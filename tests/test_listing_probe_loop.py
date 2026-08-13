"""`scripts/listing_probe.py`의 폴링 루프 — 유일하게 테스트가 없던 부분.

`SeenStore` 자체는 `test_listing_universe.py`가 검증하지만, 실제로 프로덕션에 나간
치명 버그는 `SeenStore`가 아니라 이 루프에서 났다: `seen`을 매 심볼마다
`remember(seen | {sym})`로 사본 저장했더니, 같은 폴링 주기에 두 심볼이 동시에
TRADING이 되면 두 번째 저장이 첫 번째를 지웠다. 다음 폴링에서 그 심볼이 신규로
재감지돼 기록 스레드가 하나 더 뜨고, 두 `RotatingWriter`가 같은 `.jsonl.gz`에
append하며 gzip 멤버가 엇갈려 파일이 깨졌다. 이 회귀는 `SeenStore` 단위 테스트로는
잡히지 않는다 — 루프가 `seen`을 어떻게 갱신·저장하는지가 핵심이었다.

`scripts/`는 패키지가 아니라 `importlib.util.spec_from_file_location`으로 로드한다
(`tests/test_r116_prep_baseline.py`의 기존 관례를 따른다).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time as time_module
from pathlib import Path
from types import ModuleType

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "exchange_info_20260813.json"
SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "listing_probe.py"


def _load_probe_module() -> ModuleType:
    """매 테스트마다 독립된 모듈 인스턴스를 새로 로드한다.

    `STATE_DIR`/`OUT_ROOT`는 모듈 전역이라, 테스트끼리 같은 모듈 객체를 공유하면
    monkeypatch 순서에 따라 서로 오염될 수 있다.
    """
    spec = importlib.util.spec_from_file_location("listing_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _clone_symbol(info: dict, template_symbol: str, new_symbol: str, status: str = "TRADING") -> dict:
    """실제 심볼 항목을 복제해 심볼명·상태만 바꾼다 — 가짜 유니버스를 새로 만들지 않는다."""
    template = next(s for s in info["symbols"] if s["symbol"] == template_symbol)
    clone = dict(template)
    clone["symbol"] = new_symbol
    clone["status"] = status
    return clone


class _FakeThreadingNS:
    """`listing_probe` 모듈 안에서만 `threading.Event`를 가짜로 바꿔치기한다.

    전역 `threading.Event`를 직접 patch하면 `threading.Thread.__init__`이 내부적으로
    쓰는 `self._started = Event()`까지 가짜로 바뀌어, `start()`가 실제로 스레드가
    시작하기 전에 리턴하는 경쟁 상태를 만든다("cannot join thread before it is
    started" — 직접 재현해 확인했다). `Thread`는 real threading 모듈에 위임하고
    `Event`만 가로채 실제 스레드 동기화는 건드리지 않는다.
    """

    def __init__(self, real_threading, max_iters: int) -> None:
        self._real = real_threading
        self._max_iters = max_iters

    def Event(self) -> "_CountedStop":
        return _CountedStop(self._max_iters)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class _FakeTimeNS:
    """`listing_probe` 모듈 안에서만 `time.monotonic`을 가짜 시계로 바꿔치기한다.

    전역 `time.monotonic`을 직접 patch하면 `Thread.join(timeout=...)`의 내부
    타임아웃 계산까지 가짜 시계를 쓰게 된다 — `threading.Event`와 같은 이유로
    모듈 로컬 바인딩만 바꾼다.
    """

    def __init__(self, real_time, monotonic_fn) -> None:
        self._real = real_time
        self._monotonic_fn = monotonic_fn

    def monotonic(self) -> float:
        return self._monotonic_fn()

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class _CountedStop:
    """`threading.Event`를 흉내 내되, `is_set()` 호출 횟수로 루프를 끊는다.

    실제 루프는 `stop.wait(POLL_INTERVAL_S)`로 60초씩 잠드므로, 실시간으로 여러
    폴링 주기를 재현할 수 없다. `wait()`를 즉시 반환하게 하고 `is_set()`을
    `max_iters`번까지만 False로 응답해 다중 폴링을 빠르게 구동한다.
    """

    def __init__(self, max_iters: int) -> None:
        self._max_iters = max_iters
        self._checks = 0
        self._set = False

    def is_set(self) -> bool:
        self._checks += 1
        if self._checks > self._max_iters:
            self._set = True
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, timeout: float | None = None) -> bool:
        return self._set


@pytest.fixture
def info() -> dict:
    return json.loads(FIXTURE.read_text())


def _join_recorder_threads() -> None:
    """`--once` 모드는 기록 스레드를 join하지 않고 반환하므로, 검증 전에 직접 기다린다."""
    for t in threading.enumerate():
        if t.name.startswith("rec-"):
            t.join(timeout=2)


# ── 1. 같은 폴링에서 두 심볼이 동시에 감지되면 둘 다 등록된다 (치명 버그 재현) ──────


def test_same_poll_multi_symbol_detection_registers_both(info: dict, tmp_path: Path, monkeypatch) -> None:
    """`remember(seen | {sym})` 사본 저장 버그가 재발하면 실패한다.

    같은 폴링 주기에 AAAUSDT·BBBUSDT가 동시에 TRADING이 되면, 사본 저장 방식에서는
    두 번째 저장이 첫 번째 심볼을 지운다. 그러면 다음 폴링에서 AAAUSDT가 다시
    "신규"로 잡혀 두 번째 기록 스레드가 뜨고 파일이 깨진다.
    """
    mod = _load_probe_module()
    new_a = _clone_symbol(info, "DOGEUSDT", "AAAUSDT")
    new_b = _clone_symbol(info, "DOGEUSDT", "BBBUSDT")
    info["symbols"].append(new_a)
    info["symbols"].append(new_b)

    seen_before = {s["symbol"] for s in info["symbols"]} - {"AAAUSDT", "BBBUSDT"}
    seen_path = tmp_path / "seen_symbols.json"

    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path / "socket")
    mod.SeenStore(seen_path).save(seen_before)

    recorded: list[str] = []
    monkeypatch.setattr(mod, "record", lambda sym, out_root, duration_s, max_gb, stop: recorded.append(sym))
    monkeypatch.setattr(mod, "fetch_exchange_info", lambda timeout=30.0: info)
    monkeypatch.setattr(sys, "argv", ["listing_probe.py", "--once"])

    rc = mod.main()
    _join_recorder_threads()

    assert rc == 0
    assert set(recorded) == {"AAAUSDT", "BBBUSDT"}, f"둘 다 기록을 시작해야 한다: {recorded}"

    persisted = mod.SeenStore(seen_path).load()
    assert {"AAAUSDT", "BBBUSDT"} <= persisted, (
        f"두 심볼 모두 seen에 남아야 한다 — 사본 저장 버그면 하나가 사라진다: {persisted}"
    )


# ── 2. `recording` 가드 — 이미 기록 중인 심볼에는 두 번째 스레드를 띄우지 않는다 ─────


def test_recording_guard_prevents_duplicate_thread(info: dict, tmp_path: Path, monkeypatch) -> None:
    """같은 심볼이 폴링 사이에 다시 "신규"로 보여도 `record()`는 한 번만 불려야 한다.

    두 번째 폴링 직전에 seen 파일에서 CCCUSDT를 제거해 재감지 상황을 만든다(원인은
    무관 — 목적은 "recording 세트가 없다면 두 번째 스레드가 뜨는 상태"를 재현하는
    것이다). `recording` 가드가 없으면 두 RotatingWriter가 같은 파일에 동시에
    append해 gzip이 깨진다.
    """
    mod = _load_probe_module()
    new_sym = _clone_symbol(info, "DOGEUSDT", "CCCUSDT")
    info["symbols"].append(new_sym)

    seen_path = tmp_path / "seen_symbols.json"
    seen_before = {s["symbol"] for s in info["symbols"]} - {"CCCUSDT"}
    mod.SeenStore(seen_path).save(seen_before)

    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path / "socket")

    recorded: list[str] = []
    monkeypatch.setattr(mod, "record", lambda sym, out_root, duration_s, max_gb, stop: recorded.append(sym))

    call_count = {"n": 0}

    def _fake_fetch(timeout: float = 30.0) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 2:
            # 두 번째 폴링 직전 — CCCUSDT가 seen에서 빠진 상태를 강제로 재현한다.
            mod.SeenStore(seen_path).save(seen_before)
        return info

    monkeypatch.setattr(mod, "fetch_exchange_info", _fake_fetch)
    monkeypatch.setattr(mod, "threading", _FakeThreadingNS(threading, 2))
    monkeypatch.setattr(sys, "argv", ["listing_probe.py"])

    rc = mod.main()

    assert rc == 0
    assert call_count["n"] >= 2, "두 번째 폴링까지 구동되지 않았다 — 테스트 자체가 시나리오를 재현 못했다"
    assert recorded.count("CCCUSDT") == 1, f"CCCUSDT가 두 번 이상 기록됐다 — recording 가드가 뚫렸다: {recorded}"


# ── 3. PENDING_TRADING이 48시간을 넘기면 포기하고 재감지되지 않는다 ─────────────────


def test_pending_timeout_gives_up_after_48h(info: dict, tmp_path: Path, monkeypatch) -> None:
    """48시간 넘게 TRADING이 안 되면 seen에 등록하고 기록을 시작하지 않아야 한다.

    타임아웃 처리가 없으면 영원히 PENDING으로 남아 매 폴링마다 재평가되고,
    seen에도 등록되지 않아 프로세스를 재시작할 때마다 다시 "신규"로 잡힌다.
    """
    mod = _load_probe_module()
    new_sym = _clone_symbol(info, "DOGEUSDT", "DDDUSDT", status="PENDING_TRADING")
    info["symbols"].append(new_sym)

    seen_path = tmp_path / "seen_symbols.json"
    seen_before = {s["symbol"] for s in info["symbols"]} - {"DDDUSDT"}
    mod.SeenStore(seen_path).save(seen_before)

    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path / "socket")

    recorded: list[str] = []
    monkeypatch.setattr(mod, "record", lambda sym, out_root, duration_s, max_gb, stop: recorded.append(sym))
    monkeypatch.setattr(mod, "fetch_exchange_info", lambda timeout=30.0: info)

    # 호출 순서: ①1차 폴링 setdefault ②1차 폴링 타임아웃 판정(아직) ③2차 폴링
    # setdefault(무시됨) ④2차 폴링 타임아웃 판정(48시간 초과).
    clock = iter([0.0, 10.0, 20.0, 200_000.0])
    monkeypatch.setattr(mod, "time", _FakeTimeNS(time_module, lambda: next(clock)))

    monkeypatch.setattr(mod, "threading", _FakeThreadingNS(threading, 2))
    monkeypatch.setattr(sys, "argv", ["listing_probe.py"])

    rc = mod.main()

    assert rc == 0
    assert recorded == [], f"48시간 타임아웃 대상인데 기록이 시작됐다: {recorded}"

    persisted = mod.SeenStore(seen_path).load()
    assert "DDDUSDT" in persisted, "타임아웃된 심볼이 seen에 등록되지 않았다 — 다음 폴링에 다시 감지된다"


# ── 4. 빈 상태 첫 실행은 loop 레벨에서도 기존 심볼을 신규로 오인하지 않는다 ────────


def test_first_run_through_loop_does_not_start_any_recorder(info: dict, tmp_path: Path, monkeypatch) -> None:
    """`SeenStore.bootstrap()` 단위 테스트는 이미 있지만(`test_listing_universe.py`),
    `main()`이 `bootstrap()`보다 먼저 `detect_new()`를 부르는 순서 실수를 하면
    첫 폴링에서 기존 심볼 수백 개가 전부 기록 스레드를 띄운다. 실제 루프를 통과시켜
    확인한다.
    """
    mod = _load_probe_module()

    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path / "socket")

    recorded: list[str] = []
    monkeypatch.setattr(mod, "record", lambda sym, out_root, duration_s, max_gb, stop: recorded.append(sym))
    monkeypatch.setattr(mod, "fetch_exchange_info", lambda timeout=30.0: info)
    monkeypatch.setattr(sys, "argv", ["listing_probe.py", "--once"])

    rc = mod.main()
    _join_recorder_threads()

    assert rc == 0
    assert recorded == [], f"첫 실행인데 기록이 시작됐다: {recorded[:5]}"

    persisted = mod.SeenStore(tmp_path / "seen_symbols.json").load()
    assert persisted == {s["symbol"] for s in info["symbols"]}
