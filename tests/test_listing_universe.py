"""대상 필터와 신규 상장 감지.

제목 파싱으로 크립토/주식을 구분하려던 초안은 조용히 틀렸다 (GIGADEVUSDT를
크립토로 분류했으나 거래소 기준은 HK_EQUITY). 거래소 자신의 분류만 믿는다.

두 번째 함정: 빈 상태로 시작하면 첫 폴링에서 기존 700여 심볼이 전부 "신규"가 된다.
그대로 두면 700개를 구독한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantbox.listing.universe import (
    KNOWN_TYPES,
    SeenStore,
    detect_new,
    is_target,
    target_symbols,
    unknown_types,
)

FIXTURE = Path(__file__).parent / "fixtures" / "exchange_info_20260813.json"


@pytest.fixture
def info() -> dict:
    return json.loads(FIXTURE.read_text())


def test_crypto_perpetual_is_target(info: dict) -> None:
    """알려진 크립토 무기한이 통과한다."""
    by_sym = {s["symbol"]: s for s in info["symbols"]}
    for sym in ("DOGEUSDT", "BTCUSDT", "DOSUSDT", "GRVTUSDT"):
        assert is_target(by_sym[sym]), f"{sym}이 대상에서 빠졌다"


def test_equity_and_commodity_are_excluded(info: dict) -> None:
    """주식·원자재 무기한은 제외된다 — 섞이면 −26.59% 측정이 오염된다.

    GIGADEVUSDT가 핵심 케이스다. 제목에는 TradFi 표시가 없어 정규식 필터를 통과했지만
    거래소 분류는 HK_EQUITY다.
    """
    by_sym = {s["symbol"]: s for s in info["symbols"]}
    for sym in ("AMDUSDT", "MSFTUSDT", "QQQUSDT", "NATGASUSDT", "XPTUSDT", "GIGADEVUSDT"):
        if sym in by_sym:
            assert not is_target(by_sym[sym]), f"{sym}이 대상에 들어왔다 — 크립토가 아니다"


def test_target_set_is_a_strict_subset(info: dict) -> None:
    """대상이 전체보다 작고 비어 있지 않다 — 필터가 통과만 시키거나 전부 막지 않는다."""
    targets = target_symbols(info)
    assert 0 < len(targets) < len(info["symbols"])
    assert all(s["underlyingType"] == "COIN" for s in targets.values())
    assert all(s["contractType"] == "PERPETUAL" for s in targets.values())
    assert all(s["quoteAsset"] == "USDT" for s in targets.values())


def test_non_usdt_quote_is_excluded(info: dict) -> None:
    """USDC·USD1 등 다른 정산 자산은 제외한다 — 측정 기준이 USDT다."""
    targets = target_symbols(info)
    assert not any(s.endswith("USDC") for s in targets)


def test_unknown_type_is_reported_not_swallowed(info: dict) -> None:
    """거래소가 새 분류를 추가하면 알아차려야 한다 — 조용히 제외하면 모르고 지나간다."""
    assert unknown_types(info) == set(), f"고정 자산에 미등록 분류가 있다: {unknown_types(info)}"

    mutated = {"symbols": [dict(info["symbols"][0], underlyingType="QUANTUM_FOAM")]}
    assert unknown_types(mutated) == {"QUANTUM_FOAM"}


def test_known_types_matches_fixture(info: dict) -> None:
    """상수가 실측과 일치한다 — 손으로 적은 목록이 낡으면 경보가 오작동한다."""
    assert {s["underlyingType"] for s in info["symbols"]} <= KNOWN_TYPES


def test_first_run_does_not_flood(info: dict, tmp_path: Path) -> None:
    """빈 상태 첫 실행에서 기존 심볼이 신규로 잡히면 안 된다.

    이 가드가 없으면 첫 폴링에서 700개 구독을 시도한다.

    쓰기 가능한 경로를 쓴다 — `bootstrap`은 등록 결과를 실제로 저장해야 하고,
    저장 실패를 삼키면 재시작마다 조용히 다시 부트스트랩하게 된다.
    """
    store = SeenStore(tmp_path / "seen.json")
    seen = store.bootstrap(info)

    assert len(seen) == len(info["symbols"]), "부트스트랩이 전 심볼을 등록하지 않았다"
    assert detect_new(target_symbols(info), seen) == {}


def test_detect_new_finds_only_the_addition(info: dict) -> None:
    """정말 새 심볼만 잡는다."""
    targets = target_symbols(info)
    victim = next(iter(targets))
    seen = {s["symbol"] for s in info["symbols"]} - {victim}

    assert set(detect_new(targets, seen)) == {victim}


def test_seen_store_roundtrip(tmp_path: Path) -> None:
    """재시작해도 본 심볼을 기억한다 — 잊으면 같은 이벤트를 다시 기록한다."""
    store = SeenStore(tmp_path / "seen.json")
    assert store.load() == set()

    store.save({"AAAUSDT", "BBBUSDT"})
    assert SeenStore(tmp_path / "seen.json").load() == {"AAAUSDT", "BBBUSDT"}


def test_saving_copies_loses_the_earlier_one(tmp_path: Path) -> None:
    """회귀 재현: 같은 `seen` 사본에서 두 번 저장하면 먼저 것이 사라진다.

    `listing_probe.py`가 감지된 심볼마다 `remember(seen | {sym})`처럼 원본을 복사해
    저장하면, 같은 폴링 주기에 두 심볼이 동시에 TRADING이 될 때 두 번째 저장이
    첫 번째 심볼을 지운다. 다음 폴링에서 그 심볼이 신규로 재감지돼 기록 스레드가
    하나 더 뜨고, 두 RotatingWriter가 같은 파일에 append하며 gzip 멤버가 엇갈려
    파일이 깨진다.
    """
    store = SeenStore(tmp_path / "seen.json")
    seen = {"EXISTINGUSDT"}
    store.save(seen)

    # 버그: 사본을 저장 — 두 번째 저장이 첫 번째 심볼(A)을 지운다.
    store.save(seen | {"AUSDT"})
    store.save(seen | {"BUSDT"})
    assert "AUSDT" not in store.load(), "사본 저장 방식은 A를 지운다 (회귀 재현)"

    # 수정: 제자리 갱신 — 둘 다 남는다.
    seen2 = {"EXISTINGUSDT"}
    store.save(seen2)
    seen2.add("AUSDT")
    store.save(seen2)
    seen2.add("BUSDT")
    store.save(seen2)
    assert store.load() == {"EXISTINGUSDT", "AUSDT", "BUSDT"}
