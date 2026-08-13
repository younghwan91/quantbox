"""시간 단위 gzip JSONL writer.

`scripts/collect_book_data.py`에서 추출했다. 그 수집기는 2026-08-10부터 무중단
가동 중이고 바이낸스가 2024-03-30자로 bookTicker 아카이브를 중단했으므로 여기
쌓이는 데이터는 복구 불가능하다. 추출이 동작을 바꾸면 안 된다.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from quantbox.collect.rotating_writer import MAX_QUEUE, RotatingWriter, disk_used_gb


def _drain(w: RotatingWriter, timeout: float = 5.0) -> None:
    """쓰기 스레드가 큐를 비울 때까지 기다린다."""
    end = time.monotonic() + timeout
    while w.qsize and time.monotonic() < end:
        time.sleep(0.02)
    w.flush()


def test_writes_expected_path_and_format(tmp_path: Path) -> None:
    """`<root>/<stream>/<symbol>/<YYYY-MM-DD-HH>.jsonl.gz`, 각 줄 `{recv_ns, d}`.

    기존 분석 코드(`autoresearch/spread_est.py`)가 이 레이아웃을 전제한다.
    """
    w = RotatingWriter(tmp_path)
    w.submit("bookTicker", "TESTUSDT", 123456789, {"b": "1.0", "a": "1.1"})
    _drain(w)
    w.close()

    hour = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    path = tmp_path / "bookTicker" / "TESTUSDT" / f"{hour}.jsonl.gz"
    assert path.exists(), f"기대 경로에 파일이 없다: {path}"

    with gzip.open(path, "rt") as fh:
        rec = json.loads(fh.readline())
    assert rec == {"recv_ns": 123456789, "d": {"b": "1.0", "a": "1.1"}}


def test_streams_and_symbols_are_separated(tmp_path: Path) -> None:
    """스트림·심볼별로 파일이 갈린다 — 섞이면 분석에서 되돌릴 수 없다."""
    w = RotatingWriter(tmp_path)
    w.submit("bookTicker", "AAAUSDT", 1, {"x": 1})
    w.submit("depth20", "AAAUSDT", 2, {"x": 2})
    w.submit("bookTicker", "BBBUSDT", 3, {"x": 3})
    _drain(w)
    w.close()

    files = {p.relative_to(tmp_path).parts[:2] for p in tmp_path.rglob("*.jsonl.gz")}
    assert files == {("bookTicker", "AAAUSDT"), ("depth20", "AAAUSDT"), ("bookTicker", "BBBUSDT")}
    assert w.counts["bookTicker"] == 2
    assert w.counts["depth20"] == 1


def test_overflow_is_counted_not_silent(tmp_path: Path) -> None:
    """큐가 넘치면 드랍을 센다 — 조용한 유실은 데이터에 구멍을 남기고 알려주지 않는다."""
    w = RotatingWriter(tmp_path)
    w._stop.set()  # 배수 스레드를 멈춰 큐를 강제로 채운다
    w._thread.join(timeout=5)
    for i in range(MAX_QUEUE + 10):
        w.submit("bookTicker", "TESTUSDT", i, {"i": i})

    assert w.dropped > 0, "큐 상한을 넘겼는데 드랍이 0이다"


def test_disk_used_counts_only_gz(tmp_path: Path) -> None:
    """`.gz`만 센다 — 상한 판정이 다른 파일에 오염되면 안 된다."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.jsonl.gz").write_bytes(b"0" * 1000)
    (tmp_path / "a" / "notes.txt").write_bytes(b"0" * 10_000)

    assert disk_used_gb(tmp_path) == 1000 / 1e9
    assert disk_used_gb(tmp_path / "missing") == 0.0
