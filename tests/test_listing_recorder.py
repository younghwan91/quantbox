"""한 심볼을 정해진 기간 소켓 기록한다.

이 환경은 aggTrade·kline_1m·forceOrder가 네트워크 레벨에서 차단돼 0건이 온다
(`scripts/collect_book_data.py` 독스트링). depth 계열도 막혔을 가능성이 있었으나
2026-08-13 실측에서 전부 정상임을 확인했다. 회귀하면 여기서 잡힌다.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from quantbox.listing.book import effective_half_spread_bp, parse_levels
from quantbox.listing.recorder import STREAM_SUFFIXES, record, stream_kind


def test_stream_kind_strips_symbol_and_speed() -> None:
    """저장 디렉터리 이름은 갱신 주기를 포함하지 않는다 — 주기를 바꿔도 경로가 갈리면 안 된다."""
    assert stream_kind("btcusdt@bookTicker") == "bookTicker"
    assert stream_kind("btcusdt@trade") == "trade"
    assert stream_kind("btcusdt@depth20@500ms") == "depth20"
    assert stream_kind("btcusdt@depth20") == "depth20"


def test_three_streams_are_configured() -> None:
    """세 스트림을 다 구독한다 — depth20이 빠지면 실효 스프레드를 못 잰다."""
    assert STREAM_SUFFIXES == ("bookTicker", "trade", "depth20@500ms")


@pytest.mark.slow
def test_records_all_three_streams_live(tmp_path: Path) -> None:
    """실구독 30초. 세 스트림이 모두 파일로 떨어지는가.

    실데이터만 쓴다 (프로젝트 규칙: No Mock Data). 이 환경에서 여러 스트림이 조용히
    0건을 반환하므로, 목으로는 이 검증이 성립하지 않는다.
    """
    counts = record("BTCUSDT", tmp_path, duration_s=30.0)

    for kind in ("bookTicker", "trade", "depth20"):
        assert counts.get(kind, 0) > 0, f"{kind} 스트림이 0건이다 — 차단됐거나 배선이 틀렸다"
        files = list((tmp_path / kind / "BTCUSDT").glob("*.jsonl.gz"))
        assert files, f"{kind} 파일이 없다"


@pytest.mark.slow
def test_recorded_depth_feeds_the_spread_calculation(tmp_path: Path) -> None:
    """기록한 depth20이 실효 스프레드 계산에 그대로 들어가는가.

    기록 형식과 계산 입력이 어긋나면 3주를 모으고 나서야 알게 된다. 지금 이어본다.
    """
    record("BTCUSDT", tmp_path, duration_s=30.0)
    f = next(iter((tmp_path / "depth20" / "BTCUSDT").glob("*.jsonl.gz")))

    with gzip.open(f, "rt") as fh:
        d = json.loads(fh.readline())["d"]

    bids, asks = parse_levels(d["b"]), parse_levels(d["a"])
    assert len(bids) == 20 and len(asks) == 20

    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)
    assert filled and 0.0 < eff < 20.0, f"BTCUSDT 편도 스프레드가 비현실적이다: {eff}bp"
