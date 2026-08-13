"""호가창을 실제로 걸어 얻는 편도 실효 스프레드.

최우선 호가 스프레드만 보면 "1bp인데 그 가격에 100달러어치뿐"인 상황을 못 거른다.
신규 상장 초기는 호가가 얇아 이 구분이 판정을 뒤집는다. 그래서 판정값은
목표 금액을 실제로 체결시킨 체결가중평균과 중간가의 괴리다.
"""

from __future__ import annotations

import math

import pytest

from quantbox.listing.book import (
    effective_half_spread_bp,
    fill_vwap,
    parse_levels,
    top_half_spread_bp,
)


def _book(spread_bp: float, qty_usd_per_level: float, n: int = 20, mid: float = 100.0):
    """중간가 ``mid``, 지정 스프레드, 각 호가에 균일 금액이 쌓인 호가창."""
    half = mid * spread_bp / 1e4
    tick = mid * 1e-4
    bids = [(mid - half - i * tick, qty_usd_per_level / (mid - half - i * tick)) for i in range(n)]
    asks = [(mid + half + i * tick, qty_usd_per_level / (mid + half + i * tick)) for i in range(n)]
    return bids, asks


def test_parse_levels_converts_binance_strings() -> None:
    """바이낸스는 호가를 문자열로 준다 — float 변환이 없으면 산술이 조용히 문자열 연결이 된다."""
    assert parse_levels([["0.070710", "1205399"], ["0.070700", "1231221"]]) == [
        (0.070710, 1205399.0),
        (0.070700, 1231221.0),
    ]


def test_thick_top_level_equals_top_of_book_spread() -> None:
    """최우선 호가에서 다 채워지면 실효 스프레드 == 최우선 호가 스프레드.

    이게 성립하지 않으면 계산이 틀린 것이다 — 깊이가 충분할 때는 두 값이 같아야 한다.
    """
    bids, asks = _book(spread_bp=10.0, qty_usd_per_level=10_000.0)
    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)

    assert filled
    assert eff == pytest.approx(top_half_spread_bp(bids, asks), rel=1e-9)
    assert eff == pytest.approx(10.0, rel=1e-6)  # 편도 = 스프레드의 절반 → 여기선 half=10bp


def test_thin_book_costs_more_than_top_of_book() -> None:
    """호가가 얇으면 실효 스프레드가 최우선 호가 스프레드보다 크다 — 깊이 부족이 자동 반영된다."""
    bids, asks = _book(spread_bp=10.0, qty_usd_per_level=2.0)  # 레벨당 2달러뿐
    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)

    assert filled, "20호가 × 2달러 = 40달러이므로 24달러는 채워져야 한다"
    assert eff > top_half_spread_bp(bids, asks), "얇은데 최우선 호가와 같은 값이 나왔다"


def test_unfillable_book_reports_not_filled() -> None:
    """20호가를 다 먹어도 목표 금액이 안 되면 체결 불가로 보고한다."""
    bids, asks = _book(spread_bp=10.0, qty_usd_per_level=0.5)  # 20호가 합쳐도 10달러
    _, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)

    assert not filled


def test_fill_vwap_stops_at_target() -> None:
    """목표 금액에 도달하면 남은 호가를 먹지 않는다 — 다 먹으면 비용이 과대평가된다."""
    levels = [(100.0, 1.0), (200.0, 1.0)]  # 100달러 + 200달러
    vwap, filled = fill_vwap(levels, notional_usd=100.0)

    assert filled
    assert vwap == pytest.approx(100.0), "첫 호가에서 끝나야 하는데 둘째까지 먹었다"


def test_empty_book_is_nan_not_crash() -> None:
    """빈 호가창은 예외가 아니라 nan이다 — 상장 직후 첫 스냅샷이 비어 있을 수 있다."""
    eff, filled = effective_half_spread_bp([], [], notional_usd=24.0)

    assert math.isnan(eff)
    assert not filled


def test_crossed_book_is_rejected() -> None:
    """교차 호가(bid >= ask)는 데이터 오류다 — 음수 스프레드를 조용히 내보내지 않는다."""
    eff, filled = effective_half_spread_bp([(101.0, 100.0)], [(100.0, 100.0)], notional_usd=24.0)

    assert math.isnan(eff)
    assert not filled


@pytest.mark.slow
def test_live_book_gives_sane_spread() -> None:
    """실제 depth20 스냅샷에서 값이 합리적인가.

    합성 호가창은 내가 만든 가정을 검증할 뿐이다. 실호가 구조(가격·수량 문자열,
    레벨 수, 틱 크기)에서도 동작하는지는 실구독으로만 확인된다.
    """
    import asyncio
    import json

    import websockets

    async def snap() -> dict:
        url = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@500ms"
        async with websockets.connect(url, ping_interval=20) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            return json.loads(raw)["data"]

    d = asyncio.run(snap())
    bids, asks = parse_levels(d["b"]), parse_levels(d["a"])
    assert len(bids) == 20 and len(asks) == 20, "depth20인데 20레벨이 아니다"

    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)
    assert filled, "BTCUSDT 최우선 호가로 24달러가 안 채워질 리 없다"
    assert 0.0 < eff < 20.0, f"BTCUSDT 편도 스프레드가 비현실적이다: {eff}bp"
    # 24달러는 BTC 최우선 호가에 묻히므로 최우선 호가 값과 일치해야 한다.
    assert eff == pytest.approx(top_half_spread_bp(bids, asks), rel=1e-6)
