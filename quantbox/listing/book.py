"""호가창 → 편도 실효 스프레드.

최우선 호가 스프레드는 "그 가격에 얼마나 있는가"를 말해주지 않는다. 신규 상장
초기처럼 얇은 호가에서는 그 차이가 판정을 뒤집으므로, 목표 금액을 호가창에 실제로
걸어 체결시킨 값을 쓴다. 깊이가 충분하면 두 값은 같아지고, 얇으면 실효 쪽이 커진다.
따라서 **깊이 부족을 별도 조건으로 둘 필요가 없다.**

순수 함수만 둔다 — 네트워크도 파일도 만지지 않으므로 테스트가 빠르고 결정적이다.
"""

from __future__ import annotations

import math

PriceLevel = tuple[float, float]
"""``(price, qty)``. qty는 코인 수량이지 금액이 아니다."""

_EPS = 1e-12


def parse_levels(raw: list[list[str]]) -> list[PriceLevel]:
    """바이낸스 호가(문자열 쌍)를 float 쌍으로.

    바이낸스는 ``[["0.070710", "1205399"], ...]`` 형태로 준다. 문자열 그대로 두면
    비교·산술이 사전순으로 동작해 조용히 틀린다.
    """
    return [(float(p), float(q)) for p, q in raw]


def fill_vwap(levels: list[PriceLevel], notional_usd: float) -> tuple[float, bool]:
    """``levels``를 순서대로 먹어 ``notional_usd``를 채운 체결가중평균가.

    Args:
        levels: 매수측은 가격 내림차순, 매도측은 오름차순.
        notional_usd: 채울 명목 금액(USD).

    Returns:
        ``(vwap, filled)``. 호가를 다 먹어도 목표에 못 미치면 ``filled=False``이고
        vwap은 먹은 만큼의 평균이다(호가창이 비었으면 nan).
    """
    if notional_usd <= 0:
        return float("nan"), False

    remaining = notional_usd
    spent = 0.0
    qty = 0.0
    for price, size in levels:
        if price <= 0 or size <= 0:
            continue
        take = min(remaining, price * size)
        spent += take
        qty += take / price
        remaining -= take
        if remaining <= _EPS:
            return spent / qty, True

    return (spent / qty if qty > 0 else float("nan")), False


def top_half_spread_bp(bids: list[PriceLevel], asks: list[PriceLevel]) -> float:
    """최우선 호가 기준 편도 스프레드(bp). 대조용."""
    if not bids or not asks:
        return float("nan")
    bid, ask = bids[0][0], asks[0][0]
    if bid <= 0 or ask <= bid:
        return float("nan")
    mid = (bid + ask) / 2.0
    return (ask - bid) / 2.0 / mid * 1e4


def effective_half_spread_bp(bids: list[PriceLevel], asks: list[PriceLevel], notional_usd: float) -> tuple[float, bool]:
    """``notional_usd``를 양방향으로 체결시킨 **편도** 실효 스프레드(bp).

    매수는 매도호가를, 매도는 매수호가를 걸어 각각 중간가와의 괴리를 구하고 평균낸다.
    왕복 비용은 이 값의 2배에 수수료를 더한 것이다.

    Returns:
        ``(bp, filled)``. 어느 한쪽이라도 채워지지 않으면 ``filled=False``.
        호가창이 비었거나 교차했으면 ``(nan, False)``.
    """
    if not bids or not asks:
        return float("nan"), False

    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0 or best_ask <= best_bid:
        return float("nan"), False  # 교차·비정상 호가는 음수 스프레드를 만든다

    mid = (best_bid + best_ask) / 2.0
    buy_vwap, buy_ok = fill_vwap(asks, notional_usd)
    sell_vwap, sell_ok = fill_vwap(bids, notional_usd)
    if math.isnan(buy_vwap) or math.isnan(sell_vwap):
        return float("nan"), False

    half = ((buy_vwap - mid) + (mid - sell_vwap)) / 2.0
    return half / mid * 1e4, (buy_ok and sell_ok)
