"""G11: 부분체결/더스트가 현재 사이징에서 구속력이 있는지 검사.

라이브는 명목가가 거래소 `min_notional`(≈$5)보다 작으면 진입이 거부되고, 부분체결
잔여가 그 아래로 남으면 더스트가 되어 슬롯을 막는다(`positions.py:252-255`).

백테스트는 이걸 모델링하지 않는다. 그래도 되는지는 **사이징 산수로 결정된다**:
현재 파라미터에서 최소 명목가는 $300 수준이라 $5 임계와 두 자릿수 배율로 떨어져 있다.

이 테스트는 그 여유가 사라지는 순간을 잡는다. 잔고가 급감하거나 `risk_per_trade`,
`max_concurrent`, CW 배수 하한이 바뀌면 더스트가 실제 변수가 되고, 그때는 백테스트에도
반영해야 한다.
"""

from __future__ import annotations

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

#: Binance USDT-M 최소 명목가 (심볼별로 다르나 $5가 일반적 하한).
MIN_NOTIONAL = 5.0

#: 실계좌 잔고 (2026-08). 이 잔고에서는 더스트가 문제되지 않아야 한다.
LIVE_BALANCE = 1000.0

#: 여유 배수 하한. 이 아래로 내려가면 백테스트가 무시하는 더스트/부분체결이
#: 실제 성과 변수가 되기 시작한다.
MARGIN = 10.0


def _declared(field: str):
    """선언된 프로덕션 기본값.

    `conftest.py:68`이 테스트 환경에 `VS_RISK_PER_TRADE=0.01`을 주입하므로
    `VolStraddleConfig()` 인스턴스는 라이브 사이징이 아니다. 이 가드가 지켜야 할
    대상은 프로덕션 값이므로 선언 기본값을 직접 읽는다.
    """
    info = VolStraddleConfig.model_fields[field]
    if info.default_factory is not None:
        return info.default_factory()
    return info.default


def _min_notional_at(balance: float, sl: float) -> float:
    """가장 작은 진입 명목가 — 최저 CW 배수까지 적용한 값."""
    max_concurrent = _declared("max_concurrent")
    per_slot = balance / max_concurrent
    leverage = max(1.0, min(_declared("max_leverage"), _declared("risk_per_trade") * max_concurrent / sl))
    cw_floor = min(_declared("ma_cluster_cw_multipliers"))
    return per_slot * leverage * cw_floor


def _min_operating_balance(sl: float) -> float:
    """더스트 여유 10배를 확보하는 최소 잔고.

    레버리지를 낮추면 명목가가 줄어 이 값이 **올라간다** — 사이징 축소의 숨은 대가다.
    """
    return (
        MIN_NOTIONAL
        * MARGIN
        * _declared("max_concurrent")
        / (
            max(1.0, min(_declared("max_leverage"), _declared("risk_per_trade") * _declared("max_concurrent") / sl))
            * min(_declared("ma_cluster_cw_multipliers"))
        )
    )


@pytest.mark.parametrize("strategy_sl_attr", ["ma_cluster_sl", "d_sl", "donchian_sl"])
def test_dust_not_binding_at_live_balance(strategy_sl_attr: str) -> None:
    """실계좌 잔고에서 최소 명목가가 거래소 하한의 10배를 넘어야 한다.

    2026-08-10 사이징 축소(30배 → 2.5배)로 명목가가 12분의 1이 됐다. 파산 확률을
    없애는 대가로 **최소 운용 잔고가 올라간다** — 그 경계를 여기서 고정한다.
    """
    sl = _declared(strategy_sl_attr)
    notional = _min_notional_at(LIVE_BALANCE, sl)

    assert notional > MIN_NOTIONAL * MARGIN, (
        f"{strategy_sl_attr}: 잔고 ${LIVE_BALANCE:.0f}에서 최소 명목가 ${notional:.2f} — "
        f"거래소 하한 ${MIN_NOTIONAL}에 근접했다. 더스트/부분체결을 백테스트에 반영해야 한다."
    )


@pytest.mark.parametrize("strategy_sl_attr", ["ma_cluster_sl", "d_sl", "donchian_sl"])
def test_minimum_operating_balance_is_below_live_balance(strategy_sl_attr: str) -> None:
    """운용 가능 하한 잔고가 실계좌보다 낮아야 한다.

    현재 값(2026-08-10): MA Cluster $400 / VR $200 / Donchian $300.
    잔고가 이 아래로 내려가면 진입이 거부되거나 더스트가 슬롯을 막기 시작한다.
    """
    need = _min_operating_balance(_declared(strategy_sl_attr))
    assert need < LIVE_BALANCE, (
        f"{strategy_sl_attr}: 최소 운용 잔고 ${need:,.0f} > 실계좌 ${LIVE_BALANCE:,.0f} — "
        f"이 사이징으로는 운용할 수 없다."
    )


def test_dust_becomes_binding_at_tiny_balance() -> None:
    """가드가 공허하지 않음을 보인다 — 잔고가 충분히 작으면 실제로 걸린다."""
    tiny = _min_notional_at(1.0, _declared("ma_cluster_sl"))
    assert tiny < MIN_NOTIONAL * MARGIN, "어떤 잔고에서도 안 걸린다면 이 가드는 아무것도 지키지 않는다"
