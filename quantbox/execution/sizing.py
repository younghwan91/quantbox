"""포지션 사이징 산수 — 라이브와 백테스트가 공유하는 단일 소스.

라이브는 SL 폭에서 레버리지를 역산한다: **SL 1회 = 잔고의 `risk_per_trade`**.
백테스트는 이 축이 없어서 트레이드 손익을 `pnl / max_concurrent`로 환산해 왔다.
그 모델에서는 SL 1회가 자기자본의 0.2%지만 라이브에서는 6%다 — 30배 차이다.

여기 함수들은 그 환산을 한 곳에 모은다. 사이징을 쓰는 쪽은 전부 이걸 호출한다.
"""

from __future__ import annotations

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

__all__ = ["equity_fraction_per_trade", "risk_based_leverage"]


def risk_based_leverage(sl_pct: float, cfg: VolStraddleConfig | None = None) -> float:
    """SL 폭에서 역산한 레버리지.

    `per_slot_capital × leverage × SL = balance × risk_per_trade` 을 풀면
    `leverage = risk_per_trade × max_concurrent / SL` 이다.

    Args:
        sl_pct: 해당 전략의 손절 폭 (비율, 예: 0.02 = 2%).
        cfg: 설정. 생략 시 `VolStraddleConfig()` (Rule #6 — 하드코딩 금지).

    Returns:
        `[1.0, max_leverage]` 로 클램프된 레버리지 배수.

    Raises:
        ValueError: `sl_pct <= 0` — 레버리지가 발산한다.
    """
    cfg = cfg if cfg is not None else VolStraddleConfig()
    if sl_pct <= 0:
        raise ValueError(f"sl_pct must be positive, got {sl_pct}")
    leverage = cfg.risk_per_trade * cfg.max_concurrent / sl_pct
    return max(1.0, min(cfg.max_leverage, leverage))


def equity_fraction_per_trade(sl_pct: float, cfg: VolStraddleConfig | None = None) -> float:
    """가격 변동 1단위가 자기자본에 미치는 비율.

    한 슬롯의 명목가는 `balance / max_concurrent × leverage` 이므로, 가격이 `r`만큼
    움직였을 때 자기자본 변화는 `r × leverage / max_concurrent` 다.

    레버리지가 상한에 걸리지 않으면 이 값은 `risk_per_trade / sl_pct` 와 같다 —
    즉 SL 1회가 정확히 `risk_per_trade` 손실이 된다는 설계와 일치한다.

    Args:
        sl_pct: 해당 전략의 손절 폭 (비율).
        cfg: 설정. 생략 시 `VolStraddleConfig()`.

    Returns:
        가격 변동률 → 자기자본 변동률 환산 계수.
    """
    cfg = cfg if cfg is not None else VolStraddleConfig()
    return risk_based_leverage(sl_pct, cfg) / cfg.max_concurrent
