"""2026-08-10 수익성 개편 — 설정과 배선을 고정한다.

근거: `docs/backtest_results/202608/20260810_f3191ec_trail_removal_long_hold.md`

세 변경은 독립이 아니다:
  - 트레일 비활성: 트레일 수익의 95.4%가 봉 집계 아티팩트였고 실제로는 엣지를 잘랐다
  - 보유 연장: 신호는 맞혔을 때 오래 가는 성질이라 조기 청산이 손해다
  - 사이징 축소: 30배는 켈리 정점을 넘어 P(DD>80%)=99.1%를 만들었다

사이징만 고치면 기댓값이 음수로 남고, 청산만 고치면 파산 위험이 남는다.
"""

from __future__ import annotations

import pytest

from quantbox.core.strategy_names import DONCHIAN, MA_CLUSTER, VR_DIRECTIONAL
from quantbox.execution.sizing import equity_fraction_per_trade, risk_based_leverage
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


def _declared(field: str):
    info = VolStraddleConfig.model_fields[field]
    return info.default_factory() if info.default_factory is not None else info.default


def test_ma_cluster_trail_is_disabled() -> None:
    """MA Cluster 트레일은 꺼져 있어야 한다."""
    assert _declared("ma_cluster_trail_enabled") is False


def test_trail_flag_only_affects_ma_cluster() -> None:
    """다른 전략의 트레일 동작은 건드리지 않았다 — 근거가 있는 전략에만 적용한다."""
    cfg = VolStraddleConfig()
    assert cfg.is_trail_enabled(MA_CLUSTER) is cfg.ma_cluster_trail_enabled
    assert cfg.is_trail_enabled(VR_DIRECTIONAL) is True
    assert cfg.is_trail_enabled(DONCHIAN) is True


def test_hold_is_long_enough_for_signal_to_pay() -> None:
    """보유가 신호가 값을 내는 구간(h>=18)에 있어야 한다.

    비용 제외 평균이 h=1 +0.067%에서 h=12 +0.601%로 커진다. 짧은 보유는
    마찰(트레이드당 약 0.20%)을 넘지 못한다.
    """
    assert _declared("ma_cluster_max_hold_4h") >= 18


def test_sizing_keeps_ruin_probability_at_zero() -> None:
    """SL 1회 손실이 자기자본의 1% 이하여야 한다.

    부트스트랩 실측: SL 1회 6%(30배)에서 P(DD>80%)=99.1%,
    0.5%(2.5배)에서 P(DD>80%)=0.0% / P(DD>50%)=0.3%.
    1%(5배)는 P(DD>50%)=19.8%로 경계다 — 그 위로 올라가면 이 가드가 깨진다.
    """
    sl = _declared("ma_cluster_sl")
    cfg = VolStraddleConfig(risk_per_trade=_declared("risk_per_trade"))
    loss_at_sl = sl * equity_fraction_per_trade(sl, cfg)
    assert loss_at_sl <= 0.01, f"SL 1회 = 자기자본 {loss_at_sl * 100:.2f}% — 파산 확률이 유의해진다"


def test_leverage_is_below_kelly_peak() -> None:
    """레버리지가 켈리 정점(관측상 15배 부근) 아래여야 한다.

    15배가 30배보다 수익이 높았다 — 정점을 넘으면 리스크를 더 져서 수익을 깎는다.
    """
    cfg = VolStraddleConfig(risk_per_trade=_declared("risk_per_trade"))
    assert risk_based_leverage(_declared("ma_cluster_sl"), cfg) < 15.0


@pytest.mark.parametrize(
    "module, marker",
    [
        ("quantbox/execution/positions.py", "is_trail_enabled"),
        ("quantbox/execution/reconciliation.py", "is_trail_enabled"),
        ("quantbox/scalper.py", "is_trail_enabled"),
    ],
)
def test_every_trail_placement_path_respects_the_flag(module: str, marker: str) -> None:
    """트레일을 거는 모든 경로가 플래그를 확인한다.

    복구·재배치 경로가 빠지면 진입에서 안 걸린 트레일이 리컨실 때 되살아난다.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / module).read_text()
    assert marker in src, f"{module}: 트레일 배치 경로가 플래그를 확인하지 않는다"
