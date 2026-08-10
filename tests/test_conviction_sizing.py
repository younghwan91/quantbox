"""G8 파리티: MA cluster conviction-weighted sizing — 라이브와 **같은 함수**를 쓴다.

이전에는 라이브(`strategy.py`)가 인라인으로 CW를 판정하고 백테스트가 그걸 벡터화해
다시 짰다. 값은 맞춰 놨지만 사본이라 라이브 조건이 바뀌면 조용히 갈라졌다 — 실제로
`close <= 0` 경계에서 이미 갈라져 있었다.

지금은 `strategy.vol_straddle.conviction`이 단일 소스이고 양쪽이 이를 호출한다.
여기 테스트는 (a) 판정 자체의 경계와 (b) **사본이 다시 생기지 않았는지**를 고정한다.
"""

from pathlib import Path

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.conviction import (
    CW_VOL_WINDOW,
    ma_cluster_cw_multiplier,
    ma_cluster_cw_score,
    ma_cluster_spread_pct,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"


def _flat_volume(n: int = CW_VOL_WINDOW, value: float = 100.0) -> np.ndarray:
    """스파이크 없는 거래량 — 조건2가 거짓인 기본값."""
    return np.full(n, value, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────
#  판정 경계
# ──────────────────────────────────────────────────────────────────────────


def test_all_three_conditions_give_max_score() -> None:
    """좁은 스프레드 + 거래량 스파이크 + 강한 모멘텀 → 점수 3 → 최대 배수."""
    cfg = VolStraddleConfig()
    vol = _flat_volume()
    vol[-1] = 1000.0  # mean = (19*100+1000)/20 = 145, 1000 > 1.5*145

    score = ma_cluster_cw_score(
        spread_pct=0.5,  # < 1.5%
        mom_val=cfg.ma_cluster_cw_mom_mult * cfg.ma_cluster_min_momentum * 10.0,
        volume=vol,
        cfg=cfg,
    )
    assert score == 3
    assert ma_cluster_cw_multiplier(score, cfg) == cfg.ma_cluster_cw_multipliers[3]


def test_no_conditions_give_min_score() -> None:
    """세 조건 모두 거짓 → 점수 0 → 라이브가 명목가를 절반으로 줄이는 구간."""
    cfg = VolStraddleConfig()
    score = ma_cluster_cw_score(spread_pct=3.0, mom_val=0.0, volume=_flat_volume(), cfg=cfg)
    assert score == 0
    assert ma_cluster_cw_multiplier(score, cfg) == cfg.ma_cluster_cw_multipliers[0]


def test_volume_condition_needs_20_bars() -> None:
    """20봉 미만이면 거래량 조건은 아예 평가되지 않는다 (라이브 `len >= 20` 가드)."""
    cfg = VolStraddleConfig()
    short = np.full(CW_VOL_WINDOW - 1, 100.0)
    short[-1] = 1e6  # 창이 모자라면 이 스파이크는 무시돼야 한다

    score = ma_cluster_cw_score(spread_pct=0.5, mom_val=0.0, volume=short, cfg=cfg)
    assert score == 1, "20봉 미만인데 거래량 조건이 가산됐다"


def test_spread_threshold_is_percent_units() -> None:
    """임계 비교는 퍼센트 단위 — 라이브가 `spread < threshold * 100`으로 쓴다.

    비율 단위(0.015)로 잘못 비교하면 거의 모든 봉이 조건1을 통과해 점수가 부풀려진다.
    """
    cfg = VolStraddleConfig()
    # 2.0% — 임계 1.5%보다 넓으므로 조건1은 거짓
    assert ma_cluster_cw_score(spread_pct=2.0, mom_val=0.0, volume=_flat_volume(), cfg=cfg) == 0
    # 1.0% — 통과
    assert ma_cluster_cw_score(spread_pct=1.0, mom_val=0.0, volume=_flat_volume(), cfg=cfg) == 1


def test_disabled_config_scores_zero() -> None:
    """`ma_cluster_cw_enabled=False`면 라이브는 점수를 세지 않는다."""
    cfg = VolStraddleConfig(ma_cluster_cw_enabled=False)
    vol = _flat_volume()
    vol[-1] = 1e6
    assert ma_cluster_cw_score(spread_pct=0.1, mom_val=99.0, volume=vol, cfg=cfg) == 0


def test_spread_falls_back_to_zero_on_nonpositive_close() -> None:
    """`close <= 0`이면 스프레드는 0.0 — 라이브의 폴백 동작을 그대로 따른다.

    사본 시절 백테스트는 이걸 NaN으로 두어 조건1을 탈락시켰다. 실데이터엔 없는
    경계지만, 사본이 만드는 드리프트가 실재했음을 남겨 둔다.
    """
    assert ma_cluster_spread_pct(0.0, 100.0, 101.0, 102.0, 103.0) == 0.0
    assert ma_cluster_spread_pct(100.0, 100.0, 100.0, 100.0, 100.5) == pytest.approx(0.5)


def test_multiplier_clamps_beyond_table() -> None:
    """표 길이를 넘는 점수는 클램프된다 (라이브 `min(score, len-1)`)."""
    cfg = VolStraddleConfig()
    assert ma_cluster_cw_multiplier(99, cfg) == cfg.ma_cluster_cw_multipliers[-1]


# ──────────────────────────────────────────────────────────────────────────
#  구조적 결합 — 사본이 다시 생기지 않았는가
# ──────────────────────────────────────────────────────────────────────────


def test_live_has_no_inline_cw_arithmetic() -> None:
    """라이브가 CW 조건을 다시 인라인으로 계산하면 실패한다.

    `strategy.py`는 임계값을 직접 비교하지 않고 공유 함수를 호출해야 한다.
    """
    src = (ROOT / "quantbox" / "strategy" / "vol_straddle" / "strategy.py").read_text()
    for token in ("ma_cluster_cw_spread_threshold", "ma_cluster_cw_volume_mult", "ma_cluster_cw_mom_mult"):
        assert token not in src, f"strategy.py가 CW 조건을 다시 인라인 계산한다: {token}"
    assert "ma_cluster_cw_score(" in src, "라이브가 공유 CW 판정을 호출하지 않는다"


def test_backtest_has_no_cw_copy() -> None:
    """백테스트에 CW 판정 사본이 부활하면 실패한다."""
    assert not (ROOT / "quantbox" / "backtest" / "conviction.py").exists(), (
        "backtest/conviction.py가 되살아났다 — CW 판정은 strategy 쪽 단일 소스만 둔다"
    )
    src = (ROOT / "quantbox" / "backtest" / "sim_core.py").read_text()
    for token in ("ma_cluster_cw_spread_threshold", "ma_cluster_cw_volume_mult", "ma_cluster_cw_mom_mult"):
        assert token not in src, f"sim_core.py가 CW 조건을 다시 계산한다: {token}"


# ──────────────────────────────────────────────────────────────────────────
#  sim 배선 — 실데이터
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_live_path_applies_cw_sizing_by_default() -> None:
    """`run_ma_cluster_live`는 라이브 재현이 목적이므로 CW를 기본 적용한다.

    트레이드 집합(진입/청산 봉)은 그대로여야 한다 — CW는 **사이징**이지 신호가 아니다.
    """
    from quantbox.backtest.sim_core import run_ma_cluster_live
    from quantbox.research.harness import MarketData

    cfg = VolStraddleConfig()
    md = MarketData.load(cfg.symbol_whitelist)

    base = run_ma_cluster_live(md.data, cw_sizing=False)
    cw = run_ma_cluster_live(md.data)

    assert len(base) > 0
    assert [t[:3] for t in base] == [t[:3] for t in cw], "CW가 트레이드 집합을 바꿨다 — 사이징이 신호에 샜다"

    allowed = {round(m, 10) for m in cfg.ma_cluster_cw_multipliers}
    ratios = {round(c[3] / b[3], 10) for b, c in zip(base, cw, strict=True) if b[3] != 0.0}
    assert ratios <= allowed, f"배수 테이블 밖의 스케일: {ratios - allowed}"
    assert ratios != {1.0}, "CW가 실제로 적용되지 않았다"


@pytest.mark.slow
@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_monolith_path_keeps_bit_parity_by_default() -> None:
    """`run_ma_cluster`는 기존 백테스트 숫자와의 bit-parity 계약이 우선 — CW는 opt-in."""
    from quantbox.backtest.sim_core import run_ma_cluster
    from quantbox.backtest.trade_sim import collect_ma_cluster_trades
    from quantbox.research.harness import MarketData

    md = MarketData.load(VolStraddleConfig().symbol_whitelist)

    assert run_ma_cluster(md.data) == collect_ma_cluster_trades(md.data), "기본값이 CW ON으로 뒤집혔다"

    cw = run_ma_cluster(md.data, cw_sizing=True)
    base = run_ma_cluster(md.data)
    assert [t[:3] for t in cw] == [t[:3] for t in base]
    assert any(c[3] != b[3] for b, c in zip(base, cw, strict=True)), "opt-in CW가 아무 효과도 없다"
