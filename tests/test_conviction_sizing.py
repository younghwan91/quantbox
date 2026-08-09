"""G8 파리티: MA cluster conviction-weighted sizing을 백테스트에 이식.

라이브(`strategy.py:500-516`)는 진입 시 3개 조건으로 CW 점수(0~3)를 매기고
`ma_cluster_cw_multipliers`로 명목가를 0.5~1.5배 스케일한다. 백테스트에는 이 축이
아예 없어서 모든 트레이드를 1.0배로 계산해 왔다 — 사이징 축이 통째로 달랐다.

여기 테스트는 이식된 스코어러가 **라이브와 같은 판정**을 내는지 고정한다.
"""

from pathlib import Path

import numpy as np
import pytest

from quantbox.backtest.conviction import ma_cluster_cw_mults, ma_cluster_cw_scores
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"


def _flat(n: int, value: float) -> np.ndarray:
    return np.full(n, value, dtype=np.float64)


def _bars(n: int = 25) -> dict[str, np.ndarray]:
    """CW 3조건이 모두 거짓인 기본 봉 세트 (조건별로 하나씩 켜서 쓴다)."""
    cfg = VolStraddleConfig()
    return {
        # spread = (max-min)/close*100 = 3.0% > 1.5% 임계 → 조건1 거짓
        "close": _flat(n, 100.0),
        "sma_20": _flat(n, 100.0),
        "sma_50": _flat(n, 103.0),
        "sma_100": _flat(n, 103.0),
        "sma_200": _flat(n, 103.0),
        # 평탄한 거래량 → 스파이크 없음 → 조건2 거짓
        "volume": _flat(n, 100.0),
        # |mom| = 0 → 조건3 거짓
        "momentum": _flat(n, 0.0),
        "_cfg": cfg,
    }


def _scores(b: dict) -> np.ndarray:
    return ma_cluster_cw_scores(
        b["close"], b["sma_20"], b["sma_50"], b["sma_100"], b["sma_200"], b["momentum"], b["volume"], b["_cfg"]
    )


def test_all_three_conditions_give_max_score() -> None:
    """좁은 스프레드 + 거래량 스파이크 + 강한 모멘텀 → 점수 3 → 1.5배."""
    cfg = VolStraddleConfig()
    b = _bars()
    i = 24

    # 1) tight spread: (100.5-100)/100*100 = 0.5% < 1.5%
    b["sma_50"][i] = b["sma_100"][i] = b["sma_200"][i] = 100.5
    # 2) volume spike: mean(volume[5:25]) = (19*100+1000)/20 = 145, 1000 > 1.5*145
    b["volume"][i] = 1000.0
    # 3) strong momentum: |mom| > 2.0 * min_momentum
    b["momentum"][i] = cfg.ma_cluster_cw_mom_mult * cfg.ma_cluster_min_momentum * 10.0

    scores = _scores(b)
    assert scores[i] == 3

    mults = ma_cluster_cw_mults(scores, cfg)
    assert mults[i] == cfg.ma_cluster_cw_multipliers[3]


def test_no_conditions_give_min_score() -> None:
    """세 조건 모두 거짓 → 점수 0 → 0.5배 (라이브가 명목가를 절반으로 줄이는 구간)."""
    cfg = VolStraddleConfig()
    b = _bars()
    scores = _scores(b)
    assert scores[24] == 0
    assert ma_cluster_cw_mults(scores, cfg)[24] == cfg.ma_cluster_cw_multipliers[0]


def test_volume_condition_needs_20_bars() -> None:
    """20봉 미만 구간에서는 거래량 조건이 아예 평가되지 않는다 (라이브 `len>=20` 가드)."""
    cfg = VolStraddleConfig()
    b = _bars()
    # 거래량 스파이크를 봉 10에 둔다 — 라이브라면 창이 20봉이 안 돼 점수를 안 준다.
    b["volume"][10] = 1e6
    b["sma_50"][10] = b["sma_100"][10] = b["sma_200"][10] = 100.5  # 조건1만 참
    scores = _scores(b)
    assert scores[10] == 1, "20봉 미만인데 거래량 조건이 가산됐다"
    assert ma_cluster_cw_mults(scores, cfg)[10] == cfg.ma_cluster_cw_multipliers[1]


def test_spread_threshold_is_percent_units() -> None:
    """임계 비교는 퍼센트 단위 — 라이브가 `spread < threshold*100`으로 쓴다.

    비율 단위(0.015)로 잘못 비교하면 거의 모든 봉이 조건1을 통과해 점수가 부풀려진다.
    """
    b = _bars()
    i = 24
    # spread = 2.0% — 임계 1.5%보다 넓으므로 조건1은 거짓이어야 한다.
    b["sma_50"][i] = b["sma_100"][i] = b["sma_200"][i] = 102.0
    assert _scores(b)[i] == 0


# ──────────────────────────────────────────────────────────────────────────
#  sim 배선 — 실데이터 (라이브 충실 경로에는 기본 ON, 모놀리식 재현 경로는 opt-in)
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
