"""6종 편향 회귀 모음 (Phase 1 안전망) — 이번 세션에 찾은 백테스트 편향의 재발 차단.

각 테스트는 특정 편향이 '다시 생기면' CI에서 실패한다. 4종은 기존 테스트가 커버:
  - 진입 재구현(2.32x): tests/test_sim_core_live_parity.py
  - portfolio_size=1 폭발 / data_years 6종 / _offset 정렬: tests/test_harness.py
이 파일은 누락됐던 2종을 고정한다:
  - exit 드리프트 (r119류: trailing 1봉 지연 + TP-before-SL) → canonical sim_drive 동작 고정
  - ma_cluster 펀딩 미반영 → 현상태 명시 회귀(정정은 Phase 2)

상세: docs/refactor/research_audit.md, .omc/specs/deep-interview-parity-core-refactor.md
"""

from __future__ import annotations

import pathlib

import numpy as np

from quantbox.backtest.sim_core import KIND_ACTIVE, sim_drive


def _const_arrays(n: int):
    """진입이 매 봉 후보인 최소 구성(bar_kind=ACTIVE, entry_ok=True, dir=+1)."""
    bar_kind = np.full(n, KIND_ACTIVE, dtype=np.int8)
    entry_ok = np.ones(n, dtype=np.bool_)
    direction = np.ones(n, dtype=np.int8)
    return bar_kind, entry_ok, direction


# ──────────────────────────────────────────────────────────────────────────
#  편향 #2 — exit 드리프트 (r119류). canonical sim_drive는 SL-first + 활성화 즉시.
# ──────────────────────────────────────────────────────────────────────────


def test_sl_checked_before_tp_when_not_trailing():
    """not-trailing 봉에서 SL과 TP가 같은 봉에 동시 도달하면 SL이 먼저다.

    r119류 드리프트는 TP를 먼저 체크해 같은 봉을 TP(+tp)로 청산 → PF 부풀림.
    canonical sim_drive는 SL(-sl)을 먼저. 이 테스트가 그 순서를 고정한다.
    """
    # trail_activate를 크게(0.5) 잡아 trailing 미발동 → not-trailing 경로.
    close = np.array([100.0, 100.0], dtype=np.float64)
    high = np.array([100.0, 121.0], dtype=np.float64)  # TP(120) 도달
    low = np.array([100.0, 97.0], dtype=np.float64)  # SL(98) 도달
    bar_kind, entry_ok, direction = _const_arrays(2)

    pnl, ent, ext, _d, _r = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,  # min_duration=1, max_duration=off
        0.02,
        0.20,
        0.5,
        0.005,
        5,
        0.0,
        0,  # sl, tp, trail_activate=0.5(미발동), trail_dist, max_hold, cost, warmup
        2,
        0,
    )
    assert len(pnl) == 1, "정확히 1 트레이드여야 한다"
    # SL-first면 pnl = -sl. TP-first(r119류)면 +tp가 나온다.
    assert pnl[0] == -0.02, f"SL-first 위반: pnl={pnl[0]} (TP-first 드리프트면 +0.20)"


def test_trailing_activates_same_bar_not_next():
    """peak_gain이 trail_activate에 도달한 '그 봉'에서 trailing이 즉시 켜진다.

    r119류는 활성화를 봉 끝에서 판정 → 그 봉은 아직 not-trailing(SL) 로직 → 1봉 지연.
    canonical은 peak 갱신 직후 즉시 활성화 → 같은 봉에서 trail 청산 가능.
    """
    # 진입 100. 한 봉에서 high로 peak_gain=0.03(>=trail_activate=0.01) → 즉시 활성화.
    # 같은 봉 low가 trail_level(peak-dist) 아래로 → trailing 청산.
    # trail_level = 100*(1+0.03-0.005)=102.5. low=102 <= 102.5 → trail exit.
    close = np.array([100.0, 100.0], dtype=np.float64)
    high = np.array([100.0, 103.0], dtype=np.float64)  # peak_gain=0.03
    low = np.array([100.0, 102.0], dtype=np.float64)  # trail_level=102.5 도달
    bar_kind, entry_ok, direction = _const_arrays(2)

    pnl, ent, ext, _d, _r = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,
        0.02,
        0.20,
        0.01,
        0.005,
        5,
        0.0,
        0,  # trail_activate=0.01, trail_dist=0.005
        2,
        0,
    )
    assert len(pnl) == 1
    # 활성화 즉시 → trail 청산: pnl = peak_gain - trail_dist = 0.03 - 0.005 = 0.025.
    # 1봉 지연(r119류)이면 그 봉은 not-trailing → 청산 안 됨(SL/TP 미도달) → 다음 봉 or maxhold.
    assert abs(pnl[0] - 0.025) < 1e-9, f"활성화 즉시 위반: pnl={pnl[0]} (지연이면 0.025 아님)"


def test_short_direction_symmetry():
    """숏에서도 SL-first가 유지된다(long/short 대칭 회귀)."""
    close = np.array([100.0, 100.0], dtype=np.float64)
    high = np.array([100.0, 103.0], dtype=np.float64)  # 숏 SL(102=100*1.02) 도달
    low = np.array([100.0, 79.0], dtype=np.float64)  # 숏 TP(80=100*0.80) 도달
    bar_kind = np.full(2, KIND_ACTIVE, dtype=np.int8)
    entry_ok = np.ones(2, dtype=np.bool_)
    direction = np.full(2, -1, dtype=np.int8)  # SHORT

    pnl, ent, ext, _d, _r = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,
        0.02,
        0.20,
        0.5,
        0.005,
        5,
        0.0,
        0,  # trail 미발동
        2,
        0,
    )
    assert len(pnl) == 1
    assert pnl[0] == -0.02, f"숏 SL-first 위반: pnl={pnl[0]}"


# ──────────────────────────────────────────────────────────────────────────
#  편향 #3 — ma_cluster 펀딩 미반영 (현상태 고정; 정정은 Phase 2)
# ──────────────────────────────────────────────────────────────────────────


def test_ma_cluster_funding_is_deducted():
    """ma_cluster 라이브 경로가 펀딩을 차감한다 (2026-08-10 정정).

    이전에는 VR(collect_trades)만 G3 파리티로 펀딩을 반영하고 ma_cluster(sim_core)는
    빠져 있었다. 보유가 길수록 드래그가 커지므로, **보유 기간을 비교하는 실험에서는
    이 누락이 장기 보유를 부당하게 유리하게 만든다.**

    차감은 `trade_sim.apply_funding_drag`와 같은 요율/주기를 쓴다.
    """
    import numpy as np

    from quantbox.backtest.sim_core import run_ma_cluster_live
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    cache = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache_extended"
    if not cache.is_dir():
        pytest.skip("확장 캐시 없음")

    from quantbox.research.harness import MarketData

    data = MarketData.load(["BTCUSDT"]).data
    on = run_ma_cluster_live(data, cw_sizing=False)
    off = run_ma_cluster_live(data, cw_sizing=False, funding=False)

    assert [t[:3] for t in on] == [t[:3] for t in off], "펀딩이 트레이드 집합을 바꿨다"
    drags = np.array([b[3] - a[3] for a, b in zip(on, off, strict=True)])
    assert (drags >= 0).all(), "펀딩이 손익을 늘렸다"
    assert drags.max() > 0, "펀딩이 실제로 차감되지 않았다"

    # 드래그는 보유 기간에 비례해야 한다.
    rate = VolStraddleConfig().default_funding_rate_8h
    holds = np.array([t[2] - t[1] for t in on], dtype=float)
    assert holds.max() > 0
    # entry_bar는 봉 시작을 가리키므로 종가 기준 보유는 (holds - (ratio-1)) 이다.
    expected = rate * ((holds - 15) / 32)
    np.testing.assert_allclose(drags, expected, atol=1e-12)
