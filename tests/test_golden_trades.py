"""골든 트레이드 회귀 — "기존 백테스트 숫자는 안 바뀐다"의 유일한 근거.

`tests/fixtures/golden_trades.npz` 는 모놀리식 엔진(위임 전)의 실데이터 출력을 박제한 것이다.
`trade_sim` 이 `sim_core` 에 위임하도록 바뀐 뒤에도 이 테스트가 통과해야 한다.

왜 이게 필요한가: test_sim_core_parity.py 는 모놀리식과 sim_core 를 맞대어 본다. 위임 후에는
그 비교가 자기 자신과의 비교가 되어 의미를 잃는다. 박제본만이 바깥의 기준점이다.

이 테스트가 깨지면 백테스트 숫자가 바뀐 것이다 —
**픽스처를 재생성하지 말고, 왜 바뀌었는지 먼저 밝혀라.**
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"
FIXTURE = Path(__file__).parent / "fixtures" / "golden_trades.npz"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음"),
    pytest.mark.skipif(not FIXTURE.exists(), reason="골든 픽스처 없음"),
]


@pytest.fixture(scope="module")
def prepared():
    """collect_* 가 요구하는 형태로 실데이터 준비 (r116_prep_baseline 과 동일 순서)."""
    from quantbox.backtest.trade_sim import build_resampled, precompute, precompute_all_vr
    from quantbox.research.harness import MarketData
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    cfg = VolStraddleConfig()
    md = MarketData.load(cfg.symbol_whitelist)
    data_15m = md.data

    sp, lp = cfg.vr_short, cfg.vr_long
    precompute_all_vr([data_15m], [(sp, lp)])
    precompute(data_15m, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_1h, data_4h], [(sp, lp)])
    for ds in (data_1h, data_4h):
        precompute(ds, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    return cfg, data_15m, data_1h, data_4h


@pytest.fixture(scope="module")
def golden():
    return np.load(FIXTURE, allow_pickle=False)


def _assert_matches(golden, name: str, trades: list) -> None:
    trades = sorted(trades)
    exp_n = len(golden[f"{name}_sym"])
    assert exp_n > 0, f"{name}: 골든이 비어 있다 — 픽스처가 공허하다"
    assert len(trades) == exp_n, f"{name}: 트레이드 수가 {exp_n} -> {len(trades)} 로 바뀌었다"

    np.testing.assert_array_equal(
        np.array([t[0] for t in trades], dtype="<U16"), golden[f"{name}_sym"], err_msg=f"{name}: 심볼 불일치"
    )
    np.testing.assert_array_equal(
        np.array([t[1] for t in trades], dtype=np.int64), golden[f"{name}_entry"], err_msg=f"{name}: 진입봉 불일치"
    )
    np.testing.assert_array_equal(
        np.array([t[2] for t in trades], dtype=np.int64), golden[f"{name}_exit"], err_msg=f"{name}: 청산봉 불일치"
    )
    np.testing.assert_allclose(
        np.array([t[3] for t in trades], dtype=np.float64),
        golden[f"{name}_pnl"],
        rtol=0,
        atol=1e-12,
        err_msg=f"{name}: PnL 불일치",
    )


def test_ma_cluster_matches_golden(prepared, golden):
    from quantbox.backtest.trade_sim import collect_ma_cluster_trades

    _, data_15m, _, _ = prepared
    _assert_matches(golden, "ma_cluster", collect_ma_cluster_trades(data_15m))


def test_donchian_matches_golden(prepared, golden):
    from quantbox.backtest.trade_sim import collect_donchian_trades

    _, data_15m, _, _ = prepared
    _assert_matches(golden, "donchian", collect_donchian_trades(data_15m))


def test_directional_matches_golden(prepared, golden):
    from quantbox.backtest.trade_sim import collect_trades

    cfg, data_15m, data_1h, data_4h = prepared
    ct_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "d_sl": cfg.d_sl,
        "tp": cfg.d_tp,
        "d_trail_activate": cfg.d_trail_activate,
        "d_trail_distance": cfg.d_trail_distance,
    }
    _assert_matches(golden, "directional", collect_trades(ct_cfg, data_15m, data_1h, data_4h, "D_mom8"))


def test_golden_fixture_is_nonvacuous(golden):
    """세 전략 모두 실제 트레이드가 박제돼 있어야 한다."""
    for name, minimum in (("ma_cluster", 1000), ("donchian", 1000), ("directional", 500)):
        n = len(golden[f"{name}_sym"])
        assert n >= minimum, f"{name}: {n}건뿐 — 픽스처가 잘못 생성됐다"
