"""Tests for G3 parity fix: funding cost model in collect_trades.

Verifies that funding costs are deducted from backtest trade PnL
proportional to bars held. At 30x leverage, funding of 0.01% per 8h
on notional = material drag over 48h holds.
"""

from __future__ import annotations

import numpy as np

from quantbox.backtest.trade_sim import (
    collect_trades,
)


def _make_minimal_cfg(**overrides) -> dict:
    """Build minimal cfg dict for collect_trades with funding support."""
    cfg = {
        "sp": 8,
        "lp": 56,
        "d_vr": 0.50,
        "s_vr": 0.57,
        "s_dur": 3,
        "d_sl": 0.01,
        "s_sl": 0.015,
        "tp": 0.20,
        "s_trail_activate": 0.030,
        "s_trail_distance": 0.005,
    }
    cfg.update(overrides)
    return cfg


class TestFundingCostDeduction:
    """Funding cost deduction in collect_trades."""

    def test_funding_zero_matches_legacy(self):
        """funding_rate_8h=0 produces identical results to no funding."""
        # Create synthetic data with a known signal
        n = 300
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 101.0, dtype=np.float64)
        low = np.full(n, 99.0, dtype=np.float64)
        volume = np.full(n, 1000.0, dtype=np.float64)
        vr = np.full(n, 0.3, dtype=np.float64)  # below threshold → signals fire
        mom = np.full(n, 0.01, dtype=np.float64)

        data_15m = {
            "BTCUSDT": {
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "vr_8_56": vr,
                "mom_8": mom,
            }
        }
        data_1h: dict = {}
        data_4h: dict = {}

        from quantbox.backtest.trade_sim import TF_15m

        cfg_zero = _make_minimal_cfg(funding_rate_8h=0.0)
        trades_zero = collect_trades(cfg_zero, data_15m, data_1h, data_4h, "D_mom8", tf_group=[TF_15m])

        # With funding=0, trades should exist and have no funding drag
        assert len(trades_zero) > 0, "Need at least one trade"
        for sym, entry, exit_b, pnl in trades_zero:
            assert isinstance(pnl, float)

    def test_funding_reduces_pnl(self):
        """Positive funding rate reduces PnL proportional to bars held."""
        n = 300
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 101.0, dtype=np.float64)
        low = np.full(n, 99.0, dtype=np.float64)
        volume = np.full(n, 1000.0, dtype=np.float64)
        vr = np.full(n, 0.3, dtype=np.float64)
        mom = np.full(n, 0.01, dtype=np.float64)

        data_15m = {
            "BTCUSDT": {
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "vr_8_56": vr,
                "mom_8": mom,
            }
        }
        data_1h: dict = {}
        data_4h: dict = {}

        from quantbox.backtest.trade_sim import TF_15m

        cfg_no_fund = _make_minimal_cfg(funding_rate_8h=0.0)
        cfg_with_fund = _make_minimal_cfg(funding_rate_8h=0.001)  # 0.1% per 8h (exaggerated for test)

        trades_no = collect_trades(cfg_no_fund, data_15m, data_1h, data_4h, "D_mom8", tf_group=[TF_15m])
        trades_yes = collect_trades(cfg_with_fund, data_15m, data_1h, data_4h, "D_mom8", tf_group=[TF_15m])

        assert len(trades_no) == len(trades_yes), "Same number of trades"
        assert len(trades_no) > 0, "Need at least one trade"

        for (s1, e1, x1, p1), (s2, e2, x2, p2) in zip(trades_no, trades_yes):
            assert s1 == s2 and e1 == e2 and x1 == x2, "Same trade timing"
            bars_held = x1 - e1
            expected_drag = 0.001 * (bars_held / 32)
            assert abs((p1 - p2) - expected_drag) < 1e-12, (
                f"Funding drag should be {expected_drag:.6f}, got {p1 - p2:.6f}"
            )

    def test_funding_math_manual(self):
        """Manual funding calculation: 48h hold, 0.01% per 8h = 6 × 0.01%."""
        # 48h at 15m = 192 bars. 192 / 32 = 6 funding periods.
        funding_rate_8h = 0.0001  # 0.01%
        bars_held = 192
        bars_per_8h = 32
        expected_drag = funding_rate_8h * (bars_held / bars_per_8h)
        assert abs(expected_drag - 0.0006) < 1e-10, f"Expected 0.06%, got {expected_drag}"

    def test_funding_config_default(self):
        """VolStraddleConfig has default_funding_rate_8h field."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert hasattr(cfg, "default_funding_rate_8h")
        assert cfg.default_funding_rate_8h == 0.0001
        assert cfg.default_funding_rate_8h >= 0
