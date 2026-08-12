"""Tests for the tiered slippage model in trade_sim.py."""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    cost_high_liq,
    cost_low_liq,
    cost_med_liq,
    _HIGH_LIQ_SYMS,
    _LOW_LIQ_SYMS,
    get_symbol_slippage,
    simulate_with_tiered_slippage,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ══════════════════════════════════════════════════════════════════════════════
#  get_symbol_slippage
# ══════════════════════════════════════════════════════════════════════════════


class TestGetSymbolSlippage:
    def test_high_liq_lower_than_low_liq(self) -> None:
        """High-liq symbol should have strictly lower cost than low-liq symbol."""
        assert get_symbol_slippage("BTCUSDT") < get_symbol_slippage("ICXUSDT")

    def test_high_liq_lower_than_med_liq(self) -> None:
        """High-liq symbol cheaper than mid-tier symbol."""
        assert get_symbol_slippage("ETHUSDT") < get_symbol_slippage("SANDUSDT")

    def test_med_liq_lower_than_low_liq(self) -> None:
        """Mid-tier symbol cheaper than thin/low-liq symbol."""
        assert get_symbol_slippage("SANDUSDT") < get_symbol_slippage("STORJUSDT")

    def test_high_liq_returns_correct_rate(self) -> None:
        assert get_symbol_slippage("SOLUSDT") == pytest.approx(cost_high_liq())

    def test_low_liq_returns_correct_rate(self) -> None:
        assert get_symbol_slippage("RLCUSDT") == pytest.approx(cost_low_liq())

    def test_med_liq_returns_correct_rate(self) -> None:
        # SANDUSDT is not in HIGH or LOW, so it should be MED
        assert "SANDUSDT" not in _HIGH_LIQ_SYMS
        assert "SANDUSDT" not in _LOW_LIQ_SYMS
        assert get_symbol_slippage("SANDUSDT") == pytest.approx(cost_med_liq())

    def test_conservative_adds_50pct_buffer(self) -> None:
        """Conservative mode should multiply cost by 1.5."""
        normal = get_symbol_slippage("BTCUSDT", conservative=False)
        conservative = get_symbol_slippage("BTCUSDT", conservative=True)
        assert conservative == pytest.approx(normal * 1.5)

    def test_conservative_applies_to_all_tiers(self) -> None:
        """Conservative 50% buffer applies to all three tiers."""
        for sym in ("BTCUSDT", "SANDUSDT", "ICXUSDT"):
            normal = get_symbol_slippage(sym, conservative=False)
            cons = get_symbol_slippage(sym, conservative=True)
            assert cons == pytest.approx(normal * 1.5), f"Failed for {sym}"

    def test_unknown_symbol_gets_med_cost(self) -> None:
        """Unknown/future symbol not in any tier set → medium cost."""
        cost = get_symbol_slippage("UNKNOWNUSDT")
        assert cost == pytest.approx(cost_med_liq())

    def test_tier_ordering(self) -> None:
        """HIGH < MED < LOW must hold."""
        assert cost_high_liq() < cost_med_liq() < cost_low_liq()

    def test_high_liq_matches_current_flat(self) -> None:
        """High-liq cost should equal the current flat cost (COST_DIRECTIONAL)."""
        assert cost_high_liq() == pytest.approx(COST_DIRECTIONAL)

    def test_all_whitelist_symbols_resolved(self) -> None:
        """Every symbol in the production whitelist should resolve to a valid cost."""
        cfg = VolStraddleConfig()
        for sym in cfg.symbol_whitelist:
            cost = get_symbol_slippage(sym)
            assert cost > 0, f"{sym} returned non-positive cost"
            assert cost <= cost_low_liq() * 1.5 + 1e-9, f"{sym} cost too high: {cost}"


# ══════════════════════════════════════════════════════════════════════════════
#  simulate_with_tiered_slippage
# ══════════════════════════════════════════════════════════════════════════════


def _make_minimal_data(n_bars: int = 500) -> tuple[dict, dict, dict]:
    """Build minimal synthetic OHLCV data for one symbol to test PnL adjustment."""
    np.random.seed(42)
    close = 100.0 * np.cumprod(1 + np.random.randn(n_bars) * 0.005)
    high = close * 1.003
    low = close * 0.997
    vr = np.random.uniform(0.3, 0.6, n_bars)
    mom = np.where(np.random.randn(n_bars) > 0, 1, -1).astype(float)
    data = {
        "BTCUSDT": {
            "close": close,
            "high": high,
            "low": low,
            "vr_8_56": vr,
            "mom_8": mom,
        }
    }
    return data, {}, {}


class TestSimulateWithTieredSlippage:
    """Tests using a minimal synthetic dataset (BTC only)."""

    def _make_cfg(self) -> dict:
        cfg_obj = VolStraddleConfig()
        return {
            "sp": cfg_obj.vr_short,
            "lp": cfg_obj.vr_long,
            "d_vr": cfg_obj.d_vr_threshold,
            "s_vr": 0.55,  # s_vr_threshold removed from production config
            "s_dur": 4,  # s_squeeze_dur removed from production config
            "d_sl": cfg_obj.d_sl,
            "s_sl": 0.015,  # s_sl removed from production config
            "tp": cfg_obj.d_tp,
        }

    def test_flat_mode_returns_same_as_collect_trades(self) -> None:
        """flat mode should produce identical output to collect_trades()."""
        from quantbox.backtest.trade_sim import collect_trades

        data_15m, data_1h, data_4h = _make_minimal_data()
        cfg = self._make_cfg()
        trades_flat = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
        trades_sim = simulate_with_tiered_slippage(cfg, data_15m, data_1h, data_4h, "D_mom8", "flat")
        assert len(trades_flat) == len(trades_sim)
        for (s1, e1, x1, p1), (s2, e2, x2, p2) in zip(trades_flat, trades_sim):
            assert s1 == s2
            assert e1 == e2
            assert x1 == x2
            assert p1 == pytest.approx(p2)

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode string must raise ValueError."""
        data_15m, data_1h, data_4h = _make_minimal_data()
        cfg = self._make_cfg()
        with pytest.raises(ValueError, match="mode must be"):
            simulate_with_tiered_slippage(cfg, data_15m, data_1h, data_4h, "D_mom8", "invalid")

    def test_btcusdt_flat_equals_tiered(self) -> None:
        """BTCUSDT is HIGH_LIQ so tiered cost == flat cost → PnL unchanged."""
        data_15m, data_1h, data_4h = _make_minimal_data()
        cfg = self._make_cfg()
        trades_flat = simulate_with_tiered_slippage(cfg, data_15m, data_1h, data_4h, "D_mom8", "flat")
        trades_tiered = simulate_with_tiered_slippage(cfg, data_15m, data_1h, data_4h, "D_mom8", "tiered")
        assert len(trades_flat) == len(trades_tiered)
        for (_, _, _, p_flat), (_, _, _, p_tiered) in zip(trades_flat, trades_tiered):
            assert p_flat == pytest.approx(p_tiered), "BTCUSDT (HIGH_LIQ) must not have PnL adjusted in tiered mode"

    def test_low_liq_tiered_worse_pnl(self) -> None:
        """Low-liq symbol gets higher cost in tiered mode → lower PnL than flat."""
        np.random.seed(7)
        n = 500
        close = 1.0 * np.cumprod(1 + np.random.randn(n) * 0.005)
        high = close * 1.003
        low = close * 0.997
        vr = np.random.uniform(0.3, 0.55, n)
        mom = np.where(np.random.randn(n) > 0, 1.0, -1.0)
        data_15m = {"ICXUSDT": {"close": close, "high": high, "low": low, "vr_8_56": vr, "mom_8": mom}}
        cfg = self._make_cfg()
        trades_flat = simulate_with_tiered_slippage(cfg, data_15m, {}, {}, "D_mom8", "flat")
        trades_tiered = simulate_with_tiered_slippage(cfg, data_15m, {}, {}, "D_mom8", "tiered")
        if not trades_flat:
            pytest.skip("No trades generated with this random seed")
        sum_flat = sum(p for _, _, _, p in trades_flat)
        sum_tiered = sum(p for _, _, _, p in trades_tiered)
        assert sum_tiered < sum_flat, "Low-liq symbol tiered cost must reduce total PnL vs flat"

    def test_conservative_worse_than_tiered(self) -> None:
        """Conservative mode costs more than tiered → lower PnL."""
        np.random.seed(7)
        n = 500
        close = 1.0 * np.cumprod(1 + np.random.randn(n) * 0.005)
        high = close * 1.003
        low = close * 0.997
        vr = np.random.uniform(0.3, 0.55, n)
        mom = np.where(np.random.randn(n) > 0, 1.0, -1.0)
        data_15m = {"ICXUSDT": {"close": close, "high": high, "low": low, "vr_8_56": vr, "mom_8": mom}}
        cfg = self._make_cfg()
        trades_tiered = simulate_with_tiered_slippage(cfg, data_15m, {}, {}, "D_mom8", "tiered")
        trades_cons = simulate_with_tiered_slippage(cfg, data_15m, {}, {}, "D_mom8", "conservative")
        if not trades_tiered:
            pytest.skip("No trades generated")
        sum_tiered = sum(p for _, _, _, p in trades_tiered)
        sum_cons = sum(p for _, _, _, p in trades_cons)
        assert sum_cons < sum_tiered, "Conservative mode (1.5× tiered) must be worse than plain tiered"

    def test_pnl_adjustment_arithmetic(self) -> None:
        """Verify the PnL delta formula for a known LOW_LIQ symbol."""
        # For D_mom8, delta = sym_cost - COST_DIRECTIONAL
        # For ICXUSDT (LOW_LIQ): sym_cost=0.0016, flat=0.0008 → delta=0.0008
        sym_cost = get_symbol_slippage("ICXUSDT")
        expected_delta = sym_cost - COST_DIRECTIONAL  # 0.0008
        assert expected_delta == pytest.approx(cost_low_liq() - cost_high_liq())


# ══════════════════════════════════════════════════════════════════════════════
#  VolStraddleConfig.slippage_mode validation
# ══════════════════════════════════════════════════════════════════════════════


class TestSlippageModeConfig:
    def test_default_is_flat(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.slippage_mode == "flat"

    def test_tiered_is_valid(self) -> None:
        cfg = VolStraddleConfig(slippage_mode="tiered")
        assert cfg.slippage_mode == "tiered"

    def test_conservative_is_valid(self) -> None:
        cfg = VolStraddleConfig(slippage_mode="conservative")
        assert cfg.slippage_mode == "conservative"

    def test_invalid_mode_raises(self) -> None:
        """Config must reject unknown slippage_mode values."""
        with pytest.raises(Exception):  # pydantic ValidationError
            VolStraddleConfig(slippage_mode="invalid")

    def test_another_invalid_mode_raises(self) -> None:
        with pytest.raises(Exception):
            VolStraddleConfig(slippage_mode="FLAT")  # case-sensitive

    def test_round_trip_cost_unchanged_by_slippage_mode(self) -> None:
        """round_trip_cost property is fee+slippage only; mode doesn't affect it."""
        cfg_flat = VolStraddleConfig(slippage_mode="flat")
        cfg_tiered = VolStraddleConfig(slippage_mode="tiered")
        assert cfg_flat.round_trip_cost == pytest.approx(cfg_tiered.round_trip_cost)
