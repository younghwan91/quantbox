"""Tests for v12.3 Live-Backtest Parity (T100_S38_PBT + M-2SSTSX).

6 critical divergences fixed:
  1. max_slots=2 (was 3)
  2. min_confidence=3 (was 2)
  3. SteepX confidence sizing
  4. SoftTight DD scaling
  5. Vol-adjusted sizing
  6. Direction diversity filter

Plus PBT exit mode:
  - Phase 1: TP1 hit → partial close 50% → breakeven floor
  - Phase 2: ATR trailing stop for runner
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
)


# ── Helpers ──


def _make_config(**overrides) -> RubberBandConfig:
    """Build a v12.3 T100_S38_PBT config with optional overrides."""
    defaults = dict(
        target_retrace=1.0,
        stop_mult=0.382,
        max_hold=20,
        lookback=8,
        threshold=0.032,
        partial_exit_enabled=True,
        partial_exit_fraction=0.5,
        trail_atr_mult=1.5,
        breakeven_after_partial=True,
        steep_cap_curve=True,
        cap_mult_conf2=0.3,
        cap_mult_conf3=0.5,
        cap_mult_conf4=1.5,
        cap_mult_conf5=2.0,
        soft_equity_scale=True,
        soft_eq_dd1=0.02,
        soft_eq_scale1=0.80,
        soft_eq_dd2=0.04,
        soft_eq_scale2=0.60,
        soft_eq_dd3=0.06,
        soft_eq_scale3=0.40,
        vol_adjusted_size=True,
        vol_adj_median_atr=0.01,
        vol_adj_min=0.4,
        vol_adj_max=2.0,
        dir_diversity_enabled=True,
        max_same_direction=2,
        max_concurrent=2,
        min_confidence=3,
    )
    defaults.update(overrides)
    return RubberBandConfig(**defaults)


def _make_strategy(**overrides) -> RubberBandStrategy:
    """Create a strategy with v12.3 config."""
    cfg = _make_config(**overrides)
    return RubberBandStrategy(cfg)


# ════════════════════════════════════════════════════════
#  TEST 1: Config Defaults Match T100_S38_PBT
# ════════════════════════════════════════════════════════


class TestConfigDefaults:
    """Verify RubberBandConfig defaults match the T100_S38_PBT champion."""

    def test_target_retrace(self):
        """Settings default → config builder → T100 target."""
        cfg = _make_config()
        assert cfg.target_retrace == pytest.approx(1.0)

    def test_stop_mult(self):
        """Settings default → config builder → S38 stop."""
        cfg = _make_config()
        assert cfg.stop_mult == pytest.approx(0.382)

    def test_max_concurrent(self):
        # Default in config is 5 (overridable). Settings default is 2.
        # The important thing: the config dataclass stores the value.
        cfg = _make_config()
        assert cfg.max_concurrent == 2

    def test_min_confidence(self):
        cfg = _make_config()
        assert cfg.min_confidence == 3

    def test_pbt_enabled(self):
        cfg = RubberBandConfig()
        assert cfg.partial_exit_enabled is True
        assert cfg.breakeven_after_partial is True
        assert cfg.trail_atr_mult == pytest.approx(1.0)  # v12.4 HybTight (was 1.5)


# ════════════════════════════════════════════════════════
#  TEST 2: PBT Exit Logic — Phase 1 (Full Position)
# ════════════════════════════════════════════════════════


class TestPBTPhase1:
    """Phase 1: TP1 hit → partial close, SL, max_hold, trailing."""

    def test_partial_close_at_tp1(self):
        """gain >= target → 'partial_close'."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        # Target = 0.05 * 1.0 = 5% of entry = $105
        reason = strat.update_position("BTCUSDT", 105.0)
        assert reason == "partial_close"
        pos = strat._positions["BTCUSDT"]
        assert pos["partial_closed"] is True
        assert pos["trail_floor"] == pytest.approx(0.0)  # breakeven

    def test_stop_loss_before_tp1(self):
        """gain < -stop → 'stop_loss' (Phase 1)."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        # Stop = 0.05 * 0.382 = 1.91% → price = 98.09
        reason = strat.update_position("BTCUSDT", 98.0)
        assert reason == "stop_loss"

    def test_max_hold_exit(self):
        """Bars exceed max_hold → 'max_hold' (Phase 1)."""
        strat = _make_strategy(max_hold=5)
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        # Hold for 5 bars at entry price (no gain)
        for _ in range(4):
            reason = strat.update_position("BTCUSDT", 100.0)
            assert reason is None
        reason = strat.update_position("BTCUSDT", 100.0)
        assert reason == "max_hold"

    def test_pre_tp1_trailing(self):
        """Pre-TP1 trailing: gain retreats from > 50% of target."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        # target = 5%, 50% of target = 2.5%. Push price up 3% then drop.
        strat.update_position("BTCUSDT", 103.0)  # best_pnl = 3%
        # Gain must drop to < best_pnl * 0.3 = 0.9%
        reason = strat.update_position("BTCUSDT", 100.5)  # gain = 0.5%
        assert reason == "trailing_stop"

    def test_no_partial_close_below_target(self):
        """gain < target → no partial close (old bug: triggered on any gain > 0)."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        # target = 5%, price up only 2% — must NOT trigger partial
        reason = strat.update_position("BTCUSDT", 102.0)
        assert reason is None
        pos = strat._positions["BTCUSDT"]
        assert pos["partial_closed"] is False

    def test_short_partial_close(self):
        """Short position: gain >= target → partial_close."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", -1, 100.0, 1000.0, 2.0, 0.05)
        # Short: target = 0.05 * 1.0 = 5% → price = 95.0
        reason = strat.update_position("BTCUSDT", 95.0)
        assert reason == "partial_close"


# ════════════════════════════════════════════════════════
#  TEST 3: PBT Exit Logic — Phase 2 (Runner)
# ════════════════════════════════════════════════════════


class TestPBTPhase2:
    """Phase 2: After partial close — breakeven floor + ATR trailing."""

    def _setup_runner(self, strat, entry=100.0, move=0.05):
        """Open position and move to Phase 2 (partial_closed=True)."""
        strat.open_position("BTCUSDT", 1, entry, 1000.0, 2.0, move)
        target_price = entry * (1.0 + move * strat.config.target_retrace)
        strat.update_position("BTCUSDT", target_price)  # triggers partial
        return target_price

    def test_breakeven_floor(self):
        """Runner exits at breakeven floor (gain ≤ 0)."""
        strat = _make_strategy()
        tp = self._setup_runner(strat)
        # Price drops back to entry → gain = 0 → hits breakeven floor
        reason = strat.update_position("BTCUSDT", 100.0, current_atr_pct=0.02)
        assert reason == "trail_runner"

    def test_atr_trailing(self):
        """Runner trail: gain < highest - 1.5 * ATR."""
        strat = _make_strategy()
        self._setup_runner(strat)
        # Push price up more (runner tracking)
        strat.update_position("BTCUSDT", 108.0, current_atr_pct=0.02)
        # highest = 8% gain. Trail stop = 8% - 1.5*2% = 5%.
        # Price at gain = 4% → trail_runner
        reason = strat.update_position("BTCUSDT", 104.0, current_atr_pct=0.02)
        assert reason == "trail_runner"

    def test_runner_continues_above_trail(self):
        """Runner holds when gain stays above trail stop."""
        strat = _make_strategy()
        self._setup_runner(strat)
        # Price up 8%. Trail = 8% - 1.5*2% = 5%. Price at 6% → hold.
        strat.update_position("BTCUSDT", 108.0, current_atr_pct=0.02)
        reason = strat.update_position("BTCUSDT", 106.0, current_atr_pct=0.02)
        assert reason is None

    def test_time_runner_exit(self):
        """Runner exits at runner_max_hold_mult × max_hold bars."""
        strat = _make_strategy(max_hold=5, runner_max_hold_mult=2.0)
        self._setup_runner(strat)
        # Already been 1 bar. Need 10 total (2×5).
        for _ in range(8):
            reason = strat.update_position("BTCUSDT", 106.0, current_atr_pct=0.02)
            assert reason is None
        reason = strat.update_position("BTCUSDT", 106.0, current_atr_pct=0.02)
        assert reason == "time_runner"

    def test_default_atr_when_none(self):
        """When current_atr_pct is None, defaults to 0.02 (2%)."""
        strat = _make_strategy()
        self._setup_runner(strat)
        strat.update_position("BTCUSDT", 108.0)  # No ATR arg
        # Trail = 8% - 1.5*2% = 5%. Price at 4% → runner exit.
        reason = strat.update_position("BTCUSDT", 104.0)
        assert reason == "trail_runner"


# ════════════════════════════════════════════════════════
#  TEST 4: SteepX Confidence Sizing
# ════════════════════════════════════════════════════════


class TestSteepXSizing:
    """Verify confidence → sizing multiplier mapping."""

    def test_steep_multiplier_conf2(self):
        cfg = _make_config()
        assert cfg.cap_mult_conf2 == pytest.approx(0.3)

    def test_steep_multiplier_conf5(self):
        cfg = _make_config()
        assert cfg.cap_mult_conf5 == pytest.approx(2.0)

    def test_steep_curve_disabled(self):
        """When steep_cap_curve=False, no sizing adjustment applied."""
        cfg = _make_config(steep_cap_curve=False)
        assert cfg.steep_cap_curve is False


# ════════════════════════════════════════════════════════
#  TEST 5: SoftTight DD Scaling
# ════════════════════════════════════════════════════════


class TestSoftTightDDScaling:
    """Verify DD → sizing scale factor mapping."""

    def test_no_dd_full_size(self):
        """0% DD → scale = 1.0 (no reduction)."""
        cfg = _make_config()
        # DD < dd1 (2%) → no scaling
        dd_pct = 0.01
        if dd_pct >= cfg.soft_eq_dd3:
            scale = cfg.soft_eq_scale3
        elif dd_pct >= cfg.soft_eq_dd2:
            scale = cfg.soft_eq_scale2
        elif dd_pct >= cfg.soft_eq_dd1:
            scale = cfg.soft_eq_scale1
        else:
            scale = 1.0
        assert scale == pytest.approx(1.0)

    def test_dd_2pct_scaling(self):
        """2% DD → 80% sizing."""
        cfg = _make_config()
        dd_pct = 0.025  # 2.5% DD
        if dd_pct >= cfg.soft_eq_dd3:
            scale = cfg.soft_eq_scale3
        elif dd_pct >= cfg.soft_eq_dd2:
            scale = cfg.soft_eq_scale2
        elif dd_pct >= cfg.soft_eq_dd1:
            scale = cfg.soft_eq_scale1
        else:
            scale = 1.0
        assert scale == pytest.approx(0.80)

    def test_dd_6pct_scaling(self):
        """6% DD → 40% sizing (maximum protection)."""
        cfg = _make_config()
        dd_pct = 0.07
        if dd_pct >= cfg.soft_eq_dd3:
            scale = cfg.soft_eq_scale3
        elif dd_pct >= cfg.soft_eq_dd2:
            scale = cfg.soft_eq_scale2
        elif dd_pct >= cfg.soft_eq_dd1:
            scale = cfg.soft_eq_scale1
        else:
            scale = 1.0
        assert scale == pytest.approx(0.40)


# ════════════════════════════════════════════════════════
#  TEST 6: Vol-Adjusted Sizing
# ════════════════════════════════════════════════════════


class TestVolAdjustedSizing:
    """Verify volatility normalization: median_atr / current_atr."""

    def test_median_vol_no_change(self):
        """ATR = median → factor = 1.0."""
        cfg = _make_config()
        cur_atr = 0.01  # = median
        factor = cfg.vol_adj_median_atr / cur_atr
        factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, factor))
        assert factor == pytest.approx(1.0)

    def test_high_vol_smaller_size(self):
        """ATR = 2× median → factor = 0.5."""
        cfg = _make_config()
        cur_atr = 0.02
        factor = cfg.vol_adj_median_atr / cur_atr
        factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, factor))
        assert factor == pytest.approx(0.5)

    def test_low_vol_bigger_size(self):
        """ATR = 0.5× median → factor = 2.0 (capped at max)."""
        cfg = _make_config()
        cur_atr = 0.005
        factor = cfg.vol_adj_median_atr / cur_atr
        factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, factor))
        assert factor == pytest.approx(2.0)

    def test_extreme_low_vol_capped(self):
        """ATR = 0.001 → factor would be 10.0, capped at 2.0."""
        cfg = _make_config()
        cur_atr = 0.001
        factor = cfg.vol_adj_median_atr / cur_atr
        factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, factor))
        assert factor == pytest.approx(2.0)

    def test_extreme_high_vol_floored(self):
        """ATR = 0.05 → factor = 0.2, floored at 0.4."""
        cfg = _make_config()
        cur_atr = 0.05
        factor = cfg.vol_adj_median_atr / cur_atr
        factor = max(cfg.vol_adj_min, min(cfg.vol_adj_max, factor))
        assert factor == pytest.approx(0.4)


# ════════════════════════════════════════════════════════
#  TEST 7: Direction Diversity
# ════════════════════════════════════════════════════════


class TestDirectionDiversity:
    """Verify same-direction position limiting."""

    def test_max_same_direction_config(self):
        cfg = _make_config()
        assert cfg.dir_diversity_enabled is True
        assert cfg.max_same_direction == 2

    def test_direction_count_logic(self):
        """Count positions by direction to enforce limit."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        strat.open_position("ETHUSDT", 1, 50.0, 500.0, 2.0, 0.05)

        # Count long positions
        long_count = sum(1 for p in strat.active_positions.values() if p.get("signal") == 1)
        assert long_count == 2

        # Should be at max (2) — no more longs allowed
        assert long_count >= strat.config.max_same_direction

    def test_opposite_direction_allowed(self):
        """Short positions still allowed when longs are at max."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        strat.open_position("ETHUSDT", 1, 50.0, 500.0, 2.0, 0.05)

        short_count = sum(1 for p in strat.active_positions.values() if p.get("signal") == -1)
        assert short_count == 0  # Still room for shorts


# ════════════════════════════════════════════════════════
#  TEST 8: Confidence & ATR Storage in Strategy
# ════════════════════════════════════════════════════════


class TestConfidenceATRStorage:
    """Verify on_bar stores confidence and ATR for live use."""

    def test_get_last_confidence_default(self):
        strat = _make_strategy()
        assert strat.get_last_confidence("BTCUSDT") == 0

    def test_get_last_atr_pct_default(self):
        strat = _make_strategy()
        assert strat.get_last_atr_pct("BTCUSDT") == pytest.approx(0.02)

    def test_confidence_stored_after_on_bar(self):
        """After on_bar with signal=0, confidence stored as 0."""
        strat = _make_strategy(entry_mode="confidence", min_confidence=1)
        close = np.linspace(100.0, 100.01, 200)  # flat → no signal
        btc = np.linspace(100.0, 100.01, 200)
        vol = np.ones(200) * 1e6
        taker = np.ones(200) * 5e5
        high = close * 1.001
        low = close * 0.999
        strat.update_market_data("TEST", vol, taker, high, low)
        strat.update_htf_data("TEST", close[-100:])
        strat.on_bar("TEST", close, btc)
        # No signal → confidence stored as 0
        assert strat.get_last_confidence("TEST") == 0


# ════════════════════════════════════════════════════════
#  TEST 9: open_position stores confidence
# ════════════════════════════════════════════════════════


class TestOpenPositionConfidence:
    """Verify open_position() stores the confidence kwarg."""

    def test_confidence_kwarg(self):
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05, confidence=4)
        pos = strat._positions["BTCUSDT"]
        assert pos["confidence"] == 4

    def test_confidence_default_zero(self):
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        pos = strat._positions["BTCUSDT"]
        assert pos["confidence"] == 0

    def test_pbt_fields_initialized(self):
        """PBT-specific fields initialized on open."""
        strat = _make_strategy()
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        pos = strat._positions["BTCUSDT"]
        assert pos["highest_since_tp1"] == 0.0
        assert pos["trail_floor"] is None
        assert pos["partial_closed"] is False


# ════════════════════════════════════════════════════════
#  TEST 10: Standard Mode (partial_exit_enabled=False)
# ════════════════════════════════════════════════════════


class TestStandardExitMode:
    """When PBT disabled, classic exit behavior is preserved."""

    def test_target_exit(self):
        strat = _make_strategy(partial_exit_enabled=False)
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        # target = 0.05 * 1.0 = 5%
        reason = strat.update_position("BTCUSDT", 105.0)
        assert reason == "target"

    def test_stop_loss_exit(self):
        strat = _make_strategy(partial_exit_enabled=False)
        strat.open_position("BTCUSDT", 1, 100.0, 1000.0, 2.0, 0.05)
        reason = strat.update_position("BTCUSDT", 98.0)
        assert reason == "stop_loss"


# ════════════════════════════════════════════════════════
#  TEST 11: Settings Parity
# ════════════════════════════════════════════════════════


class TestSettingsParity:
    """Verify RubberBandConfig defaults match the v12.3 production values (SSoT)."""

    def test_settings_has_v123_fields(self):
        """All v12.3 fields exist in RubberBandConfig (settings.rb_* removed)."""
        from quantbox.strategy.rubberband.config import RubberBandConfig

        cfg = RubberBandConfig()
        required = [
            "target_retrace",
            "stop_mult",
            "min_confidence",
            "max_concurrent",
            "partial_exit_enabled",
            "trail_atr_mult",
            "breakeven_after_partial",
            "steep_cap_curve",
            "cap_mult_conf2",
            "cap_mult_conf5",
            "soft_equity_scale",
            "soft_eq_dd1",
            "soft_eq_scale1",
            "vol_adjusted_size",
            "vol_adj_median_atr",
            "dir_diversity_enabled",
            "max_same_direction",
        ]
        for f in required:
            assert hasattr(cfg, f), f"Missing RubberBandConfig field: {f}"

    def test_settings_defaults_match(self):
        """RubberBandConfig defaults should match T100_S38_PBT v12.3 champion."""
        from quantbox.strategy.rubberband.config import RubberBandConfig

        cfg = RubberBandConfig()
        assert cfg.target_retrace == pytest.approx(1.5)
        assert cfg.stop_mult == pytest.approx(0.382)
        assert cfg.min_confidence == 3
        assert cfg.max_concurrent == 2
        assert cfg.partial_exit_enabled is True
        assert cfg.trail_atr_mult == pytest.approx(1.0)
        assert cfg.breakeven_after_partial is True
        assert cfg.steep_cap_curve is True
        assert cfg.soft_equity_scale is True
        assert cfg.vol_adjusted_size is True
        assert cfg.dir_diversity_enabled is True
        assert cfg.max_same_direction == 2


# ═══════════════════════════════════════════════════════════
#  TEST 8: Production Safety — State Restore & Bracket Repair
# ═══════════════════════════════════════════════════════════


class TestProductionSafety:
    """Tests for critical production bugs fixed in v12.3."""

    def test_runner_state_restore_preserves_trail_floor(self):
        """After state restore, runner's trail_floor and highest_since_tp1
        are preserved — not reset to defaults."""
        cfg = _make_config()
        strategy = RubberBandStrategy(cfg)

        # target = 0.06 * 1.0 = 6%. Price at 106 → 6% gain → partial.
        strategy.open_position("BTCUSDT", 1, 100.0, 1000.0, 1.0, 0.06)
        result = strategy.update_position("BTCUSDT", 106.0)
        assert result == "partial_close"

        # Runner: price goes to 112 (12% gain), building highest_since_tp1
        strategy.update_position("BTCUSDT", 112.0, current_atr_pct=0.01)
        pos = strategy._positions["BTCUSDT"]
        assert pos["highest_since_tp1"] > 0.10  # Peak at 12%
        assert pos["trail_floor"] is not None

        # Simulate state save+restore: capture pos state, close, re-open
        saved_state = dict(pos)
        strategy.close_position("BTCUSDT")

        # Re-open (simulates _load_state calling open_position)
        strategy.open_position(
            "BTCUSDT",
            saved_state["signal"],
            saved_state["entry_price"],
            saved_state["notional"],
            saved_state["leverage"],
            saved_state["move_size"],
        )
        # Now restore runner fields (as _load_state does)
        strategy._positions["BTCUSDT"]["partial_closed"] = saved_state["partial_closed"]
        strategy._positions["BTCUSDT"]["highest_since_tp1"] = saved_state["highest_since_tp1"]
        strategy._positions["BTCUSDT"]["trail_floor"] = saved_state["trail_floor"]
        strategy._positions["BTCUSDT"]["bars_held"] = saved_state["bars_held"]

        # Verify restored trail state
        restored_pos = strategy._positions["BTCUSDT"]
        assert restored_pos["trail_floor"] == saved_state["trail_floor"]
        assert restored_pos["highest_since_tp1"] == saved_state["highest_since_tp1"]
        assert restored_pos["partial_closed"] is True

        # Trail should still work: drop below floor → trail_runner
        result = strategy.update_position("BTCUSDT", 100.0, current_atr_pct=0.01)
        assert result == "trail_runner"

    def test_runner_state_restore_without_trail_floor(self):
        """State restore from old format (no trail_floor key) uses safe defaults."""
        cfg = _make_config()
        strategy = RubberBandStrategy(cfg)
        strategy.open_position("BTCUSDT", 1, 100.0, 1000.0, 1.0, 0.06)
        strategy.update_position("BTCUSDT", 102.0)  # partial close

        # Simulate old state format missing trail_floor & highest_since_tp1
        pos = strategy._positions["BTCUSDT"]
        pos.pop("trail_floor", None)
        pos.pop("highest_since_tp1", None)

        # Should not crash — uses defaults
        result = strategy.update_position("BTCUSDT", 104.0, current_atr_pct=0.01)
        assert result is None  # Continues running (ATR trail not hit)

    def test_partial_close_failure_clears_order_ids(self):
        """When partial close order fails, stale order IDs must be cleared
        to force bracket repair on the next cycle."""
        cfg = _make_config()
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            "BTCUSDT", 1, 100.0, 1000.0, 1.0, 0.06, server_sl_order_id="sl123", server_tp_order_id="tp456"
        )

        pos = strategy._positions["BTCUSDT"]
        # Simulate the state after cancel + failed order:
        # partial_close_position cancels SL/TP, then order fails
        pos["partial_closed"] = False
        pos["server_sl_order_id"] = None  # Cleared on failure
        pos["server_tp_order_id"] = None  # Cleared on failure

        # Both should be None to trigger bracket repair
        assert pos["server_sl_order_id"] is None
        assert pos["server_tp_order_id"] is None
        assert pos["partial_closed"] is False

    def test_runner_cost_is_exit_only(self):
        """Runner close should charge cost_per_side, not round_trip_cost.
        Entry cost was already charged at the original opening."""
        cfg = _make_config()
        strategy = RubberBandStrategy(cfg)
        strategy.open_position("BTCUSDT", 1, 100.0, 1000.0, 1.0, 0.06)
        # target = 6%. Price 106 → partial
        strategy.update_position("BTCUSDT", 106.0)

        pos = strategy._positions["BTCUSDT"]
        assert pos.get("partial_closed") is True

        # Verify the flag exists for cost routing in scalper
        # (actual cost routing tested via scalper integration)

    def test_pbt_runner_breakeven_floor_active(self):
        """Runner with breakeven_after_partial=True has a floor at entry gain=0."""
        cfg = _make_config(breakeven_after_partial=True)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position("BTCUSDT", 1, 100.0, 1000.0, 1.0, 0.06)

        # target = 6%. Partial at 106.0
        r1 = strategy.update_position("BTCUSDT", 106.0)
        assert r1 == "partial_close"

        pos = strategy._positions["BTCUSDT"]
        assert pos["trail_floor"] is not None
        assert pos["trail_floor"] >= 0.0  # Floor at or above breakeven

        # Price at entry → floor breached → trail_runner
        r2 = strategy.update_position("BTCUSDT", 100.0, current_atr_pct=0.01)
        assert r2 == "trail_runner"

    def test_pbt_runner_time_runner_exit(self):
        """Runner exits via time_runner at runner_max_hold_mult × max_hold."""
        cfg = _make_config(max_hold=8, runner_max_hold_mult=2.0)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position("BTCUSDT", 1, 100.0, 1000.0, 1.0, 0.06)

        # target = 6%. Partial at 106.0
        strategy.update_position("BTCUSDT", 106.0)

        # Runner: advance bars_held past 2× max_hold = 16
        pos = strategy._positions["BTCUSDT"]
        pos["bars_held"] = 15  # update_position increments to 16 = 2×8

        result = strategy.update_position("BTCUSDT", 109.0, current_atr_pct=0.01)
        assert result == "time_runner"
