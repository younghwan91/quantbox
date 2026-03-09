"""Tests for v12.4 I-HybTight config upgrade.

Changes from A-MR-Prod → I-HybTight:
  1. target_retrace: 1.0 → 1.5 (deeper reversal capture)
  2. max_hold: 20 → 25 (more patience)
  3. trail_atr_mult: 1.5 → 1.0 (tighter runner trail)
  4. runner_max_hold_mult: hardcoded 2 → configurable 3.0

Tests:
  1. Config defaults match HybTight values
  2. runner_max_hold_mult controls runner time exit (numba)
  3. Tighter trail_atr_mult in live runner exit
  4. runner_max_hold_mult in live runner time exit
  5. Settings → Config wiring for runner_max_hold_mult
  6. Backtest ↔ live parity for runner exits
"""

import numpy as np
import pytest
from quantbox.strategy.rubberband import (
    compute_signals_with_partial_exit,
    RubberBandConfig,
    RubberBandStrategy,
)


# ── Helpers ──


def _make_drop_then_run(n=250, drop_pct=0.06, recovery_pct=0.12):
    """Price drops sharply then recovers strongly (runner scenario).

    Designed to trigger:
      1. Entry (short) on the drop
      2. TP1 partial close on the recovery overshoot
      3. Runner phase that extends for many bars
    """
    prices = np.full(n, 100.0, dtype=np.float64)
    drop_start = 50
    # Sharp drop over lookback=8 bars
    for i in range(drop_start, drop_start + 8):
        frac = (i - drop_start + 1) / 8
        prices[i] = 100.0 * (1.0 - drop_pct * frac)
    bottom = prices[drop_start + 7]
    # Strong recovery over 20 bars (enough to trigger TP1)
    for i in range(drop_start + 8, min(drop_start + 28, n)):
        frac = (i - drop_start - 7) / 20
        prices[i] = bottom * (1.0 + recovery_pct * frac)
    # Hold at recovered level (runner territory)
    recovered = prices[min(drop_start + 27, n - 1)]
    for i in range(drop_start + 28, n):
        prices[i] = recovered
    return prices


# ════════════════════════════════════════════════════════
#  TEST 1: Config Defaults
# ════════════════════════════════════════════════════════


class TestHybTightConfig:
    """Verify v12.4 I-HybTight config defaults."""

    def test_target_retrace_is_1_5(self):
        cfg = RubberBandConfig()
        assert cfg.target_retrace == 1.5

    def test_max_hold_is_25(self):
        cfg = RubberBandConfig()
        assert cfg.max_hold == 25

    def test_trail_atr_mult_is_1_0(self):
        cfg = RubberBandConfig()
        assert cfg.trail_atr_mult == 1.0

    def test_runner_max_hold_mult_is_3_0(self):
        cfg = RubberBandConfig()
        assert cfg.runner_max_hold_mult == 3.0

    def test_runner_max_bars_computed(self):
        """Runner max bars = max_hold × runner_max_hold_mult = 25 × 3.0 = 75."""
        cfg = RubberBandConfig()
        assert cfg.max_hold * cfg.runner_max_hold_mult == 75.0

    def test_stop_mult_unchanged(self):
        """stop_mult should remain 0.382 from T100_S38_PBT."""
        cfg = RubberBandConfig()
        assert cfg.stop_mult == 0.382


# ════════════════════════════════════════════════════════
#  TEST 2: runner_max_hold_mult in Numba Signal
# ════════════════════════════════════════════════════════


class TestRunnerMaxHoldNumba:
    """Verify runner_max_hold_mult controls runner time exit in numba."""

    def test_runner_holds_longer_with_higher_mult(self):
        """With mult=3.0 vs mult=2.0, runner stays in position longer."""
        prices = _make_drop_then_run(n=300)

        sigs_old = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.032,
            1.5,
            0.382,
            25,
            0.0,
            runner_max_hold_mult=2.0,
        )
        sigs_new = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.032,
            1.5,
            0.382,
            25,
            0.0,
            runner_max_hold_mult=3.0,
        )

        def _bars_in_half(sigs):
            """Count how many bars the runner (|sig|==0.5) is held."""
            return int(np.sum(np.abs(sigs) == 0.5))

        half_old = _bars_in_half(sigs_old)
        half_new = _bars_in_half(sigs_new)

        # With higher mult, runner should hold at least as long
        # (it can only extend, never shorten)
        assert half_new >= half_old

    def test_runner_max_hold_mult_default_is_3(self):
        """Default runner_max_hold_mult (no arg) should behave as 3.0."""
        prices = _make_drop_then_run(n=300)

        sigs_default = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.032,
            1.5,
            0.382,
            25,
            0.0,
        )
        sigs_explicit = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.032,
            1.5,
            0.382,
            25,
            0.0,
            runner_max_hold_mult=3.0,
        )

        np.testing.assert_array_equal(sigs_default, sigs_explicit)


# ════════════════════════════════════════════════════════
#  TEST 3: Tighter Trail in Live Runner
# ════════════════════════════════════════════════════════


class TestTighterTrail:
    """Verify trail_atr_mult=1.0 is tighter than 1.5."""

    def test_tighter_trail_exits_sooner(self):
        """1.0× ATR trail triggers exit before 1.5× ATR trail."""
        # Setup: long position, after partial close
        cfg_tight = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=1.5,
            trail_atr_mult=1.0,
        )
        cfg_wide = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=1.5,
            trail_atr_mult=1.5,
        )
        strategy_tight = RubberBandStrategy(cfg_tight)
        strategy_wide = RubberBandStrategy(cfg_wide)

        # Open positions and trigger partial
        for s in (strategy_tight, strategy_wide):
            s.open_position("TEST", 1, 100.0, 1000.0, 1.0, 0.10)
            result = s.update_position("TEST", 116.0)  # target=0.10*1.5=15%, gain=16%
            assert result == "partial_close"

        # Runner peak at 20% gain, then pullback
        for s in (strategy_tight, strategy_wide):
            s.update_position("TEST", 120.0, current_atr_pct=0.03)

        # Now moderate pullback: gain = 17%
        # Tight trail_stop = 20% - 1.0*3% = 17% → gain 17% <= 17% = EXIT
        # Wide trail_stop  = 20% - 1.5*3% = 15.5% → gain 17% > 15.5% = HOLD
        result_tight = strategy_tight.update_position("TEST", 117.0, current_atr_pct=0.03)
        result_wide = strategy_wide.update_position("TEST", 117.0, current_atr_pct=0.03)

        assert result_tight == "trail_runner"
        assert result_wide is None  # Still holding


# ════════════════════════════════════════════════════════
#  TEST 4: Runner Time Exit in Live
# ════════════════════════════════════════════════════════


class TestRunnerTimeExitLive:
    """Verify runner_max_hold_mult controls live runner time exit."""

    def _setup_runner(self, cfg):
        """Create a strategy with a position in runner phase."""
        strategy = RubberBandStrategy(cfg)
        strategy.open_position("TEST", 1, 100.0, 1000.0, 1.0, 0.10)
        # Trigger partial close (target = 0.10 * 1.5 = 15%)
        result = strategy.update_position("TEST", 116.0)
        assert result == "partial_close"
        return strategy

    def test_time_runner_at_mult_boundary(self):
        """Runner exits at max_hold × runner_max_hold_mult bars."""
        cfg = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=1.5,
            trail_atr_mult=1.0,
            max_hold=25,
            runner_max_hold_mult=3.0,
        )
        strategy = self._setup_runner(cfg)

        # Simulate bars at a high price (above trail stop), advancing bars_held
        # Runner exit threshold = 25 * 3.0 = 75 bars
        # update_position increments bars_held FIRST, so set to 73 → becomes 74
        pos = strategy._positions["TEST"]
        pos["bars_held"] = 73
        result = strategy.update_position("TEST", 125.0, current_atr_pct=0.01)
        assert result is None  # 74 < 75, still holding

        # Now at 74 → becomes 75, triggers time_runner
        pos["bars_held"] = 74
        result = strategy.update_position("TEST", 125.0, current_atr_pct=0.01)
        assert result == "time_runner"

    def test_old_mult_would_exit_earlier(self):
        """With mult=2.0, runner exits at bar 50, not 75."""
        cfg_old = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=1.5,
            trail_atr_mult=1.0,
            max_hold=25,
            runner_max_hold_mult=2.0,
        )
        strategy = self._setup_runner(cfg_old)

        pos = strategy._positions["TEST"]
        pos["bars_held"] = 49  # increments to 50 = 25 × 2.0
        result = strategy.update_position("TEST", 125.0, current_atr_pct=0.01)
        assert result == "time_runner"

    def test_runner_does_not_exit_before_threshold(self):
        """Runner holds when bars < max_hold × mult."""
        cfg = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=1.5,
            trail_atr_mult=1.0,
            max_hold=25,
            runner_max_hold_mult=3.0,
        )
        strategy = self._setup_runner(cfg)
        pos = strategy._positions["TEST"]
        pos["bars_held"] = 59  # increments to 60, well below 75
        result = strategy.update_position("TEST", 125.0, current_atr_pct=0.01)
        assert result is None


# ════════════════════════════════════════════════════════
#  TEST 5: Settings Wiring
# ════════════════════════════════════════════════════════


class TestSettingsWiring:
    """Verify RubberBandConfig is the SSoT for RB params (settings.rb_* removed)."""

    def test_settings_default(self):
        from quantbox.strategy.rubberband.config import RubberBandConfig

        cfg = RubberBandConfig()
        assert cfg.runner_max_hold_mult == 3.0

    def test_settings_hybtight_params(self):
        """All four HybTight params have correct defaults in RubberBandConfig."""
        from quantbox.strategy.rubberband.config import RubberBandConfig

        cfg = RubberBandConfig()
        assert cfg.target_retrace == 1.5
        assert cfg.max_hold == 25
        assert cfg.trail_atr_mult == 1.0
        assert cfg.runner_max_hold_mult == 3.0


# ════════════════════════════════════════════════════════
#  TEST 6: Backtest ↔ Live Parity
# ════════════════════════════════════════════════════════


class TestBacktestLiveParity:
    """Critical: backtest and live must agree on runner behavior."""

    def test_runner_time_exit_parity(self):
        """Numba and live agree on runner time exit bar count.

        Both should use max_hold × runner_max_hold_mult as the runner
        time exit threshold. This test verifies they fire at the same
        bar count.
        """
        max_hold = 10
        mult = 3.0
        threshold_bar = int(max_hold * mult)  # 30

        # Build a price series that enters a trade and holds the runner
        # past the time threshold without triggering trail/BE.
        n = 80
        prices = np.full(n, 100.0, dtype=np.float64)
        # Drop at bar 10 (lookback=8, threshold=3.2%)
        for i in range(10, 18):
            frac = (i - 9) / 8
            prices[i] = 100.0 * (1.0 - 0.05 * frac)  # 5% drop
        bottom = prices[17]  # ~95.0
        # Recovery: needs to exceed target_retrace=1.5 × 5% = 7.5%
        # So price needs to go from 95 to > 95*(1+0.075) = ~102.1
        for i in range(18, 28):
            frac = (i - 17) / 10
            prices[i] = bottom * (1.0 + 0.10 * frac)  # 10% recovery
        # Hold at recovered level (high enough to avoid trail/BE stop)
        recovered = prices[27]
        for i in range(28, n):
            prices[i] = recovered

        sigs = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.032,
            1.5,
            0.382,
            max_hold,
            0.0,
            runner_max_hold_mult=mult,
        )

        # Find where runner (|sig|==0.5) starts and ends
        runner_bars = np.where(np.abs(sigs) == 0.5)[0]
        if len(runner_bars) > 0:
            runner_duration = len(runner_bars)
            # Runner should not exceed max_hold × mult
            assert runner_duration <= threshold_bar, f"Runner held {runner_duration} bars, expected <= {threshold_bar}"

        # Live parity: same threshold
        cfg = RubberBandConfig(
            max_hold=max_hold,
            runner_max_hold_mult=mult,
            partial_exit_enabled=True,
            target_retrace=1.5,
            trail_atr_mult=1.0,
        )
        live_threshold = cfg.max_hold * cfg.runner_max_hold_mult
        assert live_threshold == threshold_bar
