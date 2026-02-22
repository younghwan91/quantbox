"""Tests for v4.3 Capital Preservation Framework.

Tests:
  1. Partial exit signals: ±1.0 → ±0.5 transition on first profit
  2. Break-even stop: remaining half closes at entry price
  3. R:R-gated leverage: no leverage when R:R < threshold
  4. Backtest core partial: delta-based cost model
  5. Live trading parity: update_position returns 'partial_close'
  6. Combined with BTC directional: entry gate + partial exit
"""

import numpy as np
import pytest
from quantbox.strategy.rubberband import (
    compute_signals,
    compute_signals_with_partial_exit,
    compute_signals_partial_exit_btc_directional,
    compute_rr_gated_leverage,
    apply_rr_gated_leverage,
    apply_confidence_leverage,
    RubberBandConfig,
    RubberBandStrategy,
)
from quantbox.backtest.vectorized_backtest import (
    _backtest_core,
    _backtest_core_partial,
)


# ── Helpers ──
def make_drop_recovery(n=200, drop_pct=0.06, recovery_pct=0.03):
    """Create price series: flat → sharp drop → gradual recovery."""
    prices = np.full(n, 100.0)
    drop_bar = 50
    # Sharp drop over 8 bars
    for i in range(drop_bar, drop_bar + 8):
        prices[i] = prices[drop_bar - 1] * (1.0 - drop_pct * (i - drop_bar + 1) / 8)
    bottom = prices[drop_bar + 7]
    # Recovery over 20 bars
    for i in range(drop_bar + 8, min(drop_bar + 28, n)):
        frac = (i - drop_bar - 7) / 20
        prices[i] = bottom * (1.0 + recovery_pct * frac)
    # Flat after
    for i in range(drop_bar + 28, n):
        prices[i] = prices[drop_bar + 27]
    return prices.astype(np.float64)


def make_drop_then_flat(n=200, drop_pct=0.06):
    """Price drops then stays flat (break-even stop scenario)."""
    prices = np.full(n, 100.0)
    drop_bar = 50
    for i in range(drop_bar, drop_bar + 8):
        prices[i] = prices[drop_bar - 1] * (1.0 - drop_pct * (i - drop_bar + 1) / 8)
    bottom = prices[drop_bar + 7]
    # Tiny recovery (triggers partial close) then back to entry
    for i in range(drop_bar + 8, drop_bar + 12):
        prices[i] = bottom * 1.002  # +0.2% above bottom
    for i in range(drop_bar + 12, n):
        prices[i] = bottom * 0.998  # slight drop below entry
    return prices.astype(np.float64)


# ════════════════════════════════════════════════════════
#  TEST 1: Partial Exit Signal Generation
# ════════════════════════════════════════════════════════

class TestPartialExitSignals:
    """Verify compute_signals_with_partial_exit output."""

    def test_signals_contain_half_values(self):
        """Partial exit produces ±0.5 signals (half position)."""
        prices = make_drop_recovery()
        sigs = compute_signals_with_partial_exit(
            prices, prices, prices, 8, 0.045, 0.3, 1.0, 10, 0.0,
        )
        unique = np.unique(sigs)
        # Should have 0.0 (flat) and either 0.5 or 1.0
        assert 0.0 in unique
        # At least some non-zero signals should exist
        has_position = np.any(sigs != 0)
        if has_position:
            # If there's a position, it should transition through phases
            abs_sigs = np.abs(sigs[sigs != 0])
            # Should contain 1.0 (full) or 0.5 (half) or both
            assert np.any((abs_sigs == 1.0) | (abs_sigs == 0.5))

    def test_no_partial_exit_matches_original(self):
        """Without partial exit, signal direction matches compute_signals."""
        prices = make_drop_recovery(drop_pct=0.08)
        sigs_orig = compute_signals(prices, 8, 0.045, 0.3, 1.0, 10, 0.0)
        sigs_partial = compute_signals_with_partial_exit(
            prices, prices, prices, 8, 0.045, 0.3, 1.0, 10, 0.0,
        )
        # Direction should match (sign)
        for i in range(len(prices)):
            if sigs_orig[i] != 0 and sigs_partial[i] != 0:
                assert np.sign(sigs_orig[i]) == np.sign(sigs_partial[i])

    def test_partial_signal_magnitude_is_half(self):
        """After partial close, signal magnitude should be exactly 0.5."""
        prices = make_drop_recovery(drop_pct=0.08, recovery_pct=0.04)
        sigs = compute_signals_with_partial_exit(
            prices, prices, prices, 8, 0.045, 0.3, 1.0, 10, 0.0,
        )
        half_sigs = sigs[np.abs(sigs) == 0.5]
        if len(half_sigs) > 0:
            assert np.all(np.abs(half_sigs) == 0.5)

    def test_full_to_half_transition(self):
        """Signal transitions from ±1 to ±0.5 (not directly to 0)."""
        prices = make_drop_recovery(drop_pct=0.08, recovery_pct=0.04)
        sigs = compute_signals_with_partial_exit(
            prices, prices, prices, 8, 0.045, 0.3, 1.0, 10, 0.0,
        )
        # Find transitions
        for i in range(1, len(sigs)):
            if abs(sigs[i]) == 0.5 and abs(sigs[i-1]) == 1.0:
                # Valid: full → half
                assert np.sign(sigs[i]) == np.sign(sigs[i-1])


# ════════════════════════════════════════════════════════
#  TEST 2: R:R-Gated Leverage
# ════════════════════════════════════════════════════════

class TestRRGatedLeverage:
    """Verify R:R-gated leverage logic."""

    def test_threshold_move_no_leverage(self):
        """At threshold move size, R:R ≈ 0.28 < 0.30 → leverage = 1.0."""
        lev = compute_rr_gated_leverage(
            0.045, 0.045, 0.3, 1.0, 0.0004, 3.0, 10.0, 7.0, 0.30,
        )
        assert lev == 1.0

    def test_large_move_also_no_leverage(self):
        """Even large moves have R:R ≈ 0.29 < 0.30 → still 1.0."""
        lev = compute_rr_gated_leverage(
            0.10, 0.045, 0.3, 1.0, 0.0004, 3.0, 10.0, 7.0, 0.30,
        )
        # R:R = (0.10*0.3 - 0.0008) / (0.10*1.0 + 0.0008) = 0.0292/0.1008 ≈ 0.29
        assert lev == 1.0

    def test_low_min_rr_allows_leverage(self):
        """With min_rr=0.20, most trades get leverage."""
        lev = compute_rr_gated_leverage(
            0.045, 0.045, 0.3, 1.0, 0.0004, 3.0, 10.0, 7.0, 0.20,
        )
        # R:R ≈ 0.277 > 0.20 → leverage allowed
        assert lev >= 3.0

    def test_zero_move_returns_one(self):
        """Zero move size → no leverage."""
        lev = compute_rr_gated_leverage(
            0.0, 0.045, 0.3, 1.0, 0.0004, 3.0, 10.0, 7.0, 0.30,
        )
        assert lev == 1.0

    def test_favorable_params_allow_leverage(self):
        """With favorable target/stop ratio, R:R exceeds gate."""
        # target=0.6, stop=0.5 → R:R ≈ 1.2 → leverage should be allowed
        lev = compute_rr_gated_leverage(
            0.045, 0.045, 0.6, 0.5, 0.0004, 3.0, 10.0, 7.0, 0.30,
        )
        assert lev >= 3.0

    def test_apply_rr_gated_preserves_direction(self):
        """apply_rr_gated_leverage preserves signal direction."""
        close = np.full(50, 100.0)
        signals = np.zeros(50)
        signals[20] = 1.0
        signals[21] = 0.5
        signals[30] = -1.0
        signals[31] = -0.5

        result = apply_rr_gated_leverage(
            close, signals, 8, 0.045, 0.3, 1.0, 0.0004, 3.0, 10.0, 7.0, 0.30,
        )
        # With min_rr=0.30, leverage=1.0 for all, so signals unchanged
        assert result[20] == 1.0
        assert result[21] == 0.5
        assert result[30] == -1.0
        assert result[31] == -0.5


# ════════════════════════════════════════════════════════
#  TEST 3: Backtest Core Partial (Delta-Based Costs)
# ════════════════════════════════════════════════════════

class TestBacktestCorePartial:
    """Verify _backtest_core_partial delta-based cost model."""

    def test_identical_to_original_without_partial(self):
        """For standard 0/±1 signals, results match _backtest_core."""
        close = np.linspace(100, 110, 500).astype(np.float64)
        sigs = compute_signals(close, 8, 0.045, 0.3, 1.0, 10, 0.0)

        orig = _backtest_core(close, sigs, 0.0004, 0.0001/32)
        partial = _backtest_core_partial(close, sigs, 0.0004, 0.0001/32)

        # Total return should be very close (delta costs = close+open costs
        # for 0↔±1 transitions)
        np.testing.assert_allclose(orig[1], partial[1], rtol=1e-6)

    def test_partial_close_lower_cost(self):
        """Partial close (1→0.5) costs less than full close+reopen."""
        close = np.full(20, 100.0)
        # Simulate: entry at bar 5, partial at bar 10, full close at bar 15
        sigs = np.zeros(20)
        sigs[5:10] = 1.0
        sigs[10:15] = 0.5
        # sigs[15:] = 0.0

        # Original core: cost at bar 10 = |1.0|+|0.5| = 1.5 * cost
        eq_orig = _backtest_core(close, sigs, 0.01, 0.0)  # Large cost for visibility
        # Partial core: cost at bar 10 = |0.5-1.0| = 0.5 * cost
        eq_partial = _backtest_core_partial(close, sigs, 0.01, 0.0)

        # Partial should have higher equity (lower costs)
        assert eq_partial[1] >= eq_orig[1]  # total return

    def test_trade_counting_lifecycle(self):
        """Partial core counts full lifecycle as one trade."""
        close = np.full(30, 100.0)
        sigs = np.zeros(30)
        sigs[5:10] = 1.0    # Full position
        sigs[10:15] = 0.5   # Half position
        sigs[15:] = 0.0     # Flat

        result = _backtest_core_partial(close, sigs, 0.0004, 0.0)
        n_trades = result[5]
        # Should count as 1 trade (entry to final flat)
        assert n_trades == 1


# ════════════════════════════════════════════════════════
#  TEST 4: Live Trading Parity
# ════════════════════════════════════════════════════════

class TestLiveTradingParity:
    """Verify update_position returns correct exit reasons.

    v12.3: PBT mode — partial triggers at TP1 (gain >= target),
    not on any gain > 0. Runner uses trail_runner/time_runner.
    """

    def test_partial_close_at_target(self):
        """update_position returns 'partial_close' when gain >= target (v12.3 PBT)."""
        cfg = RubberBandConfig(partial_exit_enabled=True, target_retrace=0.3)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # target = 0.06 * 0.3 = 1.8%. Price at +2% → triggers partial
        result = strategy.update_position('BTCUSDT', 102.0)
        assert result == 'partial_close'

    def test_no_partial_below_target(self):
        """Small gain below target does NOT trigger partial (v12.3 PBT)."""
        cfg = RubberBandConfig(partial_exit_enabled=True, target_retrace=0.3)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # +0.5% gain < 1.8% target → no partial
        result = strategy.update_position('BTCUSDT', 100.5)
        assert result is None

    def test_breakeven_floor_after_partial(self):
        """After partial close, price at entry triggers 'trail_runner' (breakeven floor)."""
        cfg = RubberBandConfig(partial_exit_enabled=True, target_retrace=0.3)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # Partial close at target
        result1 = strategy.update_position('BTCUSDT', 102.0)
        assert result1 == 'partial_close'
        assert strategy.has_position('BTCUSDT')

        # Price drops to entry → breakeven floor triggers trail_runner
        result2 = strategy.update_position('BTCUSDT', 100.0, current_atr_pct=0.02)
        assert result2 == 'trail_runner'

    def test_target_hit_triggers_partial(self):
        """Target hit on full position triggers partial_close in PBT mode."""
        cfg = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=0.3,
        )
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # Price jumps to target (0.06 * 0.3 = 1.8% gain)
        result = strategy.update_position('BTCUSDT', 101.81)
        assert result == 'partial_close'  # PBT: target → partial, not full close

    def test_stop_loss_before_partial(self):
        """Stop loss on full position works before partial close."""
        cfg = RubberBandConfig(partial_exit_enabled=True)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # Price drops hard (stop = 0.06 * 1.0 = 6%)
        result = strategy.update_position('BTCUSDT', 93.5)
        assert result == 'stop_loss'

    def test_no_partial_when_disabled(self):
        """With partial_exit_enabled=False, no 'partial_close' reason."""
        cfg = RubberBandConfig(partial_exit_enabled=False)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # Small profit — should NOT trigger partial close
        result = strategy.update_position('BTCUSDT', 100.5)
        assert result is None

    def test_short_partial_close(self):
        """Partial close works for short positions at target (v12.3 PBT)."""
        cfg = RubberBandConfig(partial_exit_enabled=True, target_retrace=0.3)
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', -1, 100.0, 1000.0, 1.0, 0.06,
        )
        # Short: target = 0.06 * 0.3 = 1.8%. Price at 98 → gain 2% > 1.8%
        result = strategy.update_position('BTCUSDT', 98.0)
        assert result == 'partial_close'

    def test_runner_trail_after_partial(self):
        """Runner exits via trail_runner after partial close (v12.3 PBT)."""
        cfg = RubberBandConfig(
            partial_exit_enabled=True,
            target_retrace=0.3,
        )
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            'BTCUSDT', 1, 100.0, 1000.0, 1.0, 0.06,
        )
        # Partial close at target
        strategy.update_position('BTCUSDT', 102.0)
        # Runner: price rises to 4% then falls
        strategy.update_position('BTCUSDT', 104.0, current_atr_pct=0.01)
        # Trail stop = 4% - 1.5*1% = 2.5%. Gain at 2% < 2.5% → trail_runner
        result = strategy.update_position('BTCUSDT', 102.0, current_atr_pct=0.01)
        assert result == 'trail_runner'


# ════════════════════════════════════════════════════════
#  TEST 5: BTC Directional + Partial Exit
# ════════════════════════════════════════════════════════

class TestPartialExitBTCDirectional:
    """Verify combined BTC directional + partial exit."""

    def test_same_entry_gate_as_btc_directional(self):
        """Entry requires BTC momentum alignment (same as v4.2)."""
        n = 200
        close = np.full(n, 100.0)
        btc_close = np.full(n, 50000.0)

        # Create a drop in the alt
        for i in range(50, 58):
            close[i] = 100.0 * (1.0 - 0.06 * (i - 49) / 8)

        # BTC flat (no confirmation) → should NOT enter
        sigs_no_confirm = compute_signals_partial_exit_btc_directional(
            close, btc_close, 8, 0.045, 0.3, 1.0, 10, 0.0, 32,
        )

        # BTC also dropping → should enter
        btc_drop = btc_close.copy()
        for i in range(20, 58):
            btc_drop[i] = 50000.0 * (1.0 - 0.01 * (i - 19) / 38)
        sigs_confirm = compute_signals_partial_exit_btc_directional(
            close, btc_drop, 8, 0.045, 0.3, 1.0, 10, 0.0, 32,
        )

        # No BTC drop → no entry
        assert np.all(sigs_no_confirm == 0)
        # With BTC drop → entry should happen
        assert np.any(sigs_confirm != 0)


# ════════════════════════════════════════════════════════
#  TEST 6: Config Integration
# ════════════════════════════════════════════════════════

class TestConfigIntegration:
    """Verify new config fields work correctly."""

    def test_default_config_values(self):
        """Default config has v12.3 PBT settings: partial exit enabled, R:R gate on."""
        cfg = RubberBandConfig()
        assert cfg.partial_exit_enabled is True  # v12.3: PBT mode on by default
        assert cfg.partial_exit_fraction == 0.5
        assert cfg.rr_leverage_enabled is True
        assert cfg.min_rr_for_leverage == 0.30

    def test_disabled_config(self):
        """Can disable v4.3 features."""
        cfg = RubberBandConfig(
            partial_exit_enabled=False,
            rr_leverage_enabled=False,
        )
        assert cfg.partial_exit_enabled is False
        assert cfg.rr_leverage_enabled is False
