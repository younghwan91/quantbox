"""Tests for dynamic leverage (move-confidence scaling).

Validates:
  - compute_confidence_leverage formula correctness
  - apply_confidence_leverage Numba function (backtest signal scaling)
  - Edge cases: threshold-level moves, extreme moves, zero/negative inputs
  - Integration with RubberBandConfig defaults
  - Backtest-live parity: same leverage for same move
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_confidence_leverage,
    apply_confidence_leverage,
)


class TestComputeConfidenceLeverage:
    """Unit tests for the pure-Python leverage formula."""

    def test_threshold_level_move_returns_base(self):
        """Move exactly at threshold → base leverage (no excess)."""
        lev = compute_confidence_leverage(
            move_size=0.045,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert lev == pytest.approx(3.0)

    def test_below_threshold_returns_base(self):
        """Move below threshold → still base leverage (excess clamped to 0)."""
        lev = compute_confidence_leverage(
            move_size=0.03,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert lev == pytest.approx(3.0)

    def test_double_threshold_move(self):
        """Move = 2× threshold → excess = 1.0 → lev = base + scale*1.0."""
        # excess = (0.09 - 0.045) / 0.045 = 1.0
        lev = compute_confidence_leverage(
            move_size=0.09,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=20.0,
        )
        assert lev == pytest.approx(13.0)  # 3 + 10*1.0

    def test_max_leverage_cap(self):
        """Extreme move → leverage capped at max."""
        lev = compute_confidence_leverage(
            move_size=0.20,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert lev == pytest.approx(7.0)

    def test_zero_move_returns_base(self):
        """Zero move size → base leverage (safety)."""
        lev = compute_confidence_leverage(
            move_size=0.0,
            threshold=0.045,
            base_leverage=5.0,
            leverage_scale=20.0,
            max_leverage=15.0,
        )
        assert lev == pytest.approx(5.0)

    def test_negative_move_returns_base(self):
        """Negative move size → base leverage (safety)."""
        lev = compute_confidence_leverage(
            move_size=-0.05,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert lev == pytest.approx(3.0)

    def test_zero_threshold_returns_base(self):
        """Zero threshold → base leverage (avoid division by zero)."""
        lev = compute_confidence_leverage(
            move_size=0.05,
            threshold=0.0,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert lev == pytest.approx(3.0)

    def test_competition_mode_params(self):
        """A/B validated competition mode: Base=5, Scale=20, Max=15."""
        # 6% move, threshold 4.5%: excess = (0.06-0.045)/0.045 = 0.333
        lev = compute_confidence_leverage(
            move_size=0.06,
            threshold=0.045,
            base_leverage=5.0,
            leverage_scale=20.0,
            max_leverage=15.0,
        )
        expected = 5.0 + 20.0 * (0.06 - 0.045) / 0.045  # 5 + 6.667 = 11.667
        assert lev == pytest.approx(expected)

    def test_conservative_defaults(self):
        """Default config produces v4.4 flat leverage: base=1.5, scale=0, max=1.5."""
        cfg = RubberBandConfig()
        assert cfg.dynamic_leverage_enabled is True
        assert cfg.base_leverage == 1.5
        assert cfg.leverage_scale == 0.0
        assert cfg.max_leverage == 20.0  # v12.5.2: final hard cap (after MTF mult)


class TestApplyConfidenceLeverage:
    """Tests for the Numba-compiled backtest leverage scaler."""

    def test_flat_signals_unchanged(self):
        """Zero signals remain zero (no leverage on flat)."""
        close = np.linspace(100, 110, 100)
        signals = np.zeros(100)
        result = apply_confidence_leverage(
            close,
            signals,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_long_signal_scaled_positive(self):
        """Long signal (+1) → positive leverage value."""
        close = np.ones(20) * 100.0
        signals = np.zeros(20)
        signals[15] = 1.0  # Long entry at bar 15
        result = apply_confidence_leverage(
            close,
            signals,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        # move_size = |100/100 - 1| = 0 → excess = negative → base_leverage
        assert result[15] == pytest.approx(3.0)

    def test_short_signal_scaled_negative(self):
        """Short signal (-1) → negative leverage value."""
        close = np.ones(20) * 100.0
        signals = np.zeros(20)
        signals[15] = -1.0
        result = apply_confidence_leverage(
            close,
            signals,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert result[15] == pytest.approx(-3.0)

    def test_large_move_gets_more_leverage(self):
        """A 10% move should get more leverage than a 5% move."""
        n = 50
        close = np.ones(n) * 100.0

        # Simulate a 5% move at bar 20
        signals_5pct = np.zeros(n)
        signals_5pct[20] = 1.0
        close_5pct = close.copy()
        close_5pct[20] = 100.0
        close_5pct[12] = 95.24  # 20-8=12, move = 100/95.24 - 1 ≈ 5%

        lev_5 = apply_confidence_leverage(
            close_5pct,
            signals_5pct,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=20.0,
        )

        # Simulate a 10% move at bar 20
        signals_10pct = np.zeros(n)
        signals_10pct[20] = 1.0
        close_10pct = close.copy()
        close_10pct[20] = 100.0
        close_10pct[12] = 90.91  # move = 100/90.91 - 1 ≈ 10%

        lev_10 = apply_confidence_leverage(
            close_10pct,
            signals_10pct,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=20.0,
        )

        assert lev_10[20] > lev_5[20], "Bigger move should get bigger leverage"

    def test_leverage_never_exceeds_max(self):
        """Even with extreme moves, leverage stays under max."""
        n = 30
        close = np.ones(n) * 100.0
        close[5] = 50.0  # 100% move
        signals = np.zeros(n)
        signals[13] = 1.0  # 13-8=5

        result = apply_confidence_leverage(
            close,
            signals,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert abs(result[13]) <= 7.0

    def test_early_bars_pass_through(self):
        """Bars before lookback pass through unchanged."""
        close = np.ones(20) * 100.0
        signals = np.zeros(20)
        signals[3] = 1.0  # Before lookback=8

        result = apply_confidence_leverage(
            close,
            signals,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )
        assert result[3] == 1.0  # Passed through as-is


class TestBacktestLiveParity:
    """Ensure backtest and live compute the same leverage for the same move."""

    def test_same_leverage_same_move(self):
        """compute_confidence_leverage and apply_confidence_leverage agree."""
        # Create a scenario with a known move
        n = 30
        close = np.ones(n) * 100.0
        close[7] = 93.0  # 7.53% move at bar 15 (15-8=7)
        signals = np.zeros(n)
        signals[15] = 1.0

        move_size = abs(close[15] / close[7] - 1.0)  # 7.53%

        # Python function (live)
        live_lev = compute_confidence_leverage(
            move_size=move_size,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )

        # Numba function (backtest)
        bt_signals = apply_confidence_leverage(
            close,
            signals,
            lookback=8,
            threshold=0.045,
            base_leverage=3.0,
            leverage_scale=10.0,
            max_leverage=7.0,
        )

        assert bt_signals[15] == pytest.approx(live_lev, abs=1e-10), (
            "Backtest and live leverage must match for the same move"
        )
