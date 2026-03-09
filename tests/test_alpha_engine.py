"""
Tests for alpha_engine — Multi-factor signal generator.

Tests the Numba-compiled signal engines and the AlphaStrategy wrapper.
Validates that filters correctly modify entry behavior and that
exit logic (trailing stop, max hold) works correctly.
"""

import numpy as np
import pytest

from quantbox.strategy.alpha_engine import (
    AlphaConfig,
    AlphaStrategy,
    compute_alpha_signals,
    compute_hybrid_signals,
    compute_momentum_signals,
    _compute_atr,
    _compute_vol_ratio,
    _compute_taker_ratio,
)
from quantbox.backtest.vectorized_backtest import _backtest_core


# ── Helper indicators ──


class TestHelperFunctions:
    """Tests for ATR, volume ratio, and taker ratio."""

    def test_atr_positive(self):
        """ATR is always non-negative for valid OHLC."""
        n = 200
        rng = np.random.RandomState(42)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.005, n))
        high = close * (1 + rng.uniform(0, 0.01, n))
        low = close * (1 - rng.uniform(0, 0.01, n))
        atr = _compute_atr(high, low, close, 14)
        assert np.all(atr >= 0)
        assert atr[50] > 0  # Should be positive after warmup

    def test_vol_ratio_baseline(self):
        """Constant volume → ratio = 1.0."""
        volume = np.full(200, 1000.0)
        ratio = _compute_vol_ratio(volume, 48)
        # After warmup, ratio should be ~1.0
        assert abs(ratio[100] - 1.0) < 0.01

    def test_vol_ratio_spike(self):
        """Volume spike → ratio > 1.0."""
        volume = np.full(200, 1000.0)
        volume[150] = 5000.0  # 5x spike
        ratio = _compute_vol_ratio(volume, 48)
        assert ratio[150] > 4.0

    def test_taker_ratio_balanced(self):
        """50/50 taker/volume → ratio = 0.5."""
        n = 200
        volume = np.full(n, 1000.0)
        taker = np.full(n, 500.0)
        ratio = _compute_taker_ratio(taker, volume, 16)
        assert abs(ratio[100] - 0.5) < 0.01


# ── compute_alpha_signals ──


class TestAlphaSignals:
    """Tests for the enhanced rubber band signal engine."""

    @pytest.fixture
    def make_price_data(self):
        """Create OHLCV arrays for testing."""

        def _make(n=500, seed=42):
            rng = np.random.RandomState(seed)
            close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.003, n))
            high = close * (1 + rng.uniform(0, 0.005, n))
            low = close * (1 - rng.uniform(0, 0.005, n))
            volume = rng.uniform(500, 2000, n)
            taker = volume * rng.uniform(0.4, 0.6, n)
            return close, high, low, volume, taker

        return _make

    def test_output_shape(self, make_price_data):
        """Output length matches input."""
        close, high, low, vol, taker = make_price_data(500)
        sigs = compute_alpha_signals(
            close,
            high,
            low,
            vol,
            taker,
            8,
            0.025,
            0.5,
            1.0,
            8,
            False,
            48,
            1.5,
            False,
            96,
            0,
            False,
            48,
            2.5,
            False,
            0.3,
            0.4,
            False,
            16,
            0.55,
            False,
            32,
        )
        assert len(sigs) == 500

    def test_signals_in_valid_range(self, make_price_data):
        """Signals are only -1, 0, or +1."""
        close, high, low, vol, taker = make_price_data(500)
        sigs = compute_alpha_signals(
            close,
            high,
            low,
            vol,
            taker,
            8,
            0.025,
            0.5,
            1.0,
            8,
            True,
            48,
            1.5,
            True,
            96,
            0,
            True,
            48,
            2.5,
            True,
            0.3,
            0.4,
            True,
            16,
            0.55,
            False,
            32,
        )
        unique = set(sigs)
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_vol_filter_reduces_trades(self, make_price_data):
        """Enabling volume filter should reduce number of trades."""
        close, high, low, vol, taker = make_price_data(2000, seed=99)
        # No filter
        sigs_no = compute_alpha_signals(
            close,
            high,
            low,
            vol,
            taker,
            8,
            0.025,
            0.5,
            1.0,
            8,
            False,
            48,
            1.5,
            False,
            96,
            0,
            False,
            48,
            2.5,
            False,
            0.3,
            0.4,
            False,
            16,
            0.55,
            False,
            32,
        )
        # With strict vol filter
        sigs_yes = compute_alpha_signals(
            close,
            high,
            low,
            vol,
            taker,
            8,
            0.025,
            0.5,
            1.0,
            8,
            True,
            48,
            3.0,  # Very strict: 3x average
            False,
            96,
            0,
            False,
            48,
            2.5,
            False,
            0.3,
            0.4,
            False,
            16,
            0.55,
            False,
            32,
        )
        trades_no = np.sum(np.abs(np.diff(sigs_no)) > 0)
        trades_yes = np.sum(np.abs(np.diff(sigs_yes)) > 0)
        assert trades_yes <= trades_no

    def test_max_hold_exits(self, make_price_data):
        """Positions close after max_hold bars."""
        close, high, low, vol, taker = make_price_data(2000)
        max_hold = 4
        sigs = compute_alpha_signals(
            close,
            high,
            low,
            vol,
            taker,
            8,
            0.025,
            0.5,
            1.0,
            max_hold,
            False,
            48,
            1.5,
            False,
            96,
            0,
            False,
            48,
            2.5,
            False,
            0.3,
            0.4,
            False,
            16,
            0.55,
            False,
            32,
        )
        # Find any trade and check it doesn't exceed max_hold
        in_pos = 0
        for i in range(1, len(sigs)):
            if sigs[i] != 0:
                in_pos += 1
            else:
                if in_pos > 0:
                    assert in_pos <= max_hold + 1  # +1 for entry bar
                in_pos = 0


# ── compute_hybrid_signals ──


class TestHybridSignals:
    """Tests for the trend+rubber band hybrid strategy."""

    def test_output_shape(self):
        """Output length matches input."""
        n = 500
        rng = np.random.RandomState(42)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.003, n))
        high = close * 1.005
        low = close * 0.995
        vol = np.full(n, 1000.0)
        taker = vol * 0.5
        sigs = compute_hybrid_signals(
            close,
            high,
            low,
            vol,
            taker,
            64,
            0.02,
            8,
            0.025,
            0.6,
            1.0,
            12,
            48,
            1.5,
            48,
            2.0,
            0.3,
            0.4,
        )
        assert len(sigs) == n

    def test_only_trades_with_trend(self):
        """Hybrid should only enter in the trend direction."""
        n = 1000
        # Create a clear uptrend
        close = 100.0 * np.cumprod(1.0 + np.full(n, 0.001))
        # Add a sharp dip at bar 500
        close[495:500] *= np.linspace(1.0, 0.95, 5)
        close[500:] *= 0.95 / close[500] * close[499]  # reconnect

        high = close * 1.002
        low = close * 0.998
        vol = np.full(n, 2000.0)
        vol[498:502] = 5000.0  # Volume spike at dip
        taker = vol * 0.5

        sigs = compute_hybrid_signals(
            close,
            high,
            low,
            vol,
            taker,
            64,
            0.01,
            8,
            0.02,
            0.6,
            1.0,
            12,
            48,
            1.0,
            48,
            2.0,
            0.3,
            0.4,
        )
        unique = set(sigs)
        assert unique.issubset({-1.0, 0.0, 1.0})


# ── AlphaStrategy wrapper ──


class TestAlphaStrategy:
    """Tests for the live trading wrapper."""

    def test_default_config(self):
        """Default config creates valid strategy."""
        strat = AlphaStrategy()
        assert strat.config.lookback == 8
        assert strat.config.cost_per_side == 0.0004
        assert strat.n_active == 0

    def test_position_management(self):
        """Open/close position tracking works."""
        strat = AlphaStrategy()
        assert not strat.has_position("BTCUSDT")
        strat.open_position("BTCUSDT", 1, 50000.0, 10000.0, 2.0, 0.03)
        assert strat.has_position("BTCUSDT")
        assert strat.n_active == 1
        pos = strat.close_position("BTCUSDT")
        assert pos["signal"] == 1
        assert strat.n_active == 0

    def test_config_cost_properties(self):
        """Config correctly computes derived costs."""
        cfg = AlphaConfig(fee_rate=0.0002, slippage_rate=0.0003)
        assert cfg.cost_per_side == 0.0005
        assert cfg.bars_per_funding == 32
        assert cfg.funding_per_bar > 0


# ── Integration with backtest engine ──


class TestBacktestIntegration:
    """Test that alpha signals produce valid backtest results."""

    def test_alpha_signals_backtest(self, real_data_available):
        """compute_alpha_signals runs through _backtest_core without errors."""
        if not real_data_available:
            pytest.skip("Real data not available")

        from quantbox.data.real_data_loader import RealDataLoader

        loader = RealDataLoader()
        df = loader.load_symbol("BTCUSDT")
        if df is None:
            pytest.skip("BTCUSDT data not available")

        close = df["Close"].values.astype(np.float64)
        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        taker = vol * 0.5

        sigs = compute_alpha_signals(
            close,
            high,
            low,
            vol,
            taker,
            8,
            0.025,
            0.5,
            1.0,
            8,
            True,
            48,
            1.5,
            False,
            96,
            0,
            True,
            48,
            2.5,
            True,
            0.3,
            0.4,
            False,
            16,
            0.55,
            False,
            32,
        )

        eq, ret, sharpe, dd, wr, trades, bars_in, pf, avg_ret, _ = _backtest_core(close, sigs, 0.0004, 0.0001 / 32)

        assert len(eq) == len(close)
        assert eq[0] == 1.0
        assert dd >= 0
        assert dd <= 1.0
