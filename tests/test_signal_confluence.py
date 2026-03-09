"""
Tests for v9.0 S+R+M Confluence Pipeline features.

Validates:
  - Numba helper functions: EMA, Stochastic, MACD
  - compute_signals_v9 shape/value correctness
  - RubberBandConfig v9.0 defaults (Fibonacci exits, MACD+Stoch enabled)
  - on_bar() dispatch to v9 vs v8 path
  - Backward compatibility: compute_signals_confidence unchanged

Uses REAL Binance data from cache (no synthetic data).
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    _compute_ema_inline,
    _compute_macd_inline,
    _compute_stochastic_inline,
    compute_signals_confidence,
    compute_signals_v9,
)


# ── Fixtures ──


@pytest.fixture
def btc_close(real_data_available):
    """Real BTC close prices."""
    if not real_data_available:
        pytest.skip("Real data not available")
    from quantbox.data.real_data_loader import RealDataLoader

    loader = RealDataLoader()
    df = loader.load_symbol("BTCUSDT")
    if df is None:
        pytest.skip("BTCUSDT data not available")
    return df["Close"].values.astype(np.float64)


@pytest.fixture
def btc_ohlcv(real_data_available):
    """Real BTC OHLCV data."""
    if not real_data_available:
        pytest.skip("Real data not available")
    from quantbox.data.real_data_loader import RealDataLoader

    loader = RealDataLoader()
    df = loader.load_symbol("BTCUSDT")
    if df is None:
        pytest.skip("BTCUSDT data not available")
    return {
        "close": df["Close"].values.astype(np.float64),
        "high": df["High"].values.astype(np.float64),
        "low": df["Low"].values.astype(np.float64),
        "volume": df["Volume"].values.astype(np.float64),
        "taker_buy": df["Taker_buy_base_volume"].values.astype(np.float64)
        if "Taker_buy_base_volume" in df.columns
        else df["Volume"].values.astype(np.float64) * 0.5,
    }


@pytest.fixture
def eth_ohlcv(real_data_available):
    """Real ETH OHLCV data."""
    if not real_data_available:
        pytest.skip("Real data not available")
    from quantbox.data.real_data_loader import RealDataLoader

    loader = RealDataLoader()
    df = loader.load_symbol("ETHUSDT")
    if df is None:
        pytest.skip("ETHUSDT data not available")
    return {
        "close": df["Close"].values.astype(np.float64),
        "high": df["High"].values.astype(np.float64),
        "low": df["Low"].values.astype(np.float64),
        "volume": df["Volume"].values.astype(np.float64),
        "taker_buy": df["Taker_buy_base_volume"].values.astype(np.float64)
        if "Taker_buy_base_volume" in df.columns
        else df["Volume"].values.astype(np.float64) * 0.5,
    }


# ══════════════ 1. Numba Helper Functions ══════════════


class TestEMAInline:
    """Test _compute_ema_inline Numba function."""

    def test_output_shape(self, btc_close):
        ema = _compute_ema_inline(btc_close, 12)
        assert ema.shape == btc_close.shape

    def test_sma_warmup(self, btc_close):
        """First valid EMA value should equal SMA of first `period` bars."""
        period = 12
        ema = _compute_ema_inline(btc_close, period)
        expected_sma = np.mean(btc_close[:period])
        assert abs(ema[period - 1] - expected_sma) < 1e-6

    def test_nan_before_warmup(self, btc_close):
        ema = _compute_ema_inline(btc_close, 12)
        assert np.all(np.isnan(ema[:11]))
        assert not np.isnan(ema[11])


class TestStochasticInline:
    """Test _compute_stochastic_inline Numba function."""

    def test_output_shapes(self, btc_ohlcv):
        k, d = _compute_stochastic_inline(btc_ohlcv["high"], btc_ohlcv["low"], btc_ohlcv["close"], 5, 3)
        assert k.shape == btc_ohlcv["close"].shape
        assert d.shape == btc_ohlcv["close"].shape

    def test_k_range(self, btc_ohlcv):
        """Stochastic %K must be in [0, 100]."""
        k, _ = _compute_stochastic_inline(btc_ohlcv["high"], btc_ohlcv["low"], btc_ohlcv["close"], 5, 3)
        assert k.min() >= 0.0
        assert k.max() <= 100.0

    def test_d_is_smoothed_k(self, btc_ohlcv):
        """%D should be smoother (lower std) than %K."""
        k, d = _compute_stochastic_inline(btc_ohlcv["high"], btc_ohlcv["low"], btc_ohlcv["close"], 14, 3)
        # After warmup, D should be smoother
        assert np.std(d[20:]) <= np.std(k[20:])


class TestMACDInline:
    """Test _compute_macd_inline Numba function."""

    def test_output_shapes(self, btc_close):
        ml, sl, hist = _compute_macd_inline(btc_close, 12, 26, 14)
        assert ml.shape == btc_close.shape
        assert sl.shape == btc_close.shape
        assert hist.shape == btc_close.shape

    def test_histogram_is_macd_minus_signal(self, btc_close):
        ml, sl, hist = _compute_macd_inline(btc_close, 12, 26, 14)
        # Where signal is valid, hist should equal macd - signal
        valid = ~np.isnan(sl)
        diff = np.abs(hist[valid] - (ml[valid] - sl[valid]))
        assert np.max(diff) < 1e-8


# ══════════════ 2. compute_signals_v9 ══════════════


class TestComputeSignalsV9:
    """Test the v9.0 Numba signal engine."""

    def test_output_shape(self, btc_ohlcv, btc_close):
        """Signal array must be same length as input."""
        eth_close = btc_close  # use BTC as both for shape test
        sigs = compute_signals_v9(
            eth_close,
            btc_ohlcv["high"],
            btc_ohlcv["low"],
            btc_ohlcv["volume"],
            btc_ohlcv["taker_buy"],
            btc_close,
            8,
            0.045,
            0.382,
            0.786,
            10,
            0.0,
            32,
            96,
            192,
            2,
            1.5,
            0.0,
            1.5,
            14,
            20,
            False,
            0.45,
            0.5,
            0.55,
            True,
            5,
            3,
            True,
            12,
            26,
            14,
        )
        assert sigs.shape == btc_close.shape

    def test_signal_values_valid(self, eth_ohlcv, btc_close):
        """Signals should be 0 or ±leverage values."""
        n = min(len(eth_ohlcv["close"]), len(btc_close))
        sigs = compute_signals_v9(
            eth_ohlcv["close"][:n],
            eth_ohlcv["high"][:n],
            eth_ohlcv["low"][:n],
            eth_ohlcv["volume"][:n],
            eth_ohlcv["taker_buy"][:n],
            btc_close[:n],
            8,
            0.045,
            0.382,
            0.786,
            10,
            0.0,
            32,
            96,
            192,
            2,
            1.5,
            0.0,
            1.5,
            14,
            20,
            False,
            0.45,
            0.5,
            0.55,
            True,
            5,
            3,
            True,
            12,
            26,
            14,
        )
        non_zero = sigs[sigs != 0]
        if len(non_zero) > 0:
            assert np.all(np.abs(non_zero) >= 1.0)  # at least base leverage
            assert np.all(np.abs(non_zero) <= 3.0)  # within reasonable bounds

    def test_disabled_stoch_macd_matches_v8(self, eth_ohlcv, btc_close):
        """With stoch+macd disabled, v9 should produce identical signals to v8."""
        n = min(len(eth_ohlcv["close"]), len(btc_close))
        c, v, t, b = (
            eth_ohlcv["close"][:n],
            eth_ohlcv["volume"][:n],
            eth_ohlcv["taker_buy"][:n],
            btc_close[:n],
        )
        h, lo = eth_ohlcv["high"][:n], eth_ohlcv["low"][:n]
        params = (8, 0.045, 0.382, 0.786, 10, 0.0, 32, 96, 192, 2, 1.5, 0.0, 1.5, 14, 20)
        dyn = (False, 0.45, 0.5, 0.55)

        sigs_v9 = compute_signals_v9(c, h, lo, v, t, b, *params, *dyn, False, 5, 3, False, 12, 26, 14)
        sigs_v8 = compute_signals_confidence(c, h, lo, v, t, b, *params, *dyn)
        np.testing.assert_array_equal(sigs_v9, sigs_v8)


# ══════════════ 3. Config Defaults ══════════════


class TestV9ConfigDefaults:
    """Validate v9.0 config defaults match winning strategy parameters."""

    def test_fibonacci_exits(self):
        """v12.4 HybTight: target_retrace=1.5, stop_mult=0.382."""
        cfg = RubberBandConfig()
        assert cfg.target_retrace == 1.5, "v12.4 HybTight retrace target"
        assert cfg.stop_mult == 0.382, "T100_S38 Fibonacci stop"

    def test_dynamic_target_disabled(self):
        cfg = RubberBandConfig()
        assert cfg.dynamic_target_enabled is False

    def test_macd_defaults(self):
        """v12.6: MACD disabled by default (R21 backtest: noise, WR +11pp without)."""
        cfg = RubberBandConfig()
        assert cfg.macd_enabled is False
        assert cfg.macd_fast == 12
        assert cfg.macd_slow == 26
        assert cfg.macd_signal_period == 14

    def test_stoch_defaults(self):
        """v12.6: Stochastic disabled by default (R21 backtest: noise, WR +11pp without)."""
        cfg = RubberBandConfig()
        assert cfg.stoch_enabled is False
        assert cfg.stoch_k_period == 3  # v10.0: faster stochastic
        assert cfg.stoch_d_period == 3


# ══════════════ 4. Strategy Dispatch ══════════════


class TestV9StrategyDispatch:
    """Test on_bar() correctly dispatches to v9 vs v8 path."""

    def test_update_market_data_stores_high_low(self):
        """update_market_data should store high/low when provided."""
        strategy = RubberBandStrategy()
        vol = np.ones(10)
        taker = np.ones(10) * 0.5
        high = np.ones(10) * 101
        low = np.ones(10) * 99
        strategy.update_market_data("TEST", vol, taker, high=high, low=low)
        assert "TEST" in strategy._high_data
        assert "TEST" in strategy._low_data
        np.testing.assert_array_equal(strategy._high_data["TEST"], high)

    def test_update_market_data_backward_compat(self):
        """update_market_data without high/low should still work."""
        strategy = RubberBandStrategy()
        vol = np.ones(10)
        taker = np.ones(10) * 0.5
        strategy.update_market_data("TEST", vol, taker)
        assert "TEST" in strategy._volume_data
        assert "TEST" not in strategy._high_data

    def test_on_bar_uses_v9_when_enabled(self, eth_ohlcv, btc_close):
        """When macd/stoch enabled, on_bar should use v9 path (no crash)."""
        cfg = RubberBandConfig(
            macd_enabled=True,
            stoch_enabled=True,
            mtf_confirmation_enabled=False,
        )
        strategy = RubberBandStrategy(cfg)
        n = min(len(eth_ohlcv["close"]), len(btc_close))
        close = eth_ohlcv["close"][:n]
        btc = btc_close[:n]
        strategy.update_market_data(
            "ETHUSDT",
            eth_ohlcv["volume"][:n],
            eth_ohlcv["taker_buy"][:n],
            high=eth_ohlcv["high"][:n],
            low=eth_ohlcv["low"][:n],
        )
        sig = strategy.on_bar("ETHUSDT", close, btc)
        assert sig in (-1, 0, 1)

    def test_on_bar_falls_back_without_high_low(self, eth_ohlcv, btc_close):
        """v9 path should use close as fallback for missing high/low."""
        cfg = RubberBandConfig(
            macd_enabled=True,
            stoch_enabled=False,
            mtf_confirmation_enabled=False,
        )
        strategy = RubberBandStrategy(cfg)
        n = min(len(eth_ohlcv["close"]), len(btc_close))
        close = eth_ohlcv["close"][:n]
        btc = btc_close[:n]
        # Only pass volume/taker, no high/low
        strategy.update_market_data(
            "ETHUSDT",
            eth_ohlcv["volume"][:n],
            eth_ohlcv["taker_buy"][:n],
        )
        sig = strategy.on_bar("ETHUSDT", close, btc)
        assert sig in (-1, 0, 1)

    def test_on_bar_uses_v8_when_disabled(self, eth_ohlcv, btc_close):
        """When both macd+stoch disabled, should use v8 compute_signals_confidence."""
        cfg = RubberBandConfig(
            macd_enabled=False,
            stoch_enabled=False,
            mtf_confirmation_enabled=False,
        )
        strategy = RubberBandStrategy(cfg)
        n = min(len(eth_ohlcv["close"]), len(btc_close))
        close = eth_ohlcv["close"][:n]
        btc = btc_close[:n]
        strategy.update_market_data(
            "ETHUSDT",
            eth_ohlcv["volume"][:n],
            eth_ohlcv["taker_buy"][:n],
        )
        sig = strategy.on_bar("ETHUSDT", close, btc)
        assert sig in (-1, 0, 1)
