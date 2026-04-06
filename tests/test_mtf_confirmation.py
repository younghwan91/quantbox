"""Tests for v7.0 Multi-Timeframe Confirmation + Independent 4H Trading.

Tests:
  1. apply_mtf_confirmation blocks counter-trend entries
  2. apply_mtf_confirmation preserves trend-aligned entries
  3. apply_mtf_leverage scales signals correctly
  4. compute_signals_4h_independent generates valid signals
  5. RubberBandConfig has MTF fields with correct defaults
  6. on_bar applies MTF confirmation when htf data available
  7. on_bar_4h generates 4h signals
  8. Backtest/live parity: same signals from backtest and on_bar
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_signals_4h_independent,
)


# ── Fixtures ──


@pytest.fixture
def trending_close_4h():
    """4h close prices with clear uptrend (above SMA)."""
    np.random.seed(42)
    n = 100
    # Uptrend: steadily rising with small noise
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 2
    return trend + noise


@pytest.fixture
def downtrend_close_4h():
    """4h close prices with clear downtrend (below SMA)."""
    np.random.seed(42)
    n = 100
    trend = np.linspace(200, 100, n)
    noise = np.random.randn(n) * 2
    return trend + noise


@pytest.fixture
def mixed_signals_15m():
    """15m signals with both long and short entries."""
    n = 1600  # 100 4h bars × 16 15m bars
    signals = np.zeros(n)
    # Scatter some entries
    signals[300] = 1.5  # Long
    signals[301] = 1.5
    signals[302] = 1.5
    signals[500] = -1.5  # Short
    signals[501] = -1.5
    signals[800] = 1.5  # Long
    signals[1200] = -1.5  # Short
    return signals


@pytest.fixture
def v7_config():
    """v7.0 config with MTF enabled."""
    return RubberBandConfig(
        mtf_confirmation_enabled=True,
        mtf_sma_period=12,
        mtf_leverage_mult=2.0,
        htf_independent_enabled=True,
        htf_rb_lookback=3,
        htf_rb_threshold=0.10,
        htf_rb_target_retrace=0.5,
        htf_rb_stop_mult=0.8,
        htf_rb_max_hold=12,
        htf_rb_btc_sma_period=48,
    )


# ── Test apply_mtf_confirmation ──


def test_mtf_blocks_counter_trend_long(trending_close_4h, downtrend_close_4h):
    """Long signal in downtrend should be blocked."""
    n = 1600
    signals = np.zeros(n)
    signals[800] = 1.5  # Long entry at bar 800 → maps to 4h bar 50

    # Downtrend: close < SMA → long should be blocked
    filtered = apply_mtf_confirmation(signals, downtrend_close_4h, 12, 16)
    assert filtered[800] == 0.0, "Long should be blocked in downtrend"


def test_mtf_blocks_counter_trend_short(trending_close_4h):
    """Short signal in uptrend should be blocked."""
    n = 1600
    signals = np.zeros(n)
    signals[800] = -1.5  # Short entry at bar 800

    # Uptrend: close > SMA → short should be blocked
    filtered = apply_mtf_confirmation(signals, trending_close_4h, 12, 16)
    assert filtered[800] == 0.0, "Short should be blocked in uptrend"


def test_mtf_preserves_trend_aligned_long(trending_close_4h):
    """Long signal in uptrend should be preserved."""
    n = 1600
    signals = np.zeros(n)
    signals[800] = 1.5

    # Uptrend: close > SMA → long should pass
    filtered = apply_mtf_confirmation(signals, trending_close_4h, 12, 16)
    assert filtered[800] == 1.5, "Long should be preserved in uptrend"


def test_mtf_preserves_trend_aligned_short(downtrend_close_4h):
    """Short signal in downtrend should be preserved."""
    n = 1600
    signals = np.zeros(n)
    signals[800] = -1.5

    # Downtrend: close < SMA → short should pass
    filtered = apply_mtf_confirmation(signals, downtrend_close_4h, 12, 16)
    assert filtered[800] == -1.5, "Short should be preserved in downtrend"


def test_mtf_preserves_zero_signals():
    """Zero signals should remain zero."""
    close_4h = np.linspace(100, 200, 100)
    signals = np.zeros(1600)
    filtered = apply_mtf_confirmation(signals, close_4h, 12, 16)
    assert np.all(filtered == 0.0)


def test_mtf_does_not_modify_input():
    """apply_mtf_confirmation should return a copy, not modify input."""
    close_4h = np.linspace(100, 200, 100)
    signals = np.zeros(1600)
    signals[800] = 1.5
    original = signals.copy()
    _ = apply_mtf_confirmation(signals, close_4h, 12, 16)
    np.testing.assert_array_equal(signals, original)


# ── Test apply_mtf_leverage ──


def test_mtf_leverage_scales_signals():
    """Leverage multiplier should scale signal magnitude."""
    signals = np.array([0, 1.5, -1.5, 0, 1.5])
    scaled = apply_mtf_leverage(signals, 2.0)
    expected = np.array([0, 3.0, -3.0, 0, 3.0])
    np.testing.assert_array_almost_equal(scaled, expected)


def test_mtf_leverage_identity():
    """1.0x leverage should not change signals."""
    signals = np.array([0, 1.5, -1.5])
    scaled = apply_mtf_leverage(signals, 1.0)
    np.testing.assert_array_almost_equal(scaled, signals)


# ── Test compute_signals_4h_independent ──


def test_4h_signals_shape():
    """Output shape should match input."""
    np.random.seed(42)
    close = np.abs(np.random.randn(200).cumsum() + 100)
    btc = np.abs(np.random.randn(200).cumsum() + 100)
    sigs = compute_signals_4h_independent(close, btc, 3, 0.10, 0.5, 0.8, 12, 48)
    assert len(sigs) == len(close)


def test_4h_signals_values():
    """Signals should be -1, 0, or +1."""
    np.random.seed(42)
    close = np.abs(np.random.randn(200).cumsum() + 100)
    btc = np.abs(np.random.randn(200).cumsum() + 100)
    sigs = compute_signals_4h_independent(close, btc, 3, 0.10, 0.5, 0.8, 12, 48)
    unique = set(sigs)
    assert unique.issubset({-1.0, 0.0, 1.0})


def test_4h_high_threshold_no_trades():
    """Absurdly high threshold should produce no trades."""
    close = np.linspace(100, 101, 200)  # Very flat
    btc = np.linspace(100, 101, 200)
    sigs = compute_signals_4h_independent(close, btc, 3, 0.50, 0.5, 0.8, 12, 48)
    assert np.all(sigs == 0)


# ── Test RubberBandConfig defaults ──


def test_config_mtf_defaults():
    """v7.0 MTF config fields should have correct defaults."""
    cfg = RubberBandConfig()
    assert cfg.mtf_confirmation_enabled is True
    assert cfg.mtf_sma_period == 12
    assert cfg.mtf_leverage_mult == 2.5  # v10.1: boosted from 2.0
    assert cfg.htf_independent_enabled is False  # v10.2: disabled (dead code)
    assert cfg.htf_rb_lookback == 3
    assert cfg.htf_rb_threshold == 0.10
    assert cfg.htf_rb_target_retrace == 0.5
    assert cfg.htf_rb_stop_mult == 0.8
    assert cfg.htf_rb_max_hold == 12
    assert cfg.htf_rb_btc_sma_period == 48


def test_config_mtf_disabled():
    """MTF should be disableable via config."""
    cfg = RubberBandConfig(mtf_confirmation_enabled=False)
    assert cfg.mtf_confirmation_enabled is False


# ── Test RubberBandStrategy on_bar with MTF ──


def test_on_bar_applies_mtf_confirmation():
    """on_bar should apply MTF confirmation when htf data is available."""
    cfg = RubberBandConfig(
        mtf_confirmation_enabled=True,
        mtf_sma_period=6,
        mtf_leverage_mult=1.0,
        entry_mode="confidence",
        min_confidence=1,  # Low bar so we get signals
    )
    strategy = RubberBandStrategy(cfg)

    # Create data where 15m shows a big dip (long signal) but 4h is in downtrend
    np.random.seed(123)
    n = 200
    close_15m = np.ones(n) * 100
    # Big dip at the end to trigger entry
    close_15m[-1] = 91.0  # -9% drop (should trigger at 4.5% threshold)
    btc_close = np.ones(n) * 50000

    # Volume and taker data
    volume = np.ones(n) * 1000
    taker_buy = np.ones(n) * 400

    # 4h data: clear downtrend (should block long entries)
    close_4h = np.linspace(200, 100, 20)

    strategy.update_market_data("TESTUSDT", volume, taker_buy)
    strategy.update_htf_data("TESTUSDT", close_4h)

    signal = strategy.on_bar("TESTUSDT", close_15m, btc_close)
    # In downtrend, long entries should be blocked by MTF
    assert signal == 0, "Long signal should be blocked by downtrend 4h confirmation"


def test_on_bar_without_htf_data_falls_through():
    """Without htf data, on_bar should still work (no MTF filter applied)."""
    cfg = RubberBandConfig(
        mtf_confirmation_enabled=True,
        entry_mode="confidence",
    )
    strategy = RubberBandStrategy(cfg)

    # Provide volume/taker but NO htf data
    n = 200
    close = np.ones(n) * 100
    btc = np.ones(n) * 50000
    volume = np.ones(n) * 1000
    taker_buy = np.ones(n) * 400

    strategy.update_market_data("TESTUSDT", volume, taker_buy)
    # Don't call update_htf_data — MTF should be skipped gracefully
    signal = strategy.on_bar("TESTUSDT", close, btc)
    # Should not crash, just return whatever confidence signals give
    assert signal in (-1, 0, 1)


# ── Test on_bar_4h ──


def test_on_bar_4h_generates_signal():
    """on_bar_4h should use 4h independent parameters."""
    cfg = RubberBandConfig(
        htf_independent_enabled=True,
        htf_rb_lookback=3,
        htf_rb_threshold=0.10,
        htf_rb_btc_sma_period=12,
    )
    strategy = RubberBandStrategy(cfg)

    np.random.seed(42)
    close_4h = np.abs(np.random.randn(100).cumsum() + 100)
    btc_4h = np.abs(np.random.randn(100).cumsum() + 100)

    signal = strategy.on_bar_4h("TESTUSDT", close_4h, btc_4h)
    assert signal in (-1, 0, 1)


def test_on_bar_4h_disabled():
    """on_bar_4h should return 0 when htf_independent_enabled=False."""
    cfg = RubberBandConfig(htf_independent_enabled=False)
    strategy = RubberBandStrategy(cfg)

    close_4h = np.linspace(100, 200, 100)
    btc_4h = np.linspace(100, 200, 100)

    assert strategy.on_bar_4h("TESTUSDT", close_4h, btc_4h) == 0


def test_on_bar_4h_no_btc():
    """on_bar_4h should return 0 without BTC data."""
    cfg = RubberBandConfig(htf_independent_enabled=True)
    strategy = RubberBandStrategy(cfg)

    close_4h = np.linspace(100, 200, 100)
    assert strategy.on_bar_4h("TESTUSDT", close_4h, None) == 0
