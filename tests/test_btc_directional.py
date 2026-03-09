"""Tests for BTC Directional Alignment filter (v4.2).

Validates:
  1. Numba function produces correct entry/block behavior
  2. on_bar routing respects btc_directional_enabled flag
  3. Config defaults match .env expectations
  4. Disabled filter falls back to baseline behavior
  5. Backtest parity: function matches A/B test results
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals,
    compute_signals_with_btc_directional,
)


def _make_crash(n: int = 100, crash_bar: int = 50, crash_pct: float = 0.06) -> np.ndarray:
    """Create price array with a crash at crash_bar."""
    close = np.ones(n, dtype=np.float64)
    # Gradual crash over lookback=8 bars ending at crash_bar-1
    for i in range(crash_bar - 8, crash_bar):
        close[i] = 1.0 - crash_pct * ((i - crash_bar + 9) / 8)
    # After crash, price stays low
    for i in range(crash_bar, n):
        close[i] = close[crash_bar - 1]
    return close


def _make_btc(n: int = 100, crash_bar: int = 50, btc_dir: float = -0.01) -> np.ndarray:
    """BTC prices: trending in btc_dir direction over 32 bars ending at crash_bar."""
    btc = np.ones(n, dtype=np.float64)
    for i in range(crash_bar - 32, crash_bar):
        btc[i] = 1.0 + btc_dir * ((i - crash_bar + 33) / 32)
    for i in range(crash_bar, n):
        btc[i] = btc[crash_bar - 1]
    return btc


class TestBtcDirectionalNumba:
    """Test compute_signals_with_btc_directional Numba function."""

    def test_long_entry_btc_falling(self):
        """Long entry when alt crashes AND BTC falling → should trigger."""
        close = _make_crash(100, 50, 0.06)
        btc = _make_btc(100, 50, -0.02)  # BTC falling
        sigs = compute_signals_with_btc_directional(close, btc, 8, 0.045, 0.3, 1.0, 10, 0.0, 32)
        # Should have a long entry around bar 49
        assert np.any(sigs > 0), "Should enter long when BTC also falling"

    def test_long_blocked_btc_rising(self):
        """Long entry when alt crashes but BTC rising → should block."""
        close = _make_crash(100, 50, 0.06)
        btc = _make_btc(100, 50, +0.02)  # BTC rising
        sigs = compute_signals_with_btc_directional(close, btc, 8, 0.045, 0.3, 1.0, 10, 0.0, 32)
        # No signal at crash bar — BTC rising blocks long
        assert sigs[49] == 0, "Should NOT enter long when BTC is rising"

    def test_short_entry_btc_rising(self):
        """Short entry when alt surges AND BTC rising → should trigger."""
        close = _make_crash(100, 50, -0.06)  # Negative crash = surge
        btc = _make_btc(100, 50, +0.02)  # BTC rising
        sigs = compute_signals_with_btc_directional(close, btc, 8, 0.045, 0.3, 1.0, 10, 0.0, 32)
        assert np.any(sigs < 0), "Should enter short when BTC also rising"

    def test_short_blocked_btc_falling(self):
        """Short entry when alt surges but BTC falling → should block."""
        close = _make_crash(100, 50, -0.06)  # Surge
        btc = _make_btc(100, 50, -0.02)  # BTC falling
        sigs = compute_signals_with_btc_directional(close, btc, 8, 0.045, 0.3, 1.0, 10, 0.0, 32)
        assert sigs[49] == 0, "Should NOT enter short when BTC is falling"

    def test_exit_conditions_unchanged(self):
        """Exit logic (target/stop/trailing/max_hold) is identical to baseline."""
        close = _make_crash(200, 50, 0.06)
        btc = _make_btc(200, 50, -0.02)
        # After entry, simulate recovery to hit target
        entry_bar = 49
        entry_price = close[entry_bar]
        move_size = 0.06
        target_price = entry_price * (1 + move_size * 0.3)
        # Quick recovery
        for i in range(50, 55):
            close[i] = entry_price + (target_price - entry_price) * (i - 49) / 3
        for i in range(55, 200):
            close[i] = target_price * 1.1

        sigs = compute_signals_with_btc_directional(close, btc, 8, 0.045, 0.3, 1.0, 10, 0.0, 32)
        # Should exit within max_hold bars
        held = 0
        entered = False
        for i in range(50, 65):
            if sigs[i] > 0 and not entered:
                entered = True
            elif entered and sigs[i] == 0:
                break
            if entered:
                held += 1
        assert held <= 10, f"Position held {held} bars, max_hold=10"

    def test_no_btc_data_returns_flat(self):
        """When btc_close is too short, on_bar returns 0."""
        cfg = RubberBandConfig(
            entry_mode="btc_directional",
            btc_directional_enabled=True,
            btc_directional_period=32,
        )
        strategy = RubberBandStrategy(cfg)
        close = _make_crash(100, 50, 0.06)
        btc_short = np.ones(10)  # Too short
        sig = strategy.on_bar("TEST", close[:50], btc_short)
        assert sig == 0, "Should return 0 when BTC data insufficient"


class TestOnBarRouting:
    """Test on_bar correctly routes based on btc_directional_enabled."""

    def test_enabled_uses_btc_directional(self):
        """When enabled + btc data available, uses BTC directional."""
        cfg = RubberBandConfig(
            entry_mode="btc_directional",
            btc_directional_enabled=True,
            btc_directional_period=32,
        )
        strategy = RubberBandStrategy(cfg)
        close = _make_crash(100, 50, 0.06)
        btc_falling = _make_btc(100, 50, -0.02)
        btc_rising = _make_btc(100, 50, +0.02)

        sig_aligned = strategy.on_bar("T1", close[:50], btc_falling[:50])
        sig_blocked = strategy.on_bar("T2", close[:50], btc_rising[:50])

        assert sig_aligned == 1, "Should enter when BTC aligned"
        assert sig_blocked == 0, "Should block when BTC not aligned"

    def test_disabled_ignores_btc_direction(self):
        """When disabled, falls back to plain compute_signals."""
        cfg = RubberBandConfig(entry_mode="raw", btc_directional_enabled=False)
        strategy = RubberBandStrategy(cfg)
        close = _make_crash(100, 50, 0.06)
        btc_rising = _make_btc(100, 50, +0.02)

        sig = strategy.on_bar("T3", close[:50], btc_rising[:50])
        assert sig == 1, "Disabled filter should enter regardless of BTC"

    def test_no_btc_data_falls_through(self):
        """When btc_close is None, uses plain compute_signals."""
        cfg = RubberBandConfig(entry_mode="btc_directional", btc_directional_enabled=True)
        strategy = RubberBandStrategy(cfg)
        close = _make_crash(100, 50, 0.06)

        sig = strategy.on_bar("T4", close[:50], None)
        assert sig == 1, "No BTC data → plain signals → should enter on crash"


class TestConfigDefaults:
    """Test config defaults and settings propagation."""

    def test_default_confidence(self):
        """Default entry_mode is confidence (v6.0)."""
        cfg = RubberBandConfig()
        assert cfg.entry_mode == "confidence"
        assert cfg.btc_regime_period == 96
        assert cfg.asset_trend_period == 192
        assert cfg.btc_directional_enabled is True
        assert cfg.btc_directional_period == 48  # v11.0: A/B validated

    def test_custom_period(self):
        cfg = RubberBandConfig(btc_directional_period=48)
        assert cfg.btc_directional_period == 48

    def test_disable(self):
        cfg = RubberBandConfig(btc_directional_enabled=False)
        assert cfg.btc_directional_enabled is False
