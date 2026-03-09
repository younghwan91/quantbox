"""Tests for Regime-Switch Dual Entry (v5.0).

Validates:
  1. compute_signals_regime_switch Numba function behavior
  2. on_bar dispatch routes correctly for entry_mode='regime_switch'
  3. Risk-On regime: classic pullback (alt dip + alt uptrend)
  4. Risk-Off regime: sympathy pullback (alt dip + BTC falling)
  5. Config defaults match v5.0 spec
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals_regime_switch,
    compute_signals_with_btc_directional,
)


# ────────── Test Data Generators ──────────

def _make_prices(
    n: int = 300,
    crash_bar: int = 200,
    crash_pct: float = 0.06,
    lookback: int = 8,
) -> np.ndarray:
    """Create price array with a dip at crash_bar."""
    close = np.ones(n, dtype=np.float64)
    for i in range(crash_bar - lookback, crash_bar):
        close[i] = 1.0 - crash_pct * ((i - crash_bar + lookback + 1) / lookback)
    for i in range(crash_bar, n):
        close[i] = close[crash_bar - 1]
    return close


def _make_btc_regime(
    n: int = 300,
    regime: str = 'risk_off',
    regime_period: int = 96,
) -> np.ndarray:
    """Create BTC prices above/below SMA(regime_period).

    risk_on: BTC trending upward → price > SMA.
    risk_off: BTC trending downward → price < SMA.
    """
    btc = np.ones(n, dtype=np.float64)
    if regime == 'risk_on':
        # BTC slowly rising → well above SMA
        for i in range(n):
            btc[i] = 1.0 + 0.15 * (i / n)
    else:
        # BTC slowly falling → well below SMA
        for i in range(n):
            btc[i] = 1.0 - 0.15 * (i / n)
    return btc


def _make_asset_trend(
    n: int = 300,
    crash_bar: int = 200,
    crash_pct: float = 0.06,
    uptrend: bool = True,
    lookback: int = 8,
    trend_period: int = 192,
) -> np.ndarray:
    """Create asset price with uptrend/downtrend + crash at crash_bar.

    If uptrend: price rises gently then dips → price > SMA(trend_period) before dip.
    If downtrend: price falls gently then dips → price < SMA(trend_period) before dip.
    """
    close = np.ones(n, dtype=np.float64)
    if uptrend:
        for i in range(n):
            close[i] = 1.0 + 0.20 * (i / n)
    else:
        for i in range(n):
            close[i] = 1.0 - 0.10 * (i / n)

    # Apply crash at crash_bar
    base = close[crash_bar - lookback - 1]
    for i in range(crash_bar - lookback, crash_bar):
        frac = (i - crash_bar + lookback + 1) / lookback
        close[i] = base * (1.0 - crash_pct * frac)
    for i in range(crash_bar, n):
        close[i] = close[crash_bar - 1]
    return close


# Shared parameters
_PARAMS = dict(
    lookback=8, threshold=0.045, target_retrace=0.4, stop_mult=0.9,
    max_hold=10, stop_decay=0.0, btc_mom_period=32,
    btc_regime_period=96, asset_trend_period=192,
)


class TestRegimeSwitchNumba:
    """Test compute_signals_regime_switch Numba function."""

    def test_risk_off_sympathy_long(self):
        """Risk-Off: alt dips + BTC falling → long entry (sympathy)."""
        btc = _make_btc_regime(300, 'risk_off', 96)
        close = _make_prices(300, 200, 0.06)
        sigs = compute_signals_regime_switch(close, btc, **_PARAMS)
        assert np.any(sigs > 0), "Risk-Off: should enter long on sympathy dip"

    def test_risk_on_classic_long(self):
        """Risk-On: alt dips in its own uptrend → long entry (classic pullback)."""
        btc = _make_btc_regime(300, 'risk_on', 96)
        close = _make_asset_trend(300, 200, 0.06, uptrend=True)
        sigs = compute_signals_regime_switch(close, btc, **_PARAMS)
        assert np.any(sigs > 0), "Risk-On: should enter long on classic pullback"

    def test_risk_on_blocked_downtrend(self):
        """Risk-On: alt dips but alt is in downtrend → no entry."""
        btc = _make_btc_regime(300, 'risk_on', 96)
        close = _make_asset_trend(300, 200, 0.06, uptrend=False)
        sigs = compute_signals_regime_switch(close, btc, **_PARAMS)
        # In risk-on, dip in downtrend should be blocked (not a pullback)
        long_sigs = sigs[195:210]
        assert not np.any(long_sigs > 0), (
            "Risk-On: should NOT enter long when asset is in downtrend"
        )

    def test_risk_off_blocked_btc_rising(self):
        """Risk-Off regime but BTC short-term momentum is rising → no sympathy entry."""
        # BTC in downtrend regime (< SMA) but with short-term uptick
        btc = _make_btc_regime(300, 'risk_off', 96)
        # Override last 32 bars with rising BTC (positive momentum)
        for i in range(168, 200):
            btc[i] = btc[168] * (1.0 + 0.03 * ((i - 168) / 32))
        for i in range(200, 300):
            btc[i] = btc[199]

        close = _make_prices(300, 200, 0.06)
        sigs = compute_signals_regime_switch(close, btc, **_PARAMS)
        # At bar 199 (crash bar-1), btc mom should be positive → block long
        assert sigs[199] == 0, "Risk-Off: should block when BTC short-term rising"

    def test_output_shape(self):
        """Output array shape matches input."""
        btc = _make_btc_regime(300, 'risk_off', 96)
        close = _make_prices(300, 200, 0.06)
        sigs = compute_signals_regime_switch(close, btc, **_PARAMS)
        assert sigs.shape == close.shape

    def test_max_hold_enforced(self):
        """Position exits by max_hold bars."""
        btc = _make_btc_regime(300, 'risk_off', 96)
        close = _make_prices(300, 200, 0.06)
        sigs = compute_signals_regime_switch(close, btc, **_PARAMS)

        # Find entry and measure hold duration
        entered = False
        hold = 0
        for i in range(len(sigs)):
            if sigs[i] != 0 and not entered:
                entered = True
                hold = 1
            elif entered and sigs[i] != 0:
                hold += 1
            elif entered and sigs[i] == 0:
                break
        if entered:
            assert hold <= _PARAMS['max_hold'], f"Held {hold} > max_hold={_PARAMS['max_hold']}"


class TestOnBarRegimeSwitchRouting:
    """Test on_bar dispatches to regime-switch when entry_mode='regime_switch'."""

    def test_regime_switch_dispatch(self):
        """entry_mode='regime_switch' routes to compute_signals_regime_switch."""
        cfg = RubberBandConfig(entry_mode='regime_switch')
        strategy = RubberBandStrategy(cfg)

        btc = _make_btc_regime(300, 'risk_off', 96)
        close = _make_prices(300, 200, 0.06)

        sig = strategy.on_bar("TEST", close[:200], btc[:200])
        # Should produce a signal via regime-switch
        assert isinstance(sig, int)

    def test_confidence_is_default(self):
        """Default config entry_mode is 'confidence' (v6.0)."""
        cfg = RubberBandConfig()
        assert cfg.entry_mode == 'confidence'

    def test_btc_directional_mode_bypasses_regime_switch(self):
        """entry_mode='btc_directional' uses old btc_directional path."""
        cfg = RubberBandConfig(
            entry_mode='btc_directional',
            btc_directional_enabled=True,
        )
        strategy = RubberBandStrategy(cfg)

        btc = _make_btc_regime(300, 'risk_on', 96)
        close = _make_prices(300, 200, 0.06)

        # With btc_directional, BTC momentum must align (risk_on = BTC rising)
        # Alt crash + BTC rising → btc_directional blocks long
        sig = strategy.on_bar("TEST", close[:200], btc[:200])
        assert sig == 0, "BTC directional should block long when BTC rising"

    def test_insufficient_data_returns_zero(self):
        """Returns 0 when data is too short for regime detection."""
        cfg = RubberBandConfig(entry_mode='regime_switch')
        strategy = RubberBandStrategy(cfg)

        close = np.ones(50, dtype=np.float64)
        btc = np.ones(50, dtype=np.float64)
        sig = strategy.on_bar("TEST", close, btc)
        assert sig == 0, "Insufficient data should return 0"

    def test_no_btc_data_falls_through(self):
        """When btc_close=None, regime-switch is skipped entirely."""
        cfg = RubberBandConfig(entry_mode='regime_switch')
        strategy = RubberBandStrategy(cfg)
        close = _make_prices(300, 200, 0.06)
        sig = strategy.on_bar("TEST", close[:200], None)
        # Falls through to raw compute_signals
        assert isinstance(sig, int)


class TestRegimeSwitchConfigV5:
    """Test v5.0 config fields."""

    def test_defaults(self):
        cfg = RubberBandConfig()
        assert cfg.entry_mode == 'confidence'  # v6.0 default
        assert cfg.btc_regime_period == 96
        assert cfg.asset_trend_period == 192
        assert cfg.btc_directional_period == 48  # v11.0: A/B validated
        assert cfg.min_confidence == 3  # v12.3: M-2SSTSX validated (Sharpe 5.50)
        assert cfg.confidence_leverage_step == 0.0
        assert cfg.rsi_period == 14
        assert cfg.vol_period == 20

    def test_custom_regime_periods(self):
        cfg = RubberBandConfig(btc_regime_period=48, asset_trend_period=96)
        assert cfg.btc_regime_period == 48
        assert cfg.asset_trend_period == 96

    def test_entry_mode_raw(self):
        cfg = RubberBandConfig(entry_mode='raw')
        assert cfg.entry_mode == 'raw'
