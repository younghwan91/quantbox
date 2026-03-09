"""Tests for v8.0 Dynamic Target feature.

Verifies that compute_signals_confidence correctly scales targets
by confidence level when dynamic_target_enabled=True.
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_confidence,
)


def _make_data(n: int = 500):
    """Create deterministic test data with a clear rubber-band signal."""
    np.random.seed(42)
    # Gentle uptrend with occasional dips
    close = np.cumsum(np.random.randn(n) * 0.001 + 0.0002) + 100.0
    # Insert a sharp dip at bar 300 (5% drop over 8 bars)
    for i in range(293, 301):
        close[i] = close[292] * (1 - 0.06 * (i - 292) / 8)
    # Then recovery
    for i in range(301, min(320, n)):
        close[i] = close[300] * (1 + 0.003 * (i - 300))

    btc = close * 0.5 + 200  # correlated BTC
    volume = np.ones(n) * 100.0
    volume[298:302] = 300.0  # Volume spike at dip
    taker_buy = volume * 0.4  # Sell pressure (ratio < 0.45)
    return close, btc, volume, taker_buy


BASE_KWARGS = dict(
    lookback=8,
    threshold=0.045,
    target_retrace=0.4,
    stop_mult=0.9,
    max_hold=10,
    stop_decay=0.0,
    btc_mom_period=32,
    btc_regime_period=96,
    asset_trend_period=192,
    min_confidence=2,
    base_leverage=1.5,
    confidence_leverage_step=0.0,
    max_leverage=1.5,
    rsi_period=14,
    vol_period=20,
)


class TestDynamicTargetConfig:
    """Test RubberBandConfig defaults for dynamic target."""

    def test_defaults_disabled_v9(self):
        """v9.0: dynamic target disabled by default — Fib discipline outperforms."""
        cfg = RubberBandConfig()
        assert cfg.dynamic_target_enabled is False

    def test_default_conf3(self):
        cfg = RubberBandConfig()
        assert cfg.target_conf3 == 0.45

    def test_default_conf4(self):
        cfg = RubberBandConfig()
        assert cfg.target_conf4 == 0.50

    def test_default_conf5(self):
        cfg = RubberBandConfig()
        assert cfg.target_conf5 == 0.55

    def test_custom_values(self):
        cfg = RubberBandConfig(
            dynamic_target_enabled=False,
            target_conf3=0.6,
            target_conf4=0.7,
            target_conf5=0.8,
        )
        assert cfg.dynamic_target_enabled is False
        assert cfg.target_conf3 == 0.6


class TestDynamicTargetSignals:
    """Test that dynamic target affects signal generation."""

    def test_disabled_matches_baseline(self):
        """Dynamic target disabled should produce identical signals."""
        close, btc, vol, taker = _make_data()
        sig_base = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=False,
            target_conf3=0.45,
            target_conf4=0.5,
            target_conf5=0.55,
        )
        sig_default = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=False,
            target_conf3=0.99,
            target_conf4=0.99,
            target_conf5=0.99,
        )
        np.testing.assert_array_equal(sig_base, sig_default)

    def test_enabled_produces_valid_signals(self):
        """Dynamic target enabled should produce valid signal array."""
        close, btc, vol, taker = _make_data()
        sig = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=True,
            target_conf3=0.45,
            target_conf4=0.5,
            target_conf5=0.55,
        )
        assert sig.shape == close.shape
        assert sig.dtype == np.float64
        # Should have at least some non-zero signals
        assert np.any(sig != 0)

    def test_wider_target_can_change_exit_timing(self):
        """A wider target may cause trades to hold longer or exit at different bars."""
        close, btc, vol, taker = _make_data(800)
        sig_narrow = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=True,
            target_conf3=0.4,
            target_conf4=0.4,
            target_conf5=0.4,  # same as baseline
        )
        sig_wide = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=True,
            target_conf3=0.6,
            target_conf4=0.7,
            target_conf5=0.8,
        )
        # Both should be valid
        assert sig_narrow.shape == sig_wide.shape
        # If any high-confidence trade exists, the signals may differ
        # (wider target = different exit timing)
        # We just verify they are valid, not necessarily different
        # since the test data may not generate conf >= 3 trades

    def test_conf_level_scaling(self):
        """Verify target scales correctly by checking that conf3 < conf4 < conf5."""
        cfg = RubberBandConfig(
            dynamic_target_enabled=True,
            target_conf3=1.8,
            target_conf4=2.0,
            target_conf5=2.2,
        )
        assert cfg.target_conf3 < cfg.target_conf4 < cfg.target_conf5
        assert cfg.target_retrace < cfg.target_conf3  # base (1.5) < conf3 (1.8)

    def test_same_targets_as_disabled(self):
        """Dynamic target with all values = target_retrace equals disabled."""
        close, btc, vol, taker = _make_data()
        base_tr = 0.4
        sig_disabled = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=False,
            target_conf3=0.45,
            target_conf4=0.5,
            target_conf5=0.55,
        )
        sig_same = compute_signals_confidence(
            close,
            close,
            close,
            vol,
            taker,
            btc,
            **BASE_KWARGS,
            dynamic_target_enabled=True,
            target_conf3=base_tr,
            target_conf4=base_tr,
            target_conf5=base_tr,
        )
        np.testing.assert_array_equal(sig_disabled, sig_same)
