"""Tests for v12.6 Climax Volume Filter + ATR-Based Stop.

Verifies:
1. Config defaults reflect R21 champion settings (Stoch/MACD off, Climax/ATR on)
2. get_effective_stop_mult() correctly returns ATR-based stop
3. ATR stop floor and cap are enforced
4. Climax confidence evidence correctly increments confidence on volume spike
5. QuantBoxSettings.to_rb_config() maps new v12.6 fields correctly

No synthetic backtest data — unit tests only.
"""

from __future__ import annotations

import numpy as np

from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy


# ═══════════════════════ 1. Config Defaults ═══════════════════════


class TestV126ConfigDefaults:
    """v12.6 default values must match R21 champion [E]: Clmx×3.0+ATR2."""

    def test_macd_disabled_by_default(self):
        """v12.6: MACD is noise — disabled (R21 WR +11pp without)."""
        assert RubberBandConfig().macd_enabled is False

    def test_stoch_disabled_by_default(self):
        """v12.6: Stochastic is noise — disabled (R21 WR +11pp without)."""
        assert RubberBandConfig().stoch_enabled is False

    def test_climax_enabled_by_default(self):
        assert RubberBandConfig().climax_enabled is True

    def test_climax_vol_mult_default(self):
        """R21 best: 3.0× vol SMA."""
        assert RubberBandConfig().climax_vol_mult == 3.0

    def test_atr_stop_enabled_by_default(self):
        assert RubberBandConfig().atr_stop_enabled is True

    def test_atr_stop_mult_default(self):
        """R21 champion: 2.0× ATR%."""
        assert RubberBandConfig().atr_stop_mult == 2.0

    def test_atr_stop_floor_default(self):
        assert RubberBandConfig().atr_stop_floor == 0.005

    def test_atr_stop_cap_default(self):
        assert RubberBandConfig().atr_stop_cap == 0.04


# ═══════════════════════ 2. ATR Stop Calculation ═══════════════════════


def _make_strategy(
    atr_stop_enabled=True, atr_stop_mult=2.0, atr_stop_floor=0.005, atr_stop_cap=0.04
) -> RubberBandStrategy:
    """Build a strategy instance with custom v12.6 ATR stop settings."""
    cfg = RubberBandConfig(
        atr_stop_enabled=atr_stop_enabled,
        atr_stop_mult=atr_stop_mult,
        atr_stop_floor=atr_stop_floor,
        atr_stop_cap=atr_stop_cap,
        htf_period=0,  # disable HTF to isolate ATR path
    )
    return RubberBandStrategy(cfg)


class TestATRStopMult:
    """get_effective_stop_mult() must return ATR-based stop when enabled."""

    def test_basic_atr_stop(self):
        """SL = 2.0×ATR / move_size = stop_mult equivalent."""
        strat = _make_strategy()
        strat._last_atr_pct["TESTUSDT"] = 0.015  # 1.5% ATR
        close = np.ones(50)  # dummy close prices (HTF disabled)

        result = strat.get_effective_stop_mult(
            close,
            symbol="TESTUSDT",
            move_size=0.05,
        )
        # Expected: 2.0 × 0.015 / 0.05 = 0.6
        assert abs(result - 0.60) < 1e-9

    def test_atr_stop_floor_enforcement(self):
        """If 2×ATR < floor (0.5%), clamp to floor."""
        strat = _make_strategy()
        strat._last_atr_pct["TESTUSDT"] = 0.001  # 0.1% ATR → raw SL = 0.2%
        close = np.ones(50)

        result = strat.get_effective_stop_mult(
            close,
            symbol="TESTUSDT",
            move_size=0.05,
        )
        # Raw: 2.0 × 0.001 = 0.002 < floor 0.005 → clamped to 0.005
        expected = 0.005 / 0.05  # = 0.1
        assert abs(result - expected) < 1e-9

    def test_atr_stop_cap_enforcement(self):
        """If 2×ATR > cap (4%), clamp to cap."""
        strat = _make_strategy()
        strat._last_atr_pct["TESTUSDT"] = 0.03  # 3% ATR → raw SL = 6%
        close = np.ones(50)

        result = strat.get_effective_stop_mult(
            close,
            symbol="TESTUSDT",
            move_size=0.05,
        )
        # Raw: 2.0 × 0.03 = 0.06 > cap 0.04 → clamped to 0.04
        expected = 0.04 / 0.05  # = 0.8
        assert abs(result - expected) < 1e-9

    def test_fallback_to_fixed_stop_when_atr_disabled(self):
        """When atr_stop_enabled=False, returns cfg.stop_mult (legacy path)."""
        strat = _make_strategy(atr_stop_enabled=False)
        strat._last_atr_pct["TESTUSDT"] = 0.015
        close = np.ones(50)

        result = strat.get_effective_stop_mult(
            close,
            symbol="TESTUSDT",
            move_size=0.05,
        )
        # Falls through to fixed path: cfg.stop_mult = 0.382
        assert abs(result - 0.382) < 1e-6

    def test_fallback_when_symbol_not_provided(self):
        """Missing symbol → legacy fixed stop path."""
        strat = _make_strategy()
        close = np.ones(50)
        result = strat.get_effective_stop_mult(close)
        assert abs(result - 0.382) < 1e-6

    def test_fallback_when_move_size_zero(self):
        """move_size=0 → legacy fixed stop (avoid division by zero)."""
        strat = _make_strategy()
        strat._last_atr_pct["TESTUSDT"] = 0.015
        close = np.ones(50)
        result = strat.get_effective_stop_mult(
            close,
            symbol="TESTUSDT",
            move_size=0.0,
        )
        assert abs(result - 0.382) < 1e-6


# ═══════════════════════ 3. Climax Confidence Integration ═══════════════════════


def _make_confidence_data(n: int = 200, vol_spike_mult: float = 1.0):
    """Build deterministic confidence-scoring data.

    Creates a clear LONG setup:
    - Price dips 5% over lookback window (ret < -threshold)
    - BTC falls 2% over directional period (btc_mom < -0.01)
    - RSI oversold (<35): constructed by a steep downtrend
    - Taker ratio = 0.38 (sell pressure < 0.42)
    - Volume: vol_spike_mult × vol_SMA driven by last bar

    Returns:
        close, btc, volume, taker arrays.
    """
    # RSI-oversold close: stable then sharp drop
    close = np.full(n, 100.0)
    # Gradual rise then pullback at the end to yield RSI<35
    for i in range(n - 30, n):
        close[i] = 100.0 * (1 - 0.006 * (i - (n - 30)))  # -0.6% per bar = -17% over 30 bars

    # BTC: falls over directional period (48 bars)
    btc = np.full(n, 5000.0)
    btc[-1] = btc[-49] * 0.97  # -3% over 48 bars (btc_mom < -0.01)

    # Volume: base=100, last bar spiked
    volume = np.full(n, 100.0, dtype=np.float64)
    vol_period = 20
    # vol_SMA_val ≈ 100 (uniform). Last bar = vol_spike_mult × 100
    volume[-1] = 100.0 * vol_spike_mult

    # Taker: 38% buy (sell pressure → taker_ratio < 0.42)
    taker = volume * 0.38

    return close.astype(np.float64), btc.astype(np.float64), volume, taker


class TestClimaxConfidenceScoring:
    """Climax volume filter correctly adds +1 to confidence."""

    def _make_strategy(self, climax_enabled: bool, min_confidence: int = 5) -> RubberBandStrategy:
        """Strategy with controls to isolate climax contribution."""
        cfg = RubberBandConfig(
            lookback=8,
            threshold=0.042,  # threshold < 5% dip → signals
            atr_period=0,  # disable adaptive threshold
            btc_directional_period=48,
            btc_regime_period=96,
            min_confidence=min_confidence,
            climax_enabled=climax_enabled,
            climax_vol_mult=3.0,
            stoch_enabled=False,  # already default
            macd_enabled=False,  # already default
            atr_stop_enabled=False,  # test stop separately
            mtf_confirmation_enabled=False,  # disable 4h filter
        )
        strat = RubberBandStrategy(cfg)
        return strat

    def test_climax_adds_confidence_point(self):
        """With min_confidence=5 and climax vol spike, confidence reaches 5.

        vol_spike_mult=4.5 gives vol_ratio ≈ 3.83 after SMA smoothing
        (vol_SMA includes the spike bar, so raw×mult isn't the final ratio).
        Evidence: RSI<35(+1) + vol>1.5(+1) + taker<0.42(+1) + btc_mom<-0.01(+1)
        + climax vol≥3.0(+1) = 5 → signal fires.
        """
        close, btc, vol, taker = _make_confidence_data(vol_spike_mult=4.5)
        strat = self._make_strategy(climax_enabled=True, min_confidence=5)
        strat.update_market_data("TESTUSDT", vol, taker)

        sig = strat.on_bar("TESTUSDT", close, btc)
        assert sig == 1, f"Expected LONG signal with climax vol spike, got {sig}"

    def test_climax_disabled_blocks_signal_at_min_conf5(self):
        """Without climax, confidence=4 is insufficient for min_confidence=5.

        Same vol_spike_mult=4.5 → vol_ratio≈3.83 satisfies base vol>1.5 (+1)
        but climax_enabled=False blocks the 5th point → confidence=4 → no signal.
        """
        close, btc, vol, taker = _make_confidence_data(vol_spike_mult=4.5)
        strat = self._make_strategy(climax_enabled=False, min_confidence=5)
        strat.update_market_data("TESTUSDT", vol, taker)

        sig = strat.on_bar("TESTUSDT", close, btc)
        assert sig == 0, f"Expected no signal without climax at min_confidence=5, got {sig}"

    def test_no_climax_without_vol_spike(self):
        """vol_ratio=1.0 (no spike) → climax condition not triggered."""
        # With min_confidence=5 and climax enabled but no vol spike,
        # confidence=4 → no signal
        close, btc, vol, taker = _make_confidence_data(vol_spike_mult=1.0)
        strat = self._make_strategy(climax_enabled=True, min_confidence=5)
        strat.update_market_data("TESTUSDT", vol, taker)

        sig = strat.on_bar("TESTUSDT", close, btc)
        assert sig == 0, f"Expected no signal without vol spike, got {sig}"


# ═══════════════════════ 4. RubberBandConfig SSoT Verification ═══════════════════════


class TestSettingsToRbConfig:
    """RubberBandConfig is SSoT for all RB params (v12.6+).

    settings.rb_* fields have been removed. These tests verify that
    RubberBandConfig() holds the correct v12.6 production values.
    """

    def test_v126_keys_present(self):
        """All 6 v12.6 keys must exist as attributes on RubberBandConfig."""
        from quantbox.strategy.rubberband.config import RubberBandConfig

        cfg = RubberBandConfig()
        required_attrs = {
            "climax_enabled",
            "climax_vol_mult",
            "atr_stop_enabled",
            "atr_stop_mult",
            "atr_stop_floor",
            "atr_stop_cap",
        }
        missing = required_attrs - set(vars(cfg).keys())
        assert not missing, f"Attributes missing from RubberBandConfig: {missing}"

    def test_macd_disabled_in_settings(self):
        from quantbox.strategy.rubberband.config import RubberBandConfig

        assert RubberBandConfig().macd_enabled is False

    def test_stoch_disabled_in_settings(self):
        from quantbox.strategy.rubberband.config import RubberBandConfig

        assert RubberBandConfig().stoch_enabled is False

    def test_climax_enabled_in_settings(self):
        from quantbox.strategy.rubberband.config import RubberBandConfig

        assert RubberBandConfig().climax_enabled is True

    def test_atr_stop_enabled_in_settings(self):
        from quantbox.strategy.rubberband.config import RubberBandConfig

        assert RubberBandConfig().atr_stop_enabled is True
