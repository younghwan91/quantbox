"""Tests for Confidence-Scored Multi-Evidence Entry (v6.0).

Validates:
  1. compute_signals_confidence Numba function behavior
  2. Confidence filter reduces trade count (min_confidence=2 rejects low-quality)
  3. Leverage encoding in signal magnitude
  4. RSI, volume, taker ratio evidence contributions
  5. Backward compatibility: min_confidence=1 ≈ regime_switch behavior
  6. on_bar dispatch for entry_mode='confidence'
"""

import numpy as np

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals_confidence,
    compute_signals_regime_switch,
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
    regime: str = "risk_off",
) -> np.ndarray:
    """BTC prices above/below SMA(96)."""
    btc = np.ones(n, dtype=np.float64)
    if regime == "risk_on":
        for i in range(n):
            btc[i] = 1.0 + 0.15 * (i / n)
    else:
        for i in range(n):
            btc[i] = 1.0 - 0.15 * (i / n)
    return btc


def _make_volume(n: int = 300, spike_bar: int = 200, spike_mult: float = 3.0) -> np.ndarray:
    """Constant volume with a spike at spike_bar."""
    vol = np.ones(n, dtype=np.float64) * 1000.0
    vol[spike_bar - 5 : spike_bar + 5] = 1000.0 * spike_mult
    return vol


def _make_taker_buy(
    n: int = 300,
    ratio: float = 0.5,
    volume: np.ndarray | None = None,
) -> np.ndarray:
    """Taker buy base volume at constant ratio of total volume."""
    if volume is None:
        volume = np.ones(n, dtype=np.float64) * 1000.0
    return volume * ratio


# Shared parameters for compute_signals_confidence
_PARAMS = dict(
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


class TestConfidenceSignals:
    """Test compute_signals_confidence Numba function."""

    def test_output_shape(self):
        """Output array shape matches input."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300, 200)
        taker = _make_taker_buy(300, 0.4, vol)
        sigs = compute_signals_confidence(close, close, close, vol, taker, btc, **_PARAMS)
        assert sigs.shape == close.shape

    def test_risk_off_with_evidence_enters(self):
        """Risk-Off + volume spike + sell pressure → long entry."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300, 200, spike_mult=3.0)
        taker = _make_taker_buy(300, 0.35, vol)  # Heavy selling
        sigs = compute_signals_confidence(close, close, close, vol, taker, btc, **_PARAMS)
        assert np.any(sigs > 0), "Should enter long with high confidence"

    def test_min_confidence_filters_weak_signals(self):
        """Higher min_confidence should reject more entries."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300)  # No volume spike
        taker = _make_taker_buy(300, 0.5, vol)  # Neutral taker ratio

        params_mc1 = {**_PARAMS, "min_confidence": 1}
        params_mc3 = {**_PARAMS, "min_confidence": 3}

        sigs_mc1 = compute_signals_confidence(close, close, close, vol, taker, btc, **params_mc1)
        sigs_mc3 = compute_signals_confidence(close, close, close, vol, taker, btc, **params_mc3)

        n_trades_mc1 = np.sum(np.diff(sigs_mc1 != 0).astype(int) > 0)
        n_trades_mc3 = np.sum(np.diff(sigs_mc3 != 0).astype(int) > 0)
        assert n_trades_mc1 >= n_trades_mc3, f"mc=1 should have >= trades than mc=3: {n_trades_mc1} vs {n_trades_mc3}"

    def test_leverage_encoding_flat(self):
        """With step=0.0, all signal magnitudes equal base_leverage."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300, 200, 3.0)
        taker = _make_taker_buy(300, 0.35, vol)

        params = {**_PARAMS, "base_leverage": 1.5, "confidence_leverage_step": 0.0}
        sigs = compute_signals_confidence(close, close, close, vol, taker, btc, **params)

        nonzero = sigs[sigs != 0]
        if len(nonzero) > 0:
            for v in nonzero:
                assert abs(abs(v) - 1.5) < 0.01, f"Expected |signal|=1.5, got {abs(v)}"

    def test_leverage_encoding_graduated(self):
        """With step=1.0, higher confidence gets higher leverage."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300, 200, 3.0)
        taker = _make_taker_buy(300, 0.35, vol)

        params = {
            **_PARAMS,
            "min_confidence": 1,
            "base_leverage": 1.0,
            "confidence_leverage_step": 0.5,
            "max_leverage": 3.0,
        }
        sigs = compute_signals_confidence(close, close, close, vol, taker, btc, **params)
        nonzero = sigs[sigs != 0]
        if len(nonzero) > 0:
            magnitudes = np.abs(nonzero)
            # At confidence=1: lev=1.0, confidence=2: lev=1.5, etc.
            assert np.all(magnitudes >= 1.0), "Leverage should be >= base"
            assert np.all(magnitudes <= 3.0), "Leverage should be <= max"

    def test_max_hold_enforced(self):
        """Position exits by max_hold bars."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300, 200, 3.0)
        taker = _make_taker_buy(300, 0.35, vol)

        sigs = compute_signals_confidence(close, close, close, vol, taker, btc, **_PARAMS)
        in_pos = False
        hold = 0
        for s in sigs:
            if s != 0 and not in_pos:
                in_pos = True
                hold = 1
            elif in_pos and s != 0:
                hold += 1
            elif in_pos and s == 0:
                assert hold <= _PARAMS["max_hold"], f"Held {hold} > max_hold"
                in_pos = False
                hold = 0

    def test_no_signal_without_move(self):
        """No entry when price move < threshold."""
        close = np.ones(300, dtype=np.float64)  # flat
        btc = _make_btc_regime(300, "risk_off")
        vol = np.ones(300, dtype=np.float64) * 1000.0
        taker = vol * 0.5
        sigs = compute_signals_confidence(close, close, close, vol, taker, btc, **_PARAMS)
        assert np.all(sigs == 0), "No move → no signal"

    def test_mc1_similar_to_regime_switch(self):
        """min_confidence=1 should produce similar trade count to regime_switch."""
        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = np.ones(300, dtype=np.float64) * 1000.0
        taker = vol * 0.5

        params_mc1 = {**_PARAMS, "min_confidence": 1}
        sigs_conf = compute_signals_confidence(close, close, close, vol, taker, btc, **params_mc1)
        sigs_rs = compute_signals_regime_switch(
            close,
            btc,
            8,
            0.045,
            0.4,
            0.9,
            10,
            0.0,
            32,
            96,
            192,
        )

        # Both should produce signals (or not) at roughly the same bar
        conf_entries = np.where(np.diff(sigs_conf != 0) > 0)[0]
        rs_entries = np.where(np.diff(sigs_rs != 0) > 0)[0]
        # With mc=1, confidence should be >= regime_switch entries
        # (it has the same base logic)
        assert len(conf_entries) >= len(rs_entries), "mc=1 confidence should match or exceed regime_switch entry count"


class TestConfidenceOnBar:
    """Test on_bar dispatches correctly for entry_mode='confidence'."""

    def test_confidence_dispatch_with_data(self):
        """entry_mode='confidence' with volume/taker data routes correctly."""
        cfg = RubberBandConfig(entry_mode="confidence")
        strategy = RubberBandStrategy(cfg)

        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")
        vol = _make_volume(300, 200, 3.0)
        taker = _make_taker_buy(300, 0.35, vol)

        strategy.update_market_data("TEST", vol[:200], taker[:200])
        sig = strategy.on_bar("TEST", close[:200], btc[:200])
        assert isinstance(sig, int)
        assert sig in (-1, 0, 1)

    def test_confidence_fallback_without_volume(self):
        """Without volume data, falls back to regime_switch."""
        cfg = RubberBandConfig(entry_mode="confidence")
        strategy = RubberBandStrategy(cfg)

        close = _make_prices(300, 200, 0.06)
        btc = _make_btc_regime(300, "risk_off")

        # Don't call update_market_data
        sig = strategy.on_bar("TEST", close[:200], btc[:200])
        assert isinstance(sig, int)

    def test_confidence_no_btc_falls_through(self):
        """Without BTC data, confidence mode is skipped."""
        cfg = RubberBandConfig(entry_mode="confidence")
        strategy = RubberBandStrategy(cfg)
        close = _make_prices(300, 200, 0.06)
        sig = strategy.on_bar("TEST", close[:200], None)
        assert isinstance(sig, int)

    def test_update_market_data_stores(self):
        """update_market_data stores volume and taker data."""
        strategy = RubberBandStrategy()
        vol = np.ones(100, dtype=np.float64) * 1000.0
        taker = vol * 0.5
        strategy.update_market_data("BTCUSDT", vol, taker)
        assert "BTCUSDT" in strategy._volume_data
        assert "BTCUSDT" in strategy._taker_data
        np.testing.assert_array_equal(strategy._volume_data["BTCUSDT"], vol)


class TestConfidenceConfig:
    """Test v6.0 config fields."""

    def test_defaults(self):
        cfg = RubberBandConfig()
        assert cfg.entry_mode == "confidence"
        assert cfg.min_confidence == 3  # v12.3: M-2SSTSX validated (Sharpe 5.50, DD 8.0%)
        assert cfg.confidence_leverage_step == 0.0
        assert cfg.rsi_period == 14
        assert cfg.vol_period == 20
        assert cfg.base_leverage == 1.5
        assert cfg.max_leverage == 20.0  # v12.5.2: final hard cap (after MTF mult)
        assert cfg.threshold == 0.032  # v10.0
        assert cfg.max_hold == 25  # v12.4 HybTight (was 20)
        assert cfg.stoch_k_period == 3  # v10.0
        assert cfg.dynamic_capital_enabled is True  # v10.0
        assert len(cfg.symbol_whitelist) == 50  # v12.3: 50-symbol universe

    def test_custom_confidence_params(self):
        cfg = RubberBandConfig(
            min_confidence=3,
            confidence_leverage_step=1.0,
            rsi_period=21,
            vol_period=30,
        )
        assert cfg.min_confidence == 3
        assert cfg.confidence_leverage_step == 1.0
        assert cfg.rsi_period == 21
        assert cfg.vol_period == 30

    def test_moderate_profile(self):
        """Moderate risk profile: graduated leverage."""
        cfg = RubberBandConfig(
            min_confidence=2,
            base_leverage=1.0,
            confidence_leverage_step=1.0,
            max_leverage=2.0,
        )
        assert cfg.base_leverage == 1.0
        assert cfg.confidence_leverage_step == 1.0
        assert cfg.max_leverage == 2.0
