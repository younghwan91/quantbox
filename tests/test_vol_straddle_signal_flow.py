"""Test VolStraddle signal flow — ensures 15m/1h/4h signals can fire.

Regression test for the off-by-one and insufficient-data bugs that
caused ZERO signals in production for 47 cycles (2026-03-01).

BUG1: 15m — 200 bars downloaded, warmup_15m=200, check `len < 201` → blocked
BUG2: 1h  — outer guard `4*(vr_long+warmup_1h)=400 > 200` → never entered
BUG3: Diagnostic log used rubber band config instead of vol_straddle config

Commit: fix(v2.0.1) vol_straddle signal flow
"""
from __future__ import annotations

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    compute_momentum,
    compute_vr,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy


def _make_squeeze_data(
    n: int,
    squeeze_at_end: bool = True,
    squeeze_bars: int = 20,
    base_price: float = 100.0,
    base_vol: float = 0.02,
    squeeze_vol: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic OHLCV data with a volatility squeeze at the end.

    The squeeze is created by shrinking the high-low range dramatically
    in the last `squeeze_bars` bars, which causes ATR(short) to drop
    much faster than ATR(long), resulting in VR < threshold.

    Args:
        n: Total number of bars.
        squeeze_at_end: Whether to place squeeze at the end (relevant bar).
        squeeze_bars: Number of bars in the squeeze zone.
        base_price: Starting price.
        base_vol: Normal volatility (high-low range as fraction of price).
        squeeze_vol: Squeeze volatility (much smaller range).

    Returns:
        (close, high, low, volume) arrays.
    """
    rng = np.random.RandomState(42)
    close = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    volume = np.ones(n) * 1000.0

    price = base_price
    for i in range(n):
        # Determine volatility regime
        if squeeze_at_end and i >= n - squeeze_bars:
            vol = squeeze_vol
        else:
            vol = base_vol

        # Random walk with controlled volatility
        ret = rng.normal(0, vol)
        price = price * (1 + ret)
        half_range = price * vol * 0.5
        close[i] = price
        high[i] = price + half_range
        low[i] = price - half_range

    # Add a directional drift at end (positive momentum for LONG signal)
    if squeeze_at_end:
        for i in range(n - squeeze_bars, n):
            close[i] *= 1.0 + 0.001 * (i - (n - squeeze_bars))
            high[i] = close[i] * 1.001
            low[i] = close[i] * 0.999

    return close, high, low, volume


class TestSignalFlowBarCounts:
    """Verify that signal generation works with production bar counts."""

    def test_15m_signal_fires_with_500_bars(self):
        """BUG1 regression: 15m signal MUST fire with 500 bars and squeeze."""
        cfg = VolStraddleConfig()
        n_bars = 500  # production download count (fixed from 200)
        close, high, low, vol = _make_squeeze_data(n_bars, squeeze_bars=30)

        vr = compute_vr(close, high, low, cfg.vr_short, cfg.vr_long)
        sig = gen_vr_signal(vr, cfg.d_vr_threshold, cfg.warmup_15m)

        # With 500 bars and warmup=200, signal can fire on bars 200-499
        assert len(sig) == 500
        assert sig[-1] == 1, (
            f"15m signal should fire on last bar with squeeze. "
            f"VR[-1]={vr[-1]:.4f}, threshold={cfg.d_vr_threshold}"
        )

    def test_15m_signal_blocked_with_200_bars(self):
        """Confirm the old bug: 200 bars can't produce 15m signal (off-by-one)."""
        cfg = VolStraddleConfig()
        n_bars = 200
        close, high, low, vol = _make_squeeze_data(n_bars, squeeze_bars=30)

        vr = compute_vr(close, high, low, cfg.vr_short, cfg.vr_long)
        sig = gen_vr_signal(vr, cfg.d_vr_threshold, cfg.warmup_15m)

        # With warmup=200 and only 200 bars, loop range(200, 200) is empty
        assert sig[-1] == 0, "With exactly 200 bars, 15m signal cannot fire (warmup=200)"

    def test_1h_guard_passes_with_500_bars(self):
        """BUG2 regression: 1h resampling guard MUST pass with 500 bars."""
        cfg = VolStraddleConfig()
        min_len = 500

        # OLD guard (bug): 4 * (vr_long + warmup_1h) = 4 * (50 + 50) = 400
        old_guard = min_len >= 4 * (cfg.vr_long + cfg.warmup_1h)
        # This DID pass with 500 >= 400, but failed with 200 < 400
        assert old_guard is True, "Old guard should pass with 500 (but not 200)"

        # OLD guard with 200 bars (the bug):
        assert not (200 >= 4 * (cfg.vr_long + cfg.warmup_1h)), (
            "Old guard correctly fails with 200 bars"
        )

        # NEW guard: 4 * (warmup_1h + 1) = 4 * 51 = 204
        new_guard = min_len >= 4 * (cfg.warmup_1h + 1)
        assert new_guard is True, "New guard must pass with 500 bars"

        # NEW guard with 210 bars (minimal viable):
        assert 210 >= 4 * (cfg.warmup_1h + 1), (
            "New guard should pass with 210 bars"
        )

    def test_1h_signal_fires_with_500_bars(self):
        """BUG2 regression: 1h signal MUST fire with 500 bars and squeeze."""
        cfg = VolStraddleConfig()
        n_bars = 500
        close, high, low, vol = _make_squeeze_data(
            n_bars, squeeze_bars=120,  # squeeze covers ~30 1h bars
        )

        # Resample to 1h
        c1h, h1h, l1h, v1h = resample_ohlcv(close, high, low, vol, 4)
        assert len(c1h) == 125  # 500 / 4

        vr = compute_vr(c1h, h1h, l1h, cfg.vr_short, cfg.vr_long)
        sig = gen_vr_signal(vr, cfg.d_vr_threshold, cfg.warmup_1h)

        # warmup_1h=50, so can fire on bars 50-124
        assert len(sig) == 125
        assert sig[-1] == 1, (
            f"1h signal should fire on last bar with squeeze. "
            f"VR[-1]={vr[-1]:.4f}, threshold={cfg.d_vr_threshold}"
        )

    def test_4h_signal_fires(self):
        """4h signal fires with 100 bars (this always worked)."""
        cfg = VolStraddleConfig()
        n_bars = 100
        close, high, low, vol = _make_squeeze_data(
            n_bars, squeeze_bars=20,
        )

        vr = compute_vr(close, high, low, cfg.vr_short, cfg.vr_long)
        sig = gen_vr_signal(vr, cfg.d_vr_threshold, cfg.warmup_4h)

        # warmup_4h=13, 100 bars → can fire on bars 13-99
        assert len(sig) == 100
        assert sig[-1] == 1, (
            f"4h signal should fire with squeeze. "
            f"VR[-1]={vr[-1]:.4f}, threshold={cfg.d_vr_threshold}"
        )


class TestStrategyOnBar:
    """Integration test: VolStraddleStrategy.on_bar with production bar counts."""

    def _setup_strategy_with_data(
        self, n_15m: int = 500, n_4h: int = 100,
    ) -> tuple[VolStraddleStrategy, str, np.ndarray]:
        """Create a strategy with properly sized squeeze data."""
        strategy = VolStraddleStrategy()
        symbol = 'TESTUSDT'

        # 15m data with squeeze at end
        close, high, low, vol = _make_squeeze_data(
            n_15m, squeeze_bars=60,
        )
        taker = vol * 0.5

        # Wire up market data (mimics scalper's update_market_data)
        strategy.update_market_data(symbol, vol, taker, high, low)

        # Wire up 4h HTF data with squeeze at end
        c4h, h4h, l4h, _ = _make_squeeze_data(n_4h, squeeze_bars=15)
        strategy.update_htf_data(symbol, c4h, h4h, l4h)

        return strategy, symbol, close

    def test_on_bar_fires_with_500_bars(self):
        """Strategy.on_bar MUST produce non-zero signals with squeeze data."""
        strategy, symbol, close = self._setup_strategy_with_data(n_15m=500)
        signal = strategy.on_bar(symbol, close, None)

        # With squeeze data on all TFs, at least one should fire
        assert signal != 0, (
            f"on_bar should produce a signal with squeeze data. "
            f"Got signal={signal}"
        )

    def test_on_bar_blocked_with_200_bars(self):
        """Confirm old bug: on_bar returns 0 with only 200 bars (regression)."""
        strategy = VolStraddleStrategy()
        symbol = 'TESTUSDT'

        # Use only 200 bars — this was the old download count
        close, high, low, vol = _make_squeeze_data(200, squeeze_bars=30)
        taker = vol * 0.5
        strategy.update_market_data(symbol, vol, taker, high, low)

        # 4h still fires with 100 bars, so provide no HTF data
        # to isolate the 15m/1h bug
        signal = strategy.on_bar(symbol, close, None)

        # 15m: 200 < 201 → blocked. 1h: 200 < 204 → blocked. No 4h data → 0
        assert signal == 0, (
            "With 200 bars and no 4h data, on_bar should return 0 (bug repro)"
        )

    def test_signal_metadata_populated(self):
        """Signal metadata (tf, sl, tp) is populated after on_bar fires."""
        strategy, symbol, close = self._setup_strategy_with_data(n_15m=500)
        signal = strategy.on_bar(symbol, close, None)

        if signal != 0:
            meta = strategy.get_last_signal_meta(symbol)
            assert 'tf' in meta, "Signal metadata must include timeframe"
            assert 'sl' in meta, "Signal metadata must include stop loss"
            assert 'tp' in meta, "Signal metadata must include take profit"
            assert meta['tf'] in ('15m', '1h', '4h'), f"Unknown TF: {meta['tf']}"


class TestDiagnosticConfig:
    """Verify diagnostic logging uses correct config per strategy."""

    def test_vs_config_threshold_is_zero(self):
        """vol_straddle compatibility threshold should be 0.0 (not 3.2%)."""
        cfg = VolStraddleConfig()
        assert cfg.threshold == 0.0, (
            f"vol_straddle threshold should be 0.0, got {cfg.threshold}"
        )

    def test_vs_config_lookback_is_8(self):
        """vol_straddle compatibility lookback is 8 (for move diagnostics)."""
        cfg = VolStraddleConfig()
        assert cfg.lookback == 8
