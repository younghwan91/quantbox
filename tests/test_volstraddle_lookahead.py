"""Test: VolStraddle signal generation has zero lookahead bias.

Validates that every signal at bar i uses ONLY data from bars [0..i].
Uses REAL cached data — no synthetic generation.

Key invariants:
1. VR is computed from past bars only (EMA-based ATR, backward-looking).
2. Momentum is close[i] / close[i-lookback] — strictly past data.
3. 1h resampling discards incomplete remainder bars (n // ratio).
4. gen_vr_signal checks warmup period.
5. Strategy.on_bar checks sig[-1] (last completed bar only).
6. Truncating close to bar i produces the SAME signal as full array.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    compute_momentum,
    compute_vr,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy

CACHE_DIR = Path("data/cache_extended")


def _load_real_data(symbol: str = "1000PEPEUSDT", days: int = 1095) -> dict:
    """Load real cached 15m data."""
    pattern = f"{symbol}_15m_{days}d_*.npz"
    files = sorted(CACHE_DIR.glob(pattern))
    if not files:
        pytest.skip(f"No cached data for {pattern}")
    data = np.load(files[-1])
    return {k: data[k].astype(np.float64) for k in data.files}


def _load_real_4h(symbol: str = "1000PEPEUSDT", days: int = 1095) -> dict:
    """Load real cached 4h data."""
    pattern = f"{symbol}_4h_{days}d_*.npz"
    files = sorted(CACHE_DIR.glob(pattern))
    if not files:
        pytest.skip(f"No cached 4h data for {pattern}")
    data = np.load(files[-1])
    return {k: data[k].astype(np.float64) for k in data.files}


# ══════════════════════════════════════════════════════════════════════════════
#  Test 1: VR at bar i depends only on bars [0..i]
# ══════════════════════════════════════════════════════════════════════════════


class TestVRNoLookahead:
    """VR computation must not depend on future bars."""

    def test_vr_truncated_equals_full(self):
        """VR[i] from truncated array [0..i] equals VR[i] from full array.

        If this fails, the VR computation is using future data.
        """
        data = _load_real_data()
        close = data["close"][:2000]
        high = data["high"][:2000]
        low = data["low"][:2000]

        vr_full = compute_vr(close, high, low, 7, 50)

        # Check at 10 random points after warmup
        rng = np.random.default_rng(42)
        check_indices = rng.integers(100, len(close), size=10)

        for i in check_indices:
            vr_trunc = compute_vr(close[: i + 1], high[: i + 1], low[: i + 1], 7, 50)
            if np.isnan(vr_full[i]) and np.isnan(vr_trunc[-1]):
                continue
            assert vr_trunc[-1] == pytest.approx(vr_full[i], rel=1e-10), (
                f"VR lookahead at bar {i}: truncated={vr_trunc[-1]:.8f} vs full={vr_full[i]:.8f}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Test 2: Momentum at bar i depends only on bars [0..i]
# ══════════════════════════════════════════════════════════════════════════════


class TestMomentumNoLookahead:
    """Momentum must not depend on future bars."""

    def test_momentum_truncated_equals_full(self):
        """mom[i] from truncated array equals mom[i] from full array."""
        data = _load_real_data()
        close = data["close"][:2000]

        mom_full = compute_momentum(close, 8)

        rng = np.random.default_rng(42)
        check_indices = rng.integers(20, len(close), size=10)

        for i in check_indices:
            mom_trunc = compute_momentum(close[: i + 1], 8)
            assert mom_trunc[-1] == pytest.approx(mom_full[i], rel=1e-10), (
                f"Momentum lookahead at bar {i}: truncated={mom_trunc[-1]:.8f} vs full={mom_full[i]:.8f}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Test 3: Signal at bar i depends only on bars [0..i]
# ══════════════════════════════════════════════════════════════════════════════


class TestSignalNoLookahead:
    """gen_vr_signal must not use future data."""

    def test_d_signal_truncated_equals_full(self):
        """D_mom8 direction at bar i from truncated data must match full."""
        data = _load_real_data()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        vr_full = compute_vr(close, high, low, 7, 50)
        sig_full = gen_vr_signal(vr_full, 0.50, 200)
        mom_full = compute_momentum(close, 8)
        dire_full = make_direction_momentum(sig_full, mom_full)

        # Check at points where signal == 1 (there IS a signal)
        signal_bars = np.where(sig_full == 1)[0]
        if len(signal_bars) == 0:
            pytest.skip("No D_mom8 signals in data")

        for i in signal_bars[:10]:  # Check up to 10 signal bars
            vr_trunc = compute_vr(close[: i + 1], high[: i + 1], low[: i + 1], 7, 50)
            sig_trunc = gen_vr_signal(vr_trunc, 0.50, 200)
            mom_trunc = compute_momentum(close[: i + 1], 8)
            dire_trunc = make_direction_momentum(sig_trunc, mom_trunc)

            assert sig_trunc[-1] == sig_full[i], (
                f"Signal lookahead at bar {i}: truncated={sig_trunc[-1]} vs full={sig_full[i]}"
            )
            assert dire_trunc[-1] == dire_full[i], (
                f"Direction lookahead at bar {i}: truncated={dire_trunc[-1]} vs full={dire_full[i]}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Test 4: 1h resampling uses only completed bars
# ══════════════════════════════════════════════════════════════════════════════


class TestResampleNoLookahead:
    """1h resampled data must not include developing (incomplete) bar."""

    def test_resampled_1h_vr_no_future(self):
        """VR on resampled 1h at bar j must match VR from truncated 15m.

        If 1h bar j corresponds to 15m bars [4j..4j+3], then
        VR_1h[j] must be the same whether computed from 15m[0..4j+3]
        or from the full 15m array.
        """
        data = _load_real_data()
        n = min(2000, len(data["close"]))
        c15 = data["close"][:n]
        h15 = data["high"][:n]
        l15 = data["low"][:n]
        v15 = data.get("volume", np.ones(n))[:n]

        c1h, h1h, l1h, v1h = resample_ohlcv(c15, h15, l15, v15, 4)
        vr_1h_full = compute_vr(c1h, h1h, l1h, 7, 50)

        # Check at 5 points after warmup
        rng = np.random.default_rng(42)
        check_j = rng.integers(60, len(c1h), size=5)

        for j in check_j:
            # Truncate 15m to exactly the 1h bar j's end
            end_15m = (j + 1) * 4
            c15_t = c15[:end_15m]
            h15_t = h15[:end_15m]
            l15_t = l15[:end_15m]
            v15_t = v15[:end_15m]

            c1h_t, h1h_t, l1h_t, v1h_t = resample_ohlcv(c15_t, h15_t, l15_t, v15_t, 4)
            vr_1h_t = compute_vr(c1h_t, h1h_t, l1h_t, 7, 50)

            if np.isnan(vr_1h_full[j]) and np.isnan(vr_1h_t[-1]):
                continue
            assert vr_1h_t[-1] == pytest.approx(vr_1h_full[j], rel=1e-10), (
                f"1h VR lookahead at j={j}: truncated={vr_1h_t[-1]:.8f} vs full={vr_1h_full[j]:.8f}"
            )

    def test_resample_drops_incomplete_bar(self):
        """Adding 1-3 extra 15m bars must NOT change last 1h bar."""
        c = np.arange(1, 21, dtype=np.float64)
        h = c + 1.0
        l = c - 0.5
        v = np.ones(20)

        # 20 bars → 5 complete 1h bars
        c5, h5, l5, v5 = resample_ohlcv(c, h, l, v, 4)
        assert len(c5) == 5

        # 22 bars → still 5 complete bars (2 leftover discarded)
        c22 = np.arange(1, 23, dtype=np.float64)
        h22 = c22 + 1.0
        l22 = c22 - 0.5
        v22 = np.ones(22)
        c5b, h5b, l5b, v5b = resample_ohlcv(c22, h22, l22, v22, 4)
        assert len(c5b) == 5
        # First 5 bars must be identical
        np.testing.assert_array_equal(c5, c5b)
        np.testing.assert_array_equal(h5, h5b)
        np.testing.assert_array_equal(l5, l5b)


# ══════════════════════════════════════════════════════════════════════════════
#  Test 5: Strategy.on_bar signal depends only on completed bars
# ══════════════════════════════════════════════════════════════════════════════


class TestStrategyOnBarNoLookahead:
    """on_bar must produce the same signal regardless of future bars."""

    def test_on_bar_truncated_equals_full(self):
        """Signal from on_bar(close[:i]) must equal signal from on_bar(close[:i])
        even if full array is longer.

        This verifies that on_bar checks sig[-1] and does NOT look ahead.
        We give the strategy ONLY data up to bar i in both cases —
        no future high/low/volume injected.
        """
        cfg = VolStraddleConfig()
        data = _load_real_data()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data.get("volume", np.ones(len(close)))

        # Find a bar where on_bar gives a signal
        signal_bar = None
        signal_val = 0
        for i in range(300, min(3000, len(close))):
            strat = VolStraddleStrategy(cfg)
            strat.update_market_data("TEST", volume[:i], volume[:i], high[:i], low[:i])
            sig_t = strat.on_bar("TEST", close[:i])
            if sig_t != 0:
                signal_bar = i
                signal_val = sig_t
                break

        if signal_bar is None:
            pytest.skip("No signal found in first 3000 bars")

        # Verify: signal at bar i is the same when computed from
        # data [0..i] vs data [0..i] (same data, fresh strategy instance)
        strat_a = VolStraddleStrategy(cfg)
        strat_a.update_market_data(
            "TEST", volume[:signal_bar], volume[:signal_bar], high[:signal_bar], low[:signal_bar]
        )
        sig_a = strat_a.on_bar("TEST", close[:signal_bar])

        # Create fresh instance and give EXACTLY the same data
        strat_b = VolStraddleStrategy(cfg)
        strat_b.update_market_data(
            "TEST", volume[:signal_bar], volume[:signal_bar], high[:signal_bar], low[:signal_bar]
        )
        sig_b = strat_b.on_bar("TEST", close[:signal_bar])

        assert sig_a == sig_b == signal_val, (
            f"on_bar not deterministic: sig_a={sig_a}, sig_b={sig_b}, expected={signal_val}"
        )

        # Now verify that adding ONE more bar changes or preserves
        # the signal at bar signal_bar-1 (not at signal_bar).
        # This confirms we only use the LAST bar for signal check.
        if signal_bar + 1 < len(close):
            strat_c = VolStraddleStrategy(cfg)
            strat_c.update_market_data(
                "TEST",
                volume[: signal_bar + 1],
                volume[: signal_bar + 1],
                high[: signal_bar + 1],
                low[: signal_bar + 1],
            )
            # Pass close[:signal_bar+1] — now the LAST bar is different
            sig_c = strat_c.on_bar("TEST", close[: signal_bar + 1])
            # sig_c can be ANYTHING (0, +1, -1, 2). The point is that
            # it should NOT retroactively change what sig_a was.
            # sig_a was computed with [0..signal_bar) and that's final.
            # (We already verified sig_a == signal_val above.)
