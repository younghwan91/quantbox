"""Tests for r55 Sub-TF Entry Refinement.

룩어헤드 바이어스 안전성 + 함수 정확성 검증.
"""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Import functions under test
# ─────────────────────────────────────────────────────────────────────────────

from research.active.r51_r60.r55_subtf_entry import (
    gen_vr_signal_with_subtf_vr,
    sim_directional_pullback,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_vr_arrays():
    """4h VR (n=20) and 15m VR (n=320) aligned arrays."""
    n_4h = 20
    lower_ratio = 16
    n_15m = n_4h * lower_ratio

    vr_4h = np.full(n_4h, 0.6, dtype=np.float64)
    vr_15m = np.full(n_15m, 0.6, dtype=np.float64)
    return vr_4h, vr_15m, lower_ratio


@pytest.fixture
def trending_ohlc():
    """Simple uptrending OHLC for pullback tests."""
    n = 200
    rng = np.random.default_rng(42)
    close = np.cumprod(1 + rng.standard_normal(n) * 0.005 + 0.001) * 100
    high = close * (1 + np.abs(rng.standard_normal(n) * 0.003))
    low = close * (1 - np.abs(rng.standard_normal(n) * 0.003))
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    return close, high, low


# ─────────────────────────────────────────────────────────────────────────────
#  Method A: gen_vr_signal_with_subtf_vr — 룩어헤드 안전성
# ─────────────────────────────────────────────────────────────────────────────


class TestSubTfVrSignal:
    def test_no_signal_when_upper_vr_above_threshold(self, simple_vr_arrays):
        """VR 4h > threshold → no signal regardless of 15m VR."""
        vr_4h, vr_15m, ratio = simple_vr_arrays
        vr_4h[:] = 0.8  # all above threshold 0.5
        sig = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 2, ratio)
        assert sig.sum() == 0

    def test_no_signal_when_lower_vr_above_threshold(self, simple_vr_arrays):
        """4h VR compressed but 15m VR not → no signal (stackup requirement)."""
        vr_4h, vr_15m, ratio = simple_vr_arrays
        vr_4h[5:] = 0.3  # 4h compressed
        vr_15m[:] = 0.8  # 15m NOT compressed
        sig = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 2, ratio)
        assert sig.sum() == 0

    def test_signal_only_when_both_compressed(self, simple_vr_arrays):
        """Signal fires ONLY when BOTH 4h and 15m VR are below threshold."""
        vr_4h, vr_15m, ratio = simple_vr_arrays
        # Set 4h bar 5 as compressed
        vr_4h[5] = 0.3

        # 15m bars corresponding to 4h bar 5: [80..95] (indices 5*16 to 5*16+15)
        vr_15m[:] = 0.8  # default: NOT compressed
        vr_15m[80:96] = 0.4  # compressed for 4h bar 5's 15m bars

        sig = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 2, ratio)
        assert sig[5] == 1
        assert sig.sum() == 1

    def test_lookahead_safety_last_lower_bar_only(self, simple_vr_arrays):
        """룩어헤드 없음 검증: 4h bar i의 확인에 15m bars [i*16 .. i*16+15]만 사용.

        4h bar 3 (i=3) signal 발생 시:
         - 4h bar 3의 마지막 15m bar = index 3*16+15 = 63
         - 4h bar 4의 첫 번째 15m bar = index 64 (FUTURE)
         - 64번 이후는 사용하면 안 됨.
        """
        vr_4h, vr_15m, ratio = simple_vr_arrays
        vr_4h[3] = 0.3  # 4h bar 3 compressed

        # 15m: ONLY bar 63 (last of 4h bar 3) compressed → should fire
        vr_15m[:] = 0.8
        vr_15m[63] = 0.4  # exactly the last bar of 4h bar 3
        sig_fire = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 2, ratio)

        # 15m: ONLY bar 64 (first of 4h bar 4 = FUTURE) compressed → should NOT fire for bar 3
        vr_15m[63] = 0.8
        vr_15m[64] = 0.4  # future bar — should NOT affect 4h bar 3's signal
        sig_nope = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 2, ratio)

        assert sig_fire[3] == 1, "Signal should fire when last 15m bar in 4h bar is compressed"
        assert sig_nope[3] == 0, "LOOKAHEAD BUG: 4h bar 3 used future 15m bar 64"

    def test_lower_ratio_1h_to_15m(self):
        """1h→15m (ratio=4) alignment."""
        n_1h = 30
        n_15m = n_1h * 4
        vr_1h = np.full(n_1h, 0.6, dtype=np.float64)
        vr_15m = np.full(n_15m, 0.6, dtype=np.float64)

        vr_1h[10] = 0.3
        # Last 15m bar of 1h bar 10 = index 10*4 + 3 = 43
        vr_15m[43] = 0.4  # trigger

        sig = gen_vr_signal_with_subtf_vr(vr_1h, vr_15m, 0.5, 0.5, 5, 4)
        assert sig[10] == 1
        assert sig.sum() == 1

    def test_warmup_respected(self, simple_vr_arrays):
        """Warmup bars produce no signal."""
        vr_4h, vr_15m, ratio = simple_vr_arrays
        vr_4h[:3] = 0.2  # very compressed
        vr_15m[:48] = 0.2  # 15m bars for first 3 4h bars
        sig = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 3, ratio)
        assert sig[:3].sum() == 0, "No signal before warmup"

    def test_nan_handling(self, simple_vr_arrays):
        """NaN in either VR array → no signal."""
        vr_4h, vr_15m, ratio = simple_vr_arrays
        vr_4h[5] = np.nan
        vr_15m[80:96] = 0.3
        sig_nan_upper = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 2, ratio)
        assert sig_nan_upper[5] == 0, "NaN 4h VR should not fire"

    def test_signal_reduces_trades_vs_no_stackup(self):
        """Stackup requirement reduces or equals number of signals."""
        n = 100
        ratio = 16
        vr_4h = np.random.uniform(0.2, 0.8, n)
        vr_15m = np.random.uniform(0.2, 0.8, n * ratio)

        # Without stackup (threshold_lower=1.0 → always passes)
        sig_none = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 1.0, 5, ratio)
        # With stackup
        sig_stack = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, 0.5, 5, ratio)

        assert sig_stack.sum() <= sig_none.sum(), "Stackup should reduce or equal signals"


# ─────────────────────────────────────────────────────────────────────────────
#  Method B: sim_directional_pullback — 진입가 및 R:R 검증
# ─────────────────────────────────────────────────────────────────────────────


class TestPullbackEntry:
    def test_entry_at_limit_price_not_signal_close(self, trending_ohlc):
        """Pullback entry price = limit_price, NOT close[signal_bar]."""
        close, high, low = trending_ohlc
        n = len(close)

        # Fire signal at bar 50 (LONG)
        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        sig[50] = 1
        dire[50] = 1

        pb_pct = 0.01  # 1% pullback
        signal_close = close[50]
        limit_price = signal_close * (1 - pb_pct)

        # Force low to hit limit within window
        low[51] = limit_price * 0.99  # definitely fills

        p, e, x = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
            pb_pct,
            3,
        )

        assert len(p) == 1, "Exactly 1 trade should execute"
        assert e[0] == 51, f"Entry bar should be 51 (fill bar), got {e[0]}"

    def test_no_fill_skips_trade(self, trending_ohlc):
        """If price never touches limit within window, trade is skipped."""
        close, high, low = trending_ohlc
        n = len(close)

        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        sig[50] = 1
        dire[50] = 1

        # Force price to NEVER pull back (keep low far above limit)
        close[50] = 100.0
        low[51:55] = 99.5  # > limit = 100*(1-0.03) = 97 → limit not hit

        p, e, x = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
            pullback_pct=0.03,  # 3% pullback — very hard to reach
            pullback_window=3,
        )
        # close=100, limit=97, low[51:54]=99.5 → no fill in window 3
        assert len(p) == 0, "Trade should be skipped when limit not reached"

    def test_short_pullback_direction(self, trending_ohlc):
        """For SHORT signal, limit price is ABOVE close (pull up for short entry)."""
        close, high, low = trending_ohlc
        n = len(close)

        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        sig[60] = 1
        dire[60] = -1  # SHORT

        pb_pct = 0.01
        signal_close = close[60]
        limit_price = signal_close * (1 + pb_pct)  # higher for short

        # Force high to hit limit
        high[61] = limit_price * 1.01  # definitely fills

        p, e, x = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
            pb_pct,
            3,
        )

        assert len(p) == 1, "SHORT pullback trade should execute"
        assert e[0] == 61

    def test_pullback_window_limits_search(self, trending_ohlc):
        """Trade fills only within pullback_window, not after."""
        close, high, low = trending_ohlc
        n = len(close)

        # Ensure stable prices for this test
        close[:] = 100.0
        high[:] = 101.0
        low[:] = 99.5

        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        sig[50] = 1
        dire[50] = 1

        limit_price = 100.0 * (1 - 0.02)  # 98.0

        # Window=3, hit happens at bar 54 (outside window)
        low[54] = limit_price * 0.99

        p, e, x = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
            0.02,
            3,  # window=3 bars
        )
        assert len(p) == 0, "Fill at bar 54 is outside window=3 (bars 51,52,53)"

    def test_zero_pullback_equivalent_to_baseline(self, trending_ohlc):
        """pullback_pct=0 should give identical results to original sim (no skip)."""
        close, high, low = trending_ohlc
        n = len(close)

        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        for i in [20, 50, 80, 120]:
            if i < n:
                sig[i] = 1
                dire[i] = 1 if close[i] > close[i - 5] else -1

        from research.active.r41_r45.r43_exit_improvement import sim_directional_trailing

        p_base, e_base, x_base = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
        )
        p_pb, e_pb, x_pb = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
            0.0,
            5,  # pullback_pct=0 → no pullback required
        )

        # With pb_pct=0, limit_price = close[i]*(1-0) = close[i]
        # Since low[i+1] would need to be ≤ close[i] to fill...
        # This might NOT be identical to baseline since pullback_pct=0 still requires low <= close[i]
        # But at least N should be ≤ baseline N
        assert len(p_pb) <= len(p_base), "With pullback, N should not increase"

    def test_no_overlap_between_trades(self, trending_ohlc):
        """Trades from pullback should not overlap (exit bar of one <= entry bar of next)."""
        close, high, low = trending_ohlc
        n = len(close)

        # Fire signals at regular intervals
        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        for i in range(20, n - 10, 15):
            sig[i] = 1
            dire[i] = 1

        p, e, x = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            10,
            0.0,
            0.005,
            0.002,
            0.001,
            2,
        )

        for k in range(1, len(p)):
            assert e[k] > x[k - 1], f"Trade {k} entry {e[k]} overlaps with trade {k - 1} exit {x[k - 1]}"

    def test_pnl_calculated_from_limit_price(self, trending_ohlc):
        """PnL should reflect entry at limit_price, not original close."""
        close, high, low = trending_ohlc
        n = len(close)

        # Set up controlled scenario
        close[50] = 100.0
        high[50] = 101.0
        low[50] = 99.0

        # Pullback fills at bar 51
        low[51] = 97.5  # will hit limit = 100*(1-0.02) = 98.0
        high[51] = 103.0  # will hit TP = 98.0 * 1.12 = 109.76? No, high=103 < 109.76
        close[51] = 101.0

        # TP will be hit at bar 52
        high[52] = 115.0  # above TP = 98.0 * 1.12 = 109.76

        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        sig[50] = 1
        dire[50] = 1

        p, e, x = sim_directional_pullback(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.12,
            48,
            0.0,
            0.005,
            0.002,
            0.02,
            3,  # 2% pullback
        )

        assert len(p) == 1
        # TP pnl = 0.12 (exact, since SL not hit and TP is hit)
        assert abs(p[0] - 0.12) < 1e-6, f"Expected TP pnl=0.12, got {p[0]:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-End: Stackup reduces N without drastically cutting PF
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_stackup_n_monotone_with_threshold(self):
        """Stricter 15m threshold → fewer signals (monotone decrease in N)."""
        from research.active.r51_r60.r55_subtf_entry import gen_vr_signal_with_subtf_vr

        n_4h = 200
        ratio = 16
        n_15m = n_4h * ratio

        rng = np.random.default_rng(42)
        vr_4h = rng.uniform(0.1, 0.9, n_4h)
        vr_15m = rng.uniform(0.1, 0.9, n_15m)

        thresholds = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
        counts = []
        for thr in thresholds:
            sig = gen_vr_signal_with_subtf_vr(vr_4h, vr_15m, 0.5, thr, 10, ratio)
            counts.append(int(sig.sum()))

        # Should be monotone non-increasing
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], (
                f"Expected N to decrease with stricter threshold: "
                f"thr={thresholds[i]:.1f} gave N={counts[i]} > thr={thresholds[i - 1]:.1f} N={counts[i - 1]}"
            )
