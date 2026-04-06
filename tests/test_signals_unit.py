"""Unit tests for quantbox/strategy/vol_straddle/signals.py.

Tests every public function with explicit numeric assertions.
All tests are deterministic — no randomness, no I/O.

Coverage target: signals.py 14% → ≥80%.
"""

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.signals import (
    check_ma_cluster,
    compute_momentum,
    compute_squeeze_dur,
    compute_sma,
    compute_vr,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _flat_ohlcv(n: int, price: float = 100.0, spread: float = 0.001):
    """Return flat OHLCV arrays (no trend, tiny range → very low VR)."""
    close = np.full(n, price, dtype=float)
    high = close * (1 + spread)
    low = close * (1 - spread)
    volume = np.ones(n, dtype=float)
    return close, high, low, volume


def _volatile_ohlcv(n: int, price: float = 100.0, spread: float = 0.05):
    """Return OHLCV with large candle range → high VR."""
    close = np.full(n, price, dtype=float)
    high = close * (1 + spread)
    low = close * (1 - spread)
    volume = np.ones(n, dtype=float)
    return close, high, low, volume


# ══════════════════════════════════════════════════════════════════════════════
#  compute_vr
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeVR:
    """Volatility Ratio = ATR(short) / ATR(long)."""

    def test_output_length_matches_input(self):
        close, high, low, _ = _flat_ohlcv(200)
        vr = compute_vr(close, high, low, short_period=8, long_period=56)
        assert len(vr) == 200

    def test_warmup_bars_are_nan(self):
        close, high, low, _ = _flat_ohlcv(200)
        vr = compute_vr(close, high, low, short_period=8, long_period=56)
        # Before long_period-1, VR should be NaN
        assert np.all(np.isnan(vr[:55]))

    def test_post_warmup_bars_are_not_nan(self):
        close, high, low, _ = _flat_ohlcv(200)
        vr = compute_vr(close, high, low, short_period=8, long_period=56)
        assert not np.any(np.isnan(vr[56:]))

    def test_vr_is_positive(self):
        close, high, low, _ = _flat_ohlcv(200)
        vr = compute_vr(close, high, low, short_period=8, long_period=56)
        valid = vr[~np.isnan(vr)]
        assert np.all(valid > 0)

    def test_low_spread_gives_vr_near_one(self):
        """Flat price with tiny uniform spread → short ATR ≈ long ATR → VR ≈ 1."""
        close, high, low, _ = _flat_ohlcv(300, spread=0.001)
        vr = compute_vr(close, high, low, short_period=8, long_period=56)
        valid = vr[~np.isnan(vr)]
        # Uniform candles → VR converges to ~1.0
        assert np.all(np.abs(valid - 1.0) < 0.05)

    def test_too_short_array_all_nan(self):
        close, high, low, _ = _flat_ohlcv(10)
        vr = compute_vr(close, high, low, short_period=8, long_period=56)
        assert np.all(np.isnan(vr))

    def test_volatile_then_flat_vr_drops(self):
        """High-spread candles followed by flat → VR should decrease after flat starts."""
        n = 300
        close = np.full(n, 100.0)
        high_arr = np.where(np.arange(n) < 150, close * 1.05, close * 1.001)
        low_arr = np.where(np.arange(n) < 150, close * 0.95, close * 0.999)
        vr = compute_vr(close, high_arr, low_arr, short_period=8, long_period=56)
        # VR in last section should be lower than in first section
        vr_early = np.nanmean(vr[100:150])
        vr_late = np.nanmean(vr[250:300])
        assert vr_late < vr_early


# ══════════════════════════════════════════════════════════════════════════════
#  compute_momentum
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeMomentum:
    def test_output_length_matches_input(self):
        close = np.linspace(100, 110, 50)
        mom = compute_momentum(close, lookback=8)
        assert len(mom) == 50

    def test_warmup_bars_are_zero(self):
        close = np.linspace(100, 110, 50)
        mom = compute_momentum(close, lookback=8)
        assert np.all(mom[:8] == 0.0)

    def test_uptrend_gives_positive_momentum(self):
        close = np.linspace(100, 200, 100)  # strict uptrend
        mom = compute_momentum(close, lookback=8)
        assert np.all(mom[8:] > 0)

    def test_downtrend_gives_negative_momentum(self):
        close = np.linspace(200, 100, 100)  # strict downtrend
        mom = compute_momentum(close, lookback=8)
        assert np.all(mom[8:] < 0)

    def test_flat_gives_zero_momentum(self):
        close = np.full(50, 100.0)
        mom = compute_momentum(close, lookback=8)
        assert np.allclose(mom, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  compute_squeeze_dur
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSqueezeDur:
    def test_output_length_matches_input(self):
        vr = np.full(100, 0.4)
        dur = compute_squeeze_dur(vr, threshold=0.5)
        assert len(dur) == 100

    def test_all_below_threshold_counts_up(self):
        vr = np.full(10, 0.3)
        dur = compute_squeeze_dur(vr, threshold=0.5)
        expected = np.arange(1, 11, dtype=np.int64)
        np.testing.assert_array_equal(dur, expected)

    def test_reset_after_break(self):
        vr = np.array([0.3, 0.3, 0.3, 0.8, 0.3, 0.3])  # break at index 3
        dur = compute_squeeze_dur(vr, threshold=0.5)
        assert dur[2] == 3  # 3 consecutive
        assert dur[3] == 0  # reset
        assert dur[5] == 2  # restarted

    def test_nan_resets_counter(self):
        vr = np.array([0.3, 0.3, np.nan, 0.3, 0.3])
        dur = compute_squeeze_dur(vr, threshold=0.5)
        assert dur[1] == 2
        assert dur[2] == 0  # NaN resets
        assert dur[4] == 2

    def test_all_above_threshold_all_zeros(self):
        vr = np.full(10, 0.9)
        dur = compute_squeeze_dur(vr, threshold=0.5)
        assert np.all(dur == 0)


# ══════════════════════════════════════════════════════════════════════════════
#  gen_vr_signal
# ══════════════════════════════════════════════════════════════════════════════


class TestGenVRSignal:
    def test_output_length_matches_input(self):
        vr = np.full(100, 0.4)
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert len(sig) == 100

    def test_below_threshold_after_warmup_is_one(self):
        vr = np.full(200, 0.3)  # all below 0.5
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert np.all(sig[56:] == 1)

    def test_warmup_bars_are_zero(self):
        vr = np.full(200, 0.3)
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert np.all(sig[:56] == 0)

    def test_above_threshold_gives_zero(self):
        vr = np.full(200, 0.8)  # all above 0.5
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert np.all(sig == 0)

    def test_nan_gives_zero(self):
        vr = np.full(200, np.nan)
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert np.all(sig == 0)

    def test_threshold_boundary(self):
        """VR exactly at threshold → no signal (strict <)."""
        vr = np.full(200, 0.5)
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert np.all(sig == 0)

    def test_mixed_values(self):
        vr = np.full(200, 0.8)
        vr[100:150] = 0.3  # squeeze window
        sig = gen_vr_signal(vr, threshold=0.5, warmup=56)
        assert np.all(sig[:100] == 0)
        assert np.all(sig[100:150] == 1)
        assert np.all(sig[150:] == 0)


# ══════════════════════════════════════════════════════════════════════════════
#  make_direction_momentum
# ══════════════════════════════════════════════════════════════════════════════


class TestMakeDirectionMomentum:
    def test_positive_momentum_gives_long(self):
        sig = np.ones(10, dtype=np.int8)
        mom = np.full(10, 0.01)  # positive
        dire = make_direction_momentum(sig, mom)
        assert np.all(dire == 1)

    def test_negative_momentum_gives_short(self):
        sig = np.ones(10, dtype=np.int8)
        mom = np.full(10, -0.01)  # negative
        dire = make_direction_momentum(sig, mom)
        assert np.all(dire == -1)

    def test_near_zero_momentum_defaults_long(self):
        """|momentum| < 0.0001 → default LONG (+1)."""
        sig = np.ones(10, dtype=np.int8)
        mom = np.full(10, 0.00005)  # near-zero
        dire = make_direction_momentum(sig, mom)
        assert np.all(dire == 1)

    def test_negative_near_zero_defaults_long(self):
        sig = np.ones(10, dtype=np.int8)
        mom = np.full(10, -0.00005)  # near-zero negative
        dire = make_direction_momentum(sig, mom)
        assert np.all(dire == 1)

    def test_no_signal_gives_zero_direction(self):
        sig = np.zeros(10, dtype=np.int8)
        mom = np.full(10, 0.05)
        dire = make_direction_momentum(sig, mom)
        assert np.all(dire == 0)

    def test_mixed_signal_and_momentum(self):
        sig = np.array([1, 0, 1, 1, 0], dtype=np.int8)
        mom = np.array([0.01, 0.05, -0.01, 0.0, -0.02])
        dire = make_direction_momentum(sig, mom)
        assert dire[0] == 1  # sig=1, mom>0 → LONG
        assert dire[1] == 0  # sig=0 → 0
        assert dire[2] == -1  # sig=1, mom<0 → SHORT
        assert dire[3] == 1  # sig=1, mom near-zero → default LONG
        assert dire[4] == 0  # sig=0 → 0

    def test_output_dtype_is_int8(self):
        sig = np.ones(5, dtype=np.int8)
        mom = np.full(5, 0.01)
        dire = make_direction_momentum(sig, mom)
        assert dire.dtype == np.int8


# ══════════════════════════════════════════════════════════════════════════════
#  compute_sma
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSMA:
    def test_output_length_matches_input(self):
        close = np.linspace(100, 200, 50)
        sma = compute_sma(close, period=20)
        assert len(sma) == 50

    def test_warmup_bars_are_nan(self):
        close = np.linspace(100, 200, 50)
        sma = compute_sma(close, period=20)
        assert np.all(np.isnan(sma[:19]))

    def test_first_valid_bar_is_mean_of_first_n(self):
        close = np.arange(1.0, 11.0)  # [1,2,...,10]
        sma = compute_sma(close, period=5)
        assert sma[4] == pytest.approx(3.0)  # mean(1,2,3,4,5)

    def test_flat_price_sma_equals_price(self):
        close = np.full(50, 42.0)
        sma = compute_sma(close, period=10)
        np.testing.assert_allclose(sma[9:], 42.0)

    def test_too_short_returns_all_nan(self):
        close = np.array([1.0, 2.0, 3.0])
        sma = compute_sma(close, period=10)
        assert np.all(np.isnan(sma))


# ══════════════════════════════════════════════════════════════════════════════
#  check_ma_cluster
# ══════════════════════════════════════════════════════════════════════════════


def _build_ma_cluster_inputs(n: int = 250, price: float = 100.0, spread: float = 0.005):
    """Build arrays for check_ma_cluster with all SMAs converged."""
    close = np.full(n, price, dtype=float)
    # SMAs at same price → spread = 0
    sma20 = np.full(n, price)
    sma50 = np.full(n, price)
    sma100 = np.full(n, price)
    sma200 = np.full(n, price)
    # Small positive momentum
    momentum = np.full(n, 0.015)
    in_cooldown = np.zeros(1, dtype=np.int64)
    duration_out = np.zeros(1, dtype=np.int64)
    return close, sma20, sma50, sma100, sma200, momentum, in_cooldown, duration_out


class TestCheckMACluster:
    def test_no_signal_when_sma200_is_nan(self):
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs()
        sma200[-1] = np.nan
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 4, cd, dur)
        assert result == 0

    def test_no_signal_when_duration_below_min(self):
        """Cluster duration < min_duration → no signal."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        # Break cluster for all bars except last 2 (spread > cluster_pct for bars 0..n-3)
        for i in range(n - 2):
            sma20[i] = price * 1.04  # spread > 0.03 → not in cluster
        # Last 2 bars: SMAs converged → small spread (in cluster)
        for i in range(n - 2, n):
            sma20[i] = price * 1.001
        # Close above band → breakout condition met
        close[-1] = price * 1.005
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 4, cd, dur)
        assert result == 0  # only 2 bars duration < min_duration=4

    def test_no_signal_when_close_inside_band(self):
        """Close inside MA band → no breakout → 0."""
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs()
        # Close exactly at price (inside band)
        close[-1] = 100.0  # same as SMAs → inside band
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur)
        assert result == 0

    def test_long_signal_on_upside_breakout(self):
        """Close above MA band with positive momentum → +1."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        # SMAs slightly spread but within cluster_pct
        sma20[:] = price * 1.005
        sma50[:] = price * 1.003
        sma100[:] = price * 1.001
        sma200[:] = price
        # Close above cluster band
        close[-1] = price * 1.010  # above max(SMA) = sma20 = 100.5
        mom[-1] = 0.02  # positive
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur)
        assert result == 1

    def test_short_signal_on_downside_breakout(self):
        """Close below MA band with negative momentum → -1."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        sma20[:] = price * 1.005
        sma50[:] = price * 1.003
        sma100[:] = price * 1.001
        sma200[:] = price
        # Close below cluster band
        close[-1] = price * 0.988  # below min(SMA) = sma200 = 100.0
        mom[-1] = -0.02  # negative
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur)
        assert result == -1

    def test_cooldown_prevents_double_signal(self):
        """Once fired, cooldown=1 → subsequent calls return 0."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        sma20[:] = price * 1.005
        sma50[:] = price * 1.003
        sma100[:] = price * 1.001
        sma200[:] = price
        close[-1] = price * 1.010
        mom[-1] = 0.02
        # First call: fires signal, sets cooldown
        r1 = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur)
        assert r1 == 1
        assert cd[0] == 1
        # Second call: cooldown active → 0
        r2 = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur)
        assert r2 == 0

    def test_cooldown_resets_when_cluster_dissolves(self):
        """When cluster breaks (spread > cluster_pct), cooldown resets."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        cd[0] = 1  # pre-set cooldown

        # Make last bar non-cluster (large spread)
        sma20[-1] = price * 1.10  # huge spread
        sma200[-1] = price
        close[-1] = price * 1.05  # inside band, so no new signal anyway
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur)
        assert result == 0
        assert cd[0] == 0  # cooldown reset

    def test_low_momentum_gives_no_signal(self):
        """|momentum| < min_momentum → 0."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        sma20[:] = price * 1.005
        sma50[:] = price * 1.003
        sma100[:] = price * 1.001
        sma200[:] = price
        close[-1] = price * 1.010
        mom[-1] = 0.001  # below min_momentum=0.003 default
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 8, cd, dur, min_momentum=0.003)
        assert result == 0

    def test_max_duration_zombie_filter(self):
        """Cluster duration > max_duration → zombie filter → 0."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        sma20[:] = price * 1.005
        sma50[:] = price * 1.003
        sma100[:] = price * 1.001
        sma200[:] = price
        close[-1] = price * 1.010
        mom[-1] = 0.02
        # max_duration=5 but cluster spans all 250 bars → zombie
        result = check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 4, cd, dur, max_duration=5)
        assert result == 0

    def test_duration_out_populated(self):
        """duration_out[0] should reflect actual cluster length."""
        n = 250
        price = 100.0
        close, sma20, sma50, sma100, sma200, mom, cd, dur = _build_ma_cluster_inputs(n, price)
        check_ma_cluster(close, sma20, sma50, sma100, sma200, mom, 0.03, 4, cd, dur)
        assert dur[0] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  resample_ohlcv
# ══════════════════════════════════════════════════════════════════════════════


class TestResampleOHLCV:
    def test_output_length_is_n_div_ratio(self):
        n = 160  # 160 15m bars = 40 1h bars (ratio=4)
        close, high, low, volume = _flat_ohlcv(n)
        c, h, lo, v = resample_ohlcv(close, high, low, volume, ratio=4)
        assert len(c) == 40

    def test_drops_incomplete_last_bar(self):
        """161 bars / ratio=4 → 40 complete groups + 1 leftover → 40 bars."""
        close, high, low, volume = _flat_ohlcv(161)
        c, h, lo, v = resample_ohlcv(close, high, low, volume, ratio=4)
        assert len(c) == 40

    def test_close_is_last_bar_of_group(self):
        """Resampled close[0] = last of first group = close[ratio-1]."""
        n = 80
        close = np.arange(float(n))
        high = close + 1
        low = close - 1
        vol = np.ones(n)
        c, *_ = resample_ohlcv(close, high, low, vol, ratio=4)
        assert c[0] == pytest.approx(3.0)  # close[3] = last of group 0
        assert c[1] == pytest.approx(7.0)  # close[7]

    def test_high_is_max_of_group(self):
        n = 80
        close = np.ones(n) * 100.0
        high = np.arange(float(n))  # increasing highs
        low = np.zeros(n)
        vol = np.ones(n)
        _, h, *_ = resample_ohlcv(close, high, low, vol, ratio=4)
        assert h[0] == pytest.approx(3.0)  # max(0,1,2,3)

    def test_low_is_min_of_group(self):
        n = 80
        close = np.ones(n) * 100.0
        high = np.ones(n) * 200.0
        low = np.arange(float(n))  # increasing lows
        vol = np.ones(n)
        _, _, lo, _ = resample_ohlcv(close, high, low, vol, ratio=4)
        assert lo[0] == pytest.approx(0.0)  # min(0,1,2,3)

    def test_volume_is_sum_of_group(self):
        n = 80
        close = np.ones(n) * 100.0
        high = close + 1
        low = close - 1
        vol = np.ones(n) * 10.0
        _, _, _, v = resample_ohlcv(close, high, low, vol, ratio=4)
        assert v[0] == pytest.approx(40.0)  # 4 × 10

    def test_ratio_16_15m_to_4h(self):
        """Standard 15m→4h resampling: 320 bars → 20 4h bars."""
        close, high, low, vol = _flat_ohlcv(320)
        c, h, lo, v = resample_ohlcv(close, high, low, vol, ratio=16)
        assert len(c) == 20
