"""Numba-compiled signal generators for VolStraddle strategy.

Ported from research/active/mtf_vol_straddle_r31.py — same logic,
zero lookahead.  All functions operate on COMPLETED bars only.

VR = ATR(short) / ATR(long) — measuring relative volatility compression.
When VR drops below a threshold, a squeeze is detected and entry signals fire.
"""
from __future__ import annotations

import numba
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  ATR & VR (Volatility Ratio)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _atr_pct(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int,
) -> np.ndarray:
    """EMA-based ATR expressed as percentage of close price.

    Args:
        close: Close prices.
        high: High prices.
        low: Low prices.
        period: ATR computation period.

    Returns:
        Array of ATR% values (same length as input), NaN before warmup.
    """
    n = len(close)
    out = np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = (high[0] - low[0]) / close[0] if close[0] > 0 else 0.0
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        mx = max(hl, max(hc, lc))
        tr[i] = mx / close[i] if close[i] > 0 else 0.0
    if n >= period:
        atr = 0.0
        for k in range(period):
            atr += tr[k]
        atr /= period
        out[period - 1] = atr
        for i in range(period, n):
            atr = (atr * (period - 1) + tr[i]) / period
            out[i] = atr
    return out


@numba.jit(nopython=True, cache=True)
def compute_vr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    short_period: int, long_period: int,
) -> np.ndarray:
    """Compute Volatility Ratio = ATR(short) / ATR(long).

    Low VR (< threshold) indicates volatility compression / squeeze.
    Scale-invariant: same VR periods work across timeframes.

    Args:
        close: Close prices.
        high: High prices.
        low: Low prices.
        short_period: Short ATR period (default 7).
        long_period: Long ATR period (default 50).

    Returns:
        VR array (same length), NaN before long_period warmup.
    """
    atr_s = _atr_pct(close, high, low, short_period)
    atr_l = _atr_pct(close, high, low, long_period)
    n = len(close)
    vr = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_s[i]) and not np.isnan(atr_l[i]) and atr_l[i] > 0:
            vr[i] = atr_s[i] / atr_l[i]
    return vr


# ══════════════════════════════════════════════════════════════════════════════
#  Momentum & Squeeze Duration
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_momentum(close: np.ndarray, lookback: int) -> np.ndarray:
    """Simple return-based momentum: close[i] / close[i-lookback] - 1.

    Args:
        close: Close prices.
        lookback: Momentum lookback period.

    Returns:
        Momentum array (0.0 before lookback warmup).
    """
    n = len(close)
    out = np.full(n, 0.0)
    for i in range(lookback, n):
        if close[i - lookback] > 0:
            out[i] = close[i] / close[i - lookback] - 1.0
    return out


@numba.jit(nopython=True, cache=True)
def compute_squeeze_dur(vr: np.ndarray, threshold: float) -> np.ndarray:
    """Count consecutive bars where VR is below threshold.

    Args:
        vr: Volatility Ratio array.
        threshold: VR threshold below which squeeze is counted.

    Returns:
        Squeeze duration array (0 when not in squeeze).
    """
    n = len(vr)
    dur = np.zeros(n, dtype=np.int64)
    cnt = 0
    for i in range(n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        dur[i] = cnt
    return dur


# ══════════════════════════════════════════════════════════════════════════════
#  Signal Generators (zero-lookahead, each bar uses only past data)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def gen_vr_signal(
    vr: np.ndarray, threshold: float, warmup: int,
) -> np.ndarray:
    """Generate D_mom8 raw signal: 1 where VR < threshold (after warmup).

    Direction is NOT set here — see make_direction_momentum().

    Args:
        vr: Volatility Ratio array.
        threshold: VR threshold for entry trigger.
        warmup: Minimum bars before first signal.

    Returns:
        Binary signal array (0 or 1).
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_vr_dur_signal(
    vr: np.ndarray, threshold: float, min_dur: int, warmup: int,
) -> np.ndarray:
    """Generate S_d8 raw signal: 1 where VR < threshold AND squeeze ≥ min_dur.

    For straddle mode (Phase 2).

    Args:
        vr: Volatility Ratio array.
        threshold: VR threshold.
        min_dur: Minimum consecutive bars below threshold.
        warmup: Minimum bars before first signal.

    Returns:
        Binary signal array (0 or 1).
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    cnt = 0
    for i in range(n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        if i >= warmup and cnt >= min_dur:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def make_direction_momentum(
    sig: np.ndarray, momentum: np.ndarray,
) -> np.ndarray:
    """Assign direction to raw signals based on momentum.

    Positive momentum → +1 (LONG), negative → -1 (SHORT).
    Momentum near zero (|m| < 0.0001) defaults to +1.

    Args:
        sig: Raw signal array (0/1).
        momentum: Momentum array.

    Returns:
        Directional signal array (+1/-1/0).
    """
    n = len(sig)
    dire = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if sig[i] == 1:
            if momentum[i] > 0.0001:
                dire[i] = 1
            elif momentum[i] < -0.0001:
                dire[i] = -1
            else:
                dire[i] = 1
    return dire


# ══════════════════════════════════════════════════════════════════════════════
#  OHLCV Resampling (15m → 1h/4h)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def resample_ohlcv(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, ratio: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample OHLCV from lower TF to higher TF using COMPLETED bars only.

    CRITICAL: Uses only completed bars (truncates remainder).
    This prevents lookahead bias from developing (incomplete) bars.

    ratio=4 for 15m→1h, ratio=16 for 15m→4h.

    Args:
        close: Lower TF close prices.
        high: Lower TF high prices.
        low: Lower TF low prices.
        volume: Lower TF volumes.
        ratio: Resampling ratio (e.g. 4 for 15m→1h).

    Returns:
        Tuple of (close, high, low, volume) at higher TF.
    """
    n = len(close)
    n_out = n // ratio
    c_out = np.empty(n_out, dtype=np.float64)
    h_out = np.empty(n_out, dtype=np.float64)
    l_out = np.empty(n_out, dtype=np.float64)
    v_out = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        start = i * ratio
        end = start + ratio
        # Close = last bar's close (COMPLETED bar)
        c_out[i] = close[end - 1]
        # High = max high in window
        mx = high[start]
        for k in range(start + 1, end):
            if high[k] > mx:
                mx = high[k]
        h_out[i] = mx
        # Low = min low in window
        mn = low[start]
        for k in range(start + 1, end):
            if low[k] < mn:
                mn = low[k]
        l_out[i] = mn
        # Volume = sum
        vs = 0.0
        for k in range(start, end):
            vs += volume[k]
        v_out[i] = vs
    return c_out, h_out, l_out, v_out


# ══════════════════════════════════════════════════════════════════════════════
#  MA Cluster Squeeze (R44d V17)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average — NaN for indices < period-1."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for k in range(period):
        s += close[k]
    out[period - 1] = s / period
    for k in range(period, n):
        s += close[k] - close[k - period]
        out[k] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def check_ma_cluster(
    close: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    cluster_count_arr: np.ndarray,
    in_cooldown_arr: np.ndarray,
) -> int:
    """Check MA cluster squeeze signal on the LAST completed bar.

    Maintains running state via cluster_count_arr and in_cooldown_arr
    (both are 1-element arrays for mutability across calls).

    Returns:
        +1 (long), -1 (short), or 0 (no signal).
    """
    idx = len(close) - 1  # Last completed bar
    if idx < 0 or np.isnan(sma200[idx]):
        cluster_count_arr[0] = 0
        return 0

    # Compute MA band spread
    ma_max = max(sma20[idx], max(sma50[idx], max(sma100[idx], sma200[idx])))
    ma_min = min(sma20[idx], min(sma50[idx], min(sma100[idx], sma200[idx])))
    if close[idx] <= 0:
        return 0
    spread = (ma_max - ma_min) / close[idx]

    if spread <= cluster_pct:
        cluster_count_arr[0] += 1
    else:
        cluster_count_arr[0] = 0
        in_cooldown_arr[0] = 0  # Cluster dissolved → reset cooldown
        return 0

    # Duration filter
    if cluster_count_arr[0] < min_duration:
        return 0

    # One signal per cluster episode: skip if already fired
    if in_cooldown_arr[0] == 1:
        return 0

    # Breakout: close outside the cluster band
    if ma_min <= close[idx] <= ma_max:
        return 0

    # Momentum direction
    if abs(momentum[idx]) < 0.003:
        return 0

    # Mark cooldown — no more signals until cluster dissolves
    in_cooldown_arr[0] = 1

    if momentum[idx] > 0:
        return 1
    return -1
