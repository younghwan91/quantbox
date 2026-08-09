#!/usr/bin/env python3
"""
R26: Advanced Volatility Straddle Alpha Search
================================================
Improvements over R25 (all PF < 0.9):

SIGNAL IMPROVEMENTS:
  - Stricter squeeze thresholds (5th, 3rd percentile)
  - Squeeze DURATION filter (consecutive bars in squeeze ≥ N)
  - Parkinson volatility (uses high/low, 5× more efficient)
  - Vol-ratio: short_vol / long_vol < threshold
  - Volume drying up during squeeze
  - Multi-TF: 4h squeeze → 15m execution (completed bars only!)

EXIT IMPROVEMENTS:
  - ATR-adaptive SL/TP (scales to each coin's vol)
  - Trailing winner: after one side stops out, trail other side
  - Breakout straddle: enter only on range breakout (2 legs → 1 leg)

TOTAL: ~80+ variants across signal × exit × TF dimensions.
Zero lookahead.  Per-trade independent.  SL-priority.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# ── Constants ────────────────────────────────────────────────────────────────
COST_PER_SIDE = 0.0004
FEE_4LEGS = COST_PER_SIDE * 4  # classic straddle
FEE_2LEGS = COST_PER_SIDE * 2  # breakout straddle (1 entry + 1 exit)

BB_PERIOD = 20
ATR_PERIOD = 14
PCT_LOOKBACK = 200
PARKINSON_PERIOD = 20


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Indicators
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _pct_rank(arr: np.ndarray, lookback: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, 50.0)
    for i in range(lookback - 1, n):
        cur = arr[i]
        if np.isnan(cur):
            continue
        below = 0
        valid = 0
        for k in range(lookback):
            v = arr[i - lookback + 1 + k]
            if not np.isnan(v):
                valid += 1
                if v < cur:
                    below += 1
        if valid > 0:
            out[i] = below / valid * 100.0
    return out


@numba.jit(nopython=True, cache=True)
def _bbw(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += close[i - period + 1 + k]
        mean = s / period
        v = 0.0
        for k in range(period):
            d = close[i - period + 1 + k] - mean
            v += d * d
        v /= period
        if mean > 0:
            out[i] = 4.0 * np.sqrt(v) / mean
    return out


@numba.jit(nopython=True, cache=True)
def _atr_pct(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
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
def _parkinson_vol(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Parkinson volatility: uses high/low, ~5× more efficient than close-to-close."""
    n = len(high)
    out = np.full(n, np.nan)
    k = 1.0 / (4.0 * period * np.log(2.0))
    for i in range(period - 1, n):
        s = 0.0
        for j in range(period):
            idx = i - period + 1 + j
            if low[idx] > 0 and high[idx] > low[idx]:
                r = np.log(high[idx] / low[idx])
                s += r * r
        out[i] = np.sqrt(k * s)
    return out


@numba.jit(nopython=True, cache=True)
def _vol_sma(volume: np.ndarray, period: int) -> np.ndarray:
    n = len(volume)
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += volume[i]
        if i >= period:
            s -= volume[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def _consecutive_below(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Count consecutive bars where arr < threshold."""
    n = len(arr)
    out = np.zeros(n, dtype=np.int32)
    cnt = 0
    for i in range(n):
        if not np.isnan(arr[i]) and arr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        out[i] = cnt
    return out


@numba.jit(nopython=True, cache=True)
def _kc_width(atr_pct_arr: np.ndarray, mult: float) -> np.ndarray:
    n = len(atr_pct_arr)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_pct_arr[i]):
            out[i] = 2.0 * mult * atr_pct_arr[i]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Signal generation — 20+ signal variants
# ══════════════════════════════════════════════════════════════════════════════

NUM_SIGNALS = 20

SIGNAL_LABELS = [
    "V0:Random(96)",           # 0
    "V1:BBsq<10%",             # 1
    "V2:BBsq<5%",              # 2
    "V3:BBsq<3%",              # 3
    "V4:BBdur≥10",             # 4  BB squeeze duration ≥ 10 bars
    "V5:BBdur≥20",             # 5  BB squeeze duration ≥ 20 bars
    "V6:ATRcrush<5%",          # 6
    "V7:Parkinson<5%",         # 7
    "V8:VolRatio<0.5",         # 8  short_vol / long_vol < 0.5
    "V9:VolDry+BBsq",          # 9  volume drought + BB squeeze
    "V10:TTM+dur≥10",          # 10 TTM squeeze + duration ≥10
    "V11:3of5<5%",             # 11 composite: 3 of (BB, ATR, vol, range, parkinson) < 5th pct
    "V12:4of5<5%",             # 12 composite: 4 of 5 < 5th
    "V13:BBonset+dur≥10",      # 13 first bar AFTER 10+ bar squeeze (release)
    "V14:BBonset+dur≥20",      # 14 first bar AFTER 20+ bar squeeze
    "V15:InsideBar≥3",         # 15 3+ consecutive inside bars
    "V16:NarrowRange3",        # 16 3 consecutive narrowing ranges
    "V17:VolSurge+Sq",         # 17 volume surge (>2× avg) while squeezed
    "V18:BBsq<3%+VolDry",      # 18 strictest: BB<3% + volume drying
    "V19:BBdur≥30",            # 19 extreme duration
]


@numba.jit(nopython=True, cache=True)
def _gen_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    bb_period: int,
    atr_period: int,
    pct_lookback: int,
    parkinson_period: int,
) -> np.ndarray:
    n = len(close)
    signals = np.zeros((NUM_SIGNALS, n), dtype=np.int8)

    # ── Indicators ──
    bbw = _bbw(close, bb_period)
    atr = _atr_pct(close, high, low, atr_period)
    park = _parkinson_vol(high, low, parkinson_period)
    vol_sma20 = _vol_sma(volume, 20)
    vol_sma50 = _vol_sma(volume, 50)

    # Percentile ranks
    bbw_rank = _pct_rank(bbw, pct_lookback)
    atr_rank = _pct_rank(atr, pct_lookback)
    vol_rank = _pct_rank(volume, pct_lookback)
    park_rank = _pct_rank(park, pct_lookback)

    # Range % SMA for range compression
    rng = np.empty(n)
    for i in range(n):
        rng[i] = (high[i] - low[i]) / close[i] if close[i] > 0 else 0.0
    rng_sma = _vol_sma(rng, atr_period)
    rng_rank = _pct_rank(rng_sma, pct_lookback)

    # BB squeeze duration (consecutive bars with bbw_rank < 10, 5, 3)
    bb_dur10 = _consecutive_below(bbw_rank, 10.0)
    bb_dur5 = _consecutive_below(bbw_rank, 5.0)
    bb_dur3 = _consecutive_below(bbw_rank, 3.0)

    # Keltner width for TTM
    kc_w = _kc_width(atr, 1.5)

    # Short vol / long vol ratio
    atr_short = _atr_pct(close, high, low, 7)
    atr_long = _atr_pct(close, high, low, 50)

    warmup = max(pct_lookback, 50) + 1

    for i in range(warmup, n):
        # ── V0: Random ──
        if (i - warmup) % 96 == 0:
            signals[0, i] = 1

        bb10 = bbw_rank[i] < 10.0
        bb5 = bbw_rank[i] < 5.0
        bb3 = bbw_rank[i] < 3.0
        atr5 = atr_rank[i] < 5.0
        vol5 = vol_rank[i] < 5.0
        park5 = park_rank[i] < 5.0
        rng5 = rng_rank[i] < 5.0
        vol_dry = vol_rank[i] < 10.0

        # TTM: BB inside Keltner
        ttm = False
        if not np.isnan(bbw[i]) and not np.isnan(kc_w[i]):
            ttm = bbw[i] < kc_w[i]

        # Vol ratio
        vol_ratio_low = False
        if not np.isnan(atr_short[i]) and not np.isnan(atr_long[i]) and atr_long[i] > 0:
            vol_ratio_low = (atr_short[i] / atr_long[i]) < 0.5

        # ── Signals ──
        if bb10:
            signals[1, i] = 1
        if bb5:
            signals[2, i] = 1
        if bb3:
            signals[3, i] = 1
        if bb_dur10[i] >= 10:
            signals[4, i] = 1
        if bb_dur5[i] >= 20:
            signals[5, i] = 1
        if atr5:
            signals[6, i] = 1
        if park5:
            signals[7, i] = 1
        if vol_ratio_low:
            signals[8, i] = 1
        if vol_dry and bb5:
            signals[9, i] = 1
        if ttm and bb_dur10[i] >= 10:
            signals[10, i] = 1
        # Composite score (5 indicators at 5th percentile)
        score = 0
        if bb5:
            score += 1
        if atr5:
            score += 1
        if vol5:
            score += 1
        if rng5:
            score += 1
        if park5:
            score += 1
        if score >= 3:
            signals[11, i] = 1
        if score >= 4:
            signals[12, i] = 1

        # ── V15: Inside bars ──
        if i >= warmup + 3:
            inside_cnt = 0
            for k in range(1, 4):
                if high[i - k + 1] < high[i - k] and low[i - k + 1] > low[i - k]:
                    inside_cnt += 1
                else:
                    break
            if inside_cnt >= 3:
                signals[15, i] = 1

        # ── V16: Narrowing range (3 consecutive) ──
        if i >= warmup + 3:
            narrowing = True
            for k in range(1, 3):
                r_cur = high[i - k + 1] - low[i - k + 1]
                r_prev = high[i - k] - low[i - k]
                if r_cur >= r_prev:
                    narrowing = False
                    break
            if narrowing:
                signals[16, i] = 1

        # ── V17: Volume surge while squeezed ──
        if bb5 and not np.isnan(vol_sma20[i]) and vol_sma20[i] > 0:
            if volume[i] > 2.0 * vol_sma20[i]:
                signals[17, i] = 1

        # ── V18: Strictest: BB<3% + vol drying ──
        if bb3 and vol_dry:
            signals[18, i] = 1

        # ── V19: Extreme duration ──
        if bb_dur3[i] >= 30:
            signals[19, i] = 1

    # ── V13/V14: Onset signals (first bar AFTER long squeeze) ──
    for i in range(warmup + 1, n):
        # V13: first bar where bb_dur5 was ≥10 on prev bar but now broke OR signal[4] was 1 and now 0
        if signals[4, i - 1] == 1 and signals[4, i] == 0:
            signals[13, i] = 1
        if signals[5, i - 1] == 1 and signals[5, i] == 0:
            signals[14, i] = 1

    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Simulate straddles — 3 exit strategies
# ══════════════════════════════════════════════════════════════════════════════

# Exit strategy 1: Fixed SL/TP (same as R25)
@numba.jit(nopython=True, cache=True)
def sim_fixed(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    outcome = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        l_alive = True
        s_alive = True
        l_exit = 0.0
        s_exit = 0.0
        exit_bar = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if l_alive:
                if low[j] <= l_sl:
                    l_exit = -sl
                    l_alive = False
                elif high[j] >= l_tp:
                    l_exit = tp
                    l_alive = False
            if s_alive:
                if high[j] >= s_sl:
                    s_exit = -sl
                    s_alive = False
                elif low[j] <= s_tp:
                    s_exit = tp
                    s_alive = False
            if not l_alive and not s_alive:
                exit_bar = j
                break
        else:
            exit_bar = end - 1
        if l_alive:
            l_exit = (close[exit_bar] - entry) / entry
        if s_alive:
            s_exit = (entry - close[exit_bar]) / entry
        net = l_exit + s_exit - cost
        if l_exit >= tp * 0.9:
            oc = 1
        elif s_exit >= tp * 0.9:
            oc = 2
        elif l_exit <= -sl * 0.9 and s_exit <= -sl * 0.9:
            oc = 0
        else:
            oc = 3
        if cnt < cap:
            pnl[cnt] = net
            outcome[cnt] = oc
            cnt += 1
        i = exit_bar + 1
    return pnl[:cnt], outcome[:cnt]


# Exit strategy 2: ATR-adaptive SL/TP
@numba.jit(nopython=True, cache=True)
def sim_atr_adaptive(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr_arr: np.ndarray, sig: np.ndarray,
    sl_mult: float, tp_mult: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SL = sl_mult × ATR%, TP = tp_mult × ATR%.  Adapts per coin per bar."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    outcome = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        cur_atr = atr_arr[i]
        if entry <= 0 or np.isnan(cur_atr) or cur_atr <= 0:
            i += 1
            continue
        sl = sl_mult * cur_atr
        tp = tp_mult * cur_atr
        # Clamp SL/TP to reasonable range
        if sl < 0.001:
            sl = 0.001
        if sl > 0.05:
            sl = 0.05
        if tp < 0.002:
            tp = 0.002
        if tp > 0.15:
            tp = 0.15
        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        l_alive = True
        s_alive = True
        l_exit = 0.0
        s_exit = 0.0
        exit_bar = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if l_alive:
                if low[j] <= l_sl:
                    l_exit = -sl
                    l_alive = False
                elif high[j] >= l_tp:
                    l_exit = tp
                    l_alive = False
            if s_alive:
                if high[j] >= s_sl:
                    s_exit = -sl
                    s_alive = False
                elif low[j] <= s_tp:
                    s_exit = tp
                    s_alive = False
            if not l_alive and not s_alive:
                exit_bar = j
                break
        else:
            exit_bar = end - 1
        if l_alive:
            l_exit = (close[exit_bar] - entry) / entry
        if s_alive:
            s_exit = (entry - close[exit_bar]) / entry
        net = l_exit + s_exit - cost
        if l_exit >= tp * 0.9:
            oc = 1
        elif s_exit >= tp * 0.9:
            oc = 2
        elif l_exit <= -sl * 0.9 and s_exit <= -sl * 0.9:
            oc = 0
        else:
            oc = 3
        if cnt < cap:
            pnl[cnt] = net
            outcome[cnt] = oc
            cnt += 1
        i = exit_bar + 1
    return pnl[:cnt], outcome[:cnt]


# Exit strategy 3: TRAILING WINNER  ★ KEY INNOVATION ★
@numba.jit(nopython=True, cache=True)
def sim_trailing(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr_arr: np.ndarray, sig: np.ndarray,
    sl_mult: float, trail_mult: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    After one side stops out, trail the surviving side.
    Trail stop = trail_mult × ATR below max profit (long) or above min profit (short).
    No fixed TP — let winners run.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    outcome = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        cur_atr = atr_arr[i]
        if entry <= 0 or np.isnan(cur_atr) or cur_atr <= 0:
            i += 1
            continue
        sl_pct = sl_mult * cur_atr
        trail_pct = trail_mult * cur_atr
        if sl_pct < 0.001:
            sl_pct = 0.001
        if sl_pct > 0.05:
            sl_pct = 0.05
        if trail_pct < 0.002:
            trail_pct = 0.002
        if trail_pct > 0.10:
            trail_pct = 0.10

        l_sl = entry * (1.0 - sl_pct)
        s_sl = entry * (1.0 + sl_pct)

        l_alive = True
        s_alive = True
        l_exit_pnl = 0.0
        s_exit_pnl = 0.0

        # Trailing state
        l_max_price = entry  # highest seen for long
        s_min_price = entry  # lowest seen for short
        l_trailing = False   # long is now in trail mode
        s_trailing = False

        exit_bar = i
        end = min(i + 1 + max_hold, n)

        for j in range(i + 1, end):
            h = high[j]
            lo_j = low[j]

            # ── LONG side ──
            if l_alive:
                if not l_trailing:
                    # Initial SL check
                    if lo_j <= l_sl:
                        l_exit_pnl = -sl_pct
                        l_alive = False
                    elif not s_alive:
                        # Other side already out → start trailing
                        l_trailing = True
                        if h > l_max_price:
                            l_max_price = h
                else:
                    # Trailing mode
                    if h > l_max_price:
                        l_max_price = h
                    trail_stop = l_max_price * (1.0 - trail_pct)
                    if lo_j <= trail_stop:
                        l_exit_pnl = (trail_stop - entry) / entry
                        l_alive = False
                    # Also still check initial SL
                    if l_alive and lo_j <= l_sl:
                        l_exit_pnl = -sl_pct
                        l_alive = False

            # ── SHORT side ──
            if s_alive:
                if not s_trailing:
                    if h >= s_sl:
                        s_exit_pnl = -sl_pct
                        s_alive = False
                    elif not l_alive:
                        s_trailing = True
                        if lo_j < s_min_price:
                            s_min_price = lo_j
                else:
                    if lo_j < s_min_price:
                        s_min_price = lo_j
                    trail_stop = s_min_price * (1.0 + trail_pct)
                    if h >= trail_stop:
                        s_exit_pnl = (entry - trail_stop) / entry
                        s_alive = False
                    if s_alive and h >= s_sl:
                        s_exit_pnl = -sl_pct
                        s_alive = False

            if not l_alive and not s_alive:
                exit_bar = j
                break
        else:
            exit_bar = end - 1

        # Timeout closures
        if l_alive:
            l_exit_pnl = (close[exit_bar] - entry) / entry
        if s_alive:
            s_exit_pnl = (entry - close[exit_bar]) / entry

        net = l_exit_pnl + s_exit_pnl - cost

        # Classify
        if l_exit_pnl > sl_pct:
            oc = 1  # long won big
        elif s_exit_pnl > sl_pct:
            oc = 2  # short won big
        elif l_exit_pnl <= -sl_pct * 0.9 and s_exit_pnl <= -sl_pct * 0.9:
            oc = 0  # whipsaw
        else:
            oc = 3  # timeout / partial

        if cnt < cap:
            pnl[cnt] = net
            outcome[cnt] = oc
            cnt += 1
        i = exit_bar + 1
    return pnl[:cnt], outcome[:cnt]


# Exit strategy 4: BREAKOUT STRADDLE (limit orders above/below range)
@numba.jit(nopython=True, cache=True)
def sim_breakout(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr_arr: np.ndarray, sig: np.ndarray,
    break_mult: float, sl_mult: float, tp_mult: float,
    max_hold: int, cost_2leg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    On signal: place limit buy above recent high and limit sell below recent low.
    Only ONE side fills when price breaks out.  Cost = 2 legs instead of 4.
    If neither fills within wait_bars, cancel both.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    outcome = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    wait_bars = 24  # wait 6h for breakout

    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        cur_atr = atr_arr[i]
        if close[i] <= 0 or np.isnan(cur_atr) or cur_atr <= 0:
            i += 1
            continue

        # Define breakout levels: recent 10-bar high/low ± break_mult × ATR
        recent_high = close[i]
        recent_low = close[i]
        lookback = min(10, i)
        for k in range(lookback):
            if high[i - k] > recent_high:
                recent_high = high[i - k]
            if low[i - k] < recent_low:
                recent_low = low[i - k]

        atr_val = cur_atr * close[i]
        buy_level = recent_high + break_mult * atr_val
        sell_level = recent_low - break_mult * atr_val

        if buy_level <= 0 or sell_level <= 0:
            i += 1
            continue

        # Wait for breakout
        filled = False
        is_long = True
        entry = 0.0
        fill_bar = i
        limit = min(i + 1 + wait_bars, n)

        for j in range(i + 1, limit):
            if high[j] >= buy_level:
                entry = buy_level
                is_long = True
                fill_bar = j
                filled = True
                break
            if low[j] <= sell_level:
                entry = sell_level
                is_long = False
                fill_bar = j
                filled = True
                break

        if not filled:
            i = limit
            continue

        # Now manage single position with SL/TP
        sl = sl_mult * cur_atr
        tp = tp_mult * cur_atr
        if sl < 0.001:
            sl = 0.001
        if tp < 0.002:
            tp = 0.002

        exit_pnl = 0.0
        exit_bar = fill_bar
        end = min(fill_bar + 1 + max_hold, n)

        if is_long:
            l_sl = entry * (1.0 - sl)
            l_tp = entry * (1.0 + tp)
            for j in range(fill_bar + 1, end):
                if low[j] <= l_sl:
                    exit_pnl = -sl
                    exit_bar = j
                    break
                if high[j] >= l_tp:
                    exit_pnl = tp
                    exit_bar = j
                    break
            else:
                exit_bar = end - 1
                exit_pnl = (close[exit_bar] - entry) / entry
        else:
            s_sl = entry * (1.0 + sl)
            s_tp = entry * (1.0 - tp)
            for j in range(fill_bar + 1, end):
                if high[j] >= s_sl:
                    exit_pnl = -sl
                    exit_bar = j
                    break
                if low[j] <= s_tp:
                    exit_pnl = tp
                    exit_bar = j
                    break
            else:
                exit_bar = end - 1
                exit_pnl = (entry - close[exit_bar]) / entry

        net = exit_pnl - cost_2leg

        if exit_pnl >= tp * 0.9:
            oc = 1 if is_long else 2
        elif exit_pnl <= -sl * 0.9:
            oc = 0
        else:
            oc = 3

        if cnt < cap:
            pnl[cnt] = net
            outcome[cnt] = oc
            cnt += 1
        i = exit_bar + 1

    return pnl[:cnt], outcome[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_data() -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings
    settings = QuantBoxSettings()
    whitelist = settings.rb_symbol_whitelist.split(",")

    data: dict = {}
    for sym in whitelist:
        candidates = list(cache_dir.glob(f"{sym}_15m_*.npz"))
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)
        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        h = npz["high"].astype(np.float64)
        lo = npz["low"].astype(np.float64)
        v = npz["volume"].astype(np.float64)
        if len(c) < PCT_LOOKBACK + 200:
            continue
        data[sym] = {"close": c, "high": h, "low": lo, "volume": v, "bars": len(c)}
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-TF: 4h squeeze → 15m signal (completed bars only!)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _resample_4h(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Resample 15m → 4h (ratio=16). Returns 4h OHLCV and count."""
    n = len(close)
    ratio = 16
    n4 = n // ratio
    c4 = np.empty(n4, dtype=np.float64)
    h4 = np.empty(n4, dtype=np.float64)
    l4 = np.empty(n4, dtype=np.float64)
    v4 = np.empty(n4, dtype=np.float64)
    for j in range(n4):
        base = j * ratio
        hi = high[base]
        lo = low[base]
        vol = 0.0
        for k in range(ratio):
            idx = base + k
            if high[idx] > hi:
                hi = high[idx]
            if low[idx] < lo:
                lo = low[idx]
            vol += volume[idx]
        c4[j] = close[base + ratio - 1]
        h4[j] = hi
        l4[j] = lo
        v4[j] = vol
    return c4, h4, l4, v4, n4


@numba.jit(nopython=True, cache=True)
def _map_4h_squeeze_to_15m(
    bbw_4h: np.ndarray, n_4h: int, n_15m: int, pct_lookback: int,
) -> np.ndarray:
    """
    Map 4h BB squeeze status to 15m bars.
    CRITICAL: Use j_completed = (i // 16) - 1 → LAST COMPLETED 4h bar.
    """
    bbw_rank_4h = _pct_rank(bbw_4h, pct_lookback)
    out = np.zeros(n_15m, dtype=np.int8)
    ratio = 16
    for i in range(n_15m):
        j = i // ratio - 1  # LAST COMPLETED 4h bar
        if j < 0 or j >= n_4h:
            continue
        if bbw_rank_4h[j] < 5.0:
            out[i] = 1
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(all_pnl: dict, all_oc: dict) -> dict:
    summary: dict = {}
    for key in all_pnl:
        arr_list = all_pnl[key]
        if not arr_list:
            continue
        pnl_arr = np.concatenate(arr_list)
        oc_arr = np.concatenate(all_oc[key])
        n = len(pnl_arr)
        if n < 10:
            continue
        wins = int(np.sum((oc_arr == 1) | (oc_arr == 2)))
        whips = int(np.sum(oc_arr == 0))
        timeouts = int(np.sum(oc_arr == 3))
        pos = pnl_arr[pnl_arr > 0]
        neg = pnl_arr[pnl_arr < 0]
        gp = float(np.sum(pos)) if len(pos) > 0 else 0.0
        gn = float(np.abs(np.sum(neg))) if len(neg) > 0 else 1e-9
        pf = gp / gn
        ev = float(np.mean(pnl_arr)) * 10_000
        summary[key] = {
            "N": n, "Win%": wins / n * 100, "Whip%": whips / n * 100,
            "Time%": timeouts / n * 100, "PF": pf, "EV_bp": ev,
        }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  EXIT CONFIG DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Each config: (label, strategy_type, params...)
EXIT_CONFIGS = [
    # Fixed SL/TP (for reference)
    ("Fix:SL1/TP4/48h",       "fixed", 0.01, 0.04, 192),
    # ATR-adaptive
    ("ATR:SL0.5/TP2/96b",     "atr", 0.5, 2.0, 96),
    ("ATR:SL0.5/TP3/96b",     "atr", 0.5, 3.0, 96),
    ("ATR:SL0.75/TP3/192b",   "atr", 0.75, 3.0, 192),
    ("ATR:SL1.0/TP4/192b",    "atr", 1.0, 4.0, 192),
    # Trailing winner ★
    ("Trail:SL0.5/T1.0/192b", "trail", 0.5, 1.0, 192),
    ("Trail:SL0.5/T1.5/192b", "trail", 0.5, 1.5, 192),
    ("Trail:SL0.75/T1.0/192b","trail", 0.75, 1.0, 192),
    ("Trail:SL0.75/T1.5/384b","trail", 0.75, 1.5, 384),
    ("Trail:SL1.0/T1.0/384b", "trail", 1.0, 1.0, 384),
    ("Trail:SL1.0/T2.0/384b", "trail", 1.0, 2.0, 384),
    # Breakout straddle (1 leg only)
    ("Break:B0.5/SL1/TP3/96", "break", 0.5, 1.0, 3.0, 96),
    ("Break:B0.2/SL1/TP3/96", "break", 0.2, 1.0, 3.0, 96),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    commit = get_commit_short()
    now = datetime.now(timezone.utc)

    print("=" * 90)
    print("  R26: ADVANCED VOLATILITY STRADDLE — DEEP SEARCH")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Dimensions: 20 signals × 13 exits × 2 TFs = up to 520 variants")
    print("  Key innovations: trailing winner, ATR-adaptive, breakout straddle, 4h MTF")
    print("=" * 90)
    print()

    # ── Load data ──
    print("  Loading data...")
    data = load_data()
    if not data:
        print("  ❌ No data!")
        return
    total_bars = sum(d["bars"] for d in data.values())
    print(f"  Symbols: {len(data)}  |  Total bars: {total_bars:,}")
    print()

    all_pnl: dict[str, list[np.ndarray]] = {}
    all_oc: dict[str, list[np.ndarray]] = {}
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: 15m signals on 15m data
    # ══════════════════════════════════════════════════════════════════════════
    print("  ── PHASE 1: 15m signals ──")
    sym_count = 0
    for sym, d in sorted(data.items()):
        sym_count += 1
        c, h, lo, v = d["close"], d["high"], d["low"], d["volume"]
        atr = _atr_pct(c, h, lo, ATR_PERIOD)
        sig_matrix = _gen_signals(c, h, lo, v, BB_PERIOD, ATR_PERIOD, PCT_LOOKBACK, PARKINSON_PERIOD)

        for sig_idx in range(NUM_SIGNALS):
            sig = sig_matrix[sig_idx]
            for ecfg in EXIT_CONFIGS:
                label = ecfg[0]
                stype = ecfg[1]
                key = f"15m|{SIGNAL_LABELS[sig_idx]}|{label}"

                if stype == "fixed":
                    sl, tp, mh = ecfg[2], ecfg[3], ecfg[4]
                    p, o = sim_fixed(c, h, lo, sig, sl, tp, mh, FEE_4LEGS)
                elif stype == "atr":
                    sl_m, tp_m, mh = ecfg[2], ecfg[3], ecfg[4]
                    p, o = sim_atr_adaptive(c, h, lo, atr, sig, sl_m, tp_m, mh, FEE_4LEGS)
                elif stype == "trail":
                    sl_m, tr_m, mh = ecfg[2], ecfg[3], ecfg[4]
                    p, o = sim_trailing(c, h, lo, atr, sig, sl_m, tr_m, mh, FEE_4LEGS)
                elif stype == "break":
                    br_m, sl_m, tp_m, mh = ecfg[2], ecfg[3], ecfg[4], ecfg[5]
                    p, o = sim_breakout(c, h, lo, atr, sig, br_m, sl_m, tp_m, mh, FEE_2LEGS)
                else:
                    continue

                if key not in all_pnl:
                    all_pnl[key] = []
                    all_oc[key] = []
                all_pnl[key].append(p)
                all_oc[key].append(o)

        if sym_count % 10 == 0:
            print(f"    [{sym_count}/{len(data)}] {sym}")

    print(f"  Phase 1 done: {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: 4h squeeze → 15m execution (MTF)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ── PHASE 2: 4h squeeze signals (completed bars only) ──")
    t1 = time.time()
    mtf_signals_labels = ["MTF4h:BBsq<5%", "MTF4h:BBsq<3%"]
    mtf_thresholds = [5.0, 3.0]

    for sym, d in sorted(data.items()):
        c, h, lo, v = d["close"], d["high"], d["low"], d["volume"]
        n = len(c)
        atr = _atr_pct(c, h, lo, ATR_PERIOD)

        # Resample to 4h
        c4, h4, l4, v4, n4 = _resample_4h(c, h, lo, v)
        if n4 < PCT_LOOKBACK + 10:
            continue

        bbw_4h = _bbw(c4, BB_PERIOD)

        for tidx, thr in enumerate(mtf_thresholds):
            # Map 4h squeeze to 15m signal
            bbw_rank_4h = _pct_rank(bbw_4h, min(PCT_LOOKBACK, n4))

            # Build 15m signal from completed 4h bars
            sig_4h = np.zeros(n, dtype=np.int8)
            ratio = 16
            for i in range(n):
                j = i // ratio - 1  # LAST COMPLETED 4h bar (zero lookahead)
                if j < 0 or j >= n4:
                    continue
                if bbw_rank_4h[j] < thr:
                    sig_4h[i] = 1

            sig_label = mtf_signals_labels[tidx]

            # Only run with the most promising exit configs (reduce compute)
            promising_exits = [ecfg for ecfg in EXIT_CONFIGS if ecfg[1] in ("trail", "atr")]
            for ecfg in promising_exits:
                label = ecfg[0]
                stype = ecfg[1]
                key = f"4h→15m|{sig_label}|{label}"

                if stype == "atr":
                    sl_m, tp_m, mh = ecfg[2], ecfg[3], ecfg[4]
                    p, o = sim_atr_adaptive(c, h, lo, atr, sig_4h, sl_m, tp_m, mh, FEE_4LEGS)
                elif stype == "trail":
                    sl_m, tr_m, mh = ecfg[2], ecfg[3], ecfg[4]
                    p, o = sim_trailing(c, h, lo, atr, sig_4h, sl_m, tr_m, mh, FEE_4LEGS)
                else:
                    continue

                if key not in all_pnl:
                    all_pnl[key] = []
                    all_oc[key] = []
                all_pnl[key].append(p)
                all_oc[key].append(o)

    print(f"  Phase 2 done: {time.time() - t1:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  ✅ Total runtime: {elapsed:.1f}s")
    print(f"  Total variants tested: {len(all_pnl)}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  Aggregate & Report
    # ══════════════════════════════════════════════════════════════════════════
    summary = aggregate(all_pnl, all_oc)

    # ── Sort by PF and display top results ──
    print("=" * 90)
    print(f"  TOP 30 VARIANTS (by Profit Factor)")
    print("=" * 90)
    top = sorted(summary.items(), key=lambda x: -x[1]["PF"])
    for rank, (key, s) in enumerate(top[:30], 1):
        m = "✅" if s["PF"] >= 1.0 else "❌"
        print(
            f"  {rank:>2}. {m} PF={s['PF']:.3f}  EV={s['EV_bp']:>+7.1f}bp  "
            f"Win={s['Win%']:>5.1f}%  Whip={s['Whip%']:>5.1f}%  "
            f"N={s['N']:>7,}  {key}"
        )

    # ── PF ≥ 1.0 count ──
    pf_above_1 = [(k, v) for k, v in summary.items() if v["PF"] >= 1.0]
    print(f"\n  Variants with PF ≥ 1.0: {len(pf_above_1)} / {len(summary)}")
    if pf_above_1:
        print("\n  ── ALL PF ≥ 1.0 VARIANTS ──")
        for key, s in sorted(pf_above_1, key=lambda x: -x[1]["PF"]):
            print(
                f"    ✅ PF={s['PF']:.3f}  EV={s['EV_bp']:>+7.1f}bp  "
                f"Win={s['Win%']:>5.1f}%  N={s['N']:>7,}  {key}"
            )

    # ── Best by exit strategy type ──
    print(f"\n  ── BEST BY EXIT STRATEGY ──")
    for stype in ["fixed", "atr", "trail", "break"]:
        candidates = [(k, v) for k, v in summary.items() if f"|{stype[:3].title()}" in k or f"|{stype[:3].upper()}" in k or stype.capitalize()[:3] in k]
        if not candidates:
            # More flexible matching
            if stype == "fixed":
                candidates = [(k, v) for k, v in summary.items() if "Fix:" in k]
            elif stype == "atr":
                candidates = [(k, v) for k, v in summary.items() if "ATR:" in k]
            elif stype == "trail":
                candidates = [(k, v) for k, v in summary.items() if "Trail:" in k]
            elif stype == "break":
                candidates = [(k, v) for k, v in summary.items() if "Break:" in k]
        if candidates:
            best_k, best_v = max(candidates, key=lambda x: x[1]["PF"])
            m = "✅" if best_v["PF"] >= 1.0 else "❌"
            print(f"    {m} {stype.upper():<8} PF={best_v['PF']:.3f}  EV={best_v['EV_bp']:>+7.1f}bp  N={best_v['N']:>7,}  {best_k}")

    # ── Best by signal family ──
    print(f"\n  ── BEST BY SIGNAL (across all exits) ──")
    for sig_idx in range(NUM_SIGNALS):
        label_fragment = SIGNAL_LABELS[sig_idx]
        candidates = [(k, v) for k, v in summary.items() if label_fragment in k]
        if candidates:
            best_k, best_v = max(candidates, key=lambda x: x[1]["PF"])
            m = "✅" if best_v["PF"] >= 1.0 else "❌"
            print(
                f"    {m} {SIGNAL_LABELS[sig_idx]:<22} PF={best_v['PF']:.3f}  "
                f"EV={best_v['EV_bp']:>+7.1f}bp  N={best_v['N']:>7,}"
            )

    # ── MTF results ──
    mtf_results = [(k, v) for k, v in summary.items() if k.startswith("4h")]
    if mtf_results:
        print(f"\n  ── 4h→15m MTF RESULTS (top 10) ──")
        for key, s in sorted(mtf_results, key=lambda x: -x[1]["PF"])[:10]:
            m = "✅" if s["PF"] >= 1.0 else "❌"
            print(
                f"    {m} PF={s['PF']:.3f}  EV={s['EV_bp']:>+7.1f}bp  "
                f"Win={s['Win%']:>5.1f}%  N={s['N']:>7,}  {key}"
            )

    # ── Edge: trailing vs fixed vs random ──
    print(f"\n  ── TRAILING vs FIXED vs RANDOM (V1:BBsq<10%) ──")
    for ecfg in EXIT_CONFIGS:
        rnd_key = f"15m|V0:Random(96)|{ecfg[0]}"
        bb_key = f"15m|V1:BBsq<10%|{ecfg[0]}"
        if rnd_key in summary and bb_key in summary:
            r = summary[rnd_key]
            b = summary[bb_key]
            print(
                f"    {ecfg[0]:<24} Random PF={r['PF']:.3f}  "
                f"BBsq PF={b['PF']:.3f}  edge={b['PF']-r['PF']:>+.3f}"
            )

    # ── Distribution of #1 variant ──
    if top:
        best_key = top[0][0]
        if best_key in all_pnl:
            pnls = np.concatenate(all_pnl[best_key])
            print(f"\n  ── P&L DIST: {best_key} ──")
            for lbl, val in [
                ("Mean", np.mean(pnls)),
                ("Std", np.std(pnls)),
                ("P5", np.percentile(pnls, 5)),
                ("P25", np.percentile(pnls, 25)),
                ("Median", np.median(pnls)),
                ("P75", np.percentile(pnls, 75)),
                ("P95", np.percentile(pnls, 95)),
                ("Max", np.max(pnls)),
            ]:
                print(f"    {lbl:<8} {val*10000:>+8.1f} bp")

    # ── Verdict ──
    best_pf = top[0][1]["PF"] if top else 0.0
    if best_pf >= 1.10:
        verdict = "✅ PROMISING — PF ≥ 1.10.  Deep-dive & robustness check warranted."
    elif best_pf >= 1.00:
        verdict = f"⚠️  MARGINAL — Best PF={best_pf:.3f}.  Thin edge, needs slippage test."
    else:
        verdict = f"❌ NO ALPHA — Best PF={best_pf:.3f} < 1.0."

    print(f"\n{'='*90}")
    print(f"  VERDICT: {verdict}")
    print(f"  Best: {top[0][0]} → PF={best_pf:.3f}" if top else "  No results")
    print(f"  Total variants: {len(summary)}")
    print(f"{'='*90}\n")

    # Save
    out_path = ROOT / "logs" / "research" / "vol_straddle_r26_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "commit": commit,
        "date": now.isoformat(),
        "total_variants": len(summary),
        "pf_above_1_count": len(pf_above_1),
        "best_pf": round(best_pf, 4),
        "best_variant": top[0][0] if top else "",
        "top30": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in top[:30]},
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  💾 {out_path}")


if __name__ == "__main__":
    main()
