#!/usr/bin/env python3
"""
R33: Creative Strategy A/B Test
=================================
Tests fundamentally different hypotheses beyond exit mechanics (R32).
Base: G-ALL deduped (15m+1h+4h) from R31.

CATEGORIES:
  A. Signal Quality    — Deeper squeeze, volume confirmation, signal strength
  B. Asymmetric Design — Momentum-biased straddle legs, hybrid D+S
  C. Cross-Asset       — BTC regime overlay (market compression filter)
  D. Anti-Crowding     — Skip when too many symbols fire simultaneously
  E. Entry Timing      — Delay confirmation, mean-reversion proximity filter
  F. Adaptive Holding  — Hold time proportional to squeeze depth
  G. Breakout (Inverse)— Trade VR>0.65 expansion instead of compression
  H. Behavioral        — Post-loss cooldown, anti-martingale sizing
  I. Combined Best     — Stack top-performing filters

VARIANTS:
  A0-Baseline       : Current G-ALL deduped (reference)
  A1-DeepSqueeze    : VR<0.35 for D, VR<0.45 for S (deeper = higher conviction)
  A2-VolConfirm     : Require vol_pct > 70 at entry
  A3-LongSqueeze    : S_d8 → S_d16 (16-bar squeeze minimum)
  B1-MomBias-70     : Straddle 70/30 size split favoring momentum direction
  B2-MomBias-Asym   : Straddle: momentum-leg TP=7%, counter-leg TP=3%
  B3-Hybrid-DS      : Only straddle when D_mom8 also signals (confluence)
  C1-BTC-Squeeze    : Only trade when BTC is also in squeeze (VR<0.55)
  C2-BTC-Neutral    : Only trade when BTC |momentum_8| < 1% (range-bound)
  D1-AntiCrowd-3    : Skip if >3 symbols signal on same bar
  D2-AntiCrowd-5    : Skip if >5 symbols signal on same bar
  E1-Delay-1bar     : Enter 1 bar after signal (confirmation)
  E2-SMA-Proximity  : Only straddle when price within 0.5% of SMA(20)
  F1-AdaptHold-VR   : Hold time proportional to 1/VR (deeper squeeze → longer)
  G1-Breakout       : VR>0.65 + momentum directional (expansion trade)
  H1-Cooldown-5     : Skip 5 bars after a losing trade
  H2-AntiMartingale : After win: ×1.5 next PnL weight; after loss: ×0.5
  I1-BestCombo      : Combine top filters (determined after Phase 1)

All use 4-core parallel. ~20 variants.
"""

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    WARMUP_15m,
    WARMUP_1h,
    WARMUP_4h,
    _atr_pct,
    bootstrap_pf,
    build_resampled,
    compute_momentum,
    compute_squeeze_dur,
    compute_vr,
    full_stats,
    gen_vr_dur_signal,
    gen_vr_signal,
    load_15m,
    make_direction_momentum,
    pf_from_pnl,
    precompute,
    resample_ohlcv,
    vol_percentile,
)

NUM_WORKERS = 4

# Global data (set once in main, inherited by fork)
_G: dict = {}   # {"15m": {}, "1h": {}, "4h": {}, "years": 0, "btc_vr": {}, "btc_mom": {}}


def _init_worker(g: dict) -> None:
    global _G
    _G = g


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Additional indicators
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += close[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += close[i] - close[i - period]
        out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def sma_proximity(close: np.ndarray, sma_arr: np.ndarray, threshold: float) -> np.ndarray:
    """1 if close is within threshold % of SMA, else 0."""
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if not np.isnan(sma_arr[i]) and sma_arr[i] > 0:
            dist = abs(close[i] - sma_arr[i]) / sma_arr[i]
            if dist <= threshold:
                out[i] = 1
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Simulation variants
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard straddle returning (pnl, entries, exits)."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
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
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard directional returning (pnl, entries, exits)."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry
        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_straddle_mombias(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, momentum: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    mom_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Momentum-biased straddle: mom_weight on momentum leg, (1-mom_weight) on counter.
    mom_weight=0.7 means 70% size on direction of momentum, 30% against.
    PnL = mom_weight * aligned_leg + (1-mom_weight) * counter_leg - cost.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
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
        # Determine momentum direction
        mom_long = momentum[i] > 0.0001  # True = bullish momentum

        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry

        # Apply momentum bias
        if mom_long:
            net = mom_weight * le + (1.0 - mom_weight) * se - cost
        else:
            net = (1.0 - mom_weight) * le + mom_weight * se - cost

        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_straddle_asym_tp(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, momentum: np.ndarray,
    sl: float, tp_mom: float, tp_counter: float,
    max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric TP straddle: TP on momentum-leg = tp_mom, counter-leg = tp_counter."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
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
        mom_long = momentum[i] > 0.0001

        # Assign TP per leg based on momentum direction
        if mom_long:
            l_tp_pct = tp_mom
            s_tp_pct = tp_counter
        else:
            l_tp_pct = tp_counter
            s_tp_pct = tp_mom

        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + l_tp_pct)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - s_tp_pct)
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = l_tp_pct
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = s_tp_pct
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_cooldown(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    cooldown_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional with cooldown: skip cooldown_bars after a losing trade."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    cooldown_until = -1
    while i < n - 1:
        if i < cooldown_until:
            i += 1
            continue
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry
        net = trade_pnl - cost
        if net < 0:
            cooldown_until = eb + 1 + cooldown_bars
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_straddle_cooldown(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
    cooldown_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle with cooldown: skip cooldown_bars after a losing trade."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    cooldown_until = -1
    while i < n - 1:
        if i < cooldown_until:
            i += 1
            continue
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
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if net < 0:
            cooldown_until = eb + 1 + cooldown_bars
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_adapt_hold(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray, vr: np.ndarray,
    sl: float, tp: float, base_hold: int, cost: float,
    hold_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional with adaptive hold: hold = base_hold * hold_scale / VR.
    Deeper squeeze (lower VR) → longer hold time.
    Clamped to [base_hold//2, base_hold*3].
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
        # Adaptive hold
        cur_vr = vr[i] if not np.isnan(vr[i]) else 0.5
        if cur_vr < 0.05:
            cur_vr = 0.05
        adap_hold = int(base_hold * hold_scale / cur_vr)
        adap_hold = max(base_hold // 2, min(adap_hold, base_hold * 3))

        end = min(i + 1 + adap_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry
        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_straddle_adapt_hold(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, vr: np.ndarray,
    sl: float, tp: float, base_hold: int, cost: float,
    hold_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle with adaptive hold."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
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
        cur_vr = vr[i] if not np.isnan(vr[i]) else 0.5
        if cur_vr < 0.05:
            cur_vr = 0.05
        adap_hold = int(base_hold * hold_scale / cur_vr)
        adap_hold = max(base_hold // 2, min(adap_hold, base_hold * 3))

        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + adap_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Signal filter helpers
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def filter_by_mask(sig: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out signal where mask is 0."""
    n = len(sig)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if sig[i] != 0 and mask[i] == 1:
            out[i] = sig[i]
    return out


@numba.jit(nopython=True, cache=True)
def delay_signal(sig: np.ndarray, delay: int) -> np.ndarray:
    """Delay signal by `delay` bars (verification bar)."""
    n = len(sig)
    out = np.zeros(n, dtype=np.int8)
    for i in range(delay, n):
        if sig[i - delay] != 0:
            out[i] = sig[i - delay]
    return out


@numba.jit(nopython=True, cache=True)
def vol_pct_mask(vol_pct: np.ndarray, threshold: float) -> np.ndarray:
    """Mask: 1 where vol_pct >= threshold."""
    n = len(vol_pct)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if vol_pct[i] >= threshold:
            out[i] = 1
    return out


@numba.jit(nopython=True, cache=True)
def btc_squeeze_mask(btc_vr: np.ndarray, threshold: float, ratio: int) -> np.ndarray:
    """Create mask for altcoin bars based on BTC VR state.
    btc_vr is on 15m timeframe. ratio converts to target TF bars.
    Output length = len(btc_vr) // ratio (completed bars only).
    """
    n_out = len(btc_vr) // ratio
    out = np.zeros(n_out, dtype=np.int8)
    for i in range(n_out):
        # Use LAST COMPLETED bar of BTC VR for this HTF bar
        # For ratio=1 (15m), just use current index
        # For ratio=4 (1h), use the last 15m bar in this 1h window
        btc_idx = (i + 1) * ratio - 1  # last 15m bar of this TF bar
        if btc_idx < len(btc_vr) and not np.isnan(btc_vr[btc_idx]):
            if btc_vr[btc_idx] < threshold:
                out[i] = 1
    return out


@numba.jit(nopython=True, cache=True)
def btc_neutral_mask(btc_mom: np.ndarray, threshold: float, ratio: int) -> np.ndarray:
    """Mask: 1 where |BTC momentum| < threshold (range-bound market)."""
    n_out = len(btc_mom) // ratio
    out = np.zeros(n_out, dtype=np.int8)
    for i in range(n_out):
        btc_idx = (i + 1) * ratio - 1
        if btc_idx < len(btc_mom):
            if abs(btc_mom[btc_idx]) < threshold:
                out[i] = 1
    return out


@numba.jit(nopython=True, cache=True)
def momentum_filter_mask(momentum: np.ndarray, max_abs_mom: float) -> np.ndarray:
    """Mask: 1 where |momentum| < max_abs_mom (not too directional)."""
    n = len(momentum)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if abs(momentum[i]) < max_abs_mom:
            out[i] = 1
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Anti-martingale post-processing
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def apply_antimartingale(pnl: np.ndarray, win_mult: float, loss_mult: float) -> np.ndarray:
    """Apply anti-martingale sizing: after win multiply next trade by win_mult,
    after loss multiply by loss_mult. First trade is 1.0x."""
    n = len(pnl)
    out = np.empty(n, dtype=np.float64)
    mult = 1.0
    for i in range(n):
        out[i] = pnl[i] * mult
        if pnl[i] > 0:
            mult = win_mult
        elif pnl[i] < 0:
            mult = loss_mult
        else:
            mult = 1.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  G-ALL TF configs
# ══════════════════════════════════════════════════════════════════════════════

TF_15m = ("15m", 1, WARMUP_15m, 192, 384, 8)
TF_1h = ("1h", 4, WARMUP_1h, 48, 96, 8)
TF_4h = ("4h", 16, WARMUP_4h, 12, 24, 8)
G_ALL_TFS = [TF_15m, TF_1h, TF_4h]


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════


# Variant config structure:
# {
#   "desc": str,
#   "sig_filter": None | function_name,  # applied to signal before sim
#   "sim_type_D": "standard" | "cooldown" | "adapt_hold",
#   "sim_type_S": "standard" | "mombias" | "asym_tp" | "cooldown" | "adapt_hold",
#   "params": dict  # extra params for the variant
# }


VARIANTS: dict[str, dict] = {
    "A0-Baseline": {
        "desc": "Current G-ALL deduped (SL=1%/1.5%, TP=5%)",
        "type": "standard",
    },
    "A1-DeepSqueeze": {
        "desc": "Deeper squeeze threshold: D VR<0.35, S VR<0.45",
        "type": "deep_squeeze",
        "d_vr_thresh": 0.35,
        "s_vr_thresh": 0.45,
    },
    "A2-VolConfirm": {
        "desc": "Volume confirmation: vol_pct ≥ 70 at entry",
        "type": "vol_filter",
        "vol_threshold": 70.0,
    },
    "A3-LongSqueeze": {
        "desc": "S_d8 → S_d16 (16-bar squeeze minimum)",
        "type": "long_squeeze",
        "s_min_dur": 16,
    },
    "B1-MomBias70": {
        "desc": "Straddle 70/30 size split favoring momentum",
        "type": "mombias",
        "mom_weight": 0.7,
    },
    "B2-AsymTP": {
        "desc": "Straddle: mom-leg TP=7%, counter-leg TP=3%",
        "type": "asym_tp",
        "tp_mom": 0.07,
        "tp_counter": 0.03,
    },
    "B3-Confluence": {
        "desc": "Only straddle when D_mom8 also signals (signal confluence)",
        "type": "confluence",
    },
    "C1-BTC-Squeeze": {
        "desc": "Only trade when BTCUSDT VR<0.55 (market compressed)",
        "type": "btc_squeeze",
        "btc_vr_thresh": 0.55,
    },
    "C2-BTC-Neutral": {
        "desc": "Only trade when BTC |momentum_8|<1% (range-bound)",
        "type": "btc_neutral",
        "btc_mom_thresh": 0.01,
    },
    "D1-AntiCrowd3": {
        "desc": "Skip if >3 coins signal on same bar (anti-crowding)",
        "type": "anticrowd",
        "max_concurrent_sigs": 3,
    },
    "D2-AntiCrowd5": {
        "desc": "Skip if >5 coins signal on same bar",
        "type": "anticrowd",
        "max_concurrent_sigs": 5,
    },
    "E1-Delay1bar": {
        "desc": "Enter 1 bar after signal (confirmation delay)",
        "type": "delay",
        "delay_bars": 1,
    },
    "E2-SMA-Prox": {
        "desc": "Only straddle when price within 0.5% of SMA(20)",
        "type": "sma_proximity",
        "sma_period": 20,
        "sma_threshold": 0.005,
    },
    "F1-AdaptHold": {
        "desc": "Hold time ∝ 1/VR (deeper squeeze → longer hold, scale=0.5)",
        "type": "adapt_hold",
        "hold_scale": 0.5,
    },
    "G1-Breakout": {
        "desc": "VR>0.65 momentum directional (expansion trade, inverse strategy)",
        "type": "breakout",
        "vr_thresh": 0.65,
    },
    "H1-Cooldown5": {
        "desc": "Skip 5 bars after a losing trade per symbol",
        "type": "cooldown",
        "cooldown_bars": 5,
    },
    "H2-AntiMart": {
        "desc": "Anti-martingale: win→×1.5 next, loss→×0.5 next",
        "type": "antimartingale",
        "win_mult": 1.5,
        "loss_mult": 0.5,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ENGINE: run a variant with all TFs + dedup
# ══════════════════════════════════════════════════════════════════════════════


def _build_btc_masks(btc_data_15m: dict, ratio: int) -> tuple[np.ndarray, np.ndarray]:
    """Get BTC VR and momentum arrays mapped to target TF length."""
    if "BTCUSDT" not in btc_data_15m:
        # Fallback: all-ones mask (everything passes)
        return np.array([]), np.array([])
    d = btc_data_15m["BTCUSDT"]
    return d["vr"], d["mom_8"]


def _run_variant_strat(
    var_name: str, var_cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
    btc_vr_15m: np.ndarray, btc_mom_15m: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Run one variant for one strategy with dedup. Returns (deduped_pnl, n_raw, n_dedup)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    vtype = var_cfg.get("type", "standard")
    raw_pnls: list[float] = []
    deduped_pnls: list[float] = []

    # ── Anti-crowding: precompute per-bar signal counts across symbols ──
    crowd_counts: dict[str, dict[int, int]] = {}  # tf_label -> {bar_idx: count}
    if vtype == "anticrowd":
        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in G_ALL_TFS:
            d_tf = tf_data_map[ratio]
            bar_counts: dict[int, int] = {}
            for sym, d in d_tf.items():
                if strat == "D_mom8":
                    sig = gen_vr_signal(d["vr"], 0.45, warmup)
                else:
                    sig = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)
                for idx in range(len(sig)):
                    if sig[idx] != 0:
                        bar_counts[idx] = bar_counts.get(idx, 0) + 1
            crowd_counts[tf_label] = bar_counts

    for sym in sorted(data_15m.keys()):
        all_sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in G_ALL_TFS:
            d_tf = tf_data_map[ratio]
            if sym not in d_tf:
                continue
            d = d_tf[sym]
            n_bars = len(d["close"])

            mh = mh_d if strat == "D_mom8" else mh_s
            md = 8 if strat == "D_mom8" else min_dur

            # ────── Build signals based on variant type ─────────
            if vtype == "deep_squeeze":
                if strat == "D_mom8":
                    sig = gen_vr_signal(d["vr"], var_cfg["d_vr_thresh"], warmup)
                else:
                    sig = gen_vr_dur_signal(d["vr"], var_cfg["s_vr_thresh"], md, warmup)
            elif vtype == "long_squeeze" and strat == "S_d8":
                sig = gen_vr_dur_signal(d["vr"], 0.55, var_cfg["s_min_dur"], warmup)
            elif vtype == "breakout":
                # Invert: VR > threshold → signal
                vr_thresh = var_cfg["vr_thresh"]
                sig = np.zeros(n_bars, dtype=np.int8)
                for idx in range(warmup, n_bars):
                    if not np.isnan(d["vr"][idx]) and d["vr"][idx] > vr_thresh:
                        sig[idx] = 1
                if strat == "S_d8":
                    # Breakout doesn't make sense for straddle — skip
                    continue
            else:
                # Standard signal generation
                if strat == "D_mom8":
                    sig = gen_vr_signal(d["vr"], 0.45, warmup)
                else:
                    sig = gen_vr_dur_signal(d["vr"], 0.55, md, warmup)

            # ────── Apply signal filters ──────
            if vtype == "vol_filter":
                mask = vol_pct_mask(d["vol_pct"], var_cfg["vol_threshold"])
                sig = filter_by_mask(sig, mask)

            elif vtype == "btc_squeeze" and len(btc_vr_15m) > 0:
                btc_mask = btc_squeeze_mask(btc_vr_15m, var_cfg["btc_vr_thresh"], ratio)
                # Align lengths
                use_len = min(n_bars, len(btc_mask))
                sig_new = np.zeros(n_bars, dtype=np.int8)
                for idx in range(use_len):
                    if sig[idx] != 0 and btc_mask[idx] == 1:
                        sig_new[idx] = sig[idx]
                sig = sig_new

            elif vtype == "btc_neutral" and len(btc_mom_15m) > 0:
                btc_mask = btc_neutral_mask(btc_mom_15m, var_cfg["btc_mom_thresh"], ratio)
                use_len = min(n_bars, len(btc_mask))
                sig_new = np.zeros(n_bars, dtype=np.int8)
                for idx in range(use_len):
                    if sig[idx] != 0 and btc_mask[idx] == 1:
                        sig_new[idx] = sig[idx]
                sig = sig_new

            elif vtype == "anticrowd":
                max_conc = var_cfg["max_concurrent_sigs"]
                bc = crowd_counts.get(tf_label, {})
                sig_new = sig.copy()
                for idx in range(len(sig_new)):
                    if sig_new[idx] != 0 and bc.get(idx, 0) > max_conc:
                        sig_new[idx] = 0
                sig = sig_new

            elif vtype == "delay":
                sig = delay_signal(sig, var_cfg["delay_bars"])

            elif vtype == "sma_proximity" and strat == "S_d8":
                sma_arr = sma(d["close"], var_cfg["sma_period"])
                prox_mask = sma_proximity(d["close"], sma_arr, var_cfg["sma_threshold"])
                sig = filter_by_mask(sig, prox_mask)

            elif vtype == "confluence" and strat == "S_d8":
                # Only straddle when D_mom8 signal is also active
                d_sig = gen_vr_signal(d["vr"], 0.45, warmup)
                sig = filter_by_mask(sig, d_sig)

            dire = make_direction_momentum(sig, d["mom_8"]) if strat == "D_mom8" else np.zeros(1, dtype=np.int8)

            # ────── Run simulation ──────
            if strat == "D_mom8":
                if vtype == "cooldown":
                    p, e, x = sim_directional_cooldown(
                        d["close"], d["high"], d["low"], sig, dire,
                        0.01, 0.05, mh, COST_DIRECTIONAL, var_cfg["cooldown_bars"])
                elif vtype == "adapt_hold":
                    p, e, x = sim_directional_adapt_hold(
                        d["close"], d["high"], d["low"], sig, dire, d["vr"],
                        0.01, 0.05, mh, COST_DIRECTIONAL, var_cfg["hold_scale"])
                elif vtype == "breakout":
                    dire_bo = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(
                        d["close"], d["high"], d["low"], sig, dire_bo,
                        0.01, 0.05, mh, COST_DIRECTIONAL)
                else:
                    p, e, x = sim_directional_trades(
                        d["close"], d["high"], d["low"], sig, dire,
                        0.01, 0.05, mh, COST_DIRECTIONAL)
            else:  # S_d8
                if vtype == "mombias":
                    p, e, x = sim_straddle_mombias(
                        d["close"], d["high"], d["low"], sig, d["mom_8"],
                        0.015, 0.05, mh, COST_STRADDLE, var_cfg["mom_weight"])
                elif vtype == "asym_tp":
                    p, e, x = sim_straddle_asym_tp(
                        d["close"], d["high"], d["low"], sig, d["mom_8"],
                        0.015, var_cfg["tp_mom"], var_cfg["tp_counter"],
                        mh, COST_STRADDLE)
                elif vtype == "cooldown":
                    p, e, x = sim_straddle_cooldown(
                        d["close"], d["high"], d["low"], sig,
                        0.015, 0.05, mh, COST_STRADDLE, var_cfg["cooldown_bars"])
                elif vtype == "adapt_hold":
                    p, e, x = sim_straddle_adapt_hold(
                        d["close"], d["high"], d["low"], sig, d["vr"],
                        0.015, 0.05, mh, COST_STRADDLE, var_cfg["hold_scale"])
                else:
                    p, e, x = sim_straddle_trades(
                        d["close"], d["high"], d["low"], sig,
                        0.015, 0.05, mh, COST_STRADDLE)

            # Anti-martingale post-processing
            if vtype == "antimartingale":
                p = apply_antimartingale(p, var_cfg["win_mult"], var_cfg["loss_mult"])

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                all_sym_trades.append((entry_15m, exit_15m, float(p[k])))
                raw_pnls.append(float(p[k]))

        # Dedup: chronological first-come-first-served
        all_sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in all_sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True
                    break
            if not conflict:
                deduped_pnls.append(pnl_val)
                accepted.append((entry_15m, exit_15m))

    return (
        np.array(deduped_pnls) if deduped_pnls else np.empty(0),
        len(raw_pnls),
        len(deduped_pnls),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════


def _compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
    n = len(pnl)
    if n < 2:
        return {"N": n, "PF": 0.0, "EV_bp": 0.0, "t": 0.0, "WR": 0.0,
                "RR": 0.0, "ret_pct": 0.0, "max_dd_pct": 0.0, "tpd": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n * 100
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = abs(float(np.mean(losses))) if len(losses) > 0 else 1e-9
    rr = avg_win / avg_loss if avg_loss > 0 else 999.0
    pf = pf_from_pnl(pnl)
    ev = float(np.mean(pnl)) * 10000
    se = float(np.std(pnl) / np.sqrt(n)) * 10000
    t = ev / se if se > 0 else 0.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnl:
        equity *= (1.0 + p)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    total_ret = (equity - 1.0) * 100
    tpd = n / (data_years * 365) if data_years > 0 else 0.0

    return {
        "N": n, "PF": round(pf, 3), "EV_bp": round(ev, 1),
        "t": round(t, 2), "WR": round(wr, 1), "RR": round(rr, 2),
        "ret_pct": round(total_ret, 1), "max_dd_pct": round(max_dd * 100, 1),
        "tpd": round(tpd, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Workers
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_variant(args: tuple) -> dict:
    """Evaluate one variant. Uses global _G data."""
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_years = _G["years"]
    btc_vr_15m = _G["btc_vr"]
    btc_mom_15m = _G["btc_mom"]
    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    d_pnl = s_pnl = None
    for strat in ["D_mom8", "S_d8"]:
        deduped, n_raw, n_dedup = _run_variant_strat(
            var_name, var_cfg, data_15m, data_1h, data_4h, strat,
            btc_vr_15m, btc_mom_15m)
        if strat == "D_mom8":
            d_pnl = deduped
        else:
            s_pnl = deduped
        metrics = _compute_metrics(deduped, data_years)
        metrics["n_raw"] = n_raw
        metrics["n_dedup"] = n_dedup
        bs = bootstrap_pf(deduped, 1000)
        metrics["P_PF_ge_1"] = bs["P_PF_ge_1"]
        metrics["CI_lo"] = bs["CI_lo"]
        metrics["CI_hi"] = bs["CI_hi"]
        result[strat] = metrics

    # Combined
    combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)
    c_metrics = _compute_metrics(combined, data_years)
    c_bs = bootstrap_pf(combined, 1000)
    c_metrics["P_PF_ge_1"] = c_bs["P_PF_ge_1"]
    c_metrics["CI_lo"] = c_bs["CI_lo"]
    c_metrics["CI_hi"] = c_bs["CI_hi"]
    result["combined"] = c_metrics

    return result


def oos_variant(args: tuple) -> dict:
    """OOS test for one variant (67/33 split). Uses global _G."""
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_years = _G["years"]
    btc_vr_15m = _G["btc_vr"]
    btc_mom_15m = _G["btc_mom"]
    result: dict = {"name": var_name}

    for strat_key in ["D_mom8", "S_d8", "combined"]:
        is_15m: dict = {}
        oos_15m: dict = {}
        is_1h: dict = {}
        oos_1h: dict = {}
        is_4h: dict = {}
        oos_4h: dict = {}

        for sym in data_15m:
            n = len(data_15m[sym]["close"])
            cut = int(n * 0.67)
            for src, is_dst, oos_dst in [
                (data_15m, is_15m, oos_15m),
                (data_1h, is_1h, oos_1h),
                (data_4h, is_4h, oos_4h),
            ]:
                if sym not in src:
                    continue
                d = src[sym]
                n_tf = len(d["close"])
                ratio = n // n_tf if n_tf > 0 else 1
                cut_tf = cut // ratio

                for start, end_b, dst in [(0, cut_tf, is_dst), (cut_tf, n_tf, oos_dst)]:
                    seg_len = end_b - start
                    if seg_len < 100:
                        continue
                    d_seg: dict = {}
                    for key in ("close", "high", "low", "volume"):
                        d_seg[key] = d[key][start:end_b].copy()
                    d_seg["bars"] = seg_len
                    c, h, lo, v = d_seg["close"], d_seg["high"], d_seg["low"], d_seg["volume"]
                    d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
                    d_seg["mom_8"] = compute_momentum(c, 8)
                    d_seg["vol_pct"] = vol_percentile(v, min(200, seg_len // 2))
                    d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)
                    dst[sym] = d_seg

        is_years = data_years * 0.67
        oos_years = data_years * 0.33

        # Split BTC VR/mom too
        btc_cut = int(len(btc_vr_15m) * 0.67)
        is_btc_vr = btc_vr_15m[:btc_cut] if len(btc_vr_15m) > 0 else np.array([])
        oos_btc_vr = btc_vr_15m[btc_cut:] if len(btc_vr_15m) > 0 else np.array([])
        is_btc_mom = btc_mom_15m[:btc_cut] if len(btc_mom_15m) > 0 else np.array([])
        oos_btc_mom = btc_mom_15m[btc_cut:] if len(btc_mom_15m) > 0 else np.array([])

        if strat_key == "combined":
            is_d, _, _ = _run_variant_strat(var_name, var_cfg, is_15m, is_1h, is_4h, "D_mom8", is_btc_vr, is_btc_mom)
            is_s, _, _ = _run_variant_strat(var_name, var_cfg, is_15m, is_1h, is_4h, "S_d8", is_btc_vr, is_btc_mom)
            oos_d, _, _ = _run_variant_strat(var_name, var_cfg, oos_15m, oos_1h, oos_4h, "D_mom8", oos_btc_vr, oos_btc_mom)
            oos_s, _, _ = _run_variant_strat(var_name, var_cfg, oos_15m, oos_1h, oos_4h, "S_d8", oos_btc_vr, oos_btc_mom)
            is_pnl = np.concatenate([is_d, is_s]) if len(is_d) + len(is_s) > 0 else np.empty(0)
            oos_pnl = np.concatenate([oos_d, oos_s]) if len(oos_d) + len(oos_s) > 0 else np.empty(0)
        else:
            is_pnl, _, _ = _run_variant_strat(var_name, var_cfg, is_15m, is_1h, is_4h, strat_key, is_btc_vr, is_btc_mom)
            oos_pnl, _, _ = _run_variant_strat(var_name, var_cfg, oos_15m, oos_1h, oos_4h, strat_key, oos_btc_vr, oos_btc_mom)

        is_m = _compute_metrics(is_pnl, is_years)
        oos_m = _compute_metrics(oos_pnl, oos_years)
        oos_bs = bootstrap_pf(oos_pnl, 1000)
        oos_m["P_PF_ge_1"] = oos_bs["P_PF_ge_1"]
        oos_m["CI_lo"] = oos_bs["CI_lo"]
        oos_m["CI_hi"] = oos_bs["CI_hi"]
        result[strat_key] = {"IS": is_m, "OOS": oos_m}

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 90)
    print("  R33: CREATIVE STRATEGY A/B TEST")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Base: G-ALL deduped (15m+1h+4h)  |  Workers: {NUM_WORKERS}")
    print(f"  Variants: {len(VARIANTS)}")
    print("=" * 90)

    # ── Load data ──
    print("\n  Loading 15m OHLCV data...")
    data_15m = load_15m()
    total_15m = sum(d["bars"] for d in data_15m.values())
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    print(f"  Loaded {len(data_15m)} symbols, {total_15m:,} bars, ~{data_years:.1f} years")

    print("  Resampling 15m → 1h...")
    data_1h = build_resampled(data_15m, 4)
    print(f"  1h: {len(data_1h)} symbols, {sum(d['bars'] for d in data_1h.values()):,} bars")

    print("  Resampling 15m → 4h...")
    data_4h = build_resampled(data_15m, 16)
    print(f"  4h: {len(data_4h)} symbols, {sum(d['bars'] for d in data_4h.values()):,} bars")

    print("  Precomputing indicators...")
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)

    # ── Extract BTC data for cross-asset filters ──
    btc_vr_15m = np.array([])
    btc_mom_15m = np.array([])
    if "BTCUSDT" in data_15m:
        btc_vr_15m = data_15m["BTCUSDT"]["vr"].copy()
        btc_mom_15m = data_15m["BTCUSDT"]["mom_8"].copy()
        print(f"  BTC cross-asset data: {len(btc_vr_15m):,} bars")
    else:
        print("  ⚠ BTCUSDT not in whitelist — C1/C2 filters will be no-ops")

    # ── Set up global data ──
    global _G
    _G = {
        "15m": data_15m, "1h": data_1h, "4h": data_4h,
        "years": data_years,
        "btc_vr": btc_vr_15m, "btc_mom": btc_mom_15m,
    }
    print("  Done.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON")
    print("=" * 90)

    eval_args = [(vn, vc) for vn, vc in sorted(VARIANTS.items())]

    print(f"\n  Running {len(eval_args)} variants across {NUM_WORKERS} workers...")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(evaluate_variant, eval_args):
            results.append(r)
            print(f"    ✓ {r['name']}")

    results.sort(key=lambda x: x["name"])

    # ── Tables ──
    def print_table(label: str, strat_key: str) -> None:
        print(f"\n{'─' * 90}")
        print(f"  {label}")
        print(f"{'─' * 90}")
        header = (f"  {'Variant':<17} {'N':>5} {'PF':>6} {'EV_bp':>7} {'t':>5} "
                  f"{'WR%':>5} {'R:R':>5} {'Ret%':>9} {'DD%':>5} {'P≥1':>4}")
        print(header)
        print("  " + "─" * 78)
        for r in results:
            m = r[strat_key]
            print(f"  {r['name']:<17} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} {m['t']:>5.2f} "
                  f"{m['WR']:>5.1f} {m['RR']:>5.2f} {m['ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
                  f"{m['P_PF_ge_1']:>4.0f}%")

    print_table("FULL-SAMPLE: D_mom8 (deduped)", "D_mom8")
    print_table("FULL-SAMPLE: S_d8 (deduped)", "S_d8")
    print_table("FULL-SAMPLE: Combined D+S (deduped)", "combined")

    # ── Ranking ──
    print(f"\n{'─' * 90}")
    print("  RANKING by Combined EV (bp)")
    print(f"{'─' * 90}")
    baseline_ev = next(r for r in results if r["name"] == "A0-Baseline")["combined"]["EV_bp"]
    ranked = sorted(results, key=lambda x: x["combined"]["EV_bp"], reverse=True)
    for i, r in enumerate(ranked, 1):
        m = r["combined"]
        delta = m["EV_bp"] - baseline_ev
        marker = " ◄" if r["name"] == "A0-Baseline" else ""
        print(f"  #{i:>2} {r['name']:<17} EV={m['EV_bp']:>+7.1f}bp  PF={m['PF']:.3f}  "
              f"N={m['N']:>5}  DD={m['max_dd_pct']:.1f}%  "
              f"Δ_vs_A0={delta:>+6.1f}bp{marker}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: OOS for top variants
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}")
    print("  PHASE 2: OOS TEST (67/33 split) — Top 8 + Baseline")
    print(f"{'=' * 90}")

    oos_names = {"A0-Baseline"}
    for r in ranked[:8]:
        oos_names.add(r["name"])

    oos_args = [(vn, VARIANTS[vn]) for vn in sorted(oos_names)]

    print(f"\n  Running OOS for {len(oos_args)} variants...")
    oos_results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(oos_variant, oos_args):
            oos_results.append(r)
            print(f"    ✓ {r['name']}")

    oos_results.sort(key=lambda x: x["name"])

    def print_oos_table(label: str, strat_key: str) -> None:
        print(f"\n{'─' * 90}")
        print(f"  {label}")
        print(f"{'─' * 90}")
        print(f"  {'Variant':<17} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} {'OOS_N':>6} "
              f"{'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} {'P≥1':>5} {'CI_lo':>6} {'CI_hi':>6}")
        print("  " + "─" * 84)
        for r in oos_results:
            is_m = r[strat_key]["IS"]
            oos_m = r[strat_key]["OOS"]
            print(f"  {r['name']:<17} {is_m['N']:>6} {is_m['PF']:>6.3f} {is_m['EV_bp']:>+7.1f} "
                  f"{oos_m['N']:>6} {oos_m['PF']:>6.3f} {oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
                  f"{oos_m.get('P_PF_ge_1', 0):>4.0f}% "
                  f"{oos_m.get('CI_lo', 0):>6.3f} {oos_m.get('CI_hi', 0):>6.3f}")

    print_oos_table("OOS: Combined D+S", "combined")
    print_oos_table("OOS: D_mom8", "D_mom8")
    print_oos_table("OOS: S_d8", "S_d8")

    # ── OOS Ranking ──
    print(f"\n{'─' * 90}")
    print("  OOS RANKING by Combined OOS EV (bp)")
    print(f"{'─' * 90}")
    oos_baseline = next(r for r in oos_results if r["name"] == "A0-Baseline")
    bl_oos_ev = oos_baseline["combined"]["OOS"]["EV_bp"]
    oos_ranked = sorted(oos_results, key=lambda x: x["combined"]["OOS"]["EV_bp"], reverse=True)
    for i, r in enumerate(oos_ranked, 1):
        oos_m = r["combined"]["OOS"]
        delta = oos_m["EV_bp"] - bl_oos_ev
        marker = " ◄" if r["name"] == "A0-Baseline" else ""
        print(f"  #{i:>2} {r['name']:<17} OOS_EV={oos_m['EV_bp']:>+7.1f}bp  "
              f"PF={oos_m['PF']:.3f}  N={oos_m['N']:>5}  t={oos_m['t']:.2f}  "
              f"DD={oos_m['max_dd_pct']:.1f}%  Δ={delta:>+6.1f}bp{marker}")

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 90}")
    print(f"  R33 COMPLETE  |  {elapsed:.0f}s  |  {len(VARIANTS)} variants  |  {NUM_WORKERS} workers")
    print(f"{'=' * 90}")

    best_oos = oos_ranked[0]
    bm = best_oos["combined"]["OOS"]
    print(f"\n  Best OOS Combined: {best_oos['name']}")
    print(f"    N={bm['N']}  PF={bm['PF']:.3f}  EV={bm['EV_bp']:>+.1f}bp  t={bm['t']:.2f}  "
          f"WR={bm['WR']:.1f}%  DD={bm['max_dd_pct']:.1f}%")

    bl_m = oos_baseline["combined"]["OOS"]
    print(f"\n  Baseline OOS: N={bl_m['N']}  PF={bl_m['PF']:.3f}  EV={bl_m['EV_bp']:>+.1f}bp  "
          f"t={bl_m['t']:.2f}  DD={bl_m['max_dd_pct']:.1f}%")

    print(f"\n  Delta: ΔEV={bm['EV_bp'] - bl_m['EV_bp']:>+.1f}bp  "
          f"ΔPF={bm['PF'] - bl_m['PF']:>+.3f}")


if __name__ == "__main__":
    main()
