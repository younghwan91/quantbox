#!/usr/bin/env python3
"""
R32: Exit / Risk Management A/B Test
======================================
Base strategy: G-ALL deduped (15m+1h+4h) from R31.
Tests exit techniques from rubber band era applied to vol straddle:

VARIANTS:
  A-Baseline     : Current G-ALL deduped (SL=1%/1.5%, TP=5%, fixed timeout)
  B-ATR-SL       : ATR(14)-adaptive SL, clamped [0.3%, 3.0%]
  C-Trail-10     : Trailing stop 10% of max favorable excursion
  C-Trail-15     : Trailing stop 15%
  C-Trail-20     : Trailing stop 20%
  D-Partial      : D_mom8: 50% at TP1=2%, runner with BE floor + trail
  E1-SL-050      : SL sweep: 0.50%
  E2-SL-075      : SL sweep: 0.75%
  E3-SL-125      : SL sweep: 1.25%
  E4-SL-200      : SL sweep: 2.00%
  F1-TP-030      : TP sweep: 3%
  F2-TP-070      : TP sweep: 7%
  F3-TP-100      : TP sweep: 10%
  G-ATR-Trail    : ATR-adaptive SL + trailing stop (best of B × C)
  H1-Hold-Short  : Max hold ×0.5  (faster timeout)
  H2-Hold-Long   : Max hold ×1.5  (longer patience)
  H3-Hold-2x     : Max hold ×2.0  (much longer)

All variants use 4-core parallel evaluation.
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
    vol_percentile,
)

NUM_WORKERS = 4

# Global data (set once in main, inherited by fork)
_G_DATA_15m: dict = {}
_G_DATA_1h: dict = {}
_G_DATA_4h: dict = {}
_G_DATA_YEARS: float = 0.0


def _init_worker(d15: dict, d1h: dict, d4h: dict, dy: float) -> None:
    """Initialize worker process globals (called once per worker via fork)."""
    global _G_DATA_15m, _G_DATA_1h, _G_DATA_4h, _G_DATA_YEARS
    _G_DATA_15m = d15
    _G_DATA_1h = d1h
    _G_DATA_4h = d4h
    _G_DATA_YEARS = dy


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Extended sim functions for exit variants
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_atr_sl(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, atr: np.ndarray,
    atr_mult: float, sl_min: float, sl_max: float,
    tp: float, max_hold: int, cost: float,
) -> np.ndarray:
    """Straddle with ATR-adaptive SL. TP is fixed."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
        # ATR-adaptive SL
        raw_sl = atr[i] * atr_mult if not np.isnan(atr[i]) else 0.01
        sl = min(max(raw_sl, sl_min), sl_max)

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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_atr_sl(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray, atr: np.ndarray,
    atr_mult: float, sl_min: float, sl_max: float,
    tp: float, max_hold: int, cost: float,
) -> np.ndarray:
    """Directional with ATR-adaptive SL."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
        raw_sl = atr[i] * atr_mult if not np.isnan(atr[i]) else 0.01
        sl = min(max(raw_sl, sl_min), sl_max)
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_straddle_trail(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int,
    trail_pct: float, cost: float,
) -> np.ndarray:
    """Straddle with trailing stop on each leg.
    Once a leg moves trail_pct in favor, a trailing stop is activated
    that follows the best price at trail_pct distance.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
        # Long leg
        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        l_best = entry
        l_trail_active = False
        # Short leg
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        s_best = entry
        s_trail_active = False

        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                # Track best high for long leg
                if high[j] > l_best:
                    l_best = high[j]
                # Activate trailing after trail_pct gain
                if l_best >= entry * (1.0 + trail_pct):
                    l_trail_active = True
                    l_trail_stop = l_best * (1.0 - trail_pct)
                    if l_trail_stop > l_sl:
                        l_sl = l_trail_stop
                # Check exits
                if low[j] <= l_sl:
                    le = (l_sl - entry) / entry
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                # Track best low for short leg
                if low[j] < s_best:
                    s_best = low[j]
                # Activate trailing
                if s_best <= entry * (1.0 - trail_pct):
                    s_trail_active = True
                    s_trail_stop = s_best * (1.0 + trail_pct)
                    if s_trail_stop < s_sl:
                        s_sl = s_trail_stop
                # Check exits
                if high[j] >= s_sl:
                    se = (entry - s_sl) / entry
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trail(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int,
    trail_pct: float, cost: float,
) -> np.ndarray:
    """Directional with trailing stop."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
        best = entry
        trail_active = False

        for j in range(i + 1, end):
            if d == 1:
                if high[j] > best:
                    best = high[j]
                # Activate trail
                if best >= entry * (1.0 + trail_pct):
                    trail_active = True
                    trail_stop = best * (1.0 - trail_pct)
                    # Check trail stop (ensures at least breakeven or better)
                    if trail_stop > entry and low[j] <= trail_stop:
                        trade_pnl = (trail_stop - entry) / entry
                        eb = j
                        break
                # Fixed SL
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                # Fixed TP (cap)
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if low[j] < best:
                    best = low[j]
                if best <= entry * (1.0 - trail_pct):
                    trail_active = True
                    trail_stop = best * (1.0 + trail_pct)
                    if trail_stop < entry and high[j] >= trail_stop:
                        trade_pnl = (entry - trail_stop) / entry
                        eb = j
                        break
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_partial(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp1: float, tp2: float,
    partial_pct: float, max_hold: int,
    trail_pct: float, cost: float,
) -> np.ndarray:
    """Directional with partial exit: partial_pct at TP1, runner to TP2 with
    breakeven floor + trailing stop.
    PnL = partial_pct × leg1_pnl + (1-partial_pct) × leg2_pnl - cost
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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

        leg1_pnl = 0.0   # partial close PnL
        leg2_pnl = 0.0   # runner PnL
        tp1_hit = False
        best = entry

        for j in range(i + 1, end):
            if d == 1:
                if high[j] > best:
                    best = high[j]
                if not tp1_hit:
                    # Phase 1: SL or TP1
                    if low[j] <= entry * (1.0 - sl):
                        leg1_pnl = -sl
                        leg2_pnl = -sl
                        eb = j
                        break
                    if high[j] >= entry * (1.0 + tp1):
                        leg1_pnl = tp1
                        tp1_hit = True
                        best = entry * (1.0 + tp1)  # reset best to TP1
                else:
                    # Phase 2: Runner — BE floor + trail
                    runner_stop = entry  # breakeven floor
                    if best >= entry * (1.0 + trail_pct + tp1):
                        trail_stop = best * (1.0 - trail_pct)
                        if trail_stop > runner_stop:
                            runner_stop = trail_stop
                    if low[j] <= runner_stop:
                        leg2_pnl = (runner_stop - entry) / entry
                        eb = j
                        break
                    if high[j] >= entry * (1.0 + tp2):
                        leg2_pnl = tp2
                        eb = j
                        break
            else:  # SHORT
                if low[j] < best:
                    best = low[j]
                if not tp1_hit:
                    if high[j] >= entry * (1.0 + sl):
                        leg1_pnl = -sl
                        leg2_pnl = -sl
                        eb = j
                        break
                    if low[j] <= entry * (1.0 - tp1):
                        leg1_pnl = tp1
                        tp1_hit = True
                        best = entry * (1.0 - tp1)
                else:
                    runner_stop = entry
                    if best <= entry * (1.0 - trail_pct - tp1):
                        trail_stop = best * (1.0 + trail_pct)
                        if trail_stop < runner_stop:
                            runner_stop = trail_stop
                    if high[j] >= runner_stop:
                        leg2_pnl = (entry - runner_stop) / entry
                        eb = j
                        break
                    if low[j] <= entry * (1.0 - tp2):
                        leg2_pnl = tp2
                        eb = j
                        break
        else:
            # Timeout
            if d == 1:
                timeout_pnl = (close[eb] - entry) / entry
            else:
                timeout_pnl = (entry - close[eb]) / entry
            if not tp1_hit:
                leg1_pnl = timeout_pnl
                leg2_pnl = timeout_pnl
            else:
                leg2_pnl = timeout_pnl

        total = partial_pct * leg1_pnl + (1.0 - partial_pct) * leg2_pnl - cost
        if cnt < cap:
            pnl[cnt] = total
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_straddle_atr_trail(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, atr: np.ndarray,
    atr_mult: float, sl_min: float, sl_max: float,
    tp: float, max_hold: int, trail_pct: float, cost: float,
) -> np.ndarray:
    """Straddle with ATR-adaptive SL + trailing stop."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
        raw_sl = atr[i] * atr_mult if not np.isnan(atr[i]) else 0.01
        sl = min(max(raw_sl, sl_min), sl_max)

        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        l_best = entry
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        s_best = entry
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if high[j] > l_best:
                    l_best = high[j]
                if l_best >= entry * (1.0 + trail_pct):
                    l_trail = l_best * (1.0 - trail_pct)
                    if l_trail > l_sl:
                        l_sl = l_trail
                if low[j] <= l_sl:
                    le = (l_sl - entry) / entry
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if low[j] < s_best:
                    s_best = low[j]
                if s_best <= entry * (1.0 - trail_pct):
                    s_trail = s_best * (1.0 + trail_pct)
                    if s_trail < s_sl:
                        s_sl = s_trail
                if high[j] >= s_sl:
                    se = (entry - s_sl) / entry
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_atr_trail(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray, atr: np.ndarray,
    atr_mult: float, sl_min: float, sl_max: float,
    tp: float, max_hold: int, trail_pct: float, cost: float,
) -> np.ndarray:
    """Directional with ATR-adaptive SL + trailing stop."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
        raw_sl = atr[i] * atr_mult if not np.isnan(atr[i]) else 0.01
        sl = min(max(raw_sl, sl_min), sl_max)
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        best = entry

        for j in range(i + 1, end):
            if d == 1:
                if high[j] > best:
                    best = high[j]
                trail_stop = -1.0
                if best >= entry * (1.0 + trail_pct):
                    trail_stop = best * (1.0 - trail_pct)
                # Trail stop (only if above entry = profitable)
                if trail_stop > entry and low[j] <= trail_stop:
                    trade_pnl = (trail_stop - entry) / entry
                    eb = j
                    break
                # Fixed ATR SL
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if low[j] < best:
                    best = low[j]
                trail_stop = 1e18
                if best <= entry * (1.0 - trail_pct):
                    trail_stop = best * (1.0 + trail_pct)
                if trail_stop < entry and high[j] >= trail_stop:
                    trade_pnl = (entry - trail_stop) / entry
                    eb = j
                    break
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade-level sims (with entry/exit indices for dedup)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades_v2(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, atr: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    use_atr_sl: bool, atr_mult: float, sl_min: float, sl_max: float,
    use_trail: bool, trail_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Universal straddle sim with optional ATR SL + trailing. Returns (pnl, entries, exits)."""
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    ent_arr = np.empty(cap, dtype=np.int64)
    ext_arr = np.empty(cap, dtype=np.int64)
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
        # Determine SL
        if use_atr_sl and not np.isnan(atr[i]):
            raw_sl = atr[i] * atr_mult
            cur_sl = min(max(raw_sl, sl_min), sl_max)
        else:
            cur_sl = sl

        l_sl_price = entry * (1.0 - cur_sl)
        l_tp_price = entry * (1.0 + tp)
        s_sl_price = entry * (1.0 + cur_sl)
        s_tp_price = entry * (1.0 - tp)
        l_best = entry
        s_best = entry
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if high[j] > l_best:
                    l_best = high[j]
                if use_trail and l_best >= entry * (1.0 + trail_pct):
                    l_trail = l_best * (1.0 - trail_pct)
                    if l_trail > l_sl_price:
                        l_sl_price = l_trail
                if low[j] <= l_sl_price:
                    le = (l_sl_price - entry) / entry
                    la = False
                elif high[j] >= l_tp_price:
                    le = tp
                    la = False
            if sa:
                if low[j] < s_best:
                    s_best = low[j]
                if use_trail and s_best <= entry * (1.0 - trail_pct):
                    s_trail = s_best * (1.0 + trail_pct)
                    if s_trail < s_sl_price:
                        s_sl_price = s_trail
                if high[j] >= s_sl_price:
                    se = (entry - s_sl_price) / entry
                    sa = False
                elif low[j] <= s_tp_price:
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
            pnl_arr[cnt] = net
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trades_v2(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray, atr: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    use_atr_sl: bool, atr_mult: float, sl_min: float, sl_max: float,
    use_trail: bool, trail_pct: float,
    use_partial: bool, tp1: float, partial_pct: float, tp2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Universal directional sim with optional ATR SL + trail + partial exit.
    Returns (pnl, entries, exits).
    """
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    ent_arr = np.empty(cap, dtype=np.int64)
    ext_arr = np.empty(cap, dtype=np.int64)
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
        # Determine SL
        if use_atr_sl and not np.isnan(atr[i]):
            raw_sl = atr[i] * atr_mult
            cur_sl = min(max(raw_sl, sl_min), sl_max)
        else:
            cur_sl = sl
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        best = entry
        tp1_hit = False
        leg1_pnl = 0.0
        trade_pnl = 0.0
        exited = False

        for j in range(i + 1, end):
            if d == 1:
                if high[j] > best:
                    best = high[j]
                # --- Partial exit mode ---
                if use_partial and not tp1_hit:
                    if low[j] <= entry * (1.0 - cur_sl):
                        trade_pnl = -cur_sl - cost
                        eb = j
                        exited = True
                        break
                    if high[j] >= entry * (1.0 + tp1):
                        leg1_pnl = tp1
                        tp1_hit = True
                        best = entry * (1.0 + tp1)
                        continue
                if use_partial and tp1_hit:
                    # Runner: BE floor + trail
                    runner_stop = entry
                    if use_trail and best >= entry * (1.0 + trail_pct + tp1):
                        t_stop = best * (1.0 - trail_pct)
                        if t_stop > runner_stop:
                            runner_stop = t_stop
                    if low[j] <= runner_stop:
                        leg2_pnl = (runner_stop - entry) / entry
                        trade_pnl = partial_pct * leg1_pnl + (1.0 - partial_pct) * leg2_pnl - cost
                        eb = j
                        exited = True
                        break
                    if high[j] >= entry * (1.0 + tp2):
                        leg2_pnl = tp2
                        trade_pnl = partial_pct * leg1_pnl + (1.0 - partial_pct) * leg2_pnl - cost
                        eb = j
                        exited = True
                        break
                    continue
                # --- Non-partial mode ---
                # Trail
                if use_trail:
                    if best >= entry * (1.0 + trail_pct):
                        t_stop = best * (1.0 - trail_pct)
                        if t_stop > entry and low[j] <= t_stop:
                            trade_pnl = (t_stop - entry) / entry - cost
                            eb = j
                            exited = True
                            break
                # SL
                if low[j] <= entry * (1.0 - cur_sl):
                    trade_pnl = -cur_sl - cost
                    eb = j
                    exited = True
                    break
                # TP
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp - cost
                    eb = j
                    exited = True
                    break
            else:  # SHORT
                if low[j] < best:
                    best = low[j]
                if use_partial and not tp1_hit:
                    if high[j] >= entry * (1.0 + cur_sl):
                        trade_pnl = -cur_sl - cost
                        eb = j
                        exited = True
                        break
                    if low[j] <= entry * (1.0 - tp1):
                        leg1_pnl = tp1
                        tp1_hit = True
                        best = entry * (1.0 - tp1)
                        continue
                if use_partial and tp1_hit:
                    runner_stop = entry
                    if use_trail and best <= entry * (1.0 - trail_pct - tp1):
                        t_stop = best * (1.0 + trail_pct)
                        if t_stop < runner_stop:
                            runner_stop = t_stop
                    if high[j] >= runner_stop:
                        leg2_pnl = (entry - runner_stop) / entry
                        trade_pnl = partial_pct * leg1_pnl + (1.0 - partial_pct) * leg2_pnl - cost
                        eb = j
                        exited = True
                        break
                    if low[j] <= entry * (1.0 - tp2):
                        leg2_pnl = tp2
                        trade_pnl = partial_pct * leg1_pnl + (1.0 - partial_pct) * leg2_pnl - cost
                        eb = j
                        exited = True
                        break
                    continue
                if use_trail:
                    if best <= entry * (1.0 - trail_pct):
                        t_stop = best * (1.0 + trail_pct)
                        if t_stop < entry and high[j] >= t_stop:
                            trade_pnl = (entry - t_stop) / entry - cost
                            eb = j
                            exited = True
                            break
                if high[j] >= entry * (1.0 + cur_sl):
                    trade_pnl = -cur_sl - cost
                    eb = j
                    exited = True
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp - cost
                    eb = j
                    exited = True
                    break

        if not exited:
            # Timeout
            if d == 1:
                timeout_pnl = (close[eb] - entry) / entry
            else:
                timeout_pnl = (entry - close[eb]) / entry
            if use_partial and tp1_hit:
                trade_pnl = partial_pct * leg1_pnl + (1.0 - partial_pct) * timeout_pnl - cost
            else:
                trade_pnl = timeout_pnl - cost

        if cnt < cap:
            pnl_arr[cnt] = trade_pnl
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════


# G-ALL TF configs: (label, ratio, warmup, max_hold_D, max_hold_S, min_dur_S)
TF_15m = ("15m", 1, WARMUP_15m, 192, 384, 8)
TF_1h = ("1h", 4, WARMUP_1h, 48, 96, 8)
TF_4h = ("4h", 16, WARMUP_4h, 12, 24, 8)
G_ALL_TFS = [TF_15m, TF_1h, TF_4h]


# Each variant: dict with exit params for D_mom8 and S_d8
# D params: sl, tp, use_atr_sl, atr_mult, sl_min, sl_max, use_trail, trail_pct,
#           use_partial, tp1, partial_pct, tp2
# S params: sl, tp, use_atr_sl, atr_mult, sl_min, sl_max, use_trail, trail_pct


def _d_params(
    sl: float = 0.01, tp: float = 0.05,
    use_atr_sl: bool = False, atr_mult: float = 1.5, sl_min: float = 0.003, sl_max: float = 0.03,
    use_trail: bool = False, trail_pct: float = 0.015,
    use_partial: bool = False, tp1: float = 0.02, partial_pct: float = 0.5, tp2: float = 0.08,
) -> dict:
    return dict(
        sl=sl, tp=tp,
        use_atr_sl=use_atr_sl, atr_mult=atr_mult, sl_min=sl_min, sl_max=sl_max,
        use_trail=use_trail, trail_pct=trail_pct,
        use_partial=use_partial, tp1=tp1, partial_pct=partial_pct, tp2=tp2,
    )


def _s_params(
    sl: float = 0.015, tp: float = 0.05,
    use_atr_sl: bool = False, atr_mult: float = 1.5, sl_min: float = 0.003, sl_max: float = 0.03,
    use_trail: bool = False, trail_pct: float = 0.015,
) -> dict:
    return dict(
        sl=sl, tp=tp,
        use_atr_sl=use_atr_sl, atr_mult=atr_mult, sl_min=sl_min, sl_max=sl_max,
        use_trail=use_trail, trail_pct=trail_pct,
    )


VARIANTS: dict[str, dict] = {
    "A-Baseline": {
        "desc": "Current G-ALL deduped (SL=1%/1.5%, TP=5%, fixed timeout)",
        "D": _d_params(),
        "S": _s_params(),
        "hold_mult": 1.0,
    },
    "B-ATR-SL": {
        "desc": "ATR(14)-adaptive SL, clamped [0.3%, 3.0%]",
        "D": _d_params(use_atr_sl=True, atr_mult=1.5, sl_min=0.003, sl_max=0.03),
        "S": _s_params(use_atr_sl=True, atr_mult=1.5, sl_min=0.003, sl_max=0.03),
        "hold_mult": 1.0,
    },
    "C1-Trail-10": {
        "desc": "Trailing stop 1.0%",
        "D": _d_params(use_trail=True, trail_pct=0.01),
        "S": _s_params(use_trail=True, trail_pct=0.01),
        "hold_mult": 1.0,
    },
    "C2-Trail-15": {
        "desc": "Trailing stop 1.5%",
        "D": _d_params(use_trail=True, trail_pct=0.015),
        "S": _s_params(use_trail=True, trail_pct=0.015),
        "hold_mult": 1.0,
    },
    "C3-Trail-20": {
        "desc": "Trailing stop 2.0%",
        "D": _d_params(use_trail=True, trail_pct=0.02),
        "S": _s_params(use_trail=True, trail_pct=0.02),
        "hold_mult": 1.0,
    },
    "D-Partial": {
        "desc": "D_mom8: 50% at TP1=2%, runner to TP2=8% with BE+trail",
        "D": _d_params(use_partial=True, tp1=0.02, partial_pct=0.5, tp2=0.08,
                        use_trail=True, trail_pct=0.015),
        "S": _s_params(),  # Straddle unchanged (already hedged)
        "hold_mult": 1.0,
    },
    "E1-SL-050": {
        "desc": "SL sweep: D=0.50%, S=0.75%",
        "D": _d_params(sl=0.005),
        "S": _s_params(sl=0.0075),
        "hold_mult": 1.0,
    },
    "E2-SL-075": {
        "desc": "SL sweep: D=0.75%, S=1.0%",
        "D": _d_params(sl=0.0075),
        "S": _s_params(sl=0.01),
        "hold_mult": 1.0,
    },
    "E3-SL-125": {
        "desc": "SL sweep: D=1.25%, S=1.75%",
        "D": _d_params(sl=0.0125),
        "S": _s_params(sl=0.0175),
        "hold_mult": 1.0,
    },
    "E4-SL-200": {
        "desc": "SL sweep: D=2.0%, S=2.5%",
        "D": _d_params(sl=0.02),
        "S": _s_params(sl=0.025),
        "hold_mult": 1.0,
    },
    "F1-TP-030": {
        "desc": "TP sweep: 3%",
        "D": _d_params(tp=0.03),
        "S": _s_params(tp=0.03),
        "hold_mult": 1.0,
    },
    "F2-TP-070": {
        "desc": "TP sweep: 7%",
        "D": _d_params(tp=0.07),
        "S": _s_params(tp=0.07),
        "hold_mult": 1.0,
    },
    "F3-TP-100": {
        "desc": "TP sweep: 10%",
        "D": _d_params(tp=0.10),
        "S": _s_params(tp=0.10),
        "hold_mult": 1.0,
    },
    "G-ATR-Trail": {
        "desc": "ATR-adaptive SL + trailing stop 1.5%",
        "D": _d_params(use_atr_sl=True, atr_mult=1.5, sl_min=0.003, sl_max=0.03,
                        use_trail=True, trail_pct=0.015),
        "S": _s_params(use_atr_sl=True, atr_mult=1.5, sl_min=0.003, sl_max=0.03,
                        use_trail=True, trail_pct=0.015),
        "hold_mult": 1.0,
    },
    "H1-Hold-Half": {
        "desc": "Max hold ×0.5 (faster timeout)",
        "D": _d_params(),
        "S": _s_params(),
        "hold_mult": 0.5,
    },
    "H2-Hold-1.5x": {
        "desc": "Max hold ×1.5 (longer patience)",
        "D": _d_params(),
        "S": _s_params(),
        "hold_mult": 1.5,
    },
    "H3-Hold-2x": {
        "desc": "Max hold ×2.0 (much longer)",
        "D": _d_params(),
        "S": _s_params(),
        "hold_mult": 2.0,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION ENGINE (with dedup)
# ══════════════════════════════════════════════════════════════════════════════


def _precompute_atr(data: dict, period: int = 14) -> None:
    """Add ATR_pct array to each symbol's data dict."""
    for sym, d in data.items():
        if "atr_pct" not in d:
            d["atr_pct"] = _atr_pct(d["close"], d["high"], d["low"], period)


def _run_variant_for_strat(
    var_name: str, var_cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Run one variant for one strategy with dedup. Returns (deduped_pnl, raw_pnl, n_raw, n_dedup)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    params = var_cfg["D"] if strat == "D_mom8" else var_cfg["S"]
    hold_mult = var_cfg.get("hold_mult", 1.0)
    raw_pnls: list[float] = []
    deduped_pnls: list[float] = []

    for sym in sorted(data_15m.keys()):
        all_sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in G_ALL_TFS:
            d_tf = tf_data_map[ratio]
            if sym not in d_tf:
                continue
            d = d_tf[sym]

            mh = mh_d if strat == "D_mom8" else mh_s
            mh = max(1, int(mh * hold_mult))
            md = 8 if strat == "D_mom8" else min_dur

            # Create ATR array (dummy if not needed)
            atr = d.get("atr_pct", np.full(len(d["close"]), np.nan))

            if strat == "D_mom8":
                sig = gen_vr_signal(d["vr"], 0.45, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                pnl, ent, ext = sim_directional_trades_v2(
                    d["close"], d["high"], d["low"], sig, dire, atr,
                    params["sl"], params["tp"], mh, COST_DIRECTIONAL,
                    params["use_atr_sl"], params["atr_mult"],
                    params["sl_min"], params["sl_max"],
                    params["use_trail"], params["trail_pct"],
                    params["use_partial"], params["tp1"],
                    params["partial_pct"], params["tp2"],
                )
            else:
                sig = gen_vr_dur_signal(d["vr"], 0.55, md, warmup)
                pnl, ent, ext = sim_straddle_trades_v2(
                    d["close"], d["high"], d["low"], sig, atr,
                    params["sl"], params["tp"], mh, COST_STRADDLE,
                    params["use_atr_sl"], params["atr_mult"],
                    params["sl_min"], params["sl_max"],
                    params["use_trail"], params["trail_pct"],
                )

            for k in range(len(pnl)):
                entry_15m = int(ent[k]) * ratio
                exit_15m = int(ext[k]) * ratio + (ratio - 1)
                all_sym_trades.append((entry_15m, exit_15m, float(pnl[k])))
                raw_pnls.append(float(pnl[k]))

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
        np.array(raw_pnls) if raw_pnls else np.empty(0),
        len(raw_pnls),
        len(deduped_pnls),
    )


def _compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
    """Compute comprehensive metrics from PnL array."""
    n = len(pnl)
    if n < 2:
        return {"N": n, "PF": 0.0, "EV_bp": 0.0, "t": 0.0, "WR": 0.0,
                "RR": 0.0, "ret_pct": 0.0, "max_dd_pct": 0.0, "trades_per_day": 0.0}
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

    # Cumulative equity
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
    trades_per_day = n / (data_years * 365) if data_years > 0 else 0.0

    return {
        "N": n, "PF": round(pf, 3), "EV_bp": round(ev, 1),
        "t": round(t, 2), "WR": round(wr, 1), "RR": round(rr, 2),
        "ret_pct": round(total_ret, 1), "max_dd_pct": round(max_dd * 100, 1),
        "trades_per_day": round(trades_per_day, 2),
    }


def evaluate_variant(args: tuple) -> dict:
    """Evaluate one variant (called by ProcessPoolExecutor).
    args = (var_name, var_cfg)
    Uses global _G_DATA_* set by _init_worker.
    """
    var_name, var_cfg = args
    data_15m = _G_DATA_15m
    data_1h = _G_DATA_1h
    data_4h = _G_DATA_4h
    data_years = _G_DATA_YEARS
    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    d_deduped = None
    s_deduped = None
    for strat in ["D_mom8", "S_d8"]:
        deduped, raw, n_raw, n_dedup = _run_variant_for_strat(
            var_name, var_cfg, data_15m, data_1h, data_4h, strat)
        if strat == "D_mom8":
            d_deduped = deduped
        else:
            s_deduped = deduped
        metrics = _compute_metrics(deduped, data_years)
        metrics["n_raw"] = n_raw
        metrics["n_dedup"] = n_dedup
        bs = bootstrap_pf(deduped, 1000)
        metrics["P_PF_ge_1"] = bs["P_PF_ge_1"]
        metrics["CI_lo"] = bs["CI_lo"]
        metrics["CI_hi"] = bs["CI_hi"]
        result[strat] = metrics

    # Combined (reuse already-computed deduped PnLs)
    combined = np.concatenate([d_deduped, s_deduped]) if len(d_deduped) + len(s_deduped) > 0 else np.empty(0)
    c_metrics = _compute_metrics(combined, data_years)
    c_bs = bootstrap_pf(combined, 1000)
    c_metrics["P_PF_ge_1"] = c_bs["P_PF_ge_1"]
    c_metrics["CI_lo"] = c_bs["CI_lo"]
    c_metrics["CI_hi"] = c_bs["CI_hi"]
    result["combined"] = c_metrics

    return result


def oos_variant(args: tuple) -> dict:
    """OOS test for a variant (67/33 split). Returns IS and OOS metrics."""
    var_name, var_cfg = args
    data_15m = _G_DATA_15m
    data_1h = _G_DATA_1h
    data_4h = _G_DATA_4h
    data_years = _G_DATA_YEARS
    result: dict = {"name": var_name}

    for strat in ["D_mom8", "S_d8", "combined"]:
        # Split each symbol's data at 67%
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
                # Determine ratio from data
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
                    d_seg["atr_pct"] = _atr_pct(c, h, lo, 14)
                    dst[sym] = d_seg

        is_years = data_years * 0.67
        oos_years = data_years * 0.33

        if strat == "combined":
            # Run both strats and combine
            is_d, _, _, _ = _run_variant_for_strat(var_name, var_cfg, is_15m, is_1h, is_4h, "D_mom8")
            is_s, _, _, _ = _run_variant_for_strat(var_name, var_cfg, is_15m, is_1h, is_4h, "S_d8")
            oos_d, _, _, _ = _run_variant_for_strat(var_name, var_cfg, oos_15m, oos_1h, oos_4h, "D_mom8")
            oos_s, _, _, _ = _run_variant_for_strat(var_name, var_cfg, oos_15m, oos_1h, oos_4h, "S_d8")
            is_pnl = np.concatenate([is_d, is_s]) if len(is_d) + len(is_s) > 0 else np.empty(0)
            oos_pnl = np.concatenate([oos_d, oos_s]) if len(oos_d) + len(oos_s) > 0 else np.empty(0)
        else:
            is_pnl, _, _, _ = _run_variant_for_strat(
                var_name, var_cfg, is_15m, is_1h, is_4h, strat)
            oos_pnl, _, _, _ = _run_variant_for_strat(
                var_name, var_cfg, oos_15m, oos_1h, oos_4h, strat)

        is_m = _compute_metrics(is_pnl, is_years)
        oos_m = _compute_metrics(oos_pnl, oos_years)
        oos_bs = bootstrap_pf(oos_pnl, 1000)
        oos_m["P_PF_ge_1"] = oos_bs["P_PF_ge_1"]
        oos_m["CI_lo"] = oos_bs["CI_lo"]
        oos_m["CI_hi"] = oos_bs["CI_hi"]

        result[strat] = {"IS": is_m, "OOS": oos_m}

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 90)
    print("  R32: EXIT / RISK MANAGEMENT A/B TEST")
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

    print("  Precomputing indicators + ATR(14)...")
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)
    _precompute_atr(data_15m)
    _precompute_atr(data_1h)
    _precompute_atr(data_4h)

    # Set globals for main process too
    global _G_DATA_15m, _G_DATA_1h, _G_DATA_4h, _G_DATA_YEARS
    _G_DATA_15m = data_15m
    _G_DATA_1h = data_1h
    _G_DATA_4h = data_4h
    _G_DATA_YEARS = data_years

    print("  Done.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample evaluation (parallel)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON (G-ALL deduped, all variants)")
    print("=" * 90)

    # Prepare args (only variant config, data is in globals)
    eval_args = [
        (vn, vc)
        for vn, vc in sorted(VARIANTS.items())
    ]

    # Run in parallel (use initializer to set globals in workers via fork)
    print(f"\n  Running {len(eval_args)} variants across {NUM_WORKERS} workers...")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(data_15m, data_1h, data_4h, data_years),
    ) as executor:
        for r in executor.map(evaluate_variant, eval_args):
            results.append(r)
            print(f"    ✓ {r['name']}")

    # Sort by name
    results.sort(key=lambda x: x["name"])

    # ── Summary table ──
    print("\n" + "─" * 90)
    print("  FULL-SAMPLE RESULTS: D_mom8 (deduped)")
    print("─" * 90)
    header = (f"  {'Variant':<16} {'N':>5} {'PF':>6} {'EV_bp':>7} {'t':>5} "
              f"{'WR%':>5} {'R:R':>5} {'Ret%':>8} {'DD%':>5} {'P≥1':>4}")
    print(header)
    print("  " + "─" * 78)
    for r in results:
        m = r["D_mom8"]
        print(f"  {r['name']:<16} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} {m['t']:>5.2f} "
              f"{m['WR']:>5.1f} {m['RR']:>5.2f} {m['ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} "
              f"{m['P_PF_ge_1']:>4.0f}%")

    print("\n" + "─" * 90)
    print("  FULL-SAMPLE RESULTS: S_d8 (deduped)")
    print("─" * 90)
    print(header)
    print("  " + "─" * 78)
    for r in results:
        m = r["S_d8"]
        print(f"  {r['name']:<16} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} {m['t']:>5.2f} "
              f"{m['WR']:>5.1f} {m['RR']:>5.2f} {m['ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} "
              f"{m['P_PF_ge_1']:>4.0f}%")

    print("\n" + "─" * 90)
    print("  FULL-SAMPLE RESULTS: Combined D+S (deduped)")
    print("─" * 90)
    print(header)
    print("  " + "─" * 78)
    for r in results:
        m = r["combined"]
        print(f"  {r['name']:<16} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} {m['t']:>5.2f} "
              f"{m['WR']:>5.1f} {m['RR']:>5.2f} {m['ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} "
              f"{m.get('P_PF_ge_1', 0):>4.0f}%")

    # ── Ranking by EV ──
    print("\n" + "─" * 90)
    print("  RANKING by Combined EV (bp)")
    print("─" * 90)
    ranked = sorted(results, key=lambda x: x["combined"]["EV_bp"], reverse=True)
    for i, r in enumerate(ranked, 1):
        m = r["combined"]
        delta = m["EV_bp"] - results[0]["combined"]["EV_bp"]  # vs A-Baseline
        print(f"  #{i:>2} {r['name']:<16} EV={m['EV_bp']:>+7.1f}bp  PF={m['PF']:.3f}  "
              f"N={m['N']:>5}  Ret={m['ret_pct']:>+8.1f}%  DD={m['max_dd_pct']:.1f}%  "
              f"Δ_vs_A={delta:>+6.1f}bp")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: OOS test (top 6 + baseline)
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("  PHASE 2: OOS TEST (67/33 split) — Top 6 + Baseline")
    print("=" * 90)

    # Always include baseline + top 6 by combined EV
    oos_names = {"A-Baseline"}
    for r in ranked[:6]:
        oos_names.add(r["name"])

    oos_args = [
        (vn, VARIANTS[vn])
        for vn in sorted(oos_names)
    ]

    print(f"\n  Running OOS for {len(oos_args)} variants...")
    oos_results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(data_15m, data_1h, data_4h, data_years),
    ) as executor:
        for r in executor.map(oos_variant, oos_args):
            oos_results.append(r)
            print(f"    ✓ {r['name']}")

    oos_results.sort(key=lambda x: x["name"])

    print("\n" + "─" * 90)
    print("  OOS RESULTS: Combined D+S")
    print("─" * 90)
    print(f"  {'Variant':<16} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} {'OOS_N':>6} "
          f"{'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} {'P≥1':>5} {'CI_lo':>6} {'CI_hi':>6}")
    print("  " + "─" * 82)
    for r in oos_results:
        is_m = r["combined"]["IS"]
        oos_m = r["combined"]["OOS"]
        print(f"  {r['name']:<16} {is_m['N']:>6} {is_m['PF']:>6.3f} {is_m['EV_bp']:>+7.1f} "
              f"{oos_m['N']:>6} {oos_m['PF']:>6.3f} {oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
              f"{oos_m.get('P_PF_ge_1', 0):>4.0f}% "
              f"{oos_m.get('CI_lo', 0):>6.3f} {oos_m.get('CI_hi', 0):>6.3f}")

    print("\n" + "─" * 90)
    print("  OOS RESULTS: D_mom8")
    print("─" * 90)
    print(f"  {'Variant':<16} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} {'OOS_N':>6} "
          f"{'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} {'P≥1':>5}")
    print("  " + "─" * 72)
    for r in oos_results:
        is_m = r["D_mom8"]["IS"]
        oos_m = r["D_mom8"]["OOS"]
        print(f"  {r['name']:<16} {is_m['N']:>6} {is_m['PF']:>6.3f} {is_m['EV_bp']:>+7.1f} "
              f"{oos_m['N']:>6} {oos_m['PF']:>6.3f} {oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
              f"{oos_m.get('P_PF_ge_1', 0):>4.0f}%")

    print("\n" + "─" * 90)
    print("  OOS RESULTS: S_d8")
    print("─" * 90)
    print(f"  {'Variant':<16} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} {'OOS_N':>6} "
          f"{'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} {'P≥1':>5}")
    print("  " + "─" * 72)
    for r in oos_results:
        is_m = r["S_d8"]["IS"]
        oos_m = r["S_d8"]["OOS"]
        print(f"  {r['name']:<16} {is_m['N']:>6} {is_m['PF']:>6.3f} {is_m['EV_bp']:>+7.1f} "
              f"{oos_m['N']:>6} {oos_m['PF']:>6.3f} {oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
              f"{oos_m.get('P_PF_ge_1', 0):>4.0f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n\n" + "=" * 90)
    print(f"  R32 COMPLETE  |  {elapsed:.0f}s  |  {len(VARIANTS)} variants  |"
          f"  {NUM_WORKERS} workers")
    print("=" * 90)

    # Find best OOS combined
    best_oos = max(oos_results, key=lambda x: x["combined"]["OOS"]["EV_bp"])
    m = best_oos["combined"]["OOS"]
    print(f"\n  Best OOS Combined: {best_oos['name']}")
    print(f"    N={m['N']}  PF={m['PF']:.3f}  EV={m['EV_bp']:>+.1f}bp  t={m['t']:.2f}  "
          f"WR={m['WR']:.1f}%  R:R={m['RR']:.2f}  Ret={m['ret_pct']:>+.1f}%  DD={m['max_dd_pct']:.1f}%")

    # Baseline comparison
    baseline_oos = next(r for r in oos_results if r["name"] == "A-Baseline")
    bm = baseline_oos["combined"]["OOS"]
    print(f"\n  Baseline OOS Combined:")
    print(f"    N={bm['N']}  PF={bm['PF']:.3f}  EV={bm['EV_bp']:>+.1f}bp  t={bm['t']:.2f}  "
          f"WR={bm['WR']:.1f}%  R:R={bm['RR']:.2f}  Ret={bm['ret_pct']:>+.1f}%  DD={bm['max_dd_pct']:.1f}%")

    delta_ev = m["EV_bp"] - bm["EV_bp"]
    delta_pf = m["PF"] - bm["PF"]
    print(f"\n  Delta: ΔEV={delta_ev:>+.1f}bp  ΔPF={delta_pf:>+.3f}")


if __name__ == "__main__":
    main()
