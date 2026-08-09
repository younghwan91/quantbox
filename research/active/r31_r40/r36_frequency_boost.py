#!/usr/bin/env python3
"""R36: Frequency Boost Research — Novel Approaches
=====================================================
R35 tested basic N-boost levers (VR threshold, duration, hold, 4TF, CW-Tier).
Winner: A1-D50 (VR<0.50) deployed. N=702→1300 OOS (+85%), EV +64→+39bp.

Problem: Live system shows VR min ~0.78-0.80 across 50 symbols.
The VR < 0.50 threshold is rarely hit. Need fundamentally different
approaches to increase trade frequency without destroying edge.

Novel Approaches (NOT tested in R35):

  V1: Adaptive VR Percentile — VR_pctile(500-bar lookback) < 10th pctile
      → normalizes for regime shifts where VR baseline rises/falls.
      Each symbol has its OWN relative threshold.

  V2: VR Velocity — |dVR/dt| acceleration toward threshold.
      If VR is FALLING fast (velocity < -X), enter BEFORE it crosses
      the absolute threshold. Catches early squeeze entries.

  V3: Shorter VR Periods — VR(5/30) and VR(3/20) instead of VR(7/50).
      Faster-responding ATR ratio → more frequent squeeze detections.
      Trades shorter-lived compressions.

  V4: Cross-Asset Squeeze — When ≥K symbols have VR < relaxed threshold
      simultaneously, enter ALL of them. Market-wide squeeze cluster.
      Higher confidence because of cross-validation.

  V5: Max Concurrent 3-5 Slots — Portfolio-level sim with chronological
      entry queue. Current constraint of 2 slots blocks many valid signals.

  V6: Tiered VR Entry — Enter half-size at VR<0.55, full-size at VR<0.50.
      Captures looser squeezes at lower risk, tighter squeezes at full risk.

  V7: Combined Best — Merge best single levers from V1-V6.

Each variant runs D_mom8 + S_d8 (combined) across 50 symbols × 6yr,
3 TF (15m+1h+4h), with OOS 67/33 split.

Cost assumptions (same as R35):
  - Directional: 2 × 0.04% = 0.08%
  - Straddle:    4 × 0.04% = 0.16%

Usage:
    PYTHONUNBUFFERED=1 .venv/bin/python research/active/frequency_boost_r36.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    WARMUP_15m,
    WARMUP_1h,
    WARMUP_4h,
    build_resampled,
    compute_momentum,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    load_15m,
    make_direction_momentum,
    pf_from_pnl,
    bootstrap_pf,
    precompute,
    vol_percentile,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

NUM_WORKERS = 4
COST_DIRECTIONAL = 0.0008   # 2 × 0.04%
COST_STRADDLE = 0.0016      # 4 × 0.04%

# TF configs: (label, ratio, warmup, max_hold_d, max_hold_s, min_dur)
TF_15m = ("15m", 1, WARMUP_15m, 192, 384, 8)
TF_1h  = ("1h",  4, WARMUP_1h,  48,  96, 8)
TF_4h  = ("4h", 16, WARMUP_4h,  12,  24, 8)
G_3TF  = [TF_15m, TF_1h, TF_4h]


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Novel signal generators
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sma(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += arr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += arr[i] - arr[i - period]
        out[i] = s / period
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
def compute_vr_custom(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sp: int, lp: int,
) -> np.ndarray:
    """Compute VR = ATR(short) / ATR(long) with custom periods."""
    atr_s = _atr_pct(close, high, low, sp)
    atr_l = _atr_pct(close, high, low, lp)
    n = len(close)
    vr = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_s[i]) and not np.isnan(atr_l[i]) and atr_l[i] > 0:
            vr[i] = atr_s[i] / atr_l[i]
    return vr


@numba.jit(nopython=True, cache=True)
def compute_vr_pctile(vr: np.ndarray, lookback: int) -> np.ndarray:
    """Precompute rolling percentile rank of VR (0-100).

    O(n * lookback) but only called ONCE per symbol, not per variant.
    """
    n = len(vr)
    out = np.full(n, 50.0)
    for i in range(lookback, n):
        if np.isnan(vr[i]):
            continue
        cur = vr[i]
        below = 0
        valid = 0
        for k in range(i - lookback + 1, i + 1):
            if not np.isnan(vr[k]):
                valid += 1
                if vr[k] < cur:
                    below += 1
        if valid > 0:
            out[i] = below / valid * 100.0
    return out


@numba.jit(nopython=True, cache=True)
def gen_vr_percentile_signal(
    vr_pctile: np.ndarray, vr: np.ndarray,
    pctile_threshold: float, abs_cap: float, warmup: int,
) -> np.ndarray:
    """V1: Signal when precomputed VR pctile < threshold AND VR < abs_cap.

    Args:
        vr_pctile: Precomputed rolling percentile (0-100).
        vr: Raw VR array (for abs_cap check).
        pctile_threshold: Percentile threshold (e.g., 10.0 = bottom 10%).
        abs_cap: Hard max VR level (e.g., 0.70).
        warmup: Minimum bars before first signal.
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if np.isnan(vr[i]) or vr[i] >= abs_cap:
            continue
        if vr_pctile[i] < pctile_threshold:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_vr_percentile_dur_signal(
    vr_pctile: np.ndarray, vr: np.ndarray,
    pctile_threshold: float, abs_cap: float,
    min_dur: int, warmup: int,
) -> np.ndarray:
    """V1: Duration-filtered percentile squeeze signal for S_d8."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    cnt = 0
    for i in range(n):
        if i < warmup or np.isnan(vr[i]):
            cnt = 0
            continue
        if vr[i] < abs_cap and vr_pctile[i] < pctile_threshold:
            cnt += 1
        else:
            cnt = 0
        if cnt >= min_dur:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_vr_velocity_signal(
    vr: np.ndarray, abs_threshold: float, velocity_period: int,
    velocity_threshold: float, warmup: int,
) -> np.ndarray:
    """V2: Signal when VR is dropping fast toward squeeze territory.

    Fires when:
      - VR < abs_threshold (standard), OR
      - VR < abs_threshold * 1.3 AND VR velocity < velocity_threshold
        (approaching squeeze rapidly)

    velocity = (VR[i] - VR[i-velocity_period]) / velocity_period
    Negative velocity = VR is falling (compressing).
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    upper = abs_threshold * 1.3  # Relaxed zone for velocity entry

    for i in range(max(warmup, velocity_period), n):
        if np.isnan(vr[i]):
            continue
        # Standard threshold
        if vr[i] < abs_threshold:
            sig[i] = 1
            continue
        # Velocity check: VR approaching threshold from above
        if vr[i] < upper and i >= velocity_period:
            prev = vr[i - velocity_period]
            if not np.isnan(prev) and prev > 0:
                vel = (vr[i] - prev) / velocity_period
                if vel < velocity_threshold:
                    sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_vr_velocity_dur_signal(
    vr: np.ndarray, abs_threshold: float, velocity_period: int,
    velocity_threshold: float, min_dur: int, warmup: int,
) -> np.ndarray:
    """V2: Duration-filtered velocity squeeze signal for S_d8."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    upper = abs_threshold * 1.3
    cnt = 0

    for i in range(n):
        if i < max(warmup, velocity_period) or np.isnan(vr[i]):
            cnt = 0
            continue
        fired = False
        if vr[i] < abs_threshold:
            fired = True
        elif vr[i] < upper and i >= velocity_period:
            prev = vr[i - velocity_period]
            if not np.isnan(prev) and prev > 0:
                vel = (vr[i] - prev) / velocity_period
                if vel < velocity_threshold:
                    fired = True
        if fired:
            cnt += 1
        else:
            cnt = 0
        if cnt >= min_dur:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_tiered_signal(
    vr: np.ndarray, tier1_threshold: float, tier2_threshold: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """V6: Two-tier entry signal.

    Returns (sig, size_mult):
      - VR < tier2 (tighter): sig=1, size=1.0
      - tier2 <= VR < tier1 (looser): sig=1, size=0.5
      - VR >= tier1: sig=0
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    size = np.zeros(n, dtype=np.float64)

    for i in range(warmup, n):
        if np.isnan(vr[i]):
            continue
        if vr[i] < tier2_threshold:
            sig[i] = 1
            size[i] = 1.0
        elif vr[i] < tier1_threshold:
            sig[i] = 1
            size[i] = 0.5
    return sig, size


@numba.jit(nopython=True, cache=True)
def gen_tiered_dur_signal(
    vr: np.ndarray, tier1_threshold: float, tier2_threshold: float,
    min_dur: int, warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """V6: Two-tier duration-filtered signal for S_d8."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    size = np.zeros(n, dtype=np.float64)
    cnt = 0

    for i in range(n):
        if i < warmup or np.isnan(vr[i]):
            cnt = 0
            continue
        if vr[i] < tier1_threshold:
            cnt += 1
        else:
            cnt = 0
        if cnt >= min_dur:
            sig[i] = 1
            if vr[i] < tier2_threshold:
                size[i] = 1.0
            else:
                size[i] = 0.5
    return sig, size


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Simulations (identical to R35)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def sim_straddle_tiered(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, size_mult: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with size multiplier applied to PnL."""
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
        sz = size_mult[i]
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
        net = (le + se) * sz - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_tiered(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray, size_mult: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with size multiplier applied to PnL."""
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
        sz = size_mult[i]
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
            pnl[cnt] = (trade_pnl * sz) - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio-Level Simulation (V5: Max Concurrent Slots)
# ══════════════════════════════════════════════════════════════════════════════


def portfolio_sim(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> np.ndarray:
    """Simulate portfolio-level filtering with max concurrent positions.

    Args:
        all_trades: List of (symbol, entry_bar_15m, exit_bar_15m, pnl).
        max_concurrent: Max simultaneous open positions.

    Returns:
        Filtered PnL array (trades that would have been accepted).
    """
    # Sort by entry time
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted_pnl: list[float] = []
    active: list[tuple[str, int]] = []  # (symbol, exit_bar)

    for sym, entry, exit_b, pnl_val in trades:
        # Close expired positions
        active = [(s, e) for s, e in active if e > entry]
        # Check symbol dedup
        sym_active = any(s == sym for s, _ in active)
        if sym_active:
            continue
        # Check slot capacity
        if len(active) >= max_concurrent:
            continue
        # Accept trade
        active.append((sym, exit_b))
        accepted_pnl.append(pnl_val)

    return np.array(accepted_pnl) if accepted_pnl else np.empty(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-Asset Squeeze Detection (V4)
# ══════════════════════════════════════════════════════════════════════════════


def cross_asset_squeeze_mask(
    all_vr: dict[str, np.ndarray],
    relaxed_threshold: float,
    min_symbols: int,
    max_bars: int,
) -> np.ndarray:
    """V4: Generate market-wide squeeze mask.

    Returns a bar-level mask (1 = at least min_symbols are in squeeze).
    """
    mask = np.zeros(max_bars, dtype=np.int8)
    for i in range(max_bars):
        count = 0
        for sym, vr in all_vr.items():
            if i < len(vr) and not np.isnan(vr[i]) and vr[i] < relaxed_threshold:
                count += 1
        if count >= min_symbols:
            mask[i] = 1
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics (same as R35)
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
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
#  Global data (for ProcessPoolExecutor workers)
# ══════════════════════════════════════════════════════════════════════════════

_G: dict = {}


def _init_worker(g: dict) -> None:
    global _G
    _G = g


# ══════════════════════════════════════════════════════════════════════════════
#  Variant Definitions
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS: OrderedDict[str, dict] = OrderedDict()

# ── T0: Current deployed config (A1-D50 from R35) ──
VARIANTS["T0-Deployed"] = {
    "desc": "Current deployed: VR<0.50 (A1-D50)",
    "type": "standard",
    "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
}

# ── V1: Adaptive VR Percentile (with absolute cap to prevent garbage signals) ──
for lb, pct, cap in [(500, 5, 0.65), (500, 10, 0.65), (500, 10, 0.70), (500, 15, 0.70)]:
    key = f"V1-P{pct}L{lb}C{int(cap*100)}"
    VARIANTS[key] = {
        "desc": f"VR pctile<{pct}% ({lb}bar) cap<{cap}",
        "type": "percentile",
        "pct_lookback": lb, "pct_threshold": pct, "abs_cap": cap,
        "s_dur": 8,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── V2: VR Velocity (VR momentum toward squeeze) ──
for vel_p, vel_t in [(8, -0.01), (16, -0.005), (16, -0.003)]:
    key = f"V2-Vel{vel_p}T{abs(vel_t):.3f}"
    VARIANTS[key] = {
        "desc": f"VR velocity: period={vel_p}, vel<{vel_t}",
        "type": "velocity",
        "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
        "velocity_period": vel_p, "velocity_threshold": vel_t,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── V3: Shorter VR Periods ──
for sp, lp in [(5, 30), (3, 20), (5, 40)]:
    key = f"V3-VR{sp}_{lp}"
    VARIANTS[key] = {
        "desc": f"VR({sp}/{lp}) instead of VR(7/50)",
        "type": "short_vr",
        "vr_short": sp, "vr_long": lp,
        "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── V4: Cross-Asset Squeeze (relaxed threshold + market-wide confirmation) ──
for relax_vr, min_k in [(0.65, 5), (0.70, 5), (0.70, 8)]:
    key = f"V4-XSq{relax_vr:.2f}K{min_k}"
    VARIANTS[key] = {
        "desc": f"Cross-asset: VR<{relax_vr} when ≥{min_k} syms squeeze",
        "type": "cross_asset",
        "d_vr": relax_vr, "s_vr": relax_vr + 0.05, "s_dur": 6,
        "min_squeeze_symbols": min_k,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── V5: Max Concurrent Slots (portfolio-level) ──
for slots in [3, 4, 5]:
    key = f"V5-Slot{slots}"
    VARIANTS[key] = {
        "desc": f"max_concurrent={slots} (from 2)",
        "type": "portfolio",
        "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
        "max_concurrent": slots,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── V6: Tiered VR Entry (half-size at looser threshold) ──
for t1, t2 in [(0.55, 0.50), (0.60, 0.50)]:
    key = f"V6-Tier{t1:.2f}_{t2:.2f}"
    VARIANTS[key] = {
        "desc": f"Half@VR<{t1}, full@VR<{t2}",
        "type": "tiered",
        "tier1_threshold": t1, "tier2_threshold": t2,
        "s_vr_tier1": t1 + 0.05, "s_vr_tier2": t2 + 0.05,
        "s_dur": 8,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# Total variants
print(f"  Total variants defined: {len(VARIANTS)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Core Engine — Signal + Sim per variant type
# ══════════════════════════════════════════════════════════════════════════════


def _run_variant(
    var_name: str,
    var_cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
) -> list[tuple[str, int, int, float]]:
    """Run one variant for one strategy across all symbols + TFs.

    Returns list of (symbol, entry_15m, exit_15m, pnl) for portfolio sim.
    """
    vtype = var_cfg["type"]
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    d_sl = var_cfg["d_sl"]
    s_sl = var_cfg["s_sl"]
    tp = var_cfg["tp"]

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur_base in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            n_bars = len(c)

            sl = s_sl if is_straddle else d_sl
            mh = mh_s if is_straddle else mh_d

            # ── Generate signals based on variant type ──
            if vtype == "standard":
                if is_straddle:
                    sig = gen_vr_dur_signal(d["vr"], var_cfg["s_vr"], var_cfg["s_dur"], warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_signal(d["vr"], var_cfg["d_vr"], warmup)
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            elif vtype == "percentile":
                pct = var_cfg["pct_threshold"]
                cap = var_cfg["abs_cap"]
                vr_pctile = d.get("vr_pctile_500", np.full(n_bars, 50.0))
                if is_straddle:
                    sig = gen_vr_percentile_dur_signal(vr_pctile, d["vr"], pct, cap, var_cfg["s_dur"], warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_percentile_signal(vr_pctile, d["vr"], pct, cap, warmup)
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            elif vtype == "velocity":
                vel_p = var_cfg["velocity_period"]
                vel_t = var_cfg["velocity_threshold"]
                if is_straddle:
                    sig = gen_vr_velocity_dur_signal(
                        d["vr"], var_cfg["s_vr"], vel_p, vel_t, var_cfg["s_dur"], warmup,
                    )
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_velocity_signal(
                        d["vr"], var_cfg["d_vr"], vel_p, vel_t, warmup,
                    )
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            elif vtype == "short_vr":
                sp = var_cfg["vr_short"]
                lp = var_cfg["vr_long"]
                vr_custom = compute_vr_custom(c, h, lo, sp, lp)
                if is_straddle:
                    sig = gen_vr_dur_signal(vr_custom, var_cfg["s_vr"], var_cfg["s_dur"], warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_signal(vr_custom, var_cfg["d_vr"], warmup)
                    mom = compute_momentum(c, 8)
                    dire = make_direction_momentum(sig, mom)
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            elif vtype == "cross_asset":
                # Cross-asset: use standard signal BUT only when market squeeze is active
                # The mask will be applied after all TFs are processed
                if is_straddle:
                    sig = gen_vr_dur_signal(d["vr"], var_cfg["s_vr"], var_cfg.get("s_dur", 6), warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_signal(d["vr"], var_cfg["d_vr"], warmup)
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            elif vtype == "portfolio":
                # Same signals as standard, portfolio filtering done later
                if is_straddle:
                    sig = gen_vr_dur_signal(d["vr"], var_cfg["s_vr"], var_cfg["s_dur"], warmup)
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
                else:
                    sig = gen_vr_signal(d["vr"], var_cfg["d_vr"], warmup)
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            elif vtype == "tiered":
                if is_straddle:
                    sig, size = gen_tiered_dur_signal(
                        d["vr"], var_cfg["s_vr_tier1"], var_cfg["s_vr_tier2"],
                        var_cfg["s_dur"], warmup,
                    )
                    p, e, x = sim_straddle_tiered(c, h, lo, sig, size, sl, tp, mh, COST_STRADDLE)
                else:
                    sig, size = gen_tiered_signal(
                        d["vr"], var_cfg["tier1_threshold"], var_cfg["tier2_threshold"], warmup,
                    )
                    dire = make_direction_momentum(sig, d["mom_8"])
                    p, e, x = sim_directional_tiered(c, h, lo, sig, dire, size, sl, tp, mh, COST_DIRECTIONAL)
            else:
                continue

            # Convert to portfolio-level trades (15m-aligned)
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        # Per-symbol dedup (time-based overlap removal)
        sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True
                    break
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation Workers
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_variant(args: tuple) -> dict:
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_years = _G["years"]
    vtype = var_cfg["type"]

    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    d_trades = _run_variant(var_name, var_cfg, data_15m, data_1h, data_4h, "D_mom8")
    s_trades = _run_variant(var_name, var_cfg, data_15m, data_1h, data_4h, "S_d8")

    # For portfolio variants (V5), apply max_concurrent filtering
    if vtype == "portfolio":
        max_c = var_cfg["max_concurrent"]
        # Combine D + S trades and filter together
        all_combined = d_trades + s_trades
        combined_pnl = portfolio_sim(all_combined, max_c)
        d_pnl = portfolio_sim(d_trades, max_c)
        s_pnl = portfolio_sim(s_trades, max_c)
    else:
        d_pnl = np.array([t[3] for t in d_trades]) if d_trades else np.empty(0)
        s_pnl = np.array([t[3] for t in s_trades]) if s_trades else np.empty(0)
        combined_pnl = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)

    for key, pnl in [("D_mom8", d_pnl), ("S_d8", s_pnl), ("combined", combined_pnl)]:
        m = compute_metrics(pnl, data_years)
        bs = bootstrap_pf(pnl, 1000)
        m["P_PF_ge_1"] = bs["P_PF_ge_1"]
        m["CI_lo"] = bs["CI_lo"]
        m["CI_hi"] = bs["CI_hi"]
        result[key] = m

    return result


def oos_evaluate_variant(args: tuple) -> dict:
    """OOS 67/33 evaluation."""
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_years = _G["years"]
    vtype = var_cfg["type"]

    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    # Build IS/OOS splits
    is_15m: dict = {}
    oos_15m: dict = {}
    is_1h: dict = {}
    oos_1h: dict = {}
    is_4h: dict = {}
    oos_4h: dict = {}

    for sym in data_15m:
        n = len(data_15m[sym]["close"])
        cut = int(n * 0.67)

        sources = [
            (data_15m, is_15m, oos_15m),
            (data_1h, is_1h, oos_1h),
            (data_4h, is_4h, oos_4h),
        ]

        for src, is_dst, oos_dst in sources:
            if sym not in src:
                continue
            d = src[sym]
            n_tf = len(d["close"])
            ratio = n // n_tf if n_tf > 0 else 1
            cut_tf = cut // ratio

            for start, end_b, dst in [
                (0, cut_tf, is_dst),
                (cut_tf, n_tf, oos_dst),
            ]:
                seg_len = end_b - start
                if seg_len < 100:
                    continue
                d_seg: dict = {}
                for key in ("close", "high", "low", "volume"):
                    d_seg[key] = d[key][start:end_b].copy()
                d_seg["bars"] = seg_len
                c, h, lo, v = (
                    d_seg["close"], d_seg["high"], d_seg["low"], d_seg["volume"],
                )
                d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
                d_seg["mom_8"] = compute_momentum(c, 8)
                d_seg["vol_pct"] = vol_percentile(v, min(200, seg_len // 2))
                d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)
                d_seg["vr_pctile_500"] = compute_vr_pctile(d_seg["vr"], min(500, seg_len // 2))
                dst[sym] = d_seg

    is_years = data_years * 0.67
    oos_years = data_years * 0.33

    for split_label, sd_15m, sd_1h, sd_4h, years in [
        ("IS", is_15m, is_1h, is_4h, is_years),
        ("OOS", oos_15m, oos_1h, oos_4h, oos_years),
    ]:
        d_trades = _run_variant(var_name, var_cfg, sd_15m, sd_1h, sd_4h, "D_mom8")
        s_trades = _run_variant(var_name, var_cfg, sd_15m, sd_1h, sd_4h, "S_d8")

        if vtype == "portfolio":
            max_c = var_cfg["max_concurrent"]
            all_combined = d_trades + s_trades
            combined_pnl = portfolio_sim(all_combined, max_c)
            d_pnl = portfolio_sim(d_trades, max_c)
            s_pnl = portfolio_sim(s_trades, max_c)
        else:
            d_pnl = np.array([t[3] for t in d_trades]) if d_trades else np.empty(0)
            s_pnl = np.array([t[3] for t in s_trades]) if s_trades else np.empty(0)
            combined_pnl = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)

        for key, pnl in [("D_mom8", d_pnl), ("S_d8", s_pnl), ("combined", combined_pnl)]:
            m = compute_metrics(pnl, years)
            bs = bootstrap_pf(pnl, 1000)
            m["P_PF_ge_1"] = bs["P_PF_ge_1"]
            m["CI_lo"] = bs["CI_lo"]
            m["CI_hi"] = bs["CI_hi"]
            if key not in result:
                result[key] = {}
            result[key][split_label] = m

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Printing
# ══════════════════════════════════════════════════════════════════════════════


def print_table(results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 110}")
    print(f"  {label}")
    print(f"{'─' * 110}")
    header = (
        f"  {'Variant':<22} {'N':>6} {'PF':>6} {'EV_bp':>7} {'t':>5} "
        f"{'WR%':>5} {'R:R':>5} {'Ret%':>9} {'DD%':>5} {'TPD':>5} {'P≥1':>4}"
    )
    print(header)
    print("  " + "─" * 98)
    for r in results:
        m = r[strat_key]
        print(
            f"  {r['name']:<22} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
            f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['RR']:>5.2f} "
            f"{m['ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
            f"{m['tpd']:>5.2f} "
            f"{m.get('P_PF_ge_1', 0):>4.0f}%"
        )


def print_oos_table(results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 115}")
    print(f"  {label}")
    print(f"{'─' * 115}")
    print(
        f"  {'Variant':<22} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} "
        f"{'OOS_N':>6} {'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} "
        f"{'P≥1':>5} {'CI_lo':>6} {'CI_hi':>6} {'Ratio':>5}"
    )
    print("  " + "─" * 100)
    for r in results:
        if strat_key not in r or "IS" not in r[strat_key]:
            continue
        is_m = r[strat_key]["IS"]
        oos_m = r[strat_key]["OOS"]
        ratio = oos_m["PF"] / is_m["PF"] if is_m["PF"] > 0 else 0
        print(
            f"  {r['name']:<22} {is_m['N']:>6} {is_m['PF']:>6.3f} "
            f"{is_m['EV_bp']:>+7.1f} {oos_m['N']:>6} {oos_m['PF']:>6.3f} "
            f"{oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
            f"{oos_m.get('P_PF_ge_1', 0):>4.0f}% "
            f"{oos_m.get('CI_lo', 0):>6.3f} {oos_m.get('CI_hi', 0):>6.3f} "
            f"{ratio:>5.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 110)
    print("  R36: FREQUENCY BOOST RESEARCH — NOVEL APPROACHES")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Base: A1-D50 deployed (VR<0.50, 3TF, D_mom8+S_d8)")
    print(f"  Variants: {len(VARIANTS)}")
    print("=" * 110)

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

    # Precompute VR percentile for V1 variants (O(n*lookback) once, not per-variant)
    print("  Precomputing VR percentile (500-bar lookback)...")
    for dataset in [data_15m, data_1h, data_4h]:
        for sym, d in dataset.items():
            d["vr_pctile_500"] = compute_vr_pctile(d["vr"], 500)
    print("  Done.\n")

    # ── Global data ──
    global _G
    _G = {
        "15m": data_15m, "1h": data_1h, "4h": data_4h,
        "years": data_years,
    }

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 110)
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON")
    print("=" * 110)

    eval_args = [(vn, vc) for vn, vc in VARIANTS.items()]

    print(f"\n  Running {len(eval_args)} variants across {NUM_WORKERS} workers...")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(evaluate_variant, eval_args):
            results.append(r)
            print(f"    ✓ {r['name']:<22}  N={r['combined']['N']:>6}")

    results.sort(key=lambda x: x["name"])
    base_m = next(r for r in results if r["name"] == "T0-Deployed")["combined"]
    base_n = base_m["N"]
    base_ev = base_m["EV_bp"]
    base_evn = base_ev * base_n

    print_table(results, "FULL-SAMPLE: Combined D+S (deduped)", "combined")
    print_table(results, "FULL-SAMPLE: D_mom8 (deduped)", "D_mom8")
    print_table(results, "FULL-SAMPLE: S_d8 (deduped)", "S_d8")

    # ── Ranking by N ──
    print(f"\n{'─' * 110}")
    print("  RANKING by Combined N (trade count)")
    print(f"{'─' * 110}")
    ranked_n = sorted(results, key=lambda x: x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_n, 1):
        m = r["combined"]
        delta_n = m["N"] - base_n
        delta_n_pct = (m["N"] / base_n - 1) * 100 if base_n > 0 else 0
        delta_ev = m["EV_bp"] - base_ev
        evn = m["EV_bp"] * m["N"]
        delta_evn_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Deployed" else ""
        print(
            f"  #{i:>2} {r['name']:<22} N={m['N']:>6} ({delta_n:>+5}, {delta_n_pct:>+6.0f}%)  "
            f"EV={m['EV_bp']:>+7.1f}bp ({delta_ev:>+5.1f})  PF={m['PF']:.3f}  "
            f"EV×N Δ={delta_evn_pct:>+6.1f}%  t={m['t']:.2f}{marker}"
        )

    # ── Ranking by EV×N ──
    print(f"\n{'─' * 110}")
    print("  RANKING by Combined EV×N (total expected profit)")
    print(f"{'─' * 110}")
    ranked_evn = sorted(results, key=lambda x: x["combined"]["EV_bp"] * x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_evn, 1):
        m = r["combined"]
        evn = m["EV_bp"] * m["N"]
        delta_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Deployed" else ""
        print(
            f"  #{i:>2} {r['name']:<22} EV×N={evn:>+10.0f}  "
            f"(EV={m['EV_bp']:>+.1f} × N={m['N']})  "
            f"Δ={delta_pct:>+6.1f}%  DD={m['max_dd_pct']:.1f}%{marker}"
        )

    # ── Ranking by EV with N threshold ──
    print(f"\n{'─' * 110}")
    print(f"  RANKING by EV (only variants with N ≥ {int(base_n * 1.2)}, i.e. +20% freq)")
    print(f"{'─' * 110}")
    freq_threshold = int(base_n * 1.2)
    freq_filtered = [r for r in results if r["combined"]["N"] >= freq_threshold]
    ranked_ev_freq = sorted(freq_filtered, key=lambda x: x["combined"]["EV_bp"], reverse=True)
    for i, r in enumerate(ranked_ev_freq, 1):
        m = r["combined"]
        n_boost = (m["N"] / base_n - 1) * 100
        ev_change = m["EV_bp"] - base_ev
        print(
            f"  #{i:>2} {r['name']:<22} EV={m['EV_bp']:>+7.1f}bp  N={m['N']:>5} (+{n_boost:.0f}%)  "
            f"PF={m['PF']:.3f}  t={m['t']:.2f}  DD={m['max_dd_pct']:.1f}%  "
            f"ΔEV={ev_change:>+.1f}bp"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: OOS evaluation
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("  PHASE 2: OOS TEST (67/33 split)")
    print(f"{'=' * 110}")

    oos_args = [(vn, VARIANTS[vn]) for vn in sorted(VARIANTS.keys())]

    print(f"\n  Running OOS for {len(oos_args)} variants...")
    oos_results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(oos_evaluate_variant, oos_args):
            oos_results.append(r)
            oos_n = r.get("combined", {}).get("OOS", {}).get("N", 0)
            print(f"    ✓ {r['name']:<22}  OOS_N={oos_n:>5}")

    oos_results.sort(key=lambda x: x["name"])

    print_oos_table(oos_results, "OOS: Combined D+S", "combined")

    # ── OOS Rankings ──
    oos_base = next(r for r in oos_results if r["name"] == "T0-Deployed")
    bl_oos = oos_base["combined"]["OOS"]
    bl_oos_ev = bl_oos["EV_bp"]
    bl_oos_n = bl_oos["N"]
    bl_oos_evn = bl_oos_ev * bl_oos_n

    print(f"\n{'─' * 110}")
    print("  OOS RANKING by EV×N (total expected profit)")
    print(f"{'─' * 110}")
    oos_ranked_evn = sorted(
        oos_results,
        key=lambda x: x.get("combined", {}).get("OOS", {}).get("EV_bp", 0) * x.get("combined", {}).get("OOS", {}).get("N", 0),
        reverse=True,
    )
    for i, r in enumerate(oos_ranked_evn, 1):
        if "combined" not in r or "OOS" not in r["combined"]:
            continue
        oos_m = r["combined"]["OOS"]
        evn = oos_m["EV_bp"] * oos_m["N"]
        delta_pct = (evn / bl_oos_evn - 1) * 100 if bl_oos_evn != 0 else 0
        n_boost = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        marker = " ◄" if r["name"] == "T0-Deployed" else ""
        print(
            f"  #{i:>2} {r['name']:<22} EV×N={evn:>+10.0f}  "
            f"(EV={oos_m['EV_bp']:>+.1f} × N={oos_m['N']})  "
            f"N+{n_boost:.0f}%  PF={oos_m['PF']:.3f}  DD={oos_m['max_dd_pct']:.1f}%  "
            f"t={oos_m['t']:.2f}{marker}"
        )

    print(f"\n{'─' * 110}")
    print("  OOS RANKING by EV (per-trade quality)")
    print(f"{'─' * 110}")
    oos_ranked_ev = sorted(
        oos_results,
        key=lambda x: x.get("combined", {}).get("OOS", {}).get("EV_bp", -999),
        reverse=True,
    )
    for i, r in enumerate(oos_ranked_ev, 1):
        if "combined" not in r or "OOS" not in r["combined"]:
            continue
        oos_m = r["combined"]["OOS"]
        delta_ev = oos_m["EV_bp"] - bl_oos_ev
        n_boost = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        marker = " ◄" if r["name"] == "T0-Deployed" else ""
        print(
            f"  #{i:>2} {r['name']:<22} OOS_EV={oos_m['EV_bp']:>+7.1f}bp  "
            f"PF={oos_m['PF']:.3f}  N={oos_m['N']:>5} (+{n_boost:.0f}%)  "
            f"t={oos_m['t']:.2f}  DD={oos_m['max_dd_pct']:.1f}%  "
            f"CI_lo={oos_m.get('CI_lo', 0):.3f}  ΔEV={delta_ev:>+.1f}bp{marker}"
        )

    # ── Sweet Spot Analysis ──
    print(f"\n{'─' * 110}")
    print("  SWEET SPOT: OOS variants where N ≥ +20% AND EV degradation ≤ 10bp AND PF ≥ 1.20")
    print(f"{'─' * 110}")
    sweet = []
    for r in oos_results:
        if "combined" not in r or "OOS" not in r["combined"]:
            continue
        oos_m = r["combined"]["OOS"]
        n_boost_pct = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        ev_degrad = bl_oos_ev - oos_m["EV_bp"]
        if n_boost_pct >= 20 and ev_degrad <= 10 and oos_m["PF"] >= 1.20:
            sweet.append(r)
    sweet.sort(key=lambda x: x["combined"]["OOS"]["EV_bp"] * x["combined"]["OOS"]["N"], reverse=True)
    if sweet:
        for i, r in enumerate(sweet, 1):
            oos_m = r["combined"]["OOS"]
            n_boost = (oos_m["N"] / bl_oos_n - 1) * 100
            ev_loss = bl_oos_ev - oos_m["EV_bp"]
            evn = oos_m["EV_bp"] * oos_m["N"]
            print(
                f"  #{i} {r['name']:<22} N+{n_boost:.0f}%  EV={oos_m['EV_bp']:>+.1f}bp  "
                f"(ΔEV={-ev_loss:>+.1f})  PF={oos_m['PF']:.3f}  DD={oos_m['max_dd_pct']:.1f}%  "
                f"EV×N={evn:>+.0f}"
            )
    else:
        print("  ⚠️  No variant meets all 3 criteria. Check individual rankings.")

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 110}")
    print(
        f"  R36 COMPLETE  |  {elapsed:.0f}s  |  {len(VARIANTS)} variants  "
        f"|  {NUM_WORKERS} workers  |  commit: {commit}"
    )
    print(f"{'=' * 110}")

    print(f"\n  Deployed baseline (T0): N={bl_oos_n}  EV={bl_oos_ev:>+.1f}bp  "
          f"PF={bl_oos['PF']:.3f}  DD={bl_oos['max_dd_pct']:.1f}%")

    if oos_ranked_evn and "combined" in oos_ranked_evn[0] and "OOS" in oos_ranked_evn[0]["combined"]:
        best = oos_ranked_evn[0]
        bm = best["combined"]["OOS"]
        evn_val = bm["EV_bp"] * bm["N"]
        n_boost = (bm["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        print(f"\n  Best OOS EV×N: {best['name']}")
        print(
            f"    N={bm['N']} (+{n_boost:.0f}%)  EV={bm['EV_bp']:>+.1f}bp  "
            f"PF={bm['PF']:.3f}  EV×N={evn_val:>+.0f}  DD={bm['max_dd_pct']:.1f}%"
        )

    print(f"\n  ⏱️  Runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
