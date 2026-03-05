"""R43 — Multi-Vector Exit & Entry Improvement Research

Baseline: v13.0.1 (CFG_V13: VR8_56, SL1/TP5, s_dur=4, 10 slots)
  N=2,265, PF=1.280, EV=+32.1bp, WR=43.4%, Calmar=5.8

Research Vectors (all unexplored):
  V1: Trailing Stop (directional) — trail SL after initial gain
  V2: Breakeven Ratchet (directional) — move SL to entry after +X%
  V3: ATR-Scaled Exits — SL/TP proportional to per-bar ATR%
  V4: Volume Confirmation Filter — enter only on volume spike
  V5: Symbol Exclusion — filter historically losing symbols
  V6: Straddle Leg Management — trail winning leg, cut losing leg faster
  V7: Combine best vectors

Rules:
  - Baseline uses R38 canonical sim (import, no reimplementation)
  - ±1% sanity check vs PDF canonical (N=2,265, PF=1.285)
  - 10 slots production config
  - No lookahead: all indicators use completed bars only

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/exit_improvement_r43.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    _atr_pct, compute_vr, compute_momentum, compute_squeeze_dur,
    vol_percentile, gen_vr_signal, gen_vr_dur_signal,
    make_direction_momentum, resample_ohlcv,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    sim_straddle_trades, sim_directional_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF, CFG_V13,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  V1: Trailing Stop — Directional
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_trailing(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float,   # activate trailing after this % gain
    trail_distance: float,   # trailing stop distance from peak
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with trailing stop.

    Once unrealized gain reaches `trail_activate`, stop-loss becomes
    a trailing stop at `trail_distance` below the peak gain.
    TP is kept as a hard ceiling.

    Args:
        trail_activate: e.g. 0.02 means trailing activates after +2% gain
        trail_distance: e.g. 0.01 means trail 1% below peak
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
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                cur_price_gain = (close[j] - entry) / entry
                # Update peak
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                # Check trailing stop first (if active)
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        # Trailing stop hit
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break

                # Check hard TP
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break

                # Check initial SL (only before trailing activates)
                if not trailing_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break

                # Activate trailing
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                cur_gain = (entry - low[j]) / entry
                cur_price_gain = (entry - close[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break

                if high[j] >= entry * (1.0 + tp) and d == -1:
                    # Nonsense for short — check SL
                    pass
                if low[j] <= entry * (1.0 - tp) and d == -1:
                    trade_pnl = tp
                    eb = j
                    break

                if not trailing_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break

                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
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


# ══════════════════════════════════════════════════════════════════════════════
#  V2: Breakeven Ratchet — Directional
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_breakeven(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    be_activate: float,   # move SL to breakeven after this % gain
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with breakeven ratchet.

    After reaching `be_activate` gain, SL moves to entry price (breakeven).
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
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        be_active = False
        peak_gain = 0.0

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not be_active and peak_gain >= be_activate:
                    be_active = True

                # Check SL / breakeven
                if be_active:
                    if low[j] <= entry:   # breakeven
                        trade_pnl = 0.0
                        eb = j
                        break
                else:
                    if low[j] <= entry * (1.0 - sl):
                        trade_pnl = -sl
                        eb = j
                        break

                # Check TP
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not be_active and peak_gain >= be_activate:
                    be_active = True

                if be_active:
                    if high[j] >= entry:   # breakeven
                        trade_pnl = 0.0
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


# ══════════════════════════════════════════════════════════════════════════════
#  V3: ATR-Scaled Exits
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_atr_exit(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    atr: np.ndarray,
    sl_mult: float, tp_mult: float,
    max_hold: int, cost: float,
    sl_floor: float, sl_cap: float,
    tp_floor: float, tp_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with ATR-scaled SL/TP.

    SL = clamp(atr[i] * sl_mult, sl_floor, sl_cap)
    TP = clamp(atr[i] * tp_mult, tp_floor, tp_cap)
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
        if entry <= 0 or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        d = dire[i]

        # ATR-scaled SL/TP
        raw_sl = atr[i] * sl_mult
        raw_tp = atr[i] * tp_mult
        sl = min(max(raw_sl, sl_floor), sl_cap)
        tp = min(max(raw_tp, tp_floor), tp_cap)

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
def sim_straddle_atr_exit(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, atr: np.ndarray,
    sl_mult: float, tp_mult: float,
    max_hold: int, cost: float,
    sl_floor: float, sl_cap: float,
    tp_floor: float, tp_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with ATR-scaled SL/TP."""
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
        if entry <= 0 or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        raw_sl = atr[i] * sl_mult
        raw_tp = atr[i] * tp_mult
        sl = min(max(raw_sl, sl_floor), sl_cap)
        tp = min(max(raw_tp, tp_floor), tp_cap)

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


# ══════════════════════════════════════════════════════════════════════════════
#  V6: Straddle Trailing — trail winning leg after activation
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trailing(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float,   # activate trailing on winning leg after this gain
    trail_distance: float,   # trail distance from peak of winning leg
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with trailing on winning legs.

    Each leg gets its own trailing stop once it gains `trail_activate`.
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

        la = True
        sa = True
        le = 0.0
        se = 0.0
        l_peak = 0.0
        s_peak = 0.0
        l_trailing = False
        s_trailing = False
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            # Long leg
            if la:
                l_gain = (high[j] - entry) / entry
                if l_gain > l_peak:
                    l_peak = l_gain

                if not l_trailing and l_peak >= trail_activate:
                    l_trailing = True

                if l_trailing:
                    trail_lvl = entry * (1.0 + l_peak - trail_distance)
                    if low[j] <= trail_lvl:
                        le = max(l_peak - trail_distance, -sl)
                        la = False
                else:
                    if low[j] <= entry * (1.0 - sl):
                        le = -sl
                        la = False

                if la and high[j] >= entry * (1.0 + tp):
                    le = tp
                    la = False

            # Short leg
            if sa:
                s_gain = (entry - low[j]) / entry
                if s_gain > s_peak:
                    s_peak = s_gain

                if not s_trailing and s_peak >= trail_activate:
                    s_trailing = True

                if s_trailing:
                    trail_lvl = entry * (1.0 - s_peak + trail_distance)
                    if high[j] >= trail_lvl:
                        se = max(s_peak - trail_distance, -sl)
                        sa = False
                else:
                    if high[j] >= entry * (1.0 + sl):
                        se = -sl
                        sa = False

                if sa and low[j] <= entry * (1.0 - tp):
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
#  Trade collection wrappers
# ══════════════════════════════════════════════════════════════════════════════


def collect_trades_variant(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
    variant: str,
    variant_params: dict,
    exclude_symbols: set | None = None,
    min_vol_pct: float = 0.0,
) -> list[tuple[str, int, int, float]]:
    """Collect trades with variant sim function.

    variant: "baseline", "trailing", "breakeven", "atr_exit",
             "straddle_trailing"
    """
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    d_sl = cfg["d_sl"]
    s_sl = cfg["s_sl"]
    tp = cfg["tp"]
    d_vr = cfg["d_vr"]
    s_vr = cfg["s_vr"]
    s_dur = cfg["s_dur"]

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        if exclude_symbols and sym in exclude_symbols:
            continue

        sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            sl = s_sl if is_straddle else d_sl
            mh = mh_s if is_straddle else mh_d

            # Volume filter
            if min_vol_pct > 0:
                vol_pct = d.get("vol_pct")
                if vol_pct is None:
                    vol_pct = vol_percentile(d["volume"], min(200, len(c) // 2))
                    d["vol_pct"] = vol_pct

            if is_straddle:
                sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)

                # Apply volume filter by zeroing signal
                if min_vol_pct > 0:
                    vol_pct = d["vol_pct"]
                    sig_filtered = sig.copy()
                    for k in range(len(sig_filtered)):
                        if sig_filtered[k] == 1 and vol_pct[k] < min_vol_pct:
                            sig_filtered[k] = 0
                    sig = sig_filtered

                if variant == "straddle_trailing":
                    p, e, x = sim_straddle_trailing(
                        c, h, lo, sig, sl, tp, mh, COST_STRADDLE,
                        variant_params.get("trail_activate", 0.03),
                        variant_params.get("trail_distance", 0.01),
                    )
                elif variant == "atr_exit":
                    atr_lp = _atr_pct(c, h, lo, lp)
                    p, e, x = sim_straddle_atr_exit(
                        c, h, lo, sig, atr_lp,
                        variant_params.get("sl_mult", 2.0),
                        variant_params.get("tp_mult", 5.0),
                        mh, COST_STRADDLE,
                        variant_params.get("sl_floor", 0.005),
                        variant_params.get("sl_cap", 0.03),
                        variant_params.get("tp_floor", 0.02),
                        variant_params.get("tp_cap", 0.10),
                    )
                else:
                    p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
            else:
                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])

                # Apply volume filter
                if min_vol_pct > 0:
                    vol_pct = d["vol_pct"]
                    sig_filtered = sig.copy()
                    for k in range(len(sig_filtered)):
                        if sig_filtered[k] == 1 and vol_pct[k] < min_vol_pct:
                            sig_filtered[k] = 0
                    sig = sig_filtered
                    dire = make_direction_momentum(sig, d["mom_8"])

                if variant == "trailing":
                    p, e, x = sim_directional_trailing(
                        c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL,
                        variant_params.get("trail_activate", 0.02),
                        variant_params.get("trail_distance", 0.01),
                    )
                elif variant == "breakeven":
                    p, e, x = sim_directional_breakeven(
                        c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL,
                        variant_params.get("be_activate", 0.02),
                    )
                elif variant == "atr_exit":
                    atr_lp = _atr_pct(c, h, lo, lp)
                    p, e, x = sim_directional_atr_exit(
                        c, h, lo, sig, dire, atr_lp,
                        variant_params.get("sl_mult", 1.5),
                        variant_params.get("tp_mult", 5.0),
                        mh, COST_DIRECTIONAL,
                        variant_params.get("sl_floor", 0.005),
                        variant_params.get("sl_cap", 0.02),
                        variant_params.get("tp_floor", 0.02),
                        variant_params.get("tp_cap", 0.10),
                    )
                else:
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        # Per-symbol time dedup
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
#  Year-by-year analysis helpers
# ══════════════════════════════════════════════════════════════════════════════


def get_year_boundaries(data_15m: dict) -> dict[int, tuple[int, int]]:
    """Get approximate bar indices for each year (from data_15m)."""
    # Assume data starts around 2020-02-18 (6yr lookback from 2026-02-18)
    # 15m bars per year ≈ 365.25 * 24 * 4 = 35,064
    max_bars = max(len(d["close"]) for d in data_15m.values())
    bars_per_year = 365 * 24 * 4  # 35,040
    years: dict[int, tuple[int, int]] = {}
    for yr_idx in range(6):
        year = 2020 + yr_idx
        start = yr_idx * bars_per_year
        end = min((yr_idx + 1) * bars_per_year, max_bars)
        if start < max_bars:
            years[year] = (start, end)
    return years


def year_metrics(
    pnl_array: np.ndarray,
    accepted_trades: list[tuple[str, int, int, float]],
    year_bounds: dict[int, tuple[int, int]],
) -> dict[int, dict]:
    """Compute per-year metrics from accepted trades."""
    results: dict[int, dict] = {}
    for year, (start, end) in year_bounds.items():
        year_pnl = []
        for idx, (sym, entry, exit_b, pnl_val) in enumerate(accepted_trades):
            if start <= entry < end:
                year_pnl.append(pnl_val)
        if year_pnl:
            pnl_arr = np.array(year_pnl)
            n = len(pnl_arr)
            pf = pf_from_pnl(pnl_arr)
            ev = float(np.mean(pnl_arr)) * 10000
            wr = np.sum(pnl_arr > 0) / n * 100
            results[year] = {"N": n, "PF": round(pf, 2), "EV_bp": round(ev, 1), "WR": round(wr, 1)}
        else:
            results[year] = {"N": 0, "PF": 0, "EV_bp": 0, "WR": 0}
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Symbol performance analysis
# ══════════════════════════════════════════════════════════════════════════════


def symbol_performance(
    accepted_trades: list[tuple[str, int, int, float]],
) -> dict[str, dict]:
    """Compute per-symbol performance."""
    sym_pnl: dict[str, list[float]] = {}
    for sym, entry, exit_b, pnl_val in accepted_trades:
        sym_pnl.setdefault(sym, []).append(pnl_val)

    results: dict[str, dict] = {}
    for sym, pnls in sym_pnl.items():
        arr = np.array(pnls)
        n = len(arr)
        pf = pf_from_pnl(arr)
        ev = float(np.mean(arr)) * 10000
        wr = np.sum(arr > 0) / n * 100
        total_return = float(np.sum(arr)) * 10000
        results[sym] = {
            "N": n, "PF": round(pf, 3), "EV_bp": round(ev, 1),
            "WR": round(wr, 1), "total_bp": round(total_return, 0),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def print_results_table(name: str, m: dict, m_yr: dict | None = None):
    """Print a single result line."""
    print(f"  {name:<35} N={m['N']:>5}  PF={m['PF']:>5.3f}  EV={m['EV_bp']:>+6.1f}bp  "
          f"WR={m['WR']:>4.1f}%  AnnRet={m['ann_ret_pct']:>+7.1f}%  "
          f"DD={m['max_dd_pct']:>5.1f}%  Calmar={m['calmar']:>5.1f}")
    if m_yr:
        yr_str = "    Yearly: "
        for yr in sorted(m_yr.keys()):
            y = m_yr[yr]
            yr_str += f"{yr}: N={y['N']:>3} PF={y['PF']:.2f} EV={y['EV_bp']:>+5.1f}  "
        print(yr_str)


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*100}")
    print(f"  R43 — Multi-Vector Exit & Entry Improvement Research")
    print(f"  Commit: {commit}  |  Baseline: CFG_V13 (VR8_56, SL1/TP5, s_dur=4)")
    print(f"{'='*100}\n")

    t0 = time.time()

    # ── Load data ──
    print("Loading data...")
    data_15m = load_15m()
    n_symbols = len(data_15m)
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    print(f"  {n_symbols} symbols × {data_years:.1f}yr loaded in {time.time()-t0:.1f}s")

    year_bounds = get_year_boundaries(data_15m)
    SLOTS = 10

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: Baseline (R38 canonical — import, no reimplementation)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 1: Baseline Verification (R38 canonical)")
    print(f"{'─'*100}")

    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_trades = bl_d + bl_s
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_trades, SLOTS)
    bl_m = compute_metrics(bl_pnl, data_years)

    # Sanity check: ±1% vs PDF (N=2,265, PF=1.285)
    pdf_n, pdf_pf = 2265, 1.285
    n_delta = abs(bl_m["N"] - pdf_n) / pdf_n * 100
    pf_delta = abs(bl_m["PF"] - pdf_pf) / pdf_pf * 100
    if n_delta > 1 or pf_delta > 1:
        print(f"  ❌ BASELINE DRIFT! N={bl_m['N']} (Δ{n_delta:.1f}%), PF={bl_m['PF']} (Δ{pf_delta:.1f}%)")
        print(f"     Expected: N={pdf_n}, PF={pdf_pf}")
        sys.exit(1)
    print(f"  ✓ Baseline matches PDF: N={bl_m['N']} (Δ{n_delta:.2f}%), PF={bl_m['PF']} (Δ{pf_delta:.2f}%)")

    bl_yr = year_metrics(bl_pnl, bl_accepted, year_bounds)
    print_results_table("BASELINE (v13.0.1)", bl_m, bl_yr)

    # Symbol analysis for V5
    bl_sym = symbol_performance(bl_accepted)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: V1 — Trailing Stop (Directional only)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 2: V1 — Trailing Stop (Directional)")
    print(f"  After +X% gain, trail SL at Y% below peak. TP kept as hard ceiling.")
    print(f"{'─'*100}")

    trail_configs = [
        ("Trail_A2_D1", {"trail_activate": 0.02, "trail_distance": 0.01}),
        ("Trail_A3_D1", {"trail_activate": 0.03, "trail_distance": 0.01}),
        ("Trail_A3_D15", {"trail_activate": 0.03, "trail_distance": 0.015}),
        ("Trail_A2_D05", {"trail_activate": 0.02, "trail_distance": 0.005}),
        ("Trail_A4_D2", {"trail_activate": 0.04, "trail_distance": 0.02}),
        ("Trail_A15_D05", {"trail_activate": 0.015, "trail_distance": 0.005}),
    ]

    best_trail = None
    best_trail_calmar = bl_m["calmar"]

    for name, params in trail_configs:
        t1 = time.time()
        # Directional uses trailing, straddle uses baseline
        d_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
            "trailing", params,
        )
        s_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f}  [{elapsed:.1f}s]")

        # Print year-by-year deltas for promising ones
        if m["calmar"] > bl_m["calmar"] * 0.95:
            yr_str = "    Yearly ΔEV: "
            for y in sorted(yr.keys()):
                delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
                yr_str += f"{y}: {delta:>+5.1f}  "
            print(yr_str)

        if m["calmar"] > best_trail_calmar:
            best_trail_calmar = m["calmar"]
            best_trail = (name, params, m, yr)

    if best_trail:
        print(f"\n  ★ Best Trail: {best_trail[0]} — Calmar {best_trail[2]['calmar']:.1f} "
              f"(baseline {bl_m['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 3: V2 — Breakeven Ratchet (Directional only)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 3: V2 — Breakeven Ratchet (Directional)")
    print(f"  After +X% gain, move SL to entry price (zero-loss).")
    print(f"{'─'*100}")

    be_configs = [
        ("BE_2pct", {"be_activate": 0.02}),
        ("BE_3pct", {"be_activate": 0.03}),
        ("BE_4pct", {"be_activate": 0.04}),
        ("BE_15pct", {"be_activate": 0.015}),
        ("BE_1pct", {"be_activate": 0.01}),
    ]

    best_be = None
    best_be_calmar = bl_m["calmar"]

    for name, params in be_configs:
        t1 = time.time()
        d_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
            "breakeven", params,
        )
        s_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f}  [{elapsed:.1f}s]")

        if m["calmar"] > bl_m["calmar"] * 0.95:
            yr_str = "    Yearly ΔEV: "
            for y in sorted(yr.keys()):
                delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
                yr_str += f"{y}: {delta:>+5.1f}  "
            print(yr_str)

        if m["calmar"] > best_be_calmar:
            best_be_calmar = m["calmar"]
            best_be = (name, params, m, yr)

    if best_be:
        print(f"\n  ★ Best BE: {best_be[0]} — Calmar {best_be[2]['calmar']:.1f} "
              f"(baseline {bl_m['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 4: V3 — ATR-Scaled Exits (both D and S)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 4: V3 — ATR-Scaled Exits")
    print(f"  SL = ATR(56) × mult, TP = ATR(56) × mult. Clamped to floor/cap.")
    print(f"{'─'*100}")

    atr_configs = [
        # (name, d_sl_mult, d_tp_mult, s_sl_mult, s_tp_mult)
        ("ATR_SL15_TP50", {"sl_mult": 1.5, "tp_mult": 5.0,
                           "sl_floor": 0.005, "sl_cap": 0.02,
                           "tp_floor": 0.02, "tp_cap": 0.10}),
        ("ATR_SL20_TP50", {"sl_mult": 2.0, "tp_mult": 5.0,
                           "sl_floor": 0.005, "sl_cap": 0.025,
                           "tp_floor": 0.02, "tp_cap": 0.10}),
        ("ATR_SL15_TP60", {"sl_mult": 1.5, "tp_mult": 6.0,
                           "sl_floor": 0.005, "sl_cap": 0.02,
                           "tp_floor": 0.02, "tp_cap": 0.12}),
        ("ATR_SL20_TP70", {"sl_mult": 2.0, "tp_mult": 7.0,
                           "sl_floor": 0.005, "sl_cap": 0.025,
                           "tp_floor": 0.02, "tp_cap": 0.15}),
        ("ATR_SL10_TP40", {"sl_mult": 1.0, "tp_mult": 4.0,
                           "sl_floor": 0.003, "sl_cap": 0.015,
                           "tp_floor": 0.015, "tp_cap": 0.08}),
        ("ATR_SL25_TP80", {"sl_mult": 2.5, "tp_mult": 8.0,
                           "sl_floor": 0.005, "sl_cap": 0.03,
                           "tp_floor": 0.03, "tp_cap": 0.15}),
    ]

    best_atr = None
    best_atr_calmar = bl_m["calmar"]

    for name, params in atr_configs:
        t1 = time.time()
        d_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
            "atr_exit", params,
        )
        s_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "S_d8",
            "atr_exit", params,
        )
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f}  [{elapsed:.1f}s]")

        if m["calmar"] > bl_m["calmar"] * 0.95:
            yr_str = "    Yearly ΔEV: "
            for y in sorted(yr.keys()):
                delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
                yr_str += f"{y}: {delta:>+5.1f}  "
            print(yr_str)

        if m["calmar"] > best_atr_calmar:
            best_atr_calmar = m["calmar"]
            best_atr = (name, params, m, yr)

    if best_atr:
        print(f"\n  ★ Best ATR: {best_atr[0]} — Calmar {best_atr[2]['calmar']:.1f} "
              f"(baseline {bl_m['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 5: V4 — Volume Confirmation Filter
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 5: V4 — Volume Confirmation Filter")
    print(f"  Only enter when volume percentile > threshold. Hypothesis: volume confirms breakout.")
    print(f"{'─'*100}")

    vol_configs = [
        ("Vol_P50", 50.0),
        ("Vol_P60", 60.0),
        ("Vol_P70", 70.0),
        ("Vol_P80", 80.0),
        ("Vol_P40", 40.0),
    ]

    best_vol = None
    best_vol_calmar = bl_m["calmar"]

    for name, min_pct in vol_configs:
        t1 = time.time()
        d_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
            "baseline", {}, min_vol_pct=min_pct,
        )
        s_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "S_d8",
            "baseline", {}, min_vol_pct=min_pct,
        )
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f}  [{elapsed:.1f}s]")

        if m["calmar"] > bl_m["calmar"] * 0.95:
            yr_str = "    Yearly ΔEV: "
            for y in sorted(yr.keys()):
                delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
                yr_str += f"{y}: {delta:>+5.1f}  "
            print(yr_str)

        if m["calmar"] > best_vol_calmar:
            best_vol_calmar = m["calmar"]
            best_vol = (name, min_pct, m, yr)

    if best_vol:
        print(f"\n  ★ Best Vol: {best_vol[0]} — Calmar {best_vol[2]['calmar']:.1f} "
              f"(baseline {bl_m['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 6: V5 — Symbol Exclusion
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 6: V5 — Symbol Exclusion")
    print(f"  Exclude historically losing symbols from trading universe.")
    print(f"{'─'*100}")

    # Rank symbols by total contribution
    sym_sorted = sorted(bl_sym.items(), key=lambda x: x[1]["total_bp"])
    print("\n  Bottom 15 symbols (baseline):")
    print(f"  {'Symbol':<16} {'N':>4} {'PF':>6} {'EV(bp)':>8} {'WR%':>5} {'Total(bp)':>10}")
    for sym, s in sym_sorted[:15]:
        print(f"  {sym:<16} {s['N']:>4} {s['PF']:>6.3f} {s['EV_bp']:>+8.1f} {s['WR']:>5.1f} {s['total_bp']:>+10.0f}")

    losing_symbols = [sym for sym, s in sym_sorted if s["PF"] < 1.0]
    print(f"\n  Losing symbols: {len(losing_symbols)} / {len(bl_sym)}")

    excl_configs = [
        ("Excl_PF_lt_1", {sym for sym, s in bl_sym.items() if s["PF"] < 1.0}),
        ("Excl_PF_lt_09", {sym for sym, s in bl_sym.items() if s["PF"] < 0.9}),
        ("Excl_Bottom5", {sym for sym, _ in sym_sorted[:5]}),
        ("Excl_Bottom10", {sym for sym, _ in sym_sorted[:10]}),
        ("Excl_EV_neg", {sym for sym, s in bl_sym.items() if s["EV_bp"] < 0}),
    ]

    best_excl = None
    best_excl_calmar = bl_m["calmar"]

    for name, excl_set in excl_configs:
        t1 = time.time()
        d_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
            "baseline", {}, exclude_symbols=excl_set,
        )
        s_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "S_d8",
            "baseline", {}, exclude_symbols=excl_set,
        )
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<30} excl={len(excl_set):>2}  N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f}  [{elapsed:.1f}s]")

        if m["calmar"] > bl_m["calmar"] * 0.95:
            yr_str = "    Yearly ΔEV: "
            for y in sorted(yr.keys()):
                delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
                yr_str += f"{y}: {delta:>+5.1f}  "
            print(yr_str)

        if m["calmar"] > best_excl_calmar:
            best_excl_calmar = m["calmar"]
            best_excl = (name, excl_set, m, yr)

    if best_excl:
        print(f"\n  ★ Best Excl: {best_excl[0]} — Calmar {best_excl[2]['calmar']:.1f} "
              f"(baseline {bl_m['calmar']:.1f})")
        print(f"    Excluded: {sorted(best_excl[1])}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 7: V6 — Straddle Trailing
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 7: V6 — Straddle Leg Trailing")
    print(f"  Trail winning straddle legs independently.")
    print(f"{'─'*100}")

    strd_trail_configs = [
        ("S_Trail_A3_D1", {"trail_activate": 0.03, "trail_distance": 0.01}),
        ("S_Trail_A2_D05", {"trail_activate": 0.02, "trail_distance": 0.005}),
        ("S_Trail_A4_D15", {"trail_activate": 0.04, "trail_distance": 0.015}),
        ("S_Trail_A2_D1", {"trail_activate": 0.02, "trail_distance": 0.01}),
        ("S_Trail_A3_D15", {"trail_activate": 0.03, "trail_distance": 0.015}),
    ]

    best_strd = None
    best_strd_calmar = bl_m["calmar"]

    for name, params in strd_trail_configs:
        t1 = time.time()
        # Directional baseline, straddle trailing
        d_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
        s_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "S_d8",
            "straddle_trailing", params,
        )
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f}  [{elapsed:.1f}s]")

        if m["calmar"] > bl_m["calmar"] * 0.95:
            yr_str = "    Yearly ΔEV: "
            for y in sorted(yr.keys()):
                delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
                yr_str += f"{y}: {delta:>+5.1f}  "
            print(yr_str)

        if m["calmar"] > best_strd_calmar:
            best_strd_calmar = m["calmar"]
            best_strd = (name, params, m, yr)

    if best_strd:
        print(f"\n  ★ Best S_Trail: {best_strd[0]} — Calmar {best_strd[2]['calmar']:.1f} "
              f"(baseline {bl_m['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 8: Combine Best Vectors
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 8: Combine Best Improvements")
    print(f"{'─'*100}")

    # Collect all winners
    winners = []
    if best_trail:
        winners.append(("D_Trail", best_trail))
    if best_be:
        winners.append(("D_BE", best_be))
    if best_atr:
        winners.append(("ATR_Exit", best_atr))
    if best_vol:
        winners.append(("Vol_Filter", best_vol))
    if best_excl:
        winners.append(("Sym_Excl", best_excl))
    if best_strd:
        winners.append(("S_Trail", best_strd))

    print(f"  Winners found: {len(winners)}")
    for wname, w in winners:
        print(f"    {wname}: {w[0]} — Calmar {w[2]['calmar']:.1f}")

    if len(winners) == 0:
        print("\n  No improvement found over baseline. Trying aggressive combinations...")

    # Combo 1: Best directional exit + symbol exclusion
    combo_excl = best_excl[1] if best_excl else set()

    # Try: trailing D + trailing S + symbol exclusion
    combos = []

    # C1: Best D trail + best S trail
    d_trail_params = best_trail[1] if best_trail else {"trail_activate": 0.03, "trail_distance": 0.01}
    s_trail_params = best_strd[1] if best_strd else {"trail_activate": 0.03, "trail_distance": 0.01}

    combos.append(("C1_DTrail+STrail",
                    "trailing", d_trail_params,
                    "straddle_trailing", s_trail_params,
                    set()))

    # C2: D trail + S trail + symbol exclusion
    if combo_excl:
        combos.append(("C2_DTrail+STrail+Excl",
                        "trailing", d_trail_params,
                        "straddle_trailing", s_trail_params,
                        combo_excl))

    # C3: ATR exit + symbol exclusion
    atr_params = best_atr[1] if best_atr else {"sl_mult": 1.5, "tp_mult": 5.0,
                                                 "sl_floor": 0.005, "sl_cap": 0.02,
                                                 "tp_floor": 0.02, "tp_cap": 0.10}
    combos.append(("C3_ATR+Excl",
                    "atr_exit", atr_params,
                    "atr_exit", atr_params,
                    combo_excl))

    # C4: BE + S trail
    be_params = best_be[1] if best_be else {"be_activate": 0.02}
    combos.append(("C4_BE+STrail",
                    "breakeven", be_params,
                    "straddle_trailing", s_trail_params,
                    set()))

    # C5: BE + S trail + excl
    if combo_excl:
        combos.append(("C5_BE+STrail+Excl",
                        "breakeven", be_params,
                        "straddle_trailing", s_trail_params,
                        combo_excl))

    # C6: D trail + baseline S + excl
    if combo_excl:
        combos.append(("C6_DTrail+Excl",
                        "trailing", d_trail_params,
                        "baseline", {},
                        combo_excl))

    # C7: Volume filter + D trail
    vol_min = best_vol[1] if best_vol else 60.0
    combos.append(("C7_Vol+DTrail",
                    "trailing", d_trail_params,
                    "baseline", {},
                    set()))

    best_combo = None
    best_combo_calmar = bl_m["calmar"]

    for combo_name, d_var, d_params, s_var, s_params, excl in combos:
        t1 = time.time()
        d_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
            d_var, d_params, exclude_symbols=excl,
        )
        s_trades = collect_trades_variant(
            CFG_V13, data_15m, data_1h, data_4h, "S_d8",
            s_var, s_params, exclude_symbols=excl,
        )
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {combo_name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})  "
              f"Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})  [{elapsed:.1f}s]")

        yr_str = "    Yearly ΔEV: "
        for y in sorted(yr.keys()):
            delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
            yr_str += f"{y}: {delta:>+5.1f}  "
        print(yr_str)

        if m["calmar"] > best_combo_calmar:
            best_combo_calmar = m["calmar"]
            best_combo = (combo_name, m, yr)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 9: Summary
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R43 SUMMARY — {total_time:.0f}s total")
    print(f"{'='*100}")

    print(f"\n  Baseline: N={bl_m['N']}  PF={bl_m['PF']}  EV={bl_m['EV_bp']:+.1f}bp  "
          f"WR={bl_m['WR']}%  Calmar={bl_m['calmar']}")

    # Print all best per vector
    all_bests = [
        ("V1 Trail (D)", best_trail),
        ("V2 BE (D)", best_be),
        ("V3 ATR Exit", best_atr),
        ("V4 Vol Filter", best_vol),
        ("V5 Sym Excl", best_excl),
        ("V6 S Trail", best_strd),
    ]

    print(f"\n  {'Vector':<20} {'Best Variant':<30} {'PF':>6} {'EV(bp)':>8} {'Calmar':>7} {'ΔCalmar':>8}")
    print(f"  {'─'*90}")
    for vname, best in all_bests:
        if best:
            m = best[2]
            dc = m["calmar"] - bl_m["calmar"]
            print(f"  {vname:<20} {best[0]:<30} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
                  f"{m['calmar']:>7.1f} {dc:>+8.1f}")
        else:
            print(f"  {vname:<20} {'(none beat baseline)':<30}")

    if best_combo:
        m = best_combo[1]
        dc = m["calmar"] - bl_m["calmar"]
        print(f"\n  {'BEST COMBO':<20} {best_combo[0]:<30} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['calmar']:>7.1f} {dc:>+8.1f}")
        print(f"\n  Year-by-Year:")
        yr = best_combo[2]
        for y in sorted(yr.keys()):
            bl = bl_yr[y]
            bc = yr[y]
            delta_ev = bc["EV_bp"] - bl["EV_bp"]
            delta_pf = bc["PF"] - bl["PF"]
            print(f"    {y}: N={bc['N']:>3} PF={bc['PF']:.2f} ({delta_pf:>+.2f})  "
                  f"EV={bc['EV_bp']:>+6.1f}bp ({delta_ev:>+.1f})")

    # Overall verdict
    any_improvement = any(b is not None and b[2]["calmar"] > bl_m["calmar"] for _, b in all_bests)
    combo_improvement = best_combo is not None and best_combo[1]["calmar"] > bl_m["calmar"]

    if combo_improvement:
        m = best_combo[1]
        pf_gain = (m["PF"] - bl_m["PF"]) / bl_m["PF"] * 100
        ev_gain = m["EV_bp"] - bl_m["EV_bp"]
        calmar_gain = m["calmar"] - bl_m["calmar"]
        if calmar_gain > 1.0 and ev_gain > 3:
            print(f"\n  ✅ VERDICT: IMPROVEMENT — {best_combo[0]}")
            print(f"     PF {bl_m['PF']:.3f} → {m['PF']:.3f} ({pf_gain:+.1f}%)")
            print(f"     EV {bl_m['EV_bp']:+.1f}bp → {m['EV_bp']:+.1f}bp ({ev_gain:+.1f}bp)")
            print(f"     Calmar {bl_m['calmar']:.1f} → {m['calmar']:.1f} ({calmar_gain:+.1f})")
        else:
            print(f"\n  ⚠️  VERDICT: MARGINAL — {best_combo[0]}")
            print(f"     Improvement exists but marginal. Consider further iteration.")
    elif any_improvement:
        print(f"\n  ⚠️  VERDICT: MARGINAL — individual vectors improve but combination doesn't stack.")
    else:
        print(f"\n  ❌ VERDICT: NO IMPROVEMENT — current baseline is already well-optimized.")

    print(f"\n  Commit: {commit}  |  Runtime: {total_time:.0f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
