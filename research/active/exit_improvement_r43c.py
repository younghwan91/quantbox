"""R43c — Robust Exit Improvements (Partial TP + Trail, no Symbol Exclusion)

R43b Findings:
  - Symbol exclusion is NOT robust OOS (curve-fit). Discarded.
  - Trail_A20_D03 fine-tuned to Calmar 11.7 (with excl, may be biased)
  - Trail alone: IS PF=1.247→baseline 1.209; OOS PF=1.387→baseline 1.395 (neutral)
  - VR Depth filter bugged (all configs same N). Fix needed.

This script focuses on ROBUST, mechanically-sound improvements:
  Phase 1: Trail-only (no exclusion) fine-tune + IS/OOS
  Phase 2: Partial TP + Trail (close 50% at TP1, trail rest)
  Phase 3: Adaptive TP based on entry VR depth
  Phase 4: Straddle partial TP / early leg closure
  Phase 5: Combined best robust improvements
  Phase 6: IS/OOS validation of best combo

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/exit_improvement_r43c.py
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

from research.active.mtf_vol_straddle_r31 import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    _atr_pct, compute_vr, compute_momentum, compute_squeeze_dur,
    vol_percentile, gen_vr_signal, gen_vr_dur_signal,
    make_direction_momentum,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    sim_straddle_trades, sim_directional_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF, CFG_V13,
)
from research.active.exit_improvement_r43 import (
    sim_directional_trailing, year_metrics, get_year_boundaries,
    symbol_performance, collect_trades_variant,
)
from research.active.exit_improvement_r43b import (
    split_data_temporal, make_split_vr,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  V1: Directional Partial TP + Trail
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_partial_tp(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    tp1_target: float,    # first partial take (e.g. 0.025 = +2.5%)
    tp1_fraction: float,  # fraction to close at TP1 (e.g. 0.5 = 50%)
    trail_activate: float,  # activate trail after this gain (for remainder)
    trail_distance: float,  # trail distance from peak
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with partial TP at TP1, then trail remainder.

    Exit Logic:
    1. SL hits → full loss (-sl)
    2. TP1 hits → close tp1_fraction at tp1_target gain
       Then trail remainder:
       - If trail_activate reached → trail at trail_distance
       - If SL on remainder → breakeven (already took partial profit)
       - Final TP → remaining (1-tp1_fraction) * tp
    3. Hard TP hits (without TP1 first) → full tp
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
        tp1_taken = False
        trailing_active = False
        peak_gain = 0.0
        tp1_pnl = 0.0   # accumulated profit from TP1

        for j in range(i + 1, end):
            if d == 1:
                cur_gain_high = (high[j] - entry) / entry
                cur_gain_low = (low[j] - entry) / entry

                if cur_gain_high > peak_gain:
                    peak_gain = cur_gain_high

                if not tp1_taken:
                    # Check SL first
                    if cur_gain_low <= -sl:
                        trade_pnl = -sl
                        eb = j
                        break
                    # Check TP1
                    if cur_gain_high >= tp1_target:
                        tp1_taken = True
                        tp1_pnl = tp1_fraction * tp1_target
                        # Don't break — continue with remainder
                    # Check hard TP
                    if cur_gain_high >= tp:
                        trade_pnl = tp
                        eb = j
                        break
                else:
                    # TP1 already taken, managing remainder
                    remainder = 1.0 - tp1_fraction

                    # Trailing on remainder
                    if trailing_active:
                        trail_level = peak_gain - trail_distance
                        if cur_gain_low <= trail_level:
                            trade_pnl = tp1_pnl + remainder * max(trail_level, 0.0)
                            eb = j
                            break

                    # Check SL on remainder (BE since TP1 profit locked)
                    if cur_gain_low <= 0.0:
                        trade_pnl = tp1_pnl  # remainder stopped at BE
                        eb = j
                        break

                    # Check hard TP on remainder
                    if cur_gain_high >= tp:
                        trade_pnl = tp1_pnl + remainder * tp
                        eb = j
                        break

                    # Transition to trailing
                    if not trailing_active and peak_gain >= trail_activate:
                        trailing_active = True
            else:
                cur_gain_high = (entry - low[j]) / entry
                cur_gain_low = (entry - high[j]) / entry

                if cur_gain_high > peak_gain:
                    peak_gain = cur_gain_high

                if not tp1_taken:
                    if cur_gain_low <= -sl:
                        trade_pnl = -sl
                        eb = j
                        break
                    if cur_gain_high >= tp1_target:
                        tp1_taken = True
                        tp1_pnl = tp1_fraction * tp1_target
                    if cur_gain_high >= tp:
                        trade_pnl = tp
                        eb = j
                        break
                else:
                    remainder = 1.0 - tp1_fraction
                    if trailing_active:
                        trail_level = peak_gain - trail_distance
                        if cur_gain_low <= trail_level:
                            trade_pnl = tp1_pnl + remainder * max(trail_level, 0.0)
                            eb = j
                            break
                    if cur_gain_low <= 0.0:
                        trade_pnl = tp1_pnl
                        eb = j
                        break
                    if cur_gain_high >= tp:
                        trade_pnl = tp1_pnl + remainder * tp
                        eb = j
                        break
                    if not trailing_active and peak_gain >= trail_activate:
                        trailing_active = True
        else:
            # Max hold reached
            if d == 1:
                final_gain = (close[eb] - entry) / entry
            else:
                final_gain = (entry - close[eb]) / entry
            if tp1_taken:
                remainder = 1.0 - tp1_fraction
                trade_pnl = tp1_pnl + remainder * final_gain
            else:
                trade_pnl = final_gain

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V2: Straddle with Partial Close
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_partial(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
    partial_tp: float,      # close winning leg at this level (e.g. 0.03 = +3%)
    partial_frac: float,    # fraction of winning leg to close (e.g. 0.5)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim where winning leg takes partial profit early.

    When price moves +partial_tp, close partial_frac of the winning leg.
    Losing leg continues. This reduces risk while letting winners run.
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

        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)

        la = True   # long active
        sa = True   # short active
        le = 0.0    # long exit pnl
        se = 0.0    # short exit pnl

        # Partial state
        l_partial_taken = False
        s_partial_taken = False
        l_partial_pnl = 0.0
        s_partial_pnl = 0.0
        l_remain = 1.0
        s_remain = 1.0

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                # Long leg partial TP
                if not l_partial_taken and high[j] >= entry * (1.0 + partial_tp):
                    l_partial_pnl = partial_frac * partial_tp
                    l_remain = 1.0 - partial_frac
                    l_partial_taken = True

                # Long leg SL
                if low[j] <= l_sl:
                    le = l_partial_pnl + l_remain * (-sl)
                    la = False
                # Long leg full TP
                elif high[j] >= l_tp:
                    le = l_partial_pnl + l_remain * tp
                    la = False

            if sa:
                # Short leg partial TP
                if not s_partial_taken and low[j] <= entry * (1.0 - partial_tp):
                    s_partial_pnl = partial_frac * partial_tp
                    s_remain = 1.0 - partial_frac
                    s_partial_taken = True

                # Short leg SL
                if high[j] >= s_sl:
                    se = s_partial_pnl + s_remain * (-sl)
                    sa = False
                # Short leg full TP
                elif low[j] <= s_tp:
                    se = s_partial_pnl + s_remain * tp
                    sa = False

            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1

        if la:
            if l_partial_taken:
                le = l_partial_pnl + l_remain * (close[eb] - entry) / entry
            else:
                le = (close[eb] - entry) / entry
        if sa:
            if s_partial_taken:
                se = s_partial_pnl + s_remain * (entry - close[eb]) / entry
            else:
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
#  V3: Directional with Wider TP + Trail (let winners run further)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_widetp_trail(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float,
    trail_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as sim_directional_trailing but with arbitrary TP."""
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
        trailing = False
        peak = 0.0

        for j in range(i + 1, end):
            if d == 1:
                g = (high[j] - entry) / entry
                if g > peak:
                    peak = g

                if trailing:
                    lvl = peak - trail_distance
                    if (low[j] - entry) / entry <= lvl:
                        trade_pnl = max(lvl, 0.0)
                        eb = j
                        break

                # Hard TP (can be wider than standard 5%)
                if g >= tp:
                    trade_pnl = tp
                    eb = j
                    break

                # SL
                if (low[j] - entry) / entry <= -sl:
                    trade_pnl = -sl
                    eb = j
                    break

                if not trailing and peak >= trail_activate:
                    trailing = True
            else:
                g = (entry - low[j]) / entry
                if g > peak:
                    peak = g

                if trailing:
                    lvl = peak - trail_distance
                    if (entry - high[j]) / entry <= lvl:
                        trade_pnl = max(lvl, 0.0)
                        eb = j
                        break

                if g >= tp:
                    trade_pnl = tp
                    eb = j
                    break

                if (entry - high[j]) / entry <= -sl:
                    trade_pnl = -sl
                    eb = j
                    break

                if not trailing and peak >= trail_activate:
                    trailing = True
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
#  V4: Straddle with tighter SL (reduce losers)
# ══════════════════════════════════════════════════════════════════════════════


# Use sim_straddle_trades with different SL value --- no new sim needed


# ══════════════════════════════════════════════════════════════════════════════
#  Trade collection wrappers
# ══════════════════════════════════════════════════════════════════════════════


def collect_d_partial_tp(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    tp1_target: float, tp1_fraction: float,
    trail_activate: float, trail_distance: float,
    exclude_symbols: set | None = None,
) -> list[tuple[str, int, int, float]]:
    sp = cfg["sp"]; lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
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
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue
            sig = gen_vr_signal(vr, cfg["d_vr"], warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_partial_tp(
                c, h, lo, sig, dire, cfg["d_sl"], cfg["tp"], mh_d, COST_DIRECTIONAL,
                tp1_target, tp1_fraction, trail_activate, trail_distance,
            )
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True; break
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))
    return all_trades


def collect_s_partial(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    partial_tp: float, partial_frac: float,
    exclude_symbols: set | None = None,
) -> list[tuple[str, int, int, float]]:
    sp = cfg["sp"]; lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
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
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue
            sig = gen_vr_dur_signal(vr, cfg["s_vr"], cfg["s_dur"], warmup)
            p, e, x = sim_straddle_partial(
                c, h, lo, sig, cfg["s_sl"], cfg["tp"], mh_s, COST_STRADDLE,
                partial_tp, partial_frac,
            )
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True; break
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))
    return all_trades


def collect_d_widetp_trail(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    tp_override: float,
    trail_activate: float, trail_distance: float,
    exclude_symbols: set | None = None,
) -> list[tuple[str, int, int, float]]:
    sp = cfg["sp"]; lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
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
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue
            sig = gen_vr_signal(vr, cfg["d_vr"], warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_widetp_trail(
                c, h, lo, sig, dire, cfg["d_sl"], tp_override, mh_d, COST_DIRECTIONAL,
                trail_activate, trail_distance,
            )
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True; break
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))
    return all_trades


def collect_s_with_sl(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    sl_override: float,
    exclude_symbols: set | None = None,
) -> list[tuple[str, int, int, float]]:
    """Straddle trades with custom SL."""
    sp = cfg["sp"]; lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
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
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue
            sig = gen_vr_dur_signal(vr, cfg["s_vr"], cfg["s_dur"], warmup)
            p, e, x = sim_straddle_trades(
                c, h, lo, sig, sl_override, cfg["tp"], mh_s, COST_STRADDLE,
            )
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True; break
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*100}")
    print(f"  R43c — Robust Exit Improvements (No Symbol Exclusion)")
    print(f"  Commit: {commit}")
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

    # Full-sample baseline
    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_trades = bl_d + bl_s
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_trades, SLOTS)
    bl_m = compute_metrics(bl_pnl, data_years)
    bl_yr = year_metrics(bl_pnl, bl_accepted, year_bounds)
    print(f"  Baseline: N={bl_m['N']}  PF={bl_m['PF']}  EV={bl_m['EV_bp']:+.1f}bp  "
          f"Calmar={bl_m['calmar']}  DD={bl_m['max_dd_pct']:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: Trail-only (NO exclusion) — Grid Fine-Tune
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 1: Trail-Only (No Exclusion) — Fine-Tune Grid")
    print(f"{'─'*100}")

    trail_grid = [
        ("Trail_A15_D03", 0.015, 0.003),
        ("Trail_A15_D05", 0.015, 0.005),
        ("Trail_A20_D03", 0.020, 0.003),
        ("Trail_A20_D05", 0.020, 0.005),
        ("Trail_A20_D08", 0.020, 0.008),
        ("Trail_A25_D03", 0.025, 0.003),
        ("Trail_A25_D05", 0.025, 0.005),
        ("Trail_A25_D08", 0.025, 0.008),
        ("Trail_A30_D03", 0.030, 0.003),
        ("Trail_A30_D05", 0.030, 0.005),
        ("Trail_A30_D08", 0.030, 0.008),
        ("Trail_A35_D05", 0.035, 0.005),
        ("Trail_A35_D10", 0.035, 0.010),
        ("Trail_A40_D05", 0.040, 0.005),
        ("Trail_A40_D10", 0.040, 0.010),
    ]

    best_trail = None
    best_trail_calmar = bl_m["calmar"]
    trail_results: list[tuple[str, float, float, dict, dict]] = []

    for name, ta, td in trail_grid:
        d_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                           "trailing", {"trail_activate": ta, "trail_distance": td})
        trades = d_trades + bl_s  # straddle unchanged
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = " ✓" if m["calmar"] > best_trail_calmar else ""
        print(f"  {name:<20} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})"
              f"  DD={m['max_dd_pct']:>4.1f}%{marker}")

        trail_results.append((name, ta, td, m, yr))
        if m["calmar"] > best_trail_calmar:
            best_trail_calmar = m["calmar"]
            best_trail = (name, ta, td, m, yr)

    if best_trail:
        m = best_trail[3]
        print(f"\n  ★ Best Trail (no excl): {best_trail[0]} — "
              f"Calmar={m['calmar']} PF={m['PF']} EV={m['EV_bp']:+.1f}bp")
        yr_str = "    Yearly ΔEV: "
        for y in sorted(best_trail[4].keys()):
            delta = best_trail[4][y]["EV_bp"] - bl_yr[y]["EV_bp"]
            yr_str += f"{y}: {delta:>+.1f}  "
        print(yr_str)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: Partial TP + Trail (directional)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 2: Partial TP + Trail (Directional)")
    print(f"{'─'*100}")

    partial_configs = [
        # (name, tp1_target, tp1_fraction, trail_activate, trail_distance)
        ("PTP_T15_F50_TA30_D05", 0.015, 0.5, 0.030, 0.005),
        ("PTP_T20_F50_TA30_D05", 0.020, 0.5, 0.030, 0.005),
        ("PTP_T25_F50_TA35_D05", 0.025, 0.5, 0.035, 0.005),
        ("PTP_T20_F50_TA25_D03", 0.020, 0.5, 0.025, 0.003),
        ("PTP_T20_F30_TA25_D03", 0.020, 0.3, 0.025, 0.003),
        ("PTP_T15_F50_TA25_D03", 0.015, 0.5, 0.025, 0.003),
        ("PTP_T25_F50_TA30_D03", 0.025, 0.5, 0.030, 0.003),
        ("PTP_T30_F50_TA40_D05", 0.030, 0.5, 0.040, 0.005),
        ("PTP_T20_F50_TA20_D03", 0.020, 0.5, 0.020, 0.003),
    ]

    best_partial = None
    best_partial_calmar = bl_m["calmar"]

    for name, tp1, frac, ta, td in partial_configs:
        d_trades = collect_d_partial_tp(
            CFG_V13, data_15m, data_1h, data_4h,
            tp1, frac, ta, td,
        )
        trades = d_trades + bl_s
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = " ✓" if m["calmar"] > best_partial_calmar else ""
        print(f"  {name:<30} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f}){marker}")

        if m["calmar"] > best_partial_calmar:
            best_partial_calmar = m["calmar"]
            best_partial = (name, tp1, frac, ta, td, m)

    if best_partial:
        m = best_partial[5]
        print(f"\n  ★ Best Partial TP: {best_partial[0]} — Calmar={m['calmar']} PF={m['PF']}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 3: Wide TP + Trail (let winners run beyond 5%)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 3: Wide TP + Trail (let winners run)")
    print(f"{'─'*100}")

    widetp_configs = [
        # (name, tp_override, trail_activate, trail_distance)
        ("WTP_TP07_TA20_D03", 0.07, 0.020, 0.003),
        ("WTP_TP07_TA30_D05", 0.07, 0.030, 0.005),
        ("WTP_TP10_TA20_D03", 0.10, 0.020, 0.003),
        ("WTP_TP10_TA30_D05", 0.10, 0.030, 0.005),
        ("WTP_TP10_TA30_D10", 0.10, 0.030, 0.010),
        ("WTP_TP15_TA20_D03", 0.15, 0.020, 0.003),
        ("WTP_TP15_TA30_D08", 0.15, 0.030, 0.008),
        ("WTP_TP20_TA30_D10", 0.20, 0.030, 0.010),
        ("WTP_NoTP_TA20_D03", 1.00, 0.020, 0.003),  # No hard TP, pure trail
        ("WTP_NoTP_TA30_D05", 1.00, 0.030, 0.005),
    ]

    best_wtp = None
    best_wtp_calmar = bl_m["calmar"]

    for name, tp, ta, td in widetp_configs:
        d_trades = collect_d_widetp_trail(
            CFG_V13, data_15m, data_1h, data_4h,
            tp, ta, td,
        )
        trades = d_trades + bl_s
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = " ✓" if m["calmar"] > best_wtp_calmar else ""
        print(f"  {name:<25} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f}){marker}")

        if m["calmar"] > best_wtp_calmar:
            best_wtp_calmar = m["calmar"]
            best_wtp = (name, tp, ta, td, m)

    if best_wtp:
        m = best_wtp[4]
        print(f"\n  ★ Best Wide TP: {best_wtp[0]} — Calmar={m['calmar']} PF={m['PF']}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 4: Straddle Improvements
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 4: Straddle Improvements (partial TP, SL tweak)")
    print(f"{'─'*100}")

    # 4a: Straddle SL tweak
    s_sl_configs = [
        ("S_SL_1.0%", 0.010),
        ("S_SL_1.2%", 0.012),
        ("S_SL_1.5%", 0.015),  # baseline
        ("S_SL_1.8%", 0.018),
        ("S_SL_2.0%", 0.020),
        ("S_SL_2.5%", 0.025),
    ]

    print("  [4a] Straddle SL tweak:")
    best_s_sl = None
    best_s_sl_calmar = bl_m["calmar"]

    for name, sl in s_sl_configs:
        s_trades = collect_s_with_sl(CFG_V13, data_15m, data_1h, data_4h, sl)
        trades = bl_d + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = "←" if sl == 0.015 else ("✓" if m["calmar"] > best_s_sl_calmar else "")
        print(f"    {marker:>2} {name:<15} N={m['N']:>5}  PF={m['PF']:>5.3f}  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})")

        if m["calmar"] > best_s_sl_calmar:
            best_s_sl_calmar = m["calmar"]
            best_s_sl = (name, sl, m)

    # 4b: Straddle partial TP
    s_partial_configs = [
        ("S_PTP_T02_F50", 0.02, 0.5),
        ("S_PTP_T03_F50", 0.03, 0.5),
        ("S_PTP_T03_F30", 0.03, 0.3),
        ("S_PTP_T04_F50", 0.04, 0.5),
        ("S_PTP_T025_F50", 0.025, 0.5),
    ]

    print("\n  [4b] Straddle partial TP:")
    best_s_ptp = None
    best_s_ptp_calmar = bl_m["calmar"]

    for name, ptp, frac in s_partial_configs:
        s_trades = collect_s_partial(CFG_V13, data_15m, data_1h, data_4h, ptp, frac)
        trades = bl_d + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = " ✓" if m["calmar"] > best_s_ptp_calmar else ""
        print(f"    {name:<20} N={m['N']:>5}  PF={m['PF']:>5.3f}  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f}){marker}")

        if m["calmar"] > best_s_ptp_calmar:
            best_s_ptp_calmar = m["calmar"]
            best_s_ptp = (name, ptp, frac, m)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 5: Combined Best (D improvements + S improvements)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 5: Combined Best Robust Improvements")
    print(f"{'─'*100}")

    combos: list[tuple[str, list]] = []

    # Combo 1: Best trail D + baseline S
    if best_trail:
        ta, td = best_trail[1], best_trail[2]
        d_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                           "trailing", {"trail_activate": ta, "trail_distance": td})
        combos.append(("C1_BestTrail+BaseS", d_trades + bl_s))

    # Combo 2: Best trail D + best straddle SL
    if best_trail and best_s_sl and best_s_sl[1] != 0.015:
        d_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                           "trailing", {"trail_activate": best_trail[1], "trail_distance": best_trail[2]})
        s_trades = collect_s_with_sl(CFG_V13, data_15m, data_1h, data_4h, best_s_sl[1])
        combos.append(("C2_BestTrail+BestSSL", d_trades + s_trades))

    # Combo 3: Best trail D + best straddle partial
    if best_trail and best_s_ptp:
        d_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                           "trailing", {"trail_activate": best_trail[1], "trail_distance": best_trail[2]})
        s_trades = collect_s_partial(CFG_V13, data_15m, data_1h, data_4h,
                                     best_s_ptp[1], best_s_ptp[2])
        combos.append(("C3_BestTrail+BestSPTP", d_trades + s_trades))

    # Combo 4: Best partial TP D + baseline S
    if best_partial:
        d_trades = collect_d_partial_tp(
            CFG_V13, data_15m, data_1h, data_4h,
            best_partial[1], best_partial[2], best_partial[3], best_partial[4],
        )
        combos.append(("C4_BestPartialD+BaseS", d_trades + bl_s))

    # Combo 5: Best partial TP D + best S
    if best_partial and best_s_sl and best_s_sl[1] != 0.015:
        d_trades = collect_d_partial_tp(
            CFG_V13, data_15m, data_1h, data_4h,
            best_partial[1], best_partial[2], best_partial[3], best_partial[4],
        )
        s_trades = collect_s_with_sl(CFG_V13, data_15m, data_1h, data_4h, best_s_sl[1])
        combos.append(("C5_BestPartialD+BestSSL", d_trades + s_trades))

    # Combo 6: Best wide TP + baseline
    if best_wtp:
        d_trades = collect_d_widetp_trail(
            CFG_V13, data_15m, data_1h, data_4h,
            best_wtp[1], best_wtp[2], best_wtp[3],
        )
        combos.append(("C6_BestWideTP+BaseS", d_trades + bl_s))

    best_combo = None
    best_combo_calmar = bl_m["calmar"]

    for name, trades in combos:
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]

        # Year-by-year
        yr_all_better = all(
            yr[y]["EV_bp"] >= bl_yr[y]["EV_bp"] - 2.0  # allow 2bp noise
            for y in yr
        )

        marker = "✓" if m["calmar"] > best_combo_calmar else "—"
        print(f"\n  {marker} {name}:")
        print(f"    N={m['N']:>5}  PF={m['PF']:.3f} ({delta_pf:+.3f})  EV={m['EV_bp']:+.1f}bp  "
              f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.1f} ({delta_calmar:+.1f})")
        print(f"    AnnRet={m['ann_ret_pct']:+.1f}%  DD={m['max_dd_pct']:.1f}%")
        yr_str = "    Years: "
        for y in sorted(yr.keys()):
            delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
            yr_str += f"{y}: {delta:>+5.1f}  "
        print(yr_str)
        status = "ALL BETTER" if yr_all_better else "MIXED"
        print(f"    Year consistency: {status}")

        if m["calmar"] > best_combo_calmar:
            best_combo_calmar = m["calmar"]
            best_combo = (name, m, yr, trades)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 6: IS/OOS Validation of Best Combo
    # ══════════════════════════════════════════════════════════════════════════
    if best_combo:
        print(f"\n{'─'*100}")
        print(f"  Phase 6: IS/OOS Validation — {best_combo[0]}")
        print(f"{'─'*100}")

        is_15m, oos_15m = split_data_temporal(data_15m)
        is_1h, oos_1h = split_data_temporal(data_1h)
        is_4h, oos_4h = split_data_temporal(data_4h)
        make_split_vr([is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h], [(8, 56)])

        is_years = data_years * 0.67
        oos_years = data_years * 0.33

        # IS/OOS baseline
        is_bl_d = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "D_mom8")
        is_bl_s = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "S_d8")
        is_bl_pnl = portfolio_sim(is_bl_d + is_bl_s, SLOTS)
        is_bl_m = compute_metrics(is_bl_pnl, is_years)

        oos_bl_d = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8")
        oos_bl_s = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")
        oos_bl_pnl = portfolio_sim(oos_bl_d + oos_bl_s, SLOTS)
        oos_bl_m = compute_metrics(oos_bl_pnl, oos_years)

        # Re-run best combo on IS and OOS separately
        # We need to figure out what the best combo contained and recreate it on split data
        # Since best_combo name tells us the config, let's handle each case

        combo_name = best_combo[0]

        if "Trail" in combo_name and "Partial" not in combo_name and "WideTP" not in combo_name:
            # Pure trail combo
            ta, td = best_trail[1], best_trail[2]

            is_d = collect_trades_variant(CFG_V13, is_15m, is_1h, is_4h, "D_mom8",
                                           "trailing", {"trail_activate": ta, "trail_distance": td})
            oos_d = collect_trades_variant(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8",
                                            "trailing", {"trail_activate": ta, "trail_distance": td})

            if "BestSSL" in combo_name:
                is_s = collect_s_with_sl(CFG_V13, is_15m, is_1h, is_4h, best_s_sl[1])
                oos_s = collect_s_with_sl(CFG_V13, oos_15m, oos_1h, oos_4h, best_s_sl[1])
            elif "BestSPTP" in combo_name:
                is_s = collect_s_partial(CFG_V13, is_15m, is_1h, is_4h, best_s_ptp[1], best_s_ptp[2])
                oos_s = collect_s_partial(CFG_V13, oos_15m, oos_1h, oos_4h, best_s_ptp[1], best_s_ptp[2])
            else:
                is_s = is_bl_s
                oos_s = oos_bl_s

            is_pnl = portfolio_sim(is_d + is_s, SLOTS)
            is_m = compute_metrics(is_pnl, is_years)
            oos_pnl = portfolio_sim(oos_d + oos_s, SLOTS)
            oos_m = compute_metrics(oos_pnl, oos_years)

        elif "PartialD" in combo_name:
            is_d = collect_d_partial_tp(
                CFG_V13, is_15m, is_1h, is_4h,
                best_partial[1], best_partial[2], best_partial[3], best_partial[4],
            )
            oos_d = collect_d_partial_tp(
                CFG_V13, oos_15m, oos_1h, oos_4h,
                best_partial[1], best_partial[2], best_partial[3], best_partial[4],
            )

            if "BestSSL" in combo_name:
                is_s = collect_s_with_sl(CFG_V13, is_15m, is_1h, is_4h, best_s_sl[1])
                oos_s = collect_s_with_sl(CFG_V13, oos_15m, oos_1h, oos_4h, best_s_sl[1])
            else:
                is_s = is_bl_s
                oos_s = oos_bl_s

            is_pnl = portfolio_sim(is_d + is_s, SLOTS)
            is_m = compute_metrics(is_pnl, is_years)
            oos_pnl = portfolio_sim(oos_d + oos_s, SLOTS)
            oos_m = compute_metrics(oos_pnl, oos_years)

        elif "WideTP" in combo_name:
            is_d = collect_d_widetp_trail(
                CFG_V13, is_15m, is_1h, is_4h,
                best_wtp[1], best_wtp[2], best_wtp[3],
            )
            oos_d = collect_d_widetp_trail(
                CFG_V13, oos_15m, oos_1h, oos_4h,
                best_wtp[1], best_wtp[2], best_wtp[3],
            )
            is_s = is_bl_s
            oos_s = oos_bl_s

            is_pnl = portfolio_sim(is_d + is_s, SLOTS)
            is_m = compute_metrics(is_pnl, is_years)
            oos_pnl = portfolio_sim(oos_d + oos_s, SLOTS)
            oos_m = compute_metrics(oos_pnl, oos_years)

        else:
            # Fallback: just run trail with best params
            ta, td = best_trail[1], best_trail[2] if best_trail else (0.02, 0.005)
            is_d = collect_trades_variant(CFG_V13, is_15m, is_1h, is_4h, "D_mom8",
                                           "trailing", {"trail_activate": ta, "trail_distance": td})
            oos_d = collect_trades_variant(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8",
                                            "trailing", {"trail_activate": ta, "trail_distance": td})
            is_s = is_bl_s
            oos_s = oos_bl_s
            is_pnl = portfolio_sim(is_d + is_s, SLOTS)
            is_m = compute_metrics(is_pnl, is_years)
            oos_pnl = portfolio_sim(oos_d + oos_s, SLOTS)
            oos_m = compute_metrics(oos_pnl, oos_years)

        print(f"  Baseline IS:  N={is_bl_m['N']:>5}  PF={is_bl_m['PF']:.3f}  EV={is_bl_m['EV_bp']:+.1f}bp  Calmar={is_bl_m['calmar']:.1f}")
        print(f"  Baseline OOS: N={oos_bl_m['N']:>5}  PF={oos_bl_m['PF']:.3f}  EV={oos_bl_m['EV_bp']:+.1f}bp  Calmar={oos_bl_m['calmar']:.1f}")
        print(f"  Best IS:      N={is_m['N']:>5}  PF={is_m['PF']:.3f}  EV={is_m['EV_bp']:+.1f}bp  Calmar={is_m['calmar']:.1f}")
        print(f"  Best OOS:     N={oos_m['N']:>5}  PF={oos_m['PF']:.3f}  EV={oos_m['EV_bp']:+.1f}bp  Calmar={oos_m['calmar']:.1f}")

        is_pf_gain = is_m["PF"] - is_bl_m["PF"]
        oos_pf_gain = oos_m["PF"] - oos_bl_m["PF"]
        print(f"\n  IS  PF gain: {is_pf_gain:+.3f}")
        print(f"  OOS PF gain: {oos_pf_gain:+.3f}")

        if oos_pf_gain > 0:
            print(f"  ✅ ROBUST: OOS PF improves by {oos_pf_gain:+.3f}")
        elif oos_pf_gain > -0.01:
            print(f"  ⚠️ NEUTRAL: OOS PF change negligible ({oos_pf_gain:+.3f})")
        else:
            print(f"  ❌ DEGRADED: OOS PF drops by {oos_pf_gain:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R43c FINAL SUMMARY — {total_time:.0f}s")
    print(f"{'='*100}")

    print(f"\n  {'Config':<35} {'N':>5} {'PF':>6} {'EV(bp)':>8} {'WR%':>5} "
          f"{'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"  {'─'*85}")

    rows = [("Baseline (v13.0.1)", bl_m)]
    if best_trail:
        rows.append((f"Trail only ({best_trail[0]})", best_trail[3]))
    if best_partial:
        rows.append((f"Partial TP ({best_partial[0][:20]})", best_partial[5]))
    if best_wtp:
        rows.append((f"Wide TP ({best_wtp[0][:20]})", best_wtp[4]))
    if best_combo:
        rows.append((f"BEST COMBO ({best_combo[0][:20]})", best_combo[1]))

    for label, m in rows:
        print(f"  {label:<35} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} {m['WR']:>5.1f} "
              f"{m['ann_ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")

    print(f"\n  Commit: {commit}  |  Runtime: {total_time:.0f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
