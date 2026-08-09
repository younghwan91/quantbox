"""R43b — Validation & Fine-Tuning of R43 C6 (DTrail + SymExcl)

R43 found: C6_DTrail+Excl — Calmar 5.8 → 10.3 (+78%)
  - Trail_A2_D05: activate +2%, trail 0.5%
  - Excl_Bottom10: remove 10 worst symbols

This script validates:
  Phase 1: IS/OOS temporal split (67%/33%) — does C6 work OOS?
  Phase 2: Walk-forward symbol filter (rolling 2yr lookback)
  Phase 3: Fine-tune trailing stop parameters (grid around A2_D05 sweet spot)
  Phase 4: Additional directional exit improvements  
  Phase 5: VR depth scoring — deeper squeeze = better signal?
  Phase 6: Directional-only SL tweak (tighter/wider SL with trail)
  Phase 7: Best combos with validated symbol filter

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/exit_improvement_r43b.py
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
    make_direction_momentum,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    sim_straddle_trades, sim_directional_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF, CFG_V13,
)
from research.active.r41_r45.r43_exit_improvement import (
    collect_trades_variant,
    year_metrics, get_year_boundaries, symbol_performance,
    print_results_table,
)
from quantbox.backtest.trade_sim import sim_directional_trailing


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  V5b: VR Depth Signal — only enter deep squeezees
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def gen_vr_signal_depth(
    vr: np.ndarray, threshold: float, warmup: int,
    depth_lookback: int, depth_pctile: float,
) -> np.ndarray:
    """VR signal that only fires when VR is in deeper percentile of recent VR values.

    depth_lookback: how many bars to look back for VR history
    depth_pctile: only fire if current VR is below this percentile of recent VR values
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(max(warmup, depth_lookback), n):
        if np.isnan(vr[i]) or vr[i] >= threshold:
            continue
        # Check if this is a deep squeeze (VR below recent percentile)
        below = 0
        valid = 0
        for k in range(depth_lookback):
            v = vr[i - depth_lookback + k]
            if not np.isnan(v):
                valid += 1
                if vr[i] < v:
                    below += 1
        if valid > 0 and (below / valid) <= depth_pctile:
            sig[i] = 1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  V6b: Directional with tweaked SL + trail
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_directional_trail_tightsl(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float,
    trail_distance: float,
    be_activate: float,   # move SL to breakeven after this gain (before trail kicks in)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with 3-stage exit:
    1. Initial SL
    2. After be_activate gain → move SL to breakeven
    3. After trail_activate gain → trail SL at trail_distance from peak
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
        be_active = False
        peak_gain = 0.0

        for j in range(i + 1, end):
            if d == 1:
                cur_high_gain = (high[j] - entry) / entry
                if cur_high_gain > peak_gain:
                    peak_gain = cur_high_gain

                # Stage 3: Trailing stop
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, 0.0)
                        eb = j
                        break
                # Stage 2: Breakeven stop
                elif be_active:
                    if low[j] <= entry:
                        trade_pnl = 0.0
                        eb = j
                        break

                # Check hard TP
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break

                # Stage 1: Initial SL
                if not be_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break

                # Transitions
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                elif not be_active and peak_gain >= be_activate:
                    be_active = True
            else:
                cur_high_gain = (entry - low[j]) / entry
                if cur_high_gain > peak_gain:
                    peak_gain = cur_high_gain

                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, 0.0)
                        eb = j
                        break
                elif be_active:
                    if high[j] >= entry:
                        trade_pnl = 0.0
                        eb = j
                        break

                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break

                if not be_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break

                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                elif not be_active and peak_gain >= be_activate:
                    be_active = True
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


def collect_directional_trail3_fn(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    trail_activate: float, trail_distance: float, be_activate: float,
    sl_override: float | None = None,
    exclude_symbols: set | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect directional trades with 3-stage exit."""
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    d_sl = sl_override if sl_override is not None else cfg["d_sl"]

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

            p, e, x = sim_directional_trail_tightsl(
                c, h, lo, sig, dire,
                d_sl, cfg["tp"], mh_d, COST_DIRECTIONAL,
                trail_activate, trail_distance, be_activate,
            )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

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


def collect_directional_depth(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    depth_lookback: int, depth_pctile: float,
    exclude_symbols: set | None = None,
    trail_activate: float = 0.0, trail_distance: float = 0.0,
) -> list[tuple[str, int, int, float]]:
    """Collect directional trades with VR depth filter + optional trailing."""
    sp = cfg["sp"]
    lp = cfg["lp"]
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

            sig = gen_vr_signal_depth(vr, cfg["d_vr"], warmup, depth_lookback, depth_pctile)
            dire = make_direction_momentum(sig, d["mom_8"])

            if trail_activate > 0:
                p, e, x = sim_directional_trailing(
                    c, h, lo, sig, dire,
                    cfg["d_sl"], cfg["tp"], mh_d, COST_DIRECTIONAL,
                    trail_activate, trail_distance,
                )
            else:
                p, e, x = sim_directional_trades(
                    c, h, lo, sig, dire,
                    cfg["d_sl"], cfg["tp"], mh_d, COST_DIRECTIONAL,
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

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
#  IS/OOS Split Helpers
# ══════════════════════════════════════════════════════════════════════════════


def split_data_temporal(data: dict, ratio: float = 0.67) -> tuple[dict, dict]:
    """Split each symbol's data temporally into IS and OOS."""
    is_data: dict = {}
    oos_data: dict = {}
    for sym, d in data.items():
        n = len(d["close"])
        cut = int(n * ratio)
        if cut < 200 or n - cut < 200:
            continue
        for start, end_b, target in [(0, cut, is_data), (cut, n, oos_data)]:
            seg: dict = {}
            for key in d:
                if isinstance(d[key], np.ndarray) and len(d[key]) == n:
                    seg[key] = d[key][start:end_b].copy()
                else:
                    seg[key] = d[key]
            seg["bars"] = end_b - start
            target[sym] = seg
    return is_data, oos_data


def make_split_vr(datasets: list[dict], pairs: list[tuple[int, int]]) -> None:
    """Recompute VR and momentum for split data."""
    for dataset in datasets:
        for sym, d in dataset.items():
            c, h, lo = d["close"], d["high"], d["low"]
            d["mom_8"] = compute_momentum(c, 8)
            for sp, lp in pairs:
                d[f"vr_{sp}_{lp}"] = compute_vr(c, h, lo, sp, lp)


# ══════════════════════════════════════════════════════════════════════════════
#  Walk-Forward Symbol Filter
# ══════════════════════════════════════════════════════════════════════════════


def walk_forward_symbol_filter(
    data_15m: dict, data_1h: dict, data_4h: dict,
    lookback_years: float,
    min_trades: int = 10,
    min_pf: float = 1.0,
) -> dict[int, set[str]]:
    """For each year, determine which symbols to exclude based on prior performance.

    Returns dict: {year: set of symbols to exclude}
    No lookahead: year Y exclusion uses only data from years < Y.
    """
    bars_per_year = 365 * 24 * 4
    max_bars = max(len(d["close"]) for d in data_15m.values())
    n_years = int(max_bars / bars_per_year) + 1

    exclusions: dict[int, set[str]] = {}

    for yr_idx in range(n_years):
        year = 2020 + yr_idx
        # Lookback window for evaluation
        eval_end = yr_idx * bars_per_year
        eval_start = max(0, int(eval_end - lookback_years * bars_per_year))

        if eval_end <= eval_start or eval_start < bars_per_year:
            # Not enough history (first year or so)
            exclusions[year] = set()
            continue

        # Create subset of data for evaluation period
        eval_15m: dict = {}
        eval_1h: dict = {}
        eval_4h: dict = {}

        for sym in data_15m:
            n = len(data_15m[sym]["close"])
            if eval_end > n:
                continue
            # 15m slice
            seg: dict = {}
            for key in data_15m[sym]:
                if isinstance(data_15m[sym][key], np.ndarray) and len(data_15m[sym][key]) == n:
                    seg[key] = data_15m[sym][key][eval_start:eval_end].copy()
                else:
                    seg[key] = data_15m[sym][key]
            seg["bars"] = eval_end - eval_start
            c, h, lo = seg["close"], seg["high"], seg["low"]
            seg["mom_8"] = compute_momentum(c, 8)
            seg["vr_8_56"] = compute_vr(c, h, lo, 8, 56)
            eval_15m[sym] = seg

            # 1h slice
            if sym in data_1h:
                n1h = len(data_1h[sym]["close"])
                ratio = n // n1h
                s1h = eval_start // ratio
                e1h = eval_end // ratio
                seg1h: dict = {}
                for key in data_1h[sym]:
                    if isinstance(data_1h[sym][key], np.ndarray) and len(data_1h[sym][key]) == n1h:
                        seg1h[key] = data_1h[sym][key][s1h:e1h].copy()
                    else:
                        seg1h[key] = data_1h[sym][key]
                seg1h["bars"] = e1h - s1h
                c, h, lo = seg1h["close"], seg1h["high"], seg1h["low"]
                seg1h["mom_8"] = compute_momentum(c, 8)
                seg1h["vr_8_56"] = compute_vr(c, h, lo, 8, 56)
                eval_1h[sym] = seg1h

            # 4h slice
            if sym in data_4h:
                n4h = len(data_4h[sym]["close"])
                ratio4h = n // n4h
                s4h = eval_start // ratio4h
                e4h = eval_end // ratio4h
                seg4h: dict = {}
                for key in data_4h[sym]:
                    if isinstance(data_4h[sym][key], np.ndarray) and len(data_4h[sym][key]) == n4h:
                        seg4h[key] = data_4h[sym][key][s4h:e4h].copy()
                    else:
                        seg4h[key] = data_4h[sym][key]
                seg4h["bars"] = e4h - s4h
                c, h, lo = seg4h["close"], seg4h["high"], seg4h["low"]
                seg4h["mom_8"] = compute_momentum(c, 8)
                seg4h["vr_8_56"] = compute_vr(c, h, lo, 8, 56)
                eval_4h[sym] = seg4h

        # Run baseline on evaluation period
        d_trades = r38_collect_trades(CFG_V13, eval_15m, eval_1h, eval_4h, "D_mom8")
        s_trades = r38_collect_trades(CFG_V13, eval_15m, eval_1h, eval_4h, "S_d8")
        all_trades = d_trades + s_trades

        # Per-symbol performance
        sym_pnl: dict[str, list[float]] = {}
        for sym, entry, exit_b, pnl_val in all_trades:
            sym_pnl.setdefault(sym, []).append(pnl_val)

        exclude: set[str] = set()
        for sym, pnls in sym_pnl.items():
            if len(pnls) < min_trades:
                continue
            arr = np.array(pnls)
            pf = pf_from_pnl(arr)
            if pf < min_pf:
                exclude.add(sym)

        exclusions[year] = exclude

    return exclusions


def collect_trades_wf_excl(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
    variant: str,
    variant_params: dict,
    wf_exclusions: dict[int, set[str]],
) -> list[tuple[str, int, int, float]]:
    """Collect trades with walk-forward symbol exclusion.

    Each bar's exclusion set is determined by which year it falls in.
    """
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")
    bars_per_year = 365 * 24 * 4

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
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

            sl = cfg["s_sl"] if is_straddle else cfg["d_sl"]
            mh = mh_s if is_straddle else mh_d

            if is_straddle:
                sig = gen_vr_dur_signal(vr, cfg["s_vr"], cfg["s_dur"], warmup)
                p, e, x = sim_straddle_trades(c, h, lo, sig, sl, cfg["tp"], mh, COST_STRADDLE)
            else:
                sig = gen_vr_signal(vr, cfg["d_vr"], warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                if variant == "trailing":
                    p, e, x = sim_directional_trailing(
                        c, h, lo, sig, dire, sl, cfg["tp"], mh, COST_DIRECTIONAL,
                        variant_params.get("trail_activate", 0.02),
                        variant_params.get("trail_distance", 0.005),
                    )
                else:
                    p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, cfg["tp"], mh, COST_DIRECTIONAL)

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                # Check if symbol is excluded in this bar's year
                yr_idx = entry_15m // bars_per_year
                year = 2020 + yr_idx
                excl = wf_exclusions.get(year, set())
                if sym not in excl:
                    sym_trades.append((entry_15m, exit_15m, float(p[k])))

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
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*100}")
    print(f"  R43b — Validation & Fine-Tuning of R43 C6 (DTrail + SymExcl)")
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
    bl_sym = symbol_performance(bl_accepted)
    print(f"  Baseline: N={bl_m['N']}  PF={bl_m['PF']}  EV={bl_m['EV_bp']:+.1f}bp  Calmar={bl_m['calmar']}")

    # R43 C6 reference (full-sample)
    EXCL_BOTTOM10 = {'ALGOUSDT', 'ATOMUSDT', 'BCHUSDT', 'COMPUSDT', 'ENAUSDT',
                     'HBARUSDT', 'SNXUSDT', 'STXUSDT', 'SUIUSDT', 'XLMUSDT'}
    TRAIL_A2_D05 = {"trail_activate": 0.02, "trail_distance": 0.005}

    c6_d = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                   "trailing", TRAIL_A2_D05, exclude_symbols=EXCL_BOTTOM10)
    c6_s = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                   "baseline", {}, exclude_symbols=EXCL_BOTTOM10)
    c6_trades = c6_d + c6_s
    c6_pnl, c6_accepted = portfolio_sim_with_details(c6_trades, SLOTS)
    c6_m = compute_metrics(c6_pnl, data_years)
    c6_yr = year_metrics(c6_pnl, c6_accepted, year_bounds)
    print(f"  C6 Ref:   N={c6_m['N']}  PF={c6_m['PF']}  EV={c6_m['EV_bp']:+.1f}bp  Calmar={c6_m['calmar']}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: IS/OOS Validation of C6
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 1: IS/OOS Temporal Split (67%/33%)")
    print(f"{'─'*100}")

    # Split data
    is_15m, oos_15m = split_data_temporal(data_15m)
    is_1h, oos_1h = split_data_temporal(data_1h)
    is_4h, oos_4h = split_data_temporal(data_4h)
    make_split_vr([is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h], [(8, 56)])

    is_years = data_years * 0.67
    oos_years = data_years * 0.33

    # IS baseline
    is_bl_d = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "D_mom8")
    is_bl_s = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "S_d8")
    is_bl_pnl = portfolio_sim(is_bl_d + is_bl_s, SLOTS)
    is_bl_m = compute_metrics(is_bl_pnl, is_years)

    # OOS baseline
    oos_bl_d = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8")
    oos_bl_s = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")
    oos_bl_pnl = portfolio_sim(oos_bl_d + oos_bl_s, SLOTS)
    oos_bl_m = compute_metrics(oos_bl_pnl, oos_years)

    print(f"  Baseline IS:  N={is_bl_m['N']:>5}  PF={is_bl_m['PF']:.3f}  EV={is_bl_m['EV_bp']:+.1f}bp")
    print(f"  Baseline OOS: N={oos_bl_m['N']:>5}  PF={oos_bl_m['PF']:.3f}  EV={oos_bl_m['EV_bp']:+.1f}bp")

    # IS/OOS trailing
    is_trail_d = collect_trades_variant(CFG_V13, is_15m, is_1h, is_4h, "D_mom8",
                                         "trailing", TRAIL_A2_D05)
    is_trail_pnl = portfolio_sim(is_trail_d + is_bl_s, SLOTS)
    is_trail_m = compute_metrics(is_trail_pnl, is_years)

    oos_trail_d = collect_trades_variant(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8",
                                          "trailing", TRAIL_A2_D05)
    oos_trail_s = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")
    oos_trail_pnl = portfolio_sim(oos_trail_d + oos_trail_s, SLOTS)
    oos_trail_m = compute_metrics(oos_trail_pnl, oos_years)

    print(f"\n  Trail IS:     N={is_trail_m['N']:>5}  PF={is_trail_m['PF']:.3f}  EV={is_trail_m['EV_bp']:+.1f}bp  "
          f"Calmar={is_trail_m['calmar']:.1f}")
    print(f"  Trail OOS:    N={oos_trail_m['N']:>5}  PF={oos_trail_m['PF']:.3f}  EV={oos_trail_m['EV_bp']:+.1f}bp  "
          f"Calmar={oos_trail_m['calmar']:.1f}")

    # IS/OOS symbol exclusion (using IS-derived exclusion on OOS)
    # First find losing symbols from IS
    is_all_trades = is_bl_d + is_bl_s
    is_sym = symbol_performance([(s, e, x, p) for s, e, x, p in
                                  portfolio_sim_with_details(is_all_trades, SLOTS)[1]])
    is_excl = {sym for sym, s in is_sym.items() if s["PF"] < 1.0}
    print(f"\n  IS-derived exclusion: {len(is_excl)} symbols — {sorted(is_excl)}")

    # Apply IS-derived exclusion to OOS
    oos_excl_d = collect_trades_variant(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8",
                                         "trailing", TRAIL_A2_D05, exclude_symbols=is_excl)
    oos_excl_s = collect_trades_variant(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8",
                                         "baseline", {}, exclude_symbols=is_excl)
    oos_excl_pnl, oos_excl_acc = portfolio_sim_with_details(oos_excl_d + oos_excl_s, SLOTS)
    oos_excl_m = compute_metrics(oos_excl_pnl, oos_years)

    delta_pf = oos_excl_m["PF"] - oos_bl_m["PF"]
    delta_ev = oos_excl_m["EV_bp"] - oos_bl_m["EV_bp"]
    print(f"\n  C6 OOS (IS-derived excl): N={oos_excl_m['N']:>5}  PF={oos_excl_m['PF']:.3f} ({delta_pf:+.3f})  "
          f"EV={oos_excl_m['EV_bp']:+.1f}bp ({delta_ev:+.1f})  Calmar={oos_excl_m['calmar']:.1f}")

    if oos_excl_m["PF"] > oos_bl_m["PF"]:
        print(f"  ✓ C6 is ROBUST in OOS — PF improves from {oos_bl_m['PF']:.3f} to {oos_excl_m['PF']:.3f}")
    else:
        print(f"  ⚠ C6 degrades in OOS — caution required")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: Walk-Forward Symbol Filter
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 2: Walk-Forward Symbol Filter (no lookahead)")
    print(f"{'─'*100}")

    wf_configs = [
        ("WF_2yr_PF1", 2.0, 10, 1.0),
        ("WF_2yr_PF09", 2.0, 10, 0.9),
        ("WF_3yr_PF1", 3.0, 10, 1.0),
        ("WF_1yr_PF1", 1.0, 5, 1.0),
    ]

    best_wf = None
    best_wf_calmar = bl_m["calmar"]

    for name, lb_years, min_trades, min_pf in wf_configs:
        t1 = time.time()
        wf_excl = walk_forward_symbol_filter(data_15m, data_1h, data_4h, lb_years, min_trades, min_pf)

        # Print exclusion stats
        excl_counts = {yr: len(s) for yr, s in wf_excl.items()}
        print(f"\n  {name}: excl counts = {excl_counts}")

        # Collect with WF exclusion + trailing
        d_trades = collect_trades_wf_excl(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                          "trailing", TRAIL_A2_D05, wf_excl)
        s_trades = collect_trades_wf_excl(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                          "baseline", {}, wf_excl)
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        elapsed = time.time() - t1

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        verdict = "✓" if m["calmar"] > bl_m["calmar"] else "—"
        print(f"  {verdict} {name:<25} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})  [{elapsed:.1f}s]")

        yr_str = "    Yearly ΔEV: "
        for y in sorted(yr.keys()):
            delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
            yr_str += f"{y}: {delta:>+5.1f}  "
        print(yr_str)

        if m["calmar"] > best_wf_calmar:
            best_wf_calmar = m["calmar"]
            best_wf = (name, m, yr, wf_excl)

    if best_wf:
        print(f"\n  ★ Best WF: {best_wf[0]} — Calmar {best_wf[1]['calmar']:.1f} (baseline {bl_m['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 3: Fine-tune Trailing Stop (grid around A2_D05)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 3: Fine-Tune Trailing Stop (with best symbol exclusion)")
    print(f"{'─'*100}")

    # Use static exclusion for this phase (faster)
    excl = EXCL_BOTTOM10

    trail_fine = [
        ("T_A15_D03", {"trail_activate": 0.015, "trail_distance": 0.003}),
        ("T_A15_D05", {"trail_activate": 0.015, "trail_distance": 0.005}),
        ("T_A20_D03", {"trail_activate": 0.020, "trail_distance": 0.003}),
        ("T_A20_D05", {"trail_activate": 0.020, "trail_distance": 0.005}),  # R43 winner
        ("T_A20_D08", {"trail_activate": 0.020, "trail_distance": 0.008}),
        ("T_A25_D03", {"trail_activate": 0.025, "trail_distance": 0.003}),
        ("T_A25_D05", {"trail_activate": 0.025, "trail_distance": 0.005}),
        ("T_A25_D08", {"trail_activate": 0.025, "trail_distance": 0.008}),
        ("T_A30_D05", {"trail_activate": 0.030, "trail_distance": 0.005}),
        ("T_A30_D08", {"trail_activate": 0.030, "trail_distance": 0.008}),
        ("T_A30_D10", {"trail_activate": 0.030, "trail_distance": 0.010}),
        ("T_A35_D05", {"trail_activate": 0.035, "trail_distance": 0.005}),
        ("T_A35_D10", {"trail_activate": 0.035, "trail_distance": 0.010}),
    ]

    best_fine = None
    best_fine_calmar = 0

    for name, params in trail_fine:
        d_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                           "trailing", params, exclude_symbols=excl)
        s_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                           "baseline", {}, exclude_symbols=excl)
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        verdict = "✓" if m["calmar"] > c6_m["calmar"] else "—"
        print(f"  {verdict} {name:<20} N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})")

        if m["calmar"] > best_fine_calmar:
            best_fine_calmar = m["calmar"]
            best_fine = (name, params, m)

    if best_fine:
        print(f"\n  ★ Best Fine-tuned Trail: {best_fine[0]} — Calmar {best_fine[2]['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 4: 3-Stage Exit (SL → BE → Trail)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 4: 3-Stage Exit (SL → Breakeven → Trail)")
    print(f"{'─'*100}")

    stage3_configs = [
        # (name, trail_activate, trail_distance, be_activate)
        ("3S_BE1_TA3_TD05", 0.03, 0.005, 0.01),
        ("3S_BE15_TA3_TD05", 0.03, 0.005, 0.015),
        ("3S_BE1_TA2_TD05", 0.02, 0.005, 0.01),
        ("3S_BE15_TA25_TD05", 0.025, 0.005, 0.015),
        ("3S_BE1_TA25_TD03", 0.025, 0.003, 0.01),
    ]

    best_3s = None
    best_3s_calmar = c6_m["calmar"]

    for name, ta, td, be in stage3_configs:
        d_trades = collect_directional_trail3_fn(
            CFG_V13, data_15m, data_1h, data_4h,
            ta, td, be, exclude_symbols=excl,
        )
        s_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                           "baseline", {}, exclude_symbols=excl)
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_calmar = m["calmar"] - bl_m["calmar"]
        verdict = "✓" if m["calmar"] > c6_m["calmar"] else "—"
        print(f"  {verdict} {name:<25} N={m['N']:>5}  PF={m['PF']:>5.3f}  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})")

        if m["calmar"] > best_3s_calmar:
            best_3s_calmar = m["calmar"]
            best_3s = (name, m)

    if best_3s:
        print(f"\n  ★ Best 3-Stage: {best_3s[0]} — Calmar {best_3s[1]['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 5: VR Depth Scoring
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 5: VR Depth Scoring (deeper squeeze = better signal)")
    print(f"{'─'*100}")

    depth_configs = [
        # (name, lookback_bars, max_pctile)
        ("Depth_200_P30", 200, 0.30),
        ("Depth_200_P50", 200, 0.50),
        ("Depth_100_P30", 100, 0.30),
        ("Depth_100_P50", 100, 0.50),
        ("Depth_200_P20", 200, 0.20),
    ]

    best_depth = None
    best_depth_calmar = bl_m["calmar"]

    for name, lb, pctile in depth_configs:
        # Depth D + baseline S + exclusion + trailing
        d_trades = collect_directional_depth(
            CFG_V13, data_15m, data_1h, data_4h,
            lb, pctile, exclude_symbols=excl,
            trail_activate=0.02, trail_distance=0.005,
        )
        s_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                           "baseline", {}, exclude_symbols=excl)
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)

        delta_calmar = m["calmar"] - bl_m["calmar"]
        verdict = "✓" if m["calmar"] > c6_m["calmar"] else "—"
        print(f"  {verdict} {name:<25} N={m['N']:>5}  PF={m['PF']:>5.3f}  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})")

        if m["calmar"] > best_depth_calmar:
            best_depth_calmar = m["calmar"]
            best_depth = (name, m, yr)

    if best_depth:
        print(f"\n  ★ Best Depth: {best_depth[0]} — Calmar {best_depth[1]['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 6: Directional SL Tweak (tighter/wider with trail)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 6: SL Tweak + Trail (directional only)")
    print(f"{'─'*100}")

    sl_configs = [
        ("SL_0.5%_Trail", 0.005),
        ("SL_0.7%_Trail", 0.007),
        ("SL_0.8%_Trail", 0.008),
        ("SL_1.0%_Trail", 0.010),   # baseline
        ("SL_1.2%_Trail", 0.012),
        ("SL_1.5%_Trail", 0.015),
    ]

    best_sl = None
    best_sl_calmar = 0

    for name, sl_val in sl_configs:
        d_trades = collect_directional_trail3_fn(
            CFG_V13, data_15m, data_1h, data_4h,
            0.02, 0.005, 0.0,  # trail only, no BE
            sl_override=sl_val, exclude_symbols=excl,
        )
        s_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                           "baseline", {}, exclude_symbols=excl)
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = " ←" if sl_val == 0.01 else ""
        print(f"  {name:<20} N={m['N']:>5}  PF={m['PF']:>5.3f}  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f}){marker}")

        if m["calmar"] > best_sl_calmar:
            best_sl_calmar = m["calmar"]
            best_sl = (name, sl_val, m)

    if best_sl:
        print(f"\n  ★ Best SL: {best_sl[0]} — Calmar {best_sl[2]['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 7: Final Best Combination
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 7: Comprehensive Final Comparison")
    print(f"{'─'*100}")

    final_configs = []

    # F1: C6 original (trail + excl10)
    final_configs.append(("F1_C6_Original", "trailing", TRAIL_A2_D05, EXCL_BOTTOM10, None))

    # F2: Best fine-tuned trail + excl10
    if best_fine:
        final_configs.append(("F2_FineTuned", "trailing", best_fine[1], EXCL_BOTTOM10, None))

    # F3: WF-based exclusion + trail
    if best_wf:
        final_configs.append(("F3_WF_Excl", "trailing", TRAIL_A2_D05, None, best_wf[3]))

    # F4: Best SL + trail + excl10
    if best_sl and best_sl[1] != 0.01:
        final_configs.append(("F4_BestSL", "trailing", TRAIL_A2_D05, EXCL_BOTTOM10, None))

    best_final = None
    best_final_calmar = 0

    for name, d_var, d_params, static_excl, wf_excl in final_configs:
        if wf_excl is not None:
            d_trades = collect_trades_wf_excl(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                              d_var, d_params, wf_excl)
            s_trades = collect_trades_wf_excl(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                              "baseline", {}, wf_excl)
        else:
            d_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "D_mom8",
                                               d_var, d_params, exclude_symbols=static_excl)
            s_trades = collect_trades_variant(CFG_V13, data_15m, data_1h, data_4h, "S_d8",
                                               "baseline", {}, exclude_symbols=static_excl)
        trades = d_trades + s_trades
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        print(f"\n  {name}:")
        print(f"    N={m['N']:>5}  PF={m['PF']:.3f} ({delta_pf:+.3f})  EV={m['EV_bp']:+.1f}bp  "
              f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.1f} ({delta_calmar:+.1f})")
        print(f"    AnnRet={m['ann_ret_pct']:+.1f}%  DD={m['max_dd_pct']:.1f}%")
        yr_str = "    Yearly: "
        for y in sorted(yr.keys()):
            yy = yr[y]
            delta = yy["EV_bp"] - bl_yr[y]["EV_bp"]
            yr_str += f"{y}: PF={yy['PF']:.2f} EV={yy['EV_bp']:>+5.1f}({delta:>+.1f})  "
        print(yr_str)

        if m["calmar"] > best_final_calmar:
            best_final_calmar = m["calmar"]
            best_final = (name, m, yr)

    # ══════════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R43b FINAL SUMMARY — {total_time:.0f}s")
    print(f"{'='*100}")

    print(f"\n  {'Config':<30} {'N':>5} {'PF':>6} {'EV(bp)':>8} {'WR%':>5} "
          f"{'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"  {'─'*80}")
    for label, m in [("Baseline (v13.0.1)", bl_m), ("C6 (R43 winner)", c6_m)]:
        print(f"  {label:<30} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} {m['WR']:>5.1f} "
              f"{m['ann_ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")
    if best_final:
        m = best_final[1]
        print(f"  {'BEST (R43b)':<30} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} {m['WR']:>5.1f} "
              f"{m['ann_ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")

    # IS/OOS summary
    print(f"\n  IS/OOS Validation:")
    print(f"    Trail IS PF:  {is_trail_m['PF']:.3f}  (baseline {is_bl_m['PF']:.3f})")
    print(f"    Trail OOS PF: {oos_trail_m['PF']:.3f}  (baseline {oos_bl_m['PF']:.3f})")
    print(f"    Excl OOS PF:  {oos_excl_m['PF']:.3f}  (baseline {oos_bl_m['PF']:.3f})")

    is_oos_robust = (oos_trail_m["PF"] > oos_bl_m["PF"] and
                     oos_excl_m["PF"] > oos_bl_m["PF"])

    if is_oos_robust:
        print(f"\n  ✅ VALIDATED: Both trail and exclusion improve PF in OOS")
    else:
        print(f"\n  ⚠️  CAUTION: OOS validation mixed")

    print(f"\n  Commit: {commit}  |  Runtime: {total_time:.0f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
