#!/usr/bin/env python3
"""
R31 Final Analysis: G-ALL Deduped — Trade Metrics & Returns
=============================================================
Calculates WR, R:R, N, cumulative returns for the strongest variant
(G-ALL with per-symbol dedup), compared against A-15m baseline.

Includes E-15m+4h as conservative alternative.
"""

import json
import subprocess
import sys
import time
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
    bootstrap_pf,
    build_resampled,
    full_stats,
    gen_vr_dur_signal,
    gen_vr_signal,
    load_15m,
    make_direction_momentum,
    pf_from_pnl,
    precompute,
)
from research.active.r31_r40.r31_overlap_audit import (
    sim_directional_trades,
    sim_straddle_trades,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def dedup_trades_for_variant(
    data_15m: dict, data_1h: dict, data_4h: dict,
    tf_list: list, strat: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Run strategy with per-symbol dedup. Returns (deduped_pnl, raw_pnl, n_raw, n_deduped)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    raw_pnls: list[float] = []
    deduped_pnls: list[float] = []

    for sym in sorted(data_15m.keys()):
        all_sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in tf_list:
            d_tf = tf_data_map[ratio]
            if sym not in d_tf:
                continue
            d = d_tf[sym]

            if strat == "D_mom8":
                sig = gen_vr_signal(d["vr"], 0.45, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                pnl, ent, ext = sim_directional_trades(
                    d["close"], d["high"], d["low"], sig, dire,
                    0.01, 0.05, mh_d, COST_DIRECTIONAL)
            else:
                sig = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)
                pnl, ent, ext = sim_straddle_trades(
                    d["close"], d["high"], d["low"], sig,
                    0.015, 0.05, mh_s, COST_STRADDLE)

            for k in range(len(pnl)):
                entry_15m = int(ent[k]) * ratio
                exit_15m = int(ext[k]) * ratio + (ratio - 1)
                all_sym_trades.append((entry_15m, exit_15m, float(pnl[k])))
                raw_pnls.append(float(pnl[k]))

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

    return (np.array(deduped_pnls), np.array(raw_pnls),
            len(raw_pnls), len(deduped_pnls))


def compute_metrics(pnl: np.ndarray, label: str, data_years: float) -> dict:
    """Compute WR, RR, N, returns, PF, EV, t-stat, bootstrap."""
    n = len(pnl)
    if n == 0:
        return {}

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    flat = pnl[pnl == 0]

    wr = len(wins) / n * 100
    avg_win = float(np.mean(wins)) * 100 if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) * 100 if len(losses) > 0 else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    pf = pf_from_pnl(pnl)
    ev = float(np.mean(pnl))
    ev_bp = ev * 10000
    std = float(np.std(pnl))
    se = std / np.sqrt(n)
    t = ev / se if se > 0 else 0.0

    # Cumulative return (sequential, compounding)
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
    ann_ret = ((equity ** (1.0 / data_years)) - 1.0) * 100 if data_years > 0 else 0.0
    trades_per_day = n / (data_years * 365) if data_years > 0 else 0.0

    bs = bootstrap_pf(pnl, 2000)

    # Median win/loss
    med_win = float(np.median(wins)) * 100 if len(wins) > 0 else 0.0
    med_loss = float(np.median(losses)) * 100 if len(losses) > 0 else 0.0
    max_win = float(np.max(wins)) * 100 if len(wins) > 0 else 0.0
    max_loss = float(np.min(losses)) * 100 if len(losses) > 0 else 0.0

    # Win/loss streak
    streak_w = 0
    streak_l = 0
    max_streak_w = 0
    max_streak_l = 0
    for p in pnl:
        if p > 0:
            streak_w += 1
            streak_l = 0
        elif p < 0:
            streak_l += 1
            streak_w = 0
        else:
            streak_w = 0
            streak_l = 0
        max_streak_w = max(max_streak_w, streak_w)
        max_streak_l = max(max_streak_l, streak_l)

    return {
        "label": label,
        "N": n,
        "wins": len(wins),
        "losses": len(losses),
        "flat": len(flat),
        "WR": round(wr, 1),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "med_win_pct": round(med_win, 3),
        "med_loss_pct": round(med_loss, 3),
        "max_win_pct": round(max_win, 2),
        "max_loss_pct": round(max_loss, 2),
        "RR": round(rr, 3),
        "PF": round(pf, 3),
        "EV_bp": round(ev_bp, 1),
        "t_stat": round(t, 2),
        "total_ret_pct": round(total_ret, 2),
        "ann_ret_pct": round(ann_ret, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "trades_per_day": round(trades_per_day, 2),
        "max_win_streak": max_streak_w,
        "max_loss_streak": max_streak_l,
        "bootstrap_P_PF_ge_1": bs["P_PF_ge_1"],
        "bootstrap_CI_lo": bs["CI_lo"],
        "bootstrap_CI_hi": bs["CI_hi"],
    }


def per_year_dedup(
    data_15m: dict, data_1h: dict, data_4h: dict,
    tf_list: list, strat: str, data_years: float,
) -> list[dict]:
    """Per-year deduped metrics."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    max_years = 7
    bpy_15m = 365 * 24 * 4

    year_pnls: list[list[float]] = [[] for _ in range(max_years)]

    for sym in sorted(data_15m.keys()):
        n_15m = len(data_15m[sym]["close"])
        n_years_sym = n_15m // bpy_15m + 1

        for yr in range(min(n_years_sym, max_years)):
            yr_start_15m = yr * bpy_15m
            yr_end_15m = min((yr + 1) * bpy_15m, n_15m)
            if yr_end_15m - yr_start_15m < 500:
                continue

            all_sym_trades: list[tuple[int, int, float]] = []

            for tf_label, ratio, warmup, mh_d, mh_s, min_dur in tf_list:
                d_tf = tf_data_map[ratio]
                if sym not in d_tf:
                    continue
                d_full = d_tf[sym]
                n_tf = len(d_full["close"])

                # Convert 15m year boundaries to this TF's bar indices
                tf_start = yr_start_15m // ratio
                tf_end = min(yr_end_15m // ratio, n_tf)
                if tf_end - tf_start < max(100, warmup + 50):
                    continue

                # Slice data for this year
                d_yr: dict = {}
                for key in ("close", "high", "low", "volume"):
                    d_yr[key] = d_full[key][tf_start:tf_end].copy()
                from research.active.r31_r40.r31_mtf_vol_straddle import (
                    compute_momentum,
                    compute_squeeze_dur,
                    compute_vr,
                    vol_percentile,
                )
                c, h, lo, v = d_yr["close"], d_yr["high"], d_yr["low"], d_yr["volume"]
                d_yr["vr"] = compute_vr(c, h, lo, 7, 50)
                d_yr["mom_8"] = compute_momentum(c, 8)
                d_yr["vol_pct"] = vol_percentile(v, min(200, len(v) // 2))
                d_yr["sq_dur_055"] = compute_squeeze_dur(d_yr["vr"], 0.55)

                if strat == "D_mom8":
                    sig = gen_vr_signal(d_yr["vr"], 0.45, warmup)
                    dire = make_direction_momentum(sig, d_yr["mom_8"])
                    pnl, ent, ext = sim_directional_trades(
                        d_yr["close"], d_yr["high"], d_yr["low"], sig, dire,
                        0.01, 0.05, mh_d, COST_DIRECTIONAL)
                else:
                    sig = gen_vr_dur_signal(d_yr["vr"], 0.55, min_dur, warmup)
                    pnl, ent, ext = sim_straddle_trades(
                        d_yr["close"], d_yr["high"], d_yr["low"], sig,
                        0.015, 0.05, mh_s, COST_STRADDLE)

                for k in range(len(pnl)):
                    entry_15m = tf_start * ratio + int(ent[k]) * ratio
                    exit_15m = tf_start * ratio + int(ext[k]) * ratio + (ratio - 1)
                    all_sym_trades.append((entry_15m, exit_15m, float(pnl[k])))

            all_sym_trades.sort(key=lambda x: x[0])
            accepted: list[tuple[int, int]] = []
            for entry_15m, exit_15m, pnl_val in all_sym_trades:
                conflict = False
                for ae, ax in accepted:
                    if entry_15m <= ax and exit_15m >= ae:
                        conflict = True
                        break
                if not conflict:
                    year_pnls[yr].append(pnl_val)
                    accepted.append((entry_15m, exit_15m))

    results = []
    for yr in range(max_years):
        if not year_pnls[yr]:
            continue
        arr = np.array(year_pnls[yr])
        s = compute_metrics(arr, f"{2020 + yr}", 1.0)
        s["year"] = 2020 + yr
        results.append(s)
    return results


def print_metrics(m: dict) -> None:
    """Pretty print metrics."""
    print(f"\n    {'─' * 60}")
    print(f"    {m['label']}")
    print(f"    {'─' * 60}")
    print(f"    Trades:          {m['N']:>6}  ({m['trades_per_day']:.2f}/day)")
    print(f"    Wins/Losses/Flat:{m['wins']:>5} / {m['losses']:>4} / {m['flat']}")
    print(f"    Win Rate:        {m['WR']:>6.1f}%")
    print(f"    ")
    print(f"    Avg Win:         {m['avg_win_pct']:>+7.3f}%")
    print(f"    Avg Loss:        {m['avg_loss_pct']:>+7.3f}%")
    print(f"    Med Win:         {m['med_win_pct']:>+7.3f}%")
    print(f"    Med Loss:        {m['med_loss_pct']:>+7.3f}%")
    print(f"    Max Win:         {m['max_win_pct']:>+7.2f}%")
    print(f"    Max Loss:        {m['max_loss_pct']:>+7.2f}%")
    print(f"    R:R (avg):       {m['RR']:>7.3f}")
    print(f"    ")
    print(f"    Profit Factor:   {m['PF']:>7.3f}")
    print(f"    EV/trade:        {m['EV_bp']:>+7.1f} bp")
    print(f"    t-statistic:     {m['t_stat']:>7.2f}")
    print(f"    ")
    print(f"    Total Return:    {m['total_ret_pct']:>+8.2f}% (compounded)")
    print(f"    Annual Return:   {m['ann_ret_pct']:>+8.2f}%")
    print(f"    Max Drawdown:    {m['max_dd_pct']:>8.2f}%")
    print(f"    ")
    print(f"    Max Win Streak:  {m['max_win_streak']:>6}")
    print(f"    Max Loss Streak: {m['max_loss_streak']:>6}")
    print(f"    Bootstrap P(PF≥1): {m['bootstrap_P_PF_ge_1']:.0f}%  "
          f"CI=[{m['bootstrap_CI_lo']:.3f}, {m['bootstrap_CI_hi']:.3f}]")


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 80)
    print("  R31 FINAL ANALYSIS: TRADE METRICS & RETURNS")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # Load data
    print("\n  Loading data...")
    data_15m = load_15m()
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)

    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    n_syms = len(data_15m)
    total_bars = sum(d["bars"] for d in data_15m.values())
    print(f"  {n_syms} symbols, {total_bars:,} bars, {data_years:.1f} years")

    # ── Variant configs ──
    variants = {
        "A-15m (baseline)": [("15m", 1, WARMUP_15m, 192, 384, 8)],
        "E-15m+4h (conservative)": [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ],
        "G-ALL deduped (strongest)": [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("1h", 4, WARMUP_1h, 48, 96, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ],
    }

    all_metrics: dict = {}

    for var_name, tf_list in variants.items():
        print(f"\n\n  {'═' * 74}")
        print(f"  {var_name}")
        print(f"  {'═' * 74}")

        for strat in ["D_mom8", "S_d8"]:
            ded_pnl, raw_pnl, n_raw, n_ded = dedup_trades_for_variant(
                data_15m, data_1h, data_4h, tf_list, strat)
            label = f"{var_name} — {strat}"
            m = compute_metrics(ded_pnl, label, data_years)
            print_metrics(m)
            all_metrics[f"{var_name}_{strat}"] = m

        # Combined D+S metrics
        ded_d, _, _, _ = dedup_trades_for_variant(
            data_15m, data_1h, data_4h, tf_list, "D_mom8")
        ded_s, _, _, _ = dedup_trades_for_variant(
            data_15m, data_1h, data_4h, tf_list, "S_d8")
        combined = np.concatenate([ded_d, ded_s])
        label = f"{var_name} — COMBINED (D+S)"
        m_comb = compute_metrics(combined, label, data_years)
        print_metrics(m_comb)
        all_metrics[f"{var_name}_COMBINED"] = m_comb

    # ── Per-Year for G-ALL deduped ──
    print(f"\n\n  {'═' * 74}")
    print(f"  G-ALL DEDUPED — PER-YEAR BREAKDOWN")
    print(f"  {'═' * 74}")

    g_tf = variants["G-ALL deduped (strongest)"]
    for strat in ["D_mom8", "S_d8"]:
        print(f"\n  {strat}:")
        yrs = per_year_dedup(data_15m, data_1h, data_4h, g_tf, strat, data_years)
        n_pos = sum(1 for y in yrs if y["PF"] >= 1.0)
        print(f"  {n_pos}/{len(yrs)} profitable years")
        print(f"  {'Year':<6} {'N':>5} {'WR':>6} {'R:R':>6} {'PF':>6} {'EV(bp)':>8} {'Ret%':>8} {'DD%':>6}")
        print(f"  {'─' * 55}")
        for y in yrs:
            mark = "✅" if y["PF"] >= 1.0 else "❌"
            print(f"  {y['year']:<6} {y['N']:>5} {y['WR']:>5.1f}% {y['RR']:>6.2f} "
                  f"{y['PF']:>6.3f} {y['EV_bp']:>+7.1f} {y['total_ret_pct']:>+7.2f}% "
                  f"{y['max_dd_pct']:>5.1f}% {mark}")

    # ── Summary table ──
    print(f"\n\n  {'═' * 74}")
    print(f"  COMPARISON TABLE")
    print(f"  {'═' * 74}")
    print(f"\n  {'Strategy':<40} {'N':>5} {'WR':>6} {'R:R':>6} {'PF':>6} {'EV':>7} {'Ret%':>9} {'DD%':>6} {'t':>5} {'#/day':>5}")
    print(f"  {'─' * 100}")
    for key in sorted(all_metrics.keys()):
        m = all_metrics[key]
        print(f"  {m['label'][:40]:<40} {m['N']:>5} {m['WR']:>5.1f}% {m['RR']:>6.2f} "
              f"{m['PF']:>6.3f} {m['EV_bp']:>+6.1f} {m['total_ret_pct']:>+8.2f}% "
              f"{m['max_dd_pct']:>5.1f}% {m['t_stat']:>5.2f} {m['trades_per_day']:>5.2f}")

    elapsed = time.time() - t0
    print(f"\n  RUNTIME: {elapsed:.1f}s")

    # ── Save JSON ──
    out_dir = ROOT / "logs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "r31_final_metrics.json"
    save_data = {
        "commit": commit,
        "date": datetime.now(timezone.utc).isoformat(),
        "n_symbols": n_syms,
        "data_years": round(data_years, 1),
        "runtime_s": round(elapsed, 1),
        "metrics": all_metrics,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  💾 {out_path}")


if __name__ == "__main__":
    main()
