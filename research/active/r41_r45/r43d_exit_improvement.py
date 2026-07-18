"""R43d — Deep Optimization of Wide-TP + Trail (Best R43c Finding)

R43c Winner: WTP_TP15_TA20_D03
  - TP=15% (3x wider than baseline 5%), Trail_A=2%, Trail_D=0.3%
  - Calmar 5.8 → 9.4 (+62%), PF 1.280 → 1.351
  - OOS validated: PF +0.011, Calmar 14.7 → 17.8
  - ALL 6 years improved, no symbol exclusion

This script:
  Phase 1: Dense grid around TA20_D03 winner (activate 1.0-3.0%, dist 0.1-0.5%)
  Phase 2: TP sweep (5%-20%, NoTP) at best trail params
  Phase 3: Year-by-year deep analysis of top 5 configs
  Phase 4: Bootstrap confidence interval for best config
  Phase 5: Slot count sensitivity (8, 10, 12, 15)
  Phase 6: IS/OOS with multiple split ratios (50/50, 60/40, 70/30)
  Phase 7: Walk-forward 1yr blocks (2020→2021→2022→2023→2024→2025)

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/exit_improvement_r43d.py
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
    compute_vr, compute_momentum,
    gen_vr_signal, gen_vr_dur_signal,
    make_direction_momentum,
    load_15m, build_resampled, precompute, pf_from_pnl,
    bootstrap_pf,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    sim_straddle_trades, sim_directional_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF, CFG_V13,
)
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics, get_year_boundaries, symbol_performance,
)
from research.active.r41_r45.r43c_exit_improvement import (
    sim_directional_widetp_trail, collect_d_widetp_trail,
)
from research.active.r41_r45.r43b_exit_improvement import (
    split_data_temporal, make_split_vr,
)


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거, harness 단일소스


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*100}")
    print(f"  R43d — Deep Optimization: Wide-TP + Trail")
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

    # Baseline
    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_trades = bl_d + bl_s
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_trades, SLOTS)
    bl_m = compute_metrics(bl_pnl, data_years)
    bl_yr = year_metrics(bl_pnl, bl_accepted, year_bounds)
    print(f"  Baseline: N={bl_m['N']}  PF={bl_m['PF']}  EV={bl_m['EV_bp']:+.1f}bp  "
          f"Calmar={bl_m['calmar']}  DD={bl_m['max_dd_pct']:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: Dense Grid Search (activate × distance)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 1: Dense Grid — Trail Activate × Trail Distance (TP = no-cap)")
    print(f"{'─'*100}")

    # Use TP=1.0 (effectively no cap) since R43c showed TP15 ≈ NoTP
    TP_WIDE = 1.0

    activates = [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.035, 0.040]
    distances = [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.010]

    results: list[tuple[float, float, dict, dict]] = []
    best_calmar = 0.0
    best_config = None

    # Print header
    dist_header = "".join(f"{'D'+str(int(d*1000))+'‰':>8}" for d in distances)
    print(f"  {'Activate':>10} {dist_header}")
    print(f"  {'─'*10} {'─'*8 * len(distances)}")

    for ta in activates:
        row_str = f"  A{ta*100:.1f}%"
        for td in distances:
            if td >= ta:
                row_str += f"{'—':>8}"
                continue

            d_trades = collect_d_widetp_trail(
                CFG_V13, data_15m, data_1h, data_4h,
                TP_WIDE, ta, td,
            )
            trades = d_trades + bl_s
            pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
            m = compute_metrics(pnl, data_years)
            yr = year_metrics(pnl, accepted, year_bounds)
            results.append((ta, td, m, yr))

            calmar = m["calmar"]
            marker = "★" if calmar > best_calmar else ""
            row_str += f"{calmar:>6.1f}{marker:>2}"

            if calmar > best_calmar:
                best_calmar = calmar
                best_config = (ta, td, m, yr)

        print(row_str)

    if best_config:
        ta, td, m, yr = best_config
        print(f"\n  ★ Best: A={ta*100:.1f}% D={td*1000:.0f}‰ — "
              f"Calmar={m['calmar']} PF={m['PF']} EV={m['EV_bp']:+.1f}bp N={m['N']}")
        yr_str = "    ΔEV: "
        for y in sorted(yr.keys()):
            delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
            yr_str += f"{y}: {delta:>+.1f}  "
        print(yr_str)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: TP Sweep at Best Trail Params
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 2: TP Sweep at A={best_config[0]*100:.1f}% D={best_config[1]*1000:.0f}‰")
    print(f"{'─'*100}")

    ta_best, td_best = best_config[0], best_config[1]
    tp_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.00]

    best_tp = None
    best_tp_calmar = 0.0

    for tp_val in tp_values:
        d_trades = collect_d_widetp_trail(
            CFG_V13, data_15m, data_1h, data_4h,
            tp_val, ta_best, td_best,
        )
        trades = d_trades + bl_s
        pnl, accepted = portfolio_sim_with_details(trades, SLOTS)
        m = compute_metrics(pnl, data_years)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_calmar = m["calmar"] - bl_m["calmar"]
        marker = "★" if m["calmar"] > best_tp_calmar else ("←" if tp_val == 0.05 else "")
        print(f"  TP={tp_val*100:>5.0f}%  N={m['N']:>5}  PF={m['PF']:>5.3f} ({delta_pf:>+.3f})  "
              f"EV={m['EV_bp']:>+6.1f}bp  Calmar={m['calmar']:>5.1f} ({delta_calmar:>+.1f})  {marker}")

        if m["calmar"] > best_tp_calmar:
            best_tp_calmar = m["calmar"]
            best_tp = (tp_val, m)

    if best_tp:
        print(f"\n  ★ Best TP: {best_tp[0]*100:.0f}% — Calmar={best_tp[1]['calmar']}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 3: Top 5 Config Year-by-Year Deep Dive
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 3: Top 5 Configs — Year-by-Year Analysis")
    print(f"{'─'*100}")

    # Sort results by Calmar descending
    results_sorted = sorted(results, key=lambda r: r[2]["calmar"], reverse=True)

    # Baseline yearly
    print(f"\n  {'Config':<25} {'Full':>8}", end="")
    for y in sorted(bl_yr.keys()):
        print(f" {y:>8}", end="")
    print()
    print(f"  {'─'*25} {'─'*8}", end="")
    for _ in bl_yr:
        print(f" {'─'*8}", end="")
    print()

    # Print baseline
    print(f"  {'Baseline':<25} {bl_m['calmar']:>8.1f}", end="")
    for y in sorted(bl_yr.keys()):
        print(f" {bl_yr[y]['EV_bp']:>+8.1f}", end="")
    print()

    for i, (ta, td, m, yr) in enumerate(results_sorted[:5]):
        name = f"A{ta*100:.1f}_D{td*1000:.0f}"
        print(f"  {name:<25} {m['calmar']:>8.1f}", end="")
        all_better = True
        for y in sorted(yr.keys()):
            delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
            print(f" {delta:>+8.1f}", end="")
            if delta < -2.0:
                all_better = False
        status = " ✓" if all_better else " ⚠"
        print(status)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 4: Bootstrap Confidence Interval
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 4: Bootstrap CI (1000 samples)")
    print(f"{'─'*100}")

    ta_opt, td_opt = best_config[0], best_config[1]
    d_trades_opt = collect_d_widetp_trail(
        CFG_V13, data_15m, data_1h, data_4h,
        TP_WIDE, ta_opt, td_opt,
    )
    opt_trades = d_trades_opt + bl_s
    opt_pnl, opt_accepted = portfolio_sim_with_details(opt_trades, SLOTS)
    opt_m = compute_metrics(opt_pnl, data_years)

    pnl_array = opt_pnl

    # Bootstrap PF
    n_boot = 1000
    boot_pf = np.empty(n_boot)
    boot_ev = np.empty(n_boot)
    rng = np.random.default_rng(42)

    for b in range(n_boot):
        idx = rng.choice(len(pnl_array), size=len(pnl_array), replace=True)
        sample = pnl_array[idx]
        boot_pf[b] = pf_from_pnl(sample)
        boot_ev[b] = float(np.mean(sample)) * 10000  # bp

    pf_lo = np.percentile(boot_pf, 2.5)
    pf_hi = np.percentile(boot_pf, 97.5)
    ev_lo = np.percentile(boot_ev, 2.5)
    ev_hi = np.percentile(boot_ev, 97.5)

    print(f"  N = {len(pnl_array)} trades")
    print(f"  PF = {opt_m['PF']:.3f}  95% CI: [{pf_lo:.3f}, {pf_hi:.3f}]")
    print(f"  EV = {opt_m['EV_bp']:+.1f}bp  95% CI: [{ev_lo:+.1f}, {ev_hi:+.1f}]bp")

    # Baseline bootstrap
    bl_boot_pf = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(len(bl_pnl), size=len(bl_pnl), replace=True)
        bl_boot_pf[b] = pf_from_pnl(bl_pnl[idx])

    pf_diff = boot_pf - bl_boot_pf
    pct_better = np.mean(pf_diff > 0) * 100
    diff_lo = np.percentile(pf_diff, 2.5)
    diff_hi = np.percentile(pf_diff, 97.5)

    print(f"\n  PF Improvement vs Baseline:")
    print(f"    ΔPF = {opt_m['PF'] - bl_m['PF']:+.3f}  95% CI: [{diff_lo:+.3f}, {diff_hi:+.3f}]")
    print(f"    Better in {pct_better:.0f}% of bootstrap samples")
    if diff_lo > 0:
        print(f"    ✅ Statistically significant (entire CI above 0)")
    else:
        print(f"    ⚠️ CI includes 0 — improvement not statistically significant at 95%")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 5: Slot Sensitivity
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 5: Slot Count Sensitivity")
    print(f"{'─'*100}")

    slot_counts = [6, 8, 10, 12, 15]

    for sc in slot_counts:
        bl_pnl_sc = portfolio_sim(bl_trades, sc)
        bl_m_sc = compute_metrics(bl_pnl_sc, data_years)
        opt_pnl_sc = portfolio_sim(opt_trades, sc)
        opt_m_sc = compute_metrics(opt_pnl_sc, data_years)

        delta_calmar = opt_m_sc["calmar"] - bl_m_sc["calmar"]
        delta_pf = opt_m_sc["PF"] - bl_m_sc["PF"]
        marker = "←" if sc == 10 else ""
        print(f"  Slots={sc:>2}  BL: Calmar={bl_m_sc['calmar']:>5.1f} PF={bl_m_sc['PF']:.3f}  "
              f"| OPT: Calmar={opt_m_sc['calmar']:>5.1f} PF={opt_m_sc['PF']:.3f}  "
              f"| Δcalmar={delta_calmar:>+.1f}  ΔPF={delta_pf:>+.3f}  {marker}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 6: IS/OOS with Multiple Split Ratios
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 6: IS/OOS Robustness (multiple splits)")
    print(f"{'─'*100}")

    split_ratios = [0.50, 0.60, 0.67, 0.75]

    for sr in split_ratios:
        is_15m, oos_15m = split_data_temporal(data_15m, sr)
        is_1h, oos_1h = split_data_temporal(data_1h, sr)
        is_4h, oos_4h = split_data_temporal(data_4h, sr)
        make_split_vr([is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h], [(8, 56)])

        is_years = data_years * sr
        oos_years = data_years * (1.0 - sr)

        # IS/OOS baseline
        is_bl_all = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "D_mom8") + \
                    r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "S_d8")
        oos_bl_all = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8") + \
                     r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")

        is_bl_pnl_sr = portfolio_sim(is_bl_all, SLOTS)
        oos_bl_pnl_sr = portfolio_sim(oos_bl_all, SLOTS)
        is_bl_m_sr = compute_metrics(is_bl_pnl_sr, is_years)
        oos_bl_m_sr = compute_metrics(oos_bl_pnl_sr, oos_years)

        # IS/OOS optimal
        is_s = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "S_d8")
        oos_s = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")

        is_d = collect_d_widetp_trail(CFG_V13, is_15m, is_1h, is_4h,
                                       TP_WIDE, ta_opt, td_opt)
        oos_d = collect_d_widetp_trail(CFG_V13, oos_15m, oos_1h, oos_4h,
                                        TP_WIDE, ta_opt, td_opt)

        is_opt_pnl = portfolio_sim(is_d + is_s, SLOTS)
        oos_opt_pnl = portfolio_sim(oos_d + oos_s, SLOTS)
        is_opt_m = compute_metrics(is_opt_pnl, is_years)
        oos_opt_m = compute_metrics(oos_opt_pnl, oos_years)

        is_gain = is_opt_m["PF"] - is_bl_m_sr["PF"]
        oos_gain = oos_opt_m["PF"] - oos_bl_m_sr["PF"]
        robust = "✅" if oos_gain > 0 else "❌"

        print(f"  Split {sr:.0%}/{1-sr:.0%}  IS: ΔPF={is_gain:+.3f}  OOS: ΔPF={oos_gain:+.3f}  {robust}"
              f"  (OOS BL PF={oos_bl_m_sr['PF']:.3f}→{oos_opt_m['PF']:.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 7: Walk-Forward 1yr Blocks
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*100}")
    print(f"  Phase 7: Walk-Forward Annual Blocks")
    print(f"{'─'*100}")

    bars_per_year = 365 * 24 * 4
    max_bars = max(len(d["close"]) for d in data_15m.values())
    n_years_avail = int(max_bars / bars_per_year)

    print(f"\n  {'Year':<8} {'BL_PF':>8} {'OPT_PF':>8} {'ΔPF':>8} {'BL_EV':>8} "
          f"{'OPT_EV':>8} {'ΔEV':>8} {'Status':>8}")
    print(f"  {'─'*70}")

    wf_better = 0
    wf_total = 0

    for yr_idx in range(n_years_avail):
        start = yr_idx * bars_per_year
        end = min(start + bars_per_year, max_bars)
        year = 2020 + yr_idx

        # Create yearly slices
        yr_15m: dict = {}
        yr_1h: dict = {}
        yr_4h: dict = {}

        for sym in data_15m:
            n = len(data_15m[sym]["close"])
            if end > n:
                continue

            # 15m
            seg: dict = {}
            for key in data_15m[sym]:
                if isinstance(data_15m[sym][key], np.ndarray) and len(data_15m[sym][key]) == n:
                    seg[key] = data_15m[sym][key][start:end].copy()
                else:
                    seg[key] = data_15m[sym][key]
            seg["bars"] = end - start
            c, h, lo = seg["close"], seg["high"], seg["low"]
            seg["mom_8"] = compute_momentum(c, 8)
            seg["vr_8_56"] = compute_vr(c, h, lo, 8, 56)
            yr_15m[sym] = seg

            # 1h
            if sym in data_1h:
                n1h = len(data_1h[sym]["close"])
                ratio = n // n1h
                s1h, e1h = start // ratio, end // ratio
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
                yr_1h[sym] = seg1h

            # 4h
            if sym in data_4h:
                n4h = len(data_4h[sym]["close"])
                ratio4 = n // n4h
                s4h, e4h = start // ratio4, end // ratio4
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
                yr_4h[sym] = seg4h

        if not yr_15m:
            continue

        # Baseline on this year
        bl_d_yr = r38_collect_trades(CFG_V13, yr_15m, yr_1h, yr_4h, "D_mom8")
        bl_s_yr = r38_collect_trades(CFG_V13, yr_15m, yr_1h, yr_4h, "S_d8")
        bl_pnl_yr = portfolio_sim(bl_d_yr + bl_s_yr, SLOTS)
        bl_m_yr = compute_metrics(bl_pnl_yr, 1.0)

        # Optimal on this year
        opt_d_yr = collect_d_widetp_trail(CFG_V13, yr_15m, yr_1h, yr_4h,
                                           TP_WIDE, ta_opt, td_opt)
        opt_pnl_yr = portfolio_sim(opt_d_yr + bl_s_yr, SLOTS)
        opt_m_yr = compute_metrics(opt_pnl_yr, 1.0)

        delta_pf = opt_m_yr["PF"] - bl_m_yr["PF"]
        delta_ev = opt_m_yr["EV_bp"] - bl_m_yr["EV_bp"]
        status = "✓" if delta_pf > 0 else "✗"
        wf_total += 1
        if delta_pf > 0:
            wf_better += 1

        print(f"  {year:<8} {bl_m_yr['PF']:>8.3f} {opt_m_yr['PF']:>8.3f} {delta_pf:>+8.3f} "
              f"{bl_m_yr['EV_bp']:>+8.1f} {opt_m_yr['EV_bp']:>+8.1f} {delta_ev:>+8.1f} {status:>8}")

    print(f"\n  WF win rate: {wf_better}/{wf_total} years")

    # ══════════════════════════════════════════════════════════════════════════
    #  Final Summary
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R43d FINAL SUMMARY — {total_time:.0f}s")
    print(f"{'='*100}")

    ta, td = best_config[0], best_config[1]
    m = best_config[2]

    print(f"\n  OPTIMAL CONFIG: Trail Activate={ta*100:.1f}%, Trail Distance={td*1000:.0f}‰, TP=uncapped")
    print(f"\n  {'Metric':<20} {'Baseline':>12} {'Optimal':>12} {'Change':>12}")
    print(f"  {'─'*60}")
    print(f"  {'N (trades)':<20} {bl_m['N']:>12} {m['N']:>12} {m['N']-bl_m['N']:>+12}")
    print(f"  {'PF':<20} {bl_m['PF']:>12.3f} {m['PF']:>12.3f} {m['PF']-bl_m['PF']:>+12.3f}")
    print(f"  {'EV (bp)':<20} {bl_m['EV_bp']:>+12.1f} {m['EV_bp']:>+12.1f} {m['EV_bp']-bl_m['EV_bp']:>+12.1f}")
    print(f"  {'WR (%)':<20} {bl_m['WR']:>12.1f} {m['WR']:>12.1f} {m['WR']-bl_m['WR']:>+12.1f}")
    print(f"  {'AnnRet (%)':<20} {bl_m['ann_ret_pct']:>+12.1f} {m['ann_ret_pct']:>+12.1f} "
          f"{m['ann_ret_pct']-bl_m['ann_ret_pct']:>+12.1f}")
    print(f"  {'MaxDD (%)':<20} {bl_m['max_dd_pct']:>12.1f} {m['max_dd_pct']:>12.1f} "
          f"{m['max_dd_pct']-bl_m['max_dd_pct']:>+12.1f}")
    print(f"  {'Calmar':<20} {bl_m['calmar']:>12.1f} {m['calmar']:>12.1f} {m['calmar']-bl_m['calmar']:>+12.1f}")

    print(f"\n  Bootstrap PF 95% CI: [{pf_lo:.3f}, {pf_hi:.3f}]")
    print(f"  ΔPF 95% CI: [{diff_lo:+.3f}, {diff_hi:+.3f}]")
    print(f"  Better in {pct_better:.0f}% of bootstrap samples")
    print(f"  WF annual win rate: {wf_better}/{wf_total}")

    final_verdict = (
        diff_lo > 0 and
        wf_better >= wf_total - 1 and
        m["calmar"] > bl_m["calmar"] * 1.2
    )
    if final_verdict:
        print(f"\n  ✅ PRODUCTION-READY: Statistically significant, WF robust, >20% Calmar improvement")
    else:
        print(f"\n  🔬 PROMISING: Improvement found but additional validation recommended")

    print(f"\n  Commit: {commit}  |  Runtime: {total_time:.0f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
