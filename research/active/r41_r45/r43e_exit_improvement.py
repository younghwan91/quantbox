"""R43e — Production Config Validation

Quick script to compute exact metrics for the recommended production config:
  Trail activate=4.0%, Trail distance=0.3%, TP=12% (uncapped equivalent)
  vs
  Trail activate=3.0%, Trail distance=0.3%, TP=12%
  vs
  Trail activate=2.0%, Trail distance=0.3%, TP=12% (R43c winner)

Also runs bootstrap CI and IS/OOS for the production config.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    compute_vr, compute_momentum,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    CFG_V13,
)
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics, get_year_boundaries, symbol_performance,
)
from research.active.r41_r45.r43c_exit_improvement import (
    collect_d_widetp_trail,
)
from research.active.r41_r45.r43b_exit_improvement import (
    split_data_temporal, make_split_vr,
)


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*80}")
    print(f"  R43e — Production Config Validation  (commit: {commit})")
    print(f"{'='*80}\n")

    t0 = time.time()

    # Load
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])

    year_bounds = get_year_boundaries(data_15m)
    SLOTS = 10

    # Baseline
    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_pnl, bl_acc = portfolio_sim_with_details(bl_d + bl_s, SLOTS)
    bl_m = compute_metrics(bl_pnl, data_years)
    bl_yr = year_metrics(bl_pnl, bl_acc, year_bounds)

    # Production candidates
    candidates = [
        ("Baseline (v13.0.1)", None, None, None),
        ("P1: A2.0_D3_TP12%", 0.020, 0.003, 0.12),
        ("P2: A3.0_D3_TP12%", 0.030, 0.003, 0.12),
        ("P3: A4.0_D3_TP12%", 0.040, 0.003, 0.12),
        ("P4: A3.0_D5_TP12%", 0.030, 0.005, 0.12),
        ("P5: A4.0_D5_TP12%", 0.040, 0.005, 0.12),
        ("P6: A3.0_D3_TP∞",  0.030, 0.003, 1.00),
        ("P7: A4.0_D3_TP∞",  0.040, 0.003, 1.00),
    ]

    all_results: list[tuple[str, dict, dict]] = []

    print(f"  {'Config':<25} {'N':>5} {'PF':>6} {'ΔPF':>6} {'EV(bp)':>8} {'ΔEV':>6} "
          f"{'WR%':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7} {'ΔCal':>6}")
    print(f"  {'─'*97}")

    for name, ta, td, tp in candidates:
        if ta is None:
            pnl, acc = bl_pnl, bl_acc
            m = bl_m
            yr = bl_yr
        else:
            d_trades = collect_d_widetp_trail(
                CFG_V13, data_15m, data_1h, data_4h,
                tp, ta, td,
            )
            trades = d_trades + bl_s
            pnl, acc = portfolio_sim_with_details(trades, SLOTS)
            m = compute_metrics(pnl, data_years)
            yr = year_metrics(pnl, acc, year_bounds)

        delta_pf = m["PF"] - bl_m["PF"]
        delta_ev = m["EV_bp"] - bl_m["EV_bp"]
        delta_calmar = m["calmar"] - bl_m["calmar"]

        print(f"  {name:<25} {m['N']:>5} {m['PF']:>6.3f} {delta_pf:>+6.3f} "
              f"{m['EV_bp']:>+8.1f} {delta_ev:>+6.1f} {m['WR']:>5.1f} "
              f"{m['ann_ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f} {delta_calmar:>+6.1f}")

        all_results.append((name, m, yr))

    # Year-by-year for top candidates
    print(f"\n  Year-by-Year ΔEV (bp) vs Baseline:")
    print(f"  {'Config':<25}", end="")
    for y in sorted(bl_yr.keys()):
        print(f" {y:>7}", end="")
    print(f" {'All ✓?':>7}")
    print(f"  {'─'*80}")

    for name, m, yr in all_results:
        if name.startswith("Baseline"):
            continue
        print(f"  {name:<25}", end="")
        all_better = True
        for y in sorted(yr.keys()):
            delta = yr[y]["EV_bp"] - bl_yr[y]["EV_bp"]
            print(f" {delta:>+7.1f}", end="")
            if delta < -3.0:
                all_better = False
        status = "  YES" if all_better else "   NO"
        print(status)

    # ── IS/OOS for recommended config P3 (A4.0_D3_TP12%) ──
    print(f"\n{'─'*80}")
    print(f"  IS/OOS Validation — P3: A4.0_D3_TP12%")
    print(f"{'─'*80}")

    REC_TA, REC_TD, REC_TP = 0.040, 0.003, 0.12

    for sr in [0.50, 0.67, 0.75]:
        is_15m, oos_15m = split_data_temporal(data_15m, sr)
        is_1h, oos_1h = split_data_temporal(data_1h, sr)
        is_4h, oos_4h = split_data_temporal(data_4h, sr)
        make_split_vr([is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h], [(8, 56)])

        is_years = data_years * sr
        oos_years = data_years * (1.0 - sr)

        # Baseline
        is_bl = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "D_mom8") + \
                r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "S_d8")
        oos_bl = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "D_mom8") + \
                 r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")
        is_bl_pnl = portfolio_sim(is_bl, SLOTS)
        oos_bl_pnl = portfolio_sim(oos_bl, SLOTS)
        is_bl_m = compute_metrics(is_bl_pnl, is_years)
        oos_bl_m = compute_metrics(oos_bl_pnl, oos_years)

        # Optimal
        is_d = collect_d_widetp_trail(CFG_V13, is_15m, is_1h, is_4h, REC_TP, REC_TA, REC_TD)
        oos_d = collect_d_widetp_trail(CFG_V13, oos_15m, oos_1h, oos_4h, REC_TP, REC_TA, REC_TD)
        is_s = r38_collect_trades(CFG_V13, is_15m, is_1h, is_4h, "S_d8")
        oos_s = r38_collect_trades(CFG_V13, oos_15m, oos_1h, oos_4h, "S_d8")

        is_opt_pnl = portfolio_sim(is_d + is_s, SLOTS)
        oos_opt_pnl = portfolio_sim(oos_d + oos_s, SLOTS)
        is_opt_m = compute_metrics(is_opt_pnl, is_years)
        oos_opt_m = compute_metrics(oos_opt_pnl, oos_years)

        is_gain = is_opt_m["PF"] - is_bl_m["PF"]
        oos_gain = oos_opt_m["PF"] - oos_bl_m["PF"]
        robust = "✅" if oos_gain > 0 else "❌"

        print(f"  Split {sr:.0%}/{1-sr:.0%}  "
              f"IS: Δ={is_gain:+.3f} (BL {is_bl_m['PF']:.3f}→{is_opt_m['PF']:.3f})  "
              f"OOS: Δ={oos_gain:+.3f} (BL {oos_bl_m['PF']:.3f}→{oos_opt_m['PF']:.3f})  {robust}")

    # ── Bootstrap for P3 ──
    print(f"\n  Bootstrap CI (P3: A4.0_D3_TP12%):")
    d_trades = collect_d_widetp_trail(CFG_V13, data_15m, data_1h, data_4h, REC_TP, REC_TA, REC_TD)
    opt_pnl = portfolio_sim(d_trades + bl_s, SLOTS)
    opt_m = compute_metrics(opt_pnl, data_years)

    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_pf = np.empty(n_boot)
    bl_boot_pf = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(len(opt_pnl), size=len(opt_pnl), replace=True)
        boot_pf[b] = pf_from_pnl(opt_pnl[idx])
        idx2 = rng.choice(len(bl_pnl), size=len(bl_pnl), replace=True)
        bl_boot_pf[b] = pf_from_pnl(bl_pnl[idx2])

    diff = boot_pf - bl_boot_pf
    pct_better = np.mean(diff > 0) * 100
    lo, hi = np.percentile(diff, [2.5, 97.5])
    lo90, hi90 = np.percentile(diff, [5.0, 95.0])

    print(f"    PF = {opt_m['PF']:.3f}  95% CI: [{np.percentile(boot_pf, 2.5):.3f}, {np.percentile(boot_pf, 97.5):.3f}]")
    print(f"    ΔPF = {opt_m['PF'] - bl_m['PF']:+.3f}")
    print(f"    ΔPF 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    print(f"    ΔPF 90% CI: [{lo90:+.3f}, {hi90:+.3f}]")
    print(f"    Better in {pct_better:.0f}% of bootstrap samples")

    if lo > 0:
        print(f"    ✅ Significant at 95%")
    elif lo90 > 0:
        print(f"    ✅ Significant at 90%")
    else:
        print(f"    ⚠️ Not significant at 90%")

    # Win-loss breakdown
    print(f"\n  Win/Loss Analysis:")
    for label, pnl_arr in [("Baseline", bl_pnl), ("P3 (A4.0_D3)", opt_pnl)]:
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr < 0]
        avg_win = float(np.mean(wins)) * 10000 if len(wins) > 0 else 0
        avg_loss = float(np.mean(losses)) * 10000 if len(losses) > 0 else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        print(f"    {label:<20} WR={len(wins)}/{len(pnl_arr)} ({len(wins)/len(pnl_arr)*100:.1f}%)  "
              f"AvgW={avg_win:+.1f}bp  AvgL={avg_loss:+.1f}bp  R:R={rr:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.1f}s  |  Commit: {commit}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
