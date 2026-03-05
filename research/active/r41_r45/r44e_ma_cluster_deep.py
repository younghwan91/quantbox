"""R44e — MA Cluster Deep Analysis

Deep validation of V17 (4h-only MA Cluster Squeeze):
  1. Year-by-year stability (must be positive in ≥5/6 years)
  2. VR squeeze overlap analysis (trade timing correlation)
  3. Combined portfolio: VR Squeeze + MA Cluster
  4. Bootstrap PF confidence interval (P(PF≥1.0) > 95%)
  5. Per-symbol distribution

Top V17 candidates from R44d:
  A: c=3% d=8 sl=2% tp=8%   → N=1836, PF=1.285, DD=35.5%, Calmar=5.4
  B: c=3% d=8 sl=1.5% tp=8% → N=1935, PF=1.254, DD=43.0%, Calmar=3.3
  C: c=2% d=8 sl=1.5% tp=8% → N=727,  PF=1.478, DD=30.7%, Calmar=2.8
  D: c=3% d=8 sl=2% tp=12%  → N=1825, PF=1.308, DD=41.6%, Calmar=5.1
  E: c=3% d=2 sl=1.5% tp=8% → N=4018, PF=1.185, DD=48.3%, Calmar=5.4
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    load_15m, build_resampled, precompute, pf_from_pnl,
    bootstrap_pf,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    COST_DIRECTIONAL, G_3TF, CFG_V13,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r44d_ma_cluster_refined import collect_v16_trades
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics, get_year_boundaries,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Overlap analysis: temporal overlap between two trade sets
# ══════════════════════════════════════════════════════════════════════════════


def overlap_fraction(
    trades_a: list[tuple[str, int, int, float]],
    trades_b: list[tuple[str, int, int, float]],
) -> tuple[float, float, int]:
    """Compute what fraction of B's trades overlap with A's trades.

    Returns: (frac_b_overlap_a, frac_a_overlap_b, n_overlap)
    """
    if not trades_a or not trades_b:
        return 0.0, 0.0, 0

    # Build interval sets per symbol for A
    a_intervals: dict[str, list[tuple[int, int]]] = {}
    for sym, entry, exit_b, _ in trades_a:
        a_intervals.setdefault(sym, []).append((entry, exit_b))

    # Check each B trade against A's intervals
    b_overlap = 0
    for sym, entry, exit_b, _ in trades_b:
        if sym not in a_intervals:
            continue
        for ae, ax in a_intervals[sym]:
            if entry <= ax and exit_b >= ae:  # overlap
                b_overlap += 1
                break

    # Check each A trade against B's intervals
    b_intervals: dict[str, list[tuple[int, int]]] = {}
    for sym, entry, exit_b, _ in trades_b:
        b_intervals.setdefault(sym, []).append((entry, exit_b))

    a_overlap = 0
    for sym, entry, exit_b, _ in trades_a:
        if sym not in b_intervals:
            continue
        for be, bx in b_intervals[sym]:
            if entry <= bx and exit_b >= be:
                a_overlap += 1
                break

    frac_b = b_overlap / len(trades_b) if trades_b else 0.0
    frac_a = a_overlap / len(trades_a) if trades_a else 0.0
    return frac_b, frac_a, b_overlap


# ══════════════════════════════════════════════════════════════════════════════
#  Combined portfolio: merge two trade sets with slot management
# ══════════════════════════════════════════════════════════════════════════════


def combined_portfolio(
    trades_a: list[tuple[str, int, int, float]],
    trades_b: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> tuple[np.ndarray, list[tuple[str, int, int, float]]]:
    """Merge two trade sets and run through portfolio_sim_with_details."""
    combined = trades_a + trades_b
    return portfolio_sim_with_details(combined, max_concurrent)


def main() -> None:
    commit = get_commit_short()
    print(f"\nR44e — MA Cluster Deep Analysis [{commit}]")
    print("=" * 80)

    # ── Load data ──
    t0 = time.time()
    print("Loading data...")
    data_15m = load_15m()
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    vr_pairs = [(8, 56)]
    precompute_all_vr([data_15m, data_1h, data_4h], vr_pairs)
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
        precompute_mas(ds)

    bars_list = [d["bars"] for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list)) if bars_list else 0
    data_years = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)

    print(f"  {len(data_15m)} symbols, {data_years:.1f} years, "
          f"load time: {time.time() - t0:.1f}s")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  1. Baseline: VR Squeeze (CFG_V13)
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("1. BASELINE: VR Squeeze (CFG_V13)")
    print("=" * 80)
    bl_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_trades, 10)
    bl_m = compute_metrics(bl_pnl, data_years)
    bl_yr = year_metrics(bl_pnl, bl_accepted, year_bounds)
    bl_boot = bootstrap_pf(bl_pnl, 5000)

    print(f"  N={bl_m['N']}  PF={bl_m['PF']:.3f}  WR={bl_m['WR']:.1f}%  "
          f"EV={bl_m['EV_bp']:+.1f}bp  DD={bl_m['max_dd_pct']:.1f}%  "
          f"Calmar={bl_m['calmar']:.1f}")
    print(f"  Bootstrap: P(PF≥1.0)={bl_boot['P_PF_ge_1']}%  "
          f"CI=[{bl_boot['CI_lo']:.3f}, {bl_boot['CI_hi']:.3f}]  "
          f"median={bl_boot['median']:.3f}")
    print("  Yearly:")
    for yr in sorted(bl_yr.keys()):
        y = bl_yr[yr]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        print(f"    {yr}: N={y['N']:>4}  PF={y['PF']:.2f}  "
              f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  2. V17 Candidates: Deep Analysis
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("2. V17 CANDIDATES: Deep Analysis")
    print("=" * 80)

    candidates = [
        ("A: c=3% d=8 sl=2% tp=8%",   0.03, 8, 0.02, 0.08),
        ("B: c=3% d=8 sl=1.5% tp=8%", 0.03, 8, 0.015, 0.08),
        ("C: c=2% d=8 sl=1.5% tp=8%", 0.02, 8, 0.015, 0.08),
        ("D: c=3% d=8 sl=2% tp=12%",  0.03, 8, 0.02, 0.12),
        ("E: c=3% d=2 sl=1.5% tp=8%", 0.03, 2, 0.015, 0.08),
        ("F: c=3% d=4 sl=1.5% tp=8%", 0.03, 4, 0.015, 0.08),
        ("G: c=3% d=16 sl=1% tp=8%",  0.03, 16, 0.01, 0.08),
    ]

    best_candidate = None
    best_calmar = 0.0

    for name, cluster_pct, min_dur, sl, tp in candidates:
        print(f"\n  ── {name} ──")

        # Collect trades (4h only = tf_restriction=16)
        trades = collect_v16_trades(
            data_15m, data_1h, data_4h,
            cluster_pct, min_dur, sl, tp,
            tf_restriction=16)

        if not trades:
            print("    0 trades — skip")
            continue

        pnl, accepted = portfolio_sim_with_details(trades, 10)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, accepted, year_bounds)
        boot = bootstrap_pf(pnl, 5000)

        print(f"    N={m['N']}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
              f"EV={m['EV_bp']:+.1f}bp  RR={m['RR']:.2f}")
        print(f"    AnnRet={m['ann_ret_pct']:+.1f}%  DD={m['max_dd_pct']:.1f}%  "
              f"Calmar={m['calmar']:.1f}  TPD={m['tpd']:.2f}")
        print(f"    Bootstrap: P(PF≥1.0)={boot['P_PF_ge_1']}%  "
              f"CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
              f"median={boot['median']:.3f}")

        # Year-by-year
        print("    Yearly:")
        positive_years = 0
        for yr_key in sorted(yr.keys()):
            y = yr[yr_key]
            flag = "✓" if y["PF"] >= 1.0 else "❌"
            if y["PF"] >= 1.0:
                positive_years += 1
            print(f"      {yr_key}: N={y['N']:>4}  PF={y['PF']:.2f}  "
                  f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")
        print(f"    Positive years: {positive_years}/{len(yr)}")

        # Overlap with VR squeeze
        overlap_b, overlap_a, n_ovl = overlap_fraction(bl_accepted, accepted)
        print(f"    VR overlap: {overlap_b*100:.1f}% of MA trades overlap VR  "
              f"({n_ovl} trades)")
        print(f"                {overlap_a*100:.1f}% of VR trades overlap MA")

        # Per-symbol distribution
        sym_counts: dict[str, int] = {}
        sym_pnls: dict[str, list[float]] = {}
        for sym, entry, exit_b, pnl_val in accepted:
            sym_counts[sym] = sym_counts.get(sym, 0) + 1
            sym_pnls.setdefault(sym, []).append(pnl_val)
        active_syms = len(sym_counts)
        top_syms = sorted(sym_counts.items(), key=lambda x: -x[1])[:5]
        print(f"    Symbols traded: {active_syms}/50")
        top_str = ", ".join(f"{s}({n})" for s, n in top_syms)
        print(f"    Top 5: {top_str}")

        # Track best
        if m['calmar'] > best_calmar and m['N'] > 200:
            best_calmar = m['calmar']
            best_candidate = (name, trades, accepted, m, yr, boot)

    # ══════════════════════════════════════════════════════════════════════
    #  3. Combined Portfolio: VR Squeeze + Best MA Cluster
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 80)
    print("3. COMBINED PORTFOLIO: VR Squeeze + Best MA Cluster")
    print("=" * 80)

    if best_candidate:
        bc_name, bc_trades, bc_accepted, bc_m, _, _ = best_candidate
        print(f"  Best candidate: {bc_name}")
        print()

        # Try different slot allocations
        for total_slots in [10, 12, 15]:
            comb_pnl, comb_accepted = combined_portfolio(
                bl_trades, bc_trades, total_slots)
            comb_m = compute_metrics(comb_pnl, data_years)
            comb_yr = year_metrics(comb_pnl, comb_accepted, year_bounds)
            comb_boot = bootstrap_pf(comb_pnl, 5000)

            print(f"  ── {total_slots} slots ──")
            print(f"    N={comb_m['N']}  PF={comb_m['PF']:.3f}  "
                  f"WR={comb_m['WR']:.1f}%  EV={comb_m['EV_bp']:+.1f}bp")
            print(f"    AnnRet={comb_m['ann_ret_pct']:+.1f}%  "
                  f"DD={comb_m['max_dd_pct']:.1f}%  "
                  f"Calmar={comb_m['calmar']:.1f}  TPD={comb_m['tpd']:.2f}")
            print(f"    Bootstrap: P(PF≥1.0)={comb_boot['P_PF_ge_1']}%  "
                  f"CI=[{comb_boot['CI_lo']:.3f}, {comb_boot['CI_hi']:.3f}]")

            # Yearly
            print("    Yearly:")
            for yr_key in sorted(comb_yr.keys()):
                y = comb_yr[yr_key]
                flag = "✓" if y["PF"] >= 1.0 else "❌"
                print(f"      {yr_key}: N={y['N']:>4}  PF={y['PF']:.2f}  "
                      f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")

            # How many trades came from each strategy?
            n_vr = sum(1 for t in comb_accepted
                       if any(t[0] == bt[0] and t[1] == bt[1]
                              for bt in bl_accepted))
            n_ma = comb_m['N'] - n_vr
            print(f"    Composition: ~{n_vr} VR + ~{n_ma} MA = {comb_m['N']}")
            print()

        # Also test ALL candidates combined
        print("  ── All V17 candidates combined with VR (10 slots) ──")
        for name, cluster_pct, min_dur, sl, tp in candidates:
            trades = collect_v16_trades(
                data_15m, data_1h, data_4h,
                cluster_pct, min_dur, sl, tp,
                tf_restriction=16)
            if not trades:
                continue
            comb_pnl, comb_acc = combined_portfolio(bl_trades, trades, 10)
            comb_m = compute_metrics(comb_pnl, data_years)
            comb_boot = bootstrap_pf(comb_pnl, 5000)
            print(f"    VR + {name}: N={comb_m['N']:>5}  PF={comb_m['PF']:.3f}  "
                  f"DD={comb_m['max_dd_pct']:.1f}%  Calmar={comb_m['calmar']:.1f}  "
                  f"P(PF≥1)={comb_boot['P_PF_ge_1']}%")

    # ══════════════════════════════════════════════════════════════════════
    #  4. Conclusion
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 80)
    print("4. CONCLUSION")
    print("=" * 80)
    print(f"  Baseline VR Squeeze: N={bl_m['N']}  PF={bl_m['PF']:.3f}  "
          f"Calmar={bl_m['calmar']:.1f}")
    if best_candidate:
        bc_name, _, _, bc_m, _, bc_boot = best_candidate
        print(f"  Best MA Cluster:     {bc_name}")
        print(f"    N={bc_m['N']}  PF={bc_m['PF']:.3f}  "
              f"Calmar={bc_m['calmar']:.1f}  "
              f"P(PF≥1)={bc_boot['P_PF_ge_1']}%")

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
    print(f"  Commit: {commit}")


if __name__ == "__main__":
    main()
