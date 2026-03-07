#!/usr/bin/env python3
"""r47b: Fair exit-only comparison — v14.3.0 exits vs r47 best exits.

METHODOLOGY NOTE
================
Both sides use IDENTICAL entry filters:
  - No F5 TBR (removed from baseline too, to match r47 Phase 0)
  - Same S_d8 straddle trades
  - Same MA cluster entry signals

Only the EXIT parameters differ:
  Baseline:  D_mom8 ta=2.0%, td=0.3%  |  MA cluster flat TP=8%
  r47 best:  D_mom8 ta=0.5%, td=0.2%  |  MA cluster trailing ta=4%, td=0.5%

WHY N STILL CHANGES
===================
Even with identical entry filters, portfolio_sim accepted N can differ because:
  1. Re-entry effect: earlier exits free slots, allowing new entries that were
     previously blocked — this is real system behavior, not a bug.
  2. Raw D trade count also changes: earlier exits within a symbol allow the
     scanner to detect a new entry signal in the same time window.

To isolate this, we report:
  - Raw trade counts (before portfolio_sim)
  - Accepted N (after portfolio_sim slot allocation)
  - Per-trade EV (fair comparison metric, N-independent)
"""

from __future__ import annotations

import subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ── Canonical imports ──────────────────────────────────────────────────────────
from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    G_3TF,
    bootstrap_pf,
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r43_exit_improvement import year_metrics, get_year_boundaries
from research.active.r31_r40.scaling_r38 import CFG_V13
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m,
    collect_d_trades_trailing,   # baseline D: ta=2%, no-TBR
    collect_enhanced_trades,     # MA cluster: flat TP=8%
    extended_metrics,
    print_full,
    D_SL, D_TP, D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION, MA_MIN_MOMENTUM,
    MA_SL, MA_TP, MA_TF_RESTRICTION,
    VR_LN, MA_LN,
)
from research.active.r47_r50.r47_exit_ab_test import (
    collect_d_trades_custom_trail,  # r47 D: ta=0.5%, no-TBR
    collect_ma_cluster_trailing,    # r47 MA: trailing ta=4%, td=0.5%
)

# ── r47 best exit params ───────────────────────────────────────────────────────
R47_TRAIL_ACTIVATE  = 0.005   # D_mom8: was 0.020
R47_TRAIL_DISTANCE  = 0.002   # D_mom8: was 0.003
R47_MA_TRAIL_ACT    = 0.04    # MA trailing activate
R47_MA_TRAIL_DIST   = 0.005   # MA trailing distance

MAX_CONCURRENT = 10


def get_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _blended_ln(vr_n: int, ma_n: int) -> float:
    total = vr_n + ma_n
    return (vr_n * VR_LN + ma_n * MA_LN) / total if total > 0 else 1.5


# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*78}")
    print(f"  r47b: Exit-Only Comparison  |  v14.3.0 exits vs r47 best exits")
    print(f"  Commit: {commit}  |  Branch: research/r47-exit-ab-test")
    print(f"  Both sides: same entry filter (no TBR), same S_d8")
    print(f"  Only exit params differ")
    print(f"{'='*78}")

    # ── Load & precompute ──────────────────────────────────────────────────────
    print("\n  Loading 15m data...")
    data_15m = load_15m()
    if not data_15m:
        sys.exit("❌ No data found.")

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    avg_bars   = int(np.mean([len(v["close"]) for v in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)
    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  {data_years:.2f} years  |  {time.time()-t0:.1f}s")

    # ── Shared: S_d8 straddle (identical for both) ─────────────────────────────
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    print(f"\n  S_d8 (shared): raw {len(s_trades)} straddle trades")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE: same entry filter, v14.3.0 exit params (ta=2%, flat MA TP=8%)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  BASELINE  D(ta=2.0%, td=0.3%, no-TBR) + S_d8 + MA(flat TP=8%)")
    print(f"{'─'*78}")

    d_bl = collect_d_trades_trailing(CFG_V13, data_15m, data_1h, data_4h)
    ma_bl = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )
    print(f"  Raw trades: D={len(d_bl)}  S={len(s_trades)}  MA={len(ma_bl)}  total={len(d_bl)+len(s_trades)+len(ma_bl)}")

    bl_comb = d_bl + s_trades + ma_bl
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_comb, MAX_CONCURRENT)
    bl_ln  = _blended_ln(len(d_bl + s_trades), len(ma_bl))
    bl_m   = extended_metrics(bl_pnl, data_years, ln_leverage=bl_ln)
    bl_boot = bootstrap_pf(bl_pnl, 5000)
    print(f"  Accepted N (portfolio_sim, 10 slots): {bl_m['N']}")
    print_full(f"Baseline (L/N={bl_ln:.2f})", bl_m, bl_boot)

    print("\n  Baseline per-year:")
    bl_yr = year_metrics(bl_pnl, bl_accepted, year_bounds)
    for key in sorted(bl_yr.keys()):
        y = bl_yr[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        print(f"    {key}  N={y['N']:>4}  PF={y['PF']:.3f}  EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")

    # ══════════════════════════════════════════════════════════════════════════
    # r47 BEST: same entry filter, r47 exit params (ta=0.5%, MA trailing)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  r47 BEST  D(ta=0.5%, td=0.2%, no-TBR) + S_d8 + MA trailing(ta=4%,td=0.5%)")
    print(f"{'─'*78}")

    d_r47 = collect_d_trades_custom_trail(
        CFG_V13, data_15m, data_1h, data_4h,
        trail_activate=R47_TRAIL_ACTIVATE, trail_distance=R47_TRAIL_DISTANCE,
    )
    ma_r47 = collect_ma_cluster_trailing(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
        trail_activate=R47_MA_TRAIL_ACT, trail_distance=R47_MA_TRAIL_DIST,
    )
    print(f"  Raw trades: D={len(d_r47)}  S={len(s_trades)}  MA={len(ma_r47)}  total={len(d_r47)+len(s_trades)+len(ma_r47)}")

    delta_d_raw  = len(d_r47)  - len(d_bl)
    delta_ma_raw = len(ma_r47) - len(ma_bl)
    print(f"  ΔRaw vs baseline: D={delta_d_raw:+d} (re-entry from earlier exits)  MA={delta_ma_raw:+d}")

    r47_comb = d_r47 + s_trades + ma_r47
    r47_pnl, r47_accepted = portfolio_sim_with_details(r47_comb, MAX_CONCURRENT)
    r47_ln  = _blended_ln(len(d_r47 + s_trades), len(ma_r47))
    r47_m   = extended_metrics(r47_pnl, data_years, ln_leverage=r47_ln)
    r47_boot = bootstrap_pf(r47_pnl, 5000)
    print(f"  Accepted N (portfolio_sim, 10 slots): {r47_m['N']}")
    print_full(f"r47 Best (L/N={r47_ln:.2f})", r47_m, r47_boot)

    print("\n  r47 per-year:")
    r47_yr = year_metrics(r47_pnl, r47_accepted, year_bounds)
    for key in sorted(r47_yr.keys()):
        y = r47_yr[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        print(f"    {key}  N={y['N']:>4}  PF={y['PF']:.3f}  EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")

    # ══════════════════════════════════════════════════════════════════════════
    # SIDE-BY-SIDE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print("  SIDE-BY-SIDE (same entry filter, exit params only differ)")
    print(f"{'='*78}")
    print(f"  {'':38} {'Raw D':>6} {'Raw MA':>7} {'Acc N':>6} {'PF':>6} {'WR':>6} {'EV':>7} {'Sharpe':>7}")
    print(f"  {'─'*85}")

    def _row2(label: str, raw_d: int, raw_ma: int, m: dict) -> str:
        sh = m.get("sharpe", 0.0)
        return (f"  {label:<38} {raw_d:>6} {raw_ma:>7} {m['N']:>6} "
                f"{m['PF']:>6.3f} {m['WR']:>5.1f}% {m['EV_bp']:>+6.1f}bp {sh:>7.2f}")

    print(_row2("Baseline (D ta=2% | MA flat TP=8%)", len(d_bl), len(ma_bl), bl_m))
    print(_row2("r47 best (D ta=0.5% | MA trail 4%)", len(d_r47), len(ma_r47), r47_m))

    d_pf  = r47_m["PF"] - bl_m["PF"]
    d_ev  = r47_m["EV_bp"] - bl_m["EV_bp"]
    d_sh  = r47_m.get("sharpe", 0) - bl_m.get("sharpe", 0)
    d_wr  = r47_m["WR"] - bl_m["WR"]
    d_n   = r47_m["N"] - bl_m["N"]
    d_rd  = len(d_r47) - len(d_bl)
    d_rm  = len(ma_r47) - len(ma_bl)
    print(f"  {'Δ (r47 - baseline)':<38} {d_rd:>+6} {d_rm:>+7} {d_n:>+6} "
          f"{d_pf:>+6.3f} {d_wr:>+5.1f}% {d_ev:>+6.1f}bp {d_sh:>+7.2f}")

    print(f"\n  Note: ΔRaw D={d_rd:+d} and ΔRaw MA={d_rm:+d} reflect re-entry effect from")
    print(f"        earlier exits freeing slots — real system behavior, not methodology error.")
    print(f"        Per-trade EV and PF are the fair comparison metrics (N-independent).")

    print(f"\n  Per-year PF comparison:")
    print(f"  {'Year':<8} {'BL N':>5} {'BL PF':>7} {'BL EV':>8}  {'r47 N':>6} {'r47 PF':>7} {'r47 EV':>8}  Δ PF")
    print(f"  {'─'*70}")
    for yr in sorted(set(list(bl_yr.keys()) + list(r47_yr.keys()))):
        b = bl_yr.get(yr, {})
        r = r47_yr.get(yr, {})
        dpf = r.get("PF", 0) - b.get("PF", 0)
        bf = "✓" if b.get("PF", 0) >= 1.0 else "❌"
        rf = "✓" if r.get("PF", 0) >= 1.0 else "❌"
        print(f"  {yr:<8} {b.get('N',0):>5}  {b.get('PF',0):>6.3f}{bf}  {b.get('EV_bp',0):>+7.1f}bp  "
              f"{r.get('N',0):>6}  {r.get('PF',0):>6.3f}{rf}  {r.get('EV_bp',0):>+7.1f}bp  {dpf:>+.3f}")

    print(f"\n  Bootstrap 95% CI:")
    print(f"    Baseline: PF∈[{bl_boot['CI_lo']:.3f}, {bl_boot['CI_hi']:.3f}]  P(PF≥1)={bl_boot['P_PF_ge_1']:.1f}%")
    print(f"    r47 best: PF∈[{r47_boot['CI_lo']:.3f}, {r47_boot['CI_hi']:.3f}]  P(PF≥1)={r47_boot['P_PF_ge_1']:.1f}%")

    print(f"\n  L/N: VR={VR_LN}, MA={MA_LN}, blended baseline={bl_ln:.2f}, blended r47={r47_ln:.2f}")
    print(f"  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()

