#!/usr/bin/env python3
"""r47b: Full comparison — v14.3.0 canonical vs r47 best combination.

v14.3.0 (baseline):
  D_mom8 trailing (ta=2%, td=0.3%) + F5 TBR + S_d8 + MA cluster flat TP=8%

r47 best (COMBO_D0.5_MA4):
  D_mom8 trailing (ta=0.5%, td=0.2%) + NO TBR + S_d8 + MA cluster trailing (ta=4%, td=0.5%)

Side-by-side: N, PF, WR, EV, Sharpe, Sortino, CAGR*(L/N), DD*, Calmar*
Per-year breakdown for both.
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
    sim_straddle_trades,
)
from quantbox.strategy.vol_straddle.signals import gen_vr_signal, make_direction_momentum
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r43_exit_improvement import year_metrics, get_year_boundaries
from research.active.r31_r40.scaling_r38 import CFG_V13
from research.active.r41_r45.backtest_v14_3_0 import (
    # Data
    load_15m,
    # Trade collectors (baseline)
    collect_d_trades_trailing,
    collect_d_trades_with_tbr,
    collect_enhanced_trades,
    # Helpers
    extended_metrics,
    print_full,
    # Constants
    D_SL, D_TP, D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION, MA_MIN_MOMENTUM,
    MA_SL, MA_TP, MA_TF_RESTRICTION,
    VR_LN, MA_LN,
)
# r47 custom collectors
from research.active.r47_r50.r47_exit_ab_test import (
    collect_d_trades_custom_trail,
    collect_ma_cluster_trailing,
    sim_ma_cluster_trailing,
)

# ── r47 best params (from Phase 4 combination test) ───────────────────────────
# D_mom8: lower trail_activate (0.5% vs 2.0%), tighter distance (0.2% vs 0.3%)
R47_TRAIL_ACTIVATE  = 0.005   # was 0.020
R47_TRAIL_DISTANCE  = 0.002   # was 0.003
# MA Cluster: trailing exit (was flat TP=8%)
R47_MA_TRAIL_ACT    = 0.04    # activate after +4% gain
R47_MA_TRAIL_DIST   = 0.005   # trail 0.5% below peak

MAX_CONCURRENT = 10


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
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
    print(f"  r47b: Full Comparison  |  v14.3.0 vs r47 Best Combination")
    print(f"  Commit: {commit}  |  Branch: research/r47-exit-ab-test")
    print(f"{'='*78}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n  Loading 15m data...")
    data_15m = load_15m()
    if not data_15m:
        print("❌ No data found.")
        sys.exit(1)

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    avg_bars  = int(np.mean([len(v["close"]) for v in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  {data_years:.2f} years")
    print(f"  Precompute: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE: v14.3.0 — D_mom8(trailing,F5-TBR) + S_d8 + MA(flat TP=8%)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  BASELINE  v14.3.0 — D(ta=2%,F5-TBR) + S_d8 + MA(flat TP=8%)")
    print(f"{'─'*78}")

    # D trades with F5 TBR (canonical v14.3.0)
    d_bl_full   = collect_d_trades_trailing(CFG_V13, data_15m, data_1h, data_4h)
    d_bl_trades, n_blocked = collect_d_trades_with_tbr(CFG_V13, data_15m, data_1h, data_4h)
    print(f"  D_mom8 (no TBR): {len(d_bl_full)} | after F5 TBR: {len(d_bl_trades)} ({n_blocked} blocked)")

    # S_d8 straddle (reused for both baseline and r47 — same params)
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")

    # MA cluster (flat TP=8%)
    ma_bl_trades = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )

    # Combined portfolio
    bl_comb = d_bl_trades + s_trades + ma_bl_trades
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_comb, MAX_CONCURRENT)
    n_vr_bl = len(d_bl_trades + s_trades)
    n_ma_bl = len(ma_bl_trades)
    bl_ln   = _blended_ln(n_vr_bl, n_ma_bl)
    bl_m    = extended_metrics(bl_pnl, data_years, ln_leverage=bl_ln)
    bl_boot = bootstrap_pf(bl_pnl, 5000)
    print_full(f"v14.3.0 Combined (L/N={bl_ln:.2f})", bl_m, bl_boot)

    print("\n  v14.3.0 per-year:")
    bl_yr = year_metrics(bl_pnl, bl_accepted, year_bounds)
    for key in sorted(bl_yr.keys()):
        y = bl_yr[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        print(f"    {key}  N={y['N']:>4}  PF={y['PF']:.3f}  "
              f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")

    # ══════════════════════════════════════════════════════════════════════════
    # r47 BEST: D_mom8(ta=0.5%,td=0.2%) + S_d8 + MA trailing(ta=4%,td=0.5%)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print(f"  r47 BEST  — D(ta=0.5%,td=0.2%,no-TBR) + S_d8 + MA trailing(ta=4%,td=0.5%)")
    print(f"{'─'*78}")

    d_r47_trades = collect_d_trades_custom_trail(
        CFG_V13, data_15m, data_1h, data_4h,
        trail_activate=R47_TRAIL_ACTIVATE, trail_distance=R47_TRAIL_DISTANCE,
    )
    ma_r47_trades = collect_ma_cluster_trailing(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
        trail_activate=R47_MA_TRAIL_ACT, trail_distance=R47_MA_TRAIL_DIST,
    )
    print(f"  D_mom8 (r47, ta=0.5%): {len(d_r47_trades)} | MA trailing (ta=4%): {len(ma_r47_trades)}")

    r47_comb = d_r47_trades + s_trades + ma_r47_trades
    r47_pnl, r47_accepted = portfolio_sim_with_details(r47_comb, MAX_CONCURRENT)
    n_vr_r47 = len(d_r47_trades + s_trades)
    n_ma_r47 = len(ma_r47_trades)
    r47_ln   = _blended_ln(n_vr_r47, n_ma_r47)
    r47_m    = extended_metrics(r47_pnl, data_years, ln_leverage=r47_ln)
    r47_boot = bootstrap_pf(r47_pnl, 5000)
    print_full(f"r47 Best Combined (L/N={r47_ln:.2f})", r47_m, r47_boot)

    print("\n  r47 per-year:")
    r47_yr = year_metrics(r47_pnl, r47_accepted, year_bounds)
    for key in sorted(r47_yr.keys()):
        y = r47_yr[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        print(f"    {key}  N={y['N']:>4}  PF={y['PF']:.3f}  "
              f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")

    # ══════════════════════════════════════════════════════════════════════════
    # SIDE-BY-SIDE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*78}")
    hdr = f"  {'Strategy':<38} {'N':>5} {'PF':>6} {'WR':>6} {'EV':>7} {'Sharpe':>7} {'CAGR*':>8} {'DD*':>6} {'Calmar*':>8}"
    print(hdr)
    print(f"  {'─'*95}")

    def _row(label: str, m: dict) -> str:
        cagr = m.get("prod_CAGR_pct", m["ann_ret_pct"])
        dd   = m.get("prod_max_dd_pct", m["max_dd_pct"])
        cal  = m.get("prod_calmar", m["calmar"])
        sh   = m.get("sharpe", 0.0)
        return (f"  {label:<38} {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>5.1f}% "
                f"{m['EV_bp']:>+6.1f}bp {sh:>7.2f} {cagr:>+7.1f}% {dd:>5.1f}% {cal:>8.1f}")

    print(_row(f"v14.3.0 (D ta=2%,TBR | MA flat TP)", bl_m))
    print(_row(f"r47 best (D ta=0.5% | MA trail ta=4%)", r47_m))

    # Deltas
    d_pf   = r47_m["PF"]           - bl_m["PF"]
    d_ev   = r47_m["EV_bp"]        - bl_m["EV_bp"]
    d_sh   = r47_m.get("sharpe",0) - bl_m.get("sharpe",0)
    d_cagr = r47_m.get("prod_CAGR_pct", r47_m["ann_ret_pct"]) - bl_m.get("prod_CAGR_pct", bl_m["ann_ret_pct"])
    d_dd   = r47_m.get("prod_max_dd_pct", r47_m["max_dd_pct"]) - bl_m.get("prod_max_dd_pct", bl_m["max_dd_pct"])
    d_n    = r47_m["N"] - bl_m["N"]
    print(f"  {'Δ (r47 - v14.3.0)':<38} {d_n:>+5} {d_pf:>+6.3f} {'':>6} {d_ev:>+6.1f}bp {d_sh:>+7.2f} {d_cagr:>+7.1f}% {d_dd:>+5.1f}% {'':>8}")

    print(f"\n  * Production leverage: VR L/N={VR_LN:.1f}, MA L/N={MA_LN:.1f}, blended v14={bl_ln:.2f}, blended r47={r47_ln:.2f}")
    print(f"\n  Per-year PF comparison:")
    print(f"  {'Year':<8} {'v14.3 N':>7} {'v14.3 PF':>9} {'v14.3 EV':>9} {'r47 N':>6} {'r47 PF':>8} {'r47 EV':>8}  Δ PF")
    print(f"  {'─'*70}")
    all_years = sorted(set(list(bl_yr.keys()) + list(r47_yr.keys())))
    for yr in all_years:
        b = bl_yr.get(yr, {})
        r = r47_yr.get(yr, {})
        dpf = r.get("PF", 0) - b.get("PF", 0)
        bf = "✓" if b.get("PF", 0) >= 1.0 else "❌"
        rf = "✓" if r.get("PF", 0) >= 1.0 else "❌"
        print(f"  {yr:<8} {b.get('N',0):>7}   {b.get('PF',0):>7.3f}{bf}  {b.get('EV_bp',0):>+7.1f}bp "
              f"{r.get('N',0):>6}  {r.get('PF',0):>7.3f}{rf}  {r.get('EV_bp',0):>+7.1f}bp  {dpf:>+.3f}")

    print(f"\n  Bootstrap 95% CI:")
    print(f"    v14.3.0: PF∈[{bl_boot['CI_lo']:.3f}, {bl_boot['CI_hi']:.3f}]  P(PF≥1)={bl_boot['P_PF_ge_1']:.1f}%")
    print(f"    r47 best: PF∈[{r47_boot['CI_lo']:.3f}, {r47_boot['CI_hi']:.3f}]  P(PF≥1)={r47_boot['P_PF_ge_1']:.1f}%")

    print(f"\n  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
