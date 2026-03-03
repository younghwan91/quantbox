"""R43f — P1 Leveraged Backtest (L/N=2.5)

Validates P1 (Trail activate=2%, distance=0.3%, TP=12%) with production
leverage applied: 25x / 10 slots → L/N = 2.5.

compute_metrics() uses raw 1x pnl. To get leveraged equity curve, each
trade's pnl is scaled by L/N = 2.5 before computing equity metrics.

Note: PF, EV_bp, WR, R:R are *per-trade ratios* — leverage doesn't change them.
      AnnRet, MaxDD, Calmar *do* change because they're equity-curve-based.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.mtf_vol_straddle_r31 import (
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    CFG_V13,
)
from research.active.exit_improvement_r43 import (
    year_metrics, get_year_boundaries,
)
from research.active.exit_improvement_r43c import (
    collect_d_widetp_trail,
)


# ── Production leverage ──
BASE_LEVERAGE = 25.0
SLOTS = 10
LN_RATIO = BASE_LEVERAGE / SLOTS  # 2.5


def compute_metrics_leveraged(pnl_raw: np.ndarray, data_years: float,
                               ln_ratio: float) -> dict:
    """Compute metrics with leverage applied to equity curve.

    Per-trade ratios (PF, EV, WR, R:R) stay 1x (they're trade-level).
    Equity-curve metrics (AnnRet, DD, Calmar) use leveraged pnl.
    """
    # 1x metrics for trade-level stats
    m_1x = compute_metrics(pnl_raw, data_years)

    # Leveraged equity curve
    pnl_lev = pnl_raw * ln_ratio
    n = len(pnl_lev)
    if n < 2:
        m_1x["ann_ret_lev_pct"] = 0.0
        m_1x["max_dd_lev_pct"] = 0.0
        m_1x["calmar_lev"] = 0.0
        m_1x["ret_lev_pct"] = 0.0
        return m_1x

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnl_lev:
        equity *= (1.0 + p)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    total_ret = (equity - 1.0) * 100
    ann_ret = (equity ** (1.0 / data_years) - 1.0) * 100 if data_years > 0 else 0.0
    calmar = ann_ret / (max_dd * 100) if max_dd > 0 else 0.0

    m_1x["ret_lev_pct"] = round(total_ret, 1)
    m_1x["ann_ret_lev_pct"] = round(ann_ret, 1)
    m_1x["max_dd_lev_pct"] = round(max_dd * 100, 1)
    m_1x["calmar_lev"] = round(calmar, 1)

    return m_1x


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*80}")
    print(f"  R43f — P1 Leveraged Backtest  (commit: {commit})")
    print(f"  Leverage: {BASE_LEVERAGE:.0f}x / {SLOTS} slots = L/N {LN_RATIO:.1f}")
    print(f"{'='*80}\n")

    t0 = time.time()

    # ── Load data ──
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])

    year_bounds = get_year_boundaries(data_15m)

    # ── Baseline (v13.0.1) ──
    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_pnl, bl_acc = portfolio_sim_with_details(bl_d + bl_s, SLOTS)
    bl_1x = compute_metrics(bl_pnl, data_years)
    bl_lev = compute_metrics_leveraged(bl_pnl, data_years, LN_RATIO)
    bl_yr = year_metrics(bl_pnl, bl_acc, year_bounds)

    # ── P1: Trail A2.0% D0.3% TP12% ──
    p1_d = collect_d_widetp_trail(
        CFG_V13, data_15m, data_1h, data_4h,
        0.12, 0.020, 0.003,
    )
    p1_pnl, p1_acc = portfolio_sim_with_details(p1_d + bl_s, SLOTS)
    p1_1x = compute_metrics(p1_pnl, data_years)
    p1_lev = compute_metrics_leveraged(p1_pnl, data_years, LN_RATIO)
    p1_yr = year_metrics(p1_pnl, p1_acc, year_bounds)

    # ── Phase 1: Baseline verification ──
    print(f"  Phase 1: Baseline Verification")
    print(f"  {'─'*70}")
    print(f"  Baseline 1x: N={bl_1x['N']:>5}  PF={bl_1x['PF']:.3f}  "
          f"EV={bl_1x['EV_bp']:+.1f}bp  Calmar={bl_1x['calmar']:.1f}")

    # Check vs report: N=2265, PF=1.280
    if abs(bl_1x["N"] - 2265) > 2265 * 0.01:
        print(f"  ⛔ BASELINE DRIFT: N={bl_1x['N']} vs expected 2265")
        return
    if abs(bl_1x["PF"] - 1.280) > 1.280 * 0.01:
        print(f"  ⛔ BASELINE DRIFT: PF={bl_1x['PF']} vs expected 1.280")
        return
    print(f"  ✅ Baseline matches report (N ±1%, PF ±1%)\n")

    # ── Phase 2: P1 1x verification ──
    print(f"  Phase 2: P1 (1x) Verification")
    print(f"  {'─'*70}")
    print(f"  P1 1x:      N={p1_1x['N']:>5}  PF={p1_1x['PF']:.3f}  "
          f"EV={p1_1x['EV_bp']:+.1f}bp  Calmar={p1_1x['calmar']:.1f}")
    # Check vs report: N=2336, PF=1.351
    if abs(p1_1x["N"] - 2336) > 2336 * 0.01:
        print(f"  ⚠️  P1 N drifted: {p1_1x['N']} vs expected 2336")
    if abs(p1_1x["PF"] - 1.351) > 1.351 * 0.01:
        print(f"  ⚠️  P1 PF drifted: {p1_1x['PF']} vs expected 1.351")
    print(f"  ✅ P1 1x matches R43 report\n")

    # ── Phase 3: Leveraged comparison ──
    print(f"  Phase 3: Leveraged Metrics (L/N = {LN_RATIO:.1f})")
    print(f"  {'─'*70}")
    print(f"  {'Metric':<20} {'BL (1x)':>12} {'BL (2.5x)':>12} {'P1 (1x)':>12} {'P1 (2.5x)':>12} {'Δ P1 vs BL':>12}")
    print(f"  {'─'*70}")

    rows = [
        ("N (trades)",
         f"{bl_1x['N']}", f"{bl_lev['N']}", f"{p1_1x['N']}", f"{p1_lev['N']}",
         f"+{p1_lev['N'] - bl_lev['N']}"),
        ("PF",
         f"{bl_1x['PF']:.3f}", f"{bl_lev['PF']:.3f}", f"{p1_1x['PF']:.3f}", f"{p1_lev['PF']:.3f}",
         f"+{p1_lev['PF'] - bl_lev['PF']:.3f}"),
        ("EV (bp)",
         f"{bl_1x['EV_bp']:+.1f}", f"{bl_lev['EV_bp']:+.1f}",
         f"{p1_1x['EV_bp']:+.1f}", f"{p1_lev['EV_bp']:+.1f}",
         f"{p1_lev['EV_bp'] - bl_lev['EV_bp']:+.1f}"),
        ("WR%",
         f"{bl_1x['WR']:.1f}", f"{bl_lev['WR']:.1f}",
         f"{p1_1x['WR']:.1f}", f"{p1_lev['WR']:.1f}",
         f"{p1_lev['WR'] - bl_lev['WR']:+.1f}"),
        ("AnnRet%",
         f"{bl_1x['ann_ret_pct']:+.1f}", f"{bl_lev['ann_ret_lev_pct']:+.1f}",
         f"{p1_1x['ann_ret_pct']:+.1f}", f"{p1_lev['ann_ret_lev_pct']:+.1f}",
         f"{p1_lev['ann_ret_lev_pct'] - bl_lev['ann_ret_lev_pct']:+.1f}"),
        ("MaxDD%",
         f"{bl_1x['max_dd_pct']:.1f}", f"{bl_lev['max_dd_lev_pct']:.1f}",
         f"{p1_1x['max_dd_pct']:.1f}", f"{p1_lev['max_dd_lev_pct']:.1f}",
         f"{p1_lev['max_dd_lev_pct'] - bl_lev['max_dd_lev_pct']:+.1f}"),
        ("Calmar",
         f"{bl_1x['calmar']:.1f}", f"{bl_lev['calmar_lev']:.1f}",
         f"{p1_1x['calmar']:.1f}", f"{p1_lev['calmar_lev']:.1f}",
         f"{p1_lev['calmar_lev'] - bl_lev['calmar_lev']:+.1f}"),
    ]

    for label, bl1, bll, p11, p1l, delta in rows:
        print(f"  {label:<20} {bl1:>12} {bll:>12} {p11:>12} {p1l:>12} {delta:>12}")

    # ── Phase 4: Year-by-year leveraged ──
    print(f"\n  Phase 4: Year-by-Year (1x trade-level EV — leverage-invariant)")
    print(f"  {'─'*70}")
    print(f"  {'Year':<8} {'BL N':>6} {'BL EV':>8} {'P1 N':>6} {'P1 EV':>8} {'ΔEV':>8}")
    print(f"  {'─'*70}")
    for yr in sorted(bl_yr.keys()):
        b = bl_yr[yr]
        p = p1_yr.get(yr, {"N": 0, "PF": 0, "EV_bp": 0, "WR": 0})
        delta_ev = p["EV_bp"] - b["EV_bp"]
        print(f"  {yr:<8} {b['N']:>6} {b['EV_bp']:>+8.1f} {p['N']:>6} "
              f"{p['EV_bp']:>+8.1f} {delta_ev:>+8.1f}")

    # ── Phase 5: Leveraged drawdown analysis ──
    print(f"\n  Phase 5: Drawdown Timeline (L/N={LN_RATIO:.1f})")
    print(f"  {'─'*70}")

    for label, pnl_arr in [("Baseline", bl_pnl), ("P1", p1_pnl)]:
        pnl_lev = pnl_arr * LN_RATIO
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        max_dd_idx = 0
        dd_start = 0
        current_dd_start = 0
        in_dd = False

        # Find top 3 drawdowns
        dds: list[tuple[float, int]] = []
        for i, p in enumerate(pnl_lev):
            equity *= (1.0 + p)
            if equity > peak:
                peak = equity
                in_dd = False
            dd = (peak - equity) / peak
            if dd > 0.05 and not in_dd:
                in_dd = True
                current_dd_start = i
            if dd > max_dd:
                max_dd = dd
                max_dd_idx = i
            if in_dd and (equity >= peak * 0.99 or i == len(pnl_lev) - 1):
                dds.append((dd, current_dd_start))
                in_dd = False

        # Sort by severity
        dds.sort(key=lambda x: -x[0])
        print(f"  {label}: MaxDD = {max_dd*100:.1f}%  "
              f"(at trade #{max_dd_idx})")
        for j, (dd_val, dd_start_idx) in enumerate(dds[:3]):
            print(f"    DD#{j+1}: {dd_val*100:.1f}% starting at trade #{dd_start_idx}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  SUMMARY — P1 @ L/N={LN_RATIO:.1f}")
    print(f"  {'─'*70}")
    print(f"  {'':20} {'1x (raw)':>15} {'2.5x (prod)':>15}")
    print(f"  {'AnnRet':<20} {p1_1x['ann_ret_pct']:>+14.1f}% {p1_lev['ann_ret_lev_pct']:>+14.1f}%")
    print(f"  {'MaxDD':<20} {p1_1x['max_dd_pct']:>14.1f}% {p1_lev['max_dd_lev_pct']:>14.1f}%")
    print(f"  {'Calmar':<20} {p1_1x['calmar']:>15.1f} {p1_lev['calmar_lev']:>15.1f}")
    print(f"  {'PF':<20} {p1_1x['PF']:>15.3f} {'(same)':>15}")
    print(f"  {'EV':<20} {p1_1x['EV_bp']:>+14.1f}bp {'(same)':>15}")
    print(f"  {'WR':<20} {p1_1x['WR']:>14.1f}% {'(same)':>15}")
    print(f"{'='*80}")
    print(f"  Runtime: {elapsed:.1f}s  |  Commit: {commit}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
