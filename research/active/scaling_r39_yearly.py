"""R39 — Yearly Consistency + Aggressive Ranking.

Goal: Find the variant with highest compound return AND consistent yearly profits.
Rank by: compound return × calmar (aggression metric).
Show year-by-year equity for top candidates.
Unlimited slots first → slot-optimize finalists.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from research.active.mtf_vol_straddle_r31 import (
    load_15m, build_resampled, precompute, compute_vr, compute_momentum,
    compute_squeeze_dur, vol_percentile, gen_vr_signal, gen_vr_dur_signal,
    make_direction_momentum,
)
from research.active.scaling_r38 import (
    collect_trades, portfolio_sim, portfolio_sim_with_details,
    compute_metrics, precompute_all_vr,
    sim_straddle_trades, sim_directional_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF,
)

BARS_PER_YEAR = 365 * 24 * 4  # 35040


def yearly_equity(
    trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    n_years: int = 6,
) -> list[dict]:
    """Compute year-by-year metrics from portfolio-simulated trades.

    Returns list of dicts with year index, N, return%, equity multiplier.
    Trades are assigned to year by entry_15m bar index.
    """
    # Run portfolio sim to get accepted trades
    _, accepted = portfolio_sim_with_details(trades, max_concurrent)

    yearly: list[dict] = []
    for yr in range(n_years):
        yr_start = yr * BARS_PER_YEAR
        yr_end = (yr + 1) * BARS_PER_YEAR
        yr_pnl = [pnl for _, entry, _, pnl in accepted
                  if yr_start <= entry < yr_end]
        if not yr_pnl:
            yearly.append({"year": yr, "N": 0, "ret_pct": 0.0, "eq_mult": 1.0,
                           "max_dd_pct": 0.0, "pf": 0.0})
            continue

        eq = 1.0
        peak = 1.0
        max_dd = 0.0
        for p in yr_pnl:
            eq *= (1.0 + p)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        wins = [p for p in yr_pnl if p > 0]
        losses = [abs(p) for p in yr_pnl if p < 0]
        gross_win = sum(wins) if wins else 0.0
        gross_loss = sum(losses) if losses else 1e-12
        pf = gross_win / gross_loss

        yearly.append({
            "year": yr,
            "N": len(yr_pnl),
            "ret_pct": (eq - 1.0) * 100,
            "eq_mult": eq,
            "max_dd_pct": max_dd * 100,
            "pf": pf,
        })

    return yearly


def compound_return(
    trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> float:
    """Total compound return as multiplier (1.0 = breakeven)."""
    pnl = portfolio_sim(trades, max_concurrent)
    if len(pnl) == 0:
        return 1.0
    eq = 1.0
    for p in pnl:
        eq *= (1.0 + p)
    return eq


def full_metrics_with_yearly(
    trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    data_years: float,
) -> dict:
    """Full metrics + yearly breakdown."""
    pnl = portfolio_sim(trades, max_concurrent)
    m = compute_metrics(pnl, data_years)
    yr = yearly_equity(trades, max_concurrent, n_years=int(data_years))
    m["yearly"] = yr
    m["losing_years"] = sum(1 for y in yr if y["ret_pct"] < 0)
    m["min_year_ret"] = min(y["ret_pct"] for y in yr) if yr else 0.0
    return m


def main() -> None:
    t0 = time.time()
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    n_years = int(data_years)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)

    all_pairs = [
        (4,28),(4,30),(4,35),(4,40),
        (5,30),(5,35),(5,40),(5,45),
        (6,30),(6,35),(6,40),(6,42),(6,45),(6,50),
        (7,35),(7,40),(7,42),(7,45),(7,50),
        (8,40),(8,50),(8,56),
    ]
    precompute_all_vr([data_15m, data_1h, data_4h], all_pairs)

    # ═══════════════════════════════════════════════════════
    #  Phase 1: Unlimited slots — raw compound return ranking
    # ═══════════════════════════════════════════════════════
    configs: list[tuple[str, dict]] = []
    for sp, lp in all_pairs:
        for d_vr in [0.45, 0.50, 0.55]:
            s_vr = d_vr + 0.05
            for s_dur in [4, 8]:
                name = f"VR{sp}_{lp}_T{int(d_vr*100)}_D{s_dur}"
                configs.append((name, {
                    "sp": sp, "lp": lp, "d_vr": d_vr, "s_vr": s_vr,
                    "s_dur": s_dur, "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
                }))

    print(f"Phase 1: Collecting trades for {len(configs)} variants (unlimited)...")
    t1 = time.time()

    variant_trades: dict[str, list] = {}
    results: list[dict] = []

    for idx, (name, cfg) in enumerate(configs):
        if (idx + 1) % 30 == 0:
            print(f"  ... {idx+1}/{len(configs)} ({time.time()-t1:.0f}s)")

        d_t = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
        s_t = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
        trades = d_t + s_t
        variant_trades[name] = trades

        # Unlimited metrics
        m_unlim = full_metrics_with_yearly(trades, 9999, data_years)
        results.append({"name": name, "cfg": cfg, "unlim": m_unlim})

    phase1_time = time.time() - t1
    print(f"  Phase 1 done in {phase1_time:.0f}s")

    # Filter: PF > 1.0, N > 1000, no more than 1 losing year
    valid = [r for r in results
             if r["unlim"]["PF"] > 1.0
             and r["unlim"]["N"] > 1000
             and r["unlim"]["losing_years"] <= 1]

    # Sort by compound return (ann_ret already in %)
    valid.sort(key=lambda r: r["unlim"]["ann_ret_pct"], reverse=True)

    # ═══════════════════════════════════════════════════════
    #  Phase 1 Table: Top 20 by compound return (unlimited)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print(f"  Phase 1: {len(valid)} / {len(results)} passed (PF>1.0, N>1000, losing_years≤1)")
    print(f"  Sorted by: Annual Return (unlimited slots = raw signal power)")
    print(f"{'='*130}")

    hdr = (f"  {'#':>2} {'Variant':<22} {'N':>6} {'PF':>5} {'EV':>6} "
           f"{'WR':>4} {'R:R':>4} {'TPD':>5} "
           f"{'AnnRet%':>9} {'DD%':>5} {'Calmar':>7} "
           f"{'LoseYr':>6}")
    print(hdr)
    print("  " + "─" * 125)

    for rank, r in enumerate(valid[:30], 1):
        u = r["unlim"]
        print(f"  {rank:>2} {r['name']:<22} {u['N']:>6} {u['PF']:>5.3f} "
              f"{u['EV_bp']:>+6.1f} {u['WR']:>4.1f} {u['RR']:>4.2f} "
              f"{u['tpd']:>5.2f} {u['ann_ret_pct']:>+9.1f} "
              f"{u['max_dd_pct']:>5.1f} {u['calmar']:>7.1f} "
              f"{u['losing_years']:>6}")

    # References
    print(f"\n  ── References ──")
    for ref_name in ["VR5_35_T50_D8", "VR7_50_T50_D8"]:
        for r in results:
            if r["name"] == ref_name:
                u = r["unlim"]
                print(f"  -- {r['name']:<22} {u['N']:>6} {u['PF']:>5.3f} "
                      f"{u['EV_bp']:>+6.1f} {u['WR']:>4.1f} {u['RR']:>4.2f} "
                      f"{u['tpd']:>5.2f} {u['ann_ret_pct']:>+9.1f} "
                      f"{u['max_dd_pct']:>5.1f} {u['calmar']:>7.1f} "
                      f"{u['losing_years']:>6}")

    # ═══════════════════════════════════════════════════════
    #  Phase 2: Year-by-year breakdown for top 10 + refs
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print(f"  Phase 2: Year-by-Year Equity (unlimited slots)")
    print(f"  Years: Y0=2020, Y1=2021, Y2=2022, Y3=2023, Y4=2024, Y5=2025")
    print(f"{'='*130}")

    targets = valid[:10]
    for ref_name in ["VR5_35_T50_D8", "VR7_50_T50_D8"]:
        for r in results:
            if r["name"] == ref_name and r not in targets:
                targets.append(r)

    # Table header for yearly
    yr_hdr = (f"  {'Variant':<22} "
              + " | ".join(f"{'Y'+str(i)+' Ret%':>9} {'DD%':>5} {'N':>4}" for i in range(n_years)))
    print(yr_hdr)
    print("  " + "─" * (22 + n_years * 22))

    for r in targets:
        yrs = r["unlim"]["yearly"]
        parts = []
        for y in yrs:
            parts.append(f"{y['ret_pct']:>+9.1f} {y['max_dd_pct']:>5.1f} {y['N']:>4}")
        print(f"  {r['name']:<22} " + " | ".join(parts))

    # ═══════════════════════════════════════════════════════
    #  Phase 3: Slot optimization for top 10
    #  Aggressive: pick slot count that maximizes AnnRet × Calmar
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print(f"  Phase 3: Slot × Aggression Optimization")
    print(f"  Metric: AnnRet% × Calmar (aggression score)")
    print(f"{'='*130}")

    t3 = time.time()
    slot_candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]

    final_results: list[dict] = []

    for r in targets:
        name = r["name"]
        trades = variant_trades[name]
        unlim_n = len(trades)

        print(f"\n  {name} (unlimited N={unlim_n}):")
        print(f"  {'Slots':>5} {'N':>6} {'PF':>5} {'EV':>6} {'TPD':>5} "
              f"{'AnnRet%':>9} {'DD%':>5} {'Calmar':>7} {'Aggr':>9} "
              f"{'LoseYr':>6} {'Fill%':>6}")
        print(f"  {'─'*85}")

        best_aggr = 0.0
        best_slot_data = None

        for slots in slot_candidates:
            m = full_metrics_with_yearly(trades, slots, data_years)
            fill = m["N"] / unlim_n * 100 if unlim_n else 0
            aggr = m["ann_ret_pct"] * m["calmar"] if m["calmar"] > 0 else 0.0

            marker = ""
            if aggr > best_aggr:
                best_aggr = aggr
                best_slot_data = {
                    "name": name, "slots": slots, "m": m,
                    "fill": fill, "aggr": aggr,
                }
                marker = " ★"

            print(f"  {slots:>5} {m['N']:>6} {m['PF']:>5.3f} {m['EV_bp']:>+6.1f} "
                  f"{m['tpd']:>5.2f} {m['ann_ret_pct']:>+9.1f} "
                  f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f} "
                  f"{aggr:>9.0f} {m['losing_years']:>6} "
                  f"{fill:>5.1f}%{marker}")

        if best_slot_data:
            final_results.append(best_slot_data)

    phase3_time = time.time() - t3
    print(f"\n  Phase 3 done in {phase3_time:.1f}s")

    # ═══════════════════════════════════════════════════════
    #  FINAL: Aggression Ranking
    # ═══════════════════════════════════════════════════════
    final_results.sort(key=lambda x: x["aggr"], reverse=True)

    print(f"\n{'='*130}")
    print(f"  FINAL RANKING: AnnRet% × Calmar (Aggression Score)")
    print(f"{'='*130}")
    print(f"  {'#':>2} {'Variant':<22} {'Slots':>5} {'N':>6} {'PF':>5} "
          f"{'AnnRet%':>9} {'DD%':>5} {'Calmar':>7} {'Aggr':>9} {'LoseYr':>6}")
    print(f"  {'─'*90}")

    for rank, fr in enumerate(final_results, 1):
        m = fr["m"]
        tag = ""
        if fr["name"] == "VR5_35_T50_D8":
            tag = " ← A-VR5_35"
        elif fr["name"] == "VR7_50_T50_D8":
            tag = " ← Baseline"
        print(f"  {rank:>2} {fr['name']:<22} {fr['slots']:>5} {m['N']:>6} "
              f"{m['PF']:>5.3f} {m['ann_ret_pct']:>+9.1f} "
              f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f} "
              f"{fr['aggr']:>9.0f} {m['losing_years']:>6}{tag}")

    # Year-by-year for #1
    if final_results:
        winner = final_results[0]
        w_trades = variant_trades[winner["name"]]
        yrs = yearly_equity(w_trades, winner["slots"], n_years)
        print(f"\n  Winner year-by-year ({winner['name']} @{winner['slots']} slots):")
        for y in yrs:
            bar = "+" * max(0, int(y["ret_pct"] / 10)) if y["ret_pct"] > 0 else "-" * max(0, int(-y["ret_pct"] / 10))
            print(f"    Y{y['year']} (20{20+y['year']}): {y['ret_pct']:>+8.1f}%  "
                  f"DD={y['max_dd_pct']:>5.1f}%  N={y['N']:>4}  PF={y['pf']:>5.3f}  {bar}")

    total = time.time() - t0
    print(f"\n  Total R39 completed in {total:.0f}s\n")


if __name__ == "__main__":
    main()
