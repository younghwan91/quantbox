"""Optimal Slot Analysis for R40 Champion (VR8_56_SL1_TP5_D4).

각 슬롯이 equity/N 을 사용하는 현실적 모델(Model B)로
slot 수 × leverage 조합별 AnnRet, MaxDD, Sharpe, Calmar 계산.

핵심 수식:
    - 슬롯당 자본 = equity / N
    - 슬롯당 노셔널 = (equity / N) × leverage
    - 트레이드 PnL이 equity에 미치는 영향 = pnl_pct × leverage / N
    - 복리: eq *= (1 + pnl × leverage / N)

바이낸스 레버리지 한도: 대부분 심볼 20~25x

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/optimal_slots_r40.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from research.active.r31_r40.r31_mtf_vol_straddle import (
    load_15m, build_resampled, precompute,
    pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import precompute_all_vr
from research.active.r31_r40.r40d_tournament import (
    collect_trades_custom,
    portfolio_sim_levered_with_details,
    BARS_PER_YEAR,
)

CFG = {
    "sp": 8, "lp": 56,
    "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
}


def sim_realistic(
    accepted_trades: list[tuple[str, int, int, float]],
    leverage: float,
    max_slots: int,
    data_years: float,
) -> dict:
    """Simulate with realistic 1/N position sizing.

    Each trade uses equity/max_slots capital, levered by `leverage`.
    eq *= (1 + pnl_pct × leverage / max_slots)

    Also tracks:
    - Full equity curve for drawdown
    - Yearly breakdown
    - Concurrency stats
    """
    sorted_trades = sorted(accepted_trades, key=lambda x: x[1])

    # ── Equity curve ──
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    eq_series: list[float] = [1.0]

    # ── Yearly tracking ──
    n_years = int(data_years)
    yearly_start_eq: list[float] = [1.0]  # equity at start of each year
    yearly_n: list[int] = [0] * n_years
    yearly_end_eq: list[float] = [1.0] * n_years
    yearly_peak: list[float] = [1.0] * n_years
    yearly_dd: list[float] = [0.0] * n_years

    # ── Concurrency tracking ──
    active_set: list[tuple[str, int]] = []  # (sym, exit_bar)
    concurrent_at_entry: list[int] = []

    frac = leverage / max_slots

    for sym, entry, exit_b, pnl_val in sorted_trades:
        # Clean expired
        active_set = [(s, e) for s, e in active_set if e > entry]
        concurrent_at_entry.append(len(active_set) + 1)
        active_set.append((sym, exit_b))

        # Apply PnL with fractional sizing
        equity *= (1.0 + pnl_val * frac)

        # Drawdown
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
        eq_series.append(equity)

        # Year tracking
        yr = min(entry // BARS_PER_YEAR, n_years - 1)
        yearly_n[yr] += 1
        yearly_end_eq[yr] = equity

    # Build yearly returns
    yearly_results: list[dict] = []
    running_eq = 1.0
    for yr in range(n_years):
        yr_start = running_eq
        # Replay just this year's trades for DD within year
        yr_trades = [(s, e, x, p) for s, e, x, p in sorted_trades
                     if yr * BARS_PER_YEAR <= e < (yr + 1) * BARS_PER_YEAR]
        yr_eq = 1.0
        yr_peak = 1.0
        yr_dd = 0.0
        for _, _, _, pnl_val in yr_trades:
            yr_eq *= (1.0 + pnl_val * frac)
            if yr_eq > yr_peak:
                yr_peak = yr_eq
            dd = (yr_peak - yr_eq) / yr_peak
            if dd > yr_dd:
                yr_dd = dd

        yr_ret_pct = (yr_eq - 1.0) * 100
        running_eq *= yr_eq
        yearly_results.append({
            "year": yr,
            "N": len(yr_trades),
            "ret_pct": yr_ret_pct,
            "max_dd_pct": yr_dd * 100,
        })

    # ── Per-trade stats (these don't change with sizing model) ──
    pnl_arr = np.array([p * leverage for _, _, _, p in sorted_trades])
    n = len(pnl_arr)
    pf = pf_from_pnl(pnl_arr) if n > 1 else 0.0
    wr = float(np.mean(pnl_arr > 0) * 100) if n > 0 else 0.0
    ev = float(np.mean(pnl_arr) * 10000) if n > 0 else 0.0  # basis points

    # Sharpe (per-trade)
    if n > 1 and np.std(pnl_arr) > 0:
        trades_per_yr = n / data_years
        sharpe = (np.mean(pnl_arr) / np.std(pnl_arr)) * np.sqrt(trades_per_yr)
    else:
        sharpe = 0.0

    # Annualized return from final equity
    ann_ret = (equity ** (1.0 / data_years) - 1.0) * 100 if equity > 0 else -100.0

    # Calmar
    calmar = ann_ret / (max_dd * 100) if max_dd > 0 else 0.0

    # Concurrency
    conc_arr = np.array(concurrent_at_entry) if concurrent_at_entry else np.array([0])
    avg_conc = float(np.mean(conc_arr))
    max_conc = int(np.max(conc_arr))

    losing_years = sum(1 for y in yearly_results if y["ret_pct"] < 0)
    min_yr = min(y["ret_pct"] for y in yearly_results) if yearly_results else 0.0

    return {
        "N": n,
        "PF": pf,
        "WR": wr,
        "EV_bp": ev,
        "sharpe": sharpe,
        "ann_ret_pct": ann_ret,
        "max_dd_pct": max_dd * 100,
        "calmar": calmar,
        "final_eq": equity,
        "avg_concurrent": avg_conc,
        "max_concurrent": max_conc,
        "losing_years": losing_years,
        "min_year_ret": min_yr,
        "yearly": yearly_results,
        "leverage": leverage,
        "max_slots": max_slots,
        "effective_lev": leverage,  # total exposure = leverage (split across N slots)
    }


def main() -> None:
    t0 = time.time()

    print("=" * 120)
    print("  OPTIMAL SLOT ANALYSIS — R40 Champion VR8_56_SL1_TP5_D4")
    print("  Model B: Realistic 1/N equity allocation per slot")
    print("  eq *= (1 + pnl × leverage / max_slots)")
    print("=" * 120)

    # ── Data Load ──
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    n_years = int(data_years)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])

    # ── Collect trades ──
    print(f"\n  Collecting R40 champion trades...")
    d_trades = collect_trades_custom(CFG, data_15m, data_1h, data_4h, "D_mom8")
    s_trades = collect_trades_custom(CFG, data_15m, data_1h, data_4h, "S_d8")
    all_raw = d_trades + s_trades
    print(f"  Raw trades: {len(all_raw)} ({len(d_trades)} directional + {len(s_trades)} straddle)")

    # ═══════════════════════════════════════════════════════
    #  Phase 1: Full Grid — Slots × Leverage
    # ═══════════════════════════════════════════════════════
    slot_range = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    lev_range = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    print(f"\n  Phase 1: {len(slot_range)} × {len(lev_range)} = {len(slot_range)*len(lev_range)} combos")
    print(f"  Slots: {slot_range}")
    print(f"  Leverage: {lev_range}")
    print(f"  Binance max leverage: ~20-25x (hard cap)")

    results: list[dict] = []

    for slots in slot_range:
        for lev in lev_range:
            # Filter trades through portfolio sim (slot limit)
            _, accepted = portfolio_sim_levered_with_details(all_raw, slots, 1.0)
            # Now simulate with realistic 1/N sizing
            m = sim_realistic(accepted, lev, slots, data_years)
            m["config_slots"] = slots
            m["config_lev"] = lev
            m["effective_per_trade"] = lev / slots  # fraction of equity risked
            results.append(m)

    # ═══════════════════════════════════════════════════════
    #  Phase 2: Display Results
    # ═══════════════════════════════════════════════════════

    # -- Table 1: Main grid --
    print(f"\n{'─'*120}")
    print(f"  RESULTS: AnnRet% (Model B: realistic 1/N sizing)")
    print(f"{'─'*120}")
    
    header = f"  {'Slots':>5} |"
    for lev in lev_range:
        header += f" {'L='+str(lev)+'x':>10} |"
    print(header)
    print(f"  {'─'*5} |" + (f"{'─'*10}—|" * len(lev_range)))

    for slots in slot_range:
        row = f"  {slots:>5} |"
        for lev in lev_range:
            r = next(r for r in results if r["config_slots"] == slots and r["config_lev"] == lev)
            ann = r["ann_ret_pct"]
            # Mark if losing years > 1
            marker = " !" if r["losing_years"] > 1 else ""
            row += f" {ann:>+9.1f}{marker}|"
        print(row)

    print(f"\n  (!) = > 1 losing year — 불안정")

    # -- Table 2: MaxDD% grid --
    print(f"\n{'─'*120}")
    print(f"  MaxDD% (negative = larger drawdown)")
    print(f"{'─'*120}")

    header = f"  {'Slots':>5} |"
    for lev in lev_range:
        header += f" {'L='+str(lev)+'x':>10} |"
    print(header)
    print(f"  {'─'*5} |" + (f"{'─'*10}—|" * len(lev_range)))

    for slots in slot_range:
        row = f"  {slots:>5} |"
        for lev in lev_range:
            r = next(r for r in results if r["config_slots"] == slots and r["config_lev"] == lev)
            dd = r["max_dd_pct"]
            row += f" {dd:>9.1f}%|"
        print(row)

    # -- Table 3: Calmar grid --
    print(f"\n{'─'*120}")
    print(f"  Calmar Ratio (AnnRet / MaxDD — higher is better)")
    print(f"{'─'*120}")

    header = f"  {'Slots':>5} |"
    for lev in lev_range:
        header += f" {'L='+str(lev)+'x':>10} |"
    print(header)
    print(f"  {'─'*5} |" + (f"{'─'*10}—|" * len(lev_range)))

    for slots in slot_range:
        row = f"  {slots:>5} |"
        for lev in lev_range:
            r = next(r for r in results if r["config_slots"] == slots and r["config_lev"] == lev)
            cal = r["calmar"]
            row += f" {cal:>10.2f}|"
        print(row)

    # ═══════════════════════════════════════════════════════
    #  Phase 3: Top 20 by Calmar (risk-adjusted) + AnnRet
    # ═══════════════════════════════════════════════════════
    # Filter: ≤ 1 losing year, DD < 95%
    reliable = [r for r in results if r["losing_years"] <= 1 and r["max_dd_pct"] < 95]

    # Top by Calmar
    by_calmar = sorted(reliable, key=lambda r: r["calmar"], reverse=True)
    print(f"\n{'═'*120}")
    print(f"  TOP 20 BY CALMAR (risk-adjusted, ≤1 losing year, DD < 95%)")
    print(f"{'═'*120}")
    print(f"  {'#':>2} {'Slots':>5} {'Lev':>5} {'L/N':>6} {'N':>5} "
          f"{'AnnRet%':>9} {'DD%':>6} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LoseYr':>6} {'MinYr%':>8} {'AvgConc':>7} {'FinalEq':>10}")
    print(f"  {'─'*105}")

    for rank, r in enumerate(by_calmar[:20], 1):
        marker = " ★" if rank <= 3 else ""
        print(f"  {rank:>2} {r['config_slots']:>5} {r['config_lev']:>5.1f} "
              f"{r['effective_per_trade']:>6.2f} {r['N']:>5} "
              f"{r['ann_ret_pct']:>+9.1f} {r['max_dd_pct']:>6.1f} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} "
              f"{r['losing_years']:>6} {r['min_year_ret']:>+8.1f} "
              f"{r['avg_concurrent']:>7.1f} {r['final_eq']:>10.1f}x{marker}")

    # Top by AnnRet
    by_return = sorted(reliable, key=lambda r: r["ann_ret_pct"], reverse=True)
    print(f"\n{'═'*120}")
    print(f"  TOP 20 BY ANNUALIZED RETURN (pure growth, ≤1 losing year)")
    print(f"{'═'*120}")
    print(f"  {'#':>2} {'Slots':>5} {'Lev':>5} {'L/N':>6} {'N':>5} "
          f"{'AnnRet%':>9} {'DD%':>6} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LoseYr':>6} {'MinYr%':>8} {'AvgConc':>7} {'FinalEq':>10}")
    print(f"  {'─'*105}")

    for rank, r in enumerate(by_return[:20], 1):
        binance_ok = "✓" if r["config_lev"] <= 20 else "✗"
        print(f"  {rank:>2} {r['config_slots']:>5} {r['config_lev']:>5.1f} "
              f"{r['effective_per_trade']:>6.2f} {r['N']:>5} "
              f"{r['ann_ret_pct']:>+9.1f} {r['max_dd_pct']:>6.1f} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} "
              f"{r['losing_years']:>6} {r['min_year_ret']:>+8.1f} "
              f"{r['avg_concurrent']:>7.1f} {r['final_eq']:>10.1f}x {binance_ok}")

    # ═══════════════════════════════════════════════════════
    #  Phase 4: Year-by-Year for Top 5 by Calmar
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*120}")
    print(f"  YEAR-BY-YEAR: Top 5 Calmar")
    print(f"{'═'*120}")
    print(f"  {'Slots':>5} {'Lev':>4} {'L/N':>5} |"
          + " | ".join(f"  {'Y'+str(i):>5}" for i in range(n_years))
          + f" | {'AnnRet':>8}")
    print(f"  {'─'*80}")

    for r in by_calmar[:5]:
        yr_str = " | ".join(f"{y['ret_pct']:>+7.1f}" for y in r["yearly"])
        print(f"  {r['config_slots']:>5} {r['config_lev']:>4.1f} "
              f"{r['effective_per_trade']:>5.2f} | {yr_str} | {r['ann_ret_pct']:>+8.1f}")

    # ═══════════════════════════════════════════════════════
    #  Phase 5: Current vs Optimal comparison
    # ═══════════════════════════════════════════════════════
    current = next(r for r in results if r["config_slots"] == 15 and r["config_lev"] == 3.0)
    best_calmar = by_calmar[0]
    best_return = by_return[0]
    # Best within binance limits (≤ 20x)
    binance_reliable = [r for r in reliable if r["config_lev"] <= 20]
    best_binance_cal = max(binance_reliable, key=lambda r: r["calmar"]) if binance_reliable else current
    best_binance_ret = max(binance_reliable, key=lambda r: r["ann_ret_pct"]) if binance_reliable else current

    print(f"\n{'═'*120}")
    print(f"  CURRENT vs OPTIMAL COMPARISON")
    print(f"{'═'*120}")

    def print_row(label: str, r: dict):
        print(f"  {label:<30} Slots={r['config_slots']:>2}  Lev={r['config_lev']:>5.1f}x  "
              f"L/N={r['effective_per_trade']:.2f}  "
              f"AnnRet={r['ann_ret_pct']:>+7.1f}%  DD={r['max_dd_pct']:>5.1f}%  "
              f"Calmar={r['calmar']:.2f}  LoseYr={r['losing_years']}")

    print_row("Current (3x, 15 slots)", current)
    print_row("Best Calmar (any lev)", best_calmar)
    print_row("Best Return (any lev)", best_return)
    print_row("Best Calmar (≤20x)", best_binance_cal)
    print_row("Best Return (≤20x)", best_binance_ret)

    # ═══════════════════════════════════════════════════════
    #  Phase 6: Sensitivity — L/N ratio analysis
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*120}")
    print(f"  L/N RATIO ANALYSIS (leverage / max_slots)")
    print(f"  This is the effective fraction of equity each trade risks.")
    print(f"{'═'*120}")

    # Group by approximate L/N ratio
    ln_buckets: dict[str, list[dict]] = {}
    for r in reliable:
        ratio = r["effective_per_trade"]
        bucket = f"{ratio:.2f}"
        if bucket not in ln_buckets:
            ln_buckets[bucket] = []
        ln_buckets[bucket].append(r)

    print(f"\n  {'L/N':>6} {'Combos':>6} {'Avg AnnRet':>10} {'Avg DD':>8} {'Avg Calmar':>10} {'Best Combo':>25}")
    print(f"  {'─'*75}")
    for ratio_str in sorted(ln_buckets.keys(), key=float):
        bucket = ln_buckets[ratio_str]
        avg_ret = np.mean([r["ann_ret_pct"] for r in bucket])
        avg_dd = np.mean([r["max_dd_pct"] for r in bucket])
        avg_cal = np.mean([r["calmar"] for r in bucket])
        best = max(bucket, key=lambda r: r["calmar"])
        combo = f"S={best['config_slots']},L={best['config_lev']}"
        print(f"  {float(ratio_str):>6.2f} {len(bucket):>6} {avg_ret:>+10.1f} "
              f"{avg_dd:>8.1f} {avg_cal:>10.2f} {combo:>25}")

    # ═══════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*120}")
    print(f"  SUMMARY & RECOMMENDATION")
    print(f"{'═'*120}")
    print(f"\n  Current config: {current['config_slots']} slots, {current['config_lev']}x leverage")
    print(f"    → L/N = {current['effective_per_trade']:.2f}")
    print(f"    → AnnRet = {current['ann_ret_pct']:+.1f}%, MaxDD = {current['max_dd_pct']:.1f}%, "
          f"Calmar = {current['calmar']:.2f}")
    print()
    print(f"  Best Calmar (≤20x): {best_binance_cal['config_slots']} slots, "
          f"{best_binance_cal['config_lev']}x")
    print(f"    → L/N = {best_binance_cal['effective_per_trade']:.2f}")
    print(f"    → AnnRet = {best_binance_cal['ann_ret_pct']:+.1f}%, "
          f"MaxDD = {best_binance_cal['max_dd_pct']:.1f}%, "
          f"Calmar = {best_binance_cal['calmar']:.2f}")
    print()
    print(f"  Best Return (≤20x): {best_binance_ret['config_slots']} slots, "
          f"{best_binance_ret['config_lev']}x")
    print(f"    → L/N = {best_binance_ret['effective_per_trade']:.2f}")
    print(f"    → AnnRet = {best_binance_ret['ann_ret_pct']:+.1f}%, "
          f"MaxDD = {best_binance_ret['max_dd_pct']:.1f}%, "
          f"Calmar = {best_binance_ret['calmar']:.2f}")

    total = time.time() - t0
    print(f"\n  Completed in {total:.1f}s")


if __name__ == "__main__":
    main()
