"""Optimal Slot v2 — Fine Grid + Adaptive Leverage Models.

이전 분석에서 L/N(leverage/slots) 비율이 핵심임을 확인.
이번엔:
  1. 세밀한 그리드: slots 1~25, leverage 1~25 (바이낸스 한도)
  2. 적응형 레버리지 모델:
     - Fixed: L/N 고정 (이전과 동일)
     - Adaptive-Concurrency: 동시 포지션 많으면 L 낮춤
       → 실제: 빈 슬롯이 많으면 더 공격적, 꽉 차면 보수적
     - Adaptive-DD: drawdown 깊으면 L 줄임 (risk parity)
     - Target L/N: 목표 L/N=2.0~3.0을 유지하도록 동적 조절

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/optimal_slots_v2.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from research.active.mtf_vol_straddle_r31 import (
    load_15m, build_resampled, precompute,
    pf_from_pnl,
)
from research.active.scaling_r38 import precompute_all_vr
from research.active.tournament_r40 import (
    collect_trades_custom,
    portfolio_sim_levered_with_details,
    BARS_PER_YEAR,
)

CFG = {
    "sp": 8, "lp": 56,
    "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
}


def sim_fixed(
    trades: list[tuple[str, int, int, float]],
    leverage: float, max_slots: int, data_years: float,
) -> dict:
    """Fixed L/N model."""
    sorted_t = sorted(trades, key=lambda x: x[1])
    frac = leverage / max_slots
    eq = 1.0; peak = 1.0; max_dd = 0.0
    n_years = int(data_years)

    active: list[tuple[str, int]] = []
    conc_list: list[int] = []

    for sym, entry, exit_b, pnl in sorted_t:
        active = [(s, e) for s, e in active if e > entry]
        conc_list.append(len(active) + 1)
        active.append((sym, exit_b))
        eq *= (1.0 + pnl * frac)
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd

    # Yearly
    yearly = []
    for yr in range(n_years):
        yr_t = [p for _, e, _, p in sorted_t if yr * BARS_PER_YEAR <= e < (yr+1) * BARS_PER_YEAR]
        yr_eq = 1.0; yr_peak = 1.0; yr_dd = 0.0
        for p in yr_t:
            yr_eq *= (1.0 + p * frac)
            if yr_eq > yr_peak: yr_peak = yr_eq
            dd = (yr_peak - yr_eq) / yr_peak
            if dd > yr_dd: yr_dd = dd
        yearly.append({"year": yr, "N": len(yr_t), "ret_pct": (yr_eq-1)*100, "max_dd_pct": yr_dd*100})

    return _build_result(sorted_t, eq, max_dd, yearly, data_years, leverage, max_slots, conc_list, "fixed")


def sim_adaptive_concurrency(
    trades: list[tuple[str, int, int, float]],
    base_leverage: float, max_slots: int, data_years: float,
    target_ln: float,
) -> dict:
    """Adaptive: maintain target L/N ratio based on actual concurrent positions.

    When few positions are open → increase per-trade leverage (up to base_leverage).
    When many positions → decrease per-trade leverage.

    effective_lev_per_trade = target_ln  (capped at base_leverage)
    So: eq *= (1 + pnl × min(target_ln, base_leverage))

    But total leverage = effective_lev × concurrent_count ≤ base_leverage × max_slots / max_slots
    Actually: each trade allocates equity/max_slots, but levs up to target_ln × max_slots.
    
    Simpler model:
    - Budget = equity × base_leverage (total allowed notional)
    - Each trade: notional = budget / max(concurrent, 1)
    - pnl_impact = pnl × notional / equity = pnl × base_leverage / max(concurrent, 1)
    """
    sorted_t = sorted(trades, key=lambda x: x[1])
    eq = 1.0; peak = 1.0; max_dd = 0.0
    n_years = int(data_years)

    active: list[tuple[str, int]] = []
    conc_list: list[int] = []
    effective_fracs: list[float] = []

    for sym, entry, exit_b, pnl in sorted_t:
        active = [(s, e) for s, e in active if e > entry]
        n_conc = len(active) + 1
        conc_list.append(n_conc)
        active.append((sym, exit_b))

        # Adaptive: spread base_leverage across concurrent positions
        eff_frac = base_leverage / n_conc
        # Cap: each trade can't use more than base_leverage (1 slot gets full lev)
        eff_frac = min(eff_frac, base_leverage)
        effective_fracs.append(eff_frac)

        eq *= (1.0 + pnl * eff_frac)
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd

    # Yearly (approximate with fixed frac for simplicity)
    avg_frac = np.mean(effective_fracs) if effective_fracs else 1.0
    yearly = []
    for yr in range(n_years):
        yr_t = [(s, e, x, p) for s, e, x, p in sorted_t if yr * BARS_PER_YEAR <= e < (yr+1) * BARS_PER_YEAR]
        yr_eq = 1.0; yr_peak = 1.0; yr_dd = 0.0
        yr_active: list[tuple[str, int]] = []
        for sym, entry, exit_b, pnl in yr_t:
            yr_active = [(s, e) for s, e in yr_active if e > entry]
            n_conc = len(yr_active) + 1
            yr_active.append((sym, exit_b))
            frac = min(base_leverage / n_conc, base_leverage)
            yr_eq *= (1.0 + pnl * frac)
            if yr_eq > yr_peak: yr_peak = yr_eq
            dd = (yr_peak - yr_eq) / yr_peak
            if dd > yr_dd: yr_dd = dd
        yearly.append({"year": yr, "N": len(yr_t), "ret_pct": (yr_eq-1)*100, "max_dd_pct": yr_dd*100})

    return _build_result(sorted_t, eq, max_dd, yearly, data_years, base_leverage, max_slots, conc_list,
                         "adaptive_conc", extra={"avg_eff_frac": float(np.mean(effective_fracs)) if effective_fracs else 0})


def sim_adaptive_dd(
    trades: list[tuple[str, int, int, float]],
    base_leverage: float, max_slots: int, data_years: float,
    dd_threshold: float = 0.20,
    dd_reduction: float = 0.5,
) -> dict:
    """Adaptive-DD: reduce leverage when in drawdown.

    If current_dd > dd_threshold:
        effective_frac = base_frac × dd_reduction
    Else:
        effective_frac = base_frac
    """
    sorted_t = sorted(trades, key=lambda x: x[1])
    base_frac = base_leverage / max_slots
    eq = 1.0; peak = 1.0; max_dd = 0.0
    n_years = int(data_years)

    active: list[tuple[str, int]] = []
    conc_list: list[int] = []
    dd_reductions = 0

    for sym, entry, exit_b, pnl in sorted_t:
        active = [(s, e) for s, e in active if e > entry]
        conc_list.append(len(active) + 1)
        active.append((sym, exit_b))

        # Check current drawdown
        current_dd = (peak - eq) / peak if peak > 0 else 0
        if current_dd > dd_threshold:
            frac = base_frac * dd_reduction
            dd_reductions += 1
        else:
            frac = base_frac

        eq *= (1.0 + pnl * frac)
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd

    # Yearly
    yearly = []
    for yr in range(n_years):
        yr_t = [p for _, e, _, p in sorted_t if yr * BARS_PER_YEAR <= e < (yr+1) * BARS_PER_YEAR]
        yr_eq = 1.0; yr_peak = 1.0; yr_dd = 0.0
        for p in yr_t:
            yr_eq *= (1.0 + p * base_frac)  # approximate
            if yr_eq > yr_peak: yr_peak = yr_eq
            dd = (yr_peak - yr_eq) / yr_peak
            if dd > yr_dd: yr_dd = dd
        yearly.append({"year": yr, "N": len(yr_t), "ret_pct": (yr_eq-1)*100, "max_dd_pct": yr_dd*100})

    return _build_result(sorted_t, eq, max_dd, yearly, data_years, base_leverage, max_slots, conc_list,
                         "adaptive_dd", extra={"dd_reductions": dd_reductions, "pct_reduced": dd_reductions/len(sorted_t)*100 if sorted_t else 0})


def sim_target_ln(
    trades: list[tuple[str, int, int, float]],
    base_leverage: float, max_slots: int, data_years: float,
    target_ln: float,
) -> dict:
    """Target L/N: adjust effective leverage to maintain target L/N regardless of slot count.

    effective_frac = target_ln (constant per trade)
    This means: regardless of max_slots, each trade risks target_ln% of equity.
    Total risk = target_ln × concurrent_count.

    Cap at base_leverage to respect exchange limits.
    """
    sorted_t = sorted(trades, key=lambda x: x[1])
    eq = 1.0; peak = 1.0; max_dd = 0.0
    n_years = int(data_years)

    active: list[tuple[str, int]] = []
    conc_list: list[int] = []
    frac = min(target_ln, base_leverage)

    for sym, entry, exit_b, pnl in sorted_t:
        active = [(s, e) for s, e in active if e > entry]
        conc_list.append(len(active) + 1)
        active.append((sym, exit_b))

        eq *= (1.0 + pnl * frac)
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd

    yearly = []
    for yr in range(n_years):
        yr_t = [p for _, e, _, p in sorted_t if yr * BARS_PER_YEAR <= e < (yr+1) * BARS_PER_YEAR]
        yr_eq = 1.0; yr_peak = 1.0; yr_dd = 0.0
        for p in yr_t:
            yr_eq *= (1.0 + p * frac)
            if yr_eq > yr_peak: yr_peak = yr_eq
            dd = (yr_peak - yr_eq) / yr_peak
            if dd > yr_dd: yr_dd = dd
        yearly.append({"year": yr, "N": len(yr_t), "ret_pct": (yr_eq-1)*100, "max_dd_pct": yr_dd*100})

    return _build_result(sorted_t, eq, max_dd, yearly, data_years, base_leverage, max_slots, conc_list,
                         f"target_ln={target_ln}")


def _build_result(
    sorted_t: list, eq: float, max_dd: float,
    yearly: list[dict], data_years: float,
    leverage: float, max_slots: int,
    conc_list: list[int], model: str,
    extra: dict | None = None,
) -> dict:
    n = len(sorted_t)
    pnl_arr = np.array([p * leverage for _, _, _, p in sorted_t]) if n > 0 else np.empty(0)
    pf = pf_from_pnl(pnl_arr) if n > 1 else 0.0
    wr = float(np.mean(pnl_arr > 0) * 100) if n > 0 else 0.0
    ev = float(np.mean(pnl_arr) * 10000) if n > 0 else 0.0

    if n > 1 and np.std(pnl_arr) > 0:
        sharpe = (np.mean(pnl_arr) / np.std(pnl_arr)) * np.sqrt(n / data_years)
    else:
        sharpe = 0.0

    ann_ret = (eq ** (1.0 / data_years) - 1.0) * 100 if eq > 0 else -100.0
    calmar = ann_ret / (max_dd * 100) if max_dd > 0 else 0.0

    conc_arr = np.array(conc_list) if conc_list else np.array([0])
    losing_yrs = sum(1 for y in yearly if y["ret_pct"] < 0)
    min_yr = min(y["ret_pct"] for y in yearly) if yearly else 0.0

    result = {
        "N": n, "PF": pf, "WR": wr, "EV_bp": ev, "sharpe": sharpe,
        "ann_ret_pct": ann_ret, "max_dd_pct": max_dd * 100,
        "calmar": calmar, "final_eq": eq,
        "avg_concurrent": float(np.mean(conc_arr)),
        "max_concurrent": int(np.max(conc_arr)),
        "losing_years": losing_yrs, "min_year_ret": min_yr,
        "yearly": yearly, "leverage": leverage, "max_slots": max_slots,
        "ln_ratio": leverage / max_slots,
        "model": model,
    }
    if extra:
        result.update(extra)
    return result


def main() -> None:
    t0 = time.time()

    print("=" * 130)
    print("  OPTIMAL SLOT v2 — Fine Grid + Adaptive Leverage Models")
    print("  R40 Champion VR8_56_SL1_TP5_D4")
    print("=" * 130)

    # ── Data Load ──
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    n_years = int(data_years)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])

    # ── Collect trades ──
    d_trades = collect_trades_custom(CFG, data_15m, data_1h, data_4h, "D_mom8")
    s_trades = collect_trades_custom(CFG, data_15m, data_1h, data_4h, "S_d8")
    all_raw = d_trades + s_trades
    print(f"\n  Raw trades: {len(all_raw)} ({len(d_trades)} D + {len(s_trades)} S)")

    # ═══════════════════════════════════════════════════════
    #  Section 1: Fine Grid (Fixed L/N)
    # ═══════════════════════════════════════════════════════
    slot_range = list(range(1, 26))  # 1~25
    lev_range = list(range(1, 26))   # 1~25x

    print(f"\n  Section 1: Fine Grid — {len(slot_range)}×{len(lev_range)} = {len(slot_range)*len(lev_range)} combos")
    t1 = time.time()

    fixed_results: list[dict] = []
    for slots in slot_range:
        for lev in lev_range:
            _, accepted = portfolio_sim_levered_with_details(all_raw, slots, 1.0)
            m = sim_fixed(accepted, float(lev), slots, data_years)
            m["config_slots"] = slots
            m["config_lev"] = lev
            fixed_results.append(m)

    print(f"  Fixed grid done in {time.time()-t1:.1f}s")

    # Filter reliable
    reliable = [r for r in fixed_results if r["losing_years"] <= 1 and r["max_dd_pct"] < 95]
    by_calmar = sorted(reliable, key=lambda r: r["calmar"], reverse=True)
    by_return = sorted(reliable, key=lambda r: r["ann_ret_pct"], reverse=True)

    print(f"\n  {len(reliable)} / {len(fixed_results)} reliable (≤1 losing yr, DD<95%)")

    # Top 30 by Calmar
    print(f"\n{'═'*130}")
    print(f"  TOP 30 BY CALMAR — Fixed L/N (≤25x leverage, Binance compatible)")
    print(f"{'═'*130}")
    print(f"  {'#':>2} {'Slots':>5} {'Lev':>4} {'L/N':>5} {'N':>5} "
          f"{'AnnRet%':>9} {'DD%':>6} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LoseYr':>6} {'MinYr%':>8} {'AvgConc':>7}")
    print(f"  {'─'*95}")

    for rank, r in enumerate(by_calmar[:30], 1):
        marker = " ★" if rank <= 3 else ""
        print(f"  {rank:>2} {r['config_slots']:>5} {r['config_lev']:>4} "
              f"{r['ln_ratio']:>5.2f} {r['N']:>5} "
              f"{r['ann_ret_pct']:>+9.1f} {r['max_dd_pct']:>6.1f} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} "
              f"{r['losing_years']:>6} {r['min_year_ret']:>+8.1f} "
              f"{r['avg_concurrent']:>7.1f}{marker}")

    # Top 30 by Return
    print(f"\n{'═'*130}")
    print(f"  TOP 30 BY ANNUALIZED RETURN (≤1 losing yr)")
    print(f"{'═'*130}")
    print(f"  {'#':>2} {'Slots':>5} {'Lev':>4} {'L/N':>5} {'N':>5} "
          f"{'AnnRet%':>9} {'DD%':>6} {'Calmar':>7} {'LoseYr':>6} {'MinYr%':>8}")
    print(f"  {'─'*80}")

    for rank, r in enumerate(by_return[:30], 1):
        print(f"  {rank:>2} {r['config_slots']:>5} {r['config_lev']:>4} "
              f"{r['ln_ratio']:>5.2f} {r['N']:>5} "
              f"{r['ann_ret_pct']:>+9.1f} {r['max_dd_pct']:>6.1f} "
              f"{r['calmar']:>7.2f} {r['losing_years']:>6} {r['min_year_ret']:>+8.1f}")

    # ═══════════════════════════════════════════════════════
    #  Section 2: Adaptive Leverage Models
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*130}")
    print(f"  Section 2: ADAPTIVE LEVERAGE MODELS")
    print(f"  Compare 4 models at various slot/leverage combos")
    print(f"{'═'*130}")

    # Test configs: promising slot/lev pairs from Section 1
    test_configs = [
        (3, 10), (4, 10), (5, 10), (5, 15), (5, 20),
        (6, 15), (6, 20), (7, 15), (7, 20), (7, 25),
        (8, 20), (8, 25), (10, 15), (10, 20), (10, 25),
        (15, 20), (15, 25),
        # Current
        (15, 3),
    ]

    print(f"\n  {'Config':<12} {'Model':<20} {'N':>5} {'AnnRet%':>9} {'DD%':>6} "
          f"{'Calmar':>7} {'LoseYr':>6} {'MinYr%':>8} {'AvgConc':>7}")
    print(f"  {'─'*90}")

    adaptive_all: list[dict] = []

    for slots, lev in test_configs:
        _, accepted = portfolio_sim_levered_with_details(all_raw, slots, 1.0)
        label = f"S={slots},L={lev}"

        # Model 1: Fixed
        m1 = sim_fixed(accepted, float(lev), slots, data_years)
        m1["label"] = label; m1["config_slots"] = slots; m1["config_lev"] = lev

        # Model 2: Adaptive Concurrency (spread total lev across concurrent)
        m2 = sim_adaptive_concurrency(accepted, float(lev), slots, data_years, target_ln=lev/slots)
        m2["label"] = label; m2["config_slots"] = slots; m2["config_lev"] = lev

        # Model 3: Adaptive DD (reduce when in >20% drawdown)
        m3 = sim_adaptive_dd(accepted, float(lev), slots, data_years, dd_threshold=0.20, dd_reduction=0.5)
        m3["label"] = label; m3["config_slots"] = slots; m3["config_lev"] = lev

        # Model 4: Target L/N=2.0 (optimal range from Section 1)
        m4 = sim_target_ln(accepted, float(lev), slots, data_years, target_ln=2.0)
        m4["label"] = label; m4["config_slots"] = slots; m4["config_lev"] = lev

        for m, name in [(m1, "Fixed L/N"), (m2, "Adapt-Concurrency"),
                         (m3, "Adapt-DD(20%,50%)"), (m4, "Target L/N=2.0")]:
            m["model_name"] = name
            adaptive_all.append(m)
            lose_str = str(m["losing_years"])
            if m["losing_years"] > 1:
                lose_str += " !"
            print(f"  {label:<12} {name:<20} {m['N']:>5} "
                  f"{m['ann_ret_pct']:>+9.1f} {m['max_dd_pct']:>6.1f} "
                  f"{m['calmar']:>7.2f} {lose_str:>6} {m['min_year_ret']:>+8.1f} "
                  f"{m['avg_concurrent']:>7.1f}")
        print()

    # ═══════════════════════════════════════════════════════
    #  Section 3: Best of Each Model
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*130}")
    print(f"  Section 3: BEST RESULT PER MODEL (by Calmar, ≤1 losing year)")
    print(f"{'═'*130}")

    for model_name in ["Fixed L/N", "Adapt-Concurrency", "Adapt-DD(20%,50%)", "Target L/N=2.0"]:
        candidates = [m for m in adaptive_all
                      if m["model_name"] == model_name and m["losing_years"] <= 1]
        if not candidates:
            print(f"  {model_name:<20} — no reliable configs")
            continue
        best = max(candidates, key=lambda m: m["calmar"])
        print(f"  {model_name:<20} S={best['config_slots']:>2} L={best['config_lev']:>2}x  "
              f"AnnRet={best['ann_ret_pct']:>+8.1f}%  DD={best['max_dd_pct']:>5.1f}%  "
              f"Calmar={best['calmar']:.2f}  LoseYr={best['losing_years']}  "
              f"MinYr={best['min_year_ret']:>+.1f}%")

    # ═══════════════════════════════════════════════════════
    #  Section 4: Target L/N sweep
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*130}")
    print(f"  Section 4: TARGET L/N SWEEP — What's the optimal L/N ratio?")
    print(f"  Uses all accepted trades (max_slots=25 for max trade count)")
    print(f"{'═'*130}")

    _, all_accepted = portfolio_sim_levered_with_details(all_raw, 25, 1.0)
    ln_targets = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                  3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]

    print(f"\n  {'L/N':>6} {'AnnRet%':>9} {'DD%':>6} {'Calmar':>7} {'LoseYr':>6} {'MinYr%':>8} {'FinalEq':>12}")
    print(f"  {'─'*60}")

    best_ln_calmar = 0.0
    best_ln_val = 0.0
    ln_sweep: list[dict] = []

    for ln in ln_targets:
        m = sim_target_ln(all_accepted, 25.0, 25, data_years, target_ln=ln)
        ln_sweep.append(m)
        lose_marker = " !" if m["losing_years"] > 1 else ""
        print(f"  {ln:>6.2f} {m['ann_ret_pct']:>+9.1f} {m['max_dd_pct']:>6.1f} "
              f"{m['calmar']:>7.2f} {m['losing_years']:>6}{lose_marker} "
              f"{m['min_year_ret']:>+8.1f} {m['final_eq']:>12.1f}x")
        if m["losing_years"] <= 1 and m["calmar"] > best_ln_calmar:
            best_ln_calmar = m["calmar"]
            best_ln_val = ln

    print(f"\n  ★ Peak Calmar at L/N = {best_ln_val:.2f} (Calmar = {best_ln_calmar:.2f})")

    # ═══════════════════════════════════════════════════════
    #  Section 5: Year-by-Year for top configs
    # ═══════════════════════════════════════════════════════
    # Pick top 3 by Calmar from fixed grid + best adaptive
    top_fixed = by_calmar[:3]
    best_adaptive = [m for m in adaptive_all if m["losing_years"] <= 1]
    if best_adaptive:
        best_adapt_calmar = max(best_adaptive, key=lambda m: m["calmar"])
    else:
        best_adapt_calmar = None

    print(f"\n{'═'*130}")
    print(f"  Section 5: YEAR-BY-YEAR COMPARISON")
    print(f"{'═'*130}")
    print(f"  {'Label':<35} |" + " | ".join(f" {'Y'+str(i):>7}" for i in range(n_years)) + f" | {'AnnRet':>8} | {'DD%':>5}")
    print(f"  {'─'*120}")

    for r in top_fixed:
        label = f"Fixed S={r['config_slots']} L={r['config_lev']}x (L/N={r['ln_ratio']:.2f})"
        yr_str = " | ".join(f"{y['ret_pct']:>+7.1f}" for y in r["yearly"])
        print(f"  {label:<35} | {yr_str} | {r['ann_ret_pct']:>+8.1f} | {r['max_dd_pct']:>5.1f}")

    if best_adapt_calmar:
        label = f"{best_adapt_calmar['model_name']} S={best_adapt_calmar['config_slots']} L={best_adapt_calmar['config_lev']}x"
        yr_str = " | ".join(f"{y['ret_pct']:>+7.1f}" for y in best_adapt_calmar["yearly"])
        print(f"  {label:<35} | {yr_str} | {best_adapt_calmar['ann_ret_pct']:>+8.1f} | {best_adapt_calmar['max_dd_pct']:>5.1f}")

    # Current (15/3)
    current = next((r for r in fixed_results if r["config_slots"] == 15 and r["config_lev"] == 3), None)
    if current:
        label = f"Current S=15 L=3x (L/N=0.20)"
        yr_str = " | ".join(f"{y['ret_pct']:>+7.1f}" for y in current["yearly"])
        print(f"  {label:<35} | {yr_str} | {current['ann_ret_pct']:>+8.1f} | {current['max_dd_pct']:>5.1f}")

    # ═══════════════════════════════════════════════════════
    #  Section 6: Practical Recommendations
    # ═══════════════════════════════════════════════════════
    print(f"\n{'═'*130}")
    print(f"  Section 6: PRACTICAL RECOMMENDATIONS")
    print(f"{'═'*130}")

    # Find best by calmar within binance-friendly DD ranges
    tiers = [
        ("Conservative (DD<30%)", 30),
        ("Moderate (DD<50%)", 50),
        ("Aggressive (DD<70%)", 70),
        ("YOLO (DD<90%)", 90),
    ]

    for tier_name, dd_cap in tiers:
        tier_cands = [r for r in reliable if r["max_dd_pct"] < dd_cap and r["config_lev"] <= 25]
        if not tier_cands:
            print(f"\n  {tier_name}: No reliable config found")
            continue
        best_cal = max(tier_cands, key=lambda r: r["calmar"])
        best_ret = max(tier_cands, key=lambda r: r["ann_ret_pct"])
        print(f"\n  {tier_name}:")
        print(f"    Best Calmar: S={best_cal['config_slots']:>2}  L={best_cal['config_lev']:>2}x  "
              f"L/N={best_cal['ln_ratio']:.2f}  AnnRet={best_cal['ann_ret_pct']:>+.1f}%  "
              f"DD={best_cal['max_dd_pct']:.1f}%  Calmar={best_cal['calmar']:.2f}")
        print(f"    Best Return: S={best_ret['config_slots']:>2}  L={best_ret['config_lev']:>2}x  "
              f"L/N={best_ret['ln_ratio']:.2f}  AnnRet={best_ret['ann_ret_pct']:>+.1f}%  "
              f"DD={best_ret['max_dd_pct']:.1f}%  Calmar={best_ret['calmar']:.2f}")

    total = time.time() - t0
    print(f"\n  Completed in {total:.1f}s\n")


if __name__ == "__main__":
    main()
