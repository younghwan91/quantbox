"""R40 — Tournament-Grade Aggression (Binance Futures Competition Level).

바이낸스 선물 대회급 공격성. DD 무시, 신뢰성(매년 흑자)만 유지.
Leverage × Slots × SL/TP 3축 극대화.

핵심 인사이트:
  - 현재 PnL은 1x 레버리지 기준 (raw %)
  - PnL × L = Lx 레버리지 시뮬레이션 (비용도 비례 스케일)
  - 레버리지가 올라가면 compound 효과가 폭발적
  - 다만 매년 흑자(신뢰성) 유지가 대전제

Phase 1: ATR × SL/TP grid (unlimited, 1x) → raw signal power 탐색
Phase 2: Top 10 → Leverage × Slots 매트릭스 (3x ~ 20x × 10~30 slots)
Phase 3: Year-by-year reliability check → 최종 champion

Usage:
    PYTHONUNBUFFERED=1 nohup uv run python research/active/tournament_r40.py &
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


# ══════════════════════════════════════════════════════════════════════════════
#  Leveraged Portfolio Simulation
# ══════════════════════════════════════════════════════════════════════════════


def portfolio_sim_levered(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    leverage: float = 1.0,
) -> np.ndarray:
    """Portfolio sim with leverage multiplier on PnL.

    PnL × leverage correctly models leveraged trading:
      raw_move - cost → (raw_move - cost) × L = L×raw_move - L×cost

    Note: Does NOT enforce liquidation. With SL=1% and L=20x,
    max single-trade loss = 20% — no liquidation risk.
    """
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted_pnl: list[float] = []
    active: list[tuple[str, int]] = []

    for sym, entry, exit_b, pnl_val in trades:
        active = [(s, e) for s, e in active if e > entry]
        if any(s == sym for s, _ in active):
            continue
        if len(active) >= max_concurrent:
            continue
        active.append((sym, exit_b))
        accepted_pnl.append(pnl_val * leverage)

    return np.array(accepted_pnl) if accepted_pnl else np.empty(0)


def portfolio_sim_levered_with_details(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    leverage: float = 1.0,
) -> tuple[np.ndarray, list[tuple[str, int, int, float]]]:
    """Portfolio sim with leverage — returns PnL array + accepted trades."""
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted_pnl: list[float] = []
    accepted: list[tuple[str, int, int, float]] = []
    active: list[tuple[str, int]] = []

    for sym, entry, exit_b, pnl_val in trades:
        active = [(s, e) for s, e in active if e > entry]
        if any(s == sym for s, _ in active):
            continue
        if len(active) >= max_concurrent:
            continue
        active.append((sym, exit_b))
        accepted_pnl.append(pnl_val * leverage)
        accepted.append((sym, entry, exit_b, pnl_val * leverage))

    pnl = np.array(accepted_pnl) if accepted_pnl else np.empty(0)
    return pnl, accepted


def yearly_equity_levered(
    trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    leverage: float = 1.0,
    n_years: int = 6,
) -> list[dict]:
    """Year-by-year metrics with leverage."""
    _, accepted = portfolio_sim_levered_with_details(
        trades, max_concurrent, leverage,
    )

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
        pf = sum(wins) / (sum(losses) if losses else 1e-12)

        yearly.append({
            "year": yr, "N": len(yr_pnl),
            "ret_pct": (eq - 1.0) * 100,
            "eq_mult": eq,
            "max_dd_pct": max_dd * 100,
            "pf": pf,
        })
    return yearly


def levered_metrics(
    trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
    leverage: float,
    data_years: float,
) -> dict:
    """Full metrics with leverage, including yearly breakdown."""
    pnl = portfolio_sim_levered(trades, max_concurrent, leverage)
    m = compute_metrics(pnl, data_years)
    yr = yearly_equity_levered(
        trades, max_concurrent, leverage, n_years=int(data_years),
    )
    m["yearly"] = yr
    m["losing_years"] = sum(1 for y in yr if y["ret_pct"] < 0)
    m["min_year_ret"] = min(y["ret_pct"] for y in yr) if yr else 0.0
    m["leverage"] = leverage
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  Custom SL/TP Trade Collection
# ══════════════════════════════════════════════════════════════════════════════


def collect_trades_custom(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
) -> list[tuple[str, int, int, float]]:
    """Collect trades with custom SL/TP (from cfg)."""
    sp, lp = cfg["sp"], cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    sl = cfg["s_sl"] if is_straddle else cfg["d_sl"]
    tp = cfg["tp"]
    d_vr, s_vr, s_dur = cfg["d_vr"], cfg["s_vr"], cfg["s_dur"]

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_entries: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            mh = mh_s if is_straddle else mh_d

            if is_straddle:
                sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                cost = cfg.get("s_cost", COST_STRADDLE)
                p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, cost)
            else:
                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                cost = cfg.get("d_cost", COST_DIRECTIONAL)
                p, e, x = sim_directional_trades(
                    c, h, lo, sig, dire, sl, tp, mh, cost,
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_entries.append((entry_15m, exit_15m, float(p[k])))

        # Dedup: sort by entry, keep non-overlapping
        sym_entries.sort()
        last_exit = -1
        for entry, exit_b, pnl_val in sym_entries:
            if entry > last_exit:
                all_trades.append((sym, entry, exit_b, pnl_val))
                last_exit = exit_b

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()

    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    n_years = int(data_years)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)

    # ATR pairs: R39 top 5 + new aggressive candidates
    all_pairs = [
        # R39 winners
        (8, 56), (8, 50), (5, 40), (4, 40), (5, 35),
        # New aggressive — short ATR for faster signals
        (3, 20), (4, 28), (6, 35), (6, 42),
        # Ultra-wide for deep squeeze
        (10, 70), (12, 80),
        # Legacy
        (7, 50),
    ]
    precompute_all_vr([data_15m, data_1h, data_4h], all_pairs)

    # SL/TP combos to explore
    sltp_combos = [
        # (d_sl, s_sl, tp, label)
        (0.010, 0.015, 0.050, "SL1_TP5"),    # Current production
        (0.007, 0.010, 0.070, "SL07_TP7"),   # Tighter SL, wider TP
        (0.005, 0.008, 0.100, "SL05_TP10"),  # Extreme asymmetric
        (0.012, 0.018, 0.040, "SL12_TP4"),   # Wider SL, closer TP → higher WR
        (0.008, 0.012, 0.060, "SL08_TP6"),   # Moderate improvement
    ]

    # ═══════════════════════════════════════════════════════
    #  Phase 1: ATR × SL/TP grid (unlimited, 1x leverage)
    # ═══════════════════════════════════════════════════════
    print("=" * 140)
    print("  R40: TOURNAMENT-GRADE AGGRESSION — Binance Futures Competition Level")
    print("  DD IGNORED. Pure compound return. Reliability (≤1 losing year) filter only.")
    print("=" * 140)

    configs: list[tuple[str, dict]] = []
    for sp, lp in all_pairs:
        for d_sl, s_sl, tp, sltp_label in sltp_combos:
            for s_dur in [2, 4]:
                name = f"VR{sp}_{lp}_{sltp_label}_D{s_dur}"
                configs.append((name, {
                    "sp": sp, "lp": lp,
                    "d_vr": 0.50, "s_vr": 0.55,
                    "s_dur": s_dur,
                    "d_sl": d_sl, "s_sl": s_sl, "tp": tp,
                }))

    print(f"\n  Phase 1: {len(configs)} variants (12 ATR × 5 SL/TP × 2 DUR)")
    print(f"  Collecting trades...")
    t1 = time.time()

    variant_trades: dict[str, list] = {}
    results_p1: list[dict] = []

    for idx, (name, cfg) in enumerate(configs):
        if (idx + 1) % 20 == 0:
            print(f"    ... {idx+1}/{len(configs)} ({time.time()-t1:.0f}s)")

        d_t = collect_trades_custom(cfg, data_15m, data_1h, data_4h, "D_mom8")
        s_t = collect_trades_custom(cfg, data_15m, data_1h, data_4h, "S_d8")
        trades = d_t + s_t
        variant_trades[name] = trades

        # Unlimited 1x metrics
        pnl = portfolio_sim(trades, 9999)
        m = compute_metrics(pnl, data_years)
        yr = yearly_equity_levered(trades, 9999, 1.0, n_years)
        m["yearly"] = yr
        m["losing_years"] = sum(1 for y in yr if y["ret_pct"] < 0)
        m["min_year_ret"] = min(y["ret_pct"] for y in yr) if yr else 0.0

        results_p1.append({"name": name, "cfg": cfg, "m": m})

    p1_time = time.time() - t1
    print(f"  Phase 1 done in {p1_time:.0f}s")

    # Filter: PF > 1.0, N > 300, ≤ 1 losing year
    valid = [r for r in results_p1
             if r["m"]["PF"] > 1.0
             and r["m"]["N"] > 300
             and r["m"]["losing_years"] <= 1]
    valid.sort(key=lambda r: r["m"]["ann_ret_pct"], reverse=True)

    print(f"\n  {len(valid)} / {len(results_p1)} passed filter (PF>1.0, N>300, ≤1 losing year)")
    print(f"\n  Top 30 by Compound Return (1x, unlimited):")
    print(f"  {'#':>2} {'Variant':<30} {'N':>5} {'PF':>5} {'EV':>6} "
          f"{'WR':>4} {'R:R':>4} {'AnnRet%':>9} {'DD%':>5} {'LoseYr':>6}")
    print(f"  " + "─" * 100)

    for rank, r in enumerate(valid[:30], 1):
        m = r["m"]
        print(f"  {rank:>2} {r['name']:<30} {m['N']:>5} {m['PF']:>5.3f} "
              f"{m['EV_bp']:>+6.1f} {m['WR']:>4.1f} {m['RR']:>4.2f} "
              f"{m['ann_ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
              f"{m['losing_years']:>6}")

    # ═══════════════════════════════════════════════════════
    #  Phase 2: Leverage × Slots matrix for top 15
    # ═══════════════════════════════════════════════════════
    leverages = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    slot_options = [5, 10, 15, 20, 30]

    top_n = min(15, len(valid))
    phase2_targets = valid[:top_n]

    print(f"\n{'='*140}")
    print(f"  Phase 2: Leverage × Slots Matrix — Top {top_n} variants")
    print(f"  Leverage: {leverages}")
    print(f"  Slots: {slot_options}")
    print(f"  Filter: ≤ 1 losing year")
    print(f"  Rank: Pure compound return (6yr)")
    print(f"{'='*140}")

    t2 = time.time()
    phase2_results: list[dict] = []

    for r in phase2_targets:
        name = r["name"]
        trades = variant_trades[name]

        print(f"\n  ── {name} ({len(trades)} raw trades) ──")
        print(f"  {'Lev':>4} {'Slots':>5} {'N':>5} {'PF':>5} "
              f"{'AnnRet%':>9} {'DD%':>5} {'LoseYr':>6} {'MinYr%':>8} {'WR':>4}")
        print(f"  " + "─" * 70)

        for lev in leverages:
            for slots in slot_options:
                m = levered_metrics(trades, slots, lev, data_years)
                losing = m["losing_years"]
                min_yr = m["min_year_ret"]

                # Only show reliable combos (≤ 1 losing year)
                if losing <= 1:
                    print(f"  {lev:>4.1f} {slots:>5} {m['N']:>5} {m['PF']:>5.3f} "
                          f"{m['ann_ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
                          f"{losing:>6} {min_yr:>+8.1f} {m['WR']:>4.1f}")

                    phase2_results.append({
                        "name": name,
                        "lev": lev, "slots": slots,
                        "m": m,
                        "cfg": r["cfg"],
                    })

    p2_time = time.time() - t2
    print(f"\n  Phase 2 done in {p2_time:.0f}s ({len(phase2_results)} reliable combos)")

    # ═══════════════════════════════════════════════════════
    #  Phase 3: Final ranking + Year-by-Year for top 20
    # ═══════════════════════════════════════════════════════
    phase2_results.sort(key=lambda x: x["m"]["ann_ret_pct"], reverse=True)

    print(f"\n{'='*140}")
    print(f"  Phase 3: FINAL RANKING — Pure Compound Return (DD ignored)")
    print(f"  Filter: ≤ 1 losing year retained")
    print(f"{'='*140}")

    print(f"\n  {'#':>2} {'Variant':<30} {'Lev':>4} {'Slots':>5} {'N':>5} "
          f"{'AnnRet%':>10} {'DD%':>6} {'MinYr%':>8} {'PF':>5} {'WR':>4} "
          f"{'LoseYr':>6}")
    print(f"  " + "─" * 110)

    for rank, pr in enumerate(phase2_results[:30], 1):
        m = pr["m"]
        marker = " ★" if rank <= 3 else ""
        print(f"  {rank:>2} {pr['name']:<30} {pr['lev']:>4.1f} {pr['slots']:>5} "
              f"{m['N']:>5} {m['ann_ret_pct']:>+10.1f} {m['max_dd_pct']:>6.1f} "
              f"{m['min_year_ret']:>+8.1f} {m['PF']:>5.3f} {m['WR']:>4.1f} "
              f"{m['losing_years']:>6}{marker}")

    # Year-by-year for top 5
    print(f"\n  ── Year-by-Year: Top 5 Champions ──")
    print(f"  {'Variant':<30} {'Lev':>4} {'Sl':>3} "
          + " | ".join(f"{'20'+str(20+i):>8}" for i in range(n_years)))
    print(f"  " + "─" * (40 + n_years * 11))

    for pr in phase2_results[:5]:
        m = pr["m"]
        yrs = m["yearly"]
        yr_strs = [f"{y['ret_pct']:>+8.1f}" for y in yrs]
        print(f"  {pr['name']:<30} {pr['lev']:>4.1f} {pr['slots']:>3} "
              + " | ".join(yr_strs))

    # ═══════════════════════════════════════════════════════
    #  Phase 4: Kelly Criterion for overall winner
    # ═══════════════════════════════════════════════════════
    if phase2_results:
        winner = phase2_results[0]
        w_trades = variant_trades[winner["name"]]
        pnl_1x = portfolio_sim(w_trades, winner["slots"])
        if len(pnl_1x) > 0:
            wr = np.mean(pnl_1x > 0)
            avg_win = np.mean(pnl_1x[pnl_1x > 0]) if np.any(pnl_1x > 0) else 0
            avg_loss = abs(np.mean(pnl_1x[pnl_1x < 0])) if np.any(pnl_1x < 0) else 1e-12
            rr = avg_win / avg_loss
            kelly = wr - (1 - wr) / rr
            optimal_lev = kelly / avg_loss if avg_loss > 0 else 1.0

            print(f"\n  ── Kelly Criterion Analysis (Winner: {winner['name']}) ──")
            print(f"    WR: {wr:.3f}  Avg Win: {avg_win:.4f}  Avg Loss: {avg_loss:.4f}")
            print(f"    R:R = {rr:.2f}")
            print(f"    Kelly fraction f* = {kelly:.3f}")
            print(f"    Optimal Kelly leverage ≈ {optimal_lev:.1f}x")
            print(f"    Half-Kelly (conservative) ≈ {optimal_lev/2:.1f}x")
            print(f"    Currently selected: {winner['lev']}x")

    # ═══════════════════════════════════════════════════════
    #  Tournament Mode Summary
    # ═══════════════════════════════════════════════════════
    if phase2_results:
        w = phase2_results[0]
        wm = w["m"]
        print(f"\n{'='*140}")
        print(f"  🏆 TOURNAMENT CHAMPION: {w['name']}  @{w['lev']}x  {w['slots']} slots")
        print(f"     AnnRet: {wm['ann_ret_pct']:+.1f}%  |  6yr Compound: "
              f"{((1 + wm['ann_ret_pct']/100)**6 - 1)*100:+,.0f}%")
        print(f"     DD: {wm['max_dd_pct']:.1f}%  |  Min Year: {wm['min_year_ret']:+.1f}%  "
              f"|  Losing Years: {wm['losing_years']}")
        print(f"     PF: {wm['PF']:.3f}  |  WR: {wm['WR']:.1f}%  |  N: {wm['N']}")
        print(f"     Config: sp={w['cfg']['sp']}, lp={w['cfg']['lp']}, "
              f"d_sl={w['cfg']['d_sl']}, s_sl={w['cfg']['s_sl']}, tp={w['cfg']['tp']}, "
              f"s_dur={w['cfg']['s_dur']}")
        print(f"{'='*140}")

    total = time.time() - t0
    print(f"\n  Total R40 completed in {total:.0f}s\n")


if __name__ == "__main__":
    main()
