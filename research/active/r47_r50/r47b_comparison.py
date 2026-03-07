#!/usr/bin/env python3
"""r47b: Fair exit-only comparison — v14.3.0 exits vs r47 best exits.

METHODOLOGY
===========
Both sides use IDENTICAL entry filters:
  - F5 TBR gate (z_window=192, z_threshold=0.5) — same on both sides
  - Same S_d8 straddle trades
  - Same MA cluster entry signals

Only the EXIT parameters differ:
  Baseline:  D_mom8 ta=2.0%, td=0.3%  |  MA cluster flat TP=8%
  r47 best:  D_mom8 ta=0.5%, td=0.2%  |  MA cluster trailing ta=4%, td=0.5%

WHY N STILL CHANGES
===================
Even with identical entry filters (same TBR), portfolio_sim accepted N can
differ because earlier exits free slots, allowing re-entries that were
previously blocked — real system behavior, not a methodology error.

Per-trade EV and PF are N-independent and are the primary comparison metrics.
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
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.signals import gen_vr_signal, make_direction_momentum
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics,
    get_year_boundaries,
    sim_directional_trailing,
)
from research.active.r31_r40.scaling_r38 import CFG_V13
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m,
    collect_d_trades_with_tbr,   # baseline D: ta=2%, WITH TBR
    collect_enhanced_trades,     # MA cluster: flat TP=8%
    compute_tbr_z_at_bar,        # TBR gate logic
    extended_metrics,
    print_full,
    D_SL, D_TP, D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION, MA_MIN_MOMENTUM,
    MA_SL, MA_TP, MA_TF_RESTRICTION,
    VR_LN, MA_LN,
)
from research.active.r47_r50.r47_exit_ab_test import (
    collect_ma_cluster_trailing,    # r47 MA: trailing ta=4%, td=0.5%
)

# ── r47 best exit params ───────────────────────────────────────────────────────
R47_TRAIL_ACTIVATE  = 0.005   # D_mom8: was 0.020
R47_TRAIL_DISTANCE  = 0.002   # D_mom8: was 0.003
R47_MA_TRAIL_ACT    = 0.04    # MA trailing activate
R47_MA_TRAIL_DIST   = 0.005   # MA trailing distance

TBR_Z_WINDOW    = 192
TBR_Z_THRESHOLD = 0.5
MAX_CONCURRENT  = 10


def get_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _blended_ln(vr_n: int, ma_n: int) -> float:
    total = vr_n + ma_n
    return (vr_n * VR_LN + ma_n * MA_LN) / total if total > 0 else 1.5


def collect_d_trades_custom_trail_tbr(
    cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    trail_activate: float = R47_TRAIL_ACTIVATE,
    trail_distance: float = R47_TRAIL_DISTANCE,
    d_sl: float = D_SL,
    d_tp: float = D_TP,
    tbr_z_window: int = TBR_Z_WINDOW,
    tbr_z_threshold: float = TBR_Z_THRESHOLD,
) -> tuple[list, int]:
    """r47 D_mom8: custom trailing exit + same F5 TBR gate as v14.3.0 baseline.

    Mirror of collect_d_trades_with_tbr() but with parameterized trail_activate
    and trail_distance. TBR gate logic is identical.
    """
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}

    all_trades: list[tuple[str, int, int, float]] = []
    n_blocked = 0

    for sym in sorted(data_15m.keys()):
        sym_d = data_15m[sym]
        taker_buy_15m = sym_d.get("taker_buy")
        volume_15m    = sym_d.get("volume")
        if taker_buy_15m is None or volume_15m is None:
            continue

        sym_trades: list[tuple[int, int, float]] = []

        for _tf_label, ratio, warmup, mh_d, _mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            mom8 = d.get("mom_8")
            if vr is None or mom8 is None:
                continue

            sig  = gen_vr_signal(vr, cfg["d_vr"], warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, d_sl, d_tp, mh_d, COST_DIRECTIONAL,
                trail_activate=trail_activate, trail_distance=trail_distance,
            )

            for k in range(len(p)):
                entry_tf  = int(e[k])
                entry_15m = entry_tf * ratio

                # ── F5 TBR gate (identical to collect_d_trades_with_tbr) ────
                tbr_z = compute_tbr_z_at_bar(
                    taker_buy_15m, volume_15m, entry_15m, tbr_z_window
                )
                direction = int(dire[entry_tf]) if entry_tf < len(dire) else 0
                if direction > 0 and tbr_z > tbr_z_threshold:
                    n_blocked += 1
                    continue
                if direction < 0 and tbr_z < -tbr_z_threshold:
                    n_blocked += 1
                    continue
                # ─────────────────────────────────────────────────────────────

                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades, n_blocked


# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*78}")
    print(f"  r47b: Exit-Only Comparison  |  v14.3.0 exits vs r47 best exits")
    print(f"  Commit: {commit}  |  Branch: research/r47-exit-ab-test")
    print(f"  Both sides: F5 TBR gate (z_window=192, z_threshold=0.5) + same S_d8")
    print(f"  Only exit params differ (entry filter identical)")
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

    avg_bars    = int(np.mean([len(v["close"]) for v in data_15m.values()]))
    data_years  = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)
    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  {data_years:.2f} years  |  {time.time()-t0:.1f}s")

    # ── Shared: S_d8 straddle (identical for both) ─────────────────────────────
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    print(f"\n  S_d8 (shared): raw {len(s_trades)} straddle trades")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE: v14.3.0 exits (D ta=2%, MA flat TP=8%) + F5 TBR
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  BASELINE  D(ta=2.0%, td=0.3%, F5 TBR) + S_d8 + MA(flat TP=8%)")
    print(f"{'─'*78}")

    d_bl, n_bl_blocked = collect_d_trades_with_tbr(
        CFG_V13, data_15m, data_1h, data_4h,
        tbr_z_window=TBR_Z_WINDOW, tbr_z_threshold=TBR_Z_THRESHOLD,
    )
    ma_bl = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )
    print(f"  Raw trades: D={len(d_bl)} (+{n_bl_blocked} TBR-blocked)  S={len(s_trades)}  MA={len(ma_bl)}")

    bl_comb = d_bl + s_trades + ma_bl
    bl_pnl, bl_accepted = portfolio_sim_with_details(bl_comb, MAX_CONCURRENT)
    bl_ln   = _blended_ln(len(d_bl + s_trades), len(ma_bl))
    bl_m    = extended_metrics(bl_pnl, data_years, ln_leverage=bl_ln)
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
    # r47 BEST: r47 exits (D ta=0.5%, MA trailing ta=4%) + F5 TBR
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  r47 BEST  D(ta=0.5%, td=0.2%, F5 TBR) + S_d8 + MA trailing(ta=4%,td=0.5%)")
    print(f"{'─'*78}")

    d_r47, n_r47_blocked = collect_d_trades_custom_trail_tbr(
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
    print(f"  Raw trades: D={len(d_r47)} (+{n_r47_blocked} TBR-blocked)  S={len(s_trades)}  MA={len(ma_r47)}")

    delta_d_raw  = len(d_r47)  - len(d_bl)
    delta_ma_raw = len(ma_r47) - len(ma_bl)
    print(f"  ΔRaw D vs baseline: {delta_d_raw:+d}  ΔRaw MA: {delta_ma_raw:+d}  (re-entry effect from earlier exits)")

    r47_comb = d_r47 + s_trades + ma_r47
    r47_pnl, r47_accepted = portfolio_sim_with_details(r47_comb, MAX_CONCURRENT)
    r47_ln   = _blended_ln(len(d_r47 + s_trades), len(ma_r47))
    r47_m    = extended_metrics(r47_pnl, data_years, ln_leverage=r47_ln)
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
    print("  SIDE-BY-SIDE (same F5 TBR entry filter, exit params only differ)")
    print(f"{'='*78}")
    print(f"  {'':42} {'Raw D':>6} {'Raw MA':>7} {'Acc N':>6} {'PF':>6} {'WR':>6} {'EV':>7} {'Sharpe':>7}")
    print(f"  {'─'*90}")

    def _row(label: str, raw_d: int, raw_ma: int, m: dict) -> str:
        sh = m.get("sharpe", 0.0)
        return (f"  {label:<42} {raw_d:>6} {raw_ma:>7} {m['N']:>6} "
                f"{m['PF']:>6.3f} {m['WR']:>5.1f}% {m['EV_bp']:>+6.1f}bp {sh:>7.2f}")

    print(_row("Baseline (D ta=2% | MA flat TP=8%,  TBR)", len(d_bl),   len(ma_bl),   bl_m))
    print(_row("r47 best (D ta=0.5% | MA trail 4%, TBR)", len(d_r47),  len(ma_r47),  r47_m))

    d_pf = r47_m["PF"]     - bl_m["PF"]
    d_ev = r47_m["EV_bp"]  - bl_m["EV_bp"]
    d_sh = r47_m.get("sharpe", 0) - bl_m.get("sharpe", 0)
    d_wr = r47_m["WR"]     - bl_m["WR"]
    d_n  = r47_m["N"]      - bl_m["N"]
    d_rd = len(d_r47)      - len(d_bl)
    d_rm = len(ma_r47)     - len(ma_bl)
    print(f"  {'Δ (r47 - baseline)':<42} {d_rd:>+6} {d_rm:>+7} {d_n:>+6} "
          f"{d_pf:>+6.3f} {d_wr:>+5.1f}% {d_ev:>+6.1f}bp {d_sh:>+7.2f}")

    print(f"\n  TBR blocks: baseline={n_bl_blocked}  r47={n_r47_blocked}")
    if d_rd != 0:
        print(f"  ΔRaw D={d_rd:+d}: earlier exits (ta=0.5% vs 2%) allow re-entries — real system effect.")

    print(f"\n  Per-year PF comparison:")
    print(f"  {'Year':<8} {'BL N':>5} {'BL PF':>7} {'BL EV':>8}  {'r47 N':>6} {'r47 PF':>7} {'r47 EV':>8}  Δ PF")
    print(f"  {'─'*70}")
    for yr in sorted(set(list(bl_yr.keys()) + list(r47_yr.keys()))):
        b = bl_yr.get(yr, {})
        r = r47_yr.get(yr, {})
        dpf = r.get("PF", 0) - b.get("PF", 0)
        bf  = "✓" if b.get("PF", 0) >= 1.0 else "❌"
        rf  = "✓" if r.get("PF", 0) >= 1.0 else "❌"
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
