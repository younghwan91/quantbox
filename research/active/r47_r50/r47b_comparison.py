#!/usr/bin/env python3
"""r47b: 4-way comparison — v14.3.0 vs r47 exits × TBR on/off.

Combinations:
  A: v14.3.0 exits  + TBR ON   ← canonical production baseline
  B: v14.3.0 exits  + TBR OFF
  C: r47 best exits + TBR ON
  D: r47 best exits + TBR OFF

Entry filter (TBR) and exit params are varied independently.
Only S_d8 and MA entry signals are shared across all four.
"""

from __future__ import annotations

import subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

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
)
from quantbox.backtest.trade_sim import sim_directional_trailing
from research.active.r31_r40.scaling_r38 import CFG_V13
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m,
    collect_d_trades_with_tbr,
    collect_d_trades_trailing,
    collect_enhanced_trades,
    compute_tbr_z_at_bar,
    extended_metrics,
    print_full,
    D_SL, D_TP, D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION, MA_MIN_MOMENTUM,
    MA_SL, MA_TP, MA_TF_RESTRICTION,
    VR_LN, MA_LN,
)
from research.active.r47_r50.r47_exit_ab_test import (
    collect_d_trades_custom_trail,
    collect_ma_cluster_trailing,
)

R47_TRAIL_ACTIVATE = 0.005
R47_TRAIL_DISTANCE = 0.002
R47_MA_TRAIL_ACT   = 0.04
R47_MA_TRAIL_DIST  = 0.005
TBR_Z_WINDOW       = 192
TBR_Z_THRESHOLD    = 0.5
MAX_CONCURRENT     = 10


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
    """r47 custom trail + F5 TBR gate."""
    sp, lp = cfg["sp"], cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    all_trades: list = []
    n_blocked = 0

    for sym in sorted(data_15m.keys()):
        sym_d = data_15m[sym]
        taker_buy_15m = sym_d.get("taker_buy")
        volume_15m    = sym_d.get("volume")
        if taker_buy_15m is None or volume_15m is None:
            continue

        sym_trades: list = []
        for _tf_label, ratio, warmup, mh_d, _mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr, mom8 = d.get(vr_key), d.get("mom_8")
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
                tbr_z = compute_tbr_z_at_bar(taker_buy_15m, volume_15m, entry_15m, tbr_z_window)
                direction = int(dire[entry_tf]) if entry_tf < len(dire) else 0
                if direction > 0 and tbr_z > tbr_z_threshold:
                    n_blocked += 1; continue
                if direction < 0 and tbr_z < -tbr_z_threshold:
                    n_blocked += 1; continue
                sym_trades.append((entry_15m, int(x[k]) * ratio + (ratio - 1), float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            if not any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted):
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades, n_blocked


def run_combo(
    label: str,
    d_trades: list,
    s_trades: list,
    ma_trades: list,
    data_years: float,
    year_bounds: dict,
    n_tbr_blocked: int = 0,
) -> tuple[dict, dict, dict]:
    comb = d_trades + s_trades + ma_trades
    pnl, accepted = portfolio_sim_with_details(comb, MAX_CONCURRENT)
    ln  = _blended_ln(len(d_trades + s_trades), len(ma_trades))
    m   = extended_metrics(pnl, data_years, ln_leverage=ln)
    boot = bootstrap_pf(pnl, 5000)
    yr  = year_metrics(pnl, accepted, year_bounds)
    return m, boot, yr


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*82}")
    print(f"  r47b: 4-way comparison  |  v14.3.0 vs r47 exits  ×  TBR on/off")
    print(f"  Commit: {commit}  |  Branch: research/r47-exit-ab-test")
    print(f"{'='*82}")

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

    # ── Shared trades ──────────────────────────────────────────────────────────
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")

    # v14 exits (MA flat TP=8%)
    ma_v14 = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )
    # r47 exits (MA trailing)
    ma_r47 = collect_ma_cluster_trailing(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
        trail_activate=R47_MA_TRAIL_ACT, trail_distance=R47_MA_TRAIL_DIST,
    )

    # D trades: 4 variants
    # A: v14 exits + TBR ON
    d_A, n_A_blocked = collect_d_trades_with_tbr(
        CFG_V13, data_15m, data_1h, data_4h,
        tbr_z_window=TBR_Z_WINDOW, tbr_z_threshold=TBR_Z_THRESHOLD,
    )
    # B: v14 exits + TBR OFF
    d_B = collect_d_trades_trailing(CFG_V13, data_15m, data_1h, data_4h)
    n_B_blocked = 0
    # C: r47 exits + TBR ON
    d_C, n_C_blocked = collect_d_trades_custom_trail_tbr(
        CFG_V13, data_15m, data_1h, data_4h,
        trail_activate=R47_TRAIL_ACTIVATE, trail_distance=R47_TRAIL_DISTANCE,
    )
    # D: r47 exits + TBR OFF
    d_D = collect_d_trades_custom_trail(
        CFG_V13, data_15m, data_1h, data_4h,
        trail_activate=R47_TRAIL_ACTIVATE, trail_distance=R47_TRAIL_DISTANCE,
    )
    n_D_blocked = 0

    print(f"\n  D raw counts: A={len(d_A)}(+{n_A_blocked}blk)  B={len(d_B)}  C={len(d_C)}(+{n_C_blocked}blk)  D={len(d_D)}")
    print(f"  MA raw:  v14={len(ma_v14)}  r47={len(ma_r47)}  |  S_d8={len(s_trades)}")

    # Run all 4 combos
    print("\n  Running portfolio_sim for all 4 combos...", end=" ", flush=True)
    mA, bA, yrA = run_combo("A", d_A, s_trades, ma_v14, data_years, year_bounds, n_A_blocked)
    mB, bB, yrB = run_combo("B", d_B, s_trades, ma_v14, data_years, year_bounds)
    mC, bC, yrC = run_combo("C", d_C, s_trades, ma_r47, data_years, year_bounds, n_C_blocked)
    mD, bD, yrD = run_combo("D", d_D, s_trades, ma_r47, data_years, year_bounds)
    print(f"done  ({time.time()-t0:.1f}s total)")

    # ── 4-way summary table ────────────────────────────────────────────────────
    rows = [
        ("A", "v14.3.0 exits + TBR ON  ★canonical", d_A, ma_v14, mA, bA, n_A_blocked),
        ("B", "v14.3.0 exits + TBR OFF",             d_B, ma_v14, mB, bB, 0),
        ("C", "r47 best exits + TBR ON",              d_C, ma_r47, mC, bC, n_C_blocked),
        ("D", "r47 best exits + TBR OFF",             d_D, ma_r47, mD, bD, 0),
    ]

    print(f"\n{'='*82}")
    print(f"  4-WAY SUMMARY  (D trail: v14=ta2%td0.3%  |  r47=ta0.5%td0.2%)")
    print(f"  (MA: v14=flat TP8%  |  r47=trailing ta4%td0.5%)")
    print(f"{'='*82}")
    hdr = f"  {'':3} {'Label':<35} {'D raw':>6} {'MA raw':>7} {'N':>5} {'PF':>6} {'WR':>6} {'EV':>7} {'Sharpe':>7} {'MaxDD':>7}"
    print(hdr)
    print(f"  {'─'*88}")
    for tag, label, d_t, ma_t, m, b, n_blk in rows:
        blk = f"(+{n_blk}blk)" if n_blk else "      "
        maxdd = m.get("prod_max_dd_pct", 0)
        print(f"  {tag}  {label:<35} {len(d_t):>4}{blk} {len(ma_t):>7} {m['N']:>5} "
              f"{m['PF']:>6.3f} {m['WR']:>5.1f}% {m['EV_bp']:>+6.1f}bp {m.get('sharpe',0):>7.2f} {maxdd:>6.1f}%")

    print(f"\n  Bootstrap 95% CI:")
    for tag, label, *_, m, b, _ in rows:
        print(f"    {tag}: PF∈[{b['CI_lo']:.3f}, {b['CI_hi']:.3f}]  P(PF≥1)={b['P_PF_ge_1']:.1f}%")

    # ── Per-year table ─────────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"  PER-YEAR PF")
    print(f"{'='*82}")
    all_yrs = sorted(set(list(yrA) + list(yrB) + list(yrC) + list(yrD)))
    print(f"  {'Year':<6} {'A (v14+TBR)':>12} {'B (v14-TBR)':>12} {'C (r47+TBR)':>12} {'D (r47-TBR)':>12}  {'ΔA→C':>7} {'ΔB→D':>7}")
    print(f"  {'─'*72}")
    for yr in all_yrs:
        a = yrA.get(yr, {})
        b = yrB.get(yr, {})
        c = yrC.get(yr, {})
        d_yr = yrD.get(yr, {})
        fa = "✓" if a.get("PF", 0) >= 1 else "❌"
        fb = "✓" if b.get("PF", 0) >= 1 else "❌"
        fc = "✓" if c.get("PF", 0) >= 1 else "❌"
        fd = "✓" if d_yr.get("PF", 0) >= 1 else "❌"
        dac = c.get("PF", 0) - a.get("PF", 0)
        dbd = d_yr.get("PF", 0) - b.get("PF", 0)
        print(f"  {yr:<6} {a.get('N',0):>4} {a.get('PF',0):>5.3f}{fa}  "
              f"{b.get('N',0):>4} {b.get('PF',0):>5.3f}{fb}  "
              f"{c.get('N',0):>4} {c.get('PF',0):>5.3f}{fc}  "
              f"{d_yr.get('N',0):>4} {d_yr.get('PF',0):>5.3f}{fd}  "
              f"{dac:>+7.3f}  {dbd:>+7.3f}")

    # ── TBR effect (same exits, TBR on vs off) ────────────────────────────────
    print(f"\n{'='*82}")
    print(f"  TBR EFFECT  (same exits, TBR on vs off)")
    print(f"{'='*82}")
    print(f"  v14 exits:  TBR ON (A) PF={mA['PF']:.3f} N={mA['N']}  vs  TBR OFF (B) PF={mB['PF']:.3f} N={mB['N']}")
    print(f"              ΔPF={mA['PF']-mB['PF']:+.3f}  ΔEV={mA['EV_bp']-mB['EV_bp']:+.1f}bp  ΔN={mA['N']-mB['N']:+d}")
    print(f"  r47 exits:  TBR ON (C) PF={mC['PF']:.3f} N={mC['N']}  vs  TBR OFF (D) PF={mD['PF']:.3f} N={mD['N']}")
    print(f"              ΔPF={mC['PF']-mD['PF']:+.3f}  ΔEV={mC['EV_bp']-mD['EV_bp']:+.1f}bp  ΔN={mC['N']-mD['N']:+d}")

    print(f"\n{'='*82}")
    print(f"  EXIT EFFECT  (same TBR, v14 vs r47 exits)")
    print(f"{'='*82}")
    print(f"  TBR ON:   v14(A) PF={mA['PF']:.3f}  →  r47(C) PF={mC['PF']:.3f}  ΔPF={mC['PF']-mA['PF']:+.3f}  ΔEV={mC['EV_bp']-mA['EV_bp']:+.1f}bp")
    print(f"  TBR OFF:  v14(B) PF={mB['PF']:.3f}  →  r47(D) PF={mD['PF']:.3f}  ΔPF={mD['PF']-mB['PF']:+.3f}  ΔEV={mD['EV_bp']-mB['EV_bp']:+.1f}bp")

    print(f"\n  L/N: VR={VR_LN}, MA={MA_LN}  |  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*82}\n")


if __name__ == "__main__":
    main()
