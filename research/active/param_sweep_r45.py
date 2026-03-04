"""R45 — Parameter Sweep: MA Cluster + VR Squeeze Tuning

Motivation (paper trading session 2026-03-10):
  - MA Cluster: TRXUSDT entered with duration=70, mom=-0.007 (barely above 0.003).
    TP=8% never hit (max +3.11%). ADA SL'd at -2% despite reaching +2.53%.
  - VR Squeeze: 0 signals in 62 cycles (min VR=0.72, threshold=0.50).
    Not necessarily broken — market just wasn't in squeeze condition.

Issues to fix:
  1. MA: Momentum minimum 0.003 too low → test 0.005–0.03 range
  2. MA: TP=8% unreachable in current regime → test 3–6% range
  3. MA: No max_duration cap → zombie signals at duration=70+
  4. VR: Validate current params still optimal on 4.7yr data

Approach:
  - Phase 1: Validate baselines match deployed reports (±1%)
  - Phase 2: MA Cluster sweep with new min_momentum / max_duration / TP params
  - Phase 3: VR Squeeze sweep (SL/TP/trail variations)
  - Phase 4: Combined portfolio with best params from Phase 2+3
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

from research.active.mtf_vol_straddle_r31 import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    load_15m, build_resampled, precompute, pf_from_pnl,
    bootstrap_pf,
)
from research.active.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    COST_DIRECTIONAL, G_3TF, CFG_V13,
)
from research.active.ma_confluence_r44c import precompute_mas
from research.active.ma_cluster_refined_r44d import (
    collect_v16_trades,
)
from research.active.high_vol_complement_r44 import _dedup_sym_trades
from research.active.exit_improvement_r43 import (
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
#  Enhanced MA Cluster Sim: min_momentum + max_duration parameters
#
#  Original sim_ma_cluster_squeeze() has momentum threshold hardcoded at 0.003.
#  This version makes it a parameter, plus adds max_duration to filter
#  "zombie" signals that have been in cluster for 70+ bars.
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_enhanced(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    max_duration: int,
    min_momentum: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MA cluster squeeze with duration filter, max_duration cap, and
    configurable momentum minimum.

    Key changes from sim_ma_cluster_squeeze():
      1. min_momentum: configurable (was hardcoded 0.003)
      2. max_duration: skip if cluster duration exceeds this (zombie filter)
         Set to 0 or very large value to disable.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            i += 1
            continue

        # Compute MA band spread
        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            i += 1
            continue

        # Duration filter: cluster must have persisted
        if cluster_count < min_duration:
            i += 1
            continue

        # Max duration filter: skip zombie signals
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue

        # Breakout: close must be outside the cluster zone
        if close[i] <= ma_max and close[i] >= ma_min:
            i += 1
            continue

        # Momentum direction with configurable minimum
        if abs(momentum[i]) < min_momentum:
            i += 1
            continue
        d = 1 if momentum[i] > 0 else -1

        entry = close[i]

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1

        # Reset cluster after trade — one signal per cluster episode
        cluster_count = 0
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


def collect_enhanced_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    cluster_pct: float, min_duration: int,
    max_duration: int, min_momentum: float,
    sl: float, tp: float,
    tf_restriction: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect enhanced MA cluster trades with min_momentum + max_duration."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 192, 4: 48, 16: 12}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            if tf_restriction is not None and ratio != tf_restriction:
                continue
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            if "sma_200" not in d or "mom_8" not in d:
                continue
            p, e, x = sim_ma_cluster_enhanced(
                d["close"], d["high"], d["low"],
                d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"],
                d["mom_8"],
                cluster_pct, min_duration,
                max_duration, min_momentum,
                sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  VR Squeeze Sweep: test different TP/SL/VR threshold combos
# ══════════════════════════════════════════════════════════════════════════════


def collect_vr_trades_custom(
    cfg_base: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    d_vr: float, d_sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect VR squeeze D_mom8 trades with custom params."""
    cfg = dict(cfg_base)
    cfg["d_vr"] = d_vr
    cfg["d_sl"] = d_sl
    cfg["tp"] = tp
    return r38_collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"\nR45 — Parameter Sweep: MA Cluster + VR Squeeze [{commit}]")
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
    #  PHASE 1: Baseline Validation (must match ±1%)
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("PHASE 1: BASELINE VALIDATION")
    print("=" * 80)

    # VR Squeeze baseline
    bl_vr_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_vr_pnl = portfolio_sim(bl_vr_trades, 10)
    bl_vr = compute_metrics(bl_vr_pnl, data_years)

    # MA Cluster baseline (deployed: c=3% d=8 sl=2% tp=8%, 4h-only)
    bl_ma_trades = collect_v16_trades(
        data_15m, data_1h, data_4h,
        0.03, 8, 0.02, 0.08,
        tf_restriction=16)
    bl_ma_pnl = portfolio_sim(bl_ma_trades, 10)
    bl_ma = compute_metrics(bl_ma_pnl, data_years)

    print(f"  VR Squeeze:  N={bl_vr['N']}  PF={bl_vr['PF']:.3f}  "
          f"WR={bl_vr['WR']:.1f}%  EV={bl_vr['EV_bp']:+.1f}bp  "
          f"DD={bl_vr['max_dd_pct']:.1f}%")
    print(f"    Report ref: N=1574, PF=1.310, WR=25.9%")
    vr_pf_diff = abs(bl_vr['PF'] - 1.310) / 1.310 * 100
    print(f"    PF diff: {vr_pf_diff:.1f}% {'✓ PASS' if vr_pf_diff < 1.0 else '❌ FAIL'}")

    print(f"  MA Cluster:  N={bl_ma['N']}  PF={bl_ma['PF']:.3f}  "
          f"WR={bl_ma['WR']:.1f}%  EV={bl_ma['EV_bp']:+.1f}bp  "
          f"DD={bl_ma['max_dd_pct']:.1f}%")
    print(f"    Report ref: N=1836, PF=1.285, WR=38.6%")
    ma_pf_diff = abs(bl_ma['PF'] - 1.285) / 1.285 * 100
    print(f"    PF diff: {ma_pf_diff:.1f}% {'✓ PASS' if ma_pf_diff < 1.0 else '❌ FAIL'}")

    if vr_pf_diff >= 1.0 or ma_pf_diff >= 1.0:
        print("\n  ❌ BASELINE MISMATCH — aborting sweep.")
        print("  Check data or scripts before proceeding.")
        return

    print("  ✓ Both baselines validated.")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: MA Cluster Parameter Sweep
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("PHASE 2: MA CLUSTER PARAMETER SWEEP")
    print("=" * 80)
    print("  Testing: min_momentum, max_duration, TP, SL")
    print()

    ma_results: list[tuple[str, dict, list]] = []

    # ── 2A: Momentum minimum sweep (key fix for zombie signals) ──
    print("  ── 2A: Momentum Minimum Sweep ──")
    print("  (cluster=3%, duration=8, 4h-only, sl=2%)")
    print(f"  {'min_mom':>10} {'tp':>6} {'N':>6} {'PF':>7} {'WR':>6} "
          f"{'EV_bp':>8} {'DD%':>6} {'Calm':>6} {'RR':>6}")
    print("  " + "-" * 68)

    for min_mom in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]:
        for tp in [0.03, 0.04, 0.05, 0.06, 0.08]:
            label = f"mom≥{min_mom:.3f} tp={tp*100:.0f}%"
            trades = collect_enhanced_trades(
                data_15m, data_1h, data_4h,
                cluster_pct=0.03, min_duration=8,
                max_duration=0, min_momentum=min_mom,
                sl=0.02, tp=tp,
                tf_restriction=16)
            if not trades:
                continue
            pnl = portfolio_sim(trades, 10)
            m = compute_metrics(pnl, data_years)
            flag = "✓" if m["PF"] >= 1.10 else " "
            print(f"  {min_mom:>10.3f} {tp*100:>5.0f}% {m['N']:>6} "
                  f"{m['PF']:>7.3f} {m['WR']:>5.1f}% {m['EV_bp']:>+7.1f} "
                  f"{m['max_dd_pct']:>5.1f}% {m['calmar']:>5.1f} "
                  f"{m['RR']:>5.2f} {flag}")
            ma_results.append((label, m, trades))

    # ── 2B: Max duration sweep (zombie filter) ──
    print()
    print("  ── 2B: Max Duration Sweep ──")
    print("  (cluster=3%, min_duration=8, min_mom=0.003, 4h-only)")
    print(f"  {'max_dur':>8} {'sl':>6} {'tp':>6} {'N':>6} {'PF':>7} {'WR':>6} "
          f"{'EV_bp':>8} {'DD%':>6} {'Calm':>6}")
    print("  " + "-" * 66)

    for max_dur in [16, 24, 32, 48, 72, 0]:  # 0 = no limit
        for sl in [0.015, 0.02]:
            for tp in [0.04, 0.05, 0.08]:
                dur_label = f"{max_dur}" if max_dur > 0 else "∞"
                label = f"maxd={dur_label} sl={sl*100:.1f}% tp={tp*100:.0f}%"
                trades = collect_enhanced_trades(
                    data_15m, data_1h, data_4h,
                    cluster_pct=0.03, min_duration=8,
                    max_duration=max_dur, min_momentum=0.003,
                    sl=sl, tp=tp,
                    tf_restriction=16)
                if not trades:
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= 1.10 else " "
                print(f"  {dur_label:>8} {sl*100:>5.1f}% {tp*100:>5.0f}% "
                      f"{m['N']:>6} {m['PF']:>7.3f} {m['WR']:>5.1f}% "
                      f"{m['EV_bp']:>+7.1f} {m['max_dd_pct']:>5.1f}% "
                      f"{m['calmar']:>5.1f} {flag}")
                ma_results.append((label, m, trades))

    # ── 2C: Combined best factors (momentum + duration + TP) ──
    print()
    print("  ── 2C: Combined Factor Sweep ──")
    print("  (cluster=3%, min_duration=8, 4h-only)")
    print(f"  {'min_mom':>8} {'max_d':>6} {'sl':>6} {'tp':>6} {'N':>6} "
          f"{'PF':>7} {'WR':>6} {'EV_bp':>8} {'DD%':>6} {'Calm':>6}")
    print("  " + "-" * 72)

    combo_results: list[tuple[str, dict, list]] = []
    for min_mom in [0.005, 0.01, 0.015]:
        for max_dur in [24, 48, 0]:
            for sl in [0.015, 0.02]:
                for tp in [0.03, 0.04, 0.05, 0.06, 0.08]:
                    dur_label = f"{max_dur}" if max_dur > 0 else "∞"
                    label = (f"mom≥{min_mom:.3f} maxd={dur_label} "
                             f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                    trades = collect_enhanced_trades(
                        data_15m, data_1h, data_4h,
                        cluster_pct=0.03, min_duration=8,
                        max_duration=max_dur, min_momentum=min_mom,
                        sl=sl, tp=tp,
                        tf_restriction=16)
                    if not trades:
                        continue
                    pnl = portfolio_sim(trades, 10)
                    m = compute_metrics(pnl, data_years)
                    if m["PF"] >= 1.10 and m["N"] >= 300:
                        flag = "✓"
                        combo_results.append((label, m, trades))
                    else:
                        flag = " "
                    print(f"  {min_mom:>8.3f} {dur_label:>6} {sl*100:>5.1f}% "
                          f"{tp*100:>5.0f}% {m['N']:>6} {m['PF']:>7.3f} "
                          f"{m['WR']:>5.1f}% {m['EV_bp']:>+7.1f} "
                          f"{m['max_dd_pct']:>5.1f}% {m['calmar']:>5.1f} {flag}")

    # ── 2D: MA Summary ──
    print()
    print("  ── MA Cluster Top 15 (PF ≥ 1.10, N ≥ 300, sorted by Calmar) ──")
    all_ma = ma_results + combo_results
    viable_ma = [(l, m, t) for l, m, t in all_ma
                 if m["PF"] >= 1.10 and m["N"] >= 300 and m["max_dd_pct"] < 60]
    viable_ma.sort(key=lambda x: -x[1]["calmar"])
    for i, (label, m, _) in enumerate(viable_ma[:15]):
        print(f"  {i+1:>2}. {label}")
        print(f"      N={m['N']:>5}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
              f"EV={m['EV_bp']:+.1f}bp  DD={m['max_dd_pct']:.1f}%  "
              f"Calmar={m['calmar']:.1f}  RR={m['RR']:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: VR Squeeze Parameter Sweep
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 80)
    print("PHASE 3: VR SQUEEZE PARAMETER SWEEP")
    print("=" * 80)
    print("  Testing: d_vr threshold, SL, TP")
    print()

    vr_results: list[tuple[str, dict, list]] = []

    print(f"  {'d_vr':>6} {'d_sl':>6} {'tp':>6} {'N':>6} {'PF':>7} {'WR':>6} "
          f"{'EV_bp':>8} {'DD%':>6} {'Calm':>6}")
    print("  " + "-" * 66)

    for d_vr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for d_sl in [0.008, 0.01, 0.012, 0.015]:
            for tp in [0.03, 0.05, 0.08, 0.10, 0.12]:
                label = f"vr<{d_vr:.2f} sl={d_sl*100:.1f}% tp={tp*100:.0f}%"
                trades = collect_vr_trades_custom(
                    CFG_V13, data_15m, data_1h, data_4h,
                    d_vr=d_vr, d_sl=d_sl, tp=tp)
                if not trades:
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= 1.10 else " "
                print(f"  {d_vr:>6.2f} {d_sl*100:>5.1f}% {tp*100:>5.0f}% "
                      f"{m['N']:>6} {m['PF']:>7.3f} {m['WR']:>5.1f}% "
                      f"{m['EV_bp']:>+7.1f} {m['max_dd_pct']:>5.1f}% "
                      f"{m['calmar']:>5.1f} {flag}")
                vr_results.append((label, m, trades))

    # ── VR Summary ──
    print()
    print("  ── VR Squeeze Top 15 (PF ≥ 1.10, N ≥ 200, sorted by Calmar) ──")
    viable_vr = [(l, m, t) for l, m, t in vr_results
                 if m["PF"] >= 1.10 and m["N"] >= 200 and m["max_dd_pct"] < 70]
    viable_vr.sort(key=lambda x: -x[1]["calmar"])
    for i, (label, m, _) in enumerate(viable_vr[:15]):
        print(f"  {i+1:>2}. {label}")
        print(f"      N={m['N']:>5}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
              f"EV={m['EV_bp']:+.1f}bp  DD={m['max_dd_pct']:.1f}%  "
              f"Calmar={m['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 4: Combined Portfolio (Best MA + Best VR)
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 80)
    print("PHASE 4: COMBINED PORTFOLIO")
    print("=" * 80)

    # Select top 5 MA and top 3 VR combos
    top_ma = viable_ma[:5] if viable_ma else []
    top_vr = viable_vr[:3] if viable_vr else [(
        "baseline", bl_vr,
        r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    )]

    if not top_ma:
        # Fallback to baseline MA
        top_ma = [("baseline MA c=3% d=8 sl=2% tp=8%", bl_ma, bl_ma_trades)]

    print(f"  Testing {len(top_ma)} MA × {len(top_vr)} VR combinations")
    print()

    combined_results: list[tuple[str, str, dict]] = []

    for vr_label, vr_m, vr_trades in top_vr:
        for ma_label, ma_m, ma_trades in top_ma:
            comb_pnl, comb_accepted = portfolio_sim_with_details(
                vr_trades + ma_trades, 10)
            comb_m = compute_metrics(comb_pnl, data_years)
            combined_results.append((vr_label, ma_label, comb_m))

    # Sort by Calmar
    combined_results.sort(key=lambda x: -x[2]["calmar"])

    print(f"  {'Rank':>4} {'N':>6} {'PF':>7} {'WR':>6} {'EV_bp':>8} "
          f"{'DD%':>6} {'Calmar':>7}")
    print("  " + "-" * 52)
    for i, (vr_l, ma_l, m) in enumerate(combined_results[:20]):
        print(f"  {i+1:>4} {m['N']:>6} {m['PF']:>7.3f} {m['WR']:>5.1f}% "
              f"{m['EV_bp']:>+7.1f} {m['max_dd_pct']:>5.1f}% "
              f"{m['calmar']:>6.1f}")
        print(f"       VR: {vr_l}")
        print(f"       MA: {ma_l}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 5: Deep Validation of Top 3 Combined
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("PHASE 5: DEEP VALIDATION (Top 3 Combined)")
    print("=" * 80)

    for rank, (vr_l, ma_l, _) in enumerate(combined_results[:3], 1):
        print(f"\n  ── Rank #{rank} ──")
        print(f"  VR: {vr_l}")
        print(f"  MA: {ma_l}")

        # Re-collect trades for this combo
        vr_trades_sel = None
        ma_trades_sel = None
        for l, m, t in top_vr:
            if l == vr_l:
                vr_trades_sel = t
                break
        for l, m, t in top_ma:
            if l == ma_l:
                ma_trades_sel = t
                break

        if vr_trades_sel is None or ma_trades_sel is None:
            print("    ⚠️ Could not re-find trades, skipping")
            continue

        comb_pnl, comb_accepted = portfolio_sim_with_details(
            vr_trades_sel + ma_trades_sel, 10)
        comb_m = compute_metrics(comb_pnl, data_years)
        comb_yr = year_metrics(comb_pnl, comb_accepted, year_bounds)
        comb_boot = bootstrap_pf(comb_pnl, 5000)

        print(f"  N={comb_m['N']}  PF={comb_m['PF']:.3f}  WR={comb_m['WR']:.1f}%  "
              f"EV={comb_m['EV_bp']:+.1f}bp")
        print(f"  AnnRet={comb_m['ann_ret_pct']:+.1f}% (1x compound)  "
              f"DD={comb_m['max_dd_pct']:.1f}%  Calmar={comb_m['calmar']:.1f}")
        print(f"  Bootstrap: P(PF≥1.0)={comb_boot['P_PF_ge_1']}%  "
              f"CI=[{comb_boot['CI_lo']:.3f}, {comb_boot['CI_hi']:.3f}]")

        print("  Yearly:")
        positive_years = 0
        for yr_key in sorted(comb_yr.keys()):
            y = comb_yr[yr_key]
            flag = "✓" if y["PF"] >= 1.0 else "❌"
            if y["PF"] >= 1.0:
                positive_years += 1
            print(f"    {yr_key}: N={y['N']:>4}  PF={y['PF']:.2f}  "
                  f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")
        print(f"  Positive years: {positive_years}/{len(comb_yr)}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 6: vs Baseline Comparison
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 80)
    print("PHASE 6: vs CURRENT BASELINE COMPARISON")
    print("=" * 80)

    # Current deployed combo
    bl_comb_pnl, bl_comb_acc = portfolio_sim_with_details(
        bl_vr_trades + bl_ma_trades, 10)
    bl_comb = compute_metrics(bl_comb_pnl, data_years)
    bl_comb_boot = bootstrap_pf(bl_comb_pnl, 5000)

    print(f"  Current (deployed):  N={bl_comb['N']}  PF={bl_comb['PF']:.3f}  "
          f"DD={bl_comb['max_dd_pct']:.1f}%  Calmar={bl_comb['calmar']:.1f}  "
          f"P(PF≥1)={bl_comb_boot['P_PF_ge_1']}%")

    if combined_results:
        best = combined_results[0]
        print(f"  Best new combo:      N={best[2]['N']}  "
              f"PF={best[2]['PF']:.3f}  "
              f"DD={best[2]['max_dd_pct']:.1f}%  "
              f"Calmar={best[2]['calmar']:.1f}")
        print(f"    VR: {best[0]}")
        print(f"    MA: {best[1]}")

        pf_change = (best[2]['PF'] - bl_comb['PF']) / bl_comb['PF'] * 100
        cal_change = best[2]['calmar'] - bl_comb['calmar']
        print(f"\n  PF change: {pf_change:+.1f}%")
        print(f"  Calmar change: {cal_change:+.1f}")

        if pf_change < -5:
            print("\n  ⚠️ New params have LOWER PF than baseline.")
            print("  Recommend keeping current params unless Calmar improved significantly.")

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
    print(f"  Commit: {commit}")


if __name__ == "__main__":
    main()
