"""r73: Trail-Aware 3-Strategy Combined Portfolio Backtest.

r72 used flat TP=5% simulation → S_d8 PF=1.066 (vs trail-aware 4.948 in r66).
This script reruns the 3-strategy combined portfolio using actual trail exit logic
to get realistic combined Calmar/PF estimates.

Trail params (from config):
  D_mom8: trail_activate=1.0%, trail_distance=0.2%, sl=1.0%
  S_d8:   trail_activate=1.5%, trail_distance=0.2%, sl=1.5%, tp=5%
  MA Cluster: flat TP=8%, sl=2% (no trail sim available)

Expected: 3-strat trail Calmar >> r72 flat (5.6), closer to r66 S_d8 standalone (PF=4.948).

Decision framework:
  - 3-strat trail Calmar >> D+MA trail × 1.1 → Option A: accept S_d8 dominance
  - 3-strat trail Calmar ≈ D+MA trail       → Option B: max_concurrent increase (10→15)

Usage:
    uv run python research/active/r71_r75/r73_trail_combined.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)
from research.active.r41_r45.r43_exit_improvement import (
    get_year_boundaries,
    sim_directional_trailing,
    sim_straddle_trailing,
    year_metrics,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ─── MA Cluster deployed params (R45 champion) ────────────────────────────────
MA_CLUSTER_PCT    = 0.03
MA_MIN_DURATION   = 8
MA_MAX_DURATION   = 24
MA_MIN_MOMENTUM   = 0.010
MA_SL             = 0.02
MA_TP             = 0.08
MA_TF_RESTRICTION = 16  # 4h only

# G_4TF = 30m added for S_d8 (matches r66 canonical)
WARMUP_30m: int = 100


def _get_g4tf(cfg_obj: VolStraddleConfig) -> list:
    mh_30m = cfg_obj.max_hold_15m // 2
    tf_30m = ("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)
    return [tf_30m] + list(G_3TF)


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_15m(whitelist: list[str]) -> dict:
    """Load 60 whitelist symbols with _offset alignment."""
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in set(whitelist):
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}

    # Align offsets (shorter history syms start later in global bar space)
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def _dedup(sym: str, trades: list) -> list:
    """Per-symbol time dedup: reject overlapping (entry/exit) trades."""
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_d_trail(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg_obj: VolStraddleConfig,
) -> list[tuple[str, int, int, float]]:
    """D_mom8 trail-aware collection (G_3TF: 15m/1h/4h)."""
    vr_key = f"vr_{cfg_obj.vr_short}_{cfg_obj.vr_long}"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    sl = cfg_obj.d_sl
    tp = cfg_obj.d_tp  # hard ceiling (0.20), trail exits before this
    trail_act = cfg_obj.d_trail_activate
    trail_dist = cfg_obj.d_trail_distance
    d_vr = cfg_obj.d_vr_threshold

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, mh_d, _mh_s in G_3TF:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_signal(vr, d_vr, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, sl, tp, mh_d, COST_DIRECTIONAL,
                trail_act, trail_dist,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def collect_s_trail(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict,
    cfg_obj: VolStraddleConfig,
) -> list[tuple[str, int, int, float]]:
    """S_d8 trail-aware collection (G_4TF: 15m/30m/1h/4h, matching r66)."""
    vr_key = f"vr_{cfg_obj.vr_short}_{cfg_obj.vr_long}"
    g4tf = _get_g4tf(cfg_obj)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl = cfg_obj.s_sl
    tp = cfg_obj.s_tp
    trail_act = cfg_obj.s_trail_activate
    trail_dist = cfg_obj.s_trail_distance
    s_vr = cfg_obj.s_vr_threshold
    s_dur = cfg_obj.s_squeeze_dur
    mh_15m = cfg_obj.max_hold_15m

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, _mhd, mh_s in g4tf:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            # For 30m, mh_s was specified in TF tuple; otherwise use cfg.max_hold_15m//ratio
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE, trail_act, trail_dist,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def print_metrics(label: str, m: dict, boot: dict) -> None:
    print(f"  {label}")
    print(f"    N={m['N']:>5}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
          f"EV={m['EV_bp']:+.1f}bp  RR={m['RR']:.2f}")
    print(f"    AnnRet={m['ann_ret_pct']:+.1f}%(1x)  DD={m['max_dd_pct']:.1f}%  "
          f"Calmar={m['calmar']:.1f}")
    print(f"    Bootstrap: P(PF≥1.0)={boot['P_PF_ge_1']}%  "
          f"CI=[{boot['CI_lo']:.3f},{boot['CI_hi']:.3f}]")


def print_yearly(yr_data: dict) -> None:
    pos = 0
    for key in sorted(yr_data.keys()):
        y = yr_data[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        if y["PF"] >= 1.0:
            pos += 1
        print(f"    {key}: N={y['N']:>4}  PF={y['PF']:.2f}  "
              f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")
    print(f"    Positive years: {pos}/{len(yr_data)}")


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*78}")
    print(f"  r73: Trail-Aware 3-Strategy Combined Portfolio  [{commit}]")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*78}")

    cfg_obj = VolStraddleConfig()
    vr_ln = cfg_obj.base_leverage / cfg_obj.max_concurrent  # 3.0
    ma_ln = cfg_obj.ma_cluster_leverage / cfg_obj.max_concurrent  # 2.1
    max_concurrent = cfg_obj.max_concurrent  # 10

    print(f"\n  Config (from VolStraddleConfig):")
    print(f"    D_mom8: sl={cfg_obj.d_sl:.3f}  trail_act={cfg_obj.d_trail_activate:.3f}"
          f"  trail_dist={cfg_obj.d_trail_distance:.3f}  d_vr={cfg_obj.d_vr_threshold}")
    print(f"    S_d8: sl={cfg_obj.s_sl:.3f}  tp={cfg_obj.s_tp:.3f}  "
          f"trail_act={cfg_obj.s_trail_activate:.3f}  trail_dist={cfg_obj.s_trail_distance:.3f}")
    print(f"    S_d8: s_vr={cfg_obj.s_vr_threshold}  s_dur={cfg_obj.s_squeeze_dur}")
    print(f"    MA: cluster≤{MA_CLUSTER_PCT*100:.0f}%  sl={MA_SL*100:.0f}%  "
          f"tp={MA_TP*100:.0f}%  flat (no trail sim)")
    print(f"    L/N: VR={vr_ln:.1f}  MA={ma_ln:.1f}  max_concurrent={max_concurrent}")
    print(f"    f5_tbr_enabled={cfg_obj.f5_tbr_enabled}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data (60 symbols, whitelist)...")
    data_15m = load_15m(cfg_obj.symbol_whitelist)
    if not data_15m:
        print("❌ No data loaded.")
        return

    data_30m = build_resampled(data_15m, 2)
    data_1h  = build_resampled(data_15m, 4)
    data_4h  = build_resampled(data_15m, 16)
    g4tf = _get_g4tf(cfg_obj)
    vr_key = f"vr_{cfg_obj.vr_short}_{cfg_obj.vr_long}"

    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg_obj.vr_short, cfg_obj.vr_long)])
        precompute(ds, vr_sp=cfg_obj.vr_short, vr_lp=cfg_obj.vr_long)

    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    bars_list = [d["bars"] if "bars" in d else len(d["close"]) for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  "
          f"{data_years:.2f} years  |  load: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 0: S_d8 trail baseline check (vs r66: N≈7730, PF≈4.948)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 0: S_d8 TRAIL BASELINE (vs r66: N≈7730, PF≈4.948)")
    print(f"{'─'*78}")

    s_trades = collect_s_trail(data_15m, data_1h, data_4h, data_30m, cfg_obj)
    s_pnl = portfolio_sim(s_trades, max_concurrent)
    s_m = compute_metrics(s_pnl, data_years)
    s_boot = bootstrap_pf(s_pnl, 5000)

    n_diff  = abs(s_m["N"] - 7730) / 7730 * 100
    pf_diff = abs(s_m["PF"] - 4.948) / 4.948 * 100
    status  = "✓ PASS" if n_diff < 10.0 and pf_diff < 10.0 else "⚠️ DRIFT"
    print_metrics("S_d8 trail-aware (G_4TF)", s_m, s_boot)
    print(f"    vs r66 (N=7730, PF=4.948): N_diff={n_diff:.1f}%  "
          f"PF_diff={pf_diff:.1f}%  {status}")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: Individual strategy metrics (trail-aware)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 1: INDIVIDUAL STRATEGIES (trail-aware)")
    print(f"{'─'*78}")

    # D_mom8 trail
    print("\n  [Collecting D_mom8 trail trades...]")
    d_trades = collect_d_trail(data_15m, data_1h, data_4h, cfg_obj)
    d_pnl = portfolio_sim(d_trades, max_concurrent)
    d_m = compute_metrics(d_pnl, data_years)
    d_boot = bootstrap_pf(d_pnl, 5000)
    print()
    print_metrics("D_mom8 trail-aware (G_3TF)", d_m, d_boot)

    # S_d8 trail (already collected)
    print()
    print_metrics("S_d8 trail-aware (G_4TF)", s_m, s_boot)

    # MA Cluster (flat TP)
    print("\n  [Collecting MA Cluster trades (flat TP=8%)...]")
    ma_trades = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT,
        min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION,
        min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=MA_TF_RESTRICTION,
    )
    ma_pnl = portfolio_sim(ma_trades, max_concurrent)
    ma_m = compute_metrics(ma_pnl, data_years)
    ma_boot = bootstrap_pf(ma_pnl, 5000)
    print()
    print_metrics("MA Cluster (flat TP=8%, 4h only)", ma_m, ma_boot)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: 3-Strategy Combined (trail-aware)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 2: 3-STRATEGY COMBINED (trail-aware D+S+MA, 10 slots)")
    print(f"{'─'*78}")

    # OLD combined (D+MA, no straddle)
    no_s_trades = d_trades + ma_trades
    no_s_pnl, no_s_accepted = portfolio_sim_with_details(no_s_trades, max_concurrent)
    no_s_m = compute_metrics(no_s_pnl, data_years)
    no_s_boot = bootstrap_pf(no_s_pnl, 5000)
    print()
    print_metrics("OLD combined: D_mom8 + MA (no straddle)", no_s_m, no_s_boot)

    # NEW 3-strat combined
    all_trades = d_trades + s_trades + ma_trades
    comb_pnl, comb_accepted = portfolio_sim_with_details(all_trades, max_concurrent)
    comb_m = compute_metrics(comb_pnl, data_years)
    comb_boot = bootstrap_pf(comb_pnl, 5000)
    print()
    print_metrics("NEW combined: D_mom8 + S_d8 + MA (3-strat)", comb_m, comb_boot)

    # Slot attribution
    d_sym_entries = {(sym, e): "D" for sym, e, _, _ in d_trades}
    s_sym_entries = {(sym, e): "S" for sym, e, _, _ in s_trades}
    ma_sym_entries = {(sym, e): "MA" for sym, e, _, _ in ma_trades}
    sym_entry_map = {**d_sym_entries, **s_sym_entries, **ma_sym_entries}
    accepted_by = {"D": 0, "S": 0, "MA": 0}
    for sym, e, x, _ in comb_accepted:
        label = sym_entry_map.get((sym, e), "?")
        if label in accepted_by:
            accepted_by[label] += 1

    print(f"\n  Slot attribution (combined):")
    d_raw, s_raw, ma_raw = len(d_trades), len(s_trades), len(ma_trades)
    print(f"    D_mom8:   {d_raw:>5} raw → {accepted_by['D']:>5} accepted "
          f"({accepted_by['D']/d_raw*100:.1f}%)")
    print(f"    S_d8:     {s_raw:>5} raw → {accepted_by['S']:>5} accepted "
          f"({accepted_by['S']/s_raw*100:.1f}%)")
    print(f"    MA:       {ma_raw:>5} raw → {accepted_by['MA']:>5} accepted "
          f"({accepted_by['MA']/ma_raw*100:.1f}%)")

    # Year-by-year
    print(f"\n  Per-year (3-strat combined, trail-aware):")
    comb_yr = year_metrics(comb_pnl, comb_accepted, year_bounds)
    print_yearly(comb_yr)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 3: Slot sensitivity & Decision guidance
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 3: OPTION COMPARISON (for structural decision)")
    print(f"{'─'*78}")

    # Option A: current 10-slot (3-strat combined) — already computed
    # Option B: max_concurrent=15 (simulate)
    opt_b_pnl = portfolio_sim(all_trades, 15)
    opt_b_m = compute_metrics(opt_b_pnl, data_years)
    opt_b_boot = bootstrap_pf(opt_b_pnl, 5000)
    print()
    print_metrics("Option B: max_concurrent=15 (3-strat combined)", opt_b_m, opt_b_boot)

    # Slot attribution for Option B
    opt_b_pnl2, opt_b_accepted = portfolio_sim_with_details(all_trades, 15)
    acc_b = {"D": 0, "S": 0, "MA": 0}
    for sym, e, x, _ in opt_b_accepted:
        label = sym_entry_map.get((sym, e), "?")
        if label in acc_b:
            acc_b[label] += 1
    print(f"    D_mom8 in 15-slot: {acc_b['D']} accepted "
          f"({acc_b['D']/d_raw*100:.1f}% — was {accepted_by['D']/d_raw*100:.1f}% in 10-slot)")
    print(f"    S_d8 in 15-slot:   {acc_b['S']} accepted ({acc_b['S']/s_raw*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    #  Summary + Decision
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print("  SUMMARY TABLE (trail-aware)")
    print(f"{'='*78}")
    print(f"  {'Strategy':<42} {'N':>6} {'PF':>7} {'WR':>6} {'EV_bp':>8} "
          f"{'DD%':>6} {'Calmar':>7}")
    print(f"  {'─'*78}")
    rows = [
        ("D_mom8 trail (G_3TF)",              d_m,     d_boot),
        ("S_d8 trail (G_4TF, r66 canonical)", s_m,     s_boot),
        ("MA Cluster (flat TP=8%, 4h)",       ma_m,    ma_boot),
        ("--- OLD: D_mom8 + MA (no straddle)", no_s_m, no_s_boot),
        ("=== NEW: 3-strat, 10-slot",         comb_m,  comb_boot),
        ("=== NEW: 3-strat, 15-slot (Opt B)", opt_b_m, opt_b_boot),
    ]
    for label, m, _ in rows:
        print(f"  {label:<42} {m['N']:>6} {m['PF']:>7.3f} "
              f"{m['WR']:>5.1f}% {m['EV_bp']:>+7.1f} "
              f"{m['max_dd_pct']:>5.1f}% {m['calmar']:>6.1f}")

    # Decision guidance
    threshold = no_s_m["calmar"] * 1.1
    calmar_3s = comb_m["calmar"]
    print(f"\n  Decision framework:")
    print(f"    OLD D+MA Calmar:         {no_s_m['calmar']:.1f}")
    print(f"    3-strat (10-slot) Calmar: {calmar_3s:.1f}")
    print(f"    Threshold (D+MA × 1.1):  {threshold:.1f}")
    if calmar_3s >= threshold:
        print(f"    → ✅ Option A: Accept S_d8 dominance (3-strat Calmar ≥ D+MA × 1.1)")
    else:
        print(f"    → 📊 Option B/C warranted: 3-strat Calmar below threshold")
        print(f"       Option B (15-slot) Calmar: {opt_b_m['calmar']:.1f}")
        if opt_b_m["calmar"] >= threshold:
            print(f"       → ✅ Option B: Increase max_concurrent to 15")
        else:
            print(f"       → ⚠️  Consider Option C: Slot quota system")

    print(f"\n  r72 flat comparison:")
    print(f"    r72 3-strat flat: Calmar=5.6  PF=1.129")
    print(f"    r73 3-strat trail: Calmar={calmar_3s:.1f}  PF={comb_m['PF']:.3f}")
    print(f"    Trail effect: Calmar {calmar_3s/5.6:.1f}x, PF {comb_m['PF']/1.129:.1f}x vs flat sim")

    print(f"\n  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  "
          f"Commit: {commit}  Runtime: {time.time()-t0:.1f}s")
    print(f"  Note: 1x compound (multiply by L/N=3.0 for VR production)")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
