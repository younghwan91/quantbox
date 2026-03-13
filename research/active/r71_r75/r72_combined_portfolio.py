"""r72: 3-Strategy Combined Portfolio Backtest

Answers the question: 기존 백테스팅 결과는 3전략 동시 운용을 가정했는가?

Answer: 아니오. v14 6yr combined = VR(D_mom8) + MA Cluster만 (S_d8 제외).
This script runs all 3 strategies competing for the same 10 slots.

Strategies:
  - D_mom8: VR Squeeze directional (15m/1h/4h)
  - S_d8:   VR Squeeze straddle (15m/1h/4h) — G6 bug fixed in v14.17.0
  - MA Cluster: 4h MA compression breakout

Data: 60 syms · 6yr · Binance USDT-M · 0.08%/0.16% round-trip
Slots: 10 (max_concurrent)
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
    bootstrap_pf,
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r41_r45.r43_exit_improvement import (
    get_year_boundaries,
    year_metrics,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ─── MA Cluster deployed params (R45 champion, unchanged) ─────────────────────
MA_CLUSTER_PCT    = 0.03
MA_MIN_DURATION   = 8
MA_MAX_DURATION   = 24    # zombie filter
MA_MIN_MOMENTUM   = 0.010
MA_SL             = 0.02
MA_TP             = 0.08
MA_TF_RESTRICTION = 16    # 4h only


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_15m(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    data: dict = {}
    for sym in whitelist:
        # Exclude supplementary files (numtrades/quote_volume only)
        candidates = [
            f for f in cache_dir.glob(f"{sym}_15m_*.npz")
            if "numtrades" not in f.name
        ]
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)
        if not {"close", "high", "low", "volume"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {
            "close": c,
            "high": npz["high"].astype(np.float64),
            "low": npz["low"].astype(np.float64),
            "volume": npz["volume"].astype(np.float64),
            "bars": len(c),
            "days": _extract_days(best),
        }
    return data


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def print_strat(label: str, m: dict, boot: dict) -> None:
    print(f"  {label}")
    print(f"    N={m['N']:>5}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
          f"EV={m['EV_bp']:+.1f}bp  RR={m['RR']:.2f}")
    print(f"    AnnRet={m['ann_ret_pct']:+.1f}% (1x)  "
          f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}  "
          f"TPD={m['tpd']:.2f}")
    print(f"    Bootstrap: P(PF≥1.0)={boot['P_PF_ge_1']}%  "
          f"CI=[{boot['CI_lo']:.3f},{boot['CI_hi']:.3f}]  "
          f"median={boot['median']:.3f}")


def print_yearly(yr_data: dict) -> None:
    pos = 0
    for key in sorted(yr_data.keys()):
        y = yr_data[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        if y["PF"] >= 1.0:
            pos += 1
        print(f"    {key}: N={y['N']:>4}  PF={y['PF']:.2f}  "
              f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  "
              f"ret={y.get('ret_pct', 0):>+6.1f}%  {flag}")
    print(f"    Positive years: {pos}/{len(yr_data)}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*78}")
    print(f"  r72: 3-Strategy Combined Portfolio Backtest  [{commit}]")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*78}")

    # ── Config (always read from source of truth) ─────────────────────────
    cfg_obj = VolStraddleConfig()
    cfg = {
        "sp":    cfg_obj.vr_short,
        "lp":    cfg_obj.vr_long,
        "d_vr":  cfg_obj.d_vr_threshold,
        "s_vr":  cfg_obj.s_vr_threshold,
        "s_dur": cfg_obj.s_squeeze_dur,
        "d_sl":  cfg_obj.d_sl,
        "s_sl":  cfg_obj.s_sl,
        # Use s_tp (straddle-specific TP = 5%) for flat sim.
        # d_tp = 0.20 is a dead param in production (trail exits first).
        # Using s_tp=0.05 here to match the canonical baseline approach.
        "tp":    cfg_obj.s_tp,
    }
    max_concurrent = cfg_obj.max_concurrent  # 10
    vr_ln = cfg_obj.base_leverage / max_concurrent  # 3.0
    ma_ln = cfg_obj.ma_cluster_leverage / max_concurrent  # 2.1

    print(f"\n  Config (from VolStraddleConfig):")
    print(f"    VR sp={cfg['sp']} lp={cfg['lp']}  d_vr={cfg['d_vr']}  "
          f"d_sl={cfg['d_sl']*100:.0f}%  tp={cfg['tp']*100:.0f}%")
    print(f"    S d8: s_vr={cfg['s_vr']}  s_dur={cfg['s_dur']}  "
          f"s_sl={cfg['s_sl']*100:.1f}%")
    print(f"    MA: cluster≤{MA_CLUSTER_PCT*100:.0f}%  dur={MA_MIN_DURATION}-{MA_MAX_DURATION}  "
          f"mom≥{MA_MIN_MOMENTUM:.3f}  sl={MA_SL*100:.0f}%  tp={MA_TP*100:.0f}%  4h-only")
    print(f"    L/N: VR={vr_ln:.1f}  MA={ma_ln:.1f}  trail_activate={cfg_obj.d_trail_activate*100:.1f}%")
    print(f"    f5_tbr_enabled={cfg_obj.f5_tbr_enabled}")

    # ── Load data (60 symbols from whitelist) ─────────────────────────────
    print("\nLoading data (60 symbols)...")
    data_15m = load_15m(cfg_obj.symbol_whitelist)

    if not data_15m:
        print("❌ No data loaded. Check data/cache_extended/")
        return

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    # Precompute VR signals
    precompute_all_vr([data_15m, data_1h, data_4h], [(cfg["sp"], cfg["lp"])])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=cfg["sp"], vr_lp=cfg["lp"])

    # Precompute MAs for MA Cluster
    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    bars_list = [d["bars"] for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  "
          f"{data_years:.2f} years  |  load: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 0: S_d8 Baseline Check (vs r66: N≈7730, PF≈4.948)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 0: S_d8 BASELINE (must match r66 ±5%: N≈7730, PF≈4.948)")
    print(f"{'─'*78}")

    s_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
    s_pnl = portfolio_sim(s_trades, max_concurrent)
    s_m = compute_metrics(s_pnl, data_years)

    n_diff  = abs(s_m["N"] - 7730) / 7730 * 100
    pf_diff = abs(s_m["PF"] - 4.948) / 4.948 * 100
    status  = "✓ PASS" if n_diff < 5.0 and pf_diff < 5.0 else "❌ DRIFT"
    print(f"  S_d8 alone: N={s_m['N']}  PF={s_m['PF']:.3f}  "
          f"WR={s_m['WR']:.1f}%  EV={s_m['EV_bp']:+.1f}bp")
    print(f"  vs r66 (N=7730, PF=4.948): N_diff={n_diff:.1f}%  PF_diff={pf_diff:.1f}%  {status}")

    if n_diff >= 5.0 or pf_diff >= 5.0:
        print("\n⚠️  Baseline drift — checking config mismatch. Continuing anyway.")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: Each Strategy Individually
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 1: INDIVIDUAL STRATEGY METRICS")
    print(f"{'─'*78}")

    # D_mom8
    d_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
    d_pnl = portfolio_sim(d_trades, max_concurrent)
    d_m = compute_metrics(d_pnl, data_years)
    d_boot = bootstrap_pf(d_pnl, 5000)
    print_strat("D_mom8 (directional VR, flat TP)", d_m, d_boot)

    # S_d8
    s_boot = bootstrap_pf(s_pnl, 5000)
    print()
    print_strat("S_d8 Straddle (LONG+SHORT simultaneous)", s_m, s_boot)

    # VR combined (D_mom8 + S_d8) — canonical baseline check
    vr_trades = d_trades + s_trades
    vr_pnl = portfolio_sim(vr_trades, max_concurrent)
    vr_m = compute_metrics(vr_pnl, data_years)
    vr_boot = bootstrap_pf(vr_pnl, 5000)
    vr_n_diff  = abs(vr_m["N"] - 6207) / 6207 * 100
    vr_pf_diff = abs(vr_m["PF"] - 1.563) / 1.563 * 100
    vr_status  = "✓ PASS" if vr_n_diff < 5.0 and vr_pf_diff < 5.0 else "⚠️ DRIFT"
    print()
    print_strat("VR combined (D_mom8 + S_d8) — canonical", vr_m, vr_boot)
    print(f"    vs v14.5.0 canonical (N=6207, PF=1.563): "
          f"N_diff={vr_n_diff:.1f}%  PF_diff={vr_pf_diff:.1f}%  {vr_status}")

    # MA Cluster
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
    print_strat("MA Cluster (4h only, R45 champion params)", ma_m, ma_boot)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: 3-Strategy Combined Portfolio
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 2: 3-STRATEGY COMBINED (D_mom8 + S_d8 + MA Cluster, 10 slots)")
    print(f"{'─'*78}")

    all_trades = d_trades + s_trades + ma_trades
    comb_pnl, comb_accepted = portfolio_sim_with_details(all_trades, max_concurrent)
    comb_m = compute_metrics(comb_pnl, data_years)
    comb_boot = bootstrap_pf(comb_pnl, 5000)

    print_strat("Combined (D_mom8 + S_d8 + MA)", comb_m, comb_boot)

    print(f"\n  Note: straddle modeled as 1 slot (sim). Live: 2 slots (LONG+SHORT).")
    print(f"  Trade mix: D_mom8={len(d_trades)} raw → S_d8={len(s_trades)} raw → "
          f"MA={len(ma_trades)} raw → total raw={len(all_trades)}")

    accepted_by_strat = {"D_mom8": 0, "S_d8": 0, "MA": 0}
    d_sym_entries = {(sym, e): "D_mom8" for sym, e, _, _ in d_trades}
    s_sym_entries = {(sym, e): "S_d8" for sym, e, _, _ in s_trades}
    ma_sym_entries = {(sym, e): "MA" for sym, e, _, _ in ma_trades}
    sym_entry_map = {**d_sym_entries, **s_sym_entries, **ma_sym_entries}
    for sym, e, x, pnl_val in comb_accepted:
        key = (sym, e)
        strat_label = sym_entry_map.get(key, "unknown")
        if strat_label in accepted_by_strat:
            accepted_by_strat[strat_label] += 1
    print(f"  Accepted: D_mom8={accepted_by_strat['D_mom8']}  "
          f"S_d8={accepted_by_strat['S_d8']}  "
          f"MA={accepted_by_strat['MA']}  total={comb_m['N']}")

    # Per-year breakdown
    print(f"\n  Per-year (Combined 3 strategies):")
    comb_yr = year_metrics(comb_pnl, comb_accepted, year_bounds)
    print_yearly(comb_yr)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 3: Straddle Slot Sensitivity
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print("  PHASE 3: STRADDLE SLOT SENSITIVITY")
    print(f"  Live behavior: straddle uses 2 slots (LONG+SHORT separately)")
    print(f"  Backtest default: straddle uses 1 slot (combined PnL per symbol)")
    print(f"{'─'*78}")

    # Model straddle as 2 slots by duplicating the straddle trades
    # (same sym, same entry/exit bar — portfolio_sim will count them as 2 active)
    s_doubled = []
    for sym, entry, exit_b, pnl_val in s_trades:
        # Each leg is modeled as separate slots: split PnL by 2 per leg
        s_doubled.append((sym + "_LONG", entry, exit_b, pnl_val / 2))
        s_doubled.append((sym + "_SHRT", entry, exit_b, pnl_val / 2))

    all_trades_2slot = d_trades + s_doubled + ma_trades
    comb2_pnl = portfolio_sim(all_trades_2slot, max_concurrent)
    comb2_m = compute_metrics(comb2_pnl, data_years)
    comb2_boot = bootstrap_pf(comb2_pnl, 5000)

    print_strat("Combined (straddle=2 slots, pnl/2 per leg)", comb2_m, comb2_boot)

    # Without straddle (old combined: D_mom8 + MA only)
    no_straddle = d_trades + ma_trades
    ns_pnl = portfolio_sim(no_straddle, max_concurrent)
    ns_m = compute_metrics(ns_pnl, data_years)
    ns_boot = bootstrap_pf(ns_pnl, 5000)
    print()
    print_strat("Without straddle (D_mom8 + MA only — old combined)", ns_m, ns_boot)

    # ══════════════════════════════════════════════════════════════════════
    #  Summary Table
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print("  SUMMARY TABLE")
    print(f"{'='*78}")
    print(f"  {'Strategy':<42} {'N':>6} {'PF':>7} {'WR':>6} {'EV_bp':>8} "
          f"{'DD%':>6} {'Calmar':>7}")
    print(f"  {'─'*78}")
    rows = [
        ("D_mom8 (directional VR)",         d_m,     d_boot),
        ("S_d8 Straddle",                   s_m,     s_boot),
        ("MA Cluster (4h, R45)",            ma_m,    ma_boot),
        ("VR combined (D+S) — canonical",   vr_m,    vr_boot),
        ("--- OLD combined (D+MA) ---",     ns_m,    ns_boot),
        ("=== NEW 3-strat (D+S+MA, 1-slot)", comb_m, comb_boot),
        ("=== NEW 3-strat (D+S+MA, 2-slot)", comb2_m, comb2_boot),
    ]
    for label, m, _ in rows:
        print(f"  {label:<42} {m['N']:>6} {m['PF']:>7.3f} "
              f"{m['WR']:>5.1f}% {m['EV_bp']:>+7.1f} "
              f"{m['max_dd_pct']:>5.1f}% {m['calmar']:>6.1f}")

    print(f"\n  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  "
          f"Commit: {commit}  Runtime: {time.time()-t0:.1f}s")
    print(f"  Note: All metrics are 1x compound. Multiply by L/N for production.")
    print(f"  VR L/N={vr_ln:.1f} → 3-strat combined AnnRet ≈ "
          f"{comb_m['ann_ret_pct'] * vr_ln:.1f}% (approx, blended L/N)")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
