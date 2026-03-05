"""v14.0.0 Deployed Configuration — 6-Year Full Backtest

Runs the exact currently deployed parameters for both strategies:
  - VR Squeeze (CFG_V13): sp=8, lp=56, vr<0.50, sl=1%, tp=5%, s_sl=1.5%, s_dur=4
  - MA Cluster (R45):     cluster≤3%, dur≥8, max_dur=24, mom≥0.010, sl=2%, tp=8%, 4h-only

No parameter sweep — just the production numbers.
Outputs: per-strategy metrics, per-year breakdown, bootstrap CI, combined portfolio.
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
    build_resampled,
    precompute,
    precompute_all_vr,
    collect_trades,
    portfolio_sim,
    portfolio_sim_with_details,
    compute_metrics,
    bootstrap_pf,
    pf_from_pnl,
)
from research.active.r41_r45.r45_param_sweep import (
    collect_enhanced_trades,
    sim_ma_cluster_enhanced,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics,
    get_year_boundaries,
)
from research.active.r31_r40.scaling_r38 import CFG_V13


# ─── Local data loader (uses correct project ROOT) ────────────────────────────
def _extract_days(path: Path) -> int:
    """Extract days count from filename like BTCUSDT_15m_2190d_20260220.npz."""
    stem = path.stem
    for part in stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_15m() -> dict:
    """Load 15m NPZ data using correct project ROOT (data/cache_extended/)."""
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings
    settings = QuantBoxSettings()
    whitelist = [s.strip() for s in settings.rb_symbol_whitelist.split(",") if s.strip()]
    data: dict = {}
    for sym in whitelist:
        candidates = list(cache_dir.glob(f"{sym}_15m_*.npz"))
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)
        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        h = npz["high"].astype(np.float64)
        lo = npz["low"].astype(np.float64)
        v = npz["volume"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {
            "close": c, "high": h, "low": lo, "volume": v,
            "bars": len(c), "days": _extract_days(best),
        }
    return data


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def print_strategy(label: str, m: dict, boot: dict) -> None:
    print(f"  {label}")
    print(f"    N={m['N']:>5}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
          f"EV={m['EV_bp']:+.1f}bp  RR={m['RR']:.2f}")
    print(f"    AnnRet={m['ann_ret_pct']:+.1f}% (1x)  "
          f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}  "
          f"TPD={m['tpd']:.2f}")
    print(f"    Bootstrap: P(PF≥1.0)={boot['P_PF_ge_1']}%  "
          f"CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
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


# ═══════════════════════════════════════════════════════════════════════════
#  Deployed MA Cluster params (R45 champion)
# ═══════════════════════════════════════════════════════════════════════════

MA_CLUSTER_PCT   = 0.03
MA_MIN_DURATION  = 8
MA_MAX_DURATION  = 24    # zombie filter (R45)
MA_MIN_MOMENTUM  = 0.010 # momentum filter (R45)
MA_SL            = 0.02
MA_TP            = 0.08
MA_TF_RESTRICTION = 16   # 4h only


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*78}")
    print(f"  v14.0.0 Deployed Config — 6-Year Backtest  [{commit}]")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*78}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    data_15m = load_15m()

    if not data_15m:
        print("❌ No data loaded. Check data/cache_extended/")
        return

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    # Precompute VR (only sp=8, lp=56)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)

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

    # ── Phase 1: Baseline check ────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("  PHASE 1: BASELINE CHECK (must match report ±1%)")
    print(f"{'─'*78}")

    bl_vr_d = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_vr_s = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_vr_trades = bl_vr_d + bl_vr_s
    bl_vr_pnl = portfolio_sim(bl_vr_trades, 10)
    bl_vr = compute_metrics(bl_vr_pnl, data_years)

    vr_pf_diff = abs(bl_vr["PF"] - 1.280) / 1.280 * 100
    vr_n_diff  = abs(bl_vr["N"] - 2265) / 2265 * 100

    print(f"  VR Squeeze: N={bl_vr['N']}  PF={bl_vr['PF']:.3f}  WR={bl_vr['WR']:.1f}%")
    print(f"    vs report (N=2265, PF=1.280): "
          f"N_diff={vr_n_diff:.1f}%  PF_diff={vr_pf_diff:.1f}%  "
          f"{'✓ PASS' if vr_pf_diff < 1.0 else '❌ FAIL'}")

    if vr_pf_diff >= 1.0 or vr_n_diff >= 3.0:
        print("\n❌ BASELINE MISMATCH — data or code changed. Aborting.")
        return

    # ── Phase 2: Deployed MA Cluster ───────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  PHASE 2: MA CLUSTER (R45 deployed params)")
    print(f"  cluster≤{MA_CLUSTER_PCT*100:.0f}%  dur≥{MA_MIN_DURATION}  "
          f"max_dur={MA_MAX_DURATION}  mom≥{MA_MIN_MOMENTUM:.3f}  "
          f"sl={MA_SL*100:.0f}%  tp={MA_TP*100:.0f}%  4h-only")
    print(f"{'─'*78}")

    ma_trades = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT,
        min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION,
        min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=MA_TF_RESTRICTION,
    )
    ma_pnl = portfolio_sim(ma_trades, 10)
    ma_m   = compute_metrics(ma_pnl, data_years)
    ma_boot = bootstrap_pf(ma_pnl, 5000)

    print_strategy("MA Cluster R45", ma_m, ma_boot)

    # ── Phase 3: VR Squeeze ────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  PHASE 3: VR SQUEEZE (CFG_V13 — D+S combined)")
    print(f"  sp=8 lp=56 vr<0.50 d_sl=1% s_sl=1.5% tp=5% s_dur=4")
    print(f"{'─'*78}")

    vr_pnl  = bl_vr_pnl
    vr_m    = bl_vr
    vr_boot = bootstrap_pf(vr_pnl, 5000)

    print_strategy("VR Squeeze CFG_V13 (D+S)", vr_m, vr_boot)

    # ── Phase 4: Combined portfolio ────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  PHASE 4: COMBINED PORTFOLIO (VR + MA, 10 slots)")
    print(f"{'─'*78}")

    comb_all = bl_vr_trades + ma_trades
    comb_pnl, comb_accepted = portfolio_sim_with_details(comb_all, 10)
    comb_m    = compute_metrics(comb_pnl, data_years)
    comb_boot = bootstrap_pf(comb_pnl, 5000)

    print_strategy("Combined v14.0.0", comb_m, comb_boot)

    # Per-year breakdown
    print()
    print("  Per-year:")
    comb_yr = year_metrics(comb_pnl, comb_accepted, year_bounds)
    print_yearly(comb_yr)

    # ── Phase 5: VR per-year breakdown ────────────────────────────────────
    print()
    print("  VR Squeeze — per-year:")
    vr_accepted_all = portfolio_sim_with_details(bl_vr_trades, 10)[1]
    vr_yr = year_metrics(vr_pnl, vr_accepted_all, year_bounds)
    print_yearly(vr_yr)

    # ── Phase 6: MA per-year breakdown ────────────────────────────────────
    print()
    print("  MA Cluster — per-year:")
    ma_accepted_all = portfolio_sim_with_details(ma_trades, 10)[1]
    ma_yr = year_metrics(ma_pnl, ma_accepted_all, year_bounds)
    print_yearly(ma_yr)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  SUMMARY — v14.0.0 Invariant Edge (leverage + period independent)")
    print(f"{'='*78}")
    print(f"  {'Strategy':<28} {'N':>6} {'PF':>7} {'WR':>6} {'EV_bp':>8} "
          f"{'DD%':>6} {'Calmar':>7} {'P(PF≥1)':>9}")
    print(f"  {'─'*75}")
    for label, m, boot in [
        ("VR Squeeze (D+S)", vr_m, vr_boot),
        (f"MA Cluster (R45)", ma_m, ma_boot),
        ("Combined v14.0.0", comb_m, comb_boot),
    ]:
        print(f"  {label:<28} {m['N']:>6} {m['PF']:>7.3f} {m['WR']:>5.1f}% "
              f"{m['EV_bp']:>+7.1f} {m['max_dd_pct']:>5.1f}% "
              f"{m['calmar']:>6.1f} {boot['P_PF_ge_1']:>8.1f}%")

    print(f"\n  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  "
          f"Commit: {commit}  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
