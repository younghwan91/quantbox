#!/usr/bin/env python3
"""r116b: Combined D_mom8 blacklist × conviction sweep.

Takes top-2 blacklist variants × top-2 conviction variants from the prior sweeps
and runs all 4 combinations to identify best single config. Produces re-enable
recommendation if best combo PF ≥ 1.3.

Usage:
    uv run python -m research.active.r116_willy_nfi.r116b_combined_sweep

(ref: r116 plan Section 3, US-R116B-4)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled,
    compute_metrics,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r116_willy_nfi.r116_prep_baseline import (
    DATA_YEARS,
    STATE_DIR,
    _compute_sharpe,
    compute_mfe_losing,
    load_15m_data,
)
from research.active.r116_willy_nfi.r116b_conviction_sweep import (
    collect_tagged_trades,
    filter_by_min,
)
from research.active.r116_willy_nfi.r116b_symbol_blacklist_sweep import (
    VARIANTS as BLACKLIST_VARIANTS,
    compute_blacklist,
)

OUTPUT_DIR = ROOT / ".omc" / "state" / "r116b"
BASELINE_PATH = STATE_DIR / "vr_directional_metrics.json"

CFG = VolStraddleConfig()


def load_blacklist_summary() -> list[dict]:
    """Read per-variant blacklist metrics from prior sweep output."""
    variants = []
    for name, pf_th, min_n in BLACKLIST_VARIANTS:
        p = OUTPUT_DIR / f"blacklist_{name}_metrics.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
            data["_pf_threshold"] = pf_th
            data["_min_n"] = min_n
            variants.append(data)
    return variants


def load_conviction_summary() -> list[dict]:
    variants = []
    for ms in [0, 1, 2, 3]:
        p = OUTPUT_DIR / f"conviction_{ms}_metrics.json"
        if not p.exists():
            continue
        with open(p) as f:
            variants.append(json.load(f))
    return variants


def select_top_n(variants: list[dict], n: int, key: str = "PF") -> list[dict]:
    return sorted(variants, key=lambda v: v.get(key, 0), reverse=True)[:n]


def run_combo(
    blacklist: set[str],
    min_conviction: int,
    tagged_trades: list[tuple],
    data_15m: dict,
) -> dict:
    t0 = time.time()
    # Apply blacklist
    bl_filtered = [t for t in tagged_trades if t[0] not in blacklist]
    # Apply conviction gate
    final = filter_by_min(bl_filtered, min_conviction)
    pnl_arr, accepted = portfolio_sim_with_details(final, CFG.max_concurrent)
    metrics = compute_metrics(pnl_arr, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    mfe_losing = compute_mfe_losing(accepted, data_15m) if accepted else 0.0

    return {
        "combo_name": f"bl{len(blacklist)}_min{min_conviction}",
        "blacklist_count": len(blacklist),
        "blacklist": sorted(blacklist),
        "min_conviction": min_conviction,
        "after_blacklist": len(bl_filtered),
        "after_conviction": len(final),
        "N": metrics["N"],
        "WR_pct": metrics["WR"],
        "PF": metrics["PF"],
        "MDD_pct": metrics["max_dd_pct"],
        "avg_win_pct": metrics.get("avg_win_pct", 0.0),
        "avg_loss_pct": metrics.get("avg_loss_pct", 0.0),
        "sharpe": _compute_sharpe(pnl_arr),
        "calmar": metrics["calmar"],
        "ann_ret_pct": metrics["ann_ret_pct"],
        "mfe_losing_avg_pct": round(mfe_losing, 3),
        "elapsed_s": round(time.time() - t0, 1),
    }


def write_summary_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "combo", "blacklist_n", "min_conv", "N", "WR_pct", "PF",
            "MDD_pct", "Sharpe", "Calmar", "ann_ret_pct",
        ])
        for r in results:
            w.writerow([
                r["combo_name"], r["blacklist_count"], r["min_conviction"],
                r["N"], r["WR_pct"], r["PF"], r["MDD_pct"],
                r["sharpe"], r["calmar"], r["ann_ret_pct"],
            ])


def main() -> None:
    print("=" * 78)
    print("r116b — Combined Blacklist × Conviction Sweep")
    print("=" * 78)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    baseline_pf = baseline["PF"]
    baseline_mdd = baseline["MDD_pct"]
    print(f"Baseline:  PF={baseline_pf}  MDD={baseline_mdd}%  N={baseline['N']}")

    # Select top-2 blacklist × top-2 conviction
    bl_variants = load_blacklist_summary()
    conv_variants = load_conviction_summary()

    top_bl = select_top_n(bl_variants, 2)
    top_conv = select_top_n(conv_variants, 2)

    print("\nTop-2 blacklist variants:")
    for v in top_bl:
        print(f"  {v['variant']}: PF={v['PF']} MDD={v['MDD_pct']}% ({v['blacklist_count']} syms)")
    print("\nTop-2 conviction variants:")
    for v in top_conv:
        print(f"  min={v['min_conviction_score']}: PF={v['PF']} MDD={v['MDD_pct']}%")

    # Load data + precompute
    print("\nLoading data + precomputing...")
    t0 = time.time()
    data_15m = load_15m_data(CFG.symbol_whitelist)
    sp, lp = CFG.vr_short, CFG.vr_long
    precompute_all_vr([data_15m], [(sp, lp)])
    precompute(data_15m, vr_sp=sp, vr_lp=lp, mom_lb=CFG.momentum_period)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_1h, data_4h], [(sp, lp)])
    for ds in (data_1h, data_4h):
        precompute(ds, vr_sp=sp, vr_lp=lp, mom_lb=CFG.momentum_period)
    print(f"Precompute elapsed {time.time()-t0:.1f}s")

    # Tag trades once (no filters)
    print("\nCollecting tagged trades...")
    ct_cfg = {
        "sp": CFG.vr_short,
        "lp": CFG.vr_long,
        "d_vr": CFG.d_vr_threshold,
        "d_sl": CFG.d_sl,
        "tp": CFG.d_tp,
        "d_trail_activate": CFG.d_trail_activate,
        "d_trail_distance": CFG.d_trail_distance,
    }
    # Load per_symbol to compute blacklist sets
    per_symbol = baseline["per_symbol"]

    # Tag all trades
    tagged_trades = collect_tagged_trades(ct_cfg, data_15m, data_1h, data_4h)
    print(f"Raw tagged: {len(tagged_trades)}")

    # Run 4 combos (top-2 × top-2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for bl_v in top_bl:
        blacklist = compute_blacklist(per_symbol, bl_v["_pf_threshold"], bl_v["_min_n"])
        for conv_v in top_conv:
            min_score = conv_v["min_conviction_score"]
            print(f"\n--- combo: blacklist='{bl_v['variant']}' × min_conv={min_score} ---")
            res = run_combo(blacklist, min_score, tagged_trades, data_15m)
            res["blacklist_variant"] = bl_v["variant"]
            results.append(res)

            out_path = OUTPUT_DIR / f"combined_{bl_v['variant']}_min{min_score}_metrics.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)

            print(
                f"  after_bl={res['after_blacklist']}  "
                f"after_conv={res['after_conviction']}  "
                f"N={res['N']}  WR={res['WR_pct']}%  PF={res['PF']}  "
                f"MDD={res['MDD_pct']}%"
            )

    summary_path = OUTPUT_DIR / "combined_summary.csv"
    write_summary_csv(results, summary_path)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'combo':<18} {'N':>5} {'WR%':>6} {'PF':>6} {'MDD%':>6} {'Sharpe':>7} {'ΔPF':>7}")
    for r in results:
        delta = r["PF"] - baseline_pf
        print(
            f"{r['combo_name']:<18} {r['N']:>5} {r['WR_pct']:>6.1f} {r['PF']:>6.3f} "
            f"{r['MDD_pct']:>6.1f} {r['sharpe']:>7.2f} {delta:>+7.3f}"
        )

    # Best combo: highest PF with MDD constraint
    valid = [r for r in results if r["MDD_pct"] <= baseline_mdd + 3.0]
    if not valid:
        print("\n✗ No combo satisfies MDD constraint (≤ baseline + 3pp)")
        best = max(results, key=lambda r: r["PF"])
    else:
        best = max(valid, key=lambda r: r["PF"])

    print("\n" + "=" * 78)
    print(f"BEST combo: {best['combo_name']}")
    print(f"  Blacklist variant: {best.get('blacklist_variant', '?')}")
    print(f"  Blacklist size: {best['blacklist_count']}")
    print(f"  min_conviction: {best['min_conviction']}")
    print(f"  PF={best['PF']} (Δ {best['PF']-baseline_pf:+.3f})  MDD={best['MDD_pct']}%")
    print(f"  N={best['N']}  WR={best['WR_pct']}%  Sharpe={best['sharpe']}")

    if best["PF"] >= 1.3:
        print("\n✓ RE-ENABLE CANDIDATE")
        print("  Proposed .env config:")
        if best["blacklist_count"] > 0:
            print(f"    # D_mom8 symbol blacklist (PF<{bl_v['_pf_threshold']} AND N>={bl_v['_min_n']} from 2024-01 baseline)")
            print(f"    # {best['blacklist_count']} symbols: {best['blacklist'][:5]}...")
        print(f"    VS_MIN_CONVICTION_SCORE={best['min_conviction']}")
        print("    # (remove VS_MIN_CONVICTION_SCORE=5 pause flag)")
    else:
        print("\n✗ MAINTAIN PAUSE — no combo reaches PF ≥ 1.3")
        print("  Top-3 combos for future investigation:")
        top3 = sorted(results, key=lambda r: r["PF"], reverse=True)[:3]
        for r in top3:
            print(f"    {r['combo_name']}: PF={r['PF']} MDD={r['MDD_pct']}%")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
