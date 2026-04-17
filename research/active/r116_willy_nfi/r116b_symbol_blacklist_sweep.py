#!/usr/bin/env python3
"""r116b: D_mom8 symbol blacklist sweep.

For each blacklist variant, re-run D_mom8 baseline with those symbols excluded.
Compare PF/WR/MDD vs baseline to identify whether symbol selection alone can lift
D_mom8 from PF=1.092 (baseline) to PF≥1.3 (re-enable target).

Variants:
    empty:   no blacklist (control — should match baseline)
    tight:   PF < 0.80 AND N >= 10
    medium:  PF < 0.90 AND N >= 10
    loose:   PF < 1.00 AND N >= 10

Rule: never tighten SL in optimization. Blacklist only prunes symbols; does not
touch SL/TP/trail params.

Usage:
    uv run python -m research.active.r116_willy_nfi.r116b_symbol_blacklist_sweep

(ref: r116 plan Section 3, US-R116B-2)
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
    G_4TF,
    build_resampled,
    collect_trades,
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

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / ".omc" / "state" / "r116b"
BASELINE_PATH = STATE_DIR / "vr_directional_metrics.json"


# ── Variant definitions ──────────────────────────────────────────────────────
VARIANTS = [
    # (name, pf_threshold, min_N)
    ("empty", 0.0, 0),  # no blacklist
    ("tight", 0.80, 10),
    ("medium", 0.90, 10),
    ("loose", 1.00, 10),
]


def load_per_symbol_baseline() -> list[dict]:
    with open(BASELINE_PATH) as f:
        return json.load(f)["per_symbol"]


def compute_blacklist(per_symbol: list[dict], pf_threshold: float, min_n: int) -> set[str]:
    """Return symbols with PF < pf_threshold AND N >= min_n."""
    if pf_threshold <= 0:
        return set()
    return {s["symbol"] for s in per_symbol if s["PF"] < pf_threshold and s["N"] >= min_n}


def run_variant(
    variant_name: str,
    blacklist: set[str],
    cfg: VolStraddleConfig,
) -> dict:
    """Run D_mom8 backtest with given symbols excluded.

    BTCUSDT is always retained (needed for BTC momentum gate) even if blacklisted.
    """
    t0 = time.time()

    # Filter symbols: keep whitelist - blacklist, always include BTCUSDT
    symbols = [s for s in cfg.symbol_whitelist if s not in blacklist or s == "BTCUSDT"]
    data_15m = load_15m_data(symbols)

    # Double-check: remove blacklisted symbols from data_15m except BTCUSDT
    for s in list(data_15m.keys()):
        if s in blacklist and s != "BTCUSDT":
            del data_15m[s]

    sp, lp = cfg.vr_short, cfg.vr_long
    precompute_all_vr([data_15m], [(sp, lp)])
    precompute(data_15m, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_1h, data_4h], [(sp, lp)])
    for ds in (data_1h, data_4h):
        precompute(ds, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    ct_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "d_sl": cfg.d_sl,
        "tp": cfg.d_tp,
        "d_trail_activate": cfg.d_trail_activate,
        "d_trail_distance": cfg.d_trail_distance,
    }

    all_trades = collect_trades(ct_cfg, data_15m, data_1h, data_4h, "D_mom8", tf_group=G_4TF)

    # Post-filter: drop any trades on blacklisted symbols (belt-and-suspenders
    # in case collect_trades internally touches data_15m symbols we didn't prune)
    if blacklist:
        all_trades = [t for t in all_trades if t[0] not in blacklist]

    pnl_arr, accepted = portfolio_sim_with_details(all_trades, cfg.max_concurrent)
    metrics = compute_metrics(pnl_arr, DATA_YEARS, portfolio_size=cfg.max_concurrent)
    mfe_losing = compute_mfe_losing(accepted, data_15m) if accepted else 0.0

    return {
        "variant": variant_name,
        "blacklist": sorted(blacklist),
        "blacklist_count": len(blacklist),
        "symbols_kept": len([s for s in cfg.symbol_whitelist if s not in blacklist]),
        "raw_trades": len(all_trades),
        "N": metrics["N"],
        "WR_pct": metrics["WR"],
        "PF": metrics["PF"],
        "MDD_pct": metrics["max_dd_pct"],
        "avg_win_pct": metrics["avg_win_pct"],
        "avg_loss_pct": metrics["avg_loss_pct"],
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
            "variant", "blacklist_count", "symbols_kept", "N", "WR_pct", "PF",
            "MDD_pct", "Sharpe", "Calmar", "ann_ret_pct", "mfe_losing_avg_pct",
        ])
        for r in results:
            w.writerow([
                r["variant"], r["blacklist_count"], r["symbols_kept"],
                r["N"], r["WR_pct"], r["PF"], r["MDD_pct"],
                r["sharpe"], r["calmar"], r["ann_ret_pct"], r["mfe_losing_avg_pct"],
            ])


def identify_best(results: list[dict], baseline_mdd: float) -> dict | None:
    """Pick highest PF among variants with MDD ≤ baseline_MDD + 3pp (no SL regression)."""
    candidates = [r for r in results if r["MDD_pct"] <= baseline_mdd + 3.0]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["PF"])


def main() -> None:
    print("=" * 78)
    print("r116b — D_mom8 Symbol Blacklist Sweep")
    print("=" * 78)

    cfg = VolStraddleConfig()
    per_symbol = load_per_symbol_baseline()

    # Load baseline portfolio metrics for comparison
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    baseline_mdd = baseline["MDD_pct"]
    baseline_pf = baseline["PF"]

    print(f"Baseline:  PF={baseline_pf}  MDD={baseline_mdd}%  N={baseline['N']}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for name, pf_th, min_n in VARIANTS:
        blacklist = compute_blacklist(per_symbol, pf_th, min_n)
        print(f"\n--- Variant '{name}' (PF<{pf_th}, N>={min_n}) ---")
        print(f"  Blacklist ({len(blacklist)}): {sorted(blacklist)}")

        result = run_variant(name, blacklist, cfg)
        results.append(result)

        out_path = OUTPUT_DIR / f"blacklist_{name}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"  N={result['N']}  WR={result['WR_pct']}%  PF={result['PF']}  "
            f"MDD={result['MDD_pct']}%  Sharpe={result['sharpe']}  "
            f"Calmar={result['calmar']}"
        )
        print(f"  → {out_path.name}")

    summary_path = OUTPUT_DIR / "blacklist_summary.csv"
    write_summary_csv(results, summary_path)
    print(f"\nSummary CSV: {summary_path}")

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'variant':<10} {'N':>5} {'WR%':>6} {'PF':>6} {'MDD%':>6} {'Sharpe':>7} {'ΔPF':>7}")
    for r in results:
        delta = r["PF"] - baseline_pf
        print(
            f"{r['variant']:<10} {r['N']:>5} {r['WR_pct']:>6.1f} {r['PF']:>6.3f} "
            f"{r['MDD_pct']:>6.1f} {r['sharpe']:>7.2f} {delta:>+7.3f}"
        )

    best = identify_best(results, baseline_mdd)
    print("\n" + "=" * 78)
    if best:
        print(f"BEST variant: '{best['variant']}'")
        print(f"  PF={best['PF']} (Δ {best['PF']-baseline_pf:+.3f})  MDD={best['MDD_pct']}%")
        if best["PF"] >= 1.3:
            print("  ✓ Exceeds re-enable threshold PF ≥ 1.3")
        else:
            print("  ✗ Below re-enable threshold (PF ≥ 1.3 required)")
    else:
        print("NO variant satisfies MDD constraint — all regressed > 3pp MDD vs baseline")
    print("=" * 78)


if __name__ == "__main__":
    main()
