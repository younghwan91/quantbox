#!/usr/bin/env python3
"""r116b: D_mom8 min_conviction_score sweep.

Tests min_conviction ∈ {0, 1, 2, 3} as entry filter. Since b3_confluence is
reserved-always-False in production, max achievable conviction is 3/4.

Variants:
    min=0 (baseline): no filter — all signals enter
    min=1: require ≥1 condition (e2_sma OR c2_btc OR a3_quality)
    min=2: require ≥2 conditions
    min=3: require all three live conditions (e2_sma AND c2_btc AND a3_quality)

DOGE sanity check: the 2026-04-12 live trade fired at conviction 2/4 = 2/3-live.
    min=2 would ALLOW it (score 2 >= 2)
    min=3 would BLOCK it (score 2 < 3)

Rule: no SL/TP/trail changes — entry filter only. No mock data. Config-first.

Usage:
    uv run python -m research.active.r116_willy_nfi.r116b_conviction_sweep

(ref: r116 plan Section 3, US-R116B-3)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    G_4TF,
    build_resampled,
    compute_metrics,
    gen_vr_signal,
    make_direction_momentum,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r116_willy_nfi.r116_prep_baseline import (
    DATA_YEARS,
    STATE_DIR,
    _compute_sharpe,
    compute_mfe_losing,
    load_15m_data,
)

OUTPUT_DIR = ROOT / ".omc" / "state" / "r116b"
BASELINE_PATH = STATE_DIR / "vr_directional_metrics.json"

CFG = VolStraddleConfig()
COST_DIRECTIONAL = 0.0008


# ── Conviction computation (mirrors strategy.py:_compute_conviction_score) ──


def compute_conviction_at_bar(
    close: np.ndarray,
    vr: np.ndarray,
    btc_close: np.ndarray | None,
    bar_idx: int,
) -> int:
    """Return conviction score 0-3 at a specific bar. b3_confluence reserved (always False).

    Mirrors `quantbox/strategy/vol_straddle/strategy.py::_compute_conviction_score` exactly.
    """
    i = bar_idx
    score = 0

    # E2: SMA Proximity
    if CFG.e2_enabled and i >= CFG.e2_sma_period - 1:
        sma = np.mean(close[i - CFG.e2_sma_period + 1 : i + 1])
        if sma > 0:
            prox = abs(close[i] - sma) / sma
            if prox < CFG.e2_threshold:
                score += 1

    # C2: BTC Neutral
    if CFG.c2_enabled and btc_close is not None and i > CFG.momentum_period:
        btc_prev = btc_close[i - CFG.momentum_period]
        if btc_prev > 0:
            btc_mom = btc_close[i] / btc_prev - 1.0
            if abs(btc_mom) < CFG.c2_btc_mom_threshold:
                score += 1

    # B3: Confluence — RESERVED (always False, max achievable = 3/4)
    # ↑ strategy.py:670 confirms

    # A3: Quality+
    if CFG.a3_enabled:
        vr_val = float(vr[i]) if not np.isnan(vr[i]) else 999.0
        if vr_val < CFG.a3_deep_vr_threshold:
            score += 1

    return score


def collect_tagged_trades(
    ct_cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    include_direction: bool = False,
) -> list[tuple]:
    """Collect D_mom8 trades with conviction score attached. Mirrors collect_trades
    G_4TF + tf_priority=True + funding drag exactly. Adds conviction tag at entry bar.

    Returns:
        If include_direction=False: list of (symbol, entry_15m, exit_15m, pnl, conviction_score).
        If include_direction=True:  list of (symbol, entry_15m, exit_15m, pnl, conviction_score, direction).
            direction is +1 (LONG) or -1 (SHORT) from momentum at entry bar.
    """
    sp = ct_cfg["sp"]
    lp = ct_cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    d_sl = ct_cfg["d_sl"]
    d_tp = ct_cfg["tp"]
    d_vr = ct_cfg["d_vr"]
    d_trail_act = ct_cfg["d_trail_activate"]
    d_trail_dist = ct_cfg["d_trail_distance"]

    # Build 30m on the fly to mirror baseline collect_trades behavior.
    # Baseline does NOT precompute VR on 30m → the 30m TF is silently skipped
    # (vr_key lookup returns None). Mirror that exactly for parity.
    data_30m = build_resampled(data_15m, 2)
    # NOTE: intentionally no precompute_all_vr/precompute on data_30m — matches baseline.

    tf_data_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    btc_by_ratio: dict[int, np.ndarray | None] = {}
    for ratio, ds in tf_data_map.items():
        btc = ds.get("BTCUSDT")
        btc_by_ratio[ratio] = btc["close"] if btc is not None else None

    all_trades: list[tuple[str, int, int, float, int]] = []

    for sym in sorted(data_15m.keys()):
        # tf_priority mode: higher TFs first, lower-TF overlaps blocked
        sorted_tfs = sorted(G_4TF, key=lambda t: t[1], reverse=True)
        occupied: list[tuple[int, int]] = []

        for tf_label, ratio, warmup, mh_d in sorted_tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            btc_c = btc_by_ratio.get(ratio)

            sig = gen_vr_signal(vr, d_vr, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, d_sl, d_tp, mh_d, COST_DIRECTIONAL,
                d_trail_act, d_trail_dist,
            )

            for k in range(len(p)):
                bar_i = int(e[k])
                entry_15m = bar_i * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                blocked = False
                for oe, ox in occupied:
                    if entry_15m <= ox and exit_15m >= oe:
                        blocked = True
                        break
                if not blocked:
                    score = compute_conviction_at_bar(c, vr, btc_c, bar_i)
                    direction = int(dire[bar_i]) if bar_i < len(dire) else 0
                    if include_direction:
                        all_trades.append(
                            (sym, entry_15m, exit_15m, float(p[k]), score, direction)
                        )
                    else:
                        all_trades.append((sym, entry_15m, exit_15m, float(p[k]), score))
                    occupied.append((entry_15m, exit_15m))

    # Funding drag (mirrors collect_trades line 677-694)
    funding_8h = CFG.default_funding_rate_8h
    if funding_8h > 0:
        BARS_PER_8H_15M = 32
        adjusted = []
        for t in all_trades:
            bars_held = t[2] - t[1]
            funding_drag = funding_8h * (bars_held / BARS_PER_8H_15M)
            pnl_adj = t[3] - funding_drag
            adjusted.append((*t[:3], pnl_adj, *t[4:]))
        all_trades = adjusted

    return all_trades


# ── Sweep runner ──


def filter_by_min(
    tagged: list[tuple[str, int, int, float, int]],
    min_score: int,
) -> list[tuple[str, int, int, float]]:
    """Apply conviction gate and strip score."""
    return [(s, e, x, p) for s, e, x, p, sc in tagged if sc >= min_score]


def run_variant(
    min_score: int,
    tagged_trades: list[tuple[str, int, int, float, int]],
    data_15m: dict,
) -> dict:
    t0 = time.time()
    filtered = filter_by_min(tagged_trades, min_score)
    pnl_arr, accepted = portfolio_sim_with_details(filtered, CFG.max_concurrent)
    metrics = compute_metrics(pnl_arr, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    mfe_losing = compute_mfe_losing(accepted, data_15m) if accepted else 0.0

    return {
        "variant": f"min_conviction_{min_score}",
        "min_conviction_score": min_score,
        "raw_trades": len(tagged_trades),
        "filtered_trades": len(filtered),
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
            "min_conviction", "filtered_trades", "N", "WR_pct", "PF",
            "MDD_pct", "Sharpe", "Calmar", "ann_ret_pct", "mfe_losing_avg_pct",
        ])
        for r in results:
            w.writerow([
                r["min_conviction_score"], r["filtered_trades"], r["N"],
                r["WR_pct"], r["PF"], r["MDD_pct"],
                r["sharpe"], r["calmar"], r["ann_ret_pct"], r["mfe_losing_avg_pct"],
            ])


def main() -> None:
    print("=" * 78)
    print("r116b — D_mom8 min_conviction_score Sweep")
    print("=" * 78)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    print(f"Baseline (min=0):  PF={baseline['PF']}  MDD={baseline['MDD_pct']}%  N={baseline['N']}")

    # Load data + precompute once
    print("\nLoading data + precomputing VR/momentum...")
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
    print(f"Precompute elapsed {time.time()-t0:.1f}s, loaded {len(data_15m)} symbols")

    # Tag all trades with conviction ONCE
    print("\nCollecting tagged trades (with conviction scores)...")
    ct_cfg = {
        "sp": CFG.vr_short,
        "lp": CFG.vr_long,
        "d_vr": CFG.d_vr_threshold,
        "d_sl": CFG.d_sl,
        "tp": CFG.d_tp,
        "d_trail_activate": CFG.d_trail_activate,
        "d_trail_distance": CFG.d_trail_distance,
    }
    tagged_trades = collect_tagged_trades(ct_cfg, data_15m, data_1h, data_4h)
    print(f"Raw tagged trades: {len(tagged_trades)}")

    # Conviction score distribution
    from collections import Counter
    score_dist = Counter(t[4] for t in tagged_trades)
    print(f"Score distribution: {dict(sorted(score_dist.items()))}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for min_score in [0, 1, 2, 3]:
        print(f"\n--- min_conviction = {min_score} ---")
        result = run_variant(min_score, tagged_trades, data_15m)
        results.append(result)

        out_path = OUTPUT_DIR / f"conviction_{min_score}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"  filtered={result['filtered_trades']}  N={result['N']}  "
            f"WR={result['WR_pct']}%  PF={result['PF']}  "
            f"MDD={result['MDD_pct']}%  Sharpe={result['sharpe']}"
        )

    summary_path = OUTPUT_DIR / "conviction_summary.csv"
    write_summary_csv(results, summary_path)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    baseline_pf = baseline["PF"]
    print(f"{'min':<4} {'filt':>6} {'N':>5} {'WR%':>6} {'PF':>6} {'MDD%':>6} {'Sharpe':>7} {'ΔPF':>7}")
    for r in results:
        delta = r["PF"] - baseline_pf
        print(
            f"{r['min_conviction_score']:<4} {r['filtered_trades']:>6} "
            f"{r['N']:>5} {r['WR_pct']:>6.1f} {r['PF']:>6.3f} "
            f"{r['MDD_pct']:>6.1f} {r['sharpe']:>7.2f} {delta:>+7.3f}"
        )

    best = max(results, key=lambda r: r["PF"])
    print(f"\nBEST variant: min_conviction={best['min_conviction_score']}  "
          f"PF={best['PF']} (Δ {best['PF']-baseline_pf:+.3f})")
    if best["PF"] >= 1.3:
        print("  ✓ Exceeds re-enable threshold PF ≥ 1.3")
    else:
        print("  ✗ Below re-enable threshold")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
