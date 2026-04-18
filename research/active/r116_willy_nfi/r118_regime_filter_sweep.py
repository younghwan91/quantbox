#!/usr/bin/env python3
"""r118: MA regime filter sweeps — Tier A candidates from Willy/NFI review.

Tests three confident hypotheses for improving existing strategies via MA-based
regime context (not MA replacement):

    R1. D_mom8 BTC regime filter (Willy SMA200 gate):
        Block LONG when BTC < 4h SMA200 (hard downtrend)
        Block SHORT when BTC > 4h SMA200 (hard uptrend)
        Hypothesis: counter-trend entries are the main loss source.

    R2. R1 + r116b 'loose' blacklist combined:
        Test whether BTC regime filter compounds with symbol selection.

    R3. MA Cluster + 1d SMA50 direction alignment:
        Block LONG when 1d SMA50 declining (price < SMA50 1d ago)
        Block SHORT when 1d SMA50 rising
        Hypothesis: multi-TF trend alignment reduces false breakouts.

Canonical TradeEventSim. No SL/TP/param tweaks. Config-first.

Usage:
    uv run python -m research.active.r116_willy_nfi.r118_regime_filter_sweep

(ref: r116 plan — 2026-04-18 MA technique extension)
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
from research.active.r116_willy_nfi.r116b_conviction_sweep import collect_tagged_trades
from research.active.r116_willy_nfi.r116b_symbol_blacklist_sweep import (
    VARIANTS as BLACKLIST_VARIANTS,
    compute_blacklist,
)

OUTPUT_DIR = ROOT / ".omc" / "state" / "r118"
BASELINE_PATH = STATE_DIR / "vr_directional_metrics.json"
MA_BASELINE_PATH = STATE_DIR / "ma_cluster_metrics.json"

CFG = VolStraddleConfig()


# ── Helpers ─────────────────────────────────────────────────────────────────


def compute_sma_1d(arr_15m: np.ndarray, period_1d_bars: int = 50) -> np.ndarray:
    """Simple moving average on 1d resampled data. Uses cumulative 96×15m bars per day."""
    # 1 day = 96 × 15m bars. Resample close to 1d by taking last bar per day.
    bars_per_day = 96
    n_days = len(arr_15m) // bars_per_day
    if n_days < period_1d_bars + 1:
        return np.full(len(arr_15m), np.nan)
    daily_close = np.array([arr_15m[(i + 1) * bars_per_day - 1] for i in range(n_days)])
    sma = np.full(n_days, np.nan)
    for i in range(period_1d_bars - 1, n_days):
        sma[i] = daily_close[i - period_1d_bars + 1 : i + 1].mean()
    # Project back to 15m cadence: each 15m bar inherits its day's SMA value
    out = np.full(len(arr_15m), np.nan)
    for i in range(len(arr_15m)):
        d = i // bars_per_day
        if 0 <= d < n_days:
            out[i] = sma[d]
    return out


def compute_sma_4h(arr_15m: np.ndarray, period_4h_bars: int = 200) -> np.ndarray:
    """SMA on 4h resampled series, projected back to 15m cadence."""
    bars_per_4h = 16
    n_4h = len(arr_15m) // bars_per_4h
    if n_4h < period_4h_bars + 1:
        return np.full(len(arr_15m), np.nan)
    c_4h = np.array([arr_15m[(i + 1) * bars_per_4h - 1] for i in range(n_4h)])
    sma = np.full(n_4h, np.nan)
    for i in range(period_4h_bars - 1, n_4h):
        sma[i] = c_4h[i - period_4h_bars + 1 : i + 1].mean()
    out = np.full(len(arr_15m), np.nan)
    for i in range(len(arr_15m)):
        b = i // bars_per_4h
        if 0 <= b < n_4h:
            out[i] = sma[b]
    return out


# ── R1 & R2: D_mom8 regime filter ───────────────────────────────────────────


def apply_btc_regime_filter(
    tagged_trades: list[tuple],  # (sym, entry_15m, exit_15m, pnl, score, direction)
    btc_close_15m: np.ndarray,
    btc_sma200_4h_proj: np.ndarray,
    strong_threshold: float = 0.03,
) -> tuple[list[tuple[str, int, int, float]], dict]:
    """Drop trades where direction is counter to BTC 4h SMA200 regime.

    NO LOOK-AHEAD: uses only (symbol, entry_bar, direction) + BTC SMA200 state
    at entry bar. pnl is NOT consulted in the filter decision.

    Rule:
        deviation = (btc_close - btc_sma200) / btc_sma200
        if |deviation| < strong_threshold: ALLOW (weak trend, no regime signal)
        elif deviation > 0 (uptrend): BLOCK trades with direction == -1 (SHORT)
        elif deviation < 0 (downtrend): BLOCK trades with direction == +1 (LONG)

    Only direction-tagged trades (6-tuple) are filtered; 5-tuple input is
    passed through unchanged (backward compat).
    """
    kept: list[tuple[str, int, int, float]] = []
    stats = {
        "counter_trend_short_blocked": 0,
        "counter_trend_long_blocked": 0,
        "with_trend_allowed": 0,
        "weak_trend_allowed": 0,
        "no_regime_data_allowed": 0,
    }

    for t in tagged_trades:
        sym, entry_15m, exit_15m, pnl = t[0], t[1], t[2], t[3]
        direction = t[5] if len(t) >= 6 else 0
        if entry_15m >= len(btc_close_15m) or direction == 0:
            kept.append((sym, entry_15m, exit_15m, pnl))
            stats["no_regime_data_allowed"] += 1
            continue
        btc_px = float(btc_close_15m[entry_15m])
        btc_sma = float(btc_sma200_4h_proj[entry_15m])
        if np.isnan(btc_sma) or btc_sma <= 0:
            kept.append((sym, entry_15m, exit_15m, pnl))
            stats["no_regime_data_allowed"] += 1
            continue
        deviation = (btc_px - btc_sma) / btc_sma
        if abs(deviation) < strong_threshold:
            kept.append((sym, entry_15m, exit_15m, pnl))
            stats["weak_trend_allowed"] += 1
        elif deviation > 0:
            # Uptrend — block SHORT
            if direction == -1:
                stats["counter_trend_short_blocked"] += 1
                continue
            kept.append((sym, entry_15m, exit_15m, pnl))
            stats["with_trend_allowed"] += 1
        else:
            # Downtrend — block LONG
            if direction == 1:
                stats["counter_trend_long_blocked"] += 1
                continue
            kept.append((sym, entry_15m, exit_15m, pnl))
            stats["with_trend_allowed"] += 1

    return kept, stats


def run_r1_r2(data_15m: dict, data_1h: dict, data_4h: dict, tagged_trades: list, per_symbol: list) -> list[dict]:
    """Run R1 (regime only) and R2 (regime + blacklist)."""
    btc_15m = data_15m["BTCUSDT"]["close"]
    btc_sma200_4h = compute_sma_4h(btc_15m, 200)

    results: list[dict] = []

    # Baseline (no filter) — strip extras, keep only (sym, entry, exit, pnl)
    baseline_flat = [(t[0], t[1], t[2], t[3]) for t in tagged_trades]
    pnl, acc = portfolio_sim_with_details(baseline_flat, CFG.max_concurrent)
    m = compute_metrics(pnl, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    results.append({
        "variant": "baseline",
        "N": m["N"], "WR_pct": m["WR"], "PF": m["PF"], "MDD_pct": m["max_dd_pct"],
        "sharpe": _compute_sharpe(pnl), "calmar": m["calmar"], "ann_ret_pct": m["ann_ret_pct"],
    })

    # R1 — regime filter only (no blacklist)
    r1_trades, r1_stats = apply_btc_regime_filter(tagged_trades, btc_15m, btc_sma200_4h)
    pnl, acc = portfolio_sim_with_details(r1_trades, CFG.max_concurrent)
    m = compute_metrics(pnl, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    results.append({
        "variant": "R1_regime_only",
        "N": m["N"], "WR_pct": m["WR"], "PF": m["PF"], "MDD_pct": m["max_dd_pct"],
        "sharpe": _compute_sharpe(pnl), "calmar": m["calmar"], "ann_ret_pct": m["ann_ret_pct"],
        "filter_stats": r1_stats,
    })

    # R2 — regime + 'loose' blacklist
    loose_bl = compute_blacklist(per_symbol, 1.0, 10)
    r1_after_bl = [t for t in r1_trades if t[0] not in loose_bl]
    pnl, acc = portfolio_sim_with_details(r1_after_bl, CFG.max_concurrent)
    m = compute_metrics(pnl, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    results.append({
        "variant": "R2_regime_plus_loose_blacklist",
        "N": m["N"], "WR_pct": m["WR"], "PF": m["PF"], "MDD_pct": m["max_dd_pct"],
        "sharpe": _compute_sharpe(pnl), "calmar": m["calmar"], "ann_ret_pct": m["ann_ret_pct"],
        "blacklist_size": len(loose_bl),
    })

    return results


# ── R3: MA Cluster + 1d alignment ───────────────────────────────────────────


def run_r3_ma_cluster(data_15m: dict, data_4h: dict) -> list[dict]:
    """MA Cluster baseline + 1d SMA50 direction alignment filter."""
    from quantbox.backtest.trade_sim import sim_ma_cluster_trailing
    from quantbox.strategy.vol_straddle.signals import compute_momentum as _cmom

    # Run MA Cluster sim per symbol, tag each trade with direction via entry→exit close movement
    periods = CFG.ma_cluster_sma_periods
    raw_trades: list[tuple[str, int, int, float, int]] = []  # sym, e15, x15, pnl, direction(+1/-1)

    def sma_vec(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(p - 1, len(arr)):
            out[i] = arr[i - p + 1 : i + 1].mean()
        return out

    for sym in sorted(data_4h.keys()):
        if sym == "BTCUSDT":
            continue
        d = data_4h[sym]
        c, h, lo = d["close"], d["high"], d["low"]
        sma20 = sma_vec(c, periods[0])
        sma50 = sma_vec(c, periods[1])
        sma100 = sma_vec(c, periods[2])
        sma200 = sma_vec(c, periods[3])
        mom = _cmom(c, CFG.momentum_period)

        pnl_arr, ent_arr, ext_arr = sim_ma_cluster_trailing(
            c, h, lo, sma20, sma50, sma100, sma200, mom,
            CFG.ma_cluster_pct,
            CFG.ma_cluster_min_duration,
            CFG.ma_cluster_max_duration,
            CFG.ma_cluster_min_momentum,
            CFG.ma_cluster_sl,
            CFG.ma_cluster_tp,
            CFG.ma_cluster_trail_activate,
            CFG.ma_cluster_trail_distance,
            CFG.ma_cluster_max_hold_4h,
            CFG.round_trip_cost,
            200,
        )
        ratio = 16
        for i in range(len(pnl_arr)):
            ent_4h = int(ent_arr[i])
            # Direction inferred from 4h bar close momentum at entry
            direction = 1 if mom[ent_4h] > 0 else -1
            ent_15 = ent_4h * ratio
            ext_15 = int(ext_arr[i]) * ratio
            raw_trades.append((sym, ent_15, ext_15, float(pnl_arr[i]), direction))

    # Baseline (no filter)
    flat = [(s, e, x, p) for s, e, x, p, _ in raw_trades]
    pnl, _ = portfolio_sim_with_details(flat, CFG.max_concurrent)
    m = compute_metrics(pnl, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    baseline_ma = {
        "variant": "MA_baseline",
        "N": m["N"], "WR_pct": m["WR"], "PF": m["PF"], "MDD_pct": m["max_dd_pct"],
        "sharpe": _compute_sharpe(pnl), "calmar": m["calmar"], "ann_ret_pct": m["ann_ret_pct"],
    }

    # R3 — apply 1d SMA50 direction filter per-symbol
    # Block LONG when 1d SMA50 declining; block SHORT when 1d SMA50 rising.
    sma50_1d_by_sym: dict[str, np.ndarray] = {}
    for sym, d in data_15m.items():
        sma50_1d_by_sym[sym] = compute_sma_1d(d["close"], 50)

    kept: list[tuple[str, int, int, float]] = []
    block_stats = {"block_long_in_down": 0, "block_short_in_up": 0, "allow": 0}
    for sym, e15, x15, pnl_val, direction in raw_trades:
        sma1d = sma50_1d_by_sym.get(sym)
        if sma1d is None or e15 >= len(sma1d) or e15 < 96:
            kept.append((sym, e15, x15, pnl_val))
            continue
        sma_now = sma1d[e15]
        sma_prev = sma1d[max(0, e15 - 96)]  # 1 day earlier
        if np.isnan(sma_now) or np.isnan(sma_prev):
            kept.append((sym, e15, x15, pnl_val))
            continue
        rising = sma_now > sma_prev
        if direction > 0 and not rising:
            block_stats["block_long_in_down"] += 1
            continue
        if direction < 0 and rising:
            block_stats["block_short_in_up"] += 1
            continue
        kept.append((sym, e15, x15, pnl_val))
        block_stats["allow"] += 1

    pnl, _ = portfolio_sim_with_details(kept, CFG.max_concurrent)
    m = compute_metrics(pnl, DATA_YEARS, portfolio_size=CFG.max_concurrent)
    r3 = {
        "variant": "R3_ma_plus_1d_alignment",
        "N": m["N"], "WR_pct": m["WR"], "PF": m["PF"], "MDD_pct": m["max_dd_pct"],
        "sharpe": _compute_sharpe(pnl), "calmar": m["calmar"], "ann_ret_pct": m["ann_ret_pct"],
        "filter_stats": block_stats,
    }
    return [baseline_ma, r3]


def write_summary(all_results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "N", "WR_pct", "PF", "MDD_pct", "Sharpe", "Calmar", "ann_ret_pct"])
        for r in all_results:
            w.writerow([
                r["variant"], r["N"], r["WR_pct"], r["PF"], r["MDD_pct"],
                r["sharpe"], r["calmar"], r["ann_ret_pct"],
            ])


def main() -> None:
    print("=" * 78)
    print("r118 — MA Regime Filter Sweep (R1/R2/R3)")
    print("=" * 78)

    # Load baseline per_symbol for R2
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    per_symbol = baseline["per_symbol"]
    baseline_pf = baseline["PF"]

    print(f"D_mom8 baseline: PF={baseline_pf}  N={baseline['N']}")

    print("\nLoading data + precomputing VR/mom...")
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
    print(f"Data loaded in {time.time()-t0:.1f}s ({len(data_15m)} symbols)")

    print("\nCollecting D_mom8 tagged trades (with direction)...")
    ct_cfg = {
        "sp": CFG.vr_short, "lp": CFG.vr_long,
        "d_vr": CFG.d_vr_threshold, "d_sl": CFG.d_sl, "tp": CFG.d_tp,
        "d_trail_activate": CFG.d_trail_activate, "d_trail_distance": CFG.d_trail_distance,
    }
    tagged = collect_tagged_trades(ct_cfg, data_15m, data_1h, data_4h, include_direction=True)
    print(f"Raw tagged: {len(tagged)}")
    # Direction distribution sanity
    from collections import Counter as _Counter
    dir_dist = _Counter(t[5] for t in tagged)
    print(f"Direction distribution: {dict(dir_dist)}")

    print("\n── R1 + R2: D_mom8 BTC regime filter ──")
    r1r2 = run_r1_r2(data_15m, data_1h, data_4h, tagged, per_symbol)
    for r in r1r2:
        print(
            f"  {r['variant']:<38} N={r['N']:>5} WR={r['WR_pct']:>5.1f}%  "
            f"PF={r['PF']:>6.3f}  MDD={r['MDD_pct']:>5.2f}%  Sharpe={r['sharpe']:>6.2f}"
        )

    print("\n── R3: MA Cluster + 1d SMA50 alignment ──")
    r3 = run_r3_ma_cluster(data_15m, data_4h)
    for r in r3:
        print(
            f"  {r['variant']:<38} N={r['N']:>5} WR={r['WR_pct']:>5.1f}%  "
            f"PF={r['PF']:>6.3f}  MDD={r['MDD_pct']:>5.2f}%  Sharpe={r['sharpe']:>6.2f}"
        )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = r1r2 + r3
    with open(OUTPUT_DIR / "r118_regime_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    write_summary(all_results, OUTPUT_DIR / "r118_summary.csv")

    print("\n" + "=" * 78)
    print("SUMMARY vs baselines")
    print("=" * 78)
    base_pf = r1r2[0]["PF"]
    for r in r1r2[1:]:
        delta = r["PF"] - base_pf
        print(f"{r['variant']:<40} ΔPF={delta:+.3f}  (baseline {base_pf})")
    base_ma_pf = r3[0]["PF"]
    for r in r3[1:]:
        delta = r["PF"] - base_ma_pf
        print(f"{r['variant']:<40} ΔPF={delta:+.3f}  (MA baseline {base_ma_pf})")

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
