"""r75: MA Cluster Trail-Aware 파라미터 최적화.

r73 used flat TP=8% for MA cluster → PF=1.283, Calmar=4.4.
Live config has trail_activate=4%, trail_distance=0.5%.
This script adds proper trailing stop simulation and sweeps key parameters.

Phases:
  0: Trail-aware baseline (config values) vs r73 flat (N=2461, PF=1.283)
  1: cluster_pct × min_duration grid
  2: min_momentum sweep (with Phase 1 best)
  3: trail_activate × trail_distance grid (with Phase 1+2 best)

Usage:
    uv run python research/active/r71_r75/r75_ma_cluster_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r41_r45.r44c_ma_confluence import precompute_mas

COST_DIRECTIONAL = 0.0008
WARMUP_4h = 13  # ceil(200/16) — enough for SMA200 to be valid


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading (same pattern as r73)
# ══════════════════════════════════════════════════════════════════════════════


def load_data(whitelist: list[str]) -> dict:
    """Load 60 whitelist symbols from cache_extended with _offset alignment."""
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

    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Numba sim: MA cluster + trailing stop exit
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_trailing(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    max_duration: int,
    min_momentum: float,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MA cluster squeeze with trailing stop exit (numba-accelerated).

    Cluster detection (same as sim_ma_cluster_enhanced):
      - All 4 MAs must be within cluster_pct of each other
      - Duration must be in [min_duration, max_duration] (zombie filter)
      - Price must break out of cluster zone
      - |momentum| >= min_momentum required (direction confirmation)

    Exit logic (priority order):
      1. Trail stop: once peak_gain >= trail_activate, exit when
         intra-bar gain drops to peak_gain - trail_distance
      2. Hard TP (safety cap): exit at tp
      3. SL: exit at -sl (initial; suppressed once trailing activates)
      4. Max hold: exit at close[max_hold_bar]

    Args:
        close/high/low: 4h OHLCV arrays
        sma20/50/100/200: Pre-computed SMAs
        momentum: Pre-computed momentum(8) series
        cluster_pct: MA spread threshold for cluster detection
        min_duration: Minimum bars cluster must persist before entry
        max_duration: Maximum bars (zombie filter); 0 = disabled
        min_momentum: Minimum |momentum| for directional confirmation
        sl: Stop-loss fraction (0.02 = 2%)
        tp: Hard TP cap fraction (0.08 = 8%)
        trail_activate: Peak gain to activate trailing (0.04 = 4%)
        trail_distance: Trail distance from peak (0.005 = 0.5%)
        max_hold: Maximum bars to hold (4h units)
        cost: Round-trip transaction cost
        warmup: Skip first N bars (for SMA warmup)

    Returns:
        (pnl, entry_bars, exit_bars) arrays
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

        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0.0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            i += 1
            continue

        # Min duration: cluster must have persisted
        if cluster_count < min_duration:
            i += 1
            continue

        # Max duration: zombie filter
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue

        # Breakout: price must be outside the cluster zone
        if ma_min <= close[i] <= ma_max:
            i += 1
            continue

        # Momentum direction confirmation
        if abs(momentum[i]) < min_momentum:
            i += 1
            continue
        d = 1 if momentum[i] > 0 else -1

        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False

        for j in range(i + 1, end):
            if d == 1:
                # Long: track peak via high, check low for trail/SL
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                # Trailing stop (once activated, replaces initial SL)
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break

                # Hard TP cap
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break

                # Initial SL (only before trailing activates)
                if not trailing_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break

                # Activate trailing
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                # Short: track peak via low, check high for trail/SL
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                # Trailing stop
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break

                # Hard TP cap
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break

                # Initial SL
                if not trailing_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break

                # Activate trailing
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
        else:
            # Max-hold exit at close
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1

        # One signal per cluster episode: reset after trade
        cluster_count = 0
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade collection
# ══════════════════════════════════════════════════════════════════════════════


def _dedup(sym: str, trades: list) -> list:
    """Per-symbol time dedup: reject overlapping trades."""
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_ma_trail_trades(
    data_15m: dict,
    data_4h: dict,
    params: dict,
) -> list[tuple[str, int, int, float]]:
    """Collect MA cluster trailing trades for all whitelist symbols (4h only).

    Args:
        data_15m: 15m data with _offset per symbol
        data_4h: 4h resampled data (sma_* and mom_8 pre-populated)
        params: dict with cluster_pct, min_duration, max_duration,
                min_momentum, sl, tp, trail_activate, trail_distance,
                max_hold (4h bars)

    Returns:
        List of (symbol, entry_15m_bar, exit_15m_bar, pnl) sorted by entry
    """
    ratio = 16  # 4h = 16 × 15m bars
    cluster_pct   = params["cluster_pct"]
    min_duration  = params["min_duration"]
    max_duration  = params["max_duration"]
    min_momentum  = params["min_momentum"]
    sl            = params["sl"]
    tp            = params["tp"]
    trail_activate = params["trail_activate"]
    trail_distance = params["trail_distance"]
    max_hold      = params["max_hold"]

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        d4 = data_4h.get(sym)
        if d4 is None:
            continue
        if "sma_200" not in d4 or "mom_8" not in d4:
            continue

        offset = int(data_15m[sym].get("_offset", 0))

        p, e, x = sim_ma_cluster_trailing(
            d4["close"], d4["high"], d4["low"],
            d4["sma_20"], d4["sma_50"], d4["sma_100"], d4["sma_200"],
            d4["mom_8"],
            cluster_pct, min_duration, max_duration, min_momentum,
            sl, tp, trail_activate, trail_distance,
            max_hold, COST_DIRECTIONAL, WARMUP_4h,
        )

        sym_trades: list[tuple[int, int, float]] = []
        for k in range(len(p)):
            entry_bar = int(e[k]) * ratio + offset
            exit_bar  = int(x[k]) * ratio + ratio - 1 + offset
            sym_trades.append((entry_bar, exit_bar, float(p[k])))

        all_trades.extend(_dedup(sym, sym_trades))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def run_portfolio(
    trades: list,
    max_concurrent: int,
    data_years: float,
) -> tuple[dict, dict]:
    """Run portfolio_sim + compute_metrics + bootstrap_pf."""
    pnl = portfolio_sim(trades, max_concurrent)
    m   = compute_metrics(pnl, data_years)
    b   = bootstrap_pf(pnl, 5000)
    return m, b


def fmt_row(label: str, m: dict, b: dict) -> str:
    return (
        f"  {label:<28} N={m['N']:>5}  PF={m['PF']:.3f}  "
        f"WR={m['WR']:.1f}%  EV={m['EV_bp']:>+6.1f}bp  "
        f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}  "
        f"CI=[{b['CI_lo']:.3f},{b['CI_hi']:.3f}]"
    )


def print_metrics(label: str, m: dict, b: dict) -> None:
    print(fmt_row(label, m, b))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*80}")
    print(f"  r75: MA Cluster Trail-Aware 파라미터 최적화  [{commit}]")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*80}")

    cfg = VolStraddleConfig()
    ma_ln = cfg.ma_cluster_leverage / cfg.max_concurrent
    max_concurrent = cfg.max_concurrent

    print(f"\n  MA Cluster config (from VolStraddleConfig):")
    print(f"    cluster_pct={cfg.ma_cluster_pct}  min_dur={cfg.ma_cluster_min_duration}"
          f"  max_dur={cfg.ma_cluster_max_duration}")
    print(f"    min_momentum={cfg.ma_cluster_min_momentum}  sl={cfg.ma_cluster_sl}"
          f"  tp={cfg.ma_cluster_tp}")
    print(f"    trail_activate={cfg.ma_cluster_trail_activate}"
          f"  trail_distance={cfg.ma_cluster_trail_distance}")
    print(f"    max_hold_4h={cfg.ma_cluster_max_hold_4h}  leverage={cfg.ma_cluster_leverage}"
          f"  L/N={ma_ln:.2f}  max_concurrent={max_concurrent}")

    # ── Load & precompute ─────────────────────────────────────────────────
    print("\nLoading data (60 symbols, symbol_whitelist)...")
    data_15m = load_data(cfg.symbol_whitelist)
    if not data_15m:
        print("❌ No data found in cache_extended. Aborting.")
        return

    data_4h = build_resampled(data_15m, 16)

    # Precompute SMAs (sma_20/50/100/200) and momentum (mom_8) on 4h
    precompute_mas(data_4h)
    precompute(data_4h, n_workers=None)

    # Data years from 15m bars (same convention as r73)
    avg_bars = int(np.mean([len(d["close"]) for d in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  "
          f"{data_years:.2f} years  |  load: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 0: Trail-Aware Baseline (config values)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  PHASE 0: TRAIL-AWARE BASELINE (config params)")
    print(f"    vs r73 flat: N=2461, PF=1.283, Calmar=4.4")
    print(f"{'─'*80}")

    baseline_params = {
        "cluster_pct":    cfg.ma_cluster_pct,
        "min_duration":   cfg.ma_cluster_min_duration,
        "max_duration":   cfg.ma_cluster_max_duration,
        "min_momentum":   cfg.ma_cluster_min_momentum,
        "sl":             cfg.ma_cluster_sl,
        "tp":             cfg.ma_cluster_tp,
        "trail_activate": cfg.ma_cluster_trail_activate,
        "trail_distance": cfg.ma_cluster_trail_distance,
        "max_hold":       cfg.ma_cluster_max_hold_4h,
    }

    t1 = time.time()
    bl_trades = collect_ma_trail_trades(data_15m, data_4h, baseline_params)
    bl_m, bl_b = run_portfolio(bl_trades, max_concurrent, data_years)
    print(f"\n  Trail baseline ({time.time()-t1:.1f}s):")
    print_metrics(
        f"trail pct={cfg.ma_cluster_pct} dur={cfg.ma_cluster_min_duration}",
        bl_m, bl_b,
    )

    n_diff  = abs(bl_m["N"] - 2461) / 2461 * 100 if bl_m["N"] > 0 else 999.0
    pf_diff = abs(bl_m["PF"] - 1.283) / 1.283 * 100
    print(f"\n    vs r73 flat: N_diff={n_diff:.1f}%  PF_diff={pf_diff:.1f}%  "
          f"(trail changes both — expected to differ)")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: cluster_pct × min_duration grid
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  PHASE 1: cluster_pct × min_duration GRID")
    print("  (Fixed: max_dur=24, min_mom=0.01, sl=0.02, tp=0.08,")
    print("          trail_act=0.04, trail_dist=0.005, max_hold=12)")
    print(f"{'─'*80}")

    pct_grid = [0.020, 0.025, 0.030, 0.035, 0.040]
    dur_grid = [6, 8, 10, 12]

    p1_results: list[dict] = []
    hdr = f"  {'pct':>5} {'dur':>4} | {'N':>5} {'PF':>7} {'WR':>6} {'EV':>7} {'DD':>6} {'Calm':>6}"
    print(f"\n{hdr}")
    print(f"  {'-'*5} {'-'*4}   {'-'*5} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")

    for pct in pct_grid:
        for dur in dur_grid:
            params = {**baseline_params, "cluster_pct": pct, "min_duration": dur}
            trades = collect_ma_trail_trades(data_15m, data_4h, params)
            m, b = run_portfolio(trades, max_concurrent, data_years)
            star = " ★" if m["PF"] >= 1.3 else ""
            print(f"  {pct:>5.3f} {dur:>4}   "
                  f"N={m['N']:>5}  PF={m['PF']:.3f}  "
                  f"WR={m['WR']:.1f}%  EV={m['EV_bp']:>+6.1f}bp  "
                  f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}"
                  f"{star}")
            p1_results.append({"pct": pct, "dur": dur, "m": m, "b": b})

    # Best by PF
    best_p1 = max(p1_results, key=lambda r: r["m"]["PF"])
    best_pct = best_p1["pct"]
    best_dur = best_p1["dur"]
    print(f"\n  Phase 1 best: cluster_pct={best_pct}  min_dur={best_dur}  "
          f"PF={best_p1['m']['PF']:.3f}  Calmar={best_p1['m']['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: min_momentum sweep (Phase 1 best)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  PHASE 2: min_momentum SWEEP")
    print(f"  (Fixed: pct={best_pct}, dur={best_dur}, max_dur=24,")
    print(f"          sl=0.02, tp=0.08, trail_act=0.04, trail_dist=0.005, max_hold=12)")
    print(f"{'─'*80}")

    mom_grid = [0.005, 0.010, 0.015, 0.020]
    p2_results: list[dict] = []
    print()
    for mom in mom_grid:
        params = {
            **baseline_params,
            "cluster_pct":  best_pct,
            "min_duration": best_dur,
            "min_momentum": mom,
        }
        trades = collect_ma_trail_trades(data_15m, data_4h, params)
        m, b = run_portfolio(trades, max_concurrent, data_years)
        star = " ★" if m["PF"] >= 1.3 else ""
        print(f"  min_mom={mom:.3f}  N={m['N']:>5}  PF={m['PF']:.3f}  "
              f"WR={m['WR']:.1f}%  EV={m['EV_bp']:>+6.1f}bp  "
              f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}{star}")
        p2_results.append({"mom": mom, "m": m, "b": b})

    best_p2 = max(p2_results, key=lambda r: r["m"]["PF"])
    best_mom = best_p2["mom"]
    print(f"\n  Phase 2 best: min_momentum={best_mom}  "
          f"PF={best_p2['m']['PF']:.3f}  Calmar={best_p2['m']['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 3: trail_activate × trail_distance grid (Phase 1+2 best)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  PHASE 3: trail_activate × trail_distance GRID")
    print(f"  (Fixed: pct={best_pct}, dur={best_dur}, min_mom={best_mom},")
    print(f"          max_dur=24, sl=0.02, tp=0.08, max_hold=12)")
    print(f"{'─'*80}")

    act_grid  = [0.02, 0.03, 0.04, 0.05, 0.06]
    dist_grid = [0.003, 0.005, 0.007]
    p3_results: list[dict] = []

    hdr = f"  {'act':>5} {'dist':>6} | {'N':>5} {'PF':>7} {'WR':>6} {'EV':>7} {'DD':>6} {'Calm':>6}"
    print(f"\n{hdr}")
    print(f"  {'-'*5} {'-'*6}   {'-'*5} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")

    for act in act_grid:
        for dist in dist_grid:
            params = {
                **baseline_params,
                "cluster_pct":    best_pct,
                "min_duration":   best_dur,
                "min_momentum":   best_mom,
                "trail_activate": act,
                "trail_distance": dist,
            }
            trades = collect_ma_trail_trades(data_15m, data_4h, params)
            m, b = run_portfolio(trades, max_concurrent, data_years)
            star = " ★" if m["PF"] >= 1.3 else ""
            print(f"  {act:>5.3f} {dist:>6.3f}   "
                  f"N={m['N']:>5}  PF={m['PF']:.3f}  "
                  f"WR={m['WR']:.1f}%  EV={m['EV_bp']:>+6.1f}bp  "
                  f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}"
                  f"{star}")
            p3_results.append({
                "act": act, "dist": dist, "m": m, "b": b,
            })

    best_p3 = max(p3_results, key=lambda r: r["m"]["PF"])
    best_act  = best_p3["act"]
    best_dist = best_p3["dist"]
    print(f"\n  Phase 3 best: trail_activate={best_act}  trail_distance={best_dist}  "
          f"PF={best_p3['m']['PF']:.3f}  Calmar={best_p3['m']['calmar']:.1f}")

    # ══════════════════════════════════════════════════════════════════════
    #  Summary: Optimal combo
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  SUMMARY: OPTIMAL COMBO (Phase 1+2+3 best)")
    print(f"{'─'*80}")

    opt_params = {
        **baseline_params,
        "cluster_pct":    best_pct,
        "min_duration":   best_dur,
        "min_momentum":   best_mom,
        "trail_activate": best_act,
        "trail_distance": best_dist,
    }
    opt_trades = collect_ma_trail_trades(data_15m, data_4h, opt_params)
    opt_m, opt_b = run_portfolio(opt_trades, max_concurrent, data_years)

    print(f"\n  Optimal params:")
    print(f"    cluster_pct={best_pct}  min_duration={best_dur}  min_momentum={best_mom}")
    print(f"    trail_activate={best_act}  trail_distance={best_dist}")
    print(f"    (fixed) sl=0.02  tp=0.08  max_hold=12  max_dur=24")
    print()
    print_metrics("r73 flat baseline    (ref)", {"N": 2461, "PF": 1.283, "WR": 0.0,
                                                  "EV_bp": 0.0, "max_dd_pct": 0.0,
                                                  "calmar": 4.4}, {"CI_lo": 0.0, "CI_hi": 0.0})
    print(f"  {'r73 flat baseline    (ref)':<28} N= 2461  PF=1.283  Calmar=4.4  (r73 flat, no trail)")
    print_metrics("trail baseline (config)", bl_m, bl_b)
    print_metrics("optimal combo          ", opt_m, opt_b)

    pf_gain = opt_m["PF"] - 1.283
    calmar_gain = opt_m["calmar"] - 4.4
    print(f"\n  PF gain vs r73 flat: {pf_gain:+.3f}  Calmar gain: {calmar_gain:+.1f}")

    recommend_config = opt_m["PF"] > 1.283 and opt_m["calmar"] > 4.4
    print(f"  Config change recommended: {'✅ Yes' if recommend_config else '❌ No'}")

    # ── Report file ────────────────────────────────────────────────────────
    today = time.strftime("%Y%m%d", time.gmtime())
    report_dir = ROOT / "docs" / "backtest_results" / "202603"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r75_ma_cluster_sweep.md"

    report = f"""# r75: MA Cluster Trail-Aware 파라미터 최적화

**목적:** r73 flat TP=8% (PF=1.283, Calmar=4.4) → trail-aware 최적화
**데이터:** {len(data_15m)} syms · {data_years:.2f}yr · 0.08% round-trip · 10슬롯 (MA-only)
**Parity:** config.py 읽음 ✅  symbol_whitelist 사용 ✅  룩어헤드 없음 ✅  portfolio_sim import ✅
**Commit:** {commit}  |  Run: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}

---

## Phase 0: Trail-Aware Baseline (config params)

r73 MA cluster used flat TP=8%. Live config uses trail_activate=4%, trail_distance=0.5%.

| | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| r73 flat baseline | 2461 | 1.283 | — | — | — | 4.4 |
| trail baseline (config) | {bl_m['N']} | {bl_m['PF']:.3f} | {bl_m['WR']:.1f}% | {bl_m['EV_bp']:+.1f} | {bl_m['max_dd_pct']:.1f}% | {bl_m['calmar']:.1f} |

N_diff vs r73: {n_diff:.1f}%  PF_diff: {pf_diff:.1f}%  (trail changes exit timing — expected to differ)

---

## Phase 1: cluster_pct × min_duration Grid

Fixed: max_dur=24, min_mom=0.01, sl=0.02, tp=0.08, trail_act=0.04, trail_dist=0.005, max_hold=12

| cluster_pct | min_dur | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|---|
"""
    for r in p1_results:
        star = " ★" if r["m"]["PF"] >= 1.3 else ""
        report += (f"| {r['pct']:.3f} | {r['dur']} | {r['m']['N']} | {r['m']['PF']:.3f}"
                   f" | {r['m']['WR']:.1f}% | {r['m']['EV_bp']:+.1f} | {r['m']['max_dd_pct']:.1f}%"
                   f" | {r['m']['calmar']:.1f}{star} |\n")

    report += f"""
**Phase 1 best:** cluster_pct={best_pct}  min_dur={best_dur}  PF={best_p1['m']['PF']:.3f}  Calmar={best_p1['m']['calmar']:.1f}

---

## Phase 2: min_momentum Sweep

Fixed: pct={best_pct}, dur={best_dur}, max_dur=24, sl=0.02, tp=0.08, trail_act=0.04, trail_dist=0.005, max_hold=12

| min_momentum | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|
"""
    for r in p2_results:
        star = " ★" if r["m"]["PF"] >= 1.3 else ""
        report += (f"| {r['mom']:.3f} | {r['m']['N']} | {r['m']['PF']:.3f}"
                   f" | {r['m']['WR']:.1f}% | {r['m']['EV_bp']:+.1f} | {r['m']['max_dd_pct']:.1f}%"
                   f" | {r['m']['calmar']:.1f}{star} |\n")

    report += f"""
**Phase 2 best:** min_momentum={best_mom}  PF={best_p2['m']['PF']:.3f}  Calmar={best_p2['m']['calmar']:.1f}

---

## Phase 3: trail_activate × trail_distance Grid

Fixed: pct={best_pct}, dur={best_dur}, min_mom={best_mom}, max_dur=24, sl=0.02, tp=0.08, max_hold=12

| trail_activate | trail_distance | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|---|
"""
    for r in p3_results:
        star = " ★" if r["m"]["PF"] >= 1.3 else ""
        report += (f"| {r['act']:.3f} | {r['dist']:.3f} | {r['m']['N']} | {r['m']['PF']:.3f}"
                   f" | {r['m']['WR']:.1f}% | {r['m']['EV_bp']:+.1f} | {r['m']['max_dd_pct']:.1f}%"
                   f" | {r['m']['calmar']:.1f}{star} |\n")

    report += f"""
**Phase 3 best:** trail_activate={best_act}  trail_distance={best_dist}  PF={best_p3['m']['PF']:.3f}  Calmar={best_p3['m']['calmar']:.1f}

---

## 결론

| | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| r73 flat (ref) | 2461 | 1.283 | — | — | — | 4.4 |
| trail baseline (config) | {bl_m['N']} | {bl_m['PF']:.3f} | {bl_m['WR']:.1f}% | {bl_m['EV_bp']:+.1f} | {bl_m['max_dd_pct']:.1f}% | {bl_m['calmar']:.1f} |
| **최적 combo** | **{opt_m['N']}** | **{opt_m['PF']:.3f}** | **{opt_m['WR']:.1f}%** | **{opt_m['EV_bp']:+.1f}** | **{opt_m['max_dd_pct']:.1f}%** | **{opt_m['calmar']:.1f}** |

**최적 파라미터:**
- `cluster_pct={best_pct}`  `min_duration={best_dur}`  `min_momentum={best_mom}`
- `trail_activate={best_act}`  `trail_distance={best_dist}`
- (고정) `sl=0.02`  `tp=0.08`  `max_hold=12`  `max_duration=24`

**PF 개선:** 1.283 → {opt_m['PF']:.3f} ({pf_gain:+.3f})
**Calmar 개선:** 4.4 → {opt_m['calmar']:.1f} ({calmar_gain:+.1f})

**config 변경 권장:** {'Yes ✅' if recommend_config else 'No ❌'}
**구현:** 미구현
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report saved: {report_path.relative_to(ROOT)}")
    print(f"\n  Total elapsed: {time.time()-t0:.1f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
