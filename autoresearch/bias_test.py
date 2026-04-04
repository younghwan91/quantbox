#!/usr/bin/env python3
"""Bias Test Suite for Donchian + Volume Spike Strategy.

Tests:
1. Temporal stability — yearly PF breakdown (all 6 years profitable?)
2. OOS validation — train on first 4 years, test on last 2
3. Direction bias — LONG vs SHORT performance
4. Symbol concentration — top/bottom symbol contribution
5. Trade clustering — temporal distribution of trades
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (  # noqa: E402
    build_resampled,
    bootstrap_pf,
    compute_metrics,
    portfolio_sim,
    portfolio_sim_with_details,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import compute_momentum  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache_extended"
DATA_YEARS = 6.0
SLOT_LIMIT = 10

# Best config from autoresearch
DONCHIAN_PERIOD = 50
MOM_MIN = 0.02
VOL_SMA_PERIOD = 20
VOL_SPIKE_MULT = 2.5
SL = 0.015
TP = 0.15
TRAIL_ACTIVATE = 0.02
TRAIL_DISTANCE = 0.008
MAX_HOLD = 24
COST = 0.0008
WARMUP = 55


def load_data():
    cfg = VolStraddleConfig()
    whitelist = set(cfg.symbol_whitelist)
    data_15m: dict = {}
    max_bars = 0
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        d = np.load(f, allow_pickle=False)
        bars = len(d["close"])
        if bars > max_bars:
            max_bars = bars
        data_15m[sym] = {
            "close": d["close"].astype(np.float64),
            "high": d["high"].astype(np.float64),
            "low": d["low"].astype(np.float64),
            "volume": d["volume"].astype(np.float64),
        }
    for sym in data_15m:
        data_15m[sym]["_offset"] = max_bars - len(data_15m[sym]["close"])
        data_15m[sym]["mom_8"] = compute_momentum(data_15m[sym]["close"], 8)
    data_4h = build_resampled(data_15m, 16)
    for d in data_4h.values():
        d["mom_8"] = compute_momentum(d["close"], 8)
    precompute_all_vr([data_15m, data_4h], [(8, 56)])
    return data_15m, data_4h, max_bars


@numba.jit(nopython=True, cache=True)
def compute_vol_sma(volume, period):
    n = len(volume)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += volume[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += volume[i] - volume[i - period]
        out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def sim_donchian_with_direction(
    close, high, low, volume, vol_sma,
    period, mom_min, vol_mult, sl, tp, max_hold, cost,
    trail_activate, trail_distance, warmup,
):
    """Returns pnl, entry_bars, exit_bars, directions (1=LONG, -1=SHORT)."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    dirs = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        # Volume filter
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 0 or volume[i] < vol_mult * vol_sma[i]:
            i += 1
            continue
        # Donchian channel
        chan_high = -1e18
        chan_low = 1e18
        for j in range(i - period, i):
            if high[j] > chan_high:
                chan_high = high[j]
            if low[j] < chan_low:
                chan_low = low[j]
        if close[i] <= 0:
            i += 1
            continue

        d = 0
        if close[i] > chan_high:
            bp = (close[i] - chan_high) / close[i]
            if bp > mom_min:
                d = 1
        elif close[i] < chan_low:
            bp = (chan_low - close[i]) / close[i]
            if bp > mom_min:
                d = -1

        if d == 0:
            i += 1
            continue

        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j_idx in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j_idx] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j_idx] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j_idx
                        break
                else:
                    if low[j_idx] <= entry * (1.0 - sl):
                        trade_pnl = -sl
                        eb = j_idx
                        break
                if high[j_idx] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j_idx
                    break
            else:
                cur_gain = (entry - low[j_idx]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j_idx] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j_idx
                        break
                else:
                    if high[j_idx] >= entry * (1.0 + sl):
                        trade_pnl = -sl
                        eb = j_idx
                        break
                if low[j_idx] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j_idx
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            dirs[cnt] = d
            cnt += 1
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt], dirs[:cnt]


def collect_all_trades_with_meta(data_4h, data_15m):
    """Collect trades with direction and symbol metadata."""
    all_trades = []  # (sym, entry_15m, exit_15m, pnl, direction)
    for sym in sorted(data_4h.keys()):
        d = data_4h[sym]
        if "volume" not in d:
            continue
        vol_sma = compute_vol_sma(d["volume"], VOL_SMA_PERIOD)
        p, e, x, dirs = sim_donchian_with_direction(
            d["close"], d["high"], d["low"], d["volume"], vol_sma,
            DONCHIAN_PERIOD, MOM_MIN, VOL_SPIKE_MULT,
            SL, TP, MAX_HOLD, COST,
            TRAIL_ACTIVATE, TRAIL_DISTANCE, WARMUP,
        )
        offset = int(data_15m[sym].get("_offset", 0))
        for k in range(len(p)):
            entry_bar = int(e[k]) * 16 + offset
            exit_bar = int(x[k]) * 16 + 15 + offset
            all_trades.append((sym, entry_bar, exit_bar, float(p[k]), int(dirs[k])))
    return all_trades


def pf_from_pnls(pnls):
    arr = np.array(pnls)
    if len(arr) == 0:
        return 0.0
    w = arr[arr > 0].sum()
    l = abs(arr[arr <= 0].sum())
    return w / l if l > 0 else 999.0


def main():
    print("=" * 72)
    print("  BIAS TEST SUITE — Donchian + Volume Spike (4h)")
    print("=" * 72)
    print("\nLoading data...")
    data_15m, data_4h, max_bars = load_data()

    print("Collecting all trades with metadata...")
    all_trades_meta = collect_all_trades_with_meta(data_4h, data_15m)
    print(f"  Raw trades (pre-slot filter): {len(all_trades_meta)}")

    # Standard portfolio-filtered trades
    trades_for_sim = [(s, e, x, p) for s, e, x, p, d in all_trades_meta]
    pnl_arr, accepted = portfolio_sim_with_details(trades_for_sim, SLOT_LIMIT)
    accepted_set = set((s, e, x) for s, e, x, p in accepted)
    accepted_meta = [t for t in all_trades_meta if (t[0], t[1], t[2]) in accepted_set]
    print(f"  Accepted trades (10-slot): {len(accepted_meta)}")

    bars_per_year = max_bars / DATA_YEARS

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Temporal Stability — Yearly PF
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 1: TEMPORAL STABILITY — Yearly PF")
    print(f"{'=' * 72}")

    yearly: dict[int, list[float]] = {}
    for sym, entry, exit_b, pnl_val, dire in accepted_meta:
        year = min(int(2020 + entry / bars_per_year), 2025)
        yearly.setdefault(year, []).append(pnl_val)

    all_profitable = True
    print(f"\n  {'Year':>6} | {'N':>6} {'PF':>7} {'WR':>6} {'EV_bp':>7} | {'Status'}")
    print(f"  {'-'*6} | {'-'*6} {'-'*7} {'-'*6} {'-'*7} | {'-'*6}")
    for y in sorted(yearly):
        pnls = np.array(yearly[y])
        n = len(pnls)
        pf = pf_from_pnls(pnls)
        wr = len(pnls[pnls > 0]) / n * 100 if n > 0 else 0
        ev = float(np.mean(pnls)) * 10000 if n > 0 else 0
        status = "✓" if pf > 1.0 else "✗ LOSING"
        if pf <= 1.0:
            all_profitable = False
        print(f"  {y:>6} | {n:>6} {pf:7.3f} {wr:5.1f}% {ev:7.1f} | {status}")
    print(f"\n  All years profitable: {'YES' if all_profitable else 'NO — TEMPORAL BIAS DETECTED'}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: OOS Validation — First 4yr train, Last 2yr test
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 2: OUT-OF-SAMPLE VALIDATION (4yr train / 2yr test)")
    print(f"{'=' * 72}")

    cutoff = int(4.0 * bars_per_year)
    train_trades = [(s, e, x, p) for s, e, x, p, d in all_trades_meta if e < cutoff]
    test_trades = [(s, e, x, p) for s, e, x, p, d in all_trades_meta if e >= cutoff]

    train_pnl = portfolio_sim(train_trades, SLOT_LIMIT)
    test_pnl = portfolio_sim(test_trades, SLOT_LIMIT)

    train_m = compute_metrics(train_pnl, 4.0)
    test_m = compute_metrics(test_pnl, 2.0)

    train_boot = bootstrap_pf(train_pnl, 3000)
    test_boot = bootstrap_pf(test_pnl, 3000)

    print(f"\n  {'':>12} | {'N':>6} {'PF':>7} {'WR':>6} {'MaxDD':>7} {'EV_bp':>7} | {'CI':>14} {'P≥1':>5}")
    print(f"  {'-'*12} | {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7} | {'-'*14} {'-'*5}")
    print(
        f"  {'Train (4yr)':>12} | {train_m['N']:>6} {train_m['PF']:7.3f} {train_m['WR']:5.1f}% "
        f"{train_m['max_dd_pct']:6.1f}% {train_m['EV_bp']:7.1f} | "
        f"[{train_boot.get('CI_lo', 0):.2f},{train_boot.get('CI_hi', 0):.2f}] {train_boot.get('P_PF_ge_1', 0):4.0f}%"
    )
    print(
        f"  {'Test (2yr)':>12} | {test_m['N']:>6} {test_m['PF']:7.3f} {test_m['WR']:5.1f}% "
        f"{test_m['max_dd_pct']:6.1f}% {test_m['EV_bp']:7.1f} | "
        f"[{test_boot.get('CI_lo', 0):.2f},{test_boot.get('CI_hi', 0):.2f}] {test_boot.get('P_PF_ge_1', 0):4.0f}%"
    )

    if train_m["PF"] > 0:
        decay = max(0, 1 - test_m["PF"] / train_m["PF"])
        print(f"\n  OOS decay: {decay*100:.1f}% (test PF / train PF = {test_m['PF']/train_m['PF']:.2f})")
        oos_pass = test_m["PF"] > 1.3 and test_boot.get("P_PF_ge_1", 0) >= 95
        print(f"  OOS verdict: {'PASS' if oos_pass else 'FAIL — OOS DEGRADATION'}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Direction Bias — LONG vs SHORT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 3: DIRECTION BIAS — LONG vs SHORT")
    print(f"{'=' * 72}")

    long_pnls = [p for s, e, x, p, d in accepted_meta if d == 1]
    short_pnls = [p for s, e, x, p, d in accepted_meta if d == -1]

    print(f"\n  {'Direction':>10} | {'N':>6} {'PF':>7} {'WR':>6} {'EV_bp':>7}")
    print(f"  {'-'*10} | {'-'*6} {'-'*7} {'-'*6} {'-'*7}")
    for label, pnls in [("LONG", long_pnls), ("SHORT", short_pnls)]:
        arr = np.array(pnls) if pnls else np.empty(0)
        n = len(arr)
        pf = pf_from_pnls(arr) if n > 0 else 0
        wr = len(arr[arr > 0]) / n * 100 if n > 0 else 0
        ev = float(np.mean(arr)) * 10000 if n > 0 else 0
        print(f"  {label:>10} | {n:>6} {pf:7.3f} {wr:5.1f}% {ev:7.1f}")

    long_pf = pf_from_pnls(np.array(long_pnls)) if long_pnls else 0
    short_pf = pf_from_pnls(np.array(short_pnls)) if short_pnls else 0
    if long_pf > 0 and short_pf > 0:
        ratio = max(long_pf, short_pf) / min(long_pf, short_pf)
        biased = ratio > 3.0
        dominant = "LONG" if long_pf > short_pf else "SHORT"
        print(f"\n  PF ratio (max/min): {ratio:.1f}x ({dominant} dominant)")
        print(f"  Direction bias: {'YES — {dominant} ONLY' if biased else 'NO — both directions profitable'}")
    elif long_pf <= 1.0 or short_pf <= 1.0:
        losing_dir = "LONG" if long_pf <= 1.0 else "SHORT"
        print(f"\n  Direction bias: YES — {losing_dir} is unprofitable (PF≤1.0)")

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Symbol Concentration
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 4: SYMBOL CONCENTRATION")
    print(f"{'=' * 72}")

    sym_pnl: dict[str, list[float]] = {}
    for s, e, x, p, d in accepted_meta:
        sym_pnl.setdefault(s, []).append(p)

    sym_stats = []
    for s, pnls in sym_pnl.items():
        arr = np.array(pnls)
        total = float(arr.sum())
        sym_stats.append((s, len(arr), pf_from_pnls(arr), total))

    sym_stats.sort(key=lambda x: x[3], reverse=True)
    total_pnl_sum = sum(t for _, _, _, t in sym_stats)

    print(f"\n  Total symbols traded: {len(sym_stats)}")
    profitable = sum(1 for _, _, pf, _ in sym_stats if pf > 1.0)
    print(f"  Profitable symbols: {profitable}/{len(sym_stats)} ({profitable/len(sym_stats)*100:.0f}%)")

    print(f"\n  Top 10 contributors:")
    print(f"  {'Symbol':>12} | {'N':>5} {'PF':>7} {'TotalPnL':>10} {'Cum%':>6}")
    print(f"  {'-'*12} | {'-'*5} {'-'*7} {'-'*10} {'-'*6}")
    cum = 0.0
    for s, n, pf, total in sym_stats[:10]:
        cum += total
        cum_pct = cum / total_pnl_sum * 100 if total_pnl_sum > 0 else 0
        print(f"  {s:>12} | {n:>5} {pf:7.3f} {total:10.4f} {cum_pct:5.1f}%")

    print(f"\n  Bottom 5:")
    for s, n, pf, total in sym_stats[-5:]:
        print(f"  {s:>12} | {n:>5} {pf:7.3f} {total:10.4f}")

    top10_pct = sum(t for _, _, _, t in sym_stats[:10]) / total_pnl_sum * 100 if total_pnl_sum > 0 else 0
    concentrated = top10_pct > 80
    print(f"\n  Top 10 symbols = {top10_pct:.1f}% of total PnL")
    print(f"  Concentration risk: {'HIGH' if concentrated else 'OK'}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Trade Clustering — Temporal Distribution
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 5: TRADE CLUSTERING — Temporal Distribution")
    print(f"{'=' * 72}")

    # Group trades by quarter
    quarterly: dict[str, list[float]] = {}
    for s, entry, exit_b, p, d in accepted_meta:
        year = int(2020 + entry / bars_per_year)
        quarter = (int((entry % bars_per_year) / (bars_per_year / 4)) % 4) + 1
        key = f"{year}Q{quarter}"
        quarterly.setdefault(key, []).append(p)

    print(f"\n  {'Quarter':>8} | {'N':>5} {'PF':>7} {'EV_bp':>7}")
    print(f"  {'-'*8} | {'-'*5} {'-'*7} {'-'*7}")
    empty_quarters = 0
    losing_quarters = 0
    for q in sorted(quarterly):
        arr = np.array(quarterly[q])
        n = len(arr)
        pf = pf_from_pnls(arr)
        ev = float(np.mean(arr)) * 10000 if n > 0 else 0
        marker = "" if pf > 1.0 else " ← LOSING"
        if pf <= 1.0:
            losing_quarters += 1
        print(f"  {q:>8} | {n:>5} {pf:7.3f} {ev:7.1f}{marker}")

    total_q = len(quarterly)
    print(f"\n  Total quarters: {total_q}")
    print(f"  Losing quarters: {losing_quarters}/{total_q} ({losing_quarters/total_q*100:.0f}%)")

    # Check for long gaps with no trades
    entries = sorted([e for _, e, _, _, _ in accepted_meta])
    if len(entries) > 1:
        gaps = np.diff(entries)
        max_gap_bars = int(np.max(gaps))
        max_gap_days = max_gap_bars / (96)  # 96 15m-bars per day
        median_gap = float(np.median(gaps)) / 96
        print(f"  Max gap between trades: {max_gap_days:.1f} days")
        print(f"  Median gap: {median_gap:.1f} days")

    # ══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  FINAL BIAS VERDICT")
    print(f"{'=' * 72}")

    issues = []
    if not all_profitable:
        issues.append("Temporal: not all years profitable")
    if test_m["PF"] < 1.3:
        issues.append(f"OOS: test PF={test_m['PF']:.3f} < 1.3")
    if long_pf <= 1.0 or short_pf <= 1.0:
        issues.append("Direction: one direction is unprofitable")
    if concentrated:
        issues.append(f"Concentration: top 10 = {top10_pct:.0f}% of PnL")
    if losing_quarters > total_q * 0.4:
        issues.append(f"Clustering: {losing_quarters}/{total_q} losing quarters")

    if not issues:
        print("\n  ALL BIAS TESTS PASSED")
    else:
        print(f"\n  ISSUES DETECTED ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")

    print()
    print("---")
    print(f"primary_metric: {len(issues)}")


if __name__ == "__main__":
    main()
