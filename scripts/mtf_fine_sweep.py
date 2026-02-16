#!/usr/bin/env python3
"""
Fine-grained MTF sweep — find the sweet spot between more entries and controlled DD.

Explores:
  1. Fine-grained disc × stop around the promising zone (0.85-1.0 × 0.7-1.0)
  2. Different htf_block_threshold with enhanced mode
  3. Different htf_period with enhanced mode

Usage:
    uv run python scripts/mtf_fine_sweep.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import gzip
import pickle
from pathlib import Path
from itertools import product

from quantbox.strategy.rubberband import compute_signals, compute_signals_mtf
from quantbox.strategy.vol_selector import rank_symbols_by_vol
from quantbox.backtest.vectorized_backtest import _backtest_core

# ── Configuration ──
TOP_N = 3
VOL_LOOKBACK = 2880
REBALANCE = 960
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32

LOOKBACK = 8
THRESHOLD = 0.025
TARGET = 0.5
STOP = 1.0
MAX_HOLD = 8
STOP_DECAY = 0.3


def load_data():
    cache_dir = Path("data/cache")
    with open(cache_dir / "cache_index.json") as f:
        index = json.load(f)
    all_data = {}
    for key, meta in index.items():
        fpath = Path(meta["filepath"])
        symbol = meta.get("metadata", {}).get("symbol", key.split("_")[0])
        if fpath.exists():
            with gzip.open(fpath, "rb") as f:
                entry = pickle.load(f)
            df = entry.data
            if "Close" in df.columns and len(df) >= 200:
                all_data[symbol] = df
    return all_data


def run_config(close_dict, symbols, n_bars, htf_period, htf_block, disc, smult):
    """Run a single config and return key metrics."""

    def sig_func(close):
        return compute_signals_mtf(
            close, LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
            htf_period, htf_block, disc, smult,
        )

    # Pre-compute equities
    equities = {}
    results = {}
    all_trade_rets = {}
    for sym in symbols:
        close = close_dict[sym]
        sigs = sig_func(close)
        eq, ret, sharpe, dd, wr, n_trades, bars_in, pf, avg_ret, trade_rets = (
            _backtest_core(close, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        )
        equities[sym] = eq
        results[sym] = {'trades': n_trades, 'wr': wr, 'ret': ret * 100}
        all_trade_rets[sym] = trade_rets

    # Build rolling portfolio
    sym_bar_rets = {}
    for sym in symbols:
        eq = equities[sym]
        bar_rets = np.zeros(n_bars)
        for j in range(1, n_bars):
            bar_rets[j] = eq[j] / eq[j - 1] - 1.0
        sym_bar_rets[sym] = bar_rets

    portfolio_eq = np.ones(n_bars)
    selected_syms = set()

    rebalance_pts = list(range(VOL_LOOKBACK, n_bars, REBALANCE))
    if not rebalance_pts:
        rebalance_pts = [VOL_LOOKBACK]
    if rebalance_pts[-1] < n_bars:
        rebalance_pts.append(n_bars)

    for ri in range(len(rebalance_pts) - 1):
        rp = rebalance_pts[ri]
        next_rp = rebalance_pts[ri + 1]
        ws = max(0, rp - VOL_LOOKBACK)
        ranked = rank_symbols_by_vol(
            {s: close_dict[s][ws:rp] for s in symbols},
            top_n=TOP_N, lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96 * 365.25),
        )
        selected = [sym for sym, _ in ranked]
        for s in selected:
            selected_syms.add(s)
        for j in range(rp, next_rp):
            bar_ret = sum(sym_bar_rets[s][j] for s in selected) / TOP_N
            portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        bar_ret = sum(sym_bar_rets[s][j] for s in symbols) / len(symbols)
        portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    # Post-warmup metrics
    post_eq = portfolio_eq[VOL_LOOKBACK:]
    post_ret = (post_eq[-1] / post_eq[0] - 1.0) * 100
    post_peak = np.maximum.accumulate(post_eq)
    post_dd = np.max((post_peak - post_eq) / post_peak) * 100
    post_calmar = post_ret / post_dd if post_dd > 0 else 999.0

    rets = np.diff(post_eq) / post_eq[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0.0

    total_trades = sum(results[s]['trades'] for s in selected_syms)

    # Trade stats
    all_tr = []
    for s in selected_syms:
        for r in all_trade_rets[s]:
            all_tr.append(r)
    all_tr = np.array(all_tr) if all_tr else np.array([0.0])
    wins = all_tr[all_tr > 0]
    losses = all_tr[all_tr <= 0]
    win_rate = len(wins) / len(all_tr) * 100 if len(all_tr) > 0 else 0.0
    avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0.0
    avg_loss = np.mean(np.abs(losses)) * 100 if len(losses) > 0 else 0.0
    rr = avg_win / avg_loss if avg_loss > 0 else 999.0
    pf = np.sum(wins) / np.sum(np.abs(losses)) if np.sum(np.abs(losses)) > 0 else 999.0

    return {
        'ret': post_ret, 'dd': post_dd, 'calmar': post_calmar,
        'sharpe': sharpe, 'trades': total_trades, 'wr': win_rate,
        'rr': rr, 'pf': pf, 'avg_win': avg_win, 'avg_loss': avg_loss,
    }


def main():
    print("Loading data...")
    all_data = load_data()
    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    close_dict = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64)
                  for s in symbols}
    days = n_bars * 15 / 60 / 24
    print(f"Loaded {len(symbols)} symbols, {n_bars} bars ({days:.0f} days)")

    # ── 1. Fine-grained disc × stop near the sweet spot ──
    print("\n" + "█" * 60)
    print("  1. Fine-grained DISC × STOP sweep (htf=48, block=0.02)")
    print("█" * 60)

    discounts = [0.85, 0.90, 0.92, 0.95, 0.97, 1.0]
    stop_mults = [0.70, 0.80, 0.85, 0.90, 0.95, 1.0]

    results_1 = []
    for disc, sm in product(discounts, stop_mults):
        r = run_config(close_dict, symbols, n_bars, 48, 0.02, disc, sm)
        results_1.append((disc, sm, r))

    results_1.sort(key=lambda x: x[2]['calmar'], reverse=True)

    print(f"\n{'Disc':>6} {'Stop':>6} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR%':>6} {'RR':>5} {'PF':>5}")
    print("-" * 85)
    for disc, sm, r in results_1:
        print(f"{disc:>6.2f} {sm:>6.2f} {r['ret']:>+8.2f} {r['dd']:>7.2f} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} {r['trades']:>7} "
              f"{r['wr']:>5.1f}% {r['rr']:>5.2f} {r['pf']:>5.2f}")

    # ── 2. Vary block threshold with best disc/stop ──
    best_disc, best_stop = results_1[0][0], results_1[0][1]
    print(f"\n\n{'█' * 60}")
    print(f"  2. Block threshold sweep (disc={best_disc}, stop={best_stop})")
    print("█" * 60)

    block_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    results_2 = []
    for bt in block_thresholds:
        r = run_config(close_dict, symbols, n_bars, 48, bt, best_disc, best_stop)
        results_2.append((bt, r))

    results_2.sort(key=lambda x: x[1]['calmar'], reverse=True)

    print(f"\n{'Block':>8} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR%':>6} {'RR':>5}")
    print("-" * 65)
    for bt, r in results_2:
        print(f"{bt:>8.3f} {r['ret']:>+8.2f} {r['dd']:>7.2f} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} {r['trades']:>7} "
              f"{r['wr']:>5.1f}% {r['rr']:>5.2f}")

    best_block = results_2[0][0]

    # ── 3. Vary HTF period with best disc/stop/block ──
    print(f"\n\n{'█' * 60}")
    print(f"  3. HTF period sweep (disc={best_disc}, stop={best_stop}, block={best_block})")
    print("█" * 60)

    htf_periods = [24, 32, 40, 48, 56, 64, 72, 96]
    results_3 = []
    for hp in htf_periods:
        r = run_config(close_dict, symbols, n_bars, hp, best_block, best_disc, best_stop)
        results_3.append((hp, r))

    results_3.sort(key=lambda x: x[1]['calmar'], reverse=True)

    print(f"\n{'Period':>8} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR%':>6} {'RR':>5}")
    print("-" * 65)
    for hp, r in results_3:
        hrs = hp * 15 / 60
        print(f"{hp:>5} ({hrs:.0f}h) {r['ret']:>+8.2f} {r['dd']:>7.2f} "
              f"{r['calmar']:>7.2f} {r['sharpe']:>7.2f} {r['trades']:>7} "
              f"{r['wr']:>5.1f}% {r['rr']:>5.2f}")

    best_period = results_3[0][0]

    # ── Final comparison ──
    print(f"\n\n{'=' * 60}")
    print(f"  FINAL: BEST vs V1 BASELINE")
    print(f"{'=' * 60}")

    v1 = run_config(close_dict, symbols, n_bars, 48, 0.02, 1.0, 1.0)
    best_r = run_config(close_dict, symbols, n_bars, best_period, best_block,
                         best_disc, best_stop)

    print(f"\n  V1 (block-only, period=48, block=0.02):")
    print(f"    Return: {v1['ret']:+.2f}%, DD: {v1['dd']:.2f}%, Calmar: {v1['calmar']:.2f}")
    print(f"    Sharpe: {v1['sharpe']:.2f}, Trades: {v1['trades']}, WR: {v1['wr']:.1f}%")
    print(f"    Avg Win: {v1['avg_win']:.3f}%, Avg Loss: {v1['avg_loss']:.3f}%, RR: {v1['rr']:.2f}")
    print(f"    PF: {v1['pf']:.2f}")

    print(f"\n  BEST (period={best_period}, block={best_block}, disc={best_disc}, stop={best_stop}):")
    print(f"    Return: {best_r['ret']:+.2f}%, DD: {best_r['dd']:.2f}%, Calmar: {best_r['calmar']:.2f}")
    print(f"    Sharpe: {best_r['sharpe']:.2f}, Trades: {best_r['trades']}, WR: {best_r['wr']:.1f}%")
    print(f"    Avg Win: {best_r['avg_win']:.3f}%, Avg Loss: {best_r['avg_loss']:.3f}%, RR: {best_r['rr']:.2f}")
    print(f"    PF: {best_r['pf']:.2f}")

    delta_ret = best_r['ret'] - v1['ret']
    delta_dd = best_r['dd'] - v1['dd']
    delta_calmar = best_r['calmar'] - v1['calmar']
    print(f"\n  DELTA: Ret {delta_ret:+.2f}%, DD {delta_dd:+.2f}%, Calmar {delta_calmar:+.2f}")

    if best_r['calmar'] > v1['calmar']:
        print(f"\n  >>> VERDICT: Enhanced BEATS v1! Use period={best_period}, "
              f"block={best_block}, disc={best_disc}, stop={best_stop}")
    elif best_r['ret'] > v1['ret'] * 1.15 and best_r['dd'] < v1['dd'] * 1.25:
        print(f"\n  >>> VERDICT: Enhanced has 15%+ more return with manageable DD. "
              f"Consider adopting.")
    else:
        print(f"\n  >>> VERDICT: V1 (block-only) remains optimal on risk-adjusted basis.")

    # Output recommended config
    print(f"\n  RECOMMENDED PRODUCTION CONFIG:")
    print(f"    htf_period = {best_period}")
    print(f"    htf_block_threshold = {best_block}")
    print(f"    htf_agree_discount = {best_disc}")
    print(f"    htf_agree_stop_mult = {best_stop}")


if __name__ == "__main__":
    main()
