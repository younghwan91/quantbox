#!/usr/bin/env python3
"""
Mega Alpha Sweep — Find the legendary edge

Tests multiple strategy variants and parameter combinations:
  1. Enhanced Rubber Band (volume + momentum + adaptive ATR)
  2. Pure Momentum Breakout
  3. Hybrid: Momentum direction + Rubber band timing
  4. Symbol filtering (top-N only)

Outputs: ranked results sorted by Sharpe × Return for practical edge.

Usage:
    uv run python scripts/mega_sweep.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import gzip
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from itertools import product

from quantbox.strategy.alpha_engine import (
    compute_alpha_signals,
    compute_momentum_signals,
    compute_hybrid_signals,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

# ── Load data ──
CACHE = Path("data/cache")
with open(CACHE / "cache_index.json") as f:
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

symbols = sorted(all_data.keys())
n_bars = min(len(all_data[s]) for s in symbols)
print(f"Data: {len(symbols)} symbols × {n_bars} bars (15m)")
print(f"Period: {n_bars * 15 / 60 / 24:.0f} days\n")

# ── Prepare aligned arrays ──
close_arrays = {}
high_arrays = {}
low_arrays = {}
volume_arrays = {}
taker_buy_arrays = {}

for sym in symbols:
    df = all_data[sym]
    close_arrays[sym] = df["Close"].values[:n_bars].astype(np.float64)
    high_arrays[sym] = df["High"].values[:n_bars].astype(np.float64)
    low_arrays[sym] = df["Low"].values[:n_bars].astype(np.float64)
    volume_arrays[sym] = df["Volume"].values[:n_bars].astype(np.float64)
    if "Taker Buy Base Asset Volume" in df.columns:
        taker_buy_arrays[sym] = df["Taker Buy Base Asset Volume"].values[:n_bars].astype(np.float64)
    else:
        taker_buy_arrays[sym] = volume_arrays[sym] * 0.5

# Costs
COST_PER_SIDE = 0.0004   # 0.04% per side (maker + slippage)
FUNDING_PER_BAR = 0.0001 / 32  # 0.01%/8h → per 15m bar


def run_strategy_on_all(signal_func, symbols_subset=None):
    """Run a signal function on all symbols, return portfolio metrics."""
    syms = symbols_subset or symbols
    n_sym = len(syms)
    portfolio_eq = np.ones(n_bars)

    per_symbol = {}
    for sym in syms:
        sigs = signal_func(sym)
        eq, total_ret, sharpe, max_dd, wr, n_trades, bars_in, pf, avg_ret, _ = (
            _backtest_core(close_arrays[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        )
        portfolio_eq *= eq[:n_bars] ** (1.0 / n_sym)
        per_symbol[sym] = {
            'ret': total_ret * 100, 'sharpe': sharpe,
            'dd': max_dd * 100, 'wr': wr, 'trades': n_trades,
            'pf': pf, 'avg_ret': avg_ret * 100,
        }

    port_ret = (portfolio_eq[-1] - 1.0) * 100
    peak = np.maximum.accumulate(portfolio_eq)
    port_dd = np.max((peak - portfolio_eq) / peak) * 100
    rets = np.diff(portfolio_eq) / portfolio_eq[:-1]
    port_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0
    positive = sum(1 for v in per_symbol.values() if v['ret'] > 0)
    total_trades = sum(v['trades'] for v in per_symbol.values())
    avg_pf = np.mean([v['pf'] for v in per_symbol.values() if v['pf'] < 100])

    return {
        'port_ret': port_ret,
        'port_sharpe': port_sharpe,
        'port_dd': port_dd,
        'positive': positive,
        'n_sym': n_sym,
        'total_trades': total_trades,
        'avg_pf': avg_pf,
        'per_symbol': per_symbol,
        'portfolio_eq': portfolio_eq,
    }


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 1: Enhanced Rubber Band (with filters)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  STRATEGY 1: Enhanced Rubber Band (Volume + Momentum + Adaptive ATR)")
print("=" * 80)

rb_configs = [
    # (name, lookback, threshold, target, stop, max_hold,
    #  vol_filter, vol_lookback, vol_mult,
    #  momentum_filter, momentum_lookback, momentum_mode,
    #  adaptive, atr_lookback, atr_mult,
    #  trailing, trail_act, trail_dist,
    #  taker, taker_lb, taker_thr,
    #  breakout, breakout_lb)

    # Baseline (original rubber band, no filters)
    ("RB_baseline", 8, 0.025, 0.5, 1.0, 8,
     False, 48, 1.5, False, 96, 0, False, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    # Volume filter only
    ("RB+Vol1.5", 8, 0.025, 0.5, 1.0, 8,
     True, 48, 1.5, False, 96, 0, False, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    ("RB+Vol2.0", 8, 0.025, 0.5, 1.0, 8,
     True, 48, 2.0, False, 96, 0, False, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    # Momentum filter only (counter-trend mode = original logic)
    ("RB+Mom_with", 8, 0.025, 0.5, 1.0, 8,
     False, 48, 1.5, True, 96, 0, False, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    ("RB+Mom_ctr", 8, 0.025, 0.5, 1.0, 8,
     False, 48, 1.5, True, 96, 1, False, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    # Adaptive threshold only
    ("RB+ATR2.0", 8, 0.025, 0.5, 1.0, 8,
     False, 48, 1.5, False, 96, 0, True, 48, 2.0,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    ("RB+ATR3.0", 8, 0.025, 0.5, 1.0, 8,
     False, 48, 1.5, False, 96, 0, True, 48, 3.0,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    # Enhanced trailing stop only
    ("RB+Trail", 8, 0.025, 0.6, 1.0, 12,
     False, 48, 1.5, False, 96, 0, False, 48, 2.5,
     True, 0.3, 0.4, False, 16, 0.55, False, 32),

    # ── Combo: Vol + Momentum ──
    ("RB+Vol+Mom_w", 8, 0.025, 0.5, 1.0, 8,
     True, 48, 1.5, True, 96, 0, False, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    # ── Combo: Vol + ATR ──
    ("RB+Vol+ATR", 8, 0.025, 0.5, 1.0, 8,
     True, 48, 1.5, False, 96, 0, True, 48, 2.5,
     False, 0.3, 0.4, False, 16, 0.55, False, 32),

    # ── Combo: All filters ──
    ("RB+All_v1", 8, 0.025, 0.6, 1.0, 12,
     True, 48, 1.5, True, 96, 0, True, 48, 2.5,
     True, 0.3, 0.4, False, 16, 0.55, False, 32),

    ("RB+All_v2", 8, 0.025, 0.6, 1.0, 12,
     True, 48, 2.0, True, 96, 0, True, 48, 2.5,
     True, 0.25, 0.35, False, 16, 0.55, False, 32),

    # ── Tighter stop, wider target ──
    ("RB+All_wide", 8, 0.025, 0.8, 0.8, 12,
     True, 48, 1.5, True, 96, 0, True, 48, 2.5,
     True, 0.3, 0.3, False, 16, 0.55, False, 32),

    # ── Higher threshold ──
    ("RB+All_T3.5", 8, 0.035, 0.6, 1.0, 12,
     True, 48, 1.5, True, 96, 0, True, 48, 2.5,
     True, 0.3, 0.4, False, 16, 0.55, False, 32),

    # ── With taker filter ──
    ("RB+Taker", 8, 0.025, 0.6, 1.0, 12,
     True, 48, 1.5, True, 96, 0, True, 48, 2.5,
     True, 0.3, 0.4, True, 16, 0.55, False, 32),

    # ── Longer lookback ──
    ("RB+L12_All", 12, 0.025, 0.6, 1.0, 12,
     True, 48, 1.5, True, 96, 0, True, 48, 2.5,
     True, 0.3, 0.4, False, 16, 0.55, False, 32),

    # ── Shorter lookback, more aggressive ──
    ("RB+L4_All", 4, 0.02, 0.5, 1.0, 8,
     True, 48, 1.5, True, 48, 0, True, 48, 2.0,
     True, 0.3, 0.4, False, 16, 0.55, False, 32),
]

t0 = time.time()
rb_results = []

for cfg in rb_configs:
    name = cfg[0]
    params = cfg[1:]

    def sig_func(sym, p=params):
        return compute_alpha_signals(
            close_arrays[sym], high_arrays[sym], low_arrays[sym],
            volume_arrays[sym], taker_buy_arrays[sym],
            p[0], p[1], p[2], p[3], p[4],   # lookback, threshold, target, stop, max_hold
            p[5], p[6], p[7],                # vol_filter, vol_lookback, vol_mult
            p[8], p[9], p[10],               # momentum_filter, momentum_lookback, momentum_mode
            p[11], p[12], p[13],             # adaptive, atr_lookback, atr_mult
            p[14], p[15], p[16],             # trailing, trail_act, trail_dist
            p[17], p[18], p[19],             # taker, taker_lb, taker_thr
            p[20], p[21],                     # breakout, breakout_lb
        )

    result = run_strategy_on_all(sig_func)
    result['name'] = name
    rb_results.append(result)

    # Also test with top-10 symbols only
    top10 = sorted(result['per_symbol'].items(), key=lambda x: x[1]['ret'], reverse=True)[:10]
    top10_syms = [s for s, _ in top10]

    result_top10 = run_strategy_on_all(sig_func, top10_syms)
    result_top10['name'] = name + "_top10"
    rb_results.append(result_top10)

# Print results
print(f"\n{'Name':<22} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} {'PF':>6} {'Trades':>7} "
      f"{'Win':>5} {'+Sym':>5}")
print("-" * 80)
for r in sorted(rb_results, key=lambda x: x['port_sharpe'], reverse=True):
    print(f"{r['name']:<22} {r['port_ret']:+8.2f} {r['port_sharpe']:8.2f} "
          f"{r['port_dd']:7.2f} {r['avg_pf']:6.2f} {r['total_trades']:7d} "
          f"{np.mean([v['wr'] for v in r['per_symbol'].values()]):5.1f} "
          f"{r['positive']:>3}/{r['n_sym']}")

print(f"\nTime: {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 2: Pure Momentum Breakout
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STRATEGY 2: Pure Momentum Breakout")
print("=" * 80)

mom_configs = [
    # (name, fast_lb, slow_lb, breakout_lb, atr_lb, vol_lb, vol_mult, trail_atr, max_hold)
    ("MOM_16_48_24", 16, 48, 24, 24, 48, 1.5, 2.0, 32),
    ("MOM_8_32_16", 8, 32, 16, 24, 48, 1.5, 2.0, 24),
    ("MOM_32_96_48", 32, 96, 48, 48, 48, 1.0, 2.5, 48),
    ("MOM_16_48_32", 16, 48, 32, 24, 48, 1.0, 3.0, 48),
    ("MOM_fast_8_24", 8, 24, 12, 16, 32, 1.5, 1.5, 16),
    ("MOM_slow_48_144", 48, 144, 96, 48, 48, 1.0, 3.0, 96),
]

mom_results = []
t0 = time.time()

for cfg in mom_configs:
    name = cfg[0]
    params = cfg[1:]

    def sig_func(sym, p=params):
        return compute_momentum_signals(
            close_arrays[sym], high_arrays[sym], low_arrays[sym],
            volume_arrays[sym],
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        )

    result = run_strategy_on_all(sig_func)
    result['name'] = name
    mom_results.append(result)

    # Top 10
    top10 = sorted(result['per_symbol'].items(), key=lambda x: x[1]['ret'], reverse=True)[:10]
    top10_syms = [s for s, _ in top10]
    result_top10 = run_strategy_on_all(sig_func, top10_syms)
    result_top10['name'] = name + "_top10"
    mom_results.append(result_top10)

print(f"\n{'Name':<22} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} {'PF':>6} {'Trades':>7} "
      f"{'Win':>5} {'+Sym':>5}")
print("-" * 80)
for r in sorted(mom_results, key=lambda x: x['port_sharpe'], reverse=True):
    print(f"{r['name']:<22} {r['port_ret']:+8.2f} {r['port_sharpe']:8.2f} "
          f"{r['port_dd']:7.2f} {r['avg_pf']:6.2f} {r['total_trades']:7d} "
          f"{np.mean([v['wr'] for v in r['per_symbol'].values()]):5.1f} "
          f"{r['positive']:>3}/{r['n_sym']}")

print(f"\nTime: {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY 3: Hybrid (Momentum Dir + Rubber Band Timing)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  STRATEGY 3: Hybrid (Trend Direction + Rubber Band Timing)")
print("=" * 80)

hybrid_configs = [
    # (name, trend_lb, trend_thr, rb_lb, rb_thr, rb_target, rb_stop, rb_hold,
    #  vol_lb, vol_mult, atr_lb, atr_mult, trail_act, trail_dist)
    ("HYB_96_2_8_25", 96, 0.02, 8, 0.025, 0.6, 1.0, 12, 48, 1.5, 48, 2.5, 0.3, 0.4),
    ("HYB_48_15_8_20", 48, 0.015, 8, 0.020, 0.5, 1.0, 12, 48, 1.5, 48, 2.5, 0.3, 0.4),
    ("HYB_96_1_8_25", 96, 0.01, 8, 0.025, 0.6, 1.0, 12, 48, 1.5, 48, 2.5, 0.3, 0.4),
    ("HYB_96_3_8_30", 96, 0.03, 8, 0.030, 0.7, 0.8, 12, 48, 1.0, 48, 2.5, 0.25, 0.35),
    ("HYB_144_2_12_25", 144, 0.02, 12, 0.025, 0.6, 1.0, 16, 48, 1.5, 48, 2.5, 0.3, 0.4),
    ("HYB_64_2_8_25", 64, 0.02, 8, 0.025, 0.6, 1.0, 12, 48, 1.5, 48, 2.0, 0.3, 0.4),
    ("HYB_96_2_4_20", 96, 0.02, 4, 0.020, 0.5, 1.0, 8, 48, 1.5, 48, 2.5, 0.3, 0.4),
    ("HYB_noVol_96_2", 96, 0.02, 8, 0.025, 0.6, 1.0, 12, 48, 0.0, 48, 2.5, 0.3, 0.4),
    # Aggressive: wider target, tighter stop
    ("HYB_agg_96_2", 96, 0.02, 8, 0.025, 0.8, 0.7, 16, 48, 1.5, 48, 2.5, 0.25, 0.3),
    # Trend-only (no volume filter, relaxed threshold)
    ("HYB_relax", 96, 0.015, 8, 0.020, 0.6, 1.2, 16, 48, 0.0, 48, 2.5, 0.3, 0.4),
]

hybrid_results = []
t0 = time.time()

for cfg in hybrid_configs:
    name = cfg[0]
    params = cfg[1:]

    def sig_func(sym, p=params):
        return compute_hybrid_signals(
            close_arrays[sym], high_arrays[sym], low_arrays[sym],
            volume_arrays[sym], taker_buy_arrays[sym],
            p[0], p[1], p[2], p[3], p[4], p[5], p[6],
            p[7], p[8], p[9], p[10], p[11], p[12],
        )

    result = run_strategy_on_all(sig_func)
    result['name'] = name
    hybrid_results.append(result)

    # Top 10
    top10 = sorted(result['per_symbol'].items(), key=lambda x: x[1]['ret'], reverse=True)[:10]
    top10_syms = [s for s, _ in top10]
    result_top10 = run_strategy_on_all(sig_func, top10_syms)
    result_top10['name'] = name + "_top10"
    hybrid_results.append(result_top10)

print(f"\n{'Name':<22} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} {'PF':>6} {'Trades':>7} "
      f"{'Win':>5} {'+Sym':>5}")
print("-" * 80)
for r in sorted(hybrid_results, key=lambda x: x['port_sharpe'], reverse=True):
    print(f"{r['name']:<22} {r['port_ret']:+8.2f} {r['port_sharpe']:8.2f} "
          f"{r['port_dd']:7.2f} {r['avg_pf']:6.2f} {r['total_trades']:7d} "
          f"{np.mean([v['wr'] for v in r['per_symbol'].values()]):5.1f} "
          f"{r['positive']:>3}/{r['n_sym']}")

print(f"\nTime: {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
#  GRAND RANKING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  GRAND RANKING — ALL STRATEGIES")
print("=" * 80)

all_results = rb_results + mom_results + hybrid_results

# Score = Sharpe * sqrt(Return) * (1 - DD/100) — rewards high Sharpe with decent returns
for r in all_results:
    ret = max(r['port_ret'], 0.01)  # avoid sqrt of negative
    r['score'] = r['port_sharpe'] * np.sqrt(ret) * (1.0 - r['port_dd'] / 100)
    # Calmar
    r['calmar'] = r['port_ret'] / r['port_dd'] if r['port_dd'] > 0 else 0

all_results.sort(key=lambda x: x['score'], reverse=True)

print(f"\n{'Rank':<5} {'Name':<22} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} "
      f"{'Calmar':>7} {'PF':>6} {'Score':>8} {'Trades':>7} {'+Sym':>5}")
print("-" * 95)
for i, r in enumerate(all_results[:30], 1):
    print(f"{i:<5} {r['name']:<22} {r['port_ret']:+8.2f} {r['port_sharpe']:8.2f} "
          f"{r['port_dd']:7.2f} {r['calmar']:7.2f} {r['avg_pf']:6.2f} "
          f"{r['score']:8.3f} {r['total_trades']:7d} "
          f"{r['positive']:>3}/{r['n_sym']}")


# ═══════════════════════════════════════════════════════════════════════
#  DEEP DIVE: Best strategy per-symbol breakdown
# ═══════════════════════════════════════════════════════════════════════
best = all_results[0]
print(f"\n\n{'=' * 80}")
print(f"  BEST STRATEGY: {best['name']}")
print(f"{'=' * 80}")
print(f"\n  Portfolio Return: {best['port_ret']:+.2f}%")
print(f"  Sharpe:           {best['port_sharpe']:.2f}")
print(f"  Max Drawdown:     {best['port_dd']:.2f}%")
print(f"  Calmar:           {best['calmar']:.2f}")
print(f"  Avg PF:           {best['avg_pf']:.2f}")
print(f"  Total Trades:     {best['total_trades']}")

print(f"\n{'Symbol':<12} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} {'WR%':>6} "
      f"{'Trades':>7} {'PF':>6}")
print("-" * 60)
sorted_syms = sorted(best['per_symbol'].items(), key=lambda x: x[1]['ret'], reverse=True)
for sym, v in sorted_syms:
    print(f"{sym:<12} {v['ret']:+8.2f} {v['sharpe']:8.2f} "
          f"{v['dd']:7.2f} {v['wr']:6.1f} {v['trades']:7d} {v['pf']:6.2f}")


# ═══════════════════════════════════════════════════════════════════════
#  MONTHLY BREAKDOWN for top 3
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  MONTHLY BREAKDOWN — Top 3 Strategies")
print(f"{'=' * 80}")

MONTH = 2880  # 30 days in 15m bars

for r in all_results[:3]:
    eq = r['portfolio_eq']
    monthly = []
    for m in range(n_bars // MONTH):
        s, e = m * MONTH, min((m + 1) * MONTH, n_bars)
        m_ret = (eq[e - 1] / eq[s] - 1.0) * 100
        monthly.append(m_ret)

    months_str = " | ".join([f"M{i+1}: {m:+.2f}%" for i, m in enumerate(monthly)])
    print(f"  {r['name']:<22}: {months_str}")


# ═══════════════════════════════════════════════════════════════════════
#  POSITION SIZING PROJECTION
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  POSITION SIZING & COMPOUNDING PROJECTIONS")
print(f"{'=' * 80}")

for r in all_results[:5]:
    base_ret = r['port_ret']
    days = n_bars * 15 / 60 / 24

    # Conservative sizing: 1% risk, 5 slots
    risk_pct = 0.01
    max_slots = 5
    capital_per_slot = 1.0 / max_slots
    # With proper position sizing via Kelly-like approach
    # If PF > 1 and WR known, optimal fraction = (PF * WR - (1-WR)) / PF
    avg_wr = np.mean([v['wr'] for v in r['per_symbol'].values()]) / 100
    avg_pf = r['avg_pf']
    if avg_pf > 0 and avg_pf < 100:
        kelly_f = (avg_pf * avg_wr - (1 - avg_wr)) / avg_pf
    else:
        kelly_f = 0
    kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25% for safety

    # Leveraged return (using half-Kelly for safety)
    half_kelly = kelly_f * 0.5
    est_ret_with_sizing = base_ret * half_kelly * max_slots
    annual_ret = est_ret_with_sizing * (365.25 / days)

    print(f"\n  {r['name']}:")
    print(f"    Base return (equal-weight):  {base_ret:+.2f}%")
    print(f"    Kelly fraction:              {kelly_f*100:.1f}%")
    print(f"    Half-Kelly:                  {half_kelly*100:.1f}%")
    print(f"    Est. return (sized, {days:.0f}d):  {est_ret_with_sizing:+.2f}%")
    print(f"    Annualized:                  {annual_ret:+.1f}%")

print()
