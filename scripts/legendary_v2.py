#!/usr/bin/env python3
"""
Legendary Scalper v2 — Correct Portfolio Construction

Key fix: Use pre-computed equity curves from _backtest_core,
then combine via rolling symbol selection using PAST Sharpe ranking.

No bar-by-bar loops (fast). No look-ahead bias (rolling).

Usage:
    uv run python scripts/legendary_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, gzip, pickle, time
from pathlib import Path

from quantbox.strategy.alpha_engine import compute_alpha_signals, compute_hybrid_signals
from quantbox.strategy.rubberband import compute_signals
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
days = n_bars * 15 / 60 / 24
print(f"Data: {len(symbols)} symbols × {n_bars} bars (15m), {days:.0f} days\n")

# Prepare arrays
close = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64) for s in symbols}
high = {s: all_data[s]["High"].values[:n_bars].astype(np.float64) for s in symbols}
low = {s: all_data[s]["Low"].values[:n_bars].astype(np.float64) for s in symbols}
volume = {s: all_data[s]["Volume"].values[:n_bars].astype(np.float64) for s in symbols}
taker = {}
for s in symbols:
    if "Taker Buy Base Asset Volume" in all_data[s].columns:
        taker[s] = all_data[s]["Taker Buy Base Asset Volume"].values[:n_bars].astype(np.float64)
    else:
        taker[s] = volume[s] * 0.5

COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32


def get_metrics_for_window(eq_curve):
    """Compute Sharpe and return from an equity curve segment."""
    ret = eq_curve[-1] / eq_curve[0] - 1.0
    rets = np.diff(eq_curve) / eq_curve[:-1]
    std = np.std(rets)
    sharpe = np.mean(rets) / std * np.sqrt(96 * 365.25) if std > 0 else 0
    return ret, sharpe


# ═══════════════════════════════════════════════════════════════════════
#  Pre-compute all strategy signals and equity curves
# ═══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  Pre-computing strategies...")
print("=" * 80)

strategy_configs = {
    # Strategy Name: signal computation function
    'RB_original': lambda sym: compute_signals(
        close[sym], 8, 0.025, 0.5, 1.0, 8),

    'RB_enhanced': lambda sym: compute_alpha_signals(
        close[sym], high[sym], low[sym], volume[sym], taker[sym],
        8, 0.025, 0.6, 1.0, 12,
        True, 48, 1.5,    # vol filter
        False, 96, 0,     # no momentum
        True, 48, 2.5,    # adaptive ATR
        True, 0.3, 0.4,   # trailing stop
        False, 16, 0.55,  # no taker
        False, 32),        # no breakout

    'RB_vol+trail': lambda sym: compute_alpha_signals(
        close[sym], high[sym], low[sym], volume[sym], taker[sym],
        8, 0.025, 0.6, 1.0, 12,
        True, 48, 2.0,    # stricter vol filter
        False, 96, 0,
        False, 48, 2.5,
        True, 0.25, 0.35, # tighter trailing
        False, 16, 0.55,
        False, 32),

    'RB_aggressive': lambda sym: compute_alpha_signals(
        close[sym], high[sym], low[sym], volume[sym], taker[sym],
        8, 0.025, 0.8, 0.8, 16,  # wider target, tighter stop, longer hold
        True, 48, 1.5,
        False, 96, 0,
        True, 48, 2.5,
        True, 0.3, 0.3,
        False, 16, 0.55,
        False, 32),

    'Hybrid_64': lambda sym: compute_hybrid_signals(
        close[sym], high[sym], low[sym], volume[sym], taker[sym],
        64, 0.02, 8, 0.025, 0.6, 1.0, 12,
        48, 1.5, 48, 2.0, 0.3, 0.4),

    'Hybrid_48': lambda sym: compute_hybrid_signals(
        close[sym], high[sym], low[sym], volume[sym], taker[sym],
        48, 0.015, 8, 0.020, 0.5, 1.0, 12,
        48, 1.5, 48, 2.5, 0.3, 0.4),
}

# Compute signals and equity curves for each strategy × symbol
strat_results = {}  # strat -> sym -> {equity, signals, ret, sharpe, trades, wr, pf}

for strat_name, sig_func in strategy_configs.items():
    strat_results[strat_name] = {}
    for sym in symbols:
        sigs = sig_func(sym)
        eq, ret, sharpe, dd, wr, trades, bars_in, pf, avg_ret, _ = _backtest_core(
            close[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        strat_results[strat_name][sym] = {
            'equity': eq, 'signals': sigs,
            'ret': ret * 100, 'sharpe': sharpe, 'dd': dd * 100,
            'trades': trades, 'wr': wr, 'pf': pf,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Show per-strategy, all-30-symbols results
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  BASELINE: All 30 Symbols (Equal Weight)")
print("=" * 80)

for strat_name in strategy_configs:
    results = strat_results[strat_name]
    eq_all = np.ones(n_bars)
    for sym in symbols:
        eq_all *= results[sym]['equity'] ** (1.0 / len(symbols))
    port_ret = (eq_all[-1] - 1.0) * 100
    peak = np.maximum.accumulate(eq_all)
    dd = np.max((peak - eq_all) / peak) * 100
    rets = np.diff(eq_all) / eq_all[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96*365.25) if np.std(rets) > 0 else 0
    pos = sum(1 for r in results.values() if r['ret'] > 0)
    trades = sum(r['trades'] for r in results.values())
    avg_pf = np.mean([r['pf'] for r in results.values() if r['pf'] < 100])

    MONTH = 2880
    monthly = [(eq_all[min((m+1)*MONTH, n_bars)-1] / eq_all[m*MONTH] - 1.0) * 100
               for m in range(n_bars // MONTH)]
    months_str = " | ".join([f"M{i+1}:{m:+.1f}%" for i, m in enumerate(monthly)])

    print(f"  {strat_name:<18}: Ret {port_ret:+6.2f}%, Sharpe {sharpe:5.2f}, "
          f"DD {dd:5.2f}%, PF {avg_pf:4.2f}, Trades {trades:>5}, "
          f"+{pos}/30 | {months_str}")


# ═══════════════════════════════════════════════════════════════════════
#  ROLLING SYMBOL SELECTION (by PAST Sharpe)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  ROLLING SYMBOL SELECTION (Past-Sharpe Rank, No Bias)")
print("=" * 80)

RANK_LOOKBACK = 2880   # 30 days to compute ranking
REBALANCE = 960        # Re-rank every 10 days
VOL_LOOKBACK = 2880    # For volatility ranking

def rolling_portfolio(strat_name, top_n, rank_method='sharpe'):
    """Build portfolio with rolling symbol selection.

    rank_method: 'sharpe' (past Sharpe), 'vol' (realized volatility),
                 'return' (past return), 'composite' (Sharpe × sqrt(trades))
    """
    results = strat_results[strat_name]

    # Collect bar returns for each symbol
    sym_bar_rets = {}
    for sym in symbols:
        eq = results[sym]['equity']
        bar_rets = np.zeros(n_bars)
        for j in range(1, n_bars):
            bar_rets[j] = eq[j] / eq[j - 1] - 1.0
        sym_bar_rets[sym] = bar_rets

    # Build rolling portfolio
    portfolio_eq = np.ones(n_bars)
    selection_log = []

    rebalance_points = list(range(RANK_LOOKBACK, n_bars, REBALANCE))
    if rebalance_points[-1] < n_bars:
        rebalance_points.append(n_bars)  # Final point

    for ri in range(len(rebalance_points) - 1):
        rp = rebalance_points[ri]
        next_rp = rebalance_points[ri + 1]

        # Rank symbols based on PAST data only
        scores = []
        for sym in symbols:
            eq = results[sym]['equity']
            ws = max(0, rp - RANK_LOOKBACK)
            eq_window = eq[ws:rp]

            if rank_method == 'sharpe':
                _, score = get_metrics_for_window(eq_window)
            elif rank_method == 'return':
                score = eq_window[-1] / eq_window[0] - 1.0
            elif rank_method == 'vol':
                # Higher volatility = more opportunity
                log_rets = np.diff(np.log(close[sym][ws:rp]))
                score = np.std(log_rets) * np.sqrt(96 * 365.25)
            elif rank_method == 'composite':
                _, sh = get_metrics_for_window(eq_window)
                sigs = results[sym]['signals'][ws:rp]
                trade_count = np.sum(np.abs(np.diff(sigs)) > 0)
                score = sh * np.sqrt(max(trade_count, 1))
            else:
                score = 0

            scores.append((sym, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scores[:top_n]]
        selection_log.append((rp, selected))

        # Build portfolio for this window
        for j in range(rp, next_rp):
            bar_ret = 0.0
            for sym in selected:
                bar_ret += sym_bar_rets[sym][j] / top_n
            portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    # Fill initial period (before first rebalance) with all-symbol equal weight
    for j in range(1, RANK_LOOKBACK):
        bar_ret = 0.0
        for sym in symbols:
            bar_ret += sym_bar_rets[sym][j] / len(symbols)
        portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    return portfolio_eq, selection_log


# Test all combinations
print(f"\n{'Strategy':<18} {'Rank':<10} {'N':>3} {'Ret%':>8} {'Sharpe':>7} "
      f"{'DD%':>6} {'Calmar':>7} {'Months':>30}")
print("-" * 100)

best_score = -999
best_config = None
all_rolling = []

for strat_name in strategy_configs:
    for rank_method in ['sharpe', 'return', 'vol', 'composite']:
        for top_n in [5, 8, 10, 15]:
            eq, sel_log = rolling_portfolio(strat_name, top_n, rank_method)

            port_ret = (eq[-1] - 1.0) * 100
            peak = np.maximum.accumulate(eq)
            dd = np.max((peak - eq) / peak) * 100
            rets = np.diff(eq[RANK_LOOKBACK:]) / eq[RANK_LOOKBACK:-1]
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96*365.25) if np.std(rets) > 0 else 0
            calmar = port_ret / dd if dd > 0 else 0

            MONTH = 2880
            monthly = [(eq[min((m+1)*MONTH, n_bars)-1] / eq[m*MONTH] - 1.0) * 100
                       for m in range(n_bars // MONTH)]
            months_str = " ".join([f"M{i+1}:{m:+.1f}%" for i, m in enumerate(monthly)])

            score = sharpe * max(port_ret, 0.01) ** 0.5 * (1 - dd / 100)
            all_rolling.append({
                'strat': strat_name, 'rank': rank_method, 'n': top_n,
                'ret': port_ret, 'sharpe': sharpe, 'dd': dd, 'calmar': calmar,
                'monthly': monthly, 'score': score, 'eq': eq, 'sel_log': sel_log,
            })

            if score > best_score:
                best_score = score
                best_config = all_rolling[-1]

# Sort and display top results
all_rolling.sort(key=lambda x: x['score'], reverse=True)

print("\n  TOP 20 ROLLING CONFIGURATIONS:")
print(f"\n{'Rank':<5} {'Strategy':<18} {'Method':<10} {'N':>3} {'Ret%':>8} "
      f"{'Sharpe':>7} {'DD%':>6} {'Calmar':>7} {'Score':>7}")
print("-" * 80)
for i, r in enumerate(all_rolling[:20], 1):
    months_str = " ".join([f"M{j+1}:{m:+.1f}%" for j, m in enumerate(r['monthly'])])
    print(f"{i:<5} {r['strat']:<18} {r['rank']:<10} {r['n']:>3} {r['ret']:+8.2f} "
          f"{r['sharpe']:7.2f} {r['dd']:6.2f} {r['calmar']:7.2f} {r['score']:7.3f}")


# ═══════════════════════════════════════════════════════════════════════
#  Deep dive: Best rolling config
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  BEST ROLLING CONFIG: {best_config['strat']} / {best_config['rank']} / Top-{best_config['n']}")
print("=" * 80)

print(f"\n  Return:    {best_config['ret']:+.2f}%")
print(f"  Sharpe:    {best_config['sharpe']:.2f}")
print(f"  DD:        {best_config['dd']:.2f}%")
print(f"  Calmar:    {best_config['calmar']:.2f}")

# Monthly
for i, m in enumerate(best_config['monthly']):
    print(f"  Month {i+1}:   {m:+.2f}%")

# Symbol selections over time
print(f"\n  Symbol Selection Log:")
for rp, selected in best_config['sel_log']:
    day = rp * 15 / 60 / 24
    print(f"    Day {day:>5.0f}: {', '.join(selected)}")

# Per-symbol breakdown
results = strat_results[best_config['strat']]
print(f"\n  Per-symbol results ({best_config['strat']}):")
print(f"  {'Symbol':<12} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} {'Trades':>7} {'WR%':>6} {'PF':>6}")
print("  " + "-" * 58)
sorted_syms = sorted(results.items(), key=lambda x: x[1]['ret'], reverse=True)
for sym, r in sorted_syms:
    print(f"  {sym:<12} {r['ret']:+8.2f} {r['sharpe']:8.2f} {r['dd']:7.2f} "
          f"{r['trades']:7d} {r['wr']:6.1f} {r['pf']:6.2f}")


# ═══════════════════════════════════════════════════════════════════════
#  Leveraged Compounding Projections
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  LEVERAGED COMPOUNDING PROJECTIONS")
print("=" * 80)

# Take top 5 rolling configs
for ci, cfg in enumerate(all_rolling[:5], 1):
    print(f"\n  #{ci} {cfg['strat']} / {cfg['rank']} / Top-{cfg['n']}")
    base_eq = cfg['eq']

    for lev in [1, 2, 3, 5, 8, 10, 15, 20]:
        lev_eq = np.ones(n_bars)
        for j in range(1, n_bars):
            bar_ret = base_eq[j] / base_eq[j-1] - 1.0
            lev_eq[j] = lev_eq[j-1] * (1.0 + bar_ret * lev)
            # Circuit breaker: avoid total loss
            if lev_eq[j] < 0.1:
                lev_eq[j:] = lev_eq[j]
                break

        lev_ret = (lev_eq[-1] / lev_eq[0] - 1.0) * 100
        lev_peak = np.maximum.accumulate(lev_eq)
        lev_dd = np.max((lev_peak - lev_eq) / lev_peak) * 100
        annual = ((lev_eq[-1] / lev_eq[0]) ** (365.25 / days) - 1.0) * 100 if lev_eq[-1] > 0.01 else -100

        print(f"    {lev:>2}x: {lev_ret:+10.2f}% (90d), DD {lev_dd:6.2f}%, "
              f"Annual {annual:+10.1f}%")


# ═══════════════════════════════════════════════════════════════════════
#  WALK-FORWARD (proper OOS for best config)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  WALK-FORWARD VALIDATION (Top 3 Rolling Configs)")
print("=" * 80)

IS_BARS = 2880   # 30 days IS
OOS_BARS = 1440  # 15 days OOS
STEP = 1440      # 15-day step

windows = []
s = 0
while s + IS_BARS + OOS_BARS <= n_bars:
    windows.append({'is_start': s, 'is_end': s + IS_BARS,
                    'oos_start': s + IS_BARS, 'oos_end': s + IS_BARS + OOS_BARS})
    s += STEP

for ci, cfg in enumerate(all_rolling[:3], 1):
    strat_name = cfg['strat']
    rank_method = cfg['rank']
    top_n = cfg['n']
    results = strat_results[strat_name]

    print(f"\n  #{ci} {strat_name} / {rank_method} / Top-{top_n}")

    oos_rets = []
    for wi, w in enumerate(windows):
        # Rank symbols on IS data
        scores = []
        for sym in symbols:
            eq = results[sym]['equity']
            eq_is = eq[w['is_start']:w['is_end']]

            if rank_method == 'sharpe':
                _, score = get_metrics_for_window(eq_is)
            elif rank_method == 'return':
                score = eq_is[-1] / eq_is[0] - 1.0
            elif rank_method == 'vol':
                log_rets = np.diff(np.log(close[sym][w['is_start']:w['is_end']]))
                score = np.std(log_rets) * np.sqrt(96 * 365.25)
            elif rank_method == 'composite':
                _, sh = get_metrics_for_window(eq_is)
                sigs = results[sym]['signals'][w['is_start']:w['is_end']]
                trade_count = np.sum(np.abs(np.diff(sigs)) > 0)
                score = sh * np.sqrt(max(trade_count, 1))
            else:
                score = 0
            scores.append((sym, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scores[:top_n]]

        # OOS: build portfolio of selected symbols
        oos_eq = np.ones(w['oos_end'] - w['oos_start'])
        oos_trades = 0
        for sym in selected:
            eq_full = results[sym]['equity']
            eq_oos = eq_full[w['oos_start']:w['oos_end']]
            # Normalize to start at 1
            eq_norm = eq_oos / eq_oos[0]
            oos_eq *= eq_norm ** (1.0 / top_n)
            # Count trades in OOS
            sigs = results[sym]['signals'][w['oos_start']:w['oos_end']]
            oos_trades += np.sum(np.abs(np.diff(sigs)) > 0)

        w_ret = (oos_eq[-1] - 1.0) * 100
        oos_rets.append(w_ret)
        rets_w = np.diff(oos_eq) / oos_eq[:-1]
        w_sharpe = np.mean(rets_w) / np.std(rets_w) * np.sqrt(96*365.25) if np.std(rets_w) > 0 else 0

        day_start = w['oos_start'] * 15 / 60 / 24
        day_end = w['oos_end'] * 15 / 60 / 24
        print(f"    W{wi+1} (d{day_start:.0f}-{day_end:.0f}): OOS {w_ret:+6.2f}%, "
              f"Sharpe {w_sharpe:5.2f}, Trades {oos_trades:>3}")
        print(f"       Selected: {', '.join(selected[:5])}{'...' if len(selected) > 5 else ''}")

    print(f"    ─── Summary ───")
    print(f"    Avg OOS Return: {np.mean(oos_rets):+.2f}% per 15d")
    print(f"    Profitable:     {sum(1 for r in oos_rets if r > 0)}/{len(oos_rets)}")
    print(f"    Min / Max:      {min(oos_rets):+.2f}% / {max(oos_rets):+.2f}%")

    # Annualize
    compound = 1.0
    for r in oos_rets:
        compound *= (1.0 + r / 100)
    total_oos_days = len(oos_rets) * 15
    annual_oos = (compound ** (365.25 / total_oos_days) - 1.0) * 100
    print(f"    Compounded OOS: {(compound - 1.0) * 100:+.2f}% over {total_oos_days}d")
    print(f"    Annualized:     {annual_oos:+.1f}%")


# ═══════════════════════════════════════════════════════════════════════
#  GRAND SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  GRAND SUMMARY — THE LEGENDARY SCALPER")
print("=" * 80)

best = all_rolling[0]
print(f"""
  Best Configuration:
    Strategy:     {best['strat']}
    Ranking:      {best['rank']} (re-rank every 10 days using past 30 days)
    Universe:     Top {best['n']} symbols

  Full Period Performance (90 days):
    Return:       {best['ret']:+.2f}%
    Sharpe:       {best['sharpe']:.2f}
    Max DD:       {best['dd']:.2f}%
    Calmar:       {best['calmar']:.2f}

  Monthly breakdown:
    Month 1:      {best['monthly'][0]:+.2f}%
    Month 2:      {best['monthly'][1]:+.2f}%
    Month 3:      {best['monthly'][2]:+.2f}%
""")

print("  With Leverage (Kelly-guided):")
base_eq = best['eq']
for lev in [1, 3, 5, 10, 20]:
    lev_eq = np.ones(n_bars)
    for j in range(1, n_bars):
        bar_ret = base_eq[j] / base_eq[j-1] - 1.0
        lev_eq[j] = lev_eq[j-1] * (1.0 + bar_ret * lev)
        if lev_eq[j] < 0.1:
            lev_eq[j:] = lev_eq[j]
            break
    lev_ret = (lev_eq[-1] - 1.0) * 100
    annual = ((lev_eq[-1]) ** (365.25 / days) - 1.0) * 100 if lev_eq[-1] > 0 else -100
    lev_peak = np.maximum.accumulate(lev_eq)
    lev_dd = np.max((lev_peak - lev_eq) / lev_peak) * 100
    print(f"    {lev:>2}x: {lev_ret:+10.2f}% (90d) | DD {lev_dd:5.2f}% | Annual {annual:+10.1f}%")

print()
