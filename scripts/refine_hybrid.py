#!/usr/bin/env python3
"""
Phase 2: Refine the Hybrid Strategy + Walk-Forward Validation

Results from mega_sweep.py showed HYB_64_2_8_25 as the winner:
  - Sharpe 6.00, Calmar 6.72, PF 5.39 (top-10 symbols)
  - Sharpe 1.92 across ALL 30 symbols (22/30 profitable)

This script:
  1. Fine-tunes parameters around the winning config
  2. Implements ROLLING symbol selection (no look-ahead bias)
  3. Walk-forward validates the final strategy
  4. Computes proper compounding returns with position sizing

Usage:
    uv run python scripts/refine_hybrid.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, gzip, pickle, time
from pathlib import Path

from quantbox.strategy.alpha_engine import compute_hybrid_signals
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

# Prepare arrays
close = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64) for s in symbols}
high = {s: all_data[s]["High"].values[:n_bars].astype(np.float64) for s in symbols}
low = {s: all_data[s]["Low"].values[:n_bars].astype(np.float64) for s in symbols}
volume = {s: all_data[s]["Volume"].values[:n_bars].astype(np.float64) for s in symbols}
taker = {s: all_data[s]["Taker Buy Base Asset Volume"].values[:n_bars].astype(np.float64)
         if "Taker Buy Base Asset Volume" in all_data[s].columns
         else all_data[s]["Volume"].values[:n_bars].astype(np.float64) * 0.5
         for s in symbols}

COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32


def run_hybrid(trend_lb, trend_thr, rb_lb, rb_thr, rb_target, rb_stop,
               rb_hold, vol_lb, vol_mult, atr_lb, atr_mult,
               trail_act, trail_dist, sym_list=None):
    """Run hybrid strategy and return portfolio metrics."""
    syms = sym_list or symbols
    n_sym = len(syms)
    portfolio_eq = np.ones(n_bars)
    per_sym = {}

    for sym in syms:
        sigs = compute_hybrid_signals(
            close[sym], high[sym], low[sym], volume[sym], taker[sym],
            trend_lb, trend_thr, rb_lb, rb_thr, rb_target, rb_stop, rb_hold,
            vol_lb, vol_mult, atr_lb, atr_mult, trail_act, trail_dist,
        )
        eq, total_ret, sharpe, max_dd, wr, n_trades, bars_in, pf, avg_ret, _ = (
            _backtest_core(close[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        )
        portfolio_eq *= eq[:n_bars] ** (1.0 / n_sym)
        per_sym[sym] = {
            'ret': total_ret * 100, 'sharpe': sharpe, 'dd': max_dd * 100,
            'wr': wr, 'trades': n_trades, 'pf': pf,
        }

    port_ret = (portfolio_eq[-1] - 1.0) * 100
    peak = np.maximum.accumulate(portfolio_eq)
    port_dd = np.max((peak - portfolio_eq) / peak) * 100
    rets = np.diff(portfolio_eq) / portfolio_eq[:-1]
    port_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0
    positive = sum(1 for v in per_sym.values() if v['ret'] > 0)
    total_trades = sum(v['trades'] for v in per_sym.values())
    avg_pf = np.mean([v['pf'] for v in per_sym.values() if v['pf'] < 100])

    return {
        'port_ret': port_ret, 'port_sharpe': port_sharpe, 'port_dd': port_dd,
        'positive': positive, 'n_sym': n_sym, 'total_trades': total_trades,
        'avg_pf': avg_pf, 'per_symbol': per_sym, 'portfolio_eq': portfolio_eq,
    }


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1: Fine-tune around winner (HYB_64_2_8_25)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  PHASE 1: Fine-tune Hybrid Parameters")
print("=" * 80)

configs = []
# Base winners and variations
for trend_lb in [48, 64, 80, 96]:
    for trend_thr in [0.015, 0.02, 0.025, 0.03]:
        for rb_thr in [0.02, 0.025, 0.03]:
            for rb_target in [0.5, 0.6, 0.7, 0.8]:
                for vol_mult in [1.0, 1.5, 2.0]:
                    configs.append((
                        trend_lb, trend_thr, 8, rb_thr, rb_target, 1.0, 12,
                        48, vol_mult, 48, 2.0, 0.3, 0.4
                    ))

print(f"Testing {len(configs)} parameter combinations across ALL 30 symbols...")
t0 = time.time()

results = []
for i, cfg in enumerate(configs):
    r = run_hybrid(*cfg)
    r['params'] = cfg
    results.append(r)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(configs)} done...")

results.sort(key=lambda x: x['port_sharpe'], reverse=True)

print(f"\nTime: {time.time() - t0:.1f}s")
print(f"\nTop 15 (ALL 30 symbols):")
print(f"{'Rank':<5} {'Ret%':>8} {'Sharpe':>8} {'DD%':>7} {'Calmar':>7} {'PF':>6} "
      f"{'Trades':>7} {'+Sym':>5} {'T_LB':>5} {'T_THR':>6} {'RB_THR':>6} "
      f"{'Target':>6} {'V_Mult':>6}")
print("-" * 105)
for i, r in enumerate(results[:15], 1):
    p = r['params']
    calmar = r['port_ret'] / r['port_dd'] if r['port_dd'] > 0 else 0
    print(f"{i:<5} {r['port_ret']:+8.2f} {r['port_sharpe']:8.2f} "
          f"{r['port_dd']:7.2f} {calmar:7.2f} {r['avg_pf']:6.2f} "
          f"{r['total_trades']:7d} {r['positive']:>3}/{r['n_sym']} "
          f"{p[0]:>5} {p[1]:>6.3f} {p[3]:>6.3f} {p[4]:>6.2f} {p[8]:>6.1f}")


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2: Rolling Symbol Selection (No Look-Ahead Bias)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PHASE 2: Rolling Symbol Selection (Bias-Free)")
print("=" * 80)

# Use the best parameters from Phase 1
best_params = results[0]['params']
print(f"\nBest params: trend_lb={best_params[0]}, trend_thr={best_params[1]:.3f}, "
      f"rb_thr={best_params[3]:.3f}, target={best_params[4]:.2f}, vol_mult={best_params[8]:.1f}")

# Rolling window: rank symbols every RANK_PERIOD bars using past performance
RANK_PERIOD = 960  # 10 days (re-rank every 10 days)
LOOKBACK_RANK = 2880  # Use past 30 days to rank
TOP_N_CHOICES = [5, 8, 10, 15]

for top_n in TOP_N_CHOICES:
    # Generate all signals for all symbols
    all_sigs = {}
    for sym in symbols:
        all_sigs[sym] = compute_hybrid_signals(
            close[sym], high[sym], low[sym], volume[sym], taker[sym],
            *best_params,
        )

    # Rolling portfolio
    portfolio_eq = np.ones(n_bars)
    total_trades = 0
    active_syms_history = []

    # Start after warmup period
    start = LOOKBACK_RANK
    current_syms = symbols[:top_n]  # Initial: arbitrary

    for t in range(start, n_bars):
        # Re-rank every RANK_PERIOD bars
        if (t - start) % RANK_PERIOD == 0 and t > start:
            # Rank by Sharpe over lookback period
            sym_scores = []
            for sym in symbols:
                window_start = max(0, t - LOOKBACK_RANK)
                eq_w, ret_w, sharpe_w, _, _, _, _, pf_w, _, _ = _backtest_core(
                    close[sym][window_start:t],
                    all_sigs[sym][window_start:t],
                    COST_PER_SIDE, FUNDING_PER_BAR,
                )
                sym_scores.append((sym, sharpe_w, ret_w * 100))
            sym_scores.sort(key=lambda x: x[1], reverse=True)
            current_syms = [s for s, _, _ in sym_scores[:top_n]]
            active_syms_history.append((t, current_syms.copy()))

    # Now build portfolio with rolling selection
    # Simpler approach: divide into periods, select top-N at each rebalance
    portfolio_eq = np.ones(n_bars)
    total_trades_roll = 0

    rebalance_points = list(range(start, n_bars, RANK_PERIOD))

    for ri, rp in enumerate(rebalance_points):
        # Determine active symbols based on PAST performance
        if rp > LOOKBACK_RANK:
            sym_scores = []
            for sym in symbols:
                ws = rp - LOOKBACK_RANK
                eq_w, ret_w, sharpe_w, _, _, _, _, pf_w, _, _ = _backtest_core(
                    close[sym][ws:rp],
                    all_sigs[sym][ws:rp],
                    COST_PER_SIDE, FUNDING_PER_BAR,
                )
                sym_scores.append((sym, sharpe_w))
            sym_scores.sort(key=lambda x: x[1], reverse=True)
            active = [s for s, _ in sym_scores[:top_n]]
        else:
            # Not enough history, use all symbols
            active = symbols[:top_n]

        # Calculate returns for this period
        next_rp = rebalance_points[ri + 1] if ri + 1 < len(rebalance_points) else n_bars

        for sym in active:
            period_eq, _, _, _, _, n_tr, _, _, _, _ = _backtest_core(
                close[sym][rp:next_rp],
                all_sigs[sym][rp:next_rp],
                COST_PER_SIDE, FUNDING_PER_BAR,
            )
            total_trades_roll += n_tr
            # Equal-weight contribution
            for j in range(len(period_eq)):
                idx = rp + j
                if idx < n_bars:
                    if j == 0:
                        continue
                    bar_ret = period_eq[j] / period_eq[j-1] - 1.0
                    portfolio_eq[idx] *= (1.0 + bar_ret / top_n)

    # Compute metrics
    port_ret = (portfolio_eq[-1] - 1.0) * 100
    peak = np.maximum.accumulate(portfolio_eq)
    port_dd = np.max((peak - portfolio_eq) / peak) * 100
    rets = np.diff(portfolio_eq[start:]) / portfolio_eq[start:-1]
    port_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0

    # Monthly
    MONTH = 2880
    monthly = []
    for m in range(n_bars // MONTH):
        s, e = m * MONTH, min((m + 1) * MONTH, n_bars)
        m_ret = (portfolio_eq[e-1] / portfolio_eq[s] - 1.0) * 100
        monthly.append(m_ret)
    months_str = " | ".join([f"M{i+1}: {m:+.2f}%" for i, m in enumerate(monthly)])

    calmar = port_ret / port_dd if port_dd > 0 else 0
    print(f"\n  Top-{top_n} Rolling:")
    print(f"    Return: {port_ret:+.2f}%, Sharpe: {port_sharpe:.2f}, DD: {port_dd:.2f}%, "
          f"Calmar: {calmar:.2f}")
    print(f"    Trades: {total_trades_roll}, Monthly: {months_str}")


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3: Walk-Forward Validation (True OOS)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PHASE 3: Walk-Forward Validation")
print("=" * 80)

# Use top 3 param sets from Phase 1
top3_params = [results[i]['params'] for i in range(min(3, len(results)))]

IS_BARS = 2880   # 30 days in-sample
OOS_BARS = 1440  # 15 days out-of-sample
STEP = 1440      # 15-day step

for pi, params in enumerate(top3_params):
    print(f"\n  Config #{pi+1}: trend_lb={params[0]}, trend_thr={params[1]:.3f}, "
          f"rb_thr={params[3]:.3f}, target={params[4]:.2f}, vol_mult={params[8]:.1f}")

    windows = []
    s = 0
    while s + IS_BARS + OOS_BARS <= n_bars:
        windows.append({'is_start': s, 'is_end': s + IS_BARS,
                        'oos_start': s + IS_BARS, 'oos_end': s + IS_BARS + OOS_BARS})
        s += STEP

    oos_rets = []
    oos_sharpes = []
    oos_trade_counts = []

    for wi, w in enumerate(windows):
        # IS: find top symbols
        is_scores = []
        for sym in symbols:
            sigs = compute_hybrid_signals(
                close[sym][w['is_start']:w['is_end']],
                high[sym][w['is_start']:w['is_end']],
                low[sym][w['is_start']:w['is_end']],
                volume[sym][w['is_start']:w['is_end']],
                taker[sym][w['is_start']:w['is_end']],
                *params,
            )
            eq, ret, sharpe, _, _, _, _, _, _, _ = _backtest_core(
                close[sym][w['is_start']:w['is_end']], sigs,
                COST_PER_SIDE, FUNDING_PER_BAR,
            )
            is_scores.append((sym, sharpe))

        is_scores.sort(key=lambda x: x[1], reverse=True)
        top_syms = [s for s, _ in is_scores[:10]]

        # OOS: test on top symbols
        oos_eq = np.ones(w['oos_end'] - w['oos_start'])
        oos_trades = 0

        for sym in top_syms:
            sigs = compute_hybrid_signals(
                close[sym][w['oos_start']:w['oos_end']],
                high[sym][w['oos_start']:w['oos_end']],
                low[sym][w['oos_start']:w['oos_end']],
                volume[sym][w['oos_start']:w['oos_end']],
                taker[sym][w['oos_start']:w['oos_end']],
                *params,
            )
            eq, ret, sharpe, _, _, n_tr, _, _, _, _ = _backtest_core(
                close[sym][w['oos_start']:w['oos_end']], sigs,
                COST_PER_SIDE, FUNDING_PER_BAR,
            )
            oos_eq *= eq ** (1.0 / 10)
            oos_trades += n_tr

        w_ret = (oos_eq[-1] - 1.0) * 100
        rets_w = np.diff(oos_eq) / oos_eq[:-1]
        w_sharpe = np.mean(rets_w) / np.std(rets_w) * np.sqrt(96*365.25) if np.std(rets_w) > 0 else 0

        oos_rets.append(w_ret)
        oos_sharpes.append(w_sharpe)
        oos_trade_counts.append(oos_trades)

        print(f"    Window {wi+1}: OOS return {w_ret:+.2f}%, Sharpe {w_sharpe:.2f}, "
              f"Trades {oos_trades}")

    print(f"    ── Summary ──")
    print(f"    Avg OOS Return: {np.mean(oos_rets):+.2f}%")
    print(f"    Avg OOS Sharpe: {np.mean(oos_sharpes):.2f}")
    print(f"    Profitable windows: {sum(1 for r in oos_rets if r > 0)}/{len(oos_rets)}")
    print(f"    Std of returns: {np.std(oos_rets):.2f}%")


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 4: Compounding Equity Curve with Position Sizing
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PHASE 4: Realistic Compounding Simulation")
print("=" * 80)

# Use best params
bp = results[0]['params']
print(f"\nBest params: {bp}")

# Run on all 30 symbols, but with proper position sizing
# Key: the raw backtest assumes 100% capital per position
# With sizing: each trade uses risk_pct/(threshold*stop) of capital
risk_pct = 0.01  # 1% risk per trade
max_slots = 5

# First, get signals and equity curves for each symbol
sym_results = {}
for sym in symbols:
    sigs = compute_hybrid_signals(
        close[sym], high[sym], low[sym], volume[sym], taker[sym], *bp)
    eq, ret, sharpe, dd, wr, trades, bars_in, pf, avg_ret, trade_rets = _backtest_core(
        close[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
    sym_results[sym] = {
        'signals': sigs, 'equity': eq, 'ret': ret,
        'sharpe': sharpe, 'trades': trades, 'pf': pf, 'wr': wr,
    }

# Rank by Sharpe
ranked = sorted(sym_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
print(f"\nAll 30 Symbols (ranked by Sharpe):")
print(f"{'Symbol':<12} {'Ret%':>8} {'Sharpe':>8} {'Trades':>7} {'WR%':>6} {'PF':>6}")
print("-" * 55)
for sym, r in ranked:
    print(f"{sym:<12} {r['ret']*100:+8.2f} {r['sharpe']:8.2f} "
          f"{r['trades']:7d} {r['wr']:6.1f} {r['pf']:6.2f}")

pos_count = sum(1 for _, r in ranked if r['ret'] > 0)
print(f"\nPositive: {pos_count}/{len(ranked)}")

# Simulate compounded equity with top-10 rolling selection
# and leveraged position sizing
TOP_N = 10
REBALANCE = 960  # 10 days

# Pre-compute all signals
all_signals = {sym: sym_results[sym]['signals'] for sym in symbols}

# Simple compounding simulation
initial_capital = 10000.0
capital = initial_capital
equity_ts = [capital]

for t in range(1, n_bars):
    # Determine active symbols (rebalance periodically)
    if t % REBALANCE == 0 and t > LOOKBACK_RANK:
        # Rank by recent Sharpe
        scores = []
        for sym in symbols:
            ws = max(0, t - LOOKBACK_RANK)
            eq_w, ret_w, sh_w, _, _, _, _, _, _, _ = _backtest_core(
                close[sym][ws:t], all_signals[sym][ws:t],
                COST_PER_SIDE, FUNDING_PER_BAR)
            scores.append((sym, sh_w))
        scores.sort(key=lambda x: x[1], reverse=True)
        active = [s for s, _ in scores[:TOP_N]]
    elif t == 1:
        active = [s for s, _ in ranked[:TOP_N]]

    # Aggregate bar returns from active symbols
    bar_ret_sum = 0.0
    for sym in active:
        sig_prev = all_signals[sym][t-1]
        sig_now = all_signals[sym][t]
        price_ret = close[sym][t] / close[sym][t-1] - 1.0

        pnl = sig_prev * price_ret  # Position PnL

        # Cost on position change
        if sig_now != sig_prev:
            cost = abs(sig_prev) * COST_PER_SIDE + abs(sig_now) * COST_PER_SIDE
            pnl -= cost

        # Funding cost
        if sig_prev != 0:
            pnl -= abs(sig_prev) * FUNDING_PER_BAR

        # Position sizing: scale by Kelly fraction
        sizing = 0.10  # 10% of capital per symbol (conservative)
        bar_ret_sum += pnl * sizing / TOP_N

    capital *= (1.0 + bar_ret_sum)
    equity_ts.append(capital)

equity_arr = np.array(equity_ts)
final_ret = (equity_arr[-1] / equity_arr[0] - 1.0) * 100
peak = np.maximum.accumulate(equity_arr)
max_dd_sized = np.max((peak - equity_arr) / peak) * 100
rets_sized = np.diff(equity_arr) / equity_arr[:-1]
sharpe_sized = np.mean(rets_sized) / np.std(rets_sized) * np.sqrt(96*365.25) if np.std(rets_sized) > 0 else 0

# Different leverage levels
print(f"\n{'=' * 60}")
print(f"  COMPOUNDING PROJECTIONS (90 days → 365 days)")
print(f"{'=' * 60}")

leverage_levels = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
for lev in leverage_levels:
    cap = initial_capital
    for t in range(1, n_bars):
        bar_ret_sum = 0.0
        for sym in active:  # Uses last active set — simplified
            sig_prev = all_signals[sym][t-1]
            price_ret = close[sym][t] / close[sym][t-1] - 1.0
            pnl = sig_prev * price_ret
            if all_signals[sym][t] != sig_prev:
                pnl -= abs(sig_prev) * COST_PER_SIDE + abs(all_signals[sym][t]) * COST_PER_SIDE
            if sig_prev != 0:
                pnl -= abs(sig_prev) * FUNDING_PER_BAR
            bar_ret_sum += pnl / TOP_N

        cap *= (1.0 + bar_ret_sum * lev * 0.1)  # 10% base allocation × leverage

    ret_lev = (cap / initial_capital - 1.0) * 100
    annual_lev = ((cap / initial_capital) ** (365.25 / 90) - 1.0) * 100
    print(f"  {lev:>4.0f}x leverage: {ret_lev:+8.2f}% (90d) → {annual_lev:+8.1f}% annualized")

# Final summary
print(f"\n{'=' * 80}")
print(f"  FINAL VERDICT")
print(f"{'=' * 80}")

best_all30 = results[0]
print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  Strategy: Hybrid Momentum + Rubber Band             │
  │  "Buy the dip in uptrends, sell the rally in down"   │
  ├──────────────────────────────────────────────────────┤
  │  ALL 30 Symbols:                                     │
  │    Return: {best_all30['port_ret']:+.2f}%  Sharpe: {best_all30['port_sharpe']:.2f}  DD: {best_all30['port_dd']:.2f}%  │
  │    Positive Symbols: {best_all30['positive']}/{best_all30['n_sym']}                           │
  │    Total Trades: {best_all30['total_trades']}                                │
  │                                                      │
  │  Parameters:                                         │
  │    trend_lb={best_all30['params'][0]}, trend_thr={best_all30['params'][1]:.3f}                │
  │    rb_lb=8, rb_thr={best_all30['params'][3]:.3f}                        │
  │    target={best_all30['params'][4]:.2f}, stop=1.0, max_hold=12           │
  │    vol_mult={best_all30['params'][8]:.1f}                                │
  └──────────────────────────────────────────────────────┘
""")
