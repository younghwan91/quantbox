#!/usr/bin/env python3
"""
Regime diagnostic — understand WHY the strategy fails on 365 days.

Investigates:
  1. Raw rubber band (no MTF, no stop_decay) — is the BASE alpha dying?
  2. Varying threshold levels — maybe 2.5% is wrong for different regimes
  3. Realized volatility vs strategy PnL correlation
  4. BTC trend overlay (is mean-reversion dying in trending markets?)
  5. Aggressive sweep across the full parameter space on 365d data
"""

import sys
import os
import gzip
import pickle
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quantbox.strategy.rubberband import compute_signals, compute_signals_mtf
from quantbox.strategy.vol_selector import rank_symbols_by_vol
from quantbox.backtest.vectorized_backtest import _backtest_core

TOP_N = 3
VOL_LOOKBACK = 2880
REBALANCE = 960
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96


def load_extended_data():
    cache_dir = Path("data/cache_extended")
    all_data = {}
    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" in df.columns and len(df) >= 200:
            all_data[symbol] = df
    return all_data


def run_portfolio_simple(close_dict, symbols, n_bars, sig_func):
    """Run portfolio backtest with a custom signal function."""
    equities = {}
    all_trade_rets = {}
    for sym in symbols:
        close = close_dict[sym]
        sigs = sig_func(close)
        eq, ret, sharpe, dd, wr, n_trades, bars_in, pf, avg_ret, trade_rets = (
            _backtest_core(close, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        )
        equities[sym] = eq
        all_trade_rets[sym] = trade_rets

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

    # Collect trade rets from selected syms
    all_tr = []
    for s in selected_syms:
        for r in all_trade_rets[s]:
            all_tr.append(r)
    all_tr = np.array(all_tr) if all_tr else np.array([0.0])

    return portfolio_eq, all_tr


def compute_metrics(eq, start_idx=0):
    sub = eq[start_idx:]
    if len(sub) < 2:
        return {'ret': 0, 'dd': 0, 'calmar': 0, 'sharpe': 0}
    ret = (sub[-1] / sub[0] - 1.0) * 100
    peak = np.maximum.accumulate(sub)
    dd = np.max((peak - sub) / peak) * 100
    calmar = ret / dd if dd > 0 else 999.0
    rets = np.diff(sub) / sub[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0.0
    return {'ret': ret, 'dd': dd, 'calmar': calmar, 'sharpe': sharpe}


def trade_stats_brief(tr):
    if len(tr) == 0:
        return 0, 0, 0
    wins = tr[tr > 0]
    wr = len(wins) / len(tr) * 100
    pf = np.sum(wins) / np.sum(np.abs(tr[tr <= 0])) if np.sum(np.abs(tr[tr <= 0])) > 0 else 999
    return len(tr), wr, pf


def main():
    print("=" * 70)
    print("  REGIME DIAGNOSTIC — Why does the strategy fail on 365 days?")
    print("=" * 70)

    all_data = load_extended_data()
    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    close_dict = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64)
                  for s in symbols}
    days = n_bars / BARS_PER_DAY
    print(f"Loaded {len(symbols)} symbols, {n_bars:,} bars ({days:.0f} days)")

    # ═══════════════════════════════════════════════════════════
    # 1. BASE RUBBER BAND (no MTF, varying stop_decay)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  1. BASE RUBBER BAND — Is the raw alpha working?")
    print(f"{'█' * 70}")

    configs = [
        ("RB raw (no decay)",      8, 0.025, 0.5, 1.0, 8, 0.0),
        ("RB decay=0.3",           8, 0.025, 0.5, 1.0, 8, 0.3),
        ("RB decay=0.5",           8, 0.025, 0.5, 1.0, 8, 0.5),
        ("RB target=0.3",          8, 0.025, 0.3, 1.0, 8, 0.0),
        ("RB target=0.3 d=0.3",    8, 0.025, 0.3, 1.0, 8, 0.3),
        ("RB threshold=0.03",      8, 0.030, 0.5, 1.0, 8, 0.0),
        ("RB threshold=0.035",     8, 0.035, 0.5, 1.0, 8, 0.0),
        ("RB threshold=0.04",      8, 0.040, 0.5, 1.0, 8, 0.0),
        ("RB threshold=0.05",      8, 0.050, 0.5, 1.0, 8, 0.0),
        ("RB lb=12",              12, 0.025, 0.5, 1.0, 8, 0.0),
        ("RB lb=12 th=0.035",     12, 0.035, 0.5, 1.0, 8, 0.0),
        ("RB lb=4 th=0.02",        4, 0.020, 0.5, 1.0, 6, 0.0),
        ("RB lb=4 th=0.015",       4, 0.015, 0.5, 1.0, 6, 0.0),
        ("RB stop=0.5",            8, 0.025, 0.5, 0.5, 8, 0.0),
        ("RB stop=0.7",            8, 0.025, 0.5, 0.7, 8, 0.0),
        ("RB max_hold=4",          8, 0.025, 0.5, 1.0, 4, 0.0),
        ("RB max_hold=12",         8, 0.025, 0.5, 1.0, 12, 0.0),
        ("RB max_hold=16",         8, 0.025, 0.5, 1.0, 16, 0.0),
    ]

    print(f"\n  {'Config':>25} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR%':>6} {'PF':>5}")
    print("  " + "-" * 85)

    for label, lb, th, tgt, stop, mh, sd in configs:
        def sig_func(close, _lb=lb, _th=th, _tgt=tgt, _stop=stop, _mh=mh, _sd=sd):
            return compute_signals(close, _lb, _th, _tgt, _stop, _mh, _sd)

        eq, tr = run_portfolio_simple(close_dict, symbols, n_bars, sig_func)
        m = compute_metrics(eq, VOL_LOOKBACK)
        n_tr, wr, pf = trade_stats_brief(tr)
        print(f"  {label:>25} {m['ret']:>+8.2f} {m['dd']:>7.2f} "
              f"{m['calmar']:>7.2f} {m['sharpe']:>7.2f} {n_tr:>7} {wr:>5.1f}% {pf:>5.2f}")

    # ═══════════════════════════════════════════════════════════
    # 2. BTC TREND ANALYSIS — When does mean reversion work?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  2. BTC TREND vs STRATEGY PnL — Regime Analysis")
    print(f"{'█' * 70}")

    btc_close = close_dict['BTCUSDT']

    # Compute BTC monthly returns
    bars_per_month = BARS_PER_DAY * 30
    n_months = (n_bars - VOL_LOOKBACK) // bars_per_month

    # Use vanilla RB (best so far) for regime analysis
    def sig_vanilla(close):
        return compute_signals(close, 8, 0.025, 0.5, 1.0, 8, 0.0)

    eq_vanilla, _ = run_portfolio_simple(close_dict, symbols, n_bars, sig_vanilla)

    print(f"\n  {'Month':>8} {'BTC%':>8} {'Strat%':>8} {'BTC_vol':>8} {'Regime':>12}")
    print("  " + "-" * 55)

    for m in range(n_months):
        m_start = VOL_LOOKBACK + m * bars_per_month
        m_end = min(m_start + bars_per_month, n_bars)
        btc_ret = (btc_close[m_end - 1] / btc_close[m_start] - 1.0) * 100
        strat_ret = (eq_vanilla[m_end - 1] / eq_vanilla[m_start] - 1.0) * 100

        # Compute BTC realized volatility for this month
        btc_rets = np.diff(btc_close[m_start:m_end]) / btc_close[m_start:m_end - 1]
        btc_vol = np.std(btc_rets) * np.sqrt(96 * 365.25) * 100

        # Detect regime
        if abs(btc_ret) > 15:
            regime = "TRENDING"
        elif btc_vol > 60:
            regime = "HIGH_VOL"
        else:
            regime = "NORMAL"

        print(f"  M{m+1:>2}     {btc_ret:>+7.1f}% {strat_ret:>+7.1f}% {btc_vol:>7.1f}% {regime:>12}")

    # ═══════════════════════════════════════════════════════════
    # 3. EQUAL WEIGHT vs VOL SELECT — Is Vol Selection hurting?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  3. EQUAL WEIGHT vs VOL SELECTION")
    print(f"{'█' * 70}")

    # Equal weight all symbols
    def sig_vanilla_ew(close):
        return compute_signals(close, 8, 0.025, 0.5, 1.0, 8, 0.0)

    equities_ew = {}
    for sym in symbols:
        sigs = sig_vanilla_ew(close_dict[sym])
        eq_s, _, _, _, _, _, _, _, _, _ = _backtest_core(
            close_dict[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        equities_ew[sym] = eq_s

    # Equal weight portfolio
    ew_eq = np.ones(n_bars)
    for j in range(1, n_bars):
        bar_ret = 0
        for sym in symbols:
            bar_ret += equities_ew[sym][j] / equities_ew[sym][j-1] - 1.0
        bar_ret /= len(symbols)
        ew_eq[j] = ew_eq[j-1] * (1.0 + bar_ret)

    ew_m = compute_metrics(ew_eq, VOL_LOOKBACK)
    vol_m = compute_metrics(eq_vanilla, VOL_LOOKBACK)

    print(f"\n  {'Method':>15} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 50)
    print(f"  {'Equal Weight':>15} {ew_m['ret']:>+8.2f} {ew_m['dd']:>7.2f} "
          f"{ew_m['calmar']:>7.2f} {ew_m['sharpe']:>7.2f}")
    print(f"  {'Vol Top-3':>15} {vol_m['ret']:>+8.2f} {vol_m['dd']:>7.2f} "
          f"{vol_m['calmar']:>7.2f} {vol_m['sharpe']:>7.2f}")

    # Top-5
    def run_topn(n):
        eq_n, _ = run_portfolio_topn(close_dict, symbols, n_bars, sig_vanilla_ew, n)
        return compute_metrics(eq_n, VOL_LOOKBACK)

    for n in [5, 10, 15]:
        m_n = run_topn(n)
        print(f"  {'Vol Top-' + str(n):>15} {m_n['ret']:>+8.2f} {m_n['dd']:>7.2f} "
              f"{m_n['calmar']:>7.2f} {m_n['sharpe']:>7.2f}")

    # ═══════════════════════════════════════════════════════════
    # 4. PER-SYMBOL ANALYSIS — Which coins are profitable?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  4. PER-SYMBOL ANNUAL PERFORMANCE")
    print(f"{'█' * 70}")

    sym_perfs = []
    for sym in symbols:
        sigs = sig_vanilla_ew(close_dict[sym])
        eq_s, _, sharpe, dd, wr, n_trades, _, pf, _, _ = _backtest_core(
            close_dict[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        ret = (eq_s[-1] / eq_s[0] - 1.0) * 100  # Full period
        sym_perfs.append((sym, ret, dd * 100, sharpe, wr, n_trades, pf))

    sym_perfs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Symbol':>12} {'Ret%':>8} {'DD%':>7} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'PF':>5}")
    print("  " + "-" * 60)
    for sym, ret, dd, sharpe, wr, trades, pf in sym_perfs:
        marker = "★" if ret > 0 else ""
        print(f"  {sym:>12} {ret:>+8.2f} {dd:>7.2f} {sharpe:>7.2f} {wr:>5.1f}% {trades:>7} {pf:>5.2f} {marker}")

    pos_syms = sum(1 for _, ret, *_ in sym_perfs if ret > 0)
    print(f"\n  Profitable: {pos_syms}/{len(symbols)} ({pos_syms/len(symbols)*100:.0f}%)")
    print(f"  Best: {sym_perfs[0][0]} ({sym_perfs[0][1]:+.2f}%)")
    print(f"  Worst: {sym_perfs[-1][0]} ({sym_perfs[-1][1]:+.2f}%)")

    # ═══════════════════════════════════════════════════════════
    # 5. AGGRESSIVE THRESHOLD SWEEP
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  5. FULL PARAMETER SWEEP on 365d (finding if ANY config works)")
    print(f"{'█' * 70}")

    best_calmar = -999
    best_config = None
    results = []

    # Sweep key params
    for lb in [4, 6, 8, 12]:
        for th in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
            for tgt in [0.3, 0.4, 0.5, 0.6]:
                for mh in [4, 6, 8, 12]:
                    for sd in [0.0, 0.3, 0.5]:
                        def sig_func(close, _lb=lb, _th=th, _tgt=tgt, _mh=mh, _sd=sd):
                            return compute_signals(close, _lb, _th, _tgt, 1.0, _mh, _sd)
                        eq, tr = run_portfolio_simple(close_dict, symbols, n_bars, sig_func)
                        m = compute_metrics(eq, VOL_LOOKBACK)
                        n_tr, wr, pf = trade_stats_brief(tr)
                        results.append((lb, th, tgt, mh, sd, m, n_tr, wr, pf))
                        if m['calmar'] > best_calmar:
                            best_calmar = m['calmar']
                            best_config = (lb, th, tgt, mh, sd, m, n_tr, wr, pf)

    # Sort by Calmar
    results.sort(key=lambda x: x[5]['calmar'], reverse=True)

    print(f"\n  Top 20 configs (by Calmar):")
    print(f"  {'LB':>3} {'TH':>6} {'TGT':>5} {'MH':>3} {'SD':>4} "
          f"{'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7} {'Trades':>7} {'WR%':>6} {'PF':>5}")
    print("  " + "-" * 80)
    for lb, th, tgt, mh, sd, m, n_tr, wr, pf in results[:20]:
        print(f"  {lb:>3} {th:>6.3f} {tgt:>5.2f} {mh:>3} {sd:>4.1f} "
              f"{m['ret']:>+8.2f} {m['dd']:>7.2f} {m['calmar']:>7.2f} {m['sharpe']:>7.2f} "
              f"{n_tr:>7} {wr:>5.1f}% {pf:>5.2f}")

    print(f"\n  Total configs tested: {len(results)}")
    pos_configs = sum(1 for r in results if r[5]['ret'] > 0)
    print(f"  Profitable configs: {pos_configs}/{len(results)} ({pos_configs/len(results)*100:.1f}%)")

    if best_config:
        lb, th, tgt, mh, sd, m, n_tr, wr, pf = best_config
        print(f"\n  BEST CONFIG: lb={lb}, th={th}, tgt={tgt}, mh={mh}, sd={sd}")
        print(f"    Ret: {m['ret']:+.2f}%, DD: {m['dd']:.2f}%, Calmar: {m['calmar']:.2f}, "
              f"Sharpe: {m['sharpe']:.2f}")
        print(f"    Trades: {n_tr}, WR: {wr:.1f}%, PF: {pf:.2f}")

    print(f"\n  VERDICT:")
    if best_calmar > 1.0:
        print(f"  ✓ Found a viable config with Calmar > 1.0 on 365 days")
    elif best_calmar > 0:
        print(f"  ⚠ Best config is marginally profitable but weak (Calmar < 1.0)")
    else:
        print(f"  ✗ NO CONFIG IS PROFITABLE on 365 days. Alpha is dead.")


def run_portfolio_topn(close_dict, symbols, n_bars, sig_func, top_n):
    """Run portfolio with variable top_n."""
    equities = {}
    for sym in symbols:
        sigs = sig_func(close_dict[sym])
        eq_s, _, _, _, _, _, _, _, _, _ = _backtest_core(
            close_dict[sym], sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        equities[sym] = eq_s

    sym_bar_rets = {}
    for sym in symbols:
        eq = equities[sym]
        bar_rets = np.zeros(n_bars)
        for j in range(1, n_bars):
            bar_rets[j] = eq[j] / eq[j - 1] - 1.0
        sym_bar_rets[sym] = bar_rets

    portfolio_eq = np.ones(n_bars)
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
            top_n=top_n, lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96 * 365.25),
        )
        selected = [sym for sym, _ in ranked]
        for j in range(rp, next_rp):
            bar_ret = sum(sym_bar_rets[s][j] for s in selected) / top_n
            portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        bar_ret = sum(sym_bar_rets[s][j] for s in symbols) / len(symbols)
        portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    all_tr = np.array([0.0])  # placeholder
    return portfolio_eq, all_tr


if __name__ == "__main__":
    main()
