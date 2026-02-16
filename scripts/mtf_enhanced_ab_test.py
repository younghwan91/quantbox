#!/usr/bin/env python3
"""
MTF Enhanced A/B Test — Block-only vs 3-mode (agree+boost entries+tighter stop)

Compares three configurations on the 30-symbol portfolio:
  A) Baseline:      No MTF filter
  B) MTF v1:        Block-only (htf_agree_discount=1.0, htf_agree_stop_mult=1.0)
  C) MTF Enhanced:  3-mode (agree=lower threshold+tighter stop, neutral=normal, against=block)

Also sweeps htf_agree_discount and htf_agree_stop_mult to find optimal values.

Per-trade statistics: trade count, win rate, avg win/loss, risk-reward ratio,
expected trades per day.

Usage:
    uv run python scripts/mtf_enhanced_ab_test.py
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
VOL_LOOKBACK = 2880     # 30 days
REBALANCE = 960          # 10 days
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32

# Base strategy params
LOOKBACK = 8
THRESHOLD = 0.025
TARGET = 0.5
STOP = 1.0
MAX_HOLD = 8
STOP_DECAY = 0.3

# MTF params
HTF_PERIOD = 48
HTF_BLOCK_THRESHOLD = 0.02


def load_data():
    """Load all cached symbols."""
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


def precompute_equities(close_dict, signal_func):
    """Pre-compute equity curves and trade returns for all symbols.

    Returns:
        equities: dict[str, np.ndarray] — equity curve per symbol
        results: dict[str, dict] — per-symbol metrics
        all_trade_returns: dict[str, np.ndarray] — per-symbol trade returns
    """
    equities = {}
    results = {}
    all_trade_returns = {}

    for sym, close in close_dict.items():
        sigs = signal_func(close)
        eq, ret, sharpe, dd, wr, n_trades, bars_in, pf, avg_ret, trade_rets = (
            _backtest_core(close, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        )
        equities[sym] = eq
        results[sym] = {
            'ret': ret * 100, 'sharpe': sharpe, 'dd': dd * 100,
            'trades': n_trades, 'wr': wr, 'pf': pf, 'avg_ret': avg_ret * 100,
        }
        all_trade_returns[sym] = trade_rets

    return equities, results, all_trade_returns


def build_rolling_portfolio(equities, close_dict, symbols, n_bars):
    """Build rolling vol-selected portfolio from pre-computed equities.

    Returns portfolio equity curve and list of selected symbols per window.
    """
    sym_bar_rets = {}
    for sym in symbols:
        eq = equities[sym]
        bar_rets = np.zeros(n_bars)
        for j in range(1, n_bars):
            bar_rets[j] = eq[j] / eq[j - 1] - 1.0
        sym_bar_rets[sym] = bar_rets

    portfolio_eq = np.ones(n_bars)
    selection_log = []

    rebalance_pts = list(range(VOL_LOOKBACK, n_bars, REBALANCE))
    if len(rebalance_pts) == 0:
        rebalance_pts = [VOL_LOOKBACK]
    if rebalance_pts[-1] < n_bars:
        rebalance_pts.append(n_bars)

    for ri in range(len(rebalance_pts) - 1):
        rp = rebalance_pts[ri]
        next_rp = rebalance_pts[ri + 1]

        ws = max(0, rp - VOL_LOOKBACK)
        sym_closes_window = {s: close_dict[s][ws:rp] for s in symbols}
        ranked = rank_symbols_by_vol(
            sym_closes_window, top_n=TOP_N,
            lookback_bars=VOL_LOOKBACK,
            annualize_factor=np.sqrt(96 * 365.25),
        )
        selected = [sym for sym, _ in ranked]
        selection_log.append((rp, selected))

        for j in range(rp, next_rp):
            bar_ret = 0.0
            for sym in selected:
                bar_ret += sym_bar_rets[sym][j] / TOP_N
            portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    # Fill warmup with all-symbol equal weight
    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        bar_ret = 0.0
        for sym in symbols:
            bar_ret += sym_bar_rets[sym][j] / len(symbols)
        portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    return portfolio_eq, selection_log


def compute_portfolio_stats(portfolio_eq, results, selection_log, n_bars, label,
                            all_trade_returns=None):
    """Compute portfolio metrics + per-trade stats from selected symbols only."""
    port_ret = (portfolio_eq[-1] - 1.0) * 100
    peak = np.maximum.accumulate(portfolio_eq)
    port_dd = np.max((peak - portfolio_eq) / peak) * 100

    # Use only post-warmup for Sharpe
    warmup_end = VOL_LOOKBACK
    rets = np.diff(portfolio_eq[warmup_end:]) / portfolio_eq[warmup_end:-1]
    port_sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25)
                   if np.std(rets) > 0 else 0.0)
    calmar = port_ret / port_dd if port_dd > 0 else 999.0

    # Aggregate trade stats across selected symbols only
    selected_syms = set()
    for _, sel in selection_log:
        for s in sel:
            selected_syms.add(s)

    total_trades = sum(results[s]['trades'] for s in selected_syms)
    days = n_bars * 15 / 60 / 24
    trades_per_day = total_trades / days if days > 0 else 0

    # Collect all trade returns from selected symbols
    trade_ret_list = []
    if all_trade_returns is not None:
        for s in selected_syms:
            for r in all_trade_returns[s]:
                trade_ret_list.append(r)

    if trade_ret_list:
        tr_arr = np.array(trade_ret_list)
        wins = tr_arr[tr_arr > 0]
        losses = tr_arr[tr_arr <= 0]
        win_rate = len(wins) / len(tr_arr) * 100
        avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0.0
        avg_loss = np.mean(np.abs(losses)) * 100 if len(losses) > 0 else 0.0
        rr = avg_win / avg_loss if avg_loss > 0 else 999.0
        pf = np.sum(wins) / np.sum(np.abs(losses)) if np.sum(np.abs(losses)) > 0 else 999.0
    else:
        win_rate = avg_win = avg_loss = rr = pf = 0.0

    # Post-warmup DD (real DD)
    post_warmup_eq = portfolio_eq[warmup_end:]
    post_peak = np.maximum.accumulate(post_warmup_eq)
    real_dd = np.max((post_peak - post_warmup_eq) / post_peak) * 100
    real_ret = (post_warmup_eq[-1] / post_warmup_eq[0] - 1.0) * 100
    real_calmar = real_ret / real_dd if real_dd > 0 else 999.0

    return {
        "label": label,
        "return_pct": port_ret,
        "sharpe": port_sharpe,
        "max_dd_pct": port_dd,
        "calmar": calmar,
        "real_return_pct": real_ret,
        "real_dd_pct": real_dd,
        "real_calmar": real_calmar,
        "n_trades": total_trades,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "rr_ratio": rr,
        "profit_factor": pf,
        "selected_symbols": sorted(selected_syms),
    }


def print_result(r):
    """Pretty-print a result dict."""
    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"{'='*60}")
    print(f"  Full Period (incl. warmup):")
    print(f"    Return:         {r['return_pct']:+.2f}%")
    print(f"    Max DD:         {r['max_dd_pct']:.2f}%")
    print(f"    Calmar:         {r['calmar']:.2f}")
    print(f"  Post-Warmup (real strategy):")
    print(f"    Return:         {r['real_return_pct']:+.2f}%")
    print(f"    Max DD:         {r['real_dd_pct']:.2f}%")
    print(f"    Calmar:         {r['real_calmar']:.2f}")
    print(f"    Sharpe:         {r['sharpe']:.2f}")
    print(f"  ─── Trade Stats ───")
    print(f"  Total Trades:     {r['n_trades']}")
    print(f"  Trades/Day:       {r['trades_per_day']:.1f}")
    print(f"  Win Rate:         {r['win_rate']:.1f}%")
    print(f"  Avg Win:          {r['avg_win_pct']:.3f}%")
    print(f"  Avg Loss:         {r['avg_loss_pct']:.3f}%")
    print(f"  Risk-Reward:      {r['rr_ratio']:.2f}")
    print(f"  Profit Factor:    {r['profit_factor']:.2f}")
    if r.get('selected_symbols'):
        syms_str = ', '.join(r['selected_symbols'][:8])
        if len(r['selected_symbols']) > 8:
            syms_str += f"... ({len(r['selected_symbols'])} total)"
        print(f"  Selected Syms:    {syms_str}")


def main():
    print("Loading data...")
    all_data = load_data()
    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    days = n_bars * 15 / 60 / 24
    print(f"Loaded {len(symbols)} symbols, {n_bars} bars ({days:.0f} days)")

    close_dict = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64)
                  for s in symbols}

    # ── Signal functions ──
    def sig_baseline(close):
        return compute_signals(close, LOOKBACK, THRESHOLD, TARGET, STOP,
                               MAX_HOLD, STOP_DECAY)

    def sig_mtf_v1(close):
        return compute_signals_mtf(close, LOOKBACK, THRESHOLD, TARGET, STOP,
                                   MAX_HOLD, STOP_DECAY,
                                   HTF_PERIOD, HTF_BLOCK_THRESHOLD,
                                   1.0, 1.0)

    def sig_mtf_enhanced(close):
        return compute_signals_mtf(close, LOOKBACK, THRESHOLD, TARGET, STOP,
                                   MAX_HOLD, STOP_DECAY,
                                   HTF_PERIOD, HTF_BLOCK_THRESHOLD,
                                   0.8, 0.7)

    print("\n" + "█" * 60)
    print("  A/B TEST: Baseline vs MTF v1 vs MTF Enhanced")
    print("█" * 60)

    ab_results = []
    for label, sig_func in [
        ("A) Baseline (no MTF)", sig_baseline),
        ("B) MTF v1 (block-only)", sig_mtf_v1),
        ("C) MTF Enhanced (disc=0.8, stop=0.7)", sig_mtf_enhanced),
    ]:
        equities, results, trade_rets = precompute_equities(close_dict, sig_func)
        port_eq, sel_log = build_rolling_portfolio(equities, close_dict, symbols, n_bars)
        stats = compute_portfolio_stats(port_eq, results, sel_log, n_bars, label,
                                        trade_rets)
        ab_results.append(stats)
        print_result(stats)

    # ── Parameter Sweep ──
    print("\n\n" + "█" * 60)
    print("  SWEEP: htf_agree_discount × htf_agree_stop_mult")
    print("█" * 60)

    discounts = [0.6, 0.7, 0.8, 0.9, 1.0]
    stop_mults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    sweep_results = []
    for disc, sm in product(discounts, stop_mults):
        def make_sig_func(d, s):
            def f(close):
                return compute_signals_mtf(close, LOOKBACK, THRESHOLD, TARGET, STOP,
                                          MAX_HOLD, STOP_DECAY,
                                          HTF_PERIOD, HTF_BLOCK_THRESHOLD, d, s)
            return f

        sig_func = make_sig_func(disc, sm)
        label = f"disc={disc:.1f}, stop={sm:.1f}"
        equities, results, trade_rets = precompute_equities(close_dict, sig_func)
        port_eq, sel_log = build_rolling_portfolio(equities, close_dict, symbols, n_bars)
        stats = compute_portfolio_stats(port_eq, results, sel_log, n_bars, label,
                                        trade_rets)
        sweep_results.append(stats)

    # Sort by real calmar (post-warmup)
    sweep_results.sort(key=lambda x: x["real_calmar"], reverse=True)

    print(f"\n{'Disc':>6} {'Stop':>6} {'RealRet%':>10} {'RealDD%':>9} "
          f"{'RCalmar':>8} {'Sharpe':>7} {'Trades':>7} "
          f"{'WR%':>6} {'RR':>6} {'PF':>6}")
    print("-" * 90)
    for r in sweep_results:
        parts = r["label"].split(", ")
        disc = parts[0].split("=")[1]
        sm = parts[1].split("=")[1]
        print(f"{disc:>6} {sm:>6} {r['real_return_pct']:>+10.2f} "
              f"{r['real_dd_pct']:>9.2f} {r['real_calmar']:>8.2f} "
              f"{r['sharpe']:>7.2f} {r['n_trades']:>7} "
              f"{r['win_rate']:>5.1f}% {r['rr_ratio']:>6.2f} "
              f"{r['profit_factor']:>6.2f}")

    best = sweep_results[0]

    # ── Best config detailed ──
    print_result(best)

    # ── Final comparison table ──
    print(f"\n\n{'='*60}")
    print(f"  FINAL COMPARISON (post-warmup metrics)")
    print(f"{'='*60}")
    print(f"{'Config':>40} {'RealRet':>8} {'RealDD':>7} {'RCalm':>6} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR':>6} {'RR':>6} {'T/Day':>6}")
    print("-" * 105)
    for r in ab_results:
        print(f"{r['label']:>40} {r['real_return_pct']:>+7.1f}% "
              f"{r['real_dd_pct']:>6.1f}% {r['real_calmar']:>6.2f} "
              f"{r['sharpe']:>7.2f} {r['n_trades']:>7} "
              f"{r['win_rate']:>5.1f}% {r['rr_ratio']:>6.2f} "
              f"{r['trades_per_day']:>5.1f}")
    print(f"{'BEST: '+best['label']:>40} {best['real_return_pct']:>+7.1f}% "
          f"{best['real_dd_pct']:>6.1f}% {best['real_calmar']:>6.2f} "
          f"{best['sharpe']:>7.2f} {best['n_trades']:>7} "
          f"{best['win_rate']:>5.1f}% {best['rr_ratio']:>6.2f} "
          f"{best['trades_per_day']:>5.1f}")

    # ── Expected stats for production ──
    print(f"\n\n{'='*60}")
    print(f"  PRODUCTION EXPECTATIONS (best config)")
    print(f"{'='*60}")
    post_days = (n_bars - VOL_LOOKBACK) * 15 / 60 / 24
    trades_post = best['n_trades']
    tpd = trades_post / post_days if post_days > 0 else 0
    tpw = tpd * 7
    print(f"  Expected Trades/Day:    {tpd:.1f}")
    print(f"  Expected Trades/Week:   {tpw:.1f}")
    print(f"  Win Rate:               {best['win_rate']:.1f}%")
    print(f"  Risk-Reward Ratio:      {best['rr_ratio']:.2f}")
    print(f"  Profit Factor:          {best['profit_factor']:.2f}")
    print(f"  Avg Win:                {best['avg_win_pct']:.3f}%")
    print(f"  Avg Loss:               {best['avg_loss_pct']:.3f}%")
    print(f"  Expected Return/Week:   {best['real_return_pct'] / (post_days/7):+.2f}%")


if __name__ == "__main__":
    main()
