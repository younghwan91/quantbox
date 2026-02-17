#!/usr/bin/env python3
"""
Long-term strategy validation — 365 days across multiple market regimes.

Tests:
  1. Full-period performance (12 months)
  2. Rolling 90-day windows (walk-forward style)
  3. Regime analysis (drawdown months vs rally months)
  4. Parameter stability (how sensitive are results to param changes)
  5. Monthly breakdown
  6. Realistic live projections with haircuts

Usage:
    uv run python scripts/longterm_backtest.py
"""

import sys
import os
import gzip
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quantbox.strategy.rubberband import compute_signals_mtf
from quantbox.strategy.vol_selector import rank_symbols_by_vol
from quantbox.backtest.vectorized_backtest import _backtest_core

# ── Production config (v3.3) ──
LOOKBACK = 8
THRESHOLD = 0.025
TARGET = 0.5
STOP = 1.0
MAX_HOLD = 8
STOP_DECAY = 0.3
HTF_PERIOD = 64
HTF_BLOCK = 0.01
HTF_DISC = 0.85
HTF_STOP = 0.95

TOP_N = 3
VOL_LOOKBACK = 2880  # 30 days
REBALANCE = 960  # 10 days
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96


def load_extended_data():
    """Load 365-day extended data."""
    cache_dir = Path("data/cache_extended")
    all_data = {}
    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" in df.columns and len(df) >= 200:
            all_data[symbol] = df
    return all_data


def run_portfolio(close_dict, symbols, n_bars, htf_period=HTF_PERIOD,
                  htf_block=HTF_BLOCK, disc=HTF_DISC, smult=HTF_STOP):
    """Run portfolio backtest and return equity curve + trade stats."""

    def sig_func(close):
        return compute_signals_mtf(
            close, LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY,
            htf_period, htf_block, disc, smult,
        )

    equities = {}
    all_trade_rets = {}
    sym_results = {}
    for sym in symbols:
        close = close_dict[sym]
        sigs = sig_func(close)
        eq, ret, sharpe, dd, wr, n_trades, bars_in, pf, avg_ret, trade_rets = (
            _backtest_core(close, sigs, COST_PER_SIDE, FUNDING_PER_BAR)
        )
        equities[sym] = eq
        sym_results[sym] = {
            'ret': ret * 100, 'dd': dd * 100, 'sharpe': sharpe,
            'WR': wr, 'trades': n_trades
        }
        all_trade_rets[sym] = trade_rets

    # Bar returns per symbol
    sym_bar_rets = {}
    for sym in symbols:
        eq = equities[sym]
        bar_rets = np.zeros(n_bars)
        for j in range(1, n_bars):
            bar_rets[j] = eq[j] / eq[j - 1] - 1.0
        sym_bar_rets[sym] = bar_rets

    # Portfolio with vol-selection
    portfolio_eq = np.ones(n_bars)
    selected_history = []

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
        selected_history.append((rp, selected))
        for j in range(rp, next_rp):
            bar_ret = sum(sym_bar_rets[s][j] for s in selected) / TOP_N
            portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    # Pre-warmup: equal weight
    for j in range(1, min(VOL_LOOKBACK, n_bars)):
        bar_ret = sum(sym_bar_rets[s][j] for s in symbols) / len(symbols)
        portfolio_eq[j] = portfolio_eq[j - 1] * (1.0 + bar_ret)

    # Collect all trade returns from selected symbols
    selected_syms = set()
    for _, sels in selected_history:
        for s in sels:
            selected_syms.add(s)

    all_tr = []
    for s in selected_syms:
        for r in all_trade_rets[s]:
            all_tr.append(r)
    all_tr = np.array(all_tr) if all_tr else np.array([0.0])

    return portfolio_eq, all_tr, selected_history, sym_results


def compute_metrics(eq, start_idx=0):
    """Compute performance metrics from equity curve."""
    sub = eq[start_idx:]
    ret = (sub[-1] / sub[0] - 1.0) * 100
    peak = np.maximum.accumulate(sub)
    dd = np.max((peak - sub) / peak) * 100
    calmar = ret / dd if dd > 0 else 999.0

    rets = np.diff(sub) / sub[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(96 * 365.25) if np.std(rets) > 0 else 0.0

    return {'ret': ret, 'dd': dd, 'calmar': calmar, 'sharpe': sharpe}


def trade_stats(trade_rets):
    """Compute trade-level statistics."""
    if len(trade_rets) == 0:
        return {}
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets <= 0]
    wr = len(wins) / len(trade_rets) * 100
    avg_w = np.mean(wins) * 100 if len(wins) > 0 else 0
    avg_l = np.mean(np.abs(losses)) * 100 if len(losses) > 0 else 0
    rr = avg_w / avg_l if avg_l > 0 else 999
    pf = np.sum(wins) / np.sum(np.abs(losses)) if np.sum(np.abs(losses)) > 0 else 999
    return {
        'total': len(trade_rets), 'wr': wr, 'avg_win': avg_w,
        'avg_loss': avg_l, 'rr': rr, 'pf': pf,
        'best': np.max(trade_rets) * 100, 'worst': np.min(trade_rets) * 100,
        'median': np.median(trade_rets) * 100,
    }


def main():
    print("=" * 70)
    print("  LONG-TERM STRATEGY VALIDATION — 365 days")
    print("=" * 70)

    print("\nLoading extended data...")
    all_data = load_extended_data()
    if not all_data:
        print("ERROR: No extended data found. Run download_extended_data.py first.")
        return

    symbols = sorted(all_data.keys())
    n_bars = min(len(all_data[s]) for s in symbols)
    close_dict = {s: all_data[s]["Close"].values[:n_bars].astype(np.float64)
                  for s in symbols}
    days = n_bars / BARS_PER_DAY
    months = days / 30.44

    # Get date range
    some_df = list(all_data.values())[0]
    date_start = some_df.index[0]
    date_end = some_df.index[n_bars - 1]

    print(f"Loaded {len(symbols)} symbols, {n_bars:,} bars ({days:.0f} days / {months:.1f} months)")
    print(f"Date range: {date_start} → {date_end}")
    print(f"Config: lookback={LOOKBACK}, threshold={THRESHOLD}, target={TARGET}")
    print(f"  stop={STOP}, max_hold={MAX_HOLD}, stop_decay={STOP_DECAY}")
    print(f"  htf_period={HTF_PERIOD}, block={HTF_BLOCK}, disc={HTF_DISC}, stop_mult={HTF_STOP}")

    # ═══════════════════════════════════════════════════════════
    # 1. FULL-PERIOD BACKTEST
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  1. FULL-PERIOD PERFORMANCE ({days:.0f} days)")
    print(f"{'█' * 70}")

    eq, trade_rets, sel_hist, sym_res = run_portfolio(
        close_dict, symbols, n_bars
    )

    post = compute_metrics(eq, VOL_LOOKBACK)
    ts = trade_stats(trade_rets)
    trades_per_day = ts['total'] / (days - VOL_LOOKBACK / BARS_PER_DAY) if ts else 0

    print(f"\n  Return:       {post['ret']:+.2f}%")
    print(f"  Max DD:       {post['dd']:.2f}%")
    print(f"  Calmar:       {post['calmar']:.2f}")
    print(f"  Sharpe:       {post['sharpe']:.2f}")
    print(f"  Trades:       {ts.get('total', 0):,}")
    print(f"  Trades/Day:   {trades_per_day:.1f}")
    print(f"  Win Rate:     {ts.get('wr', 0):.1f}%")
    print(f"  Avg Win:      {ts.get('avg_win', 0):.3f}%")
    print(f"  Avg Loss:     {ts.get('avg_loss', 0):.3f}%")
    print(f"  Risk/Reward:  {ts.get('rr', 0):.2f}")
    print(f"  Profit Factor:{ts.get('pf', 0):.2f}")
    print(f"  Best Trade:   {ts.get('best', 0):+.3f}%")
    print(f"  Worst Trade:  {ts.get('worst', 0):+.3f}%")
    print(f"  Median Trade: {ts.get('median', 0):+.3f}%")

    # ═══════════════════════════════════════════════════════════
    # 2. MONTHLY BREAKDOWN
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  2. MONTHLY BREAKDOWN")
    print(f"{'█' * 70}")

    bars_per_month = BARS_PER_DAY * 30
    n_months = (n_bars - VOL_LOOKBACK) // bars_per_month

    print(f"\n  {'Month':>8} {'Return%':>9} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Eq_start':>10} {'Eq_end':>10}")
    print("  " + "-" * 68)

    monthly_returns = []
    for m in range(n_months):
        m_start = VOL_LOOKBACK + m * bars_per_month
        m_end = min(m_start + bars_per_month, n_bars)
        m_eq = eq[m_start:m_end]
        m_ret = (m_eq[-1] / m_eq[0] - 1.0) * 100
        m_peak = np.maximum.accumulate(m_eq)
        m_dd = np.max((m_peak - m_eq) / m_peak) * 100
        m_calmar = m_ret / m_dd if m_dd > 0 else 999.0

        m_rets = np.diff(m_eq) / m_eq[:-1]
        m_sharpe = np.mean(m_rets) / np.std(m_rets) * np.sqrt(96 * 365.25) if np.std(m_rets) > 0 else 0.0

        monthly_returns.append(m_ret)
        print(f"  M{m+1:>2}       {m_ret:>+8.2f}% {m_dd:>6.2f}% {m_calmar:>7.2f} "
              f"{m_sharpe:>7.2f} {m_eq[0]:>10.4f} {m_eq[-1]:>10.4f}")

    # Remaining bars
    remaining_start = VOL_LOOKBACK + n_months * bars_per_month
    if remaining_start < n_bars - 1:
        r_eq = eq[remaining_start:]
        r_ret = (r_eq[-1] / r_eq[0] - 1.0) * 100
        monthly_returns.append(r_ret)
        r_days = (n_bars - remaining_start) / BARS_PER_DAY
        print(f"  Rem ({r_days:.0f}d) {r_ret:>+8.2f}%")

    pos_months = sum(1 for r in monthly_returns if r > 0)
    neg_months = sum(1 for r in monthly_returns if r <= 0)
    print(f"\n  Positive months: {pos_months}/{len(monthly_returns)} ({pos_months/len(monthly_returns)*100:.0f}%)")
    print(f"  Best month:  {max(monthly_returns):+.2f}%")
    print(f"  Worst month: {min(monthly_returns):+.2f}%")
    print(f"  Avg month:   {np.mean(monthly_returns):+.2f}%")
    print(f"  Std month:   {np.std(monthly_returns):.2f}%")

    # ═══════════════════════════════════════════════════════════
    # 3. ROLLING 90-DAY WINDOWS (Walk-Forward)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  3. ROLLING 90-DAY WINDOWS (Walk-Forward Stability)")
    print(f"{'█' * 70}")

    window_bars = BARS_PER_DAY * 90
    step_bars = BARS_PER_DAY * 30  # 30-day step
    windows = []

    w_start = VOL_LOOKBACK
    while w_start + window_bars <= n_bars:
        w_eq = eq[w_start:w_start + window_bars]
        w_m = compute_metrics(w_eq, 0)
        windows.append((w_start, w_m))
        w_start += step_bars

    print(f"\n  {'Window':>8} {'Start_bar':>10} {'Ret%':>8} {'DD%':>7} "
          f"{'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 55)
    for i, (ws, wm) in enumerate(windows):
        print(f"  W{i+1:>2}       {ws:>10} {wm['ret']:>+8.2f} {wm['dd']:>7.2f} "
              f"{wm['calmar']:>7.2f} {wm['sharpe']:>7.2f}")

    pos_windows = sum(1 for _, wm in windows if wm['ret'] > 0)
    calmars = [wm['calmar'] for _, wm in windows]
    print(f"\n  Profitable windows: {pos_windows}/{len(windows)} ({pos_windows/len(windows)*100:.0f}%)")
    print(f"  Min Calmar: {min(calmars):.2f}")
    print(f"  Max Calmar: {max(calmars):.2f}")
    print(f"  Avg Calmar: {np.mean(calmars):.2f}")
    print(f"  Std Calmar: {np.std(calmars):.2f}")

    worst_window = min(windows, key=lambda x: x[1]['ret'])
    print(f"  Worst window: Return {worst_window[1]['ret']:+.2f}%, DD {worst_window[1]['dd']:.2f}%")

    # ═══════════════════════════════════════════════════════════
    # 4. PARAMETER SENSITIVITY (Stability Check)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  4. PARAMETER SENSITIVITY (How fragile are we?)")
    print(f"{'█' * 70}")

    # Test neighboring parameters — if results fall off a cliff, we're overfit
    param_tests = [
        ("Production",   HTF_PERIOD, HTF_BLOCK, HTF_DISC, HTF_STOP),
        ("disc=0.80",    HTF_PERIOD, HTF_BLOCK, 0.80,     HTF_STOP),
        ("disc=0.90",    HTF_PERIOD, HTF_BLOCK, 0.90,     HTF_STOP),
        ("stop=0.90",    HTF_PERIOD, HTF_BLOCK, HTF_DISC, 0.90),
        ("stop=1.00",    HTF_PERIOD, HTF_BLOCK, HTF_DISC, 1.00),
        ("block=0.005",  HTF_PERIOD, 0.005,     HTF_DISC, HTF_STOP),
        ("block=0.02",   HTF_PERIOD, 0.02,      HTF_DISC, HTF_STOP),
        ("period=48",    48,         HTF_BLOCK, HTF_DISC, HTF_STOP),
        ("period=96",    96,         HTF_BLOCK, HTF_DISC, HTF_STOP),
        ("No MTF",       0,          HTF_BLOCK, 1.0,      1.0),
    ]

    print(f"\n  {'Config':>14} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Trades':>7}")
    print("  " + "-" * 60)

    sensitivity_results = []
    for label, hp, hb, hd, hs in param_tests:
        eq_t, tr_t, _, _ = run_portfolio(close_dict, symbols, n_bars, hp, hb, hd, hs)
        m = compute_metrics(eq_t, VOL_LOOKBACK)
        ts_t = trade_stats(tr_t)
        sensitivity_results.append((label, m, ts_t))
        print(f"  {label:>14} {m['ret']:>+8.2f} {m['dd']:>7.2f} "
              f"{m['calmar']:>7.2f} {m['sharpe']:>7.2f} {ts_t.get('total', 0):>7}")

    # Check stability
    prod = sensitivity_results[0]
    neighbors = [r for r in sensitivity_results[1:-1]]  # exclude No MTF
    neighbor_calmars = [r[1]['calmar'] for r in neighbors]
    neighbor_rets = [r[1]['ret'] for r in neighbors]

    calmar_stability = min(neighbor_calmars) / prod[1]['calmar'] * 100 if prod[1]['calmar'] > 0 else 0
    ret_stability = min(neighbor_rets) / prod[1]['ret'] * 100 if prod[1]['ret'] > 0 else 0

    print(f"\n  Stability analysis:")
    print(f"    Neighbor min Calmar: {min(neighbor_calmars):.2f} ({calmar_stability:.0f}% of production)")
    print(f"    Neighbor min Return: {min(neighbor_rets):+.2f}% ({ret_stability:.0f}% of production)")
    if calmar_stability >= 60:
        print(f"    ✓ STABLE: Neighbors retain ≥60% of production Calmar")
    else:
        print(f"    ⚠ FRAGILE: Calmar drops >40% with small param changes")

    # ═══════════════════════════════════════════════════════════
    # 5. DRAWDOWN ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  5. DRAWDOWN ANALYSIS")
    print(f"{'█' * 70}")

    post_eq = eq[VOL_LOOKBACK:]
    peak = np.maximum.accumulate(post_eq)
    dd_curve = (peak - post_eq) / peak * 100

    # Find top 5 drawdown episodes
    in_dd = False
    dd_episodes = []
    dd_start = 0
    for i in range(len(dd_curve)):
        if dd_curve[i] > 0.5 and not in_dd:  # DD starts >0.5%
            in_dd = True
            dd_start = i
        elif dd_curve[i] < 0.1 and in_dd:  # DD ends
            in_dd = False
            max_dd_in_episode = np.max(dd_curve[dd_start:i])
            duration_bars = i - dd_start
            dd_episodes.append((dd_start, i, max_dd_in_episode, duration_bars))

    if in_dd:  # Still in DD at end
        max_dd_in_episode = np.max(dd_curve[dd_start:])
        dd_episodes.append((dd_start, len(dd_curve)-1, max_dd_in_episode, len(dd_curve)-1-dd_start))

    dd_episodes.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 5 Drawdown Episodes:")
    print(f"  {'#':>3} {'Depth%':>8} {'Duration':>10} {'Recovery':>10}")
    print("  " + "-" * 35)
    for i, (ds, de, depth, dur_bars) in enumerate(dd_episodes[:5]):
        dur_days = dur_bars / BARS_PER_DAY
        print(f"  {i+1:>3} {depth:>7.2f}% {dur_days:>8.1f}d    bar {ds}-{de}")

    # Time underwater
    underwater_bars = np.sum(dd_curve > 0.5)
    underwater_pct = underwater_bars / len(dd_curve) * 100
    print(f"\n  Time underwater (>0.5% DD): {underwater_pct:.1f}% of total time")
    print(f"  Avg DD when in drawdown: {np.mean(dd_curve[dd_curve > 0]):.2f}%")

    # ═══════════════════════════════════════════════════════════
    # 6. REALISTIC LIVE TRADING PROJECTIONS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  6. REALISTIC LIVE TRADING PROJECTIONS")
    print(f"{'█' * 70}")

    # Apply haircuts
    backtest_annual_ret = post['ret'] * (365.25 / (days - VOL_LOOKBACK / BARS_PER_DAY))
    
    # Conservative haircuts for live vs backtest
    slippage_haircut = 0.85  # 15% worse execution
    regime_change_haircut = 0.80  # 20% for market regime changes
    latency_haircut = 0.95  # 5% for exchange latency / missed fills
    black_swan_haircut = 0.90  # 10% for unexpected events

    total_haircut = slippage_haircut * regime_change_haircut * latency_haircut * black_swan_haircut
    
    live_annual_ret = backtest_annual_ret * total_haircut
    live_monthly_ret = (1 + live_annual_ret/100) ** (1/12) - 1
    live_dd = post['dd'] * 1.3  # DD is typically 30% worse in live

    print(f"\n  Backtest Results (annualized):")
    print(f"    Annual Return:  {backtest_annual_ret:+.1f}%")
    print(f"    Max DD:         {post['dd']:.1f}%")
    print(f"    Calmar:         {post['calmar']:.2f}")

    print(f"\n  Haircuts Applied:")
    print(f"    Execution/Slippage:   {(1-slippage_haircut)*100:.0f}%")
    print(f"    Regime Change:        {(1-regime_change_haircut)*100:.0f}%")
    print(f"    Latency/Missed Fills: {(1-latency_haircut)*100:.0f}%")
    print(f"    Black Swan Buffer:    {(1-black_swan_haircut)*100:.0f}%")
    print(f"    Total Haircut:        {(1-total_haircut)*100:.1f}%")

    print(f"\n  REALISTIC LIVE PROJECTIONS:")
    print(f"    Expected Annual Return: {live_annual_ret:+.1f}%")
    print(f"    Expected Monthly Ret:   {live_monthly_ret*100:+.2f}%")
    print(f"    Expected Max DD:        {live_dd:.1f}%")
    print(f"    Expected Calmar:        {live_annual_ret/live_dd:.2f}")

    capitals = [1000, 5000, 10000, 50000]
    print(f"\n  Capital Projections (12 months):")
    print(f"  {'Capital':>10} {'Expected P&L':>14} {'Worst Case':>12} {'Monthly':>10}")
    print("  " + "-" * 50)
    for cap in capitals:
        expected_pnl = cap * live_annual_ret / 100
        worst_loss = cap * live_dd / 100
        monthly_pnl = cap * live_monthly_ret
        print(f"  ${cap:>9,} ${expected_pnl:>12,.0f} -${worst_loss:>10,.0f} ${monthly_pnl:>8,.0f}/mo")

    # ═══════════════════════════════════════════════════════════
    # 7. PARAMETER TUNING RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'█' * 70}")
    print(f"  7. PARAMETER TUNING RECOMMENDATIONS")
    print(f"{'█' * 70}")

    print(f"""
  Based on the analysis:

  TUNING FREQUENCY:
    • Signal params (lookback, threshold, target, stop, max_hold):
      → NEVER tune during live. These are structural properties of the
        mean-reversion alpha. Re-optimize only if the market structure
        fundamentally changes (e.g., crypto regulation, exchange changes).
    
    • MTF params (htf_period, block, discount, stop_mult):
      → RE-VALIDATE every 3-6 months. These adapt to macro volatility
        regimes. Run the parameter sweep on the latest 180 days.
      → DO NOT tune more frequently — that's overfitting to noise.
    
    • Vol selection (top_n, rebalance interval):  
      → These are SELF-ADAPTIVE by design. No tuning needed.
    
    • Risk params (max_concurrent, risk_per_trade, DD breaker):
      → Adjust based on CAPITAL SIZE, not performance.
        Larger capital → lower risk_per_trade, more max_concurrent.

  MONITORING CHECKLIST (Weekly):
    1. Win rate: Flag if drops below 48% for 2+ weeks
    2. Avg trade duration: Flag if >6 bars consistently
    3. Trades/day: Flag if drops below 5/day (market too quiet)
    4. Max DD: Halt & review if exceeds {live_dd:.1f}%
    5. Profit factor: Flag if drops below 0.95 for 2+ weeks

  REGIME DETECTION (Auto-halt triggers):
    • 10%+ session DD → circuit breaker fires automatically
    • 5%+ daily loss → halts new entries for 24h
    • If strategy loses money 5 consecutive days → manual review
    """)

    # ═══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════
    print(f"{'=' * 70}")
    print(f"  FINAL VERDICT")
    print(f"{'=' * 70}")

    # Robustness score
    score = 0
    checks = []

    # Check 1: Positive return over full period
    if post['ret'] > 0:
        score += 1
        checks.append("✓ Full-period return positive")
    else:
        checks.append("✗ Full-period return NEGATIVE")

    # Check 2: >60% of rolling windows profitable
    if pos_windows / len(windows) >= 0.6:
        score += 1
        checks.append(f"✓ {pos_windows}/{len(windows)} windows profitable (≥60%)")
    else:
        checks.append(f"✗ Only {pos_windows}/{len(windows)} windows profitable (<60%)")

    # Check 3: >60% of months profitable
    if pos_months / len(monthly_returns) >= 0.6:
        score += 1
        checks.append(f"✓ {pos_months}/{len(monthly_returns)} months profitable (≥60%)")
    else:
        checks.append(f"✗ Only {pos_months}/{len(monthly_returns)} months profitable (<60%)")

    # Check 4: Parameter stable
    if calmar_stability >= 60:
        score += 1
        checks.append(f"✓ Parameter stable ({calmar_stability:.0f}% calmar retained)")
    else:
        checks.append(f"✗ Parameter fragile ({calmar_stability:.0f}% calmar retained)")

    # Check 5: Max DD manageable
    if post['dd'] < 15:
        score += 1
        checks.append(f"✓ Max DD < 15% ({post['dd']:.1f}%)")
    else:
        checks.append(f"✗ Max DD ≥ 15% ({post['dd']:.1f}%)")

    # Check 6: Calmar > 1
    if post['calmar'] > 1.0:
        score += 1
        checks.append(f"✓ Calmar > 1.0 ({post['calmar']:.2f})")
    else:
        checks.append(f"✗ Calmar < 1.0 ({post['calmar']:.2f})")

    # Check 7: Worst monthly drawdown manageable
    if min(monthly_returns) > -10:
        score += 1
        checks.append(f"✓ No single month > -10% ({min(monthly_returns):+.2f}%)")
    else:
        checks.append(f"✗ Single month > -10% ({min(monthly_returns):+.2f}%)")

    print(f"\n  Robustness Score: {score}/7")
    for c in checks:
        print(f"    {c}")

    if score >= 6:
        print(f"\n  >>> LIVE-READY. Deploy with confidence.")
        print(f"  >>> Expected: {live_annual_ret:+.1f}% annual, {live_dd:.1f}% max DD")
    elif score >= 4:
        print(f"\n  >>> CAUTIOUSLY LIVE-READY. Start with small capital.")
        print(f"  >>> Monitor closely for first 30 days.")
    else:
        print(f"\n  >>> NOT READY. Strategy needs more work before live deployment.")
        print(f"  >>> Review failing checks above.")


if __name__ == "__main__":
    main()
