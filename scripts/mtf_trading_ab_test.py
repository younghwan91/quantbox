#!/usr/bin/env python3
"""
Multi-Timeframe Trading A/B Test — v6.1 Research

Tests whether trading rubber band signals on MULTIPLE timeframes independently
(4h, 15m, and potentially 5m) reduces portfolio MDD through diversification.

Key hypotheses:
  H1: 4h rubber band signals generate alpha independently
  H2: 4h + 15m combined reduces MDD via uncorrelated entries
  H3: 4h trend as MTF confirmation for 15m entries improves Sharpe
  H4: Capital allocation across TFs optimizes risk-adjusted return

Approach:
  Phase 1: Sweep 4h RB parameters (threshold, lookback, target, stop, max_hold)
  Phase 2: Sweep 5m RB parameters (if data available)
  Phase 3: Portfolio combination (15m + 4h equity curves, varying allocation)
  Phase 4: MTF confirmation (4h trend validates 15m entry)

Usage:
    uv run python scripts/mtf_trading_ab_test.py
"""

import sys
import gzip
import pickle
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import numba

from quantbox.strategy.rubberband import (
    compute_signals_confidence,
    compute_signals_regime_switch,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

# ── Configuration ──
CACHE_DIR = Path("data/cache_extended")
COST_PER_SIDE = 0.0004   # 0.04% per side (fee + slippage)
FUNDING_PER_BAR_15M = 0.0001 / 32   # 0.01% per 8h, 15m bar = 1/32
FUNDING_PER_BAR_4H = 0.0001 / 2     # 0.01% per 8h, 4h bar = 1/2
FUNDING_PER_BAR_5M = 0.0001 / 96    # 0.01% per 8h, 5m bar = 1/96

TOP_10 = [
    "FETUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "DOTUSDT",
    "OPUSDT", "SHIBUSDT", "DOGEUSDT", "LTCUSDT", "ATOMUSDT"
]


def load_data(symbol: str, interval: str) -> dict | None:
    """Load cached OHLCV+taker data for a symbol and interval."""
    cache_file = CACHE_DIR / f"{symbol}_{interval}_365d.pkl.gz"
    if not cache_file.exists():
        return None
    with gzip.open(cache_file, 'rb') as f:
        df = pickle.load(f)
    return {
        'close': df['Close'].values.astype(np.float64),
        'volume': df['Volume'].values.astype(np.float64),
        'taker_buy': df['Taker Buy Base'].values.astype(np.float64),
        'high': df['High'].values.astype(np.float64),
        'low': df['Low'].values.astype(np.float64),
        'open': df['Open'].values.astype(np.float64),
        'index': df.index,
    }


@numba.jit(nopython=True, cache=True)
def _compute_sma(arr, period):
    n = len(arr)
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += arr[i]
        if i >= period:
            s -= arr[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def compute_signals_4h(
    close, btc_close,
    lookback, threshold, target_retrace, stop_mult,
    max_hold, btc_sma_period,
):
    """Rubber band signals adapted for 4h timeframe.
    
    4h differences from 15m:
    - Lookback: 3-6 bars (12-24h vs 2h)
    - Threshold: higher (larger moves on 4h = more noise filter)
    - Max hold: 6-12 bars (1-2 days vs 2.5h)
    - No asset trend filter (4h has too few bars for reliable SMA)
    - BTC regime via longer SMA (24-48 bars = 4-8 days)
    """
    n = len(close)
    sig = np.zeros(n)
    btc_sma = _compute_sma(btc_close, btc_sma_period)
    
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_sma_period)
    
    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0
        
        if position == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue
                # BTC regime: above SMA = risk-on, below = risk-off
                btc_mom = btc_close[i] / btc_close[i - min(lookback * 2, i)] - 1.0
                
                # Simple regime-aligned entry
                if ret < -threshold:
                    # Long: buy the dip
                    position = 1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                elif ret > threshold:
                    # Short: sell the rip
                    position = -1
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            
            if gain > best_pnl:
                best_pnl = gain
            
            target = move_size * target_retrace
            effective_stop = move_size * stop_mult
            
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def compute_signals_4h_regime(
    close, btc_close,
    lookback, threshold, target_retrace, stop_mult,
    max_hold, btc_sma_period, btc_mom_period,
):
    """4h signals with BTC regime switch (adapted from v6.0 15m logic)."""
    n = len(close)
    sig = np.zeros(n)
    btc_sma = _compute_sma(btc_close, btc_sma_period)
    
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    start_bar = max(lookback, btc_sma_period, btc_mom_period)
    
    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0
        
        if position == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue
                
                is_risk_on = btc_close[i] > btc_sma[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False
                
                if is_risk_on:
                    # Risk-On: buy dips, short rips
                    if ret < -threshold:
                        position = 1
                        entered = True
                    elif ret > threshold:
                        position = -1
                        entered = True
                else:
                    # Risk-Off: sympathy trades with BTC direction
                    if ret < -threshold and btc_mom < 0:
                        position = 1
                        entered = True
                    elif ret > threshold and btc_mom > 0:
                        position = -1
                        entered = True
                
                if entered:
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            
            if gain > best_pnl:
                best_pnl = gain
            
            target = move_size * target_retrace
            effective_stop = move_size * stop_mult
            
            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0
        
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def resample_equity_to_15m(equity_4h, n_15m, ratio=16):
    """Expand 4h equity curve to 15m resolution via forward-fill.
    
    Each 4h bar spans 16 fifteen-minute bars (4h = 240min / 15min = 16).
    """
    out = np.ones(n_15m)
    for i in range(n_15m):
        j = min(i // ratio, len(equity_4h) - 1)
        out[i] = equity_4h[j]
    return out


@numba.jit(nopython=True, cache=True)
def combine_equity_curves(eq1, eq2, weight1):
    """Combine two equity curves with capital allocation weights.
    
    Returns portfolio equity = weight1 * eq1 + (1-weight1) * eq2
    Rebalancing at each bar.
    """
    n = min(len(eq1), len(eq2))
    combined = np.ones(n)
    w2 = 1.0 - weight1
    for i in range(1, n):
        r1 = eq1[i] / eq1[i-1] - 1.0
        r2 = eq2[i] / eq2[i-1] - 1.0
        combined[i] = combined[i-1] * (1.0 + weight1 * r1 + w2 * r2)
    return combined


def compute_metrics(equity):
    """Compute strategy metrics from equity curve."""
    n = len(equity)
    if n < 2:
        return {}
    
    total_ret = equity[-1] / equity[0] - 1.0
    
    # Sharpe (annualized, assuming 15m bars → 35040/year)
    returns = np.diff(equity) / equity[:-1]
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(35040)
    else:
        sharpe = 0.0
    
    # Max DD
    peak = equity[0]
    max_dd = 0.0
    for i in range(n):
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
    
    calmar = total_ret / max_dd if max_dd > 0 else 0.0
    
    return {
        'return': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
    }


def compute_metrics_4h(equity):
    """Compute strategy metrics using 4h bar frequency."""
    n = len(equity)
    if n < 2:
        return {}
    
    total_ret = equity[-1] / equity[0] - 1.0
    
    returns = np.diff(equity) / equity[:-1]
    # 4h bars → 2190/year
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(2190)
    else:
        sharpe = 0.0
    
    peak = equity[0]
    max_dd = 0.0
    for i in range(n):
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
    
    calmar = total_ret / max_dd if max_dd > 0 else 0.0
    
    return {
        'return': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
    }


def run_4h_sweep(all_data_4h, btc_4h):
    """Phase 1: Sweep 4h rubber band parameters."""
    print("\n" + "=" * 70)
    print("PHASE 1: 4H RUBBER BAND PARAMETER SWEEP")
    print("=" * 70)
    
    # 4h parameter ranges (different from 15m)
    lookbacks = [3, 4, 6]               # 12h, 16h, 24h
    thresholds = [0.04, 0.06, 0.08, 0.10]  # 4-10% (larger moves on 4h)
    targets = [0.3, 0.4, 0.5]           # 30-50% retracement
    stops = [0.8, 0.9, 1.0]             # stop multiplier
    max_holds = [6, 8, 12]              # 1-2 days
    btc_sma_periods = [24, 48]          # 4-8 days
    
    configs = list(product(lookbacks, thresholds, targets, stops, max_holds, btc_sma_periods))
    print(f"Testing {len(configs)} configurations across {len(all_data_4h)} symbols")
    
    results = []
    
    for lb, th, tgt, stp, mh, bsp in configs:
        port_returns = []
        port_equities = []
        total_trades = 0
        
        for sym, data in all_data_4h.items():
            c = data['close']
            sigs = compute_signals_4h_regime(
                c, btc_4h['close'],
                lb, th, tgt, stp, mh, bsp,
                min(lb * 2, bsp),  # btc_mom_period
            )
            
            eq, ret, sh, dd, wr, nt, bim, pf, atr, tr = _backtest_core(
                c, sigs, COST_PER_SIDE, FUNDING_PER_BAR_4H,
            )
            port_equities.append(eq)
            port_returns.append(ret)
            total_trades += nt
        
        if not port_equities:
            continue
        
        # Equal-weight portfolio
        combined = port_equities[0].copy()
        for eq in port_equities[1:]:
            min_len = min(len(combined), len(eq))
            combined = combined[:min_len]
            for j in range(1, min_len):
                r = eq[j] / eq[j-1] - 1.0
                combined[j] = combined[j-1] * (1.0 + combined[j] / combined[j-1] - 1.0 + r / len(port_equities))
        
        # Recompute properly: average returns
        n_syms = len(port_equities)
        min_len = min(len(e) for e in port_equities)
        combined = np.ones(min_len)
        for i in range(1, min_len):
            avg_r = 0.0
            for eq in port_equities:
                avg_r += (eq[i] / eq[i-1] - 1.0) / n_syms
            combined[i] = combined[i-1] * (1.0 + avg_r)
        
        m = compute_metrics_4h(combined)
        m['lookback'] = lb
        m['threshold'] = th
        m['target'] = tgt
        m['stop'] = stp
        m['max_hold'] = mh
        m['btc_sma'] = bsp
        m['n_trades'] = total_trades
        results.append(m)
    
    # Sort by Calmar
    results.sort(key=lambda x: x.get('calmar', 0), reverse=True)
    
    print(f"\n{'─' * 70}")
    print(f"TOP 10 CONFIGURATIONS (by Calmar):")
    print(f"{'─' * 70}")
    print(f"{'LB':>3} {'TH':>5} {'TGT':>4} {'STP':>4} {'MH':>3} {'BSP':>4} | "
          f"{'Ret':>7} {'DD':>6} {'Calmar':>7} {'Sharpe':>7} {'Trades':>6}")
    
    for r in results[:10]:
        print(f"{r['lookback']:3d} {r['threshold']:5.2f} {r['target']:4.1f} "
              f"{r['stop']:4.1f} {r['max_hold']:3d} {r['btc_sma']:4d} | "
              f"{r.get('return',0)*100:+6.1f}% {r.get('max_dd',0)*100:5.1f}% "
              f"{r.get('calmar',0):7.2f} {r.get('sharpe',0):7.2f} {r.get('n_trades',0):6d}")
    
    print(f"\n{'─' * 70}")
    print(f"BOTTOM 5 (worst):")
    for r in results[-5:]:
        print(f"{r['lookback']:3d} {r['threshold']:5.2f} {r['target']:4.1f} "
              f"{r['stop']:4.1f} {r['max_hold']:3d} {r['btc_sma']:4d} | "
              f"{r.get('return',0)*100:+6.1f}% {r.get('max_dd',0)*100:5.1f}% "
              f"{r.get('calmar',0):7.2f} {r.get('sharpe',0):7.2f} {r.get('n_trades',0):6d}")
    
    return results


def run_portfolio_combination(all_data_15m, all_data_4h, btc_15m, btc_4h, best_4h_params):
    """Phase 3: Test 15m + 4h portfolio allocation for MDD reduction."""
    print("\n" + "=" * 70)
    print("PHASE 3: PORTFOLIO COMBINATION (15m + 4h)")
    print("=" * 70)
    
    # v6.0 baseline: 15m confidence signals
    print("\nComputing v6.0 15m baseline...")
    eq_15m_per_sym = []
    for sym in TOP_10:
        if sym not in all_data_15m:
            continue
        d = all_data_15m[sym]
        sigs = compute_signals_confidence(
            d['close'], d['volume'], d['taker_buy'], btc_15m['close'],
            8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
            2, 1.5, 0.0, 1.5, 14, 20,
        )
        eq, *_ = _backtest_core(d['close'], sigs, COST_PER_SIDE, FUNDING_PER_BAR_15M)
        eq_15m_per_sym.append(eq)
    
    n_15m_syms = len(eq_15m_per_sym)
    min_len_15m = min(len(e) for e in eq_15m_per_sym)
    portfolio_15m = np.ones(min_len_15m)
    for i in range(1, min_len_15m):
        avg_r = sum((e[i] / e[i-1] - 1.0) for e in eq_15m_per_sym) / n_15m_syms
        portfolio_15m[i] = portfolio_15m[i-1] * (1.0 + avg_r)
    
    m_15m = compute_metrics(portfolio_15m)
    print(f"  15m-only: Ret={m_15m['return']*100:+.1f}% DD={m_15m['max_dd']*100:.1f}% "
          f"Calmar={m_15m['calmar']:.2f} Sharpe={m_15m['sharpe']:.2f}")
    
    # Compute 4h equity curves with best params
    print(f"\nComputing 4h signals with best params: {best_4h_params}")
    eq_4h_per_sym = []
    for sym in TOP_10:
        if sym not in all_data_4h:
            continue
        d = all_data_4h[sym]
        p = best_4h_params
        sigs = compute_signals_4h_regime(
            d['close'], btc_4h['close'],
            p['lookback'], p['threshold'], p['target'], p['stop'],
            p['max_hold'], p['btc_sma'],
            min(p['lookback'] * 2, p['btc_sma']),
        )
        eq, *_ = _backtest_core(d['close'], sigs, COST_PER_SIDE, FUNDING_PER_BAR_4H)
        eq_4h_per_sym.append(eq)
    
    n_4h_syms = len(eq_4h_per_sym)
    min_len_4h = min(len(e) for e in eq_4h_per_sym)
    portfolio_4h = np.ones(min_len_4h)
    for i in range(1, min_len_4h):
        avg_r = sum((e[i] / e[i-1] - 1.0) for e in eq_4h_per_sym) / n_4h_syms
        portfolio_4h[i] = portfolio_4h[i-1] * (1.0 + avg_r)
    
    m_4h = compute_metrics_4h(portfolio_4h)
    print(f"  4h-only:  Ret={m_4h['return']*100:+.1f}% DD={m_4h['max_dd']*100:.1f}% "
          f"Calmar={m_4h['calmar']:.2f} Sharpe={m_4h['sharpe']:.2f}")
    
    # Resample 4h equity to 15m resolution for combining
    portfolio_4h_resampled = resample_equity_to_15m(portfolio_4h, min_len_15m, 16)
    
    # Test different capital allocations
    allocations = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    print(f"\n{'─' * 70}")
    print(f"CAPITAL ALLOCATION: weight_15m × 15m + (1-weight_15m) × 4h")
    print(f"{'─' * 70}")
    print(f"{'W_15m':>6} {'W_4h':>5} | {'Ret':>7} {'DD':>6} {'Calmar':>7} {'Sharpe':>7}")
    
    best_combo = None
    best_calmar = 0
    
    for w15 in allocations:
        combined = combine_equity_curves(portfolio_15m, portfolio_4h_resampled, w15)
        m = compute_metrics(combined)
        print(f"{w15:6.2f} {1-w15:5.2f} | {m['return']*100:+6.1f}% {m['max_dd']*100:5.1f}% "
              f"{m['calmar']:7.2f} {m['sharpe']:7.2f}")
        
        if m['calmar'] > best_calmar:
            best_calmar = m['calmar']
            best_combo = {'w15': w15, 'w4h': 1-w15, **m}
    
    print(f"\n✓ Best combo: {best_combo['w15']*100:.0f}% 15m + {best_combo['w4h']*100:.0f}% 4h")
    print(f"  Ret={best_combo['return']*100:+.1f}% DD={best_combo['max_dd']*100:.1f}% "
          f"Calmar={best_combo['calmar']:.2f} Sharpe={best_combo['sharpe']:.2f}")
    
    # Key comparison
    print(f"\n{'═' * 70}")
    print(f"MDD REDUCTION ANALYSIS:")
    print(f"{'═' * 70}")
    print(f"  15m-only MDD:  {m_15m['max_dd']*100:.1f}%")
    print(f"  4h-only MDD:   {m_4h['max_dd']*100:.1f}%")
    print(f"  Combined MDD:  {best_combo['max_dd']*100:.1f}%")
    dd_reduction = (m_15m['max_dd'] - best_combo['max_dd']) / m_15m['max_dd'] * 100
    print(f"  MDD reduction: {dd_reduction:+.1f}% vs 15m-only")
    
    return best_combo, m_15m, m_4h


def run_mtf_confirmation(all_data_15m, all_data_4h, btc_15m, btc_4h):
    """Phase 4: Test 4h trend as confirmation filter for 15m entries."""
    print("\n" + "=" * 70)
    print("PHASE 4: MTF CONFIRMATION (4h trend validates 15m entry)")
    print("=" * 70)
    
    # Concept: only take 15m entries when 4h trend agrees
    # 4h trend = close > SMA(period) on 4h
    # Long 15m entry → only if 4h is bullish (or oversold)
    # Short 15m entry → only if 4h is bearish (or overbought)
    
    confirmation_periods = [6, 12, 24]  # 4h SMA periods (1d, 2d, 4d)
    
    print(f"\n{'─' * 70}")
    print(f"{'Period':>7} | {'Ret':>7} {'DD':>6} {'Calmar':>7} {'Sharpe':>7} {'Trades':>6}")
    
    # Baseline without MTF confirmation
    baseline_eqs = []
    baseline_trades = 0
    for sym in TOP_10:
        if sym not in all_data_15m:
            continue
        d = all_data_15m[sym]
        sigs = compute_signals_confidence(
            d['close'], d['volume'], d['taker_buy'], btc_15m['close'],
            8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
            2, 1.5, 0.0, 1.5, 14, 20,
        )
        eq, _, _, _, _, nt, *_ = _backtest_core(d['close'], sigs, COST_PER_SIDE, FUNDING_PER_BAR_15M)
        baseline_eqs.append(eq)
        baseline_trades += nt
    
    min_len = min(len(e) for e in baseline_eqs)
    port_base = np.ones(min_len)
    for i in range(1, min_len):
        avg_r = sum((e[i] / e[i-1] - 1.0) for e in baseline_eqs) / len(baseline_eqs)
        port_base[i] = port_base[i-1] * (1.0 + avg_r)
    m_base = compute_metrics(port_base)
    print(f"{'No MTF':>7} | {m_base['return']*100:+6.1f}% {m_base['max_dd']*100:5.1f}% "
          f"{m_base['calmar']:7.2f} {m_base['sharpe']:7.2f} {baseline_trades:6d}")
    
    for period_4h in confirmation_periods:
        mtf_eqs = []
        mtf_trades = 0
        
        for sym in TOP_10:
            if sym not in all_data_15m or sym not in all_data_4h:
                continue
            d15 = all_data_15m[sym]
            d4h = all_data_4h[sym]
            
            # Compute 4h SMA trend
            close_4h = d4h['close']
            sma_4h = _compute_sma(close_4h, period_4h)
            
            # Compute 15m signals normally
            sigs = compute_signals_confidence(
                d15['close'], d15['volume'], d15['taker_buy'], btc_15m['close'],
                8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
                2, 1.5, 0.0, 1.5, 14, 20,
            )
            
            # Apply MTF confirmation: zero out signals that disagree with 4h trend
            n_15m = len(sigs)
            for i in range(n_15m):
                if sigs[i] == 0:
                    continue
                # Map 15m bar index to 4h bar index
                j = min(i // 16, len(close_4h) - 1)
                if np.isnan(sma_4h[j]):
                    continue
                
                trend_up = close_4h[j] > sma_4h[j]
                
                if sigs[i] > 0 and not trend_up:
                    # Long but 4h bearish → block
                    sigs[i] = 0
                elif sigs[i] < 0 and trend_up:
                    # Short but 4h bullish → block
                    sigs[i] = 0
            
            eq, _, _, _, _, nt, *_ = _backtest_core(d15['close'], sigs, COST_PER_SIDE, FUNDING_PER_BAR_15M)
            mtf_eqs.append(eq)
            mtf_trades += nt
        
        min_len = min(len(e) for e in mtf_eqs)
        port = np.ones(min_len)
        for i in range(1, min_len):
            avg_r = sum((e[i] / e[i-1] - 1.0) for e in mtf_eqs) / len(mtf_eqs)
            port[i] = port[i-1] * (1.0 + avg_r)
        
        m = compute_metrics(port)
        label = f"4h_{period_4h}"
        print(f"{label:>7} | {m['return']*100:+6.1f}% {m['max_dd']*100:5.1f}% "
              f"{m['calmar']:7.2f} {m['sharpe']:7.2f} {mtf_trades:6d}")
    
    return m_base


def run_5m_exploration(btc_5m):
    """Phase 2: Test 5m rubber band signals if data available."""
    print("\n" + "=" * 70)
    print("PHASE 2: 5M RUBBER BAND PARAMETER SWEEP")
    print("=" * 70)
    
    # Check for 5m data
    available = []
    all_data_5m = {}
    for sym in TOP_10:
        data = load_data(sym, "5m")
        if data is not None:
            available.append(sym)
            all_data_5m[sym] = data
    
    if not available:
        print("⚠ No 5m data available. Run: uv run python scripts/download_mtf_data.py --interval 5m")
        return None, None
    
    print(f"5m data available for {len(available)}/{len(TOP_10)} symbols")
    
    # 5m parameter ranges (faster mean reversion)
    lookbacks = [12, 16, 24]            # 1h, 1.3h, 2h (in 5m bars)
    thresholds = [0.03, 0.035, 0.04, 0.045]  # 3-4.5% (smaller on shorter TF)
    targets = [0.3, 0.4, 0.5]
    stops = [0.8, 0.9, 1.0]
    max_holds = [12, 18, 24]            # 1h, 1.5h, 2h
    
    configs = list(product(lookbacks, thresholds, targets, stops, max_holds))
    print(f"Testing {len(configs)} configs")
    
    results = []
    for lb, th, tgt, stp, mh in configs:
        eqs = []
        total_trades = 0
        
        for sym, data in all_data_5m.items():
            c = data['close']
            bc = btc_5m['close'] if btc_5m else c  # fallback
            
            # Simple 5m signals (no regime switch — too fast for regime)
            n = len(c)
            sig = np.zeros(n)
            position = 0
            entry_price = 0.0
            move_size = 0.0
            hold_count = 0
            best_pnl = 0.0
            
            for i in range(lb, n):
                ret = c[i] / c[i - lb] - 1.0
                if position == 0:
                    if ret < -th:
                        position = 1
                        entry_price = c[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                    elif ret > th:
                        position = -1
                        entry_price = c[i]
                        move_size = abs(ret)
                        hold_count = 0
                        best_pnl = 0.0
                else:
                    hold_count += 1
                    gain = (c[i] - entry_price) / entry_price if position == 1 else (entry_price - c[i]) / entry_price
                    if gain > best_pnl:
                        best_pnl = gain
                    target = move_size * tgt
                    effective_stop = move_size * stp
                    if gain >= target or gain < -effective_stop or hold_count >= mh:
                        position = 0
                    elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                        position = 0
                sig[i] = position
            
            eq, _, _, _, _, nt, *_ = _backtest_core(c, sig, COST_PER_SIDE, FUNDING_PER_BAR_5M)
            eqs.append(eq)
            total_trades += nt
        
        if not eqs:
            continue
        
        n_syms = len(eqs)
        min_len = min(len(e) for e in eqs)
        combined = np.ones(min_len)
        for i in range(1, min_len):
            avg_r = sum((e[i] / e[i-1] - 1.0) for e in eqs) / n_syms
            combined[i] = combined[i-1] * (1.0 + avg_r)
        
        # 5m metrics (5m bars → 105120/year)
        returns = np.diff(combined) / combined[:-1]
        total_ret = combined[-1] - 1.0
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(105120) if np.std(returns) > 0 else 0
        peak = combined[0]
        max_dd = 0.0
        for v in combined:
            if v > peak: peak = v
            dd = (peak - v) / peak
            if dd > max_dd: max_dd = dd
        calmar = total_ret / max_dd if max_dd > 0 else 0
        
        results.append({
            'lookback': lb, 'threshold': th, 'target': tgt,
            'stop': stp, 'max_hold': mh,
            'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
            'calmar': calmar, 'n_trades': total_trades,
        })
    
    results.sort(key=lambda x: x.get('calmar', 0), reverse=True)
    
    print(f"\n{'─' * 70}")
    print(f"TOP 10 5M CONFIGS (by Calmar):")
    print(f"{'─' * 70}")
    print(f"{'LB':>3} {'TH':>5} {'TGT':>4} {'STP':>4} {'MH':>3} | "
          f"{'Ret':>7} {'DD':>6} {'Calmar':>7} {'Sharpe':>7} {'Trades':>6}")
    
    for r in results[:10]:
        print(f"{r['lookback']:3d} {r['threshold']:5.3f} {r['target']:4.1f} "
              f"{r['stop']:4.1f} {r['max_hold']:3d} | "
              f"{r['return']*100:+6.1f}% {r['max_dd']*100:5.1f}% "
              f"{r['calmar']:7.2f} {r['sharpe']:7.2f} {r['n_trades']:6d}")
    
    return results, all_data_5m


def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME TRADING A/B TEST")
    print("Goal: Reduce MDD through TF diversification")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading data...")
    all_data_15m = {}
    all_data_4h = {}
    
    for sym in TOP_10:
        d15 = load_data(sym, "15m")
        if d15:
            all_data_15m[sym] = d15
        d4h = load_data(sym, "4h")
        if d4h:
            all_data_4h[sym] = d4h
    
    btc_15m = load_data("BTCUSDT", "15m")
    btc_4h = load_data("BTCUSDT", "4h")
    btc_5m = load_data("BTCUSDT", "5m")
    
    print(f"  15m: {len(all_data_15m)} symbols loaded")
    print(f"  4h:  {len(all_data_4h)} symbols loaded")
    
    if not btc_4h:
        print("❌ No 4h BTC data. Run: python scripts/download_mtf_data.py --interval 4h")
        return
    
    # Numba warmup
    print("\n⚡ Numba warmup...")
    dummy = np.random.randn(100).cumsum() + 100
    dummy = np.abs(dummy)
    _compute_sma(dummy, 5)
    compute_signals_4h(dummy, dummy, 3, 0.05, 0.4, 0.9, 6, 12)
    compute_signals_4h_regime(dummy, dummy, 3, 0.05, 0.4, 0.9, 6, 12, 6)
    _backtest_core(dummy, np.zeros(100), 0.0004, 0.0001)
    
    # Phase 1: 4h sweep
    results_4h = run_4h_sweep(all_data_4h, btc_4h)
    
    if results_4h and results_4h[0].get('calmar', 0) > 0:
        best_4h = results_4h[0]
    else:
        print("\n⚠ No viable 4h configs found. Trying relaxed parameters...")
        best_4h = {'lookback': 4, 'threshold': 0.06, 'target': 0.4,
                    'stop': 0.9, 'max_hold': 8, 'btc_sma': 24}
    
    # Phase 2: 5m exploration (if data available)
    results_5m, all_data_5m = run_5m_exploration(btc_5m)
    
    # Phase 3: Portfolio combination
    best_combo, m_15m, m_4h = run_portfolio_combination(
        all_data_15m, all_data_4h, btc_15m, btc_4h, best_4h,
    )
    
    # Phase 4: MTF confirmation
    run_mtf_confirmation(all_data_15m, all_data_4h, btc_15m, btc_4h)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n15m-only (v6.0):      Ret={m_15m['return']*100:+.1f}% DD={m_15m['max_dd']*100:.1f}% "
          f"Calmar={m_15m['calmar']:.2f} Sharpe={m_15m['sharpe']:.2f}")
    print(f"4h-only (best):       Ret={m_4h['return']*100:+.1f}% DD={m_4h['max_dd']*100:.1f}% "
          f"Calmar={m_4h['calmar']:.2f} Sharpe={m_4h['sharpe']:.2f}")
    if best_combo:
        print(f"Combined ({best_combo['w15']*100:.0f}%/{ best_combo['w4h']*100:.0f}%): "
              f"Ret={best_combo['return']*100:+.1f}% DD={best_combo['max_dd']*100:.1f}% "
              f"Calmar={best_combo['calmar']:.2f} Sharpe={best_combo['sharpe']:.2f}")
    
    if results_5m:
        best_5m = results_5m[0]
        print(f"5m-only (best):       Ret={best_5m['return']*100:+.1f}% DD={best_5m['max_dd']*100:.1f}% "
              f"Calmar={best_5m['calmar']:.2f} Sharpe={best_5m['sharpe']:.2f}")
    
    print(f"\n{'─' * 70}")
    print("Decision framework:")
    print("  - If 4h + 15m reduces MDD by >10%: implement MTF portfolio")
    print("  - If MTF confirmation improves Sharpe: add 4h trend filter")
    print("  - If uncertain (marginal improvement): stick with 15m-only (v6.0)")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
