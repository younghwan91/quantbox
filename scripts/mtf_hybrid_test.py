#!/usr/bin/env python3
"""
v7.0 Hybrid MTF Strategy Test

Tests the optimal combination:
  1. 15m confidence signals + 4h SMA trend confirmation
  2. Independent 4h rubber band trades
  3. Portfolio combination with leverage recovery

Key insight from mtf_trading_ab_test.py:
  - MTF confirmation (4h_12) → DD 2.2%, Calmar 19.64
  - With DD this low, we can safely apply leverage to recover return
  - Independent 4h trades add uncorrelated alpha

Usage:
    uv run python scripts/mtf_hybrid_test.py
"""

import sys
import gzip
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import numba

from quantbox.strategy.rubberband import compute_signals_confidence
from quantbox.backtest.vectorized_backtest import _backtest_core

CACHE_DIR = Path("data/cache_extended")
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR_15M = 0.0001 / 32
FUNDING_PER_BAR_4H = 0.0001 / 2

TOP_10 = [
    "FETUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "DOTUSDT",
    "OPUSDT", "SHIBUSDT", "DOGEUSDT", "LTCUSDT", "ATOMUSDT"
]


def load_data(symbol: str, interval: str) -> dict | None:
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
def apply_mtf_confirmation(signals_15m, close_4h, sma_period_4h, ratio=16):
    """Zero out 15m signals that disagree with 4h trend direction.
    
    Rule: Only take long when 4h close > SMA, short when 4h close < SMA.
    This filters out counter-trend entries that tend to lose.
    
    Args:
        signals_15m: 15m signal array (modified in-place)
        close_4h: 4h close prices
        sma_period_4h: SMA period on 4h (12 = 2 days)
        ratio: 15m bars per 4h bar (16)
    
    Returns:
        Filtered signal array
    """
    sma_4h = _compute_sma(close_4h, sma_period_4h)
    n = len(signals_15m)
    out = signals_15m.copy()
    
    for i in range(n):
        if out[i] == 0.0:
            continue
        j = min(i // ratio, len(close_4h) - 1)
        if np.isnan(sma_4h[j]):
            continue
        
        trend_up = close_4h[j] > sma_4h[j]
        
        # Block counter-trend entries
        if out[i] > 0 and not trend_up:
            out[i] = 0.0
        elif out[i] < 0 and trend_up:
            out[i] = 0.0
    
    return out


@numba.jit(nopython=True, cache=True)
def compute_signals_4h_regime(
    close, btc_close,
    lookback, threshold, target_retrace, stop_mult,
    max_hold, btc_sma_period, btc_mom_period,
):
    """4h rubber band signals with BTC regime switch."""
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
                    if ret < -threshold:
                        position = 1
                        entered = True
                    elif ret > threshold:
                        position = -1
                        entered = True
                else:
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
def apply_leverage_to_signals(signals, leverage):
    """Scale signal magnitude by leverage factor."""
    n = len(signals)
    out = np.zeros(n)
    for i in range(n):
        out[i] = signals[i] * leverage
    return out


@numba.jit(nopython=True, cache=True)
def resample_equity_to_15m(equity_4h, n_15m, ratio=16):
    """Expand 4h equity to 15m resolution via forward-fill."""
    out = np.ones(n_15m)
    for i in range(n_15m):
        j = min(i // ratio, len(equity_4h) - 1)
        out[i] = equity_4h[j]
    return out


@numba.jit(nopython=True, cache=True)
def combine_equity_curves(eq1, eq2, weight1):
    """Combine two equity curves with capital allocation."""
    n = min(len(eq1), len(eq2))
    combined = np.ones(n)
    w2 = 1.0 - weight1
    for i in range(1, n):
        r1 = eq1[i] / eq1[i-1] - 1.0
        r2 = eq2[i] / eq2[i-1] - 1.0
        combined[i] = combined[i-1] * (1.0 + weight1 * r1 + w2 * r2)
    return combined


def compute_metrics(equity, bars_per_year=35040):
    """Compute strategy metrics from equity curve."""
    n = len(equity)
    if n < 2:
        return {}
    total_ret = equity[-1] / equity[0] - 1.0
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year) if np.std(returns) > 0 else 0
    peak = equity[0]
    max_dd = 0.0
    for i in range(n):
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
    calmar = total_ret / max_dd if max_dd > 0 else 0
    return {'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd, 'calmar': calmar}


def portfolio_equity(eqs, bars_per_year=35040):
    """Equal-weight portfolio equity from list of equity curves."""
    n_syms = len(eqs)
    min_len = min(len(e) for e in eqs)
    combined = np.ones(min_len)
    for i in range(1, min_len):
        avg_r = sum((e[i] / e[i-1] - 1.0) for e in eqs) / n_syms
        combined[i] = combined[i-1] * (1.0 + avg_r)
    return combined


def main():
    print("=" * 70)
    print("v7.0 HYBRID MTF STRATEGY TEST")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    all_15m = {}
    all_4h = {}
    for sym in TOP_10:
        d = load_data(sym, "15m")
        if d:
            all_15m[sym] = d
        d = load_data(sym, "4h")
        if d:
            all_4h[sym] = d
    
    btc_15m = load_data("BTCUSDT", "15m")
    btc_4h = load_data("BTCUSDT", "4h")
    print(f"  15m: {len(all_15m)} symbols, 4h: {len(all_4h)} symbols")
    
    # Warmup
    dummy = np.abs(np.random.randn(100).cumsum() + 100)
    _compute_sma(dummy, 5)
    apply_mtf_confirmation(np.zeros(100), dummy, 12, 16)
    compute_signals_4h_regime(dummy, dummy, 3, 0.05, 0.4, 0.9, 6, 12, 6)
    _backtest_core(dummy, np.zeros(100), 0.0004, 0.0001)
    
    # ── BASELINE: v6.0 15m only ──
    print("\n" + "=" * 70)
    print("BASELINE: v6.0 15m Confidence (no MTF)")
    print("=" * 70)
    
    eq_15m_base = []
    trades_15m_base = 0
    for sym in TOP_10:
        if sym not in all_15m:
            continue
        d = all_15m[sym]
        sigs = compute_signals_confidence(
            d['close'], d['volume'], d['taker_buy'], btc_15m['close'],
            8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
            2, 1.5, 0.0, 1.5, 14, 20,
        )
        eq, _, _, _, _, nt, *_ = _backtest_core(d['close'], sigs, COST_PER_SIDE, FUNDING_PER_BAR_15M)
        eq_15m_base.append(eq)
        trades_15m_base += nt
    
    port_base = portfolio_equity(eq_15m_base)
    m_base = compute_metrics(port_base)
    print(f"  Ret={m_base['return']*100:+.1f}%  DD={m_base['max_dd']*100:.1f}%  "
          f"Calmar={m_base['calmar']:.2f}  Sharpe={m_base['sharpe']:.2f}  Trades={trades_15m_base}")
    
    # ── TEST A: MTF-confirmed 15m only ──
    print("\n" + "=" * 70)
    print("TEST A: 15m + 4h SMA(12) MTF Confirmation")
    print("=" * 70)
    
    for sma_period in [6, 12, 18, 24]:
        eq_list = []
        total_trades = 0
        for sym in TOP_10:
            if sym not in all_15m or sym not in all_4h:
                continue
            d15 = all_15m[sym]
            d4h = all_4h[sym]
            sigs = compute_signals_confidence(
                d15['close'], d15['volume'], d15['taker_buy'], btc_15m['close'],
                8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
                2, 1.5, 0.0, 1.5, 14, 20,
            )
            sigs_filtered = apply_mtf_confirmation(sigs, d4h['close'], sma_period, 16)
            eq, _, _, _, _, nt, *_ = _backtest_core(d15['close'], sigs_filtered, COST_PER_SIDE, FUNDING_PER_BAR_15M)
            eq_list.append(eq)
            total_trades += nt
        
        port = portfolio_equity(eq_list)
        m = compute_metrics(port)
        print(f"  SMA({sma_period:2d}): Ret={m['return']*100:+6.1f}%  DD={m['max_dd']*100:5.1f}%  "
              f"Calmar={m['calmar']:7.2f}  Sharpe={m['sharpe']:5.2f}  Trades={total_trades}")
    
    # ── TEST B: MTF-confirmed 15m + Leverage recovery ──
    print("\n" + "=" * 70)
    print("TEST B: MTF-confirmed 15m + Leverage (recover return)")
    print("=" * 70)
    print("(Using SMA_12 confirmation, varying leverage)")
    
    leverages = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for lev in leverages:
        eq_list = []
        total_trades = 0
        for sym in TOP_10:
            if sym not in all_15m or sym not in all_4h:
                continue
            d15 = all_15m[sym]
            d4h = all_4h[sym]
            sigs = compute_signals_confidence(
                d15['close'], d15['volume'], d15['taker_buy'], btc_15m['close'],
                8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
                2, 1.5, 0.0, 1.5, 14, 20,
            )
            sigs_filtered = apply_mtf_confirmation(sigs, d4h['close'], 12, 16)
            sigs_lev = apply_leverage_to_signals(sigs_filtered, lev)
            eq, _, _, _, _, nt, *_ = _backtest_core(d15['close'], sigs_lev, COST_PER_SIDE, FUNDING_PER_BAR_15M)
            eq_list.append(eq)
            total_trades += nt
        
        port = portfolio_equity(eq_list)
        m = compute_metrics(port)
        dd_vs_base = m['max_dd'] / m_base['max_dd'] if m_base['max_dd'] > 0 else 0
        print(f"  {lev:.1f}x: Ret={m['return']*100:+7.1f}%  DD={m['max_dd']*100:5.1f}%  "
              f"Calmar={m['calmar']:7.2f}  Sharpe={m['sharpe']:5.2f}  "
              f"DD_ratio={dd_vs_base:.2f}x baseline")
    
    # ── TEST C: Independent 4h trades ──
    print("\n" + "=" * 70)
    print("TEST C: Independent 4H Rubber Band Strategy")
    print("=" * 70)
    
    eq_4h_list = []
    trades_4h = 0
    for sym in TOP_10:
        if sym not in all_4h:
            continue
        d = all_4h[sym]
        sigs = compute_signals_4h_regime(
            d['close'], btc_4h['close'],
            3, 0.10, 0.5, 0.8, 12, 48, 6,
        )
        eq, _, _, _, _, nt, *_ = _backtest_core(d['close'], sigs, COST_PER_SIDE, FUNDING_PER_BAR_4H)
        eq_4h_list.append(eq)
        trades_4h += nt
    
    port_4h = portfolio_equity(eq_4h_list)
    m_4h = compute_metrics(port_4h, bars_per_year=2190)
    print(f"  4h-only: Ret={m_4h['return']*100:+.1f}%  DD={m_4h['max_dd']*100:.1f}%  "
          f"Calmar={m_4h['calmar']:.2f}  Sharpe={m_4h['sharpe']:.2f}  Trades={trades_4h}")
    
    # ── TEST D: v7.0 HYBRID — MTF-confirmed 15m + independent 4h ──
    print("\n" + "=" * 70)
    print("TEST D: v7.0 HYBRID (MTF-confirmed 15m + independent 4h)")
    print("=" * 70)
    
    # Compute MTF-confirmed 15m portfolio equity (with leverage sweep)
    best_hybrid = None
    best_calmar = 0
    
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for w15 in [0.5, 0.6, 0.7, 0.8, 0.9]:
            eq_15m_mtf = []
            for sym in TOP_10:
                if sym not in all_15m or sym not in all_4h:
                    continue
                d15 = all_15m[sym]
                d4h = all_4h[sym]
                sigs = compute_signals_confidence(
                    d15['close'], d15['volume'], d15['taker_buy'], btc_15m['close'],
                    8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
                    2, 1.5, 0.0, 1.5, 14, 20,
                )
                sigs_filtered = apply_mtf_confirmation(sigs, d4h['close'], 12, 16)
                sigs_lev = apply_leverage_to_signals(sigs_filtered, lev)
                eq, *_ = _backtest_core(d15['close'], sigs_lev, COST_PER_SIDE, FUNDING_PER_BAR_15M)
                eq_15m_mtf.append(eq)
            
            port_15m_mtf = portfolio_equity(eq_15m_mtf)
            
            # Resample 4h to 15m and combine
            port_4h_resampled = resample_equity_to_15m(port_4h, len(port_15m_mtf), 16)
            combined = combine_equity_curves(port_15m_mtf, port_4h_resampled, w15)
            m = compute_metrics(combined)
            
            if m['calmar'] > best_calmar:
                best_calmar = m['calmar']
                best_hybrid = {
                    'leverage': lev, 'w15': w15, 'w4h': 1 - w15, **m,
                }
    
    print(f"\n  BEST HYBRID: {best_hybrid['leverage']:.1f}x lev, "
          f"{best_hybrid['w15']*100:.0f}%/{best_hybrid['w4h']*100:.0f}% split")
    print(f"  Ret={best_hybrid['return']*100:+.1f}%  DD={best_hybrid['max_dd']*100:.1f}%  "
          f"Calmar={best_hybrid['calmar']:.2f}  Sharpe={best_hybrid['sharpe']:.2f}")
    
    # Show top 10 hybrid configs
    print(f"\n{'─' * 70}")
    print(f"TOP HYBRID CONFIGURATIONS:")
    print(f"{'─' * 70}")
    print(f"{'Lev':>4} {'W15':>4} {'W4h':>4} | {'Ret':>8} {'DD':>6} {'Calmar':>7} {'Sharpe':>7}")
    
    # Recompute for all and sort
    all_hybrids = []
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
        eq_15m_mtf = []
        for sym in TOP_10:
            if sym not in all_15m or sym not in all_4h:
                continue
            d15 = all_15m[sym]
            d4h = all_4h[sym]
            sigs = compute_signals_confidence(
                d15['close'], d15['volume'], d15['taker_buy'], btc_15m['close'],
                8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
                2, 1.5, 0.0, 1.5, 14, 20,
            )
            sigs_filtered = apply_mtf_confirmation(sigs, d4h['close'], 12, 16)
            sigs_lev = apply_leverage_to_signals(sigs_filtered, lev)
            eq, *_ = _backtest_core(d15['close'], sigs_lev, COST_PER_SIDE, FUNDING_PER_BAR_15M)
            eq_15m_mtf.append(eq)
        
        port_15m_mtf = portfolio_equity(eq_15m_mtf)
        port_4h_r = resample_equity_to_15m(port_4h, len(port_15m_mtf), 16)
        
        for w15 in [0.5, 0.6, 0.7, 0.8, 0.9]:
            combined = combine_equity_curves(port_15m_mtf, port_4h_r, w15)
            m = compute_metrics(combined)
            all_hybrids.append({
                'leverage': lev, 'w15': w15, 'w4h': 1 - w15, **m,
            })
    
    all_hybrids.sort(key=lambda x: x.get('calmar', 0), reverse=True)
    for h in all_hybrids[:15]:
        print(f"{h['leverage']:4.1f} {h['w15']:4.1f} {h['w4h']:4.1f} | "
              f"{h['return']*100:+7.1f}% {h['max_dd']*100:5.1f}% "
              f"{h['calmar']:7.2f} {h['sharpe']:7.2f}")
    
    # ── GRAND COMPARISON ──
    print("\n" + "=" * 70)
    print("GRAND COMPARISON")
    print("=" * 70)
    
    dd_reduce_pct = (m_base['max_dd'] - best_hybrid['max_dd']) / m_base['max_dd'] * 100
    
    print(f"\n  {'Strategy':30s} | {'Ret':>8} {'DD':>6} {'Calmar':>7} {'Sharpe':>7}")
    print(f"  {'─' * 68}")
    print(f"  {'v6.0 (15m only)':30s} | {m_base['return']*100:+7.1f}% {m_base['max_dd']*100:5.1f}% "
          f"{m_base['calmar']:7.2f} {m_base['sharpe']:7.2f}")
    print(f"  {'v7.0 hybrid':30s} | {best_hybrid['return']*100:+7.1f}% {best_hybrid['max_dd']*100:5.1f}% "
          f"{best_hybrid['calmar']:7.2f} {best_hybrid['sharpe']:7.2f}")
    print(f"  {'4h-only ref':30s} | {m_4h['return']*100:+7.1f}% {m_4h['max_dd']*100:5.1f}% "
          f"{m_4h['calmar']:7.2f} {m_4h['sharpe']:7.2f}")
    
    print(f"\n  MDD reduction: {dd_reduce_pct:+.1f}% ({m_base['max_dd']*100:.1f}% → {best_hybrid['max_dd']*100:.1f}%)")
    calmar_improve = (best_hybrid['calmar'] - m_base['calmar']) / m_base['calmar'] * 100
    print(f"  Calmar improvement: {calmar_improve:+.1f}% ({m_base['calmar']:.2f} → {best_hybrid['calmar']:.2f})")
    
    # ── ROBUSTNESS: Walk-forward validation ──
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Walk-Forward Validation (4 quarters)")
    print("=" * 70)
    
    lev = best_hybrid['leverage']
    w15 = best_hybrid['w15']
    
    # Split 365 days into 4 quarters (approximate)
    n_bars_15m = min(len(e) for e in eq_15m_base)
    q_size = n_bars_15m // 4
    
    for q in range(4):
        start = q * q_size
        end = min((q + 1) * q_size, n_bars_15m)
        
        # Base 15m
        base_returns = []
        for eq in eq_15m_base:
            eq_q = eq[start:end]
            if len(eq_q) > 1:
                r = eq_q[-1] / eq_q[0] - 1.0
                base_returns.append(r)
        
        # Hybrid 15m (MTF confirmed + leverage)
        hybrid_eqs = []
        for sym in TOP_10:
            if sym not in all_15m or sym not in all_4h:
                continue
            d15 = all_15m[sym]
            d4h = all_4h[sym]
            sigs = compute_signals_confidence(
                d15['close'], d15['volume'], d15['taker_buy'], btc_15m['close'],
                8, 0.045, 0.4, 0.9, 10, 0.0, 32, 96, 192,
                2, 1.5, 0.0, 1.5, 14, 20,
            )
            sigs_filtered = apply_mtf_confirmation(sigs, d4h['close'], 12, 16)
            sigs_lev = apply_leverage_to_signals(sigs_filtered, lev)
            eq, *_ = _backtest_core(d15['close'], sigs_lev, COST_PER_SIDE, FUNDING_PER_BAR_15M)
            hybrid_eqs.append(eq)
        
        port_h = portfolio_equity(hybrid_eqs)
        
        # Quarter slice
        eq_slice = port_h[start:end] if end <= len(port_h) else port_h[start:]
        m_q = compute_metrics(eq_slice)
        
        base_q = portfolio_equity(eq_15m_base)
        eq_base_slice = base_q[start:end] if end <= len(base_q) else base_q[start:]
        m_bq = compute_metrics(eq_base_slice)
        
        print(f"  Q{q+1}: Base: Ret={m_bq.get('return',0)*100:+6.1f}% DD={m_bq.get('max_dd',0)*100:5.1f}% | "
              f"Hybrid: Ret={m_q.get('return',0)*100:+6.1f}% DD={m_q.get('max_dd',0)*100:5.1f}% | "
              f"DD reduction: {((m_bq.get('max_dd',0.001) - m_q.get('max_dd',0)) / m_bq.get('max_dd',0.001))*100:+.0f}%")
    
    print(f"\n{'─' * 70}")
    if dd_reduce_pct > 20 and best_hybrid['calmar'] > m_base['calmar']:
        print("✓ DECISION: Implement v7.0 hybrid MTF strategy")
        print(f"  Config: 15m confidence + 4h SMA(12) confirmation + {lev:.1f}x leverage")
        print(f"  Portfolio: {w15*100:.0f}% 15m / {(1-w15)*100:.0f}% 4h")
        print(f"  4h params: lookback=3, threshold=0.10, target=0.5, stop=0.8, max_hold=12, btc_sma=48")
    else:
        print("✗ Hybrid improvement insufficient. Stay with v6.0.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
