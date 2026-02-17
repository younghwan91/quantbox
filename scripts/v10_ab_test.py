#!/usr/bin/env python3
"""v10 A/B Test — Extract more alpha from idle capital.

Problem: 98.4% of bars have ZERO positions. Capital is doing nothing.
Goal: Find safe ways to increase trade frequency without degrading Calmar.

Hypotheses from winning_strategy.md:
  H1: Lower threshold (4.5% → 3.5%/3.0%/2.5%) — capture smaller moves
  H2: RSI divergence as entry confirmation (was -18% in v8 but v9 changes landscape)
  H3: Lower min_confidence from 2 → 1 (accept more entries with MTF as quality gate)
  H4: Shorter lookback (8 → 6/5/4) — faster signal detection
  H5: Dynamic threshold — lower threshold when MTF confirms trend alignment
  H6: EMA regime filter (winning strategy's EMA stack) instead of SMA

Tests run on Top-12 profitable symbols with MTF (production parity).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from quantbox.strategy.rubberband import (
    compute_signals_v9, apply_mtf_confirmation, apply_mtf_leverage,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

CACHE = Path('data/cache_extended')
COST = 0.0004
FUNDING = 0.0001 / 32

# Top-12 symbols by Calmar (from slot analysis)
TOP12 = [
    'ARBUSDT', 'SOLUSDT', 'FILUSDT', 'ETHUSDT', 'XLMUSDT', 'ADAUSDT',
    'UNIUSDT', 'DOTUSDT', 'XRPUSDT', 'INJUSDT', 'NEARUSDT', 'AAVEUSDT',
]

# Production baseline params
BASE = dict(
    lookback=8, threshold=0.045, target_retrace=0.382, stop_mult=0.786,
    max_hold=10, stop_decay=0.0, btc_dir_period=0, btc_regime_period=96,
    asset_trend_period=192, min_confidence=2, base_leverage=1.5,
    conf_lev_step=0.0, max_leverage=1.5, rsi_period=14, vol_period=20,
    dynamic_target=False, target_c3=0.45, target_c4=0.5, target_c5=0.55,
    stoch_enabled=True, stoch_k=5, stoch_d=3,
    macd_enabled=True, macd_fast=12, macd_slow=26, macd_signal=14,
)


def load_data() -> dict:
    """Load all symbol data into memory."""
    btc_15m = pd.read_pickle(CACHE / 'BTCUSDT_15m_365d.pkl.gz')['Close'].values.astype(np.float64)
    data = {'btc_15m': btc_15m}
    
    for sym in TOP12:
        f15 = CACHE / f'{sym}_15m_365d.pkl.gz'
        f4h = CACHE / f'{sym}_4h_365d.pkl.gz'
        if not f15.exists():
            continue
        df = pd.read_pickle(f15)
        entry = {
            'close': df['Close'].values.astype(np.float64),
            'high': df['High'].values.astype(np.float64),
            'low': df['Low'].values.astype(np.float64),
            'volume': df['Volume'].values.astype(np.float64),
            'taker': df['Taker Buy Base'].values.astype(np.float64),
        }
        if f4h.exists():
            entry['close_4h'] = pd.read_pickle(f4h)['Close'].values.astype(np.float64)
        data[sym] = entry
    return data


def run_config(data: dict, params: dict, use_mtf: bool = True) -> dict:
    """Run backtest with given params across Top-12 symbols."""
    btc = data['btc_15m']
    results = []
    
    for sym in TOP12:
        if sym not in data:
            continue
        d = data[sym]
        close = d['close']
        n = len(close)
        btc_n = btc[:n] if len(btc) >= n else np.concatenate([btc, np.full(n - len(btc), btc[-1])])
        
        signals = compute_signals_v9(
            close, d['high'], d['low'], d['volume'], d['taker'], btc_n,
            params['lookback'], params['threshold'], params['target_retrace'],
            params['stop_mult'], params['max_hold'], params['stop_decay'],
            params['btc_dir_period'], params['btc_regime_period'],
            params['asset_trend_period'], params['min_confidence'],
            params['base_leverage'], params['conf_lev_step'],
            params['max_leverage'], params['rsi_period'], params['vol_period'],
            params['dynamic_target'], params['target_c3'], params['target_c4'],
            params['target_c5'], params['stoch_enabled'], params['stoch_k'],
            params['stoch_d'], params['macd_enabled'], params['macd_fast'],
            params['macd_slow'], params['macd_signal'],
        )
        
        if use_mtf and 'close_4h' in d:
            signals = apply_mtf_confirmation(signals, d['close_4h'], 12)
            signals = apply_mtf_leverage(signals, 2.0)
        
        r = _backtest_core(close, signals, COST, FUNDING)
        ret, dd, nt, wr = float(r[1]), float(r[3]), int(r[5]), float(r[4])
        cal = ret / dd if dd > 0 else 0.0
        results.append((sym, ret * 100, dd * 100, cal, nt, wr))
    
    total_ret = sum(r[1] for r in results)
    total_trades = sum(r[4] for r in results)
    avg_calmar = np.mean([r[3] for r in results]) if results else 0
    profitable = sum(1 for r in results if r[1] > 0)
    worst_dd = max(r[2] for r in results) if results else 0
    
    return {
        'total_ret': total_ret,
        'trades': total_trades,
        'avg_calmar': avg_calmar,
        'profitable': profitable,
        'worst_dd': worst_dd,
        'per_symbol': results,
    }


def print_result(name: str, r: dict, baseline: dict | None = None):
    """Print comparison result."""
    delta_ret = f" ({r['total_ret'] - baseline['total_ret']:+.1f}%)" if baseline else ""
    delta_cal = f" ({r['avg_calmar'] - baseline['avg_calmar']:+.2f})" if baseline else ""
    delta_t = f" ({r['trades'] - baseline['trades']:+d})" if baseline else ""
    
    print(f"  {name:<35} ret={r['total_ret']:>7.1f}%{delta_ret:<10} "
          f"trades={r['trades']:>4}{delta_t:<8} "
          f"avg_cal={r['avg_calmar']:>6.2f}{delta_cal:<8} "
          f"DD={r['worst_dd']:>5.1f}%  "
          f"prof={r['profitable']}/12")


def main():
    print("Loading data...")
    data = load_data()
    
    # ═══ BASELINE ═══
    print("\n" + "=" * 100)
    print("  v9.0 PRODUCTION BASELINE (Top-12 with MTF)")
    print("=" * 100)
    baseline = run_config(data, BASE)
    print_result("BASELINE (v9.0 production)", baseline)
    
    # ═══ H1: Lower Threshold ═══
    print("\n" + "=" * 100)
    print("  H1: THRESHOLD SWEEP (lower threshold → more entries)")
    print("=" * 100)
    for th in [0.040, 0.035, 0.030, 0.025, 0.020]:
        p = {**BASE, 'threshold': th}
        r = run_config(data, p)
        print_result(f"threshold={th:.3f}", r, baseline)
    
    # ═══ H2: Lower Min Confidence ═══
    print("\n" + "=" * 100)
    print("  H2: MIN CONFIDENCE SWEEP (lower bar → more entries, MTF as quality gate)")
    print("=" * 100)
    for mc in [1, 0]:
        p = {**BASE, 'min_confidence': mc}
        r = run_config(data, p)
        print_result(f"min_confidence={mc}", r, baseline)
    
    # ═══ H3: Shorter Lookback ═══
    print("\n" + "=" * 100)
    print("  H3: LOOKBACK SWEEP (shorter lookback → faster detection)")
    print("=" * 100)
    for lb in [6, 5, 4, 3]:
        p = {**BASE, 'lookback': lb}
        r = run_config(data, p)
        print_result(f"lookback={lb}", r, baseline)
    
    # ═══ H4: Combined — Lower threshold + lower confidence ═══
    print("\n" + "=" * 100)
    print("  H4: COMBINED — threshold + confidence (maximize entries)")
    print("=" * 100)
    for th, mc in [(0.035, 1), (0.030, 1), (0.035, 2), (0.030, 2), (0.025, 1)]:
        p = {**BASE, 'threshold': th, 'min_confidence': mc}
        r = run_config(data, p)
        print_result(f"th={th}, conf={mc}", r, baseline)
    
    # ═══ H5: Shorter lookback + lower threshold ═══
    print("\n" + "=" * 100)
    print("  H5: LOOKBACK + THRESHOLD COMBO")
    print("=" * 100)
    for lb, th in [(6, 0.035), (6, 0.030), (5, 0.035), (5, 0.030), (4, 0.030)]:
        p = {**BASE, 'lookback': lb, 'threshold': th}
        r = run_config(data, p)
        print_result(f"lb={lb}, th={th}", r, baseline)
    
    # ═══ H6: Max hold sweep ═══
    print("\n" + "=" * 100)
    print("  H6: MAX HOLD SWEEP (longer hold → capture more reversion)")
    print("=" * 100)
    for mh in [12, 15, 20, 8, 6]:
        p = {**BASE, 'max_hold': mh}
        r = run_config(data, p)
        print_result(f"max_hold={mh}", r, baseline)
    
    # ═══ H7: Target retrace sweep ═══
    print("\n" + "=" * 100)
    print("  H7: TARGET RETRACE SWEEP (wider target → more per trade)")
    print("=" * 100)
    for tr in [0.300, 0.350, 0.400, 0.450, 0.500, 0.618]:
        p = {**BASE, 'target_retrace': tr}
        r = run_config(data, p)
        print_result(f"target={tr:.3f}", r, baseline)
    
    # ═══ H8: MACD signal period ═══
    print("\n" + "=" * 100)
    print("  H8: MACD SIGNAL PERIOD (faster → more responsive)")
    print("=" * 100)
    for sp in [9, 10, 12, 16, 18]:
        p = {**BASE, 'macd_signal': sp}
        r = run_config(data, p)
        print_result(f"macd_signal={sp}", r, baseline)
    
    # ═══ H9: Stochastic period sweep ═══
    print("\n" + "=" * 100)
    print("  H9: STOCHASTIC %K PERIOD SWEEP")
    print("=" * 100)
    for sk in [3, 7, 9, 14]:
        p = {**BASE, 'stoch_k': sk}
        r = run_config(data, p)
        print_result(f"stoch_k={sk}", r, baseline)
    
    # ═══ H10: No stoch (MACD-only) vs no MACD (stoch-only) ═══
    print("\n" + "=" * 100)
    print("  H10: INDICATOR ABLATION (what's carrying the alpha?)")
    print("=" * 100)
    p_macd_only = {**BASE, 'stoch_enabled': False}
    r = run_config(data, p_macd_only)
    print_result("MACD only (no stoch)", r, baseline)
    
    p_stoch_only = {**BASE, 'macd_enabled': False}
    r = run_config(data, p_stoch_only)
    print_result("Stoch only (no MACD)", r, baseline)
    
    p_neither = {**BASE, 'macd_enabled': False, 'stoch_enabled': False}
    r = run_config(data, p_neither)
    print_result("Neither (v6 baseline)", r, baseline)
    
    # ═══ TOP COMBOS ═══
    print("\n" + "=" * 100)
    print("  BEST COMBINATIONS (cross the winners)")
    print("=" * 100)
    combos = [
        ('th=0.035 + lb=6', {'threshold': 0.035, 'lookback': 6}),
        ('th=0.035 + lb=6 + mh=12', {'threshold': 0.035, 'lookback': 6, 'max_hold': 12}),
        ('th=0.030 + conf=1', {'threshold': 0.030, 'min_confidence': 1}),
        ('th=0.035 + conf=1 + lb=6', {'threshold': 0.035, 'min_confidence': 1, 'lookback': 6}),
        ('th=0.030 + conf=1 + lb=6', {'threshold': 0.030, 'min_confidence': 1, 'lookback': 6}),
        ('th=0.030 + lb=5 + mh=12', {'threshold': 0.030, 'lookback': 5, 'max_hold': 12}),
        ('th=0.035 + target=0.350', {'threshold': 0.035, 'target_retrace': 0.350}),
        ('FULL AGGRO: th=0.030+lb=5+conf=1+mh=12', {
            'threshold': 0.030, 'lookback': 5, 'min_confidence': 1, 'max_hold': 12
        }),
    ]
    for name, overrides in combos:
        p = {**BASE, **overrides}
        r = run_config(data, p)
        print_result(name, r, baseline)


if __name__ == '__main__':
    main()
