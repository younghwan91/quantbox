#!/usr/bin/env python3
"""Diagnostic: compare portfolio return computation methods."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig, compute_signals_v9,
    apply_mtf_confirmation, apply_mtf_leverage,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

settings = get_settings()
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'
COST = 0.0004
FUNDING = 0.01 / 100 / 32
BARS_PER_DAY = 96.0

symbols = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT',
]

cfg = RubberBandConfig()
cfg.lookback = settings.rb_lookback
cfg.threshold = settings.rb_threshold
cfg.target_retrace = settings.rb_target_retrace
cfg.stop_mult = settings.rb_stop_mult
cfg.max_hold = settings.rb_max_hold
cfg.stop_decay = settings.rb_stop_decay
cfg.btc_directional_period = settings.rb_btc_directional_period
cfg.btc_regime_period = settings.rb_btc_regime_period
cfg.asset_trend_period = settings.rb_asset_trend_period
cfg.min_confidence = settings.rb_min_confidence
cfg.rsi_period = settings.rb_rsi_period
cfg.vol_period = settings.rb_vol_period
cfg.base_leverage = settings.rb_base_leverage
cfg.confidence_leverage_step = settings.rb_confidence_leverage_step
cfg.max_leverage = settings.rb_max_leverage
cfg.stoch_enabled = settings.rb_stoch_enabled
cfg.stoch_k_period = settings.rb_stoch_k_period
cfg.stoch_d_period = settings.rb_stoch_d_period
cfg.macd_enabled = settings.rb_macd_enabled
cfg.macd_fast = settings.rb_macd_fast
cfg.macd_slow = settings.rb_macd_slow
cfg.macd_signal_period = settings.rb_macd_signal_period
cfg.dynamic_target_enabled = settings.rb_dynamic_target_enabled
cfg.target_conf3 = settings.rb_target_conf3
cfg.target_conf4 = settings.rb_target_conf4
cfg.target_conf5 = settings.rb_target_conf5

mtf_enabled = settings.rb_mtf_confirmation_enabled
mtf_sma = settings.rb_mtf_sma_period
mtf_lev = settings.rb_mtf_leverage_mult

print(f'Settings: threshold={cfg.threshold}, lookback={cfg.lookback}, max_hold={cfg.max_hold}')
print(f'Leverage: base={cfg.base_leverage}, max={cfg.max_leverage}, mtf_mult={mtf_lev}')
print(f'MTF: enabled={mtf_enabled}, sma={mtf_sma}')
print()

# Load BTC data
btc_d = np.load(CACHE_DIR / 'BTCUSDT_15m_365d_20260218.npz')
btc_close = btc_d['close']
btc_4h_d = np.load(CACHE_DIR / 'BTCUSDT_4h_365d_20260218.npz')
btc_4h = btc_4h_d['close']

print(f'BTC: {len(btc_close)} bars (15m), {len(btc_4h)} bars (4h)')
print(f'BTC price range: {btc_close[0]:.0f} → {btc_close[-1]:.0f} ({(btc_close[-1]/btc_close[0]-1)*100:+.1f}%)')
print()

results = {}
for sym in symbols:
    f15 = CACHE_DIR / f'{sym}_15m_365d_20260218.npz'
    f4h = CACHE_DIR / f'{sym}_4h_365d_20260218.npz'
    if not f15.exists() or not f4h.exists():
        print(f'SKIP {sym}: missing cache')
        continue
    d = np.load(f15)
    c, h, l, v, t = d['close'], d['high'], d['low'], d['volume'], d['taker_buy']
    d4h = np.load(f4h)
    close_4h = d4h['close']

    n = min(len(c), len(btc_close))
    c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc_close[:n]

    sigs = compute_signals_v9(
        c, h, l, v, t, b,
        cfg.lookback, cfg.threshold, cfg.target_retrace, cfg.stop_mult,
        cfg.max_hold, cfg.stop_decay, cfg.btc_directional_period,
        cfg.btc_regime_period, cfg.asset_trend_period,
        cfg.min_confidence, cfg.base_leverage,
        cfg.confidence_leverage_step, cfg.max_leverage,
        cfg.rsi_period, cfg.vol_period,
        cfg.dynamic_target_enabled,
        cfg.target_conf3, cfg.target_conf4, cfg.target_conf5,
        cfg.stoch_enabled, cfg.stoch_k_period, cfg.stoch_d_period,
        cfg.macd_enabled, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
    )

    n_raw = int(np.sum(sigs != 0))

    if mtf_enabled and close_4h is not None:
        sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
        n_post_mtf = int(np.sum(sigs != 0))
        sigs = apply_mtf_leverage(sigs, mtf_lev)
    else:
        n_post_mtf = n_raw

    eq, total_ret, sharpe, max_dd, wr, n_trades, bars_in, pf, avg_ret, _ = \
        _backtest_core(c, sigs, COST, FUNDING, BARS_PER_DAY)

    # Also compute max signal magnitude
    max_sig = float(np.max(np.abs(sigs)))

    results[sym] = {
        'ret': total_ret * 100,
        'eq_final': eq[-1],
        'trades': int(n_trades),
        'wr': wr,
        'mdd': max_dd * 100,
        'equity': eq,
        'n_raw': n_raw,
        'n_post_mtf': n_post_mtf,
        'max_sig': max_sig,
        'price_change': (c[-1] / c[0] - 1) * 100,
    }

print(f'Loaded {len(results)} symbols')
print()

# Per-symbol results
print(f'{"Symbol":<16} {"Ret%":>8} {"EqFinal":>10} {"Trades":>6} {"WR%":>6} {"MDD%":>6} {"RawSig":>7} {"MTFSig":>7} {"MaxLev":>7} {"PxChg%":>7}')
print('-' * 100)
per_sym_rets = []
per_sym_eqs = []
for sym in sorted(results.keys()):
    r = results[sym]
    per_sym_rets.append(r['ret'])
    per_sym_eqs.append(r['eq_final'])
    print(f'{sym:<16} {r["ret"]:+8.1f} {r["eq_final"]:10.4f} {r["trades"]:6d} {r["wr"]:6.1f} {r["mdd"]:6.1f} {r["n_raw"]:7d} {r["n_post_mtf"]:7d} {r["max_sig"]:7.2f} {r["price_change"]:+7.1f}')

print()
n_syms = len(results)

# Method 1: Geometric mean (what A/B test & run_portfolio use)
min_len = min(len(r['equity']) for r in results.values())
port_eq_geo = np.ones(min_len)
for r in results.values():
    port_eq_geo *= r['equity'][:min_len] ** (1.0 / n_syms)
geo_ret = (port_eq_geo[-1] - 1.0) * 100

# Method 2: Arithmetic mean of equity curves
port_eq_arith = np.zeros(min_len)
for r in results.values():
    port_eq_arith += r['equity'][:min_len] / n_syms
arith_eq_ret = (port_eq_arith[-1] - 1.0) * 100

# Method 3: Arithmetic sum of per-symbol returns
sum_ret = sum(per_sym_rets)

# Method 4: Arithmetic mean of per-symbol returns
mean_ret = float(np.mean(per_sym_rets))

# Method 5: Geometric mean of final equities
geo_mean_eq = float(np.prod(np.array(per_sym_eqs)) ** (1.0 / n_syms))
geo_mean_ret = (geo_mean_eq - 1.0) * 100

print(f'=== Portfolio Return Comparison ({n_syms} symbols) ===')
print(f'Method 1 - Geometric portfolio equity (A/B test):       {geo_ret:+.1f}%')
print(f'Method 2 - Arithmetic mean of equity curves:             {arith_eq_ret:+.1f}%')
print(f'Method 3 - Arithmetic SUM of per-symbol returns:         {sum_ret:+.1f}%')
print(f'Method 4 - Arithmetic MEAN of per-symbol returns:        {mean_ret:+.1f}%')
print(f'Method 5 - Geometric mean of final equities (endpoint):  {geo_mean_ret:+.1f}%')
print()
if geo_ret != 0:
    print(f'Ratio (SUM / geometric):     {sum_ret/geo_ret:.1f}x')
    print(f'Ratio (arith-eq / geometric): {arith_eq_ret/geo_ret:.1f}x')
print()

# Summary stats
print(f'Per-symbol return stats:')
print(f'  Min:    {min(per_sym_rets):+.1f}%')
print(f'  Max:    {max(per_sym_rets):+.1f}%')
print(f'  Median: {float(np.median(per_sym_rets)):+.1f}%')
print(f'  Std:    {float(np.std(per_sym_rets)):.1f}%')
print(f'  +ive:   {sum(1 for r in per_sym_rets if r > 0)}/{n_syms}')
total_trades = sum(r['trades'] for r in results.values())
print(f'  Total trades: {total_trades}')
