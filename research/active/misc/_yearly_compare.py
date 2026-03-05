#!/usr/bin/env python3
"""Year-by-year comparison: v12.5.3 intra-bar vs close-based."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

from research.active.misc.v1253_intrabar_6yr_backtest import (
    load_15m, load_4h, scan_entries_risk_off_only, apply_mtf_filter,
    compute_atr_pct, simulate_portfolio, simulate_portfolio_close_based,
    LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
    ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
    STOCH_ENABLED, STOCH_K, STOCH_D, MACD_ENABLED, MACD_FAST, MACD_SLOW,
    MACD_SIGNAL, ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP, MTF_SMA_PERIOD,
)

SYMBOLS_50 = [
    'BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT',
    'ADAUSDT','AVAXUSDT','LINKUSDT','XLMUSDT','DOTUSDT','1000PEPEUSDT',
    'ARBUSDT','FETUSDT','1000SHIBUSDT','OPUSDT','ATOMUSDT','HBARUSDT',
    'ICPUSDT','FILUSDT','TRXUSDT','SUIUSDT','TONUSDT','LTCUSDT',
    'BCHUSDT','UNIUSDT','NEARUSDT','APTUSDT','AAVEUSDT','INJUSDT',
    'ETCUSDT','GRTUSDT','STXUSDT','EGLDUSDT','AXSUSDT','SANDUSDT',
    'THETAUSDT','ALGOUSDT','RUNEUSDT','DYDXUSDT','CRVUSDT','LDOUSDT',
    'COMPUSDT','SNXUSDT','VETUSDT','APEUSDT','ENAUSDT','TAOUSDT','WLDUSDT','WIFUSDT',
]

btc_data = load_15m('BTCUSDT')
btc_close = btc_data['close']

all_data, all_4h = {}, {}
for sym in SYMBOLS_50:
    d = load_15m(sym)
    if d is None: continue
    all_data[sym] = d
    h4 = load_4h(sym)
    if h4 is not None: all_4h[sym] = h4

available = [s for s in SYMBOLS_50 if s in all_data]
sym_dirs, sym_confs, sym_moves = {}, {}, {}
sym_close_d, sym_high_d, sym_low_d, sym_atr_d = {}, {}, {}, {}

for sym in available:
    d = all_data[sym]
    n = min(len(d['close']), len(btc_close))
    c, h, l, v, t = d['close'][:n], d['high'][:n], d['low'][:n], d['volume'][:n], d['taker_buy'][:n]
    b = btc_close[:n]
    dirs, confs, moves = scan_entries_risk_off_only(
        c, h, l, v, t, b, LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
        ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
        STOCH_ENABLED, STOCH_K, STOCH_D, MACD_ENABLED, MACD_FAST, MACD_SLOW,
        MACD_SIGNAL, ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP)
    c4h = all_4h.get(sym)
    if c4h is not None:
        dirs, confs, moves = apply_mtf_filter(dirs, confs, moves, c4h, MTF_SMA_PERIOD, ratio=16)
    sym_dirs[sym] = dirs; sym_confs[sym] = confs; sym_moves[sym] = moves
    sym_close_d[sym] = c; sym_high_d[sym] = h; sym_low_d[sym] = l
    atr = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)
    sym_atr_d[sym] = atr

max_n = max(len(sym_close_d[s]) for s in sym_close_d)
for sym in list(sym_close_d.keys()):
    for d_ref, fill in [
        (sym_close_d, 'first'), (sym_high_d, 'first'), (sym_low_d, 'first'),
        (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
        (sym_atr_d, 'nan'),
    ]:
        arr = d_ref[sym]
        if len(arr) < max_n:
            pad_n = max_n - len(arr)
            fv = arr[0] if fill == 'first' else (np.nan if fill == 'nan' else 0.0)
            d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

r_ib = simulate_portfolio(sym_close_d, sym_high_d, sym_low_d, sym_dirs, sym_confs, sym_moves, sym_atr=sym_atr_d)
r_cb = simulate_portfolio_close_based(sym_close_d, sym_high_d, sym_low_d, sym_dirs, sym_confs, sym_moves, sym_atr=sym_atr_d)

eq_ib = r_ib['equity_curve']
eq_cb = r_cb['equity_curve']
bars_per_year = 96 * 365

years = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

print(f"\n{'='*80}")
print("  YEAR-BY-YEAR: v12.5.3 Intra-Bar vs v12.4.1 Close-Based")
print(f"{'='*80}")
print(f"  {'Year':<12} {'IB Return':>10} {'IB DD':>8}  {'CB Return':>10} {'CB DD':>8}  {'Δ Return':>10}")
print(f"  {'-'*68}")

for y in range(min(6, len(eq_ib) // bars_per_year)):
    s = y * bars_per_year
    e = min((y + 1) * bars_per_year, len(eq_ib)) - 1

    ib_ret = (eq_ib[e] / eq_ib[s] - 1) * 100
    cb_ret = (eq_cb[e] / eq_cb[s] - 1) * 100

    ib_pk, ib_dd = eq_ib[s], 0.0
    cb_pk, cb_dd = eq_cb[s], 0.0
    for j in range(s, e + 1):
        if eq_ib[j] > ib_pk: ib_pk = eq_ib[j]
        d = (ib_pk - eq_ib[j]) / ib_pk
        if d > ib_dd: ib_dd = d
        if eq_cb[j] > cb_pk: cb_pk = eq_cb[j]
        d = (cb_pk - eq_cb[j]) / cb_pk
        if d > cb_dd: cb_dd = d

    print(f"  {years[y]:<12} {ib_ret:>+9.1f}% {ib_dd*100:>7.1f}%  {cb_ret:>+9.1f}% {cb_dd*100:>7.1f}%  {ib_ret-cb_ret:>+9.1f}%")

# Totals
ib_total = (eq_ib[-1] / eq_ib[0] - 1) * 100
cb_total = (eq_cb[-1] / eq_cb[0] - 1) * 100
print(f"  {'-'*68}")
print(f"  {'TOTAL':<12} {ib_total:>+9.1f}% {r_ib['max_dd_pct']:>7.1f}%  {cb_total:>+9.1f}% {r_cb['max_dd_pct']:>7.1f}%  {ib_total-cb_total:>+9.1f}%")

print(f"\n{'='*80}")
print("  EXIT REASON BREAKDOWN")
print(f"{'='*80}")
print(f"  {'Reason':<16} {'v12.5.3 IB':>12} {'v12.4.1 CB':>12} {'Δ':>10}")
print(f"  {'-'*52}")
all_reasons = sorted(set(list(r_ib['exit_reasons'].keys()) + list(r_cb['exit_reasons'].keys())))
for r in all_reasons:
    ib_n = r_ib['exit_reasons'].get(r, 0)
    cb_n = r_cb['exit_reasons'].get(r, 0)
    print(f"  {r:<16} {ib_n:>12,} {cb_n:>12,} {ib_n-cb_n:>+10,}")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<16} {r_ib['n_trades']:>12,} {r_cb['n_trades']:>12,} {r_ib['n_trades']-r_cb['n_trades']:>+10,}")
