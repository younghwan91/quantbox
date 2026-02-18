"""Leverage A/B test for adaptive 2.5x_f1 variant.
Tests 14 combinations of base_leverage x mtf_leverage_mult.
Walk-forward validated (IS=182d, OOS=183d).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    apply_mtf_confirmation, apply_mtf_leverage,
    RubberBandConfig,
)
from quantbox.backtest.vectorized_backtest import _backtest_core
from binance.client import Client

# Import from the sweep script (same directory)
from rb_adaptive_sweep import compute_signals_adaptive, download_15m, download_4h

settings = get_settings()
client = Client(
    settings.binance_api_key,
    settings.binance_api_secret,
)

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT',
]

DAYS = 365
COST = 0.0004
FUNDING = 0.01 / 100 / 32
BARS_PER_DAY = 96.0
IS_BARS = int(182 * BARS_PER_DAY)

# Adaptive threshold params (chosen variant: 2.5x_f1)
ATR_PERIOD = 48
ATR_MULTIPLIER = 2.5
THRESHOLD_FLOOR = 0.01
THRESHOLD_CAP = 0.06

# Build config from settings
cfg = RubberBandConfig()
for attr in [
    'lookback', 'target_retrace', 'stop_mult', 'max_hold', 'stop_decay',
    'btc_directional_period', 'btc_regime_period', 'asset_trend_period',
    'min_confidence', 'rsi_period', 'vol_period', 'stoch_enabled',
    'stoch_k_period', 'stoch_d_period', 'macd_enabled', 'macd_fast',
    'macd_slow', 'macd_signal_period', 'dynamic_target_enabled',
    'target_conf3', 'target_conf4', 'target_conf5',
]:
    setattr(cfg, attr, getattr(settings, f'rb_{attr}'))


# --- Load data ---
print("Loading data...")
btc_data = download_15m(client, 'BTCUSDT', DAYS)
btc_close = btc_data['close']

all_data: dict = {}
all_4h: dict = {}
for sym in SYMBOLS:
    d = download_15m(client, sym, DAYS)
    h4 = download_4h(client, sym, DAYS)
    if d:
        all_data[sym] = d
    if h4 is not None:
        all_4h[sym] = h4
print(f"Loaded {len(all_data)} symbols (15m) + {len(all_4h)} symbols (4h)")

# --- Leverage grid ---
LEV_GRID = [
    # (base_lev, conf_step, max_lev, mtf_mult, label)
    (1.0, 0.0, 1.0, 1.0, '1.0x_mtf1.0 (no lev)'),
    (1.0, 0.0, 1.0, 1.5, '1.0x_mtf1.5'),
    (1.0, 0.0, 1.0, 2.0, '1.0x_mtf2.0'),
    (1.0, 0.0, 1.0, 2.5, '1.0x_mtf2.5'),
    (1.0, 0.0, 1.0, 3.0, '1.0x_mtf3.0'),
    (1.5, 0.0, 1.5, 1.0, '1.5x_mtf1.0'),
    (1.5, 0.0, 1.5, 1.5, '1.5x_mtf1.5'),
    (1.5, 0.0, 1.5, 2.0, '1.5x_mtf2.0'),
    (1.5, 0.0, 1.5, 2.5, '1.5x_mtf2.5 (current)'),
    (1.5, 0.0, 1.5, 3.0, '1.5x_mtf3.0'),
    (2.0, 0.0, 2.0, 1.0, '2.0x_mtf1.0'),
    (2.0, 0.0, 2.0, 1.5, '2.0x_mtf1.5'),
    (2.0, 0.0, 2.0, 2.0, '2.0x_mtf2.0'),
    (2.0, 0.0, 2.0, 2.5, '2.0x_mtf2.5'),
]

print(f"\nLeverage A/B test for adaptive 2.5x_f1 (ATR mult={ATR_MULTIPLIER}, floor={THRESHOLD_FLOOR * 100}%)")
print(f"Symbols: {len(all_data)}, Period: {DAYS}d")
print(f"Walk-forward: IS={IS_BARS} bars | OOS=remainder\n")


def portfolio_metrics(res: dict) -> dict:
    """Compute portfolio-level metrics from per-symbol results."""
    ns = len(res)
    ml = min(len(r['equity']) for r in res.values())
    pe = np.ones(ml)
    for r in res.values():
        pe *= r['equity'][:ml] ** (1.0 / ns)
    ret = (pe[-1] - 1) * 100
    pk = np.maximum.accumulate(pe)
    dd = np.max((pk - pe) / pk) * 100
    rets = np.diff(pe) / pe[:-1]
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(BARS_PER_DAY * 365.25)
          if np.std(rets) > 0 else 0)
    cal = ret / dd if dd > 0 else 0
    trd = sum(r['trades'] for r in res.values())
    wr_avg = np.mean([r['wr'] for r in res.values()])
    sr = sum(r['ret'] for r in res.values())
    return {
        'ret': ret, 'sum_ret': sr, 'sharpe': sh, 'mdd': dd,
        'calmar': cal, 'trades': trd, 'wr': wr_avg,
    }


results = []
for base_lev, conf_step, max_lev, mtf_mult, label in LEV_GRID:
    res_full: dict = {}
    res_is: dict = {}
    res_oos: dict = {}

    for sym, data in all_data.items():
        c4h = all_4h.get(sym)
        c, h, l = data['close'], data['high'], data['low']
        v, t = data['volume'], data['taker_buy']
        n = min(len(c), len(btc_close))
        cc, hh, ll, vv, tt, bb = c[:n], h[:n], l[:n], v[:n], t[:n], btc_close[:n]

        sigs = compute_signals_adaptive(
            cc, hh, ll, vv, tt, bb,
            cfg.lookback, cfg.target_retrace, cfg.stop_mult, cfg.max_hold,
            cfg.stop_decay, cfg.btc_directional_period, cfg.btc_regime_period,
            cfg.asset_trend_period, cfg.min_confidence,
            base_lev, conf_step, max_lev,
            cfg.rsi_period, cfg.vol_period,
            cfg.dynamic_target_enabled, cfg.target_conf3, cfg.target_conf4, cfg.target_conf5,
            cfg.stoch_enabled, cfg.stoch_k_period, cfg.stoch_d_period,
            cfg.macd_enabled, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
            atr_period=ATR_PERIOD, atr_multiplier=ATR_MULTIPLIER,
            threshold_floor=THRESHOLD_FLOOR, threshold_cap=THRESHOLD_CAP,
        )

        if settings.rb_mtf_confirmation_enabled and c4h is not None:
            sigs = apply_mtf_confirmation(sigs, c4h, settings.rb_mtf_sma_period, ratio=16)
            sigs = apply_mtf_leverage(sigs, mtf_mult)

        # Full period
        eq, ret, sh, dd, wr, nt, *_ = _backtest_core(cc, sigs, COST, FUNDING, BARS_PER_DAY)
        res_full[sym] = {'ret': ret * 100, 'sharpe': sh, 'mdd': dd * 100, 'wr': wr, 'trades': int(nt), 'equity': eq}

        # In-sample
        split = min(IS_BARS, n)
        eq2, ret2, sh2, dd2, wr2, nt2, *_ = _backtest_core(cc[:split], sigs[:split], COST, FUNDING, BARS_PER_DAY)
        res_is[sym] = {'ret': ret2 * 100, 'sharpe': sh2, 'mdd': dd2 * 100, 'wr': wr2, 'trades': int(nt2), 'equity': eq2}

        # Out-of-sample
        eq3, ret3, sh3, dd3, wr3, nt3, *_ = _backtest_core(cc[split:], sigs[split:], COST, FUNDING, BARS_PER_DAY)
        res_oos[sym] = {'ret': ret3 * 100, 'sharpe': sh3, 'mdd': dd3 * 100, 'wr': wr3, 'trades': int(nt3), 'equity': eq3}

    pf = portfolio_metrics(res_full)
    pis = portfolio_metrics(res_is)
    poos = portfolio_metrics(res_oos)
    oos_ratio = poos['sharpe'] / pis['sharpe'] if pis['sharpe'] > 0 else 0
    eff_lev = base_lev * mtf_mult
    results.append((label, eff_lev, pf, pis, poos, oos_ratio))
    print(f"  {label:<28} eff={eff_lev:.1f}x  SumRet={pf['sum_ret']:+9.0f}%"
          f"  Shrp={pf['sharpe']:5.2f}  MDD={pf['mdd']:5.1f}%"
          f"  OOS/IS={oos_ratio:.2f}  Trd={pf['trades']}")

# --- Summary table ---
print()
print('=' * 120)
print('  LEVERAGE A/B TEST -- Adaptive 2.5x_f1, 365d, 20 symbols')
print('=' * 120)
header = (f"  {'Config':<28} {'Eff.Lev':>7} {'SumRet%':>9} {'GeoRet%':>8}"
          f" {'Sharpe':>7} {'MDD%':>6} {'Calmar':>7} {'OOS/IS':>7}"
          f" {'Trades':>7} {'WR%':>6}")
print(header)
print(f"  {'-' * 28} {'-' * 7} {'-' * 9} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6}")
for label, eff, pf, pis, poos, oos_r in results:
    marker = ' <--' if '(current)' in label else ''
    print(f"  {label:<28} {eff:7.1f} {pf['sum_ret']:+9.0f} {pf['ret']:+8.1f}"
          f" {pf['sharpe']:7.2f} {pf['mdd']:6.1f} {pf['calmar']:7.2f}"
          f" {oos_r:7.2f} {pf['trades']:7d} {pf['wr'] * 100:5.1f}{marker}")

best_sharpe = max(results, key=lambda x: x[2]['sharpe'])
best_oos = max(results, key=lambda x: x[4]['sharpe'])
best_calmar = max(results, key=lambda x: x[2]['calmar'])
print(f"\n  Best Full Sharpe:  {best_sharpe[0]} (Sharpe={best_sharpe[2]['sharpe']:.2f})")
print(f"  Best OOS Sharpe:   {best_oos[0]} (OOS Sharpe={best_oos[4]['sharpe']:.2f})")
print(f"  Best Calmar:       {best_calmar[0]} (Calmar={best_calmar[2]['calmar']:.2f})")
print('=' * 120)

# --- Sharpe-to-MDD efficiency ---
print("\n  Sharpe/MDD Efficiency (higher = better risk-adjusted):")
for label, eff, pf, *_ in results:
    ratio = pf['sharpe'] / pf['mdd'] if pf['mdd'] > 0 else 0
    marker = ' <--' if '(current)' in label else ''
    print(f"    {label:<28} Sharpe/MDD={ratio:.3f}{marker}")
