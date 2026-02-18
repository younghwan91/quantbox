#!/usr/bin/env python3
"""
Adaptive Threshold Sweep — Walk-Forward Validated
==================================================
Grid: 3 multipliers × 6 floors = 18 variants
Walk-forward: first 182d (IS) vs last 183d (OOS) → detect overfitting
Last-30d signal count: would it actually trade in the current regime?

Reuses cached data from rb_adaptive_ab_test.py (data/cache_extended/*.npz).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import time
from datetime import datetime, timedelta, timezone
import numpy as np
import numba
from binance.client import Client
from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_v9,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    _compute_sma_inline,
    _compute_rsi_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

# ═══════════════════════════════════════════════════════════════════════
#  ATR Computation (Numba)
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def compute_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int) -> np.ndarray:
    """ATR as percentage of close price (EMA-smoothed)."""
    n = len(close)
    atr_pct = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    if n < period:
        return atr_pct
    atr = 0.0
    for i in range(period):
        atr += tr[i]
    atr /= period
    if close[period - 1] > 0:
        atr_pct[period - 1] = atr / close[period - 1]
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr = alpha * tr[i] + (1 - alpha) * atr
        if close[i] > 0:
            atr_pct[i] = atr / close[i]
    return atr_pct


# ═══════════════════════════════════════════════════════════════════════
#  Volatility-Adaptive Signal Generator
# ═══════════════════════════════════════════════════════════════════════

def compute_signals_adaptive(
    close, high, low, volume, taker_buy, btc_close,
    lookback, target_retrace, stop_mult, max_hold, stop_decay,
    btc_mom_period, btc_regime_period, asset_trend_period,
    min_confidence, base_leverage, confidence_leverage_step, max_leverage,
    rsi_period, vol_period,
    dynamic_target_enabled, target_conf3, target_conf4, target_conf5,
    stoch_enabled, stoch_k_period, stoch_d_period,
    macd_enabled, macd_fast, macd_slow, macd_signal_period,
    atr_period=48, atr_multiplier=2.0,
    threshold_floor=0.015, threshold_cap=0.06,
):
    """Volatility-adaptive compute_signals_v9 — per-bar threshold."""
    n = len(close)
    sig = np.zeros(n)

    atr_pct = compute_atr_pct(high, low, close, atr_period)
    btc_sma = _compute_sma_inline(btc_close, btc_regime_period)
    asset_sma = _compute_sma_inline(close, asset_trend_period)
    rsi = _compute_rsi_inline(close, rsi_period)
    vol_sma = _compute_sma_inline(volume, vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    if stoch_enabled:
        stoch_k, stoch_d = _compute_stochastic_inline(
            high, low, close, stoch_k_period, stoch_d_period)
    else:
        stoch_k = np.full(n, 50.0)
        stoch_d = np.full(n, 50.0)

    if macd_enabled:
        _, _, macd_hist = _compute_macd_inline(
            close, macd_fast, macd_slow, macd_signal_period)
    else:
        macd_hist = np.full(n, 0.0)

    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    lev = base_leverage
    trade_target_retrace = target_retrace
    start_bar = max(lookback, btc_mom_period, btc_regime_period,
                    asset_trend_period, rsi_period, vol_period, atr_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        atr_val = atr_pct[i]
        if np.isnan(atr_val):
            threshold = 0.032
        else:
            threshold = min(threshold_cap, max(threshold_floor, atr_multiplier * atr_val))

        if position == 0:
            if abs(ret) > threshold:
                if np.isnan(btc_sma[i]):
                    continue
                is_risk_on = btc_close[i] > btc_sma[i]
                btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
                entered = False
                confidence = 1

                if is_risk_on:
                    if np.isnan(asset_sma[i]):
                        continue
                    is_uptrend = close[i] > asset_sma[i]

                    if ret < -threshold and is_uptrend:
                        if rsi[i] < 40:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.45:
                            confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02:
                            confidence += 1
                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i - 1] <= stoch_d[i - 1]:
                                confidence += 1
                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > threshold and not is_uptrend:
                        if rsi[i] > 60:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.55:
                            confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02:
                            confidence += 1
                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i - 1] >= stoch_d[i - 1]:
                                confidence += 1
                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = -1
                            entered = True
                else:
                    if ret < -threshold and btc_mom < 0:
                        if rsi[i] < 35:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] < 0.42:
                            confidence += 1
                        if btc_mom < -0.01:
                            confidence += 1
                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i - 1] <= stoch_d[i - 1]:
                                confidence += 1
                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = 1
                            entered = True

                    elif ret > threshold and btc_mom > 0:
                        if rsi[i] > 65:
                            confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                            confidence += 1
                        if taker_ratio[i] > 0.58:
                            confidence += 1
                        if btc_mom > 0.01:
                            confidence += 1
                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i - 1] >= stoch_d[i - 1]:
                                confidence += 1
                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                                confidence += 1
                        if confidence >= min_confidence:
                            position = -1
                            entered = True

                if entered:
                    entry_price = close[i]
                    move_size = abs(ret)
                    hold_count = 0
                    best_pnl = 0.0
                    lev = min(base_leverage + confidence_leverage_step * (confidence - 1),
                              max_leverage)
                    if dynamic_target_enabled:
                        if confidence >= 5:
                            trade_target_retrace = target_conf5
                        elif confidence >= 4:
                            trade_target_retrace = target_conf4
                        elif confidence >= 3:
                            trade_target_retrace = target_conf3
                        else:
                            trade_target_retrace = target_retrace
                    else:
                        trade_target_retrace = target_retrace
                    sig[i] = position * lev
                    continue

        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price

            if gain > best_pnl:
                best_pnl = gain

            time_factor = 1.0 - stop_decay * (hold_count / max_hold)
            effective_stop = move_size * stop_mult * time_factor
            target = move_size * trade_target_retrace

            if gain >= target:
                position = 0
            elif gain < -effective_stop:
                position = 0
            elif best_pnl > target * 0.5 and gain < best_pnl * 0.3:
                position = 0
            elif hold_count >= max_hold:
                position = 0

            if position == 0:
                sig[i] = 0
                continue

        if position != 0:
            sig[i] = position * lev

    return sig


# ═══════════════════════════════════════════════════════════════════════
#  Data Download (reuses cache from rb_adaptive_ab_test.py)
# ═══════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, interval: str, days: int) -> Path:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return CACHE_DIR / f'{symbol}_{interval}_{days}d_{today}.npz'


def download_15m(client, symbol, days):
    cache = _cache_path(symbol, '15m', days)
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines = []
    cur = start_ms
    while cur < end_ms:
        klines = client.futures_klines(
            symbol=symbol, interval='15m',
            startTime=cur, endTime=end_ms, limit=1500)
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        time.sleep(0.35)

    if len(all_klines) < 200:
        return None
    all_klines = all_klines[:-1]

    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in all_klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in all_klines], dtype=np.float64)
    volume = np.array([float(k[5]) for k in all_klines], dtype=np.float64)
    taker_buy = np.array([float(k[9]) for k in all_klines], dtype=np.float64)

    np.savez_compressed(cache, close=close, high=high, low=low,
                        volume=volume, taker_buy=taker_buy)
    return {'close': close, 'high': high, 'low': low,
            'volume': volume, 'taker_buy': taker_buy}


def download_4h(client, symbol, days):
    cache = _cache_path(symbol, '4h', days)
    if cache.exists():
        d = np.load(cache)
        return d['close']

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines = []
    cur = start_ms
    while cur < end_ms:
        klines = client.futures_klines(
            symbol=symbol, interval='4h',
            startTime=cur, endTime=end_ms, limit=1500)
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        time.sleep(0.35)

    if len(all_klines) < 12:
        return None
    all_klines = all_klines[:-1]
    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    np.savez_compressed(cache, close=close)
    return close


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    settings = get_settings()
    client = Client(settings.binance_api_key, settings.binance_api_secret)
    print("Connected to Binance\n")

    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
    ]

    DAYS = 365
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32
    BARS_PER_DAY = 96.0

    # Walk-forward split: first 182 days IS, last 183 days OOS
    IS_BARS = int(182 * BARS_PER_DAY)   # ~17,472
    LAST_30D_BARS = int(30 * BARS_PER_DAY)  # 2,880

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

    print(f"Config: threshold={cfg.threshold}, lookback={cfg.lookback}, "
          f"max_hold={cfg.max_hold}")
    print(f"MTF: enabled={mtf_enabled}, sma={mtf_sma}, lev_mult={mtf_lev}")
    print(f"Walk-forward split: IS={IS_BARS} bars (182d) | OOS=remainder (183d)")
    print()

    # ── Download data ──
    print(f"Loading {DAYS}-day data (cached)...")
    btc_data = download_15m(client, 'BTCUSDT', DAYS)
    btc_close = btc_data['close']
    btc_4h = download_4h(client, 'BTCUSDT', DAYS)
    print(f"  BTC: {len(btc_close)} bars (15m), {len(btc_4h)} bars (4h)")

    all_data = {}
    all_4h = {}
    for sym in symbols:
        d = download_15m(client, sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        h4 = download_4h(client, sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4
        print(f"  {sym}: {len(d['close'])} (15m), {len(h4) if h4 is not None else 0} (4h)")

    print(f"\nLoaded {len(all_data)} symbols\n")

    # ── JIT warmup ──
    print("Warming up Numba JIT...")
    dummy = np.random.randn(300).cumsum() + 100.0
    dummy_h = dummy + 0.5
    dummy_l = dummy - 0.5
    compute_atr_pct(dummy_h, dummy_l, dummy, 14)
    _backtest_core(dummy, np.zeros(300), COST, 0.0, BARS_PER_DAY)
    print("JIT compiled\n")

    # ═══════════════════════════════════════════════════════════════════
    #  Parameter Grid
    # ═══════════════════════════════════════════════════════════════════
    #  multiplier × floor = 18 variants
    #  + fixed baseline
    MULTIPLIERS = [2.0, 2.5, 3.0]
    FLOORS = [0.010, 0.012, 0.015, 0.018, 0.020, 0.025]
    ATR_PERIOD = 48
    CAP = 0.06

    grid = []
    for mult in MULTIPLIERS:
        for floor_val in FLOORS:
            name = f"{mult}x_f{floor_val*100:.0f}" if floor_val * 100 == int(floor_val * 100) \
                else f"{mult}x_f{floor_val*1000:.0f}"
            # Clean naming
            fp = floor_val * 100
            if fp == int(fp):
                fname = f"{int(fp)}"
            else:
                fname = f"{fp:.1f}"
            name = f"{mult}x_f{fname}"
            grid.append({
                'name': name, 'atr_period': ATR_PERIOD, 'atr_mult': mult,
                'floor': floor_val, 'cap': CAP,
            })

    print(f"Testing {len(grid)} adaptive variants + fixed baseline")
    print(f"Multipliers: {MULTIPLIERS}")
    print(f"Floors: {[f'{f*100:.1f}%' for f in FLOORS]}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    #  Signal + Backtest helpers
    # ═══════════════════════════════════════════════════════════════════

    def _gen_fixed(data, btc, close_4h):
        c, h, l, v, t = data['close'], data['high'], data['low'], data['volume'], data['taker_buy']
        n = min(len(c), len(btc))
        c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]
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
        if mtf_enabled and close_4h is not None:
            sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
            sigs = apply_mtf_leverage(sigs, mtf_lev)
        return c, sigs

    def _gen_adaptive(data, btc, close_4h, acfg):
        c, h, l, v, t = data['close'], data['high'], data['low'], data['volume'], data['taker_buy']
        n = min(len(c), len(btc))
        c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]
        sigs = compute_signals_adaptive(
            c, h, l, v, t, b,
            cfg.lookback, cfg.target_retrace, cfg.stop_mult,
            cfg.max_hold, cfg.stop_decay, cfg.btc_directional_period,
            cfg.btc_regime_period, cfg.asset_trend_period,
            cfg.min_confidence, cfg.base_leverage,
            cfg.confidence_leverage_step, cfg.max_leverage,
            cfg.rsi_period, cfg.vol_period,
            cfg.dynamic_target_enabled,
            cfg.target_conf3, cfg.target_conf4, cfg.target_conf5,
            cfg.stoch_enabled, cfg.stoch_k_period, cfg.stoch_d_period,
            cfg.macd_enabled, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
            atr_period=acfg['atr_period'], atr_multiplier=acfg['atr_mult'],
            threshold_floor=acfg['floor'], threshold_cap=acfg['cap'],
        )
        if mtf_enabled and close_4h is not None:
            sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
            sigs = apply_mtf_leverage(sigs, mtf_lev)
        return c, sigs

    def _bt(close, signals):
        eq, total_ret, sharpe, max_dd, wr, n_trades, *_ = (
            _backtest_core(close, signals, COST, FUNDING, BARS_PER_DAY))
        calmar = (total_ret * 100 / (max_dd * 100)) if max_dd > 0 else 0.0
        return {
            'ret': total_ret * 100, 'sharpe': sharpe,
            'mdd': max_dd * 100, 'calmar': calmar,
            'wr': wr, 'trades': int(n_trades), 'equity': eq,
        }

    def _portfolio(results):
        """Compute portfolio ΣRet, GeoRet, Sharpe, MDD."""
        n_syms = len(results)
        if n_syms == 0:
            return {'ret': 0, 'sum_ret': 0, 'sharpe': 0, 'mdd': 0,
                    'calmar': 0, 'trades': 0, 'wr': 0}
        min_len = min(len(r['equity']) for r in results.values())
        port_eq = np.ones(min_len)
        for r in results.values():
            port_eq *= r['equity'][:min_len] ** (1.0 / n_syms)
        ret = (port_eq[-1] - 1.0) * 100
        peak = np.maximum.accumulate(port_eq)
        dd = np.max((peak - port_eq) / peak) * 100
        rets = np.diff(port_eq) / port_eq[:-1]
        sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(BARS_PER_DAY * 365.25)
                  if np.std(rets) > 0 else 0.0)
        calmar = ret / dd if dd > 0 else 0.0
        trades = sum(r['trades'] for r in results.values())
        wr = np.mean([r['wr'] for r in results.values()])
        sum_ret = sum(r['ret'] for r in results.values())
        return {'ret': ret, 'sum_ret': sum_ret, 'sharpe': sharpe, 'mdd': dd,
                'calmar': calmar, 'trades': trades, 'wr': wr}

    def _count_last_30d_signals(close, signals):
        """Count entry signals in the last 30 days of data."""
        n = len(signals)
        start = max(0, n - LAST_30D_BARS)
        count = 0
        in_pos = False
        for i in range(start, n):
            if not in_pos and signals[i] != 0:
                count += 1
                in_pos = True
            elif in_pos and signals[i] == 0:
                in_pos = False
        return count

    # ═══════════════════════════════════════════════════════════════════
    #  Run everything: full period + walk-forward IS/OOS
    # ═══════════════════════════════════════════════════════════════════

    all_results = []  # list of dicts

    # ── Fixed baseline ──
    print("Running FIXED 3.2% baseline... ", end='', flush=True)
    res_full = {}
    res_is = {}
    res_oos = {}
    last30_trades = 0
    for sym, data in all_data.items():
        c4h = all_4h.get(sym)
        c, sigs = _gen_fixed(data, btc_close, c4h)
        # Full period
        res_full[sym] = _bt(c, sigs)
        # Walk-forward: IS (first 182d)
        split = min(IS_BARS, len(c))
        res_is[sym] = _bt(c[:split], sigs[:split])
        # Walk-forward: OOS (last 183d)
        res_oos[sym] = _bt(c[split:], sigs[split:])
        # Last 30d trades
        last30_trades += _count_last_30d_signals(c, sigs)

    pf = _portfolio(res_full)
    pis = _portfolio(res_is)
    poos = _portfolio(res_oos)
    oos_ratio = poos['sharpe'] / pis['sharpe'] if pis['sharpe'] > 0 else 0.0
    all_results.append({
        'name': 'Fixed 3.2%',
        'full': pf, 'is': pis, 'oos': poos,
        'oos_ratio': oos_ratio,
        'last30': last30_trades,
    })
    print(f"ΣRet={pf['sum_ret']:+.0f}% | OOS/IS={oos_ratio:.2f} | L30d={last30_trades}")

    # ── Adaptive grid ──
    for idx, acfg in enumerate(grid):
        name = acfg['name']
        print(f"Running [{idx+1}/{len(grid)}] {name}... ", end='', flush=True)
        res_full = {}
        res_is = {}
        res_oos = {}
        last30_trades = 0
        for sym, data in all_data.items():
            c4h = all_4h.get(sym)
            c, sigs = _gen_adaptive(data, btc_close, c4h, acfg)
            res_full[sym] = _bt(c, sigs)
            split = min(IS_BARS, len(c))
            res_is[sym] = _bt(c[:split], sigs[:split])
            res_oos[sym] = _bt(c[split:], sigs[split:])
            last30_trades += _count_last_30d_signals(c, sigs)

        pf = _portfolio(res_full)
        pis = _portfolio(res_is)
        poos = _portfolio(res_oos)
        oos_ratio = poos['sharpe'] / pis['sharpe'] if pis['sharpe'] > 0 else 0.0
        all_results.append({
            'name': name,
            'full': pf, 'is': pis, 'oos': poos,
            'oos_ratio': oos_ratio,
            'last30': last30_trades,
        })
        print(f"ΣRet={pf['sum_ret']:+.0f}% | OOS/IS={oos_ratio:.2f} | L30d={last30_trades}")

    # ═══════════════════════════════════════════════════════════════════
    #  Output Tables
    # ═══════════════════════════════════════════════════════════════════

    print()
    print("=" * 130)
    print("  FULL-PERIOD RESULTS (365 days)")
    print("=" * 130)
    hdr = (f"  {'Strategy':<16} {'ΣRet%':>9} {'GeoRet%':>8} {'Sharpe':>7} "
           f"{'MDD%':>6} {'Calmar':>7} {'Trades':>7} {'WR%':>6} {'L30d':>5}")
    print(hdr)
    print(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
    for r in all_results:
        f = r['full']
        print(f"  {r['name']:<16} {f['sum_ret']:+9.0f} {f['ret']:+8.1f} "
              f"{f['sharpe']:7.2f} {f['mdd']:6.1f} {f['calmar']:7.2f} "
              f"{f['trades']:7d} {f['wr']*100:5.1f} {r['last30']:5d}")

    # ── Walk-forward table ──
    print()
    print("=" * 130)
    print("  WALK-FORWARD: IN-SAMPLE (first 182d) vs OUT-OF-SAMPLE (last 183d)")
    print("  OOS/IS ratio near 1.0 = robust, <0.5 = overfitted")
    print("=" * 130)
    hdr2 = (f"  {'Strategy':<16} {'IS Shrp':>7} {'IS ΣRet':>9} {'IS Trd':>7} "
            f"│ {'OOS Shrp':>8} {'OOS ΣRet':>9} {'OOS Trd':>7} "
            f"│ {'OOS/IS':>6} {'⚠ Overfit':>10}")
    print(hdr2)
    print(f"  {'-'*16} {'-'*7} {'-'*9} {'-'*7} ┼ {'-'*8} {'-'*9} {'-'*7} ┼ {'-'*6} {'-'*10}")
    for r in all_results:
        is_ = r['is']
        oos = r['oos']
        ratio = r['oos_ratio']
        flag = '✓ OK' if ratio >= 0.7 else ('⚠ WARN' if ratio >= 0.5 else '✗ OVERFIT')
        print(f"  {r['name']:<16} {is_['sharpe']:7.2f} {is_['sum_ret']:+9.0f} "
              f"{is_['trades']:7d} │ {oos['sharpe']:8.2f} {oos['sum_ret']:+9.0f} "
              f"{oos['trades']:7d} │ {ratio:6.2f} {flag:>10}")

    # ── Heatmap: Sharpe (full period) ──
    print()
    print("=" * 80)
    print("  HEATMAP: Full-period Sharpe  (rows=multiplier, cols=floor)")
    print("=" * 80)
    print(f"  {'':>6}", end='')
    for fl in FLOORS:
        print(f"  f{fl*100:4.1f}%", end='')
    print()
    print(f"  {'':>6}", end='')
    for _ in FLOORS:
        print(f"  {'-'*7}", end='')
    print()
    idx = 1  # skip fixed baseline (idx 0)
    for mult in MULTIPLIERS:
        print(f"  {mult}x  ", end='')
        for _ in FLOORS:
            r = all_results[idx]
            sh = r['full']['sharpe']
            print(f"  {sh:7.2f}", end='')
            idx += 1
        print()

    # ── Heatmap: OOS/IS ratio ──
    print()
    print("=" * 80)
    print("  HEATMAP: OOS/IS Sharpe Ratio  (>0.7 = robust, <0.5 = overfitted)")
    print("=" * 80)
    print(f"  {'':>6}", end='')
    for fl in FLOORS:
        print(f"  f{fl*100:4.1f}%", end='')
    print()
    print(f"  {'':>6}", end='')
    for _ in FLOORS:
        print(f"  {'-'*7}", end='')
    print()
    idx = 1
    for mult in MULTIPLIERS:
        print(f"  {mult}x  ", end='')
        for _ in FLOORS:
            r = all_results[idx]
            ratio = r['oos_ratio']
            print(f"  {ratio:7.2f}", end='')
            idx += 1
        print()

    # ── Heatmap: Last 30d trades ──
    print()
    print("=" * 80)
    print("  HEATMAP: Last-30d Signal Count  (0 = dead in current regime)")
    print("=" * 80)
    print(f"  {'':>6}", end='')
    for fl in FLOORS:
        print(f"  f{fl*100:4.1f}%", end='')
    print()
    print(f"  {'':>6}", end='')
    for _ in FLOORS:
        print(f"  {'-'*7}", end='')
    print()
    idx = 1
    for mult in MULTIPLIERS:
        print(f"  {mult}x  ", end='')
        for _ in FLOORS:
            r = all_results[idx]
            print(f"  {r['last30']:7d}", end='')
            idx += 1
        print()

    # ── Heatmap: Trades (full) ──
    print()
    print("=" * 80)
    print("  HEATMAP: Total Trades (365d)")
    print("=" * 80)
    print(f"  {'':>6}", end='')
    for fl in FLOORS:
        print(f"  f{fl*100:4.1f}%", end='')
    print()
    print(f"  {'':>6}", end='')
    for _ in FLOORS:
        print(f"  {'-'*7}", end='')
    print()
    idx = 1
    for mult in MULTIPLIERS:
        print(f"  {mult}x  ", end='')
        for _ in FLOORS:
            r = all_results[idx]
            print(f"  {r['full']['trades']:7d}", end='')
            idx += 1
        print()

    # ── Sweet spot recommendation ──
    print()
    print("=" * 130)
    print("  SWEET SPOT CANDIDATES  (OOS/IS ≥ 0.6, L30d > 0, sorted by OOS Sharpe)")
    print("=" * 130)
    candidates = [r for r in all_results
                  if r['oos_ratio'] >= 0.6 and r['last30'] > 0]
    candidates.sort(key=lambda r: r['oos']['sharpe'], reverse=True)
    if candidates:
        print(f"  {'Strategy':<16} {'Full Shrp':>9} {'OOS Shrp':>9} {'OOS/IS':>7} "
              f"{'ΣRet':>9} {'OOS ΣRet':>9} {'Trades':>7} {'L30d':>5} {'WR%':>6}")
        print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*9} {'-'*7} {'-'*5} {'-'*6}")
        for r in candidates[:10]:
            f = r['full']
            o = r['oos']
            print(f"  {r['name']:<16} {f['sharpe']:9.2f} {o['sharpe']:9.2f} "
                  f"{r['oos_ratio']:7.2f} {f['sum_ret']:+9.0f} {o['sum_ret']:+9.0f} "
                  f"{f['trades']:7d} {r['last30']:5d} {f['wr']*100:5.1f}")
    else:
        print("  No variants meet the criteria. Consider relaxing L30d threshold.")

    # ── Fixed baseline reference ──
    fb = all_results[0]
    print(f"\n  Reference: Fixed 3.2% → Full Sharpe={fb['full']['sharpe']:.2f}, "
          f"OOS/IS={fb['oos_ratio']:.2f}, L30d={fb['last30']}")
    print()


if __name__ == '__main__':
    main()
