#!/usr/bin/env python3
"""
A/B Test: Fixed Threshold (3.2%) vs Volatility-Adaptive Threshold
=================================================================
Runs both strategies on 365 days of real Binance 15m data across 20 symbols.

Strategy A (Control):  Fixed threshold = 3.2%
Strategy B (Adaptive): threshold = max(floor, k × ATR_pct(atr_lookback))
  where ATR_pct = ATR / close, so threshold scales with recent realized volatility.

Uses EXACT same pipeline as production:
  compute_signals → apply_mtf_confirmation → apply_mtf_leverage → backtest_core
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import time
import json
import hashlib
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
from quantbox.backtest.vectorized_backtest import (
    _backtest_core,
)

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
#  (EXACT replica of compute_signals_v9, only threshold is per-bar)
# ═══════════════════════════════════════════════════════════════════════

def compute_signals_adaptive(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    taker_buy: np.ndarray,
    btc_close: np.ndarray,
    lookback: int,
    target_retrace: float,
    stop_mult: float,
    max_hold: int,
    stop_decay: float,
    btc_mom_period: int,
    btc_regime_period: int,
    asset_trend_period: int,
    min_confidence: int,
    base_leverage: float,
    confidence_leverage_step: float,
    max_leverage: float,
    rsi_period: int,
    vol_period: int,
    dynamic_target_enabled: bool,
    target_conf3: float,
    target_conf4: float,
    target_conf5: float,
    stoch_enabled: bool,
    stoch_k_period: int,
    stoch_d_period: int,
    macd_enabled: bool,
    macd_fast: int,
    macd_slow: int,
    macd_signal_period: int,
    # ── Adaptive params ──
    atr_period: int = 48,
    atr_multiplier: float = 2.0,
    threshold_floor: float = 0.015,
    threshold_cap: float = 0.06,
) -> np.ndarray:
    """Volatility-adaptive compute_signals_v9.

    IDENTICAL logic to compute_signals_v9, except:
    threshold[i] = clip(k * ATR_pct(atr_period)[i], floor, cap)
    """
    n = len(close)
    sig = np.zeros(n)

    # Pre-compute ATR-based per-bar thresholds
    atr_pct = compute_atr_pct(high, low, close, atr_period)

    # Pre-compute indicators (same as compute_signals_v9)
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
    start_bar = max(lookback, btc_mom_period, btc_regime_period, asset_trend_period,
                    rsi_period, vol_period, atr_period)

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        # ── Per-bar adaptive threshold ──
        atr_val = atr_pct[i]
        if np.isnan(atr_val):
            threshold = 0.032  # fallback
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
                    # ═══ RISK-ON: Classic Pullback + Confidence ═══
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
                    # ═══ RISK-OFF: Sympathy Pullback + Confidence ═══
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
            # ═══ EXIT LOGIC (exact copy from compute_signals_v9) ═══
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
#  Data Download
# ═══════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, interval: str, days: int) -> Path:
    """Generate cache file path based on symbol, interval, and date."""
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return CACHE_DIR / f'{symbol}_{interval}_{days}d_{today}.npz'


def download_15m(client: Client, symbol: str, days: int) -> dict | None:
    """Download 15m OHLCV data for a symbol (with disk cache)."""
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
            startTime=cur, endTime=end_ms, limit=1500,
        )
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        time.sleep(0.35)  # Conservative rate limiting

    if len(all_klines) < 200:
        return None

    # Drop incomplete last bar (parity with live)
    all_klines = all_klines[:-1]

    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in all_klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in all_klines], dtype=np.float64)
    volume = np.array([float(k[5]) for k in all_klines], dtype=np.float64)
    taker_buy = np.array([float(k[9]) for k in all_klines], dtype=np.float64)

    # Save cache
    np.savez_compressed(cache, close=close, high=high, low=low,
                        volume=volume, taker_buy=taker_buy)

    return {'close': close, 'high': high, 'low': low,
            'volume': volume, 'taker_buy': taker_buy}


def download_4h(client: Client, symbol: str, days: int) -> np.ndarray | None:
    """Download 4h close data for MTF confirmation (with disk cache)."""
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
            startTime=cur, endTime=end_ms, limit=1500,
        )
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        time.sleep(0.35)  # Conservative rate limiting

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

    # Load settings into config
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
    print()

    # ── Download BTC first ──
    print(f"Downloading {DAYS}-day data...")
    btc_data = download_15m(client, 'BTCUSDT', DAYS)
    btc_close = btc_data['close']
    btc_4h = download_4h(client, 'BTCUSDT', DAYS)
    print(f"  BTC: {len(btc_close)} bars (15m), "
          f"{len(btc_4h) if btc_4h is not None else 0} bars (4h)")

    # ── Download all symbols ──
    all_data = {}
    all_4h = {}
    for sym in symbols:
        d = download_15m(client, sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}: insufficient data")
            continue
        h4 = download_4h(client, sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4
        print(f"  {sym}: {len(d['close'])} bars (15m), "
              f"{len(h4) if h4 is not None else 0} (4h)")
        time.sleep(0.05)

    print(f"\nDownloaded {len(all_data)} symbols\n")

    # ── JIT warmup ──
    print("Warming up Numba JIT...")
    dummy = np.random.randn(300).cumsum() + 100.0
    dummy_h = dummy + 0.5
    dummy_l = dummy - 0.5
    compute_atr_pct(dummy_h, dummy_l, dummy, 14)
    _backtest_core(dummy, np.zeros(300), COST, 0.0, BARS_PER_DAY)
    print("JIT compiled\n")

    # ═══════════════════════════════════════════════════════════════════
    #  Adaptive threshold parameter grid
    # ═══════════════════════════════════════════════════════════════════
    adaptive_configs = [
        {'name': 'Adpt_2x_48', 'atr_period': 48, 'atr_mult': 2.0,
         'floor': 0.015, 'cap': 0.06},
        {'name': 'Adpt_2.5x_48', 'atr_period': 48, 'atr_mult': 2.5,
         'floor': 0.015, 'cap': 0.06},
        {'name': 'Adpt_3x_48', 'atr_period': 48, 'atr_mult': 3.0,
         'floor': 0.015, 'cap': 0.06},
        {'name': 'Adpt_2x_96', 'atr_period': 96, 'atr_mult': 2.0,
         'floor': 0.015, 'cap': 0.06},
        {'name': 'Adpt_2x_48_f10', 'atr_period': 48, 'atr_mult': 2.0,
         'floor': 0.010, 'cap': 0.06},
    ]

    # ═══════════════════════════════════════════════════════════════════
    #  Helper: generate signals + MTF pipeline
    # ═══════════════════════════════════════════════════════════════════

    def generate_signals_fixed(data: dict, btc: np.ndarray,
                               close_4h: np.ndarray | None) -> tuple:
        """Strategy A: fixed threshold."""
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

    def generate_signals_adaptive(data: dict, btc: np.ndarray,
                                  close_4h: np.ndarray | None,
                                  atr_period: int, atr_mult: float,
                                  floor: float, cap: float) -> tuple:
        """Strategy B: adaptive threshold."""
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
            atr_period=atr_period, atr_multiplier=atr_mult,
            threshold_floor=floor, threshold_cap=cap,
        )

        if mtf_enabled and close_4h is not None:
            sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
            sigs = apply_mtf_leverage(sigs, mtf_lev)

        return c, sigs

    def run_bt(close: np.ndarray, signals: np.ndarray) -> dict:
        eq, total_ret, sharpe, max_dd, wr, n_trades, bars_in, pf, avg_ret, _ = (
            _backtest_core(close, signals, COST, FUNDING, BARS_PER_DAY)
        )
        calmar = (total_ret * 100 / (max_dd * 100)) if max_dd > 0 else 0.0
        return {
            'ret': total_ret * 100,
            'sharpe': sharpe,
            'mdd': max_dd * 100,
            'calmar': calmar,
            'wr': wr,
            'trades': int(n_trades),
            'pf': pf,
            'equity': eq,
        }

    def portfolio_metrics(results: dict) -> dict:
        """Compute portfolio metrics (both sum and geometric mean)."""
        n_syms = len(results)
        if n_syms == 0:
            return {'ret': 0, 'sum_ret': 0, 'sharpe': 0, 'mdd': 0, 'calmar': 0,
                    'trades': 0, 'wr': 0, 'avg_ret': 0}
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
        avg_ret = np.mean([r['ret'] for r in results.values()])
        sum_ret = sum(r['ret'] for r in results.values())  # ΣRet (your 4000% figure)
        return {'ret': ret, 'sum_ret': sum_ret, 'sharpe': sharpe, 'mdd': dd,
                'calmar': calmar, 'trades': trades, 'wr': wr, 'avg_ret': avg_ret}

    # ═══════════════════════════════════════════════════════════════════
    #  Run A: Fixed threshold
    # ═══════════════════════════════════════════════════════════════════
    print("Running Strategy A (Fixed threshold = {:.1%})...".format(cfg.threshold))
    results_a = {}
    for sym, data in all_data.items():
        close_4h = all_4h.get(sym)
        c, sigs = generate_signals_fixed(data, btc_close, close_4h)
        results_a[sym] = run_bt(c, sigs)
    pa = portfolio_metrics(results_a)
    print(f"  A: ΣRet={pa['sum_ret']:+.0f}%, GeoPort={pa['ret']:+.1f}%, Sharpe={pa['sharpe']:.2f}, "
          f"MDD={pa['mdd']:.1f}%, Trades={pa['trades']}")

    # ═══════════════════════════════════════════════════════════════════
    #  Run B: Adaptive variants
    # ═══════════════════════════════════════════════════════════════════
    all_pb = {}
    all_results_b = {}
    for acfg in adaptive_configs:
        name = acfg['name']
        print(f"\nRunning Strategy B [{name}]...")
        results_b = {}
        for sym, data in all_data.items():
            close_4h = all_4h.get(sym)
            c, sigs = generate_signals_adaptive(
                data, btc_close, close_4h,
                acfg['atr_period'], acfg['atr_mult'],
                acfg['floor'], acfg['cap'],
            )
            results_b[sym] = run_bt(c, sigs)
        pb = portfolio_metrics(results_b)
        all_pb[name] = pb
        all_results_b[name] = results_b
        print(f"  B[{name}]: ΣRet={pb['sum_ret']:+.0f}%, GeoPort={pb['ret']:+.1f}%, "
              f"Sharpe={pb['sharpe']:.2f}, MDD={pb['mdd']:.1f}%, Trades={pb['trades']}")

    # ═══════════════════════════════════════════════════════════════════
    #  Print per-symbol comparison (best adaptive vs fixed)
    # ═══════════════════════════════════════════════════════════════════
    best_name = max(all_pb, key=lambda k: all_pb[k]['calmar'])
    best_pb = all_pb[best_name]
    best_results_b = all_results_b[best_name]

    print()
    print("=" * 110)
    print(f"  A/B TEST — {DAYS}d, {len(all_data)} Symbols")
    print(f"  A = Fixed {cfg.threshold:.1%}  |  B = {best_name} (best Calmar)")
    print("=" * 110)
    print()
    print(f"{'Symbol':<16} {'--- A (Fixed) ---':^40}  {'--- B (Adaptive) ---':^40}")
    print(f"{'':16} {'Ret%':>7} {'Shrp':>6} {'MDD%':>6} {'Calm':>6} {'#Tr':>5} {'WR':>5}"
          f"  {'Ret%':>7} {'Shrp':>6} {'MDD%':>6} {'Calm':>6} {'#Tr':>5} {'WR':>5}")
    print("-" * 110)

    for sym in sorted(results_a.keys()):
        a = results_a[sym]
        b = best_results_b[sym]
        print(
            f"{sym:<16} "
            f"{a['ret']:+7.1f} {a['sharpe']:6.2f} {a['mdd']:6.1f} "
            f"{a['calmar']:6.2f} {a['trades']:5d} {a['wr']*100:4.0f}%"
            f"  {b['ret']:+7.1f} {b['sharpe']:6.2f} {b['mdd']:6.1f} "
            f"{b['calmar']:6.2f} {b['trades']:5d} {b['wr']*100:4.0f}%"
        )

    print("-" * 110)
    print(
        f"{'ΣRet (sum)':<16} "
        f"{pa['sum_ret']:+7.0f} {'':6} {'':6} "
        f"{'':6} {'':5} {'':5}"
        f"  {best_pb['sum_ret']:+7.0f} {'':6} {'':6} "
        f"{'':6} {'':5} {'':5}"
    )
    print(
        f"{'GeoPort (mean)':<16} "
        f"{pa['ret']:+7.1f} {pa['sharpe']:6.2f} {pa['mdd']:6.1f} "
        f"{pa['calmar']:6.2f} {pa['trades']:5d} {pa['wr']*100:4.0f}%"
        f"  {best_pb['ret']:+7.1f} {best_pb['sharpe']:6.2f} {best_pb['mdd']:6.1f} "
        f"{best_pb['calmar']:6.2f} {best_pb['trades']:5d} {best_pb['wr']*100:4.0f}%"
    )

    # ── All variants table ──
    print()
    print("=" * 90)
    print("  ALL VARIANTS")
    print("=" * 90)
    print(f"  {'Strategy':<22} {'ΣRet%':>9} {'GeoRet%':>8} {'Sharpe':>8} {'MDD%':>8} {'Calmar':>8} {'Trades':>8} {'WR%':>6}")
    print(f"  {'-'*22} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'Strategy':<22} {'ΣRet%':>9} {'GeoRet%':>8} {'Sharpe':>8} {'MDD%':>8} {'Calmar':>8} {'Trades':>8} {'WR%':>6}")
    print(f"  {'-'*22} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'A: Fixed '+f'{cfg.threshold:.1%}':<22} "
          f"{pa['sum_ret']:+9.0f} {pa['ret']:+8.1f} {pa['sharpe']:8.2f} {pa['mdd']:8.1f} "
          f"{pa['calmar']:8.2f} {pa['trades']:8d} {pa['wr']*100:5.1f}")
    for acfg in adaptive_configs:
        n = acfg['name']
        p = all_pb[n]
        marker = '  <--BEST' if n == best_name else ''
        print(f"  {'B: '+n:<22} "
              f"{p['sum_ret']:+9.0f} {p['ret']:+8.1f} {p['sharpe']:8.2f} {p['mdd']:8.1f} "
              f"{p['calmar']:8.2f} {p['trades']:8d} {p['wr']*100:5.1f}{marker}")

    # ── Verdict ──
    print()
    print("=" * 90)
    print("  VERDICT")
    print("=" * 90)
    metrics = [
        ('ΣReturn', pa['sum_ret'], best_pb['sum_ret'], True),
        ('GeoReturn', pa['ret'], best_pb['ret'], True),
        ('Sharpe', pa['sharpe'], best_pb['sharpe'], True),
        ('MaxDD', pa['mdd'], best_pb['mdd'], False),
        ('Calmar', pa['calmar'], best_pb['calmar'], True),
        ('Trades', pa['trades'], best_pb['trades'], True),
    ]
    a_wins = b_wins = 0
    for mname, va, vb, higher_better in metrics:
        w = 'B' if (vb > va if higher_better else vb < va) else 'A'
        if w == 'A': a_wins += 1
        else: b_wins += 1
        d = vb - va
        print(f"  {mname:<10}: A={va:>8.2f}  B={vb:>8.2f}  Delta={d:+.2f}  Winner={w}")

    overall = 'A (Fixed)' if a_wins > b_wins else f'B ({best_name})'
    print(f"\n  WINNER: {overall} ({max(a_wins, b_wins)}/{a_wins+b_wins} metrics)\n")

    # ── ATR diagnostics ──
    print("=" * 90)
    print("  ADAPTIVE THRESHOLD DISTRIBUTION (best variant)")
    print("=" * 90)
    acfg_best = next(c for c in adaptive_configs if c['name'] == best_name)
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']:
        if sym not in all_data:
            continue
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        atr_pct = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n],
                                  acfg_best['atr_period'])
        valid = atr_pct[~np.isnan(atr_pct)]
        thresholds = np.clip(acfg_best['atr_mult'] * valid,
                             acfg_best['floor'], acfg_best['cap'])
        print(f"  {sym:<16} ATR%: p10={np.percentile(valid,10)*100:.3f}% "
              f"p50={np.percentile(valid,50)*100:.3f}% "
              f"p90={np.percentile(valid,90)*100:.3f}%  "
              f"Thr: p10={np.percentile(thresholds,10)*100:.2f}% "
              f"p50={np.percentile(thresholds,50)*100:.2f}% "
              f"p90={np.percentile(thresholds,90)*100:.2f}%")
    print()


if __name__ == '__main__':
    main()
