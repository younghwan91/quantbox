#!/usr/bin/env python3
"""
R23: New Alpha Search — 4 Dimensions
======================================
After R22 proved PF=0.877 for rubber band 15m MR at ZERO cost,
we search for alpha across 4 fundamentally different approaches:

  A) Timeframe: Does mean-reversion work better on 1h/4h?
  B) Momentum:  Does following the trend beat fading it?
  C) Regime:    Ranging→MR, Trending→Momentum (adaptive)?
  D) Volume:    Can order flow / volume patterns generate edge?

All tests: zero cost, 1x leverage, 6yr, 50 symbols, zero lookahead.

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/alpha_search_r23.py
"""

import os
import sys
import time
import json
import numpy as np
import numba
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
    compute_atr_pct,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache_extended'
DATA_DAYS = 2190
INITIAL_CAPITAL = 2095.0
FUNDING_PER_BAR_15m = 0.01 / 100 / 32  # 0.01% per 8h ÷ 32 15m-bars

# Production entry parameters (v12.5.3)
LOOKBACK = 8
THRESHOLD = 0.032
BTC_MOM_PERIOD = 48
BTC_REGIME_PERIOD = 96
RSI_PERIOD = 14
VOL_PERIOD = 20
ATR_PERIOD = 48
ATR_MULT = 2.0
THR_FLOOR = 0.008
THR_CAP = 0.06

SYMBOLS_50 = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT', 'EGLDUSDT',
    'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT',
    'CRVUSDT', 'LDOUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'VETUSDT',
    'EOSUSDT', 'APEUSDT',
]


# ═══════════════════════════════════════════════════════════════════
# Data Loading & Resampling
# ═══════════════════════════════════════════════════════════════════
def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    pattern = f"{symbol}_{interval}_{days}d_"
    for f in sorted(CACHE_DIR.glob(f"{pattern}*.npz"), reverse=True):
        return f
    return None


def load_all_15m() -> dict:
    """Load 15m OHLCV for all symbols."""
    print("📊 Loading 15m data...")
    data = {}
    for sym in SYMBOLS_50:
        f = _find_cache(sym, '15m', DATA_DAYS)
        if f is None:
            continue
        d = np.load(f)
        data[sym] = {
            'close': d['close'].astype(np.float64),
            'high': d['high'].astype(np.float64),
            'low': d['low'].astype(np.float64),
            'volume': d['volume'].astype(np.float64),
            'taker_buy': d['taker_buy'].astype(np.float64),
        }
    print(f"  Loaded {len(data)} symbols, {len(next(iter(data.values()))['close']):,} bars each")
    return data


@numba.jit(nopython=True, cache=True)
def _resample_ohlcv(close, high, low, volume, taker_buy, factor):
    """Resample 15m → higher TF. factor=4 for 1h, 16 for 4h."""
    n = len(close)
    n_out = n // factor
    rc = np.zeros(n_out)
    rh = np.zeros(n_out)
    rl = np.zeros(n_out)
    rv = np.zeros(n_out)
    rt = np.zeros(n_out)
    for j in range(n_out):
        s = j * factor
        rc[j] = close[s + factor - 1]
        mx = high[s]
        mn = low[s]
        vs = 0.0
        ts = 0.0
        for k in range(s, s + factor):
            if high[k] > mx:
                mx = high[k]
            if low[k] < mn:
                mn = low[k]
            vs += volume[k]
            ts += taker_buy[k]
        rh[j] = mx
        rl[j] = mn
        rv[j] = vs
        rt[j] = ts
    return rc, rh, rl, rv, rt


def resample_all(data: dict, factor: int) -> dict:
    """Resample all symbols to higher TF."""
    out = {}
    for sym, d in data.items():
        c, h, l, v, t = _resample_ohlcv(
            d['close'], d['high'], d['low'], d['volume'], d['taker_buy'], factor)
        out[sym] = {'close': c, 'high': h, 'low': l, 'volume': v, 'taker_buy': t}
    return out


# ═══════════════════════════════════════════════════════════════════
# Entry Scanners
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_mr_momentum(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period,
    min_confidence, rsi_period, vol_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
    flip_direction,   # True=momentum (follow move), False=MR (fade move)
    use_btc_filter,   # True=require BTC alignment, False=no BTC
):
    """Unified scanner for mean-reversion AND momentum.

    Returns:
        dirs: entry direction (+1 LONG, -1 SHORT, 0 none)
        confs: confidence score
        moves: absolute move size (|return|)
        atrs: ATR% at entry bar
    """
    n = min(len(close), len(btc_close))
    dirs = np.zeros(n)
    confs = np.zeros(n)
    moves = np.zeros(n)
    atrs = np.full(n, np.nan)

    rsi = _compute_rsi_inline(close[:n], rsi_period)
    vol_sma = _compute_sma_inline(volume[:n], vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    if atr_period > 0:
        atr_pct = compute_atr_pct(high[:n], low[:n], close[:n], atr_period)
    else:
        atr_pct = np.full(n, np.nan)

    start = max(lookback, btc_mom_period, rsi_period, vol_period)
    if atr_period > start:
        start = atr_period

    for i in range(start, n):
        ret = close[i] / close[i - lookback] - 1.0

        # Adaptive threshold
        if atr_period > 0 and not np.isnan(atr_pct[i]):
            thr = max(threshold_floor, min(atr_multiplier * atr_pct[i], threshold_cap))
            atrs[i] = atr_pct[i]
        else:
            thr = threshold
            atrs[i] = abs(ret)

        if abs(ret) <= thr:
            continue

        # BTC alignment
        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

        direction = 0
        confidence = 0

        # ═══ DOWNWARD MOVE ═══
        if ret < -thr:
            if use_btc_filter and btc_mom >= 0:
                continue  # BTC not aligned for Risk-Off
            if flip_direction:
                direction = -1   # Momentum: follow down
            else:
                direction = 1    # MR: fade down

            # Confidence: oversold indicators for MR, trend-strength for momentum
            if not flip_direction:
                # MR: want oversold
                if rsi[i] < 35:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] < 0.42:
                    confidence += 1
                if btc_mom < -0.01:
                    confidence += 1
            else:
                # Momentum: want continuation signals
                if rsi[i] < 45 and rsi[i] > 15:
                    confidence += 1  # room to fall further
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1  # volume confirms
                if taker_ratio[i] < 0.42:
                    confidence += 1  # sellers in control
                if btc_mom < -0.01:
                    confidence += 1  # BTC cascade

            if confidence < min_confidence:
                direction = 0

        # ═══ UPWARD MOVE ═══
        elif ret > thr:
            if use_btc_filter and btc_mom <= 0:
                continue
            if flip_direction:
                direction = 1    # Momentum: follow up
            else:
                direction = -1   # MR: fade up

            if not flip_direction:
                if rsi[i] > 65:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] > 0.58:
                    confidence += 1
                if btc_mom > 0.01:
                    confidence += 1
            else:
                if rsi[i] > 55 and rsi[i] < 85:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] > 0.58:
                    confidence += 1
                if btc_mom > 0.01:
                    confidence += 1

            if confidence < min_confidence:
                direction = 0

        if direction != 0:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            moves[i] = abs(ret)

    return dirs, confs, moves, atrs


# ═══════════════════════════════════════════════════════════════════
# Regime Detection
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def compute_efficiency_ratio(close, period):
    """Kaufman's Efficiency Ratio: |net_change| / sum(|bar_changes|).

    Close to 1.0 = trending (direct move), close to 0.0 = ranging (choppy).
    """
    n = len(close)
    er = np.full(n, np.nan)
    for i in range(period, n):
        net = abs(close[i] - close[i - period])
        total = 0.0
        for j in range(i - period + 1, i + 1):
            total += abs(close[j] - close[j - 1])
        er[i] = net / total if total > 0 else 0.0
    return er


def apply_regime_filter(dirs, confs, moves, er, threshold, keep_below):
    """Filter entries by efficiency ratio.

    Args:
        keep_below: True = keep entries where ER < threshold (range-bound).
                    False = keep entries where ER > threshold (trending).
    """
    od = dirs.copy()
    for i in range(len(dirs)):
        if dirs[i] == 0.0:
            continue
        if np.isnan(er[i]):
            od[i] = 0.0
            continue
        if keep_below and er[i] >= threshold:
            od[i] = 0.0
        elif not keep_below and er[i] <= threshold:
            od[i] = 0.0
    return od, confs, moves


# ═══════════════════════════════════════════════════════════════════
# Volume Signal Scanners
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_volume_climax(close, high, low, volume, taker_buy,
                       vol_period, vol_mult, atr_period, rsi_period):
    """D1: Volume Climax Reversal.

    High volume spike + candle rejection (long wick) = exhaustion reversal.
    """
    n = len(close)
    dirs = np.zeros(n)
    confs = np.zeros(n)
    sizes = np.zeros(n)  # ATR as size proxy

    vol_sma = _compute_sma_inline(volume, vol_period)
    atr = compute_atr_pct(high, low, close, atr_period)
    rsi = _compute_rsi_inline(close, rsi_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    start = max(vol_period, atr_period, rsi_period) + 1

    for i in range(start, n):
        if np.isnan(vol_sma[i]) or np.isnan(atr[i]):
            continue
        if volume[i] < vol_mult * vol_sma[i]:
            continue  # need volume spike

        # Use close[i-1] as open proxy
        bar_range = high[i] - low[i]
        if bar_range < 1e-12:
            continue

        lower_wick = min(close[i], close[i - 1]) - low[i]
        upper_wick = high[i] - max(close[i], close[i - 1])

        direction = 0
        confidence = 0

        # LONG: Long lower wick = buying at lows (rejection of downside)
        if lower_wick > 0.5 * bar_range and rsi[i] < 45:
            direction = 1
            confidence += 1  # wick rejection itself
            if taker_ratio[i] < 0.45:
                confidence += 1  # sellers exhausted
            if rsi[i] < 30:
                confidence += 1
            if close[i] > close[i - 1]:
                confidence += 1  # green close (bullish)
            if volume[i] > vol_mult * 1.5 * vol_sma[i]:
                confidence += 1

        # SHORT: Long upper wick = rejection of upside
        elif upper_wick > 0.5 * bar_range and rsi[i] > 55:
            direction = -1
            confidence += 1
            if taker_ratio[i] > 0.55:
                confidence += 1  # buyers exhausted
            if rsi[i] > 70:
                confidence += 1
            if close[i] < close[i - 1]:
                confidence += 1  # red close (bearish)
            if volume[i] > vol_mult * 1.5 * vol_sma[i]:
                confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            sizes[i] = atr[i]

    return dirs, confs, sizes


@numba.jit(nopython=True, cache=True)
def scan_taker_exhaustion(close, high, low, volume, taker_buy,
                          vol_period, atr_period, rsi_period,
                          ratio_lo, ratio_hi, lookback):
    """D2: Taker Exhaustion.

    Extreme taker buy ratio = one side exhausted → reversal.
    """
    n = len(close)
    dirs = np.zeros(n)
    confs = np.zeros(n)
    sizes = np.zeros(n)

    atr = compute_atr_pct(high, low, close, atr_period)
    rsi = _compute_rsi_inline(close, rsi_period)
    vol_sma = _compute_sma_inline(volume, vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    # Rolling average of taker ratio
    tr_sma = _compute_sma_inline(taker_ratio, lookback)

    start = max(vol_period, atr_period, rsi_period, lookback) + 1

    for i in range(start, n):
        if np.isnan(atr[i]) or np.isnan(tr_sma[i]):
            continue

        direction = 0
        confidence = 0

        # Sellers exhausted (low taker buy = heavy selling) → LONG
        if taker_ratio[i] < ratio_lo:
            direction = 1
            confidence += 1
            if rsi[i] < 35:
                confidence += 1
            if close[i] > close[i - 1]:
                confidence += 1  # bouncing
            if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                confidence += 1  # volume spike
            if tr_sma[i] < 0.47:
                confidence += 1  # sustained selling pressure

        # Buyers exhausted (high taker buy = heavy buying) → SHORT
        elif taker_ratio[i] > ratio_hi:
            direction = -1
            confidence += 1
            if rsi[i] > 65:
                confidence += 1
            if close[i] < close[i - 1]:
                confidence += 1
            if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                confidence += 1
            if tr_sma[i] > 0.53:
                confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            sizes[i] = atr[i]

    return dirs, confs, sizes


@numba.jit(nopython=True, cache=True)
def scan_cvd_divergence(close, high, low, volume, taker_buy,
                        lookback, atr_period, rsi_period):
    """D3: CVD Divergence.

    Price makes new extreme but Cumulative Volume Delta doesn't confirm.
    Hidden buying: price new low + CVD rising.
    Hidden selling: price new high + CVD falling.
    """
    n = len(close)
    dirs = np.zeros(n)
    confs = np.zeros(n)
    sizes = np.zeros(n)

    atr = compute_atr_pct(high, low, close, atr_period)
    rsi = _compute_rsi_inline(close, rsi_period)

    # Cumulative Volume Delta: net buy pressure per bar
    cvd = np.zeros(n)
    for i in range(n):
        delta = 2.0 * taker_buy[i] - volume[i]
        cvd[i] = (cvd[i - 1] + delta) if i > 0 else delta

    start = max(lookback, atr_period, rsi_period) + 1

    for i in range(start, n):
        if np.isnan(atr[i]):
            continue

        # Find min/max of price and CVD over lookback window
        price_min = close[i]
        price_max = close[i]
        cvd_min = cvd[i]
        cvd_max = cvd[i]

        for j in range(i - lookback, i):
            if close[j] < price_min:
                price_min = close[j]
            if close[j] > price_max:
                price_max = close[j]
            if cvd[j] < cvd_min:
                cvd_min = cvd[j]
            if cvd[j] > cvd_max:
                cvd_max = cvd[j]

        direction = 0
        confidence = 0

        # Calculate a relative CVD position
        cvd_range = cvd_max - cvd_min
        if cvd_range < 1e-12:
            continue

        # Bullish divergence: price near lookback low, CVD NOT at low
        price_range = price_max - price_min
        if price_range < 1e-12:
            continue

        price_pct = (close[i] - price_min) / price_range  # 0=at low, 1=at high
        cvd_pct = (cvd[i] - cvd_min) / cvd_range

        if price_pct < 0.1 and cvd_pct > 0.3:
            # Price at bottom but CVD well above its low → hidden buying
            direction = 1
            confidence += 1
            if rsi[i] < 40:
                confidence += 1
            if cvd[i] > cvd[i - 1]:
                confidence += 1  # CVD currently rising
            if close[i] > close[i - 1]:
                confidence += 1

        elif price_pct > 0.9 and cvd_pct < 0.7:
            # Price at top but CVD well below its high → hidden selling
            direction = -1
            confidence += 1
            if rsi[i] > 60:
                confidence += 1
            if cvd[i] < cvd[i - 1]:
                confidence += 1
            if close[i] < close[i - 1]:
                confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            sizes[i] = atr[i]

    return dirs, confs, sizes


@numba.jit(nopython=True, cache=True)
def scan_vwap_deviation(close, high, low, volume, taker_buy,
                        vwap_period, atr_period, rsi_period, dev_mult):
    """D4: VWAP Mean-Reversion.

    When price deviates significantly from volume-weighted fair price,
    fade the deviation.
    """
    n = len(close)
    dirs = np.zeros(n)
    confs = np.zeros(n)
    sizes = np.zeros(n)

    atr = compute_atr_pct(high, low, close, atr_period)
    rsi = _compute_rsi_inline(close, rsi_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    # Rolling VWAP via cumulative sums
    cv_cum = np.zeros(n + 1)
    v_cum = np.zeros(n + 1)
    for i in range(n):
        cv_cum[i + 1] = cv_cum[i] + close[i] * volume[i]
        v_cum[i + 1] = v_cum[i] + volume[i]

    vwap = np.full(n, np.nan)
    for i in range(vwap_period - 1, n):
        s = i - vwap_period + 1
        sv = v_cum[i + 1] - v_cum[s]
        if sv > 0:
            vwap[i] = (cv_cum[i + 1] - cv_cum[s]) / sv

    start = max(vwap_period, atr_period, rsi_period) + 1

    for i in range(start, n):
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or atr[i] < 1e-12:
            continue

        # Deviation in ATR units
        dev = (close[i] - vwap[i]) / (close[i] * atr[i])

        direction = 0
        confidence = 0

        if dev < -dev_mult:  # below VWAP
            direction = 1  # Long to revert
            confidence += 1
            if rsi[i] < 35:
                confidence += 1
            if dev < -dev_mult * 1.5:
                confidence += 1  # very far below
            if close[i] > close[i - 1]:
                confidence += 1  # starting to revert
            if taker_ratio[i] < 0.45:
                confidence += 1  # selling pressure declining

        elif dev > dev_mult:  # above VWAP
            direction = -1
            confidence += 1
            if rsi[i] > 65:
                confidence += 1
            if dev > dev_mult * 1.5:
                confidence += 1
            if close[i] < close[i - 1]:
                confidence += 1
            if taker_ratio[i] > 0.55:
                confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            sizes[i] = atr[i]

    return dirs, confs, sizes


@numba.jit(nopython=True, cache=True)
def scan_volume_enhanced_mr(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period,
    rsi_period, vol_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
    vol_requirement,  # require volume > vol_requirement * vol_sma
):
    """D5: Volume-Enhanced Rubber Band.

    Same as MR but REQUIRE high volume as hard filter (not just confidence).
    """
    n = min(len(close), len(btc_close))
    dirs = np.zeros(n)
    confs = np.zeros(n)
    moves = np.zeros(n)
    atrs = np.full(n, np.nan)

    rsi = _compute_rsi_inline(close[:n], rsi_period)
    vol_sma = _compute_sma_inline(volume[:n], vol_period)
    atr_pct = compute_atr_pct(high[:n], low[:n], close[:n], atr_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    start = max(lookback, btc_mom_period, rsi_period, vol_period, atr_period)

    for i in range(start, n):
        ret = close[i] / close[i - lookback] - 1.0

        if not np.isnan(atr_pct[i]):
            thr = max(threshold_floor, min(atr_multiplier * atr_pct[i], threshold_cap))
            atrs[i] = atr_pct[i]
        else:
            thr = threshold
            atrs[i] = abs(ret)

        if abs(ret) <= thr:
            continue

        # HARD volume filter — must meet requirement
        if np.isnan(vol_sma[i]) or volume[i] < vol_requirement * vol_sma[i]:
            continue

        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

        direction = 0
        confidence = 0

        if ret < -thr and btc_mom < 0:
            direction = 1
            if rsi[i] < 35:
                confidence += 1
            if taker_ratio[i] < 0.44:
                confidence += 1
            if btc_mom < -0.01:
                confidence += 1
            confidence += 1  # volume already confirmed

        elif ret > thr and btc_mom > 0:
            direction = -1
            if rsi[i] > 65:
                confidence += 1
            if taker_ratio[i] > 0.56:
                confidence += 1
            if btc_mom > 0.01:
                confidence += 1
            confidence += 1

        # min_confidence=2 since volume is already required
        if confidence < 2:
            direction = 0

        if direction != 0:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            moves[i] = abs(ret)

    return dirs, confs, moves, atrs


@numba.jit(nopython=True, cache=True)
def scan_volume_breakout(close, high, low, volume, taker_buy,
                         lookback, vol_period, vol_mult, atr_period, rsi_period):
    """D6: Volume Breakout Momentum.

    Price breaks N-bar high/low WITH volume confirmation → follow breakout.
    """
    n = len(close)
    dirs = np.zeros(n)
    confs = np.zeros(n)
    sizes = np.zeros(n)

    vol_sma = _compute_sma_inline(volume, vol_period)
    atr = compute_atr_pct(high, low, close, atr_period)
    rsi = _compute_rsi_inline(close, rsi_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    start = max(lookback, vol_period, atr_period, rsi_period) + 1

    for i in range(start, n):
        if np.isnan(vol_sma[i]) or np.isnan(atr[i]):
            continue
        if volume[i] < vol_mult * vol_sma[i]:
            continue  # need volume spike

        # Check for N-bar breakout
        highest = close[i]
        lowest = close[i]
        for j in range(i - lookback, i):
            if close[j] > highest:
                highest = close[j]
            if close[j] < lowest:
                lowest = close[j]

        direction = 0
        confidence = 0

        # Breakout above N-bar high → LONG momentum
        if close[i] >= highest:
            direction = 1
            confidence += 1  # breakout itself
            if taker_ratio[i] > 0.52:
                confidence += 1  # buyers dominating
            if rsi[i] > 55 and rsi[i] < 85:
                confidence += 1  # momentum but not exhausted
            if close[i] > close[i - 1]:
                confidence += 1

        # Breakout below N-bar low → SHORT momentum
        elif close[i] <= lowest:
            direction = -1
            confidence += 1
            if taker_ratio[i] < 0.48:
                confidence += 1
            if rsi[i] < 45 and rsi[i] > 15:
                confidence += 1
            if close[i] < close[i - 1]:
                confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            sizes[i] = atr[i]

    return dirs, confs, sizes


# ═══════════════════════════════════════════════════════════════════
# Trade Simulator (Per-Trade Independent Analysis)
# ═══════════════════════════════════════════════════════════════════
def simulate_trades(
    sym_close: dict, sym_high: dict, sym_low: dict,
    sym_dirs: dict, sym_sizes: dict,
    target_mult: float = 1.5,
    stop_mult: float = 0.382,
    max_hold: int = 25,
    trail_active: float = 0.5,
    trail_factor: float = 0.3,
    funding_per_bar: float = FUNDING_PER_BAR_15m,
) -> list[dict]:
    """Simulate each trade independently (no portfolio constraints).

    Args:
        sym_sizes: dict[sym] → array of sizing values (move_size or ATR).
    """
    trades = []
    for sym in sorted(sym_close.keys()):
        close = sym_close[sym]
        high = sym_high[sym]
        low = sym_low[sym]
        d_arr = sym_dirs[sym]
        s_arr = sym_sizes[sym]
        n = len(close)

        for entry_bar in range(n):
            if d_arr[entry_bar] == 0.0:
                continue

            d = int(d_arr[entry_bar])
            size = float(s_arr[entry_bar])
            if size <= 0:
                continue

            ep = close[entry_bar]
            target = size * target_mult
            stop = size * stop_mult
            best_pnl = 0.0
            mfe = 0.0
            mae = 0.0
            exit_bar = -1
            exit_reason = 'time'
            exit_gain = 0.0

            for j in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
                if d == 1:
                    gain = (close[j] - ep) / ep
                    worst = (low[j] - ep) / ep
                    best = (high[j] - ep) / ep
                else:
                    gain = (ep - close[j]) / ep
                    worst = (ep - high[j]) / ep
                    best = (ep - low[j]) / ep

                if best > mfe:
                    mfe = best
                if worst < mae:
                    mae = worst
                if best > best_pnl:
                    best_pnl = best

                # SL priority (intra-bar)
                if worst < -stop:
                    exit_bar = j
                    exit_reason = 'stop'
                    exit_gain = -stop
                    break
                elif best >= target:
                    exit_bar = j
                    exit_reason = 'target'
                    exit_gain = target
                    break
                elif best_pnl > target * trail_active and worst < best_pnl * trail_factor:
                    exit_bar = j
                    exit_reason = 'trail'
                    exit_gain = best_pnl * trail_factor
                    break
                elif j - entry_bar >= max_hold:
                    exit_bar = j
                    exit_reason = 'time'
                    exit_gain = gain
                    break

            if exit_bar == -1:
                continue

            trades.append({
                'raw_gain': exit_gain,
                'bars_held': exit_bar - entry_bar,
                'mfe': mfe,
                'mae': mae,
                'exit_reason': exit_reason,
            })

    return trades


def compute_stats(trades: list[dict], cost_per_side: float = 0.0,
                  funding_per_bar: float = FUNDING_PER_BAR_15m) -> dict:
    """Compute PF, WR, EV, etc. from raw trades."""
    if not trades:
        return {'n': 0, 'wr': 0, 'pf': 0, 'rr': 0, 'ev': 0}

    pnls = []
    for t in trades:
        raw = t['raw_gain']
        cost = 2 * cost_per_side
        funding = funding_per_bar * t['bars_held']
        pnls.append(raw - cost - funding)

    pnls = np.array(pnls)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n * 100 if n > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    pf = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else 999
    ev = float(np.mean(pnls))

    # Exit breakdown
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    return {
        'n': n, 'wr': wr, 'avg_win': avg_win * 100, 'avg_loss': avg_loss * 100,
        'rr': rr, 'pf': pf, 'ev': ev * 100, 'reasons': reasons,
    }


def print_result(label: str, stats: dict):
    """Print one-line result."""
    n = stats['n']
    if n == 0:
        print(f"  {label:<35s}  N=0 (no signals)")
        return
    exits = stats.get('reasons', {})
    stop_pct = exits.get('stop', 0) / n * 100 if n > 0 else 0
    print(f"  {label:<35s}  N={n:>7,d}  WR={stats['wr']:5.1f}%"
          f"  R:R={stats['rr']:.2f}  PF={stats['pf']:.3f}"
          f"  EV={stats['ev']:+.3f}bp  SL%={stop_pct:.0f}%")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    commit = os.popen("git rev-parse --short HEAD").read().strip()
    now = datetime.now(timezone.utc)

    print("=" * 80)
    print(f"  R23: NEW ALPHA SEARCH — 4 DIMENSIONS")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Data: {DATA_DAYS}d (6yr), {len(SYMBOLS_50)} symbols")
    print(f"  All variants: ZERO cost, 1x leverage, zero lookahead")
    print("=" * 80)

    # ═══ Load & Resample ═══
    data_15m = load_all_15m()
    symbols = sorted(data_15m.keys())
    n_bars_15m = min(len(data_15m[s]['close']) for s in symbols)

    print(f"\n  Resampling 15m → 1h and 4h...")
    data_1h = resample_all(data_15m, 4)
    data_4h = resample_all(data_15m, 16)
    n_bars_1h = min(len(data_1h[s]['close']) for s in symbols)
    n_bars_4h = min(len(data_4h[s]['close']) for s in symbols)
    print(f"  1h: {n_bars_1h:,} bars | 4h: {n_bars_4h:,} bars")

    btc_15m = data_15m['BTCUSDT']['close'][:n_bars_15m]
    btc_1h = data_1h['BTCUSDT']['close'][:n_bars_1h]
    btc_4h = data_4h['BTCUSDT']['close'][:n_bars_4h]

    # Helper dicts for trade simulator
    def to_dicts(data, n_bars):
        sc = {s: data[s]['close'][:n_bars] for s in symbols}
        sh = {s: data[s]['high'][:n_bars] for s in symbols}
        sl_ = {s: data[s]['low'][:n_bars] for s in symbols}
        return sc, sh, sl_

    sc_15, sh_15, sl_15 = to_dicts(data_15m, n_bars_15m)
    sc_1h, sh_1h, sl_1h = to_dicts(data_1h, n_bars_1h)
    sc_4h, sh_4h, sl_4h = to_dicts(data_4h, n_bars_4h)

    all_results = []

    def run_variant(label, desc, sym_close, sym_high, sym_low,
                    sym_dirs, sym_sizes,
                    target_mult=1.5, stop_mult=0.382, max_hold=25,
                    trail_active=0.5, trail_factor=0.3,
                    funding_per_bar=FUNDING_PER_BAR_15m):
        trades = simulate_trades(
            sym_close, sym_high, sym_low, sym_dirs, sym_sizes,
            target_mult, stop_mult, max_hold, trail_active, trail_factor,
            funding_per_bar)
        stats = compute_stats(trades, cost_per_side=0.0, funding_per_bar=funding_per_bar)
        print_result(label, stats)
        all_results.append({'label': label, 'desc': desc, **stats})
        return stats

    # ══════════════════════════════════════════════════════════════
    # PHASE A: TIMEFRAME
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  PHASE A: HIGHER TIMEFRAME MEAN-REVERSION")
    print(f"  Does MR work better on 1h or 4h?")
    print(f"{'═' * 80}\n")

    # Scan MR on each timeframe
    # Scale lookback/periods for higher TFs:
    # 1h: same lookback=8 means 8 hours (vs 2h on 15m)
    # 4h: lookback=8 means 32 hours
    # Adjust btc_mom and ATR periods proportionally
    tf_configs = [
        ('A1-MR-15m', '15m MR baseline (R22 reference)',
         data_15m, btc_15m, n_bars_15m, sc_15, sh_15, sl_15,
         8, 48, ATR_PERIOD, 3, FUNDING_PER_BAR_15m, 25),
        ('A2-MR-1h', '1h MR (lookback=8h)',
         data_1h, btc_1h, n_bars_1h, sc_1h, sh_1h, sl_1h,
         8, 12, 12, 2, FUNDING_PER_BAR_15m * 4, 25),
        ('A3-MR-1h-c3', '1h MR conf≥3',
         data_1h, btc_1h, n_bars_1h, sc_1h, sh_1h, sl_1h,
         8, 12, 12, 3, FUNDING_PER_BAR_15m * 4, 25),
        ('A4-MR-4h', '4h MR (lookback=32h)',
         data_4h, btc_4h, n_bars_4h, sc_4h, sh_4h, sl_4h,
         8, 3, 3, 2, FUNDING_PER_BAR_15m * 16, 12),
    ]

    for tf_label, tf_desc, tf_data, tf_btc, n_bars, sc, sh, sl_, \
            lb, btc_mom_p, atr_p, min_c, fpb, mh in tf_configs:
        sym_dirs = {}
        sym_sizes = {}
        for s in symbols:
            d = tf_data[s]
            c = d['close'][:n_bars]
            h = d['high'][:n_bars]
            lo = d['low'][:n_bars]
            v = d['volume'][:n_bars]
            t = d['taker_buy'][:n_bars]
            btc = tf_btc[:n_bars]

            dirs, _, moves, _ = scan_mr_momentum(
                c, h, lo, v, t, btc,
                lb, THRESHOLD, btc_mom_p, min_c, RSI_PERIOD,
                VOL_PERIOD, atr_p, ATR_MULT, THR_FLOOR, THR_CAP,
                False, True)
            sym_dirs[s] = dirs
            sym_sizes[s] = moves

        run_variant(tf_label, tf_desc, sc, sh, sl_, sym_dirs, sym_sizes,
                    funding_per_bar=fpb, max_hold=mh)

    # ══════════════════════════════════════════════════════════════
    # PHASE B: MOMENTUM
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  PHASE B: MOMENTUM (FOLLOW THE TREND)")
    print(f"  Does following the move beat fading it?")
    print(f"{'═' * 80}\n")

    mom_configs = [
        # label, desc, flip, btc_filter, conf, tgt, stop, max_hold, tf
        ('B1-Mom-15m', '15m momentum, BTC aligned, baseline exits',
         True, True, 3, 1.5, 0.382, 25, '15m'),
        ('B2-Mom-15m-asym', '15m momentum, asymmetric (2.5x tgt, 0.15 stop)',
         True, True, 3, 2.5, 0.15, 30, '15m'),
        ('B3-Mom-15m-noBTC', '15m momentum, no BTC filter, conf≥2',
         True, False, 2, 1.5, 0.382, 25, '15m'),
        ('B4-Mom-15m-trail', '15m momentum, trail-heavy (3x tgt, 0.2 stop)',
         True, True, 2, 3.0, 0.20, 40, '15m'),
        ('B5-Mom-1h', '1h momentum, BTC aligned',
         True, True, 2, 1.5, 0.382, 25, '1h'),
    ]

    for m_label, m_desc, flip, btc_filt, conf, tgt, stp, mh, tf in mom_configs:
        if tf == '15m':
            tf_data, tf_btc, n_bars = data_15m, btc_15m, n_bars_15m
            sc, sh, sl_ = sc_15, sh_15, sl_15
            lb, btc_p, atr_p = 8, 48, ATR_PERIOD
            fpb = FUNDING_PER_BAR_15m
        else:
            tf_data, tf_btc, n_bars = data_1h, btc_1h, n_bars_1h
            sc, sh, sl_ = sc_1h, sh_1h, sl_1h
            lb, btc_p, atr_p = 8, 12, 12
            fpb = FUNDING_PER_BAR_15m * 4

        sym_dirs = {}
        sym_sizes = {}
        for s in symbols:
            d = tf_data[s]
            c = d['close'][:n_bars]
            h = d['high'][:n_bars]
            lo = d['low'][:n_bars]
            v = d['volume'][:n_bars]
            t = d['taker_buy'][:n_bars]
            btc = tf_btc[:n_bars]

            dirs, _, moves, _ = scan_mr_momentum(
                c, h, lo, v, t, btc,
                lb, THRESHOLD, btc_p, conf, RSI_PERIOD,
                VOL_PERIOD, atr_p, ATR_MULT, THR_FLOOR, THR_CAP,
                flip, btc_filt)
            sym_dirs[s] = dirs
            sym_sizes[s] = moves

        run_variant(m_label, m_desc, sc, sh, sl_, sym_dirs, sym_sizes,
                    target_mult=tgt, stop_mult=stp, max_hold=mh,
                    funding_per_bar=fpb)

    # ══════════════════════════════════════════════════════════════
    # PHASE C: REGIME-CONDITIONAL
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  PHASE C: REGIME-CONDITIONAL")
    print(f"  Range → MR, Trend → Momentum, Adaptive?")
    print(f"{'═' * 80}\n")

    # Pre-compute efficiency ratios per symbol
    ER_PERIOD = 48  # 12 hours of 15m bars
    print(f"  Computing Efficiency Ratio (period={ER_PERIOD})...")
    sym_er = {}
    for s in symbols:
        sym_er[s] = compute_efficiency_ratio(data_15m[s]['close'][:n_bars_15m], ER_PERIOD)

    # Pre-compute MR and Momentum entries (15m)
    print(f"  Scanning MR + Momentum entries (15m)...")
    sym_mr_dirs = {}
    sym_mr_moves = {}
    sym_mom_dirs = {}
    sym_mom_moves = {}

    for s in symbols:
        d = data_15m[s]
        c = d['close'][:n_bars_15m]
        h = d['high'][:n_bars_15m]
        lo = d['low'][:n_bars_15m]
        v = d['volume'][:n_bars_15m]
        t = d['taker_buy'][:n_bars_15m]
        btc = btc_15m

        dirs_mr, _, moves_mr, _ = scan_mr_momentum(
            c, h, lo, v, t, btc,
            8, THRESHOLD, 48, 3, RSI_PERIOD, VOL_PERIOD,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            False, True)
        sym_mr_dirs[s] = dirs_mr
        sym_mr_moves[s] = moves_mr

        dirs_mom, _, moves_mom, _ = scan_mr_momentum(
            c, h, lo, v, t, btc,
            8, THRESHOLD, 48, 2, RSI_PERIOD, VOL_PERIOD,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            True, True)
        sym_mom_dirs[s] = dirs_mom
        sym_mom_moves[s] = moves_mom

    # C1: Range-only MR (ER < 0.3)
    sym_c1_dirs = {}
    for s in symbols:
        fd, _, _ = apply_regime_filter(
            sym_mr_dirs[s], None, None, sym_er[s], 0.3, keep_below=True)
        sym_c1_dirs[s] = fd
    run_variant('C1-Range-MR-0.3', 'MR only when ER<0.3 (range)',
                sc_15, sh_15, sl_15, sym_c1_dirs, sym_mr_moves)

    # C2: Range-only MR (ER < 0.2) — stricter
    sym_c2_dirs = {}
    for s in symbols:
        fd, _, _ = apply_regime_filter(
            sym_mr_dirs[s], None, None, sym_er[s], 0.2, keep_below=True)
        sym_c2_dirs[s] = fd
    run_variant('C2-Range-MR-0.2', 'MR only when ER<0.2 (choppy)',
                sc_15, sh_15, sl_15, sym_c2_dirs, sym_mr_moves)

    # C3: Trend-only Momentum (ER > 0.5)
    sym_c3_dirs = {}
    for s in symbols:
        fd, _, _ = apply_regime_filter(
            sym_mom_dirs[s], None, None, sym_er[s], 0.5, keep_below=False)
        sym_c3_dirs[s] = fd
    run_variant('C3-Trend-Mom-0.5', 'Momentum only when ER>0.5 (trending)',
                sc_15, sh_15, sl_15, sym_c3_dirs, sym_mom_moves,
                target_mult=2.0, stop_mult=0.20)

    # C4: Trend-only Momentum (ER > 0.4) — less strict
    sym_c4_dirs = {}
    for s in symbols:
        fd, _, _ = apply_regime_filter(
            sym_mom_dirs[s], None, None, sym_er[s], 0.4, keep_below=False)
        sym_c4_dirs[s] = fd
    run_variant('C4-Trend-Mom-0.4', 'Momentum only when ER>0.4',
                sc_15, sh_15, sl_15, sym_c4_dirs, sym_mom_moves,
                target_mult=2.0, stop_mult=0.20)

    # C5: Adaptive — Range → MR, Trend → Momentum
    sym_c5_dirs = {}
    sym_c5_sizes = {}
    for s in symbols:
        c5_dirs = np.zeros_like(sym_mr_dirs[s])
        c5_sizes = np.zeros_like(sym_mr_moves[s])
        er = sym_er[s]
        for i in range(len(c5_dirs)):
            if np.isnan(er[i]):
                continue
            # Range: use MR
            if er[i] < 0.3 and sym_mr_dirs[s][i] != 0:
                c5_dirs[i] = sym_mr_dirs[s][i]
                c5_sizes[i] = sym_mr_moves[s][i]
            # Trend: use Momentum
            elif er[i] > 0.5 and sym_mom_dirs[s][i] != 0:
                c5_dirs[i] = sym_mom_dirs[s][i]
                c5_sizes[i] = sym_mom_moves[s][i]
        sym_c5_dirs[s] = c5_dirs
        sym_c5_sizes[s] = c5_sizes
    run_variant('C5-Adaptive', 'ER<0.3→MR, ER>0.5→Mom',
                sc_15, sh_15, sl_15, sym_c5_dirs, sym_c5_sizes,
                target_mult=1.5, stop_mult=0.30)

    # C6: Adaptive with wider regime bands
    sym_c6_dirs = {}
    sym_c6_sizes = {}
    for s in symbols:
        c6_dirs = np.zeros_like(sym_mr_dirs[s])
        c6_sizes = np.zeros_like(sym_mr_moves[s])
        er = sym_er[s]
        for i in range(len(c6_dirs)):
            if np.isnan(er[i]):
                continue
            if er[i] < 0.25 and sym_mr_dirs[s][i] != 0:
                c6_dirs[i] = sym_mr_dirs[s][i]
                c6_sizes[i] = sym_mr_moves[s][i]
            elif er[i] > 0.45 and sym_mom_dirs[s][i] != 0:
                c6_dirs[i] = sym_mom_dirs[s][i]
                c6_sizes[i] = sym_mom_moves[s][i]
        sym_c6_dirs[s] = c6_dirs
        sym_c6_sizes[s] = c6_sizes
    run_variant('C6-Adaptive-wide', 'ER<0.25→MR, ER>0.45→Mom',
                sc_15, sh_15, sl_15, sym_c6_dirs, sym_c6_sizes,
                target_mult=1.5, stop_mult=0.30)

    # ══════════════════════════════════════════════════════════════
    # PHASE D: VOLUME-BASED ALPHA
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  PHASE D: VOLUME-BASED ALPHA")
    print(f"  Can order flow / volume patterns generate edge?")
    print(f"{'═' * 80}\n")

    # D1: Volume Climax Reversal
    print(f"  Scanning D1: Volume Climax Reversal...")
    sym_d1_dirs = {}
    sym_d1_sizes = {}
    for s in symbols:
        d = data_15m[s]
        c = d['close'][:n_bars_15m]
        h = d['high'][:n_bars_15m]
        lo = d['low'][:n_bars_15m]
        v = d['volume'][:n_bars_15m]
        t = d['taker_buy'][:n_bars_15m]
        dirs, _, sizes = scan_volume_climax(c, h, lo, v, t,
                                            VOL_PERIOD, 2.5, ATR_PERIOD, RSI_PERIOD)
        sym_d1_dirs[s] = dirs
        sym_d1_sizes[s] = sizes

    run_variant('D1-VolClimax-2.5x', 'Volume climax reversal (>2.5x SMA)',
                sc_15, sh_15, sl_15, sym_d1_dirs, sym_d1_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=20)

    # D1b: Stronger climax requirement
    sym_d1b_dirs = {}
    sym_d1b_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_volume_climax(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            VOL_PERIOD, 3.5, ATR_PERIOD, RSI_PERIOD)
        sym_d1b_dirs[s] = dirs
        sym_d1b_sizes[s] = sizes
    run_variant('D1b-VolClimax-3.5x', 'Volume climax (>3.5x SMA, stricter)',
                sc_15, sh_15, sl_15, sym_d1b_dirs, sym_d1b_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=20)

    # D2: Taker Exhaustion
    print(f"  Scanning D2: Taker Exhaustion...")
    sym_d2_dirs = {}
    sym_d2_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_taker_exhaustion(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            VOL_PERIOD, ATR_PERIOD, RSI_PERIOD,
            0.35, 0.65, 20)
        sym_d2_dirs[s] = dirs
        sym_d2_sizes[s] = sizes
    run_variant('D2-TakerExhaust', 'Taker exhaustion (ratio <0.35/>0.65)',
                sc_15, sh_15, sl_15, sym_d2_dirs, sym_d2_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=20)

    # D2b: More moderate thresholds
    sym_d2b_dirs = {}
    sym_d2b_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_taker_exhaustion(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            VOL_PERIOD, ATR_PERIOD, RSI_PERIOD,
            0.38, 0.62, 20)
        sym_d2b_dirs[s] = dirs
        sym_d2b_sizes[s] = sizes
    run_variant('D2b-TakerMod', 'Taker exhaustion moderate (0.38/0.62)',
                sc_15, sh_15, sl_15, sym_d2b_dirs, sym_d2b_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=20)

    # D3: CVD Divergence
    print(f"  Scanning D3: CVD Divergence...")
    sym_d3_dirs = {}
    sym_d3_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_cvd_divergence(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            48, ATR_PERIOD, RSI_PERIOD)
        sym_d3_dirs[s] = dirs
        sym_d3_sizes[s] = sizes
    run_variant('D3-CVD-Div-48', 'CVD divergence (lookback=48, 12h)',
                sc_15, sh_15, sl_15, sym_d3_dirs, sym_d3_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=20)

    # D3b: Shorter lookback for divergence
    sym_d3b_dirs = {}
    sym_d3b_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_cvd_divergence(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            24, ATR_PERIOD, RSI_PERIOD)
        sym_d3b_dirs[s] = dirs
        sym_d3b_sizes[s] = sizes
    run_variant('D3b-CVD-Div-24', 'CVD divergence (lookback=24, 6h)',
                sc_15, sh_15, sl_15, sym_d3b_dirs, sym_d3b_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=20)

    # D4: VWAP Mean-Reversion
    print(f"  Scanning D4: VWAP Deviation...")
    sym_d4_dirs = {}
    sym_d4_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_vwap_deviation(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            96, ATR_PERIOD, RSI_PERIOD, 2.0)
        sym_d4_dirs[s] = dirs
        sym_d4_sizes[s] = sizes
    run_variant('D4-VWAP-96-2.0', 'VWAP MR (period=96/24h, dev>2.0 ATR)',
                sc_15, sh_15, sl_15, sym_d4_dirs, sym_d4_sizes,
                target_mult=1.5, stop_mult=1.0, max_hold=25)

    # D4b: Weekly VWAP
    sym_d4b_dirs = {}
    sym_d4b_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_vwap_deviation(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            672, ATR_PERIOD, RSI_PERIOD, 2.5)
        sym_d4b_dirs[s] = dirs
        sym_d4b_sizes[s] = sizes
    run_variant('D4b-VWAP-672-2.5', 'Weekly VWAP MR (period=672/7d, dev>2.5)',
                sc_15, sh_15, sl_15, sym_d4b_dirs, sym_d4b_sizes,
                target_mult=1.5, stop_mult=1.0, max_hold=30)

    # D5: Volume-Enhanced MR (rubber band + hard volume filter)
    print(f"  Scanning D5: Volume-Enhanced MR...")
    sym_d5_dirs = {}
    sym_d5_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, moves, _ = scan_volume_enhanced_mr(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m], btc_15m,
            8, THRESHOLD, 48, RSI_PERIOD, VOL_PERIOD,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            2.0)
        sym_d5_dirs[s] = dirs
        sym_d5_sizes[s] = moves
    run_variant('D5-VolEnhanced-2x', 'MR + require vol>2x SMA',
                sc_15, sh_15, sl_15, sym_d5_dirs, sym_d5_sizes)

    # D5b: Stronger volume requirement
    sym_d5b_dirs = {}
    sym_d5b_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, moves, _ = scan_volume_enhanced_mr(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m], btc_15m,
            8, THRESHOLD, 48, RSI_PERIOD, VOL_PERIOD,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            3.0)
        sym_d5b_dirs[s] = dirs
        sym_d5b_sizes[s] = moves
    run_variant('D5b-VolEnhanced-3x', 'MR + require vol>3x SMA',
                sc_15, sh_15, sl_15, sym_d5b_dirs, sym_d5b_sizes)

    # D6: Volume Breakout Momentum
    print(f"  Scanning D6: Volume Breakout Momentum...")
    sym_d6_dirs = {}
    sym_d6_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_volume_breakout(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            20, VOL_PERIOD, 2.0, ATR_PERIOD, RSI_PERIOD)
        sym_d6_dirs[s] = dirs
        sym_d6_sizes[s] = sizes
    run_variant('D6-VolBreakout-20', 'Vol breakout (20-bar, >2x SMA)',
                sc_15, sh_15, sl_15, sym_d6_dirs, sym_d6_sizes,
                target_mult=2.0, stop_mult=1.0, max_hold=25)

    # D6b: Longer lookback
    sym_d6b_dirs = {}
    sym_d6b_sizes = {}
    for s in symbols:
        d = data_15m[s]
        dirs, _, sizes = scan_volume_breakout(
            d['close'][:n_bars_15m], d['high'][:n_bars_15m],
            d['low'][:n_bars_15m], d['volume'][:n_bars_15m],
            d['taker_buy'][:n_bars_15m],
            48, VOL_PERIOD, 2.5, ATR_PERIOD, RSI_PERIOD)
        sym_d6b_dirs[s] = dirs
        sym_d6b_sizes[s] = sizes
    run_variant('D6b-VolBreak-48', 'Vol breakout (48-bar/12h, >2.5x)',
                sc_15, sh_15, sl_15, sym_d6b_dirs, sym_d6b_sizes,
                target_mult=2.5, stop_mult=1.0, max_hold=30)

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY — ALL VARIANTS RANKED BY PF (zero cost)")
    print(f"  Runtime: {elapsed:.0f}s")
    print(f"{'═' * 80}\n")

    ranked = sorted(all_results, key=lambda x: x.get('pf', 0), reverse=True)
    print(f"  {'Label':<35s}  {'N':>8s}  {'WR%':>5s}  {'R:R':>5s}"
          f"  {'PF':>6s}  {'EV(bp)':>7s}  {'SL%':>4s}")
    print(f"  {'─' * 80}")

    best_pf = 0
    for r in ranked:
        n = r.get('n', 0)
        if n == 0:
            continue
        exits = r.get('reasons', {})
        sl_pct = exits.get('stop', 0) / n * 100 if n > 0 else 0
        pf = r.get('pf', 0)
        marker = " ★" if pf >= 1.0 else ""
        print(f"  {r['label']:<35s}  {n:>8,d}  {r['wr']:5.1f}  {r['rr']:5.2f}"
              f"  {pf:6.3f}  {r['ev']:+7.3f}  {sl_pct:4.0f}%{marker}")
        if pf > best_pf:
            best_pf = pf

    print()
    if best_pf >= 1.0:
        print(f"  ★ POSITIVE EDGE FOUND — best PF = {best_pf:.3f}")
        print(f"  → Deep-dive recommended: portfolio sim, cost sensitivity, yearly stability")
    else:
        print(f"  ❌ NO VARIANT achieves PF ≥ 1.0 at zero cost.")
        delta = 1.0 - best_pf
        print(f"  Best PF = {best_pf:.3f} (gap to break-even: {delta:.3f})")
        print(f"  → Consider: different asset classes, different data (orderbook, funding rate)")

    # ═══ Save Results ═══
    results_path = Path(__file__).parent.parent.parent / 'logs' / 'research' / 'alpha_search_r23_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable items
    for r in all_results:
        r.pop('reasons', None)
    with open(results_path, 'w') as f:
        json.dump({
            'commit': commit,
            'timestamp': now.isoformat(),
            'variants': all_results,
            'best_pf': best_pf,
        }, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")


if __name__ == "__main__":
    main()
