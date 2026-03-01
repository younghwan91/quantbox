#!/usr/bin/env python3
"""Download actual data and run parity backtest for the live trading period.

Downloads 60 days of 15m + 4h data for the exact 15 symbols traded live
(+ BTCUSDT), then runs a bar-by-bar backtest that mirrors live on_bar()
logic EXACTLY:
  - Risk-Off only entry (no Risk-On path)
  - ATR adaptive threshold (atr_period=48, atr_multiplier=2.0)
  - Confidence scoring with climax volume filter
  - ATR-based dynamic stop loss (atr_stop_mult=2.0, floor=0.5%, cap=4.0%)
  - Intra-bar SL/TP using high/low
  - PBT partial close (50% at TP1, runner with trailing)
  - MTF 4h SMA(12) confirmation filter
  - max_concurrent=2 slot limit (portfolio-level)

Usage:
    cd /home/user/Documents/git/quantbox
    .venv/bin/python scripts/backtest_live_period.py
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband.signals import (
    compute_atr_pct,
    _compute_rsi_inline,
    _compute_sma_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
)

# ═══════════════════════════════════════════════════════════════════
# LIVE PARAMS (from .env — verified)
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032  # fallback (ATR adaptive overrides)
TARGET_RETRACE = 1.5
STOP_MULT = 0.382  # legacy fallback
MAX_HOLD = 25
STOP_DECAY = 0.0
BTC_DIRECTIONAL_PERIOD = 48
BTC_REGIME_PERIOD = 96
ASSET_TREND_PERIOD = 192
MIN_CONFIDENCE = 3
BASE_LEVERAGE = 1.5
CONFIDENCE_LEVERAGE_STEP = 0.0
MAX_LEVERAGE = 20.0
RSI_PERIOD = 14
VOL_PERIOD = 20
STOCH_ENABLED = False
MACD_ENABLED = False
CLIMAX_ENABLED = False  # v12.6 feature — NOT in v12.5.3 (live log commit ca2ceba)
CLIMAX_VOL_MULT = 3.0
ATR_PERIOD = 48
ATR_MULTIPLIER = 2.0
THRESHOLD_FLOOR = 0.008
THRESHOLD_CAP = 0.06
MTF_SMA_PERIOD = 12
MTF_ENABLED = True
PARTIAL_EXIT_ENABLED = True
PARTIAL_EXIT_FRACTION = 0.5
RUNNER_MAX_HOLD_MULT = 3.0
MAX_CONCURRENT = 2
ATR_STOP_ENABLED = False  # EVIDENCE: all 25 live trades show eff_stop=0.3820 = fixed stop
ATR_STOP_MULT = 2.0
ATR_STOP_FLOOR = 0.005
ATR_STOP_CAP = 0.04
MTF_LEVERAGE_MULT = 2.5  # MTF confirmed → 1.5 × 2.5 = 3.75 → ceil(4) = 4x
EFF_LEVERAGE = 4  # what live actually used (all 25 trades show lev=4x)
COST_PER_SIDE = 0.0004  # 0.04% taker fee

# ═══════════════════════════════════════════════════════════════════
# SYMBOLS
# ═══════════════════════════════════════════════════════════════════
LIVE_SYMBOLS = [
    'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT',
    '1000PEPEUSDT', 'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT',
    'ATOMUSDT', 'HBARUSDT', 'ICPUSDT', 'FILUSDT', 'TRXUSDT',
    'SUIUSDT', 'TONUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT',
    'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT', 'ETCUSDT',
    'GRTUSDT', 'STXUSDT', 'EGLDUSDT', 'AXSUSDT', 'SANDUSDT',
    'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT',
    'LDOUSDT', 'COMPUSDT', 'SNXUSDT', 'VETUSDT', 'APEUSDT',
    'ENAUSDT', 'TAOUSDT', 'WLDUSDT', 'WIFUSDT',
]
BTC_SYMBOL = 'BTCUSDT'
ALL_SYMBOLS = [BTC_SYMBOL] + LIVE_SYMBOLS

# Live period (UTC): Feb 22 16:32 → Feb 26 ~now
LIVE_START = datetime(2026, 2, 22, 16, 30, tzinfo=timezone.utc)
LIVE_END = datetime(2026, 2, 26, 23, 59, tzinfo=timezone.utc)

DOWNLOAD_DAYS = 60  # enough warmup for all indicators
CACHE_DIR = ROOT / 'data' / 'cache_parity'


# ═══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD (with local caching)
# ═══════════════════════════════════════════════════════════════════
def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}_{DOWNLOAD_DAYS}d.npz"


def download_klines(client, symbol: str, interval: str, days: int) -> list:
    """Download klines from Binance futures API in chunks."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    all_klines = []
    chunk_ms = 1500 * (15 * 60 * 1000 if interval == '15m' else 4 * 60 * 60 * 1000)

    cur = start_ms
    retries = 0
    while cur < end_ms:
        chunk_end = min(cur + chunk_ms, end_ms)
        try:
            klines = client.futures_klines(
                symbol=symbol, interval=interval,
                startTime=cur, endTime=chunk_end, limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            cur = int(klines[-1][0]) + 1
            retries = 0
            time.sleep(0.25)  # avoid rate limit (2400 req/min)
        except Exception as e:
            retries += 1
            err_str = str(e)
            if '-1003' in err_str:  # rate limit
                wait = min(60, 10 * retries)
                print(f"\n    ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif retries > 3:
                print(f"\n    ✗ {symbol} {interval} failed after 3 retries: {e}")
                return []
            else:
                time.sleep(2 ** retries)

    # Remove last incomplete bar
    if all_klines and int(all_klines[-1][6]) > end_ms:
        all_klines = all_klines[:-1]

    return all_klines


def klines_to_arrays(klines: list) -> dict:
    """Convert raw Binance klines to numpy arrays."""
    close = np.array([float(k[4]) for k in klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in klines], dtype=np.float64)
    volume = np.array([float(k[5]) for k in klines], dtype=np.float64)
    taker_buy = np.array([float(k[9]) for k in klines], dtype=np.float64)
    timestamps = np.array([int(k[0]) for k in klines], dtype=np.int64)
    return {
        'close': close, 'high': high, 'low': low,
        'volume': volume, 'taker_buy': taker_buy,
        'timestamps': timestamps,
    }


def download_all(client) -> dict:
    """Download 15m + 4h data for all symbols (with local npz caching)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for i, sym in enumerate(ALL_SYMBOLS):
        print(f"  [{i+1}/{len(ALL_SYMBOLS)}] {sym}", end='')

        # Check 15m cache
        cache_15m = _cache_path(sym, '15m')
        if cache_15m.exists():
            d15 = dict(np.load(cache_15m))
            print(f"  ✓ 15m cached: {len(d15['close']):,} bars", end='')
        else:
            klines_15m = download_klines(client, sym, '15m', DOWNLOAD_DAYS)
            if len(klines_15m) < 200:
                print(f"  ✗ 15m: only {len(klines_15m)} bars, skipping")
                continue
            d15 = klines_to_arrays(klines_15m)
            np.savez_compressed(cache_15m, **d15)
            print(f"  ✓ 15m: {len(d15['close']):,} bars", end='')

        # Check 4h cache
        cache_4h = _cache_path(sym, '4h')
        if cache_4h.exists():
            d4h = np.load(cache_4h)
            close_4h = d4h['close']
            ts_4h = d4h['timestamps']
            print(f"  ✓ 4h cached: {len(close_4h):,} bars")
        else:
            klines_4h = download_klines(client, sym, '4h', DOWNLOAD_DAYS)
            if len(klines_4h) < MTF_SMA_PERIOD + 1:
                print(f"  ✗ 4h: only {len(klines_4h)} bars, skipping")
                continue
            close_4h = np.array([float(k[4]) for k in klines_4h], dtype=np.float64)
            ts_4h = np.array([int(k[0]) for k in klines_4h], dtype=np.int64)
            np.savez_compressed(cache_4h, close=close_4h, timestamps=ts_4h)
            print(f"  ✓ 4h: {len(close_4h):,} bars")

        data[sym] = {
            '15m': d15,
            '4h_close': close_4h,
            '4h_timestamps': ts_4h,
        }

    return data


# ═══════════════════════════════════════════════════════════════════
# PARITY BACKTEST (mirrors live on_bar + exit exactly)
# ═══════════════════════════════════════════════════════════════════
def run_parity_backtest(data: dict) -> list[dict]:
    """Run bar-by-bar parity backtest over the live period.

    Iterates 15m bars chronologically across all symbols,
    evaluating entry/exit at each bar exactly as on_bar() + exit logic does.

    Returns list of trade dicts.
    """
    btc_data = data.get(BTC_SYMBOL)
    if btc_data is None:
        print("ERROR: No BTC data")
        return []

    btc_15m = btc_data['15m']
    btc_close = btc_15m['close']
    btc_ts = btc_15m['timestamps']

    # Find bar indices for live period
    live_start_ms = int(LIVE_START.timestamp() * 1000)
    live_end_ms = int(LIVE_END.timestamp() * 1000)

    # Pre-compute BTC indicators (full array for warmup)
    btc_sma = _compute_sma_inline(btc_close, BTC_REGIME_PERIOD)

    # Positions and trades
    positions = {}  # symbol -> position dict
    trades = []
    n_bars = len(btc_close)

    # Pre-compute per-symbol indicators
    sym_indicators = {}
    for sym in LIVE_SYMBOLS:
        if sym not in data:
            continue
        d = data[sym]['15m']
        close = d['close']
        high = d['high']
        low = d['low']
        vol = d['volume']
        taker = d['taker_buy']
        ts = d['timestamps']

        n = len(close)
        rsi = _compute_rsi_inline(close, RSI_PERIOD)
        vol_sma = _compute_sma_inline(vol, VOL_PERIOD)
        atr_pct = compute_atr_pct(high, low, close, ATR_PERIOD) if ATR_PERIOD > 0 else np.full(n, np.nan)

        taker_ratio = np.full(n, 0.5)
        for j in range(n):
            if vol[j] > 0:
                taker_ratio[j] = taker[j] / vol[j]

        # 4h data for MTF
        close_4h = data[sym].get('4h_close')
        ts_4h = data[sym].get('4h_timestamps')
        sma_4h = _compute_sma_inline(close_4h, MTF_SMA_PERIOD) if close_4h is not None else None

        sym_indicators[sym] = {
            'close': close, 'high': high, 'low': low,
            'volume': vol, 'taker_buy': taker,
            'timestamps': ts,
            'rsi': rsi, 'vol_sma': vol_sma, 'atr_pct': atr_pct,
            'taker_ratio': taker_ratio,
            'close_4h': close_4h, 'ts_4h': ts_4h, 'sma_4h': sma_4h,
        }

    # ── Bar-by-bar simulation ──
    # We need to align all symbols to the same timestamps as BTC
    start_bar = max(LOOKBACK, BTC_DIRECTIONAL_PERIOD, BTC_REGIME_PERIOD,
                    ASSET_TREND_PERIOD, RSI_PERIOD, VOL_PERIOD)
    if ATR_PERIOD > start_bar:
        start_bar = ATR_PERIOD

    print(f"\n{'='*70}")
    print(f"  Running parity backtest: {n_bars} BTC bars, start_bar={start_bar}")
    print(f"  Live window: {LIVE_START.strftime('%Y-%m-%d %H:%M')} → "
          f"{LIVE_END.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*70}\n")

    for i in range(start_bar, n_bars):
        bar_ts = btc_ts[i]
        in_live_period = live_start_ms <= bar_ts <= live_end_ms

        # ── EXIT: Check all open positions ──
        closed_this_bar = []
        for sym, pos in list(positions.items()):
            ind = sym_indicators.get(sym)
            if ind is None:
                continue

            # Find corresponding bar index for this symbol
            si = _find_bar_idx(ind['timestamps'], bar_ts)
            if si is None or si < 1:
                continue

            pos['bars_held'] += 1
            entry_price = pos['entry_price']
            signal = pos['signal']
            move_size = pos['move_size']
            eff_stop = pos['eff_stop']

            c = ind['close'][si]
            h = ind['high'][si]
            l = ind['low'][si]

            if signal == 1:
                gain = (c - entry_price) / entry_price
                worst_gain = (l - entry_price) / entry_price
                best_gain = (h - entry_price) / entry_price
            else:
                gain = (entry_price - c) / entry_price
                worst_gain = (entry_price - h) / entry_price
                best_gain = (entry_price - l) / entry_price

            if best_gain > pos['best_pnl']:
                pos['best_pnl'] = best_gain

            # SL distance from ATR stop
            sl_dist = eff_stop  # already in absolute fraction
            target = move_size * TARGET_RETRACE

            # Time decay on stop
            time_factor = 1.0 - STOP_DECAY * (pos['bars_held'] / MAX_HOLD)

            exit_reason = None
            exit_pnl = None

            # ── PBT logic ──
            if PARTIAL_EXIT_ENABLED and not pos.get('partial_closed', False):
                # Phase 1: Check TP1 (intra-bar via high/low)
                if best_gain >= target:
                    # Partial close 50% at TP1
                    pos['partial_closed'] = True
                    tp1_pnl = target  # assumed fill at target
                    pos['tp1_pnl'] = tp1_pnl

                    # Now check if SL also hit this bar (SL priority)
                    if worst_gain < -sl_dist * time_factor:
                        # Both hit — assume SL hit first (conservative)
                        # 50% closed at TP1, 50% stopped out
                        exit_reason = 'SL_AFTER_TP1'
                        runner_pnl = -sl_dist * time_factor
                        exit_pnl = PARTIAL_EXIT_FRACTION * tp1_pnl + (1 - PARTIAL_EXIT_FRACTION) * runner_pnl
                    else:
                        # TP1 hit, runner continues
                        pos['highest_since_tp1'] = best_gain
                        pos['trail_floor'] = 0.0  # breakeven floor
                        continue  # runner stays open

                # Phase 1: SL check (before TP1)
                if worst_gain < -sl_dist * time_factor:
                    exit_reason = 'SL'
                    exit_pnl = -sl_dist * time_factor  # stopped out on full position

                # Phase 1: max_hold timeout
                if exit_reason is None and pos['bars_held'] >= MAX_HOLD:
                    exit_reason = 'TIMEOUT'
                    exit_pnl = gain

                # Phase 1: trailing stop (best_pnl > 50% of target and current drops to 30% of best)
                if exit_reason is None and pos['best_pnl'] > target * 0.5 and worst_gain < pos['best_pnl'] * 0.3:
                    exit_reason = 'TRAIL'
                    exit_pnl = pos['best_pnl'] * 0.3  # approximate exit at trail level

            elif PARTIAL_EXIT_ENABLED and pos.get('partial_closed', False):
                # Phase 2: Runner (50% remaining)
                runner_max = int(MAX_HOLD * RUNNER_MAX_HOLD_MULT)

                # Update runner peak
                if best_gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = best_gain

                # Runner SL: breakeven (0%) or trailing
                runner_sl = max(0.0, pos.get('trail_floor', 0.0))
                # Trail: if highest > 2× target, trail at 50% of peak
                peak = pos.get('highest_since_tp1', 0)
                if peak > target * 2:
                    runner_sl = max(runner_sl, peak * 0.5)
                    pos['trail_floor'] = runner_sl

                if worst_gain < runner_sl:
                    exit_reason = 'RUNNER_SL'
                    exit_pnl = pos['tp1_pnl'] * PARTIAL_EXIT_FRACTION + runner_sl * (1 - PARTIAL_EXIT_FRACTION)
                elif pos['bars_held'] >= runner_max:
                    exit_reason = 'RUNNER_TIMEOUT'
                    exit_pnl = pos['tp1_pnl'] * PARTIAL_EXIT_FRACTION + gain * (1 - PARTIAL_EXIT_FRACTION)
                else:
                    continue  # runner stays open

            else:
                # Non-PBT exit (shouldn't happen with current config)
                if worst_gain < -sl_dist * time_factor:
                    exit_reason = 'SL'
                    exit_pnl = -sl_dist * time_factor
                elif best_gain >= target:
                    exit_reason = 'TP'
                    exit_pnl = target
                elif pos['best_pnl'] > target * 0.5 and worst_gain < pos['best_pnl'] * 0.3:
                    exit_reason = 'TRAIL'
                    exit_pnl = pos['best_pnl'] * 0.3
                elif pos['bars_held'] >= MAX_HOLD:
                    exit_reason = 'TIMEOUT'
                    exit_pnl = gain

            if exit_reason is not None:
                # Deduct costs
                cost = 2 * COST_PER_SIDE  # entry + exit
                net_pnl = exit_pnl - cost

                # ── DEBUG: Log exit ──
                bar_dt = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
                print(f"  [{bar_dt.strftime('%m-%d %H:%M')}] EXIT {sym} "
                      f"{'L' if signal==1 else 'S'} {exit_reason} "
                      f"pnl={net_pnl*100:+.2f}% bars={pos['bars_held']}")

                trade = {
                    'symbol': sym,
                    'signal': signal,
                    'entry_bar_ts': pos['entry_bar_ts'],
                    'exit_bar_ts': bar_ts,
                    'entry_price': entry_price,
                    'exit_reason': exit_reason,
                    'gross_pnl_pct': exit_pnl * 100,
                    'net_pnl_pct': net_pnl * 100,
                    'bars_held': pos['bars_held'],
                    'confidence': pos['confidence'],
                    'leverage': pos['leverage'],
                    'move_size_pct': move_size * 100,
                    'eff_stop_pct': sl_dist * 100,
                    'in_live_period': (
                        pos['entry_bar_ts'] >= live_start_ms and
                        pos['entry_bar_ts'] <= live_end_ms
                    ),
                }
                trades.append(trade)
                closed_this_bar.append(sym)

        for sym in closed_this_bar:
            del positions[sym]

        # ── ENTRY: Scan symbols (only during live period) ──
        if not in_live_period:
            continue

        available_slots = MAX_CONCURRENT - len(positions)
        if available_slots <= 0:
            continue

        # Collect candidates with move strength
        candidates = []
        for sym in LIVE_SYMBOLS:
            if sym in positions:
                continue
            ind = sym_indicators.get(sym)
            if ind is None:
                continue

            si = _find_bar_idx(ind['timestamps'], bar_ts)
            if si is None or si < start_bar:
                continue

            close = ind['close']
            high = ind['high']
            low = ind['low']
            vol = ind['volume']
            rsi = ind['rsi']
            vol_sma = ind['vol_sma']
            taker_ratio = ind['taker_ratio']
            atr_pct_arr = ind['atr_pct']

            # ATR adaptive threshold
            if ATR_PERIOD > 0 and not np.isnan(atr_pct_arr[si]):
                thr = ATR_MULTIPLIER * atr_pct_arr[si]
                thr = max(THRESHOLD_FLOOR, min(THRESHOLD_CAP, thr))
            else:
                thr = THRESHOLD

            # 8-bar return
            ret = close[si] / close[si - LOOKBACK] - 1.0
            if abs(ret) <= thr:
                continue

            # BTC regime
            btc_sma_val = btc_sma[i]
            if np.isnan(btc_sma_val):
                continue

            # BTC momentum
            btc_mom = btc_close[i] / btc_close[i - BTC_DIRECTIONAL_PERIOD] - 1.0

            rsi_val = rsi[si]
            vol_sma_val = vol_sma[si]
            vol_ratio = vol[si] / vol_sma_val if (not np.isnan(vol_sma_val) and vol_sma_val > 0) else 0.0
            taker_val = taker_ratio[si]

            # ── Risk-Off ONLY confidence scoring (matches live on_bar) ──
            confidence = 0
            signal = 0

            if ret < -thr and btc_mom < 0:
                # LONG: alt dips with BTC → sympathy → mean revert
                if rsi_val < 35:
                    confidence += 1
                if vol_ratio > 1.5:
                    confidence += 1
                if taker_val < 0.42:
                    confidence += 1
                if btc_mom < -0.01:
                    confidence += 1
                # Climax volume
                if CLIMAX_ENABLED and vol_sma_val > 0 and vol_ratio >= CLIMAX_VOL_MULT:
                    confidence += 1
                if confidence >= MIN_CONFIDENCE:
                    signal = 1

            elif ret > thr and btc_mom > 0:
                # SHORT: alt pumps with BTC bounce → fades back
                if rsi_val > 65:
                    confidence += 1
                if vol_ratio > 1.5:
                    confidence += 1
                if taker_val > 0.58:
                    confidence += 1
                if btc_mom > 0.01:
                    confidence += 1
                # Climax volume
                if CLIMAX_ENABLED and vol_sma_val > 0 and vol_ratio >= CLIMAX_VOL_MULT:
                    confidence += 1
                if confidence >= MIN_CONFIDENCE:
                    signal = -1

            if signal == 0:
                continue

            # ── MTF 4h confirmation ──
            if MTF_ENABLED and ind['sma_4h'] is not None:
                close_4h = ind['close_4h']
                sma_4h = ind['sma_4h']
                ts_4h = ind['ts_4h']
                # Find the 4h bar corresponding to this 15m bar
                j4h = _find_4h_bar_idx(ts_4h, bar_ts)
                if j4h is not None and j4h >= MTF_SMA_PERIOD and not np.isnan(sma_4h[j4h]):
                    trend_up = close_4h[j4h] > sma_4h[j4h]
                    if signal > 0 and not trend_up:
                        continue  # Block long in 4h downtrend
                    if signal < 0 and trend_up:
                        continue  # Block short in 4h uptrend

            # ATR-based stop distance (DISABLED in live — evidence from logs)
            atr_val = atr_pct_arr[si]
            move_size = abs(ret)
            if ATR_STOP_ENABLED and not np.isnan(atr_val):
                sl_dist = ATR_STOP_MULT * atr_val
                sl_dist = max(ATR_STOP_FLOOR, min(ATR_STOP_CAP, sl_dist))
            else:
                # Live used fixed 0.382 × move_size (confirmed from all 25 trades)
                sl_dist = move_size * STOP_MULT

            candidates.append((sym, signal, abs(ret), confidence, si, move_size, sl_dist))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

        # ── DEBUG: Log all candidates and entries ──
        if candidates:
            bar_dt = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
            open_syms = list(positions.keys())
            cand_str = ', '.join(
                f"{c[0]}({'L' if c[1]==1 else 'S'} c={c[3]} mv={c[2]*100:.2f}%)"
                for c in candidates[:8]
            )
            print(f"  [{bar_dt.strftime('%m-%d %H:%M')}] "
                  f"slots={available_slots}/{MAX_CONCURRENT} "
                  f"open={open_syms} "
                  f"cands={len(candidates)}: {cand_str}")

        for sym, signal, abs_move, confidence, si, move_size, sl_dist in candidates[:available_slots]:
            ind = sym_indicators[sym]
            lev = EFF_LEVERAGE  # 4x (confirmed from live logs)

            positions[sym] = {
                'signal': signal,
                'entry_price': ind['close'][si],
                'entry_bar_ts': bar_ts,
                'move_size': move_size,
                'eff_stop': sl_dist,
                'bars_held': 0,
                'best_pnl': 0.0,
                'confidence': confidence,
                'leverage': lev,
                'partial_closed': False,
            }

    # Close any remaining positions at last bar
    for sym, pos in positions.items():
        ind = sym_indicators.get(sym)
        if ind is None:
            continue
        si = len(ind['close']) - 1
        entry_price = pos['entry_price']
        signal = pos['signal']
        c = ind['close'][si]
        if signal == 1:
            gain = (c - entry_price) / entry_price
        else:
            gain = (entry_price - c) / entry_price

        if pos.get('partial_closed', False):
            exit_pnl = pos['tp1_pnl'] * PARTIAL_EXIT_FRACTION + gain * (1 - PARTIAL_EXIT_FRACTION)
        else:
            exit_pnl = gain

        net_pnl = exit_pnl - 2 * COST_PER_SIDE
        trades.append({
            'symbol': sym,
            'signal': signal,
            'entry_bar_ts': pos['entry_bar_ts'],
            'exit_bar_ts': ind['timestamps'][si],
            'entry_price': entry_price,
            'exit_reason': 'STILL_OPEN',
            'gross_pnl_pct': exit_pnl * 100,
            'net_pnl_pct': net_pnl * 100,
            'bars_held': pos['bars_held'],
            'confidence': pos['confidence'],
            'leverage': pos['leverage'],
            'move_size_pct': pos['move_size'] * 100,
            'eff_stop_pct': pos['eff_stop'] * 100,
            'in_live_period': True,
        })

    return trades


def _find_bar_idx(timestamps: np.ndarray, target_ts: int) -> int | None:
    """Find the index of the bar with the closest timestamp <= target_ts."""
    idx = np.searchsorted(timestamps, target_ts, side='right') - 1
    if idx < 0 or idx >= len(timestamps):
        return None
    # Allow up to 15 minutes tolerance (900000 ms)
    if abs(timestamps[idx] - target_ts) > 900_000:
        return None
    return int(idx)


def _find_4h_bar_idx(ts_4h: np.ndarray, target_ts: int) -> int | None:
    """Find the LAST COMPLETED 4h bar index before a 15m timestamp.

    The live system downloads klines including the current developing 4h bar,
    whose close = latest trade price (NOT the final bar close). Using the
    current bar's final close would introduce lookahead bias.

    To match live behavior: we return j4h-1 (last completed bar) when
    target_ts falls WITHIN a 4h bar. When target_ts exactly matches a 4h
    bar open, that bar just started so the last completed bar is j4h-1.
    At the bar boundary (4h close = next 4h open), the completed bar is j4h.
    """
    idx = np.searchsorted(ts_4h, target_ts, side='right') - 1
    if idx < 0 or idx >= len(ts_4h):
        return None

    # Check if target_ts falls within this bar (not yet closed)
    # 4h bar spans 4*60*60*1000 = 14_400_000 ms
    bar_open = ts_4h[idx]
    bar_end = bar_open + 14_400_000  # 4h in ms
    if target_ts < bar_end:
        # Target is within this bar → bar not yet completed → use previous
        return int(idx - 1) if idx > 0 else None
    return int(idx)


# ═══════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════
def print_report(trades: list[dict]):
    """Print comparison report."""
    live_trades = [t for t in trades if t['in_live_period']]
    all_trades = trades

    print(f"\n{'='*70}")
    print(f"  PARITY BACKTEST RESULTS")
    print(f"  Period: {LIVE_START.strftime('%Y-%m-%d %H:%M')} → "
          f"{LIVE_END.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Symbols: {len(LIVE_SYMBOLS)}")
    print(f"  Max concurrent: {MAX_CONCURRENT}")
    print(f"{'='*70}")

    if not live_trades:
        print("\n  ⚠ No trades in live period!")
        return

    n = len(live_trades)
    wins = sum(1 for t in live_trades if t['net_pnl_pct'] > 0)
    wr = wins / n * 100 if n > 0 else 0

    total_pnl = sum(t['net_pnl_pct'] for t in live_trades)
    avg_pnl = total_pnl / n if n > 0 else 0

    # Leveraged PnL (approximate)
    total_lev_pnl = sum(t['net_pnl_pct'] * EFF_LEVERAGE for t in live_trades)

    sl_trades = [t for t in live_trades if 'SL' in (t['exit_reason'] or '')]
    tp_trades = [t for t in live_trades if 'TP' in (t['exit_reason'] or '') or t['exit_reason'] == 'RUNNER_SL']
    timeout_trades = [t for t in live_trades if 'TIMEOUT' in (t['exit_reason'] or '')]

    print(f"\n  ── BACKTEST (expected) ──")
    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  Net PnL:    {total_pnl:+.3f}% (unleveraged)")
    print(f"  Lev PnL:    {total_lev_pnl:+.3f}% (leveraged at {EFF_LEVERAGE}x)")
    print(f"  Avg Trade:  {avg_pnl:+.3f}%")
    print(f"  SL exits:   {len(sl_trades)}")
    print(f"  TP/Runner:  {len(tp_trades)}")
    print(f"  Timeouts:   {len(timeout_trades)}")

    print(f"\n  ── LIVE (actual, from logs) ──")
    print(f"  Trades:     25")
    print(f"  Win Rate:   36.0%")
    print(f"  Net PnL:    +$14.79 (+0.76% on $1,957.76)")
    print(f"  SL exits:   16")
    print(f"  TP exits:   9")

    print(f"\n  ── PER-TRADE DETAIL (backtest) ──")
    print(f"  {'Symbol':<14} {'Dir':>4} {'Conf':>4} {'Move%':>6} {'SL%':>6} "
          f"{'PnL%':>7} {'Exit':<15} {'Bars':>4}")
    print(f"  {'-'*65}")
    for t in sorted(live_trades, key=lambda x: x['entry_bar_ts']):
        side = 'L' if t['signal'] == 1 else 'S'
        entry_dt = datetime.fromtimestamp(t['entry_bar_ts']/1000, tz=timezone.utc)
        print(f"  {t['symbol']:<14} {side:>4} {t['confidence']:>4} "
              f"{t['move_size_pct']:>6.2f} {t['eff_stop_pct']:>6.2f} "
              f"{t['net_pnl_pct']:>+7.3f} {t['exit_reason']:<15} {t['bars_held']:>4}  "
              f"{entry_dt.strftime('%m-%d %H:%M')}")

    # Exit reason breakdown
    reasons = {}
    for t in live_trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0.0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['net_pnl_pct']

    print(f"\n  ── EXIT REASON BREAKDOWN ──")
    for reason, stats in sorted(reasons.items()):
        avg = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {reason:<20} {stats['count']:>3} trades  "
              f"total={stats['pnl']:>+7.3f}%  avg={avg:>+7.3f}%")


def main():
    from binance.client import Client

    settings = get_settings()
    print("🔌 Connecting to Binance...")
    client = Client(settings.binance_api_key, settings.binance_api_secret)

    print(f"\n📥 Downloading {DOWNLOAD_DAYS}-day data for {len(ALL_SYMBOLS)} symbols...")
    data = download_all(client)

    if BTC_SYMBOL not in data:
        print("FATAL: Could not download BTC data")
        return

    available_syms = [s for s in LIVE_SYMBOLS if s in data]
    print(f"\n✓ Data ready: {len(available_syms)}/{len(LIVE_SYMBOLS)} symbols + BTC")

    # Show data range
    btc_ts = data[BTC_SYMBOL]['15m']['timestamps']
    print(f"  BTC 15m range: "
          f"{datetime.fromtimestamp(btc_ts[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} → "
          f"{datetime.fromtimestamp(btc_ts[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

    trades = run_parity_backtest(data)
    print_report(trades)


if __name__ == '__main__':
    main()
