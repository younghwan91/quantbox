#!/usr/bin/env python3
"""
R24: Illiquid Altcoin Alpha Search
====================================
R22/R23 proved: 15m OHLCV on top-50 liquid coins has NO alpha (best PF=0.905).

Hypothesis: Illiquid markets are less efficient → MR signals may work because:
  - Fewer algo bots → mispricing persists longer
  - Lower liquidity → single large order causes exaggerated moves → mean-revert
  - Small capital ($2,095) is NOT size-constrained

Approach:
  1. Query Binance for ALL USDT-M futures symbols
  2. Rank by 24h quote volume → take bottom 30-80 (excl. top 50)
  3. Download 15m OHLCV (max available history)
  4. Run SAME scanners from R23 on illiquid universe
  5. Compare PF: illiquid vs liquid (R23 baseline)

**ZERO LOOKAHEAD** — all scanners use only past data. MTF uses completed bars only.

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/illiquid_alpha_r24.py
"""

import os
import sys
import time
import json
import numpy as np
import numba
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    compute_atr_pct,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache_illiquid'
INITIAL_CAPITAL = 2095.0
FUNDING_PER_BAR_15m = 0.01 / 100 / 32  # 0.01% per 8h ÷ 32 bars

# Entry parameters (same as R22/R23 baseline)
LOOKBACK = 8
THRESHOLD = 0.032
BTC_MOM_PERIOD = 48
RSI_PERIOD = 14
VOL_PERIOD = 20
ATR_PERIOD = 48
ATR_MULT = 2.0
THR_FLOOR = 0.008
THR_CAP = 0.06

# Existing top-50 liquid symbols (to EXCLUDE from illiquid universe)
LIQUID_50 = {
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT', 'EGLDUSDT',
    'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT',
    'CRVUSDT', 'LDOUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'VETUSDT',
    'EOSUSDT', 'APEUSDT',
    # Additional high-volume symbols that may not be in whitelist
    'WIFUSDT', 'ENAUSDT', 'TAOUSDT', 'WLDUSDT', 'POLUSDT',
    'TIAUSDT', 'JUPUSDT', 'PENGUUSDT', 'TRUMPUSDT', 'PENDLEUSDT',
}

# Slippage model for illiquid coins (conservative, per side)
# Standard liquid: 0.04% (fee + slippage)
# Illiquid: wider spread → 0.06~0.10% per side
COST_PER_SIDE_LIQUID = 0.0004
COST_PER_SIDE_ILLIQUID = 0.0008  # 2x higher cost for illiquid


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Discover & Classify Symbols
# ═══════════════════════════════════════════════════════════════════
def discover_symbols() -> tuple[list[dict], list[dict]]:
    """Query Binance for all USDT-M futures. Return (illiquid, liquid) symbol infos.

    Returns:
        illiquid: list of dicts with 'symbol', 'quoteVolume24h', 'maxLeverage'
        liquid: same format for liquid (top ~50)
    """
    from binance.client import Client

    settings = get_settings()
    client = Client(settings.binance_api_key, settings.binance_api_secret)

    # Get all futures symbols with volume
    print("🔍 Querying Binance USDT-M futures exchange info...")
    exchange_info = client.futures_exchange_info()

    # Filter active USDT-M perpetual contracts only
    active_symbols = []
    for s in exchange_info['symbols']:
        if (s['contractType'] == 'PERPETUAL'
                and s['quoteAsset'] == 'USDT'
                and s['status'] == 'TRADING'):
            active_symbols.append(s['symbol'])

    print(f"  Active USDT-M perpetuals: {len(active_symbols)}")

    # Get 24h tickers for volume data
    print("  Fetching 24h tickers for volume ranking...")
    tickers = client.futures_ticker()
    ticker_map = {t['symbol']: t for t in tickers}

    # Build ranked list
    all_syms = []
    for sym in active_symbols:
        if sym not in ticker_map:
            continue
        t = ticker_map[sym]
        vol_24h = float(t.get('quoteVolume', 0))
        all_syms.append({
            'symbol': sym,
            'quoteVolume24h': vol_24h,
            'lastPrice': float(t.get('lastPrice', 0)),
        })

    # Sort by volume descending
    all_syms.sort(key=lambda x: x['quoteVolume24h'], reverse=True)

    # Classify
    illiquid = []
    liquid = []
    for i, s in enumerate(all_syms):
        sym = s['symbol']
        if sym in LIQUID_50:
            liquid.append(s)
        else:
            # Only include symbols with SOME volume (>$500K/day) to avoid dead coins
            # and exclude very new listings (data < 90 days likely)
            if s['quoteVolume24h'] >= 500_000:
                illiquid.append(s)

    print(f"\n  📊 Classification:")
    print(f"     Liquid (top 50 whitelist): {len(liquid)}")
    print(f"     Illiquid candidates (>$500K vol): {len(illiquid)}")
    if illiquid:
        # Show volume distribution
        vols = [s['quoteVolume24h'] / 1e6 for s in illiquid]
        print(f"     Illiquid vol range: ${min(vols):.1f}M — ${max(vols):.1f}M / 24h")
        print(f"     Illiquid median vol: ${np.median(vols):.1f}M / 24h")

    # Show top 20 illiquid by volume
    print(f"\n  Top 20 illiquid candidates:")
    for i, s in enumerate(illiquid[:20]):
        print(f"    {i+1:2d}. {s['symbol']:<16s}  vol=${s['quoteVolume24h']/1e6:>8.1f}M"
              f"  price=${s['lastPrice']:.4f}")

    # Show bottom 20 illiquid
    if len(illiquid) > 20:
        print(f"\n  Bottom 20 illiquid (thinnest):")
        for i, s in enumerate(illiquid[-20:]):
            print(f"    {len(illiquid)-19+i:2d}. {s['symbol']:<16s}"
                  f"  vol=${s['quoteVolume24h']/1e6:>8.1f}M"
                  f"  price=${s['lastPrice']:.4f}")

    return illiquid, liquid


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Download Data
# ═══════════════════════════════════════════════════════════════════
def download_illiquid_data(symbols: list[dict], max_symbols: int = 80,
                           target_days: int = 730) -> dict:
    """Download 15m klines for illiquid symbols.

    Downloads the maximum available history (up to target_days).
    Saves to data/cache_illiquid/ as npz files.

    Args:
        symbols: list of symbol info dicts from discover_symbols()
        max_symbols: max number of symbols to download
        target_days: target history length in days

    Returns:
        dict[symbol] → {'close', 'high', 'low', 'volume', 'taker_buy', 'n_days'}
    """
    from binance.client import Client

    settings = get_settings()
    client = Client(settings.binance_api_key, settings.binance_api_secret)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime('%Y%m%d')

    syms_to_download = [s['symbol'] for s in symbols[:max_symbols]]
    data = {}
    skipped = 0

    print(f"\n📥 Downloading 15m data for {len(syms_to_download)} illiquid symbols...")
    print(f"   Target: {target_days}d | Cache: {CACHE_DIR}")

    for idx, sym in enumerate(syms_to_download):
        # Check cache first
        cached = _find_cache(sym, target_days)
        if cached is not None:
            d = np.load(cached)
            n_bars = len(d['close'])
            if n_bars >= 96 * 30:  # at least 30 days
                data[sym] = {
                    'close': d['close'].astype(np.float64),
                    'high': d['high'].astype(np.float64),
                    'low': d['low'].astype(np.float64),
                    'volume': d['volume'].astype(np.float64),
                    'taker_buy': d['taker_buy'].astype(np.float64),
                }
                print(f"  [{idx+1:3d}/{len(syms_to_download)}] {sym:<16s} "
                      f"cached {n_bars:>7,d} bars ({n_bars/96:.0f}d)")
                continue
            # Else: too short, re-download

        # Download from Binance
        print(f"  [{idx+1:3d}/{len(syms_to_download)}] {sym:<16s} downloading...", end='', flush=True)
        try:
            klines = _download_klines(client, sym, '15m', target_days)
        except Exception as e:
            print(f" ERROR: {e}")
            skipped += 1
            continue

        if len(klines) < 96 * 30:  # less than 30 days
            print(f" SKIP ({len(klines)} bars < 30d min)")
            skipped += 1
            continue

        c = np.array([float(k[4]) for k in klines], dtype=np.float64)  # close
        h = np.array([float(k[2]) for k in klines], dtype=np.float64)  # high
        lo = np.array([float(k[3]) for k in klines], dtype=np.float64)  # low
        v = np.array([float(k[5]) for k in klines], dtype=np.float64)  # volume
        t = np.array([float(k[9]) for k in klines], dtype=np.float64)  # taker buy base

        # Save
        n_days = len(c) // 96
        out_path = CACHE_DIR / f'{sym}_15m_{target_days}d_{today}.npz'
        np.savez_compressed(out_path, close=c, high=h, low=lo, volume=v, taker_buy=t)

        data[sym] = {'close': c, 'high': h, 'low': lo, 'volume': v, 'taker_buy': t}
        print(f" {len(c):>7,d} bars ({n_days}d)")

        time.sleep(0.3)  # Be nice to API

    print(f"\n  ✅ Downloaded: {len(data)} symbols | Skipped: {skipped}")
    return data


def _find_cache(symbol: str, target_days: int) -> Path | None:
    """Find cached npz file for symbol."""
    pattern = f"{symbol}_15m_{target_days}d_"
    for f in sorted(CACHE_DIR.glob(f"{pattern}*.npz"), reverse=True):
        return f
    return None


def _download_klines(client, symbol: str, interval: str, days: int) -> list:
    """Download klines with pagination, drop developing (last) bar."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines = []
    cur = start_ms

    while cur < end_ms:
        try:
            batch = client.futures_klines(
                symbol=symbol, interval=interval,
                startTime=cur, endTime=end_ms, limit=1500)
            if not batch:
                break
            all_klines.extend(batch)
            cur = batch[-1][0] + 1  # next bar open time
            time.sleep(0.15)
        except Exception as e:
            if 'Invalid symbol' in str(e):
                raise
            time.sleep(3)

    # Drop developing bar (last bar may be incomplete)
    if all_klines:
        all_klines = all_klines[:-1]

    return all_klines


# ═══════════════════════════════════════════════════════════════════
# Load R23's top-50 data for direct comparison
# ═══════════════════════════════════════════════════════════════════
CACHE_EXTENDED = Path(__file__).parent.parent.parent / 'data' / 'cache_extended'

def load_liquid_data() -> dict:
    """Load 15m OHLCV for top-50 liquid symbols (same as R23)."""
    print("📊 Loading liquid (top-50) data for comparison...")
    data = {}
    for sym in sorted(LIQUID_50):
        # Try 2190d first, then 1095d
        f = None
        for days in [2190, 1095, 365]:
            pattern = f"{sym}_15m_{days}d_"
            candidates = sorted(CACHE_EXTENDED.glob(f"{pattern}*.npz"), reverse=True)
            if candidates:
                f = candidates[0]
                break

        if f is None:
            continue
        d = np.load(f)
        if 'close' not in d:
            continue
        data[sym] = {
            'close': d['close'].astype(np.float64),
            'high': d['high'].astype(np.float64),
            'low': d['low'].astype(np.float64),
            'volume': d['volume'].astype(np.float64),
            'taker_buy': d['taker_buy'].astype(np.float64),
        }
    print(f"  Loaded {len(data)} liquid symbols")
    return data


# ═══════════════════════════════════════════════════════════════════
# Scanners (from R23 — exact copy, no modification = no new lookahead risk)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_mr(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period,
    min_confidence, rsi_period, vol_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
    use_btc_filter,
):
    """Mean-reversion scanner. ZERO lookahead.

    All indicators computed from bar[0..i] only.
    BTW: btc_close[i] is BTC's close at same bar i — no cross-TF lookahead.

    Returns:
        dirs, confs, moves, atrs
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

    atr_pct = np.full(n, np.nan)
    if atr_period > 0:
        atr_pct = compute_atr_pct(high[:n], low[:n], close[:n], atr_period)

    start = max(lookback, btc_mom_period, rsi_period, vol_period)
    if atr_period > start:
        start = atr_period

    for i in range(start, n):
        # Return over lookback window: close[i] vs close[i-lookback]
        # Both are known at bar i — no lookahead
        ret = close[i] / close[i - lookback] - 1.0

        # Adaptive threshold from ATR (computed as trailing EMA of past bars)
        if atr_period > 0 and not np.isnan(atr_pct[i]):
            thr = max(threshold_floor, min(atr_multiplier * atr_pct[i], threshold_cap))
            atrs[i] = atr_pct[i]
        else:
            thr = threshold
            atrs[i] = abs(ret)

        if abs(ret) <= thr:
            continue

        # BTC momentum: btc_close[i] vs btc_close[i-period]
        # Same timeframe, both known — no lookahead
        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

        direction = 0
        confidence = 0

        if ret < -thr:
            if use_btc_filter and btc_mom >= 0:
                continue
            direction = 1  # MR: fade down → LONG

            if rsi[i] < 35:
                confidence += 1
            if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                confidence += 1
            if taker_ratio[i] < 0.42:
                confidence += 1
            if btc_mom < -0.01:
                confidence += 1

            if confidence < min_confidence:
                direction = 0

        elif ret > thr:
            if use_btc_filter and btc_mom <= 0:
                continue
            direction = -1  # MR: fade up → SHORT

            if rsi[i] > 65:
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


@numba.jit(nopython=True, cache=True)
def scan_volume_climax(close, high, low, volume, taker_buy,
                       vol_period, vol_mult, atr_period, rsi_period):
    """Volume Climax Reversal (from R23-D1). No lookahead.

    All indicators use data[:i+1]. close[i-1] used as open proxy (known at bar i).
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

    start = max(vol_period, atr_period, rsi_period) + 1

    for i in range(start, n):
        if np.isnan(vol_sma[i]) or np.isnan(atr[i]):
            continue
        if volume[i] < vol_mult * vol_sma[i]:
            continue

        bar_range = high[i] - low[i]
        if bar_range < 1e-12:
            continue

        lower_wick = min(close[i], close[i - 1]) - low[i]
        upper_wick = high[i] - max(close[i], close[i - 1])

        direction = 0
        confidence = 0

        if lower_wick > 0.5 * bar_range and rsi[i] < 45:
            direction = 1
            confidence += 1
            if taker_ratio[i] < 0.45:
                confidence += 1
            if rsi[i] < 30:
                confidence += 1
            if close[i] > close[i - 1]:
                confidence += 1
            if volume[i] > vol_mult * 1.5 * vol_sma[i]:
                confidence += 1

        elif upper_wick > 0.5 * bar_range and rsi[i] > 55:
            direction = -1
            confidence += 1
            if taker_ratio[i] > 0.55:
                confidence += 1
            if rsi[i] > 70:
                confidence += 1
            if close[i] < close[i - 1]:
                confidence += 1
            if volume[i] > vol_mult * 1.5 * vol_sma[i]:
                confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            sizes[i] = atr[i]

    return dirs, confs, sizes


@numba.jit(nopython=True, cache=True)
def compute_efficiency_ratio(close, period):
    """Kaufman ER. Trailing computation — no lookahead."""
    n = len(close)
    er = np.full(n, np.nan)
    for i in range(period, n):
        net = abs(close[i] - close[i - period])
        total = 0.0
        for j in range(i - period + 1, i + 1):
            total += abs(close[j] - close[j - 1])
        er[i] = net / total if total > 0 else 0.0
    return er


# ═══════════════════════════════════════════════════════════════════
# Trade Simulator (from R23 — exact copy)
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
    """Per-trade independent simulation.

    NO lookahead in exit logic:
    - Each bar j > entry_bar: uses high[j], low[j], close[j] which are known at bar j
    - SL checked on worst (intra-bar low for long, high for short)
    - SL priority before TP (conservative)
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
                    best_bar = (high[j] - ep) / ep
                else:
                    gain = (ep - close[j]) / ep
                    worst = (ep - high[j]) / ep
                    best_bar = (ep - low[j]) / ep

                if best_bar > mfe:
                    mfe = best_bar
                if worst < mae:
                    mae = worst
                if best_bar > best_pnl:
                    best_pnl = best_bar

                # SL first (conservative)
                if worst < -stop:
                    exit_bar = j
                    exit_reason = 'stop'
                    exit_gain = -stop
                    break
                elif best_bar >= target:
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
                'sym': sym,
            })

    return trades


def compute_stats(trades: list[dict], cost_per_side: float = 0.0,
                  funding_per_bar: float = FUNDING_PER_BAR_15m) -> dict:
    """PF, WR, EV from trades. cost_per_side applied to both entry and exit."""
    if not trades:
        return {'n': 0, 'wr': 0, 'pf': 0, 'rr': 0, 'ev': 0, 'reasons': {}}

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

    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    return {
        'n': n, 'wr': wr, 'avg_win': avg_win * 100, 'avg_loss': avg_loss * 100,
        'rr': rr, 'pf': pf, 'ev': ev * 100, 'reasons': reasons,
    }


def print_result(label: str, stats: dict):
    """One-line result."""
    n = stats['n']
    if n == 0:
        print(f"  {label:<40s}  N=0 (no signals)")
        return
    exits = stats.get('reasons', {})
    stop_pct = exits.get('stop', 0) / n * 100 if n > 0 else 0
    print(f"  {label:<40s}  N={n:>7,d}  WR={stats['wr']:5.1f}%"
          f"  R:R={stats['rr']:.2f}  PF={stats['pf']:.3f}"
          f"  EV={stats['ev']:+.3f}bp  SL%={stop_pct:.0f}%")


def run_on_universe(label_prefix: str, data: dict, btc_close: np.ndarray,
                    cost_per_side: float, all_results: list) -> list[dict]:
    """Run all scanner variants on a given universe. Return results.

    This is the core comparison function: same scanners, same params, different coins.
    """
    symbols = sorted(data.keys())
    if 'BTCUSDT' not in data:
        # For illiquid set, BTC is external reference
        pass
    n_bars = min(len(data[s]['close']) for s in symbols)
    btc = btc_close[:n_bars]

    sc = {s: data[s]['close'][:n_bars] for s in symbols}
    sh = {s: data[s]['high'][:n_bars] for s in symbols}
    sl_ = {s: data[s]['low'][:n_bars] for s in symbols}

    results = []

    def _run(label, desc, sym_dirs, sym_sizes, **kwargs):
        tgt = kwargs.get('target_mult', 1.5)
        stp = kwargs.get('stop_mult', 0.382)
        mh = kwargs.get('max_hold', 25)
        fpb = kwargs.get('funding_per_bar', FUNDING_PER_BAR_15m)

        trades = simulate_trades(sc, sh, sl_, sym_dirs, sym_sizes,
                                 tgt, stp, mh, funding_per_bar=fpb)

        # Zero cost
        stats_zero = compute_stats(trades, cost_per_side=0.0, funding_per_bar=fpb)
        # Realistic cost
        stats_cost = compute_stats(trades, cost_per_side=cost_per_side, funding_per_bar=fpb)

        print_result(f"{label} [zero-cost]", stats_zero)
        print_result(f"{label} [cost={cost_per_side*100:.2f}%]", stats_cost)

        result = {
            'label': label,
            'desc': desc,
            'n': stats_zero['n'],
            'zero_cost': {k: v for k, v in stats_zero.items() if k != 'reasons'},
            'with_cost': {k: v for k, v in stats_cost.items() if k != 'reasons'},
        }
        results.append(result)
        all_results.append(result)
        return stats_zero

    # ── V1: MR baseline (same as R22/R23 A-Prod) ──
    print(f"\n  --- {label_prefix} V1: MR Baseline (conf≥3, BTC filter) ---")
    sym_dirs, sym_sizes = {}, {}
    for s in symbols:
        d = data[s]
        c = d['close'][:n_bars]
        h = d['high'][:n_bars]
        lo = d['low'][:n_bars]
        v = d['volume'][:n_bars]
        t = d['taker_buy'][:n_bars]

        dirs, _, moves, _ = scan_mr(
            c, h, lo, v, t, btc,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, 3, RSI_PERIOD,
            VOL_PERIOD, ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            True)
        sym_dirs[s] = dirs
        sym_sizes[s] = moves

    _run(f'{label_prefix}-V1-MR-c3-BTC', 'MR conf≥3, BTC filter',
         sym_dirs, sym_sizes)

    # ── V2: MR, no BTC filter, conf≥2 (more signals) ──
    print(f"\n  --- {label_prefix} V2: MR NoBTC (conf≥2) ---")
    sym_dirs2, sym_sizes2 = {}, {}
    for s in symbols:
        d = data[s]
        c = d['close'][:n_bars]
        h = d['high'][:n_bars]
        lo = d['low'][:n_bars]
        v = d['volume'][:n_bars]
        t = d['taker_buy'][:n_bars]

        dirs, _, moves, _ = scan_mr(
            c, h, lo, v, t, btc,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, 2, RSI_PERIOD,
            VOL_PERIOD, ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            False)
        sym_dirs2[s] = dirs
        sym_sizes2[s] = moves

    _run(f'{label_prefix}-V2-MR-c2-noBTC', 'MR conf≥2, no BTC filter',
         sym_dirs2, sym_sizes2)

    # ── V3: MR, wider threshold (THR_FLOOR=0.015) for higher vol ──
    print(f"\n  --- {label_prefix} V3: MR Wide Threshold ---")
    sym_dirs3, sym_sizes3 = {}, {}
    for s in symbols:
        d = data[s]
        c = d['close'][:n_bars]
        h = d['high'][:n_bars]
        lo = d['low'][:n_bars]
        v = d['volume'][:n_bars]
        t = d['taker_buy'][:n_bars]

        dirs, _, moves, _ = scan_mr(
            c, h, lo, v, t, btc,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, 2, RSI_PERIOD,
            VOL_PERIOD, ATR_PERIOD, ATR_MULT, 0.015, 0.08,  # wider threshold range
            True)
        sym_dirs3[s] = dirs
        sym_sizes3[s] = moves

    _run(f'{label_prefix}-V3-MR-wideThr', 'MR conf≥2, wider threshold (1.5%–8%)',
         sym_dirs3, sym_sizes3)

    # ── V4: Volume Climax Reversal ──
    print(f"\n  --- {label_prefix} V4: Volume Climax Reversal ---")
    sym_dirs4, sym_sizes4 = {}, {}
    for s in symbols:
        d = data[s]
        c = d['close'][:n_bars]
        h = d['high'][:n_bars]
        lo = d['low'][:n_bars]
        v = d['volume'][:n_bars]
        t = d['taker_buy'][:n_bars]

        dirs, _, sizes = scan_volume_climax(c, h, lo, v, t,
                                            VOL_PERIOD, 2.5, ATR_PERIOD, RSI_PERIOD)
        sym_dirs4[s] = dirs
        sym_sizes4[s] = sizes

    _run(f'{label_prefix}-V4-VolClimax', 'Volume climax reversal (>2.5x SMA)',
         sym_dirs4, sym_sizes4,
         target_mult=2.0, stop_mult=1.0, max_hold=20)

    # ── V5: Regime-conditional (range only → MR) ──
    print(f"\n  --- {label_prefix} V5: Regime MR (ER<0.2 only) ---")
    ER_PERIOD = 48
    sym_dirs5, sym_sizes5 = {}, {}
    for s in symbols:
        c = data[s]['close'][:n_bars]
        er = compute_efficiency_ratio(c, ER_PERIOD)
        # Start from V2 (more signals, no BTC filter) and filter by regime
        d5 = sym_dirs2[s].copy()
        for i in range(len(d5)):
            if d5[i] == 0.0:
                continue
            if np.isnan(er[i]) or er[i] >= 0.2:
                d5[i] = 0.0
        sym_dirs5[s] = d5
        sym_sizes5[s] = sym_sizes2[s]  # same sizing

    _run(f'{label_prefix}-V5-Regime-MR', 'MR only in range regime (ER<0.2)',
         sym_dirs5, sym_sizes5)

    # ── V6: Wider stop (ATR-based) for volatile illiquid coins ──
    print(f"\n  --- {label_prefix} V6: MR Wide Stop (0.6x move) ---")
    _run(f'{label_prefix}-V6-MR-wideStop', 'MR conf≥3 + wider stop (0.6x move)',
         sym_dirs, sym_sizes,
         stop_mult=0.6, target_mult=1.8, max_hold=30)

    # ── V7: Tight stop, quick take-profit ──
    print(f"\n  --- {label_prefix} V7: MR Tight (0.25 stop, 1.0 tgt) ---")
    _run(f'{label_prefix}-V7-MR-tight', 'MR conf≥3 + tight stop/tgt',
         sym_dirs, sym_sizes,
         stop_mult=0.25, target_mult=1.0, max_hold=15)

    return results


# ═══════════════════════════════════════════════════════════════════
# Per-Symbol Breakdown
# ═══════════════════════════════════════════════════════════════════
def per_symbol_analysis(trades: list[dict], cost_per_side: float,
                        funding_per_bar: float = FUNDING_PER_BAR_15m) -> dict:
    """Aggregate trades by symbol. Return best/worst performers."""
    by_sym = {}
    for t in trades:
        s = t['sym']
        if s not in by_sym:
            by_sym[s] = []
        pnl = t['raw_gain'] - 2 * cost_per_side - funding_per_bar * t['bars_held']
        by_sym[s].append(pnl)

    results = {}
    for s, pnls in by_sym.items():
        arr = np.array(pnls)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        pf = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else 999
        results[s] = {
            'n': len(arr),
            'wr': len(wins) / len(arr) * 100,
            'pf': pf,
            'ev': float(np.mean(arr)) * 100,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    commit = os.popen("git rev-parse --short HEAD").read().strip()
    now = datetime.now(timezone.utc)

    print("=" * 85)
    print(f"  R24: ILLIQUID ALTCOIN ALPHA SEARCH")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Hypothesis: Illiquid markets have less competition → MR alpha may exist")
    print(f"  All variants: ZERO LOOKAHEAD, per-trade independent test")
    print("=" * 85)

    # ═══ Phase 1: Discover Symbols ═══
    print(f"\n{'═' * 85}")
    print(f"  PHASE 1: SYMBOL DISCOVERY")
    print(f"{'═' * 85}")

    illiquid_syms, liquid_syms = discover_symbols()

    # ═══ Phase 2: Download Data ═══
    print(f"\n{'═' * 85}")
    print(f"  PHASE 2: DATA DOWNLOAD")
    print(f"{'═' * 85}")

    illiquid_data = download_illiquid_data(illiquid_syms, max_symbols=80, target_days=730)

    # Also need BTC as reference (guaranteed present in liquid data or download)
    if 'BTCUSDT' in illiquid_data:
        btc_close_illiquid = illiquid_data['BTCUSDT']['close']
        # Remove BTC from illiquid set (it's the most liquid coin)
        del illiquid_data['BTCUSDT']
    else:
        # Download BTC separately if not in illiquid set
        from binance.client import Client
        settings = get_settings()
        client = Client(settings.binance_api_key, settings.binance_api_secret)
        print("  Downloading BTC reference data...")
        klines = _download_klines(client, 'BTCUSDT', '15m', 730)
        btc_close_illiquid = np.array([float(k[4]) for k in klines], dtype=np.float64)

    # Load liquid data for comparison
    liquid_data = load_liquid_data()

    # Determine common bar count for illiquid set
    if illiquid_data:
        illiq_symbols = sorted(illiquid_data.keys())
        illiq_bars = [len(illiquid_data[s]['close']) for s in illiq_symbols]
        min_bars = min(illiq_bars)
        max_bars = max(illiq_bars)
        print(f"\n  📊 Illiquid data summary:")
        print(f"     Symbols: {len(illiq_symbols)}")
        print(f"     Bars range: {min_bars:,} — {max_bars:,} ({min_bars//96}d — {max_bars//96}d)")

        # Compute per-symbol ATR for characterization
        print(f"\n  📊 Volatility characterization (illiquid):")
        illiq_atrs = []
        for s in illiq_symbols[:10]:  # sample
            d = illiquid_data[s]
            n = min(len(d['close']), len(d['high']), len(d['low']))
            atr = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)
            valid = atr[~np.isnan(atr)]
            if len(valid) > 0:
                illiq_atrs.append(float(np.median(valid)))
                print(f"     {s:<16s}  median ATR%={np.median(valid)*100:.2f}%"
                      f"  bars={n:,d} ({n//96}d)")

        if illiq_atrs:
            print(f"     Overall median ATR%: {np.median(illiq_atrs)*100:.2f}%")

    # ═══ Phase 3: Run Alpha Tests ═══
    print(f"\n{'═' * 85}")
    print(f"  PHASE 3: ALPHA TEST — ILLIQUID UNIVERSE")
    print(f"{'═' * 85}")

    all_results = []

    if illiquid_data:
        illiq_results = run_on_universe(
            'ILLIQ', illiquid_data, btc_close_illiquid,
            COST_PER_SIDE_ILLIQUID, all_results)

    # ═══ Phase 4: Comparison — same scanners on liquid universe ═══
    print(f"\n{'═' * 85}")
    print(f"  PHASE 4: COMPARISON — LIQUID UNIVERSE (TOP-50)")
    print(f"{'═' * 85}")

    if liquid_data:
        btc_close_liquid = liquid_data.get('BTCUSDT', {}).get('close', btc_close_illiquid)
        liq_results = run_on_universe(
            'LIQUID', liquid_data, btc_close_liquid,
            COST_PER_SIDE_LIQUID, all_results)

    # ═══ Phase 5: Per-Symbol Breakdown (Illiquid V1 only) ═══
    print(f"\n{'═' * 85}")
    print(f"  PHASE 5: PER-SYMBOL BREAKDOWN (ILLIQUID V1-MR)")
    print(f"{'═' * 85}")

    if illiquid_data:
        illiq_symbols = sorted(illiquid_data.keys())
        n_bars = min(len(illiquid_data[s]['close']) for s in illiq_symbols)
        btc = btc_close_illiquid[:n_bars]
        sc = {s: illiquid_data[s]['close'][:n_bars] for s in illiq_symbols}
        sh = {s: illiquid_data[s]['high'][:n_bars] for s in illiq_symbols}
        sl_ = {s: illiquid_data[s]['low'][:n_bars] for s in illiq_symbols}

        sym_dirs, sym_sizes = {}, {}
        for s in illiq_symbols:
            d = illiquid_data[s]
            c = d['close'][:n_bars]
            h = d['high'][:n_bars]
            lo = d['low'][:n_bars]
            v = d['volume'][:n_bars]
            t = d['taker_buy'][:n_bars]
            dirs, _, moves, _ = scan_mr(
                c, h, lo, v, t, btc,
                LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, 3, RSI_PERIOD,
                VOL_PERIOD, ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP, True)
            sym_dirs[s] = dirs
            sym_sizes[s] = moves

        trades = simulate_trades(sc, sh, sl_, sym_dirs, sym_sizes)
        sym_stats = per_symbol_analysis(trades, COST_PER_SIDE_ILLIQUID)

        # Sort by PF
        ranked = sorted(sym_stats.items(), key=lambda x: x[1]['pf'], reverse=True)

        print(f"\n  {'Symbol':<16s}  {'N':>6s}  {'WR%':>5s}  {'PF':>6s}  {'EV(bp)':>7s}")
        print(f"  {'─' * 50}")

        above_1 = 0
        for sym, st in ranked:
            if st['n'] < 10:
                continue
            marker = " ★" if st['pf'] >= 1.0 else ""
            print(f"  {sym:<16s}  {st['n']:>6d}  {st['wr']:5.1f}  {st['pf']:6.3f}  {st['ev']:+7.2f}{marker}")
            if st['pf'] >= 1.0:
                above_1 += 1

        print(f"\n  Symbols with PF ≥ 1.0 (with cost): {above_1}/{len([s for s, st in ranked if st['n'] >= 10])}")

    # ═══ Summary ═══
    elapsed = time.time() - t0
    print(f"\n{'═' * 85}")
    print(f"  FINAL SUMMARY")
    print(f"  Runtime: {elapsed:.0f}s")
    print(f"{'═' * 85}\n")

    # Side-by-side comparison
    print(f"  {'Variant':<40s}  {'N':>8s}  {'PF(0)':>6s}  {'PF($)':>6s}  {'WR%':>5s}")
    print(f"  {'─' * 75}")

    for r in all_results:
        zc = r.get('zero_cost', {})
        wc = r.get('with_cost', {})
        pf_z = zc.get('pf', 0)
        pf_c = wc.get('pf', 0)
        wr = zc.get('wr', 0)
        n = r.get('n', 0)
        marker = " ★" if pf_z >= 1.0 else ("◆" if pf_c >= 1.0 else "")
        print(f"  {r['label']:<40s}  {n:>8,d}  {pf_z:>6.3f}  {pf_c:>6.3f}  {wr:>5.1f}{marker}")

    # Extract best
    illiq_best_pf = max((r['zero_cost']['pf'] for r in all_results
                         if r['label'].startswith('ILLIQ') and r.get('n', 0) > 0), default=0)
    liq_best_pf = max((r['zero_cost']['pf'] for r in all_results
                        if r['label'].startswith('LIQUID') and r.get('n', 0) > 0), default=0)

    print(f"\n  Best PF (zero cost):")
    print(f"    Illiquid:  {illiq_best_pf:.3f}")
    print(f"    Liquid:    {liq_best_pf:.3f}")
    print(f"    Delta:     {illiq_best_pf - liq_best_pf:+.3f}")

    if illiq_best_pf > liq_best_pf and illiq_best_pf >= 1.0:
        print(f"\n  ★ ILLIQUID EDGE CONFIRMED — PF {illiq_best_pf:.3f} > liquid {liq_best_pf:.3f}")
        print(f"  → Proceed to slippage stress test + small live pilot")
    elif illiq_best_pf > liq_best_pf:
        print(f"\n  ◆ ILLIQUID BETTER BUT STILL < 1.0 — PF {illiq_best_pf:.3f}")
        print(f"  → Illiquidity improves PF but not enough for profitability")
    else:
        print(f"\n  ❌ NO IMPROVEMENT — illiquid PF {illiq_best_pf:.3f} ≤ liquid {liq_best_pf:.3f}")
        print(f"  → Market microstructure alone does not create alpha for these signals")

    # ═══ Save Results ═══
    results_path = (Path(__file__).parent.parent.parent
                    / 'logs' / 'research' / 'illiquid_alpha_r24_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean non-serializable items
    clean_results = []
    for r in all_results:
        cr = {k: v for k, v in r.items()}
        clean_results.append(cr)

    with open(results_path, 'w') as f:
        json.dump({
            'commit': commit,
            'timestamp': now.isoformat(),
            'illiquid_symbols': sorted(illiquid_data.keys()) if illiquid_data else [],
            'n_illiquid': len(illiquid_data) if illiquid_data else 0,
            'n_liquid': len(liquid_data) if liquid_data else 0,
            'cost_illiquid': COST_PER_SIDE_ILLIQUID,
            'cost_liquid': COST_PER_SIDE_LIQUID,
            'variants': clean_results,
            'illiq_best_pf': illiq_best_pf,
            'liq_best_pf': liq_best_pf,
        }, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")


if __name__ == "__main__":
    main()
