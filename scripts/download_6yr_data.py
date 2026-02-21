#!/usr/bin/env python3
"""
6-year (2190d) data download for crash-era validation.
Covers Luna collapse (May 2022), FTX collapse (Nov 2022),
COVID crash (Mar 2020), and China ban (May 2021).

Downloads 15m OHLCV+taker_buy and 4h close for all 50 symbols.
Coins that didn't exist 6 years ago download from their listing date.
Saves to data/cache_extended/ as {SYMBOL}_{interval}_{DAYS}d_{date}.npz

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/download_6yr_data.py
"""

import sys
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

DAYS = 2190  # 6 years
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'

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


def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    matches = sorted(CACHE_DIR.glob(f'{symbol}_{interval}_{days}d_*.npz'), reverse=True)
    return matches[0] if matches else None


def _cache_path(symbol: str, interval: str, days: int) -> Path:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return CACHE_DIR / f'{symbol}_{interval}_{days}d_{today}.npz'


def download_klines(client, symbol: str, interval: str, days: int) -> list:
    """Download klines from Binance futures. Returns raw kline list."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    all_klines, cur = [], start_ms
    retries = 0
    while cur < end_ms:
        try:
            klines = client.futures_klines(
                symbol=symbol, interval=interval,
                startTime=cur, endTime=end_ms, limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            cur = klines[-1][0] + 1
            retries = 0
            time.sleep(0.25)
        except Exception as e:
            retries += 1
            if retries > 5:
                print(f"    ⚠ Giving up after 5 retries: {e}")
                break
            wait = min(2 ** retries, 30)
            print(f"    ⚠ Retry {retries} in {wait}s: {e}")
            time.sleep(wait)

    # Remove last incomplete bar
    if all_klines:
        all_klines = all_klines[:-1]
    return all_klines


def download_15m(client, symbol: str, days: int) -> dict | None:
    """Download 15m OHLCV + taker_buy. Skip if cached."""
    cache = _find_cache(symbol, '15m', days)
    if cache is not None and cache.exists():
        d = np.load(cache)
        n = len(d['close'])
        print(f"    ✓ 15m cached: {n:,} bars ({n/96:.0f}d)")
        return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}

    klines = download_klines(client, symbol, '15m', days)
    if len(klines) < 200:
        return None

    close = np.array([float(k[4]) for k in klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in klines], dtype=np.float64)
    volume = np.array([float(k[5]) for k in klines], dtype=np.float64)
    taker_buy = np.array([float(k[9]) for k in klines], dtype=np.float64)

    out = _cache_path(symbol, '15m', days)
    np.savez_compressed(out, close=close, high=high, low=low,
                        volume=volume, taker_buy=taker_buy)
    print(f"    ✓ 15m saved: {len(close):,} bars ({len(close)/96:.0f}d)")
    return {'close': close, 'high': high, 'low': low,
            'volume': volume, 'taker_buy': taker_buy}


def download_4h(client, symbol: str, days: int) -> np.ndarray | None:
    """Download 4h close. Skip if cached."""
    cache = _find_cache(symbol, '4h', days)
    if cache is not None and cache.exists():
        d = np.load(cache)
        n = len(d['close'])
        print(f"    ✓ 4h  cached: {n:,} bars ({n/6:.0f}d)")
        return d['close']

    klines = download_klines(client, symbol, '4h', days)
    if len(klines) < 12:
        return None

    close = np.array([float(k[4]) for k in klines], dtype=np.float64)
    out = _cache_path(symbol, '4h', days)
    np.savez_compressed(out, close=close)
    print(f"    ✓ 4h  saved: {len(close):,} bars ({len(close)/6:.0f}d)")
    return close


def main():
    from quantbox.core.settings import get_settings
    from binance.client import Client

    settings = get_settings()
    print("🔌 Connecting to Binance...")
    client = Client(settings.binance_api_key, settings.binance_api_secret)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS)

    print(f"\n{'='*70}")
    print(f"  📊 6-YEAR DATA DOWNLOAD ({DAYS} days)")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  Symbols: {len(SYMBOLS_50)}")
    print(f"  Intervals: 15m (OHLCV+taker) + 4h (close)")
    print(f"  Note: coins listed after {start_date.strftime('%Y-%m')} will have shorter history")
    print(f"{'='*70}\n")

    success_15m = 0
    success_4h = 0
    failed = []
    bar_counts = {}

    for idx, sym in enumerate(SYMBOLS_50):
        print(f"  [{idx+1}/{len(SYMBOLS_50)}] {sym}")

        d15 = download_15m(client, sym, DAYS)
        if d15 is not None:
            success_15m += 1
            bar_counts[sym] = len(d15['close'])
        else:
            failed.append(sym)
            print(f"    ✗ 15m failed")
            continue

        d4h = download_4h(client, sym, DAYS)
        if d4h is not None:
            success_4h += 1
        else:
            print(f"    ✗ 4h failed")

    # Summary
    print(f"\n{'='*70}")
    print(f"  ✅ DOWNLOAD COMPLETE")
    print(f"  15m: {success_15m}/{len(SYMBOLS_50)}")
    print(f"  4h:  {success_4h}/{len(SYMBOLS_50)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # Data coverage summary
    if bar_counts:
        sorted_by_bars = sorted(bar_counts.items(), key=lambda x: x[1])
        shortest = sorted_by_bars[0]
        longest = sorted_by_bars[-1]
        print(f"\n  Data range:")
        print(f"    Longest:  {longest[0]} = {longest[1]:,} bars ({longest[1]/96:.0f} days)")
        print(f"    Shortest: {shortest[0]} = {shortest[1]:,} bars ({shortest[1]/96:.0f} days)")
        print(f"    Coverage categories:")

        ranges = [(0, 365), (365, 730), (730, 1095), (1095, 1460), (1460, 1825), (1825, 2190+1)]
        labels = ['<1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr']
        for (lo, hi), label in zip(ranges, labels):
            count = sum(1 for bars in bar_counts.values() if lo * 96 <= bars < hi * 96)
            if count > 0:
                print(f"      {label}: {count} symbols")

    cache_size = sum(f.stat().st_size for f in CACHE_DIR.glob(f"*_{DAYS}d_*.npz"))
    print(f"\n  Cache size (6yr): {cache_size / 1024 / 1024:.1f} MB")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
