#!/usr/bin/env python3
"""
Multi-timeframe data download — 5m, 1h, 4h for MTF trading research.

Downloads 365 days of data for each timeframe to complement
the existing 15m data in data/cache_extended/.

Usage:
    uv run python scripts/download_mtf_data.py
    uv run python scripts/download_mtf_data.py --interval 5m
    uv run python scripts/download_mtf_data.py --interval 4h
"""

import os
import sys
import gzip
import pickle
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ── Configuration ──
DAYS = 365

BARS_PER_DAY = {
    "5m": 288,
    "15m": 96,
    "1h": 24,
    "4h": 6,
}

TOP_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    "SUIUSDT", "TONUSDT", "XLMUSDT", "DOTUSDT", "HBARUSDT",
    "LTCUSDT", "SHIBUSDT", "BCHUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "PEPEUSDT", "ICPUSDT", "AAVEUSDT", "FETUSDT",
    "FILUSDT", "ARBUSDT", "ATOMUSDT", "OPUSDT", "INJUSDT"
]

OUTPUT_DIR = Path("data/cache_extended")

INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


def download_symbol_chunked(client, symbol: str, interval: str,
                             start: datetime, end: datetime) -> pd.DataFrame:
    """Download klines in 30-day chunks."""
    all_dfs = []
    chunk_start = start
    chunk_size = timedelta(days=30)

    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_size, end)
        start_ms = int(chunk_start.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        klines = []
        current_ms = start_ms

        while current_ms < end_ms:
            try:
                batch = client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=str(current_ms),
                    end_str=str(end_ms),
                    limit=1000
                )
                if not batch:
                    break
                klines.extend(batch)
                current_ms = batch[-1][6] + 1  # Close time + 1ms
                time.sleep(0.1)
            except Exception as e:
                print(f"  ⚠ Error at {datetime.utcfromtimestamp(current_ms/1000)}: {e}")
                time.sleep(2)
                current_ms += INTERVAL_MS[interval] * 1000

        if klines:
            columns = [
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base', 'Taker Buy Quote', 'Ignore'
            ]
            df = pd.DataFrame(klines, columns=columns)
            df = df.astype(float)
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df.set_index('Open Time', inplace=True)
            all_dfs.append(df)

        chunk_start = chunk_end

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs)
    result = result[~result.index.duplicated(keep='first')]
    result.sort_index(inplace=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Download multi-TF data")
    parser.add_argument("--interval", default="all",
                        choices=["5m", "1h", "4h", "all"],
                        help="Timeframe to download (default: all)")
    args = parser.parse_args()

    from quantbox.core.settings import get_settings
    from binance.client import Client

    settings = get_settings()

    print("🔌 Connecting to Binance...")
    client = Client(settings.binance_api_key, settings.binance_api_secret)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS)

    intervals = ["5m", "1h", "4h"] if args.interval == "all" else [args.interval]

    for interval in intervals:
        bpd = BARS_PER_DAY[interval]
        print(f"\n{'='*60}")
        print(f"📊 Downloading {interval} data")
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        print(f"   Expected bars/coin: ~{DAYS * bpd:,}")
        print(f"   Coins: {len(TOP_COINS)}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = []

        for i, symbol in enumerate(TOP_COINS):
            cache_file = OUTPUT_DIR / f"{symbol}_{interval}_{DAYS}d.pkl.gz"

            if cache_file.exists():
                print(f"  [{i+1}/{len(TOP_COINS)}] {symbol} ({interval}): cached ✓")
                success += 1
                continue

            print(f"  [{i+1}/{len(TOP_COINS)}] {symbol} ({interval}): downloading...")
            try:
                df = download_symbol_chunked(client, symbol, interval, start_date, end_date)
                if len(df) > 0:
                    with gzip.open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                    print(f"    ✓ {len(df)} bars")
                    success += 1
                else:
                    failed.append(symbol)
                    print(f"    ✗ No data")
            except Exception as e:
                failed.append(symbol)
                print(f"    ✗ Error: {e}")

        print(f"\n  {interval}: {success}/{len(TOP_COINS)} coins downloaded")
        if failed:
            print(f"  Failed: {', '.join(failed)}")

    cache_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.pkl.gz"))
    print(f"\n📦 Total cache size: {cache_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
