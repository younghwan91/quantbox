#!/usr/bin/env python3
"""
Extended 15m data download — 365 days (1 year) for long-term strategy validation.

Downloads in monthly chunks to avoid memory issues and provides progress tracking.
Saves directly to data/cache_extended/ to avoid overwriting existing 90-day cache.

Usage:
    uv run python scripts/download_extended_data.py
"""

import os
import sys
import json
import gzip
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from tqdm import tqdm

# ── Configuration ──
INTERVAL = "15m"
DAYS = 365  # 1 year
BARS_PER_DAY = 96  # 15m = 96 bars/day

# Same 30 coins as production
TOP_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    "SUIUSDT", "TONUSDT", "XLMUSDT", "DOTUSDT", "HBARUSDT",
    "LTCUSDT", "SHIBUSDT", "BCHUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "PEPEUSDT", "ICPUSDT", "AAVEUSDT", "FETUSDT",
    "FILUSDT", "ARBUSDT", "ATOMUSDT", "OPUSDT", "INJUSDT"
]

OUTPUT_DIR = Path("data/cache_extended")


def download_symbol_chunked(client, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download klines in 30-day chunks to handle rate limits and memory."""
    from binance.client import Client
    
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
                    interval=INTERVAL,
                    start_str=str(current_ms),
                    end_str=str(end_ms),
                    limit=1000
                )
                if not batch:
                    break
                klines.extend(batch)
                current_ms = batch[-1][6] + 1  # Close time + 1ms
                time.sleep(0.1)  # Rate limit
            except Exception as e:
                print(f"  ⚠ Error at {datetime.utcfromtimestamp(current_ms/1000)}: {e}")
                time.sleep(2)
                current_ms += 15 * 60 * 1000 * 1000  # Skip ahead
        
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
    from quantbox.core.settings import get_settings
    from binance.client import Client
    
    settings = get_settings()
    
    print("🔌 Connecting to Binance...")
    client = Client(settings.binance_api_key, settings.binance_api_secret)
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS)
    
    print(f"\n📊 Extended Data Download")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"   Days: {DAYS}")
    print(f"   Expected bars/coin: ~{DAYS * BARS_PER_DAY:,}")
    print(f"   Coins: {len(TOP_COINS)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    failed = []
    
    for i, symbol in enumerate(TOP_COINS):
        cache_file = OUTPUT_DIR / f"{symbol}_15m_{DAYS}d.pkl.gz"
        
        # Skip if already downloaded
        if cache_file.exists():
            print(f"  [{i+1}/{len(TOP_COINS)}] {symbol}: cached, loading...")
            with gzip.open(cache_file, 'rb') as f:
                df = pickle.load(f)
            all_data[symbol] = df
            print(f"    ✓ {len(df)} bars, {df.index.min()} → {df.index.max()}")
            continue
        
        print(f"  [{i+1}/{len(TOP_COINS)}] {symbol}: downloading...")
        try:
            df = download_symbol_chunked(client, symbol, start_date, end_date)
            if len(df) > 0:
                all_data[symbol] = df
                # Save to disk
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                print(f"    ✓ {len(df)} bars, {df.index.min()} → {df.index.max()}")
            else:
                failed.append(symbol)
                print(f"    ✗ No data")
        except Exception as e:
            failed.append(symbol)
            print(f"    ✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Download Complete!")
    print(f"   Success: {len(all_data)}/{len(TOP_COINS)} coins")
    if failed:
        print(f"   Failed: {', '.join(failed)}")
    
    total_bars = sum(len(df) for df in all_data.values())
    print(f"   Total bars: {total_bars:,}")
    
    # Date range alignment check
    if all_data:
        starts = [df.index.min() for df in all_data.values()]
        ends = [df.index.max() for df in all_data.values()]
        print(f"   Common range: {max(starts)} → {min(ends)}")
        common_bars = len(list(all_data.values())[0].loc[max(starts):min(ends)])
        print(f"   Common bars: ~{common_bars:,}")
    
    cache_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.pkl.gz"))
    print(f"   Cache size: {cache_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
