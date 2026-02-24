#!/usr/bin/env python3
"""Download 6yr data for 4 new whitelist symbols."""
import sys
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from quantbox.core.settings import get_settings
from binance.client import Client

settings = get_settings()
client = Client(settings.binance_api_key, settings.binance_api_secret)

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'
DAYS = 2190
NEW_SYMBOLS = ['ENAUSDT', 'TAOUSDT', 'WLDUSDT', 'WIFUSDT']


def download_klines(sym, interval, days):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    all_k, cur = [], start_ms
    while cur < end_ms:
        try:
            kl = client.futures_klines(
                symbol=sym, interval=interval,
                startTime=cur, endTime=end_ms, limit=1500)
            if not kl:
                break
            all_k.extend(kl)
            cur = kl[-1][0] + 1
            time.sleep(0.25)
        except Exception as e:
            print(f'  ERR {e}')
            time.sleep(5)
    if all_k:
        all_k = all_k[:-1]
    return all_k


for sym in NEW_SYMBOLS:
    print(f'[{sym}]')
    # 15m
    kl = download_klines(sym, '15m', DAYS)
    if len(kl) >= 200:
        c = np.array([float(k[4]) for k in kl], dtype=np.float64)
        h = np.array([float(k[2]) for k in kl], dtype=np.float64)
        lo = np.array([float(k[3]) for k in kl], dtype=np.float64)
        v = np.array([float(k[5]) for k in kl], dtype=np.float64)
        t = np.array([float(k[9]) for k in kl], dtype=np.float64)
        today = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = CACHE_DIR / f'{sym}_15m_{DAYS}d_{today}.npz'
        np.savez_compressed(out, close=c, high=h, low=lo, volume=v, taker_buy=t)
        print(f'  15m: {len(c):,} bars ({len(c)/96:.0f}d)')
    else:
        print(f'  15m: SKIP ({len(kl)} bars)')
    # 4h
    kl = download_klines(sym, '4h', DAYS)
    if len(kl) >= 12:
        c = np.array([float(k[4]) for k in kl], dtype=np.float64)
        today = datetime.now(timezone.utc).strftime('%Y%m%d')
        out = CACHE_DIR / f'{sym}_4h_{DAYS}d_{today}.npz'
        np.savez_compressed(out, close=c)
        print(f'  4h:  {len(c):,} bars ({len(c)/6:.0f}d)')
    else:
        print(f'  4h:  SKIP ({len(kl)} bars)')

print('DONE')
