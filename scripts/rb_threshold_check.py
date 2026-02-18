#!/usr/bin/env python3
"""Check how often the rubber band threshold is breached on recent 15m data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from binance.client import Client
from quantbox.core.settings import get_settings

s = get_settings()
client = Client(s.binance_api_key, s.binance_api_secret)

symbols = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT', 'ICPUSDT',
    '1000SHIBUSDT', 'FILUSDT',
]

lookback = s.rb_lookback
threshold = s.rb_threshold

print(f'Scanning last 7 days of 15m data, lookback={lookback}, threshold={threshold*100:.1f}%')
print(f'A rubber band signal requires |8-bar return| > {threshold*100:.1f}%')
print()

total_breaches = 0
for sym in symbols:
    klines = client.futures_klines(symbol=sym, interval='15m', limit=672)
    close = np.array([float(k[4]) for k in klines])
    breaches = 0
    max_move = 0.0
    all_moves = []
    for i in range(lookback, len(close)):
        ret = close[i] / close[i - lookback] - 1.0
        all_moves.append(abs(ret))
        if abs(ret) > threshold:
            breaches += 1
        max_move = max(max_move, abs(ret))
    pct_50 = np.percentile(all_moves, 50) * 100
    pct_95 = np.percentile(all_moves, 95) * 100
    pct_99 = np.percentile(all_moves, 99) * 100
    total_breaches += breaches
    print(
        f'{sym:>15s}: breach={breaches:3d}  '
        f'max={max_move*100:.2f}%  p50={pct_50:.2f}%  p95={pct_95:.2f}%  p99={pct_99:.2f}%'
    )

print(f'\nTotal threshold breaches (7d): {total_breaches}')
print(f'≈ {total_breaches/7:.0f}/day across {len(symbols)} symbols')
print(f'≈ {total_breaches/7/len(symbols):.1f}/symbol/day')
print()

# Compare with backtest period volatility
print('--- Now checking 365-day historical volatility for comparison ---')
print()
total_hist = 0
for sym in symbols[:5]:  # Just top 5 to save time
    klines = client.futures_klines(symbol=sym, interval='15m', limit=1000)
    close = np.array([float(k[4]) for k in klines])
    moves = [abs(close[i] / close[i - lookback] - 1.0) for i in range(lookback, len(close))]
    breaches = sum(1 for m in moves if m > threshold)
    pct_95 = np.percentile(moves, 95) * 100
    pct_99 = np.percentile(moves, 99) * 100
    total_hist += breaches
    period_days = len(close) / 96  # 96 bars per day at 15m
    print(
        f'{sym:>15s}: breach={breaches:3d} in {period_days:.0f}d  '
        f'p95={pct_95:.2f}%  p99={pct_99:.2f}%  '
        f'≈{breaches/max(period_days,1):.1f}/day'
    )
