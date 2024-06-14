import json
from binance.client import Client
from pair_selector import CryptoPairSelector
from binance_downloader import BinanceDownloader
import datetime as dt
from datetime import timedelta

# Backtester code for testing trading strategies

TESTNET = 0

with open('binance_id.json', 'r') as file:
    config = json.load(file)
    api_key = config['api_key']
    api_secret = config['api_secret']

client = Client(api_key=api_key, api_secret=api_secret, testnet=TESTNET)

# 0) define test configuration
# 1) donwload historical data
# 2) select trading pairs
# 3) run backtest
# 4) evaluate performance and plot results

"""
Fetch and save klines for a given coin and interval.

Interval options:
- '1m': 1 minute
- '3m': 3 minutes
- '5m': 5 minutes
- '15m': 15 minutes
- '30m': 30 minutes
- '1h': 1 hour
- '2h': 2 hours
- '4h': 4 hours
- '6h': 6 hours
- '8h': 8 hours
- '12h': 12 hours
- '1d': 1 day
- '3d': 3 days
- '1w': 1 week
- '1M': 1 month
"""

train_day = 2
test_day = 1
total_day = train_day + test_day
train_ratio = train_day / total_day

config = {'SAVING_DIR': 'data',
        'QUOTE_COIN': 'USDT',
        'N_COINS': 20,
        'CSV_FILENAME': 'trading_coins.csv',
        'INTERVAL': '5m',
        'START_STR': f'{total_day} day ago',
        # 'END_STR': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'END_STR': (dt.datetime.now() - timedelta(hours=0)).strftime("%Y-%m-%d %H:%M:%S"),
        'TRAIN_RATIO': train_ratio, 
        'COINT_THRESHOLD': 0.01,
        'TRADING_CHARGE': 0.0005,
        'CORR_THRESHOLD': 0.5,
          }
downloader = BinanceDownloader(config, client)
downloader.download_data()

selector = CryptoPairSelector(client, config)
coint_pairs, data = selector.select_trading_pairs(split=True)
selector.observe_all_pairs(data, coint_pairs, ENABLE_PLOT=1)