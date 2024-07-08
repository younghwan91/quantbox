import json
from binance.client import Client
from pair_selector import CryptoPairSelector
from binance_downloader import BinanceDownloader
import datetime as dt
from datetime import timedelta
import numpy as np
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
# Configurations
TESTNET = 0
hour_per_day = 24
generate_training_data = 1
plot_data = 0

# Load API credentials
with open('binance.json', 'r') as file:
    config = json.load(file)
    api_key = config['api_key']
    api_secret = config['api_secret']

client = Client(api_key=api_key, api_secret=api_secret, testnet=TESTNET)

# Set parameters for training and testing
train_unit = hour_per_day * 7
test_unit = 2
total_unit = train_unit + test_unit
train_ratio = train_unit / total_unit

train_data = []
for hour_diff in range(0, hour_per_day * 7, test_unit):
    config = {
        'SAVING_DIR': 'data',
        'QUOTE_COIN': 'USDT',
        'N_COINS': 200,
        'CSV_FILENAME': 'trading_coins.csv',
        'INTERVAL': '15m',
        'START_STR': f'{total_unit + hour_diff} hours ago',
        'END_STR': (dt.datetime.now() - timedelta(hours=hour_diff)).strftime("%Y-%m-%d %H:%M:%S"),
        'TRAIN_RATIO': train_ratio, 
        'COINT_THRESHOLD': 0.05,
        'TRADING_CHARGE': 0.0005,
        'CORR_THRESHOLD': 0.5,
    }

    # Download data using BinanceDownloader
    downloader = BinanceDownloader(config, client)
    downloader.download_data()

    # Select trading pairs using the refactored CryptoPairSelector
    selector = CryptoPairSelector(client, config)
    coint_pairs, data = selector.select_trading_pairs()

    # Observe and collect training data
    data_samples = selector.observe_all_pairs(data, coint_pairs, ENABLE_PLOT=plot_data)
    train_data.extend(data_samples)

    if generate_training_data:
        np.save(f'train_data_{train_unit}_{test_unit}.npy', train_data)
    else:
        break
