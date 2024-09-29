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
PLOT_DATA = 0
DAYS = 90
FORMATION_DAYS = 4
TRADING_WINDOW = 24
HOURS_IN_DAY = 24
FORMATION_WINDOW = int(HOURS_IN_DAY * FORMATION_DAYS)
TOTAL_WINDOW_LENGTH = FORMATION_WINDOW + TRADING_WINDOW
WINDOW_RATIO = FORMATION_WINDOW / TOTAL_WINDOW_LENGTH

# Load API credentials
with open('binance.json', 'r') as file:
    config = json.load(file)
    api_key = config['api_key']
    api_secret = config['api_secret']

client = Client(api_key=api_key, api_secret=api_secret, testnet=TESTNET)

# Set parameters for training and testing

config = {
    'SAVING_DIR': 'data',
    'QUOTE_COIN': 'USDT',
    'N_COINS': 200,
    'CSV_FILENAME': 'trading_coins.csv',
    'INTERVAL': '15m',
    'TRAIN_RATIO': WINDOW_RATIO, 
    'COINT_THRESHOLD': 0.05,
    'TRADING_CHARGE': 0.0005,
    'CORR_THRESHOLD': 0.5,
}

train_data, train_pair_data = [], []
for hour_diff in range(0, HOURS_IN_DAY * DAYS, TRADING_WINDOW):

    # Set start and end time for data download
    config['START_STR'] = f'{TOTAL_WINDOW_LENGTH + hour_diff} hours ago'
    config['END_STR'] = (dt.datetime.now() - timedelta(hours=hour_diff)).strftime("%Y-%m-%d %H:%M:%S")

    # Download data using BinanceDownloader
    downloader = BinanceDownloader(config, client)
    downloader.download_data()

    # Select trading pairs using the refactored CryptoPairSelector
    selector = CryptoPairSelector(config)
    coint_pairs, data = selector.select_trading_pairs()
    pair_data = selector.save_pairs_and_data(coint_pairs, data)
    
    train_pair_data.append(pair_data)
    np.save(f'./training_data/pair_data_{FORMATION_WINDOW}_{TRADING_WINDOW}_{config["INTERVAL"]}_{DAYS}.npy', train_pair_data)