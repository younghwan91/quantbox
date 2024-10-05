import json
from binance.client import Client
from pair_selector import CryptoPairSelector
from binance_downloader import BinanceDownloader
import datetime as dt
from datetime import timedelta
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed output
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
        RotatingFileHandler(
            "crypto_downloader.log", 
            maxBytes=10 * 1024 * 1024,  # 10 MB max size
            backupCount=3  # Keep up to 3 backup logs
        )
    ]
)

"""
Fetch and save klines for a given coin and interval.
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""

# Configurations
TESTNET = 0
PLOT_DATA = 0
DAYS = 90
HOURS_IN_DAY = 24
FORMATION_DAYS = 4
TRADING_WINDOW = 24
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
    'THRESHOLD': 2.5,
}

train_data, train_pair_data = [], []
for hour_diff in range(0, HOURS_IN_DAY * DAYS, TRADING_WINDOW): # in hours

    end_time = dt.datetime.now(dt.timezone.utc) - timedelta(hours=hour_diff)
    start_time = end_time - timedelta(hours=TOTAL_WINDOW_LENGTH)
    # Set start and end time for data download
    config['START_STR'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    config['END_STR'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Downloading data from {config['START_STR']} to {config['END_STR']}")

    # Download data using BinanceDownloader
    downloader = BinanceDownloader(config, client)
    downloader.download_data()

    # Select trading pairs using the refactored CryptoPairSelector
    selector = CryptoPairSelector(config)
    coint_pairs, data = selector.select_trading_pairs()
    pair_data = selector.save_pairs_and_data(coint_pairs, data)
    
    train_pair_data.append(pair_data)
    np.save(f'./training_data/pair_data_{FORMATION_WINDOW}_{TRADING_WINDOW}_{config["INTERVAL"]}_{DAYS}.npy', train_pair_data)