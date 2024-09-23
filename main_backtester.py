import json
from binance.client import Client
from pair_selector import CryptoPairSelector
from binance_downloader import BinanceDownloader
import datetime as dt
from datetime import timedelta
import numpy as np
import tqdm

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
HOURS_IN_DAY = 24
TESTNET = 0
PLOT_DATA = 0
DOWMLOAD_DATA = 1

# Load API credentials
with open('binance.json', 'r') as file:
    config = json.load(file)
    api_key = config['api_key']
    api_secret = config['api_secret']

client = Client(api_key=api_key, api_secret=api_secret, testnet=TESTNET)

# Set parameters for training and testing
train_time = int(HOURS_IN_DAY * 12)
test_time = 24
total_time = train_time + test_time
train_ratio = train_time / total_time

config = {
    'SAVING_DIR': 'data',
    'QUOTE_COIN': 'USDT',
    'N_COINS': 100,
    'CSV_FILENAME': 'trading_coins.csv',
    'INTERVAL': '1h',
    'TRAIN_RATIO': train_ratio, 
    'COINT_THRESHOLD': 0.05,
    'TRADING_CHARGE': 0.0005,
    'CORR_THRESHOLD': 0.5,
}

'''
Download data and save trading pairs for training and testing.
'''
days = 30
if DOWMLOAD_DATA:
    train_data, train_pair_data = [], []
    for hour_diff in range(0, HOURS_IN_DAY * days, test_time):

        # Set start and end time for data download
        config['START_STR'] = f'{total_time + hour_diff} hours ago'
        config['END_STR'] = (dt.datetime.now() - timedelta(hours=hour_diff)).strftime("%Y-%m-%d %H:%M:%S")

        # Download data using BinanceDownloader
        downloader = BinanceDownloader(config, client)
        downloader.download_data()

        # Select trading pairs using the refactored CryptoPairSelector
        selector = CryptoPairSelector(client, config)
        coint_pairs, data = selector.select_trading_pairs()
        pair_data = selector.save_pairs_and_data(coint_pairs, data)
        
        train_pair_data.append(pair_data)
        np.save(f'./training_data/pair_data_{train_time}_{test_time}_{config["INTERVAL"]}_{days}.npy', train_pair_data)

'''
Backtest trading pairs and save the data.
'''

train_pair_data = np.load(f'./training_data/pair_data_{train_time}_{test_time}_{config["INTERVAL"]}_{days}.npy', allow_pickle=True)
selector = CryptoPairSelector(client, config)

train_samples = []
for t in tqdm.tqdm(range(len(train_pair_data))):
    pair_data_t = train_pair_data[t]
    for pair_name, pair_price  in pair_data_t.items():
        sample = selector.extract_features(pair_price)
        if sample:
            train_samples.append(sample)

train_samples = np.array(train_samples)
np.save(f'./training_data/train_samples_{train_time}_{test_time}_{config["INTERVAL"]}.npy', train_samples)


exit()
# Observe and collect training data
data_samples = selector.backtest_pairs(data, coint_pairs, ENABLE_PLOT=PLOT_DATA)
train_data.extend(data_samples)

np.save(f'./training_data/train_{train_time}_{test_time}_{config["INTERVAL"]}.npy', train_data)