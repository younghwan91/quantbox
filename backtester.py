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

config = {'SAVING_DIR': 'data',
        'QUOTE_COIN': 'USDT',
        'N_COINS': 50,
        'CSV_FILENAME': 'trading_coins.csv',
        'INTERVAL': '5m',
        'START_STR': '2 day ago',
        # 'END_STR': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'END_STR': (dt.datetime.now() - timedelta(hours=0)).strftime("%Y-%m-%d %H:%M:%S"),
        'MAX_N_NPD': None,
        'TRAIN_RATIO': 0.5, 
        'COINT_THRESHOLD': 0.01,
        'TRADING_CHARGE': 0.0005,
          }
downloader = BinanceDownloader(config, client)
downloader.download_data()

selector = CryptoPairSelector(client, config)
coint_pairs, data = selector.select_trading_pairs(split=True)
selector.observe_all_pairs(data, coint_pairs, ENABLE_PLOT=1)