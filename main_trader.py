from trader import CryptoTrader
from binance.client import Client
import json
import datetime as dt
import os

ENABLE_PLOT = 0
TESTNET = 0
TRADE = 1

hour_per_day = 24
total_hours = hour_per_day*2

CONFIG = {
    "FOLDER_NAME": 'futures',
    "SAVING_DIR": 'data',
    "CSV_FILENAME": "pairs_list.csv",
    "N_COINS": 100,
    "INTERVAL": '15m',
    "START_STR": f'{total_hours} hour ago',
    "END_STR": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "COINT_THRESHOLD": 0.01,
    "TRAIN_RATIO": 1.0,
    "QUOTE_COIN": 'USDT',
    "TRADING_CHARGE": 0.0005,
    "TRADAING_TAX": 0,
    "MAX_TRADE_PORTION": 0.02,
    "CORR_THRESHOLD": 0.3,
}

saving_dir = os.path.join(CONFIG["SAVING_DIR"], CONFIG["FOLDER_NAME"], CONFIG["INTERVAL"])
CONFIG["SAVING_DIR"] = saving_dir

if TESTNET:
    api_key = 'YOUR_BINANCE_API_KEY'
    api_secret = 'YOUR_BINANCE_API_SECRET'
else:
    with open('binance.json', 'r') as file:
        config = json.load(file)
        api_key = config['api_key']
        api_secret = config['api_secret']

client = Client(api_key=api_key, api_secret=api_secret, testnet=TESTNET)
trader = CryptoTrader(client, CONFIG, ENABLE_PLOT)
trader.print_futures_balance()

if TRADE:
    trader.run_trading_cycle()
 