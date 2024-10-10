from trader import CryptoTrader
from binance.client import Client
import json
import datetime as dt
import os
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
            "crypto_trader.log", 
            maxBytes=10 * 1024 * 1024,  # 10 MB max size
            backupCount=3  # Keep up to 3 backup logs
        )
    ]
)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('binance').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

ENABLE_PLOT = 1
TESTNET = 0
TRADE = 1
HOURS_IN_DAY = 24

FORMATION_DAYS = 7
FORMATION_WINDOW = HOURS_IN_DAY*FORMATION_DAYS

CONFIG = {
    "FOLDER_NAME": 'futures',
    "SAVING_DIR": 'data',
    "CSV_FILENAME": "pairs_list.csv",
    "N_COINS": 200,
    "INTERVAL": '15m',
    "FORMATION_TIME_WINDOW": FORMATION_WINDOW,
    "COINT_THRESHOLD": 0.05,
    "TRAIN_RATIO": 1.0,
    "QUOTE_COIN": 'USDT',
    "TRADING_CHARGE": 0.0005,
    "SLIPPAGE": 0.005,
    "MAX_TRADE_PORTION": 0.02,
    "CORR_THRESHOLD": 0.5,
    "NORMALITY_THRESHOLD": 0.05,
    "STOP_LOSS": 1.553,
    "RISK_REWARD_RATIO": 1.763,
    "THRESHOLD": 2.5,
    "FRAC_HOLD": 0.448,
    "LEVERAGE": 1,
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
 