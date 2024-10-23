# main_backtest.py

import json
from binance.client import Client
import datetime as dt
from datetime import timedelta
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import sys
import os

from binance_downloader import BinanceDownloader
from backtest_preprocessor import BacktestPreprocessor
from pair_trading_model import PairTradingModel  # Ensure this is correctly refactored if needed

def setup_logging():
    """
    Configures logging to output to both stdout and a rotating log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Stream Handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating File Handler for log files
    file_handler = RotatingFileHandler(
        "crypto_downloader.log", 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Suppress less important logs from specific libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('binance').setLevel(logging.WARNING)
    logging.getLogger('statsmodels').setLevel(logging.WARNING)

def main(DOWNLOAD_DATA):
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Main')

    # Configuration Variables
    TESTNET = False  
    DAYS = 30
    HOURS_IN_DAY = 24
    FORMATION_DAYS = 7
    TRADING_WINDOW = 24
    FORMATION_WINDOW = int(HOURS_IN_DAY * FORMATION_DAYS)
    TOTAL_WINDOW_LENGTH = FORMATION_WINDOW + TRADING_WINDOW
    WINDOW_RATIO = FORMATION_WINDOW / TOTAL_WINDOW_LENGTH
    INTERVAL = '30m'

    # Load API credentials
    try:
        with open('binance.json', 'r') as file:
            config_data = json.load(file)
            api_key = config_data['api_key']
            api_secret = config_data['api_secret']
    except Exception as e:
        logger.error(f"Error loading Binance API credentials: {e}")
        sys.exit(1)

    # Initialize Binance Client
    client = Client(api_key=api_key, api_secret=api_secret, testnet=TESTNET)

    # Define configurations for BinanceDownloader
    downloader_config = {
        'SAVING_DIR': 'data',
        'QUOTE_COIN': 'USDT',
        'N_COINS': 20,
        'CSV_FILENAME': 'trading_coins.csv',
        'INTERVAL': INTERVAL,
        'WINDOW_RATIO': WINDOW_RATIO, 
        'COINT_THRESHOLD': 0.05,
        'CORR_THRESHOLD': 0.5,
    }

    # Set start and end time for data download
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - timedelta(hours=DAYS * HOURS_IN_DAY)
    downloader_config['START_STR'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    downloader_config['END_STR'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Downloading data from {downloader_config['START_STR']} to {downloader_config['END_STR']}")
    
    if DOWNLOAD_DATA:
        # Initialize BinanceDownloader
        downloader = BinanceDownloader(downloader_config, client)

        # Step 1: Download data for backtesting
        logger.info("Starting data download for backtesting...")
        downloader.download_data_for_backtest()

    # Step 2: Prepare pairs for backtesting using BacktestPreprocessor
    logger.info("Preparing pairs for backtesting...")
    preprocessor = BacktestPreprocessor(config=downloader_config)
    preprocessor.prepare_pairs_for_backtest(
        window_size=str(TOTAL_WINDOW_LENGTH)+'h', 
        overlap=str(FORMATION_WINDOW)+'h',
        output_filename="pairs_for_backtest.csv"  # Optional: specify if different
    )

    # Define configurations for PairTradingModel
    model_config = {
        'FORMATION_WINDOW': FORMATION_WINDOW,  # e.g., 7 days * 24 hours = 168
        'TRADING_WINDOW': TRADING_WINDOW,      # e.g., 24 hours
        'TOTAL_WINDOW': TOTAL_WINDOW_LENGTH,   # FORMATION_WINDOW + TRADING_WINDOW
        'WINDOW_RATIO': WINDOW_RATIO,          # e.g., 168 / 192 = 0.875
        'TIMEFRAME': INTERVAL,                 # e.g., '30m'
        'DAYS': DAYS,                           # e.g., 30
        'TRADING_FEE': 0.0005,                 # Example trading fee
        'DATA_DIR': 'data/backtest',
        'PAIRS_CSV': 'pairs_for_backtest.csv',
        'PRECOMPUTED_FILE': 'precomputed_values.pkl'
    }

    # Initialize PairTradingModel
    model = PairTradingModel(model_config)

    # Paths
    pairs_csv_path = os.path.join(model_config['DATA_DIR'], model_config['PAIRS_CSV'])
    precomputed_filepath = os.path.join(model_config['DATA_DIR'], model_config['PRECOMPUTED_FILE'])

    # Step 3: Precompute hedge ratios and z-scores or load them if already computed
    if not os.path.exists(precomputed_filepath):
        logger.info("Precomputed values not found. Starting precomputation...")
        # Load pairs information
        pairs_info = model.load_pairs_info(pairs_csv_path)

        # Precompute hedge ratios and z-scores
        model.precompute_hedge_ratios(pairs_info)

        # Save precomputed values
        model.save_precomputed_values(precomputed_filepath)
    else:
        logger.info("Precomputed values found. Loading them...")
        # Load precomputed values
        model.load_precomputed_values(precomputed_filepath)

    # Step 4: Optimize trading parameters
    logger.info("Starting optimization of trading parameters...")
    best_params, max_fval = model.optimize_parameters()
    logger.info(f'Best parameters: {best_params} with max function value: {max_fval}')

if __name__ == "__main__":
    DOWNLOAD_DATA = False
    main(DOWNLOAD_DATA)
