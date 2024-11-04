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
import argparse  # Add argparse import

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
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def main(DOWNLOAD_DATA, PREPROCESS_DATA, DAYS, FORMATION_DAYS, TRADING_WINDOW, INTERVAL, HOUR_SHIFT, N_COINS):
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Main')

    # Configuration Variables
    TESTNET = False  
    HOURS_IN_DAY = 24
    FORMATION_WINDOW = int(HOURS_IN_DAY * FORMATION_DAYS)
    TOTAL_WINDOW_LENGTH = FORMATION_WINDOW + TRADING_WINDOW
    WINDOW_RATIO = FORMATION_WINDOW / TOTAL_WINDOW_LENGTH
    # INTERVAL and HOUR_SHIFT are now passed as arguments

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
        'N_COINS': N_COINS,  # Use parsed n_coins
        'CSV_FILENAME': 'trading_coins_backtest.csv',
        'INTERVAL': INTERVAL,  # Use parsed interval
        'WINDOW_RATIO': WINDOW_RATIO, 
        'COINT_THRESHOLD': 0.05,
        'CORR_THRESHOLD': 0.5,
        'PAIRS_CSV': 'pairs_for_backtest.csv',
        'PRECOMPUTED_FILE': 'precomputed_values.pkl',
        'START_STR': None,  # Will be set below
        'END_STR': None     # Will be set below
    }

    # Define configurations for PairTradingModel
    model_config = {
        'FORMATION_WINDOW': FORMATION_WINDOW,  
        'TRADING_WINDOW': TRADING_WINDOW,      
        'TOTAL_WINDOW': TOTAL_WINDOW_LENGTH,   
        'WINDOW_RATIO': WINDOW_RATIO,          
        'TIMEFRAME': INTERVAL,                 
        'DAYS': DAYS,                           
        'TRADING_FEE': 0.0005,                 
        'DATA_DIR': 'data/backtest',
        'HOUR_SHIFT': HOUR_SHIFT  # Use parsed hour_shift
    }    

    # Set start and end time for data download
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - timedelta(hours=DAYS * HOURS_IN_DAY)
    downloader_config['START_STR'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    downloader_config['END_STR'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Downloading data from {downloader_config['START_STR']} to {downloader_config['END_STR']}")
    
    if PREPROCESS_DATA:
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
            overlap=str(TOTAL_WINDOW_LENGTH-HOUR_SHIFT)+'h',
        )

    model = PairTradingModel(model_config)
    logger.info("Precomputed values found. Loading them...")
    # Load and split precomputed values
    precomputed_filepath = os.path.join(model_config['DATA_DIR'], 'precomputed_values.pkl')
    model.load_precomputed_values(precomputed_filepath)

    # Step 4: Optimize trading parameters
    logger.info("Starting optimization of trading parameters...")
    best_params, max_fval, test_fval = model.optimize_parameters()
    logger.info(f'Best parameters: {best_params} with max function value: {max_fval}')
    logger.info(f'Test function value: {test_fval}')

if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Run backtest with optional data download and preprocessing.")
    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Flag to download data before preprocessing.'
    )
    parser.add_argument(
        '--preprocess-data',
        action='store_true',
        help='Flag to preprocess data for backtesting.'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days of data to use.'
    )
    parser.add_argument(
        '--formation-days',
        type=int,
        default=3,
        help='Number of formation days.'
    )
    parser.add_argument(
        '--trading-window',
        type=int,
        default=24,
        help='Trading window in hours.'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='30m',
        help='Data interval.'
    )
    parser.add_argument(
        '--hour-shift',
        type=int,
        default=4,
        help='Number of hours to shift the window.'
    )
    parser.add_argument(
        '--n-coins',
        type=int,
        default=50,
        help='Number of coins to consider.'
    )
    args = parser.parse_args()
    
    main(
        DOWNLOAD_DATA=args.download_data, 
        PREPROCESS_DATA=args.preprocess_data,
        DAYS=args.days,
        FORMATION_DAYS=args.formation_days,
        TRADING_WINDOW=args.trading_window,
        INTERVAL=args.interval,
        HOUR_SHIFT=args.hour_shift,
        N_COINS=args.n_coins
    )
