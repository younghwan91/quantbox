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
from pair_trading_model_drl import PairTradingModelDRL


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
    logging.getLogger('stable_baselines3').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)


def main(DOWNLOAD_DATA, PREPROCESS_DATA, DOWNLOAD_AND_PREPROCESS_DATA_ONLY, DAYS, FORMATION_DAYS, TRADING_WINDOW, INTERVAL, 
         HOUR_SHIFT, N_COINS, MODE, SAVE_MODEL, LOAD_MODEL, MODEL_PATH):
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Main')

    # Configuration Variables
    TESTNET = False
    HOURS_IN_DAY = 24
    FORMATION_WINDOW = int(HOURS_IN_DAY * FORMATION_DAYS)
    TOTAL_WINDOW_LENGTH = FORMATION_WINDOW + TRADING_WINDOW
    WINDOW_RATIO = FORMATION_WINDOW / TOTAL_WINDOW_LENGTH

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
        'N_COINS': N_COINS,
        'CSV_FILENAME': 'trading_coins_backtest.csv',
        'INTERVAL': INTERVAL,
        'WINDOW_RATIO': WINDOW_RATIO,
        'COINT_THRESHOLD': 0.05,
        'CORR_THRESHOLD': 0.5,
        'PAIRS_CSV': 'pairs_for_backtest.csv',
        'PRECOMPUTED_FILE': 'precomputed_values.pkl',
        'START_STR': None,
        'END_STR': None
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
        'HOUR_SHIFT': HOUR_SHIFT,
        'WINDOW_SIZE': 128  # For DRL CNN window size
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

    if DOWNLOAD_AND_PREPROCESS_DATA_ONLY:
        logger.info("Data download and preprocessing complete. Exiting...")
        sys.exit(0)

    # Step 4: Perform action based on selected mode
    if MODE == 'optimize':    
        # Step 3: Initialize PairTradingModel
        model = PairTradingModel(model_config)
        logger.info("Loading precomputed values...")
        # Load and split precomputed values
        precomputed_filepath = os.path.join(model_config['DATA_DIR'], 'precomputed_values.pkl')
        model.load_precomputed_values(precomputed_filepath)
        # Optimize trading parameters
        logger.info("Starting optimization of trading parameters...")
        best_params, max_fval, test_fval = model.optimize_parameters()
        logger.info(f'Best parameters: {best_params} with max function value: {max_fval}')
        logger.info(f'Test function value: {test_fval}')
        
    elif MODE == 'drl':
        # Step 3: Initialize PairTradingModelDRL
        model = PairTradingModelDRL(model_config)
        logger.info("Loading precomputed values...")
        # Load and split precomputed values
        precomputed_filepath = os.path.join(model_config['DATA_DIR'], 'precomputed_values.pkl')
        model.load_precomputed_values(precomputed_filepath)
        
        if LOAD_MODEL:
            model.load_model(MODEL_PATH)
        else:
            # Train and test DRL agent
            logger.info("Starting DRL training...")
            model.train_agent(total_timesteps=100000)
        
        logger.info("Testing DRL agent...")
        model.test_agent()
        logger.info("Simulating trading with DRL agent...")
        model.simulate_trading()
        
        if SAVE_MODEL:
            model.save_model(MODEL_PATH)
    else:
        logger.error(f"Invalid mode selected: {MODE}. Choose 'optimize' or 'drl'.")


if __name__ == "__main__":
    # Call main() with hard-coded default values
    main(
        DOWNLOAD_DATA=False,                     # Flag to download data before preprocessing.
        PREPROCESS_DATA=False,                   # Flag to preprocess data for backtesting.
        DOWNLOAD_AND_PREPROCESS_DATA_ONLY=False,  # If True, the script will only download and preprocess data and then exit.
        DAYS=30,                                # Number of days of data to use.
        FORMATION_DAYS=5,                       # Number of formation days.
        TRADING_WINDOW=24,                      # Trading window in hours.
        INTERVAL='30m',                          # Data interval (e.g., '5m' for 5 minutes).
        HOUR_SHIFT=11,                          # Number of hours to shift the window.
        N_COINS=50,                             # Number of coins to consider.
        MODE='optimize',                        # Mode to run: 'optimize' for parameter optimization, 'drl' for deep reinforcement learning.
        SAVE_MODEL=False,                       # Flag to save the trained model.
        LOAD_MODEL=False,                       # Flag to load a pre-trained model.
        MODEL_PATH='pair_trading_model.zip'     # Path to save or load the model.
    )

