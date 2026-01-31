# main_backtest.py

from binance.client import Client
import datetime as dt
from datetime import timedelta
from binance_downloader import BinanceDownloader
from backtest_preprocessor import BacktestPreprocessor
from pair_trading_model import PairTradingModel  
from backtest_utils import setup_logging, interval_to_minutes
import sys
import json
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

def main(download_data, preprocess_data, days, formation_bars, trading_bars, interval, 
         hour_shift, n_coins, mode):
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Main')

    # --- Calculate windows in hours from bars ---
    interval_in_minutes = interval_to_minutes(interval)
    formation_window = (formation_bars * interval_in_minutes) / 60  # in hours
    trading_window = (trading_bars * interval_in_minutes) / 60      # in hours

    # Configuration Variables
    testnet = False
    total_window_length = formation_window + trading_window
    window_ratio = formation_window / total_window_length

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
    client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)

    # Set start and end time for data download
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - timedelta(hours=days * 24)
    
    # Define configurations for BinanceDownloader
    downloader_config = {
        'SAVING_DIR': 'data',
        'QUOTE_COIN': 'USDT',
        'N_COINS': n_coins,
        'CSV_FILENAME': 'trading_coins_backtest.csv',
        'INTERVAL': interval,
        'START_STR': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'END_STR': end_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Define configurations for BacktestPreprocessor
    preprocessor_config = {
        'SAVING_DIR': 'data',
        'CSV_FILENAME': 'trading_coins_backtest.csv',
        'PAIRS_CSV': 'pairs_for_backtest.csv',
        'PRECOMPUTED_FILE': 'precomputed_values.pkl',
        'WINDOW_RATIO': window_ratio,
        'COINT_THRESHOLD': 0.05,
        'CORR_THRESHOLD': 0.5,
    }

    # Define configurations for PairTradingModel
    model_config = {
        'FORMATION_WINDOW': formation_window,
        'TRADING_WINDOW': trading_window,
        'TOTAL_WINDOW': total_window_length,
        'WINDOW_RATIO': window_ratio,
        'TIMEFRAME': interval,
        'DAYS': days,
        'TRADING_FEE': 0.0005,
        'DATA_DIR': 'data/backtest',
        'HOUR_SHIFT': hour_shift,
        'PRECOMPUTED_FILE': preprocessor_config['PRECOMPUTED_FILE'],
    }

    logger.info(f"Downloading data from {downloader_config['START_STR']} to {downloader_config['END_STR']}")

    if preprocess_data:
        if download_data:
            # Initialize BinanceDownloader
            downloader = BinanceDownloader(downloader_config, client)

            # Step 1: Download data for backtesting
            logger.info("Starting data download for backtesting...")
            downloader.download_data_for_backtest()

        # Step 2: Prepare pairs for backtesting using BacktestPreprocessor
        logger.info("Preparing pairs for backtesting...")
        preprocessor = BacktestPreprocessor(config=preprocessor_config)
        preprocessor.prepare_pairs_for_backtest(
            window_size=str(total_window_length)+'h',
            hour_shift=hour_shift
        )

    # Step 4: Perform action based on selected mode
    if mode == 'eda': # Exploratory Data Analysis (EDA)
        pass

    elif mode == 'optimize':    
        # Step 3: Initialize PairTradingModel
        model = PairTradingModel(model_config)
        logger.info("Loading precomputed values...")
        # Load and split precomputed values
        precomputed_filepath = os.path.join(model_config['DATA_DIR'], model_config['PRECOMPUTED_FILE'])
        model.load_precomputed_values(precomputed_filepath)
        # Optimize trading parameters
        logger.info("Starting optimization of trading parameters...")
        best_params, max_fval, test_fval = model.optimize_parameters()
        logger.info(f'Best parameters: {best_params} with max function value: {max_fval}')
        logger.info(f'Test function value: {test_fval}')
        
    else:
        logger.error(f"Invalid mode selected: {mode}. Choose 'optimize' or 'drl'.")

if __name__ == "__main__":
    # Call main() with hard-coded default values
    main(
        download_data= False,                     # Flag to download data before preprocessing.
        preprocess_data = True,                   # Flag to preprocess data for backtesting.
        days=10,                                # Number of days of data to use.
        hour_shift=24+1,                          # Number of hours to shift the window.
        formation_bars=241,                       # Number of formation bars.
        trading_bars=24,                       # Number of trading bars.
        interval='30m',                          # Data interval (e.g., '5m' for 5 minutes).
        n_coins=50,                             # Number of coins to consider.
        mode='optimize',                        # Mode to run: 'eda', 'optimize', or 'drl'.
    )

