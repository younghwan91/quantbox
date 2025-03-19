# main_backtest.py

from binance.client import Client
import datetime as dt
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from binance_downloader import BinanceDownloader
from backtest_preprocessor import BacktestPreprocessor
from pair_trading_model import PairTradingModel  
import sys
import json
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

def setup_logging():

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Stream Handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating File Handler for log files
    file_handler = RotatingFileHandler(
        "crypto_backtest.log",
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


def main(download_data, preprocess_data, download_and_preprocess_data_only, days, formation_days, trading_window, interval, 
         hour_shift, n_coins, mode, save_model, load_model, model_path):
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Main')

    # Configuration Variables
    testnet = False
    hour_per_day = 24
    formation_window = int(hour_per_day * formation_days)
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

    # Define configurations for BinanceDownloader
    downloader_config = {
        'SAVING_DIR': 'data',
        'QUOTE_COIN': 'USDT',
        'N_COINS': n_coins,
        'CSV_FILENAME': 'trading_coins_backtest.csv',
        'INTERVAL': interval,
        'WINDOW_RATIO': window_ratio,
        'COINT_THRESHOLD': 0.05,
        'CORR_THRESHOLD': 0.5,
        'PAIRS_CSV': 'pairs_for_backtest.csv',
        'PRECOMPUTED_FILE': 'precomputed_values.pkl',
        'START_STR': None,
        'END_STR': None
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
        'WINDOW_SIZE': 128  # For DRL CNN window size
    }

    # Set start and end time for data download
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - timedelta(hours=days * hour_per_day)
    downloader_config['START_STR'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    downloader_config['END_STR'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
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
        preprocessor = BacktestPreprocessor(config=downloader_config)
        preprocessor.prepare_pairs_for_backtest(
            window_size=str(total_window_length)+'h',
            overlap=str(total_window_length-hour_shift)+'h',
        )

    if download_and_preprocess_data_only:
        logger.info("Data download and preprocessing complete. Exiting...")
        sys.exit(0)

    # Step 4: Perform action based on selected mode
    if mode == 'eda' # Exploratory Data Analysis
        pass
    elif mode == 'optimize':    
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
        
    elif mode == 'drl':
        pass
    else:
        logger.error(f"Invalid mode selected: {mode}. Choose 'optimize' or 'drl'.")


if __name__ == "__main__":
    # Call main() with hard-coded default values
    main(
        download_data= False,                     # Flag to download data before preprocessing.
        preprocess_data = True,                   # Flag to preprocess data for backtesting.
        download_and_preprocess_data_only=False,  # If True, the script will only download and preprocess data and then exit.
        days=10,                                # Number of days of data to use.
        formation_days=2,                       # Number of formation days.
        trading_window=24,                      # Trading window in hours.
        interval='5m',                          # Data interval (e.g., '5m' for 5 minutes).
        hour_shift=11,                          # Number of hours to shift the window.
        n_coins=50,                             # Number of coins to consider.
        mode='eda',                        # Mode to run: 'eda', 'optimize', or 'drl'.
        save_model=False,                       # Flag to save the trained model.
        load_model=False,                       # Flag to load a pre-trained model.
        model_path='pair_trading_model.zip'     # Path to save or load the model.
    )

