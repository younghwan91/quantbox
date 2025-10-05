import logging
import sys
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import statsmodels.api as sm

def setup_logging(log_file="crypto_backtest.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Stream Handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating File Handler for log files
    file_handler = RotatingFileHandler(
        log_file,
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

def convert_to_timestamp(date_str):
    """
    Convert a date string to a Unix timestamp in milliseconds.
    """
    return int(pd.to_datetime(date_str).timestamp() * 1000)

def convert_to_str(timestamp):
    """
    Convert a Unix timestamp in milliseconds to a date string.
    """
    return pd.to_datetime(timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')

def interval_to_milliseconds(interval):
    """
    Convert Binance interval string to milliseconds.
    """
    ms = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
    }
    return ms.get(interval, None)

def calculate_half_life(spread):
    """
    Calculates the half-life of the spread.
    """
    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    spread_lag = spread_lag.iloc[1:]
    spread_ret = spread_ret.iloc[1:]
    
    model = sm.OLS(spread_ret, sm.add_constant(spread_lag))
    result = model.fit()
    lambda_param = result.params.iloc[1]
    if lambda_param >= 0:
        return 100
    half_life = -np.log(2) / lambda_param
    return half_life

def interval_to_minutes(interval_str):
    unit = interval_str[-1]
    value = int(interval_str[:-1])
    if unit == 'm':
        return value
    elif unit == 'h':
        return value * 60
    elif unit == 'd':
        return value * 60 * 24
    else:
        raise ValueError(f"Unsupported interval format: {interval_str}")