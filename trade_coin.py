import os
import pickle
import logging
import numpy as np
from binance.client import Client
from binance.enums import *
import pandas as pd

# Configuration for logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('trade_coin.log')])

# Load API keys from environment variables or configuration file
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

client = Client(API_KEY, API_SECRET)

def load_best_pair_info(filename='info_best_pair.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_current_price(symbol):
    """Get the current price for a symbol from Binance."""
    try:
        avg_price = client.get_avg_price(symbol=symbol)
        return float(avg_price['price'])
    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {e}")
        return None

def place_order(symbol, side, quantity):
    """Place an order on Binance."""
    try:
        order = client.order_market(
            symbol=symbol,
            side=side,
            quantity=quantity
        )
        logging.info(f"Order placed: {order}")
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")

def calculate_quantity(symbol, allocation, price):
    """Calculate the quantity to trade based on allocation and current price."""
    balance = float(client.get_asset_balance(asset=symbol.replace('USDT', ''))['free'])
    return round(allocation / price, 5)  # Adjust precision based on trading pair

def trade_best_pair(info_best_pair, allocation):
    best_pair = info_best_pair['best_pair']
    symbol1 = best_pair[0]
    symbol2 = best_pair[1]

    # Get current prices
    price1 = get_current_price(symbol1)
    price2 = get_current_price(symbol2)

    if not price1 or not price2:
        logging.error("Unable to fetch current prices, aborting trade.")
        return

    # Calculate quantities to trade
    quantity1 = calculate_quantity(symbol1, allocation, price1)
    quantity2 = calculate_quantity(symbol2, allocation, price2)

    # Get the z-score
    current_spread = np.log(price1) - (info_best_pair['hedge_ratio'] * np.log(price2) + info_best_pair['intercept'])
    z_score = (current_spread - info_best_pair['mean_spread']) / info_best_pair['std_spread']

    logging.info(f"Current Z-score: {z_score}")

    # Trading logic
    if z_score > info_best_pair['upper_bound']:
        # Short symbol1 and long symbol2
        logging.info(f"Shorting {symbol1} and longing {symbol2}")
        place_order(symbol1, SIDE_SELL, quantity1)
        place_order(symbol2, SIDE_BUY, quantity2)
    elif z_score < info_best_pair['lower_bound']:
        # Long symbol1 and short symbol2
        logging.info(f"Longing {symbol1} and shorting {symbol2}")
        place_order(symbol1, SIDE_BUY, quantity1)
        place_order(symbol2, SIDE_SELL, quantity2)
    else:
        logging.info("No trading signal.")

if __name__ == "__main__":
    # Load best pair info
    info_best_pair = load_best_pair_info()

    # Example allocation amount in USDT
    allocation = 100  # Adjust this based on your risk management strategy

    trade_best_pair(info_best_pair, allocation)
