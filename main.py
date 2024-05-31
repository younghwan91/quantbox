from binance_downloader import BinanceDownloader
from pair_search import CryptoPairSelector
from config import CONFIG
from binance.client import Client
import numpy as np
import json
import math
import time
import os

def find_cointegrated_pairs(client, ENABLE_PLOT): 
    print("Finding cointegrated pairs...")
    selector = CryptoPairSelector(client, CONFIG)
    coint_pairs, data = selector.select_trading_pairs()
    best_pair, best_sharpe_ratio, info_pair = selector.select_best_pair(data, coint_pairs, ENABLE_PLOT)

    return best_pair, best_sharpe_ratio, info_pair

def download_data(client):
    print("Downloading data...")
    downloader = BinanceDownloader(CONFIG, client)
    downloader.download_data()

def get_symbol_info(client, symbol):
    try:
        info = client.futures_exchange_info()
        if info:
            for symbol_info in info['symbols']:
                if symbol_info['symbol'] == symbol:
                    filters = symbol_info['filters']
                    step_size = None
                    tick_size = None
                    min_notional = None
                    quantity_precision = symbol_info['quantityPrecision']
                    for f in filters:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            tick_size = float(f['tickSize'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            min_notional = float(f['notional'])
                    return step_size, tick_size, min_notional, quantity_precision
        else:
            print(f"No symbol info found for {symbol}")
            return None, None, None, None
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None, None, None, None


def round_step_size(quantity, step_size):
    return math.floor(quantity / step_size) * step_size

def simulate_order(client, symbol, side, quantity):
    try:
        step_size, tick_size, min_notional, quantity_precision = get_symbol_info(client, symbol)
        rounded_quantity = round_step_size(quantity, step_size)
        rounded_quantity = round(rounded_quantity, quantity_precision)
        
        # Fetch the current price
        price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
        notional = rounded_quantity * price
        print(f"Quantity: {quantity}, Rounded Quantity: {rounded_quantity}")
        print(f"Step size: {step_size}, Tick size: {tick_size}, Min notional: {min_notional}, Quantity precision: {quantity_precision}")
        print(f"Price: {price}, Notional: {notional}")
        if notional < min_notional:
            raise ValueError(f"Order notional {notional} is less than the minimum notional {min_notional}")
        
        return rounded_quantity, notional
    except Exception as e:
        print(f"An exception occurred during simulation: {e}")
        return None, None

def place_order(client, symbol, side, rounded_quantity):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=rounded_quantity,
        )
        return order
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None

def get_current_positions(client):
    try:
        positions = client.futures_account()['positions']
        current_positions = [position for position in positions if float(position['positionAmt']) != 0.0]
        return current_positions
    except Exception as e:
        print(f"An exception occurred while fetching positions: {e}")
        return []


def close_position(client, symbol, quantity):
    try:
        side = 'SELL' if quantity > 0 else 'BUY'
        quantity = abs(quantity)
        step_size, tick_size, min_notional, quantity_precision = get_symbol_info(client, symbol)
        rounded_quantity = round_step_size(quantity, step_size)
        rounded_quantity = round(rounded_quantity, quantity_precision)
        
        if rounded_quantity == 0.0:
            print(f"Rounded quantity is zero for {symbol}, unable to close position")
            return

        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=rounded_quantity,
        )
        print(f"Closed position for {symbol}: {order}")
    except Exception as e:
        print(f"An exception occurred while closing position: {e}")

def close_all_positions(client):
    current_positions = get_current_positions(client)
    for position in current_positions:
        symbol = position['symbol']
        quantity = float(position['positionAmt'])
        print(f"Closing position for {symbol}, Quantity: {quantity}")
        close_position(client, symbol, quantity)

def print_futures_balance(client):
    try:
        balances = client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == 'USDT':
                print(f"Asset: {balance['asset']}")
                print(f"Available Balance: {balance['availableBalance']}")
                print(f"Cross Wallet Balance: {balance['crossWalletBalance']}")
                print(f"Cross Unrealized PNL: {balance['crossUnPnl']}")
                print(f"Total Wallet Balance: {balance['balance']}")
                break
    except Exception as e:
        print(f"An exception occurred while fetching futures account balance: {e}")

def get_available_balance(client, asset='USDT'):
    try:
        balances = client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == asset:
                available_balance = float(balance['availableBalance'])
                print(f"Available balance for {asset}: {available_balance}")
                return available_balance
    except Exception as e:
        print(f"An exception occurred while fetching futures account balance: {e}")
        return 0.0

if __name__ == '__main__':
    ENABLE_DOWNLOAD = 1
    ENABLE_PLOT = 0
    testnet = 0

    with open('binance_id.json', 'r') as file:
        config = json.load(file)
        api_key = config['api_key']
        api_secret = config['api_secret']    

    if testnet:
        api_key = 'YOUR_BINANCE_API_KEY'
        api_secret = 'YOUR_BINANCE_API_SECRET'

    client = Client(api_key= api_key, api_secret= api_secret, testnet=testnet)
    print_futures_balance(client)
    
    try:
        account_info = client.futures_account()
        print("Account info retrieved successfully:")
    except Exception as e:
        print(f"Error with mainnet: {e}")

    while True:
        # Close all current positions
        close_all_positions(client)
        # if there is any open orders, do exit
        open_orders = client.futures_get_open_orders()
        if open_orders:
            print("There are open orders, exiting...")
            break
        
        capital = get_available_balance(client)
        capital = capital * (4/5) # Use 80% of the available balance        

        if ENABLE_DOWNLOAD:
            download_data(client)
        
        best_pair, best_sharpe_ratio, info_pair = find_cointegrated_pairs(client, ENABLE_PLOT)
        print(f"Expected instant return: {info_pair['instant_return']*100}")
        if best_pair is not None and best_sharpe_ratio > 0.1:
            
            coin1, coin2 = best_pair
            hedge_ratio = info_pair['hedge_ratio']
            intercept = info_pair['intercept']
            trading_charge = CONFIG['TRADING_CHARGE']
            signal = info_pair['signal'].iloc[-1]
            upper_bound = info_pair['upper_bound']
            lower_bound = info_pair['lower_bound']
            mean_spread = info_pair['mean_spread']
            std_spread = info_pair['std_spread']

            E = capital * (1-2*trading_charge)
            print(f"Capital: {capital}, Trading charge: {trading_charge}, Effective capital: {E}")
            print(f"Hedge ratio: {hedge_ratio}")
            S1 = E / (1 + hedge_ratio)
            S2 = hedge_ratio * S1
            qty1 = S1 / info_pair['price1']
            qty2 = S2 / info_pair['price2']
            coin1_trade = info_pair['trade_coin1'] # 1 order buy, -1 order sell 
            coin2_trade = info_pair['trade_coin2']
            print(f"Trading pair: {coin1} {coin2}")
            print(qty1, qty2)
            print(coin1_trade, coin2_trade)
            
            # Simulate orders
            rounded_qty1, cost1 = simulate_order(client, coin1, coin1_trade, qty1)
            rounded_qty2, cost2 = simulate_order(client, coin2, coin2_trade, qty2)
            
            if rounded_qty1 and rounded_qty2:
                print(f"Rounded Quantity for {coin1}: {rounded_qty1}, Cost: {cost1}")
                print(f"Rounded Quantity for {coin2}: {rounded_qty2}, Cost: {cost2}")
                print(f"Percentage difference from capital: {(cost1+cost2)/capital*100}")

                # Place orders if both simulations are successful
                order1 = place_order(client, coin1, coin1_trade, rounded_qty1)
                order2 = place_order(client, coin2, coin2_trade, rounded_qty2)

                price1 = float(client.futures_symbol_ticker(symbol=coin1)['price'])
                price2 = float(client.futures_symbol_ticker(symbol=coin2)['price'])
                init_spread = np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
                init_z_spread = (init_spread-mean_spread)/std_spread
                print(f"Initial Z-spread: {init_z_spread}")

                cnt = 0
                while True:
                    # check the prices
                    price1 = float(client.futures_symbol_ticker(symbol=coin1)['price'])
                    price2 = float(client.futures_symbol_ticker(symbol=coin2)['price'])
                    #compute the z-spread
                    spread_i = np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
                    z_spread_i = (spread_i-mean_spread)/std_spread

                    if signal == 1 and z_spread_i > 0:
                        close_all_positions(client)
                        break
                    elif signal == 1 and z_spread_i < (init_z_spread-2):
                        close_all_positions(client)
                        break
                    elif signal == -1 and z_spread_i < 0:
                        close_all_positions(client)
                        break
                    elif signal == -1 and z_spread_i > (init_z_spread+2):
                        close_all_positions(client)
                        break
                    else:
                        cnt += 1
                        print(f"Holding positions... {cnt}, Z-spread: {z_spread_i}, Signal: {signal}, Upper bound: {upper_bound}, Lower bound: {lower_bound}")

                    time.sleep(10)
            else:
                print("Error placing orders")
                time.sleep(60)
        else:
            print("No cointegrated pairs found")
            time.sleep(60)
            