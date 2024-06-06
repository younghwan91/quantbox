import numpy as np
import math
import time
from binance_downloader import BinanceDownloader
from pair_selector import CryptoPairSelector

class CryptoTrader:
    def __init__(self, client, config, enable_plot=0):
        self.client = client
        self.config = config
        self.enable_plot = enable_plot

    def find_cointegrated_pairs(self): 
        print("Finding cointegrated pairs...")
        selector = CryptoPairSelector(self.client, self.config)
        coint_pairs, data = selector.select_trading_pairs()
        best_pair, best_sharpe_ratio, info_pair = selector.select_best_pair(data, coint_pairs, self.enable_plot)
        return best_pair, best_sharpe_ratio, info_pair

    def download_data(self):
        print("Downloading data...")
        downloader = BinanceDownloader(self.config, self.client)
        downloader.download_data()

    def get_symbol_info(self, symbol):
        try:
            info = self.client.futures_exchange_info()
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

    def round_step_size(self, quantity, step_size):
        return math.floor(quantity / step_size) * step_size

    def simulate_order(self, symbol, side, quantity):
        try:
            step_size, tick_size, min_notional, quantity_precision = self.get_symbol_info(symbol)
            rounded_quantity = self.round_step_size(quantity, step_size)
            rounded_quantity = round(rounded_quantity, quantity_precision)
            price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
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

    def place_order(self, symbol, side, rounded_quantity):
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=rounded_quantity,
            )
            return order
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None

    def get_current_positions(self):
        try:
            positions = self.client.futures_account()['positions']
            current_positions = [position for position in positions if float(position['positionAmt']) != 0.0]
            return current_positions
        except Exception as e:
            print(f"An exception occurred while fetching positions: {e}")
            return []

    def close_position(self, symbol, quantity):
        try:
            side = 'SELL' if quantity > 0 else 'BUY'
            quantity = abs(quantity)
            step_size, tick_size, min_notional, quantity_precision = self.get_symbol_info(symbol)
            rounded_quantity = self.round_step_size(quantity, step_size)
            rounded_quantity = round(rounded_quantity, quantity_precision)
            if rounded_quantity == 0.0:
                print(f"Rounded quantity is zero for {symbol}, unable to close position")
                return
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=rounded_quantity,
            )
            print(f"Closed position for {symbol}: {order}")
        except Exception as e:
            print(f"An exception occurred while closing position: {e}")

    def close_all_positions(self):
        current_positions = self.get_current_positions()
        for position in current_positions:
            symbol = position['symbol']
            quantity = float(position['positionAmt'])
            print(f"Closing position for {symbol}, Quantity: {quantity}")
            self.close_position(symbol, quantity)

    def print_futures_balance(self):
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == 'USDT':
                    print("-"*50)
                    print(f"Asset: {balance['asset']}")
                    print(f"Available Balance: {float(balance['availableBalance']):.2f}")
                    print(f"Cross Wallet Balance: {float(balance['crossWalletBalance']):.2f}")
                    print(f"Cross Unrealized PNL: {float(balance['crossUnPnl']):.2f}")
                    print(f"Total Wallet Balance: {float(balance['balance']):.2f}")
                    print("-"*50)
                    break
        except Exception as e:
            print(f"An exception occurred while fetching futures account balance: {e}")

    def get_available_balance(self, asset='USDT'):
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == asset:
                    available_balance = float(balance['availableBalance'])
                    print(f"Available balance for {asset}: {available_balance}")
                    return available_balance
        except Exception as e:
            print(f"An exception occurred while fetching futures account balance: {e}")
            return 0.0

    def run_trading_cycle(self):
        while True:
            self.close_all_positions()
            open_orders = self.client.futures_get_open_orders()
            if open_orders:
                print("There are open orders, exiting...")
                break
            
            capital = self.get_available_balance()
            capital = capital

            self.download_data()
            
            best_pair, best_sharpe_ratio, info_pair = self.find_cointegrated_pairs()
            if best_pair is not None and best_sharpe_ratio > 0.1:
                
                coin1, coin2 = best_pair
                hedge_ratio = info_pair['hedge_ratio']
                intercept = info_pair['intercept']
                trading_charge = self.config['TRADING_CHARGE']
                signal = info_pair['signal'].iloc[-1]
                upper_bound = info_pair['upper_bound']
                lower_bound = info_pair['lower_bound']
                mean_spread = info_pair['mean_spread']
                std_spread = info_pair['std_spread']

                E = capital * (1-2*trading_charge)
                S1 = E / (1 + hedge_ratio)
                S2 = hedge_ratio * S1
                qty1 = S1 / info_pair['price1']
                qty2 = S2 / info_pair['price2']
                coin1_trade = info_pair['trade_coin1']
                coin2_trade = info_pair['trade_coin2']

                print("-"*50)
                print(f"Capital: {capital}, Effective capital: {E}")
                print(f"Hedge ratio: {hedge_ratio}")
                print(f"Trading pair: {coin1} {coin2}")
                print(f"Quantity 1: {qty1:.4f}, Quantity 2: {qty2:.4f}")
                print(f"Trade Coin 1: {coin1_trade}, Trade Coin 2: {coin2_trade}")
                print("-"*50)
                
                rounded_qty1, cost1 = self.simulate_order(coin1, coin1_trade, qty1)
                rounded_qty2, cost2 = self.simulate_order(coin2, coin2_trade, qty2)
                
                if rounded_qty1 and rounded_qty2:
                    print(f"Rounded Quantity for {coin1}: {rounded_qty1}, Cost: {cost1}")
                    print(f"Rounded Quantity for {coin2}: {rounded_qty2}, Cost: {cost2}")
                    print(f"Percentage difference from capital: {(cost1+cost2)/capital*100:.2f}%")
                    order1 = self.place_order(coin1, coin1_trade, rounded_qty1)
                    order2 = self.place_order(coin2, coin2_trade, rounded_qty2)

                    price1 = float(self.client.futures_symbol_ticker(symbol=coin1)['price'])
                    price2 = float(self.client.futures_symbol_ticker(symbol=coin2)['price'])
                    init_spread = np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
                    init_z_spread = (init_spread-mean_spread)/std_spread
                    print(f"Initial Z-spread: {init_z_spread:.2f}")

                    cnt = 0
                    

                    while True:
                        price1 = float(self.client.futures_symbol_ticker(symbol=coin1)['price'])
                        price2 = float(self.client.futures_symbol_ticker(symbol=coin2)['price'])
                        spread_i = np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
                        z_spread_i = (spread_i-mean_spread)/std_spread

                        print(f"\033[A\033[KHolding positions for the {cnt}th time. Z-spread: {z_spread_i:.2f}, Signal: {signal}, Upper bound: {upper_bound:.2f}, Lower bound: {lower_bound:.2f}")
                        
                        if signal == 1 and z_spread_i > 0:
                            self.close_all_positions()
                            break
                        elif signal == 1 and z_spread_i < (init_z_spread-2):
                            self.close_all_positions()
                            break
                        elif signal == -1 and z_spread_i < 0:
                            self.close_all_positions()
                            break
                        elif signal == -1 and z_spread_i > (init_z_spread+2):
                            self.close_all_positions()
                            break
                        else:
                            cnt += 1
                            
                        time.sleep(30)
                else:
                    print("Error placing orders")
                    time.sleep(60)
            else:
                print("No cointegrated pairs found")
                time.sleep(60)