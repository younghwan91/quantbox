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

    def validate_order(self, symbol, side, quantity):
        try:
            step_size, tick_size, min_notional, quantity_precision = self.get_symbol_info(symbol)
            rounded_quantity = self.round_step_size(quantity, step_size)
            rounded_quantity = round(rounded_quantity, quantity_precision)
            price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
            notional = rounded_quantity * price
            
            print("\n" + "="*60)
            print(f"{'Validating Order':<30} | {symbol} | {side}")
            print("="*60)
            print(f"{'Initial Quantity':<30}: {quantity:.4f}")
            print(f"{'Rounded Quantity':<30}: {rounded_quantity:.4f}")
            print(f"{'Step Size':<30}: {step_size}")
            print(f"{'Tick Size':<30}: {tick_size}")
            print(f"{'Min Notional':<30}: {min_notional}")
            print(f"{'Quantity Precision':<30}: {quantity_precision}")
            print(f"{'Price':<30}: {price:.4f}")
            print(f"{'Order Notional':<30}: {notional:.4f}")
            print("="*60 + "\n")
            
            if notional < min_notional:
                raise ValueError(f"Order notional {notional:.4f} is less than the minimum notional {min_notional:.4f}")
            
            return rounded_quantity, notional
        except Exception as e:
            print(f"An exception occurred during validation: {e}")
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
                    print("="*50)
                    print(f"{'Futures Account Balance':^50}")
                    print("="*50)
                    print(f"{'Asset':<25}: {balance['asset']}")
                    print(f"{'Available Balance':<25}: {float(balance['availableBalance']):.2f} USDT")
                    print(f"{'Cross Wallet Balance':<25}: {float(balance['crossWalletBalance']):.2f} USDT")
                    print(f"{'Cross Unrealized PNL':<25}: {float(balance['crossUnPnl']):.2f} USDT")
                    print(f"{'Total Wallet Balance':<25}: {float(balance['balance']):.2f} USDT")
                    print("="*50)
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

            self.download_data()
            
            best_pair, best_sharpe_ratio, info_pair = self.find_cointegrated_pairs()
            if best_pair is not None:
                
                coin1, coin2 = best_pair
                hedge_ratio = info_pair['hedge_ratio']
                intercept = info_pair['intercept']
                trading_charge = self.config['TRADING_CHARGE']
                signal = info_pair['signal'].iloc[-1]
                upper_bound = info_pair['upper_bound']
                lower_bound = info_pair['lower_bound']
                mean_spread = info_pair['mean_spread']
                std_spread = info_pair['std_spread']
                kalman_model = info_pair['kalman_model']

                E = capital * (1 - 2 * trading_charge)
                S1 = E / (1 + hedge_ratio)
                S2 = hedge_ratio * S1
                qty1 = S1 / info_pair['price1']
                qty2 = S2 / info_pair['price2']
                coin1_trade = info_pair['trade_coin1']
                coin2_trade = info_pair['trade_coin2']

                print("\n" + "="*60)
                print(f"{'Capital':<20}: {capital}")
                print(f"{'Effective Capital':<20}: {E:.2f}")
                print(f"{'Hedge Ratio':<20}: {hedge_ratio:.4f}")
                print(f"{'Trading Pair':<20}: {coin1} / {coin2}")
                print(f"{'Quantity 1':<20}: {qty1:.4f}")
                print(f"{'Quantity 2':<20}: {qty2:.4f}")
                print(f"{'Trade Coin 1':<20}: {coin1_trade}")
                print(f"{'Trade Coin 2':<20}: {coin2_trade}")
                print("="*60 + "\n")
                
                rounded_qty1, cost1 = self.validate_order(coin1, coin1_trade, qty1)
                rounded_qty2, cost2 = self.validate_order(coin2, coin2_trade, qty2)
                spread_0, z_spread_0 = self.compute_spread(coin1, coin2, hedge_ratio, intercept, mean_spread, std_spread)
                print(f"{'Spread':<30}: {spread_0:.4f}")
                print(f"{'Z-spread':<30}: {z_spread_0:.4f}")

                if rounded_qty1 and rounded_qty2 and (z_spread_0 > upper_bound or z_spread_0 < lower_bound):
                    capital_usage_percentage = (cost1 + cost2) / capital * 100
                    print(f"{'Capital usage percentage':<40}: {capital_usage_percentage:.2f}%\n") 

                    order1 = self.place_order(coin1, coin1_trade, rounded_qty1)
                    order2 = self.place_order(coin2, coin2_trade, rounded_qty2)

                    if order1 is None or order2 is None:
                        print("Error placing orders")
                        self.close_all_positions()
                        time.sleep(60)
                        continue

                    cnt = 0

                    while True:

                        if cnt == 0:
                            spread_i, z_spread_i = spread_0, z_spread_0
                        else:
                            price1 = self.get_price(coin1)
                            price2 = self.get_price(coin2)
                            hedge_ratio, intercept = kalman_model.update(np.log(price1), np.log(price2))
                            spread_i, z_spread_i = self.compute_spread_from_prices(price1, price2, hedge_ratio, intercept, mean_spread, std_spread)
                            
                        print(f"Iteration {cnt}, Spread: {spread_i:.4f}, Z-spread: {z_spread_i:.4f}, Signal: {signal}, Bounds: ({upper_bound:.2f}, {lower_bound:.2f})")

                        if self.should_close_position(signal, z_spread_i, z_spread_0, upper_bound, lower_bound):
                            self.close_all_positions()
                            break
                        else:
                            cnt += 1
                            
                        interval_seconds = self.convert_interval_to_seconds(self.config['INTERVAL'])
                        time.sleep(interval_seconds)
                        
                else:
                    print("Error placing orders")
                    time.sleep(60)
            else:
                print("No cointegrated pairs found")
                time.sleep(60)

    def compute_spread(self, coin1, coin2, hedge_ratio, intercept, mean_spread, std_spread):
        price1 = float(self.client.futures_symbol_ticker(symbol=coin1)['price'])
        price2 = float(self.client.futures_symbol_ticker(symbol=coin2)['price'])
        return self.compute_spread_from_prices(price1, price2, hedge_ratio, intercept, mean_spread, std_spread)
    
    def compute_spread_from_prices(self, price1, price2, hedge_ratio, intercept, mean_spread, std_spread):
        spread = np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
        z_spread = (spread - mean_spread) / std_spread
        return spread, z_spread

    def should_close_position(self, signal, z_spread_i, init_z_spread, upper_bound, lower_bound):
        if signal == 1 and z_spread_i > 0:
            return True
        elif signal == 1 and z_spread_i < (init_z_spread - 2):
            return True
        elif signal == -1 and z_spread_i < 0:
            return True
        elif signal == -1 and z_spread_i > (init_z_spread + 2):
            return True
        else:
            return False

    def get_price(self, coin):
        return float(self.client.futures_symbol_ticker(symbol=coin)['price'])

    def convert_interval_to_seconds(self, interval):
        unit = interval[-1]
        amount = int(interval[:-1])
        
        if unit == 'm':
            return amount * 60
        elif unit == 'h':
            return amount * 3600
        elif unit == 'd':
            return amount * 86400
        elif unit == 'w':
            return amount * 604800
        elif unit == 'M':
            # Rough approximation, as months can vary in length
            return amount * 2592000
        else:
            raise ValueError("Invalid interval format")
