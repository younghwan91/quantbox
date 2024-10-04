import numpy as np
import math
import time
from datetime import datetime, timedelta
from binance_downloader import BinanceDownloader
from pair_selector import CryptoPairSelector
import logging

logger = logging.getLogger('CryptoTrader')

class CryptoTrader:
    def __init__(self, client, config, enable_plot=0):
        self.client = client
        self.config = config
        self.enable_plot = enable_plot
        logging.info("CryptoTrader initialized with config: %s", config)
        
    def find_cointegrated_pairs(self): 
        logging.info("Finding cointegrated pairs...")
        
        selector = CryptoPairSelector(self.config)
        coint_pairs, data = selector.select_trading_pairs()
        suitable_pairs_sorted = selector.select_best_pair(data, coint_pairs, self.enable_plot)
        if suitable_pairs_sorted:
            logging.info("Suitable pairs found: %s", suitable_pairs_sorted)
        else:
            logging.warning("No cointegrated pairs found.")
        return suitable_pairs_sorted

    def download_data(self):
        logging.info("Downloading data...")
        downloader = BinanceDownloader(self.config, self.client)
        downloader.download_data()
        logging.debug("Data downloaded successfully.")

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
                        logging.debug("Symbol info for %s: step_size=%s, tick_size=%s, min_notional=%s, quantity_precision=%s",
                                      symbol, step_size, tick_size, min_notional, quantity_precision)
                        return step_size, tick_size, min_notional, quantity_precision
            logging.error("No symbol info found for %s", symbol)
            return None, None, None, None
        except Exception as e:
            logging.error("An exception occurred while fetching symbol info for %s: %s", symbol, e)
            return None, None, None, None

    def round_step_size(self, quantity, step_size):
        rounded = math.floor(quantity / step_size) * step_size
        logging.debug("Rounded quantity: original=%s, step_size=%s, rounded=%s", quantity, step_size, rounded)
        return rounded

    def validate_order(self, symbol, side, quantity):
        try:
            step_size, tick_size, min_notional, quantity_precision = self.get_symbol_info(symbol)
            if None in (step_size, tick_size, min_notional, quantity_precision):
                logging.error("Symbol info incomplete for %s. Cannot validate order.", symbol)
                return None, None

            rounded_quantity = self.round_step_size(quantity, step_size)
            rounded_quantity = round(rounded_quantity, quantity_precision)
            price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
            notional = rounded_quantity * price
            
            logging.info("Validating Order for %s | Side: %s", symbol, side)
            logging.debug("Initial Quantity: %.4f, Rounded Quantity: %.4f, Step Size: %s, Tick Size: %s, Min Notional: %s, Quantity Precision: %s, Price: %.4f, Order Notional: %.4f",
                          quantity, rounded_quantity, step_size, tick_size, min_notional, quantity_precision, price, notional)
            
            if notional < min_notional:
                logging.warning("Order notional %.4f is less than the minimum notional %.4f for %s", notional, min_notional, symbol)
                raise ValueError(f"Order notional {notional:.4f} is less than the minimum notional {min_notional:.4f}")
            
            return rounded_quantity, notional
        except Exception as e:
            logging.error("An exception occurred during order validation for %s: %s", symbol, e)
            return None, None

    def place_order(self, symbol, side, rounded_quantity):
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=self.config['LEVERAGE'])
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=rounded_quantity,
            )
            logging.info("Placed order for %s | Side: %s | Quantity: %s | Order ID: %s", symbol, side, rounded_quantity, order.get('orderId'))
            return order
        except Exception as e:
            logging.error("An exception occurred while placing order for %s: %s", symbol, e)
            return None

    def place_order_with_retry(self, symbol, side, quantity, max_retries=3, backoff_factor=2):
        attempt = 0
        while attempt < max_retries:
            order = self.place_order(symbol, side, quantity)
            if order is not None:
                return order
            attempt += 1
            wait_time = backoff_factor ** attempt
            logging.warning("Retrying to place order for %s in %s seconds... (Attempt %s/%s)", symbol, wait_time, attempt, max_retries)
            time.sleep(wait_time)
        logging.error("Failed to place order for %s after %s attempts.", symbol, max_retries)
        return None

    def is_order_filled(self, symbol, order_id):
        try:
            order = self.client.futures_get_order(symbol=symbol, orderId=order_id)
            executed_qty = float(order['executedQty'])
            original_qty = float(order['origQty'])
            filled = executed_qty >= original_qty
            logging.debug("Order %s for %s filled: %s (Executed: %s, Original: %s)", order_id, symbol, filled, executed_qty, original_qty)
            return filled
        except Exception as e:
            logging.error("An exception occurred while checking order status for %s (Order ID: %s): %s", symbol, order_id, e)
            return False

    def get_current_positions(self):
        try:
            positions = self.client.futures_account()['positions']
            current_positions = [position for position in positions if float(position['positionAmt']) != 0.0]
            logging.debug("Current open positions: %s", current_positions)
            return current_positions
        except Exception as e:
            logging.error("An exception occurred while fetching positions: %s", e)
            return []

    def close_position(self, symbol, quantity):
        try:
            side = 'SELL' if quantity > 0 else 'BUY'
            quantity = abs(quantity)
            step_size, tick_size, min_notional, quantity_precision = self.get_symbol_info(symbol)
            if None in (step_size, tick_size, min_notional, quantity_precision):
                logging.error("Symbol info incomplete for %s. Cannot close position.", symbol)
                return
            
            rounded_quantity = self.round_step_size(quantity, step_size)
            rounded_quantity = round(rounded_quantity, quantity_precision)
            if rounded_quantity == 0.0:
                logging.warning("Rounded quantity is zero for %s, unable to close position", symbol)
                return
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=rounded_quantity,
            )
            logging.info("Closed position for %s | Side: %s | Quantity: %s | Order ID: %s", symbol, side, rounded_quantity, order.get('orderId'))
        except Exception as e:
            logging.error("An exception occurred while closing position for %s: %s", symbol, e)


    def close_all_positions(self):
        current_positions = self.get_current_positions()
        if not current_positions:
            logging.info("No open positions to close.")
            return
        
        for position in current_positions:
            symbol = position['symbol']
            quantity = float(position['positionAmt'])
            logging.info("Closing position for %s, Quantity: %s", symbol, quantity)
            self.close_position(symbol, quantity)

        time.sleep(2)  # Wait for positions to close
        remaining_positions = self.get_current_positions()
        if remaining_positions:
            logging.error("Some positions are still open after attempting to close all. Remaining positions: %s", remaining_positions)
            raise RuntimeError("Failed to close all positions. Stopping further actions.")
        else:
            logging.info("All positions closed successfully.")            

    def print_futures_balance(self):
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == 'USDT':
                    logging.info("Futures Account Balance:")
                    logging.info("="*50)
                    logging.info("Asset: %s", balance['asset'])
                    logging.info("Available Balance: %.2f USDT", float(balance['availableBalance']))
                    logging.info("Cross Wallet Balance: %.2f USDT", float(balance['crossWalletBalance']))
                    logging.info("Cross Unrealized PNL: %.2f USDT", float(balance['crossUnPnl']))
                    logging.info("Total Wallet Balance: %.2f USDT", float(balance['balance']))
                    logging.info("="*50)
                    break
        except Exception as e:
            logging.error("An exception occurred while fetching futures account balance: %s", e)


    def get_available_balance(self, asset='USDT'):
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == asset:
                    available_balance = float(balance['availableBalance'])
                    logging.info("Available balance for %s: %.2f", asset, available_balance)
                    return available_balance
            logging.warning("Asset %s not found in futures account balance.", asset)
            return 0.0
        except Exception as e:
            logging.error("An exception occurred while fetching futures account balance for %s: %s", asset, e)
            return 0.0
        

    def run_trading_cycle(self):
        logging.info("Starting trading cycle...")
        while True:
            try:
                open_positions = self.get_current_positions()
                if open_positions:
                    logging.info("Open positions detected. Closing all positions before proceeding.")
                    self.close_all_positions()
                
                capital = self.get_available_balance()
                capital = capital * 1.0

                self.download_data()
                
                suitable_pairs_sorted = self.find_cointegrated_pairs()
                if not suitable_pairs_sorted:
                    logging.info("No suitable pairs found. Retrying after delay.")
                    time.sleep(60)
                    continue # Restart the trading cycle
                
                for pair, info_pair in suitable_pairs_sorted:
                    
                    coin1, coin2 = pair
                    hedge_ratio = info_pair['hedge_ratio']
                    intercept = info_pair['intercept']
                    mean_spread = info_pair['mean_spread']
                    std_spread = info_pair['std_spread']
                    half_life = info_pair['half_life']

                    latest_price1 = self.get_price(coin1)
                    latest_price2 = self.get_price(coin2)
                    spread_0, z_spread_0 = self.compute_spread_from_prices(latest_price1, latest_price2, hedge_ratio, intercept, mean_spread, std_spread)
                    logging.info(f"Latest Spread for Pair {pair}: Spread={spread_0:.4f}, Z-Spread={z_spread_0:.4f}, Half-Life={half_life:.2f}")

                    if self.config['THRESHOLD'] < np.abs(z_spread_0) < 5.0:
                        logging.info(f"Pair {pair} still meets z-score criteria with Z-Spread={z_spread_0:.4f}")
                    
                        # Allocate Signal Based on the Latest Z-Spread
                        signal = 1 if z_spread_0 < 0 else -1
                        trade_coin1 = 'BUY' if signal == 1 else 'SELL'
                        trade_coin2 = 'SELL' if signal == 1 else 'BUY'
                        logging.info(f"Allocated Signal for Pair {pair}: {trade_coin1}/{trade_coin2} based on Z-Spread: {z_spread_0:.4f}")

                        trading_charge = self.config['TRADING_CHARGE'] + self.config['SLIPPAGE']
                        stop_loss = self.config['STOP_LOSS']
                        risk_reward_ratio = self.config['RISK_REWARD_RATIO']
                        frac_hold = self.config['FRAC_HOLD']
                        
                        E = capital * (1 - 2 * trading_charge)
                        S1 = E / (1 + hedge_ratio)
                        S2 = hedge_ratio * S1

                        qty1 = S1 / latest_price1
                        qty2 = S2 / latest_price2

                        logging.info("Trading Parameters for Pair %s/%s:", coin1, coin2)
                        logging.debug("Capital: %s, Effective Capital: %.2f, Hedge Ratio: %.4f, Quantity1: %.4f, Quantity2: %.4f",
                                capital, E, hedge_ratio, qty1, qty2)
  
                        rounded_qty1, cost1 = self.validate_order(coin1, trade_coin1, qty1)
                        rounded_qty2, cost2 = self.validate_order(coin2, trade_coin2, qty2)

                        if rounded_qty1 is None or rounded_qty2 is None:
                            logging.error(f"Order validation failed for pair {pair}. Skipping to next pair.")
                            continue  # Skip to the next pair
                        else:
                            capital_usage_percentage = (cost1 + cost2) / capital * 100
                            logging.info("Capital usage percentage: %.2f%%", capital_usage_percentage)

                        # Log Validated Order Details
                        logging.info("Validated Orders:")
                        logging.debug("Coin1: %s, Action: %s, Quantity: %.4f, Cost: %.4f", coin1, trade_coin1, rounded_qty1, cost1)
                        logging.debug("Coin2: %s, Action: %s, Quantity: %.4f, Cost: %.4f", coin2, trade_coin2, rounded_qty2, cost2)

                        order1 = self.place_order_with_retry(coin1, trade_coin1, rounded_qty1)
                        order2 = self.place_order_with_retry(coin2, trade_coin2, rounded_qty2)

                        if order1 is None or order2 is None:
                            logging.error("Error placing orders. Closing all positions and retrying after delay.")
                            self.close_all_positions()
                            time.sleep(60)
                            continue

                        order1_filled = self.is_order_filled(coin1, order1['orderId'])
                        order2_filled = self.is_order_filled(coin2, order2['orderId'])

                        if not order1_filled or not order2_filled:
                            logging.warning("One or both orders are not filled. Closing all positions and searching for new pairs.")
                            self.close_all_positions()
                            time.sleep(60)
                            continue  # Restart the trading cycle
                        
                        position_open_time = datetime.now()
                        max_hold_seconds = frac_hold * 24 * 3600

                        cnt = 0
                        while True:
                            if cnt == 0:
                                spread_i, z_spread_i = spread_0, z_spread_0
                            else:
                                price1 = self.get_price(coin1)
                                price2 = self.get_price(coin2)
                                spread_i, z_spread_i = self.compute_spread_from_prices(price1, price2, hedge_ratio, intercept, mean_spread, std_spread)
                                
                            current_time = datetime.now()
                            elapsed_time = (current_time - position_open_time).total_seconds()
                            logging.info(">> Iter. %s, Time: %s, Spread: %.4f, Z-spread: %.4f, Signal: %s", cnt, current_time.strftime("%H:%M:%S"), spread_i, z_spread_i, signal)
                            
                            if elapsed_time > max_hold_seconds:
                                logging.info("Maximum holding time of %.2f hours reached. Closing all positions.", frac_hold * 24)
                                self.close_all_positions()
                                break

                            if self.should_close_position(signal, z_spread_i, z_spread_0, stop_loss, risk_reward_ratio):
                                logging.info("Conditions met to close positions based on spread thresholds.")
                                self.close_all_positions()
                                break
                            else:
                                cnt += 1
                                
                            interval_seconds = self.convert_interval_to_seconds(self.config['INTERVAL'])
                            logging.debug("Sleeping for %s seconds before next check.", interval_seconds)
                            time.sleep(interval_seconds)
                        
                        break # Exit the for-loop and restart the while-loop
                            
                    else:
                        logging.info(f"Trading signals not strong enough for pair {pair}. Trying next pair.")
                        continue  # Try the next pair
            
            except Exception as e:
                logging.error("An exception occurred in trading cycle: %s", e)
                self.close_all_positions()
                time.sleep(60)
    
    def compute_spread_from_prices(self, price1, price2, hedge_ratio, intercept, mean_spread, std_spread):
        spread = np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
        z_spread = (spread - mean_spread) / std_spread
        logging.debug("Spread calculation: spread=%.4f, z_spread=%.4f", spread, z_spread)
        return spread, z_spread

    def should_close_position(self, signal, z_spread_i, init_z_spread, stop_loss, risk_reward_ratio):
        close = False
        if signal == 1:
            if z_spread_i > init_z_spread + (risk_reward_ratio * stop_loss) or z_spread_i < init_z_spread - stop_loss:
                close = True
        elif signal == -1:
            if z_spread_i < init_z_spread - (risk_reward_ratio * stop_loss) or z_spread_i > init_z_spread + stop_loss:
                close = True
        if close:
            logging.info("Position closure condition met: z_spread_i=%.4f, init_z_spread=%.4f", z_spread_i, init_z_spread)
        return close

    def get_price(self, coin):
        try:
            price = float(self.client.futures_symbol_ticker(symbol=coin)['price'])
            logging.debug("Fetched price for %s: %.4f", coin, price)
            return price
        except Exception as e:
            logging.error("An exception occurred while fetching price for %s: %s", coin, e)
            return 0.0

    def convert_interval_to_seconds(self, interval):
        try:
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
        except Exception as e:
            logging.error("Error converting interval '%s' to seconds: %s", interval, e)
            raise
