# binance_downloader.py

import os
import csv
import pandas as pd
from binance.enums import HistoricalKlinesType
from tqdm import tqdm
import numpy as np
import logging
import time
import shutil

logger = logging.getLogger('BinanceDownloader')

class BinanceDownloader:
    def __init__(self, config, client):
        self.client = client
        self.config = config
        self.trading_coins = self.get_trading_coins(
            self.client, config["QUOTE_COIN"], config["N_COINS"]
        )
        self.saving_dir = config["SAVING_DIR"]
        self.backtest_dir = os.path.join(self.saving_dir, 'backtest')
        self.ensure_directories()

    def ensure_directories(self):
        # Ensure the main saving directory exists
        os.makedirs(self.saving_dir, exist_ok=True)
        
        try:
            if os.path.exists(self.backtest_dir):
                shutil.rmtree(self.backtest_dir)
                logger.debug(f"Deleted entire directory: {self.backtest_dir}")
            os.makedirs(self.backtest_dir, exist_ok=True)
            logger.info(f"Created directory: {self.backtest_dir}")
        except Exception as e:
            logger.error(f"Failed to reset {self.backtest_dir}. Reason: {e}") 

    def get_trading_coins(self, client, quote_coin, n_coins, sort_by_volume=True):
        futures_exchange_info = client.futures_exchange_info()

        trading_coins = [
            info['symbol']
            for info in futures_exchange_info['symbols']
            if info['symbol'].endswith(quote_coin) and info['status'] == 'TRADING'
        ]

        logger.info(f"Number of trading pairs: {len(trading_coins)}")
        assert n_coins <= len(trading_coins), "N_COINS is greater than the number of trading coins"

        if sort_by_volume:
            logger.info("Fetching 24-hour trading volume for each coin")
            volume_dict = {}
            for coin in trading_coins:
                try:
                    ticker = client.get_ticker(symbol=coin)
                    volume_dict[coin] = float(ticker['quoteVolume'])
                except Exception as e:
                    logger.error(f"Error fetching ticker for {coin}: {e}")
                    continue

            # Sort coins based on trading volume in ascending order
            sorted_coins = sorted(volume_dict.items(), key=lambda x: x[1])  # Remove reverse=True
            num_coins = len(sorted_coins)
            logger.info(f"Number of trading pairs after sorting by volume: {num_coins}")

            logger.info(f"Selecting bottom {n_coins} coins based on trading volume")
            selected_coins = [coin for coin, volume in sorted_coins[:n_coins]]
        else:
            logger.info(f"Selecting first {n_coins} coins")
            selected_coins = trading_coins[:n_coins]

        return selected_coins

    def save_trading_coins(self):
        csv_path = os.path.join(self.saving_dir, self.config["CSV_FILENAME"])
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.trading_coins)
        logger.info(f"Saved trading coins to {self.config['CSV_FILENAME']}")

    def fetch_and_save_klines(self, coin, interval, start_str, end_str):
        """
        Fetch and save klines for a given coin and interval.
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=coin,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=1000,
                klines_type=HistoricalKlinesType.FUTURES
            )
            if not klines:
                logger.warning(f"No klines fetched for {coin}.")
                return

            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close',
                'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            df = df.astype(float)
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df.set_index('Open Time', inplace=True)
            save_path = os.path.join(self.saving_dir, f'{coin}.pkl')
            df.to_pickle(save_path)
            logger.info(f"Saved klines for {coin} to {save_path}")
        except Exception as e:
            logger.error(f"Error fetching klines for {coin}: {e}")

    def fetch_and_save_klines_for_backtest(self, coin, interval, start_str, end_str):
        """
        Fetch and save klines for a given coin and interval in multiple batches 
        due to the 1000 kline limit per request.
        Combines all batches into a single DataFrame and saves as one pickle file.
        """
        start_ts = self._convert_to_timestamp(start_str)
        end_ts = self._convert_to_timestamp(end_str)
        limit = 1000
        step = self._interval_to_milliseconds(interval)

        logger.info(f"Starting download for {coin} from {start_str} to {end_str} with interval {interval}")

        current_ts = start_ts
        all_klines = []  # List to accumulate klines

        while current_ts < end_ts:
            batch_start = current_ts
            batch_end = min(current_ts + limit * step, end_ts)

            # Convert timestamps back to string format required by Binance
            batch_start_str = self._convert_to_str(batch_start)
            batch_end_str = self._convert_to_str(batch_end)

            try:
                klines = self.client.get_historical_klines(
                    symbol=coin,
                    interval=interval,
                    start_str=batch_start_str,
                    end_str=batch_end_str,
                    limit=limit,
                    klines_type=HistoricalKlinesType.FUTURES
                )
                if not klines:
                    logger.info(f"No more klines fetched for {coin}. Ending download.")
                    break

                all_klines.extend(klines)
                logger.info(f"Fetched {len(klines)} klines for {coin} from {batch_start_str} to {batch_end_str}")

                # Update the current timestamp to the last kline's close time plus 1 millisecond
                last_kline_close_time = klines[-1][6]  # 'Close Time' is at index 6
                current_ts = last_kline_close_time + 1

                # To respect API rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching klines for {coin} between {batch_start_str} and {batch_end_str}: {e}")
                # Skip to the next batch after logging the error
                current_ts = batch_end + 1
                continue

        if all_klines:
            # Convert all accumulated klines to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close',
                'Volume', 'Close Time', 'Quote Asset Volume',
                'Number of Trades', 'Taker Buy Base Asset Volume',
                'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            df = df.astype(float)
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df.set_index('Open Time', inplace=True)

            # Sort the DataFrame by index to ensure chronological order
            df.sort_index(inplace=True)

            # Save the combined DataFrame as a single pickle file
            save_path = os.path.join(self.backtest_dir, f'{coin}_backtest.pkl')
            df.to_pickle(save_path)
            logger.info(f"Saved {len(df)} klines to {save_path}")
        else:
            logger.warning(f"No klines fetched for {coin}. No file saved.")

    def download_data(self):
        """
        Download klines for all trading coins and save them.
        """
        self.save_trading_coins()

        for coin in tqdm(self.trading_coins, desc="Downloading klines"):
            self.fetch_and_save_klines(
                coin,
                self.config["INTERVAL"],
                self.config["START_STR"],
                self.config["END_STR"]
            )

    def download_data_for_backtest(self):
        """
        Download klines specifically for backtesting, handling multiple batches.
        """
        self.save_trading_coins()

        for coin in tqdm(self.trading_coins, desc="Downloading backtest data"):
            self.fetch_and_save_klines_for_backtest(
                coin,
                self.config["INTERVAL"],
                self.config["START_STR"],
                self.config["END_STR"]
            )

    def _convert_to_timestamp(self, date_str):
        """
        Convert a date string to a Unix timestamp in milliseconds.
        """
        return int(pd.to_datetime(date_str).timestamp() * 1000)

    def _convert_to_str(self, timestamp):
        """
        Convert a Unix timestamp in milliseconds to a date string.
        """
        return pd.to_datetime(timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')

    def _interval_to_milliseconds(self, interval):
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
