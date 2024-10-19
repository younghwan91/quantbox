import os
import csv
import pandas as pd
from binance.enums import HistoricalKlinesType
from tqdm import tqdm
import numpy as np
import logging
import time
from itertools import combinations
from datetime import datetime
from statsmodels.tsa.stattools import coint
from scipy.stats import shapiro


logger = logging.getLogger('BinanceDownloader')

class BinanceDownloader:
    def __init__(self, config, client):
        self.client = client
        self.saving_dir = config["SAVING_DIR"]
        self.trading_coins = self.get_trading_coins(self.client, config["QUOTE_COIN"], config["N_COINS"])
        self.config = config
        self.ensure_directories()

    def ensure_directories(self):
        # Ensure the main saving directory exists
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        
        # Ensure the 'backtest' subdirectory exists
        self.backtest_dir = os.path.join(self.saving_dir, 'backtest')
        if not os.path.exists(self.backtest_dir):
            os.makedirs(self.backtest_dir)
        
        # Ensure the 'pairs' subdirectory within 'backtest' exists
        self.pairs_dir = os.path.join(self.backtest_dir, 'pairs')
        if not os.path.exists(self.pairs_dir):
            os.makedirs(self.pairs_dir)

    def get_trading_coins(self, client, quote_coin, n_coins, sort_by_volume=True):
        futures_exchange_info = client.futures_exchange_info()

        trading_coins = [info['symbol'] for info in futures_exchange_info['symbols'] if info['symbol'].endswith(quote_coin) and info['status'] == 'TRADING']

        print(f"Number of trading pairs: {len(trading_coins)}")
        assert n_coins <= len(trading_coins), "N_COINS is greater than the number of trading coins"

        if sort_by_volume:
            logger.info("Fetching 24-hour trading volume for each coin")
            volume_dict = {}
            for coin in trading_coins:
                try:
                    ticker = client.get_ticker(symbol=coin)
                    volume_dict[coin] = float(ticker['volume'])
                except Exception as e:
                    logger.error(f"Error fetching ticker for {coin}: {e}")
                    continue

            # Sort coins based on trading volume in descending order
            sorted_coins = sorted(volume_dict.items(), key=lambda x: x[1], reverse=True)
            num_coins = len(sorted_coins)
            logger.info(f"Number of trading pairs: {num_coins}")

            logger.info(f"Selecting top {n_coins} coins based on trading volume")
            top_coins = [coin for coin, volume in sorted_coins[:n_coins]]
        else:
            logger.info(f"Selecting first {n_coins} coins")
            top_coins = trading_coins[:n_coins]

        return top_coins


    def save_trading_coins(self):
        with open(os.path.join(self.saving_dir, self.config["CSV_FILENAME"]), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.trading_coins)

    def fetch_and_save_klines(self, coin, interval, start_str, end_str):
        
        """
        Fetch and save klines for a given coin and interval.
        1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        
        klines = self.client.get_historical_klines(
            symbol=coin,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=1000,
            klines_type=HistoricalKlinesType.FUTURES
        )
        df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 
                            'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
        df = df.astype(float)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        df.to_pickle(os.path.join(self.saving_dir, f'{coin}.pkl'))

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
                # Implement retry logic or skip to next batch
                # For simplicity, we'll skip to the next batch after logging the error
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

        self.save_trading_coins()

        for coin in tqdm(self.trading_coins):
            self.fetch_and_save_klines(coin, self.config["INTERVAL"], self.config["START_STR"], self.config["END_STR"])
    
    def download_data_for_backtest(self):

        self.save_trading_coins()

        for coin in tqdm(self.trading_coins, desc="Downloading backtest data"):
            self.fetch_and_save_klines_for_backtest(
                coin, 
                self.config["INTERVAL"], 
                self.config["START_STR"], 
                self.config["END_STR"]
            )

    def prepare_pairs_for_backtest(
        self, 
        window_size: str, 
        overlap: str, 
        output_filename: str = "pairs_for_backtest.csv"
    ):
        """
        Prepares pair trading data by aligning all coin data to a common end time,
        slicing into windows with specified size and overlap from end to start,
        generating all possible unique coin pairs within each window, and saving 
        the pairs in chronological order.

        Parameters:
        - window_size (str): Size of each window (e.g., '2D' for 2 days).
        - overlap (str): Overlap between windows (e.g., '4H' for 4 hours).
        - output_filename (str): Filename to save the pairs data.
        """
        logger.info("Preparing pairs for backtest")

        # Step 1: Load all coin data
        coin_data = self._load_all_coin_data()
        logger.info(f"Loaded data for {len(coin_data)} coins")

        if not coin_data:
            logger.error("No coin data available to prepare pairs.")
            return

        # Step 2: Find common end time
        common_end_time = self._find_common_end_time(coin_data)
        logger.info(f"Common end time for all coins: {common_end_time}")

        # Step 3: Truncate all DataFrames to the common end time
        coin_data = {coin: df[:common_end_time] for coin, df in coin_data.items()}

        # Step 4: Slice data into windows from end to start
        windows = self._generate_time_windows_reverse(coin_data, window_size, overlap)
        logger.info(f"Generated {len(windows)} windows")

        if not windows:
            logger.error("No windows generated. Check window size and overlap.")
            return

        # Step 5: Generate all possible unique pairs within each window
        all_pairs = self._generate_all_pairs_reverse(coin_data, windows)
        logger.info(f"Generated {len(all_pairs)} coin pairs for backtest")

        if not all_pairs:
            logger.warning("No pairs generated. Check if window sizes are appropriate.")
            return

        # Step 6: Filter pairs based on correlation and cointegration
        correlation_threshold = self.config.get('CORR_THRESHOLD', 0.5)
        pvalue_threshold = self.config.get('COINT_THRESHOLD', 0.05)

        filtered_pairs = self._filter_pairs(all_pairs, correlation_threshold, pvalue_threshold)
        logger.info(f"Filtered down to {len(filtered_pairs)} coin pairs after applying thresholds")

        if not filtered_pairs:
            logger.warning("No pairs left after filtering. Consider lowering thresholds.")
            return

        # Step 7: Save the filtered pairs data
        self._save_pairs(filtered_pairs, output_filename)
        logger.info(f"Saved filtered pairs data to {output_filename}")

    def _filter_pairs(self, all_pairs, corr_threshold, pvalue_threshold):
        """
        Filters pairs based on Pearson correlation and cointegration test.

        Parameters:
        - all_pairs (list of dicts): The pairs data.
        - corr_threshold (float): Minimum Pearson correlation to qualify.
        - pvalue_threshold (float): Maximum p-value for cointegration to qualify.

        Returns:
        - list of dicts: Filtered pairs data.
        """
        filtered = []
        for pair in tqdm(all_pairs, desc="Filtering pairs"):
            data1 = pair['data1']['Close']
            data2 = pair['data2']['Close']
            log_data1 = np.log(data1)
            log_data2 = np.log(data2)

            # Filter 1: Pearson correlation
            correlation = data1.corr(data2)
            if np.abs(correlation) < corr_threshold:
                continue 
            
            # Filter 2: Hedge ratio
            try:
                hedge_ratio = np.polyfit(log_data2, log_data1, 1)[0]
            except np.RankWarning as e:
                logger.warning(f"Polyfit failed: {e}")
                continue
            hedge_const = 3.0
            if (1 / hedge_const) > hedge_ratio or hedge_ratio > hedge_const:
                continue
            
            # Filter 3: Z-score normality test
            spread = log_data1 - hedge_ratio * log_data2
            z_spread = (spread - np.mean(spread)) / np.std(spread)
            if np.abs(z_spread.iloc[-1]) < 2:
                continue

            # Filter 4: Shapiro-Wilk normality test
            shapiro_result = shapiro(spread)
            if shapiro_result.pvalue < 0.05:
                continue

            # Filter 5: Cointegration test
            _, p_value, _ = coint(log_data1, log_data2)
            if p_value > pvalue_threshold:
                continue  

            filtered.append(pair)

        return filtered

    def _load_all_coin_data(self):
        """
        Loads all coin data from pickle files into a dictionary.

        Returns:
        - dict: A dictionary with coin symbols as keys and DataFrames as values.
        """
        coin_data = {}
        for coin in self.trading_coins:
            file_path = os.path.join(self.backtest_dir, f"{coin}_backtest.pkl")
            if os.path.exists(file_path):
                df = pd.read_pickle(file_path)
                if not df.empty:
                    coin_data[coin] = df
                    logger.debug(f"Loaded data for {coin}")
                else:
                    logger.warning(f"Data file for {coin} is empty.")
            else:
                logger.warning(f"Data file for {coin} not found at {file_path}")
        return coin_data

    def _find_common_end_time(self, coin_data):
        """
        Finds the latest common timestamp where all coins have data.

        Parameters:
        - coin_data (dict): Dictionary of coin DataFrames.

        Returns:
        - pd.Timestamp: The common end timestamp.
        """
        end_times = [df.index[-1] for df in coin_data.values()]
        common_end_time = min(end_times)
        return common_end_time

    def _generate_time_windows_reverse(self, coin_data, window_size: str, overlap: str):
        """
        Generates a list of (start, end) tuples representing each window, 
        slicing from end to start.

        Parameters:
        - coin_data (dict): Dictionary of coin DataFrames.
        - window_size (str): Window size (e.g., '2D').
        - overlap (str): Overlap duration (e.g., '4H').

        Returns:
        - list of tuples: Each tuple contains (window_start, window_end).
        """
        # Assuming all DataFrames have the same index frequency
        first_coin = next(iter(coin_data.values()))
        start_time = first_coin.index[0]
        end_time = first_coin.index[-1]

        window_delta = pd.to_timedelta(window_size)
        overlap_delta = pd.to_timedelta(overlap)
        step_delta = window_delta - overlap_delta

        windows = []
        current_end = end_time
        while current_end - window_delta >= start_time:
            current_start = current_end - window_delta
            windows.append((current_start, current_end))
            current_end -= step_delta

        return windows

    def _generate_all_pairs_reverse(self, coin_data, windows):
        """
        Generates all possible unique coin pairs within each window, 
        considering only coins with complete data in the window.

        Parameters:
        - coin_data (dict): Dictionary of coin DataFrames.
        - windows (list of tuples): List of (start, end) window timestamps.

        Returns:
        - list of dicts: Each dict contains 'window_start', 'window_end', 'pair', 'data1', 'data2'.
        """
        all_pairs = []
        coins = list(coin_data.keys())

        for window_start, window_end in tqdm(windows, desc="Generating pairs"):
            # Determine which coins have complete data for the window
            available_coins = []
            for coin in coins:
                df = coin_data[coin]
                if window_start in df.index and window_end in df.index:
                    # Ensure there are no missing timestamps within the window
                    window_df = df.loc[window_start:window_end]
                    expected_length = int((window_end - window_start) / (df.index[1] - df.index[0])) + 1
                    if len(window_df) == expected_length:
                        available_coins.append(coin)
                else:
                    continue

            if len(available_coins) < 2:
                logger.warning(f"Not enough coins with complete data in window {window_start} to {window_end}")
                continue

            # Generate all unique pairs from the available coins
            coin_pairs = list(combinations(available_coins, 2))

            for pair in coin_pairs:
                coin1, coin2 = pair
                df1 = coin_data[coin1].loc[window_start:window_end]
                df2 = coin_data[coin2].loc[window_start:window_end]

                # Additional check to ensure both DataFrames have data for the entire window
                if df1.empty or df2.empty:
                    logger.warning(f"No data for pair {pair} in window {window_start} to {window_end}")
                    continue

                pair_data = {
                    'window_start': window_start,
                    'window_end': window_end,
                    'pair': pair,
                    'data1': df1,
                    'data2': df2
                }
                all_pairs.append(pair_data)

        return all_pairs

    def _save_pairs(self, all_pairs, output_filename):
        """
        Saves the pairs data to a CSV file with references to pickle files.

        Parameters:
        - all_pairs (list of dicts): The pairs data.
        - output_filename (str): The filename to save the data.
        """
        output_path = os.path.join(self.saving_dir, output_filename)
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['window_start', 'window_end', 'pair', 'data1_path', 'data2_path'])

            for idx, pair in enumerate(tqdm(all_pairs, desc="Saving pairs")):
                window_start = pair['window_start'].strftime('%Y-%m-%d %H:%M:%S')
                window_end = pair['window_end'].strftime('%Y-%m-%d %H:%M:%S')
                pair_tuple = pair['pair']
                
                # Define paths to the sliced data pickles
                data1_path = os.path.join(
                    self.pairs_dir, 
                    f"{pair_tuple[0]}_{pair['window_start'].strftime('%Y%m%d%H%M%S')}_window.pkl"
                )
                data2_path = os.path.join(
                    self.pairs_dir, 
                    f"{pair_tuple[1]}_{pair['window_start'].strftime('%Y%m%d%H%M%S')}_window.pkl"
                )
                
                # Save the sliced DataFrames
                pair['data1'].to_pickle(data1_path)
                pair['data2'].to_pickle(data2_path)
                
                # Write the row to CSV
                writer.writerow([
                    window_start, 
                    window_end, 
                    f"{pair_tuple[0]}-{pair_tuple[1]}", 
                    data1_path, 
                    data2_path
                ])

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
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000  # Approximation
        }
        return ms.get(interval, None)