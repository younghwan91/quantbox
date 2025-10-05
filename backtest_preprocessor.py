# backtest_preprocessor.py

import os
import csv
import pickle
import logging
import random
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import statsmodels.api as sm  
import numba_utils as nu
from backtest_utils import calculate_half_life
from statsmodels.tsa.stattools import coint

logger = logging.getLogger('BacktestPreprocessor')
    
class BacktestPreprocessor:
    def __init__(self, config):
        self.saving_dir = config["SAVING_DIR"]
        self.backtest_dir = os.path.join(self.saving_dir, 'backtest')
        self.config = config
        self.trading_coins = self._load_trading_coins()

        self.pairs_csv_path = os.path.join(self.backtest_dir, config["PAIRS_CSV"])
        self.precomputed_path = os.path.join(self.backtest_dir, config["PRECOMPUTED_FILE"])

    def _load_trading_coins(self):
        csv_path = os.path.join(self.saving_dir, self.config["CSV_FILENAME"])
        if not os.path.exists(csv_path):
            logger.error(f"Trading coins CSV file not found at {csv_path}")
            return []
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header:
                trading_coins = header
            else:
                trading_coins = []
        logger.info(f"Loaded {len(trading_coins)} trading coins from CSV")
        return trading_coins
    
    def _filter_top_coins_by_volume(self, coin_data, windows, top_n=50):
        """
        Filters the top N coins by 24-hour rolling quote volume for each time window.

        Args:
            coin_data (dict): Dictionary of coin DataFrames.
            windows (list): List of (start, end) tuples representing time windows.
            top_n (int): Number of top coins to select based on 24-hour rolling volume.

        Returns:
            list: A list where each element is a list of top N coin names for a specific window.
        """
        logger.info("Filtering top coins by 24-hour rolling quote volume")
        top_coins_per_window = []

        for window_start, window_end in windows:
            # Calculate the total 24-hour volume for each coin at the end of the window
            volumes = {}
            for coin, df in coin_data.items():
                if window_end in df.index:
                    # Ensure the 24-hour rolling volume column exists
                    if '24h_Quote_Volume' not in df.columns:
                        inferred_freq = pd.infer_freq(df.index)
                        if inferred_freq is None:
                            time_diffs = df.index.to_series().diff().dropna()
                            median_delta = time_diffs.median()
                            inferred_freq_minutes = median_delta.total_seconds() / 60
                        else:
                            inferred_freq_minutes = pd.Timedelta(inferred_freq).total_seconds() / 60
                        rows_per_24h = int(round((24 * 60) / inferred_freq_minutes))
                        df['24h_Quote_Volume'] = df['Quote Asset Volume'].rolling(window=rows_per_24h, min_periods=1).sum()

                    # Get the 24-hour rolling volume at the end of the window
                    volumes[coin] = df.loc[window_end, '24h_Quote_Volume']

            # Sort coins by volume and select the top N
            # warning if number of coins are less than top_n
            if len(volumes) < top_n:
                logger.warning(f"Only {len(volumes)} coins available for window ending at {window_end}, which is less than the requested top_n={top_n}. Selecting all available coins.")
                top_n = len(volumes)

            top_coins = sorted(volumes, key=volumes.get, reverse=True)[:top_n]
            top_coins_per_window.append(top_coins)

        return top_coins_per_window

    def prepare_pairs_for_backtest(
        self, 
        window_size: str, 
        hour_shift: int, 
    ):
        """
        Prepares pair trading data by aligning all coin data to a common end time,
        slicing into windows with specified size and overlap from end to start,
        generating all possible unique coin pairs within each window, and saving 
        the pairs in chronological order.
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
        windows = self._generate_time_windows(coin_data, window_size, hour_shift)
        logger.info(f"Generated {len(windows)} windows")

        if not windows:
            logger.error("No windows generated. Check window size and overlap.")
            return
        
        # Step 5: Filter top 50 coins by 24-hour rolling quote volume
        top_coins_per_window = self._filter_top_coins_by_volume(coin_data, windows)
        logger.info(f"Allocated top {len(top_coins_per_window)} coins for each of the {len(windows)} windows")

        # Step 6 & 7: Generate and filter pairs on the fly
        correlation_threshold = self.config.get('CORR_THRESHOLD', 0.5)
        pvalue_threshold = self.config.get('COINT_THRESHOLD', 0.05)
        window_ratio = self.config.get('WINDOW_RATIO') 

        filtered_pairs = []
        for pair in self._generate_all_pairs(coin_data, windows, top_coins_per_window, correlation_threshold, pvalue_threshold, window_ratio):
            filtered_pairs.append(pair)
        logger.info(f"Filtered down to {len(filtered_pairs)} coin pairs after applying thresholds")

        if not filtered_pairs:
            logger.warning("No pairs left after filtering. Consider lowering thresholds.")
            return

        # Step 8: Save the filtered and precomputed pairs data
        self._save_pairs_info(filtered_pairs)
        logger.info(f"Saved precomputed pairs data to {self.precomputed_path}")

    def _load_all_coin_data(self):
        """
        Loads all coin data from pickle files into a dictionary.
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
        """
        end_times = [df.index[-1] for df in coin_data.values()]
        common_end_time = min(end_times)
        return common_end_time

    def _generate_time_windows(self, coin_data, window_size: str, hour_shift: int):
        """
        Generates a list of (start, end) tuples representing each window, 
        slicing from end to start.
        """
        # Assuming all DataFrames have the same index frequency
        first_coin = next(iter(coin_data.values()))
        start_time = first_coin.index[0]
        end_time = first_coin.index[-1]

        window_delta = pd.to_timedelta(window_size)
        step_delta = pd.to_timedelta(hour_shift, unit='h')

        windows = []
        current_end = end_time
        while current_end - window_delta >= start_time:
            current_start = current_end - window_delta
            windows.append((current_start, current_end))
            current_end -= step_delta

        return windows

    def _generate_all_pairs(self, coin_data, windows, top_coins_per_window, corr_threshold, pvalue_threshold, window_ratio):
        """
        Generates all possible unique coin pairs within each window, 
        considering only coins with complete data in the window.

        Args:
            coin_data (dict): Dictionary of coin DataFrames.
            windows (list): List of (start, end) tuples representing time windows.
            top_coins_per_window (list): List of top coins for each window.
            corr_threshold (float): Correlation threshold for filtering pairs.
            pvalue_threshold (float): P-value threshold for filtering pairs.
            window_ratio (float): Ratio of training to total window size.

        Returns:
            list: Filtered pairs with precomputed data for each window.
        """
        max_pairs_per_window = 50
        logger.info(f"Generating pairs with max {max_pairs_per_window} pairs per window")

        all_pairs = []

        for i, (window_start, window_end) in enumerate(tqdm(windows, position=0, leave=True)):
            coins = top_coins_per_window[i]
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
            random.shuffle(coin_pairs)  # Shuffle to avoid bias

            n = np.min([500, len(coin_pairs)])
            n_min_npd_pairs = []
            for pair in coin_pairs:
                coin1, coin2 = pair
                df1 = coin_data[coin1].loc[window_start:window_end]
                df2 = coin_data[coin2].loc[window_start:window_end]
                price1 = df1['Close'].values
                price2 = df2['Close'].values
                price1 = price1/price1[0]
                price2 = price2/price2[0]
                # normalized price difference
                split_idx = int(window_ratio * len(price1))
                npd =  np.sum((price1[:split_idx]-price2[:split_idx])**2) / len(price1)
                n_min_npd_pairs.append((pair, npd))

            n_min_npd_pairs = sorted(n_min_npd_pairs, key=lambda x: x[1])
            coin_pairs = [pair for pair, npd in n_min_npd_pairs[:n]]
    
            filtered_pairs = []

            for pair in coin_pairs:
                coin1, coin2 = pair
                df1 = coin_data[coin1].loc[window_start:window_end]
                df2 = coin_data[coin2].loc[window_start:window_end]

                # Additional check to ensure both DataFrames have identical indexes
                if not df1.index.equals(df2.index):
                    logger.warning(f"Time indexes do not match for pair {pair} in window {window_start} to {window_end}")
                    continue

                pair_data = {
                    'window_start': window_start,
                    'window_end': window_end,
                    'pair': pair,
                    'data1': df1[['High', 'Low', 'Close', 'Volume']],
                    'data2': df2[['High', 'Low', 'Close', 'Volume']]
                }
                
                close1 = pair_data['data1']['Close'].values.astype(np.float64)
                close2 = pair_data['data2']['Close'].values.astype(np.float64)
                if self._filter_pair(close1, close2, corr_threshold, pvalue_threshold, window_ratio):
                    filtered_pairs.append(pair_data)

                if len(filtered_pairs) >= max_pairs_per_window:
                    break

            all_pairs.extend(filtered_pairs)

        return all_pairs

    def _filter_pair(self, close1, close2, corr_threshold, pvalue_threshold, window_ratio):


        try:

            split_idx = int(window_ratio * len(close1))
            train_data1 = close1[:split_idx]
            train_data2 = close2[:split_idx]

            if (train_data1 <= 0).any() or (train_data2 <= 0).any():
                return False

            log_train_data1 = np.log(train_data1)
            log_train_data2 = np.log(train_data2)

            log_returns1 = nu.log_returns_numba(train_data1)
            log_returns2 = nu.log_returns_numba(train_data2)

            corr = nu.correlation_numba(log_returns1, log_returns2)
            if corr < 0.6:
                return False

            # hedge_ratio = nu.compute_slope(log_train_data2, log_train_data1)
            intercept, beta = nu.ols_regression(log_train_data2, log_train_data1)
            hedge_const = 4.0
            if not (1 / hedge_const) < beta < hedge_const:
                return False

            spread = log_train_data1 - beta * log_train_data2

            spread_std = spread.std()
            spread_mean = spread.mean()

            z_spread = (spread - spread_mean) / spread_std            
            if spread_std < 0.01:
                return False
            
            if np.abs(spread[-1]-spread_mean) < 0.02:
                return False

            jarque_bera_pvalue = nu.jarque_bera_numba(z_spread)
            if jarque_bera_pvalue < 0.05:
                return False

            results= coint(log_train_data1, log_train_data2)    
            if results[1] > 0.05:
                return False

            # coint_pvalue = nu.cointegration_adf_pvalue(log_train_data1, log_train_data2, hedge_ratio)
            # if coint_pvalue > pvalue_threshold:
            #     return False

            # t_stat = nu.cointegration_test(log_train_data2, log_train_data1, intercept, beta)
            # if t_stat < -4: # 3.7
            #     return False
            
            # hurst_exponent = nu.hurst(spread)
            # print(f"Hurst Exponent: {hurst_exponent}")
            # if hurst_exponent > 0.45:
            #     return False
            
            return True
        except Exception as e:
            logger.error(f"Error filtering pair: {e}")
            return False

    def _save_pairs_info(self, all_pairs):
        """
        Save precomputed hedge ratios and z-scores directly to a pickle file.
        """
        precomputed_data = []
        for pair in tqdm(all_pairs, desc="Saving and Precomputing"):
            window_start = pair['window_start'].strftime('%Y-%m-%d %H:%M:%S')
            window_end = pair['window_end'].strftime('%Y-%m-%d %H:%M:%S')
            pair_tuple = pair['pair']

            # Compute hedge ratio and z-score here
            try:
                data1 = pair['data1']['Close']
                data2 = pair['data2']['Close']

                split_idx = int(self.config['WINDOW_RATIO'] * len(data1))
                train_data1 = data1[:split_idx]
                train_data2 = data2[:split_idx]
                corr_coef = np.corrcoef(train_data1, train_data2)[0, 1]
                hedge_ratio = np.polyfit(np.log(train_data2), np.log(train_data1), 1)[0]
                spread = np.log(data1) - hedge_ratio * np.log(data2)
                train_spread = spread[:split_idx]
                train_mean = train_spread.mean()
                train_std = train_spread.std()
                z_spread = (spread - train_mean) / train_std
                half_life = calculate_half_life(train_spread)

                precomputed_data.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'pair': pair_tuple,
                    'hedge_ratio': hedge_ratio,
                    'spread': spread,  # Added spread
                    'z_spread': z_spread,
                    'split_index': split_idx,  # Added split_index
                    'high1': pair['data1']['High'],
                    'high2': pair['data2']['High'],
                    'low1': pair['data1']['Low'],
                    'low2': pair['data2']['Low'],
                    'vol1': pair['data1']['Volume'],
                    'vol2': pair['data2']['Volume'],
                    'close1': data1,
                    'close2': data2,
                    'corr_coef': corr_coef,
                    'half_life': half_life,
                })
            except Exception as e:
                logger.error(f"Error precomputing pair {pair_tuple}: {e}")
                continue

        # Save all precomputed data to a single pickle file
        with open(self.precomputed_path, 'wb') as f:
            pickle.dump(precomputed_data, f)
        logger.info(f"Saved precomputed data to {self.precomputed_path}")