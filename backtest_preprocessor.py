# backtest_preprocessor.py

import os
import csv
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import numpy as np
import logging
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import shapiro, skew, kurtosis, jarque_bera
import matplotlib.pyplot as plt
import pickle
import statsmodels.api as sm  # Added import for half-life calculation
import ta

# Configure logging (if not already configured in main application)
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

    def prepare_pairs_for_backtest(
        self, 
        window_size: str, 
        overlap: str, 
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
        windows = self._generate_time_windows(coin_data, window_size, overlap)
        logger.info(f"Generated {len(windows)} windows")

        if not windows:
            logger.error("No windows generated. Check window size and overlap.")
            return

        # Step 5 & 6: Generate and filter pairs on the fly
        correlation_threshold = self.config.get('CORR_THRESHOLD', 0.5)
        pvalue_threshold = self.config.get('COINT_THRESHOLD', 0.05)
        window_ratio = self.config.get('WINDOW_RATIO') 

        filtered_pairs = []
        for pair in self._generate_all_pairs(coin_data, windows, correlation_threshold, pvalue_threshold, window_ratio):
            filtered_pairs.append(pair)
        logger.info(f"Filtered down to {len(filtered_pairs)} coin pairs after applying thresholds")

        if not filtered_pairs:
            logger.warning("No pairs left after filtering. Consider lowering thresholds.")
            return

        # Step 7: Save the filtered and precomputed pairs data
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

    def _generate_time_windows(self, coin_data, window_size: str, overlap: str):
        """
        Generates a list of (start, end) tuples representing each window, 
        slicing from end to start.
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

    def _generate_all_pairs(self, coin_data, windows, corr_threshold, pvalue_threshold, window_ratio):
        """
        Generates all possible unique coin pairs within each window, 
        considering only coins with complete data in the window.
        """
        all_pairs = []
        coins = list(coin_data.keys())

        for window_start, window_end in tqdm(windows, desc="Generating pairs and filtering"):
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

                # Additional check to ensure both DataFrames have identical indexes
                if not df1.index.equals(df2.index):
                    logger.warning(f"Time indexes do not match for pair {pair} in window {window_start} to {window_end}")
                    continue

                pair_data = {
                    'window_start': window_start,
                    'window_end': window_end,
                    'pair': pair,
                    'data1': df1[['Close', 'Volume']],
                    'data2': df2[['Close', 'Volume']]
                }
                # Apply filtering here
                if self._filter_pair(pair_data, corr_threshold, pvalue_threshold, window_ratio):
                    all_pairs.append(pair_data)

        return all_pairs

    def _filter_pair(self, pair, corr_threshold, pvalue_threshold, window_ratio):
        """
        Filters a single pair based on hedge ratio and half-life criteria.
        """
        try:
            split_idx = int(window_ratio * len(pair['data1']['Close']))
            train_data1 = pair['data1']['Close'][:split_idx]
            train_data2 = pair['data2']['Close'][:split_idx]

            if (train_data1 <= 0).any() or (train_data2 <= 0).any():
                logger.warning(f"Non-positive values in pair {pair['pair']}.")
                return False

            log_train_data1 = np.log(train_data1)
            log_train_data2 = np.log(train_data2)
            
            # Check for correlation
            corr = np.corrcoef(log_train_data1, log_train_data2)[0, 1]
            if np.abs(corr) < 0.5:
                return False
            
            hedge_ratio = np.polyfit(log_train_data2, log_train_data1, 1)[0]
            hedge_const = 3.0
            if not (1 / hedge_const) < hedge_ratio < hedge_const:
                return False

            spread = log_train_data1 - hedge_ratio * log_train_data2
            spread_std = spread.std()
            spread_mean = spread.mean()
            if spread_std < 0.01:
                return False

            z_spread = (spread - spread_mean) / spread_std 
            if np.abs(z_spread.iloc[-1]) < 2:
                return False
            
            # jb_pvalue = jarque_bera(spread)[1] # financial data is often not normally distributed
            # if jb_pvalue < 0.05:
            #     return False            

            adf_pvalue = adfuller(spread)[1]
            if adf_pvalue > pvalue_threshold:
                return False

            return True
        except Exception as e:
            logger.warning(f"Filtering failed for pair {pair['pair']}: {e}")
            return False

    def _calculate_half_life(self, spread):
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
            return None
        half_life = -np.log(2) / lambda_param
        return half_life

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

                hedge_ratio = np.polyfit(np.log(train_data2), np.log(train_data1), 1)[0]
                spread = np.log(data1) - hedge_ratio * np.log(data2)
                train_spread = spread[:split_idx]
                train_mean = train_spread.mean()
                train_std = train_spread.std()
                z_spread = (spread - train_mean) / train_std

                # Calculate RSI for data1 and data2
                rsi_1 = ta.momentum.RSIIndicator(data1).rsi()
                rsi_2 = ta.momentum.RSIIndicator(data2).rsi()

                precomputed_data.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'pair': pair_tuple,
                    'hedge_ratio': hedge_ratio,
                    'spread': spread,  # Added spread
                    'z_spread': z_spread,
                    'split_index': split_idx,  # Added split_index
                    'vol1': pair['data1']['Volume'],
                    'vol2': pair['data2']['Volume'],
                    'rsi1': rsi_1,
                    'rsi2': rsi_2
                })
            except Exception as e:
                logger.error(f"Error precomputing pair {pair_tuple}: {e}")
                continue

        # Save all precomputed data to a single pickle file
        with open(self.precomputed_path, 'wb') as f:
            pickle.dump(precomputed_data, f)
        logger.info(f"Saved precomputed data to {self.precomputed_path}")