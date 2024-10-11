import os
import csv
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import tqdm
import logging
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro
from statsmodels.api import OLS, add_constant
from datetime import datetime
import pytz

logger = logging.getLogger('PairSelector')

# dark background
plt.style.use('dark_background')

class CoinDataLoader:
    def __init__(self, config):
        self.config = config
        self.directory = config['SAVING_DIR']
        self.csv_filename = config['CSV_FILENAME']
        self.n_coins = config['N_COINS']
    
    def load_selected_trading_coins(self):
        logging.info(f"Loading trading coins from {self.csv_filename}...")
        with open(os.path.join(self.directory, self.csv_filename), 'r') as file:
            reader = csv.reader(file)
            coins = [coin for row in reader for coin in row[:self.n_coins]]
            assert len(coins) >= self.n_coins, "N_COINS is greater than the number of trading coins"
            return coins
    
    def load_coin_data(self, trading_pairs_usdt):
        data = {pair: pd.read_pickle(os.path.join(self.directory, f'{pair}.pkl')) for pair in trading_pairs_usdt}
        return data

class CoinAnalysis:
    def __init__(self, config):
        self.config = config
        self.train_ratio = config['TRAIN_RATIO']
        self.coint_threshold = config['COINT_THRESHOLD']
        self.corr_threshold = config['CORR_THRESHOLD']
        self.normality_threshold = config.get('NORMALITY_THRESHOLD', 0.05)
        self.split = self.train_ratio < 1

    def find_valid_coins(self, data):
        items = list(data.items())
        for coin, df in items:
            coin_diff = np.log(df['Close']).diff().dropna()
            coin_diff = coin_diff[coin_diff != 0]
            if len(coin_diff) == 0:
                logging.info(f'No price difference for {coin}')
                continue
            else:
                try:
                    adf_test = adfuller(coin_diff)
                    if adf_test[1] < 0.01:
                        data[coin] = df
                    else:
                        logging.info(f'{coin} is not integrated of order 1')
                        data.pop(coin)
                except ValueError:
                    logging.info(f'Sample size too short for {coin}')
                    data.pop(coin)
                    continue 

        return data

    def calculate_normalized_close_prices(self, data):
        normalized_prices = {}
        for coin, df in data.items():
            prices = df['Close']
            normalized_prices[coin] = prices / prices.iloc[0]
        return normalized_prices

    def calculate_npd(self, normalized_prices, pairs):
        pairs_npd = []
        for coin1, coin2 in pairs:
            common_indices = normalized_prices[coin1].index.intersection(normalized_prices[coin2].index)
            prices1 = normalized_prices[coin1].loc[common_indices]
            prices2 = normalized_prices[coin2].loc[common_indices]

            if self.split:
                split_index = int(len(prices1) * self.train_ratio)
                npd = np.sum((prices1[:split_index] - prices2[:split_index]) ** 2)
            else:
                npd = np.sum((prices1 - prices2) ** 2)
            pairs_npd.append((coin1, coin2, npd))
        return pairs_npd
    
    def calculate_correlation(self, normalized_prices, pairs):
        pairs_corr = []
        for coin1, coin2 in pairs:
            common_indices = normalized_prices[coin1].index.intersection(normalized_prices[coin2].index)
            prices1 = normalized_prices[coin1].loc[common_indices]
            prices2 = normalized_prices[coin2].loc[common_indices]

            if prices1.isnull().any() or prices2.isnull().any():
                continue
            if np.isinf(prices1).any() or np.isinf(prices2).any():
                continue

            if self.split:
                split_index = int(len(prices1) * self.train_ratio)
                corr = np.corrcoef(prices1[:split_index], prices2[:split_index])[0, 1]
            else:
                corr = np.corrcoef(prices1, prices2)[0, 1]

            if corr > self.corr_threshold:
                pairs_corr.append((coin1, coin2))

        return pairs_corr

    def identify_pairs_candidates(self, selected_pairs, data):
        coint_pairs = []
        for coin1, coin2 in tqdm.tqdm(selected_pairs):
            try:
                common_indices = data[coin1].index.intersection(data[coin2].index)
                prices1 = np.log(data[coin1].loc[common_indices]['Close'])
                prices2 = np.log(data[coin2].loc[common_indices]['Close'])
                split_index = int(len(prices1) * self.train_ratio)

                # Filter 1: Hedge ratio
                hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
                hedge_const = 3.0
                if (1 / hedge_const) <= hedge_ratio <= hedge_const:

                    # Filter 2: Normality test
                    spread = prices1 - hedge_ratio * prices2
                    shapiro_result = shapiro(spread)
                    if shapiro_result.pvalue > self.normality_threshold:  
                        
                        # Filter 3: Cointegration test
                        if self.split:
                            result = coint(prices1[:split_index], prices2[:split_index])
                        else:
                            result = coint(prices1, prices2)

                        if result[1] < self.coint_threshold:
                            coint_pairs.append((coin1, coin2))

            except Exception as e:
                logging.error(f"Error in testing pair: {(coin1, coin2)}: {e}")

        return coint_pairs

class PairSelector:
    def __init__(self, config):
        self.config = config
        self.analysis = CoinAnalysis(config)
        self.threshold = config['THRESHOLD']
        self.save_plot_dir = './saved_plots' 
        os.makedirs(self.save_plot_dir, exist_ok=True)
        logging.info(f"Plot save directory set to: {self.save_plot_dir}")

    def select_trading_pairs(self, data):
        logging.info(">>> Starting pair selection process...")
        # Step 1: Find valid coins based on stationarity
        data = self.analysis.find_valid_coins(data)

        # Step 2: Calculate normalized close prices
        normalized_prices = self.analysis.calculate_normalized_close_prices(data)

        # Step 3: Generate all possible pairs
        coins = list(data.keys())
        all_pairs = list(itertools.combinations(coins, 2))

        # Step 4: Calculate correlation to filter pairs
        chosen_pairs = self.analysis.calculate_correlation(normalized_prices, all_pairs)
        logging.info(f"Number of pairs after correlation filtering: {len(chosen_pairs)}")

        # Step 5: Identify cointegrated pairs with normal spread
        coint_pairs = self.analysis.identify_pairs_candidates(chosen_pairs, data)
        logging.info(f"Number of cointegrated pairs: {len(coint_pairs)} / {len(chosen_pairs)}")

        return coint_pairs, data

    def save_pairs_and_data(self, cointegrated_pairs, data):
        logging.info(">>> Saving pairs and their data...")
        pair_data = {}
        for pair in cointegrated_pairs:
            coin1, coin2 = pair
            if coin1 in data and coin2 in data:
                coin1_data = data[coin1]
                coin2_data = data[coin2]
                pair_data[pair] = {
                    # Open, High, Low, Close, Volume of each coin
                    coin1: coin1_data[['Open', 'High', 'Low', 'Close', 'Volume']],
                    coin2: coin2_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                }
            else:
                logging.warning(f"Data for one of the coins in the pair {pair} is missing. Skipping.")
        logging.info(f"Saved data for {len(pair_data)} pairs.")
        return pair_data

    def _calculate_half_life(self, spread: pd.Series) -> float:
        spread_lagged = spread.shift(1)
        delta_spread = spread - spread_lagged

        spread_lagged = spread_lagged.dropna()
        delta_spread = delta_spread.dropna()

        X = spread_lagged.values.reshape(-1, 1)
        y = delta_spread.values.reshape(-1, 1)
        X = add_constant(X)
        model = OLS(y, X).fit()

        beta = model.params[1]
        theta = -np.log(1 + beta)
        half_life = np.log(2) / theta if theta != 0 else np.inf

        return half_life

    def _calculate_hurst_exponent(self, ts):
        lags = range(2, 100)
        tau = [np.std(np.subtract(ts[lag:].to_numpy(), ts[:-lag].to_numpy())) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def select_best_pair(self, data, cointegrated_pairs, ENABLE_PLOT=False):
        logging.info(">>> Selecting the best trading pair...")
        suitable_pairs = []

        for pair in tqdm.tqdm(cointegrated_pairs, desc="Selecting Best Pair"):
            coin1, coin2 = pair
            common_indices = data[coin1].index.intersection(data[coin2].index)

            if len(common_indices) < 50:
                logging.debug(f"Not enough data for pair {pair}. Skipping.")
                continue    

            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])

            try:
                # Compute hedge ratio and intercept
                hedge_ratio, init_intercept = np.polyfit(log_price_2, log_price_1, 1)
                hedge_ratio_series = pd.Series(hedge_ratio, index=log_price_1.index)
                intercept_series = pd.Series(init_intercept, index=log_price_1.index)

                # Calculate spread and z-score
                spread = log_price_1 - (hedge_ratio_series * log_price_2 + intercept_series)
                mean_spread = spread.mean()
                std_spread = spread.std()
                z_spread = (spread - mean_spread) / std_spread

                if self.threshold < np.abs(z_spread.iloc[-1]) < 5.0:
                    logging.info(f"Pair: {pair}, Z-Spread: {z_spread.iloc[-1]:.4f} meets criteria.")

                    # Calculate half-life and Hurst exponent
                    half_life = self._calculate_half_life(spread)
                    # hurst = self._calculate_hurst_exponent(spread)
                    p_coint = coint(log_price_1, log_price_2)[1]

                    pair_info = {
                        'hedge_ratio': hedge_ratio,
                        'intercept': init_intercept,
                        'mean_spread': mean_spread,
                        'std_spread': std_spread,
                        'half_life': half_life,
                        'p_coint': p_coint,
                        'z_spread': z_spread.iloc[-1],
                    }

                    suitable_pairs.append((pair, pair_info))

                    # Plotting and Saving the Spread as PNG
                    if ENABLE_PLOT:
                        try:
                            plt.figure(figsize=(14, 7))
                            plt.plot(spread, label='Spread', color='green')
                            plt.axhline(mean_spread, color='red', linestyle='--', label='Mean Spread')
                            plt.title(f"Spread for Pair: {coin1}/{coin2}")
                            plt.legend()
                            
                            # Create a filename with pair name and current timestamp
                            kst = pytz.timezone('Asia/Seoul')
                            timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
                            filename = f"{coin1}_{coin2}_{timestamp}.png"
                            filepath = os.path.join(self.save_plot_dir, filename)
                            
                            plt.savefig(filepath)
                            logging.info(f"Saved spread plot for pair {pair} as {filepath}")
                            plt.close()
                        except Exception as e:
                            logging.error(f"Failed to save plot for pair {pair}: {e}")

            except np.linalg.LinAlgError as e:
                logging.warning(f"LinAlgError in processing pair {pair}: {e}")
            except Exception as e:
                logging.error(f"Error in processing pair {pair}: {e}")
    
        if not suitable_pairs:
            logging.warning("No suitable pairs found based on the given criteria.")
            return []

        # Sort suitable_pairs by half-life in ascending order
        suitable_pairs_sorted = sorted(suitable_pairs, key=lambda x: x[1]['half_life'] * (x[1]['p_coint']))

        for idx, (pair, info) in enumerate(suitable_pairs_sorted, start=1):
            logging.info(f"{idx}. Pair: {pair}, Z-Spread: {info['z_spread']:.4f}, Half-Life: {info['half_life']:.4f}, Coint: {info['p_coint']:.4f}")    

        return suitable_pairs_sorted

class CryptoPairSelector:
    def __init__(self, config):
        self.config = config
        self.loader = CoinDataLoader(config)
        self.selector = PairSelector(config)

    def select_trading_pairs(self):
        try:
            logging.info(">>> Loading trading coins...")
            trading_pairs_usdt = self.loader.load_selected_trading_coins()
            data = self.loader.load_coin_data(trading_pairs_usdt)
            coint_pairs, data = self.selector.select_trading_pairs(data)
            return coint_pairs, data
        except Exception as e:
            logging.error(f"Error in select_trading_pairs: {e}")
            raise

    def save_pairs_and_data(self, cointegrated_pairs, data):
        try:
            return self.selector.save_pairs_and_data(cointegrated_pairs, data)
        except Exception as e:
            logging.error(f"Error in save_pairs_and_data: {e}")
            raise

    def select_best_pair(self, data, cointegrated_pairs, ENABLE_PLOT=False):
        try:
            return self.selector.select_best_pair(data, cointegrated_pairs, ENABLE_PLOT)
        except Exception as e:
            logging.error(f"Error in select_best_pair: {e}")
            raise
