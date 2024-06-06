import os
import csv
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import tqdm
import scipy.stats as stats
import statsmodels.api as sm
from pykalman import KalmanFilter
import logging

plt.style.use('ggplot')

class CryptoPairSelector:
    def __init__(self, client, config):
        self.client = client
        self.directory = config['SAVING_DIR']
        self.interval = config['INTERVAL']  # Assuming you want to use the interval from the config
        self.n_coins = config['N_COINS']
        self.max_n_npd = config['MAX_N_NPD']
        self.train_ratio = config['TRAIN_RATIO']
        self.csv_filename = config['CSV_FILENAME']
        self.coint_threshold = config['COINT_THRESHOLD']
        self.trading_charge = config['TRADING_CHARGE']
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',  # Round to seconds
                            handlers=[logging.StreamHandler(),  # This will print to console
                            logging.FileHandler('crypto_pair_selector.log')])  # This will write to a log file
        
    def load_selected_trading_coins(self):
        logging.info(f"Loading trading coins from {self.csv_filename}...")
        with open(os.path.join(self.directory, self.csv_filename), 'r') as file:
            reader = csv.reader(file)
            coins = [coin for row in reader for coin in row[:self.n_coins]]
            assert len(coins) >= self.n_coins, "N_COINS is greater than the number of trading coins"
            return coins

    def find_valid_coins(self, data):
        valid_coins = []
        for coin, df in data.items():
            coin_diff = np.log(df['Close']).diff().dropna()
            coin_diff = coin_diff[coin_diff != 0]
            if len(coin_diff) == 0:
                logging.info(f'No price difference for {coin}')
                continue
            else:
                try:
                    adf_test = adfuller(coin_diff)
                    if adf_test[1] < 0.01:
                        valid_coins.append(coin)
                    else:
                        logging.info(f'{coin} is not integrated of order 1')
                except ValueError:
                    logging.info(f'Sample size too short for {coin}')
                    continue 

        logging.info(f'Number of coins integrated of order 1: {len(valid_coins)}')

        return valid_coins

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
            split_index = int(len(prices1) * self.train_ratio)
            npd = np.sum((prices1[:split_index] - prices2[:split_index]) ** 2)
            pairs_npd.append((coin1, coin2, npd))
        return pairs_npd

    def find_cointegrated_pairs(self, selected_pairs, data, split):

        coint_pairs = []
        for coin1, coin2 in tqdm.tqdm(selected_pairs):
            common_indices = data[coin1].index.intersection(data[coin2].index)
            prices1 = np.log(data[coin1].loc[common_indices]['Close'])
            prices2 = np.log(data[coin2].loc[common_indices]['Close'])
            split_index = int(len(prices1) * self.train_ratio)
            try:
                if split:
                    result_train = coint(prices1[:split_index], prices2[:split_index])
                    if result_train[1] < self.coint_threshold:
                        coint_pairs.append((coin1, coin2))
                else:
                    result = coint(prices1, prices2)
                    if result[1] < self.coint_threshold:  
                        coint_pairs.append((coin1, coin2))

            except Exception as e:
                logging.error(f"Error in cointegration test for pair: {(coin1, coin2)}: {e}")
            
        return coint_pairs

    def select_trading_pairs(self, split = False):

        logging.info(">>> Loading trading coins...")
        trading_pairs_usdt = self.load_selected_trading_coins()
        data = {pair: pd.read_pickle(os.path.join(self.directory, f'{pair}.pkl')) for pair in trading_pairs_usdt}

        logging.info(">>> Finding valid coins...")
        valid_coins = self.find_valid_coins(data)
        trading_pairs_usdt = [coin for coin in trading_pairs_usdt if coin in valid_coins]
        data = {coin: df for coin, df in data.items() if coin in valid_coins}
        
        logging.info(">>> Calculating normalized prices...")
        all_pairs = list(itertools.combinations(trading_pairs_usdt, 2))
        normalized_prices = self.calculate_normalized_close_prices(data)

        logging.info(">>> Calculating NPD with train ratio...")
        npd_pairs = self.calculate_npd(normalized_prices, all_pairs)
        npd_pairs.sort(key=lambda x: x[2])

        if self.max_n_npd is None:
            selected_npd = [pair[:2] for pair in npd_pairs]
        else:
            selected_npd = [pair[:2] for pair in npd_pairs[:self.max_n_npd]]

        logging.info(">>> Finding cointegrated pairs...")
        coint_pairs = self.find_cointegrated_pairs(selected_npd, data, split)

        logging.info(f"Length of selected NPD pairs for cointegration testing: {len(selected_npd)} / {len(all_pairs)}")
        logging.info(f"Length of cointegrated pairs: {len(coint_pairs)} / {len(selected_npd)}")
        logging.info(f"Cointegrated pairs: {coint_pairs}")

        return coint_pairs, data

    def generate_signal(self, z_spread, threshold_long, threshold_short):
        # Initialize signal with 0
        signal = pd.Series(0, index=z_spread.index)

        # initial signal
        if z_spread.iloc[0] > threshold_long:
            signal.iloc[0] = -1
        elif z_spread.iloc[0] < threshold_short:
            signal.iloc[0] = 1
        
        # loop
        for t in range(1, len(z_spread)):
            if signal.iloc[t-1] == 0:  # if we were in no position
                if z_spread.iloc[t] <= threshold_long:
                    signal.iloc[t] = 1
                elif z_spread.iloc[t] >= threshold_short:
                    signal.iloc[t] = -1
                else:
                    signal.iloc[t] = 0
            elif signal.iloc[t-1] == 1:  # if we were in a long position
                if z_spread.iloc[t] >= 0:
                    signal.iloc[t] = 0
                else:
                    signal.iloc[t] = signal.iloc[t-1]
            else:  # if we were in a short position
                if z_spread.iloc[t] <= 0:
                    signal.iloc[t] = 0
                else:
                    signal.iloc[t] = signal.iloc[t-1]
        
        return signal 

    def get_Kalman_params(self, log_price_1, log_price_2, init_hedge_ratio, init_intercept):
        

        X = log_price_2.values.reshape(-1, 1)
        y = log_price_1.values.reshape(-1, 1)
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_cov = 10
        obs_mat = sm.add_constant(X, prepend=False)[:, np.newaxis]

        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                        initial_state_mean=np.array([init_hedge_ratio, init_intercept]),
                        initial_state_covariance=np.array([[1, 0], 
                                                            [0, 1]]),
                        transition_matrices=np.eye(2),
                        observation_matrices=obs_mat,
                        observation_covariance=obs_cov,
                        transition_covariance=trans_cov)
        
        state_means, _ = kf.filter(y)
        hedge_ratio = state_means[:, 0]
        intercept = state_means[:, 1]

        return hedge_ratio, intercept

    def calculate_optimal_thresholds(self, mean_spread, std_spread, T):
        f = lambda s0: -(s0-(self.trading_charge*4))*T*(1-stats.norm.cdf(s0, mean_spread, std_spread))
        x = np.linspace(0, 0.05, 100)
        f_values = [f(s0) for s0 in x]
        s0 = x[np.argmin(f_values)]
        lower_bound = -(s0-mean_spread)/std_spread
        upper_bound = (s0-mean_spread)/std_spread
        return lower_bound, upper_bound, s0

    def calculate_sharpe_ratio(self, returns):
        mean_returns = returns.mean()
        std_returns = returns.std()
        sharpes = mean_returns / std_returns
        return sharpes
    
    def backtest_pair(self, spread, signal):
        spread_return = spread.diff()
        traded_return = signal * spread_return.shift(-1)
        traded_return.fillna(0, inplace=True)
        cumulated_return = (traded_return.cumsum()+1)*100
        return cumulated_return, traded_return
    
    def observe_all_pairs(self, data, cointegrated_pairs, ENABLE_PLOT):
        for coin1, coin2 in cointegrated_pairs:
            common_indices = data[coin1].index.intersection(data[coin2].index)
            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
            
            # Perform linear regression to find the hedge ratio
            split_index = int(len(log_price_1) * self.train_ratio)
            init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
            hedge_ratio, intercept = self.get_Kalman_params(log_price_1, log_price_2, init_hedge_ratio, init_intercept)
            
            spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)
            kalman_spread = log_price_1 - (hedge_ratio * log_price_2 + intercept)
            # Z-score the spread
            train_spread = spread[:split_index]
            mean_spread = train_spread.mean()
            kalman_mean_spread = kalman_spread.mean()
            std_spread = train_spread.std()
            kalman_std_spread = kalman_spread.std()
            z_spread = (spread - mean_spread) / std_spread
            kalman_z_spread = (kalman_spread - kalman_mean_spread) / kalman_std_spread

            # Determine the optimal thresholds
            T = len(spread)
            lower_bound, upper_bound, s0 = self.calculate_optimal_thresholds(mean_spread, std_spread, T)
            kalman_lower_bound, kalman_upper_bound, kalman_s0 = self.calculate_optimal_thresholds(kalman_mean_spread, kalman_std_spread, T)

            # Generate signal
            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            kalman_signal = self.generate_signal(kalman_z_spread, kalman_lower_bound, kalman_upper_bound)
            
            # Backtest the pair
            cumulated_return, traded_return = self.backtest_pair(spread, signal)
            kalman_cumulated_return, kalman_traded_return = self.backtest_pair(kalman_spread, kalman_signal)

            # Calculate the Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(traded_return)                                
            kalman_shape_ratio = self.calculate_sharpe_ratio(kalman_traded_return)

            # check if z spread at the beginnging split is outside the bounds. if not skip
            print(z_spread.iloc[split_index], upper_bound, lower_bound, sharpe_ratio)
            if z_spread.iloc[split_index] < upper_bound and z_spread.iloc[split_index] > lower_bound:
                continue

            if ENABLE_PLOT:
                # self.plot_pair(coin1, coin2, spread, z_spread, signal, cumulated_return)
                split_index = int(len(spread) * self.train_ratio)

                plt.figure(figsize=(20, 10))
                plt.plot(z_spread)
                plt.plot(kalman_z_spread, 'g')
                plt.axvline(z_spread.index[split_index], color='r', linestyle='--')
                plt.plot(z_spread.index, signal)
                plt.axhline(upper_bound, color='r', linestyle='--')
                plt.axhline(lower_bound, color='r', linestyle='--')
                plt.axhline(0.0)
                plt.ylabel('Z-score')
                plt.grid(True)
                plt.show()

    def select_best_pair(self, data, cointegrated_pairs, ENABLE_PLOT):
        best_pair = None
        best_sharpe_ratio = -np.inf

        info_pair = {}
        for coin1, coin2 in cointegrated_pairs:
            common_indices = data[coin1].index.intersection(data[coin2].index)
            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
            
            # Perform linear regression to find the hedge ratio
            init_hedge_ratio, init_intercept = np.polyfit(log_price_2, log_price_1, 1)
            # hedge_ratio, intercept = self.get_Kalman_params(log_price_1, log_price_2, init_hedge_ratio, init_intercept)
            hedge_ratio = init_hedge_ratio
            intercept = init_intercept
            spread = log_price_1 - (hedge_ratio * log_price_2 + intercept)

            # Z-score the spread
            mean_spread = spread.mean()
            std_spread = spread.std()
            z_spread = (spread - mean_spread) / std_spread

            # Determine the optimal thresholds
            T = len(spread)
            lower_bound, upper_bound, _ = self.calculate_optimal_thresholds(mean_spread, std_spread, T)

            # Generate signal
            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            
            # Backtest the pair
            cumulated_return, traded_return = self.backtest_pair(spread, signal)

            # Calculate the Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(traded_return)                                

            if sharpe_ratio > best_sharpe_ratio and (z_spread.iloc[-1] > upper_bound or z_spread.iloc[-1] < lower_bound) and hedge_ratio > 0: # Only trade if the current spread is outside the bounds
                best_sharpe_ratio = sharpe_ratio
                best_pair = (coin1, coin2)
                info_pair = {
                    'cumulated_return': cumulated_return,
                    'traded_return': traded_return,
                    'spread': spread,
                    'signal': signal,
                    # 'hedge_ratio': hedge_ratio[-1],
                    # 'intercept': intercept[-1],
                    'hedge_ratio': hedge_ratio,
                    'intercept': intercept,
                    'mean_spread': mean_spread,
                    'std_spread': std_spread,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'z_spread': z_spread,
                    'instant_return': np.abs(spread.iloc[-1]) - (2*self.trading_charge),
                    'price1': data[coin1].loc[common_indices]['Close'].iloc[-1],
                    'price2': data[coin2].loc[common_indices]['Close'].iloc[-1],
                    'trade_coin1': 'SELL' if signal.iloc[-1] == 1 else 'BUY', # Swap BUY and SELL
                    'trade_coin2': 'BUY' if signal.iloc[-1] == 1 else 'SELL' # Swap BUY and SELL
                }
        logging.info(f"Best pair: {best_pair} with Sharpe Ratio: {best_sharpe_ratio:.4f}")
        try:
            if ENABLE_PLOT:
                self.plot_best_pair(best_pair, info_pair)
        except Exception as e:
            logging.error(f"Error plotting best pair: {e}")

        return best_pair, best_sharpe_ratio, info_pair            

    def plot_z_score(self, info_pair, best_pair):
        plt.subplot(3, 1, 1)
        plt.plot(info_pair['z_spread'])
        plt.plot(info_pair['z_spread'].index, info_pair['signal'])
        plt.axhline(info_pair['upper_bound'], color='r', linestyle='--')
        plt.axhline(info_pair['lower_bound'], color='r', linestyle='--')
        plt.axhline(0.0)
        plt.ylabel('Z-score')
        plt.title(f'{best_pair[0]} and {best_pair[1]} Z-score')
        plt.grid(True)

    def plot_spread(self, info_pair, best_pair):
        plt.subplot(3, 1, 2)
        plt.plot(info_pair['spread'])
        plt.axhline((info_pair['upper_bound']*info_pair['std_spread'])+info_pair['mean_spread'], color='r', linestyle='--')
        plt.axhline((info_pair['lower_bound']*info_pair['std_spread'])+info_pair['mean_spread'], color='r', linestyle='--')
        plt.axhline(0.0)
        plt.ylabel('Spread')
        plt.title(f'{best_pair[0]} and {best_pair[1]} Spread (stdev: {info_pair["std_spread"]:.4f})')
        plt.grid(True)

    def plot_cumulated_return(self, info_pair, best_pair):
        plt.subplot(3, 1, 3)
        plt.plot(info_pair['cumulated_return'])
        plt.ylabel('Cumulated Return (%)')
        plt.title(f'{best_pair[0]} and {best_pair[1]} Cumulated Return')
        plt.grid(True)

    def plot_best_pair(self, best_pair, info_pair):
        """Plot the results for the best pair."""
        plt.figure(figsize=(20, 10))
        self.plot_z_score(info_pair, best_pair)
        self.plot_spread(info_pair, best_pair)
        self.plot_cumulated_return(info_pair, best_pair)
        plt.tight_layout()
        plt.show()
