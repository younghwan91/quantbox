import os
import csv
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import tqdm
import scipy.stats as stats
import logging
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro
import ta
from joblib import load
from statsmodels.api import OLS, add_constant
from statsmodels.robust.robust_linear_model import RLM

# dark background
plt.style.use('dark_background')

class CoinDataLoader:
    def __init__(self, directory, csv_filename, n_coins):
        self.directory = directory
        self.csv_filename = csv_filename
        self.n_coins = n_coins
    
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
    def __init__(self, train_ratio, coint_threshold, corr_threshold, normality_threshold=0.9):
        if train_ratio < 1:
            self.split = True
        else:
            self.split = False

        self.train_ratio = train_ratio
        self.coint_threshold = coint_threshold
        self.corr_threshold = corr_threshold
        self.normality_threshold = normality_threshold

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
                    if shapiro_result.statistic > self.normality_threshold:  
                        
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
    def __init__(self, analysis):
        self.analysis = analysis

    def select_trading_pairs(self, data):
        logging.info(">>> Finding valid coins...")
        data = self.analysis.find_valid_coins(data)
        
        logging.info(">>> Calculating normalized prices...")
        coins = list(data.keys())
        all_pairs = list(itertools.combinations(coins, 2))
        normalized_prices = self.analysis.calculate_normalized_close_prices(data)

        logging.info(">>> Calculating correlation with train ratio...")  # 1차적인 필터링
        chosen_pairs = self.analysis.calculate_correlation(normalized_prices, all_pairs)
        print(f"Number of pairs: {len(chosen_pairs)}")

        logging.info(">>> Coinegration and Normality test...")  # 2차적인 필터링
        coint_pairs = self.analysis.identify_pairs_candidates(chosen_pairs, data)

        logging.info(f"Number of all pairs before filtering: {len(all_pairs)}")
        logging.info(f"Number of chosen pairs - correlation: {len(chosen_pairs)}")
        logging.info(f"Number of cointegrated pairs: {len(coint_pairs)} / {len(chosen_pairs)}")

        return coint_pairs, data
    
    def save_pairs_and_data(self, cointegrated_pairs, data):
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
                logging.info(f"Data for one of the coins in the pair {pair} is missing.")
                
        return pair_data

class Backtester:
    def __init__(self, trading_charge, train_ratio):
        self.trading_charge = trading_charge
        self.train_ratio = train_ratio
        self.stop_loss_ratio = 2

    def generate_signal(self, z_spread, threshold_long, threshold_short):
        signal = pd.Series(0, index=z_spread.index)

        if z_spread.iloc[0] > threshold_long:
            signal.iloc[0] = -1
        elif z_spread.iloc[0] < threshold_short:
            signal.iloc[0] = 1
        
        for t in range(1, len(z_spread)):
            if signal.iloc[t - 1] == 0:
                if z_spread.iloc[t] <= threshold_long:
                    signal.iloc[t] = 1
                elif z_spread.iloc[t] >= threshold_short:
                    signal.iloc[t] = -1
                else:
                    signal.iloc[t] = 0
            elif signal.iloc[t - 1] == 1:
                if z_spread.iloc[t] >= 0:
                    signal.iloc[t] = 0
                else:
                    signal.iloc[t] = signal.iloc[t - 1]
            else:
                if z_spread.iloc[t] <= 0:
                    signal.iloc[t] = 0
                else:
                    signal.iloc[t] = signal.iloc[t - 1]
        
        return signal 

    def calculate_optimal_thresholds(self, mean_spread, std_spread, T):
        f = lambda s0: -(s0 - (self.trading_charge * 4)) * T * (1 - stats.norm.cdf(s0, mean_spread, std_spread))
        x = np.linspace(0, 0.05, 100)
        f_values = [f(s0) for s0 in x]
        s0 = x[np.argmin(f_values)]
        lower_bound = -(s0 - mean_spread) / std_spread
        upper_bound = (s0 - mean_spread) / std_spread
        return lower_bound, upper_bound, s0

    def _calculate_sharpe_ratio(self, returns):
        mean_returns = returns.mean()
        std_returns = returns.std()
        sharpes = mean_returns / std_returns
        return sharpes
    
    def _calculate_half_life_ols(self, spread: pd.Series) -> float:
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
        half_life = np.log(2) / theta
        
        return half_life

    def _calculate_hurst_exponent(self, ts):
        lags = range(2, 100)
        tau = [np.std(np.subtract(ts[lag:].to_numpy(), ts[:-lag].to_numpy())) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def backtest_pairs(self, data, cointegrated_pairs, ENABLE_PLOT):

        data_samples = []
        for coin1, coin2 in cointegrated_pairs:
            common_indices = data[coin1].index.intersection(data[coin2].index)
            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
            
            split_index = int(len(log_price_1) * self.train_ratio)
            init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
            spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)

            hurst = self._calculate_hurst_exponent(spread[:split_index])
            print(f"Hurst Exponent: {hurst}")
            half_life_methods = self.calculate_half_life_methods(spread[:split_index])
            print(f"Half-life methods: {half_life_methods}")
            correlation = np.corrcoef(log_price_1[:split_index], log_price_2[:split_index])[0, 1]

            train_spread = spread[:split_index]
            mean_spread = train_spread.mean()
            std_spread = train_spread.std()
            z_spread = (spread - mean_spread) / std_spread
  
            T = len(spread)
            lower_bound, upper_bound, s0 = self.calculate_optimal_thresholds(mean_spread, std_spread, T)
            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            print(f"Z-spread at split index: {z_spread.iloc[split_index]}")
            print(f"Upper bound: {upper_bound}, Lower bound: {lower_bound}")

            data_sample = self.collect_training_data(data, coin1, coin2, z_spread, signal, std_spread, half_life_methods, hurst, upper_bound, correlation, split_index)
            condition1 = z_spread.iloc[split_index] > upper_bound
            condition2 = z_spread.iloc[split_index] < lower_bound

            if (condition1 or condition2) and (upper_bound * std_spread + mean_spread > 0.005):
                data_samples.append(data_sample)

                if ENABLE_PLOT:
                    split_index = int(len(spread) * self.train_ratio)

                    plt.figure(figsize=(20, 10))
                    plt.subplot(2, 1, 1)
                    plt.plot(z_spread)
                    plt.axvline(z_spread.index[split_index], color='r', linestyle='--')
                    plt.plot(z_spread.index, signal)
                    plt.axhline(upper_bound, color='r', linestyle='--')
                    plt.axhline(lower_bound, color='r', linestyle='--')
                    plt.axhline((np.abs(z_spread.iloc[split_index]) * self.stop_loss_ratio), color='g', linestyle='--')
                    plt.axhline(-(np.abs(z_spread.iloc[split_index]) * self.stop_loss_ratio), color='g', linestyle='--')
                    plt.axhline(z_spread.iloc[split_index], color = 'y', linestyle='--')
                    plt.axhline(0.0)
                    plt.ylim(-4, 4)
                    plt.ylabel('Z-score')
                    plt.grid(True)

                    plt.subplot(2, 1, 2)
                    plt.plot(z_spread.index, spread)
                    plt.axhline((upper_bound * std_spread) + mean_spread, color='r', linestyle='--')
                    plt.axhline((lower_bound * std_spread) + mean_spread, color='r', linestyle='--')
                    plt.axhline(0.0)
                    plt.axhline(np.abs(spread.iloc[split_index])*self.stop_loss_ratio, color='g', linestyle='--')
                    plt.axhline(-np.abs(spread.iloc[split_index])*self.stop_loss_ratio, color='g', linestyle='--')
                    plt.axhline(spread.iloc[split_index], color='y', linestyle='--')
                    plt.grid(True)

                    plt.show()

        return data_samples
    
    def optimize_filter_params(self, data):
        coin1, coin2 = data.keys()

        common_indices = data[coin1].index.intersection(data[coin2].index)
        log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
        log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
        
        split_index = int(len(log_price_1) * self.train_ratio)
        init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
        spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)

        train_spread = spread[:split_index]
        mean_spread = train_spread.mean()
        std_spread = train_spread.std()
        z_spread = (spread - mean_spread) / std_spread

        T = len(spread)
        lower_bound, upper_bound, s0 = self.calculate_optimal_thresholds(mean_spread, std_spread, T)

        condition1 = z_spread.iloc[split_index] > upper_bound
        condition2 = z_spread.iloc[split_index] < lower_bound

        if (condition1 or condition2) and np.abs(spread.iloc[split_index]) > 0.005:

            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            
            z_spread_i = z_spread.iloc[split_index]
            signal_i = signal.iloc[split_index]

            observe_z_spread = z_spread.iloc[split_index:]

            open_position = 0 
            stop_loss = z_spread_i*self.stop_loss_ratio

            for i in range(1, len(observe_z_spread)): # check if it generates 2:1 win-loss ratio 
                if signal_i == 1 and observe_z_spread.iloc[i] < stop_loss:
                    open_position = 0
                    break
                elif signal_i == -1 and observe_z_spread.iloc[i] > stop_loss:
                    open_position = 0
                    break
                elif signal_i == 1 and observe_z_spread.iloc[i] > 0:
                    open_position = 1
                    break
                elif signal_i == -1 and observe_z_spread.iloc[i] < 0:
                    open_position = 1
                    break   
            features.append(open_position)
        else:
            features = None

        return features
        
    def select_pair(self, data, cointegrated_pairs, ENABLE_PLOT):

        best_pair = None

        pair_info = {}
        for coin1, coin2 in cointegrated_pairs:
            common_indices = data[coin1].index.intersection(data[coin2].index)
            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
            
            init_hedge_ratio, init_intercept = np.polyfit(log_price_2, log_price_1, 1)
            hedge_ratio = pd.Series(init_hedge_ratio, index=log_price_1.index)
            intercept = pd.Series(init_intercept, index=log_price_1.index)

            spread = log_price_1 - (hedge_ratio * log_price_2 + intercept)

            mean_spread = spread.mean()
            std_spread = spread.std()
            z_spread = (spread - mean_spread) / std_spread

            T = len(spread)
            lower_bound, upper_bound, _ = self.calculate_optimal_thresholds(mean_spread, std_spread, T)
            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            cumulated_return, traded_return = self.backtest_pair(spread, signal)
            half_life = self._calculate_half_life(spread)
            hurst = self._calculate_hurst_exponent(spread)


            condition_3 = z_spread.iloc[-1] > upper_bound or z_spread.iloc[-1] < lower_bound
            condition_4 = upper_bound * std_spread + mean_spread > 0.005

            if condition_3 and condition_4:
                best_pair = (coin1, coin2)
                pair_info = {
                    'cumulated_return': cumulated_return,
                    'traded_return': traded_return,
                    'spread': spread,
                    'signal': signal,
                    'hedge_ratio': hedge_ratio[-1],
                    'intercept': intercept[-1],
                    'mean_spread': mean_spread,
                    'std_spread': std_spread,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'z_spread': z_spread,
                    'half_life': half_life,
                    'instant_return': np.abs(spread.iloc[-1]) - (2 * self.trading_charge),
                    'price1': data[coin1].loc[common_indices]['Close'].iloc[-1],
                    'price2': data[coin2].loc[common_indices]['Close'].iloc[-1],
                    'trade_coin1': 'BUY' if signal.iloc[-1] == 1 else 'SELL',
                    'trade_coin2': 'SELL' if signal.iloc[-1] == 1 else 'BUY'
                }
        try:
            if ENABLE_PLOT:
                self.plot_best_pair(best_pair, pair_info)
        except Exception as e:
            logging.error(f"Error plotting best pair: {e}")

        return best_pair, pair_info

    def plot_best_pair(self, best_pair, info_pair):
        plt.figure(figsize=(20, 10))
        self.plot_z_score(info_pair, best_pair)
        self.plot_spread(info_pair, best_pair)
        self.plot_cumulated_return(info_pair, best_pair)
        plt.tight_layout()
        plt.show()

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
        plt.axhline((info_pair['upper_bound'] * info_pair['std_spread']) + info_pair['mean_spread'], color='r', linestyle='--')
        plt.axhline((info_pair['lower_bound'] * info_pair['std_spread']) + info_pair['mean_spread'], color='r', linestyle='--')
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

class CryptoPairSelector:
    def __init__(self, client, config):
        self.client = client
        self.loader = CoinDataLoader(config['SAVING_DIR'], config['CSV_FILENAME'], config['N_COINS'])
        self.analysis = CoinAnalysis(config['TRAIN_RATIO'], config['COINT_THRESHOLD'], config['CORR_THRESHOLD'])
        self.selector = PairSelector(self.analysis)
        self.backtester = Backtester(config['TRADING_CHARGE'], config['TRAIN_RATIO'])
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',  # Round to seconds
                            handlers=[logging.StreamHandler(),  # This will print to console
                                      logging.FileHandler('crypto_pair_selector.log')])  # This will write to a log file
    
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

    def save_pairs_and_data(self, cointegrated_pairs, data): # already filtered. we try to optimize the saving pairs
        try:
            return self.selector.save_pairs_and_data(cointegrated_pairs, data)
        except Exception as e:
            logging.error(f"Error in save_pairs_and_data: {e}")
            raise
    
    def optimize_filter_params(self, cointegrated_pairs, data):
        pass

    def backtest_pairs(self, data, cointegrated_pairs, ENABLE_PLOT):
        try:
            return self.backtester.backtest_pairs(data, cointegrated_pairs, ENABLE_PLOT)
        except Exception as e:
            logging.error(f"Error in backtest_pairs: {e}")
            raise

    def select_best_pair(self, data, cointegrated_pairs, ENABLE_PLOT):
        try:
            return self.backtester.select_pair(data, cointegrated_pairs, ENABLE_PLOT)
        except Exception as e:
            logging.error(f"Error in select_best_pair: {e}")
            raise
