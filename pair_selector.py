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
import logging
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro
import ta
from joblib import load
#dark background
plt.style.use('dark_background')

class CryptoPairSelector:
    def __init__(self, client, config):
        self.client = client
        self.directory = config['SAVING_DIR']
        self.interval = config['INTERVAL']  # Assuming you want to use the interval from the config
        self.n_coins = config['N_COINS']
        self.train_ratio = config['TRAIN_RATIO']
        self.csv_filename = config['CSV_FILENAME']
        self.coint_threshold = config['COINT_THRESHOLD']
        self.trading_charge = config['TRADING_CHARGE']
        self.corr_threshold = config['CORR_THRESHOLD']
        self.normality_threshold = 0.9
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
                        features = self.calculate_features(data = df)
                        data[coin] = pd.concat([df, features], axis=1)
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

    def calculate_npd(self, normalized_prices, pairs, split = False):

        pairs_npd = []
        for coin1, coin2 in pairs:
            common_indices = normalized_prices[coin1].index.intersection(normalized_prices[coin2].index)
            prices1 = normalized_prices[coin1].loc[common_indices]
            prices2 = normalized_prices[coin2].loc[common_indices]
            if split:
                split_index = int(len(prices1) * self.train_ratio)
                npd = np.sum((prices1[:split_index] - prices2[:split_index]) ** 2)
            else:
                npd = np.sum((prices1 - prices2) ** 2)
            pairs_npd.append((coin1, coin2, npd))

        return pairs_npd
    
    def calculate_correlation(self, normalized_prices, pairs, split = False):

        pairs_corr = []
        for coin1, coin2 in pairs:
            common_indices = normalized_prices[coin1].index.intersection(normalized_prices[coin2].index)
            prices1 = normalized_prices[coin1].loc[common_indices]
            prices2 = normalized_prices[coin2].loc[common_indices]
            if split:
                split_index = int(len(prices1) * self.train_ratio)
                corr = np.corrcoef(prices1[:split_index], prices2[:split_index])[0, 1]
            else:
                corr = np.corrcoef(prices1, prices2)[0, 1]

            if corr > self.corr_threshold:
                pairs_corr.append((coin1, coin2))

        return pairs_corr

    def identify_pairs_candidates(self, selected_pairs, data, split):
        
        coint_pairs = []
        for coin1, coin2 in tqdm.tqdm(selected_pairs):
            try:
                common_indices = data[coin1].index.intersection(data[coin2].index)
                prices1 = np.log(data[coin1].loc[common_indices]['Close'])
                prices2 = np.log(data[coin2].loc[common_indices]['Close'])
                split_index = int(len(prices1) * self.train_ratio)

                # Calculate hedge ratio
                hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
                hedge_const = 3.0
                if (1/hedge_const) <= hedge_ratio <= hedge_const:
                    # Calculate spread for the normality test
                    spread = prices1 - hedge_ratio * prices2
                    
                    # Perform Shapiro-Wilk test
                    shapiro_result = shapiro(spread)
                    if shapiro_result.statistic > self.normality_threshold:  
                        if split:
                            result = coint(prices1[:split_index], prices2[:split_index])
                        else:
                            result = coint(prices1, prices2)

                        if result[1] < self.coint_threshold:
                            coint_pairs.append((coin1, coin2))

            except Exception as e:
                logging.error(f"Error in testing pair: {(coin1, coin2)}: {e}")

        return coint_pairs

    def _calculate_hurst_exponent(self, ts):
        lags = range(2, 100)
        tau = [np.std(np.subtract(ts[lag:].to_numpy(), ts[:-lag].to_numpy())) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def calculate_features(self, data, window_rsi=14, window_stochastic=14, window_ema=20, atr_windows=[14, 20, 50]):
        features = pd.DataFrame(index=data.index)

        # Calculate RSI (Relative Strength Index) and normalize it to range between 0 and 1
        features['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=window_rsi).rsi() / 100

        # Calculate Stochastic Oscillator (normalized features: %K and %D)
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=window_stochastic)
        features['Stoch_k'] = stoch.stoch() / 100
        features['Stoch_d'] = stoch.stoch_signal() / 100

        # Calculate VWAP (Volume Weighted Average Price)
        vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # Calculate P/V Ratio (Price to VWAP Ratio)
        features['P/V_Ratio'] = data['Close'] / vwap

        # Calculate NPVI (Normalized Price to VWAP Indicator)
        min_price = data['Close'].min()
        max_price = data['Close'].max()
        features['NPVI'] = (data['Close'] - min_price) / (max_price - min_price)

        # Calculate ATR for multiple windows and normalize it
        for window in atr_windows:
            atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=window).average_true_range()
            features[f'Normalized_ATR_{window}'] = atr / data['Close']

        # Normalize volume using a moving average of the volume over a certain window
        volume_ma = data['Volume'].rolling(window=window_ema).mean()
        features['Normalized_Volume'] = data['Volume'] / volume_ma

        return features

    def select_trading_pairs(self, split = False):
        
        logging.info(">>> Loading trading coins...")
        trading_pairs_usdt = self.load_selected_trading_coins()
        data = {pair: pd.read_pickle(os.path.join(self.directory, f'{pair}.pkl')) for pair in trading_pairs_usdt}

        logging.info(">>> Finding valid coins...")
        data = self.find_valid_coins(data)
        
        logging.info(">>> Calculating normalized prices...")
        trading_pairs_usdt = list(data.keys())
        all_pairs = list(itertools.combinations(trading_pairs_usdt, 2))
        normalized_prices = self.calculate_normalized_close_prices(data)

        # logging.info(">>> Calculating NPD with train ratio...")
        # npd_pairs = self.calculate_npd(normalized_prices, all_pairs)
        # npd_pairs.sort(key=lambda x: x[2])

        logging.info(">>> Calculating correlation with train ratio...") # 1차적인 필터링
        chosen_pairs = self.calculate_correlation(normalized_prices, all_pairs)
        print(f"Number of pairs: {len(chosen_pairs)}")

        logging.info(">>> Coinegration and Normality test...") # 2차적인 필터링
        coint_pairs = self.identify_pairs_candidates(chosen_pairs, data, split)

        logging.info(f"Number of all pairs before filtering: {len(all_pairs)}")
        logging.info(f"Number of chosen pairs - correlation: {len(chosen_pairs)}")
        logging.info(f"Number of cointegrated pairs: {len(coint_pairs)} / {len(chosen_pairs)}")
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

    def calculate_optimal_thresholds(self, mean_spread, std_spread, T):
        f = lambda s0: -(s0-(self.trading_charge*4))*T*(1-stats.norm.cdf(s0, mean_spread, std_spread))
        x = np.linspace(0, 0.05, 100)
        f_values = [f(s0) for s0 in x]
        s0 = x[np.argmin(f_values)]
        lower_bound = -(s0-mean_spread)/std_spread
        upper_bound = (s0-mean_spread)/std_spread
        return lower_bound, upper_bound, s0

    def _calculate_sharpe_ratio(self, returns):
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
    
    def _calculate_half_life(self, spread: pd.Series) -> float:

        # Calculate the first difference of the spread
        spread_lagged = spread.shift(1)
        delta_spread = spread - spread_lagged

        # Drop NaN values
        spread_lagged = spread_lagged.dropna()
        delta_spread = delta_spread.dropna()

        # Regress delta_spread on spread_lagged
        X = spread_lagged.values.reshape(-1, 1)
        y = delta_spread.values.reshape(-1, 1)
        X = add_constant(X)
        model = OLS(y, X).fit()
        
        # Get the coefficient for spread_lagged
        beta = model.params[1]
        
        # Calculate theta
        theta = -np.log(1 + beta)
        
        # Calculate half-life
        half_life = np.log(2) / theta
        
        return half_life

    def collect_training_data(self, data, coin1, coin2, z_spread, signal, std_spread, half_life, hurst, split_index):
        '''
        Collects training data for the model.

        Features:
        "Position",
        "Z_Spread",
        "Std_Spread",
        "Coin1_RSI",
        "Coin1_Stoch_k",
        "Coin1_Stoch_d",
        "Coin1_P/V_Ratio",
        "Coin1_NPVI",
        "Coin1_Normalized_ATR_14",
        "Coin1_Normalized_ATR_20",
        "Coin1_Normalized_ATR_50",
        "Coin1_Normalized_Volume",
        "Coin2_RSI",
        "Coin2_Stoch_k",
        "Coin2_Stoch_d",
        "Coin2_P/V_Ratio",
        "Coin2_NPVI",
        "Coin2_Normalized_ATR_14",
        "Coin2_Normalized_ATR_20",
        "Coin2_Normalized_ATR_50",
        "Coin2_Normalized_Volume",
        "half_life",
        "Hurst_Exponent",
        "Choose_Pair"
        '''

        common_indices = data[coin1].index.intersection(data[coin2].index)
        # Input features
        coin1_data = data[coin1].loc[common_indices]
        coin2_data = data[coin2].loc[common_indices]
        train_sample = []
        list_of_features = ['RSI', 'Stoch_k', 'Stoch_d', 'P/V_Ratio', 'NPVI', 'Normalized_ATR_14', 'Normalized_ATR_20', 'Normalized_ATR_50', 'Normalized_Volume']
        position = signal.iloc[split_index]
        z_spread = z_spread.iloc[split_index]
        train_sample.append(position)
        train_sample.append(z_spread)
        train_sample.append(std_spread)
        train_sample.extend(coin1_data.iloc[split_index][list_of_features].tolist())
        train_sample.extend(coin2_data.iloc[split_index][list_of_features].tolist())
        train_sample.append(half_life)
        train_sample.append(hurst)

        # Output feature
        observe_signal = signal.iloc[split_index:]
        choose_pair = len(np.unique(observe_signal)) > 1
        train_sample.append(choose_pair)

        return train_sample


    def observe_all_pairs(self, data, cointegrated_pairs, ENABLE_PLOT):
        data_samples = []
        for coin1, coin2 in cointegrated_pairs:

            common_indices = data[coin1].index.intersection(data[coin2].index)
            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
            
            split_index = int(len(log_price_1) * self.train_ratio)
            init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
            spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)

            hurst= self._calculate_hurst_exponent(spread[:split_index])
            print(f"Hurst Exponent: {hurst}")
            half_life = self._calculate_half_life(spread[:split_index])
            print(f"Half-life: {half_life}") 

            train_spread = spread[:split_index]
            mean_spread = train_spread.mean()
            std_spread = train_spread.std()
            z_spread = (spread - mean_spread) / std_spread
  
            T = len(spread)
            lower_bound, upper_bound, s0 = self.calculate_optimal_thresholds(mean_spread, std_spread, T)
            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            cumulated_return, traded_return = self.backtest_pair(spread, signal)
            sharpe_ratio = self._calculate_sharpe_ratio(traded_return)                                
            print(f"Z-spread at split index: {z_spread.iloc[split_index]}")
            print(f"Upper bound: {upper_bound}, Lower bound: {lower_bound}")

            data_sample = self.collect_training_data(data, coin1, coin2, z_spread, signal, std_spread, half_life, hurst, split_index)
            condition1 = z_spread.iloc[split_index] > upper_bound
            condition2 = z_spread.iloc[split_index] < lower_bound
            if condition1 or condition2:
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
                    plt.axhline(0.0)
                    plt.ylim(-4, 4)
                    plt.ylabel('Z-score')
                    plt.grid(True)

                    plt.subplot(2, 1, 2)
                    plt.plot(z_spread.index, spread)
                    plt.grid(True)

                    plt.show()

        return data_samples

    def select_best_pair(self, data, cointegrated_pairs, ENABLE_PLOT):
        # Load the trained model
        model = load('trained_model.joblib')
        scaler = load('scaler.joblib')

        best_pair = None
        highest_probability = -np.inf

        pair_info = {}
        for coin1, coin2 in cointegrated_pairs:
            common_indices = data[coin1].index.intersection(data[coin2].index)
            log_price_1 = np.log(data[coin1].loc[common_indices]['Close'])
            log_price_2 = np.log(data[coin2].loc[common_indices]['Close'])
            
            # Perform linear regression to find the hedge ratio
            init_hedge_ratio, init_intercept = np.polyfit(log_price_2, log_price_1, 1)
            hedge_ratio = pd.Series(init_hedge_ratio, index=log_price_1.index)
            intercept = pd.Series(init_intercept, index=log_price_1.index)

            spread = log_price_1 - (hedge_ratio * log_price_2 + intercept)

            # Z-score the spread
            mean_spread = spread.mean()
            std_spread = spread.std()
            z_spread = (spread - mean_spread) / std_spread

            # Determine the optimal thresholds
            T = len(spread)
            lower_bound, upper_bound, _ = self.calculate_optimal_thresholds(mean_spread, std_spread, T)
            signal = self.generate_signal(z_spread, lower_bound, upper_bound)
            cumulated_return, traded_return = self.backtest_pair(spread, signal)
            half_life = self._calculate_half_life(spread)
            hurst = self._calculate_hurst_exponent(spread)

            data_sample = self.collect_training_data(data, coin1, coin2, z_spread, signal, std_spread, half_life, hurst, split_index=-1)
            if np.isnan(data_sample).any():
                continue
            scaled_features = scaler.transform(data_sample)
            probability = model.predict_proba(scaled_features)[0][1]

            if probability > 0.5 and probability > highest_probability and (z_spread.iloc[-1] > upper_bound or z_spread.iloc[-1] < lower_bound):
                best_pair = (coin1, coin2)
                highest_probability = probability
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
                    'instant_return': np.abs(spread.iloc[-1]) - (2*self.trading_charge),
                    'price1': data[coin1].loc[common_indices]['Close'].iloc[-1],
                    'price2': data[coin2].loc[common_indices]['Close'].iloc[-1],
                    'trade_coin1': 'BUY' if signal.iloc[-1] == 1 else 'SELL',
                    'trade_coin2': 'SELL' if signal.iloc[-1] == 1 else 'BUY' 
                }
        logging.info(f"Best pair: {best_pair} with probability: {highest_probability:.4f}")
        try:
            if ENABLE_PLOT:
                self.plot_best_pair(best_pair, pair_info)
        except Exception as e:
            logging.error(f"Error plotting best pair: {e}")

        return best_pair, highest_probability, pair_info

    def plot_best_pair(self, best_pair, info_pair):
        """Plot the results for the best pair."""
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