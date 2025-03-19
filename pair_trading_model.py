# pair_trading_model.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from scipy.stats import shapiro
from scipy.optimize import differential_evolution
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import ta
from numba import njit
from pykalman import KalmanFilter
import mykalman as mk
import statsmodels.api as sm
from mykalman import KalmanModel
plt.style.use('ggplot')


def rolling_vwap(prices, volumes, window):
    # Ensure inputs are float64 arrays to avoid Numba type issues
    prices = np.array(prices, dtype=np.float64)
    volumes = np.array(volumes, dtype=np.float64)
    
    n = len(prices)
    if n == 0 or window <= 0:
        return np.array([], dtype=np.float64)

    # Prepare output array, filled with NaN
    vwap_values = np.full(n, np.nan, dtype=np.float64)

    # Cumulative sums for volumes and (price * volume)
    cumsum_vol = np.zeros(n, dtype=np.float64)
    cumsum_pv  = np.zeros(n, dtype=np.float64)  # price * volume

    # Initialize first element
    cumsum_vol[0] = volumes[0]
    cumsum_pv[0]  = prices[0] * volumes[0]

    # Compute cumulative sums
    for i in range(1, n):
        cumsum_vol[i] = cumsum_vol[i - 1] + volumes[i]
        cumsum_pv[i]  = cumsum_pv[i - 1] + prices[i] * volumes[i]

    # Compute rolling VWAP
    for i in range(n):
        # We can only compute once we have "window" observations
        if i + 1 >= window:
            if i >= window:
                total_vol = cumsum_vol[i] - cumsum_vol[i - window]
                total_pv  = cumsum_pv[i]  - cumsum_pv[i - window]
            else:
                # For initial window, just use the data up to i
                total_vol = cumsum_vol[i]
                total_pv  = cumsum_pv[i]

            if total_vol > 0:
                vwap_values[i] = total_pv / total_vol

    return vwap_values


@njit
def compute_vw_rsi_nb(close, volume, window):
    """
    Numba njit를 사용해 Volume Weighted RSI (VWRSI)를 계산합니다.
    
    Parameters:
        close (np.ndarray): 종가 배열.
        volume (np.ndarray): 거래량 배열.
        window (int): 계산에 사용할 기간 (예: 14)
        
    Returns:
        np.ndarray: VWRSI 값이 담긴 배열 (초기 window-1 인덱스는 np.nan)
    """
    n = close.shape[0]
    vw_rsi = np.empty(n, dtype=np.float64)
    
    # 초기값 설정: 초기 window-1 값은 계산 불가하므로 np.nan 할당.
    for i in range(n):
        vw_rsi[i] = np.nan
    
    # 1. 종가 변화(diff) 계산 (첫 번째 값은 0 처리)
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = close[i] - close[i-1]
    
    # 2. 상승(gain)과 하락(loss) 분리 (하락은 절대값)
    gain = np.empty(n, dtype=np.float64)
    loss = np.empty(n, dtype=np.float64)
    for i in range(n):
        if diff[i] > 0:
            gain[i] = diff[i]
            loss[i] = 0.0
        elif diff[i] < 0:
            gain[i] = 0.0
            loss[i] = -diff[i]
        else:
            gain[i] = 0.0
            loss[i] = 0.0
    
    # 3. 거래량 가중치 적용
    weighted_gain = np.empty(n, dtype=np.float64)
    weighted_loss = np.empty(n, dtype=np.float64)
    for i in range(n):
        weighted_gain[i] = gain[i] * np.sqrt(volume[i])
        weighted_loss[i] = loss[i] * np.sqrt(volume[i])
    
    # 4. rolling window를 사용해 기간 동안의 가중합 계산 및 VWRSI 산출
    for i in range(window - 1, n):
        sum_gain = 0.0
        sum_loss = 0.0
        for j in range(i - window + 1, i + 1):
            sum_gain += weighted_gain[j]
            sum_loss += weighted_loss[j]
        
        # 하락의 합계가 0이면 RSI는 100으로 설정
        if sum_loss == 0.0:
            vw_rsi[i] = 100.0
        else:
            rs = sum_gain / sum_loss
            vw_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return vw_rsi

@njit
def total_least_squares(x, y):
    """
    Total Least Squares를 이용하여 x와 y 데이터에 대한 선형 회귀
    (y = a*x + b)를 수행합니다.
    
    Parameters:
        x (np.ndarray): 독립 변수 데이터 (1D array)
        y (np.ndarray): 종속 변수 데이터 (1D array)
    
    Returns:
        a (float): 회귀선의 기울기
        b (float): 회귀선의 절편
    """
    n = x.shape[0]
    
    # x와 y의 평균 계산
    x_mean = 0.0
    y_mean = 0.0
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    # 중심화된 데이터에 대해 공분산 행렬의 요소 계산: S_xx, S_xy, S_yy
    S_xx = 0.0
    S_xy = 0.0
    S_yy = 0.0
    for i in range(n):
        dx = x[i] - x_mean
        dy = y[i] - y_mean
        S_xx += dx * dx
        S_xy += dx * dy
        S_yy += dy * dy

    # 2x2 공분산 행렬: [[S_xx, S_xy], [S_xy, S_yy]]
    # eigenvalue 문제를 풀어 작은 고유값에 해당하는 eigenvector를 찾습니다.
    # delta = sqrt((S_xx - S_yy)^2 + 4*S_xy^2)
    delta = np.sqrt((S_xx - S_yy)**2 + 4.0 * S_xy**2)
    # 두 고유값: lambda1 >= lambda2
    lambda1 = (S_xx + S_yy + delta) / 2.0
    lambda2 = (S_xx + S_yy - delta) / 2.0  # 최소 고유값

    # lambda2에 해당하는 eigenvector [v0, v1]를 구합니다.
    # (S_xx - lambda2)*v0 + S_xy*v1 = 0  =>  v1 = - (S_xx - lambda2)/S_xy  (S_xy ≠ 0)
    if np.abs(S_xy) > 1e-12:
        v0 = 1.0
        v1 = - (S_xx - lambda2) / S_xy
    else:
        # S_xy가 거의 0이면, 행렬이 대각행렬에 가까워집니다.
        if S_xx < S_yy:
            v0 = 1.0
            v1 = 0.0
        else:
            v0 = 0.0
            v1 = 1.0

    # TLS에서는 eigenvector [v0, v1]에 대해 선형모델은
    # v0 + v1*a = 0  =>  a = -v0 / v1 로 얻을 수 있습니다.
    if np.abs(v1) < 1e-12:
        a = np.inf  # v1이 0에 가까우면 기울기는 무한대로 처리
    else:
        a = -v0 / v1

    # 절편은 중심화한 평균을 이용: b = y_mean - a*x_mean
    b = y_mean - a * x_mean

    return a, b

def _calculate_half_life(spread):
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
        return np.inf  # Return a large value if spread is not mean-reverting
    half_life = -np.log(2) / lambda_param
    return half_life 

def calculate_half_life_with_pvalue(spread):

    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    spread_lag = spread_lag.iloc[1:]
    spread_ret = spread_ret.iloc[1:]
    
    model = sm.OLS(spread_ret, sm.add_constant(spread_lag))
    result = model.fit()
    
    lambda_param = result.params.iloc[1]
    stderr = result.bse.iloc[1]  # Standard error of lambda
    p_value = result.pvalues.iloc[1]  # P-value of lambda
    
    if lambda_param >= 0:
        return np.inf, stderr, p_value  # Not mean-reverting
    
    half_life = -np.log(2) / lambda_param
    
    return half_life, stderr, p_value

logger = logging.getLogger('PairTradingModel')

class PairTradingModel:
    def __init__(self, config):
        self.formation_window = config['FORMATION_WINDOW']
        self.timeframe = config['TIMEFRAME']
        self.days = config['DAYS']
        self.window_ratio = config['WINDOW_RATIO']
        self.trading_fee = config.get('TRADING_FEE', 0.0005)
        self.precomputed_values = None

        self.empty_list = []
        self.counter = 0

    def load_precomputed_values(self, filepath):
        """
        Load precomputed values from a pickle file and split into training and testing sets.
        """
        try:
            with open(filepath, 'rb') as f:
                all_values = pickle.load(f)
            logger.info(f"Loaded precomputed values from {filepath}")
            # sort the precomputed values by 'window_start' key
            all_values = sorted(all_values, key=lambda x: x['window_start'])
            
            split_index = int(len(all_values) * 0.8)  # 80% for training
            self.training_values = all_values[:split_index]
            self.testing_values = all_values[split_index:]
            logger.info(f"Split precomputed values into {len(self.training_values)} training and {len(self.testing_values)} testing records")

        except Exception as e:
            logger.error(f"Error loading precomputed values: {e}")

    def simulate_trading(self, entry_thres, stop_loss, dataset=None, slippage=0.005, verbose=False, test_plot=False):
        """
        Simulate trading based on precomputed hedge ratios and z-scores.
        """
        assert dataset is not None or self.precomputed_values is not None, "No dataset or precomputed values provided."
        

        trade_count = 0
       
        for record in dataset:
            spread = record['spread']
            z_spread = record['z_spread']
            split_index = record['split_index']
            close1 = record['close1']
            close2 = record['close2']
            vol1 = record['vol1']
            vol2 = record['vol2']

            # if np.abs(z_spread.iloc[split_index-1]) > entry_thres:   

            trade_count += 1
            z_spread_0 = z_spread.iloc[split_index-1]
            spread_0 = spread.iloc[split_index-1]

            observe_spread = spread.iloc[split_index:]
            observe_z_spread = z_spread.iloc[split_index:]

            signal_0 = -1 if z_spread_0 > 0 else 1
            profit, trade_duration = 0, 0
            testing_time = len(observe_z_spread)
            max_trading_time = int(testing_time)

            fig, axes = plt.subplots(5, 2, figsize=(30, 15))
            log_close1 = np.log(close1)
            log_close2 = np.log(close2)             


                  
            Y = np.column_stack([log_close1[:], log_close2[:]])
            model = KalmanModel(model_type="mu_gamma_ddot", Y=Y, split_index=split_index)
            state_all = model.run_smoother()
            print(f"state_last: {state_all[split_index-1]}")

            # Standard OLS for comparison
            X = sm.add_constant(log_close2[:split_index])
            model_fit = sm.OLS(log_close1[:split_index], X).fit()
            intercept, beta = model_fit.params
            spread_ = log_close1 - (beta * log_close2 + intercept)  
            spread_ = spread_ - spread_[:split_index].mean()
            spread_kalman = log_close1 - (state_all[:, 1] * log_close2 + state_all[:, 0])
            
            N_bars = _calculate_half_life(spread_[:split_index])
            half_life0, stderr0, p_value0 = calculate_half_life_with_pvalue(spread_[:split_index])
            print(N_bars, p_value0, p_value0 < 0.05)

            rsi1 = ta.momentum.RSIIndicator(close1, window=14).rsi()
            rsi2 = ta.momentum.RSIIndicator(close2, window=14).rsi()
            # rsi1 = compute_vw_rsi_nb(np.array(close1), np.array(vol1), 14)
            # rsi2 = compute_vw_rsi_nb(np.array(close2), np.array(vol2), 14)

            #plot log close
            log_close1_hat = beta * log_close2 + intercept
            # forecasted_state, forecasted_cov = model.forecast_state_with_covariance(steps=int(N_bars))
            # forecasted_mu = forecasted_state[0]
            # forecasted_gamma = forecasted_state[1]
            # kalman_spread = log_close1 - (forecasted_gamma * log_close2 + forecasted_mu)
            # kalman_spread = kalman_spread + (spread_.iloc[split_index] - kalman_spread.iloc[split_index])

            x = log_close1.index  # time axis
            axes[0,0].plot(x, spread_, label="Spread")
            axes[0,0].axhline(y=spread_[:split_index].mean(), color='r', linestyle='--')

            axes[0,0].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            if N_bars < 50:
                axes[0,0].axvline(x=log_close1.index[split_index-1+int(N_bars)], color='k', linestyle='-')
                axes[0,0].axvline(x=log_close1.index[split_index-1+int(2*N_bars)], color='k', linestyle='-')
            
            axes[0,0].set_xlabel("Time")
            axes[0,0].set_ylim([-0.1, 0.1])
            axes[0,0].legend()
            axes[0,0].grid(True)

            log_vol1 = np.log(vol1)
            log_vol2 = np.log(vol2)
            axes[1,0].plot(close1.index, rsi1, label="RSI1")
            axes[1,0].plot(close1.index, rsi2, label="RSI2")
            axes[1,0].plot(close1.index, rsi1 - rsi2, label="RSI Spread", linewidth=3)
            axes[1,0].axhline(y=0, color='r', linestyle='--')
            axes[1,0].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            # horizontal line at 30 and 70 and 50
            axes[1,0].axhline(y=30, color='r', linestyle='--')
            axes[1,0].axhline(y=70, color='r', linestyle='--')
            axes[1,0].axhline(y=50, color='r', linestyle='--')

            axes[1,0].set_title("RSI")
            axes[1,0].set_xlabel("Time")
            axes[1,0].grid(True)
            axes[1,0].legend()

            score_rsi = np.min([1, np.abs(rsi1-rsi2).iloc[split_index-1]/50])
            print(f'score_rsi: {score_rsi}')

            dist1 = (close1.iloc[split_index-1] - ta.trend.sma_indicator(close1, window=200).iloc[split_index-1]) / ta.trend.sma_indicator(close1, window=200).iloc[split_index-1]
            ma7 = ta.trend.sma_indicator(close1, window=7)
            vwap100 = rolling_vwap(close1.values, vol1.values, 100)
            ma20 = ta.trend.sma_indicator(close1, window=20)
            ma50 = ta.trend.sma_indicator(close1, window=50)
            ma200 = ta.trend.sma_indicator(close1, window=200)
            boll20 = ta.volatility.bollinger_mavg(close1, window=200)
            boll_std = close1.rolling(window=100).std()
            boll_upper = boll20 + 2 * boll_std
            boll_lower = boll20 - 2 * boll_std
            pct_bound1 = (close1.iloc[split_index-1]-boll20.iloc[split_index-1])/boll_std.iloc[split_index-1]
            print(f'dist, boll: {dist1}, {pct_bound1}')

            axes[0,1].plot(close1.index, close1, label="Close1")
            axes[0,1].plot(close1.index, ma7, label="SMA7")
            axes[0,1].plot(close1.index, ma20, label="SMA20")
            axes[0,1].plot(close1.index, ma50, label="SMA50")
            axes[0,1].plot(close1.index, ma200, label="SMA200")
            axes[0,1].plot(close1.index, vwap100, label="VWAP100")
            axes[0,1].fill_between(close1.index, boll_upper, boll_lower, color='gray', alpha=0.5)

            # axes[0,1].plot(close1.index, ta.volume.volume_weighted_average_price(close1, vol1, window=100), label="VWAP100")
            axes[0,1].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            axes[0,1].set_xlabel("Time")
            axes[0,1].set_title("Close1")
            axes[0,1].grid(True)
            axes[0,1].legend()

            dist2 = (close2.iloc[split_index-1] - ta.trend.sma_indicator(close2, window=200).iloc[split_index-1]) / ta.trend.sma_indicator(close2, window=200).iloc[split_index-1]
            ma7 = ta.trend.sma_indicator(close2, window=7)
            vwap100 = rolling_vwap(close2.values, vol2.values, 100)
            ma20 = ta.trend.sma_indicator(close2, window=20)
            ma50 = ta.trend.sma_indicator(close2, window=50)
            ma200 = ta.trend.sma_indicator(close2, window=200)
            boll20 = ta.volatility.bollinger_mavg(close2, window=200)
            # Calculate the standard deviation directly
            boll_std = close2.rolling(window=100).std()
            boll_upper = boll20 + 2 * boll_std
            boll_lower = boll20 - 2 * boll_std
            pct_bound2 = (close2.iloc[split_index-1]-boll20.iloc[split_index-1])/boll_std.iloc[split_index-1]
            print(f'dist, boll: {dist2}, {pct_bound2}')
            axes[1,1].plot(close2.index, close2, label="Close2")
            axes[1,1].plot(close2.index, ma7, label="SMA7")
            axes[1,1].plot(close2.index, ma20, label="SMA20")
            axes[1,1].plot(close2.index, ma50, label="SMA50")
            axes[1,1].plot(close2.index, ma200, label="SMA200")
            axes[1,1].plot(close2.index, vwap100, label="VWAP100")
            #boll fill
            axes[1,1].fill_between(close2.index, boll_upper, boll_lower, color='gray', alpha=0.5)
            score_bb = np.min([1, np.abs(pct_bound1-pct_bound2)/2])
            print(f'score_bb: {score_bb}')
            score_dist = np.min([1, np.abs(dist1-dist2)/0.1])
            print(f'score_dist: {score_dist}')


            # axes[1,1].plot(close2.index, ta.volume.volume_weighted_average_price(close2, vol2, window=100), label="VWAP100")
            axes[1,1].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            axes[1,1].set_xlabel("Time")
            axes[1,1].set_title("Close2")
            axes[1,1].grid(True)
            axes[1,1].legend()

            # Plot the regression lines on the same axis
            axes[2,0].plot(log_close1.index, log_close1, label="Log Close1")
            axes[2,0].plot(log_close1.index, log_close1_hat, label="Log Close2")
            axes[2,0].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            axes[2,0].set_xlabel("Time")
            axes[2,0].set_title("Log Close")
            axes[2,0].grid(True)
            axes[2,0].legend()

            # plot bar chart of volume vol 1 and mean
            last_avg_vol1 = vol1[split_index-5:split_index].mean()
            vol1_median = vol1[:split_index].median()
            axes[3,0].bar(log_close1.index, vol1, label="Volume1", alpha=0.5, width=0.001)
            axes[3,0].axhline(y=vol1_median, color='r', linestyle='--')
            axes[3,0].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            axes[3,0].set_xlabel("Time")
            axes[3,0].set_title(f"Volume1: {vol1_median < last_avg_vol1}")
            axes[3,0].grid(True)

            # plot bar chart of volume vol 2 and mean
            last_avg_vol2 = vol2[split_index-5:split_index].mean()
            vol2_median = vol2[:split_index].median()
            axes[4,0].bar(log_close1.index, vol2, label="Volume2", alpha=0.5, width=0.001)
            axes[4,0].axhline(y=vol2_median, color='r', linestyle='--')
            axes[4,0].axvline(x=log_close1.index[split_index-1], color='g', linestyle='-')
            axes[4,0].set_xlabel("Time")
            axes[4,0].set_title(f"Volume2: {vol2_median < last_avg_vol2}")
            axes[4,0].grid(True)

            condition1 = N_bars is not None and N_bars < 50
            condition2 = np.sign(rsi1- rsi2).iloc[split_index-1] == np.sign(z_spread_0)
            condition3 = vol1_median < last_avg_vol1
            condition4 = vol2_median < last_avg_vol2
            trade_now = condition1 and condition2 and (condition3 or condition4)
            axes[0,0].set_title(f"Trade: {trade_now}")
            plt.tight_layout()
            plt.show()

        return 0

    def objective(self, params):
        """
        Objective function for optimization (to be minimized).
        """
        entry_thres, stop_loss, = params
        return -self.simulate_trading(entry_thres, stop_loss, dataset=self.training_values, verbose=False, test_plot=True)
        

    def optimize_parameters(self):
        """
        Optimize trading parameters using differential evolution on training data.
        """
        bounds = [
            (2, 2.5),    # entry_thres
            (0.5, 2),    # stop_loss
        ]

        logger.info("Starting parameter optimization using differential evolution...")
        result = differential_evolution(
            self.objective,
            bounds,
            strategy='best1bin',
            maxiter=30,
            popsize=50,
            tol=0.001,
            disp=True
        )

        entry_thres, stop_loss, = result.x

        logger.info("Optimal Parameters:")
        logger.info(f"Entry Threshold: {entry_thres:.3f}")
        logger.info(f"Stop Loss: {stop_loss:.3f}")
        logger.info(f"Optimal Cost Function Value: {-result.fun:.5f}")  # Return positive value

        # Test on testing set
        test_fval = self.simulate_trading(
            entry_thres, stop_loss,
            dataset=self.testing_values,
            verbose=True
        )
        logger.info(f"Test Function Value: {test_fval:.3f}%")

        return result.x, -result.fun, test_fval
