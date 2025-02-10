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
        
        total_profit, cnt_win, cnt_loss, cnt_draw, cnt_profit = 0, 0, 0, 0, 0
        total_trade_duration = 0
        total_profit_per_duration = 0
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


            # Create three subplots vertically stacked
            fig, axes = plt.subplots(3, 1, figsize=(20, 15))

            log_close1 = np.log(close1)
            log_close2 = np.log(close2)                   
            beta = np.polyfit(log_close1[:split_index], log_close2[:split_index], 1)[0] # reverse the order of the two assets
            print(beta)
            spread_ = log_close1 - beta * log_close2  

            x = mdates.date2num(log_close1.index)
            # x = np.linspace(0,1,len(log_close1))
            x = x - x[0]
            x = x.reshape(-1, 1)
            y = spread_
            model = LinearRegression()
            model.fit(x[:split_index], y[:split_index])
            # print slope and intercept
            print(model.coef_, model.intercept_)
            # compute R^2
            r_sq = model.score(x[:split_index], y[:split_index])
            print('coefficient of determination:', r_sq)
            y_pred = model.predict(x)
            axes[0].plot(log_close1.index, spread_, label="Spread")
            axes[0].plot(log_close1.index, y_pred, label="Spread Regression", linestyle='--')
            axes[0].axhline(y=spread_[:split_index].mean(), color='r', linestyle='--')
            axes[0].axvline(x=log_close1.index[split_index], color='g', linestyle='-')
            axes[0].set_title(f"R2 = {r_sq:.3f} and slope = {model.coef_[0]:.3f}")
            axes[0].set_xlabel("Time")
            # grid on
            axes[0].grid(True)

            log_vol1 = np.log(vol1)
            log_vol2 = np.log(vol2)
            epsilon = y_pred - spread_
            div_log_vol = log_vol1 / log_vol2
            # scatter plot of delta_epsilon and delta_sum_log_vol
            axes[1].scatter(div_log_vol, epsilon, label="Epsilon vs Sum Log Volume")
            axes[1].set_title("Epsilon vs Sum Log Volume")
            axes[1].set_xlabel("Sum Log Volume")
            axes[1].set_ylabel("Epsilon")
            # grid on
            axes[1].grid(True)
            axes[1].legend()

            # volume analysis log differenc3 between the two assets
            log_diff_vol = log_vol1 - log_vol2
            # plot moving average of log difference volume
            ma10 = log_diff_vol.rolling(window=10).mean()
            ma20 = log_diff_vol.rolling(window=20).mean()
            ma40 = log_diff_vol.rolling(window=40).mean()
            axes[2].plot(log_close1.index, ma10, label="MA5")
            axes[2].plot(log_close1.index, ma20, label="MA20")
            axes[2].plot(log_close1.index, ma40, label="MA40")
            axes[2].plot(log_close1.index, log_diff_vol, label="Log Difference Volume")
            axes[2].axhline(y=log_diff_vol[:split_index].mean(), color='r', linestyle='--')
            axes[2].axvline(x=log_close1.index[split_index], color='g', linestyle='-')
            axes[2].set_title("Log Difference Volume")
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Log Difference Volume")
            # grid on
            axes[2].grid(True)
            axes[2].legend()




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
