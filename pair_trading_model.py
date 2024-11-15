# pair_trading_model.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from scipy.stats import shapiro
from scipy.optimize import differential_evolution
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates

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

            if np.abs(z_spread.iloc[split_index-1]) > entry_thres:                    
                exit_condition = False
                trade_count += 1
                z_spread_0 = z_spread.iloc[split_index-1]
                spread_0 = spread.iloc[split_index-1]

                observe_spread = spread.iloc[split_index:]
                observe_z_spread = z_spread.iloc[split_index:]

                signal_0 = -1 if z_spread_0 > 0 else 1
                profit, trade_duration = 0, 0
                testing_time = len(observe_z_spread)

                max_trading_time = int(testing_time)

                if test_plot and 1==1:
                    plt.figure(figsize=(20, 10))
                    # Plot z-spread
                    plt.plot(z_spread, '.-')
                    plt.axvline(x=spread.index[split_index], color='r', linestyle='--')
                    plt.axhline(y=0, color='k', linewidth=2)
                    plt.axhline(y=z_spread_0, color='g', linestyle='-')
                    plt.ylim(-6, 6)
                    # Draw horizontal lines between 0 and z_spread_0
                    plt.axhline(y=z_spread_0/2, color='y', linestyle='--')
                    plt.axhline(y=z_spread_0/4, color='y', linestyle='--')
                    plt.axhline(y=z_spread_0*3/4, color='y', linestyle='--')
                    plt.show()
                    plt.close()

                # Open initial 0.25 position immediately
                position_size = 1
                cumulative_position = position_size * signal_0
                opened_positions = [{
                    'position_size': position_size,
                    'entry_price': spread_0
                }]

                # Define remaining z-score levels and position sizes
                z_levels = np.linspace(z_spread_0, 0, num=4)[1:]
                position_sizes = [0.0, 0.0, 0]

                for i in range(0, max_trading_time):
                    current_spread = observe_spread.iloc[i]
                    current_z_spread = observe_z_spread.iloc[i]
                    trade_duration += 1

                    # Calculate current profit based on opened positions
                    current_profit = 0
                    for pos in opened_positions:
                        pos_profit = (current_spread - pos['entry_price']) * pos['position_size'] * signal_0
                        current_profit += pos_profit
                    current_profit -= (2 * self.trading_fee) + slippage

                    # Check for trailing stop loss hit
                    if (signal_0 == 1 and current_z_spread > 0) or (signal_0 == -1 and current_z_spread < 0):
                        exit_condition = True
                    if (signal_0 == 1 and current_z_spread < -stop_loss+ z_spread_0) or (signal_0 == -1 and current_z_spread > stop_loss + z_spread_0):
                        exit_condition = True

                    # Open positions as z-score reaches the levels
                    while len(z_levels) > 0 and (
                        (signal_0 == 1 and current_z_spread >= z_levels[0]) or 
                        (signal_0 == -1 and current_z_spread <= z_levels[0])
                    ):
                        position_size = position_sizes.pop(0)
                        cumulative_position += position_size * signal_0
                        opened_positions.append({
                            'entrty_z_spread': z_levels[0],
                            'position_size': position_size,
                            'entry_price': current_spread
                        })
                        z_levels = z_levels[1:]

                    if not test_plot:
                        plt.figure(figsize=(20, 10))
                        # Plot z-spread
                        plt.plot(z_spread, '.-')
                        plt.axvline(x=spread.index[split_index], color='r', linestyle='--')
                        plt.axhline(y=0, color='k', linewidth=4)
                        plt.axhline(y=z_spread_0, color='g', linestyle='-')
                        plt.ylim(-6, 6)
                        # Draw horizontal lines between 0 and z_spread_0
                        plt.axhline(y=z_spread_0/2, color='y', linestyle='--')
                        plt.axhline(y=z_spread_0/4, color='y', linestyle='--')
                        plt.axhline(y=z_spread_0*3/4, color='y', linestyle='--')
                        plt.title(f'Profit: {current_profit:.4f} ')
                        plt.plot(spread.index[split_index+i], current_z_spread, 'ro')
                        plt.show()
                        plt.close()
                    


                    # Exit trade if exit condition met
                    if exit_condition or i == max_trading_time - 1:
                        # Calculate profit based on all opened positions
                        profit = current_profit
                        total_profit += profit
                        # Update counters (cnt_win, cnt_loss, etc.)
                        if profit > 0:
                            cnt_win += 1
                            cnt_profit += 1
                        elif profit < 0:
                            cnt_loss += 1
                        else:
                            cnt_draw += 1

                        break

                total_trade_duration += trade_duration
                profit_per_duration = profit / trade_duration if trade_duration > 0 else 0
                total_profit_per_duration += profit_per_duration

        if trade_count == 0:
            logger.warning("No trades were made during simulation.")
            return 0

        avg_profit_per_duration = total_profit_per_duration / trade_count
        avg_profit_per_trade = total_profit / trade_count


        N_DAYS = self.days
        N_avg_trade = trade_count / N_DAYS
        N_trade_per_day = N_avg_trade
        func_val = avg_profit_per_trade * N_trade_per_day * 100

        if verbose:
            logger.info(f'Trade Count: {trade_count} | Avg. Number of Trades: {N_avg_trade:.3f} | Trades/Day: {N_trade_per_day:.3f}')
            logger.info(f'Win Rate: {cnt_win / trade_count:.3f} | Loss Rate: {cnt_loss / trade_count:.3f} | Draw Rate: {cnt_draw / trade_count:.3f} | Profit Rate: {cnt_profit / trade_count:.3f}')
            logger.info('-' * 20 + ' Parameters ' + '-' * 20)
            logger.info(f'Threshold: {entry_thres:.3f} | Stop Loss: {stop_loss:.3f}')
            logger.info(f'Avg Profit/Duration: {avg_profit_per_duration * 100:.3f}% | Avg Profit/Trade: {avg_profit_per_trade * 100:.3f}%/trade | Function Value: {func_val:.3f}%')
            logger.info('=' * 50)

        return func_val

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
