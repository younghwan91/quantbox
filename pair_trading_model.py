# pair_trading_model.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from scipy.stats import shapiro
from scipy.optimize import differential_evolution
import pickle
import os

logger = logging.getLogger('PairTradingModel')

class PairTradingModel:
    def __init__(self, config):
        self.formation_window = config['FORMATION_WINDOW']
        self.timeframe = config['TIMEFRAME']
        self.days = config['DAYS']
        self.window_ratio = config['WINDOW_RATIO']
        self.trading_fee = config.get('TRADING_FEE', 0.0005)
        self.precomputed_values = None

    def load_precomputed_values(self, filepath):
        """
        Load precomputed values from a pickle file.
        """
        try:
            with open(filepath, 'rb') as f:
                self.precomputed_values = pickle.load(f)
            logger.info(f"Loaded precomputed values from {filepath}")
        except Exception as e:
            logger.error(f"Error loading precomputed values: {e}")

    def simulate_trading(self, entry_thres, risk_reward_ratio, stop_loss, frac_hold, slippage=0.006):
        """
        Simulate trading based on precomputed hedge ratios and z-scores.
        """
        total_profit, cnt_win, cnt_loss, cnt_draw, cnt_profit = 0, 0, 0, 0, 0
        total_trade_duration = 0
        total_profit_per_duration = 0
        trade_count = 0
        returns = []

        for record in self.precomputed_values:
            spread = record['spread']
            z_spread = record['z_spread']
            split_index = record['split_index']

            if np.abs(z_spread.iloc[split_index]) > entry_thres:
                trade_count += 1
                z_spread_0 = z_spread.iloc[split_index]
                spread_0 = spread.iloc[split_index]

                observe_spread = spread.iloc[split_index:]
                observe_z_spread = z_spread.iloc[split_index:]

                signal_0 = -1 if z_spread_0 > 0 else 1
                profit, trade_duration = 0, 0
                testing_time = len(observe_z_spread)
                max_trading_time = int(frac_hold * testing_time)

                for i in range(1, max_trading_time):
                    current_spread = observe_spread.iloc[i]
                    current_z_spread = observe_z_spread.iloc[i]
                    trade_duration += 1

                    if signal_0 == 1:
                        if (current_z_spread < z_spread_0 - stop_loss) or (current_z_spread > z_spread_0 + risk_reward_ratio * stop_loss):
                            profit = (current_spread - spread_0) - (2 * self.trading_fee) - slippage
                            total_profit += profit
                            if current_z_spread > z_spread_0 + risk_reward_ratio * stop_loss:
                                cnt_win += 1
                            else:
                                cnt_loss += 1
                            returns.append(profit)
                            break
                        if i == max_trading_time - 1:
                            profit = (current_spread - spread_0) - (2 * self.trading_fee) - slippage
                            total_profit += profit
                            cnt_draw += 1
                            returns.append(profit)
                            break

                    elif signal_0 == -1:
                        if (current_z_spread > z_spread_0 + stop_loss) or (current_z_spread < z_spread_0 - risk_reward_ratio * stop_loss):
                            profit = (spread_0 - current_spread) - (2 * self.trading_fee) - slippage
                            total_profit += profit
                            if current_z_spread < z_spread_0 - risk_reward_ratio * stop_loss:
                                cnt_win += 1
                            else:
                                cnt_loss += 1
                            returns.append(profit)
                            break
                        if i == max_trading_time - 1:
                            profit = (spread_0 - current_spread) - (2 * self.trading_fee) - slippage
                            total_profit += profit
                            cnt_draw += 1
                            returns.append(profit)
                            break                

                if profit > 0:
                    cnt_profit += 1

                total_trade_duration += trade_duration
                profit_per_duration = profit / trade_duration if trade_duration > 0 else 0
                total_profit_per_duration += profit_per_duration

        if trade_count == 0:
            logger.warning("No trades were made during simulation.")
            return 0

        avg_profit_per_duration = total_profit_per_duration / trade_count
        avg_profit_per_trade = total_profit / trade_count

        T_TOTAL = 24
        T_SEARCH = 0
        N_DAYS = self.days
        N_effective = (T_TOTAL - T_SEARCH * (1 / frac_hold - 1)) / (T_TOTAL * frac_hold)
        N_avg_trade = trade_count / N_DAYS
        N_trade_per_day = min(N_effective, N_avg_trade)
        func_val = avg_profit_per_trade * N_trade_per_day * 100

        logger.info('=' * 50)
        logger.info(f'Trade Count: {trade_count} | Effective Trades: {N_effective:.3f} | Avg. Number of Trades: {N_avg_trade:.3f} | Trades/Day: {N_trade_per_day:.3f}')
        logger.info(f'Win Rate: {cnt_win / trade_count:.3f} | Loss Rate: {cnt_loss / trade_count:.3f} | Draw Rate: {cnt_draw / trade_count:.3f} | Profit Rate: {cnt_profit / trade_count:.3f}')
        logger.info('-' * 20 + ' Parameters ' + '-' * 20)
        logger.info(f'Threshold: {entry_thres:.3f} | Stop Loss: {stop_loss:.3f} | Risk-Reward: {risk_reward_ratio:.3f} | Frac. Hold: {frac_hold:.3f}')
        logger.info(f'Avg Profit/Duration: {avg_profit_per_duration * 100:.3f}% | Avg Profit/Trade: {avg_profit_per_trade * 100:.3f}%/trade | Function Value: {func_val:.3f}%')
        logger.info('=' * 50)

        return func_val

    def objective(self, params):
        """
        Objective function for optimization (to be minimized).
        """
        entry_thres, risk_reward_ratio, stop_loss, frac_hold = params
        return -self.simulate_trading(entry_thres, risk_reward_ratio, stop_loss, frac_hold)

    def optimize_parameters(self):
        """
        Optimize trading parameters using differential evolution.
        """
        bounds = [
            (2, 3),    # entry_thres
            (1, 3),      # risk_reward_ratio
            (0.5, 2),    # stop_loss
            (0.2, 0.9)    # frac_hold
        ]

        logger.info("Starting parameter optimization using differential evolution...")
        result = differential_evolution(
            self.objective,
            bounds,
            strategy='best1bin',
            maxiter=30,
            popsize=20,
            tol=0.001,
            disp=True
        )

        entry_thres, risk_reward_ratio, stop_loss, frac_hold = result.x

        logger.info("Optimal Parameters:")
        logger.info(f"Entry Threshold: {entry_thres:.3f}")
        logger.info(f"Risk-Reward Ratio: {risk_reward_ratio:.3f}")
        logger.info(f"Stop Loss: {stop_loss:.3f}")
        logger.info(f"Fraction of Hold: {frac_hold:.3f}")
        logger.info(f"Optimal Cost Function Value: {-result.fun:.5f}")  # Return positive value

        return result.x, -result.fun  # Return the best params and the corresponding profit
