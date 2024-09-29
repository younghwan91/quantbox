import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
import logging
import pickle
import os

# Configuration Variables
FORMATION_WINDOW = 120
TRADING_WINDOW = 24
TOTAL_WINDOW = FORMATION_WINDOW + TRADING_WINDOW
WINDOW_RATIO = FORMATION_WINDOW / TOTAL_WINDOW
TIMEFRAME = '15m'
DAYS = 90
TRADING_FEE = 0.00045


class PairTradingModel:
    def __init__(self, formation_window, timeframe, days, window_ratio):
        self.formation_window = formation_window
        self.timeframe = timeframe
        self.days = days
        self.window_ratio = window_ratio
        self.precomputed_values = None

    def load_data(self, filepath):
        return np.load(filepath, allow_pickle=True)

    def precompute_hedge_ratios(self, train_pair_data):
        precomputed_values = []
        for t in tqdm.tqdm(range(len(train_pair_data))):
            pair_data_t = train_pair_data[t]
            for pair_name, pair_price in pair_data_t.items():
                common_indices = pair_price[pair_name[0]].index.intersection(pair_price[pair_name[1]].index)
                log_price_1 = np.log(pair_price[pair_name[0]].loc[common_indices]['Close'])
                log_price_2 = np.log(pair_price[pair_name[1]].loc[common_indices]['Close'])
                
                split_index = int(len(log_price_1) * self.window_ratio)
                init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
                spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)
                train_spread = spread[:split_index]
                mean_spread = train_spread.mean()
                std_spread = train_spread.std()
                z_spread = (spread - mean_spread) / std_spread

                precomputed_values.append((spread, z_spread, split_index))

        self.precomputed_values = precomputed_values

    def save_precomputed_values(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.precomputed_values, f)

    def simulate_trading(self, entry_thres, risk_reward_ratio, stop_loss, frac_hold, slippage=0.006):
        total_profit, cnt_win, cnt_loss, cnt_draw, cnt_profit = 0, 0, 0, 0, 0
        total_trade_duration = 0
        total_profit_per_duration = 0
        trade_count = 0
        returns = []

        for spread, z_spread, split_index in self.precomputed_values:
            
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
                        if current_z_spread < z_spread_0 - stop_loss or current_z_spread > z_spread_0 + risk_reward_ratio * stop_loss:
                            profit = (current_spread - spread_0) - (2*TRADING_FEE) - slippage
                            total_profit += profit
                            if current_z_spread > z_spread_0 + risk_reward_ratio * stop_loss:
                                cnt_win += 1
                            else:
                                cnt_loss += 1
                            returns.append(profit)
                            break                        
                        if i == max_trading_time - 1:
                            profit = (current_spread - spread_0) - (2*TRADING_FEE) - slippage
                            total_profit += profit
                            cnt_draw += 1
                            returns.append(profit)
                            break

                    elif signal_0 == -1:
                        if current_z_spread > z_spread_0 + stop_loss or current_z_spread < z_spread_0 - risk_reward_ratio * stop_loss:
                            profit = (spread_0 - current_spread) - (2*TRADING_FEE) - slippage
                            total_profit += profit
                            if current_z_spread < z_spread_0 - risk_reward_ratio * stop_loss:
                                cnt_win += 1
                            else:
                                cnt_loss += 1
                            returns.append(profit)
                            break
                        if i == max_trading_time - 1:
                            profit = (spread_0 - current_spread) - (2*TRADING_FEE) - slippage
                            total_profit += profit
                            cnt_draw += 1
                            returns.append(profit)
                            break                
                
                if profit > 0:
                    cnt_profit += 1
                
                total_trade_duration += trade_duration
                profit_per_duration = profit / trade_duration if trade_duration > 0 else 0
                total_profit_per_duration += profit_per_duration
                
        avg_profit_per_duration = total_profit_per_duration / trade_count if trade_count > 0 else 0
        avg_profit_per_trade = total_profit / trade_count if trade_count > 0 else 0

        T_TOTAL = 24
        T_SEARCH = 1
        N_DAYS = DAYS*2
        N_effective = (T_TOTAL-T_SEARCH*(1/frac_hold-1)) / (T_TOTAL*frac_hold)
        N_avg_trade = trade_count / N_DAYS
        N_trade_per_day = min(N_effective, N_avg_trade)
        func_val = avg_profit_per_trade * N_trade_per_day

        print('='*50)
        print(f'Trade Count:  {trade_count}')
        print(f'Win Rate:     {cnt_win/trade_count:.3f}')
        print(f'Loss Rate:    {cnt_loss/trade_count:.3f}')
        print(f'Draw Rate:    {cnt_draw/trade_count:.3f}')
        print(f'Profit Rate:  {cnt_profit/trade_count:.3f}')
        print(f'Number of Trades per Day: {N_trade_per_day:.3f}')
        print(f'Number of Effective Trades: {N_effective:.3f}')
        print(f'Avg. Number of Trades: {N_avg_trade:.3f}')
        print('-'*20 + 'Parameters' + '-'*20)
        print(f'Threshold:    {entry_thres:.3f}')
        print(f'Stop Loss:    {stop_loss:.3f}')
        print(f'Risk-Reward:  {risk_reward_ratio:.3f}')
        print(f'Frac. hold:   {frac_hold:.3f}')
        print(f'Avg Profit per Duration:{avg_profit_per_duration*100:.3f}%')
        print(f'Avg Profit per Trade:   {avg_profit_per_trade*100:.3f}%/trade')
        print(f'Function value:         {func_val*100:.3f}%')
        print('='*50)

        # return fval/frac_hold
        return func_val*100

    def objective(self, params):
        entry_thres, risk_reward_ratio, stop_loss, frac_hold = params
        return -self.simulate_trading(entry_thres, risk_reward_ratio, stop_loss, frac_hold)

    def optimize_parameters(self):
        # Define bounds for each parameter
        bounds = [(1, 2.5),    # entry_thres
                (1, 3), # risk_reward_ratio
                (0.5, 2), # stop_loss
                (0.2, 0.9)]   # frac_hold

        result = differential_evolution(self.objective, bounds, strategy='best1bin', maxiter=30, popsize=20, tol=0.001)
        entry_thres, risk_reward_ratio, stop_loss, frac_hold = result.x

        # Print the results
        print("Optimal Parameters:")
        print(f"Entry Threshold: {entry_thres:.3f}")
        print(f"Risk-Reward Ratio: {risk_reward_ratio:.3f}")
        print(f"Stop Loss: {stop_loss:.3f}")
        print(f"Fraction of Hold: {frac_hold:.3f}")
        print(f"Optimal Cost Function Value: {-result.fun:.5f}")  # Return positive value

        return result.x, -result.fun  # Return the best params and the corresponding profit

# Running the Process
if __name__ == "__main__":
    model = PairTradingModel(FORMATION_WINDOW, TIMEFRAME, DAYS, WINDOW_RATIO)
    
    # Load training data
    filepath = f'./training_data/pair_data_{FORMATION_WINDOW}_{TRADING_WINDOW}_{TIMEFRAME}_{DAYS}.npy'
    train_pair_data = model.load_data(filepath)
    
    # Precompute values and save
    pkl_filepath = f'./training_data/precomputed_{FORMATION_WINDOW}_{TIMEFRAME}_{DAYS}.pkl'
    if not os.path.exists(pkl_filepath):
        model.precompute_hedge_ratios(train_pair_data)
        model.save_precomputed_values(pkl_filepath)

    # Load precomputed values
    with open(pkl_filepath, 'rb') as f:
        print('Loading precomputed values...')
        model.precomputed_values = pickle.load(f)
        print(f'Length of precomputed values: {len(model.precomputed_values)}')

    # Optimize the parameters
    best_params, max_fval = model.optimize_parameters()
    print(f'Best parameters: {best_params} with max function value: {max_fval}')