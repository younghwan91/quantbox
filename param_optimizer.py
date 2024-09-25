import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
import pickle
from pyswarm import pso
import os

# Configuration Variables
train_hour = 120
test_hour = 24
total_hour = train_hour + test_hour
train_ratio = train_hour / total_hour
timeframe = '15m'
days = 90
trading_fee = 0.00045


class PairTradingModel:
    def __init__(self, train_hour, timeframe, days, train_ratio):
        self.train_hour = train_hour
        self.timeframe = timeframe
        self.days = days
        self.train_ratio = train_ratio
        self.precomputed_values = None

    def load_data(self, filepath):
        return np.load(filepath, allow_pickle=True)

    def calculate_optimal_thresholds(self, mean_spread, std_spread, T):
        f = lambda s0: -(s0 - (0.0005 * 4)) * T * (1 - stats.norm.cdf(s0, mean_spread, std_spread))
        x = np.linspace(0, 0.05, 100)
        f_values = [f(s0) for s0 in x]
        s0 = x[np.argmin(f_values)]
        lower_bound = -(s0 - mean_spread) / std_spread
        upper_bound = (s0 - mean_spread) / std_spread
        return lower_bound, upper_bound, s0

    def generate_signal_t(self, z_spread, split_index, threshold_long, threshold_short):
        if z_spread.iloc[split_index] <= threshold_long:
            return 1
        elif z_spread.iloc[split_index] >= threshold_short:
            return -1
        return 0

    def precompute_hedge_ratios(self, train_pair_data):
        precomputed_values = []
        for t in tqdm.tqdm(range(len(train_pair_data))):
            pair_data_t = train_pair_data[t]
            for pair_name, pair_price in pair_data_t.items():
                common_indices = pair_price[pair_name[0]].index.intersection(pair_price[pair_name[1]].index)
                log_price_1 = np.log(pair_price[pair_name[0]].loc[common_indices]['Close'])
                log_price_2 = np.log(pair_price[pair_name[1]].loc[common_indices]['Close'])
                
                split_index = int(len(log_price_1) * self.train_ratio)
                init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
                spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)
                train_spread = spread[:split_index]
                mean_spread = train_spread.mean()
                std_spread = train_spread.std()
                z_spread = (spread - mean_spread) / std_spread
                # T = len(spread)
                # lower_bound, upper_bound, s0 = self.calculate_optimal_thresholds(mean_spread, std_spread, T)
                # signal_t = self.generate_signal_t(z_spread, split_index, lower_bound, upper_bound)

                precomputed_values.append((spread, z_spread, split_index))

        self.precomputed_values = precomputed_values

    def save_precomputed_values(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.precomputed_values, f)

    @staticmethod
    def calculate_sharpe_ratio(returns):
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return mean_return / std_return

    def cost_function(self, entry_thres, risk_reward_ratio, stop_loss, frac_hold, slippage=0.006):
        total_profit, cnt_win, cnt_loss, cnt_draw, cnt_profit = 0, 0, 0, 0, 0
        
        total_trade_duration = 0
        total_profit_per_duration = 0
        trade_count = 0
        returns = []
        for spread, z_spread, split_index in self.precomputed_values:
            std_spread = spread[:split_index].std()
            
            if np.abs(z_spread.iloc[split_index]) > entry_thres:
                trade_count += 1
                z_spread_0 = z_spread.iloc[split_index]
                spread_0 = spread.iloc[split_index]

                observe_spread = spread.iloc[split_index:]
                observe_z_spread = z_spread.iloc[split_index:]

                signal_0 = -1 if z_spread_0 > 0 else 1

                profit = 0
                trade_duration = 0
                testing_time = len(observe_z_spread)
                max_trading_time = int(frac_hold * testing_time)

                for i in range(1, max_trading_time):
                    current_spread = observe_spread.iloc[i]
                    current_z_spread = observe_z_spread.iloc[i]
                    trade_duration += 1
                    
                    if signal_0 == 1:
                        if current_z_spread < z_spread_0 - stop_loss or current_z_spread > z_spread_0 + risk_reward_ratio * stop_loss:
                            profit = (current_spread - spread_0) - (2*trading_fee) - slippage
                            total_profit += profit
                            if current_z_spread > z_spread_0 + risk_reward_ratio * stop_loss:
                                cnt_win += 1
                            else:
                                cnt_loss += 1
                            returns.append(profit)
                            break                        
                        if i == max_trading_time - 1:
                            profit = (current_spread - spread_0) - (2*trading_fee) - slippage
                            total_profit += profit
                            cnt_draw += 1
                            returns.append(profit)
                            break

                    elif signal_0 == -1:
                        if current_z_spread > z_spread_0 + stop_loss or current_z_spread < z_spread_0 - risk_reward_ratio * stop_loss:
                            profit = (spread_0 - current_spread) - (2*trading_fee) - slippage
                            total_profit += profit
                            if current_z_spread < z_spread_0 - risk_reward_ratio * stop_loss:
                                cnt_win += 1
                            else:
                                cnt_loss += 1
                            returns.append(profit)
                            break
                        if i == max_trading_time - 1:
                            profit = (spread_0 - current_spread) - (2*trading_fee) - slippage
                            total_profit += profit
                            cnt_draw += 1
                            returns.append(profit)
                            break                
                
                if profit > 0:
                    cnt_profit += 1
                
                total_trade_duration += trade_duration
                profit_per_duration = profit / trade_duration if trade_duration > 0 else 0
                total_profit_per_duration += profit_per_duration
                
                # print(signal_0)
                # print(profit)
                # plt.plot(z_spread)
                # plt.axhline(0, color='black', linestyle='--')
                # plt.axvline(z_spread.index[split_index], color='red', linestyle='--')
                # if signal_0 == 1:
                #     plt.axhline(z_spread_0 - stop_loss, color='green', linestyle='--')
                #     plt.axhline(z_spread_0 + risk_reward_ratio * stop_loss, color='blue', linestyle='--')
                # elif signal_0 == -1:
                #     plt.axhline(z_spread_0 + stop_loss, color='green', linestyle='--')
                #     plt.axhline(z_spread_0 - risk_reward_ratio * stop_loss, color='blue', linestyle='--')
                # plt.show()

        avg_profit_per_duration = total_profit_per_duration / trade_count if trade_count > 0 else 0
        avg_profit_per_trade = total_profit / trade_count if trade_count > 0 else 0

        T_total = 24
        T_search = 1
        N_days = 90*1.5
        N_effective = (T_total-T_search*(1/frac_hold-1)) / (T_total*frac_hold)
        N_trade_avg = trade_count / N_days
        N_trade_per_day = min(N_effective, N_trade_avg)
        func_val = avg_profit_per_trade * N_trade_per_day

        print('='*50)
        print(f'Trade Count:  {trade_count}')
        print(f'Win Rate:     {cnt_win/trade_count:.3f}')
        print(f'Loss Rate:    {cnt_loss/trade_count:.3f}')
        print(f'Draw Rate:    {cnt_draw/trade_count:.3f}')
        print(f'Profit Rate:  {cnt_profit/trade_count:.3f}')
        print(f'Number of Trades per Day: {N_trade_per_day:.3f}')
        print(f'Number of Effective Trades: {N_effective:.3f}')
        print(f'Avg. Number of Trades: {N_trade_avg:.3f}')
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
        return func_val

    def objective(self, params):
        entry_thres, risk_reward_ratio, stop_loss, frac_hold = params
        return -self.cost_function(entry_thres, risk_reward_ratio, stop_loss, frac_hold)

    def optimize_parameters(self):
        # Define bounds for each parameter
        bounds = [(1, 2.5),    # entry_thres
                (1, 3), # risk_reward_ratio
                (0.5, 2), # stop_loss
                (0.2, 0.9)]   # frac_hold

        # Call the differential_evolution optimizer
        result = differential_evolution(self.objective, bounds, strategy='best1bin', maxiter=30, popsize=20, tol=0.001)

        # Unpack the optimized parameters
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
    model = PairTradingModel(train_hour, timeframe, days, train_ratio)
    
    # Load training data
    filepath = f'./training_data/pair_data_{train_hour}_{test_hour}_{timeframe}_{days}.npy'
    train_pair_data = model.load_data(filepath)
    
    # Precompute values and save
    pkl_filepath = f'./training_data/precomputed_{train_hour}_{timeframe}_{days}.pkl'
    if not os.path.exists(pkl_filepath):
        model.precompute_hedge_ratios(train_pair_data)
        model.save_precomputed_values(f'./training_data/precomputed_{train_hour}_{timeframe}_{days}.pkl')

    # Load precomputed values
    with open(f'./training_data/precomputed_{train_hour}_{timeframe}_{days}.pkl', 'rb') as f:
        print('Loading precomputed values...')
        model.precomputed_values = pickle.load(f)
        print(f'Length of precomputed values: {len(model.precomputed_values)}')
    # Optimize the parameters
    best_params, max_fval = model.optimize_parameters()
    print(f'Best parameters: {best_params} with max function value: {max_fval}')