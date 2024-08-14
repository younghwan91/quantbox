# Step 1: Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
import pickle
from deap import base, creator, tools, algorithms

train_pair_data = np.load('./training_data/pair_data_24_6_5m_30.npy', allow_pickle=True)

def calculate_optimal_thresholds(mean_spread, std_spread, T):
    f = lambda s0: -(s0 - (0.0005 * 4)) * T * (1 - stats.norm.cdf(s0, mean_spread, std_spread))
    x = np.linspace(0, 0.05, 100)
    f_values = [f(s0) for s0 in x]
    s0 = x[np.argmin(f_values)]
    lower_bound = -(s0 - mean_spread) / std_spread
    upper_bound = (s0 - mean_spread) / std_spread
    return lower_bound, upper_bound, s0

def generate_signal(z_spread, threshold_long, threshold_short):
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

# Precompute hedge ratios and intercepts

# precomputed_values = []
# for t in tqdm.tqdm(range(int(len(train_pair_data)))):
#     pair_data_t = train_pair_data[t]
#     for pair_name, pair_price in pair_data_t.items():
#         common_indices = pair_price[pair_name[0]].index.intersection(pair_price[pair_name[1]].index)
#         log_price_1 = np.log(pair_price[pair_name[0]].loc[common_indices]['Close'])
#         log_price_2 = np.log(pair_price[pair_name[1]].loc[common_indices]['Close'])
        
#         split_index = int(len(log_price_1) * 0.8)
#         init_hedge_ratio, init_intercept = np.polyfit(log_price_2[:split_index], log_price_1[:split_index], 1)
#         spread = log_price_1 - (init_hedge_ratio * log_price_2 + init_intercept)
#         train_spread = spread[:split_index]
#         mean_spread = train_spread.mean()
#         std_spread = train_spread.std()
#         z_spread = (spread - mean_spread) / std_spread
#         T = len(spread)
#         lower_bound, upper_bound, s0 = calculate_optimal_thresholds(mean_spread, std_spread, T)
#         signal = generate_signal(z_spread, lower_bound, upper_bound)

#         precomputed_values.append((spread, z_spread, split_index, signal, lower_bound, upper_bound))

# with open('precomputed_values.pkl', 'wb') as f:
#     pickle.dump(precomputed_values, f)

# Function to generate X and y with different thresholds
def cost_function(precomputed_values, threshold, risk_reward_ratio, stop_loss):
    print(f'Generating X and y with threshold: {threshold}, risk-reward: {risk_reward_ratio}, stop loss: {stop_loss}')
    profit = 0
    cnt_win = 1
    cnt_loss = 1
    for spread, z_spread, split_index, signal, lower_bound, upper_bound in tqdm.tqdm(precomputed_values):
        if np.abs(z_spread.iloc[split_index]) > threshold:
            z_spread_i = z_spread.iloc[split_index]
            signal_i = signal.iloc[split_index]
            spread_i = spread.iloc[split_index]
            observe_z_spread = z_spread.iloc[split_index:]

            for i in range(1, len(observe_z_spread)): # check if it generates 2:1 win-loss ratio 
                if signal_i == 1 and observe_z_spread.iloc[i] < z_spread_i-(stop_loss/np.sqrt(np.abs(z_spread_i))):
                    cnt_loss += 1
                    break
                elif signal_i == -1 and observe_z_spread.iloc[i] > z_spread_i+(stop_loss/np.sqrt(np.abs(z_spread_i))):
                    cnt_loss += 1
                    break
                elif signal_i == 1 and observe_z_spread.iloc[i] > z_spread_i+(risk_reward_ratio*stop_loss/np.sqrt(np.abs(z_spread_i))):
                    cnt_win += 1
                    break
                elif signal_i == -1 and observe_z_spread.iloc[i] < z_spread_i-(risk_reward_ratio*stop_loss/np.sqrt(np.abs(z_spread_i))):
                    cnt_win += 1
                    break

            if i == len(observe_z_spread)-1:
                cnt_loss += 1

    print(f'Win: {cnt_win}, Loss: {cnt_loss}')
    print(cnt_win*risk_reward_ratio - cnt_loss)
    return cnt_win*risk_reward_ratio - cnt_loss

def objective(params, precomputed_values):
    threshold, win_z, loss_z = params
    return -cost_function(precomputed_values, threshold, win_z, loss_z)

# Load precomputed values
with open('precomputed_values.pkl', 'rb') as f:
    precomputed_values = pickle.load(f)

# Initial guess for threshold, win_z, and loss_z
initial_guess = [1, 1, 1]

# Bounds for threshold, win_z, and loss_z
bounds = [(0.1, 4), (0.5, 3), (0.5, 4.0)]

# Optimize
result = minimize(objective, initial_guess, args=(precomputed_values,), bounds=bounds, method='Nelder-Mead')

# Extract the best parameters
best_threshold, best_win_z, best_loss_z = result.x
max_profit = -result.fun

print(f'Best parameters: threshold={best_threshold}, win_z={best_win_z}, loss_z={best_loss_z} with profit: {max_profit}')