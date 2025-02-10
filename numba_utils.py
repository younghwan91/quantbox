# numba_utils.py
import numpy as np
from numba import njit
from math import erf

@njit
def correlation_numba(x, y):
    n = len(x)
    x_mean = np.sum(x) / n
    y_mean = np.sum(y) / n
    numerator = 0.0
    denominator_x = 0.0
    denominator_y = 0.0

    for i in range(n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator_x += (x[i] - x_mean)**2
        denominator_y += (y[i] - y_mean)**2

    return numerator / np.sqrt(denominator_x * denominator_y)


@njit
def log_returns_numba(prices):
    n = len(prices)
    returns = np.empty(n - 1)
    for i in range(n - 1):
        returns[i] = np.log(prices[i + 1]) - np.log(prices[i])
    return returns

@njit
def normalized_price_distance(price1, price2):
    
    t = len(price1)

    log_price1 = np.log(price1)
    log_price2 = np.log(price2)
    
    norm_log_price1 = log_price1 - log_price1[0]
    norm_log_price2 = log_price2 - log_price2[0]
    
    squared_diff = (norm_log_price1 - norm_log_price2) ** 2
    
    return np.sum(squared_diff) / t

@njit
def compute_slope(x, y):
    n = len(x)
    
    # Calculate means of x and y
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    # Compute covariance between x and y and variance of x
    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        diff_x = x[i] - mean_x
        diff_y = y[i] - mean_y
        cov_xy += diff_x * diff_y
        var_x += diff_x * diff_x

    # Avoid division by zero
    if var_x == 0:
        return np.nan  # You could also raise an error if preferred

    return cov_xy / var_x

@njit
def linear_regression(x, y):

    n = len(x)
    if n == 0:
        return np.nan, np.nan

    # Compute means of x and y.
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    # Compute covariance between x and y and variance of x.
    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        diff_x = x[i] - mean_x
        diff_y = y[i] - mean_y
        cov_xy += diff_x * diff_y
        var_x += diff_x * diff_x

    # Avoid division by zero.
    if var_x == 0:
        return np.nan, np.nan

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    return intercept, slope

@njit
def r_squared(x, y, intercept, slope):

    n = len(y)
    if n == 0:
        return np.nan

    # Compute the mean of y.
    mean_y = 0.0
    for i in range(n):
        mean_y += y[i]
    mean_y /= n

    # Compute total sum of squares and residual sum of squares.
    ss_tot = 0.0
    ss_res = 0.0
    for i in range(n):
        y_pred = intercept + slope * x[i]
        diff = y[i] - y_pred
        ss_res += diff * diff
        diff_mean = y[i] - mean_y
        ss_tot += diff_mean * diff_mean

    # If there's no variance in y, we treat it as a perfect fit.
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - ss_res / ss_tot



@njit
def filter_pair_numba(data1, data2, corr_threshold, window_ratio):
    """
    Minimal example of a 'filter' function in Numba.
    """
    n = len(data1)
    if n < 2:
        return False

    split_idx = int(window_ratio * n)
    if split_idx < 2:
        return False

    train_data1 = data1[:split_idx]
    train_data2 = data2[:split_idx]

    # 1) Positivity check
    for i in range(len(train_data1)):
        if train_data1[i] <= 0:
            return False
    for i in range(len(train_data2)):
        if train_data2[i] <= 0:
            return False

    # 2) Log returns
    log_returns1 = log_returns_numba(train_data1)
    log_returns2 = log_returns_numba(train_data2)

    # 3) Correlation
    corr = correlation_numba(log_returns1, log_returns2)
    if corr < 0.8 :
        return False
    
    # beta = compute_slope(log_returns2, log_returns1)
    # spread = log_returns1 - beta * log_returns2
    # t = np.linspace(0,1,len(spread))
    # intercept, slope = linear_regression(t, spread)
    # r2 = r_squared(t, spread, intercept, slope)
    # if r2 < 0.5:
    #     return False

    return True
