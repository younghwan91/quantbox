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
    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        returns[i] = np.log(prices[i + 1]) - np.log(prices[i])
    return returns

@njit
def normal_cdf(z):
    """
    Numba-friendly approximation to standard normal CDF 
    using the error function.
    """
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

@njit
def conditional_forecast(A, B, rho):
    """
    For bivariate normal assumption:
       A_next ~ Normal(mu_A + rho*(sigma_A/sigma_B)*(B[-1]-mu_B), 
                       sigma_A^2*(1 - rho^2))
    """
    n = A.shape[0]
    mu_A = np.sum(A) / n
    mu_B = np.sum(B) / n

    # sample standard deviations
    sigma_A = np.sqrt(np.sum((A - mu_A)**2) / (n - 1))
    sigma_B = np.sqrt(np.sum((B - mu_B)**2) / (n - 1))

    # mean
    mu_cond = mu_A + rho * (sigma_A / sigma_B) * (B[-1] - mu_B)
    # variance
    sigma_cond2 = sigma_A**2 * (1.0 - rho**2)

    return mu_cond, sigma_cond2

@njit
def filter_pair_numba(data1, data2, corr_threshold, window_ratio):
    """
    Fully Numba-jitted function. Returns True if the pair passes
    the filter; otherwise False.

    data1, data2 must be float64 arrays (prices).
    corr_threshold: e.g. 0.8
    window_ratio: e.g. 0.8 (80% of data as 'train')
    """
    n = data1.shape[0]
    if n < 2:
        return False

    split_idx = int(window_ratio * n)
    if split_idx < 2:
        return False

    # 1) Check positivity in the training set
    for i in range(split_idx):
        if data1[i] <= 0.0:
            return False
        if data2[i] <= 0.0:
            return False

    # 2) Log returns
    log_returns1 = log_returns_numba(data1[:split_idx])
    log_returns2 = log_returns_numba(data2[:split_idx])

    # 3) Correlation
    corr = correlation_numba(log_returns1, log_returns2)
    if corr < corr_threshold:
        return False

    # 4) Conditional forecast
    mu_cond, sigma_cond2 = conditional_forecast(log_returns1, log_returns2, corr)
    if sigma_cond2 <= 1e-14:
        return False

    # 5) Probability thresholds
    current_return = log_returns1[-1]
    threshold_long = current_return + 0.02
    threshold_short = current_return - 0.02

    z_long = (threshold_long - mu_cond) / np.sqrt(sigma_cond2)
    z_short = (threshold_short - mu_cond) / np.sqrt(sigma_cond2)

    prob_long = 1.0 - normal_cdf(z_long)   # P(X >= threshold_long)
    prob_short = normal_cdf(z_short)       # P(X <= threshold_short)

    go_long = (prob_long >= 0.60)
    go_short = (prob_short >= 0.60)

    if (not go_long) and (not go_short):
        return False

    return True

# -------------------------------------------------------------------------
# Test / demonstration
if __name__ == "__main__":
    # Generate some fake data
    prices1 = np.linspace(10.0, 12.0, 100).astype(np.float64)
    prices2 = np.linspace(5.0, 5.5, 100).astype(np.float64)

    # Example call
    passed = filter_pair_numba(prices1, prices2, corr_threshold=0.8, window_ratio=0.8)
    print("Pair passed filter:", passed)
