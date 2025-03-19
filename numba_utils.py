# numba_utils.py
import numpy as np
from numba import njit

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
def cointegration_adf_pvalue(logP1, logP2, beta):
    """
    Perform a simplified ADF test on the spread defined as:
        z_t = logP1 + beta * logP2
    via the regression:
        Δz_t = α + γ * z_{t-1} + error.
    
    The function computes the t-statistic for the lagged level (γ)
    and then returns an approximate p-value by linearly interpolating
    between typical critical values for the ADF test.
    
    Parameters
    ----------
    logP1 : numpy.ndarray
        Log-prices for asset 1.
    logP2 : numpy.ndarray
        Log-prices for asset 2.
    beta : float
        The hedge ratio.
    
    Returns
    -------
    p_val : float
        An approximate p-value for the ADF test.
    
    Note
    ----
    Typical critical values (for a model with an intercept, no trend) are:
      - at 1%: -3.43
      - at 5%: -2.86
      - at 10%: -2.57
    Here we define a simple interpolation using the following points:
      t_stat = -4.00  -> p = 0.0
      t_stat = -3.43  -> p = 0.01
      t_stat = -2.86  -> p = 0.05
      t_stat = -2.57  -> p = 0.10
      t_stat =  0.00  -> p = 1.0
    Values outside the range are pegged at 0.0 or 1.0.
    """
    n = logP1.shape[0]
    
    # Compute the spread: z_t = logP1 + beta * logP2
    z = np.empty(n)
    for i in range(n):
        z[i] = logP1[i] + beta * logP2[i]
    
    # Set up the regression for t = 1, ..., n-1:
    # Δz[t] = z[t+1] - z[t] and regressor is [1, z[t]]
    m = n - 1
    y = np.empty(m)
    X = np.empty((m, 2))
    for t in range(m):
        y[t] = z[t+1] - z[t]
        X[t, 0] = 1.0       # constant
        X[t, 1] = z[t]      # lagged level
    
    # OLS: b = (X'X)^{-1} X'y
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, y)
    b = np.linalg.solve(XtX, XtY)
    
    # Residuals and variance estimate
    res = y - np.dot(X, b)
    dof = m - 2
    sigma2 = 0.0
    for i in range(m):
        sigma2 += res[i] * res[i]
    sigma2 /= dof
    
    # Manually compute the (1,1) element of (X'X)^{-1}
    a = XtX[0, 0]
    b_val = XtX[0, 1]
    c_val = XtX[1, 0]
    d = XtX[1, 1]
    det = a * d - b_val * c_val
    inv_11 = a / det  # corresponds to the variance of gamma's estimator
    
    # Standard error and t-statistic for gamma (coefficient on z_{t-1})
    se_gamma = np.sqrt(sigma2 * inv_11)
    t_stat = b[1] / se_gamma
    
    # --- Approximate p-value from the t_stat ---
    # Define interpolation points:
    xp = np.array([-4.0, -3.43, -2.86, -2.57, 0.0])
    fp = np.array([0.0, 0.01, 0.05, 0.10, 1.0])
    p_val = np.interp(t_stat, xp, fp)
    
    return p_val


@njit
def ols_regression(x, y):
    """
    단순 OLS 회귀: y = intercept + beta*x 를 계산합니다.
    """
    n = x.shape[0]
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n
    
    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        cov_xy += (x[i] - mean_x) * (y[i] - mean_y)
        var_x += (x[i] - mean_x) ** 2
    beta = cov_xy / var_x
    intercept = mean_y - beta * mean_x
    return intercept, beta


@njit
def cointegration_test(x, y, intercept, beta):
    """
    Engle–Granger 코인티그레이션 테스트의 간단한 구현 예제입니다.
    
    1. y와 x에 대해 OLS 회귀를 수행하여 잔차(residuals)를 계산합니다.
    2. 잔차에 대해, lag 1의 단순 ADF 테스트(차분 Δe_t = gamma * e_(t-1) + error)를 수행하여 
       gamma의 t‑statistic을 계산합니다.
       
    반환값:
      - t_stat: ADF 회귀에서의 t‑statistic (잔차의 단위근 검정 통계량)
      - intercept, beta: cointegrating 회귀의 계수
    """
    # Step 1: OLS 회귀로 잔차 계산
    # intercept, beta = ols_regression(x, y)
    n = x.shape[0]
    residuals = np.empty(n)
    for i in range(n):
        residuals[i] = y[i] - (intercept + beta * x[i])
    
    # Step 2: 잔차에 대한 단순 ADF 테스트 (지연 1)
    T = n - 1
    delta = np.empty(T)
    lagged = np.empty(T)
    for t in range(1, n):
        delta[t - 1] = residuals[t] - residuals[t - 1]
        lagged[t - 1] = residuals[t - 1]
    
    # delta = gamma * lagged + error 의 OLS 추정
    num = 0.0
    den = 0.0
    for t in range(T):
        num += lagged[t] * delta[t]
        den += lagged[t] * lagged[t]
    gamma = num / den
    
    # OLS 잔차(error) 계산
    error = np.empty(T)
    for t in range(T):
        error[t] = delta[t] - gamma * lagged[t]
    
    # gamma의 표준오차(standard error) 계산
    err_sq_sum = 0.0
    for t in range(T):
        err_sq_sum += error[t] ** 2
    sigma2 = err_sq_sum / (T - 1)
    se = np.sqrt(sigma2 / den)
    
    # t-statistic 계산
    t_stat = gamma / se
    
    return t_stat

@njit
def jarque_bera_numba(x):
    n = x.shape[0]
    # Compute the mean
    mean = np.mean(x)
    # Center the data
    diff = x - mean
    # Compute second, third, and fourth central moments
    m2 = np.mean(diff ** 2)
    m3 = np.mean(diff ** 3)
    m4 = np.mean(diff ** 4)
    
    # Calculate skewness and kurtosis
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    
    # Jarque-Bera test statistic: JB = n/6 * [skew^2 + (1/4)*(kurt - 3)^2]
    jb = n / 6.0 * (skew ** 2 + 0.25 * (kurt - 3) ** 2)
    
    # For a chi-square distribution with 2 degrees of freedom,
    # the survival function is exp(-jb/2) so p-value = exp(-jb/2)
    p_value = np.exp(-jb / 2.0)
    
    return p_value

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
