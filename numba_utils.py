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

def hurst(ts):
        
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.var(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # Use a linear fit to estimate the Hurst exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]/2.0