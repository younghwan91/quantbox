import numpy as np
import pandas as pd

def _calculate_hurst_exponent(time_series):
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:].to_numpy(), time_series[:-lag].to_numpy())) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# Example usage
prices = pd.Series([100, 102, 101, 104, 103, 105, 104, 106, 105])
hurst_exponent = _calculate_hurst_exponent(prices)
print(f'Hurst exponent: {hurst_exponent}')