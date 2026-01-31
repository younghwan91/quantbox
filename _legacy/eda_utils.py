import numpy as np
# ta adx

def create_target_column(precomputed_values, stop_loss_threshold=1.0):
    for item in precomputed_values:
        z_spread = item['z_spread']
        split_index = item['split_index']
        
        # Entry z-score is last point of formation window
        if split_index <= 0 or split_index >= len(z_spread):
            item['position'] = 0
            item['trade'] = 0
            continue
        
        val_init = z_spread.iloc[split_index - 1]
        position = -1 * np.sign(val_init)
        trade = 0
        
        if position == 0:
            item['position'] = 0
            item['trade'] = 0
            continue

        # Evaluate over trading window (post-formation)
        for val in z_spread.iloc[split_index:]:
            diff = val - val_init

            # Stop-loss: moved further from mean
            if (position < 0 and diff > stop_loss_threshold) or \
               (position > 0 and diff < -stop_loss_threshold):
                trade = 0
                break

            # Take-profit: mean reversion toward zero
            if (position < 0 and val < 0) or (position > 0 and val > 0):
                trade = 1
                break

        item['position'] = position
        item['trade'] = trade
        item['rr_ratio'] = np.abs(val_init) / stop_loss_threshold

    return precomputed_values

def calculate_rsi(series, window=14):
    """
    Calculates the Relative Strength Index (RSI) for a given series.
    
    Args:
        series (pd.Series): The price series to calculate RSI for.
        window (int): The lookback period for RSI calculation.
    
    Returns:
        pd.Series: The RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_features(precomputed_values, ma_windows=[7, 15, 50, 200], rsi_window=14, adx_window=14):
    """
    Adds moving averages, RSI, ADX, percentage difference features, and VWAP percentage differences to precomputed_values.
    
    Args:
        precomputed_values (list): List of dictionaries containing 'close1', 'close2', 'vol1', 'vol2', 'high1', 'high2', 'low1', 'low2', and 'split_index'.
        ma_windows (list): List of moving average window sizes.
        rsi_window (int): The lookback period for RSI calculation.
        adx_window (int): The lookback period for ADX calculation.
    
    Returns:
        list: Updated precomputed_values with moving averages, RSI, ADX, percentage difference features, and VWAP percentage differences.
    """
    for item in precomputed_values:
        close1 = item['close1']
        close2 = item['close2']
        high1 = item['high1']
        high2 = item['high2']
        low1 = item['low1']
        low2 = item['low2']
        volume1 = item['vol1']
        volume2 = item['vol2']
        z_spread = item['z_spread']
        spread = item['spread']
        split_index = item['split_index']
        
        # Calculate moving averages for close1 and close2
        for window in ma_windows:
            item[f'close1_ma_{window}'] = close1.rolling(window=window).mean()
            item[f'close2_ma_{window}'] = close2.rolling(window=window).mean()
        
        # Calculate RSI for close1 and close2
        item['close1_rsi'] = calculate_rsi(close1, window=rsi_window)
        item['close2_rsi'] = calculate_rsi(close2, window=rsi_window)

        # Calculate percentage difference features and RSI at split_index-1
        split_idx = split_index - 1
        item['z_spread_split'] = z_spread.iloc[split_idx]
        item['var_spread'] = spread.iloc[:split_idx].var()

        for window in ma_windows:
            if split_idx >= window - 1:  # Ensure enough data for the moving average
                close1_ma = item[f'close1_ma_{window}'].iloc[split_idx]
                close2_ma = item[f'close2_ma_{window}'].iloc[split_idx]
                close1_price = close1.iloc[split_idx]
                close2_price = close2.iloc[split_idx]
                
                item[f'close1_ma_{window}_pct_diff'] = (close1_price - close1_ma) / close1_ma * 100
                item[f'close2_ma_{window}_pct_diff'] = (close2_price - close2_ma) / close2_ma * 100
            else:
                item[f'close1_ma_{window}_pct_diff'] = None
                item[f'close2_ma_{window}_pct_diff'] = None
        
        item['close1_rsi_split'] = item['close1_rsi'].iloc[split_idx]
        item['close2_rsi_split'] = item['close2_rsi'].iloc[split_idx]

        # Calculate VWAP percentage differences for the last 100 bars ending at split_idx
        window_vwap = 100
        if split_idx >= window_vwap - 1:
            start_idx = split_idx - window_vwap + 1
            vwap1 = (close1.iloc[start_idx:split_idx+1] * volume1.iloc[start_idx:split_idx+1]).sum() / volume1.iloc[start_idx:split_idx+1].sum()
            vwap2 = (close2.iloc[start_idx:split_idx+1] * volume2.iloc[start_idx:split_idx+1]).sum() / volume2.iloc[start_idx:split_idx+1].sum()
            close1_price = close1.iloc[split_idx]
            close2_price = close2.iloc[split_idx]
            item['close1_vwap_100_pct_diff'] = (close1_price - vwap1) / vwap1 * 100
            item['close2_vwap_100_pct_diff'] = (close2_price - vwap2) / vwap2 * 100
        else:
            item['close1_vwap_100_pct_diff'] = None
            item['close2_vwap_100_pct_diff'] = None
        
        # Calculate z-score slope before entry
        z_slope_window = 10
        if split_index >= z_slope_window:
            y = z_spread.iloc[split_index - z_slope_window:split_index].values
            x = np.arange(z_slope_window)
            slope = np.polyfit(x, y, 1)[0]
            item['z_slope_pre_split'] = slope
        else:
            item['z_slope_pre_split'] = None
        
        z_vol_window = 20
        if split_index >= z_vol_window:
            item['z_volatility'] = z_spread.iloc[split_index - z_vol_window:split_index].std()
        else:
            item['z_volatility'] = None

        z_mean_window = 20
        if split_index >= z_mean_window:
            z_mean = z_spread.iloc[split_index - z_mean_window:split_index].mean()
            item['z_spread_deviation'] = item['z_spread_split'] - z_mean
        else:
            item['z_spread_deviation'] = None

    return precomputed_values