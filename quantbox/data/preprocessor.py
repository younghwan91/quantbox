"""
Data preprocessing for pair trading.

Handles:
- Data alignment across pairs
- Feature engineering
- Spread calculation
- Statistical computations
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


logger = logging.getLogger('DataPreprocessor')


@dataclass
class PairData:
    """Container for processed pair data."""
    symbol1: str
    symbol2: str
    
    # Price data
    close1: pd.Series = None
    close2: pd.Series = None
    
    # Log prices
    log_close1: pd.Series = None
    log_close2: pd.Series = None
    
    # Spread data
    spread: pd.Series = None
    z_score: pd.Series = None
    
    # Hedge parameters
    hedge_ratio: float = 1.0
    intercept: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 1.0
    
    # Statistics
    half_life: float = 0.0
    correlation: float = 0.0
    cointegration_pvalue: float = 1.0
    
    # Training/Testing split
    split_index: int = 0
    
    # Additional features
    features: Dict[str, pd.Series] = field(default_factory=dict)


class DataPreprocessor:
    """
    Preprocessor for pair trading data.
    """
    
    def __init__(self, config=None):
        self.config = config
        
    def align_data(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        column: str = 'Close'
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Align two DataFrames to common timestamps.
        
        Args:
            data1: First DataFrame
            data2: Second DataFrame
            column: Column to extract
            
        Returns:
            Tuple of aligned Series
        """
        # Get common index
        common_idx = data1.index.intersection(data2.index)
        
        if len(common_idx) == 0:
            raise ValueError("No common timestamps found")
        
        series1 = data1.loc[common_idx, column]
        series2 = data2.loc[common_idx, column]
        
        return series1, series2
    
    def calculate_hedge_ratio(
        self,
        y: np.ndarray,
        x: np.ndarray,
        method: str = 'ols'
    ) -> Tuple[float, float]:
        """
        Calculate hedge ratio using specified method.
        
        Args:
            y: Dependent variable (log prices)
            x: Independent variable (log prices)
            method: 'ols' or 'tls' (total least squares)
            
        Returns:
            Tuple of (hedge_ratio, intercept)
        """
        if method == 'ols':
            # Simple OLS: y = intercept + beta * x
            coeffs = np.polyfit(x, y, 1)
            hedge_ratio = coeffs[0]
            intercept = coeffs[1]
        
        elif method == 'tls':
            # Total Least Squares
            hedge_ratio, intercept = self._total_least_squares(x, y)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return hedge_ratio, intercept
    
    def _total_least_squares(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate hedge ratio using Total Least Squares."""
        n = len(x)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Compute covariance matrix elements
        S_xx = np.sum((x - x_mean) ** 2)
        S_xy = np.sum((x - x_mean) * (y - y_mean))
        S_yy = np.sum((y - y_mean) ** 2)
        
        # Find eigenvalue/eigenvector
        delta = np.sqrt((S_xx - S_yy) ** 2 + 4 * S_xy ** 2)
        lambda2 = (S_xx + S_yy - delta) / 2
        
        if abs(S_xy) > 1e-12:
            v0 = 1.0
            v1 = -(S_xx - lambda2) / S_xy
        else:
            v0 = 0.0 if S_xx >= S_yy else 1.0
            v1 = 1.0 if S_xx >= S_yy else 0.0
        
        # Calculate slope and intercept
        if abs(v1) < 1e-12:
            hedge_ratio = np.inf
        else:
            hedge_ratio = -v0 / v1
        
        intercept = y_mean - hedge_ratio * x_mean
        
        return hedge_ratio, intercept
    
    def calculate_spread(
        self,
        log_price1: pd.Series,
        log_price2: pd.Series,
        hedge_ratio: float,
        intercept: float
    ) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Spread = log(P1) - (hedge_ratio * log(P2) + intercept)
        """
        spread = log_price1 - (hedge_ratio * log_price2 + intercept)
        return spread
    
    def calculate_z_score(
        self,
        spread: pd.Series,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate z-score of spread.
        
        Args:
            spread: Spread series
            mean: Pre-calculated mean (uses spread mean if None)
            std: Pre-calculated std (uses spread std if None)
            lookback: Rolling window for mean/std (if None, uses full series)
            
        Returns:
            Z-score series
        """
        if lookback:
            rolling_mean = spread.rolling(window=lookback).mean()
            rolling_std = spread.rolling(window=lookback).std()
            z_score = (spread - rolling_mean) / rolling_std
        else:
            mean = mean if mean is not None else spread.mean()
            std = std if std is not None else spread.std()
            z_score = (spread - mean) / std
        
        return z_score
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean-reversion half-life using AR(1) model.
        
        Half-life = -ln(2) / ln(1 + lambda)
        where lambda is the AR(1) coefficient
        """
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        
        # Remove NaN
        spread_lag = spread_lag.iloc[1:]
        spread_ret = spread_ret.iloc[1:]
        
        # OLS regression: spread_ret = alpha + lambda * spread_lag
        try:
            import statsmodels.api as sm
            X = sm.add_constant(spread_lag)
            model = sm.OLS(spread_ret, X).fit()
            lambda_param = model.params.iloc[1]
            
            if lambda_param >= 0:
                return np.inf  # Not mean-reverting
            
            half_life = -np.log(2) / np.log(1 + lambda_param)
            return max(0, half_life)
            
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.inf
    
    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[float, float]:
        """
        Test for cointegration between two series.
        
        Returns:
            Tuple of (t-statistic, p-value)
        """
        try:
            from statsmodels.tsa.stattools import coint
            t_stat, p_value, _ = coint(series1, series2)
            return t_stat, p_value
        except Exception as e:
            logger.warning(f"Error in cointegration test: {e}")
            return 0.0, 1.0
    
    def test_stationarity(self, series: pd.Series) -> Tuple[float, float]:
        """
        Test stationarity using ADF test.
        
        Returns:
            Tuple of (t-statistic, p-value)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna())
            return result[0], result[1]
        except Exception as e:
            logger.warning(f"Error in stationarity test: {e}")
            return 0.0, 1.0
    
    def calculate_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> float:
        """Calculate Pearson correlation."""
        return series1.corr(series2)
    
    def calculate_hurst_exponent(
        self,
        series: pd.Series,
        max_lag: int = 100
    ) -> float:
        """
        Calculate Hurst exponent.
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        lags = range(2, min(max_lag, len(series) // 2))
        
        tau = [
            np.std(np.subtract(series[lag:].values, series[:-lag].values))
            for lag in lags
        ]
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] / 2.0
    
    def process_pair(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        train_ratio: float = 0.8,
        hedge_method: str = 'ols'
    ) -> PairData:
        """
        Process a trading pair and calculate all statistics.
        
        Args:
            data1: OHLCV data for symbol1
            data2: OHLCV data for symbol2
            symbol1: First symbol name
            symbol2: Second symbol name
            train_ratio: Ratio of data for training
            hedge_method: Method for hedge ratio calculation
            
        Returns:
            PairData object with all computed values
        """
        # Align data
        close1, close2 = self.align_data(data1, data2, 'Close')
        
        # Calculate split index
        split_index = int(len(close1) * train_ratio)
        
        # Log prices
        log_close1 = np.log(close1)
        log_close2 = np.log(close2)
        
        # Training data
        train_log1 = log_close1.iloc[:split_index]
        train_log2 = log_close2.iloc[:split_index]
        
        # Calculate hedge ratio on training data
        hedge_ratio, intercept = self.calculate_hedge_ratio(
            train_log1.values,
            train_log2.values,
            method=hedge_method
        )
        
        # Calculate spread
        spread = self.calculate_spread(log_close1, log_close2, hedge_ratio, intercept)
        
        # Training spread statistics
        train_spread = spread.iloc[:split_index]
        spread_mean = train_spread.mean()
        spread_std = train_spread.std()
        
        # Z-score
        z_score = self.calculate_z_score(spread, spread_mean, spread_std)
        
        # Statistics on training data
        half_life = self.calculate_half_life(train_spread)
        correlation = self.calculate_correlation(train_log1, train_log2)
        _, coint_pvalue = self.test_cointegration(train_log1, train_log2)
        
        return PairData(
            symbol1=symbol1,
            symbol2=symbol2,
            close1=close1,
            close2=close2,
            log_close1=log_close1,
            log_close2=log_close2,
            spread=spread,
            z_score=z_score,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            correlation=correlation,
            cointegration_pvalue=coint_pvalue,
            split_index=split_index
        )
    
    def add_technical_features(
        self,
        pair_data: PairData,
        windows: List[int] = [7, 14, 21, 50, 200]
    ) -> PairData:
        """
        Add technical analysis features to pair data.
        """
        close1 = pair_data.close1
        close2 = pair_data.close2
        spread = pair_data.spread
        
        # Moving averages
        for w in windows:
            pair_data.features[f'ma_{w}_1'] = close1.rolling(w).mean()
            pair_data.features[f'ma_{w}_2'] = close2.rolling(w).mean()
            pair_data.features[f'spread_ma_{w}'] = spread.rolling(w).mean()
        
        # RSI
        pair_data.features['rsi_14_1'] = self._calculate_rsi(close1, 14)
        pair_data.features['rsi_14_2'] = self._calculate_rsi(close2, 14)
        
        # Volatility
        pair_data.features['vol_20_1'] = close1.pct_change().rolling(20).std()
        pair_data.features['vol_20_2'] = close2.pct_change().rolling(20).std()
        pair_data.features['spread_vol_20'] = spread.rolling(20).std()
        
        # Z-score derivative
        pair_data.features['z_score_diff'] = pair_data.z_score.diff()
        pair_data.features['z_score_ma_5'] = pair_data.z_score.rolling(5).mean()
        
        return pair_data
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_training_dataset(
        self,
        pairs_data: List[PairData],
        lookback: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset from multiple pairs.
        
        Returns:
            Tuple of (features, labels)
        """
        X_list = []
        y_list = []
        
        for pair_data in pairs_data:
            # Use training portion only
            z_score = pair_data.z_score.iloc[:pair_data.split_index]
            
            for i in range(lookback, len(z_score) - 1):
                # Features: last N z-scores
                features = z_score.iloc[i-lookback:i].values
                
                # Label: direction of next z-score change
                next_z = z_score.iloc[i + 1]
                current_z = z_score.iloc[i]
                
                # Binary label: mean reversion occurred
                if current_z > 0 and next_z < current_z:
                    label = 1  # Reversion down
                elif current_z < 0 and next_z > current_z:
                    label = 1  # Reversion up
                else:
                    label = 0  # No reversion
                
                X_list.append(features)
                y_list.append(label)
        
        return np.array(X_list), np.array(y_list)
