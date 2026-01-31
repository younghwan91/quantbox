"""
Test fixtures and shared utilities for tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='h')
    
    # Generate random walk
    price = 100
    prices = []
    for _ in range(n):
        price *= np.exp(np.random.randn() * 0.02)
        prices.append(price)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    return df


@pytest.fixture
def sample_pair_data():
    """Generate correlated pair data for testing."""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='h')
    
    # Generate correlated prices
    common_factor = np.cumsum(np.random.randn(n) * 0.01)
    noise1 = np.cumsum(np.random.randn(n) * 0.005)
    noise2 = np.cumsum(np.random.randn(n) * 0.005)
    
    price1 = 100 * np.exp(common_factor + noise1)
    price2 = 50 * np.exp(common_factor * 1.2 + noise2)
    
    df1 = pd.DataFrame({
        'Open': price1,
        'High': price1 * 1.01,
        'Low': price1 * 0.99,
        'Close': price1,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    df2 = pd.DataFrame({
        'Open': price2,
        'High': price2 * 1.01,
        'Low': price2 * 0.99,
        'Close': price2,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    return {'BTCUSDT': df1, 'ETHUSDT': df2}


@pytest.fixture
def multi_symbol_data(sample_pair_data):
    """Generate multi-symbol data."""
    np.random.seed(43)
    n = len(sample_pair_data['BTCUSDT'])
    dates = sample_pair_data['BTCUSDT'].index
    
    data = sample_pair_data.copy()
    
    # Add more symbols
    for symbol, base_price in [('BNBUSDT', 300), ('SOLUSDT', 25), ('ADAUSDT', 0.5)]:
        price = base_price * np.exp(np.cumsum(np.random.randn(n) * 0.015))
        
        data[symbol] = pd.DataFrame({
            'Open': price,
            'High': price * 1.01,
            'Low': price * 0.99,
            'Close': price,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
    
    return data


@pytest.fixture
def config():
    """Sample configuration."""
    from quantbox.core.config import Config
    return Config()


@pytest.fixture
def backtest_config():
    """Sample backtest configuration."""
    from quantbox.backtest.runner import BacktestConfig
    return BacktestConfig(
        initial_capital=10000.0,
        trading_fee=0.0005,
        slippage=0.001,
        formation_periods=100,
        trading_periods=50,
        window_shift=25
    )
