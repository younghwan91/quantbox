"""
Test fixtures and shared utilities for tests.

IMPORTANT: All test data comes from REAL Binance Futures data cached locally.
NO synthetic/fake data generation is allowed.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def _get_real_data_loader():
    """Get the real data loader instance."""
    from quantbox.data.real_data_loader import RealDataLoader
    return RealDataLoader()


@pytest.fixture(scope="session")
def real_data_available():
    """Check if real data is available in cache."""
    loader = _get_real_data_loader()
    symbols = loader.get_available_symbols()
    return len(symbols) >= 2


@pytest.fixture
def sample_ohlcv(real_data_available):
    """Load REAL OHLCV data from Binance cache."""
    if not real_data_available:
        pytest.skip("Real data not available - run data downloader first")
    
    loader = _get_real_data_loader()
    df = loader.load_symbol('BTCUSDT')
    
    if df is None:
        # Try any available symbol
        symbols = loader.get_available_symbols()
        if symbols:
            df = loader.load_symbol(symbols[0])
    
    if df is None:
        pytest.skip("No real data available - run data downloader first")
    
    return df


@pytest.fixture
def sample_pair_data(real_data_available):
    """Load REAL correlated pair data from Binance cache."""
    if not real_data_available:
        pytest.skip("Real data not available - run data downloader first")
    
    loader = _get_real_data_loader()
    data = loader.load_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        min_bars=100,
        align_data=True
    )
    
    if len(data) < 2:
        pytest.skip("Insufficient real data - need BTCUSDT and ETHUSDT")
    
    return data


@pytest.fixture
def multi_symbol_data(real_data_available):
    """Load REAL multi-symbol data from Binance cache."""
    if not real_data_available:
        pytest.skip("Real data not available - run data downloader first")
    
    loader = _get_real_data_loader()
    data = loader.load_data(
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'],
        min_bars=100,
        align_data=True
    )
    
    if len(data) < 3:
        pytest.skip("Insufficient real data - need at least 3 symbols")
    
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
