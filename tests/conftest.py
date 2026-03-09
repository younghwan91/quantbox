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


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """매 테스트마다 get_settings() 캐시를 초기화.

    @lru_cache 싱글턴이 테스트 간 환경변수 오염을 일으키는 것을 방지한다.
    """
    from quantbox.core.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_state_file(tmp_path):
    """Redirect scalper state file to tmp_path so tests never corrupt
    the production state (data/state/scalper_state.json).

    This is CRITICAL — without this, any test calling _sync_server_orders()
    or _save_state() would overwrite the live bot's position state.
    """
    import quantbox.scalper as _mod
    import quantbox.core.scalper_state as _state_mod

    orig_file = _mod.STATE_FILE
    orig_state_file = _state_mod.STATE_FILE
    orig_state_dir = _state_mod.STATE_DIR
    _state_mod.STATE_DIR = tmp_path
    _state_mod.STATE_FILE = tmp_path / "scalper_state.json"
    _mod.STATE_FILE = tmp_path / "scalper_state.json"
    yield
    _mod.STATE_FILE = orig_file
    _state_mod.STATE_FILE = orig_state_file
    _state_mod.STATE_DIR = orig_state_dir


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
    df = loader.load_symbol("BTCUSDT")

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
    data = loader.load_data(symbols=["BTCUSDT", "ETHUSDT"], min_bars=100, align_data=True)

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
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"], min_bars=100, align_data=True
    )

    if len(data) < 3:
        pytest.skip("Insufficient real data - need at least 3 symbols")

    return data


@pytest.fixture
def config():
    """Sample configuration."""
    from quantbox.core.config import Config

    return Config()
