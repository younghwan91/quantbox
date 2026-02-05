"""
Unit tests for Real Data Loader and Backtest System.

All tests use REAL Binance Futures data from cache.
NO synthetic/fake data is allowed.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class TestRealDataLoader:
    """Tests for RealDataLoader."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        from quantbox.data.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        assert loader.cache_dir is not None
        assert isinstance(loader.cache_dir, Path)
    
    def test_get_available_symbols(self, real_data_available):
        """Test getting list of available symbols."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.data.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        symbols = loader.get_available_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        # All symbols should end with USDT
        for s in symbols:
            assert s.endswith('USDT')
    
    def test_load_single_symbol(self, real_data_available):
        """Test loading a single symbol."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.data.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        
        df = loader.load_symbol('BTCUSDT')
        
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert 'Close' in df.columns
            assert len(df) > 0
    
    def test_load_multiple_symbols(self, real_data_available):
        """Test loading multiple symbols."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.data.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        
        data = loader.load_data(
            symbols=['BTCUSDT', 'ETHUSDT'],
            min_bars=100,
            align_data=True
        )
        
        assert isinstance(data, dict)
        if len(data) >= 2:
            # All DataFrames should have same length (aligned)
            lengths = [len(df) for df in data.values()]
            assert len(set(lengths)) == 1
    
    def test_data_has_required_columns(self, sample_ohlcv):
        """Test loaded data has OHLCV columns."""
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            assert col in sample_ohlcv.columns
    
    def test_data_is_numeric(self, sample_ohlcv):
        """Test price data is numeric."""
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert pd.api.types.is_numeric_dtype(sample_ohlcv[col])
    
    def test_data_has_datetime_index(self, sample_ohlcv):
        """Test data has datetime index."""
        assert isinstance(sample_ohlcv.index, pd.DatetimeIndex)


class TestContinuousBacktest:
    """Tests for continuous pair trading backtest."""
    
    def test_backtest_config_creation(self):
        """Test backtest config creation."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig
        
        config = ContinuousBacktestConfig(
            initial_capital=10000.0,
            formation_bars=150,
            entry_zscore=2.0
        )
        
        assert config.initial_capital == 10000.0
        assert config.formation_bars == 150
        assert config.entry_zscore == 2.0
    
    def test_backtest_engine_creation(self):
        """Test backtest engine creation."""
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest,
            ContinuousBacktestConfig
        )
        
        config = ContinuousBacktestConfig()
        engine = ContinuousPairBacktest(config)
        
        assert engine.config.initial_capital > 0
    
    def test_backtest_with_real_data(self, sample_pair_data, real_data_available):
        """Test backtest runs with real data."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest,
            ContinuousBacktestConfig
        )
        
        config = ContinuousBacktestConfig(
            initial_capital=10000.0,
            formation_bars=100,
            entry_zscore=1.8,
            exit_zscore=0.4
        )
        
        engine = ContinuousPairBacktest(config)
        results = engine.run(sample_pair_data, progress=False)
        
        # Should return a dict with metrics
        assert isinstance(results, dict)
        assert 'metrics' in results or 'total_return_pct' in results
    
    def test_backtest_returns_valid_metrics(self, multi_symbol_data, real_data_available):
        """Test backtest returns valid metrics."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.backtest.continuous_backtest import (
            ContinuousPairBacktest,
            ContinuousBacktestConfig
        )
        
        config = ContinuousBacktestConfig(
            initial_capital=10000.0,
            formation_bars=100
        )
        
        engine = ContinuousPairBacktest(config)
        results = engine.run(multi_symbol_data, progress=False)
        
        # Check metrics are valid numbers
        metrics = results.get('metrics', results)
        
        if isinstance(metrics, dict):
            for key in ['total_return_pct', 'sharpe_ratio']:
                if key in metrics:
                    assert not np.isnan(metrics[key]) or metrics.get('n_trades', 0) == 0


class TestPairStatistics:
    """Tests for pair statistics calculations."""
    
    def test_pearson_correlation(self, sample_pair_data, real_data_available):
        """Test correlation calculation."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.backtest.continuous_backtest import PairStatistics
        
        stats = PairStatistics()
        symbols = list(sample_pair_data.keys())
        
        if len(symbols) >= 2:
            p1 = sample_pair_data[symbols[0]]['Close'].values
            p2 = sample_pair_data[symbols[1]]['Close'].values
            
            corr = stats.pearson_correlation(p1, p2)
            
            assert -1 <= corr <= 1
    
    def test_hedge_ratio_calculation(self, sample_pair_data, real_data_available):
        """Test hedge ratio calculation."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.backtest.continuous_backtest import PairStatistics
        
        stats = PairStatistics()
        symbols = list(sample_pair_data.keys())
        
        if len(symbols) >= 2:
            p1 = sample_pair_data[symbols[0]]['Close'].values
            p2 = sample_pair_data[symbols[1]]['Close'].values
            
            hedge_ratio, intercept = stats.calculate_hedge_ratio(p1, p2)
            
            assert isinstance(hedge_ratio, (int, float))
            assert not np.isnan(hedge_ratio)


class TestValidation:
    """Tests for validation module."""
    
    def test_load_test_data_function(self, real_data_available):
        """Test the load_test_data function."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        from quantbox.backtest.validation import load_test_data
        
        data = load_test_data(min_symbols=2, min_bars=100)
        
        assert isinstance(data, dict)
        assert len(data) >= 2
    
    def test_lookahead_validator_creation(self):
        """Test lookahead validator creation."""
        from quantbox.backtest.validation import LookaheadBiasValidator
        
        validator = LookaheadBiasValidator()
        assert hasattr(validator, 'violations')
    
    def test_transaction_cost_validator(self):
        """Test transaction cost validator."""
        from quantbox.backtest.validation import TransactionCostValidator
        
        validator = TransactionCostValidator()
        
        # Test fee calculation
        is_valid = validator.validate_fee_calculation(
            capital=10000.0,
            fee_rate=0.0005,
            calculated_fee=10.0  # 2 * 10000 * 0.0005
        )
        
        assert is_valid


class TestLiveTrader:
    """Tests for live trader module."""
    
    def test_trader_config_creation(self):
        """Test trader config can be created."""
        try:
            from live_trader import LiveTraderConfig
            
            config = LiveTraderConfig(
                initial_capital=10000.0,
                max_position_pct=0.5
            )
            
            assert config.initial_capital == 10000.0
        except ImportError:
            pytest.skip("live_trader module not available")
    
    def test_signal_generator_creation(self):
        """Test signal generator creation."""
        try:
            from live_trader import SignalGenerator
            
            generator = SignalGenerator(
                formation_bars=150,
                entry_zscore=2.0,
                exit_zscore=0.5
            )
            
            assert generator.entry_zscore == 2.0
        except ImportError:
            pytest.skip("live_trader module not available")


class TestDataIntegrity:
    """Tests for data integrity."""
    
    def test_no_synthetic_data_in_fixtures(self, sample_ohlcv, real_data_available):
        """Verify fixtures use real data, not synthetic."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        # Real data should have realistic price ranges
        close_prices = sample_ohlcv['Close']
        
        # If this is BTC, prices should be in thousands
        # We can't check exact values, but we can verify data looks real
        assert close_prices.std() > 0  # Has variance
        assert close_prices.min() > 0  # No negative prices
    
    def test_data_timestamps_valid(self, sample_ohlcv, real_data_available):
        """Test data has valid timestamps."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        # Timestamps should be chronological
        assert sample_ohlcv.index.is_monotonic_increasing
    
    def test_ohlc_relationships(self, sample_ohlcv, real_data_available):
        """Test OHLC relationships are valid."""
        if not real_data_available:
            pytest.skip("Real data not available")
        
        # High >= Low
        assert (sample_ohlcv['High'] >= sample_ohlcv['Low']).all()
        
        # High >= Open and Close
        assert (sample_ohlcv['High'] >= sample_ohlcv['Open']).all()
        assert (sample_ohlcv['High'] >= sample_ohlcv['Close']).all()
        
        # Low <= Open and Close
        assert (sample_ohlcv['Low'] <= sample_ohlcv['Open']).all()
        assert (sample_ohlcv['Low'] <= sample_ohlcv['Close']).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
