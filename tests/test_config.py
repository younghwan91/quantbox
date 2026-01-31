"""
Tests for core configuration module.
"""

import pytest
import tempfile
import os

from quantbox.core.config import (
    Config, ExchangeConfig, DataConfig, StrategyConfig,
    RiskConfig, BacktestConfig, OptimizationConfig, TradingMode
)


class TestExchangeConfig:
    """Tests for ExchangeConfig."""
    
    def test_default_values(self):
        config = ExchangeConfig()
        assert config.name == "binance"
        assert config.testnet is True
        assert config.rate_limit == 1200
    
    def test_custom_values(self):
        config = ExchangeConfig(
            name="bybit",
            api_key="test_key",
            api_secret="test_secret",
            testnet=False
        )
        assert config.name == "bybit"
        assert config.api_key == "test_key"
        assert config.testnet is False


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_timeframes(self):
        config = DataConfig()
        assert "1h" in config.timeframes
        assert "4h" in config.timeframes
    
    def test_cache_settings(self):
        config = DataConfig(
            cache_enabled=True,
            cache_dir="/tmp/test_cache"
        )
        assert config.cache_enabled is True
        assert config.cache_dir == "/tmp/test_cache"


class TestStrategyConfig:
    """Tests for StrategyConfig."""
    
    def test_default_thresholds(self):
        config = StrategyConfig()
        assert config.entry_zscore == 2.0
        assert config.exit_zscore == 0.5
        assert config.stop_loss_zscore == 3.5
    
    def test_filter_settings(self):
        config = StrategyConfig(
            min_correlation=0.7,
            max_cointegration_pvalue=0.01
        )
        assert config.min_correlation == 0.7
        assert config.max_cointegration_pvalue == 0.01


class TestRiskConfig:
    """Tests for RiskConfig."""
    
    def test_position_limits(self):
        config = RiskConfig(
            max_position_size=0.2,
            max_portfolio_risk=0.1
        )
        assert config.max_position_size == 0.2
        assert config.max_portfolio_risk == 0.1
    
    def test_risk_constraints(self):
        config = RiskConfig(
            max_daily_loss=500.0,
            max_drawdown=0.2
        )
        assert config.max_daily_loss == 500.0
        assert config.max_drawdown == 0.2


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_initialization(self):
        config = Config()
        assert config.mode == TradingMode.BACKTEST
        assert isinstance(config.exchange, ExchangeConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_from_dict(self):
        data = {
            'mode': 'live',
            'exchange': {'name': 'bybit', 'testnet': False},
            'strategy': {'entry_zscore': 2.5}
        }
        config = Config.from_dict(data)
        assert config.mode == TradingMode.LIVE
        assert config.exchange.name == 'bybit'
        assert config.strategy.entry_zscore == 2.5
    
    def test_to_dict(self):
        config = Config()
        data = config.to_dict()
        assert 'mode' in data
        assert 'exchange' in data
        assert 'strategy' in data
    
    def test_validate(self):
        config = Config()
        # Should not raise for valid config
        config.validate()
        
        # Invalid config should raise
        config.strategy.entry_zscore = -1
        with pytest.raises(ValueError):
            config.validate()
    
    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = Config()
            config.strategy.entry_zscore = 3.0
            config.save(temp_path)
            
            loaded = Config.load(temp_path)
            assert loaded.strategy.entry_zscore == 3.0
        finally:
            os.unlink(temp_path)


class TestTradingMode:
    """Tests for TradingMode enum."""
    
    def test_modes(self):
        assert TradingMode.BACKTEST.value == "backtest"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.LIVE.value == "live"
