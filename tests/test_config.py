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
        assert config.testnet is False
        assert config.rate_limit == 0.5
        assert config.max_retries == 3
    
    def test_custom_values(self):
        config = ExchangeConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        assert config.api_key == "test_key"
        assert config.testnet is True


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_values(self):
        config = DataConfig()
        assert "1h" in config.intervals
        assert config.quote_currency == "USDT"
        assert config.n_coins == 50
    
    def test_custom_values(self):
        config = DataConfig(
            cache_dir="/tmp/test_cache",
            intervals=["5m", "15m"]
        )
        assert config.cache_dir == "/tmp/test_cache"
        assert "5m" in config.intervals


class TestStrategyConfig:
    """Tests for StrategyConfig."""
    
    def test_default_thresholds(self):
        config = StrategyConfig()
        assert config.entry_zscore == 2.0
        assert config.exit_zscore == 0.0  # Note: actual default
        assert config.stop_loss_zscore == 3.5
    
    def test_custom_values(self):
        config = StrategyConfig(
            correlation_threshold=0.7,
            cointegration_pvalue=0.01,
            entry_zscore=2.5
        )
        assert config.correlation_threshold == 0.7
        assert config.cointegration_pvalue == 0.01
        assert config.entry_zscore == 2.5


class TestRiskConfig:
    """Tests for RiskConfig."""
    
    def test_default_values(self):
        config = RiskConfig()
        assert config.max_position_size == 0.1
        assert config.max_drawdown == 0.2
        assert config.leverage == 1
    
    def test_custom_values(self):
        config = RiskConfig(
            max_position_size=0.2,
            max_drawdown=0.3,
            leverage=2
        )
        assert config.max_position_size == 0.2
        assert config.max_drawdown == 0.3
        assert config.leverage == 2


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_initialization(self):
        # Note: Config validates that entry > exit, so we need valid params
        config = Config(
            strategy=StrategyConfig(
                entry_zscore=2.0,
                exit_zscore=0.0,
                stop_loss_zscore=3.5
            )
        )
        assert config.mode == TradingMode.BACKTEST
        assert isinstance(config.exchange, ExchangeConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_to_dict(self):
        config = Config(
            strategy=StrategyConfig(
                entry_zscore=2.0,
                exit_zscore=0.0,
                stop_loss_zscore=3.5
            )
        )
        data = config.to_dict()
        assert 'mode' in data
        assert 'exchange' in data
        assert 'strategy' in data
    
    def test_validation_entry_exit(self):
        # Invalid: entry <= exit should raise
        with pytest.raises(ValueError):
            Config(
                strategy=StrategyConfig(
                    entry_zscore=1.0,
                    exit_zscore=2.0,  # exit > entry is invalid
                    stop_loss_zscore=3.5
                )
            )
    
    def test_validation_stop_loss(self):
        # Invalid: stop_loss <= entry should raise
        with pytest.raises(ValueError):
            Config(
                strategy=StrategyConfig(
                    entry_zscore=2.0,
                    exit_zscore=0.0,
                    stop_loss_zscore=1.5  # stop_loss < entry is invalid
                )
            )


class TestTradingMode:
    """Tests for TradingMode enum."""
    
    def test_modes(self):
        assert TradingMode.BACKTEST.value == "backtest"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.LIVE.value == "live"
