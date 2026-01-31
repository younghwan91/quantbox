"""
Configuration management for QuantBox trading framework.

Provides centralized configuration handling with validation,
environment-based settings, and type safety.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import os
from pathlib import Path


class TradingMode(Enum):
    """Trading execution mode."""
    BACKTEST = "backtest"
    PAPER = "paper"  
    LIVE = "live"


class Interval(Enum):
    """Supported trading intervals."""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    D1 = "1d"
    
    @property
    def minutes(self) -> int:
        """Convert interval to minutes."""
        mapping = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "1d": 1440
        }
        return mapping[self.value]
    
    @property
    def milliseconds(self) -> int:
        """Convert interval to milliseconds."""
        return self.minutes * 60 * 1000


@dataclass
class ExchangeConfig:
    """Exchange connection configuration."""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    rate_limit: float = 0.5  # seconds between requests
    max_retries: int = 3
    retry_delay: float = 2.0
    
    @classmethod
    def from_file(cls, filepath: str) -> "ExchangeConfig":
        """Load exchange config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            api_key=data.get('api_key', ''),
            api_secret=data.get('api_secret', ''),
            testnet=data.get('testnet', False)
        )


@dataclass
class DataConfig:
    """Data handling configuration."""
    cache_dir: str = "data/cache"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    quote_currency: str = "USDT"
    n_coins: int = 50
    intervals: List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    cache_expiry_hours: int = 24
    enable_cache: bool = True


@dataclass
class StrategyConfig:
    """Strategy-specific configuration."""
    # Pair selection parameters
    correlation_threshold: float = 0.5
    cointegration_pvalue: float = 0.05
    normality_pvalue: float = 0.05
    min_half_life: float = 1.0
    max_half_life: float = 100.0
    hedge_ratio_bounds: tuple = (0.25, 4.0)
    
    # Entry/Exit parameters  
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    stop_loss_zscore: float = 3.5
    max_holding_periods: int = 100
    
    # Formation/Trading window
    formation_periods: int = 240  # in bars
    trading_periods: int = 24  # in bars


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.1  # 10% of portfolio
    max_drawdown: float = 0.2  # 20% max drawdown
    leverage: int = 1
    trading_fee: float = 0.0005  # 0.05%
    slippage: float = 0.001  # 0.1%
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.10  # 10%


@dataclass
class BacktestConfig:
    """Backtesting-specific configuration."""
    initial_capital: float = 10000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    window_shift_hours: int = 24
    train_test_split: float = 0.8
    walk_forward: bool = True
    monte_carlo_runs: int = 1000


@dataclass 
class OptimizationConfig:
    """Optimization parameters configuration."""
    method: str = "differential_evolution"  # or "grid_search", "bayesian"
    max_iterations: int = 100
    population_size: int = 50
    tolerance: float = 0.001
    n_jobs: int = -1  # Use all CPUs
    
    # Parameter bounds for optimization
    param_bounds: Dict[str, tuple] = field(default_factory=lambda: {
        "entry_zscore": (1.5, 3.5),
        "stop_loss_zscore": (2.0, 5.0),
        "exit_zscore": (-0.5, 0.5),
        "max_holding_periods": (10, 200),
    })


@dataclass
class Config:
    """
    Master configuration class that aggregates all config components.
    
    Usage:
        config = Config.from_file("config.json")
        # or
        config = Config(
            mode=TradingMode.BACKTEST,
            exchange=ExchangeConfig(...),
            ...
        )
    """
    mode: TradingMode = TradingMode.BACKTEST
    interval: str = "15m"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "quantbox.log"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._create_directories()
    
    def _validate(self):
        """Validate configuration values."""
        if self.strategy.entry_zscore <= self.strategy.exit_zscore:
            raise ValueError("entry_zscore must be greater than exit_zscore")
        
        if self.strategy.stop_loss_zscore <= self.strategy.entry_zscore:
            raise ValueError("stop_loss_zscore must be greater than entry_zscore")
        
        if not 0 < self.backtest.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        if self.risk.max_position_size <= 0 or self.risk.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data.cache_dir, self.data.raw_dir, self.data.processed_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, filepath: str) -> "Config":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            mode=TradingMode(data.get('mode', 'backtest')),
            interval=data.get('interval', '15m'),
            exchange=ExchangeConfig(**data.get('exchange', {})),
            data=DataConfig(**data.get('data', {})),
            strategy=StrategyConfig(**data.get('strategy', {})),
            risk=RiskConfig(**data.get('risk', {})),
            backtest=BacktestConfig(**data.get('backtest', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            log_level=data.get('log_level', 'INFO'),
            log_file=data.get('log_file', 'quantbox.log'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mode': self.mode.value,
            'interval': self.interval,
            'exchange': {
                'testnet': self.exchange.testnet,
                'rate_limit': self.exchange.rate_limit,
            },
            'data': {
                'cache_dir': self.data.cache_dir,
                'quote_currency': self.data.quote_currency,
                'n_coins': self.data.n_coins,
                'intervals': self.data.intervals,
            },
            'strategy': {
                'correlation_threshold': self.strategy.correlation_threshold,
                'cointegration_pvalue': self.strategy.cointegration_pvalue,
                'entry_zscore': self.strategy.entry_zscore,
                'exit_zscore': self.strategy.exit_zscore,
                'stop_loss_zscore': self.strategy.stop_loss_zscore,
                'formation_periods': self.strategy.formation_periods,
                'trading_periods': self.strategy.trading_periods,
            },
            'risk': {
                'max_position_size': self.risk.max_position_size,
                'leverage': self.risk.leverage,
                'trading_fee': self.risk.trading_fee,
                'slippage': self.risk.slippage,
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'train_test_split': self.backtest.train_test_split,
            },
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> Config:
    """Get default configuration for quick start."""
    return Config()
