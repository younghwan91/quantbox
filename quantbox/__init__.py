"""
QuantBox - Binance Futures Pair Trading Framework

A quantitative trading framework for cryptocurrency pair trading
on Binance Futures with continuous backtesting and live trading.

Features:
- Kalman filter-based spread tracking with adaptive z-score
- Multi-layer defense: Z-Score Cap, Vol Gate, BTC Crash Gate
- Continuous pair backtesting with realistic cost modeling
- Systematic multi-timeframe data caching
- Flexible filter pipeline for pair selection
- Discord bot for live monitoring and control

Usage:
    from quantbox import Config, DataDownloader
    from quantbox.backtest import run_continuous_backtest

Author: QuantBox Team
Version: 1.3.0
"""

__version__ = "1.3.0"
__author__ = "QuantBox Team"

# Core components
from quantbox.core.config import Config, TradingMode
from quantbox.core.events import (
    Event, EventType, MarketEvent, SignalEvent,
    OrderEvent, FillEvent, EventQueue, SignalType
)

# Data components
from quantbox.data.cache import DataCache, TimeframedCache
from quantbox.data.downloader import DataDownloader
from quantbox.data.preprocessor import DataPreprocessor, PairData

# Strategy components
from quantbox.strategy.base import Strategy, StatefulStrategy
from quantbox.strategy.filters import (
    FilterPipeline, CorrelationFilter, CointegrationFilter,
    HalfLifeFilter, HurstFilter
)
from quantbox.strategy.pair_selector import PairSelector, PairCandidate

# Backtest components
from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
    TradeResult,
    run_continuous_backtest,
)
from quantbox.backtest.metrics import PerformanceMetrics, calculate_metrics

__all__ = [
    # Version
    "__version__",

    # Core
    "Config",
    "TradingMode",
    "Event",
    "EventType",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "EventQueue",
    "SignalType",

    # Data
    "DataCache",
    "TimeframedCache",
    "DataDownloader",
    "DataPreprocessor",
    "PairData",

    # Strategy
    "Strategy",
    "StatefulStrategy",
    "FilterPipeline",
    "CorrelationFilter",
    "CointegrationFilter",
    "HalfLifeFilter",
    "HurstFilter",
    "PairSelector",
    "PairCandidate",

    # Backtest
    "ContinuousBacktestConfig",
    "ContinuousPairBacktest",
    "TradeResult",
    "run_continuous_backtest",
    "PerformanceMetrics",
    "calculate_metrics",
]
