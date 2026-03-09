"""
QuantBox - Binance Futures Rubber Band Scalper

A quantitative trading framework for single-asset mean-reversion scalping
on Binance USDT-M Perpetual Futures.

Features:
- Rubber Band strategy: fade extreme 15m moves, ride the snap-back
- Numba-compiled signal generation and vectorized backtesting
- LimitOrderChaser for maker-only execution
- Multi-layer risk management (drawdown, daily loss, position limits)
- Discord notifications for live monitoring
- Atomic state persistence for crash recovery

Usage:
    from quantbox.strategy import RubberBandConfig, RubberBandStrategy
    from quantbox.backtest import VectorizedBacktest, BacktestConfig

Author: QuantBox Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "QuantBox Team"

# Core components
from quantbox.core.config import Config, TradingMode
from quantbox.core.events import (
    Event,
    EventType,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    EventQueue,
    SignalType,
)

# Data components
from quantbox.data.cache import DataCache, TimeframedCache
from quantbox.data.downloader import DataDownloader

# Strategy components
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals,
    calculate_position_size,
)

# Backtest components
from quantbox.backtest.vectorized_backtest import (
    BacktestConfig,
    BacktestResult,
    VectorizedBacktest,
)

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
    # Strategy
    "RubberBandConfig",
    "RubberBandStrategy",
    "compute_signals",
    "calculate_position_size",
    # Backtest
    "BacktestConfig",
    "BacktestResult",
    "VectorizedBacktest",
]
