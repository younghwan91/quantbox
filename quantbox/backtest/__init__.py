"""
Backtesting module - Simulation and performance analysis
"""

from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
    TradeResult,
    ExitReason,
    run_continuous_backtest,
)
from quantbox.backtest.metrics import PerformanceMetrics, calculate_metrics

__all__ = [
    "ContinuousBacktestConfig",
    "ContinuousPairBacktest",
    "TradeResult",
    "ExitReason",
    "run_continuous_backtest",
    "PerformanceMetrics",
    "calculate_metrics",
]
