"""
Core module - Trading engine, execution, and configuration
"""

from quantbox.core.engine import TradingEngine
from quantbox.core.config import Config
from quantbox.core.events import Event, EventType, SignalEvent, OrderEvent, FillEvent
from quantbox.core.portfolio import Portfolio
from quantbox.core.execution import ExecutionHandler, BacktestExecutionHandler, LiveExecutionHandler

__all__ = [
    "TradingEngine",
    "Config",
    "Event",
    "EventType", 
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "Portfolio",
    "ExecutionHandler",
    "BacktestExecutionHandler",
    "LiveExecutionHandler",
]
