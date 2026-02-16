"""
Core module - Configuration, events, and settings.
"""

from quantbox.core.config import Config
from quantbox.core.events import Event, EventType, SignalEvent, OrderEvent, FillEvent
from quantbox.core.settings import QuantBoxSettings, get_settings
from quantbox.core.limit_chaser import LimitOrderChaser
from quantbox.core.state_persistence import save_state, load_state, clear_state

__all__ = [
    "Config",
    "Event",
    "EventType",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "QuantBoxSettings",
    "get_settings",
    "LimitOrderChaser",
    "save_state",
    "load_state",
    "clear_state",
]
