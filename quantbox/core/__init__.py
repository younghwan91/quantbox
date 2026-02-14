"""
Core module - Configuration, events, and trading slots.
"""

from quantbox.core.config import Config
from quantbox.core.events import Event, EventType, SignalEvent, OrderEvent, FillEvent
from quantbox.core.trader_slot import (
    TraderSlotState,
    get_occupied_symbols,
    get_idle_slots,
    get_active_slots,
    calculate_slot_capital,
    is_pair_available,
    calculate_margin_from_notional,
    calculate_kelly_from_trades,
    calculate_kelly_discrete,
)

__all__ = [
    "Config",
    "Event",
    "EventType",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "TraderSlotState",
    "get_occupied_symbols",
    "get_idle_slots",
    "get_active_slots",
    "calculate_slot_capital",
    "is_pair_available",
    "calculate_margin_from_notional",
    "calculate_kelly_from_trades",
    "calculate_kelly_discrete",
]
