"""
TraderSlot - Independent trading slot for multi-trader architecture.

Each slot operates independently with its own position state,
but shares capital and symbol exclusion with other slots.
Max 5 concurrent traders can run independently, each searching for
and executing pair trades without blocking others.

Design:
    - Each TraderSlotState holds one pair position at a time
    - Slots share a symbol exclusion set (no duplicate symbols across slots)
    - Capital is allocated per-slot from a shared pool
    - All helper functions are pure (no side effects) for testability
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class TraderSlotState:
    """State for a single independent trader slot.

    Each slot can independently hold one pair position at a time.
    Multiple slots enable concurrent pair trading.

    Args:
        slot_id: Unique identifier for this slot (0-indexed).
        position_open: Whether this slot currently holds a position.
        current_pair: The (symbol1, symbol2) pair being traded, or None.
        position_info: Dict with entry details (prices, hedge ratio, capital, etc.).
    """

    slot_id: int
    position_open: bool = False
    current_pair: Optional[tuple[str, str]] = None
    position_info: Optional[Dict] = None

    @property
    def occupied_symbols(self) -> set[str]:
        """Get symbols in use by this slot."""
        if self.position_open and self.current_pair:
            return {self.current_pair[0], self.current_pair[1]}
        return set()

    def reset(self):
        """Reset slot to idle state."""
        self.position_open = False
        self.current_pair = None
        self.position_info = None


def get_occupied_symbols(slots: list[TraderSlotState]) -> set[str]:
    """Get all symbols currently occupied across all active slots.

    Args:
        slots: List of all trader slots.

    Returns:
        Set of symbol strings that are currently in use.
    """
    occupied: set[str] = set()
    for slot in slots:
        occupied |= slot.occupied_symbols
    return occupied


def get_idle_slots(slots: list[TraderSlotState]) -> list[TraderSlotState]:
    """Get all idle (non-positioned) slots.

    Args:
        slots: List of all trader slots.

    Returns:
        List of slots that are not currently holding a position.
    """
    return [s for s in slots if not s.position_open]


def get_active_slots(slots: list[TraderSlotState]) -> list[TraderSlotState]:
    """Get all active (positioned) slots.

    Args:
        slots: List of all trader slots.

    Returns:
        List of slots that are currently holding a position.
    """
    return [s for s in slots if s.position_open]


def calculate_slot_capital(
    available_capital: float,
    active_slots: list[TraderSlotState],
    max_traders: int,
    position_size_pct: float,
) -> float:
    """Calculate capital allocation for a new slot entry.

    Uses total book value (available + allocated across active slots)
    multiplied by position_size_pct, capped at available capital.

    Args:
        available_capital: Unallocated cash (free balance).
        active_slots: Currently active (positioned) slots.
        max_traders: Maximum number of concurrent traders.
        position_size_pct: Fraction of total book value per trade.

    Returns:
        Capital to allocate for the new trade. Always >= 0.

    Raises:
        ValueError: If available_capital is negative.
    """
    if available_capital < 0:
        raise ValueError(
            f"available_capital cannot be negative: {available_capital}"
        )

    total_allocated = sum(
        s.position_info["capital"]
        for s in active_slots
        if s.position_info and "capital" in s.position_info
    )
    total_book_value = available_capital + total_allocated

    target_capital = total_book_value * position_size_pct

    # Never allocate more than available
    return max(0.0, min(target_capital, available_capital))


def is_pair_available(
    pair: tuple[str, str],
    occupied_symbols: set[str],
) -> bool:
    """Check if a pair's symbols are available (not used by other slots).

    Args:
        pair: (symbol1, symbol2) to check.
        occupied_symbols: Set of symbols already in use.

    Returns:
        True if neither symbol in the pair is occupied.
    """
    return pair[0] not in occupied_symbols and pair[1] not in occupied_symbols
