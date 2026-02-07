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
    - Leverage multiplies effective notional without increasing margin
    - Kelly criterion computed from actual spread PnL distribution
    - All helper functions are pure (no side effects) for testability
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np


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
    leverage: int = 1,
) -> float:
    """Calculate capital allocation for a new slot entry.

    Uses total book value (available + allocated across active slots)
    multiplied by position_size_pct, then amplified by leverage.
    The result is the effective notional exposure, not the margin required.

    For pair trading with leverage=2x:
      margin used = slot_capital_without_leverage
      notional exposure = margin * leverage

    Args:
        available_capital: Unallocated cash (free balance).
        active_slots: Currently active (positioned) slots.
        max_traders: Maximum number of concurrent traders.
        position_size_pct: Fraction of total book value per trade.
        leverage: Leverage multiplier (1 = no leverage, 2 = 2x, etc.).

    Returns:
        Effective notional capital for the trade. Always >= 0.

    Raises:
        ValueError: If available_capital is negative or leverage < 1.
    """
    if available_capital < 0:
        raise ValueError(
            f"available_capital cannot be negative: {available_capital}"
        )
    if leverage < 1:
        raise ValueError(
            f"leverage must be >= 1: {leverage}"
        )

    total_allocated = sum(
        s.position_info["capital"]
        for s in active_slots
        if s.position_info and "capital" in s.position_info
    )
    total_book_value = available_capital + total_allocated

    # Margin allocation (how much cash we actually lock up)
    target_margin = total_book_value * position_size_pct

    # Never allocate more margin than available cash
    margin = max(0.0, min(target_margin, available_capital))

    # Effective notional = margin × leverage
    return margin * leverage


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


def calculate_margin_from_notional(notional: float, leverage: int) -> float:
    """Calculate actual margin (cash locked) from notional exposure.

    This is the inverse of the leverage multiplication in calculate_slot_capital.
    Used to record how much cash is actually consumed from the pool.

    Args:
        notional: Total notional exposure (what calculate_slot_capital returns).
        leverage: Leverage multiplier.

    Returns:
        Margin (cash) required. Always >= 0.

    Raises:
        ValueError: If leverage < 1.
    """
    if leverage < 1:
        raise ValueError(f"leverage must be >= 1: {leverage}")
    return notional / leverage


def calculate_kelly_from_trades(
    trade_returns_pct: list[float] | np.ndarray,
    fraction: float = 0.5,
) -> float:
    """Calculate Kelly-optimal position sizing from actual trade PnL distribution.

    Unlike traditional Kelly (which assumes binary win/loss),
    this uses the continuous Kelly formula for a distribution of returns:

        f* = mean(R) / var(R)

    where R is the per-trade return distribution. This correctly handles
    pair trading's bounded risk-reward profile.

    Half-Kelly (fraction=0.5) is applied by default for safety.

    Args:
        trade_returns_pct: Array of per-trade net return percentages
            (e.g., [1.2, -0.5, 0.8, -0.3, ...]).
            These should be NET of fees and slippage.
        fraction: Kelly fraction (0.5 = half-Kelly, 1.0 = full Kelly).
            Must be in (0, 1].

    Returns:
        Optimal position_size_pct (fraction of capital to allocate per trade).
        Clamped to [0.01, 1.0]. Returns 0.0 if insufficient data or
        negative expectancy.

    Raises:
        ValueError: If fraction not in (0, 1].
    """
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1]: {fraction}")

    returns = np.asarray(trade_returns_pct, dtype=np.float64)

    # Need minimum sample size for statistical significance
    if len(returns) < 10:
        return 0.0

    mean_r = np.mean(returns)
    var_r = np.var(returns, ddof=1)  # unbiased variance

    # Negative expectancy → don't trade
    if mean_r <= 0:
        return 0.0

    # Variance too small → likely overfitting or trivial returns
    if var_r < 1e-10:
        return 0.0

    # Continuous Kelly: f* = μ / σ²
    kelly_raw = mean_r / var_r

    # Apply fractional Kelly
    kelly = kelly_raw * fraction

    # Convert from "fraction of bankroll per unit of return" to position_size_pct
    # For pair trading returns in %, divide by 100 to get decimal
    kelly_pct = kelly / 100.0

    # Safety clamp: minimum 1%, maximum 100% of capital
    return float(np.clip(kelly_pct, 0.01, 1.0))


def calculate_kelly_discrete(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    fraction: float = 0.5,
) -> float:
    """Calculate Kelly criterion using discrete win/loss model.

    Traditional Kelly formula:
        f* = (p * b - q) / b

    where:
        p = win probability
        q = 1 - p (loss probability)
        b = avg_win / avg_loss (payoff ratio)

    This is a simpler alternative to calculate_kelly_from_trades when
    only summary statistics are available.

    Args:
        win_rate: Win probability as fraction (e.g., 0.65 for 65%).
        avg_win_pct: Average winning trade return in % (positive).
        avg_loss_pct: Average losing trade return in % (positive, absolute value).
        fraction: Kelly fraction (0.5 = half-Kelly).

    Returns:
        Optimal position_size_pct. Clamped to [0.01, 1.0].
        Returns 0.0 if negative expectancy.

    Raises:
        ValueError: If win_rate not in [0, 1] or fraction not in (0, 1].
    """
    if not (0 <= win_rate <= 1):
        raise ValueError(f"win_rate must be in [0, 1]: {win_rate}")
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1]: {fraction}")
    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return 0.0

    p = win_rate
    q = 1 - p
    b = avg_win_pct / avg_loss_pct  # payoff ratio

    # Kelly: f* = (p*b - q) / b
    kelly_raw = (p * b - q) / b

    if kelly_raw <= 0:
        return 0.0

    kelly = kelly_raw * fraction

    # Clamp to [0.01, 1.0]
    return float(np.clip(kelly, 0.01, 1.0))
