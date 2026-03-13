"""Session-level exposure cap.

Prevents new position entries when total open notional exceeds
balance × max_exposure_multiplier. This is a safety guard against:
- Config errors (accidental leverage/slot count changes)
- Edge cases in dynamic capital mode
- Runaway position accumulation
"""

import logging

logger = logging.getLogger("ExposureCap")


class ExposureGuard:
    """Checks whether new entries should be blocked based on total exposure.

    Usage:
        guard = ExposureGuard(max_multiplier=35.0)

        # Before each new entry:
        if not guard.can_open(
            current_notional=total_open_notional,
            new_notional=proposed_notional,
            balance=current_balance,
        ):
            logger.warning("Exposure cap hit — blocking new entry")
            return
    """

    def __init__(self, max_multiplier: float = 35.0):
        if max_multiplier <= 0:
            raise ValueError(f"max_multiplier must be positive, got {max_multiplier}")
        self._max_multiplier = max_multiplier
        self._blocked_count: int = 0

    def can_open(
        self,
        current_notional: float,
        new_notional: float,
        balance: float,
    ) -> bool:
        """Check if a new position can be opened within exposure limits.

        Args:
            current_notional: Sum of absolute notional of all open positions.
            new_notional: Proposed notional for the new position.
            balance: Current wallet balance.

        Returns:
            True if the new position is within limits.
        """
        if balance <= 0:
            logger.error("Balance is zero or negative — blocking entry")
            self._blocked_count += 1
            return False

        total = current_notional + new_notional
        max_allowed = balance * self._max_multiplier

        if total > max_allowed:
            ratio = total / balance
            logger.warning(
                f"Exposure cap hit: {ratio:.1f}x > {self._max_multiplier:.1f}x limit. "
                f"Current={current_notional:.0f}, New={new_notional:.0f}, "
                f"Balance={balance:.0f}, Max={max_allowed:.0f}"
            )
            self._blocked_count += 1
            return False

        return True

    @property
    def blocked_count(self) -> int:
        """Number of entries blocked by exposure cap."""
        return self._blocked_count

    @property
    def max_multiplier(self) -> float:
        return self._max_multiplier

    def utilization(self, current_notional: float, balance: float) -> float:
        """Current exposure utilization as a fraction (0.0 to 1.0+).

        Args:
            current_notional: Total open notional.
            balance: Current balance.

        Returns:
            Utilization ratio. Values > 1.0 mean over-limit.
        """
        if balance <= 0:
            return float("inf")
        return current_notional / (balance * self._max_multiplier)
