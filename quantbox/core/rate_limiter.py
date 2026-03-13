"""API rate limiter with sliding window and exponential backoff.

Binance Futures API limits:
- 2400 request weight per minute (IP-based)
- Individual endpoint weights vary (1-20)

This module provides:
1. Sliding-window weight tracking
2. Pre-request wait calculation
3. Exponential backoff on 429/rate-limit errors
4. Circuit breaker after consecutive failures
"""

import re
import time
import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("RateLimiter")


@dataclass
class RateLimiter:
    """Sliding-window API rate limiter with exponential backoff."""

    max_weight_per_minute: int = 2400
    # Safety margin: start throttling at 80% usage
    throttle_threshold: float = 0.80

    # Backoff settings
    base_backoff_s: float = 2.0
    max_backoff_s: float = 60.0
    circuit_break_after: int = 5  # consecutive failures to trip circuit

    # Internal state
    _window: deque = field(default_factory=deque)  # (timestamp, weight) pairs
    _consecutive_failures: int = 0
    _circuit_open: bool = False
    _circuit_open_until: float = 0.0
    _last_backoff_s: float = 0.0

    def record_request(self, weight: int = 1) -> None:
        """Record an API request with its weight."""
        self._prune_window()
        self._window.append((time.monotonic(), weight))

    def current_usage(self) -> int:
        """Get current weight usage in the sliding window."""
        self._prune_window()
        return sum(w for _, w in self._window)

    def wait_if_needed(self, weight: int = 1) -> float:
        """Calculate and sleep if approaching rate limit.

        Returns:
            Seconds waited (0.0 if no wait needed).
        """
        # Check circuit breaker first
        if self._circuit_open:
            remaining = self._circuit_open_until - time.monotonic()
            if remaining > 0:
                logger.warning(f"Circuit breaker open — waiting {remaining:.1f}s")
                time.sleep(remaining)
                return remaining
            else:
                self._circuit_open = False
                self._consecutive_failures = 0
                logger.info("Circuit breaker reset")

        self._prune_window()
        current = sum(w for _, w in self._window)
        threshold = self.max_weight_per_minute * self.throttle_threshold

        if current + weight > threshold:
            # How long until enough weight expires?
            target = threshold - weight
            wait_s = 0.0
            cumulative = current
            for ts, w in self._window:
                expires_at = ts + 60.0
                wait_s = max(0.0, expires_at - time.monotonic())
                cumulative -= w
                if cumulative <= target:
                    break

            if wait_s > 0:
                logger.info(
                    f"Rate limit throttle: waiting {wait_s:.1f}s (usage={current}/{self.max_weight_per_minute})"
                )
                time.sleep(wait_s)
                return wait_s

        return 0.0

    def record_success(self) -> None:
        """Record a successful API call — resets backoff state."""
        self._consecutive_failures = 0
        self._last_backoff_s = 0.0

    def record_failure(self, retry_after_s: int | None = None) -> float:
        """Record a failed API call. Returns backoff duration in seconds.

        Args:
            retry_after_s: Retry-After header value if available.

        Returns:
            Number of seconds to wait before retrying.
        """
        self._consecutive_failures += 1

        if retry_after_s is not None and retry_after_s > 0:
            backoff = float(retry_after_s)
        else:
            # Exponential backoff: 2, 4, 8, 16, 32, 60, 60, 60...
            backoff = min(
                self.base_backoff_s * (2 ** (self._consecutive_failures - 1)),
                self.max_backoff_s,
            )

        self._last_backoff_s = backoff

        # Trip circuit breaker after N consecutive failures
        if self._consecutive_failures >= self.circuit_break_after:
            self._circuit_open = True
            self._circuit_open_until = time.monotonic() + backoff
            logger.critical(
                f"Circuit breaker OPEN after {self._consecutive_failures} "
                f"consecutive failures. No new entries for {backoff:.0f}s."
            )
        else:
            logger.warning(f"API failure #{self._consecutive_failures} — backoff {backoff:.1f}s")

        return backoff

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        if self._circuit_open:
            if time.monotonic() >= self._circuit_open_until:
                self._circuit_open = False
                self._consecutive_failures = 0
                return False
            return True
        return False

    def _prune_window(self) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = time.monotonic() - 60.0
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    @staticmethod
    def parse_rate_limit_error(error_str: str) -> tuple[bool, int | None]:
        """Parse Binance error for rate limit info.

        Returns:
            (is_rate_limit, retry_after_seconds_or_None)
        """
        if "429" in error_str or "Too Many Requests" in error_str:
            match = re.search(r'Retry-After["\']?\s*[:=]\s*(\d+)', error_str)
            if match:
                return True, int(match.group(1))
            return True, None
        if "-1015" in error_str:  # Binance specific: Too many new orders
            return True, None
        return False, None
