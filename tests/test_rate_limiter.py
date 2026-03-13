"""Tests for quantbox.core.rate_limiter — sliding window + exponential backoff."""

import time
from unittest.mock import patch

import pytest

from quantbox.core.rate_limiter import RateLimiter


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_limiter(**kwargs) -> RateLimiter:
    """Build a RateLimiter with small defaults for fast tests."""
    defaults = dict(
        max_weight_per_minute=100,
        throttle_threshold=0.80,
        base_backoff_s=2.0,
        max_backoff_s=60.0,
        circuit_break_after=5,
    )
    defaults.update(kwargs)
    return RateLimiter(**defaults)


# ── Sliding Window ───────────────────────────────────────────────────────


class TestSlidingWindow:
    """Tests for request recording and window pruning."""

    def test_record_request_tracks_weight(self):
        rl = _make_limiter()
        assert rl.current_usage() == 0

        rl.record_request(weight=5)
        assert rl.current_usage() == 5

        rl.record_request(weight=10)
        assert rl.current_usage() == 15

    def test_record_request_default_weight(self):
        rl = _make_limiter()
        rl.record_request()
        assert rl.current_usage() == 1

    @patch("quantbox.core.rate_limiter.time")
    def test_window_prunes_old_entries(self, mock_time):
        """Entries older than 60s are removed from the window."""
        t0 = 1000.0
        mock_time.monotonic.return_value = t0
        mock_time.sleep = time.sleep  # not needed but safe

        rl = _make_limiter()
        rl.record_request(weight=10)  # recorded at t=1000

        # Advance 30s — still in window
        mock_time.monotonic.return_value = t0 + 30
        assert rl.current_usage() == 10

        # Advance 61s — pruned
        mock_time.monotonic.return_value = t0 + 61
        assert rl.current_usage() == 0


# ── Throttling (wait_if_needed) ──────────────────────────────────────────


class TestThrottle:
    """Tests for pre-request throttle logic."""

    def test_wait_if_needed_no_wait_under_threshold(self):
        """No wait when usage is well below 80% threshold."""
        rl = _make_limiter(max_weight_per_minute=100)
        # Add 50 weight — 50% of limit, well under 80%
        rl.record_request(weight=50)
        waited = rl.wait_if_needed(weight=1)
        assert waited == 0.0

    @patch("quantbox.core.rate_limiter.time")
    def test_wait_if_needed_waits_near_limit(self, mock_time):
        """Waits when approaching the throttle threshold."""
        t0 = 1000.0
        mock_time.monotonic.return_value = t0

        # threshold = 100 * 0.80 = 80
        rl = _make_limiter(max_weight_per_minute=100, throttle_threshold=0.80)

        # Add 79 weight at t=1000 — just under threshold
        rl.record_request(weight=79)

        # Requesting weight=5 → 79+5=84 > 80 → needs to wait
        # The oldest entry (t=1000) expires at t=1060
        # After pruning 79, cumulative=0 ≤ target(75) → wait_s = 1060 - now
        # Simulate monotonic returning t0 during the loop check, then advancing
        call_count = 0

        def advancing_monotonic():
            nonlocal call_count
            call_count += 1
            # First calls: during _prune_window + threshold calc
            # Later: inside the loop's expires_at - monotonic()
            return t0

        mock_time.monotonic.side_effect = advancing_monotonic
        # sleep should just record (we verify it was called with > 0)
        slept_values: list[float] = []
        mock_time.sleep.side_effect = lambda s: slept_values.append(s)

        waited = rl.wait_if_needed(weight=5)

        assert waited > 0
        assert len(slept_values) == 1
        assert slept_values[0] == waited


# ── Backoff ──────────────────────────────────────────────────────────────


class TestBackoff:
    """Tests for exponential backoff on failures."""

    def test_record_success_resets_failures(self):
        rl = _make_limiter()
        rl.record_failure()
        rl.record_failure()
        assert rl._consecutive_failures == 2

        rl.record_success()
        assert rl._consecutive_failures == 0
        assert rl._last_backoff_s == 0.0

    def test_record_failure_exponential_backoff(self):
        """Backoff doubles: 2, 4, 8, 16, 32."""
        rl = _make_limiter(base_backoff_s=2.0, max_backoff_s=60.0, circuit_break_after=99)

        expected = [2.0, 4.0, 8.0, 16.0, 32.0]
        for i, exp in enumerate(expected, 1):
            backoff = rl.record_failure()
            assert backoff == exp, f"failure #{i}: expected {exp}, got {backoff}"

    def test_record_failure_max_cap(self):
        """Backoff is capped at max_backoff_s."""
        rl = _make_limiter(base_backoff_s=2.0, max_backoff_s=10.0, circuit_break_after=99)

        # 2, 4, 8, 10, 10, 10...
        for _ in range(3):
            rl.record_failure()  # 2, 4, 8

        backoff = rl.record_failure()  # 16 capped to 10
        assert backoff == 10.0

        backoff = rl.record_failure()  # 32 capped to 10
        assert backoff == 10.0

    def test_record_failure_retry_after_override(self):
        """retry_after_s from header takes priority over exponential calc."""
        rl = _make_limiter(base_backoff_s=2.0)

        # First failure would normally be 2.0, but retry_after overrides
        backoff = rl.record_failure(retry_after_s=30)
        assert backoff == 30.0

    def test_record_failure_retry_after_zero_uses_exponential(self):
        """retry_after_s=0 falls back to exponential backoff."""
        rl = _make_limiter(base_backoff_s=2.0)
        backoff = rl.record_failure(retry_after_s=0)
        assert backoff == 2.0

    def test_record_failure_retry_after_none_uses_exponential(self):
        """retry_after_s=None falls back to exponential backoff."""
        rl = _make_limiter(base_backoff_s=2.0)
        backoff = rl.record_failure(retry_after_s=None)
        assert backoff == 2.0


# ── Circuit Breaker ──────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for circuit breaker tripping and recovery."""

    def test_circuit_breaker_trips(self):
        """Circuit breaker trips after N consecutive failures."""
        rl = _make_limiter(circuit_break_after=3, base_backoff_s=2.0)

        rl.record_failure()  # #1
        rl.record_failure()  # #2
        assert not rl._circuit_open

        rl.record_failure()  # #3 — trips
        assert rl._circuit_open
        assert rl.is_circuit_open

    @patch("quantbox.core.rate_limiter.time")
    def test_circuit_breaker_resets(self, mock_time):
        """Circuit breaker resets after timeout expires."""
        t0 = 1000.0
        mock_time.monotonic.return_value = t0
        mock_time.sleep = time.sleep

        rl = _make_limiter(circuit_break_after=3, base_backoff_s=2.0)

        # Trip it (3 failures → backoff = 2*2^2 = 8s)
        rl.record_failure()
        rl.record_failure()
        rl.record_failure()  # circuit opens, backoff=8s, open_until=1008
        assert rl.is_circuit_open

        # Still open at t=1005
        mock_time.monotonic.return_value = t0 + 5
        assert rl.is_circuit_open

        # Expired at t=1009
        mock_time.monotonic.return_value = t0 + 9
        assert not rl.is_circuit_open
        assert rl._consecutive_failures == 0

    @patch("quantbox.core.rate_limiter.time")
    def test_wait_if_needed_respects_circuit_breaker(self, mock_time):
        """wait_if_needed waits for circuit breaker to expire."""
        t0 = 1000.0
        mock_time.monotonic.return_value = t0

        rl = _make_limiter(circuit_break_after=2, base_backoff_s=5.0)

        # Trip circuit (2 failures → backoff = 5*2^1 = 10s, open_until=1010)
        rl.record_failure()
        rl.record_failure()
        assert rl._circuit_open

        # wait_if_needed should see circuit is open and wait
        slept: list[float] = []
        mock_time.sleep.side_effect = lambda s: slept.append(s)

        waited = rl.wait_if_needed(weight=1)
        assert waited > 0
        assert len(slept) == 1


# ── Error Parsing ────────────────────────────────────────────────────────


class TestParseRateLimitError:
    """Tests for Binance error string parsing."""

    def test_parse_rate_limit_429(self):
        is_rl, retry = RateLimiter.parse_rate_limit_error("APIError(code=429): Too Many Requests")
        assert is_rl is True
        assert retry is None

    def test_parse_rate_limit_429_with_retry_after(self):
        is_rl, retry = RateLimiter.parse_rate_limit_error("APIError(code=429): Too Many Requests, Retry-After: 30")
        assert is_rl is True
        assert retry == 30

    def test_parse_rate_limit_1015(self):
        is_rl, retry = RateLimiter.parse_rate_limit_error("APIError(code=-1015): Too many new orders")
        assert is_rl is True
        assert retry is None

    def test_parse_non_rate_limit_error(self):
        is_rl, retry = RateLimiter.parse_rate_limit_error("APIError(code=-2019): Margin is insufficient.")
        assert is_rl is False
        assert retry is None

    def test_parse_too_many_requests_text(self):
        is_rl, retry = RateLimiter.parse_rate_limit_error("HTTPError: Too Many Requests")
        assert is_rl is True
        assert retry is None
