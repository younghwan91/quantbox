"""Tests for Discord bot reconnection and Heartbeat module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from quantbox.core.heartbeat import Heartbeat
from quantbox.notifications.bot import DiscordBotRunner


# ---------------------------------------------------------------------------
# DiscordBotRunner reconnection tests
# ---------------------------------------------------------------------------


class TestBotReconnect:
    """DiscordBotRunner reconnection logic."""

    def _make_runner(self, max_attempts: int = 5) -> DiscordBotRunner:
        runner = DiscordBotRunner(token="fake-token", channel_id=123)
        runner._max_reconnect_attempts = max_attempts
        return runner

    @patch("quantbox.notifications.bot.time.sleep")
    def test_reconnect_on_first_failure(self, mock_sleep: MagicMock):
        """Bot retries after first crash, succeeds on second attempt."""
        runner = self._make_runner()
        call_count = 0

        def fake_run():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network down")
            # Second call succeeds (clean return)

        runner._run_event_loop = fake_run
        runner._run_with_reconnect()

        assert call_count == 2
        assert runner.reconnect_count == 1
        mock_sleep.assert_called_once_with(2.0)  # base backoff

    @patch("quantbox.notifications.bot.time.sleep")
    def test_reconnect_exponential_backoff(self, mock_sleep: MagicMock):
        """Backoff doubles: 2, 4, 8, 16, then gives up at max_attempts=5."""
        runner = self._make_runner(max_attempts=5)
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"fail #{call_count}")

        runner._run_event_loop = always_fail
        runner._run_with_reconnect()

        assert call_count == 5
        assert runner.reconnect_count == 5
        backoffs = [c.args[0] for c in mock_sleep.call_args_list]
        assert backoffs == [2.0, 4.0, 8.0, 16.0]  # 4 sleeps before 5th fail

    @patch("quantbox.notifications.bot.time.sleep")
    def test_reconnect_max_attempts(self, mock_sleep: MagicMock):
        """Stops after max_reconnect_attempts failures."""
        runner = self._make_runner(max_attempts=3)

        def always_fail():
            raise RuntimeError("boom")

        runner._run_event_loop = always_fail
        runner._run_with_reconnect()

        assert runner.reconnect_count == 3
        assert mock_sleep.call_count == 2  # 2 sleeps, 3rd fail exits

    def test_reconnect_clean_shutdown(self):
        """No retry on clean exit (no exception)."""
        runner = self._make_runner()
        runner._run_event_loop = MagicMock()  # Returns cleanly

        runner._run_with_reconnect()

        runner._run_event_loop.assert_called_once()
        assert runner.reconnect_count == 0

    @patch("quantbox.notifications.bot.time.sleep")
    def test_reconnect_count_property(self, mock_sleep: MagicMock):
        """reconnect_count tracks the number of failed attempts."""
        runner = self._make_runner(max_attempts=4)
        call_count = 0

        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")

        runner._run_event_loop = fail_twice
        runner._run_with_reconnect()

        assert runner.reconnect_count == 2
        assert call_count == 3  # 2 fails + 1 success

    def test_reset_reconnect_count(self):
        """reset_reconnect_count() resets the counter to 0."""
        runner = self._make_runner()
        runner._reconnect_count = 3

        assert runner.reconnect_count == 3
        runner.reset_reconnect_count()
        assert runner.reconnect_count == 0

    @patch("quantbox.notifications.bot.time.sleep")
    def test_reconnect_backoff_capped_at_60(self, mock_sleep: MagicMock):
        """Backoff is capped at 60 seconds."""
        runner = self._make_runner(max_attempts=8)

        def always_fail():
            raise RuntimeError("fail")

        runner._run_event_loop = always_fail
        runner._run_with_reconnect()

        backoffs = [c.args[0] for c in mock_sleep.call_args_list]
        # 2, 4, 8, 16, 32, 60, 60  (7 sleeps before 8th fail)
        assert backoffs == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]

    @patch("quantbox.notifications.bot.time.sleep")
    def test_reconnect_stop_requested(self, mock_sleep: MagicMock):
        """_stop_requested prevents further reconnection attempts."""
        runner = self._make_runner()

        def fail_and_stop():
            runner._stop_requested = True
            raise RuntimeError("fail")

        runner._run_event_loop = fail_and_stop
        runner._run_with_reconnect()

        # Only 1 call — after the exception, _stop_requested is checked
        assert runner.reconnect_count == 1


# ---------------------------------------------------------------------------
# Heartbeat tests
# ---------------------------------------------------------------------------


class TestHeartbeat:
    """Heartbeat stall-detection tests."""

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_beat_resets_timer(self, mock_mono: MagicMock):
        """beat() resets the stall timer."""
        mock_mono.return_value = 100.0
        hb = Heartbeat(stall_threshold_s=10.0)

        mock_mono.return_value = 105.0
        assert hb.seconds_since_last_beat() == pytest.approx(5.0)

        hb.beat()
        assert hb.seconds_since_last_beat() == pytest.approx(0.0)

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_is_stalled_after_threshold(self, mock_mono: MagicMock):
        """Stall detected after threshold exceeded."""
        mock_mono.return_value = 0.0
        hb = Heartbeat(stall_threshold_s=10.0)

        mock_mono.return_value = 11.0
        assert hb.is_stalled() is True

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_not_stalled_within_threshold(self, mock_mono: MagicMock):
        """No stall within threshold."""
        mock_mono.return_value = 0.0
        hb = Heartbeat(stall_threshold_s=10.0)

        mock_mono.return_value = 9.0
        assert hb.is_stalled() is False

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_total_beats(self, mock_mono: MagicMock):
        """Counter increments with each beat()."""
        mock_mono.return_value = 0.0
        hb = Heartbeat()

        assert hb.total_beats == 0
        hb.beat()
        hb.beat()
        hb.beat()
        assert hb.total_beats == 3

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_stall_alerted_flag(self, mock_mono: MagicMock):
        """stall_alerted flag tracks alert state."""
        mock_mono.return_value = 0.0
        hb = Heartbeat()

        assert hb.stall_alerted is False
        hb.stall_alerted = True
        assert hb.stall_alerted is True

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_beat_clears_stall_alert(self, mock_mono: MagicMock):
        """beat() clears the stall_alerted flag."""
        mock_mono.return_value = 0.0
        hb = Heartbeat()

        hb.stall_alerted = True
        assert hb.stall_alerted is True

        hb.beat()
        assert hb.stall_alerted is False

    @patch("quantbox.core.heartbeat.time.monotonic")
    def test_heartbeat_stall_boundary(self, mock_mono: MagicMock):
        """Exactly at threshold is not stalled (> not >=)."""
        mock_mono.return_value = 0.0
        hb = Heartbeat(stall_threshold_s=10.0)

        mock_mono.return_value = 10.0
        assert hb.is_stalled() is False
