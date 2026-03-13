"""Tests for DiscordNotifier critical alert methods (T4)."""

from unittest.mock import patch, MagicMock

import pytest

from quantbox.notifications.discord import (
    DiscordNotifier,
    COLOR_GREEN,
    COLOR_YELLOW,
)


@pytest.fixture
def notifier():
    return DiscordNotifier(webhook_url="YOUR_DISCORD_WEBHOOK_URL")


@pytest.fixture
def mock_post():
    with patch("quantbox.notifications.discord.requests.post") as m:
        m.return_value = MagicMock(status_code=204)
        yield m


# ==================== notify_critical ====================


def test_notify_critical_sends_embed(notifier, mock_post):
    result = notifier.notify_critical("State corrupt", "Details here")
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert embed["title"] == "🔴 CRITICAL: State corrupt"
    assert embed["color"] == 0xFF0000
    assert embed["description"] == "Details here"


def test_notify_critical_cooldown(notifier, mock_post):
    with patch("time.time", return_value=1000.0):
        assert notifier.notify_critical("X", "msg") is True
    with patch("time.time", return_value=1200.0):
        assert notifier.notify_critical("X", "msg") is False
    assert mock_post.call_count == 1


def test_notify_critical_different_keys_no_cooldown(notifier, mock_post):
    with patch("time.time", return_value=1000.0):
        assert notifier.notify_critical("A", "msg", alert_key="key_a") is True
        assert notifier.notify_critical("B", "msg", alert_key="key_b") is True
    assert mock_post.call_count == 2


def test_notify_critical_truncates_message(notifier, mock_post):
    long_msg = "x" * 2000
    notifier.notify_critical("title", long_msg)
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert len(embed["description"]) == 1000


# ==================== notify_api_failure ====================


def test_notify_api_failure_under_threshold(notifier, mock_post):
    result = notifier.notify_api_failure("Balance fetch", "timeout", consecutive_failures=2)
    assert result is False
    mock_post.assert_not_called()


def test_notify_api_failure_sends_after_3(notifier, mock_post):
    result = notifier.notify_api_failure("Balance fetch", "timeout", consecutive_failures=3)
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert "Balance fetch" in embed["title"]
    assert embed["color"] == COLOR_YELLOW


def test_notify_api_failure_red_after_5(notifier, mock_post):
    result = notifier.notify_api_failure("Order placement", "500", consecutive_failures=5)
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert embed["color"] == 0xFF0000
    fields_by_name = {f["name"]: f["value"] for f in embed["fields"]}
    assert fields_by_name["연속 실패"] == "5회"


def test_notify_api_failure_cooldown(notifier, mock_post):
    with patch("time.time", return_value=1000.0):
        assert notifier.notify_api_failure("Op", "err", 3) is True
    with patch("time.time", return_value=1100.0):
        assert notifier.notify_api_failure("Op", "err", 3) is False
    assert mock_post.call_count == 1


# ==================== notify_circuit_breaker ====================


def test_notify_circuit_breaker_open(notifier, mock_post):
    result = notifier.notify_circuit_breaker(is_open=True, failures=10, backoff_s=60)
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert "OPEN" in embed["title"]
    assert embed["color"] == 0xFF0000
    assert "10회" in embed["description"]
    assert "60초" in embed["description"]


def test_notify_circuit_breaker_close(notifier, mock_post):
    result = notifier.notify_circuit_breaker(is_open=False, failures=0, backoff_s=0)
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert "RESET" in embed["title"]
    assert embed["color"] == COLOR_GREEN
    assert "정상 복구" in embed["description"]


# ==================== notify_state_restored ====================


def test_notify_state_restored(notifier, mock_post):
    result = notifier.notify_state_restored("scalper_state_backup_20260313.json")
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert "백업 복원" in embed["title"]
    assert embed["color"] == COLOR_YELLOW
    assert "scalper_state_backup_20260313.json" in embed["description"]


# ==================== notify_reconciliation_failure ====================


def test_notify_reconciliation_under_5(notifier, mock_post):
    result = notifier.notify_reconciliation_failure("BTCUSDT", "timeout", consecutive=4)
    assert result is False
    mock_post.assert_not_called()


def test_notify_reconciliation_sends_after_5(notifier, mock_post):
    result = notifier.notify_reconciliation_failure("BTCUSDT", "timeout", consecutive=5)
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert "BTCUSDT" in embed["title"]
    assert embed["color"] == 0xFF0000
    assert "5회" in embed["description"]


def test_notify_reconciliation_cooldown(notifier, mock_post):
    with patch("time.time", return_value=1000.0):
        assert notifier.notify_reconciliation_failure("ETHUSDT", "err", 5) is True
    with patch("time.time", return_value=1500.0):
        assert notifier.notify_reconciliation_failure("ETHUSDT", "err", 5) is False
    # 600s cooldown, so only 1 call
    assert mock_post.call_count == 1


# ==================== notify_heartbeat ====================


def test_notify_heartbeat_sends(notifier, mock_post):
    result = notifier.notify_heartbeat(stall_seconds=600)
    assert result is True
    embed = mock_post.call_args[1]["json"]["embeds"][0]
    assert "정지 감지" in embed["title"]
    assert embed["color"] == 0xFF0000
    assert "600초" in embed["description"]


def test_notify_heartbeat_cooldown(notifier, mock_post):
    with patch("time.time", return_value=1000.0):
        assert notifier.notify_heartbeat(900) is True
    with patch("time.time", return_value=1200.0):
        assert notifier.notify_heartbeat(1100) is False
    assert mock_post.call_count == 1


# ==================== _should_alert ====================


def test_should_alert_allows_after_cooldown(notifier):
    with patch("time.time", return_value=1000.0):
        assert notifier._should_alert("key", cooldown_s=60) is True
    with patch("time.time", return_value=1061.0):
        assert notifier._should_alert("key", cooldown_s=60) is True


def test_should_alert_blocks_within_cooldown(notifier):
    with patch("time.time", return_value=1000.0):
        assert notifier._should_alert("key", cooldown_s=60) is True
    with patch("time.time", return_value=1059.0):
        assert notifier._should_alert("key", cooldown_s=60) is False


def test_disabled_notifier_returns_false():
    with patch.dict("os.environ", {}, clear=True):
        n = DiscordNotifier(webhook_url=None)
    assert n.notify_critical("title", "msg") is False
    assert n.notify_api_failure("op", "err", 5) is False
    assert n.notify_circuit_breaker(True, 5, 30) is False
    assert n.notify_state_restored("backup.json") is False
    assert n.notify_reconciliation_failure("SYM", "err", 10) is False
    assert n.notify_heartbeat(999) is False
