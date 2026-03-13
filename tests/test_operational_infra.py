"""Tests for operational infrastructure: log cleanup and Discord alert fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from quantbox.core.bot_utils import cleanup_old_logs, cleanup_all_log_dirs
from quantbox.notifications.discord import DiscordNotifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_URL = "https://discord.example.com/webhook/fake"


def _make_old_log(path: Path, name: str, age_days: float = 40) -> Path:
    """Create a .log file and back-date its mtime."""
    f = path / name
    f.write_text("old log content")
    old_mtime = time.time() - age_days * 86400
    os.utime(f, (old_mtime, old_mtime))
    return f


# ===========================================================================
# Task A: Log Cleanup
# ===========================================================================


def test_cleanup_old_logs_removes_old_files(tmp_path):
    """Old .log files are deleted; recent one survives."""
    _make_old_log(tmp_path, "session_old1.log", age_days=40)
    _make_old_log(tmp_path, "session_old2.log", age_days=35)
    _make_old_log(tmp_path, "session_old3.log", age_days=31)

    # Recent file — mtime = now (within last 10 s guard)
    recent = tmp_path / "session_recent.log"
    recent.write_text("current session")

    deleted = cleanup_old_logs(tmp_path, max_age_days=30)

    assert deleted == 3
    assert not (tmp_path / "session_old1.log").exists()
    assert not (tmp_path / "session_old2.log").exists()
    assert not (tmp_path / "session_old3.log").exists()
    assert recent.exists()


def test_cleanup_old_logs_skips_non_log_files(tmp_path):
    """Non-.log files are never deleted even if old."""
    _make_old_log(tmp_path, "old.log", age_days=40)  # should be deleted
    old_json = tmp_path / "state.json"
    old_json.write_text("{}")
    old_mtime = time.time() - 40 * 86400
    os.utime(old_json, (old_mtime, old_mtime))

    deleted = cleanup_old_logs(tmp_path, max_age_days=30)

    assert deleted == 1  # only the .log file
    assert old_json.exists()


def test_cleanup_old_logs_never_raises(tmp_path):
    """Calling on a non-existent directory returns 0, never raises."""
    nonexistent = tmp_path / "does_not_exist"
    result = cleanup_old_logs(nonexistent, max_age_days=30)
    assert result == 0


def test_cleanup_old_logs_returns_count(tmp_path):
    """Return value equals the exact number of deleted files."""
    n = 5
    for i in range(n):
        _make_old_log(tmp_path, f"old_{i}.log", age_days=40)

    result = cleanup_old_logs(tmp_path, max_age_days=30)
    assert result == n


def test_cleanup_old_logs_max_age_days_zero_deletes_hour_old(tmp_path):
    """max_age_days=0 triggers 1-hour cutoff (maintenance mode)."""
    _make_old_log(tmp_path, "ninety_min.log", age_days=0.0625)  # 90 min

    # A file only 30 minutes old — should survive
    young = tmp_path / "thirty_min.log"
    young.write_text("young")
    young_mtime = time.time() - 30 * 60
    os.utime(young, (young_mtime, young_mtime))

    deleted = cleanup_old_logs(tmp_path, max_age_days=0)

    assert deleted == 1
    assert young.exists()


def test_cleanup_all_log_dirs_walks_subdirs(tmp_path):
    """cleanup_all_log_dirs descends into subdirectories."""
    sub1 = tmp_path / "bot" / "vol_straddle"
    sub2 = tmp_path / "bot" / "paper"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    _make_old_log(sub1, "a.log", age_days=40)
    _make_old_log(sub2, "b.log", age_days=40)
    _make_old_log(tmp_path, "root.log", age_days=40)

    total = cleanup_all_log_dirs(base=tmp_path, max_age_days=30)
    assert total == 3


def test_cleanup_all_log_dirs_nonexistent_base(tmp_path):
    """Returns 0 and doesn't raise when base doesn't exist."""
    result = cleanup_all_log_dirs(base=tmp_path / "missing", max_age_days=30)
    assert result == 0


# ===========================================================================
# Task B: Discord Alert Fallback
# ===========================================================================


def test_discord_fallback_writes_on_failure(tmp_path, monkeypatch):
    """When requests.post raises, the embed is written to the fallback JSONL."""
    fallback_path = tmp_path / "alerts" / "missed_alerts.jsonl"
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", fallback_path)

    def mock_post(*args, **kwargs):
        raise ConnectionError("Discord is down")

    monkeypatch.setattr(requests, "post", mock_post)

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    embed = {"title": "Test Alert", "color": 0xFF0000, "description": "something broke"}
    result = notifier.send(embed=embed)

    assert result is False
    assert fallback_path.exists()

    records = [json.loads(l) for l in fallback_path.read_text().strip().splitlines()]
    assert len(records) == 1
    assert records[0]["title"] == "Test Alert"
    assert records[0]["level"] == "critical"
    assert records[0]["description"] == "something broke"
    assert "ts" in records[0]


def test_discord_fallback_yellow_mapped_to_warning(tmp_path, monkeypatch):
    """Yellow embed color is mapped to 'warning' level."""
    from quantbox.notifications.discord import COLOR_YELLOW

    fallback_path = tmp_path / "alerts" / "missed_alerts.jsonl"
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", fallback_path)

    monkeypatch.setattr(requests, "post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError()))

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    notifier.send(embed={"title": "Warning", "color": COLOR_YELLOW})

    records = [json.loads(l) for l in fallback_path.read_text().strip().splitlines()]
    assert records[0]["level"] == "warning"


def test_discord_fallback_does_not_affect_local_logs(tmp_path, monkeypatch):
    """Even when fallback write fails (read-only path), Discord failure is silent."""
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    os.chmod(str(readonly_dir), 0o555)  # no write permission

    monkeypatch.setattr(DiscordNotifier, "_fallback_path", readonly_dir / "alerts.jsonl")

    def mock_post(*args, **kwargs):
        raise ConnectionError("Discord down")

    monkeypatch.setattr(requests, "post", mock_post)

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    # Must NOT raise despite Discord failure AND fallback write failure
    result = notifier.send(embed={"title": "Test", "color": 0xFF0000})
    assert result is False

    # Restore permissions so tmp_path cleanup works
    os.chmod(str(readonly_dir), 0o755)


def test_get_missed_alerts_returns_written_alerts(tmp_path, monkeypatch):
    """get_missed_alerts() returns entries written during failed sends."""
    fallback_path = tmp_path / "alerts" / "missed_alerts.jsonl"
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", fallback_path)

    def mock_post(*args, **kwargs):
        raise ConnectionError("Discord down")

    monkeypatch.setattr(requests, "post", mock_post)

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    notifier.send(embed={"title": "Alert 1", "color": 0xFF0000})

    alerts = DiscordNotifier.get_missed_alerts()
    assert len(alerts) == 1
    assert alerts[0]["title"] == "Alert 1"


def test_get_missed_alerts_clear(tmp_path, monkeypatch):
    """After clear=True, subsequent call returns empty list."""
    fallback_path = tmp_path / "alerts" / "missed_alerts.jsonl"
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", fallback_path)

    def mock_post(*args, **kwargs):
        raise ConnectionError("Discord down")

    monkeypatch.setattr(requests, "post", mock_post)

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    notifier.send(embed={"title": "Alert 1", "color": 0xFF0000})

    alerts = DiscordNotifier.get_missed_alerts(clear=True)
    assert len(alerts) == 1
    assert not fallback_path.exists()

    # Second read returns empty
    alerts_after = DiscordNotifier.get_missed_alerts()
    assert len(alerts_after) == 0


def test_get_missed_alerts_empty_when_no_file(tmp_path, monkeypatch):
    """get_missed_alerts() returns [] when no fallback file exists."""
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", tmp_path / "nonexistent.jsonl")
    assert DiscordNotifier.get_missed_alerts() == []


def test_discord_no_fallback_when_disabled(tmp_path, monkeypatch):
    """Fallback write is skipped when _fallback_enabled=False."""
    fallback_path = tmp_path / "alerts" / "missed_alerts.jsonl"
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", fallback_path)
    monkeypatch.setattr(DiscordNotifier, "_fallback_enabled", False)

    def mock_post(*args, **kwargs):
        raise ConnectionError("Discord down")

    monkeypatch.setattr(requests, "post", mock_post)

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    notifier.send(embed={"title": "Test", "color": 0xFF0000})

    assert not fallback_path.exists()


def test_discord_429_retry_once(tmp_path, monkeypatch):
    """On 429, waits retry_after seconds and retries exactly once."""
    fallback_path = tmp_path / "alerts" / "missed_alerts.jsonl"
    monkeypatch.setattr(DiscordNotifier, "_fallback_path", fallback_path)

    call_count = []
    slept: list[float] = []

    def mock_sleep(secs: float) -> None:
        slept.append(secs)

    monkeypatch.setattr(time, "sleep", mock_sleep)

    def mock_post(*args, **kwargs):
        call_count.append(1)
        if len(call_count) == 1:
            resp = MagicMock()
            resp.status_code = 429
            resp.json.return_value = {"retry_after": 0.5}
            return resp
        resp = MagicMock()
        resp.status_code = 204
        return resp

    monkeypatch.setattr(requests, "post", mock_post)

    notifier = DiscordNotifier(webhook_url=_FAKE_URL)
    result = notifier.send(embed={"title": "Rate limited", "color": 0x00C853})

    assert result is True
    assert len(call_count) == 2  # initial + one retry
    assert len(slept) == 1
    assert slept[0] == pytest.approx(0.5)
    # No fallback written since send ultimately succeeded
    assert not fallback_path.exists()
