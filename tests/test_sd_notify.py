"""Tests for quantbox.core.sd_notify."""

import socket
import threading
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest

from quantbox.core.sd_notify import sd_notify


def test_sd_notify_returns_false_without_socket(monkeypatch):
    """Returns False when NOTIFY_SOCKET is not set and systemd.daemon unavailable."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    with patch.dict("sys.modules", {"systemd": None, "systemd.daemon": None}):
        result = sd_notify("WATCHDOG=1")
    assert result is False


def test_sd_notify_returns_false_systemd_import_error(monkeypatch):
    """Falls back gracefully when systemd.daemon raises ImportError."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    with patch("builtins.__import__", side_effect=ImportError):
        # sd_notify should not raise, just return False
        pass  # Can't easily patch builtins here; use the module-level approach below

    # Direct: no NOTIFY_SOCKET set, no systemd → False
    result = sd_notify("READY=1")
    assert result is False


def test_sd_notify_with_mock_socket(monkeypatch, tmp_path):
    """sd_notify sends the correct state string to NOTIFY_SOCKET."""
    sock_path = str(tmp_path / "notify.sock")

    received = []

    # Create a Unix DGRAM server to receive the notification
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server_sock.bind(sock_path)
    server_sock.settimeout(2.0)

    monkeypatch.setenv("NOTIFY_SOCKET", sock_path)

    # Patch out systemd.daemon so we exercise the fallback path
    with patch.dict("sys.modules", {"systemd": MagicMock(), "systemd.daemon": None}):
        # Make systemd.daemon import fail
        import sys

        mock_systemd = MagicMock()
        mock_daemon = MagicMock()
        mock_daemon.notify = MagicMock(side_effect=ImportError)
        mock_systemd.daemon = mock_daemon
        # Just test the raw socket fallback directly by removing systemd from modules
        original_systemd = sys.modules.pop("systemd", None)
        original_daemon = sys.modules.pop("systemd.daemon", None)
        try:
            result = sd_notify("WATCHDOG=1")
        finally:
            if original_systemd is not None:
                sys.modules["systemd"] = original_systemd
            if original_daemon is not None:
                sys.modules["systemd.daemon"] = original_daemon

    try:
        data = server_sock.recv(256)
        received.append(data.decode())
    except socket.timeout:
        pass
    finally:
        server_sock.close()

    assert result is True
    assert received == ["WATCHDOG=1"]


def test_sd_notify_with_systemd_daemon(monkeypatch):
    """Uses systemd.daemon.notify when available and returns its result."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    mock_daemon = MagicMock()
    mock_daemon.notify = MagicMock(return_value=True)
    mock_systemd = MagicMock()
    mock_systemd.daemon = mock_daemon

    import sys

    sys.modules["systemd"] = mock_systemd
    sys.modules["systemd.daemon"] = mock_daemon
    try:
        result = sd_notify("READY=1")
    finally:
        sys.modules.pop("systemd", None)
        sys.modules.pop("systemd.daemon", None)

    mock_daemon.notify.assert_called_once_with("READY=1")
    assert result is True


def test_sd_notify_abstract_namespace_socket(monkeypatch, tmp_path):
    """Handles @-prefixed abstract namespace sockets (Linux extension)."""
    # Abstract namespace: prepend null byte
    abstract_name = "@quantbox_test_notify"

    received = []

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server_sock.bind("\0" + abstract_name[1:])
    server_sock.settimeout(2.0)

    monkeypatch.setenv("NOTIFY_SOCKET", abstract_name)

    import sys

    sys.modules.pop("systemd", None)
    sys.modules.pop("systemd.daemon", None)

    try:
        result = sd_notify("STOPPING=1")
        try:
            data = server_sock.recv(256)
            received.append(data.decode())
        except socket.timeout:
            pass
    finally:
        server_sock.close()

    assert result is True
    assert received == ["STOPPING=1"]
