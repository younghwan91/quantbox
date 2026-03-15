"""Lightweight systemd sd_notify helper.

Uses systemd.daemon if available, falls back to raw $NOTIFY_SOCKET,
no-ops gracefully when not under systemd.
"""

import logging
import os
import socket

logger = logging.getLogger("sd_notify")


def sd_notify(state: str) -> bool:
    """Send sd_notify state string to systemd.

    Common states: "READY=1", "WATCHDOG=1", "STOPPING=1"
    Returns True if notification was sent, False otherwise.
    """
    # Try systemd.daemon first
    try:
        import systemd.daemon

        return systemd.daemon.notify(state)
    except ImportError:
        pass

    # Fallback: raw NOTIFY_SOCKET
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        return False

    try:
        if notify_socket.startswith("@"):
            notify_socket = "\0" + notify_socket[1:]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.connect(notify_socket)
            sock.sendall(state.encode())
            return True
        finally:
            sock.close()
    except Exception as e:
        logger.debug(f"sd_notify failed: {e}")
        return False
