"""Main loop heartbeat monitor.

Tracks the last successful cycle time and detects stalls.
"""

import logging
import time

from quantbox.core.sd_notify import sd_notify

logger = logging.getLogger("Heartbeat")


class Heartbeat:
    """Tracks main loop cycle timing for stall detection.

    Usage:
        hb = Heartbeat(stall_threshold_s=300)

        # In each cycle:
        hb.beat()

        # Check for stalls:
        if hb.is_stalled():
            alert(hb.seconds_since_last_beat())
    """

    def __init__(self, stall_threshold_s: float = 300.0):
        self._stall_threshold_s = stall_threshold_s
        self._last_beat: float = time.monotonic()
        self._total_beats: int = 0
        self._stall_alerted: bool = False

    def beat(self) -> None:
        """Record a successful cycle completion."""
        self._last_beat = time.monotonic()
        self._total_beats += 1
        self._stall_alerted = False
        sd_notify("WATCHDOG=1")

    @staticmethod
    def notify_ready() -> None:
        """Signal systemd that the process finished startup and is ready."""
        sd_notify("READY=1")

    def is_stalled(self) -> bool:
        """Check if the main loop appears stalled."""
        return self.seconds_since_last_beat() > self._stall_threshold_s

    def seconds_since_last_beat(self) -> float:
        """Seconds since last recorded beat."""
        return time.monotonic() - self._last_beat

    @property
    def total_beats(self) -> int:
        """Total number of beats recorded."""
        return self._total_beats

    @property
    def stall_alerted(self) -> bool:
        """Whether a stall alert has already been sent this stall period."""
        return self._stall_alerted

    @stall_alerted.setter
    def stall_alerted(self, value: bool) -> None:
        self._stall_alerted = value
