"""Diagnostic JSONL Logger (v12.0).

Structured per-symbol-per-cycle data for parameter optimization.
Writes to logs/diagnostics/diagnostics.jsonl — load with pd.read_json(lines=True).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path


class DiagnosticLogger:
    """Append-only JSONL writer for structured diagnostic data.

    Zero impact on strategy logic — pure observation layer.
    Rotates at 50MB to prevent disk fill.
    """

    def __init__(self, log_dir: Path = Path("logs/diagnostics")):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Can't create dir — all log methods become no-ops
        self._path = log_dir / "diagnostics.jsonl"
        self._fh: logging.handlers.RotatingFileHandler | None = None
        self._logger = logging.getLogger("Diagnostics")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Don't spam main log
        try:
            fh = logging.handlers.RotatingFileHandler(
                str(self._path),
                maxBytes=50 * 1024 * 1024,
                backupCount=3,
            )
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(fh)
            self._fh = fh
        except Exception:
            pass  # Degrade gracefully — diagnostics are optional

    def log_scan(self, cycle: int, timestamp: str, diags: list[dict]) -> None:
        """Write one JSONL record per cycle with all symbol diagnostics."""
        if self._fh is None:
            return
        record = {
            "type": "scan",
            "cycle": cycle,
            "ts": timestamp,
            "symbols": diags,
        }
        self._logger.debug(json.dumps(record, separators=(",", ":")))

    def log_entry(self, cycle: int, timestamp: str, symbol: str, signal: int, diag: dict, sizing: dict) -> None:
        """Write entry diagnostic when a position is opened."""
        if self._fh is None:
            return
        record = {
            "type": "entry",
            "cycle": cycle,
            "ts": timestamp,
            "symbol": symbol,
            "signal": signal,
            "diag": diag,
            "sizing": sizing,
        }
        self._logger.debug(json.dumps(record, separators=(",", ":")))

    def log_exit(self, cycle: int, timestamp: str, trade: dict) -> None:
        """Write exit diagnostic when a position is closed."""
        if self._fh is None:
            return
        record = {
            "type": "exit",
            "cycle": cycle,
            "ts": timestamp,
            "trade": trade,
        }
        self._logger.debug(json.dumps(record, separators=(",", ":")))


_diag_logger = DiagnosticLogger()
