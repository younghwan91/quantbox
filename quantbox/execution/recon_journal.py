"""Reconciliation audit journal — persistent SQLite event log.

Records every reconciliation action for long-term pattern analysis:
- Ghost positions detected and closed
- Reverse ghosts detected and closed
- Server SL/TP fills detected
- Bracket repairs (SL/TP re-placement)
- Failed operations

Schema designed for efficient querying by symbol, event type, and time range.
"""

import sqlite3
import time
import json
import logging
from pathlib import Path
from enum import Enum

logger = logging.getLogger("ReconJournal")


class ReconEventType(str, Enum):
    """Types of reconciliation events."""

    GHOST_DETECTED = "ghost_detected"
    GHOST_CLOSED = "ghost_closed"
    REVERSE_GHOST = "reverse_ghost"
    REVERSE_GHOST_CLOSED = "reverse_ghost_closed"
    SERVER_SL_FILL = "server_sl_fill"
    SERVER_TP_FILL = "server_tp_fill"
    BRACKET_REPAIR = "bracket_repair"
    BRACKET_REPAIR_FAIL = "bracket_repair_fail"
    POSITION_ADOPTED = "position_adopted"
    EXPOSURE_BLOCKED = "exposure_blocked"


class ReconJournal:
    """SQLite-backed reconciliation event journal.

    Usage:
        journal = ReconJournal(Path("data/journal/recon.db"))
        journal.record(
            event_type=ReconEventType.GHOST_DETECTED,
            symbol="BTCUSDT",
            action="Removed from state",
            result="success",
            details={"position_side": "LONG", "bars_held": 42}
        )

        # Query recent events
        events = journal.query_recent(hours=24)
        ghost_count = journal.count_by_type(ReconEventType.GHOST_DETECTED, hours=168)
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS recon_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT 'success',
                details_json TEXT NOT NULL DEFAULT '{}',
                session_id TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recon_timestamp
            ON recon_events(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recon_symbol
            ON recon_events(symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recon_type
            ON recon_events(event_type)
        """)
        self._conn.commit()

    def record(
        self,
        event_type: ReconEventType,
        symbol: str,
        action: str = "",
        result: str = "success",
        details: dict | None = None,
        session_id: str = "",
    ) -> int:
        """Record a reconciliation event.

        Args:
            event_type: Type of event.
            symbol: Trading symbol (e.g., "BTCUSDT").
            action: What was done (e.g., "auto-closed at market").
            result: "success" or "failure" or other status.
            details: Additional JSON-serializable data.
            session_id: Current trading session identifier.

        Returns:
            Row ID of the inserted event.
        """
        cursor = self._conn.execute(
            """INSERT INTO recon_events
               (timestamp, event_type, symbol, action, result, details_json, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                event_type.value,
                symbol,
                action,
                result,
                json.dumps(details or {}),
                session_id,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        logger.debug(f"Recorded {event_type.value}: {symbol} — {action} ({result})")
        return row_id

    def query_recent(self, hours: float = 24) -> list[dict]:
        """Get events from the last N hours.

        Returns:
            List of event dicts.
        """
        cutoff = time.time() - (hours * 3600)
        rows = self._conn.execute(
            "SELECT * FROM recon_events WHERE timestamp > ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_by_symbol(self, symbol: str, limit: int = 100) -> list[dict]:
        """Get events for a specific symbol."""
        rows = self._conn.execute(
            "SELECT * FROM recon_events WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_by_type(self, event_type: ReconEventType, hours: float = 168) -> int:
        """Count events of a given type in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        row = self._conn.execute(
            "SELECT COUNT(*) FROM recon_events WHERE event_type = ? AND timestamp > ?",
            (event_type.value, cutoff),
        ).fetchone()
        return row[0]

    def summary(self, hours: float = 168) -> dict[str, int]:
        """Get event count summary by type for the last N hours."""
        cutoff = time.time() - (hours * 3600)
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM recon_events WHERE timestamp > ? GROUP BY event_type",
            (cutoff,),
        ).fetchall()
        return {r["event_type"]: r["cnt"] for r in rows}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["details"] = json.loads(d.pop("details_json", "{}"))
        return d
