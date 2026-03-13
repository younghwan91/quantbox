"""Tests for quantbox.execution.recon_journal."""

from pathlib import Path
from unittest.mock import patch

import pytest

from quantbox.execution.recon_journal import ReconEventType, ReconJournal

NOW = 1_700_000_000.0


@pytest.fixture
def journal(tmp_path: Path) -> ReconJournal:
    j = ReconJournal(tmp_path / "journal" / "recon.db")
    yield j
    j.close()


# ── 1. DB creation ──────────────────────────────────────────────


def test_create_journal_creates_db(tmp_path: Path) -> None:
    db_path = tmp_path / "sub" / "recon.db"
    j = ReconJournal(db_path)
    assert db_path.exists()
    j.close()


# ── 2. record() returns row id ──────────────────────────────────


def test_record_returns_row_id(journal: ReconJournal) -> None:
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        row_id = journal.record(
            event_type=ReconEventType.GHOST_DETECTED,
            symbol="BTCUSDT",
        )
    assert isinstance(row_id, int)
    assert row_id > 0


# ── 3. record → query_recent round-trip ─────────────────────────


def test_record_and_query_recent(journal: ReconJournal) -> None:
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        journal.record(
            event_type=ReconEventType.GHOST_CLOSED,
            symbol="ETHUSDT",
            action="Removed from state",
            result="success",
        )
        events = journal.query_recent(hours=1)

    assert len(events) == 1
    e = events[0]
    assert e["event_type"] == "ghost_closed"
    assert e["symbol"] == "ETHUSDT"
    assert e["action"] == "Removed from state"
    assert e["result"] == "success"


# ── 4. query_recent respects time window ─────────────────────────


def test_query_recent_respects_time_window(journal: ReconJournal) -> None:
    old_ts = NOW - 7200  # 2 hours ago
    new_ts = NOW

    with patch("quantbox.execution.recon_journal.time.time", return_value=old_ts):
        journal.record(ReconEventType.GHOST_DETECTED, symbol="OLDUSDT")
    with patch("quantbox.execution.recon_journal.time.time", return_value=new_ts):
        journal.record(ReconEventType.GHOST_DETECTED, symbol="NEWUSDT")

    # query last 1 hour — should only get new event
    with patch("quantbox.execution.recon_journal.time.time", return_value=new_ts):
        events = journal.query_recent(hours=1)

    assert len(events) == 1
    assert events[0]["symbol"] == "NEWUSDT"


# ── 5. query_by_symbol ──────────────────────────────────────────


def test_query_by_symbol(journal: ReconJournal) -> None:
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        journal.record(ReconEventType.BRACKET_REPAIR, symbol="BTCUSDT")
        journal.record(ReconEventType.GHOST_DETECTED, symbol="ETHUSDT")
        journal.record(ReconEventType.BRACKET_REPAIR_FAIL, symbol="BTCUSDT")

    btc = journal.query_by_symbol("BTCUSDT")
    assert len(btc) == 2
    assert all(e["symbol"] == "BTCUSDT" for e in btc)


# ── 6. count_by_type ────────────────────────────────────────────


def test_count_by_type(journal: ReconJournal) -> None:
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        journal.record(ReconEventType.SERVER_SL_FILL, symbol="BTCUSDT")
        journal.record(ReconEventType.SERVER_SL_FILL, symbol="ETHUSDT")
        journal.record(ReconEventType.SERVER_TP_FILL, symbol="BTCUSDT")

        count = journal.count_by_type(ReconEventType.SERVER_SL_FILL, hours=1)
    assert count == 2


# ── 7. summary groups by type ───────────────────────────────────


def test_summary_groups_by_type(journal: ReconJournal) -> None:
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        journal.record(ReconEventType.GHOST_DETECTED, symbol="A")
        journal.record(ReconEventType.GHOST_DETECTED, symbol="B")
        journal.record(ReconEventType.BRACKET_REPAIR, symbol="C")

        s = journal.summary(hours=1)

    assert s["ghost_detected"] == 2
    assert s["bracket_repair"] == 1
    assert len(s) == 2


# ── 8. details JSON round-trip ──────────────────────────────────


def test_record_with_details(journal: ReconJournal) -> None:
    details = {"position_side": "LONG", "bars_held": 42, "pnl": -0.0031}
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        journal.record(
            ReconEventType.REVERSE_GHOST_CLOSED,
            symbol="SUIUSDT",
            details=details,
        )
        events = journal.query_recent(hours=1)

    assert events[0]["details"] == details


# ── 9. multiple events ordering ─────────────────────────────────


def test_multiple_events_ordering(journal: ReconJournal) -> None:
    for i in range(5):
        with patch(
            "quantbox.execution.recon_journal.time.time",
            return_value=NOW + i,
        ):
            journal.record(ReconEventType.GHOST_DETECTED, symbol=f"SYM{i}")

    with patch(
        "quantbox.execution.recon_journal.time.time",
        return_value=NOW + 10,
    ):
        events = journal.query_recent(hours=1)

    assert len(events) == 5
    # DESC order — newest first
    assert events[0]["symbol"] == "SYM4"
    assert events[-1]["symbol"] == "SYM0"


# ── 10. enum values are strings ─────────────────────────────────


def test_event_type_enum_values() -> None:
    for member in ReconEventType:
        assert isinstance(member.value, str)
        assert member.value == member.value.lower()
        assert "_" in member.value or member.value.isalpha()


# ── 11. empty journal queries ───────────────────────────────────


def test_empty_journal_queries(journal: ReconJournal) -> None:
    assert journal.query_recent(hours=24) == []
    assert journal.query_by_symbol("BTCUSDT") == []
    assert journal.count_by_type(ReconEventType.GHOST_DETECTED) == 0
    assert journal.summary() == {}


# ── 12. close and reopen — persistence ──────────────────────────


def test_close_and_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "recon.db"
    j1 = ReconJournal(db_path)

    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        j1.record(ReconEventType.POSITION_ADOPTED, symbol="XRPUSDT", action="adopted")
    j1.close()

    j2 = ReconJournal(db_path)
    with patch("quantbox.execution.recon_journal.time.time", return_value=NOW):
        events = j2.query_recent(hours=1)
    assert len(events) == 1
    assert events[0]["symbol"] == "XRPUSDT"
    assert events[0]["event_type"] == "position_adopted"
    j2.close()
