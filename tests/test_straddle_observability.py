"""Tests for Step 4: Pair-Level Observability and PnL Tracking.

straddle-pair-coordination.md Step 4 acceptance criteria:
- _init_db() runs ALTER TABLE trades ADD COLUMN straddle_pair_id TEXT wrapped in try/except (idempotent)
- _SCHEMA includes straddle_pair_id TEXT for fresh DBs
- record() INSERT includes straddle_pair_id
- straddle_pair_id appears in journal.sqlite trade records
- get_pair_pnl returns correct net PnL for completed pair
- !status shows straddle pair statistics
- Journal migration: existing records get straddle_pair_id=NULL
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from quantbox.core.trade_journal import TradeJournal
from tests.conftest import _build_vs_scalper


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_trade(pair_id: str | None = None, signal: int = 1, pnl_usd: float = 50.0) -> dict:
    return {
        "symbol": "TESTUSDT",
        "signal": signal,
        "entry_price": 1.0,
        "exit_price": 1.05 if signal == 1 else 0.95,
        "pnl_pct": 5.0,
        "pnl_usd": pnl_usd,
        "mfe_pct": 5.0,
        "mae_pct": -1.0,
        "notional": 1000.0,
        "bars_held": 10,
        "reason": "server_tp",
        "time": "2026-03-15T10:00:00+00:00",
        "strategy_type": "straddle",
        "entry_time": "2026-03-15T00:00:00+00:00",
        "balance_before": 1000.0,
        "balance_after": 1050.0,
        "straddle_pair_id": pair_id,
    }


# ─── Journal schema migration ─────────────────────────────────────────────────


class TestJournalSchemaMigration:
    def test_fresh_db_has_straddle_pair_id_column(self, tmp_path):
        """Fresh DB created by _SCHEMA includes straddle_pair_id column."""
        db = tmp_path / "test.sqlite"
        TradeJournal(str(db), session="test")

        conn = sqlite3.connect(db)
        info = conn.execute("PRAGMA table_info(trades)").fetchall()
        conn.close()

        col_names = [row[1] for row in info]
        assert "straddle_pair_id" in col_names

    def test_init_db_idempotent_on_existing_db(self, tmp_path):
        """Calling _init_db() twice does not raise (ALTER TABLE is idempotent)."""
        db = tmp_path / "test.sqlite"
        TradeJournal(str(db), session="test")
        # Second instantiation should not raise
        TradeJournal(str(db), session="test")

    def test_migration_on_old_db_without_column(self, tmp_path):
        """Old DB without straddle_pair_id gets the column added on first _init_db."""
        db = tmp_path / "old.sqlite"
        # Create old-style DB without straddle_pair_id
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy_type TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl_pct REAL,
                pnl_usd REAL,
                mfe_pct REAL,
                mae_pct REAL,
                notional REAL,
                bars_held INTEGER,
                entry_time TEXT,
                exit_time TEXT,
                exit_reason TEXT,
                balance_before REAL,
                balance_after REAL,
                session TEXT,
                commit_sha TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("INSERT INTO trades (symbol, session) VALUES ('BTCUSDT', 'live')")
        conn.commit()
        conn.close()

        # _init_db should add the column without error
        TradeJournal(str(db), session="live")

        conn2 = sqlite3.connect(db)
        info = conn2.execute("PRAGMA table_info(trades)").fetchall()
        existing_rows = conn2.execute("SELECT straddle_pair_id FROM trades").fetchall()
        conn2.close()

        col_names = [row[1] for row in info]
        assert "straddle_pair_id" in col_names
        # Existing rows have NULL for the new column
        assert existing_rows[0][0] is None


# ─── straddle_pair_id in trade records ────────────────────────────────────────


class TestTradeRecordPairId:
    def test_record_with_pair_id_stored(self, tmp_path):
        """Trade record with straddle_pair_id is stored in DB."""
        db = tmp_path / "journal.sqlite"
        journal = TradeJournal(str(db), session="test")

        trade = _make_trade(pair_id="abc12345")
        journal.record(trade)

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT straddle_pair_id FROM trades WHERE symbol='TESTUSDT'").fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "abc12345"

    def test_record_without_pair_id_is_null(self, tmp_path):
        """Non-straddle trade has straddle_pair_id=NULL."""
        db = tmp_path / "journal.sqlite"
        journal = TradeJournal(str(db), session="test")

        trade = _make_trade(pair_id=None)
        journal.record(trade)

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT straddle_pair_id FROM trades WHERE symbol='TESTUSDT'").fetchone()
        conn.close()

        assert row is not None
        assert row[0] is None

    def test_record_multiple_legs_same_pair(self, tmp_path):
        """Both LONG and SHORT legs of a pair stored with same pair_id."""
        db = tmp_path / "journal.sqlite"
        journal = TradeJournal(str(db), session="test")

        pair_id = "pair9999"
        long_trade = _make_trade(pair_id=pair_id, signal=1, pnl_usd=50.0)
        long_trade["time"] = "2026-03-15T10:00:00+00:00"
        short_trade = _make_trade(pair_id=pair_id, signal=-1, pnl_usd=-20.0)
        short_trade["time"] = "2026-03-15T11:00:00+00:00"

        journal.record(long_trade)
        journal.record(short_trade)

        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT direction, pnl_usd, straddle_pair_id FROM trades WHERE straddle_pair_id=?",
            (pair_id,),
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert all(r[2] == pair_id for r in rows)


# ─── get_pair_pnl utility ─────────────────────────────────────────────────────


class TestGetPairPnl:
    def test_pair_pnl_net_calculation(self, tmp_path):
        """get_pair_pnl returns correct net PnL for completed pair."""
        db = tmp_path / "journal.sqlite"
        journal = TradeJournal(str(db), session="test")

        pair_id = "testpair01"
        long_trade = _make_trade(pair_id=pair_id, signal=1, pnl_usd=80.0)
        long_trade["time"] = "2026-03-15T10:00:00+00:00"
        short_trade = _make_trade(pair_id=pair_id, signal=-1, pnl_usd=-30.0)
        short_trade["time"] = "2026-03-15T11:00:00+00:00"

        journal.record(long_trade)
        journal.record(short_trade)

        result = journal.get_pair_pnl(pair_id)

        assert result is not None
        assert result["pair_id"] == pair_id
        assert result["long_pnl"] == pytest.approx(80.0)
        assert result["short_pnl"] == pytest.approx(-30.0)
        assert result["net_pnl"] == pytest.approx(50.0)

    def test_pair_pnl_returns_none_for_unknown_pair(self, tmp_path):
        """get_pair_pnl returns None for pair_id with no trades."""
        db = tmp_path / "journal.sqlite"
        journal = TradeJournal(str(db), session="test")

        result = journal.get_pair_pnl("nonexistent_pair")
        assert result is None

    def test_pair_pnl_includes_bars_and_reason(self, tmp_path):
        """get_pair_pnl includes bars_held and exit_reason for each leg."""
        db = tmp_path / "journal.sqlite"
        journal = TradeJournal(str(db), session="test")

        pair_id = "testpair02"
        long_trade = _make_trade(pair_id=pair_id, signal=1, pnl_usd=60.0)
        long_trade["time"] = "2026-03-15T10:00:00+00:00"
        long_trade["bars_held"] = 8
        long_trade["reason"] = "server_tp"
        short_trade = _make_trade(pair_id=pair_id, signal=-1, pnl_usd=-10.0)
        short_trade["time"] = "2026-03-15T11:00:00+00:00"
        short_trade["bars_held"] = 12
        short_trade["reason"] = "server_sl"

        journal.record(long_trade)
        journal.record(short_trade)

        result = journal.get_pair_pnl(pair_id)
        assert result is not None
        assert result.get("long_bars") == 8
        assert result.get("short_bars") == 12
        assert result.get("long_reason") == "server_tp"
        assert result.get("short_reason") == "server_sl"


# ─── !status pair statistics ──────────────────────────────────────────────────


class TestStatusPairStats:
    def test_status_shows_pair_count(self):
        """!status output includes straddle pair count."""
        scalper = _build_vs_scalper()

        pair_id = "stat_pair01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": {
                "symbol": "TESTUSDT",
                "signal": 1,
                "is_straddle": True,
                "straddle_pair_id": pair_id,
                "entry_price": 1.0,
                "notional": 1000.0,
                "bars_held": 3,
                "best_pnl": 0.0,
                "position_side": "LONG",
            },
            "TESTUSDT:SHORT": {
                "symbol": "TESTUSDT",
                "signal": -1,
                "is_straddle": True,
                "straddle_pair_id": pair_id,
                "entry_price": 1.0,
                "notional": 1000.0,
                "bars_held": 3,
                "best_pnl": 0.0,
                "position_side": "SHORT",
            },
        }
        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": "1.0"})

        result = scalper._cmd_status()

        assert "1쌍" in result or "Straddle" in result

    def test_status_shows_orphan_count(self):
        """!status output includes orphaned leg count."""
        scalper = _build_vs_scalper()

        pair_id = "stat_pair02"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": {
                "symbol": "TESTUSDT",
                "signal": 1,
                "is_straddle": True,
                "straddle_pair_id": pair_id,
                "entry_price": 1.0,
                "notional": 1000.0,
                "bars_held": 3,
                "best_pnl": 0.0,
                "position_side": "LONG",
                "straddle_orphan_since_cycle": 2,  # orphaned
            },
        }
        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": "1.0"})

        result = scalper._cmd_status()

        assert "고아=1" in result or "orphan" in result.lower()

    def test_status_no_straddle_positions(self):
        """!status works normally when no straddle positions."""
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {}

        result = scalper._cmd_status()

        assert "0쌍" in result or "Straddle" in result
        assert "고아=0" in result
