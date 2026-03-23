"""Tests for quantbox.core.trade_journal — TradeJournal SQLite persistence."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from quantbox.core.trade_journal import TradeJournal


# ─────────────────────────── Helpers ────────────────────────────────────────


def _journal(tmp_path: Path, session: str = "live") -> TradeJournal:
    """Create a TradeJournal backed by a temp SQLite file."""
    return TradeJournal(db_path=str(tmp_path / "journal.sqlite"), session=session)


def _trade(
    symbol: str = "BTCUSDT",
    signal: int = 1,
    pnl_usd: float = 100.0,
    pnl_pct: float = 1.5,
    reason: str = "trailing_stop",
    strategy_type="vr_directional",
    exit_time: str = "2026-01-01T12:00:00+00:00",
    mfe_pct: float = 2.0,
    mae_pct: float = 0.3,
) -> dict:
    return {
        "symbol": symbol,
        "signal": signal,
        "entry_price": 50000.0,
        "exit_price": 50750.0,
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "notional": 3000.0,
        "bars_held": 10,
        "reason": reason,
        "time": exit_time,
        "strategy_type": strategy_type,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "entry_time": "2026-01-01T10:00:00+00:00",
        "balance_before": 1000.0,
        "balance_after": 1100.0,
    }


# ─────────────────────────── DB init ────────────────────────────────────────


class TestInit:
    def test_creates_db_file(self, tmp_path):
        j = _journal(tmp_path)
        assert (tmp_path / "journal.sqlite").exists()

    def test_creates_trades_table(self, tmp_path):
        j = _journal(tmp_path)
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "trades" in tables
        conn.close()

    def test_creates_unique_index(self, tmp_path):
        j = _journal(tmp_path)
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        indexes = {r[1] for r in conn.execute("SELECT * FROM sqlite_master WHERE type='index'")}
        assert "idx_trades_key" in indexes
        conn.close()

    def test_idempotent_init(self, tmp_path):
        """Creating a second journal on the same DB should not crash."""
        _journal(tmp_path)
        _journal(tmp_path)  # no exception


# ─────────────────────────── record() ───────────────────────────────────────


class TestRecord:
    def test_record_inserts_row(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade())
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        assert n == 1

    def test_record_fields_stored_correctly(self, tmp_path):
        j = _journal(tmp_path)
        t = _trade(symbol="ETHUSDT", pnl_usd=-50.0, reason="stop_loss")
        j.record(t)
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        row = conn.execute("SELECT * FROM trades").fetchone()
        conn.close()
        # sqlite3 row is positional — check via column names
        conn2 = sqlite3.connect(tmp_path / "journal.sqlite")
        conn2.row_factory = sqlite3.Row
        row2 = conn2.execute("SELECT * FROM trades").fetchone()
        conn2.close()
        assert row2["symbol"] == "ETHUSDT"
        assert row2["pnl_usd"] == pytest.approx(-50.0)
        assert row2["exit_reason"] == "stop_loss"
        assert row2["session"] == "live"

    def test_direction_long(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade(signal=1))
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT direction FROM trades").fetchone()
        conn.close()
        assert row["direction"] == "LONG"

    def test_direction_short(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade(signal=-1))
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT direction FROM trades").fetchone()
        conn.close()
        assert row["direction"] == "SHORT"

    def test_duplicate_ignored(self, tmp_path):
        """Same symbol+exit_time should not be inserted twice (UNIQUE constraint)."""
        j = _journal(tmp_path)
        j.record(_trade())
        j.record(_trade())  # duplicate
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        assert n == 1

    def test_different_exit_times_both_inserted(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade(exit_time="2026-01-01T12:00:00+00:00"))
        j.record(_trade(exit_time="2026-01-01T13:00:00+00:00"))
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        assert n == 2

    def test_record_does_not_raise_on_bad_data(self, tmp_path):
        """record() must never crash the trading loop."""
        j = _journal(tmp_path)
        j.record({})  # missing all fields — no exception

    def test_mfe_mae_stored(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade(mfe_pct=3.5, mae_pct=0.8))
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT mfe_pct, mae_pct FROM trades").fetchone()
        conn.close()
        assert row["mfe_pct"] == pytest.approx(3.5)
        assert row["mae_pct"] == pytest.approx(0.8)

    def test_session_label_paper(self, tmp_path):
        j = _journal(tmp_path, session="paper")
        j.record(_trade())
        conn = sqlite3.connect(tmp_path / "journal.sqlite")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT session FROM trades").fetchone()
        conn.close()
        assert row["session"] == "paper"


# ─────────────────────────── get_stats() ────────────────────────────────────


class TestGetStats:
    def _populate(self, j: TradeJournal) -> None:
        trades = [
            _trade("BTCUSDT", 1, 100.0, 1.5, "trailing_stop", "vr_directional", "2026-01-01T10:00:00+00:00"),
            _trade("BTCUSDT", -1, -50.0, -0.8, "stop_loss", "vr_directional", "2026-01-01T11:00:00+00:00"),
            _trade("ETHUSDT", 1, 80.0, 1.2, "take_profit", "ma_cluster", "2026-01-01T12:00:00+00:00"),
            _trade("ETHUSDT", -1, -30.0, -0.5, "stop_loss", "ma_cluster", "2026-01-01T13:00:00+00:00"),
        ]
        for t in trades:
            j.record(t)

    def test_overall_stats(self, tmp_path):
        j = _journal(tmp_path)
        self._populate(j)
        s = j.get_stats()
        assert s["n"] == 4
        assert s["wins"] == 2
        assert s["wr"] == pytest.approx(0.5)
        assert s["total_pnl_usd"] == pytest.approx(100.0)  # 100-50+80-30

    def test_profit_factor(self, tmp_path):
        j = _journal(tmp_path)
        self._populate(j)
        s = j.get_stats()
        # gross_profit=180, gross_loss=80 → PF=2.25
        assert s["profit_factor"] == pytest.approx(180.0 / 80.0)

    def test_filter_by_symbol(self, tmp_path):
        j = _journal(tmp_path)
        self._populate(j)
        s = j.get_stats(symbol="BTCUSDT")
        assert s["n"] == 2
        assert s["total_pnl_usd"] == pytest.approx(50.0)

    def test_filter_by_strategy(self, tmp_path):
        j = _journal(tmp_path)
        self._populate(j)
        s_vr = j.get_stats(strategy="vr_directional")
        s_ma = j.get_stats(strategy="ma_cluster")
        assert s_vr["n"] == 2
        assert s_ma["n"] == 2

    def test_empty_db_returns_zeros(self, tmp_path):
        j = _journal(tmp_path)
        s = j.get_stats()
        assert s["n"] == 0
        assert s["wr"] == 0.0
        assert s["profit_factor"] == float("inf")

    def test_all_wins_profit_factor_inf(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade(pnl_usd=100.0, exit_time="2026-01-01T10:00:00+00:00"))
        s = j.get_stats()
        assert s["profit_factor"] == float("inf")


# ─────────────────────────── get_recent() ───────────────────────────────────


class TestGetRecent:
    def test_returns_list_of_dicts(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade())
        result = j.get_recent(limit=5)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_respects_limit(self, tmp_path):
        j = _journal(tmp_path)
        for i in range(10):
            j.record(_trade(exit_time=f"2026-01-01T{i:02d}:00:00+00:00"))
        result = j.get_recent(limit=3)
        assert len(result) == 3

    def test_most_recent_first(self, tmp_path):
        j = _journal(tmp_path)
        j.record(_trade(exit_time="2026-01-01T10:00:00+00:00", pnl_usd=10.0))
        j.record(_trade(exit_time="2026-01-01T11:00:00+00:00", pnl_usd=20.0))
        result = j.get_recent(limit=2)
        assert result[0]["exit_time"] == "2026-01-01T11:00:00+00:00"

    def test_empty_db_returns_empty_list(self, tmp_path):
        j = _journal(tmp_path)
        assert j.get_recent() == []
