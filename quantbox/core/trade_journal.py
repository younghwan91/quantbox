"""Trade Journal — SQLite persistence for every closed trade.

Allows instant SQL queries across all sessions:
    SELECT symbol, COUNT(*), SUM(pnl_usd) FROM trades GROUP BY symbol;
    SELECT strategy_type, AVG(pnl_pct) FROM trades GROUP BY strategy_type;
    SELECT strftime('%H', exit_time), AVG(pnl_pct) FROM trades GROUP BY 1;
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from quantbox.core.bot_utils import GIT_COMMIT

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    strategy_type   TEXT,
    direction       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    pnl_pct         REAL,
    pnl_usd         REAL,
    mfe_pct         REAL,
    mae_pct         REAL,
    notional        REAL,
    bars_held       INTEGER,
    entry_time      TEXT,
    exit_time       TEXT,
    exit_reason     TEXT,
    balance_before  REAL,
    balance_after   REAL,
    session         TEXT,
    commit_sha      TEXT,
    straddle_pair_id TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_key ON trades(symbol, exit_time);
"""


def migrate_strategy_types(db_path: str | Path) -> int:
    """Migrate old strategy_type values to canonical names. Idempotent.

    Wraps both UPDATEs in a single transaction for atomicity.
    Safe to call on a fresh DB (no-op if trades table doesn't exist).
    """
    conn = sqlite3.connect(db_path)
    count = 0
    try:
        with conn:  # explicit transaction — atomic commit or rollback
            for old, new in [("vr_squeeze", "vr_directional"), ("straddle", "vr_straddle")]:
                cursor = conn.execute("UPDATE trades SET strategy_type = ? WHERE strategy_type = ?", (new, old))
                count += cursor.rowcount
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet (fresh DB)
    finally:
        conn.close()
    return count


class TradeJournal:
    """Persistent SQLite trade journal.

    Args:
        db_path: Path to SQLite database file. Created if it does not exist.
        session: ``'live'`` or ``'paper'``.
    """

    def __init__(self, db_path: str = "data/journal.sqlite", session: str = "live") -> None:
        self.db_path = Path(db_path)
        self.session = session
        self._commit_sha = GIT_COMMIT.get("short", "unknown")
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)
            # Schema migration: 기존 DB에 straddle_pair_id 컬럼 추가 (멱등성 보장)
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN straddle_pair_id TEXT")
                conn.commit()
            except Exception:
                pass  # 이미 존재하는 컬럼
        # Naming migration: vr_squeeze→vr_directional, straddle→vr_straddle (멱등)
        n = migrate_strategy_types(self.db_path)
        if n > 0:
            logger.info(f"Migrated {n} trade records to canonical strategy names")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, trade: dict) -> None:
        """Persist a closed trade record.

        Args:
            trade: Dict produced by ``PositionMixin._close_position()``.
                Expected keys: symbol, signal, entry_price, exit_price,
                pnl_pct, pnl_usd, notional, bars_held, reason, time,
                strategy_type, mfe_pct, mae_pct, entry_time,
                balance_before, balance_after.
        """
        direction = "LONG" if trade.get("signal", 1) == 1 else "SHORT"
        row = (
            trade.get("symbol"),
            trade.get("strategy_type", "vr_directional"),
            direction,
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("pnl_pct"),
            trade.get("pnl_usd"),
            trade.get("mfe_pct"),
            trade.get("mae_pct"),
            trade.get("notional"),
            trade.get("bars_held"),
            trade.get("entry_time"),
            trade.get("time"),  # exit_time
            trade.get("reason"),
            trade.get("balance_before"),
            trade.get("balance_after"),
            self.session,
            self._commit_sha,
            trade.get("straddle_pair_id"),
        )
        sql = """
            INSERT OR IGNORE INTO trades
                (symbol, strategy_type, direction, entry_price, exit_price,
                 pnl_pct, pnl_usd, mfe_pct, mae_pct, notional, bars_held,
                 entry_time, exit_time, exit_reason,
                 balance_before, balance_after, session, commit_sha, straddle_pair_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, row)
        except Exception as exc:  # never crash the trading loop
            logger.warning(f"[TradeJournal] record failed: {exc}")

    def get_stats(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        session: str | None = None,
    ) -> dict:
        """Return aggregate stats filtered by optional dimensions.

        Returns:
            Dict with keys: n, wins, wr, total_pnl_usd, avg_pnl_pct,
            profit_factor, avg_mfe_pct, avg_mae_pct.
        """
        where: list[str] = []
        params: list[str] = []
        if symbol:
            where.append("symbol = ?")
            params.append(symbol)
        if strategy:
            where.append("strategy_type = ?")
            params.append(strategy)
        if session:
            where.append("session = ?")
            params.append(session)

        clause = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT
                COUNT(*)                                  AS n,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(pnl_usd)                              AS total_pnl_usd,
                AVG(pnl_pct)                              AS avg_pnl_pct,
                SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END) AS gross_profit,
                ABS(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END)) AS gross_loss,
                AVG(mfe_pct)                              AS avg_mfe_pct,
                AVG(mae_pct)                              AS avg_mae_pct
            FROM trades {clause}
        """
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()

        n = row["n"] or 0
        wins = row["wins"] or 0
        gross_profit = row["gross_profit"] or 0.0
        gross_loss = row["gross_loss"] or 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "n": n,
            "wins": wins,
            "wr": wins / n if n > 0 else 0.0,
            "total_pnl_usd": row["total_pnl_usd"] or 0.0,
            "avg_pnl_pct": row["avg_pnl_pct"] or 0.0,
            "profit_factor": pf,
            "avg_mfe_pct": row["avg_mfe_pct"] or 0.0,
            "avg_mae_pct": row["avg_mae_pct"] or 0.0,
        }

    def get_pair_pnl(self, pair_id: str) -> dict | None:
        """페어 단위 PnL 조회 (스트래들 LONG + SHORT 합산).

        Args:
            pair_id: straddle_pair_id 값.

        Returns:
            Dict with keys: pair_id, long_pnl, short_pnl, net_pnl,
            long_bars, short_bars, long_reason, short_reason.
            None if no trades found for this pair.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT direction, pnl_usd, bars_held, exit_reason FROM trades WHERE straddle_pair_id = ?",
                (pair_id,),
            ).fetchall()
        if not rows:
            return None
        result: dict = {"pair_id": pair_id, "long_pnl": 0.0, "short_pnl": 0.0, "net_pnl": 0.0}
        for row in rows:
            direction, pnl_usd, bars, reason = row["direction"], row["pnl_usd"], row["bars_held"], row["exit_reason"]
            pnl_usd = pnl_usd or 0.0
            if direction == "LONG":
                result["long_pnl"] = pnl_usd
                result["long_bars"] = bars
                result["long_reason"] = reason
            else:
                result["short_pnl"] = pnl_usd
                result["short_bars"] = bars
                result["short_reason"] = reason
        result["net_pnl"] = result["long_pnl"] + result["short_pnl"]
        return result

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Return the most recent ``limit`` trades as list of dicts."""
        sql = """
            SELECT symbol, strategy_type, direction, pnl_pct, pnl_usd,
                   exit_reason, bars_held, exit_time
            FROM trades ORDER BY exit_time DESC LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]
