#!/usr/bin/env python3
"""Query trade journal for performance analysis.

Usage:
    python scripts/query_journal.py                     # overall stats
    python scripts/query_journal.py --symbol BTCUSDT    # per-symbol
    python scripts/query_journal.py --strategy ma_cluster
    python scripts/query_journal.py --days 7            # last 7 days
    python scripts/query_journal.py --by-symbol         # rank all symbols
    python scripts/query_journal.py --by-hour           # hourly breakdown
    python scripts/query_journal.py --by-reason         # by exit reason
    python scripts/query_journal.py --recent 20         # last N trades
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt(val: float | None, fmt: str = ".2f") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _print_table(headers: list[str], rows: list[tuple]) -> None:
    col_widths = [len(h) for h in headers]
    str_rows = [[str(v) if v is not None else "—" for v in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "  ".join("-" * w for w in col_widths)
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print(sep)
    for row in str_rows:
        print("  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))


def overall_stats(conn: sqlite3.Connection, where: str = "", params: list = []) -> None:
    sql = f"""
        SELECT
            COUNT(*)                                              AS n,
            SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END)        AS wins,
            ROUND(SUM(pnl_usd), 2)                               AS total_pnl,
            ROUND(AVG(pnl_pct), 3)                               AS avg_pnl_pct,
            ROUND(AVG(mfe_pct), 3)                               AS avg_mfe,
            ROUND(AVG(mae_pct), 3)                               AS avg_mae,
            ROUND(SUM(CASE WHEN pnl_usd>0 THEN pnl_usd ELSE 0 END), 2) AS gross_profit,
            ROUND(ABS(SUM(CASE WHEN pnl_usd<0 THEN pnl_usd ELSE 0 END)), 2) AS gross_loss,
            ROUND(AVG(bars_held), 1)                             AS avg_bars
        FROM trades {where}
    """
    r = conn.execute(sql, params).fetchone()
    n = r["n"] or 0
    wins = r["wins"] or 0
    gp = r["gross_profit"] or 0
    gl = r["gross_loss"] or 0
    pf = f"{gp / gl:.3f}" if gl > 0 else "∞"
    wr = f"{wins / n * 100:.1f}%" if n > 0 else "N/A"

    print(f"\n{'─' * 40}")
    print(f"  Trades:         {n}")
    print(f"  Win Rate:       {wr}  ({wins}W / {n - wins}L)")
    print(f"  Profit Factor:  {pf}")
    print(f"  Total PnL:      ${r['total_pnl']:+.2f}" if r["total_pnl"] else "  Total PnL:      $0.00")
    print(f"  Avg PnL%:       {_fmt(r['avg_pnl_pct'])}%")
    print(f"  Avg MFE:        {_fmt(r['avg_mfe'])}%")
    print(f"  Avg MAE:        {_fmt(r['avg_mae'])}%")
    print(f"  Avg Hold:       {_fmt(r['avg_bars'], '.1f')} bars")
    print(f"{'─' * 40}\n")


def by_symbol(conn: sqlite3.Connection, where: str = "", params: list = []) -> None:
    sql = f"""
        SELECT
            symbol,
            strategy_type,
            COUNT(*)                                              AS n,
            SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END)        AS wins,
            ROUND(SUM(pnl_usd), 2)                               AS total_pnl,
            ROUND(AVG(pnl_pct), 2)                               AS avg_pnl
        FROM trades {where}
        GROUP BY symbol, strategy_type
        ORDER BY total_pnl DESC
    """
    rows = conn.execute(sql, params).fetchall()
    print("\n  Breakdown by symbol:\n")
    _print_table(
        ["Symbol", "Strategy", "N", "Wins", "Total PnL", "Avg PnL%"],
        [(r["symbol"], r["strategy_type"], r["n"], r["wins"],
          f"${r['total_pnl']:+.2f}", f"{r['avg_pnl']:+.2f}%") for r in rows],
    )
    print()


def by_hour(conn: sqlite3.Connection, where: str = "", params: list = []) -> None:
    sql = f"""
        SELECT
            CAST(strftime('%H', exit_time) AS INTEGER)           AS hour,
            COUNT(*)                                              AS n,
            ROUND(AVG(pnl_pct), 2)                               AS avg_pnl,
            ROUND(SUM(pnl_usd), 2)                               AS total_pnl
        FROM trades {where}
        GROUP BY hour ORDER BY hour
    """
    rows = conn.execute(sql, params).fetchall()
    print("\n  Breakdown by hour (UTC):\n")
    _print_table(
        ["Hour (UTC)", "N", "Avg PnL%", "Total PnL"],
        [(f"{r['hour']:02d}:00", r["n"], f"{r['avg_pnl']:+.2f}%", f"${r['total_pnl']:+.2f}") for r in rows],
    )
    print()


def by_reason(conn: sqlite3.Connection, where: str = "", params: list = []) -> None:
    sql = f"""
        SELECT
            exit_reason,
            COUNT(*)                                              AS n,
            SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END)        AS wins,
            ROUND(SUM(pnl_usd), 2)                               AS total_pnl,
            ROUND(AVG(pnl_pct), 2)                               AS avg_pnl
        FROM trades {where}
        GROUP BY exit_reason ORDER BY n DESC
    """
    rows = conn.execute(sql, params).fetchall()
    print("\n  Breakdown by exit reason:\n")
    _print_table(
        ["Exit Reason", "N", "Wins", "Total PnL", "Avg PnL%"],
        [(r["exit_reason"], r["n"], r["wins"],
          f"${r['total_pnl']:+.2f}", f"{r['avg_pnl']:+.2f}%") for r in rows],
    )
    print()


def recent_trades(conn: sqlite3.Connection, limit: int = 20) -> None:
    sql = """
        SELECT symbol, direction, strategy_type,
               ROUND(pnl_pct, 2) AS pnl_pct,
               ROUND(pnl_usd, 2) AS pnl_usd,
               exit_reason, bars_held, exit_time
        FROM trades ORDER BY exit_time DESC LIMIT ?
    """
    rows = conn.execute(sql, (limit,)).fetchall()
    print(f"\n  Last {limit} trades:\n")
    _print_table(
        ["Symbol", "Dir", "Strategy", "PnL%", "PnL$", "Reason", "Bars", "Exit Time"],
        [(r["symbol"], r["direction"], r["strategy_type"],
          f"{r['pnl_pct']:+.2f}%" if r["pnl_pct"] else "—",
          f"${r['pnl_usd']:+.2f}" if r["pnl_usd"] else "—",
          r["exit_reason"], r["bars_held"],
          (r["exit_time"] or "")[:19]) for r in rows],
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query trade journal")
    parser.add_argument("--db", default="data/journal.sqlite")
    parser.add_argument("--symbol", help="Filter by symbol")
    parser.add_argument("--strategy", choices=["vr_squeeze", "ma_cluster"], help="Filter by strategy")
    parser.add_argument("--session", choices=["live", "paper"], help="Filter by session")
    parser.add_argument("--days", type=int, help="Last N days only")
    parser.add_argument("--by-symbol", action="store_true", help="Rank by symbol")
    parser.add_argument("--by-hour", action="store_true", help="Hourly breakdown")
    parser.add_argument("--by-reason", action="store_true", help="By exit reason")
    parser.add_argument("--recent", type=int, metavar="N", help="Show last N trades")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Journal DB not found: {args.db}")
        print("Run: python scripts/backfill_journal.py")
        sys.exit(1)

    conn = _connect(args.db)

    # Build WHERE clause
    conditions: list[str] = []
    params: list = []
    if args.symbol:
        conditions.append("symbol = ?"); params.append(args.symbol)
    if args.strategy:
        conditions.append("strategy_type = ?"); params.append(args.strategy)
    if args.session:
        conditions.append("session = ?"); params.append(args.session)
    if args.days:
        conditions.append(f"exit_time >= datetime('now', '-{args.days} days')")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    label_parts = []
    if args.symbol: label_parts.append(f"symbol={args.symbol}")
    if args.strategy: label_parts.append(f"strategy={args.strategy}")
    if args.session: label_parts.append(f"session={args.session}")
    if args.days: label_parts.append(f"last {args.days} days")
    label = " | ".join(label_parts) or "all trades"

    print(f"\n  Trade Journal — {label}")
    overall_stats(conn, where, params)

    if args.by_symbol:
        by_symbol(conn, where, params)
    if args.by_hour:
        by_hour(conn, where, params)
    if args.by_reason:
        by_reason(conn, where, params)
    if args.recent:
        recent_trades(conn, args.recent)

    # Default: if no breakdown flag, show all three
    if not any([args.by_symbol, args.by_hour, args.by_reason, args.recent]):
        by_symbol(conn, where, params)
        by_reason(conn, where, params)
        recent_trades(conn, 10)


if __name__ == "__main__":
    main()
