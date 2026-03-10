#!/usr/bin/env python3
"""Trade journal viewer — python3 scripts/journal.py [command]

Commands:
  (없음)     오늘 트레이드 + 전체 요약
  all        전체 트레이드 내역
  symbols    심볼별 손익 랭킹
  strategy   전략별 성과 비교
  hours      시간대별 손익 (UTC)
  last N     최근 N개 트레이드 (기본 20)
"""
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).parent.parent / "data" / "journal.sqlite"
REAL_FILTER = "ABS(pnl_pct) < 10 AND symbol NOT LIKE 'TEST%' AND symbol NOT LIKE '%:%'"


def connect():
    if not DB.exists():
        print(f"[오류] {DB} 없음 — 라이브 세션을 먼저 실행하세요.")
        sys.exit(1)
    return sqlite3.connect(DB)


def fmt_row(r):
    sym, strat, direction, pnl_usd, reason, exit_time = r
    sign = "+" if pnl_usd >= 0 else ""
    tag = "✅" if pnl_usd > 0 else "❌"
    t = str(exit_time)[:16]
    return f"  {tag} {t}  {sym:12s} {direction:5s} {strat:10s}  ${sign}{pnl_usd:7.2f}  {reason}"


def summary(conn):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT COUNT(*), SUM(pnl_usd),
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
               MIN(exit_time), MAX(exit_time)
        FROM trades WHERE {REAL_FILTER}
    """)
    n, total, wins, first, last = cur.fetchone()
    if not n:
        print("트레이드 없음")
        return
    wr = wins / n * 100
    sign = "+" if total >= 0 else ""
    print(f"─── 전체 요약 ({first[:10]} ~ {last[:10]}) ───")
    print(f"  거래 수: {n}건  승률: {wr:.1f}%  합계: ${sign}{total:.2f}")
    print()


def today(conn):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol, strategy_type, direction, pnl_usd, exit_reason, exit_time
        FROM trades
        WHERE DATE(exit_time) >= DATE('now', '-1 day')
          AND {REAL_FILTER}
        ORDER BY exit_time DESC
    """)
    rows = cur.fetchall()
    print(f"─── 최근 24시간 트레이드 ({len(rows)}건) ───")
    if not rows:
        print("  없음")
    for r in rows:
        print(fmt_row(r))
    print()


def last_n(conn, n=20):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol, strategy_type, direction, pnl_usd, exit_reason, exit_time
        FROM trades WHERE {REAL_FILTER}
        ORDER BY id DESC LIMIT {n}
    """)
    rows = cur.fetchall()
    print(f"─── 최근 {n}개 트레이드 ───")
    for r in reversed(rows):
        print(fmt_row(r))
    print()


def by_symbol(conn):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol,
               COUNT(*) as n,
               SUM(pnl_usd) as total,
               ROUND(100.0 * SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) / COUNT(*), 0) as wr
        FROM trades WHERE {REAL_FILTER}
        GROUP BY symbol ORDER BY total ASC
    """)
    rows = cur.fetchall()
    print("─── 심볼별 손익 (손실순) ───")
    for sym, n, total, wr in rows:
        bar = "█" * int(abs(total) / 10) if total else ""
        sign = "+" if total >= 0 else ""
        tag = "🟢" if total >= 0 else "🔴"
        print(f"  {tag} {sym:12s}  {n:2d}건  WR={wr:.0f}%  ${sign}{total:7.2f}  {bar}")
    print()


def by_strategy(conn):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT strategy_type,
               COUNT(*) as n,
               SUM(pnl_usd) as total,
               ROUND(100.0 * SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as wr,
               ROUND(AVG(bars_held), 1) as avg_bars
        FROM trades WHERE {REAL_FILTER}
        GROUP BY strategy_type
    """)
    rows = cur.fetchall()
    print("─── 전략별 성과 ───")
    for strat, n, total, wr, avg_bars in rows:
        sign = "+" if total >= 0 else ""
        tag = "🟢" if total >= 0 else "🔴"
        print(f"  {tag} {strat:12s}  {n:2d}건  WR={wr:.1f}%  합계=${sign}{total:.2f}  평균보유={avg_bars}바")
    print()


def by_hour(conn):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT CAST(strftime('%H', exit_time) AS INT) as h,
               COUNT(*) as n,
               ROUND(SUM(pnl_usd), 2) as total
        FROM trades WHERE {REAL_FILTER}
        GROUP BY h ORDER BY h
    """)
    rows = cur.fetchall()
    print("─── 시간대별 손익 (UTC) ───")
    for h, n, total in rows:
        bar_len = int(abs(total) / 5)
        bar = ("+" if total >= 0 else "-") * bar_len
        sign = "+" if total >= 0 else ""
        print(f"  {h:02d}:00  {n:2d}건  ${sign}{total:7.2f}  {bar}")
    print()


def all_trades(conn):
    cur = conn.cursor()
    cur.execute(f"""
        SELECT symbol, strategy_type, direction, pnl_usd, exit_reason, exit_time
        FROM trades WHERE {REAL_FILTER}
        ORDER BY id
    """)
    rows = cur.fetchall()
    print(f"─── 전체 트레이드 ({len(rows)}건) ───")
    for r in rows:
        print(fmt_row(r))
    print()


def main():
    args = sys.argv[1:]
    conn = connect()

    if not args:
        summary(conn)
        today(conn)
        by_strategy(conn)
    elif args[0] == "all":
        summary(conn)
        all_trades(conn)
    elif args[0] == "symbols":
        by_symbol(conn)
    elif args[0] == "strategy":
        by_strategy(conn)
    elif args[0] == "hours":
        by_hour(conn)
    elif args[0] == "last":
        n = int(args[1]) if len(args) > 1 else 20
        last_n(conn, n)
    else:
        print(__doc__)

    conn.close()


if __name__ == "__main__":
    main()
