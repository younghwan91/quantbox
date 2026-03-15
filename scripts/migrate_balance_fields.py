#!/usr/bin/env python3
"""Balance fields migration — balance_source 컬럼 추가 및 NULL backfill.

스키마:
    balance_source TEXT — 'measured' | 'interpolated' | NULL

Usage:
    python scripts/migrate_balance_fields.py [--dry-run] [--db-path PATH]
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


DEFAULT_DB = "data/journal.sqlite"

# straddle 페어 매칭 허용 시간 간격 (초)
PAIR_MATCH_WINDOW_SECS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_stats(conn: sqlite3.Connection, label: str) -> None:
    """NULL 현황 및 balance_source 분포를 출력한다."""
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                          AS total,
            SUM(CASE WHEN balance_before IS NULL THEN 1 ELSE 0 END) AS null_before,
            SUM(CASE WHEN balance_after  IS NULL THEN 1 ELSE 0 END) AS null_after
        FROM trades
        """
    ).fetchone()
    print(f"\n[{label}]")
    print(f"  total rows       : {row[0]}")
    print(f"  NULL balance_before : {row[1]}")
    print(f"  NULL balance_after  : {row[2]}")

    # balance_source 분포 (컬럼이 있을 때만)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "balance_source" in cols:
        dist = conn.execute(
            """
            SELECT COALESCE(balance_source, 'NULL') AS src, COUNT(*) AS n
            FROM trades
            GROUP BY balance_source
            ORDER BY n DESC
            """
        ).fetchall()
        print("  balance_source distribution:")
        for src, n in dist:
            print(f"    {src:15s}: {n}")


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------


def _add_balance_source_column(conn: sqlite3.Connection, dry_run: bool) -> None:
    """balance_source TEXT 컬럼 추가 — 멱등성 보장."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "balance_source" in cols:
        print("  [skip] balance_source 컬럼이 이미 존재합니다.")
        return
    if dry_run:
        print("  [dry]  ALTER TABLE trades ADD COLUMN balance_source TEXT")
        return
    conn.execute("ALTER TABLE trades ADD COLUMN balance_source TEXT")
    conn.commit()
    print("  [done] balance_source 컬럼 추가 완료.")


def _mark_measured(conn: sqlite3.Connection, dry_run: bool) -> int:
    """balance_before 또는 balance_after 가 non-NULL 인 레코드를 'measured' 로 표시."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "balance_source" not in cols:
        # dry-run 에서 컬럼이 아직 없으므로 단순 카운트만
        n = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE balance_before IS NOT NULL OR balance_after IS NOT NULL"
        ).fetchone()[0]
        print(f"  [dry]  {n} 개 레코드 → balance_source = 'measured'")
        return n
    sql_count = """
        SELECT COUNT(*) FROM trades
        WHERE (balance_before IS NOT NULL OR balance_after IS NOT NULL)
          AND balance_source IS NULL
    """
    n = conn.execute(sql_count).fetchone()[0]
    if n == 0:
        print("  [skip] 'measured' 로 표시할 신규 레코드 없음.")
        return 0
    if dry_run:
        print(f"  [dry]  {n} 개 레코드 → balance_source = 'measured'")
        return n
    conn.execute(
        """
        UPDATE trades
        SET balance_source = 'measured'
        WHERE (balance_before IS NOT NULL OR balance_after IS NOT NULL)
          AND balance_source IS NULL
        """
    )
    conn.commit()
    print(f"  [done] {n} 개 레코드 → balance_source = 'measured'")
    return n


def _interpolate_balance_after(conn: sqlite3.Connection, dry_run: bool) -> int:
    """NULL balance_after + known balance_before + known pnl_usd 에서 추정."""
    sql_count = """
        SELECT COUNT(*) FROM trades
        WHERE balance_after IS NULL
          AND balance_before IS NOT NULL
          AND pnl_usd IS NOT NULL
    """
    n = conn.execute(sql_count).fetchone()[0]
    if n == 0:
        print("  [skip] balance_after 추정 대상 없음.")
        return 0
    if dry_run:
        print(f"  [dry]  {n} 개 레코드 → balance_after = balance_before + pnl_usd (interpolated)")
        return n
    conn.execute(
        """
        UPDATE trades
        SET balance_after  = balance_before + pnl_usd,
            balance_source = 'interpolated'
        WHERE balance_after IS NULL
          AND balance_before IS NOT NULL
          AND pnl_usd IS NOT NULL
        """
    )
    conn.commit()
    print(f"  [done] {n} 개 레코드 balance_after 추정 완료.")
    return n


def _interpolate_balance_before_from_prev(conn: sqlite3.Connection, dry_run: bool) -> int:
    """NULL balance_before 를 이전 레코드의 balance_after 에서 추정.

    exit_time 기준으로 정렬한 직전 레코드의 balance_after 를 사용한다.
    """
    # id 순서로 정렬된 전체 rows 를 가져온다
    rows = conn.execute(
        "SELECT id, balance_before, balance_after FROM trades ORDER BY exit_time, id"
    ).fetchall()

    updates: list[tuple[float, int]] = []
    prev_balance_after: float | None = None

    for row_id, bal_before, bal_after in rows:
        if bal_before is None and prev_balance_after is not None:
            updates.append((prev_balance_after, row_id))
        # 현재 레코드의 balance_after 를 다음 반복을 위해 보존
        if bal_after is not None:
            prev_balance_after = bal_after

    if not updates:
        print("  [skip] balance_before 추정 대상 없음.")
        return 0
    if dry_run:
        print(f"  [dry]  {len(updates)} 개 레코드 → balance_before 이전 레코드 balance_after 에서 추정")
        return len(updates)

    conn.executemany(
        "UPDATE trades SET balance_before = ?, balance_source = 'interpolated' WHERE id = ?",
        updates,
    )
    conn.commit()
    print(f"  [done] {len(updates)} 개 레코드 balance_before 추정 완료.")
    return len(updates)


def _match_straddle_pairs(conn: sqlite3.Connection, dry_run: bool) -> int:
    """같은 symbol 에서 entry_time ±2분 이내 LONG+SHORT 페어를 straddle_pair_id 로 연결.

    이미 straddle_pair_id 가 있는 레코드는 건너뛴다.
    """
    # NULL straddle_pair_id 를 가진 레코드를 symbol 별로 묶어 매칭
    rows = conn.execute(
        """
        SELECT id, symbol, direction, entry_time
        FROM trades
        WHERE straddle_pair_id IS NULL
          AND entry_time IS NOT NULL
        ORDER BY symbol, entry_time
        """
    ).fetchall()

    # symbol 별로 그룹화
    from collections import defaultdict
    from datetime import datetime, timezone

    by_symbol: dict[str, list[tuple[int, str, str, datetime]]] = defaultdict(list)
    for row_id, symbol, direction, entry_time_str in rows:
        try:
            dt = datetime.fromisoformat(entry_time_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        by_symbol[symbol].append((row_id, symbol, direction, dt))

    pair_updates: list[tuple[str, int]] = []
    pair_counter = 0

    for symbol, entries in by_symbol.items():
        longs = [(rid, dt) for rid, _sym, d, dt in entries if d == "LONG"]
        shorts = [(rid, dt) for rid, _sym, d, dt in entries if d == "SHORT"]

        matched_long_ids: set[int] = set()
        matched_short_ids: set[int] = set()

        for long_id, long_dt in longs:
            if long_id in matched_long_ids:
                continue
            for short_id, short_dt in shorts:
                if short_id in matched_short_ids:
                    continue
                diff = abs((long_dt - short_dt).total_seconds())
                if diff <= PAIR_MATCH_WINDOW_SECS:
                    pair_counter += 1
                    pair_id = f"pair_{symbol}_{pair_counter:04d}"
                    pair_updates.append((pair_id, long_id))
                    pair_updates.append((pair_id, short_id))
                    matched_long_ids.add(long_id)
                    matched_short_ids.add(short_id)
                    break

    if not pair_updates:
        print("  [skip] 매칭 가능한 straddle 페어 없음.")
        return 0
    if dry_run:
        print(f"  [dry]  {pair_counter} 쌍 ({len(pair_updates)} 레코드) straddle_pair_id 연결 예정")
        return pair_counter
    conn.executemany(
        "UPDATE trades SET straddle_pair_id = ? WHERE id = ?",
        pair_updates,
    )
    conn.commit()
    print(f"  [done] {pair_counter} 쌍 ({len(pair_updates)} 레코드) straddle_pair_id 연결 완료.")
    return pair_counter


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def migrate(db_path: str, dry_run: bool) -> None:
    """전체 마이그레이션을 실행한다.

    Args:
        db_path: SQLite DB 파일 경로.
        dry_run: True 이면 변경 없이 현황만 출력.
    """
    path = Path(db_path).resolve()
    if not path.exists():
        print(f"[ERROR] DB 파일을 찾을 수 없습니다: {path}")
        return

    # 백업 (idempotent — 이미 있으면 skip)
    backup_path = Path(str(path) + ".backup_premigration")
    if not backup_path.exists():
        if dry_run:
            print(f"[dry]  백업 생성 예정: {backup_path}")
        else:
            shutil.copy2(path, backup_path)
            print(f"[done] 백업 생성: {backup_path}")
    else:
        print(f"[skip] 백업 이미 존재: {backup_path}")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    print("\n=== 마이그레이션 전 현황 ===")
    _print_stats(conn, "BEFORE")

    print("\n=== 마이그레이션 단계 ===")

    print("\n[1] balance_source 컬럼 추가")
    _add_balance_source_column(conn, dry_run)

    print("\n[2] non-NULL balance 레코드 → 'measured' 표시")
    _mark_measured(conn, dry_run)

    print("\n[3] NULL balance_after 추정 (balance_before + pnl_usd)")
    _interpolate_balance_after(conn, dry_run)

    print("\n[4] NULL balance_before 추정 (이전 레코드 balance_after)")
    _interpolate_balance_before_from_prev(conn, dry_run)

    print("\n[5] straddle 페어 매칭 (±2분 이내 LONG+SHORT)")
    _match_straddle_pairs(conn, dry_run)

    print("\n=== 마이그레이션 후 현황 ===")
    _print_stats(conn, "AFTER")

    conn.close()

    if dry_run:
        print("\n[DRY RUN 완료] 실제 변경 없음. --dry-run 플래그를 제거하면 적용됩니다.")
    else:
        print("\n[마이그레이션 완료]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", default=DEFAULT_DB, help=f"SQLite DB 경로 (기본값: {DEFAULT_DB})")
    parser.add_argument("--dry-run", action="store_true", help="변경 없이 현황만 출력")
    args = parser.parse_args()

    migrate(db_path=args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
