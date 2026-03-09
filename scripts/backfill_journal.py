#!/usr/bin/env python3
"""Backfill trade_journal from existing log files.

Parses all scalper_live_*.log files and inserts CLOSED trade records
into data/journal.sqlite. Safe to re-run — duplicates are ignored.

Usage:
    python scripts/backfill_journal.py
    python scripts/backfill_journal.py --db data/journal.sqlite --log-dir logs/bot/vol_straddle
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantbox.core.trade_journal import TradeJournal

# Example log line:
# 2026-03-09 22:45:22,124 [INFO] Scalper - [CRVUSDT:LONG] CLOSED (trailing_stop): LONG PnL=+3.39% ($+36.21) MFE=+4.66% MAE=+0.00% held=39 bars

_CLOSED_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] Scalper - "
    r"\[(?P<sym_raw>[^\]]+)\] CLOSED \((?P<reason>[^)]+)\): (?P<dir>LONG|SHORT) "
    r"PnL=(?P<pnl_pct>[+-][\d.]+)% \(\$(?P<pnl_usd>[+-][\d.]+)\)"
    r"(?: MFE=(?P<mfe>[+-][\d.]+)%)?"
    r"(?: MAE=(?P<mae>[+-][\d.]+)%)?"
    r"(?: held=(?P<bars>\d+) bars)?"
)

# Entry line (for entry_time):
# [CRVUSDT] Position opened: LONG TF=15m SL=1.0% TP=12.0% ... (vr_squeeze)
_OPENED_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] VolStraddle - "
    r"\[(?P<sym_raw>[^\]]+)\] Position opened: (?P<dir>LONG|SHORT)"
    r".*?\((?P<strategy>vr_squeeze|ma_cluster)\)"
)


def _parse_symbol(sym_raw: str) -> tuple[str, str | None]:
    """'CRVUSDT:LONG' → ('CRVUSDT', 'LONG'), 'CRVUSDT' → ('CRVUSDT', None)."""
    if ":" in sym_raw:
        parts = sym_raw.rsplit(":", 1)
        return parts[0], parts[1]
    return sym_raw, None


def _ts_to_iso(ts_str: str) -> str:
    """'2026-03-09 22:45:22' → '2026-03-09T22:45:22+00:00'."""
    return ts_str.replace(" ", "T") + "+00:00"


def backfill(log_dir: str, db_path: str, session: str = "live", dry_run: bool = False) -> int:
    log_path = Path(log_dir)
    journal = TradeJournal(db_path=db_path, session=session)

    pattern = "scalper_live_*.log" if session == "live" else "scalper_paper_*.log"
    log_files = sorted(log_path.glob(pattern))

    if not log_files:
        print(f"No log files found matching {log_path / pattern}")
        return 0

    print(f"Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  {f.name}")

    # First pass: collect entry timestamps per session
    entry_map: dict[str, str] = {}  # sym_key → iso timestamp
    strategy_map: dict[str, str] = {}  # sym_key → strategy_type

    total_inserted = 0

    for log_file in log_files:
        text = log_file.read_text(errors="replace")

        # Build entry map from this file
        for m in _OPENED_RE.finditer(text):
            sym_raw = m.group("sym_raw")
            sym, direction = _parse_symbol(sym_raw)
            key = f"{sym}:{direction or m.group('dir')}"
            entry_map[key] = _ts_to_iso(m.group("ts"))
            strategy_map[key] = m.group("strategy")

        # Parse CLOSED lines
        for m in _CLOSED_RE.finditer(text):
            sym_raw = m.group("sym_raw")
            sym, _pos_side = _parse_symbol(sym_raw)
            direction = m.group("dir")
            key = f"{sym}:{direction}"

            exit_time = _ts_to_iso(m.group("ts"))
            trade = {
                "symbol": sym,
                "signal": 1 if direction == "LONG" else -1,
                "entry_price": None,
                "exit_price": None,
                "pnl_pct": float(m.group("pnl_pct")),
                "pnl_usd": float(m.group("pnl_usd")),
                "notional": None,
                "bars_held": int(m.group("bars")) if m.group("bars") else None,
                "reason": m.group("reason"),
                "time": exit_time,
                "strategy_type": strategy_map.get(key, "vr_squeeze"),
                "mfe_pct": float(m.group("mfe")) if m.group("mfe") else None,
                "mae_pct": float(m.group("mae")) if m.group("mae") else None,
                "entry_time": entry_map.get(key),
                "balance_before": None,
                "balance_after": None,
            }

            if dry_run:
                print(f"  [DRY] {sym} {direction} {m.group('reason')} {m.group('pnl_usd')} @ {exit_time}")
            else:
                journal.record(trade)
                total_inserted += 1

    if not dry_run:
        print(f"\n✓ Backfill complete: {total_inserted} record(s) processed (duplicates ignored)")
    else:
        print(f"\n[DRY RUN] {total_inserted} record(s) would be inserted")

    return total_inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill trade journal from log files")
    parser.add_argument("--db", default="data/journal.sqlite", help="SQLite DB path")
    parser.add_argument("--log-dir", default="logs/bot/vol_straddle", help="Log directory")
    parser.add_argument("--session", default="live", choices=["live", "paper"])
    parser.add_argument("--dry-run", action="store_true", help="Print records without inserting")
    args = parser.parse_args()

    backfill(log_dir=args.log_dir, db_path=args.db, session=args.session, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
