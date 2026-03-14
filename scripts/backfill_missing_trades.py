"""
Backfill 3 missing trades into journal.sqlite.

Root causes:
- SUSHIUSDT: server-side algo SL triggered at 22:03 UTC 03-13;
  session detected but journal INSERT may have been missed.
- BANDUSDT: reverse ghost (Incident #6) reconciliation forced close;
  reconciliation.py force-close path didn't call journal.record().
- CRVUSDT: session gap during v14.x upgrade — position was tracked
  but journal was not initialised at the time of close.

Run: python3 scripts/backfill_missing_trades.py
"""
import sqlite3
from pathlib import Path

DB = Path("data/journal.sqlite")

MISSING_TRADES = [
    # ── SUSHIUSDT LONG (ma_cluster) ──────────────────────────────────────
    # Entry: 03-13 04:00 UTC, Exit: 03-13 22:03 UTC via server algo SL
    # Binance fills confirmed: 13854 qty, entry=0.2134, exit=0.2090
    # Total realizedPnL = -60.69 (sum of all exit fills)
    {
        "symbol":        "SUSHIUSDT",
        "strategy_type": "ma_cluster",
        "direction":     "LONG",
        "entry_price":   0.2134,
        "exit_price":    0.2090,
        "pnl_pct":       (0.2090 - 0.2134) / 0.2134 * 100,   # -2.062%
        "pnl_usd":       -60.69,
        "mfe_pct":       None,   # peak_gain not recoverable
        "mae_pct":       None,
        "notional":      2956.44,
        "bars_held":     72,
        "entry_time":    "2026-03-13T04:00:21+00:00",
        "exit_time":     "2026-03-13T22:03:28+00:00",
        "exit_reason":   "server_sl",
        "balance_before": None,
        "balance_after":  None,
        "session":       "manual_backfill",
        "commit_sha":    "7a07658",
    },
    # ── BANDUSDT LONG (reconciliation close — ghost position) ─────────────
    # Incident #6: position opened but not in state → reconciliation forced
    # market close → journal.record() not called (known gap in reconciliation.py)
    # Binance fills: entry 11:45, exit 14:25, +$10.36
    {
        "symbol":        "BANDUSDT",
        "strategy_type": "unknown",    # ghost — strategy unknown
        "direction":     "LONG",
        "entry_price":   0.2348,
        "exit_price":    0.2410,
        "pnl_pct":       (0.2410 - 0.2348) / 0.2348 * 100,   # +2.641%
        "pnl_usd":       10.36,
        "mfe_pct":       None,
        "mae_pct":       None,
        "notional":      392.43,
        "bars_held":     11,
        "entry_time":    "2026-03-13T11:45:21+00:00",
        "exit_time":     "2026-03-13T14:25:35+00:00",
        "exit_reason":   "reconciliation_close",
        "balance_before": None,
        "balance_after":  None,
        "session":       "manual_backfill",
        "commit_sha":    "7a07658",
    },
    # ── CRVUSDT SHORT (vr_squeeze) ────────────────────────────────────────
    # Session gap during upgrade: position entered 03-12 04:00, SL triggered
    # 03-12 09:44 (6h hold, 23 bars). Binance fills confirm -$65.33.
    {
        "symbol":        "CRVUSDT",
        "strategy_type": "vr_squeeze",
        "direction":     "SHORT",
        "entry_price":   0.2300,
        "exit_price":    0.2350,
        "pnl_pct":       (0.2300 - 0.2350) / 0.2300 * 100,   # -2.174%
        "pnl_usd":       -65.33,
        "mfe_pct":       None,
        "mae_pct":       None,
        "notional":      3005.18,
        "bars_held":     23,
        "entry_time":    "2026-03-12T04:00:34+00:00",
        "exit_time":     "2026-03-12T09:44:07+00:00",
        "exit_reason":   "server_sl",
        "balance_before": None,
        "balance_after":  None,
        "session":       "manual_backfill",
        "commit_sha":    "7a07658",
    },
]


def main() -> None:
    if not DB.exists():
        print(f"ERROR: {DB} not found.")
        return

    conn = sqlite3.connect(DB)
    inserted = 0
    skipped = 0

    for t in MISSING_TRADES:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO trades
                   (symbol, strategy_type, direction, entry_price, exit_price,
                    pnl_pct, pnl_usd, mfe_pct, mae_pct, notional, bars_held,
                    entry_time, exit_time, exit_reason,
                    balance_before, balance_after, session, commit_sha)
                   VALUES
                   (:symbol, :strategy_type, :direction, :entry_price, :exit_price,
                    :pnl_pct, :pnl_usd, :mfe_pct, :mae_pct, :notional, :bars_held,
                    :entry_time, :exit_time, :exit_reason,
                    :balance_before, :balance_after, :session, :commit_sha)""",
                t,
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                print(f"  ✅ Inserted: {t['symbol']} {t['direction']} "
                      f"exit={t['exit_time']} pnl_usd=${t['pnl_usd']:.2f}")
                inserted += 1
            else:
                print(f"  ⏭  Skipped (already exists): {t['symbol']} exit={t['exit_time']}")
                skipped += 1
        except Exception as e:
            print(f"  ❌ Error inserting {t['symbol']}: {e}")

    conn.commit()
    conn.close()

    total = conn.execute("SELECT count(*) FROM trades").fetchone()[0] if False else None
    print(f"\nDone. Inserted={inserted} Skipped={skipped}")
    print("Verify with: python3 -c \"import sqlite3; conn=sqlite3.connect('data/journal.sqlite'); "
          "print(conn.execute('SELECT count(*) FROM trades').fetchone())\"")


if __name__ == "__main__":
    main()
