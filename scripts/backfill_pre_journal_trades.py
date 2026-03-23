"""Backfill live trades from March 1–7, 2026 that predate journal initialization.

The journal.sqlite was initialized on 2026-03-08. These trades were executed live
on Binance USDT-M perpetuals but never recorded. Reconstructed from Binance fill API.

Notes:
- strategy_type='vr_squeeze' for directional trades (estimated; session state unavailable)
- ETCUSDT on 2026-03-07 was a straddle (LONG+SHORT simultaneously)
- Trades closing at 2026-03-07T19:30 were mass-closed by drawdown_halt event
- TRBUSDT: income API shows -$60.92 but fills predate 2026-03-01 (cannot reconstruct)
- STXUSDT had two closes; only the significant one (-$68.65) is included
- Safe to re-run: INSERT OR IGNORE on (symbol, exit_time) unique index
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).parent.parent / "data" / "journal.sqlite"
COMMIT = "7c0eb41"

TRADES = [
    dict(symbol="WLDUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=0.4322, exit_price=0.4243,
         pnl_pct=(0.4243 - 0.4322) / 0.4322 * 100, pnl_usd=-219.96,
         mfe_pct=None, mae_pct=None, notional=11888.0, bars_held=2,
         entry_time="2026-03-04T16:45:00+00:00", exit_time="2026-03-04T17:19:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="LTCUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=57.49, exit_price=56.34,
         pnl_pct=(56.34 - 57.49) / 57.49 * 100, pnl_usd=-73.66,
         mfe_pct=None, mae_pct=None, notional=3682.0, bars_held=31,
         entry_time="2026-03-04T20:00:00+00:00", exit_time="2026-03-05T03:39:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="AVAXUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=9.448, exit_price=9.256,
         pnl_pct=(9.256 - 9.448) / 9.448 * 100, pnl_usd=-43.17,
         mfe_pct=None, mae_pct=None, notional=2116.0, bars_held=56,
         entry_time="2026-03-05T04:00:00+00:00", exit_time="2026-03-05T17:55:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="ETCUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=8.8074, exit_price=8.628,
         pnl_pct=(8.628 - 8.8074) / 8.8074 * 100, pnl_usd=-43.17,
         mfe_pct=None, mae_pct=None, notional=2110.0, bars_held=16,
         entry_time="2026-03-05T12:00:00+00:00", exit_time="2026-03-05T16:04:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="AAVEUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=118.39, exit_price=116.01,
         pnl_pct=(116.01 - 118.39) / 118.39 * 100, pnl_usd=-38.08,
         mfe_pct=None, mae_pct=None, notional=1894.0, bars_held=6,
         entry_time="2026-03-06T08:00:00+00:00", exit_time="2026-03-06T09:34:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="DOGEUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=0.0935, exit_price=0.0934,
         pnl_pct=(0.0934 - 0.0935) / 0.0935 * 100, pnl_usd=-1.53,
         mfe_pct=None, mae_pct=None, notional=4757.0, bars_held=45,
         entry_time="2026-03-06T01:15:00+00:00", exit_time="2026-03-06T12:23:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="STXUSDT", strategy_type="vr_directional", direction="SHORT",
         entry_price=0.2646, exit_price=0.27,
         pnl_pct=(0.2646 - 0.27) / 0.2646 * 100, pnl_usd=-68.65,
         mfe_pct=None, mae_pct=None, notional=3346.0, bars_held=30,
         entry_time="2026-03-06T00:00:00+00:00", exit_time="2026-03-06T07:31:00+00:00",
         exit_reason="server_sl"),

    # ETC straddle pair on March 7 (LONG+SHORT open simultaneously)
    dict(symbol="ETCUSDT", strategy_type="vr_straddle", direction="LONG",
         entry_price=8.2323, exit_price=8.108,
         pnl_pct=(8.108 - 8.2323) / 8.2323 * 100, pnl_usd=-101.27,
         mfe_pct=None, mae_pct=None, notional=6583.0, bars_held=61,
         entry_time="2026-03-07T00:30:00+00:00", exit_time="2026-03-07T15:39:00+00:00",
         exit_reason="server_sl"),

    dict(symbol="ETCUSDT", strategy_type="vr_straddle", direction="SHORT",
         entry_price=8.2284, exit_price=8.107,
         pnl_pct=(8.2284 - 8.107) / 8.2284 * 100, pnl_usd=97.04,
         mfe_pct=None, mae_pct=None, notional=6579.0, bars_held=76,
         entry_time="2026-03-07T00:30:00+00:00", exit_time="2026-03-07T19:30:00+00:00",
         exit_reason="drawdown_halt"),

    dict(symbol="SUIUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=0.9015, exit_price=0.8977,
         pnl_pct=(0.8977 - 0.9015) / 0.9015 * 100, pnl_usd=-27.78,
         mfe_pct=None, mae_pct=None, notional=6590.0, bars_held=78,
         entry_time="2026-03-06T23:45:00+00:00", exit_time="2026-03-07T19:15:00+00:00",
         exit_reason="drawdown_halt"),

    dict(symbol="COMPUSDT", strategy_type="vr_directional", direction="SHORT",
         entry_price=17.5143, exit_price=17.31,
         pnl_pct=(17.5143 - 17.31) / 17.5143 * 100, pnl_usd=35.23,
         mfe_pct=None, mae_pct=None, notional=3074.0, bars_held=110,
         entry_time="2026-03-06T16:00:00+00:00", exit_time="2026-03-07T19:30:00+00:00",
         exit_reason="drawdown_halt"),

    dict(symbol="AAVEUSDT", strategy_type="vr_directional", direction="LONG",
         entry_price=109.39, exit_price=108.74,
         pnl_pct=(108.74 - 109.39) / 109.39 * 100, pnl_usd=-14.53,
         mfe_pct=None, mae_pct=None, notional=2418.0, bars_held=15,
         entry_time="2026-03-07T15:45:00+00:00", exit_time="2026-03-07T19:30:00+00:00",
         exit_reason="drawdown_halt"),
]


def run() -> None:
    conn = sqlite3.connect(DB)
    inserted = skipped = 0
    for t in TRADES:
        conn.execute(
            """INSERT OR IGNORE INTO trades
               (symbol,strategy_type,direction,entry_price,exit_price,pnl_pct,pnl_usd,
                mfe_pct,mae_pct,notional,bars_held,entry_time,exit_time,exit_reason,
                balance_before,balance_after,session,commit_sha)
               VALUES (:symbol,:strategy_type,:direction,:entry_price,:exit_price,:pnl_pct,:pnl_usd,
                :mfe_pct,:mae_pct,:notional,:bars_held,:entry_time,:exit_time,:exit_reason,
                NULL,NULL,'pre_journal_backfill',:commit_sha)""",
            {**t, "commit_sha": COMMIT},
        )
        if conn.execute("SELECT changes()").fetchone()[0]:
            print(f"  ✅ {t['symbol']} {t['direction']} {t['exit_time'][:16]} pnl=${t['pnl_usd']:+.2f}")
            inserted += 1
        else:
            print(f"  ⏭  skip {t['symbol']} {t['direction']} {t['exit_time'][:16]}")
            skipped += 1
    conn.commit()
    n = conn.execute("SELECT count(*) FROM trades WHERE session='pre_journal_backfill'").fetchone()[0]
    print(f"\nInserted={inserted} Skipped={skipped}  Total pre_journal_backfill: {n}")


if __name__ == "__main__":
    run()
