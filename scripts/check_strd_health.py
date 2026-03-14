#!/usr/bin/env python3
"""STRD cliff risk monitor — checks live WR/PF against alert thresholds.

Usage:
    python3 scripts/check_strd_health.py

Alerts:
    - WR < 55% (last 20 STRD trades)
    - PF < 2.0 (last 50 STRD trades)
    - 3+ consecutive losses

Background: s_trail_activate=0.015 is a sharp cliff peak (r77/r81).
±0.001 change → PF drops 34%. Monthly monitoring essential.
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "journal.sqlite"

# Alert thresholds (from plan T6)
WR_ALERT = 0.55          # WR below this → warning
PF_ALERT = 2.0           # PF below this → warning
WR_WINDOW = 20           # recent trades for WR check
PF_WINDOW = 50           # recent trades for PF check
CONSEC_LOSS_ALERT = 3    # consecutive losses → warning


def main() -> None:
    if not DB_PATH.exists():
        print("❌ journal.sqlite not found")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # All STRD trades, most recent first
    cur.execute("""
        SELECT symbol, direction, pnl_pct, pnl_usd, exit_reason,
               entry_time, exit_time, bars_held, mfe_pct, mae_pct
        FROM trades
        WHERE strategy_type = 'straddle'
        ORDER BY exit_time DESC
    """)
    rows = cur.fetchall()
    n_total = len(rows)

    if n_total == 0:
        print("ℹ️  No STRD trades in journal yet.")
        return

    print(f"=== STRD Health Monitor ({n_total} total trades) ===\n")

    # --- Recent trades table ---
    print(f"{'Symbol':<12} {'Dir':<6} {'PnL%':>8} {'PnL$':>8} {'MFE%':>7} {'MAE%':>7} {'Bars':>5} {'Exit':<12} {'Time'}")
    print("-" * 100)
    for r in rows[:WR_WINDOW]:
        pnl_pct = r["pnl_pct"] if r["pnl_pct"] else 0.0
        # DB stores pnl_pct as percentage (e.g., -1.37 = -1.37%)
        # Values >100 are likely basis points from older entries
        if abs(pnl_pct) > 100:
            pnl_pct /= 100  # basis points → percent
        pnl_usd = r["pnl_usd"] if r["pnl_usd"] else 0
        mfe = r["mfe_pct"] if r["mfe_pct"] else 0.0
        mae = r["mae_pct"] if r["mae_pct"] else 0.0
        bars = r["bars_held"] if r["bars_held"] else 0
        exit_r = r["exit_reason"] or ""
        exit_t = (r["exit_time"] or "")[:16]
        print(f"{r['symbol']:<12} {r['direction'] or '':<6} {pnl_pct:>+7.2f}% {pnl_usd:>+8.2f} {mfe:>+6.2f}% {mae:>+6.2f}% {bars:>5} {exit_r:<12} {exit_t}")

    # --- WR check (last WR_WINDOW) ---
    recent = rows[:WR_WINDOW]
    wins = sum(1 for r in recent if (r["pnl_usd"] or 0) > 0)
    wr = wins / len(recent) if recent else 0

    print(f"\n--- WR (last {min(len(recent), WR_WINDOW)} trades) ---")
    print(f"  Wins: {wins}/{len(recent)} = {wr:.1%}")
    if wr < WR_ALERT:
        print(f"  ⚠️  WR {wr:.1%} < {WR_ALERT:.0%} threshold — cliff edge risk!")
    else:
        print(f"  ✅ WR OK (≥ {WR_ALERT:.0%})")

    # --- PF check (last PF_WINDOW) ---
    pf_trades = rows[:PF_WINDOW]
    gross_win = sum((r["pnl_usd"] or 0) for r in pf_trades if (r["pnl_usd"] or 0) > 0)
    gross_loss = abs(sum((r["pnl_usd"] or 0) for r in pf_trades if (r["pnl_usd"] or 0) < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n--- PF (last {min(len(pf_trades), PF_WINDOW)} trades) ---")
    print(f"  Gross Win: ${gross_win:.2f} | Gross Loss: ${gross_loss:.2f}")
    print(f"  PF: {pf:.2f}")
    if pf < PF_ALERT:
        print(f"  ⚠️  PF {pf:.2f} < {PF_ALERT:.1f} threshold — review trail params!")
    else:
        print(f"  ✅ PF OK (≥ {PF_ALERT:.1f})")

    # --- Consecutive losses ---
    consec = 0
    max_consec = 0
    for r in rows:
        if (r["pnl_usd"] or 0) < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    # Current streak (from most recent)
    current_streak = 0
    for r in rows:
        if (r["pnl_usd"] or 0) < 0:
            current_streak += 1
        else:
            break

    print(f"\n--- Loss Streaks ---")
    print(f"  Current: {current_streak} consecutive losses")
    print(f"  Max ever: {max_consec}")
    if current_streak >= CONSEC_LOSS_ALERT:
        print(f"  ⚠️  {current_streak} consecutive losses ≥ {CONSEC_LOSS_ALERT} — monitor closely!")
    else:
        print(f"  ✅ Streak OK (< {CONSEC_LOSS_ALERT})")

    # --- Summary ---
    total_pnl = sum((r["pnl_usd"] or 0) for r in rows)
    avg_pnl = total_pnl / n_total if n_total else 0

    print(f"\n--- Lifetime STRD Summary ---")
    print(f"  Total PnL: ${total_pnl:+.2f} ({n_total} trades)")
    print(f"  Avg PnL/trade: ${avg_pnl:+.2f}")
    total_wr = sum(1 for r in rows if (r["pnl_usd"] or 0) > 0) / n_total
    print(f"  Lifetime WR: {total_wr:.1%}")

    # --- Action recommendations ---
    alerts = []
    if wr < WR_ALERT:
        alerts.append("WR below threshold")
    if pf < PF_ALERT:
        alerts.append("PF below threshold")
    if current_streak >= CONSEC_LOSS_ALERT:
        alerts.append(f"{current_streak} consecutive losses")

    if alerts:
        print(f"\n🚨 ACTION NEEDED: {', '.join(alerts)}")
        print("   Consider: straddle_max_slots 8→6 or 4 (gradual reduction)")
        print("   Review: s_trail_activate cliff edge (r77/r81 analysis)")
    else:
        print("\n✅ STRD health OK — no action needed.")

    conn.close()


if __name__ == "__main__":
    main()
