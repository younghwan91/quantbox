#!/usr/bin/env python3
"""r117: Donchian backtest-live parity audit.

Live Donchian (N=9) shows PF 0.75, vs backtest PF 4.30 — 5.7x divergence.
This script extracts live trades from journal.sqlite and replays each
through TradeEventSim to identify the divergence source.

Methodology:
1. Load live Donchian trades from data/journal.sqlite.
2. For each trade, identify the 4h bar index corresponding to entry_time.
3. Run sim_donchian_trailing on the symbol's OHLCV series.
4. Find the sim trade that fires on or near the same entry bar.
5. Compare live vs sim: exit_reason, bars_held, pnl_pct.
6. Classify divergence: NO_SIM_SIGNAL / SAME_EXIT / DIFFERENT_EXIT.

Outputs:
    .omc/state/r117/donchian_parity.json — per-trade delta detail
    docs/backtest_results/202604/<date>_<commit>_donchian_parity.md — report

Decision rule (PAUSE Donchian if):
    |avg(delta_pnl_pct)| > 2.0%
    AND
    worst-3 live trades all show live_pnl_pct < sim_pnl_pct - 3.0%

(ref: r116 plan r117 follow-up, US-URGENT-1)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import os

from dotenv import load_dotenv

from quantbox.backtest.trade_sim import build_resampled, sim_donchian_trailing
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from research.active.r116_willy_nfi.r116_prep_baseline import load_15m_data

OUTPUT_DIR = ROOT / ".omc" / "state" / "r117"
JOURNAL_DB = ROOT / "data" / "journal.sqlite"

# Tolerance for bar-index matching: live entry may fire up to 1 bar offset
# from sim due to process-level vs bar-close timing.
BAR_MATCH_TOLERANCE = 1

# Decision thresholds (see module docstring)
PAUSE_AVG_DELTA_THRESHOLD = 2.0  # %
PAUSE_WORST3_DELTA_THRESHOLD = 3.0  # %


def fetch_15m_klines_to_extend(
    symbols: list[str],
    from_ts: datetime,
    to_ts: datetime,
) -> dict[str, dict]:
    """Fetch 15m OHLCV from Binance for the live window and return as
    a data dict compatible with load_15m_data schema (close/high/low/volume arrays).

    Overlap with cache is expected but harmless — we'll concat then dedup.
    """
    load_dotenv()
    from binance.client import Client

    client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_API_SECRET"])
    start_ms = int(from_ts.timestamp() * 1000)
    end_ms = int(to_ts.timestamp() * 1000)

    out: dict[str, dict] = {}
    for sym in symbols:
        klines = []
        cursor = start_ms
        while cursor < end_ms:
            batch = client.futures_klines(
                symbol=sym,
                interval="15m",
                startTime=cursor,
                endTime=end_ms,
                limit=1000,
            )
            if not batch:
                break
            klines.extend(batch)
            last_open = batch[-1][0]
            if last_open <= cursor:
                break
            cursor = last_open + 15 * 60 * 1000
        if not klines:
            continue
        # kline format: [open_time, open, high, low, close, volume, ...]
        c_arr = np.array([float(k[4]) for k in klines], dtype=np.float64)
        h_arr = np.array([float(k[2]) for k in klines], dtype=np.float64)
        l_arr = np.array([float(k[3]) for k in klines], dtype=np.float64)
        v_arr = np.array([float(k[5]) for k in klines], dtype=np.float64)
        ts_arr = np.array([int(k[0]) for k in klines], dtype=np.int64)
        out[sym] = {
            "close": c_arr,
            "high": h_arr,
            "low": l_arr,
            "volume": v_arr,
            "open_ms": ts_arr,
        }
    return out


def extend_data_15m(
    data_15m: dict,
    fresh: dict,
    cache_end: datetime,
) -> dict:
    """Append fresh klines (post-cache_end) to existing data_15m arrays.

    Only keeps fresh bars whose open_time > cache_end to avoid duplicate rows.
    """
    cache_end_ms = int(cache_end.timestamp() * 1000)
    for sym, f in fresh.items():
        mask = f["open_ms"] > cache_end_ms
        if not mask.any():
            continue
        c_new = f["close"][mask]
        h_new = f["high"][mask]
        l_new = f["low"][mask]
        v_new = f["volume"][mask]
        if sym not in data_15m:
            data_15m[sym] = {
                "close": c_new, "high": h_new, "low": l_new, "volume": v_new,
                "bars": len(c_new), "days": len(c_new) / (4 * 24),
            }
        else:
            d = data_15m[sym]
            d["close"] = np.concatenate([d["close"], c_new])
            d["high"] = np.concatenate([d["high"], h_new])
            d["low"] = np.concatenate([d["low"], l_new])
            d["volume"] = np.concatenate([d["volume"], v_new])
            d["bars"] = len(d["close"])
            d["days"] = d["bars"] / (4 * 24)
    return data_15m


def load_live_donchian_trades() -> list[dict]:
    """Pull all Donchian trades from journal.sqlite."""
    conn = sqlite3.connect(JOURNAL_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """SELECT symbol, direction, entry_price, exit_price, pnl_pct, pnl_usd,
                  bars_held, exit_reason, entry_time, exit_time, notional
           FROM trades
           WHERE strategy_type='donchian'
           ORDER BY entry_time""",
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def bar_index_from_time(
    entry_ts: datetime,
    data_4h: dict,
    symbol: str,
    effective_end: datetime,
) -> int:
    """Return the 4h bar index corresponding to entry_ts.

    `effective_end` is the timestamp of the LAST 4h bar in data_4h (post-extension).
    Each 4h bar covers 4 hours. Index is computed by counting 4h steps back
    from effective_end.
    """
    if entry_ts > effective_end:
        return -1
    if symbol not in data_4h:
        return -1
    n_4h = len(data_4h[symbol]["close"])
    seconds_from_end = (effective_end - entry_ts).total_seconds()
    bars_from_end = int(seconds_from_end / (4 * 3600))
    idx = n_4h - 1 - bars_from_end
    if idx < 0 or idx >= n_4h:
        return -1
    return idx


def run_sim_donchian_for_symbol(
    cfg: VolStraddleConfig,
    data_4h_sym: dict,
) -> list[tuple[int, int, float]]:
    """Return list of (entry_bar_4h, exit_bar_4h, pnl_pct) for Donchian
    sim-trades on a single symbol's 4h OHLCV.
    """
    c = data_4h_sym["close"]
    h = data_4h_sym["high"]
    lo = data_4h_sym["low"]
    vol = data_4h_sym["volume"]
    warmup = cfg.donchian_lookback + cfg.donchian_vol_sma
    pnl_arr, ent_arr, ext_arr = sim_donchian_trailing(
        c, h, lo, vol,
        cfg.donchian_lookback,
        cfg.donchian_vol_sma,
        cfg.donchian_vol_mult,
        cfg.donchian_breakout_pct,
        cfg.donchian_cooldown_bars,
        cfg.donchian_sl,
        cfg.donchian_tp,
        cfg.donchian_trail_activate,
        cfg.donchian_trail_distance,
        cfg.donchian_max_hold_4h,
        cfg.round_trip_cost,
        warmup,
    )
    return [(int(e), int(x), float(p)) for p, e, x in zip(pnl_arr, ent_arr, ext_arr)]


def find_matching_sim_trade(
    sim_trades: list[tuple[int, int, float]],
    live_entry_bar: int,
) -> tuple[int, int, float] | None:
    """Find sim-trade whose entry_bar is within ±BAR_MATCH_TOLERANCE of live."""
    if live_entry_bar < 0:
        return None
    for st in sim_trades:
        if abs(st[0] - live_entry_bar) <= BAR_MATCH_TOLERANCE:
            return st
    return None


def classify(live: dict, sim: tuple[int, int, float] | None) -> str:
    """Classify divergence type."""
    if sim is None:
        if live.get("_out_of_cache"):
            return "OUT_OF_CACHE"
        return "NO_SIM_SIGNAL"
    live_pnl = float(live["pnl_pct"])
    sim_pnl = sim[2] * 100.0  # sim returns fraction; convert to %
    if abs(live_pnl - sim_pnl) < 0.3:
        return "IDENTICAL"
    if (live_pnl < 0) == (sim_pnl < 0):
        return "SAME_SIGN_DIFFERENT_MAG"
    return "OPPOSITE_SIGN"


def build_per_trade_row(live: dict, sim: tuple[int, int, float] | None) -> dict:
    live_pnl = float(live["pnl_pct"])
    sim_pnl = sim[2] * 100.0 if sim else None
    delta = (live_pnl - sim_pnl) if sim_pnl is not None else None
    sim_bars = (sim[1] - sim[0]) if sim else None
    return {
        "symbol": live["symbol"],
        "entry_time": live["entry_time"],
        "direction": live["direction"],
        "live_pnl_pct": round(live_pnl, 3),
        "live_pnl_usd": round(float(live["pnl_usd"]), 2),
        "live_bars_held": live["bars_held"],
        "live_exit_reason": live["exit_reason"],
        "live_notional": round(float(live["notional"]), 2),
        "sim_pnl_pct": round(sim_pnl, 3) if sim_pnl is not None else None,
        "sim_bars_held_4h": sim_bars,
        "delta_pct": round(delta, 3) if delta is not None else None,
        "classification": classify(live, sim),
    }


def compute_aggregates(rows: list[dict], live_trades: list[dict]) -> dict:
    live_deltas = [r["delta_pct"] for r in rows if r["delta_pct"] is not None]
    sim_pnls = [r["sim_pnl_pct"] for r in rows if r["sim_pnl_pct"] is not None]
    live_pnls = [r["live_pnl_pct"] for r in rows]

    live_gw = sum(max(0, p["pnl_usd"]) for p in live_trades)
    live_gl = sum(max(0, -p["pnl_usd"]) for p in live_trades)
    live_pf = live_gw / live_gl if live_gl > 0 else float("inf")

    sim_wins = [p for p in sim_pnls if p > 0]
    sim_losses = [p for p in sim_pnls if p < 0]
    sim_pf = sum(sim_wins) / abs(sum(sim_losses)) if sim_losses else float("inf")

    return {
        "n_live": len(live_trades),
        "n_matched_sim": sum(1 for r in rows if r["sim_pnl_pct"] is not None),
        "n_out_of_cache": sum(1 for r in rows if r["classification"] == "OUT_OF_CACHE"),
        "n_no_sim_signal": sum(1 for r in rows if r["classification"] == "NO_SIM_SIGNAL"),
        "n_identical": sum(1 for r in rows if r["classification"] == "IDENTICAL"),
        "n_diff_mag": sum(1 for r in rows if r["classification"] == "SAME_SIGN_DIFFERENT_MAG"),
        "n_opposite": sum(1 for r in rows if r["classification"] == "OPPOSITE_SIGN"),
        "live_pf_usd": round(live_pf, 3),
        "live_avg_pnl_pct": round(float(np.mean(live_pnls)), 3) if live_pnls else 0.0,
        "sim_pf_matched": round(sim_pf, 3) if sim_losses else None,
        "sim_avg_pnl_pct": round(float(np.mean(sim_pnls)), 3) if sim_pnls else None,
        "delta_avg_pct": round(float(np.mean(live_deltas)), 3) if live_deltas else None,
        "delta_worst3_avg_pct": (
            round(float(np.mean(sorted(live_deltas)[:3])), 3)
            if len(live_deltas) >= 3 else None
        ),
    }


def make_recommendation(agg: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if agg["delta_avg_pct"] is None:
        return "INSUFFICIENT_DATA", [
            "No matched sim trades — cache window (2024-01 to 2026-04-12) predates live Donchian activity.",
            "Cannot audit parity without overlapping data. Extend cache to cover live period first.",
        ]

    pause_avg = abs(agg["delta_avg_pct"]) > PAUSE_AVG_DELTA_THRESHOLD
    pause_worst3 = (
        agg["delta_worst3_avg_pct"] is not None
        and agg["delta_worst3_avg_pct"] < -PAUSE_WORST3_DELTA_THRESHOLD
    )
    if pause_avg and pause_worst3:
        reasons.append(f"avg(delta_pnl_pct)={agg['delta_avg_pct']}% exceeds {PAUSE_AVG_DELTA_THRESHOLD}% threshold.")
        reasons.append(
            f"worst-3 live trades avg delta {agg['delta_worst3_avg_pct']}% "
            f"is below {-PAUSE_WORST3_DELTA_THRESHOLD}% threshold."
        )
        return "PAUSE_DONCHIAN", reasons
    if pause_avg:
        reasons.append(
            f"avg delta {agg['delta_avg_pct']}% exceeds threshold but worst-3 does not (single-trade skew)."
        )
    if pause_worst3:
        reasons.append(
            f"worst-3 delta {agg['delta_worst3_avg_pct']}% bad, but aggregate is within bounds."
        )
    if not (pause_avg or pause_worst3):
        reasons.append(
            f"avg delta {agg['delta_avg_pct']}% within ±{PAUSE_AVG_DELTA_THRESHOLD}% threshold."
        )
    reasons.append(
        f"Matched {agg['n_matched_sim']}/{agg['n_live']} trades; "
        f"N matched too small for conclusive pause decision (recommend collect more)."
    )
    return "KEEP_RUNNING", reasons


def write_report(
    rows: list[dict],
    agg: dict,
    recommendation: str,
    reasons: list[str],
    out_path: Path,
) -> None:
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    ap = lines.append
    ap(f"# r117 Donchian Backtest-Live Parity Audit")
    ap("")
    ap(f"**Commit:** {commit}")
    ap(f"**Date:** {date_str}")
    ap(f"**Source:** data/journal.sqlite (live trades) vs sim_donchian_trailing (backtest)")
    ap(f"**Scope:** All Donchian live trades to date (N={agg['n_live']}).")
    ap("")
    ap("## 1. Executive Summary")
    ap("")
    ap(f"**Recommendation: {recommendation}**")
    ap("")
    for r in reasons:
        ap(f"- {r}")
    ap("")
    ap("## 2. Aggregate Comparison")
    ap("")
    ap("| Metric | Live | Sim (matched) | Delta |")
    ap("|--------|------|---------------|-------|")
    ap(f"| N | {agg['n_live']} | {agg['n_matched_sim']} | — |")
    ap(f"| PF (USD) | {agg['live_pf_usd']} | {agg['sim_pf_matched']} | — |")
    ap(f"| Avg PnL % | {agg['live_avg_pnl_pct']} | {agg['sim_avg_pnl_pct']} | {agg['delta_avg_pct']} |")
    ap(f"| Worst-3 avg delta % | — | — | {agg['delta_worst3_avg_pct']} |")
    ap("")
    ap("### Classification counts")
    ap("")
    ap(f"- IDENTICAL (|delta| < 0.3%): {agg['n_identical']}")
    ap(f"- SAME_SIGN_DIFFERENT_MAG: {agg['n_diff_mag']}")
    ap(f"- OPPOSITE_SIGN: {agg['n_opposite']}")
    ap(f"- NO_SIM_SIGNAL (live fired, sim did not): {agg['n_no_sim_signal']}")
    ap(f"- OUT_OF_CACHE (live entry after cache window end): {agg['n_out_of_cache']}")
    ap("")
    ap("## 3. Per-Trade Detail")
    ap("")
    ap("| Symbol | Entry | Dir | Live PnL% | Live USD | Live reason | Sim PnL% | Delta% | Class |")
    ap("|--------|-------|-----|-----------|----------|-------------|----------|--------|-------|")
    for r in rows:
        ap(
            f"| {r['symbol']} | {r['entry_time'][:16]} | {r['direction']} | "
            f"{r['live_pnl_pct']} | ${r['live_pnl_usd']} | {r['live_exit_reason']} | "
            f"{r['sim_pnl_pct'] if r['sim_pnl_pct'] is not None else '—'} | "
            f"{r['delta_pct'] if r['delta_pct'] is not None else '—'} | {r['classification']} |"
        )
    ap("")
    ap("## 4. Methodology Notes")
    ap("")
    ap(
        "- Sim trades matched to live by entry_bar index within "
        f"±{BAR_MATCH_TOLERANCE} 4h-bar tolerance (accounts for live-vs-sim "
        "bar-close timing micro-skew)."
    )
    ap(
        "- Live entry_time beyond cache WINDOW_END (2026-04-12T16:18 UTC) "
        "cannot be replayed without cache extension — flagged OUT_OF_CACHE."
    )
    ap("- Sim uses `sim_donchian_trailing` with cfg values from VolStraddleConfig (no hardcoded params).")
    ap(
        f"- Pause thresholds: |avg_delta|>{PAUSE_AVG_DELTA_THRESHOLD}% AND "
        f"worst-3 avg < -{PAUSE_WORST3_DELTA_THRESHOLD}% (both conditions must fire)."
    )
    ap("")
    ap("## 5. Follow-ups")
    ap("")
    if agg["n_out_of_cache"] == agg["n_live"]:
        ap(
            "- **Blocker:** All live Donchian trades are out-of-cache (live activity "
            "post-dates 2026-04-12 cache). Parity audit requires cache refresh to "
            "cover live window. File a follow-up r117a for cache extension before "
            "any PAUSE decision."
        )
    ap(
        "- Regardless of parity finding, live Donchian N=9 is too small for "
        "statistical confidence. Accumulate 20+ trades before drawing hard conclusions."
    )
    ap("- If recommendation is KEEP_RUNNING, revisit audit when live N ≥ 20.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    print("=" * 78)
    print("r117 — Donchian Backtest-Live Parity Audit")
    print("=" * 78)

    from research.active.r116_willy_nfi.r116_prep_baseline import (
        WINDOW_END as CACHE_END,
    )

    cfg = VolStraddleConfig()
    live_trades = load_live_donchian_trades()
    print(f"Loaded {len(live_trades)} live Donchian trades from journal.sqlite")
    if not live_trades:
        print("NO live trades — exiting (nothing to audit)")
        return

    # Load data for all symbols that have live trades (+ BTCUSDT for consistency)
    syms_needed = sorted({t["symbol"] for t in live_trades})
    print(f"Loading 15m data for {len(syms_needed)} symbols...")
    data_15m = load_15m_data(syms_needed)

    # Determine if extension is needed (any live trade beyond cache_end)
    max_exit_ts = max(parse_iso(t["exit_time"]) for t in live_trades)
    effective_end = CACHE_END
    if max_exit_ts > CACHE_END:
        fetch_until = max_exit_ts + timedelta(days=1)
        print(
            f"Live window extends beyond cache (max exit {max_exit_ts.isoformat()}). "
            f"Fetching fresh 15m klines for {len(syms_needed)} symbols..."
        )
        fresh = fetch_15m_klines_to_extend(syms_needed, CACHE_END, fetch_until)
        data_15m = extend_data_15m(data_15m, fresh, CACHE_END)
        # Recompute effective_end based on extended data_15m (15m bars × 15 min from cache_start)
        # Using any symbol's fresh last kline timestamp as authoritative
        last_ms = max(int(f["open_ms"][-1]) for f in fresh.values() if len(f["open_ms"]) > 0)
        effective_end = datetime.fromtimestamp(last_ms / 1000 + 15 * 60, tz=timezone.utc)
        print(f"Extended data_15m: effective end now {effective_end.isoformat()}")

    data_4h = build_resampled(data_15m, 16)

    # Run sim per symbol
    print("Running sim_donchian_trailing per symbol...")
    sim_by_symbol: dict[str, list[tuple[int, int, float]]] = {}
    for sym in data_4h:
        if sym == "BTCUSDT":
            continue
        sim_by_symbol[sym] = run_sim_donchian_for_symbol(cfg, data_4h[sym])

    # Build per-trade comparison
    rows: list[dict] = []
    for lt in live_trades:
        sym = lt["symbol"]
        entry_ts = parse_iso(lt["entry_time"])
        bar_idx = bar_index_from_time(entry_ts, data_4h, sym, effective_end)
        if bar_idx == -1:
            lt["_out_of_cache"] = True
        sim_trade = (
            find_matching_sim_trade(sim_by_symbol.get(sym, []), bar_idx)
            if bar_idx >= 0 else None
        )
        rows.append(build_per_trade_row(lt, sim_trade))

    # Aggregate + recommendation
    agg = compute_aggregates(rows, live_trades)
    recommendation, reasons = make_recommendation(agg)

    # Output JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "donchian_parity.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "recommendation": recommendation,
                "reasons": reasons,
                "aggregate": agg,
                "per_trade": rows,
            },
            f,
            indent=2,
        )

    # Report
    import subprocess
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
    ).decode().strip()
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    report_path = ROOT / "docs" / "backtest_results" / "202604" / (
        f"{date_str}_{commit}_donchian_parity.md"
    )
    write_report(rows, agg, recommendation, reasons, report_path)

    # Print summary to stdout
    print()
    print("=" * 78)
    print(f"AGGREGATE: {json.dumps(agg, indent=2)}")
    print(f"RECOMMENDATION: {recommendation}")
    for r in reasons:
        print(f"  - {r}")
    print(f"\nJSON:    {json_path}")
    print(f"REPORT:  {report_path}")


if __name__ == "__main__":
    main()
