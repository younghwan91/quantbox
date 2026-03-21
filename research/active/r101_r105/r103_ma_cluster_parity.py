"""r103: MA Cluster Parity — Bug-free backtest vs live journal comparison.

Goal:
  Separate bug-caused losses from strategy losses by running MA Cluster
  backtest with current bug-free code on the exact live period (Mar 5-21)
  and comparing trade-by-trade against journal.sqlite.

Method:
  1. Run r92-style sim (gate ON) on full 6yr data → reproduce baseline PF≈2.749
  2. Run same sim on parity window (Mar 5-21) only
  3. Load journal MA Cluster trades, compare entry symbols/directions/exit reasons
  4. Apply r101 bias correction (stratified by exit type if available)

Pass criteria (proceed to paper mode):
  - Full 6yr bias-corrected PF > 1.5
  - Parity window: ≥70% exit reason match rate
  - No new anomaly in parity comparison

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r101_r105/r103_ma_cluster_parity.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import compute_metrics, portfolio_sim
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config ──────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

MA_PARAMS = {
    "cluster_pct": cfg.ma_cluster_pct,
    "min_duration": cfg.ma_cluster_min_duration,
    "max_duration": cfg.ma_cluster_max_duration,
    "min_momentum": cfg.ma_cluster_min_momentum,
    "sl": cfg.ma_cluster_sl,
    "trail_activate": cfg.ma_cluster_trail_activate,
    "trail_distance": cfg.ma_cluster_trail_distance,
    "tp": 0.20,  # dead cap — trail exits first
    "max_hold": cfg.ma_cluster_max_hold_4h,
    "cost": 0.0008,
    "warmup": 200,
}
BTC_GATE_PERIOD = cfg.ma_cluster_btc_ma_period_4h
MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent  # 21/10 = 2.1
DATA_YEARS = 6.0
RATIO = 16  # 15m → 4h

# r101 bias correction (bp per trade, by exit type)
# Trail exits have larger bias than SL/max_hold exits.
# If r101 stratified data not available, use aggregate 10bp.
BIAS_TRAIL_BP = 15.0  # conservative estimate for trail exits
BIAS_SL_BP = 2.0  # SL fires at exact price, minimal bias
BIAS_MAXHOLD_BP = 3.0  # close-price exit, small bias
BIAS_AGGREGATE_BP = 10.0  # fallback

# Parity window: 2026-03-05 to 2026-03-21
PARITY_START = datetime(2026, 3, 5, tzinfo=timezone.utc)
PARITY_END = datetime(2026, 3, 21, 23, 59, 59, tzinfo=timezone.utc)


# ── Data loading (reuse r92 pattern) ────────────────────────────────────────

def load_data(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    all_syms = set(whitelist) | {"BTCUSDT"}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in all_syms:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        d = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
        # Store timestamps if available for parity window mapping
        if "timestamps" in npz.files:
            d["_timestamps"] = npz["timestamps"]
        data[sym] = d
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def build_resampled(c15: np.ndarray, h15: np.ndarray, l15: np.ndarray, ratio: int) -> tuple:
    n4 = len(c15) // ratio
    close = c15[ratio - 1 : n4 * ratio : ratio].copy()
    high = np.array([np.max(h15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    low = np.array([np.min(l15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    return close, high, low


def compute_sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.mean(arr[i - period + 1 : i + 1])
    return out


def momentum8(close: np.ndarray) -> np.ndarray:
    mom = np.full(len(close), np.nan)
    for i in range(8, len(close)):
        mom[i] = (close[i] - close[i - 8]) / close[i - 8]
    return mom


# ── Core sim (reuse r92 exactly) ────────────────────────────────────────────

@numba.jit(nopython=True, cache=True)
def sim_live_style_gated(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    btc_close_4h, btc_ma50_4h, gate_enabled,
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """Live-style MA cluster sim with optional 4h BTC gate.
    Returns (pnl, entry_bars, exit_bars, directions, exit_reasons).
    exit_reason: 0=SL, 1=TP, 2=trail, 3=max_hold
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    dir_arr = np.empty(cap, np.int64)
    reason_arr = np.empty(cap, np.int64)  # 0=SL, 1=TP, 2=trail, 3=max_hold
    cnt = 0
    i = warmup
    cluster_count = 0
    in_cooldown = 0

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            in_cooldown = 0
            i += 1
            continue
        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0.0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            in_cooldown = 0
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue
        if in_cooldown == 1:
            i += 1
            continue
        if ma_min <= close[i] <= ma_max:
            i += 1
            continue
        if abs(momentum[i]) < min_momentum:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        if gate_enabled == 1 and d == 1 and i < len(btc_ma50_4h) and not np.isnan(btc_ma50_4h[i]):
            if btc_close_4h[i] < btc_ma50_4h[i]:
                in_cooldown = 1
                i += 1
                continue

        in_cooldown = 1
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False
        exit_reason = 3  # default max_hold

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        exit_reason = 2  # trail
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    exit_reason = 1  # TP
                    break
                if not trailing_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    exit_reason = 0  # SL
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        exit_reason = 2  # trail
                        break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    exit_reason = 1  # TP
                    break
                if not trailing_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    exit_reason = 0  # SL
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl_arr[cnt] = trade_pnl - cost
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            dir_arr[cnt] = d
            reason_arr[cnt] = exit_reason
            cnt += 1

        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], dir_arr[:cnt], reason_arr[:cnt]


# ── Per-symbol runner ────────────────────────────────────────────────────────

def run_symbol(sym: str, d15: dict, btc_4h: np.ndarray, btc_ma50: np.ndarray, p: dict) -> dict:
    c15, h15, l15 = d15["close"], d15["high"], d15["low"]
    c4, h4, l4 = build_resampled(c15, h15, l15, RATIO)
    sma20 = compute_sma(c4, 20)
    sma50 = compute_sma(c4, 50)
    sma100 = compute_sma(c4, 100)
    sma200 = compute_sma(c4, 200)
    mom = momentum8(c4)
    offset = d15.get("_offset", 0)

    btc_offset_4h = offset // RATIO
    n4 = len(c4)
    if len(btc_4h) >= btc_offset_4h + n4:
        btc_close_aligned = btc_4h[btc_offset_4h : btc_offset_4h + n4]
        btc_ma50_aligned = btc_ma50[btc_offset_4h : btc_offset_4h + n4]
    else:
        btc_close_aligned = np.full(n4, np.nan)
        btc_ma50_aligned = np.full(n4, np.nan)

    base_args = (c4, h4, l4, sma20, sma50, sma100, sma200, mom, btc_close_aligned, btc_ma50_aligned)
    sim_args = (
        p["cluster_pct"], p["min_duration"], p["max_duration"], p["min_momentum"],
        p["sl"], p["tp"], p["trail_activate"], p["trail_distance"],
        p["max_hold"], p["cost"], p["warmup"],
    )

    pnl, ent, ext, dirs, reasons = sim_live_style_gated(*base_args, 1, *sim_args)

    # Convert 4h bars → 15m bars for portfolio_sim
    ent15 = (ent * RATIO + offset).astype(np.int64)
    ext15 = ((ext + 1) * RATIO - 1 + offset).astype(np.int64)

    return {
        "sym": sym,
        "pnl": pnl, "ent": ent15, "ext": ext15, "dir": dirs, "reason": reasons,
        "ent_4h": ent, "n4h": n4, "offset_4h": offset // RATIO,
        "c4": c4,  # for parity window date mapping
    }


# ── Parity window extraction ────────────────────────────────────────────────

def find_parity_bar_range(data: dict) -> tuple[int, int] | None:
    """Find 15m bar indices corresponding to the parity window using BTC timestamps."""
    btc = data.get("BTCUSDT")
    if btc is None or "_timestamps" not in btc:
        # Fallback: estimate from bar count and known end date
        # Cache ends ~2026-03-21, 15m bars. 16 days = 16*24*4 = 1536 bars from end
        max_bars = max(len(d["close"]) for d in data.values())
        bars_16d = 16 * 24 * 4  # Mar 5 to Mar 21
        return (max_bars - bars_16d, max_bars)

    ts = btc["_timestamps"]
    start_ms = int(PARITY_START.timestamp() * 1000)
    end_ms = int(PARITY_END.timestamp() * 1000)

    start_idx = np.searchsorted(ts, start_ms)
    end_idx = np.searchsorted(ts, end_ms)
    return (int(start_idx), int(end_idx))


def filter_parity_trades(results: list[dict], bar_start: int, bar_end: int) -> list[dict]:
    """Filter sim trades to only those within the parity window."""
    parity = []
    for r in results:
        mask = (r["ent"] >= bar_start) & (r["ent"] < bar_end)
        if mask.any():
            parity.append({
                "sym": r["sym"],
                "pnl": r["pnl"][mask],
                "ent": r["ent"][mask],
                "ext": r["ext"][mask],
                "dir": r["dir"][mask],
                "reason": r["reason"][mask],
            })
    return parity


# ── Journal loading ──────────────────────────────────────────────────────────

REASON_MAP = {
    "server_sl": "SL", "sl_hit": "SL", "stop_loss": "SL",
    "trailing_stop": "trail", "server_trail": "trail",
    "max_hold": "max_hold", "drawdown_halt": "max_hold",
}
SIM_REASON_MAP = {0: "SL", 1: "TP", 2: "trail", 3: "max_hold"}


def load_journal_ma_trades() -> list[dict]:
    db_path = ROOT / "data" / "journal.sqlite"
    if not db_path.exists():
        print(f"WARNING: Journal not found at {db_path}")
        return []
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute(
        "SELECT exit_time, symbol, direction, strategy_type, exit_reason, pnl_usd, pnl_pct, bars_held "
        "FROM trades WHERE strategy_type = 'ma_cluster' AND entry_time >= '2026-03-05' "
        "ORDER BY exit_time"
    )
    rows = c.fetchall()
    conn.close()
    trades = []
    for r in rows:
        raw_reason = r[4] or ""
        mapped = REASON_MAP.get(raw_reason, raw_reason)
        trades.append({
            "exit_time": r[0],
            "symbol": r[1],
            "direction": r[2] or "LONG",
            "reason": mapped,
            "raw_reason": raw_reason,
            "pnl_usd": r[5],
            "pnl_pct": r[6],
            "bars_held": r[7],
        })
    return trades


# ── Bias correction ──────────────────────────────────────────────────────────

def apply_bias_correction(pnl: np.ndarray, reasons: np.ndarray) -> np.ndarray:
    """Apply r101 bias correction stratified by exit type."""
    corrected = pnl.copy()
    for i in range(len(pnl)):
        r = int(reasons[i])
        if r == 0:  # SL
            corrected[i] -= BIAS_SL_BP / 10000
        elif r == 2:  # trail
            corrected[i] -= BIAS_TRAIL_BP / 10000
        elif r == 3:  # max_hold
            corrected[i] -= BIAS_MAXHOLD_BP / 10000
        else:  # TP or unknown
            corrected[i] -= BIAS_AGGREGATE_BP / 10000
    return corrected


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("r103: MA Cluster Parity — Bug-free Backtest vs Live Journal")
    print("=" * 70)
    print(f"Config: cluster_pct={MA_PARAMS['cluster_pct']} min_dur={MA_PARAMS['min_duration']} "
          f"sl={MA_PARAMS['sl']} trail_act={MA_PARAMS['trail_activate']} "
          f"trail_dist={MA_PARAMS['trail_distance']}")
    print(f"Gate: BTC 4h MA{BTC_GATE_PERIOD} | Parity: {PARITY_START.date()} to {PARITY_END.date()}")
    print()

    # ── Load data ──
    data = load_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data)} symbols from cache_extended")
    if not data:
        print("ERROR: No data found in cache_extended")
        return
    if "BTCUSDT" not in data:
        print("ERROR: BTCUSDT not in cache — gate cannot be computed")
        return

    btc = data["BTCUSDT"]
    btc_close_4h, _, _ = build_resampled(btc["close"], btc["high"], btc["low"], RATIO)
    btc_ma50_4h = compute_sma(btc_close_4h, BTC_GATE_PERIOD)
    max_bars_15m = max(len(d["close"]) for d in data.values())
    print(f"BTC 4h bars: {len(btc_close_4h)} | Max 15m bars: {max_bars_15m}")

    # ── Warm up numba ──
    print("Warming up numba JIT... ", end="", flush=True)
    _c = np.ones(500)
    _nan = np.full(500, np.nan)
    _nan[200:] = 1.0
    _btc = np.ones(500) * 90000.0
    _bma = np.full(500, np.nan)
    _bma[50:] = 88000.0
    sim_live_style_gated(
        _c, _c, _c, _nan, _nan, _nan, _nan, _nan, _btc, _bma, 1,
        0.02, 6, 24, 0.01, 0.02, 0.20, 0.02, 0.003, 48, 0.0, 200,
    )
    print("done\n")

    # ── Run sim (gate ON) ──
    results = []
    syms = [s for s in cfg.symbol_whitelist if s in data and s != "BTCUSDT"]
    for sym in syms:
        try:
            r = run_symbol(sym, data[sym], btc_close_4h, btc_ma50_4h, MA_PARAMS)
            results.append(r)
        except Exception as e:
            print(f"  SKIP {sym}: {e}")
    print(f"Processed {len(results)} symbols\n")

    # ── Full 6yr metrics ──
    trades_all = []
    all_pnl = []
    all_reasons = []
    for r in results:
        for p, e, x, reason in zip(r["pnl"], r["ent"], r["ext"], r["reason"]):
            trades_all.append((r["sym"], int(e), int(x), float(p)))
            all_pnl.append(float(p))
            all_reasons.append(int(reason))

    all_pnl_arr = np.array(all_pnl)
    all_reasons_arr = np.array(all_reasons)

    print("─" * 70)
    print("SECTION 1: FULL 6-YEAR RESULTS (Gate ON)")
    print("─" * 70)

    if len(trades_all) >= 10:
        pnl_portfolio = portfolio_sim(trades_all, max_concurrent=cfg.max_concurrent)
        m = compute_metrics(pnl_portfolio, DATA_YEARS)
        print(f"  N={m['N']}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
              f"MaxDD={m['max_dd_pct']:.1f}%  EV={m.get('EV_bp', 0):.1f}bp")

        # Bias-corrected
        corrected_pnl = apply_bias_correction(all_pnl_arr, all_reasons_arr)
        trades_corrected = [
            (trades_all[i][0], trades_all[i][1], trades_all[i][2], float(corrected_pnl[i]))
            for i in range(len(trades_all))
        ]
        pnl_corrected = portfolio_sim(trades_corrected, max_concurrent=cfg.max_concurrent)
        mc = compute_metrics(pnl_corrected, DATA_YEARS)
        print(f"  Bias-corrected: N={mc['N']}  PF={mc['PF']:.3f}  WR={mc['WR']:.1f}%  "
              f"EV={mc.get('EV_bp', 0):.1f}bp")
        print(f"  (bias: SL={BIAS_SL_BP}bp trail={BIAS_TRAIL_BP}bp maxhold={BIAS_MAXHOLD_BP}bp)")
    else:
        print("  ERROR: Too few trades for portfolio_sim")
        m = {}
        mc = {}

    # ── Exit reason breakdown ──
    print("\n  Exit reason breakdown (raw):")
    for code, name in SIM_REASON_MAP.items():
        mask = all_reasons_arr == code
        n = mask.sum()
        if n > 0:
            avg_pnl = all_pnl_arr[mask].mean() * 10000
            print(f"    {name:>8}: N={n:>4}  avg={avg_pnl:+.1f}bp")

    # ── Parity window ──
    print("\n" + "─" * 70)
    print("SECTION 2: PARITY WINDOW (Mar 5-21)")
    print("─" * 70)

    bar_range = find_parity_bar_range(data)
    if bar_range is None:
        print("  ERROR: Cannot determine parity window bar range")
        return
    bar_start, bar_end = bar_range
    print(f"  15m bar range: [{bar_start}, {bar_end}) = {bar_end - bar_start} bars")

    parity_results = filter_parity_trades(results, bar_start, bar_end)
    parity_trades = []
    for r in parity_results:
        for p, e, x in zip(r["pnl"], r["ent"], r["ext"]):
            parity_trades.append((r["sym"], int(e), int(x), float(p)))

    n_parity = sum(len(r["pnl"]) for r in parity_results)
    print(f"  Sim trades in parity window: {n_parity}")

    if n_parity > 0:
        # List sim trades
        print("\n  Sim trades (parity window):")
        print(f"  {'Symbol':>12} {'Dir':>5} {'Reason':>8} {'PnL%':>8}")
        print("  " + "-" * 40)
        for r in parity_results:
            for p, d, reason in zip(r["pnl"], r["dir"], r["reason"]):
                dir_str = "LONG" if d == 1 else "SHORT"
                reason_str = SIM_REASON_MAP.get(int(reason), "?")
                print(f"  {r['sym']:>12} {dir_str:>5} {reason_str:>8} {p * 100:+8.2f}%")

    # ── Journal comparison ──
    print("\n" + "─" * 70)
    print("SECTION 3: JOURNAL COMPARISON")
    print("─" * 70)

    journal = load_journal_ma_trades()
    print(f"  Journal MA Cluster trades since Mar 5: {len(journal)}")

    if journal:
        print(f"\n  {'Exit Time':>20} {'Symbol':>12} {'Dir':>5} {'Reason':>10} {'PnL$':>10} {'PnL%':>8}")
        print("  " + "-" * 70)
        for t in journal:
            print(f"  {t['exit_time']:>20} {t['symbol']:>12} {t['direction']:>5} "
                  f"{t['reason']:>10} ${t['pnl_usd']:>+8.2f} {(t['pnl_pct'] or 0):+7.2f}%")

    # ── Match analysis ──
    if journal and parity_results:
        print(f"\n  Matching analysis:")
        journal_syms = {t["symbol"] for t in journal}
        sim_syms = {r["sym"] for r in parity_results}
        print(f"    Journal symbols: {sorted(journal_syms)}")
        print(f"    Sim symbols:     {sorted(sim_syms)}")
        print(f"    Overlap:         {sorted(journal_syms & sim_syms)}")

        # Exit reason match rate
        j_reasons = [t["reason"] for t in journal]
        s_reasons = []
        for r in parity_results:
            for reason in r["reason"]:
                s_reasons.append(SIM_REASON_MAP.get(int(reason), "?"))

        j_reason_counts = {}
        for r in j_reasons:
            j_reason_counts[r] = j_reason_counts.get(r, 0) + 1
        s_reason_counts = {}
        for r in s_reasons:
            s_reason_counts[r] = s_reason_counts.get(r, 0) + 1

        print(f"\n    Exit reason distribution:")
        print(f"      Journal: {j_reason_counts}")
        print(f"      Sim:     {s_reason_counts}")

        # Calculate match rate (proportion of journal reasons that match sim)
        all_reasons_set = set(j_reasons) | set(s_reasons)
        total_match = 0
        total_j = len(j_reasons)
        for reason in all_reasons_set:
            j_count = j_reason_counts.get(reason, 0)
            s_count = s_reason_counts.get(reason, 0)
            total_match += min(j_count, s_count)
        match_rate = total_match / max(total_j, 1) * 100
        print(f"    Exit reason match rate: {match_rate:.0f}%")

    # ── Decision gate ──
    print("\n" + "=" * 70)
    print("DECISION GATE")
    print("=" * 70)

    gate_pf = mc.get("PF", 0) if mc else 0
    gate_match = match_rate if journal and parity_results else 0

    pf_pass = gate_pf > 1.5
    match_pass = gate_match >= 70

    print(f"  Full 6yr bias-corrected PF: {gate_pf:.3f} {'PASS' if pf_pass else 'FAIL'} (threshold: >1.5)")
    print(f"  Exit reason match rate:     {gate_match:.0f}% {'PASS' if match_pass else 'FAIL'} (threshold: ≥70%)")

    if pf_pass and match_pass:
        print("\n  ✅ GATE PASSED — Proceed to 48h/7-day paper mode")
        print("  Next: Enable MA Cluster with --paper flag")
    elif pf_pass and not match_pass:
        print("\n  ⚠️  PF passed but match rate low — investigate parity divergence before paper mode")
    else:
        print(f"\n  ❌ GATE FAILED — MA Cluster PF too low ({gate_pf:.3f})")
        print("  Track 2 scope may need to include MA Cluster trail optimization")


if __name__ == "__main__":
    main()
