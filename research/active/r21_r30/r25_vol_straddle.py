#!/usr/bin/env python3
"""
R25: Volatility Straddle Alpha Search
======================================
Hypothesis
----------
Volatility expansion is predictable even when direction is not.
Enter LONG + SHORT simultaneously when a vol-compression signal fires.
One side is guaranteed to lose (SL hit); the other may profit (TP hit).

Straddle P&L math
-----------------
  WIN (one TP + one SL):   net = +tp% − sl% − 4×fee
  WHIPSAW (both SL):       net = −2×sl% − 4×fee
  TIMEOUT (close both):    net ≈ 0 − 4×fee

Key advantage: direction-neutral. Funding fees cancel (long + short).
Cost = 4 × commission per straddle (2 entries + 2 exits).

Signals tested (11 variants)
-----------------------------
  V0  Random baseline (every 96 bars)
  V1  Bollinger Band Squeeze (BBW at bottom 10%ile of 200 bars)
  V2  ATR Crush (ATR% at bottom 10%ile)
  V3  Volume Drought (volume at bottom 10%ile)
  V4  Range Compression (SMA of bar range at bottom 10%ile)
  V5  BB + Volume combined
  V6  TTM Squeeze (BB inside Keltner Channel)
  V7  BB Squeeze onset (first bar of new squeeze)
  V8  ATR Crush onset
  V9  Compression Score ≥ 3 (of 5 base indicators)
  V10 Compression Score ≥ 4

All signals use only completed / past data.  Zero lookahead bias.
Per-trade independent test.  No position overlap.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

# ── Project path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ── Constants ────────────────────────────────────────────────────────────────
COST_PER_SIDE = 0.0004  # 0.04% per side (maker/taker avg for liquid)
COST_PER_STRADDLE = COST_PER_SIDE * 4  # 4 legs: 2 open + 2 close

# Indicator parameters
BB_PERIOD = 20
ATR_PERIOD = 14
PCT_LOOKBACK = 200  # Percentile-rank lookback window
SQUEEZE_PCT = 10.0  # Bottom 10th percentile → squeeze signal

# Straddle configs: (label, sl_pct, tp_pct, max_hold_bars)
STRADDLE_CONFIGS = [
    ("A:SL0.5/TP1.5/24h", 0.005, 0.015, 96),
    ("B:SL0.5/TP2.0/24h", 0.005, 0.020, 96),
    ("C:SL1.0/TP2.0/24h", 0.010, 0.020, 96),
    ("D:SL1.0/TP3.0/24h", 0.010, 0.030, 96),
    ("E:SL1.0/TP4.0/48h", 0.010, 0.040, 192),
]

SIGNAL_NAMES = [
    "V0-Random",
    "V1-BB-Squeeze",
    "V2-ATR-Crush",
    "V3-Vol-Drought",
    "V4-Range-Compress",
    "V5-BB+Vol",
    "V6-TTM-Squeeze",
    "V7-BB-Onset",
    "V8-ATR-Onset",
    "V9-Score≥3",
    "V10-Score≥4",
]
NUM_SIGNALS = len(SIGNAL_NAMES)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Indicator helpers
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _percentile_rank(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling percentile rank.  0 = lowest in window, 100 = highest."""
    n = len(arr)
    out = np.full(n, 50.0)
    for i in range(lookback - 1, n):
        cur = arr[i]
        if np.isnan(cur):
            continue
        below = 0
        valid = 0
        for k in range(lookback):
            v = arr[i - lookback + 1 + k]
            if not np.isnan(v):
                valid += 1
                if v < cur:
                    below += 1
        if valid > 0:
            out[i] = below / valid * 100.0
    return out


@numba.jit(nopython=True, cache=True)
def _bbw(close: np.ndarray, period: int) -> np.ndarray:
    """Bollinger Band Width = 2 × mult × std / SMA  (mult=2)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += close[i - period + 1 + k]
        mean = s / period
        v = 0.0
        for k in range(period):
            d = close[i - period + 1 + k] - mean
            v += d * d
        v /= period
        if mean > 0:
            out[i] = 4.0 * np.sqrt(v) / mean  # 2*2*std / mean
    return out


@numba.jit(nopython=True, cache=True)
def _atr_pct(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> np.ndarray:
    """ATR% = EMA-style ATR / close."""
    n = len(close)
    out = np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = (high[0] - low[0]) / close[0] if close[0] > 0 else 0.0
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        mx = hl
        if hc > mx:
            mx = hc
        if lc > mx:
            mx = lc
        tr[i] = mx / close[i] if close[i] > 0 else 0.0
    if n >= period:
        atr = 0.0
        for k in range(period):
            atr += tr[k]
        atr /= period
        out[period - 1] = atr
        for i in range(period, n):
            atr = (atr * (period - 1) + tr[i]) / period
            out[i] = atr
    return out


@numba.jit(nopython=True, cache=True)
def _range_pct_sma(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> np.ndarray:
    """SMA of bar-range% = (high − low) / close."""
    n = len(close)
    out = np.full(n, np.nan)
    buf = np.empty(n)
    for i in range(n):
        buf[i] = (high[i] - low[i]) / close[i] if close[i] > 0 else 0.0
    s = 0.0
    for i in range(n):
        s += buf[i]
        if i >= period:
            s -= buf[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def _kc_width(
    close: np.ndarray,
    atr_pct_arr: np.ndarray,
    period: int,
    mult: float,
) -> np.ndarray:
    """Keltner Channel width (as fraction of close) = 2 × mult × ATR%."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(atr_pct_arr[i]):
            out[i] = 2.0 * mult * atr_pct_arr[i]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Signal generation (all 11 variants in one pass)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _gen_signals_all(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    bb_period: int,
    atr_period: int,
    pct_lookback: int,
    squeeze_pct: float,
) -> np.ndarray:
    """
    Generate all 11 signal arrays at once.

    Returns
    -------
    int8 array of shape (11, n).
    Row i → SIGNAL_NAMES[i].
    """
    n = len(close)
    signals = np.zeros((11, n), dtype=np.int8)

    # ── Indicators ──
    bbw = _bbw(close, bb_period)
    atr = _atr_pct(close, high, low, atr_period)
    rng = _range_pct_sma(close, high, low, atr_period)
    kc_w = _kc_width(close, atr, bb_period, 1.5)

    bbw_rank = _percentile_rank(bbw, pct_lookback)
    atr_rank = _percentile_rank(atr, pct_lookback)
    vol_rank = _percentile_rank(volume, pct_lookback)
    rng_rank = _percentile_rank(rng, pct_lookback)

    warmup = pct_lookback

    # V0: Random — every 96 bars after warmup
    for i in range(warmup, n):
        if (i - warmup) % 96 == 0:
            signals[0, i] = 1

    # V1-V6 continuous  +  V9/V10 composite score
    for i in range(warmup, n):
        bb_sq = bbw_rank[i] < squeeze_pct
        atr_sq = atr_rank[i] < squeeze_pct
        vol_sq = vol_rank[i] < squeeze_pct
        rng_sq = rng_rank[i] < squeeze_pct
        ttm = False
        if not np.isnan(bbw[i]) and not np.isnan(kc_w[i]):
            ttm = bbw[i] < kc_w[i]

        if bb_sq:
            signals[1, i] = 1
        if atr_sq:
            signals[2, i] = 1
        if vol_sq:
            signals[3, i] = 1
        if rng_sq:
            signals[4, i] = 1
        if bb_sq and vol_sq:
            signals[5, i] = 1
        if ttm:
            signals[6, i] = 1

        score = 0
        if bb_sq:
            score += 1
        if atr_sq:
            score += 1
        if vol_sq:
            score += 1
        if rng_sq:
            score += 1
        if ttm:
            score += 1
        if score >= 3:
            signals[9, i] = 1
        if score >= 4:
            signals[10, i] = 1

    # V7: BB squeeze onset (0→1 transition)
    for i in range(warmup + 1, n):
        if signals[1, i] == 1 and signals[1, i - 1] == 0:
            signals[7, i] = 1

    # V8: ATR crush onset
    for i in range(warmup + 1, n):
        if signals[2, i] == 1 and signals[2, i - 1] == 0:
            signals[8, i] = 1

    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Straddle simulation
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def simulate_straddles(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
    cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate non-overlapping straddles on signal bars.

    Per straddle: open LONG + SHORT at close[signal_bar].
    Walk forward bar-by-bar; SL priority within each bar.
    No new straddle until the current one fully resolves.

    Returns
    -------
    pnl     : float64 — per-trade net P&L (fraction of notional per side)
    outcome : int64   — 0=whipsaw, 1=long-win, 2=short-win, 3=timeout
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    outcome = np.empty(cap, dtype=np.int64)
    cnt = 0

    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        l_sl = entry * (1.0 - sl_pct)
        l_tp = entry * (1.0 + tp_pct)
        s_sl = entry * (1.0 + sl_pct)
        s_tp = entry * (1.0 - tp_pct)

        l_alive = True
        s_alive = True
        l_exit = 0.0
        s_exit = 0.0
        exit_bar = i
        end = min(i + 1 + max_hold, n)
        done = False

        for j in range(i + 1, end):
            h = high[j]
            lo = low[j]

            # ── LONG: SL priority ──
            if l_alive:
                if lo <= l_sl:
                    l_exit = -sl_pct
                    l_alive = False
                elif h >= l_tp:
                    l_exit = tp_pct
                    l_alive = False

            # ── SHORT: SL priority ──
            if s_alive:
                if h >= s_sl:
                    s_exit = -sl_pct
                    s_alive = False
                elif lo <= s_tp:
                    s_exit = tp_pct
                    s_alive = False

            if not l_alive and not s_alive:
                exit_bar = j
                done = True
                break

        if not done:
            exit_bar = end - 1

        # Timeout — close surviving legs at market
        if l_alive:
            c = close[exit_bar]
            l_exit = (c - entry) / entry
        if s_alive:
            c = close[exit_bar]
            s_exit = (entry - c) / entry

        net = l_exit + s_exit - cost

        # Classify
        if l_exit >= tp_pct * 0.9:
            oc = 1  # long won
        elif s_exit >= tp_pct * 0.9:
            oc = 2  # short won
        elif l_exit <= -sl_pct * 0.9 and s_exit <= -sl_pct * 0.9:
            oc = 0  # whipsaw
        else:
            oc = 3  # timeout / partial

        if cnt < cap:
            pnl[cnt] = net
            outcome[cnt] = oc
            cnt += 1

        # Advance past this straddle (no overlap)
        i = exit_bar + 1

    return pnl[:cnt], outcome[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def _extract_days(path: Path) -> int:
    """Extract day count from filename like BTCUSDT_15m_2190d_20260220.npz."""
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_data() -> dict:
    """Load top-50 liquid 15m data from data/cache_extended/."""
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings

    settings = QuantBoxSettings()
    whitelist = settings.rb_symbol_whitelist.split(",")

    data: dict = {}
    for sym in whitelist:
        candidates = list(cache_dir.glob(f"{sym}_15m_*.npz"))
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)

        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(npz.files)):
            print(f"  ⚠️  {sym}: missing keys {required - set(npz.files)}, skip")
            continue

        c = npz["close"].astype(np.float64)
        h = npz["high"].astype(np.float64)
        lo = npz["low"].astype(np.float64)
        v = npz["volume"].astype(np.float64)

        if len(c) < PCT_LOOKBACK + 200:
            continue

        data[sym] = {"close": c, "high": h, "low": lo, "volume": v, "bars": len(c)}

    print(f"  Loaded {len(data)} liquid symbols")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregation & reporting
# ══════════════════════════════════════════════════════════════════════════════


def aggregate(all_pnl: dict, all_oc: dict) -> dict:
    """Aggregate per-symbol arrays into a summary dict keyed by variant label."""
    summary: dict = {}
    for key in all_pnl:
        pnl_arr = np.concatenate(all_pnl[key]) if all_pnl[key] else np.array([])
        oc_arr = np.concatenate(all_oc[key]) if all_oc[key] else np.array([], dtype=np.int64)
        n = len(pnl_arr)
        if n == 0:
            continue

        wins = int(np.sum((oc_arr == 1) | (oc_arr == 2)))
        whips = int(np.sum(oc_arr == 0))
        timeouts = int(np.sum(oc_arr == 3))

        pos = pnl_arr[pnl_arr > 0]
        neg = pnl_arr[pnl_arr < 0]
        gross_pos = float(np.sum(pos)) if len(pos) > 0 else 0.0
        gross_neg = float(np.abs(np.sum(neg))) if len(neg) > 0 else 1e-9
        pf = gross_pos / gross_neg
        ev_bp = float(np.mean(pnl_arr)) * 10_000

        summary[key] = {
            "N": n,
            "Win%": wins / n * 100,
            "Whip%": whips / n * 100,
            "Time%": timeouts / n * 100,
            "PF": pf,
            "EV_bp": ev_bp,
        }
    return summary


def print_table(summary: dict) -> None:
    hdr = (
        f"  {'Signal':<20} {'Config':<18} {'N':>7}  "
        f"{'Win%':>5}  {'Whip%':>5}  {'Time%':>5}  {'PF':>6}  {'EV(bp)':>8}"
    )
    sep = "  " + "─" * 85
    print(hdr)
    print(sep)
    for key in sorted(summary.keys()):
        s = summary[key]
        parts = key.split("|")
        sig = parts[0] if len(parts) > 0 else key
        cfg = parts[1] if len(parts) > 1 else ""
        print(
            f"  {sig:<20} {cfg:<18} {s['N']:>7}  "
            f"{s['Win%']:>5.1f}  {s['Whip%']:>5.1f}  {s['Time%']:>5.1f}  "
            f"{s['PF']:>6.3f}  {s['EV_bp']:>+8.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    now = datetime.now(timezone.utc)

    print("=" * 85)
    print("  R25: VOLATILITY STRADDLE ALPHA SEARCH")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Hypothesis: Vol compression → vol expansion (direction-neutral)")
    print("  Strategy: Enter LONG + SHORT simultaneously; one side SL, other TP")
    print(f"  Cost: 4 × {COST_PER_SIDE*100:.2f}% = {COST_PER_STRADDLE*100:.2f}% per straddle")
    print(f"  Signals: {NUM_SIGNALS} variants  |  Configs: {len(STRADDLE_CONFIGS)} TP/SL combos")
    print("=" * 85)
    print()

    # ── Break-even table ──
    print("  BREAK-EVEN WIN RATES (theoretical)")
    print("  " + "─" * 60)
    for label, sl, tp, mh in STRADDLE_CONFIGS:
        win_net = tp - sl - COST_PER_STRADDLE
        loss_net = 2 * sl + COST_PER_STRADDLE
        be_wr = loss_net / (win_net + loss_net) * 100
        print(
            f"    {label:<22}  WIN={win_net*100:>+5.2f}%  "
            f"WHIP={-loss_net*100:>6.2f}%  BE-WR={be_wr:>5.1f}%"
        )
    print()

    # ── Load data ──
    print("  Loading data...")
    data = load_data()
    if not data:
        print("  ❌ No data loaded!")
        return

    total_bars = sum(d["bars"] for d in data.values())
    min_bars = min(d["bars"] for d in data.values())
    max_bars = max(d["bars"] for d in data.values())
    print(f"  Total bars: {total_bars:,}  |  Range: {min_bars:,} — {max_bars:,}")
    print()

    # ── Run backtest ──
    print("  Running straddle simulations...")
    all_pnl: dict[str, list[np.ndarray]] = {}
    all_oc: dict[str, list[np.ndarray]] = {}
    t0 = time.time()

    sym_count = 0
    for sym, d in sorted(data.items()):
        sym_count += 1
        c = d["close"]
        h = d["high"]
        lo = d["low"]
        v = d["volume"]

        # Generate all 11 signal arrays at once
        sig_matrix = _gen_signals_all(
            c, h, lo, v, BB_PERIOD, ATR_PERIOD, PCT_LOOKBACK, SQUEEZE_PCT
        )

        for sig_idx in range(NUM_SIGNALS):
            sig = sig_matrix[sig_idx]

            for cfg_label, sl, tp, mh in STRADDLE_CONFIGS:
                key = f"{SIGNAL_NAMES[sig_idx]}|{cfg_label}"

                pnl_arr, oc_arr = simulate_straddles(c, h, lo, sig, sl, tp, mh, COST_PER_STRADDLE)

                if key not in all_pnl:
                    all_pnl[key] = []
                    all_oc[key] = []
                all_pnl[key].append(pnl_arr)
                all_oc[key].append(oc_arr)

        if sym_count % 10 == 0:
            print(f"    [{sym_count}/{len(data)}] {sym} done")

    elapsed = time.time() - t0
    print(f"  ✅ Simulation complete: {elapsed:.1f}s")
    print()

    # ── Aggregate ──
    summary = aggregate(all_pnl, all_oc)

    # ══════════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 85)
    print(f"  FULL RESULTS  (runtime: {elapsed:.1f}s)")
    print("=" * 85)
    print()
    print_table(summary)
    print()

    # ── Top 10 by PF ──
    print("  ── TOP 10 VARIANTS (by Profit Factor) ──")
    top10 = sorted(summary.items(), key=lambda x: -x[1]["PF"])[:10]
    for rank, (key, s) in enumerate(top10, 1):
        marker = "✅" if s["PF"] >= 1.0 else "❌"
        print(
            f"    {rank:>2}. {marker} {key:<42} "
            f"PF={s['PF']:.3f}  EV={s['EV_bp']:>+6.2f}bp  "
            f"Win={s['Win%']:.1f}%  N={s['N']:,}"
        )
    print()

    # ── Edge vs Random ──
    print("  ── EDGE VS RANDOM (Win% difference per config) ──")
    for cfg_label, sl, tp, mh in STRADDLE_CONFIGS:
        rk = f"V0-Random|{cfg_label}"
        if rk not in summary:
            continue
        rnd_wr = summary[rk]["Win%"]
        rnd_pf = summary[rk]["PF"]
        print(f"\n  Config {cfg_label}  (Random: Win%={rnd_wr:.1f}% PF={rnd_pf:.3f})")
        for sig_idx in range(1, NUM_SIGNALS):
            key = f"{SIGNAL_NAMES[sig_idx]}|{cfg_label}"
            if key not in summary:
                continue
            s = summary[key]
            edge_wr = s["Win%"] - rnd_wr
            edge_pf = s["PF"] - rnd_pf
            if edge_wr > 2.0:
                m = "✅"
            elif edge_wr > 0:
                m = "⚠️ "
            else:
                m = "❌"
            print(
                f"    {m} {SIGNAL_NAMES[sig_idx]:<20} "
                f"Win%={s['Win%']:>5.1f}  edge={edge_wr:>+5.1f}pp  "
                f"PF={s['PF']:.3f} ({edge_pf:>+.3f})"
            )
    print()

    # ── Break-even analysis ──
    print("  ── BREAK-EVEN ANALYSIS (actual vs required) ──")
    for cfg_label, sl, tp, mh in STRADDLE_CONFIGS:
        win_net = tp - sl - COST_PER_STRADDLE
        loss_net = 2 * sl + COST_PER_STRADDLE
        be_wr = loss_net / (win_net + loss_net) * 100

        best_sig = "none"
        best_wr = 0.0
        for sig_idx in range(1, NUM_SIGNALS):
            key = f"{SIGNAL_NAMES[sig_idx]}|{cfg_label}"
            if key in summary and summary[key]["Win%"] > best_wr:
                best_wr = summary[key]["Win%"]
                best_sig = SIGNAL_NAMES[sig_idx]

        gap = best_wr - be_wr
        marker = "✅" if gap > 0 else "❌"
        print(
            f"  {marker} {cfg_label:<22}  BE-WR={be_wr:>5.1f}%  "
            f"Best={best_wr:>5.1f}% ({best_sig:<20})  gap={gap:>+5.1f}pp"
        )
    print()

    # ── Distribution of best variant ──
    best_key = top10[0][0] if top10 else None
    if best_key and best_key in all_pnl:
        pnls = np.concatenate(all_pnl[best_key])
        print(f"  ── P&L DISTRIBUTION: {best_key} ──")
        print(f"    N trades:  {len(pnls):,}")
        print(f"    Mean:      {np.mean(pnls)*10000:>+8.2f} bp")
        print(f"    Std:       {np.std(pnls)*10000:>8.2f} bp")
        print(f"    Min:       {np.min(pnls)*10000:>+8.2f} bp")
        print(f"    P25:       {np.percentile(pnls, 25)*10000:>+8.2f} bp")
        print(f"    Median:    {np.median(pnls)*10000:>+8.2f} bp")
        print(f"    P75:       {np.percentile(pnls, 75)*10000:>+8.2f} bp")
        print(f"    Max:       {np.max(pnls)*10000:>+8.2f} bp")
        print()

    # ── Final verdict ──
    best_pf = top10[0][1]["PF"] if top10 else 0.0
    if best_pf >= 1.10:
        verdict = "✅ PROMISING — best PF ≥ 1.10. Worth deeper investigation."
    elif best_pf >= 1.00:
        verdict = "⚠️  MARGINAL — best PF ≥ 1.0 but slim. May not survive friction."
    else:
        verdict = "❌ NO ALPHA — all variants PF < 1.0. Straddle costs exceed signal edge."

    print("=" * 85)
    print(f"  VERDICT: {verdict}")
    if top10:
        print(f"  Best variant: {top10[0][0]} → PF={best_pf:.3f}")
    print("=" * 85)
    print()

    # ── Save ──
    out_path = ROOT / "logs" / "research" / "vol_straddle_r25_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "commit": commit,
        "date": now.isoformat(),
        "cost_per_straddle": COST_PER_STRADDLE,
        "signals": SIGNAL_NAMES,
        "configs": [c[0] for c in STRADDLE_CONFIGS],
        "summary": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in summary.items()
        },
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
