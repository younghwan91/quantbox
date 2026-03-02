"""R42 — Regime-Adaptive VR Squeeze Parameters

Research Question:
  Can VR squeeze profitability improve in low-volatility periods (2023, 2025)
  by adapting entry/exit parameters to the current volatility regime?

Approach:
  1. Classify each bar into volatility regime using BTC ATR(56) percentile:
     - low_vol:  BTC ATR%ile < 33rd  → market is calm
     - mid_vol:  33rd–67th           → normal
     - high_vol: > 67th              → volatile
  2. For each regime, test different VR parameters:
     - VR threshold (entry sensitivity)
     - SL / TP (exit levels)
     - max_hold (position duration)
  3. Compare:
     - STATIC: v13.0 baseline (VR8_56, d_vr=0.50, SL=1%, TP=5%)
     - ADAPTIVE: regime-conditional parameter sets
  4. Measure by year to check 2023/2025 improvement without hurting good years.

Hypothesis:
  - Low-vol  → lower VR threshold (easier entry), tighter TP (smaller wins),
               tighter SL (cut faster), shorter max_hold
  - High-vol → standard/tighter VR threshold (be selective), wider TP
               (let winners run), wider SL (give room)

No look-ahead: regime classification uses COMPLETED 4h bars only.

Data: 54 symbols × 6yr 15m cache (cache_extended).

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/regime_adaptive_r42.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.mtf_vol_straddle_r31 import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    _atr_pct, compute_vr, compute_momentum, compute_squeeze_dur,
    vol_percentile, gen_vr_signal, gen_vr_dur_signal,
    make_direction_momentum, resample_ohlcv,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.scaling_r38 import (
    compute_metrics, precompute_all_vr,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Regime Classification
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_atr_pct_percentile(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    atr_period: int, lookback: int,
) -> np.ndarray:
    """Rolling percentile of ATR% over `lookback` bars.

    Returns array of same length, values 0.0–1.0.
    Uses completed bars only (no lookahead).
    """
    atr = _atr_pct(close, high, low, atr_period)
    n = len(close)
    pct = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        cur = atr[i]
        if np.isnan(cur):
            continue
        below = 0
        valid = 0
        for k in range(lookback):
            v = atr[i - lookback + 1 + k]
            if not np.isnan(v):
                valid += 1
                if v < cur:
                    below += 1
        if valid > 0:
            pct[i] = below / valid
    return pct


@numba.jit(nopython=True, cache=True)
def classify_regime_from_pctile(
    pctile: np.ndarray, lo_thresh: float, hi_thresh: float,
) -> np.ndarray:
    """Classify: 0=low_vol, 1=mid_vol, 2=high_vol. NaN → 1 (mid default)."""
    n = len(pctile)
    regime = np.ones(n, dtype=np.int8)  # default mid
    for i in range(n):
        if np.isnan(pctile[i]):
            regime[i] = 1
        elif pctile[i] < lo_thresh:
            regime[i] = 0
        elif pctile[i] > hi_thresh:
            regime[i] = 2
        else:
            regime[i] = 1
    return regime


def build_btc_regime_4h(
    data_15m: dict,
    atr_period: int = 56,
    lookback: int = 250,  # ~42 days of 4h bars
    lo_thresh: float = 0.33,
    hi_thresh: float = 0.67,
) -> np.ndarray:
    """Build BTC volatility regime array at 4h resolution.

    Returns int8 array: 0=low_vol, 1=mid_vol, 2=high_vol.
    Uses BTC ATR(56) percentile over 250 bars of 4h.
    """
    btc = data_15m.get("BTCUSDT")
    if btc is None:
        raise ValueError("BTCUSDT not in data_15m")

    c, h, lo, v = resample_ohlcv(btc["close"], btc["high"], btc["low"], btc["volume"], 16)
    pctile = compute_atr_pct_percentile(c, h, lo, atr_period, lookback)
    regime = classify_regime_from_pctile(pctile, lo_thresh, hi_thresh)
    return regime


def expand_regime_to_tf(regime_4h: np.ndarray, ratio: int, n_bars: int) -> np.ndarray:
    """Expand 4h regime to lower TF.

    ratio=16 for 15m, ratio=4 for 1h, ratio=1 for 4h.
    Uses COMPLETED 4h bar only: bar i at lower TF uses regime_4h[i // ratio_from_4h - 1].
    This avoids look-ahead: the 4h bar being formed is NOT used.
    """
    n_4h = len(regime_4h)
    out = np.ones(n_bars, dtype=np.int8)  # default mid
    for i in range(n_bars):
        # Which completed 4h bar does this bar correspond to?
        # For 4h data (ratio=1): use bar i-1 (the previous completed bar)
        # For 1h data (ratio=4): use bar (i // 4) - 1
        # For 15m data (ratio=16): use bar (i // 16) - 1
        j = i // ratio - 1 if ratio > 1 else i - 1
        if 0 <= j < n_4h:
            out[i] = regime_4h[j]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Regime-Adaptive Simulation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RegimeParams:
    """VR squeeze parameters for a single volatility regime."""
    d_vr: float       # directional VR threshold
    s_vr: float       # straddle VR threshold
    s_dur: int        # straddle min squeeze duration
    d_sl: float       # directional stop-loss
    s_sl: float       # straddle stop-loss
    tp: float         # take-profit (both modes)
    d_max_hold: int   # directional max hold (in TF bars)
    s_max_hold: int   # straddle max hold (in TF bars)


@numba.jit(nopython=True, cache=True)
def sim_regime_directional(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vr: np.ndarray, momentum: np.ndarray, regime: np.ndarray,
    warmup: int,
    # Per-regime params: arrays of length 3 [low_vol, mid_vol, high_vol]
    d_vr_arr: np.ndarray, d_sl_arr: np.ndarray, tp_arr: np.ndarray,
    mh_arr: np.ndarray,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional trade sim with regime-adaptive parameters.

    At each potential entry bar i, uses params from regime[i].
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    while i < n - 1:
        r = regime[i]
        # Check signal: VR < threshold for this regime
        if np.isnan(vr[i]) or vr[i] >= d_vr_arr[r]:
            i += 1
            continue
        # Direction from momentum
        if momentum[i] > 0.0001:
            d = 1
        elif momentum[i] < -0.0001:
            d = -1
        else:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        sl = d_sl_arr[r]
        tp = tp_arr[r]
        max_hold = int(mh_arr[r])
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0

        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_regime_straddle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vr: np.ndarray, regime: np.ndarray,
    warmup: int,
    # Per-regime params: arrays of length 3 [low_vol, mid_vol, high_vol]
    s_vr_arr: np.ndarray, s_dur_arr: np.ndarray,
    s_sl_arr: np.ndarray, tp_arr: np.ndarray,
    mh_arr: np.ndarray,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle trade sim with regime-adaptive parameters.

    Squeeze duration is counted per-regime (resets if regime changes).
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0

    # Track squeeze duration
    sq_cnt = 0
    i = 0
    while i < n - 1:
        r = regime[i]
        vr_thresh = s_vr_arr[r]
        min_dur = int(s_dur_arr[r])

        if not np.isnan(vr[i]) and vr[i] < vr_thresh:
            sq_cnt += 1
        else:
            sq_cnt = 0

        if i < warmup or sq_cnt < min_dur:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        sl = s_sl_arr[r]
        tp = tp_arr[r]
        max_hold = int(mh_arr[r])

        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
        sq_cnt = 0  # reset after trade

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection (regime-adaptive)
# ══════════════════════════════════════════════════════════════════════════════

TF_DEFS = [
    # (label, 15m_ratio, warmup, default_mh_d, default_mh_s)
    ("15m", 1, WARMUP_15m, 192, 384),
    ("1h",  4, WARMUP_1h,   48,  96),
    ("4h", 16, WARMUP_4h,   12,  24),
]


def collect_regime_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    regime_4h: np.ndarray,
    params_per_regime: list[RegimeParams],
    strat: str,
    sp: int = 8, lp: int = 56,
) -> list[tuple[str, int, int, float]]:
    """Collect trades with regime-adaptive parameters.

    Returns list of (symbol, entry_15m, exit_15m, pnl).
    """
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    # Build per-regime arrays for numba
    d_vr_arr = np.array([p.d_vr for p in params_per_regime])
    s_vr_arr = np.array([p.s_vr for p in params_per_regime])
    s_dur_arr = np.array([p.s_dur for p in params_per_regime], dtype=np.float64)
    d_sl_arr = np.array([p.d_sl for p in params_per_regime])
    s_sl_arr = np.array([p.s_sl for p in params_per_regime])
    tp_arr = np.array([p.tp for p in params_per_regime])
    d_mh_arr = np.array([p.d_max_hold for p in params_per_regime], dtype=np.float64)
    s_mh_arr = np.array([p.s_max_hold for p in params_per_regime], dtype=np.float64)

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, _, _ in TF_DEFS:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            n_bars = len(c)
            # Expand regime to this TF
            # ratio is 15m-to-TF ratio. We need 4h-to-TF ratio.
            # 4h = 16 × 15m. TF = ratio × 15m.
            # So 4h-to-TF ratio = 16 / ratio
            regime_ratio = 16 // ratio
            regime_tf = expand_regime_to_tf(regime_4h, regime_ratio, n_bars)

            if is_straddle:
                p, e, x = sim_regime_straddle(
                    c, h, lo, vr, regime_tf, warmup,
                    s_vr_arr, s_dur_arr, s_sl_arr, tp_arr, s_mh_arr,
                    COST_STRADDLE,
                )
            else:
                mom = d.get("mom_8")
                if mom is None:
                    mom = compute_momentum(c, 8)
                p, e, x = sim_regime_directional(
                    c, h, lo, vr, mom, regime_tf, warmup,
                    d_vr_arr, d_sl_arr, tp_arr, d_mh_arr,
                    COST_DIRECTIONAL,
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        # Per-symbol time dedup
        sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True
                    break
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades


def fast_regime_pnl(
    data_15m: dict, data_1h: dict, data_4h: dict,
    regime_4h: np.ndarray,
    params_per_regime: list[RegimeParams],
    strat: str,
    sp: int = 8, lp: int = 56,
) -> np.ndarray:
    """Fast PnL array (no trade metadata). For grid scanning."""
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    d_vr_arr = np.array([p.d_vr for p in params_per_regime])
    s_vr_arr = np.array([p.s_vr for p in params_per_regime])
    s_dur_arr = np.array([p.s_dur for p in params_per_regime], dtype=np.float64)
    d_sl_arr = np.array([p.d_sl for p in params_per_regime])
    s_sl_arr = np.array([p.s_sl for p in params_per_regime])
    tp_arr = np.array([p.tp for p in params_per_regime])
    d_mh_arr = np.array([p.d_max_hold for p in params_per_regime], dtype=np.float64)
    s_mh_arr = np.array([p.s_max_hold for p in params_per_regime], dtype=np.float64)

    all_pnl: list[float] = []

    for sym in sorted(data_15m.keys()):
        sym_entries: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, _, _ in TF_DEFS:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            n_bars = len(c)
            regime_ratio = 16 // ratio
            regime_tf = expand_regime_to_tf(regime_4h, regime_ratio, n_bars)

            if is_straddle:
                p, e, x = sim_regime_straddle(
                    c, h, lo, vr, regime_tf, warmup,
                    s_vr_arr, s_dur_arr, s_sl_arr, tp_arr, s_mh_arr,
                    COST_STRADDLE,
                )
            else:
                mom = d.get("mom_8")
                if mom is None:
                    mom = compute_momentum(c, 8)
                p, e, x = sim_regime_directional(
                    c, h, lo, vr, mom, regime_tf, warmup,
                    d_vr_arr, d_sl_arr, tp_arr, d_mh_arr,
                    COST_DIRECTIONAL,
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_entries.append((entry_15m, exit_15m, float(p[k])))

        # Fast dedup
        sym_entries.sort()
        last_exit = -1
        for entry, exit_b, pnl_val in sym_entries:
            if entry > last_exit:
                all_pnl.append(pnl_val)
                last_exit = exit_b

    return np.array(all_pnl) if all_pnl else np.empty(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio Simulation (from R38)
# ══════════════════════════════════════════════════════════════════════════════


def portfolio_sim(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> np.ndarray:
    """Filter trades by max concurrent positions, no same-symbol overlap."""
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted_pnl: list[float] = []
    active: list[tuple[str, int]] = []

    for sym, entry, exit_b, pnl_val in trades:
        active = [(s, e) for s, e in active if e > entry]
        if any(s == sym for s, _ in active):
            continue
        if len(active) >= max_concurrent:
            continue
        active.append((sym, exit_b))
        accepted_pnl.append(pnl_val)

    return np.array(accepted_pnl) if accepted_pnl else np.empty(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Year-by-Year Analysis
# ══════════════════════════════════════════════════════════════════════════════


def estimate_bar_dates(n_bars: int, days: int) -> np.ndarray:
    """Estimate date index for each 15m bar.

    Returns year as float for each bar. Uses linear interpolation from
    data end date (approx 2026-02-20) backward.
    """
    # Data ends at ~2026-02-20, 96 bars/day
    from datetime import datetime, timedelta
    end_date = datetime(2026, 2, 20)
    start_date = end_date - timedelta(days=days)
    years = np.empty(n_bars, dtype=np.float64)
    for i in range(n_bars):
        frac = i / max(n_bars - 1, 1)
        dt = start_date + timedelta(days=frac * days)
        years[i] = dt.year + (dt.timetuple().tm_yday - 1) / 365.25
    return years


def pnl_by_year(
    trades: list[tuple[str, int, int, float]],
    n_15m: int, data_days: int,
) -> dict[int, np.ndarray]:
    """Split trades into yearly PnL arrays."""
    bar_years = estimate_bar_dates(n_15m, data_days)
    yearly: dict[int, list[float]] = {}
    for sym, entry, exit_b, pnl_val in trades:
        if 0 <= entry < n_15m:
            yr = int(bar_years[entry])
            if yr not in yearly:
                yearly[yr] = []
            yearly[yr].append(pnl_val)
    return {yr: np.array(v) for yr, v in sorted(yearly.items())}


# ══════════════════════════════════════════════════════════════════════════════
#  Regime Distribution Diagnostics
# ══════════════════════════════════════════════════════════════════════════════


def regime_distribution(regime_4h: np.ndarray, n_15m: int, data_days: int) -> None:
    """Print regime distribution by year."""
    bar_years = estimate_bar_dates(n_15m, data_days)
    # Expand regime to 15m
    regime_15m = expand_regime_to_tf(regime_4h, 16, n_15m)

    print("\n  ── Regime Distribution (by year) ──")
    print(f"  {'Year':>6} {'Low%':>6} {'Mid%':>6} {'High%':>6} {'Bars':>8}")
    print(f"  {'─'*40}")

    years = sorted(set(int(y) for y in bar_years))
    for yr in years:
        mask = np.array([int(bar_years[i]) == yr for i in range(n_15m)])
        n = mask.sum()
        if n == 0:
            continue
        lo_pct = np.sum((regime_15m == 0) & mask) / n * 100
        mid_pct = np.sum((regime_15m == 1) & mask) / n * 100
        hi_pct = np.sum((regime_15m == 2) & mask) / n * 100
        print(f"  {yr:>6} {lo_pct:>5.1f}% {mid_pct:>5.1f}% {hi_pct:>5.1f}% {n:>8,}")

    total = len(regime_15m)
    lo_total = np.sum(regime_15m == 0) / total * 100
    mid_total = np.sum(regime_15m == 1) / total * 100
    hi_total = np.sum(regime_15m == 2) / total * 100
    print(f"  {'Total':>6} {lo_total:>5.1f}% {mid_total:>5.1f}% {hi_total:>5.1f}% {total:>8,}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    t0 = time.time()
    print(f"R42 — Regime-Adaptive VR Squeeze Parameters [{commit}]")
    print("=" * 120)

    # ── Load data ──
    print("Loading data...")
    data_15m = load_15m()
    data_days = max(d.get("days", 0) for d in data_15m.values())
    data_years = data_days / 365
    n_15m = max(len(d["close"]) for d in data_15m.values())
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)

    # Precompute VR for additional ATR pairs
    all_pairs = [(8, 56), (7, 50), (5, 35), (6, 42)]
    precompute_all_vr([data_15m, data_1h, data_4h], all_pairs)

    print(f"  {len(data_15m)} symbols, {data_days}d ({data_years:.1f}yr), "
          f"{n_15m:,} bars (15m)")
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # ── Build regime ──
    print("\nBuilding BTC volatility regime (ATR(56) percentile on 4h)...")
    regime_4h = build_btc_regime_4h(data_15m)
    regime_distribution(regime_4h, n_15m, data_days)

    # ═════════════════════════════════════════════════════════════════════
    #  Phase 1: Baseline (static params = v13.0)
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  Phase 1: BASELINE (static v13.0 params)")
    print(f"{'='*120}")

    baseline_params = RegimeParams(
        d_vr=0.50, s_vr=0.55, s_dur=4, d_sl=0.01, s_sl=0.015, tp=0.05,
        d_max_hold=192, s_max_hold=384,
    )
    # Static = same params for all regimes
    static_params = [baseline_params, baseline_params, baseline_params]

    for strat in ["D_mom8", "S_d8"]:
        for sp, lp in [(8, 56)]:
            pnl = fast_regime_pnl(
                data_15m, data_1h, data_4h, regime_4h,
                static_params, strat, sp, lp,
            )
            m = compute_metrics(pnl, data_years)
            print(f"\n  {strat} VR{sp}_{lp} (static): "
                  f"N={m['N']:,}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
                  f"t={m['t']:.2f}  WR={m['WR']:.1f}%  R:R={m['RR']:.2f}  "
                  f"TPD={m['tpd']:.2f}  AnnRet={m['ann_ret_pct']:+.1f}%  "
                  f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}")

    # Combined baseline
    d_pnl_bl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                               static_params, "D_mom8", 8, 56)
    s_pnl_bl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                               static_params, "S_d8", 8, 56)
    # Collect trades for portfolio sim + yearly
    d_trades_bl = collect_regime_trades(data_15m, data_1h, data_4h, regime_4h,
                                       static_params, "D_mom8", 8, 56)
    s_trades_bl = collect_regime_trades(data_15m, data_1h, data_4h, regime_4h,
                                       static_params, "S_d8", 8, 56)
    all_trades_bl = d_trades_bl + s_trades_bl
    bl_pnl_10s = portfolio_sim(all_trades_bl, 10)
    bl_m = compute_metrics(bl_pnl_10s, data_years)
    print(f"\n  COMBINED (10 slots): "
          f"N={bl_m['N']:,}  PF={bl_m['PF']:.3f}  EV={bl_m['EV_bp']:+.1f}bp  "
          f"t={bl_m['t']:.2f}  WR={bl_m['WR']:.1f}%  "
          f"AnnRet={bl_m['ann_ret_pct']:+.1f}%  DD={bl_m['max_dd_pct']:.1f}%  "
          f"Calmar={bl_m['calmar']:.1f}")

    # Year-by-year baseline
    yearly_bl = pnl_by_year(all_trades_bl, n_15m, data_days)
    print(f"\n  {'Year':>6} {'N':>5} {'PF':>5} {'EV(bp)':>7} {'WR%':>5} {'R:R':>4}")
    print(f"  {'─'*40}")
    for yr, yr_pnl in yearly_bl.items():
        if len(yr_pnl) < 5:
            continue
        yr_m = compute_metrics(yr_pnl, 1.0)
        print(f"  {yr:>6} {yr_m['N']:>5} {yr_m['PF']:>5.2f} {yr_m['EV_bp']:>+7.1f} "
              f"{yr_m['WR']:>5.1f} {yr_m['RR']:>4.2f}")

    # ═════════════════════════════════════════════════════════════════════
    #  Phase 2: Regime-Adaptive Grid Sweep
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  Phase 2: REGIME-ADAPTIVE PARAMETER SWEEP")
    print(f"{'='*120}")

    # Sweep dimensions (per-regime):
    # Low-vol (regime 0): relax entry, tighten exits
    #   d_vr: [0.40, 0.45, 0.50, 0.55, 0.60]  — wider range to find sweet spot
    #   tp:   [0.02, 0.03, 0.04, 0.05]
    #   d_sl: [0.005, 0.008, 0.01, 0.015]
    #
    # Mid-vol (regime 1): keep near baseline
    #   d_vr: [0.45, 0.50, 0.55]
    #   tp:   [0.03, 0.05, 0.07]
    #   d_sl: [0.008, 0.01, 0.015]
    #
    # High-vol (regime 2): be selective, wider targets
    #   d_vr: [0.40, 0.45, 0.50]
    #   tp:   [0.05, 0.07, 0.10]
    #   d_sl: [0.01, 0.015, 0.02]

    # Too many combos if fully independent. Strategy:
    # Fix mid-vol to baseline, sweep low-vol and high-vol around it.
    # This reduces the search space dramatically.

    # Step 2a: Sweep LOW-VOL params (mid/high fixed to baseline)
    print("\n  ── Step 2a: Sweep LOW-VOL params (mid/high = baseline) ──")
    low_vol_sweeps: list[tuple[str, list[RegimeParams]]] = []
    for d_vr_lo in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for tp_lo in [0.02, 0.03, 0.04, 0.05]:
            for sl_lo in [0.005, 0.008, 0.01, 0.015]:
                name = f"LO_vr{int(d_vr_lo*100)}_tp{int(tp_lo*100)}_sl{int(sl_lo*1000)}"
                params = [
                    RegimeParams(d_vr=d_vr_lo, s_vr=d_vr_lo+0.05, s_dur=4,
                                 d_sl=sl_lo, s_sl=sl_lo*1.5, tp=tp_lo,
                                 d_max_hold=192, s_max_hold=384),
                    baseline_params,  # mid = baseline
                    baseline_params,  # high = baseline
                ]
                low_vol_sweeps.append((name, params))

    print(f"  Scanning {len(low_vol_sweeps)} low-vol variants...")
    t1 = time.time()
    low_results: list[dict] = []
    for idx, (name, params) in enumerate(low_vol_sweeps):
        if (idx + 1) % 20 == 0:
            print(f"    ... {idx+1}/{len(low_vol_sweeps)} ({time.time()-t1:.0f}s)")

        d_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                                params, "D_mom8", 8, 56)
        s_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                                params, "S_d8", 8, 56)
        combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl)+len(s_pnl) > 0 else np.empty(0)
        m = compute_metrics(combined, data_years)
        low_results.append({"name": name, "params": params, "m": m})

    low_results.sort(key=lambda r: r["m"]["calmar"], reverse=True)
    print(f"\n  Top 10 low-vol variants (by Calmar):")
    print(f"  {'Variant':<35} {'N':>5} {'PF':>5} {'EV':>7} {'t':>5} "
          f"{'WR%':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"  {'─'*95}")
    for r in low_results[:10]:
        m = r["m"]
        print(f"  {r['name']:<35} {m['N']:>5} {m['PF']:>5.3f} {m['EV_bp']:>+7.1f} "
              f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['ann_ret_pct']:>+8.1f} "
              f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")
    print(f"  ... (baseline ref: PF={bl_m['PF']:.3f}, Calmar={bl_m['calmar']:.1f})")

    # Step 2b: Sweep HIGH-VOL params (low = best from 2a, mid = baseline)
    print(f"\n  ── Step 2b: Sweep HIGH-VOL params (low = best-2a, mid = baseline) ──")
    best_low = low_results[0]["params"][0]  # Best low-vol params

    high_vol_sweeps: list[tuple[str, list[RegimeParams]]] = []
    for d_vr_hi in [0.35, 0.40, 0.45, 0.50]:
        for tp_hi in [0.05, 0.07, 0.10, 0.15]:
            for sl_hi in [0.01, 0.015, 0.02, 0.025]:
                name = f"HI_vr{int(d_vr_hi*100)}_tp{int(tp_hi*100)}_sl{int(sl_hi*1000)}"
                params = [
                    best_low,          # low = best from 2a
                    baseline_params,   # mid = baseline
                    RegimeParams(d_vr=d_vr_hi, s_vr=d_vr_hi+0.05, s_dur=4,
                                 d_sl=sl_hi, s_sl=sl_hi*1.5, tp=tp_hi,
                                 d_max_hold=192, s_max_hold=384),
                ]
                high_vol_sweeps.append((name, params))

    print(f"  Scanning {len(high_vol_sweeps)} high-vol variants...")
    t2 = time.time()
    high_results: list[dict] = []
    for idx, (name, params) in enumerate(high_vol_sweeps):
        if (idx + 1) % 20 == 0:
            print(f"    ... {idx+1}/{len(high_vol_sweeps)} ({time.time()-t2:.0f}s)")

        d_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                                params, "D_mom8", 8, 56)
        s_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                                params, "S_d8", 8, 56)
        combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl)+len(s_pnl) > 0 else np.empty(0)
        m = compute_metrics(combined, data_years)
        high_results.append({"name": name, "params": params, "m": m})

    high_results.sort(key=lambda r: r["m"]["calmar"], reverse=True)
    print(f"\n  Top 10 high-vol variants (by Calmar):")
    print(f"  {'Variant':<35} {'N':>5} {'PF':>5} {'EV':>7} {'t':>5} "
          f"{'WR%':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"  {'─'*95}")
    for r in high_results[:10]:
        m = r["m"]
        print(f"  {r['name']:<35} {m['N']:>5} {m['PF']:>5.3f} {m['EV_bp']:>+7.1f} "
              f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['ann_ret_pct']:>+8.1f} "
              f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")

    # Step 2c: Sweep MID-VOL params (low/high = best from 2a/2b)
    print(f"\n  ── Step 2c: Sweep MID-VOL params (low = best-2a, high = best-2b) ──")
    best_high = high_results[0]["params"][2]  # Best high-vol params

    mid_vol_sweeps: list[tuple[str, list[RegimeParams]]] = []
    for d_vr_mid in [0.45, 0.48, 0.50, 0.52, 0.55]:
        for tp_mid in [0.03, 0.04, 0.05, 0.07]:
            for sl_mid in [0.008, 0.01, 0.012, 0.015]:
                name = f"MID_vr{int(d_vr_mid*100)}_tp{int(tp_mid*100)}_sl{int(sl_mid*1000)}"
                params = [
                    best_low,          # low = best from 2a
                    RegimeParams(d_vr=d_vr_mid, s_vr=d_vr_mid+0.05, s_dur=4,
                                 d_sl=sl_mid, s_sl=sl_mid*1.5, tp=tp_mid,
                                 d_max_hold=192, s_max_hold=384),
                    best_high,         # high = best from 2b
                ]
                mid_vol_sweeps.append((name, params))

    print(f"  Scanning {len(mid_vol_sweeps)} mid-vol variants...")
    t3 = time.time()
    mid_results: list[dict] = []
    for idx, (name, params) in enumerate(mid_vol_sweeps):
        if (idx + 1) % 20 == 0:
            print(f"    ... {idx+1}/{len(mid_vol_sweeps)} ({time.time()-t3:.0f}s)")

        d_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                                params, "D_mom8", 8, 56)
        s_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_4h,
                                params, "S_d8", 8, 56)
        combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl)+len(s_pnl) > 0 else np.empty(0)
        m = compute_metrics(combined, data_years)
        mid_results.append({"name": name, "params": params, "m": m})

    mid_results.sort(key=lambda r: r["m"]["calmar"], reverse=True)
    print(f"\n  Top 10 mid-vol variants (by Calmar):")
    print(f"  {'Variant':<35} {'N':>5} {'PF':>5} {'EV':>7} {'t':>5} "
          f"{'WR%':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"  {'─'*95}")
    for r in mid_results[:10]:
        m = r["m"]
        print(f"  {r['name']:<35} {m['N']:>5} {m['PF']:>5.3f} {m['EV_bp']:>+7.1f} "
              f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['ann_ret_pct']:>+8.1f} "
              f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")

    # ═════════════════════════════════════════════════════════════════════
    #  Phase 3: Best Adaptive vs Baseline — Detailed Comparison
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  Phase 3: BEST ADAPTIVE vs BASELINE — Detailed Comparison")
    print(f"{'='*120}")

    best_mid = mid_results[0]["params"][1]
    best_adaptive = [best_low, best_mid, best_high]

    print(f"\n  Best adaptive params:")
    for label, p in zip(["LOW_VOL", "MID_VOL", "HIGH_VOL"], best_adaptive):
        print(f"    {label}: d_vr={p.d_vr:.2f} s_vr={p.s_vr:.2f} "
              f"d_sl={p.d_sl:.3f} s_sl={p.s_sl:.3f} "
              f"tp={p.tp:.3f} s_dur={p.s_dur}")

    # Full trade collection for adaptive
    d_trades_adp = collect_regime_trades(data_15m, data_1h, data_4h, regime_4h,
                                         best_adaptive, "D_mom8", 8, 56)
    s_trades_adp = collect_regime_trades(data_15m, data_1h, data_4h, regime_4h,
                                         best_adaptive, "S_d8", 8, 56)
    all_trades_adp = d_trades_adp + s_trades_adp

    # Slot sweep
    print(f"\n  {'Mode':<12} {'Slots':>5} {'N':>5} {'PF':>5} {'EV':>7} {'t':>5} "
          f"{'WR%':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"  {'─'*80}")

    for slots in [5, 10, 15]:
        pnl_bl = portfolio_sim(all_trades_bl, slots)
        m_bl = compute_metrics(pnl_bl, data_years)
        pnl_adp = portfolio_sim(all_trades_adp, slots)
        m_adp = compute_metrics(pnl_adp, data_years)
        print(f"  {'BASELINE':<12} {slots:>5} {m_bl['N']:>5} {m_bl['PF']:>5.3f} "
              f"{m_bl['EV_bp']:>+7.1f} {m_bl['t']:>5.2f} {m_bl['WR']:>5.1f} "
              f"{m_bl['ann_ret_pct']:>+8.1f} {m_bl['max_dd_pct']:>5.1f} {m_bl['calmar']:>7.1f}")
        print(f"  {'ADAPTIVE':<12} {slots:>5} {m_adp['N']:>5} {m_adp['PF']:>5.3f} "
              f"{m_adp['EV_bp']:>+7.1f} {m_adp['t']:>5.2f} {m_adp['WR']:>5.1f} "
              f"{m_adp['ann_ret_pct']:>+8.1f} {m_adp['max_dd_pct']:>5.1f} {m_adp['calmar']:>7.1f}")
        # Delta
        d_ev = m_adp["EV_bp"] - m_bl["EV_bp"]
        d_cal = m_adp["calmar"] - m_bl["calmar"]
        d_n = m_adp["N"] - m_bl["N"]
        print(f"  {'  Δ':<12} {'':>5} {d_n:>+5} {'':>5} {d_ev:>+7.1f} {'':>5} "
              f"{'':>5} {'':>8} {'':>5} {d_cal:>+7.1f}")
        print()

    # Year-by-year comparison (10 slots)
    pnl_bl_10 = portfolio_sim(all_trades_bl, 10)
    pnl_adp_10 = portfolio_sim(all_trades_adp, 10)

    yearly_bl = pnl_by_year(all_trades_bl, n_15m, data_days)
    yearly_adp = pnl_by_year(all_trades_adp, n_15m, data_days)

    print(f"\n  Year-by-Year Comparison (unlimited slots, no portfolio filter):")
    print(f"  {'Year':>6} {'':>3} {'N':>5} {'PF':>5} {'EV(bp)':>7} {'WR%':>5} {'R:R':>4}")
    print(f"  {'─'*50}")
    all_years = sorted(set(list(yearly_bl.keys()) + list(yearly_adp.keys())))
    for yr in all_years:
        bl_yr = yearly_bl.get(yr, np.empty(0))
        adp_yr = yearly_adp.get(yr, np.empty(0))
        if len(bl_yr) < 5 and len(adp_yr) < 5:
            continue
        m_bl_yr = compute_metrics(bl_yr, 1.0) if len(bl_yr) >= 2 else {"N": len(bl_yr), "PF": 0, "EV_bp": 0, "WR": 0, "RR": 0}
        m_adp_yr = compute_metrics(adp_yr, 1.0) if len(adp_yr) >= 2 else {"N": len(adp_yr), "PF": 0, "EV_bp": 0, "WR": 0, "RR": 0}
        print(f"  {yr:>6} BL  {m_bl_yr['N']:>5} {m_bl_yr['PF']:>5.2f} "
              f"{m_bl_yr['EV_bp']:>+7.1f} {m_bl_yr['WR']:>5.1f} {m_bl_yr['RR']:>4.2f}")
        print(f"  {'':>6} ADP {m_adp_yr['N']:>5} {m_adp_yr['PF']:>5.2f} "
              f"{m_adp_yr['EV_bp']:>+7.1f} {m_adp_yr['WR']:>5.1f} {m_adp_yr['RR']:>4.2f}")
        d_ev = m_adp_yr["EV_bp"] - m_bl_yr["EV_bp"]
        marker = " ✓" if d_ev > 0 else " ✗" if d_ev < -5 else ""
        print(f"  {'':>6}   Δ {'':>5} {'':>5} {d_ev:>+7.1f}{marker}")

    # ═════════════════════════════════════════════════════════════════════
    #  Phase 4: Additional ATR pair + regime threshold sweep
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  Phase 4: Regime Threshold Sensitivity (33/67 vs alternatives)")
    print(f"{'='*120}")

    # Test different regime thresholds
    for lo_t, hi_t in [(0.25, 0.75), (0.30, 0.70), (0.33, 0.67), (0.40, 0.60)]:
        regime_alt = build_btc_regime_4h(data_15m, lo_thresh=lo_t, hi_thresh=hi_t)

        # Use best adaptive params from Phase 3
        d_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_alt,
                                best_adaptive, "D_mom8", 8, 56)
        s_pnl = fast_regime_pnl(data_15m, data_1h, data_4h, regime_alt,
                                best_adaptive, "S_d8", 8, 56)
        combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl)+len(s_pnl) > 0 else np.empty(0)
        m = compute_metrics(combined, data_years)

        label = f"Regime {lo_t:.2f}/{hi_t:.2f}"
        print(f"  {label:<25} N={m['N']:>5}  PF={m['PF']:.3f}  "
              f"EV={m['EV_bp']:+.1f}bp  WR={m['WR']:.1f}%  "
              f"AnnRet={m['ann_ret_pct']:+.1f}%  Calmar={m['calmar']:.1f}")

    # ═════════════════════════════════════════════════════════════════════
    #  Summary
    # ═════════════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*120}")
    print(f"  R42 Complete — {total_time:.0f}s total [{commit}]")
    print(f"{'='*120}")

    bl_m_final = compute_metrics(bl_pnl_10s, data_years)
    adp_m_final = compute_metrics(pnl_adp_10, data_years)

    print(f"\n  BASELINE:  PF={bl_m_final['PF']:.3f}  EV={bl_m_final['EV_bp']:+.1f}bp  "
          f"Calmar={bl_m_final['calmar']:.1f}  N={bl_m_final['N']}")
    print(f"  ADAPTIVE:  PF={adp_m_final['PF']:.3f}  EV={adp_m_final['EV_bp']:+.1f}bp  "
          f"Calmar={adp_m_final['calmar']:.1f}  N={adp_m_final['N']}")

    d_ev = adp_m_final["EV_bp"] - bl_m_final["EV_bp"]
    d_cal = adp_m_final["calmar"] - bl_m_final["calmar"]
    verdict = "IMPROVEMENT" if d_ev > 5 and d_cal > 0 else "NO IMPROVEMENT" if d_ev < -5 else "MARGINAL"

    print(f"\n  Verdict: {verdict}")
    print(f"    EV delta:    {d_ev:+.1f} bp")
    print(f"    Calmar delta: {d_cal:+.1f}")

    if verdict == "IMPROVEMENT":
        print(f"\n  → Regime-adaptive params improve over static baseline.")
        print(f"    Review year-by-year breakdown above to confirm no regression.")
        print(f"    Next: implement regime detection in live VolStraddleStrategy.")
    elif verdict == "MARGINAL":
        print(f"\n  → Marginal difference. Static params may be robust enough.")
        print(f"    Regime adaptation adds complexity without clear benefit.")
    else:
        print(f"\n  → Static params are better. Regime adaptation hurts performance.")
        print(f"    VR squeeze may already be regime-agnostic by design.")


if __name__ == "__main__":
    main()
