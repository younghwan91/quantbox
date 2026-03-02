"""R38 — Scaling Research: Coins × Slots × Correlation × Leverage

Research Questions:
1. A-VR5_35 validation: WR, RR, TPD, AnnRet (pending from R37)
2. Coin expansion: Does adding mid/small-cap symbols improve portfolio?
3. Slot expansion: max_concurrent = 2→6, does it compound or just correlate?
4. Cross-symbol correlation: Are simultaneous trades correlated?
5. Leverage vs Diversification: N slots × 1x vs 1 slot × Nx

Data:
  - 50 symbols × 6yr 15m (cache_extended) — Current
  - 38 symbols × 15m (cache_illiquid) — New, mid/small-cap
  - All 88 combined

Author: QuantBox R38 (frequency-boost branch)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.active.mtf_vol_straddle_r31 import (
    WARMUP_15m,
    WARMUP_1h,
    WARMUP_4h,
    build_resampled,
    compute_momentum,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    load_15m,
    make_direction_momentum,
    pf_from_pnl,
    bootstrap_pf,
    precompute,
    vol_percentile,
)

ROOT = Path(__file__).resolve().parent.parent.parent

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

COST_DIRECTIONAL = 0.0008   # 2 × 0.04%
COST_STRADDLE = 0.0016      # 4 × 0.04%

TF_15m = ("15m", 1, WARMUP_15m, 192, 384)
TF_1h  = ("1h",  4, WARMUP_1h,  48,  96)
TF_4h  = ("4h", 16, WARMUP_4h,  12,  24)
G_3TF  = [TF_15m, TF_1h, TF_4h]

# A-VR5_35 config (selected from R37)
CFG_A = {"sp": 5, "lp": 35, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
         "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05}

# Baseline (current deployed)
CFG_BL = {"sp": 7, "lp": 50, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
          "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05}


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Sim (from R37, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
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
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
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


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection (with symbol + timing info for portfolio sim)
# ══════════════════════════════════════════════════════════════════════════════


def collect_trades(
    cfg: dict,
    data_15m: dict, data_1h: dict, data_4h: dict,
    strat: str,
) -> list[tuple[str, int, int, float]]:
    """Collect all trades as (symbol, entry_15m, exit_15m, pnl).

    Unlike R37 _run_variant which returns flat pnl array,
    this preserves symbol and timing for portfolio simulation.
    """
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    d_sl = cfg["d_sl"]
    s_sl = cfg["s_sl"]
    tp = cfg["tp"]
    d_vr = cfg["d_vr"]
    s_vr = cfg["s_vr"]
    s_dur = cfg["s_dur"]

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s in G_3TF:
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

            sl = s_sl if is_straddle else d_sl
            mh = mh_s if is_straddle else mh_d

            if is_straddle:
                sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
            else:
                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

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


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio Simulation (max_concurrent filtering)
# ══════════════════════════════════════════════════════════════════════════════


def portfolio_sim(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> np.ndarray:
    """Filter trades by max concurrent positions, no same-symbol overlap."""
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted_pnl: list[float] = []
    active: list[tuple[str, int]] = []  # (symbol, exit_bar)

    for sym, entry, exit_b, pnl_val in trades:
        active = [(s, e) for s, e in active if e > entry]
        sym_active = any(s == sym for s, _ in active)
        if sym_active:
            continue
        if len(active) >= max_concurrent:
            continue
        active.append((sym, exit_b))
        accepted_pnl.append(pnl_val)

    return np.array(accepted_pnl) if accepted_pnl else np.empty(0)


def portfolio_sim_with_details(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> tuple[np.ndarray, list[tuple[str, int, int, float]]]:
    """Portfolio sim that also returns accepted trades with metadata."""
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted: list[tuple[str, int, int, float]] = []
    active: list[tuple[str, int]] = []

    for sym, entry, exit_b, pnl_val in trades:
        active = [(s, e) for s, e in active if e > entry]
        sym_active = any(s == sym for s, _ in active)
        if sym_active:
            continue
        if len(active) >= max_concurrent:
            continue
        active.append((sym, exit_b))
        accepted.append((sym, entry, exit_b, pnl_val))

    pnl = np.array([t[3] for t in accepted]) if accepted else np.empty(0)
    return pnl, accepted


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics (enhanced from R37)
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
    n = len(pnl)
    if n < 2:
        return {"N": n, "PF": 0.0, "EV_bp": 0.0, "t": 0.0, "WR": 0.0,
                "RR": 0.0, "ret_pct": 0.0, "max_dd_pct": 0.0, "tpd": 0.0,
                "ann_ret_pct": 0.0, "calmar": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n * 100
    avg_win = float(np.mean(wins)) * 100 if len(wins) > 0 else 0.0
    avg_loss = abs(float(np.mean(losses))) * 100 if len(losses) > 0 else 1e-9
    rr = avg_win / avg_loss if avg_loss > 0 else 999.0
    pf = pf_from_pnl(pnl)
    ev = float(np.mean(pnl)) * 10000
    se = float(np.std(pnl) / np.sqrt(n)) * 10000
    t_stat = ev / se if se > 0 else 0.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnl:
        equity *= (1.0 + p)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    total_ret = (equity - 1.0) * 100
    tpd = n / (data_years * 365) if data_years > 0 else 0.0
    ann_ret = (equity ** (1.0 / data_years) - 1.0) * 100 if data_years > 0 else 0.0
    calmar = ann_ret / (max_dd * 100) if max_dd > 0 else 0.0

    return {
        "N": n, "PF": round(pf, 3), "EV_bp": round(ev, 1),
        "t": round(t_stat, 2), "WR": round(wr, 1), "RR": round(rr, 2),
        "avg_win_pct": round(avg_win, 3), "avg_loss_pct": round(avg_loss, 3),
        "ret_pct": round(total_ret, 1), "max_dd_pct": round(max_dd * 100, 1),
        "tpd": round(tpd, 2), "ann_ret_pct": round(ann_ret, 1),
        "calmar": round(calmar, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loaders
# ══════════════════════════════════════════════════════════════════════════════


def load_illiquid_15m() -> dict:
    """Load mid/small-cap symbols from cache_illiquid."""
    cache_dir = ROOT / "data" / "cache_illiquid"
    data: dict = {}
    for f in sorted(cache_dir.glob("*_15m_*.npz")):
        sym = f.name.split("_15m_")[0]
        if sym in data:
            # Keep longest
            if _extract_days(f) <= data[sym]["days"]:
                continue
        npz = np.load(f)
        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        h = npz["high"].astype(np.float64)
        lo = npz["low"].astype(np.float64)
        v = npz["volume"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {
            "close": c, "high": h, "low": lo, "volume": v,
            "bars": len(c), "days": _extract_days(f),
        }
    return data


def _extract_days(path: Path) -> int:
    """Extract days from filename like BTCUSDT_15m_2190d_20260220.npz."""
    name = path.stem
    for part in name.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def precompute_all_vr(datasets: list[dict], pairs: list[tuple[int, int]]) -> None:
    """Precompute VR for all ATR pairs in all datasets."""
    for dataset in datasets:
        for sym, d in dataset.items():
            c, h, lo = d["close"], d["high"], d["low"]
            for sp, lp in pairs:
                key = f"vr_{sp}_{lp}"
                if key not in d:
                    d[key] = compute_vr(c, h, lo, sp, lp)


# ══════════════════════════════════════════════════════════════════════════════
#  Correlation Analysis
# ══════════════════════════════════════════════════════════════════════════════


def compute_pairwise_correlation(
    trades: list[tuple[str, int, int, float]],
    bin_size: int = 96,  # 96 × 15m = 1 day
) -> tuple[float, int, dict[str, float]]:
    """Compute pairwise correlation of trade PnLs binned by time.

    Returns: (mean_correlation, n_pairs, per_pair_corr)
    """
    # Bin trades by symbol and time bucket
    sym_buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sym, entry, exit_b, pnl_val in trades:
        bucket = entry // bin_size
        sym_buckets[sym][bucket].append(pnl_val)

    # Average PnL per bucket per symbol
    sym_series: dict[str, dict[int, float]] = {}
    for sym, buckets in sym_buckets.items():
        sym_series[sym] = {b: np.mean(v) for b, v in buckets.items()}

    # Pairwise correlation
    symbols = sorted(sym_series.keys())
    all_corrs: list[float] = []
    pair_corrs: dict[str, float] = {}

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1, s2 = symbols[i], symbols[j]
            common = set(sym_series[s1].keys()) & set(sym_series[s2].keys())
            if len(common) < 10:
                continue
            v1 = np.array([sym_series[s1][b] for b in sorted(common)])
            v2 = np.array([sym_series[s2][b] for b in sorted(common)])
            if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
                continue
            corr = float(np.corrcoef(v1, v2)[0, 1])
            all_corrs.append(corr)
            pair_corrs[f"{s1[:6]}-{s2[:6]}"] = round(corr, 3)

    mean_corr = float(np.mean(all_corrs)) if all_corrs else 0.0
    return mean_corr, len(all_corrs), pair_corrs


def compute_concurrent_pnl_correlation(
    trades: list[tuple[str, int, int, float]],
) -> float:
    """Measure correlation of simultaneously open trades.

    For each trade, find other trades overlapping in time,
    compute correlation of their PnLs. This captures the real
    diversification benefit of multi-slot.
    """
    sorted_trades = sorted(trades, key=lambda x: x[1])
    overlap_pairs: list[tuple[float, float]] = []

    for i in range(len(sorted_trades)):
        sym_i, entry_i, exit_i, pnl_i = sorted_trades[i]
        for j in range(i + 1, len(sorted_trades)):
            sym_j, entry_j, exit_j, pnl_j = sorted_trades[j]
            if entry_j > exit_i:
                break  # No more overlaps
            if sym_i == sym_j:
                continue  # Same symbol, skip
            # Overlapping trades
            overlap_pairs.append((pnl_i, pnl_j))

    if len(overlap_pairs) < 10:
        return 0.0

    v1 = np.array([p[0] for p in overlap_pairs])
    v2 = np.array([p[1] for p in overlap_pairs])
    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        return 0.0
    return float(np.corrcoef(v1, v2)[0, 1])


# ══════════════════════════════════════════════════════════════════════════════
#  Leverage vs Diversification Comparison
# ══════════════════════════════════════════════════════════════════════════════


def leverage_vs_div(
    trades: list[tuple[str, int, int, float]],
    data_years: float,
    slot_range: list[int] = [1, 2, 3, 4, 5, 6],
) -> list[dict]:
    """Compare N slots × 1x leverage vs 1 slot × Nx leverage.

    For N slots: portfolio_sim with max_concurrent=N.
    For 1×Nx: portfolio_sim with max_concurrent=1, but PnL scaled by N.
    """
    results: list[dict] = []

    for n_slots in slot_range:
        # Multi-slot: spread across N positions
        pnl_multi = portfolio_sim(trades, n_slots)
        m = compute_metrics(pnl_multi, data_years)
        m["label"] = f"{n_slots}slot×1x"
        m["n_slots"] = n_slots
        results.append(m)

        # Single-slot with Nx leverage: same N trades, each PnL × N
        if n_slots > 1:
            pnl_1slot = portfolio_sim(trades, 1)
            # Scale PnL by N (simulating Nx leverage on 1 trade)
            pnl_levered = pnl_1slot * n_slots
            m_lev = compute_metrics(pnl_levered, data_years)
            m_lev["label"] = f"1slot×{n_slots}x"
            m_lev["n_slots"] = 1
            results.append(m_lev)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main Research
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    t0 = time.time()
    print(f"R38 Scaling Research — commit {commit}")
    print("=" * 110)

    # ── Load Data ──
    print("\n[1/6] Loading data...")
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)

    vr_pairs = [(5, 35), (7, 50)]
    precompute_all_vr([data_15m, data_1h, data_4h], vr_pairs)

    n_sym = len(data_15m)
    total_bars = sum(d["bars"] for d in data_15m.values())
    print(f"  Core: {n_sym} symbols, {total_bars:,} 15m bars, {data_years:.1f}yr")

    # Load illiquid
    illiq_15m = load_illiquid_15m()
    if illiq_15m:
        illiq_years = max(d.get("days", 0) for d in illiq_15m.values()) / 365
        illiq_1h = build_resampled(illiq_15m, 4)
        illiq_4h = build_resampled(illiq_15m, 16)
        precompute(illiq_15m)
        precompute(illiq_1h)
        precompute(illiq_4h)
        precompute_all_vr([illiq_15m, illiq_1h, illiq_4h], vr_pairs)
        print(f"  Illiquid: {len(illiq_15m)} symbols, "
              f"{sum(d['bars'] for d in illiq_15m.values()):,} 15m bars, "
              f"{illiq_years:.1f}yr")
    else:
        illiq_1h, illiq_4h = {}, {}
        illiq_years = 0
        print("  Illiquid: 0 symbols (no data found)")

    # Merged data
    all_15m = {**data_15m, **illiq_15m}
    all_1h = {**data_1h, **illiq_1h}
    all_4h = {**data_4h, **illiq_4h}
    all_years = max(data_years, illiq_years) if illiq_15m else data_years
    print(f"  Combined: {len(all_15m)} symbols")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 1: A-VR5_35 Full Validation (pending from R37)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART 1: A-VR5_35 Full Validation")
    print("=" * 110)

    for label, cfg in [("A-VR5_35", CFG_A), ("Baseline(7/50)", CFG_BL)]:
        d_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "D_mom8")
        s_trades = collect_trades(cfg, data_15m, data_1h, data_4h, "S_d8")
        combined = d_trades + s_trades

        d_pnl = np.array([t[3] for t in d_trades]) if d_trades else np.empty(0)
        s_pnl = np.array([t[3] for t in s_trades]) if s_trades else np.empty(0)
        c_pnl = np.array([t[3] for t in combined]) if combined else np.empty(0)

        print(f"\n  ── {label} (Full 6yr, {n_sym} coins, no slot limit) ──")
        header = (f"  {'Strategy':<10} {'N':>6} {'PF':>6} {'EV_bp':>7} {'t':>5} "
                  f"{'WR%':>5} {'R:R':>5} {'AvgW%':>7} {'AvgL%':>7} "
                  f"{'TPD':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
        print(header)
        print("  " + "─" * 105)

        for tag, pnl in [("D_mom8", d_pnl), ("S_d8", s_pnl), ("Combined", c_pnl)]:
            m = compute_metrics(pnl, data_years)
            print(f"  {tag:<10} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
                  f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['RR']:>5.2f} "
                  f"{m['avg_win_pct']:>7.3f} {m['avg_loss_pct']:>7.3f} "
                  f"{m['tpd']:>5.2f} {m['ann_ret_pct']:>+8.1f} "
                  f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")

    # OOS split
    print(f"\n  ── IS/OOS Split (67/33) ──")
    # Manual OOS split
    is_15m: dict = {}
    oos_15m: dict = {}
    is_1h: dict = {}
    oos_1h: dict = {}
    is_4h: dict = {}
    oos_4h: dict = {}

    for sym in data_15m:
        n_bars = len(data_15m[sym]["close"])
        cut = int(n_bars * 0.67)
        for src, is_d, oos_d in [
            (data_15m, is_15m, oos_15m),
            (data_1h, is_1h, oos_1h),
            (data_4h, is_4h, oos_4h),
        ]:
            if sym not in src:
                continue
            d = src[sym]
            n_tf = len(d["close"])
            ratio = n_bars // n_tf if n_tf > 0 else 1
            cut_tf = cut // ratio
            for start, end_b, dst in [(0, cut_tf, is_d), (cut_tf, n_tf, oos_d)]:
                seg_len = end_b - start
                if seg_len < 100:
                    continue
                d_seg: dict = {}
                for key in ("close", "high", "low", "volume"):
                    d_seg[key] = d[key][start:end_b].copy()
                d_seg["bars"] = seg_len
                d_seg["days"] = d.get("days", 0)
                c = d_seg["close"]
                h = d_seg["high"]
                lo = d_seg["low"]
                v = d_seg["volume"]
                d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
                d_seg["mom_8"] = compute_momentum(c, 8)
                d_seg["vol_pct"] = vol_percentile(v, min(200, seg_len // 2))
                d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)
                for sp, lp in vr_pairs:
                    d_seg[f"vr_{sp}_{lp}"] = compute_vr(c, h, lo, sp, lp)
                dst[sym] = d_seg

    is_years = data_years * 0.67
    oos_years = data_years * 0.33

    header2 = (f"  {'Config':<18} {'Split':>4} {'N':>6} {'PF':>6} {'EV_bp':>7} "
               f"{'t':>5} {'WR%':>5} {'R:R':>5} {'TPD':>5} {'AnnRet%':>8} {'DD%':>5}")
    print(header2)
    print("  " + "─" * 90)

    for label, cfg in [("A-VR5_35", CFG_A), ("Baseline(7/50)", CFG_BL)]:
        for split_label, s15, s1h, s4h, yrs in [
            ("IS", is_15m, is_1h, is_4h, is_years),
            ("OOS", oos_15m, oos_1h, oos_4h, oos_years),
        ]:
            d_t = collect_trades(cfg, s15, s1h, s4h, "D_mom8")
            s_t = collect_trades(cfg, s15, s1h, s4h, "S_d8")
            c_pnl = np.array([t[3] for t in d_t + s_t]) if d_t + s_t else np.empty(0)
            m = compute_metrics(c_pnl, yrs)
            print(f"  {label:<18} {split_label:>4} {m['N']:>6} {m['PF']:>6.3f} "
                  f"{m['EV_bp']:>+7.1f} {m['t']:>5.2f} {m['WR']:>5.1f} "
                  f"{m['RR']:>5.2f} {m['tpd']:>5.2f} {m['ann_ret_pct']:>+8.1f} "
                  f"{m['max_dd_pct']:>5.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 2: Coin Expansion (50 → 88 symbols)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART 2: Coin Expansion")
    print("=" * 110)

    sets = [
        ("Core 50", data_15m, data_1h, data_4h, data_years),
    ]
    if illiq_15m:
        sets.append(("Illiquid 38", illiq_15m, illiq_1h, illiq_4h, illiq_years))
        sets.append(("Combined 88", all_15m, all_1h, all_4h, all_years))

    header3 = (f"  {'Dataset':<15} {'N':>6} {'PF':>6} {'EV_bp':>7} {'t':>5} "
               f"{'WR%':>5} {'R:R':>5} {'TPD':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print(f"\n  A-VR5_35 across different symbol pools (no slot limit)")
    print(header3)
    print("  " + "─" * 95)

    all_trade_sets: dict[str, list] = {}
    for ds_name, d15, d1h, d4h, yrs in sets:
        d_t = collect_trades(CFG_A, d15, d1h, d4h, "D_mom8")
        s_t = collect_trades(CFG_A, d15, d1h, d4h, "S_d8")
        combined = d_t + s_t
        all_trade_sets[ds_name] = combined
        c_pnl = np.array([t[3] for t in combined]) if combined else np.empty(0)
        m = compute_metrics(c_pnl, yrs)
        print(f"  {ds_name:<15} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
              f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['RR']:>5.2f} "
              f"{m['tpd']:>5.2f} {m['ann_ret_pct']:>+8.1f} "
              f"{m['max_dd_pct']:>5.1f} {m['calmar']:>7.1f}")

    # Per-symbol breakdown for illiquid
    if illiq_15m:
        print(f"\n  Per-symbol trade count (Illiquid, combined D+S):")
        sym_counts: dict[str, int] = defaultdict(int)
        for sym, _, _, _ in all_trade_sets.get("Illiquid 38", []):
            sym_counts[sym] += 1
        top = sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for sym, cnt in top:
            print(f"    {sym:<20} {cnt:>4} trades")
        if len(sym_counts) > 15:
            print(f"    ... and {len(sym_counts) - 15} more symbols")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 3: Slot Expansion (max_concurrent = 1..8)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART 3: Slot Expansion (max_concurrent)")
    print("=" * 110)

    # Use core 50 trades
    core_trades = all_trade_sets.get("Core 50", [])

    print(f"\n  Portfolio-level simulation with max_concurrent limits (Core 50, A-VR5_35)")
    header4 = (f"  {'Slots':>5} {'N':>6} {'PF':>6} {'EV_bp':>7} {'t':>5} "
               f"{'WR%':>5} {'TPD':>5} {'AnnRet%':>8} {'DD%':>5} {'Calmar':>7} {'FillRate':>9}")
    print(header4)
    print("  " + "─" * 85)

    unlimited_n = len(core_trades)
    for slots in [1, 2, 3, 4, 5, 6, 8, 999]:
        pnl = portfolio_sim(core_trades, slots)
        m = compute_metrics(pnl, data_years)
        fill_rate = m["N"] / unlimited_n * 100 if unlimited_n > 0 else 0
        slot_label = "∞" if slots == 999 else str(slots)
        print(f"  {slot_label:>5} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
              f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['tpd']:>5.2f} "
              f"{m['ann_ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} "
              f"{m['calmar']:>7.1f} {fill_rate:>8.1f}%")

    # Combined 88 if available
    if "Combined 88" in all_trade_sets:
        combo_trades = all_trade_sets["Combined 88"]
        print(f"\n  Portfolio-level simulation (Combined 88, A-VR5_35)")
        print(header4)
        print("  " + "─" * 85)
        unlimited_n_88 = len(combo_trades)
        for slots in [1, 2, 3, 4, 5, 6, 8, 999]:
            pnl = portfolio_sim(combo_trades, slots)
            m = compute_metrics(pnl, all_years)
            fill_rate = m["N"] / unlimited_n_88 * 100 if unlimited_n_88 > 0 else 0
            slot_label = "∞" if slots == 999 else str(slots)
            print(f"  {slot_label:>5} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
                  f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['tpd']:>5.2f} "
                  f"{m['ann_ret_pct']:>+8.1f} {m['max_dd_pct']:>5.1f} "
                  f"{m['calmar']:>7.1f} {fill_rate:>8.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 4: Cross-Symbol Correlation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART 4: Cross-Symbol Correlation")
    print("=" * 110)

    mean_corr, n_pairs, pair_corrs = compute_pairwise_correlation(core_trades)
    concurrent_corr = compute_concurrent_pnl_correlation(core_trades)

    print(f"\n  Pairwise daily-binned correlation:")
    print(f"    Mean ρ = {mean_corr:.3f}  (across {n_pairs} symbol pairs)")
    print(f"  Concurrent trade PnL correlation:")
    print(f"    ρ_concurrent = {concurrent_corr:.3f}")

    # Top/bottom 5 pairs
    if pair_corrs:
        sorted_pairs = sorted(pair_corrs.items(), key=lambda x: x[1])
        print(f"\n  Most negative correlation (best diversification):")
        for pair, c in sorted_pairs[:5]:
            print(f"    {pair:<20} ρ = {c:+.3f}")
        print(f"\n  Most positive correlation (highest redundancy):")
        for pair, c in sorted_pairs[-5:]:
            print(f"    {pair:<20} ρ = {c:+.3f}")

    # Interpretation for aggressive trader
    if concurrent_corr < 0.3:
        div_verdict = "LOW correlation → Multi-slot STRONGLY preferred over leverage"
    elif concurrent_corr < 0.5:
        div_verdict = "MODERATE correlation → Multi-slot still preferred, some diversification"
    else:
        div_verdict = "HIGH correlation → Leverage may be more capital-efficient"
    print(f"\n  → Verdict: {div_verdict}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 5: Leverage vs Diversification
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART 5: Leverage vs Diversification")
    print("=" * 110)

    lev_results = leverage_vs_div(core_trades, data_years)

    print(f"\n  {'Label':<12} {'N':>6} {'PF':>6} {'EV_bp':>7} {'TPD':>5} "
          f"{'AnnRet%':>8} {'DD%':>5} {'Calmar':>7}")
    print("  " + "─" * 65)
    for r in lev_results:
        print(f"  {r['label']:<12} {r['N']:>6} {r['PF']:>6.3f} {r['EV_bp']:>+7.1f} "
              f"{r['tpd']:>5.2f} {r['ann_ret_pct']:>+8.1f} "
              f"{r['max_dd_pct']:>5.1f} {r['calmar']:>7.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 6: Summary & Recommendations
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART 6: Aggressive Deployment Recommendations")
    print("=" * 110)

    # Best slot count by Calmar
    best_slot_calmar = 0
    best_slot_n = 2
    for slots in range(1, 9):
        pnl = portfolio_sim(core_trades, slots)
        m = compute_metrics(pnl, data_years)
        if m["calmar"] > best_slot_calmar:
            best_slot_calmar = m["calmar"]
            best_slot_n = slots

    # Marginal return per additional slot
    print(f"\n  Marginal return per additional slot:")
    prev_ann = 0.0
    for slots in range(1, 9):
        pnl = portfolio_sim(core_trades, slots)
        m = compute_metrics(pnl, data_years)
        marginal = m["ann_ret_pct"] - prev_ann
        delta = f"+{marginal:.1f}%" if slots > 1 else f"base"
        print(f"    {slots} slots: AnnRet={m['ann_ret_pct']:+.1f}% DD={m['max_dd_pct']:.1f}% "
              f"Calmar={m['calmar']:.1f} ({delta})")
        prev_ann = m["ann_ret_pct"]

    # Optimal config summary
    pnl_opt = portfolio_sim(core_trades, best_slot_n)
    m_opt = compute_metrics(pnl_opt, data_years)
    print(f"\n  ★ Optimal aggressive config:")
    print(f"    Strategy: A-VR5_35 (sp=5, lp=35)")
    print(f"    Slots: {best_slot_n} (Calmar {best_slot_calmar:.1f})")
    print(f"    N={m_opt['N']} trades, TPD={m_opt['tpd']:.2f}")
    print(f"    AnnRet={m_opt['ann_ret_pct']:+.1f}% DD={m_opt['max_dd_pct']:.1f}%")
    print(f"    Concurrent ρ = {concurrent_corr:.3f}")

    if concurrent_corr < 0.3:
        print(f"    → Low correlation: {best_slot_n}slot×1x >> 1slot×{best_slot_n}x")
    elif concurrent_corr < 0.5:
        print(f"    → Moderate corr: {best_slot_n}slot×1x > 1slot×{best_slot_n}x")
    else:
        print(f"    → High corr: Consider hybrid (fewer slots + moderate leverage)")

    elapsed = time.time() - t0
    print(f"\n  R38 completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
