#!/usr/bin/env python3
"""Autoresearch: Comprehensive Robustness Test for All Successful Strategies.

Tests each strategy under a matrix of realistic conditions:
  - Cost: 1x, 2x, 3x (fee + slippage)
  - Entry slippage: 0%, 0.1%, 0.2% (adverse entry price on breakout bars)
  - Combined worst-case: 3x cost + 0.2% slippage

Primary metric: PF at worst-case scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.sim_core import run_ma_cluster_live  # noqa: E402
from quantbox.backtest.trade_sim import (  # noqa: E402
    G_4TF,
    build_resampled,
    bootstrap_pf,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import compute_momentum  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = ROOT / "data" / "cache_extended"
DATA_YEARS = 6.0
SLOT_LIMIT = 10


def load_data():
    cfg = VolStraddleConfig()
    whitelist = set(cfg.symbol_whitelist)
    data_15m: dict = {}
    max_bars = 0
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        d = np.load(f, allow_pickle=False)
        bars = len(d["close"])
        if bars > max_bars:
            max_bars = bars
        data_15m[sym] = {
            "close": d["close"].astype(np.float64),
            "high": d["high"].astype(np.float64),
            "low": d["low"].astype(np.float64),
            "volume": d["volume"].astype(np.float64),
        }
    for sym in data_15m:
        data_15m[sym]["_offset"] = max_bars - len(data_15m[sym]["close"])
        data_15m[sym]["mom_8"] = compute_momentum(data_15m[sym]["close"], 8)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    for ds in [data_1h, data_4h]:
        for d in ds.values():
            d["mom_8"] = compute_momentum(d["close"], 8)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    return data_15m, data_1h, data_4h, max_bars


def collect_existing_trades(data_15m, data_1h, data_4h):
    cfg = VolStraddleConfig()
    d_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "s_vr": cfg.s_vr_threshold,
        "s_dur": cfg.s_squeeze_dur,
        "d_sl": cfg.d_sl,
        "s_sl": cfg.s_sl,
        "tp": cfg.d_tp,
        "d_trail_activate": cfg.d_trail_activate,
        "d_trail_distance": cfg.d_trail_distance,
        "s_trail_activate": cfg.s_trail_activate,
        "s_trail_distance": cfg.s_trail_distance,
        "funding_rate_8h": cfg.default_funding_rate_8h,
    }
    vr_trades = collect_trades(
        d_cfg, data_15m, data_1h, data_4h, "D_mom8", tf_group=G_4TF
    )
    # 라이브 규칙(check_ma_cluster 호출)으로 백테스트한다. 재구현판
    # collect_ma_cluster_trades 는 클러스터 에피소드당 재진입으로 2.32x 과대 —
    # autoresearch 가 ma_cluster 를 과대평가해 잘못된 포트폴리오를 학습했다. (2026-07-18)
    ma_trades = run_ma_cluster_live(data_15m, data_4h)
    return vr_trades + ma_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Trade sim WITH entry slippage support
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_with_slippage(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    dire: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,
    trail_distance: float,
    warmup: int,
    entry_slippage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trade sim with entry slippage — adverse entry price on signal bars.

    entry_slippage: fraction of price added adversely.
    LONG:  entry = close[i] * (1 + entry_slippage)  — pay more
    SHORT: entry = close[i] * (1 - entry_slippage)  — get less
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        base_entry = close[i]
        if base_entry <= 0:
            i += 1
            continue
        d = dire[i]
        # Apply adverse entry slippage
        if d == 1:
            entry = base_entry * (1.0 + entry_slippage)
        else:
            entry = base_entry * (1.0 - entry_slippage)

        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                else:
                    if low[j] <= entry * (1.0 - sl):
                        trade_pnl = -sl
                        eb = j
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
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
#  Signal generators for each strategy
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_vol_sma(volume, period):
    n = len(volume)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += volume[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += volume[i] - volume[i - period]
        out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def gen_donchian_vol(close, high, low, volume, period, mom_min, vol_sma, vol_mult):
    n = len(close)
    sig = np.zeros(n, dtype=np.int64)
    dire = np.zeros(n, dtype=np.int64)
    for i in range(period, n):
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 0:
            continue
        if volume[i] < vol_mult * vol_sma[i]:
            continue
        chan_high = -1e18
        chan_low = 1e18
        for j in range(i - period, i):
            if high[j] > chan_high:
                chan_high = high[j]
            if low[j] < chan_low:
                chan_low = low[j]
        if close[i] <= 0:
            continue
        if close[i] > chan_high:
            breakout_pct = (close[i] - chan_high) / close[i]
            if breakout_pct > mom_min:
                sig[i] = 1
                dire[i] = 1
        elif close[i] < chan_low:
            breakout_pct = (chan_low - close[i]) / close[i]
            if breakout_pct > mom_min:
                sig[i] = 1
                dire[i] = -1
    return sig, dire


@numba.jit(nopython=True, cache=True)
def gen_high_vol_momentum(vr, momentum, vr_min, mom_min):
    n = len(vr)
    sig = np.zeros(n, dtype=np.int64)
    dire = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if np.isnan(vr[i]) or np.isnan(momentum[i]):
            continue
        if vr[i] > vr_min and abs(momentum[i]) > mom_min:
            sig[i] = 1
            dire[i] = 1 if momentum[i] > 0 else -1
    return sig, dire


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy definitions
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "donchian_vol": {
        "tf": "4h",
        "tf_ratio": 16,
        "sl": 0.015,
        "tp": 0.15,
        "trail_activate": 0.02,
        "trail_distance": 0.008,
        "max_hold": 24,
        "warmup": 55,
    },
    "high_vol_momentum": {
        "tf": "1h",
        "tf_ratio": 4,
        "sl": 0.015,
        "tp": 0.15,
        "trail_activate": 0.03,
        "trail_distance": 0.01,
        "max_hold": 48,
        "warmup": 60,
    },
}


def collect_strategy_trades(
    name: str,
    cfg: dict,
    data_tf: dict,
    data_15m: dict,
    cost: float,
    entry_slippage: float,
) -> list[tuple[str, int, int, float]]:
    """Collect trades for a strategy with given cost and slippage."""
    tf_ratio = cfg["tf_ratio"]
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_tf.keys()):
        d = data_tf[sym]

        if name == "donchian_vol":
            if "volume" not in d:
                continue
            vol_sma = compute_vol_sma(d["volume"], 20)
            sig, dire = gen_donchian_vol(
                d["close"], d["high"], d["low"], d["volume"],
                50, 0.02, vol_sma, 2.5,
            )
        elif name == "high_vol_momentum":
            vr_key = "vr_8_56"
            if vr_key not in d or "mom_8" not in d:
                continue
            sig, dire = gen_high_vol_momentum(d[vr_key], d["mom_8"], 2.0, 0.03)
        else:
            continue

        p, e, x = sim_with_slippage(
            d["close"], d["high"], d["low"],
            sig, dire,
            cfg["sl"], cfg["tp"], cfg["max_hold"], cost,
            cfg["trail_activate"], cfg["trail_distance"],
            cfg["warmup"], entry_slippage,
        )

        offset = int(data_15m[sym].get("_offset", 0))
        for k in range(len(p)):
            entry_bar = int(e[k]) * tf_ratio + offset
            exit_bar = int(x[k]) * tf_ratio + tf_ratio - 1 + offset
            all_trades.append((sym, entry_bar, exit_bar, float(p[k])))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Main — Robustness matrix
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 72)
    print("  COMPREHENSIVE ROBUSTNESS TEST — ALL SUCCESSFUL STRATEGIES")
    print("=" * 72)
    print("\nLoading data...")
    data_15m, data_1h, data_4h, max_bars = load_data()

    tf_data = {"1h": data_1h, "4h": data_4h}

    print("Collecting existing strategy trades...")
    existing_trades = collect_existing_trades(data_15m, data_1h, data_4h)
    existing_pnl = portfolio_sim(existing_trades, SLOT_LIMIT)
    existing_m = compute_metrics(existing_pnl, DATA_YEARS)
    print(f"  Existing baseline: N={existing_m['N']}  PF={existing_m['PF']:.3f}\n")

    # Test matrix
    cost_levels = [
        ("1x", 0.0008),
        ("2x", 0.0016),
        ("3x", 0.0024),
    ]
    slippage_levels = [
        ("0%", 0.0),
        ("0.1%", 0.001),
        ("0.2%", 0.002),
    ]

    for strat_name, strat_cfg in STRATEGIES.items():
        data_tf = tf_data[strat_cfg["tf"]]

        print(f"\n{'=' * 72}")
        print(f"  STRATEGY: {strat_name} ({strat_cfg['tf']})")
        print(f"  SL={strat_cfg['sl']*100:.1f}%  Trail={strat_cfg['trail_activate']*100:.0f}%/{strat_cfg['trail_distance']*100:.1f}%")
        print(f"{'=' * 72}")
        print()

        # Header
        print(f"  {'Cost':>6} {'Slip':>6} | {'PF':>6} {'N':>6} {'WR':>6} {'MaxDD':>7} {'EV_bp':>7} | {'CombPF':>7} {'Delta':>7} | {'Boot':>5} {'P≥1':>5}")
        print(f"  {'-'*6} {'-'*6} | {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} | {'-'*7} {'-'*7} | {'-'*5} {'-'*5}")

        best_worst_pf = 0.0
        for cost_label, cost_val in cost_levels:
            for slip_label, slip_val in slippage_levels:
                trades = collect_strategy_trades(
                    strat_name, strat_cfg, data_tf, data_15m, cost_val, slip_val,
                )
                if not trades:
                    print(f"  {cost_label:>6} {slip_label:>6} | NO TRADES")
                    continue

                pnl = portfolio_sim(trades, SLOT_LIMIT)
                m = compute_metrics(pnl, DATA_YEARS)
                boot = bootstrap_pf(pnl, 2000)

                # Combined portfolio
                combined_pnl = portfolio_sim(existing_trades + trades, SLOT_LIMIT)
                cm = compute_metrics(combined_pnl, DATA_YEARS)
                delta = cm["PF"] - existing_m["PF"]

                marker = ""
                if cost_label == "1x" and slip_label == "0%":
                    marker = " ← base"
                elif cost_label == "3x" and slip_label == "0.2%":
                    marker = " ← WORST"

                print(
                    f"  {cost_label:>6} {slip_label:>6} | "
                    f"{m['PF']:6.3f} {m['N']:6d} {m['WR']:5.1f}% {m['max_dd_pct']:6.1f}% {m['EV_bp']:7.1f} | "
                    f"{cm['PF']:7.3f} {delta:+7.3f} | "
                    f"[{boot.get('CI_lo', 0):.2f}] {boot.get('P_PF_ge_1', 0):4.0f}%{marker}"
                )

                if cost_label == "3x" and slip_label == "0.2%":
                    best_worst_pf = m["PF"]

        # Non-overlap (only need to compute once — signals don't change with cost)
        base_trades = collect_strategy_trades(
            strat_name, strat_cfg, data_tf, data_15m, 0.0008, 0.0,
        )
        non_overlap_raw = len(base_trades)
        existing_intervals: dict[str, list[tuple[int, int]]] = {}
        for sym, entry, exit_b, _ in existing_trades:
            existing_intervals.setdefault(sym, []).append((entry, exit_b))
        non_overlapping = 0
        for sym, entry, exit_b, _ in base_trades:
            intervals = existing_intervals.get(sym, [])
            overlaps = False
            for e_entry, e_exit in intervals:
                if entry <= e_exit and exit_b >= e_entry:
                    overlaps = True
                    break
            if not overlaps:
                non_overlapping += 1
        no_pct = non_overlapping / non_overlap_raw * 100 if non_overlap_raw > 0 else 0

        print(f"\n  Non-overlap with existing: {no_pct:.1f}% ({non_overlapping}/{non_overlap_raw})")
        print(f"  Worst-case PF (3x cost + 0.2% slip): {best_worst_pf:.3f}")
        verdict = "PASS" if best_worst_pf > 1.3 else ("MARGINAL" if best_worst_pf > 1.0 else "FAIL")
        print(f"  Verdict: {verdict}")

    # Final summary
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print("  Strategies that survive worst-case (3x cost + 0.2% slippage):")
    print("  with PF > 1.3 → PRODUCTION VIABLE")
    print("  with PF > 1.0 → MARGINAL (needs more investigation)")
    print("  with PF < 1.0 → FAIL (not robust)")
    print()
    print("---")
    print("primary_metric: 0")


if __name__ == "__main__":
    main()
