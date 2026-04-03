#!/usr/bin/env python3
"""MetaScore: backtest 결과의 실전 생존 확률 측정.

4-metric composite score (0-1):
1. Slippage Robustness — 비용 2x/3x에서도 PF>1?
2. Temporal Stability — 매년 profitable?
3. Parameter Plateau — ±20%에서도 안정?
4. OOS Decay — 최근 2년 성능 감소율?

Validation: production config >= 0.7, pct=0.15 failure case < 0.5.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (  # noqa: E402
    build_resampled,
    collect_ma_cluster_trades,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    precompute_all_vr,
    G_4TF,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import compute_momentum  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
#  FIXED: Data loading
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = ROOT / "data" / "cache_extended"
DATA_YEARS = 6.0
SLOT_LIMIT = 10


def load_data():
    cfg = VolStraddleConfig()
    whitelist = set(cfg.symbol_whitelist)
    data_15m, max_bars = {}, 0
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
        data_15m[sym]["mom_8"] = compute_momentum(data_15m[sym]["close"], cfg.momentum_period)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    for d in data_1h.values():
        d["mom_8"] = compute_momentum(d["close"], cfg.momentum_period)
    for d in data_4h.values():
        d["mom_8"] = compute_momentum(d["close"], cfg.momentum_period)
    precompute_all_vr([data_15m, data_1h, data_4h], [(cfg.vr_short, cfg.vr_long)])
    return data_15m, data_1h, data_4h, max_bars


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT: MetaScore implementation — agent hacks this section
# ══════════════════════════════════════════════════════════════════════════════

# Weights for composite score
W_SLIPPAGE = 0.20
W_TEMPORAL = 0.20
W_PLATEAU = 0.20
W_DECAY = 0.20
W_FILTER = 0.20  # filter effectiveness — penalizes disabled filters


def score_slippage_robustness(
    collect_fn, collect_kwargs: dict, base_cost: float = 0.0008
) -> float:
    """Test PF at 1x, 2x, 3x cost. Score = fraction of multipliers where PF > 1.0."""
    multipliers = [1, 2, 3]
    passes = 0
    for mult in multipliers:
        kwargs = {**collect_kwargs, "cost": base_cost * mult}
        trades = collect_fn(**kwargs)
        pnl = portfolio_sim(trades, max_concurrent=SLOT_LIMIT)
        m = compute_metrics(pnl, DATA_YEARS)
        if m.get("PF", 0) > 1.0:
            passes += 1
    return passes / len(multipliers)


def score_temporal_stability(trades: list, max_bars: int) -> float:
    """Score = fraction of years with PF > 1.0."""
    if not trades:
        return 0.0
    bars_per_year = max_bars / DATA_YEARS
    yearly: dict[int, list[float]] = {}
    for sym, entry, exit_b, pnl_val in trades:
        year = min(int(2020 + entry / bars_per_year), 2025)
        yearly.setdefault(year, []).append(pnl_val)

    profitable_years = 0
    total_years = 0
    for y in sorted(yearly):
        pnls = np.array(yearly[y])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
        total_years += 1
        if pf > 1.0:
            profitable_years += 1
    return profitable_years / total_years if total_years > 0 else 0.0


def score_parameter_plateau(
    collect_fn, collect_kwargs: dict, param_name: str, base_value: float
) -> float:
    """Test PF at param ±10%, ±20%. Score = fraction of variants with PF within 80% of base PF."""
    # First get base PF
    trades = collect_fn(**collect_kwargs)
    pnl = portfolio_sim(trades, max_concurrent=SLOT_LIMIT)
    base_m = compute_metrics(pnl, DATA_YEARS)
    base_pf = base_m.get("PF", 0)
    if base_pf <= 0:
        return 0.0

    offsets = [-0.20, -0.10, 0.10, 0.20]
    stable = 0
    for off in offsets:
        test_val = base_value * (1 + off)
        test_kwargs = {**collect_kwargs, param_name: test_val}
        trades = collect_fn(**test_kwargs)
        pnl = portfolio_sim(trades, max_concurrent=SLOT_LIMIT)
        m = compute_metrics(pnl, DATA_YEARS)
        test_pf = m.get("PF", 0)
        if test_pf >= base_pf * 0.80:
            stable += 1
    return stable / len(offsets)


def score_oos_decay(trades: list, max_bars: int) -> float:
    """Compare PF of recent 2 years vs first 4 years. Score = 1 - decay_rate (clamped 0-1)."""
    if not trades:
        return 0.0
    bars_per_year = max_bars / DATA_YEARS
    cutoff_bar = int(4.0 * bars_per_year)  # first 4 years

    early = [p for _, e, _, p in trades if e < cutoff_bar]
    late = [p for _, e, _, p in trades if e >= cutoff_bar]

    if not early or not late:
        return 0.0

    early_arr = np.array(early)
    late_arr = np.array(late)

    def pf(arr):
        w = arr[arr > 0].sum()
        l = abs(arr[arr <= 0].sum())
        return w / l if l > 0 else 999

    early_pf = pf(early_arr)
    late_pf = pf(late_arr)

    if early_pf <= 0:
        return 0.0

    decay_rate = max(0, 1 - late_pf / early_pf)  # 0 = no decay, 1 = total decay
    return max(0, min(1, 1 - decay_rate))


def score_filter_effectiveness(
    collect_fn, collect_kwargs: dict, param_name: str, base_param_value: float,
    max_bars: int,
) -> float:
    """Measure filter selectivity via trades per symbol per year.

    A genuine squeeze/cluster filter should fire rarely per symbol.
    If trades_per_sym_per_year > 30, the filter is too loose.

    Score: <=10 → 1.0, 10-30 → linear decay, >30 → 0.0.
    """
    raw_trades = collect_fn(**collect_kwargs)
    raw_n = len(raw_trades)
    if raw_n == 0:
        return 1.0

    n_symbols = len(set(s for s, _, _, _ in raw_trades))
    if n_symbols == 0:
        return 1.0

    tpsy = raw_n / n_symbols / DATA_YEARS  # trades per symbol per year

    if tpsy <= 10:
        return 1.0
    elif tpsy <= 30:
        return 1.0 - (tpsy - 10) / 20.0
    else:
        return 0.0


def compute_meta_score(
    collect_fn,
    collect_kwargs: dict,
    param_name: str,
    base_param_value: float,
    max_bars: int,
) -> dict:
    """Compute composite MetaScore."""
    trades = collect_fn(**collect_kwargs)
    pnl = portfolio_sim(trades, max_concurrent=SLOT_LIMIT)
    base_metrics = compute_metrics(pnl, DATA_YEARS)

    s1 = score_slippage_robustness(collect_fn, collect_kwargs)
    s2 = score_temporal_stability(trades, max_bars)
    s3 = score_parameter_plateau(collect_fn, collect_kwargs, param_name, base_param_value)
    s4 = score_oos_decay(trades, max_bars)
    s5 = score_filter_effectiveness(
        collect_fn, collect_kwargs, param_name, base_param_value, max_bars
    )

    # Filter is a multiplicative gate — if filter is broken, nothing else matters
    base_score = (
        W_SLIPPAGE * s1 + W_TEMPORAL * s2 + W_PLATEAU * s3 + W_DECAY * s4
    ) / (W_SLIPPAGE + W_TEMPORAL + W_PLATEAU + W_DECAY)
    composite = base_score * s5

    return {
        "composite": composite,
        "slippage_robustness": s1,
        "temporal_stability": s2,
        "parameter_plateau": s3,
        "oos_decay": s4,
        "filter_effectiveness": s5,
        "base_pf": base_metrics.get("PF", 0),
        "base_n": base_metrics.get("N", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Test harness
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    data_15m, data_1h, data_4h, max_bars = load_data()

    # ── Test Case 1: Production MA Cluster config (should score >= 0.7) ──
    print("\n" + "=" * 60)
    print("  TEST 1: Production MA Cluster (pct=0.02, dur=6)")
    print("=" * 60)
    prod_kwargs = {
        "data_15m": data_15m, "data_4h": data_4h,
        "cluster_pct": 0.02, "min_duration": 6,
        "trail_activate": 0.01, "trail_distance": 0.005, "sl": 0.02,
    }
    prod_score = compute_meta_score(
        collect_ma_cluster_trades, prod_kwargs,
        param_name="cluster_pct", base_param_value=0.02, max_bars=max_bars,
    )
    for k, v in prod_score.items():
        print(f"  {k:25s}: {v:.3f}")
    prod_pass = "PASS" if prod_score["composite"] >= 0.7 else "FAIL"
    print(f"  VERDICT: {prod_pass} (target >= 0.7)")

    # ── Test Case 2: Failed pct=0.15 config (should score < 0.5) ──
    print("\n" + "=" * 60)
    print("  TEST 2: Failed config (pct=0.15, dur=1)")
    print("=" * 60)
    bad_kwargs = {
        "data_15m": data_15m, "data_4h": data_4h,
        "cluster_pct": 0.15, "min_duration": 1,
        "trail_activate": 0.01, "trail_distance": 0.005, "sl": 0.02,
    }
    bad_score = compute_meta_score(
        collect_ma_cluster_trades, bad_kwargs,
        param_name="cluster_pct", base_param_value=0.15, max_bars=max_bars,
    )
    for k, v in bad_score.items():
        print(f"  {k:25s}: {v:.3f}")
    bad_pass = "PASS" if bad_score["composite"] < 0.5 else "FAIL"
    print(f"  VERDICT: {bad_pass} (target < 0.5)")

    # ── Test Case 3: Production D_mom8 (should score >= 0.7) ──
    print("\n" + "=" * 60)
    print("  TEST 3: Production D_mom8 (d_vr=0.50)")
    print("=" * 60)
    cfg = VolStraddleConfig()
    d_cfg = {
        "sp": cfg.vr_short, "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold, "s_vr": cfg.s_vr_threshold,
        "s_dur": cfg.s_squeeze_dur, "d_sl": cfg.d_sl, "s_sl": cfg.s_sl,
        "tp": cfg.d_tp,
        "d_trail_activate": cfg.d_trail_activate,
        "d_trail_distance": cfg.d_trail_distance,
        "s_trail_activate": cfg.s_trail_activate,
        "s_trail_distance": cfg.s_trail_distance,
        "funding_rate_8h": cfg.default_funding_rate_8h,
    }

    def collect_d_mom8(**kw):
        c = {**d_cfg, **{k: v for k, v in kw.items() if k not in ("data_15m", "data_1h", "data_4h")}}
        return collect_trades(
            c, kw.get("data_15m", data_15m),
            kw.get("data_1h", data_1h), kw.get("data_4h", data_4h),
            strat="D_mom8", tf_group=G_4TF,
        )

    d_kwargs = {"data_15m": data_15m, "data_1h": data_1h, "data_4h": data_4h}
    d_score = compute_meta_score(
        collect_d_mom8, d_kwargs,
        param_name="d_vr", base_param_value=cfg.d_vr_threshold, max_bars=max_bars,
    )
    for k, v in d_score.items():
        print(f"  {k:25s}: {v:.3f}")
    d_pass = "PASS" if d_score["composite"] >= 0.7 else "FAIL"
    print(f"  VERDICT: {d_pass} (target >= 0.7)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"---")
    print(f"primary_metric: {prod_score['composite']:.3f}")
    print(f"prod_ma:        {prod_score['composite']:.3f} [{prod_pass}]")
    print(f"bad_ma:         {bad_score['composite']:.3f} [{bad_pass}]")
    print(f"prod_d:         {d_score['composite']:.3f} [{d_pass}]")
    all_pass = prod_pass == "PASS" and bad_pass == "PASS" and d_pass == "PASS"
    print(f"all_tests:      {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
