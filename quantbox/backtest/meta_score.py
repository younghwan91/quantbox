"""MetaScore: backtest 결과의 실전 생존 확률 측정.

5-metric composite score (0-1):
1. Slippage Robustness — 비용 2x/3x에서도 PF>1?
2. Temporal Stability — 매년 profitable?
3. Parameter Plateau — ±20%에서도 안정?
4. OOS Decay — 최근 2년 성능 감소율?
5. Filter Effectiveness — 필터가 실제로 선별하는가? (multiplicative gate)

Usage:
    from quantbox.backtest.meta_score import compute_meta_score
    score = compute_meta_score(collect_fn, kwargs, "cluster_pct", 0.02, max_bars)
    print(score["composite"])  # 0.0 - 1.0
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from quantbox.backtest.trade_sim import compute_metrics, portfolio_sim

DEFAULT_SLOT_LIMIT = 10
DEFAULT_DATA_YEARS = 6.0


def score_slippage_robustness(
    collect_fn: Callable[..., list],
    collect_kwargs: dict[str, Any],
    slot_limit: int = DEFAULT_SLOT_LIMIT,
    data_years: float = DEFAULT_DATA_YEARS,
    base_cost: float = 0.0008,
) -> float:
    """Test PF at 1x, 2x, 3x cost. Score = fraction where PF > 1.0."""
    multipliers = [1, 2, 3]
    passes = 0
    for mult in multipliers:
        kwargs = {**collect_kwargs, "cost": base_cost * mult}
        trades = collect_fn(**kwargs)
        pnl = portfolio_sim(trades, max_concurrent=slot_limit)
        m = compute_metrics(pnl, data_years)
        if m.get("PF", 0) > 1.0:
            passes += 1
    return passes / len(multipliers)


def score_temporal_stability(
    trades: list[tuple[str, int, int, float]],
    max_bars: int,
    data_years: float = DEFAULT_DATA_YEARS,
    start_year: int = 2020,
) -> float:
    """Score = fraction of years with PF > 1.0."""
    if not trades:
        return 0.0
    bars_per_year = max_bars / data_years
    end_year = start_year + int(data_years) - 1
    yearly: dict[int, list[float]] = {}
    for _, entry, _, pnl_val in trades:
        year = min(int(start_year + entry / bars_per_year), end_year)
        yearly.setdefault(year, []).append(pnl_val)

    profitable_years = 0
    total_years = 0
    for pnls_list in yearly.values():
        pnls = np.array(pnls_list)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
        total_years += 1
        if pf > 1.0:
            profitable_years += 1
    return profitable_years / total_years if total_years > 0 else 0.0


def score_parameter_plateau(
    collect_fn: Callable[..., list],
    collect_kwargs: dict[str, Any],
    param_name: str,
    base_value: float,
    slot_limit: int = DEFAULT_SLOT_LIMIT,
    data_years: float = DEFAULT_DATA_YEARS,
) -> float:
    """Test PF at param +/-10%, +/-20%. Score = fraction within 80% of base PF."""
    trades = collect_fn(**collect_kwargs)
    pnl = portfolio_sim(trades, max_concurrent=slot_limit)
    base_m = compute_metrics(pnl, data_years)
    base_pf = base_m.get("PF", 0)
    if base_pf <= 0:
        return 0.0

    offsets = [-0.20, -0.10, 0.10, 0.20]
    stable = 0
    for off in offsets:
        test_val = base_value * (1 + off)
        test_kwargs = {**collect_kwargs, param_name: test_val}
        test_trades = collect_fn(**test_kwargs)
        test_pnl = portfolio_sim(test_trades, max_concurrent=slot_limit)
        m = compute_metrics(test_pnl, data_years)
        if m.get("PF", 0) >= base_pf * 0.80:
            stable += 1
    return stable / len(offsets)


def score_oos_decay(
    trades: list[tuple[str, int, int, float]],
    max_bars: int,
    data_years: float = DEFAULT_DATA_YEARS,
    is_split: float = 4.0,
) -> float:
    """Compare PF of recent years vs early years. Score = 1 - decay_rate."""
    if not trades:
        return 0.0
    bars_per_year = max_bars / data_years
    cutoff_bar = int(is_split * bars_per_year)

    early = [p for _, e, _, p in trades if e < cutoff_bar]
    late = [p for _, e, _, p in trades if e >= cutoff_bar]

    if not early or not late:
        return 0.0

    def pf(arr: np.ndarray) -> float:
        w = arr[arr > 0].sum()
        lo = abs(arr[arr <= 0].sum())
        return w / lo if lo > 0 else 999

    early_pf = pf(np.array(early))
    late_pf = pf(np.array(late))

    if early_pf <= 0:
        return 0.0

    decay_rate = max(0.0, 1 - late_pf / early_pf)
    return max(0.0, min(1.0, 1 - decay_rate))


def score_filter_effectiveness(
    collect_fn: Callable[..., list],
    collect_kwargs: dict[str, Any],
    data_years: float = DEFAULT_DATA_YEARS,
    max_tpsy: float = 30.0,
) -> float:
    """Measure filter selectivity via trades per symbol per year.

    Multiplicative gate: if filter is disabled (tpsy > 30), composite = 0.
    Score: <=10 -> 1.0, 10-30 -> linear decay, >30 -> 0.0.
    """
    raw_trades = collect_fn(**collect_kwargs)
    raw_n = len(raw_trades)
    if raw_n == 0:
        return 1.0

    n_symbols = len(set(s for s, _, _, _ in raw_trades))
    if n_symbols == 0:
        return 1.0

    tpsy = raw_n / n_symbols / data_years

    if tpsy <= 10:
        return 1.0
    elif tpsy <= max_tpsy:
        return 1.0 - (tpsy - 10) / (max_tpsy - 10)
    else:
        return 0.0


def compute_meta_score(
    collect_fn: Callable[..., list],
    collect_kwargs: dict[str, Any],
    param_name: str,
    base_param_value: float,
    max_bars: int,
    *,
    slot_limit: int = DEFAULT_SLOT_LIMIT,
    data_years: float = DEFAULT_DATA_YEARS,
) -> dict[str, float]:
    """Compute composite MetaScore.

    Returns dict with composite score (0-1) and individual sub-scores.
    Filter effectiveness is a multiplicative gate — if 0, composite = 0.
    """
    trades = collect_fn(**collect_kwargs)
    pnl = portfolio_sim(trades, max_concurrent=slot_limit)
    base_metrics = compute_metrics(pnl, data_years)

    s1 = score_slippage_robustness(collect_fn, collect_kwargs, slot_limit, data_years)
    s2 = score_temporal_stability(trades, max_bars, data_years)
    s3 = score_parameter_plateau(collect_fn, collect_kwargs, param_name, base_param_value, slot_limit, data_years)
    s4 = score_oos_decay(trades, max_bars, data_years)
    s5 = score_filter_effectiveness(collect_fn, collect_kwargs, data_years)

    # Filter is multiplicative gate — disabled filter zeroes everything
    base_score = (s1 + s2 + s3 + s4) / 4.0
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
