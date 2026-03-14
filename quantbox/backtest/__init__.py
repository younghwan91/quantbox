"""Backtesting module — two engines for different strategy families.

Engines:
  - VectorizedBacktest: Signal-array engine for strategies that embed
    exit logic in the signal (e.g. RubberBand).  Uses close-to-close PnL.
  - Trade-event engine (trade_sim): Discrete SL/TP simulation with
    intra-bar high/low resolution + cross-symbol slot contention.
    Canonical engine for VolStraddle (VR Squeeze) and any strategy
    with explicit stop-loss / take-profit.

Usage:
    # Signal-array (RubberBand)
    from quantbox.backtest import VectorizedBacktest, BacktestConfig

    # Trade-event (VolStraddle / any SL-TP strategy)
    from quantbox.backtest.trade_sim import (
        collect_trades, portfolio_sim, portfolio_sim_with_details,
        get_straddle_syms, compute_metrics,
        sim_directional_trades, sim_straddle_trades,
    )
"""

from quantbox.backtest.vectorized_backtest import (
    BacktestConfig,
    BacktestResult,
    VectorizedBacktest,
)

# Trade-event engine — import submodule for lazy access
from quantbox.backtest import trade_sim  # noqa: F401

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "VectorizedBacktest",
    "trade_sim",
]
