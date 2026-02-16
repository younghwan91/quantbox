"""
Strategy module — Rubber Band mean-reversion scalper + Alpha Engine.

Production strategy: RB_original + Volatility-ranked Top-5 symbols.
Walk-forward validated: +35% (90d), Sharpe 4.67, OOS +34% compounded.
"""

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals,
    calculate_position_size,
)
from quantbox.strategy.alpha_engine import (
    AlphaConfig,
    AlphaStrategy,
    compute_alpha_signals,
    compute_hybrid_signals,
)
from quantbox.strategy.vol_selector import (
    VolSelectorConfig,
    VolatilitySelector,
    compute_realized_vol,
    rank_symbols_by_vol,
)

__all__ = [
    # Original rubber band
    "RubberBandConfig",
    "RubberBandStrategy",
    "compute_signals",
    "calculate_position_size",
    # Alpha engine (enhanced + hybrid)
    "AlphaConfig",
    "AlphaStrategy",
    "compute_alpha_signals",
    "compute_hybrid_signals",
    # Volatility-based symbol selection
    "VolSelectorConfig",
    "VolatilitySelector",
    "compute_realized_vol",
    "rank_symbols_by_vol",
]
