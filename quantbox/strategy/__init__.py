"""
Strategy module — Production strategy: vol_straddle (VR Squeeze).

Infrastructure:
  - RubberBandScalper: Base execution class (all strategies run on this)
  - RubberBand: Mean-reversion strategy (inactive, infrastructure only)
  - Alpha/VolSelector: Backtest utilities
"""

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals,
    compute_signals_mtf,
    compute_signals_with_btc_directional,
    compute_signals_regime_switch,
    compute_signals_confidence,
    compute_signals_4h_independent,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    calculate_position_size,
    compute_confidence_leverage,
    apply_confidence_leverage,
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
    # Rubber band (infrastructure base)
    "RubberBandConfig",
    "RubberBandStrategy",
    "compute_signals",
    "compute_signals_with_btc_directional",
    "compute_signals_regime_switch",
    "compute_signals_confidence",
    "calculate_position_size",
    "compute_confidence_leverage",
    "apply_confidence_leverage",
    # Alpha engine (backtest utilities)
    "AlphaConfig",
    "AlphaStrategy",
    "compute_alpha_signals",
    "compute_hybrid_signals",
    # Volatility-based symbol selection (backtest utilities)
    "VolSelectorConfig",
    "VolatilitySelector",
    "compute_realized_vol",
    "rank_symbols_by_vol",
]
