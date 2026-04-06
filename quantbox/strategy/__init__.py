"""
Strategy module — Production strategies: VR Directional, MA Cluster, Donchian.

Infrastructure:
  - TradingStrategy: Protocol formalising the shared strategy interface.
  - VolSelector: Backtest utility (compute_realized_vol)

DEPRECATED (Phase 2.5, refactor/mega-v2):
  - RubberBand strategy is deprecated. Module files kept for reference.
    Do not import from quantbox.strategy.rubberband for new code.
"""

from quantbox.strategy.protocol import TradingStrategy
from quantbox.strategy.vol_selector import (
    VolSelectorConfig,
    VolatilitySelector,
    compute_realized_vol,
    rank_symbols_by_vol,
)

__all__ = [
    # Strategy protocol (type-checking interface)
    "TradingStrategy",
    # Volatility-based symbol selection (backtest utilities)
    "VolSelectorConfig",
    "VolatilitySelector",
    "compute_realized_vol",
    "rank_symbols_by_vol",
]
