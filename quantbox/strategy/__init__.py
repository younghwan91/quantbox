"""
Strategy module — Rubber Band + Elliott Wave + Alpha Engine.

Strategies:
  - RubberBand: Mean-reversion scalper (15m/4h)
  - Elliott: ABC wave pattern + Fibonacci validation (4h/15m)
  - Alpha: Multi-factor alpha engine (hybrid)
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
from quantbox.strategy.elliott import (
    ElliottConfig,
    ElliottStrategy,
    compute_elliott_signals,
    scan_multi_scale_patterns,
)

__all__ = [
    # Original rubber band
    "RubberBandConfig",
    "RubberBandStrategy",
    "compute_signals",
    "compute_signals_with_btc_directional",
    "compute_signals_regime_switch",
    "compute_signals_confidence",
    "calculate_position_size",
    "compute_confidence_leverage",
    "apply_confidence_leverage",
    # Elliott Wave (ABC pattern + Fibonacci)
    "ElliottConfig",
    "ElliottStrategy",
    "compute_elliott_signals",
    "scan_multi_scale_patterns",
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
