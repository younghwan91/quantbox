"""Rubber Band Scalper Strategy subpackage.

Re-exports all public symbols for backward compatibility.
Import from `quantbox.strategy.rubberband` as before.
"""

from quantbox.strategy.rubberband.config import RubberBandConfig
from quantbox.strategy.rubberband.signals import (
    _compute_ema_inline,
    _compute_macd_inline,
    _compute_rsi_inline,
    _compute_sma_inline,
    _compute_stochastic_inline,
    apply_confidence_leverage,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    apply_rr_gated_leverage,
    calculate_position_size,
    compute_atr_pct,
    compute_confidence_leverage,
    compute_rr_gated_leverage,
    compute_signals,
    compute_signals_4h_independent,
    compute_signals_confidence,
    compute_signals_mtf,
    compute_signals_partial_exit_btc_directional,
    compute_signals_regime_switch,
    compute_signals_v9,
    compute_signals_with_btc_directional,
    compute_signals_with_partial_exit,
    compute_signals_with_regime,
)
from quantbox.strategy.rubberband.strategy import RubberBandStrategy

__all__ = [
    "RubberBandConfig",
    "RubberBandStrategy",
    "calculate_position_size",
    "compute_atr_pct",
    "compute_confidence_leverage",
    "compute_rr_gated_leverage",
    "compute_signals",
    "compute_signals_confidence",
    "compute_signals_v9",
    "compute_signals_with_regime",
]
