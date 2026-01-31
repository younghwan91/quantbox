"""
Utility modules for QuantBox trading framework.
"""

from quantbox.utils.numba_ops import (
    calculate_rolling_zscore,
    calculate_ema,
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_max_drawdown,
    correlation_matrix,
    ols_hedge_ratio,
    rolling_correlation,
    calculate_sharpe_ratio,
    optimal_position_sizing,
    kelly_criterion,
)

from quantbox.utils.costs import (
    TransactionCosts,
    SlippageModel,
    create_binance_futures_costs,
    create_conservative_costs,
)

__all__ = [
    # Numba-accelerated calculations
    'calculate_rolling_zscore',
    'calculate_ema',
    'calculate_half_life',
    'calculate_hurst_exponent',
    'calculate_max_drawdown',
    'correlation_matrix',
    'ols_hedge_ratio',
    'rolling_correlation',
    'calculate_sharpe_ratio',
    'optimal_position_sizing',
    'kelly_criterion',
    # Cost models
    'TransactionCosts',
    'SlippageModel',
    'create_binance_futures_costs',
    'create_conservative_costs',
]
