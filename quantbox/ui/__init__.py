"""
Analysis UI module - Streamlit-based dashboard for backtesting and analysis.
"""

from quantbox.ui.dashboard import run_dashboard
from quantbox.ui.components import (
    plot_equity_curve,
    plot_drawdown,
    plot_trades,
    metrics_card,
)

__all__ = [
    "run_dashboard",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_trades",
    "metrics_card",
]
