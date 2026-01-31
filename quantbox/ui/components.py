"""
UI components for visualization.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from quantbox.backtest.metrics import PerformanceMetrics


def check_plotly():
    """Check if Plotly is available."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for UI components. Install with: pip install plotly")


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve",
    height: int = 500
) -> "go.Figure":
    """
    Create interactive equity curve plot.
    
    Args:
        equity_curve: DataFrame with 'equity' column
        benchmark: Optional benchmark series
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly figure
    """
    check_plotly()
    
    fig = go.Figure()
    
    # Main equity curve
    if 'equity' in equity_curve.columns:
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#2E86AB', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
    
    # Benchmark
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1.5, dash='dash'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=height,
        template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        hovermode='x unified'
    )
    
    return fig


def plot_drawdown(
    equity_curve: pd.DataFrame,
    title: str = "Drawdown",
    height: int = 300
) -> "go.Figure":
    """
    Create drawdown visualization.
    """
    check_plotly()
    
    if 'drawdown' in equity_curve.columns:
        drawdown = equity_curve['drawdown'] * 100
    else:
        equity = equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#E74C3C', width=1),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=height,
        template='plotly_white',
        yaxis=dict(autorange='reversed')  # Drawdown goes down
    )
    
    return fig


def plot_trades(
    trades: pd.DataFrame,
    title: str = "Trade Analysis",
    height: int = 400
) -> "go.Figure":
    """
    Create trade analysis visualization.
    """
    check_plotly()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative P&L', 'Trade Returns Distribution',
            'Win/Loss by Exit Reason', 'P&L vs Holding Period'
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    if 'pnl' in trades.columns and len(trades) > 0:
        # Cumulative PnL
        cum_pnl = trades['pnl'].cumsum()
        colors = ['#27AE60' if p >= 0 else '#E74C3C' for p in cum_pnl]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cum_pnl))),
                y=cum_pnl,
                mode='lines+markers',
                name='Cumulative PnL',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Trade returns distribution
        fig.add_trace(
            go.Histogram(
                x=trades['pnl'],
                name='PnL Distribution',
                marker_color='#3498DB',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Win/Loss by exit reason
        if 'exit_reason' in trades.columns:
            reason_pnl = trades.groupby('exit_reason')['pnl'].sum()
            colors = ['#27AE60' if v > 0 else '#E74C3C' for v in reason_pnl.values]
            
            fig.add_trace(
                go.Bar(
                    x=reason_pnl.index,
                    y=reason_pnl.values,
                    name='PnL by Reason',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # PnL vs Holding period
        if 'bars_held' in trades.columns:
            colors = ['#27AE60' if p > 0 else '#E74C3C' for p in trades['pnl']]
            
            fig.add_trace(
                go.Scatter(
                    x=trades['bars_held'],
                    y=trades['pnl'],
                    mode='markers',
                    name='Trades',
                    marker=dict(color=colors, size=8, opacity=0.6)
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_monthly_heatmap(
    monthly_returns: pd.DataFrame,
    title: str = "Monthly Returns (%)",
    height: int = 400
) -> "go.Figure":
    """
    Create monthly returns heatmap.
    """
    check_plotly()
    
    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns.values,
        x=monthly_returns.columns,
        y=monthly_returns.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(monthly_returns.values, 1),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
        template='plotly_white'
    )
    
    return fig


def metrics_card(metrics: PerformanceMetrics) -> Dict[str, Any]:
    """
    Convert metrics to card data for display.
    """
    return {
        'return': {
            'label': 'Total Return',
            'value': f"{metrics.total_return_pct:.2f}%",
            'delta': metrics.total_return_pct,
            'positive': metrics.total_return_pct > 0
        },
        'sharpe': {
            'label': 'Sharpe Ratio',
            'value': f"{metrics.sharpe_ratio:.2f}",
            'delta': metrics.sharpe_ratio,
            'positive': metrics.sharpe_ratio > 1.0
        },
        'max_dd': {
            'label': 'Max Drawdown',
            'value': f"{metrics.max_drawdown:.2f}%",
            'delta': -metrics.max_drawdown,
            'positive': metrics.max_drawdown < 20
        },
        'win_rate': {
            'label': 'Win Rate',
            'value': f"{metrics.win_rate:.1f}%",
            'delta': metrics.win_rate - 50,
            'positive': metrics.win_rate > 50
        },
        'trades': {
            'label': 'Total Trades',
            'value': str(metrics.total_trades),
            'delta': None,
            'positive': True
        },
        'profit_factor': {
            'label': 'Profit Factor',
            'value': f"{metrics.profit_factor:.2f}",
            'delta': metrics.profit_factor - 1,
            'positive': metrics.profit_factor > 1
        }
    }


def plot_optimization_surface(
    results: pd.DataFrame,
    param1: str,
    param2: str,
    score_col: str = 'score',
    title: str = "Optimization Surface",
    height: int = 500
) -> "go.Figure":
    """
    Create 3D optimization surface plot.
    """
    check_plotly()
    
    # Pivot data for surface
    pivot = results.pivot_table(
        values=score_col,
        index=param1,
        columns=param2,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=[go.Surface(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        hovertemplate=f'{param1}: %{{y}}<br>{param2}: %{{x}}<br>Score: %{{z:.3f}}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis_title=param2,
            yaxis_title=param1,
            zaxis_title='Score'
        ),
        height=height
    )
    
    return fig


def plot_convergence(
    history: list,
    title: str = "Optimization Convergence",
    height: int = 300
) -> "go.Figure":
    """
    Plot optimization convergence history.
    """
    check_plotly()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=history,
        mode='lines+markers',
        name='Best Score',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Iteration',
        yaxis_title='Best Score',
        height=height,
        template='plotly_white'
    )
    
    return fig


def plot_pair_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    title: str = "Pair Correlation Matrix",
    height: int = 600
) -> "go.Figure":
    """
    Create correlation matrix heatmap.
    """
    check_plotly()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=8),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig
