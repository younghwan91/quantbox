"""
Streamlit dashboard for backtesting and analysis.
"""

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger('Dashboard')


def run_dashboard(
    results: Dict = None,
    data: Dict[str, pd.DataFrame] = None,
    port: int = 8501
):
    """
    Run Streamlit dashboard.
    
    Args:
        results: Pre-loaded backtest results
        data: Pre-loaded market data
        port: Port to run on
    """
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return
    
    # Create dashboard app
    create_dashboard_app(results, data)


def create_dashboard_app(
    results: Dict = None,
    data: Dict[str, pd.DataFrame] = None
):
    """Create Streamlit dashboard application."""
    
    try:
        import streamlit as st
        from quantbox.ui.components import (
            plot_equity_curve, plot_drawdown, plot_trades,
            metrics_card, plot_monthly_heatmap
        )
        from quantbox.backtest.runner import BacktestRunner, BacktestConfig
        from quantbox.backtest.metrics import calculate_monthly_returns
    except ImportError as e:
        print(f"Required dependencies not installed: {e}")
        return
    
    # Page config
    st.set_page_config(
        page_title="QuantBox Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Header
    st.title("📈 QuantBox Trading Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.selectbox(
            "Mode",
            ["Backtest", "Analysis", "Optimization"]
        )
        
        st.divider()
        
        if mode == "Backtest":
            st.subheader("Backtest Parameters")
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000
            )
            
            trading_fee = st.slider(
                "Trading Fee (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01
            ) / 100
            
            entry_zscore = st.slider(
                "Entry Z-Score",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.1
            )
            
            exit_zscore = st.slider(
                "Exit Z-Score",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
            
            st.divider()
            
            run_backtest = st.button("🚀 Run Backtest", type="primary")
    
    # Main content
    if mode == "Backtest":
        _render_backtest_tab(results)
    elif mode == "Analysis":
        _render_analysis_tab(results, data)
    elif mode == "Optimization":
        _render_optimization_tab(data)


def _render_backtest_tab(results: Dict = None):
    """Render backtest results tab."""
    try:
        import streamlit as st
        from quantbox.ui.components import (
            plot_equity_curve, plot_drawdown, plot_trades, metrics_card
        )
    except ImportError:
        return
    
    if results is None:
        st.info("📊 Run a backtest to see results here, or load existing results.")
        
        # Upload option
        uploaded_file = st.file_uploader("Upload Results (pickle)", type=['pkl'])
        if uploaded_file:
            import pickle
            results = pickle.load(uploaded_file)
        else:
            return
    
    metrics = results.get('metrics')
    equity_curve = results.get('equity_curve', pd.DataFrame())
    trades = results.get('trades', pd.DataFrame())
    
    # Metrics cards
    st.subheader("📊 Performance Metrics")
    
    if metrics:
        cards = metrics_card(metrics)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                cards['return']['label'],
                cards['return']['value'],
                delta=f"{cards['return']['delta']:.1f}%"
            )
        
        with col2:
            st.metric(
                cards['sharpe']['label'],
                cards['sharpe']['value']
            )
        
        with col3:
            st.metric(
                cards['max_dd']['label'],
                cards['max_dd']['value'],
                delta=f"{cards['max_dd']['delta']:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                cards['win_rate']['label'],
                cards['win_rate']['value']
            )
        
        with col5:
            st.metric(
                cards['trades']['label'],
                cards['trades']['value']
            )
        
        with col6:
            st.metric(
                cards['profit_factor']['label'],
                cards['profit_factor']['value']
            )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💰 Equity Curve")
        if not equity_curve.empty:
            fig = plot_equity_curve(equity_curve)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No equity curve data available")
    
    with col2:
        st.subheader("📉 Drawdown")
        if not equity_curve.empty:
            fig = plot_drawdown(equity_curve)
            st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    st.subheader("🔄 Trade Analysis")
    if not trades.empty:
        fig = plot_trades(trades)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade table
        with st.expander("📋 Trade Details"):
            st.dataframe(trades, use_container_width=True)
    else:
        st.warning("No trade data available")


def _render_analysis_tab(results: Dict = None, data: Dict = None):
    """Render analysis tab."""
    try:
        import streamlit as st
        from quantbox.ui.components import plot_pair_correlation_matrix
    except ImportError:
        return
    
    st.subheader("📈 Market Analysis")
    
    if data is None:
        st.info("Load market data to perform analysis")
        return
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    # Build correlation matrix
    closes = pd.DataFrame({
        symbol: df['Close'] 
        for symbol, df in data.items()
        if 'Close' in df.columns
    })
    
    if not closes.empty:
        corr_matrix = closes.pct_change().corr()
        fig = plot_pair_correlation_matrix(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pair performance
    if results and 'pair_performance' in results:
        st.subheader("👥 Pair Performance")
        pair_perf = results['pair_performance']
        st.dataframe(pair_perf, use_container_width=True)


def _render_optimization_tab(data: Dict = None):
    """Render optimization tab."""
    try:
        import streamlit as st
        from quantbox.ui.components import plot_convergence, plot_optimization_surface
    except ImportError:
        return
    
    st.subheader("🎯 Parameter Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Optimization Settings**")
        
        method = st.selectbox(
            "Method",
            ["Differential Evolution", "Grid Search", "Random Search", "Bayesian"]
        )
        
        objective = st.selectbox(
            "Objective",
            ["Sharpe Ratio", "Calmar Ratio", "Total Return", "Profit Factor"]
        )
        
        st.markdown("**Parameter Bounds**")
        
        entry_z_min = st.number_input("Entry Z-Score Min", value=1.0)
        entry_z_max = st.number_input("Entry Z-Score Max", value=3.0)
        
        exit_z_min = st.number_input("Exit Z-Score Min", value=0.1)
        exit_z_max = st.number_input("Exit Z-Score Max", value=1.5)
        
        if st.button("🚀 Run Optimization", type="primary"):
            st.info("Optimization started... This may take a while.")
    
    with col2:
        st.markdown("**Results**")
        st.info("Run optimization to see results here")


# CLI entry point
def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantBox Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on")
    parser.add_argument("--results", type=str, help="Path to results file")
    
    args = parser.parse_args()
    
    results = None
    if args.results:
        import pickle
        with open(args.results, 'rb') as f:
            results = pickle.load(f)
    
    # Run with streamlit
    import subprocess
    import sys
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.port", str(args.port)
    ])


if __name__ == "__main__":
    # When run directly by streamlit
    create_dashboard_app()
