"""
QuantBox Trading Dashboard

Web-based UI for:
1. Running and visualizing backtests
2. Monitoring live trading
3. Analyzing performance metrics
4. Comparing strategies

Built with Streamlit for easy deployment.

Run: streamlit run quantbox/ui/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
import json
import time
import threading

# Page config
st.set_page_config(
    page_title="QuantBox Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    """Create equity curve chart with drawdown overlay"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown")
    )
    
    # Equity line
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00D4AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ),
        row=1, col=1
    )
    
    # Cash line
    if 'cash' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['cash'],
                mode='lines',
                name='Cash',
                line=dict(color='#FFB800', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    # Drawdown
    if 'drawdown' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['drawdown'] * 100,
                mode='lines',
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='#FF4444', width=1),
                fillcolor='rgba(255, 68, 68, 0.3)'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig


def create_trades_chart(trades_df: pd.DataFrame, equity_df: pd.DataFrame = None) -> go.Figure:
    """Create chart showing trade entries and exits"""
    
    fig = go.Figure()
    
    if equity_df is not None and not equity_df.empty:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            )
        )
    
    if not trades_df.empty:
        # Winning trades
        wins = trades_df[trades_df['net_pnl'] > 0]
        losses = trades_df[trades_df['net_pnl'] <= 0]
        
        if not wins.empty:
            fig.add_trace(
                go.Scatter(
                    x=wins['exit_time'],
                    y=wins['net_pnl'].cumsum() + (equity_df['equity'].iloc[0] if equity_df is not None else 10000),
                    mode='markers',
                    name='Winning Trades',
                    marker=dict(color='#00FF88', size=10, symbol='triangle-up'),
                    text=wins['pair'] + '<br>PnL: $' + wins['net_pnl'].round(2).astype(str),
                    hoverinfo='text+x'
                )
            )
        
        if not losses.empty:
            fig.add_trace(
                go.Scatter(
                    x=losses['exit_time'],
                    y=losses['net_pnl'].cumsum() + (equity_df['equity'].iloc[0] if equity_df is not None else 10000),
                    mode='markers',
                    name='Losing Trades',
                    marker=dict(color='#FF4444', size=10, symbol='triangle-down'),
                    text=losses['pair'] + '<br>PnL: $' + losses['net_pnl'].round(2).astype(str),
                    hoverinfo='text+x'
                )
            )
    
    fig.update_layout(
        title='Trade Distribution',
        height=400,
        template='plotly_dark',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_returns_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """Create histogram of trade returns"""
    
    fig = go.Figure()
    
    if not trades_df.empty and 'return_pct' in trades_df.columns:
        returns = trades_df['return_pct']
        
        # Separate wins and losses for coloring
        colors = ['#00FF88' if r > 0 else '#FF4444' for r in returns]
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name='Returns',
                marker_color='#00D4AA',
                opacity=0.7
            )
        )
        
        # Add mean line
        mean_return = returns.mean()
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )
    
    fig.update_layout(
        title='Trade Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Count',
        height=300,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_exit_reasons_pie(trades_df: pd.DataFrame) -> go.Figure:
    """Create pie chart of exit reasons"""
    
    fig = go.Figure()
    
    if not trades_df.empty and 'exit_reason' in trades_df.columns:
        reasons = trades_df['exit_reason'].value_counts()
        
        colors = {
            'mean_reversion': '#00FF88',
            'stop_loss': '#FF4444',
            'max_holding': '#FFB800',
            'end_of_period': '#888888'
        }
        
        fig.add_trace(
            go.Pie(
                labels=reasons.index,
                values=reasons.values,
                marker_colors=[colors.get(r, '#888888') for r in reasons.index],
                hole=0.4,
                textinfo='percent+label'
            )
        )
    
    fig.update_layout(
        title='Exit Reasons',
        height=300,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_monthly_returns_heatmap(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> go.Figure:
    """Create monthly returns heatmap"""
    
    fig = go.Figure()
    
    if not equity_df.empty:
        # Calculate daily returns
        equity_df = equity_df.copy()
        equity_df['daily_return'] = equity_df['equity'].pct_change() * 100
        
        # Group by month
        equity_df['year'] = equity_df.index.year
        equity_df['month'] = equity_df.index.month
        
        monthly = equity_df.groupby(['year', 'month'])['daily_return'].sum().reset_index()
        monthly_pivot = monthly.pivot(index='year', columns='month', values='daily_return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_pivot.values,
            x=month_names[:len(monthly_pivot.columns)],
            y=monthly_pivot.index.astype(str),
            colorscale=[
                [0, '#FF4444'],
                [0.5, '#333333'],
                [1, '#00FF88']
            ],
            zmid=0,
            text=np.round(monthly_pivot.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Monthly Returns (%)',
        height=250,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def display_metrics_cards(metrics: Dict):
    """Display key metrics as cards"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = metrics.get('total_return_pct', 0)
        color = "normal" if total_return >= 0 else "inverse"
        st.metric(
            label="Total Return",
            value=f"{total_return:.2f}%",
            delta=f"${metrics.get('total_return', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics.get('sharpe_ratio', 0):.2f}",
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics.get('max_drawdown', 0):.2f}%",
        )
    
    with col4:
        st.metric(
            label="Win Rate",
            value=f"{metrics.get('win_rate', 0):.1f}%",
        )
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="Total Trades",
            value=f"{metrics.get('n_trades', 0)}"
        )
    
    with col6:
        st.metric(
            label="Profit Factor",
            value=f"{metrics.get('profit_factor', 0):.2f}"
        )
    
    with col7:
        st.metric(
            label="Avg Trade",
            value=f"${metrics.get('avg_trade', 0):.2f}"
        )
    
    with col8:
        st.metric(
            label="Market Exposure",
            value=f"{metrics.get('in_market_pct', 0):.1f}%"
        )


def run_backtest_page():
    """Backtest configuration and execution page"""
    
    st.header("🎯 Backtest Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Capital & Costs")
        initial_capital = st.number_input("Initial Capital ($)", value=10000.0, step=1000.0)
        trading_fee = st.number_input("Trading Fee (%)", value=0.05, step=0.01, format="%.2f") / 100
        slippage = st.number_input("Slippage (%)", value=0.1, step=0.01, format="%.2f") / 100
        
        st.subheader("Formation Period")
        formation_bars = st.slider("Formation Bars", 50, 500, 240)
    
    with col2:
        st.subheader("Entry/Exit")
        entry_zscore = st.slider("Entry Z-score", 1.5, 3.5, 2.0, 0.1)
        exit_zscore = st.slider("Exit Z-score", 0.0, 1.5, 0.5, 0.1)
        stop_loss_zscore = st.slider("Stop Loss Z-score", 2.0, 5.0, 3.5, 0.1)
        max_holding = st.slider("Max Holding Bars", 20, 200, 100)
    
    st.subheader("Pair Selection")
    col3, col4 = st.columns(2)
    
    with col3:
        min_correlation = st.slider("Min Correlation", 0.3, 0.9, 0.6, 0.05)
        max_coint_pvalue = st.slider("Max Coint P-value", 0.01, 0.10, 0.05, 0.01)
    
    with col4:
        min_half_life = st.slider("Min Half-Life", 1, 20, 2)
        max_half_life = st.slider("Max Half-Life", 20, 100, 50)
    
    if st.button("🚀 Run Backtest", type="primary"):
        run_backtest_with_config({
            'initial_capital': initial_capital,
            'trading_fee': trading_fee,
            'slippage': slippage,
            'formation_bars': formation_bars,
            'entry_zscore': entry_zscore,
            'exit_zscore': exit_zscore,
            'stop_loss_zscore': stop_loss_zscore,
            'max_holding_bars': max_holding,
            'min_correlation': min_correlation,
            'max_cointegration_pvalue': max_coint_pvalue,
            'min_half_life': min_half_life,
            'max_half_life': max_half_life
        })


def run_backtest_with_config(config: Dict):
    """Execute backtest and display results"""
    
    with st.spinner("Running backtest..."):
        try:
            from quantbox.backtest.continuous_backtest import (
                ContinuousPairBacktest,
                ContinuousBacktestConfig
            )
            from quantbox.data.real_data_loader import RealDataLoader
            
            # Load REAL market data only (no synthetic data)
            loader = RealDataLoader()
            data = loader.load_data(min_bars=500, align_data=True)
            
            if len(data) < 2:
                st.error("Insufficient real market data. Please run the data downloader first.")
                return
            
            # Run backtest
            bt_config = ContinuousBacktestConfig(**config)
            engine = ContinuousPairBacktest(bt_config)
            results = engine.run(data, progress=False)
            
            # Store results in session state
            st.session_state['backtest_results'] = results
            
            st.success("Backtest completed!")
            
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def results_page():
    """Display backtest results"""
    
    st.header("📊 Backtest Results")
    
    if 'backtest_results' not in st.session_state:
        st.warning("No backtest results available. Run a backtest first!")
        return
    
    results = st.session_state['backtest_results']
    metrics = results.get('metrics', {})
    equity_df = results.get('equity_curve', pd.DataFrame())
    trades_df = results.get('trades', pd.DataFrame())
    
    # Metrics cards
    display_metrics_cards(metrics)
    
    st.divider()
    
    # Equity curve
    if not equity_df.empty:
        st.plotly_chart(create_equity_chart(equity_df), use_container_width=True)
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        if not trades_df.empty:
            st.plotly_chart(create_returns_distribution(trades_df), use_container_width=True)
    
    with col2:
        if not trades_df.empty:
            st.plotly_chart(create_exit_reasons_pie(trades_df), use_container_width=True)
    
    # Monthly heatmap
    if not equity_df.empty:
        st.plotly_chart(create_monthly_returns_heatmap(trades_df, equity_df), use_container_width=True)
    
    # Trades table
    st.subheader("📋 Trade Log")
    if not trades_df.empty:
        st.dataframe(
            trades_df.style.format({
                'capital': '${:,.2f}',
                'gross_pnl': '${:,.2f}',
                'net_pnl': '${:,.2f}',
                'fees': '${:,.2f}',
                'slippage': '${:,.2f}',
                'return_pct': '{:.2f}%',
                'entry_z': '{:.2f}',
                'exit_z': '{:.2f}',
                'hedge_ratio': '{:.4f}'
            }).applymap(
                lambda x: 'color: #00FF88' if isinstance(x, (int, float)) and x > 0 else 'color: #FF4444',
                subset=['net_pnl']
            ),
            use_container_width=True
        )
    
    # Detailed metrics
    with st.expander("📈 Detailed Metrics"):
        st.json(metrics)


def live_trading_page():
    """Live trading monitoring page"""
    
    st.header("🔴 Live Trading Monitor")
    
    st.warning("⚠️ Live trading involves real financial risk. Use paper trading mode first!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mode = st.radio(
            "Trading Mode",
            ["Paper Trading", "Live Trading"],
            horizontal=True
        )
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Configuration
    with st.expander("⚙️ Live Trading Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            capital = st.number_input("Trading Capital ($)", value=1000.0, step=100.0)
            timeframe = st.selectbox("Timeframe", ['15m', '1h', '4h'], index=1)
            n_coins = st.slider("Coins to Analyze", 10, 50, 30)
        
        with col2:
            check_interval = st.slider("Check Interval (seconds)", 30, 300, 60)
            entry_z = st.slider("Entry Z-score", 1.5, 3.0, 2.0, 0.1)
            exit_z = st.slider("Exit Z-score", 0.0, 1.0, 0.5, 0.1)
    
    # Status display
    st.subheader("📊 Current Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Mode", "PAPER" if mode == "Paper Trading" else "LIVE")
    
    with status_col2:
        st.metric("Status", "Idle")
    
    with status_col3:
        st.metric("Position", "None")
    
    # Start/Stop controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_btn = st.button("▶️ Start Trading", type="primary", disabled=True)
    
    with col2:
        stop_btn = st.button("⏹️ Stop Trading", disabled=True)
    
    st.info("🔧 Live trading controls are disabled in the dashboard for safety. Use the command line interface.")
    
    st.code("""
# Paper trading (recommended first):
python -m quantbox.trading.live_trader --paper --capital 1000

# Live trading (use with caution!):
python -m quantbox.trading.live_trader --live --capital 1000
    """, language="bash")
    
    # Recent activity log
    st.subheader("📜 Activity Log")
    st.text_area(
        "Log",
        "No recent activity. Start trading to see logs here.",
        height=200,
        disabled=True
    )


def validation_page():
    """Validation and testing page"""
    
    st.header("🔍 System Validation")
    
    st.markdown("""
    This page runs comprehensive validation tests to ensure:
    1. **No Lookahead Bias** - Only historical data is used for decisions
    2. **Correct Fee Calculation** - Transaction costs are accurately modeled
    3. **Consistent Logic** - Backtest and live trading use identical logic
    4. **Capital Tracking** - Money flows are correctly tracked
    """)
    
    if st.button("🧪 Run Validation Tests", type="primary"):
        with st.spinner("Running validation tests..."):
            try:
                from quantbox.backtest.validation import run_all_validation_tests
                
                # Capture output
                import io
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                
                passed, failed = run_all_validation_tests()
                
                output = buffer.getvalue()
                sys.stdout = old_stdout
                
                # Display results
                if failed == 0:
                    st.success(f"✅ All {passed} tests passed!")
                else:
                    st.error(f"❌ {failed} tests failed, {passed} passed")
                
                st.code(output)
                
            except Exception as e:
                st.error(f"Validation failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def main():
    """Main dashboard entry point"""
    
    # Sidebar navigation
    st.sidebar.title("📈 QuantBox")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Run Backtest", "📊 Results", "🔴 Live Trading", "🔍 Validation"],
        label_visibility="hidden"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **QuantBox** is a professional-grade 
    cryptocurrency pair trading framework.
    
    - Event-driven architecture
    - Lookahead bias prevention
    - Accurate cost modeling
    - Live/Paper trading
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("v1.0.0 | [GitHub](https://github.com)")
    
    # Page routing
    if page == "🎯 Run Backtest":
        run_backtest_page()
    elif page == "📊 Results":
        results_page()
    elif page == "🔴 Live Trading":
        live_trading_page()
    elif page == "🔍 Validation":
        validation_page()


if __name__ == "__main__":
    main()
