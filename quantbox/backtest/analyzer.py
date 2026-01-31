"""
Backtest analyzer with visualization capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


logger = logging.getLogger('BacktestAnalyzer')


class BacktestAnalyzer:
    """
    Analyzer for backtest results with visualization.
    """
    
    def __init__(self, results: Dict):
        self.results = results
        self.metrics = results.get('metrics')
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('trades', pd.DataFrame())
        
        # Set style
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-darkgrid')
    
    def summary(self) -> str:
        """Get text summary of results."""
        if not self.metrics:
            return "No metrics available"
        
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║  Total Return:        {self.metrics.total_return_pct:>10.2f}%                        ║
║  Annualized Return:   {self.metrics.annualized_return:>10.2f}%                        ║
║  Max Drawdown:        {self.metrics.max_drawdown:>10.2f}%                        ║
║  Sharpe Ratio:        {self.metrics.sharpe_ratio:>10.2f}                          ║
║  Sortino Ratio:       {self.metrics.sortino_ratio:>10.2f}                          ║
║  Calmar Ratio:        {self.metrics.calmar_ratio:>10.2f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  Total Trades:        {self.metrics.total_trades:>10}                          ║
║  Winning Trades:      {self.metrics.winning_trades:>10}                          ║
║  Win Rate:            {self.metrics.win_rate:>10.1f}%                        ║
║  Profit Factor:       {self.metrics.profit_factor:>10.2f}                          ║
║  Average Trade:       ${self.metrics.avg_trade:>9.2f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  Final Capital:       ${self.metrics.final_capital:>9.2f}                          ║
║  Peak Capital:        ${self.metrics.peak_capital:>9.2f}                          ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    def plot_equity_curve(
        self,
        benchmark: Optional[pd.Series] = None,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ):
        """Plot equity curve."""
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if self.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.equity_curve.index, self.equity_curve['equity'], 
                label='Strategy', linewidth=2, color='#2E86AB')
        
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values,
                   label='Benchmark', linewidth=1.5, alpha=0.7, color='gray')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown(
        self,
        figsize: tuple = (12, 4),
        save_path: Optional[str] = None
    ):
        """Plot drawdown chart."""
        if not HAS_MATPLOTLIB or self.equity_curve.empty:
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'drawdown' in self.equity_curve.columns:
            drawdown = self.equity_curve['drawdown'] * 100
        else:
            equity = self.equity_curve['equity']
            peak = equity.expanding().max()
            drawdown = (peak - equity) / peak * 100
        
        ax.fill_between(drawdown.index, 0, drawdown, 
                       color='#E74C3C', alpha=0.5)
        ax.plot(drawdown.index, drawdown, color='#E74C3C', linewidth=1)
        
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_ylim(bottom=0)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_returns_distribution(
        self,
        figsize: tuple = (10, 5),
        save_path: Optional[str] = None
    ):
        """Plot returns distribution."""
        if not HAS_MATPLOTLIB or self.trades.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Returns histogram
        if 'pnl_pct' in self.trades.columns:
            returns = self.trades['pnl_pct']
        elif 'pnl' in self.trades.columns:
            returns = self.trades['pnl']
        else:
            return
        
        axes[0].hist(returns, bins=30, color='#3498DB', alpha=0.7, edgecolor='white')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].axvline(x=returns.mean(), color='green', linestyle='--', linewidth=2,
                       label=f'Mean: {returns.mean():.2f}')
        axes[0].set_title('Trade Returns Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns(
        self,
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None
    ):
        """Plot monthly returns heatmap."""
        if not HAS_MATPLOTLIB or self.equity_curve.empty:
            return
        
        from quantbox.backtest.metrics import calculate_monthly_returns
        
        monthly = calculate_monthly_returns(self.equity_curve)
        
        if monthly.empty:
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(monthly.values, aspect='auto', cmap='RdYlGn',
                       vmin=-10, vmax=10)
        
        # Set ticks
        ax.set_xticks(range(len(monthly.columns)))
        ax.set_xticklabels(monthly.columns)
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)')
        
        # Add values
        for i in range(len(monthly.index)):
            for j in range(len(monthly.columns)):
                val = monthly.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                           color=color, fontsize=8)
        
        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_trade_analysis(
        self,
        figsize: tuple = (14, 10),
        save_path: Optional[str] = None
    ):
        """Comprehensive trade analysis plots."""
        if not HAS_MATPLOTLIB or self.trades.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cumulative PnL
        if 'pnl' in self.trades.columns:
            cumulative_pnl = self.trades['pnl'].cumsum()
            axes[0, 0].plot(range(len(cumulative_pnl)), cumulative_pnl, 
                          color='#2E86AB', linewidth=2)
            axes[0, 0].fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                                   where=cumulative_pnl >= 0, color='green', alpha=0.3)
            axes[0, 0].fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                                   where=cumulative_pnl < 0, color='red', alpha=0.3)
            axes[0, 0].axhline(y=0, color='gray', linestyle='--')
            axes[0, 0].set_title('Cumulative P&L', fontweight='bold')
            axes[0, 0].set_xlabel('Trade #')
            axes[0, 0].set_ylabel('P&L ($)')
        
        # 2. Win/Loss distribution
        if 'pnl' in self.trades.columns:
            wins = self.trades[self.trades['pnl'] > 0]['pnl']
            losses = self.trades[self.trades['pnl'] <= 0]['pnl']
            
            axes[0, 1].bar(['Wins', 'Losses'], 
                          [len(wins), len(losses)],
                          color=['#27AE60', '#E74C3C'])
            axes[0, 1].set_title('Win/Loss Count', fontweight='bold')
            axes[0, 1].set_ylabel('Number of Trades')
        
        # 3. P&L by exit reason
        if 'exit_reason' in self.trades.columns and 'pnl' in self.trades.columns:
            reason_pnl = self.trades.groupby('exit_reason')['pnl'].sum()
            colors = ['#27AE60' if v > 0 else '#E74C3C' for v in reason_pnl.values]
            axes[1, 0].bar(reason_pnl.index, reason_pnl.values, color=colors)
            axes[1, 0].set_title('P&L by Exit Reason', fontweight='bold')
            axes[1, 0].set_ylabel('Total P&L ($)')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Holding period vs P&L
        if 'bars_held' in self.trades.columns and 'pnl' in self.trades.columns:
            colors = ['#27AE60' if p > 0 else '#E74C3C' for p in self.trades['pnl']]
            axes[1, 1].scatter(self.trades['bars_held'], self.trades['pnl'],
                             c=colors, alpha=0.6)
            axes[1, 1].axhline(y=0, color='gray', linestyle='--')
            axes[1, 1].set_title('Holding Period vs P&L', fontweight='bold')
            axes[1, 1].set_xlabel('Bars Held')
            axes[1, 1].set_ylabel('P&L ($)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report(
        self,
        output_dir: str = "reports",
        report_name: str = "backtest_report"
    ):
        """Generate comprehensive HTML report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        plots = {}
        
        if HAS_MATPLOTLIB:
            plots['equity'] = os.path.join(output_dir, f"{report_name}_equity.png")
            self.plot_equity_curve(save_path=plots['equity'])
            plt.close()
            
            plots['drawdown'] = os.path.join(output_dir, f"{report_name}_drawdown.png")
            self.plot_drawdown(save_path=plots['drawdown'])
            plt.close()
            
            plots['trades'] = os.path.join(output_dir, f"{report_name}_trades.png")
            self.plot_trade_analysis(save_path=plots['trades'])
            plt.close()
        
        # Generate HTML
        html = self._generate_html_report(plots)
        
        report_path = os.path.join(output_dir, f"{report_name}.html")
        with open(report_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def _generate_html_report(self, plots: Dict) -> str:
        """Generate HTML report content."""
        metrics_html = ""
        if self.metrics:
            metrics_html = f"""
            <table class="metrics-table">
                <tr><td>Total Return</td><td>{self.metrics.total_return_pct:.2f}%</td></tr>
                <tr><td>Max Drawdown</td><td>{self.metrics.max_drawdown:.2f}%</td></tr>
                <tr><td>Sharpe Ratio</td><td>{self.metrics.sharpe_ratio:.2f}</td></tr>
                <tr><td>Win Rate</td><td>{self.metrics.win_rate:.1f}%</td></tr>
                <tr><td>Profit Factor</td><td>{self.metrics.profit_factor:.2f}</td></tr>
                <tr><td>Total Trades</td><td>{self.metrics.total_trades}</td></tr>
            </table>
            """
        
        plots_html = ""
        for name, path in plots.items():
            plots_html += f'<img src="{path}" alt="{name}" class="plot">\n'
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics-table {{ width: 50%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .metrics-table td:first-child {{ font-weight: bold; color: #555; }}
        .metrics-table td:last-child {{ text-align: right; color: #2c3e50; }}
        .plot {{ max-width: 100%; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Backtest Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Metrics</h2>
        {metrics_html}
        
        <h2>Charts</h2>
        {plots_html}
    </div>
</body>
</html>
"""
