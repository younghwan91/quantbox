"""
Performance metrics calculation for backtesting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in periods
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Profit/Loss
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Holding periods
    avg_holding_period: float = 0.0
    avg_winning_holding: float = 0.0
    avg_losing_holding: float = 0.0
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Capital
    final_capital: float = 0.0
    peak_capital: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return_pct': f"{self.total_return_pct:.2f}%",
            'annualized_return': f"{self.annualized_return:.2f}%",
            'max_drawdown': f"{self.max_drawdown:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'calmar_ratio': f"{self.calmar_ratio:.2f}",
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.1f}%",
            'profit_factor': f"{self.profit_factor:.2f}",
            'avg_trade': f"${self.avg_trade:.2f}",
            'final_capital': f"${self.final_capital:.2f}",
        }
    
    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Return: {self.total_return_pct:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Max DD: {self.max_drawdown:.2f}% | "
            f"Win Rate: {self.win_rate:.1f}% | "
            f"Trades: {self.total_trades}"
        )


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: List,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity_curve: DataFrame with equity values
        trades: List of TradeRecord objects
        initial_capital: Starting capital
        risk_free_rate: Risk-free rate for Sharpe calculation
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()
    
    if equity_curve.empty:
        return metrics
    
    # Extract equity values
    if 'equity' in equity_curve.columns:
        equity = equity_curve['equity'].values
    else:
        equity = equity_curve.iloc[:, 0].values
    
    # Basic returns
    metrics.final_capital = equity[-1]
    metrics.peak_capital = np.max(equity)
    metrics.total_return = equity[-1] - initial_capital
    metrics.total_return_pct = (metrics.total_return / initial_capital) * 100
    
    # Calculate returns series
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]
    
    if len(returns) > 0:
        # Volatility
        metrics.volatility = np.std(returns) * 100
        metrics.annualized_volatility = metrics.volatility * np.sqrt(periods_per_year)
        
        # Annualized return
        total_periods = len(equity)
        years = total_periods / periods_per_year
        if years > 0 and equity[-1] > 0:
            metrics.annualized_return = ((equity[-1] / initial_capital) ** (1 / years) - 1) * 100
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate / periods_per_year
        if np.std(excess_returns) > 0:
            metrics.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                metrics.sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    
    # Drawdown analysis
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    metrics.max_drawdown = np.max(drawdown) * 100
    
    # Max drawdown duration
    in_drawdown = drawdown > 0
    drawdown_periods = []
    current_period = 0
    for dd in in_drawdown:
        if dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    if current_period > 0:
        drawdown_periods.append(current_period)
    metrics.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    
    # Calmar ratio
    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
    
    # Trade statistics
    if trades:
        pnls = [t.pnl for t in trades]
        holding_periods = [t.bars_held for t in trades]
        
        metrics.total_trades = len(trades)
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades * 100) if metrics.total_trades > 0 else 0
        
        # Profit/Loss
        metrics.gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        metrics.gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')
        
        metrics.avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        metrics.avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        metrics.avg_trade = np.mean(pnls) if pnls else 0
        
        metrics.largest_win = max(pnls) if pnls else 0
        metrics.largest_loss = min(pnls) if pnls else 0
        
        # Holding periods
        metrics.avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        metrics.avg_winning_holding = np.mean([t.bars_held for t in winning_trades]) if winning_trades else 0
        metrics.avg_losing_holding = np.mean([t.bars_held for t in losing_trades]) if losing_trades else 0
        
        # Consecutive wins/losses
        win_streak, loss_streak = 0, 0
        max_win_streak, max_loss_streak = 0, 0
        for t in trades:
            if t.pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        metrics.max_consecutive_wins = max_win_streak
        metrics.max_consecutive_losses = max_loss_streak
    
    return metrics


def calculate_rolling_metrics(
    equity_curve: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    """
    if 'equity' not in equity_curve.columns:
        return pd.DataFrame()
    
    equity = equity_curve['equity']
    returns = equity.pct_change()
    
    rolling = pd.DataFrame(index=equity_curve.index)
    
    # Rolling returns
    rolling['rolling_return'] = equity / equity.shift(window) - 1
    
    # Rolling volatility
    rolling['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling['rolling_sharpe'] = (rolling_mean / rolling_std) * np.sqrt(252)
    
    # Rolling max drawdown
    def max_dd(x):
        peak = x.expanding().max()
        dd = (peak - x) / peak
        return dd.max()
    
    rolling['rolling_max_dd'] = equity.rolling(window).apply(max_dd, raw=False)
    
    return rolling


def calculate_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns table.
    """
    if 'equity' not in equity_curve.columns:
        return pd.DataFrame()
    
    equity = equity_curve['equity'].copy()
    equity.index = pd.to_datetime(equity.index)
    
    # Resample to monthly (ME = Month End for pandas >= 2.2)
    monthly = equity.resample('ME').last()
    monthly_returns = monthly.pct_change()
    
    # Pivot to year x month format
    monthly_returns.index = pd.MultiIndex.from_arrays([
        monthly_returns.index.year,
        monthly_returns.index.month
    ])
    
    pivot = monthly_returns.unstack()
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
    
    # Add yearly return
    pivot['Year'] = pivot.sum(axis=1)
    
    return pivot * 100  # Convert to percentage


def compare_strategies(
    results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Compare multiple strategy results.
    
    Args:
        results: Dict of strategy_name -> backtest results
        
    Returns:
        Comparison DataFrame
    """
    comparison = []
    
    for name, result in results.items():
        metrics = result.get('metrics', PerformanceMetrics())
        comparison.append({
            'Strategy': name,
            'Return %': metrics.total_return_pct,
            'Ann. Return %': metrics.annualized_return,
            'Max DD %': metrics.max_drawdown,
            'Sharpe': metrics.sharpe_ratio,
            'Sortino': metrics.sortino_ratio,
            'Win Rate %': metrics.win_rate,
            'Profit Factor': metrics.profit_factor,
            'Total Trades': metrics.total_trades,
            'Avg Trade': metrics.avg_trade,
        })
    
    return pd.DataFrame(comparison)


def monte_carlo_analysis(
    trades: List,
    initial_capital: float,
    num_simulations: int = 1000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Run Monte Carlo simulation on trade sequence.
    
    Randomly reorders trades to estimate distribution of outcomes.
    """
    if not trades:
        return {}
    
    pnls = [t.pnl for t in trades]
    
    final_capitals = []
    max_drawdowns = []
    
    for _ in range(num_simulations):
        # Shuffle trade order
        shuffled_pnls = np.random.permutation(pnls)
        
        # Calculate equity curve
        equity = [initial_capital]
        for pnl in shuffled_pnls:
            equity.append(equity[-1] + pnl)
        
        equity = np.array(equity)
        
        # Calculate metrics
        final_capitals.append(equity[-1])
        
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdowns.append(np.max(drawdown) * 100)
    
    # Calculate statistics
    lower_pct = (1 - confidence_level) / 2 * 100
    upper_pct = (1 - (1 - confidence_level) / 2) * 100
    
    return {
        'final_capital': {
            'mean': np.mean(final_capitals),
            'std': np.std(final_capitals),
            'median': np.median(final_capitals),
            'lower_ci': np.percentile(final_capitals, lower_pct),
            'upper_ci': np.percentile(final_capitals, upper_pct),
            'min': np.min(final_capitals),
            'max': np.max(final_capitals),
        },
        'max_drawdown': {
            'mean': np.mean(max_drawdowns),
            'std': np.std(max_drawdowns),
            'median': np.median(max_drawdowns),
            'lower_ci': np.percentile(max_drawdowns, lower_pct),
            'upper_ci': np.percentile(max_drawdowns, upper_pct),
            'worst': np.max(max_drawdowns),
        },
        'probability_of_loss': np.mean(np.array(final_capitals) < initial_capital),
        'num_simulations': num_simulations,
    }
