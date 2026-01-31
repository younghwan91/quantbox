"""
Tests for backtest metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quantbox.backtest.metrics import (
    PerformanceMetrics, calculate_metrics, calculate_monthly_returns,
    monte_carlo_analysis, compare_strategies
)
from quantbox.backtest.runner import TradeRecord


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve."""
    n = 252  # One year of daily data
    
    # Upward trending with noise
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01 + 0.0003  # Small positive drift
    equity = 10000 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    return pd.DataFrame({
        'equity': equity,
        'cash': equity * 0.3,
        'unrealized_pnl': equity * 0.1
    }, index=dates)


@pytest.fixture
def sample_trades():
    """Generate sample trades."""
    trades = []
    
    np.random.seed(42)
    base_time = datetime(2023, 1, 1)
    
    for i in range(50):
        pnl = np.random.randn() * 100 + 20  # Slight positive bias
        
        trade = TradeRecord(
            pair=('BTCUSDT', 'ETHUSDT'),
            entry_time=base_time,
            exit_time=base_time,
            entry_z_score=2.0,
            exit_z_score=0.5,
            entry_spread=0.1,
            exit_spread=0.05,
            signal=1 if np.random.random() > 0.5 else -1,
            pnl=pnl,
            pnl_pct=pnl / 1000,
            bars_held=np.random.randint(5, 50),
            exit_reason='signal',
            hedge_ratio=0.5
        )
        trades.append(trade)
    
    return trades


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""
    
    def test_default_values(self):
        metrics = PerformanceMetrics()
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.total_trades == 0
    
    def test_to_dict(self):
        metrics = PerformanceMetrics(
            total_return_pct=15.5,
            sharpe_ratio=1.5,
            max_drawdown=10.0
        )
        
        data = metrics.to_dict()
        assert '15.50%' in data['total_return_pct']
        assert '1.50' in data['sharpe_ratio']
    
    def test_summary(self):
        metrics = PerformanceMetrics(
            total_return_pct=15.5,
            sharpe_ratio=1.5,
            max_drawdown=10.0,
            win_rate=60.0,
            total_trades=100
        )
        
        summary = metrics.summary()
        assert '15.50%' in summary
        assert '1.50' in summary
        assert '60.0%' in summary


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_basic_calculation(self, sample_equity_curve, sample_trades):
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000.0
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == len(sample_trades)
        assert metrics.final_capital > 0
    
    def test_empty_equity_curve(self, sample_trades):
        metrics = calculate_metrics(
            equity_curve=pd.DataFrame(),
            trades=sample_trades,
            initial_capital=10000.0
        )
        
        assert metrics.total_return == 0.0
    
    def test_empty_trades(self, sample_equity_curve):
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=[],
            initial_capital=10000.0
        )
        
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
    
    def test_drawdown_calculation(self, sample_equity_curve):
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=[],
            initial_capital=10000.0
        )
        
        # Drawdown should be between 0 and 100
        assert 0 <= metrics.max_drawdown <= 100
    
    def test_win_loss_stats(self, sample_trades):
        metrics = calculate_metrics(
            equity_curve=pd.DataFrame({'equity': [10000, 11000]}),
            trades=sample_trades,
            initial_capital=10000.0
        )
        
        assert metrics.winning_trades + metrics.losing_trades == metrics.total_trades
        assert 0 <= metrics.win_rate <= 100


class TestMonteCarloAnalysis:
    """Tests for monte_carlo_analysis function."""
    
    def test_basic_simulation(self, sample_trades):
        results = monte_carlo_analysis(
            trades=sample_trades,
            initial_capital=10000.0,
            num_simulations=100
        )
        
        assert 'final_capital' in results
        assert 'max_drawdown' in results
        assert results['num_simulations'] == 100
    
    def test_empty_trades(self):
        results = monte_carlo_analysis(
            trades=[],
            initial_capital=10000.0
        )
        
        assert results == {}
    
    def test_confidence_intervals(self, sample_trades):
        results = monte_carlo_analysis(
            trades=sample_trades,
            initial_capital=10000.0,
            num_simulations=1000,
            confidence_level=0.95
        )
        
        fc = results['final_capital']
        assert fc['lower_ci'] <= fc['median'] <= fc['upper_ci']


class TestCompareStrategies:
    """Tests for compare_strategies function."""
    
    def test_comparison(self):
        results = {
            'Strategy A': {
                'metrics': PerformanceMetrics(
                    total_return_pct=15.0,
                    sharpe_ratio=1.5,
                    max_drawdown=10.0,
                    win_rate=60.0,
                    profit_factor=2.0,
                    total_trades=100
                )
            },
            'Strategy B': {
                'metrics': PerformanceMetrics(
                    total_return_pct=20.0,
                    sharpe_ratio=1.2,
                    max_drawdown=15.0,
                    win_rate=55.0,
                    profit_factor=1.8,
                    total_trades=150
                )
            }
        }
        
        comparison = compare_strategies(results)
        
        assert len(comparison) == 2
        assert 'Strategy' in comparison.columns
        assert 'Sharpe' in comparison.columns


class TestMonthlyReturns:
    """Tests for calculate_monthly_returns function."""
    
    def test_monthly_calculation(self, sample_equity_curve):
        monthly = calculate_monthly_returns(sample_equity_curve)
        
        # Should have monthly columns
        assert len(monthly.columns) > 0
    
    def test_empty_curve(self):
        monthly = calculate_monthly_returns(pd.DataFrame())
        assert monthly.empty
