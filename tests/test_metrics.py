"""
Tests for backtest metrics calculation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from quantbox.backtest.metrics import (
    PerformanceMetrics, calculate_metrics, monte_carlo_analysis,
    compare_strategies, calculate_monthly_returns
)


# Mock TradeRecord for testing
@dataclass
class MockTradeRecord:
    pnl: float
    bars_held: int = 1


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_default_values(self):
        metrics = PerformanceMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
    
    def test_to_dict(self):
        metrics = PerformanceMetrics(
            total_return_pct=25.0,
            sharpe_ratio=1.5,
            max_drawdown=10.0
        )
        
        data = metrics.to_dict()
        assert 'total_return_pct' in data
        assert 'sharpe_ratio' in data
    
    def test_summary(self):
        metrics = PerformanceMetrics(
            total_return_pct=25.0,
            sharpe_ratio=1.5,
            total_trades=100,
            win_rate=55.0
        )
        
        summary = metrics.summary()
        assert 'Return' in summary
        assert 'Sharpe' in summary


@pytest.fixture
def sample_equity_curve():
    """Create sample equity curve for testing."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    # Simulate equity curve with some growth and volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)  # ~12.5% annual, 32% vol
    equity = 10000 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'equity': equity,
        'timestamp': dates
    }).set_index('timestamp')


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    np.random.seed(42)
    n_trades = 50
    
    trades = []
    for i in range(n_trades):
        pnl = np.random.normal(50, 200)  # Average $50 profit with high variance
        trades.append({
            'pnl': pnl,
            'return_pct': pnl / 10000,  # Assuming $10k position size
            'duration': np.random.randint(1, 48),
            'symbol': f"PAIR{i % 5}"
        })
    
    return trades


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_basic_calculation(self, sample_equity_curve, sample_trades):
        trades = [MockTradeRecord(pnl=t['pnl'], bars_held=t['duration']) for t in sample_trades]
        metrics = calculate_metrics(sample_equity_curve, trades, initial_capital=10000.0)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == len(sample_trades)
    
    def test_empty_equity_curve(self, sample_trades):
        empty_curve = pd.DataFrame(columns=['equity'])
        trades = [MockTradeRecord(pnl=t['pnl'], bars_held=t['duration']) for t in sample_trades]
        metrics = calculate_metrics(empty_curve, trades, initial_capital=10000.0)
        
        assert metrics.total_return == 0.0
    
    def test_empty_trades(self, sample_equity_curve):
        metrics = calculate_metrics(sample_equity_curve, [], initial_capital=10000.0)
        
        assert metrics.total_trades == 0
    
    def test_drawdown_calculation(self, sample_equity_curve, sample_trades):
        trades = [MockTradeRecord(pnl=t['pnl'], bars_held=t['duration']) for t in sample_trades]
        metrics = calculate_metrics(sample_equity_curve, trades, initial_capital=10000.0)
        
        # Max drawdown should be non-negative
        assert metrics.max_drawdown >= 0.0
        # Max drawdown should be reasonable for our test data
        assert metrics.max_drawdown < 100.0
    
    def test_win_loss_stats(self, sample_equity_curve, sample_trades):
        trades = [MockTradeRecord(pnl=t['pnl'], bars_held=t['duration']) for t in sample_trades]
        metrics = calculate_metrics(sample_equity_curve, trades, initial_capital=10000.0)
        
        # Win rate should be between 0 and 100 (percentage)
        assert 0.0 <= metrics.win_rate <= 100.0


class TestMonteCarloAnalysis:
    """Tests for monte_carlo_analysis function."""
    
    def test_basic_simulation(self, sample_trades):
        trades = [MockTradeRecord(pnl=t['pnl'], bars_held=t['duration']) for t in sample_trades]
        results = monte_carlo_analysis(trades, initial_capital=10000.0, num_simulations=100)
        
        assert 'final_capital' in results
        assert 'max_drawdown' in results
    
    def test_empty_trades(self):
        results = monte_carlo_analysis([], initial_capital=10000.0, num_simulations=100)
        
        assert results == {}
    
    def test_confidence_intervals(self, sample_trades):
        trades = [MockTradeRecord(pnl=t['pnl'], bars_held=t['duration']) for t in sample_trades]
        results = monte_carlo_analysis(trades, initial_capital=10000.0, num_simulations=1000)
        
        final_capital = results['final_capital']
        # Lower CI should be less than upper CI
        assert final_capital['lower_ci'] <= final_capital['upper_ci']


class TestCompareStrategies:
    """Tests for compare_strategies function."""
    
    def test_comparison(self):
        results = {
            'strategy_a': {
                'metrics': PerformanceMetrics(
                    total_return_pct=20.0,
                    sharpe_ratio=1.2,
                    max_drawdown=15.0,
                    total_trades=50
                )
            },
            'strategy_b': {
                'metrics': PerformanceMetrics(
                    total_return_pct=30.0,
                    sharpe_ratio=1.5,
                    max_drawdown=20.0,
                    total_trades=75
                )
            }
        }
        
        comparison = compare_strategies(results)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2


class TestMonthlyReturns:
    """Tests for calculate_monthly_returns function."""
    
    def test_monthly_calculation(self, sample_equity_curve):
        monthly = calculate_monthly_returns(sample_equity_curve)
        
        assert isinstance(monthly, pd.DataFrame)
        # Should have monthly columns
        if not monthly.empty:
            assert len(monthly.columns) >= 1
    
    def test_empty_curve(self):
        empty_curve = pd.DataFrame(columns=['equity'])
        monthly = calculate_monthly_returns(empty_curve)
        
        assert monthly.empty
