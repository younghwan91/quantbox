"""
Tests for optimization module.
"""

import pytest
import pandas as pd
import numpy as np

from quantbox.optimization.optimizer import (
    GridSearchOptimizer, RandomSearchOptimizer,
    DifferentialEvolutionOptimizer, OptimizationResult
)
from quantbox.optimization.objective import (
    SharpeObjective, CalmarObjective, MultiObjective,
    create_objective
)
from quantbox.backtest.metrics import PerformanceMetrics


@pytest.fixture
def mock_metrics():
    """Create mock performance metrics."""
    return PerformanceMetrics(
        total_return_pct=15.0,
        sharpe_ratio=1.5,
        calmar_ratio=2.0,
        sortino_ratio=2.5,
        max_drawdown=10.0,
        win_rate=60.0,
        profit_factor=2.0,
        total_trades=50
    )


class TestObjectiveFunctions:
    """Tests for objective functions."""
    
    def test_sharpe_objective(self, mock_metrics):
        obj = SharpeObjective()
        score = obj(mock_metrics)
        
        assert score == mock_metrics.sharpe_ratio
        assert obj.name == "Sharpe Ratio"
    
    def test_sharpe_min_trades(self):
        obj = SharpeObjective(min_trades=100)
        metrics = PerformanceMetrics(sharpe_ratio=2.0, total_trades=50)
        
        score = obj(metrics)
        assert score == -10.0  # Penalty for too few trades
    
    def test_calmar_objective(self, mock_metrics):
        obj = CalmarObjective()
        score = obj(mock_metrics)
        
        assert score == mock_metrics.calmar_ratio
    
    def test_calmar_drawdown_limit(self):
        obj = CalmarObjective(max_drawdown_limit=5.0)
        metrics = PerformanceMetrics(calmar_ratio=2.0, max_drawdown=10.0, total_trades=50)
        
        score = obj(metrics)
        assert score == -10.0  # Exceeds drawdown limit
    
    def test_multi_objective(self, mock_metrics):
        obj = MultiObjective(
            objectives=[SharpeObjective(), CalmarObjective()],
            weights=[0.5, 0.5]
        )
        
        score = obj(mock_metrics)
        assert isinstance(score, float)
    
    def test_create_objective_factory(self):
        obj = create_objective('sharpe')
        assert isinstance(obj, SharpeObjective)
        
        obj = create_objective('calmar')
        assert isinstance(obj, CalmarObjective)
    
    def test_create_multi_objective(self):
        obj = create_objective('multi', objectives=['sharpe', 'calmar'])
        assert isinstance(obj, MultiObjective)


class TestOptimizationResult:
    """Tests for OptimizationResult."""
    
    def test_to_dataframe(self):
        result = OptimizationResult(
            best_params={'a': 1.0, 'b': 2.0},
            best_score=1.5,
            best_metrics=PerformanceMetrics(),
            all_params=[{'a': 1.0, 'b': 2.0}, {'a': 1.5, 'b': 2.5}],
            all_scores=[1.5, 1.2],
            method='test',
            num_evaluations=2
        )
        
        df = result.to_dataframe()
        assert len(df) == 2
        assert 'a' in df.columns
        assert 'score' in df.columns


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""
    
    def test_initialization(self):
        bounds = {'entry_zscore': (1.0, 3.0), 'exit_zscore': (0.1, 1.0)}
        optimizer = GridSearchOptimizer(bounds)
        
        assert optimizer.param_bounds == bounds
        assert optimizer.maximize is True
    
    def test_param_steps(self):
        bounds = {'a': (0.0, 1.0)}
        optimizer = GridSearchOptimizer(bounds, param_steps={'a': 3})
        
        assert optimizer.param_steps['a'] == 3


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""
    
    def test_initialization(self):
        bounds = {'a': (0.0, 1.0)}
        optimizer = RandomSearchOptimizer(bounds, n_iterations=50)
        
        assert optimizer.n_iterations == 50
    
    def test_seed_reproducibility(self):
        bounds = {'a': (0.0, 1.0)}
        
        opt1 = RandomSearchOptimizer(bounds, seed=42)
        opt2 = RandomSearchOptimizer(bounds, seed=42)
        
        # Same seed should give same random state
        val1 = opt1.rng.random()
        val2 = opt2.rng.random()
        assert val1 == val2


class TestDifferentialEvolutionOptimizer:
    """Tests for DifferentialEvolutionOptimizer."""
    
    def test_initialization(self):
        bounds = {'a': (0.0, 1.0)}
        optimizer = DifferentialEvolutionOptimizer(
            bounds,
            population_size=10,
            max_iterations=20
        )
        
        assert optimizer.population_size == 10
        assert optimizer.max_iterations == 20
    
    def test_init_population(self):
        bounds_dict = {'a': (0.0, 1.0), 'b': (2.0, 4.0)}
        optimizer = DifferentialEvolutionOptimizer(bounds_dict, population_size=5)
        
        bounds = np.array([[0.0, 1.0], [2.0, 4.0]])
        pop = optimizer._init_population(bounds)
        
        assert pop.shape == (5, 2)
        assert np.all(pop[:, 0] >= 0.0) and np.all(pop[:, 0] <= 1.0)
        assert np.all(pop[:, 1] >= 2.0) and np.all(pop[:, 1] <= 4.0)
