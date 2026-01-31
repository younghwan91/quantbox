"""
Parameter optimization engines for strategy tuning.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from itertools import product
import warnings

from quantbox.backtest.runner import BacktestRunner, BacktestConfig
from quantbox.backtest.metrics import PerformanceMetrics


logger = logging.getLogger('Optimizer')


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    best_params: Dict[str, float]
    best_score: float
    best_metrics: PerformanceMetrics
    
    # All evaluated points
    all_params: List[Dict]
    all_scores: List[float]
    
    # Optimization metadata
    method: str
    num_evaluations: int
    convergence_history: List[float] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for params, score in zip(self.all_params, self.all_scores):
            row = {**params, 'score': score}
            data.append(row)
        return pd.DataFrame(data).sort_values('score', ascending=False)


class ParameterOptimizer(ABC):
    """Base class for parameter optimizers."""
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        objective_func: Callable = None,
        maximize: bool = True
    ):
        """
        Args:
            param_bounds: Dict of parameter -> (min, max) bounds
            objective_func: Function that takes metrics and returns score
            maximize: Whether to maximize (True) or minimize (False) objective
        """
        self.param_bounds = param_bounds
        self.objective_func = objective_func or self._default_objective
        self.maximize = maximize
        
        # Results tracking
        self.all_params: List[Dict] = []
        self.all_scores: List[float] = []
        self.best_params: Dict = None
        self.best_score: float = float('-inf') if maximize else float('inf')
        self.best_metrics: PerformanceMetrics = None
    
    @abstractmethod
    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> OptimizationResult:
        """Run optimization."""
        pass
    
    def _default_objective(self, metrics: PerformanceMetrics) -> float:
        """Default objective: Sharpe ratio."""
        return metrics.sharpe_ratio
    
    def _evaluate(
        self,
        params: Dict,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[float, PerformanceMetrics]:
        """Evaluate a parameter set."""
        
        # Create config from params
        config = BacktestConfig(**params)
        runner = BacktestRunner(config)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = runner.run_backtest(data, progress=False)
        
        metrics = results.get('metrics', PerformanceMetrics())
        score = self.objective_func(metrics)
        
        # Handle NaN scores
        if np.isnan(score):
            score = float('-inf') if self.maximize else float('inf')
        
        return score, metrics
    
    def _update_best(
        self,
        params: Dict,
        score: float,
        metrics: PerformanceMetrics
    ):
        """Update best result if improved."""
        self.all_params.append(params.copy())
        self.all_scores.append(score)
        
        is_better = (
            (self.maximize and score > self.best_score) or
            (not self.maximize and score < self.best_score)
        )
        
        if is_better:
            self.best_params = params.copy()
            self.best_score = score
            self.best_metrics = metrics


class GridSearchOptimizer(ParameterOptimizer):
    """
    Grid search optimizer.
    
    Exhaustively searches all combinations of parameter values.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        param_steps: Dict[str, int] = None,
        objective_func: Callable = None,
        maximize: bool = True
    ):
        super().__init__(param_bounds, objective_func, maximize)
        self.param_steps = param_steps or {k: 5 for k in param_bounds.keys()}
    
    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        progress: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """Run grid search optimization."""
        
        # Generate parameter grid
        param_names = list(self.param_bounds.keys())
        param_values = []
        
        for name in param_names:
            low, high = self.param_bounds[name]
            steps = self.param_steps.get(name, 5)
            values = np.linspace(low, high, steps)
            param_values.append(values)
        
        # All combinations
        combinations = list(product(*param_values))
        total = len(combinations)
        
        logger.info(f"Grid search: {total} combinations to evaluate")
        
        iterator = tqdm(combinations, desc="Grid Search") if progress else combinations
        
        for values in iterator:
            params = dict(zip(param_names, values))
            score, metrics = self._evaluate(params, data)
            self._update_best(params, score, metrics)
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_metrics=self.best_metrics,
            all_params=self.all_params,
            all_scores=self.all_scores,
            method='grid_search',
            num_evaluations=total
        )


class RandomSearchOptimizer(ParameterOptimizer):
    """
    Random search optimizer.
    
    Randomly samples parameter space. Often more efficient than grid search.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 100,
        objective_func: Callable = None,
        maximize: bool = True,
        seed: int = None
    ):
        super().__init__(param_bounds, objective_func, maximize)
        self.n_iterations = n_iterations
        self.rng = np.random.default_rng(seed)
    
    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        progress: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """Run random search optimization."""
        
        logger.info(f"Random search: {self.n_iterations} iterations")
        
        iterator = range(self.n_iterations)
        if progress:
            iterator = tqdm(iterator, desc="Random Search")
        
        for _ in iterator:
            # Sample random parameters
            params = {}
            for name, (low, high) in self.param_bounds.items():
                params[name] = self.rng.uniform(low, high)
            
            score, metrics = self._evaluate(params, data)
            self._update_best(params, score, metrics)
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_metrics=self.best_metrics,
            all_params=self.all_params,
            all_scores=self.all_scores,
            method='random_search',
            num_evaluations=self.n_iterations
        )


class DifferentialEvolutionOptimizer(ParameterOptimizer):
    """
    Differential Evolution optimizer.
    
    Global optimization using evolutionary strategy.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        population_size: int = 15,
        max_iterations: int = 50,
        mutation: float = 0.8,
        recombination: float = 0.7,
        objective_func: Callable = None,
        maximize: bool = True,
        seed: int = None
    ):
        super().__init__(param_bounds, objective_func, maximize)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.mutation = mutation
        self.recombination = recombination
        self.rng = np.random.default_rng(seed)
        
        self.convergence_history = []
    
    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        progress: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """Run differential evolution optimization."""
        
        param_names = list(self.param_bounds.keys())
        n_dims = len(param_names)
        bounds = np.array([self.param_bounds[p] for p in param_names])
        
        # Initialize population
        population = self._init_population(bounds)
        fitness = np.full(self.population_size, float('-inf') if self.maximize else float('inf'))
        
        # Evaluate initial population
        for i, individual in enumerate(population):
            params = dict(zip(param_names, individual))
            score, metrics = self._evaluate(params, data)
            fitness[i] = score
            self._update_best(params, score, metrics)
        
        logger.info(f"DE: pop_size={self.population_size}, max_iter={self.max_iterations}")
        
        # Evolution loop
        iterator = range(self.max_iterations)
        if progress:
            iterator = tqdm(iterator, desc="Differential Evolution")
        
        for generation in iterator:
            for i in range(self.population_size):
                # Mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = self.rng.choice(candidates, 3, replace=False)
                
                mutant = population[a] + self.mutation * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = self.rng.random(n_dims) < self.recombination
                if not cross_points.any():
                    cross_points[self.rng.integers(0, n_dims)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                params = dict(zip(param_names, trial))
                score, metrics = self._evaluate(params, data)
                self._update_best(params, score, metrics)
                
                is_better = (
                    (self.maximize and score > fitness[i]) or
                    (not self.maximize and score < fitness[i])
                )
                
                if is_better:
                    population[i] = trial
                    fitness[i] = score
            
            # Track convergence
            self.convergence_history.append(self.best_score)
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_metrics=self.best_metrics,
            all_params=self.all_params,
            all_scores=self.all_scores,
            method='differential_evolution',
            num_evaluations=len(self.all_scores),
            convergence_history=self.convergence_history
        )
    
    def _init_population(self, bounds: np.ndarray) -> np.ndarray:
        """Initialize random population."""
        population = self.rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            (self.population_size, len(bounds))
        )
        return population


class BayesianOptimizer(ParameterOptimizer):
    """
    Bayesian optimization using Gaussian Process surrogate model.
    
    Efficient for expensive objective functions.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 50,
        n_initial: int = 5,
        exploration_weight: float = 2.0,
        objective_func: Callable = None,
        maximize: bool = True,
        seed: int = None
    ):
        super().__init__(param_bounds, objective_func, maximize)
        self.n_iterations = n_iterations
        self.n_initial = n_initial
        self.exploration_weight = exploration_weight
        self.rng = np.random.default_rng(seed)
        
        self.X_observed = []
        self.y_observed = []
    
    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        progress: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """Run Bayesian optimization."""
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            logger.warning("sklearn not available, falling back to random search")
            fallback = RandomSearchOptimizer(
                self.param_bounds, self.n_iterations, 
                self.objective_func, self.maximize
            )
            return fallback.optimize(data, progress)
        
        param_names = list(self.param_bounds.keys())
        bounds = np.array([self.param_bounds[p] for p in param_names])
        
        # Initial random exploration
        for _ in range(self.n_initial):
            params = {
                name: self.rng.uniform(low, high)
                for name, (low, high) in self.param_bounds.items()
            }
            x = np.array([params[p] for p in param_names])
            
            score, metrics = self._evaluate(params, data)
            self._update_best(params, score, metrics)
            
            self.X_observed.append(x)
            self.y_observed.append(score if self.maximize else -score)
        
        # Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        logger.info(f"Bayesian optimization: {self.n_iterations} iterations")
        
        remaining = self.n_iterations - self.n_initial
        iterator = range(remaining)
        if progress:
            iterator = tqdm(iterator, desc="Bayesian Optimization")
        
        for _ in iterator:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            
            # Fit GP
            gp.fit(X, y)
            
            # Find next point using acquisition function
            x_next = self._acquisition_search(gp, bounds)
            
            # Evaluate
            params = dict(zip(param_names, x_next))
            score, metrics = self._evaluate(params, data)
            self._update_best(params, score, metrics)
            
            self.X_observed.append(x_next)
            self.y_observed.append(score if self.maximize else -score)
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            best_metrics=self.best_metrics,
            all_params=self.all_params,
            all_scores=self.all_scores,
            method='bayesian_optimization',
            num_evaluations=len(self.all_scores)
        )
    
    def _acquisition_search(
        self,
        gp,
        bounds: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """Find point that maximizes Expected Improvement."""
        
        from scipy.stats import norm
        
        # Best observed value
        y_best = max(self.y_observed)
        
        # Random samples
        X_samples = self.rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            (n_samples, len(bounds))
        )
        
        # Calculate EI for each sample
        mu, sigma = gp.predict(X_samples, return_std=True)
        
        with np.errstate(divide='warn'):
            Z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-8] = 0.0
        
        # Add exploration bonus
        ei += self.exploration_weight * sigma
        
        return X_samples[np.argmax(ei)]


def optimize_strategy(
    data: Dict[str, pd.DataFrame],
    param_bounds: Dict[str, Tuple[float, float]],
    method: str = 'differential_evolution',
    objective: str = 'sharpe',
    **kwargs
) -> OptimizationResult:
    """
    Convenience function for strategy optimization.
    
    Args:
        data: Historical data
        param_bounds: Parameter bounds
        method: 'grid', 'random', 'de', or 'bayesian'
        objective: 'sharpe', 'calmar', 'return', or 'sortino'
        
    Returns:
        OptimizationResult
    """
    # Create objective function
    if objective == 'sharpe':
        obj_func = lambda m: m.sharpe_ratio
    elif objective == 'calmar':
        obj_func = lambda m: m.calmar_ratio
    elif objective == 'return':
        obj_func = lambda m: m.total_return_pct
    elif objective == 'sortino':
        obj_func = lambda m: m.sortino_ratio
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Create optimizer
    if method == 'grid':
        optimizer = GridSearchOptimizer(param_bounds, objective_func=obj_func, **kwargs)
    elif method == 'random':
        optimizer = RandomSearchOptimizer(param_bounds, objective_func=obj_func, **kwargs)
    elif method in ['de', 'differential_evolution']:
        optimizer = DifferentialEvolutionOptimizer(param_bounds, objective_func=obj_func, **kwargs)
    elif method in ['bayesian', 'bo']:
        optimizer = BayesianOptimizer(param_bounds, objective_func=obj_func, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimizer.optimize(data)
