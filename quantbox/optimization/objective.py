"""
Objective functions for optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
import numpy as np

from quantbox.backtest.metrics import PerformanceMetrics


class ObjectiveFunction(ABC):
    """Base class for objective functions."""
    
    @abstractmethod
    def __call__(self, metrics: PerformanceMetrics) -> float:
        """Calculate objective value."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Objective name."""
        pass


class SharpeObjective(ObjectiveFunction):
    """
    Sharpe ratio objective.
    
    Maximizes risk-adjusted returns.
    """
    
    def __init__(self, min_trades: int = 10):
        self.min_trades = min_trades
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        # Penalize if too few trades
        if metrics.total_trades < self.min_trades:
            return -10.0
        
        sharpe = metrics.sharpe_ratio
        
        # Handle edge cases
        if np.isnan(sharpe) or np.isinf(sharpe):
            return -10.0
        
        return sharpe
    
    @property
    def name(self) -> str:
        return "Sharpe Ratio"


class CalmarObjective(ObjectiveFunction):
    """
    Calmar ratio objective.
    
    Maximizes return relative to maximum drawdown.
    """
    
    def __init__(self, min_trades: int = 10, max_drawdown_limit: float = 50.0):
        self.min_trades = min_trades
        self.max_drawdown_limit = max_drawdown_limit
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        if metrics.total_trades < self.min_trades:
            return -10.0
        
        # Hard constraint on max drawdown
        if metrics.max_drawdown > self.max_drawdown_limit:
            return -10.0
        
        calmar = metrics.calmar_ratio
        
        if np.isnan(calmar) or np.isinf(calmar):
            return -10.0
        
        return calmar
    
    @property
    def name(self) -> str:
        return "Calmar Ratio"


class SortinoObjective(ObjectiveFunction):
    """
    Sortino ratio objective.
    
    Focuses on downside risk only.
    """
    
    def __init__(self, min_trades: int = 10):
        self.min_trades = min_trades
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        if metrics.total_trades < self.min_trades:
            return -10.0
        
        sortino = metrics.sortino_ratio
        
        if np.isnan(sortino) or np.isinf(sortino):
            return -10.0
        
        return sortino
    
    @property
    def name(self) -> str:
        return "Sortino Ratio"


class ReturnObjective(ObjectiveFunction):
    """
    Total return objective.
    
    Maximizes absolute return (ignoring risk).
    """
    
    def __init__(self, min_trades: int = 10):
        self.min_trades = min_trades
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        if metrics.total_trades < self.min_trades:
            return -1000.0
        
        return metrics.total_return_pct
    
    @property
    def name(self) -> str:
        return "Total Return"


class ProfitFactorObjective(ObjectiveFunction):
    """
    Profit factor objective.
    
    Maximizes ratio of gross profit to gross loss.
    """
    
    def __init__(self, min_trades: int = 20):
        self.min_trades = min_trades
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        if metrics.total_trades < self.min_trades:
            return 0.0
        
        pf = metrics.profit_factor
        
        if np.isnan(pf) or np.isinf(pf):
            return 0.0
        
        return pf
    
    @property
    def name(self) -> str:
        return "Profit Factor"


class WinRateObjective(ObjectiveFunction):
    """
    Win rate objective with profit factor constraint.
    """
    
    def __init__(self, min_trades: int = 20, min_profit_factor: float = 1.0):
        self.min_trades = min_trades
        self.min_profit_factor = min_profit_factor
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        if metrics.total_trades < self.min_trades:
            return 0.0
        
        if metrics.profit_factor < self.min_profit_factor:
            return 0.0
        
        return metrics.win_rate
    
    @property
    def name(self) -> str:
        return "Win Rate"


class MultiObjective(ObjectiveFunction):
    """
    Multi-objective function combining multiple criteria.
    
    Uses weighted sum approach.
    """
    
    def __init__(
        self,
        objectives: List[ObjectiveFunction] = None,
        weights: List[float] = None
    ):
        if objectives is None:
            # Default: Sharpe + Calmar
            objectives = [SharpeObjective(), CalmarObjective()]
            weights = [0.5, 0.5]
        
        self.objectives = objectives
        self.weights = weights or [1.0 / len(objectives)] * len(objectives)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        score = 0.0
        for obj, weight in zip(self.objectives, self.weights):
            obj_score = obj(metrics)
            # Normalize to similar scales
            score += weight * self._normalize(obj, obj_score)
        return score
    
    def _normalize(self, obj: ObjectiveFunction, score: float) -> float:
        """Normalize score to [0, 1] range approximately."""
        if isinstance(obj, (SharpeObjective, SortinoObjective)):
            # Sharpe typically -3 to 3
            return (score + 3) / 6
        elif isinstance(obj, CalmarObjective):
            # Calmar typically 0 to 5
            return score / 5
        elif isinstance(obj, ReturnObjective):
            # Return typically -100 to 200
            return (score + 100) / 300
        elif isinstance(obj, ProfitFactorObjective):
            # PF typically 0 to 3
            return score / 3
        elif isinstance(obj, WinRateObjective):
            # Win rate 0 to 100
            return score / 100
        else:
            return score
    
    @property
    def name(self) -> str:
        names = [obj.name for obj in self.objectives]
        return f"Multi({', '.join(names)})"


class ConstrainedObjective(ObjectiveFunction):
    """
    Objective with hard constraints.
    """
    
    def __init__(
        self,
        base_objective: ObjectiveFunction,
        constraints: Dict[str, Any] = None
    ):
        """
        Args:
            base_objective: Primary objective function
            constraints: Dict with constraint specifications:
                - min_trades: Minimum number of trades
                - max_drawdown: Maximum drawdown percentage
                - min_win_rate: Minimum win rate percentage
                - min_profit_factor: Minimum profit factor
        """
        self.base_objective = base_objective
        self.constraints = constraints or {}
    
    def __call__(self, metrics: PerformanceMetrics) -> float:
        # Check constraints
        if 'min_trades' in self.constraints:
            if metrics.total_trades < self.constraints['min_trades']:
                return -100.0
        
        if 'max_drawdown' in self.constraints:
            if metrics.max_drawdown > self.constraints['max_drawdown']:
                return -100.0
        
        if 'min_win_rate' in self.constraints:
            if metrics.win_rate < self.constraints['min_win_rate']:
                return -100.0
        
        if 'min_profit_factor' in self.constraints:
            if metrics.profit_factor < self.constraints['min_profit_factor']:
                return -100.0
        
        # All constraints satisfied
        return self.base_objective(metrics)
    
    @property
    def name(self) -> str:
        return f"Constrained({self.base_objective.name})"


def create_objective(
    name: str,
    **kwargs
) -> ObjectiveFunction:
    """
    Factory function to create objective functions.
    
    Args:
        name: 'sharpe', 'calmar', 'sortino', 'return', 'profit_factor', 'multi'
        **kwargs: Additional arguments for the objective
        
    Returns:
        ObjectiveFunction instance
    """
    objectives = {
        'sharpe': SharpeObjective,
        'calmar': CalmarObjective,
        'sortino': SortinoObjective,
        'return': ReturnObjective,
        'profit_factor': ProfitFactorObjective,
        'win_rate': WinRateObjective,
    }
    
    if name == 'multi':
        obj_names = kwargs.get('objectives', ['sharpe', 'calmar'])
        weights = kwargs.get('weights')
        objs = [objectives[n]() for n in obj_names]
        return MultiObjective(objs, weights)
    
    if name not in objectives:
        raise ValueError(f"Unknown objective: {name}. Choose from {list(objectives.keys())}")
    
    return objectives[name](**kwargs)
