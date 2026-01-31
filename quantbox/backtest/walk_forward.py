"""
Walk-forward analysis for robust strategy evaluation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from quantbox.backtest.runner import BacktestRunner, BacktestConfig
from quantbox.backtest.metrics import PerformanceMetrics, calculate_metrics


logger = logging.getLogger('WalkForward')


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Optimized parameters
    best_params: Dict = None
    train_metrics: PerformanceMetrics = None
    test_metrics: PerformanceMetrics = None
    
    # Efficiency ratio
    efficiency_ratio: float = 0.0  # test_return / train_return


class WalkForwardAnalysis:
    """
    Walk-forward optimization and analysis.
    
    Splits data into rolling train/test windows, optimizes parameters
    on training data, and validates on out-of-sample test data.
    """
    
    def __init__(
        self,
        train_periods: int = 500,  # Bars for training
        test_periods: int = 100,   # Bars for testing
        step_periods: int = 100,   # Bars to step forward
        min_trades: int = 10       # Minimum trades for valid window
    ):
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_periods = step_periods
        self.min_trades = min_trades
        
        # Results
        self.windows: List[WalkForwardWindow] = []
        self.combined_equity: pd.DataFrame = None
        self.combined_trades: List = []
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        param_ranges: Dict[str, Tuple[float, float]],
        optimization_func: callable = None,
        progress: bool = True
    ) -> Dict:
        """
        Run walk-forward analysis.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            param_ranges: Dict of parameter -> (min, max) bounds
            optimization_func: Custom optimization function
            progress: Show progress bar
            
        Returns:
            Dict with combined results
        """
        # Generate windows
        windows = self._generate_windows(data)
        logger.info(f"Generated {len(windows)} walk-forward windows")
        
        if not windows:
            return {}
        
        # Process each window
        iterator = tqdm(enumerate(windows), total=len(windows), desc="Walk-Forward") if progress else enumerate(windows)
        
        for i, (train_start, train_end, test_start, test_end) in iterator:
            window_result = self._process_window(
                i, data, 
                train_start, train_end, test_start, test_end,
                param_ranges, optimization_func
            )
            if window_result:
                self.windows.append(window_result)
        
        # Combine results
        return self._compile_results()
    
    def _generate_windows(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward windows."""
        first_df = next(iter(data.values()))
        timestamps = first_df.index.tolist()
        total_bars = len(timestamps)
        
        windows = []
        total_window = self.train_periods + self.test_periods
        
        i = 0
        while i + total_window <= total_bars:
            train_start = timestamps[i]
            train_end = timestamps[i + self.train_periods - 1]
            test_start = timestamps[i + self.train_periods]
            test_end = timestamps[min(i + total_window - 1, total_bars - 1)]
            
            windows.append((train_start, train_end, test_start, test_end))
            i += self.step_periods
        
        return windows
    
    def _process_window(
        self,
        window_id: int,
        data: Dict[str, pd.DataFrame],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        param_ranges: Dict,
        optimization_func: callable
    ) -> Optional[WalkForwardWindow]:
        """Process a single walk-forward window."""
        
        # Split data
        train_data = {
            symbol: df.loc[train_start:train_end].copy()
            for symbol, df in data.items()
            if train_start in df.index
        }
        
        test_data = {
            symbol: df.loc[test_start:test_end].copy()
            for symbol, df in data.items()
            if test_start in df.index
        }
        
        if not train_data or not test_data:
            return None
        
        # Optimize on training data
        if optimization_func:
            best_params, train_metrics = optimization_func(train_data, param_ranges)
        else:
            best_params, train_metrics = self._default_optimize(train_data, param_ranges)
        
        if not best_params:
            return None
        
        # Test on out-of-sample data
        test_metrics = self._test_params(test_data, best_params)
        
        # Calculate efficiency ratio
        efficiency = 0.0
        if train_metrics and train_metrics.total_return_pct != 0:
            efficiency = test_metrics.total_return_pct / train_metrics.total_return_pct
        
        return WalkForwardWindow(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            efficiency_ratio=efficiency
        )
    
    def _default_optimize(
        self,
        train_data: Dict[str, pd.DataFrame],
        param_ranges: Dict
    ) -> Tuple[Dict, PerformanceMetrics]:
        """Default grid search optimization."""
        
        best_params = None
        best_metrics = None
        best_score = float('-inf')
        
        # Simple grid search
        param_names = list(param_ranges.keys())
        param_values = [
            np.linspace(r[0], r[1], 5) for r in param_ranges.values()
        ]
        
        from itertools import product
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            # Run backtest
            config = BacktestConfig(**params)
            runner = BacktestRunner(config)
            results = runner.run_backtest(train_data, progress=False)
            
            metrics = results.get('metrics', PerformanceMetrics())
            
            # Score by Sharpe ratio
            score = metrics.sharpe_ratio
            
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics
    
    def _test_params(
        self,
        test_data: Dict[str, pd.DataFrame],
        params: Dict
    ) -> PerformanceMetrics:
        """Test parameters on out-of-sample data."""
        
        config = BacktestConfig(**params)
        runner = BacktestRunner(config)
        results = runner.run_backtest(test_data, progress=False)
        
        return results.get('metrics', PerformanceMetrics())
    
    def _compile_results(self) -> Dict:
        """Compile walk-forward results."""
        
        if not self.windows:
            return {}
        
        # Aggregate metrics
        test_returns = [w.test_metrics.total_return_pct for w in self.windows if w.test_metrics]
        train_returns = [w.train_metrics.total_return_pct for w in self.windows if w.train_metrics]
        efficiencies = [w.efficiency_ratio for w in self.windows]
        
        # Summary statistics
        summary = {
            'num_windows': len(self.windows),
            'avg_test_return': np.mean(test_returns) if test_returns else 0,
            'std_test_return': np.std(test_returns) if test_returns else 0,
            'avg_train_return': np.mean(train_returns) if train_returns else 0,
            'avg_efficiency': np.mean(efficiencies) if efficiencies else 0,
            'min_efficiency': min(efficiencies) if efficiencies else 0,
            'max_efficiency': max(efficiencies) if efficiencies else 0,
            'positive_windows': sum(1 for r in test_returns if r > 0),
            'negative_windows': sum(1 for r in test_returns if r <= 0),
        }
        
        # Window details
        window_details = [
            {
                'window': w.window_id,
                'train_start': w.train_start,
                'train_end': w.train_end,
                'test_start': w.test_start,
                'test_end': w.test_end,
                'best_params': w.best_params,
                'train_return': w.train_metrics.total_return_pct if w.train_metrics else 0,
                'test_return': w.test_metrics.total_return_pct if w.test_metrics else 0,
                'efficiency': w.efficiency_ratio,
            }
            for w in self.windows
        ]
        
        return {
            'summary': summary,
            'windows': pd.DataFrame(window_details),
            'raw_windows': self.windows,
        }
    
    def get_recommended_params(self) -> Dict:
        """
        Get recommended parameters based on walk-forward analysis.
        
        Returns average of parameters weighted by test performance.
        """
        if not self.windows:
            return {}
        
        # Weight by positive test returns
        weighted_params = {}
        total_weight = 0
        
        for w in self.windows:
            if w.test_metrics and w.test_metrics.total_return_pct > 0:
                weight = w.test_metrics.total_return_pct
                for key, value in w.best_params.items():
                    if key not in weighted_params:
                        weighted_params[key] = 0
                    weighted_params[key] += value * weight
                total_weight += weight
        
        if total_weight > 0:
            for key in weighted_params:
                weighted_params[key] /= total_weight
        
        return weighted_params
