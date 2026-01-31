"""
Sensitivity analysis for parameter robustness testing.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from quantbox.backtest.runner import BacktestRunner, BacktestConfig
from quantbox.backtest.metrics import PerformanceMetrics


logger = logging.getLogger('SensitivityAnalyzer')


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a single parameter."""
    parameter: str
    values: List[float]
    scores: List[float]
    metrics: List[Dict]
    
    # Sensitivity metrics
    mean_score: float
    std_score: float
    score_range: float
    sensitivity_index: float  # Normalized sensitivity
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'value': self.values,
            'score': self.scores
        })


class SensitivityAnalyzer:
    """
    Analyze strategy sensitivity to parameter changes.
    
    Helps identify robust parameter regions and unstable parameters.
    """
    
    def __init__(
        self,
        base_params: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]] = None,
        objective_func: callable = None
    ):
        """
        Args:
            base_params: Baseline parameter values
            param_bounds: Optional bounds for each parameter
            objective_func: Scoring function (metrics -> score)
        """
        self.base_params = base_params.copy()
        self.param_bounds = param_bounds or {}
        self.objective_func = objective_func or (lambda m: m.sharpe_ratio)
        
        self.results: Dict[str, SensitivityResult] = {}
    
    def analyze_parameter(
        self,
        data: Dict[str, pd.DataFrame],
        param_name: str,
        values: List[float] = None,
        n_points: int = 11,
        progress: bool = True
    ) -> SensitivityResult:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            data: Historical data
            param_name: Parameter to analyze
            values: Specific values to test (optional)
            n_points: Number of points if values not specified
            
        Returns:
            SensitivityResult
        """
        if values is None:
            # Generate values around baseline
            if param_name in self.param_bounds:
                low, high = self.param_bounds[param_name]
            else:
                base = self.base_params.get(param_name, 1.0)
                low, high = base * 0.5, base * 1.5
            
            values = np.linspace(low, high, n_points)
        
        scores = []
        all_metrics = []
        
        iterator = values
        if progress:
            iterator = tqdm(values, desc=f"Analyzing {param_name}")
        
        for value in iterator:
            params = self.base_params.copy()
            params[param_name] = value
            
            config = BacktestConfig(**params)
            runner = BacktestRunner(config)
            results = runner.run_backtest(data, progress=False)
            
            metrics = results.get('metrics', PerformanceMetrics())
            score = self.objective_func(metrics)
            
            scores.append(score)
            all_metrics.append(metrics.to_dict())
        
        # Calculate sensitivity metrics
        scores_array = np.array(scores)
        valid_scores = scores_array[~np.isnan(scores_array)]
        
        mean_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0
        std_score = np.std(valid_scores) if len(valid_scores) > 0 else 0
        score_range = np.ptp(valid_scores) if len(valid_scores) > 0 else 0
        
        # Sensitivity index: coefficient of variation
        sensitivity_index = std_score / abs(mean_score) if mean_score != 0 else float('inf')
        
        result = SensitivityResult(
            parameter=param_name,
            values=list(values),
            scores=scores,
            metrics=all_metrics,
            mean_score=mean_score,
            std_score=std_score,
            score_range=score_range,
            sensitivity_index=sensitivity_index
        )
        
        self.results[param_name] = result
        return result
    
    def analyze_all(
        self,
        data: Dict[str, pd.DataFrame],
        n_points: int = 11,
        progress: bool = True
    ) -> Dict[str, SensitivityResult]:
        """
        Analyze sensitivity to all parameters.
        """
        for param_name in self.base_params.keys():
            self.analyze_parameter(data, param_name, n_points=n_points, progress=progress)
        
        return self.results
    
    def get_robust_parameters(
        self,
        sensitivity_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Get parameters that show low sensitivity (robust).
        
        Args:
            sensitivity_threshold: Maximum sensitivity index for robust params
            
        Returns:
            Dict of robust parameters
        """
        robust = {}
        for param_name, result in self.results.items():
            if result.sensitivity_index < sensitivity_threshold:
                # Use value with best score
                best_idx = np.argmax(result.scores)
                robust[param_name] = result.values[best_idx]
        return robust
    
    def get_sensitive_parameters(
        self,
        sensitivity_threshold: float = 0.5
    ) -> List[str]:
        """
        Get parameters that show high sensitivity (require careful tuning).
        """
        sensitive = []
        for param_name, result in self.results.items():
            if result.sensitivity_index >= sensitivity_threshold:
                sensitive.append(param_name)
        return sensitive
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all sensitivity analyses."""
        rows = []
        for param_name, result in self.results.items():
            rows.append({
                'parameter': param_name,
                'base_value': self.base_params.get(param_name),
                'mean_score': result.mean_score,
                'std_score': result.std_score,
                'score_range': result.score_range,
                'sensitivity_index': result.sensitivity_index,
                'robust': result.sensitivity_index < 0.5
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('sensitivity_index', ascending=True)
    
    def plot_sensitivity(
        self,
        param_name: str = None,
        figsize: tuple = (12, 8),
        save_path: str = None
    ):
        """Plot sensitivity analysis results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available")
            return
        
        if param_name:
            results_to_plot = {param_name: self.results[param_name]}
        else:
            results_to_plot = self.results
        
        n_params = len(results_to_plot)
        
        if n_params == 0:
            return
        
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for ax, (name, result) in zip(axes, results_to_plot.items()):
            ax.plot(result.values, result.scores, 'o-', linewidth=2, markersize=6)
            
            # Highlight baseline
            if name in self.base_params:
                base_val = self.base_params[name]
                ax.axvline(x=base_val, color='red', linestyle='--', 
                          label=f'Baseline: {base_val:.2f}')
            
            ax.set_title(f'{name}\nSensitivity: {result.sensitivity_index:.3f}', 
                        fontweight='bold')
            ax.set_xlabel(name)
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for ax in axes[n_params:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(
        self,
        param1: str,
        param2: str,
        data: Dict[str, pd.DataFrame],
        n_points: int = 10,
        figsize: tuple = (10, 8),
        save_path: str = None
    ):
        """
        Plot 2D sensitivity heatmap for two parameters.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        # Generate value grids
        if param1 in self.param_bounds:
            low1, high1 = self.param_bounds[param1]
        else:
            base1 = self.base_params.get(param1, 1.0)
            low1, high1 = base1 * 0.5, base1 * 1.5
        
        if param2 in self.param_bounds:
            low2, high2 = self.param_bounds[param2]
        else:
            base2 = self.base_params.get(param2, 1.0)
            low2, high2 = base2 * 0.5, base2 * 1.5
        
        values1 = np.linspace(low1, high1, n_points)
        values2 = np.linspace(low2, high2, n_points)
        
        # Evaluate grid
        scores = np.zeros((n_points, n_points))
        
        for i, v1 in enumerate(tqdm(values1, desc="2D Sensitivity")):
            for j, v2 in enumerate(values2):
                params = self.base_params.copy()
                params[param1] = v1
                params[param2] = v2
                
                config = BacktestConfig(**params)
                runner = BacktestRunner(config)
                results = runner.run_backtest(data, progress=False)
                
                metrics = results.get('metrics', PerformanceMetrics())
                scores[i, j] = self.objective_func(metrics)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(scores, aspect='auto', cmap='RdYlGn', origin='lower',
                      extent=[low2, high2, low1, high1])
        
        # Mark baseline
        if param1 in self.base_params and param2 in self.base_params:
            ax.plot(self.base_params[param2], self.base_params[param1], 
                   'k*', markersize=15, label='Baseline')
        
        # Mark best
        best_idx = np.unravel_index(np.argmax(scores), scores.shape)
        ax.plot(values2[best_idx[1]], values1[best_idx[0]], 
               'r*', markersize=15, label='Best')
        
        ax.set_xlabel(param2, fontsize=12)
        ax.set_ylabel(param1, fontsize=12)
        ax.set_title(f'Sensitivity: {param1} vs {param2}', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score')
        
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, scores


def robustness_test(
    data: Dict[str, pd.DataFrame],
    base_params: Dict[str, float],
    param_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 100,
    perturbation_pct: float = 0.1
) -> Dict:
    """
    Monte Carlo robustness test.
    
    Randomly perturbs parameters and measures stability.
    
    Args:
        data: Historical data
        base_params: Baseline parameters
        param_bounds: Parameter bounds
        n_samples: Number of random samples
        perturbation_pct: Max perturbation as percentage of value
        
    Returns:
        Dict with robustness metrics
    """
    rng = np.random.default_rng()
    scores = []
    
    for _ in tqdm(range(n_samples), desc="Robustness Test"):
        # Perturb parameters
        params = {}
        for name, base in base_params.items():
            delta = base * perturbation_pct * rng.uniform(-1, 1)
            value = base + delta
            
            # Clip to bounds
            if name in param_bounds:
                low, high = param_bounds[name]
                value = np.clip(value, low, high)
            
            params[name] = value
        
        # Evaluate
        config = BacktestConfig(**params)
        runner = BacktestRunner(config)
        results = runner.run_backtest(data, progress=False)
        
        metrics = results.get('metrics', PerformanceMetrics())
        scores.append(metrics.sharpe_ratio)
    
    scores = np.array(scores)
    valid_scores = scores[~np.isnan(scores)]
    
    return {
        'mean_score': np.mean(valid_scores),
        'std_score': np.std(valid_scores),
        'min_score': np.min(valid_scores),
        'max_score': np.max(valid_scores),
        'positive_ratio': np.mean(valid_scores > 0),
        'consistency': 1 - (np.std(valid_scores) / (np.mean(valid_scores) + 1e-8)),
        'n_samples': n_samples,
        'perturbation_pct': perturbation_pct
    }
