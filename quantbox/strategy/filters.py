"""
Flexible filter system for pair selection.

Allows composable filters that can be combined to create
custom pair selection criteria.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging
from scipy import stats


logger = logging.getLogger('Filters')


@dataclass
class FilterResult:
    """Result of a filter operation."""
    passed: bool
    score: float = 0.0
    reason: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseFilter(ABC):
    """
    Base class for pair filters.
    
    All filters should inherit from this and implement
    the apply() method.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        """
        Apply filter to a pair of series.
        
        Args:
            series1: First price series
            series2: Second price series
            **kwargs: Additional parameters
            
        Returns:
            FilterResult with pass/fail and score
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self.enabled})"


class CorrelationFilter(BaseFilter):
    """Filter pairs by price correlation."""
    
    def __init__(
        self,
        min_correlation: float = 0.5,
        max_correlation: float = 0.99,
        use_returns: bool = True,
        enabled: bool = True
    ):
        super().__init__("correlation", enabled)
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.use_returns = use_returns
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        if self.use_returns:
            s1 = np.log(series1).diff().dropna()
            s2 = np.log(series2).diff().dropna()
        else:
            s1 = series1
            s2 = series2
        
        # Align series
        common_idx = s1.index.intersection(s2.index)
        s1 = s1.loc[common_idx]
        s2 = s2.loc[common_idx]
        
        correlation = np.corrcoef(s1, s2)[0, 1]
        
        passed = self.min_correlation <= correlation <= self.max_correlation
        score = correlation if passed else 0.0
        
        return FilterResult(
            passed=passed,
            score=score,
            reason=f"Correlation: {correlation:.4f}" if passed else 
                   f"Correlation {correlation:.4f} outside [{self.min_correlation}, {self.max_correlation}]",
            metadata={'correlation': correlation}
        )


class CointegrationFilter(BaseFilter):
    """Filter pairs by cointegration test."""
    
    def __init__(
        self,
        max_pvalue: float = 0.05,
        enabled: bool = True
    ):
        super().__init__("cointegration", enabled)
        self.max_pvalue = max_pvalue
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        try:
            from statsmodels.tsa.stattools import coint
            
            log1 = np.log(series1)
            log2 = np.log(series2)
            
            t_stat, p_value, _ = coint(log1, log2)
            
            passed = p_value < self.max_pvalue
            score = 1 - p_value if passed else 0.0
            
            return FilterResult(
                passed=passed,
                score=score,
                reason=f"Cointegration p-value: {p_value:.4f}" if passed else
                       f"P-value {p_value:.4f} > {self.max_pvalue}",
                metadata={'t_stat': t_stat, 'p_value': p_value}
            )
        
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
            return FilterResult(False, 0.0, f"Test failed: {e}")


class HalfLifeFilter(BaseFilter):
    """Filter pairs by mean-reversion half-life."""
    
    def __init__(
        self,
        min_half_life: float = 1.0,
        max_half_life: float = 100.0,
        enabled: bool = True
    ):
        super().__init__("half_life", enabled)
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        try:
            # Calculate spread
            log1 = np.log(series1)
            log2 = np.log(series2)
            
            # OLS hedge ratio
            hedge_ratio = np.polyfit(log2, log1, 1)[0]
            spread = log1 - hedge_ratio * log2
            
            # Calculate half-life
            half_life = self._calculate_half_life(spread)
            
            passed = self.min_half_life <= half_life <= self.max_half_life
            
            # Score: prefer medium half-lives
            if passed:
                optimal = (self.min_half_life + self.max_half_life) / 2
                score = 1 - abs(half_life - optimal) / (self.max_half_life - self.min_half_life)
            else:
                score = 0.0
            
            return FilterResult(
                passed=passed,
                score=score,
                reason=f"Half-life: {half_life:.2f}" if passed else
                       f"Half-life {half_life:.2f} outside [{self.min_half_life}, {self.max_half_life}]",
                metadata={'half_life': half_life, 'hedge_ratio': hedge_ratio}
            )
        
        except Exception as e:
            logger.warning(f"Half-life calculation failed: {e}")
            return FilterResult(False, 0.0, f"Calculation failed: {e}")
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of spread."""
        import statsmodels.api as sm
        
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        
        spread_lag = spread_lag.iloc[1:]
        spread_ret = spread_ret.iloc[1:]
        
        model = sm.OLS(spread_ret, sm.add_constant(spread_lag))
        result = model.fit()
        
        lambda_param = result.params.iloc[1]
        
        if lambda_param >= 0:
            return np.inf
        
        return -np.log(2) / lambda_param


class HedgeRatioFilter(BaseFilter):
    """Filter pairs by hedge ratio bounds."""
    
    def __init__(
        self,
        min_ratio: float = 0.25,
        max_ratio: float = 4.0,
        enabled: bool = True
    ):
        super().__init__("hedge_ratio", enabled)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        log1 = np.log(series1)
        log2 = np.log(series2)
        
        hedge_ratio = np.polyfit(log2, log1, 1)[0]
        
        passed = self.min_ratio <= hedge_ratio <= self.max_ratio
        
        # Score: prefer hedge ratios closer to 1
        if passed:
            score = 1 - abs(np.log(hedge_ratio)) / np.log(self.max_ratio)
        else:
            score = 0.0
        
        return FilterResult(
            passed=passed,
            score=score,
            reason=f"Hedge ratio: {hedge_ratio:.4f}" if passed else
                   f"Hedge ratio {hedge_ratio:.4f} outside [{self.min_ratio}, {self.max_ratio}]",
            metadata={'hedge_ratio': hedge_ratio}
        )


class NormalityFilter(BaseFilter):
    """Filter pairs by spread normality (Shapiro-Wilk or Jarque-Bera)."""
    
    def __init__(
        self,
        min_pvalue: float = 0.05,
        test: str = 'jarque_bera',  # 'shapiro' or 'jarque_bera'
        enabled: bool = True
    ):
        super().__init__("normality", enabled)
        self.min_pvalue = min_pvalue
        self.test = test
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        try:
            # Calculate spread
            log1 = np.log(series1)
            log2 = np.log(series2)
            hedge_ratio = np.polyfit(log2, log1, 1)[0]
            spread = log1 - hedge_ratio * log2
            
            # Standardize
            z_spread = (spread - spread.mean()) / spread.std()
            
            # Test normality
            if self.test == 'shapiro':
                # Shapiro-Wilk (limited to 5000 samples)
                sample = z_spread.values[:5000]
                stat, p_value = stats.shapiro(sample)
            else:
                # Jarque-Bera
                stat, p_value = stats.jarque_bera(z_spread.values)
            
            passed = p_value > self.min_pvalue
            score = p_value if passed else 0.0
            
            return FilterResult(
                passed=passed,
                score=score,
                reason=f"{self.test} p-value: {p_value:.4f}" if passed else
                       f"Spread not normal (p={p_value:.4f})",
                metadata={'test': self.test, 'statistic': stat, 'p_value': p_value}
            )
        
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return FilterResult(False, 0.0, f"Test failed: {e}")


class VolumeFilter(BaseFilter):
    """Filter pairs by minimum trading volume."""
    
    def __init__(
        self,
        min_volume: float = 1e6,  # Minimum 24h volume in quote currency
        enabled: bool = True
    ):
        super().__init__("volume", enabled)
        self.min_volume = min_volume
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        volume1: Optional[pd.Series] = None,
        volume2: Optional[pd.Series] = None,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        if volume1 is None or volume2 is None:
            return FilterResult(True, 0.5, "Volume data not provided")
        
        avg_vol1 = volume1.mean()
        avg_vol2 = volume2.mean()
        min_vol = min(avg_vol1, avg_vol2)
        
        passed = min_vol >= self.min_volume
        
        if passed:
            score = min(1.0, min_vol / (self.min_volume * 10))
        else:
            score = 0.0
        
        return FilterResult(
            passed=passed,
            score=score,
            reason=f"Min volume: {min_vol:,.0f}" if passed else
                   f"Volume {min_vol:,.0f} < {self.min_volume:,.0f}",
            metadata={'volume1': avg_vol1, 'volume2': avg_vol2}
        )


class StationarityFilter(BaseFilter):
    """Filter pairs by spread stationarity (ADF test)."""
    
    def __init__(
        self,
        max_pvalue: float = 0.05,
        enabled: bool = True
    ):
        super().__init__("stationarity", enabled)
        self.max_pvalue = max_pvalue
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Calculate spread
            log1 = np.log(series1)
            log2 = np.log(series2)
            hedge_ratio = np.polyfit(log2, log1, 1)[0]
            spread = log1 - hedge_ratio * log2
            
            # ADF test
            result = adfuller(spread.dropna())
            adf_stat = result[0]
            p_value = result[1]
            
            passed = p_value < self.max_pvalue
            score = 1 - p_value if passed else 0.0
            
            return FilterResult(
                passed=passed,
                score=score,
                reason=f"ADF p-value: {p_value:.4f}" if passed else
                       f"Spread not stationary (p={p_value:.4f})",
                metadata={'adf_stat': adf_stat, 'p_value': p_value}
            )
        
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return FilterResult(False, 0.0, f"Test failed: {e}")


class HurstFilter(BaseFilter):
    """Filter pairs by Hurst exponent (mean-reversion indicator)."""
    
    def __init__(
        self,
        max_hurst: float = 0.5,  # < 0.5 indicates mean-reversion
        enabled: bool = True
    ):
        super().__init__("hurst", enabled)
        self.max_hurst = max_hurst
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, 1.0, "Filter disabled")
        
        try:
            # Calculate spread
            log1 = np.log(series1)
            log2 = np.log(series2)
            hedge_ratio = np.polyfit(log2, log1, 1)[0]
            spread = log1 - hedge_ratio * log2
            
            # Calculate Hurst exponent
            hurst = self._calculate_hurst(spread.values)
            
            passed = hurst < self.max_hurst
            
            # Score: lower Hurst = better mean-reversion
            if passed:
                score = (self.max_hurst - hurst) / self.max_hurst
            else:
                score = 0.0
            
            return FilterResult(
                passed=passed,
                score=score,
                reason=f"Hurst exponent: {hurst:.4f}" if passed else
                       f"Hurst {hurst:.4f} >= {self.max_hurst} (not mean-reverting)",
                metadata={'hurst': hurst}
            )
        
        except Exception as e:
            logger.warning(f"Hurst calculation failed: {e}")
            return FilterResult(False, 0.0, f"Calculation failed: {e}")
    
    def _calculate_hurst(self, ts: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        lags = range(2, min(max_lag, len(ts) // 4))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] / 2.0


class FilterPipeline:
    """
    Pipeline for combining multiple filters.
    
    Filters are applied in order and a pair must pass all
    enabled filters to be selected.
    """
    
    def __init__(self, filters: List[BaseFilter] = None):
        self.filters = filters or []
        self._results_cache: Dict[tuple, Dict] = {}
    
    def add_filter(self, filter: BaseFilter):
        """Add a filter to the pipeline."""
        self.filters.append(filter)
        return self
    
    def remove_filter(self, name: str):
        """Remove a filter by name."""
        self.filters = [f for f in self.filters if f.name != name]
        return self
    
    def enable_filter(self, name: str):
        """Enable a filter by name."""
        for f in self.filters:
            if f.name == name:
                f.enabled = True
        return self
    
    def disable_filter(self, name: str):
        """Disable a filter by name."""
        for f in self.filters:
            if f.name == name:
                f.enabled = False
        return self
    
    def apply(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> Tuple[bool, float, Dict[str, FilterResult]]:
        """
        Apply all filters to a pair.
        
        Args:
            series1: First price series
            series2: Second price series
            **kwargs: Additional parameters passed to filters
            
        Returns:
            Tuple of (passed_all, combined_score, results_dict)
        """
        results = {}
        passed_all = True
        total_score = 0.0
        num_enabled = 0
        
        for filter in self.filters:
            if not filter.enabled:
                continue
            
            result = filter.apply(series1, series2, **kwargs)
            results[filter.name] = result
            
            if not result.passed:
                passed_all = False
            
            total_score += result.score
            num_enabled += 1
        
        # Average score
        combined_score = total_score / num_enabled if num_enabled > 0 else 0.0
        
        return passed_all, combined_score, results
    
    def get_failing_filters(
        self,
        series1: pd.Series,
        series2: pd.Series,
        **kwargs
    ) -> List[str]:
        """Get list of filter names that failed."""
        _, _, results = self.apply(series1, series2, **kwargs)
        return [name for name, result in results.items() if not result.passed]
    
    def __repr__(self):
        filters_str = ", ".join(f.name for f in self.filters if f.enabled)
        return f"FilterPipeline([{filters_str}])"


def create_default_pipeline() -> FilterPipeline:
    """Create a default filter pipeline for pair trading."""
    return FilterPipeline([
        CorrelationFilter(min_correlation=0.5, max_correlation=0.99),
        CointegrationFilter(max_pvalue=0.05),
        HalfLifeFilter(min_half_life=1.0, max_half_life=100.0),
        HedgeRatioFilter(min_ratio=0.25, max_ratio=4.0),
        StationarityFilter(max_pvalue=0.05),
    ])


def create_strict_pipeline() -> FilterPipeline:
    """Create a strict filter pipeline with all filters."""
    return FilterPipeline([
        CorrelationFilter(min_correlation=0.6, max_correlation=0.95),
        CointegrationFilter(max_pvalue=0.01),
        HalfLifeFilter(min_half_life=2.0, max_half_life=50.0),
        HedgeRatioFilter(min_ratio=0.33, max_ratio=3.0),
        NormalityFilter(min_pvalue=0.05),
        StationarityFilter(max_pvalue=0.01),
        HurstFilter(max_hurst=0.45),
    ])
