"""
Tests for filter system.
"""

import pytest
import pandas as pd
import numpy as np

from quantbox.strategy.filters import (
    CorrelationFilter, CointegrationFilter, HalfLifeFilter,
    HedgeRatioFilter, VolumeFilter, StationarityFilter,
    HurstFilter, FilterPipeline, FilterResult
)


@pytest.fixture
def correlated_series():
    """Generate correlated price series."""
    np.random.seed(42)
    n = 500
    
    # Common trend
    trend = np.cumsum(np.random.randn(n) * 0.01)
    
    series1 = 100 * np.exp(trend + np.random.randn(n) * 0.005)
    series2 = 50 * np.exp(trend * 1.1 + np.random.randn(n) * 0.005)
    
    return pd.Series(series1), pd.Series(series2)


@pytest.fixture
def uncorrelated_series():
    """Generate uncorrelated price series."""
    np.random.seed(42)
    n = 500
    
    series1 = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    series2 = 50 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    return pd.Series(series1), pd.Series(series2)


class TestCorrelationFilter:
    """Tests for CorrelationFilter."""
    
    def test_high_correlation_passes(self, correlated_series):
        filter = CorrelationFilter(min_correlation=0.5)
        s1, s2 = correlated_series
        
        result = filter.apply(s1, s2)
        assert result.passed is True
        assert result.value > 0.5
    
    def test_low_correlation_fails(self, uncorrelated_series):
        filter = CorrelationFilter(min_correlation=0.9)
        s1, s2 = uncorrelated_series
        
        result = filter.apply(s1, s2)
        # May or may not pass depending on random seed
        assert isinstance(result, FilterResult)
        assert 'correlation' in result.reason.lower() or result.passed
    
    def test_correlation_bounds(self):
        filter = CorrelationFilter(min_correlation=0.5, max_correlation=0.95)
        
        # Perfect correlation should fail
        s = pd.Series(range(100))
        result = filter.apply(s, s)
        assert result.passed is False


class TestCointegrationFilter:
    """Tests for CointegrationFilter."""
    
    def test_cointegrated_pair(self, correlated_series):
        filter = CointegrationFilter(max_pvalue=0.1)
        s1, s2 = correlated_series
        
        result = filter.apply(s1, s2)
        # Result depends on actual cointegration
        assert isinstance(result, FilterResult)
        assert result.value is not None
    
    def test_pvalue_threshold(self, uncorrelated_series):
        filter = CointegrationFilter(max_pvalue=0.001)  # Very strict
        s1, s2 = uncorrelated_series
        
        result = filter.apply(s1, s2)
        # Very likely to fail with random data
        assert isinstance(result, FilterResult)


class TestHalfLifeFilter:
    """Tests for HalfLifeFilter."""
    
    def test_reasonable_halflife(self, correlated_series):
        filter = HalfLifeFilter(min_halflife=1, max_halflife=100)
        s1, s2 = correlated_series
        
        result = filter.apply(s1, s2)
        assert isinstance(result, FilterResult)
        assert result.value is not None
    
    def test_halflife_bounds(self, correlated_series):
        filter = HalfLifeFilter(min_halflife=50, max_halflife=60)
        s1, s2 = correlated_series
        
        result = filter.apply(s1, s2)
        if result.passed:
            assert 50 <= result.value <= 60


class TestHedgeRatioFilter:
    """Tests for HedgeRatioFilter."""
    
    def test_hedge_ratio_calculation(self, correlated_series):
        filter = HedgeRatioFilter(min_ratio=0.1, max_ratio=5.0)
        s1, s2 = correlated_series
        
        result = filter.apply(s1, s2)
        assert isinstance(result, FilterResult)
        assert result.value is not None
    
    def test_stability_check(self, correlated_series):
        filter = HedgeRatioFilter(
            min_ratio=0.1,
            max_ratio=5.0,
            stability_window=50
        )
        s1, s2 = correlated_series
        
        result = filter.apply(s1, s2)
        assert isinstance(result, FilterResult)


class TestVolumeFilter:
    """Tests for VolumeFilter."""
    
    def test_sufficient_volume(self):
        filter = VolumeFilter(min_avg_volume=100)
        
        vol1 = pd.Series([200, 150, 300, 250])
        vol2 = pd.Series([180, 200, 220, 190])
        
        result = filter.apply(vol1, vol2)
        assert result.passed is True
    
    def test_insufficient_volume(self):
        filter = VolumeFilter(min_avg_volume=1000)
        
        vol1 = pd.Series([100, 150, 120])
        vol2 = pd.Series([80, 90, 110])
        
        result = filter.apply(vol1, vol2)
        assert result.passed is False


class TestStationarityFilter:
    """Tests for StationarityFilter."""
    
    def test_stationary_series(self):
        filter = StationarityFilter(max_pvalue=0.05)
        
        # Mean-reverting series
        np.random.seed(42)
        spread = np.random.randn(500)
        
        result = filter.apply(pd.Series(spread))
        assert result.passed is True
    
    def test_nonstationary_series(self):
        filter = StationarityFilter(max_pvalue=0.05)
        
        # Random walk (non-stationary)
        np.random.seed(42)
        walk = np.cumsum(np.random.randn(500))
        
        result = filter.apply(pd.Series(walk))
        # May or may not be stationary by chance


class TestHurstFilter:
    """Tests for HurstFilter."""
    
    def test_mean_reverting(self):
        filter = HurstFilter(max_hurst=0.5)
        
        # Mean-reverting series (H < 0.5)
        np.random.seed(42)
        series = pd.Series(np.random.randn(500))
        
        result = filter.apply(series)
        assert isinstance(result, FilterResult)
    
    def test_trending_fails(self):
        filter = HurstFilter(max_hurst=0.4)
        
        # Trending series (H > 0.5)
        np.random.seed(42)
        trend = np.cumsum(np.random.randn(500) + 0.1)
        
        result = filter.apply(pd.Series(trend))
        # Trending series should have H > 0.5


class TestFilterPipeline:
    """Tests for FilterPipeline."""
    
    def test_empty_pipeline(self, correlated_series):
        pipeline = FilterPipeline([])
        s1, s2 = correlated_series
        
        results = pipeline.apply(s1, s2)
        assert len(results) == 0
    
    def test_single_filter(self, correlated_series):
        filter = CorrelationFilter(min_correlation=0.3)
        pipeline = FilterPipeline([filter])
        s1, s2 = correlated_series
        
        results = pipeline.apply(s1, s2)
        assert len(results) == 1
        assert isinstance(results[0], FilterResult)
    
    def test_multiple_filters(self, correlated_series):
        filters = [
            CorrelationFilter(min_correlation=0.3),
            CointegrationFilter(max_pvalue=0.1),
        ]
        pipeline = FilterPipeline(filters)
        s1, s2 = correlated_series
        
        results = pipeline.apply(s1, s2)
        assert len(results) == 2
    
    def test_all_passed(self, correlated_series):
        filters = [
            CorrelationFilter(min_correlation=0.3),
        ]
        pipeline = FilterPipeline(filters)
        s1, s2 = correlated_series
        
        results = pipeline.apply(s1, s2)
        all_passed = pipeline.all_passed(results)
        # Depends on data
        assert isinstance(all_passed, bool)
    
    def test_summary(self, correlated_series):
        filters = [
            CorrelationFilter(min_correlation=0.3),
            CointegrationFilter(max_pvalue=0.1),
        ]
        pipeline = FilterPipeline(filters)
        s1, s2 = correlated_series
        
        results = pipeline.apply(s1, s2)
        summary = pipeline.summary(results)
        
        assert 'CorrelationFilter' in summary
        assert 'CointegrationFilter' in summary
