"""
Tests for strategy filters.
"""

import pytest
import numpy as np
import pandas as pd

from quantbox.strategy.filters import (
    FilterResult, BaseFilter, CorrelationFilter, CointegrationFilter,
    HalfLifeFilter, HedgeRatioFilter, VolumeFilter, StationarityFilter,
    HurstFilter, FilterPipeline
)


# Fixture: create correlated price series
@pytest.fixture
def correlated_series():
    np.random.seed(42)
    n = 200
    base = np.cumsum(np.random.randn(n)) + 100
    noise = np.random.randn(n) * 0.5
    
    series1 = pd.Series(np.abs(base + noise), name="asset1")
    series2 = pd.Series(np.abs(base * 0.5 + 50 + noise * 0.3), name="asset2")
    return series1, series2


@pytest.fixture
def uncorrelated_series():
    np.random.seed(42)
    n = 200
    series1 = pd.Series(np.abs(np.cumsum(np.random.randn(n)) + 100))
    series2 = pd.Series(np.abs(np.cumsum(np.random.randn(n)) + 100))
    return series1, series2


class TestFilterResult:
    """Tests for FilterResult dataclass."""
    
    def test_creation(self):
        result = FilterResult(passed=True, score=0.85, reason="Test passed")
        assert result.passed is True
        assert result.score == 0.85
        assert result.reason == "Test passed"
    
    def test_default_metadata(self):
        result = FilterResult(passed=True, score=1.0)
        assert result.metadata == {}


class TestCorrelationFilter:
    """Tests for CorrelationFilter."""
    
    def test_high_correlation_passes(self, correlated_series):
        series1, series2 = correlated_series
        filter = CorrelationFilter(min_correlation=0.5, max_correlation=0.99)
        
        result = filter.apply(series1, series2)
        assert result.passed == True
        assert result.score > 0.5
    
    def test_low_correlation_fails(self, uncorrelated_series):
        series1, series2 = uncorrelated_series
        filter = CorrelationFilter(min_correlation=0.9)
        
        result = filter.apply(series1, series2)
        # Uncorrelated series should fail high correlation threshold
        assert result.score < 0.9
    
    def test_filter_disabled(self, correlated_series):
        series1, series2 = correlated_series
        filter = CorrelationFilter(enabled=False)
        
        result = filter.apply(series1, series2)
        assert result.passed == True
        assert "disabled" in result.reason.lower()


class TestCointegrationFilter:
    """Tests for CointegrationFilter."""
    
    def test_cointegrated_pair(self, correlated_series):
        series1, series2 = correlated_series
        filter = CointegrationFilter(max_pvalue=0.1)
        
        result = filter.apply(series1, series2)
        # Correlated series may or may not be cointegrated
        assert 'p_value' in result.metadata or 'error' in str(result.reason).lower()
    
    def test_filter_disabled(self, correlated_series):
        series1, series2 = correlated_series
        filter = CointegrationFilter(enabled=False)
        
        result = filter.apply(series1, series2)
        assert result.passed == True


class TestHalfLifeFilter:
    """Tests for HalfLifeFilter."""
    
    def test_filter_creation(self):
        filter = HalfLifeFilter(min_half_life=1, max_half_life=100)
        assert filter.min_half_life == 1
        assert filter.max_half_life == 100
    
    def test_filter_apply(self, correlated_series):
        series1, series2 = correlated_series
        filter = HalfLifeFilter(min_half_life=1, max_half_life=200)
        
        result = filter.apply(series1, series2)
        assert isinstance(result, FilterResult)
    
    def test_filter_disabled(self, correlated_series):
        series1, series2 = correlated_series
        filter = HalfLifeFilter(enabled=False)
        
        result = filter.apply(series1, series2)
        assert result.passed == True


class TestHedgeRatioFilter:
    """Tests for HedgeRatioFilter."""
    
    def test_filter_creation(self):
        filter = HedgeRatioFilter(min_ratio=0.25, max_ratio=4.0)
        assert filter.min_ratio == 0.25
        assert filter.max_ratio == 4.0
    
    def test_filter_apply(self, correlated_series):
        series1, series2 = correlated_series
        filter = HedgeRatioFilter(min_ratio=0.1, max_ratio=10.0)
        
        result = filter.apply(series1, series2)
        assert isinstance(result, FilterResult)
        assert 'hedge_ratio' in result.metadata


class TestVolumeFilter:
    """Tests for VolumeFilter."""
    
    def test_filter_creation(self):
        filter = VolumeFilter(min_volume=1000000)
        assert filter.min_volume == 1000000
    
    def test_sufficient_volume(self):
        filter = VolumeFilter(min_volume=100)
        
        series1 = pd.Series([100] * 50)
        series2 = pd.Series([100] * 50)
        volume1 = pd.Series([1000] * 50)
        volume2 = pd.Series([1000] * 50)
        
        result = filter.apply(series1, series2, volume1=volume1, volume2=volume2)
        assert result.passed == True
    
    def test_insufficient_volume(self):
        filter = VolumeFilter(min_volume=10000)
        
        series1 = pd.Series([100] * 50)
        series2 = pd.Series([100] * 50)
        volume1 = pd.Series([100] * 50)
        volume2 = pd.Series([100] * 50)
        
        result = filter.apply(series1, series2, volume1=volume1, volume2=volume2)
        assert result.passed == False


class TestStationarityFilter:
    """Tests for StationarityFilter."""
    
    def test_filter_creation(self):
        filter = StationarityFilter(max_pvalue=0.05)
        assert filter.max_pvalue == 0.05
    
    def test_filter_apply(self, correlated_series):
        series1, series2 = correlated_series
        filter = StationarityFilter(max_pvalue=0.1)
        
        result = filter.apply(series1, series2)
        assert isinstance(result, FilterResult)


class TestHurstFilter:
    """Tests for HurstFilter."""
    
    def test_filter_creation(self):
        filter = HurstFilter(max_hurst=0.5)
        assert filter.max_hurst == 0.5
    
    def test_mean_reverting_series(self, correlated_series):
        series1, series2 = correlated_series
        filter = HurstFilter(max_hurst=0.6)
        
        result = filter.apply(series1, series2)
        assert isinstance(result, FilterResult)


class TestFilterPipeline:
    """Tests for FilterPipeline."""
    
    def test_empty_pipeline(self, correlated_series):
        series1, series2 = correlated_series
        pipeline = FilterPipeline([])
        
        passed, score, results = pipeline.apply(series1, series2)
        assert passed is True  # Empty pipeline passes
    
    def test_single_filter(self, correlated_series):
        series1, series2 = correlated_series
        pipeline = FilterPipeline([
            CorrelationFilter(min_correlation=0.3)
        ])
        
        passed, score, results = pipeline.apply(series1, series2)
        assert isinstance(passed, (bool, np.bool_))
        assert 'correlation' in results
    
    def test_multiple_filters(self, correlated_series):
        series1, series2 = correlated_series
        pipeline = FilterPipeline([
            CorrelationFilter(min_correlation=0.3),
            CointegrationFilter(max_pvalue=0.1),
        ])
        
        passed, score, results = pipeline.apply(series1, series2)
        assert isinstance(passed, (bool, np.bool_))
        assert len(results) >= 1
    
    def test_add_filter(self, correlated_series):
        series1, series2 = correlated_series
        pipeline = FilterPipeline([])
        pipeline.add_filter(CorrelationFilter(min_correlation=0.3))
        
        assert len(pipeline.filters) == 1
