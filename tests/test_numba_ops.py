"""
Unit tests for numba operations.

Tests the Numba-accelerated calculation functions.
"""

import pytest
import numpy as np


class TestNumbaOps:
    """Tests for numba operations."""
    
    def test_calculate_rolling_zscore(self):
        """Test rolling z-score calculation."""
        from quantbox.utils.numba_ops import calculate_rolling_zscore
        
        prices = np.array([100.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0])
        lookback = 5
        
        zscore = calculate_rolling_zscore(prices, lookback)
        
        assert len(zscore) == len(prices)
        # First lookback-1 values should be NaN
        assert np.isnan(zscore[:lookback-1]).all()
        # Rest should be valid
        assert not np.isnan(zscore[lookback-1:]).any()
    
    def test_calculate_half_life(self):
        """Test half-life calculation."""
        from quantbox.utils.numba_ops import calculate_half_life
        
        # Create mean-reverting series
        np.random.seed(42)
        spread = np.cumsum(np.random.randn(100) - 0.1 * np.arange(100) * 0.01)
        
        half_life = calculate_half_life(spread)
        
        assert isinstance(half_life, (int, float))
        # Half-life should be positive if mean-reverting
    
    def test_calculate_hurst_exponent(self):
        """Test Hurst exponent calculation."""
        from quantbox.utils.numba_ops import calculate_hurst_exponent
        
        # Random walk
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200)) + 100
        
        hurst = calculate_hurst_exponent(prices)
        
        assert 0 <= hurst <= 1
    
    def test_ols_hedge_ratio(self):
        """Test OLS hedge ratio calculation."""
        from quantbox.utils.numba_ops import ols_hedge_ratio
        
        np.random.seed(42)
        x = np.cumsum(np.random.randn(100)) + 100
        y = 2.0 * x + np.random.randn(100) * 5 + 50  # y ≈ 2*x + noise
        
        # ols_hedge_ratio(y, x) returns (hedge_ratio, r_squared)
        hedge_ratio, r_squared = ols_hedge_ratio(y, x)
        
        # Hedge ratio should be close to 2.0
        assert 1.5 < hedge_ratio < 2.5
        # R-squared should be reasonably high
        assert r_squared > 0.7
    
    def test_kelly_criterion(self):
        """Test Kelly criterion calculation."""
        from quantbox.utils.numba_ops import kelly_criterion
        
        # Win rate 60%, win size 1.0, loss size 0.5
        kelly = kelly_criterion(
            win_rate=0.6,
            avg_win=1.0,
            avg_loss=0.5,
            fraction=0.5  # Half-Kelly
        )
        
        assert 0 <= kelly <= 1
    
    def test_optimal_position_sizing(self):
        """Test optimal position sizing."""
        from quantbox.utils.numba_ops import optimal_position_sizing
        
        # optimal_position_sizing(equity, volatility, target_vol, max_leverage)
        position = optimal_position_sizing(
            equity=10000.0,
            volatility=0.20,
            target_vol=0.20,
            max_leverage=3.0
        )
        
        # With target_vol == volatility, leverage should be 1x
        assert position == 10000.0  # equity * 1.0 leverage


class TestNumbaOpsEdgeCases:
    """Edge case tests for numba operations."""
    
    def test_zscore_single_value(self):
        """Test z-score with minimal data."""
        from quantbox.utils.numba_ops import calculate_rolling_zscore
        
        prices = np.array([100.0])
        zscore = calculate_rolling_zscore(prices, lookback=5)
        
        assert len(zscore) == 1
        assert np.isnan(zscore[0])
    
    def test_half_life_constant_series(self):
        """Test half-life with constant series."""
        from quantbox.utils.numba_ops import calculate_half_life
        
        spread = np.ones(100) * 10.0
        half_life = calculate_half_life(spread)
        
        # Should handle gracefully
        assert isinstance(half_life, (int, float))
    
    def test_hedge_ratio_identical_series(self):
        """Test hedge ratio with identical series."""
        from quantbox.utils.numba_ops import ols_hedge_ratio
        
        x = np.arange(100, dtype=float)
        y = x.copy()
        
        hedge_ratio, r_squared = ols_hedge_ratio(y, x)
        
        # Hedge ratio should be ~1.0
        assert 0.99 < hedge_ratio < 1.01
        # R-squared should be ~1.0
        assert r_squared > 0.99


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
