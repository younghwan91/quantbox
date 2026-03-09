"""
Tests for vol_selector — Volatility-based symbol selection.

Tests the core vol ranking module that drives production symbol selection.
"""

import numpy as np
import pytest

from quantbox.strategy.vol_selector import (
    compute_realized_vol,
    rank_symbols_by_vol,
    VolatilitySelector,
    VolSelectorConfig,
)


# ── compute_realized_vol ──


class TestComputeRealizedVol:
    """Tests for the Numba-compiled realized vol function."""

    def test_constant_prices_zero_vol(self):
        """Constant prices → zero volatility."""
        close = np.full(100, 100.0)
        vol = compute_realized_vol(close, 187.08)
        assert vol == 0.0

    def test_trending_prices_nonzero_vol(self):
        """Consistently moving prices → positive volatility."""
        # 1% daily drift at 15m bars (96 per day)
        close = 100.0 * np.cumprod(1.0 + np.random.RandomState(42).normal(0.0001, 0.001, 1000))
        vol = compute_realized_vol(close, np.sqrt(96 * 365.25))
        assert vol > 0.0

    def test_higher_noise_higher_vol(self):
        """More volatile prices → higher realized vol."""
        rng = np.random.RandomState(42)
        low_vol = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.001, 500))
        high_vol = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 500))
        ann = np.sqrt(96 * 365.25)
        assert compute_realized_vol(high_vol, ann) > compute_realized_vol(low_vol, ann)

    def test_single_element_returns_zero(self):
        """Edge case: single price → 0."""
        assert compute_realized_vol(np.array([100.0]), 187.08) == 0.0

    def test_two_elements(self):
        """Two prices: single return → zero variance (std of 1 value = 0)."""
        close = np.array([100.0, 105.0])
        vol = compute_realized_vol(close, 187.08)
        # Single return: std of [x] around mean x is 0
        assert vol == 0.0

    def test_three_elements_nonzero(self):
        """Three prices with different returns → nonzero vol."""
        close = np.array([100.0, 105.0, 102.0])
        vol = compute_realized_vol(close, 187.08)
        assert vol > 0.0


# ── rank_symbols_by_vol ──


class TestRankSymbolsByVol:
    """Tests for symbol ranking by volatility."""

    def test_ranking_order(self):
        """Most volatile symbol should rank first."""
        rng = np.random.RandomState(42)
        n = 500
        symbols = {
            "STABLE": np.full(n, 1.0),  # Zero vol
            "LOW": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.001, n)),
            "HIGH": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n)),
        }
        ranked = rank_symbols_by_vol(symbols, top_n=3)
        assert ranked[0][0] == "HIGH"
        assert ranked[-1][0] == "STABLE"

    def test_top_n_limits(self):
        """Only returns requested number of symbols."""
        symbols = {f"SYM{i}": np.full(100, float(i + 1)) for i in range(10)}
        ranked = rank_symbols_by_vol(symbols, top_n=3)
        assert len(ranked) == 3

    def test_top_n_larger_than_universe(self):
        """Requesting more than available returns all."""
        symbols = {"A": np.ones(100), "B": np.ones(100)}
        ranked = rank_symbols_by_vol(symbols, top_n=10)
        assert len(ranked) == 2

    def test_lookback_uses_recent_data(self):
        """Lookback window should use most recent bars."""
        rng = np.random.RandomState(42)
        # First 500 bars: A volatile, B stable
        a_first = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 500))
        b_first = np.full(500, 100.0)
        # Last 500 bars: B volatile, A stable
        a_last = np.full(500, a_first[-1])
        b_last = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 500))

        symbols = {
            "A": np.concatenate([a_first, a_last]),
            "B": np.concatenate([b_first, b_last]),
        }
        # Using only last 500 bars → B should rank higher
        ranked = rank_symbols_by_vol(symbols, top_n=2, lookback_bars=500)
        assert ranked[0][0] == "B"


# ── VolatilitySelector ──


class TestVolatilitySelector:
    """Tests for the stateful selector class."""

    def test_should_rebalance_initial(self):
        """First rebalance after warmup period."""
        sel = VolatilitySelector(VolSelectorConfig(vol_lookback_bars=100))
        assert not sel.should_rebalance(50)
        assert sel.should_rebalance(100)

    def test_should_rebalance_interval(self):
        """Rebalance triggers after configured interval."""
        cfg = VolSelectorConfig(vol_lookback_bars=100, rebalance_interval_bars=50)
        sel = VolatilitySelector(cfg)

        symbols = {"A": np.ones(200), "B": np.ones(200)}
        sel.rebalance(symbols, current_bar=100)

        assert not sel.should_rebalance(120)
        assert sel.should_rebalance(150)

    def test_rebalance_returns_symbols(self):
        """Rebalance returns the top-N selected symbols."""
        rng = np.random.RandomState(42)
        cfg = VolSelectorConfig(top_n=2, vol_lookback_bars=100)
        sel = VolatilitySelector(cfg)

        symbols = {
            "LOW": np.full(200, 1.0),
            "MED": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.005, 200)),
            "HIGH": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 200)),
        }
        result = sel.rebalance(symbols, current_bar=200)
        assert len(result) == 2
        assert "HIGH" in result
        assert "LOW" not in result

    def test_selected_symbols_property(self):
        """selected_symbols returns current selection."""
        sel = VolatilitySelector(VolSelectorConfig(top_n=1))
        symbols = {
            "A": np.full(200, 1.0),
            "B": 100.0 * np.cumprod(np.ones(200) + 0.01),
        }
        sel.rebalance(symbols)
        assert len(sel.selected_symbols) == 1

    def test_is_selected(self):
        """is_selected checks membership correctly."""
        rng = np.random.RandomState(42)
        sel = VolatilitySelector(VolSelectorConfig(top_n=1))
        symbols = {
            "STABLE": np.full(200, 100.0),
            "VOLATILE": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 200)),
        }
        sel.rebalance(symbols)
        assert sel.is_selected("VOLATILE")
        assert not sel.is_selected("STABLE")

    def test_rebalance_count(self):
        """Tracks number of rebalances."""
        sel = VolatilitySelector()
        assert sel.rebalance_count == 0
        sel.rebalance({"A": np.ones(100)})
        assert sel.rebalance_count == 1
        sel.rebalance({"A": np.ones(100)})
        assert sel.rebalance_count == 2


# ── Integration with real data ──


class TestVolSelectorRealData:
    """Integration tests using cached Binance data."""

    def test_real_data_ranking(self, real_data_available):
        """Vol ranking works on real data."""
        if not real_data_available:
            pytest.skip("Real data not available")

        from quantbox.data.real_data_loader import RealDataLoader

        loader = RealDataLoader()
        all_symbols = loader.get_available_symbols()
        if len(all_symbols) < 5:
            pytest.skip("Need at least 5 symbols")

        symbol_closes = {}
        for sym in all_symbols[:10]:
            df = loader.load_symbol(sym)
            if df is not None and "Close" in df.columns:
                symbol_closes[sym] = df["Close"].values.astype(np.float64)

        ranked = rank_symbols_by_vol(symbol_closes, top_n=5)
        assert len(ranked) <= 5
        assert all(isinstance(s, str) for s, _ in ranked)
        # Vols should be descending
        vols = [v for _, v in ranked]
        assert vols == sorted(vols, reverse=True)
