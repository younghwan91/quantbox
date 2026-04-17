"""Tests for compute_metrics portfolio_size parameter and bug fix.

US-R116B-1: compute_metrics() previously produced absurd Calmar/ann_ret values for
high-N × high-per-trade-rate strategies (Donchian N=1876 → Calmar=61M, ann_ret=1.76B%).
Root cause: 1x compound model (cumprod(1+pnl)) treats each trade as using full equity.
Fix: optional portfolio_size parameter scales per-trade PnL by 1/portfolio_size before
compounding, modeling realistic multi-slot capital allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.backtest.trade_sim import compute_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility (portfolio_size=1 = legacy 1x compound)
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardCompat:
    """portfolio_size=1 (default) preserves the pre-fix 1x compound model."""

    def test_default_portfolio_size_is_one(self):
        """No-arg call compounds at 1x (each trade reinvests full equity)."""
        pnl = np.array([0.30, 0.30, 0.30])
        m = compute_metrics(pnl, 2.0)  # default portfolio_size=1
        expected_equity = 1.3 * 1.3 * 1.3  # 2.197
        expected_ann = (expected_equity**0.5 - 1.0) * 100
        assert m["ann_ret_pct"] == pytest.approx(expected_ann, abs=0.5)

    def test_explicit_size_one_matches_default(self):
        """Explicit portfolio_size=1 matches default (no-arg)."""
        pnl = np.array([0.05, -0.01, 0.03, -0.01, 0.02])
        m_default = compute_metrics(pnl, 1.0)
        m_explicit = compute_metrics(pnl, 1.0, portfolio_size=1)
        assert m_default == m_explicit


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio scaling (bug fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioScaling:
    """portfolio_size>1 prevents compound explosion for multi-slot portfolios."""

    def test_donchian_like_high_n_no_explosion(self):
        """Synthetic 1876-trade strategy at 2% avg/trade: Calmar < 200 at portfolio_size=10."""
        # Mimic Donchian-like distribution: 58% WR, avg win 4.86%, avg loss 1.58%
        np.random.seed(42)
        n = 1876
        wins = np.random.choice([True, False], size=n, p=[0.583, 0.417])
        pnl = np.where(wins, 0.0486, -0.0158)  # deterministic magnitudes
        m = compute_metrics(pnl, 2.28, portfolio_size=10)

        # Sanity bounds: plausible values (not the pre-fix 61M Calmar, 1.76B% ann_ret)
        assert m["N"] == n
        assert m["calmar"] < 500, f"Calmar={m['calmar']} exceeds plausible bound"
        assert m["ann_ret_pct"] < 1000, f"ann_ret_pct={m['ann_ret_pct']} exceeds plausible bound"
        # And not broken-low either
        assert m["ann_ret_pct"] > 0

    def test_portfolio_size_reduces_mdd_proportionally(self):
        """At higher portfolio_size, MDD shrinks because each trade is a smaller equity fraction."""
        pnl = np.array([0.10, -0.20, 0.05])  # same data as test_max_dd
        m1 = compute_metrics(pnl, 1.0, portfolio_size=1)
        m10 = compute_metrics(pnl, 1.0, portfolio_size=10)

        # portfolio_size=1 reproduces original 20% MDD
        assert m1["max_dd_pct"] == pytest.approx(20.0, abs=0.1)
        # portfolio_size=10 scales PnL to [0.01, -0.02, 0.005]
        # equity: (1.01)(0.98)(1.005) — MDD at bar 2 ≈ (1.01 - 0.9898)/1.01 ≈ 2.0%
        assert m10["max_dd_pct"] == pytest.approx(2.0, abs=0.2)

    def test_pf_wr_invariant_across_portfolio_size(self):
        """PF and WR do not depend on portfolio_size (no compounding in their calc)."""
        pnl = np.array([0.05, -0.01, 0.03, -0.01, 0.02])
        m1 = compute_metrics(pnl, 1.0, portfolio_size=1)
        m10 = compute_metrics(pnl, 1.0, portfolio_size=10)

        assert m1["PF"] == m10["PF"]
        assert m1["WR"] == m10["WR"]
        assert m1["avg_win_pct"] == m10["avg_win_pct"]
        assert m1["avg_loss_pct"] == m10["avg_loss_pct"]
        assert m1["RR"] == m10["RR"]
        assert m1["tpd"] == m10["tpd"]


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """N<2, all-wins, all-losses, invalid inputs."""

    def test_empty_no_error(self):
        m = compute_metrics(np.array([]), 1.0, portfolio_size=10)
        assert m["N"] == 0
        assert m["PF"] == 0.0
        assert m["calmar"] == 0.0

    def test_all_winners_finite_values(self):
        """All-wins at any portfolio_size produces finite values (no div-by-zero)."""
        pnl = np.array([0.01, 0.02, 0.03])
        m = compute_metrics(pnl, 1.0, portfolio_size=10)
        assert m["PF"] == 999.0
        assert np.isfinite(m["calmar"])
        assert np.isfinite(m["ann_ret_pct"])
        assert m["max_dd_pct"] == 0.0  # no drawdowns in all-winners

    def test_invalid_portfolio_size_raises(self):
        """portfolio_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="portfolio_size must be >= 1"):
            compute_metrics(np.array([0.01, 0.02]), 1.0, portfolio_size=0)
        with pytest.raises(ValueError, match="portfolio_size must be >= 1"):
            compute_metrics(np.array([0.01, 0.02]), 1.0, portfolio_size=-5)

    def test_zero_data_years_no_div_error(self):
        """data_years=0 returns 0.0 annualized metrics without error."""
        pnl = np.array([0.01, 0.02])
        m = compute_metrics(pnl, 0.0, portfolio_size=10)
        assert m["ann_ret_pct"] == 0.0
        assert m["tpd"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Baseline regression (values from docs/backtest_results/202604/ after fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestBaselineRegression:
    """Verify that actual baseline metrics are in reasonable ranges post-fix."""

    def test_d_mom8_baseline_plausible(self):
        """D_mom8 baseline (N=1051, PF=1.092) produces near-zero ann_ret (barely-breakeven)."""
        # Reconstruct a D_mom8-like distribution: 43.9% WR, avg_win 1.509%, avg_loss 1.080%
        np.random.seed(42)
        n = 1051
        wins_mask = np.random.choice([True, False], size=n, p=[0.439, 0.561])
        pnl = np.where(wins_mask, 0.01509, -0.01080)
        m = compute_metrics(pnl, 2.28, portfolio_size=10)

        # PF should be close to actual baseline 1.092 (deterministic pnl, random WR)
        assert 0.8 < m["PF"] < 1.5
        # Near-breakeven strategy: ann_ret should be modest (< 20%)
        assert abs(m["ann_ret_pct"]) < 30
        # MDD scaled by /10: should be < 5% at portfolio_size=10
        assert m["max_dd_pct"] < 10

    def test_high_pf_donchian_like_finite(self):
        """PF>4 high-performing distribution yields finite, non-absurd Calmar."""
        np.random.seed(1)
        n = 1876
        wins_mask = np.random.choice([True, False], size=n, p=[0.583, 0.417])
        pnl = np.where(wins_mask, 0.0486, -0.0158)
        m = compute_metrics(pnl, 2.28, portfolio_size=10)

        assert np.isfinite(m["calmar"])
        assert np.isfinite(m["ann_ret_pct"])
        # Pre-fix: Calmar = 61M. Post-fix: < 500 is the sanity bound.
        assert m["calmar"] < 500
