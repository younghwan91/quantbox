"""Tests for r61 SMA-distance and squeeze-duration analysis.

Covers:
- compute_entry_attrs: SMA distance and VR depth calculation
- bucket_analysis: correct grouping and WR computation
- threshold_sweep: "above" and "below" modes
- wr_stats: basic stats on known data
- Data integrity: 60 symbols loaded
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.active.r51_r60.r61_sma_filter import (
    bucket_analysis,
    compute_entry_attrs,
    evaluate_with_portfolio_sim,
    threshold_sweep,
    wr_stats,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_close(n: int = 100, base: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    ret = rng.normal(0, 0.005, n)
    c = base * np.cumprod(1 + ret)
    return c.astype(np.float64)


def _make_vr(n: int = 100, val: float = 0.4) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


# ── wr_stats ─────────────────────────────────────────────────────────────────


class TestWrStats:
    def test_empty(self):
        wr, pf, ev = wr_stats([])
        assert wr == 0.0
        assert pf == 0.0
        assert ev == 0.0

    def test_all_wins(self):
        wr, pf, ev = wr_stats([0.01, 0.02, 0.03])
        assert wr == 1.0
        assert pf == float("inf")
        assert ev > 0

    def test_all_losses(self):
        wr, pf, ev = wr_stats([-0.01, -0.02])
        assert wr == 0.0
        assert pf == 0.0
        assert ev < 0

    def test_mixed(self):
        pnls = [0.05, -0.01, 0.03, -0.01]  # 2 wins, 2 losses
        wr, pf, ev = wr_stats(pnls)
        assert wr == pytest.approx(0.5)
        assert pf == pytest.approx((0.05 + 0.03) / (0.01 + 0.01))
        assert ev == pytest.approx(np.mean(pnls) * 10000)

    def test_returns_float(self):
        wr, pf, ev = wr_stats([0.01, -0.01])
        assert isinstance(wr, float)
        assert isinstance(pf, float)
        assert isinstance(ev, float)


# ── compute_entry_attrs ───────────────────────────────────────────────────────


class TestComputeEntryAttrs:
    def test_sma_dist_flat_price(self):
        """Flat price = SMA distance near 0."""
        c = np.full(50, 100.0, dtype=np.float64)
        vr = _make_vr(50, 0.40)
        sma_dist, vr_depth = compute_entry_attrs(c, vr, 30, True, squeeze_dur=5)
        assert sma_dist == pytest.approx(0.0, abs=1e-10)

    def test_sma_dist_elevated_price(self):
        """Price 5% above SMA → sma_dist ≈ 0.05."""
        c = np.full(50, 100.0, dtype=np.float64)
        c[30] = 105.0  # just the entry bar elevated
        vr = _make_vr(50, 0.40)
        sma_dist, _ = compute_entry_attrs(c, vr, 30, True, squeeze_dur=1)
        # sma is avg of bars 11..30, bar 30=105, others=100 → sma = (19*100+105)/20 = 100.25
        expected = abs(105.0 - 100.25) / 100.25
        assert sma_dist == pytest.approx(expected, rel=0.01)

    def test_sma_dist_insufficient_history(self):
        """Too few bars → returns 0."""
        c = np.full(10, 100.0, dtype=np.float64)
        vr = _make_vr(10, 0.40)
        sma_dist, _ = compute_entry_attrs(c, vr, 5, True, squeeze_dur=0, sma_period=20)
        assert sma_dist == 0.0

    def test_vr_depth_below_threshold(self):
        """VR=0.40, S_VR=0.57 → depth=0.17."""
        c = _make_close(50)
        vr = _make_vr(50, 0.40)
        _, vr_depth = compute_entry_attrs(c, vr, 30, is_straddle=True, squeeze_dur=5)
        cfg = VolStraddleConfig()
        expected = max(0.0, cfg.s_vr_threshold - 0.40)
        assert vr_depth == pytest.approx(expected, rel=0.01)

    def test_vr_depth_above_threshold(self):
        """VR=0.70 > S_VR=0.57 → depth=0 (clamped)."""
        c = _make_close(50)
        vr = _make_vr(50, 0.70)
        _, vr_depth = compute_entry_attrs(c, vr, 30, is_straddle=True, squeeze_dur=0)
        assert vr_depth == 0.0

    def test_vr_depth_directional(self):
        """D_mom8: uses d_vr_threshold (0.5)."""
        c = _make_close(50)
        vr = _make_vr(50, 0.30)
        _, vr_depth = compute_entry_attrs(c, vr, 30, is_straddle=False, squeeze_dur=0)
        cfg = VolStraddleConfig()
        expected = max(0.0, cfg.d_vr_threshold - 0.30)
        assert vr_depth == pytest.approx(expected, rel=0.01)

    def test_nan_vr_treated_as_zero(self):
        """NaN VR → treated as vr=0 → depth = s_vr_threshold (full depth)."""
        c = _make_close(50)
        vr = np.full(50, np.nan)
        _, vr_depth = compute_entry_attrs(c, vr, 30, is_straddle=True, squeeze_dur=0)
        cfg = VolStraddleConfig()
        # NaN → vr_val=0.0 → depth = s_vr - 0.0 = s_vr
        assert vr_depth == pytest.approx(cfg.s_vr_threshold, rel=0.01)


# ── bucket_analysis ───────────────────────────────────────────────────────────


class TestBucketAnalysis:
    def _make_trades(self, attrs: list[float], pnls: list[float]) -> list[tuple]:
        """Create minimal trade tuples: (sym, entry, exit, pnl, attr_at_4, attr_at_5, attr_at_6)."""
        return [("SYM", 0, 1, pnl, 0, attr, 0.0) for attr, pnl in zip(attrs, pnls)]

    def test_bucket_counts(self, capsys):
        """Trades are correctly assigned to buckets."""
        attrs = [0.001, 0.003, 0.007, 0.015, 0.025]
        pnls = [0.01, 0.01, 0.01, 0.01, 0.01]
        trades = self._make_trades(attrs, pnls)
        buckets = [("low", 0.0, 0.005), ("mid", 0.005, 0.020), ("high", 0.020, 1.0)]
        results = bucket_analysis(trades, 5, buckets, "test")
        assert results[0]["N"] == 2  # 0.001, 0.003
        assert results[1]["N"] == 2  # 0.007, 0.015
        assert results[2]["N"] == 1  # 0.025

    def test_bucket_wr_calculation(self, capsys):
        """WR computed correctly per bucket."""
        # 2 wins at low, 2 losses at high
        attrs = [0.001, 0.003, 0.015, 0.025]
        pnls = [0.01, 0.02, -0.01, -0.02]
        trades = self._make_trades(attrs, pnls)
        buckets = [("low", 0.0, 0.005), ("high", 0.005, 1.0)]
        results = bucket_analysis(trades, 5, buckets, "test")
        assert results[0]["WR"] == pytest.approx(1.0)
        assert results[1]["WR"] == pytest.approx(0.0)

    def test_empty_bucket(self, capsys):
        """Empty bucket returns N=0."""
        trades = self._make_trades([0.001], [0.01])
        buckets = [("all", 0.0, 0.002), ("empty", 0.002, 1.0)]
        results = bucket_analysis(trades, 5, buckets, "test")
        assert results[1]["N"] == 0


# ── threshold_sweep ───────────────────────────────────────────────────────────


class TestThresholdSweep:
    def _make_trades(self, attrs: list[float], pnls: list[float]) -> list[tuple]:
        return [("SYM", 0, 1, pnl, 0, attr, 0.0) for attr, pnl in zip(attrs, pnls)]

    def test_above_mode(self, capsys):
        """'above' keeps trades where attr > threshold."""
        attrs = [0.001, 0.005, 0.010, 0.020]
        pnls = [0.01] * 4
        trades = self._make_trades(attrs, pnls)
        results = threshold_sweep(trades, 5, [0.005, 0.010], "above", "test", 1.0)
        # threshold=0.005: keeps 0.010, 0.020 (N=2)
        assert results[0]["N"] == 2
        # threshold=0.010: keeps 0.020 (N=1)
        assert results[1]["N"] == 1

    def test_below_mode(self, capsys):
        """'below' keeps trades where attr < threshold."""
        attrs = [0.001, 0.005, 0.010, 0.020]
        pnls = [0.01] * 4
        trades = self._make_trades(attrs, pnls)
        results = threshold_sweep(trades, 5, [0.010, 0.020], "below", "test", 1.0)
        # threshold=0.010: keeps 0.001, 0.005 (N=2)
        assert results[0]["N"] == 2
        # threshold=0.020: keeps 0.001, 0.005, 0.010 (N=3)
        assert results[1]["N"] == 3

    def test_pct_kept_correct(self, capsys):
        """pct_kept = N_kept / total * 100."""
        attrs = [0.001, 0.010, 0.020, 0.030]
        pnls = [0.01] * 4
        trades = self._make_trades(attrs, pnls)
        results = threshold_sweep(trades, 5, [0.015], "above", "test", 1.0)
        # keeps 0.020, 0.030 → N=2 / 4 = 50%
        assert results[0]["pct_kept"] == pytest.approx(50.0)

    def test_empty_filter_returns_zero(self, capsys):
        """Threshold above all values → N=0."""
        attrs = [0.001, 0.002]
        pnls = [0.01] * 2
        trades = self._make_trades(attrs, pnls)
        results = threshold_sweep(trades, 5, [0.999], "above", "test", 1.0)
        assert results[0]["N"] == 0


# ── evaluate_with_portfolio_sim ───────────────────────────────────────────────


class TestEvaluateWithPortfolioSim:
    def _make_sym_trades(self, n: int, pnl_val: float, attrs: tuple) -> list[tuple]:
        """Create trades as (sym, entry, exit, pnl, sq_dur, sma_dist, vr_depth)."""
        trades = []
        spacing = 500  # non-overlapping entries
        for i in range(n):
            entry = i * spacing
            exit_ = entry + 100
            # spread pnl slightly around pnl_val
            p = pnl_val + (i % 3 - 1) * 0.001
            trades.append(("SYMA", entry, exit_, p, attrs[0], attrs[1], attrs[2]))
        return trades

    def test_baseline_returns_metrics(self):
        """Baseline (no filter) returns valid N, WR, PF."""
        s_trades = self._make_sym_trades(20, 0.01, (5, 0.005, 0.10))
        d_trades = self._make_sym_trades(10, 0.02, (0, 0.003, 0.08))
        r = evaluate_with_portfolio_sim(s_trades, d_trades, lambda t: True, lambda t: True, 1.0, "test_baseline")
        assert r["N"] > 0
        assert 0 < r["WR"] <= 100
        assert r["PF"] > 0

    def test_filter_reduces_n(self):
        """Strict filter reduces N vs no filter."""
        s_trades = self._make_sym_trades(20, 0.01, (5, 0.005, 0.10))
        d_trades = self._make_sym_trades(10, 0.02, (0, 0.003, 0.08))
        r_all = evaluate_with_portfolio_sim(s_trades, d_trades, lambda t: True, lambda t: True, 1.0, "all")
        # Filter out all s_trades (sma_dist=0.005, require >0.01)
        r_strict = evaluate_with_portfolio_sim(s_trades, d_trades, lambda t: t[5] > 0.01, lambda t: True, 1.0, "strict")
        assert r_strict["N"] < r_all["N"]

    def test_empty_trades_returns_zero(self):
        """No trades after filtering → N=0."""
        s_trades = self._make_sym_trades(5, 0.01, (5, 0.003, 0.10))
        d_trades = []
        r = evaluate_with_portfolio_sim(
            s_trades,
            d_trades,
            lambda t: t[5] > 100.0,  # impossible filter
            lambda t: True,
            1.0,
            "empty",
        )
        assert r["N"] == 0


# ── Config parity checks ──────────────────────────────────────────────────────


class TestConfigParity:
    def test_production_params(self):
        """All key production params match expected values."""
        cfg = VolStraddleConfig()
        assert cfg.s_vr_threshold == pytest.approx(0.57)
        assert cfg.d_vr_threshold == pytest.approx(0.5)
        assert cfg.s_squeeze_dur == 3
        assert cfg.s_trail_activate == pytest.approx(0.015)
        assert cfg.d_trail_activate == pytest.approx(0.010)
        assert cfg.s_trail_distance == pytest.approx(0.002)  # r94: 0.001→0.002 (too tight for live)
        assert cfg.d_trail_distance == pytest.approx(0.002)
        assert cfg.max_concurrent == 10
        assert len(cfg.symbol_whitelist) == 59  # 2026-03-19: RLCUSDT removed (notional limit)

    def test_ln_ratio(self):
        """L/N must be 3.0 for VR squeeze."""
        cfg = VolStraddleConfig()
        ln = cfg.base_leverage / cfg.max_concurrent
        assert ln == pytest.approx(3.0)
