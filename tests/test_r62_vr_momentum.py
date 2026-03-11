"""Tests for r62: VR Momentum Filter analysis.

Tests cover:
  - compute_vr_momentum: dVR and bars_since_min computation
  - collect_s_d8_tagged / collect_d_mom8_tagged: tuple structure and tag ranges
  - dvr_bucket_analysis / bsm_bucket_analysis: bucket coverage
  - dvr_threshold_sweep / bsm_threshold_sweep: monotonic N decrease
  - evaluate_portfolio: portfolio_sim integration
  - Config parity: production parameters
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.active.r61_r65.r62_vr_momentum import (
    BSM_WINDOW,
    D_TRAIL_ACT,
    D_VR,
    S_TRAIL_ACT,
    S_VR,
    bsm_bucket_analysis,
    bsm_threshold_sweep,
    collect_d_mom8_tagged,
    collect_s_d8_tagged,
    compute_vr_momentum,
    dvr_bucket_analysis,
    dvr_threshold_sweep,
    evaluate_portfolio,
    wr_stats,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_vr(values: list[float]) -> np.ndarray:
    """Build VR array from plain list."""
    return np.array(values, dtype=np.float64)


def _make_minimal_sym_data(n: int = 400) -> dict:
    """Build minimal symbol data dict for inject into collect functions."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.01)
    close = np.clip(close, 0.01, None)
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    # Synthetic VR: alternating low (squeeze) and high regions
    vr = np.full(n, 0.6)
    # Create squeeze windows to trigger signals
    for start in range(50, n - 50, 80):
        vr[start : start + 20] = 0.40  # deep squeeze → both S and D signals
    vr_key = "vr_8_56"
    mom = np.zeros(n)
    mom[8:] = close[8:] / close[:-8] - 1.0
    return {
        "close": close,
        "high": high,
        "low": low,
        vr_key: vr,
        "vr": vr,
        "mom_8": mom,
        "_offset": 0,
    }


def _make_flat_resampled(data_15m: dict, ratio: int) -> dict:
    """Build minimal resampled data dict (1 symbol, simplified)."""
    c = data_15m["close"]
    n_bars = max(1, len(c) // ratio)
    flat_close = np.array([c[i * ratio] for i in range(n_bars)])
    flat_high = flat_close * 1.005
    flat_low = flat_close * 0.995
    vr_15m = data_15m.get("vr_8_56", np.full(len(c), 0.6))
    flat_vr = np.array([vr_15m[i * ratio] for i in range(n_bars)])
    mom = np.zeros(n_bars)
    mom[8:] = flat_close[8:] / flat_close[:-8] - 1.0
    return {
        "close": flat_close,
        "high": flat_high,
        "low": flat_low,
        "vr_8_56": flat_vr,
        "vr": flat_vr,
        "mom_8": mom,
        "_offset": 0,
    }


@pytest.fixture
def sym_data_15m():
    return _make_minimal_sym_data(600)


@pytest.fixture
def three_tf_data(sym_data_15m):
    d15 = {"TESTSYM": sym_data_15m}
    d1h = {"TESTSYM": _make_flat_resampled(sym_data_15m, 4)}
    d4h = {"TESTSYM": _make_flat_resampled(sym_data_15m, 16)}
    return d15, d1h, d4h, ["TESTSYM"]


# ── Tests: compute_vr_momentum ────────────────────────────────────────────────


class TestComputeVrMomentum:
    def test_dvr_positive_when_rising(self):
        vr = _make_vr([0.5, 0.45, 0.42, 0.44, 0.48])
        dvr, _ = compute_vr_momentum(vr, 4)
        assert dvr == pytest.approx(0.48 - 0.44, abs=1e-10)
        assert dvr > 0

    def test_dvr_negative_when_falling(self):
        vr = _make_vr([0.5, 0.48, 0.45, 0.42])
        dvr, _ = compute_vr_momentum(vr, 3)
        assert dvr == pytest.approx(0.42 - 0.45, abs=1e-10)
        assert dvr < 0

    def test_dvr_zero_at_bar_0(self):
        vr = _make_vr([0.5, 0.48, 0.45])
        dvr, _ = compute_vr_momentum(vr, 0)
        assert dvr == 0.0

    def test_dvr_nan_previous_bar(self):
        vr = _make_vr([np.nan, 0.48, 0.45])
        dvr, _ = compute_vr_momentum(vr, 1)
        assert dvr == 0.0

    def test_dvr_nan_current_bar(self):
        vr = _make_vr([0.5, np.nan, 0.45])
        dvr, _ = compute_vr_momentum(vr, 1)
        assert dvr == 0.0

    def test_bsm_at_current_bottom(self):
        # VR decreasing — current bar is the minimum
        vr = _make_vr([0.5, 0.48, 0.46, 0.44, 0.42])
        _, bsm = compute_vr_momentum(vr, 4)
        assert bsm == 0

    def test_bsm_one_bar_ago(self):
        # Minimum was 1 bar ago
        vr = _make_vr([0.5, 0.48, 0.44, 0.42, 0.45])  # min at index 3
        _, bsm = compute_vr_momentum(vr, 4)
        assert bsm == 1

    def test_bsm_two_bars_ago(self):
        # Minimum was 2 bars ago
        vr = _make_vr([0.5, 0.48, 0.42, 0.45, 0.47])  # min at index 2
        _, bsm = compute_vr_momentum(vr, 4)
        assert bsm == 2

    def test_bsm_respects_window(self):
        # Window = 3 — don't look beyond 3 bars back
        long_vr = _make_vr([0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        _, bsm = compute_vr_momentum(long_vr, 6, window=3)
        # window=3: looks at [6-3:7] = [3,4,5,6] = [0.5,0.5,0.5,0.5]
        # minimum = 0.5 at index 0 of segment → bsm = 3
        assert bsm == 3

    def test_bsm_all_nan(self):
        vr = _make_vr([np.nan, np.nan, np.nan])
        _, bsm = compute_vr_momentum(vr, 2)
        assert bsm == 0  # fallback

    def test_bsm_nonnegative(self):
        vr = _make_vr([0.5, 0.48, 0.46, 0.44, 0.44, 0.45, 0.47])
        for i in range(1, len(vr)):
            _, bsm = compute_vr_momentum(vr, i)
            assert bsm >= 0

    def test_return_types(self):
        vr = _make_vr([0.5, 0.48, 0.50])
        dvr, bsm = compute_vr_momentum(vr, 2)
        assert isinstance(dvr, float)
        assert isinstance(bsm, int)


# ── Tests: wr_stats ───────────────────────────────────────────────────────────


class TestWrStats:
    def test_all_wins(self):
        pnls = [0.01, 0.02, 0.03]
        wr, pf, ev = wr_stats(pnls)
        assert wr == pytest.approx(1.0)
        assert np.isinf(pf)
        assert ev > 0

    def test_all_losses(self):
        pnls = [-0.01, -0.02]
        wr, pf, ev = wr_stats(pnls)
        assert wr == pytest.approx(0.0)
        assert pf == pytest.approx(0.0)
        assert ev < 0

    def test_empty(self):
        wr, pf, ev = wr_stats([])
        assert wr == 0.0 and pf == 0.0 and ev == 0.0

    def test_mixed(self):
        pnls = [0.02, 0.02, -0.01]  # 2 wins, 1 loss
        wr, pf, ev = wr_stats(pnls)
        assert wr == pytest.approx(2 / 3)
        assert pf == pytest.approx(0.04 / 0.01)
        assert ev == pytest.approx(np.mean(pnls) * 10000)


# ── Tests: collect functions ──────────────────────────────────────────────────


class TestCollectTagged:
    def test_s_d8_returns_tuples_7_fields(self, three_tf_data):
        d15, d1h, d4h, syms = three_tf_data
        trades = collect_s_d8_tagged(d15, d1h, d4h, syms)
        if trades:
            assert len(trades[0]) == 7

    def test_d_mom8_returns_tuples_8_fields(self, three_tf_data):
        d15, d1h, d4h, syms = three_tf_data
        trades = collect_d_mom8_tagged(d15, d1h, d4h, syms)
        if trades:
            assert len(trades[0]) == 8

    def test_s_d8_dvr_is_float(self, three_tf_data):
        d15, d1h, d4h, syms = three_tf_data
        trades = collect_s_d8_tagged(d15, d1h, d4h, syms)
        for t in trades[:20]:
            assert isinstance(t[5], float), f"dvr should be float, got {type(t[5])}"

    def test_s_d8_bsm_is_nonnegative_int(self, three_tf_data):
        d15, d1h, d4h, syms = three_tf_data
        trades = collect_s_d8_tagged(d15, d1h, d4h, syms)
        for t in trades[:20]:
            assert isinstance(t[6], int)
            assert t[6] >= 0

    def test_s_d8_sq_dur_positive(self, three_tf_data):
        d15, d1h, d4h, syms = three_tf_data
        trades = collect_s_d8_tagged(d15, d1h, d4h, syms)
        for t in trades[:20]:
            assert t[4] > 0  # sq_dur always ≥ 1

    def test_d_mom8_direction_valid(self, three_tf_data):
        d15, d1h, d4h, syms = three_tf_data
        trades = collect_d_mom8_tagged(d15, d1h, d4h, syms)
        for t in trades[:20]:
            assert t[7] in ("LONG", "SHORT")

    def test_empty_syms_returns_empty(self, three_tf_data):
        d15, d1h, d4h, _ = three_tf_data
        assert collect_s_d8_tagged(d15, d1h, d4h, []) == []
        assert collect_d_mom8_tagged(d15, d1h, d4h, []) == []

    def test_missing_sym_skipped(self, three_tf_data):
        d15, d1h, d4h, _ = three_tf_data
        trades = collect_s_d8_tagged(d15, d1h, d4h, ["NOSYM"])
        assert trades == []


# ── Tests: bucket analysis ────────────────────────────────────────────────────


class TestBucketAnalysis:
    def _make_trades(self) -> list[tuple]:
        """Fake S_d8 tagged trades: (sym, entry, exit, pnl, sq_dur, dvr, bsm)"""
        return [
            ("A", 0, 10, 0.02, 3, -0.025, 0),  # strong falling, win
            ("A", 0, 10, -0.015, 5, -0.010, 0),  # weak falling, loss
            ("A", 0, 10, 0.03, 2, 0.005, 1),  # weak rising, win
            ("A", 0, 10, -0.01, 2, 0.025, 2),  # strong rising, loss
            ("A", 0, 10, 0.02, 3, 0.015, 3),  # weak rising, win
            ("A", 0, 10, 0.01, 4, 0.000, 4),  # flat/zero, win
        ]

    def test_dvr_buckets_cover_all_trades(self, capsys):
        trades = self._make_trades()
        results = dvr_bucket_analysis(trades, "Test dVR buckets")
        total = sum(r["N"] for r in results)
        assert total == len(trades)

    def test_bsm_buckets_cover_all_trades(self, capsys):
        trades = self._make_trades()
        results = bsm_bucket_analysis(trades, "Test BSM buckets")
        total = sum(r["N"] for r in results)
        assert total == len(trades)

    def test_dvr_bucket_wr_range(self, capsys):
        trades = self._make_trades()
        results = dvr_bucket_analysis(trades, "WR check")
        for r in results:
            if r["N"] > 0:
                assert 0.0 <= r["WR"] <= 1.0

    def test_bsm_bucket_wr_range(self, capsys):
        trades = self._make_trades()
        results = bsm_bucket_analysis(trades, "BSM WR check")
        for r in results:
            if r["N"] > 0:
                assert 0.0 <= r["WR"] <= 1.0


# ── Tests: threshold sweeps ────────────────────────────────────────────────────


class TestThresholdSweeps:
    def _make_trades(self) -> list[tuple]:
        trades = []
        for i, (dvr, bsm, pnl) in enumerate(
            [
                (-0.03, 0, -0.01),
                (-0.01, 0, -0.01),
                (0.0, 1, 0.02),
                (0.005, 1, 0.02),
                (0.01, 2, 0.02),
                (0.02, 3, 0.03),
                (0.025, 4, 0.03),
                (0.03, 5, 0.01),
            ]
        ):
            trades.append(("A", i, i + 10, pnl, 3, dvr, bsm))
        return trades

    def test_dvr_sweep_n_decreases_monotonically_rising(self, capsys):
        trades = self._make_trades()
        results = dvr_threshold_sweep(trades, "test", 1.0)
        ns = [r["N"] for r in results]
        # As threshold increases, N should not increase (≥ threshold means fewer trades)
        for i in range(len(ns) - 1):
            assert ns[i] >= ns[i + 1], f"N should not increase: {ns[i]} < {ns[i + 1]}"

    def test_bsm_sweep_n_decreases_monotonically(self, capsys):
        trades = self._make_trades()
        results = bsm_threshold_sweep(trades, "test", 1.0)
        ns = [r["N"] for r in results]
        for i in range(len(ns) - 1):
            assert ns[i] >= ns[i + 1]

    def test_dvr_sweep_k0_includes_all(self, capsys):
        trades = self._make_trades()
        results = dvr_threshold_sweep(trades, "test", 1.0)
        # First threshold (-0.030) should keep all trades with dvr > -0.030
        # At least the most lenient threshold should have highest N
        assert results[0]["N"] >= results[-1]["N"]

    def test_bsm_k0_keeps_all(self, capsys):
        trades = self._make_trades()
        results = bsm_threshold_sweep(trades, "test", 1.0)
        assert results[0]["N"] == len(trades)  # bsm>=0 keeps everything


# ── Tests: portfolio evaluation ───────────────────────────────────────────────


class TestEvaluatePortfolio:
    def _make_s_trades(self) -> list[tuple]:
        return [("AAAUSDT", i * 200, i * 200 + 100, 0.02 if i % 2 == 0 else -0.01, 3, 0.005, 1) for i in range(50)]

    def _make_d_trades(self) -> list[tuple]:
        return [
            ("BBBUSDT", i * 200, i * 200 + 100, 0.03 if i % 3 != 0 else -0.015, 0, 0.010, 2, "LONG") for i in range(30)
        ]

    def test_baseline_returns_dict(self):
        s = self._make_s_trades()
        d = self._make_d_trades()
        r = evaluate_portfolio(s, d, lambda t: True, lambda t: True, 6.0, "test")
        assert "N" in r and "WR" in r and "PF" in r
        assert r["N"] > 0

    def test_filter_reduces_n(self):
        s = self._make_s_trades()
        d = self._make_d_trades()
        r_all = evaluate_portfolio(s, d, lambda t: True, lambda t: True, 6.0, "all")
        r_filt = evaluate_portfolio(s, d, lambda t: t[5] > 0.003, lambda t: True, 6.0, "filtered")
        assert r_filt["N"] <= r_all["N"]

    def test_empty_trades_returns_zero_n(self):
        r = evaluate_portfolio([], [], lambda t: True, lambda t: True, 6.0, "empty")
        assert r["N"] == 0

    def test_wr_in_valid_range(self):
        s = self._make_s_trades()
        d = self._make_d_trades()
        r = evaluate_portfolio(s, d, lambda t: True, lambda t: True, 6.0, "test")
        assert 0.0 <= r["WR"] <= 100.0


# ── Tests: Config parity ──────────────────────────────────────────────────────


class TestConfigParity:
    def test_s_vr_threshold(self):
        assert S_VR == 0.57, f"S_VR={S_VR} — must match production Combo2"

    def test_d_vr_threshold(self):
        assert D_VR == 0.5, f"D_VR={D_VR}"

    def test_s_trail_activate(self):
        assert S_TRAIL_ACT == 0.015, f"S_TRAIL_ACT={S_TRAIL_ACT}"

    def test_d_trail_activate(self):
        assert D_TRAIL_ACT == 0.005, f"D_TRAIL_ACT={D_TRAIL_ACT}"

    def test_bsm_window_positive(self):
        assert BSM_WINDOW > 0
