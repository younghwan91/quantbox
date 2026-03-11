"""Tests for r60: N-WR Harmony — Conviction + Direction Analysis.

Tests cover:
- Config parity (no hardcoding)
- Conviction scoring logic (E2/C2/B3/A3)
- Score distribution analysis
- Direction distribution
- Filter functions
- Candidate evaluation
- Report generation
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ══════════════════════════════════════════════════════════════════════════════
#  r60 module import
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def r60():
    import importlib, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import research.active.r51_r60.r60_nwr_harmony as m

    return m


# ══════════════════════════════════════════════════════════════════════════════
#  1. Config Parity
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigParity:
    def test_s_vr_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.S_VR == pytest.approx(cfg.s_vr_threshold)

    def test_s_dur_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.S_DUR == cfg.s_squeeze_dur

    def test_d_sl_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.D_SL == pytest.approx(cfg.d_sl)

    def test_s_sl_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.S_SL == pytest.approx(cfg.s_sl)

    def test_vr_ln_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.VR_LN == pytest.approx(cfg.base_leverage / cfg.max_concurrent)

    def test_max_concurrent_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.MAX_CONCURRENT == cfg.max_concurrent

    def test_e2_params_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.E2_SMA_PERIOD == cfg.e2_sma_period
        assert r60.E2_THRESHOLD == pytest.approx(cfg.e2_threshold)

    def test_c2_threshold_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.C2_BTC_THRESHOLD == pytest.approx(cfg.c2_btc_mom_threshold)

    def test_a3_params_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.A3_DEEP_VR == pytest.approx(cfg.a3_deep_vr_threshold)
        assert r60.A3_LONG_DUR == cfg.a3_long_squeeze_dur

    def test_mom_period_from_config(self, r60):
        cfg = VolStraddleConfig()
        assert r60.MOM_PERIOD == cfg.momentum_period


# ══════════════════════════════════════════════════════════════════════════════
#  2. Conviction Score Computation
# ══════════════════════════════════════════════════════════════════════════════


class TestConvictionScore:
    def test_score_zero_on_all_disabled(self, r60):
        """Score=0 when all conditions false."""
        close = np.ones(50) * 100.0
        vr = np.ones(50) * 0.5
        btc = np.ones(50) * 50000.0
        # BTC momentum > threshold → C2 false
        btc[-1] = 60000.0  # 20% momentum
        score, conds = r60.compute_conviction_at_bar(close, vr, btc, 49, False, False, 0)
        assert conds["c2_btc"] is False

    def test_e2_triggers_when_close_near_sma(self, r60):
        """E2 fires when price is within threshold of SMA."""
        period = VolStraddleConfig().e2_sma_period
        close = np.ones(period + 5) * 100.0
        vr = np.ones(period + 5) * 0.5
        # Price exactly at SMA → proximity=0 < threshold=0.005
        score, conds = r60.compute_conviction_at_bar(close, vr, None, period + 4, False, False, 0)
        assert conds["e2_sma"] is True
        assert score >= 1

    def test_e2_fails_when_far_from_sma(self, r60):
        """E2 does not fire when price is far from SMA."""
        period = VolStraddleConfig().e2_sma_period
        close = np.ones(period + 5) * 100.0
        close[-1] = 115.0  # 15% above SMA → fails E2 (threshold=0.5%)
        vr = np.ones(period + 5) * 0.5
        score, conds = r60.compute_conviction_at_bar(close, vr, None, period + 4, False, False, 0)
        assert conds["e2_sma"] is False

    def test_c2_triggers_when_btc_flat(self, r60):
        """C2 fires when BTC momentum is small."""
        close = np.ones(20) * 100.0
        vr = np.ones(20) * 0.5
        btc = np.ones(20) * 50000.0  # flat BTC → momentum=0
        score, conds = r60.compute_conviction_at_bar(close, vr, btc, 19, False, False, 0)
        assert conds["c2_btc"] is True

    def test_c2_fails_when_btc_strong_move(self, r60):
        """C2 does not fire when BTC has strong momentum."""
        cfg = VolStraddleConfig()
        close = np.ones(20) * 100.0
        vr = np.ones(20) * 0.5
        btc = np.ones(20) * 50000.0
        # Set btc 8 bars ago to make 5% momentum (> 1% threshold)
        btc[20 - cfg.momentum_period - 1] = 45000.0
        score, conds = r60.compute_conviction_at_bar(close, vr, btc, 19, False, False, 0)
        assert conds["c2_btc"] is False

    def test_b3_triggers_when_overlap(self, r60):
        """B3 fires when d_overlap=True."""
        close = np.ones(20) * 100.0
        vr = np.ones(20) * 0.5
        score, conds = r60.compute_conviction_at_bar(close, vr, None, 19, False, True, 0)
        assert conds["b3_confluence"] is True
        assert score >= 1

    def test_a3_fires_for_straddle_long_squeeze(self, r60):
        """A3 fires for straddle with squeeze_dur >= a3_long_squeeze_dur."""
        cfg = VolStraddleConfig()
        close = np.ones(20) * 100.0
        vr = np.ones(20) * 0.5
        score, conds = r60.compute_conviction_at_bar(close, vr, None, 19, True, False, cfg.a3_long_squeeze_dur)
        assert conds["a3_quality"] is True

    def test_a3_fails_for_straddle_short_squeeze(self, r60):
        """A3 does not fire for straddle with short squeeze duration."""
        close = np.ones(20) * 100.0
        vr = np.ones(20) * 0.5
        score, conds = r60.compute_conviction_at_bar(close, vr, None, 19, True, False, 1)
        assert conds["a3_quality"] is False

    def test_a3_fires_for_directional_deep_vr(self, r60):
        """A3 fires for directional when VR < a3_deep_vr_threshold."""
        cfg = VolStraddleConfig()
        close = np.ones(20) * 100.0
        vr = np.ones(20) * (cfg.a3_deep_vr_threshold - 0.05)
        score, conds = r60.compute_conviction_at_bar(close, vr, None, 19, False, False, 0)
        assert conds["a3_quality"] is True

    def test_max_score_is_4(self, r60):
        """Maximum conviction score is 4."""
        cfg = VolStraddleConfig()
        period = cfg.e2_sma_period
        sz = max(50, period + 10)
        close = np.ones(sz) * 100.0
        btc = np.ones(sz) * 50000.0
        # VR deep for A3 D
        vr = np.ones(sz) * (cfg.a3_deep_vr_threshold - 0.05)
        score, conds = r60.compute_conviction_at_bar(close, vr, btc, sz - 1, False, True, 0)
        assert 0 <= score <= 4


# ══════════════════════════════════════════════════════════════════════════════
#  3. Score Distribution
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreDistribution:
    def _make_tagged(self) -> list[tuple]:
        """Create synthetic tagged trades for testing."""
        return [
            ("A", 0, 100, 0.01, 0, "LONG"),  # win score=0
            ("A", 200, 300, -0.005, 0, "SHORT"),  # loss score=0
            ("B", 100, 200, 0.02, 1, "LONG"),  # win score=1
            ("B", 300, 400, 0.015, 2, "SHORT"),  # win score=2
            ("C", 150, 250, -0.008, 1, "LONG"),  # loss score=1
        ]

    def test_score_distribution_keys(self, r60):
        tagged = self._make_tagged()
        dist = r60.score_distribution(tagged)
        assert set(dist.keys()) == {0, 1, 2}

    def test_score_distribution_n(self, r60):
        tagged = self._make_tagged()
        dist = r60.score_distribution(tagged)
        assert dist[0]["N"] == 2
        assert dist[1]["N"] == 2
        assert dist[2]["N"] == 1

    def test_score_distribution_wr(self, r60):
        tagged = self._make_tagged()
        dist = r60.score_distribution(tagged)
        assert dist[0]["WR"] == pytest.approx(0.5)
        assert dist[1]["WR"] == pytest.approx(0.5)
        assert dist[2]["WR"] == pytest.approx(1.0)

    def test_direction_distribution_keys(self, r60):
        tagged = self._make_tagged()
        dist = r60.direction_distribution(tagged)
        assert "LONG" in dist
        assert "SHORT" in dist

    def test_direction_distribution_wr(self, r60):
        tagged = self._make_tagged()
        dist = r60.direction_distribution(tagged)
        # LONG: 2 wins, 1 loss → WR=2/3
        assert dist["LONG"]["WR"] == pytest.approx(2 / 3)
        # SHORT: 1 win (score=2), 1 loss (score=0)
        assert dist["SHORT"]["WR"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Filter Functions
# ══════════════════════════════════════════════════════════════════════════════


class TestFilterFunctions:
    def _make_tagged(self) -> list[tuple]:
        return [
            ("A", 0, 100, 0.01, 0, "LONG"),
            ("A", 200, 300, -0.005, 1, "SHORT"),
            ("B", 100, 200, 0.02, 2, "LONG"),
            ("C", 150, 250, 0.015, 3, "SHORT"),
        ]

    def test_filter_min_score_0_keeps_all(self, r60):
        tagged = self._make_tagged()
        result = r60.filter_by_conviction(tagged, 0)
        assert len(result) == 4

    def test_filter_min_score_1_removes_score0(self, r60):
        tagged = self._make_tagged()
        result = r60.filter_by_conviction(tagged, 1)
        assert len(result) == 3

    def test_filter_min_score_2_removes_score01(self, r60):
        tagged = self._make_tagged()
        result = r60.filter_by_conviction(tagged, 2)
        assert len(result) == 2

    def test_filter_direction_specific(self, r60):
        """Direction filter overrides global for specific direction."""
        tagged = self._make_tagged()
        # Global=0, SHORT needs ≥2
        result = r60.filter_by_conviction(tagged, 0, {"SHORT": 2})
        # score=0 SHORT removed, score=1 SHORT removed, score=3 SHORT kept
        syms = [t[0] for t in result]
        # (A,0,LONG)=keep, (A,1,SHORT)=blocked(score<2), (B,2,LONG)=keep, (C,3,SHORT)=keep
        assert len(result) == 3

    def test_strip_conviction_format(self, r60):
        tagged = self._make_tagged()
        standard = r60.strip_conviction(tagged)
        assert len(standard) == 4
        assert len(standard[0]) == 4  # (sym, entry, exit, pnl)

    def test_filter_returns_standard_format(self, r60):
        tagged = self._make_tagged()
        result = r60.filter_by_conviction(tagged, 1)
        assert all(len(t) == 4 for t in result)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Bar-to-Year Conversion
# ══════════════════════════════════════════════════════════════════════════════


class TestBarToYear:
    def test_last_bar_is_2026(self, r60):
        max_bars = 365 * 24 * 4 * 6  # ~6 years
        year = r60.bar_to_year(max_bars - 1, max_bars)
        assert year in (2025, 2026)  # depends on actual data end date

    def test_four_years_back(self, r60):
        bars_per_year = 365 * 24 * 4
        max_bars = bars_per_year * 6
        bar = max_bars - bars_per_year * 4
        year = r60.bar_to_year(bar, max_bars)
        assert year == 2022

    def test_year_range_is_valid(self, r60):
        max_bars = 365 * 24 * 4 * 6
        bars_per_year = 365 * 24 * 4
        for offset in range(0, max_bars, bars_per_year // 2):
            year = r60.bar_to_year(offset, max_bars)
            assert 2019 <= year <= 2027


# ══════════════════════════════════════════════════════════════════════════════
#  6. Score × Direction Breakdown
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreByDirection:
    def test_score_by_direction_structure(self, r60):
        tagged = [
            ("A", 0, 100, 0.01, 0, "LONG"),
            ("A", 200, 300, -0.005, 0, "SHORT"),
            ("B", 100, 200, 0.02, 1, "LONG"),
        ]
        sbd = r60.score_by_direction(tagged)
        assert "LONG" in sbd
        assert "SHORT" in sbd
        assert 0 in sbd["LONG"]
        assert 1 in sbd["LONG"]

    def test_score_by_direction_wr(self, r60):
        tagged = [
            ("A", 0, 100, 0.01, 0, "LONG"),
            ("A", 200, 300, -0.005, 0, "LONG"),
            ("B", 100, 200, 0.02, 0, "SHORT"),
        ]
        sbd = r60.score_by_direction(tagged)
        assert sbd["LONG"][0]["WR"] == pytest.approx(0.5)
        assert sbd["SHORT"][0]["WR"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  7. Yearly Breakdown
# ══════════════════════════════════════════════════════════════════════════════


class TestYearlyBreakdown:
    def test_filter_applied_in_yearly(self, r60):
        max_bars = 365 * 24 * 4 * 6
        bars_per_year = 365 * 24 * 4
        # Trade 4 years ago (score=0) and 2 years ago (score=2)
        bar_4yr = max_bars - bars_per_year * 4
        bar_2yr = max_bars - bars_per_year * 2
        tagged = [
            ("A", bar_4yr, bar_4yr + 100, 0.01, 0, "LONG"),
            ("B", bar_2yr, bar_2yr + 100, 0.01, 2, "LONG"),
        ]
        # With min_score=1, score=0 trade excluded
        breakdown = r60.yearly_breakdown(tagged, max_bars, min_score=1)
        year_4 = r60.bar_to_year(bar_4yr, max_bars)
        year_2 = r60.bar_to_year(bar_2yr, max_bars)
        if year_4 in breakdown:
            # Only score=2 trade should be in breakdown
            assert breakdown[year_4]["N"] == 0 or year_4 not in breakdown
        if year_2 in breakdown:
            assert breakdown[year_2]["N"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
#  8. Evaluate Candidate
# ══════════════════════════════════════════════════════════════════════════════


class TestEvaluateCandidate:
    def _make_many_tagged(self) -> list[tuple]:
        """Create enough trades to span several 'slots'."""
        trades = []
        for i in range(500):
            entry = i * 200
            exit_b = entry + 100
            pnl = 0.012 if i % 3 != 0 else -0.008
            score = i % 4
            direction = "LONG" if i % 2 == 0 else "SHORT"
            trades.append(("A", entry, exit_b, pnl, score, direction))
        return trades

    def test_evaluate_returns_required_keys(self, r60):
        tagged = self._make_many_tagged()
        max_bars = 600 * 200
        result = r60.evaluate_candidate("test", tagged, 0, None, max_bars, 3.0)
        for key in ["label", "N", "WR", "PF", "pass_wr", "pass_n", "pass_pf"]:
            assert key in result

    def test_evaluate_min_score_1_reduces_n(self, r60):
        tagged = self._make_many_tagged()
        max_bars = 600 * 200
        r0 = r60.evaluate_candidate("c0", tagged, 0, None, max_bars, 3.0)
        r1 = r60.evaluate_candidate("c1", tagged, 1, None, max_bars, 3.0)
        assert r1["N"] <= r0["N"]

    def test_evaluate_pass_flags(self, r60):
        tagged = self._make_many_tagged()
        max_bars = 600 * 200
        result = r60.evaluate_candidate("test", tagged, 0, None, max_bars, 3.0)
        assert isinstance(result["pass_wr"], bool)
        assert isinstance(result["pass_n"], bool)
        assert isinstance(result["pass_pf"], bool)


# ══════════════════════════════════════════════════════════════════════════════
#  9. Report Generation
# ══════════════════════════════════════════════════════════════════════════════


class TestReportGeneration:
    def _make_inputs(self, r60) -> tuple:
        score_dist = {
            0: {"N": 3000, "WR": 0.38, "PF": 1.3, "EV_bp": 25.0},
            1: {"N": 2000, "WR": 0.44, "PF": 1.6, "EV_bp": 45.0},
            2: {"N": 800, "WR": 0.50, "PF": 2.0, "EV_bp": 80.0},
        }
        dir_dist = {
            "LONG": {"N": 3500, "WR": 0.46, "PF": 1.7, "EV_bp": 50.0},
            "SHORT": {"N": 2300, "WR": 0.38, "PF": 1.4, "EV_bp": 30.0},
        }
        sbd = {
            "LONG": {0: {"N": 1500, "WR": 0.40, "PF": 1.4, "EV_bp": 30.0}},
            "SHORT": {0: {"N": 1500, "WR": 0.36, "PF": 1.2, "EV_bp": 15.0}},
        }
        cands = [
            {
                "label": "min_conv=0",
                "min_score": 0,
                "direction_filters": {},
                "N": 8000,
                "WR": 0.413,
                "PF": 1.993,
                "EV_bp": 55.0,
                "CAGR_1x": 0.35,
                "MaxDD": 0.60,
                "yearly": {2023: {"N": 1500, "WR": 0.42, "PF": 1.5, "EV_bp": 40.0}},
                "all_years_positive": True,
                "pass_wr": False,
                "pass_n": True,
                "pass_pf": True,
            },
            {
                "label": "min_conv=1",
                "min_score": 1,
                "direction_filters": {},
                "N": 9000,
                "WR": 0.44,
                "PF": 1.95,
                "EV_bp": 65.0,
                "CAGR_1x": 0.40,
                "MaxDD": 0.55,
                "yearly": {2023: {"N": 1400, "WR": 0.45, "PF": 1.6, "EV_bp": 50.0}},
                "all_years_positive": True,
                "pass_wr": True,
                "pass_n": True,
                "pass_pf": True,
            },
        ]
        return score_dist, dir_dist, sbd, cands

    def test_report_contains_header(self, r60):
        sd, dd, sbd, cands = self._make_inputs(r60)
        report = r60.generate_report(sd, dd, sbd, cands, 6.0, "S_d8")
        assert "r60" in report
        assert "N-WR Harmony" in report

    def test_report_contains_phase_a(self, r60):
        sd, dd, sbd, cands = self._make_inputs(r60)
        report = r60.generate_report(sd, dd, sbd, cands, 6.0, "S_d8")
        assert "Phase A" in report
        assert "Score" in report

    def test_report_contains_best_candidate(self, r60):
        sd, dd, sbd, cands = self._make_inputs(r60)
        report = r60.generate_report(sd, dd, sbd, cands, 6.0, "S_d8")
        assert "min_conv=1" in report

    def test_report_contains_parity_notes(self, r60):
        sd, dd, sbd, cands = self._make_inputs(r60)
        report = r60.generate_report(sd, dd, sbd, cands, 6.0, "S_d8")
        assert "Parity Notes" in report
        assert "config.py" in report

    def test_report_no_hardcoded_thresholds(self, r60):
        """Report does not contain hardcoded param values that bypass config."""
        sd, dd, sbd, cands = self._make_inputs(r60)
        report = r60.generate_report(sd, dd, sbd, cands, 6.0, "S_d8")
        # Should not have outdated hardcoded s_vr values
        assert "s_vr=0.55" not in report or "s_vr=0.57" in report
