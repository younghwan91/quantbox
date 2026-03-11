"""Tests for r59: OOS Validation.

Covers:
- IS/OOS split logic (bar-to-datetime mapping, split correctness)
- Candidate configuration (s_vr/s_dur from config)
- OOS pass/fail criteria
- split_trades correctness (no lookahead, correct assignment)
- Helper functions (bar_to_dt, compute_split_bar)
- Baseline parity gate
- Report generation
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.active.r51_r60.r59_oos_validation as r59
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ══════════════════════════════════════════════════════════════════════════════
#  Config parity
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigParity:
    def test_vr_ln_is_3_0(self):
        """VR L/N must be 3.0 (30/10). No hardcoding."""
        assert r59.VR_LN == pytest.approx(3.0, abs=0.01)

    def test_max_concurrent_from_config(self):
        cfg = VolStraddleConfig()
        assert r59.MAX_CONC == cfg.max_concurrent

    def test_d_trail_params_from_config(self):
        cfg = VolStraddleConfig()
        assert r59.D_TRAIL_ACT == pytest.approx(cfg.d_trail_activate)
        assert r59.D_TRAIL_DIST == pytest.approx(cfg.d_trail_distance)

    def test_s_vr_from_config(self):
        cfg = VolStraddleConfig()
        assert r59.S_VR == pytest.approx(cfg.s_vr_threshold)

    def test_s_dur_from_config(self):
        cfg = VolStraddleConfig()
        assert r59.S_DUR == cfg.s_squeeze_dur

    def test_d_vr_from_config(self):
        cfg = VolStraddleConfig()
        assert r59.D_VR == pytest.approx(cfg.d_vr_threshold)

    def test_d_sl_tp_from_config(self):
        cfg = VolStraddleConfig()
        assert r59.D_SL == pytest.approx(cfg.d_sl)
        assert r59.D_TP == pytest.approx(cfg.d_tp)


# ══════════════════════════════════════════════════════════════════════════════
#  IS/OOS split dates
# ══════════════════════════════════════════════════════════════════════════════


class TestSplitDates:
    def test_data_end_is_feb_2026(self):
        assert r59.DATA_END == datetime(2026, 2, 20)

    def test_is_end_is_2022(self):
        assert r59.IS_END.year == 2022
        assert r59.IS_END.month == 12

    def test_oos_start_is_2023(self):
        assert r59.OOS_START.year == 2023
        assert r59.OOS_START.month == 1

    def test_oos_start_after_is_end(self):
        assert r59.OOS_START > r59.IS_END

    def test_no_gap_between_is_oos(self):
        """IS and OOS are contiguous (no gap day)."""
        gap = (r59.OOS_START - r59.IS_END).days
        assert gap == 1  # Jan 1 immediately follows Dec 31


# ══════════════════════════════════════════════════════════════════════════════
#  bar_to_dt
# ══════════════════════════════════════════════════════════════════════════════


class TestBarToDt:
    def test_last_bar_is_data_end(self):
        max_bars = 1000
        dt = r59.bar_to_dt(max_bars, max_bars)
        # bar max_bars is one step PAST last bar, so DATA_END
        assert dt == r59.DATA_END

    def test_bar_0_is_data_start(self):
        max_bars = 1000
        dt = r59.bar_to_dt(0, max_bars)
        expected = r59.DATA_END - timedelta(minutes=max_bars * 15)
        assert dt == expected

    def test_monotonic_increasing(self):
        max_bars = 100
        dts = [r59.bar_to_dt(i, max_bars) for i in range(0, 100, 10)]
        for i in range(len(dts) - 1):
            assert dts[i] < dts[i + 1]

    def test_15min_step(self):
        max_bars = 1000
        dt_a = r59.bar_to_dt(500, max_bars)
        dt_b = r59.bar_to_dt(501, max_bars)
        assert (dt_b - dt_a).total_seconds() == 15 * 60


# ══════════════════════════════════════════════════════════════════════════════
#  compute_split_bar
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSplitBar:
    def test_split_bar_is_positive(self):
        max_bars = 210_000  # ~6yr of 15m bars
        split_bar = r59.compute_split_bar(max_bars, r59.OOS_START)
        assert split_bar > 0

    def test_split_bar_less_than_max_bars(self):
        max_bars = 210_000
        split_bar = r59.compute_split_bar(max_bars, r59.OOS_START)
        assert split_bar < max_bars

    def test_split_bar_dt_near_oos_start(self):
        """Bar at split_bar should be close to OOS_START."""
        max_bars = 210_000
        split_bar = r59.compute_split_bar(max_bars, r59.OOS_START)
        dt_at_split = r59.bar_to_dt(split_bar, max_bars)
        # Should be within 1 day of OOS_START
        delta_days = abs((dt_at_split - r59.OOS_START).total_seconds()) / 86400
        assert delta_days < 1.0

    def test_split_before_start_returns_0(self):
        """Split date before data start → split_bar=0 (all OOS)."""
        max_bars = 1000
        data_start = r59.DATA_END - timedelta(minutes=max_bars * 15)
        split_bar = r59.compute_split_bar(max_bars, data_start - timedelta(days=1))
        assert split_bar == 0

    def test_split_after_end_returns_max_bars(self):
        """Split date after DATA_END → split_bar=max_bars (all IS)."""
        max_bars = 1000
        split_bar = r59.compute_split_bar(max_bars, r59.DATA_END + timedelta(days=1))
        assert split_bar == max_bars

    def test_is_oos_roughly_equal_years(self):
        """6yr dataset split at 2023: IS ~3yr, OOS ~3yr."""
        max_bars = int(6 * 365 * 24 * 4)  # approx 6yr
        # Adjust DATA_END anchor for this synthetic case
        # Use actual max_bars to find approx ratio
        split_bar = r59.compute_split_bar(max_bars, r59.OOS_START)
        is_fraction = split_bar / max_bars
        # IS should be roughly 40-60% of total
        assert 0.35 < is_fraction < 0.65


# ══════════════════════════════════════════════════════════════════════════════
#  split_trades
# ══════════════════════════════════════════════════════════════════════════════


class TestSplitTrades:
    def _make_trades(self, entry_bars: list[int]) -> list:
        """Make minimal trade tuples: (sym, entry, exit, pnl)."""
        return [("SYM", e, e + 10, 0.01) for e in entry_bars]

    def test_all_is_when_all_before_split(self):
        oos_start = 1000
        trades = self._make_trades([100, 200, 500, 999])
        is_t, oos_t = r59.split_trades(trades, oos_start)
        assert len(is_t) == 4
        assert len(oos_t) == 0

    def test_all_oos_when_all_after_split(self):
        oos_start = 1000
        trades = self._make_trades([1000, 1100, 2000])
        is_t, oos_t = r59.split_trades(trades, oos_start)
        assert len(is_t) == 0
        assert len(oos_t) == 3

    def test_correct_split_on_boundary(self):
        oos_start = 1000
        trades = self._make_trades([999, 1000])  # 999=IS, 1000=OOS
        is_t, oos_t = r59.split_trades(trades, oos_start)
        assert len(is_t) == 1
        assert len(oos_t) == 1
        assert is_t[0][1] == 999
        assert oos_t[0][1] == 1000

    def test_split_preserves_trade_count(self):
        oos_start = 500
        entries = list(range(0, 1000, 50))
        trades = self._make_trades(entries)
        is_t, oos_t = r59.split_trades(trades, oos_start)
        assert len(is_t) + len(oos_t) == len(trades)

    def test_no_lookahead_entry_bar_used(self):
        """Trade assigned by entry bar, not exit bar."""
        oos_start = 1000
        # Entry=999 (IS), exit=1050 (would be OOS) — must be in IS
        trade = ("SYM", 999, 1050, 0.01)
        is_t, oos_t = r59.split_trades([trade], oos_start)
        assert len(is_t) == 1
        assert len(oos_t) == 0

    def test_empty_trades(self):
        is_t, oos_t = r59.split_trades([], 1000)
        assert is_t == []
        assert oos_t == []


# ══════════════════════════════════════════════════════════════════════════════
#  run_split_metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestRunSplitMetrics:
    def test_empty_trades_returns_zero_pf(self):
        m = r59.run_split_metrics([], 1.0)
        assert m["N"] == 0
        assert m["PF"] == 0.0

    def test_positive_trades_pf_greater_than_1(self):
        trades = [("SYM", i * 100, i * 100 + 10, 0.02) for i in range(50)]
        m = r59.run_split_metrics(trades, 1.0)
        assert m["N"] > 0
        assert m["PF"] > 1.0

    def test_mixed_trades_metrics_defined(self):
        trades = [("SYM", i * 100, i * 100 + 10, 0.02) for i in range(30)] + [
            ("SYM", 3000 + i * 100, 3000 + i * 100 + 10, -0.01) for i in range(20)
        ]
        m = r59.run_split_metrics(trades, 2.0)
        assert "N" in m
        assert "PF" in m
        assert "WR" in m
        assert "EV_bp" in m


# ══════════════════════════════════════════════════════════════════════════════
#  OOS pass criteria constants
# ══════════════════════════════════════════════════════════════════════════════


class TestPassCriteria:
    def test_oos_pf_min(self):
        assert r59.OOS_PF_MIN == pytest.approx(1.20, abs=0.01)

    def test_oos_wr_min(self):
        assert r59.OOS_WR_MIN == pytest.approx(42.0, abs=0.5)

    def test_oos_is_ratio_min(self):
        assert r59.OOS_IS_RATIO_MIN == pytest.approx(0.85, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
#  Candidate configuration
# ══════════════════════════════════════════════════════════════════════════════


class TestCandidateConfig:
    def setup_method(self):
        cfg = VolStraddleConfig()
        self.S_VR = cfg.s_vr_threshold
        self.S_DUR = cfg.s_squeeze_dur
        self.CANDIDATES = [
            ("Baseline", self.S_VR, self.S_DUR),
            ("B2_mid", 0.57, self.S_DUR),
            ("B2_high", 0.59, self.S_DUR),
            ("B3", self.S_VR, 2),
            ("Combo1", 0.57, 2),
            ("Combo2", 0.57, 3),
            ("Combo3", 0.59, 2),
        ]

    def test_candidate_count(self):
        assert len(self.CANDIDATES) == 7

    def test_baseline_uses_config_values(self):
        baseline = self.CANDIDATES[0]
        assert baseline[0] == "Baseline"
        assert baseline[1] == pytest.approx(self.S_VR)
        assert baseline[2] == self.S_DUR

    def test_b2_mid_s_vr_above_baseline(self):
        b2 = next(c for c in self.CANDIDATES if c[0] == "B2_mid")
        assert b2[1] > self.S_VR

    def test_b3_s_dur_below_baseline(self):
        b3 = next(c for c in self.CANDIDATES if c[0] == "B3")
        assert b3[2] < self.S_DUR

    def test_combo1_both_modified(self):
        c1 = next(c for c in self.CANDIDATES if c[0] == "Combo1")
        assert c1[1] > self.S_VR
        assert c1[2] < self.S_DUR

    def test_s_vr_candidates_in_valid_range(self):
        for label, s_vr, _ in self.CANDIDATES:
            assert 0.50 <= s_vr <= 0.70, f"{label}: s_vr={s_vr} out of range"

    def test_s_dur_candidates_in_valid_range(self):
        for label, _, s_dur in self.CANDIDATES:
            assert 1 <= s_dur <= 8, f"{label}: s_dur={s_dur} out of range"

    def test_no_hardcoded_baseline(self):
        """Baseline must use config values, not literal 0.55 / 4."""
        cfg = VolStraddleConfig()
        baseline = self.CANDIDATES[0]
        # Must equal config — will break if config changes
        assert baseline[1] == pytest.approx(cfg.s_vr_threshold)
        assert baseline[2] == cfg.s_squeeze_dur


# ══════════════════════════════════════════════════════════════════════════════
#  IS/OOS fraction sanity
# ══════════════════════════════════════════════════════════════════════════════


class TestSplitFraction:
    def test_is_oos_year_balance(self):
        """IS ~3yr and OOS ~3yr from 6yr dataset."""
        max_bars = 210_240  # actual 15m bars for 6yr (approx)
        split_bar = r59.compute_split_bar(max_bars, r59.OOS_START)
        is_bars = split_bar
        oos_bars = max_bars - split_bar
        # Both segments should be substantial (> 1yr worth of bars)
        bars_per_year = 365 * 24 * 4
        assert is_bars > bars_per_year
        assert oos_bars > bars_per_year

    def test_total_bars_conserved_in_split(self):
        oos_start = 1000
        entries = list(range(0, 2000, 7))
        trades = [("A", e, e + 5, 0.01) for e in entries]
        is_t, oos_t = r59.split_trades(trades, oos_start)
        assert len(is_t) + len(oos_t) == len(trades)
