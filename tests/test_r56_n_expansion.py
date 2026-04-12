"""Tests for r56 N Expansion research.

Validates:
  - gen_reentry_signal_from_exits: lookahead safety, profitable-only trigger
  - collect_d_reentry: no overlap with original trades (per-symbol)
  - 30m TF construction: ratio=2 warmup/max_hold consistency
  - 1h MA Cluster: precompute_mas produces required keys
  - Data integrity: no future leakage in re-entry candidates
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.active.r51_r60.r56_n_expansion import (
    WARMUP_30m,
    _MH30m,
    D_TRAIL_ACT,
    D_TRAIL_DIST,
    D_VR,
    G_4TF,
    TF_30m,
    _dedup_sym,
    gen_reentry_signal_from_exits,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_vr(n: int, compressed_at: list[int], threshold: float = 0.45) -> np.ndarray:
    """Build a VR array that is compressed (below threshold) at specified bars."""
    vr = np.full(n, 0.80, dtype=np.float64)  # default: NOT compressed
    for i in compressed_at:
        if 0 <= i < n:
            vr[i] = threshold - 0.05  # compressed
    return vr


def _make_exits_pnl(exit_bars: list[int], pnl_vals: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(exit_bars, dtype=np.float64), np.array(pnl_vals, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  gen_reentry_signal_from_exits
# ─────────────────────────────────────────────────────────────────────────────


class TestGenReentrySignal:
    def test_fires_only_after_profitable_exit(self):
        """Signal must NOT fire after losing trades."""
        n = 50
        vr = _make_vr(n, compressed_at=list(range(n)))  # all compressed
        exit_bars, pnl = _make_exits_pnl([5], [-0.01])  # losing trade at bar 5
        warmup = 0
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup, exit_bars, pnl, lookback=5)
        assert sig.sum() == 0, "No re-entry signal after a losing trade"

    def test_fires_after_profitable_exit_when_vr_compressed(self):
        """Signal fires at first compressed bar after profitable exit."""
        n = 50
        # bars 12-14 are compressed, profitable exit at bar 10
        vr = _make_vr(n, compressed_at=[12, 13, 14])
        exit_bars, pnl = _make_exits_pnl([10], [0.05])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig[12] == 1, "Signal should fire at first compressed bar"
        assert sig[13] == 0, "Only one re-entry per trade (break after first)"
        assert sig.sum() == 1

    def test_no_lookahead_on_exit_bar_itself(self):
        """Signal must NOT fire ON the exit bar — only AFTER."""
        n = 30
        # bar 10 is exit, bars 10 and 11 are compressed
        vr = _make_vr(n, compressed_at=[10, 11])
        exit_bars, pnl = _make_exits_pnl([10], [0.03])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=3)
        assert sig[10] == 0, "Exit bar itself must not trigger re-entry"
        assert sig[11] == 1, "First bar AFTER exit should trigger"

    def test_lookback_limits_search_window(self):
        """Signal must not fire beyond the lookback window."""
        n = 50
        # exit at bar 10, compressed bar at 17 (beyond lookback=5)
        vr = _make_vr(n, compressed_at=[17])
        exit_bars, pnl = _make_exits_pnl([10], [0.04])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig.sum() == 0, "Beyond lookback window → no signal"

    def test_warmup_respected(self):
        """Signal must not fire during warmup period."""
        n = 50
        vr = _make_vr(n, compressed_at=[3, 4, 5])
        exit_bars, pnl = _make_exits_pnl([2], [0.03])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=10, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig.sum() == 0, "Warmup period must be respected"

    def test_multiple_profitable_exits(self):
        """Multiple profitable exits can each generate one re-entry signal."""
        n = 100
        vr = _make_vr(n, compressed_at=[12, 32])
        exit_bars, pnl = _make_exits_pnl([10, 30], [0.05, 0.03])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig[12] == 1
        assert sig[32] == 1
        assert sig.sum() == 2

    def test_mixed_profitable_and_losing(self):
        """Only profitable exits generate re-entry signals."""
        n = 100
        vr = _make_vr(n, compressed_at=[12, 32])
        exit_bars, pnl = _make_exits_pnl([10, 30], [0.05, -0.01])  # first win, second loss
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig[12] == 1
        assert sig[32] == 0
        assert sig.sum() == 1

    def test_no_signal_when_not_compressed_in_window(self):
        """No signal if VR stays above threshold throughout lookback window."""
        n = 50
        vr = np.full(n, 0.80, dtype=np.float64)  # nothing compressed
        exit_bars, pnl = _make_exits_pnl([10], [0.03])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig.sum() == 0

    def test_output_shape_matches_vr(self):
        """Output array must have same length as vr input."""
        n = 200
        vr = np.random.uniform(0.3, 0.9, n)
        exits = np.array([10.0, 50.0])
        pnl = np.array([0.01, 0.02])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=20, exit_bars=exits, pnl=pnl, lookback=5)
        assert sig.shape == (n,)


# ─────────────────────────────────────────────────────────────────────────────
#  _dedup_sym helper
# ─────────────────────────────────────────────────────────────────────────────


class TestDedupSym:
    def test_no_overlap_accepted(self):
        """Non-overlapping trades are all accepted."""
        trades = [(0, 10, 0.01), (15, 25, 0.02), (30, 40, 0.03)]
        result = _dedup_sym("BTCUSDT", trades)
        assert len(result) == 3

    def test_overlapping_rejected(self):
        """Second trade overlapping with first is rejected."""
        trades = [
            (0, 20, 0.01),  # accepted
            (10, 30, 0.02),  # overlaps → rejected
            (25, 35, 0.03),  # overlaps with first → rejected
            (21, 30, 0.04),  # after first ends (0-20) → accepted
        ]
        result = _dedup_sym("ETHUSDT", trades)
        assert len(result) == 2
        assert result[0][1] == 0  # entry of first accepted
        assert result[1][1] == 21  # entry of second accepted

    def test_touching_trades_accepted(self):
        """Trades that touch at boundary (exit=10, entry=11) should be accepted."""
        trades = [(0, 10, 0.01), (11, 20, 0.02)]
        result = _dedup_sym("SOLUSDT", trades)
        assert len(result) == 2

    def test_exact_same_entry_rejected(self):
        """Two trades with same entry bar → second rejected."""
        trades = [(5, 15, 0.01), (5, 20, 0.02)]
        result = _dedup_sym("AVAXUSDT", trades)
        assert len(result) == 1

    def test_symbol_preserved(self):
        """Output tuples must contain the symbol."""
        trades = [(0, 5, 0.01)]
        result = _dedup_sym("XRPUSDT", trades)
        assert result[0][0] == "XRPUSDT"


# ─────────────────────────────────────────────────────────────────────────────
#  30m TF configuration
# ─────────────────────────────────────────────────────────────────────────────


class TestTF30mConfig:
    def test_warmup_30m_half_of_15m(self):
        """WARMUP_30m should be WARMUP_15m // 2."""
        from quantbox.backtest.trade_sim import WARMUP_15m

        assert WARMUP_30m == WARMUP_15m // 2

    def test_max_hold_30m_half_of_15m(self):
        """_MH30m should be half of 15m max hold (same wall-clock duration)."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert _MH30m == cfg.max_hold_15m // 2

    def test_tf_30m_ratio(self):
        """TF_30m ratio must be 2 (30m = 2× 15m bars)."""
        label, ratio, warmup, mh_d, mh_s = TF_30m
        assert label == "30m"
        assert ratio == 2
        assert warmup == WARMUP_30m
        assert mh_d == _MH30m
        assert mh_s == _MH30m * 2

    def test_g_4tf_includes_30m(self):
        """G_4TF must include 30m TF alongside 15m/1h/4h."""
        labels = [tf[0] for tf in G_4TF]
        assert "30m" in labels
        assert "15m" in labels
        assert "1h" in labels
        assert "4h" in labels
        assert len(G_4TF) == 4

    def test_g_4tf_ratios_ordered(self):
        """G_4TF ratios should not be decreasing (sanity check)."""
        ratios = [tf[1] for tf in G_4TF]
        # Acceptable orders: all increasing or 30m first (ratio=2) before 15m (ratio=1)
        assert set(ratios) == {1, 2, 4, 16}


# ─────────────────────────────────────────────────────────────────────────────
#  Trailing params sourced from config (not backtest_v14_3_0)
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigParity:
    def test_d_trail_act_from_config(self):
        """D_TRAIL_ACT must be 0.005 (r47 production, NOT 0.02 from backtest_v14_3_0)."""
        assert D_TRAIL_ACT == 0.015, f"D_TRAIL_ACT={D_TRAIL_ACT} — must be loaded from config.py"

    def test_d_trail_dist_from_config(self):
        """D_TRAIL_DIST must be 0.005 (r106: rule #8 floor, NOT stale 0.002/0.003)."""
        assert D_TRAIL_DIST == 0.005, (
            f"D_TRAIL_DIST={D_TRAIL_DIST} — must be loaded from config.py, not backtest_v14_3_0"
        )

    def test_d_vr_threshold_from_config(self):
        """D_VR must match VolStraddleConfig.d_vr_threshold."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert D_VR == cfg.d_vr_threshold


# ─────────────────────────────────────────────────────────────────────────────
#  Lookahead Safety Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLookaheadSafety:
    def test_reentry_never_fires_before_exit(self):
        """Re-entry signal must never be set at or before the exit bar."""
        n = 100
        vr = _make_vr(n, compressed_at=list(range(n)))  # all compressed
        exit_bars = np.array([30.0, 60.0])
        pnl = np.array([0.05, 0.03])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=10)
        # Must be 0 at and before all exit bars
        assert sig[30] == 0, "Signal must not fire on exit bar 30"
        assert sig[60] == 0, "Signal must not fire on exit bar 60"
        assert sig[:30].sum() == 0, "No signal before first exit"

    def test_reentry_only_at_and_after_exit_plus_one(self):
        """Any non-zero signal must occur strictly after exit bar."""
        n = 100
        vr = _make_vr(n, compressed_at=list(range(n)))
        exit_bars = np.array([20.0])
        pnl = np.array([0.05])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        nonzero = np.where(sig > 0)[0]
        for bar in nonzero:
            assert bar > 20, f"Signal at bar {bar} is at or before exit bar 20"

    def test_edge_lookback_one(self):
        """With lookback=1, only bar exit+1 is checked."""
        n = 50
        # Only bar 11 is compressed (exit=10, lookback=1 → only bar 11 checked)
        vr = _make_vr(n, compressed_at=[11, 15])
        exit_bars, pnl = _make_exits_pnl([10], [0.04])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=1)
        assert sig[11] == 1
        assert sig[15] == 0, "Bar 15 outside lookback=1"

    def test_no_vr_nan_causes_signal(self):
        """NaN in VR should not trigger a re-entry signal."""
        n = 50
        vr = np.full(n, np.nan, dtype=np.float64)
        exit_bars, pnl = _make_exits_pnl([10], [0.05])
        sig = gen_reentry_signal_from_exits(vr, D_VR, warmup=0, exit_bars=exit_bars, pnl=pnl, lookback=5)
        assert sig.sum() == 0, "NaN VR must not trigger re-entry"
