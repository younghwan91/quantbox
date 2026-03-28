"""Tests for SL/trail exit priority in trailing stop sim functions.

G5 parity fix (refactor/mega-v2): once trailing activates, trail level is
always tighter than the original SL. On the exchange, both are independent
server-side orders — the tighter one fires first. The backtest must match:
when trail is active, check trail first (skip SL since it's below trail).

When trail is NOT active, SL fires normally.

Scenarios:
  1. Same-bar SL+trail trigger → trail wins (tighter stop)
  2. SL fires normally before trail activates
  3. Trail fires cleanly after activation
  4. Crash through both levels after trail active → trail level exit (not SL)
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.backtest.trade_sim import (
    sim_directional_trailing,
    sim_straddle_trailing,
)


def _make_bars(
    bars: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create close/high/low arrays from (open, high, low, close) tuples."""
    c = np.array([b[3] for b in bars], dtype=np.float64)
    h = np.array([b[1] for b in bars], dtype=np.float64)
    lo = np.array([b[2] for b in bars], dtype=np.float64)
    return c, h, lo


# ══════════════════════════════════════════════════════════════════════════════
#  Straddle — Trail Priority Tests (G5 parity fix)
# ══════════════════════════════════════════════════════════════════════════════


class TestStraddleTrailPriority:
    """When trail is active, trail level (tighter) wins over SL."""

    SL = 0.015  # 1.5%
    TP = 0.20
    MAX_HOLD = 192
    COST = 0.0016
    TRAIL_ACT = 0.015  # 1.5% (== SL)
    TRAIL_DIST = 0.001  # 0.1%

    def test_same_bar_sl_and_trail_trail_wins(self):
        """Bar has high >= trail_activate AND low <= SL → trail wins (tighter)."""
        # Bar 0: entry at 100
        # Bar 1: high=101.6 (+1.6%), low=98.4 (-1.6%)
        # LONG: peak=0.016 > trail_activate → trail_level=101.5, low < trail → trail fires at 0.015
        # SHORT: peak=0.016 > trail_activate → trail_level=98.5, high > trail → trail fires at 0.015
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # bar 0: entry
            (100.0, 101.6, 98.4, 100.0),  # bar 1: both SL and trail trigger
            (100.0, 100.0, 100.0, 100.0),  # bar 2: filler
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])

        pnl, ent, ext = sim_straddle_trailing(
            close,
            high,
            low,
            sig,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        # Half-size per leg: trail fires at max(peak - trail_dist, -sl) = 0.015
        long_trail = max(0.016 - self.TRAIL_DIST, -self.SL)  # 0.015
        short_trail = max(0.016 - self.TRAIL_DIST, -self.SL)  # 0.015
        expected = 0.5 * (long_trail + short_trail - self.COST)
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_sl_fires_before_trail_activates(self):
        """SL fires normally when trail hasn't activated yet."""
        # Bar 1: low touches SL but high doesn't reach trail_activate
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            (100.0, 100.5, 98.4, 99.0),  # high=+0.5% < trail_activate=1.5%, low < SL
            (99.0, 99.0, 99.0, 99.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_straddle_trailing(
            close,
            high,
            low,
            sig,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        # LONG: SL fires (trail not active), SHORT: close at bar 1
        assert pnl[0] < 0, f"Expected loss when SL fires before trail activation, got {pnl[0]}"

    def test_trail_fires_when_sl_not_hit(self):
        """Trail fires normally when bar high/low doesn't reach SL."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            (100.0, 102.0, 101.0, 101.5),  # LONG: +2% gain, trail activates
            # peak=0.02, trail_lvl=100*(1+0.02-0.001)=101.9
            # low=101.3 < 101.9 → trail hit at 0.019
            (101.5, 102.1, 101.3, 101.5),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_straddle_trailing(
            close,
            high,
            low,
            sig,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        assert pnl[0] > 0.0, f"Expected positive trail exit, got {pnl[0]}"

    def test_crash_through_both_after_trail_active_trail_wins(self):
        """Crash through SL after trail active → trail level exit (not SL)."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            # Bar 1: LONG gains +1.6%, trail activates. Low stays above trail_lvl.
            # trail_lvl = 100*(1+0.016-0.001) = 101.5, low=101.51 > 101.5
            # SHORT: high=101.6 > 100*(1+0.015)=101.5 → SHORT trail activates too
            (101.0, 101.6, 101.51, 101.55),
            # Bar 2: crash — low=98.0 (below SL=98.5).
            # With G5 fix: trail is active, trail fires (tighter than SL).
            # LONG: peak still 0.016, trail_lvl=101.5, low=98 < trail → trail at 0.015
            (101.55, 101.6, 98.0, 99.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])

        pnl, _, ext = sim_straddle_trailing(
            close,
            high,
            low,
            sig,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        # LONG: trail at max(0.016-0.001, -0.015) = 0.015
        # SHORT: SL at bar 1 (high=101.6 > 101.5) — wait, SHORT trail also activates
        # SHORT peak from bar 1: low=101.51, but entry=100, short gain=(100-101.51)/100=-0.0151 (negative)
        # Actually SHORT: gain = (entry - low) / entry. low=101.51 → gain = (100-101.51)/100 = -0.0151 < 0
        # So SHORT peak_gain=0, trail NOT active. SHORT SL: high=101.6 > 100*(1+0.015)=101.5 → SL fires bar 1
        # Net: LONG trail(0.015) + SHORT SL(-0.015) - cost
        # This is approximately break-even minus cost
        assert pnl[0] > -0.01, f"Expected trail to mitigate crash, got {pnl[0]}"

    def test_massive_range_bar_trail_wins_both_legs(self):
        """Massive range bar: both legs trail (not double SL)."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # high=103 (+3%), low=97 (-3%). Both legs: peak > trail_activate
            (100.0, 103.0, 97.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_straddle_trailing(
            close,
            high,
            low,
            sig,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        # Half-size: 0.5*0.029 + 0.5*0.029 - cost ≈ +2.7%
        assert pnl[0] > 0.025, f"Expected positive from trail on both legs, got {pnl[0]}"

    def test_no_conflict_bars_produce_correct_results(self):
        """Normal scenario: trail activates and fires cleanly on separate bars."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            (100.0, 100.5, 99.8, 100.3),
            (100.3, 102.0, 100.2, 101.8),  # LONG: trail activates
            # peak=0.02, trail_lvl=100*(1+0.02-0.001)=101.9
            # low=101.0 < 101.9 → trail fires
            (101.8, 102.0, 101.0, 101.5),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0, 0.0])

        pnl, _, _ = sim_straddle_trailing(
            close,
            high,
            low,
            sig,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        assert pnl[0] > -self.COST, f"Expected non-negative trail result, got {pnl[0]}"


# ══════════════════════════════════════════════════════════════════════════════
#  Directional — Trail Priority Tests (G5 parity fix)
# ══════════════════════════════════════════════════════════════════════════════


class TestDirectionalTrailPriority:
    """When trail is active, trail level (tighter) wins over SL."""

    SL = 0.01  # 1%
    TP = 0.20
    MAX_HOLD = 192
    COST = 0.0008
    TRAIL_ACT = 0.01  # 1%
    TRAIL_DIST = 0.002  # 0.2%

    def test_same_bar_sl_and_trail_long_trail_wins(self):
        """LONG: bar has high >= trail_activate AND low <= SL → trail wins."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # high=101.1 (+1.1%, above trail_activate=1.0%)
            # low=98.9 (-1.1%, below SL=1.0% at 99.0)
            # Trail activates, trail_level=100*(1+0.011-0.002)=100.9
            # low=98.9 < trail_level → trail fires at max(0.009, -0.01) = 0.009
            (100.0, 101.1, 98.9, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])
        dire = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        # Trail wins: max(0.011 - 0.002, -0.01) = 0.009
        expected = 0.009 - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_same_bar_sl_and_trail_short_trail_wins(self):
        """SHORT: same-bar SL+trail → trail wins."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            # low=98.9 (+1.1% gain for SHORT), high=101.1 (above SL at 101.0)
            # Trail: peak=0.011, trail_level=100*(1-0.011+0.002)=99.1
            # high=101.1 > trail_level → trail fires at max(0.009, -0.01) = 0.009
            (100.0, 101.1, 98.9, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])
        dire = np.array([-1.0, 0.0, 0.0])

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        expected = 0.009 - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_sl_fires_before_trail_activates(self):
        """SL fires normally when trail hasn't activated yet."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            # high=100.5 (+0.5% < trail_activate=1%), low=98.9 (below SL at 99.0)
            (100.0, 100.5, 98.9, 99.0),
            (99.0, 99.0, 99.0, 99.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])
        dire = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        expected = -self.SL - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected SL exit {expected}, got {pnl[0]}"

    def test_trail_fires_when_sl_not_hit_long(self):
        """LONG trail fires normally when SL level not breached."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.5, 99.5, 101.2),  # +1.5% gain, trail activates
            # peak=0.015, trail_lvl=100*(1+0.015-0.002)=101.3
            # low=101.0 < 101.3 → trail hit
            (101.2, 101.5, 101.0, 101.2),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])
        dire = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        expected = 0.013 - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_trail_fires_when_sl_not_hit_short(self):
        """SHORT trail fires normally when SL level not breached."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.5, 98.5, 98.8),  # +1.5% gain for SHORT
            # peak=0.015, trail_lvl=100*(1-0.015+0.002)=98.7
            # high=99.0 > 98.7 → trail hit
            (98.8, 99.0, 98.5, 98.8),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])
        dire = np.array([-1.0, 0.0, 0.0])

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        expected = 0.013 - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_crash_after_trail_active_trail_wins(self):
        """Crash through SL after trail active → trail level exit (not -sl)."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            # Bar 1: +1.1% gain → trail activates.
            # trail_lvl = 100*(1+0.011-0.002) = 100.9, low=101.01 > 100.9 → no fire
            (100.0, 101.1, 101.01, 101.05),
            # Bar 2: crash — low=98.5 (below SL=99.0).
            # Trail active: trail_lvl=100.9 (from bar 1 peak), low=98.5 < 100.9
            # Trail fires at max(0.011-0.002, -0.01) = 0.009 (NOT -sl)
            (101.05, 101.1, 98.5, 99.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.array([1.0, 0.0, 0.0])
        dire = np.array([1.0, 0.0, 0.0])

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        # G5 fix: trail wins → positive PnL (not -sl)
        expected = max(0.011 - self.TRAIL_DIST, -self.SL) - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Trail should fire (tighter than SL): expected {expected}, got {pnl[0]}"

    def test_backward_compat_no_conflict(self):
        """Trades without SL/trail conflicts produce identical results."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.3, 99.7, 100.0),
            (100.0, 100.5, 99.5, 100.2),
            (100.2, 100.8, 99.8, 100.5),
            (100.5, 101.2, 100.0, 101.0),
            # peak=0.012 > trail_activate=0.01 → trail active
            # trail_lvl = 100*(1+0.012-0.002) = 101.0
            # low=100.5, not below trail_lvl
            (101.0, 101.3, 100.5, 101.1),
            # trail_lvl = 100*(1+0.013-0.002) = 101.1
            # low=100.8 < 101.1 → trail fires at 0.011
            (101.1, 101.3, 100.8, 101.0),
        ]
        close, high, low = _make_bars(bars)
        sig = np.zeros(len(bars))
        sig[0] = 1.0
        dire = np.zeros(len(bars))
        dire[0] = 1.0

        pnl, _, _ = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            self.SL,
            self.TP,
            self.MAX_HOLD,
            self.COST,
            self.TRAIL_ACT,
            self.TRAIL_DIST,
        )

        assert len(pnl) == 1
        assert pnl[0] > 0, f"Expected positive trail exit, got {pnl[0]}"
