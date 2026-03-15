"""Tests for SL-first priority in trailing stop sim functions.

Validates that server SL fires BEFORE client-side trail when both
conditions are met in the same bar. This matches live behavior where
server SL is intra-bar (instant) and trail fires at bar close.

Root cause fix for trail-before-SL bias discovered in H5 investigation.
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
#  Straddle — SL Priority Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestStraddleSLPriority:
    """SL must fire before trail in sim_straddle_trailing."""

    SL = 0.015  # 1.5%
    TP = 0.20
    MAX_HOLD = 192
    COST = 0.0016
    TRAIL_ACT = 0.015  # 1.5% (== SL)
    TRAIL_DIST = 0.001  # 0.1%

    def test_same_bar_sl_and_trail_long_leg_sl_wins(self):
        """Bar has high >= trail_activate AND low <= SL → LONG SL fires."""
        # Bar 0: entry at 100
        # Bar 1: high=101.6 (+1.6%, above trail_activate=1.5%)
        #         low=98.4 (-1.6%, below SL=1.5% at 98.5)
        # SL should fire for LONG, not trail
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
        # LONG leg: SL = -1.5%, SHORT leg: high=101.6 → SL at +1.5% → -1.5%
        # Both legs SL → net = -1.5% + -1.5% - cost
        long_sl = -self.SL
        short_sl = -self.SL  # high=101.6 > 100*(1+0.015)=101.5
        expected = long_sl + short_sl - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_same_bar_sl_and_trail_short_leg_sl_wins(self):
        """Bar has low >= trail_activate (SHORT) AND high >= SL → SHORT SL fires."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            (100.0, 101.6, 98.4, 100.0),  # SL+trail for both legs
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

        # Both legs hit SL
        assert pnl[0] < -0.02, f"Expected double-SL loss, got {pnl[0]}"

    def test_trail_fires_when_sl_not_hit(self):
        """Trail fires normally when bar high/low doesn't reach SL."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # LONG: high=102 (+2%, activates trail), low=101 (above SL=98.5)
            (100.0, 102.0, 101.0, 101.5),
            # LONG: high=102.1, low=101.3 → trail_lvl = 100*(1+0.02-0.001)=101.9
            # low=101.3 < 101.9 → trail hit at +1.9%
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
        # LONG exits via trail ~+1.9%, SHORT still active, closes at bar 2 close
        # Net should be positive (trail win > SL loss)
        # LONG trail: peak=0.021 → trail = max(0.021-0.001, -0.015) = 0.020
        # SHORT: close at 101.5 → (100-101.5)/100 = -0.015 → SL at bar 1
        # Actually SHORT high=102 > 100*(1+0.015)=101.5 → SL at bar 1
        # Net = 0.020 + (-0.015) - 0.0016 = 0.0034
        assert pnl[0] > 0.0, f"Expected positive trail exit, got {pnl[0]}"

    def test_double_sl_straddle_produces_large_loss(self):
        """When both legs SL, loss should be ~-3% + cost (not hidden by trail)."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # Massive range bar: high=103, low=97 → both legs SL
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
        expected_loss = -self.SL * 2 - self.COST
        assert abs(pnl[0] - expected_loss) < 1e-10, f"Expected {expected_loss:.4f}, got {pnl[0]:.4f}"

    def test_sl_fires_even_when_trail_was_active(self):
        """If trail was active from previous bars but current bar hits SL, SL wins."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # Bar 1: LONG peak=+1.6% → trail activates. Low stays above trail_lvl.
            # trail_lvl = 100*(1+0.016-0.001) = 101.5, low=101.51 > 101.5 → no trail fire
            # SHORT: high=101.6 > 100*(1+0.015)=101.5 → SHORT SL fires
            (101.0, 101.6, 101.51, 101.55),
            # Bar 2: crash — low=98.0 (below SL=98.5). SL fires for LONG
            # despite trail being active.
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
        # LONG: SL at bar 2 → le = -0.015
        # SHORT: SL at bar 1 → se = -0.015
        # net = -0.015 + (-0.015) - cost = -0.0316
        assert pnl[0] < -0.02, f"Expected SL loss, got {pnl[0]}"

    def test_no_conflict_bars_produce_correct_results(self):
        """Normal scenario: trail activates and fires cleanly on separate bars."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # Bar 1: small up, no SL, no trail activate
            (100.0, 100.5, 99.8, 100.3),
            # Bar 2: LONG gains +2%, trail activates
            (100.3, 102.0, 100.2, 101.8),
            # Bar 3: price dips below trail level but ABOVE SL
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
        # Trail exit is profitable
        assert pnl[0] > -self.COST, f"Expected non-negative trail result, got {pnl[0]}"


# ══════════════════════════════════════════════════════════════════════════════
#  Directional — SL Priority Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDirectionalSLPriority:
    """SL must fire before trail in sim_directional_trailing."""

    SL = 0.01  # 1%
    TP = 0.20
    MAX_HOLD = 192
    COST = 0.0008
    TRAIL_ACT = 0.01  # 1%
    TRAIL_DIST = 0.002  # 0.2%

    def test_same_bar_sl_and_trail_long_sl_wins(self):
        """LONG: bar has high >= trail_activate AND low <= SL → SL fires."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # high=101.1 (+1.1%, above trail_activate=1.0%)
            # low=98.9 (-1.1%, below SL=1.0% at 99.0)
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
        expected = -self.SL - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_same_bar_sl_and_trail_short_sl_wins(self):
        """SHORT: bar has low showing gain >= trail AND high >= SL → SL fires."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # low=98.9 (-1.1% = +1.1% gain for SHORT, above trail_activate)
            # high=101.1 (+1.1%, above SL=1.0% at 101.0 for SHORT)
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
        expected = -self.SL - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_trail_fires_when_sl_not_hit_long(self):
        """LONG trail fires normally when SL level not breached."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),  # entry
            # Bar 1: +1.5% gain, activates trail, no SL
            (100.0, 101.5, 99.5, 101.2),
            # Bar 2: dips to trail level but above SL
            # peak=0.015, trail_lvl = 100*(1+0.015-0.002) = 101.3
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
        # Trail exit: peak_gain=0.015, pnl = 0.015 - 0.002 = 0.013
        expected = 0.013 - self.COST
        assert abs(pnl[0] - expected) < 1e-10, f"Expected {expected}, got {pnl[0]}"

    def test_trail_fires_when_sl_not_hit_short(self):
        """SHORT trail fires normally when SL level not breached."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            # Bar 1: -1.5% = +1.5% gain for SHORT, activates trail
            (100.0, 100.5, 98.5, 98.8),
            # Bar 2: bounces to trail level but below SL
            # peak=0.015, trail_lvl = 100*(1-0.015+0.002) = 98.7
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

    def test_sl_fires_even_when_trail_active(self):
        """SL fires even after trail was activated on a previous bar."""
        bars = [
            (100.0, 100.0, 100.0, 100.0),
            # Bar 1: +1.1% gain → trail activates (>=1.0%). Low stays above trail_lvl.
            # trail_lvl = 100*(1+0.011-0.002) = 100.9, low=101.01 > 100.9 → no trail fire
            (100.0, 101.1, 101.01, 101.05),
            # Bar 2: crash — low=98.5 (below SL=99.0). SL should fire
            # despite trail being active.
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
        expected = -self.SL - self.COST
        assert abs(pnl[0] - expected) < 1e-10, (
            f"SL should fire even with trail active: expected {expected}, got {pnl[0]}"
        )

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
        # Trail exit, no SL conflict
        assert pnl[0] > 0, f"Expected positive trail exit, got {pnl[0]}"
