"""Tests for trailing stop exit logic.

Validates that directional trades use trail (activate→distance→hard TP)
and MA cluster uses ma_cluster_trail params.
"""

from __future__ import annotations

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy


@pytest.fixture
def cfg() -> VolStraddleConfig:
    """Default config with trailing stop params (r106: rule #8 floor applied)."""
    return VolStraddleConfig(
        d_sl=0.01,  # 1%
        d_tp=0.12,  # 12% hard ceiling
        d_trail_activate=0.02,  # activate after +2%
        d_trail_distance=0.005,  # r106: 0.5% behind peak (rule #8: callbackRate >= 0.5%)
        trail_server_native=False,  # test client-side trail logic
    )


@pytest.fixture
def strat(cfg: VolStraddleConfig) -> VolStraddleStrategy:
    return VolStraddleStrategy(cfg)


def _open_long(strat: VolStraddleStrategy, symbol: str = "BTCUSDT", entry: float = 100.0) -> None:
    strat.open_position(
        symbol=symbol,
        signal=1,
        entry_price=entry,
        notional=1000.0,
        leverage=25,
        move_size=1.0,
        sl=strat.config.d_sl,
        tp=strat.config.d_tp,
        max_hold=192,
    )


def _open_short(strat: VolStraddleStrategy, symbol: str = "BTCUSDT", entry: float = 100.0) -> None:
    strat.open_position(
        symbol=symbol,
        signal=-1,
        entry_price=entry,
        notional=1000.0,
        leverage=25,
        move_size=1.0,
        sl=strat.config.d_sl,
        tp=strat.config.d_tp,
        max_hold=192,
    )


class TestTrailingStopActivation:
    """Trail activates after peak gain reaches d_trail_activate."""

    def test_trail_not_active_before_threshold(self, strat: VolStraddleStrategy):
        """Trail should NOT activate below 2% gain."""
        _open_long(strat)
        # Price rises 1.5% — below 2% activate threshold
        result = strat.update_position("BTCUSDT", 101.5, current_high=101.5, current_low=100.5)
        assert result is None
        pos = strat._positions["BTCUSDT"]
        assert pos["trail_active"] is False

    def test_trail_activates_at_threshold(self, strat: VolStraddleStrategy):
        """Trail activates when peak gain >= 2%."""
        _open_long(strat)
        # Price rises 2.5% — above 2% activate threshold
        # Low must stay above trail_stop (2.5%-0.5%=2.0% → $102.0)
        result = strat.update_position("BTCUSDT", 102.5, current_high=102.5, current_low=102.3)
        assert result is None
        pos = strat._positions["BTCUSDT"]
        assert pos["trail_active"] is True
        assert pos["peak_gain"] >= 0.02

    def test_trail_activates_short(self, strat: VolStraddleStrategy):
        """Trail activates for short position too."""
        _open_short(strat)
        # Price drops 2.5%, high stays above trail stop
        # peak_gain=2.5%, trail_stop=2.0%, trail_level=100*(1-0.025+0.005)=97.8
        # high must be < 97.8 to NOT trigger trail
        result = strat.update_position("BTCUSDT", 97.5, current_high=97.7, current_low=97.5)
        assert result is None
        pos = strat._positions["BTCUSDT"]
        assert pos["trail_active"] is True


class TestTrailingStopExit:
    """Trail fires when price retraces trail_distance from peak."""

    def test_trail_exit_long(self, strat: VolStraddleStrategy):
        """Long: peak at +3%, retrace 0.5% from peak → trail hit."""
        _open_long(strat, entry=100.0)
        # Bar 1: push to +3% peak, low above trail stop
        # trail_stop_gain = 3% - 0.5% = 2.5%  → trail_level = $102.5
        strat.update_position("BTCUSDT", 103.0, current_high=103.0, current_low=102.8)
        assert strat._positions["BTCUSDT"]["trail_active"] is True

        # Bar 2: retrace — low touches trail stop
        # low = 102.4 < 102.5 → trail hit
        result = strat.update_position("BTCUSDT", 102.6, current_high=102.9, current_low=102.4)
        assert result == "trailing_stop"

    def test_trail_exit_short(self, strat: VolStraddleStrategy):
        """Short: peak at +3%, retrace 0.5% from peak → trail hit."""
        _open_short(strat, entry=100.0)
        # Bar 1: push to +3% peak (price drops), high below trail stop
        # trail_stop_gain = 2.5%, trail_level = 100*(1-0.03+0.005) = 97.5
        strat.update_position("BTCUSDT", 97.0, current_high=97.2, current_low=97.0)
        assert strat._positions["BTCUSDT"]["trail_active"] is True

        # Bar 2: retrace — high touches trail
        # high = 97.6 > 97.5 → trail hit
        result = strat.update_position("BTCUSDT", 97.2, current_high=97.6, current_low=97.0)
        assert result == "trailing_stop"

    def test_no_trail_exit_when_still_above(self, strat: VolStraddleStrategy):
        """Trail doesn't fire if retrace is less than distance."""
        _open_long(strat, entry=100.0)
        # Push to +3%, keep low above trail stop
        strat.update_position("BTCUSDT", 103.0, current_high=103.0, current_low=102.8)
        assert strat._positions["BTCUSDT"]["trail_active"] is True
        # Retrace but low is still above trail stop (102.7)
        result = strat.update_position("BTCUSDT", 102.9, current_high=103.0, current_low=102.8)
        assert result is None


class TestSLBeforeTrail:
    """SL fires normally before trail activates."""

    def test_sl_hit_before_trail(self, strat: VolStraddleStrategy):
        """SL hits at -1% before trail can activate."""
        _open_long(strat, entry=100.0)
        result = strat.update_position("BTCUSDT", 99.0, current_high=100.2, current_low=98.9)
        assert result == "stop_loss"

    def test_sl_disabled_after_trail_active(self, strat: VolStraddleStrategy):
        """Once trail is active, original SL does not fire."""
        _open_long(strat, entry=100.0)
        # Activate trail at +3%, low above trail stop
        strat.update_position("BTCUSDT", 103.0, current_high=103.0, current_low=102.8)
        assert strat._positions["BTCUSDT"]["trail_active"] is True

        # Now a bar with low at original SL level (-1%).
        # Trail fires as 'trailing_stop', not 'stop_loss'.
        # trail_stop_gain = 3% - 0.5% = 2.5%
        # worst_gain = (99.0 - 100) / 100 = -1% < 2.5% → trail fires
        result = strat.update_position("BTCUSDT", 99.0, current_high=100.0, current_low=99.0)
        assert result == "trailing_stop"


class TestHardTP:
    """Hard TP at 12% fires as safety ceiling."""

    def test_hard_tp_fires(self, strat: VolStraddleStrategy):
        """Price reaches 12% → hard TP fires (when trail not hit)."""
        _open_long(strat, entry=100.0)
        # Gradual smooth rise — low always above trail stop
        # Bar 1: +5%, trail activates
        strat.update_position("BTCUSDT", 105.0, current_high=105.0, current_low=104.8)
        # Bar 2: +8%
        strat.update_position("BTCUSDT", 108.0, current_high=108.0, current_low=107.5)
        # Bar 3: +12% — hard TP fires
        # peak_gain=12%, trail_stop=11.5%, trail_level=111.5
        # low=111.8 > 111.5 → trail does NOT fire → hard TP fires
        result = strat.update_position("BTCUSDT", 112.0, current_high=112.0, current_low=111.8)
        assert result == "target"


class TestMaxHold:
    """Max hold exit works for directional positions."""

    def test_max_hold_directional(self, strat: VolStraddleStrategy):
        _open_long(strat)
        pos = strat._positions["BTCUSDT"]
        pos["bars_held"] = 191  # next update will be 192
        result = strat.update_position("BTCUSDT", 100.5, current_high=100.5, current_low=100.0)
        assert result == "max_hold"


class TestPeakGainTracking:
    """Peak gain ratchets up and never decreases."""

    def test_peak_gain_only_increases(self, strat: VolStraddleStrategy):
        _open_long(strat, entry=100.0)
        strat.update_position("BTCUSDT", 103.0, current_high=103.0, current_low=102.8)
        assert strat._positions["BTCUSDT"]["peak_gain"] == pytest.approx(0.03, abs=1e-6)

        # Price drops — peak should NOT decrease
        # But trail is active (peak 3% > 2%), so worst_gain must
        # stay above trail_stop (2.7%) to not exit
        strat.update_position("BTCUSDT", 102.9, current_high=103.0, current_low=102.8)
        assert strat._positions["BTCUSDT"]["peak_gain"] == pytest.approx(0.03, abs=1e-6)

        # New high — peak increases
        # peak becomes 4%, trail_stop = 3.7%, low must be > $103.7
        result = strat.update_position("BTCUSDT", 104.0, current_high=104.0, current_low=103.8)
        assert result is None
        assert strat._positions["BTCUSDT"]["peak_gain"] == pytest.approx(0.04, abs=1e-6)


class TestConfigDefaults:
    """Verify config defaults match r47 exit params."""

    def test_default_config(self):
        cfg = VolStraddleConfig()
        assert cfg.d_tp == 0.20  # r68: raised from 0.12 — dead parameter, now explicit safety cap
        assert cfg.d_trail_activate == 0.040  # r69→current: 0.010→0.040
        assert cfg.d_trail_distance == 0.005  # r106: 0.5% (rule #8)
        assert cfg.d_sl == 0.01
        # Leverage production (half-Kelly)
        assert cfg.base_leverage == 30.0
        assert cfg.max_leverage == 30.0
        assert cfg.max_concurrent == 10


class TestStateRestoreTrail:
    """Trail state persists across position restore."""

    def test_trail_fields_in_position(self, strat: VolStraddleStrategy):
        """trail_active and peak_gain exist in position dict."""
        _open_long(strat)
        pos = strat._positions["BTCUSDT"]
        assert "trail_active" in pos
        assert "peak_gain" in pos
        assert pos["trail_active"] is False
        assert pos["peak_gain"] == 0.0
