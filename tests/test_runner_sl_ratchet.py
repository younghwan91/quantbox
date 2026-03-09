"""
v12.4.4 Hotfix Tests — Runner SL Ratchet Stale _server_sl_pct

When a runner's server SL is re-placed at breakeven (via _close_position
failure or bracket repair), _server_sl_pct must be reset to 0.0 so that
_ratchet_runner_sl can immediately re-ratchet on the next cycle.

Without the fix, _server_sl_pct retains the old ratcheted value (e.g. 2.5%),
_ratchet_runner_sl computes improvement < 0.5%, skips ratcheting, and the
server SL stays stuck at breakeven — losing all trailing stop protection.
"""

from unittest.mock import MagicMock, patch
import pytest


def _make_scalper(paper: bool = True, **config_overrides):
    """Create a minimal RubberBandScalper for testing."""
    from quantbox.scalper import RubberBandScalper

    mock_client = MagicMock()
    mock_client.futures_account_balance.return_value = [
        {"asset": "USDT", "availableBalance": "1000.0"},
    ]
    mock_client._request_futures_api.return_value = {"algoId": 99999}

    with (
        patch("quantbox.scalper.load_dotenv"),
        patch("quantbox.scalper.Client", return_value=mock_client),
        patch("quantbox.scalper.LimitOrderChaser"),
    ):
        scalper = RubberBandScalper(paper_mode=paper)

    scalper.client = mock_client
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(scalper.config, k, v)

    return scalper


def _make_runner_position(scalper, symbol="ETHUSDT", ratcheted_pct=0.025):
    """Open a position and set it up as a ratcheted runner."""
    scalper.strategy.open_position(
        symbol=symbol,
        signal=1,
        entry_price=100.0,
        notional=50.0,
        leverage=2,
        move_size=0.05,
        server_sl_order_id="sl_ratcheted",
        server_tp_order_id=None,
    )
    pos = scalper.strategy._positions[symbol]
    pos["partial_closed"] = True
    pos["trail_floor"] = 0.0
    pos["highest_since_tp1"] = 0.04
    pos["_computed_trail_stop"] = ratcheted_pct
    pos["_server_sl_pct"] = ratcheted_pct  # Previously ratcheted
    return pos


class TestCloseFailureResetsRatchetState:
    """_close_position failure on runner must reset _server_sl_pct to 0.0."""

    def test_runner_close_failure_resets_server_sl_pct(self):
        """After close failure, _server_sl_pct must be 0.0 (not stale)."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_be_new")
        scalper._place_order = MagicMock(return_value=None)  # Close FAILS

        pos = _make_runner_position(scalper, ratcheted_pct=0.025)
        assert pos["_server_sl_pct"] == 0.025  # pre-condition

        result = scalper._close_position("ETHUSDT", "trail_runner", 100.5)
        assert result is False

        pos = scalper.strategy._positions["ETHUSDT"]
        # CRITICAL: _server_sl_pct must be reset so ratchet can re-fire
        assert pos["_server_sl_pct"] == 0.0, (
            f"_server_sl_pct should be 0.0 after BE SL re-placement, got {pos['_server_sl_pct']}"
        )
        assert pos["server_sl_order_id"] == "sl_be_new"

    def test_phase1_close_failure_does_not_reset(self):
        """Phase 1 (non-runner) failure should NOT touch _server_sl_pct."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
        )
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_new")
        scalper._place_order = MagicMock(return_value=None)

        scalper.strategy.open_position(
            symbol="ETHUSDT",
            signal=1,
            entry_price=100.0,
            notional=100.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id="sl_old",
            server_tp_order_id=None,
        )
        # Phase 1: partial_closed=False (default)

        scalper._close_position("ETHUSDT", "trailing_stop", 99.0)

        pos = scalper.strategy._positions["ETHUSDT"]
        # Phase 1 doesn't use _server_sl_pct; should remain default
        assert pos.get("_server_sl_pct", 0.0) == 0.0


class TestBracketRepairResetsRatchetState:
    """Bracket repair for runner must reset _server_sl_pct to 0.0."""

    def test_bracket_repair_runner_resets_server_sl_pct(self):
        """When runner SL is repaired at BE, _server_sl_pct is reset."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()

        pos = _make_runner_position(scalper, ratcheted_pct=0.030)
        # Simulate externally cancelled SL → None
        pos["server_sl_order_id"] = None
        pos["server_tp_order_id"] = None

        # Mock: _check_order_filled not needed (SL already None)
        # _place_server_stop_loss returns new algo ID
        scalper._place_server_stop_loss = MagicMock(return_value="sl_repaired")
        scalper._place_server_take_profit = MagicMock(return_value="tp_new")

        scalper._sync_server_orders()

        pos = scalper.strategy._positions["ETHUSDT"]
        assert pos["server_sl_order_id"] == "sl_repaired"
        # CRITICAL: _server_sl_pct must be reset for ratchet to work
        assert pos["_server_sl_pct"] == 0.0, (
            f"_server_sl_pct should be 0.0 after bracket repair BE SL, got {pos['_server_sl_pct']}"
        )

    def test_bracket_repair_phase1_does_not_reset(self):
        """Phase 1 bracket repair uses original stop_mult, no reset needed."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
        )
        scalper._save_state = MagicMock()

        scalper.strategy.open_position(
            symbol="ETHUSDT",
            signal=1,
            entry_price=100.0,
            notional=100.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id=None,
            server_tp_order_id=None,
        )

        scalper._place_server_stop_loss = MagicMock(return_value="sl_repaired")
        scalper._place_server_take_profit = MagicMock()

        scalper._sync_server_orders()

        pos = scalper.strategy._positions["ETHUSDT"]
        assert pos["server_sl_order_id"] == "sl_repaired"
        # Phase 1: _server_sl_pct should stay default
        assert pos.get("_server_sl_pct", 0.0) == 0.0


class TestRatchetRecoveryAfterReset:
    """Integration: After _server_sl_pct reset, _ratchet_runner_sl fires."""

    def test_ratchet_fires_after_close_failure_reset(self):
        """Simulate close failure → reset → ratchet fires next cycle."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()

        # Step 1: Close failure re-places at BE
        scalper._place_server_stop_loss = MagicMock(return_value="sl_be")
        scalper._place_order = MagicMock(return_value=None)

        pos = _make_runner_position(scalper, ratcheted_pct=0.025)
        scalper._close_position("ETHUSDT", "trail_runner", 100.5)

        pos = scalper.strategy._positions["ETHUSDT"]
        assert pos["_server_sl_pct"] == 0.0  # Reset confirmed

        # Step 2: Next cycle — simulate ratchet with trail_stop at 2.6%
        pos["_computed_trail_stop"] = 0.026
        pos["server_sl_order_id"] = "sl_be"

        # Mock algo order for ratchet
        scalper._place_algo_order = MagicMock(return_value="sl_ratcheted_new")

        scalper._ratchet_runner_sl("ETHUSDT", pos)

        # Ratchet must fire because improvement = 2.6% - 0% = 2.6% > 0.5%
        scalper._place_algo_order.assert_called_once()
        assert pos["_server_sl_pct"] == 0.026
        assert pos["server_sl_order_id"] == "sl_ratcheted_new"

    def test_ratchet_blocked_without_fix(self):
        """Demonstrate the bug: without reset, ratchet would NOT fire.

        This test verifies that when _server_sl_pct is stale (simulating
        the pre-fix state), _ratchet_runner_sl skips ratcheting because
        improvement is below the 0.5% threshold.
        """
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()

        pos = _make_runner_position(scalper, ratcheted_pct=0.025)
        pos["server_sl_order_id"] = "sl_be"  # SL at BE
        # Simulate STALE _server_sl_pct (the bug scenario):
        pos["_server_sl_pct"] = 0.025  # Not reset
        pos["_computed_trail_stop"] = 0.026  # Only 0.1% improvement

        scalper._place_algo_order = MagicMock(return_value="sl_new")

        scalper._ratchet_runner_sl("ETHUSDT", pos)

        # With stale _server_sl_pct=2.5%, improvement=0.1% < 0.5% → skip
        scalper._place_algo_order.assert_not_called()
        # SL stays at breakeven — THIS IS THE BUG the fix prevents
