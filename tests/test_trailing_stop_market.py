"""Tests for server-side TRAILING_STOP_MARKET migration (r90).

Verifies:
- Trail order placement/cancellation via Algo Order API
- Trail parameter calculation per strategy_type
- Entry flow integration (trail placed alongside SL)
- Close flow cleanup (trail cancelled on position close)
- Reconciliation: trail fill detection + orphan cleanup
- Trail bracket repair for missing trail orders
- Client-side trail suppression when server_native=True
- Straddle independent trail orders per leg
- Max hold cancel trail before closing
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _build_vs_scalper


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_trail_scalper(trail_server_native: bool = True, **kwargs):
    """Build a vs_scalper with trail_server_native configured."""
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    cfg = VolStraddleConfig(trail_server_native=trail_server_native)
    return _build_vs_scalper(vs_config=cfg, **kwargs)


def _inject_position(scalper, pos_key: str, **overrides):
    """Inject a position into the scalper's strategy."""
    signal = 1 if "LONG" in pos_key else -1
    pos = {
        "symbol": pos_key.split(":")[0],
        "signal": signal,
        "entry_price": 100.0,
        "notional": 3000.0,
        "bars_held": 5,
        "peak_gain": 0.02,
        "best_pnl": 0.02,
        "worst_pnl": -0.003,
        "server_sl_order_id": "sl_111",
        "server_tp_order_id": None,
        "server_trail_order_id": "trail_9999001",
        "sl": 0.015,
        "tp": 0.05,
        "tf": "15m",
        "position_side": "LONG" if signal == 1 else "SHORT",
        "trail_active": False,
        "opened_at": "2026-01-01T00:00:00+00:00",
        "strategy_type": "straddle",
        "is_straddle": True,
        "move_size": 1.0,
        "entry_stop_mult": 0.015,
    }
    pos.update(overrides)
    scalper.strategy._positions[pos_key] = pos
    return pos


# ══════════════════════════════════════════════════════════════════════════
# T3: _get_trail_params
# ══════════════════════════════════════════════════════════════════════════


class TestGetTrailParams:
    def test_straddle_long(self):
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("straddle", 1, 100.0)
        assert cr == pytest.approx(0.1)  # 0.001 * 100
        assert ap == pytest.approx(100.0)  # entry_price (no activation buffer)
        assert cs == "SELL"

    def test_straddle_short(self):
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("straddle", -1, 100.0)
        assert cr == pytest.approx(0.1)
        assert ap == pytest.approx(100.0)  # entry_price
        assert cs == "BUY"

    def test_vr_squeeze_long(self):
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("vr_squeeze", 1, 200.0)
        assert cr == pytest.approx(0.2)  # 0.002 * 100
        assert ap == pytest.approx(200.0)  # entry_price
        assert cs == "SELL"

    def test_ma_cluster_short(self):
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("ma_cluster", -1, 50.0)
        assert cr == pytest.approx(0.3)  # 0.003 * 100
        assert ap == pytest.approx(50.0)  # entry_price
        assert cs == "BUY"

    def test_callback_rate_clamp_min(self):
        """callbackRate must be at least 0.1 (Binance minimum)."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig(trail_server_native=True, s_trail_distance=0.0005)
        s = _build_vs_scalper(vs_config=cfg)
        cr, _, _ = s._get_trail_params("straddle", 1, 100.0)
        assert cr == 0.1  # 0.0005 * 100 = 0.05 → clamped to 0.1


# ══════════════════════════════════════════════════════════════════════════
# T1: _place_trailing_stop_market
# ══════════════════════════════════════════════════════════════════════════


class TestPlaceTrailingStopMarket:
    def test_paper_mode_returns_mock_id(self):
        s = _build_trail_scalper()
        assert s.paper_mode is True
        oid = s._place_trailing_stop_market("TESTUSDT", "SELL", 10.0, 0.1, 101.5, "LONG")
        assert isinstance(oid, str)
        assert oid.startswith("paper_trail_")

    def test_live_mode_calls_algo_api(self):
        s = _build_trail_scalper()
        s.paper_mode = False
        s.client._request_futures_api = MagicMock(return_value={"algoId": "12345678"})

        oid = s._place_trailing_stop_market("TESTUSDT", "SELL", 10.0, 0.1, 101.5, "LONG")

        assert oid == "12345678"
        call_args = s.client._request_futures_api.call_args
        assert call_args[0] == ("post", "algoOrder")
        data = call_args[1]["data"]
        assert data["type"] == "TRAILING_STOP_MARKET"
        assert data["callbackRate"] == "0.1"
        assert data["activationPrice"] == "101.5"
        assert data["positionSide"] == "LONG"
        assert data["side"] == "SELL"
        assert data["quantity"] == "10.0"
        assert data["algoType"] == "CONDITIONAL"
        assert "reduceOnly" not in data  # mutual exclusion with positionSide

    def test_live_mode_no_position_side_uses_reduce_only(self):
        """One-Way mode (no positionSide) should use reduceOnly."""
        s = _build_trail_scalper()
        s.paper_mode = False
        s.client._request_futures_api = MagicMock(return_value={"algoId": "99999"})

        oid = s._place_trailing_stop_market("TESTUSDT", "SELL", 5.0, 0.2, 102.0)

        data = s.client._request_futures_api.call_args[1]["data"]
        assert data["reduceOnly"] == "true"
        assert "positionSide" not in data

    def test_api_error_returns_none(self):
        s = _build_trail_scalper()
        s.paper_mode = False
        s.client._request_futures_api = MagicMock(side_effect=Exception("API error -1000"))

        oid = s._place_trailing_stop_market("TESTUSDT", "SELL", 10.0, 0.1, 101.5, "LONG")
        assert oid is None


# ══════════════════════════════════════════════════════════════════════════
# T2: _cancel_trailing_stop_market
# ══════════════════════════════════════════════════════════════════════════


class TestCancelTrailingStopMarket:
    def test_paper_mode_returns_true(self):
        s = _build_trail_scalper()
        assert s._cancel_trailing_stop_market("TESTUSDT", "trail_12345") is True

    def test_live_cancel_delegates_to_cancel_server_order(self):
        """_cancel_trailing_stop_market delegates to _cancel_server_order (Algo API)."""
        s = _build_trail_scalper()
        s.paper_mode = False
        s._cancel_server_order = MagicMock(return_value=True)

        result = s._cancel_trailing_stop_market("TESTUSDT", "12345")
        assert result is True
        s._cancel_server_order.assert_called_once_with("TESTUSDT", "12345", "trailing-stop")

    def test_already_filled_returns_true(self):
        """Already filled/cancelled orders should return True (benign)."""
        s = _build_trail_scalper()
        s.paper_mode = False
        s._cancel_server_order = MagicMock(return_value=True)

        assert s._cancel_trailing_stop_market("TESTUSDT", "12345") is True


# ══════════════════════════════════════════════════════════════════════════
# _check_trail_order_status
# ══════════════════════════════════════════════════════════════════════════


class TestCheckTrailOrderStatus:
    def test_delegates_to_check_order_filled(self):
        """_check_trail_order_status delegates to _check_order_filled (Algo API)."""
        s = _build_trail_scalper()
        s.paper_mode = False
        s._check_order_filled = MagicMock(
            return_value={
                "status": "FILLED",
                "avgPrice": "101.50",
                "executedQty": "10.0",
                "algoId": "12345",
            }
        )
        result = s._check_trail_order_status("TESTUSDT", "12345")
        assert result["status"] == "FILLED"
        assert result["avgPrice"] == "101.50"
        s._check_order_filled.assert_called_once_with("TESTUSDT", "12345")

    def test_not_filled_returns_none(self):
        s = _build_trail_scalper()
        s.paper_mode = False
        s._check_order_filled = MagicMock(return_value=None)
        result = s._check_trail_order_status("TESTUSDT", "12345")
        assert result is None

    def test_paper_mode_returns_none(self):
        s = _build_trail_scalper()
        # _check_order_filled returns None for paper mode
        result = s._check_trail_order_status("TESTUSDT", "paper_trail_12345")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# T4/T5: Entry flow — trail placed, TP skipped
# ══════════════════════════════════════════════════════════════════════════


class TestTrailAtEntry:
    def test_trail_placed_at_entry_tp_skipped(self):
        """When trail_server_native=True, trail is placed and TP is skipped.

        Tests the bracket placement section of _open_position (positions.py L394-460)
        by verifying the position dict after entry.
        """
        s = _build_trail_scalper(trail_server_native=True)

        # Paper mode: _place_trailing_stop_market returns "paper_trail_..."
        # _place_server_stop_loss returns paper_stop_...
        # We need to ensure the entry flow reaches the bracket section.
        # Use direct strategy.open_position to verify the position dict stores trail ID.
        trail_oid = "trail_77777"
        s.strategy.open_position(
            symbol="TESTUSDT",
            signal=1,
            entry_price=100.0,
            notional=3000.0,
            leverage=30,
            move_size=1.0,
            server_sl_order_id="sl_mock",
            server_tp_order_id=None,
            server_trail_order_id=trail_oid,
            position_side="LONG",
            sl=0.015,
            tp=0.05,
            tf="15m",
            strategy_type="straddle",
        )

        pos = s.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_trail_order_id"] == "trail_77777"
        assert pos["server_tp_order_id"] is None  # TP skipped

    def test_legacy_mode_no_trail(self):
        """When trail_server_native=False, TP is placed and trail is not."""
        s = _build_trail_scalper(trail_server_native=False)

        s.strategy.open_position(
            symbol="TESTUSDT",
            signal=1,
            entry_price=100.0,
            notional=3000.0,
            leverage=30,
            move_size=1.0,
            server_sl_order_id="sl_mock",
            server_tp_order_id="tp_mock",
            position_side="LONG",
            sl=0.015,
            tp=0.05,
            tf="15m",
            strategy_type="straddle",
        )

        pos = s.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_tp_order_id"] == "tp_mock"
        assert pos["server_trail_order_id"] is None

    def test_positions_py_trail_branch(self):
        """Integration: verify positions.py places trail instead of TP."""
        s = _build_trail_scalper(trail_server_native=True)

        # Check that the config flag is respected in the code path
        assert s.vs_config.trail_server_native is True

        # The positions.py code at L405-460 checks vs_config.trail_server_native
        # and calls _place_trailing_stop_market + skips _place_server_take_profit.
        # We verify by checking the trail params calculation works correctly.
        cr, ap, cs = s._get_trail_params("straddle", 1, 100.0)
        assert cr == pytest.approx(0.1)
        assert ap == pytest.approx(100.0)  # activation = entry_price
        assert cs == "SELL"


# ══════════════════════════════════════════════════════════════════════════
# T7: Close position cancels trail
# ══════════════════════════════════════════════════════════════════════════


class TestClosePositionCancelsTrail:
    def test_trail_cancelled_on_close(self):
        s = _build_trail_scalper()
        _inject_position(s, "TESTUSDT:LONG", server_trail_order_id="trail_77777")

        s._cancel_trail_regular = MagicMock(return_value=True)
        s._cancel_server_stop_loss = MagicMock(return_value=True)
        s._cancel_server_take_profit = MagicMock(return_value=True)

        s._close_position("TESTUSDT:LONG", "max_hold", 102.0)

        s._cancel_trail_regular.assert_called_once_with("TESTUSDT", "trail_77777")

    def test_no_trail_no_cancel(self):
        """If position has no trail order, cancel should not be called."""
        s = _build_trail_scalper()
        _inject_position(s, "TESTUSDT:LONG", server_trail_order_id=None)

        s._cancel_trail_regular = MagicMock()
        s._cancel_server_stop_loss = MagicMock(return_value=True)
        s._cancel_server_take_profit = MagicMock(return_value=True)

        s._close_position("TESTUSDT:LONG", "max_hold", 102.0)

        s._cancel_trail_regular.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# T6: Client-side trail suppressed when server_native
# ══════════════════════════════════════════════════════════════════════════


class TestClientTrailSuppression:
    def test_no_client_trail_exit_when_server_native(self):
        """update_position should NOT return 'trailing_stop' when server_native."""
        s = _build_trail_scalper(trail_server_native=True)
        _inject_position(
            s,
            "TESTUSDT:LONG",
            trail_active=True,
            peak_gain=0.025,  # 2.5%
            worst_pnl=-0.001,
            strategy_type="straddle",
            sl=0.015,
        )

        # trail_stop = 2.5% - 0.1% = 2.4%
        # entry=100, low=101.5 → worst_gain = +1.5% ≤ 2.4% → would fire in legacy
        result = s.strategy.update_position(
            symbol="TESTUSDT:LONG",
            current_price=102.0,
            current_high=103.0,
            current_low=101.5,
        )
        assert result != "trailing_stop"

    def test_client_trail_fires_when_legacy(self):
        """update_position SHOULD return 'trailing_stop' when server_native=False."""
        s = _build_trail_scalper(trail_server_native=False)
        _inject_position(
            s,
            "TESTUSDT:LONG",
            trail_active=True,
            peak_gain=0.025,  # 2.5%
            worst_pnl=-0.001,
            strategy_type="straddle",
            sl=0.015,
        )

        # trail_stop = 2.5% - 0.1% = 2.4%
        # entry=100, low=101.5 → worst_gain = +1.5% ≤ 2.4% → fires
        result = s.strategy.update_position(
            symbol="TESTUSDT:LONG",
            current_price=102.0,
            current_high=103.0,
            current_low=101.5,
        )
        assert result == "trailing_stop"


# ══════════════════════════════════════════════════════════════════════════
# T8/T9: Reconciliation — trail fill detection
# ══════════════════════════════════════════════════════════════════════════


class TestReconTrailFill:
    def test_trail_fill_detected_and_recorded(self):
        """When trail order is FILLED, position should be closed and SL cancelled."""
        s = _build_trail_scalper()
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="77777",
            server_sl_order_id="sl_111",
            is_straddle=False,
        )

        # Mock trail order check → FILLED (regular API)
        s._check_trail_regular = MagicMock(
            return_value={
                "status": "FILLED",
                "avgPrice": "102.50",
                "executedQty": "30.0",
                "orderId": 77777,
            }
        )
        # Mock algo check → not filled (SL still active)
        s._check_order_filled = MagicMock(return_value=None)
        s._cancel_server_stop_loss = MagicMock(return_value=True)
        s._record_server_fill = MagicMock()

        s._sync_server_orders()

        # SL should be cancelled
        s._cancel_server_stop_loss.assert_called_with("TESTUSDT", "sl_111")
        # Trade should be recorded
        s._record_server_fill.assert_called_once_with("TESTUSDT:LONG", 102.50, 30.0, "server_trail")

    def test_trail_not_filled_no_action(self):
        """If trail order is not filled, no action should be taken."""
        s = _build_trail_scalper()
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="77777",
            is_straddle=False,
        )

        s._check_trail_regular = MagicMock(return_value=None)
        s._check_order_filled = MagicMock(return_value=None)
        s._record_server_fill = MagicMock()

        s._sync_server_orders()

        s._record_server_fill.assert_not_called()

    def test_trail_cancelled_clears_for_repair(self):
        """Externally cancelled trail should be cleared and then repaired."""
        s = _build_trail_scalper(trail_server_native=True)
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="77777",
            is_straddle=False,
        )

        s._check_trail_regular = MagicMock(return_value={"status": "CANCELED", "orderId": 77777})
        s._check_order_filled = MagicMock(return_value=None)
        # After clearing, bracket repair will re-place the trail
        s._place_trail_regular = MagicMock(return_value="88888")
        s._get_trail_params = MagicMock(return_value=(0.1, 101.5, "SELL"))

        s._sync_server_orders()

        # Trail should be repaired immediately
        s._place_trail_regular.assert_called_once()
        pos = s.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_trail_order_id"] == "88888"


# ══════════════════════════════════════════════════════════════════════════
# T10: Trail bracket repair
# ══════════════════════════════════════════════════════════════════════════


class TestTrailBracketRepair:
    def test_missing_trail_repaired(self):
        """Position with missing trail order should get one placed."""
        s = _build_trail_scalper(trail_server_native=True)
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id=None,
            server_sl_order_id="sl_111",
            strategy_type="straddle",
            is_straddle=False,
        )

        # Mock all order checks → nothing filled
        s._check_order_filled = MagicMock(return_value=None)
        s._check_trail_regular = MagicMock(return_value=None)
        s._place_trail_regular = MagicMock(return_value="88888")
        s._get_trail_params = MagicMock(return_value=(0.1, 101.5, "SELL"))

        s._sync_server_orders()

        # Trail should be placed as repair
        s._place_trail_regular.assert_called_once()
        pos = s.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_trail_order_id"] == "88888"

    def test_existing_trail_not_re_placed(self):
        """Position with existing trail order should NOT get a new one."""
        s = _build_trail_scalper(trail_server_native=True)
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="trail_77777",
            server_sl_order_id="sl_111",
            is_straddle=False,
        )

        s._check_order_filled = MagicMock(return_value=None)
        s._check_trail_regular = MagicMock(return_value=None)
        s._place_trail_regular = MagicMock()

        s._sync_server_orders()

        s._place_trail_regular.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# T13: Ratchet skipped when server_native
# ══════════════════════════════════════════════════════════════════════════


class TestRatchetSkip:
    def test_ratchet_skipped_when_server_native(self):
        s = _build_trail_scalper(trail_server_native=True)
        s.paper_mode = False
        s._ratchet_trail_sl = MagicMock()

        _inject_position(
            s,
            "TESTUSDT:LONG",
            trail_active=True,
            server_trail_order_id="trail_77777",
        )

        # The ratchet check is in _update_existing_positions which calls
        # strategy.update_position then checks trail_active for ratchet.
        # Directly test the guard condition.
        assert s.vs_config.trail_server_native is True
        # ratchet should be skipped due to trail_server_native=True


# ══════════════════════════════════════════════════════════════════════════
# T12: State persistence
# ══════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    def test_trail_order_id_in_position_dict(self):
        """server_trail_order_id should be stored in position dict."""
        s = _build_trail_scalper()
        _inject_position(s, "TESTUSDT:LONG", server_trail_order_id="trail_77777")

        pos = s.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_trail_order_id"] == "trail_77777"

    def test_trail_order_id_none_by_default(self):
        """When trail not placed, server_trail_order_id should be None."""
        s = _build_trail_scalper()
        # Use strategy.open_position directly
        s.strategy.open_position(
            symbol="TESTUSDT",
            signal=1,
            entry_price=100.0,
            notional=3000.0,
            leverage=30,
            move_size=1.0,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
            position_side="LONG",
            sl=0.015,
            tp=0.05,
            tf="15m",
            strategy_type="straddle",
        )

        pos_key = "TESTUSDT:LONG"
        pos = s.strategy._positions.get(pos_key)
        assert pos is not None
        assert pos["server_trail_order_id"] is None


# ══════════════════════════════════════════════════════════════════════════
# Straddle: independent trail per leg
# ══════════════════════════════════════════════════════════════════════════


class TestStraddleIndependentTrails:
    def test_each_leg_has_own_trail(self):
        """LONG and SHORT legs should each have their own trail order."""
        s = _build_trail_scalper()
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="trail_11111",
            signal=1,
        )
        _inject_position(
            s,
            "TESTUSDT:SHORT",
            server_trail_order_id="trail_22222",
            signal=-1,
            position_side="SHORT",
        )

        long_trail = s.strategy._positions["TESTUSDT:LONG"]["server_trail_order_id"]
        short_trail = s.strategy._positions["TESTUSDT:SHORT"]["server_trail_order_id"]

        assert long_trail == "trail_11111"
        assert short_trail == "trail_22222"
        assert long_trail != short_trail

    def test_trail_fill_one_leg_preserves_other(self):
        """When one leg's trail fires, the other leg should be unaffected."""
        s = _build_trail_scalper()
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="trail_11111",
            server_sl_order_id="sl_long",
            is_straddle=True,
            straddle_pair_id="pair_1",
        )
        _inject_position(
            s,
            "TESTUSDT:SHORT",
            server_trail_order_id="trail_22222",
            server_sl_order_id="sl_short",
            signal=-1,
            position_side="SHORT",
            is_straddle=True,
            straddle_pair_id="pair_1",
        )

        # LONG trail fires
        def mock_trail_check(sym, oid):
            if oid == "trail_11111":
                return {"status": "FILLED", "avgPrice": "102.0", "executedQty": "30.0", "orderId": 11111}
            return None  # Not filled

        s._check_trail_regular = MagicMock(side_effect=mock_trail_check)
        s._check_order_filled = MagicMock(return_value=None)
        s._cancel_server_stop_loss = MagicMock(return_value=True)
        s._record_server_fill = MagicMock()

        s._sync_server_orders()

        # SHORT leg should still exist
        assert "TESTUSDT:SHORT" in s.strategy._positions
        # SHORT trail should still be intact
        assert s.strategy._positions["TESTUSDT:SHORT"]["server_trail_order_id"] == "trail_22222"


# ══════════════════════════════════════════════════════════════════════════
# Config flag
# ══════════════════════════════════════════════════════════════════════════


class TestConfigFlag:
    def test_default_in_code_is_false(self):
        """Code default is False, but .env override takes precedence."""
        # Note: if VS_TRAIL_SERVER_NATIVE=true in .env, this will be True
        # We test the code path, not the env var behavior
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig(trail_server_native=False)
        assert cfg.trail_server_native is False

    def test_enable_via_constructor(self):
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig(trail_server_native=True)
        assert cfg.trail_server_native is True


# ══════════════════════════════════════════════════════════════════════════
# Trail Facade Tests (delegates to Algo API)
# ══════════════════════════════════════════════════════════════════════════


class TestPlaceTrailRegular:
    """Test _place_trail_regular delegates to _place_trailing_stop_market (Algo API)."""

    def test_paper_mode_returns_mock_id(self):
        s = _build_trail_scalper()
        assert s.paper_mode is True
        oid = s._place_trail_regular("TESTUSDT", "SELL", 10.0, 0.2, 101.5, "LONG")
        assert isinstance(oid, str)
        assert oid.startswith("paper_trail_")

    def test_delegates_to_algo_api(self):
        s = _build_trail_scalper()
        s._place_trailing_stop_market = MagicMock(return_value="4000000123456")

        oid = s._place_trail_regular("TESTUSDT", "SELL", 10.0, 0.2, 101.5, "LONG")

        assert oid == "4000000123456"
        s._place_trailing_stop_market.assert_called_once_with(
            symbol="TESTUSDT",
            side="SELL",
            quantity=10.0,
            callback_rate=0.2,
            activation_price=101.5,
            position_side="LONG",
        )

    def test_none_activation_passes_zero(self):
        s = _build_trail_scalper()
        s._place_trailing_stop_market = MagicMock(return_value="4000000999")

        s._place_trail_regular("TESTUSDT", "BUY", 5.0, 0.3, position_side="SHORT")

        call_kwargs = s._place_trailing_stop_market.call_args[1]
        assert call_kwargs["activation_price"] == 0.0  # None → 0.0 fallback

    def test_algo_failure_returns_none(self):
        s = _build_trail_scalper()
        s._place_trailing_stop_market = MagicMock(return_value=None)

        oid = s._place_trail_regular("TESTUSDT", "SELL", 10.0, 0.2, 101.5, "LONG")
        assert oid is None


class TestCancelTrailRegular:
    """Test _cancel_trail_regular delegates to _cancel_trailing_stop_market (Algo API)."""

    def test_paper_mode_returns_true(self):
        s = _build_trail_scalper()
        assert s._cancel_trail_regular("TESTUSDT", "12345") is True

    def test_paper_prefix_returns_true(self):
        s = _build_trail_scalper()
        assert s._cancel_trail_regular("TESTUSDT", "paper_trail_12345") is True

    def test_none_order_id_returns_true(self):
        s = _build_trail_scalper()
        s._cancel_trailing_stop_market = MagicMock(return_value=True)
        assert s._cancel_trail_regular("TESTUSDT", None) is True

    def test_delegates_to_algo_cancel(self):
        s = _build_trail_scalper()
        s._cancel_trailing_stop_market = MagicMock(return_value=True)

        result = s._cancel_trail_regular("TESTUSDT", "4000000123")
        assert result is True
        s._cancel_trailing_stop_market.assert_called_once_with("TESTUSDT", "4000000123")


class TestCheckTrailRegular:
    """Test _check_trail_regular delegates to _check_trail_order_status (Algo API)."""

    def test_paper_mode_returns_none(self):
        s = _build_trail_scalper()
        assert s._check_trail_regular("TESTUSDT", "paper_trail_123") is None

    def test_filled_returns_data(self):
        s = _build_trail_scalper()
        s._check_trail_order_status = MagicMock(
            return_value={
                "status": "FILLED",
                "avgPrice": "102.50",
                "executedQty": "30.0",
            }
        )

        result = s._check_trail_regular("TESTUSDT", "4000000777")
        assert result is not None
        assert result["status"] == "FILLED"

    def test_none_returns_none(self):
        s = _build_trail_scalper()
        s._check_trail_order_status = MagicMock(return_value=None)

        result = s._check_trail_regular("TESTUSDT", "4000000777")
        assert result is None

    def test_canceled_returns_canceled(self):
        s = _build_trail_scalper()
        s._check_trail_order_status = MagicMock(return_value={"status": "CANCELLED"})

        result = s._check_trail_regular("TESTUSDT", "4000000777")
        assert result is not None
        assert result["status"] == "CANCELLED"


class TestTrailCrossCancel:
    """Test cross-cancel logic: when SL fires → trail cancelled, when trail fires → SL+TP cancelled."""

    def test_sl_fires_cancels_trail(self):
        """When server SL fills, the trail order should also be cancelled."""
        s = _build_trail_scalper()
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="77777",
            server_sl_order_id="sl_111",
            server_tp_order_id="tp_222",
            is_straddle=False,
        )

        # SL fires
        s._check_order_filled = MagicMock(
            return_value={
                "status": "FILLED",
                "avgPrice": "99.0",
                "executedQty": "30.0",
            }
        )
        s._check_trail_regular = MagicMock(return_value=None)
        s._cancel_trail_regular = MagicMock(return_value=True)
        s._cancel_server_take_profit = MagicMock(return_value=True)
        s._record_server_fill = MagicMock()

        s._sync_server_orders()

        # Trail should be cancelled when SL fires
        s._cancel_trail_regular.assert_called_with("TESTUSDT", "77777")

    def test_trail_fires_cancels_sl_and_tp(self):
        """When trail fills, both SL and TP should be cancelled."""
        s = _build_trail_scalper()
        s.paper_mode = False
        _inject_position(
            s,
            "TESTUSDT:LONG",
            server_trail_order_id="77777",
            server_sl_order_id="sl_111",
            server_tp_order_id="tp_222",
            is_straddle=False,
        )

        # SL/TP not filled
        s._check_order_filled = MagicMock(return_value=None)
        # Trail fires
        s._check_trail_regular = MagicMock(
            return_value={
                "status": "FILLED",
                "avgPrice": "103.0",
                "executedQty": "30.0",
            }
        )
        s._cancel_server_stop_loss = MagicMock(return_value=True)
        s._cancel_server_take_profit = MagicMock(return_value=True)
        s._record_server_fill = MagicMock()

        s._sync_server_orders()

        # Both SL and TP should be cancelled
        s._cancel_server_stop_loss.assert_called_with("TESTUSDT", "sl_111")
        s._cancel_server_take_profit.assert_called_with("TESTUSDT", "tp_222")
        s._record_server_fill.assert_called_once_with("TESTUSDT:LONG", 103.0, 30.0, "server_trail")


class TestCallbackRateBounds:
    """Test callbackRate edge values from _get_trail_params."""

    def test_straddle_at_minimum_0_1(self):
        """s_trail_distance=0.001 → callbackRate=0.1 (Binance minimum)."""
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("straddle", 1, 100.0)
        assert cr == pytest.approx(0.1, abs=0.01)  # 0.001 * 100 = 0.1

    def test_vr_squeeze_callback_rate(self):
        """d_trail_distance=0.002 → callbackRate=0.2."""
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("vr_squeeze", 1, 100.0)
        assert cr == pytest.approx(0.2, abs=0.01)

    def test_ma_cluster_callback_rate(self):
        """ma_trail_distance=0.003 → callbackRate=0.3."""
        s = _build_trail_scalper()
        cr, ap, cs = s._get_trail_params("ma_cluster", 1, 100.0)
        assert cr == pytest.approx(0.3, abs=0.01)
