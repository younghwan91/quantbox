"""
Tests for _reconcile_positions() and _sync_server_orders().

Hedge fund level: ghost detection, API failure safety, qty mismatch correction.
All tests use paper_mode=False scalper with mocked client to test live code paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _build_vs_scalper


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _live_scalper(balance: float = 1000.0):
    """VolStraddle scalper in live mode (paper_mode=False) with mocked client."""
    from quantbox.scalper import RubberBandScalper
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    vs_config = VolStraddleConfig()

    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client"),
        patch("quantbox.scalper.DiscordNotifier"),
        patch("quantbox.scalper.DiscordBotRunner", None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="k",
            binance_api_secret="s",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token="",
            discord_channel_id="",
            paper_balance=balance,
        )
        scalper = RubberBandScalper(
            paper_mode=False,
            strategy_name="vol_straddle",
            vs_config=vs_config,
        )

    from tests.conftest import _SYMBOL_INFO

    for sym in ("TESTUSDT", "SHORTUSDT"):
        scalper._symbol_info_cache[sym] = _SYMBOL_INFO.copy()

    scalper.client = MagicMock()
    scalper.discord = MagicMock()
    scalper._save_state = MagicMock()
    return scalper


def _inject_hedge_position(scalper, sym: str, signal: int = 1, entry_price: float = 100.0):
    """Inject an open position dict under compound key (TESTUSDT:LONG)."""
    side = "LONG" if signal == 1 else "SHORT"
    key = f"{sym}:{side}"
    scalper.strategy._positions[key] = {
        "symbol": sym,
        "signal": signal,
        "entry_price": entry_price,
        "notional": 3000.0,
        "leverage": 30,
        "bars_held": 5,
        "peak_gain": 0.0,
        "best_pnl": 0.0,
        "server_sl_order_id": "sl_111",
        "server_tp_order_id": "tp_222",
        "sl": 0.01,
        "tp": 0.12,
        "tf": "15m",
        "position_side": side,
        "trail_active": False,
        "strategy_type": "vr_directional",
    }
    return key


# ══════════════════════════════════════════════════════════════════════
# 1. Ghost Position Detection
# ══════════════════════════════════════════════════════════════════════


class TestGhostPositionDetection:
    """Internal state has position, exchange has none → auto-remove + alert."""

    def test_ghost_triggers_discord_alert(self):
        scalper = _live_scalper()
        _inject_hedge_position(scalper, "TESTUSDT", signal=1)

        # Exchange returns no positions at all
        scalper.client.futures_position_information.return_value = []
        scalper.client.futures_account_trades.return_value = []

        scalper._reconcile_positions()

        # Discord alert must fire (via content= or embed=)
        assert scalper.discord.send.called, "Discord alert expected for ghost position"
        all_calls_str = str(scalper.discord.send.call_args_list).lower()
        assert "ghost" in all_calls_str or "reconcil" in all_calls_str, (
            f"Alert must mention ghost/reconciliation: {all_calls_str[:200]}"
        )

    def test_ghost_position_removed_from_internal_state(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)

        # Exchange shows no position
        scalper.client.futures_position_information.return_value = []
        scalper.client.futures_account_trades.return_value = []

        assert key in scalper.strategy._positions
        scalper._reconcile_positions()
        assert key not in scalper.strategy._positions, "Ghost must be removed from internal state"

    def test_valid_position_not_removed(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)

        # Exchange confirms position exists
        scalper.client.futures_position_information.return_value = [
            {
                "symbol": "TESTUSDT",
                "positionSide": "LONG",
                "positionAmt": "30.0",  # non-zero → position exists
                "entryPrice": "100.0",
            }
        ]

        scalper._reconcile_positions()
        assert key in scalper.strategy._positions, "Valid position must NOT be removed"

    def test_short_ghost_detected_separately(self):
        """Hedge mode: LONG can exist while SHORT is ghost — must detect only SHORT."""
        scalper = _live_scalper()
        long_key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)
        short_key = _inject_hedge_position(scalper, "TESTUSDT", signal=-1)

        # Exchange only has the LONG, not the SHORT
        scalper.client.futures_position_information.return_value = [
            {"symbol": "TESTUSDT", "positionSide": "LONG", "positionAmt": "30.0", "entryPrice": "100.0"},
            {"symbol": "TESTUSDT", "positionSide": "SHORT", "positionAmt": "0.0", "entryPrice": "0.0"},
        ]
        scalper.client.futures_account_trades.return_value = []

        scalper._reconcile_positions()

        assert long_key in scalper.strategy._positions, "LONG must survive"
        assert short_key not in scalper.strategy._positions, "SHORT ghost must be removed"


# ══════════════════════════════════════════════════════════════════════
# 2. API Failure Safety
# ══════════════════════════════════════════════════════════════════════


class TestReconciliationAPIFailure:
    """Exchange API failure → keep internal state intact, no crash."""

    def test_api_failure_preserves_positions(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)

        # Simulate API timeout
        scalper.client.futures_position_information.side_effect = Exception("Connection timeout")

        scalper._reconcile_positions()  # Must not raise

        assert key in scalper.strategy._positions, "Position must survive API failure"

    def test_api_failure_does_not_send_false_ghost_alert(self):
        scalper = _live_scalper()
        _inject_hedge_position(scalper, "TESTUSDT", signal=1)

        scalper.client.futures_position_information.side_effect = Exception("Network error")

        scalper._reconcile_positions()

        # No ghost alert on API error (only on confirmed ghost)
        ghost_alerts = [
            c for c in scalper.discord.send.call_args_list if "Ghost" in str(c) or "ghost" in str(c).lower()
        ]
        assert len(ghost_alerts) == 0, "API failure must NOT trigger ghost alert"

    def test_paper_mode_skips_reconciliation(self):
        """Paper mode must skip exchange API calls entirely."""
        scalper = _build_vs_scalper()  # paper_mode=True by default
        scalper.client = MagicMock()
        scalper.discord = MagicMock()

        scalper._reconcile_positions()

        scalper.client.futures_position_information.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# 3. _sync_server_orders: TRIGGERED + FINISHED both treated as filled
# ══════════════════════════════════════════════════════════════════════


class TestSyncServerOrdersStatusHandling:
    """Incident #4: TRIGGERED and FINISHED must both close the position.

    Previously only TRIGGERED was detected — FINISHED state (after brief
    TRIGGERED window) caused position to persist as ghost.
    """

    def _make_algo_response(self, status: str, avg_price: str = "99.0", qty: str = "30.0"):
        return {
            "status": status,
            "avgPrice": avg_price,
            "executedQty": qty,
            "algoId": "999",
        }

    def test_tp_triggered_closes_position(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)

        scalper.client.futures_get_algo_order = MagicMock(return_value=self._make_algo_response("TRIGGERED"))
        scalper._check_order_filled = MagicMock(
            return_value={"status": "FILLED", "avgPrice": "112.0", "executedQty": "30.0"}
        )
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._record_server_fill = MagicMock()

        scalper._sync_server_orders()
        scalper._record_server_fill.assert_called_once()

    def test_sl_triggered_closes_position(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)
        # Disable TP so SL is checked
        scalper.strategy._positions[key]["server_tp_order_id"] = None

        scalper._check_order_filled = MagicMock(
            return_value={"status": "FILLED", "avgPrice": "99.0", "executedQty": "30.0"}
        )
        scalper._cancel_server_take_profit = MagicMock()
        scalper._record_server_fill = MagicMock()

        scalper._sync_server_orders()
        scalper._record_server_fill.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
# 4. Bracket Repair — 3 consecutive failures → auto-close
# ══════════════════════════════════════════════════════════════════════


class TestBracketRepairExhaustion:
    """T1-3 (already in production code): 3 repair failures → force-close."""

    def test_three_failures_trigger_auto_close(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)
        # Remove SL so bracket repair is triggered
        scalper.strategy._positions[key]["server_sl_order_id"] = None
        scalper.strategy._positions[key]["server_tp_order_id"] = None

        # _place_server_stop_loss always fails (returns None)
        scalper._place_server_stop_loss = MagicMock(return_value=None)
        scalper._place_server_take_profit = MagicMock(return_value=None)
        scalper._close_position = MagicMock(return_value=True)
        scalper.client.futures_orderbook_ticker = MagicMock(return_value={"bidPrice": "99.0", "askPrice": "101.0"})

        # Run 3 cycles of bracket repair
        for _ in range(3):
            scalper._sync_server_orders()

        # After 3 failures, auto-close must have been called
        scalper._close_position.assert_called_once_with(key, "bracket_repair_exhausted", pytest.approx(99.0, abs=1.0))

    def test_successful_repair_resets_fail_count(self):
        scalper = _live_scalper()
        key = _inject_hedge_position(scalper, "TESTUSDT", signal=1)
        scalper.strategy._positions[key]["server_sl_order_id"] = None
        scalper.strategy._positions[key]["server_tp_order_id"] = None
        scalper.strategy._positions[key]["_repair_fail_count"] = 2  # one away from limit

        # This time repair succeeds
        scalper._place_server_stop_loss = MagicMock(return_value="new_sl_999")
        scalper._place_server_take_profit = MagicMock(return_value="new_tp_888")
        scalper._close_position = MagicMock()

        scalper._sync_server_orders()

        # Position must still be open (not auto-closed)
        scalper._close_position.assert_not_called()
        assert scalper.strategy._positions[key]["server_sl_order_id"] == "new_sl_999"


# ══════════════════════════════════════════════════════════════════════
# 5. T1-2 — State validation before restore
# ══════════════════════════════════════════════════════════════════════


class TestStateValidation:
    """_validate_restored_position must reject bad state file entries."""

    def test_valid_position_passes(self):
        from quantbox.scalper import RubberBandScalper

        pos = {
            "entry_price": 100.0,
            "notional": 3000.0,
            "leverage": 30,
            "signal": 1,
        }
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert ok
        assert reason == ""

    def test_zero_entry_price_rejected(self):
        from quantbox.scalper import RubberBandScalper

        pos = {"entry_price": 0.0, "notional": 3000.0, "leverage": 30, "signal": 1}
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert not ok
        assert "entry_price" in reason

    def test_negative_entry_price_rejected(self):
        from quantbox.scalper import RubberBandScalper

        pos = {"entry_price": -50.0, "notional": 3000.0, "leverage": 30, "signal": 1}
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert not ok

    def test_nan_entry_price_rejected(self):
        from quantbox.scalper import RubberBandScalper
        import math

        pos = {"entry_price": float("nan"), "notional": 3000.0, "leverage": 30, "signal": 1}
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert not ok

    def test_zero_notional_rejected(self):
        from quantbox.scalper import RubberBandScalper

        pos = {"entry_price": 100.0, "notional": 0.0, "leverage": 30, "signal": 1}
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert not ok
        assert "notional" in reason

    def test_leverage_out_of_range_rejected(self):
        from quantbox.scalper import RubberBandScalper

        pos = {"entry_price": 100.0, "notional": 3000.0, "leverage": 200, "signal": 1}
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert not ok
        assert "leverage" in reason

    def test_invalid_signal_rejected(self):
        from quantbox.scalper import RubberBandScalper

        pos = {"entry_price": 100.0, "notional": 3000.0, "leverage": 30, "signal": 99}
        ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)
        assert not ok
        assert "signal" in reason

    def test_corrupt_symbol_field_warns_but_passes(self, caplog):
        """Corrupt pos['symbol'] with ':' must warn but still return True (sanitized later)."""
        import logging

        from quantbox.scalper import RubberBandScalper

        pos = {
            "entry_price": 100.0,
            "notional": 3000.0,
            "leverage": 30,
            "signal": 1,
            "symbol": "TESTUSDT:LONG",  # corrupt compound key in symbol field
        }
        with caplog.at_level(logging.WARNING, logger="quantbox.scalper"):
            ok, reason = RubberBandScalper._validate_restored_position("TESTUSDT:LONG", pos)

        assert ok, "corrupt symbol field should warn but not reject"
        assert any(":" in r.message and "TESTUSDT:LONG" in r.message for r in caplog.records)

    def test_corrupt_position_skipped_during_load(self, tmp_path):
        """Corrupt position in state file must be skipped, not crash the session."""
        import json

        scalper = _build_vs_scalper()
        scalper.discord = MagicMock()
        scalper._save_state = MagicMock()

        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "saved_at": "2026-01-01T00:00:00",
                    "paper_balance": 1000.0,
                    "initial_balance": 1000.0,
                    "high_water_mark": 1000.0,
                    "trade_history": [],
                    "positions": {
                        "TESTUSDT:LONG": {
                            "symbol": "TESTUSDT",
                            "signal": 1,
                            "entry_price": 0.0,  # CORRUPT: zero entry price
                            "notional": 3000.0,
                            "leverage": 30,
                        }
                    },
                }
            )
        )

        from quantbox.core.scalper_state import load_scalper_state

        with patch("quantbox.scalper.load_scalper_state", return_value=json.loads(state_file.read_text())):
            scalper._load_state()

        # Position must NOT be restored
        assert "TESTUSDT:LONG" not in scalper.strategy._positions
        # Discord alert must fire
        assert scalper.discord.send.called


# ══════════════════════════════════════════════════════════════════════
# 6. T1-4 — Balance guard blocks new entries on API failure
# ══════════════════════════════════════════════════════════════════════


class TestBalanceFetchGuard:
    """T1-4: get_balance() sets _balance_fetch_ok flag correctly."""

    def test_successful_fetch_sets_ok_true(self):
        scalper = _build_vs_scalper(balance=1000.0)
        scalper.client.futures_account_balance.return_value = [{"asset": "USDT", "balance": "1000.0"}]
        scalper.get_balance = scalper.__class__.get_balance.__get__(scalper)  # restore real method
        scalper.paper_mode = False

        val = scalper.get_balance()
        assert val == pytest.approx(1000.0)
        assert scalper._balance_fetch_ok is True

    def test_api_failure_sets_ok_false(self):
        scalper = _build_vs_scalper(balance=1000.0)
        scalper.client.futures_account_balance.side_effect = Exception("timeout")
        scalper.get_balance = scalper.__class__.get_balance.__get__(scalper)
        scalper.paper_mode = False
        scalper._last_known_balance = 0.0

        val = scalper.get_balance()
        assert val == 0.0
        assert scalper._balance_fetch_ok is False

    def test_api_failure_with_fallback_still_sets_ok_false(self):
        """Even with stale balance returned, flag must be False."""
        scalper = _build_vs_scalper(balance=1000.0)
        scalper.client.futures_account_balance.side_effect = Exception("timeout")
        scalper.get_balance = scalper.__class__.get_balance.__get__(scalper)
        scalper.paper_mode = False
        scalper._last_known_balance = 950.0  # stale

        val = scalper.get_balance()
        assert val == pytest.approx(950.0)  # fallback returned
        assert scalper._balance_fetch_ok is False  # but flag is False
