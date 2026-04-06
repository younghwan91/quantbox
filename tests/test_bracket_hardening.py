"""Tests for bracket protection hardening (Step 3).

Covers:
- _verify_algo_order_active: correct status returns
- _check_bracket_staleness: detects EXPIRED/CANCELLED SL orders
- _verify_ratchet: success path, failure path, no-SL force-close trigger
- Ratchet history logging
"""

from __future__ import annotations

from unittest.mock import MagicMock


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_scalper(paper_mode: bool = False) -> MagicMock:
    """Build a minimal ScalperProtocol mock for bracket mixin tests."""
    scalper = MagicMock()
    scalper.paper_mode = paper_mode
    scalper.discord = MagicMock()
    scalper.discord.send = MagicMock()
    scalper.strategy = MagicMock()
    scalper.strategy.active_positions = {}
    scalper.strategy._positions = {}
    return scalper


# ── _verify_algo_order_active ────────────────────────────────────────────────


class TestVerifyAlgoOrderActive:
    def test_paper_mode_always_returns_working(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper(paper_mode=True)
        result = BracketMixin._verify_algo_order_active(scalper, "12345", "BTCUSDT")
        assert result == "WORKING"

    def test_returns_algo_status_from_response(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.return_value = {"algoStatus": "WORKING"}
        result = BracketMixin._verify_algo_order_active(scalper, "12345", "BTCUSDT")
        assert result == "WORKING"

    def test_returns_expired_status(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.return_value = {"algoStatus": "EXPIRED"}
        result = BracketMixin._verify_algo_order_active(scalper, "12345", "BTCUSDT")
        assert result == "EXPIRED"

    def test_returns_cancelled_status(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.return_value = {"algoStatus": "CANCELLED"}
        result = BracketMixin._verify_algo_order_active(scalper, "99", "ETHUSDT")
        assert result == "CANCELLED"

    def test_returns_none_on_query_failure(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.side_effect = Exception("network error")
        result = BracketMixin._verify_algo_order_active(scalper, "99", "ETHUSDT")
        assert result is None

    def test_falls_back_to_status_field(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        # No algoStatus key, falls back to 'status'
        scalper._algo_api.get_order.return_value = {"status": "WORKING"}
        result = BracketMixin._verify_algo_order_active(scalper, "99", "ETHUSDT")
        assert result == "WORKING"

    def test_returns_none_for_list_response(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.return_value = ["item1"]
        result = BracketMixin._verify_algo_order_active(scalper, "99", "ETHUSDT")
        assert result is None


# ── _check_bracket_staleness ─────────────────────────────────────────────────


class TestCheckBracketStaleness:
    def _setup_position(self, scalper: MagicMock, sym: str, sl_oid: str, api_sym: str) -> dict:
        pos = {
            "symbol": api_sym,
            "server_sl_order_id": sl_oid,
            "signal": 1,
        }
        scalper.strategy.active_positions = {sym: pos}
        scalper.strategy._positions = {sym: pos}
        return pos

    def test_paper_mode_skips(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper(paper_mode=True)
        scalper.strategy.active_positions = {"BTCUSDT:LONG": {}}
        BracketMixin._check_bracket_staleness(scalper)
        scalper._algo_api.get_order.assert_not_called()

    def test_detects_expired_sl(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        pos = self._setup_position(scalper, "BTCUSDT:LONG", "111", "BTCUSDT")
        scalper._verify_algo_order_active = MagicMock(return_value="EXPIRED")

        BracketMixin._check_bracket_staleness(scalper)

        assert pos["server_sl_order_id"] is None
        scalper.discord.send.assert_called_once()
        assert "EXPIRED" in scalper.discord.send.call_args[1]["content"]

    def test_detects_cancelled_sl(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        pos = self._setup_position(scalper, "ETHUSDT:SHORT", "222", "ETHUSDT")
        scalper._verify_algo_order_active = MagicMock(return_value="CANCELLED")

        BracketMixin._check_bracket_staleness(scalper)

        assert pos["server_sl_order_id"] is None
        scalper.discord.send.assert_called_once()

    def test_working_sl_updates_verified_at(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        pos = self._setup_position(scalper, "SOLUSDT:LONG", "333", "SOLUSDT")
        scalper._verify_algo_order_active = MagicMock(return_value="WORKING")

        BracketMixin._check_bracket_staleness(scalper)

        assert pos["server_sl_order_id"] == "333"  # unchanged
        assert "bracket_sl_verified_at" in pos
        scalper.discord.send.assert_not_called()

    def test_skips_positions_without_sl(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        pos = {"symbol": "BTCUSDT", "server_sl_order_id": None}
        scalper.strategy.active_positions = {"BTCUSDT:LONG": pos}
        scalper.strategy._positions = {"BTCUSDT:LONG": pos}

        BracketMixin._check_bracket_staleness(scalper)
        scalper._algo_api.get_order.assert_not_called()

    def test_skips_paper_sl_order_ids(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        pos = {"symbol": "BTCUSDT", "server_sl_order_id": "paper_sl_123456"}
        scalper.strategy.active_positions = {"BTCUSDT:LONG": pos}
        scalper.strategy._positions = {"BTCUSDT:LONG": pos}

        BracketMixin._check_bracket_staleness(scalper)
        scalper._algo_api.get_order.assert_not_called()


# ── _verify_ratchet ───────────────────────────────────────────────────────────


class TestVerifyRatchet:
    def test_success_logs_ratchet_history(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._verify_algo_order_active = MagicMock(return_value="WORKING")
        pos = {"symbol": "BTCUSDT", "server_sl_order_id": "555"}

        result = BracketMixin._verify_ratchet(scalper, "BTCUSDT:LONG", pos, expected_price=95000.0)

        assert result is True
        assert len(pos["_ratchet_history"]) == 1
        assert pos["_ratchet_history"][0]["price"] == 95000.0
        assert pos["_ratchet_history"][0]["verified"] is True
        assert pos["last_ratchet_price"] == 95000.0
        assert "last_ratchet_at" in pos

    def test_failure_keeps_old_sl_and_alerts(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.return_value = {"algoStatus": "EXPIRED"}
        pos = {"symbol": "BTCUSDT", "server_sl_order_id": "555"}

        result = BracketMixin._verify_ratchet(scalper, "BTCUSDT:LONG", pos, expected_price=95000.0)

        assert result is False
        # SL order ID must NOT be cleared — old SL kept active
        assert pos["server_sl_order_id"] == "555"
        scalper.discord.send.assert_called_once()
        assert "래칫 검증 실패" in scalper.discord.send.call_args[1]["content"]

    def test_no_sl_at_all_triggers_critical_alert(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        pos = {"symbol": "BTCUSDT", "server_sl_order_id": None}

        result = BracketMixin._verify_ratchet(scalper, "BTCUSDT:LONG", pos, expected_price=95000.0)

        assert result is False
        scalper.discord.send.assert_called_once()
        assert "SL 없음" in scalper.discord.send.call_args[1]["content"]
        # No API call — no SL to check
        scalper._algo_api.get_order.assert_not_called()

    def test_ratchet_history_appends_multiple(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._verify_algo_order_active = MagicMock(return_value="WORKING")
        pos = {"symbol": "BTCUSDT", "server_sl_order_id": "777", "_ratchet_history": []}

        BracketMixin._verify_ratchet(scalper, "BTCUSDT:LONG", pos, expected_price=90000.0)
        BracketMixin._verify_ratchet(scalper, "BTCUSDT:LONG", pos, expected_price=92000.0)

        assert len(pos["_ratchet_history"]) == 2
        assert pos["last_ratchet_price"] == 92000.0

    def test_verify_ratchet_none_query_failure(self):
        from quantbox.execution.brackets import BracketMixin

        scalper = _make_scalper()
        scalper._algo_api.get_order.side_effect = Exception("timeout")
        pos = {"symbol": "ETHUSDT", "server_sl_order_id": "999"}

        # Query fails → status=None → not WORKING → returns False
        result = BracketMixin._verify_ratchet(scalper, "ETHUSDT:SHORT", pos, expected_price=3000.0)
        assert result is False
        # Old SL preserved
        assert pos["server_sl_order_id"] == "999"
