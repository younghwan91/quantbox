"""Tests for _cancel_server_order bool return (Step 0.1).

Verifies that _cancel_server_order, _cancel_server_stop_loss, and
_cancel_server_take_profit all return bool and that benign errors
return True while unexpected errors return False.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from binance.exceptions import BinanceAPIException

from quantbox.execution.brackets import BracketMixin


def _make_scalper(paper_mode: bool = False) -> MagicMock:
    """Build a minimal mock ScalperProtocol for BracketMixin methods."""
    scalper = MagicMock()
    scalper.paper_mode = paper_mode
    return scalper


class TestCancelServerOrderBoolReturn:
    def test_returns_true_on_success(self) -> None:
        scalper = _make_scalper()
        scalper._algo_api.cancel_order.return_value = {"algoId": 123, "status": "CANCELLED"}

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "123", "stop-loss")

        assert result is True

    def test_returns_true_on_paper_mode(self) -> None:
        scalper = _make_scalper(paper_mode=True)

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "paper_sl_123", "stop-loss")

        assert result is True

    def test_returns_true_when_order_id_empty(self) -> None:
        scalper = _make_scalper()

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "", "stop-loss")

        assert result is True

    def test_returns_true_on_benign_error_unknown(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=400), 400, '{"code": -2011, "msg": "Unknown order"}')
        scalper._algo_api.cancel_order.side_effect = exc

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999", "stop-loss")

        assert result is True

    def test_returns_true_on_benign_error_minus_4000(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=400), 400, '{"code": -4000, "msg": "Invalid order id"}')
        scalper._algo_api.cancel_order.side_effect = exc

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999", "stop-loss")

        assert result is True

    def test_returns_true_on_benign_error_minus_20012(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=400), 400, '{"code": -20012, "msg": "Order already closed"}')
        scalper._algo_api.cancel_order.side_effect = exc

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999", "stop-loss")

        assert result is True

    def test_returns_true_on_benign_error_minus_25029(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=400), 400, '{"code": -25029, "msg": "Algo order not found"}')
        scalper._algo_api.cancel_order.side_effect = exc

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999", "stop-loss")

        assert result is True

    def test_returns_false_on_unexpected_api_error(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=500), 500, '{"code": -1000, "msg": "An unknown error"}')
        scalper._algo_api.cancel_order.side_effect = exc

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999", "stop-loss")

        assert result is False

    def test_returns_false_on_unexpected_exception(self) -> None:
        scalper = _make_scalper()
        scalper._algo_api.cancel_order.side_effect = RuntimeError("network failure")

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999", "stop-loss")

        assert result is False


class TestCancelServerStopLossBoolReturn:
    def test_returns_true_on_success(self) -> None:
        scalper = _make_scalper()
        scalper._algo_api.cancel_order.return_value = {"algoId": 123}
        # Wire _cancel_server_order to the real BracketMixin implementation
        scalper._cancel_server_order = lambda sym, oid, lbl="order": BracketMixin._cancel_server_order(
            scalper, sym, oid, lbl
        )

        result = BracketMixin._cancel_server_stop_loss(scalper, "BTCUSDT", "123")

        assert isinstance(result, bool)
        assert result is True

    def test_returns_false_on_failure(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=500), 500, '{"code": -1000, "msg": "error"}')
        scalper._algo_api.cancel_order.side_effect = exc
        scalper._cancel_server_order = lambda sym, oid, lbl="order": BracketMixin._cancel_server_order(
            scalper, sym, oid, lbl
        )

        result = BracketMixin._cancel_server_stop_loss(scalper, "BTCUSDT", "999")

        assert isinstance(result, bool)
        assert result is False


class TestCancelServerTakeProfitBoolReturn:
    def test_returns_true_on_success(self) -> None:
        scalper = _make_scalper()
        scalper._algo_api.cancel_order.return_value = {"algoId": 456}
        scalper._cancel_server_order = lambda sym, oid, lbl="order": BracketMixin._cancel_server_order(
            scalper, sym, oid, lbl
        )

        result = BracketMixin._cancel_server_take_profit(scalper, "ETHUSDT", "456")

        assert isinstance(result, bool)
        assert result is True

    def test_returns_false_on_failure(self) -> None:
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=500), 500, '{"code": -1000, "msg": "error"}')
        scalper._algo_api.cancel_order.side_effect = exc
        scalper._cancel_server_order = lambda sym, oid, lbl="order": BracketMixin._cancel_server_order(
            scalper, sym, oid, lbl
        )

        result = BracketMixin._cancel_server_take_profit(scalper, "ETHUSDT", "999")

        assert isinstance(result, bool)
        assert result is False


class TestCallerCompatibility:
    """Existing callers that ignore the return value still work (bool is truthy/falsy)."""

    def test_none_replacement_truthy_on_success(self) -> None:
        """Callers that previously got None (always falsy) now get True on success."""
        scalper = _make_scalper()
        scalper._algo_api.cancel_order.return_value = {}

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "123")

        # True is truthy — callers using `if result:` work correctly
        assert bool(result) is True

    def test_none_replacement_falsy_on_failure(self) -> None:
        """Callers that check the result get False on failure."""
        scalper = _make_scalper()
        exc = BinanceAPIException(MagicMock(status_code=500), 500, '{"code": -1000, "msg": "error"}')
        scalper._algo_api.cancel_order.side_effect = exc

        result = BracketMixin._cancel_server_order(scalper, "BTCUSDT", "999")

        assert bool(result) is False
