"""T2-3: SL/TP bracket safety tests.

Covers:
1. SL price guard — _place_server_stop_loss() returns None when entry_price=0
2. TP price guard — _place_server_take_profit() returns None when entry_price=0
3. WOULD_TRIGGER sentinel — _place_algo_order returning "WOULD_TRIGGER" propagates
4. _check_order_filled returns filled dict on TRIGGERED/FINISHED algo status
5. Paper mode SL returns "paper_sl_" prefix string
6. Paper mode TP returns "paper_tp_" prefix string
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.test_server_oco import _make_scalper


# ─────────────────────────── Helpers ──────────────────────────────

_SYM_INFO = {"tick_size": 0.01, "price_precision": 2}

_SL_KWARGS = dict(
    symbol="TESTUSDT",
    signal=1,
    quantity=30.0,
    entry_price=100.0,
    move_size=0.03,
    stop_mult=1.0,
    position_side="LONG",
)

_TP_KWARGS = dict(
    symbol="TESTUSDT",
    signal=1,
    quantity=30.0,
    entry_price=100.0,
    move_size=0.03,
    target_retrace=4.0,
    position_side="LONG",
)


def _live_scalper(**overrides):
    """Live-mode RubberBand scalper with mocked client."""
    sc = _make_scalper(paper=False, **overrides)
    sc.client._request_futures_api.return_value = {"algoId": 12345}
    sc.discord = MagicMock()
    sc.get_symbol_info = MagicMock(return_value=_SYM_INFO)
    return sc


# ─────────────────────────── Tests ────────────────────────────────


class TestSLGuard:
    """_place_server_stop_loss entry_price guard."""

    def test_sl_returns_none_when_entry_price_zero(self):
        """Live mode: entry_price=0 must return None (T2-1 guard)."""
        sc = _live_scalper()
        result = sc._place_server_stop_loss(
            **{**_SL_KWARGS, "entry_price": 0.0},
        )
        assert result is None

    def test_sl_returns_str_on_valid_entry_price(self):
        """Live mode: valid entry returns an algoId string."""
        sc = _live_scalper()
        result = sc._place_server_stop_loss(**_SL_KWARGS)
        assert result is not None
        assert isinstance(result, str)


class TestTPGuard:
    """_place_server_take_profit entry_price guard."""

    def test_tp_returns_none_when_entry_price_zero(self):
        """Live mode: entry_price=0 must return None (T2-1 guard)."""
        sc = _live_scalper()
        result = sc._place_server_take_profit(
            **{**_TP_KWARGS, "entry_price": 0.0},
        )
        assert result is None

    def test_tp_returns_str_on_valid_entry_price(self):
        """Live mode: valid entry returns an algoId string."""
        sc = _live_scalper()
        result = sc._place_server_take_profit(**_TP_KWARGS)
        assert result is not None
        assert isinstance(result, str)


class TestWouldTriggerSentinel:
    """WOULD_TRIGGER propagation from _place_algo_order."""

    def test_sl_returns_would_trigger_sentinel(self):
        """-2021 path: _place_algo_order returns WOULD_TRIGGER → SL returns it."""
        sc = _live_scalper()
        sc._place_algo_order = MagicMock(return_value="WOULD_TRIGGER")
        result = sc._place_server_stop_loss(**_SL_KWARGS)
        assert result == "WOULD_TRIGGER"

    def test_tp_does_not_propagate_would_trigger(self):
        """TP has no sentinel logic; None from _place_algo_order returns None."""
        sc = _live_scalper()
        sc._place_algo_order = MagicMock(return_value=None)
        result = sc._place_server_take_profit(**_TP_KWARGS)
        assert result is None


class TestCheckOrderFilled:
    """_check_order_filled behaviour for algo order states."""

    def _setup_live_sc_with_algo_status(self, status: str) -> object:
        sc = _live_scalper()
        sc.client._request_futures_api.return_value = {
            "algoStatus": status,
            "actualPrice": "100.5",
            "actualQty": "30.0",
            "triggerPrice": "100.0",
            "orderType": "STOP",
        }
        # No account trades → fallback to actualPrice
        sc.client.futures_account_trades.return_value = []
        return sc

    def test_triggered_status_returns_filled_dict(self):
        """TRIGGERED algoStatus → returns dict with status=FILLED."""
        sc = self._setup_live_sc_with_algo_status("TRIGGERED")
        result = sc._check_order_filled("TESTUSDT", "99999")
        assert result is not None
        assert result["status"] == "FILLED"
        assert "avgPrice" in result
        assert "executedQty" in result

    def test_finished_status_returns_filled_dict(self):
        """FINISHED algoStatus → returns dict with status=FILLED."""
        sc = self._setup_live_sc_with_algo_status("FINISHED")
        result = sc._check_order_filled("TESTUSDT", "99999")
        assert result is not None
        assert result["status"] == "FILLED"

    def test_working_status_returns_none(self):
        """WORKING algoStatus → order not yet filled, returns None."""
        sc = self._setup_live_sc_with_algo_status("WORKING")
        result = sc._check_order_filled("TESTUSDT", "99999")
        assert result is None

    def test_cancelled_status_returns_cancelled_dict(self):
        """CANCELLED algoStatus → returns dict with status=CANCELLED."""
        sc = self._setup_live_sc_with_algo_status("CANCELLED")
        result = sc._check_order_filled("TESTUSDT", "99999")
        assert result is not None
        assert result["status"] == "CANCELLED"

    def test_paper_mode_returns_none(self):
        """Paper mode: _check_order_filled always returns None."""
        sc = _make_scalper(paper=True)
        result = sc._check_order_filled("TESTUSDT", "paper_sl_123")
        assert result is None

    def test_empty_order_id_returns_none(self):
        """Empty/None order_id returns None without API call."""
        sc = _live_scalper()
        assert sc._check_order_filled("TESTUSDT", "") is None
        assert sc._check_order_filled("TESTUSDT", None) is None


class TestPaperModeBrackets:
    """Paper mode SL/TP return paper_ prefix strings."""

    def test_paper_sl_returns_paper_prefix(self):
        """Paper mode SL must start with 'paper_sl_'."""
        sc = _make_scalper(paper=True)
        result = sc._place_server_stop_loss(**_SL_KWARGS)
        assert result is not None
        assert result.startswith("paper_sl_")

    def test_paper_tp_returns_paper_prefix(self):
        """Paper mode TP must start with 'paper_tp_'."""
        sc = _make_scalper(paper=True)
        result = sc._place_server_take_profit(**_TP_KWARGS)
        assert result is not None
        assert result.startswith("paper_tp_")

    def test_paper_sl_short_signal(self):
        """Paper SL for SHORT signal also returns paper_sl_ prefix."""
        sc = _make_scalper(paper=True)
        result = sc._place_server_stop_loss(**{**_SL_KWARGS, "signal": -1, "position_side": "SHORT"})
        assert result is not None
        assert result.startswith("paper_sl_")

    def test_paper_tp_short_signal(self):
        """Paper TP for SHORT signal also returns paper_tp_ prefix."""
        sc = _make_scalper(paper=True)
        result = sc._place_server_take_profit(**{**_TP_KWARGS, "signal": -1, "position_side": "SHORT"})
        assert result is not None
        assert result.startswith("paper_tp_")
