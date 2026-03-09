"""
v12.4.5 Hotfix Test — _save_state after successful partial close.

Without the fix, a crash between partial close and end-of-cycle _save_state
leaves the state file with 100% notional + partial_closed=False.
On restart: bracket repair uses wrong qty, PnL is double-counted on the
runner leg, and the partial close trade is lost from history.
"""

from unittest.mock import MagicMock, patch, call
import pytest


def _make_scalper(**config_overrides):
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
        scalper = RubberBandScalper(paper_mode=True)

    scalper.client = mock_client
    for k, v in config_overrides.items():
        setattr(scalper.config, k, v)
    return scalper


class TestPartialCloseSavesState:
    """_partial_close_position must call _save_state after success."""

    def test_save_state_called_after_partial_close_success(self):
        """Verify _save_state is called immediately after partial close."""
        scalper = _make_scalper(
            partial_exit_enabled=True,
            breakeven_after_partial=True,
            partial_exit_fraction=0.5,
        )
        scalper._save_state = MagicMock()

        scalper.strategy.open_position(
            symbol="ETHUSDT",
            signal=1,
            entry_price=100.0,
            notional=100.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id=None,
        )
        # Force partial_closed=True so update_position returned 'partial_close'
        scalper.strategy._positions["ETHUSDT"]["partial_closed"] = True

        # Mock: paper mode auto-fills at orderbook price
        scalper.client.futures_orderbook_ticker.return_value = {
            "askPrice": "102.0",
            "bidPrice": "102.0",
        }

        result = scalper._partial_close_position("ETHUSDT", 102.0)
        assert result is True

        # _save_state must have been called at least once
        assert scalper._save_state.call_count >= 1, "_save_state was not called after successful partial close"

    def test_state_reflects_partial_after_save(self):
        """After partial close, saved state has reduced notional."""
        scalper = _make_scalper(
            partial_exit_enabled=True,
            breakeven_after_partial=True,
            partial_exit_fraction=0.5,
        )

        # Track what state was saved
        saved_states = []
        original_save = scalper._save_state

        def capture_save():
            pos = scalper.strategy._positions.get("ETHUSDT")
            if pos:
                saved_states.append(
                    {
                        "notional": pos["notional"],
                        "partial_closed": pos["partial_closed"],
                    }
                )

        scalper._save_state = capture_save

        scalper.strategy.open_position(
            symbol="ETHUSDT",
            signal=1,
            entry_price=100.0,
            notional=100.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id=None,
        )
        scalper.strategy._positions["ETHUSDT"]["partial_closed"] = True

        scalper.client.futures_orderbook_ticker.return_value = {
            "askPrice": "102.0",
            "bidPrice": "102.0",
        }

        scalper._partial_close_position("ETHUSDT", 102.0)

        # At least one save must capture the reduced notional
        assert len(saved_states) >= 1, "No state saved after partial close"
        last = saved_states[-1]
        assert last["notional"] < 100.0, f"Notional should be ~50 after 50% partial, got {last['notional']}"
        assert last["partial_closed"] is True

    def test_save_state_not_called_on_failure(self):
        """If partial close order fails, _save_state should NOT be called
        (state is handled by the failure path's bracket re-placement)."""
        scalper = _make_scalper(
            partial_exit_enabled=True,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_new")

        scalper.strategy.open_position(
            symbol="ETHUSDT",
            signal=1,
            entry_price=100.0,
            notional=100.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id=None,
        )

        # Paper mode returns fill result, so switch to non-paper
        # and mock _place_order to return None (failure)
        scalper.paper_mode = False
        scalper._place_order = MagicMock(return_value=None)

        result = scalper._partial_close_position("ETHUSDT", 102.0)
        assert result is False

        # _save_state should NOT be called from the success path
        # (the failure path doesn't call _save_state either — it just
        # restores brackets; end-of-cycle handles the save)
        scalper._save_state.assert_not_called()
