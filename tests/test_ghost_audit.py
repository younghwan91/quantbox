"""Tests for ghost fill audit logging (P1).

Verifies that when _reconcile_positions() detects a ghost position
(internally tracked but missing on exchange), it:
1. Queries futures_account_trades() with limit=50
2. Queries futures_get_all_orders() to identify the trigger order
3. Logs structured audit lines (ERROR level) with fill details
4. Sends enriched Discord notification
5. Falls back gracefully when API calls fail
"""

from unittest.mock import MagicMock, patch, call
import logging
import pytest

from quantbox.core.config import Config


# ─────────────────────────── Fixtures ──────────────────────────────


def _make_scalper(paper: bool = False):
    """Create a minimal RubberBandScalper for ghost audit testing."""
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
    return scalper


# ─────────────────────────── Tests ──────────────────────────────


class TestGhostFillAudit:
    """Verify ghost reconcile logs full audit trail."""

    def test_ghost_logs_fill_details(self, caplog):
        """Ghost detection should log orderId, price, qty, realizedPnl,
        commission, time, and internal state (SL/TP algo IDs)."""
        scalper = _make_scalper()

        # Open internal position (exchange will say it doesn't exist)
        scalper.strategy.open_position(
            symbol="BTCUSDT",
            signal=1,
            entry_price=100.0,
            notional=200.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id="5001",
            server_tp_order_id="6001",
        )

        # Exchange: no position
        scalper.client.futures_position_information.return_value = []

        # Fake fill from account trades
        scalper.client.futures_account_trades.return_value = [
            {
                "orderId": 77001,
                "side": "SELL",
                "price": "101.50",
                "qty": "2.0",
                "realizedPnl": "3.00",
                "commission": "0.0610",
                "commissionAsset": "USDT",
                "time": 1740000000000,  # epoch ms
                "maker": False,
                "buyer": False,
            },
        ]

        # Matching order details
        scalper.client.futures_get_all_orders.return_value = [
            {
                "orderId": 77001,
                "clientOrderId": "algo_tp_btc",
                "type": "LIMIT",
                "side": "SELL",
                "origType": "TAKE_PROFIT_MARKET",
                "status": "FILLED",
                "stopPrice": "101.50",
                "price": "0",
                "updateTime": 1740000000000,
            },
        ]

        with caplog.at_level(logging.ERROR, logger="Scalper"):
            scalper._reconcile_positions()

        log_text = caplog.text

        # Verify structured audit lines
        assert "GHOST FILL AUDIT" in log_text
        assert "orderId=77001" in log_text
        assert "price=101.5000" in log_text
        assert "qty=2.000000" in log_text
        assert "realizedPnl=+3.0000" in log_text
        assert "commission=0.061000" in log_text
        assert "TAKER" in log_text
        assert "trigger_order:" in log_text
        assert "TAKE_PROFIT_MARKET" in log_text
        assert "server_sl_id=5001" in log_text
        assert "server_tp_id=6001" in log_text

        # Position should be cleaned up
        assert not scalper.strategy.has_position("BTCUSDT")

    def test_ghost_no_fills_found(self, caplog):
        """When no matching fills exist, log that explicitly."""
        scalper = _make_scalper()

        scalper.strategy.open_position(
            symbol="ETHUSDT",
            signal=-1,
            entry_price=3000.0,
            notional=300.0,
            leverage=1,
            move_size=0.04,
        )

        scalper.client.futures_position_information.return_value = []
        scalper.client.futures_account_trades.return_value = []

        with caplog.at_level(logging.ERROR, logger="Scalper"):
            scalper._reconcile_positions()

        assert "NO MATCHING FILLS FOUND" in caplog.text
        assert not scalper.strategy.has_position("ETHUSDT")

    def test_ghost_api_failure_graceful(self, caplog):
        """Trade fetch failure => still cleans up, logs warning."""
        scalper = _make_scalper()

        scalper.strategy.open_position(
            symbol="SOLUSDT",
            signal=1,
            entry_price=50.0,
            notional=100.0,
            leverage=2,
            move_size=0.03,
        )

        scalper.client.futures_position_information.return_value = []
        scalper.client.futures_account_trades.side_effect = Exception(
            "APIError(-1021): Timestamp error",
        )

        with caplog.at_level(logging.WARNING, logger="Scalper"):
            scalper._reconcile_positions()

        assert "trade fetch failed" in caplog.text
        # Position still cleaned up even though audit data unavailable
        assert not scalper.strategy.has_position("SOLUSDT")

    def test_ghost_discord_embed_has_audit(self):
        """Discord notification should include fill audit summary."""
        scalper = _make_scalper()
        scalper.discord = MagicMock()

        scalper.strategy.open_position(
            symbol="BTCUSDT",
            signal=1,
            entry_price=100.0,
            notional=200.0,
            leverage=2,
            move_size=0.05,
            server_sl_order_id="5001",
        )

        scalper.client.futures_position_information.return_value = []
        scalper.client.futures_account_trades.return_value = [
            {
                "orderId": 88001,
                "side": "SELL",
                "price": "99.00",
                "qty": "2.0",
                "realizedPnl": "-2.00",
                "commission": "0.060",
                "commissionAsset": "USDT",
                "time": 1740000000000,
                "maker": False,
                "buyer": False,
            },
        ]
        scalper.client.futures_get_all_orders.return_value = [
            {
                "orderId": 88001,
                "clientOrderId": "algo_sl_btc",
                "type": "STOP_MARKET",
                "side": "SELL",
                "origType": "STOP_MARKET",
                "status": "FILLED",
                "stopPrice": "99.00",
                "price": "0",
                "updateTime": 1740000000000,
            },
        ]

        scalper._reconcile_positions()

        # Find Discord send calls with embed containing Ghost Fill Audit
        discord_calls = scalper.discord.send.call_args_list
        audit_embeds = [c for c in discord_calls if c[1].get("embed", {}).get("title", "").startswith("👻")]
        assert len(audit_embeds) >= 1
        embed = audit_embeds[0][1]["embed"]
        assert "Ghost Fill Audit" in embed["title"]
        assert "$99.00" in embed["description"]
        assert "STOP_MARKET" in embed["description"]
