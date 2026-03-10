"""
Tests for hedge mode activation and reconciliation compound keys.

CRITICAL #1: _ensure_position_mode() must call futures_change_position_mode()
CRITICAL #2: _reconcile_positions() must handle LONG+SHORT for same symbol
"""

import pytest
from unittest.mock import MagicMock, patch, call

from binance.exceptions import BinanceAPIException

from quantbox.scalper import RubberBandScalper
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


@pytest.fixture
def mock_vs_scalper(tmp_path):
    """Create a vol_straddle scalper with mocked exchange client."""
    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client") as MockClient,
        patch("quantbox.scalper.DiscordNotifier") as MockDiscord,
        patch("quantbox.scalper.DiscordBotRunner", None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="test_key",
            binance_api_secret="test_secret",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token="",
            discord_channel_id="",
            paper_balance=1000.0,
        )
        scalper = RubberBandScalper(
            paper_mode=False,
            strategy_name="vol_straddle",
            vs_config=VolStraddleConfig(hedge_mode=True),
            journal_db_path=":memory:",
        )

    scalper.client = MagicMock()
    scalper.discord = MagicMock()
    return scalper


# ══════════════════════════════════════════════════════════════════════════
#  CRITICAL #1: Hedge Mode Activation
# ══════════════════════════════════════════════════════════════════════════


class TestEnsurePositionMode:
    """_ensure_position_mode() must switch Binance account to hedge mode."""

    def test_skip_in_paper_mode(self, tmp_path):
        """Paper mode never calls Binance API."""
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
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=True,
                strategy_name="vol_straddle",
                vs_config=VolStraddleConfig(hedge_mode=True),
            )
        scalper.client = MagicMock()
        scalper._ensure_position_mode()
        scalper.client.futures_get_position_mode.assert_not_called()

    def test_already_hedge_mode_no_change(self, mock_vs_scalper):
        """If account is already in hedge mode, no switch needed."""
        scalper = mock_vs_scalper
        scalper.client.futures_get_position_mode.return_value = {
            "dualSidePosition": True,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_get_position_mode.assert_called_once()
        scalper.client.futures_change_position_mode.assert_not_called()

    def test_switch_to_hedge_mode(self, mock_vs_scalper):
        """Switches from one-way to hedge when config says hedge."""
        scalper = mock_vs_scalper
        scalper.client.futures_get_position_mode.return_value = {
            "dualSidePosition": False,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_change_position_mode.assert_called_once_with(
            dualSidePosition=True,
        )

    def test_open_positions_prevent_switch(self, mock_vs_scalper):
        """Error -4068 (open positions) logs error + Discord alert."""
        scalper = mock_vs_scalper
        scalper.client.futures_get_position_mode.return_value = {
            "dualSidePosition": False,
        }
        resp = MagicMock(status_code=400)
        resp.text = "APIError(code=-4068): ..."
        scalper.client.futures_change_position_mode.side_effect = BinanceAPIException(resp, resp.status_code, resp.text)

        scalper._ensure_position_mode()

        scalper.discord.send.assert_called_once()
        msg = scalper.discord.send.call_args[1]["content"]
        assert "포지션 모드 전환 실패" in msg

    def test_one_way_mode_for_rubberband(self, tmp_path):
        """RubberBand strategy should want one-way mode."""
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
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=False,
                strategy_name="rubberband",
                journal_db_path=":memory:",
            )
        scalper.client = MagicMock()
        scalper.discord = MagicMock()

        # Account in one-way mode → no change needed
        scalper.client.futures_get_position_mode.return_value = {
            "dualSidePosition": False,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_change_position_mode.assert_not_called()

    def test_rubberband_switches_to_oneway_if_hedge(self, tmp_path):
        """RubberBand on hedge-mode account → switch to one-way."""
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
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=False,
                strategy_name="rubberband",
                journal_db_path=":memory:",
            )
        scalper.client = MagicMock()
        scalper.discord = MagicMock()

        # Account in hedge mode → needs switch to one-way
        scalper.client.futures_get_position_mode.return_value = {
            "dualSidePosition": True,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_change_position_mode.assert_called_once_with(
            dualSidePosition=False,
        )


# ══════════════════════════════════════════════════════════════════════════
#  CRITICAL #3: _load_state Compound Key Bug (Incident 2026-03-08)
# ══════════════════════════════════════════════════════════════════════════


class TestLoadStateCompoundKeyBug:
    """Ensure _load_state correctly parses compound keys from state file.

    Bug: _load_state passed 'SUIUSDT:LONG' as `symbol` to open_position,
    which set pos['symbol'] = 'SUIUSDT:LONG'. Any subsequent Binance API
    call using that symbol failed with -1022 (invalid signature) because
    ':' is not a valid character in a Binance symbol.

    Fix: _load_state now strips the position_side suffix from the key
    and passes symbol='SUIUSDT', position_side='LONG' separately.
    """

    def test_load_state_strips_long_suffix(self, mock_vs_scalper, tmp_path):
        """After _load_state, pos['symbol'] must be the pure Binance symbol."""
        scalper = mock_vs_scalper
        scalper.paper_mode = False

        # Simulate a corrupted state file where keys have :LONG/:SHORT suffix
        import json

        state = {
            "saved_at": "2026-03-08T03:00:00",
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "trade_history": [],
            "positions": {
                "SUIUSDT:LONG": {
                    "symbol": "SUIUSDT:LONG",  # corrupted field (pre-fix state)
                    "signal": 1,
                    "entry_price": 0.9015,
                    "notional": 6589.0,
                    "leverage": 30,
                    "move_size": 1.0,
                    "bars_held": 5,
                    "server_sl_order_id": None,
                    "server_tp_order_id": None,
                    "sl": 0.01,
                    "tp": 0.12,
                    "tf": "15m",
                    "max_hold": 48,
                    "is_straddle": False,
                    "position_side": None,  # also corrupted (pre-fix)
                    "trail_active": False,
                    "peak_gain": 0.0,
                },
            },
            "strategy_name": "vol_straddle",
        }
        state_file = tmp_path / "scalper_state.json"
        state_file.write_text(json.dumps(state))

        import quantbox.scalper as scalper_module
        from pathlib import Path

        original_state_file = scalper_module.STATE_FILE
        scalper_module.STATE_FILE = Path(state_file)
        try:
            scalper._load_state()
        finally:
            scalper_module.STATE_FILE = original_state_file

        pos = scalper.strategy._positions.get("SUIUSDT:LONG")
        assert pos is not None, "Position should be restored under compound key"
        assert pos["symbol"] == "SUIUSDT", f"pos['symbol'] must be pure Binance symbol, got: {pos['symbol']}"
        assert pos["position_side"] == "LONG", f"position_side must be 'LONG', got: {pos['position_side']}"

    def test_load_state_strips_short_suffix(self, mock_vs_scalper, tmp_path):
        """SHORT compound key also correctly parsed."""
        scalper = mock_vs_scalper
        scalper.paper_mode = False

        import json

        state = {
            "saved_at": "2026-03-08T03:00:00",
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "trade_history": [],
            "positions": {
                "COMPUSDT:SHORT": {
                    "symbol": "COMPUSDT:SHORT",
                    "signal": -1,
                    "entry_price": 50.0,
                    "notional": 1500.0,
                    "leverage": 30,
                    "move_size": 1.0,
                    "bars_held": 2,
                    "server_sl_order_id": "999",
                    "server_tp_order_id": "888",
                    "sl": 0.01,
                    "tp": 0.12,
                    "tf": "15m",
                    "max_hold": 48,
                    "is_straddle": False,
                    "position_side": None,
                    "trail_active": False,
                    "peak_gain": 0.0,
                },
            },
            "strategy_name": "vol_straddle",
        }
        state_file = tmp_path / "scalper_state.json"
        state_file.write_text(json.dumps(state))

        import quantbox.scalper as scalper_module
        from pathlib import Path

        original_state_file = scalper_module.STATE_FILE
        scalper_module.STATE_FILE = Path(state_file)
        try:
            scalper._load_state()
        finally:
            scalper_module.STATE_FILE = original_state_file

        pos = scalper.strategy._positions.get("COMPUSDT:SHORT")
        assert pos is not None
        assert pos["symbol"] == "COMPUSDT"
        assert pos["position_side"] == "SHORT"
        assert pos["server_sl_order_id"] == "999"

    def test_reconcile_api_sym_strips_compound_suffix(self, mock_vs_scalper):
        """_reconcile_positions must not use compound key as Binance symbol.

        If pos['symbol'] is 'SUIUSDT:LONG' (corrupted), the exchange lookup
        must still find 'SUIUSDT:LONG' in exchange_map correctly (using
        the defensive split in reconciliation.py).
        """
        scalper = mock_vs_scalper

        # Simulate corrupted state where symbol has compound suffix
        scalper.strategy._positions = {
            "SUIUSDT:LONG": {
                "symbol": "SUIUSDT:LONG",  # corrupted
                "signal": 1,
                "entry_price": 0.9015,
                "notional": 6589.0,
                "bars_held": 5,
                "position_side": "LONG",
                "server_sl_order_id": None,
                "server_tp_order_id": None,
            },
        }

        # Exchange has the position (SUIUSDT LONG)
        scalper.client.futures_position_information.return_value = [
            {"symbol": "SUIUSDT", "positionAmt": "7309.8", "positionSide": "LONG"},
            {"symbol": "SUIUSDT", "positionAmt": "0", "positionSide": "SHORT"},
        ]

        scalper._reconcile_positions()

        # Should NOT report ghost (position exists on exchange)
        assert not any("Ghost" in str(c) for c in scalper.discord.send.call_args_list)


class TestReconciliationHedgeMode:
    """_reconcile_positions() must not overwrite LONG with SHORT."""

    def test_both_legs_found_no_ghost(self, mock_vs_scalper):
        """When exchange has both LONG and SHORT, no ghost detected."""
        scalper = mock_vs_scalper

        # Internal state: BTCUSDT:LONG + BTCUSDT:SHORT
        scalper.strategy._positions = {
            "BTCUSDT:LONG": {
                "symbol": "BTCUSDT",
                "signal": 1,
                "entry_price": 50000.0,
                "notional": 100.0,
                "bars_held": 3,
                "position_side": "LONG",
                "server_sl_order_id": None,
                "server_tp_order_id": None,
            },
            "BTCUSDT:SHORT": {
                "symbol": "BTCUSDT",
                "signal": -1,
                "entry_price": 50000.0,
                "notional": 100.0,
                "bars_held": 3,
                "position_side": "SHORT",
                "server_sl_order_id": None,
                "server_tp_order_id": None,
            },
        }

        # Exchange returns TWO entries for BTCUSDT
        scalper.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.002", "positionSide": "LONG"},
            {"symbol": "BTCUSDT", "positionAmt": "-0.002", "positionSide": "SHORT"},
        ]

        scalper._reconcile_positions()

        # No ghost detection → no error Discord notification
        assert not any("Ghost" in str(c) for c in scalper.discord.send.call_args_list)

    def test_one_leg_missing_ghost_detected(self, mock_vs_scalper):
        """When exchange only has LONG, SHORT leg should be ghost."""
        scalper = mock_vs_scalper

        scalper.strategy._positions = {
            "BTCUSDT:LONG": {
                "symbol": "BTCUSDT",
                "signal": 1,
                "entry_price": 50000.0,
                "notional": 100.0,
                "bars_held": 3,
                "position_side": "LONG",
                "server_sl_order_id": None,
                "server_tp_order_id": None,
            },
            "BTCUSDT:SHORT": {
                "symbol": "BTCUSDT",
                "signal": -1,
                "entry_price": 50000.0,
                "notional": 100.0,
                "bars_held": 3,
                "position_side": "SHORT",
                "server_sl_order_id": None,
                "server_tp_order_id": None,
            },
        }

        # Exchange only has LONG
        scalper.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.002", "positionSide": "LONG"},
        ]

        # Mock the trade fetch for ghost audit
        scalper.client.futures_account_trades.return_value = []
        scalper._record_server_fill = MagicMock()

        scalper._reconcile_positions()

        # SHORT should be detected as ghost
        scalper._record_server_fill.assert_called_once()
        call_args = scalper._record_server_fill.call_args
        assert call_args[0][0] == "BTCUSDT:SHORT"
        assert call_args[0][2] == 0  # qty
        assert call_args[0][3] == "ghost_reconcile"

    def test_oneway_mode_uses_plain_key(self, mock_vs_scalper):
        """One-way mode positions use plain symbol key correctly."""
        scalper = mock_vs_scalper

        scalper.strategy._positions = {
            "ETHUSDT": {
                "symbol": "ETHUSDT",
                "signal": 1,
                "entry_price": 3000.0,
                "notional": 60.0,
                "bars_held": 5,
                "server_sl_order_id": None,
                "server_tp_order_id": None,
                # No position_side → one-way mode
            },
        }

        scalper.client.futures_position_information.return_value = [
            {"symbol": "ETHUSDT", "positionAmt": "0.02", "positionSide": "BOTH"},
        ]

        scalper._reconcile_positions()

        # No ghost detection
        assert not any("Ghost" in str(c) for c in scalper.discord.send.call_args_list)

    def test_qty_mismatch_uses_correct_leg(self, mock_vs_scalper):
        """Qty check uses the correct leg's amount, not the other side."""
        scalper = mock_vs_scalper

        scalper.strategy._positions = {
            "BTCUSDT:LONG": {
                "symbol": "BTCUSDT",
                "signal": 1,
                "entry_price": 50000.0,
                "notional": 100.0,  # internal: 0.002 BTC
                "bars_held": 3,
                "position_side": "LONG",
                "server_sl_order_id": None,
                "server_tp_order_id": None,
            },
        }

        # Exchange: LONG has 0.002, SHORT has 0.010 (different trade)
        scalper.client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.002", "positionSide": "LONG"},
            {"symbol": "BTCUSDT", "positionAmt": "-0.010", "positionSide": "SHORT"},
        ]

        scalper._reconcile_positions()

        # Should NOT report qty mismatch (LONG matches)
        pos = scalper.strategy._positions["BTCUSDT:LONG"]
        assert pos["notional"] == 100.0  # unchanged


# ══════════════════════════════════════════════════════════════════════════
#  CRITICAL #5: Hedge Mode close must NOT trigger leverage change (v14.4.2)
#  Incident: trail exit passed reduce_only=False (hedge mode), leverage=1
#  → futures_change_leverage(leverage=1) → -2028 → ABORTING order
# ══════════════════════════════════════════════════════════════════════════


class TestHedgeCloseNoLeverageChange:
    """_place_order must NOT call futures_change_leverage for close orders.

    In Hedge Mode, close orders use positionSide instead of reduceOnly,
    so reduce_only=False is passed. The guard must also check leverage > 1
    to distinguish entry (explicit leverage) from close (default leverage=1).
    """

    def test_close_order_skips_leverage_change(self, mock_vs_scalper):
        """Hedge mode close: leverage change must NOT be called."""
        scalper = mock_vs_scalper

        scalper.client.futures_orderbook_ticker.return_value = {
            "askPrice": "1.000",
            "bidPrice": "0.999",
        }
        scalper.client.futures_create_algo_order.return_value = {
            "algoId": 9999,
            "clientAlgoId": "test",
            "code": "200",
            "msg": "success",
        }
        # Simulate LimitOrderChaser fill
        scalper._chaser = MagicMock()
        scalper._chaser.chase_limit_order.return_value = MagicMock(
            filled=True,
            avg_price=0.999,
            filled_qty=100.0,
            maker_fill=True,
        )

        # close order: hedge mode → reduce_only=False, leverage defaults to 1
        result = scalper._place_order(
            "SUIUSDT",
            "SELL",
            100.0,
            reduce_only=False,  # hedge mode close
            position_side="LONG",
            # leverage not passed → defaults to 1
        )

        scalper.client.futures_change_leverage.assert_not_called()

    def test_entry_order_does_set_leverage(self, mock_vs_scalper):
        """Entry order with explicit leverage MUST change leverage."""
        scalper = mock_vs_scalper

        scalper.client.futures_change_leverage.return_value = {"leverage": 30}
        scalper._chaser = MagicMock()
        scalper._chaser.chase_limit_order.return_value = MagicMock(
            filled=True,
            avg_price=1.000,
            filled_qty=3000.0,
            maker_fill=True,
        )

        scalper._place_order(
            "SUIUSDT",
            "BUY",
            3000.0,
            leverage=30,  # explicit entry leverage
            reduce_only=False,
            position_side="LONG",
        )

        scalper.client.futures_change_leverage.assert_called_once_with(
            symbol="SUIUSDT",
            leverage=30,
        )

    def test_one_way_close_still_skips_leverage(self, mock_vs_scalper):
        """One-Way mode close (reduce_only=True) must also skip leverage change."""
        scalper = mock_vs_scalper

        scalper._chaser = MagicMock()
        scalper._chaser.chase_limit_order.return_value = MagicMock(
            filled=True,
            avg_price=1.0,
            filled_qty=10.0,
            maker_fill=True,
        )

        scalper._place_order(
            "BTCUSDT",
            "SELL",
            10.0,
            reduce_only=True,  # one-way close
        )

        scalper.client.futures_change_leverage.assert_not_called()
