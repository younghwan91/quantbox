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
        patch('quantbox.scalper.get_settings') as mock_settings,
        patch('quantbox.scalper.Client') as MockClient,
        patch('quantbox.scalper.DiscordNotifier') as MockDiscord,
        patch('quantbox.scalper.DiscordBotRunner', None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key='test_key',
            binance_api_secret='test_secret',
            binance_testnet=False,
            discord_webhook_url='',
            discord_bot_token='',
            discord_channel_id='',
            paper_balance=1000.0,
        )
        scalper = RubberBandScalper(
            paper_mode=False,
            strategy_name='vol_straddle',
            vs_config=VolStraddleConfig(hedge_mode=True),
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
            patch('quantbox.scalper.get_settings') as mock_settings,
            patch('quantbox.scalper.Client'),
            patch('quantbox.scalper.DiscordNotifier'),
            patch('quantbox.scalper.DiscordBotRunner', None),
        ):
            mock_settings.return_value = MagicMock(
                binance_api_key='k', binance_api_secret='s',
                binance_testnet=False, discord_webhook_url='',
                discord_bot_token='', discord_channel_id='',
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=True,
                strategy_name='vol_straddle',
                vs_config=VolStraddleConfig(hedge_mode=True),
            )
        scalper.client = MagicMock()
        scalper._ensure_position_mode()
        scalper.client.futures_get_position_mode.assert_not_called()

    def test_already_hedge_mode_no_change(self, mock_vs_scalper):
        """If account is already in hedge mode, no switch needed."""
        scalper = mock_vs_scalper
        scalper.client.futures_get_position_mode.return_value = {
            'dualSidePosition': True,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_get_position_mode.assert_called_once()
        scalper.client.futures_change_position_mode.assert_not_called()

    def test_switch_to_hedge_mode(self, mock_vs_scalper):
        """Switches from one-way to hedge when config says hedge."""
        scalper = mock_vs_scalper
        scalper.client.futures_get_position_mode.return_value = {
            'dualSidePosition': False,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_change_position_mode.assert_called_once_with(
            dualSidePosition=True,
        )

    def test_open_positions_prevent_switch(self, mock_vs_scalper):
        """Error -4068 (open positions) logs error + Discord alert."""
        scalper = mock_vs_scalper
        scalper.client.futures_get_position_mode.return_value = {
            'dualSidePosition': False,
        }
        resp = MagicMock(status_code=400)
        resp.text = 'APIError(code=-4068): ...'
        scalper.client.futures_change_position_mode.side_effect = (
            BinanceAPIException(resp, resp.status_code, resp.text)
        )

        scalper._ensure_position_mode()

        scalper.discord.send.assert_called_once()
        msg = scalper.discord.send.call_args[1]['content']
        assert '포지션 모드 전환 실패' in msg

    def test_one_way_mode_for_rubberband(self, tmp_path):
        """RubberBand strategy should want one-way mode."""
        with (
            patch('quantbox.scalper.get_settings') as mock_settings,
            patch('quantbox.scalper.Client'),
            patch('quantbox.scalper.DiscordNotifier'),
            patch('quantbox.scalper.DiscordBotRunner', None),
        ):
            mock_settings.return_value = MagicMock(
                binance_api_key='k', binance_api_secret='s',
                binance_testnet=False, discord_webhook_url='',
                discord_bot_token='', discord_channel_id='',
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=False,
                strategy_name='rubberband',
            )
        scalper.client = MagicMock()
        scalper.discord = MagicMock()

        # Account in one-way mode → no change needed
        scalper.client.futures_get_position_mode.return_value = {
            'dualSidePosition': False,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_change_position_mode.assert_not_called()

    def test_rubberband_switches_to_oneway_if_hedge(self, tmp_path):
        """RubberBand on hedge-mode account → switch to one-way."""
        with (
            patch('quantbox.scalper.get_settings') as mock_settings,
            patch('quantbox.scalper.Client'),
            patch('quantbox.scalper.DiscordNotifier'),
            patch('quantbox.scalper.DiscordBotRunner', None),
        ):
            mock_settings.return_value = MagicMock(
                binance_api_key='k', binance_api_secret='s',
                binance_testnet=False, discord_webhook_url='',
                discord_bot_token='', discord_channel_id='',
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=False,
                strategy_name='rubberband',
            )
        scalper.client = MagicMock()
        scalper.discord = MagicMock()

        # Account in hedge mode → needs switch to one-way
        scalper.client.futures_get_position_mode.return_value = {
            'dualSidePosition': True,
        }

        scalper._ensure_position_mode()

        scalper.client.futures_change_position_mode.assert_called_once_with(
            dualSidePosition=False,
        )


# ══════════════════════════════════════════════════════════════════════════
#  CRITICAL #2: Reconciliation Compound Keys
# ══════════════════════════════════════════════════════════════════════════


class TestReconciliationHedgeMode:
    """_reconcile_positions() must not overwrite LONG with SHORT."""

    def test_both_legs_found_no_ghost(self, mock_vs_scalper):
        """When exchange has both LONG and SHORT, no ghost detected."""
        scalper = mock_vs_scalper

        # Internal state: BTCUSDT:LONG + BTCUSDT:SHORT
        scalper.strategy._positions = {
            'BTCUSDT:LONG': {
                'symbol': 'BTCUSDT',
                'signal': 1,
                'entry_price': 50000.0,
                'notional': 100.0,
                'bars_held': 3,
                'position_side': 'LONG',
                'server_sl_order_id': None,
                'server_tp_order_id': None,
            },
            'BTCUSDT:SHORT': {
                'symbol': 'BTCUSDT',
                'signal': -1,
                'entry_price': 50000.0,
                'notional': 100.0,
                'bars_held': 3,
                'position_side': 'SHORT',
                'server_sl_order_id': None,
                'server_tp_order_id': None,
            },
        }

        # Exchange returns TWO entries for BTCUSDT
        scalper.client.futures_position_information.return_value = [
            {'symbol': 'BTCUSDT', 'positionAmt': '0.002', 'positionSide': 'LONG'},
            {'symbol': 'BTCUSDT', 'positionAmt': '-0.002', 'positionSide': 'SHORT'},
        ]

        scalper._reconcile_positions()

        # No ghost detection → no error Discord notification
        assert not any(
            'Ghost' in str(c) for c in scalper.discord.send.call_args_list
        )

    def test_one_leg_missing_ghost_detected(self, mock_vs_scalper):
        """When exchange only has LONG, SHORT leg should be ghost."""
        scalper = mock_vs_scalper

        scalper.strategy._positions = {
            'BTCUSDT:LONG': {
                'symbol': 'BTCUSDT',
                'signal': 1,
                'entry_price': 50000.0,
                'notional': 100.0,
                'bars_held': 3,
                'position_side': 'LONG',
                'server_sl_order_id': None,
                'server_tp_order_id': None,
            },
            'BTCUSDT:SHORT': {
                'symbol': 'BTCUSDT',
                'signal': -1,
                'entry_price': 50000.0,
                'notional': 100.0,
                'bars_held': 3,
                'position_side': 'SHORT',
                'server_sl_order_id': None,
                'server_tp_order_id': None,
            },
        }

        # Exchange only has LONG
        scalper.client.futures_position_information.return_value = [
            {'symbol': 'BTCUSDT', 'positionAmt': '0.002', 'positionSide': 'LONG'},
        ]

        # Mock the trade fetch for ghost audit
        scalper.client.futures_account_trades.return_value = []
        scalper._record_server_fill = MagicMock()

        scalper._reconcile_positions()

        # SHORT should be detected as ghost
        scalper._record_server_fill.assert_called_once()
        call_args = scalper._record_server_fill.call_args
        assert call_args[0][0] == 'BTCUSDT:SHORT'
        assert call_args[0][2] == 0  # qty
        assert call_args[0][3] == 'ghost_reconcile'

    def test_oneway_mode_uses_plain_key(self, mock_vs_scalper):
        """One-way mode positions use plain symbol key correctly."""
        scalper = mock_vs_scalper

        scalper.strategy._positions = {
            'ETHUSDT': {
                'symbol': 'ETHUSDT',
                'signal': 1,
                'entry_price': 3000.0,
                'notional': 60.0,
                'bars_held': 5,
                'server_sl_order_id': None,
                'server_tp_order_id': None,
                # No position_side → one-way mode
            },
        }

        scalper.client.futures_position_information.return_value = [
            {'symbol': 'ETHUSDT', 'positionAmt': '0.02', 'positionSide': 'BOTH'},
        ]

        scalper._reconcile_positions()

        # No ghost detection
        assert not any(
            'Ghost' in str(c) for c in scalper.discord.send.call_args_list
        )

    def test_qty_mismatch_uses_correct_leg(self, mock_vs_scalper):
        """Qty check uses the correct leg's amount, not the other side."""
        scalper = mock_vs_scalper

        scalper.strategy._positions = {
            'BTCUSDT:LONG': {
                'symbol': 'BTCUSDT',
                'signal': 1,
                'entry_price': 50000.0,
                'notional': 100.0,  # internal: 0.002 BTC
                'bars_held': 3,
                'position_side': 'LONG',
                'server_sl_order_id': None,
                'server_tp_order_id': None,
            },
        }

        # Exchange: LONG has 0.002, SHORT has 0.010 (different trade)
        scalper.client.futures_position_information.return_value = [
            {'symbol': 'BTCUSDT', 'positionAmt': '0.002', 'positionSide': 'LONG'},
            {'symbol': 'BTCUSDT', 'positionAmt': '-0.010', 'positionSide': 'SHORT'},
        ]

        scalper._reconcile_positions()

        # Should NOT report qty mismatch (LONG matches)
        pos = scalper.strategy._positions['BTCUSDT:LONG']
        assert pos['notional'] == 100.0  # unchanged
