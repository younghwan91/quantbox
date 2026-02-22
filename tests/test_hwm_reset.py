"""
v12.4.6 Hotfix Test — HWM reset on restart with 0 positions.

Without the fix, _load_state restores stale HWM from a previous session.
If balance has dropped (due to SL fills while bot was down), the bot
immediately calculates DD > 10% and halts — even though there are no
positions and the losses are already realized.
"""

from unittest.mock import MagicMock, patch
import pytest


def _make_scalper(**config_overrides):
    """Create a minimal RubberBandScalper for testing."""
    from quantbox.scalper import RubberBandScalper

    mock_client = MagicMock()
    mock_client.futures_account_balance.return_value = [
        {'asset': 'USDT', 'availableBalance': '2052.70'},
    ]
    mock_client._request_futures_api.return_value = {'algoId': 99999}

    with patch('quantbox.scalper.load_dotenv'), \
         patch('quantbox.scalper.Client', return_value=mock_client), \
         patch('quantbox.scalper.LimitOrderChaser'):
        scalper = RubberBandScalper(paper_mode=True)

    scalper.client = mock_client
    for k, v in config_overrides.items():
        setattr(scalper.config, k, v)
    return scalper


class TestHWMResetOnRestart:
    """HWM must reset to current balance when no positions survive restart."""

    def test_hwm_resets_when_no_positions(self):
        """With 0 positions in state, HWM should match current balance."""
        from quantbox.scalper import _save_scalper_state, PAPER_STATE_FILE

        scalper = _make_scalper(max_drawdown_pct=0.10)

        # Simulate: old state with high HWM but 0 positions
        state = {
            'version': '2.0.0',
            'saved_at': '2026-02-21T00:00:00+00:00',
            'paper_mode': True,
            'paper_balance': 2052.70,
            'initial_balance': 2200.0,
            'high_water_mark': 2283.80,  # Stale HWM from previous session
            'positions': {},  # No positions
            'trade_history': [{'pnl_usd': -10.0, 'symbol': 'TEST', 'signal': 1,
                               'entry_price': 100, 'exit_price': 99,
                               'pnl_pct': -1.0, 'notional': 100,
                               'bars_held': 5, 'reason': 'stop_loss',
                               'time': '2026-02-21T00:00:00'}],
            'cycle_count': 0,
        }
        _save_scalper_state(state, filepath=PAPER_STATE_FILE)

        scalper._load_state()

        # HWM should be reset to current balance (2052.70), NOT stale 2283.80
        assert scalper._high_water_mark == pytest.approx(2052.70, abs=0.01), (
            f"HWM should be current balance 2052.70, got {scalper._high_water_mark}"
        )

        # Trade history should be cleared — old trades are settled
        assert scalper._trade_history == [], (
            "Trade history should be empty for fresh session with 0 positions"
        )

        # DD check should NOT trigger halt
        risk_blocked = scalper._check_risk_limits(2052.70)
        assert risk_blocked is False, "Should not halt with HWM == balance"
        assert scalper._drawdown_halt is False

    def test_hwm_preserved_when_positions_exist(self):
        """With active positions in state, HWM should be preserved."""
        from quantbox.scalper import _save_scalper_state, PAPER_STATE_FILE

        scalper = _make_scalper(max_drawdown_pct=0.10)

        state = {
            'version': '2.0.0',
            'saved_at': '2026-02-21T00:00:00+00:00',
            'paper_mode': True,
            'paper_balance': 2052.70,
            'initial_balance': 2200.0,
            'high_water_mark': 2283.80,
            'positions': {
                'ETHUSDT': {
                    'signal': 1, 'entry_price': 2500.0, 'notional': 100.0,
                    'leverage': 2, 'move_size': 0.05, 'bars_held': 3,
                    'best_pnl': 0.01, 'partial_closed': False,
                    'server_sl_order_id': 'sl_1', 'server_tp_order_id': None,
                    'entry_stop_mult': 0.382,
                },
            },
            'trade_history': [],
            'cycle_count': 0,
        }
        _save_scalper_state(state, filepath=PAPER_STATE_FILE)

        scalper._load_state()

        # HWM should be PRESERVED because there's an active position
        assert scalper._high_water_mark == pytest.approx(2283.80, abs=0.01), (
            f"HWM should be preserved at 2283.80, got {scalper._high_water_mark}"
        )

    def test_trade_history_cleared_when_no_positions(self):
        """Trade history should be cleared when no positions exist."""
        from quantbox.scalper import _save_scalper_state, PAPER_STATE_FILE

        scalper = _make_scalper()

        trades = [
            {'pnl_usd': 15.0, 'symbol': 'A', 'signal': 1, 'entry_price': 100,
             'exit_price': 101, 'pnl_pct': 1.0, 'notional': 100,
             'bars_held': 5, 'reason': 'target', 'time': '2026-02-21T00:00:00'},
            {'pnl_usd': -10.0, 'symbol': 'B', 'signal': -1, 'entry_price': 50,
             'exit_price': 51, 'pnl_pct': -2.0, 'notional': 50,
             'bars_held': 3, 'reason': 'stop_loss', 'time': '2026-02-21T01:00:00'},
        ]
        state = {
            'version': '2.0.0', 'saved_at': '2026-02-21T00:00:00+00:00',
            'paper_mode': True, 'paper_balance': 2000.0,
            'initial_balance': 2000.0, 'high_water_mark': 2500.0,
            'positions': {}, 'trade_history': trades, 'cycle_count': 0,
        }
        _save_scalper_state(state, filepath=PAPER_STATE_FILE)

        scalper._load_state()

        # Trade history cleared — these trades are from a finished session
        assert scalper._trade_history == []
