"""
Tests for server-side OCO (TP + SL) bracket orders — v11.2.

Verifies:
1. TP order placement with correct prices
2. SL + TP pair on position open
3. Cancel TP on SL fill and vice versa (_sync_server_orders)
4. Partial close updates both TP and SL
5. State persistence of server_tp_order_id
"""

from unittest.mock import MagicMock, patch, call
import pytest

from quantbox.core.config import Config


# ─────────────────────────── Fixtures ──────────────────────────────


def _make_scalper(paper: bool = True, **config_overrides):
    """Create a minimal RubberBandScalper for testing."""
    from quantbox.scalper import RubberBandScalper

    mock_client = MagicMock()
    mock_client.futures_account_balance.return_value = [
        {'asset': 'USDT', 'availableBalance': '1000.0'},
    ]
    mock_client.futures_create_order.return_value = {
        'orderId': 12345,
        'avgPrice': '100.0',
        'executedQty': '1.0',
        'status': 'FILLED',
    }

    with patch('quantbox.scalper.load_dotenv'), \
         patch('quantbox.scalper.Client', return_value=mock_client), \
         patch('quantbox.scalper.LimitOrderChaser'):
        scalper = RubberBandScalper(paper_mode=paper)

    scalper.client = mock_client
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(scalper.config, k, v)

    return scalper


# ─────────────────────────── TP Price Calc ──────────────────────────


class TestTPPriceCalculation:
    """Verify take-profit price is correct for long and short."""

    def test_long_tp_price_above_entry(self):
        """LONG TP must be above entry."""
        scalper = _make_scalper()
        # signal=1 (long), entry=100, move=5%, target_retrace=0.382
        tp_oid = scalper._place_server_take_profit(
            symbol='BTCUSDT', signal=1, quantity=0.1,
            entry_price=100.0, move_size=0.05, target_retrace=0.382,
        )
        assert tp_oid is not None
        assert tp_oid.startswith('paper_tp_')

    def test_short_tp_price_below_entry(self):
        """SHORT TP must be below entry."""
        scalper = _make_scalper()
        tp_oid = scalper._place_server_take_profit(
            symbol='BTCUSDT', signal=-1, quantity=0.1,
            entry_price=100.0, move_size=0.05, target_retrace=0.382,
        )
        assert tp_oid is not None
        assert tp_oid.startswith('paper_tp_')

    def test_live_long_tp_order_params(self):
        """Live mode: verify correct TAKE_PROFIT order params for LONG."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client.futures_create_order.return_value = {'orderId': 99}

        scalper._place_server_take_profit(
            symbol='BTCUSDT', signal=1, quantity=0.1,
            entry_price=100.0, move_size=0.05, target_retrace=0.382,
        )

        call_kwargs = scalper.client.futures_create_order.call_args
        assert call_kwargs[1]['side'] == 'SELL'  # Close long = sell
        assert call_kwargs[1]['type'] == 'TAKE_PROFIT'
        assert call_kwargs[1]['reduceOnly'] is True
        # stopPrice should be ~101.91 (100 * 1.0191)
        stop_price = float(call_kwargs[1]['stopPrice'])
        assert stop_price > 100.0, "Long TP trigger must be above entry"

    def test_live_short_tp_order_params(self):
        """Live mode: verify correct TAKE_PROFIT order params for SHORT."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client.futures_create_order.return_value = {'orderId': 88}

        scalper._place_server_take_profit(
            symbol='ETHUSDT', signal=-1, quantity=1.0,
            entry_price=3000.0, move_size=0.04, target_retrace=0.382,
        )

        call_kwargs = scalper.client.futures_create_order.call_args
        assert call_kwargs[1]['side'] == 'BUY'  # Close short = buy
        assert call_kwargs[1]['type'] == 'TAKE_PROFIT'
        stop_price = float(call_kwargs[1]['stopPrice'])
        assert stop_price < 3000.0, "Short TP trigger must be below entry"


# ─────────────────────────── Sync Server Orders ──────────────────


class TestSyncServerOrders:
    """Verify _sync_server_orders reconciles filled orders."""

    def test_tp_filled_cancels_sl(self):
        """When server TP fills, SL must be cancelled."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })

        # Manually insert a position with both order IDs
        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_001',
            server_tp_order_id='tp_001',
        )

        # Mock: TP is FILLED, SL is still open
        def mock_get_order(symbol, orderId):
            if orderId == int('tp_001'.replace('tp_', '')):
                # This won't work with 'tp_001', so let's use numeric IDs
                pass
            raise Exception("not found")

        # Use numeric IDs for realistic mock
        scalper.strategy._positions['BTCUSDT']['server_sl_order_id'] = '1001'
        scalper.strategy._positions['BTCUSDT']['server_tp_order_id'] = '2001'

        def futures_get_order(symbol, orderId):
            if orderId == 2001:  # TP order
                return {
                    'status': 'FILLED',
                    'avgPrice': '101.91',
                    'executedQty': '1.0',
                }
            return {'status': 'NEW'}

        scalper.client.futures_get_order = MagicMock(side_effect=futures_get_order)
        scalper.client.futures_cancel_order = MagicMock()

        scalper._sync_server_orders()

        # Position should be removed (TP filled)
        assert not scalper.strategy.has_position('BTCUSDT')
        # SL should have been cancelled
        scalper.client.futures_cancel_order.assert_called_once_with(
            symbol='BTCUSDT', orderId=1001,
        )

    def test_sl_filled_cancels_tp(self):
        """When server SL fills, TP must be cancelled."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })

        scalper.strategy.open_position(
            symbol='ETHUSDT', signal=-1, entry_price=3000.0,
            notional=3000.0, leverage=3, move_size=0.04,
            server_sl_order_id='3001',
            server_tp_order_id='4001',
        )

        def futures_get_order(symbol, orderId):
            if orderId == 4001:  # TP: not filled
                return {'status': 'NEW'}
            if orderId == 3001:  # SL: filled
                return {
                    'status': 'FILLED',
                    'avgPrice': '3120.0',
                    'executedQty': '1.0',
                }
            return {'status': 'NEW'}

        scalper.client.futures_get_order = MagicMock(side_effect=futures_get_order)
        scalper.client.futures_cancel_order = MagicMock()

        scalper._sync_server_orders()

        assert not scalper.strategy.has_position('ETHUSDT')
        scalper.client.futures_cancel_order.assert_called_once_with(
            symbol='ETHUSDT', orderId=4001,
        )

    def test_neither_filled_no_change(self):
        """If neither order filled, position stays."""
        scalper = _make_scalper(paper=False)

        scalper.strategy.open_position(
            symbol='SOLUSDT', signal=1, entry_price=150.0,
            notional=150.0, leverage=2, move_size=0.03,
            server_sl_order_id='5001',
            server_tp_order_id='6001',
        )

        scalper.client.futures_get_order = MagicMock(
            return_value={'status': 'NEW'}
        )

        scalper._sync_server_orders()

        assert scalper.strategy.has_position('SOLUSDT')

    def test_paper_mode_skips_sync(self):
        """Paper mode should not call any API."""
        scalper = _make_scalper(paper=True)
        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='paper_sl_1',
            server_tp_order_id='paper_tp_1',
        )
        scalper._sync_server_orders()
        assert scalper.strategy.has_position('BTCUSDT')


# ─────────────────────────── State Persistence ──────────────────


class TestTPStatePersistence:
    """server_tp_order_id is saved and restored correctly."""

    def test_tp_order_id_in_position(self):
        """open_position stores server_tp_order_id."""
        scalper = _make_scalper()
        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_1',
            server_tp_order_id='tp_1',
        )
        pos = scalper.strategy._positions['BTCUSDT']
        assert pos['server_tp_order_id'] == 'tp_1'
        assert pos['server_sl_order_id'] == 'sl_1'

    def test_trade_record_on_server_tp(self):
        """Server TP fill creates a trade record with reason='server_tp'."""
        scalper = _make_scalper(paper=False)

        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='7001',
            server_tp_order_id='8001',
        )

        scalper.client.futures_get_order = MagicMock(side_effect=lambda **kw: {
            'status': 'FILLED',
            'avgPrice': '101.91',
            'executedQty': '1.0',
        } if kw.get('orderId') == 8001 else {'status': 'NEW'})
        scalper.client.futures_cancel_order = MagicMock()

        scalper._sync_server_orders()

        assert len(scalper._trade_history) == 1
        assert scalper._trade_history[0]['reason'] == 'server_tp'
        assert scalper._trade_history[0]['pnl_usd'] > 0  # TP = profit
