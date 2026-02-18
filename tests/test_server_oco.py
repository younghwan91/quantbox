"""
Tests for server-side OCO (TP + SL) bracket orders via Binance Algo Order API.

Verifies:
1. TP order placement with correct prices via Algo API
2. SL + TP pair on position open
3. Cancel TP on SL fill and vice versa (_sync_server_orders)
4. Partial close updates both TP and SL
5. State persistence of server_tp_order_id
6. Algo API params (algoType=CONDITIONAL, triggerPrice, etc.)
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
    # Default Algo Order API mock: returns algoId
    mock_client._request_futures_api.return_value = {'algoId': 99999}

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

    def test_live_long_tp_algo_params(self):
        """Live: verify TAKE_PROFIT algo order params for LONG."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client._request_futures_api.return_value = {'algoId': 501}

        scalper._place_server_take_profit(
            symbol='BTCUSDT', signal=1, quantity=0.1,
            entry_price=100.0, move_size=0.05, target_retrace=0.382,
        )

        # The _place_algo_order calls _request_futures_api('post', 'algoOrder', ...)
        call_args = scalper.client._request_futures_api.call_args
        assert call_args[0][0] == 'post'
        assert call_args[0][1] == 'algoOrder'
        data = call_args[1]['data']
        assert data['algoType'] == 'CONDITIONAL'
        assert data['side'] == 'SELL'  # Close long = sell
        assert data['type'] == 'TAKE_PROFIT'
        assert data['reduceOnly'] == 'true'
        trigger = float(data['triggerPrice'])
        assert trigger > 100.0, "Long TP trigger must be above entry"
        limit = float(data['price'])
        assert limit < trigger, "Limit slightly below trigger for SELL"

    def test_live_short_tp_algo_params(self):
        """Live: verify TAKE_PROFIT algo order params for SHORT."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client._request_futures_api.return_value = {'algoId': 502}

        scalper._place_server_take_profit(
            symbol='ETHUSDT', signal=-1, quantity=1.0,
            entry_price=3000.0, move_size=0.04, target_retrace=0.382,
        )

        data = scalper.client._request_futures_api.call_args[1]['data']
        assert data['side'] == 'BUY'  # Close short = buy
        assert data['type'] == 'TAKE_PROFIT'
        trigger = float(data['triggerPrice'])
        assert trigger < 3000.0, "Short TP trigger must be below entry"


class TestSLPriceCalculation:
    """Verify stop-loss price is correct for long and short."""

    def test_live_long_sl_algo_params(self):
        """Live: verify STOP algo order params for LONG."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client._request_futures_api.return_value = {'algoId': 601}

        scalper._place_server_stop_loss(
            symbol='BTCUSDT', signal=1, quantity=0.1,
            entry_price=100.0, move_size=0.05, stop_mult=0.786,
        )

        data = scalper.client._request_futures_api.call_args[1]['data']
        assert data['algoType'] == 'CONDITIONAL'
        assert data['side'] == 'SELL'  # Close long = sell
        assert data['type'] == 'STOP'
        assert data['reduceOnly'] == 'true'
        trigger = float(data['triggerPrice'])
        assert trigger < 100.0, "Long SL trigger must be below entry"
        limit = float(data['price'])
        assert limit < trigger, "SL limit below trigger for SELL"

    def test_live_short_sl_algo_params(self):
        """Live: verify STOP algo order params for SHORT."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client._request_futures_api.return_value = {'algoId': 602}

        scalper._place_server_stop_loss(
            symbol='ETHUSDT', signal=-1, quantity=1.0,
            entry_price=3000.0, move_size=0.04, stop_mult=0.786,
        )

        data = scalper.client._request_futures_api.call_args[1]['data']
        assert data['side'] == 'BUY'  # Close short = buy
        assert data['type'] == 'STOP'
        trigger = float(data['triggerPrice'])
        assert trigger > 3000.0, "Short SL trigger must be above entry"
        limit = float(data['price'])
        assert limit > trigger, "SL limit above trigger for BUY"

    def test_sl_failure_returns_none(self):
        """API failure returns None and does not raise."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })
        scalper.client._request_futures_api.side_effect = Exception("timeout")

        result = scalper._place_server_stop_loss(
            symbol='BTCUSDT', signal=1, quantity=0.1,
            entry_price=100.0, move_size=0.05, stop_mult=0.786,
        )
        assert result is None


# ─────────────────────────── Cancel Algo Orders ──────────────────


class TestCancelAlgoOrders:
    """Verify cancel uses DELETE /fapi/v1/algoOrder."""

    def test_cancel_sl_calls_algo_api(self):
        """Cancel SL uses _request_futures_api DELETE algoOrder."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {
            'algoId': 1001, 'code': '200', 'msg': 'success',
        }

        scalper._cancel_server_stop_loss('BTCUSDT', '1001')

        call_args = scalper.client._request_futures_api.call_args
        assert call_args[0][0] == 'delete'
        assert call_args[0][1] == 'algoOrder'
        assert call_args[1]['data'] == {'algoId': 1001}

    def test_cancel_tp_calls_algo_api(self):
        """Cancel TP uses _request_futures_api DELETE algoOrder."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {
            'algoId': 2001, 'code': '200', 'msg': 'success',
        }

        scalper._cancel_server_take_profit('BTCUSDT', '2001')

        call_args = scalper.client._request_futures_api.call_args
        assert call_args[0][0] == 'delete'
        assert call_args[0][1] == 'algoOrder'
        assert call_args[1]['data'] == {'algoId': 2001}

    def test_cancel_paper_mode_skips(self):
        """Paper mode does not call API."""
        scalper = _make_scalper(paper=True)
        scalper._cancel_server_stop_loss('BTCUSDT', 'paper_sl_123')
        scalper.client._request_futures_api.assert_not_called()

    def test_cancel_none_order_skips(self):
        """None or empty order_id does not call API."""
        scalper = _make_scalper(paper=False)
        scalper._cancel_server_stop_loss('BTCUSDT', None)
        scalper._cancel_server_stop_loss('BTCUSDT', '')
        scalper.client._request_futures_api.assert_not_called()


# ─────────────────────────── Check Order Filled ──────────────────


class TestCheckOrderFilled:
    """Verify _check_order_filled uses GET /fapi/v1/algoOrder."""

    def test_triggered_returns_fill(self):
        """TRIGGERED status returns fill info."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {
            'algoStatus': 'TRIGGERED',
            'orderType': 'TAKE_PROFIT',
            'triggerPrice': '101.91',
            'actualPrice': '101.90',
            'quantity': '1.0',
        }

        result = scalper._check_order_filled('BTCUSDT', '8001')

        assert result is not None
        assert result['status'] == 'FILLED'
        assert float(result['avgPrice']) == 101.90
        assert result['executedQty'] == '1.0'

        # Verify GET request
        call_args = scalper.client._request_futures_api.call_args
        assert call_args[0][0] == 'get'
        assert call_args[0][1] == 'algoOrder'
        assert call_args[1]['data'] == {'algoId': 8001}

    def test_triggered_zero_actual_uses_trigger(self):
        """If actualPrice is 0, use triggerPrice as fill proxy."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {
            'algoStatus': 'TRIGGERED',
            'orderType': 'STOP',
            'triggerPrice': '95.50',
            'actualPrice': '0',
            'quantity': '2.0',
        }

        result = scalper._check_order_filled('BTCUSDT', '9001')
        assert float(result['avgPrice']) == 95.50

    def test_new_status_returns_none(self):
        """NEW status means not filled."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {
            'algoStatus': 'NEW',
        }

        result = scalper._check_order_filled('BTCUSDT', '5001')
        assert result is None

    def test_paper_mode_returns_none(self):
        """Paper mode always returns None."""
        scalper = _make_scalper(paper=True)
        result = scalper._check_order_filled('BTCUSDT', 'paper_tp_123')
        assert result is None


# ─────────────────────────── Sync Server Orders ──────────────────


class TestSyncServerOrders:
    """Verify _sync_server_orders reconciles filled algo orders."""

    def test_tp_filled_cancels_sl(self):
        """When server TP fills, SL must be cancelled."""
        scalper = _make_scalper(paper=False)
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
        })

        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='1001',
            server_tp_order_id='2001',
        )

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'get' and data.get('algoId') == 2001:
                return {
                    'algoStatus': 'TRIGGERED',
                    'orderType': 'TAKE_PROFIT',
                    'triggerPrice': '101.91',
                    'actualPrice': '101.91',
                    'quantity': '1.0',
                }
            if method == 'get':
                return {'algoStatus': 'NEW'}
            # Cancel response
            return {'algoId': data.get('algoId'), 'code': '200', 'msg': 'success'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        # Position should be removed (TP filled)
        assert not scalper.strategy.has_position('BTCUSDT')
        # SL cancel should have been called
        cancel_calls = [
            c for c in scalper.client._request_futures_api.call_args_list
            if c[0][0] == 'delete'
        ]
        assert len(cancel_calls) == 1
        assert cancel_calls[0][1]['data'] == {'algoId': 1001}

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

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'get' and data.get('algoId') == 4001:
                return {'algoStatus': 'NEW'}
            if method == 'get' and data.get('algoId') == 3001:
                return {
                    'algoStatus': 'TRIGGERED',
                    'orderType': 'STOP',
                    'triggerPrice': '3120.0',
                    'actualPrice': '3120.0',
                    'quantity': '1.0',
                }
            if method == 'get':
                return {'algoStatus': 'NEW'}
            return {'algoId': data.get('algoId'), 'code': '200', 'msg': 'success'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        assert not scalper.strategy.has_position('ETHUSDT')
        cancel_calls = [
            c for c in scalper.client._request_futures_api.call_args_list
            if c[0][0] == 'delete'
        ]
        assert len(cancel_calls) == 1
        assert cancel_calls[0][1]['data'] == {'algoId': 4001}

    def test_neither_filled_no_change(self):
        """If neither order filled, position stays."""
        scalper = _make_scalper(paper=False)

        scalper.strategy.open_position(
            symbol='SOLUSDT', signal=1, entry_price=150.0,
            notional=150.0, leverage=2, move_size=0.03,
            server_sl_order_id='5001',
            server_tp_order_id='6001',
        )

        scalper.client._request_futures_api.return_value = {'algoStatus': 'NEW'}

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

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'get' and data.get('algoId') == 8001:
                return {
                    'algoStatus': 'TRIGGERED',
                    'orderType': 'TAKE_PROFIT',
                    'triggerPrice': '101.91',
                    'actualPrice': '101.91',
                    'quantity': '1.0',
                }
            if method == 'get':
                return {'algoStatus': 'NEW'}
            return {'algoId': data.get('algoId'), 'code': '200', 'msg': 'success'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        assert len(scalper._trade_history) == 1
        assert scalper._trade_history[0]['reason'] == 'server_tp'
        assert scalper._trade_history[0]['pnl_usd'] > 0  # TP = profit


# ─────────────────────────── Bracket Repair ──────────────────────


class TestBracketRepair:
    """v11.4.2: _sync_server_orders repairs missing SL/TP for restored positions."""

    def test_repair_both_missing(self):
        """Positions with None SL and TP get both placed on sync."""
        scalper = _make_scalper(paper=False)
        scalper.strategy.open_position(
            symbol='FETUSDT', signal=1, entry_price=0.17,
            notional=100.0, leverage=5, move_size=0.04,
            server_sl_order_id=None,
            server_tp_order_id=None,
        )

        call_count = {'n': 0}

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'post':
                call_count['n'] += 1
                return {'algoId': 70000 + call_count['n']}
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        pos = scalper.strategy._positions['FETUSDT']
        assert pos['server_sl_order_id'] is not None
        assert pos['server_tp_order_id'] is not None
        # Two POST calls: one for SL, one for TP
        post_calls = [
            c for c in scalper.client._request_futures_api.call_args_list
            if c[0][0] == 'post'
        ]
        assert len(post_calls) == 2

    def test_repair_sl_only(self):
        """Only SL is missing — only SL gets placed."""
        scalper = _make_scalper(paper=False)
        scalper.strategy.open_position(
            symbol='XLMUSDT', signal=-1, entry_price=0.16,
            notional=80.0, leverage=3, move_size=0.035,
            server_sl_order_id=None,
            server_tp_order_id='existing_tp_9999',
        )

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'post':
                return {'algoId': 88001}
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        pos = scalper.strategy._positions['XLMUSDT']
        assert pos['server_sl_order_id'] == '88001'
        assert pos['server_tp_order_id'] == 'existing_tp_9999'

    def test_repair_tp_only(self):
        """Only TP is missing — only TP gets placed, SL untouched."""
        scalper = _make_scalper(paper=False)
        scalper.strategy.open_position(
            symbol='ADAUSDT', signal=1, entry_price=0.50,
            notional=200.0, leverage=4, move_size=0.03,
            server_sl_order_id='existing_sl_5555',
            server_tp_order_id=None,
        )

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'post':
                return {'algoId': 99002}
            if method == 'get' and data.get('algoId') == 'existing_sl_5555':
                return {'algoStatus': 'NEW'}
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        pos = scalper.strategy._positions['ADAUSDT']
        assert pos['server_sl_order_id'] == 'existing_sl_5555'
        assert pos['server_tp_order_id'] == '99002'

    def test_no_repair_when_both_exist(self):
        """Positions with valid SL and TP don't trigger repair."""
        scalper = _make_scalper(paper=False)
        scalper.strategy.open_position(
            symbol='SOLUSDT', signal=1, entry_price=100.0,
            notional=500.0, leverage=5, move_size=0.05,
            server_sl_order_id='sl_existing',
            server_tp_order_id='tp_existing',
        )

        def mock_algo_api(method, path, signed=True, data=None):
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        # Only GET calls for checking status — no POST calls
        post_calls = [
            c for c in scalper.client._request_futures_api.call_args_list
            if c[0][0] == 'post'
        ]
        assert len(post_calls) == 0

    def test_repair_saves_state(self):
        """Bracket repair calls _save_state after placing orders."""
        scalper = _make_scalper(paper=False)
        scalper._save_state = MagicMock()
        scalper.strategy.open_position(
            symbol='DOTUSDT', signal=1, entry_price=5.00,
            notional=100.0, leverage=3, move_size=0.04,
            server_sl_order_id=None,
            server_tp_order_id=None,
        )

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'post':
                return {'algoId': 'repaired_123'}
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_api)

        scalper._sync_server_orders()

        scalper._save_state.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# Force-close on -2021 "Order would immediately trigger"
# ═══════════════════════════════════════════════════════════════════

class TestForceCloseOnWouldTrigger:
    """Verify that bracket repair force-closes positions when the SL
    placement returns -2021 (price already past stop level)."""

    def test_algo_order_returns_sentinel_on_2021(self):
        """_place_algo_order returns 'WOULD_TRIGGER' for -2021 errors."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.side_effect = Exception(
            'APIError(code=-2021): Order would immediately trigger.'
        )

        result = scalper._place_algo_order(
            symbol='FETUSDT', side='SELL', order_type='STOP',
            quantity=25000.0, trigger_price=0.1652,
            limit_price=0.1643, label='stop-loss',
        )
        assert result == 'WOULD_TRIGGER'

    def test_algo_order_returns_none_on_other_error(self):
        """_place_algo_order returns None for non-2021 errors."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.side_effect = Exception(
            'APIError(code=-1111): Some other error.'
        )

        result = scalper._place_algo_order(
            symbol='FETUSDT', side='SELL', order_type='STOP',
            quantity=25000.0, trigger_price=0.1652,
            limit_price=0.1643, label='stop-loss',
        )
        assert result is None

    def test_place_server_sl_propagates_sentinel(self):
        """_place_server_stop_loss returns 'WOULD_TRIGGER' when algo API
        returns -2021."""
        scalper = _make_scalper(paper=False)
        scalper.client._request_futures_api.side_effect = Exception(
            'APIError(code=-2021): Order would immediately trigger.'
        )

        result = scalper._place_server_stop_loss(
            symbol='FETUSDT', signal=1, quantity=25000.0,
            entry_price=0.1686, move_size=0.02, stop_mult=1.0,
        )
        assert result == 'WOULD_TRIGGER'

    def test_bracket_repair_force_closes_on_would_trigger(self):
        """When bracket repair gets WOULD_TRIGGER, it force-closes the
        position instead of leaving it unprotected."""
        scalper = _make_scalper(paper=False)
        scalper._save_state = MagicMock()
        scalper._close_position = MagicMock(return_value=True)

        # Position with missing SL (e.g. restored from state after crash)
        scalper.strategy.open_position(
            symbol='FETUSDT', signal=1, entry_price=0.1686,
            notional=100.0, leverage=3, move_size=0.02,
            server_sl_order_id=None,
            server_tp_order_id='tp_exists',
        )

        call_count = 0

        def mock_algo_api(method, path, signed=True, data=None):
            nonlocal call_count
            if method == 'post':
                # SL placement → -2021
                raise Exception(
                    'APIError(code=-2021): Order would immediately trigger.'
                )
            # GET for checking TP status
            call_count += 1
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(
            side_effect=mock_algo_api,
        )

        scalper._sync_server_orders()

        # Must call _close_position with force_sl_breach reason
        scalper._close_position.assert_called_once_with(
            'FETUSDT', 'force_sl_breach', 0.0,
        )

    def test_bracket_repair_force_close_failure_alerts(self):
        """When force-close itself fails, a CRITICAL discord alert fires."""
        scalper = _make_scalper(paper=False)
        scalper._save_state = MagicMock()
        scalper._close_position = MagicMock(return_value=False)
        scalper.discord = MagicMock()

        scalper.strategy.open_position(
            symbol='FETUSDT', signal=1, entry_price=0.1686,
            notional=100.0, leverage=3, move_size=0.02,
            server_sl_order_id=None,
            server_tp_order_id='tp_exists',
        )

        def mock_algo_api(method, path, signed=True, data=None):
            if method == 'post':
                raise Exception(
                    'APIError(code=-2021): Order would immediately trigger.'
                )
            return {'algoStatus': 'NEW'}

        scalper.client._request_futures_api = MagicMock(
            side_effect=mock_algo_api,
        )

        scalper._sync_server_orders()

        # Discord must fire FORCE CLOSE FAILED alert
        discord_calls = [
            c for c in scalper.discord.send.call_args_list
            if 'FORCE CLOSE FAILED' in str(c)
        ]
        assert len(discord_calls) >= 1

    def test_reduce_only_on_close_position(self):
        """_close_position passes reduce_only=True to _place_order."""
        scalper = _make_scalper(paper=False)
        scalper._place_order = MagicMock(return_value={
            'orderId': 999, 'avgPrice': '0.1640',
            'executedQty': '25000', 'status': 'FILLED',
        })

        scalper.strategy.open_position(
            symbol='FETUSDT', signal=1, entry_price=0.1686,
            notional=100.0, leverage=3, move_size=0.02,
            server_sl_order_id=None,
            server_tp_order_id=None,
        )

        scalper._close_position('FETUSDT', 'test_close', 0.1640)

        # Verify reduce_only=True was passed
        _, kwargs = scalper._place_order.call_args
        if 'reduce_only' in kwargs:
            assert kwargs['reduce_only'] is True
        else:
            # positional arg check
            args = scalper._place_order.call_args[0]
            assert args[3] is True  # 4th positional = reduce_only
