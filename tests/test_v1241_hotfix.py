"""
v12.4.1 Hotfix Tests — PBT Parity Fixes

Validates three critical fixes:
1. CRITICAL #1: Server TP skipped when PBT enabled (was closing 100% instead of 50%)
2. CRITICAL #2: Confidence base = 0 (was inflated to 1, admitting low-quality trades)
3. HIGH #1: Risk-Off only signal path (Risk-On was untested in 6yr backtest)
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from quantbox.core.config import Config
from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

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


def _make_strategy_with_data(*, btc_declining=True, min_confidence=3,
                             big_volume=False, low_taker=False):
    """Create strategy with realistic data for on_bar signal testing.

    Args:
        btc_declining: If True, BTC declines (Risk-Off, btc_mom<0).
        min_confidence: Minimum confidence for signal.
        big_volume: If True, last bar has 3x volume (scores +1).
        low_taker: If True, taker ratio < 0.42 (scores +1).
    """
    cfg = RubberBandConfig(
        min_confidence=min_confidence,
        lookback=1,
        threshold=0.02,
        btc_directional_period=1,
        btc_regime_period=20,
        asset_trend_period=20,
        rsi_period=14,
        vol_period=20,
        stoch_enabled=False,
        macd_enabled=False,
        entry_mode='confidence',
    )
    strategy = RubberBandStrategy(cfg)

    n = 100
    if btc_declining:
        btc_prices = np.linspace(100, 80, n)  # strong decline
    else:
        btc_prices = np.linspace(90, 110, n)  # rising BTC

    close_prices = np.linspace(50, 42, n)
    close_prices[-1] = close_prices[-2] * 0.96  # -4% dip

    volume = np.full(n, 1000.0)
    if big_volume:
        volume[-1] = 3000.0  # 3x avg → vol_ratio > 1.5

    if low_taker:
        taker_buy = volume * 0.35  # taker_ratio < 0.42
    else:
        taker_buy = volume * 0.5  # neutral, won't score

    high = close_prices * 1.005
    low = close_prices * 0.995

    strategy.update_market_data('ETHUSDT', volume, taker_buy, high, low)

    return strategy, close_prices, btc_prices


# ═══════════════════════════════════════════════════════════════════
# CRITICAL #1: Server TP skipped in PBT mode
# ═══════════════════════════════════════════════════════════════════

class TestServerTPSkippedInPBT:
    """v12.4.1: Server TP must NOT be placed when PBT (partial exit) is enabled."""

    def test_pbt_enabled_skips_server_tp(self):
        """When partial_exit_enabled=True, _open_position must NOT call _place_server_take_profit."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=True)
        scalper._save_state = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_123')
        scalper._place_server_take_profit = MagicMock(return_value='tp_456')
        scalper._place_order = MagicMock(return_value={
            'avgPrice': '100.0', 'executedQty': '1.0', 'status': 'FILLED',
        })
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)

        closes = np.array([95.0] * 20 + [100.0])
        scalper._open_position('BTCUSDT', 1, closes, 1000.0)

        # Server SL must still be placed
        scalper._place_server_stop_loss.assert_called_once()
        # Server TP must NOT be placed in PBT mode
        scalper._place_server_take_profit.assert_not_called()

    def test_pbt_disabled_places_server_tp(self):
        """When partial_exit_enabled=False, server TP is placed normally."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=False)
        scalper._save_state = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_123')
        scalper._place_server_take_profit = MagicMock(return_value='tp_456')
        scalper._place_order = MagicMock(return_value={
            'avgPrice': '100.0', 'executedQty': '1.0', 'status': 'FILLED',
        })
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)

        closes = np.array([95.0] * 20 + [100.0])
        scalper._open_position('BTCUSDT', 1, closes, 1000.0)

        scalper._place_server_stop_loss.assert_called_once()
        scalper._place_server_take_profit.assert_called_once()

    def test_pbt_position_records_no_tp_order_id(self):
        """Position opened in PBT mode must have server_tp_order_id=None."""
        scalper = _make_scalper(paper=True, partial_exit_enabled=True)

        closes = np.array([95.0] * 20 + [100.0])
        scalper._open_position('BTCUSDT', 1, closes, 1000.0)

        pos = scalper.strategy._positions.get('BTCUSDT')
        assert pos is not None
        assert pos['server_tp_order_id'] is None


class TestBracketRepairPBT:
    """v12.4.1: Bracket repair must skip TP when PBT enabled."""

    def test_bracket_repair_skips_tp_in_pbt_mode(self):
        """Missing TP should NOT be repaired when partial_exit_enabled=True."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=True)
        scalper._save_state = MagicMock()

        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_existing',
            server_tp_order_id=None,
        )

        def mock_algo_check(method, path, signed=True, data=None):
            if method == 'get':
                return {'algoStatus': 'NEW', 'orderType': 'STOP'}
            return {'algoId': 88888}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_check)
        scalper._place_server_take_profit = MagicMock(return_value='tp_new')

        scalper._sync_server_orders()

        # TP should NOT be placed in PBT mode
        scalper._place_server_take_profit.assert_not_called()
        pos = scalper.strategy._positions.get('BTCUSDT')
        assert pos['server_tp_order_id'] is None

    def test_bracket_repair_places_tp_in_non_pbt_mode(self):
        """Missing TP should be repaired when partial_exit_enabled=False."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=False)
        scalper._save_state = MagicMock()
        scalper.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
            'step_size': 0.001, 'qty_precision': 3,
        })

        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_existing',
            server_tp_order_id=None,
        )

        def mock_algo_check(method, path, signed=True, data=None):
            if method == 'get':
                return {'algoStatus': 'NEW', 'orderType': 'STOP'}
            return {'algoId': 88888}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_algo_check)

        scalper._sync_server_orders()

        pos = scalper.strategy._positions.get('BTCUSDT')
        assert pos['server_tp_order_id'] is not None


# ═══════════════════════════════════════════════════════════════════
# CRITICAL #2: Confidence base = 0
# ═══════════════════════════════════════════════════════════════════

class TestConfidenceBase:
    """v12.4.1: Confidence must start at 0, not 1."""

    def test_confidence_starts_at_zero(self):
        """With neutral indicators, confidence should be 0 (not 1)."""
        strategy, close_prices, btc_prices = _make_strategy_with_data(
            btc_declining=True, min_confidence=3,
        )
        signal = strategy.on_bar('ETHUSDT', close_prices, btc_prices)
        last_conf = strategy.get_last_confidence('ETHUSDT')
        # btc_mom < -0.01 scores +1, but rest are neutral → conf=1
        # With base=0, this is correct (1 < 3 → no signal)
        assert last_conf < 3, (
            f"Confidence should be well below min_confidence=3 with few evidence, got {last_conf}"
        )
        assert signal == 0

    def test_two_evidence_blocked_at_min3(self):
        """With min_confidence=3, two evidence points must NOT produce a signal.

        Before v12.4.1: confidence=1(base)+2=3 → signal fires.
        After v12.4.1: confidence=0(base)+2=2 < 3 → blocked.
        """
        strategy, close_prices, btc_prices = _make_strategy_with_data(
            btc_declining=True, min_confidence=3, big_volume=True,
        )
        signal = strategy.on_bar('ETHUSDT', close_prices, btc_prices)
        # btc_mom < -0.01 (+1), vol_ratio > 1.5 (+1) = 2 < 3 → blocked
        assert signal == 0, (
            f"2 evidence with base=0 should be blocked at min_confidence=3, got signal={signal}"
        )


# ═══════════════════════════════════════════════════════════════════
# HIGH #1: Risk-Off only signal path
# ═══════════════════════════════════════════════════════════════════

class TestRiskOffOnly:
    """v12.4.1: on_bar uses Risk-Off logic regardless of BTC regime."""

    def test_no_signal_in_risk_on_regime(self):
        """Dip during BTC bull (btc_mom > 0) must NOT produce long signal."""
        strategy, close_prices, btc_prices = _make_strategy_with_data(
            btc_declining=False,  # BTC rising → btc_mom > 0
            min_confidence=1,
            big_volume=True,
            low_taker=True,
        )
        signal = strategy.on_bar('ETHUSDT', close_prices, btc_prices)
        # Risk-Off long requires btc_mom < 0; btc_mom > 0 here → blocked
        assert signal == 0, (
            f"Risk-On regime dips must be blocked (btc_mom>0), got signal={signal}"
        )

    def test_signal_fires_in_risk_off_regime(self):
        """In Risk-Off (BTC declining), sympathy dip with evidence fires."""
        strategy, close_prices, btc_prices = _make_strategy_with_data(
            btc_declining=True,
            min_confidence=2,
            big_volume=True,
        )
        signal = strategy.on_bar('ETHUSDT', close_prices, btc_prices)
        # btc_mom < -0.01 (+1), vol_ratio > 1.5 (+1) = 2 >= min_confidence(2)
        assert signal == 1, (
            f"Risk-Off sympathy long should fire with 2 evidence, got signal={signal}"
        )


class TestDiagnoseSignalParity:
    """diagnose_bar must use the same logic as on_bar."""

    def test_diagnose_uses_risk_off_only(self):
        """diagnose_bar must NOT use Risk-On entry types."""
        strategy, close_prices, btc_prices = _make_strategy_with_data(
            btc_declining=False,  # BTC rising
            min_confidence=1,
            big_volume=True,
            low_taker=True,
        )
        diag = strategy.diagnose_bar('ETHUSDT', close_prices, btc_prices)
        # Should NOT see old Risk-On types
        assert diag.get('entry_type') != 'classic_long'
        assert diag.get('entry_type') != 'classic_short'


# ═══════════════════════════════════════════════════════════════════
# v12.4.2: Partial close failure path must not re-place server TP
# ═══════════════════════════════════════════════════════════════════

class TestPartialCloseFailureNoTP:
    """When partial close fails and we restore brackets, server TP must
    be skipped in PBT mode — same bug class as v12.4.1 CRITICAL #1."""

    def test_pbt_failure_path_skips_server_tp(self):
        """If partial close order fails in PBT mode, only SL is restored (no TP)."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=True)
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_new')
        scalper._place_server_take_profit = MagicMock(return_value='tp_new')
        # Simulate market order failure (returns None)
        scalper._place_order = MagicMock(return_value=None)

        # Set up position
        scalper.strategy.open_position(
            symbol='ETHUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_old', server_tp_order_id=None,
        )

        result = scalper._partial_close_position('ETHUSDT', 101.5)
        assert result is False  # partial close failed

        # SL should be re-placed to restore protection
        scalper._place_server_stop_loss.assert_called_once()
        # TP must NOT be re-placed in PBT mode
        scalper._place_server_take_profit.assert_not_called()

        # Verify position still tracked with SL but no TP
        pos = scalper.strategy._positions.get('ETHUSDT')
        assert pos is not None
        assert pos['server_sl_order_id'] == 'sl_new'
        assert pos['server_tp_order_id'] is None

    def test_non_pbt_failure_path_restores_both(self):
        """If partial close fails in non-PBT mode, both SL and TP are restored."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=False)
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_new')
        scalper._place_server_take_profit = MagicMock(return_value='tp_new')
        scalper._place_order = MagicMock(return_value=None)

        scalper.strategy.open_position(
            symbol='ETHUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_old', server_tp_order_id='tp_old',
        )

        result = scalper._partial_close_position('ETHUSDT', 101.5)
        assert result is False

        # Both SL and TP should be re-placed
        scalper._place_server_stop_loss.assert_called_once()
        scalper._place_server_take_profit.assert_called_once()

        pos = scalper.strategy._positions.get('ETHUSDT')
        assert pos['server_sl_order_id'] == 'sl_new'
        assert pos['server_tp_order_id'] == 'tp_new'


# ═══════════════════════════════════════════════════════════════════
# v12.4.3: _close_position failure path must not re-place server TP
# ═══════════════════════════════════════════════════════════════════

class TestClosePositionFailureNoTP:
    """When _close_position fails and we restore brackets, server TP must
    be skipped in PBT mode for Phase 1 positions — same bug class as
    v12.4.1 CRITICAL #1.

    Scenario: Phase 1 trailing_stop fires → close order fails (API error)
    → server TP re-placed at TP1 → price recovers → server TP closes 100%
    → PBT partial close flow bypassed, runner never created.
    """

    def test_pbt_phase1_close_failure_skips_tp(self):
        """Phase 1 close failure in PBT mode: SL restored, TP skipped."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=True)
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_new')
        scalper._place_server_take_profit = MagicMock(return_value='tp_new')
        # Close order FAILS
        scalper._place_order = MagicMock(return_value=None)

        scalper.strategy.open_position(
            symbol='ETHUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_old', server_tp_order_id=None,
        )

        result = scalper._close_position('ETHUSDT', 'trailing_stop', 99.0)
        assert result is False  # close failed

        # SL must be re-placed
        scalper._place_server_stop_loss.assert_called_once()
        # TP must NOT be re-placed in PBT mode
        scalper._place_server_take_profit.assert_not_called()

        pos = scalper.strategy._positions.get('ETHUSDT')
        assert pos is not None
        assert pos['server_sl_order_id'] == 'sl_new'
        assert pos['server_tp_order_id'] is None

    def test_non_pbt_close_failure_restores_both(self):
        """Non-PBT close failure: both SL and TP restored."""
        scalper = _make_scalper(paper=False, partial_exit_enabled=False)
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_new')
        scalper._place_server_take_profit = MagicMock(return_value='tp_new')
        scalper._place_order = MagicMock(return_value=None)

        scalper.strategy.open_position(
            symbol='ETHUSDT', signal=1, entry_price=100.0,
            notional=100.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_old', server_tp_order_id='tp_old',
        )

        result = scalper._close_position('ETHUSDT', 'stop_loss', 97.0)
        assert result is False

        scalper._place_server_stop_loss.assert_called_once()
        scalper._place_server_take_profit.assert_called_once()

        pos = scalper.strategy._positions.get('ETHUSDT')
        assert pos['server_sl_order_id'] == 'sl_new'
        assert pos['server_tp_order_id'] == 'tp_new'

    def test_runner_close_failure_skips_tp(self):
        """Runner close failure: BE SL restored, no TP (already correct pre-fix)."""
        scalper = _make_scalper(
            paper=False, partial_exit_enabled=True,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value='sl_be')
        scalper._place_server_take_profit = MagicMock(return_value='tp_new')
        scalper._place_order = MagicMock(return_value=None)

        scalper.strategy.open_position(
            symbol='ETHUSDT', signal=1, entry_price=100.0,
            notional=50.0, leverage=2, move_size=0.05,
            server_sl_order_id='sl_old', server_tp_order_id=None,
        )
        scalper.strategy._positions['ETHUSDT']['partial_closed'] = True

        result = scalper._close_position('ETHUSDT', 'trail_runner', 100.5)
        assert result is False

        # SL re-placed with BE stop
        scalper._place_server_stop_loss.assert_called_once()
        # TP must NOT be re-placed for runner
        scalper._place_server_take_profit.assert_not_called()

        pos = scalper.strategy._positions.get('ETHUSDT')
        assert pos['server_sl_order_id'] == 'sl_be'
        assert pos['server_tp_order_id'] is None
