"""PnL 정확도 테스트.

실거래 PnL 계산의 3가지 개선사항을 검증:
1. 진입가: get_price() ticker → 실제 체결 avgPrice 사용
2. 청산가: get_price() mark → 실제 체결 avgPrice 사용
3. 펀딩비: 고정 0.01% → 바이낸스 API 실시간 조회

copilot-instructions: Test-Code Parity 준수.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from quantbox.live_trader import LivePairTrader


@pytest.fixture
def mock_env(monkeypatch):
    """필수 환경변수 설정."""
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("MAX_TRADERS", "3")
    monkeypatch.setenv("LEVERAGE", "2")
    monkeypatch.setenv("ENTRY_ZSCORE", "2.0")
    monkeypatch.setenv("EXIT_ZSCORE", "0.5")
    monkeypatch.setenv("STOP_LOSS_ZSCORE", "1.5")
    monkeypatch.setenv("MAX_HOLDING_HOURS", "16")
    monkeypatch.setenv("USE_DYNAMIC_EXIT", "false")
    monkeypatch.setenv("TRADING_FEE", "0.0005")
    monkeypatch.setenv("SLIPPAGE", "0.001")
    monkeypatch.setenv("FUNDING_RATE_8H", "0.0001")
    monkeypatch.setenv("N_COINS", "10")


@pytest.fixture
def trader(mock_env):
    """LivePairTrader with mocked Binance client (live mode for API tests)."""
    with patch("quantbox.live_trader.Client"):
        t = LivePairTrader(paper_mode=False)
        t.discord = MagicMock()
        yield t


@pytest.fixture
def paper_trader(mock_env):
    """LivePairTrader in paper mode."""
    with patch("quantbox.live_trader.Client"):
        t = LivePairTrader(paper_mode=True)
        t.discord = MagicMock()
        yield t


# ============================================================
# 1. Entry Price: 실제 체결가 사용
# ============================================================

class TestEntryFillPrice:
    """진입 시 entry_price가 실제 체결 avgPrice를 사용하는지 검증."""

    def test_entry_price_uses_fill_price_not_ticker(self, trader):
        """entry_price1/2가 order['avgPrice']에서 온다."""
        slot_id = 0
        ticker_price1 = 100.0  # get_price() 반환값
        ticker_price2 = 50.0
        fill_price1 = 100.05   # 실제 체결가 (더 비쌈)
        fill_price2 = 49.95    # 실제 체결가 (더 쌈)

        trader.get_price = MagicMock(side_effect=[ticker_price1, ticker_price2])
        trader.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
            'step_size': 0.001, 'quantity_precision': 3,
            'min_notional': 5,
        })
        trader.round_quantity = MagicMock(side_effect=lambda q, *a, **kw: q)
        trader.calculate_zscore = MagicMock(return_value=-2.5)

        # place_order returns dict with avgPrice
        trader.place_order = MagicMock(side_effect=[
            {'orderId': 1, 'status': 'FILLED', 'executedQty': '1.0',
             'avgPrice': str(fill_price1), 'makerFill': True, 'chaseCount': 1},
            {'orderId': 2, 'status': 'FILLED', 'executedQty': '2.0',
             'avgPrice': str(fill_price2), 'makerFill': True, 'chaseCount': 1},
        ])

        from quantbox.strategy.pair_selector import PairCandidate
        pair = PairCandidate(symbol1='BTCUSDT', symbol2='ETHUSDT', combined_score=0.9)
        pair_info = {
            'hedge_ratio': 1.0,
            'spread_mean': 0.0,
            'spread_std': 1.0,
            'half_life': 10.0,
        }
        capital = 1000.0

        result = trader.execute_pair_trade(slot_id, pair, pair_info, capital)
        assert result is True

        pos = trader.slot_positions[slot_id]
        assert pos['entry_price1'] == fill_price1, \
            f"entry_price1 should be fill price {fill_price1}, got {pos['entry_price1']}"
        assert pos['entry_price2'] == fill_price2, \
            f"entry_price2 should be fill price {fill_price2}, got {pos['entry_price2']}"

    def test_entry_price_fallback_to_ticker_when_no_avgprice(self, trader):
        """avgPrice가 0이거나 없으면 ticker price로 fallback."""
        slot_id = 0
        ticker_price1 = 100.0
        ticker_price2 = 50.0

        trader.get_price = MagicMock(side_effect=[ticker_price1, ticker_price2])
        trader.get_symbol_info = MagicMock(return_value={
            'tick_size': 0.01, 'price_precision': 2,
            'step_size': 0.001, 'quantity_precision': 3,
            'min_notional': 5,
        })
        trader.round_quantity = MagicMock(side_effect=lambda q, *a, **kw: q)
        trader.calculate_zscore = MagicMock(return_value=-2.5)

        # avgPrice = '0' → should fallback
        trader.place_order = MagicMock(side_effect=[
            {'orderId': 1, 'status': 'FILLED', 'executedQty': '1.0',
             'avgPrice': '0', 'makerFill': True, 'chaseCount': 1},
            {'orderId': 2, 'status': 'FILLED', 'executedQty': '2.0',
             'avgPrice': '0', 'makerFill': True, 'chaseCount': 1},
        ])

        from quantbox.strategy.pair_selector import PairCandidate
        pair = PairCandidate(symbol1='BTCUSDT', symbol2='ETHUSDT', combined_score=0.9)
        pair_info = {
            'hedge_ratio': 1.0, 'spread_mean': 0.0,
            'spread_std': 1.0, 'half_life': 10.0,
        }

        trader.execute_pair_trade(slot_id, pair, pair_info, 1000.0)
        pos = trader.slot_positions[slot_id]
        assert pos['entry_price1'] == ticker_price1
        assert pos['entry_price2'] == ticker_price2


# ============================================================
# 2. Exit Price: 실제 청산 체결가 사용
# ============================================================

class TestExitFillPrice:
    """청산 시 PnL이 실제 체결 avgPrice 기반인지 검증."""

    def test_close_slot_returns_fill_prices(self, trader):
        """close_slot_positions가 실제 체결가를 반환한다."""
        slot_id = 0
        trader.slots[slot_id].position_open = True
        trader.slots[slot_id].current_pair = ("BTCUSDT", "ETHUSDT")
        trader.slot_positions[slot_id] = {
            'symbol1': 'BTCUSDT', 'symbol2': 'ETHUSDT',
            'qty1': 0.01, 'qty2': -0.1,
        }
        trader.slot_entry_times[slot_id] = datetime.now(timezone.utc)

        fill1 = {'orderId': 1, 'avgPrice': '51000.0', 'executedQty': '0.01'}
        fill2 = {'orderId': 2, 'avgPrice': '3200.0', 'executedQty': '0.1'}
        trader.close_position = MagicMock(side_effect=[fill1, fill2])

        exit_prices = trader.close_slot_positions(slot_id)

        assert exit_prices['exit_price1'] == 51000.0
        assert exit_prices['exit_price2'] == 3200.0

    def test_close_slot_returns_zero_on_failure(self, trader):
        """청산 실패 시 exit_price=0."""
        slot_id = 1
        trader.slots[slot_id].position_open = True
        trader.slots[slot_id].current_pair = ("AAUSDT", "BBUSDT")
        trader.slot_positions[slot_id] = {
            'symbol1': 'AAUSDT', 'symbol2': 'BBUSDT',
            'qty1': 1.0, 'qty2': -1.0,
        }
        trader.slot_entry_times[slot_id] = datetime.now(timezone.utc)

        # Leg1 succeeds, Leg2 fails
        trader.close_position = MagicMock(side_effect=[
            {'orderId': 1, 'avgPrice': '10.0', 'executedQty': '1.0'},
            None,  # failed
        ])

        exit_prices = trader.close_slot_positions(slot_id)
        assert exit_prices['exit_price1'] == 10.0
        assert exit_prices['exit_price2'] == 0.0

    def test_close_position_returns_order_dict(self, trader):
        """close_position이 bool이 아닌 dict를 반환한다."""
        trader.get_symbol_info = MagicMock(return_value={
            'step_size': 0.001, 'quantity_precision': 3,
        })
        expected_order = {
            'orderId': 999, 'status': 'FILLED',
            'executedQty': '0.5', 'avgPrice': '100.0',
        }
        trader.place_order = MagicMock(return_value=expected_order)

        result = trader.close_position("BTCUSDT", 0.5)
        assert isinstance(result, dict)
        assert result['avgPrice'] == '100.0'

    def test_close_position_returns_none_on_failure(self, trader):
        """close_position 실패 시 None 반환."""
        trader.get_symbol_info = MagicMock(return_value={
            'step_size': 0.001, 'quantity_precision': 3,
        })
        trader.place_order = MagicMock(return_value=None)

        result = trader.close_position("BTCUSDT", 0.5)
        assert result is None


# ============================================================
# 3. Funding Rate: 실시간 조회
# ============================================================

class TestFundingRate:
    """펀딩비 계산이 실시간 API 값을 사용하는지 검증."""

    def test_fetch_funding_rate_from_api(self, trader):
        """_fetch_funding_rate가 Binance API에서 가져온다."""
        trader.client.futures_mark_price = MagicMock(return_value={
            'lastFundingRate': '0.0003',  # 0.03%
        })

        rate = trader._fetch_funding_rate("BTCUSDT")
        assert rate == 0.0003
        trader.client.futures_mark_price.assert_called_once_with(symbol="BTCUSDT")

    def test_fetch_funding_rate_fallback_on_error(self, trader):
        """API 실패 시 config 기본값으로 fallback."""
        trader.client.futures_mark_price = MagicMock(side_effect=Exception("API error"))

        rate = trader._fetch_funding_rate("BTCUSDT")
        assert rate == trader.config['funding_rate_8h']  # 0.0001

    def test_fetch_funding_rate_fallback_on_zero(self, trader):
        """펀딩비가 0이면 config 기본값 사용."""
        trader.client.futures_mark_price = MagicMock(return_value={
            'lastFundingRate': '0',
        })

        rate = trader._fetch_funding_rate("BTCUSDT")
        # rate == 0 → fallback to config
        assert rate == trader.config['funding_rate_8h']  # 0.0001

    def test_calculate_funding_cost_live(self, trader):
        """_calculate_funding_cost가 양 레그의 실시간 펀딩비를 사용."""
        # BTC: 0.03%, ETH: 0.05%
        trader.client.futures_mark_price = MagicMock(side_effect=[
            {'lastFundingRate': '0.0003'},
            {'lastFundingRate': '0.0005'},
        ])

        cost = trader._calculate_funding_cost(
            "BTCUSDT", "ETHUSDT",
            notional=1000.0,
            holding_hours=8.0,
        )

        # avg rate = (0.0003 + 0.0005) / 2 = 0.0004
        # cost = 0.0004 × 2 legs × $1000 × 1.0 period = $0.80
        assert abs(cost - 0.80) < 0.01

    def test_calculate_funding_cost_paper_uses_config(self, paper_trader):
        """Paper 모드에서는 config 고정값 사용."""
        trader = paper_trader

        cost = trader._calculate_funding_cost(
            "BTCUSDT", "ETHUSDT",
            notional=1000.0,
            holding_hours=16.0,
        )

        # rate = 0.0001, periods = 16/8 = 2
        # cost = 0.0001 × 2 × 1000 × 2 = $0.40
        assert abs(cost - 0.40) < 0.01

    def test_calculate_funding_cost_proportional_to_hours(self, paper_trader):
        """펀딩비가 보유시간에 비례한다."""
        trader = paper_trader

        cost_8h = trader._calculate_funding_cost("A", "B", 1000.0, 8.0)
        cost_16h = trader._calculate_funding_cost("A", "B", 1000.0, 16.0)

        assert abs(cost_16h / cost_8h - 2.0) < 0.01


# ============================================================
# 4. PnL Pipeline 통합 테스트
# ============================================================

class TestPnLPipeline:
    """진입 → 청산 전체 파이프라인에서 PnL이 실제 체결가 기반인지."""

    def test_trade_history_includes_cost_breakdown(self, trader):
        """trade_history에 fee_cost, funding_cost, slip_cost 기록 확인."""
        slot_id = 0

        # Setup active position with fill prices
        trader.slots[slot_id].position_open = True
        trader.slots[slot_id].current_pair = ("BTCUSDT", "ETHUSDT")
        trader.slot_positions[slot_id] = {
            'symbol1': 'BTCUSDT',
            'symbol2': 'ETHUSDT',
            'signal': 1,
            'qty1': 0.01,
            'qty2': -0.1,
            'entry_price1': 50000.0,  # actual fill from entry
            'entry_price2': 3000.0,
            'entry_zscore': -2.5,
            'hedge_ratio': 1.0,
            'spread_mean': 0.0,
            'spread_std': 1.0,
            'half_life': 10.0,
            'capital': 500.0,
            'notional': 1000.0,
            'leverage': 2,
        }
        trader.slot_entry_times[slot_id] = datetime.now(timezone.utc) - timedelta(hours=10)

        # Mock prices and z-score for monitor
        trader.get_price = MagicMock(side_effect=[51000.0, 2950.0])
        trader.calculate_zscore = MagicMock(return_value=-0.3)  # < exit_z → close

        # Mock close → return fill prices
        exit_fill1 = {'orderId': 1, 'avgPrice': '51005.0', 'executedQty': '0.01'}
        exit_fill2 = {'orderId': 2, 'avgPrice': '2948.0', 'executedQty': '0.1'}
        trader.close_position = MagicMock(side_effect=[exit_fill1, exit_fill2])

        # Mock funding API
        trader.client.futures_mark_price = MagicMock(side_effect=[
            {'lastFundingRate': '0.0003'},
            {'lastFundingRate': '0.0005'},
        ])

        result = trader.monitor_slot(slot_id)
        assert result is False  # position closed

        # Verify trade_history
        assert len(trader.trade_history) == 1
        trade = trader.trade_history[0]

        assert 'raw_pnl' in trade
        assert 'fee_cost' in trade
        assert 'funding_cost' in trade
        assert 'slip_cost' in trade
        assert trade['fee_cost'] > 0
        assert trade['funding_cost'] > 0
        assert trade['slip_cost'] == 0.0  # live mode, no simulated slip
        assert abs(trade['pnl'] - (trade['raw_pnl'] - trade['fee_cost'] - trade['funding_cost'] - trade['slip_cost'])) < 0.01

    def test_pnl_uses_exit_fill_not_mark_price(self, paper_trader):
        """PnL이 mark price가 아닌 실제 청산 체결가 기반인지."""
        trader = paper_trader
        slot_id = 0

        trader.slots[slot_id].position_open = True
        trader.slots[slot_id].current_pair = ("AAUSDT", "BBUSDT")
        entry_p1, entry_p2 = 100.0, 50.0
        trader.slot_positions[slot_id] = {
            'symbol1': 'AAUSDT', 'symbol2': 'BBUSDT',
            'signal': 1,
            'qty1': 10.0, 'qty2': -20.0,
            'entry_price1': entry_p1, 'entry_price2': entry_p2,
            'entry_zscore': -2.5,
            'hedge_ratio': 1.0,
            'spread_mean': 0.0, 'spread_std': 1.0, 'half_life': 10.0,
            'capital': 500.0, 'notional': 1000.0, 'leverage': 2,
        }
        trader.slot_entry_times[slot_id] = datetime.now(timezone.utc) - timedelta(hours=5)

        # Mark price at monitor time
        mark_p1, mark_p2 = 105.0, 48.0
        trader.get_price = MagicMock(side_effect=[mark_p1, mark_p2])
        trader.calculate_zscore = MagicMock(return_value=-0.1)  # → close

        # Actual fill prices (differ from mark)
        fill_p1, fill_p2 = 104.5, 48.2
        trader.close_position = MagicMock(side_effect=[
            {'orderId': 1, 'avgPrice': str(fill_p1), 'executedQty': '10.0'},
            {'orderId': 2, 'avgPrice': str(fill_p2), 'executedQty': '20.0'},
        ])

        trader.monitor_slot(slot_id)

        trade = trader.trade_history[0]
        # raw_pnl should use fill prices, not mark prices
        expected_raw = (fill_p1 - entry_p1) * 10.0 + (fill_p2 - entry_p2) * (-20.0)
        assert abs(trade['raw_pnl'] - expected_raw) < 0.01, \
            f"raw_pnl should be {expected_raw:.2f} (fill-based), got {trade['raw_pnl']:.2f}"
