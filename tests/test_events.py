"""
Tests for event system.
"""

import pytest
from datetime import datetime

from quantbox.core.events import (
    Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    SignalType, OrderType, OrderSide, EventQueue, PositionEvent, RiskEvent,
    create_pair_signal, create_pair_orders
)


class TestEventTypes:
    """Tests for event type enums."""
    
    def test_event_types(self):
        assert EventType.MARKET.value == "MARKET"
        assert EventType.SIGNAL.value == "SIGNAL"
        assert EventType.ORDER.value == "ORDER"
        assert EventType.FILL.value == "FILL"
    
    def test_signal_types(self):
        assert SignalType.LONG.value == "LONG"
        assert SignalType.SHORT.value == "SHORT"
        assert SignalType.EXIT_LONG.value == "EXIT_LONG"
        assert SignalType.EXIT_SHORT.value == "EXIT_SHORT"


class TestMarketEvent:
    """Tests for MarketEvent."""
    
    def test_creation(self):
        event = MarketEvent(
            symbol="BTCUSDT",
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=1000.0
        )
        
        assert event.type == EventType.MARKET
        assert event.symbol == "BTCUSDT"
        assert event.close == 102.0
    
    def test_with_data_dict(self):
        event = MarketEvent(
            symbol="ETHUSDT",
            data={'extra_field': 'value'}
        )
        assert event.data['extra_field'] == 'value'


class TestSignalEvent:
    """Tests for SignalEvent."""
    
    def test_long_signal(self):
        event = SignalEvent(
            signal_type=SignalType.LONG,
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            z_score=2.5,
            hedge_ratio=0.5
        )
        
        assert event.type == EventType.SIGNAL
        assert event.signal_type == SignalType.LONG
        assert event.symbol1 == "BTCUSDT"
        assert event.symbol2 == "ETHUSDT"
        assert event.z_score == 2.5
    
    def test_with_metadata(self):
        event = SignalEvent(
            signal_type=SignalType.SHORT,
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            z_score=-2.5,
            hedge_ratio=0.5,
            metadata={'strategy': 'pair_trading'}
        )
        
        assert event.metadata['strategy'] == 'pair_trading'


class TestOrderEvent:
    """Tests for OrderEvent."""
    
    def test_market_order(self):
        event = OrderEvent(
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        assert event.type == EventType.ORDER
        assert event.order_type == OrderType.MARKET
        assert event.side == OrderSide.BUY
        assert event.quantity == 1.0
    
    def test_limit_order(self):
        event = OrderEvent(
            symbol="ETHUSDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=2.0,
            price=1800.0
        )
        
        assert event.order_type == OrderType.LIMIT
        assert event.price == 1800.0


class TestFillEvent:
    """Tests for FillEvent."""
    
    def test_fill_creation(self):
        event = FillEvent(
            order_id="order123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=25.0
        )
        
        assert event.type == EventType.FILL
        assert event.fill_quantity == 1.0
        assert event.fill_price == 50000.0
        assert event.commission == 25.0
    
    def test_total_cost(self):
        event = FillEvent(
            order_id="order456",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_quantity=2.0,
            fill_price=50000.0,
            commission=50.0,
            slippage=10.0
        )
        
        # total_cost = qty * price + commission + slippage
        expected = 2.0 * 50000.0 + 50.0 + 10.0
        assert event.total_cost == expected


class TestEventQueue:
    """Tests for EventQueue."""
    
    def test_empty_queue(self):
        queue = EventQueue()
        assert queue.empty() is True
    
    def test_put_and_get(self):
        queue = EventQueue()
        event = MarketEvent(symbol="BTCUSDT")
        
        queue.put(event)
        assert queue.empty() is False
        
        retrieved = queue.get(block=False)
        assert retrieved.symbol == "BTCUSDT"
        assert queue.empty() is True
    
    def test_fifo_order(self):
        queue = EventQueue()
        
        events = [
            MarketEvent(symbol="BTC"),
            MarketEvent(symbol="ETH"),
            MarketEvent(symbol="BNB"),
        ]
        
        for event in events:
            queue.put(event)
        
        assert queue.get(block=False).symbol == "BTC"
        assert queue.get(block=False).symbol == "ETH"
        assert queue.get(block=False).symbol == "BNB"
    
    def test_register_handler(self):
        queue = EventQueue()
        results = []
        
        def handler(event):
            results.append(event.symbol)
        
        queue.register_handler(EventType.MARKET, handler)
        
        event = MarketEvent(symbol="BTCUSDT")
        queue.process(event)
        
        assert "BTCUSDT" in results


class TestPairSignalFactory:
    """Tests for pair signal factory function."""
    
    def test_create_pair_signal(self):
        signal = create_pair_signal(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            signal_type=SignalType.LONG,
            z_score=2.5,
            hedge_ratio=0.5,
            strategy_id="pair_v1"
        )
        
        assert signal.symbol1 == "BTCUSDT"
        assert signal.symbol2 == "ETHUSDT"
        assert signal.signal_type == SignalType.LONG
        assert signal.z_score == 2.5
        assert signal.hedge_ratio == 0.5


class TestPairOrders:
    """Tests for pair order creation."""
    
    def test_create_pair_orders_long(self):
        signal = SignalEvent(
            signal_type=SignalType.LONG,
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            z_score=2.5,
            hedge_ratio=0.5
        )
        
        orders = create_pair_orders(
            signal=signal,
            capital=10000.0,
            price1=50000.0,
            price2=2000.0
        )
        
        assert len(orders) == 2
        assert orders[0].symbol == "BTCUSDT"
        assert orders[0].side == OrderSide.BUY
        assert orders[1].symbol == "ETHUSDT"
        assert orders[1].side == OrderSide.SELL
    
    def test_create_pair_orders_short(self):
        signal = SignalEvent(
            signal_type=SignalType.SHORT,
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            z_score=-2.5,
            hedge_ratio=0.5
        )
        
        orders = create_pair_orders(
            signal=signal,
            capital=10000.0,
            price1=50000.0,
            price2=2000.0
        )
        
        assert len(orders) == 2
        assert orders[0].side == OrderSide.SELL  # Short spread
        assert orders[1].side == OrderSide.BUY
