"""
Tests for event system.
"""

import pytest
from datetime import datetime

from quantbox.core.events import (
    Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    RiskEvent, PositionEvent, EventQueue, SignalType, OrderType, OrderSide
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
        assert SignalType.CLOSE_ALL.value == "CLOSE_ALL"


class TestMarketEvent:
    """Tests for MarketEvent."""
    
    def test_creation(self):
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            data={'close': 50000.0}
        )
        assert event.type == EventType.MARKET
        assert event.symbol == "BTCUSDT"
        assert event.data['close'] == 50000.0
    
    def test_str_representation(self):
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            data={}
        )
        str_rep = str(event)
        assert "MARKET" in str_rep
        assert "BTCUSDT" in str_rep


class TestSignalEvent:
    """Tests for SignalEvent."""
    
    def test_long_signal(self):
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=0.8
        )
        assert event.signal_type == SignalType.LONG
        assert event.strength == 0.8
    
    def test_with_metadata(self):
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.SHORT,
            strength=0.5,
            metadata={'z_score': -2.5}
        )
        assert event.metadata['z_score'] == -2.5


class TestOrderEvent:
    """Tests for OrderEvent."""
    
    def test_market_order(self):
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1
        )
        assert event.order_type == OrderType.MARKET
        assert event.side == OrderSide.BUY
        assert event.quantity == 0.1
    
    def test_limit_order(self):
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0
        )
        assert event.order_type == OrderType.LIMIT
        assert event.price == 51000.0


class TestFillEvent:
    """Tests for FillEvent."""
    
    def test_fill_creation(self):
        event = FillEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            commission=2.5
        )
        assert event.price == 50000.0
        assert event.commission == 2.5
    
    def test_fill_value(self):
        event = FillEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            commission=2.5
        )
        # Value = quantity * price
        expected_value = 0.1 * 50000.0
        assert event.quantity * event.price == expected_value


class TestEventQueue:
    """Tests for EventQueue."""
    
    def test_empty_queue(self):
        queue = EventQueue()
        assert queue.is_empty()
        assert queue.get() is None
    
    def test_put_and_get(self):
        queue = EventQueue()
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            data={}
        )
        
        queue.put(event)
        assert not queue.is_empty()
        
        retrieved = queue.get()
        assert retrieved == event
        assert queue.is_empty()
    
    def test_fifo_order(self):
        queue = EventQueue()
        
        events = [
            MarketEvent(datetime.now(), "BTC", {}),
            MarketEvent(datetime.now(), "ETH", {}),
            MarketEvent(datetime.now(), "BNB", {}),
        ]
        
        for e in events:
            queue.put(e)
        
        for expected in events:
            assert queue.get().symbol == expected.symbol
    
    def test_clear(self):
        queue = EventQueue()
        queue.put(MarketEvent(datetime.now(), "BTC", {}))
        queue.put(MarketEvent(datetime.now(), "ETH", {}))
        
        queue.clear()
        assert queue.is_empty()
    
    def test_size(self):
        queue = EventQueue()
        assert len(queue) == 0
        
        queue.put(MarketEvent(datetime.now(), "BTC", {}))
        assert len(queue) == 1
        
        queue.put(MarketEvent(datetime.now(), "ETH", {}))
        assert len(queue) == 2
