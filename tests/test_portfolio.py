"""
Tests for portfolio management.
"""

import pytest
from datetime import datetime

from quantbox.core.portfolio import (
    Position, PairPosition, Portfolio, PortfolioSnapshot
)
from quantbox.core.events import OrderSide


class TestPosition:
    """Tests for Position class."""
    
    def test_creation(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        assert pos.symbol == "BTCUSDT"
        assert pos.quantity == 0.1
        assert pos.entry_price == 50000.0
    
    def test_pnl_long(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        
        # Price increased
        pnl = pos.unrealized_pnl(52000.0)
        expected = 0.1 * (52000.0 - 50000.0)
        assert abs(pnl - expected) < 0.01
    
    def test_pnl_short(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.SELL
        )
        
        # Price decreased (profit for short)
        pnl = pos.unrealized_pnl(48000.0)
        expected = 0.1 * (50000.0 - 48000.0)
        assert abs(pnl - expected) < 0.01
    
    def test_value(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        
        value = pos.market_value(52000.0)
        assert abs(value - 5200.0) < 0.01
    
    def test_pnl_pct(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        
        pct = pos.unrealized_pnl_pct(55000.0)
        expected = (55000.0 - 50000.0) / 50000.0 * 100
        assert abs(pct - expected) < 0.01


class TestPairPosition:
    """Tests for PairPosition class."""
    
    def test_creation(self):
        pos = PairPosition(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            quantity1=0.1,
            quantity2=1.5,
            entry_price1=50000.0,
            entry_price2=3000.0,
            hedge_ratio=0.5,
            is_long_spread=True
        )
        assert pos.symbol1 == "BTCUSDT"
        assert pos.hedge_ratio == 0.5
        assert pos.is_long_spread is True
    
    def test_spread_pnl(self):
        pos = PairPosition(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            quantity1=0.1,
            quantity2=1.5,
            entry_price1=50000.0,
            entry_price2=3000.0,
            hedge_ratio=0.5,
            is_long_spread=True
        )
        
        pnl = pos.unrealized_pnl(52000.0, 3100.0)
        assert isinstance(pnl, float)


class TestPortfolio:
    """Tests for Portfolio class."""
    
    def test_initialization(self):
        portfolio = Portfolio(initial_capital=10000.0)
        assert portfolio.cash == 10000.0
        assert portfolio.total_equity() == 10000.0
    
    def test_add_position(self):
        portfolio = Portfolio(initial_capital=10000.0)
        
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        
        portfolio.add_position(pos)
        assert "BTCUSDT" in portfolio.positions
    
    def test_remove_position(self):
        portfolio = Portfolio(initial_capital=10000.0)
        
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        
        portfolio.add_position(pos)
        portfolio.remove_position("BTCUSDT")
        assert "BTCUSDT" not in portfolio.positions
    
    def test_update_cash(self):
        portfolio = Portfolio(initial_capital=10000.0)
        
        portfolio.update_cash(-500.0)
        assert portfolio.cash == 9500.0
        
        portfolio.update_cash(200.0)
        assert portfolio.cash == 9700.0
    
    def test_equity_calculation(self):
        portfolio = Portfolio(initial_capital=10000.0)
        
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            side=OrderSide.BUY
        )
        portfolio.add_position(pos)
        
        prices = {"BTCUSDT": 52000.0}
        equity = portfolio.total_equity(prices)
        
        # Cash + unrealized PnL
        expected = 10000.0 + 0.1 * (52000.0 - 50000.0)
        assert abs(equity - expected) < 0.01
    
    def test_snapshot(self):
        portfolio = Portfolio(initial_capital=10000.0)
        
        snapshot = portfolio.take_snapshot(datetime.now())
        
        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.equity == 10000.0
        assert snapshot.cash == 10000.0


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot."""
    
    def test_creation(self):
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            equity=10000.0,
            cash=8000.0,
            positions_value=2000.0,
            unrealized_pnl=200.0,
            realized_pnl=0.0,
            drawdown=0.0
        )
        assert snapshot.equity == 10000.0
        assert snapshot.positions_value == 2000.0
    
    def test_to_dict(self):
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            equity=10000.0,
            cash=8000.0,
            positions_value=2000.0,
            unrealized_pnl=200.0,
            realized_pnl=0.0,
            drawdown=0.0
        )
        
        data = snapshot.to_dict()
        assert 'equity' in data
        assert 'cash' in data
        assert data['equity'] == 10000.0
