"""
Tests for portfolio management.
"""

import pytest
from datetime import datetime
import numpy as np

from quantbox.core.portfolio import (
    Position, PairPosition, Portfolio, PortfolioSnapshot
)
from quantbox.core.events import FillEvent, OrderSide


class TestPosition:
    """Tests for Position class."""
    
    def test_creation(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=50000.0
        )
        
        assert pos.symbol == "BTCUSDT"
        assert pos.quantity == 1.0
        assert pos.avg_entry_price == 50000.0
    
    def test_is_long(self):
        pos = Position(symbol="BTCUSDT", quantity=1.0)
        assert pos.is_long is True
        assert pos.is_short is False
    
    def test_is_short(self):
        pos = Position(symbol="BTCUSDT", quantity=-1.0)
        assert pos.is_long is False
        assert pos.is_short is True
    
    def test_is_flat(self):
        pos = Position(symbol="BTCUSDT", quantity=0.0)
        assert pos.is_flat is True
    
    def test_market_value(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=2.0,
            current_price=50000.0
        )
        assert pos.market_value == 100000.0
    
    def test_unrealized_pnl_long(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=55000.0
        )
        # Long: (current - entry) * qty = (55000 - 50000) * 1 = 5000
        assert pos.unrealized_pnl == 5000.0
    
    def test_unrealized_pnl_short(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=-1.0,
            avg_entry_price=50000.0,
            current_price=45000.0
        )
        # Short: (current - entry) * qty = (45000 - 50000) * -1 = 5000
        assert pos.unrealized_pnl == 5000.0
    
    def test_return_pct(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=55000.0
        )
        # Return: unrealized_pnl / cost_basis = 5000 / 50000 = 0.1
        assert abs(pos.return_pct - 0.1) < 1e-6
    
    def test_update_price(self):
        pos = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=50000.0
        )
        pos.update_price(55000.0)
        assert pos.current_price == 55000.0


class TestPairPosition:
    """Tests for PairPosition class."""
    
    def test_creation(self):
        pos1 = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=50000.0
        )
        pos2 = Position(
            symbol="ETHUSDT",
            quantity=-10.0,
            avg_entry_price=2000.0,
            current_price=2000.0
        )
        
        pair_pos = PairPosition(
            position1=pos1,
            position2=pos2,
            hedge_ratio=0.5,
            entry_z_score=2.5
        )
        
        assert pair_pos.symbol1 == "BTCUSDT"
        assert pair_pos.symbol2 == "ETHUSDT"
        assert pair_pos.hedge_ratio == 0.5
    
    def test_is_long_spread(self):
        pos1 = Position(symbol="BTCUSDT", quantity=1.0)  # Long
        pos2 = Position(symbol="ETHUSDT", quantity=-1.0)  # Short
        
        pair_pos = PairPosition(
            position1=pos1,
            position2=pos2,
            hedge_ratio=0.5
        )
        
        assert pair_pos.is_long_spread is True
    
    def test_total_pnl(self):
        pos1 = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=52000.0
        )
        pos2 = Position(
            symbol="ETHUSDT",
            quantity=-10.0,
            avg_entry_price=2000.0,
            current_price=1950.0
        )
        
        pair_pos = PairPosition(
            position1=pos1,
            position2=pos2,
            hedge_ratio=0.5
        )
        
        # pos1 pnl: (52000 - 50000) * 1 = 2000
        # pos2 pnl: (1950 - 2000) * -10 = 500
        # total: 2500
        assert pair_pos.unrealized_pnl == 2500.0


class TestPortfolio:
    """Tests for Portfolio class."""
    
    def test_initialization(self):
        portfolio = Portfolio(initial_capital=10000.0)
        
        assert portfolio.initial_capital == 10000.0
        assert portfolio.cash == 10000.0
        assert portfolio.total_equity == 10000.0
    
    def test_process_fill_buy(self):
        portfolio = Portfolio(initial_capital=100000.0)
        
        fill = FillEvent(
            order_id="order1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=25.0
        )
        
        portfolio.process_fill(fill)
        
        assert "BTCUSDT" in portfolio.positions
        assert portfolio.positions["BTCUSDT"].quantity == 1.0
        assert portfolio.cash < 100000.0  # Cash reduced
    
    def test_process_fill_sell(self):
        portfolio = Portfolio(initial_capital=100000.0)
        
        # First buy
        buy_fill = FillEvent(
            order_id="order1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            fill_quantity=2.0,
            fill_price=50000.0,
            commission=50.0
        )
        portfolio.process_fill(buy_fill)
        
        # Then sell half
        sell_fill = FillEvent(
            order_id="order2",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=55000.0,
            commission=27.5
        )
        portfolio.process_fill(sell_fill)
        
        assert portfolio.positions["BTCUSDT"].quantity == 1.0
    
    def test_total_unrealized_pnl(self):
        portfolio = Portfolio(initial_capital=100000.0)
        
        # Add position
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=55000.0
        )
        
        assert portfolio.total_unrealized_pnl == 5000.0
    
    def test_current_drawdown(self):
        portfolio = Portfolio(initial_capital=10000.0)
        portfolio.peak_equity = 12000.0
        portfolio.cash = 9000.0
        
        # Drawdown: (12000 - 9000) / 12000 = 0.25
        assert abs(portfolio.current_drawdown - 0.25) < 1e-6
    
    def test_update_prices(self):
        portfolio = Portfolio(initial_capital=100000.0)
        
        portfolio.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            quantity=1.0,
            avg_entry_price=50000.0,
            current_price=50000.0
        )
        
        portfolio.update_prices({"BTCUSDT": 55000.0})
        
        assert portfolio.positions["BTCUSDT"].current_price == 55000.0


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot."""
    
    def test_creation(self):
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_equity=10000.0,
            cash=5000.0,
            positions_value=5000.0,
            unrealized_pnl=500.0,
            realized_pnl=100.0,
            num_positions=2,
            drawdown=0.05
        )
        
        assert snapshot.total_equity == 10000.0
        assert snapshot.cash == 5000.0
        assert snapshot.positions_value == 5000.0
        assert snapshot.num_positions == 2
