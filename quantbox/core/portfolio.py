"""
Portfolio management for tracking positions, PnL, and performance.

Unified portfolio class that works identically in backtest and live modes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from quantbox.core.events import (
    FillEvent, PositionEvent, OrderSide, EventType
)


@dataclass
class Position:
    """Represents a single position in an asset."""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    # Pair trading specific
    pair_symbol: Optional[str] = None
    hedge_ratio: float = 1.0
    
    # Timestamps
    entry_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-10
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.avg_entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.is_flat:
            return 0.0
        return (self.current_price - self.avg_entry_price) * self.quantity
    
    @property
    def total_pnl(self) -> float:
        """Total PnL including realized and unrealized."""
        return self.realized_pnl + self.unrealized_pnl - self.commission_paid
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if abs(self.cost_basis) < 1e-10:
            return 0.0
        return self.unrealized_pnl / abs(self.cost_basis)
    
    def update_price(self, price: float):
        """Update current market price."""
        self.current_price = price
        self.last_update = datetime.utcnow()
    
    def update_from_fill(self, fill: FillEvent):
        """Update position from a fill event."""
        fill_qty = fill.fill_quantity if fill.side == OrderSide.BUY else -fill.fill_quantity
        
        if self.is_flat:
            # New position
            self.avg_entry_price = fill.fill_price
            self.quantity = fill_qty
            self.entry_time = fill.execution_time
        else:
            # Existing position
            if (self.is_long and fill_qty > 0) or (self.is_short and fill_qty < 0):
                # Adding to position - update average price
                total_value = (self.quantity * self.avg_entry_price + 
                              fill_qty * fill.fill_price)
                self.quantity += fill_qty
                self.avg_entry_price = total_value / self.quantity
            else:
                # Reducing or closing position
                close_qty = min(abs(fill_qty), abs(self.quantity))
                self.realized_pnl += (fill.fill_price - self.avg_entry_price) * close_qty * np.sign(self.quantity)
                self.quantity += fill_qty
                
                if self.is_flat:
                    self.avg_entry_price = 0.0
        
        self.current_price = fill.fill_price
        self.commission_paid += fill.commission
        self.last_update = fill.execution_time


@dataclass
class PairPosition:
    """Represents a paired position for pair trading."""
    position1: Position
    position2: Position
    hedge_ratio: float
    entry_z_score: float = 0.0
    current_z_score: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def symbol1(self) -> str:
        return self.position1.symbol
    
    @property
    def symbol2(self) -> str:
        return self.position2.symbol
    
    @property
    def is_long_spread(self) -> bool:
        """Long spread: long position1, short position2."""
        return self.position1.is_long and self.position2.is_short
    
    @property
    def is_flat(self) -> bool:
        return self.position1.is_flat and self.position2.is_flat
    
    @property
    def total_market_value(self) -> float:
        return abs(self.position1.market_value) + abs(self.position2.market_value)
    
    @property
    def unrealized_pnl(self) -> float:
        return self.position1.unrealized_pnl + self.position2.unrealized_pnl
    
    @property
    def realized_pnl(self) -> float:
        return self.position1.realized_pnl + self.position2.realized_pnl
    
    @property
    def total_pnl(self) -> float:
        return self.position1.total_pnl + self.position2.total_pnl


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""
    timestamp: datetime
    total_equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int
    drawdown: float = 0.0


class Portfolio:
    """
    Portfolio manager for tracking positions and performance.
    
    Works identically for both backtest and live trading modes.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.0005,
        slippage: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.pair_positions: Dict[Tuple[str, str], PairPosition] = {}
        
        # Performance tracking
        self.equity_curve: List[PortfolioSnapshot] = []
        self.trades: List[Dict] = []
        self.peak_equity = initial_capital
        
        # Timestamp
        self.current_time = datetime.utcnow()
    
    @property
    def total_equity(self) -> float:
        """Total portfolio equity (cash + positions value)."""
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized PnL across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized PnL."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.total_equity) / self.peak_equity
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown experienced."""
        if not self.equity_curve:
            return 0.0
        return max(snap.drawdown for snap in self.equity_curve)
    
    def update_time(self, timestamp: datetime):
        """Update current time."""
        self.current_time = timestamp
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
        
        # Update peak equity
        current_equity = self.total_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
    
    def process_fill(self, fill: FillEvent):
        """Process a fill event and update portfolio."""
        symbol = fill.symbol
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        old_quantity = position.quantity
        
        # Update position
        position.update_from_fill(fill)
        
        # Update cash
        fill_value = fill.fill_quantity * fill.fill_price
        if fill.side == OrderSide.BUY:
            self.cash -= fill_value + fill.commission
        else:
            self.cash += fill_value - fill.commission
        
        # Record trade if position changed direction or closed
        if (old_quantity * position.quantity < 0) or position.is_flat:
            self._record_trade(fill, old_quantity, position)
        
        # Clean up flat positions
        if position.is_flat:
            del self.positions[symbol]
    
    def _record_trade(self, fill: FillEvent, old_quantity: float, position: Position):
        """Record a completed trade."""
        trade_record = {
            'timestamp': fill.execution_time,
            'symbol': fill.symbol,
            'side': 'SELL' if old_quantity > 0 else 'BUY',
            'quantity': abs(old_quantity),
            'entry_price': position.avg_entry_price if not position.is_flat else 0,
            'exit_price': fill.fill_price,
            'realized_pnl': position.realized_pnl,
            'commission': fill.commission,
        }
        self.trades.append(trade_record)
    
    def record_snapshot(self):
        """Record current portfolio state."""
        snapshot = PortfolioSnapshot(
            timestamp=self.current_time,
            total_equity=self.total_equity,
            cash=self.cash,
            positions_value=self.total_equity - self.cash,
            unrealized_pnl=self.total_unrealized_pnl,
            realized_pnl=self.total_realized_pnl,
            num_positions=len(self.positions),
            drawdown=self.current_drawdown
        )
        self.equity_curve.append(snapshot)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_position_size(self, symbol: str) -> float:
        """Get position size for a symbol."""
        pos = self.get_position(symbol)
        return pos.quantity if pos else 0.0
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self.positions and not self.positions[symbol].is_flat
    
    def calculate_order_quantity(
        self,
        symbol: str,
        target_value: float,
        current_price: float,
        side: OrderSide
    ) -> float:
        """
        Calculate order quantity considering fees and slippage.
        """
        # Account for fees and slippage
        effective_price = current_price * (1 + self.slippage) if side == OrderSide.BUY else current_price * (1 - self.slippage)
        fee = target_value * self.trading_fee
        
        available_value = target_value - fee
        quantity = available_value / effective_price
        
        return quantity
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {}
        
        # Convert to numpy for calculations
        equity = np.array([snap.total_equity for snap in self.equity_curve])
        returns = np.diff(equity) / equity[:-1]
        
        # Basic metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        sharpe_ratio = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            # Annualize assuming daily snapshots
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / peak
        max_dd = np.max(drawdowns)
        
        # Win rate
        winning_trades = [t for t in self.trades if t['realized_pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Profit factor
        gross_profit = sum(t['realized_pnl'] for t in self.trades if t['realized_pnl'] > 0)
        gross_loss = abs(sum(t['realized_pnl'] for t in self.trades if t['realized_pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_equity': equity[-1],
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(self.trades) - len(winning_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_trade_pnl': np.mean([t['realized_pnl'] for t in self.trades]) if self.trades else 0,
            'total_commission': sum(t['commission'] for t in self.trades),
        }
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        data = [
            {
                'timestamp': snap.timestamp,
                'equity': snap.total_equity,
                'cash': snap.cash,
                'positions_value': snap.positions_value,
                'unrealized_pnl': snap.unrealized_pnl,
                'realized_pnl': snap.realized_pnl,
                'drawdown': snap.drawdown,
            }
            for snap in self.equity_curve
        ]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        df.set_index('timestamp', inplace=True)
        return df
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.pair_positions.clear()
        self.equity_curve.clear()
        self.trades.clear()
        self.peak_equity = self.initial_capital
        self.current_time = datetime.utcnow()
