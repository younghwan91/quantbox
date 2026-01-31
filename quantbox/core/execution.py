"""
Execution handlers for backtest and live trading.

Provides unified interface for order execution that works identically
in both backtest and live trading modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging
import time
import math

from quantbox.core.events import (
    OrderEvent, FillEvent, OrderSide, OrderType, EventQueue
)
from quantbox.core.config import Config


logger = logging.getLogger('ExecutionHandler')


class ExecutionHandler(ABC):
    """
    Abstract base class for execution handlers.
    
    Subclasses implement either backtest or live execution.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.trading_fee = config.risk.trading_fee
        self.slippage = config.risk.slippage
    
    @abstractmethod
    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """Execute an order and return fill event."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass
    
    def calculate_slippage(self, price: float, side: OrderSide) -> float:
        """Calculate slippage-adjusted price."""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission."""
        return quantity * price * self.trading_fee


class BacktestExecutionHandler(ExecutionHandler):
    """
    Execution handler for backtesting.
    
    Simulates order execution using historical data with
    configurable slippage and fee models.
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.current_prices: Dict[str, float] = {}
        self.current_time: datetime = datetime.now(timezone.utc)
        self.order_history: List[OrderEvent] = []
        self.fill_history: List[FillEvent] = []
        
        # Simulation settings
        self.use_ohlc_slippage = True  # Use high/low for slippage
        self.current_ohlc: Dict[str, Dict] = {}  # symbol -> {open, high, low, close}
    
    def update_market_data(
        self, 
        timestamp: datetime,
        prices: Dict[str, float],
        ohlc: Optional[Dict[str, Dict]] = None
    ):
        """Update current market data for simulation."""
        self.current_time = timestamp
        self.current_prices = prices
        if ohlc:
            self.current_ohlc = ohlc
    
    def get_current_price(self, symbol: str) -> float:
        """Get current simulated price."""
        return self.current_prices.get(symbol, 0.0)
    
    def get_execution_price(self, symbol: str, side: OrderSide) -> float:
        """
        Get execution price considering slippage.
        
        Uses OHLC data if available for more realistic simulation.
        """
        base_price = self.get_current_price(symbol)
        
        if self.use_ohlc_slippage and symbol in self.current_ohlc:
            ohlc = self.current_ohlc[symbol]
            if side == OrderSide.BUY:
                # Buy at higher price (ask simulation)
                return min(ohlc['high'], base_price * (1 + self.slippage))
            else:
                # Sell at lower price (bid simulation)
                return max(ohlc['low'], base_price * (1 - self.slippage))
        else:
            return self.calculate_slippage(base_price, side)
    
    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """
        Execute order in backtest simulation.
        
        Simulates immediate fill for market orders with slippage.
        """
        self.order_history.append(order)
        
        if order.order_type != OrderType.MARKET:
            logger.warning(f"Only MARKET orders supported in backtest, got {order.order_type}")
            return None
        
        # Get execution price with slippage
        fill_price = self.get_execution_price(order.symbol, order.side)
        
        if fill_price <= 0:
            logger.error(f"Invalid price for {order.symbol}: {fill_price}")
            return None
        
        # Calculate commission
        commission = self.calculate_commission(order.quantity, fill_price)
        
        # Calculate actual slippage amount
        base_price = self.get_current_price(order.symbol)
        slippage_amount = abs(fill_price - base_price) * order.quantity
        
        # Create fill event
        fill = FillEvent(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount,
            execution_time=self.current_time
        )
        
        self.fill_history.append(fill)
        logger.debug(
            f"FILL: {order.symbol} {order.side.value} {order.quantity:.6f} @ {fill_price:.4f} "
            f"(commission: {commission:.4f}, slippage: {slippage_amount:.4f})"
        )
        
        return fill
    
    def get_fill_rate(self) -> float:
        """Get percentage of orders that were filled."""
        if not self.order_history:
            return 0.0
        return len(self.fill_history) / len(self.order_history)


class LiveExecutionHandler(ExecutionHandler):
    """
    Execution handler for live trading.
    
    Interfaces with exchange API for real order execution.
    """
    
    def __init__(self, config: Config, client=None):
        super().__init__(config)
        self.client = client
        self.leverage = config.risk.leverage
        self.max_retries = config.exchange.max_retries
        self.retry_delay = config.exchange.retry_delay
        
        # Symbol info cache
        self._symbol_info: Dict[str, Dict] = {}
    
    def set_client(self, client):
        """Set the exchange client."""
        self.client = client
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading rules from exchange."""
        if symbol in self._symbol_info:
            return self._symbol_info[symbol]
        
        try:
            info = self.client.futures_exchange_info()
            for symbol_info in info['symbols']:
                if symbol_info['symbol'] == symbol:
                    filters = symbol_info['filters']
                    result = {
                        'quantity_precision': symbol_info['quantityPrecision'],
                        'price_precision': symbol_info['pricePrecision'],
                    }
                    for f in filters:
                        if f['filterType'] == 'LOT_SIZE':
                            result['step_size'] = float(f['stepSize'])
                            result['min_qty'] = float(f['minQty'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            result['tick_size'] = float(f['tickSize'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            result['min_notional'] = float(f['notional'])
                    
                    self._symbol_info[symbol] = result
                    return result
            
            logger.error(f"Symbol info not found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None
    
    def round_quantity(self, quantity: float, symbol: str) -> float:
        """Round quantity to exchange-compatible precision."""
        info = self.get_symbol_info(symbol)
        if not info:
            return quantity
        
        step_size = info.get('step_size', 0.001)
        precision = info.get('quantity_precision', 3)
        
        rounded = math.floor(quantity / step_size) * step_size
        return round(rounded, precision)
    
    def round_price(self, price: float, symbol: str) -> float:
        """Round price to exchange-compatible precision."""
        info = self.get_symbol_info(symbol)
        if not info:
            return price
        
        tick_size = info.get('tick_size', 0.01)
        precision = info.get('price_precision', 2)
        
        rounded = round(price / tick_size) * tick_size
        return round(rounded, precision)
    
    def validate_order(self, order: OrderEvent) -> Tuple[bool, str]:
        """Validate order against exchange rules."""
        info = self.get_symbol_info(order.symbol)
        if not info:
            return False, "Could not fetch symbol info"
        
        # Check minimum quantity
        if order.quantity < info.get('min_qty', 0):
            return False, f"Quantity below minimum: {order.quantity} < {info['min_qty']}"
        
        # Check minimum notional
        price = self.get_current_price(order.symbol)
        notional = order.quantity * price
        if notional < info.get('min_notional', 0):
            return False, f"Notional below minimum: {notional} < {info['min_notional']}"
        
        return True, "Valid"
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from exchange."""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """Execute order on exchange."""
        if not self.client:
            logger.error("No exchange client configured")
            return None
        
        # Validate order
        is_valid, message = self.validate_order(order)
        if not is_valid:
            logger.error(f"Order validation failed: {message}")
            return None
        
        # Round quantity
        rounded_qty = self.round_quantity(order.quantity, order.symbol)
        
        # Set leverage
        try:
            self.client.futures_change_leverage(
                symbol=order.symbol, 
                leverage=self.leverage
            )
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")
        
        # Execute with retry
        for attempt in range(self.max_retries):
            try:
                result = self.client.futures_create_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    type=order.order_type.value,
                    quantity=rounded_qty
                )
                
                # Create fill event from result
                fill = FillEvent(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    fill_quantity=float(result['executedQty']),
                    fill_price=float(result.get('avgPrice', result.get('price', 0))),
                    commission=0.0,  # Will be calculated from trades
                    exchange_order_id=str(result['orderId']),
                    execution_time=datetime.now(timezone.utc)
                )
                
                logger.info(
                    f"Order executed: {order.symbol} {order.side.value} "
                    f"{fill.fill_quantity} @ {fill.fill_price}"
                )
                
                return fill
                
            except Exception as e:
                logger.warning(
                    f"Order execution attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        
        logger.error(f"Failed to execute order after {self.max_retries} attempts")
        return None
    
    def close_position(self, symbol: str, quantity: float) -> Optional[FillEvent]:
        """Close an existing position."""
        side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
        
        order = OrderEvent(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type=OrderType.MARKET
        )
        
        return self.execute_order(order)
    
    def close_all_positions(self) -> List[FillEvent]:
        """Close all open positions."""
        fills = []
        
        try:
            positions = self.client.futures_account()['positions']
            for pos in positions:
                qty = float(pos['positionAmt'])
                if abs(qty) > 0:
                    fill = self.close_position(pos['symbol'], qty)
                    if fill:
                        fills.append(fill)
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
        
        return fills
    
    def get_account_balance(self, asset: str = 'USDT') -> float:
        """Get available account balance."""
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == asset:
                    return float(balance['availableBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return 0.0
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions."""
        try:
            positions = self.client.futures_account()['positions']
            return [
                {
                    'symbol': pos['symbol'],
                    'quantity': float(pos['positionAmt']),
                    'entry_price': float(pos['entryPrice']),
                    'unrealized_pnl': float(pos['unrealizedProfit']),
                    'leverage': int(pos['leverage'])
                }
                for pos in positions
                if abs(float(pos['positionAmt'])) > 0
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []


def create_execution_handler(config: Config, client=None) -> ExecutionHandler:
    """
    Factory function to create appropriate execution handler.
    """
    from quantbox.core.config import TradingMode
    
    if config.mode == TradingMode.BACKTEST:
        return BacktestExecutionHandler(config)
    else:
        handler = LiveExecutionHandler(config, client)
        return handler
