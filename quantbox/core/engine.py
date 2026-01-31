"""
Main trading engine that orchestrates all components.

Provides unified interface for both backtesting and live trading.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from quantbox.core.config import Config, TradingMode
from quantbox.core.events import (
    EventQueue, EventType, Event, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, OrderSide, SignalType, create_pair_orders
)
from quantbox.core.portfolio import Portfolio
from quantbox.core.execution import (
    ExecutionHandler, BacktestExecutionHandler, LiveExecutionHandler,
    create_execution_handler
)


logger = logging.getLogger('TradingEngine')


class TradingEngine:
    """
    Main trading engine that coordinates all components.
    
    Provides unified interface for backtest and live trading,
    ensuring identical logic execution in both modes.
    """
    
    def __init__(
        self,
        config: Config,
        client=None
    ):
        self.config = config
        self.client = client
        
        # Initialize components
        self.portfolio = Portfolio(
            initial_capital=config.backtest.initial_capital,
            trading_fee=config.risk.trading_fee,
            slippage=config.risk.slippage
        )
        
        self.execution = create_execution_handler(config, client)
        self.event_queue = EventQueue()
        
        # Strategy registry
        self.strategies: Dict[str, Any] = {}
        self.active_strategy: Optional[str] = None
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.current_time: datetime = datetime.utcnow()
        
        # Callbacks for custom event handling
        self._on_signal_callbacks: List[Callable] = []
        self._on_fill_callbacks: List[Callable] = []
        self._on_bar_callbacks: List[Callable] = []
        
        # State
        self.is_running = False
        
        # Register event handlers
        self._setup_event_handlers()
        
        logger.info(f"TradingEngine initialized in {config.mode.value} mode")
    
    def _setup_event_handlers(self):
        """Setup event handlers for the event queue."""
        self.event_queue.register_handler(EventType.SIGNAL, self._handle_signal)
        self.event_queue.register_handler(EventType.ORDER, self._handle_order)
        self.event_queue.register_handler(EventType.FILL, self._handle_fill)
        self.event_queue.register_handler(EventType.MARKET, self._handle_market)
    
    def register_strategy(self, name: str, strategy):
        """Register a trading strategy."""
        self.strategies[name] = strategy
        if not self.active_strategy:
            self.active_strategy = name
        logger.info(f"Registered strategy: {name}")
    
    def set_active_strategy(self, name: str):
        """Set the active trading strategy."""
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not registered")
        self.active_strategy = name
    
    def on_signal(self, callback: Callable):
        """Register callback for signal events."""
        self._on_signal_callbacks.append(callback)
    
    def on_fill(self, callback: Callable):
        """Register callback for fill events."""
        self._on_fill_callbacks.append(callback)
    
    def on_bar(self, callback: Callable):
        """Register callback for new bar events."""
        self._on_bar_callbacks.append(callback)
    
    def _handle_market(self, event: MarketEvent):
        """Handle new market data."""
        self.current_prices[event.symbol] = event.close
        self.portfolio.update_prices(self.current_prices)
    
    def _handle_signal(self, event: SignalEvent):
        """Handle trading signal."""
        logger.info(
            f"Signal: {event.signal_type.value} {event.symbol1}/{event.symbol2} "
            f"z-score: {event.z_score:.3f}"
        )
        
        # Execute callbacks
        for callback in self._on_signal_callbacks:
            callback(event)
        
        # Check if we should act on signal
        if not self._validate_signal(event):
            return
        
        # Generate orders from signal
        orders = self._generate_orders_from_signal(event)
        
        for order in orders:
            self.event_queue.put(order)
    
    def _validate_signal(self, signal: SignalEvent) -> bool:
        """Validate if signal should be acted upon."""
        # Check position limits
        if self.portfolio.has_position(signal.symbol1) or \
           self.portfolio.has_position(signal.symbol2):
            if signal.signal_type not in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.CLOSE_ALL]:
                logger.debug(f"Already have position for pair, skipping signal")
                return False
        
        # Check drawdown limit
        if self.portfolio.current_drawdown > self.config.risk.max_drawdown:
            logger.warning(f"Max drawdown exceeded, skipping signal")
            return False
        
        return True
    
    def _generate_orders_from_signal(self, signal: SignalEvent) -> List[OrderEvent]:
        """Generate orders from a trading signal."""
        # Get prices
        price1 = self.current_prices.get(signal.symbol1, 0)
        price2 = self.current_prices.get(signal.symbol2, 0)
        
        if price1 <= 0 or price2 <= 0:
            logger.error(f"Invalid prices for order generation")
            return []
        
        # Calculate position size
        available_capital = self.portfolio.cash * self.config.risk.max_position_size
        
        # Create pair orders
        orders = create_pair_orders(
            signal=signal,
            capital=available_capital,
            price1=price1,
            price2=price2,
            fee_rate=self.config.risk.trading_fee
        )
        
        return orders
    
    def _handle_order(self, event: OrderEvent):
        """Handle order event."""
        logger.debug(f"Order: {event.side.value} {event.quantity:.6f} {event.symbol}")
        
        # Execute order
        fill = self.execution.execute_order(event)
        
        if fill:
            self.event_queue.put(fill)
    
    def _handle_fill(self, event: FillEvent):
        """Handle fill event."""
        logger.info(
            f"Fill: {event.side.value} {event.fill_quantity:.6f} {event.symbol} "
            f"@ {event.fill_price:.4f}"
        )
        
        # Update portfolio
        self.portfolio.process_fill(event)
        
        # Execute callbacks
        for callback in self._on_fill_callbacks:
            callback(event)
    
    def update_market_data(
        self,
        timestamp: datetime,
        data: Dict[str, Dict[str, float]]
    ):
        """
        Update market data.
        
        Args:
            timestamp: Current timestamp
            data: Dict of symbol -> {open, high, low, close, volume}
        """
        self.current_time = timestamp
        self.portfolio.update_time(timestamp)
        
        # Update prices
        for symbol, ohlcv in data.items():
            self.current_prices[symbol] = ohlcv['close']
            
            # Create market event
            market_event = MarketEvent(
                timestamp=timestamp,
                symbol=symbol,
                open=ohlcv.get('open', 0),
                high=ohlcv.get('high', 0),
                low=ohlcv.get('low', 0),
                close=ohlcv['close'],
                volume=ohlcv.get('volume', 0)
            )
            self.event_queue.put(market_event)
        
        # Update execution handler for backtest
        if isinstance(self.execution, BacktestExecutionHandler):
            self.execution.update_market_data(
                timestamp, 
                self.current_prices,
                {s: d for s, d in data.items()}
            )
        
        # Execute bar callbacks
        for callback in self._on_bar_callbacks:
            callback(timestamp, data)
    
    def process_events(self):
        """Process all events in the queue."""
        self.event_queue.process_all()
    
    def generate_signal(
        self,
        symbol1: str,
        symbol2: str,
        signal_type: SignalType,
        z_score: float,
        hedge_ratio: float,
        **kwargs
    ):
        """Generate a trading signal."""
        signal = SignalEvent(
            signal_type=signal_type,
            symbol1=symbol1,
            symbol2=symbol2,
            z_score=z_score,
            hedge_ratio=hedge_ratio,
            strategy_id=self.active_strategy or "",
            metadata=kwargs
        )
        self.event_queue.put(signal)
    
    def close_pair_position(self, symbol1: str, symbol2: str):
        """Close a pair position."""
        pos1 = self.portfolio.get_position(symbol1)
        pos2 = self.portfolio.get_position(symbol2)
        
        if pos1 and not pos1.is_flat:
            side = OrderSide.SELL if pos1.quantity > 0 else OrderSide.BUY
            order = OrderEvent(
                symbol=symbol1,
                side=side,
                quantity=abs(pos1.quantity)
            )
            self.event_queue.put(order)
        
        if pos2 and not pos2.is_flat:
            side = OrderSide.SELL if pos2.quantity > 0 else OrderSide.BUY
            order = OrderEvent(
                symbol=symbol2,
                side=side,
                quantity=abs(pos2.quantity)
            )
            self.event_queue.put(order)
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        progress: bool = True
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            strategy_func: Function that generates signals
            progress: Show progress bar
            
        Returns:
            Dict with performance metrics
        """
        if self.config.mode != TradingMode.BACKTEST:
            raise ValueError("Engine must be in BACKTEST mode")
        
        self.is_running = True
        self.portfolio.reset()
        
        # Get common timestamps
        first_df = next(iter(data.values()))
        timestamps = first_df.index.tolist()
        
        logger.info(f"Starting backtest with {len(timestamps)} bars")
        
        iterator = tqdm(timestamps, desc="Backtesting") if progress else timestamps
        
        for timestamp in iterator:
            if not self.is_running:
                break
            
            # Gather data for all symbols at this timestamp
            bar_data = {}
            for symbol, df in data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    bar_data[symbol] = {
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'volume': row.get('Volume', 0)
                    }
            
            # Update market data
            self.update_market_data(timestamp, bar_data)
            
            # Run strategy logic
            strategy_func(self, timestamp, bar_data)
            
            # Process all events
            self.process_events()
            
            # Record portfolio snapshot
            self.portfolio.record_snapshot()
        
        self.is_running = False
        
        # Return results
        return {
            'metrics': self.portfolio.get_performance_metrics(),
            'equity_curve': self.portfolio.get_equity_dataframe(),
            'trades': self.portfolio.get_trades_dataframe()
        }
    
    def run_live(self, strategy_func: Callable, interval_seconds: int = 60):
        """
        Run live trading loop.
        
        Args:
            strategy_func: Function that generates signals
            interval_seconds: Seconds between strategy updates
        """
        import time
        
        if self.config.mode == TradingMode.BACKTEST:
            raise ValueError("Cannot run live trading in BACKTEST mode")
        
        if not self.client:
            raise ValueError("Exchange client required for live trading")
        
        self.is_running = True
        logger.info("Starting live trading loop")
        
        try:
            while self.is_running:
                try:
                    # Get current market data
                    bar_data = self._fetch_current_data()
                    
                    # Update market data
                    self.update_market_data(datetime.utcnow(), bar_data)
                    
                    # Run strategy
                    strategy_func(self, datetime.utcnow(), bar_data)
                    
                    # Process events
                    self.process_events()
                    
                    # Wait for next interval
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in live trading loop: {e}")
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            self.stop()
    
    def _fetch_current_data(self) -> Dict[str, Dict]:
        """Fetch current market data from exchange."""
        data = {}
        
        # Get prices for tracked symbols
        for symbol in self.current_prices.keys():
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                data[symbol] = {
                    'open': float(ticker['price']),
                    'high': float(ticker['price']),
                    'low': float(ticker['price']),
                    'close': float(ticker['price']),
                    'volume': 0
                }
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
        
        return data
    
    def stop(self):
        """Stop the trading engine."""
        self.is_running = False
        logger.info("Trading engine stopped")
    
    def get_results(self) -> Dict:
        """Get current results/metrics."""
        return {
            'metrics': self.portfolio.get_performance_metrics(),
            'equity_curve': self.portfolio.get_equity_dataframe(),
            'trades': self.portfolio.get_trades_dataframe(),
            'current_positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.portfolio.positions.items()
            }
        }
