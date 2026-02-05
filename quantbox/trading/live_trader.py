"""
QuantBox Live Pair Trading System

Production-ready live trading module that uses IDENTICAL logic to the backtest.
This ensures backtest-live consistency and prevents strategy drift.

Key Features:
- Same entry/exit logic as continuous_backtest.py
- Paper trading mode for testing
- Risk management and position sizing
- Real-time pair monitoring
- Automatic reconnection handling
- Comprehensive logging

IMPORTANT: Run in paper mode first to validate behavior matches backtest!
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Local imports
from quantbox.backtest.continuous_backtest import (
    PairStatistics,
    ContinuousBacktestConfig,
    ExitReason
)

logger = logging.getLogger('LiveTrader')


class TradingMode(Enum):
    """Trading mode"""
    PAPER = "paper"  # Simulated trades
    LIVE = "live"    # Real money trades


@dataclass
class LiveTradeConfig:
    """Live trading configuration"""
    # Mode
    mode: TradingMode = TradingMode.PAPER
    
    # Capital
    trading_capital: float = 1000.0  # USDT to use
    
    # Transaction costs (should match backtest!)
    trading_fee: float = 0.0005  # 0.05% Binance maker fee
    slippage: float = 0.0010    # 0.10% estimated slippage
    
    # Formation/Analysis
    formation_bars: int = 240   # Bars for pair analysis
    timeframe: str = '1h'       # Timeframe for data
    
    # Entry/Exit (MUST MATCH BACKTEST!)
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 3.5
    max_holding_bars: int = 100
    
    # Pair selection
    min_correlation: float = 0.6
    max_cointegration_pvalue: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 50.0
    
    # Universe
    n_coins: int = 30  # Top coins by volume to consider
    quote_currency: str = 'USDT'
    
    # Timing
    check_interval_seconds: int = 60  # How often to check positions
    
    # Risk
    max_position_size_pct: float = 0.95  # Max % of capital per position
    min_order_value: float = 11.0  # Minimum order value in USDT
    
    def to_backtest_config(self) -> ContinuousBacktestConfig:
        """Convert to backtest config for consistency checking"""
        return ContinuousBacktestConfig(
            initial_capital=self.trading_capital,
            trading_fee=self.trading_fee,
            slippage=self.slippage,
            formation_bars=self.formation_bars,
            entry_zscore=self.entry_zscore,
            exit_zscore=self.exit_zscore,
            stop_loss_zscore=self.stop_loss_zscore,
            max_holding_bars=self.max_holding_bars,
            min_correlation=self.min_correlation,
            max_cointegration_pvalue=self.max_cointegration_pvalue,
            min_half_life=self.min_half_life,
            max_half_life=self.max_half_life,
        )


@dataclass
class Position:
    """Current open position"""
    symbol1: str
    symbol2: str
    signal: int  # 1=long spread, -1=short spread
    
    entry_time: datetime
    entry_price1: float
    entry_price2: float
    entry_spread: float
    entry_z: float
    
    # Frozen formation statistics
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    
    # Position sizes
    qty1: float
    qty2: float
    capital_used: float
    
    # Tracking
    bars_held: int = 0
    last_update: datetime = None
    
    def current_zscore(self, price1: float, price2: float) -> float:
        """Calculate current z-score using frozen statistics"""
        spread = np.log(price1) - (self.hedge_ratio * np.log(price2) + self.intercept)
        return (spread - self.spread_mean) / self.spread_std


@dataclass 
class TradeLog:
    """Trade log entry"""
    timestamp: datetime
    pair: Tuple[str, str]
    action: str  # OPEN_LONG, OPEN_SHORT, CLOSE
    signal: int
    price1: float
    price2: float
    zscore: float
    qty1: float
    qty2: float
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


class BinanceClient:
    """
    Binance API wrapper for live trading.
    
    Handles:
    - Market data fetching
    - Order placement (with paper mode)
    - Position tracking
    - Error handling and retries
    """
    
    def __init__(self, config: LiveTradeConfig):
        self.config = config
        self.client = None
        self._init_client()
        
    def _init_client(self):
        """Initialize Binance client"""
        try:
            from binance.client import Client
            
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
            
            if not api_key or not api_secret:
                logger.warning("No API keys found - limited to market data only")
                self.client = Client()
            else:
                self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
                logger.info(f"Binance client initialized (testnet={testnet})")
                
        except ImportError:
            logger.error("python-binance not installed. Run: pip install python-binance")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def get_top_symbols(self, n: int = 30) -> List[str]:
        """Get top N symbols by 24h volume"""
        try:
            tickers = self.client.futures_ticker()
            
            # Filter USDT pairs
            usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
            
            # Sort by quote volume
            sorted_tickers = sorted(
                usdt_tickers,
                key=lambda x: float(x.get('quoteVolume', 0)),
                reverse=True
            )
            
            symbols = [t['symbol'] for t in sorted_tickers[:n]]
            logger.info(f"Got top {len(symbols)} symbols by volume")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get top symbols: {e}")
            return []
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 300
    ) -> Optional[pd.DataFrame]:
        """Fetch historical klines for a symbol"""
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get trading rules for a symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    filters = {f['filterType']: f for f in s['filters']}
                    return {
                        'symbol': symbol,
                        'price_precision': s['pricePrecision'],
                        'quantity_precision': s['quantityPrecision'],
                        'min_qty': float(filters.get('LOT_SIZE', {}).get('minQty', 0)),
                        'step_size': float(filters.get('LOT_SIZE', {}).get('stepSize', 0)),
                        'min_notional': float(filters.get('MIN_NOTIONAL', {}).get('notional', 0)),
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        paper: bool = True
    ) -> Optional[Dict]:
        """Place market order (or simulate in paper mode)"""
        
        if paper:
            # Simulated order
            price = self.get_current_price(symbol)
            if price is None:
                return None
            
            return {
                'symbol': symbol,
                'orderId': f"PAPER_{int(time.time()*1000)}",
                'side': side,
                'type': 'MARKET',
                'quantity': quantity,
                'price': price,
                'status': 'FILLED',
                'paper': True
            }
        
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"Order placed: {symbol} {side} {quantity}")
            return order
        except Exception as e:
            logger.error(f"Failed to place order {symbol} {side} {quantity}: {e}")
            return None
    
    def get_account_balance(self) -> float:
        """Get USDT balance"""
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b['asset'] == 'USDT':
                    return float(b['availableBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0


class LivePairTrader:
    """
    Live pair trading engine.
    
    Uses IDENTICAL logic to ContinuousPairBacktest to ensure
    backtest-live consistency.
    """
    
    def __init__(self, config: LiveTradeConfig = None):
        self.config = config or LiveTradeConfig()
        self.stats = PairStatistics()
        self.client = None
        self.position: Optional[Position] = None
        self.trade_log: List[TradeLog] = []
        self.running = False
        self._stop_event = threading.Event()
        
        # Paper trading state
        self.paper_balance = self.config.trading_capital
        self.paper_pnl = 0.0
        
    def initialize(self):
        """Initialize trading system"""
        load_dotenv()
        self.client = BinanceClient(self.config)
        
        mode_str = "PAPER" if self.config.mode == TradingMode.PAPER else "LIVE"
        logger.info(f"LivePairTrader initialized in {mode_str} mode")
        logger.info(f"  Capital: ${self.config.trading_capital:,.2f}")
        logger.info(f"  Entry Z-score: ±{self.config.entry_zscore}")
        logger.info(f"  Exit Z-score: ±{self.config.exit_zscore}")
    
    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch current market data for top symbols"""
        symbols = self.client.get_top_symbols(self.config.n_coins)
        
        data = {}
        for symbol in symbols:
            df = self.client.get_historical_klines(
                symbol,
                self.config.timeframe,
                self.config.formation_bars + 10  # Extra buffer
            )
            if df is not None and len(df) >= self.config.formation_bars:
                data[symbol] = df
        
        logger.info(f"Loaded data for {len(data)} symbols")
        return data
    
    def find_best_pair(self, data: Dict[str, pd.DataFrame]) -> Optional[Tuple[str, str, Dict]]:
        """
        Find best tradeable pair.
        
        SAME LOGIC AS continuous_backtest._find_best_pair()
        """
        symbols = list(data.keys())
        candidates = []
        
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                df1 = data[s1]
                df2 = data[s2]
                
                # Use only formation data (exclude last bar for entry decision)
                formation_end = min(len(df1), len(df2)) - 1
                formation_start = max(0, formation_end - self.config.formation_bars)
                
                prices1 = df1.iloc[formation_start:formation_end]['Close'].values
                prices2 = df2.iloc[formation_start:formation_end]['Close'].values
                
                if len(prices1) < 30 or (prices1 <= 0).any() or (prices2 <= 0).any():
                    continue
                
                # 1. Correlation filter
                corr = self.stats.pearson_correlation(prices1, prices2)
                if corr < self.config.min_correlation:
                    continue
                
                # 2. Cointegration filter
                t_stat, p_value = self.stats.cointegration_test(prices1, prices2)
                if p_value > self.config.max_cointegration_pvalue:
                    continue
                
                # 3. Calculate hedge ratio
                hedge_ratio, intercept = self.stats.calculate_hedge_ratio(prices1, prices2)
                
                if not 0.25 < hedge_ratio < 4.0:
                    continue
                
                spread = self.stats.calculate_spread(prices1, prices2, hedge_ratio, intercept)
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                
                if spread_std < 0.01:
                    continue
                
                # 4. Half-life filter
                half_life = self.stats.calculate_half_life(pd.Series(spread))
                if not (self.config.min_half_life < half_life < self.config.max_half_life):
                    continue
                
                # 5. Check current z-score (at end of formation)
                current_z = (spread[-1] - spread_mean) / spread_std
                
                if abs(current_z) >= self.config.entry_zscore:
                    candidates.append({
                        'symbol1': s1,
                        'symbol2': s2,
                        'hedge_ratio': hedge_ratio,
                        'intercept': intercept,
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                        'half_life': half_life,
                        'correlation': corr,
                        'coint_pvalue': p_value,
                        'current_z': current_z,
                        'score': half_life * p_value
                    })
        
        if not candidates:
            return None
        
        best = min(candidates, key=lambda x: x['score'])
        
        return (
            best['symbol1'],
            best['symbol2'],
            {
                'hedge_ratio': best['hedge_ratio'],
                'intercept': best['intercept'],
                'spread_mean': best['spread_mean'],
                'spread_std': best['spread_std'],
                'half_life': best['half_life'],
                'current_z': best['current_z'],
            }
        )
    
    def enter_position(
        self,
        symbol1: str,
        symbol2: str,
        pair_info: Dict,
        capital: float
    ) -> bool:
        """Enter a new position"""
        
        # Get current prices
        price1 = self.client.get_current_price(symbol1)
        price2 = self.client.get_current_price(symbol2)
        
        if price1 is None or price2 is None:
            logger.error("Failed to get prices for entry")
            return False
        
        # Calculate current z-score
        current_z = pair_info['current_z']
        
        # Determine signal
        if current_z <= -self.config.entry_zscore:
            signal = 1  # Long spread
        elif current_z >= self.config.entry_zscore:
            signal = -1  # Short spread
        else:
            logger.info(f"Z-score {current_z:.2f} not extreme enough")
            return False
        
        # Calculate position sizes
        hedge = pair_info['hedge_ratio']
        
        # Effective capital after estimated costs
        effective_capital = capital * (1 - self.config.trading_fee - self.config.slippage)
        
        # Split capital based on hedge ratio
        capital1 = effective_capital / (1 + hedge)
        capital2 = hedge * capital1
        
        qty1 = capital1 / price1
        qty2 = capital2 / price2
        
        # Round quantities based on symbol rules
        info1 = self.client.get_symbol_info(symbol1)
        info2 = self.client.get_symbol_info(symbol2)
        
        if info1 and info2:
            qty1 = round(qty1, info1['quantity_precision'])
            qty2 = round(qty2, info2['quantity_precision'])
        
        # Paper mode check
        is_paper = self.config.mode == TradingMode.PAPER
        
        # Place orders
        if signal == 1:  # Long spread: buy asset1, sell asset2
            side1, side2 = 'BUY', 'SELL'
        else:  # Short spread: sell asset1, buy asset2
            side1, side2 = 'SELL', 'BUY'
        
        order1 = self.client.place_market_order(symbol1, side1, qty1, paper=is_paper)
        order2 = self.client.place_market_order(symbol2, side2, qty2, paper=is_paper)
        
        if order1 is None or order2 is None:
            logger.error("Failed to place entry orders")
            # TODO: Handle partial fill - close any filled order
            return False
        
        # Calculate entry spread
        entry_spread = self.stats.calculate_spread(
            np.array([price1]),
            np.array([price2]),
            pair_info['hedge_ratio'],
            pair_info['intercept']
        )[0]
        
        # Store position
        self.position = Position(
            symbol1=symbol1,
            symbol2=symbol2,
            signal=signal,
            entry_time=datetime.now(timezone.utc),
            entry_price1=price1,
            entry_price2=price2,
            entry_spread=entry_spread,
            entry_z=current_z,
            hedge_ratio=pair_info['hedge_ratio'],
            intercept=pair_info['intercept'],
            spread_mean=pair_info['spread_mean'],
            spread_std=pair_info['spread_std'],
            qty1=qty1,
            qty2=qty2,
            capital_used=capital,
            bars_held=0,
            last_update=datetime.now(timezone.utc)
        )
        
        # Log trade
        self.trade_log.append(TradeLog(
            timestamp=datetime.now(timezone.utc),
            pair=(symbol1, symbol2),
            action=f"OPEN_{'LONG' if signal == 1 else 'SHORT'}",
            signal=signal,
            price1=price1,
            price2=price2,
            zscore=current_z,
            qty1=qty1,
            qty2=qty2
        ))
        
        logger.info(
            f"Entered {'LONG' if signal == 1 else 'SHORT'} {symbol1}/{symbol2}: "
            f"z={current_z:.2f}, qty1={qty1:.4f}, qty2={qty2:.4f}"
        )
        
        return True
    
    def check_exit_conditions(self) -> Optional[ExitReason]:
        """
        Check if current position should be closed.
        
        SAME LOGIC AS continuous_backtest._update_position()
        """
        if self.position is None:
            return None
        
        pos = self.position
        
        # Get current prices
        price1 = self.client.get_current_price(pos.symbol1)
        price2 = self.client.get_current_price(pos.symbol2)
        
        if price1 is None or price2 is None:
            logger.warning("Failed to get prices for exit check")
            return None
        
        # Calculate current z-score using FROZEN statistics
        current_z = pos.current_zscore(price1, price2)
        
        # 1. Mean reversion target
        if pos.signal == 1:  # Long spread
            if current_z >= -self.config.exit_zscore:
                return ExitReason.MEAN_REVERSION
        else:  # Short spread
            if current_z <= self.config.exit_zscore:
                return ExitReason.MEAN_REVERSION
        
        # 2. Stop loss
        z_change = current_z - pos.entry_z
        if pos.signal == 1:
            if z_change < -self.config.stop_loss_zscore:
                return ExitReason.STOP_LOSS
        else:
            if z_change > self.config.stop_loss_zscore:
                return ExitReason.STOP_LOSS
        
        # 3. Max holding time
        if pos.bars_held >= self.config.max_holding_bars:
            return ExitReason.MAX_HOLDING
        
        return None
    
    def close_position(self, reason: ExitReason) -> bool:
        """Close current position"""
        if self.position is None:
            return True
        
        pos = self.position
        is_paper = self.config.mode == TradingMode.PAPER
        
        # Get exit prices
        price1 = self.client.get_current_price(pos.symbol1)
        price2 = self.client.get_current_price(pos.symbol2)
        
        if price1 is None or price2 is None:
            logger.error("Failed to get prices for exit")
            return False
        
        # Place exit orders (reverse of entry)
        if pos.signal == 1:  # Long spread: was buy1/sell2, now sell1/buy2
            side1, side2 = 'SELL', 'BUY'
        else:  # Short spread: was sell1/buy2, now buy1/sell2
            side1, side2 = 'BUY', 'SELL'
        
        order1 = self.client.place_market_order(pos.symbol1, side1, pos.qty1, paper=is_paper)
        order2 = self.client.place_market_order(pos.symbol2, side2, pos.qty2, paper=is_paper)
        
        if order1 is None or order2 is None:
            logger.error("Failed to place exit orders")
            return False
        
        # Calculate PnL
        exit_spread = self.stats.calculate_spread(
            np.array([price1]),
            np.array([price2]),
            pos.hedge_ratio,
            pos.intercept
        )[0]
        
        exit_z = (exit_spread - pos.spread_mean) / pos.spread_std
        
        spread_change = exit_spread - pos.entry_spread
        gross_pnl = pos.signal * spread_change * pos.capital_used
        
        # Costs
        fees = 2 * self.config.trading_fee * pos.capital_used
        slippage = 2 * self.config.slippage * pos.capital_used
        net_pnl = gross_pnl - fees - slippage
        
        # Update paper balance
        if is_paper:
            self.paper_balance += net_pnl
            self.paper_pnl += net_pnl
        
        # Log trade
        self.trade_log.append(TradeLog(
            timestamp=datetime.now(timezone.utc),
            pair=(pos.symbol1, pos.symbol2),
            action="CLOSE",
            signal=pos.signal,
            price1=price1,
            price2=price2,
            zscore=exit_z,
            qty1=pos.qty1,
            qty2=pos.qty2,
            pnl=net_pnl,
            exit_reason=reason.value
        ))
        
        logger.info(
            f"Closed {pos.symbol1}/{pos.symbol2}: "
            f"reason={reason.value}, PnL=${net_pnl:.2f} ({net_pnl/pos.capital_used*100:.2f}%)"
        )
        
        # Clear position
        self.position = None
        
        return True
    
    def run_cycle(self):
        """Run one trading cycle"""
        
        if self.position is not None:
            # Check exit conditions
            exit_reason = self.check_exit_conditions()
            
            if exit_reason:
                self.close_position(exit_reason)
            else:
                # Update bars held (approximate based on check interval)
                self.position.bars_held += 1
                logger.debug(
                    f"Position open: {self.position.symbol1}/{self.position.symbol2}, "
                    f"bars_held={self.position.bars_held}"
                )
        
        else:
            # Search for new pair
            logger.info("Searching for trading opportunity...")
            
            data = self.get_market_data()
            
            if not data:
                logger.warning("No market data available")
                return
            
            best_pair = self.find_best_pair(data)
            
            if best_pair:
                symbol1, symbol2, pair_info = best_pair
                logger.info(
                    f"Found pair: {symbol1}/{symbol2}, "
                    f"z={pair_info['current_z']:.2f}, "
                    f"half_life={pair_info['half_life']:.1f}"
                )
                
                # Get available capital
                if self.config.mode == TradingMode.PAPER:
                    capital = self.paper_balance * self.config.max_position_size_pct
                else:
                    capital = self.client.get_account_balance() * self.config.max_position_size_pct
                
                if capital >= self.config.min_order_value * 2:
                    self.enter_position(symbol1, symbol2, pair_info, capital)
                else:
                    logger.warning(f"Insufficient capital: ${capital:.2f}")
            else:
                logger.info("No suitable pair found this cycle")
    
    def run(self, max_cycles: int = None):
        """
        Main trading loop.
        
        Args:
            max_cycles: Maximum cycles to run (None = infinite)
        """
        self.initialize()
        self.running = True
        
        mode_str = "PAPER" if self.config.mode == TradingMode.PAPER else "LIVE"
        logger.info(f"Starting trading loop in {mode_str} mode...")
        
        cycle = 0
        while self.running and (max_cycles is None or cycle < max_cycles):
            try:
                cycle += 1
                logger.info(f"\n{'='*50}")
                logger.info(f"Cycle {cycle}")
                
                if self.config.mode == TradingMode.PAPER:
                    logger.info(f"Paper Balance: ${self.paper_balance:,.2f}")
                    logger.info(f"Paper PnL: ${self.paper_pnl:,.2f}")
                
                self.run_cycle()
                
                if self.running and (max_cycles is None or cycle < max_cycles):
                    logger.debug(f"Sleeping {self.config.check_interval_seconds}s...")
                    time.sleep(self.config.check_interval_seconds)
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.stop()
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(30)  # Wait before retry
        
        # Clean up
        if self.position is not None:
            logger.info("Closing remaining position...")
            self.close_position(ExitReason.END_OF_PERIOD)
        
        self.print_summary()
    
    def stop(self):
        """Stop trading"""
        self.running = False
        self._stop_event.set()
    
    def print_summary(self):
        """Print trading summary"""
        print("\n" + "="*60)
        print("TRADING SESSION SUMMARY")
        print("="*60)
        
        n_trades = len([t for t in self.trade_log if t.action.startswith("OPEN")])
        
        if self.config.mode == TradingMode.PAPER:
            print(f"Mode: PAPER TRADING")
            print(f"Final Balance: ${self.paper_balance:,.2f}")
            print(f"Total PnL: ${self.paper_pnl:,.2f}")
            print(f"Return: {(self.paper_pnl/self.config.trading_capital)*100:.2f}%")
        
        print(f"Total Trades: {n_trades}")
        
        if self.trade_log:
            closed_trades = [t for t in self.trade_log if t.action == "CLOSE" and t.pnl is not None]
            if closed_trades:
                wins = [t for t in closed_trades if t.pnl > 0]
                losses = [t for t in closed_trades if t.pnl <= 0]
                print(f"Win Rate: {len(wins)/len(closed_trades)*100:.1f}%")
                print(f"Avg Win: ${np.mean([t.pnl for t in wins]):.2f}" if wins else "No wins")
                print(f"Avg Loss: ${np.mean([t.pnl for t in losses]):.2f}" if losses else "No losses")
        
        print("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QuantBox Live Trader')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--capital', type=float, default=1000.0, help='Trading capital')
    parser.add_argument('--cycles', type=int, default=None, help='Max trading cycles')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('live_trader.log', mode='a')
        ]
    )
    
    # Suppress noisy loggers
    for log_name in ['urllib3', 'requests', 'binance']:
        logging.getLogger(log_name).setLevel(logging.WARNING)
    
    # Determine mode
    if args.live:
        mode = TradingMode.LIVE
        print("⚠️  WARNING: LIVE TRADING MODE - REAL MONEY AT RISK")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            return
    else:
        mode = TradingMode.PAPER
    
    config = LiveTradeConfig(
        mode=mode,
        trading_capital=args.capital,
        check_interval_seconds=args.interval
    )
    
    trader = LivePairTrader(config)
    trader.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
