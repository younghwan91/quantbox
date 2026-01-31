"""
Backtest runner with comprehensive simulation capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from quantbox.core.config import Config
from quantbox.core.engine import TradingEngine
from quantbox.core.events import SignalType, SignalEvent
from quantbox.strategy.pair_strategy import PairTradingStrategy
from quantbox.strategy.pair_selector import PairSelector, PairCandidate


logger = logging.getLogger('BacktestRunner')


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    initial_capital: float = 10000.0
    trading_fee: float = 0.0005
    slippage: float = 0.001
    
    # Window settings
    formation_periods: int = 240
    trading_periods: int = 24
    window_shift: int = 24  # Bars to shift between windows
    
    # Strategy parameters
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 3.5
    max_holding_periods: int = 100
    
    # Pair selection
    max_pairs_per_window: int = 10
    min_correlation: float = 0.5
    max_cointegration_pvalue: float = 0.05


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    pair: Tuple[str, str]
    entry_time: datetime
    exit_time: datetime
    entry_z_score: float
    exit_z_score: float
    entry_spread: float
    exit_spread: float
    signal: int  # 1 for long, -1 for short
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str
    hedge_ratio: float


class BacktestRunner:
    """
    Comprehensive backtest runner for pair trading.
    
    Features:
    - Rolling window backtesting
    - Walk-forward analysis
    - Monte Carlo simulation
    - Transaction cost modeling
    """
    
    def __init__(
        self,
        config: BacktestConfig = None
    ):
        self.config = config or BacktestConfig()
        
        # Results storage
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self.pair_performance: Dict[Tuple, List[TradeRecord]] = {}
        
        # State
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        pair_selector: PairSelector = None,
        strategy: PairTradingStrategy = None,
        progress: bool = True
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            pair_selector: Pair selection logic
            strategy: Trading strategy
            progress: Show progress bar
            
        Returns:
            Dict with results and metrics
        """
        # Initialize
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        
        # Create default components if not provided
        if strategy is None:
            strategy = PairTradingStrategy(config={
                'entry_zscore': self.config.entry_zscore,
                'exit_zscore': self.config.exit_zscore,
                'stop_loss_zscore': self.config.stop_loss_zscore,
                'max_holding_periods': self.config.max_holding_periods,
            })
        
        # Get common timestamps
        first_df = next(iter(data.values()))
        timestamps = first_df.index.tolist()
        
        # Generate windows
        windows = self._generate_windows(timestamps)
        logger.info(f"Generated {len(windows)} trading windows")
        
        # Run backtest
        iterator = tqdm(windows, desc="Backtesting") if progress else windows
        
        for window_start, window_end, trade_start, trade_end in iterator:
            self._process_window(
                data, window_start, window_end, trade_start, trade_end,
                pair_selector, strategy
            )
        
        # Calculate final metrics
        results = self._compile_results()
        
        return results
    
    def _generate_windows(
        self,
        timestamps: List[datetime]
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate rolling windows for backtest.
        
        Returns list of (formation_start, formation_end, trade_start, trade_end)
        """
        windows = []
        total_bars = self.config.formation_periods + self.config.trading_periods
        
        i = 0
        while i + total_bars <= len(timestamps):
            formation_start = timestamps[i]
            formation_end = timestamps[i + self.config.formation_periods - 1]
            trade_start = timestamps[i + self.config.formation_periods]
            trade_end = timestamps[min(i + total_bars - 1, len(timestamps) - 1)]
            
            windows.append((formation_start, formation_end, trade_start, trade_end))
            i += self.config.window_shift
        
        return windows
    
    def _process_window(
        self,
        data: Dict[str, pd.DataFrame],
        formation_start: datetime,
        formation_end: datetime,
        trade_start: datetime,
        trade_end: datetime,
        pair_selector: PairSelector,
        strategy: PairTradingStrategy
    ):
        """Process a single trading window."""
        
        # Get formation period data
        formation_data = {
            symbol: df.loc[formation_start:formation_end]
            for symbol, df in data.items()
            if formation_start in df.index
        }
        
        # Select pairs for this window
        if pair_selector:
            candidates = pair_selector.select_pairs(formation_data, progress=False)
            tradeable = pair_selector.select_tradeable_pairs(
                candidates,
                min_z_score=self.config.entry_zscore,
                max_z_score=5.0
            )
        else:
            # Use all valid pairs
            tradeable = []
        
        # Limit pairs
        pairs_to_trade = tradeable[:self.config.max_pairs_per_window]
        
        if not pairs_to_trade:
            return
        
        # Initialize strategy with pairs
        strategy.reset()
        for candidate in pairs_to_trade:
            s1, s2 = candidate.symbol1, candidate.symbol2
            if s1 in formation_data and s2 in formation_data:
                strategy.add_pair(candidate, formation_data[s1], formation_data[s2])
        
        # Simulate trading period
        trading_data = {
            symbol: df.loc[trade_start:trade_end]
            for symbol, df in data.items()
        }
        
        self._simulate_trading(
            trading_data, strategy, pairs_to_trade
        )
    
    def _simulate_trading(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: PairTradingStrategy,
        pairs: List[PairCandidate]
    ):
        """Simulate trading for a period."""
        
        # Get timestamps
        first_df = next(iter(data.values()))
        timestamps = first_df.index.tolist()
        
        # Track positions
        positions: Dict[Tuple, Dict] = {}  # pair -> position info
        
        for timestamp in timestamps:
            # Prepare bar data
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
            
            # Get strategy signal
            signal = strategy.on_bar(timestamp, bar_data)
            
            if signal:
                self._process_signal(
                    signal, timestamp, bar_data, positions
                )
            
            # Update equity
            self._update_equity(timestamp, bar_data, positions)
        
        # Close any remaining positions at end
        for pair, pos in list(positions.items()):
            if pos['is_open']:
                self._close_position(
                    pair, pos, timestamps[-1], bar_data, "end_of_window"
                )
    
    def _process_signal(
        self,
        signal: Dict,
        timestamp: datetime,
        bar_data: Dict[str, Dict],
        positions: Dict
    ):
        """Process a trading signal."""
        
        pair = (signal['symbol1'], signal['symbol2'])
        signal_type = signal['type']
        
        if signal_type in [SignalType.LONG, SignalType.SHORT]:
            # Entry signal
            if pair not in positions or not positions[pair].get('is_open', False):
                price1 = bar_data[signal['symbol1']]['close']
                price2 = bar_data[signal['symbol2']]['close']
                
                # Calculate position size
                capital_per_pair = self.current_capital / self.config.max_pairs_per_window
                
                positions[pair] = {
                    'is_open': True,
                    'signal': 1 if signal_type == SignalType.LONG else -1,
                    'entry_time': timestamp,
                    'entry_z_score': signal['z_score'],
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'hedge_ratio': signal['hedge_ratio'],
                    'capital': capital_per_pair,
                    'bars_held': 0
                }
                
        elif signal_type == SignalType.CLOSE_ALL:
            # Exit signal
            if pair in positions and positions[pair].get('is_open', False):
                self._close_position(
                    pair, positions[pair], timestamp, bar_data,
                    signal.get('reason', 'signal')
                )
                positions[pair]['is_open'] = False
    
    def _close_position(
        self,
        pair: Tuple[str, str],
        position: Dict,
        timestamp: datetime,
        bar_data: Dict,
        reason: str
    ):
        """Close a position and record the trade."""
        
        s1, s2 = pair
        exit_price1 = bar_data[s1]['close']
        exit_price2 = bar_data[s2]['close']
        
        # Calculate spread change
        entry_spread = np.log(position['entry_price1']) - position['hedge_ratio'] * np.log(position['entry_price2'])
        exit_spread = np.log(exit_price1) - position['hedge_ratio'] * np.log(exit_price2)
        spread_change = exit_spread - entry_spread
        
        # Calculate PnL (simplified)
        signal = position['signal']
        gross_pnl = signal * spread_change * position['capital']
        
        # Subtract fees
        fee = 2 * self.config.trading_fee * position['capital']  # Entry + exit
        slippage = 2 * self.config.slippage * position['capital']
        net_pnl = gross_pnl - fee - slippage
        
        pnl_pct = net_pnl / position['capital'] if position['capital'] > 0 else 0
        
        # Record trade
        trade = TradeRecord(
            pair=pair,
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_z_score=position['entry_z_score'],
            exit_z_score=0,  # Would need to recalculate
            entry_spread=entry_spread,
            exit_spread=exit_spread,
            signal=signal,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            bars_held=position['bars_held'],
            exit_reason=reason,
            hedge_ratio=position['hedge_ratio']
        )
        
        self.trades.append(trade)
        
        # Update capital
        self.current_capital += net_pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Track pair performance
        if pair not in self.pair_performance:
            self.pair_performance[pair] = []
        self.pair_performance[pair].append(trade)
    
    def _update_equity(
        self,
        timestamp: datetime,
        bar_data: Dict,
        positions: Dict
    ):
        """Update equity curve."""
        
        # Calculate unrealized PnL
        unrealized_pnl = 0
        for pair, pos in positions.items():
            if pos.get('is_open', False):
                s1, s2 = pair
                if s1 in bar_data and s2 in bar_data:
                    current_price1 = bar_data[s1]['close']
                    current_price2 = bar_data[s2]['close']
                    
                    entry_spread = np.log(pos['entry_price1']) - pos['hedge_ratio'] * np.log(pos['entry_price2'])
                    current_spread = np.log(current_price1) - pos['hedge_ratio'] * np.log(current_price2)
                    
                    spread_change = current_spread - entry_spread
                    unrealized_pnl += pos['signal'] * spread_change * pos['capital']
        
        total_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'drawdown': (self.peak_capital - total_equity) / self.peak_capital if self.peak_capital > 0 else 0
        })
    
    def _compile_results(self) -> Dict:
        """Compile backtest results."""
        
        from quantbox.backtest.metrics import calculate_metrics
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        # Convert trades to DataFrame
        trades_data = [
            {
                'pair': f"{t.pair[0]}/{t.pair[1]}",
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'signal': 'LONG' if t.signal == 1 else 'SHORT',
                'entry_z_score': t.entry_z_score,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct * 100,
                'bars_held': t.bars_held,
                'exit_reason': t.exit_reason,
            }
            for t in self.trades
        ]
        trades_df = pd.DataFrame(trades_data)
        
        # Calculate metrics
        metrics = calculate_metrics(
            equity_curve=equity_df,
            trades=self.trades,
            initial_capital=self.config.initial_capital
        )
        
        return {
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': trades_df,
            'pair_performance': self._summarize_pair_performance(),
            'config': self.config
        }
    
    def _summarize_pair_performance(self) -> pd.DataFrame:
        """Summarize performance by pair."""
        
        summaries = []
        for pair, trades in self.pair_performance.items():
            pnls = [t.pnl for t in trades]
            pnl_pcts = [t.pnl_pct for t in trades]
            
            summaries.append({
                'pair': f"{pair[0]}/{pair[1]}",
                'num_trades': len(trades),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls) if pnls else 0,
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
                'avg_bars_held': np.mean([t.bars_held for t in trades]) if trades else 0,
            })
        
        return pd.DataFrame(summaries)


def run_simple_backtest(
    data: Dict[str, pd.DataFrame],
    config: Dict = None
) -> Dict:
    """
    Convenience function for running a simple backtest.
    """
    bt_config = BacktestConfig(**config) if config else BacktestConfig()
    runner = BacktestRunner(bt_config)
    return runner.run_backtest(data)
