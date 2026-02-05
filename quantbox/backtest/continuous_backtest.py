"""
Continuous Pair Trading Backtest Engine

This module implements a high-frequency pair trading backtest system that:
1. Continuously searches for new pairs after each position close
2. Maximizes market participation
3. Strictly prevents lookahead bias
4. Accurately models transaction costs

Key Design Principles:
- Formation period and trading period are completely separated
- Hedge ratio and spread statistics are FIXED at position entry time
- Only information available at time T is used for decisions at time T
- Realistic fee and slippage modeling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

logger = logging.getLogger('ContinuousBacktest')


class ExitReason(Enum):
    """Trade exit reasons"""
    MEAN_REVERSION = "mean_reversion"
    STOP_LOSS = "stop_loss"
    MAX_HOLDING = "max_holding"
    END_OF_PERIOD = "end_of_period"
    PAIR_UNAVAILABLE = "pair_unavailable"


@dataclass
class TradeResult:
    """Individual trade result"""
    pair: Tuple[str, str]
    entry_time: datetime
    exit_time: datetime
    entry_z: float
    exit_z: float
    entry_spread: float
    exit_spread: float
    signal: int  # 1=long spread, -1=short spread
    capital_allocated: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    net_return_pct: float
    bars_held: int
    exit_reason: ExitReason
    hedge_ratio: float
    spread_mean: float
    spread_std: float


@dataclass
class BacktestState:
    """Backtest state tracking"""
    current_capital: float
    peak_capital: float
    position_open: bool = False
    current_pair: Optional[Tuple[str, str]] = None
    position_info: Optional[Dict] = None
    trades: List[TradeResult] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    search_count: int = 0  # How many times we searched for pairs
    no_trade_periods: int = 0  # Periods with no suitable pair found


@dataclass 
class ContinuousBacktestConfig:
    """Configuration for continuous backtest"""
    # Capital
    initial_capital: float = 10000.0
    
    # Transaction costs (per side)
    trading_fee: float = 0.0005  # 0.05% per trade (Binance maker)
    slippage: float = 0.0010    # 0.10% slippage estimate
    
    # Formation window for pair selection
    formation_bars: int = 240   # Bars to analyze for pair selection
    
    # Entry/Exit thresholds
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5    # Exit when z-score returns to this level
    stop_loss_zscore: float = 3.5  # Max adverse z-score movement
    
    # Holding limits
    max_holding_bars: int = 100
    
    # Pair selection criteria
    min_correlation: float = 0.6
    max_cointegration_pvalue: float = 0.05
    min_half_life: float = 2.0   # Minimum half-life in bars
    max_half_life: float = 50.0  # Maximum half-life in bars
    
    # Risk management
    position_size_pct: float = 1.0  # Fraction of capital per trade
    
    # Pair universe
    min_volume_rank: int = 50   # Only consider top N coins by volume
    
    @property
    def total_round_trip_cost(self) -> float:
        """Total cost for entry + exit"""
        return 2 * (self.trading_fee + self.slippage)


class PairStatistics:
    """Calculate pair statistics with strict lookahead prevention"""
    
    @staticmethod
    def calculate_hedge_ratio(prices1: np.ndarray, prices2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate hedge ratio using OLS regression on log prices.
        
        log(P1) = alpha + beta * log(P2) + epsilon
        
        Returns:
            (hedge_ratio, intercept)
        """
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        
        # OLS: minimize sum((log_p1 - alpha - beta*log_p2)^2)
        X = np.column_stack([np.ones(len(log_p2)), log_p2])
        coeffs = np.linalg.lstsq(X, log_p1, rcond=None)[0]
        
        intercept, hedge_ratio = coeffs
        return hedge_ratio, intercept
    
    @staticmethod
    def calculate_spread(
        prices1: np.ndarray, 
        prices2: np.ndarray,
        hedge_ratio: float,
        intercept: float
    ) -> np.ndarray:
        """Calculate spread series"""
        log_p1 = np.log(prices1)
        log_p2 = np.log(prices2)
        return log_p1 - (hedge_ratio * log_p2 + intercept)
    
    @staticmethod
    def calculate_zscore(
        spread: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """Calculate z-score series"""
        if std <= 0:
            return np.zeros_like(spread)
        return (spread - mean) / std
    
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using OLS.
        
        Fits: delta_spread = lambda * spread_lag + epsilon
        Half-life = -ln(2) / ln(1 + lambda)
        """
        spread_arr = spread.values if isinstance(spread, pd.Series) else spread
        
        spread_lag = spread_arr[:-1]
        spread_diff = spread_arr[1:] - spread_arr[:-1]
        
        # OLS regression
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        try:
            coeffs = np.linalg.lstsq(X, spread_diff, rcond=None)[0]
            lambda_param = coeffs[1]
            
            if lambda_param >= 0:
                return np.inf  # Not mean-reverting
            
            half_life = -np.log(2) / np.log(1 + lambda_param)
            return max(0, half_life)
        except:
            return np.inf
    
    @staticmethod
    def cointegration_test(prices1: np.ndarray, prices2: np.ndarray) -> Tuple[float, float]:
        """
        Perform Engle-Granger cointegration test.
        
        Returns:
            (t_statistic, p_value)
        """
        try:
            from statsmodels.tsa.stattools import coint
            t_stat, p_value, _ = coint(np.log(prices1), np.log(prices2))
            return t_stat, p_value
        except:
            return 0.0, 1.0
    
    @staticmethod
    def pearson_correlation(prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate correlation of log returns"""
        log_ret1 = np.diff(np.log(prices1))
        log_ret2 = np.diff(np.log(prices2))
        
        if len(log_ret1) < 2:
            return 0.0
        
        return np.corrcoef(log_ret1, log_ret2)[0, 1]


class ContinuousPairBacktest:
    """
    Continuous pair trading backtest engine.
    
    Key Features:
    - Searches for new pairs after each position close
    - Strictly prevents lookahead bias
    - Accurate transaction cost modeling
    - Comprehensive performance tracking
    """
    
    def __init__(self, config: ContinuousBacktestConfig = None):
        self.config = config or ContinuousBacktestConfig()
        self.stats = PairStatistics()
        
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_idx: int = None,
        end_idx: int = None,
        progress: bool = True
    ) -> Dict:
        """
        Run continuous backtest.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
                  Must have 'Close' column, optionally 'Volume'
            start_idx: Starting bar index (default: after formation period)
            end_idx: Ending bar index
            progress: Show progress bar
            
        Returns:
            Dictionary with results, metrics, trades
        """
        # Validate and prepare data
        data = self._prepare_data(data)
        if not data:
            raise ValueError("No valid data provided")
        
        # Get aligned timestamps
        symbols = list(data.keys())
        timestamps = data[symbols[0]].index.tolist()
        
        # Determine trading range
        start_idx = start_idx or self.config.formation_bars
        end_idx = end_idx or len(timestamps)
        
        if start_idx >= end_idx:
            raise ValueError("Not enough data for backtest")
        
        # Initialize state
        state = BacktestState(
            current_capital=self.config.initial_capital,
            peak_capital=self.config.initial_capital
        )
        
        logger.info(f"Starting continuous backtest")
        logger.info(f"  Period: {timestamps[start_idx]} to {timestamps[end_idx-1]}")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Initial capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"  Round-trip cost: {self.config.total_round_trip_cost*100:.2f}%")
        
        # Main backtest loop
        iterator = range(start_idx, end_idx)
        if progress:
            iterator = tqdm(iterator, desc="Backtesting")
        
        for bar_idx in iterator:
            timestamp = timestamps[bar_idx]
            
            # Get current bar data
            bar_data = self._get_bar_data(data, bar_idx)
            
            if state.position_open:
                # Update existing position
                self._update_position(state, bar_data, bar_idx, timestamp)
            else:
                # Search for new pair
                self._search_and_enter(state, data, bar_idx, timestamp)
            
            # Record equity
            self._record_equity(state, bar_data, timestamp)
        
        # Close any remaining position at end
        if state.position_open:
            final_bar = self._get_bar_data(data, end_idx - 1)
            self._close_position(state, final_bar, end_idx - 1, 
                               timestamps[end_idx - 1], ExitReason.END_OF_PERIOD)
        
        # Compile results
        return self._compile_results(state)
    
    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare and validate data"""
        prepared = {}
        
        # Find common timestamps
        common_idx = None
        for symbol, df in data.items():
            if 'Close' not in df.columns:
                logger.warning(f"Skipping {symbol}: no Close column")
                continue
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
        
        if common_idx is None or len(common_idx) < self.config.formation_bars + 10:
            return {}
        
        # Align all data to common index
        for symbol, df in data.items():
            if 'Close' not in df.columns:
                continue
            aligned = df.loc[common_idx].copy()
            if len(aligned) == len(common_idx):
                prepared[symbol] = aligned
        
        return prepared
    
    def _get_bar_data(
        self, 
        data: Dict[str, pd.DataFrame],
        bar_idx: int
    ) -> Dict[str, Dict]:
        """Get bar data for all symbols at given index"""
        bar_data = {}
        for symbol, df in data.items():
            if bar_idx < len(df):
                row = df.iloc[bar_idx]
                bar_data[symbol] = {
                    'close': row['Close'],
                    'high': row.get('High', row['Close']),
                    'low': row.get('Low', row['Close']),
                    'volume': row.get('Volume', 0)
                }
        return bar_data
    
    def _search_and_enter(
        self,
        state: BacktestState,
        data: Dict[str, pd.DataFrame],
        bar_idx: int,
        timestamp: datetime
    ):
        """Search for valid pair and enter position"""
        state.search_count += 1
        
        # Get formation period data (ONLY past data, no lookahead)
        formation_start = bar_idx - self.config.formation_bars
        if formation_start < 0:
            state.no_trade_periods += 1
            return
        
        # Find best pair from formation data
        best_pair = self._find_best_pair(
            data, formation_start, bar_idx - 1  # Exclude current bar!
        )
        
        if best_pair is None:
            state.no_trade_periods += 1
            return
        
        # Enter position using CURRENT bar price (slight lookahead acceptable for execution)
        symbol1, symbol2, pair_info = best_pair
        
        # Get entry prices at current bar
        entry_price1 = data[symbol1].iloc[bar_idx]['Close']
        entry_price2 = data[symbol2].iloc[bar_idx]['Close']
        
        # Calculate entry spread and z-score using FORMATION statistics (frozen)
        entry_spread = self.stats.calculate_spread(
            np.array([entry_price1]),
            np.array([entry_price2]),
            pair_info['hedge_ratio'],
            pair_info['intercept']
        )[0]
        
        entry_z = (entry_spread - pair_info['spread_mean']) / pair_info['spread_std']
        
        # Determine signal direction
        if entry_z <= -self.config.entry_zscore:
            signal = 1  # Long spread (expect z to increase)
        elif entry_z >= self.config.entry_zscore:
            signal = -1  # Short spread (expect z to decrease)
        else:
            # Z-score not extreme enough anymore (market moved)
            state.no_trade_periods += 1
            return
        
        # Calculate position size
        capital_allocated = state.current_capital * self.config.position_size_pct
        
        # Store position info
        state.position_open = True
        state.current_pair = (symbol1, symbol2)
        state.position_info = {
            'entry_bar': bar_idx,
            'entry_time': timestamp,
            'entry_price1': entry_price1,
            'entry_price2': entry_price2,
            'entry_spread': entry_spread,
            'entry_z': entry_z,
            'signal': signal,
            'capital': capital_allocated,
            # Frozen formation statistics (NO UPDATES during trade)
            'hedge_ratio': pair_info['hedge_ratio'],
            'intercept': pair_info['intercept'],
            'spread_mean': pair_info['spread_mean'],
            'spread_std': pair_info['spread_std'],
        }
        
        logger.debug(
            f"Entered {symbol1}/{symbol2} at bar {bar_idx}: "
            f"z={entry_z:.2f}, signal={'LONG' if signal==1 else 'SHORT'}"
        )
    
    def _find_best_pair(
        self,
        data: Dict[str, pd.DataFrame],
        start_idx: int,
        end_idx: int
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        Find best tradeable pair from formation period.
        
        CRITICAL: Only uses data from [start_idx, end_idx] - NO FUTURE DATA!
        """
        symbols = list(data.keys())
        candidates = []
        
        # Generate all pairs
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                # Get formation period prices ONLY
                prices1 = data[s1].iloc[start_idx:end_idx+1]['Close'].values
                prices2 = data[s2].iloc[start_idx:end_idx+1]['Close'].values
                
                if len(prices1) < 30 or len(prices2) < 30:
                    continue
                
                # Check minimum criteria
                if (prices1 <= 0).any() or (prices2 <= 0).any():
                    continue
                
                # 1. Correlation filter
                corr = self.stats.pearson_correlation(prices1, prices2)
                if corr < self.config.min_correlation:
                    continue
                
                # 2. Cointegration filter
                t_stat, p_value = self.stats.cointegration_test(prices1, prices2)
                if p_value > self.config.max_cointegration_pvalue:
                    continue
                
                # 3. Calculate hedge ratio and spread statistics
                hedge_ratio, intercept = self.stats.calculate_hedge_ratio(prices1, prices2)
                
                # Check hedge ratio bounds (avoid extreme leverage)
                if not 0.25 < hedge_ratio < 4.0:
                    continue
                
                spread = self.stats.calculate_spread(prices1, prices2, hedge_ratio, intercept)
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                
                if spread_std < 0.001:  # Too stable (lowered threshold)
                    continue
                
                # 4. Half-life filter
                half_life = self.stats.calculate_half_life(pd.Series(spread))
                if not (self.config.min_half_life < half_life < self.config.max_half_life):
                    continue
                
                # Z-score at end of formation (for tracking, not filtering)
                current_z = (spread[-1] - spread_mean) / spread_std
                
                # Add candidate WITHOUT requiring z-score at entry level
                # Entry decision will be made based on CURRENT bar in _search_and_enter
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
                    # Score: lower = better (faster mean reversion + stronger cointegration)
                    'score': half_life * p_value
                })
        
        if not candidates:
            return None
        
        # Select best pair (lowest score = fastest mean reversion + strongest cointegration)
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
            }
        )
    
    def _update_position(
        self,
        state: BacktestState,
        bar_data: Dict[str, Dict],
        bar_idx: int,
        timestamp: datetime
    ):
        """Update existing position, check for exit conditions"""
        pos = state.position_info
        s1, s2 = state.current_pair
        
        # Check data availability
        if s1 not in bar_data or s2 not in bar_data:
            self._close_position(state, bar_data, bar_idx, timestamp, 
                               ExitReason.PAIR_UNAVAILABLE)
            return
        
        # Calculate current spread using FROZEN hedge ratio
        current_price1 = bar_data[s1]['close']
        current_price2 = bar_data[s2]['close']
        
        current_spread = self.stats.calculate_spread(
            np.array([current_price1]),
            np.array([current_price2]),
            pos['hedge_ratio'],
            pos['intercept']
        )[0]
        
        # Calculate current z-score using FROZEN statistics
        current_z = (current_spread - pos['spread_mean']) / pos['spread_std']
        
        # Calculate bars held
        bars_held = bar_idx - pos['entry_bar']
        
        # Check exit conditions
        exit_reason = None
        
        # 1. Mean reversion target
        if pos['signal'] == 1:  # Long spread
            if current_z >= -self.config.exit_zscore:
                exit_reason = ExitReason.MEAN_REVERSION
        else:  # Short spread
            if current_z <= self.config.exit_zscore:
                exit_reason = ExitReason.MEAN_REVERSION
        
        # 2. Stop loss
        z_change = current_z - pos['entry_z']
        if pos['signal'] == 1:  # Long spread expects z to increase
            if z_change < -self.config.stop_loss_zscore:
                exit_reason = ExitReason.STOP_LOSS
        else:  # Short spread expects z to decrease
            if z_change > self.config.stop_loss_zscore:
                exit_reason = ExitReason.STOP_LOSS
        
        # 3. Max holding period
        if bars_held >= self.config.max_holding_bars:
            exit_reason = ExitReason.MAX_HOLDING
        
        if exit_reason:
            self._close_position(state, bar_data, bar_idx, timestamp, exit_reason)
    
    def _close_position(
        self,
        state: BacktestState,
        bar_data: Dict[str, Dict],
        bar_idx: int,
        timestamp: datetime,
        exit_reason: ExitReason
    ):
        """Close current position and record trade"""
        pos = state.position_info
        s1, s2 = state.current_pair
        
        # Get exit prices
        exit_price1 = bar_data.get(s1, {}).get('close', pos['entry_price1'])
        exit_price2 = bar_data.get(s2, {}).get('close', pos['entry_price2'])
        
        # Calculate exit spread using FROZEN hedge ratio
        exit_spread = self.stats.calculate_spread(
            np.array([exit_price1]),
            np.array([exit_price2]),
            pos['hedge_ratio'],
            pos['intercept']
        )[0]
        
        exit_z = (exit_spread - pos['spread_mean']) / pos['spread_std']
        
        # Calculate PnL from spread change
        spread_change = exit_spread - pos['entry_spread']
        gross_pnl = pos['signal'] * spread_change * pos['capital']
        
        # Calculate transaction costs
        # Fees: entry + exit both sides
        total_fees = self.config.trading_fee * 2 * pos['capital']
        
        # Slippage: proportional to position size
        total_slippage = self.config.slippage * 2 * pos['capital']
        
        net_pnl = gross_pnl - total_fees - total_slippage
        net_return_pct = (net_pnl / pos['capital']) * 100
        
        bars_held = bar_idx - pos['entry_bar']
        
        # Record trade
        trade = TradeResult(
            pair=state.current_pair,
            entry_time=pos['entry_time'],
            exit_time=timestamp,
            entry_z=pos['entry_z'],
            exit_z=exit_z,
            entry_spread=pos['entry_spread'],
            exit_spread=exit_spread,
            signal=pos['signal'],
            capital_allocated=pos['capital'],
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage_cost=total_slippage,
            net_pnl=net_pnl,
            net_return_pct=net_return_pct,
            bars_held=bars_held,
            exit_reason=exit_reason,
            hedge_ratio=pos['hedge_ratio'],
            spread_mean=pos['spread_mean'],
            spread_std=pos['spread_std']
        )
        
        state.trades.append(trade)
        
        # Update capital
        state.current_capital += net_pnl
        state.peak_capital = max(state.peak_capital, state.current_capital)
        
        # Reset position state
        state.position_open = False
        state.current_pair = None
        state.position_info = None
        
        logger.debug(
            f"Closed {s1}/{s2}: reason={exit_reason.value}, "
            f"PnL=${net_pnl:.2f} ({net_return_pct:.2f}%)"
        )
    
    def _record_equity(
        self,
        state: BacktestState,
        bar_data: Dict[str, Dict],
        timestamp: datetime
    ):
        """Record equity curve point"""
        # Calculate unrealized PnL if in position
        unrealized_pnl = 0.0
        
        if state.position_open and state.current_pair:
            pos = state.position_info
            s1, s2 = state.current_pair
            
            if s1 in bar_data and s2 in bar_data:
                current_price1 = bar_data[s1]['close']
                current_price2 = bar_data[s2]['close']
                
                current_spread = self.stats.calculate_spread(
                    np.array([current_price1]),
                    np.array([current_price2]),
                    pos['hedge_ratio'],
                    pos['intercept']
                )[0]
                
                spread_change = current_spread - pos['entry_spread']
                unrealized_pnl = pos['signal'] * spread_change * pos['capital']
        
        total_equity = state.current_capital + unrealized_pnl
        drawdown = (state.peak_capital - total_equity) / state.peak_capital if state.peak_capital > 0 else 0
        
        state.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': state.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'drawdown': drawdown,
            'in_position': state.position_open
        })
    
    def _compile_results(self, state: BacktestState) -> Dict:
        """Compile final backtest results"""
        # Create equity DataFrame
        equity_df = pd.DataFrame(state.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        metrics = self._calculate_metrics(state, equity_df)
        
        # Create trades DataFrame
        trades_data = []
        for t in state.trades:
            trades_data.append({
                'pair': f"{t.pair[0]}/{t.pair[1]}",
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'signal': 'LONG' if t.signal == 1 else 'SHORT',
                'entry_z': t.entry_z,
                'exit_z': t.exit_z,
                'capital': t.capital_allocated,
                'gross_pnl': t.gross_pnl,
                'fees': t.fees,
                'slippage': t.slippage_cost,
                'net_pnl': t.net_pnl,
                'return_pct': t.net_return_pct,
                'bars_held': t.bars_held,
                'exit_reason': t.exit_reason.value,
                'hedge_ratio': t.hedge_ratio
            })
        trades_df = pd.DataFrame(trades_data)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': trades_df,
            'state': state,
            'config': self.config
        }
    
    def _calculate_metrics(
        self, 
        state: BacktestState,
        equity_df: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics"""
        trades = state.trades
        
        # Basic metrics
        total_return = state.current_capital - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        # Trade statistics
        n_trades = len(trades)
        if n_trades == 0:
            return {
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'n_trades': 0,
                'win_rate': 0,
                'avg_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'message': 'No trades executed'
            }
        
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        
        win_rate = len(winning) / n_trades * 100
        avg_trade = np.mean([t.net_pnl for t in trades])
        avg_winner = np.mean([t.net_pnl for t in winning]) if winning else 0
        avg_loser = np.mean([t.net_pnl for t in losing]) if losing else 0
        
        # Profit factor
        gross_profit = sum(t.net_pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.net_pnl for t in losing)) if losing else 1e-10
        profit_factor = gross_profit / gross_loss
        
        # Drawdown
        max_dd = equity_df['drawdown'].max() * 100 if not equity_df.empty else 0
        
        # Calculate Sharpe ratio
        if not equity_df.empty and len(equity_df) > 1:
            equity_vals = equity_df['equity'].values
            returns = np.diff(equity_vals) / equity_vals[:-1]
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason.value
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Total fees and slippage paid
        total_fees = sum(t.fees for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        
        # Market participation rate
        if not equity_df.empty:
            in_market_pct = equity_df['in_position'].mean() * 100
        else:
            in_market_pct = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_capital': state.current_capital,
            'peak_capital': state.peak_capital,
            
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            
            'avg_trade': avg_trade,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'largest_win': max(t.net_pnl for t in trades),
            'largest_loss': min(t.net_pnl for t in trades),
            
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            
            'avg_holding_bars': np.mean([t.bars_held for t in trades]),
            'exit_reasons': exit_reasons,
            
            'total_fees_paid': total_fees,
            'total_slippage_paid': total_slippage,
            'total_costs': total_fees + total_slippage,
            
            'in_market_pct': in_market_pct,
            'search_count': state.search_count,
            'no_trade_periods': state.no_trade_periods,
        }


def run_continuous_backtest(
    data: Dict[str, pd.DataFrame],
    config: Dict = None
) -> Dict:
    """
    Convenience function to run continuous backtest.
    
    Args:
        data: Symbol -> OHLCV DataFrame
        config: Configuration overrides
        
    Returns:
        Backtest results
    """
    cfg = ContinuousBacktestConfig(**(config or {}))
    engine = ContinuousPairBacktest(cfg)
    return engine.run(data)
