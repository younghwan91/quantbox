"""
Pair trading strategy implementation.

Implements a complete pair trading strategy with:
- Entry/exit logic based on z-score
- Position management
- Risk management
- Integration with trading engine
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from quantbox.strategy.base import StatefulStrategy
from quantbox.strategy.pair_selector import PairCandidate
from quantbox.core.events import SignalType
from quantbox.data.preprocessor import DataPreprocessor


logger = logging.getLogger('PairTradingStrategy')


class PairTradingStrategy(StatefulStrategy):
    """
    Mean-reversion pair trading strategy.
    
    Trades pairs of cointegrated assets by going long the spread
    when z-score is negative (below mean) and short when positive.
    """
    
    DEFAULT_PARAMS = {
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
        'stop_loss_zscore': 3.5,
        'max_holding_periods': 100,
        'lookback_periods': 240,
        'recalculate_hedge': True,
        'use_dynamic_zscore': False,
        'dynamic_lookback': 50,
    }
    
    def __init__(
        self,
        name: str = "PairTrading",
        config: Dict = None
    ):
        params = {**self.DEFAULT_PARAMS, **(config or {})}
        super().__init__(name, params)
        
        self.preprocessor = DataPreprocessor()
        
        # Pair data cache
        self._pair_cache: Dict[Tuple[str, str], Dict] = {}
        
        # Price history for hedge recalculation
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._max_history = params['lookback_periods'] * 2
    
    def update_price_history(
        self,
        timestamp: datetime,
        data: Dict[str, Dict[str, float]]
    ):
        """Update price history for all symbols."""
        for symbol, ohlcv in data.items():
            if symbol not in self._price_history:
                self._price_history[symbol] = []
            
            self._price_history[symbol].append((timestamp, ohlcv['close']))
            
            # Trim old data
            if len(self._price_history[symbol]) > self._max_history:
                self._price_history[symbol] = self._price_history[symbol][-self._max_history:]
    
    def get_price_series(self, symbol: str) -> pd.Series:
        """Get price series for a symbol."""
        if symbol not in self._price_history:
            return pd.Series(dtype=float)
        
        history = self._price_history[symbol]
        return pd.Series(
            [p for _, p in history],
            index=[t for t, _ in history]
        )
    
    def initialize_pair(
        self,
        symbol1: str,
        symbol2: str,
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ):
        """
        Initialize a trading pair with historical data.
        """
        pair_data = self.preprocessor.process_pair(
            data1, data2, symbol1, symbol2,
            train_ratio=1.0  # Use all data for initial calculation
        )
        
        self._pair_cache[(symbol1, symbol2)] = {
            'hedge_ratio': pair_data.hedge_ratio,
            'intercept': pair_data.intercept,
            'spread_mean': pair_data.spread_mean,
            'spread_std': pair_data.spread_std,
            'half_life': pair_data.half_life,
            'last_updated': datetime.now(timezone.utc)
        }
        
        logger.info(
            f"Initialized pair {symbol1}/{symbol2}: "
            f"hedge={pair_data.hedge_ratio:.4f}, "
            f"half_life={pair_data.half_life:.1f}"
        )
    
    def get_pair_params(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[Dict]:
        """Get cached pair parameters."""
        return self._pair_cache.get((symbol1, symbol2))
    
    def calculate_spread(
        self,
        price1: float,
        price2: float,
        hedge_ratio: float,
        intercept: float = 0.0
    ) -> float:
        """Calculate spread from prices."""
        return np.log(price1) - (hedge_ratio * np.log(price2) + intercept)
    
    def calculate_z_score(
        self,
        spread: float,
        spread_mean: float,
        spread_std: float
    ) -> float:
        """Calculate z-score from spread."""
        if spread_std == 0:
            return 0.0
        return (spread - spread_mean) / spread_std
    
    def recalculate_hedge_ratio(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[Dict]:
        """
        Recalculate hedge ratio from recent price history.
        """
        series1 = self.get_price_series(symbol1)
        series2 = self.get_price_series(symbol2)
        
        if len(series1) < self.config['lookback_periods']:
            return None
        
        # Align series
        common_idx = series1.index.intersection(series2.index)
        s1 = series1.loc[common_idx]
        s2 = series2.loc[common_idx]
        
        # Calculate new hedge ratio
        log1 = np.log(s1)
        log2 = np.log(s2)
        
        hedge_ratio, intercept = np.polyfit(log2, log1, 1)
        
        # Calculate spread statistics
        spread = log1 - (hedge_ratio * log2 + intercept)
        
        return {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'last_updated': datetime.now(timezone.utc)
        }
    
    def should_enter_long(
        self,
        z_score: float,
        state: Dict
    ) -> bool:
        """
        Check if should enter long position.
        
        Long spread when z-score is significantly negative
        (spread below mean, expect reversion up).
        """
        if state['signal'] != 0:
            return False  # Already in position
        
        return z_score <= -self.config['entry_zscore']
    
    def should_enter_short(
        self,
        z_score: float,
        state: Dict
    ) -> bool:
        """
        Check if should enter short position.
        
        Short spread when z-score is significantly positive
        (spread above mean, expect reversion down).
        """
        if state['signal'] != 0:
            return False  # Already in position
        
        return z_score >= self.config['entry_zscore']
    
    def should_exit(
        self,
        z_score: float,
        state: Dict
    ) -> Tuple[bool, str]:
        """
        Check if should exit position.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        if state['signal'] == 0:
            return False, ""
        
        entry_z = state['entry_z_score']
        signal = state['signal']
        bars_held = state['bars_held']
        
        # Exit on mean reversion
        exit_threshold = self.config['exit_zscore']
        if signal == 1 and z_score >= -exit_threshold:
            return True, "mean_reversion"
        if signal == -1 and z_score <= exit_threshold:
            return True, "mean_reversion"
        
        # Stop loss
        stop_loss = self.config['stop_loss_zscore']
        if signal == 1 and z_score <= entry_z - stop_loss:
            return True, "stop_loss"
        if signal == -1 and z_score >= entry_z + stop_loss:
            return True, "stop_loss"
        
        # Max holding period
        if bars_held >= self.config['max_holding_periods']:
            return True, "max_holding"
        
        return False, ""
    
    def on_bar(
        self,
        timestamp: datetime,
        data: Dict[str, Dict[str, float]]
    ) -> Optional[Dict]:
        """
        Process new bar of data.
        
        This is called by the trading engine for each new bar.
        """
        # Update price history
        self.update_price_history(timestamp, data)
        
        # Process each tracked pair
        signals = []
        
        for (symbol1, symbol2), params in self._pair_cache.items():
            if symbol1 not in data or symbol2 not in data:
                continue
            
            signal = self._process_pair_bar(
                symbol1, symbol2,
                data[symbol1]['close'],
                data[symbol2]['close'],
                params,
                timestamp
            )
            
            if signal:
                signals.append(signal)
        
        return signals[0] if signals else None
    
    def _process_pair_bar(
        self,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float,
        params: Dict,
        timestamp: datetime
    ) -> Optional[Dict]:
        """Process a single pair for the current bar."""
        
        # Check if should recalculate hedge ratio
        if self.config['recalculate_hedge']:
            age_hours = (timestamp - params['last_updated']).total_seconds() / 3600
            if age_hours > 24:  # Recalculate daily
                new_params = self.recalculate_hedge_ratio(symbol1, symbol2)
                if new_params:
                    params.update(new_params)
                    self._pair_cache[(symbol1, symbol2)] = params
        
        # Calculate current spread and z-score
        spread = self.calculate_spread(
            price1, price2,
            params['hedge_ratio'],
            params.get('intercept', 0)
        )
        
        if self.config['use_dynamic_zscore']:
            # Use rolling z-score
            z_score = self._calculate_dynamic_zscore(symbol1, symbol2, spread)
        else:
            z_score = self.calculate_z_score(
                spread,
                params['spread_mean'],
                params['spread_std']
            )
        
        # Get pair state
        state = self.get_pair_state(symbol1, symbol2)
        state['bars_held'] += 1 if state['signal'] != 0 else 0
        
        # Check for exit
        should_exit, exit_reason = self.should_exit(z_score, state)
        if should_exit:
            self.reset_pair_state(symbol1, symbol2)
            return {
                'type': SignalType.CLOSE_ALL,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'z_score': z_score,
                'reason': exit_reason,
                'hedge_ratio': params['hedge_ratio']
            }
        
        # Check for entry
        if self.should_enter_long(z_score, state):
            self.set_pair_state(
                symbol1, symbol2,
                entry_time=timestamp,
                entry_z_score=z_score,
                entry_spread=spread,
                signal=1,
                bars_held=0
            )
            return {
                'type': SignalType.LONG,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'z_score': z_score,
                'hedge_ratio': params['hedge_ratio']
            }
        
        if self.should_enter_short(z_score, state):
            self.set_pair_state(
                symbol1, symbol2,
                entry_time=timestamp,
                entry_z_score=z_score,
                entry_spread=spread,
                signal=-1,
                bars_held=0
            )
            return {
                'type': SignalType.SHORT,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'z_score': z_score,
                'hedge_ratio': params['hedge_ratio']
            }
        
        return None
    
    def _calculate_dynamic_zscore(
        self,
        symbol1: str,
        symbol2: str,
        current_spread: float
    ) -> float:
        """Calculate z-score using rolling statistics."""
        series1 = self.get_price_series(symbol1)
        series2 = self.get_price_series(symbol2)
        
        lookback = self.config['dynamic_lookback']
        
        if len(series1) < lookback:
            return 0.0
        
        # Get recent spreads
        common_idx = series1.index.intersection(series2.index)[-lookback:]
        s1 = series1.loc[common_idx]
        s2 = series2.loc[common_idx]
        
        params = self._pair_cache.get((symbol1, symbol2), {})
        hedge_ratio = params.get('hedge_ratio', 1.0)
        intercept = params.get('intercept', 0.0)
        
        log1 = np.log(s1)
        log2 = np.log(s2)
        spreads = log1 - (hedge_ratio * log2 + intercept)
        
        mean = spreads.mean()
        std = spreads.std()
        
        if std == 0:
            return 0.0
        
        return (current_spread - mean) / std
    
    def calculate_signal(
        self,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float
    ) -> Optional[Dict]:
        """
        Calculate trading signal for a pair.
        
        This can be called directly for manual signal generation.
        """
        params = self.get_pair_params(symbol1, symbol2)
        if not params:
            logger.warning(f"No parameters for pair {symbol1}/{symbol2}")
            return None
        
        return self._process_pair_bar(
            symbol1, symbol2,
            price1, price2,
            params,
            datetime.now(timezone.utc)
        )
    
    def add_pair(
        self,
        candidate: PairCandidate,
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ):
        """
        Add a pair candidate to the strategy.
        """
        self.initialize_pair(
            candidate.symbol1,
            candidate.symbol2,
            data1, data2
        )
        
        # Update with candidate statistics
        params = self._pair_cache[(candidate.symbol1, candidate.symbol2)]
        params['half_life'] = candidate.half_life
        params['correlation'] = candidate.correlation
    
    def remove_pair(self, symbol1: str, symbol2: str):
        """Remove a pair from the strategy."""
        key = (symbol1, symbol2)
        if key in self._pair_cache:
            del self._pair_cache[key]
        self.reset_pair_state(symbol1, symbol2)
    
    def get_active_pairs(self) -> List[Tuple[str, str]]:
        """Get list of active trading pairs."""
        return list(self._pair_cache.keys())
    
    def get_pair_status(
        self,
        symbol1: str,
        symbol2: str
    ) -> Dict:
        """Get current status of a pair."""
        params = self.get_pair_params(symbol1, symbol2)
        state = self.get_pair_state(symbol1, symbol2)
        
        return {
            'symbols': (symbol1, symbol2),
            'hedge_ratio': params.get('hedge_ratio') if params else None,
            'half_life': params.get('half_life') if params else None,
            'position': state['signal'],
            'bars_held': state['bars_held'],
            'entry_z_score': state['entry_z_score'],
        }
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._pair_cache.clear()
        self._price_history.clear()
