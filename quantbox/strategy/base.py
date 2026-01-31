"""
Base strategy interface.

All strategies should inherit from the Strategy base class
to ensure compatibility with the trading engine.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import logging


logger = logging.getLogger('Strategy')


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Strategies must implement:
    - on_bar(): Called on each new bar of data
    - on_signal(): Called to generate trading signals
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.is_active = True
        
        # State
        self._positions: Dict[str, float] = {}
        self._signals: List[Dict] = []
        
    @abstractmethod
    def on_bar(
        self,
        timestamp: datetime,
        data: Dict[str, Dict[str, float]]
    ) -> Optional[Dict]:
        """
        Called on each new bar of market data.
        
        Args:
            timestamp: Current bar timestamp
            data: Dict of symbol -> OHLCV data
            
        Returns:
            Signal dict or None
        """
        pass
    
    @abstractmethod
    def calculate_signal(self, **kwargs) -> Optional[Dict]:
        """
        Calculate trading signal based on current state.
        
        Returns:
            Signal dict with keys: type, symbols, parameters
        """
        pass
    
    def set_position(self, symbol: str, quantity: float):
        """Update internal position tracking."""
        self._positions[symbol] = quantity
    
    def get_position(self, symbol: str) -> float:
        """Get current position for symbol."""
        return self._positions.get(symbol, 0.0)
    
    def has_position(self, symbol: str) -> bool:
        """Check if has open position."""
        return abs(self.get_position(symbol)) > 1e-10
    
    def reset(self):
        """Reset strategy state."""
        self._positions.clear()
        self._signals.clear()
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return self.config.copy()
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.config.update(kwargs)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        return True
    
    def log_signal(self, signal: Dict):
        """Log a generated signal."""
        self._signals.append({
            'timestamp': datetime.utcnow(),
            **signal
        })
        logger.info(f"[{self.name}] Signal: {signal}")


class StatefulStrategy(Strategy):
    """
    Strategy with state management for pair trading.
    """
    
    def __init__(self, name: str, config: Dict = None):
        super().__init__(name, config)
        
        # Pair state
        self._pair_state: Dict[tuple, Dict] = {}
    
    def get_pair_state(self, symbol1: str, symbol2: str) -> Dict:
        """Get state for a trading pair."""
        key = (symbol1, symbol2)
        if key not in self._pair_state:
            self._pair_state[key] = {
                'entry_time': None,
                'entry_z_score': 0.0,
                'entry_spread': 0.0,
                'signal': 0,  # -1 short, 0 flat, 1 long
                'bars_held': 0,
            }
        return self._pair_state[key]
    
    def set_pair_state(self, symbol1: str, symbol2: str, **kwargs):
        """Set state for a trading pair."""
        state = self.get_pair_state(symbol1, symbol2)
        state.update(kwargs)
    
    def reset_pair_state(self, symbol1: str, symbol2: str):
        """Reset state for a trading pair."""
        key = (symbol1, symbol2)
        if key in self._pair_state:
            del self._pair_state[key]
    
    def reset(self):
        """Reset all state."""
        super().reset()
        self._pair_state.clear()
