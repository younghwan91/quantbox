"""
Real Market Data Loader

This module provides the ONLY source of data for backtesting and analysis.
NO synthetic/fake data generation is allowed anywhere in the codebase.

Usage:
    from quantbox.data.real_data_loader import RealDataLoader
    
    loader = RealDataLoader()
    data = loader.load_data()  # Returns Dict[str, pd.DataFrame]
"""

import json
import gzip
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger('RealDataLoader')


class RealDataLoader:
    """
    Loads REAL market data from Binance Futures cache.
    
    This is the ONLY acceptable source of data for:
    - Backtesting
    - Analysis
    - Testing
    - Validation
    
    NO synthetic/fake data is ever generated or used.
    """
    
    # Standard symbols for testing when specific symbols are not required
    DEFAULT_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 
        'BNBUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT'
    ]
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Path to cache directory. Defaults to 'data/cache' in project root.
        """
        if cache_dir is None:
            # Try to find project root by looking for cache_index.json
            current = Path(__file__).resolve().parent
            for parent in [current] + list(current.parents):
                test_path = parent / 'data' / 'cache' / 'cache_index.json'
                if test_path.exists():
                    cache_dir = parent / 'data' / 'cache'
                    break
            
            # Fallback: look for pyproject.toml and use data/cache
            if cache_dir is None:
                current = Path(__file__).resolve().parent
                for parent in list(current.parents):
                    if (parent / 'pyproject.toml').exists():
                        cache_dir = parent / 'data' / 'cache'
                        break
            
            if cache_dir is None:
                cache_dir = Path('data/cache')
        
        self.cache_dir = Path(cache_dir)
        self._index = None
        self._data_cache = {}
    
    @property
    def index(self) -> Dict:
        """Load and cache the index file."""
        if self._index is None:
            index_file = self.cache_dir / "cache_index.json"
            if index_file.exists():
                with open(index_file) as f:
                    self._index = json.load(f)
            else:
                self._index = {}
        return self._index
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available symbols in cache."""
        symbols = set()
        for key, info in self.index.items():
            symbol = info.get('metadata', {}).get('symbol', key.split('_')[0])
            if symbol and 'USDC' not in symbol:  # Skip stablecoins
                symbols.add(symbol)
        return sorted(symbols)
    
    def load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol.
        
        Args:
            symbol: Symbol name (e.g., 'BTCUSDT')
            
        Returns:
            DataFrame with OHLCV data, or None if not found
        """
        if symbol in self._data_cache:
            return self._data_cache[symbol].copy()
        
        for key, info in self.index.items():
            cached_symbol = info.get('metadata', {}).get('symbol', key.split('_')[0])
            if cached_symbol != symbol:
                continue
            
            filepath = Path(info['filepath'])
            if not filepath.exists():
                continue
            
            try:
                with gzip.open(filepath, 'rb') as f:
                    obj = pickle.load(f)
                
                # Handle CacheEntry objects
                if hasattr(obj, 'data'):
                    df = obj.data
                else:
                    df = obj
                
                if not isinstance(df, pd.DataFrame):
                    continue
                
                # Normalize column names
                df = self._normalize_columns(df)
                
                if 'Close' not in df.columns:
                    logger.warning(f"No Close column in {symbol}")
                    continue
                
                self._data_cache[symbol] = df
                return df.copy()
                
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        return None
    
    def load_data(
        self,
        symbols: List[str] = None,
        min_bars: int = 100,
        align_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of symbols to load. If None, loads DEFAULT_SYMBOLS.
            min_bars: Minimum number of bars required per symbol.
            align_data: Whether to align all data to common time range.
            
        Returns:
            Dict mapping symbol names to DataFrames
        """
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS
        
        data = {}
        for symbol in symbols:
            df = self.load_symbol(symbol)
            if df is not None and len(df) >= min_bars:
                data[symbol] = df
        
        if not data:
            raise ValueError(
                f"No data available. Please ensure cache exists at {self.cache_dir}. "
                "Run the downloader to fetch real data from Binance."
            )
        
        if align_data and len(data) > 1:
            data = self._align_data(data)
        
        logger.info(f"Loaded {len(data)} symbols with real market data")
        return data
    
    def load_all(self, min_bars: int = 100, align_data: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all available symbols from cache.
        
        Args:
            min_bars: Minimum number of bars required per symbol.
            align_data: Whether to align all data to common time range.
            
        Returns:
            Dict mapping symbol names to DataFrames
        """
        symbols = self.get_available_symbols()
        return self.load_data(symbols=symbols, min_bars=min_bars, align_data=align_data)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        rename_map = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        return df
    
    def _align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all DataFrames to common time range."""
        if not data:
            return data
        
        # Find common time range
        common_start = max(df.index.min() for df in data.values())
        common_end = min(df.index.max() for df in data.values())
        
        aligned = {}
        for symbol, df in data.items():
            aligned_df = df.loc[common_start:common_end]
            if len(aligned_df) > 0:
                aligned[symbol] = aligned_df
        
        logger.info(f"Aligned data: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars")
        return aligned
    
    def get_data_info(self) -> Dict:
        """Get information about available cached data."""
        info = {
            'cache_dir': str(self.cache_dir),
            'cache_exists': self.cache_dir.exists(),
            'symbols': self.get_available_symbols(),
            'n_symbols': len(self.get_available_symbols()),
        }
        
        if info['n_symbols'] > 0:
            # Get sample info
            sample_symbol = info['symbols'][0]
            sample_df = self.load_symbol(sample_symbol)
            if sample_df is not None:
                info['sample_bars'] = len(sample_df)
                info['date_range'] = {
                    'start': str(sample_df.index.min()),
                    'end': str(sample_df.index.max())
                }
        
        return info


# Global instance for convenience
_default_loader = None


def get_real_data(
    symbols: List[str] = None,
    min_bars: int = 100,
    align_data: bool = True,
    cache_dir: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenient function to load real market data.
    
    Args:
        symbols: List of symbols to load. If None, loads default symbols.
        min_bars: Minimum number of bars required.
        align_data: Whether to align data to common time range.
        cache_dir: Optional custom cache directory.
        
    Returns:
        Dict mapping symbol names to DataFrames
        
    Raises:
        ValueError: If no data is available
        
    Example:
        >>> data = get_real_data(['BTCUSDT', 'ETHUSDT'])
        >>> print(f"Loaded {len(data)} symbols")
    """
    global _default_loader
    
    if _default_loader is None or (cache_dir is not None):
        _default_loader = RealDataLoader(cache_dir)
    
    return _default_loader.load_data(
        symbols=symbols,
        min_bars=min_bars,
        align_data=align_data
    )
