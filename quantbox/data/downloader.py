"""
Data downloader for Binance exchange.

Provides efficient data download with:
- Rate limiting
- Batch processing
- Error handling and retries
- Integration with caching system
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from quantbox.core.config import Config, Interval


logger = logging.getLogger('DataDownloader')


class DataDownloader:
    """
    Binance data downloader with intelligent caching integration.
    """
    
    # Column names for OHLCV data
    KLINE_COLUMNS = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]
    
    def __init__(
        self,
        client,
        config: Config,
        cache=None
    ):
        self.client = client
        self.config = config
        self.cache = cache
        
        # Rate limiting
        self.rate_limit = config.exchange.rate_limit
        self.last_request_time = 0
        
        # Batch settings
        self.batch_limit = 1000  # Max klines per request
        
        logger.info("DataDownloader initialized")
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        mapping = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return mapping.get(interval, 60 * 60 * 1000)
    
    def _datetime_to_ms(self, dt: datetime) -> int:
        """Convert datetime to milliseconds timestamp."""
        return int(dt.timestamp() * 1000)
    
    def _ms_to_datetime(self, ms: int) -> datetime:
        """Convert milliseconds timestamp to datetime."""
        return datetime.utcfromtimestamp(ms / 1000)
    
    def get_trading_coins(
        self,
        quote_currency: str = "USDT",
        n_coins: int = 50,
        sort_by_volume: bool = True
    ) -> List[str]:
        """
        Get list of trading coins.
        
        Args:
            quote_currency: Quote currency to filter by
            n_coins: Number of coins to return
            sort_by_volume: Sort by 24h trading volume
            
        Returns:
            List of trading pair symbols
        """
        try:
            exchange_info = self.client.futures_exchange_info()
            
            # Filter active trading pairs
            trading_coins = [
                info['symbol']
                for info in exchange_info['symbols']
                if info['symbol'].endswith(quote_currency) 
                and info['status'] == 'TRADING'
            ]
            
            logger.info(f"Found {len(trading_coins)} tradable pairs")
            
            if sort_by_volume:
                # Get volumes
                volume_dict = {}
                for coin in tqdm(trading_coins, desc="Fetching volumes"):
                    try:
                        self._rate_limit_wait()
                        ticker = self.client.get_ticker(symbol=coin)
                        volume_dict[coin] = float(ticker['quoteVolume'])
                    except Exception as e:
                        logger.debug(f"Failed to get volume for {coin}: {e}")
                
                # Sort by volume
                sorted_coins = sorted(
                    volume_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                trading_coins = [coin for coin, _ in sorted_coins[:n_coins]]
            else:
                trading_coins = trading_coins[:n_coins]
            
            logger.info(f"Selected top {len(trading_coins)} coins")
            return trading_coins
            
        except Exception as e:
            logger.error(f"Failed to get trading coins: {e}")
            return []
    
    def download_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download kline data for a symbol.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            start: Start datetime
            end: End datetime
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first
        if use_cache and self.cache:
            cached = self.cache.get(symbol, interval, start, end)
            if cached is not None:
                return cached
        
        # Download in batches
        start_ms = self._datetime_to_ms(start)
        end_ms = self._datetime_to_ms(end)
        interval_ms = self._interval_to_ms(interval)
        
        all_klines = []
        current_ms = start_ms
        
        while current_ms < end_ms:
            try:
                self._rate_limit_wait()
                
                batch_end = min(current_ms + self.batch_limit * interval_ms, end_ms)
                
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=str(current_ms),
                    end_str=str(batch_end),
                    limit=self.batch_limit
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_ms = klines[-1][6] + 1  # Close time + 1ms
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} at {current_ms}: {e}")
                current_ms += interval_ms * self.batch_limit
        
        if not all_klines:
            logger.warning(f"No data downloaded for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=self.KLINE_COLUMNS)
        df = df.astype(float)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Cache result
        if use_cache and self.cache:
            self.cache.put(symbol, interval, start, end, df)
        
        logger.info(f"Downloaded {len(df)} bars for {symbol}")
        return df
    
    def download_multiple(
        self,
        symbols: List[str],
        interval: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
        progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download kline data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            interval: Time interval
            start: Start datetime
            end: End datetime
            use_cache: Whether to use caching
            progress: Show progress bar
            
        Returns:
            Dict of symbol -> DataFrame
        """
        results = {}
        
        iterator = tqdm(symbols, desc="Downloading") if progress else symbols
        
        for symbol in iterator:
            try:
                df = self.download_klines(symbol, interval, start, end, use_cache)
                if df is not None and len(df) > 0:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
        
        logger.info(f"Downloaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def download_multi_timeframe(
        self,
        symbols: List[str],
        intervals: List[str],
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download data for multiple symbols and timeframes.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            start: Start datetime
            end: End datetime
            use_cache: Whether to use caching
            
        Returns:
            Dict of symbol -> interval -> DataFrame
        """
        results = {}
        
        total = len(symbols) * len(intervals)
        with tqdm(total=total, desc="Downloading multi-TF") as pbar:
            for symbol in symbols:
                results[symbol] = {}
                for interval in intervals:
                    try:
                        df = self.download_klines(
                            symbol, interval, start, end, use_cache
                        )
                        if df is not None:
                            results[symbol][interval] = df
                    except Exception as e:
                        logger.error(f"Failed {symbol}/{interval}: {e}")
                    pbar.update(1)
        
        return results
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        try:
            self._rate_limit_wait()
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price > 0:
                prices[symbol] = price
        return prices


def create_downloader(client, config: Config, cache=None) -> DataDownloader:
    """Factory function to create data downloader."""
    return DataDownloader(client, config, cache)
