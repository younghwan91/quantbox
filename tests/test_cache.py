"""
Tests for data caching system.
"""

import pytest
import tempfile
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from quantbox.data.cache import (
    MemoryCache, DiskCache, DataCache, TimeframedCache, CacheEntry
)


class TestMemoryCache:
    """Tests for in-memory cache."""
    
    def test_put_and_get(self):
        cache = MemoryCache(max_size=100)
        
        entry = CacheEntry(key="key1", data="value1")
        cache.put(entry)
        
        result = cache.get("key1")
        assert result is not None
        assert result.data == "value1"
    
    def test_missing_key(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None
    
    def test_remove(self):
        cache = MemoryCache()
        entry = CacheEntry(key="key1", data="value1")
        cache.put(entry)
        cache.remove("key1")
        assert cache.get("key1") is None
    
    def test_clear(self):
        cache = MemoryCache()
        cache.put(CacheEntry(key="key1", data="value1"))
        cache.put(CacheEntry(key="key2", data="value2"))
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_expiration(self):
        cache = MemoryCache(max_size=100)
        entry = CacheEntry(
            key="key1", 
            data="value1",
            expires_at=datetime.now(timezone.utc) + timedelta(milliseconds=100)
        )
        cache.put(entry)
        
        assert cache.get("key1") is not None
        
        time.sleep(0.2)
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2)
        
        cache.put(CacheEntry(key="key1", data="value1"))
        cache.put(CacheEntry(key="key2", data="value2"))
        cache.put(CacheEntry(key="key3", data="value3"))
        
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
    
    def test_size(self):
        cache = MemoryCache(max_size=100)
        assert cache.size == 0
        
        cache.put(CacheEntry(key="key1", data="value1"))
        assert cache.size == 1


class TestDiskCache:
    """Tests for disk-based cache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_put_and_get(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        entry = CacheEntry(key="test_df", data=df)
        cache.put(entry)
        
        retrieved = cache.get("test_df")
        assert retrieved is not None
        pd.testing.assert_frame_equal(df, retrieved.data)
    
    def test_compression(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir, compress=True)
        
        df = pd.DataFrame({'a': range(1000), 'b': range(1000)})
        entry = CacheEntry(key="test_df", data=df)
        cache.put(entry)
        
        retrieved = cache.get("test_df")
        assert retrieved is not None
        pd.testing.assert_frame_equal(df, retrieved.data)
    
    def test_missing_key(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        assert cache.get("nonexistent") is None
    
    def test_remove(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        entry = CacheEntry(key="key1", data="value1")
        cache.put(entry)
        assert cache.get("key1") is not None
        
        cache.remove("key1")
        assert cache.get("key1") is None
    
    def test_clear(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        cache.put(CacheEntry(key="key1", data="value1"))
        cache.put(CacheEntry(key="key2", data="value2"))
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestDataCache:
    """Tests for two-tier DataCache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_put_and_get(self, temp_cache_dir):
        cache = DataCache(cache_dir=temp_cache_dir)
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        
        cache.put("BTCUSDT", "1h", start, end, df)
        retrieved = cache.get("BTCUSDT", "1h", start, end)
        
        assert retrieved is not None
        pd.testing.assert_frame_equal(df, retrieved)
    
    def test_cache_miss(self, temp_cache_dir):
        cache = DataCache(cache_dir=temp_cache_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        
        result = cache.get("BTCUSDT", "1h", start, end)
        assert result is None
        assert cache._stats['misses'] == 1


class TestTimeframedCache:
    """Tests for TimeframedCache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_put_and_get(self, temp_cache_dir):
        cache = TimeframedCache(base_dir=temp_cache_dir)
        
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103]
        }, index=pd.date_range('2023-01-01', periods=3, freq='h'))
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        
        cache.put("BTCUSDT", "1h", start, end, df)
        
        retrieved = cache.get("BTCUSDT", "1h", start, end)
        assert retrieved is not None
        pd.testing.assert_frame_equal(df, retrieved)
    
    def test_multiple_timeframes(self, temp_cache_dir):
        cache = TimeframedCache(base_dir=temp_cache_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        
        for tf in ['1h', '4h', '1d']:
            df = pd.DataFrame({
                'close': [100, 101, 102]
            }, index=pd.date_range('2023-01-01', periods=3, freq=tf))
            
            cache.put("BTCUSDT", tf, start, end, df)
        
        for tf in ['1h', '4h', '1d']:
            assert cache.get("BTCUSDT", tf, start, end) is not None
    
    def test_get_all_stats(self, temp_cache_dir):
        cache = TimeframedCache(base_dir=temp_cache_dir)
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        df = pd.DataFrame({'close': [100]})
        
        cache.put("BTCUSDT", "1h", start, end, df)
        
        stats = cache.get_all_stats()
        assert isinstance(stats, dict)
