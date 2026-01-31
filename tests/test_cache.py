"""
Tests for data caching system.
"""

import pytest
import tempfile
import os
import time
import pandas as pd
import numpy as np

from quantbox.data.cache import (
    MemoryCache, DiskCache, DataCache, TimeframedCache
)


class TestMemoryCache:
    """Tests for in-memory cache."""
    
    def test_set_and_get(self):
        cache = MemoryCache(max_size=100)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_missing_key(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"
    
    def test_delete(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_clear(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_expiration(self):
        cache = MemoryCache(default_ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")
        
        # Should exist immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_contains(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        
        assert "key1" in cache
        assert "key2" not in cache
    
    def test_stats(self):
        cache = MemoryCache()
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1


class TestDiskCache:
    """Tests for disk-based cache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_set_and_get(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cache.set("test_df", df)
        
        retrieved = cache.get("test_df")
        pd.testing.assert_frame_equal(df, retrieved)
    
    def test_compression(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir, compress=True)
        
        df = pd.DataFrame({'a': range(1000), 'b': range(1000)})
        cache.set("test_df", df)
        
        retrieved = cache.get("test_df")
        pd.testing.assert_frame_equal(df, retrieved)
    
    def test_missing_key(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        assert cache.get("nonexistent") is None
    
    def test_delete(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        cache.set("key1", "value1")
        assert cache.get("key1") is not None
        
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_clear(self, temp_cache_dir):
        cache = DiskCache(cache_dir=temp_cache_dir)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestDataCache:
    """Tests for two-tier DataCache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_memory_first(self, temp_cache_dir):
        cache = DataCache(cache_dir=temp_cache_dir)
        
        cache.set("key1", "value1")
        
        # Should be in memory cache
        assert cache.memory_cache.get("key1") == "value1"
    
    def test_disk_fallback(self, temp_cache_dir):
        cache = DataCache(cache_dir=temp_cache_dir)
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        cache.set("test_df", df, persist=True)
        
        # Clear memory cache
        cache.memory_cache.clear()
        
        # Should retrieve from disk
        retrieved = cache.get("test_df")
        pd.testing.assert_frame_equal(df, retrieved)
    
    def test_prefetch(self, temp_cache_dir):
        cache = DataCache(cache_dir=temp_cache_dir)
        
        # Store some data on disk
        for i in range(3):
            cache.set(f"key_{i}", f"value_{i}", persist=True)
        
        # Clear memory
        cache.memory_cache.clear()
        
        # Prefetch
        cache.prefetch([f"key_{i}" for i in range(3)])
        
        # Should now be in memory
        for i in range(3):
            assert cache.memory_cache.get(f"key_{i}") is not None


class TestTimeframedCache:
    """Tests for TimeframedCache."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_store_and_retrieve(self, temp_cache_dir):
        cache = TimeframedCache(cache_dir=temp_cache_dir)
        
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103]
        }, index=pd.date_range('2023-01-01', periods=3, freq='h'))
        
        cache.store("BTCUSDT", "1h", df)
        
        retrieved = cache.get("BTCUSDT", "1h")
        pd.testing.assert_frame_equal(df, retrieved)
    
    def test_multiple_timeframes(self, temp_cache_dir):
        cache = TimeframedCache(cache_dir=temp_cache_dir)
        
        for tf in ['1h', '4h', '1d']:
            df = pd.DataFrame({
                'Close': [100, 101, 102]
            }, index=pd.date_range('2023-01-01', periods=3, freq=tf))
            
            cache.store("BTCUSDT", tf, df)
        
        for tf in ['1h', '4h', '1d']:
            assert cache.get("BTCUSDT", tf) is not None
    
    def test_list_symbols(self, temp_cache_dir):
        cache = TimeframedCache(cache_dir=temp_cache_dir)
        
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            df = pd.DataFrame({'Close': [100]})
            cache.store(symbol, "1h", df)
        
        symbols = cache.list_symbols()
        assert set(symbols) == {'BTCUSDT', 'ETHUSDT', 'BNBUSDT'}
