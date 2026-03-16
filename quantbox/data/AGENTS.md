<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-16 -->

# quantbox/data

## Purpose

Market data acquisition and caching layer. Downloads OHLCV klines from Binance with intelligent 2-tier caching (memory + disk), respects Binance rate limits (2400 req/min), and enforces symbol whitelist policy. All data is real — no synthetic or mock data.

## Files

| File | Purpose | Key Exports |
| --- | --- | --- |
| `downloader.py` | Binance kline downloader with rate limiting | `DataDownloader` class |
| `real_data_loader.py` | High-level data loading API (single call → full OHLCV) | `RealDataLoader` class |
| `cache.py` | 2-tier cache system (memory LRU + disk persistent) | `MemoryCache`, `DiskCache`, `DataCache` |
| `__init__.py` | Module exports | — |

## Architecture: 2-Tier Cache

**Memory Tier (L1):**

- LRU cache, max 100 entries
- Fast access (< 1ms)
- Expires after TTL (default 24h)
- Lost on process restart

**Disk Tier (L2):**

- Gzip-compressed pickle files in `data/cache/` directory
- Survives process restarts
- Expires based on age (default 7 days)
- Fallback when memory cache misses

**Flow:**

1. Request symbol/interval/range
2. Check memory cache (hit → return)
3. Check disk cache (hit → load to memory, return)
4. Download from Binance (subject to rate limit)
5. Cache to memory and disk
6. Return data

## Binance Rate Limiting

Binance allows **2400 requests per minute** (40 req/sec average). The system enforces this via:

- **Per-downloader throttle:** `_rate_limit_wait()` method adds 25ms delay between requests (40 req/sec)
- **Batch downloads:** Request up to 1000 klines per API call (1 request for 1000 bars, not 1 request per bar)
- **Cache reuse:** Repeated requests for same symbol/interval hit cache (0 API cost)

**Example:** Downloading 200-bar history for 100 symbols:

- Without cache: 100 requests → 2.5 seconds
- With cache hit: 100 memory lookups → < 100ms
- Disk cache: 100 disk reads → ~1 second

## Symbol Whitelist Enforcement

**Policy:** Only download data for symbols in `cfg.symbol_whitelist` (config-driven). Never use `exclude_symbols` pattern.

**Rationale:** Whitelist is explicit (start with known good set), exclude list is implicit (easy to forget to update).

**Implementation:**

- `RealDataLoader` validates each symbol against whitelist before download
- Raises error if symbol not whitelisted (fail-fast)
- Backtest scripts import whitelist from config, not hardcoded

**Code pattern:**

```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
loader = RealDataLoader()
for symbol in cfg.symbol_whitelist:
    ohlcv = loader.load(symbol, interval="15m", n_bars=200)
```python

## Critical Rules for Agents

1. **No mock data ever.** All backtests and live trading use real Binance data via `RealDataLoader`. Never generate synthetic data with `np.random` or fake OHLCV arrays.

2. **Cache is transparent.** From the caller's perspective, `loader.load()` returns fresh data. Caching is internal; never assume stale data in memory.

3. **Respect rate limits.** If downloading data for 100+ symbols in a loop, the rate limiter will pause between requests. Don't try to bypass it.

4. **Symbol whitelist is source of truth.** If a symbol is not in `cfg.symbol_whitelist`, don't download it. Config drives the universe, not ad-hoc lists.

5. **Interval must be valid.** Binance supports: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`. Invalid interval → ValueError.

6. **Downloaded data is complete bars only.** Never use incomplete (still-forming) candles for signals. Download completed bars; skip the current bar.

7. **Cache eviction is LRU.** When memory cache exceeds 100 entries, least-recently-used entry is dropped. Recent symbols are kept.

8. **Disk cache is best-effort.** If disk write fails, the downloader continues without crashing. Cache is a performance optimization, not required for correctness.

## For AI Agents

### Task: Download data for backtesting

Use `RealDataLoader`:

```python
from quantbox.data.real_data_loader import RealDataLoader
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

cfg = VolStraddleConfig()
loader = RealDataLoader()

for symbol in cfg.symbol_whitelist:
    ohlcv = loader.load(symbol, interval="15m", n_bars=200)
    # ohlcv is ndarray shape (200, 5): [open, high, low, close, volume]
```python

### Task: Check cache status

```python
from quantbox.data.cache import DataCache
cache = DataCache()
# Memory cache: cache.memory_cache.get(key)
# Disk cache: cache.disk_cache.get(key)
```python

### Task: Clear stale cache entries

```python
from quantbox.data.cache import DataCache
cache = DataCache()
cache.disk_cache.cleanup()  # Remove files older than 7 days
```python

### Task: Add a new symbol to trading universe

1. Update `.env` or environment variable: `VS_SYMBOL_WHITELIST=BTCUSDT,ETHUSDT,NEWUSDT`
2. Restart the app
3. First data request for NEWUSDT will download from Binance and cache it
4. Subsequent requests hit cache

### Task: Monitor API usage

The `DataDownloader` logs each request. Enable DEBUG logging:

```python
import logging
logging.getLogger("DataDownloader").setLevel(logging.DEBUG)
```python

Each log shows timestamp, symbol, interval, and number of klines fetched.

## Cache TTL and Expiration

**Memory cache:** 24 hours (default)

- Old entries are dropped when memory cache fills beyond 100 entries
- LRU algorithm: least-recently-used first

**Disk cache:** 7 days (default)

- Files older than 7 days are removed by `cleanup()` call
- Call `cleanup()` weekly to prevent disk bloat

**Intraday updates:**

- If symbol is trading actively, latest bar is incomplete (still forming)
- Download fresh data every cycle (minute or 15m depending on strategy)
- Old cache entries are overwritten with new data
- No stale data issues because cache is validated per-request

## Testing Considerations

- **No network access in unit tests.** Mock `RealDataLoader.load()` to return fixture OHLCV arrays.
- **Cache isolation.** Use temporary cache directory for each test run.
- **Whitelist validation.** Test that non-whitelisted symbols are rejected.
- **Rate limit compliance.** Integration tests must not exceed 40 req/sec average.
- **Cache invalidation.** Verify disk cache cleanup removes old files correctly.
