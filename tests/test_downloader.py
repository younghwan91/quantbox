"""Unit tests for quantbox/data/downloader.py.

All external I/O (Binance client, cache) is mocked.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantbox.data.downloader import DataDownloader, create_downloader


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_config(rate_limit: float = 0.0) -> MagicMock:
    cfg = MagicMock()
    cfg.exchange.rate_limit = rate_limit
    return cfg


def _make_kline(open_time_ms: int, close_time_ms: int, price: float = 100.0) -> list:
    """Build a single Binance kline list entry."""
    return [
        open_time_ms,  # Open Time
        str(price),  # Open
        str(price + 1),  # High
        str(price - 1),  # Low
        str(price),  # Close
        "500.0",  # Volume
        close_time_ms,  # Close Time
        "50000.0",  # Quote Asset Volume
        100,  # Number of Trades
        "250.0",  # Taker Buy Base Asset Volume
        "25000.0",  # Taker Buy Quote Asset Volume
        "0",  # Ignore
    ]


def _make_klines(n: int, interval_ms: int = 60_000, start_ms: int = 1_600_000_000_000) -> list:
    """Generate n synthetic klines starting at start_ms."""
    klines = []
    for i in range(n):
        open_ms = start_ms + i * interval_ms
        close_ms = open_ms + interval_ms - 1
        klines.append(_make_kline(open_ms, close_ms, price=100.0 + i))
    return klines


def _make_downloader(rate_limit: float = 0.0, cache=None) -> DataDownloader:
    client = MagicMock()
    config = _make_config(rate_limit)
    return DataDownloader(client, config, cache)


START = datetime(2024, 1, 1)
END = datetime(2024, 1, 2)

# ──────────────────────────────────────────────────────────────────────────────
#  init & factory
# ──────────────────────────────────────────────────────────────────────────────


class TestInit:
    def test_batch_limit_is_1000(self):
        dl = _make_downloader()
        assert dl.batch_limit == 1000

    def test_factory_function(self):
        client = MagicMock()
        config = _make_config()
        dl = create_downloader(client, config)
        assert isinstance(dl, DataDownloader)
        assert dl.client is client


# ──────────────────────────────────────────────────────────────────────────────
#  _interval_to_ms
# ──────────────────────────────────────────────────────────────────────────────


class TestIntervalToMs:
    def test_15m(self):
        dl = _make_downloader()
        assert dl._interval_to_ms("15m") == 15 * 60 * 1000

    def test_4h(self):
        dl = _make_downloader()
        assert dl._interval_to_ms("4h") == 4 * 60 * 60 * 1000

    def test_1d(self):
        dl = _make_downloader()
        assert dl._interval_to_ms("1d") == 24 * 60 * 60 * 1000

    def test_unknown_defaults_to_1h(self):
        dl = _make_downloader()
        assert dl._interval_to_ms("99x") == 60 * 60 * 1000


# ──────────────────────────────────────────────────────────────────────────────
#  _datetime_to_ms / _ms_to_datetime
# ──────────────────────────────────────────────────────────────────────────────


class TestDatetimeConversion:
    def test_datetime_to_ms_is_integer(self):
        dl = _make_downloader()
        dt = datetime(2024, 1, 1)
        ms = dl._datetime_to_ms(dt)
        assert isinstance(ms, int)
        assert ms > 0

    def test_ms_to_datetime_returns_datetime(self):
        dl = _make_downloader()
        # 2024-01-01 00:00:00 UTC in milliseconds
        ms = 1_704_067_200_000
        dt = dl._ms_to_datetime(ms)
        assert isinstance(dt, datetime)
        assert dt.year == 2024


# ──────────────────────────────────────────────────────────────────────────────
#  download_klines
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadKlines:
    def test_returns_dataframe_with_ohlcv(self):
        dl = _make_downloader()
        interval_ms = 15 * 60 * 1000
        klines = _make_klines(10, interval_ms)
        # Only one batch needed; return empty on second call to stop loop
        dl.client.get_historical_klines.side_effect = [klines, []]
        df = dl.download_klines("BTCUSDT", "15m", START, END)
        assert df is not None
        assert len(df) == 10
        assert "Close" in df.columns
        assert "Volume" in df.columns

    def test_returns_none_on_empty_response(self):
        dl = _make_downloader()
        dl.client.get_historical_klines.return_value = []
        df = dl.download_klines("BTCUSDT", "15m", START, END)
        assert df is None

    def test_index_is_datetime(self):
        dl = _make_downloader()
        interval_ms = 60 * 1000
        klines = _make_klines(5, interval_ms)
        dl.client.get_historical_klines.side_effect = [klines, []]
        df = dl.download_klines("BTCUSDT", "1m", START, END)
        assert df is not None
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_deduplicates_rows(self):
        """If the same candle appears twice, it should be deduped."""
        dl = _make_downloader()
        interval_ms = 60 * 1000
        klines = _make_klines(5, interval_ms)
        # Duplicate the batch
        dl.client.get_historical_klines.side_effect = [klines + klines, []]
        df = dl.download_klines("BTCUSDT", "1m", START, END)
        assert df is not None
        assert not df.index.duplicated().any()

    def test_uses_cache_when_available(self):
        """Cache hit → client never called."""
        cached_df = pd.DataFrame({"Close": [100.0]}, index=[datetime(2024, 1, 1)])
        cache = MagicMock()
        cache.get.return_value = cached_df
        dl = _make_downloader(cache=cache)
        result = dl.download_klines("BTCUSDT", "15m", START, END)
        dl.client.get_historical_klines.assert_not_called()
        assert result is cached_df

    def test_skips_cache_when_disabled(self):
        """use_cache=False → cache.get never called even if cache provided."""
        cache = MagicMock()
        cache.get.return_value = None
        dl = _make_downloader(cache=cache)
        interval_ms = 60 * 1000
        klines = _make_klines(3, interval_ms)
        dl.client.get_historical_klines.side_effect = [klines, []]
        dl.download_klines("BTCUSDT", "1m", START, END, use_cache=False)
        cache.get.assert_not_called()

    def test_stores_to_cache_after_download(self):
        """After download, result should be put into cache."""
        cache = MagicMock()
        cache.get.return_value = None  # Cache miss
        dl = _make_downloader(cache=cache)
        interval_ms = 60 * 1000
        klines = _make_klines(3, interval_ms)
        dl.client.get_historical_klines.side_effect = [klines, []]
        dl.download_klines("BTCUSDT", "1m", START, END)
        cache.put.assert_called_once()

    def test_api_exception_returns_none(self):
        """If API throws, download_klines returns None gracefully."""
        dl = _make_downloader()
        dl.client.get_historical_klines.side_effect = Exception("Network error")
        df = dl.download_klines("BTCUSDT", "15m", START, END)
        assert df is None

    def test_numeric_columns_are_float(self):
        """All OHLCV columns should be float after conversion."""
        dl = _make_downloader()
        interval_ms = 60 * 1000
        klines = _make_klines(3, interval_ms)
        dl.client.get_historical_klines.side_effect = [klines, []]
        df = dl.download_klines("BTCUSDT", "1m", START, END)
        assert df is not None
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert df[col].dtype == float, f"{col} is not float"


# ──────────────────────────────────────────────────────────────────────────────
#  download_multiple
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadMultiple:
    def test_returns_dict_of_dataframes(self):
        dl = _make_downloader()
        interval_ms = 60 * 1000
        klines = _make_klines(3, interval_ms)
        dl.client.get_historical_klines.side_effect = [klines, [], klines, []]
        result = dl.download_multiple(["BTCUSDT", "ETHUSDT"], "1m", START, END, progress=False)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result

    def test_skips_failed_symbols(self):
        """If download_klines returns None for a symbol, it's excluded from results."""
        dl = _make_downloader()
        interval_ms = 60 * 1000
        klines = _make_klines(3, interval_ms)
        # Patch download_klines directly: FAILUSDT returns None, ETHUSDT returns data
        original = dl.download_klines

        def _mock_download_klines(symbol, *args, **kwargs):
            if symbol == "FAILUSDT":
                return None
            return original(symbol, *args, **kwargs)

        dl.download_klines = _mock_download_klines
        dl.client.get_historical_klines.side_effect = [klines, []]
        result = dl.download_multiple(["FAILUSDT", "ETHUSDT"], "1m", START, END, progress=False)
        assert "FAILUSDT" not in result
        assert "ETHUSDT" in result


# ──────────────────────────────────────────────────────────────────────────────
#  get_trading_coins
# ──────────────────────────────────────────────────────────────────────────────


class TestGetTradingCoins:
    def test_filters_by_quote_currency(self):
        dl = _make_downloader()
        dl.client.futures_exchange_info.return_value = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},
                {"symbol": "ETHBTC", "status": "TRADING"},  # wrong quote
                {"symbol": "XRPUSDT", "status": "BREAK"},  # not TRADING
            ]
        }
        coins = dl.get_trading_coins(quote_currency="USDT", sort_by_volume=False)
        assert coins == ["BTCUSDT"]

    def test_returns_empty_on_exception(self):
        dl = _make_downloader()
        dl.client.futures_exchange_info.side_effect = Exception("API error")
        coins = dl.get_trading_coins()
        assert coins == []


# ──────────────────────────────────────────────────────────────────────────────
#  get_latest_price
# ──────────────────────────────────────────────────────────────────────────────


class TestGetLatestPrice:
    def test_returns_float_price(self):
        dl = _make_downloader()
        dl.client.futures_symbol_ticker.return_value = {"price": "42000.5"}
        price = dl.get_latest_price("BTCUSDT")
        assert price == pytest.approx(42000.5)

    def test_returns_zero_on_exception(self):
        dl = _make_downloader()
        dl.client.futures_symbol_ticker.side_effect = Exception("fail")
        price = dl.get_latest_price("BTCUSDT")
        assert price == 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  get_latest_prices (plural)
# ──────────────────────────────────────────────────────────────────────────────


class TestGetLatestPrices:
    def test_returns_valid_prices_only(self):
        dl = _make_downloader()
        dl.client.futures_symbol_ticker.side_effect = [
            {"price": "100.0"},
            Exception("fail"),
            {"price": "200.0"},
        ]
        prices = dl.get_latest_prices(["AAVEUSDT", "FAILUSDT", "ETHUSDT"])
        assert "AAVEUSDT" in prices
        assert "FAILUSDT" not in prices
        assert "ETHUSDT" in prices
