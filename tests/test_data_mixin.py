"""Unit tests for quantbox/execution/data_mixin.py and quantbox/core/bot_utils.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantbox.execution.data_mixin import DataMixin


# ──────────────────────────────────────────────────────────────────────────────
#  Minimal DataMixin host (no Binance, no settings)
# ──────────────────────────────────────────────────────────────────────────────


def _make_host(whitelist: list[str] | None = None, exclude: list[str] | None = None):
    """Instantiate DataMixin with a minimal host that has the expected attrs."""

    class Host(DataMixin):
        def __init__(self):
            self.client = MagicMock()
            self._symbol_info_cache: dict = {}
            self._exchange_info_fetched: bool = False
            self._cfg = MagicMock()
            self._cfg.symbol_whitelist = whitelist or []
            self._cfg.exclude_symbols = exclude or []
            self._cfg.min_volume_rank = 50

    return Host()


def _make_kline(open_ms: int, close_ms: int, price: float = 100.0) -> list:
    """Return a single Binance kline row."""
    return [
        open_ms,  # 0  Open Time
        str(price),  # 1  Open
        str(price + 1),  # 2  High
        str(price - 1),  # 3  Low
        str(price),  # 4  Close
        "500.0",  # 5  Volume
        close_ms,  # 6  Close Time
        "50000.0",  # 7  Quote Volume
        100,  # 8  Trades
        "250.0",  # 9  Taker Buy Base
        "25000.0",  # 10 Taker Buy Quote
        "0",  # 11 Ignore
    ]


# ──────────────────────────────────────────────────────────────────────────────
#  DataMixin.download_klines
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadKlines:
    def test_returns_close_array_by_default(self):
        host = _make_host()
        klines = [_make_kline(i * 60_000, (i + 1) * 60_000 - 1, 100.0 + i) for i in range(5)]
        # Include one extra (the "live" candle that should be dropped)
        host.client.futures_klines.return_value = klines
        result = host.download_klines("BTCUSDT", n_bars=4)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # last bar (live) is dropped

    def test_returns_5tuple_when_full(self):
        host = _make_host()
        klines = [_make_kline(i * 60_000, (i + 1) * 60_000 - 1) for i in range(6)]
        host.client.futures_klines.return_value = klines
        result = host.download_klines("BTCUSDT", n_bars=5, full=True)
        assert isinstance(result, tuple)
        close, high, low, volume, taker = result
        assert len(close) == 5
        assert len(high) == 5

    def test_drops_last_incomplete_bar(self):
        """Binance returns the live (incomplete) candle — we must drop it."""
        host = _make_host()
        # 4 closed bars + 1 live bar = 5 total
        klines = [_make_kline(i * 1000, i * 1000 + 999, 100.0 + i) for i in range(5)]
        host.client.futures_klines.return_value = klines
        close = host.download_klines("BTCUSDT", n_bars=4)
        # close[-1] should be bar index 3 (not 4, the live bar)
        assert close[-1] == pytest.approx(103.0)

    def test_returns_none_on_empty_response(self):
        host = _make_host()
        host.client.futures_klines.return_value = []
        assert host.download_klines("BTCUSDT") is None

    def test_returns_none_on_single_bar_after_drop(self):
        """If only 1 kline returned, after dropping the live bar → empty → None."""
        host = _make_host()
        host.client.futures_klines.return_value = [_make_kline(0, 999)]
        assert host.download_klines("BTCUSDT") is None

    def test_returns_none_on_exception(self):
        host = _make_host()
        host.client.futures_klines.side_effect = Exception("Network error")
        assert host.download_klines("BTCUSDT") is None

    def test_uses_correct_interval(self):
        host = _make_host()
        klines = [_make_kline(i * 1000, i * 1000 + 999) for i in range(3)]
        host.client.futures_klines.return_value = klines
        host.download_klines("BTCUSDT", n_bars=2, interval="4h")
        host.client.futures_klines.assert_called_once_with(
            symbol="BTCUSDT",
            interval="4h",
            limit=3,  # n_bars + 1
        )


# ──────────────────────────────────────────────────────────────────────────────
#  DataMixin.get_symbol_info
# ──────────────────────────────────────────────────────────────────────────────


class TestGetSymbolInfo:
    def _exchange_info_response(self, symbol: str = "BTCUSDT"):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "quantityPrecision": 3,
                    "pricePrecision": 2,
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }

    def test_fetches_and_caches_symbol_info(self):
        host = _make_host()
        host.client.futures_exchange_info.return_value = self._exchange_info_response("BTCUSDT")
        info = host.get_symbol_info("BTCUSDT")
        assert info["step_size"] == pytest.approx(0.001)
        assert info["tick_size"] == pytest.approx(0.01)
        assert info["min_notional"] == pytest.approx(5.0)
        # Second call uses cache — client not called again
        host.get_symbol_info("BTCUSDT")
        host.client.futures_exchange_info.assert_called_once()

    def test_cache_hit_skips_api(self):
        host = _make_host()
        host._symbol_info_cache["BTCUSDT"] = {"step_size": 0.001}
        info = host.get_symbol_info("BTCUSDT")
        assert info["step_size"] == 0.001
        host.client.futures_exchange_info.assert_not_called()

    def test_returns_empty_dict_on_api_error(self):
        host = _make_host()
        host.client.futures_exchange_info.side_effect = Exception("API fail")
        info = host.get_symbol_info("XYZUSDT")
        assert info == {}

    def test_returns_empty_dict_for_unknown_symbol(self):
        host = _make_host()
        host.client.futures_exchange_info.return_value = self._exchange_info_response("BTCUSDT")
        info = host.get_symbol_info("UNKNOWNUSDT")
        assert info == {}


# ──────────────────────────────────────────────────────────────────────────────
#  DataMixin.get_trading_symbols
# ──────────────────────────────────────────────────────────────────────────────


class TestGetTradingSymbols:
    def _setup_exchange(self, host, symbols: list[str]):
        host.client.futures_exchange_info.return_value = {
            "symbols": [{"symbol": s, "status": "TRADING"} for s in symbols]
        }
        host.client.futures_ticker.return_value = [
            {"symbol": s, "quoteVolume": str((i + 1) * 1_000_000)} for i, s in enumerate(symbols)
        ]

    def test_whitelist_filters_to_tradable(self):
        host = _make_host(whitelist=["BTCUSDT", "ETHUSDT"])
        self._setup_exchange(host, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        syms = host.get_trading_symbols()
        assert "BTCUSDT" in syms
        assert "ETHUSDT" in syms
        assert "SOLUSDT" not in syms

    def test_exclude_symbols_removed_from_whitelist(self):
        host = _make_host(whitelist=["BTCUSDT", "ETHUSDT"], exclude=["ETHUSDT"])
        self._setup_exchange(host, ["BTCUSDT", "ETHUSDT"])
        syms = host.get_trading_symbols()
        assert "ETHUSDT" not in syms
        assert "BTCUSDT" in syms

    def test_no_whitelist_uses_top_n_by_volume(self):
        host = _make_host(whitelist=[])
        host._cfg.min_volume_rank = 2
        self._setup_exchange(host, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        syms = host.get_trading_symbols()
        # Should return top 2 by volume (highest quoteVolume = last in list)
        assert len(syms) <= 2

    def test_returns_empty_on_exception(self):
        host = _make_host()
        host.client.futures_exchange_info.side_effect = Exception("fail")
        syms = host.get_trading_symbols()
        assert syms == []


# ──────────────────────────────────────────────────────────────────────────────
#  bot_utils.get_git_commit / setup_logging
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
#  DataMixin._preload_max_notional
# ──────────────────────────────────────────────────────────────────────────────


def _make_bracket_response(symbol_caps: dict[str, float]) -> list[dict]:
    """Build a mock futures_leverage_bracket() response."""
    return [
        {
            "symbol": sym,
            "brackets": [
                {"bracket": 1, "initialLeverage": 50, "notionalCap": cap, "notionalFloor": 0},
            ],
        }
        for sym, cap in symbol_caps.items()
    ]


def _exchange_info_for(*symbols: str) -> dict:
    """Minimal exchange_info response for given symbols."""
    return {
        "symbols": [
            {
                "symbol": sym,
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
            for sym in symbols
        ]
    }


class TestPreloadMaxNotional:
    def test_bulk_preload_sets_max_notional(self):
        """Bulk preload populates max_notional for all cached symbols."""
        host = _make_host()
        host.client.futures_exchange_info.return_value = _exchange_info_for("ENJUSDT", "BTCUSDT")
        host.client.futures_leverage_bracket.return_value = _make_bracket_response(
            {"ENJUSDT": 5000.0, "BTCUSDT": 1_000_000.0}
        )

        host.get_symbol_info("ENJUSDT")  # triggers exchange_info + preload

        assert host._symbol_info_cache["ENJUSDT"]["max_notional"] == pytest.approx(5000.0)
        assert host._symbol_info_cache["BTCUSDT"]["max_notional"] == pytest.approx(1_000_000.0)
        # Only one bulk call, not per-symbol
        host.client.futures_leverage_bracket.assert_called_once_with()

    def test_preload_called_once_after_exchange_info(self):
        """Bracket bulk load is called exactly once, when exchange_info is first fetched."""
        host = _make_host()
        host.client.futures_exchange_info.return_value = _exchange_info_for("BTCUSDT")
        host.client.futures_leverage_bracket.return_value = _make_bracket_response({"BTCUSDT": 1_000_000.0})

        host.get_symbol_info("BTCUSDT")
        host.get_symbol_info("BTCUSDT")  # second call — cache hit, no reload

        host.client.futures_leverage_bracket.assert_called_once()

    def test_preload_failure_falls_back_to_lazy(self):
        """If bulk preload fails, per-symbol lazy-load is used as fallback."""
        host = _make_host()
        host.client.futures_exchange_info.return_value = _exchange_info_for("ENJUSDT")
        host.client.futures_leverage_bracket.side_effect = Exception("rate limit")

        # First call: exchange_info OK, bulk preload fails, max_notional not set yet
        host.get_symbol_info("ENJUSDT")
        assert "max_notional" not in host._symbol_info_cache["ENJUSDT"]

        # Second call (cache hit): lazy-load per-symbol
        host.client.futures_leverage_bracket.side_effect = None
        host.client.futures_leverage_bracket.return_value = _make_bracket_response({"ENJUSDT": 5000.0})
        info = host.get_symbol_info("ENJUSDT")
        assert info["max_notional"] == pytest.approx(5000.0)

    def test_symbol_absent_from_bracket_response_gets_none(self):
        """Symbol not in bracket response gets max_notional=None sentinel."""
        host = _make_host()
        host.client.futures_exchange_info.return_value = _exchange_info_for("NEWUSDT")
        host.client.futures_leverage_bracket.return_value = []  # empty response

        host.get_symbol_info("NEWUSDT")
        assert host._symbol_info_cache["NEWUSDT"]["max_notional"] is None


class TestGetGitCommit:
    def test_returns_dict_with_expected_keys(self):
        from quantbox.core.bot_utils import get_git_commit

        info = get_git_commit()
        assert "short" in info
        assert "full" in info
        assert "message" in info
        assert "date" in info

    def test_graceful_on_subprocess_failure(self):
        """Even if git is not available, returns a dict (not raises)."""
        from quantbox.core.bot_utils import get_git_commit

        with patch("subprocess.run", side_effect=Exception("no git")):
            info = get_git_commit()
        assert isinstance(info, dict)


class TestSetupLogging:
    def test_creates_log_file_and_returns_path(self, tmp_path, monkeypatch):
        """setup_logging creates log dir and returns a Path."""
        from quantbox.core.bot_utils import setup_logging
        import logging

        # Point log directory to tmp_path
        monkeypatch.chdir(tmp_path)
        path = setup_logging(strategy_name="vol_straddle", paper_mode=True)
        assert path.exists()
        assert "paper" in path.name
        # Cleanup handlers to avoid polluting other tests
        root = logging.getLogger()
        for h in root.handlers[:]:
            try:
                h.close()
                root.removeHandler(h)
            except Exception:
                pass

    def test_setup_logging_with_git_commit_header(self, tmp_path, monkeypatch):
        """With git_commit dict, header is written to log file."""
        from quantbox.core.bot_utils import setup_logging
        import logging

        monkeypatch.chdir(tmp_path)
        git = {"short": "abc1234", "full": "abc1234abcd", "message": "test commit", "date": "2024-01-01"}
        path = setup_logging(strategy_name="test", paper_mode=True, git_commit=git)
        content = path.read_text()
        assert "abc1234" in content
        root = logging.getLogger()
        for h in root.handlers[:]:
            try:
                h.close()
                root.removeHandler(h)
            except Exception:
                pass
