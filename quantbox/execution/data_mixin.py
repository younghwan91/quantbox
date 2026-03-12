"""DataMixin — market data fetching for RubberBandScalper.

Extracted from scalper.py to reduce file size and isolate concerns.
Methods in this mixin rely on instance attributes set by RubberBandScalper.__init__:
    self.client               — Binance Client
    self._cfg                 — strategy config (RubberBandConfig / VolStraddleConfig)
    self._symbol_info_cache   — dict[str, dict]
    self._exchange_info_fetched — bool
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantbox.core.protocol import ScalperProtocol

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("Scalper")


class DataMixin:
    """Market data fetching: symbol universe, symbol info, klines."""

    def get_trading_symbols(self: "ScalperProtocol") -> list[str]:
        """Get top-N USDT-M futures symbols by volume."""
        try:
            exchange_info = self.client.futures_exchange_info()
            usdt_symbols = [
                s["symbol"]
                for s in exchange_info["symbols"]
                if s["symbol"].endswith("USDT") and s["status"] == "TRADING"
            ]

            tickers = self.client.futures_ticker()
            volumes = {t["symbol"]: float(t["quoteVolume"]) for t in tickers if t["symbol"] in usdt_symbols}
            sorted_syms = sorted(volumes, key=lambda s: volumes.get(s, 0.0), reverse=True)

            # v11.1 fix: When whitelist is set, use whitelist directly
            # (only filter to actively trading symbols). Volume rank
            # is irrelevant when we have a curated, A/B-validated list.
            if self._cfg.symbol_whitelist:
                whitelist_set = set(self._cfg.symbol_whitelist)
                exclude_set = set(self._cfg.exclude_symbols)
                tradable = set(volumes.keys())  # actively trading symbols
                result = [s for s in sorted_syms if s in whitelist_set and s not in exclude_set]
                missing = whitelist_set - tradable
                if missing:
                    logger.warning(f"Whitelist symbols not tradable: {missing}")
                logger.info(f"Universe: {len(result)} symbols (whitelist={len(whitelist_set)})")
            else:
                # No whitelist: fall back to top-N by volume
                result = [s for s in sorted_syms[: self._cfg.min_volume_rank] if s not in self._cfg.exclude_symbols]
                logger.info(f"Universe: {len(result)} symbols")
            return result
        except Exception as e:
            logger.error(f"Symbol fetch failed: {e}")
            return []

    def _preload_max_notional(self: "ScalperProtocol") -> None:
        """Bulk-load max_notional for all symbols from Binance leverage brackets.

        Called once after the exchange_info bulk fetch populates _symbol_info_cache.
        Uses a single API call (no symbol arg) to retrieve all 800+ symbol brackets,
        then sets 'max_notional' on every cached symbol in one pass.

        The per-symbol lazy-load in get_symbol_info() remains as a fallback for
        symbols added after this preload or if this call fails.
        """
        try:
            resp = self.client.futures_leverage_bracket()
            bracket_map = {item["symbol"]: item["brackets"] for item in resp}
            loaded = 0
            for sym, cached in self._symbol_info_cache.items():
                if "max_notional" in cached:
                    continue  # already set (lazy-load beat us)
                sym_brackets = bracket_map.get(sym, [])
                if sym_brackets:
                    # Brackets sorted highest leverage first; bracket[0] gives
                    # the cap for the highest available leverage tier.
                    cached["max_notional"] = float(sym_brackets[0]["notionalCap"])
                    loaded += 1
                else:
                    cached["max_notional"] = None  # not on futures or unknown
            logger.info(
                f"Preloaded max_notional for {loaded}/{len(self._symbol_info_cache)} symbols "
                f"({len(bracket_map)} total in bracket response)"
            )
        except Exception as e:
            logger.warning(
                f"Bulk leverage bracket preload failed: {e} — "
                f"max_notional will be lazy-loaded per symbol on first order"
            )

    def get_symbol_info(self: "ScalperProtocol", symbol: str) -> dict[str, Any]:
        """Get tick_size, step_size, min_notional, max_notional for a symbol (cached)."""
        if symbol in self._symbol_info_cache:
            cached = self._symbol_info_cache[symbol]
            # Lazy-load max_notional from leverage brackets on first access.
            # Use None as sentinel to avoid repeated failed API calls.
            if "max_notional" not in cached:
                try:
                    brackets = self.client.futures_leverage_bracket(symbol=symbol)
                    if brackets:
                        sym_brackets = brackets[0].get("brackets", [])
                        if sym_brackets:
                            # Brackets are sorted highest leverage first.
                            # The first bracket's notionalCap is the max allowed
                            # at the highest leverage tier.
                            cached["max_notional"] = float(sym_brackets[0]["notionalCap"])
                        else:
                            cached["max_notional"] = None  # sentinel: tried, unavailable
                    else:
                        cached["max_notional"] = None
                except Exception as be:
                    logger.debug(f"[{symbol}] Leverage bracket fetch failed: {be}")
                    cached["max_notional"] = None  # sentinel: tried, failed
            return cached

        try:
            if not self._exchange_info_fetched:
                info = self.client.futures_exchange_info()
                for sym_info in info["symbols"]:
                    result: dict[str, Any] = {
                        "quantity_precision": sym_info["quantityPrecision"],
                        "price_precision": sym_info["pricePrecision"],
                    }
                    for f in sym_info["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            result["step_size"] = float(f["stepSize"])
                            result["min_qty"] = float(f["minQty"])
                        elif f["filterType"] == "PRICE_FILTER":
                            result["tick_size"] = float(f["tickSize"])
                        elif f["filterType"] == "MIN_NOTIONAL":
                            result["min_notional"] = float(f.get("notional", 5))
                    self._symbol_info_cache[sym_info["symbol"]] = result
                self._exchange_info_fetched = True
                # Bulk-load max_notional for all symbols now that the cache is
                # populated.  1 API call covers all 800+ symbols at once.
                self._preload_max_notional()
            return self._symbol_info_cache.get(symbol, {})
        except Exception as e:
            logger.error(f"Symbol info fetch failed for {symbol}: {e}")
            return {}

    def download_klines(
        self: "ScalperProtocol",
        symbol: str,
        n_bars: int = 200,
        interval: str = "15m",
        full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Download recent COMPLETED klines (incomplete bar dropped).

        Binance futures_klines always includes the currently-open (incomplete)
        candle as the last element. We request n_bars+1 and drop the last one
        to ensure backtest-live parity: signals are computed only on CLOSED bars.

        Args:
            symbol: Trading symbol.
            n_bars: Number of completed bars to return.
            interval: Candle interval ('15m', '4h', etc.).
            full: If True, returns (close, high, low, volume, taker_buy) tuple.

        Returns:
            Array of close prices (default), or (close, high, low, volume,
            taker_buy) tuple if full=True, or None on failure.
        """
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=n_bars + 1,  # +1 to compensate for dropping incomplete bar
            )
            if not klines:
                return None
            # Drop the last (currently-open/incomplete) candle for backtest parity.
            # The last kline from Binance is ALWAYS the live candle that hasn't
            # closed yet — using it would compute signals on partial data that
            # never existed in the backtest.
            klines = klines[:-1]
            if not klines:
                return None
            close = np.array([float(k[4]) for k in klines], dtype=np.float64)
            if full:
                high = np.array([float(k[2]) for k in klines], dtype=np.float64)
                low = np.array([float(k[3]) for k in klines], dtype=np.float64)
                volume = np.array([float(k[5]) for k in klines], dtype=np.float64)
                taker_buy = np.array([float(k[9]) for k in klines], dtype=np.float64)
                return close, high, low, volume, taker_buy
            return close
        except Exception as e:
            logger.error(f"Klines fetch failed for {symbol} ({interval}): {e}")
            return None
