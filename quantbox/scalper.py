#!/usr/bin/env python3
"""
QuantBox Rubber Band Scalper — Single-Asset Mean-Reversion Live Trader

Fades extreme short-term moves using the rubber band strategy.
Much simpler than the pair trader: no spread, no hedge ratio, no pair search.

Usage:
    python -m quantbox.scalper --paper         # Paper trading (simulation)
    python -m quantbox.scalper --paper --once   # Single cycle then exit
    python -m quantbox.scalper                  # LIVE trading (real orders!)

Architecture:
    1. Wait for 15m bar close
    2. Fetch latest klines for top-N symbols + BTC
    3. Compute rubber band signals (Numba-compiled)
    4. Manage existing positions (exit on target/stop/trailing/max_hold)
    5. Open new positions on fresh signals (LimitOrderChaser execution)
    6. Persist state atomically (crash recovery)

Safety features:
    - Max drawdown circuit breaker (10% session DD → halt all)
    - Daily loss limit (5% → stop new entries)
    - BTC crash gate (regime filter blocks entries in extreme trends)
    - Max concurrent positions (5)
    - Risk-sized positions (1% risk per trade)
    - Atomic state persistence for crash recovery
"""

import argparse
import json
import logging
import logging.handlers
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

from quantbox.core.limit_chaser import (
    ChaseConfig, ChaseFillStatus, ChaseResult, LimitOrderChaser,
)
from quantbox.core.settings import get_settings
from quantbox.core.terminal_ui import (
    Color,
    confirm_live_trading,
    format_cycle_header,
    format_entry,
    format_exit,
    format_scan_summary,
    format_session_summary,
    print_banner,
    print_startup_complete,
    print_system_info,
)
from quantbox.notifications.bot import (
    BotCommand, CommandType, DiscordBotRunner,
)
from quantbox.notifications.discord import DiscordNotifier
from quantbox.strategy.rubberband import (
    RubberBandConfig, RubberBandStrategy, calculate_position_size,
    compute_confidence_leverage,
)
from quantbox.strategy.elliott import ElliottConfig, ElliottStrategy

load_dotenv()

# ──────────────────────── Logging ──────────────────────────────────────

def _setup_logging() -> Path:
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'scalper_{ts}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                str(log_file), mode='a',
                maxBytes=50 * 1024 * 1024, backupCount=5,
            ),
        ],
    )
    for name in ('urllib3', 'requests', 'binance'):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


_log_path = _setup_logging()
logger = logging.getLogger('Scalper')


# ──────────────── Diagnostic JSONL Logger (v12.0) ─────────────────────
# Structured per-symbol-per-cycle data for parameter optimization.
# Writes to logs/diagnostics.jsonl — load with pd.read_json(lines=True).

class DiagnosticLogger:
    """Append-only JSONL writer for structured diagnostic data.

    Zero impact on strategy logic — pure observation layer.
    Rotates at 50MB to prevent disk fill.
    """

    def __init__(self, log_dir: Path = Path('logs')):
        self._path = log_dir / 'diagnostics.jsonl'
        self._fh: logging.handlers.RotatingFileHandler | None = None
        self._logger = logging.getLogger('Diagnostics')
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # Don't spam main log
        try:
            fh = logging.handlers.RotatingFileHandler(
                str(self._path), maxBytes=50 * 1024 * 1024, backupCount=3,
            )
            fh.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(fh)
            self._fh = fh
        except Exception:
            pass  # Degrade gracefully — diagnostics are optional

    def log_scan(self, cycle: int, timestamp: str, diags: list[dict]) -> None:
        """Write one JSONL record per cycle with all symbol diagnostics."""
        if self._fh is None:
            return
        record = {
            'type': 'scan',
            'cycle': cycle,
            'ts': timestamp,
            'symbols': diags,
        }
        self._logger.debug(json.dumps(record, separators=(',', ':')))

    def log_entry(self, cycle: int, timestamp: str, symbol: str,
                  signal: int, diag: dict, sizing: dict) -> None:
        """Write entry diagnostic when a position is opened."""
        if self._fh is None:
            return
        record = {
            'type': 'entry',
            'cycle': cycle,
            'ts': timestamp,
            'symbol': symbol,
            'signal': signal,
            'diag': diag,
            'sizing': sizing,
        }
        self._logger.debug(json.dumps(record, separators=(',', ':')))

    def log_exit(self, cycle: int, timestamp: str, trade: dict) -> None:
        """Write exit diagnostic when a position is closed."""
        if self._fh is None:
            return
        record = {
            'type': 'exit',
            'cycle': cycle,
            'ts': timestamp,
            'trade': trade,
        }
        self._logger.debug(json.dumps(record, separators=(',', ':')))


_diag_logger = DiagnosticLogger()

# ──────────────────────── State File ──────────────────────────────────

STATE_DIR = Path('data/state')
STATE_FILE = STATE_DIR / 'scalper_state.json'


def _save_scalper_state(state: dict, filepath: Path = STATE_FILE) -> None:
    """Atomic JSON write."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(filepath.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, str(filepath))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_scalper_state(filepath: Path = STATE_FILE) -> dict | None:
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"State load failed: {e}")
        return None


# ──────────────────────── Scalper Class ────────────────────────────────

class RubberBandScalper:
    """Live rubber band scalper for Binance USDT-M Futures.

    Attributes:
        paper_mode: If True, simulates orders without touching the exchange.
        config: RubberBandConfig with strategy parameters.
        strategy: Stateful RubberBandStrategy instance.
    """

    def __init__(
        self,
        paper_mode: bool = False,
        config: RubberBandConfig | None = None,
        strategy_name: str = 'rubberband',
        elliott_config: ElliottConfig | None = None,
    ):
        self.paper_mode = paper_mode
        self.config = config or RubberBandConfig()
        self.strategy_name = strategy_name

        if strategy_name == 'elliott':
            self.elliott_config = elliott_config or ElliottConfig()
            self.strategy = ElliottStrategy(self.elliott_config)
        else:
            self.strategy = RubberBandStrategy(self.config)

        # Binance client
        settings = get_settings()
        self.client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=settings.binance_testnet,
        )

        # Execution engine
        self._chaser = LimitOrderChaser(
            client=self.client,
            config=ChaseConfig(
                max_chase_ticks=10,
                chase_interval_seconds=1.0,
                timeout_seconds=30.0,
                fallback_to_market=True,
            ),
            paper_mode=self.paper_mode,
        )

        # Notifications
        self.discord = DiscordNotifier(webhook_url=settings.discord_webhook_url)

        # Discord Bot (interactive commands)
        self._bot_runner: DiscordBotRunner | None = None
        if settings.discord_bot_token and settings.discord_channel_id:
            self._bot_runner = DiscordBotRunner(
                token=settings.discord_bot_token,
                channel_id=settings.discord_channel_id,
            )

        # Paper trading state
        self._paper_balance = settings.paper_balance

        # Risk tracking
        self._initial_balance: float = 0.0
        self._high_water_mark: float = 0.0
        self._drawdown_halt: bool = False

        # Symbol info cache (avoid repeated exchange_info calls)
        self._symbol_info_cache: dict[str, dict] = {}
        self._exchange_info_fetched: bool = False

        # Trade history
        self._trade_history: list[dict] = []
        self._cycle_count: int = 0

        # Control flags
        self._stop_requested: bool = False
        self._paused: bool = False

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"RubberBandScalper initialized ({mode_str})")
        logger.info(
            f"Config: lookback={self.config.lookback}, "
            f"threshold={self.config.threshold}, "
            f"target={self.config.target_retrace}, "
            f"stop={self.config.stop_mult}x, "
            f"max_hold={self.config.max_hold}"
        )

    # ==================== Balance & Risk ====================

    def get_balance(self) -> float:
        """Get current USDT balance."""
        if self.paper_mode:
            return self._paper_balance
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b['asset'] == 'USDT':
                    return float(b['balance'])
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
        return 0.0

    def _check_risk_limits(self, balance: float) -> bool:
        """Check session drawdown limit.

        Returns True if trading should be blocked.
        """
        # Update high water mark
        if balance > self._high_water_mark:
            self._high_water_mark = balance

        # Session drawdown check
        if self._high_water_mark > 0:
            dd = (self._high_water_mark - balance) / self._high_water_mark
            if dd >= self.config.max_drawdown_pct:
                if not self._drawdown_halt:
                    logger.critical(
                        f"DRAWDOWN HALT: {dd*100:.1f}% DD "
                        f"(HWM=${self._high_water_mark:.2f}, "
                        f"now=${balance:.2f})"
                    )
                    self.discord.send(content=(
                        f"🚨 **DRAWDOWN HALT** — "
                        f"{dd*100:.1f}% drawdown triggered. "
                        f"Closing all positions."
                    ))
                    self._drawdown_halt = True
                return True

        return False

    # ==================== Market Data =====================

    def get_trading_symbols(self) -> list[str]:
        """Get top-N USDT-M futures symbols by volume."""
        try:
            exchange_info = self.client.futures_exchange_info()
            usdt_symbols = [
                s['symbol'] for s in exchange_info['symbols']
                if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
            ]

            tickers = self.client.futures_ticker()
            volumes = {
                t['symbol']: float(t['quoteVolume'])
                for t in tickers if t['symbol'] in usdt_symbols
            }
            sorted_syms = sorted(volumes, key=volumes.get, reverse=True)

            # v11.1 fix: When whitelist is set, use whitelist directly
            # (only filter to actively trading symbols). Volume rank
            # is irrelevant when we have a curated, A/B-validated list.
            if self.config.symbol_whitelist:
                whitelist_set = set(self.config.symbol_whitelist)
                exclude_set = set(self.config.exclude_symbols)
                tradable = set(volumes.keys())  # actively trading symbols
                result = [
                    s for s in sorted_syms
                    if s in whitelist_set and s not in exclude_set
                ]
                missing = whitelist_set - tradable
                if missing:
                    logger.warning(f"Whitelist symbols not tradable: {missing}")
                logger.info(f"Universe: {len(result)} symbols (whitelist={len(whitelist_set)})")
            else:
                # No whitelist: fall back to top-N by volume
                result = [
                    s for s in sorted_syms[:self.config.min_volume_rank]
                    if s not in self.config.exclude_symbols
                ]
                logger.info(f"Universe: {len(result)} symbols")
            return result
        except Exception as e:
            logger.error(f"Symbol fetch failed: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> dict:
        """Get tick_size, step_size, min_notional for a symbol (cached)."""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        try:
            if not self._exchange_info_fetched:
                info = self.client.futures_exchange_info()
                for sym_info in info['symbols']:
                    result = {
                        'quantity_precision': sym_info['quantityPrecision'],
                        'price_precision': sym_info['pricePrecision'],
                    }
                    for f in sym_info['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            result['step_size'] = float(f['stepSize'])
                            result['min_qty'] = float(f['minQty'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            result['tick_size'] = float(f['tickSize'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            result['min_notional'] = float(f.get('notional', 5))
                    self._symbol_info_cache[sym_info['symbol']] = result
                self._exchange_info_fetched = True
            return self._symbol_info_cache.get(symbol, {})
        except Exception as e:
            logger.error(f"Symbol info fetch failed for {symbol}: {e}")
            return {}

    def download_klines(
        self, symbol: str, n_bars: int = 200, interval: str = '15m',
        full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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

    # ==================== Order Execution ====================

    def _place_order(
        self, symbol: str, side: str, quantity: float,
        leverage: int = 1, reduce_only: bool = False,
    ) -> dict | None:
        """Place order via LimitOrderChaser.

        Args:
            symbol: Trading symbol.
            side: 'BUY' or 'SELL'.
            quantity: Order quantity.
            leverage: Leverage to set before order.
            reduce_only: If True, pass reduceOnly=True to exchange.
                Must be True when closing positions to avoid margin errors.

        Returns dict with avgPrice, executedQty on success, None on failure.
        """
        if self.paper_mode:
            # Simulate fill at current market price
            try:
                book = self.client.futures_orderbook_ticker(symbol=symbol)
                price = float(
                    book['askPrice'] if side == 'BUY' else book['bidPrice']
                )
            except Exception:
                return None
            return {
                'avgPrice': str(price),
                'executedQty': str(quantity),
                'status': 'FILLED',
                'makerFill': True,
            }

        try:
            # Set leverage — v11.1: verify it was applied correctly
            actual_leverage = leverage
            try:
                resp = self.client.futures_change_leverage(
                    symbol=symbol, leverage=leverage,
                )
                actual_leverage = int(resp.get('leverage', leverage))
                if actual_leverage != leverage:
                    logger.warning(
                        f"[{symbol}] Leverage mismatch: requested={leverage}x, "
                        f"actual={actual_leverage}x"
                    )
            except BinanceAPIException as e:
                if '-4028' in str(e):
                    # Leverage too high — try lower
                    try:
                        resp = self.client.futures_change_leverage(
                            symbol=symbol, leverage=min(leverage, 20),
                        )
                        actual_leverage = int(resp.get('leverage', min(leverage, 20)))
                    except Exception:
                        logger.error(
                            f"[{symbol}] Leverage fallback also failed — "
                            f"ABORTING order to prevent unknown leverage"
                        )
                        return None
                else:
                    logger.error(
                        f"[{symbol}] Leverage set failed: {e} — ABORTING order"
                    )
                    return None

            info = self.get_symbol_info(symbol)
            tick_size = info.get('tick_size', 0.01)
            price_precision = info.get('price_precision', 2)
            step_size = info.get('step_size', 0.001)
            qty_precision = info.get('quantity_precision', 3)

            result: ChaseResult = self._chaser.chase_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                tick_size=tick_size,
                price_precision=price_precision,
                step_size=step_size,
                quantity_precision=qty_precision,
                reduce_only=reduce_only,
            )

            if result.status == ChaseFillStatus.NO_POSITION:
                logger.warning(
                    f"[{symbol}] NO POSITION on exchange — "
                    f"reduceOnly rejected. Position already closed."
                )
                return {'status': 'NO_POSITION'}

            if result.status in (
                ChaseFillStatus.FILLED, ChaseFillStatus.FALLBACK_MARKET
            ):
                if result.avg_price <= 0:
                    logger.error(
                        f"[{symbol}] FILLED but avg_price=0 → treating as FAILED"
                    )
                    return None
                fill_type = "MAKER" if result.maker_fill else "TAKER"
                logger.info(
                    f"[{symbol}] {fill_type} {side} "
                    f"{result.filled_quantity:.6f} @ {result.avg_price:.4f} "
                    f"(chase={result.chase_count})"
                )
                return {
                    'avgPrice': str(result.avg_price),
                    'executedQty': str(result.filled_quantity),
                    'status': 'FILLED',
                    'makerFill': result.maker_fill,
                }
            else:
                logger.error(
                    f"[{symbol}] Order FAILED: {side} {quantity:.6f} "
                    f"(status={result.status.value})"
                )
                return None

        except Exception as e:
            logger.error(f"[{symbol}] Order error: {e}")
            return None

    def _round_qty(self, qty: float, symbol: str) -> float:
        """Round quantity to exchange step_size."""
        info = self.get_symbol_info(symbol)
        step = info.get('step_size', 0.001)
        prec = info.get('quantity_precision', 3)
        rounded = math.floor(qty / step) * step
        return round(rounded, prec)

    # ==================== Server-Side Orders (Algo Order API) ====================

    def _place_algo_order(
        self, symbol: str, side: str, order_type: str,
        quantity: float, trigger_price: float,
        limit_price: float | None = None,
        label: str = 'order',
    ) -> str | None:
        """Place a conditional order via Binance Algo Order API.

        Binance migrated STOP/STOP_MARKET/TAKE_PROFIT/TAKE_PROFIT_MARKET
        to POST /fapi/v1/algoOrder with algoType=CONDITIONAL.

        Args:
            symbol: Trading symbol.
            side: 'BUY' or 'SELL'.
            order_type: 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', etc.
            quantity: Position quantity.
            trigger_price: Price at which the order triggers.
            limit_price: Limit price for STOP/TAKE_PROFIT. None for MARKET types.
            label: Description for logging.

        Returns:
            algoId string, or None on failure.
        """
        if self.paper_mode:
            return f"paper_{label}_{int(time.time() * 1000)}"

        params: dict = {
            'algoType': 'CONDITIONAL',
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': str(quantity),
            'triggerPrice': str(trigger_price),
            'reduceOnly': 'true',
        }
        if limit_price is not None:
            params['price'] = str(limit_price)
            params['timeInForce'] = 'GTC'

        lp_str = f"${limit_price:.6f}" if limit_price is not None else "N/A"
        logger.debug(
            f"[{symbol}] Algo order request: {label} "
            f"side={side} type={order_type} qty={quantity:.6f} "
            f"trigger=${trigger_price:.6f} limit={lp_str}"
        )

        try:
            resp = self.client._request_futures_api(
                'post', 'algoOrder', signed=True, data=params,
            )
            algo_id = str(resp['algoId'])
            logger.info(
                f"[{symbol}] SERVER {label.upper()} placed via Algo API: "
                f"{side} {quantity:.6f} @ trigger=${trigger_price:.6f} "
                f"limit={lp_str} (algoId={algo_id})"
            )
            return algo_id
        except Exception as e:
            err_str = str(e)
            logger.error(
                f"[{symbol}] SERVER {label.upper()} FAILED: {e}. "
                f"params={params}"
            )
            # -2021: "Order would immediately trigger" — price already past
            # the stop level. Return sentinel so caller can force-close.
            if '-2021' in err_str:
                return 'WOULD_TRIGGER'
            return None

    def _place_server_stop_loss(
        self, symbol: str, signal: int, quantity: float,
        entry_price: float, move_size: float, stop_mult: float,
    ) -> str | None:
        """Place a server-side stop-loss order on Binance as a safety net.

        This ensures positions are protected even if the client process crashes.
        The stop price is set at the strategy's stop-loss level (move_size * stop_mult)
        with a 0.5% buffer beyond to account for slippage in fast markets.

        Uses Binance Algo Order API (POST /fapi/v1/algoOrder, algoType=CONDITIONAL).

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            quantity: Position quantity.
            entry_price: Fill price.
            move_size: Magnitude of the triggering move.
            stop_mult: Stop loss multiplier.

        Returns:
            algoId string, or None on failure (position still tracked).
        """
        if self.paper_mode:
            return f"paper_sl_{int(time.time() * 1000)}"

        # Stop price: entry ± (move_size * stop_mult) with 0.5% buffer
        stop_distance = move_size * stop_mult
        buffer = 0.005  # 0.5% extra beyond stop to ensure fill

        if signal == 1:  # Long: stop below entry
            stop_price = entry_price * (1.0 - stop_distance)
            limit_price = stop_price * (1.0 - buffer)  # Sell limit below stop
            close_side = 'SELL'
        else:  # Short: stop above entry
            stop_price = entry_price * (1.0 + stop_distance)
            limit_price = stop_price * (1.0 + buffer)  # Buy limit above stop
            close_side = 'BUY'

        info = self.get_symbol_info(symbol)
        tick_size = info.get('tick_size', 0.01)
        price_precision = info.get('price_precision', 2)

        # Round prices to tick size
        stop_price = round(
            round(stop_price / tick_size) * tick_size, price_precision
        )
        limit_price = round(
            round(limit_price / tick_size) * tick_size, price_precision
        )

        dist_pct = stop_distance * 100
        logger.info(
            f"[{symbol}] SL calc: entry=${entry_price:.4f} "
            f"move={move_size*100:.2f}% × stop_mult={stop_mult:.4f} "
            f"→ dist={dist_pct:.2f}% trigger=${stop_price:.4f} "
            f"limit=${limit_price:.4f} side={close_side}"
        )

        algo_id = self._place_algo_order(
            symbol=symbol, side=close_side, order_type='STOP',
            quantity=quantity, trigger_price=stop_price,
            limit_price=limit_price, label='stop-loss',
        )
        if algo_id == 'WOULD_TRIGGER':
            logger.error(
                f"[{symbol}] SL WOULD IMMEDIATELY TRIGGER — "
                f"price already past stop level ${stop_price:.4f}. "
                f"Returning sentinel for force-close."
            )
            self.discord.send(content=(
                f"\U0001f6a8 **SL WOULD TRIGGER** [{symbol}] — "
                f"Price blew past SL ${stop_price:.4f}. "
                f"Force-close required."
            ))
            return 'WOULD_TRIGGER'
        if algo_id is None:
            logger.error(
                f"[{symbol}] Position is UNPROTECTED — "
                f"client-side stop still active."
            )
            self.discord.send(content=(
                f"\u26a0\ufe0f **SERVER STOP FAILED** [{symbol}] — "
                f"Position unprotected on exchange side. "
                f"Client-side stop still active."
            ))
        return algo_id

    def _place_server_take_profit(
        self, symbol: str, signal: int, quantity: float,
        entry_price: float, move_size: float, target_retrace: float,
    ) -> str | None:
        """Place a server-side take-profit order on Binance.

        This captures intra-bar TP moves that would otherwise be missed
        by the 15m polling cycle. Combined with the server stop-loss,
        this creates an OCO-like bracket around the position.

        Uses Binance Algo Order API (POST /fapi/v1/algoOrder, algoType=CONDITIONAL).

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            quantity: Position quantity.
            entry_price: Fill price.
            move_size: Magnitude of the triggering move.
            target_retrace: Fraction of move to target (e.g. 0.382).

        Returns:
            algoId string, or None on failure.
        """
        if self.paper_mode:
            return f"paper_tp_{int(time.time() * 1000)}"

        # TP price: entry ± (move_size * target_retrace)
        target_distance = move_size * target_retrace
        buffer = 0.002  # 0.2% inside target to ensure fill

        if signal == 1:  # Long: TP above entry
            trigger_price = entry_price * (1.0 + target_distance)
            limit_price = trigger_price * (1.0 - buffer)
            close_side = 'SELL'
        else:  # Short: TP below entry
            trigger_price = entry_price * (1.0 - target_distance)
            limit_price = trigger_price * (1.0 + buffer)
            close_side = 'BUY'

        info = self.get_symbol_info(symbol)
        tick_size = info.get('tick_size', 0.01)
        price_precision = info.get('price_precision', 2)

        # Round prices to tick size
        trigger_price = round(
            round(trigger_price / tick_size) * tick_size, price_precision
        )
        limit_price = round(
            round(limit_price / tick_size) * tick_size, price_precision
        )

        dist_pct = target_distance * 100
        logger.info(
            f"[{symbol}] TP calc: entry=${entry_price:.4f} "
            f"move={move_size*100:.2f}% × retrace={target_retrace:.3f} "
            f"→ dist={dist_pct:.2f}% trigger=${trigger_price:.4f} "
            f"limit=${limit_price:.4f} side={close_side}"
        )

        algo_id = self._place_algo_order(
            symbol=symbol, side=close_side, order_type='TAKE_PROFIT',
            quantity=quantity, trigger_price=trigger_price,
            limit_price=limit_price, label='take-profit',
        )
        if algo_id is None:
            logger.warning(
                f"[{symbol}] Server TP failed — "
                f"will rely on client-side TP check."
            )
        return algo_id

    def _cancel_server_order(self, symbol: str, order_id: str,
                             label: str = 'order') -> None:
        """Cancel a server-side algo order (stop-loss or take-profit).

        Uses Binance Algo Order API (DELETE /fapi/v1/algoOrder).
        Must be called BEFORE placing a close order to avoid the server
        order triggering while we're trying to close.

        Args:
            symbol: Trading symbol.
            order_id: The algoId to cancel.
            label: Description for logging (e.g. 'stop-loss', 'take-profit').
        """
        if self.paper_mode or not order_id or order_id.startswith('paper_'):
            return

        try:
            resp = self.client._request_futures_api(
                'delete', 'algoOrder', signed=True,
                data={'algoId': int(order_id)},
            )
            logger.info(
                f"[{symbol}] Server {label} cancelled: "
                f"algoId={order_id} resp={resp}"
            )
        except BinanceAPIException as e:
            err_str = str(e)
            if 'Unknown' in err_str or '-2011' in err_str or '-25029' in err_str:
                # Already cancelled, filled, or expired — OK
                logger.debug(
                    f"[{symbol}] Server {label} already gone: "
                    f"algoId={order_id} ({e})"
                )
            else:
                logger.warning(
                    f"[{symbol}] Failed to cancel server {label}: "
                    f"algoId={order_id} error={e}"
                )
        except Exception as e:
            logger.warning(
                f"[{symbol}] Unexpected error cancelling server {label}: "
                f"algoId={order_id} error={e}"
            )

    def _cancel_server_stop_loss(self, symbol: str, order_id: str) -> None:
        """Cancel the server-side stop-loss."""
        self._cancel_server_order(symbol, order_id, 'stop-loss')

    def _cancel_server_take_profit(self, symbol: str, order_id: str) -> None:
        """Cancel the server-side take-profit."""
        self._cancel_server_order(symbol, order_id, 'take-profit')

    def _check_order_filled(self, symbol: str, order_id: str) -> dict | None:
        """Check if a server-side algo order has been filled (TRIGGERED).

        Uses Binance Algo Order API (GET /fapi/v1/algoOrder).
        Algo orders transition to algoStatus='TRIGGERED' when filled,
        then an actual order is placed. We check for TRIGGERED status.

        Returns a dict with 'avgPrice' and 'executedQty' if filled,
        None otherwise.
        """
        if self.paper_mode or not order_id or order_id.startswith('paper_'):
            return None

        try:
            resp = self.client._request_futures_api(
                'get', 'algoOrder', signed=True,
                data={'algoId': int(order_id)},
            )
            algo_status = resp.get('algoStatus', '')
            logger.debug(
                f"[{symbol}] Algo order check: algoId={order_id} "
                f"status={algo_status} type={resp.get('orderType', '?')}"
            )

            if algo_status == 'TRIGGERED':
                # When triggered, the algo order creates an actual order.
                # Extract fill info from the algo order response.
                actual_price = resp.get('actualPrice', '0')
                actual_qty = resp.get('quantity', '0')
                trigger_price = resp.get('triggerPrice', '0')

                logger.info(
                    f"[{symbol}] Algo order TRIGGERED: algoId={order_id} "
                    f"type={resp.get('orderType')} "
                    f"triggerPrice={trigger_price} "
                    f"actualPrice={actual_price} qty={actual_qty}"
                )

                # Use triggerPrice as fill proxy if actualPrice is zero
                fill_price = float(actual_price)
                if fill_price == 0:
                    fill_price = float(trigger_price)

                return {
                    'avgPrice': str(fill_price),
                    'executedQty': actual_qty,
                    'status': 'FILLED',
                    'algoId': order_id,
                }
        except Exception as e:
            logger.debug(
                f"[{symbol}] Algo order status check failed: "
                f"algoId={order_id} error={e}"
            )
        return None

    def _sync_server_orders(self) -> None:
        """Check if any server-side TP/SL orders filled between cycles.

        v11.2: Reconciles internal state with exchange reality.
        Called at the start of each cycle before processing positions.
        If a server TP/SL fired intra-bar, we record the trade and cancel
        the counterpart order.
        """
        if self.paper_mode:
            return

        for sym in list(self.strategy.active_positions):
            pos = self.strategy._positions.get(sym)
            if not pos:
                continue

            tp_oid = pos.get('server_tp_order_id')
            sl_oid = pos.get('server_sl_order_id')
            logger.debug(
                f"[{sym}] Sync check: sl_algoId={sl_oid} tp_algoId={tp_oid} "
                f"entry=${pos.get('entry_price', 0):.4f} "
                f"signal={'LONG' if pos.get('signal') == 1 else 'SHORT'}"
            )

            # Check TP order first (the happy path)
            if tp_oid:
                filled = self._check_order_filled(sym, tp_oid)
                if filled:
                    fill_price = float(filled['avgPrice'])
                    fill_qty = float(filled['executedQty'])
                    logger.info(
                        f"[{sym}] Server TP TRIGGERED intra-bar: "
                        f"fill=${fill_price:.4f} qty={fill_qty:.6f}"
                    )
                    # Cancel the counterpart SL
                    if sl_oid:
                        self._cancel_server_stop_loss(sym, sl_oid)
                    self._record_server_fill(
                        sym, fill_price, fill_qty, 'server_tp',
                    )
                    continue

            # Check SL order
            if sl_oid:
                filled = self._check_order_filled(sym, sl_oid)
                if filled:
                    fill_price = float(filled['avgPrice'])
                    fill_qty = float(filled['executedQty'])
                    logger.info(
                        f"[{sym}] Server SL TRIGGERED intra-bar: "
                        f"fill=${fill_price:.4f} qty={fill_qty:.6f}"
                    )
                    # Cancel the counterpart TP
                    if tp_oid:
                        self._cancel_server_take_profit(sym, tp_oid)
                    self._record_server_fill(
                        sym, fill_price, fill_qty, 'server_sl',
                    )
                    continue

            # ═══ v11.4.2 BRACKET REPAIR: Place missing SL/TP for restored positions ═══
            # Positions restored from state may have None algoIds (e.g. old session
            # failed to place them). Detect and repair to ensure server-side protection.
            missing_sl = sl_oid is None
            missing_tp = tp_oid is None

            if missing_sl or missing_tp:
                quantity = pos['notional'] / pos['entry_price']
                quantity = self._round_qty(quantity, sym)
                signal = pos['signal']
                entry_price = pos['entry_price']
                move_size = pos.get('move_size', 0.03)
                stop_mult = pos.get(
                    'entry_stop_mult', self.config.stop_mult,
                )

                if missing_sl:
                    new_sl = self._place_server_stop_loss(
                        symbol=sym, signal=signal, quantity=quantity,
                        entry_price=entry_price, move_size=move_size,
                        stop_mult=stop_mult,
                    )
                    if new_sl == 'WOULD_TRIGGER':
                        # ═══ FORCE-CLOSE: Price blew past SL level ═══
                        # The stop price is already behind the market.
                        # Close the position immediately at market with
                        # reduceOnly to prevent the "margin insufficient"
                        # error that killed us before.
                        logger.critical(
                            f"[{sym}] BRACKET REPAIR → FORCE CLOSE: "
                            f"Price past SL. Closing at market."
                        )
                        self.discord.send(content=(
                            f"\U0001f6a8 **FORCE CLOSE** [{sym}] — "
                            f"Price blew past SL. Emergency market close."
                        ))
                        closed = self._close_position(
                            sym, 'force_sl_breach', 0.0,
                        )
                        if closed:
                            logger.info(
                                f"[{sym}] FORCE CLOSE SUCCESS — "
                                f"position liquidated at market."
                            )
                        else:
                            logger.critical(
                                f"[{sym}] FORCE CLOSE FAILED — "
                                f"MANUAL INTERVENTION REQUIRED!"
                            )
                            self.discord.send(content=(
                                f"\U0001f6a8\U0001f6a8 **FORCE CLOSE FAILED** "
                                f"[{sym}] — MANUAL CLOSE REQUIRED NOW!"
                            ))
                        continue  # Skip TP repair — position gone
                    elif new_sl:
                        pos['server_sl_order_id'] = new_sl
                        logger.info(
                            f"[{sym}] BRACKET REPAIR: SL placed "
                            f"algoId={new_sl}"
                        )
                    else:
                        logger.error(
                            f"[{sym}] BRACKET REPAIR FAILED: "
                            f"Could not place SL — position UNPROTECTED"
                        )

                if missing_tp:
                    new_tp = self._place_server_take_profit(
                        symbol=sym, signal=signal, quantity=quantity,
                        entry_price=entry_price, move_size=move_size,
                        target_retrace=self.config.target_retrace,
                    )
                    if new_tp:
                        pos['server_tp_order_id'] = new_tp
                        logger.info(
                            f"[{sym}] BRACKET REPAIR: TP placed "
                            f"algoId={new_tp}"
                        )
                    else:
                        logger.warning(
                            f"[{sym}] BRACKET REPAIR: TP failed — "
                            f"client-side TP still active"
                        )

                if missing_sl or missing_tp:
                    self._save_state()
                    sl_id = pos.get('server_sl_order_id')
                    tp_id = pos.get('server_tp_order_id')
                    logger.info(
                        f"[{sym}] Bracket repair complete: "
                        f"SL={'OK' if sl_id else 'MISSING'} "
                        f"(algoId={sl_id}) "
                        f"TP={'OK' if tp_id else 'MISSING'} "
                        f"(algoId={tp_id})"
                    )

    def _record_server_fill(
        self, symbol: str, fill_price: float, fill_qty: float, reason: str,
    ) -> None:
        """Record a trade closed by a server-side order (TP or SL).

        Removes the position from internal tracking and logs the PnL.
        """
        pos = self.strategy.close_position(symbol)
        if not pos:
            return

        # PnL calculation
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        notional = fill_qty * fill_price
        pnl_usd = pnl_pct * notional
        cost = notional * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        trade = {
            'symbol': symbol,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': notional,
            'bars_held': pos['bars_held'],
            'reason': reason,
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)

        emoji = '🎯' if reason == 'server_tp' else '🛑'
        logger.info(
            f"[{symbol}] {emoji} {reason.upper()} FILLED: "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"held={pos['bars_held']} bars"
        )
        cum_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins_total = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        wr = wins_total / n_trades * 100 if n_trades > 0 else 0
        hold_time = pos['bars_held'] * 15

        self.discord.send(embed={
            'title': f"{emoji} {symbol} {reason.upper().replace('_', ' ')} FILLED",
            'color': 0x00C853 if net_pnl > 0 else 0xFF1744,
            'description': (
                'Server-side TP hit intra-bar!' if reason == 'server_tp'
                else 'Server-side SL triggered (safety net).'
            ),
            'fields': [
                {'name': 'Direction', 'value': 'LONG' if pos['signal'] == 1 else 'SHORT', 'inline': True},
                {'name': 'PnL', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': 'Net $', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': 'Entry', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': 'Exit', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': 'Hold Time', 'value': f'{hold_time}m ({pos["bars_held"]} bars)', 'inline': True},
                {'name': 'Cum PnL', 'value': f'${cum_pnl:+,.2f} ({n_trades}T, {wr:.0f}%WR)', 'inline': False},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })

    # ==================== Position Management ====================

    def _open_position(
        self, symbol: str, signal: int, close_prices: np.ndarray,
        balance: float,
    ) -> bool:
        """Open a new position.

        Args:
            symbol: Trading symbol.
            signal: +1 (long) or -1 (short).
            close_prices: Recent close prices.
            balance: Current balance.

        Returns:
            True if position was opened.
        """
        cfg = self.config
        entry_price = close_prices[-1]

        # Measure the move that triggered the signal
        lookback_price = close_prices[-cfg.lookback - 1]
        move_size = abs(entry_price / lookback_price - 1.0)

        # ── Elliott: 2% exploratory sizing (winning_strategy §4.2) ──
        if self.strategy_name == 'elliott':
            ecfg = getattr(self, 'elliott_config', ElliottConfig())
            margin = balance * ecfg.exploratory_pct
            leverage = ecfg.nominal_leverage
            notional = margin * leverage
            if notional < 10.0:
                logger.debug(f"[{symbol}] Elliott notional too small: ${notional:.2f}")
                return False
        else:
            # Position sizing — v10.0: dynamic capital allocation
            if cfg.dynamic_capital_enabled:
                n_active = self.strategy.n_active
                per_slot_capital = balance / max(1, n_active + 1)
                # Safety: cap at 50% of balance to prevent over-concentration
                per_slot_capital = min(per_slot_capital, balance * 0.5)
            else:
                per_slot_capital = balance / cfg.max_concurrent
            notional, leverage = calculate_position_size(
                capital=per_slot_capital,
                risk_per_trade=cfg.risk_per_trade,
                entry_price=entry_price,
                move_size=move_size,
                stop_mult=cfg.stop_mult,
                max_leverage=cfg.max_leverage,
            )

            # Dynamic leverage: override with move-confidence scaling
            if cfg.dynamic_leverage_enabled:
                if cfg.rr_leverage_enabled:
                    # v4.3: R:R-gated leverage — only deploy when R:R justifies it
                    from quantbox.strategy.rubberband import compute_rr_gated_leverage
                    conf_lev = compute_rr_gated_leverage(
                        move_size=move_size,
                        threshold=cfg.threshold,
                        target_retrace=cfg.target_retrace,
                        stop_mult=cfg.stop_mult,
                        cost_per_side=cfg.cost_per_side,
                        base_leverage=cfg.base_leverage,
                        leverage_scale=cfg.leverage_scale,
                        max_leverage=cfg.max_leverage,
                        min_rr=cfg.min_rr_for_leverage,
                    )
                else:
                    conf_lev = compute_confidence_leverage(
                        move_size=move_size,
                        threshold=cfg.threshold,
                        base_leverage=cfg.base_leverage,
                        leverage_scale=cfg.leverage_scale,
                        max_leverage=cfg.max_leverage,
                    )
                notional = per_slot_capital * conf_lev
                leverage = conf_lev

            # ═══ v10.2 PARITY FIX: Apply MTF leverage multiplier ═══
            if cfg.mtf_confirmation_enabled and cfg.mtf_leverage_mult > 1.0:
                if self.strategy.was_mtf_confirmed(symbol):
                    leverage *= cfg.mtf_leverage_mult
                    notional = per_slot_capital * leverage
                    logger.debug(
                        f"[{symbol}] MTF leverage boost: "
                        f"{leverage/cfg.mtf_leverage_mult:.2f}x → {leverage:.2f}x"
                    )

        if notional < 10.0:  # Min notional $10
            logger.debug(f"[{symbol}] Notional too small: ${notional:.2f}")
            return False

        quantity = notional / entry_price
        quantity = self._round_qty(quantity, symbol)
        if quantity <= 0:
            return False

        # Check min notional
        info = self.get_symbol_info(symbol)
        min_notional = info.get('min_notional', 5.0)
        if quantity * entry_price < min_notional:
            logger.debug(f"[{symbol}] Below min notional ${min_notional}")
            return False

        side = 'BUY' if signal > 0 else 'SELL'
        lev_int = max(1, min(math.ceil(leverage), 20))

        result = self._place_order(symbol, side, quantity, leverage=lev_int)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])

        # Compute effective stop (HTF agree parity)
        eff_stop = self.strategy.get_effective_stop_mult(close_prices)

        notional = fill_qty * fill_price
        direction = 'LONG' if signal > 0 else 'SHORT'
        logger.info(
            f"[{symbol}] OPENED {direction} qty={fill_qty:.6f} "
            f"@ ${fill_price:.4f} (notional=${notional:.2f}, lev={lev_int}x, "
            f"move={move_size*100:.2f}%, eff_stop={eff_stop:.4f})"
        )

        # ═══ Place server-side bracket orders via Algo API ═══
        sl_order_id = self._place_server_stop_loss(
            symbol=symbol,
            signal=signal,
            quantity=fill_qty,
            entry_price=fill_price,
            move_size=move_size,
            stop_mult=eff_stop,
        )

        # Captures intra-bar TP hits that 15m polling would miss.
        tp_order_id = self._place_server_take_profit(
            symbol=symbol,
            signal=signal,
            quantity=fill_qty,
            entry_price=fill_price,
            move_size=move_size,
            target_retrace=cfg.target_retrace,
        )

        sl_status = "OK" if sl_order_id else "FAILED"
        tp_status = "OK" if tp_order_id else "FAILED"
        logger.info(
            f"[{symbol}] Server bracket: SL={sl_status} "
            f"(algoId={sl_order_id}) TP={tp_status} "
            f"(algoId={tp_order_id})"
        )

        # Record in strategy
        self.strategy.open_position(
            symbol=symbol,
            signal=signal,
            entry_price=fill_price,
            notional=fill_qty * fill_price,
            leverage=lev_int,
            move_size=move_size,
            entry_stop_mult=eff_stop,
            server_sl_order_id=sl_order_id,
            server_tp_order_id=tp_order_id,
        )

        # Paper balance adjustment
        if self.paper_mode:
            cost = fill_qty * fill_price * cfg.cost_per_side
            self._paper_balance -= cost

        _side_str = 'LONG' if signal == 1 else 'SHORT'
        print(format_entry(
            symbol=symbol,
            side=_side_str,
            price=fill_price,
            leverage=lev_int,
            size_usd=fill_qty * fill_price,
        ))
        logger.info(
            f"[{symbol}] OPENED {_side_str} "
            f"qty={fill_qty:.6f} @ ${fill_price:.4f} "
            f"(notional=${fill_qty * fill_price:.2f}, "
            f"lev={lev_int}x, move={move_size*100:.2f}%)"
        )

        # v12.0: Entry diagnostic to JSONL
        if hasattr(self.strategy, 'diagnose_bar'):
            entry_diag = self.strategy.diagnose_bar(symbol, close_prices)
            _diag_logger.log_entry(
                self._cycle_count,
                datetime.now(timezone.utc).isoformat(),
                symbol, signal, entry_diag,
                {'notional': round(fill_qty * fill_price, 2),
                 'leverage': lev_int, 'move_size': round(move_size, 6),
                 'fill_price': fill_price, 'tp_pct': round(cfg.target_retrace * 100, 2),
                 'sl_mult': round(eff_stop, 4)},
            )

        # Compute TP/SL prices for notification
        tp_distance = move_size * cfg.target_retrace
        sl_distance = move_size * eff_stop
        if signal == 1:
            tp_price = fill_price * (1.0 + tp_distance)
            sl_price = fill_price * (1.0 - sl_distance)
        else:
            tp_price = fill_price * (1.0 - tp_distance)
            sl_price = fill_price * (1.0 + sl_distance)

        balance_after = self.get_balance()
        n_active = self.strategy.n_active
        self.discord.send(embed={
            'title': f"{'🟢' if signal == 1 else '🔴'} {symbol} "
                     f"{'LONG' if signal == 1 else 'SHORT'}",
            'color': 0x00C853 if signal == 1 else 0xFF1744,
            'fields': [
                {'name': 'Entry', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': 'Size', 'value': f'${fill_qty * fill_price:.2f}',
                 'inline': True},
                {'name': 'Leverage', 'value': f'{lev_int}x', 'inline': True},
                {'name': 'Move', 'value': f'{move_size*100:.2f}%', 'inline': True},
                {'name': 'TP', 'value': f'${tp_price:.4f}', 'inline': True},
                {'name': 'SL', 'value': f'${sl_price:.4f}', 'inline': True},
                {'name': 'Slots', 'value': f'{n_active}/{cfg.max_concurrent}', 'inline': True},
                {'name': 'Balance', 'value': f'${balance_after:,.2f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    def _partial_close_position(
        self, symbol: str, current_price: float,
    ) -> bool:
        """Partially close 50% of a position (v4.3 capital preservation).

        Closes half the position to lock in profit. The remaining half
        continues with a break-even stop. Does NOT remove the position
        from the strategy — it stays tracked with partial_closed=True.

        Returns True if partial close succeeded.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        # v11.1: Cancel old server SL (quantity will change)
        sl_oid = pos.get('server_sl_order_id')
        if sl_oid:
            self._cancel_server_stop_loss(symbol, sl_oid)

        # v11.2: Cancel old server TP (quantity will change)
        tp_oid = pos.get('server_tp_order_id')
        if tp_oid:
            self._cancel_server_take_profit(symbol, tp_oid)

        fraction = self.config.partial_exit_fraction
        full_quantity = pos['notional'] / pos['entry_price']
        close_qty = self._round_qty(full_quantity * fraction, symbol)
        if close_qty <= 0:
            return False

        # Reverse the entry side to close
        side = 'SELL' if pos['signal'] == 1 else 'BUY'

        result = self._place_order(symbol, side, close_qty, reduce_only=True)
        if result is None:
            # Failed: revert partial_closed flag
            pos['partial_closed'] = False
            return False

        fill_price = float(result['avgPrice'])

        # PnL on the partial close
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        partial_notional = close_qty * fill_price
        pnl_usd = pnl_pct * partial_notional
        cost = partial_notional * self.config.cost_per_side
        net_pnl = pnl_usd - cost

        # Update position: reduce notional by closed amount
        pos['notional'] -= partial_notional

        # v11.1: Place new server SL for remaining quantity
        remaining_qty = self._round_qty(
            pos['notional'] / pos['entry_price'], symbol,
        )
        if remaining_qty > 0 and not self.paper_mode:
            new_sl_oid = self._place_server_stop_loss(
                symbol=symbol,
                signal=pos['signal'],
                quantity=remaining_qty,
                entry_price=pos['entry_price'],
                move_size=pos['move_size'],
                stop_mult=0.0001,  # Break-even stop (tiny distance)
            )
            pos['server_sl_order_id'] = new_sl_oid

            # v11.2: Place new TP for remaining half at same target
            new_tp_oid = self._place_server_take_profit(
                symbol=symbol,
                signal=pos['signal'],
                quantity=remaining_qty,
                entry_price=pos['entry_price'],
                move_size=pos['move_size'],
                target_retrace=self.config.target_retrace,
            )
            pos['server_tp_order_id'] = new_tp_oid

        # Paper mode
        if self.paper_mode:
            self._paper_balance += pnl_usd - cost

        # Trade record (partial)
        trade = {
            'symbol': symbol,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': partial_notional,
            'bars_held': pos['bars_held'],
            'reason': 'partial_close',
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)

        logger.info(
            f"[{symbol}] PARTIAL CLOSE ({fraction*100:.0f}%): "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"→ remaining ${pos['notional']:.2f}, BE stop active"
        )
        self.discord.send(embed={
            'title': f"🔒 {symbol} PARTIAL CLOSE ({fraction*100:.0f}%)",
            'color': 0xFFC107,
            'description': 'Capital secured. Remaining rides with break-even stop.',
            'fields': [
                {'name': 'Direction', 'value': 'LONG' if pos['signal'] == 1 else 'SHORT', 'inline': True},
                {'name': 'PnL', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': 'Locked $', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': 'Entry', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': 'Exit', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': 'Remaining', 'value': f'${pos["notional"]:.2f}', 'inline': True},
                {'name': 'Stop', 'value': 'Break-even', 'inline': True},
                {'name': 'Bars Held', 'value': str(pos['bars_held']), 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    # ── Elliott: Add-to-position methods ──

    def _add_to_position(
        self, symbol: str, current_price: float, balance: float,
    ) -> bool:
        """Add primary capital (8%) to an exploratory Elliott position.

        Called when update_position returns 'confirm_entry' — the
        divergence + MACD/Stochastic confluence has been validated.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            balance: Current account balance.

        Returns:
            True if additional capital was deployed.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        ecfg = getattr(self, 'elliott_config', ElliottConfig())
        add_margin = balance * ecfg.primary_pct
        add_notional = add_margin * ecfg.nominal_leverage
        if add_notional < 10.0:
            return False

        add_qty = add_notional / current_price
        add_qty = self._round_qty(add_qty, symbol)
        if add_qty <= 0:
            return False

        side = 'BUY' if pos['signal'] == 1 else 'SELL'
        result = self._place_order(symbol, side, add_qty)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])

        # Update position with blended entry
        old_notional = pos['notional']
        new_notional = fill_qty * fill_price
        total_notional = old_notional + new_notional
        blended_entry = (
            (pos['entry_price'] * old_notional + fill_price * new_notional)
            / total_notional
        )
        pos['entry_price'] = blended_entry
        pos['notional'] = total_notional

        logger.info(
            f"[{symbol}] CONFIRMED: Added 8% (${new_notional:.2f}) → "
            f"total ${total_notional:.2f} @ blended ${blended_entry:.4f}"
        )
        self.discord.send(embed={
            'title': f"✅ {symbol} CONFIRMED — Primary Entry",
            'color': 0x2196F3,
            'description': 'Divergence validated. Upgraded to 10% margin.',
            'fields': [
                {'name': 'Added', 'value': f'${new_notional:.2f}', 'inline': True},
                {'name': 'Total', 'value': f'${total_notional:.2f}', 'inline': True},
                {'name': 'Blended Entry', 'value': f'${blended_entry:.4f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    def _pyramid_position(
        self, symbol: str, current_price: float, balance: float,
    ) -> bool:
        """Pyramid into a winning Elliott position using unrealized profits.

        Called when update_position returns 'pyramid' — Wave 3/C momentum
        confirmed with EMA alignment. Uses unrealized profit as margin.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            balance: Current account balance.

        Returns:
            True if pyramid order was filled.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        # Unrealized PnL
        if pos['signal'] == 1:
            gain_pct = (current_price - pos['entry_price']) / pos['entry_price']
        else:
            gain_pct = (pos['entry_price'] - current_price) / pos['entry_price']

        unrealized = gain_pct * pos['notional']
        if unrealized < 20.0:  # Min $20 profit for pyramid
            return False

        ecfg = getattr(self, 'elliott_config', ElliottConfig())
        # Use unrealized profit as margin for pyramid
        pyramid_notional = min(unrealized * ecfg.nominal_leverage, pos['notional'] * 0.5)
        pyramid_qty = pyramid_notional / current_price
        pyramid_qty = self._round_qty(pyramid_qty, symbol)
        if pyramid_qty <= 0:
            return False

        side = 'BUY' if pos['signal'] == 1 else 'SELL'
        result = self._place_order(symbol, side, pyramid_qty)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])
        new_notional = fill_qty * fill_price

        # Update position
        old_notional = pos['notional']
        total_notional = old_notional + new_notional
        blended = (
            (pos['entry_price'] * old_notional + fill_price * new_notional)
            / total_notional
        )
        pos['entry_price'] = blended
        pos['notional'] = total_notional

        logger.info(
            f"[{symbol}] PYRAMID: Added ${new_notional:.2f} from "
            f"${unrealized:.2f} unrealized → total ${total_notional:.2f}"
        )
        self.discord.send(embed={
            'title': f"📈 {symbol} PYRAMID — Wave 3/C Momentum",
            'color': 0x4CAF50,
            'description': 'Pyramiding with unrealized profits. Stop trailed to BE.',
            'fields': [
                {'name': 'Added', 'value': f'${new_notional:.2f}', 'inline': True},
                {'name': 'Total', 'value': f'${total_notional:.2f}', 'inline': True},
                {'name': 'Unrealized', 'value': f'${unrealized:.2f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    def _close_position(
        self, symbol: str, reason: str, current_price: float,
    ) -> bool:
        """Close an existing position.

        Returns True if closed successfully.  Also returns True when the
        exchange reports NO_POSITION (-4118/-2022) because the position is
        already gone — we just clean up our internal state.
        """
        pos = self.strategy._positions.get(symbol)
        if pos is None:
            return False

        quantity = pos['notional'] / pos['entry_price']
        quantity = self._round_qty(quantity, symbol)
        if quantity <= 0:
            return False

        # v11.1: Cancel server-side stop-loss BEFORE closing
        sl_oid = pos.get('server_sl_order_id')
        if sl_oid:
            self._cancel_server_stop_loss(symbol, sl_oid)

        # v11.2: Cancel server-side take-profit BEFORE closing
        tp_oid = pos.get('server_tp_order_id')
        if tp_oid:
            self._cancel_server_take_profit(symbol, tp_oid)

        # Reverse the entry side
        side = 'SELL' if pos['signal'] == 1 else 'BUY'

        result = self._place_order(symbol, side, quantity, reduce_only=True)

        # ── NO_POSITION: exchange says position already gone ──
        if isinstance(result, dict) and result.get('status') == 'NO_POSITION':
            logger.warning(
                f"[{symbol}] Position already closed on exchange — "
                f"cleaning up internal state (reason={reason})"
            )
            self.discord.send(content=(
                f"\u26a0\ufe0f **NO POSITION** [{symbol}] — "
                f"Position gone on exchange. State cleanup."
            ))
            # Remove from internal tracking
            self.strategy.close_position(symbol)
            self._save_state()
            return True

        if result is None:
            # Failed to close — position stays in state for retry
            # SL/TP already cancelled, so re-place them
            new_sl_oid = self._place_server_stop_loss(
                symbol=symbol,
                signal=pos['signal'],
                quantity=quantity,
                entry_price=pos['entry_price'],
                move_size=pos['move_size'],
                stop_mult=pos.get('entry_stop_mult', self.config.stop_mult),
            )
            new_tp_oid = self._place_server_take_profit(
                symbol=symbol,
                signal=pos['signal'],
                quantity=quantity,
                entry_price=pos['entry_price'],
                move_size=pos['move_size'],
                target_retrace=self.config.target_retrace,
            )
            # Update existing position's server order IDs
            # Guard against sentinel values being stored as order IDs
            if new_sl_oid and new_sl_oid != 'WOULD_TRIGGER':
                pos['server_sl_order_id'] = new_sl_oid
            else:
                pos['server_sl_order_id'] = None
            pos['server_tp_order_id'] = new_tp_oid
            self._save_state()
            return False

        # ── Successful close — NOW remove from state ──
        pos = self.strategy.close_position(symbol)
        if pos is None:
            return False

        fill_price = float(result['avgPrice'])

        # PnL calculation
        if pos['signal'] == 1:
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']

        pnl_usd = pnl_pct * pos['notional']
        cost = pos['notional'] * self.config.round_trip_cost
        net_pnl = pnl_usd - cost

        # Paper balance
        if self.paper_mode:
            self._paper_balance += pnl_usd
            close_cost = quantity * fill_price * self.config.cost_per_side
            self._paper_balance -= close_cost

        # Trade record
        trade = {
            'symbol': symbol,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': net_pnl,
            'notional': pos['notional'],
            'bars_held': pos['bars_held'],
            'reason': reason,
            'time': datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade)

        # v12.0: Exit diagnostic to JSONL
        _diag_logger.log_exit(self._cycle_count,
                              datetime.now(timezone.utc).isoformat(), trade)

        _exit_side = 'LONG' if pos['signal'] == 1 else 'SHORT'
        print(format_exit(
            symbol=symbol,
            side=_exit_side,
            entry_price=pos['entry_price'],
            exit_price=fill_price,
            pnl_usd=net_pnl,
            pnl_pct=pnl_pct * 100,
            reason=reason,
            bars_held=pos['bars_held'],
        ))
        logger.info(
            f"[{symbol}] CLOSED ({reason}): {_exit_side} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"held={pos['bars_held']} bars"
        )
        # Session stats for notification
        cum_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins_total = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        wr = wins_total / n_trades * 100 if n_trades > 0 else 0
        balance_after = self.get_balance()
        hold_time = pos['bars_held'] * 15  # minutes

        self.discord.send(embed={
            'title': f"{'✅' if net_pnl > 0 else '❌'} {symbol} CLOSED — {reason.upper().replace('_', ' ')}",
            'color': 0x00C853 if net_pnl > 0 else 0xFF1744,
            'fields': [
                {'name': 'Direction', 'value': 'LONG' if pos['signal'] == 1 else 'SHORT', 'inline': True},
                {'name': 'PnL', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': 'Net $', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': 'Entry', 'value': f'${pos["entry_price"]:.4f}', 'inline': True},
                {'name': 'Exit', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': 'Hold Time', 'value': f'{hold_time}m ({pos["bars_held"]} bars)', 'inline': True},
                {'name': 'Cum PnL', 'value': f'${cum_pnl:+,.2f} ({n_trades} trades)', 'inline': True},
                {'name': 'Win Rate', 'value': f'{wr:.0f}%', 'inline': True},
                {'name': 'Balance', 'value': f'${balance_after:,.2f}', 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })
        return True

    # ==================== Main Cycle ====================

    def _cycle(self, symbols: list[str]) -> None:
        """Process one 15m bar across all symbols.

        1. Process bot commands (pause/stop/close)
        2. Get BTC klines for regime filter
        3. Manage existing positions
        4. Check for new entries (skip if paused)
        5. Save state
        """
        self._cycle_count += 1
        balance = self.get_balance()

        # 0. Process interactive bot commands
        self._process_bot_commands()
        if self._stop_requested:
            return

        header = format_cycle_header(
            cycle=self._cycle_count,
            balance=balance,
            n_active=self.strategy.n_active,
            max_slots=self.config.max_concurrent,
        )
        print(header)
        logger.info(
            f"Cycle #{self._cycle_count} | "
            f"Balance=${balance:.2f} | "
            f"Positions={self.strategy.n_active}/{self.config.max_concurrent}"
        )

        # 1. Risk check
        risk_blocked = self._check_risk_limits(balance)

        # 2. Close positions on drawdown halt
        if self._drawdown_halt:
            for sym in list(self.strategy.active_positions):
                close = self.download_klines(sym, 2)
                price = close[-1] if close is not None else 0
                self._close_position(sym, 'drawdown_halt', price)
            self._save_state()
            return

        # 3. Fetch BTC klines for regime filter
        btc_close = self.download_klines('BTCUSDT', 200)

        # 3b. v7.0: Fetch BTC 4h klines for MTF / Elliott macro bias
        btc_close_4h = None
        if self.strategy_name == 'elliott':
            # Elliott needs 600+ bars of BTC 4h for 200/500 EMA macro bias
            btc_close_4h = self.download_klines('BTCUSDT', 600, interval='4h')
            if btc_close_4h is not None:
                self.strategy.update_btc_data(btc_close_4h)
        elif self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
            btc_close_4h = self.download_klines('BTCUSDT', 100, interval='4h')

        # 3c. v11.2: Sync server-side TP/SL orders (check if filled intra-bar)
        self._sync_server_orders()

        # 4. Process existing positions (always, even when paused)
        for sym in list(self.strategy.active_positions):
            close = self.download_klines(sym, 200)
            if close is None:
                continue

            current_price = close[-1]
            exit_reason = self.strategy.update_position(sym, current_price)
            if exit_reason:
                if exit_reason == 'partial_close':
                    self._partial_close_position(sym, current_price)
                elif exit_reason == 'confirm_entry':
                    # Elliott: upgrade 2% exploratory → add 8% primary
                    self._add_to_position(sym, current_price, balance)
                elif exit_reason == 'pyramid':
                    # Elliott: pyramid using unrealized profits
                    self._pyramid_position(sym, current_price, balance)
                elif exit_reason == 'cancel_exploration':
                    # Elliott: confirmation timed out → close pilot
                    self._close_position(sym, exit_reason, current_price)
                else:
                    self._close_position(sym, exit_reason, current_price)

        # 5. Look for new entries (skip if risk blocked or paused)
        if self._paused:
            logger.info("⏸️ Paused — skipping new entries")
        elif not risk_blocked:
            available_slots = (
                self.config.max_concurrent - self.strategy.n_active
            )
            if available_slots > 0:
                self._scan_entries(symbols, btc_close, balance, available_slots,
                                   btc_close_4h=btc_close_4h)

        # 6. Save state
        self._save_state()

        # 7. Periodic summary
        if self._cycle_count % 4 == 0:  # Every hour
            self._log_summary(balance)

    def _scan_entries(
        self,
        symbols: list[str],
        btc_close: np.ndarray | None,
        balance: float,
        max_entries: int,
        btc_close_4h: np.ndarray | None = None,
    ) -> None:
        """Scan all symbols for entry signals.

        v7.0: Downloads volume/taker data for confidence scoring,
        and 4h data for MTF confirmation filter.
        Prioritizes symbols with the largest moves (strongest alpha signal).
        """
        candidates: list[tuple[str, int, float, np.ndarray]] = []
        moves_seen: list[tuple[str, float]] = []  # diagnostic: (symbol, abs_move)
        scan_diags: list[dict] = []  # v12.0: per-symbol diagnostics

        for sym in symbols:
            # Skip symbols we already have positions in
            if self.strategy.has_position(sym):
                continue

            # v9.0: Download full klines (close + high + low + volume + taker)
            result = self.download_klines(sym, 200, full=True)
            if result is None:
                continue
            close, high, low, volume, taker_buy = result
            if len(close) < self.config.lookback + 1:
                continue

            # Wire up volume/taker/high/low data for v9 confluence scoring
            self.strategy.update_market_data(sym, volume, taker_buy, high, low)

            # v7.0: Download 4h data for MTF confirmation / Elliott wave detection
            if self.strategy_name == 'elliott':
                # Elliott needs 600+ bars of 4h OHLC for 500 EMA + ZigZag
                result_4h = self.download_klines(sym, 600, interval='4h', full=True)
                if result_4h is not None:
                    c4h, h4h, l4h, _, _ = result_4h
                    self.strategy.update_htf_data(sym, c4h, h4h, l4h)
            elif self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
                close_4h = self.download_klines(sym, 100, interval='4h')
                if close_4h is not None:
                    self.strategy.update_htf_data(sym, close_4h)

            signal = self.strategy.on_bar(sym, close, btc_close)

            # v12.0: Collect full diagnostic context (read-only, after on_bar)
            if self.strategy_name != 'elliott' and hasattr(self.strategy, 'diagnose_bar'):
                diag = self.strategy.diagnose_bar(sym, close, btc_close)
                diag['signal_actual'] = signal  # cross-verify with on_bar
                scan_diags.append(diag)

            # Track 8-bar move for diagnostics
            abs_move = abs(close[-1] / close[-self.config.lookback - 1] - 1.0)
            moves_seen.append((sym, abs_move))

            if signal != 0:
                # Measure move strength for prioritization
                move = abs(
                    close[-1] / close[-self.config.lookback - 1] - 1.0
                )
                candidates.append((sym, signal, move, close))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

        # ── Diagnostic: log scan summary every cycle ──
        top_movers = sorted(moves_seen, key=lambda x: x[1], reverse=True)[:3]
        print(format_scan_summary(len(candidates), top_movers, self.config.threshold))
        if candidates:
            cand_strs = [f"{s}({m*100:.1f}%)" for s, _, m, _ in candidates[:3]]
            logger.info(f"🎯 {len(candidates)} candidates: {', '.join(cand_strs)}")
        else:
            if moves_seen:
                top3 = [f"{s}({m*100:.2f}%)" for s, m in top_movers]
                logger.info(
                    f"No candidates — top moves: {', '.join(top3)} "
                    f"(need >{self.config.threshold*100:.1f}%)"
                )

        # v12.0: Write structured diagnostics to JSONL
        if scan_diags:
            ts = datetime.now(timezone.utc).isoformat()
            _diag_logger.log_scan(self._cycle_count, ts, scan_diags)

        entries_made = 0
        for sym, signal, move, close in candidates:
            if entries_made >= max_entries:
                break
            if self._open_position(sym, signal, close, balance):
                entries_made += 1

        if entries_made > 0:
            logger.info(
                f"Opened {entries_made} new positions "
                f"(from {len(candidates)} candidates)"
            )

    def _log_summary(self, balance: float) -> None:
        """Log hourly summary."""
        positions = self.strategy.active_positions
        n_trades = len(self._trade_history)
        if n_trades == 0:
            return

        wins = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        total_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        wr = wins / n_trades * 100 if n_trades > 0 else 0

        logger.info(
            f"[Summary] Trades={n_trades}, WR={wr:.0f}%, "
            f"PnL=${total_pnl:+.2f}, Balance=${balance:.2f}, "
            f"Active={len(positions)}"
        )

        # Active position details
        for sym, pos in positions.items():
            logger.info(
                f"  [{sym}] {'L' if pos['signal'] == 1 else 'S'} "
                f"entry=${pos['entry_price']:.4f} "
                f"bars={pos['bars_held']} "
                f"best_pnl={pos['best_pnl']*100:.2f}%"
            )

    # ==================== Discord Bot Commands ====================

    def _process_bot_commands(self) -> None:
        """Poll the bot command queue and handle all pending commands.

        Called at the start of each cycle. Non-blocking.
        """
        if self._bot_runner is None:
            return

        for cmd in self._bot_runner.get_all_commands():
            try:
                result = self._handle_command(cmd)
                cmd.set_response(result)
            except Exception as e:
                logger.error(f"Bot command error: {e}")
                cmd.set_response(f"❌ Internal error: {e}")

    def _handle_command(self, cmd: BotCommand) -> str:
        """Dispatch a BotCommand to the appropriate handler.

        Args:
            cmd: The parsed bot command.

        Returns:
            Response text to send back to Discord.
        """
        handlers = {
            CommandType.STATUS: lambda: self._cmd_status(),
            CommandType.CLOSE_ALL: lambda: self._cmd_close_all(),
            CommandType.CLOSE_SLOT: lambda: self._cmd_close_slot(
                cmd.args.get('slot_id', -1) if cmd.args else -1,
            ),
            CommandType.PAUSE: lambda: self._cmd_pause(),
            CommandType.RESUME: lambda: self._cmd_resume(),
            CommandType.STOP: lambda: self._cmd_stop(),
            CommandType.PNL: lambda: self._cmd_pnl(),
            CommandType.BLACKLIST: lambda: self._cmd_blacklist(cmd.args),
        }
        handler = handlers.get(cmd.command)
        if handler is None:
            return "❓ Unknown command"
        return handler()

    def _cmd_status(self) -> str:
        """!status — Show balance, positions, and active state."""
        balance = self.get_balance()
        positions = self.strategy.active_positions
        n_trades = len(self._trade_history)
        total_pnl = sum(t['pnl_usd'] for t in self._trade_history)

        status = "⏸️ PAUSED" if self._paused else "▶️ ACTIVE"
        if self._drawdown_halt:
            status = "🛑 DD HALT"

        lines = [
            f"```",
            f"📊 QuantBox Status — {status}",
            f"{'─' * 40}",
            f"Balance:    ${balance:,.2f}",
            f"Session PnL: ${total_pnl:+,.2f} ({n_trades} trades)",
            f"Positions:  {len(positions)}/{self.config.max_concurrent}",
            f"Cycle:      #{self._cycle_count}",
            f"{'─' * 40}",
        ]

        if positions:
            for sym, pos in positions.items():
                direction = "L" if pos['signal'] == 1 else "S"
                bars = pos.get('bars_held', 0)
                best = pos.get('best_pnl', 0) * 100
                notional = pos.get('notional', 0)
                lines.append(
                    f"  {direction} {sym}: "
                    f"${pos['entry_price']:.4f} | "
                    f"${notional:.0f} | "
                    f"{bars}bars | best={best:+.1f}%"
                )
        else:
            lines.append("  No open positions")

        lines.append("```")
        return '\n'.join(lines)

    def _cmd_close_all(self) -> str:
        """!close all — Close all open positions."""
        positions = list(self.strategy.active_positions.keys())
        if not positions:
            return "📭 No positions to close"

        closed = 0
        for sym in positions:
            price_arr = self.download_klines(sym, 2)
            price = price_arr[-1] if price_arr is not None else 0
            if self._close_position(sym, 'bot_close_all', price):
                closed += 1

        self._save_state()
        return f"✅ Closed {closed}/{len(positions)} positions"

    def _cmd_close_slot(self, slot_id: int) -> str:
        """!close <N> — Close the Nth active position (0-indexed).

        Since the scalper uses symbol names instead of fixed slots,
        positions are numbered in order of their dict keys.

        Args:
            slot_id: 0-based index of the position to close.
        """
        positions = list(self.strategy.active_positions.items())
        if not positions:
            return "📭 No positions to close"

        if slot_id < 0 or slot_id >= len(positions):
            symbols = [f"{i}: {s}" for i, (s, _) in enumerate(positions)]
            return (
                f"❌ Invalid index {slot_id}. "
                f"Active positions:\n" + '\n'.join(symbols)
            )

        sym, pos = positions[slot_id]
        price_arr = self.download_klines(sym, 2)
        price = price_arr[-1] if price_arr is not None else 0
        if self._close_position(sym, 'bot_close', price):
            self._save_state()
            return f"✅ Closed position {slot_id}: {sym}"
        return f"❌ Failed to close {sym}"

    def _cmd_pause(self) -> str:
        """!pause — Stop new entries, keep monitoring existing."""
        if self._paused:
            return "⏸️ Already paused"
        self._paused = True
        logger.info("Bot command: PAUSED (no new entries)")
        return "⏸️ New entries paused. Existing positions still monitored."

    def _cmd_resume(self) -> str:
        """!resume — Resume normal operation."""
        if not self._paused:
            return "▶️ Already active"
        self._paused = False
        logger.info("Bot command: RESUMED")
        return "▶️ Trading resumed"

    def _cmd_stop(self) -> str:
        """!stop — Close all positions and shut down."""
        self._stop_requested = True
        # Close all positions first
        positions = list(self.strategy.active_positions.keys())
        closed = 0
        for sym in positions:
            price_arr = self.download_klines(sym, 2)
            price = price_arr[-1] if price_arr is not None else 0
            if self._close_position(sym, 'bot_stop', price):
                closed += 1
        self._save_state()
        logger.info(f"Bot command: STOP (closed {closed} positions)")
        return (
            f"🛑 Stopping... Closed {closed} positions. "
            f"Scalper will exit after this cycle."
        )

    def _cmd_pnl(self) -> str:
        """!pnl — Session PnL summary."""
        trades = self._trade_history
        if not trades:
            return "📭 No trades this session"

        total_pnl = sum(t['pnl_usd'] for t in trades)
        wins = [t for t in trades if t['pnl_usd'] > 0]
        losses = [t for t in trades if t['pnl_usd'] <= 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        avg_win = (sum(t['pnl_usd'] for t in wins) / len(wins)) if wins else 0
        avg_loss = (
            sum(t['pnl_usd'] for t in losses) / len(losses)
        ) if losses else 0
        best = max(trades, key=lambda t: t['pnl_usd'])
        worst = min(trades, key=lambda t: t['pnl_usd'])

        balance = self.get_balance()
        session_return = (
            (balance - self._initial_balance) / self._initial_balance * 100
            if self._initial_balance > 0 else 0
        )

        return (
            f"```\n"
            f"📈 Session PnL Summary\n"
            f"{'─' * 36}\n"
            f"Total PnL:    ${total_pnl:+,.2f}\n"
            f"Session ROI:  {session_return:+.2f}%\n"
            f"Trades:       {len(trades)} "
            f"(W:{len(wins)} / L:{len(losses)})\n"
            f"Win Rate:     {wr:.1f}%\n"
            f"Avg Win:      ${avg_win:+,.2f}\n"
            f"Avg Loss:     ${avg_loss:+,.2f}\n"
            f"Best Trade:   {best['symbol']} ${best['pnl_usd']:+,.2f}\n"
            f"Worst Trade:  {worst['symbol']} ${worst['pnl_usd']:+,.2f}\n"
            f"Balance:      ${balance:,.2f}\n"
            f"```"
        )

    def _cmd_blacklist(self, args: dict | None = None) -> str:
        """!blacklist — View or clear the exclude list.

        Args:
            args: Optional dict with 'action': 'clear' to reset.
        """
        if args and args.get('action') == 'clear':
            self.config.exclude_symbols.clear()
            return "✅ Blacklist cleared"

        excludes = self.config.exclude_symbols
        if not excludes:
            return "📭 Blacklist is empty"
        return f"🚫 Blacklist ({len(excludes)}): " + ', '.join(sorted(excludes))

    # ==================== State Persistence ====================

    def _save_state(self) -> None:
        """Persist positions and trade history."""
        state = {
            'version': '2.0.0',
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'paper_mode': self.paper_mode,
            'paper_balance': self._paper_balance,
            'initial_balance': self._initial_balance,
            'high_water_mark': self._high_water_mark,

            'positions': self.strategy.active_positions,
            'trade_history': self._trade_history[-100:],  # Keep last 100
            'cycle_count': self._cycle_count,
        }
        try:
            _save_scalper_state(state)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _load_state(self) -> None:
        """Restore state from disk."""
        state = _load_scalper_state()
        if state is None:
            return

        logger.info(
            f"Restoring state from {state.get('saved_at', 'unknown')}"
        )

        self._paper_balance = state.get('paper_balance', self._paper_balance)
        self._initial_balance = state.get('initial_balance', 0)
        self._high_water_mark = state.get('high_water_mark', 0)

        self._trade_history = state.get('trade_history', [])
        self._cycle_count = 0  # Always reset per session (new log file each run)

        # Restore positions
        positions = state.get('positions', {})
        for sym, pos in positions.items():
            self.strategy.open_position(
                symbol=sym,
                signal=pos['signal'],
                entry_price=pos['entry_price'],
                notional=pos['notional'],
                leverage=pos.get('leverage', 1),
                move_size=pos.get('move_size', 0.03),
                entry_stop_mult=pos.get('entry_stop_mult'),
                server_sl_order_id=pos.get('server_sl_order_id'),
                server_tp_order_id=pos.get('server_tp_order_id'),
            )
            # Restore bars_held
            if sym in self.strategy._positions:
                self.strategy._positions[sym]['bars_held'] = pos.get(
                    'bars_held', 0
                )
                self.strategy._positions[sym]['best_pnl'] = pos.get(
                    'best_pnl', 0.0
                )
                self.strategy._positions[sym]['partial_closed'] = pos.get(
                    'partial_closed', False
                )

        logger.info(
            f"Restored {len(positions)} positions, "
            f"{len(self._trade_history)} trade records"
        )

    # ==================== Entry Points ====================

    def run_once(self) -> None:
        """Run a single trading cycle."""
        balance = self.get_balance()
        self._initial_balance = balance
        self._high_water_mark = balance
        self._daily_start_balance = balance
        self._load_state()

        symbols = self.get_trading_symbols()
        if not symbols:
            logger.error("No symbols available")
            return

        self._print_startup_banner(balance, len(symbols))
        self._cycle(symbols)

    def run_continuous(self) -> None:
        """Run the scalper continuously, cycling every 15 minutes.

        Waits for bar close, processes all symbols, repeats.
        Starts/stops the Discord bot for interactive commands.
        """
        # Setup
        balance = self.get_balance()
        self._initial_balance = balance
        self._high_water_mark = balance
        self._daily_start_balance = balance
        self._load_state()

        mode_str = "PAPER" if self.paper_mode else "LIVE"
        n_symbols = len(self.config.symbol_whitelist) if self.config.symbol_whitelist else 0
        self._print_startup_banner(balance, n_symbols)
        logger.info(f"Starting continuous trading ({mode_str})")
        logger.info(f"Initial balance: ${balance:.2f}")

        # Start Discord bot for interactive commands
        if self._bot_runner:
            self._bot_runner.start()
            logger.info("Discord Bot started for interactive commands")

        n_symbols = len(self.config.symbol_whitelist) if self.config.symbol_whitelist else 0
        self.discord.send(embed={
            'title': '🚀 QuantBox Scalper Started',
            'color': 0x2979FF,
            'description': (
                f"Rubber Band mean-reversion scalper initialized.\n"
                f"Use `!help` for interactive commands."
            ),
            'fields': [
                {'name': 'Mode', 'value': f"**{mode_str}**", 'inline': True},
                {'name': 'Balance', 'value': f'${balance:,.2f}', 'inline': True},
                {'name': 'Slots', 'value': f'{self.strategy.n_active}/{self.config.max_concurrent}', 'inline': True},
                {'name': 'Strategy', 'value': (
                    f"LB={self.config.lookback} | "
                    f"Thr={self.config.threshold} | "
                    f"TP={self.config.target_retrace} | "
                    f"SL={self.config.stop_mult}x"
                ), 'inline': False},
                {'name': 'Universe', 'value': f'{n_symbols} symbols', 'inline': True},
                {'name': 'Max Hold', 'value': f'{self.config.max_hold} bars', 'inline': True},
                {'name': 'Leverage', 'value': (
                    f'Base {self.config.base_leverage}x, '
                    f'MTF {self.config.mtf_leverage_mult}x'
                ), 'inline': True},
            ],
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })

        # Pre-fetch exchange info
        self.get_symbol_info('BTCUSDT')

        # Get trading universe (refresh every 6 hours)
        symbols = self.get_trading_symbols()
        last_universe_refresh = time.monotonic()
        UNIVERSE_REFRESH_INTERVAL = 6 * 3600  # 6 hours

        while not self._stop_requested:
            try:
                # Refresh universe periodically
                if time.monotonic() - last_universe_refresh > UNIVERSE_REFRESH_INTERVAL:
                    symbols = self.get_trading_symbols()
                    last_universe_refresh = time.monotonic()
                    if not symbols:
                        logger.warning("No symbols — retrying in 60s")
                        time.sleep(60)
                        continue

                # Wait for next 15m bar close
                self._wait_for_bar()

                # Process cycle
                self._cycle(symbols)

            except KeyboardInterrupt:
                logger.info("Shutdown requested (KeyboardInterrupt)")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                self.discord.send(embed={
                    'title': '🚨 Cycle Error',
                    'color': 0xFF1744,
                    'description': str(e)[:500],
                    'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
                })
                time.sleep(60)

        # Clean shutdown
        logger.info("Shutting down...")
        self._save_state()

        # Stop Discord bot
        if self._bot_runner:
            self._bot_runner.stop()

        total_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        balance_now = self.get_balance()
        session_return = (
            (balance_now - self._initial_balance) / self._initial_balance * 100
            if self._initial_balance > 0 else 0
        )

        best = max(self._trade_history, key=lambda t: t['pnl_usd']) if self._trade_history else None
        worst = min(self._trade_history, key=lambda t: t['pnl_usd']) if self._trade_history else None

        # Terminal summary
        print(format_session_summary(
            cycles=self._cycle_count,
            n_trades=n_trades,
            win_rate=wr,
            total_pnl=total_pnl,
            balance=balance_now,
            session_roi=session_return,
            best_trade=best,
            worst_trade=worst,
        ))

        fields = [
            {'name': 'Cycles', 'value': str(self._cycle_count), 'inline': True},
            {'name': 'Trades', 'value': f'{n_trades} (WR {wr:.0f}%)', 'inline': True},
            {'name': 'Total PnL', 'value': f'${total_pnl:+,.2f}', 'inline': True},
            {'name': 'Balance', 'value': f'${balance_now:,.2f}', 'inline': True},
            {'name': 'Session ROI', 'value': f'{session_return:+.2f}%', 'inline': True},
        ]
        if best:
            fields.append({'name': 'Best', 'value': f"{best['symbol']} ${best['pnl_usd']:+,.2f}", 'inline': True})
        if worst:
            fields.append({'name': 'Worst', 'value': f"{worst['symbol']} ${worst['pnl_usd']:+,.2f}", 'inline': True})

        self.discord.send(embed={
            'title': '🛑 QuantBox Scalper Stopped',
            'color': 0xFF1744 if total_pnl < 0 else 0x00C853,
            'fields': fields,
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })

    def _print_startup_banner(self, balance: float, n_symbols: int) -> None:
        """Print the startup banner with system info."""
        settings = get_settings()
        mode_str = 'PAPER' if self.paper_mode else 'LIVE'
        print_banner()
        print_system_info(
            mode=mode_str,
            strategy=self.strategy_name,
            balance=balance,
            n_symbols=n_symbols,
            max_slots=self.config.max_concurrent,
            leverage_base=self.config.base_leverage,
            leverage_max=self.config.max_leverage,
            lookback=self.config.lookback,
            threshold=self.config.threshold,
            target=self.config.target_retrace,
            stop_mult=self.config.stop_mult,
            max_hold=self.config.max_hold,
            entry_mode=self.config.entry_mode,
            min_confidence=self.config.min_confidence,
            testnet=settings.binance_testnet,
        )
        print_startup_complete(mode_str)

    def _next_bar_time(self) -> datetime:
        """Calculate the next 15-minute bar close time (+5s buffer)."""
        now = datetime.now(timezone.utc)
        next_boundary = ((now.minute // 15) + 1) * 15
        if next_boundary >= 60:
            target = now.replace(
                minute=0, second=5, microsecond=0
            ) + timedelta(hours=1)
        else:
            target = now.replace(
                minute=next_boundary, second=5, microsecond=0
            )
        return target

    def _wait_for_bar(self) -> None:
        """Wait for the next 15m bar close, polling every 5s for commands.

        While waiting:
        - Process bot commands (~5s latency for responsive interaction)
        - Sync server-side TP/SL order fills (every 60s to avoid API spam)
        - Check for stop requests
        """
        target = self._next_bar_time()
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining > 0:
            logger.debug(f"Waiting {remaining:.0f}s for next bar close")

        BOT_POLL_INTERVAL = 5.0    # Check bot commands every 5s
        SYNC_INTERVAL = 60.0       # Sync server orders every 60s
        last_sync = time.monotonic()

        while True:
            now = datetime.now(timezone.utc)
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                break

            # Process bot commands while waiting (~5s response time)
            self._process_bot_commands()
            if self._stop_requested:
                break

            # Sync server-side TP/SL fills periodically (not every 5s)
            if time.monotonic() - last_sync >= SYNC_INTERVAL:
                self._sync_server_orders()
                last_sync = time.monotonic()

            # Sleep 5s max for responsive bot commands
            sleep_secs = min(BOT_POLL_INTERVAL, remaining)
            time.sleep(sleep_secs)


# ──────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantBox Scalper — Rubber Band / Elliott Wave'
    )
    parser.add_argument(
        '--paper', action='store_true',
        help='Paper trading mode (no real orders)'
    )
    parser.add_argument(
        '--once', action='store_true',
        help='Run a single cycle then exit'
    )
    parser.add_argument(
        '--lookback', type=int, default=None,
        help='Signal lookback bars (default: from .env/config)'
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Min move to trigger (default: from .env/config)'
    )
    parser.add_argument(
        '--max-slots', type=int, default=None,
        help='Max concurrent positions (default: from .env/config)'
    )
    parser.add_argument(
        '--strategy', choices=['rubberband', 'elliott'], default='rubberband',
        help='Trading strategy: rubberband (default) or elliott (wave ABC)'
    )
    args = parser.parse_args()

    # Build config from .env settings (pydantic-settings) + CLI overrides
    settings = get_settings()
    config = RubberBandConfig(
        lookback=args.lookback if args.lookback is not None else settings.rb_lookback,
        threshold=args.threshold if args.threshold is not None else settings.rb_threshold,
        target_retrace=settings.rb_target_retrace,
        stop_mult=settings.rb_stop_mult,
        max_hold=settings.rb_max_hold,
        stop_decay=settings.rb_stop_decay,
        htf_period=settings.rb_htf_period,
        htf_block_threshold=settings.rb_htf_block_threshold,
        htf_agree_discount=settings.rb_htf_agree_discount,
        htf_agree_stop_mult=settings.rb_htf_agree_stop_mult,
        dynamic_leverage_enabled=settings.rb_dynamic_leverage_enabled,
        base_leverage=settings.rb_base_leverage,
        leverage_scale=settings.rb_leverage_scale,
        max_leverage=settings.rb_max_leverage,
        btc_directional_enabled=settings.rb_btc_directional_enabled,
        btc_directional_period=settings.rb_btc_directional_period,
        partial_exit_enabled=settings.rb_partial_exit_enabled,
        partial_exit_fraction=settings.rb_partial_exit_fraction,
        rr_leverage_enabled=settings.rb_rr_leverage_enabled,
        min_rr_for_leverage=settings.rb_min_rr_for_leverage,
        max_concurrent=args.max_slots if args.max_slots is not None else 5,
        # v6.0: Confidence scoring
        entry_mode=settings.rb_entry_mode,
        btc_regime_period=settings.rb_btc_regime_period,
        asset_trend_period=settings.rb_asset_trend_period,
        min_confidence=settings.rb_min_confidence,
        confidence_leverage_step=settings.rb_confidence_leverage_step,
        rsi_period=settings.rb_rsi_period,
        vol_period=settings.rb_vol_period,
        # v7.0: Multi-timeframe confirmation
        mtf_confirmation_enabled=settings.rb_mtf_confirmation_enabled,
        mtf_sma_period=settings.rb_mtf_sma_period,
        mtf_leverage_mult=settings.rb_mtf_leverage_mult,
        # v8.0: Dynamic target (disabled in v9.0 — Fib discipline outperforms)
        dynamic_target_enabled=settings.rb_dynamic_target_enabled,
        target_conf3=settings.rb_target_conf3,
        target_conf4=settings.rb_target_conf4,
        target_conf5=settings.rb_target_conf5,
        # v9.0: S+R+M Confluence (MACD + Stochastic evidence)
        macd_enabled=settings.rb_macd_enabled,
        macd_fast=settings.rb_macd_fast,
        macd_slow=settings.rb_macd_slow,
        macd_signal_period=settings.rb_macd_signal_period,
        stoch_enabled=settings.rb_stoch_enabled,
        stoch_k_period=settings.rb_stoch_k_period,
        stoch_d_period=settings.rb_stoch_d_period,
        # v11.4: Adaptive Threshold (Walk-Forward Validated)
        atr_period=settings.rb_atr_period,
        atr_multiplier=settings.rb_atr_multiplier,
        threshold_floor=settings.rb_threshold_floor,
        threshold_cap=settings.rb_threshold_cap,
        # v10.0: Symbol whitelist & dynamic capital
        symbol_whitelist=[s.strip() for s in settings.rb_symbol_whitelist.split(',') if s.strip()],
        dynamic_capital_enabled=settings.rb_dynamic_capital_enabled,
        # v7.0: Independent 4H rubber band
        htf_independent_enabled=settings.rb_htf_independent_enabled,
        htf_rb_lookback=settings.rb_htf_rb_lookback,
        htf_rb_threshold=settings.rb_htf_rb_threshold,
        htf_rb_target_retrace=settings.rb_htf_rb_target_retrace,
        htf_rb_stop_mult=settings.rb_htf_rb_stop_mult,
        htf_rb_max_hold=settings.rb_htf_rb_max_hold,
        htf_rb_btc_sma_period=settings.rb_htf_rb_btc_sma_period,
    )

    scalper = RubberBandScalper(
        paper_mode=args.paper,
        config=config,
        strategy_name=args.strategy,
    )

    # ── Live trading confirmation gate ──
    if not args.paper:
        if not confirm_live_trading():
            sys.exit(0)

    if args.once:
        scalper.run_once()
    else:
        scalper.run_continuous()


if __name__ == '__main__':
    main()
