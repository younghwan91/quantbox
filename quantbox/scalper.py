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
import subprocess
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
    compute_atr_pct, compute_confidence_leverage,
)
from quantbox.strategy.elliott import ElliottConfig, ElliottStrategy
from quantbox.strategy.vol_straddle import VolStraddleConfig, VolStraddleStrategy
from quantbox.core.commands import CommandsMixin
from quantbox.execution.brackets import BracketMixin
from quantbox.execution.positions import PositionMixin
from quantbox.execution.reconciliation import ReconciliationMixin

load_dotenv()

# ──────────────────────── Git Commit Tracking ─────────────────────────


def _get_git_commit() -> dict[str, str]:
    """Get current git commit info for log traceability.

    Returns:
        Dict with 'short', 'full', 'message', 'date' keys.
        All values are empty strings if git is unavailable.
    """
    info = {'short': '', 'full': '', 'message': '', 'date': ''}
    try:
        info['short'] = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        info['full'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        info['message'] = subprocess.run(
            ['git', 'log', '-1', '--format=%s'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        info['date'] = subprocess.run(
            ['git', 'log', '-1', '--format=%ci'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        pass
    return info


GIT_COMMIT = _get_git_commit()

# ──────────────────────── Logging ──────────────────────────────────────

def _setup_logging() -> Path:
    log_dir = Path('logs/bot')
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'scalper_{ts}.log'

    # Write git commit header before configuring logging handlers
    # so the log file starts with version traceability info
    if GIT_COMMIT['short']:
        with open(log_file, 'w') as f:
            f.write(f"# Commit: {GIT_COMMIT['short']} ({GIT_COMMIT['date']})\n")
            f.write(f"# Message: {GIT_COMMIT['message']}\n")
            f.write(f"# Full SHA: {GIT_COMMIT['full']}\n")
            f.write("#\n")

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


# ──────────────── Diagnostic Logger ─────────────────────
from quantbox.utils.diagnostics import DiagnosticLogger, _diag_logger  # noqa: E402

# ──────────────────────── State File ──────────────────────────────────

STATE_DIR = Path('data/state')
STATE_FILE = STATE_DIR / 'scalper_state.json'
PAPER_STATE_FILE = STATE_DIR / 'scalper_state_paper.json'


def _save_scalper_state(state: dict, filepath: Path | None = None) -> None:
    """Atomic JSON write."""
    if filepath is None:
        filepath = STATE_FILE
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


def _load_scalper_state(filepath: Path | None = None) -> dict | None:
    if filepath is None:
        filepath = STATE_FILE
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"State load failed: {e}")
        return None


# ──────────────────────── Scalper Class ────────────────────────────────

class RubberBandScalper(
    BracketMixin, ReconciliationMixin, PositionMixin, CommandsMixin,
):
    """Live rubber band scalper for Binance USDT-M Futures.

    Composes execution logic from mixin classes:
    - BracketMixin: Server-side SL/TP via Algo Order API
    - ReconciliationMixin: Exchange state synchronization
    - PositionMixin: Position lifecycle (open/close/partial/pyramid)
    - CommandsMixin: Discord bot command handlers

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
        vs_config: VolStraddleConfig | None = None,
    ):
        self.paper_mode = paper_mode
        self.config = config or RubberBandConfig()
        self.strategy_name = strategy_name

        if strategy_name == 'elliott':
            self.elliott_config = elliott_config or ElliottConfig()
            self.strategy = ElliottStrategy(self.elliott_config)
        elif strategy_name == 'vol_straddle':
            self.vs_config = vs_config or VolStraddleConfig()
            self.strategy = VolStraddleStrategy(self.vs_config)
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
        logger.info(f"RubberBandScalper initialized ({mode_str}) strategy={strategy_name}")
        if strategy_name == 'vol_straddle':
            vs = self.vs_config
            logger.info(
                f"Config: VR({vs.vr_short}/{vs.vr_long}) "
                f"D_threshold={vs.d_vr_threshold}, "
                f"SL={vs.d_sl*100:.1f}%, TP={vs.d_tp*100:.1f}%, "
                f"TF={'15m' if vs.enable_15m else ''}"
                f"{'|1h' if vs.enable_1h else ''}"
                f"{'|4h' if vs.enable_4h else ''}"
            )
        else:
            _sl_str = (f"ATR×{self.config.atr_stop_mult}[{self.config.atr_stop_floor*100:.1f}%-{self.config.atr_stop_cap*100:.0f}%]"
                       if self.config.atr_stop_enabled else f"{self.config.stop_mult}×move")
            logger.info(
                f"Config: lookback={self.config.lookback}, "
                f"threshold={self.config.threshold}, "
                f"target={self.config.target_retrace}, "
                f"stop={_sl_str}, "
                f"max_hold={self.config.max_hold}"
            )

    # ==================== Balance & Risk ====================

    @property
    def _cfg(self):
        """Active strategy config for shared fields."""
        if self.strategy_name == 'vol_straddle':
            return self.vs_config
        return self.config

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
        # v12.5.1: Return last known balance instead of 0.0 to prevent
        # false drawdown halt on transient API failures.
        if hasattr(self, '_last_known_balance') and self._last_known_balance > 0:
            logger.warning(
                f"Using last known balance ${self._last_known_balance:.2f} "
                f"due to API failure"
            )
            return self._last_known_balance
        return 0.0

    def _check_risk_limits(self, balance: float) -> bool:
        """Check session drawdown limit.

        Returns True if trading should be blocked.
        """
        # v12.5.1: Track last known good balance for API failure fallback
        if balance > 0:
            self._last_known_balance = balance

        # Update high water mark
        if balance > self._high_water_mark:
            self._high_water_mark = balance

        # Session drawdown check
        if self._high_water_mark > 0:
            dd = (self._high_water_mark - balance) / self._high_water_mark
            if dd >= self._cfg.max_drawdown_pct:
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
            if self._cfg.symbol_whitelist:
                whitelist_set = set(self._cfg.symbol_whitelist)
                exclude_set = set(self._cfg.exclude_symbols)
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
                    s for s in sorted_syms[:self._cfg.min_volume_rank]
                    if s not in self._cfg.exclude_symbols
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
            # v12.4.7: Skip leverage change for reduce-only orders.
            # Changing leverage on an open position can fail with
            # -2028 "insufficient margin balance" if the new leverage
            # is lower than what the position requires.  reduce_only
            # orders don't change margin requirements.
            if not reduce_only:
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
            max_slots=self._cfg.max_concurrent,
        )
        print(header)

        # ── Enhanced cycle header with HWM / DD / cum PnL ──
        hwm = self._high_water_mark if self._high_water_mark > 0 else balance
        dd_pct = (hwm - balance) / hwm * 100 if hwm > 0 else 0.0
        cum_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        n_trades = len(self._trade_history)
        logger.info(
            f"Cycle #{self._cycle_count} | "
            f"Balance=${balance:.2f} | "
            f"HWM=${hwm:.2f} | DD={dd_pct:.1f}% | "
            f"Positions={self.strategy.n_active}/{self._cfg.max_concurrent} | "
            f"Trades={n_trades} CumPnL=${cum_pnl:+.2f}"
        )

        # ── Position status moved AFTER _sync_server_orders ──
        # (SL/TP status must be checked after sync, not before)

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
        elif self.strategy_name == 'vol_straddle':
            # VolStraddle doesn't use BTC as filter — skip 4h BTC fetch
            pass
        elif self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
            btc_close_4h = self.download_klines('BTCUSDT', 100, interval='4h')

        # 3c. v11.2: Sync server-side TP/SL orders (check if filled intra-bar)
        self._sync_server_orders()

        # v12.5.1: Reconcile positions with exchange AFTER sync
        if not self.paper_mode:
            self._reconcile_positions()

        # v12.5.1: Log position status AFTER sync (SL/TP verified)
        for sym, pos in self.strategy.active_positions.items():
            try:
                mark = self.client.futures_mark_price(symbol=sym)
                cur_px = float(mark['markPrice'])
            except Exception:
                cur_px = pos.get('entry_price', 0)
            entry_px = pos.get('entry_price', 0)
            if entry_px > 0:
                if pos['signal'] == 1:
                    upnl_pct = (cur_px - entry_px) / entry_px * 100
                else:
                    upnl_pct = (entry_px - cur_px) / entry_px * 100
            else:
                upnl_pct = 0.0
            sl_id = pos.get('server_sl_order_id')
            tp_id = pos.get('server_tp_order_id')
            bars = pos.get('bars_held', 0)
            direction = 'L' if pos.get('signal') == 1 else 'S'
            logger.info(
                f"  [{sym}] {direction} entry=${entry_px:.4f} "
                f"mark=${cur_px:.4f} uPnL={upnl_pct:+.2f}% bars={bars} "
                f"SL={'✓' if sl_id else '✗'} TP={'✓' if tp_id else '✗'}"
            )

        # 4. Process existing positions (always, even when paused)
        for sym in list(self.strategy.active_positions):
            # v12.3.2: Download full OHLC for active positions to compute
            # fresh ATR. Previously only close was downloaded, so runner
            # trailing used stale ATR from entry time (on_bar never called
            # for symbols with open positions).
            result_full = self.download_klines(sym, 200, full=True)
            if result_full is None:
                continue
            close, high, low, _vol, _tb = result_full

            current_price = close[-1]

            # v12.3.2: Compute FRESH ATR for PBT runner trailing
            # VolStraddle doesn't use ATR-based trailing; pass default.
            if self.strategy_name == 'vol_straddle':
                current_atr = 0.02
            elif self.config.atr_period > 0 and len(close) >= self.config.atr_period:
                atr_arr = compute_atr_pct(high, low, close, self.config.atr_period)
                atr_val = float(atr_arr[-1])
                current_atr = atr_val if not np.isnan(atr_val) else 0.02
            else:
                current_atr = 0.02

            exit_reason = self.strategy.update_position(
                sym, current_price, current_atr_pct=current_atr,
                current_high=float(high[-1]), current_low=float(low[-1]),
            )
            if exit_reason:
                logger.info(
                    f"  [{sym}] Exit signal: {exit_reason} @ ${current_price:.4f}"
                )
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
            else:
                pos = self.strategy._positions.get(sym)
                if pos:
                    _mh = pos.get('max_hold', self.config.max_hold)
                    logger.debug(
                        f"  [{sym}] HOLD: bars={pos.get('bars_held', 0)}/{_mh} "
                        f"best_pnl={pos.get('best_pnl', 0)*100:.2f}%"
                    )
                    # v12.3.3: Ratchet server SL for runners to protect profits
                    if (pos.get('partial_closed')
                            and self.config.breakeven_after_partial
                            and not self.paper_mode):
                        self._ratchet_runner_sl(sym, pos)

        # 5. Look for new entries (skip if risk blocked or paused)
        if self._paused:
            logger.info("⏸️ Paused — skipping new entries")
        elif not risk_blocked:
            available_slots = (
                self._cfg.max_concurrent - self.strategy.n_active
            )
            if available_slots > 0:
                self._scan_entries(symbols, btc_close, balance, available_slots,
                                   btc_close_4h=btc_close_4h)

        # 6. Save state
        self._save_state()

        # 7. Periodic summary (every cycle for active positions, full every hour)
        if self.strategy.n_active > 0 or self._cycle_count % 4 == 0:
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
            # VolStraddle needs 500 bars: 15m warmup=200 needs ≥201 bars,
            # 1h resampling needs ≥4×(warmup_1h+1)=204 bars. 500 covers both.
            n_bars = 500 if self.strategy_name == 'vol_straddle' else 200
            result = self.download_klines(sym, n_bars, full=True)
            if result is None:
                continue
            close, high, low, volume, taker_buy = result
            if len(close) < self._cfg.lookback + 1:
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
            elif self.strategy_name == 'vol_straddle':
                # VolStraddle needs full 4h OHLCV for VR computation
                result_4h = self.download_klines(sym, 100, interval='4h', full=True)
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
            _lb = self._cfg.lookback
            abs_move = abs(close[-1] / close[-_lb - 1] - 1.0)
            moves_seen.append((sym, abs_move))

            if signal != 0:
                # Measure move strength for prioritization
                move = abs(
                    close[-1] / close[-_lb - 1] - 1.0
                )
                candidates.append((sym, signal, move, close))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

        # ── Diagnostic: log scan summary every cycle ──
        top_movers = sorted(moves_seen, key=lambda x: x[1], reverse=True)[:3]
        if self.strategy_name == 'vol_straddle':
            # VR strategy: show VR threshold + actual VR values
            print(format_scan_summary(len(candidates), top_movers, 0.0))
            if candidates:
                cand_strs = [f"{s}({m*100:.1f}%)" for s, _, m, _ in candidates[:3]]
                logger.info(f"🎯 {len(candidates)} candidates: {', '.join(cand_strs)}")
            else:
                if moves_seen:
                    # Show top 3 moves with VR for all TFs
                    top3_with_vr = []
                    for s, m in top_movers:
                        vr_all = self.strategy.get_vr_all_tf(s)
                        vr_str = f"15m={vr_all['15m']:.2f}/1h={vr_all['1h']:.2f}/4h={vr_all['4h']:.2f}"
                        top3_with_vr.append(f"{s}({m*100:.2f}%, {vr_str})")
                    logger.info(
                        f"No candidates — top moves: {', '.join(top3_with_vr)} "
                        f"(threshold VR<{self.vs_config.d_vr_threshold} any TF)"
                    )
        else:
            print(format_scan_summary(len(candidates), top_movers, self.config.threshold))
            if candidates:
                cand_strs = [f"{s}({m*100:.1f}%)" for s, _, m, _ in candidates[:3]]
                logger.info(f"🎯 {len(candidates)} candidates: {', '.join(cand_strs)}")
            else:
                if moves_seen:
                    top3 = [f"{s}({m*100:.2f}%)" for s, m in top_movers]
                    logger.info(
                        f"No candidates — top moves: {', '.join(top3)} "
                        f"(base >{self.config.threshold*100:.1f}%, adaptive per-symbol)"
                    )

        # v12.0: Write structured diagnostics to JSONL
        if scan_diags:
            ts = datetime.now(timezone.utc).isoformat()
            _diag_logger.log_scan(self._cycle_count, ts, scan_diags)

        entries_made = 0
        for sym, signal, move, close in candidates:
            if entries_made >= max_entries:
                break

            # ═══ v12.3 PARITY: Direction Diversity Filter ═══
            if self._cfg.dir_diversity_enabled:
                same_dir = sum(
                    1 for p in self.strategy.active_positions.values()
                    if p.get('signal') == signal
                )
                if same_dir >= self._cfg.max_same_direction:
                    logger.debug(
                        f"[{sym}] DirDiversity blocked: "
                        f"{same_dir} {'LONG' if signal == 1 else 'SHORT'} "
                        f"already open (max={self._cfg.max_same_direction})"
                    )
                    continue

            if self._open_position(sym, signal, close, balance):
                entries_made += 1

        if entries_made > 0:
            logger.info(
                f"Opened {entries_made} new positions "
                f"(from {len(candidates)} candidates)"
            )

    def _log_summary(self, balance: float) -> None:
        """Log periodic summary — always, even with 0 trades."""
        positions = self.strategy.active_positions
        n_trades = len(self._trade_history)
        wins = sum(1 for t in self._trade_history if t['pnl_usd'] > 0)
        total_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        wr = wins / n_trades * 100 if n_trades > 0 else 0

        hwm = self._high_water_mark if self._high_water_mark > 0 else balance
        dd_pct = (hwm - balance) / hwm * 100 if hwm > 0 else 0.0

        logger.info(
            f"[Summary] Trades={n_trades}, WR={wr:.0f}%, "
            f"PnL=${total_pnl:+.2f}, Balance=${balance:.2f}, "
            f"HWM=${hwm:.2f}, DD={dd_pct:.1f}%, "
            f"Active={len(positions)}"
        )

        # Active position details
        for sym, pos in positions.items():
            logger.info(
                f"  [{sym}] {'L' if pos['signal'] == 1 else 'S'} "
                f"entry=${pos['entry_price']:.4f} "
                f"bars={pos['bars_held']}/{pos.get('max_hold', self.config.max_hold)} "
                f"best_pnl={pos.get('best_pnl', 0)*100:.2f}% "
                f"SL={pos.get('server_sl_order_id') or 'NONE'} "
                f"TP={pos.get('server_tp_order_id') or 'NONE'}"
            )

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
        filepath = PAPER_STATE_FILE if self.paper_mode else STATE_FILE
        try:
            _save_scalper_state(state, filepath=filepath)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _load_state(self) -> None:
        """Restore state from disk."""
        filepath = PAPER_STATE_FILE if self.paper_mode else STATE_FILE
        state = _load_scalper_state(filepath=filepath)
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

        # v12.4.6: Reset session state when no positions survive restart.
        # Persisting HWM across sessions with 0 positions causes
        # immediate drawdown halt on restart (stale HWM >> current balance).
        # Trade history from a finished session is also stale — it inflates
        # the !pnl summary with trades that already settled into the balance.
        # If positions exist, keep saved HWM + history to track their lifecycle.
        if not positions:
            balance = self.get_balance()
            self._high_water_mark = balance
            self._initial_balance = balance
            self._trade_history = []
            logger.info(
                f"No positions to restore — fresh session: "
                f"HWM reset to ${balance:.2f}, trade history cleared"
            )

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
            # Restore bars_held and PBT runner state
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
                # v12.3 PBT: Restore runner trailing state
                # Without these, a restarted runner loses its breakeven
                # floor and peak tracking — violating PBT safety promise.
                self.strategy._positions[sym]['highest_since_tp1'] = pos.get(
                    'highest_since_tp1', 0.0
                )
                self.strategy._positions[sym]['trail_floor'] = pos.get(
                    'trail_floor', None
                )
                self.strategy._positions[sym]['confidence'] = pos.get(
                    'confidence', 0
                )
                # v12.3.3: Restore server SL ratchet state for runners.
                # Without these, a restarted runner triggers a redundant
                # ratchet API call and has a brief inconsistency window.
                self.strategy._positions[sym]['_computed_trail_stop'] = pos.get(
                    '_computed_trail_stop'
                )
                self.strategy._positions[sym]['_server_sl_pct'] = pos.get(
                    '_server_sl_pct', 0.0
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
        n_symbols = len(self._cfg.symbol_whitelist) if self._cfg.symbol_whitelist else 0
        self._print_startup_banner(balance, n_symbols)
        logger.info(f"Starting continuous trading ({mode_str})")
        logger.info(f"Initial balance: ${balance:.2f}")

        # Start Discord bot for interactive commands
        if self._bot_runner:
            self._bot_runner.start()
            logger.info("Discord Bot started for interactive commands")

        n_symbols = len(self._cfg.symbol_whitelist) if self._cfg.symbol_whitelist else 0
        commit_str = f"`{GIT_COMMIT['short']}`" if GIT_COMMIT['short'] else 'unknown'
        n_recovered = self.strategy.n_active

        if self.strategy_name == 'vol_straddle':
            vc = self.vs_config
            self.discord.send(embed={
                'title': '🚀 QuantBox VolStraddle 시작',
                'color': 0x2979FF,
                'description': (
                    f"VolStraddle D_mom8 (A1-D50) 초기화 완료.\n"
                    f"`!help`로 명령어 확인."
                ),
                'fields': [
                    {'name': '모드', 'value': f"**{mode_str}**", 'inline': True},
                    {'name': '잔고', 'value': f'${balance:,.2f}', 'inline': True},
                    {'name': '슬롯', 'value': f'{n_recovered}/{vc.max_concurrent}', 'inline': True},
                    {'name': '── 전략 설정 ──', 'value': (
                        f"VR({vc.vr_short}/{vc.vr_long}) < {vc.d_vr_threshold:.2f}\n"
                        f"TP={vc.d_tp*100:.0f}% | SL={vc.d_sl*100:.0f}% | "
                        f"Mom={vc.momentum_period}bar"
                    ), 'inline': False},
                    {'name': 'TF', 'value': (
                        f"{'15m ' if vc.enable_15m else ''}"
                        f"{'1h ' if vc.enable_1h else ''}"
                        f"{'4h' if vc.enable_4h else ''}"
                    ).strip(), 'inline': True},
                    {'name': '레버리지', 'value': f'{vc.base_leverage}x', 'inline': True},
                    {'name': '유니버스', 'value': f'{n_symbols}종목', 'inline': True},
                    {'name': '커밋', 'value': commit_str, 'inline': True},
                ],
                'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
            })
        else:
            eff_lev = self.config.base_leverage * self.config.mtf_leverage_mult
            _mtf_str = f"ON (SMA {self.config.mtf_sma_period})" if self.config.mtf_confirmation_enabled else "OFF"
            _adaptive_str = (
                f"적응형 {self.config.atr_multiplier}x "
                f"(floor {self.config.threshold_floor*100:.1f}%, cap {self.config.threshold_cap*100:.1f}%)"
                if self.config.atr_period > 0
                else f"고정 {self.config.threshold*100:.1f}%"
            )
            self.discord.send(embed={
                'title': '🚀 QuantBox 스캘퍼 시작',
                'color': 0x2979FF,
                'description': (
                    f"러버밴드 평균회귀 스캘퍼 초기화 완료.\n"
                    f"`!help`로 명령어 확인."
                ),
                'fields': [
                    {'name': '모드', 'value': f"**{mode_str}**", 'inline': True},
                    {'name': '잔고', 'value': f'${balance:,.2f}', 'inline': True},
                    {'name': '슬롯', 'value': f'{n_recovered}/{self.config.max_concurrent}', 'inline': True},
                    {'name': '── 전략 설정 ──', 'value': (
                        f"진입: {_adaptive_str}\n"
                        f"TP={self.config.target_retrace} | "
                        f"SL={'ATR×'+str(self.config.atr_stop_mult)+'['+str(round(self.config.atr_stop_floor*100,1))+'%-'+str(round(self.config.atr_stop_cap*100))+'%]' if self.config.atr_stop_enabled else str(self.config.stop_mult)+'×move'} | "
                        f"MaxHold={self.config.max_hold}바"
                    ), 'inline': False},
                    {'name': 'MTF', 'value': _mtf_str, 'inline': True},
                    {'name': '레버리지', 'value': (
                        f'Base {self.config.base_leverage}x × '
                        f'MTF {self.config.mtf_leverage_mult}x = '
                        f'{eff_lev:.1f}x'
                    ), 'inline': True},
                    {'name': '유니버스', 'value': f'{n_symbols}종목', 'inline': True},
                    {'name': '커밋', 'value': commit_str, 'inline': True},
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
                    'title': '🚨 사이클 오류',
                    'color': 0xFF1744,
                    'description': f'포지션 {self.strategy.n_active}개 활성 중\n```\n{str(e)[:400]}\n```',
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
            {'name': '사이클', 'value': str(self._cycle_count), 'inline': True},
            {'name': '거래', 'value': f'{n_trades}건 (승률 {wr:.0f}%)', 'inline': True},
            {'name': '총 수익', 'value': f'${total_pnl:+,.2f}', 'inline': True},
            {'name': '잔고', 'value': f'${balance_now:,.2f}', 'inline': True},
            {'name': '세션 ROI', 'value': f'{session_return:+.2f}%', 'inline': True},
        ]
        if best:
            fields.append({'name': '최고 수익', 'value': f"{best['symbol']} ${best['pnl_usd']:+,.2f}", 'inline': True})
        if worst:
            fields.append({'name': '최대 손실', 'value': f"{worst['symbol']} ${worst['pnl_usd']:+,.2f}", 'inline': True})

        self.discord.send(embed={
            'title': '🛑 QuantBox 스켈퍼 종료',
            'color': 0xFF1744 if total_pnl < 0 else 0x00C853,
            'fields': fields,
            'footer': {'text': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')},
        })

    def _print_startup_banner(self, balance: float, n_symbols: int) -> None:
        """Print the startup banner with system info."""
        settings = get_settings()
        mode_str = 'PAPER' if self.paper_mode else 'LIVE'
        print_banner()
        if GIT_COMMIT['short']:
            logger.info(
                f"Git: {GIT_COMMIT['short']} — {GIT_COMMIT['message']}"
            )
        if self.strategy_name == 'vol_straddle':
            vc = self.vs_config
            print_system_info(
                mode=mode_str,
                strategy='vol_straddle',
                balance=balance,
                n_symbols=n_symbols,
                max_slots=vc.max_concurrent,
                leverage_base=vc.base_leverage,
                leverage_max=vc.max_leverage,
                lookback=vc.momentum_period,
                threshold=vc.d_vr_threshold,
                target=vc.d_tp,
                stop_mult=vc.d_sl,
                max_hold=vc.max_hold_15m,
                entry_mode='vr_compression',
                min_confidence=0.0,
                testnet=settings.binance_testnet,
            )
        else:
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
            next_str = target.strftime('%H:%M:%S UTC')
            logger.info(f"Waiting {remaining:.0f}s for next bar close ({next_str})")

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
                self._save_state()  # v12.3.1: persist after inter-cycle sync
                last_sync = time.monotonic()

            # Sleep 5s max for responsive bot commands
            sleep_secs = min(BOT_POLL_INTERVAL, remaining)
            time.sleep(sleep_secs)


# ──────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantBox Scalper — Rubber Band / Elliott Wave / VolStraddle'
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
        '--strategy', choices=['rubberband', 'elliott', 'vol_straddle'],
        default='vol_straddle',
        help='Trading strategy: vol_straddle (default), rubberband, or elliott'
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
        trail_atr_mult=settings.rb_trail_atr_mult,
        breakeven_after_partial=settings.rb_breakeven_after_partial,
        runner_max_hold_mult=settings.rb_runner_max_hold_mult,
        pre_tp1_trail_activation=settings.rb_pre_tp1_trail_activation,
        pre_tp1_trail_factor=settings.rb_pre_tp1_trail_factor,
        rr_leverage_enabled=settings.rb_rr_leverage_enabled,
        min_rr_for_leverage=settings.rb_min_rr_for_leverage,
        max_concurrent=args.max_slots if args.max_slots is not None else settings.rb_max_concurrent,
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
        # v12.3: SteepX Confidence Sizing
        steep_cap_curve=settings.rb_steep_cap_curve,
        cap_mult_conf2=settings.rb_cap_mult_conf2,
        cap_mult_conf3=settings.rb_cap_mult_conf3,
        cap_mult_conf4=settings.rb_cap_mult_conf4,
        cap_mult_conf5=settings.rb_cap_mult_conf5,
        # v12.3: SoftTight DD Scaling
        soft_equity_scale=settings.rb_soft_equity_scale,
        soft_eq_dd1=settings.rb_soft_eq_dd1,
        soft_eq_scale1=settings.rb_soft_eq_scale1,
        soft_eq_dd2=settings.rb_soft_eq_dd2,
        soft_eq_scale2=settings.rb_soft_eq_scale2,
        soft_eq_dd3=settings.rb_soft_eq_dd3,
        soft_eq_scale3=settings.rb_soft_eq_scale3,
        # v12.3: Vol-Adjusted Sizing
        vol_adjusted_size=settings.rb_vol_adjusted_size,
        vol_adj_median_atr=settings.rb_vol_adj_median_atr,
        vol_adj_min=settings.rb_vol_adj_min,
        vol_adj_max=settings.rb_vol_adj_max,
        # v12.3: Direction Diversity
        dir_diversity_enabled=settings.rb_dir_diversity_enabled,
        max_same_direction=settings.rb_max_same_direction,
        # v12.6: Climax Volume Filter + ATR-Based Stop
        climax_enabled=settings.rb_climax_enabled,
        climax_vol_mult=settings.rb_climax_vol_mult,
        atr_stop_enabled=settings.rb_atr_stop_enabled,
        atr_stop_mult=settings.rb_atr_stop_mult,
        atr_stop_floor=settings.rb_atr_stop_floor,
        atr_stop_cap=settings.rb_atr_stop_cap,
    )

    scalper = RubberBandScalper(
        paper_mode=args.paper,
        config=config,
        strategy_name=args.strategy,
        vs_config=VolStraddleConfig(
            vr_short=settings.vs_vr_short,
            vr_long=settings.vs_vr_long,
            d_vr_threshold=settings.vs_d_vr_threshold,
            d_sl=settings.vs_d_sl,
            d_tp=settings.vs_d_tp,
            momentum_period=settings.vs_momentum_period,
            s_vr_threshold=settings.vs_s_vr_threshold,
            s_squeeze_dur=settings.vs_s_squeeze_dur,
            s_sl=settings.vs_s_sl,
            s_tp=settings.vs_s_tp,
            straddle_enabled=settings.vs_straddle_enabled,
            max_hold_15m=settings.vs_max_hold_15m,
            max_hold_1h=settings.vs_max_hold_1h,
            max_hold_4h=settings.vs_max_hold_4h,
            enable_15m=settings.vs_enable_15m,
            enable_1h=settings.vs_enable_1h,
            enable_4h=settings.vs_enable_4h,
            base_leverage=settings.vs_base_leverage,
            max_leverage=settings.vs_max_leverage,
            max_concurrent=args.max_slots if args.max_slots is not None else settings.vs_max_concurrent,
            symbol_whitelist=[s.strip() for s in (
                settings.vs_symbol_whitelist or settings.rb_symbol_whitelist
            ).split(',') if s.strip()],
            max_drawdown_pct=settings.max_drawdown_pct,
            soft_equity_scale=settings.rb_soft_equity_scale,
            soft_eq_dd1=settings.rb_soft_eq_dd1,
            soft_eq_scale1=settings.rb_soft_eq_scale1,
            soft_eq_dd2=settings.rb_soft_eq_dd2,
            soft_eq_scale2=settings.rb_soft_eq_scale2,
            soft_eq_dd3=settings.rb_soft_eq_dd3,
            soft_eq_scale3=settings.rb_soft_eq_scale3,
        ) if args.strategy == 'vol_straddle' else None,
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
