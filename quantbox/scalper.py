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
from quantbox.notifications.discord import DiscordNotifier
from quantbox.strategy.rubberband import (
    RubberBandConfig, RubberBandStrategy, calculate_position_size,
    compute_confidence_leverage,
)

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
    ):
        self.paper_mode = paper_mode
        self.config = config or RubberBandConfig()
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

        # Paper trading state
        self._paper_balance = settings.paper_balance

        # Risk tracking
        self._initial_balance: float = 0.0
        self._high_water_mark: float = 0.0
        self._daily_start_balance: float = 0.0
        self._daily_start_time: datetime = datetime.now(timezone.utc)
        self._drawdown_halt: bool = False
        self._daily_halt: bool = False

        # Symbol info cache (avoid repeated exchange_info calls)
        self._symbol_info_cache: dict[str, dict] = {}
        self._exchange_info_fetched: bool = False

        # Trade history
        self._trade_history: list[dict] = []
        self._cycle_count: int = 0

        # Stop flag (for testing)
        self._stop_requested: bool = False

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
        """Check drawdown and daily loss limits.

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

        # Daily loss check (reset at UTC midnight)
        now = datetime.now(timezone.utc)
        if now.date() > self._daily_start_time.date():
            self._daily_start_balance = balance
            self._daily_start_time = now
            self._daily_halt = False

        if self._daily_start_balance > 0:
            daily_loss = (
                (self._daily_start_balance - balance)
                / self._daily_start_balance
            )
            if daily_loss >= self.config.daily_loss_limit_pct:
                if not self._daily_halt:
                    logger.warning(
                        f"DAILY LOSS LIMIT: {daily_loss*100:.1f}% "
                        f"(start=${self._daily_start_balance:.2f})"
                    )
                    self._daily_halt = True
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

            # Filter exclusions
            result = [
                s for s in sorted_syms[:self.config.min_volume_rank]
                if s not in self.config.exclude_symbols
            ]

            # v10.0: Apply symbol whitelist filter
            if self.config.symbol_whitelist:
                whitelist_set = set(self.config.symbol_whitelist)
                result = [s for s in result if s in whitelist_set]
                logger.info(f"Universe: {len(result)} symbols (whitelist={len(whitelist_set)})")
            else:
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
        """Download recent klines.

        Args:
            symbol: Trading symbol.
            n_bars: Number of bars to fetch.
            interval: Candle interval ('15m', '4h', etc.).
            full: If True, returns (close, volume, taker_buy) tuple.

        Returns:
            Array of close prices (default), or (close, volume, taker_buy) tuple
            if full=True, or None on failure.
        """
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=n_bars,
            )
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
        leverage: int = 1,
    ) -> dict | None:
        """Place order via LimitOrderChaser.

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
            # Set leverage
            try:
                self.client.futures_change_leverage(
                    symbol=symbol, leverage=leverage,
                )
            except BinanceAPIException as e:
                if '-4028' in str(e):
                    # Leverage too high — try lower
                    try:
                        self.client.futures_change_leverage(
                            symbol=symbol, leverage=min(leverage, 20),
                        )
                    except Exception:
                        pass
                else:
                    logger.warning(f"[{symbol}] Leverage set failed: {e}")

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
            )

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

        side = 'BUY' if signal == 1 else 'SELL'
        lev_int = max(1, min(int(leverage), 20))

        result = self._place_order(symbol, side, quantity, leverage=lev_int)
        if result is None:
            return False

        fill_price = float(result['avgPrice'])
        fill_qty = float(result['executedQty'])

        # Compute effective stop (HTF agree parity)
        eff_stop = self.strategy.get_effective_stop_mult(close_prices)

        # Record in strategy
        self.strategy.open_position(
            symbol=symbol,
            signal=signal,
            entry_price=fill_price,
            notional=fill_qty * fill_price,
            leverage=lev_int,
            move_size=move_size,
            entry_stop_mult=eff_stop,
        )

        # Paper balance adjustment
        if self.paper_mode:
            cost = fill_qty * fill_price * cfg.cost_per_side
            self._paper_balance -= cost

        logger.info(
            f"[{symbol}] OPENED {'LONG' if signal == 1 else 'SHORT'} "
            f"qty={fill_qty:.6f} @ ${fill_price:.4f} "
            f"(notional=${fill_qty * fill_price:.2f}, "
            f"lev={lev_int}x, move={move_size*100:.2f}%)"
        )

        self.discord.send(embed={
            'title': f"{'🟢' if signal == 1 else '🔴'} {symbol} "
                     f"{'LONG' if signal == 1 else 'SHORT'}",
            'color': 0x00C853 if signal == 1 else 0xFF1744,
            'fields': [
                {'name': 'Price', 'value': f'${fill_price:.4f}', 'inline': True},
                {'name': 'Size', 'value': f'${fill_qty * fill_price:.2f}',
                 'inline': True},
                {'name': 'Leverage', 'value': f'{lev_int}x', 'inline': True},
                {'name': 'Move', 'value': f'{move_size*100:.2f}%', 'inline': True},
            ],
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

        fraction = self.config.partial_exit_fraction
        full_quantity = pos['notional'] / pos['entry_price']
        close_qty = self._round_qty(full_quantity * fraction, symbol)
        if close_qty <= 0:
            return False

        # Reverse the entry side to close
        side = 'SELL' if pos['signal'] == 1 else 'BUY'

        result = self._place_order(symbol, side, close_qty)
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
            'fields': [
                {'name': 'PnL', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': 'Net $', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': 'Remaining', 'value': f'${pos["notional"]:.2f}',
                 'inline': True},
                {'name': 'Stop', 'value': 'Break-even', 'inline': True},
            ],
        })
        return True

    def _close_position(
        self, symbol: str, reason: str, current_price: float,
    ) -> bool:
        """Close an existing position.

        Returns True if closed successfully.
        """
        pos = self.strategy.close_position(symbol)
        if pos is None:
            return False

        quantity = pos['notional'] / pos['entry_price']
        quantity = self._round_qty(quantity, symbol)
        if quantity <= 0:
            return False

        # Reverse the entry side
        side = 'SELL' if pos['signal'] == 1 else 'BUY'

        result = self._place_order(symbol, side, quantity)
        if result is None:
            # Failed to close — re-add position for retry
            self.strategy.open_position(
                symbol=symbol,
                signal=pos['signal'],
                entry_price=pos['entry_price'],
                notional=pos['notional'],
                leverage=pos['leverage'],
                move_size=pos['move_size'],
                entry_stop_mult=pos.get('entry_stop_mult'),
            )
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

        emoji = '💰' if net_pnl > 0 else '💸'
        logger.info(
            f"[{symbol}] CLOSED ({reason}): "
            f"{'LONG' if pos['signal'] == 1 else 'SHORT'} "
            f"PnL={pnl_pct*100:+.2f}% (${net_pnl:+.2f}) "
            f"held={pos['bars_held']} bars"
        )
        self.discord.send(embed={
            'title': f"{emoji} {symbol} CLOSED ({reason})",
            'color': 0x00C853 if net_pnl > 0 else 0xFF1744,
            'fields': [
                {'name': 'PnL', 'value': f'{pnl_pct*100:+.2f}%', 'inline': True},
                {'name': 'Net $', 'value': f'${net_pnl:+.2f}', 'inline': True},
                {'name': 'Bars', 'value': str(pos['bars_held']), 'inline': True},
                {'name': 'Entry', 'value': f'${pos["entry_price"]:.4f}',
                 'inline': True},
                {'name': 'Exit', 'value': f'${fill_price:.4f}', 'inline': True},
            ],
        })
        return True

    # ==================== Main Cycle ====================

    def _cycle(self, symbols: list[str]) -> None:
        """Process one 15m bar across all symbols.

        1. Get BTC klines for regime filter
        2. Manage existing positions
        3. Check for new entries
        4. Save state
        """
        self._cycle_count += 1
        balance = self.get_balance()

        logger.info(
            f"── Cycle #{self._cycle_count} | "
            f"Balance=${balance:.2f} | "
            f"Positions={self.strategy.n_active}/{self.config.max_concurrent} ──"
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

        # 3b. v7.0: Fetch BTC 4h klines for MTF confirmation
        btc_close_4h = None
        if self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
            btc_close_4h = self.download_klines('BTCUSDT', 100, interval='4h')

        # 4. Process existing positions
        for sym in list(self.strategy.active_positions):
            close = self.download_klines(sym, 200)
            if close is None:
                continue

            current_price = close[-1]
            exit_reason = self.strategy.update_position(sym, current_price)
            if exit_reason:
                if exit_reason == 'partial_close':
                    # v4.3: partial close 50%, keep remaining with BE stop
                    self._partial_close_position(sym, current_price)
                else:
                    self._close_position(sym, exit_reason, current_price)

        # 5. Look for new entries (skip if risk blocked)
        if not risk_blocked and not self._daily_halt:
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

            # v7.0: Download 4h data for MTF confirmation
            if self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
                close_4h = self.download_klines(sym, 100, interval='4h')
                if close_4h is not None:
                    self.strategy.update_htf_data(sym, close_4h)

            signal = self.strategy.on_bar(sym, close, btc_close)
            if signal != 0:
                # Measure move strength for prioritization
                move = abs(
                    close[-1] / close[-self.config.lookback - 1] - 1.0
                )
                candidates.append((sym, signal, move, close))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

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
            'daily_start_balance': self._daily_start_balance,
            'daily_start_time': self._daily_start_time.isoformat(),
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
        self._daily_start_balance = state.get('daily_start_balance', 0)
        self._trade_history = state.get('trade_history', [])
        self._cycle_count = state.get('cycle_count', 0)

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
            )
            # Restore bars_held
            if sym in self.strategy._positions:
                self.strategy._positions[sym]['bars_held'] = pos.get(
                    'bars_held', 0
                )
                self.strategy._positions[sym]['best_pnl'] = pos.get(
                    'best_pnl', 0.0
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

        self._cycle(symbols)

    def run_continuous(self) -> None:
        """Run the scalper continuously, cycling every 15 minutes.

        Waits for bar close, processes all symbols, repeats.
        """
        # Setup
        balance = self.get_balance()
        self._initial_balance = balance
        self._high_water_mark = balance
        self._daily_start_balance = balance
        self._load_state()

        mode_str = "PAPER" if self.paper_mode else "LIVE"
        logger.info(f"Starting continuous trading ({mode_str})")
        logger.info(f"Initial balance: ${balance:.2f}")

        self.discord.send(embed={
            'title': '🚀 Scalper Started',
            'color': 0x2979FF,
            'fields': [
                {'name': 'Mode', 'value': mode_str, 'inline': True},
                {'name': 'Balance', 'value': f'${balance:.2f}', 'inline': True},
                {'name': 'Max Slots',
                 'value': str(self.config.max_concurrent), 'inline': True},
            ],
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
                time.sleep(60)  # Wait before retry

        # Clean shutdown
        logger.info("Shutting down...")
        self._save_state()

        total_pnl = sum(t['pnl_usd'] for t in self._trade_history)
        self.discord.send(embed={
            'title': '🛑 Scalper Stopped',
            'color': 0xFF1744,
            'fields': [
                {'name': 'Cycles', 'value': str(self._cycle_count),
                 'inline': True},
                {'name': 'Trades', 'value': str(len(self._trade_history)),
                 'inline': True},
                {'name': 'PnL', 'value': f'${total_pnl:+.2f}',
                 'inline': True},
            ],
        })

    def _wait_for_bar(self) -> None:
        """Sleep until the next 15-minute bar closes (+5s buffer)."""
        now = datetime.now(timezone.utc)
        minute = now.minute
        second = now.second

        # Next 15m boundary
        next_boundary = ((minute // 15) + 1) * 15
        if next_boundary >= 60:
            target = now.replace(
                minute=0, second=5, microsecond=0
            ) + timedelta(hours=1)
        else:
            target = now.replace(
                minute=next_boundary, second=5, microsecond=0
            )

        sleep_secs = (target - now).total_seconds()
        if sleep_secs > 0:
            logger.debug(f"Waiting {sleep_secs:.0f}s for next bar close")
            time.sleep(sleep_secs)


# ──────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QuantBox Rubber Band Scalper'
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
        '--lookback', type=int, default=8,
        help='Signal lookback bars (default: 8)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.025,
        help='Min move to trigger (default: 0.025 = 2.5%%)'
    )
    parser.add_argument(
        '--max-slots', type=int, default=5,
        help='Max concurrent positions (default: 5)'
    )
    args = parser.parse_args()

    # Build config from .env settings (pydantic-settings) + CLI overrides
    settings = get_settings()
    config = RubberBandConfig(
        lookback=args.lookback or settings.rb_lookback,
        threshold=args.threshold or settings.rb_threshold,
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
        max_concurrent=args.max_slots,
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

    scalper = RubberBandScalper(paper_mode=args.paper, config=config)

    if args.once:
        scalper.run_once()
    else:
        scalper.run_continuous()


if __name__ == '__main__':
    main()
