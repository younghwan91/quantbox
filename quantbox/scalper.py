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
import logging
import logging.handlers
import math
import time
from datetime import datetime, timezone

import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

from quantbox.core.limit_chaser import (
    ChaseConfig,
    ChaseFillStatus,
    ChaseResult,
    LimitOrderChaser,
)
from quantbox.core.settings import get_settings
from quantbox.core.terminal_ui import (
    format_cycle_header,
    format_scan_summary,
    format_session_summary,
)
from quantbox.notifications.bot import (
    DiscordBotRunner,
)
from quantbox.notifications.discord import DiscordNotifier
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_atr_pct,
)
from quantbox.strategy.elliott import ElliottConfig, ElliottStrategy
from quantbox.strategy.vol_straddle import VolStraddleConfig, VolStraddleStrategy
from quantbox.core.bot_utils import GIT_COMMIT, setup_logging
from quantbox.core.commands import CommandsMixin
from quantbox.core.loop_mixin import LoopMixin
from quantbox.core.scalper_state import (
    PAPER_STATE_FILE,
    STATE_FILE,
    load_scalper_state,
    save_scalper_state,
)
from quantbox.execution.brackets import BracketMixin
from quantbox.execution.data_mixin import DataMixin
from quantbox.execution.positions import PositionMixin
from quantbox.execution.reconciliation import ReconciliationMixin
from quantbox.core.trade_journal import TradeJournal
from quantbox.core.divergence_monitor import DivergenceMonitor

load_dotenv()

# ──────────────────────── Logging ──────────────────────────────────────

logger = logging.getLogger("Scalper")


# ──────────────── Diagnostic Logger ─────────────────────
from quantbox.utils.diagnostics import _diag_logger  # noqa: E402

# ──────────────────────── Scalper Class ────────────────────────────────


class RubberBandScalper(
    LoopMixin,
    DataMixin,
    BracketMixin,
    ReconciliationMixin,
    PositionMixin,
    CommandsMixin,
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
        strategy_name: str = "rubberband",
        elliott_config: ElliottConfig | None = None,
        vs_config: VolStraddleConfig | None = None,
        journal_db_path: str | None = None,
    ):
        self.paper_mode = paper_mode
        self.config = config or RubberBandConfig()
        self.strategy_name = strategy_name

        if strategy_name == "elliott":
            self.elliott_config = elliott_config or ElliottConfig()
            self.strategy = ElliottStrategy(self.elliott_config)
        elif strategy_name == "vol_straddle":
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

        # Daily loss circuit breaker
        self._daily_realized_pnl: float = 0.0  # cumulative today's closed pnl (USD)
        self._daily_loss_date: str = ""  # UTC date when tracking started (YYYY-MM-DD)
        self._daily_loss_blocked: bool = False  # True = new entries blocked for today

        # Symbol info cache (avoid repeated exchange_info calls)
        self._symbol_info_cache: dict[str, dict] = {}
        self._exchange_info_fetched: bool = False

        # Trade history
        self._trade_history: list[dict] = []
        self._cycle_count: int = 0

        # Trade journal (persistent SQLite)
        _session_str = "paper" if paper_mode else "live"
        if journal_db_path is not None:
            _db_path = journal_db_path
        else:
            _db_path = "data/journal_paper.sqlite" if paper_mode else "data/journal.sqlite"
        self._journal = TradeJournal(db_path=_db_path, session=_session_str)

        # Live-Backtest divergence monitor
        _vs_cfg = self.vs_config if strategy_name == "vol_straddle" else None
        self._divergence_monitor = DivergenceMonitor(
            min_trades=_vs_cfg.divergence_min_trades if _vs_cfg else 30,
            pf_threshold=_vs_cfg.divergence_pf_threshold if _vs_cfg else 0.80,
            wr_gap_pp=_vs_cfg.divergence_wr_gap_pp if _vs_cfg else 10.0,
            cooldown_hours=_vs_cfg.divergence_cooldown_hours if _vs_cfg else 24.0,
        )

        # Control flags
        self._stop_requested: bool = False
        self._paused: bool = False

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"RubberBandScalper initialized ({mode_str}) strategy={strategy_name}")
        if strategy_name == "vol_straddle":
            vs = self.vs_config
            logger.info(
                f"Config: VR({vs.vr_short}/{vs.vr_long}) "
                f"D_threshold={vs.d_vr_threshold}, "
                f"SL={vs.d_sl * 100:.1f}%, TP={vs.d_tp * 100:.1f}%, "
                f"TF={'15m' if vs.enable_15m else ''}"
                f"{'|1h' if vs.enable_1h else ''}"
                f"{'|4h' if vs.enable_4h else ''}"
            )
        else:
            _sl_str = (
                f"ATR×{self.config.atr_stop_mult}[{self.config.atr_stop_floor * 100:.1f}%-{self.config.atr_stop_cap * 100:.0f}%]"
                if self.config.atr_stop_enabled
                else f"{self.config.stop_mult}×move"
            )
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
        if self.strategy_name == "vol_straddle":
            return self.vs_config
        return self.config

    def get_balance(self) -> float:
        """Get current USDT balance."""
        if self.paper_mode:
            return self._paper_balance
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["balance"])
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
        # v12.5.1: Return last known balance instead of 0.0 to prevent
        # false drawdown halt on transient API failures.
        if hasattr(self, "_last_known_balance") and self._last_known_balance > 0:
            logger.warning(f"Using last known balance ${self._last_known_balance:.2f} due to API failure")
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
                        f"DRAWDOWN HALT: {dd * 100:.1f}% DD (HWM=${self._high_water_mark:.2f}, now=${balance:.2f})"
                    )
                    self.discord.send(
                        content=(f"🚨 **DRAWDOWN HALT** — {dd * 100:.1f}% drawdown triggered. Closing all positions.")
                    )
                    self._drawdown_halt = True
                return True

        return False

    # ==================== Order Execution ====================

    def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int = 1,
        reduce_only: bool = False,
        position_side: str | None = None,
    ) -> dict | None:
        """Place order via LimitOrderChaser.

        Args:
            symbol: Trading symbol.
            side: 'BUY' or 'SELL'.
            quantity: Order quantity.
            leverage: Leverage to set before order.
            reduce_only: If True, pass reduceOnly=True to exchange.
                Must be True when closing positions to avoid margin errors.
            position_side: Hedge Mode position side ('LONG'/'SHORT').
                When set, positionSide is sent instead of reduceOnly.
                None = One-Way Mode (existing behavior).

        Returns dict with avgPrice, executedQty on success, None on failure.
        """
        if self.paper_mode:
            # Simulate fill at current market price
            try:
                book = self.client.futures_orderbook_ticker(symbol=symbol)
                price = float(book["askPrice"] if side == "BUY" else book["bidPrice"])
            except Exception:
                return None
            return {
                "avgPrice": str(price),
                "executedQty": str(quantity),
                "status": "FILLED",
                "makerFill": True,
            }

        try:
            # v12.4.7: Skip leverage change for reduce-only orders.
            # Changing leverage on an open position can fail with
            # -2028 "insufficient margin balance" if the new leverage
            # is lower than what the position requires.  reduce_only
            # orders don't change margin requirements.
            #
            # v14.4.2: Also skip leverage change when leverage==1 (default).
            # In Hedge Mode, close orders pass reduce_only=False (positionSide
            # is used instead), so reduce_only alone is not a sufficient guard.
            # Close / partial-close / add-to-position calls do NOT pass
            # leverage explicitly, so it defaults to 1.  Attempting to set
            # leverage=1 on a live 30x position triggers -2028.
            # Only entry orders pass leverage > 1 explicitly.
            if not reduce_only and leverage > 1:
                # Set leverage — v11.1: verify it was applied correctly
                actual_leverage = leverage
                try:
                    resp = self.client.futures_change_leverage(
                        symbol=symbol,
                        leverage=leverage,
                    )
                    actual_leverage = int(resp.get("leverage", leverage))
                    if actual_leverage != leverage:
                        logger.warning(
                            f"[{symbol}] Leverage mismatch: requested={leverage}x, actual={actual_leverage}x"
                        )
                except BinanceAPIException as e:
                    if "-4028" in str(e):
                        # Leverage too high for this symbol — try reduced
                        fallback_lev = max(20, leverage // 2)
                        try:
                            resp = self.client.futures_change_leverage(
                                symbol=symbol,
                                leverage=fallback_lev,
                            )
                            actual_leverage = int(resp.get("leverage", fallback_lev))
                            logger.warning(
                                f"[{symbol}] Leverage {leverage}x not supported, fell back to {actual_leverage}x"
                            )
                        except Exception:
                            logger.error(
                                f"[{symbol}] Leverage fallback also failed — ABORTING order to prevent unknown leverage"
                            )
                            return None
                    else:
                        logger.error(f"[{symbol}] Leverage set failed: {e} — ABORTING order")
                        return None

            info = self.get_symbol_info(symbol)
            tick_size = info.get("tick_size", 0.01)
            price_precision = info.get("price_precision", 2)
            step_size = info.get("step_size", 0.001)
            qty_precision = info.get("quantity_precision", 3)

            result: ChaseResult = self._chaser.chase_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                tick_size=tick_size,
                price_precision=price_precision,
                step_size=step_size,
                quantity_precision=qty_precision,
                reduce_only=reduce_only,
                position_side=position_side,
            )

            if result.status == ChaseFillStatus.NO_POSITION:
                logger.warning(f"[{symbol}] NO POSITION on exchange — reduceOnly rejected. Position already closed.")
                return {"status": "NO_POSITION"}

            if result.status in (ChaseFillStatus.FILLED, ChaseFillStatus.FALLBACK_MARKET):
                if result.avg_price <= 0:
                    logger.error(f"[{symbol}] FILLED but avg_price=0 → treating as FAILED")
                    return None
                fill_type = "MAKER" if result.maker_fill else "TAKER"
                logger.info(
                    f"[{symbol}] {fill_type} {side} "
                    f"{result.filled_quantity:.6f} @ {result.avg_price:.4f} "
                    f"(chase={result.chase_count})"
                )
                return {
                    "avgPrice": str(result.avg_price),
                    "executedQty": str(result.filled_quantity),
                    "status": "FILLED",
                    "makerFill": result.maker_fill,
                }
            else:
                logger.error(f"[{symbol}] Order FAILED: {side} {quantity:.6f} (status={result.status.value})")
                return None

        except Exception as e:
            logger.error(f"[{symbol}] Order error: {e}")
            return None

    def _round_qty(self, qty: float, symbol: str) -> float:
        """Round quantity to exchange step_size."""
        info = self.get_symbol_info(symbol)
        step = info.get("step_size", 0.001)
        prec = info.get("quantity_precision", 3)
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
        cum_pnl = sum(t["pnl_usd"] for t in self._trade_history)
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
            # Close any remaining positions
            for pos_key in list(self.strategy.active_positions):
                api_sym = pos_key.split(":")[0] if ":" in pos_key else pos_key
                close = self.download_klines(api_sym, 2)
                price = close[-1] if close is not None else 0
                self._close_position(pos_key, "drawdown_halt", price)

            # v14.0.1: Auto-recover when DD drops below threshold
            # after all positions are closed.
            if self.strategy.n_active == 0:
                hwm = self._high_water_mark if self._high_water_mark > 0 else balance
                dd = (hwm - balance) / hwm if hwm > 0 else 0.0
                if dd < self._cfg.max_drawdown_pct:
                    logger.info(
                        f"DD recovered to {dd * 100:.1f}% "
                        f"(threshold={self._cfg.max_drawdown_pct * 100:.0f}%) "
                        f"— resuming trading"
                    )
                    self.discord.send(
                        content=(
                            f"\u2705 **DD Recovered** — "
                            f"{dd * 100:.1f}% (below {self._cfg.max_drawdown_pct * 100:.0f}% threshold). "
                            f"Resuming trading."
                        )
                    )
                    self._drawdown_halt = False
                else:
                    logger.info(
                        f"DD halt active — {dd * 100:.1f}% DD "
                        f"(need <{self._cfg.max_drawdown_pct * 100:.0f}% to resume). "
                        f"All positions closed, waiting for HWM reset on restart."
                    )
                    self._save_state()
                    return
            else:
                self._save_state()
                return

        # 3. Fetch BTC klines for regime filter
        btc_close = self.download_klines("BTCUSDT", 200)

        # 3b. v7.0: Fetch BTC 4h klines for MTF / Elliott macro bias
        btc_close_4h = None
        if self.strategy_name == "elliott":
            # Elliott needs 600+ bars of BTC 4h for 200/500 EMA macro bias
            btc_close_4h = self.download_klines("BTCUSDT", 600, interval="4h")
            if btc_close_4h is not None:
                self.strategy.update_btc_data(btc_close_4h)
        elif self.strategy_name == "vol_straddle":
            # VolStraddle doesn't use BTC as filter — skip 4h BTC fetch
            pass
        elif self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
            btc_close_4h = self.download_klines("BTCUSDT", 100, interval="4h")

        # 3c. v11.2: Sync server-side TP/SL orders (check if filled intra-bar)
        self._sync_server_orders()

        # v12.5.1: Reconcile positions with exchange AFTER sync
        if not self.paper_mode:
            self._reconcile_positions()

        # v12.5.1: Log position status AFTER sync (SL/TP verified)
        for pos_key, pos in self.strategy.active_positions.items():
            api_sym = pos.get("symbol", pos_key.split(":")[0])
            try:
                mark = self.client.futures_mark_price(symbol=api_sym)
                cur_px = float(mark["markPrice"])
            except Exception:
                cur_px = pos.get("entry_price", 0)
            entry_px = pos.get("entry_price", 0)
            if entry_px > 0:
                if pos["signal"] == 1:
                    upnl_pct = (cur_px - entry_px) / entry_px * 100
                else:
                    upnl_pct = (entry_px - cur_px) / entry_px * 100
            else:
                upnl_pct = 0.0
            sl_id = pos.get("server_sl_order_id")
            tp_id = pos.get("server_tp_order_id")
            # Use bars_held+1 to match the value AFTER this cycle's update,
            # consistent with the Summary section printed later.
            bars = pos.get("bars_held", 0) + 1
            direction = "L" if pos.get("signal") == 1 else "S"
            logger.info(
                f"  [{pos_key}] {direction} entry=${entry_px:.4f} "
                f"mark=${cur_px:.4f} uPnL={upnl_pct:+.2f}% bars={bars} "
                f"SL={'✓' if sl_id else '✗'} TP={'✓' if tp_id else '✗'}"
            )

        # 4. Process existing positions (always, even when paused)
        for pos_key in list(self.strategy.active_positions):
            # Extract Binance symbol from compound key (e.g., BTCUSDT:LONG → BTCUSDT)
            api_sym = pos_key.split(":")[0] if ":" in pos_key else pos_key

            # v12.3.2: Download full OHLC for active positions to compute
            # fresh ATR. Previously only close was downloaded, so runner
            # trailing used stale ATR from entry time (on_bar never called
            # for symbols with open positions).
            result_full = self.download_klines(api_sym, 200, full=True)
            if result_full is None:
                continue
            close, high, low, _vol, _tb = result_full

            current_price = close[-1]

            # v12.3.2: Compute FRESH ATR for PBT runner trailing
            # VolStraddle uses R43 trailing stop (client-side, no ATR needed).
            if self.strategy_name == "vol_straddle":
                current_atr = 0.02
            elif self.config.atr_period > 0 and len(close) >= self.config.atr_period:
                atr_arr = compute_atr_pct(high, low, close, self.config.atr_period)
                atr_val = float(atr_arr[-1])
                current_atr = atr_val if not np.isnan(atr_val) else 0.02
            else:
                current_atr = 0.02

            exit_reason = self.strategy.update_position(
                pos_key,
                current_price,
                current_atr_pct=current_atr,
                current_high=float(high[-1]),
                current_low=float(low[-1]),
            )
            if exit_reason:
                logger.info(f"  [{pos_key}] Exit signal: {exit_reason} @ ${current_price:.4f}")
                # Straddle leg tracking: log which leg exits first
                if ":" in pos_key and self.strategy_name == "vol_straddle":
                    api_sym, side = pos_key.split(":", 1)
                    other_side = "SHORT" if side == "LONG" else "LONG"
                    other_key = f"{api_sym}:{other_side}"
                    other_pos = self.strategy._positions.get(other_key)
                    if other_pos:
                        entry_o = other_pos.get("entry_price", 0)
                        if entry_o > 0:
                            if other_pos["signal"] == 1:
                                o_gain = (current_price - entry_o) / entry_o
                            else:
                                o_gain = (entry_o - current_price) / entry_o
                            logger.info(
                                f"  [STRADDLE] {side} leg closed ({exit_reason}), "
                                f"{other_side} still open "
                                f"uPnL={o_gain * 100:+.2f}% "
                                f"bars={other_pos.get('bars_held', 0)}"
                            )
                    else:
                        logger.info(f"  [STRADDLE] Last leg ({side}) closed ({exit_reason}) — straddle fully unwound")
                if exit_reason == "partial_close":
                    self._partial_close_position(pos_key, current_price)
                elif exit_reason == "confirm_entry":
                    # Elliott: upgrade 2% exploratory → add 8% primary
                    self._add_to_position(pos_key, current_price, balance)
                elif exit_reason == "pyramid":
                    # Elliott: pyramid using unrealized profits
                    self._pyramid_position(pos_key, current_price, balance)
                elif exit_reason == "cancel_exploration":
                    # Elliott: confirmation timed out → close pilot
                    self._close_position(pos_key, exit_reason, current_price)
                else:
                    self._close_position(pos_key, exit_reason, current_price)
            else:
                pos = self.strategy._positions.get(pos_key)
                if pos:
                    _mh = pos.get("max_hold", self.config.max_hold)
                    logger.debug(
                        f"  [{pos_key}] HOLD: bars={pos.get('bars_held', 0)}/{_mh} "
                        f"best_pnl={pos.get('best_pnl', 0) * 100:.2f}%"
                    )
                    # v12.3.3: Ratchet server SL for runners to protect profits
                    if pos.get("partial_closed") and self.config.breakeven_after_partial and not self.paper_mode:
                        self._ratchet_runner_sl(pos_key, pos)
                    # r65: Ratchet server SL to trail_stop_gain (bar-close gap fix)
                    if (
                        pos.get("trail_active")
                        and not self.paper_mode
                        and self.strategy_name == "vol_straddle"
                        and getattr(self.vs_config, "trail_server_ratchet_enabled", True)
                    ):
                        self._ratchet_trail_sl(pos_key, pos)

        # 5. Look for new entries (skip if risk blocked or paused)
        if self._paused:
            logger.info("⏸️ Paused — skipping new entries")
        elif not risk_blocked:
            # ── G7: Daily Loss Circuit Breaker ──
            if self.strategy_name == "vol_straddle" and self.vs_config.daily_loss_limit_pct > 0:
                today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if self._daily_loss_date != today_utc:
                    # New UTC day — reset counter
                    self._daily_loss_date = today_utc
                    self._daily_realized_pnl = 0.0
                    self._daily_loss_blocked = False
                    logger.info(f"G7 DailyBreaker: reset for {today_utc}")
                # Accumulate closed PnL from today's trades
                self._daily_realized_pnl = sum(
                    t["pnl_usd"] for t in self._trade_history if t.get("time", "").startswith(today_utc)
                )
                limit_usd = balance * self.vs_config.daily_loss_limit_pct
                if not self._daily_loss_blocked and self._daily_realized_pnl < -limit_usd:
                    self._daily_loss_blocked = True
                    logger.warning(
                        f"⛔ G7 DailyBreaker TRIGGERED: "
                        f"today PnL=${self._daily_realized_pnl:.2f} < "
                        f"-${limit_usd:.2f} ({self.vs_config.daily_loss_limit_pct * 100:.0f}% limit). "
                        f"No new entries until UTC midnight."
                    )
                    self.discord.send(
                        content=(
                            f"⛔ **Daily Loss Limit Hit** — "
                            f"Realized PnL today: ${self._daily_realized_pnl:.2f} "
                            f"(limit -${limit_usd:.2f}). "
                            f"No new entries until midnight UTC."
                        )
                    )
                if self._daily_loss_blocked:
                    logger.info(
                        f"G7 DailyBreaker active: today PnL=${self._daily_realized_pnl:.2f}. Skipping new entries."
                    )
            if not self._daily_loss_blocked:
                available_slots = self._cfg.max_concurrent - self.strategy.n_active
                if available_slots > 0:
                    self._scan_entries(symbols, btc_close, balance, available_slots, btc_close_4h=btc_close_4h)

        # 6. Save state
        self._save_state()

        # 8. Divergence check (every 10 trades)
        self._check_divergence()

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
            n_bars = 500 if self.strategy_name == "vol_straddle" else 200
            result = self.download_klines(sym, n_bars, full=True)
            if result is None:
                continue
            close, high, low, volume, taker_buy = result
            if len(close) < self._cfg.lookback + 1:
                continue

            # Wire up volume/taker/high/low data for v9 confluence scoring
            self.strategy.update_market_data(sym, volume, taker_buy, high, low)

            # v7.0: Download 4h data for MTF confirmation / Elliott wave detection
            if self.strategy_name == "elliott":
                # Elliott needs 600+ bars of 4h OHLC for 500 EMA + ZigZag
                result_4h = self.download_klines(sym, 600, interval="4h", full=True)
                if result_4h is not None:
                    c4h, h4h, l4h, _, _ = result_4h
                    self.strategy.update_htf_data(sym, c4h, h4h, l4h)
            elif self.strategy_name == "vol_straddle":
                # VolStraddle needs full 4h OHLCV for VR + MA Cluster.
                # MA Cluster uses SMA(200) → need ≥201 bars; 300 for safe warmup.
                n_bars_4h = 300 if (self.vs_config and self.vs_config.ma_cluster_enabled) else 100
                result_4h = self.download_klines(sym, n_bars_4h, interval="4h", full=True)
                if result_4h is not None:
                    c4h, h4h, l4h, _, _ = result_4h
                    self.strategy.update_htf_data(sym, c4h, h4h, l4h)
            elif self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
                close_4h = self.download_klines(sym, 100, interval="4h")
                if close_4h is not None:
                    self.strategy.update_htf_data(sym, close_4h)

            signal = self.strategy.on_bar(sym, close, btc_close)

            # v12.0: Collect full diagnostic context (read-only, after on_bar)
            if self.strategy_name != "elliott" and hasattr(self.strategy, "diagnose_bar"):
                diag = self.strategy.diagnose_bar(sym, close, btc_close)
                diag["signal_actual"] = signal  # cross-verify with on_bar
                scan_diags.append(diag)

            # Track 8-bar move for diagnostics
            _lb = self._cfg.lookback
            abs_move = abs(close[-1] / close[-_lb - 1] - 1.0)
            moves_seen.append((sym, abs_move))

            if signal != 0:
                # Measure move strength for prioritization
                move = abs(close[-1] / close[-_lb - 1] - 1.0)
                if signal == 2:
                    # Straddle: expand into LONG + SHORT candidates
                    candidates.append((sym, 1, move, close))
                    candidates.append((sym, -1, move, close))
                else:
                    candidates.append((sym, signal, move, close))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

        # ── Diagnostic: log scan summary every cycle ──
        top_movers = sorted(moves_seen, key=lambda x: x[1], reverse=True)[:3]
        if self.strategy_name == "vol_straddle":
            # VR strategy: show VR threshold + actual VR values
            print(format_scan_summary(len(candidates), top_movers, 0.0))
            if candidates:
                cand_strs = [f"{s}({m * 100:.1f}%)" for s, _, m, _ in candidates[:3]]
                logger.info(f"🎯 {len(candidates)} candidates: {', '.join(cand_strs)}")
            else:
                if moves_seen:
                    # Show top 3 moves with VR for all TFs
                    top3_with_vr = []
                    for s, m in top_movers:
                        vr_all = self.strategy.get_vr_all_tf(s)
                        vr_str = f"15m={vr_all['15m']:.2f}/1h={vr_all['1h']:.2f}/4h={vr_all['4h']:.2f}"
                        top3_with_vr.append(f"{s}({m * 100:.2f}%, {vr_str})")
                    logger.info(
                        f"No candidates — top moves: {', '.join(top3_with_vr)} "
                        f"(threshold VR<{self.vs_config.d_vr_threshold} any TF)"
                    )

                    # Count symbols near squeeze thresholds for strategy health
                    near_d = 0  # VR < d_threshold + 0.05
                    near_s = 0  # VR < s_threshold + 0.05
                    in_squeeze = 0  # VR < d_threshold
                    global_min_vr = 999.0
                    closest_sym = ""
                    for s, _ in moves_seen:
                        vr_all = self.strategy.get_vr_all_tf(s)
                        min_vr = min(vr_all.values())
                        if min_vr < global_min_vr:
                            global_min_vr = min_vr
                            closest_sym = s
                        if min_vr < self.vs_config.d_vr_threshold:
                            in_squeeze += 1
                        elif min_vr < self.vs_config.d_vr_threshold + 0.05:
                            near_d += 1
                        if min_vr < self.vs_config.s_vr_threshold + 0.05:
                            near_s += 1
                    logger.info(
                        f"Squeeze monitor: {in_squeeze} in squeeze, "
                        f"{near_d} near D threshold, {near_s} near S threshold "
                        f"(of {len(moves_seen)} symbols) "
                        f"min_vr={global_min_vr:.3f} ({closest_sym})"
                    )

                    # ── MA Cluster monitor (parallel to VR Squeeze monitor) ──
                    if (
                        self.vs_config
                        and self.vs_config.ma_cluster_enabled
                        and hasattr(self.strategy, "_ma_cluster_state")
                        and self.strategy._ma_cluster_state
                    ):
                        _mcs = self.strategy._ma_cluster_state
                        _in_cluster = {s: d for s, d in _mcs.items() if d["duration"] > 0}
                        _zombie = sum(1 for d in _in_cluster.values() if d["rejection"].startswith("zombie"))
                        _mom_fail = sum(1 for d in _in_cluster.values() if d["rejection"].startswith("momentum_fail"))
                        _cooldown = sum(1 for d in _in_cluster.values() if d["rejection"] == "cooldown")
                        _pre_dur = sum(1 for d in _in_cluster.values() if d["rejection"].startswith("pre_duration"))
                        _no_break = sum(1 for d in _in_cluster.values() if d["rejection"] == "no_breakout")
                        _fired = sum(1 for d in _in_cluster.values() if d["rejection"] == "fired")
                        _top3 = sorted(
                            _in_cluster.items(),
                            key=lambda x: x[1]["duration"],
                            reverse=True,
                        )[:3]
                        _top3_str = ", ".join(
                            f"{s}({d['duration']}bars,sp={d['spread']:.1f}%,{d['rejection']})" for s, d in _top3
                        )
                        logger.info(
                            f"MA Cluster monitor: {len(_in_cluster)} in cluster "
                            f"(zombie={_zombie}, mom_fail={_mom_fail}, "
                            f"cooldown={_cooldown}, pre_dur={_pre_dur}, "
                            f"no_break={_no_break}, fired={_fired}) "
                            f"(of {len(_mcs)} scanned) "
                            f"— top: {_top3_str or 'none'}"
                        )
        else:
            print(format_scan_summary(len(candidates), top_movers, self.config.threshold))
            if candidates:
                cand_strs = [f"{s}({m * 100:.1f}%)" for s, _, m, _ in candidates[:3]]
                logger.info(f"🎯 {len(candidates)} candidates: {', '.join(cand_strs)}")
            else:
                if moves_seen:
                    top3 = [f"{s}({m * 100:.2f}%)" for s, m in top_movers]
                    logger.info(
                        f"No candidates — top moves: {', '.join(top3)} "
                        f"(base >{self.config.threshold * 100:.1f}%, adaptive per-symbol)"
                    )

        # v12.0: Write structured diagnostics to JSONL
        if scan_diags:
            ts = datetime.now(timezone.utc).isoformat()
            _diag_logger.log_scan(self._cycle_count, ts, scan_diags)

        # ── G6: BTC Trend Gate — compute once, apply to all LONG candidates ──
        btc_ma = None
        if (
            self.strategy_name == "vol_straddle"
            and self.vs_config.btc_long_filter_enabled
            and btc_close is not None
            and len(btc_close) >= self.vs_config.btc_long_ma_period
        ):
            btc_ma = float(np.mean(btc_close[-self.vs_config.btc_long_ma_period :]))
            btc_cur = float(btc_close[-1])
            btc_above_ma = btc_cur >= btc_ma
            logger.info(
                f"G6 BTC Trend Gate: BTC=${btc_cur:.1f} "
                f"MA{self.vs_config.btc_long_ma_period}=${btc_ma:.1f} "
                f"→ {'ABOVE (LONG OK)' if btc_above_ma else 'BELOW (LONG BLOCKED)'}"
            )

        entries_made = 0
        for sym, signal, move, close in candidates:
            if entries_made >= max_entries:
                break

            # ── G6: BTC Trend Gate — block LONG when BTC below MA ──
            if signal == 1 and btc_ma is not None:
                btc_cur = float(btc_close[-1])
                if btc_cur < btc_ma:
                    logger.info(
                        f"[{sym}] G6 LONG suppressed: BTC ${btc_cur:.1f} < "
                        f"MA{self.vs_config.btc_long_ma_period} ${btc_ma:.1f}"
                    )
                    continue

            # ═══ v12.3 PARITY: Direction Diversity Filter ═══
            if self._cfg.dir_diversity_enabled:
                same_dir = sum(1 for p in self.strategy.active_positions.values() if p.get("signal") == signal)
                if same_dir >= self._cfg.max_same_direction:
                    logger.info(
                        f"[{sym}] DirDiversity blocked: "
                        f"{same_dir} {'LONG' if signal == 1 else 'SHORT'} "
                        f"already open (max={self._cfg.max_same_direction})"
                    )
                    continue

            if self._open_position(sym, signal, close, balance):
                entries_made += 1

        if entries_made > 0:
            logger.info(f"Opened {entries_made} new positions (from {len(candidates)} candidates)")

    # ==================== State Persistence ====================

    def _check_divergence(self) -> None:
        """Check live performance vs backtest baseline every 10 trades."""
        n_trades = len(self._trade_history)
        if n_trades == 0 or n_trades % 10 != 0:
            return
        try:
            stats = self._journal.get_stats(session=self._journal.session)
            alert = self._divergence_monitor.check(stats)
            if alert:
                logger.warning("Divergence alert fired: %s", alert.message)
                self.discord.notify_divergence_alert(alert)
        except Exception as exc:
            logger.warning("Divergence check error: %s", exc)

    def _save_state(self) -> None:
        """Persist positions and trade history."""
        state = {
            "version": "2.0.0",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "paper_mode": self.paper_mode,
            "paper_balance": self._paper_balance,
            "initial_balance": self._initial_balance,
            "high_water_mark": self._high_water_mark,
            "positions": self.strategy.active_positions,
            "trade_history": self._trade_history[-100:],  # Keep last 100
            "cycle_count": self._cycle_count,
        }
        # v14.0: Persist MA cluster cooldown for crash recovery.
        # Without this, a restart during an active cluster episode would
        # re-fire a duplicate signal on the same cluster.
        if self.strategy_name == "vol_straddle" and hasattr(self.strategy, "_ma_cluster_cooldown"):
            state["ma_cluster_cooldown"] = {sym: int(arr[0]) for sym, arr in self.strategy._ma_cluster_cooldown.items()}
        filepath = PAPER_STATE_FILE if self.paper_mode else STATE_FILE
        try:
            save_scalper_state(state, filepath=filepath)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _load_state(self) -> None:
        """Restore state from disk."""
        filepath = PAPER_STATE_FILE if self.paper_mode else STATE_FILE
        state = load_scalper_state(filepath=filepath)
        if state is None:
            return

        logger.info(f"Restoring state from {state.get('saved_at', 'unknown')}")

        self._paper_balance = state.get("paper_balance", self._paper_balance)
        self._initial_balance = state.get("initial_balance", 0)
        self._high_water_mark = state.get("high_water_mark", 0)

        self._trade_history = state.get("trade_history", [])
        self._cycle_count = 0  # Always reset per session (new log file each run)

        # Restore positions
        positions = state.get("positions", {})

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
            logger.info(f"No positions to restore — fresh session: HWM reset to ${balance:.2f}, trade history cleared")

        for sym, pos in positions.items():
            # Parse compound key (e.g. 'SUIUSDT:LONG') to extract the clean
            # Binance API symbol and position_side separately.
            # Without this, open_position stores the compound key as 'symbol',
            # causing -1022 signature errors when it's sent to the Binance API.
            parts = sym.rsplit(":", 1)
            if len(parts) == 2 and parts[1] in ("LONG", "SHORT"):
                api_sym = parts[0]
                restored_position_side = parts[1]
            else:
                api_sym = sym
                restored_position_side = pos.get("position_side")

            self.strategy.open_position(
                symbol=api_sym,
                signal=pos["signal"],
                entry_price=pos["entry_price"],
                notional=pos["notional"],
                leverage=pos.get("leverage", 1),
                move_size=pos.get("move_size", 0.03),
                entry_stop_mult=pos.get("entry_stop_mult"),
                server_sl_order_id=pos.get("server_sl_order_id"),
                server_tp_order_id=pos.get("server_tp_order_id"),
                position_side=restored_position_side,
                # Pass stored metadata so log shows correct strategy/SL/TP at restore
                sl=pos.get("sl"),
                tp=pos.get("tp"),
                tf=pos.get("tf"),
                max_hold=pos.get("max_hold"),
                strategy_type=pos.get("strategy_type", "vr_squeeze"),
                is_straddle=pos.get("is_straddle", False),
            )
            # Restore bars_held and PBT runner state
            if sym in self.strategy._positions:
                self.strategy._positions[sym]["bars_held"] = pos.get("bars_held", 0)
                self.strategy._positions[sym]["best_pnl"] = pos.get("best_pnl", 0.0)
                self.strategy._positions[sym]["partial_closed"] = pos.get("partial_closed", False)
                # v12.3 PBT: Restore runner trailing state
                # Without these, a restarted runner loses its breakeven
                # floor and peak tracking — violating PBT safety promise.
                self.strategy._positions[sym]["highest_since_tp1"] = pos.get("highest_since_tp1", 0.0)
                self.strategy._positions[sym]["trail_floor"] = pos.get("trail_floor", None)
                self.strategy._positions[sym]["confidence"] = pos.get("confidence", 0)
                # v12.3.3: Restore server SL ratchet state for runners.
                # Without these, a restarted runner triggers a redundant
                # ratchet API call and has a brief inconsistency window.
                self.strategy._positions[sym]["_computed_trail_stop"] = pos.get("_computed_trail_stop")
                self.strategy._positions[sym]["_server_sl_pct"] = pos.get("_server_sl_pct", 0.0)
                # R43: Trailing stop state for directional trades
                self.strategy._positions[sym]["trail_active"] = pos.get("trail_active", False)
                self.strategy._positions[sym]["peak_gain"] = pos.get("peak_gain", 0.0)
                self.strategy._positions[sym]["is_straddle"] = pos.get("is_straddle", False)
                # v14.0: Restore per-signal fields (sl/tp/tf/max_hold/strategy_type).
                # Without this, crash recovery defaults to VR squeeze params
                # for ALL positions — breaking MA cluster SL/TP/max_hold/exit routing.
                self.strategy._positions[sym]["strategy_type"] = pos.get("strategy_type", "vr_squeeze")
                self.strategy._positions[sym]["sl"] = pos.get("sl", self.strategy._positions[sym].get("sl"))
                self.strategy._positions[sym]["tp"] = pos.get("tp", self.strategy._positions[sym].get("tp"))
                self.strategy._positions[sym]["tf"] = pos.get("tf", self.strategy._positions[sym].get("tf"))
                self.strategy._positions[sym]["max_hold"] = pos.get(
                    "max_hold", self.strategy._positions[sym].get("max_hold")
                )

        logger.info(f"Restored {len(positions)} positions, {len(self._trade_history)} trade records")

        # v14.0: Restore MA cluster cooldown state
        if self.strategy_name == "vol_straddle" and hasattr(self.strategy, "_ma_cluster_cooldown"):
            cooldown_data = state.get("ma_cluster_cooldown", {})
            for sym, val in cooldown_data.items():
                arr = np.zeros(1, dtype=np.int64)
                arr[0] = val
                self.strategy._ma_cluster_cooldown[sym] = arr
            if cooldown_data:
                logger.info(f"Restored MA cluster cooldown for {len(cooldown_data)} symbols")

    # ==================== Entry Points ====================

    def _ensure_position_mode(self) -> None:
        """Ensure Binance account position mode matches config.

        Hedge mode (dualSidePosition=true) is REQUIRED for VolStraddle
        S_d8 straddle to open simultaneous LONG+SHORT on the same symbol.
        Without this, all positionSide=LONG/SHORT orders fail with -4059.

        One-Way mode is required for RubberBand/Elliott (they use reduceOnly).
        """
        if self.paper_mode:
            return

        want_hedge = self.strategy_name == "vol_straddle" and self.vs_config.hedge_mode

        try:
            resp = self.client.futures_get_position_mode()
            current_hedge = resp.get("dualSidePosition", False)
        except Exception as e:
            logger.error(f"Position mode check failed: {e}")
            return

        if current_hedge == want_hedge:
            mode_label = "Hedge" if current_hedge else "One-Way"
            logger.info(f"Position mode: {mode_label} (OK)")
            return

        try:
            self.client.futures_change_position_mode(
                dualSidePosition=want_hedge,
            )
            mode_label = "Hedge" if want_hedge else "One-Way"
            logger.info(f"Position mode switched to {mode_label}")
        except BinanceAPIException as e:
            if "-4059" in str(e):
                # Already in desired mode (race condition)
                pass
            elif "-4068" in str(e):
                # Cannot change while positions are open
                logger.error(
                    "Cannot switch position mode — open positions exist. Close all positions first, then restart."
                )
                self.discord.send(
                    content=(
                        "\U0001f6a8 **포지션 모드 전환 실패** — "
                        "열린 포지션이 있어 전환 불가. 모든 포지션 청산 후 재시작 필요."
                    )
                )
            else:
                logger.error(f"Position mode change failed: {e}")

    def run_once(self) -> None:
        """Run a single trading cycle."""
        self._ensure_position_mode()
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
        self._ensure_position_mode()
        balance = self.get_balance()
        self._initial_balance = balance
        self._high_water_mark = balance
        self._daily_start_balance = balance
        self._load_state()

        # Refresh balance after state load (paper_balance may have been
        # restored from saved state, overriding the settings default).
        balance = self.get_balance()

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
        commit_str = f"`{GIT_COMMIT['short']}`" if GIT_COMMIT["short"] else "unknown"
        n_recovered = self.strategy.n_active

        if self.strategy_name == "vol_straddle":
            vc = self.vs_config
            self.discord.send(
                embed={
                    "title": "🚀 QuantBox VolStraddle 시작",
                    "color": 0x2979FF,
                    "description": ("VolStraddle D_mom8 (A1-D50) 초기화 완료.\n`!help`로 명령어 확인."),
                    "fields": [
                        {"name": "모드", "value": f"**{mode_str}**", "inline": True},
                        {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                        {"name": "슬롯", "value": f"{n_recovered}/{vc.max_concurrent}", "inline": True},
                        {
                            "name": "── 전략 설정 ──",
                            "value": (
                                f"VR({vc.vr_short}/{vc.vr_long}) < {vc.d_vr_threshold:.2f}\n"
                                f"TP={vc.d_tp * 100:.0f}% | SL={vc.d_sl * 100:.0f}% | "
                                f"Mom={vc.momentum_period}bar"
                            ),
                            "inline": False,
                        },
                        {
                            "name": "TF",
                            "value": (
                                f"{'15m ' if vc.enable_15m else ''}"
                                f"{'1h ' if vc.enable_1h else ''}"
                                f"{'4h' if vc.enable_4h else ''}"
                            ).strip(),
                            "inline": True,
                        },
                        {"name": "레버리지", "value": f"{vc.base_leverage}x", "inline": True},
                        {
                            "name": "CW-Tier",
                            "value": (f"{'ON' if vc.cw_enabled else 'OFF'} (min={vc.min_conviction_score})"),
                            "inline": True,
                        },
                        {"name": "유니버스", "value": f"{n_symbols}종목", "inline": True},
                        {"name": "커밋", "value": commit_str, "inline": True},
                    ],
                    "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                }
            )
        else:
            eff_lev = self.config.base_leverage * self.config.mtf_leverage_mult
            _mtf_str = f"ON (SMA {self.config.mtf_sma_period})" if self.config.mtf_confirmation_enabled else "OFF"
            _adaptive_str = (
                f"적응형 {self.config.atr_multiplier}x "
                f"(floor {self.config.threshold_floor * 100:.1f}%, cap {self.config.threshold_cap * 100:.1f}%)"
                if self.config.atr_period > 0
                else f"고정 {self.config.threshold * 100:.1f}%"
            )
            self.discord.send(
                embed={
                    "title": "🚀 QuantBox 스캘퍼 시작",
                    "color": 0x2979FF,
                    "description": ("러버밴드 평균회귀 스캘퍼 초기화 완료.\n`!help`로 명령어 확인."),
                    "fields": [
                        {"name": "모드", "value": f"**{mode_str}**", "inline": True},
                        {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                        {"name": "슬롯", "value": f"{n_recovered}/{self.config.max_concurrent}", "inline": True},
                        {
                            "name": "── 전략 설정 ──",
                            "value": (
                                f"진입: {_adaptive_str}\n"
                                f"TP={self.config.target_retrace} | "
                                f"SL={'ATR×' + str(self.config.atr_stop_mult) + '[' + str(round(self.config.atr_stop_floor * 100, 1)) + '%-' + str(round(self.config.atr_stop_cap * 100)) + '%]' if self.config.atr_stop_enabled else str(self.config.stop_mult) + '×move'} | "
                                f"MaxHold={self.config.max_hold}바"
                            ),
                            "inline": False,
                        },
                        {"name": "MTF", "value": _mtf_str, "inline": True},
                        {
                            "name": "레버리지",
                            "value": (
                                f"Base {self.config.base_leverage}x × "
                                f"MTF {self.config.mtf_leverage_mult}x = "
                                f"{eff_lev:.1f}x"
                            ),
                            "inline": True,
                        },
                        {"name": "유니버스", "value": f"{n_symbols}종목", "inline": True},
                        {"name": "커밋", "value": commit_str, "inline": True},
                    ],
                    "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                }
            )

        # Pre-fetch exchange info
        self.get_symbol_info("BTCUSDT")

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
                self.discord.send(
                    embed={
                        "title": "🚨 사이클 오류",
                        "color": 0xFF1744,
                        "description": f"포지션 {self.strategy.n_active}개 활성 중\n```\n{str(e)[:400]}\n```",
                        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                    }
                )
                time.sleep(60)

        # Clean shutdown
        logger.info("Shutting down...")
        self._save_state()

        # Stop Discord bot
        if self._bot_runner:
            self._bot_runner.stop()

        total_pnl = sum(t["pnl_usd"] for t in self._trade_history)
        n_trades = len(self._trade_history)
        wins = sum(1 for t in self._trade_history if t["pnl_usd"] > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        balance_now = self.get_balance()
        session_return = (
            (balance_now - self._initial_balance) / self._initial_balance * 100 if self._initial_balance > 0 else 0
        )

        best = max(self._trade_history, key=lambda t: t["pnl_usd"]) if self._trade_history else None
        worst = min(self._trade_history, key=lambda t: t["pnl_usd"]) if self._trade_history else None

        # Terminal summary
        print(
            format_session_summary(
                cycles=self._cycle_count,
                n_trades=n_trades,
                win_rate=wr,
                total_pnl=total_pnl,
                balance=balance_now,
                session_roi=session_return,
                best_trade=best,
                worst_trade=worst,
            )
        )

        fields = [
            {"name": "사이클", "value": str(self._cycle_count), "inline": True},
            {"name": "거래", "value": f"{n_trades}건 (승률 {wr:.0f}%)", "inline": True},
            {"name": "총 수익", "value": f"${total_pnl:+,.2f}", "inline": True},
            {"name": "잔고", "value": f"${balance_now:,.2f}", "inline": True},
            {"name": "세션 ROI", "value": f"{session_return:+.2f}%", "inline": True},
        ]
        if best:
            fields.append({"name": "최고 수익", "value": f"{best['symbol']} ${best['pnl_usd']:+,.2f}", "inline": True})
        if worst:
            fields.append(
                {"name": "최대 손실", "value": f"{worst['symbol']} ${worst['pnl_usd']:+,.2f}", "inline": True}
            )

        self.discord.send(
            embed={
                "title": "🛑 QuantBox 스켈퍼 종료",
                "color": 0xFF1744 if total_pnl < 0 else 0x00C853,
                "fields": fields,
                "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            }
        )


# ──────────────────────── CLI ──────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="QuantBox Scalper — Rubber Band / Elliott Wave / VolStraddle")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode (no real orders)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--lookback", type=int, default=None, help="Signal lookback bars (default: from .env/config)")
    parser.add_argument("--threshold", type=float, default=None, help="Min move to trigger (default: from .env/config)")
    parser.add_argument(
        "--max-slots", type=int, default=None, help="Max concurrent positions (default: from .env/config)"
    )
    parser.add_argument(
        "--strategy",
        choices=["rubberband", "elliott", "vol_straddle"],
        default="vol_straddle",
        help="Trading strategy: vol_straddle (default), rubberband, or elliott",
    )
    args = parser.parse_args()

    setup_logging(strategy_name=args.strategy, paper_mode=args.paper, git_commit=GIT_COMMIT)

    # Build config from RubberBandConfig defaults (config.py is SSoT) + CLI overrides
    config = RubberBandConfig()
    if args.lookback is not None:
        config.lookback = args.lookback
    if args.threshold is not None:
        config.threshold = args.threshold
    if args.max_slots is not None:
        config.max_concurrent = args.max_slots

    scalper = RubberBandScalper(
        paper_mode=args.paper,
        config=config,
        strategy_name=args.strategy,
        vs_config=VolStraddleConfig(**({"max_concurrent": args.max_slots} if args.max_slots is not None else {}))
        if args.strategy == "vol_straddle"
        else None,
    )

    if args.once:
        scalper.run_once()
    else:
        scalper.run_continuous()


if __name__ == "__main__":
    main()
