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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantbox.core.types import ApiSymbol, Side, PositionSide
    from quantbox.strategy.protocol import TradingStrategy

import argparse
import logging
import logging.handlers
import math
import time
import uuid
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
    format_position_line,
    format_scan_summary,
    format_section_header,
    format_session_summary,
)
from quantbox.notifications.bot import (
    DiscordBotRunner,
)
from quantbox.notifications.discord import DiscordNotifier
from quantbox.strategy.vol_straddle import VolStraddleConfig, VolStraddleStrategy
from quantbox.core.bot_utils import GIT_COMMIT, setup_logging
from quantbox.core.strategy_names import normalize_strategy_type
from quantbox.core.commands import CommandsMixin
from quantbox.execution.algo_api import AlgoApiClient
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
from quantbox.execution.preflight import OrderPreflight
from quantbox.execution.risk_guard import RiskGuard
from quantbox.execution.reconciliation import ReconciliationMixin
from quantbox.core.trade_journal import TradeJournal  # noqa: F401 (patched by tests)
from quantbox.core.divergence_monitor import DivergenceMonitor  # noqa: F401 (patched by tests)
from quantbox.core.lifecycle import (
    create_divergence_monitor,
    create_reliability_modules,
    create_trade_journal,
)

# ── Reliability modules (Phase 3 integration) ──
from quantbox.core.heartbeat import Heartbeat  # noqa: F401 (patched by tests)
from quantbox.core.rate_limiter import RateLimiter
from quantbox.core.sd_notify import sd_notify
from quantbox.execution.exposure import ExposureGuard  # noqa: F401 (patched by tests)
from quantbox.execution.inflight import InFlightTracker, OrderIntent, PendingOrder  # noqa: F401 (InFlightTracker patched by tests)
from quantbox.execution.recon_journal import ReconJournal  # noqa: F401 (patched by tests)

load_dotenv()

# ──────────────────────── Logging ──────────────────────────────────────

logger = logging.getLogger("Scalper")


# ──────────────── Diagnostic Logger ─────────────────────
from quantbox.utils.diagnostics import _diag_logger  # noqa: E402

# ──────────────────────── Helpers ──────────────────────────────────────────


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
        config: Strategy config (VolStraddleConfig for production).
        strategy: Stateful strategy instance implementing TradingStrategy.
    """

    def __init__(
        self,
        paper_mode: bool = False,
        config=None,
        strategy_name: str = "rubberband",
        vs_config: VolStraddleConfig | None = None,
        journal_db_path: str | None = None,
    ):
        self.paper_mode = paper_mode
        self.strategy_name = strategy_name

        _valid_strategies = {"rubberband", "vol_straddle"}
        if strategy_name not in _valid_strategies:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Valid: {sorted(_valid_strategies)}")

        self.strategy: TradingStrategy
        if strategy_name == "vol_straddle":
            self.vs_config = vs_config or VolStraddleConfig()
            self.config = self.vs_config
            self.strategy = VolStraddleStrategy(self.vs_config)
        else:
            # DEPRECATED: RubberBand strategy — kept for test backward compat
            from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy

            self.config = config if config is not None else RubberBandConfig()
            # trail_server_native=False: rubberband doesn't use server-side trailing stops
            self.vs_config = vs_config or VolStraddleConfig(trail_server_native=False)
            self.strategy = RubberBandStrategy(self.config)

        # Binance client
        settings = get_settings()
        self.client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=settings.binance_testnet,
        )
        self._algo_api = AlgoApiClient(self.client)

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

        # Process stability: consecutive error tracking and safe mode
        self._consecutive_errors: int = 0
        self._safe_mode: bool = False  # True = skip new entries, only manage existing

        # Symbol info cache (avoid repeated exchange_info calls)
        self._symbol_info_cache: dict[str, dict] = {}
        self._exchange_info_fetched: bool = False

        # Trade history
        self._trade_history: list[dict] = []
        self._cycle_count: int = 0

        # Trade journal (persistent SQLite)
        self._journal = create_trade_journal(paper_mode, journal_db_path)

        # Live-Backtest divergence monitor
        self._divergence_monitor = create_divergence_monitor(self.vs_config)

        # ── Reliability modules (Phase 3) ──
        _reliability = create_reliability_modules(self.vs_config)
        self._rate_limiter = _reliability["rate_limiter"]
        self._heartbeat = _reliability["heartbeat"]
        self._exposure_guard = _reliability["exposure_guard"]
        self._inflight = _reliability["inflight"]
        self._recon_journal = _reliability["recon_journal"]

        # Risk gate helper (stateless, injected)
        self._risk_guard = RiskGuard(self.vs_config)

        # Restart tracking & post-restart reconciliation
        self._restart_count: int = 0
        self._last_restart_at: str | None = None
        self._last_full_reconcile_at: str | None = None
        self._startup_cycles_remaining: int = 0
        self._clean_cycles_count: int = 0  # consecutive clean cycles since last error

        # Per-cycle balance cache — reset at start of each reconciliation cycle
        self._cycle_balance: float | None = None

        # Control flags
        self._stop_requested: bool = False
        self._paused: bool = False

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"RubberBandScalper initialized ({mode_str}) strategy={strategy_name}")
        vs = self.vs_config
        logger.info(
            f"Config: VR({vs.vr_short}/{vs.vr_long}) "
            f"D_threshold={vs.d_vr_threshold}, "
            f"SL={vs.d_sl * 100:.1f}%, TP={vs.d_tp * 100:.1f}%, "
            f"TF={'15m' if vs.enable_15m else ''}"
            f"{'|1h' if vs.enable_1h else ''}"
            f"{'|4h' if vs.enable_4h else ''}"
        )

    # ==================== Balance & Risk ====================

    @property
    def _cfg(self):
        """Active strategy config."""
        return self.vs_config

    def get_balance(self) -> float:
        """Get current USDT balance.

        Sets self._balance_fetch_ok = True on success, False on API failure.
        Callers must check this flag before opening new positions (T1-4).
        """
        if self.paper_mode:
            self._balance_fetch_ok = True
            return self._paper_balance
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    val = float(b["balance"])
                    self._balance_fetch_ok = True
                    return val
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
        # T1-4: Mark stale — callers will block new entries on this flag
        self._balance_fetch_ok = False
        # v12.5.1: Return last known balance instead of 0.0 to prevent
        # false drawdown halt on transient API failures.
        if hasattr(self, "_last_known_balance") and self._last_known_balance > 0:
            logger.warning(f"Using last known balance ${self._last_known_balance:.2f} due to API failure")
            return self._last_known_balance
        logger.critical("Balance fetch failed and no fallback available — new entries will be blocked")
        return 0.0

    def _get_cycle_balance(self) -> float:
        """사이클당 1회 balance API 호출 캐시 — 중복 API 호출 방지."""
        if self._cycle_balance is None:
            self._cycle_balance = self.get_balance()
        return self._cycle_balance

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

    def _verify_invariants(self, balance: float) -> list[str]:
        """T3-5: Check position invariants every cycle.

        Returns list of violation descriptions. Empty = all OK.
        Logs violations and sends a single Discord alert if any found.
        Does NOT close positions automatically — alert and log only.
        """
        import re

        violations: list[str] = []

        # 1. Position count
        n_active = self.strategy.n_active
        max_c = self._cfg.max_concurrent
        if n_active > max_c:
            violations.append(f"position_count={n_active} > max_concurrent={max_c}")

        # 2-5. Per-position checks
        safe_balance = max(balance, 1.0)
        for pos_key, pos in self.strategy._positions.items():
            entry_price = pos.get("entry_price", 0)
            notional = pos.get("notional", 0)
            signal = pos.get("signal")
            api_sym = pos.get("symbol", pos_key.split(":")[0] if ":" in pos_key else pos_key)

            if not (0 < entry_price < 1e7):
                violations.append(f"[{pos_key}] entry_price={entry_price!r} out of range")
            if not (0 < notional < safe_balance * max_c * 125 + 1):
                violations.append(f"[{pos_key}] notional={notional!r} out of range")
            if signal not in (-1, 0, 1, 2):
                violations.append(f"[{pos_key}] invalid signal={signal!r}")
            if not re.match(r"^[A-Z]{2,12}USDT$", api_sym):
                violations.append(f"[{pos_key}] malformed symbol={api_sym!r}")

        if violations:
            self.discord.send(
                content=(
                    f"\u26a0\ufe0f **불변량 위반** ({len(violations)}개) — "
                    + "; ".join(violations[:3])
                    + ("..." if len(violations) > 3 else "")
                )
            )

        return violations

    # ==================== Order Execution ====================

    def _place_order(
        self,
        symbol: "ApiSymbol",
        side: "Side",
        quantity: float,
        leverage: int = 1,
        reduce_only: bool = False,
        position_side: "PositionSide | None" = None,
        paper_fill_price: float | None = None,
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
            paper_fill_price: Override fill price in paper mode (e.g. SL/TP
                trigger price). When None, paper mode fetches live orderbook.

        Returns dict with avgPrice, executedQty on success, None on failure.
        """
        # ── Pre-flight validation (paper AND live) ──────────────────────────
        # Run before paper-mode early return so that bugs like compound keys
        # (':' in symbol → -1022) are caught during paper testing, not only
        # when the same code path hits the live exchange for the first time.
        info = self.get_symbol_info(symbol)
        preflight = OrderPreflight.validate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=leverage,
            reduce_only=reduce_only,
            sym_info=info,
        )
        if not preflight.ok:
            logger.error(f"[{symbol}] Order preflight FAILED: {preflight.reason}")
            return None

        if self.paper_mode:
            if paper_fill_price is not None:
                price = paper_fill_price
            else:
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
            tick_size = info.get("tick_size", 0.01)
            price_precision = info.get("price_precision", 2)
            step_size = info.get("step_size", 0.001)
            qty_precision = info.get("quantity_precision", 3)

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

            # ── InFlight: record pending BEFORE API call ──
            _inflight_key: str | None = None
            try:
                if self._inflight is not None:
                    _intent = (
                        OrderIntent.EXIT if reduce_only or (leverage <= 1 and position_side) else OrderIntent.ENTRY
                    )
                    _pending = PendingOrder(
                        intent=_intent,
                        symbol=symbol,
                        side=side,
                        position_side=position_side,
                        quantity=quantity,
                        order_type="LIMIT",
                    )
                    _inflight_key = self._inflight.record_pending(_pending)
            except Exception:
                pass

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
                # T1-1: Generate stable clientOrderId for audit trail.
                # Format: QB_{symbol}_{side}_{unix_ms} — unique per order attempt.
                # Entry orders pass leverage>1, close orders default to leverage=1.
                client_order_id_prefix=(
                    f"QB_{symbol}_{side}_{int(time.time() * 1000)}" if not self.paper_mode else None
                ),
            )

            # ── InFlight: confirm after API call ──
            try:
                if self._inflight is not None and _inflight_key:
                    self._inflight.confirm(_inflight_key)
            except Exception:
                pass

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
                # ── Rate limiter: record successful API call ──
                try:
                    if self._rate_limiter is not None:
                        self._rate_limiter.record_request(weight=1)
                        self._rate_limiter.record_success()
                except Exception:
                    pass
                return {
                    "avgPrice": str(result.avg_price),
                    "executedQty": str(result.filled_quantity),
                    "status": "FILLED",
                    "makerFill": result.maker_fill,
                }
            else:
                logger.error(f"[{symbol}] Order FAILED: {side} {quantity:.6f} (status={result.status.value})")
                return None

        except BinanceAPIException as e:
            # ── Rate limiter: detect 429 / rate limit errors ──
            try:
                if self._rate_limiter is not None:
                    is_rate, retry = RateLimiter.parse_rate_limit_error(str(e))
                    if is_rate:
                        backoff = self._rate_limiter.record_failure(retry)
                        logger.warning(f"[{symbol}] Rate limit hit — backing off {backoff:.1f}s")
                        time.sleep(backoff)
            except Exception:
                pass
            logger.error(f"[{symbol}] Order error: {e}")
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

    def _preflight(self) -> tuple[float, bool]:
        """Rate limit, balance fetch, bot commands. Returns (balance, stop_requested)."""
        self._cycle_count += 1

        try:
            if self._rate_limiter is not None:
                self._rate_limiter.wait_if_needed(weight=5)
        except Exception as e:
            logger.error(f"RateLimiter.wait_if_needed error: {e}")

        balance = self.get_balance()

        self._process_bot_commands()
        return balance, self._stop_requested

    def _log_cycle_header(self, balance: float) -> None:
        """Print cycle header and log HWM / DD / cumPnL stats."""
        header = format_cycle_header(
            cycle=self._cycle_count,
            balance=balance,
            n_active=self.strategy.n_active,
            max_slots=self._cfg.max_concurrent,
        )
        print(header)

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

    def _check_risk_and_halt(self, balance: float) -> tuple[bool, bool]:
        """Risk limits + invariants + drawdown halt management.

        Returns:
            (halt_stop, risk_blocked) — if halt_stop is True, caller should
            return immediately. risk_blocked is forwarded to entry scanning.
        """
        risk_blocked = self._check_risk_limits(balance)

        # T3-5: Invariant check every cycle
        violations = self._verify_invariants(balance)
        if violations:
            logger.error(f"INVARIANT VIOLATIONS ({len(violations)}): " + "; ".join(violations))

        if self._drawdown_halt:
            # Close any remaining positions
            for pos_key in list(self.strategy.active_positions):
                api_sym = pos_key.split(":")[0] if ":" in pos_key else pos_key
                close = self.download_klines(api_sym, 2)
                price = float(close[-1]) if close is not None else 0.0
                self._close_position(pos_key, "drawdown_halt", price)

            # v14.0.1: Auto-recover when DD drops below threshold
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
                    # Fall through — cycle continues normally after recovery
                else:
                    logger.info(
                        f"DD halt active — {dd * 100:.1f}% DD "
                        f"(need <{self._cfg.max_drawdown_pct * 100:.0f}% to resume). "
                        f"All positions closed, waiting for HWM reset on restart."
                    )
                    self._save_state()
                    return True, True
            else:
                self._save_state()
                return True, True

        return False, risk_blocked

    def _sync_exchange(self) -> tuple[object, object]:
        """Fetch BTC klines, sync server orders, reconcile, log position status.

        Returns (btc_close, btc_close_4h) arrays.
        """
        # Fetch BTC klines for regime filter
        btc_close = self.download_klines("BTCUSDT", 200)

        btc_close_4h = None
        # r92: Download BTC 4h data for MA cluster 4h BTC gate.
        # Gate prevents MA cluster LONGs during multi-day BTC downtrends.
        if self.vs_config and self.vs_config.ma_cluster_btc_gate_enabled:
            btc_close_4h = self.download_klines("BTCUSDT", 100, interval="4h")

        # v11.2: Sync server-side TP/SL orders (check if filled intra-bar)
        self._sync_server_orders()

        # v12.5.1: Reconcile positions with exchange AFTER sync
        if not self.paper_mode:
            self._reconcile_positions()

        # v12.5.1: Log position status AFTER sync (SL/TP verified)
        if self.strategy.active_positions:
            logger.info(format_section_header("POSITIONS"))
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
            # Use bars_held+1 to match the value AFTER this cycle's update,
            # consistent with the Summary section printed later.
            bars = pos.get("bars_held", 0) + 1
            direction = "L" if pos.get("signal") == 1 else "S"
            logger.info(
                format_position_line(
                    pos_key=pos_key,
                    strategy_type=pos.get("strategy_type", "vr_directional"),
                    direction=direction,
                    entry_price=entry_px,
                    current_price=cur_px,
                    upnl_pct=upnl_pct,
                    bars=bars,
                    max_hold=pos.get("max_hold", 192),
                    sl_pct=pos.get("sl", 0),
                    sl_active=bool(pos.get("server_sl_order_id")),
                    tp_pct=pos.get("tp", 0),
                    tp_active=bool(pos.get("server_tp_order_id")),
                    trail_active=bool(pos.get("trail_active")),
                    peak_gain=pos.get("peak_gain", 0) if pos.get("trail_active") else None,
                )
            )

        return btc_close, btc_close_4h

    def _manage_active_positions(self) -> None:
        """Process existing positions: update, exit, ratchet SL."""
        for pos_key in list(self.strategy.active_positions):
            # Extract Binance symbol from compound key (e.g., BTCUSDT:LONG → BTCUSDT)
            api_sym = pos_key.split(":")[0] if ":" in pos_key else pos_key

            # v12.3.2: Download full OHLC for active positions to compute fresh ATR.
            result_full = self.download_klines(api_sym, 200, full=True)
            if result_full is None:
                continue
            close, high, low, _vol, _tb = result_full

            current_price = close[-1]

            # v12.3.2: Compute FRESH ATR for PBT runner trailing
            # VolStraddle uses R43 trailing stop (client-side, no ATR needed).
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
                if ":" in pos_key:
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
                elif exit_reason in ("confirm_entry", "pyramid", "cancel_exploration"):
                    self._close_position(pos_key, exit_reason, current_price)
                else:
                    # ── Step 2.3: Capture counterpart before close (straddle pair context) ──
                    counterpart_key_pre_close = None
                    if ":" in pos_key:
                        pos_pre = self.strategy._positions.get(pos_key)
                        if pos_pre and pos_pre.get("is_straddle"):
                            counterpart_key_pre_close = self._get_pair_counterpart(pos_key)

                    # Compute SL/TP trigger fill price for all modes.
                    # Live: _place_order ignores this (exchange fills at real price).
                    # Paper: _place_order uses this as the simulated fill price.
                    close_price = current_price
                    if exit_reason in ("stop_loss", "target"):
                        pos_data = self.strategy._positions.get(pos_key)
                        if pos_data:
                            entry = pos_data["entry_price"]
                            sig = pos_data["signal"]
                            sl_pct = pos_data.get("sl", self.vs_config.d_sl)
                            tp_pct = pos_data.get("tp", self.vs_config.d_tp)
                            if exit_reason == "stop_loss":
                                close_price = entry * (1 - sl_pct) if sig == 1 else entry * (1 + sl_pct)
                            elif exit_reason == "target":
                                close_price = entry * (1 + tp_pct) if sig == 1 else entry * (1 - tp_pct)

                    self._close_position(pos_key, exit_reason, close_price)

                    # ── Step 2.3: Verify counterpart brackets after straddle leg close ──
                    if counterpart_key_pre_close:
                        cpart = self.strategy._positions.get(counterpart_key_pre_close)
                        if cpart:
                            if not cpart.get("server_sl_order_id"):
                                logger.warning(
                                    f"[{counterpart_key_pre_close}] 스트래들 다리 청산 후 SL 없음 — bracket repair 필요"
                                )
                                # Trigger next-cycle bracket repair by ensuring the flag is clear
                                cpart["_repair_fail_count"] = 0
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
                    # r90: Skip ratchet when Binance handles trail natively
                    if (
                        pos.get("trail_active")
                        and not self.paper_mode
                        and getattr(self.vs_config, "trail_server_ratchet_enabled", True)
                        and not getattr(self.vs_config, "trail_server_native", False)
                    ):
                        self._ratchet_trail_sl(pos_key, pos)
                    # ── Fallback: deferred server trail ──
                    # If trail wasn't placed at entry (API failure), place it now
                    # when trail_activate% is reached. Uses current price since
                    # activation threshold is already met.
                    if (
                        pos.get("trail_active")
                        and not pos.get("server_trail_order_id")
                        and not self.paper_mode
                        and getattr(self.vs_config, "trail_server_native", False)
                    ):
                        api_sym = pos_key.split(":")[0]
                        strategy_type = pos.get("strategy_type", "vr_directional")
                        signal = pos.get("signal", 1)
                        entry_price = pos.get("entry_price", current_price)
                        pos_side = pos.get("position_side")
                        notional = pos.get("notional", 0.0)
                        qty = notional / entry_price if entry_price > 0 else 0.0
                        if qty > 0:
                            cr, _, cs = self._get_trail_params(strategy_type, signal, entry_price)
                            trail_id = self._place_trail_regular(
                                symbol=api_sym,
                                side=cs,
                                quantity=qty,
                                callback_rate=cr,
                                activation_price=current_price,
                                position_side=pos_side,
                            )
                            if trail_id:
                                pos["server_trail_order_id"] = trail_id
                                logger.info(
                                    f"[{pos_key}] Fallback server trail PLACED: "
                                    f"orderId={trail_id} callbackRate={cr}% "
                                    f"activationPrice=${current_price:.4f} "
                                    f"(peak={pos.get('peak_gain', 0) * 100:+.2f}%)"
                                )
                            else:
                                logger.warning(f"[{pos_key}] Fallback server trail FAILED to place")

    def _scan_entry_opportunities(
        self,
        symbols: list[str],
        btc_close: object,
        balance: float,
        btc_close_4h: object,
        risk_blocked: bool = False,
    ) -> None:
        """Gate and run entry scanning: paused, risk, daily breaker, balance stale."""
        if self._paused:
            logger.info("⏸️ Paused — skipping new entries")
            return

        if risk_blocked:
            return

        # ── G7: Daily Loss Circuit Breaker ──
        if self.vs_config.daily_loss_limit_pct > 0:
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
            if not self._daily_loss_blocked and self._risk_guard.check_daily_loss(self._daily_realized_pnl, balance):
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
                logger.info(f"G7 DailyBreaker active: today PnL=${self._daily_realized_pnl:.2f}. Skipping new entries.")
                return

        # T1-4: Block new entries when balance is stale (API failure).
        if not getattr(self, "_balance_fetch_ok", True):
            logger.warning(
                f"T1-4 Balance fetch FAILED — skipping new entries this cycle "
                f"(using {'stale $' + str(round(balance, 2)) if balance > 0 else 'NO DATA'})"
            )
            if not getattr(self, "_balance_alert_sent", False):
                self.discord.send(
                    content=(
                        f"⚠️ **잔고 조회 실패** — 이번 사이클 신규 진입 차단. "
                        f"{'최종 잔고 $' + str(round(balance, 2)) + ' 사용 중' if balance > 0 else '잔고 불명 (0)'}. "
                        f"거래소 API 확인 요망."
                    )
                )
                self._balance_alert_sent = True
            return

        self._balance_alert_sent = False  # Reset alert flag on recovery
        available_slots = self._cfg.max_concurrent - self.strategy.n_active
        if available_slots > 0:
            self._scan_entries(symbols, btc_close, balance, available_slots, btc_close_4h=btc_close_4h)

    def _cycle_housekeeping(self, balance: float) -> None:
        """Save state, divergence check, summary log, heartbeat."""
        self._save_state()
        self._check_divergence()
        if self.strategy.n_active > 0 or self._cycle_count % 4 == 0:
            self._log_summary(balance)
        try:
            if self._heartbeat is not None:
                self._heartbeat.beat()
        except Exception as e:
            logger.error(f"Heartbeat.beat error: {e}")

    def _cycle(self, symbols: list[str]) -> None:
        """Process one 15m bar across all symbols.

        Orchestrates seven private methods — each testable in isolation:
          1. _preflight()                 rate limit, balance, bot commands
          2. _log_cycle_header()          logging only
          3. _check_risk_and_halt()       risk limits, DD halt (may return early)
          4. _sync_exchange()             BTC klines, server sync, reconcile
          5. _manage_active_positions()   per-position update / exit / ratchet
          6. _scan_entry_opportunities()  paused / daily-breaker / scan gates
          7. _cycle_housekeeping()        save, divergence, summary, heartbeat
        """
        if self._startup_cycles_remaining > 0:
            self._startup_cycles_remaining -= 1

        balance, stop = self._preflight()
        if stop:
            return

        self._log_cycle_header(balance)

        halt_stop, risk_blocked = self._check_risk_and_halt(balance)
        if halt_stop:
            return

        btc_close, btc_close_4h = self._sync_exchange()
        self._manage_active_positions()
        self._scan_entry_opportunities(symbols, btc_close, balance, btc_close_4h, risk_blocked=risk_blocked)
        self._cycle_housekeeping(balance)

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
        candidates: list[tuple[str, int, float, np.ndarray, bool]] = []
        moves_seen: list[tuple[str, float]] = []  # diagnostic: (symbol, abs_move)
        scan_diags: list[dict] = []  # v12.0: per-symbol diagnostics

        for sym in symbols:
            # Skip symbols we already have positions in
            if self.strategy.has_position(sym):
                continue

            # v9.0: Download full klines (close + high + low + volume + taker)
            # VolStraddle needs 500 bars: 15m warmup=200 needs ≥201 bars,
            # 1h resampling needs ≥4×(warmup_1h+1)=204 bars. 500 covers both.
            n_bars = 500
            result = self.download_klines(sym, n_bars, full=True)
            if result is None:
                continue
            close, high, low, volume, taker_buy = result
            if len(close) < self._cfg.lookback + 1:
                continue

            # Wire up volume/taker/high/low data for v9 confluence scoring
            self.strategy.update_market_data(sym, volume, taker_buy, high, low)

            # v7.0: Download 4h data for MTF confirmation
            # VolStraddle needs full 4h OHLCV for VR + MA Cluster.
            # MA Cluster uses SMA(200) → need ≥201 bars; 300 for safe warmup.
            n_bars_4h = 300 if (self.vs_config and self.vs_config.ma_cluster_enabled) else 100
            result_4h = self.download_klines(sym, n_bars_4h, interval="4h", full=True)
            if result_4h is not None:
                c4h, h4h, l4h, _, _ = result_4h
                self.strategy.update_htf_data(sym, c4h, h4h, l4h)

            signal = self.strategy.on_bar(sym, close, btc_close)

            # r92: MA cluster 4h BTC gate — block MA cluster LONGs during BTC downtrends.
            # Applied immediately after on_bar so blocked signals never reach candidates.
            if (
                signal == 1
                and self.vs_config
                and self.vs_config.ma_cluster_btc_gate_enabled
                and getattr(self.strategy, "_last_signal_source", {}).get(sym) == "ma_cluster"
                and btc_close_4h is not None
                and len(btc_close_4h) >= self.vs_config.ma_cluster_btc_ma_period_4h
            ):
                period = self.vs_config.ma_cluster_btc_ma_period_4h
                btc_4h_ma = float(np.mean(btc_close_4h[-period:]))
                btc_4h_cur = float(btc_close_4h[-1])
                if btc_4h_cur < btc_4h_ma:
                    logger.info(
                        f"[{sym}] MA 4h gate LONG suppressed: BTC 4h ${btc_4h_cur:.1f} < MA{period} ${btc_4h_ma:.1f}"
                    )
                    signal = 0

            # v12.0: Collect full diagnostic context (read-only, after on_bar)
            if hasattr(self.strategy, "diagnose_bar"):
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
                    # Straddle: expand into LONG + SHORT candidates.
                    # is_straddle=True so G6 and DirDiversity filters are skipped
                    # (straddle is direction-neutral — BTC trend gate does not apply).
                    # Generate pair ID now so both legs share it at entry time.
                    pair_id = str(uuid.uuid4())[:8]
                    candidates.append((sym, 1, move, close, True, pair_id))
                    candidates.append((sym, -1, move, close, True, pair_id))
                else:
                    candidates.append((sym, signal, move, close, False, None))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

        # ── Diagnostic: log scan summary every cycle ──
        top_movers = sorted(moves_seen, key=lambda x: x[1], reverse=True)[:3]
        # VR strategy: show VR threshold + actual VR values
        print(format_scan_summary(len(candidates), top_movers, 0.0))
        logger.info(format_section_header("SIGNALS"))
        if candidates:
            cand_strs = [f"{s}({m * 100:.1f}%)" for s, _, m, _, _, _ in candidates[:3]]
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

        # v12.0: Write structured diagnostics to JSONL
        if scan_diags:
            ts = datetime.now(timezone.utc).isoformat()
            _diag_logger.log_scan(self._cycle_count, ts, scan_diags)

        # ── G6: BTC Trend Gate — compute once, apply to all LONG candidates ──
        btc_ma = self._risk_guard.compute_btc_gate(btc_close)

        entries_made = 0
        straddle_opened: dict[str, set[int]] = {}  # {sym: {signals opened}} for rollback
        straddle_blocked: set[str] = set()  # symbols blocked by straddle slot cap
        for sym, signal, move, close, is_straddle, straddle_pair_id in candidates:
            if entries_made >= max_entries:
                break

            # ── Straddle slot cap — skip both legs if cap reached ──
            if is_straddle:
                if sym in straddle_blocked:
                    continue  # second leg of an already-blocked pair
                if sym not in straddle_opened:
                    # First leg: upfront check via RiskGuard
                    remaining_slots = max_entries - entries_made
                    active_straddle_pairs = sum(
                        1 for k, p in self.strategy._positions.items() if ":LONG" in k and p.get("is_straddle", False)
                    )
                    straddle_ok, straddle_reason = self._risk_guard.allow_straddle(
                        active_straddle_pairs, remaining_slots
                    )
                    if not straddle_ok:
                        straddle_blocked.add(sym)
                        logger.info(f"[{sym}] Straddle BLOCKED: {straddle_reason}")
                        continue

            # ── G6: BTC Trend Gate — block directional LONG when BTC below MA ──
            # Straddle is direction-neutral (LONG+SHORT) → G6 does not apply.
            if signal == 1 and not is_straddle and not self._risk_guard.allow_long(btc_close, btc_ma):
                # btc_close is not None here: allow_long returns False only when both args are set
                btc_cur_log = float(btc_close[-1]) if btc_close is not None else 0.0
                logger.info(
                    f"[{sym}] G6 LONG suppressed: BTC ${btc_cur_log:.1f} < "
                    f"MA{self.vs_config.btc_long_ma_period} ${btc_ma:.1f}"
                )
                continue

            # ═══ v12.3 PARITY: Direction Diversity Filter ═══
            # Straddle is direction-neutral → diversity filter does not apply.
            if not is_straddle:
                dir_ok, dir_reason = self._risk_guard.allow_direction(signal, self.strategy.active_positions)
                if not dir_ok:
                    logger.info(f"[{sym}] DirDiversity blocked: {dir_reason}")
                    continue

            # ── Straddle: pair-aware entry with retry for second leg ──
            if is_straddle and sym not in straddle_opened:
                # LONG leg is always first (signal=1 candidate appears before signal=-1)
                if self._open_position(sym, signal, close, balance, straddle_pair_id=straddle_pair_id):
                    entries_made += 1
                    straddle_opened.setdefault(sym, set()).add(signal)
                # LONG failed — second leg (SHORT) skipped when LONG not in straddle_opened
                continue

            if is_straddle and sym in straddle_opened and 1 in straddle_opened[sym]:
                # Second leg (SHORT): attempt with retry
                cfg_vs = self.vs_config
                short_opened = False
                for retry in range(cfg_vs.entry_retry_count + 1):
                    if self._open_position(sym, signal, close, balance, straddle_pair_id=straddle_pair_id):
                        entries_made += 1
                        straddle_opened[sym].add(signal)
                        short_opened = True
                        break
                    if retry < cfg_vs.entry_retry_count:
                        logger.warning(
                            f"[{sym}] SHORT 진입 실패, 재시도 {retry + 1}/{cfg_vs.entry_retry_count} "
                            f"({cfg_vs.entry_retry_delay_s}s 후)"
                        )
                        time.sleep(cfg_vs.entry_retry_delay_s)
                if not short_opened:
                    # All retries failed — SHORT never opened, straddle incomplete
                    # Mark LONG as needing rollback
                    straddle_opened[sym].discard(-1)  # ensure only LONG is in set
                continue

            # Directional entry (non-straddle)
            if self._open_position(sym, signal, close, balance):
                entries_made += 1

        # ── Straddle half-open rollback ──
        # If only LONG opened and SHORT never succeeded, rollback LONG immediately.
        for sym, signals_opened in straddle_opened.items():
            if len(signals_opened) == 1 and 1 in signals_opened:
                # Only LONG opened — SHORT failed all retries
                pos_key = f"{sym}:LONG"
                logger.error(f"[{sym}] Straddle incomplete: LONG 진입됐으나 SHORT 실패. LONG 롤백 — 단방향 노출 방지.")
                self.discord.send(
                    content=(
                        f"⚠️ **[{sym}] Straddle half-open** — LONG 진입됐으나 SHORT 실패. LONG 롤백 중 (재시도 소진)."
                    )
                )
                # Use last close price as rollback price
                close_arr = next((c for s, sig, _, c, is_s, _ in candidates if s == sym and is_s), None)
                rollback_price = float(close_arr[-1]) if close_arr is not None else 0.0
                cfg_vs = self.vs_config
                pos = self.strategy._positions.get(pos_key)

                # Rollback via _close_position with retry
                rollback_ok = False
                for attempt in range(cfg_vs.rollback_retry_count + 1):
                    if self._close_position(pos_key, "straddle_rollback", rollback_price):
                        rollback_ok = True
                        break
                    if attempt < cfg_vs.rollback_retry_count:
                        logger.warning(f"[{pos_key}] 롤백 실패, 재시도 {attempt + 1}/{cfg_vs.rollback_retry_count}")
                        time.sleep(cfg_vs.rollback_retry_delay_s)

                if not rollback_ok:
                    # All _close_position attempts failed — market order fallback
                    # CRITICAL (Rule #1): NO leverage parameter — avoids futures_change_leverage in close path
                    pos = self.strategy._positions.get(pos_key)
                    if pos is not None:
                        api_sym = pos_key.split(":")[0]
                        close_side = "SELL"  # LONG → close with SELL
                        qty = self._round_qty(pos["notional"] / pos["entry_price"], api_sym)
                        fallback_ok = False
                        if qty > 0 and not self.paper_mode:
                            try:
                                order_kwargs: dict = {
                                    "symbol": api_sym,
                                    "side": close_side,
                                    "type": "MARKET",
                                    "quantity": qty,
                                    "positionSide": "LONG",
                                }
                                self.client.futures_create_order(**order_kwargs)
                                fallback_ok = True
                                logger.info(f"[{pos_key}] 롤백 시장가 주문 성공 (fallback)")
                                # Clean up internal state
                                self.strategy.close_position(pos_key)
                                self._save_state()
                            except Exception as e:
                                logger.critical(f"[{pos_key}] 롤백 완전 실패: {e}")
                        elif self.paper_mode:
                            # Paper mode: just clean up state
                            self.strategy.close_position(pos_key)
                            self._save_state()
                            fallback_ok = True

                        if not fallback_ok:
                            logger.critical(f"[{pos_key}] 롤백 완전 실패 — 수동 청산 필요!")
                            pos["rollback_failed"] = True
                            self._save_state()
                            self.discord.send(
                                content=(f"🚨 **[{pos_key}] 롤백 완전 실패** — 시장가 주문도 실패. 수동 청산 필요!")
                            )
                    else:
                        logger.warning(f"[{pos_key}] 롤백 시도 중 포지션이 이미 없음 — 무시")

                entries_made -= 1

                # ── Set rollback cooldown to prevent re-entry loop ──
                cooldown_cycles = self.vs_config.straddle_rollback_cooldown_cycles
                self.strategy._straddle_rollback_cooldown[sym] = cooldown_cycles
                self._save_state()  # Persist immediately — crash before save restarts the loop
                logger.info(
                    f"[{sym}] Straddle rollback cooldown set: {cooldown_cycles} cycles (~{cooldown_cycles * 15}min)"
                )

                # ── Close orphaned counterpart on exchange (reverse ghost prevention) ──
                # The SHORT leg from a prior straddle may still be open on exchange
                # without internal state tracking it, causing perpetual rollback loops.
                if not self.paper_mode:
                    try:
                        ex_positions = self.client.futures_position_information(symbol=sym)
                        for ep in ex_positions:
                            if ep.get("positionSide") == "SHORT" and abs(float(ep.get("positionAmt", 0))) > 0:
                                orphan_qty = abs(float(ep["positionAmt"]))
                                logger.warning(
                                    f"[{sym}] Orphaned SHORT on exchange "
                                    f"(qty={orphan_qty}) — closing to prevent rollback loop"
                                )
                                self.client.futures_create_order(
                                    symbol=sym,
                                    side="BUY",
                                    type="MARKET",
                                    quantity=orphan_qty,
                                    positionSide="SHORT",
                                )
                                logger.info(f"[{sym}] Orphaned SHORT closed successfully")
                                self.discord.send(
                                    content=(f"🧹 **[{sym}] Orphaned SHORT 청산** — qty={orphan_qty}, 롤백 루프 방지")
                                )
                    except Exception as e:
                        logger.error(f"[{sym}] Failed to close orphaned counterpart: {e}")

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

    def _post_restart_reconcile(self) -> None:
        """Post-restart reconciliation: verify exchange state matches internal state.

        Called once after _load_state(). Ensures:
        1. API connectivity (gate)
        2. Reconciliation cooldown (skip if < 60s since last)
        3. Restart counter persistence
        4. Position qty verification
        5. Bracket (SL/TP) verification
        6. Orphan detection (alert-only, no auto-close)
        """
        if self.paper_mode:
            return

        # 1. API connectivity gate
        api_ok = False
        for attempt in range(3):
            try:
                self.client.futures_time()
                api_ok = True
                break
            except Exception as e:
                backoff = [5, 15, 45][attempt]
                logger.warning(f"API connectivity check failed (attempt {attempt + 1}/3): {e}. Retrying in {backoff}s")
                time.sleep(backoff)

        if not api_ok:
            logger.critical("API connectivity gate FAILED after 3 retries. Entering safe mode.")
            self._safe_mode = True
            self.discord.send(content="🚨 **API 연결 실패** — 3회 재시도 후 실패. Safe mode 진입. 신규 진입 중단.")
            return

        # 2. Reconciliation cooldown
        if self._last_full_reconcile_at:
            try:
                last_dt = datetime.fromisoformat(self._last_full_reconcile_at)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                if elapsed < 60:
                    logger.info(f"Reconciliation cooldown: last reconcile {elapsed:.0f}s ago (< 60s). Skipping.")
                    return
            except Exception:
                pass

        # 3. Increment restart_count FIRST (before any trading logic)
        self._restart_count = getattr(self, "_restart_count", 0) + 1
        self._last_restart_at = datetime.now(timezone.utc).isoformat()
        self._save_state()  # Persist immediately
        logger.info(f"Restart #{self._restart_count} recorded at {self._last_restart_at}")

        # 4-6. Full reconciliation
        try:
            # Query exchange positions
            exchange_positions = self.client.futures_position_information()
            exchange_map: dict[str, float] = {}
            for ep in exchange_positions:
                amt = float(ep.get("positionAmt", 0))
                if abs(amt) < 1e-8:
                    continue
                sym = ep["symbol"]
                ps = ep.get("positionSide", "BOTH")
                key = f"{sym}:{ps}" if ps in ("LONG", "SHORT") else sym
                exchange_map[key] = amt

            # 4. Position qty verification
            for pos_key in list(self.strategy.active_positions):
                pos = self.strategy._positions.get(pos_key)
                if not pos:
                    continue
                raw_sym = pos.get("symbol", pos_key)
                api_sym = raw_sym.split(":")[0] if ":" in raw_sym else raw_sym
                pos_side = pos.get("position_side")
                lookup_key = f"{api_sym}:{pos_side}" if pos_side in ("LONG", "SHORT") else api_sym

                if lookup_key not in exchange_map:
                    # Position in state but not on exchange — mark reconcile_pending during startup grace
                    if getattr(self, "_startup_cycles_remaining", 0) > 0:
                        pos["_reconcile_pending"] = True
                        logger.warning(f"[{pos_key}] Not found on exchange — marked reconcile_pending (startup grace)")
                    continue

                # Qty mismatch check
                exchange_qty = abs(exchange_map[lookup_key])
                internal_qty = pos["notional"] / pos["entry_price"]
                qty_diff_pct = abs(exchange_qty - internal_qty) / max(internal_qty, 1e-8) * 100
                if qty_diff_pct > 1.0:  # 1% threshold
                    logger.warning(
                        f"[{pos_key}] POST-RESTART QTY MISMATCH: internal={internal_qty:.6f} "
                        f"exchange={exchange_qty:.6f} diff={qty_diff_pct:.1f}%. Correcting."
                    )
                    pos["notional"] = exchange_qty * pos["entry_price"]
                    self.discord.send(
                        content=(
                            f"⚠️ **[{pos_key}] 수량 불일치 보정**: 내부={internal_qty:.6f} → 거래소={exchange_qty:.6f}"
                        )
                    )

            # 5. Bracket verification (check SL/TP algo orders exist)
            try:
                open_algo_orders = self._algo_api.list_open_orders()
                algo_order_symbols: set[str] = set()
                if isinstance(open_algo_orders, dict):
                    for order in open_algo_orders.get("orders", []):
                        algo_order_symbols.add(order.get("symbol", ""))
                elif isinstance(open_algo_orders, list):
                    for order in open_algo_orders:
                        algo_order_symbols.add(order.get("symbol", ""))
            except Exception as e:
                logger.warning(f"Failed to query algo orders for bracket verification: {e}")
                algo_order_symbols = set()

            for pos_key in list(self.strategy.active_positions):
                pos = self.strategy._positions.get(pos_key)
                if not pos:
                    continue
                sl_oid = pos.get("server_sl_order_id")
                if not sl_oid:
                    logger.warning(f"[{pos_key}] No SL bracket — will be placed during next cycle")
                    self.discord.send(content=f"⚠️ **[{pos_key}] SL 브라켓 누락** — 다음 사이클에서 재배치 예정")

            # 6. Orphan detection: exchange positions NOT in state → ALERT ONLY
            tracked_keys: set[str] = set()
            for pos_key in self.strategy.active_positions:
                pos = self.strategy._positions.get(pos_key)
                if not pos:
                    continue
                raw_sym = pos.get("symbol", pos_key)
                api_sym = raw_sym.split(":")[0] if ":" in raw_sym else raw_sym
                pos_side = pos.get("position_side")
                if pos_side in ("LONG", "SHORT"):
                    tracked_keys.add(f"{api_sym}:{pos_side}")
                else:
                    tracked_keys.add(api_sym)

            for ex_key, ex_amt in exchange_map.items():
                if ex_key in tracked_keys:
                    continue
                # ALERT ONLY — do NOT auto-close, do NOT auto-adopt
                logger.error(
                    f"[{ex_key}] ORPHAN DETECTED: Exchange position qty={abs(ex_amt):.6f} "
                    f"has no internal state. Alert-only (no auto-close)."
                )
                self.discord.send(
                    content=(
                        f"🔍 **[{ex_key}] 고아 포지션 발견** — "
                        f"거래소에 qty={abs(ex_amt):.4f} 존재, 내부 상태 없음.\n"
                        f"`!adopt {ex_key}` 또는 `!close-orphan {ex_key}` 명령으로 처리해주세요."
                    )
                )
                try:
                    if self._recon_journal:
                        from quantbox.execution.recon_journal import ReconEventType

                        self._recon_journal.record(
                            event_type=ReconEventType.REVERSE_GHOST,
                            symbol=ex_key.split(":")[0],
                            action="orphan_alert_only",
                            result="alert_sent",
                            details={"qty": abs(ex_amt), "position_key": ex_key},
                        )
                except Exception:
                    pass

            # Stale inflight order adoption
            if self._inflight:
                stale = self._inflight.get_stale_orders(max_age_s=300)
                for key, order in stale.items():
                    if order.intent == OrderIntent.ENTRY:
                        api_sym = order.symbol
                        pos_side = order.position_side
                        lookup = f"{api_sym}:{pos_side}" if pos_side in ("LONG", "SHORT") else api_sym
                        if lookup in exchange_map:
                            logger.info(f"Stale entry order {key}: position exists on exchange — clearing inflight")
                        else:
                            logger.info(f"Stale entry order {key}: no position on exchange — discarding")
                        self._inflight.confirm(key)
                    elif order.intent in (OrderIntent.BRACKET_SL, OrderIntent.BRACKET_TP):
                        logger.info(f"Stale bracket order {key}: clearing inflight (brackets verified separately)")
                        self._inflight.confirm(key)

            # Update last_full_reconcile_at
            self._last_full_reconcile_at = datetime.now(timezone.utc).isoformat()
            self._save_state()
            logger.info("Post-restart reconciliation completed successfully")

        except Exception as e:
            logger.error(f"Post-restart reconciliation error: {e}", exc_info=True)
            self.discord.send(content=f"⚠️ **재시작 후 조정 오류**: {str(e)[:300]}")

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
            "daily_loss_blocked": self._daily_loss_blocked,
            "daily_realized_pnl": self._daily_realized_pnl,
            "daily_loss_date": self._daily_loss_date,
            "restart_count": self._restart_count,
            "last_restart_at": self._last_restart_at,
            "last_full_reconcile_at": self._last_full_reconcile_at,
        }
        # v14.0: Persist MA cluster cooldown for crash recovery.
        # Without this, a restart during an active cluster episode would
        # re-fire a duplicate signal on the same cluster.
        if hasattr(self.strategy, "_ma_cluster_cooldown"):
            state["ma_cluster_cooldown"] = {sym: int(arr[0]) for sym, arr in self.strategy._ma_cluster_cooldown.items()}
        if hasattr(self.strategy, "_straddle_rollback_cooldown"):
            active = {s: v for s, v in self.strategy._straddle_rollback_cooldown.items() if v > 0}
            if active:
                state["straddle_rollback_cooldown"] = active
        filepath = PAPER_STATE_FILE if self.paper_mode else STATE_FILE
        try:
            save_scalper_state(state, filepath=filepath)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    @staticmethod
    def _validate_restored_position(sym: str, pos: dict) -> tuple[bool, str]:
        """T1-2: Validate a position dict loaded from state file before restoring.

        Returns (is_valid, error_reason). If False, the position is skipped
        and a Discord alert is sent. Prevents ZeroDivisionError and silent
        corruption from propagating into live trading logic.
        """
        entry_price = pos.get("entry_price", 0)
        if not isinstance(entry_price, (int, float)) or entry_price != entry_price or entry_price <= 0:
            return False, f"entry_price={entry_price!r} must be > 0"

        notional = pos.get("notional", 0)
        if not isinstance(notional, (int, float)) or notional != notional or notional <= 0:
            return False, f"notional={notional!r} must be > 0"

        leverage = pos.get("leverage", 1)
        if not isinstance(leverage, (int, float)) or not (1 <= float(leverage) <= 125):
            return False, f"leverage={leverage!r} must be in [1, 125]"

        signal = pos.get("signal")
        if signal not in (-1, 0, 1, 2):
            return False, f"signal={signal!r} must be in {{-1, 0, 1, 2}}"

        # T1-5: Warn on corrupt compound-key symbol fields (Incident #4 guard).
        # Sanitization already happens via api_sym extraction in _load_state;
        # we emit a warning here so the corruption is visible in logs early.
        symbol_field = pos.get("symbol", "")
        if symbol_field and ":" in symbol_field:
            logger.warning(
                f"[{sym}] pos['symbol']={symbol_field!r} contains ':' — corrupt compound key "
                f"(will be sanitized by _load_state, see Incident #4)"
            )

        return True, ""

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

        raw_history = state.get("trade_history", [])
        # A1: Validate trade_history records — drop corrupt entries (missing/non-numeric pnl fields)
        valid_history = []
        for rec in raw_history:
            if not isinstance(rec, dict):
                logger.warning(f"Trade history: dropping non-dict record {rec!r}")
                continue
            pnl_usd = rec.get("pnl_usd")
            pnl_pct = rec.get("pnl_pct")
            if not (isinstance(pnl_usd, (int, float)) and isinstance(pnl_pct, (int, float))):
                logger.warning(f"Trade history: dropping corrupt record (non-numeric pnl) — {rec!r}")
                continue
            valid_history.append(rec)
        if len(valid_history) < len(raw_history):
            logger.warning(
                f"Trade history: dropped {len(raw_history) - len(valid_history)} corrupt records "
                f"({len(valid_history)}/{len(raw_history)} kept)"
            )
        self._trade_history = valid_history
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

        # A2: Guard — HWM/initial_balance corrupt (0 or negative) → reset to current balance
        # Prevents ZeroDivisionError in (hwm - balance) / hwm and DD halt triggering immediately
        if self._high_water_mark <= 0 or self._initial_balance <= 0:
            balance = self.get_balance()
            if self._high_water_mark <= 0:
                logger.warning(
                    f"Restored HWM={self._high_water_mark:.2f} invalid — resetting to balance ${balance:.2f}"
                )
                self._high_water_mark = balance
            if self._initial_balance <= 0:
                logger.warning(
                    f"Restored initial_balance={self._initial_balance:.2f} invalid — resetting to balance ${balance:.2f}"
                )
                self._initial_balance = balance

        for sym, pos in positions.items():
            # T1-2: Validate critical fields before restoring to prevent
            # ZeroDivisionError, -1111 API errors, or silent bad state.
            valid, reason = self._validate_restored_position(sym, pos)
            if not valid:
                logger.error(f"[{sym}] SKIPPING corrupted position from state file: {reason}")
                self.discord.send(
                    content=(f"⚠️ **상태파일 오류** [{sym}] — {reason}. 해당 포지션 복원 건너뜀. 수동 확인 필요.")
                )
                continue

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
                strategy_type=normalize_strategy_type(pos.get("strategy_type", "vr_directional")),
                is_straddle=pos.get("is_straddle", False),
                server_trail_order_id=pos.get("server_trail_order_id"),
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
                # r65: Restore trail ratchet level to avoid redundant API call on restart
                self.strategy._positions[sym]["_trail_sl_ratcheted"] = pos.get("_trail_sl_ratcheted", -999.0)
                # R43: Trailing stop state for directional trades
                self.strategy._positions[sym]["trail_active"] = pos.get("trail_active", False)
                self.strategy._positions[sym]["peak_gain"] = pos.get("peak_gain", 0.0)
                self.strategy._positions[sym]["is_straddle"] = pos.get("is_straddle", False)
                # Restore straddle pair linkage (Step 3 — pair coordination)
                if pos.get("straddle_pair_id"):
                    self.strategy._positions[sym]["straddle_pair_id"] = pos["straddle_pair_id"]
                if pos.get("straddle_orphan_since_cycle") is not None:
                    self.strategy._positions[sym]["straddle_orphan_since_cycle"] = pos["straddle_orphan_since_cycle"]
                # v14.0: Restore per-signal fields (sl/tp/tf/max_hold/strategy_type).
                # Without this, crash recovery defaults to VR directional params
                # for ALL positions — breaking MA cluster SL/TP/max_hold/exit routing.
                self.strategy._positions[sym]["strategy_type"] = normalize_strategy_type(
                    pos.get("strategy_type", "vr_directional")
                )
                self.strategy._positions[sym]["sl"] = pos.get("sl", self.strategy._positions[sym].get("sl"))
                self.strategy._positions[sym]["tp"] = pos.get("tp", self.strategy._positions[sym].get("tp"))
                self.strategy._positions[sym]["tf"] = pos.get("tf", self.strategy._positions[sym].get("tf"))
                self.strategy._positions[sym]["max_hold"] = pos.get(
                    "max_hold", self.strategy._positions[sym].get("max_hold")
                )

        # balance_before backfill: 이전 버전에서 balance_before 없이 저장된 포지션 복구
        _backfill_balance: float | None = None
        for sym, pos in self.strategy._positions.items():
            if pos.get("balance_before") is None:
                if _backfill_balance is None:
                    _backfill_balance = self.get_balance()
                pos["balance_before"] = _backfill_balance
                logger.info(f"[{sym}] balance_before backfilled: ${_backfill_balance:,.2f}")

        # Step 3.2: 구 상태 포맷 스트래들 페어 재연결 (straddle_pair_id 없는 경우)
        for sym, pos in self.strategy._positions.items():
            if pos.get("is_straddle") and not pos.get("straddle_pair_id"):
                api_sym = pos.get("symbol", sym.split(":")[0])
                my_side = sym.split(":")[-1] if ":" in sym else None
                opposite_side = "SHORT" if my_side == "LONG" else "LONG" if my_side == "SHORT" else None
                if opposite_side:
                    counterpart_key = f"{api_sym}:{opposite_side}"
                    counterpart = self.strategy._positions.get(counterpart_key)
                    if counterpart and counterpart.get("is_straddle"):
                        pair_id = str(uuid.uuid4())[:8]
                        pos["straddle_pair_id"] = pair_id
                        counterpart["straddle_pair_id"] = pair_id
                        logger.info(f"[{sym}] 스트래들 페어 재연결: {counterpart_key} (pair_id={pair_id})")
                    else:
                        # 카운터파트 없음 — 즉시 고아 타이머 시작
                        pos["straddle_orphan_since_cycle"] = 0
                        logger.warning(f"[{sym}] 스트래들 레그 카운터파트 없음 — 고아로 표시됨")

        # Restore DailyBreaker state (survives crash/restart within same UTC day)
        self._daily_loss_blocked = state.get("daily_loss_blocked", False)
        self._daily_realized_pnl = state.get("daily_realized_pnl", 0.0)
        self._daily_loss_date = state.get("daily_loss_date", "")

        # Restore restart tracking
        self._restart_count = state.get("restart_count", 0)
        self._last_restart_at = state.get("last_restart_at")
        self._last_full_reconcile_at = state.get("last_full_reconcile_at")

        logger.info(f"Restored {len(positions)} positions, {len(self._trade_history)} trade records")

        # v14.0: Restore MA cluster cooldown state
        if hasattr(self.strategy, "_ma_cluster_cooldown"):
            cooldown_data = state.get("ma_cluster_cooldown", {})
            for sym, val in cooldown_data.items():
                arr: np.ndarray = np.zeros(1, dtype=np.int64)
                arr[0] = val
                self.strategy._ma_cluster_cooldown[sym] = arr
            if cooldown_data:
                logger.info(f"Restored MA cluster cooldown for {len(cooldown_data)} symbols")

        # v15.0: Restore straddle rollback cooldown
        if hasattr(self.strategy, "_straddle_rollback_cooldown"):
            rb_cooldown = state.get("straddle_rollback_cooldown", {})
            for sym, val in rb_cooldown.items():
                self.strategy._straddle_rollback_cooldown[sym] = val
            if rb_cooldown:
                logger.info(f"Restored straddle rollback cooldown for {len(rb_cooldown)} symbols: {rb_cooldown}")

        # Startup exchange cross-check: warn immediately if exchange has
        # positions not present in restored state (reverse ghost candidates).
        if not self.paper_mode:
            self._startup_exchange_crosscheck()

        # ── InFlight: check for stale orders from previous session ──
        try:
            if self._inflight is not None:
                stale = self._inflight.get_stale_orders(max_age_s=300)
                if stale:
                    logger.warning(f"Found {len(stale)} stale in-flight orders from previous session")
                    for key, order in stale.items():
                        logger.warning(f"  Stale: {key} → {order.symbol} {order.intent.value}")
                    self._inflight.clear_all()
        except Exception as e:
            logger.error(f"InFlight stale check error: {e}")

    def _startup_exchange_crosscheck(self) -> None:
        """Cross-check exchange positions against restored internal state at startup.

        Detects positions that exist on the exchange but were not in the state
        file — these are 'reverse ghosts' with no SL/TP protection.
        Logs a WARNING and sends a Discord alert immediately so the operator
        is aware before the first reconciliation cycle fires.
        """
        try:
            exchange_positions = self.client.futures_position_information()
        except Exception as exc:
            logger.warning(f"Startup exchange cross-check failed: {exc}")
            return

        for p in exchange_positions:
            amt = float(p.get("positionAmt", "0"))
            if amt == 0:
                continue
            sym = p["symbol"]
            pos_side = p.get("positionSide", "BOTH")
            check_key = f"{sym}:{pos_side}" if pos_side in ("LONG", "SHORT") else sym
            if check_key not in self.strategy.active_positions:
                notional = abs(float(p.get("notional", "0")))
                upnl = float(p.get("unRealizedProfit", "0"))
                logger.warning(
                    f"[{check_key}] STARTUP CROSSCHECK: Untracked position on exchange! "
                    f"notional=${notional:.2f} uPnL={upnl:+.2f} — "
                    f"No SL/TP. Will be auto-closed in first reconciliation cycle."
                )
                self.discord.send(
                    content=(
                        f"⚠️ **Startup: Untracked position [{check_key}]** — "
                        f"notional=${notional:.2f} uPnL={upnl:+.2f}. "
                        f"No SL/TP protection. Auto-close pending (next reconciliation cycle)."
                    )
                )

    # ==================== Entry Points ====================

    def _ensure_position_mode(self) -> None:
        """Ensure Binance account position mode matches config.

        Hedge mode (dualSidePosition=true) is REQUIRED for VolStraddle
        S_d8 straddle to open simultaneous LONG+SHORT on the same symbol.
        Without this, all positionSide=LONG/SHORT orders fail with -4059.

        One-Way mode is required for RubberBand (uses reduceOnly).
        """
        if self.paper_mode:
            return

        want_hedge = self.vs_config.hedge_mode

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
        self._startup_cycles_remaining = 2
        self._post_restart_reconcile()

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
        self._startup_cycles_remaining = 2
        self._post_restart_reconcile()

        # Refresh balance after state load (paper_balance may have been
        # restored from saved state, overriding the settings default).
        balance = self.get_balance()

        mode_str = "PAPER" if self.paper_mode else "LIVE"
        n_symbols = len(self._cfg.symbol_whitelist) if self._cfg.symbol_whitelist else 0
        self._print_startup_banner(balance, n_symbols)
        logger.info(f"Starting continuous trading ({mode_str})")
        logger.info(f"Initial balance: ${balance:.2f}")
        sd_notify("READY=1")

        # Start Discord bot for interactive commands
        if self._bot_runner:
            self._bot_runner.start()
            logger.info("Discord Bot started for interactive commands")

        n_symbols = len(self._cfg.symbol_whitelist) if self._cfg.symbol_whitelist else 0
        commit_str = f"`{GIT_COMMIT['short']}`" if GIT_COMMIT["short"] else "unknown"
        n_recovered = self.strategy.n_active

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

        # Pre-fetch exchange info
        self.get_symbol_info("BTCUSDT")

        # Get trading universe (refresh every 6 hours)
        symbols = self.get_trading_symbols()
        last_universe_refresh = time.monotonic()
        UNIVERSE_REFRESH_INTERVAL = 6 * 3600  # 6 hours

        while not self._stop_requested:
            try:
                # ── Heartbeat stall check ──
                try:
                    if (
                        self._heartbeat is not None
                        and self._heartbeat.is_stalled()
                        and not self._heartbeat.stall_alerted
                    ):
                        stall_s = self._heartbeat.seconds_since_last_beat()
                        logger.critical(f"Main loop stall detected: {stall_s:.0f}s since last cycle")
                        self.discord.notify_heartbeat(stall_s)
                        self._heartbeat.stall_alerted = True
                except Exception as e:
                    logger.error(f"Heartbeat stall check error: {e}")

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

                # Reset error counter on successful cycle
                self._consecutive_errors = 0
                if self._safe_mode:
                    self._safe_mode = False
                    logger.info("Safe mode deactivated: successful cycle completed")

                # Auto-reset restart_count after 16 consecutive clean cycles
                self._clean_cycles_count += 1
                if self._clean_cycles_count >= 16 and self._restart_count > 0:
                    logger.info(
                        f"16 consecutive clean cycles — resetting restart_count from {self._restart_count} to 0"
                    )
                    self._restart_count = 0
                    self._clean_cycles_count = 0
                    self._save_state()

            except KeyboardInterrupt:
                logger.info("Shutdown requested (KeyboardInterrupt)")
                break
            except Exception as e:
                self._consecutive_errors += 1
                self._clean_cycles_count = 0  # Reset clean streak on any error
                logger.error(f"Cycle error ({self._consecutive_errors}/3): {e}", exc_info=True)
                self.discord.send(
                    embed={
                        "title": "🚨 사이클 오류",
                        "color": 0xFF1744,
                        "description": (
                            f"포지션 {self.strategy.n_active}개 활성 중\n"
                            f"연속오류: {self._consecutive_errors}/3\n"
                            f"```\n{str(e)[:400]}\n```"
                        ),
                        "footer": {"text": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                    }
                )

                # Safe mode after 3 consecutive failures
                if self._consecutive_errors >= 3 and not self._safe_mode:
                    self._safe_mode = True
                    logger.critical("SAFE MODE ACTIVATED: 3 consecutive cycle errors. Skipping new entries.")
                    self.discord.send(content="⛔ **SAFE MODE** — 3연속 오류. 신규 진입 중단, 기존 포지션만 관리.")

                # Exponential backoff for connection errors
                if isinstance(e, (ConnectionError, TimeoutError)):
                    backoff = min(30 * (2 ** (self._consecutive_errors - 1)), 120)
                    logger.info(f"Connection error backoff: {backoff}s")
                    time.sleep(backoff)
                else:
                    time.sleep(60)

        # Clean shutdown
        logger.info("Shutting down...")
        self._save_state()

        # Stop Discord bot
        if self._bot_runner:
            self._bot_runner.stop()

        # Close ReconJournal DB
        try:
            if self._recon_journal is not None:
                self._recon_journal.close()
        except Exception:
            pass

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
    parser = argparse.ArgumentParser(description="QuantBox Scalper — Production strategy: vol_straddle")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode (no real orders)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--lookback", type=int, default=None, help="Signal lookback bars (default: from .env/config)")
    parser.add_argument("--threshold", type=float, default=None, help="Min move to trigger (default: from .env/config)")
    parser.add_argument(
        "--max-slots", type=int, default=None, help="Max concurrent positions (default: from .env/config)"
    )
    parser.add_argument(
        "--strategy",
        choices=["vol_straddle"],
        default="vol_straddle",
        help="Trading strategy: vol_straddle (default, production)",
    )
    args = parser.parse_args()

    setup_logging(strategy_name=args.strategy, paper_mode=args.paper, git_commit=GIT_COMMIT)

    vs_config = VolStraddleConfig(**({"max_concurrent": args.max_slots} if args.max_slots is not None else {}))

    scalper = RubberBandScalper(
        paper_mode=args.paper,
        strategy_name=args.strategy,
        vs_config=vs_config,
    )

    if args.once:
        scalper.run_once()
    else:
        scalper.run_continuous()


if __name__ == "__main__":
    main()
