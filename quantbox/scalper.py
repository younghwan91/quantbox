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
from quantbox.execution.preflight import OrderPreflight
from quantbox.execution.reconciliation import ReconciliationMixin
from quantbox.core.trade_journal import TradeJournal
from quantbox.core.divergence_monitor import DivergenceMonitor

# ── Reliability modules (Phase 3 integration) ──
from pathlib import Path

from quantbox.core.heartbeat import Heartbeat
from quantbox.core.rate_limiter import RateLimiter
from quantbox.execution.exposure import ExposureGuard
from quantbox.execution.inflight import InFlightTracker, OrderIntent, PendingOrder
from quantbox.execution.recon_journal import ReconJournal

load_dotenv()

# ──────────────────────── Logging ──────────────────────────────────────

logger = logging.getLogger("Scalper")


# ──────────────── Diagnostic Logger ─────────────────────
from quantbox.utils.diagnostics import _diag_logger  # noqa: E402

# ──────────────────────── Helpers ──────────────────────────────────────────


def _healed_strategy_type(pos: dict) -> str:
    """Return strategy_type, auto-healing straddle positions mislabeled as vr_squeeze.

    Positions opened before commit 0ac2e2b (strategy_type propagation fix) were
    stored with strategy_type='vr_squeeze' even when is_straddle=True.  Detect
    and correct on state restore so the dashboard / journal always show 'straddle'.
    """
    raw = pos.get("strategy_type", "vr_squeeze")
    if pos.get("is_straddle", False) and raw == "vr_squeeze":
        return "straddle"
    return raw


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
        vs_config: VolStraddleConfig | None = None,
        journal_db_path: str | None = None,
    ):
        self.paper_mode = paper_mode
        self.config = config or RubberBandConfig()
        self.strategy_name = strategy_name

        _valid_strategies = {"rubberband", "vol_straddle"}
        if strategy_name not in _valid_strategies:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Valid: {sorted(_valid_strategies)}")

        self.strategy: TradingStrategy
        if strategy_name == "vol_straddle":
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

        # ── Reliability modules (Phase 3) ──
        try:
            self._rate_limiter = RateLimiter()
        except Exception as e:
            logger.error(f"RateLimiter init failed: {e}")
            self._rate_limiter = None

        try:
            self._heartbeat = Heartbeat(stall_threshold_s=300.0)
        except Exception as e:
            logger.error(f"Heartbeat init failed: {e}")
            self._heartbeat = None

        try:
            _max_mult = self.vs_config.max_exposure_multiplier if strategy_name == "vol_straddle" else 35.0
            self._exposure_guard = ExposureGuard(max_multiplier=_max_mult)
        except Exception as e:
            logger.error(f"ExposureGuard init failed: {e}")
            self._exposure_guard = None

        try:
            inflight_path = Path("data/state/inflight_orders.json")
            self._inflight = InFlightTracker(persist_path=inflight_path)
        except Exception as e:
            logger.error(f"InFlightTracker init failed: {e}")
            self._inflight = None

        try:
            self._recon_journal = ReconJournal(Path("data/journal/recon.db"))
        except Exception as e:
            logger.error(f"ReconJournal init failed: {e}")
            self._recon_journal = None

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
        if self.strategy_name == "vol_straddle":
            pass  # VolStraddle doesn't use BTC as filter
        elif self.config.mtf_confirmation_enabled or self.config.htf_independent_enabled:
            btc_close_4h = self.download_klines("BTCUSDT", 100, interval="4h")

        # v11.2: Sync server-side TP/SL orders (check if filled intra-bar)
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
                elif exit_reason in ("confirm_entry", "pyramid", "cancel_exploration"):
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
            n_bars = 500 if self.strategy_name == "vol_straddle" else 200
            result = self.download_klines(sym, n_bars, full=True)
            if result is None:
                continue
            close, high, low, volume, taker_buy = result
            if len(close) < self._cfg.lookback + 1:
                continue

            # Wire up volume/taker/high/low data for v9 confluence scoring
            self.strategy.update_market_data(sym, volume, taker_buy, high, low)

            # v7.0: Download 4h data for MTF confirmation
            if self.strategy_name == "vol_straddle":
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
                    candidates.append((sym, 1, move, close, True))
                    candidates.append((sym, -1, move, close, True))
                else:
                    candidates.append((sym, signal, move, close, False))

        # Prioritize strongest moves
        candidates.sort(key=lambda x: x[2], reverse=True)

        # ── Diagnostic: log scan summary every cycle ──
        top_movers = sorted(moves_seen, key=lambda x: x[1], reverse=True)[:3]
        if self.strategy_name == "vol_straddle":
            # VR strategy: show VR threshold + actual VR values
            print(format_scan_summary(len(candidates), top_movers, 0.0))
            if candidates:
                cand_strs = [f"{s}({m * 100:.1f}%)" for s, _, m, _, _ in candidates[:3]]
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
                cand_strs = [f"{s}({m * 100:.1f}%)" for s, _, m, _, _ in candidates[:3]]
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
        straddle_opened: dict[str, set[int]] = {}  # {sym: {signals opened}} for rollback
        for sym, signal, move, close, is_straddle in candidates:
            if entries_made >= max_entries:
                break

            # ── G6: BTC Trend Gate — block directional LONG when BTC below MA ──
            # Straddle is direction-neutral (LONG+SHORT) → G6 does not apply.
            if signal == 1 and not is_straddle and btc_ma is not None and btc_close is not None:
                btc_cur = float(btc_close[-1])
                if btc_cur < btc_ma:
                    logger.info(
                        f"[{sym}] G6 LONG suppressed: BTC ${btc_cur:.1f} < "
                        f"MA{self.vs_config.btc_long_ma_period} ${btc_ma:.1f}"
                    )
                    continue

            # ═══ v12.3 PARITY: Direction Diversity Filter ═══
            # Straddle is direction-neutral → diversity filter does not apply.
            if self._cfg.dir_diversity_enabled and not is_straddle:
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
                if is_straddle:
                    straddle_opened.setdefault(sym, set()).add(signal)

        # ── Straddle half-open rollback ──
        # If only one leg of a straddle opened (e.g., SHORT failed -2027),
        # immediately close the orphaned leg to restore delta-neutral intent.
        for sym, signals_opened in straddle_opened.items():
            if len(signals_opened) == 1:
                opened_signal = next(iter(signals_opened))
                side_str = "LONG" if opened_signal == 1 else "SHORT"
                pos_key = f"{sym}:{side_str}"
                logger.error(
                    f"[{sym}] Straddle incomplete: only {side_str} opened (counterpart failed). "
                    f"Rolling back {side_str} to avoid unintended directional exposure."
                )
                self.discord.send(
                    content=(
                        f"⚠️ **[{sym}] Straddle half-open** — {side_str} opened but counterpart failed. "
                        f"Auto-closing {side_str} (rollback)."
                    )
                )
                # Use last close price as rollback price
                close_arr = next((c for s, sig, _, c, is_s in candidates if s == sym and is_s), None)
                rollback_price = float(close_arr[-1]) if close_arr is not None else 0.0
                closed = self._close_position(pos_key, "straddle_rollback", rollback_price)
                if not closed:
                    logger.error(f"[{sym}] Straddle rollback FAILED — manual intervention required!")
                    self.discord.send(
                        content=(f"🚨 **[{sym}] Straddle rollback FAILED** — {side_str} is orphaned. Close manually.")
                    )
                entries_made -= 1

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
                strategy_type=_healed_strategy_type(pos),
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
                # r65: Restore trail ratchet level to avoid redundant API call on restart
                self.strategy._positions[sym]["_trail_sl_ratcheted"] = pos.get("_trail_sl_ratcheted", -999.0)
                # R43: Trailing stop state for directional trades
                self.strategy._positions[sym]["trail_active"] = pos.get("trail_active", False)
                self.strategy._positions[sym]["peak_gain"] = pos.get("peak_gain", 0.0)
                self.strategy._positions[sym]["is_straddle"] = pos.get("is_straddle", False)
                # v14.0: Restore per-signal fields (sl/tp/tf/max_hold/strategy_type).
                # Without this, crash recovery defaults to VR squeeze params
                # for ALL positions — breaking MA cluster SL/TP/max_hold/exit routing.
                self.strategy._positions[sym]["strategy_type"] = _healed_strategy_type(pos)
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
                arr: np.ndarray = np.zeros(1, dtype=np.int64)
                arr[0] = val
                self.strategy._ma_cluster_cooldown[sym] = arr
            if cooldown_data:
                logger.info(f"Restored MA cluster cooldown for {len(cooldown_data)} symbols")

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
        choices=["rubberband", "vol_straddle"],
        default="vol_straddle",
        help="Trading strategy: vol_straddle (default, production) or rubberband",
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
