"""ScalperProtocol — structural type for the Mixin-composed Scalper class.

The live trading bot is assembled from multiple Mixin classes (PositionMixin,
ReconciliationMixin, BracketMixin, DataMixin, LoopMixin, CommandsMixin) that
call each other's methods via ``self``.  mypy cannot infer these cross-mixin
references and raises hundreds of ``attr-defined`` errors.

The standard solution is to annotate cross-mixin method arguments as
``self: ScalperProtocol`` using Python's ``Protocol`` pattern.  The Protocol
declares the complete interface of the composed Scalper class so that each
Mixin can type-check against it without importing the concrete class (which
would create circular imports).

Usage in Mixin files::

    from __future__ import annotations
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from quantbox.core.protocol import ScalperProtocol

    class ReconciliationMixin:
        def _sync_server_orders(self: ScalperProtocol) -> None:  # cross-mixin
            self.strategy.get_positions()  # mypy now resolves .strategy
            ...

Only annotate methods that reference cross-mixin attributes.  Methods that
only use ``self``'s own attributes do not need the annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime as _datetime


class ScalperProtocol(Protocol):
    """Structural type for the fully-composed RubberBandScalper class.

    Every attribute and method here corresponds to something defined in one
    of the Mixin classes or in ``scalper.py`` itself.  This is intentionally
    permissive (``Any`` for complex types) — the goal is to resolve
    ``attr-defined`` errors, not to achieve full parametric typing.
    """

    # ── Core state ────────────────────────────────────────────────────────
    client: Any
    """binance.client.Client instance."""

    config: Any
    """Active strategy config (RubberBandConfig or VolStraddleConfig)."""

    vs_config: Any
    """VolStraddleConfig — present when strategy_name == 'vol_straddle'."""

    strategy: Any
    """Strategy instance (RubberBandStrategy or VolStraddleStrategy)."""

    strategy_name: str
    """``'rubberband'`` or ``'vol_straddle'``."""

    paper_mode: bool
    """True = paper trading (no real orders placed)."""

    discord: Any
    """DiscordNotifier instance."""

    # ── Balance / capital tracking ────────────────────────────────────────
    _paper_balance: float
    _initial_balance: float
    _high_water_mark: float

    # ── Position / trade state ────────────────────────────────────────────
    _trade_history: list[dict[str, Any]]  # TradeRecord dicts

    # ── Session control ───────────────────────────────────────────────────
    _paused: bool
    _drawdown_halt: bool
    _stop_requested: bool
    _cycle_count: int

    # ── Bot runner (Discord command listener) ─────────────────────────────
    _bot_runner: Any

    # ── Internal config cache ─────────────────────────────────────────────
    _cfg: Any
    """Cached config shorthand used in loop_mixin."""

    # ── Symbol info cache (DataMixin) ─────────────────────────────────────
    _symbol_info_cache: dict[str, Any]
    _exchange_info_fetched: bool

    # ── Trade journal (optional) ──────────────────────────────────────────
    _journal: Any

    # ── Recon journal (optional) ────────────────────────────────────────
    _recon_journal: Any

    # ── Exchange notional cache (set by _reconcile_positions each cycle) ──
    _cached_exchange_notionals: dict[str, float]

    # ── Methods: balance & symbol info ───────────────────────────────────
    _cycle_balance: float | None

    def get_balance(self) -> float:
        """Return current USDT futures wallet balance."""
        ...

    def _get_cycle_balance(self) -> float:
        """사이클당 1회 balance API 호출 캐시 — 중복 API 호출 방지."""
        ...

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Return cached symbol metadata (step_size, tick_size, max_notional …)."""
        ...

    def _preload_max_notional(self) -> None:
        """Bulk-preload max_notional for all symbols at session start."""
        ...

    def _round_qty(self, qty: float, symbol: str) -> float:
        """Round quantity to exchange step_size for the given symbol."""
        ...

    def download_klines(
        self,
        symbol: str,
        n_bars: int = 200,
        interval: str = "15m",
        full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...] | None:
        """Download completed OHLCV klines from Binance."""
        ...

    # ── Methods: order placement ──────────────────────────────────────────
    def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int = 1,
        reduce_only: bool = False,
        position_side: str | None = None,
    ) -> dict[str, Any] | None:
        """Place order via LimitOrderChaser; return fill dict or None."""
        ...

    def _place_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        trigger_price: float,
        limit_price: float | None = None,
        label: str = "order",
        position_side: str | None = None,
    ) -> str | None:
        """Place server-side algo (STOP_MARKET / TAKE_PROFIT_MARKET) order."""
        ...

    def _cancel_server_order(self, symbol: str, order_id: str, label: str = "order") -> bool:
        """Cancel a server-side algo order by ID. Returns True on success or benign error."""
        ...

    def _lookup_actual_fill(self, symbol: str, expected_qty: str, order_type: str) -> dict[str, Any] | None:
        """Look up actual fill price/qty from recent account trades."""
        ...

    def _place_server_stop_loss(
        self,
        symbol: str,
        signal: int,
        quantity: float,
        entry_price: float,
        move_size: float,
        stop_mult: float,
        position_side: str | None = None,
    ) -> str | None:
        """Place server-side SL algo order; return algo order ID or None."""
        ...

    def _place_server_take_profit(
        self,
        symbol: str,
        signal: int,
        quantity: float,
        entry_price: float,
        move_size: float,
        target_retrace: float,
        position_side: str | None = None,
    ) -> str | None:
        """Place server-side TP algo order; return algo order ID or None."""
        ...

    def _cancel_server_stop_loss(self, symbol: str, order_id: str) -> bool:
        """Cancel a server-side SL algo order. Returns True on success or benign error."""
        ...

    def _cancel_server_take_profit(self, symbol: str, order_id: str) -> bool:
        """Cancel a server-side TP algo order. Returns True on success or benign error."""
        ...

    def _get_trail_params(
        self,
        strategy_type: str,
        signal: int,
        entry_price: float,
    ) -> tuple[float, float, str]:
        """Calculate TRAILING_STOP_MARKET params. Returns (callbackRate, activationPrice, closeSide)."""
        ...

    def _place_trailing_stop_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float,
        activation_price: float,
        position_side: str | None = None,
    ) -> str | None:
        """Place TRAILING_STOP_MARKET via Algo Order API. Returns algoId or None."""
        ...

    def _cancel_trailing_stop_market(self, symbol: str, order_id: str) -> bool:
        """Cancel a TRAILING_STOP_MARKET order (algo API). Returns True on success or benign error."""
        ...

    def _check_trail_order_status(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        """Check if TRAILING_STOP_MARKET order has been filled (algo API). Returns status dict or None."""
        ...

    def _place_trail_regular(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float,
        activation_price: float | None = None,
        position_side: str | None = None,
    ) -> str | None:
        """Place TRAILING_STOP_MARKET. Delegates to Algo API. Returns algoId or None."""
        ...

    def _cancel_trail_regular(self, symbol: str, order_id: str) -> bool:
        """Cancel a trail order. Delegates to Algo API. Returns True on success."""
        ...

    def _check_trail_regular(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        """Check trail order status. Delegates to Algo API. Returns status dict or None."""
        ...

    def _verify_algo_order_active(self, algo_id: str, symbol: str) -> str | None:
        """Verify an algo order is still active. Returns status string or None."""
        ...

    def _check_bracket_staleness(self) -> None:
        """Check if any SL algo orders have expired or been cancelled."""
        ...

    def _verify_ratchet(self, sym: str, pos: dict, expected_price: float) -> bool:
        """Verify a ratchet SL update was accepted. Returns True if OK."""
        ...

    def _check_order_filled(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        """Check if a server-side algo order has been filled; return fill dict or None."""
        ...

    # ── Methods: position lifecycle ───────────────────────────────────────
    def _get_pair_counterpart(self, pos_key: str) -> str | None:
        """Find counterpart position key for a straddle pair."""
        ...

    def _close_position(self, symbol: str, reason: str, current_price: float) -> bool:
        """Close position by key (may be compound 'SYMBOL:SIDE'); return success flag."""
        ...

    # ── Methods: reconciliation (cross-mixin) ─────────────────────────────
    def _record_server_fill(
        self,
        symbol: str,
        fill_price: float,
        fill_qty: float,
        reason: str,
    ) -> None:
        """Record a trade closed by a server-side order (TP or SL)."""
        ...

    def _handle_server_partial_tp_fill(
        self,
        symbol: str,
        pos: dict[str, Any],
        fill_price: float,
        fill_qty: float,
    ) -> None:
        """Handle server TP fill for 50% qty (PBT mode Phase 1 → Phase 2)."""
        ...

    # ── Methods: state persistence ────────────────────────────────────────
    def _save_state(self) -> None:
        """Atomically persist positions + trade history to JSON state file."""
        ...

    # ── Methods: reconciliation (cross-mixin from ReconciliationMixin) ────
    def _sync_server_orders(self) -> None:
        """Check server bracket orders for fills; repair missing brackets."""
        ...

    def _reconcile_positions(self) -> None:
        """Cross-reference internal state with exchange; remove ghost positions."""
        ...

    # ── Methods: bot commands (cross-mixin from CommandsMixin) ────────────
    def _process_bot_commands(self) -> None:
        """Process pending Discord bot commands (pause/stop/close-all etc.)."""
        ...

    def _handle_command(self, cmd: Any) -> str:
        """Dispatch a BotCommand to the appropriate handler."""
        ...

    def _cmd_adopt(self, position_key: str) -> str:
        """Adopt an orphan exchange position into internal state."""
        ...

    def _cmd_close_orphan(self, position_key: str) -> str:
        """Market close an orphan exchange position."""
        ...

    def _cmd_status(self) -> str: ...
    def _cmd_close_all(self) -> str: ...
    def _cmd_close_slot(self, slot_id: int) -> str: ...
    def _cmd_pause(self) -> str: ...
    def _cmd_resume(self) -> str: ...
    def _cmd_stop(self) -> str: ...
    def _cmd_pnl(self) -> str: ...
    def _cmd_blacklist(self, args: dict[str, Any] | None = None) -> str: ...

    # ── Methods: loop / timing (cross-mixin from LoopMixin) ───────────────
    def _log_vr_distribution(self) -> None:
        """Log VR signal distribution for the current session."""
        ...

    def _next_bar_time(self) -> "_datetime":
        """Return the datetime of the next 15m bar close."""
        ...
