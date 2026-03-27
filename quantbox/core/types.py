"""Domain type definitions — Single Source of Truth.

All primitive NewTypes, Literal aliases, and TypedDicts used across the
execution layer are defined here.  Importing from this module instead of
using raw ``str`` / ``dict`` lets mypy catch cross-boundary mistakes (e.g.
passing a compound PositionKey where an ApiSymbol is expected) at
pre-commit time rather than at live-trading runtime.

Usage example::

    from quantbox.core.types import ApiSymbol, PositionKey, PositionDict

    def _place_order(self, symbol: ApiSymbol, ...) -> ...: ...
    def _close_position(self, pos_key: PositionKey, ...) -> ...: ...
"""

from __future__ import annotations

from typing import Literal, NewType

from typing import TypedDict

# ── Primitive domain types ────────────────────────────────────────────────────

ApiSymbol = NewType("ApiSymbol", str)
"""Pure Binance API symbol, e.g. ``"SUIUSDT"``.

Must never contain a colon (``':'``).  Compound keys like ``"SUIUSDT:LONG"``
are ``PositionKey`` values and must be split before any API call.

Incident #4 (2026-03-08): compound key was passed directly to
``futures_create_order`` → ``APIError(-1022)`` + bracket repair failure.
"""

PositionKey = NewType("PositionKey", str)
"""Internal position dict key.

One-way mode: ``"SUIUSDT"`` (same as ApiSymbol).
Hedge mode:   ``"SUIUSDT:LONG"`` / ``"SUIUSDT:SHORT"``.

Use ``to_api_symbol(pos_key)`` to obtain the ``ApiSymbol`` for API calls.
"""

Side = Literal["BUY", "SELL"]
"""Order side accepted by Binance futures API."""

PositionSide = Literal["LONG", "SHORT", "BOTH"]
"""Hedge-mode position side (``BOTH`` = one-way mode)."""

SignalValue = Literal[-1, 0, 1, 2]
"""Strategy signal value.

-1 = short entry
 0 = no signal
+1 = long entry
+2 = straddle (vol_straddle strategy: open both LONG and SHORT)
"""

StraddlePairId = NewType("StraddlePairId", str)
"""UUID string linking the LONG and SHORT legs of a straddle pair.

Both legs of the same straddle share the same ``StraddlePairId``.
``None`` for directional (non-straddle) positions.
"""


# ── Utility functions ─────────────────────────────────────────────────────────


def to_api_symbol(key: str, pos: dict | None = None) -> str:
    """Extract Binance API symbol from a position key or position dict.

    Handles both one-way (``"BTCUSDT"``) and hedge-mode (``"BTCUSDT:LONG"``) keys.
    If *pos* is provided and contains a ``"symbol"`` field, uses that first.

    Incident #4 (2026-03-08): compound key sent to API → ``-1022`` error.
    """
    if pos and "symbol" in pos:
        sym: str = str(pos["symbol"])
        return sym.split(":")[0] if ":" in sym else sym
    return key.split(":")[0] if ":" in key else key


def round_to_tick(price: float, tick_size: float, price_precision: int) -> float:
    """Round a price to the exchange tick size.

    Binance rejects orders with prices that are not a multiple of tick_size.
    Uses round-half-even to snap to the nearest tick, then rounds to the
    symbol's price precision to avoid floating-point trailing digits.
    """
    return round(round(price / tick_size) * tick_size, price_precision)


# ── TypedDict: symbol metadata (from exchange_info + leverage brackets) ───────


class SymInfoDict(TypedDict, total=False):
    """Symbol metadata returned by ``DataMixin.get_symbol_info()``.

    ``total=False`` because the cache is built incrementally: exchange_info
    is fetched first (step_size, tick_size, min_qty, min_notional,
    quantity_precision, price_precision) and max_notional is added later via
    bulk leverage-bracket preload or per-symbol lazy fetch.
    """

    step_size: float
    """LOT_SIZE filter step size — minimum order quantity increment."""

    tick_size: float
    """PRICE_FILTER tick size — minimum price increment."""

    min_qty: float
    """LOT_SIZE minimum quantity."""

    min_notional: float
    """MIN_NOTIONAL filter minimum order value in USDT."""

    max_notional: float | None
    """Maximum position notional at current leverage from leverage brackets.

    ``None`` = API fetch attempted but failed or symbol not found.
    Checked before order placement to avoid ``-2027`` position-limit errors.
    """

    quantity_precision: int
    """Number of decimal places for quantity rounding."""

    price_precision: int
    """Number of decimal places for price rounding."""


# ── TypedDict: live position state ────────────────────────────────────────────


class PositionDict(TypedDict, total=False):
    """Position state stored in ``strategy._positions[pos_key]``.

    ``total=False`` because some fields are strategy-specific (vol_straddle
    vs rubberband) and not always present.  Access optional fields with
    ``.get()`` and provide defaults.
    """

    # ── Core ──────────────────────────────────────────────────────────────
    symbol: str
    """Pure Binance API symbol (ApiSymbol).  Never contains ``':'``."""

    signal: int
    """Entry signal: +1 (long) / -1 (short)."""

    entry_price: float
    """Fill price at position open."""

    notional: float
    """Position notional value in USDT at open (qty × entry_price)."""

    leverage: float
    """Applied leverage multiplier."""

    move_size: float
    """Normalised move size used for SL calculation (1.0 for vol_straddle)."""

    # ── Time tracking ─────────────────────────────────────────────────────
    bars_held: int
    """Number of completed 15m bars this position has been open."""

    max_hold: int
    """Maximum bars before forced exit (default 192 = 48 h)."""

    opened_at: str
    """ISO-8601 UTC timestamp of position open."""

    # ── SL / TP ───────────────────────────────────────────────────────────
    sl: float
    """Stop-loss distance as a fraction (e.g. 0.01 = 1 %)."""

    tp: float
    """Take-profit distance as a fraction (e.g. 0.12 = 12 %)."""

    entry_stop_mult: float
    """SL fraction used at entry — same as ``sl`` for vol_straddle."""

    server_sl_order_id: str | None
    """Binance algo order ID for the server-side stop-loss.  None = not placed."""

    server_tp_order_id: str | None
    """Binance algo order ID for the server-side take-profit.  None = not placed."""

    # ── Trailing stop (R43+) ──────────────────────────────────────────────
    trail_active: bool
    """True once the trailing stop has been activated (peak_gain ≥ threshold)."""

    peak_gain: float
    """Maximum intrabar PnL fraction observed since open."""

    trail_floor: float | None
    """Current trailing stop floor as fraction from entry.  None = not set."""

    # ── Bracket repair ────────────────────────────────────────────────────
    _repair_fail_count: int
    """Consecutive SL bracket-repair failures.  ≥ 3 triggers auto-close."""

    # ── Strategy metadata ─────────────────────────────────────────────────
    strategy_type: str
    """``'vr_directional'``, ``'vr_straddle'``, or ``'ma_cluster'``."""

    tf: str
    """Signal timeframe label: ``'15m'``, ``'1h'``, ``'4h'``."""

    is_straddle: bool
    """True if this position is a leg of a vol_straddle pair (both LONG and SHORT legs are True)."""

    position_side: str | None
    """Hedge mode position side (``'LONG'`` / ``'SHORT'``).  None = one-way."""

    # ── PBT / partial close (rubberband) ─────────────────────────────────
    partial_closed: bool
    """True if a partial close has been executed (PBT flow)."""

    best_pnl: float
    """Best PnL fraction observed (for MFE tracking)."""

    worst_pnl: float
    """Worst PnL fraction observed (for MAE tracking)."""

    highest_since_tp1: float
    """Highest price since partial close for rubberband trail."""

    confidence: int
    """Signal confidence score (0–4, used for CW-Tier sizing)."""

    # ── Balance tracking (Incident #4 / balance_before fix) ──────────────
    balance_before: float | None
    """Account balance at the time this position was opened.

    Captured immediately before ``_place_order()`` and stored in the
    position dict so that the closed ``TradeRecord`` can correctly report
    ``balance_before`` (open-time) vs ``balance_after`` (close-time).

    ``None`` for positions restored from a state file written before
    this field was introduced (fallback: close-time balance used).
    """


# ── TypedDict: closed trade record ───────────────────────────────────────────


class TradeRecord(TypedDict, total=False):
    """Entry in ``scalper._trade_history`` list."""

    symbol: str
    side: str
    strategy_type: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usd: float
    bars_held: int
    exit_reason: str
    entry_time: str
    exit_time: str
    tf: str

    # ── Balance P&L attribution ────────────────────────────────────────────
    balance_before: float
    """Account balance at position open time (captured by positions.py)."""

    balance_after: float
    """Account balance at position close time."""
