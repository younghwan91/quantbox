"""Strategy protocol — formal interface for trading strategies.

Both VolStraddleStrategy and RubberBandStrategy implement this protocol
via structural subtyping (duck typing).  This Protocol exists for:

1. Type checking (mypy will verify scalper.py uses only protocol methods)
2. Documentation (new strategies know exactly what to implement)
3. IDE support (autocomplete on self.strategy)

No runtime cost — Protocol is purely a type-checking construct.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TradingStrategy(Protocol):
    """Protocol for trading strategies used by RubberBandScalper.

    Any class implementing these methods/properties can be used as a
    strategy.  No inheritance required — structural subtyping handles it.

    Properties:
        n_active: Number of currently open positions.
        active_positions: Snapshot dict of open positions keyed by
            symbol (one-way) or ``symbol:SIDE`` (hedge mode).

    Signal flow per cycle (called by scalper):
        1. ``update_market_data()`` — feed 15m OHLCV
        2. ``update_htf_data()``    — feed 4h close (+ optional high/low)
        3. ``on_bar()``             — generate signal (-1/0/1/2)
        4. ``update_position()``    — check exit conditions
    """

    # ── properties ──────────────────────────────────────────────────

    @property
    def n_active(self) -> int:
        """Number of active positions."""
        ...

    @property
    def active_positions(self) -> dict[str, dict]:
        """Return copy of all active positions."""
        ...

    # ── data feed ───────────────────────────────────────────────────

    def update_market_data(
        self,
        symbol: str,
        volume: np.ndarray,
        taker_buy: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> None:
        """Store 15m OHLCV data before ``on_bar()``."""
        ...

    def update_htf_data(
        self,
        symbol: str,
        close_4h: np.ndarray,
        high_4h: np.ndarray | None = None,
        low_4h: np.ndarray | None = None,
        volume_4h: np.ndarray | None = None,
    ) -> None:
        """Store 4h candle data for multi-TF analysis."""
        ...

    # ── signal generation ───────────────────────────────────────────

    def on_bar(
        self,
        symbol: str,
        close_prices: np.ndarray,
        btc_close: np.ndarray | None = None,
    ) -> int:
        """Generate a trading signal for *symbol*.

        Returns:
            -1 (short), 0 (no signal), 1 (long), or 2 (straddle).
        """
        ...

    def diagnose_bar(
        self,
        symbol: str,
        close_prices: np.ndarray,
        btc_close: np.ndarray | None = None,
    ) -> dict:
        """Return diagnostic dict for current bar (debug/logging)."""
        ...

    # ── position management ─────────────────────────────────────────

    def has_position(self, symbol: str) -> bool:
        """Check whether a position is open for *symbol*."""
        ...

    def open_position(
        self,
        symbol: str,
        signal: int,
        entry_price: float,
        notional: float,
        leverage: float,
        move_size: float,
        entry_stop_mult: float | None = None,
        server_sl_order_id: str | None = None,
        server_tp_order_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a new position opened by the scalper."""
        ...

    def close_position(self, symbol: str) -> dict | None:
        """Remove and return position info, or ``None`` if not found."""
        ...

    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_atr_pct: float | None = None,
        current_high: float | None = None,
        current_low: float | None = None,
    ) -> str | None:
        """Evaluate exit conditions for an open position.

        Returns:
            Exit reason string (e.g. ``"sl"``, ``"trailing_stop"``)
            or ``None`` if the position should remain open.
        """
        ...

    # ── query helpers ───────────────────────────────────────────────

    def get_effective_stop_mult(
        self,
        close_prices: np.ndarray,
        symbol: str | None = None,
        move_size: float | None = None,
    ) -> float:
        """Return effective stop-loss multiplier for sizing."""
        ...

    def get_last_confidence(self, symbol: str) -> int:
        """Return last confidence score (0–N) for *symbol*."""
        ...

    def was_mtf_confirmed(self, symbol: str) -> bool:
        """Whether the last signal was multi-timeframe confirmed."""
        ...

    def get_last_atr_pct(self, symbol: str) -> float:
        """Return last ATR as a fraction of price for *symbol*."""
        ...
