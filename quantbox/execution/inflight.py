"""In-flight order tracking for crash recovery.

Records order intent BEFORE sending to exchange, confirms AFTER.
On restart, pending intents indicate orders that may have filled
during a crash — the reconciliation layer can adopt rather than
force-close these positions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("InFlight")


class OrderIntent(str, Enum):
    """What the order is trying to accomplish."""

    ENTRY = "entry"
    EXIT = "exit"
    BRACKET_SL = "bracket_sl"
    BRACKET_TP = "bracket_tp"


@dataclass
class PendingOrder:
    """An order that has been sent to the exchange but not yet confirmed."""

    intent: OrderIntent
    symbol: str
    side: str  # BUY or SELL
    position_side: str | None  # LONG, SHORT, or None
    quantity: float
    order_type: str  # MARKET, STOP_MARKET, etc.
    created_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "symbol": self.symbol,
            "side": self.side,
            "position_side": self.position_side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "created_at": self.created_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PendingOrder:
        return cls(
            intent=OrderIntent(d["intent"]),
            symbol=d["symbol"],
            side=d["side"],
            position_side=d.get("position_side"),
            quantity=float(d["quantity"]),
            order_type=d.get("order_type", "MARKET"),
            created_at=float(d.get("created_at", 0)),
            extra=d.get("extra", {}),
        )


class InFlightTracker:
    """Tracks orders between API submission and state persistence.

    Usage::

        tracker = InFlightTracker(path)

        # Before placing order:
        order_key = tracker.record_pending(PendingOrder(...))

        # After order confirmed:
        tracker.confirm(order_key)

        # On restart:
        stale = tracker.get_stale_orders(max_age_s=300)
        for key, order in stale.items():
            # Check exchange, adopt or close
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._pending: dict[str, PendingOrder] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    def record_pending(self, order: PendingOrder) -> str:
        """Record an order as pending BEFORE sending to exchange.

        Returns:
            Unique key for this pending order.
        """
        key = f"{order.symbol}:{order.position_side or 'NONE'}:{order.intent.value}:{int(order.created_at * 1000)}"
        self._pending[key] = order
        self._save()
        logger.debug(f"Pending order recorded: {key}")
        return key

    def confirm(self, key: str) -> PendingOrder | None:
        """Mark an order as confirmed (filled or rejected).

        Removes from pending list and persists.

        Returns:
            The removed PendingOrder, or None if key not found.
        """
        order = self._pending.pop(key, None)
        if order:
            self._save()
            logger.debug(f"Order confirmed: {key}")
        return order

    def confirm_by_symbol(
        self,
        symbol: str,
        intent: OrderIntent,
        position_side: str | None = None,
    ) -> list[PendingOrder]:
        """Confirm all pending orders matching symbol + intent + position_side.

        Useful when the exact key isn't available (e.g., bracket orders).

        Returns:
            List of confirmed (removed) orders.
        """
        confirmed: list[PendingOrder] = []
        keys_to_remove: list[str] = []
        for key, order in self._pending.items():
            if (
                order.symbol == symbol
                and order.intent == intent
                and (position_side is None or order.position_side == position_side)
            ):
                keys_to_remove.append(key)
                confirmed.append(order)
        for key in keys_to_remove:
            del self._pending[key]
        if confirmed:
            self._save()
        return confirmed

    def get_pending(self) -> dict[str, PendingOrder]:
        """Get all currently pending orders."""
        return dict(self._pending)

    def get_stale_orders(self, max_age_s: float = 300) -> dict[str, PendingOrder]:
        """Get pending orders older than max_age_s.

        These likely represent orders from a previous session
        that weren't confirmed due to a crash.

        Args:
            max_age_s: Orders older than this are considered stale.

        Returns:
            Dict of stale order key -> PendingOrder.
        """
        cutoff = time.time() - max_age_s
        return {k: v for k, v in self._pending.items() if v.created_at < cutoff}

    def clear_all(self) -> int:
        """Clear all pending orders. Returns count cleared."""
        count = len(self._pending)
        self._pending.clear()
        self._save()
        return count

    def _save(self) -> None:
        """Persist pending orders to disk."""
        if not self._persist_path:
            return
        data = {k: v.to_dict() for k, v in self._pending.items()}
        tmp = self._persist_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._persist_path)
        except OSError as e:
            logger.error(f"Failed to save inflight state: {e}")

    def _load(self) -> None:
        """Load pending orders from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            self._pending = {k: PendingOrder.from_dict(v) for k, v in data.items()}
            if self._pending:
                logger.info(f"Loaded {len(self._pending)} pending orders from disk")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load inflight state: {e}")
            self._pending = {}
