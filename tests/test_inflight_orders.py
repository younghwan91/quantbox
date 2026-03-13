"""Tests for in-flight order tracking (crash recovery)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quantbox.execution.inflight import InFlightTracker, OrderIntent, PendingOrder


# ── helpers ──────────────────────────────────────────────────────────────


def _make_order(
    symbol: str = "BTCUSDT",
    intent: OrderIntent = OrderIntent.ENTRY,
    side: str = "BUY",
    position_side: str | None = "LONG",
    quantity: float = 0.01,
    order_type: str = "MARKET",
    created_at: float | None = None,
    extra: dict | None = None,
) -> PendingOrder:
    kw: dict = dict(
        intent=intent,
        symbol=symbol,
        side=side,
        position_side=position_side,
        quantity=quantity,
        order_type=order_type,
    )
    if created_at is not None:
        kw["created_at"] = created_at
    if extra is not None:
        kw["extra"] = extra
    return PendingOrder(**kw)


# ── 1. record_pending returns key ────────────────────────────────────────


def test_record_pending_returns_key() -> None:
    tracker = InFlightTracker()
    order = _make_order(symbol="ETHUSDT", intent=OrderIntent.ENTRY, position_side="SHORT")
    key = tracker.record_pending(order)

    assert "ETHUSDT" in key
    assert "entry" in key
    assert "SHORT" in key


# ── 2. confirm removes pending ──────────────────────────────────────────


def test_confirm_removes_pending() -> None:
    tracker = InFlightTracker()
    key = tracker.record_pending(_make_order())

    removed = tracker.confirm(key)
    assert removed is not None
    assert tracker.get_pending() == {}


# ── 3. confirm unknown key returns None ──────────────────────────────────


def test_confirm_unknown_key_returns_none() -> None:
    tracker = InFlightTracker()
    assert tracker.confirm("nonexistent:key") is None


# ── 4. confirm_by_symbol matches ─────────────────────────────────────────


def test_confirm_by_symbol_matches() -> None:
    tracker = InFlightTracker()
    tracker.record_pending(_make_order(symbol="SUIUSDT", intent=OrderIntent.BRACKET_SL, position_side="LONG"))
    tracker.record_pending(
        _make_order(symbol="SUIUSDT", intent=OrderIntent.BRACKET_SL, position_side="LONG", created_at=1.0)
    )

    confirmed = tracker.confirm_by_symbol("SUIUSDT", OrderIntent.BRACKET_SL, "LONG")
    assert len(confirmed) == 2
    assert tracker.get_pending() == {}


# ── 5. confirm_by_symbol no match ────────────────────────────────────────


def test_confirm_by_symbol_no_match() -> None:
    tracker = InFlightTracker()
    tracker.record_pending(_make_order(symbol="BTCUSDT", intent=OrderIntent.ENTRY))

    confirmed = tracker.confirm_by_symbol("ETHUSDT", OrderIntent.ENTRY)
    assert confirmed == []
    assert len(tracker.get_pending()) == 1


# ── 6. get_pending returns copy ──────────────────────────────────────────


def test_get_pending_returns_copy() -> None:
    tracker = InFlightTracker()
    tracker.record_pending(_make_order())

    snapshot = tracker.get_pending()
    snapshot.clear()

    assert len(tracker.get_pending()) == 1


# ── 7. get_stale_orders age filter ───────────────────────────────────────


def test_get_stale_orders_age_filter() -> None:
    tracker = InFlightTracker()

    with patch("quantbox.execution.inflight.time.time", return_value=1000.0):
        tracker.record_pending(_make_order(symbol="OLD", created_at=500.0))
        tracker.record_pending(_make_order(symbol="NEW", created_at=900.0))

    with patch("quantbox.execution.inflight.time.time", return_value=1000.0):
        stale = tracker.get_stale_orders(max_age_s=300)

    assert len(stale) == 1
    syms = [o.symbol for o in stale.values()]
    assert "OLD" in syms
    assert "NEW" not in syms


# ── 8. get_stale_orders none when fresh ──────────────────────────────────


def test_get_stale_orders_none_when_fresh() -> None:
    tracker = InFlightTracker()

    with patch("quantbox.execution.inflight.time.time", return_value=1000.0):
        tracker.record_pending(_make_order(created_at=999.0))

    with patch("quantbox.execution.inflight.time.time", return_value=1000.0):
        stale = tracker.get_stale_orders(max_age_s=300)

    assert stale == {}


# ── 9. clear_all ─────────────────────────────────────────────────────────


def test_clear_all() -> None:
    tracker = InFlightTracker()
    tracker.record_pending(_make_order(symbol="A"))
    tracker.record_pending(_make_order(symbol="B", created_at=1.0))
    tracker.record_pending(_make_order(symbol="C", created_at=2.0))

    count = tracker.clear_all()
    assert count == 3
    assert tracker.get_pending() == {}


# ── 10. persistence save + load ──────────────────────────────────────────


def test_persistence_save_load(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "inflight.json"  # type: ignore[operator]

    tracker1 = InFlightTracker(persist_path=path)
    tracker1.record_pending(_make_order(symbol="XRPUSDT", quantity=5.5))

    tracker2 = InFlightTracker(persist_path=path)
    pending = tracker2.get_pending()

    assert len(pending) == 1
    order = list(pending.values())[0]
    assert order.symbol == "XRPUSDT"
    assert order.quantity == 5.5


# ── 11. persistence corrupt file ─────────────────────────────────────────


def test_persistence_corrupt_file(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "inflight.json"  # type: ignore[operator]
    path.write_text("{{{not valid json")  # type: ignore[union-attr]

    tracker = InFlightTracker(persist_path=path)
    assert tracker.get_pending() == {}


# ── 12. PendingOrder round-trip serialization ────────────────────────────


def test_pending_order_to_dict_from_dict() -> None:
    original = _make_order(
        symbol="SOLUSDT",
        intent=OrderIntent.EXIT,
        side="SELL",
        position_side="LONG",
        quantity=1.23,
        order_type="STOP_MARKET",
        created_at=1700000000.123,
        extra={"reason": "trail_hit"},
    )

    d = original.to_dict()
    restored = PendingOrder.from_dict(d)

    assert restored.intent == original.intent
    assert restored.symbol == original.symbol
    assert restored.side == original.side
    assert restored.position_side == original.position_side
    assert restored.quantity == original.quantity
    assert restored.order_type == original.order_type
    assert restored.created_at == original.created_at
    assert restored.extra == original.extra


# ── 13. full lifecycle: record → confirm → empty ─────────────────────────


def test_record_confirm_lifecycle() -> None:
    tracker = InFlightTracker()
    order = _make_order()

    key = tracker.record_pending(order)
    assert len(tracker.get_pending()) == 1

    removed = tracker.confirm(key)
    assert removed is not None
    assert removed.symbol == order.symbol
    assert tracker.get_pending() == {}


# ── 14. multiple symbols tracked independently ──────────────────────────


def test_multiple_symbols_independent() -> None:
    tracker = InFlightTracker()
    k1 = tracker.record_pending(_make_order(symbol="BTCUSDT"))
    k2 = tracker.record_pending(_make_order(symbol="ETHUSDT"))
    k3 = tracker.record_pending(_make_order(symbol="SOLUSDT"))

    assert len(tracker.get_pending()) == 3

    tracker.confirm(k2)
    remaining = tracker.get_pending()
    assert len(remaining) == 2
    syms = {o.symbol for o in remaining.values()}
    assert syms == {"BTCUSDT", "SOLUSDT"}


# ── 15. OrderIntent enum values ──────────────────────────────────────────


def test_order_intent_enum() -> None:
    assert OrderIntent.ENTRY.value == "entry"
    assert OrderIntent.EXIT.value == "exit"
    assert OrderIntent.BRACKET_SL.value == "bracket_sl"
    assert OrderIntent.BRACKET_TP.value == "bracket_tp"

    assert OrderIntent("entry") == OrderIntent.ENTRY
    assert OrderIntent("bracket_tp") == OrderIntent.BRACKET_TP

    assert len(OrderIntent) == 4


# ── 16. confirm_by_symbol with position_side=None matches all ────────────


def test_confirm_by_symbol_position_side_none_matches_all() -> None:
    tracker = InFlightTracker()
    tracker.record_pending(_make_order(symbol="AVAXUSDT", intent=OrderIntent.EXIT, position_side="LONG"))
    tracker.record_pending(
        _make_order(symbol="AVAXUSDT", intent=OrderIntent.EXIT, position_side="SHORT", created_at=1.0)
    )

    confirmed = tracker.confirm_by_symbol("AVAXUSDT", OrderIntent.EXIT, position_side=None)
    assert len(confirmed) == 2
    assert tracker.get_pending() == {}


# ── 17. persistence confirm removes from disk ───────────────────────────


def test_persistence_confirm_removes_from_disk(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "inflight.json"  # type: ignore[operator]
    tracker = InFlightTracker(persist_path=path)
    key = tracker.record_pending(_make_order())

    tracker.confirm(key)

    tracker2 = InFlightTracker(persist_path=path)
    assert tracker2.get_pending() == {}
