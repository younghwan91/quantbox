"""T6: Crash Recovery Integration Tests.

Tests crash recovery paths across InFlightTracker, RateLimiter,
Heartbeat, DiscordNotifier, VolStraddleConfig validators, and
scalper state persistence.

14 integration tests covering:
- InFlight order persistence across simulated crashes
- RateLimiter circuit breaker and exponential backoff
- Heartbeat stall detection and recovery
- Corrupt file recovery
- Discord alert graceful degradation (disabled mode)
- Config validator catches invalid trailing stop params
- Full crash recovery lifecycle
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from quantbox.core.heartbeat import Heartbeat
from quantbox.core.rate_limiter import RateLimiter
from quantbox.execution.inflight import InFlightTracker, OrderIntent, PendingOrder
from quantbox.notifications.discord import DiscordNotifier


# ─────────────────────────── Helpers ───────────────────────────


def _make_entry_order(
    symbol: str = "BTCUSDT",
    side: str = "BUY",
    position_side: str | None = "LONG",
    quantity: float = 0.01,
    created_at: float | None = None,
) -> PendingOrder:
    """Create a PendingOrder for testing."""
    return PendingOrder(
        intent=OrderIntent.ENTRY,
        symbol=symbol,
        side=side,
        position_side=position_side,
        quantity=quantity,
        order_type="MARKET",
        created_at=created_at or time.time(),
    )


# ─────────────── InFlightTracker Crash Recovery ────────────────


class TestInFlightCrashRecovery:
    """Tests 1-3: InFlightTracker persistence across simulated crashes."""

    def test_inflight_order_survives_crash(self, tmp_path: Path) -> None:
        """Record a pending entry → crash → new tracker loads the stale order."""
        persist = tmp_path / "inflight.json"
        t_crash = 1_700_000_000.0

        # Session 1: record an order, then "crash" (no confirm)
        with patch("time.time", return_value=t_crash):
            tracker1 = InFlightTracker(persist_path=persist)
            order = _make_entry_order(symbol="ETHUSDT", created_at=t_crash)
            key = tracker1.record_pending(order)

        assert persist.exists(), "inflight file must be persisted"

        # Session 2: new tracker loads from same file
        with patch("time.time", return_value=t_crash + 600):
            tracker2 = InFlightTracker(persist_path=persist)
            stale = tracker2.get_stale_orders(max_age_s=300)

        assert key in stale, "unconfirmed order should appear as stale"
        assert stale[key].symbol == "ETHUSDT"
        assert stale[key].intent == OrderIntent.ENTRY

    def test_inflight_order_confirm_clears(self, tmp_path: Path) -> None:
        """Record → confirm → verify cleared from disk."""
        persist = tmp_path / "inflight.json"
        tracker = InFlightTracker(persist_path=persist)
        order = _make_entry_order(symbol="SOLUSDT")
        key = tracker.record_pending(order)

        removed = tracker.confirm(key)
        assert removed is not None
        assert removed.symbol == "SOLUSDT"

        # Verify disk is clean
        data = json.loads(persist.read_text())
        assert key not in data
        assert len(tracker.get_pending()) == 0

    def test_inflight_multiple_symbols_independent(self, tmp_path: Path) -> None:
        """3 symbols recorded → confirm 1 → other 2 still pending."""
        persist = tmp_path / "inflight.json"
        tracker = InFlightTracker(persist_path=persist)

        keys = {}
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            order = _make_entry_order(symbol=sym)
            keys[sym] = tracker.record_pending(order)

        # Confirm only ETH
        tracker.confirm(keys["ETHUSDT"])

        pending = tracker.get_pending()
        assert keys["BTCUSDT"] in pending
        assert keys["SOLUSDT"] in pending
        assert keys["ETHUSDT"] not in pending
        assert len(pending) == 2


# ─────────────── RateLimiter Circuit Breaker ────────────────


class TestRateLimiterCircuitBreaker:
    """Tests 4-6: Circuit breaker and exponential backoff."""

    def test_circuit_breaker_blocks_after_failures(self) -> None:
        """5 consecutive failures → circuit opens."""
        rl = RateLimiter(circuit_break_after=5)

        for i in range(4):
            rl.record_failure()
            assert not rl._circuit_open, f"circuit should NOT open after {i + 1} failures"

        # 5th failure trips the breaker
        backoff = rl.record_failure()
        assert rl._circuit_open is True
        assert backoff > 0

        # is_circuit_open property also reflects this
        # (need monotonic to be before _circuit_open_until)
        with patch("time.monotonic", return_value=time.monotonic()):
            assert rl.is_circuit_open is True

    def test_circuit_resets_on_success(self) -> None:
        """Open circuit → record_success → circuit closed."""
        rl = RateLimiter(circuit_break_after=5)

        for _ in range(5):
            rl.record_failure()
        assert rl._circuit_open is True

        rl.record_success()
        assert rl._consecutive_failures == 0
        assert rl._last_backoff_s == 0.0

    def test_backoff_exponential_sequence(self) -> None:
        """Verify backoff: 2, 4, 8, 16, 32, 60, 60 (capped at max_backoff_s)."""
        rl = RateLimiter(
            base_backoff_s=2.0,
            max_backoff_s=60.0,
            circuit_break_after=100,  # high so circuit doesn't interfere
        )
        expected = [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        actual = [rl.record_failure() for _ in expected]
        assert actual == expected


# ─────────────── Heartbeat Stall Detection ────────────────


class TestHeartbeatStall:
    """Tests 7-8: Stall detection with time control."""

    def test_stall_detection(self) -> None:
        """Beat → advance time past threshold → stalled."""
        t0 = 1000.0
        with patch("time.monotonic", return_value=t0):
            hb = Heartbeat(stall_threshold_s=300.0)
            hb.beat()
        assert hb.total_beats == 1

        # 301s later → stalled
        with patch("time.monotonic", return_value=t0 + 301):
            assert hb.is_stalled() is True
            assert hb.seconds_since_last_beat() > 300

    def test_beat_clears_stall(self) -> None:
        """Stalled → beat() → no longer stalled."""
        t0 = 1000.0
        with patch("time.monotonic", return_value=t0):
            hb = Heartbeat(stall_threshold_s=60.0)
            hb.beat()

        # Advance into stall
        with patch("time.monotonic", return_value=t0 + 120):
            assert hb.is_stalled() is True

        # Beat clears it
        t1 = t0 + 121
        with patch("time.monotonic", return_value=t1):
            hb.beat()

        with patch("time.monotonic", return_value=t1 + 10):
            assert hb.is_stalled() is False
            assert hb.total_beats == 2


# ───────────────── State + InFlight Persistence ─────────────────


class TestStatePersistence:
    """Tests 9-10: State file save/load and corruption recovery."""

    def test_state_save_load_with_inflight(self, tmp_path: Path) -> None:
        """Save InFlightTracker to disk, load in new instance, verify pending."""
        persist = tmp_path / "inflight.json"

        # Write 2 pending orders
        tracker1 = InFlightTracker(persist_path=persist)
        o1 = _make_entry_order(symbol="BTCUSDT", created_at=1_700_000_100.0)
        o2 = _make_entry_order(
            symbol="ETHUSDT",
            side="SELL",
            position_side="SHORT",
            created_at=1_700_000_200.0,
        )
        k1 = tracker1.record_pending(o1)
        k2 = tracker1.record_pending(o2)

        # Fresh load
        tracker2 = InFlightTracker(persist_path=persist)
        pending = tracker2.get_pending()

        assert k1 in pending
        assert k2 in pending
        assert pending[k1].symbol == "BTCUSDT"
        assert pending[k2].symbol == "ETHUSDT"
        assert pending[k2].side == "SELL"
        assert pending[k2].position_side == "SHORT"

    def test_corrupt_inflight_file_recovery(self, tmp_path: Path) -> None:
        """Garbage data in inflight file → empty state, no crash."""
        persist = tmp_path / "inflight.json"
        persist.write_text("{{{{NOT VALID JSON!!!!@@@")

        tracker = InFlightTracker(persist_path=persist)
        assert len(tracker.get_pending()) == 0

        # Tracker is still functional after recovery
        order = _make_entry_order(symbol="AVAXUSDT")
        key = tracker.record_pending(order)
        assert key in tracker.get_pending()


# ─────────────── Discord Alert Integration ────────────────


class TestDiscordAlertDisabled:
    """Tests 11-12: Discord notifier graceful degradation."""

    def test_critical_alert_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disabled notifier → notify_critical returns False, no error."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        notifier = DiscordNotifier(webhook_url=None)
        assert notifier.enabled is False

        result = notifier.notify_critical(
            title="Test Critical",
            message="System on fire",
            alert_key="test_crit",
        )
        assert result is False

    def test_circuit_breaker_alert_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """notify_circuit_breaker(open/close) handles gracefully when disabled."""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        notifier = DiscordNotifier(webhook_url=None)

        result_open = notifier.notify_circuit_breaker(is_open=True, failures=5, backoff_s=32.0)
        result_close = notifier.notify_circuit_breaker(is_open=False, failures=0, backoff_s=0.0)
        assert result_open is False
        assert result_close is False


# ─────────────── Config Validator ────────────────


class TestConfigValidator:
    """Test 13: Config validation catches invalid trailing stop params."""

    def test_invalid_trail_distance_gt_activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """d_trail_distance > d_trail_activate → ValueError."""
        # Clear any existing VS_ env vars that could interfere
        import os

        for key in list(os.environ):
            if key.startswith("VS_"):
                monkeypatch.delenv(key, raising=False)

        # Set invalid: distance=0.05 > activate=0.01
        monkeypatch.setenv("VS_D_TRAIL_DISTANCE", "0.05")
        monkeypatch.setenv("VS_D_TRAIL_ACTIVATE", "0.01")

        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        with pytest.raises(ValueError, match="d_trail_distance must be less than d_trail_activate"):
            VolStraddleConfig()


# ─────────────── Full Lifecycle ────────────────


class TestFullCrashRecoveryLifecycle:
    """Test 14: End-to-end crash recovery simulation."""

    def test_full_crash_recovery_lifecycle(self, tmp_path: Path) -> None:
        """Full lifecycle: record → crash → reload → find stale → confirm_by_symbol → clean."""
        persist = tmp_path / "inflight.json"
        t0 = 1_700_000_000.0

        # ── Phase 1: Pre-crash session ──
        with patch("time.time", return_value=t0):
            tracker = InFlightTracker(persist_path=persist)
            entry = PendingOrder(
                intent=OrderIntent.ENTRY,
                symbol="SUIUSDT",
                side="BUY",
                position_side="LONG",
                quantity=100.0,
                order_type="MARKET",
                created_at=t0,
            )
            key = tracker.record_pending(entry)

            # Also record a bracket SL
            bracket = PendingOrder(
                intent=OrderIntent.BRACKET_SL,
                symbol="SUIUSDT",
                side="SELL",
                position_side="LONG",
                quantity=100.0,
                order_type="STOP_MARKET",
                created_at=t0,
            )
            tracker.record_pending(bracket)

        assert len(tracker.get_pending()) == 2

        # ── Phase 2: Crash (tracker destroyed, file persists) ──
        del tracker

        # ── Phase 3: Recovery session ──
        with patch("time.time", return_value=t0 + 600):
            tracker2 = InFlightTracker(persist_path=persist)

            # All 2 orders are pending
            assert len(tracker2.get_pending()) == 2

            # Both are stale (600s > 300s default)
            stale = tracker2.get_stale_orders(max_age_s=300)
            assert len(stale) == 2

            # Recovery: confirm all SUIUSDT ENTRY orders by symbol
            confirmed_entries = tracker2.confirm_by_symbol(
                symbol="SUIUSDT",
                intent=OrderIntent.ENTRY,
                position_side="LONG",
            )
            assert len(confirmed_entries) == 1
            assert confirmed_entries[0].symbol == "SUIUSDT"
            assert confirmed_entries[0].intent == OrderIntent.ENTRY

            # Bracket still pending
            remaining = tracker2.get_pending()
            assert len(remaining) == 1
            remaining_order = list(remaining.values())[0]
            assert remaining_order.intent == OrderIntent.BRACKET_SL

            # Confirm bracket too
            confirmed_brackets = tracker2.confirm_by_symbol(
                symbol="SUIUSDT",
                intent=OrderIntent.BRACKET_SL,
                position_side="LONG",
            )
            assert len(confirmed_brackets) == 1

            # ── Phase 4: Clean state ──
            assert len(tracker2.get_pending()) == 0

            # Disk is clean
            data = json.loads(persist.read_text())
            assert len(data) == 0
