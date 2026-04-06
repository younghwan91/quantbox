"""Characterization tests for _cycle() — capture current behavior before refactoring.

These tests serve as the safety net during the _cycle() extraction refactor.
Every test here must pass both before and after the refactor.

The tests use heavy mocking to isolate _cycle() from external I/O:
- download_klines: returns flat numpy arrays
- _save_state: no-op
- _sync_server_orders: no-op
- _reconcile_positions: no-op
- discord.send: captured
- _close_position: captured
- _scan_entries: captured
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock

from tests.conftest import _build_vs_scalper


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

PRICE = 100.0
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def _make_scalper(balance: float = 1000.0) -> object:
    """Build a fully-mocked VolStraddle scalper ready for _cycle() tests."""
    s = _build_vs_scalper(balance=balance)
    # Silence network calls
    s._save_state = MagicMock()
    s._sync_server_orders = MagicMock()
    s._reconcile_positions = MagicMock()
    s._check_divergence = MagicMock()
    s._log_summary = MagicMock()
    s._process_bot_commands = MagicMock()
    s._scan_entries = MagicMock()
    s._close_position = MagicMock()
    s._partial_close_position = MagicMock()
    s.discord = MagicMock()

    # Flat kline data (close-only and full OHLCV)
    closes = np.full(210, PRICE)
    highs = closes + 1.0
    lows = closes - 1.0
    vols = np.full(210, 1000.0)
    tbs = np.full(210, 500.0)

    def _dl(symbol, n, interval="15m", full=False):
        if full:
            return closes[:n], highs[:n], lows[:n], vols[:n], tbs[:n]
        return closes[:n]

    s.download_klines = MagicMock(side_effect=_dl)
    return s


def _inject_pos(scalper, key: str = "TESTUSDT:LONG", **overrides) -> dict:
    pos = {
        "symbol": key.split(":")[0] if ":" in key else key,
        "signal": 1 if "LONG" in key else -1,
        "entry_price": PRICE,
        "notional": 3000.0,
        "bars_held": 5,
        "peak_gain": 0.02,
        "best_pnl": 0.02,
        "worst_pnl": -0.005,
        "server_sl_order_id": "sl_111",
        "server_tp_order_id": "tp_222",
        "sl": 0.01,
        "tp": 0.20,
        "tf": "15m",
        "position_side": "LONG" if "LONG" in key else "SHORT",
        "trail_active": False,
        "opened_at": "2026-01-01T00:00:00+00:00",
        "strategy_type": "vr_directional",
        "max_hold": 192,
    }
    pos.update(overrides)
    scalper.strategy._positions[key] = pos
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Early-return paths
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleEarlyReturns:
    """_cycle() must return early in specific conditions without touching entries."""

    def test_stop_requested_returns_before_header(self):
        """_stop_requested=True → return immediately after bot commands, no entries."""
        s = _make_scalper()
        s._stop_requested = True
        s._cycle(SYMBOLS)
        s._scan_entries.assert_not_called()
        s._save_state.assert_not_called()

    def test_drawdown_halt_with_positions_closes_and_returns(self):
        """DD halt active + positions → close all → save state → return."""
        s = _make_scalper()
        s._drawdown_halt = True
        s._high_water_mark = 1000.0
        s.get_balance = MagicMock(return_value=500.0)  # 50% DD → still halted
        _inject_pos(s, "AAAUSDT:LONG")
        _inject_pos(s, "BBBUSDT:SHORT")

        s._cycle(SYMBOLS)

        # Both positions must be closed
        assert s._close_position.call_count == 2
        closed_keys = {c.args[0] for c in s._close_position.call_args_list}
        assert "AAAUSDT:LONG" in closed_keys
        assert "BBBUSDT:SHORT" in closed_keys

        # No new entries attempted
        s._scan_entries.assert_not_called()
        # State saved before returning
        s._save_state.assert_called()

    def test_drawdown_halt_no_positions_still_halted_returns(self):
        """DD halt active + no positions + DD still high → save + return."""
        s = _make_scalper()
        s._drawdown_halt = True
        s._high_water_mark = 1000.0
        s.get_balance = MagicMock(return_value=600.0)  # 40% DD > threshold
        s._cfg.max_drawdown_pct = 0.30

        s._cycle(SYMBOLS)

        s._scan_entries.assert_not_called()
        s._save_state.assert_called()

    def test_drawdown_halt_recovers_when_dd_heals(self):
        """DD halt + no positions + DD below threshold → halt cleared, cycle continues."""
        s = _make_scalper()
        s._drawdown_halt = True
        s._high_water_mark = 1000.0
        s.get_balance = MagicMock(return_value=950.0)  # 5% DD < 30% threshold
        s._cfg.max_drawdown_pct = 0.30

        s._cycle(SYMBOLS)

        # Halt must be cleared
        assert s._drawdown_halt is False
        # Discord notification sent (recovery)
        s.discord.send.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# Position management flow
# ─────────────────────────────────────────────────────────────────────────────


class TestCyclePositionManagement:
    """_cycle() must update and exit positions correctly."""

    def test_position_with_exit_signal_calls_close(self):
        """update_position returns exit reason → _close_position called."""
        s = _make_scalper()
        _inject_pos(s, "TESTUSDT:LONG")

        # Simulate exit signal from strategy
        s.strategy.update_position = MagicMock(return_value="sl_hit")

        s._cycle(SYMBOLS)

        s._close_position.assert_called_once()
        args = s._close_position.call_args.args
        assert args[0] == "TESTUSDT:LONG"
        assert args[1] == "sl_hit"

    def test_position_without_exit_signal_not_closed(self):
        """update_position returns None → _close_position NOT called."""
        s = _make_scalper()
        _inject_pos(s, "TESTUSDT:LONG")
        s.strategy.update_position = MagicMock(return_value=None)

        s._cycle(SYMBOLS)

        s._close_position.assert_not_called()

    def test_partial_close_calls_partial_method(self):
        """exit_reason='partial_close' → _partial_close_position, not _close_position."""
        s = _make_scalper()
        _inject_pos(s, "TESTUSDT:LONG")
        s.strategy.update_position = MagicMock(return_value="partial_close")

        s._cycle(SYMBOLS)

        s._partial_close_position.assert_called_once()
        s._close_position.assert_not_called()

    def test_trail_active_position_calls_ratchet(self):
        """trail_active=True + paper_mode=False + trail_server_native=False → _ratchet_trail_sl called."""
        s = _make_scalper()
        s.paper_mode = False
        # Ensure server native is off so ratchet fires
        s.vs_config.trail_server_native = False
        _inject_pos(s, "TESTUSDT:LONG", trail_active=True)
        s.strategy.update_position = MagicMock(return_value=None)
        s._ratchet_trail_sl = MagicMock()

        s._cycle(SYMBOLS)

        s._ratchet_trail_sl.assert_called_once()

    def test_trail_ratchet_skipped_in_paper_mode(self):
        """trail_active=True + paper_mode=True → _ratchet_trail_sl NOT called."""
        s = _make_scalper()
        s.paper_mode = True
        _inject_pos(s, "TESTUSDT:LONG", trail_active=True)
        s.strategy.update_position = MagicMock(return_value=None)
        s._ratchet_trail_sl = MagicMock()

        s._cycle(SYMBOLS)

        s._ratchet_trail_sl.assert_not_called()

    def test_multiple_positions_all_updated(self):
        """Multiple positions: all get update_position called."""
        s = _make_scalper()
        _inject_pos(s, "AAAUSDT:LONG")
        _inject_pos(s, "BBBUSDT:SHORT")
        s.strategy.update_position = MagicMock(return_value=None)

        s._cycle(SYMBOLS)

        assert s.strategy.update_position.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Entry scanning gates
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleEntryGates:
    """_cycle() must respect all entry gate conditions."""

    def test_paused_skips_entries(self):
        """_paused=True → _scan_entries never called."""
        s = _make_scalper()
        s._paused = True

        s._cycle(SYMBOLS)

        s._scan_entries.assert_not_called()

    def test_risk_blocked_skips_entries(self):
        """Risk limits breached → _scan_entries never called."""
        s = _make_scalper()
        s._check_risk_limits = MagicMock(return_value=True)

        s._cycle(SYMBOLS)

        s._scan_entries.assert_not_called()

    def test_balance_fetch_failure_skips_entries(self):
        """_balance_fetch_ok=False → _scan_entries not called, Discord alerted."""
        s = _make_scalper()
        s._balance_fetch_ok = False
        s._balance_alert_sent = False

        s._cycle(SYMBOLS)

        s._scan_entries.assert_not_called()
        s.discord.send.assert_called()

    def test_balance_alert_not_resent_when_already_sent(self):
        """Balance failure alert sent only once (flag guard)."""
        s = _make_scalper()
        s._balance_fetch_ok = False
        s._balance_alert_sent = True

        s._cycle(SYMBOLS)

        s.discord.send.assert_not_called()

    def test_daily_loss_blocked_skips_entries(self):
        """_daily_loss_blocked=True → _scan_entries not called."""
        s = _make_scalper()
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        s._daily_loss_date = today_utc  # same day → no reset
        s._daily_loss_blocked = True
        s.vs_config.daily_loss_limit_pct = 0.05

        s._cycle(SYMBOLS)

        s._scan_entries.assert_not_called()

    def test_full_slots_skips_entries(self):
        """All slots occupied → _scan_entries not called."""
        s = _make_scalper()
        # Fill all 10 slots
        for i in range(10):
            _inject_pos(s, f"SYM{i:02d}USDT:LONG")
        s.strategy.update_position = MagicMock(return_value=None)

        s._cycle(SYMBOLS)

        s._scan_entries.assert_not_called()

    def test_available_slots_triggers_scan_entries(self):
        """Slots available + no gates → _scan_entries called."""
        s = _make_scalper()
        # No positions, no blocks

        s._cycle(SYMBOLS)

        s._scan_entries.assert_called_once()
        call_args = s._scan_entries.call_args
        # symbols passed correctly
        assert call_args.args[0] == SYMBOLS


# ─────────────────────────────────────────────────────────────────────────────
# Housekeeping: save, sync, reconcile
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleHousekeeping:
    """_cycle() must always call housekeeping functions in every normal cycle."""

    def test_save_state_always_called(self):
        """Normal cycle → _save_state always called (even when no positions)."""
        s = _make_scalper()
        s._cycle(SYMBOLS)
        s._save_state.assert_called()

    def test_sync_server_orders_called(self):
        """_sync_server_orders called every cycle."""
        s = _make_scalper()
        s._cycle(SYMBOLS)
        s._sync_server_orders.assert_called_once()

    def test_reconcile_called_in_live_mode(self):
        """_reconcile_positions called in live mode (not paper)."""
        s = _make_scalper()
        s.paper_mode = False
        s._cycle(SYMBOLS)
        s._reconcile_positions.assert_called_once()

    def test_reconcile_skipped_in_paper_mode(self):
        """_reconcile_positions skipped in paper mode."""
        s = _make_scalper()
        s.paper_mode = True
        s._cycle(SYMBOLS)
        s._reconcile_positions.assert_not_called()

    def test_cycle_count_increments(self):
        """_cycle_count increments by 1 each call."""
        s = _make_scalper()
        initial = s._cycle_count
        s._cycle(SYMBOLS)
        assert s._cycle_count == initial + 1

    def test_heartbeat_called(self):
        """Heartbeat.beat() called on successful cycle."""
        s = _make_scalper()
        s._heartbeat = MagicMock()
        s._cycle(SYMBOLS)
        s._heartbeat.beat.assert_called_once()

    def test_heartbeat_failure_does_not_crash_cycle(self):
        """Heartbeat error → cycle continues, save_state still called."""
        s = _make_scalper()
        s._heartbeat = MagicMock()
        s._heartbeat.beat.side_effect = RuntimeError("hb down")
        s._cycle(SYMBOLS)
        # save_state still called despite heartbeat failure
        s._save_state.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# Execution order
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleOrder:
    """Critical ordering: sync before positions, positions before entries, save last."""

    def test_sync_before_position_update(self):
        """_sync_server_orders called before update_position."""
        s = _make_scalper()
        _inject_pos(s, "TESTUSDT:LONG")
        call_order = []

        original_sync = s._sync_server_orders
        s._sync_server_orders = MagicMock(side_effect=lambda: call_order.append("sync"))
        s.strategy.update_position = MagicMock(side_effect=lambda *a, **kw: call_order.append("update") or None)

        s._cycle(SYMBOLS)

        sync_idx = call_order.index("sync")
        update_idx = call_order.index("update")
        assert sync_idx < update_idx, "sync_server_orders must run before update_position"

    def test_save_state_called_after_entries(self):
        """_save_state called after _scan_entries (not before)."""
        s = _make_scalper()
        call_order = []

        s._scan_entries = MagicMock(side_effect=lambda *a, **kw: call_order.append("entries"))
        s._save_state = MagicMock(side_effect=lambda: call_order.append("save"))

        s._cycle(SYMBOLS)

        entries_idx = call_order.index("entries")
        save_idx = call_order.index("save")
        assert entries_idx < save_idx, "_save_state must run after _scan_entries"
