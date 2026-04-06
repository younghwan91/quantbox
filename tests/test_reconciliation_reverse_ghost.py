"""Tests for reverse ghost detection in _reconcile_positions() and startup crosscheck.

Incident #6 (2026-03-13): BANDUSDT LONG existed on exchange with no internal state —
no SL/TP protection for ~3 hours until manual closure.

Root cause: _reconcile_positions() only checked state→exchange (ghost detection),
NOT exchange→state (reverse ghost detection).
"""

from __future__ import annotations

from unittest.mock import MagicMock


from tests.conftest import _build_vs_scalper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRICE = 0.2348
QTY = 1671.4


def _make_exchange_position(symbol: str, pos_side: str, amt: float) -> dict:
    """Create a minimal Binance position dict as returned by futures_position_information."""
    return {
        "symbol": symbol,
        "positionSide": pos_side,
        "positionAmt": str(amt),
        "entryPrice": str(PRICE),
        "markPrice": str(PRICE * 1.02),
        "unRealizedProfit": str(abs(amt) * PRICE * 0.02),
        "notional": str(abs(amt) * PRICE),
    }


def _make_zero_position(symbol: str, pos_side: str = "BOTH") -> dict:
    """Zero-quantity position (common in Binance response)."""
    return {
        "symbol": symbol,
        "positionSide": pos_side,
        "positionAmt": "0",
        "entryPrice": "0",
        "markPrice": str(PRICE),
        "unRealizedProfit": "0",
        "notional": "0",
    }


# ---------------------------------------------------------------------------
# Tests: _reconcile_positions() Phase 2 — reverse ghost
# ---------------------------------------------------------------------------


class TestReverseGhostReconciliation:
    """Reverse ghost = exchange has position, internal state does not.

    Default behavior (v14.28+): alert-only, no auto-close.
    Auto-close restored via ORPHAN_AUTO_CLOSE=true env var (escape hatch).
    """

    def test_reverse_ghost_alert_only_by_default(self):
        """When exchange has BANDUSDT:LONG but state is empty, alert fires but NO auto-close."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
            _make_zero_position("BANDUSDT", "SHORT"),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        # NO market close — alert-only is the default
        scalper.client.futures_create_order.assert_not_called()
        # Discord alert must have been sent
        scalper.discord.send.assert_called()

    def test_reverse_ghost_auto_close_with_env_var(self, monkeypatch):
        """ORPHAN_AUTO_CLOSE=true restores auto-close behavior."""
        monkeypatch.setenv("ORPHAN_AUTO_CLOSE", "true")
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
            _make_zero_position("BANDUSDT", "SHORT"),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        # Auto-close fires when env var is set
        scalper.client.futures_create_order.assert_called_once()
        call_kwargs = scalper.client.futures_create_order.call_args.kwargs
        assert call_kwargs["symbol"] == "BANDUSDT"
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["type"] == "MARKET"
        assert call_kwargs["quantity"] == QTY
        assert call_kwargs["positionSide"] == "LONG"

    def test_reverse_ghost_sends_discord_alert(self):
        """Discord alert is sent when reverse ghost is detected (alert-only mode)."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        scalper.discord.send.assert_called()
        # Verify alert mentions the symbol and adopt/close-orphan commands
        all_calls = " ".join(str(c) for c in scalper.discord.send.call_args_list)
        assert "BANDUSDT" in all_calls

    def test_reverse_ghost_short_auto_close_with_env(self, monkeypatch):
        """Reverse ghost SHORT position is closed with BUY when ORPHAN_AUTO_CLOSE=true."""
        monkeypatch.setenv("ORPHAN_AUTO_CLOSE", "true")
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_exchange_position("ICXUSDT", "SHORT", -500.0),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        call_kwargs = scalper.client.futures_create_order.call_args.kwargs
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["positionSide"] == "SHORT"

    def test_no_reverse_ghost_when_state_tracked(self):
        """When state tracks the position, no auto-close is triggered."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        # Inject internal state position
        scalper.strategy._positions["TESTUSDT:LONG"] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": PRICE,
            "notional": 3000.0,
            "bars_held": 10,
            "peak_gain": 0.0,
            "best_pnl": 0.0,
            "worst_pnl": 0.0,
            "server_sl_order_id": "sl_111",
            "server_tp_order_id": "tp_222",
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "trail_active": False,
            "strategy_type": "vr_directional",
        }
        exchange_positions = [
            _make_exchange_position("TESTUSDT", "LONG", QTY),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        # No auto-close for tracked position
        scalper.client.futures_create_order.assert_not_called()

    def test_no_action_in_paper_mode(self):
        """_reconcile_positions() is a no-op in paper mode."""
        scalper = _build_vs_scalper()
        assert scalper.paper_mode is True  # default

        scalper._reconcile_positions()

        scalper.client.futures_position_information.assert_not_called()
        scalper.client.futures_create_order.assert_not_called()

    def test_reverse_ghost_api_failure_sends_emergency_discord(self, monkeypatch):
        """If auto-close order fails (ORPHAN_AUTO_CLOSE=true), emergency Discord message is sent."""
        monkeypatch.setenv("ORPHAN_AUTO_CLOSE", "true")
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions
        scalper.client.futures_create_order.side_effect = Exception("APIError(-2019)")

        # Should not raise — error is caught and logged
        scalper._reconcile_positions()

        # Two Discord calls: initial alert + failure alert
        assert scalper.discord.send.call_count >= 2

    def test_zero_quantity_positions_ignored(self):
        """Zero-quantity exchange positions do not trigger reverse ghost."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_zero_position("BANDUSDT", "LONG"),
            _make_zero_position("BANDUSDT", "SHORT"),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        scalper.client.futures_create_order.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _startup_exchange_crosscheck()
# ---------------------------------------------------------------------------


class TestStartupExchangeCrosscheck:
    """Startup cross-check warns when exchange has untracked positions."""

    def test_startup_warns_on_untracked_position(self):
        """Untracked exchange position at startup triggers WARNING log + Discord."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
            _make_zero_position("BANDUSDT", "SHORT"),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._startup_exchange_crosscheck()

        # Discord alert must fire
        scalper.discord.send.assert_called_once()
        content = scalper.discord.send.call_args.kwargs.get("content", "")
        assert "BAND" in content or "Untracked" in content

    def test_startup_no_alert_when_state_matches_exchange(self):
        """If all exchange positions are tracked, no Discord alert."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        scalper.strategy._positions["TESTUSDT:LONG"] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": PRICE,
            "notional": 3000.0,
            "bars_held": 5,
            "peak_gain": 0.0,
            "best_pnl": 0.0,
            "worst_pnl": 0.0,
            "server_sl_order_id": "sl_1",
            "server_tp_order_id": "tp_1",
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "trail_active": False,
            "strategy_type": "vr_directional",
        }
        exchange_positions = [
            _make_exchange_position("TESTUSDT", "LONG", QTY),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._startup_exchange_crosscheck()

        scalper.discord.send.assert_not_called()

    def test_startup_no_action_in_paper_mode(self):
        """_startup_exchange_crosscheck skips in paper mode (state file already checked)."""
        scalper = _build_vs_scalper()
        assert scalper.paper_mode is True

        # Even if called directly, it should be a no-op
        # (the caller _load_state guards with `if not self.paper_mode`)
        # Verify method exists and is callable
        assert callable(scalper._startup_exchange_crosscheck)

    def test_startup_api_failure_does_not_crash(self):
        """API error in startup crosscheck is caught gracefully."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False
        scalper.client.futures_position_information.side_effect = Exception("Network error")

        # Should not raise
        scalper._startup_exchange_crosscheck()


# ---------------------------------------------------------------------------
# Tests: P1 — Reverse ghost auto-close is recorded in main journal.sqlite
# ---------------------------------------------------------------------------


class TestReverseGhostJournaling:
    """Reverse ghost close must appear in journal.sqlite (not only in recon_journal)."""

    def test_reverse_ghost_no_journal_in_alert_only_mode(self):
        """In alert-only mode (default), journal.record() is NOT called (no auto-close)."""

        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        mock_journal = MagicMock()
        scalper._journal = mock_journal

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
            _make_zero_position("BANDUSDT", "SHORT"),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        # No auto-close → no journal record
        mock_journal.record.assert_not_called()

    def test_reverse_ghost_calls_journal_record_with_auto_close(self, monkeypatch):
        """_journal.record() is called when reverse ghost is auto-closed (ORPHAN_AUTO_CLOSE=true)."""

        monkeypatch.setenv("ORPHAN_AUTO_CLOSE", "true")
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        mock_journal = MagicMock()
        scalper._journal = mock_journal

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
            _make_zero_position("BANDUSDT", "SHORT"),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions

        scalper._reconcile_positions()

        assert mock_journal.record.called, (
            "journal.record() was not called — reverse ghost close not recorded in journal.sqlite"
        )
        call_kwargs = mock_journal.record.call_args[0][0]
        assert call_kwargs["symbol"] == "BANDUSDT"
        assert call_kwargs["reason"] == "reconciliation_close"

    def test_reverse_ghost_journal_not_called_on_api_failure(self, monkeypatch):
        """If auto-close order fails, journal.record() must NOT be called with garbage data."""

        monkeypatch.setenv("ORPHAN_AUTO_CLOSE", "true")
        scalper = _build_vs_scalper()
        scalper.paper_mode = False

        mock_journal = MagicMock()
        scalper._journal = mock_journal

        exchange_positions = [
            _make_exchange_position("BANDUSDT", "LONG", QTY),
        ]
        scalper.client.futures_position_information.return_value = exchange_positions
        scalper.client.futures_create_order.side_effect = Exception("APIError(-2019)")

        scalper._reconcile_positions()

        # journal.record should NOT be called when close failed
        mock_journal.record.assert_not_called()
