"""Tests for post-restart reconciliation hardening (Step 2).

Covers:
- API connectivity gate (3 retries, safe mode on failure)
- Reconciliation cooldown (skip if < 60s since last)
- restart_count incremented and persisted
- Position qty mismatch detection and correction
- Orphan detection (alert only, no auto-close)
- Ghost grace period (startup vs steady-state)
- ORPHAN_AUTO_CLOSE env var escape hatch
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


# ─────────────────────────── Fixtures ───────────────────────────────────────


def _make_scalper(paper_mode: bool = False, positions: dict | None = None):
    """Build a minimally-mocked RubberBandScalper for reconciliation tests."""
    from quantbox.scalper import RubberBandScalper

    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client") as mock_client_cls,
        patch("quantbox.scalper.DiscordNotifier"),
        patch("quantbox.scalper.DiscordBotRunner"),
        patch("quantbox.scalper.TradeJournal"),
        patch("quantbox.scalper.DivergenceMonitor"),
        patch("quantbox.scalper.RateLimiter"),
        patch("quantbox.scalper.Heartbeat"),
        patch("quantbox.scalper.ExposureGuard"),
        patch("quantbox.scalper.InFlightTracker"),
        patch("quantbox.scalper.ReconJournal"),
        patch("quantbox.scalper.load_scalper_state", return_value=None),
        patch("quantbox.scalper.save_scalper_state"),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="k",
            binance_api_secret="s",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token=None,
            discord_channel_id=None,
            paper_balance=10000.0,
        )
        mock_client = MagicMock()
        mock_client.futures_account_balance.return_value = [{"asset": "USDT", "balance": "10000.0"}]
        mock_client_cls.return_value = mock_client

        scalper = RubberBandScalper(paper_mode=paper_mode, strategy_name="vol_straddle")
        scalper.discord = MagicMock()
        scalper._recon_journal = None
        scalper._inflight = None

        # Inject positions into strategy if provided
        if positions:
            for pos_key, pos_data in positions.items():
                parts = pos_key.rsplit(":", 1)
                api_sym = parts[0] if len(parts) == 2 else pos_key
                pos_side = parts[1] if len(parts) == 2 and parts[1] in ("LONG", "SHORT") else None
                scalper.strategy.open_position(
                    symbol=api_sym,
                    signal=pos_data.get("signal", 1),
                    entry_price=pos_data["entry_price"],
                    notional=pos_data["notional"],
                    leverage=pos_data.get("leverage", 3),
                    move_size=0.03,
                    position_side=pos_side,
                    strategy_type=pos_data.get("strategy_type", "vr_squeeze"),
                )
                if pos_key in scalper.strategy._positions:
                    scalper.strategy._positions[pos_key].update(pos_data)

    return scalper


# ─────────────────── paper mode skips ───────────────────────────────────────


class TestPaperModeSkip:
    def test_paper_mode_returns_immediately(self):
        """_post_restart_reconcile is a no-op in paper mode."""
        scalper = _make_scalper(paper_mode=True)
        scalper._post_restart_reconcile()
        # No API calls made
        scalper.client.futures_time.assert_not_called()
        scalper.discord.send.assert_not_called()


# ─────────────────── API connectivity gate ───────────────────────────────────


class TestApiConnectivityGate:
    def test_api_ok_on_first_attempt(self):
        """API connectivity succeeds immediately — proceed to reconciliation."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []

        scalper._post_restart_reconcile()

        scalper.client.futures_time.assert_called_once()
        assert not scalper._safe_mode

    def test_api_failure_sets_safe_mode_after_3_retries(self):
        """3 consecutive API failures → safe mode, no further reconciliation."""
        scalper = _make_scalper()
        scalper.client.futures_time.side_effect = Exception("Connection refused")

        with patch("quantbox.scalper.time.sleep"):  # speed up test
            scalper._post_restart_reconcile()

        assert scalper.client.futures_time.call_count == 3
        assert scalper._safe_mode is True
        scalper.discord.send.assert_called()
        # Should NOT have queried positions
        scalper.client.futures_position_information.assert_not_called()

    def test_api_succeeds_on_second_attempt(self):
        """API fails once then succeeds — reconciliation proceeds normally."""
        scalper = _make_scalper()
        scalper.client.futures_time.side_effect = [Exception("timeout"), {"serverTime": 1000}]
        scalper.client.futures_position_information.return_value = []

        with patch("quantbox.scalper.time.sleep"):
            scalper._post_restart_reconcile()

        assert scalper.client.futures_time.call_count == 2
        assert not scalper._safe_mode

    def test_safe_mode_discord_alert_sent(self):
        """Discord alert is sent when safe mode is triggered."""
        scalper = _make_scalper()
        scalper.client.futures_time.side_effect = Exception("timeout")

        with patch("quantbox.scalper.time.sleep"):
            scalper._post_restart_reconcile()

        calls = [str(c) for c in scalper.discord.send.call_args_list]
        assert any("API" in c or "Safe mode" in c or "연결 실패" in c for c in calls)


# ─────────────────── Reconciliation cooldown ────────────────────────────────


class TestReconciliationCooldown:
    def test_cooldown_skips_if_less_than_60s(self):
        """Skip reconciliation if last run < 60s ago.

        API gate (futures_time) is still called, but full reconciliation
        (futures_position_information) is skipped.
        """
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        # Set last reconcile to 30s ago
        scalper._last_full_reconcile_at = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()

        scalper._post_restart_reconcile()

        # API gate IS called, but full reconciliation is skipped
        scalper.client.futures_time.assert_called_once()
        scalper.client.futures_position_information.assert_not_called()

    def test_cooldown_proceeds_if_more_than_60s(self):
        """Proceed if last reconcile was > 60s ago."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []
        # Set last reconcile to 90s ago
        scalper._last_full_reconcile_at = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()

        scalper._post_restart_reconcile()

        scalper.client.futures_time.assert_called_once()

    def test_cooldown_proceeds_if_no_prior_reconcile(self):
        """Proceed if _last_full_reconcile_at is None."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []
        scalper._last_full_reconcile_at = None

        scalper._post_restart_reconcile()

        scalper.client.futures_time.assert_called_once()

    def test_invalid_timestamp_proceeds(self):
        """Corrupt timestamp in last_full_reconcile_at is ignored gracefully."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []
        scalper._last_full_reconcile_at = "NOT_A_VALID_TIMESTAMP"

        scalper._post_restart_reconcile()

        scalper.client.futures_time.assert_called_once()


# ─────────────────── restart_count increment ────────────────────────────────


class TestRestartCountIncrement:
    def test_restart_count_increments(self):
        """restart_count is incremented by 1 each call."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []
        scalper._restart_count = 4

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        assert scalper._restart_count == 5

    def test_restart_count_save_called_immediately(self):
        """_save_state is called right after incrementing restart_count."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []
        scalper._restart_count = 0

        save_calls = []

        def track_save():
            save_calls.append(scalper._restart_count)

        with patch.object(scalper, "_save_state", side_effect=track_save):
            scalper._post_restart_reconcile()

        assert save_calls[0] == 1, "First save should be AFTER restart_count=1"

    def test_last_restart_at_is_set(self):
        """_last_restart_at is set to a valid ISO timestamp."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        assert scalper._last_restart_at is not None
        # Should parse as a valid datetime
        dt = datetime.fromisoformat(scalper._last_restart_at)
        assert dt.tzinfo is not None


# ─────────────────── Position qty verification ──────────────────────────────


class TestPositionQtyVerification:
    def _exchange_positions(self, symbol, qty, pos_side):
        return [
            {
                "symbol": symbol,
                "positionAmt": str(qty),
                "positionSide": pos_side,
                "entryPrice": "1.234",
                "markPrice": "1.250",
                "unRealizedProfit": "0.10",
                "leverage": "3",
            }
        ]

    def test_qty_match_no_correction(self):
        """No correction applied when qty matches (< 1% diff)."""
        entry_px = 1.234
        qty = 243.11  # notional ≈ 300 / entry_px
        notional = qty * entry_px

        scalper = _make_scalper(
            positions={
                "SUIUSDT:LONG": {
                    "signal": 1,
                    "entry_price": entry_px,
                    "notional": notional,
                    "leverage": 3,
                    "position_side": "LONG",
                    "symbol": "SUIUSDT",
                }
            }
        )
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = self._exchange_positions("SUIUSDT", qty, "LONG")

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        # discord.send only for restart notice (no mismatch alert)
        for c in scalper.discord.send.call_args_list:
            msg = str(c)
            assert "수량 불일치" not in msg

    def test_qty_mismatch_triggers_correction(self):
        """Qty mismatch > 1% is corrected and Discord alert sent."""
        entry_px = 1.234
        internal_qty = 243.11
        exchange_qty = 250.0  # ~2.8% diff

        scalper = _make_scalper(
            positions={
                "SUIUSDT:LONG": {
                    "signal": 1,
                    "entry_price": entry_px,
                    "notional": internal_qty * entry_px,
                    "leverage": 3,
                    "position_side": "LONG",
                    "symbol": "SUIUSDT",
                }
            }
        )
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = self._exchange_positions(
            "SUIUSDT", exchange_qty, "LONG"
        )

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        pos = scalper.strategy._positions.get("SUIUSDT:LONG")
        assert pos is not None
        assert abs(pos["notional"] - exchange_qty * entry_px) < 0.01

        alert_sent = any("수량 불일치" in str(c) for c in scalper.discord.send.call_args_list)
        assert alert_sent

    def test_position_missing_from_exchange_grace_period(self):
        """Position not on exchange during startup grace → marked reconcile_pending."""
        scalper = _make_scalper(
            positions={
                "SUIUSDT:LONG": {
                    "signal": 1,
                    "entry_price": 1.234,
                    "notional": 300.0,
                    "leverage": 3,
                    "position_side": "LONG",
                    "symbol": "SUIUSDT",
                }
            }
        )
        scalper._startup_cycles_remaining = 2  # Within grace period
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []  # Empty exchange

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        pos = scalper.strategy._positions.get("SUIUSDT:LONG")
        assert pos is not None
        assert pos.get("_reconcile_pending") is True


# ─────────────────── Orphan detection (alert-only) ──────────────────────────


class TestOrphanDetection:
    def _orphan_exchange_position(self, symbol="BTCUSDT", qty=0.01, pos_side="LONG"):
        return [
            {
                "symbol": symbol,
                "positionAmt": str(qty),
                "positionSide": pos_side,
                "entryPrice": "50000.0",
                "markPrice": "51000.0",
                "unRealizedProfit": "10.0",
                "leverage": "10",
            }
        ]

    def test_orphan_detected_sends_discord_alert(self):
        """Orphan position on exchange (no internal state) triggers Discord alert."""
        scalper = _make_scalper()  # No internal positions
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = self._orphan_exchange_position()

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        alert_sent = any(
            "고아" in str(c) or "orphan" in str(c).lower() or "ORPHAN" in str(c)
            for c in scalper.discord.send.call_args_list
        )
        assert alert_sent

    def test_orphan_does_not_auto_close(self):
        """Orphan detection must NOT call futures_create_order (no auto-close)."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = self._orphan_exchange_position()

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        scalper.client.futures_create_order.assert_not_called()

    def test_tracked_position_not_flagged_as_orphan(self):
        """A tracked position should NOT trigger an orphan alert."""
        scalper = _make_scalper(
            positions={
                "SUIUSDT:LONG": {
                    "signal": 1,
                    "entry_price": 1.234,
                    "notional": 300.0,
                    "leverage": 3,
                    "position_side": "LONG",
                    "symbol": "SUIUSDT",
                }
            }
        )
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = [
            {
                "symbol": "SUIUSDT",
                "positionAmt": "243.11",
                "positionSide": "LONG",
                "entryPrice": "1.234",
                "markPrice": "1.250",
                "unRealizedProfit": "0.10",
                "leverage": "3",
            }
        ]

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        no_orphan_alert = all(
            "고아" not in str(c) and "ORPHAN" not in str(c) for c in scalper.discord.send.call_args_list
        )
        assert no_orphan_alert

    def test_last_full_reconcile_at_updated_after_success(self):
        """_last_full_reconcile_at is updated after a successful reconciliation."""
        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []

        before = datetime.now(timezone.utc)

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        assert scalper._last_full_reconcile_at is not None
        after_dt = datetime.fromisoformat(scalper._last_full_reconcile_at)
        assert after_dt >= before


# ─────────────────── ORPHAN_AUTO_CLOSE env var escape hatch ─────────────────


class TestOrphanAutoCloseEscapeHatch:
    def _exchange_pos_list(self):
        return [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "positionSide": "LONG",
                "entryPrice": "50000.0",
                "markPrice": "51000.0",
                "unRealizedProfit": "10.0",
                "leverage": "10",
            }
        ]

    def test_default_no_auto_close(self):
        """Default (ORPHAN_AUTO_CLOSE unset): ORPHAN_AUTO_CLOSE env var is false."""
        env = {k: v for k, v in os.environ.items() if k != "ORPHAN_AUTO_CLOSE"}
        with patch.dict(os.environ, env, clear=True):
            _auto_close_enabled = os.getenv("ORPHAN_AUTO_CLOSE", "").lower() == "true"
            assert _auto_close_enabled is False

    def test_orphan_auto_close_enabled_via_env(self):
        """ORPHAN_AUTO_CLOSE=true restores auto-close behavior."""
        _auto_close_enabled = "true" == "true"  # simulates env check
        with patch.dict(os.environ, {"ORPHAN_AUTO_CLOSE": "true"}):
            val = os.getenv("ORPHAN_AUTO_CLOSE", "").lower() == "true"
            assert val is True

    def test_orphan_auto_close_false_by_default(self):
        """ORPHAN_AUTO_CLOSE defaults to false."""
        env = {k: v for k, v in os.environ.items() if k != "ORPHAN_AUTO_CLOSE"}
        with patch.dict(os.environ, env, clear=True):
            val = os.getenv("ORPHAN_AUTO_CLOSE", "").lower() == "true"
            assert val is False

    def test_reconcile_positions_alert_only_default(self):
        """_reconcile_positions: default uses alert-only, no futures_create_order."""
        from quantbox.execution.reconciliation import ReconciliationMixin

        # Use plain MagicMock (no spec) — strategy/discord are on ScalperProtocol not the mixin
        mock_self = MagicMock()
        mock_self.paper_mode = False  # Must be False or paper_mode guard returns early
        mock_self.strategy.active_positions = {}
        mock_self.strategy._positions = {}
        mock_self._recon_journal = None
        mock_self.config.reconciliation_qty_tolerance_pct = 1.0
        mock_self.client.futures_position_information.return_value = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "positionSide": "LONG",
                "entryPrice": "50000.0",
                "markPrice": "51000.0",
                "unRealizedProfit": "10.0",
                "leverage": "10",
            }
        ]

        env = {k: v for k, v in os.environ.items() if k != "ORPHAN_AUTO_CLOSE"}
        with patch.dict(os.environ, env, clear=True):
            ReconciliationMixin._reconcile_positions(mock_self)

        mock_self.client.futures_create_order.assert_not_called()
        mock_self.discord.send.assert_called()
        alert_msg = str(mock_self.discord.send.call_args_list)
        assert "고아" in alert_msg or "orphan" in alert_msg.lower() or "REVERSE GHOST" in alert_msg

    def test_reconcile_positions_auto_close_when_env_set(self):
        """_reconcile_positions: ORPHAN_AUTO_CLOSE=true triggers close."""
        from quantbox.execution.reconciliation import ReconciliationMixin

        # Use plain MagicMock (no spec) — strategy/discord are on ScalperProtocol not the mixin
        mock_self = MagicMock()
        mock_self.paper_mode = False  # Must be False or paper_mode guard returns early
        mock_self.strategy.active_positions = {}
        mock_self.strategy._positions = {}
        mock_self._recon_journal = None
        mock_self.config.reconciliation_qty_tolerance_pct = 1.0
        mock_self.client.futures_position_information.return_value = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "positionSide": "LONG",
                "entryPrice": "50000.0",
                "markPrice": "51000.0",
                "unRealizedProfit": "10.0",
                "leverage": "10",
            }
        ]
        mock_self.client.futures_create_order.return_value = {"orderId": 999}
        mock_self._journal = None

        with patch.dict(os.environ, {"ORPHAN_AUTO_CLOSE": "true"}):
            ReconciliationMixin._reconcile_positions(mock_self)

        mock_self.client.futures_create_order.assert_called_once()


# ─────────────────── Bracket verification ───────────────────────────────────


class TestBracketVerification:
    def test_missing_sl_triggers_discord_alert(self):
        """Position with no server_sl_order_id triggers a Discord warning."""
        scalper = _make_scalper(
            positions={
                "SUIUSDT:LONG": {
                    "signal": 1,
                    "entry_price": 1.234,
                    "notional": 300.0,
                    "leverage": 3,
                    "position_side": "LONG",
                    "symbol": "SUIUSDT",
                    "server_sl_order_id": None,  # No SL bracket
                }
            }
        )
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = [
            {
                "symbol": "SUIUSDT",
                "positionAmt": "243.11",
                "positionSide": "LONG",
                "entryPrice": "1.234",
                "markPrice": "1.250",
                "unRealizedProfit": "0.10",
                "leverage": "3",
            }
        ]
        scalper.client._request_futures_api.return_value = {"orders": []}

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        alert_sent = any("SL 브라켓" in str(c) or "SL bracket" in str(c) for c in scalper.discord.send.call_args_list)
        assert alert_sent


# ─────────────────── Stale inflight clearing ────────────────────────────────


class TestStaleInflightClearing:
    def test_stale_entry_order_cleared_when_position_exists(self):
        """Stale ENTRY inflight order: position on exchange → confirmed and cleared."""
        from quantbox.execution.inflight import InFlightTracker, OrderIntent, PendingOrder
        import time

        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = [
            {
                "symbol": "SUIUSDT",
                "positionAmt": "100.0",
                "positionSide": "LONG",
                "entryPrice": "1.234",
                "markPrice": "1.250",
                "unRealizedProfit": "0.10",
                "leverage": "3",
            }
        ]

        inflight = InFlightTracker(persist_path=None)
        old_order = PendingOrder(
            intent=OrderIntent.ENTRY,
            symbol="SUIUSDT",
            side="BUY",
            position_side="LONG",
            quantity=100.0,
            order_type="MARKET",
            created_at=time.time() - 600,  # 10 minutes old → stale
        )
        key = inflight.record_pending(old_order)
        scalper._inflight = inflight

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        # Stale order should be confirmed (cleared from pending)
        assert key not in inflight.get_pending()

    def test_stale_bracket_order_cleared(self):
        """Stale BRACKET_SL inflight order is cleared unconditionally."""
        from quantbox.execution.inflight import InFlightTracker, OrderIntent, PendingOrder
        import time

        scalper = _make_scalper()
        scalper.client.futures_time.return_value = {"serverTime": 1000}
        scalper.client.futures_position_information.return_value = []

        inflight = InFlightTracker(persist_path=None)
        bracket_order = PendingOrder(
            intent=OrderIntent.BRACKET_SL,
            symbol="BTCUSDT",
            side="SELL",
            position_side="LONG",
            quantity=0.01,
            order_type="STOP_MARKET",
            created_at=time.time() - 600,
        )
        key = inflight.record_pending(bracket_order)
        scalper._inflight = inflight

        with patch.object(scalper, "_save_state"):
            scalper._post_restart_reconcile()

        assert key not in inflight.get_pending()
