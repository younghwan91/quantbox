"""Tests for state migration: v14.27 state loads cleanly with new fields defaulting safely."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_v14_27_state() -> dict:
    """Minimal v14.27 state dict (no restart_count / last_restart_at / last_full_reconcile_at fields)."""
    return {
        "version": "2.0.0",
        "saved_at": "2026-03-15T10:00:00+00:00",
        "paper_mode": False,
        "paper_balance": 10000.0,
        "initial_balance": 10000.0,
        "high_water_mark": 10100.0,
        "positions": {},
        "trade_history": [],
        "cycle_count": 42,
        "daily_loss_blocked": False,
        "daily_realized_pnl": 0.0,
        "daily_loss_date": "2026-03-15",
        # NOTE: restart_count, last_restart_at, last_full_reconcile_at intentionally absent
    }


def _make_v14_27_state_with_position() -> dict:
    state = _make_v14_27_state()
    state["positions"] = {
        "SUIUSDT:LONG": {
            "symbol": "SUIUSDT",
            "signal": 1,
            "entry_price": 1.234,
            "notional": 300.0,
            "leverage": 3,
            "move_size": 0.03,
            "position_side": "LONG",
            "bars_held": 2,
            "best_pnl": 0.01,
            "partial_closed": False,
            "highest_since_tp1": 0.0,
            "trail_floor": None,
            "confidence": 0,
            "_computed_trail_stop": None,
            "_server_sl_pct": 0.0,
            "_trail_sl_ratcheted": -999.0,
            "trail_active": False,
            "peak_gain": 0.0,
            "is_straddle": False,
            "strategy_type": "vr_directional",
            "sl": 0.03,
            "tp": 0.06,
            "tf": "15m",
            "max_hold": 8,
        }
    }
    return state


class TestStateMigration:
    """Verify v14.27 state loads with safe defaults for all new fields."""

    def test_old_state_missing_restart_fields_defaults_to_zero(self, tmp_path):
        """restart_count / last_restart_at / last_full_reconcile_at default to safe values."""
        from quantbox.scalper import RubberBandScalper

        old_state = _make_v14_27_state()

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
            patch("quantbox.scalper.load_scalper_state", return_value=old_state),
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
            mock_client_cls.return_value = MagicMock()
            mock_client_cls.return_value.futures_account_balance.return_value = [
                {"asset": "USDT", "balance": "10000.0"}
            ]

            scalper = RubberBandScalper(paper_mode=False, strategy_name="vol_straddle")
            scalper._load_state()

        # New fields should default safely
        assert scalper._restart_count == 0
        assert scalper._last_restart_at is None
        assert scalper._last_full_reconcile_at is None

    def test_old_state_with_new_fields_restores_correctly(self, tmp_path):
        """If state already has restart fields (new format), they are restored."""
        from quantbox.scalper import RubberBandScalper

        state_with_fields = _make_v14_27_state()
        state_with_fields["restart_count"] = 5
        state_with_fields["last_restart_at"] = "2026-03-15T09:00:00+00:00"
        state_with_fields["last_full_reconcile_at"] = "2026-03-15T09:01:00+00:00"

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
            patch("quantbox.scalper.load_scalper_state", return_value=state_with_fields),
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
            mock_client_cls.return_value = MagicMock()
            mock_client_cls.return_value.futures_account_balance.return_value = [
                {"asset": "USDT", "balance": "10000.0"}
            ]

            scalper = RubberBandScalper(paper_mode=False, strategy_name="vol_straddle")
            scalper._load_state()

        assert scalper._restart_count == 5
        assert scalper._last_restart_at == "2026-03-15T09:00:00+00:00"
        assert scalper._last_full_reconcile_at == "2026-03-15T09:01:00+00:00"

    def test_save_state_includes_restart_fields(self, tmp_path):
        """_save_state persists restart_count, last_restart_at, last_full_reconcile_at."""
        from quantbox.scalper import RubberBandScalper

        saved_states: list[dict] = []

        def capture_save(state, filepath=None):
            saved_states.append(state)

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
            patch("quantbox.scalper.save_scalper_state", side_effect=capture_save),
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
            mock_client_cls.return_value = MagicMock()
            mock_client_cls.return_value.futures_account_balance.return_value = [
                {"asset": "USDT", "balance": "10000.0"}
            ]

            scalper = RubberBandScalper(paper_mode=False, strategy_name="vol_straddle")
            scalper._restart_count = 3
            scalper._last_restart_at = "2026-03-15T08:00:00+00:00"
            scalper._last_full_reconcile_at = "2026-03-15T08:01:00+00:00"
            scalper._save_state()

        assert saved_states, "save_scalper_state should have been called"
        state = saved_states[-1]
        assert state["restart_count"] == 3
        assert state["last_restart_at"] == "2026-03-15T08:00:00+00:00"
        assert state["last_full_reconcile_at"] == "2026-03-15T08:01:00+00:00"

    def test_daily_loss_fields_still_present_in_save(self, tmp_path):
        """daily_loss_* fields are still saved (not removed by this migration)."""
        from quantbox.scalper import RubberBandScalper

        saved_states: list[dict] = []

        def capture_save(state, filepath=None):
            saved_states.append(state)

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
            patch("quantbox.scalper.save_scalper_state", side_effect=capture_save),
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
            mock_client_cls.return_value = MagicMock()
            mock_client_cls.return_value.futures_account_balance.return_value = [
                {"asset": "USDT", "balance": "10000.0"}
            ]

            scalper = RubberBandScalper(paper_mode=False, strategy_name="vol_straddle")
            scalper._daily_loss_blocked = True
            scalper._daily_realized_pnl = -150.0
            scalper._daily_loss_date = "2026-03-15"
            scalper._save_state()

        assert saved_states
        state = saved_states[-1]
        assert state["daily_loss_blocked"] is True
        assert state["daily_realized_pnl"] == -150.0
        assert state["daily_loss_date"] == "2026-03-15"

    def test_old_state_with_position_restores_without_error(self, tmp_path):
        """v14.27 state with a position loads cleanly without KeyError on new fields."""
        from quantbox.scalper import RubberBandScalper

        old_state = _make_v14_27_state_with_position()

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
            patch("quantbox.scalper.load_scalper_state", return_value=old_state),
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
            mock_client_cls.return_value = MagicMock()
            mock_client_cls.return_value.futures_account_balance.return_value = [
                {"asset": "USDT", "balance": "10000.0"}
            ]

            scalper = RubberBandScalper(paper_mode=False, strategy_name="vol_straddle")
            # Should not raise
            scalper._load_state()

        # Defaults applied
        assert scalper._restart_count == 0
        assert scalper._last_restart_at is None
        assert scalper._last_full_reconcile_at is None
        # Position was restored
        assert scalper.strategy.n_active == 1
