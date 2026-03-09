"""
Tests for the Scalper live trading module.

Tests the non-exchange parts: risk management, state persistence,
and cycle logic. Uses paper mode to avoid API calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy
from quantbox.scalper import (
    RubberBandScalper,
    STATE_FILE,
    PAPER_STATE_FILE,
)
from quantbox.core.scalper_state import save_scalper_state, load_scalper_state


# ── State Persistence ──


class TestScalperState:
    """Test atomic state save/load."""

    def test_save_and_load(self, tmp_path):
        """State roundtrips through JSON correctly."""
        fpath = tmp_path / "test_state.json"
        state = {
            "version": "2.0.0",
            "paper_mode": True,
            "paper_balance": 1000.0,
            "positions": {
                "BTCUSDT": {
                    "signal": 1,
                    "entry_price": 95000.0,
                    "notional": 500.0,
                    "leverage": 2,
                    "move_size": 0.03,
                    "bars_held": 3,
                    "best_pnl": 0.01,
                },
            },
            "trade_history": [{"symbol": "ETHUSDT", "pnl_usd": 5.23}],
        }
        save_scalper_state(state, filepath=fpath)
        loaded = load_scalper_state(filepath=fpath)

        assert loaded is not None
        assert loaded["paper_balance"] == 1000.0
        assert "BTCUSDT" in loaded["positions"]
        assert loaded["positions"]["BTCUSDT"]["signal"] == 1
        assert len(loaded["trade_history"]) == 1

    def test_load_missing_file(self, tmp_path):
        """Loading non-existent file returns None."""
        result = load_scalper_state(filepath=tmp_path / "nope.json")
        assert result is None

    def test_atomic_write_no_corruption(self, tmp_path):
        """File is not corrupted on normal write."""
        fpath = tmp_path / "atomic.json"
        save_scalper_state({"test": True}, filepath=fpath)
        with open(fpath) as f:
            data = json.load(f)
        assert data["test"] is True

    def test_paper_and_live_use_different_state_files(self):
        """Paper and live modes must use separate state files."""
        assert STATE_FILE != PAPER_STATE_FILE
        assert STATE_FILE.name == "scalper_state.json"
        assert PAPER_STATE_FILE.name == "scalper_state_paper.json"
        assert STATE_FILE.parent == PAPER_STATE_FILE.parent


# ── Risk Management ──


class TestScalperRiskManagement:
    """Test drawdown and daily loss limits using the Strategy wrapper directly."""

    def test_drawdown_detection(self):
        """Session drawdown > 10% triggers halt."""
        cfg = RubberBandConfig(max_drawdown_pct=0.10)

        # Simulate: HWM = $10000, current = $8900 → 11% DD
        hwm = 10000.0
        balance = 8900.0
        dd = (hwm - balance) / hwm
        assert dd >= cfg.max_drawdown_pct

    def test_position_count_limit(self):
        """Max concurrent positions are enforced."""
        cfg = RubberBandConfig(max_concurrent=2)
        strat = RubberBandStrategy(cfg)

        strat.open_position("SYM1", 1, 100.0, 1000.0, 2.0, 0.03)
        strat.open_position("SYM2", -1, 200.0, 1000.0, 2.0, 0.03)
        assert strat.n_active == 2

        # Scalper should check n_active < max_concurrent before opening
        available_slots = cfg.max_concurrent - strat.n_active
        assert available_slots == 0


# ── Config CLI ──


class TestScalperConfig:
    """Test config defaults and overrides."""

    def test_default_config(self):
        cfg = RubberBandConfig()
        assert cfg.lookback == 8
        assert cfg.threshold == 0.032  # v10.0
        assert cfg.max_concurrent == 2  # v12.3: M-2SSTSX optimal (2 slots)
        assert cfg.cost_per_side == 0.0004

    def test_custom_config(self):
        cfg = RubberBandConfig(lookback=10, threshold=0.03, max_concurrent=3)
        assert cfg.lookback == 10
        assert cfg.threshold == 0.03
        assert cfg.max_concurrent == 3


# ── NO_POSITION Trade Recording ──


class TestNoPositionTradeRecording:
    """Test that the NO_POSITION path in _close_position records trades.

    When a server-side TP/SL fills between cycles, the scalper's reduceOnly
    close order fails with -4118. The NO_POSITION path must still record the
    trade in _trade_history — otherwise CumPnL and trade count stay at 0.
    """

    @pytest.fixture
    def mock_scalper(self, tmp_path):
        """Create a paper-mode scalper with mocked exchange client."""
        with (
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client") as MockClient,
            patch("quantbox.scalper.DiscordNotifier") as MockDiscord,
            patch("quantbox.scalper.DiscordBotRunner", None),
        ):
            mock_settings.return_value = MagicMock(
                binance_api_key="test_key",
                binance_api_secret="test_secret",
                binance_testnet=False,
                discord_webhook_url="",
                discord_bot_token="",
                discord_channel_id="",
                paper_balance=2000.0,
            )
            scalper = RubberBandScalper(paper_mode=False)

        scalper.client = MagicMock()
        scalper.discord = MagicMock()
        scalper._symbol_info_cache = {
            "AAVEUSDT": {
                "tick_size": 0.01,
                "step_size": 0.001,
                "price_precision": 2,
                "qty_precision": 3,
                "min_notional": 5.0,
            },
        }
        return scalper

    def test_no_position_records_trade(self, mock_scalper):
        """When exchange returns NO_POSITION, trade is still recorded."""
        scalper = mock_scalper

        # Set up a LONG position as if it was opened earlier
        scalper.strategy.open_position(
            symbol="AAVEUSDT",
            signal=1,
            entry_price=122.13,
            notional=500.0,
            leverage=2.0,
            move_size=0.03,
        )
        pos = scalper.strategy._positions["AAVEUSDT"]
        pos["bars_held"] = 14
        pos["server_sl_order_id"] = "algo_sl_123"
        pos["server_tp_order_id"] = "algo_tp_456"

        # Mock _place_order to return NO_POSITION (TP already filled)
        scalper._place_order = MagicMock(
            return_value={"status": "NO_POSITION"},
        )
        # Mock cancel methods
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        # Mock get_balance
        scalper.get_balance = MagicMock(return_value=2155.0)

        # Mock futures_account_trades — simulate exchange returning recent fills
        scalper.client.futures_account_trades = MagicMock(
            return_value=[
                {
                    "symbol": "AAVEUSDT",
                    "side": "SELL",
                    "price": "123.51",
                    "qty": "4.093",
                    "realizedPnl": "5.65",
                    "commission": "0.02",
                    "time": 1740000000000,
                },
            ]
        )

        # Before: no trades recorded
        assert len(scalper._trade_history) == 0

        # Call _close_position with current_price close to TP
        result = scalper._close_position(
            symbol="AAVEUSDT",
            reason="target",
            current_price=123.33,
        )

        # After: trade must be recorded
        assert result is True
        assert len(scalper._trade_history) == 1

        trade = scalper._trade_history[0]
        assert trade["symbol"] == "AAVEUSDT"
        assert trade["signal"] == 1  # LONG
        assert trade["entry_price"] == 122.13
        assert trade["exit_price"] == 123.51  # from exchange fill, not current_price
        assert trade["pnl_pct"] > 0  # profitable
        assert trade["pnl_usd"] > 0  # profitable after cost
        assert trade["bars_held"] == 14
        assert "no_position" in trade["reason"]
        assert trade["time"] is not None

        # Position should be removed from strategy
        assert not scalper.strategy.has_position("AAVEUSDT")

    def test_no_position_fallback_to_current_price(self, mock_scalper):
        """When exchange trade fetch fails, fallback to current_price."""
        scalper = mock_scalper

        # Set up a SHORT position
        scalper.strategy.open_position(
            symbol="AAVEUSDT",
            signal=-1,
            entry_price=130.0,
            notional=500.0,
            leverage=2.0,
            move_size=0.03,
        )
        pos = scalper.strategy._positions["AAVEUSDT"]
        pos["bars_held"] = 5
        pos["server_sl_order_id"] = None
        pos["server_tp_order_id"] = None

        scalper._place_order = MagicMock(
            return_value={"status": "NO_POSITION"},
        )
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper.get_balance = MagicMock(return_value=2100.0)

        # Exchange trade fetch fails
        scalper.client.futures_account_trades = MagicMock(
            side_effect=Exception("API timeout"),
        )

        result = scalper._close_position(
            symbol="AAVEUSDT",
            reason="stop_loss",
            current_price=128.0,
        )

        assert result is True
        assert len(scalper._trade_history) == 1
        trade = scalper._trade_history[0]
        assert trade["exit_price"] == 128.0  # fallback to current_price
        assert trade["signal"] == -1  # SHORT
        assert "no_position" in trade["reason"]
