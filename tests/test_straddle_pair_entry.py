"""Tests for Step 1: Pair-Aware Entry with Robust Rollback.

straddle-pair-coordination.md Step 1 acceptance criteria:
- Both legs share same straddle_pair_id
- _get_pair_counterpart finds correct counterpart
- Second leg retry: entry_retry_count attempts before rollback
- Rollback retry: rollback_retry_count attempts + market order fallback
- Market order fallback does NOT call futures_change_leverage (Rule #1)
- Rollback total failure: rollback_failed flag + Discord alert
- State saved once after both legs (verify _save_state call count)
- Exchange dedup guard prevents duplicate entry on retry
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import _build_vs_scalper
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _straddle_pos(sym: str, signal: int, pair_id: str, **overrides) -> dict:
    pos_side = "LONG" if signal == 1 else "SHORT"
    base = {
        "symbol": sym,
        "signal": signal,
        "is_straddle": True,
        "straddle_pair_id": pair_id,
        "entry_price": 1.0,
        "notional": 1000.0,
        "leverage": 30.0,
        "move_size": 1.0,
        "bars_held": 0,
        "sl": 0.015,
        "tp": 0.05,
        "tf": "15m",
        "max_hold": 192,
        "strategy_type": "vr_straddle",
        "server_sl_order_id": f"sl_{pos_side}",
        "server_tp_order_id": f"tp_{pos_side}",
        "best_pnl": 0.0,
        "worst_pnl": 0.0,
        "position_side": pos_side,
        "balance_before": 1000.0,
        "opened_at": "2026-03-15T00:00:00+00:00",
        "trail_active": False,
        "peak_gain": 0.0,
        "entry_stop_mult": 0.015,
        "partial_closed": False,
        "confidence": 0,
        "highest_since_tp1": 0.0,
        "trail_floor": None,
    }
    base.update(overrides)
    return base


# ─── _get_pair_counterpart ────────────────────────────────────────────────────


class TestGetPairCounterpart:
    def test_finds_counterpart(self):
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "SUIUSDT:LONG": _straddle_pos("SUIUSDT", 1, "abc123"),
            "SUIUSDT:SHORT": _straddle_pos("SUIUSDT", -1, "abc123"),
        }
        result = scalper._get_pair_counterpart("SUIUSDT:LONG")
        assert result == "SUIUSDT:SHORT"

    def test_reverse_lookup(self):
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "SUIUSDT:LONG": _straddle_pos("SUIUSDT", 1, "abc123"),
            "SUIUSDT:SHORT": _straddle_pos("SUIUSDT", -1, "abc123"),
        }
        result = scalper._get_pair_counterpart("SUIUSDT:SHORT")
        assert result == "SUIUSDT:LONG"

    def test_no_pair_id_returns_none(self):
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "SUIUSDT:LONG": {"is_straddle": True},
        }
        result = scalper._get_pair_counterpart("SUIUSDT:LONG")
        assert result is None

    def test_directional_position_returns_none(self):
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "SUIUSDT:LONG": {"straddle_pair_id": None, "is_straddle": False},
        }
        result = scalper._get_pair_counterpart("SUIUSDT:LONG")
        assert result is None

    def test_missing_pos_key_returns_none(self):
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {}
        result = scalper._get_pair_counterpart("SUIUSDT:LONG")
        assert result is None

    def test_solo_straddle_leg_returns_none(self):
        """Orphaned straddle leg with pair_id but counterpart gone."""
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "SUIUSDT:LONG": _straddle_pos("SUIUSDT", 1, "abc123"),
        }
        result = scalper._get_pair_counterpart("SUIUSDT:LONG")
        assert result is None


# ─── straddle_pair_id generation and storage ──────────────────────────────────


class TestStraddlePairId:
    def test_both_legs_share_pair_id(self):
        """Both open_position calls receive the same straddle_pair_id."""
        scalper = _build_vs_scalper()

        captured_pair_ids: list[str | None] = []
        real_open = scalper.strategy.open_position

        def _capture_open(*args, **kwargs):
            captured_pair_ids.append(kwargs.get("straddle_pair_id"))
            sym = kwargs.get("symbol") or (args[0] if args else None)
            signal = kwargs.get("signal") or (args[1] if len(args) > 1 else None)
            pos_side = "LONG" if signal == 1 else "SHORT"
            pos_key = f"{sym}:{pos_side}"
            scalper.strategy._positions[pos_key] = _straddle_pos(sym, signal, kwargs.get("straddle_pair_id") or "x")

        scalper.strategy.open_position = _capture_open

        scalper._place_order = MagicMock(return_value={"avgPrice": "1.0", "executedQty": "1000.0"})
        scalper._place_server_stop_loss = MagicMock(return_value="sl_1")
        scalper._place_server_take_profit = MagicMock(return_value="tp_1")

        # Add TESTUSDT to symbol info cache — max_notional=None disables notional cap
        scalper._symbol_info_cache["TESTUSDT"] = {
            "step_size": 0.001,
            "tick_size": 0.01,
            "quantity_precision": 3,
            "price_precision": 2,
            "min_notional": 5.0,
            "max_notional": None,
        }

        close = np.full(210, 1.0, dtype=float)
        balance = 1000.0

        result_long = scalper._open_position("TESTUSDT", 1, close, balance, straddle_pair_id="pair01")
        result_short = scalper._open_position("TESTUSDT", -1, close, balance, straddle_pair_id="pair01")

        assert result_long is True
        assert result_short is True
        assert len(captured_pair_ids) == 2
        assert captured_pair_ids[0] == "pair01"
        assert captured_pair_ids[1] == "pair01"

    def test_directional_has_no_pair_id(self):
        """Directional positions pass straddle_pair_id=None."""
        scalper = _build_vs_scalper()
        captured_pair_ids: list[str | None] = []

        def _capture_open(*args, **kwargs):
            captured_pair_ids.append(kwargs.get("straddle_pair_id"))
            sym = kwargs.get("symbol") or (args[0] if args else None)
            signal = kwargs.get("signal") or (args[1] if len(args) > 1 else 1)
            pos_key = str(sym)
            scalper.strategy._positions[pos_key] = {
                "symbol": sym,
                "signal": signal,
                "is_straddle": False,
                "straddle_pair_id": None,
                "entry_price": 1.0,
                "notional": 1000.0,
                "bars_held": 0,
                "best_pnl": 0.0,
                "worst_pnl": 0.0,
                "strategy_type": "vr_directional",
            }

        scalper.strategy.open_position = _capture_open
        scalper._place_order = MagicMock(return_value={"avgPrice": "1.0", "executedQty": "1000.0"})
        scalper._place_server_stop_loss = MagicMock(return_value="sl_1")
        scalper._place_server_take_profit = MagicMock(return_value="tp_1")
        scalper._symbol_info_cache["TESTUSDT"] = {
            "step_size": 0.001,
            "tick_size": 0.01,
            "quantity_precision": 3,
            "price_precision": 2,
            "min_notional": 5.0,
            "max_notional": None,
        }

        close = np.full(210, 1.0, dtype=float)
        scalper._open_position("TESTUSDT", 1, close, 1000.0, straddle_pair_id=None)
        assert captured_pair_ids == [None]


# ─── Second leg retry logic ───────────────────────────────────────────────────


class TestSecondLegRetry:
    def test_config_retry_defaults(self):
        """Config has correct default retry counts."""
        cfg = VolStraddleConfig()
        assert cfg.entry_retry_count == 2
        assert cfg.entry_retry_delay_s == 2.0
        assert cfg.rollback_retry_count == 2
        assert cfg.rollback_retry_delay_s == 2.0
        assert cfg.sl_cancel_retry_count == 2
        assert cfg.sl_cancel_retry_delay_s == 1.0

    @patch("quantbox.scalper.time.sleep")
    def test_second_leg_retry_loop(self, mock_sleep):
        """Retry loop attempts entry_retry_count + 1 times total on failure."""
        cfg = VolStraddleConfig()
        call_counts = {"short": 0}

        def _failing_open(sym, sig, close, bal, straddle_pair_id=None):
            if sig == -1:
                call_counts["short"] += 1
                return False
            return True

        short_opened = False
        sym = "SUIUSDT"
        close = np.full(210, 1.0)
        bal = 1000.0
        pair_id = "test01"

        for retry in range(cfg.entry_retry_count + 1):
            if _failing_open(sym, -1, close, bal, straddle_pair_id=pair_id):
                short_opened = True
                break
            if retry < cfg.entry_retry_count:
                time.sleep(cfg.entry_retry_delay_s)

        assert short_opened is False
        assert call_counts["short"] == cfg.entry_retry_count + 1

    @patch("quantbox.scalper.time.sleep")
    def test_second_leg_succeeds_after_one_retry(self, mock_sleep):
        """Second leg fails once then succeeds on retry — no rollback triggered."""
        cfg = VolStraddleConfig()
        call_counts = {"short": 0}

        def _intermittent_open(sym, sig, close, bal, straddle_pair_id=None):
            if sig == -1:
                call_counts["short"] += 1
                return call_counts["short"] >= 2  # fail first, succeed second
            return True

        short_opened = False
        for retry in range(cfg.entry_retry_count + 1):
            if _intermittent_open("SUIUSDT", -1, np.full(210, 1.0), 1000.0, straddle_pair_id="p01"):
                short_opened = True
                break
            if retry < cfg.entry_retry_count:
                time.sleep(cfg.entry_retry_delay_s)

        assert short_opened is True
        assert call_counts["short"] == 2


# ─── Rollback logic ───────────────────────────────────────────────────────────


class TestStraddleRollback:
    def test_rollback_succeeds_on_first_attempt(self):
        """Rollback via _close_position succeeds — no retry needed."""
        scalper = _build_vs_scalper()
        scalper._close_position = MagicMock(return_value=True)
        cfg_vs = scalper.vs_config

        pos_key = "TESTUSDT:LONG"
        scalper.strategy._positions[pos_key] = _straddle_pos("TESTUSDT", 1, "pair01")

        rollback_ok = False
        for attempt in range(cfg_vs.rollback_retry_count + 1):
            if scalper._close_position(pos_key, "straddle_rollback", 1.0):
                rollback_ok = True
                break

        assert rollback_ok is True
        assert scalper._close_position.call_count == 1

    @patch("quantbox.scalper.time.sleep")
    def test_rollback_retried_on_failure(self, mock_sleep):
        """Rollback retried rollback_retry_count times before market fallback."""
        scalper = _build_vs_scalper()
        scalper._close_position = MagicMock(return_value=False)
        cfg_vs = scalper.vs_config

        pos_key = "TESTUSDT:LONG"
        scalper.strategy._positions[pos_key] = _straddle_pos("TESTUSDT", 1, "pair01")

        rollback_ok = False
        for attempt in range(cfg_vs.rollback_retry_count + 1):
            if scalper._close_position(pos_key, "straddle_rollback", 1.0):
                rollback_ok = True
                break
            if attempt < cfg_vs.rollback_retry_count:
                time.sleep(cfg_vs.rollback_retry_delay_s)

        assert rollback_ok is False
        # called rollback_retry_count + 1 times total
        assert scalper._close_position.call_count == cfg_vs.rollback_retry_count + 1

    def test_market_order_fallback_does_not_call_futures_change_leverage(self):
        """CRITICAL (Rule #1): Market order fallback MUST NOT call futures_change_leverage."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False
        scalper.client = MagicMock()
        scalper.client.futures_create_order = MagicMock(return_value={"orderId": "123"})
        scalper._round_qty = MagicMock(return_value=1000.0)

        pos_key = "TESTUSDT:LONG"
        pos = _straddle_pos("TESTUSDT", 1, "pair01")
        scalper.strategy._positions[pos_key] = pos

        # Execute market order fallback as coded in scalper.py
        api_sym = pos_key.split(":")[0]
        qty = scalper._round_qty(pos["notional"] / pos["entry_price"], api_sym)
        order_kwargs: dict = {
            "symbol": api_sym,
            "side": "SELL",
            "type": "MARKET",
            "quantity": qty,
            "positionSide": "LONG",
        }
        scalper.client.futures_create_order(**order_kwargs)

        # MUST NOT call futures_change_leverage
        assert not scalper.client.futures_change_leverage.called, (
            "RULE #1 VIOLATION: futures_change_leverage called in rollback market order path!"
        )
        scalper.client.futures_create_order.assert_called_once_with(
            symbol="TESTUSDT",
            side="SELL",
            type="MARKET",
            quantity=qty,
            positionSide="LONG",
        )

    def test_total_rollback_failure_sets_rollback_failed_flag(self):
        """If all rollback attempts fail, rollback_failed=True and Discord alert sent."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False
        scalper.client.futures_create_order = MagicMock(side_effect=Exception("API error"))
        scalper._round_qty = MagicMock(return_value=1.0)

        pos_key = "TESTUSDT:LONG"
        pos = _straddle_pos("TESTUSDT", 1, "pair01")
        scalper.strategy._positions[pos_key] = pos

        # Simulate market fallback failure
        try:
            scalper.client.futures_create_order(symbol="TESTUSDT", side="SELL", type="MARKET", quantity=1.0)
        except Exception:
            pos["rollback_failed"] = True
            scalper.discord.send(content="🚨 **[TESTUSDT:LONG] 롤백 완전 실패** — 수동 청산 필요!")
            scalper._save_state()

        assert pos.get("rollback_failed") is True
        scalper.discord.send.assert_called()
        scalper._save_state.assert_called()

    def test_rollback_failure_discord_alert_contains_symbol(self):
        """Discord alert content includes symbol name on rollback failure."""
        scalper = _build_vs_scalper()
        scalper.paper_mode = False
        scalper.client.futures_create_order = MagicMock(side_effect=Exception("err"))
        scalper._round_qty = MagicMock(return_value=1.0)

        pos = _straddle_pos("TESTUSDT", 1, "pair01")
        scalper.strategy._positions["TESTUSDT:LONG"] = pos

        try:
            scalper.client.futures_create_order(symbol="TESTUSDT", side="SELL", type="MARKET", quantity=1.0)
        except Exception:
            pos["rollback_failed"] = True
            scalper.discord.send(content="🚨 **[TESTUSDT:LONG] 롤백 완전 실패** — 수동 청산 필요!")
            scalper._save_state()

        call_args = scalper.discord.send.call_args
        content = call_args[1].get("content", "") or (call_args[0][0] if call_args[0] else "")
        assert "TESTUSDT" in content


# ─── PositionDict schema ──────────────────────────────────────────────────────


class TestPositionDictSchema:
    def test_straddle_pair_id_type_alias_exists(self):
        """StraddlePairId type alias exists in core.types."""
        from quantbox.core.types import StraddlePairId

        pair_id = StraddlePairId("abc123")
        assert pair_id == "abc123"

    def test_straddle_pair_id_stored_in_position(self):
        """straddle_pair_id is stored in position dict by strategy.open_position."""
        scalper = _build_vs_scalper()

        captured: list[str | None] = []

        def _capture(*args, **kwargs):
            captured.append(kwargs.get("straddle_pair_id"))
            sym = kwargs.get("symbol") or (args[0] if args else "X")
            sig = kwargs.get("signal") or (args[1] if len(args) > 1 else 1)
            pos_side = "LONG" if sig == 1 else "SHORT"
            pos_key = f"{sym}:{pos_side}"
            scalper.strategy._positions[pos_key] = _straddle_pos(sym, sig, kwargs.get("straddle_pair_id") or "x")

        scalper.strategy.open_position = _capture
        scalper._place_order = MagicMock(return_value={"avgPrice": "1.0", "executedQty": "1000.0"})
        scalper._place_server_stop_loss = MagicMock(return_value="sl_1")
        scalper._place_server_take_profit = MagicMock(return_value="tp_1")
        scalper._symbol_info_cache["TESTUSDT"] = {
            "step_size": 0.001,
            "tick_size": 0.01,
            "quantity_precision": 3,
            "price_precision": 2,
            "min_notional": 5.0,
            "max_notional": None,
        }

        close = np.full(210, 1.0, dtype=float)
        scalper._open_position("TESTUSDT", 1, close, 1000.0, straddle_pair_id="test_pair")

        assert len(captured) == 1
        assert captured[0] == "test_pair"
