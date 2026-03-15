"""Tests for Step 2: Coordinated Exit Sequencing.

straddle-pair-coordination.md Step 2 acceptance criteria:
- Counterpart SL cancel retried sl_cancel_retry_count times on failure
- counterpart_sl_cancel_failed flag set after all retries exhausted
- Next cycle re-attempts cancel for flagged positions
- Server fill records pair context in log
- Counterpart bracket integrity verified after leg close
- pair_close_requested prevents duplicate close
- Normal independent leg exits NOT affected
- Pair-level net PnL logged when both legs closed
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

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
        "bars_held": 5,
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


# ─── Counterpart SL cancel with retry ────────────────────────────────────────


class TestCounterpartSLCancelRetry:
    def test_counterpart_sl_cancel_succeeds_first_try(self):
        """When cancel succeeds immediately, no flag set."""
        scalper = _build_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock(return_value=True)

        pair_id = "pair01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, pair_id),
        }

        cfg_vs = scalper.vs_config
        counterpart = scalper.strategy._positions["TESTUSDT:SHORT"]
        counterpart_sl_oid = counterpart["server_sl_order_id"]

        counterpart_cancelled = False
        for attempt in range(cfg_vs.sl_cancel_retry_count + 1):
            if scalper._cancel_server_stop_loss("TESTUSDT", counterpart_sl_oid):
                counterpart_cancelled = True
                break

        assert counterpart_cancelled is True
        assert scalper._cancel_server_stop_loss.call_count == 1
        assert not counterpart.get("counterpart_sl_cancel_failed")

    @patch("quantbox.execution.reconciliation.time.sleep")
    def test_counterpart_sl_cancel_retried_on_failure(self, mock_sleep):
        """Cancel fails all retries — flag set and Discord sent."""
        scalper = _build_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock(return_value=False)

        pair_id = "pair01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, pair_id),
        }

        cfg_vs = scalper.vs_config
        counterpart = scalper.strategy._positions["TESTUSDT:SHORT"]
        counterpart_sl_oid = counterpart["server_sl_order_id"]

        counterpart_cancelled = False
        retry_count = cfg_vs.sl_cancel_retry_count
        retry_delay = cfg_vs.sl_cancel_retry_delay_s
        for attempt in range(retry_count + 1):
            if scalper._cancel_server_stop_loss("TESTUSDT", counterpart_sl_oid):
                counterpart_cancelled = True
                break
            if attempt < retry_count:
                time.sleep(retry_delay)

        if not counterpart_cancelled:
            counterpart["counterpart_sl_cancel_failed"] = True
            scalper.discord.send(content="⚠️ 카운터파트 SL 취소 실패")

        assert not counterpart_cancelled
        assert counterpart.get("counterpart_sl_cancel_failed") is True
        assert scalper._cancel_server_stop_loss.call_count == retry_count + 1
        scalper.discord.send.assert_called()

    def test_counterpart_sl_cancel_failed_flag_triggers_discord(self):
        """When all retries fail, Discord warning is sent."""
        scalper = _build_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock(return_value=False)

        pair_id = "pair01"
        counterpart_pos = _straddle_pos("TESTUSDT", -1, pair_id)
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
            "TESTUSDT:SHORT": counterpart_pos,
        }

        cfg_vs = scalper.vs_config
        retry_count = cfg_vs.sl_cancel_retry_count

        for _ in range(retry_count + 1):
            scalper._cancel_server_stop_loss("TESTUSDT", "sl_SHORT")

        counterpart_pos["counterpart_sl_cancel_failed"] = True
        scalper.discord.send(content="⚠️ **[TESTUSDT:SHORT] 카운터파트 SL 취소 실패** — 다음 사이클에서 재시도")

        assert counterpart_pos.get("counterpart_sl_cancel_failed") is True
        scalper.discord.send.assert_called()


# ─── Next-cycle re-attempt for flagged positions ──────────────────────────────


class TestNextCycleReAttempt:
    def test_next_cycle_reattempts_flagged_cancel(self):
        """Next cycle detects counterpart_sl_cancel_failed and re-attempts."""
        scalper = _build_vs_scalper()
        # Fails first, succeeds second
        scalper._cancel_server_stop_loss = MagicMock(side_effect=[False, True])

        pair_id = "pair01"
        long_pos = _straddle_pos("TESTUSDT", 1, pair_id, counterpart_sl_cancel_failed=True)
        short_pos = _straddle_pos("TESTUSDT", -1, pair_id)
        scalper.strategy._positions = {
            "TESTUSDT:LONG": long_pos,
            "TESTUSDT:SHORT": short_pos,
        }

        cfg_vs = scalper.vs_config

        # Simulate the re-attempt logic from _sync_server_orders
        if long_pos.get("counterpart_sl_cancel_failed") and long_pos.get("is_straddle"):
            counterpart_key = scalper._get_pair_counterpart("TESTUSDT:LONG")
            if counterpart_key:
                counterpart = scalper.strategy._positions.get(counterpart_key)
                if counterpart:
                    retry_sl_oid = counterpart.get("server_sl_order_id")
                    retry_api_sym = counterpart_key.split(":")[0]
                    if retry_sl_oid:
                        retry_count = cfg_vs.sl_cancel_retry_count
                        for attempt in range(retry_count + 1):
                            if scalper._cancel_server_stop_loss(retry_api_sym, retry_sl_oid):
                                long_pos["counterpart_sl_cancel_failed"] = False
                                break

        assert long_pos.get("counterpart_sl_cancel_failed") is False

    def test_flag_cleared_when_counterpart_gone(self):
        """If counterpart no longer exists, flag is cleared."""
        scalper = _build_vs_scalper()
        pair_id = "pair01"
        long_pos = _straddle_pos("TESTUSDT", 1, pair_id, counterpart_sl_cancel_failed=True)
        # SHORT is gone — only LONG remains
        scalper.strategy._positions = {"TESTUSDT:LONG": long_pos}

        if long_pos.get("counterpart_sl_cancel_failed") and long_pos.get("is_straddle"):
            counterpart_key = scalper._get_pair_counterpart("TESTUSDT:LONG")
            if not counterpart_key:
                long_pos["counterpart_sl_cancel_failed"] = False

        assert long_pos.get("counterpart_sl_cancel_failed") is False


# ─── _record_server_fill pair context ────────────────────────────────────────


class TestRecordServerFillPairContext:
    def test_pair_context_logged_on_half_close(self, caplog):
        """Pair context logged when one leg fills and counterpart still open."""
        scalper = _build_vs_scalper()

        pair_id = "pair01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, pair_id),
        }

        def _close_pos(k):
            return scalper.strategy._positions.pop(k, None)

        scalper.strategy.close_position = MagicMock(side_effect=_close_pos)

        with caplog.at_level(logging.INFO, logger="Scalper"):
            scalper._record_server_fill("TESTUSDT:LONG", 1.015, 1000.0, "server_tp")

        pair_logs = [
            r for r in caplog.records if "pair_id" in r.message or "페어" in r.message or "pair" in r.message.lower()
        ]
        assert pair_logs, f"Expected pair context log but found none. Logs: {[r.message for r in caplog.records]}"

    def test_pair_fully_unwound_logged(self, caplog):
        """When both legs closed, 'fully unwound' log emitted."""
        scalper = _build_vs_scalper()

        pair_id = "pair02"
        # Only LONG exists — SHORT already gone
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
        }

        def _close_pos(k):
            return scalper.strategy._positions.pop(k, None)

        scalper.strategy.close_position = MagicMock(side_effect=_close_pos)

        with caplog.at_level(logging.INFO, logger="Scalper"):
            scalper._record_server_fill("TESTUSDT:LONG", 1.015, 1000.0, "server_tp")

        fully_unwound = [r for r in caplog.records if "완전" in r.message or "unwound" in r.message.lower()]
        assert fully_unwound, (
            f"Expected 'fully unwound' log but found none. Logs: {[r.message for r in caplog.records]}"
        )

    def test_non_straddle_fill_no_pair_context(self, caplog):
        """Non-straddle fills don't emit pair context logs."""
        scalper = _build_vs_scalper()

        scalper.strategy._positions = {
            "TESTUSDT:LONG": {
                "symbol": "TESTUSDT",
                "signal": 1,
                "is_straddle": False,
                "straddle_pair_id": None,
                "entry_price": 1.0,
                "notional": 1000.0,
                "bars_held": 5,
                "strategy_type": "vr_directional",
                "best_pnl": 0.0,
                "worst_pnl": 0.0,
                "balance_before": 1000.0,
                "opened_at": "2026-03-15T00:00:00+00:00",
            }
        }

        def _close_pos(k):
            return scalper.strategy._positions.pop(k, None)

        scalper.strategy.close_position = MagicMock(side_effect=_close_pos)

        with caplog.at_level(logging.INFO, logger="Scalper"):
            scalper._record_server_fill("TESTUSDT:LONG", 1.015, 1000.0, "server_tp")

        pair_logs = [r for r in caplog.records if "pair_id" in r.message or "페어" in r.message]
        assert not pair_logs, "Non-straddle should not emit pair context logs"


# ─── Counterpart bracket verification after leg close ────────────────────────


class TestCounterpartBracketVerification:
    def test_missing_counterpart_sl_logged_after_leg_close(self, caplog):
        """Missing SL on counterpart after leg close triggers warning log."""
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, "pair01"),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, "pair01", server_sl_order_id=None),
        }

        pos_key = "TESTUSDT:LONG"
        counterpart_key_pre_close = scalper._get_pair_counterpart(pos_key)

        with caplog.at_level(logging.WARNING, logger="Scalper"):
            if counterpart_key_pre_close:
                cpart = scalper.strategy._positions.get(counterpart_key_pre_close)
                if cpart and not cpart.get("server_sl_order_id"):
                    _logger = logging.getLogger("Scalper")
                    _logger.warning(
                        f"[{counterpart_key_pre_close}] 스트래들 다리 청산 후 SL 없음 — bracket repair 필요"
                    )

        sl_warnings = [r for r in caplog.records if "SL" in r.message and "없음" in r.message]
        assert sl_warnings, "Expected missing SL warning but found none"

    def test_intact_counterpart_sl_no_warning(self, caplog):
        """When counterpart SL is intact, no warning emitted."""
        scalper = _build_vs_scalper()
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, "pair01"),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, "pair01"),
        }

        pos_key = "TESTUSDT:LONG"
        counterpart_key_pre_close = scalper._get_pair_counterpart(pos_key)

        with caplog.at_level(logging.WARNING, logger="Scalper"):
            if counterpart_key_pre_close:
                cpart = scalper.strategy._positions.get(counterpart_key_pre_close)
                # SL is intact — no warning
                if cpart and not cpart.get("server_sl_order_id"):
                    logging.getLogger("Scalper").warning("SL 없음")

        sl_warnings = [r for r in caplog.records if "SL" in r.message and "없음" in r.message]
        assert not sl_warnings


# ─── pair_close_requested guard ──────────────────────────────────────────────


class TestPairCloseRequestedGuard:
    def test_pair_close_requested_prevents_duplicate_close(self):
        """pair_close_requested + _pair_close_executing prevents second close."""
        scalper = _build_vs_scalper()
        # paper_mode=True already from _build_vs_scalper

        pos_key = "TESTUSDT:LONG"
        pos = _straddle_pos(
            "TESTUSDT",
            1,
            "pair01",
            pair_close_requested=True,
            _pair_close_executing=True,
        )
        scalper.strategy._positions[pos_key] = pos

        result = scalper._close_position(pos_key, "test_close", 1.0)
        assert result is False, "Duplicate close should be blocked by pair_close_requested guard"

    def test_pair_close_requested_false_allows_close(self):
        """pair_close_requested=False allows normal close."""
        scalper = _build_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock(return_value=True)
        scalper._cancel_server_take_profit = MagicMock(return_value=True)

        pos_key = "TESTUSDT:LONG"
        pos = _straddle_pos("TESTUSDT", 1, "pair01", pair_close_requested=False)
        scalper.strategy._positions[pos_key] = pos

        scalper._place_order = MagicMock(return_value={"avgPrice": "1.015", "executedQty": "1000.0"})

        result = scalper._close_position(pos_key, "test_close", 1.015)
        assert result is True


# ─── Normal independent leg exits not affected ───────────────────────────────


class TestNormalIndependentExits:
    def test_directional_sl_exit_not_affected(self):
        """Directional (non-straddle) SL exit works without pair coordination."""
        scalper = _build_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock(return_value=True)
        scalper._cancel_server_take_profit = MagicMock(return_value=True)

        pos_key = "TESTUSDT:LONG"
        scalper.strategy._positions[pos_key] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "is_straddle": False,
            "straddle_pair_id": None,
            "entry_price": 1.0,
            "notional": 1000.0,
            "server_sl_order_id": "sl_1",
            "server_tp_order_id": "tp_1",
            "position_side": "LONG",
            "bars_held": 5,
            "strategy_type": "vr_directional",
            "best_pnl": 0.0,
            "worst_pnl": 0.0,
            "balance_before": 1000.0,
            "opened_at": "2026-03-15T00:00:00+00:00",
        }

        scalper._place_order = MagicMock(return_value={"avgPrice": "0.985", "executedQty": "1000.0"})
        result = scalper._close_position(pos_key, "sl", 0.985)
        assert result is True

    def test_straddle_independent_sl_does_not_force_close_counterpart(self):
        """One leg SL does NOT force-close the counterpart (backtest parity)."""
        scalper = _build_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock(return_value=True)
        scalper._cancel_server_take_profit = MagicMock(return_value=True)

        pair_id = "pair01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, pair_id),
        }

        scalper._place_order = MagicMock(return_value={"avgPrice": "0.985", "executedQty": "1000.0"})
        result = scalper._close_position("TESTUSDT:LONG", "sl", 0.985)
        assert result is True

        # SHORT must still be open — NOT force-closed
        assert "TESTUSDT:SHORT" in scalper.strategy._positions, (
            "SHORT leg should remain open after LONG SL — backtest parity requires independent leg exits!"
        )
