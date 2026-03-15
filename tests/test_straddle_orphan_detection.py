"""Tests for Step 3: Intra-Session Straddle Orphan Detection with Auto-Close.

straddle-pair-coordination.md Step 3 acceptance criteria:
- Orphaned leg detected when counterpart removed
- straddle_orphan_since_cycle set on first detection
- Auto-close after orphan_auto_close_cycles
- Auto-close goes through _close_position (SL cancel before close)
- Brackets verified during grace period
- Startup pair re-linking (old state format)
- Startup orphan marking (no counterpart found)
- Health monitor pair check
"""

from __future__ import annotations

import logging
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


def _build_live_scalper(**kwargs):
    """Build a scalper with paper_mode=False so _reconcile_positions runs."""
    scalper = _build_vs_scalper(**kwargs)
    scalper.paper_mode = False
    return scalper


# ─── Phase 3 orphan detection in _reconcile_positions ────────────────────────


class TestOrphanDetection:
    def _exchange_both(self):
        return [
            {"symbol": "TESTUSDT", "positionSide": "LONG", "positionAmt": "1000", "entryPrice": "1.0"},
            {"symbol": "TESTUSDT", "positionSide": "SHORT", "positionAmt": "-1000", "entryPrice": "1.0"},
        ]

    def _exchange_long_only(self):
        return [
            {"symbol": "TESTUSDT", "positionSide": "LONG", "positionAmt": "1000", "entryPrice": "1.0"},
        ]

    def test_pair_intact_no_orphan_flag(self):
        """Pair fully intact — straddle_orphan_since_cycle stays None."""
        scalper = _build_live_scalper()
        pair_id = "pair01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
            "TESTUSDT:SHORT": _straddle_pos("TESTUSDT", -1, pair_id),
        }
        scalper._cycle_count = 10

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_both())
        scalper._reconcile_positions()

        pos_long = scalper.strategy._positions.get("TESTUSDT:LONG")
        assert pos_long is not None
        assert pos_long.get("straddle_orphan_since_cycle") is None

    def test_orphan_detected_on_counterpart_removal(self):
        """When counterpart removed from _positions, orphan_since_cycle is set."""
        scalper = _build_live_scalper()
        pair_id = "orphan01"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
        }
        scalper._cycle_count = 5

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_long_only())
        scalper._reconcile_positions()

        pos = scalper.strategy._positions.get("TESTUSDT:LONG")
        assert pos is not None
        assert pos.get("straddle_orphan_since_cycle") == 5

    def test_orphan_since_cycle_set_only_once(self):
        """Second call to _reconcile_positions does not reset orphan_since_cycle."""
        # Use large grace so auto-close won't trigger within the test window
        cfg = VolStraddleConfig(orphan_auto_close_cycles=20)
        scalper = _build_live_scalper(vs_config=cfg)
        pair_id = "orphan02"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
        }
        scalper._cycle_count = 3

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_long_only())

        scalper._reconcile_positions()
        assert scalper.strategy._positions["TESTUSDT:LONG"]["straddle_orphan_since_cycle"] == 3

        # Advance cycle — should NOT update orphan_since_cycle
        scalper._cycle_count = 7
        scalper._reconcile_positions()
        assert scalper.strategy._positions["TESTUSDT:LONG"]["straddle_orphan_since_cycle"] == 3

    def test_orphan_flag_cleared_when_counterpart_returns(self):
        """If counterpart reappears (e.g. after adopt), orphan flag is cleared."""
        scalper = _build_live_scalper()
        pair_id = "orphan03"
        long_pos = _straddle_pos("TESTUSDT", 1, pair_id, straddle_orphan_since_cycle=2)
        short_pos = _straddle_pos("TESTUSDT", -1, pair_id)
        scalper.strategy._positions = {
            "TESTUSDT:LONG": long_pos,
            "TESTUSDT:SHORT": short_pos,
        }
        scalper._cycle_count = 5

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_both())
        scalper._reconcile_positions()

        assert "straddle_orphan_since_cycle" not in scalper.strategy._positions["TESTUSDT:LONG"]

    def test_auto_close_after_grace_period(self):
        """Orphaned leg is auto-closed after orphan_auto_close_cycles."""
        cfg = VolStraddleConfig(orphan_auto_close_cycles=4)
        scalper = _build_live_scalper(vs_config=cfg)
        pair_id = "orphan04"
        # orphan_since_cycle=0, current_cycle=4 → auto-close triggers
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id, straddle_orphan_since_cycle=0),
        }
        scalper._cycle_count = 4

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_long_only())
        scalper.client.futures_orderbook_ticker = MagicMock(return_value={"bidPrice": "1.0", "askPrice": "1.0"})
        scalper._close_position = MagicMock(return_value=True)

        scalper._reconcile_positions()

        scalper._close_position.assert_called_once_with(
            "TESTUSDT:LONG", "straddle_orphan_auto_close", pytest.approx(1.0)
        )

    def test_no_auto_close_within_grace_period(self):
        """Orphaned leg within grace period is NOT auto-closed."""
        cfg = VolStraddleConfig(orphan_auto_close_cycles=4)
        scalper = _build_live_scalper(vs_config=cfg)
        pair_id = "orphan05"
        # orphan_since_cycle=0, current_cycle=2 → still in grace
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id, straddle_orphan_since_cycle=0),
        }
        scalper._cycle_count = 2

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_long_only())
        scalper._close_position = MagicMock(return_value=True)

        scalper._reconcile_positions()

        scalper._close_position.assert_not_called()

    def test_orphaned_leg_sl_missing_logged(self, caplog):
        """During grace period, missing SL on orphaned leg triggers warning."""
        cfg = VolStraddleConfig(orphan_auto_close_cycles=4)
        scalper = _build_live_scalper(vs_config=cfg)
        pair_id = "orphan06"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos(
                "TESTUSDT",
                1,
                pair_id,
                straddle_orphan_since_cycle=0,
                server_sl_order_id=None,
            ),
        }
        scalper._cycle_count = 1  # within grace

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_long_only())
        scalper._close_position = MagicMock(return_value=True)

        with caplog.at_level(logging.WARNING, logger="Scalper"):
            scalper._reconcile_positions()

        assert any("SL 브라켓 없음" in r.message for r in caplog.records)

    def test_discord_alert_on_orphan_detection(self):
        """Discord message sent when orphan first detected."""
        scalper = _build_live_scalper()
        pair_id = "orphan07"
        scalper.strategy._positions = {
            "TESTUSDT:LONG": _straddle_pos("TESTUSDT", 1, pair_id),
        }
        scalper._cycle_count = 1

        scalper.client.futures_position_information = MagicMock(return_value=self._exchange_long_only())

        scalper._reconcile_positions()

        scalper.discord.send.assert_called()
        # Find the orphan-detection call among all discord.send calls
        orphan_alert_found = False
        for call in scalper.discord.send.call_args_list:
            content = call[1].get("content", "") if call[1] else ""
            if "고아" in content or "스트래들" in content:
                orphan_alert_found = True
                break
        assert orphan_alert_found


# ─── Startup pair re-linking (Step 3.2) ──────────────────────────────────────


def _relink_state_pos(signal: int, pair_id: str | None = None) -> dict:
    """Position dict with all required fields for _load_state."""
    pos_side = "LONG" if signal == 1 else "SHORT"
    pos = {
        "symbol": "TESTUSDT",
        "signal": signal,
        "is_straddle": True,
        "entry_price": 1.0,
        "notional": 1000.0,
        "leverage": 30,
        "move_size": 1.0,
        "position_side": pos_side,
        "strategy_type": "straddle",
        "sl": 0.015,
        "tp": 0.05,
        "tf": "15m",
        "max_hold": 192,
        "bars_held": 3,
    }
    if pair_id is not None:
        pos["straddle_pair_id"] = pair_id
    return pos


class TestStartupPairRelinking:
    def _build_state(self, positions: dict) -> dict:
        return {
            "saved_at": "2026-03-15T00:00:00+00:00",
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "trade_history": [],
            "positions": positions,
        }

    def test_old_format_positions_relinked(self):
        """Positions with is_straddle but no straddle_pair_id get re-linked."""
        scalper = _build_vs_scalper()

        state = self._build_state(
            {
                "TESTUSDT:LONG": _relink_state_pos(1),
                "TESTUSDT:SHORT": _relink_state_pos(-1),
            }
        )

        with patch("quantbox.scalper.load_scalper_state", return_value=state):
            scalper._load_state()

        long_restored = scalper.strategy._positions.get("TESTUSDT:LONG")
        short_restored = scalper.strategy._positions.get("TESTUSDT:SHORT")

        assert long_restored is not None
        assert short_restored is not None
        assert long_restored.get("straddle_pair_id") is not None
        assert short_restored.get("straddle_pair_id") is not None
        assert long_restored["straddle_pair_id"] == short_restored["straddle_pair_id"]

    def test_old_format_no_counterpart_marked_orphan(self):
        """Straddle leg with no counterpart after re-link is marked as orphaned."""
        scalper = _build_vs_scalper()

        state = self._build_state({"TESTUSDT:LONG": _relink_state_pos(1)})

        with patch("quantbox.scalper.load_scalper_state", return_value=state):
            scalper._load_state()

        long_restored = scalper.strategy._positions.get("TESTUSDT:LONG")
        assert long_restored is not None
        assert long_restored.get("straddle_orphan_since_cycle") == 0

    def test_positions_with_pair_id_not_relinked(self):
        """Positions that already have straddle_pair_id are not modified."""
        scalper = _build_vs_scalper()

        state = self._build_state(
            {
                "TESTUSDT:LONG": _relink_state_pos(1, pair_id="existing_id"),
                "TESTUSDT:SHORT": _relink_state_pos(-1, pair_id="existing_id"),
            }
        )

        with patch("quantbox.scalper.load_scalper_state", return_value=state):
            scalper._load_state()

        long_r = scalper.strategy._positions.get("TESTUSDT:LONG")
        assert long_r is not None
        assert long_r.get("straddle_pair_id") == "existing_id"


# ─── Health monitor pair check (Step 3.3) ────────────────────────────────────


class TestHealthMonitorPairCheck:
    def test_no_orphans_ok_msg(self, capsys):
        """No orphaned legs → ok_msg with pair count."""
        from scripts.health_monitor import check_straddle_pairs

        state = {
            "positions": {
                "TESTUSDT:LONG": {"is_straddle": True, "straddle_pair_id": "p1"},
                "TESTUSDT:SHORT": {"is_straddle": True, "straddle_pair_id": "p1"},
            }
        }
        check_straddle_pairs(state)
        out = capsys.readouterr().out
        assert "[OK]" in out
        assert "고아=0" in out

    def test_orphaned_leg_alerts(self, capsys):
        """Leg with straddle_orphan_since_cycle set → alert."""
        from scripts.health_monitor import check_straddle_pairs

        state = {
            "positions": {
                "TESTUSDT:LONG": {
                    "is_straddle": True,
                    "straddle_pair_id": "p1",
                    "straddle_orphan_since_cycle": 3,
                },
            }
        }
        with patch("scripts.health_monitor._notifier", None):
            check_straddle_pairs(state)

        out = capsys.readouterr()
        combined = out.out + out.err
        assert "ALERT" in combined or "고아" in combined

    def test_multiple_orphans_critical(self):
        """Multiple orphaned legs triggers critical=True."""
        from scripts.health_monitor import check_straddle_pairs

        state = {
            "positions": {
                "AAUSDT:LONG": {"is_straddle": True, "straddle_pair_id": "p1", "straddle_orphan_since_cycle": 1},
                "BBUSDT:LONG": {"is_straddle": True, "straddle_pair_id": "p2", "straddle_orphan_since_cycle": 2},
            }
        }
        with patch("scripts.health_monitor.alert") as mock_alert:
            check_straddle_pairs(state)

        mock_alert.assert_called_once()
        _args, kwargs = mock_alert.call_args
        assert kwargs.get("critical") is True

    def test_non_straddle_positions_ignored(self, capsys):
        """Non-straddle positions are not checked."""
        from scripts.health_monitor import check_straddle_pairs

        state = {
            "positions": {
                "BTCUSDT:LONG": {"is_straddle": False, "straddle_pair_id": None},
            }
        }
        check_straddle_pairs(state)
        out = capsys.readouterr().out
        assert "[OK]" in out
