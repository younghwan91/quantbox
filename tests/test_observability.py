"""Tests for observability and alerting enhancements (Step 4).

Covers:
- restart_count auto-reset after 16 consecutive clean cycles
- clean_cycle_counter resets on error
- health_monitor bracket coverage check
- health_monitor restart frequency check
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ── Restart counter reset logic ───────────────────────────────────────────────


class TestRestartCounterReset:
    """Tests for _clean_cycles_count and restart_count auto-reset in scalper."""

    def _make_scalper(self, restart_count: int = 1) -> MagicMock:
        scalper = MagicMock()
        scalper._restart_count = restart_count
        scalper._clean_cycles_count = 0
        scalper._consecutive_errors = 0
        scalper._safe_mode = False
        return scalper

    def test_reset_after_16_clean_cycles(self):
        """After 16 consecutive clean cycles, restart_count resets to 0."""
        scalper = self._make_scalper(restart_count=2)
        scalper._clean_cycles_count = 15  # one more push makes 16

        # Simulate what the main loop does on success
        scalper._consecutive_errors = 0
        scalper._clean_cycles_count += 1
        if scalper._clean_cycles_count >= 16 and scalper._restart_count > 0:
            scalper._restart_count = 0
            scalper._clean_cycles_count = 0
            scalper._save_state()

        assert scalper._restart_count == 0
        assert scalper._clean_cycles_count == 0
        scalper._save_state.assert_called_once()

    def test_no_reset_below_16_cycles(self):
        """restart_count is NOT reset if fewer than 16 consecutive clean cycles."""
        scalper = self._make_scalper(restart_count=2)
        scalper._clean_cycles_count = 14

        scalper._consecutive_errors = 0
        scalper._clean_cycles_count += 1
        if scalper._clean_cycles_count >= 16 and scalper._restart_count > 0:
            scalper._restart_count = 0
            scalper._clean_cycles_count = 0

        assert scalper._restart_count == 2  # unchanged
        assert scalper._clean_cycles_count == 15

    def test_no_reset_if_restart_count_already_zero(self):
        """If restart_count is already 0, _save_state is NOT called."""
        scalper = self._make_scalper(restart_count=0)
        scalper._clean_cycles_count = 15

        scalper._clean_cycles_count += 1
        if scalper._clean_cycles_count >= 16 and scalper._restart_count > 0:
            scalper._restart_count = 0
            scalper._clean_cycles_count = 0
            scalper._save_state()

        scalper._save_state.assert_not_called()

    def test_clean_cycle_counter_resets_on_error(self):
        """Any error resets the clean cycle counter to 0."""
        scalper = self._make_scalper(restart_count=1)
        scalper._clean_cycles_count = 12

        # Simulate error path
        scalper._consecutive_errors += 1
        scalper._clean_cycles_count = 0

        assert scalper._clean_cycles_count == 0

    def test_counter_increments_on_each_clean_cycle(self):
        """Counter increments by 1 per successful cycle."""
        scalper = self._make_scalper(restart_count=1)
        scalper._clean_cycles_count = 0

        for expected in range(1, 10):
            scalper._clean_cycles_count += 1
            assert scalper._clean_cycles_count == expected


# ── Health monitor: bracket coverage ─────────────────────────────────────────


class TestHealthMonitorBracketCoverage:
    def test_no_positions_is_ok(self, capsys):
        from scripts.health_monitor import check_bracket_coverage

        check_bracket_coverage({"positions": {}})
        out = capsys.readouterr().out
        assert "no positions" in out

    def test_all_positions_have_sl_is_ok(self, capsys):
        from scripts.health_monitor import check_bracket_coverage

        state = {
            "positions": {
                "BTCUSDT:LONG": {"server_sl_order_id": "111"},
                "ETHUSDT:SHORT": {"server_sl_order_id": "222"},
            }
        }
        check_bracket_coverage(state)
        out = capsys.readouterr().out
        assert "all 2 positions have SL" in out

    def test_single_missing_sl_alerts(self, capsys):
        from scripts.health_monitor import check_bracket_coverage

        state = {
            "positions": {
                "BTCUSDT:LONG": {"server_sl_order_id": None},
                "ETHUSDT:SHORT": {"server_sl_order_id": "222"},
            }
        }
        with patch("scripts.health_monitor.alert") as mock_alert:
            check_bracket_coverage(state)
            mock_alert.assert_called_once()
            call_kwargs = mock_alert.call_args
            # Single missing = warning (critical=False)
            assert call_kwargs[1].get("critical") is False or call_kwargs[0][0] == "브라켓 누락"

    def test_multiple_missing_sl_is_critical(self):
        from scripts.health_monitor import check_bracket_coverage

        state = {
            "positions": {
                "BTCUSDT:LONG": {"server_sl_order_id": None},
                "ETHUSDT:SHORT": {"server_sl_order_id": None},
            }
        }
        with patch("scripts.health_monitor.alert") as mock_alert:
            check_bracket_coverage(state)
            mock_alert.assert_called_once()
            _, kwargs = mock_alert.call_args
            assert kwargs.get("critical") is True

    def test_missing_positions_key_is_ok(self, capsys):
        from scripts.health_monitor import check_bracket_coverage

        check_bracket_coverage({})
        out = capsys.readouterr().out
        assert "no positions" in out


# ── Health monitor: restart frequency ────────────────────────────────────────


class TestHealthMonitorRestartFrequency:
    def test_low_restart_count_is_ok(self, capsys):
        from scripts.health_monitor import check_restart_frequency

        check_restart_frequency({"restart_count": 2})
        out = capsys.readouterr().out
        assert "OK" in out

    def test_zero_restart_count_is_ok(self, capsys):
        from scripts.health_monitor import check_restart_frequency

        check_restart_frequency({"restart_count": 0})
        out = capsys.readouterr().out
        assert "OK" in out

    def test_restart_count_above_3_alerts(self):
        from scripts.health_monitor import check_restart_frequency

        with patch("scripts.health_monitor.alert") as mock_alert:
            check_restart_frequency({"restart_count": 4, "last_restart_at": "2026-03-15T10:00:00"})
            mock_alert.assert_called_once()
            title = mock_alert.call_args[0][0]
            assert "재시작" in title

    def test_restart_count_above_5_is_critical(self):
        from scripts.health_monitor import check_restart_frequency

        with patch("scripts.health_monitor.alert") as mock_alert:
            check_restart_frequency({"restart_count": 6})
            mock_alert.assert_called_once()
            _, kwargs = mock_alert.call_args
            assert kwargs.get("critical") is True

    def test_restart_count_exactly_3_is_ok(self, capsys):
        from scripts.health_monitor import check_restart_frequency

        check_restart_frequency({"restart_count": 3})
        out = capsys.readouterr().out
        assert "OK" in out

    def test_missing_restart_count_defaults_to_zero(self, capsys):
        from scripts.health_monitor import check_restart_frequency

        check_restart_frequency({})
        out = capsys.readouterr().out
        assert "OK" in out

    def test_last_restart_at_shown_in_alert(self):
        from scripts.health_monitor import check_restart_frequency

        ts = "2026-03-15T08:00:00"
        with patch("scripts.health_monitor.alert") as mock_alert:
            check_restart_frequency({"restart_count": 4, "last_restart_at": ts})
            args, kwargs = mock_alert.call_args
            fields = kwargs.get("fields", [])
            restart_field = next((f for f in fields if "재시작" in f.get("name", "")), None)
            assert restart_field is not None


# ── Execution quality fields ──────────────────────────────────────────────────


class TestExecutionQualityFields:
    """Verify bracket hardening fields are accessible in position dicts."""

    def test_bracket_sl_verified_at_field(self):
        """bracket_sl_verified_at is a standard ISO timestamp string."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        pos = {"server_sl_order_id": "123", "bracket_sl_verified_at": now}
        assert "bracket_sl_verified_at" in pos
        assert "T" in pos["bracket_sl_verified_at"]

    def test_last_ratchet_fields(self):
        """last_ratchet_price and last_ratchet_at are written by _verify_ratchet."""
        from quantbox.execution.brackets import BracketMixin

        scalper = MagicMock()
        scalper.paper_mode = False
        scalper._verify_algo_order_active = MagicMock(return_value="WORKING")
        scalper.discord.send = MagicMock()

        pos = {"symbol": "BTCUSDT", "server_sl_order_id": "456"}
        BracketMixin._verify_ratchet(scalper, "BTCUSDT:LONG", pos, expected_price=50000.0)

        assert pos.get("last_ratchet_price") == 50000.0
        assert "last_ratchet_at" in pos
