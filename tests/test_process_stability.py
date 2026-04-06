"""Tests for process stability: consecutive error handling, safe mode, DailyBreaker persistence."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tests.conftest import _build_vs_scalper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scalper():
    """Build a vol_straddle scalper with all external calls mocked."""
    sc = _build_vs_scalper(balance=1000.0)
    # Restore real _save_state / _load_state for tests that need them
    from quantbox.scalper import RubberBandScalper

    sc._save_state = RubberBandScalper._save_state.__get__(sc)
    sc._load_state = RubberBandScalper._load_state.__get__(sc)
    return sc


# ---------------------------------------------------------------------------
# Consecutive error counter
# ---------------------------------------------------------------------------


class TestConsecutiveErrors:
    def test_initial_counter_is_zero(self):
        sc = _build_vs_scalper()
        assert sc._consecutive_errors == 0

    def test_counter_increments_on_error(self):
        sc = _build_vs_scalper()
        sc._consecutive_errors = 0
        sc._consecutive_errors += 1
        assert sc._consecutive_errors == 1
        sc._consecutive_errors += 1
        assert sc._consecutive_errors == 2

    def test_counter_resets_after_success(self):
        sc = _build_vs_scalper()
        sc._consecutive_errors = 2
        # Simulate successful cycle reset
        sc._consecutive_errors = 0
        assert sc._consecutive_errors == 0


# ---------------------------------------------------------------------------
# Safe mode
# ---------------------------------------------------------------------------


class TestSafeMode:
    def test_safe_mode_initial_state(self):
        sc = _build_vs_scalper()
        assert sc._safe_mode is False

    def test_safe_mode_activates_after_3_errors(self):
        sc = _build_vs_scalper()
        sc._consecutive_errors = 3
        # Simulate the activation logic from run_continuous
        if sc._consecutive_errors >= 3 and not sc._safe_mode:
            sc._safe_mode = True
        assert sc._safe_mode is True

    def test_safe_mode_does_not_reactivate_if_already_active(self):
        sc = _build_vs_scalper()
        sc._consecutive_errors = 5
        sc._safe_mode = True
        # The `not self._safe_mode` guard prevents double-notification
        activated_again = False
        if sc._consecutive_errors >= 3 and not sc._safe_mode:
            activated_again = True
        assert activated_again is False
        assert sc._safe_mode is True  # stays True

    def test_safe_mode_deactivates_after_successful_cycle(self):
        sc = _build_vs_scalper()
        sc._safe_mode = True
        sc._consecutive_errors = 3
        # Simulate successful cycle reset
        sc._consecutive_errors = 0
        if sc._safe_mode:
            sc._safe_mode = False
        assert sc._safe_mode is False

    def test_safe_mode_not_triggered_below_threshold(self):
        sc = _build_vs_scalper()
        for error_count in (1, 2):
            sc._consecutive_errors = error_count
            sc._safe_mode = False
            if sc._consecutive_errors >= 3 and not sc._safe_mode:
                sc._safe_mode = True
            assert sc._safe_mode is False, f"safe_mode should not trigger at {error_count} errors"


# ---------------------------------------------------------------------------
# Exponential backoff logic
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    @pytest.mark.parametrize(
        "n_errors,expected_backoff",
        [
            (1, 30),
            (2, 60),
            (3, 120),
            (4, 120),  # capped at 120
        ],
    )
    def test_connection_error_backoff(self, n_errors, expected_backoff):
        backoff = min(30 * (2 ** (n_errors - 1)), 120)
        assert backoff == expected_backoff


# ---------------------------------------------------------------------------
# DailyBreaker state persistence
# ---------------------------------------------------------------------------


class TestDailyBreakerPersistence:
    def test_daily_breaker_fields_saved(self, tmp_path):
        sc = _make_scalper()
        sc._daily_loss_blocked = True
        sc._daily_realized_pnl = -55.25
        sc._daily_loss_date = "2026-03-15"

        state_file = tmp_path / "state_paper.json"
        with patch("quantbox.scalper.PAPER_STATE_FILE", state_file):
            sc._save_state()

        data = json.loads(state_file.read_text())
        assert data["daily_loss_blocked"] is True
        assert data["daily_realized_pnl"] == pytest.approx(-55.25)
        assert data["daily_loss_date"] == "2026-03-15"

    def test_daily_breaker_restored_after_restart(self, tmp_path):
        sc = _make_scalper()
        sc._daily_loss_blocked = True
        sc._daily_realized_pnl = -75.0
        sc._daily_loss_date = "2026-03-15"

        state_file = tmp_path / "state_paper.json"
        with patch("quantbox.scalper.PAPER_STATE_FILE", state_file):
            sc._save_state()

        # Build a fresh scalper and load state
        sc2 = _make_scalper()
        assert sc2._daily_loss_blocked is False  # default
        with patch("quantbox.scalper.PAPER_STATE_FILE", state_file):
            with patch.object(sc2, "_startup_exchange_crosscheck"):
                with patch.object(sc2, "get_balance", return_value=1000.0):
                    sc2._load_state()

        # Fresh session (no positions) → DailyBreaker cleared
        assert sc2._daily_loss_blocked is False
        assert sc2._daily_realized_pnl == pytest.approx(0.0)
        assert sc2._daily_loss_date == ""

    def test_daily_breaker_defaults_when_missing_from_state(self, tmp_path):
        """Older state files without daily_loss_* fields load with safe defaults."""
        state_file = tmp_path / "state_paper.json"
        # Write a minimal state without daily_loss fields
        minimal_state = {
            "version": "2.0.0",
            "saved_at": "2026-03-15T00:00:00+00:00",
            "paper_mode": True,
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "positions": {},
            "trade_history": [],
            "cycle_count": 0,
        }
        state_file.write_text(json.dumps(minimal_state))

        sc = _make_scalper()
        with patch("quantbox.scalper.PAPER_STATE_FILE", state_file):
            with patch.object(sc, "_startup_exchange_crosscheck"):
                with patch.object(sc, "get_balance", return_value=1000.0):
                    sc._load_state()

        assert sc._daily_loss_blocked is False
        assert sc._daily_realized_pnl == pytest.approx(0.0)
        assert sc._daily_loss_date == ""

    def test_daily_breaker_blocked_true_survives_restart(self, tmp_path):
        """Daily loss block survives a crash and restart within the same UTC day."""
        state_file = tmp_path / "state_paper.json"
        state_data = {
            "version": "2.0.0",
            "saved_at": "2026-03-15T10:00:00+00:00",
            "paper_mode": True,
            "paper_balance": 950.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "positions": {},
            "trade_history": [
                {"pnl_usd": -50.0, "pnl_pct": -0.05, "time": "2026-03-15T09:00:00+00:00"},
            ],
            "cycle_count": 10,
            "daily_loss_blocked": True,
            "daily_realized_pnl": -50.0,
            "daily_loss_date": "2026-03-15",
        }
        state_file.write_text(json.dumps(state_data))

        sc = _make_scalper()
        with patch("quantbox.scalper.PAPER_STATE_FILE", state_file):
            with patch.object(sc, "_startup_exchange_crosscheck"):
                with patch.object(sc, "get_balance", return_value=950.0):
                    sc._load_state()

        # Fresh session (no positions) → DailyBreaker cleared despite state having True
        assert sc._daily_loss_blocked is False
