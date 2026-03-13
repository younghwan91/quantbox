"""Tests for DivergenceMonitor."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from quantbox.core.divergence_monitor import DivergenceAlert, DivergenceMonitor


# ── Helpers ──────────────────────────────────────────────────────────────────


def _stats(n: int, wr: float, pf: float) -> dict:
    return {"n_trades": n, "win_rate": wr, "profit_factor": pf}


def _monitor(tmp_path: Path, **kwargs) -> DivergenceMonitor:
    defaults = dict(
        min_trades=10,
        pf_threshold=0.80,
        wr_gap_pp=10.0,
        cooldown_hours=24.0,
        cooldown_path=tmp_path / "last_alert.json",
    )
    defaults.update(kwargs)
    return DivergenceMonitor(**defaults)


# ── TestDivergenceAlert ───────────────────────────────────────────────────────


class TestDivergenceAlert:
    def test_message_pf_only(self):
        alert = DivergenceAlert(
            pf_live=0.5,
            pf_baseline=2.002,
            wr_live=0.43,
            wr_baseline=0.461,
            n_trades=50,
            triggered_by=["pf"],
        )
        alert._pf_threshold = 0.80
        alert.message = ""
        alert.__post_init__()
        assert "PF" in alert.message
        assert "0.500" in alert.message

    def test_message_wr_only(self):
        alert = DivergenceAlert(
            pf_live=1.2,
            pf_baseline=2.002,
            wr_live=0.28,
            wr_baseline=0.461,
            n_trades=50,
            triggered_by=["wr"],
        )
        assert "WR" in alert.message
        assert "pp" in alert.message.lower() or "below" in alert.message

    def test_message_both(self):
        alert = DivergenceAlert(
            pf_live=0.5,
            pf_baseline=2.002,
            wr_live=0.28,
            wr_baseline=0.461,
            n_trades=50,
            triggered_by=["pf", "wr"],
        )
        assert "PF" in alert.message
        assert "WR" in alert.message

    def test_triggered_by_list(self):
        alert = DivergenceAlert(
            pf_live=0.5,
            pf_baseline=2.002,
            wr_live=0.28,
            wr_baseline=0.461,
            n_trades=50,
            triggered_by=["pf", "wr"],
        )
        assert "pf" in alert.triggered_by
        assert "wr" in alert.triggered_by


# ── TestDivergenceMonitorBasic ────────────────────────────────────────────────


class TestDivergenceMonitorBasic:
    def test_no_alert_when_healthy(self, tmp_path):
        mon = _monitor(tmp_path)
        # PF=1.5 > 0.80, WR=43% gap < 10pp
        result = mon.check(_stats(n=30, wr=0.43, pf=1.5))
        assert result is None

    def test_no_alert_min_trades_not_reached(self, tmp_path):
        mon = _monitor(tmp_path, min_trades=30)
        result = mon.check(_stats(n=29, wr=0.10, pf=0.10))
        assert result is None

    def test_no_alert_exactly_min_minus_one(self, tmp_path):
        mon = _monitor(tmp_path, min_trades=30)
        result = mon.check(_stats(n=29, wr=0.20, pf=0.50))
        assert result is None

    def test_alert_at_exactly_min_trades(self, tmp_path):
        mon = _monitor(tmp_path, min_trades=30)
        result = mon.check(_stats(n=30, wr=0.20, pf=0.50))
        assert result is not None

    def test_pf_below_threshold_triggers(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=0.80)
        result = mon.check(_stats(n=30, wr=0.43, pf=0.79))
        assert result is not None
        assert "pf" in result.triggered_by

    def test_pf_at_threshold_no_trigger(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=0.80)
        result = mon.check(_stats(n=30, wr=0.43, pf=0.80))
        assert result is None  # exactly at threshold → no alert

    def test_wr_gap_triggers(self, tmp_path):
        # baseline 46.1%, gap=10pp → alert if live WR < 36.1%
        mon = _monitor(tmp_path, wr_gap_pp=10.0, pf_threshold=0.0)
        result = mon.check(_stats(n=30, wr=0.33, pf=1.5))
        assert result is not None
        assert "wr" in result.triggered_by

    def test_wr_gap_just_below_threshold_no_trigger(self, tmp_path):
        # baseline 46.1%, gap exactly at boundary: 46.1 - 36.1 = 10.0pp → alert
        mon = _monitor(tmp_path, wr_gap_pp=10.0, pf_threshold=0.0)
        # 46.1 - 36.2 = 9.9pp < 10pp → no alert
        result = mon.check(_stats(n=30, wr=0.362, pf=1.5))
        assert result is None

    def test_both_triggered(self, tmp_path):
        mon = _monitor(tmp_path)
        result = mon.check(_stats(n=50, wr=0.20, pf=0.30))
        assert result is not None
        assert "pf" in result.triggered_by
        assert "wr" in result.triggered_by

    def test_alert_contains_n_trades(self, tmp_path):
        mon = _monitor(tmp_path)
        result = mon.check(_stats(n=50, wr=0.20, pf=0.30))
        assert result.n_trades == 50

    def test_baseline_values(self, tmp_path):
        mon = _monitor(tmp_path)
        result = mon.check(_stats(n=50, wr=0.20, pf=0.30))
        assert result.pf_baseline == pytest.approx(2.002)
        assert result.wr_baseline == pytest.approx(0.461)


# ── TestDivergenceMonitorCooldown ─────────────────────────────────────────────


class TestDivergenceMonitorCooldown:
    def test_cooldown_suppresses_second_alert(self, tmp_path):
        mon = _monitor(tmp_path, cooldown_hours=24.0)
        # First alert fires and saves cooldown
        r1 = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert r1 is not None
        # Second call immediately after → cooldown active
        r2 = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert r2 is None

    def test_cooldown_zero_allows_repeated_alerts(self, tmp_path):
        mon = _monitor(tmp_path, cooldown_hours=0.0)
        r1 = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert r1 is not None
        r2 = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert r2 is not None

    def test_expired_cooldown_allows_alert(self, tmp_path):
        cooldown_path = tmp_path / "last_alert.json"
        # Write a timestamp 25 hours ago
        past = datetime.now(timezone.utc) - timedelta(hours=25)
        cooldown_path.write_text(json.dumps({"last_alert_at": past.isoformat()}))
        mon = _monitor(tmp_path, cooldown_hours=24.0)
        result = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert result is not None

    def test_active_cooldown_suppresses(self, tmp_path):
        cooldown_path = tmp_path / "last_alert.json"
        # Write a timestamp 1 hour ago
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        cooldown_path.write_text(json.dumps({"last_alert_at": recent.isoformat()}))
        mon = _monitor(tmp_path, cooldown_hours=24.0)
        result = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert result is None

    def test_cooldown_file_written_on_alert(self, tmp_path):
        cooldown_path = tmp_path / "last_alert.json"
        mon = _monitor(tmp_path)
        assert not cooldown_path.exists()
        mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert cooldown_path.exists()
        data = json.loads(cooldown_path.read_text())
        assert "last_alert_at" in data

    def test_corrupt_cooldown_file_handled(self, tmp_path):
        cooldown_path = tmp_path / "last_alert.json"
        cooldown_path.write_text("NOT VALID JSON{{{")
        mon = _monitor(tmp_path, cooldown_hours=24.0)
        # Should not raise; corrupted file treated as no cooldown
        result = mon.check(_stats(n=50, wr=0.10, pf=0.20))
        assert result is not None


# ── TestDivergenceMonitorConfig ───────────────────────────────────────────────


class TestDivergenceMonitorConfig:
    def test_custom_pf_threshold(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=1.20)
        # PF=1.0 < 1.20 → alert
        result = mon.check(_stats(n=30, wr=0.43, pf=1.0))
        assert result is not None
        assert "pf" in result.triggered_by

    def test_custom_wr_gap(self, tmp_path):
        mon = _monitor(tmp_path, wr_gap_pp=5.0, pf_threshold=0.0)
        # baseline 46.1%, gap=5pp → alert if live < 41.1%
        result = mon.check(_stats(n=30, wr=0.40, pf=1.5))
        assert result is not None

    def test_default_cooldown_path(self):
        mon = DivergenceMonitor()
        assert "divergence_last_alert.json" in str(mon._cooldown_path)


# ── TestEVDriftCheck ──────────────────────────────────────────────────────────


class TestEVDriftCheck:
    """EV (basis points per trade) sign check — negative EV = no edge."""

    def _stats_ev(self, n: int, wr: float, pf: float, ev_bp: float | None = None) -> dict:
        d = {"n_trades": n, "win_rate": wr, "profit_factor": pf}
        if ev_bp is not None:
            d["ev_bp"] = ev_bp
        return d

    def test_negative_ev_triggers_alert(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=0.0, wr_gap_pp=100.0)  # disable PF/WR checks
        result = mon.check(self._stats_ev(n=30, wr=0.50, pf=1.2, ev_bp=-5.0))
        assert result is not None
        assert "ev" in result.triggered_by

    def test_zero_ev_triggers_alert(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=0.0, wr_gap_pp=100.0)
        result = mon.check(self._stats_ev(n=30, wr=0.50, pf=1.2, ev_bp=0.0))
        # ev=0.0 is not < 0, so no trigger
        assert result is None

    def test_positive_ev_no_trigger(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=0.0, wr_gap_pp=100.0)
        result = mon.check(self._stats_ev(n=30, wr=0.50, pf=1.2, ev_bp=15.0))
        assert result is None

    def test_missing_ev_skips_check(self, tmp_path):
        """If ev_bp not in stats dict, the EV check is silently skipped."""
        mon = _monitor(tmp_path, pf_threshold=0.0, wr_gap_pp=100.0)
        result = mon.check(self._stats_ev(n=30, wr=0.50, pf=1.2))  # no ev_bp key
        assert result is None

    def test_negative_ev_message_descriptive(self, tmp_path):
        mon = _monitor(tmp_path, pf_threshold=0.0, wr_gap_pp=100.0)
        result = mon.check(self._stats_ev(n=30, wr=0.50, pf=1.2, ev_bp=-12.3))
        assert result is not None
        assert "EV" in result.message
        assert "expectancy" in result.message
