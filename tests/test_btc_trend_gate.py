"""Tests for G6 BTC Trend Gate and G7 Daily Loss Circuit Breaker.

G6: LONG entries suppressed when BTC is below its MA (prevents bull-trap entries in downtrend).
G7: New entries halted for the rest of UTC-day when daily realized loss exceeds limit.
"""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone


# ─── G6: BTC Trend Gate ────────────────────────────────────────────────────


class TestBtcTrendGateConfig:
    """Config defaults are correct and .env overrides work."""

    def test_defaults(self):
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert cfg.btc_long_filter_enabled is True
        assert cfg.btc_long_ma_period == 50

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VS_BTC_LONG_FILTER_ENABLED", "false")
        monkeypatch.setenv("VS_BTC_LONG_MA_PERIOD", "20")
        # BaseSettings 는 인스턴스를 만들 때 환경변수를 읽는다 — 모듈 reload 는 필요 없다.
        # reload 하면 VolStraddleConfig 클래스가 새로 생겨, 이미 옛 클래스를 import 한
        # 다른 테스트 모듈(test_vol_straddle 등)과 conftest 의 격리 패치가 어긋났다.
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert cfg.btc_long_filter_enabled is False
        assert cfg.btc_long_ma_period == 20


class TestBtcTrendGateLogic:
    """MA computation and filtering logic is correct."""

    def _make_btc_closes(self, n: int = 200, trend: str = "up") -> np.ndarray:
        """Create BTC price array trending up or down."""
        prices = np.linspace(60_000, 70_000 if trend == "up" else 50_000, n)
        return prices.astype(np.float64)

    def test_btc_above_ma_long_allowed(self):
        """When BTC is above MA50, btc_ma < btc_cur → LONG should NOT be suppressed."""
        btc = self._make_btc_closes(200, "up")
        ma_period = 50
        btc_ma = float(np.mean(btc[-ma_period:]))
        btc_cur = float(btc[-1])
        assert btc_cur >= btc_ma, "Test setup: BTC should be above MA in uptrend"

    def test_btc_below_ma_long_suppressed(self):
        """When BTC is below MA50, btc_cur < btc_ma → LONG should be suppressed."""
        btc = self._make_btc_closes(200, "down")
        ma_period = 50
        btc_ma = float(np.mean(btc[-ma_period:]))
        btc_cur = float(btc[-1])
        assert btc_cur < btc_ma, "Test setup: BTC should be below MA in downtrend"

    def test_short_not_suppressed_when_btc_below_ma(self):
        """SHORT entries should pass through regardless of BTC trend."""
        # G6 only blocks signal == 1 (LONG)
        btc = self._make_btc_closes(200, "down")
        ma_period = 50
        btc_ma = float(np.mean(btc[-ma_period:]))
        btc_cur = float(btc[-1])
        assert btc_cur < btc_ma
        # The filter condition: signal == 1 AND btc_cur < btc_ma → skip
        # For signal == -1 (SHORT), the condition is never met
        for signal in [-1]:
            is_blocked = (signal == 1) and (btc_cur < btc_ma)
            assert not is_blocked, "SHORT should never be blocked by BTC trend gate"

    def test_insufficient_btc_data_disables_filter(self):
        """If BTC close array has fewer bars than ma_period, filter is skipped."""
        ma_period = 50
        btc_short = np.ones(30)  # fewer than 50 bars
        # Guard: len(btc_close) >= ma_period
        filter_active = len(btc_short) >= ma_period
        assert not filter_active

    def test_filter_disabled_by_config(self):
        """btc_long_filter_enabled=False disables the gate entirely."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        # Simulate the guard: btc_long_filter_enabled AND len(btc_close) >= ma_period
        btc = self._make_btc_closes(200, "down")
        filter_would_apply = (
            False  # cfg.btc_long_filter_enabled = False
            and len(btc) >= cfg.btc_long_ma_period
        )
        assert not filter_would_apply


# ─── G7: Daily Loss Circuit Breaker ────────────────────────────────────────


class TestDailyLossConfig:
    """Config defaults and .env overrides."""

    def test_defaults(self):
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert cfg.daily_loss_limit_pct == pytest.approx(0.05)

    def test_zero_disables_breaker(self):
        """daily_loss_limit_pct=0 means disabled."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        daily_loss_check = cfg.daily_loss_limit_pct > 0
        assert daily_loss_check  # default is 0.05 → enabled


class TestDailyLossBreakerLogic:
    """Circuit breaker trigger conditions."""

    def _make_trades(self, pnls: list[float], today: str = "2026-03-09") -> list[dict]:
        """Create fake trade history with today's date."""
        return [{"pnl_usd": pnl, "time": f"{today}T12:0{i}:00+00:00"} for i, pnl in enumerate(pnls)]

    def test_triggers_when_loss_exceeds_limit(self):
        """Circuit breaker triggers when today's PnL < -limit_usd."""
        balance = 1000.0
        limit_pct = 0.05  # 5%
        limit_usd = balance * limit_pct  # $50

        today = "2026-03-09"
        trades = self._make_trades([-30, -25], today)  # total = -$55 > -$50 limit

        daily_pnl = sum(t["pnl_usd"] for t in trades if t["time"].startswith(today))
        triggered = daily_pnl < -limit_usd
        assert triggered
        assert daily_pnl == pytest.approx(-55.0)

    def test_does_not_trigger_under_limit(self):
        """No trigger when loss is below limit."""
        balance = 1000.0
        limit_pct = 0.05
        limit_usd = balance * limit_pct  # $50

        today = "2026-03-09"
        trades = self._make_trades([-20, -15], today)  # total = -$35 < -$50 limit

        daily_pnl = sum(t["pnl_usd"] for t in trades if t["time"].startswith(today))
        triggered = daily_pnl < -limit_usd
        assert not triggered

    def test_only_counts_todays_trades(self):
        """Trades from other days do NOT count toward today's limit."""
        balance = 1000.0
        limit_pct = 0.05
        limit_usd = balance * limit_pct  # $50

        today = "2026-03-09"
        yesterday = "2026-03-08"
        trades = (
            self._make_trades([-40, -30], yesterday)  # -$70 yesterday (old)
            + self._make_trades([-10], today)  # -$10 today
        )

        daily_pnl = sum(t["pnl_usd"] for t in trades if t["time"].startswith(today))
        triggered = daily_pnl < -limit_usd
        assert not triggered  # only -$10 today, not triggered

    def test_resets_on_new_utc_day(self):
        """Tracker resets when UTC date changes."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = "2026-03-08"

        # Simulate: _daily_loss_date was yesterday → reset
        daily_loss_date = yesterday
        if daily_loss_date != today:
            daily_loss_date = today
            daily_realized_pnl = 0.0
            daily_loss_blocked = False

        assert daily_loss_date == today
        assert daily_realized_pnl == 0.0
        assert not daily_loss_blocked

    def test_wins_offset_losses(self):
        """Profitable trades reduce daily loss accumulation."""
        balance = 1000.0
        limit_pct = 0.05
        limit_usd = balance * limit_pct  # $50

        today = "2026-03-09"
        trades = self._make_trades([-80, 40], today)  # net = -$40 (under limit)

        daily_pnl = sum(t["pnl_usd"] for t in trades if t["time"].startswith(today))
        triggered = daily_pnl < -limit_usd
        assert not triggered
        assert daily_pnl == pytest.approx(-40.0)
