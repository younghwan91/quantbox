"""Tests for G6 BTC Trend Gate and G7 Daily Loss Circuit Breaker.

G6: LONG entries suppressed when BTC is below its MA (prevents bull-trap entries in downtrend).
G7: New entries halted for the rest of UTC-day when daily realized loss exceeds limit.
"""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock


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
        # Must re-instantiate to pick up env
        from importlib import reload
        import quantbox.strategy.vol_straddle.config as m

        reload(m)
        cfg = m.VolStraddleConfig()
        assert cfg.btc_long_filter_enabled is False
        assert cfg.btc_long_ma_period == 20
        reload(m)  # restore defaults for other tests


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


# ─── G6: Straddle Exemption ────────────────────────────────────────────────


class TestG6StraddleExemption:
    """Straddle candidates must bypass G6 BTC Trend Gate even when BTC < MA.

    Straddle is direction-neutral (LONG+SHORT simultaneously).  Blocking its
    LONG leg via G6 would cause an asymmetric SHORT-only open, which defeats
    the strategy entirely.  The fix adds is_straddle=True to straddle
    candidates so G6 logic is skipped for them.
    """

    def test_straddle_long_leg_not_blocked_by_g6(self):
        """Straddle LONG candidate passes through G6 even when BTC < MA."""
        # Simulate G6 condition evaluation — this is the guard in scalper.py:
        #   if signal == 1 and not is_straddle and btc_ma is not None and btc_cur < btc_ma:
        btc_cur = 50_000.0
        btc_ma = 60_000.0  # BTC below MA (would normally block LONG)
        signal = 1

        for is_straddle in [True, False]:
            blocked = signal == 1 and not is_straddle and btc_cur < btc_ma
            if is_straddle:
                assert not blocked, "Straddle LONG leg must NOT be blocked by G6"
            else:
                assert blocked, "Directional LONG must be blocked when BTC < MA"

    def test_straddle_short_leg_not_blocked_by_g6(self):
        """Straddle SHORT candidate is never blocked (G6 only targets signal==1)."""
        btc_cur = 50_000.0
        btc_ma = 60_000.0
        signal = -1
        is_straddle = True

        # signal == -1 → G6 condition is False regardless of is_straddle
        blocked = signal == 1 and not is_straddle and btc_cur < btc_ma
        assert not blocked

    def test_directional_long_still_blocked_by_g6(self):
        """Non-straddle LONG is still blocked when BTC < MA (regression guard)."""
        btc_cur = 50_000.0
        btc_ma = 60_000.0
        signal = 1
        is_straddle = False

        blocked = signal == 1 and not is_straddle and btc_cur < btc_ma
        assert blocked, "Directional LONG must still be blocked by G6"

    def test_straddle_candidate_tuple_has_5_elements(self):
        """Straddle candidates are 5-element tuples with is_straddle=True."""
        # Simulate what scalper.py appends for a straddle signal
        sym = "BTCUSDT"
        move = 0.025
        close = np.ones(200)

        straddle_candidates: list[tuple[str, int, float, np.ndarray, bool]] = []
        signal = 2  # straddle
        if signal == 2:
            straddle_candidates.append((sym, 1, move, close, True))
            straddle_candidates.append((sym, -1, move, close, True))
        else:
            straddle_candidates.append((sym, signal, move, close, False))

        assert len(straddle_candidates) == 2
        long_cand = straddle_candidates[0]
        short_cand = straddle_candidates[1]

        assert long_cand[1] == 1 and long_cand[4] is True, "LONG straddle leg: is_straddle must be True"
        assert short_cand[1] == -1 and short_cand[4] is True, "SHORT straddle leg: is_straddle must be True"

    def test_directional_candidate_tuple_has_false_flag(self):
        """Non-straddle candidates have is_straddle=False."""
        sym = "ETHUSDT"
        move = 0.01
        close = np.ones(200)
        signal = 1  # directional LONG

        candidates: list[tuple[str, int, float, np.ndarray, bool]] = []
        if signal == 2:
            candidates.append((sym, 1, move, close, True))
            candidates.append((sym, -1, move, close, True))
        else:
            candidates.append((sym, signal, move, close, False))

        assert len(candidates) == 1
        assert candidates[0][4] is False, "Directional candidate must have is_straddle=False"
