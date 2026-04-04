"""Unit tests for Donchian Channel Breakout + Volume Spike signal.

Tests check_donchian_breakout() from signals.py and the config validator
constraints for donchian_* parameters.

All tests use hand-crafted numpy arrays — no randomness, no I/O.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import check_donchian_breakout


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

# Default signal params used across most tests
_LOOKBACK = 10
_VOL_SMA = 5
_VOL_MULT = 2.0
_BREAKOUT_PCT = 0.01  # 1%
_COOLDOWN_BARS = 3

# min_needed = max(10, 5) + 1 = 11, so we need at least 12 bars (idx >= 11)
_MIN_BARS = 20  # safe margin


def _make_cooldown() -> np.ndarray:
    """Return a fresh 1-element int64 cooldown array."""
    return np.zeros(1, dtype=np.int64)


def _flat_arrays(n: int, price: float = 100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flat OHLCV arrays with normal (non-spike) volume."""
    close = np.full(n, price, dtype=np.float64)
    high = close * 1.001
    low = close * 0.999
    volume = np.full(n, 100.0, dtype=np.float64)
    return close, high, low, volume


def _call(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    cooldown_arr: np.ndarray | None = None,
    lookback: int = _LOOKBACK,
    vol_sma_period: int = _VOL_SMA,
    vol_mult: float = _VOL_MULT,
    breakout_pct: float = _BREAKOUT_PCT,
    cooldown_bars: int = _COOLDOWN_BARS,
) -> int:
    """Thin wrapper so tests don't repeat all positional args."""
    if cooldown_arr is None:
        cooldown_arr = _make_cooldown()
    return check_donchian_breakout(
        close,
        high,
        low,
        volume,
        lookback,
        vol_sma_period,
        vol_mult,
        breakout_pct,
        cooldown_arr,
        cooldown_bars,
    )


def _make_config(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> VolStraddleConfig:
    """Create VolStraddleConfig with env overrides, clearing other VS_ vars."""
    import os

    for key in list(os.environ):
        if key.startswith("VS_"):
            monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return VolStraddleConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  check_donchian_breakout — signal logic
# ══════════════════════════════════════════════════════════════════════════════


class TestLongBreakout:
    """TC-1: LONG breakout — close above channel high with volume spike → +1."""

    def test_long_breakout_returns_one(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)

        # Channel high (bars 0..n-2) is at most high[i] = 100.1
        # Set last bar close well above → 3% breakout (> 1% threshold)
        close[-1] = 103.5
        high[-1] = 104.0
        low[-1] = 103.0

        # Volume spike: last bar 5× normal (> 2× vol_sma)
        volume[-1] = 500.0

        result = _call(close, high, low, volume)
        assert result == 1


class TestShortBreakout:
    """TC-2: SHORT breakout — close below channel low with volume spike → -1."""

    def test_short_breakout_returns_minus_one(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)

        # Channel low (completed bars) is at most low[i] = 99.9
        # Set last bar close well below → 3% below channel low
        close[-1] = 96.5
        high[-1] = 97.0
        low[-1] = 96.0

        # Volume spike
        volume[-1] = 500.0

        result = _call(close, high, low, volume)
        assert result == -1


class TestNoVolumeSpike:
    """TC-3: Breakout price but no volume spike → 0."""

    def test_no_volume_spike_returns_zero(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)

        # Price breaks out above channel
        close[-1] = 103.5
        high[-1] = 104.0
        low[-1] = 103.0

        # Volume equal to SMA — not a spike (vol_mult=2.0 requires > 2× SMA)
        # volume is uniformly 100.0 → vol_sma_val = 100.0; 100 < 200 → no spike
        volume[-1] = 100.0

        result = _call(close, high, low, volume)
        assert result == 0


class TestNoBreakout:
    """TC-4: Volume spike but close within channel → 0."""

    def test_no_breakout_returns_zero(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)

        # Last bar close stays inside channel (no breakout)
        close[-1] = 100.05  # still below chan_high ≈ 100.1
        high[-1] = 100.15
        low[-1] = 99.95

        # Volume spike present
        volume[-1] = 500.0

        result = _call(close, high, low, volume)
        assert result == 0


class TestBreakoutPctThreshold:
    """TC-5: Close > chan_high but breakout_pct not exceeded → 0."""

    def test_below_breakout_pct_returns_zero(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)

        # chan_high = max(high[0..n-2]) = 100.1
        # Set close[n-1] just 0.5% above chan_high: (100.6 - 100.1)/100.1 ≈ 0.005 < 0.01
        close[-1] = 100.6
        high[-1] = 100.7
        low[-1] = 100.5

        # Volume spike
        volume[-1] = 500.0

        result = _call(close, high, low, volume, breakout_pct=0.01)
        assert result == 0

    def test_above_breakout_pct_returns_one(self):
        """Same setup but close 2% above chan_high → exceeds 1% threshold → +1."""
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)

        # chan_high ≈ 100.1; 2% above = 102.1
        close[-1] = 102.2
        high[-1] = 102.5
        low[-1] = 102.0

        volume[-1] = 500.0

        result = _call(close, high, low, volume, breakout_pct=0.01)
        assert result == 1


class TestCooldownActive:
    """TC-6: After signal fires, cooldown_arr counts down → returns 0 during cooldown."""

    def test_cooldown_suppresses_next_signals(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)
        close[-1] = 103.5
        high[-1] = 104.0
        low[-1] = 103.0
        volume[-1] = 500.0

        cd = _make_cooldown()

        # First call fires the signal
        r1 = _call(close, high, low, volume, cooldown_arr=cd, cooldown_bars=3)
        assert r1 == 1
        assert cd[0] == 3  # cooldown set

        # Subsequent calls during cooldown return 0 (even with same price/volume)
        r2 = _call(close, high, low, volume, cooldown_arr=cd, cooldown_bars=3)
        assert r2 == 0
        assert cd[0] == 2  # decremented

        r3 = _call(close, high, low, volume, cooldown_arr=cd, cooldown_bars=3)
        assert r3 == 0
        assert cd[0] == 1


class TestCooldownExpires:
    """TC-7: After cooldown_bars calls, signal can fire again."""

    def test_signal_fires_after_cooldown_expires(self):
        n = _MIN_BARS
        price = 100.0
        close, high, low, volume = _flat_arrays(n, price)
        close[-1] = 103.5
        high[-1] = 104.0
        low[-1] = 103.0
        volume[-1] = 500.0

        cd = _make_cooldown()
        cooldown_bars = 2

        # Fire signal
        r1 = _call(close, high, low, volume, cooldown_arr=cd, cooldown_bars=cooldown_bars)
        assert r1 == 1
        assert cd[0] == cooldown_bars

        # Drain cooldown (cooldown_bars calls)
        for _ in range(cooldown_bars):
            r = _call(close, high, low, volume, cooldown_arr=cd, cooldown_bars=cooldown_bars)
            assert r == 0

        assert cd[0] == 0

        # Now signal fires again
        r_after = _call(close, high, low, volume, cooldown_arr=cd, cooldown_bars=cooldown_bars)
        assert r_after == 1


class TestInsufficientData:
    """TC-8: Arrays shorter than min_needed = max(lookback, vol_sma_period) + 1 → 0."""

    def test_too_few_bars_returns_zero(self):
        # lookback=10, vol_sma=5 → min_needed=11 → need idx >= 11 → len >= 12
        # Use len=11 → idx=10 < 11 → returns 0
        n = 11
        close, high, low, volume = _flat_arrays(n)
        close[-1] = 103.5
        volume[-1] = 500.0

        result = _call(close, high, low, volume, lookback=10, vol_sma_period=5)
        assert result == 0

    def test_exact_min_needed_returns_signal(self):
        """len = min_needed + 1 = 12 → idx = 11 >= 11 → can produce signal."""
        n = 12
        close, high, low, volume = _flat_arrays(n)
        close[-1] = 103.5
        high[-1] = 104.0
        low[-1] = 103.0
        volume[-1] = 500.0

        result = _call(close, high, low, volume, lookback=10, vol_sma_period=5)
        assert result == 1


# ══════════════════════════════════════════════════════════════════════════════
#  VolStraddleConfig — donchian validator constraints
# ══════════════════════════════════════════════════════════════════════════════

# All config validator tests enable donchian to trigger the constraint block.
_DONCHIAN_ENABLED = {"VS_DONCHIAN_ENABLED": "true"}


class TestDonchianTrailDistance:
    """TC-9: donchian_trail_distance < 0.005 (rule #8) → ValidationError."""

    def test_trail_distance_too_small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_trail_distance must be >= 0.005"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_TRAIL_DISTANCE": "0.003",
                },
            )

    def test_trail_distance_at_minimum_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exactly 0.005 satisfies rule #8 (>= 0.005)."""
        cfg = _make_config(
            monkeypatch,
            {
                **_DONCHIAN_ENABLED,
                "VS_DONCHIAN_TRAIL_DISTANCE": "0.005",
                "VS_DONCHIAN_TRAIL_ACTIVATE": "0.02",  # must be > trail_distance
            },
        )
        assert cfg.donchian_trail_distance == pytest.approx(0.005)


class TestDonchianLeverage:
    """TC-10: donchian_leverage outside [1, 50] → ValidationError."""

    def test_leverage_too_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_leverage cap exceeded"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_LEVERAGE": "60",
                },
            )

    def test_leverage_below_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_leverage must be >= 1"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_LEVERAGE": "0.5",
                },
            )

    def test_leverage_boundary_50_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_config(
            monkeypatch,
            {
                **_DONCHIAN_ENABLED,
                "VS_DONCHIAN_LEVERAGE": "50",
            },
        )
        assert cfg.donchian_leverage == pytest.approx(50.0)


class TestDonchianSL:
    """TC-11: donchian_sl <= 0 → ValidationError."""

    def test_sl_zero_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_sl must be positive"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_SL": "0",
                },
            )

    def test_sl_negative_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_sl must be positive"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_SL": "-0.01",
                },
            )


class TestDonchianTPGreaterThanSL:
    """TC-12: donchian_tp <= donchian_sl → ValidationError."""

    def test_tp_less_than_sl_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_tp must be > donchian_sl"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_TP": "0.01",
                    "VS_DONCHIAN_SL": "0.015",
                },
            )

    def test_tp_equal_sl_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_tp must be > donchian_sl"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_TP": "0.015",
                    "VS_DONCHIAN_SL": "0.015",
                },
            )


class TestDonchianTrailDistanceLessThanActivate:
    """donchian_trail_distance must be < donchian_trail_activate → ValidationError."""

    def test_distance_equals_activate_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_trail_distance must be < donchian_trail_activate"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_TRAIL_DISTANCE": "0.01",
                    "VS_DONCHIAN_TRAIL_ACTIVATE": "0.01",
                },
            )

    def test_distance_exceeds_activate_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="donchian_trail_distance must be < donchian_trail_activate"):
            _make_config(
                monkeypatch,
                {
                    **_DONCHIAN_ENABLED,
                    "VS_DONCHIAN_TRAIL_DISTANCE": "0.02",
                    "VS_DONCHIAN_TRAIL_ACTIVATE": "0.015",
                },
            )


class TestDonchianDisabledSkipsValidation:
    """When donchian_enabled=False (default), constraints are not checked."""

    def test_disabled_skips_sl_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # donchian_enabled=False (default) — sl=0 should NOT raise
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "false",
                "VS_DONCHIAN_SL": "0",
            },
        )
        assert not cfg.donchian_enabled
