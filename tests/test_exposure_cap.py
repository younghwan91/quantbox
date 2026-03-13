"""Tests for ExposureGuard (session-level exposure cap).

Validates that total portfolio notional is capped at
balance × max_exposure_multiplier before new entries.
"""

import math

import pytest

from quantbox.execution.exposure import ExposureGuard


class TestCanOpenUnderLimit:
    def test_well_under_limit(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert guard.can_open(current_notional=10_000, new_notional=3_000, balance=1_000)

    def test_just_under_limit(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        # 34,999 + 1 = 35,000 == limit → not strictly over → True
        assert guard.can_open(current_notional=34_999, new_notional=1, balance=1_000)


class TestCanOpenAtLimit:
    def test_exactly_at_limit(self) -> None:
        """Total == max_allowed is NOT over limit (strict >)."""
        guard = ExposureGuard(max_multiplier=35.0)
        assert guard.can_open(current_notional=32_000, new_notional=3_000, balance=1_000)


class TestCanOpenOverLimit:
    def test_over_limit_blocked(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert not guard.can_open(current_notional=34_000, new_notional=2_000, balance=1_000)

    def test_slightly_over_limit(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert not guard.can_open(current_notional=35_000, new_notional=1, balance=1_000)


class TestBlockedCountIncrements:
    def test_counter_increments_on_block(self) -> None:
        guard = ExposureGuard(max_multiplier=10.0)
        assert guard.blocked_count == 0

        guard.can_open(current_notional=9_000, new_notional=2_000, balance=1_000)
        assert guard.blocked_count == 1

        guard.can_open(current_notional=9_000, new_notional=2_000, balance=1_000)
        assert guard.blocked_count == 2

    def test_counter_does_not_increment_on_allow(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        guard.can_open(current_notional=1_000, new_notional=1_000, balance=1_000)
        assert guard.blocked_count == 0


class TestZeroBalance:
    def test_zero_balance_blocks(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert not guard.can_open(current_notional=0, new_notional=100, balance=0)
        assert guard.blocked_count == 1


class TestNegativeBalance:
    def test_negative_balance_blocks(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert not guard.can_open(current_notional=0, new_notional=100, balance=-500)
        assert guard.blocked_count == 1


class TestUtilization:
    def test_utilization_calculation(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        # 17,500 / (1,000 × 35) = 0.5
        assert guard.utilization(current_notional=17_500, balance=1_000) == pytest.approx(0.5)

    def test_utilization_at_limit(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert guard.utilization(current_notional=35_000, balance=1_000) == pytest.approx(1.0)

    def test_utilization_over_limit(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert guard.utilization(current_notional=70_000, balance=1_000) == pytest.approx(2.0)

    def test_utilization_zero_balance_returns_inf(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        assert math.isinf(guard.utilization(current_notional=1_000, balance=0))


class TestInvalidMultiplier:
    def test_zero_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="max_multiplier must be positive"):
            ExposureGuard(max_multiplier=0)

    def test_negative_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="max_multiplier must be positive"):
            ExposureGuard(max_multiplier=-10.0)


class TestNormal10SlotsPasses:
    """Simulate 10 slots at 30x leverage with $1000 balance — all should pass."""

    def test_all_10_slots_within_limit(self) -> None:
        guard = ExposureGuard(max_multiplier=35.0)
        balance = 1_000.0
        per_slot_notional = balance * 30.0 / 10  # $3,000

        for i in range(10):
            current = per_slot_notional * i
            result = guard.can_open(
                current_notional=current,
                new_notional=per_slot_notional,
                balance=balance,
            )
            assert result, f"Slot {i + 1} should be allowed (total={current + per_slot_notional})"

        assert guard.blocked_count == 0


class TestEleventhSlotBlocked:
    """With multiplier=30.0 (no headroom), 11th entry gets blocked."""

    def test_11th_slot_blocked_exact_cap(self) -> None:
        guard = ExposureGuard(max_multiplier=30.0)
        balance = 1_000.0
        per_slot_notional = balance * 30.0 / 10  # $3,000
        ten_slots_notional = per_slot_notional * 10  # $30,000

        result = guard.can_open(
            current_notional=ten_slots_notional,
            new_notional=per_slot_notional,
            balance=balance,
        )
        assert not result, "11th slot should be blocked ($33,000 > $30,000 cap)"
        assert guard.blocked_count == 1

    def test_cw_boosted_slot_blocked(self) -> None:
        """CW-tier 2.5x boost on slot 11 exceeds 35x cap."""
        guard = ExposureGuard(max_multiplier=35.0)
        balance = 1_000.0
        per_slot_notional = balance * 30.0 / 10  # $3,000
        ten_slots_notional = per_slot_notional * 10  # $30,000
        boosted_notional = per_slot_notional * 2.5  # $7,500 (CW max conviction)

        result = guard.can_open(
            current_notional=ten_slots_notional,
            new_notional=boosted_notional,
            balance=balance,
        )
        assert not result, "CW-boosted 11th slot ($37,500 > $35,000)"
        assert guard.blocked_count == 1


class TestConfigHasMaxExposureMultiplier:
    """VolStraddleConfig must have the new field with correct default."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(__import__("os").environ):
            if key.startswith("VS_"):
                monkeypatch.delenv(key, raising=False)
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert cfg.max_exposure_multiplier == 35.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(__import__("os").environ):
            if key.startswith("VS_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("VS_MAX_EXPOSURE_MULTIPLIER", "50.0")
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        assert cfg.max_exposure_multiplier == 50.0


class TestConfigValidation:
    """max_exposure_multiplier must be >= base_leverage."""

    def test_below_base_leverage_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import ValidationError

        for key in list(__import__("os").environ):
            if key.startswith("VS_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("VS_MAX_EXPOSURE_MULTIPLIER", "10.0")
        monkeypatch.setenv("VS_BASE_LEVERAGE", "30.0")
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        with pytest.raises(ValidationError, match="max_exposure_multiplier must be >= base_leverage"):
            VolStraddleConfig()
