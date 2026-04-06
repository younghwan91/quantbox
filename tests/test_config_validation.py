"""Tests for VolStraddleConfig._validate_safety_constraints.

Each test injects an invalid value via env var (VS_ prefix) and asserts
that Pydantic raises ValidationError with a message containing the
relevant field names and values.
"""

import pytest
from pydantic import ValidationError

from quantbox.strategy.vol_straddle.config import VolStraddleConfig


class TestDefaultConfigPasses:
    """Default config must pass all validators without error."""

    def test_default_config_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any env overrides that might interfere
        for key in list(__import__("os").environ):
            if key.startswith("VS_"):
                monkeypatch.delenv(key, raising=False)
        cfg = VolStraddleConfig()
        assert cfg.d_sl > 0
        assert cfg.base_leverage >= 1


def _make_config(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> VolStraddleConfig:
    """Create VolStraddleConfig with env var overrides, clearing other VS_ vars."""
    for key in list(__import__("os").environ):
        if key.startswith("VS_"):
            monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return VolStraddleConfig()


class TestDirectionalSL:
    def test_d_sl_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_sl must be positive"):
            _make_config(monkeypatch, {"VS_D_SL": "0"})

    def test_d_sl_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_sl must be positive"):
            _make_config(monkeypatch, {"VS_D_SL": "-0.01"})


class TestTPGreaterThanSL:
    def test_tp_equal_sl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_tp must be greater than d_sl"):
            _make_config(monkeypatch, {"VS_D_TP": "0.01", "VS_D_SL": "0.01"})

    def test_tp_less_than_sl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_tp must be greater than d_sl"):
            _make_config(monkeypatch, {"VS_D_TP": "0.005", "VS_D_SL": "0.01"})


class TestBaseLeverage:
    def test_base_leverage_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="base_leverage must be >= 1"):
            _make_config(monkeypatch, {"VS_BASE_LEVERAGE": "0.5"})

    def test_base_leverage_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="base_leverage must be >= 1"):
            _make_config(monkeypatch, {"VS_BASE_LEVERAGE": "-1"})


class TestMaxLeverage:
    def test_max_less_than_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="max_leverage must be >= base_leverage"):
            _make_config(
                monkeypatch,
                {
                    "VS_BASE_LEVERAGE": "30",
                    "VS_MAX_LEVERAGE": "20",
                },
            )


class TestMaxConcurrent:
    def test_max_concurrent_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="max_concurrent must be >= 1"):
            _make_config(monkeypatch, {"VS_MAX_CONCURRENT": "0"})

    def test_max_concurrent_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="max_concurrent must be >= 1"):
            _make_config(monkeypatch, {"VS_MAX_CONCURRENT": "-1"})


class TestDTrailDistance:
    def test_d_trail_distance_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_trail_distance must be positive"):
            _make_config(monkeypatch, {"VS_D_TRAIL_DISTANCE": "0"})

    def test_d_trail_distance_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_trail_distance must be positive"):
            _make_config(monkeypatch, {"VS_D_TRAIL_DISTANCE": "-0.001"})


class TestDTrailDistanceLessThanActivate:
    def test_distance_equals_activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_trail_distance must be less than d_trail_activate"):
            _make_config(
                monkeypatch,
                {
                    "VS_D_TRAIL_DISTANCE": "0.01",
                    "VS_D_TRAIL_ACTIVATE": "0.01",
                },
            )

    def test_distance_exceeds_activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="d_trail_distance must be less than d_trail_activate"):
            _make_config(
                monkeypatch,
                {
                    "VS_D_TRAIL_DISTANCE": "0.02",
                    "VS_D_TRAIL_ACTIVATE": "0.01",
                },
            )


class TestMAClusterTrail:
    def test_ma_trail_distance_exceeds_activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(
            ValidationError, match="ma_cluster_trail_distance must be less than ma_cluster_trail_activate"
        ):
            _make_config(
                monkeypatch,
                {
                    "VS_MA_CLUSTER_TRAIL_DISTANCE": "0.05",
                    "VS_MA_CLUSTER_TRAIL_ACTIVATE": "0.02",
                },
            )

    def test_ma_trail_distance_equals_activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(
            ValidationError, match="ma_cluster_trail_distance must be less than ma_cluster_trail_activate"
        ):
            _make_config(
                monkeypatch,
                {
                    "VS_MA_CLUSTER_TRAIL_DISTANCE": "0.02",
                    "VS_MA_CLUSTER_TRAIL_ACTIVATE": "0.02",
                },
            )


class TestMaxDrawdownAboveSoftDD:
    def test_max_dd_equals_soft_dd3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="max_drawdown_pct must be greater than soft_eq_dd3"):
            _make_config(
                monkeypatch,
                {
                    "VS_MAX_DRAWDOWN_PCT": "0.06",
                    "VS_SOFT_EQ_DD3": "0.06",
                },
            )

    def test_max_dd_below_soft_dd3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="max_drawdown_pct must be greater than soft_eq_dd3"):
            _make_config(
                monkeypatch,
                {
                    "VS_MAX_DRAWDOWN_PCT": "0.05",
                    "VS_SOFT_EQ_DD3": "0.06",
                },
            )


class TestDailyLossLimit:
    def test_daily_loss_limit_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="daily_loss_limit_pct must be positive"):
            _make_config(monkeypatch, {"VS_DAILY_LOSS_LIMIT_PCT": "0"})

    def test_daily_loss_limit_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="daily_loss_limit_pct must be positive"):
            _make_config(monkeypatch, {"VS_DAILY_LOSS_LIMIT_PCT": "-0.01"})


class TestMAClusterSL:
    def test_ma_cluster_sl_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="ma_cluster_sl must be positive"):
            _make_config(monkeypatch, {"VS_MA_CLUSTER_SL": "0"})

    def test_ma_cluster_sl_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="ma_cluster_sl must be positive"):
            _make_config(monkeypatch, {"VS_MA_CLUSTER_SL": "-0.01"})


class TestFeeRate:
    def test_fee_rate_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="fee_rate must be non-negative"):
            _make_config(monkeypatch, {"VS_FEE_RATE": "-0.001"})

    def test_fee_rate_zero_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fee_rate=0 is valid (no-fee scenario)."""
        cfg = _make_config(monkeypatch, {"VS_FEE_RATE": "0"})
        assert cfg.fee_rate == 0.0


class TestMultipleViolations:
    """Multiple violations should all appear in the error message."""

    def test_multiple_errors_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _make_config(
                monkeypatch,
                {
                    "VS_D_SL": "0",
                    "VS_FEE_RATE": "-0.01",
                },
            )
        msg = str(exc_info.value)
        assert "d_sl must be positive" in msg
        assert "fee_rate must be non-negative" in msg
