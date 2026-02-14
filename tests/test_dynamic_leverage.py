"""
Tests for dynamic leverage feature.

Covers:
- calculate_dynamic_leverage: formula correctness, edge cases, clamping
- estimate_cost_rate: cost computation
- Integration with calculate_slot_capital: leverage type float
- Backtest _enter_slot: dynamic leverage applied correctly
- Parity: live and backtest use same formula
"""

import pytest
import numpy as np

from quantbox.core.trader_slot import (
    calculate_dynamic_leverage,
    estimate_cost_rate,
    calculate_slot_capital,
    calculate_margin_from_notional,
    TraderSlotState,
)


# ============================================================
# estimate_cost_rate
# ============================================================


class TestEstimateCostRate:
    """Tests for estimate_cost_rate helper."""

    def test_basic_cost_rate(self):
        """Round-trip cost = 2*(fee+slip) + 2*funding*bars/bars_per_funding."""
        rate = estimate_cost_rate(
            trading_fee=0.0005,
            slippage=0.001,
            funding_rate_8h=0.0001,
            expected_holding_bars=40,
            bar_interval_minutes=15,
        )
        # bars_per_funding = 8*60/15 = 32
        # round_trip = 2*(0.0005+0.001) = 0.003
        # funding = 2*0.0001*40/32 = 0.00025
        expected = 0.003 + 0.00025
        assert abs(rate - expected) < 1e-10

    def test_zero_funding(self):
        """No funding → only round-trip costs."""
        rate = estimate_cost_rate(
            trading_fee=0.0005,
            slippage=0.001,
            funding_rate_8h=0.0,
            expected_holding_bars=40,
        )
        assert abs(rate - 0.003) < 1e-10

    def test_zero_holding(self):
        """Zero holding → no funding, only round-trip."""
        rate = estimate_cost_rate(
            trading_fee=0.0005,
            slippage=0.001,
            funding_rate_8h=0.0001,
            expected_holding_bars=0,
        )
        assert abs(rate - 0.003) < 1e-10

    def test_different_bar_interval(self):
        """5m bars → bars_per_funding = 96."""
        rate = estimate_cost_rate(
            trading_fee=0.0005,
            slippage=0.001,
            funding_rate_8h=0.0001,
            expected_holding_bars=48,
            bar_interval_minutes=5,
        )
        # bars_per_funding = 8*60/5 = 96
        # funding = 2*0.0001*48/96 = 0.0001
        expected = 0.003 + 0.0001
        assert abs(rate - expected) < 1e-10


# ============================================================
# calculate_dynamic_leverage
# ============================================================


class TestCalculateDynamicLeverage:
    """Tests for calculate_dynamic_leverage."""

    def test_formula_correctness(self):
        """Verify leverage = target_r / (pct * risk_per_unit)."""
        # With stop_z=3.5, anchor_std=0.01, HR=1.0, cost_rate=0.003
        # risk_per_unit = 3.5 * 0.01 / 2.0 + 0.003 = 0.0175 + 0.003 = 0.0205
        # leverage = 0.01 / (0.2 * 0.0205) = 0.01 / 0.0041 = 2.439..
        lev = calculate_dynamic_leverage(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=3.5,
            risk_std=0.01,
            hedge_ratio=1.0,
            cost_rate=0.003,
            max_leverage=10.0,
        )
        expected = 0.01 / (0.2 * 0.0205)
        assert abs(lev - expected) < 1e-6

    def test_r_equals_target(self):
        """End-to-end: R computed with returned leverage should equal target."""
        target_r = 0.01
        pct = 0.2
        stop_z = 3.5
        anchor_std = 0.008
        hr = 1.2
        cost_rate = 0.00325
        balance = 10000.0

        lev = calculate_dynamic_leverage(
            target_r_pct=target_r,
            position_size_pct=pct,
            stop_loss_zscore=stop_z,
            risk_std=anchor_std,
            hedge_ratio=hr,
            cost_rate=cost_rate,
            max_leverage=20.0,
        )

        notional = balance * pct * lev
        stop_loss = stop_z * anchor_std * notional / (1 + hr)
        total_cost = cost_rate * notional
        R = stop_loss + total_cost
        R_pct = R / balance

        assert abs(R_pct - target_r) < 1e-8

    def test_clamp_max_leverage(self):
        """Leverage should not exceed max_leverage."""
        # Very small risk_std → would compute huge leverage
        lev = calculate_dynamic_leverage(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=3.5,
            risk_std=0.0001,  # very small
            hedge_ratio=1.0,
            cost_rate=0.00001,
            max_leverage=10.0,
        )
        assert lev == 10.0

    def test_clamp_min_leverage(self):
        """Leverage should not go below 1.0."""
        # Very large risk_std → would compute leverage < 1
        lev = calculate_dynamic_leverage(
            target_r_pct=0.001,  # very small target
            position_size_pct=0.5,
            stop_loss_zscore=3.5,
            risk_std=0.1,  # huge std
            hedge_ratio=1.0,
            cost_rate=0.01,
            max_leverage=10.0,
        )
        assert lev == 1.0

    def test_zero_risk_std_returns_max(self):
        """If risk_per_unit ≤ 0 (degenerate), return max_leverage."""
        lev = calculate_dynamic_leverage(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=3.5,
            risk_std=0.0,
            hedge_ratio=1.0,
            cost_rate=0.0,
            max_leverage=5.0,
        )
        assert lev == 5.0

    def test_higher_target_r_gives_higher_leverage(self):
        """More risk tolerance → higher leverage."""
        kwargs = dict(
            position_size_pct=0.2,
            stop_loss_zscore=3.5,
            risk_std=0.01,
            hedge_ratio=1.0,
            cost_rate=0.003,
            max_leverage=20.0,
        )
        lev_1pct = calculate_dynamic_leverage(target_r_pct=0.01, **kwargs)
        lev_2pct = calculate_dynamic_leverage(target_r_pct=0.02, **kwargs)
        assert lev_2pct > lev_1pct

    def test_higher_hr_gives_higher_leverage(self):
        """Higher HR → lower risk per unit → higher leverage."""
        kwargs = dict(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=3.5,
            risk_std=0.01,
            cost_rate=0.003,
            max_leverage=20.0,
        )
        lev_hr05 = calculate_dynamic_leverage(hedge_ratio=0.5, **kwargs)
        lev_hr15 = calculate_dynamic_leverage(hedge_ratio=1.5, **kwargs)
        assert lev_hr15 > lev_hr05  # HR=1.5 → lower risk per unit

    def test_higher_anchor_std_gives_lower_leverage(self):
        """Higher spread volatility → more risk → lower leverage."""
        kwargs = dict(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=3.5,
            hedge_ratio=1.0,
            cost_rate=0.003,
            max_leverage=20.0,
        )
        lev_low = calculate_dynamic_leverage(risk_std=0.005, **kwargs)
        lev_high = calculate_dynamic_leverage(risk_std=0.02, **kwargs)
        assert lev_low > lev_high

    def test_invalid_target_r_raises(self):
        """target_r_pct <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="target_r_pct"):
            calculate_dynamic_leverage(
                target_r_pct=0.0,
                position_size_pct=0.2,
                stop_loss_zscore=3.5,
                risk_std=0.01,
                hedge_ratio=1.0,
                cost_rate=0.003,
            )

    def test_invalid_pct_raises(self):
        """position_size_pct <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="position_size_pct"):
            calculate_dynamic_leverage(
                target_r_pct=0.01,
                position_size_pct=0.0,
                stop_loss_zscore=3.5,
                risk_std=0.01,
                hedge_ratio=1.0,
                cost_rate=0.003,
            )

    def test_invalid_max_leverage_raises(self):
        """max_leverage < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_leverage"):
            calculate_dynamic_leverage(
                target_r_pct=0.01,
                position_size_pct=0.2,
                stop_loss_zscore=3.5,
                risk_std=0.01,
                hedge_ratio=1.0,
                cost_rate=0.003,
                max_leverage=0.5,
            )


# ============================================================
# Integration: dynamic leverage with slot capital
# ============================================================


class TestDynamicLeverageIntegration:
    """Integration tests: dynamic leverage flows through capital allocation."""

    def test_slot_capital_with_float_leverage(self):
        """calculate_slot_capital should accept float leverage."""
        capital = calculate_slot_capital(
            available_capital=1000.0,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=2.44,
        )
        # margin = 1000 * 0.2 = 200, notional = 200 * 2.44 = 488
        assert abs(capital - 488.0) < 1e-6

    def test_margin_from_notional_with_float_leverage(self):
        """calculate_margin_from_notional should accept float leverage."""
        margin = calculate_margin_from_notional(488.0, 2.44)
        assert abs(margin - 200.0) < 1e-6

    def test_round_trip_r_equals_target(self):
        """Full round-trip: compute leverage, compute capital, verify R."""
        balance = 5000.0
        pct = 0.2
        target_r = 0.01
        stop_z = 3.5
        anchor_std = 0.012
        hr = 0.8
        cost_rate = estimate_cost_rate(0.0005, 0.001, 0.0001, 40)

        lev = calculate_dynamic_leverage(
            target_r_pct=target_r,
            position_size_pct=pct,
            stop_loss_zscore=stop_z,
            risk_std=anchor_std,
            hedge_ratio=hr,
            cost_rate=cost_rate,
            max_leverage=20.0,
        )

        notional = calculate_slot_capital(
            available_capital=balance,
            active_slots=[],
            max_traders=5,
            position_size_pct=pct,
            leverage=lev,
        )

        # Worst-case R = stop_loss + cost
        stop_loss = stop_z * anchor_std * notional / (1 + hr)
        total_cost = cost_rate * notional
        R = stop_loss + total_cost

        assert abs(R / balance - target_r) < 1e-8

    def test_vol_inverse_preserves_risk(self):
        """With vol-inverse sizing, R should stay ≤ target for non-anchor pairs."""
        balance = 10000.0
        pct = 0.2
        target_r = 0.01
        stop_z = 3.5
        anchor_std = 0.008
        pair_spread_std = 0.016  # 2x anchor
        hr = 1.0
        cost_rate = 0.00325

        lev = calculate_dynamic_leverage(
            target_r_pct=target_r,
            position_size_pct=pct,
            stop_loss_zscore=stop_z,
            risk_std=anchor_std,  # Use anchor_std when vol-inverse is ON
            hedge_ratio=hr,
            cost_rate=cost_rate,
            max_leverage=20.0,
        )

        base_notional = calculate_slot_capital(
            available_capital=balance,
            active_slots=[],
            max_traders=5,
            position_size_pct=pct,
            leverage=lev,
        )

        # Vol-inverse scales down for higher-vol pair
        vol_ratio = min(anchor_std / pair_spread_std, 1.0)
        actual_notional = base_notional * vol_ratio

        # Stop-loss uses pair's spread_std but notional is reduced
        stop_loss = stop_z * pair_spread_std * actual_notional / (1 + hr)
        total_cost = cost_rate * actual_notional
        R = stop_loss + total_cost
        R_pct = R / balance

        # Non-anchor pair R should be ≤ target (vol-inverse normalizes stop-loss,
        # but cost is also reduced by vol_ratio → R_non_anchor < R_anchor)
        assert R_pct <= target_r + 1e-8


class TestDynamicLeverageDisabled:
    """Tests for target_r_pct=0 (dynamic leverage disabled)."""

    def test_backtest_config_disabled(self):
        """When target_r_pct=0, should use fixed leverage."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig

        cfg = ContinuousBacktestConfig(
            target_r_pct=0.0,
            leverage=3,
        )
        assert cfg.target_r_pct == 0.0
        assert cfg.leverage == 3

    def test_backtest_config_enabled(self):
        """When target_r_pct>0, params should be set."""
        from quantbox.backtest.continuous_backtest import ContinuousBacktestConfig

        cfg = ContinuousBacktestConfig(
            target_r_pct=0.01,
            max_leverage=8.0,
        )
        assert cfg.target_r_pct == 0.01
        assert cfg.max_leverage == 8.0


class TestDynamicLeverageRealistic:
    """Realistic scenario tests with typical market parameters."""

    def test_typical_pair_1pct_risk(self):
        """Typical pair trading scenario targeting R=1%."""
        # Typical values: anchor_std ≈ 0.005-0.015, HR ≈ 0.5-1.5
        cost_rate = estimate_cost_rate(0.0005, 0.001, 0.0001, 40)

        lev = calculate_dynamic_leverage(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=1.5,  # stop_loss_zscore from .env
            risk_std=0.008,
            hedge_ratio=1.0,
            cost_rate=cost_rate,
        )

        # Should be a reasonable leverage value
        assert 1.0 <= lev <= 10.0
        # With these params: risk_per_unit = 1.5*0.008/2 + 0.00325 = 0.006+0.00325 = 0.00925
        # leverage = 0.01 / (0.2 * 0.00925) = 0.01 / 0.00185 ≈ 5.4x
        assert abs(lev - 0.01 / (0.2 * (1.5 * 0.008 / 2 + cost_rate))) < 0.1

    def test_high_vol_pair_gets_lower_leverage(self):
        """High-vol pair → lower leverage to keep R at 1%."""
        cost_rate = estimate_cost_rate(0.0005, 0.001, 0.0001, 40)

        lev_low_vol = calculate_dynamic_leverage(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=1.5,
            risk_std=0.005,
            hedge_ratio=1.0,
            cost_rate=cost_rate,
        )

        lev_high_vol = calculate_dynamic_leverage(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=1.5,
            risk_std=0.02,
            hedge_ratio=1.0,
            cost_rate=cost_rate,
        )

        assert lev_low_vol > lev_high_vol

    def test_unbalanced_pair_hr_05(self):
        """Unbalanced pair (HR=0.5) → higher per-unit risk → lower leverage."""
        cost_rate = estimate_cost_rate(0.0005, 0.001, 0.0001, 40)
        kwargs = dict(
            target_r_pct=0.01,
            position_size_pct=0.2,
            stop_loss_zscore=1.5,
            risk_std=0.01,
            cost_rate=cost_rate,
        )

        lev_balanced = calculate_dynamic_leverage(hedge_ratio=1.0, **kwargs)
        lev_unbalanced = calculate_dynamic_leverage(hedge_ratio=0.5, **kwargs)

        # HR=0.5 → divisor is 1.5 instead of 2.0 → more risk per unit
        assert lev_unbalanced < lev_balanced
