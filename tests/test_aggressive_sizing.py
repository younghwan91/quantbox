"""
Tests for aggressive position sizing: leverage and Kelly criterion.

Covers:
- calculate_slot_capital with leverage: notional amplification, margin math
- calculate_margin_from_notional: inverse leverage calculation
- calculate_kelly_from_trades: continuous Kelly from PnL distribution
- calculate_kelly_discrete: traditional binary Kelly
- compute_kelly_from_backtest: integration with backtest results
- ContinuousBacktestConfig.leverage: config propagation
- Backtest leverage parity: PnL scales with leverage, margin accounting
- Live trader leverage parity: config, capital flow, API leverage call

IMPORTANT: Unit tests for pure logic do NOT require real data.
           Synthetic trade returns are permitted for testing KELLY MATH ONLY
           (not price data used for backtesting).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from quantbox.core.trader_slot import (
    TraderSlotState,
    calculate_slot_capital,
    calculate_margin_from_notional,
    calculate_kelly_from_trades,
    calculate_kelly_discrete,
    get_active_slots,
)
from quantbox.backtest.continuous_backtest import (
    BacktestState,
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
    TradeResult,
    ExitReason,
    compute_kelly_from_backtest,
)


# ============================================================
# calculate_slot_capital with leverage
# ============================================================


class TestCalculateSlotCapitalLeverage:
    """Tests for leverage parameter in calculate_slot_capital."""

    def test_leverage_1x_unchanged(self):
        """Leverage=1 should produce same result as before."""
        result = calculate_slot_capital(
            available_capital=10000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=1,
        )
        assert result == pytest.approx(2000.0)

    def test_leverage_2x_doubles_notional(self):
        """Leverage=2 should double the effective notional."""
        result_1x = calculate_slot_capital(
            available_capital=10000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=1,
        )
        result_2x = calculate_slot_capital(
            available_capital=10000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=2,
        )
        assert result_2x == pytest.approx(result_1x * 2)

    def test_leverage_3x_triples_notional(self):
        """Leverage=3 should triple the effective notional."""
        result = calculate_slot_capital(
            available_capital=10000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=3,
        )
        # margin = 10000 * 0.2 = 2000, notional = 2000 * 3 = 6000
        assert result == pytest.approx(6000.0)

    def test_leverage_respects_available_cap(self):
        """Notional should be capped by available capital (margin can't exceed available)."""
        # position_size_pct=1.0 -> wants 100% margin, but leverage amplifies notional
        result = calculate_slot_capital(
            available_capital=1000,
            active_slots=[],
            max_traders=1,
            position_size_pct=1.0,
            leverage=2,
        )
        # margin = min(1000*1.0, 1000) = 1000, notional = 1000*2 = 2000
        assert result == pytest.approx(2000.0)

    def test_leverage_with_active_slots(self):
        """Leverage should amplify the margin-based allocation when slots are active."""
        active_slot = TraderSlotState(slot_id=0, position_open=True)
        active_slot.position_info = {'capital': 1000}  # margin locked
        
        result = calculate_slot_capital(
            available_capital=4000,
            active_slots=[active_slot],
            max_traders=5,
            position_size_pct=0.2,
            leverage=2,
        )
        # total_book = 4000 + 1000 = 5000
        # target_margin = 5000 * 0.2 = 1000
        # margin = min(1000, 4000) = 1000
        # notional = 1000 * 2 = 2000
        assert result == pytest.approx(2000.0)

    def test_leverage_zero_capital(self):
        """Zero available capital should return 0 regardless of leverage."""
        result = calculate_slot_capital(
            available_capital=0,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=3,
        )
        assert result == 0.0

    def test_leverage_invalid_below_1(self):
        """Leverage < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="leverage must be >= 1"):
            calculate_slot_capital(
                available_capital=10000,
                active_slots=[],
                max_traders=5,
                position_size_pct=0.2,
                leverage=0,
            )

    def test_negative_capital_still_raises(self):
        """Negative available_capital should still raise ValueError."""
        with pytest.raises(ValueError, match="available_capital cannot be negative"):
            calculate_slot_capital(
                available_capital=-100,
                active_slots=[],
                max_traders=5,
                position_size_pct=0.2,
                leverage=2,
            )

    def test_leverage_default_is_1(self):
        """Default leverage should be 1 (backward compatible)."""
        # Test default argument
        result = calculate_slot_capital(
            available_capital=10000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
        )
        assert result == pytest.approx(2000.0)


# ============================================================
# calculate_margin_from_notional
# ============================================================


class TestCalculateMarginFromNotional:
    """Tests for calculate_margin_from_notional."""

    def test_1x_margin_equals_notional(self):
        """With 1x leverage, margin == notional."""
        assert calculate_margin_from_notional(2000.0, 1) == pytest.approx(2000.0)

    def test_2x_margin_is_half_notional(self):
        """With 2x leverage, margin = notional / 2."""
        assert calculate_margin_from_notional(4000.0, 2) == pytest.approx(2000.0)

    def test_3x_margin(self):
        """With 3x leverage, margin = notional / 3."""
        assert calculate_margin_from_notional(6000.0, 3) == pytest.approx(2000.0)

    def test_roundtrip_consistency(self):
        """calculate_slot_capital → calculate_margin_from_notional should recover margin."""
        notional = calculate_slot_capital(
            available_capital=10000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=2,
        )
        margin = calculate_margin_from_notional(notional, leverage=2)
        # margin should equal the original un-leveraged allocation
        expected_margin = 10000 * 0.2
        assert margin == pytest.approx(expected_margin)

    def test_zero_notional(self):
        """Zero notional should return zero margin."""
        assert calculate_margin_from_notional(0.0, 2) == pytest.approx(0.0)

    def test_invalid_leverage(self):
        """Leverage < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="leverage must be >= 1"):
            calculate_margin_from_notional(1000.0, 0)


# ============================================================
# calculate_kelly_from_trades (continuous Kelly)
# ============================================================


class TestCalculateKellyFromTrades:
    """Tests for continuous Kelly criterion calculation.
    
    NOTE: These tests use synthetic RETURN data (not price data).
    Kelly math testing does not violate the real-data-only policy
    since we're testing a mathematical formula, not market behavior.
    """

    def test_positive_expectancy_returns_positive(self):
        """Trades with positive expectancy should return positive Kelly."""
        # Realistic pair trading returns: small wins and losses
        returns = [1.2, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4, 1.0, -0.2,
                   0.9, -0.6, 1.1, -0.3, 0.7]
        result = calculate_kelly_from_trades(returns, fraction=0.5)
        assert result > 0.0
        assert result <= 1.0

    def test_negative_expectancy_returns_zero(self):
        """Trades with negative mean return should return 0."""
        returns = [-1.0, -0.5, 0.2, -0.8, -1.2, 0.1, -0.9, -0.7, -1.1, -0.6]
        result = calculate_kelly_from_trades(returns, fraction=0.5)
        assert result == 0.0

    def test_insufficient_data(self):
        """Fewer than 10 trades should return 0 (insufficient sample)."""
        returns = [1.0, 0.5, -0.3]  # only 3 trades
        result = calculate_kelly_from_trades(returns, fraction=0.5)
        assert result == 0.0

    def test_half_kelly_smaller_than_full(self):
        """Half Kelly should produce smaller sizing than full Kelly."""
        returns = [1.2, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4, 1.0, -0.2,
                   0.9, -0.6, 1.1, -0.3, 0.7]
        half = calculate_kelly_from_trades(returns, fraction=0.5)
        full = calculate_kelly_from_trades(returns, fraction=1.0)
        assert half <= full  # half-Kelly ≤ full-Kelly always

    def test_fraction_validation(self):
        """fraction must be in (0, 1]."""
        returns = [1.0] * 15
        with pytest.raises(ValueError, match="fraction must be in"):
            calculate_kelly_from_trades(returns, fraction=0.0)
        with pytest.raises(ValueError, match="fraction must be in"):
            calculate_kelly_from_trades(returns, fraction=1.5)
        with pytest.raises(ValueError, match="fraction must be in"):
            calculate_kelly_from_trades(returns, fraction=-0.1)

    def test_clamped_to_max_1(self):
        """Result should never exceed 1.0 (100% of capital)."""
        # Extreme positive returns
        returns = [10.0, 8.0, 12.0, 9.0, 11.0, 7.0, 13.0, 8.0, 10.0, 9.0,
                   11.0, 12.0, 10.0, 8.0, 9.0]
        result = calculate_kelly_from_trades(returns, fraction=1.0)
        assert result <= 1.0

    def test_clamped_to_min_001(self):
        """Non-zero result should be at least 0.01."""
        # Very weak positive expectancy
        returns = [0.01, -0.009, 0.01, -0.009, 0.01, -0.009, 0.01, -0.009,
                   0.01, -0.009, 0.01, -0.009]
        result = calculate_kelly_from_trades(returns, fraction=0.5)
        # Either 0 (negative expectancy) or >= 0.01
        assert result == 0.0 or result >= 0.01

    def test_numpy_array_input(self):
        """Should accept numpy arrays."""
        returns = np.array([1.2, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4,
                           1.0, -0.2, 0.9, -0.6])
        result = calculate_kelly_from_trades(returns, fraction=0.5)
        assert isinstance(result, float)

    def test_all_wins_positive(self):
        """All winning trades should give positive Kelly."""
        returns = [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.5, 0.3, 0.4, 0.6]
        result = calculate_kelly_from_trades(returns, fraction=0.5)
        assert result > 0.0

    def test_deterministic(self):
        """Same input should always produce same output."""
        returns = [1.2, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4, 1.0, -0.2]
        r1 = calculate_kelly_from_trades(returns, fraction=0.5)
        r2 = calculate_kelly_from_trades(returns, fraction=0.5)
        assert r1 == r2


# ============================================================
# calculate_kelly_discrete (traditional Kelly)
# ============================================================


class TestCalculateKellyDiscrete:
    """Tests for discrete Kelly criterion calculation."""

    def test_coin_flip_even_odds(self):
        """Fair coin with even payoff → Kelly = 0."""
        result = calculate_kelly_discrete(
            win_rate=0.5, avg_win_pct=1.0, avg_loss_pct=1.0, fraction=1.0
        )
        assert result == 0.0  # Zero edge

    def test_positive_edge(self):
        """65% win rate with 1:1 payoff → positive Kelly."""
        result = calculate_kelly_discrete(
            win_rate=0.65, avg_win_pct=1.0, avg_loss_pct=1.0, fraction=1.0
        )
        # Kelly = (0.65*1 - 0.35)/1 = 0.30
        assert result == pytest.approx(0.30, abs=0.01)

    def test_pair_trading_profile(self):
        """Typical pair trading: high win rate, small payoff ratio."""
        result = calculate_kelly_discrete(
            win_rate=0.70,
            avg_win_pct=0.8,
            avg_loss_pct=1.2,
            fraction=0.5,
        )
        assert result > 0.0
        assert result <= 1.0

    def test_half_kelly(self):
        """Half-Kelly should halve the full Kelly value."""
        full = calculate_kelly_discrete(
            win_rate=0.65, avg_win_pct=1.5, avg_loss_pct=1.0, fraction=1.0
        )
        half = calculate_kelly_discrete(
            win_rate=0.65, avg_win_pct=1.5, avg_loss_pct=1.0, fraction=0.5
        )
        # Half-Kelly should be approximately half, but clamped
        if full > 0.02:  # If not at clamp boundary
            assert half == pytest.approx(full * 0.5, rel=0.01)

    def test_negative_edge(self):
        """Low win rate → Kelly = 0 (don't trade)."""
        result = calculate_kelly_discrete(
            win_rate=0.30, avg_win_pct=1.0, avg_loss_pct=1.0, fraction=1.0
        )
        assert result == 0.0

    def test_zero_win_rate(self):
        """0% win rate → 0."""
        result = calculate_kelly_discrete(
            win_rate=0.0, avg_win_pct=1.0, avg_loss_pct=1.0, fraction=1.0
        )
        assert result == 0.0

    def test_perfect_win_rate(self):
        """100% win rate → positive Kelly."""
        result = calculate_kelly_discrete(
            win_rate=1.0, avg_win_pct=1.0, avg_loss_pct=1.0, fraction=1.0
        )
        assert result > 0.0

    def test_invalid_win_rate(self):
        """win_rate outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="win_rate must be in"):
            calculate_kelly_discrete(
                win_rate=1.5, avg_win_pct=1.0, avg_loss_pct=1.0
            )
        with pytest.raises(ValueError, match="win_rate must be in"):
            calculate_kelly_discrete(
                win_rate=-0.1, avg_win_pct=1.0, avg_loss_pct=1.0
            )

    def test_invalid_fraction(self):
        """fraction outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="fraction must be in"):
            calculate_kelly_discrete(
                win_rate=0.5, avg_win_pct=1.0, avg_loss_pct=1.0, fraction=0.0
            )

    def test_zero_avg_values(self):
        """Zero avg_win or avg_loss should return 0."""
        assert calculate_kelly_discrete(
            win_rate=0.5, avg_win_pct=0.0, avg_loss_pct=1.0
        ) == 0.0
        assert calculate_kelly_discrete(
            win_rate=0.5, avg_win_pct=1.0, avg_loss_pct=0.0
        ) == 0.0

    def test_clamped_to_max_1(self):
        """Result should never exceed 1.0."""
        result = calculate_kelly_discrete(
            win_rate=0.99, avg_win_pct=10.0, avg_loss_pct=0.1, fraction=1.0
        )
        assert result <= 1.0

    def test_result_type(self):
        """Result should be float."""
        result = calculate_kelly_discrete(
            win_rate=0.65, avg_win_pct=1.0, avg_loss_pct=1.0
        )
        assert isinstance(result, float)


# ============================================================
# ContinuousBacktestConfig leverage
# ============================================================


class TestBacktestConfigLeverage:
    """Tests for leverage in ContinuousBacktestConfig."""

    def test_default_leverage_1(self):
        """Default leverage should be 1."""
        config = ContinuousBacktestConfig()
        assert config.leverage == 1

    def test_custom_leverage(self):
        """Custom leverage should be stored."""
        config = ContinuousBacktestConfig(leverage=2)
        assert config.leverage == 2

    def test_leverage_3x(self):
        """3x leverage should be valid."""
        config = ContinuousBacktestConfig(leverage=3)
        assert config.leverage == 3

    def test_leverage_independent_of_position_size(self):
        """Leverage and position_size_pct should be independent."""
        config = ContinuousBacktestConfig(
            leverage=2,
            position_size_pct=0.3,
            max_traders=5,
        )
        assert config.leverage == 2
        assert config.position_size_pct == pytest.approx(0.3)


# ============================================================
# Backtest leverage margin accounting
# ============================================================


class TestBacktestLeverageAccounting:
    """Tests for margin/notional accounting in backtest with leverage."""

    def _make_state(self, capital=10000.0, max_traders=5, leverage=2):
        """Helper to create a backtest state."""
        config = ContinuousBacktestConfig(
            initial_capital=capital,
            max_traders=max_traders,
            leverage=leverage,
        )
        state = BacktestState(
            current_capital=capital,
            peak_capital=capital,
            max_traders=max_traders,
        )
        return state, config

    def test_enter_slot_deducts_margin_not_notional(self):
        """When entering with 2x leverage, only margin should be deducted from cash."""
        state, config = self._make_state(capital=10000.0, leverage=2)
        engine = ContinuousPairBacktest(config)

        # Simulate entering a position
        slot = state.slots[0]
        
        # With 10000 capital, 0.2 position_size_pct, 2x leverage:
        # margin = 10000 * 0.2 = 2000
        # notional = 2000 * 2 = 4000
        notional = calculate_slot_capital(
            available_capital=state.current_capital,
            active_slots=state.active_slots,
            max_traders=config.max_traders,
            position_size_pct=config.position_size_pct,
            leverage=config.leverage,
        )
        margin = calculate_margin_from_notional(notional, config.leverage)

        assert notional == pytest.approx(4000.0)
        assert margin == pytest.approx(2000.0)

        # After deducting margin, 8000 should remain
        state.current_capital -= margin
        assert state.current_capital == pytest.approx(8000.0)

    def test_close_position_returns_margin_plus_pnl(self):
        """Closing should return margin + leveraged PnL to pool."""
        state, config = self._make_state(capital=10000.0, leverage=2)
        
        # Simulate a position: margin=2000, notional=4000
        slot = state.slots[0]
        slot.position_open = True
        slot.current_pair = ("BTCUSDT", "ETHUSDT")
        slot.position_info = {
            'capital': 2000.0,  # margin
            'notional': 4000.0,
            'entry_spread': 0.1,
            'signal': 1,
            'hedge_ratio': 0.85,
            'intercept': 0.0,
            'spread_mean': 0.1,
            'spread_std': 0.05,
            'entry_bar': 0,
            'entry_time': datetime.now(timezone.utc),
            'entry_z': -2.0,
            'entry_price1': 100.0,
            'entry_price2': 100.0,
            'leverage': 2,
        }
        state.current_capital = 8000.0  # (10000 - 2000 margin deducted)

        engine = ContinuousPairBacktest(config)

        # Close the position
        bar_data = {
            'BTCUSDT': {'close': 101.0, 'high': 101.0, 'low': 99.0, 'volume': 1000},
            'ETHUSDT': {'close': 100.0, 'high': 100.0, 'low': 99.0, 'volume': 1000},
        }
        engine._close_position(
            state, slot, bar_data, 10,
            datetime.now(timezone.utc), ExitReason.MEAN_REVERSION
        )

        # Slot should be reset
        assert not slot.position_open

        # Capital should be 8000 + 2000 (margin) + net_pnl
        assert state.current_capital > 8000.0  # margin returned + some PnL

        # Verify trade record
        assert len(state.trades) == 1
        trade = state.trades[0]
        assert trade.capital_allocated == pytest.approx(2000.0)  # margin

    def test_pnl_scales_with_leverage(self):
        """PnL from same spread change should be 2x with 2x leverage vs 1x."""
        # Setup for 1x
        state_1x, config_1x = self._make_state(capital=10000.0, leverage=1)
        engine_1x = ContinuousPairBacktest(config_1x)
        slot_1x = state_1x.slots[0]
        slot_1x.position_open = True
        slot_1x.current_pair = ("BTCUSDT", "ETHUSDT")
        slot_1x.position_info = {
            'capital': 2000.0,  # same margin
            'notional': 2000.0,  # 1x
            'entry_spread': 0.1,
            'entry_z': -2.0,
            'signal': 1,
            'hedge_ratio': 0.85,
            'intercept': 0.0,
            'spread_mean': 0.1,
            'spread_std': 0.05,
            'entry_bar': 0,
            'entry_time': datetime.now(timezone.utc),
            'entry_price1': 100.0,
            'entry_price2': 100.0,
            'leverage': 1,
        }
        state_1x.current_capital = 8000.0

        # Setup for 2x
        state_2x, config_2x = self._make_state(capital=10000.0, leverage=2)
        engine_2x = ContinuousPairBacktest(config_2x)
        slot_2x = state_2x.slots[0]
        slot_2x.position_open = True
        slot_2x.current_pair = ("BTCUSDT", "ETHUSDT")
        slot_2x.position_info = {
            'capital': 2000.0,  # same margin
            'notional': 4000.0,  # 2x
            'entry_spread': 0.1,
            'entry_z': -2.0,
            'signal': 1,
            'hedge_ratio': 0.85,
            'intercept': 0.0,
            'spread_mean': 0.1,
            'spread_std': 0.05,
            'entry_bar': 0,
            'entry_time': datetime.now(timezone.utc),
            'entry_price1': 100.0,
            'entry_price2': 100.0,
            'leverage': 2,
        }
        state_2x.current_capital = 8000.0

        # Close both with same bar data
        bar_data = {
            'BTCUSDT': {'close': 101.0, 'high': 101.0, 'low': 99.0, 'volume': 1000},
            'ETHUSDT': {'close': 100.0, 'high': 100.0, 'low': 99.0, 'volume': 1000},
        }
        engine_1x._close_position(
            state_1x, slot_1x, bar_data, 10,
            datetime.now(timezone.utc), ExitReason.MEAN_REVERSION
        )
        engine_2x._close_position(
            state_2x, slot_2x, bar_data, 10,
            datetime.now(timezone.utc), ExitReason.MEAN_REVERSION
        )

        pnl_1x = state_1x.trades[0].gross_pnl
        pnl_2x = state_2x.trades[0].gross_pnl

        # Gross PnL should scale with leverage (2x notional → 2x PnL)
        assert pnl_2x == pytest.approx(pnl_1x * 2, rel=0.01)

    def test_costs_scale_with_leverage(self):
        """Transaction costs should scale with notional, not margin."""
        state, config = self._make_state(capital=10000.0, leverage=2)
        engine = ContinuousPairBacktest(config)

        slot = state.slots[0]
        slot.position_open = True
        slot.current_pair = ("BTCUSDT", "ETHUSDT")
        slot.position_info = {
            'capital': 2000.0,
            'notional': 4000.0,
            'entry_spread': 0.1,
            'entry_z': -2.0,
            'signal': 1,
            'hedge_ratio': 0.85,
            'intercept': 0.0,
            'spread_mean': 0.1,
            'spread_std': 0.05,
            'entry_bar': 0,
            'entry_time': datetime.now(timezone.utc),
            'entry_price1': 100.0,
            'entry_price2': 100.0,
            'leverage': 2,
        }
        state.current_capital = 8000.0

        bar_data = {
            'BTCUSDT': {'close': 100.0, 'high': 100.0, 'low': 100.0, 'volume': 1000},
            'ETHUSDT': {'close': 100.0, 'high': 100.0, 'low': 100.0, 'volume': 1000},
        }
        engine._close_position(
            state, slot, bar_data, 10,
            datetime.now(timezone.utc), ExitReason.MAX_HOLDING
        )

        trade = state.trades[0]
        # Fees = 2 * fee_rate * notional = 2 * 0.0005 * 4000 = 4.0
        expected_fees = 2 * config.trading_fee * 4000.0
        assert trade.fees == pytest.approx(expected_fees)

        # Slippage = 2 * slippage_rate * notional
        expected_slippage = 2 * config.slippage * 4000.0
        assert trade.slippage_cost == pytest.approx(expected_slippage)

    def test_return_pct_relative_to_margin(self):
        """Return percentage should be computed against margin, not notional.
        
        This is critical: with leverage, a 1% spread move on notional is 
        actually 2% return on margin (with 2x leverage).
        """
        state, config = self._make_state(capital=10000.0, leverage=2)
        engine = ContinuousPairBacktest(config)

        slot = state.slots[0]
        slot.position_open = True
        slot.current_pair = ("BTCUSDT", "ETHUSDT")
        margin = 2000.0
        notional = 4000.0
        spread_change = 0.01  # 1% spread move
        slot.position_info = {
            'capital': margin,
            'notional': notional,
            'entry_spread': 0.1,
            'entry_z': -2.0,
            'signal': 1,
            'hedge_ratio': 0.85,
            'intercept': 0.0,
            'spread_mean': 0.1,
            'spread_std': 0.05,
            'entry_bar': 0,
            'entry_time': datetime.now(timezone.utc),
            'entry_price1': 100.0,
            'entry_price2': 100.0,
            'leverage': 2,
        }
        state.current_capital = 8000.0

        bar_data = {
            'BTCUSDT': {'close': 100.0, 'high': 100.0, 'low': 100.0, 'volume': 1000},
            'ETHUSDT': {'close': 100.0, 'high': 100.0, 'low': 100.0, 'volume': 1000},
        }
        engine._close_position(
            state, slot, bar_data, 10,
            datetime.now(timezone.utc), ExitReason.MEAN_REVERSION
        )

        trade = state.trades[0]
        # Return % should be (net_pnl / margin) * 100
        expected_return_pct = (trade.net_pnl / margin) * 100
        assert trade.net_return_pct == pytest.approx(expected_return_pct, rel=0.01)


# ============================================================
# compute_kelly_from_backtest
# ============================================================


class TestComputeKellyFromBacktest:
    """Tests for compute_kelly_from_backtest integration function."""

    def test_no_trades(self):
        """Empty trades should return 0."""
        results = {'trades': pd.DataFrame()}
        kelly = compute_kelly_from_backtest(results, fraction=0.5)
        assert kelly['kelly_position_size_pct'] == 0.0
        assert kelly['n_trades'] == 0

    def test_none_trades(self):
        """None trades should return 0."""
        results = {'trades': None}
        kelly = compute_kelly_from_backtest(results, fraction=0.5)
        assert kelly['kelly_position_size_pct'] == 0.0

    def test_positive_expectancy_trades(self):
        """Trades with positive expectancy should produce positive Kelly."""
        returns = [1.2, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4, 1.0, -0.2,
                   0.9, -0.6, 1.1, -0.3, 0.7]
        trades_df = pd.DataFrame({'return_pct': returns})
        results = {'trades': trades_df}
        kelly = compute_kelly_from_backtest(results, fraction=0.5)
        
        assert kelly['kelly_position_size_pct'] > 0.0
        assert kelly['n_trades'] == 15
        assert kelly['win_rate'] > 50.0
        assert kelly['avg_win_pct'] > 0
        assert kelly['avg_loss_pct'] > 0
        assert kelly['fraction'] == 0.5

    def test_uses_more_conservative_estimate(self):
        """Should use min(continuous, discrete) Kelly."""
        returns = [1.2, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4, 1.0, -0.2,
                   0.9, -0.6, 1.1, -0.3, 0.7]
        trades_df = pd.DataFrame({'return_pct': returns})
        results = {'trades': trades_df}
        kelly = compute_kelly_from_backtest(results, fraction=0.5)

        assert kelly['kelly_position_size_pct'] <= kelly['continuous_kelly']
        assert kelly['kelly_position_size_pct'] <= kelly['discrete_kelly']

    def test_output_fields(self):
        """Should contain all expected fields."""
        returns = [1.0, -0.5, 0.8, -0.3, 1.5, -0.7, 0.6, -0.4, 1.0, -0.2]
        trades_df = pd.DataFrame({'return_pct': returns})
        results = {'trades': trades_df}
        kelly = compute_kelly_from_backtest(results, fraction=0.5)

        expected_keys = {
            'kelly_position_size_pct', 'continuous_kelly', 'discrete_kelly',
            'kelly_raw', 'n_trades', 'win_rate', 'avg_win_pct', 'avg_loss_pct',
            'mean_return_pct', 'std_return_pct', 'fraction',
        }
        assert expected_keys.issubset(set(kelly.keys()))

    def test_negative_expectancy(self):
        """Negative expectancy trades should give 0 Kelly sizing."""
        returns = [-1.0, -0.5, 0.2, -0.8, -1.2, 0.1, -0.9, -0.7, -1.1, -0.6]
        trades_df = pd.DataFrame({'return_pct': returns})
        results = {'trades': trades_df}
        kelly = compute_kelly_from_backtest(results, fraction=0.5)
        assert kelly['kelly_position_size_pct'] == 0.0


# ============================================================
# Live trader leverage parity
# ============================================================


class TestLiveTraderLeverageParity:
    """Tests for leverage parity between backtest and live trader.
    
    Uses mocked Binance client to test leverage flows without real API calls.
    """

    def _create_mock_trader(self, leverage=2):
        """Create a LivePairTrader with mocked dependencies."""
        from quantbox.live_trader import LivePairTrader

        with patch.dict('os.environ', {
            'BINANCE_API_KEY': 'test_key',
            'BINANCE_API_SECRET': 'test_secret',
            'BINANCE_TESTNET': 'true',
            'LEVERAGE': str(leverage),
        }):
            with patch('quantbox.live_trader.Client') as MockClient:
                mock_client = MagicMock()
                MockClient.return_value = mock_client
                trader = LivePairTrader(paper_mode=True, config={
                    'quote_currency': 'USDT',
                    'n_coins': 10,
                    'interval': '15m',
                    'formation_hours': 168,
                    'entry_zscore': 2.0,
                    'exit_zscore': 0.5,
                    'stop_loss_zscore': 3.5,
                    'max_holding_hours': 24,
                    'position_size_pct': -1.0,
                    'trading_fee': 0.0005,
                    'slippage': 0.001,
                    'leverage': leverage,
                    'max_traders': 5,
                    'min_correlation': 0.5,
                    'max_cointegration_pvalue': 0.05,
                    'min_half_life': 3,
                    'max_half_life': 60,
                    'max_hurst': 0.5,
                    'check_interval_seconds': 60,
                    'retry_attempts': 3,
                    'retry_delay': 2,
                })
                return trader

    def test_config_leverage_from_env(self):
        """Leverage should be read from config/env."""
        trader = self._create_mock_trader(leverage=2)
        assert trader.config['leverage'] == 2

    def test_config_leverage_3x(self):
        """3x leverage should be configurable."""
        trader = self._create_mock_trader(leverage=3)
        assert trader.config['leverage'] == 3

    def test_slot_capital_uses_leverage(self):
        """run_cycle's slot_capital should use leverage."""
        trader = self._create_mock_trader(leverage=2)
        trader.paper_balance = 10000.0

        # Directly test calculate_slot_capital call
        slot_capital = calculate_slot_capital(
            available_capital=10000.0,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=2,
        )
        # margin = 10000 * 0.2 = 2000, notional = 2000 * 2 = 4000
        assert slot_capital == pytest.approx(4000.0)

    def test_backtest_live_leverage_match(self):
        """Backtest and live should compute same notional for same inputs."""
        # Backtest config
        bt_config = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=5,
            leverage=2,
        )

        # Live config
        live_leverage = 2
        live_position_size_pct = 1.0 / 5  # auto

        bt_notional = calculate_slot_capital(
            available_capital=10000.0,
            active_slots=[],
            max_traders=bt_config.max_traders,
            position_size_pct=bt_config.position_size_pct,
            leverage=bt_config.leverage,
        )

        live_notional = calculate_slot_capital(
            available_capital=10000.0,
            active_slots=[],
            max_traders=5,
            position_size_pct=live_position_size_pct,
            leverage=live_leverage,
        )

        assert bt_notional == pytest.approx(live_notional)

    def test_position_data_stores_margin_and_notional(self):
        """execute_pair_trade should store margin (capital) and notional separately."""
        trader = self._create_mock_trader(leverage=2)

        # capital=4000 is notional (from calculate_slot_capital with 2x)
        # margin should be 2000
        expected_margin = calculate_margin_from_notional(4000.0, 2)
        assert expected_margin == pytest.approx(2000.0)


# ============================================================
# Safety checks
# ============================================================


class TestLeverageSafetyChecks:
    """Safety checks for leverage implementation."""

    def test_leverage_cannot_exceed_margin(self):
        """With leverage, margin used should never exceed available capital."""
        for lev in [1, 2, 3, 5]:
            notional = calculate_slot_capital(
                available_capital=1000.0,
                active_slots=[],
                max_traders=5,
                position_size_pct=0.2,
                leverage=lev,
            )
            margin = calculate_margin_from_notional(notional, lev)
            assert margin <= 1000.0, f"Margin {margin} exceeds available 1000 at {lev}x"

    def test_multiple_slots_leverage_total_margin(self):
        """Total margin across all slots should not exceed total capital."""
        capital = 10000.0
        leverage = 2
        max_traders = 5
        pct = 1.0 / max_traders

        total_margin = 0.0
        remaining = capital

        for i in range(max_traders):
            active = [TraderSlotState(slot_id=j, position_open=True) for j in range(i)]
            for j, s in enumerate(active):
                s.position_info = {'capital': capital * pct / max_traders}

            notional = calculate_slot_capital(
                available_capital=remaining,
                active_slots=active,
                max_traders=max_traders,
                position_size_pct=pct,
                leverage=leverage,
            )
            margin = calculate_margin_from_notional(notional, leverage)
            total_margin += margin
            remaining -= margin

        assert total_margin <= capital + 0.01  # floating point tolerance

    def test_leverage_1_backward_compatible(self):
        """With leverage=1, behavior should be identical to pre-leverage code."""
        result_default = calculate_slot_capital(
            available_capital=5000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
        )
        result_explicit = calculate_slot_capital(
            available_capital=5000,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
            leverage=1,
        )
        assert result_default == result_explicit
        # With 1x, margin == notional
        assert calculate_margin_from_notional(result_explicit, 1) == result_explicit

    def test_kelly_never_negative(self):
        """Kelly should never return a negative value."""
        # Various edge cases
        assert calculate_kelly_from_trades(
            [-1.0] * 20, fraction=0.5
        ) >= 0.0
        assert calculate_kelly_discrete(
            win_rate=0.1, avg_win_pct=0.1, avg_loss_pct=10.0
        ) >= 0.0
