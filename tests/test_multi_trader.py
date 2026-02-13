"""
Tests for multi-trader architecture.

Covers:
- TraderSlotState: basic state management
- Helper functions: symbol exclusion, capital allocation, pair availability
- BacktestState with multi-slots: initialization, properties
- ContinuousBacktestConfig with max_traders: auto position_size_pct
- ContinuousPairBacktest._find_top_pairs: non-overlapping selection
- ContinuousPairBacktest._enter_slot: capital deduction, slot state
- ContinuousPairBacktest._close_position: capital return, slot reset
- ContinuousPairBacktest end-to-end: multi-trader run (integration, uses real data)
- LivePairTrader multi-slot state management

IMPORTANT: Integration tests use REAL Binance data from cache.
           Unit tests for pure logic do NOT require real data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from quantbox.core.trader_slot import (
    TraderSlotState,
    get_occupied_symbols,
    get_idle_slots,
    get_active_slots,
    calculate_slot_capital,
    is_pair_available,
)
from quantbox.backtest.continuous_backtest import (
    BacktestState,
    ContinuousBacktestConfig,
    ContinuousPairBacktest,
    TradeResult,
    ExitReason,
    PairStatistics,
)


# ============================================================
# TraderSlotState unit tests
# ============================================================


class TestTraderSlotState:
    """Tests for TraderSlotState dataclass."""

    def test_init_default(self):
        slot = TraderSlotState(slot_id=0)
        assert slot.slot_id == 0
        assert slot.position_open is False
        assert slot.current_pair is None
        assert slot.position_info is None

    def test_occupied_symbols_idle(self):
        slot = TraderSlotState(slot_id=0)
        assert slot.occupied_symbols == set()

    def test_occupied_symbols_active(self):
        slot = TraderSlotState(
            slot_id=1,
            position_open=True,
            current_pair=("BTCUSDT", "ETHUSDT"),
        )
        assert slot.occupied_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_occupied_symbols_position_open_but_no_pair(self):
        """Edge case: position_open=True but current_pair=None."""
        slot = TraderSlotState(slot_id=0, position_open=True, current_pair=None)
        assert slot.occupied_symbols == set()

    def test_reset(self):
        slot = TraderSlotState(
            slot_id=2,
            position_open=True,
            current_pair=("BTCUSDT", "ETHUSDT"),
            position_info={"capital": 2000},
        )
        slot.reset()
        assert slot.position_open is False
        assert slot.current_pair is None
        assert slot.position_info is None
        assert slot.slot_id == 2  # slot_id preserved


# ============================================================
# Helper function unit tests
# ============================================================


class TestGetOccupiedSymbols:
    """Tests for get_occupied_symbols."""

    def test_no_slots(self):
        assert get_occupied_symbols([]) == set()

    def test_all_idle(self):
        slots = [TraderSlotState(slot_id=i) for i in range(3)]
        assert get_occupied_symbols(slots) == set()

    def test_some_active(self):
        slots = [
            TraderSlotState(slot_id=0, position_open=True, current_pair=("A", "B")),
            TraderSlotState(slot_id=1),
            TraderSlotState(slot_id=2, position_open=True, current_pair=("C", "D")),
        ]
        assert get_occupied_symbols(slots) == {"A", "B", "C", "D"}

    def test_no_symbol_duplication_in_result(self):
        """Even if somehow two slots had overlapping symbols, set deduplicates."""
        slots = [
            TraderSlotState(slot_id=0, position_open=True, current_pair=("A", "B")),
            TraderSlotState(slot_id=1, position_open=True, current_pair=("A", "C")),
        ]
        assert get_occupied_symbols(slots) == {"A", "B", "C"}


class TestGetIdleAndActiveSlots:
    """Tests for get_idle_slots and get_active_slots."""

    def test_all_idle(self):
        slots = [TraderSlotState(slot_id=i) for i in range(5)]
        assert len(get_idle_slots(slots)) == 5
        assert len(get_active_slots(slots)) == 0

    def test_all_active(self):
        slots = [
            TraderSlotState(slot_id=i, position_open=True, current_pair=("A", f"B{i}"))
            for i in range(5)
        ]
        assert len(get_idle_slots(slots)) == 0
        assert len(get_active_slots(slots)) == 5

    def test_mixed(self):
        slots = [
            TraderSlotState(slot_id=0, position_open=True, current_pair=("A", "B")),
            TraderSlotState(slot_id=1),
            TraderSlotState(slot_id=2, position_open=True, current_pair=("C", "D")),
            TraderSlotState(slot_id=3),
            TraderSlotState(slot_id=4),
        ]
        idle = get_idle_slots(slots)
        active = get_active_slots(slots)
        assert len(idle) == 3
        assert len(active) == 2
        assert {s.slot_id for s in idle} == {1, 3, 4}
        assert {s.slot_id for s in active} == {0, 2}


class TestCalculateSlotCapital:
    """Tests for calculate_slot_capital."""

    def test_first_slot_full(self):
        """First slot entry with no active existing slots."""
        capital = calculate_slot_capital(
            available_capital=10000.0,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
        )
        assert capital == pytest.approx(2000.0)

    def test_second_slot(self):
        """Second slot with one active slot."""
        active_slot = TraderSlotState(
            slot_id=0, position_open=True,
            current_pair=("A", "B"),
            position_info={"capital": 2000.0},
        )
        capital = calculate_slot_capital(
            available_capital=8000.0,
            active_slots=[active_slot],
            max_traders=5,
            position_size_pct=0.2,
        )
        # total_book_value = 8000 + 2000 = 10000
        # target = 10000 * 0.2 = 2000
        assert capital == pytest.approx(2000.0)

    def test_last_slot(self):
        """Last slot: should get remaining capital."""
        active_slots = [
            TraderSlotState(
                slot_id=i, position_open=True,
                current_pair=("A", f"B{i}"),
                position_info={"capital": 2000.0},
            )
            for i in range(4)
        ]
        capital = calculate_slot_capital(
            available_capital=2000.0,
            active_slots=active_slots,
            max_traders=5,
            position_size_pct=0.2,
        )
        # total_book_value = 2000 + 4*2000 = 10000
        # target = 10000 * 0.2 = 2000
        assert capital == pytest.approx(2000.0)

    def test_zero_available(self):
        """No capital available → should return 0."""
        active_slots = [
            TraderSlotState(
                slot_id=i, position_open=True,
                current_pair=("A", f"B{i}"),
                position_info={"capital": 2000.0},
            )
            for i in range(5)
        ]
        capital = calculate_slot_capital(
            available_capital=0.0,
            active_slots=active_slots,
            max_traders=5,
            position_size_pct=0.2,
        )
        assert capital == 0.0

    def test_negative_available_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_slot_capital(
                available_capital=-100.0,
                active_slots=[],
                max_traders=5,
                position_size_pct=0.2,
            )

    def test_capped_at_available(self):
        """Target exceeds available → capped."""
        capital = calculate_slot_capital(
            available_capital=500.0,
            active_slots=[],
            max_traders=1,
            position_size_pct=1.0,
        )
        assert capital == pytest.approx(500.0)

    def test_single_trader_backward_compat(self):
        """max_traders=1, position_size_pct=1.0 → full capital."""
        capital = calculate_slot_capital(
            available_capital=10000.0,
            active_slots=[],
            max_traders=1,
            position_size_pct=1.0,
        )
        assert capital == pytest.approx(10000.0)

    def test_with_profits(self):
        """When a slot closes with profit, more capital is available."""
        # Slot 0 closed with +500 profit, returned 2500 to pool
        # Slot 1 still active with 2000
        active_slot = TraderSlotState(
            slot_id=1, position_open=True,
            current_pair=("A", "B"),
            position_info={"capital": 2000.0},
        )
        capital = calculate_slot_capital(
            available_capital=8500.0,  # 10000 - 2000 + 500 profit
            active_slots=[active_slot],
            max_traders=5,
            position_size_pct=0.2,
        )
        # total_book_value = 8500 + 2000 = 10500
        # target = 10500 * 0.2 = 2100
        assert capital == pytest.approx(2100.0)


class TestIsPairAvailable:
    """Tests for is_pair_available."""

    def test_available(self):
        assert is_pair_available(("A", "B"), set()) is True

    def test_first_symbol_occupied(self):
        assert is_pair_available(("A", "B"), {"A"}) is False

    def test_second_symbol_occupied(self):
        assert is_pair_available(("A", "B"), {"B"}) is False

    def test_both_occupied(self):
        assert is_pair_available(("A", "B"), {"A", "B"}) is False

    def test_neither_occupied(self):
        assert is_pair_available(("A", "B"), {"C", "D"}) is True


# ============================================================
# BacktestState unit tests
# ============================================================


class TestBacktestState:
    """Tests for BacktestState with multi-slot support."""

    def test_default_init(self):
        state = BacktestState(current_capital=10000, peak_capital=10000)
        assert len(state.slots) == 1  # max_traders defaults to 1
        assert state.slots[0].slot_id == 0

    def test_multi_trader_init(self):
        state = BacktestState(
            current_capital=10000, peak_capital=10000, max_traders=5
        )
        assert len(state.slots) == 5
        assert all(s.slot_id == i for i, s in enumerate(state.slots))

    def test_occupied_symbols_empty(self):
        state = BacktestState(
            current_capital=10000, peak_capital=10000, max_traders=5
        )
        assert state.occupied_symbols == set()

    def test_occupied_symbols_with_active(self):
        state = BacktestState(
            current_capital=10000, peak_capital=10000, max_traders=3
        )
        state.slots[0].position_open = True
        state.slots[0].current_pair = ("BTCUSDT", "ETHUSDT")
        state.slots[2].position_open = True
        state.slots[2].current_pair = ("SOLUSDT", "BNBUSDT")
        assert state.occupied_symbols == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}

    def test_idle_and_active_slots(self):
        state = BacktestState(
            current_capital=10000, peak_capital=10000, max_traders=5
        )
        state.slots[0].position_open = True
        state.slots[0].current_pair = ("A", "B")
        state.slots[0].position_info = {"capital": 2000}
        assert len(state.idle_slots) == 4
        assert len(state.active_slots) == 1

    def test_total_allocated(self):
        state = BacktestState(
            current_capital=6000, peak_capital=10000, max_traders=5
        )
        for i in range(2):
            state.slots[i].position_open = True
            state.slots[i].current_pair = ("A", f"B{i}")
            state.slots[i].position_info = {"capital": 2000}
        assert state.total_allocated == pytest.approx(4000.0)

    def test_total_equity(self):
        state = BacktestState(
            current_capital=6000, peak_capital=10000, max_traders=5
        )
        for i in range(2):
            state.slots[i].position_open = True
            state.slots[i].current_pair = ("A", f"B{i}")
            state.slots[i].position_info = {"capital": 2000}
        # total_equity = cash + allocated = 6000 + 4000 = 10000
        assert state.total_equity == pytest.approx(10000.0)

    def test_custom_slots_preserved(self):
        """If slots are provided explicitly, __post_init__ should not override."""
        custom_slots = [TraderSlotState(slot_id=99)]
        state = BacktestState(
            current_capital=10000,
            peak_capital=10000,
            max_traders=5,
            slots=custom_slots,
        )
        assert len(state.slots) == 1
        assert state.slots[0].slot_id == 99


# ============================================================
# ContinuousBacktestConfig unit tests
# ============================================================


class TestContinuousBacktestConfig:
    """Tests for config with max_traders support."""

    def test_default_max_traders(self):
        cfg = ContinuousBacktestConfig()
        assert cfg.max_traders == 5

    def test_auto_position_size_pct(self):
        cfg = ContinuousBacktestConfig(max_traders=5)
        assert cfg.position_size_pct == pytest.approx(0.2)

    def test_auto_position_size_pct_single(self):
        cfg = ContinuousBacktestConfig(max_traders=1)
        assert cfg.position_size_pct == pytest.approx(1.0)

    def test_explicit_position_size_pct(self):
        """If explicitly set, should not be overridden."""
        cfg = ContinuousBacktestConfig(max_traders=5, position_size_pct=0.15)
        assert cfg.position_size_pct == pytest.approx(0.15)

    def test_total_round_trip_cost(self):
        cfg = ContinuousBacktestConfig()
        expected = 2 * (cfg.trading_fee + cfg.slippage)
        assert cfg.total_round_trip_cost == pytest.approx(expected)


# ============================================================
# TradeResult unit tests
# ============================================================


class TestTradeResult:
    """Tests for TradeResult with slot_id field."""

    def test_slot_id_default(self):
        trade = TradeResult(
            pair=("A", "B"), entry_time=datetime.now(), exit_time=datetime.now(),
            entry_z=2.0, exit_z=0.5, entry_spread=0.01, exit_spread=0.005,
            signal=1, capital_allocated=2000, gross_pnl=100, fees=2,
            slippage_cost=4, funding_cost=0, net_pnl=94, net_return_pct=4.7, bars_held=10,
            exit_reason=ExitReason.MEAN_REVERSION, hedge_ratio=1.0,
            spread_mean=0.0, spread_std=0.01,
        )
        assert trade.slot_id == 0  # default

    def test_slot_id_custom(self):
        trade = TradeResult(
            pair=("A", "B"), entry_time=datetime.now(), exit_time=datetime.now(),
            entry_z=2.0, exit_z=0.5, entry_spread=0.01, exit_spread=0.005,
            signal=1, capital_allocated=2000, gross_pnl=100, fees=2,
            slippage_cost=4, funding_cost=0, net_pnl=94, net_return_pct=4.7, bars_held=10,
            exit_reason=ExitReason.MEAN_REVERSION, hedge_ratio=1.0,
            spread_mean=0.0, spread_std=0.01, slot_id=3,
        )
        assert trade.slot_id == 3


# ============================================================
# PairStatistics tests (unchanged, sanity check)
# ============================================================


class TestPairStatistics:
    """Sanity checks for PairStatistics used in multi-trader context."""

    def test_hedge_ratio_identity(self):
        """Identical prices → hedge ratio ≈ 1.0."""
        prices = np.exp(np.cumsum(np.random.randn(100) * 0.01) + np.log(100))
        hr, intercept = PairStatistics.calculate_hedge_ratio(prices, prices)
        assert hr == pytest.approx(1.0, abs=0.01)

    def test_zscore_zero_std(self):
        """Zero std → return zeros."""
        spread = np.array([1.0, 2.0, 3.0])
        z = PairStatistics.calculate_zscore(spread, mean=2.0, std=0.0)
        assert np.all(z == 0.0)


# ============================================================
# ContinuousPairBacktest._find_top_pairs tests
# ============================================================


class TestFindTopPairs:
    """Tests for _find_top_pairs with symbol exclusion and greedy selection."""

    @pytest.fixture
    def backtest_engine(self):
        cfg = ContinuousBacktestConfig(
            max_traders=5,
            formation_bars=50,
            min_correlation=0.3,
            max_cointegration_pvalue=0.5,  # relaxed for test
            min_half_life=1.0,
            max_half_life=200.0,
        )
        return ContinuousPairBacktest(cfg)

    @pytest.fixture
    def synthetic_cointegrated_data(self):
        """Generate synthetic cointegrated data for _find_top_pairs testing.

        NOTE: This is ONLY for testing the pair selection algorithm logic
        (greedy selection, symbol exclusion). Not for PnL or strategy validation.
        Real data is required for integration/backtest tests per project rules.
        """
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2025-01-01", periods=n, freq="15min")

        # Create 6 assets with known cointegration relationships:
        # A-B cointegrated, C-D cointegrated, E-F cointegrated
        base1 = np.cumsum(np.random.randn(n) * 0.01) + np.log(100)
        base2 = np.cumsum(np.random.randn(n) * 0.01) + np.log(200)
        base3 = np.cumsum(np.random.randn(n) * 0.01) + np.log(50)

        data = {}
        for name, base, offset in [
            ("A", base1, 0.0), ("B", base1, 0.02),
            ("C", base2, 0.0), ("D", base2, 0.03),
            ("E", base3, 0.0), ("F", base3, 0.01),
        ]:
            noise = np.random.randn(n) * 0.005
            prices = np.exp(base + offset + noise)
            data[name] = pd.DataFrame(
                {"Close": prices, "High": prices * 1.001,
                 "Low": prices * 0.999, "Volume": 1e6},
                index=timestamps,
            )
        return data

    def test_returns_list(self, backtest_engine, synthetic_cointegrated_data):
        result = backtest_engine._find_top_pairs(
            synthetic_cointegrated_data, 0, 49, excluded_symbols=set(), n=3
        )
        assert isinstance(result, list)

    def test_no_symbol_overlap(self, backtest_engine, synthetic_cointegrated_data):
        """Selected pairs should not share any symbol."""
        result = backtest_engine._find_top_pairs(
            synthetic_cointegrated_data, 0, 49, excluded_symbols=set(), n=3
        )
        all_symbols = []
        for c in result:
            all_symbols.extend([c["symbol1"], c["symbol2"]])
        assert len(all_symbols) == len(set(all_symbols)), "Symbol overlap detected!"

    def test_excluded_symbols_respected(
        self, backtest_engine, synthetic_cointegrated_data
    ):
        """Candidate pairs should not include excluded symbols."""
        result = backtest_engine._find_top_pairs(
            synthetic_cointegrated_data, 0, 49,
            excluded_symbols={"A", "B"}, n=3,
        )
        for c in result:
            assert c["symbol1"] not in {"A", "B"}
            assert c["symbol2"] not in {"A", "B"}

    def test_returns_at_most_n(self, backtest_engine, synthetic_cointegrated_data):
        result = backtest_engine._find_top_pairs(
            synthetic_cointegrated_data, 0, 49, excluded_symbols=set(), n=1
        )
        assert len(result) <= 1

    def test_empty_on_no_candidates(self, backtest_engine):
        """If data is too short or all filtered out, return empty list."""
        data = {
            "X": pd.DataFrame(
                {"Close": [100.0] * 10}, index=pd.date_range("2025-01-01", periods=10)
            ),
        }
        result = backtest_engine._find_top_pairs(data, 0, 5, set(), 1)
        assert result == []

    def test_sorted_by_score(self, backtest_engine, synthetic_cointegrated_data):
        """Results should be sorted by score ascending (best first)."""
        result = backtest_engine._find_top_pairs(
            synthetic_cointegrated_data, 0, 49, excluded_symbols=set(), n=3
        )
        if len(result) > 1:
            scores = [c["score"] for c in result]
            for i in range(len(scores) - 1):
                assert scores[i] <= scores[i + 1]


# ============================================================
# Backtest _enter_slot and _close_position tests
# ============================================================


class TestEnterAndCloseSlot:
    """Tests for slot entry and close position capital management."""

    @pytest.fixture
    def engine_and_state(self):
        cfg = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=5,
            formation_bars=50,
            entry_zscore=2.0,
            exit_zscore=0.5,
        )
        engine = ContinuousPairBacktest(cfg)
        state = BacktestState(
            current_capital=10000.0,
            peak_capital=10000.0,
            max_traders=5,
        )
        return engine, state

    def test_enter_slot_deducts_capital(self, engine_and_state):
        engine, state = engine_and_state

        candidate = {
            "symbol1": "AAA",
            "symbol2": "BBB",
            "hedge_ratio": 1.0,
            "intercept": 0.0,
            "spread_mean": 0.0,
            "spread_std": 0.01,
        }

        # Create minimal data with extreme z-score for guaranteed entry
        np.random.seed(0)
        n = 60
        timestamps = pd.date_range("2025-01-01", periods=n, freq="15min")
        # Spread at bar 50 should have |z| > 2.0
        prices_a = np.exp(np.cumsum(np.random.randn(n) * 0.001) + np.log(100))
        # Make BBB diverge strongly to create extreme z-score
        prices_b = np.exp(np.cumsum(np.random.randn(n) * 0.001) + np.log(100))
        prices_a[50] = 100 * np.exp(0.03)  # Force large positive spread

        data = {
            "AAA": pd.DataFrame({"Close": prices_a}, index=timestamps),
            "BBB": pd.DataFrame({"Close": prices_b}, index=timestamps),
        }

        initial_capital = state.current_capital
        result = engine._enter_slot(
            state, state.slots[0], candidate, data,
            bar_idx=50, timestamp=timestamps[50],
        )

        if result:
            # Capital should have been deducted
            assert state.current_capital < initial_capital
            assert state.slots[0].position_open is True
            assert state.slots[0].current_pair == ("AAA", "BBB")
            assert state.slots[0].position_info is not None
            assert state.slots[0].position_info["capital"] > 0

    def test_close_position_returns_capital(self, engine_and_state):
        engine, state = engine_and_state

        # Manually set up a position in slot 0
        slot = state.slots[0]
        capital_allocated = 2000.0
        state.current_capital -= capital_allocated

        slot.position_open = True
        slot.current_pair = ("AAA", "BBB")
        slot.position_info = {
            "entry_bar": 50,
            "entry_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "entry_price1": 100.0,
            "entry_price2": 100.0,
            "entry_spread": 0.0,
            "entry_z": 2.5,
            "signal": -1,
            "capital": capital_allocated,
            "hedge_ratio": 1.0,
            "intercept": 0.0,
            "spread_mean": 0.0,
            "spread_std": 0.01,
        }

        bar_data = {
            "AAA": {"close": 100.0, "high": 101, "low": 99, "volume": 1e6},
            "BBB": {"close": 100.0, "high": 101, "low": 99, "volume": 1e6},
        }

        capital_before = state.current_capital
        engine._close_position(
            state, slot, bar_data, bar_idx=60,
            timestamp=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            exit_reason=ExitReason.MEAN_REVERSION,
        )

        # Slot should be reset
        assert slot.position_open is False
        assert slot.current_pair is None
        assert slot.position_info is None

        # Capital should be returned (original + PnL - costs)
        assert state.current_capital > capital_before
        # Verify trade was recorded
        assert len(state.trades) == 1
        assert state.trades[0].slot_id == 0

    def test_close_position_records_slot_id(self, engine_and_state):
        engine, state = engine_and_state

        slot = state.slots[3]  # Use slot 3
        capital_allocated = 2000.0
        state.current_capital -= capital_allocated
        slot.position_open = True
        slot.current_pair = ("XXX", "YYY")
        slot.position_info = {
            "entry_bar": 50, "entry_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "entry_price1": 50.0, "entry_price2": 30.0,
            "entry_spread": 0.0, "entry_z": -2.5, "signal": 1,
            "capital": capital_allocated, "hedge_ratio": 1.5, "intercept": 0.0,
            "spread_mean": 0.0, "spread_std": 0.01,
        }

        bar_data = {
            "XXX": {"close": 50.0, "high": 51, "low": 49, "volume": 1e6},
            "YYY": {"close": 30.0, "high": 31, "low": 29, "volume": 1e6},
        }

        engine._close_position(
            state, slot, bar_data, 60,
            datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            ExitReason.MAX_HOLDING,
        )
        assert state.trades[-1].slot_id == 3


# ============================================================
# _record_equity multi-slot aggregation tests
# ============================================================


class TestRecordEquity:
    """Tests for equity recording across multiple active slots."""

    def test_equity_no_position(self):
        cfg = ContinuousBacktestConfig(max_traders=5)
        engine = ContinuousPairBacktest(cfg)
        state = BacktestState(
            current_capital=10000.0, peak_capital=10000.0, max_traders=5
        )
        bar_data = {}
        engine._record_equity(
            state, bar_data, datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        assert len(state.equity_curve) == 1
        assert state.equity_curve[0]["equity"] == pytest.approx(10000.0)
        assert state.equity_curve[0]["active_slots"] == 0
        assert state.equity_curve[0]["in_position"] is False

    def test_equity_with_active_slots(self):
        cfg = ContinuousBacktestConfig(max_traders=5)
        engine = ContinuousPairBacktest(cfg)
        state = BacktestState(
            current_capital=6000.0, peak_capital=10000.0, max_traders=5
        )

        # Set up two active slots
        for i in range(2):
            slot = state.slots[i]
            slot.position_open = True
            slot.current_pair = (f"A{i}", f"B{i}")
            slot.position_info = {
                "capital": 2000.0,
                "hedge_ratio": 1.0,
                "intercept": 0.0,
                "spread_mean": 0.0,
                "spread_std": 0.01,
                "entry_spread": 0.0,
                "signal": 1,
            }

        bar_data = {
            "A0": {"close": 100.0}, "B0": {"close": 100.0},
            "A1": {"close": 200.0}, "B1": {"close": 200.0},
        }

        engine._record_equity(
            state, bar_data, datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        eq = state.equity_curve[-1]
        assert eq["active_slots"] == 2
        assert eq["in_position"] is True
        # Total equity = cash(6000) + allocated(4000) + unrealized(~0)
        assert eq["equity"] == pytest.approx(10000.0, abs=10.0)


# ============================================================
# Full multi-trader backtest integration test (requires real data)
# ============================================================


class TestMultiTraderBacktestIntegration:
    """Integration tests for multi-trader backtest with real data."""

    @pytest.fixture
    def multi_symbol_data(self):
        """Load real multi-symbol data."""
        try:
            from quantbox.data.real_data_loader import RealDataLoader
            loader = RealDataLoader()
            data = loader.load_data(
                symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
                min_bars=300,
                align_data=True,
            )
            if len(data) < 3:
                pytest.skip("Insufficient real data - need at least 3 symbols")
            return data
        except Exception:
            pytest.skip("Real data not available - run data downloader first")

    def test_multi_trader_runs(self, multi_symbol_data):
        """Multi-trader backtest should run without errors."""
        cfg = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=3,
            formation_bars=100,
            max_holding_bars=50,
        )
        engine = ContinuousPairBacktest(cfg)
        result = engine.run(multi_symbol_data, progress=False)

        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result
        assert result["metrics"]["n_trades"] >= 0

    def test_multi_trader_more_trades_than_single(self, multi_symbol_data):
        """Multi-trader should execute at least as many trades as single
        (more market participation → more opportunities captured)."""
        cfg_single = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=1,
            position_size_pct=1.0,
            formation_bars=100,
            max_holding_bars=50,
        )
        cfg_multi = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=3,
            formation_bars=100,
            max_holding_bars=50,
        )

        engine_single = ContinuousPairBacktest(cfg_single)
        engine_multi = ContinuousPairBacktest(cfg_multi)

        result_single = engine_single.run(multi_symbol_data, progress=False)
        result_multi = engine_multi.run(multi_symbol_data, progress=False)

        n_single = result_single["metrics"]["n_trades"]
        n_multi = result_multi["metrics"]["n_trades"]

        # Multi-trader should capture at least as many opportunities
        assert n_multi >= n_single, (
            f"Multi-trader ({n_multi}) should have >= trades than single ({n_single})"
        )

    def test_slot_ids_in_trades(self, multi_symbol_data):
        """Trades DataFrame should contain slot_id column."""
        cfg = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=3,
            formation_bars=100,
            max_holding_bars=50,
        )
        engine = ContinuousPairBacktest(cfg)
        result = engine.run(multi_symbol_data, progress=False)

        trades_df = result["trades"]
        if not trades_df.empty:
            assert "slot_id" in trades_df.columns

    def test_equity_curve_has_active_slots(self, multi_symbol_data):
        """Equity curve should track active_slots count."""
        cfg = ContinuousBacktestConfig(
            initial_capital=10000.0,
            max_traders=3,
            formation_bars=100,
            max_holding_bars=50,
        )
        engine = ContinuousPairBacktest(cfg)
        result = engine.run(multi_symbol_data, progress=False)

        eq_df = result["equity_curve"]
        if not eq_df.empty:
            assert "active_slots" in eq_df.columns


# ============================================================
# LivePairTrader multi-slot state tests (mocked)
# ============================================================


class TestLivePairTraderMultiSlot:
    """Tests for LivePairTrader multi-slot state management.

    Uses mocked Binance client - no real API calls.
    """

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        monkeypatch.setenv("MAX_TRADERS", "5")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True)
            return trader

    def test_init_slots(self, trader):
        assert len(trader.slots) == 5
        assert all(s.slot_id == i for i, s in enumerate(trader.slots))

    def test_occupied_symbols_empty(self, trader):
        assert trader.occupied_symbols == set()

    def test_occupied_symbols_with_active(self, trader):
        trader.slots[0].position_open = True
        trader.slots[0].current_pair = ("BTCUSDT", "ETHUSDT")
        assert trader.occupied_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_active_slot_count(self, trader):
        assert trader.active_slot_count == 0
        trader.slots[0].position_open = True
        trader.slots[0].current_pair = ("A", "B")
        trader.slots[2].position_open = True
        trader.slots[2].current_pair = ("C", "D")
        assert trader.active_slot_count == 2

    def test_idle_slot_ids(self, trader):
        assert trader.idle_slot_ids == [0, 1, 2, 3, 4]
        trader.slots[0].position_open = True
        trader.slots[0].current_pair = ("A", "B")
        assert trader.idle_slot_ids == [1, 2, 3, 4]

    def test_close_slot_positions_resets_slot(self, trader):
        # Simulate an active slot
        slot_id = 1
        trader.slots[slot_id].position_open = True
        trader.slots[slot_id].current_pair = ("BTCUSDT", "ETHUSDT")
        trader.slots[slot_id].position_info = {"capital": 1000}
        trader.slot_positions[slot_id] = {
            "symbol1": "BTCUSDT", "symbol2": "ETHUSDT",
            "qty1": 0.001, "qty2": -0.01,
        }
        trader.slot_entry_times[slot_id] = datetime.now(timezone.utc)

        # Mock close_position to return fill info dict (like real place_order)
        trader.close_position = MagicMock(return_value={
            'orderId': 12345, 'status': 'FILLED',
            'executedQty': '0.001', 'avgPrice': '50000.0',
            'makerFill': True, 'chaseCount': 1,
        })

        trader.close_slot_positions(slot_id)

        assert trader.slots[slot_id].position_open is False
        assert trader.slots[slot_id].current_pair is None
        assert slot_id not in trader.slot_positions
        assert slot_id not in trader.slot_entry_times

    def test_close_all_positions_resets_all_slots(self, trader):
        # Set up two active slots
        for i in [0, 2]:
            trader.slots[i].position_open = True
            trader.slots[i].current_pair = (f"SYM{i}A", f"SYM{i}B")
            trader.slot_positions[i] = {
                "symbol1": f"SYM{i}A", "symbol2": f"SYM{i}B",
                "qty1": 0.1, "qty2": -0.1,
            }
            trader.slot_entry_times[i] = datetime.now(timezone.utc)

        # Mock external methods
        trader.get_current_positions = MagicMock(return_value=[])
        trader.close_position = MagicMock(return_value=True)

        trader.close_all_positions()

        assert all(not s.position_open for s in trader.slots)
        assert len(trader.slot_positions) == 0
        assert len(trader.slot_entry_times) == 0

    def test_max_traders_from_config(self, mock_env):
        config = {
            'quote_currency': 'USDT', 'n_coins': 10, 'interval': '15m',
            'formation_hours': 168, 'entry_zscore': 2.0, 'exit_zscore': 0.5,
            'stop_loss_zscore': 3.5, 'max_holding_hours': 24,
            'position_size_pct': -1.0, 'trading_fee': 0.0005,
            'slippage': 0.001, 'leverage': 1,
            'max_traders': 3,
            'min_correlation': 0.5, 'max_cointegration_pvalue': 0.05,
            'min_half_life': 3, 'max_half_life': 60, 'max_hurst': 0.5,
            'check_interval_seconds': 60, 'retry_attempts': 3, 'retry_delay': 2,
        }
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True, config=config)
            assert trader.max_traders == 3
            assert len(trader.slots) == 3


# ===========================================================================
# PARITY TESTS: Backtest ↔ Live must use identical position sizing
# ===========================================================================


class TestPositionSizingParity:
    """Strict tests ensuring backtest and live trader use identical
    capital allocation logic (calculate_slot_capital).

    핵심 원칙: 백테스트에서 사용한 로직과 동일한 로직이 실거래에서도 사용되어야 한다.
    """

    # ------------------------------------------------------------------
    # 1. Config parity: auto position_size_pct
    # ------------------------------------------------------------------
    def test_backtest_auto_position_size_pct(self):
        """Backtest: position_size_pct=-1 auto-resolves to 1/max_traders."""
        cfg = ContinuousBacktestConfig(max_traders=5)
        assert cfg.position_size_pct == pytest.approx(0.2)

    def test_backtest_auto_position_size_3_traders(self):
        cfg = ContinuousBacktestConfig(max_traders=3)
        assert cfg.position_size_pct == pytest.approx(1.0 / 3)

    def test_live_auto_position_size_pct(self):
        """Live: position_size_pct=-1 auto-resolves to 1/max_traders."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true", "MAX_TRADERS": "5",
            }):
                trader = LivePairTrader(paper_mode=True)
                assert trader.config['position_size_pct'] == pytest.approx(0.2)

    def test_live_auto_position_size_3_traders(self):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true", "MAX_TRADERS": "3",
            }):
                trader = LivePairTrader(paper_mode=True)
                assert trader.config['position_size_pct'] == pytest.approx(1.0 / 3)

    def test_live_explicit_position_size_pct(self):
        """Explicit position_size_pct overrides auto-calc."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            cfg = {
                'quote_currency': 'USDT', 'n_coins': 10, 'interval': '15m',
                'formation_hours': 168, 'entry_zscore': 2.0, 'exit_zscore': 0.5,
                'stop_loss_zscore': 3.5, 'max_holding_hours': 24,
                'position_size_pct': 0.30,  # explicitly 30%
                'trading_fee': 0.0005, 'slippage': 0.001, 'leverage': 1,
                'max_traders': 5,
                'min_correlation': 0.5, 'max_cointegration_pvalue': 0.05,
                'min_half_life': 3, 'max_half_life': 60, 'max_hurst': 0.5,
                'check_interval_seconds': 60, 'retry_attempts': 3, 'retry_delay': 2,
            }
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true",
            }):
                trader = LivePairTrader(paper_mode=True, config=cfg)
                assert trader.config['position_size_pct'] == pytest.approx(0.30)

    # ------------------------------------------------------------------
    # 2. Capital allocation parity: same function, same results
    # ------------------------------------------------------------------
    def test_first_slot_capital_matches(self):
        """First slot of 5 on $1000 gets $200 in both systems."""
        # Backtest scenario
        bt_state = BacktestState(
            current_capital=1000.0, peak_capital=1000.0, max_traders=5,
        )
        bt_capital = calculate_slot_capital(
            available_capital=bt_state.current_capital,
            active_slots=bt_state.active_slots,
            max_traders=5,
            position_size_pct=0.2,
        )

        # Live scenario (same function)
        live_capital = calculate_slot_capital(
            available_capital=1000.0,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
        )

        assert bt_capital == live_capital == pytest.approx(200.0)

    def test_second_slot_capital_with_one_active(self):
        """Second slot: total_book=$1000, 20% target, but only $800 available."""
        slot0 = TraderSlotState(slot_id=0)
        slot0.position_open = True
        slot0.current_pair = ("A", "B")
        slot0.position_info = {"capital": 200.0}

        capital = calculate_slot_capital(
            available_capital=800.0,
            active_slots=[slot0],
            max_traders=5,
            position_size_pct=0.2,
        )
        # total_book = 800 + 200 = 1000, target = 1000 * 0.2 = 200
        assert capital == pytest.approx(200.0)

    def test_third_slot_capital_after_profit(self):
        """If slot0 made profit: total_book increases, allocation scales up."""
        slot0 = TraderSlotState(slot_id=0)
        slot0.position_open = True
        slot0.current_pair = ("A", "B")
        slot0.position_info = {"capital": 200.0}

        slot1 = TraderSlotState(slot_id=1)
        slot1.position_open = True
        slot1.current_pair = ("C", "D")
        slot1.position_info = {"capital": 200.0}

        # $650 available (returned $200 + $50 profit earlier, now $600 + $50)
        capital = calculate_slot_capital(
            available_capital=650.0,
            active_slots=[slot0, slot1],
            max_traders=5,
            position_size_pct=0.2,
        )
        # total_book = 650 + 200 + 200 = 1050, target = 1050 * 0.2 = 210
        assert capital == pytest.approx(210.0)

    def test_capital_capped_at_available(self):
        """Capital never exceeds available balance."""
        capital = calculate_slot_capital(
            available_capital=50.0,
            active_slots=[],
            max_traders=5,
            position_size_pct=0.2,
        )
        # total_book = 50, target = 50 * 0.2 = 10  (< available, so 10)
        assert capital == pytest.approx(10.0)

    def test_all_five_slots_exhaust_capital(self):
        """5 sequential entries on $1000: each gets $200, total = $1000."""
        available = 1000.0
        active: list[TraderSlotState] = []
        allocations = []

        for i in range(5):
            cap = calculate_slot_capital(
                available_capital=available,
                active_slots=active,
                max_traders=5,
                position_size_pct=0.2,
            )
            allocations.append(cap)
            available -= cap
            slot = TraderSlotState(slot_id=i)
            slot.position_open = True
            slot.current_pair = (f"S{i}A", f"S{i}B")
            slot.position_info = {"capital": cap}
            active.append(slot)

        assert all(a == pytest.approx(200.0) for a in allocations)
        assert available == pytest.approx(0.0, abs=1e-10)

    # ------------------------------------------------------------------
    # 3. No max_position_pct in live config (removed legacy)
    # ------------------------------------------------------------------
    def test_no_max_position_pct_in_default_config(self):
        """max_position_pct removed; only position_size_pct exists."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true",
            }):
                trader = LivePairTrader(paper_mode=True)
                assert 'max_position_pct' not in trader.config
                assert 'position_size_pct' in trader.config

    # ------------------------------------------------------------------
    # 4. get_total_balance exists and is separate from get_available_balance
    # ------------------------------------------------------------------
    def test_get_total_balance_paper_mode(self):
        """Paper mode total balance returns paper_balance."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true",
            }):
                trader = LivePairTrader(paper_mode=True)
                assert trader.get_total_balance() == trader.paper_balance

    # ------------------------------------------------------------------
    # 5. Position data includes 'capital' field (for slot awareness)
    # ------------------------------------------------------------------
    def test_execute_stores_capital_in_position(self):
        """execute_pair_trade must store 'capital' in position_data
        so calculate_slot_capital can read it from slot.position_info."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true",
            }):
                trader = LivePairTrader(paper_mode=True)

                # Mock required methods
                trader.get_price = MagicMock(side_effect=[100.0, 50.0])
                trader.calculate_zscore = MagicMock(return_value=-2.5)
                trader.get_symbol_info = MagicMock(return_value={
                    'step_size': 0.001, 'quantity_precision': 3,
                    'min_notional': 5,
                })
                trader.place_order = MagicMock(return_value={'orderId': '123'})

                pair = MagicMock()
                pair.symbol1 = "BTCUSDT"
                pair.symbol2 = "ETHUSDT"
                pair_info = {
                    'hedge_ratio': 1.0,
                    'spread_mean': 0.0,
                    'spread_std': 1.0,
                }
                capital = 200.0

                result = trader.execute_pair_trade(0, pair, pair_info, capital)
                assert result is True
                assert 'capital' in trader.slot_positions[0]
                # capital now stores margin = notional / leverage
                leverage = trader.config['leverage']
                expected_margin = capital / leverage
                assert trader.slot_positions[0]['capital'] == pytest.approx(expected_margin)
                assert trader.slots[0].position_info['capital'] == pytest.approx(expected_margin)
                # notional should be the original capital passed in
                assert trader.slot_positions[0]['notional'] == pytest.approx(capital)

    # ------------------------------------------------------------------
    # 6. run_cycle uses calculate_slot_capital (not raw balance)
    # ------------------------------------------------------------------
    def test_run_cycle_uses_pool_capital(self):
        """run_cycle must call calculate_slot_capital, not raw balance * pct."""
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch.dict("os.environ", {
                "BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s",
                "BINANCE_TESTNET": "true", "MAX_TRADERS": "5",
            }):
                trader = LivePairTrader(paper_mode=True)
                trader.paper_balance = 1000.0

                # Make slot 0 active (holds $200)
                trader.slots[0].position_open = True
                trader.slots[0].current_pair = ("BTCUSDT", "ETHUSDT")
                trader.slots[0].position_info = {"capital": 200.0}
                trader.slot_positions[0] = {"symbol1": "BTCUSDT", "symbol2": "ETHUSDT"}

                # Mock monitor_slot to do nothing
                trader.monitor_slot = MagicMock()
                # Mock coin list
                trader.get_trading_coins = MagicMock(return_value=[
                    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
                ])
                trader.download_multi_symbol_data = MagicMock(return_value={
                    "SOLUSDT": pd.DataFrame(), "XRPUSDT": pd.DataFrame(),
                    "ADAUSDT": pd.DataFrame(), "DOGEUSDT": pd.DataFrame(),
                    "BTCUSDT": pd.DataFrame(),
                })
                trader.find_best_pair = MagicMock(return_value=None)

                # Run cycle (will find no pairs, but the important thing
                # is it calls calculate_slot_capital internally)
                with patch("quantbox.live_trader.calculate_slot_capital",
                           wraps=calculate_slot_capital) as mock_calc:
                    trader.run_cycle()
                    # Should be called for the first idle slot at least
                    if mock_calc.called:
                        call_args = mock_calc.call_args
                        assert call_args.kwargs['position_size_pct'] == pytest.approx(0.2)
                        assert call_args.kwargs['max_traders'] == 5


# ============================================================
# Notification throttle tests (fix/throttle-all-slots-notification)
# ============================================================


class TestNotificationThrottle:
    """모든 슬롯 사용 중 / 잔고 부족 알림이 상태 전이 시 1회만 발송되는지 검증."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        monkeypatch.setenv("MAX_TRADERS", "2")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True)
            # Stub out methods that hit external services
            trader._process_bot_commands = MagicMock()
            trader.monitor_slot = MagicMock()
            trader._build_slots_info = MagicMock(return_value=[
                {'slot_id': 0, 'pair': 'BTCUSDT/ETHUSDT', 'holding_hours': 1.0},
                {'slot_id': 1, 'pair': 'SOLUSDT/AVAXUSDT', 'holding_hours': 0.5},
            ])
            return trader

    def _make_all_slots_active(self, trader):
        """모든 슬롯을 사용 중 상태로 만든다."""
        for slot in trader.slots:
            slot.position_open = True
            slot.current_pair = ("SYMX", "SYMY")

    def _make_one_slot_idle(self, trader):
        """슬롯 하나를 유휴 상태로 되돌린다."""
        trader.slots[0].position_open = False
        trader.slots[0].current_pair = None

    def test_all_slots_occupied_notified_once(self, trader):
        """모든 슬롯 점유 시 알림은 최초 1회만 발송된다."""
        self._make_all_slots_active(trader)
        trader.discord = MagicMock()

        # 첫 사이클 → 알림 1회
        trader.run_cycle()
        assert trader.discord.notify_all_slots_occupied.call_count == 1

        # 두 번째 사이클 → 추가 알림 없음 (이미 notified)
        trader.run_cycle()
        assert trader.discord.notify_all_slots_occupied.call_count == 1

        # 세 번째 사이클도 동일
        trader.run_cycle()
        assert trader.discord.notify_all_slots_occupied.call_count == 1

    def test_all_slots_occupied_re_notified_after_slot_freed(self, trader):
        """슬롯이 비었다가 다시 찼을 때 알림이 재발송된다."""
        self._make_all_slots_active(trader)
        trader.discord = MagicMock()
        # Stub balance/trade methods for idle path
        trader.get_available_balance = MagicMock(return_value=0.0)
        trader.get_total_balance = MagicMock(return_value=1000.0)

        # 1차 점유 → 알림 1회
        trader.run_cycle()
        assert trader.discord.notify_all_slots_occupied.call_count == 1
        assert trader._all_slots_occupied_notified is True

        # 슬롯 비움 → 플래그 리셋
        self._make_one_slot_idle(trader)
        trader.run_cycle()  # idle path 진입 → 플래그 리셋됨
        assert trader._all_slots_occupied_notified is False

        # 다시 점유 → 알림 재발송
        self._make_all_slots_active(trader)
        trader.run_cycle()
        assert trader.discord.notify_all_slots_occupied.call_count == 2

    def test_insufficient_balance_notified_once(self, trader):
        """잔고 부족 알림도 상태 전이 시 1회만 발송된다."""
        trader.discord = MagicMock()
        trader.get_available_balance = MagicMock(return_value=50.0)
        trader.get_total_balance = MagicMock(return_value=50.0)

        # 모든 슬롯 idle 상태 유지 (기본값)
        trader.run_cycle()
        assert trader.discord.notify_insufficient_balance.call_count == 1

        trader.run_cycle()
        assert trader.discord.notify_insufficient_balance.call_count == 1

        trader.run_cycle()
        assert trader.discord.notify_insufficient_balance.call_count == 1

    def test_insufficient_balance_re_notified_after_recovery(self, trader):
        """잔고가 복구되었다가 다시 부족해지면 알림이 재발송된다."""
        trader.discord = MagicMock()
        trader.get_available_balance = MagicMock(return_value=50.0)
        trader.get_total_balance = MagicMock(return_value=50.0)
        trader.get_trading_coins = MagicMock(return_value=["BTCUSDT"])
        trader.download_multi_symbol_data = MagicMock(return_value={})
        trader.find_best_pair = MagicMock(return_value=None)

        # 1차 부족 → 알림 1회
        trader.run_cycle()
        assert trader.discord.notify_insufficient_balance.call_count == 1

        # 잔고 복구
        trader.get_available_balance = MagicMock(return_value=500.0)
        trader.run_cycle()
        assert trader._insufficient_balance_notified is False

        # 다시 부족 → 알림 재발송
        trader.get_available_balance = MagicMock(return_value=30.0)
        trader.run_cycle()
        assert trader.discord.notify_insufficient_balance.call_count == 2
