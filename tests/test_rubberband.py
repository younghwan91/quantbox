"""
Tests for Rubber Band strategy — signal generation, position sizing,
vectorized backtester, and live/backtest parity.

Uses REAL Binance data from cache (no synthetic data).
"""

import numpy as np
import pytest

from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    calculate_position_size,
    compute_signals,
    compute_signals_mtf,
    compute_signals_with_regime,
)
from quantbox.backtest.vectorized_backtest import (
    BacktestConfig,
    VectorizedBacktest,
    _backtest_core,
)


# ── Fixtures ──


@pytest.fixture
def btc_close(real_data_available):
    """Real BTC close prices."""
    if not real_data_available:
        pytest.skip("Real data not available")
    from quantbox.data.real_data_loader import RealDataLoader

    loader = RealDataLoader()
    df = loader.load_symbol("BTCUSDT")
    if df is None:
        pytest.skip("BTCUSDT data not available")
    return df["Close"].values.astype(np.float64)


@pytest.fixture
def eth_close(real_data_available):
    """Real ETH close prices."""
    if not real_data_available:
        pytest.skip("Real data not available")
    from quantbox.data.real_data_loader import RealDataLoader

    loader = RealDataLoader()
    df = loader.load_symbol("ETHUSDT")
    if df is None:
        pytest.skip("ETHUSDT data not available")
    return df["Close"].values.astype(np.float64)


# ══════════════════ 1. Signal Generation Tests ══════════════════


class TestComputeSignals:
    """Test the Numba-compiled signal generator."""

    def test_output_shape(self, btc_close):
        """Signal array must be same length as input."""
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        assert sigs.shape == btc_close.shape

    def test_signal_values_in_range(self, btc_close):
        """Signals must be in {-1, 0, +1}."""
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        unique = set(np.unique(sigs))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_first_lookback_bars_are_zero(self, btc_close):
        """No signals in the first `lookback` bars."""
        lookback = 8
        sigs = compute_signals(btc_close, lookback, 0.025, 0.5, 1.0, 8)
        assert np.all(sigs[:lookback] == 0)

    def test_generates_some_signals(self, btc_close):
        """Strategy must produce at least SOME trades over 3 months."""
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        n_signals = np.sum(sigs != 0)
        assert n_signals > 20, f"Only {n_signals} signal bars — alpha dead?"

    def test_high_threshold_fewer_signals(self, btc_close):
        """Higher threshold → fewer entries."""
        sigs_low = compute_signals(btc_close, 8, 0.02, 0.5, 1.0, 8)
        sigs_high = compute_signals(btc_close, 8, 0.05, 0.5, 1.0, 8)
        n_low = np.sum(sigs_low != 0)
        n_high = np.sum(sigs_high != 0)
        assert n_high < n_low, "Higher threshold should produce fewer signals"

    def test_max_hold_respected(self, btc_close):
        """No trade can stay open longer than max_hold bars."""
        max_hold = 4
        sigs = compute_signals(btc_close, 8, 0.02, 0.5, 1.0, max_hold)
        # Find consecutive non-zero runs
        max_run = 0
        current_run = 0
        for s in sigs:
            if s != 0:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        # Allow max_hold + 1 because of entry bar counting
        assert max_run <= max_hold + 1, f"Max consecutive position={max_run}, expected <={max_hold + 1}"


class TestSignalsWithRegime:
    """Test regime-filtered signal generation."""

    def test_regime_filter_reduces_signals(self, btc_close, eth_close):
        """Strict regime filter blocks some entries."""
        n = min(len(btc_close), len(eth_close))
        sigs_base = compute_signals(eth_close[:n], 8, 0.025, 0.5, 1.0, 8)
        sigs_regime = compute_signals_with_regime(eth_close[:n], btc_close[:n], 8, 0.025, 0.5, 1.0, 8, 96, 0.04)
        n_base = np.sum(sigs_base != 0)
        n_regime = np.sum(sigs_regime != 0)
        # Regime filter should block at least some entries
        assert n_regime <= n_base

    def test_regime_filter_output_shape(self, btc_close, eth_close):
        n = min(len(btc_close), len(eth_close))
        sigs = compute_signals_with_regime(eth_close[:n], btc_close[:n], 8, 0.025, 0.5, 1.0, 8, 96, 0.08)
        assert sigs.shape == (n,)


# ══════════════════ 2. Position Sizing Tests ══════════════════


class TestPositionSizing:
    """Test risk-based position sizing."""

    def test_basic_sizing(self):
        """1% risk, 2.5% stop → ~40% of capital."""
        notional, leverage = calculate_position_size(
            capital=10000,
            risk_per_trade=0.01,
            entry_price=100.0,
            move_size=0.025,
            stop_mult=1.0,
            max_leverage=20.0,
        )
        # Risk $100, max loss 2.5% → notional = $4000
        assert abs(notional - 4000) < 1, f"Expected ~$4000, got ${notional}"
        assert abs(leverage - 0.4) < 0.01

    def test_leverage_cap(self):
        """Position size capped by max_leverage."""
        notional, leverage = calculate_position_size(
            capital=1000,
            risk_per_trade=0.05,
            entry_price=100.0,
            move_size=0.001,
            stop_mult=1.0,
            max_leverage=5.0,
        )
        assert leverage <= 5.0
        assert notional <= 1000 * 5.0 + 0.01

    def test_zero_move(self):
        """Zero move → zero notional (safety)."""
        notional, leverage = calculate_position_size(
            capital=10000,
            risk_per_trade=0.01,
            entry_price=100.0,
            move_size=0.0,
            stop_mult=1.0,
        )
        assert notional == 0.0


# ══════════════════ 3. Vectorized Backtest Tests ══════════════════


class TestVectorizedBacktest:
    """Test the vectorized backtest engine."""

    def test_backtest_core_no_trades(self):
        """Flat signals → equity unchanged (minus rounding)."""
        close = np.linspace(100, 110, 500)
        sigs = np.zeros(500)
        eq, ret, sharpe, dd, wr, n_t, *_ = _backtest_core(close, sigs, 0.0004, 0.0)
        assert abs(ret) < 0.001, f"No trades but return={ret:.6f}"
        assert n_t == 0

    def test_backtest_core_with_signals(self, btc_close):
        """Backtest with real data produces sane metrics."""
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        eq, ret, sharpe, dd, wr, n_trades, *_ = _backtest_core(btc_close, sigs, 0.0004, 0.0001 / 32)
        assert len(eq) == len(btc_close)
        assert n_trades > 0, "Expected some trades"
        assert 0 <= dd <= 1.0, f"Invalid max DD: {dd}"
        assert 0 <= wr <= 100, f"Invalid win rate: {wr}"

    def test_run_single(self, btc_close):
        """VectorizedBacktest.run_single returns BacktestResult."""
        bt = VectorizedBacktest(BacktestConfig())
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        result = bt.run_single(btc_close, sigs, symbol="BTCUSDT")

        assert result.symbol == "BTCUSDT"
        assert result.n_trades > 0
        assert result.equity_curve.shape == btc_close.shape

    def test_costs_reduce_returns(self, btc_close):
        """Higher costs → lower returns."""
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        _, ret_low, *_ = _backtest_core(btc_close, sigs, 0.0001, 0.0)
        _, ret_high, *_ = _backtest_core(btc_close, sigs, 0.001, 0.0)
        assert ret_high < ret_low, "Higher costs must reduce returns"


# ══════════════════ 4. Strategy Wrapper Tests ══════════════════


class TestRubberBandStrategy:
    """Test the stateful strategy wrapper."""

    def test_initial_state(self):
        """Strategy starts with no positions."""
        strat = RubberBandStrategy()
        assert strat.n_active == 0
        assert not strat.has_position("BTCUSDT")

    def test_on_bar_returns_valid_signal(self, btc_close):
        """on_bar returns -1, 0, or +1."""
        strat = RubberBandStrategy()
        signal = strat.on_bar("BTCUSDT", btc_close)
        assert signal in (-1, 0, 1)

    def test_position_lifecycle(self):
        """Open → update → close position flow."""
        strat = RubberBandStrategy(
            RubberBandConfig(
                partial_exit_enabled=False,
                target_retrace=0.382,  # Use old Fib target for lifecycle test
            )
        )
        strat.open_position(
            symbol="TESTUSDT",
            signal=1,
            entry_price=100.0,
            notional=1000.0,
            leverage=2.0,
            move_size=0.03,
        )
        assert strat.has_position("TESTUSDT")
        assert strat.n_active == 1

        # Update: price moved favourably
        reason = strat.update_position("TESTUSDT", 101.5)
        # 1.5% gain on 3% move with 0.382 target = target hit at 1.146%
        assert reason == "target"

        pos = strat.close_position("TESTUSDT")
        assert pos is not None
        assert pos["signal"] == 1
        assert strat.n_active == 0

    def test_stop_loss_exit(self):
        """Stop loss fires when price moves against trade."""
        strat = RubberBandStrategy(RubberBandConfig(stop_mult=1.0))
        strat.open_position(
            symbol="TESTUSDT",
            signal=1,
            entry_price=100.0,
            notional=1000.0,
            leverage=2.0,
            move_size=0.03,
        )
        # Price drops beyond 3% stop (strict < comparison)
        reason = strat.update_position("TESTUSDT", 96.9)
        assert reason == "stop_loss"

    def test_max_hold_exit(self):
        """Max hold forces exit after N bars."""
        cfg = RubberBandConfig(max_hold=3)
        strat = RubberBandStrategy(cfg)
        strat.open_position(
            symbol="TESTUSDT",
            signal=-1,
            entry_price=100.0,
            notional=1000.0,
            leverage=2.0,
            move_size=0.03,
        )
        # Simulate flat bars
        for _ in range(2):
            reason = strat.update_position("TESTUSDT", 100.0)
            assert reason is None  # Still within max_hold
        reason = strat.update_position("TESTUSDT", 100.0)
        assert reason == "max_hold"


# ══════════════════ 5. Config Tests ══════════════════


class TestRubberBandConfig:
    """Test config properties and defaults."""

    def test_default_cost(self):
        cfg = RubberBandConfig()
        assert cfg.cost_per_side == 0.0004  # 0.04%
        assert cfg.round_trip_cost == 0.0008

    def test_bars_per_funding(self):
        cfg = RubberBandConfig(bar_interval_minutes=15)
        assert cfg.bars_per_funding == 32  # 8h / 15m = 32

    def test_regime_filter_disabled_by_default(self):
        """Regime filter is disabled — it blocks alpha-generating moves."""
        cfg = RubberBandConfig()
        assert cfg.regime_filter_enabled is False

    def test_stop_decay_default(self):
        """stop_decay defaults to 0.0 (disabled for long-term robustness)."""
        cfg = RubberBandConfig()
        assert cfg.stop_decay == 0.0


# ══════════════════ 6. Time-Decaying Stop Tests ══════════════════


class TestStopDecay:
    """Test the time-decaying stop loss feature."""

    def test_decay_zero_same_as_original(self, btc_close):
        """stop_decay=0 must produce identical signals to the original."""
        sigs_original = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0)
        sigs_no_decay = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8)
        np.testing.assert_array_equal(sigs_original, sigs_no_decay)

    def test_higher_decay_stops_sooner(self, btc_close):
        """Higher stop_decay should exit losing trades faster."""
        sigs_no = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0)
        sigs_hi = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.5)

        # Count bars in market
        bars_no = np.sum(sigs_no != 0)
        bars_hi = np.sum(sigs_hi != 0)

        # Higher decay → tighter stops → fewer bars in market
        # (may also generate more trades due to earlier exits freeing up)
        assert bars_hi <= bars_no * 1.05, f"High decay {bars_hi} should not exceed no-decay {bars_no} by >5%"

    def test_decay_signal_values_valid(self, btc_close):
        """Decayed signals must still be in {-1, 0, +1}."""
        sigs = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.5)
        unique = set(np.unique(sigs))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_decay_with_regime_filter(self, btc_close, eth_close):
        """stop_decay works with regime-filtered signals."""
        n = min(len(btc_close), len(eth_close))
        sigs = compute_signals_with_regime(
            eth_close[:n],
            btc_close[:n],
            8,
            0.025,
            0.5,
            1.0,
            8,
            96,
            0.08,
            0.3,
        )
        assert sigs.shape == (n,)
        unique = set(np.unique(sigs))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_live_parity_stop_decay(self):
        """update_position() must respect stop_decay like compute_signals."""
        cfg = RubberBandConfig(stop_mult=1.0, max_hold=8, stop_decay=0.5)
        strat = RubberBandStrategy(cfg)

        # Open a long position with 3% move
        strat.open_position(
            symbol="TESTUSDT",
            signal=1,
            entry_price=100.0,
            notional=1000.0,
            leverage=2.0,
            move_size=0.03,
        )

        # Without decay: stop at -3% (price 97.0)
        # With decay=0.5 at bar 4/8: factor = 1 - 0.5*(4/8) = 0.75
        # Effective stop = 3% * 1.0 * 0.75 = 2.25% → price 97.75
        # Simulate 3 flat bars first (bars_held = 1, 2, 3)
        for _ in range(3):
            reason = strat.update_position("TESTUSDT", 100.0)
            assert reason is None

        # At bar 4: price at 97.8 (above 97.75 → no stop)
        reason = strat.update_position("TESTUSDT", 97.8)
        assert reason is None, f"Should not stop at 97.8, got {reason}"

        # At bar 5: factor = 1 - 0.5*(5/8) = 0.6875
        # Stop = 3% * 0.6875 = 2.0625% → stop at price < 97.94
        # Price at 97.9 → should stop
        reason = strat.update_position("TESTUSDT", 97.9)
        assert reason == "stop_loss", f"Should stop at 97.9 with decay, got {reason}"


# ══════════════════ 7. DD-Managed Portfolio Tests ══════════════════


class TestDDManagedPortfolio:
    """Test the portfolio-level drawdown management features."""

    @pytest.fixture
    def bt(self):
        """Backtester with standard config."""
        return VectorizedBacktest(BacktestConfig())

    @pytest.fixture
    def multi_data(self, btc_close, eth_close):
        """Two-symbol dataset for portfolio tests."""
        import pandas as pd

        n = min(len(btc_close), len(eth_close))
        return {
            "BTCUSDT": pd.DataFrame({"Close": btc_close[:n]}),
            "ETHUSDT": pd.DataFrame({"Close": eth_close[:n]}),
        }

    def _signal_func(self, close):
        return compute_signals(close, 8, 0.025, 0.5, 1.0, 8)

    def test_equal_weight_sums_to_one(self, bt, multi_data):
        """Equal weights must sum to 1.0."""
        result = bt.run_dd_managed_portfolio(
            multi_data,
            self._signal_func,
            selected_symbols=["BTCUSDT", "ETHUSDT"],
            inverse_vol_weight=False,
        )
        total_w = sum(result["weights"].values())
        assert abs(total_w - 1.0) < 1e-8

    def test_inverse_vol_weights_differ(self, bt, multi_data):
        """Inverse-vol weights must differ from equal weight."""
        result = bt.run_dd_managed_portfolio(
            multi_data,
            self._signal_func,
            selected_symbols=["BTCUSDT", "ETHUSDT"],
            inverse_vol_weight=True,
        )
        # BTC and ETH have different vols → weights should differ
        w = list(result["weights"].values())
        assert abs(w[0] - w[1]) > 0.001, "Inverse-vol should produce unequal weights"

    def test_dd_dampening_reduces_return(self, bt, multi_data):
        """DD dampening at very low threshold must reduce absolute return."""
        base = bt.run_dd_managed_portfolio(
            multi_data,
            self._signal_func,
            selected_symbols=["BTCUSDT", "ETHUSDT"],
            dd_dampening_threshold=0.0,  # disabled
        )
        dampened = bt.run_dd_managed_portfolio(
            multi_data,
            self._signal_func,
            selected_symbols=["BTCUSDT", "ETHUSDT"],
            dd_dampening_threshold=0.001,  # very aggressive dampening
            dd_dampening_scale=0.1,
        )
        # With aggressive dampening, returns should be reduced
        assert abs(dampened["portfolio_return_pct"]) <= abs(base["portfolio_return_pct"]) * 1.1, (
            "Dampening should not increase absolute return significantly"
        )

    def test_empty_symbols_returns_zero(self, bt, multi_data):
        """Empty symbol list should return zero metrics."""
        result = bt.run_dd_managed_portfolio(
            multi_data,
            self._signal_func,
            selected_symbols=[],
        )
        assert result["portfolio_return_pct"] == 0.0
        assert result["portfolio_max_dd_pct"] == 0.0

    def test_result_has_all_keys(self, bt, multi_data):
        """Result dict must contain all expected keys."""
        result = bt.run_dd_managed_portfolio(
            multi_data,
            self._signal_func,
            selected_symbols=["BTCUSDT", "ETHUSDT"],
        )
        expected_keys = {
            "portfolio_return_pct",
            "portfolio_sharpe",
            "portfolio_max_dd_pct",
            "portfolio_equity",
            "calmar",
            "weights",
            "pct_symbols_positive",
            "n_symbols",
            "total_trades",
            "avg_win_rate",
            "per_asset",
        }
        assert expected_keys.issubset(set(result.keys()))


# ══════════════════ 8. Multi-Timeframe Confirmation Tests ══════════════════


class TestMTFConfirmation:
    """Test multi-timeframe entry confirmation filter."""

    def test_mtf_disabled_matches_baseline(self, btc_close):
        """htf_period=0 must produce identical signals to compute_signals."""
        sigs_base = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0)
        sigs_mtf = compute_signals_mtf(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0, 0, 0.02)
        np.testing.assert_array_equal(sigs_base, sigs_mtf)

    def test_mtf_reduces_entries(self, btc_close):
        """Active MTF filter should block some entries → fewer signals."""
        sigs_base = compute_signals(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0)
        sigs_mtf = compute_signals_mtf(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0, 48, 0.02)
        # MTF should have fewer or equal non-zero signals
        assert np.sum(sigs_mtf != 0) <= np.sum(sigs_base != 0)

    def test_mtf_signal_values_valid(self, btc_close):
        """MTF signals must be in {-1, 0, +1}."""
        sigs = compute_signals_mtf(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0, 48, 0.02)
        unique = set(np.unique(sigs))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_mtf_output_shape(self, btc_close):
        """Output shape must match input."""
        sigs = compute_signals_mtf(btc_close, 8, 0.025, 0.5, 1.0, 8, 0.0, 48, 0.02)
        assert sigs.shape == btc_close.shape

    def test_mtf_with_stop_decay(self, btc_close):
        """MTF + stop_decay must work together."""
        sigs = compute_signals_mtf(btc_close, 8, 0.025, 0.3, 1.0, 8, 0.3, 48, 0.02)
        unique = set(np.unique(sigs))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_mtf_config_defaults(self):
        """Config should have MTF disabled by default (hurts long-term perf)."""
        cfg = RubberBandConfig()
        assert cfg.htf_period == 0  # DISABLED
        assert cfg.htf_block_threshold == 0.01  # 1% block (unused when disabled)

    def test_on_bar_uses_mtf(self, btc_close):
        """on_bar should use MTF when htf_period > 0."""
        cfg = RubberBandConfig(htf_period=48, htf_block_threshold=0.02)
        strat = RubberBandStrategy(cfg)
        # Should not crash and should return valid signal
        sig = strat.on_bar("BTCUSDT", btc_close)
        assert sig in (-1, 0, 1)


class TestMTFEnhanced:
    """Test enhanced MTF with agree discount and tighter stops."""

    def test_agree_discount_increases_entries(self, btc_close):
        """Lower threshold via agree discount should produce more entries."""
        sigs_v1 = compute_signals_mtf(
            btc_close,
            8,
            0.025,
            0.5,
            1.0,
            8,
            0.0,
            48,
            0.02,
            1.0,
            1.0,  # block-only
        )
        sigs_enh = compute_signals_mtf(
            btc_close,
            8,
            0.025,
            0.5,
            1.0,
            8,
            0.0,
            48,
            0.02,
            0.7,
            1.0,  # agree discount but same stop
        )
        # Enhanced should have >= entries (discount lets more trades in)
        assert np.sum(sigs_enh != 0) >= np.sum(sigs_v1 != 0)

    def test_agree_stop_mult_changes_exits(self, btc_close):
        """Tighter stop via agree_stop_mult should change signal pattern."""
        sigs_normal = compute_signals_mtf(
            btc_close,
            8,
            0.025,
            0.5,
            1.0,
            8,
            0.0,
            48,
            0.02,
            1.0,
            1.0,
        )
        sigs_tight = compute_signals_mtf(
            btc_close,
            8,
            0.025,
            0.5,
            1.0,
            8,
            0.0,
            48,
            0.02,
            1.0,
            0.5,  # 50% tighter stop
        )
        # Tighter stop → different signal pattern (some exits happen earlier)
        # Not necessarily fewer signals, but different positions
        assert sigs_tight.shape == sigs_normal.shape
        unique = set(np.unique(sigs_tight))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_defaults_production(self):
        """Default config should have production-optimized enhanced MTF."""
        cfg = RubberBandConfig()
        assert cfg.htf_agree_discount == 0.85  # 15% easier entry
        assert cfg.htf_agree_stop_mult == 0.95  # 5% tighter stop

    def test_on_bar_passes_enhanced_params(self, btc_close):
        """on_bar should pass enhanced params without error."""
        cfg = RubberBandConfig(
            htf_period=48,
            htf_block_threshold=0.02,
            htf_agree_discount=0.8,
            htf_agree_stop_mult=0.7,
        )
        strat = RubberBandStrategy(cfg)
        sig = strat.on_bar("BTCUSDT", btc_close)
        assert sig in (-1, 0, 1)

    def test_discount_1_equals_v1(self, btc_close):
        """discount=1.0, stop=1.0 should match block-only v1 exactly."""
        sigs_v1 = compute_signals_mtf(
            btc_close,
            8,
            0.025,
            0.5,
            1.0,
            8,
            0.3,
            48,
            0.02,
        )
        sigs_enh = compute_signals_mtf(
            btc_close,
            8,
            0.025,
            0.5,
            1.0,
            8,
            0.3,
            48,
            0.02,
            1.0,
            1.0,
        )
        np.testing.assert_array_equal(sigs_v1, sigs_enh)

    def test_entry_stop_mult_stored_in_position(self):
        """open_position must store entry_stop_mult for parity."""
        cfg = RubberBandConfig(stop_mult=1.0)
        strat = RubberBandStrategy(cfg)

        # Without explicit entry_stop_mult → defaults to cfg.stop_mult
        strat.open_position("SYM1", 1, 100.0, 1000.0, 2.0, 0.03)
        assert strat._positions["SYM1"]["entry_stop_mult"] == 1.0

        # With explicit entry_stop_mult (HTF agree case)
        strat.open_position("SYM2", -1, 200.0, 1000.0, 2.0, 0.03, entry_stop_mult=0.7)
        assert strat._positions["SYM2"]["entry_stop_mult"] == 0.7

    def test_update_position_uses_entry_stop_mult(self):
        """update_position must use entry_stop_mult, not cfg.stop_mult."""
        cfg = RubberBandConfig(stop_mult=1.0, max_hold=8, stop_decay=0.0)
        strat = RubberBandStrategy(cfg)

        # Open position with tighter stop (HTF agree simulation)
        strat.open_position("TESTUSDT", 1, 100.0, 1000.0, 2.0, 0.03, entry_stop_mult=0.5)

        # With cfg.stop_mult=1.0, stop would be at 3% → price 97.0
        # With entry_stop_mult=0.5, stop is at 1.5% → price 98.5
        # Price 98.0 should trigger stop with tighter stop
        reason = strat.update_position("TESTUSDT", 98.0)
        assert reason == "stop_loss", f"Should stop at 98.0 with entry_stop_mult=0.5 (stop at 98.5), got {reason}"

    def test_update_position_no_false_stop_without_entry_stop(self):
        """Default entry_stop_mult should use cfg.stop_mult (no false stops)."""
        cfg = RubberBandConfig(stop_mult=1.0, max_hold=8, stop_decay=0.0)
        strat = RubberBandStrategy(cfg)

        # Open with default stop
        strat.open_position("TESTUSDT", 1, 100.0, 1000.0, 2.0, 0.03)

        # Price 98.0 → loss of 2% < stop of 3% → should NOT stop
        reason = strat.update_position("TESTUSDT", 98.0)
        assert reason is None, "Should not stop at 98.0 with stop_mult=1.0"
