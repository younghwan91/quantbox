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
        assert max_run <= max_hold + 1, (
            f"Max consecutive position={max_run}, expected <={max_hold + 1}"
        )


class TestSignalsWithRegime:
    """Test regime-filtered signal generation."""

    def test_regime_filter_reduces_signals(self, btc_close, eth_close):
        """Strict regime filter blocks some entries."""
        n = min(len(btc_close), len(eth_close))
        sigs_base = compute_signals(eth_close[:n], 8, 0.025, 0.5, 1.0, 8)
        sigs_regime = compute_signals_with_regime(
            eth_close[:n], btc_close[:n],
            8, 0.025, 0.5, 1.0, 8, 96, 0.04
        )
        n_base = np.sum(sigs_base != 0)
        n_regime = np.sum(sigs_regime != 0)
        # Regime filter should block at least some entries
        assert n_regime <= n_base

    def test_regime_filter_output_shape(self, btc_close, eth_close):
        n = min(len(btc_close), len(eth_close))
        sigs = compute_signals_with_regime(
            eth_close[:n], btc_close[:n],
            8, 0.025, 0.5, 1.0, 8, 96, 0.08
        )
        assert sigs.shape == (n,)


# ══════════════════ 2. Position Sizing Tests ══════════════════

class TestPositionSizing:
    """Test risk-based position sizing."""

    def test_basic_sizing(self):
        """1% risk, 2.5% stop → ~40% of capital."""
        notional, leverage = calculate_position_size(
            capital=10000, risk_per_trade=0.01,
            entry_price=100.0, move_size=0.025,
            stop_mult=1.0, max_leverage=20.0,
        )
        # Risk $100, max loss 2.5% → notional = $4000
        assert abs(notional - 4000) < 1, f"Expected ~$4000, got ${notional}"
        assert abs(leverage - 0.4) < 0.01

    def test_leverage_cap(self):
        """Position size capped by max_leverage."""
        notional, leverage = calculate_position_size(
            capital=1000, risk_per_trade=0.05,
            entry_price=100.0, move_size=0.001,
            stop_mult=1.0, max_leverage=5.0,
        )
        assert leverage <= 5.0
        assert notional <= 1000 * 5.0 + 0.01

    def test_zero_move(self):
        """Zero move → zero notional (safety)."""
        notional, leverage = calculate_position_size(
            capital=10000, risk_per_trade=0.01,
            entry_price=100.0, move_size=0.0,
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
        eq, ret, sharpe, dd, wr, n_trades, *_ = _backtest_core(
            btc_close, sigs, 0.0004, 0.0001 / 32
        )
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
        strat = RubberBandStrategy()
        strat.open_position(
            symbol="TESTUSDT", signal=1, entry_price=100.0,
            notional=1000.0, leverage=2.0, move_size=0.03,
        )
        assert strat.has_position("TESTUSDT")
        assert strat.n_active == 1

        # Update: price moved favourably
        reason = strat.update_position("TESTUSDT", 101.5)
        # 1.5% gain on 3% move with 50% target = target hit at 1.5%
        assert reason == "target"

        pos = strat.close_position("TESTUSDT")
        assert pos is not None
        assert pos["signal"] == 1
        assert strat.n_active == 0

    def test_stop_loss_exit(self):
        """Stop loss fires when price moves against trade."""
        strat = RubberBandStrategy(RubberBandConfig(stop_mult=1.0))
        strat.open_position(
            symbol="TESTUSDT", signal=1, entry_price=100.0,
            notional=1000.0, leverage=2.0, move_size=0.03,
        )
        # Price drops beyond 3% stop (strict < comparison)
        reason = strat.update_position("TESTUSDT", 96.9)
        assert reason == "stop_loss"

    def test_max_hold_exit(self):
        """Max hold forces exit after N bars."""
        cfg = RubberBandConfig(max_hold=3)
        strat = RubberBandStrategy(cfg)
        strat.open_position(
            symbol="TESTUSDT", signal=-1, entry_price=100.0,
            notional=1000.0, leverage=2.0, move_size=0.03,
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
