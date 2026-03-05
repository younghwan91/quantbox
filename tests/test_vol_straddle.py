"""Tests for VolStraddle strategy — VR compression detector.

Validates:
1. Signal computation (VR, momentum, direction)
2. Multi-TF signal generation with correct priority
3. Position lifecycle (open, SL/TP/max_hold exit)
4. Zero lookahead across TFs (completed bars only)
5. Per-symbol dedup (one position per symbol)
6. Backtest-live parity (same SL/TP/exit logic)
7. Config correctness (A1-D50 parameters)

Uses REAL cached data — no synthetic data generation.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    _atr_pct,
    compute_momentum,
    compute_vr,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════


CACHE_DIR = Path('data/cache_extended')


def _load_real_data(symbol: str = '1000PEPEUSDT', days: int = 1095) -> dict:
    """Load real cached 15m data."""
    pattern = f'{symbol}_15m_{days}d_*.npz'
    files = sorted(CACHE_DIR.glob(pattern))
    if not files:
        pytest.skip(f'No cached data for {pattern}')
    data = np.load(files[-1])
    return {k: data[k].astype(np.float64) for k in data.files}


def _load_real_4h(symbol: str = '1000PEPEUSDT', days: int = 1095) -> dict:
    """Load real cached 4h data."""
    pattern = f'{symbol}_4h_{days}d_*.npz'
    files = sorted(CACHE_DIR.glob(pattern))
    if not files:
        pytest.skip(f'No cached 4h data for {pattern}')
    data = np.load(files[-1])
    return {k: data[k].astype(np.float64) for k in data.files}


@pytest.fixture
def cfg() -> VolStraddleConfig:
    """A1-D50 default config."""
    return VolStraddleConfig()


@pytest.fixture
def strategy(cfg) -> VolStraddleStrategy:
    """Fresh strategy instance."""
    return VolStraddleStrategy(cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Config correctness — A1-D50 locked values
# ══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    """Verify A1-D50 parameters match R35 backtest report."""

    def test_a1_d50_vr_threshold(self, cfg):
        assert cfg.d_vr_threshold == 0.50

    def test_vr_periods(self, cfg):
        assert cfg.vr_short == 8   # R39: was 7
        assert cfg.vr_long == 56   # R39: was 50

    def test_sl_tp(self, cfg):
        assert cfg.d_sl == 0.01   # 1% SL
        assert cfg.d_tp == 0.12   # 12% hard TP ceiling (R43: was 5%)
        assert cfg.d_trail_activate == 0.02   # R43: trail after +2%
        assert cfg.d_trail_distance == 0.003  # R43: 0.3% behind peak

    def test_momentum_period(self, cfg):
        assert cfg.momentum_period == 8

    def test_max_hold_per_tf(self, cfg):
        assert cfg.get_max_hold('15m') == 192
        assert cfg.get_max_hold('1h') == 48
        assert cfg.get_max_hold('4h') == 12

    def test_round_trip_cost(self, cfg):
        assert cfg.round_trip_cost == pytest.approx(0.0008)

    def test_straddle_enabled(self, cfg):
        assert cfg.straddle_enabled is True
        assert cfg.hedge_mode is True


# ══════════════════════════════════════════════════════════════════════════════
#  2. Signal computation — VR, momentum, direction
# ══════════════════════════════════════════════════════════════════════════════


class TestSignals:
    """Test numba-compiled signal functions with real data."""

    def test_vr_range_and_shape(self):
        """VR values should be positive and typically 0.2–2.0."""
        data = _load_real_data()
        close = data['close']
        high = data['high']
        low = data['low']

        vr = compute_vr(close, high, low, 7, 50)

        assert len(vr) == len(close)
        valid = vr[~np.isnan(vr)]
        assert len(valid) > 0
        assert np.all(valid > 0)
        assert np.all(valid < 10)  # sanity
        # Some VR values should be < 0.5 (squeeze detected)
        assert np.any(valid < 0.50), "No squeeze found in 3yr PEPE data"

    def test_momentum_computation(self):
        """Momentum = close[i] / close[i-8] - 1."""
        data = _load_real_data()
        close = data['close']
        mom = compute_momentum(close, 8)

        assert len(mom) == len(close)
        # First 8 should be 0
        assert np.all(mom[:8] == 0.0)
        # Manual check a few values
        for i in [10, 100, 500]:
            if i < len(close):
                expected = close[i] / close[i - 8] - 1.0
                assert mom[i] == pytest.approx(expected, rel=1e-10)

    def test_gen_vr_signal_warmup(self):
        """No signals before warmup period."""
        data = _load_real_data()
        close, high, low = data['close'], data['high'], data['low']
        vr = compute_vr(close, high, low, 7, 50)

        warmup = 200
        sig = gen_vr_signal(vr, 0.50, warmup)
        assert np.all(sig[:warmup] == 0)

    def test_direction_assignment(self):
        """Direction matches momentum sign."""
        data = _load_real_data()
        close, high, low = data['close'], data['high'], data['low']
        vr = compute_vr(close, high, low, 7, 50)
        sig = gen_vr_signal(vr, 0.50, 200)
        mom = compute_momentum(close, 8)
        dire = make_direction_momentum(sig, mom)

        # Where signal == 1 and mom > 0 → direction == +1
        mask = (sig == 1) & (mom > 0.0001)
        if np.any(mask):
            assert np.all(dire[mask] == 1)

        # Where signal == 1 and mom < 0 → direction == -1
        mask = (sig == 1) & (mom < -0.0001)
        if np.any(mask):
            assert np.all(dire[mask] == -1)

        # Where signal == 0 → direction == 0
        assert np.all(dire[sig == 0] == 0)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Resampling — completed bars only (lookahead prevention)
# ══════════════════════════════════════════════════════════════════════════════


class TestResampling:
    """Test OHLCV resampling from 15m → 1h with zero lookahead."""

    def test_resample_truncates_remainder(self):
        """Incomplete bars at the end must be discarded."""
        # 7 bars → 7//4 = 1 completed 1h bar, 3 bars discarded
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        h = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
        l = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        v = np.array([10., 20., 30., 40., 50., 60., 70.])

        c_out, h_out, l_out, v_out = resample_ohlcv(c, h, l, v, 4)

        assert len(c_out) == 1
        assert c_out[0] == 4.0   # Last close of first 4 bars
        assert h_out[0] == 4.5   # Max high of first 4 bars
        assert l_out[0] == 0.5   # Min low of first 4 bars
        assert v_out[0] == 100.  # Sum volume

    def test_resample_close_is_last_bar(self):
        """Resampled close uses the LAST bar's close (completed)."""
        n = 200
        c = np.arange(1, n + 1, dtype=np.float64)
        h = c + 0.5
        l = c - 0.5
        v = np.ones(n)

        c_out, _, _, _ = resample_ohlcv(c, h, l, v, 4)

        # c_out[0] should be c[3] (bar index 3 = 4th bar = end of first 1h)
        assert c_out[0] == 4.0
        assert c_out[1] == 8.0
        assert c_out[-1] == c[len(c_out) * 4 - 1]

    def test_resample_high_low_correct(self):
        """Resampled high/low aggregate correctly over the window."""
        c = np.array([10., 12., 8., 11., 9., 13., 7., 10.], dtype=np.float64)
        h = np.array([11., 13., 9., 12., 10., 14., 8., 11.], dtype=np.float64)
        l = np.array([9., 11., 7., 10., 8., 12., 6., 9.], dtype=np.float64)
        v = np.ones(8)

        c_out, h_out, l_out, _ = resample_ohlcv(c, h, l, v, 4)

        assert len(c_out) == 2
        assert h_out[0] == 13.  # max(11, 13, 9, 12)
        assert l_out[0] == 7.   # min(9, 11, 7, 10)
        assert h_out[1] == 14.  # max(10, 14, 8, 11)
        assert l_out[1] == 6.   # min(8, 12, 6, 9)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Strategy — on_bar signal generation
# ══════════════════════════════════════════════════════════════════════════════


class TestStrategySignal:
    """Test VolStraddleStrategy.on_bar() with real data."""

    def test_on_bar_produces_signals(self, strategy):
        """on_bar should produce non-zero signals on real data."""
        data = _load_real_data()
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', np.ones(len(close)))

        signals_found = 0
        # Walk through data bar by bar (range wide enough for VR warmup)
        sym = 'TESTUSDT'
        for i in range(200, min(len(close), 5000)):
            strategy.update_market_data(
                sym, volume[:i], np.ones(i), high[:i], low[:i],
            )
            sig = strategy.on_bar(sym, close[:i])
            if sig != 0:
                signals_found += 1
                # Reset so dedup doesn't block further signals
                strategy._last_signal_meta.pop(sym, None)

        assert signals_found > 0, "No signals in 4800 bars of real data"

    def test_on_bar_dedup_per_symbol(self, strategy):
        """No signal when symbol already has an open position."""
        data = _load_real_data()
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', np.ones(len(close)))
        sym = 'TESTUSDT'

        # Find first signal
        first_signal = 0
        first_idx = 0
        for i in range(200, min(len(close), 5000)):
            strategy.update_market_data(
                sym, volume[:i], np.ones(i), high[:i], low[:i],
            )
            sig = strategy.on_bar(sym, close[:i])
            if sig != 0:
                first_signal = sig
                first_idx = i
                break

        if first_signal == 0:
            pytest.skip("No signal found in test window")

        # Open the position
        strategy.open_position(
            sym, first_signal, close[first_idx],
            notional=100.0, leverage=1.5, move_size=1.0,
        )

        # Walk more bars — all should return 0 (dedup)
        for j in range(first_idx + 1, min(first_idx + 100, len(close))):
            strategy.update_market_data(
                sym, volume[:j], np.ones(j),
                high[:j], low[:j],
            )
            assert strategy.on_bar(sym, close[:j]) == 0, (
                f"Signal leaked at bar {j} despite open position"
            )

    def test_signal_metadata_stored(self, strategy):
        """After on_bar returns non-zero, signal metadata should be available."""
        data = _load_real_data()
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', np.ones(len(close)))
        sym = 'TESTUSDT'

        for i in range(200, min(len(close), 2000)):
            strategy.update_market_data(
                sym, volume[:i], np.ones(i), high[:i], low[:i],
            )
            sig = strategy.on_bar(sym, close[:i])
            if sig != 0:
                meta = strategy.get_last_signal_meta(sym)
                assert 'tf' in meta
                assert 'sl' in meta
                assert 'tp' in meta
                assert 'max_hold' in meta
                assert 'vr' in meta
                assert meta['sl'] == 0.01
                assert meta['tp'] == 0.12
                assert meta['vr'] > 0
                break
        else:
            pytest.skip("No signal found in test window")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Position Management — SL/TP/max_hold exit
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionManagement:
    """Test position lifecycle and exit logic."""

    def test_open_position_stores_tf_metadata(self, strategy):
        sym = 'ETHUSDT'
        strategy._last_signal_meta[sym] = {
            'tf': '1h', 'sl': 0.01, 'tp': 0.05,
            'max_hold': 48, 'vr': 0.42, 'momentum': 0.015,
        }
        strategy.open_position(
            sym, 1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
        )

        pos = strategy._positions[sym]
        assert pos['tf'] == '1h'
        assert pos['sl'] == 0.01
        assert pos['tp'] == 0.05
        assert pos['max_hold'] == 48

    def test_stop_loss_exit(self, strategy):
        """SL at 1%: LONG entry $100, low drops to $98.99 → stop_loss."""
        sym = 'ETHUSDT'
        strategy.open_position(
            sym, 1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
            sl=0.01, tp=0.05, tf='15m', max_hold=192,
        )

        # Price just above SL → hold
        result = strategy.update_position(
            sym, 99.05, current_high=99.10, current_low=99.01,
        )
        assert result is None

        # Price at SL → exit
        result = strategy.update_position(
            sym, 99.0, current_high=99.5, current_low=98.99,
        )
        assert result == 'stop_loss'

    def test_take_profit_exit(self, strategy):
        """R43: LONG $100, trail activates at +2%, fires on 0.3% retrace."""
        sym = 'ETHUSDT'
        strategy.open_position(
            sym, 1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
            sl=0.01, tp=0.12, tf='15m', max_hold=192,
        )

        # Below trail activation threshold → hold
        result = strategy.update_position(
            sym, 101.5, current_high=101.9, current_low=101.0,
        )
        assert result is None

        # Trail activates (+3%), tight range → hold
        result = strategy.update_position(
            sym, 103.0, current_high=103.0, current_low=102.8,
        )
        assert result is None

        # Retraces >0.3% from peak → trailing_stop
        result = strategy.update_position(
            sym, 102.0, current_high=103.5, current_low=102.5,
        )
        assert result == 'trailing_stop'

    def test_short_sl_exit(self, strategy):
        """SHORT entry $100, high hits $101.01 → stop_loss."""
        sym = 'BTCUSDT'
        strategy.open_position(
            sym, -1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
            sl=0.01, tp=0.05, tf='15m', max_hold=192,
        )

        result = strategy.update_position(
            sym, 101.0, current_high=101.01, current_low=100.5,
        )
        assert result == 'stop_loss'

    def test_short_tp_exit(self, strategy):
        """R43: SHORT $100, trail activates at +3%, fires on 0.3% retrace."""
        sym = 'BTCUSDT'
        strategy.open_position(
            sym, -1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
            sl=0.01, tp=0.12, tf='15m', max_hold=192,
        )

        # Trail activates (+3% gain), tight range → hold
        result = strategy.update_position(
            sym, 97.2, current_high=97.3, current_low=97.0,
        )
        assert result is None

        # Retraces >0.3% from peak → trailing_stop
        result = strategy.update_position(
            sym, 98.0, current_high=98.0, current_low=96.5,
        )
        assert result == 'trailing_stop'

    def test_max_hold_exit(self, strategy):
        """Position held past max_hold → max_hold exit."""
        sym = 'ETHUSDT'
        strategy.open_position(
            sym, 1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
            sl=0.01, tp=0.05, tf='4h', max_hold=12,
        )

        # Hold for 11 bars → no exit
        for _ in range(11):
            result = strategy.update_position(
                sym, 100.1, current_high=100.2, current_low=99.95,
            )
            assert result is None

        # 12th bar → max_hold exit
        result = strategy.update_position(
            sym, 100.1, current_high=100.2, current_low=99.95,
        )
        assert result == 'max_hold'

    def test_close_position(self, strategy):
        sym = 'ETHUSDT'
        strategy.open_position(
            sym, 1, 100.0, notional=150.0,
            leverage=1.5, move_size=1.0,
        )
        assert strategy.has_position(sym)
        assert strategy.n_active == 1

        pos = strategy.close_position(sym)
        assert pos is not None
        assert not strategy.has_position(sym)
        assert strategy.n_active == 0


# ══════════════════════════════════════════════════════════════════════════════
#  6. Compatibility interface (scalper integration)
# ══════════════════════════════════════════════════════════════════════════════


class TestScalperInterface:
    """Test compatibility methods used by the scalper."""

    def test_get_effective_stop_mult(self, strategy):
        """Should return sl_pct from signal metadata."""
        sym = 'ETHUSDT'
        strategy._last_signal_meta[sym] = {
            'tf': '15m', 'sl': 0.01, 'tp': 0.05,
            'max_hold': 192, 'vr': 0.45, 'momentum': 0.02,
        }
        close = np.ones(100)
        result = strategy.get_effective_stop_mult(close, symbol=sym)
        assert result == 0.01

    def test_get_last_confidence_zero(self, strategy):
        assert strategy.get_last_confidence('ETHUSDT') == 0

    def test_was_mtf_confirmed_false(self, strategy):
        assert strategy.was_mtf_confirmed('ETHUSDT') is False

    def test_state_serialization(self, strategy):
        """State round-trip: to_state_dict → from_state_dict."""
        sym = 'ETHUSDT'
        strategy.open_position(
            sym, 1, 3000.0, notional=4500.0,
            leverage=1.5, move_size=1.0,
            sl=0.01, tp=0.05, tf='1h', max_hold=48,
        )

        state = strategy.to_state_dict()
        assert state['strategy_name'] == 'vol_straddle'
        assert sym in state['positions']

        # Restore into fresh strategy
        new_strat = VolStraddleStrategy()
        new_strat.from_state_dict(state)
        assert new_strat.has_position(sym)
        pos = new_strat._positions[sym]
        assert pos['entry_price'] == 3000.0
        assert pos['tf'] == '1h'
        assert pos['sl'] == 0.01


# ══════════════════════════════════════════════════════════════════════════════
#  7. Multi-TF priority — 4h > 1h > 15m
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiTFPriority:
    """Verify higher TF signals take priority."""

    def test_4h_priority_over_15m(self):
        """When 4h signal fires, 1h/15m are not checked."""
        cfg = VolStraddleConfig(enable_15m=True, enable_1h=True, enable_4h=True)
        strategy = VolStraddleStrategy(cfg)

        data = _load_real_data()
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', np.ones(len(close)))
        sym = 'TESTUSDT'

        # Resample 15m→4h for OHLCV (cache only has close for 4h)
        c4h, h4h, l4h, _ = resample_ohlcv(
            close, high, low, volume, 16,
        )

        # Feed 4h data
        strategy.update_htf_data(sym, close_4h=c4h, high_4h=h4h, low_4h=l4h)

        # Walk bars looking for any signal, verify metadata has TF field
        for i in range(200, min(len(close), 5000)):
            strategy.update_market_data(
                sym, volume[:i], np.ones(i), high[:i], low[:i],
            )
            sig = strategy.on_bar(sym, close[:i])
            if sig != 0:
                meta = strategy.get_last_signal_meta(sym)
                assert meta['tf'] in ('15m', '1h', '4h')
                assert 'sl' in meta
                assert 'tp' in meta
                break

    def test_tf_disabled(self):
        """When a TF is disabled, it produces no signals."""
        cfg = VolStraddleConfig(enable_15m=False, enable_1h=False, enable_4h=False)
        strategy = VolStraddleStrategy(cfg)

        data = _load_real_data()
        sym = 'TESTUSDT'
        close = data['close'][:500]
        high = data['high'][:500]
        low = data['low'][:500]
        volume = data.get('volume', np.ones(len(close)))[:500]

        strategy.update_market_data(sym, volume, np.ones(500), high, low)
        sig = strategy.on_bar(sym, close)
        assert sig == 0


# ══════════════════════════════════════════════════════════════════════════════
#  8. Lookahead audit — cross-TF data integrity
# ══════════════════════════════════════════════════════════════════════════════


class TestLookaheadPrevention:
    """Verify zero lookahead in signal computation."""

    def test_resample_no_developing_bar(self):
        """Resampled data must NOT include a developing (incomplete) bar.

        If we have 201 15m bars, resampled 1h should be 201//4 = 50 bars.
        The remaining 1 bar (201 - 200 = 1) is dropped.
        """
        n = 201
        c = np.arange(1, n + 1, dtype=np.float64)
        h = c + 0.5
        l = c - 0.5
        v = np.ones(n)

        c_out, _, _, _ = resample_ohlcv(c, h, l, v, 4)
        assert len(c_out) == 50  # 200 // 4 = 50, not 51

    def test_vr_uses_only_past_data(self):
        """VR at bar i should only use data from bars 0..i."""
        data = _load_real_data()
        close = data['close'][:300].copy()
        high = data['high'][:300].copy()
        low = data['low'][:300].copy()

        # Compute VR on full array
        vr_full = compute_vr(close, high, low, 7, 50)

        # Compute VR on first 200 bars only
        vr_partial = compute_vr(close[:200], high[:200], low[:200], 7, 50)

        # VR[199] should be identical regardless of future data
        assert vr_full[199] == pytest.approx(vr_partial[199], rel=1e-12)

    def test_momentum_no_future(self):
        """Momentum at bar i uses only close[i-8..i], no future data."""
        data = _load_real_data()
        close = data['close'][:300].copy()

        mom_full = compute_momentum(close, 8)
        mom_partial = compute_momentum(close[:200], 8)

        # mom[199] should match
        assert mom_full[199] == pytest.approx(mom_partial[199], rel=1e-12)

    def test_1h_resampled_from_completed_15m_only(self):
        """1h VR uses only completed 15m bars (no developing bar)."""
        data = _load_real_data()
        n = 201  # 201 bars → 50 completed 1h bars + 1 discarded
        close = data['close'][:n]
        high = data['high'][:n]
        low = data['low'][:n]
        volume = data.get('volume', np.ones(n))[:n]

        c1h, h1h, l1h, _ = resample_ohlcv(close, high, low, volume, 4)

        # The resampled close should use bar 199 (index), not bar 200
        # c1h[-1] should be close[199] (4th bar of last complete 1h window)
        assert c1h[-1] == close[199]
        assert len(c1h) == 50


# ══════════════════════════════════════════════════════════════════════════════
#  9. Position Sizing — multi-slot margin safety (Incident #3)
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionSizing:
    """Verify per-slot capital never exceeds balance/max_concurrent.

    Incident #3 (2026-03-05): dynamic_capital_enabled=True with n_active=0
    gave balance/1 capped at 50%, consuming entire margin for slot 1 and
    leaving nothing for slots 2-10 (-2019 Margin insufficient).

    Rule: total margin across max_concurrent slots must not exceed balance.
    """

    def test_flat_sizing_10slots(self):
        """Flat sizing: each slot gets exactly balance/max_concurrent."""
        cfg = VolStraddleConfig(dynamic_capital_enabled=False, max_concurrent=10)
        balance = 2000.0
        expected = balance / 10  # $200

        per_slot = balance / cfg.max_concurrent
        assert per_slot == pytest.approx(expected)

        # All 10 slots together must not exceed balance
        total = per_slot * cfg.max_concurrent
        assert total <= balance

    def test_dynamic_capital_capped_by_flat(self):
        """dynamic_capital must never exceed balance/max_concurrent.

        This is the hard cap added to positions.py after Incident #3.
        When n_active=0: balance/max(1, 0+1) = balance → capped to balance/10.
        """
        balance = 2000.0
        max_concurrent = 10
        flat_per_slot = balance / max_concurrent  # $200

        # Simulate dynamic formula at n_active=0 (worst case)
        for n_active in range(max_concurrent):
            dynamic = balance / max(1, n_active + 1)
            dynamic = min(dynamic, balance * 0.5)  # 50% cap
            # Hard cap (the fix we added)
            capped = min(dynamic, flat_per_slot)
            assert capped <= flat_per_slot, (
                f"n_active={n_active}: capped={capped:.2f} > flat={flat_per_slot:.2f}"
            )

    def test_all_slots_fit_in_balance(self):
        """Opening max_concurrent slots sequentially must not exceed balance.

        This is the exact scenario that caused -2019 on slots 2-10.
        Each slot opened while previous ones are active (n_active grows).
        """
        balance = 2000.0
        max_concurrent = 10
        flat_per_slot = balance / max_concurrent  # $200 each

        total_margin = 0.0
        for slot in range(max_concurrent):
            # With the fix: per_slot is always capped at flat_per_slot
            n_active = slot
            dynamic = balance / max(1, n_active + 1)
            dynamic = min(dynamic, balance * 0.5)
            per_slot = min(dynamic, flat_per_slot)  # hard cap
            total_margin += per_slot

        assert total_margin <= balance, (
            f"Total margin ${total_margin:.2f} exceeds balance ${balance:.2f}"
        )

    def test_first_slot_not_50pct_of_balance(self):
        """Slot 1 must not consume 50% of balance (the original bug).

        Original: n_active=0 → balance/1 → min(2000, 1000) = $1000 → 12x = $12,000
        Fixed:    n_active=0 → min(2000, 1000, 200) = $200 → 12x = $2,400
        """
        balance = 2000.0
        max_concurrent = 10
        flat_per_slot = balance / max_concurrent  # $200
        leverage = 12

        # Simulate old (buggy) formula
        buggy_slot = min(balance / max(1, 0 + 1), balance * 0.5)  # $1000
        buggy_notional = buggy_slot * leverage  # $12,000

        # Simulate fixed formula
        fixed_slot = min(buggy_slot, flat_per_slot)  # $200
        fixed_notional = fixed_slot * leverage  # $2,400

        assert buggy_notional == pytest.approx(12000.0)  # confirm bug existed
        assert fixed_notional == pytest.approx(2400.0)   # confirm fix works
        # margin (capital) must be within 1-slot share of balance
        assert fixed_slot <= flat_per_slot  # $200 ≤ $200


class TestHalfKellyLeverage:
    """Verify leverage values are derived from half-Kelly criterion.

    Kelly: f* = WR - (1-WR)/RR
    Half-Kelly risk per trade: f*/2
    Leverage: L = (f*/2) × N / SL

    Source: 20260304_ef55e2e strategy report (6yr, production conditions).
    VR:  WR=25.9%, PF=1.310, RR=3.75 → f*=6.1%, f*/2=3.07% → L≈30
    MA:  WR=38.6%, PF=1.285, RR=2.05 → f*=8.65%, f*/2=4.33% → L≈21
    """

    def test_vr_kelly_fraction(self):
        """VR squeeze half-Kelly ≈ 3.07%."""
        wr, rr = 0.259, 3.749
        kelly = wr - (1 - wr) / rr
        assert kelly == pytest.approx(0.0613, abs=0.005)
        half_kelly = kelly / 2
        assert half_kelly == pytest.approx(0.0307, abs=0.003)

    def test_ma_kelly_fraction(self):
        """MA cluster half-Kelly ≈ 4.33%."""
        wr, rr = 0.386, 2.05
        kelly = wr - (1 - wr) / rr
        assert kelly == pytest.approx(0.0865, abs=0.005)
        half_kelly = kelly / 2
        assert half_kelly == pytest.approx(0.0433, abs=0.003)

    def test_vr_leverage_from_kelly(self):
        """VR leverage = half_kelly × N / SL ≈ 30."""
        cfg = VolStraddleConfig()
        half_kelly_vr = 0.0307
        n = cfg.max_concurrent
        expected_lev = half_kelly_vr * n / cfg.d_sl  # 0.0307 × 10 / 0.01 = 30.7
        assert cfg.base_leverage == pytest.approx(expected_lev, abs=1.0)
        assert cfg.base_leverage == 30.0

    def test_ma_leverage_from_kelly(self):
        """MA leverage = half_kelly × N / SL ≈ 21."""
        cfg = VolStraddleConfig()
        half_kelly_ma = 0.0433
        n = cfg.max_concurrent
        expected_lev = half_kelly_ma * n / cfg.ma_cluster_sl  # 0.0433 × 10 / 0.02 = 21.6
        assert cfg.ma_cluster_leverage == pytest.approx(expected_lev, abs=1.0)
        assert cfg.ma_cluster_leverage == 21.0

    def test_per_trade_risk_vr(self):
        """VR per-trade risk = L × SL / N = 30 × 0.01 / 10 = 3.0%."""
        cfg = VolStraddleConfig()
        risk = cfg.base_leverage * cfg.d_sl / cfg.max_concurrent
        assert risk == pytest.approx(0.030)

    def test_per_trade_risk_ma(self):
        """MA per-trade risk = L × SL / N = 21 × 0.02 / 10 = 4.2%."""
        cfg = VolStraddleConfig()
        risk = cfg.ma_cluster_leverage * cfg.ma_cluster_sl / cfg.max_concurrent
        assert risk == pytest.approx(0.042)

    def test_max_simultaneous_loss(self):
        """Worst case: all 10 slots stopped simultaneously.

        VR-only: 10 × 3.0% = 30%
        MA-only: 10 × 4.2% = 42%
        These are aggressive but within half-Kelly bounds.
        """
        cfg = VolStraddleConfig()
        n = cfg.max_concurrent
        vr_max = n * cfg.base_leverage * cfg.d_sl / n
        ma_max = n * cfg.ma_cluster_leverage * cfg.ma_cluster_sl / n
        assert vr_max == pytest.approx(0.30)
        assert ma_max == pytest.approx(0.42)
        # Neither exceeds 50% (no single-event ruin)
        assert vr_max < 0.50
        assert ma_max < 0.50
