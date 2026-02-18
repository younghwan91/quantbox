"""
Tests for Elliott Wave Strategy — ABC pattern + Fibonacci validation.

Uses REAL Binance data (cached) for strategy integration tests,
and small deterministic arrays for indicator unit tests.
"""

import numpy as np
import pytest

from quantbox.strategy.elliott import (
    ElliottConfig,
    ElliottStrategy,
    compute_atr,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_stochastic,
    check_macd_reversal,
    check_stoch_cross,
    compute_elliott_signals,
    detect_impulse_abc_patterns,
    detect_bearish_divergence,
    detect_bullish_divergence,
    detect_ending_diagonal,
    detect_triangle,
    zigzag_pivots,
)


# ═══════════════════════════════════════════════════════════════════════
#  Indicator unit tests (deterministic small arrays)
# ═══════════════════════════════════════════════════════════════════════


class TestComputeEMA:
    """EMA computation tests."""

    def test_single_value(self):
        close = np.array([100.0])
        ema = compute_ema(close, 10)
        assert ema[0] == 100.0

    def test_constant_series(self):
        close = np.full(50, 42.0)
        ema = compute_ema(close, 20)
        np.testing.assert_allclose(ema[-1], 42.0, atol=0.01)

    def test_trending_up(self):
        close = np.arange(1.0, 101.0)
        ema = compute_ema(close, 10)
        # EMA should lag behind close in uptrend
        assert ema[-1] < close[-1]
        assert ema[-1] > close[-10]


class TestComputeRSI:
    """RSI computation tests."""

    def test_constant_price_gives_50(self):
        close = np.full(30, 100.0)
        rsi = compute_rsi(close, 14)
        assert rsi[-1] == pytest.approx(50.0, abs=1.0)

    def test_strong_uptrend_high_rsi(self):
        close = np.arange(100.0, 150.0)  # Straight up
        rsi = compute_rsi(close, 14)
        assert rsi[-1] > 90.0

    def test_strong_downtrend_low_rsi(self):
        close = np.arange(150.0, 100.0, -1.0)  # Straight down
        rsi = compute_rsi(close, 14)
        assert rsi[-1] < 10.0

    def test_output_shape(self):
        close = np.random.randn(100).cumsum() + 100
        rsi = compute_rsi(close, 14)
        assert len(rsi) == len(close)


class TestComputeMACD:
    """MACD computation tests."""

    def test_output_shapes(self):
        close = np.random.randn(100).cumsum() + 100
        macd_line, signal_line, histogram = compute_macd(close, 12, 26, 9)
        assert len(macd_line) == 100
        assert len(signal_line) == 100
        assert len(histogram) == 100

    def test_constant_gives_zero(self):
        close = np.full(100, 50.0)
        macd_line, _, histogram = compute_macd(close, 12, 26, 9)
        # MACD should converge to 0 for constant price
        assert abs(macd_line[-1]) < 0.01
        assert abs(histogram[-1]) < 0.01


class TestComputeStochastic:
    """Stochastic %K/%D tests."""

    def test_output_shapes(self):
        n = 50
        high = np.random.randn(n).cumsum() + 110
        low = high - 5
        close = (high + low) / 2
        k, d = compute_stochastic(high, low, close, 14, 3)
        assert len(k) == n
        assert len(d) == n

    def test_range_0_100(self):
        n = 100
        close = np.random.randn(n).cumsum() + 100
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))
        k, d = compute_stochastic(high, low, close, 14, 3)
        assert np.all(k >= 0) and np.all(k <= 100)
        assert np.all(d >= 0) and np.all(d <= 100)


class TestComputeATR:
    """ATR computation tests."""

    def test_constant_price_zero_atr(self):
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)
        atr = compute_atr(high, low, close, 14)
        assert atr[-1] == pytest.approx(0.0, abs=0.01)

    def test_positive_atr(self):
        n = 50
        close = np.random.randn(n).cumsum() + 100
        high = close + 1.0
        low = close - 1.0
        atr = compute_atr(high, low, close, 14)
        assert atr[-1] > 0


# ═══════════════════════════════════════════════════════════════════════
#  ZigZag tests
# ═══════════════════════════════════════════════════════════════════════


class TestZigZagPivots:
    """ZigZag swing detection tests."""

    def test_simple_triangle(self):
        """A clear up-down-up pattern should produce pivots."""
        n = 60
        # Build a clear triangle: up → peak → down → valley → up
        prices = np.concatenate([
            np.linspace(100, 120, 20),  # Up 20%
            np.linspace(120, 95, 20),   # Down to 95
            np.linspace(95, 115, 20),   # Back up
        ])
        high = prices + 0.5
        low = prices - 0.5

        pidx, pprice, ptype, n_piv = zigzag_pivots(high, low, 0.05)
        assert n_piv >= 3  # At least low-high-low or high-low-high

    def test_alternating_types(self):
        """Pivots must alternate between +1 and -1."""
        prices = np.concatenate([
            np.linspace(100, 130, 15),
            np.linspace(130, 90, 15),
            np.linspace(90, 125, 15),
            np.linspace(125, 85, 15),
        ])
        high = prices + 0.2
        low = prices - 0.2

        pidx, pprice, ptype, n_piv = zigzag_pivots(high, low, 0.03)
        for i in range(1, n_piv):
            assert ptype[i] != ptype[i - 1], f"Pivots {i-1} and {i} same type"

    def test_flat_market_few_pivots(self):
        """Flat market with 3% threshold → very few pivots."""
        rng = np.random.RandomState(42)
        prices = np.full(100, 100.0) + rng.randn(100) * 0.5
        high = prices + 0.2
        low = prices - 0.2

        _, _, _, n_piv = zigzag_pivots(high, low, 0.03)
        assert n_piv <= 4  # Few or no meaningful swings

    def test_minimum_data(self):
        """Edge case: very short data."""
        high = np.array([100.0])
        low = np.array([99.0])
        _, _, _, n_piv = zigzag_pivots(high, low, 0.03)
        assert n_piv == 0


# ═══════════════════════════════════════════════════════════════════════
#  Divergence detection tests
# ═══════════════════════════════════════════════════════════════════════


class TestDivergence:
    """RSI divergence detection tests."""

    def test_bullish_divergence_detected(self):
        """Lower low in price + higher low in RSI → bullish divergence."""
        n = 50
        # Uptrending base so most points are NOT local minima
        close = np.linspace(100, 108, n)
        # Clear first dip (within lookback window)
        close[24] = 97.0
        close[25] = 94.0  # Local min 1
        close[26] = 97.0
        # Clear second dip (lower low)
        close[40] = 94.0
        close[41] = 92.0  # Local min 2 (lower)
        close[42] = 94.0

        # RSI: second dip has higher RSI (divergence)
        rsi = np.full(n, 50.0)
        rsi[25] = 25.0  # RSI at first low
        rsi[41] = 30.0  # Higher RSI at lower price = divergence

        result = detect_bullish_divergence(close, rsi, 30)
        assert result is True

    def test_no_divergence_when_aligned(self):
        """Both lower price and lower RSI → no divergence."""
        n = 50
        close = np.linspace(100, 108, n)
        close[24] = 97.0
        close[25] = 94.0
        close[26] = 97.0
        close[40] = 94.0
        close[41] = 92.0
        close[42] = 94.0

        rsi = np.full(n, 50.0)
        rsi[25] = 30.0
        rsi[41] = 20.0  # Also lower → aligned, no divergence

        result = detect_bullish_divergence(close, rsi, 30)
        assert result is False

    def test_bearish_divergence_detected(self):
        """Higher high in price + lower high in RSI → bearish div."""
        n = 50
        close = np.linspace(100, 92, n)  # Slight downtrend base
        close[24] = 112.0
        close[25] = 115.0  # Local max 1
        close[26] = 112.0
        close[40] = 117.0
        close[41] = 120.0  # Local max 2 (higher high)
        close[42] = 117.0

        rsi = np.full(n, 50.0)
        rsi[25] = 80.0
        rsi[41] = 75.0   # Lower high in RSI

        result = detect_bearish_divergence(close, rsi, 30)
        assert result is True


class TestMACDReversal:
    """MACD histogram reversal tests."""

    def test_bullish_reversal(self):
        # declining negative then reversal up: was falling, now rising
        hist = np.array([-3.0, -5.0, -4.0, -2.0])
        assert check_macd_reversal(hist, 1) is True

    def test_bearish_reversal(self):
        # rising positive then reversal down: was rising, now falling
        hist = np.array([3.0, 5.0, 4.0, 2.0])
        assert check_macd_reversal(hist, -1) is True

    def test_no_reversal(self):
        hist = np.array([-5.0, -6.0, -7.0])  # Still declining
        assert check_macd_reversal(hist, 1) is False


class TestStochCross:
    """Stochastic crossover tests."""

    def test_bullish_cross_oversold(self):
        k = np.array([15.0, 22.0])
        d = np.array([20.0, 20.0])
        assert check_stoch_cross(k, d, 1) is True

    def test_bearish_cross_overbought(self):
        k = np.array([82.0, 78.0])
        d = np.array([78.0, 80.0])
        assert check_stoch_cross(k, d, -1) is True

    def test_cross_outside_zone_rejected(self):
        """Cross at 50 (neither oversold/overbought) → rejected."""
        k = np.array([48.0, 52.0])
        d = np.array([50.0, 50.0])
        assert check_stoch_cross(k, d, 1) is False


# ═══════════════════════════════════════════════════════════════════════
#  5-Wave Impulse + ABC pattern detection tests
# ═══════════════════════════════════════════════════════════════════════


class TestABCPatternDetection:
    """5-wave impulse + ABC corrective wave pattern detection tests.

    Uses 9-pivot sequences per winning_strategy §3.1.
    """

    def _make_pivots(self, sequence):
        """Build pivot arrays from (price, type) sequence."""
        n = len(sequence)
        idx = np.arange(n, dtype=np.int64) * 10  # Spread out indices
        prices = np.array([s[0] for s in sequence], dtype=np.float64)
        types = np.array([s[1] for s in sequence], dtype=np.int64)
        return idx, prices, types, n

    def test_valid_bearish_abc_long(self):
        """Valid 5-wave bullish impulse + bearish ABC → long signal.

        Impulse UP: W0(100)→W1(150)→W2(125)→W3(225)→W4(190)→W5(250)
          wave1=50, w2_retrace=0.50, w3_ext=2.0, w4_retrace=0.35
          W2>W0 ✓, W3>W1 ✓, W4>W1 ✓ (no wave4/wave1 overlap)
        ABC DOWN: W5(250)→A(195)→B(230)→C(165)
          wave_a=55, b_retrace=0.636, c_vs_a=1.182
          total_impulse=150, fib618=157.3, c_fib_dist=0.051
        """
        sequence = [
            (100.0, -1),   # W0 (low)
            (150.0, 1),    # W1 (high)
            (125.0, -1),   # W2 (low — 0.50 retrace of W1)
            (225.0, 1),    # W3 (high — 2.0x extension of W1)
            (190.0, -1),   # W4 (low — 0.35 retrace of W3, above W1)
            (250.0, 1),    # W5 (high)
            (195.0, -1),   # A (low)
            (230.0, 1),    # B (high — 0.636 retrace of A)
            (165.0, -1),   # C (low — near 0.618 fib of impulse)
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, cfg)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p['signal'] == 1  # Long
        assert p['type'] == 'bullish_impulse_abc_long'
        # TP1 = W5 (prior impulse peak), TP2 = C + 1.0 × total_impulse
        assert p['target_1'] == pytest.approx(250.0, rel=1e-3)  # W5 price
        assert p['target_2'] == pytest.approx(165.0 + 1.0 * 150.0, rel=1e-3)  # C + impulse
        assert p['stop'] == pytest.approx(165.0 * 0.995, rel=1e-3)  # C × (1 - 0.5%)
        assert 'quality' in p
        assert p['quality'] > 0  # should have positive quality score

    def test_valid_bullish_abc_short(self):
        """Valid 5-wave bearish impulse + bullish ABC → short signal.

        Impulse DOWN: W0(250)→W1(200)→W2(225)→W3(125)→W4(155)→W5(100)
          wave1=50, w2_retrace=0.50, w3_ext=2.0, w4_retrace=0.40
          W2<W0 ✓, W3<W1 ✓, W4<W1 ✓
        ABC UP: W5(100)→A(157)→B(130)→C(185)
          wave_a=57, b_ret=0.474, c_vs_a=0.965
          total_impulse=150, fib618=100+92.7=192.7, dist=|185-192.7|/150=0.051
        """
        sequence = [
            (250.0, 1),    # W0 (high)
            (200.0, -1),   # W1 (low)
            (225.0, 1),    # W2 (high — 0.50 retrace)
            (125.0, -1),   # W3 (low — 2.0x extension)
            (155.0, 1),    # W4 (high — 0.40 retrace, below W1)
            (100.0, -1),   # W5 (low)
            (157.0, 1),    # A (high)
            (130.0, -1),   # B (low — 0.474 retrace of A)
            (185.0, 1),    # C (high — near 0.618 fib)
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, cfg)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p['signal'] == -1  # Short
        assert p['type'] == 'bearish_impulse_abc_short'
        # TP1 = W5 (prior impulse low), TP2 = C - 1.0 × total_impulse
        assert p['target_1'] == pytest.approx(100.0, rel=1e-3)  # W5 price
        assert p['target_2'] == pytest.approx(185.0 - 1.0 * 150.0, rel=1e-3)  # C - impulse
        assert 'quality' in p
        assert p['quality'] > 0  # should have positive quality score

    def test_invalid_b_exceeds_origin(self):
        """B exceeding W5 (A origin) invalidates the pattern."""
        sequence = [
            (100.0, -1), (150.0, 1), (125.0, -1), (225.0, 1),
            (190.0, -1), (250.0, 1),
            (195.0, -1),
            (255.0, 1),   # B > W5(250) → INVALID
            (165.0, -1),
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, cfg)
        assert len(patterns) == 0

    def test_wave4_intruding_wave1_rejected(self):
        """Wave 4 entering Wave 1 territory → INVALID (cardinal rule)."""
        sequence = [
            (100.0, -1), (150.0, 1), (125.0, -1), (225.0, 1),
            (145.0, -1),  # W4=145 < W1=150 → INVALID overlap
            (250.0, 1), (195.0, -1), (230.0, 1), (165.0, -1),
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, cfg)
        assert len(patterns) == 0

    def test_insufficient_pivots(self):
        """Less than 9 pivots → no patterns."""
        idx = np.arange(5, dtype=np.int64) * 10
        prices = np.array([100.0, 150.0, 125.0, 200.0, 180.0])
        types = np.array([-1, 1, -1, 1, -1], dtype=np.int64)
        cfg = ElliottConfig()
        patterns = detect_impulse_abc_patterns(idx, prices, types, 5, cfg)
        assert len(patterns) == 0


# ═══════════════════════════════════════════════════════════════════════
#  Wedge and Triangle detection tests
# ═══════════════════════════════════════════════════════════════════════


class TestWedgeTriangle:
    """Ending diagonal and triangle pattern tests."""

    def test_converging_wedge_detected(self):
        """Alternating pivots with decreasing ranges → wedge."""
        # Swing highs: 100, 98, 96 (declining)
        # Swing lows: 80, 83, 86 (rising) → ranges shrinking
        prices = np.array([80.0, 100.0, 83.0, 98.0, 86.0, 96.0, 88.0, 95.0])
        types = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.int64)
        result = detect_ending_diagonal(prices, types, 8, min_pivots=6)
        assert result is True

    def test_non_converging_no_wedge(self):
        """Non-converging pivots → no wedge."""
        prices = np.array([80.0, 110.0, 70.0, 120.0, 60.0, 130.0])
        types = np.array([-1, 1, -1, 1, -1, 1], dtype=np.int64)
        result = detect_ending_diagonal(prices, types, 6, min_pivots=6)
        assert result is False

    def test_triangle_contracting(self):
        """Decreasing highs + increasing lows → triangle."""
        prices = np.array([
            90.0, 110.0, 92.0, 108.0, 94.0, 106.0, 96.0, 105.0,
        ])
        types = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.int64)
        result = detect_triangle(prices, types, 8)
        assert result is True


# ═══════════════════════════════════════════════════════════════════════
#  Strategy class integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestElliottStrategy:
    """ElliottStrategy class integration tests."""

    def test_macro_bias_no_data(self):
        """Without BTC data → neutral bias."""
        s = ElliottStrategy()
        assert s.macro_bias() == 'neutral'

    def test_macro_bias_bullish(self):
        """BTC above both 200/500 EMA → long bias."""
        s = ElliottStrategy()
        # Create uptrending BTC 4h: 600 bars, steadily climbing
        btc = np.linspace(50000, 70000, 600)
        s.update_btc_data(btc)
        assert s.macro_bias() == 'long'

    def test_macro_bias_bearish(self):
        """BTC below both 200/500 EMA → short bias."""
        s = ElliottStrategy()
        # Downtrending BTC 4h
        btc = np.linspace(70000, 50000, 600)
        s.update_btc_data(btc)
        assert s.macro_bias() == 'short'

    def test_position_lifecycle_basic(self):
        """Open → update → close lifecycle."""
        s = ElliottStrategy()
        s.open_position(
            symbol='ETHUSDT', signal=1, entry_price=3000.0,
            notional=100.0, leverage=20, move_size=0.03,
        )
        assert s.has_position('ETHUSDT')
        assert s.n_active == 1

        # Close
        info = s.close_position('ETHUSDT')
        assert info is not None
        assert not s.has_position('ETHUSDT')
        assert s.n_active == 0

    def test_exploratory_confirmation_flow(self):
        """Exploratory → confirm_entry when divergence fires."""
        s = ElliottStrategy()
        s.open_position(
            symbol='ETHUSDT', signal=1, entry_price=3000.0,
            notional=100.0, leverage=20, move_size=0.03,
        )
        pos = s._positions['ETHUSDT']
        assert pos['phase'] == 'exploratory'

        # Simulate: no divergence data → stays exploratory, eventually cancels
        for _ in range(4):
            result = s.update_position('ETHUSDT', 3010.0)
            if result == 'cancel_exploration':
                break

        assert result == 'cancel_exploration'

    def test_stop_loss_triggers(self):
        """Price hitting stop → 'stop_loss' returned."""
        s = ElliottStrategy()
        s._pending_patterns['ETHUSDT'] = {
            'target_1': 3200.0,
            'target_2': 3500.0,
            'stop': 2900.0,
            'impulse_size': 500.0,
        }
        s.open_position(
            symbol='ETHUSDT', signal=1, entry_price=3000.0,
            notional=100.0, leverage=20, move_size=0.03,
        )
        # Force past exploratory phase
        s._positions['ETHUSDT']['phase'] = 'primary'

        result = s.update_position('ETHUSDT', 2890.0)  # Below stop
        assert result == 'stop_loss'

    def test_partial_close_at_target(self):
        """Price reaching target_1 → 'partial_close'."""
        s = ElliottStrategy()
        s._pending_patterns['ETHUSDT'] = {
            'target_1': 3200.0,
            'target_2': 3500.0,
            'stop': 2900.0,
            'impulse_size': 500.0,
        }
        s.open_position(
            symbol='ETHUSDT', signal=1, entry_price=3000.0,
            notional=100.0, leverage=20, move_size=0.03,
        )
        s._positions['ETHUSDT']['phase'] = 'primary'

        result = s.update_position('ETHUSDT', 3250.0)  # Above target_1
        assert result == 'partial_close'
        assert s._positions['ETHUSDT']['partial_closed'] is True
        assert s._positions['ETHUSDT']['phase'] == 'trailing'

    def test_breakeven_stop_after_partial(self):
        """After partial close, price back to entry → 'breakeven_stop'."""
        s = ElliottStrategy()
        s._pending_patterns['ETHUSDT'] = {
            'target_1': 3200.0,
            'target_2': 3500.0,
            'stop': 2900.0,
            'impulse_size': 500.0,
        }
        s.open_position(
            symbol='ETHUSDT', signal=1, entry_price=3000.0,
            notional=100.0, leverage=20, move_size=0.03,
        )
        s._positions['ETHUSDT']['phase'] = 'primary'

        # Partial close
        s.update_position('ETHUSDT', 3250.0)

        # Price returns to entry → breakeven stop
        result = s.update_position('ETHUSDT', 2990.0)
        assert result == 'breakeven_stop'

    def test_on_bar_returns_zero_without_data(self):
        """on_bar without 4h data returns 0."""
        s = ElliottStrategy()
        close_15m = np.random.randn(200).cumsum() + 100
        assert s.on_bar('ETHUSDT', close_15m) == 0

    def test_interface_compatibility(self):
        """ElliottStrategy has the same interface as RubberBandStrategy."""
        s = ElliottStrategy()
        # Check all required interface methods exist
        assert hasattr(s, 'on_bar')
        assert hasattr(s, 'on_bar_4h')
        assert hasattr(s, 'update_position')
        assert hasattr(s, 'open_position')
        assert hasattr(s, 'close_position')
        assert hasattr(s, 'update_market_data')
        assert hasattr(s, 'update_htf_data')
        assert hasattr(s, 'has_position')
        assert hasattr(s, 'active_positions')
        assert hasattr(s, 'n_active')
        assert hasattr(s, 'was_mtf_confirmed')
        assert hasattr(s, 'get_effective_stop_mult')
        assert hasattr(s, '_positions')
        assert hasattr(s, '_mtf_confirmed')


# ═══════════════════════════════════════════════════════════════════════
#  Backtest signal generation test (real data)
# ═══════════════════════════════════════════════════════════════════════


class TestBacktestSignals:
    """Elliott backtest signal generation on real BTCUSDT data."""

    @pytest.fixture
    def btc_4h_data(self):
        """Load real BTC 4h data from cache."""
        try:
            from quantbox.data.real_data_loader import RealDataLoader
            loader = RealDataLoader()
            df = loader.load_symbol('BTCUSDT')
            if df is None or len(df) < 600:
                pytest.skip("Need 600+ bars of BTC data")
            # Normalize column names (loader may use upper-case)
            df.columns = [c.lower() for c in df.columns]
            if 'close' not in df.columns:
                pytest.skip("Data missing 'close' column")
            return df
        except Exception:
            pytest.skip("Real data not available")

    def test_signals_shape_matches_input(self, btc_4h_data):
        """Signal array must match input length."""
        df = btc_4h_data
        close = df['close'].values[-600:].astype(np.float64)
        high = df['high'].values[-600:].astype(np.float64)
        low = df['low'].values[-600:].astype(np.float64)

        signals, _ = compute_elliott_signals(close, high, low, close)
        assert len(signals) == len(close)

    def test_signals_are_valid_values(self, btc_4h_data):
        """Signals must be valid leverage values: 0, ±0.4, ±1.0, ±2.0."""
        df = btc_4h_data
        close = df['close'].values[-600:].astype(np.float64)
        high = df['high'].values[-600:].astype(np.float64)
        low = df['low'].values[-600:].astype(np.float64)

        signals, _ = compute_elliott_signals(close, high, low, close)
        valid_values = {-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0}
        unique = set(signals)
        assert unique.issubset(valid_values), f"Unexpected signal values: {unique - valid_values}"


# ═══════════════════════════════════════════════════════════════════════
#  v2 target & confluence tests
# ═══════════════════════════════════════════════════════════════════════


class TestV2TargetImprovements:
    """Verify v2 changes: TP1=W5, TP2=impulse projection, lookback=12."""

    def _make_pivots(self, sequence):
        n = len(sequence)
        idx = np.arange(n, dtype=np.int64) * 10
        prices = np.array([s[0] for s in sequence], dtype=np.float64)
        types = np.array([s[1] for s in sequence], dtype=np.int64)
        return idx, prices, types, n

    def test_bullish_tp1_is_w5_not_b(self):
        """TP1 must target W5 (structural resistance), NOT B (correction bounce)."""
        sequence = [
            (100.0, -1), (150.0, 1), (125.0, -1), (225.0, 1),
            (190.0, -1), (250.0, 1),  # W5 = 250
            (195.0, -1), (230.0, 1),  # B = 230
            (165.0, -1),              # C = 165
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, ElliottConfig())
        assert len(patterns) >= 1
        p = patterns[0]
        # TP1 = W5 = 250 (NOT B = 230)
        assert p['target_1'] == pytest.approx(250.0, rel=1e-3)
        assert p['target_1'] != pytest.approx(230.0, rel=1e-3)

    def test_bearish_tp1_is_w5_not_b(self):
        """Bearish TP1 must target W5 (structural support), NOT B."""
        sequence = [
            (250.0, 1), (200.0, -1), (225.0, 1), (125.0, -1),
            (155.0, 1), (100.0, -1),  # W5 = 100
            (157.0, 1), (130.0, -1),  # B = 130
            (185.0, 1),               # C = 185
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, ElliottConfig())
        assert len(patterns) >= 1
        p = patterns[0]
        # TP1 = W5 = 100 (NOT B = 130)
        assert p['target_1'] == pytest.approx(100.0, rel=1e-3)
        assert p['target_1'] != pytest.approx(130.0, rel=1e-3)

    def test_tp2_uses_full_impulse_not_wave1(self):
        """TP2 must use total impulse (W5-W0), not internal wave1 (W1-W0)."""
        sequence = [
            (100.0, -1), (150.0, 1), (125.0, -1), (225.0, 1),
            (190.0, -1), (250.0, 1),
            (195.0, -1), (230.0, 1), (165.0, -1),
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        patterns = detect_impulse_abc_patterns(idx, prices, types, n, ElliottConfig())
        p = patterns[0]
        total_impulse = 250.0 - 100.0  # W5 - W0 = 150
        wave1_only = 150.0 - 100.0      # W1 - W0 = 50
        # TP2 = C + 1.0 × total_impulse = 165 + 150 = 315
        assert p['target_2'] == pytest.approx(165.0 + 1.0 * total_impulse, rel=1e-3)
        # Must NOT be the old formula using wave1
        assert p['target_2'] != pytest.approx(165.0 + 1.618 * wave1_only, rel=1e-3)

    def test_pattern_lookback_bars_is_12(self):
        """Default lookback must be 12 bars (2 days at 4h), not 5."""
        cfg = ElliottConfig()
        assert cfg.pattern_lookback_bars == 12
