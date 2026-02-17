"""
Tests for Elliott Wave Strategy — Maeoknam Championship Method.

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
    detect_abc_patterns,
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
        prices = np.full(100, 100.0) + np.random.randn(100) * 0.5
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
#  ABC pattern detection tests
# ═══════════════════════════════════════════════════════════════════════


class TestABCPatternDetection:
    """ABC corrective wave pattern detection tests."""

    def _make_pivots(self, sequence):
        """Build pivot arrays from (price, type) sequence."""
        n = len(sequence)
        idx = np.arange(n, dtype=np.int64) * 10  # Spread out indices
        prices = np.array([s[0] for s in sequence], dtype=np.float64)
        types = np.array([s[1] for s in sequence], dtype=np.int64)
        return idx, prices, types, n

    def test_valid_bearish_abc_long(self):
        """Valid bearish ABC → long signal."""
        # Low(100) → High(200) → Low_A(130) → High_B(175) → Low_C(120)
        #  impulse = 100
        #  wave_a = 200-130=70, wave_b = 175-130=45, wave_c = 175-120=55
        #  b_retrace = 45/70 = 0.643 ✓ (0.40-0.72)
        #  c_vs_a = 55/70 = 0.786 → slightly below 0.80 min
        # Adjust: Low_C = 115 → wave_c = 175-115 = 60, c_vs_a = 60/70 = 0.857 ✓
        # C near 0.618 Fib: H - 0.618*(H-L) = 200 - 61.8 = 138.2
        # c_fib_dist = |115 - 138.2| / 100 = 0.232 → too far
        # Adjust for better Fib: target C at ~138
        # Let's use: Low(100) → High(200) → A(140) → B(180) → C(138)
        #  wave_a = 60, wave_b = 40, wave_c = 42
        #  b_retrace = 40/60 = 0.667 ✓
        #  c_vs_a = 42/60 = 0.70 → below min. Adjust again.
        # Better: Low(100)→High(200)→A(130)→B(172)→C(138)
        #  wave_a=70, wave_b=42, wave_c=34 → c_vs_a=0.486 too low
        # Let me just build a valid setup directly:
        # Impulse L→H: 100→200 (impulse=100, fib618=138.2)
        # A: 200→120 (wave_a=80)
        # B: 120→160 (wave_b=40, b_ret=40/80=0.50 ✓)
        # C: 160→130 (wave_c=30) → c_vs_a = 30/80 = 0.375 low
        # Need c_vs_a >= 0.80: wave_c >= 64
        # C: 160→96 (wave_c=64) → c_vs_a = 0.80 ✓
        # C=96, fib618=138.2, dist=|96-138.2|/100=0.422 too far
        # Make impulse smaller or C closer to fib:
        # Impulse L→H: 100→150 (impulse=50, fib618=119.1)
        # A: 150→120 (wave_a=30)
        # B: 120→138 (wave_b=18, b_ret=0.60 ✓)
        # C: 138→112 (wave_c=26, c_vs_a=0.867 ✓)
        # C=112, fib618=119.1, dist=|112-119.1|/50=0.142 → close to tolerance
        # Reduce tolerance slightly: Use C=118
        # C: 138→118 (wave_c=20, c_vs_a=0.667) → too low again
        # Let me use C=114
        # C: 138→114 (wave_c=24, c_vs_a=0.80 ✓)
        # dist=|114-119.1|/50=0.102 ✓
        sequence = [
            (100.0, -1),   # L (swing low)
            (150.0, 1),    # H (swing high — impulse end)
            (120.0, -1),   # A (swing low — first correction leg)
            (138.0, 1),    # B (swing high — partial retrace)
            (114.0, -1),   # C (swing low — entry zone)
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_abc_patterns(idx, prices, types, n, cfg)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p['signal'] == 1  # Long
        assert p['type'] == 'bearish_abc_long'
        assert p['target_1'] == 138.0  # B level
        assert p['target_2'] == 150.0  # H level
        assert p['stop'] < 114.0  # Below C

    def test_valid_bullish_abc_short(self):
        """Valid bullish ABC → short signal."""
        # High(150) → Low(100) → High_A(130) → Low_B(112) → High_C(136)
        # impulse = 50 (down), fib618 = 100 + 0.618*50 = 130.9
        # wave_a = 130-100 = 30, wave_b = 130-112 = 18, wave_c = 136-112 = 24
        # b_retrace = 18/30 = 0.60 ✓
        # c_vs_a = 24/30 = 0.80 ✓
        # c_fib_dist = |136-130.9|/50 = 0.102 ✓
        sequence = [
            (150.0, 1),    # H (swing high)
            (100.0, -1),   # L (swing low — impulse end)
            (130.0, 1),    # A (swing high — correction up)
            (112.0, -1),   # B (swing low — partial retrace)
            (136.0, 1),    # C (swing high — entry zone)
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_abc_patterns(idx, prices, types, n, cfg)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p['signal'] == -1  # Short
        assert p['type'] == 'bullish_abc_short'

    def test_invalid_b_exceeds_origin(self):
        """B exceeding the origin invalidates the pattern."""
        # B goes above H → invalid per winning_strategy §3.1
        sequence = [
            (100.0, -1),
            (150.0, 1),    # H
            (120.0, -1),
            (155.0, 1),    # B > H → INVALID
            (110.0, -1),
        ]
        idx, prices, types, n = self._make_pivots(sequence)
        cfg = ElliottConfig()
        patterns = detect_abc_patterns(idx, prices, types, n, cfg)
        assert len(patterns) == 0

    def test_insufficient_pivots(self):
        """Less than 5 pivots → no patterns."""
        idx = np.array([0, 10, 20], dtype=np.int64)
        prices = np.array([100.0, 120.0, 105.0])
        types = np.array([-1, 1, -1], dtype=np.int64)
        cfg = ElliottConfig()
        patterns = detect_abc_patterns(idx, prices, types, 3, cfg)
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

        signals = compute_elliott_signals(close, high, low, close)
        assert len(signals) == len(close)

    def test_signals_are_valid_values(self, btc_4h_data):
        """Signals must be -1, 0, or +1."""
        df = btc_4h_data
        close = df['close'].values[-600:].astype(np.float64)
        high = df['high'].values[-600:].astype(np.float64)
        low = df['low'].values[-600:].astype(np.float64)

        signals = compute_elliott_signals(close, high, low, close)
        unique = set(signals)
        assert unique.issubset({-1.0, 0.0, 1.0})
