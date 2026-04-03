"""Tests for MA Cluster Squeeze (R44d V17) integration.

Validates:
  1. SMA computation correctness
  2. MA cluster signal detection (cluster → breakout → signal)
  3. One-signal-per-episode logic (cooldown until cluster dissolves)
  4. Integration with VolStraddleStrategy (on_bar returns MA cluster signal)
  5. Signal metadata (leverage_override, strategy_type, SL/TP)
  6. Position exit (simple SL/TP, no trailing stop for MA cluster)
  7. Config defaults
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    check_ma_cluster,
    compute_momentum,
    compute_sma,
)
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy


# ══════════════════════════════════════════════════════════════════════════════
#  SMA
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSma:
    def test_basic_sma(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = compute_sma(close, 3)
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        assert sma[2] == pytest.approx(2.0)
        assert sma[3] == pytest.approx(3.0)
        assert sma[4] == pytest.approx(4.0)

    def test_sma_period_equals_length(self):
        close = np.array([10.0, 20.0, 30.0])
        sma = compute_sma(close, 3)
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        assert sma[2] == pytest.approx(20.0)

    def test_sma_too_short(self):
        close = np.array([1.0, 2.0])
        sma = compute_sma(close, 5)
        assert all(np.isnan(sma))


# ══════════════════════════════════════════════════════════════════════════════
#  check_ma_cluster
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckMaCluster:
    def _make_converging_data(
        self,
        n: int = 250,
        cluster_start: int = 200,
        cluster_bars: int = 20,
        breakout_up: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic 4h data where SMAs converge then breakout."""
        rng = np.random.default_rng(42)
        # Base price trending up slowly
        close = 100.0 + np.cumsum(rng.normal(0.01, 0.3, n))
        close = np.maximum(close, 50.0)  # floor

        # Compute real SMAs
        sma20 = compute_sma(close, 20)
        sma50 = compute_sma(close, 50)
        sma100 = compute_sma(close, 100)
        sma200 = compute_sma(close, 200)

        # Force convergence: make all SMAs very close for cluster_bars
        for i in range(cluster_start, min(cluster_start + cluster_bars, n)):
            avg = close[i]
            sma20[i] = avg * 1.001
            sma50[i] = avg * 0.999
            sma100[i] = avg * 1.002
            sma200[i] = avg * 0.998

        # Force breakout on last bar
        if breakout_up:
            close[-1] = close[-2] * 1.05  # 5% jump above cluster
        else:
            close[-1] = close[-2] * 0.95  # 5% drop below cluster

        # Force momentum direction
        mom = compute_momentum(close, 8)
        return close, sma20, sma50, sma100, sma200, mom

    def test_no_signal_before_duration(self):
        """No signal when cluster hasn't persisted long enough."""
        # Only 3 bars of convergence (< min_duration=8)
        n = 210
        close = np.full(n, 100.0)
        sma20 = np.full(n, 110.0)  # 10% spread — NOT clustered
        sma50 = np.full(n, 90.0)
        sma100 = np.full(n, 100.0)
        sma200 = np.full(n, 100.0)

        # Only last 3 bars are clustered
        for i in range(n - 3, n):
            sma20[i] = 100.3
            sma50[i] = 99.7
            sma100[i] = 100.1
            sma200[i] = 99.9

        close[-1] = 105.0  # breakout
        mom = np.zeros(n)
        mom[-1] = 0.05

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig == 0
        assert duration_out[0] == 3  # Only 3 bars of cluster

    def test_signal_after_duration(self):
        """Signal fires when cluster persists >= min_duration with breakout."""
        n = 250
        close = np.full(n, 100.0)
        # All SMAs within 1% for all bars (well above min_duration=8)
        sma20 = np.full(n, 100.5)
        sma50 = np.full(n, 99.5)
        sma100 = np.full(n, 100.2)
        sma200 = np.full(n, 99.8)

        # Breakout on last bar
        close[-1] = 105.0  # Way above cluster band

        # Momentum positive
        mom = np.zeros(n)
        mom[-1] = 0.05

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig == 1  # Positive momentum → long
        assert duration_out[0] >= 8

    def test_idempotent_repeated_calls(self):
        """Calling with the same data multiple times fires only once (cooldown)."""
        n = 50
        close = np.full(n, 100.0)
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        mom = np.full(n, 0.05)
        close[-1] = 105.0

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        # First call → signal fires
        sig1 = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig1 == 1

        # Second call with SAME data → blocked by cooldown
        sig2 = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig2 == 0
        assert cooldown_arr[0] == 1  # Still in cooldown

    def test_cooldown_blocks_second_signal(self):
        """After signal fires, no second signal until cluster dissolves."""
        n = 50
        close = np.full(n, 100.0)
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        mom = np.full(n, 0.05)

        # Breakout
        close[-1] = 105.0

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig != 0

        # Try with new breakout bar but cluster still active
        close_next = np.full(51, 100.0)
        close_next[-1] = 106.0
        sma20_next = np.full(51, 100.3)
        sma50_next = np.full(51, 99.7)
        sma100_next = np.full(51, 100.1)
        sma200_next = np.full(51, 99.9)
        mom_next = np.full(51, 0.05)

        sig2 = check_ma_cluster(
            close_next,
            sma20_next,
            sma50_next,
            sma100_next,
            sma200_next,
            mom_next,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig2 == 0  # Blocked by cooldown

    def test_cooldown_resets_on_cluster_dissolve(self):
        """Cooldown resets when cluster spread exceeds threshold."""
        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        # First: fire a signal to set cooldown
        n = 50
        close = np.full(n, 100.0)
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        mom = np.full(n, 0.05)
        close[-1] = 105.0

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig != 0
        assert cooldown_arr[0] == 1

        # Now dissolve: spread > cluster_pct
        close2 = np.full(10, 100.0)
        sma20_2 = np.full(10, 110.0)  # 10% spread — way over cluster_pct
        sma50_2 = np.full(10, 90.0)
        sma100_2 = np.full(10, 100.0)
        sma200_2 = np.full(10, 100.0)
        mom2 = np.full(10, 0.05)

        sig2 = check_ma_cluster(
            close2,
            sma20_2,
            sma50_2,
            sma100_2,
            sma200_2,
            mom2,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig2 == 0
        assert cooldown_arr[0] == 0  # Reset
        assert duration_out[0] == 0  # No cluster

    def test_short_signal_on_negative_momentum(self):
        """Negative momentum → short signal."""
        n = 50
        close = np.full(n, 100.0)
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        mom = np.full(n, -0.05)  # Negative momentum

        close[-1] = 95.0  # Below cluster band

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig == -1

    def test_no_signal_when_close_inside_band(self):
        """No signal when close is inside the MA band (no breakout)."""
        n = 50
        close = np.full(n, 100.0)  # Close stays inside
        sma20 = np.full(n, 100.5)
        sma50 = np.full(n, 99.5)
        sma100 = np.full(n, 100.3)
        sma200 = np.full(n, 99.7)
        mom = np.full(n, 0.05)

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
        )
        assert sig == 0  # No breakout → no signal

    def test_min_momentum_filters_weak_signals(self):
        """Signals with |momentum| below min_momentum are rejected (R45)."""
        n = 50
        close = np.full(n, 100.0)
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        # Weak momentum: 0.007 → above old 0.003 but below new 0.01
        mom = np.full(n, 0.007)
        close[-1] = 105.0  # breakout

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        # With min_momentum=0.01, this weak signal is rejected
        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
            min_momentum=0.01,
        )
        assert sig == 0

        # With default (0.003), this would fire
        cooldown_arr[0] = 0
        sig2 = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
            min_momentum=0.003,
        )
        assert sig2 == 1

    def test_max_duration_filters_zombie_signals(self):
        """Signals with duration > max_duration are rejected (R45 zombie filter)."""
        n = 100
        close = np.full(n, 100.0)
        # All bars clustered = duration = n
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        mom = np.full(n, 0.05)
        close[-1] = 105.0  # breakout

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        # With max_duration=24, duration=100 is way over → rejected
        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
            max_duration=24,
        )
        assert sig == 0
        assert duration_out[0] == 100  # Duration correctly computed

        # With max_duration=0 (disabled), it fires
        cooldown_arr[0] = 0
        sig2 = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
            max_duration=0,
        )
        assert sig2 == 1

    def test_max_duration_allows_fresh_cluster(self):
        """Cluster within max_duration range fires normally."""
        n = 50
        close = np.full(n, 100.0)
        # First 30 bars NOT clustered, last 20 bars clustered
        sma20 = np.full(n, 110.0)
        sma50 = np.full(n, 90.0)
        sma100 = np.full(n, 100.0)
        sma200 = np.full(n, 100.0)

        for i in range(n - 20, n):
            sma20[i] = 100.3
            sma50[i] = 99.7
            sma100[i] = 100.1
            sma200[i] = 99.9

        mom = np.full(n, 0.05)
        close[-1] = 105.0

        cooldown_arr = np.zeros(1, dtype=np.int64)
        duration_out = np.zeros(1, dtype=np.int64)

        # duration=20, max_duration=24 → allowed
        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown_arr,
            duration_out=duration_out,
            max_duration=24,
        )
        assert sig == 1
        assert duration_out[0] == 20


# ══════════════════════════════════════════════════════════════════════════════
#  Config Defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestMaClusterConfig:
    def test_defaults(self):
        cfg = VolStraddleConfig()
        assert cfg.ma_cluster_enabled is True  # r92 gate PASS: re-enabled 2026-03-18
        assert cfg.ma_cluster_pct == 0.15  # autoresearch apr03: 2%→15% (33x N, PF +25%)
        assert cfg.ma_cluster_min_duration == 1  # autoresearch apr03: 6→1 (N +2x, PF stable)
        assert cfg.ma_cluster_max_duration == 24
        assert cfg.ma_cluster_min_momentum == 0.01
        assert cfg.ma_cluster_sl == 0.02
        assert cfg.ma_cluster_tp == 0.08
        assert cfg.ma_cluster_max_hold_4h == 12
        assert cfg.ma_cluster_leverage == 21.0
        assert cfg.ma_cluster_sma_periods == [20, 50, 100, 200]

    def test_half_kelly_leverage(self):
        """Leverage derived from half-Kelly criterion.

        VR:  f*=6.1%, f*/2=3.07%, L=0.0307×10/0.01≈30
        MA:  f*=8.65%, f*/2=4.33%, L=0.0433×10/0.02≈21
        Per-trade risk = L × SL / N.
        """
        cfg = VolStraddleConfig()
        n = cfg.max_concurrent
        vr_risk_pct = cfg.base_leverage * cfg.d_sl / n
        ma_risk_pct = cfg.ma_cluster_leverage * cfg.ma_cluster_sl / n
        # VR half-Kelly ≈ 3.07%
        assert 0.025 < vr_risk_pct < 0.035, f"VR risk {vr_risk_pct:.4f} not near half-Kelly 3.07%"
        # MA half-Kelly ≈ 4.33%
        assert 0.038 < ma_risk_pct < 0.048, f"MA risk {ma_risk_pct:.4f} not near half-Kelly 4.33%"

    def test_disable_ma_cluster(self):
        cfg = VolStraddleConfig(ma_cluster_enabled=False)
        assert cfg.ma_cluster_enabled is False


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy Integration
# ══════════════════════════════════════════════════════════════════════════════


class TestStrategyIntegration:
    def _make_strategy(self, **kwargs) -> VolStraddleStrategy:
        defaults = dict(
            # Disable VR signals so we only test MA cluster
            enable_15m=False,
            enable_1h=False,
            enable_4h=True,
            d_vr_threshold=0.01,  # Very low → unlikely to fire
            ma_cluster_enabled=True,
            ma_cluster_max_duration=0,  # Disable zombie filter for basic tests
        )
        defaults.update(kwargs)
        cfg = VolStraddleConfig(**defaults)
        return VolStraddleStrategy(cfg)

    def _build_4h_data(
        self,
        n: int = 250,
        cluster_bars: int = 20,
        breakout_pct: float = 0.05,
    ) -> dict[str, np.ndarray]:
        """Build 4h OHLCV where SMAs converge then breakout."""
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 101.0, dtype=np.float64)
        low = np.full(n, 99.0, dtype=np.float64)

        # Force breakout on last bar
        close[-1] = 100.0 * (1 + breakout_pct)
        high[-1] = close[-1] * 1.01
        low[-1] = close[-1] * 0.99

        return {"close": close, "high": high, "low": low}

    def test_ma_cluster_signal_via_on_bar(self):
        """Strategy returns non-zero via MA cluster when VR doesn't fire."""
        strategy = self._make_strategy()

        n = 250
        data_4h = self._build_4h_data(n=n, breakout_pct=0.05)

        # Store 4h data
        strategy.update_htf_data("BTCUSDT", data_4h["close"], data_4h["high"], data_4h["low"])

        # Need 15m close_prices for on_bar (even though 15m is disabled)
        close_15m = np.full(500, 100.0)

        # Update market data (high/low for VR computation)
        strategy.update_market_data(
            "BTCUSDT",
            np.ones(500),
            np.ones(500),
            np.full(500, 101.0),
            np.full(500, 99.0),
        )

        # Stateless: first call should fire (cluster duration computed from data)
        sig = strategy.on_bar("BTCUSDT", close_15m)

        # Should get a signal from MA cluster (all SMAs = 100, close = 105)
        # The 4h data is static (all 100), so SMAs all converge at 100.
        # Breakout happens because close[-1]=105 > ma_max ≈ 100.
        assert sig != 0
        meta = strategy.get_last_signal_meta("BTCUSDT")
        assert meta["strategy_type"] == "ma_cluster"
        assert meta["sl"] == 0.02
        assert meta["tp"] == 0.08
        assert meta["tf"] == "4h"
        assert meta["leverage_override"] == 21.0
        # max_hold should be converted: 12 bars × 16 (4h→15m) = 192
        assert meta["max_hold"] == 12 * 16

    def test_ma_cluster_disabled(self):
        """No MA cluster signal when disabled."""
        strategy = self._make_strategy(ma_cluster_enabled=False)

        data_4h = self._build_4h_data(n=250, breakout_pct=0.05)
        strategy.update_htf_data("BTCUSDT", data_4h["close"], data_4h["high"], data_4h["low"])

        close_15m = np.full(500, 100.0)
        strategy.update_market_data(
            "BTCUSDT",
            np.ones(500),
            np.ones(500),
            np.full(500, 101.0),
            np.full(500, 99.0),
        )

        sig = strategy.on_bar("BTCUSDT", close_15m)
        # MA cluster disabled → no ma_cluster metadata
        meta = strategy.get_last_signal_meta("BTCUSDT")
        assert meta.get("strategy_type") != "ma_cluster"

    def test_position_exit_trail_for_ma_cluster(self):
        """MA cluster uses trailing stop (trail_activate=2%, trail_distance=0.3%, r75).

        Below trail_activate: no trail. Above: trail activates and fires on retrace.
        Uses trail_server_native=False to test client-side trail exit.
        """
        strategy = self._make_strategy(trail_server_native=False)
        cfg = strategy.config

        strategy.open_position(
            symbol="BTCUSDT",
            signal=1,
            entry_price=100.0,
            notional=1200.0,
            leverage=12.0,
            move_size=1.0,
            sl=0.02,
            tp=0.08,
            tf="4h",
            max_hold=12,
            position_side="LONG",
        )
        key = "BTCUSDT:LONG"
        strategy._positions[key]["strategy_type"] = "ma_cluster"

        trail_activate = cfg.ma_cluster_trail_activate  # 0.01
        trail_distance = cfg.ma_cluster_trail_distance  # 0.005

        # Phase 1: price below trail_activate (0.5% < 1%) — no trail yet
        result = strategy.update_position(
            key,
            100.5,
            current_high=100.5,
            current_low=100.3,
        )
        assert result is None
        assert strategy._positions[key]["trail_active"] is False

        # Phase 2: price past trail_activate (3% > 1%) — trail activates
        result = strategy.update_position(
            key,
            103.0,
            current_high=103.0,
            current_low=102.8,
        )
        assert result is None
        assert strategy._positions[key]["trail_active"] is True

        # Phase 3: retrace past trail_distance → trailing_stop exit
        peak_gain = strategy._positions[key]["peak_gain"]
        trail_stop_gain = peak_gain - trail_distance
        low_exit = 100.0 * (1.0 + trail_stop_gain - 0.001)
        result = strategy.update_position(
            key,
            low_exit,
            current_high=103.0,
            current_low=low_exit,
        )
        assert result == "trailing_stop", f"Expected trailing_stop, got {result}"

    def test_ma_cluster_stop_loss(self):
        """MA cluster position hits SL at 2%."""
        strategy = self._make_strategy()

        strategy.open_position(
            symbol="ETHUSDT",
            signal=1,
            entry_price=100.0,
            notional=1200.0,
            leverage=12.0,
            move_size=1.0,
            sl=0.02,
            tp=0.08,
            tf="4h",
            max_hold=12,
            position_side="LONG",
        )
        key = "ETHUSDT:LONG"
        strategy._positions[key]["strategy_type"] = "ma_cluster"

        # Drop 2.1% — should hit SL
        result = strategy.update_position(
            key,
            97.9,
            current_high=100.0,
            current_low=97.8,
        )
        assert result == "stop_loss"

    def test_ma_cluster_take_profit(self):
        """MA cluster hard TP ceiling at 8% (safety rail; r47 trailing fires first normally).

        Use tight bars (current_high = current_low + 0.05) so trailing stop
        never fires — worst_gain stays 0.05% above trail_stop on each bar.
        Hard TP at 8% still triggers as 'target'.
        """
        strategy = self._make_strategy()

        strategy.open_position(
            symbol="ETHUSDT",
            signal=-1,
            entry_price=100.0,
            notional=1200.0,
            leverage=12.0,
            move_size=1.0,
            sl=0.02,
            tp=0.08,
            tf="4h",
            max_hold=12,
            position_side="SHORT",
        )
        key = "ETHUSDT:SHORT"
        strategy._positions[key]["strategy_type"] = "ma_cluster"

        # Gradual drop with tight bars — trail never fires
        for p in [98, 96, 94]:
            result = strategy.update_position(
                key,
                p,
                current_high=p + 0.05,
                current_low=p,
            )
            assert result is None, f"Premature exit at {p}: {result}"

        # Hit hard TP at 8% drop with tight bar
        tp_price = 92.0  # 8% drop from 100
        result = strategy.update_position(
            key,
            tp_price,
            current_high=tp_price + 0.05,
            current_low=tp_price,
        )
        assert result == "target", f"Expected hard TP 'target', got {result}"

    def test_ma_cluster_trailing_stop(self):
        """MA cluster trailing stop fires after 4% gain retraces 0.5% (r47).

        trail_activate=4%, trail_distance=0.5%.
        """
        strategy = self._make_strategy(trail_server_native=False)

        strategy.open_position(
            symbol="ETHUSDT",
            signal=-1,
            entry_price=100.0,
            notional=1200.0,
            leverage=12.0,
            move_size=1.0,
            sl=0.02,
            tp=0.08,
            tf="4h",
            max_hold=12,
            position_side="SHORT",
        )
        key = "ETHUSDT:SHORT"
        cfg = strategy.config
        strategy._positions[key]["strategy_type"] = "ma_cluster"

        trail_activate = cfg.ma_cluster_trail_activate  # 0.04
        trail_distance = cfg.ma_cluster_trail_distance  # 0.005

        # Phase 1: Drop to activate with tight bar (tight high)
        peak = 100.0 * (1.0 - trail_activate - 0.005)  # ~95.5
        result = strategy.update_position(
            key,
            peak,
            current_high=peak * 1.001,
            current_low=peak,
        )
        assert result is None, f"Should activate trail but not exit, got {result}"
        pos = strategy._positions[key]
        assert pos["trail_active"] is True

        # Phase 2: Bounce past trail distance → exit
        peak_gain = pos["peak_gain"]
        # For SHORT: worst_gain = (entry - high)/entry; trail fires when worst_gain <= trail_stop_gain
        trail_stop_gain = peak_gain - trail_distance
        high_breach = 100.0 * (1.0 - trail_stop_gain) + 0.5
        result = strategy.update_position(
            key,
            high_breach - 1,
            current_high=high_breach,
            current_low=peak * 0.995,
        )
        assert result == "trailing_stop", f"Expected trailing_stop, got {result}"

    def test_ma_cluster_max_hold(self):
        """MA cluster position exits at max_hold (12 bars)."""
        strategy = self._make_strategy()

        strategy.open_position(
            symbol="BNBUSDT",
            signal=1,
            entry_price=100.0,
            notional=1200.0,
            leverage=12.0,
            move_size=1.0,
            sl=0.02,
            tp=0.08,
            tf="4h",
            max_hold=12,
            position_side="LONG",
        )
        key = "BNBUSDT:LONG"
        strategy._positions[key]["strategy_type"] = "ma_cluster"

        # Simulate 11 bars — no exit (price flat, below trail_activate=1%)
        for _ in range(11):
            result = strategy.update_position(
                key,
                100.3,
                current_high=100.5,
                current_low=100.0,
            )
            assert result is None

        # 12th bar — max hold
        result = strategy.update_position(
            key,
            100.3,
            current_high=100.5,
            current_low=100.0,
        )
        assert result == "max_hold"


# ══════════════════════════════════════════════════════════════════════════════
#  v14.0 Audit: Stateless Duration & Edge Cases
# ══════════════════════════════════════════════════════════════════════════════


class TestStatelessDuration:
    """Verify the stateless backward scan produces correct duration."""

    def test_duration_output_matches_cluster_bars(self):
        """duration_out[0] reflects actual consecutive cluster bars."""
        n = 30
        close = np.full(n, 100.0)
        sma20 = np.full(n, 110.0)  # NOT clustered
        sma50 = np.full(n, 90.0)
        sma100 = np.full(n, 100.0)
        sma200 = np.full(n, 100.0)
        mom = np.full(n, 0.05)

        # Last 15 bars are clustered
        for i in range(n - 15, n):
            sma20[i] = 100.3
            sma50[i] = 99.7
            sma100[i] = 100.1
            sma200[i] = 99.9

        close[-1] = 105.0  # breakout

        cooldown = np.zeros(1, dtype=np.int64)
        dur = np.zeros(1, dtype=np.int64)

        check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown,
            duration_out=dur,
        )
        assert dur[0] == 15

    def test_duration_zero_when_last_bar_not_clustered(self):
        """If last bar isn't clustered, duration=0 even if earlier bars were."""
        n = 20
        close = np.full(n, 100.0)
        # All bars clustered EXCEPT the last one
        sma20 = np.full(n, 100.3)
        sma50 = np.full(n, 99.7)
        sma100 = np.full(n, 100.1)
        sma200 = np.full(n, 99.9)
        # Dissolve cluster on last bar
        sma20[-1] = 110.0
        sma50[-1] = 90.0
        mom = np.full(n, 0.05)

        cooldown = np.zeros(1, dtype=np.int64)
        dur = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown,
            duration_out=dur,
        )
        assert sig == 0
        assert dur[0] == 0

    def test_sma200_min_data_requirement(self):
        """check_ma_cluster returns 0 when SMA200 is NaN (insufficient data)."""
        n = 100  # Not enough for SMA(200)
        close = np.full(n, 100.0)
        sma200 = np.full(n, np.nan)  # All NaN
        sma20 = np.full(n, 100.0)
        sma50 = np.full(n, 100.0)
        sma100 = np.full(n, 100.0)
        mom = np.full(n, 0.05)

        cooldown = np.zeros(1, dtype=np.int64)
        dur = np.zeros(1, dtype=np.int64)

        sig = check_ma_cluster(
            close,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            cluster_pct=0.03,
            min_duration=8,
            in_cooldown_arr=cooldown,
            duration_out=dur,
        )
        assert sig == 0
        assert dur[0] == 0


class TestMaxHoldTfConversion:
    """Verify max_hold is correctly converted from native TF to 15m bars."""

    def test_4h_max_hold_in_signal_meta(self):
        """MA cluster signal metadata has max_hold = 12 * 16 = 192."""
        from quantbox.strategy.vol_straddle.strategy import _TF_TO_15M

        assert _TF_TO_15M == {"15m": 1, "30m": 2, "1h": 4, "4h": 16}

        strategy = VolStraddleStrategy(
            VolStraddleConfig(
                enable_15m=False,
                enable_1h=False,
                enable_4h=True,
                ma_cluster_enabled=True,
                ma_cluster_max_duration=0,
            )
        )

        # Build data where MA cluster fires
        n = 250
        data = {
            "close": np.full(n, 100.0, dtype=np.float64),
            "high": np.full(n, 101.0, dtype=np.float64),
            "low": np.full(n, 99.0, dtype=np.float64),
        }
        data["close"][-1] = 105.0  # breakout

        strategy.update_htf_data("TESTUSDT", data["close"], data["high"], data["low"])
        strategy.update_market_data(
            "TESTUSDT",
            np.ones(500),
            np.ones(500),
            np.full(500, 101.0),
            np.full(500, 99.0),
        )

        sig = strategy.on_bar("TESTUSDT", np.full(500, 100.0))
        if sig != 0:
            meta = strategy.get_last_signal_meta("TESTUSDT")
            assert meta["max_hold"] == 12 * 16  # 192 bars of 15m

    def test_vr_1h_max_hold_conversion(self):
        """VR 1h signal should have max_hold = 48 * 4 = 192."""
        from quantbox.strategy.vol_straddle.strategy import _TF_TO_15M

        cfg = VolStraddleConfig()
        expected = cfg.get_max_hold("1h") * _TF_TO_15M["1h"]
        assert expected == 48 * 4  # 192
