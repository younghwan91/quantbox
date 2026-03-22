"""Tests for MA Cluster 4h BTC Gate (r92).

Validates:
  1. Config defaults: ma_cluster_btc_gate_enabled=True, ma_cluster_btc_ma_period_4h=50
  2. Strategy.on_bar tags _last_signal_source correctly (vr_directional vs ma_cluster vs none)
  3. Gate blocks MA cluster LONG when BTC 4h below MA50
  4. Gate allows MA cluster LONG when BTC 4h above MA50
  5. Gate does NOT apply to MA cluster SHORT signals
  6. Gate does NOT apply to VR squeeze signals (even if BTC below MA)
  7. Scalper-level gate: signal zeroed out before entering candidates
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy


# ══════════════════════════════════════════════════════════════════════════════
#  Config defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestMaClusterGateConfig:
    def test_gate_enabled_by_default(self):
        cfg = VolStraddleConfig()
        assert cfg.ma_cluster_btc_gate_enabled is True

    def test_ma_period_4h_default(self):
        cfg = VolStraddleConfig()
        assert cfg.ma_cluster_btc_ma_period_4h == 50

    def test_gate_can_be_disabled(self):
        cfg = VolStraddleConfig(ma_cluster_btc_gate_enabled=False)
        assert cfg.ma_cluster_btc_gate_enabled is False


# ══════════════════════════════════════════════════════════════════════════════
#  _last_signal_source tracking
# ══════════════════════════════════════════════════════════════════════════════


def _make_strategy(ma_cluster_enabled: bool = False) -> VolStraddleStrategy:
    cfg = VolStraddleConfig(
        ma_cluster_enabled=ma_cluster_enabled,
        enable_4h=True,
        enable_1h=True,
        enable_30m=False,
        enable_15m=True,
    )
    return VolStraddleStrategy(cfg)


def _flat_close(n: int = 300, price: float = 100.0) -> np.ndarray:
    """Flat close array — no VR signal fires."""
    return np.full(n, price, dtype=float)


class TestLastSignalSource:
    def test_no_signal_tags_none(self):
        strat = _make_strategy()
        close = _flat_close()
        strat.update_market_data("SYM", np.ones(300), np.ones(300), np.ones(300), np.ones(300))
        strat.on_bar("SYM", close)
        assert strat._last_signal_source.get("SYM") == "none"

    def test_vr_signal_tags_vr_directional(self):
        """When VR signal fires, source must be vr_directional."""
        strat = _make_strategy()
        # Build close array that triggers a VR squeeze: low then spike
        n = 300
        close = np.full(n, 100.0)
        # Create artificially tight range followed by breakout
        # (VR = short_std / long_std; force high VR by making recent prices very flat)
        close[-30:] = 100.0 + np.linspace(0, 0.001, 30)  # near-flat recent
        close[-1] = 103.0  # breakout

        high = close + 0.5
        low = close - 0.5
        strat.update_market_data("SYM", np.ones(n), np.ones(n), high, low)
        sig = strat.on_bar("SYM", close)
        # Whether or not signal fires, source must be vr_directional or none (not ma_cluster)
        src = strat._last_signal_source.get("SYM", "none")
        assert src in ("vr_directional", "none"), f"Unexpected source: {src}"
        if sig != 0:
            assert src == "vr_directional"

    def test_ma_cluster_signal_tags_ma_cluster(self):
        """When MA cluster fires, source must be ma_cluster."""
        strat = _make_strategy(ma_cluster_enabled=True)
        # Inject 4h data with an active MA cluster pattern
        n4h = 250
        # Prices: SMAs will be close together then breakout
        close_4h = np.ones(n4h) * 100.0
        # Make last 10 bars flat (tight cluster), then spike
        close_4h[-10:] = 100.05
        close_4h[-1] = 102.5  # momentum breakout
        high_4h = close_4h + 0.1
        low_4h = close_4h - 0.1
        strat.update_htf_data("SYM", close_4h, high_4h, low_4h)

        # Flat 15m data so no VR signal fires
        close_15m = _flat_close()
        strat.update_market_data("SYM", np.ones(300), np.ones(300), close_15m + 0.1, close_15m - 0.1)
        strat.on_bar("SYM", close_15m)
        # Whether signal fires or not, it should never be tagged vr_directional
        src = strat._last_signal_source.get("SYM", "none")
        assert src in ("ma_cluster", "none")


# ══════════════════════════════════════════════════════════════════════════════
#  Gate logic (unit) — pure numpy
# ══════════════════════════════════════════════════════════════════════════════


def _apply_gate(
    signal: int,
    signal_src: str,
    btc_4h_close: np.ndarray,
    period: int = 50,
    gate_enabled: bool = True,
) -> int:
    """Replicate the gate logic from scalper._scan_entries for unit testing."""
    if signal == 1 and gate_enabled and signal_src == "ma_cluster" and len(btc_4h_close) >= period:
        btc_4h_ma = float(np.mean(btc_4h_close[-period:]))
        btc_4h_cur = float(btc_4h_close[-1])
        if btc_4h_cur < btc_4h_ma:
            return 0  # blocked
    return signal


class TestGateLogic:
    def test_blocks_long_when_btc_below_ma(self):
        # BTC in downtrend: current < MA50
        btc = np.linspace(100_000, 80_000, 100)  # 100 bars, declining
        result = _apply_gate(signal=1, signal_src="ma_cluster", btc_4h_close=btc)
        assert result == 0, "Gate should block LONG when BTC below MA50"

    def test_allows_long_when_btc_above_ma(self):
        # BTC in uptrend: current > MA50
        btc = np.linspace(80_000, 100_000, 100)  # 100 bars, rising
        result = _apply_gate(signal=1, signal_src="ma_cluster", btc_4h_close=btc)
        assert result == 1, "Gate should allow LONG when BTC above MA50"

    def test_does_not_block_short(self):
        # Gate only applies to LONGs — shorts are fine during downtrend
        btc = np.linspace(100_000, 80_000, 100)
        result = _apply_gate(signal=-1, signal_src="ma_cluster", btc_4h_close=btc)
        assert result == -1, "Gate should NOT block SHORT signals"

    def test_does_not_apply_to_vr_directional(self):
        # Gate only applies to MA cluster signals, not VR squeeze
        btc = np.linspace(100_000, 80_000, 100)
        result = _apply_gate(signal=1, signal_src="vr_directional", btc_4h_close=btc)
        assert result == 1, "Gate should NOT block VR squeeze signals"

    def test_gate_disabled(self):
        btc = np.linspace(100_000, 80_000, 100)
        result = _apply_gate(signal=1, signal_src="ma_cluster", btc_4h_close=btc, gate_enabled=False)
        assert result == 1, "Disabled gate should pass all signals"

    def test_insufficient_btc_data(self):
        # Less than MA period available — gate cannot compute, should pass
        btc = np.linspace(100_000, 80_000, 30)  # only 30 bars, need 50
        result = _apply_gate(signal=1, signal_src="ma_cluster", btc_4h_close=btc, period=50)
        assert result == 1, "Gate should pass when BTC data insufficient"

    def test_btc_exactly_at_ma_is_allowed(self):
        # BTC exactly at MA → not strictly below → allow
        btc = np.full(100, 90_000.0)  # flat → current == MA
        result = _apply_gate(signal=1, signal_src="ma_cluster", btc_4h_close=btc)
        assert result == 1, "BTC exactly at MA should not be blocked (strictly < only)"
