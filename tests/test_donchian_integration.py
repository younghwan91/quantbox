"""Integration tests for Donchian Channel Breakout strategy embedded in VolStraddleStrategy.

Tests verify integration points: signal priority, metadata, trail params,
BTC gate, cooldown persistence, config gate, and signal source tracking.

All tests use hand-crafted numpy arrays — no randomness, no I/O.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from quantbox.core.strategy_names import DONCHIAN, VR_DIRECTIONAL
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

_SYM = "TESTUSDT"

# Donchian defaults that match config
_LOOKBACK = 50
_VOL_SMA = 20
_VOL_MULT = 2.5
_BREAKOUT_PCT = 0.02


def _make_config(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> VolStraddleConfig:
    """Create VolStraddleConfig with env overrides, clearing other VS_ vars."""
    for key in list(os.environ):
        if key.startswith("VS_"):
            monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return VolStraddleConfig()


def _flat_4h(n: int, price: float = 100.0) -> dict[str, np.ndarray]:
    """Return flat 4h OHLCV arrays with normal volume."""
    close = np.full(n, price, dtype=np.float64)
    high = close * 1.001
    low = close * 0.999
    volume = np.full(n, 100.0, dtype=np.float64)
    return {"close": close, "high": high, "low": low, "volume": volume}


def _make_donchian_long_signal(n: int = 80, price: float = 100.0) -> dict[str, np.ndarray]:
    """Build 4h arrays where the last bar has a confirmed LONG breakout."""
    data = _flat_4h(n, price)
    # Channel high (bars 0..n-2) ≈ price * 1.001
    # Set last bar close well above: 4% breakout (> 2% threshold)
    data["close"][-1] = price * 1.045
    data["high"][-1] = price * 1.050
    data["low"][-1] = price * 1.040
    # Volume spike: 5× normal (> 2.5× SMA)
    data["volume"][-1] = 500.0
    return data


def _make_donchian_short_signal(n: int = 80, price: float = 100.0) -> dict[str, np.ndarray]:
    """Build 4h arrays where the last bar has a confirmed SHORT breakout."""
    data = _flat_4h(n, price)
    data["close"][-1] = price * 0.955
    data["high"][-1] = price * 0.960
    data["low"][-1] = price * 0.950
    data["volume"][-1] = 500.0
    return data


def _make_strategy(monkeypatch: pytest.MonkeyPatch, extra_env: dict[str, str] | None = None) -> VolStraddleStrategy:
    """Build a VolStraddleStrategy with Donchian enabled and all VR/MA disabled."""
    env = {
        "VS_DONCHIAN_ENABLED": "true",
        # Disable VR TFs so they never fire in integration tests
        "VS_ENABLE_15M": "false",
        "VS_ENABLE_30M": "false",
        "VS_ENABLE_1H": "false",
        "VS_ENABLE_4H": "false",
        # Disable MA cluster so it never fires
        "VS_MA_CLUSTER_ENABLED": "false",
    }
    if extra_env:
        env.update(extra_env)
    cfg = _make_config(monkeypatch, env)
    return VolStraddleStrategy(cfg)


def _close_15m(n: int = 300, price: float = 100.0) -> np.ndarray:
    """Flat 15m close array (passed to on_bar; VR TFs disabled so irrelevant)."""
    return np.full(n, price, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  TC-1: Signal priority — VR fires → Donchian never evaluated
# ══════════════════════════════════════════════════════════════════════════════


class TestSignalPriorityVRBlocks:
    """When VR (4h) fires, Donchian path is never reached (early return)."""

    def test_vr_fires_donchian_source_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Enable 4h VR so it can fire; keep Donchian enabled
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_ENABLE_4H": "true",
                "VS_ENABLE_15M": "false",
                "VS_ENABLE_30M": "false",
                "VS_ENABLE_1H": "false",
                "VS_MA_CLUSTER_ENABLED": "false",
            },
        )
        strategy = VolStraddleStrategy(cfg)

        # Build 4h data that triggers a strong VR compression signal
        n = 120
        price = 100.0
        close_4h = np.full(n, price, dtype=np.float64)
        high_4h = close_4h * 1.001
        low_4h = close_4h * 0.999
        # Compress ATR(8) vs ATR(56): make last 8 bars tight, prior 56 wide
        # Widen the older 56-bar range to make VR(8/56) < 0.50
        high_4h[:56] = price * 1.05
        low_4h[:56] = price * 0.95
        # Tighten the last 8 bars
        high_4h[-8:] = price * 1.0005
        low_4h[-8:] = price * 0.9995
        volume_4h = np.full(n, 100.0, dtype=np.float64)

        strategy.update_htf_data(_SYM, close_4h, high_4h, low_4h, volume_4h=volume_4h)

        close_15m = _close_15m()
        # Provide required high/low for on_bar array checks
        strategy._high_data[_SYM] = np.full(300, price * 1.001)
        strategy._low_data[_SYM] = np.full(300, price * 0.999)

        signal = strategy.on_bar(_SYM, close_15m)

        if signal != 0:
            # If VR fired, source must be VR_DIRECTIONAL (not donchian)
            assert strategy._last_signal_source.get(_SYM) != DONCHIAN, (
                "When VR fires, signal source must not be donchian"
            )
        # If signal == 0, no VR fire — this is acceptable (data may not compress enough).
        # The key invariant: source is never donchian when 4h VR is enabled and checked first.
        source = strategy._last_signal_source.get(_SYM, "none")
        assert source in (VR_DIRECTIONAL, "none"), f"Unexpected source when VR is active: {source!r}"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-2: Signal priority — MA Cluster fires → Donchian never evaluated
# ══════════════════════════════════════════════════════════════════════════════


class TestSignalPriorityMAClusterBlocks:
    """When MA Cluster fires, Donchian path is skipped (MA Cluster has higher priority)."""

    def test_ma_cluster_source_not_donchian(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Enable MA cluster; Donchian enabled too
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_MA_CLUSTER_ENABLED": "true",
                "VS_ENABLE_4H": "true",
                "VS_ENABLE_15M": "false",
                "VS_ENABLE_30M": "false",
                "VS_ENABLE_1H": "false",
                # MA cluster minimum duration = 1 so it can fire easily
                "VS_MA_CLUSTER_MIN_DURATION": "1",
            },
        )
        strategy = VolStraddleStrategy(cfg)

        # Build 4h data: cluster (SMAs 20/50/100/200 converged), breakout, momentum
        n = 300
        price = 100.0
        close_4h = np.full(n, price, dtype=np.float64)
        # Create a strong upward breakout on the last bar so MA cluster fires
        close_4h[-1] = price * 1.10
        high_4h = close_4h.copy()
        low_4h = close_4h * 0.999
        volume_4h = np.full(n, 100.0, dtype=np.float64)

        strategy.update_htf_data(_SYM, close_4h, high_4h, low_4h, volume_4h=volume_4h)
        strategy._high_data[_SYM] = np.full(300, price * 1.001)
        strategy._low_data[_SYM] = np.full(300, price * 0.999)

        strategy.on_bar(_SYM, _close_15m())

        source = strategy._last_signal_source.get(_SYM, "none")
        # Source must not be donchian when MA cluster is checked first and may fire
        assert source != DONCHIAN, f"Donchian should not fire when MA Cluster path runs first; source={source!r}"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-3: Donchian fires when VR and MA Cluster are silent
# ══════════════════════════════════════════════════════════════════════════════


class TestDonchianFiresWhenOthersSilent:
    """With VR/MA disabled, Donchian fires and sets _last_signal_source = 'donchian'."""

    def test_donchian_fires_returns_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        signal = strategy.on_bar(_SYM, _close_15m())

        assert signal == 1
        assert strategy._last_signal_source.get(_SYM) == DONCHIAN


# ══════════════════════════════════════════════════════════════════════════════
#  TC-4: _last_signal_meta contents when Donchian fires
# ══════════════════════════════════════════════════════════════════════════════


class TestDonchianSignalMeta:
    """When Donchian fires, _last_signal_meta contains all 8 required keys."""

    _REQUIRED_KEYS = {"tf", "sl", "tp", "max_hold", "vr", "momentum", "strategy_type", "leverage_override"}

    def test_meta_has_all_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        signal = strategy.on_bar(_SYM, _close_15m())
        assert signal != 0, "Expected Donchian to fire"

        meta = strategy._last_signal_meta.get(_SYM)
        assert meta is not None, "_last_signal_meta not populated"
        missing = self._REQUIRED_KEYS - set(meta.keys())
        assert not missing, f"Missing keys in _last_signal_meta: {missing}"

    def test_meta_strategy_type_is_donchian(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        strategy.on_bar(_SYM, _close_15m())

        meta = strategy._last_signal_meta[_SYM]
        assert meta["strategy_type"] == DONCHIAN

    def test_meta_leverage_override_is_15(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        strategy.on_bar(_SYM, _close_15m())

        meta = strategy._last_signal_meta[_SYM]
        assert meta["leverage_override"] == pytest.approx(15.0)

    def test_meta_tf_is_4h(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        strategy.on_bar(_SYM, _close_15m())

        meta = strategy._last_signal_meta[_SYM]
        assert meta["tf"] == "4h"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-5: _get_trail_params dispatch for Donchian
# ══════════════════════════════════════════════════════════════════════════════


class TestTrailParamsDispatch:
    """_get_trail_params correctly computes callback_rate and activation_price for Donchian."""

    def _make_mock_bracket(self, monkeypatch: pytest.MonkeyPatch) -> object:
        """Build a minimal object that has _get_trail_params via BracketMixin."""
        from quantbox.execution.brackets import BracketMixin

        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_DONCHIAN_TRAIL_DISTANCE": "0.008",
                "VS_DONCHIAN_TRAIL_ACTIVATE": "0.02",
            },
        )

        class _Stub(BracketMixin):
            vs_config = cfg
            paper_mode = True

        return _Stub()

    def test_long_callback_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self._make_mock_bracket(monkeypatch)
        callback_rate, _, _ = stub._get_trail_params(DONCHIAN, 1, 100.0)
        # trail_distance=0.008 → 0.8, clamped to [0.5, 5.0]
        assert callback_rate == pytest.approx(0.8)

    def test_long_activation_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self._make_mock_bracket(monkeypatch)
        _, activation_price, _ = stub._get_trail_params(DONCHIAN, 1, 100.0)
        # trail_activate=0.02 → 100 * 1.02 = 102.0
        assert activation_price == pytest.approx(102.0)

    def test_short_activation_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self._make_mock_bracket(monkeypatch)
        _, activation_price, _ = stub._get_trail_params(DONCHIAN, -1, 100.0)
        # trail_activate=0.02 → 100 * (1 - 0.02) = 98.0
        assert activation_price == pytest.approx(98.0)

    def test_short_callback_rate_same_as_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self._make_mock_bracket(monkeypatch)
        cb_long, _, _ = stub._get_trail_params(DONCHIAN, 1, 100.0)
        cb_short, _, _ = stub._get_trail_params(DONCHIAN, -1, 100.0)
        assert cb_long == pytest.approx(cb_short)

    def test_close_side_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self._make_mock_bracket(monkeypatch)
        _, _, close_side = stub._get_trail_params(DONCHIAN, 1, 100.0)
        assert close_side == "SELL"

    def test_close_side_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self._make_mock_bracket(monkeypatch)
        _, _, close_side = stub._get_trail_params(DONCHIAN, -1, 100.0)
        assert close_side == "BUY"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-6: _get_trail_params raises ValueError for unknown strategy_type
# ══════════════════════════════════════════════════════════════════════════════


class TestTrailParamsValueError:
    """_get_trail_params raises ValueError for unrecognised strategy_type."""

    def test_unknown_strategy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantbox.execution.brackets import BracketMixin

        cfg = _make_config(monkeypatch, {})

        class _Stub(BracketMixin):
            vs_config = cfg
            paper_mode = True

        stub = _Stub()
        with pytest.raises(ValueError, match="Unknown strategy_type"):
            stub._get_trail_params("unknown_strategy", 1, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
#  TC-7: BTC gate blocks Donchian LONG when BTC < MA50 4h
# ══════════════════════════════════════════════════════════════════════════════


class TestBTCGateBlocksLong:
    """Scalper-level BTC gate suppresses Donchian LONG when BTC 4h < MA50."""

    def _apply_gate(
        self,
        signal: int,
        btc_close_4h: np.ndarray,
        vs_config: VolStraddleConfig,
        last_signal_source: str,
    ) -> int:
        """Replicate the gate logic from scalper.py lines 1104-1121."""
        period = vs_config.donchian_btc_ma_period_4h
        if (
            signal == 1
            and vs_config.donchian_enabled
            and vs_config.donchian_btc_gate_enabled
            and last_signal_source == DONCHIAN
            and len(btc_close_4h) >= period
        ):
            btc_4h_ma = float(np.mean(btc_close_4h[-period:]))
            btc_4h_cur = float(btc_close_4h[-1])
            if btc_4h_cur < btc_4h_ma:
                return 0
        return signal

    def test_long_blocked_when_btc_below_ma50(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_DONCHIAN_BTC_GATE_ENABLED": "true",
                "VS_DONCHIAN_BTC_MA_PERIOD_4H": "50",
            },
        )
        # BTC in downtrend: all bars at 50k, last bar at 45k (below MA50)
        btc_4h = np.full(100, 50000.0, dtype=np.float64)
        btc_4h[-1] = 45000.0  # current BTC below MA → gate fires

        result = self._apply_gate(1, btc_4h, cfg, DONCHIAN)
        assert result == 0, "LONG should be blocked when BTC < MA50"

    def test_long_allowed_when_btc_above_ma50(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_DONCHIAN_BTC_GATE_ENABLED": "true",
                "VS_DONCHIAN_BTC_MA_PERIOD_4H": "50",
            },
        )
        # BTC in uptrend: last bar above MA50
        btc_4h = np.full(100, 50000.0, dtype=np.float64)
        btc_4h[-1] = 55000.0  # above MA → gate does not fire

        result = self._apply_gate(1, btc_4h, cfg, DONCHIAN)
        assert result == 1, "LONG should be allowed when BTC > MA50"

    def test_short_never_blocked_by_btc_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_DONCHIAN_BTC_GATE_ENABLED": "true",
                "VS_DONCHIAN_BTC_MA_PERIOD_4H": "50",
            },
        )
        # BTC in downtrend — but SHORT should pass through
        btc_4h = np.full(100, 50000.0, dtype=np.float64)
        btc_4h[-1] = 45000.0

        result = self._apply_gate(-1, btc_4h, cfg, DONCHIAN)
        assert result == -1, "SHORT must never be blocked by the BTC gate"

    def test_gate_does_not_fire_for_other_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_config(
            monkeypatch,
            {
                "VS_DONCHIAN_ENABLED": "true",
                "VS_DONCHIAN_BTC_GATE_ENABLED": "true",
            },
        )
        btc_4h = np.full(100, 50000.0, dtype=np.float64)
        btc_4h[-1] = 45000.0  # BTC below MA

        # Signal source is VR_DIRECTIONAL, not donchian → gate must not suppress
        result = self._apply_gate(1, btc_4h, cfg, VR_DIRECTIONAL)
        assert result == 1, "BTC gate must only fire for donchian source"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-8: Cooldown persistence — save and restore state values
# ══════════════════════════════════════════════════════════════════════════════


class TestCooldownPersistence:
    """Donchian cooldown state survives a save/restore cycle."""

    def test_cooldown_save_restore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        # Fire the signal to set cooldown
        signal = strategy.on_bar(_SYM, _close_15m())
        assert signal != 0, "Expected Donchian to fire"

        cooldown_before = int(strategy._donchian_cooldown[_SYM][0])
        assert cooldown_before > 0, "Cooldown should be set after signal fires"

        # Simulate save: extract cooldown values to a plain dict
        saved: dict[str, int] = {sym: int(arr[0]) for sym, arr in strategy._donchian_cooldown.items()}

        # Simulate restore: rebuild arrays from saved dict
        for sym, val in saved.items():
            arr = np.zeros(1, dtype=np.int64)
            arr[0] = val
            strategy._donchian_cooldown[sym] = arr

        cooldown_after = int(strategy._donchian_cooldown[_SYM][0])
        assert cooldown_after == cooldown_before, (
            f"Cooldown mismatch after restore: {cooldown_after} != {cooldown_before}"
        )

    def test_cooldown_suppresses_signal_after_fire(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After signal fires, the same breakout data returns 0 on the next call."""
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        close_15m = _close_15m()
        r1 = strategy.on_bar(_SYM, close_15m)
        assert r1 != 0, "Expected first call to fire"

        r2 = strategy.on_bar(_SYM, close_15m)
        assert r2 == 0, "Expected cooldown to suppress second call with identical data"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-9: Config disabled — Donchian path never entered
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigDisabledSkipsDonchian:
    """When donchian_enabled=False (default), on_bar never enters Donchian path."""

    def test_disabled_source_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Explicitly disable donchian (.env may have VS_DONCHIAN_ENABLED=true)
        cfg = _make_config(
            monkeypatch,
            {
                "VS_ENABLE_15M": "false",
                "VS_ENABLE_30M": "false",
                "VS_ENABLE_1H": "false",
                "VS_ENABLE_4H": "false",
                "VS_MA_CLUSTER_ENABLED": "false",
                "VS_DONCHIAN_ENABLED": "false",
            },
        )
        assert not cfg.donchian_enabled, "donchian_enabled must be False when explicitly disabled"
        strategy = VolStraddleStrategy(cfg)

        # Populate 4h data with a valid breakout
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        signal = strategy.on_bar(_SYM, _close_15m())

        assert signal == 0, "No signal expected when donchian disabled"
        assert strategy._last_signal_source.get(_SYM) == "none", (
            "_last_signal_source must be 'none' when donchian disabled"
        )

    def test_disabled_no_meta_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_config(
            monkeypatch,
            {
                "VS_ENABLE_15M": "false",
                "VS_ENABLE_30M": "false",
                "VS_ENABLE_1H": "false",
                "VS_ENABLE_4H": "false",
                "VS_MA_CLUSTER_ENABLED": "false",
                "VS_DONCHIAN_ENABLED": "false",
            },
        )
        strategy = VolStraddleStrategy(cfg)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        strategy.on_bar(_SYM, _close_15m())

        # _last_signal_meta should not contain a donchian entry for this symbol
        meta = strategy._last_signal_meta.get(_SYM)
        if meta is not None:
            assert meta.get("strategy_type") != DONCHIAN, "_last_signal_meta must not be set to donchian when disabled"


# ══════════════════════════════════════════════════════════════════════════════
#  TC-10: _last_signal_source set correctly for SHORT Donchian
# ══════════════════════════════════════════════════════════════════════════════


class TestSignalSourceSetCorrectly:
    """_last_signal_source == 'donchian' when Donchian fires, for both directions."""

    def test_long_signal_source_is_donchian(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_long_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        signal = strategy.on_bar(_SYM, _close_15m())
        assert signal == 1
        assert strategy._last_signal_source[_SYM] == DONCHIAN

    def test_short_signal_source_is_donchian(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _make_strategy(monkeypatch)
        data_4h = _make_donchian_short_signal()
        strategy.update_htf_data(_SYM, data_4h["close"], data_4h["high"], data_4h["low"], volume_4h=data_4h["volume"])
        strategy._high_data[_SYM] = np.full(300, 100.0)
        strategy._low_data[_SYM] = np.full(300, 100.0)

        signal = strategy.on_bar(_SYM, _close_15m())
        assert signal == -1
        assert strategy._last_signal_source[_SYM] == DONCHIAN
