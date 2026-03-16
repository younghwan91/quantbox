"""Scenario runner: end-to-end integration tests for every exit path.

Paper mode 시나리오 러너 — 가격을 주입하여 SL/TP/trailing/max_hold/partial_close
각 경로가 실제 전략 코드(update_position)를 통해 올바르게 동작하는지 검증.

pytest로 실행 가능. 바이낸스 API 불필요.
모든 시나리오는 strategy.open_position → update_position 반복 → exit_reason 검증.

검증 항목:
  1. stop_loss: 가격 역행 → SL hit
  2. target: 가격 순행 → TP hit
  3. trailing_stop: 가격 순행 (trail activate) → 역행 → trail hit
  4. max_hold: 시간 초과 → 강제 청산
  5. ma_cluster SL/TP: MA Cluster 전략 별도 SL/TP 검증
  6. position sizing: balance/leverage/DD scaling 공식 검증
  7. MFE/MAE tracking: 생애 극값 정확성
"""

import numpy as np
import pytest

from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────


def _open_vr_long(
    strategy: VolStraddleStrategy,
    symbol: str = "TESTUSDT",
    entry_price: float = 100.0,
    notional: float = 1000.0,
    leverage: float = 25.0,
    sl: float | None = None,
    tp: float | None = None,
    max_hold: int | None = None,
) -> str:
    """Open a VR squeeze LONG position. Returns position key."""
    cfg = strategy.config
    _sl = sl if sl is not None else cfg.d_sl
    _tp = tp if tp is not None else cfg.d_tp
    _max_hold = max_hold if max_hold is not None else cfg.max_hold_15m

    pos_key = f"{symbol}:LONG" if cfg.hedge_mode else symbol
    strategy.open_position(
        symbol=symbol,
        signal=1,
        entry_price=entry_price,
        notional=notional,
        leverage=leverage,
        move_size=1.0,
        sl=_sl,
        tp=_tp,
        max_hold=_max_hold,
        position_side="LONG" if cfg.hedge_mode else None,
    )
    return pos_key


def _open_vr_short(
    strategy: VolStraddleStrategy,
    symbol: str = "TESTUSDT",
    entry_price: float = 100.0,
    notional: float = 1000.0,
    leverage: float = 25.0,
    sl: float | None = None,
    tp: float | None = None,
    max_hold: int | None = None,
) -> str:
    """Open a VR squeeze SHORT position. Returns position key."""
    cfg = strategy.config
    _sl = sl if sl is not None else cfg.d_sl
    _tp = tp if tp is not None else cfg.d_tp
    _max_hold = max_hold if max_hold is not None else cfg.max_hold_15m

    pos_key = f"{symbol}:SHORT" if cfg.hedge_mode else symbol
    strategy.open_position(
        symbol=symbol,
        signal=-1,
        entry_price=entry_price,
        notional=notional,
        leverage=leverage,
        move_size=1.0,
        sl=_sl,
        tp=_tp,
        max_hold=_max_hold,
        position_side="SHORT" if cfg.hedge_mode else None,
    )
    return pos_key


def _open_ma_cluster(
    strategy: VolStraddleStrategy,
    symbol: str = "TESTUSDT",
    signal: int = 1,
    entry_price: float = 100.0,
    notional: float = 1000.0,
    leverage: float = 12.0,
) -> str:
    """Open an MA Cluster position. Returns position key."""
    cfg = strategy.config
    pos_key = f"{symbol}:{'LONG' if signal == 1 else 'SHORT'}" if cfg.hedge_mode else symbol

    # Set signal metadata to ma_cluster type
    strategy._last_signal_meta[symbol] = {
        "tf": "4h",
        "sl": cfg.ma_cluster_sl,
        "tp": cfg.ma_cluster_tp,
        "max_hold": cfg.ma_cluster_max_hold_4h * 16,
        "vr": 0.0,
        "momentum": 0.05,
        "strategy_type": "ma_cluster",
        "ma_cluster_spread": 1.5,
        "ma_cluster_duration": 10,
        "leverage_override": leverage,
    }
    strategy.open_position(
        symbol=symbol,
        signal=signal,
        entry_price=entry_price,
        notional=notional,
        leverage=leverage,
        move_size=1.0,
        sl=cfg.ma_cluster_sl,
        tp=cfg.ma_cluster_tp,
        max_hold=cfg.ma_cluster_max_hold_4h * 16,
        position_side=("LONG" if signal == 1 else "SHORT") if cfg.hedge_mode else None,
    )
    return pos_key


def _tick(
    strategy: VolStraddleStrategy,
    pos_key: str,
    price: float,
    high: float | None = None,
    low: float | None = None,
) -> str | None:
    """Simulate one 15m bar. Returns exit_reason or None."""
    _high = high if high is not None else price
    _low = low if low is not None else price
    return strategy.update_position(
        pos_key,
        price,
        current_atr_pct=0.02,
        current_high=_high,
        current_low=_low,
    )


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> VolStraddleStrategy:
    """Fresh strategy with default config + hedge mode."""
    cfg = VolStraddleConfig(trail_server_native=False)
    return VolStraddleStrategy(cfg)


@pytest.fixture
def cfg() -> VolStraddleConfig:
    return VolStraddleConfig(trail_server_native=False)


# ══════════════════════════════════════════════════════════
#  Scenario 1: STOP LOSS — LONG
# ══════════════════════════════════════════════════════════


class TestStopLoss:
    """가격 역행 → SL hit 검증."""

    def test_long_sl_hit(self, strategy: VolStraddleStrategy, cfg: VolStraddleConfig):
        """LONG entry=100, SL=1% → low=98.9 triggers SL."""
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        sl_pct = cfg.d_sl  # 0.01
        sl_price = 100.0 * (1.0 - sl_pct)  # 99.0

        # Bar 1: price holds above SL
        reason = _tick(strategy, pos_key, 99.5, high=100.0, low=99.2)
        assert reason is None, f"Should hold, got {reason}"

        # Bar 2: low touches SL level exactly
        reason = _tick(strategy, pos_key, 99.0, high=99.5, low=sl_price)
        assert reason == "stop_loss", f"Expected stop_loss, got {reason}"

    def test_short_sl_hit(self, strategy: VolStraddleStrategy, cfg: VolStraddleConfig):
        """SHORT entry=100, SL=1% → high=101.1 triggers SL."""
        pos_key = _open_vr_short(strategy, entry_price=100.0)
        sl_pct = cfg.d_sl  # 0.01
        sl_price = 100.0 * (1.0 + sl_pct)  # 101.0

        # Bar 1: holds
        reason = _tick(strategy, pos_key, 100.5, high=100.8, low=100.0)
        assert reason is None

        # Bar 2: high pierces SL
        reason = _tick(strategy, pos_key, 100.8, high=sl_price + 0.1, low=100.5)
        assert reason == "stop_loss"

    def test_sl_not_triggered_when_trailing_active(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """SL check is skipped when trailing stop is active (R43 parity).

        Config: trail_activate=r47 value, trail_distance=r47 value, SL=1%.
        Trail fires as trailing_stop, never as stop_loss.
        """
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        trail_activate = cfg.d_trail_activate  # 0.02
        trail_distance = cfg.d_trail_distance  # 0.003

        # Activate trail with tight bar (low close to high → no trail hit yet)
        activate_price = 100.0 * (1.0 + trail_activate + 0.001)  # 102.1
        tight_low = activate_price * 0.999  # keep low tight
        reason = _tick(strategy, pos_key, activate_price, high=activate_price, low=tight_low)
        assert reason is None, "Should activate trail but not exit yet"

        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True, "Trail should be active"

        # Drop past where SL would be (entry * 0.99 = 99.0)
        # Since trail is active, this should trigger trailing_stop not stop_loss
        sl_breach = 100.0 * (1.0 - cfg.d_sl - 0.001)  # ~98.9
        reason = _tick(strategy, pos_key, sl_breach, high=100.0, low=sl_breach)
        assert reason == "trailing_stop", (
            f"When trailing active, exit must be trailing_stop not stop_loss, got {reason}"
        )


# ══════════════════════════════════════════════════════════
#  Scenario 2: TAKE PROFIT — Hard TP ceiling
# ══════════════════════════════════════════════════════════


class TestTakeProfit:
    """가격 순행 → TP hit 검증."""

    def test_long_tp_hit(self, strategy: VolStraddleStrategy, cfg: VolStraddleConfig):
        """LONG entry=100, TP=12% → high=112.1 triggers TP.

        r47: trail_activate=0.5%, trail_distance=0.2%. Tight bars (low=close-0.05)
        keep worst_gain above trail_stop so trail never fires before TP.
        """
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        tp_pct = cfg.d_tp  # 0.12
        tp_price = 100.0 * (1.0 + tp_pct)  # 112.0

        # Gradual rise with very tight bars (low = close - 0.05)
        for p in [101, 103, 105, 107, 109, 111]:
            reason = _tick(strategy, pos_key, p, high=p + 0.1, low=p - 0.05)
            assert reason is None, f"Premature exit at {p}: {reason}"

        # Hit TP with tight bar
        reason = _tick(strategy, pos_key, tp_price, high=tp_price + 0.1, low=tp_price - 0.05)
        assert reason == "target", f"Expected target, got {reason}"

    def test_short_tp_hit(self, strategy: VolStraddleStrategy, cfg: VolStraddleConfig):
        """SHORT entry=100, TP=12% → low=87.9 triggers TP.

        r47: tight-bar approach (high=close+0.05) keeps worst_gain above trail_stop.
        """
        pos_key = _open_vr_short(strategy, entry_price=100.0)
        tp_pct = cfg.d_tp
        tp_price = 100.0 * (1.0 - tp_pct)  # 88.0

        # Drop with tight bars (high close to close)
        for p in [98, 96, 94, 92, 90]:
            reason = _tick(strategy, pos_key, p, high=p + 0.05, low=p - 0.1)
            assert reason is None, f"Premature exit at {p}: {reason}"

        # Hit TP with tight bar
        reason = _tick(strategy, pos_key, tp_price, high=tp_price + 0.05, low=tp_price - 0.1)
        assert reason == "target", f"Expected target, got {reason}"


# ══════════════════════════════════════════════════════════
#  Scenario 3: TRAILING STOP
# ══════════════════════════════════════════════════════════


class TestTrailingStop:
    """Trailing stop activation + hit 검증 (R43)."""

    def test_trail_activation_and_hit_long(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """LONG: price rises to activate → retraces → trail_stop hit.

        r47: trail_activate=0.5%, trail_distance=0.2%.
        Activation bar must have tight low (close to high) to avoid
        same-bar trail trigger.
        """
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        trail_activate = cfg.d_trail_activate
        trail_distance = cfg.d_trail_distance

        # Phase 1: Rise to activate with tight bar
        peak = 100.0 * (1.0 + trail_activate + 0.005)  # 102.5
        tight_low = peak * 0.999  # low within 0.1% of peak
        reason = _tick(strategy, pos_key, peak, high=peak, low=tight_low)
        assert reason is None, f"Should activate trail but not exit, got {reason}"

        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True
        assert pos["peak_gain"] >= trail_activate

        # Phase 2: Continue rising (update peak) — tight bar
        higher_peak = peak * 1.005  # ~103.0
        reason = _tick(strategy, pos_key, higher_peak, high=higher_peak, low=higher_peak * 0.999)
        assert reason is None

        # Phase 3: Retrace past trail distance → exit
        peak_gain = pos["peak_gain"]
        trail_stop_level = 100.0 * (1.0 + peak_gain - trail_distance)
        low_breach = trail_stop_level - 0.5
        reason = _tick(strategy, pos_key, trail_stop_level, high=higher_peak - 1, low=low_breach)
        assert reason == "trailing_stop", f"Expected trailing_stop, got {reason}"

    def test_trail_activation_and_hit_short(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """SHORT: price drops to activate → bounces → trail_stop hit.

        r47: trail_activate=0.5%, trail_distance=0.2%.
        Activation bar must have tight high.
        """
        pos_key = _open_vr_short(strategy, entry_price=100.0)
        trail_activate = cfg.d_trail_activate
        trail_distance = cfg.d_trail_distance

        # Phase 1: Drop to activate with tight bar
        trough = 100.0 * (1.0 - trail_activate - 0.005)  # 97.5
        tight_high = trough * 1.001  # high within 0.1% of trough
        reason = _tick(strategy, pos_key, trough, high=tight_high, low=trough)
        assert reason is None, f"Should activate trail but not exit, got {reason}"

        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True

        # Phase 2: Drop more with tight bar
        lower = trough * 0.995  # ~97.0
        reason = _tick(strategy, pos_key, lower, high=lower * 1.001, low=lower)
        assert reason is None

        # Phase 3: Bounce past trail distance → exit
        peak_gain = pos["peak_gain"]
        trail_stop_gain = peak_gain - trail_distance
        # SHORT worst_gain = (entry - high) / entry
        # Trail fires when worst_gain <= trail_stop_gain
        # high that triggers: entry * (1 - trail_stop_gain) + margin
        high_breach = 100.0 * (1.0 - trail_stop_gain) + 0.5
        reason = _tick(strategy, pos_key, high_breach - 1, high=high_breach, low=lower + 1)
        assert reason == "trailing_stop", f"Expected trailing_stop, got {reason}"

    def test_trail_not_activated_below_threshold(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """Trailing stop should NOT activate below threshold."""
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        trail_activate = cfg.d_trail_activate

        # Rise but not enough
        almost = 100.0 * (1.0 + trail_activate * 0.8)
        reason = _tick(strategy, pos_key, almost, high=almost, low=100.0)
        assert reason is None

        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is False


# ══════════════════════════════════════════════════════════
#  Scenario 4: MAX HOLD — time-based exit
# ══════════════════════════════════════════════════════════


class TestMaxHold:
    """보유 기간 초과 → 강제 청산."""

    def test_max_hold_exit_at_limit(self, strategy: VolStraddleStrategy):
        """Position held exactly max_hold bars → triggers exit."""
        max_bars = 10  # Short for testing
        pos_key = _open_vr_long(
            strategy,
            entry_price=100.0,
            max_hold=max_bars,
        )

        # Tick max_hold-1 times at breakeven (no SL/TP hit)
        for i in range(max_bars - 1):
            reason = _tick(strategy, pos_key, 100.0)
            assert reason is None, f"Premature exit at bar {i + 1}: {reason}"

        # Bar max_hold → exit
        reason = _tick(strategy, pos_key, 100.0)
        assert reason == "max_hold", f"Expected max_hold, got {reason}"

    def test_max_hold_not_triggered_before_limit(
        self,
        strategy: VolStraddleStrategy,
    ):
        """3/10 bars held → no exit."""
        pos_key = _open_vr_long(
            strategy,
            entry_price=100.0,
            max_hold=10,
        )
        for _ in range(3):
            reason = _tick(strategy, pos_key, 100.0)
            assert reason is None


# ══════════════════════════════════════════════════════════
#  Scenario 5: MA CLUSTER — different SL/TP
# ══════════════════════════════════════════════════════════


class TestMACluster:
    """MA Cluster 전략의 SL/TP/max_hold 검증 (simple exit, no trailing)."""

    def test_ma_cluster_sl_hit(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """MA Cluster LONG, SL=2% → low=97.9 → stop_loss."""
        pos_key = _open_ma_cluster(strategy, signal=1, entry_price=100.0)
        sl = cfg.ma_cluster_sl  # 0.02

        # Verify strategy_type stored correctly
        pos = strategy._positions[pos_key]
        assert pos.get("strategy_type") == "ma_cluster"

        # Hold
        reason = _tick(strategy, pos_key, 99.0, high=99.5, low=98.5)
        assert reason is None

        # Hit SL
        sl_price = 100.0 * (1.0 - sl) - 0.1
        reason = _tick(strategy, pos_key, sl_price, high=99.0, low=sl_price)
        assert reason == "stop_loss"

    def test_ma_cluster_tp_hit(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """MA Cluster LONG hard TP ceiling at 8% (r47: trailing fires first normally).

        Tight bars (high=p+0.05, low=p) keep worst_gain above trail_stop,
        so hard TP ceiling is reached first.
        """
        pos_key = _open_ma_cluster(strategy, signal=1, entry_price=100.0)
        tp = cfg.ma_cluster_tp  # 0.08

        # Approach with tight bars — trail never fires
        for p in [102, 104, 106]:
            reason = _tick(strategy, pos_key, p, high=p + 0.05, low=p)
            assert reason is None, f"Premature exit at {p}: {reason}"

        # Hit hard TP ceiling
        tp_price = 100.0 * (1.0 + tp)
        reason = _tick(strategy, pos_key, tp_price, high=tp_price + 0.05, low=tp_price)
        assert reason == "target", f"Expected target, got {reason}"

    def test_ma_cluster_uses_ma_trail_params(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """MA Cluster uses ma_cluster_trail_activate (4%), not d_trail_activate (0.5%).

        A gain just below ma_cluster_trail_activate should NOT activate trailing.
        """
        pos_key = _open_ma_cluster(strategy, signal=1, entry_price=100.0)
        ma_trail_activate = cfg.ma_cluster_trail_activate  # 0.04

        # Rise to just below MA trail threshold (but well above d_trail_activate=0.5%)
        below_ma = 100.0 * (1.0 + ma_trail_activate * 0.9)  # 3.6% — above D threshold, below MA
        reason = _tick(strategy, pos_key, below_ma, high=below_ma, low=below_ma * 0.999)
        assert reason is None

        pos = strategy._positions[pos_key]
        assert pos.get("trail_active", False) is False, (
            "MA cluster should NOT activate trail below ma_cluster_trail_activate"
        )

    def test_ma_cluster_max_hold(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """MA Cluster max_hold = ma_cluster_max_hold_4h * 16 bars."""
        max_bars = cfg.ma_cluster_max_hold_4h * 16
        pos_key = _open_ma_cluster(strategy, signal=1, entry_price=100.0)

        # Tick up to max_hold-1
        for i in range(max_bars - 1):
            reason = _tick(strategy, pos_key, 100.0)
            assert reason is None, f"Premature exit at bar {i + 1}: {reason}"

        # Max hold bar
        reason = _tick(strategy, pos_key, 100.0)
        assert reason == "max_hold"


# ══════════════════════════════════════════════════════════
#  Scenario 6: MFE / MAE TRACKING
# ══════════════════════════════════════════════════════════


class TestMFEMAE:
    """MFE(best_pnl)와 MAE(worst_pnl)가 정확히 기록되는지 검증."""

    def test_mfe_records_peak(self, strategy: VolStraddleStrategy):
        """LONG: high=105 → MFE should be ~+5%."""
        pos_key = _open_vr_long(strategy, entry_price=100.0)

        _tick(strategy, pos_key, 103, high=105, low=101)
        pos = strategy._positions[pos_key]
        assert pos["best_pnl"] == pytest.approx(0.05, abs=0.001)

        # Next bar lower → MFE stays at peak
        _tick(strategy, pos_key, 101, high=102, low=100.5)
        assert pos["best_pnl"] == pytest.approx(0.05, abs=0.001)

    def test_mae_records_trough(self, strategy: VolStraddleStrategy):
        """LONG: low=97 → MAE should be ~-3%."""
        pos_key = _open_vr_long(strategy, entry_price=100.0)

        _tick(strategy, pos_key, 99, high=100, low=97)
        pos = strategy._positions[pos_key]
        assert pos["worst_pnl"] == pytest.approx(-0.03, abs=0.001)

        # Next bar higher → MAE stays at trough
        _tick(strategy, pos_key, 100, high=100.5, low=99)
        assert pos["worst_pnl"] == pytest.approx(-0.03, abs=0.001)

    def test_short_mfe_mae(self, strategy: VolStraddleStrategy):
        """SHORT: best when price drops, worst when price rises."""
        pos_key = _open_vr_short(strategy, entry_price=100.0)

        _tick(strategy, pos_key, 98, high=101, low=96)
        pos = strategy._positions[pos_key]
        # MFE = (entry - low) / entry = (100-96)/100 = 0.04
        assert pos["best_pnl"] == pytest.approx(0.04, abs=0.001)
        # MAE = (entry - high) / entry = (100-101)/100 = -0.01
        assert pos["worst_pnl"] == pytest.approx(-0.01, abs=0.001)


# ══════════════════════════════════════════════════════════
#  Scenario 7: POSITION SIZING VERIFICATION
# ══════════════════════════════════════════════════════════


class TestPositionSizing:
    """Position sizing 공식 검증 (전략 레벨)."""

    def test_vr_leverage_30x(self, cfg: VolStraddleConfig):
        """VR squeeze: base_leverage=30x (half-Kelly)."""
        assert cfg.base_leverage == 30.0

    def test_ma_cluster_leverage_21x(self, cfg: VolStraddleConfig):
        """MA cluster: leverage_override=21x (half-Kelly)."""
        assert cfg.ma_cluster_leverage == 21.0

    def test_slot_capital_10_concurrent(self, cfg: VolStraddleConfig):
        """max_concurrent=10 → per_slot = balance / 10."""
        balance = 1000.0
        per_slot = balance / cfg.max_concurrent
        assert per_slot == pytest.approx(100.0)

    def test_notional_equals_slot_times_leverage(self, cfg: VolStraddleConfig):
        """notional = per_slot_capital × leverage."""
        balance = 1000.0
        per_slot = balance / cfg.max_concurrent
        notional_vr = per_slot * cfg.base_leverage
        notional_ma = per_slot * cfg.ma_cluster_leverage

        assert notional_vr == pytest.approx(3000.0)
        assert notional_ma == pytest.approx(2100.0)


# ══════════════════════════════════════════════════════════
#  Scenario 8: SL/TP PRIORITY ORDER
# ══════════════════════════════════════════════════════════


class TestExitPriority:
    """같은 바에서 SL과 TP가 동시에 가능할 때 우선순위 검증."""

    def test_sl_and_tp_same_bar_long(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """LONG: high hits TP AND low hits SL in same bar.

        Actual priority: Trail check (activates + fires) > SL > TP.
        With trail_activate=2% and high at +12%, trail activates AND the
        large range will trigger trail_stop (since low is way below peak).
        This is the CORRECT production behavior.
        """
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        sl_pct = cfg.d_sl
        tp_pct = cfg.d_tp

        # Extreme bar: low below SL, high above TP
        extreme_low = 100.0 * (1.0 - sl_pct) - 1.0
        extreme_high = 100.0 * (1.0 + tp_pct) + 1.0
        reason = _tick(strategy, pos_key, 100.0, high=extreme_high, low=extreme_low)
        # Trail activates (high is +12% > 2%) and immediately fires
        # (low is way below peak - 0.3%)
        assert reason == "trailing_stop", f"Trail activates+fires on extreme bar, got {reason}"

    def test_sl_and_tp_same_bar_ma_cluster(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """MA Cluster r47: trailing stop checked before SL.

        When trail is NOT yet active (gain below ma_cluster_trail_activate),
        SL fires on extreme low. When trail IS active, trailing_stop fires instead.
        This test verifies SL fires when trail has not yet activated.
        """
        pos_key = _open_ma_cluster(strategy, signal=1, entry_price=100.0)
        sl_pct = cfg.ma_cluster_sl  # 2%

        # Bar that hits SL but NOT trail_activate (price rises 1.5%, below ma_cluster_trail_activate=2%)
        sl_low = 100.0 * (1.0 - sl_pct) - 1.0  # 97 — breaches SL
        below_trail_high = 100.0 * 1.015  # 101.5 — 1.5% gain, below ma_cluster_trail_activate (2%, r75)
        reason = _tick(strategy, pos_key, sl_low, high=below_trail_high, low=sl_low)
        assert reason == "stop_loss", f"Expected stop_loss when trail not active, got {reason}"


# ══════════════════════════════════════════════════════════
#  Scenario 9: DIRECTION COVERAGE (long + short symmetry)
# ══════════════════════════════════════════════════════════


class TestDirectionSymmetry:
    """LONG과 SHORT이 대칭적으로 동작하는지 검증."""

    def test_long_and_short_sl_symmetric(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """Same % move triggers SL for both directions."""
        sl_pct = cfg.d_sl

        # LONG: SL at entry * (1 - sl_pct)
        long_key = _open_vr_long(strategy, symbol="TEST1USDT", entry_price=100.0)
        long_sl = 100.0 * (1.0 - sl_pct) - 0.01
        reason = _tick(strategy, long_key, long_sl, low=long_sl)
        assert reason == "stop_loss"

        # SHORT: SL at entry * (1 + sl_pct)
        short_key = _open_vr_short(strategy, symbol="TEST2USDT", entry_price=100.0)
        short_sl = 100.0 * (1.0 + sl_pct) + 0.01
        reason = _tick(strategy, short_key, short_sl, high=short_sl)
        assert reason == "stop_loss"

    def test_long_and_short_tp_symmetric(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """Same % move triggers TP for both directions."""
        tp_pct = cfg.d_tp

        long_key = _open_vr_long(strategy, symbol="TEST3USDT", entry_price=100.0)
        long_tp = 100.0 * (1.0 + tp_pct) + 0.01
        reason = _tick(strategy, long_key, long_tp, high=long_tp)
        assert reason == "target"

        short_key = _open_vr_short(strategy, symbol="TEST4USDT", entry_price=100.0)
        short_tp = 100.0 * (1.0 - tp_pct) - 0.01
        reason = _tick(strategy, short_key, short_tp, low=short_tp)
        assert reason == "target"


# ══════════════════════════════════════════════════════════
#  Scenario 10: EDGE CASES
# ══════════════════════════════════════════════════════════


class TestEdgeCases:
    """경계값 및 에지 케이스 검증."""

    def test_exact_sl_boundary_long(self, strategy: VolStraddleStrategy, cfg: VolStraddleConfig):
        """Low == exact SL price → should trigger (worst_gain == -sl_pct)."""
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        sl_price = 100.0 * (1.0 - cfg.d_sl)
        reason = _tick(strategy, pos_key, sl_price, high=100.0, low=sl_price)
        assert reason == "stop_loss"

    def test_just_above_sl_boundary_long(
        self,
        strategy: VolStraddleStrategy,
        cfg: VolStraddleConfig,
    ):
        """Low just above SL → should NOT trigger."""
        pos_key = _open_vr_long(strategy, entry_price=100.0)
        almost_sl = 100.0 * (1.0 - cfg.d_sl) + 0.01
        reason = _tick(strategy, pos_key, almost_sl, high=100.0, low=almost_sl)
        assert reason is None

    def test_position_survives_sideways(self, strategy: VolStraddleStrategy):
        """50 bars of sideways movement → no exit."""
        pos_key = _open_vr_long(strategy, entry_price=100.0, max_hold=100)
        for _ in range(50):
            reason = _tick(strategy, pos_key, 100.0, high=100.2, low=99.8)
            assert reason is None

    def test_multiple_positions_independent(
        self,
        strategy: VolStraddleStrategy,
    ):
        """Multiple positions exit independently."""
        key1 = _open_vr_long(strategy, symbol="AAAUSDT", entry_price=100.0, max_hold=5)
        key2 = _open_vr_long(strategy, symbol="BBBUSDT", entry_price=200.0, max_hold=10)

        for _ in range(4):
            _tick(strategy, key1, 100.0)
            _tick(strategy, key2, 200.0)

        # key1 exits at bar 5
        reason1 = _tick(strategy, key1, 100.0)
        reason2 = _tick(strategy, key2, 200.0)
        assert reason1 == "max_hold"
        assert reason2 is None  # still 5 bars to go
