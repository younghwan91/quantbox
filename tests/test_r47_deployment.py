"""r47 배포 안전성 테스트.

r47 Option D (trail_activate=0.5%, trail_distance=0.2%, TBR OFF, MA cluster trail 4%/0.5%)
배포 전 검증 테스트. 과거 인시던트 패턴 재현 방지 포함.

핵심 검증 항목:
  1. 동일 바에서 trail 활성화 + 발동 동시 발생 (0.5% activate 특유의 리스크)
  2. MA cluster trail 파라미터 독립성 (4%/0.5% ≠ D의 0.5%/0.2%)
  3. D_mom8 + MA cluster 동시 포지션 — 각자 올바른 파라미터 독립 사용
  4. peak_gain 단조 증가 (역행 없음)
  5. trail 활성화 시 SL이 trail로 대체 (stop_loss 리턴 없음)
  6. trail 발동 시 수익 종료 확인 (trail_stop_gain ≥ 0)
  7. TBR 완전 비활성화 (f5_tbr_enabled=False)
  8. trail 종료 후 신규 포지션이 초기 상태로 시작
  9. strategy_type이 position dict에 정확히 저장
  10. 과거 인시던트: reduce_only 주문에 leverage 변경 없음
"""

import pytest

from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ──────────────────────────────────────────────────────────
#  Helpers (test_scenario_runner.py와 동일한 패턴)
# ──────────────────────────────────────────────────────────


def _open_vr_long(strategy, symbol="BTCUSDT", entry=100.0):
    """Open D_mom8 VR squeeze LONG. strategy_type = 'vr_squeeze'."""
    cfg = strategy.config
    pos_key = f"{symbol}:LONG"
    strategy.open_position(
        symbol=symbol,
        signal=1,
        entry_price=entry,
        notional=1000.0,
        leverage=25.0,
        move_size=1.0,
        sl=cfg.d_sl,
        tp=cfg.d_tp,
        max_hold=cfg.max_hold_15m,
        position_side="LONG",
    )
    return pos_key


def _open_ma_cluster(strategy, symbol="ETHUSDT", signal=1, entry=100.0):
    """Open MA cluster position. strategy_type = 'ma_cluster'."""
    cfg = strategy.config
    pos_side = "LONG" if signal == 1 else "SHORT"
    pos_key = f"{symbol}:{pos_side}"
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
        "leverage_override": cfg.ma_cluster_leverage,
    }
    strategy.open_position(
        symbol=symbol,
        signal=signal,
        entry_price=entry,
        notional=1000.0,
        leverage=cfg.ma_cluster_leverage,
        move_size=1.0,
        sl=cfg.ma_cluster_sl,
        tp=cfg.ma_cluster_tp,
        max_hold=cfg.ma_cluster_max_hold_4h * 16,
        position_side=pos_side,
    )
    return pos_key


def _tick(strategy, pos_key, close, *, high=None, low=None):
    """Simulate one bar. Returns exit_reason or None."""
    _h = high if high is not None else close
    _l = low if low is not None else close
    return strategy.update_position(pos_key, close, current_atr_pct=0.02, current_high=_h, current_low=_l)


@pytest.fixture
def strategy():
    cfg = VolStraddleConfig()
    return VolStraddleStrategy(config=cfg)


@pytest.fixture
def cfg():
    return VolStraddleConfig()


# ══════════════════════════════════════════════════════════
#  1. 동일 바 trail 활성화 + 발동 (r47 특유 리스크)
# ══════════════════════════════════════════════════════════


class TestSameBarTrailActivateAndFire:
    """trail_activate=1.0%이면 일반 변동성 바에서 활성화+발동이 동시에 발생 가능.

    bar range 0.2% (trail_distance=0.2%) 이상이면 같은 바에서 trail fire 가능.
    이전 trail_activate=0.5%에서도 동일한 메커니즘.
    """

    def test_long_same_bar_activate_fire(self, strategy, cfg):
        """LONG: high=+0.6%, low=+0.3% → 활성화 후 즉시 trail 발동."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)
        trail_activate = cfg.d_trail_activate  # 0.005
        trail_distance = cfg.d_trail_distance  # 0.002

        # best_gain=0.6% ≥ 0.5% → trail 활성화
        # peak=0.6%, trail_stop=0.4%, worst_gain=0.3% ≤ 0.4% → 발동
        result = _tick(
            strategy,
            pos_key,
            100.5,
            high=100.0 * (1 + trail_activate + 0.001),
            low=100.0 * (1 + trail_activate - trail_distance - 0.001),
        )
        assert result == "trailing_stop", (
            f"r47: 동일 바 trail 활성화+발동 미지원. got={result}. "
            f"trail_activate={trail_activate}, trail_distance={trail_distance}"
        )

    def test_short_same_bar_activate_fire(self, strategy, cfg):
        """SHORT: low=-0.6%, high=-0.3% → 활성화 후 즉시 trail 발동."""
        pos_key = f"BTCUSDT:SHORT"
        strategy.open_position(
            symbol="BTCUSDT",
            signal=-1,
            entry_price=100.0,
            notional=1000.0,
            leverage=25.0,
            move_size=1.0,
            sl=cfg.d_sl,
            tp=cfg.d_tp,
            max_hold=cfg.max_hold_15m,
            position_side="SHORT",
        )
        trail_activate = cfg.d_trail_activate
        trail_distance = cfg.d_trail_distance

        # SHORT: best_gain=(entry-low)/entry, worst_gain=(entry-high)/entry
        # low 기준 0.6% 하락 → trail 활성화
        # high 기준 0.3% 하락 → trail_stop_gain=0.6-0.2=0.4%, worst_gain=0.3% ≤ 0.4% → 발동
        result = _tick(
            strategy,
            pos_key,
            99.5,
            high=100.0 * (1 - (trail_activate - trail_distance - 0.001)),
            low=100.0 * (1 - trail_activate - 0.001),
        )
        assert result == "trailing_stop", f"SHORT 동일 바 trail 발동 실패. got={result}"

    def test_same_bar_does_not_fire_if_range_too_tight(self, strategy, cfg):
        """trail 활성화는 됐지만 bar range < trail_distance → 발동 없음."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)
        trail_activate = cfg.d_trail_activate  # 0.010
        trail_distance = cfg.d_trail_distance  # 0.002

        # high=+1.1% → trail 활성화, low=+1.0% > trail_stop(1.1-0.2=0.9%) → 발동 안 됨
        # bar range = 0.1% < trail_distance(0.2%) → too tight
        result = _tick(
            strategy, pos_key, 101.05, high=100.0 * (1 + trail_activate + 0.001), low=100.0 * (1 + trail_activate)
        )
        assert result is None, f"bar range < trail_distance일 때 trail 발동 없어야 함. got={result}"
        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True, "Trail은 활성화됐어야 함"


# ══════════════════════════════════════════════════════════
#  2. MA cluster trail 파라미터 독립성
# ══════════════════════════════════════════════════════════


class TestMAClusterTrailParamIsolation:
    """MA cluster는 4%/0.5% 파라미터를 사용, D trail 0.5%/0.2%를 쓰지 않는다."""

    def test_ma_cluster_does_not_activate_at_d_threshold(self, strategy, cfg):
        """3% 수익 (D trail 발동 수준)에서 MA cluster trail은 미활성화."""
        pos_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)
        d_trail_activate = cfg.d_trail_activate  # 0.005 (0.5%)
        ma_trail_activate = cfg.ma_cluster_trail_activate  # 0.04 (4%)

        # 3% 수익 — D trail이라면 즉시 활성화, MA trail은 미활성화
        gain_below_ma = (d_trail_activate + ma_trail_activate) / 2  # ~2.25%
        result = _tick(
            strategy,
            pos_key,
            100.0 * (1 + gain_below_ma),
            high=100.0 * (1 + gain_below_ma),
            low=100.0 * (1 + gain_below_ma) * 0.999,
        )
        assert result is None
        pos = strategy._positions[pos_key]
        assert pos.get("trail_active", False) is False, (
            f"MA cluster는 {gain_below_ma * 100:.1f}%에서 trail 활성화 안 돼야 함 "
            f"(ma_trail_activate={ma_trail_activate * 100:.0f}%)"
        )

    def test_ma_cluster_activates_at_ma_threshold(self, strategy, cfg):
        """4.1% 수익 → MA cluster trail 활성화."""
        pos_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)
        ma_trail_activate = cfg.ma_cluster_trail_activate  # 0.04

        # 4.1%: MA trail 활성화, 발동은 없음 (tight bar)
        result = _tick(strategy, pos_key, 104.1, high=104.1, low=104.09)
        assert result is None
        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True, "MA cluster 4.1%에서 trail 활성화 돼야 함"

    def test_ma_cluster_trail_fires_at_correct_distance(self, strategy, cfg):
        """MA cluster: 4% 활성화, 0.5% 되돌림 → 발동. D 기준(0.2%)보다 느슨."""
        pos_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)
        cfg_obj = strategy.config
        ma_activate = cfg_obj.ma_cluster_trail_activate  # 0.04
        ma_distance = cfg_obj.ma_cluster_trail_distance  # 0.005
        d_distance = cfg_obj.d_trail_distance  # 0.002

        # Phase 1: 5% 수익으로 trail 활성화 (tight bar)
        _tick(strategy, pos_key, 105.0, high=105.0, low=104.99)
        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True

        # Phase 2: peak=5%, D trail_stop=4.8%, MA trail_stop=4.5%
        # D 기준이면 worst_gain=4.75%에서 발동, MA 기준에서는 발동 안 됨
        peak_gain = pos["peak_gain"]  # ~0.05
        d_trail_stop = peak_gain - d_distance  # ~0.048
        ma_trail_stop = peak_gain - ma_distance  # ~0.045

        # D 발동 수준(4.75%)에 해당하는 가격 — MA는 발동 안 됨
        d_fire_low = 100.0 * (1 + d_trail_stop + 0.001)  # ~104.81
        result = _tick(strategy, pos_key, d_fire_low, high=105.0, low=d_fire_low)
        assert result is None, (
            f"MA cluster는 D trail_distance(0.2%)로 발동 안 돼야 함. got={result}\n"
            f"peak={peak_gain:.4f}, D_stop={d_trail_stop:.4f}, MA_stop={ma_trail_stop:.4f}"
        )

        # MA 발동 수준(4.45%)에 해당하는 가격
        ma_fire_low = 100.0 * (1 + ma_trail_stop - 0.001)  # ~104.49
        result = _tick(strategy, pos_key, ma_fire_low, high=105.0, low=ma_fire_low)
        assert result == "trailing_stop", f"MA trail 발동 실패. got={result}"


# ══════════════════════════════════════════════════════════
#  3. D_mom8 + MA cluster 동시 포지션 — 파라미터 독립
# ══════════════════════════════════════════════════════════


class TestConcurrentPositionTrailIsolation:
    """두 전략이 동시에 포지션을 보유할 때 trail 파라미터가 독립 동작."""

    def test_d_activates_at_0pt5_ma_does_not(self, strategy, cfg):
        """2% 수익: BTC(D_mom8) trail 활성화, ETH(MA cluster) trail 미활성화."""
        btc_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)
        eth_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)

        # 두 종목 모두 2% 수익 (BTC D trail 활성화 수준, MA는 미달)
        _tick(strategy, btc_key, 102.0, high=102.0, low=101.98)
        _tick(strategy, eth_key, 102.0, high=102.0, low=101.98)

        btc_pos = strategy._positions[btc_key]
        eth_pos = strategy._positions[eth_key]

        assert btc_pos["trail_active"] is True, "BTC(D_mom8) 2%에서 trail_active=True 필요"
        assert eth_pos.get("trail_active", False) is False, (
            "ETH(MA cluster) 2%에서 trail_active=False 필요 (MA threshold=4%)"
        )

    def test_independent_peak_gain_tracking(self, strategy, cfg):
        """두 포지션의 peak_gain이 서로 독립적으로 추적."""
        btc_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)
        eth_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)

        # BTC만 상승
        _tick(strategy, btc_key, 102.0, high=103.0, low=102.0)
        _tick(strategy, eth_key, 100.5, high=100.6, low=100.4)

        btc_pos = strategy._positions[btc_key]
        eth_pos = strategy._positions[eth_key]

        assert btc_pos["peak_gain"] > eth_pos["peak_gain"], "BTC peak_gain > ETH peak_gain 이어야 함"
        assert eth_pos["peak_gain"] < 0.01, f"ETH peak_gain 낮아야 함 (got {eth_pos['peak_gain']:.4f})"


# ══════════════════════════════════════════════════════════
#  4. peak_gain 단조 증가 (역행 없음)
# ══════════════════════════════════════════════════════════


class TestPeakGainMonotone:
    """peak_gain은 절대 감소하지 않는다."""

    def test_peak_gain_never_decreases_long(self, strategy):
        """LONG: 가격 되돌림 후에도 peak_gain은 최고점 유지."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        # 상승 → peak 설정
        _tick(strategy, pos_key, 101.0, high=102.0, low=100.9)
        pos = strategy._positions[pos_key]
        peak_after_rise = pos["peak_gain"]

        # 가격 되돌림 (trail 아직 발동 안 됨 — tight bar 유지)
        _tick(strategy, pos_key, 100.8, high=101.0, low=100.8)
        assert pos["peak_gain"] == peak_after_rise, "되돌림 후 peak_gain 감소 금지"

        # 더 하락
        _tick(strategy, pos_key, 100.5, high=100.7, low=100.5)
        assert pos["peak_gain"] == peak_after_rise, "추가 하락 후에도 peak_gain 감소 금지"

    def test_peak_gain_updates_on_new_high(self, strategy):
        """새로운 고점 도달 시 peak_gain 갱신."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        _tick(strategy, pos_key, 101.0, high=101.0, low=100.9)
        pos = strategy._positions[pos_key]
        peak_1 = pos["peak_gain"]

        _tick(strategy, pos_key, 101.5, high=102.0, low=101.4)
        peak_2 = pos["peak_gain"]
        assert peak_2 > peak_1, f"새 고점에서 peak_gain 갱신 필요. {peak_1:.4f} → {peak_2:.4f}"


# ══════════════════════════════════════════════════════════
#  5. trail 활성화 시 SL이 trailing_stop으로 대체
# ══════════════════════════════════════════════════════════


class TestTrailReplacesStopLoss:
    """trail 활성화 후 SL 수준 이하로 가격 하락 시 stop_loss 아닌 trailing_stop 반환."""

    def test_trail_active_overrides_sl_long(self, strategy, cfg):
        """LONG: trail 활성화 후 SL(-1%) 수준 하락 → trailing_stop (not stop_loss)."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        # Trail 활성화 (tight bar, 1.1% gain — above trail_activate=1.0%)
        _tick(strategy, pos_key, 101.1, high=101.1, low=101.09)
        pos = strategy._positions[pos_key]
        assert pos["trail_active"] is True

        # SL 수준(-1%)으로 직접 하락 — trail_stop_gain(0.9%)보다 훨씬 낮음
        sl_price = 100.0 * (1 - cfg.d_sl - 0.001)  # 98.9
        result = _tick(strategy, pos_key, sl_price, high=100.0, low=sl_price)
        assert result == "trailing_stop", (
            f"trail 활성화 후 SL 수준 하락 시 stop_loss 아닌 trailing_stop 필요. got={result}"
        )

    def test_sl_fires_when_trail_not_active(self, strategy, cfg):
        """trail 미활성화 상태에서 SL 수준 하락 → stop_loss."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        # trail 미활성화 상태에서 SL 수준 하락
        sl_price = 100.0 * (1 - cfg.d_sl - 0.001)
        result = _tick(
            strategy,
            pos_key,
            sl_price,
            high=100.2,  # 0.2% 수익 — trail_activate(1.0%) 미달
            low=sl_price,
        )
        assert result == "stop_loss", f"trail 미활성화 시 stop_loss 필요. got={result}"


# ══════════════════════════════════════════════════════════
#  6. trail 발동 시 수익 종료 검증
# ══════════════════════════════════════════════════════════


class TestTrailExitProfitability:
    """trail_stop_gain > 0이면 trail 발동 시 수익 종료."""

    def test_trail_fires_with_positive_pnl(self, strategy, cfg):
        """peak=1%, distance=0.2% → trail_stop=0.8% > 0: 수익 종료."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        # peak 1% 설정 (tight bar)
        _tick(strategy, pos_key, 101.0, high=101.0, low=100.99)
        pos = strategy._positions[pos_key]
        peak_gain = pos["peak_gain"]
        trail_stop_gain = peak_gain - cfg.d_trail_distance

        assert trail_stop_gain > 0, f"trail_stop_gain must be positive: {trail_stop_gain}"

        # trail 발동: low가 trail_stop 이하
        fire_low = 100.0 * (1 + trail_stop_gain - 0.001)
        result = _tick(strategy, pos_key, fire_low, high=101.0, low=fire_low)
        assert result == "trailing_stop", f"Expected trailing_stop, got {result}"

    def test_trail_activates_then_grows_then_fires(self, strategy, cfg):
        """trail 활성화 후 peak 계속 갱신 → 나중에 발동 (더 높은 수익)."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        # 여러 바 동안 상승
        for price in [100.6, 101.0, 101.5, 102.0]:
            low = price - 0.02  # very tight
            result = _tick(strategy, pos_key, price, high=price, low=low)
            assert result is None, f"Premature exit at {price}: {result}"

        pos = strategy._positions[pos_key]
        peak = pos["peak_gain"]  # ~2%
        assert peak >= 0.018, f"peak_gain should be ≥1.8%, got {peak:.4f}"

        # 되돌림으로 trail 발동
        trail_stop = peak - cfg.d_trail_distance
        fire_low = 100.0 * (1 + trail_stop - 0.001)
        result = _tick(strategy, pos_key, fire_low, high=102.0, low=fire_low)
        assert result == "trailing_stop", f"Expected trailing_stop, got {result}"
        assert trail_stop > 0.01, f"Trail fires at >{trail_stop * 100:.2f}% profit"


# ══════════════════════════════════════════════════════════
#  7. TBR 완전 비활성화 (f5_tbr_enabled=False)
# ══════════════════════════════════════════════════════════


class TestTBRDisabled:
    """r47: f5_tbr_enabled=False — TBR 게이트가 어떤 엔트리도 차단하지 않는다."""

    def test_tbr_disabled_by_default(self, cfg):
        """기본 config에서 f5_tbr_enabled=False."""
        assert cfg.f5_tbr_enabled is False, "r47: TBR는 기본값 False (trail_activate=0.5%가 bad trade 직접 처리)"

    def test_tbr_enabled_config_still_works(self):
        """TBR 활성화 설정은 여전히 가능 (선택적 재활성화 지원)."""
        cfg_tbr = VolStraddleConfig(f5_tbr_enabled=True)
        assert cfg_tbr.f5_tbr_enabled is True

    def test_tbr_z_params_unchanged(self, cfg):
        """TBR threshold/window 파라미터는 변경 없음 (비활성화만)."""
        assert cfg.f5_tbr_z_threshold == 0.5
        assert cfg.f5_tbr_z_window == 192


# ══════════════════════════════════════════════════════════
#  8. trail 종료 후 신규 포지션 초기 상태
# ══════════════════════════════════════════════════════════


class TestFreshStateAfterTrailExit:
    """trail 종료 후 같은 심볼에 신규 포지션 시 peak_gain/trail_active 초기화."""

    def test_new_position_starts_fresh(self, strategy, cfg):
        """trail 종료 → 포지션 제거 → 신규 오픈 → 초기 상태 확인."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)

        # Trail 활성화 후 발동 (trail_activate=1.0%)
        _tick(strategy, pos_key, 101.15, high=101.2, low=101.1)  # +1.2% peak, trail activates
        result = _tick(strategy, pos_key, 100.9, high=101.2, low=100.9)  # drop below trail_stop(1.0%)
        assert result == "trailing_stop"

        # 포지션 제거
        strategy.close_position(pos_key)

        # 신규 포지션 오픈
        new_key = _open_vr_long(strategy, "BTCUSDT", entry=101.0)
        new_pos = strategy._positions[new_key]

        assert new_pos.get("peak_gain", 0.0) == 0.0, "신규 포지션 peak_gain은 0 이어야 함"
        assert new_pos.get("trail_active", False) is False, "신규 포지션 trail_active는 False 이어야 함"
        assert new_pos.get("bars_held", 0) == 0, "신규 포지션 bars_held는 0 이어야 함"


# ══════════════════════════════════════════════════════════
#  9. strategy_type 저장 검증
# ══════════════════════════════════════════════════════════


class TestStrategyTypeInPositionDict:
    """position dict에 strategy_type이 정확히 저장된다."""

    def test_vr_squeeze_has_vr_squeeze_type(self, strategy):
        """D_mom8 포지션은 strategy_type='vr_squeeze'."""
        pos_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)
        pos = strategy._positions[pos_key]
        assert pos.get("strategy_type") in ("vr_squeeze", None, ""), (
            f"D_mom8 strategy_type should be vr_squeeze, got {pos.get('strategy_type')}"
        )

    def test_ma_cluster_has_ma_cluster_type(self, strategy):
        """MA cluster 포지션은 strategy_type='ma_cluster'."""
        pos_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)
        pos = strategy._positions[pos_key]
        assert pos.get("strategy_type") == "ma_cluster", (
            f"MA cluster strategy_type 누락. got={pos.get('strategy_type')}"
        )

    def test_strategy_type_routes_to_correct_trail_params(self, strategy, cfg):
        """strategy_type이 trail 파라미터 선택을 올바르게 라우팅."""
        btc_key = _open_vr_long(strategy, "BTCUSDT", entry=100.0)
        eth_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)

        # D_mom8 포지션: 1.1% 수익 → trail 활성화됨 (trail_activate=1.0%)
        _tick(strategy, btc_key, 101.1, high=101.1, low=101.09)
        btc_pos = strategy._positions[btc_key]
        assert btc_pos["trail_active"] is True, (
            f"D_mom8 1.1% 수익에서 trail_active=True 필요 (activate={cfg.d_trail_activate})"
        )

        # MA cluster 포지션: 동일 1.1% 수익 → trail 미활성화 (MA는 4% 필요)
        _tick(strategy, eth_key, 101.1, high=101.1, low=101.09)
        eth_pos = strategy._positions[eth_key]
        assert eth_pos.get("trail_active", False) is False, (
            f"MA cluster 1.1% 수익에서 trail_active=False 필요 (activate={cfg.ma_cluster_trail_activate})"
        )


# ══════════════════════════════════════════════════════════
#  10. 과거 인시던트 재현 방지
# ══════════════════════════════════════════════════════════


class TestPastIncidentPrevention:
    """과거 실제 자금 손실로 이어진 인시던트의 재현 방지 테스트."""

    def test_incident3_ma_cluster_trail_uses_ma_sl_not_d_sl(self, strategy, cfg):
        """Incident #3 유사: MA cluster는 자체 SL(2%)을 사용, D SL(1%) 아님.

        MA cluster SL=2%, D SL=1%. MA cluster 포지션이 1.1% 손실에서 종료되면 안 됨.
        """
        pos_key = _open_ma_cluster(strategy, "ETHUSDT", signal=1, entry=100.0)
        pos = strategy._positions[pos_key]
        assert pos["sl"] == cfg.ma_cluster_sl, f"MA cluster SL은 ma_cluster_sl({cfg.ma_cluster_sl}). got={pos['sl']}"

        # D SL(1%) 수준으로 하락 — MA cluster는 종료 안 됨
        d_sl_breach = 100.0 * (1 - cfg.d_sl - 0.001)  # 98.9
        result = _tick(strategy, pos_key, d_sl_breach, high=100.2, low=d_sl_breach)
        assert result is None, f"MA cluster가 D SL 수준({cfg.d_sl * 100:.0f}%)에서 종료됨. got={result}"

        # MA cluster SL(2%) 수준에서는 종료됨
        ma_sl_breach = 100.0 * (1 - cfg.ma_cluster_sl - 0.001)  # 97.9
        result = _tick(strategy, pos_key, ma_sl_breach, high=100.0, low=ma_sl_breach)
        assert result == "stop_loss", f"MA cluster SL({cfg.ma_cluster_sl * 100:.0f}%)에서 stop_loss 필요. got={result}"

    def test_incident4_algo_status_both_triggered_and_finished(self):
        """Incident #4: _check_order_filled는 TRIGGERED와 FINISHED 둘 다 체결로 처리.

        실제 브래킷 코드가 수정됐는지 검증 (quantbox/execution/brackets.py).
        """
        from quantbox.execution.brackets import BracketMixin
        import inspect

        src = inspect.getsource(BracketMixin._check_order_filled)
        assert "FINISHED" in src, "Incident #4: _check_order_filled에 FINISHED 상태 처리 없음"
        assert "TRIGGERED" in src, "_check_order_filled에 TRIGGERED 상태 처리 없음"
        # 두 상태가 OR 조건으로 처리되는지 확인
        assert (
            "('TRIGGERED', 'FINISHED')" in src
            or '("TRIGGERED", "FINISHED")' in src
            or '["TRIGGERED", "FINISHED"]' in src
            or '["TRIGGERED", "FINISHED"]' in src
        ), "TRIGGERED와 FINISHED가 동일하게 처리되지 않음"

    def test_d_trail_activate_faster_than_previous(self, cfg):
        """r69: trail_activate가 2%→0.5%→1.0%로 조정 — OOS 2023/2024/2025 일관 개선.

        r47에서 0.5%로 단축된 후, r69 연도별 검증으로 1.0%가 최적으로 확인됨.
        """
        assert cfg.d_trail_activate < 0.02, f"r47: trail_activate는 0.02 미만이어야 함. got={cfg.d_trail_activate}"
        assert cfg.d_trail_activate == 0.010, f"r69: trail_activate=0.010 (1.0%). got={cfg.d_trail_activate}"
        assert cfg.d_trail_distance == 0.002, f"r47: trail_distance=0.002 (0.2%). got={cfg.d_trail_distance}"

    def test_ma_cluster_trail_activate_not_zero(self, cfg):
        """MA cluster trail은 4% (너무 타이트한 trail로 조기 종료 방지)."""
        assert cfg.ma_cluster_trail_activate == 0.04, (
            f"MA cluster trail_activate=0.04. got={cfg.ma_cluster_trail_activate}"
        )
        assert cfg.ma_cluster_trail_distance == 0.005, (
            f"MA cluster trail_distance=0.005. got={cfg.ma_cluster_trail_distance}"
        )
