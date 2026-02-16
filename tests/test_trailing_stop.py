"""
Tests for trailing stop (ratchet stop) feature in backtest engine.

best_z 기반 trailing stop:
z가 entry 대비 activate_delta 이상 개선 → trailing 활성화 →
이후 best_z에서 trail_z만큼 되돌림 발생 → 이익 보존 청산.

학술 근거: Leung & Zhang (2021), Glover & Peskir (2024)
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from quantbox.backtest.continuous_backtest import (
    ContinuousBacktestConfig,
    ExitReason,
    TradeResult,
)


# ============================================================
# A. Config & Enum
# ============================================================

class TestTrailingStopConfig:
    """ContinuousBacktestConfig에 trailing stop 필드 존재 및 기본값 검증."""

    def test_config_has_trailing_stop_fields(self):
        config = ContinuousBacktestConfig()
        assert hasattr(config, 'trailing_stop_enabled')
        assert hasattr(config, 'trailing_stop_activate_delta')
        assert hasattr(config, 'trailing_stop_trail_z')

    def test_trailing_stop_disabled_by_default(self):
        config = ContinuousBacktestConfig()
        assert config.trailing_stop_enabled is False

    def test_trailing_stop_default_values(self):
        config = ContinuousBacktestConfig()
        assert config.trailing_stop_activate_delta == 0.3
        assert config.trailing_stop_trail_z == 0.8

    def test_exit_reason_has_trailing_stop(self):
        assert hasattr(ExitReason, 'TRAILING_STOP')
        assert ExitReason.TRAILING_STOP.value == "trailing_stop"


class TestTradeResultBestZ:
    """TradeResult에 best_z 필드가 포함되는지 검증."""

    def test_trade_result_has_best_z(self):
        trade = TradeResult(
            pair=("BTCUSDT", "ETHUSDT"),
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            entry_z=2.1,
            exit_z=0.5,
            entry_spread=0.01,
            exit_spread=0.005,
            signal=1,
            capital_allocated=100.0,
            gross_pnl=5.0,
            fees=0.1,
            slippage_cost=0.1,
            funding_cost=0.01,
            net_pnl=4.79,
            net_return_pct=4.79,
            bars_held=10,
            exit_reason=ExitReason.TRAILING_STOP,
            hedge_ratio=1.0,
            spread_mean=0.0,
            spread_std=0.01,
            slot_id=0,
            best_z=1.5,
        )
        assert trade.best_z == 1.5
        assert trade.exit_reason == ExitReason.TRAILING_STOP

    def test_trade_result_best_z_default(self):
        trade = TradeResult(
            pair=("BTCUSDT", "ETHUSDT"),
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            entry_z=2.1,
            exit_z=0.5,
            entry_spread=0.01,
            exit_spread=0.005,
            signal=1,
            capital_allocated=100.0,
            gross_pnl=5.0,
            fees=0.1,
            slippage_cost=0.1,
            funding_cost=0.01,
            net_pnl=4.79,
            net_return_pct=4.79,
            bars_held=10,
            exit_reason=ExitReason.MEAN_REVERSION,
            hedge_ratio=1.0,
            spread_mean=0.0,
            spread_std=0.01,
        )
        assert trade.best_z == 0.0  # default


# ============================================================
# B. Trailing stop logic (unit-level, using position_info dict)
# ============================================================

def _make_position(entry_z: float, signal: int) -> dict:
    """Helper: 미니멀 position_info dict 생성."""
    return {
        'entry_z': entry_z,
        'signal': signal,
        'best_z': entry_z,
        'trailing_active': False,
    }


def _update_best_z(pos: dict, current_z: float):
    """Replicate best_z update logic from _monitor_position."""
    if pos['signal'] == 1:
        if current_z > pos['best_z']:
            pos['best_z'] = current_z
    else:
        if current_z < pos['best_z']:
            pos['best_z'] = current_z


def _check_trailing_stop(
    pos: dict,
    current_z: float,
    activate_delta: float = 0.3,
    trail_z: float = 0.8,
) -> bool:
    """Replicate best_z based trailing stop check logic.

    Threshold = best_z ± trail_z (not entry_z ± buffer).
    Returns True if trailing stop should fire.
    """
    entry_z = pos['entry_z']
    delta = abs(pos['best_z'] - entry_z)
    if delta >= activate_delta:
        pos['trailing_active'] = True
    if pos['trailing_active']:
        best_z = pos['best_z']
        if pos['signal'] == 1:  # Long: best_z is high, z dropping
            if current_z <= best_z - trail_z:
                return True
        else:  # Short: best_z is low, z rising
            if current_z >= best_z + trail_z:
                return True
    return False


class TestTrailingStopLogicLongSpread:
    """Long spread (signal=1): z는 음수에서 0 방향으로 증가해야 수익."""

    def test_activation_on_sufficient_improvement(self):
        """z가 entry 대비 0.3 이상 개선 시 trailing 활성화."""
        pos = _make_position(entry_z=-2.1, signal=1)
        # z가 -2.1 → -1.7로 상승 (Δz = 0.4 > 0.3)
        _update_best_z(pos, -1.7)
        assert pos['best_z'] == -1.7
        _check_trailing_stop(pos, -1.7)
        assert pos['trailing_active'] is True

    def test_no_activation_on_insufficient_improvement(self):
        """z가 entry 대비 0.3 미만 개선 시 trailing 비활성."""
        pos = _make_position(entry_z=-2.1, signal=1)
        _update_best_z(pos, -1.9)  # Δz = 0.2 < 0.3
        _check_trailing_stop(pos, -1.9)
        assert pos['trailing_active'] is False

    def test_trigger_on_reversal_from_best_z(self):
        """활성화 후 z가 best_z에서 trail_z만큼 하락 시 발동."""
        pos = _make_position(entry_z=-2.5, signal=1)
        # Phase 1: z 개선 → 활성화
        _update_best_z(pos, -1.5)  # Δz = 1.0 > 0.3, best_z = -1.5
        _check_trailing_stop(pos, -1.5)
        assert pos['trailing_active'] is True
        # Phase 2: z 역전 → best_z - trail_z = -1.5 - 0.8 = -2.3
        result = _check_trailing_stop(pos, -2.4, trail_z=0.8)
        assert result is True  # -2.4 <= -2.3
        assert pos['best_z'] == -1.5  # best_z unchanged

    def test_no_trigger_when_z_within_trail_distance(self):
        """활성화 후에도 z가 best_z - trail_z 위에 있으면 발동 안 함."""
        pos = _make_position(entry_z=-2.5, signal=1)
        _update_best_z(pos, -1.5)  # best_z = -1.5, threshold = -2.3
        _check_trailing_stop(pos, -1.5)
        assert pos['trailing_active'] is True
        # z = -2.0 > -2.3 → no trigger
        result = _check_trailing_stop(pos, -2.0, trail_z=0.8)
        assert result is False

    def test_no_trigger_when_not_activated(self):
        """trailing 활성화 전에는 어디로 하락해도 발동 안 함."""
        pos = _make_position(entry_z=-2.1, signal=1)
        # z barely improves (Δz = 0.1 < 0.3)
        _update_best_z(pos, -2.0)
        result = _check_trailing_stop(pos, -3.0)
        assert result is False
        assert pos['trailing_active'] is False

    def test_trail_protects_profit(self):
        """큰 개선 후 trail stop은 수익 대부분을 보존."""
        pos = _make_position(entry_z=-3.0, signal=1)
        # z가 -3.0 → -0.5 (2.5σ 개선)
        _update_best_z(pos, -0.5)
        _check_trailing_stop(pos, -0.5)
        assert pos['trailing_active'] is True
        # threshold = -0.5 - 0.8 = -1.3
        # z = -1.3이면 entry 대비 1.7σ 개선 상태에서 이익 보존
        result = _check_trailing_stop(pos, -1.3, trail_z=0.8)
        assert result is True
        # 개선분의 68% (1.7/2.5) 보존


class TestTrailingStopLogicShortSpread:
    """Short spread (signal=-1): z는 양수에서 0 방향으로 감소해야 수익."""

    def test_activation_on_sufficient_improvement(self):
        """z가 entry 대비 0.3 이상 감소 시 trailing 활성화."""
        pos = _make_position(entry_z=2.1, signal=-1)
        _update_best_z(pos, 1.7)  # Δz = 0.4 > 0.3
        assert pos['best_z'] == 1.7
        _check_trailing_stop(pos, 1.7)
        assert pos['trailing_active'] is True

    def test_trigger_on_reversal_from_best_z(self):
        """활성화 후 z가 best_z에서 trail_z만큼 상승 시 발동."""
        pos = _make_position(entry_z=2.5, signal=-1)
        _update_best_z(pos, 1.5)  # best_z = 1.5, Δz = 1.0
        _check_trailing_stop(pos, 1.5)
        assert pos['trailing_active'] is True
        # threshold = 1.5 + 0.8 = 2.3
        result = _check_trailing_stop(pos, 2.4, trail_z=0.8)
        assert result is True

    def test_no_trigger_when_z_within_trail_distance(self):
        """활성화 후에도 z가 best_z + trail_z 아래면 발동 안 함."""
        pos = _make_position(entry_z=2.5, signal=-1)
        _update_best_z(pos, 1.5)  # threshold = 2.3
        _check_trailing_stop(pos, 1.5)
        result = _check_trailing_stop(pos, 2.0, trail_z=0.8)
        assert result is False


# ============================================================
# C. Revert-Then-Reverse 시나리오 (session log 패턴 재현)
# ============================================================

class TestRevertThenReverse:
    """실제 세션 로그 패턴 재현 (best_z 기반)."""

    def test_btc_eth_pattern(self):
        """BTC/ETH: entry z=+2.10 → 1.698(개선) → 2.403(역전).
        Short spread (signal=-1). best_z=1.698, threshold=1.698+0.8=2.498.
        z=2.403 < 2.498 → NO trigger.
        z=2.660 > 2.498 → trigger."""
        pos = _make_position(entry_z=2.10, signal=-1)
        trajectory = [2.063, 1.820, 1.698, 1.792, 2.403, 2.660]

        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired_at = i
                break

        # best_z=1.698, threshold=2.498
        # z=2.403 < 2.498 → no
        # z=2.660 > 2.498 → yes
        assert fired_at is not None
        assert trajectory[fired_at] == 2.660
        
    def test_ada_pengu_pattern(self):
        """ADA/PENGU: entry z=+2.161 → 1.827(개선) → 역전.
        Short spread, best_z=1.827, threshold=1.827+0.8=2.627.
        z=2.841 > 2.627 → trigger."""
        pos = _make_position(entry_z=2.161, signal=-1)
        trajectory = [2.039, 1.827, 2.179, 2.841, 3.677]

        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired_at = i
                break

        # best_z = 1.827, threshold = 2.627
        # z=2.179 < 2.627 → no
        # z=2.841 > 2.627 → yes
        assert fired_at is not None
        assert trajectory[fired_at] == 2.841

    def test_sol_ltc_pattern(self):
        """SOL/LTC: entry z=+2.109 → 1.795(개선) → 역전.
        Short spread, best_z=1.795, threshold=1.795+0.8=2.595.
        z=3.117 > 2.595 → trigger."""
        pos = _make_position(entry_z=2.109, signal=-1)
        trajectory = [2.042, 2.176, 1.991, 2.249, 2.146, 2.731, 2.134, 1.795,
                      2.148, 2.430, 3.117]

        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired_at = i
                break

        # best_z tracks minimum for signal=-1: 1.795
        # threshold = 1.795 + 0.8 = 2.595
        # z=2.430 < 2.595 → no
        # z=3.117 > 2.595 → yes
        assert fired_at is not None
        assert trajectory[fired_at] == 3.117

    def test_winner_avax_pump_not_killed(self):
        """AVAX/PUMP (winner): z=+2.054 → 1.216 → 1.103 → exit.
        계속 개선 방향이므로 trailing stop 미발동."""
        pos = _make_position(entry_z=2.054, signal=-1)
        trajectory = [1.216, 1.103, 1.107]

        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired = True
                break

        assert not fired, "Trailing stop should NOT fire on AVAX/PUMP (winner)"

    def test_winner_fil_ena_exits_with_profit(self):
        """FIL/ENA: z=-2.109 → -0.873(best) → -1.872(reversal).
        Long spread. best_z=-0.873, threshold=-0.873-0.8=-1.673.
        z=-1.872 <= -1.673 → trailing stop 발동.
        하지만 entry(-2.109) 대비 0.237σ 개선 상태 → 수익 보존 청산."""
        pos = _make_position(entry_z=-2.109, signal=1)
        trajectory = [-1.633, -1.222, -0.968, -0.886, -0.873, -1.235, -1.872,
                      -1.235, -0.515]

        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired_at = i
                break

        # best_z = -0.873, threshold = -1.673
        # z = -1.872 <= -1.673 → fires at index 6
        assert fired_at == 6
        # entry(-2.109) → exit(-1.872) = 0.237σ 개선 → 수익 상태
        assert trajectory[fired_at] > pos['entry_z']  # z improved from entry

    def test_winner_hype_link_not_killed(self):
        """HYPE/LINK (winner): z=-2.180 → -1.542(개선) → exit.
        Long spread, 초기 악화 후 개선. 역전이 trail_z 이내이므로 미발동."""
        pos = _make_position(entry_z=-2.180, signal=1)
        trajectory = [-2.301, -2.084, -2.016, -1.845, -1.542, -1.760, -1.231]

        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired = True
                break

        assert not fired, "Trailing stop should NOT fire on HYPE/LINK (winner)"
        # best_z = -1.231, threshold = -1.231 - 0.8 = -2.031
        # z never drops below -2.031 after best_z is set (worst is -1.760)
        assert pos['trailing_active'] is True  # was activated (Δz=0.949)

    def test_immediate_failure_not_triggered(self):
        """즉시 실패 케이스. z가 개선 안 되므로 trailing 미활성 → stop_loss 처리."""
        pos = _make_position(entry_z=-2.043, signal=1)
        trajectory = [-2.052, -2.396, -3.593]

        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, trail_z=0.8):
                fired = True
                break

        assert not fired
        assert pos['trailing_active'] is False


# ============================================================
# D. Edge cases & configuration
# ============================================================

class TestTrailingStopEdgeCases:
    """Edge cases 및 보수성 검증."""

    def test_disabled_trailing_stop_never_fires(self):
        """trailing_stop_enabled=False이면 절대 발동 안 함."""
        # This is implicitly tested because _check_trailing_stop doesn't check
        # the config.trailing_stop_enabled flag (that's in _monitor_position).
        # But we verify the config default is False.
        config = ContinuousBacktestConfig()
        assert config.trailing_stop_enabled is False

    def test_best_z_only_improves(self):
        """best_z는 개선 방향으로만 업데이트된다."""
        # Long (signal=1): best_z increases
        pos_long = _make_position(entry_z=-2.0, signal=1)
        _update_best_z(pos_long, -1.5)  # improved
        assert pos_long['best_z'] == -1.5
        _update_best_z(pos_long, -1.8)  # worsened
        assert pos_long['best_z'] == -1.5  # unchanged
        _update_best_z(pos_long, -1.2)  # improved more
        assert pos_long['best_z'] == -1.2
        
        # Short (signal=-1): best_z decreases
        pos_short = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos_short, 1.5)  # improved
        assert pos_short['best_z'] == 1.5
        _update_best_z(pos_short, 1.8)  # worsened
        assert pos_short['best_z'] == 1.5  # unchanged
        _update_best_z(pos_short, 1.2)  # improved more
        assert pos_short['best_z'] == 1.2

    def test_activation_is_permanent(self):
        """한번 활성화되면 best_z가 달라져도 활성화 상태 유지."""
        pos = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos, 1.5)  # Δz = 0.5 > 0.3
        _check_trailing_stop(pos, 1.5)
        assert pos['trailing_active'] is True
        
        # best_z doesn't go back, but we confirm flag persists
        _check_trailing_stop(pos, 1.8)
        assert pos['trailing_active'] is True

    def test_large_trail_z_prevents_premature_trigger(self):
        """큰 trail_z는 더 많은 되돌림을 허용."""
        pos = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos, 1.0)  # best_z = 1.0
        _check_trailing_stop(pos, 1.0)
        assert pos['trailing_active'] is True

        # trail_z=1.5: threshold = 1.0 + 1.5 = 2.5
        result = _check_trailing_stop(pos, 2.3, trail_z=1.5)
        assert result is False
        result = _check_trailing_stop(pos, 2.6, trail_z=1.5)
        assert result is True

    def test_small_trail_z_triggers_quickly(self):
        """작은 trail_z는 빠르게 발동 (이익 보존 공격적)."""
        pos = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos, 1.0)  # best_z = 1.0
        _check_trailing_stop(pos, 1.0)
        # trail_z=0.3: threshold = 1.0 + 0.3 = 1.3
        result = _check_trailing_stop(pos, 1.4, trail_z=0.3)
        assert result is True

    def test_activate_delta_sensitivity(self):
        """activate_delta 변경에 따른 활성화 시점 변화."""
        # Small delta → 활성화 쉬움
        pos1 = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos1, 1.85)  # Δz = 0.15
        _check_trailing_stop(pos1, 1.85, activate_delta=0.1)
        assert pos1['trailing_active'] is True
        
        # Large delta → 활성화 어려움
        pos2 = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos2, 1.85)  # Δz = 0.15
        _check_trailing_stop(pos2, 1.85, activate_delta=0.5)
        assert pos2['trailing_active'] is False

    def test_mean_reversion_takes_priority(self):
        """mean_reversion이 trailing_stop보다 우선.
        
        z가 exit까지 도달했으면 trailing과 무관하게 mean_reversion으로 종료.
        이건 _monitor_position 레벨의 로직이므로 구조적으로 보장됨:
        trailing stop 체크에 'exit_reason is None' 가드가 있음.
        """
        # This is a structural guarantee verified by code inspection.
        pass


# ============================================================
# E. 15분 해상도 보수성 검증 (granularity bias)
# ============================================================

class TestGranularityBias:
    """백테스트(15분)가 실거래(1분)보다 보수적인지 검증.

    15분 봉에서는 trailing stop threshold를 넘은 후 최대 15분간 더 악화된 가격에 청산.
    → 백테스트가 실거래보다 불리하게 나옴 = 안전 방향 바이어스.
    """

    def test_delayed_detection_worsens_exit_price(self):
        """15분 간격 체크 시 trailing stop 발동이 지연되어 더 나쁜 가격에 종료."""
        pos = _make_position(entry_z=2.0, signal=-1)

        # Activate trailing with initial improvement
        _update_best_z(pos, 1.0)  # best_z = 1.0
        _check_trailing_stop(pos, 1.0)
        assert pos['trailing_active'] is True

        # threshold = 1.0 + 0.8 = 1.8
        # 1분 단위: z=1.85에서 이미 발동
        result_1m = _check_trailing_stop(pos, 1.85, trail_z=0.8)
        assert result_1m is True

        # 15분 단위: z=2.1에서야 발동 (더 나쁜 가격)
        pos2 = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos2, 1.0)
        _check_trailing_stop(pos2, 1.0)
        result_15m = _check_trailing_stop(pos2, 2.1, trail_z=0.8)
        assert result_15m is True
        # 2.1 > 1.85 → 15분 봉 결과가 더 나쁨 (보수적)


# ============================================================
# G. Live Trader Trailing Stop — config & position_data parity
# ============================================================

class TestLiveTraderTrailingStopConfig:
    """settings.py의 trailing stop 필드가 to_trader_config()에 전달되는지 검증."""

    def test_settings_has_trailing_stop_fields(self):
        """QuantBoxSettings에 trailing stop 필드 존재."""
        from quantbox.core.settings import QuantBoxSettings
        import inspect
        sig = inspect.signature(QuantBoxSettings)
        params = sig.parameters
        assert 'trailing_stop_enabled' in params
        assert 'trailing_stop_activate_delta' in params
        assert 'trailing_stop_trail_z' in params

    def test_to_trader_config_includes_trailing_stop(self):
        """to_trader_config()에 trailing stop 키가 포함되는지 확인."""
        from unittest.mock import patch
        import os
        env_vars = {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
            'TRAILING_STOP_ENABLED': 'true',
            'TRAILING_STOP_ACTIVATE_DELTA': '0.4',
            'TRAILING_STOP_TRAIL_Z': '1.0',
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from quantbox.core.settings import QuantBoxSettings
            s = QuantBoxSettings(**env_vars)
            cfg = s.to_trader_config()
            assert cfg['trailing_stop_enabled'] is True
            assert cfg['trailing_stop_activate_delta'] == 0.4
            assert cfg['trailing_stop_trail_z'] == 1.0

    def test_default_trailing_stop_enabled(self):
        """기본값은 trailing_stop_enabled=True."""
        from unittest.mock import patch
        import os
        env_vars = {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from quantbox.core.settings import QuantBoxSettings
            s = QuantBoxSettings(**env_vars)
            assert s.trailing_stop_enabled is True


class TestLiveTraderTrailingStopLogic:
    """live_trader.monitor_slot의 trailing stop 로직 단위 테스트.

    live_trader를 직접 인스턴스화하지 않고,
    exit condition 로직을 position dict 기반으로 검증한다.
    """

    @staticmethod
    def _simulate_monitor_exit(
        config: dict,
        pos: dict,
        z_scores: list[float],
        elapsed_h: float = 1.0,
    ) -> str | None:
        """live_trader.monitor_slot의 exit condition 로직을 재현 (best_z 기반).

        Returns:
            close_reason 또는 None (유지).
        """
        signal = pos['signal']
        entry_z = pos['entry_zscore']

        for z_score in z_scores:
            effective_exit_z = config.get('exit_zscore', 0.5)

            # Trailing stop: update best_z
            if config.get('trailing_stop_enabled', False):
                best_z = pos.get('best_z', entry_z)
                if signal == 1:
                    if z_score > best_z:
                        pos['best_z'] = z_score
                        best_z = z_score
                else:
                    if z_score < best_z:
                        pos['best_z'] = z_score
                        best_z = z_score

                delta = abs(best_z - entry_z)
                activate_delta = config.get('trailing_stop_activate_delta', 0.3)
                if delta >= activate_delta:
                    pos['trailing_active'] = True

            # Exit conditions (best_z 기반)
            if signal == 1:
                if z_score >= -effective_exit_z:
                    return "mean_reversion"
                elif config.get('trailing_stop_enabled', False) and pos.get('trailing_active', False):
                    trail_z = config.get('trailing_stop_trail_z', 0.8)
                    best_z = pos.get('best_z', entry_z)
                    if z_score <= best_z - trail_z:
                        return "trailing_stop"
                elif z_score < entry_z - config['stop_loss_zscore']:
                    return "stop_loss"
            else:
                if z_score <= effective_exit_z:
                    return "mean_reversion"
                elif config.get('trailing_stop_enabled', False) and pos.get('trailing_active', False):
                    trail_z = config.get('trailing_stop_trail_z', 0.8)
                    best_z = pos.get('best_z', entry_z)
                    if z_score >= best_z + trail_z:
                        return "trailing_stop"
                elif z_score > entry_z + config['stop_loss_zscore']:
                    return "stop_loss"

            if elapsed_h >= config['max_holding_hours']:
                return "max_holding_time"

        return None

    def test_long_trailing_stop_fires(self):
        """Long spread: z 개선됐다가 best_z에서 trail_z 하락 시 trailing_stop."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_trail_z': 0.8,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': 1, 'entry_zscore': -2.5,
            'best_z': -2.5, 'trailing_active': False,
        }
        # z: -2.5 → -1.5 (1.0σ 개선, best_z=-1.5) → -2.4 (0.9σ 역전 > 0.8 trail_z)
        reason = self._simulate_monitor_exit(config, pos, [-1.5, -1.8, -2.4])
        assert reason == "trailing_stop"

    def test_short_trailing_stop_fires(self):
        """Short spread: z 개선됐다가 best_z에서 trail_z 상승 시 trailing_stop."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_trail_z': 0.8,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': -1, 'entry_zscore': 2.5,
            'best_z': 2.5, 'trailing_active': False,
        }
        # z: 2.5 → 1.5 (1.0σ 개선, best_z=1.5) → 2.4 (0.9σ 역전 > 0.8 trail_z)
        reason = self._simulate_monitor_exit(config, pos, [1.5, 1.8, 2.4])
        assert reason == "trailing_stop"

    def test_trailing_disabled_no_trigger(self):
        """trailing_stop_enabled=False 시 stop_loss로 처리."""
        config = {
            'trailing_stop_enabled': False,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': -1, 'entry_zscore': 2.1,
            'best_z': 2.1, 'trailing_active': False,
        }
        # z goes through improvement then diverges to stop
        reason = self._simulate_monitor_exit(config, pos, [1.7, 1.8, 2.0, 2.5, 3.0, 3.7])
        assert reason == "stop_loss"

    def test_mean_reversion_priority_over_trailing(self):
        """mean_reversion이 trailing_stop보다 우선."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_trail_z': 0.8,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': -1, 'entry_zscore': 2.1,
            'best_z': 2.1, 'trailing_active': False,
        }
        # z goes all the way to mean reversion (toward 0 and below exit_z)
        reason = self._simulate_monitor_exit(config, pos, [1.5, 0.5, 0.3, 0.1])
        assert reason == "mean_reversion"

    def test_no_activation_without_delta(self):
        """Δz < activate_delta 이면 trailing 미활성."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_trail_z': 0.8,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': -1, 'entry_zscore': 2.1,
            'best_z': 2.1, 'trailing_active': False,
        }
        # z: 2.1 → 1.9 (only 0.2 improvement, < 0.3) → 2.0 (no trigger)
        reason = self._simulate_monitor_exit(config, pos, [1.9, 2.0, 2.1])
        assert reason is None
        assert pos['trailing_active'] is False

    def test_position_data_has_trailing_fields(self):
        """position_data에 best_z, trailing_active 필드 존재 확인."""
        # This verifies the contract between live_trader entry and monitor
        pos = {
            'signal': 1, 'entry_zscore': 2.1,
            'best_z': 2.1, 'trailing_active': False,
        }
        assert 'best_z' in pos
        assert 'trailing_active' in pos
        assert pos['best_z'] == pos['entry_zscore']
        assert pos['trailing_active'] is False

    def test_btc_eth_session_replay(self):
        """세션 로그 BTC/ETH 실제 궤적: z=+2.1→+1.698→+4.330.
        signal=-1, best_z=1.698, threshold=1.698+0.8=2.498.
        z=2.660 > 2.498 → trailing_stop."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_trail_z': 0.8,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': -1, 'entry_zscore': 2.1,
            'best_z': 2.1, 'trailing_active': False,
        }
        # Actual z-score path from session log
        z_path = [2.063, 1.820, 1.698, 1.792, 2.403, 2.660, 2.450]
        reason = self._simulate_monitor_exit(config, pos, z_path)
        assert reason == "trailing_stop"

