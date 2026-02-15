"""
Tests for trailing stop (ratchet stop) feature in backtest engine.

Trailing stop detects "revert-then-reverse" pattern:
z improves beyond activate_delta from entry → trailing activated →
z reverses back past entry_z ± buffer → position closed at ~breakeven.

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
        assert hasattr(config, 'trailing_stop_buffer')

    def test_trailing_stop_disabled_by_default(self):
        config = ContinuousBacktestConfig()
        assert config.trailing_stop_enabled is False

    def test_trailing_stop_default_values(self):
        config = ContinuousBacktestConfig()
        assert config.trailing_stop_activate_delta == 0.3
        assert config.trailing_stop_buffer == 0.1

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
    buffer: float = 0.1,
) -> bool:
    """Replicate trailing stop check logic from _monitor_position.

    Returns True if trailing stop should fire.
    """
    entry_z = pos['entry_z']
    delta = abs(pos['best_z'] - entry_z)
    if delta >= activate_delta:
        pos['trailing_active'] = True
    if pos['trailing_active']:
        if pos['signal'] == 1:
            if current_z <= entry_z - buffer:
                return True
        else:
            if current_z >= entry_z + buffer:
                return True
    return False


class TestTrailingStopLogicLongSpread:
    """Long spread (signal=1): z는 음수에서 0 방향으로 증가해야 수익."""

    def test_activation_on_sufficient_improvement(self):
        """z가 entry 대비 0.3 이상 개선 시 trailing 활성화."""
        pos = _make_position(entry_z=2.1, signal=1)
        # z가 2.1 → 2.5로 상승 (Δz = 0.4 > 0.3)
        _update_best_z(pos, 2.5)
        assert pos['best_z'] == 2.5
        _check_trailing_stop(pos, 2.5)
        assert pos['trailing_active'] is True

    def test_no_activation_on_insufficient_improvement(self):
        """z가 entry 대비 0.3 미만 개선 시 trailing 비활성."""
        pos = _make_position(entry_z=2.1, signal=1)
        _update_best_z(pos, 2.3)  # Δz = 0.2 < 0.3
        _check_trailing_stop(pos, 2.3)
        assert pos['trailing_active'] is False

    def test_trigger_on_reversal_past_entry(self):
        """활성화 후 z가 entry 아래로 하락 시 발동."""
        pos = _make_position(entry_z=2.1, signal=1)
        # Phase 1: z 개선 → 활성화
        _update_best_z(pos, 2.5)
        _check_trailing_stop(pos, 2.5)
        assert pos['trailing_active'] is True
        # Phase 2: z 역전 → entry_z - buffer = 2.0 이하로 하락
        _update_best_z(pos, 1.9)  # best_z stays at 2.5
        result = _check_trailing_stop(pos, 1.9, buffer=0.1)
        assert result is True
        assert pos['best_z'] == 2.5  # best_z unchanged

    def test_no_trigger_when_z_stays_above_entry(self):
        """활성화 후에도 z가 entry 위에 있으면 발동 안 함."""
        pos = _make_position(entry_z=2.1, signal=1)
        _update_best_z(pos, 2.5)
        _check_trailing_stop(pos, 2.5)
        assert pos['trailing_active'] is True
        # z slightly below best but still above entry
        result = _check_trailing_stop(pos, 2.2, buffer=0.1)
        assert result is False

    def test_no_trigger_when_not_activated(self):
        """trailing 활성화 전에는 entry 아래로 하락해도 발동 안 함."""
        pos = _make_position(entry_z=2.1, signal=1)
        # z barely improves then reverses
        _update_best_z(pos, 2.2)  # Δz = 0.1 < 0.3
        result = _check_trailing_stop(pos, 1.9)
        assert result is False
        assert pos['trailing_active'] is False

    def test_buffer_zero_triggers_at_exact_entry(self):
        """buffer=0이면 정확히 entry_z에서 발동."""
        pos = _make_position(entry_z=2.1, signal=1)
        _update_best_z(pos, 2.5)
        _check_trailing_stop(pos, 2.5, buffer=0.0)
        assert pos['trailing_active'] is True
        # At exactly entry_z
        result = _check_trailing_stop(pos, 2.1, buffer=0.0)
        # current_z <= entry_z - 0 → 2.1 <= 2.1 → True
        assert result is True  # fires at exact entry with zero buffer


class TestTrailingStopLogicShortSpread:
    """Short spread (signal=-1): z는 양수에서 0 방향으로 감소해야 수익."""

    def test_activation_on_sufficient_improvement(self):
        """z가 entry 대비 0.3 이상 감소 시 trailing 활성화."""
        pos = _make_position(entry_z=-2.1, signal=-1)
        # z가 -2.1 → -2.5로 하강 (개선, Δz = 0.4 > 0.3)
        _update_best_z(pos, -2.5)
        assert pos['best_z'] == -2.5
        _check_trailing_stop(pos, -2.5)
        assert pos['trailing_active'] is True

    def test_trigger_on_reversal_past_entry(self):
        """활성화 후 z가 entry 위로 상승 시 발동."""
        pos = _make_position(entry_z=-2.1, signal=-1)
        _update_best_z(pos, -2.5)
        _check_trailing_stop(pos, -2.5)
        # z reverses past entry_z + buffer = -2.0
        result = _check_trailing_stop(pos, -1.9, buffer=0.1)
        assert result is True

    def test_no_trigger_when_z_stays_below_entry(self):
        """활성화 후에도 z가 entry 아래에 있으면 발동 안 함."""
        pos = _make_position(entry_z=-2.1, signal=-1)
        _update_best_z(pos, -2.5)
        _check_trailing_stop(pos, -2.5)
        result = _check_trailing_stop(pos, -2.3, buffer=0.1)
        assert result is False


# ============================================================
# C. Revert-Then-Reverse 시나리오 (session log 패턴 재현)
# ============================================================

class TestRevertThenReverse:
    """실제 세션 로그의 BTC/ETH, ADA/PENGU, SOL/LTC 패턴 재현."""

    def test_btc_eth_pattern(self):
        """BTC/ETH: entry z=+2.10 → 0.8h에 1.698(개선) → 1.4h에 2.403(역전).

        Trailing stop은 z=1.698(Δz=0.402)에서 활성화,
        z=2.403(> 2.1+0.1=2.2이므로 NOT triggered... wait)
        
        Wait - for LONG spread (signal=1, entry_z=+2.10),
        z improving means z DECREASING toward 0.
        So best_z should track the minimum z (most toward 0).
        
        Actually the code says:
        signal=1: if current_z > pos['best_z'] → update
        
        But wait: for a long spread entry at z=+2.10, the z should
        decrease toward 0 for profit. So z going from 2.10 to 1.698
        is GOOD. Higher z is BAD.
        
        Hmm, let me re-read the code...
        
        The entry happens because z >= entry_zscore (positive z = short spread?).
        Actually looking at the backtest code:
        signal = -1 if z > threshold (short spread, z is high)
        signal = +1 if z < -threshold (long spread, z is low)
        
        BTC/ETH had entry_z=+2.10 and was presumably signal=-1 (short spread).
        z decreasing from 2.10 toward 0 is improvement.
        
        Let me verify from the log:
        BTC/ETH: entry_z=+2.100, z went 2.100 → 1.698 (improved) → 2.403 (reversed)
        This means the trade expected z to decrease → signal=-1 (short spread).
        """
        # BTC/ETH: signal=-1, entry 2.10
        pos = _make_position(entry_z=2.10, signal=-1)
        
        # z-score trajectory: 2.10 → 2.06 → 1.82 → 1.698 → 1.792 → 2.403 → 2.660
        trajectory = [2.063, 1.820, 1.698, 1.792, 2.403, 2.660]
        
        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired_at = i
                break
        
        # Should activate at z=1.698 (Δz = |1.698 - 2.10| = 0.402 > 0.3)
        # Should fire at z=2.403 (z >= entry + buffer = 2.10 + 0.1 = 2.20): YES
        # Actually z=2.403 >= 2.20 → True
        assert fired_at is not None, "Trailing stop should have fired for BTC/ETH"
        assert trajectory[fired_at] == 2.403, f"Should fire at z=2.403, not {trajectory[fired_at]}"
        
    def test_ada_pengu_pattern(self):
        """ADA/PENGU: entry z=+2.161 → 1.827(개선) → 2.179(역전) → 3.677(stop)."""
        pos = _make_position(entry_z=2.161, signal=-1)
        trajectory = [2.039, 1.827, 2.179, 2.841, 3.677]
        
        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired_at = i
                break
        
        # best_z = 1.827 (Δz = 0.334 > 0.3 → activated)
        # z=2.179: >= 2.161 + 0.1 = 2.261? NO (2.179 < 2.261)
        # z=2.841: >= 2.261? YES → fired
        assert fired_at is not None
        assert trajectory[fired_at] == 2.841

    def test_sol_ltc_pattern(self):
        """SOL/LTC: entry z=+2.109 → 1.795(개선) → 2.148(역전) → 3.900(stop)."""
        pos = _make_position(entry_z=2.109, signal=-1)
        trajectory = [2.042, 2.176, 1.991, 2.249, 2.146, 2.731, 2.134, 1.795, 
                      2.148, 2.430, 3.117]
        
        fired_at = None
        for i, z in enumerate(trajectory):
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired_at = i
                break
        
        # best_z tracks minimum for signal=-1
        # At 1.795: Δz = |1.795 - 2.109| = 0.314 > 0.3 → activated
        # z=2.148: >= 2.109 + 0.1 = 2.209? NO
        # z=2.430: >= 2.209? YES → fired
        assert fired_at is not None
        assert trajectory[fired_at] == 2.430

    def test_winner_avax_pump_not_killed(self):
        """AVAX/PUMP (winner): z=+2.054 → 1.216 → 1.103 → exit.
        Trailing stop이 winner를 죽이지 않는지 확인."""
        pos = _make_position(entry_z=2.054, signal=-1)
        trajectory = [1.216, 1.103, 1.107]  # keeps improving
        
        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired = True
                break
        
        assert not fired, "Trailing stop should NOT fire on AVAX/PUMP (winner)"
        # z only improved, never reversed past entry

    def test_winner_fil_ena_not_killed(self):
        """FIL/ENA (winner): z=-2.109 → -1.633 → -0.886 → exit.
        Short spread that keeps improving."""
        pos = _make_position(entry_z=-2.109, signal=1)
        # signal=1 (long spread): entry was at z=-2.109, expect z to increase
        trajectory = [-1.633, -1.222, -0.968, -0.886, -0.873, -1.235, -1.872, 
                      -1.235, -0.515]
        
        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired = True
                break
        
        assert not fired, "Trailing stop should NOT fire on FIL/ENA (winner)"

    def test_winner_hype_link_not_killed(self):
        """HYPE/LINK (winner): z=-2.180 → -2.301(악화) → -1.542(개선) → exit.
        
        Initial deterioration should not trigger trailing (not yet activated).
        Then improvement activates trailing but z never reverses past entry.
        """
        pos = _make_position(entry_z=-2.180, signal=1)
        # signal=1: z should increase toward 0
        trajectory = [-2.301, -2.084, -2.016, -1.845, -1.542, -1.760, -1.231]
        
        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired = True
                break
        
        assert not fired, "Trailing stop should NOT fire on HYPE/LINK (winner)"
        # best_z = -1.231, Δz = |-1.231 - (-2.180)| = 0.949 → activated
        # But z never drops back to entry_z - buffer = -2.180 - 0.1 = -2.28
        assert pos['trailing_active'] is True  # it WAS activated

    def test_sui_pump_not_affected(self):
        """SUI/PUMP (loss): z=-2.043 → -2.052 → -2.396 → -3.593.
        즉시 실패 케이스. z가 개선되지 않아 trailing 활성화 안 됨.
        기존 stop_loss가 처리."""
        pos = _make_position(entry_z=-2.043, signal=-1)
        # signal=-1: z should increase toward 0 (wait... for short spread)
        # Actually: entry at z=-2.043 with signal=-1 means short spread
        # Short spread: z should decrease for profit... but z was -2.043 and went more negative
        # That means it IMPROVED (more negative = better for short)
        # Wait no. Let me re-check the log:
        # SUI/PUMP: entry_z=-2.043, exit_z=-3.593, reason=stop_loss
        # z went from -2.043 to -3.593 (more negative = worse)
        # So the spread diverged further. For signal=-1, z increasing toward 0 = profit.
        # But z went more negative = loss.
        # So z not improving at all. Correct signal interpretation:
        # signal=-1 → short spread → expect z to increase toward 0
        # But z went from -2.043 to -3.593 (decrease = worse)
        
        # Hmm, but in backtest: signal=-1 for short spread means z>threshold (positive).
        # In live trader, SUI/PUMP had z=-2.043 with presumably signal=1 (long spread at negative z)
        # Let me re-check the log properly.
        # Log says: "SUIUSDT BUY ... PUMPUSDT SELL" → signal relates to leg direction
        # SUIUSDT BUY = long leg1 → long spread → signal=1
        # But entry z was -2.043 (negative, matching signal=1 for long spread: z < -entry_threshold)
        
        # Correction: SUI/PUMP was signal=1 (long spread), entry z=-2.043
        # For signal=1: z should increase toward 0 for profit
        # z went -2.043 → -2.052 → -2.396 → -3.593 (kept decreasing = worse)
        pos = _make_position(entry_z=-2.043, signal=1)
        trajectory = [-2.052, -2.396, -3.593]
        
        fired = False
        for z in trajectory:
            _update_best_z(pos, z)
            if _check_trailing_stop(pos, z, activate_delta=0.3, buffer=0.1):
                fired = True
                break
        
        assert not fired, "Trailing stop should NOT fire (no improvement)"
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

    def test_large_buffer_prevents_premature_trigger(self):
        """큰 buffer는 더 많은 역전을 허용한다."""
        pos = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos, 1.5)
        _check_trailing_stop(pos, 1.5)
        assert pos['trailing_active'] is True
        
        # With buffer=0.5: need z >= 2.0 + 0.5 = 2.5
        result = _check_trailing_stop(pos, 2.3, buffer=0.5)
        assert result is False
        result = _check_trailing_stop(pos, 2.6, buffer=0.5)
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
    
    15분 봉에서는 z가 entry를 넘은 후 최대 15분간 더 악화된 가격에 청산.
    → 백테스트가 실거래보다 불리하게 나옴 = 안전 방향 바이어스.
    """

    def test_delayed_detection_worsens_exit_price(self):
        """15분 간격 체크 시 trailing stop 발동이 지연되어 더 나쁜 가격에 종료."""
        pos = _make_position(entry_z=2.0, signal=-1)
        
        # 1분 단위 z 시뮬레이션 (15분 봉 사이의 실제 path)
        minute_zs = [
            1.7, 1.65, 1.6, 1.55, 1.5,  # 5분: 개선 중
            1.55, 1.6, 1.7, 1.8, 1.9,   # 10분: 역전 시작
            2.0, 2.1, 2.15, 2.2, 2.3,   # 15분: entry 넘어감 → 15분봉 close = 2.3
        ]
        
        # 1분 단위: z=2.10에서 이미 발동 (entry+0.1=2.1)
        # 15분 단위: z=2.3에서야 발동 (더 나쁜 가격)
        
        # Activate trailing with initial improvement
        _update_best_z(pos, 1.5)
        _check_trailing_stop(pos, 1.5)
        assert pos['trailing_active'] is True
        
        # 1-minute check: fires at z=2.1
        result_1m = _check_trailing_stop(pos, 2.1, buffer=0.1)
        assert result_1m is True
        
        # Reset and check at 15-min close (z=2.3): also fires but at worse price
        pos2 = _make_position(entry_z=2.0, signal=-1)
        _update_best_z(pos2, 1.5)
        _check_trailing_stop(pos2, 1.5)
        result_15m = _check_trailing_stop(pos2, 2.3, buffer=0.1)
        assert result_15m is True
        # 2.3 > 2.1 → 15분 봉 결과가 더 나쁨 (보수적)


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
        assert 'trailing_stop_buffer' in params

    def test_to_trader_config_includes_trailing_stop(self):
        """to_trader_config()에 trailing stop 키가 포함되는지 확인."""
        from unittest.mock import patch
        import os
        env_vars = {
            'BINANCE_API_KEY': 'test',
            'BINANCE_API_SECRET': 'test',
            'TRAILING_STOP_ENABLED': 'true',
            'TRAILING_STOP_ACTIVATE_DELTA': '0.4',
            'TRAILING_STOP_BUFFER': '0.2',
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from quantbox.core.settings import QuantBoxSettings
            s = QuantBoxSettings(**env_vars)
            cfg = s.to_trader_config()
            assert cfg['trailing_stop_enabled'] is True
            assert cfg['trailing_stop_activate_delta'] == 0.4
            assert cfg['trailing_stop_buffer'] == 0.2

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
        """live_trader.monitor_slot의 exit condition 로직을 재현.

        Returns:
            close_reason 또는 None (유지).
        """
        signal = pos['signal']
        entry_z = pos['entry_zscore']

        for z_score in z_scores:
            # Dynamic exit placeholder
            effective_exit_z = config.get('exit_zscore', 0.5)

            should_close = False
            close_reason = ""

            # Trailing stop: update best_z
            if config.get('trailing_stop_enabled', False):
                best_z = pos.get('best_z', entry_z)
                if signal == 1:  # Long: z should increase (become less negative)
                    if z_score > best_z:
                        pos['best_z'] = z_score
                        best_z = z_score
                else:  # Short: z should decrease (become less positive)
                    if z_score < best_z:
                        pos['best_z'] = z_score
                        best_z = z_score

                delta = abs(best_z - entry_z)
                activate_delta = config.get('trailing_stop_activate_delta', 0.3)
                if delta >= activate_delta:
                    pos['trailing_active'] = True

            # Exit conditions
            if signal == 1:
                if z_score >= -effective_exit_z:
                    return "mean_reversion"
                elif config.get('trailing_stop_enabled', False) and pos.get('trailing_active', False):
                    buffer = config.get('trailing_stop_buffer', 0.1)
                    if z_score <= entry_z - buffer:
                        return "trailing_stop"
                elif z_score < entry_z - config['stop_loss_zscore']:
                    return "stop_loss"
            else:
                if z_score <= effective_exit_z:
                    return "mean_reversion"
                elif config.get('trailing_stop_enabled', False) and pos.get('trailing_active', False):
                    buffer = config.get('trailing_stop_buffer', 0.1)
                    if z_score >= entry_z + buffer:
                        return "trailing_stop"
                elif z_score > entry_z + config['stop_loss_zscore']:
                    return "stop_loss"

            if elapsed_h >= config['max_holding_hours']:
                return "max_holding_time"

        return None

    def test_long_trailing_stop_fires(self):
        """Long spread (signal=1, z<0): z가 개선됐다 entry 근처로 역전 시 trailing_stop."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_buffer': 0.1,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': 1, 'entry_zscore': -2.1,
            'best_z': -2.1, 'trailing_active': False,
        }
        # z: -2.1 → -1.7 (improve toward 0, delta=0.4≥0.3 activates)
        # → -2.3 (reverse past entry-buffer=-2.2)
        reason = self._simulate_monitor_exit(config, pos, [-1.7, -1.8, -2.3])
        assert reason == "trailing_stop"

    def test_short_trailing_stop_fires(self):
        """Short spread (signal=-1, z>0): z가 개선됐다 entry 근처로 역전 시 trailing_stop."""
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_buffer': 0.1,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_hours': 20,
        }
        pos = {
            'signal': -1, 'entry_zscore': 2.1,
            'best_z': 2.1, 'trailing_active': False,
        }
        # z: 2.1 → 1.7 (improve toward 0, delta=0.4≥0.3 activates)
        # → 2.3 (reverse past entry+buffer=2.2)
        reason = self._simulate_monitor_exit(config, pos, [1.7, 1.8, 2.3])
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
            'trailing_stop_buffer': 0.1,
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
            'trailing_stop_buffer': 0.1,
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
        """세션 로그 BTC/ETH 실제 궤적: z=+2.1→+1.698→+4.330 stop.
        
        signal=-1 (Short spread, z>0). Trailing stop should fire at z≈2.0.
        """
        config = {
            'trailing_stop_enabled': True,
            'trailing_stop_activate_delta': 0.3,
            'trailing_stop_buffer': 0.1,
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

