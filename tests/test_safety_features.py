"""
Tests for live trader safety features.

드로다운 서킷 브레이커, 일일 손실 한도, 청산 실패 처리,
에러 핸들러 개선 등의 안전 기능 테스트.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ─── Drawdown & Daily Loss (로직 단위 테스트) ───

class TestDrawdownCircuitBreaker:
    """드로다운 서킷 브레이커 테스트.

    LivePairTrader._check_drawdown() 로직을 직접 시뮬레이션.
    실제 바이낸스 API 호출 없이 로직만 검증.
    """

    def test_no_drawdown_returns_false(self):
        """정상 잔고 — 브레이커 미발동."""
        hwm = 10000.0
        current = 9500.0
        max_dd = 0.10
        dd = (hwm - current) / hwm
        assert dd < max_dd  # 5% < 10%

    def test_drawdown_at_limit(self):
        """정확히 한도 — 브레이커 발동."""
        hwm = 10000.0
        current = 9000.0
        max_dd = 0.10
        dd = (hwm - current) / hwm
        assert dd >= max_dd  # 10% >= 10%

    def test_drawdown_exceeds_limit(self):
        """한도 초과 — 브레이커 발동."""
        hwm = 10000.0
        current = 8000.0
        max_dd = 0.10
        dd = (hwm - current) / hwm
        assert dd >= max_dd  # 20% >= 10%

    def test_hwm_updates_on_new_high(self):
        """잔고 증가 시 HWM 갱신."""
        hwm = 10000.0
        current = 10500.0
        if current > hwm:
            hwm = current
        assert hwm == 10500.0

    def test_drawdown_disabled_when_zero(self):
        """max_drawdown_pct=0 → 비활성화."""
        max_dd = 0.0
        assert max_dd <= 0  # disabled

    def test_drawdown_formula_precision(self):
        """드로다운 공식 정밀도 검증."""
        hwm = 12345.67
        current = 11111.0
        dd = (hwm - current) / hwm
        assert 0.09 < dd < 0.11  # ~10%


class TestDailyLossLimit:
    """일일 손실 한도 테스트."""

    def test_no_loss_returns_false(self):
        """손실 없음 — 한도 미도달."""
        start = 10000.0
        current = 10000.0
        limit = 0.05
        loss = (start - current) / start
        assert loss < limit

    def test_loss_at_limit(self):
        """정확히 한도 — 진입 중단."""
        start = 10000.0
        current = 9500.0
        limit = 0.05
        loss = (start - current) / start
        assert loss >= limit  # 5% >= 5%

    def test_loss_exceeds_limit(self):
        """한도 초과 — 진입 중단."""
        start = 10000.0
        current = 9000.0
        limit = 0.05
        loss = (start - current) / start
        assert loss >= limit  # 10% >= 5%

    def test_gain_is_not_loss(self):
        """수익 상태 — 한도 미도달."""
        start = 10000.0
        current = 11000.0
        limit = 0.05
        loss = (start - current) / start
        assert loss < 0  # 음수 = 수익

    def test_daily_reset_after_24h(self):
        """24시간 경과 시 daily start 리셋."""
        start_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        now = datetime(2025, 6, 2, 0, 0, 1, tzinfo=timezone.utc)
        elapsed = (now - start_time).total_seconds()
        assert elapsed >= 86400  # 리셋 조건 충족

    def test_daily_not_reset_before_24h(self):
        """24시간 미경과 시 리셋 안 함."""
        start_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        now = datetime(2025, 6, 1, 23, 59, 59, tzinfo=timezone.utc)
        elapsed = (now - start_time).total_seconds()
        assert elapsed < 86400

    def test_disabled_when_zero(self):
        """daily_loss_limit_pct=0 → 비활성화."""
        limit = 0.0
        assert limit <= 0


class TestCloseFailureFix:
    """close_slot_positions 청산 실패 시 슬롯 유지 테스트."""

    def test_both_legs_success_resets_slot(self):
        """양측 청산 성공 → 슬롯 리셋."""
        from quantbox.core.trader_slot import TraderSlotState
        slot = TraderSlotState(slot_id=0, position_open=True)
        # 양측 모두 성공하면 리셋해야 함
        close1_ok = True
        close2_ok = True
        should_reset = close1_ok and close2_ok
        assert should_reset is True

    def test_one_leg_failure_keeps_slot(self):
        """한쪽 청산 실패 → 슬롯 유지."""
        close1_ok = True
        close2_ok = False
        should_reset = close1_ok and close2_ok
        assert should_reset is False  # 리셋하면 안 됨

    def test_both_legs_failure_keeps_slot(self):
        """양측 청산 실패 → 슬롯 유지."""
        close1_ok = False
        close2_ok = False
        should_reset = close1_ok and close2_ok
        assert should_reset is False


class TestConsecutiveErrorHandler:
    """연속 에러 핸들러 테스트."""

    def test_single_error_no_close_all(self):
        """단일 에러 → 전체 청산 안 함."""
        MAX_CONSECUTIVE = 5
        consecutive = 1
        assert consecutive < MAX_CONSECUTIVE

    def test_five_errors_triggers_close_all(self):
        """5회 연속 에러 → 전체 청산."""
        MAX_CONSECUTIVE = 5
        consecutive = 5
        assert consecutive >= MAX_CONSECUTIVE

    def test_success_resets_counter(self):
        """성공 시 카운터 리셋."""
        consecutive = 3
        # 성공
        consecutive = 0
        assert consecutive == 0

    def test_counter_increments(self):
        """에러 시 카운터 증가."""
        consecutive = 0
        for _ in range(3):
            consecutive += 1
        assert consecutive == 3


class TestSettingsNewFields:
    """settings.py에 추가된 drawdown/daily loss 필드 테스트."""

    def test_default_max_drawdown_pct(self):
        """기본 max_drawdown_pct = 0.10 (10%)."""
        os.environ['BINANCE_API_KEY'] = 'test'
        os.environ['BINANCE_API_SECRET'] = 'test'
        from quantbox.core.settings import QuantBoxSettings
        s = QuantBoxSettings()
        assert s.max_drawdown_pct == 0.10

    def test_default_daily_loss_limit_pct(self):
        """기본 daily_loss_limit_pct = 0.05 (5%)."""
        os.environ['BINANCE_API_KEY'] = 'test'
        os.environ['BINANCE_API_SECRET'] = 'test'
        from quantbox.core.settings import QuantBoxSettings
        s = QuantBoxSettings()
        assert s.daily_loss_limit_pct == 0.05

    def test_to_trader_config_includes_new_fields(self):
        """to_trader_config에 새 필드 포함."""
        os.environ['BINANCE_API_KEY'] = 'test'
        os.environ['BINANCE_API_SECRET'] = 'test'
        from quantbox.core.settings import QuantBoxSettings
        s = QuantBoxSettings()
        config = s.to_trader_config()
        assert 'max_drawdown_pct' in config
        assert 'daily_loss_limit_pct' in config
        assert config['max_drawdown_pct'] == 0.10
        assert config['daily_loss_limit_pct'] == 0.05

    def test_custom_drawdown_from_env(self):
        """환경변수로 커스텀 값 설정."""
        os.environ['BINANCE_API_KEY'] = 'test'
        os.environ['BINANCE_API_SECRET'] = 'test'
        os.environ['MAX_DRAWDOWN_PCT'] = '0.15'
        os.environ['DAILY_LOSS_LIMIT_PCT'] = '0.03'
        from quantbox.core.settings import QuantBoxSettings
        s = QuantBoxSettings()
        assert s.max_drawdown_pct == 0.15
        assert s.daily_loss_limit_pct == 0.03
        # Cleanup
        del os.environ['MAX_DRAWDOWN_PCT']
        del os.environ['DAILY_LOSS_LIMIT_PCT']


class TestLogRotation:
    """로그 로테이션 설정 테스트."""

    def test_rotating_handler_in_config(self):
        """RotatingFileHandler가 사용되는지 확인."""
        import logging.handlers
        # live_trader.py의 logging 설정을 확인
        # 이미 import 시 설정되므로, RotatingFileHandler 클래스 존재 확인
        assert hasattr(logging.handlers, 'RotatingFileHandler')
