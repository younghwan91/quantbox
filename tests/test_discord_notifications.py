"""
Tests for quantbox.notifications.discord.DiscordNotifier.

모든 send 호출은 requests.post를 mock하여 실제 웹훅 전송 없이 검증한다.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from quantbox.notifications.discord import (
    DiscordNotifier,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_BLUE,
    _now_str,
)


# ==================== Fixtures ====================


@pytest.fixture
def notifier():
    """웹훅 URL이 설정된 DiscordNotifier."""
    return DiscordNotifier(webhook_url="YOUR_DISCORD_WEBHOOK_URL")


@pytest.fixture
def disabled_notifier():
    """웹훅 URL이 없는 (비활성) DiscordNotifier."""
    with patch.dict("os.environ", {}, clear=True):
        return DiscordNotifier(webhook_url=None)


@pytest.fixture
def mock_post():
    """requests.post를 mock하여 204 반환."""
    with patch("quantbox.notifications.discord.requests.post") as m:
        m.return_value = MagicMock(status_code=204)
        yield m


# ==================== Basic Send ====================


class TestSendBasic:
    """send() 메서드 기본 동작."""

    def test_send_disabled_returns_false(self, disabled_notifier):
        """웹훅 비활성 시 False 반환, 네트워크 호출 없음."""
        result = disabled_notifier.send(content="test")
        assert result is False

    def test_send_content_only(self, notifier, mock_post):
        """텍스트 전송."""
        result = notifier.send(content="hello")
        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["content"] == "hello"
        assert "embeds" not in payload

    def test_send_embed_only(self, notifier, mock_post):
        """embed 전송."""
        embed = {"title": "Test", "color": COLOR_BLUE}
        result = notifier.send(embed=embed)
        assert result is True
        payload = mock_post.call_args[1]["json"]
        assert payload["embeds"] == [embed]
        assert "content" not in payload

    def test_send_with_timeout(self, notifier, mock_post):
        """timeout=5 설정."""
        notifier.send(content="t")
        assert mock_post.call_args[1]["timeout"] == 5

    def test_send_network_error_returns_false(self, notifier):
        """네트워크 에러 시 False 반환 (트레이딩 중단 방지)."""
        with patch("quantbox.notifications.discord.requests.post", side_effect=Exception("timeout")):
            result = notifier.send(content="test")
            assert result is False

    def test_send_non_204_returns_false(self, notifier):
        """서버가 204 외 상태코드 반환 시 False."""
        with patch("quantbox.notifications.discord.requests.post") as m:
            m.return_value = MagicMock(status_code=400)
            result = notifier.send(content="test")
            assert result is False


# ==================== Bot Lifecycle ====================


class TestBotLifecycle:
    """notify_start / notify_stop."""

    def test_notify_start_fields(self, notifier, mock_post):
        """시작 알림에 모드, 잔고, 레버리지, 슬롯 정보 포함."""
        notifier.notify_start("LIVE", 1000.0, leverage=2, max_traders=5)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN
        assert "트레이딩 시작" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "모드" in field_names
        assert "잔고" in field_names
        assert "레버리지" in field_names
        assert "최대 슬롯" in field_names
        # Verify values
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["모드"] == "LIVE"
        assert fields["잔고"] == "$1,000.00"
        assert fields["레버리지"] == "2x"

    def test_notify_stop_profit(self, notifier, mock_post):
        """종료 시 수익이면 녹색."""
        notifier.notify_stop("사용자 종료", total_pnl=50.0, n_trades=3)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["총 PnL"] == "$+50.00"

    def test_notify_stop_loss(self, notifier, mock_post):
        """종료 시 손실이면 빨간색."""
        notifier.notify_stop("에러", total_pnl=-25.0, n_trades=5)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED


# ==================== Position Events ====================


class TestPositionOpen:
    """notify_open."""

    def test_open_long_green(self, notifier, mock_post):
        """LONG 진입은 녹색."""
        notifier.notify_open(
            "BTCUSDT",
            "ETHUSDT",
            "[S0] LONG Spread",
            zscore=-2.5,
            capital=500.0,
            slot_id=0,
            leverage=2,
            margin=250.0,
            available_after=750.0,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["페어"] == "BTCUSDT/ETHUSDT"
        assert fields["슬롯"] == "S0"
        assert fields["노셔널"] == "$500.00"
        assert fields["마진"] == "$250.00"
        assert fields["레버리지"] == "2x"
        assert fields["잔여 가용"] == "$750.00"

    def test_open_short_red(self, notifier, mock_post):
        """SHORT 진입은 빨간색."""
        notifier.notify_open(
            "BTCUSDT",
            "ETHUSDT",
            "[S1] SHORT Spread",
            zscore=2.5,
            capital=500.0,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED

    def test_open_no_leverage_field_when_1x(self, notifier, mock_post):
        """레버리지 1x면 필드 생략."""
        notifier.notify_open(
            "BTCUSDT",
            "ETHUSDT",
            "LONG",
            zscore=-2.0,
            capital=100.0,
            leverage=1,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        field_names = [f["name"] for f in embed["fields"]]
        assert "레버리지" not in field_names


class TestPositionClose:
    """notify_close."""

    def test_close_profit(self, notifier, mock_post):
        """수익 청산: 녹색, ✅ 아이콘."""
        notifier.notify_close(
            "BTCUSDT",
            "ETHUSDT",
            pnl=12.5,
            pnl_pct=2.5,
            reason="mean_reversion",
            slot_id=1,
            holding_time="6.3h",
            cumulative_pnl=45.0,
            balance_after=1045.0,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN
        assert "✅" in embed["title"]
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["슬롯"] == "S1"
        assert "$+12.50" in fields["PnL"]
        assert "2.50%" in fields["PnL"]
        assert fields["보유 시간"] == "6.3h"
        assert fields["누적 PnL"] == "$+45.00"
        assert fields["잔고"] == "$1,045.00"

    def test_close_loss(self, notifier, mock_post):
        """손실 청산: 빨간색, ❌ 아이콘."""
        notifier.notify_close(
            "BTCUSDT",
            "ETHUSDT",
            pnl=-5.0,
            pnl_pct=-1.0,
            reason="stop_loss",
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED
        assert "❌" in embed["title"]

    def test_close_optional_fields_omitted(self, notifier, mock_post):
        """선택적 필드 미제공 시 포함되지 않음."""
        notifier.notify_close(
            "BTCUSDT",
            "ETHUSDT",
            pnl=0.0,
            pnl_pct=0.0,
            reason="timeout",
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        field_names = [f["name"] for f in embed["fields"]]
        assert "보유 시간" not in field_names
        assert "잔고" not in field_names


# ==================== Cycle Summary ====================


class TestCycleSummary:
    """notify_cycle_summary."""

    def test_cycle_summary_with_active_and_idle(self, notifier, mock_post):
        """활성+유휴 슬롯 혼합 상태."""
        slots_info = [
            {
                "slot_id": 0,
                "status": "active",
                "pair": "BTCUSDT/ETHUSDT",
                "unrealized_pnl": 5.0,
                "holding_hours": 2.3,
                "entry_z": -2.1,
                "current_z": -0.8,
            },
            {"slot_id": 1, "status": "idle"},
            {
                "slot_id": 2,
                "status": "active",
                "pair": "SOLUSDT/AVAXUSDT",
                "unrealized_pnl": -3.0,
                "holding_hours": 1.1,
                "entry_z": 2.5,
                "current_z": 1.2,
            },
        ]
        notifier.notify_cycle_summary(
            slots_info=slots_info,
            available_capital=500.0,
            total_capital=1000.0,
            total_unrealized_pnl=2.0,
            session_pnl=10.0,
            n_trades_session=3,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        # Net positive → green
        assert embed["color"] == COLOR_GREEN
        # Check slot status text is present
        slot_field = next(f for f in embed["fields"] if f["name"] == "📊 슬롯 상태")
        assert "S0" in slot_field["value"]
        assert "S1" in slot_field["value"]
        assert "대기" in slot_field["value"]
        # Capital fields
        field_names = [f["name"] for f in embed["fields"]]
        assert "가용 자본" in field_names
        assert "미실현 PnL" in field_names

    def test_cycle_summary_all_idle(self, notifier, mock_post):
        """모든 슬롯 유휴 상태."""
        slots_info = [
            {"slot_id": 0, "status": "idle"},
            {"slot_id": 1, "status": "idle"},
        ]
        notifier.notify_cycle_summary(
            slots_info=slots_info,
            available_capital=1000.0,
            total_capital=1000.0,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN  # net=0 → green

    def test_cycle_summary_negative_net(self, notifier, mock_post):
        """미실현+세션 음수면 빨간색."""
        notifier.notify_cycle_summary(
            slots_info=[],
            available_capital=900.0,
            total_capital=1000.0,
            total_unrealized_pnl=-50.0,
            session_pnl=-30.0,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED


# ==================== Daily Report ====================


class TestDailyReport:
    """notify_daily_report / should_send_daily_report."""

    def test_daily_report_all_fields(self, notifier, mock_post):
        """일일 리포트에 모든 필드 포함."""
        notifier.notify_daily_report(
            total_balance=1050.0,
            daily_pnl=50.0,
            daily_return_pct=5.0,
            cumulative_pnl=50.0,
            cumulative_return_pct=5.0,
            n_trades_today=4,
            n_trades_total=10,
            win_rate=60.0,
            max_drawdown_pct=3.5,
            active_slots=2,
            max_slots=5,
            leverage=2,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_GREEN
        assert "📈" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "총 잔고" in field_names
        assert "당일 PnL" in field_names
        assert "승률" in field_names
        assert "최대 DD" in field_names
        assert "레버리지" in field_names

    def test_daily_report_loss_red(self, notifier, mock_post):
        """일일 손실이면 빨간색."""
        notifier.notify_daily_report(
            total_balance=950.0,
            daily_pnl=-50.0,
            daily_return_pct=-5.0,
            cumulative_pnl=-50.0,
            cumulative_return_pct=-5.0,
            n_trades_today=2,
            n_trades_total=2,
            win_rate=0.0,
            max_drawdown_pct=5.0,
            active_slots=0,
            max_slots=5,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED
        assert "📉" in embed["title"]

    def test_should_send_first_time(self, notifier):
        """첫 호출 시 True."""
        assert notifier.should_send_daily_report() is True

    def test_should_not_send_within_24h(self, notifier):
        """24시간 이내엔 False."""
        notifier._last_daily_report = datetime.now(timezone.utc)
        assert notifier.should_send_daily_report() is False

    def test_should_send_after_24h(self, notifier):
        """24시간 경과 후 True."""
        notifier._last_daily_report = datetime.now(timezone.utc) - timedelta(hours=25)
        assert notifier.should_send_daily_report() is True

    def test_daily_report_updates_timestamp(self, notifier, mock_post):
        """notify_daily_report가 _last_daily_report를 업데이트."""
        assert notifier._last_daily_report is None
        notifier.notify_daily_report(
            total_balance=1000.0,
            daily_pnl=0.0,
            daily_return_pct=0.0,
            cumulative_pnl=0.0,
            cumulative_return_pct=0.0,
            n_trades_today=0,
            n_trades_total=0,
            win_rate=0.0,
            max_drawdown_pct=0.0,
            active_slots=0,
            max_slots=5,
        )
        assert notifier._last_daily_report is not None
        assert notifier.should_send_daily_report() is False


# ==================== Warnings ====================


class TestWarnings:
    """Warning / error notifications."""

    def test_notify_error(self, notifier, mock_post):
        """에러 알림에 빨간색."""
        notifier.notify_error("Connection timeout")
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED
        assert "Connection timeout" in embed["description"]

    def test_notify_error_truncation(self, notifier, mock_post):
        """에러 메시지 500자 제한."""
        long_msg = "A" * 1000
        notifier.notify_error(long_msg)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert len(embed["description"]) == 500

    def test_notify_warning(self, notifier, mock_post):
        """경고 알림에 노란색."""
        notifier.notify_warning("API Rate Limit", "Approaching rate limit")
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_YELLOW
        assert "API Rate Limit" in embed["title"]

    def test_notify_insufficient_balance(self, notifier, mock_post):
        """잔고 부족 경고."""
        notifier.notify_insufficient_balance(available=50.0, required=100.0)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_YELLOW
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["가용 잔고"] == "$50.00"
        assert fields["최소 필요"] == "$100.00"

    def test_notify_all_slots_occupied(self, notifier, mock_post):
        """모든 슬롯 사용 중 알림."""
        slots = [
            {"slot_id": 0, "pair": "BTCUSDT/ETHUSDT", "holding_hours": 3.5},
            {"slot_id": 1, "pair": "SOLUSDT/AVAXUSDT", "holding_hours": 1.2},
        ]
        notifier.notify_all_slots_occupied(slots)
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_YELLOW
        assert "S0" in embed["description"]
        assert "BTCUSDT/ETHUSDT" in embed["description"]

    def test_notify_drawdown_alert(self, notifier, mock_post):
        """낙폭 경고."""
        notifier.notify_drawdown_alert(
            current_dd_pct=8.5,
            threshold_pct=5.0,
            balance=920.0,
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == COLOR_RED
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["현재 낙폭"] == "8.50%"
        assert fields["임계치"] == "5.00%"
        assert fields["잔고"] == "$920.00"


# ==================== Utility ====================


class TestUtility:
    """Helper functions."""

    def test_now_str_format(self):
        """_now_str이 UTC 시간 문자열 반환."""
        result = _now_str()
        assert "UTC" in result
        # Parse should work
        datetime.strptime(result, "%Y-%m-%d %H:%M UTC")

    def test_notifier_enabled_flag(self):
        """웹훅 URL 유무에 따른 enabled 플래그."""
        n = DiscordNotifier(webhook_url="https://test.com")
        assert n.enabled is True

        with patch.dict("os.environ", {}, clear=True):
            n2 = DiscordNotifier(webhook_url=None)
            assert n2.enabled is False

    def test_notifier_env_fallback(self):
        """환경변수에서 URL 읽기."""
        with patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://env.com"}):
            n = DiscordNotifier()
            assert n.webhook_url == "https://env.com"
            assert n.enabled is True
