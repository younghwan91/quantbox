"""
실패한 심볼의 블랙리스트 (사이클 간 지속) 테스트.

XAGUSDT -4411 에러 같은 경우, 한 번 실패한 심볼이 TTL 동안 재시도되지 않도록
보장한다. Mock을 사용하여 거래소 호출 없이 순수 로직만 검증한다.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from quantbox.notifications.bot import CommandType, BotCommand


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def trader():
    """LivePairTrader를 API 호출 없이 인스턴스화.

    binance.Client, DiscordNotifier, DiscordBotRunner 등을 모두 mock 처리한다.
    """
    with (
        patch("quantbox.live_trader.Client"),
        patch("quantbox.live_trader.DiscordNotifier"),
        patch("quantbox.live_trader.DiscordBotRunner"),
        patch("quantbox.live_trader.LimitOrderChaser"),
        patch("quantbox.live_trader.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.binance_api_key = "test"
        settings.binance_api_secret = "test"
        settings.binance_testnet = True
        settings.discord_webhook_url = ""
        settings.discord_bot_token = ""
        settings.discord_channel_id = 0
        settings.paper_balance = 1000.0
        settings.to_trader_config.return_value = {
            'leverage': 2,
            'max_traders': 5,
            'position_size_pct': -1,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'stop_loss_zscore': 1.5,
            'max_holding_bars': 64,
            'use_dynamic_exit': True,
            'dynamic_exit_base': 0.1,
            'dynamic_exit_range': 0.9,
            'min_correlation': 0.6,
            'min_half_life': 2.0,
            'max_half_life': 80.0,
            'max_cointegration_pvalue': 0.05,
            'max_hurst': 0.5,
            'hurst_threshold': 0.5,
            'trading_fee': 0.0005,
            'slippage': 0.001,
            'funding_rate_8h': 0.0001,
            'check_interval_seconds': 60,
            'n_coins': 40,
        }
        mock_settings.return_value = settings

        from quantbox.live_trader import LivePairTrader
        t = LivePairTrader(paper_mode=True)
        return t


# ── Tests ─────────────────────────────────────────────────


class TestBlacklistPersistence:
    """블랙리스트가 사이클 간 지속되는지 검증."""

    def test_initial_blacklist_empty(self, trader):
        """초기 상태에서 블랙리스트는 비어있어야 한다."""
        assert trader._failed_symbols == {}

    def test_add_failed_symbol(self, trader):
        """심볼 추가 시 블랙리스트에 등록된다."""
        now = datetime.now(timezone.utc)
        trader._failed_symbols['XAGUSDT'] = now
        trader._failed_symbols['XAUUSDT'] = now

        assert 'XAGUSDT' in trader._failed_symbols
        assert 'XAUUSDT' in trader._failed_symbols
        assert len(trader._failed_symbols) == 2

    def test_blacklist_survives_cycle_excluded_rebuild(self, trader):
        """occupied_symbols에서 excluded를 재생성해도 블랙리스트가 포함된다."""
        now = datetime.now(timezone.utc)
        trader._failed_symbols['XAGUSDT'] = now
        trader._failed_symbols['XAUUSDT'] = now

        # occupied_symbols는 비어있지만 블랙리스트가 excluded에 합쳐져야 함
        occupied = set()  # 아무 포지션도 없는 상태
        excluded = occupied | set(trader._failed_symbols.keys())

        assert 'XAGUSDT' in excluded
        assert 'XAUUSDT' in excluded


class TestBlacklistTTL:
    """TTL 기반 자동 만료 검증."""

    def test_fresh_symbols_not_expired(self, trader):
        """방금 추가된 심볼은 만료되지 않는다."""
        now = datetime.now(timezone.utc)
        trader._failed_symbols['XAGUSDT'] = now

        trader._clean_expired_failures()

        assert 'XAGUSDT' in trader._failed_symbols

    def test_old_symbols_expire(self, trader):
        """TTL이 지난 심볼은 자동으로 제거된다."""
        old = datetime.now(timezone.utc) - timedelta(seconds=trader.BLACKLIST_TTL_SECONDS + 1)
        trader._failed_symbols['XAGUSDT'] = old
        trader._failed_symbols['XAUUSDT'] = old

        trader._clean_expired_failures()

        assert trader._failed_symbols == {}

    def test_partial_expiry(self, trader):
        """일부만 만료되면 만료된 것만 제거한다."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(seconds=trader.BLACKLIST_TTL_SECONDS + 1)

        trader._failed_symbols['XAGUSDT'] = old      # 만료됨
        trader._failed_symbols['BTCUSDT'] = now       # 아직 유효

        trader._clean_expired_failures()

        assert 'XAGUSDT' not in trader._failed_symbols
        assert 'BTCUSDT' in trader._failed_symbols

    def test_clean_empty_blacklist_no_error(self, trader):
        """빈 블랙리스트에서 clean 호출해도 에러 없이 통과."""
        trader._clean_expired_failures()
        assert trader._failed_symbols == {}


class TestBlacklistDiscordCommands:
    """Discord !blacklist 명령어 테스트."""

    def test_cmd_blacklist_empty(self, trader):
        """블랙리스트가 비어있을 때 응답 확인."""
        result = trader._cmd_blacklist()
        assert "비어있음" in result

    def test_cmd_blacklist_with_entries(self, trader):
        """블랙리스트에 항목이 있을 때 심볼이 출력된다."""
        now = datetime.now(timezone.utc)
        trader._failed_symbols['XAGUSDT'] = now
        trader._failed_symbols['XAUUSDT'] = now

        result = trader._cmd_blacklist()
        assert 'XAGUSDT' in result
        assert 'XAUUSDT' in result
        assert '2개 심볼' in result

    def test_cmd_blacklist_clear(self, trader):
        """!blacklist clear로 전체 초기화."""
        now = datetime.now(timezone.utc)
        trader._failed_symbols['XAGUSDT'] = now
        trader._failed_symbols['XAUUSDT'] = now

        result = trader._cmd_blacklist_clear()
        assert trader._failed_symbols == {}
        assert '2개 심볼 제거' in result

    def test_cmd_blacklist_clear_empty(self, trader):
        """이미 비어있는 블랙리스트 초기화."""
        result = trader._cmd_blacklist_clear()
        assert '0개 심볼 제거' in result

    def test_handle_command_routes_blacklist(self, trader):
        """_handle_command가 BLACKLIST 명령을 올바르게 라우팅한다."""
        cmd = BotCommand(command=CommandType.BLACKLIST)
        result = trader._handle_command(cmd)
        assert "비어있음" in result

    def test_handle_command_routes_blacklist_clear(self, trader):
        """_handle_command가 BLACKLIST clear 명령을 올바르게 라우팅한다."""
        now = datetime.now(timezone.utc)
        trader._failed_symbols['TESTUSDT'] = now

        cmd = BotCommand(command=CommandType.BLACKLIST, args={'action': 'clear'})
        result = trader._handle_command(cmd)
        assert trader._failed_symbols == {}
        assert '제거' in result


class TestBotCommandParsing:
    """bot.py의 _parse_command에서 !blacklist 파싱 검증."""

    def test_parse_blacklist(self):
        """!blacklist 명령 파싱."""
        from quantbox.notifications.bot import TradingBot
        import queue

        bot = TradingBot(command_queue=queue.Queue(), channel_id=123)
        cmd = bot._parse_command('!blacklist', ['!blacklist'])

        assert cmd is not None
        assert cmd.command == CommandType.BLACKLIST
        assert cmd.args is None

    def test_parse_blacklist_clear(self):
        """!blacklist clear 명령 파싱."""
        from quantbox.notifications.bot import TradingBot
        import queue

        bot = TradingBot(command_queue=queue.Queue(), channel_id=123)
        cmd = bot._parse_command('!blacklist', ['!blacklist', 'clear'])

        assert cmd is not None
        assert cmd.command == CommandType.BLACKLIST
        assert cmd.args == {'action': 'clear'}
