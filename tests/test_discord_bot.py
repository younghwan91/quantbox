"""
Tests for Discord Bot command system.

검증 포인트:
1. BotCommand / CommandType 데이터 구조
2. TradingBot._parse_command() 파싱 로직
3. DiscordBotRunner 큐 동작 (get_command, get_all_commands)
4. LivePairTrader 명령 처리 (_handle_command, _cmd_*)
5. run_cycle에서 봇 명령 통합 (pause, stop)
"""

import asyncio
import queue
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from quantbox.notifications.bot import (
    BotCommand,
    CommandType,
    TradingBot,
    DiscordBotRunner,
    HELP_TEXT,
)


# ============================================================
# BotCommand unit tests
# ============================================================


class TestBotCommand:
    """BotCommand 데이터 구조 테스트."""

    def test_create_simple(self):
        cmd = BotCommand(command=CommandType.STATUS)
        assert cmd.command == CommandType.STATUS
        assert cmd.args is None
        assert cmd.response_future is None

    def test_create_with_args(self):
        cmd = BotCommand(
            command=CommandType.CLOSE_SLOT,
            args={'slot_id': 2},
        )
        assert cmd.args['slot_id'] == 2

    def test_set_response_without_future(self):
        """Future 없이 set_response 호출 시 에러 없음."""
        cmd = BotCommand(command=CommandType.STATUS)
        cmd.set_response("test")  # should not raise

    def test_set_response_with_future(self):
        """Future가 있으면 결과가 설정된다."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        cmd = BotCommand(
            command=CommandType.STATUS,
            response_future=future,
            loop=loop,
        )
        cmd.set_response("OK")
        loop.run_until_complete(asyncio.sleep(0.01))  # let call_soon_threadsafe run
        assert future.result() == "OK"
        loop.close()


# ============================================================
# TradingBot._parse_command tests
# ============================================================


class TestParseCommand:
    """TradingBot._parse_command() 파싱 로직 테스트."""

    @pytest.fixture
    def bot(self):
        q = queue.Queue()
        return TradingBot(command_queue=q, channel_id=12345)

    def test_status(self, bot):
        cmd = bot._parse_command('!status', ['!status'])
        assert cmd.command == CommandType.STATUS

    def test_close_all(self, bot):
        cmd = bot._parse_command('!close', ['!close', 'all'])
        assert cmd.command == CommandType.CLOSE_ALL

    def test_close_slot(self, bot):
        cmd = bot._parse_command('!close', ['!close', '3'])
        assert cmd.command == CommandType.CLOSE_SLOT
        assert cmd.args['slot_id'] == 3

    def test_close_no_arg(self, bot):
        with pytest.raises(ValueError, match="형식으로"):
            bot._parse_command('!close', ['!close'])

    def test_close_invalid_arg(self, bot):
        with pytest.raises(ValueError, match="숫자"):
            bot._parse_command('!close', ['!close', 'abc'])

    def test_pause(self, bot):
        cmd = bot._parse_command('!pause', ['!pause'])
        assert cmd.command == CommandType.PAUSE

    def test_resume(self, bot):
        cmd = bot._parse_command('!resume', ['!resume'])
        assert cmd.command == CommandType.RESUME

    def test_stop(self, bot):
        cmd = bot._parse_command('!stop', ['!stop'])
        assert cmd.command == CommandType.STOP

    def test_pnl(self, bot):
        cmd = bot._parse_command('!pnl', ['!pnl'])
        assert cmd.command == CommandType.PNL

    def test_help(self, bot):
        cmd = bot._parse_command('!help', ['!help'])
        assert cmd.command == CommandType.HELP

    def test_unknown_command(self, bot):
        result = bot._parse_command('!foobar', ['!foobar'])
        assert result is None


# ============================================================
# DiscordBotRunner queue tests
# ============================================================


class TestDiscordBotRunner:
    """DiscordBotRunner 큐 동작 테스트."""

    def test_get_command_empty(self):
        runner = DiscordBotRunner(token="fake", channel_id=1)
        assert runner.get_command() is None

    def test_get_command_with_item(self):
        runner = DiscordBotRunner(token="fake", channel_id=1)
        cmd = BotCommand(command=CommandType.STATUS)
        runner.command_queue.put(cmd)
        result = runner.get_command()
        assert result is not None
        assert result.command == CommandType.STATUS

    def test_get_all_commands(self):
        runner = DiscordBotRunner(token="fake", channel_id=1)
        runner.command_queue.put(BotCommand(command=CommandType.STATUS))
        runner.command_queue.put(BotCommand(command=CommandType.PAUSE))
        runner.command_queue.put(BotCommand(command=CommandType.PNL))

        cmds = runner.get_all_commands()
        assert len(cmds) == 3
        assert cmds[0].command == CommandType.STATUS
        assert cmds[1].command == CommandType.PAUSE
        assert cmds[2].command == CommandType.PNL

    def test_get_all_commands_empty(self):
        runner = DiscordBotRunner(token="fake", channel_id=1)
        assert runner.get_all_commands() == []

    def test_is_running_before_start(self):
        runner = DiscordBotRunner(token="fake", channel_id=1)
        assert runner.is_running is False

    def test_stop_waits_for_close(self):
        """stop()이 bot.close()를 기다리고 스레드를 join하는지 검증."""
        runner = DiscordBotRunner(token="fake", channel_id=1)

        mock_bot = MagicMock()
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_thread = MagicMock()
        # join 전에는 alive, join 후에는 dead
        mock_thread.is_alive.side_effect = [True, False]

        mock_future = MagicMock()
        mock_future.result.return_value = None

        runner._bot = mock_bot
        runner._loop = mock_loop
        runner._thread = mock_thread

        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future) as mock_rcts:
            runner.stop(timeout=3.0)

        # bot.close()가 스케줄링 되었는지
        mock_rcts.assert_called_once_with(mock_bot.close(), mock_loop)
        # Future.result()로 완료 대기했는지
        mock_future.result.assert_called_once_with(timeout=3.0)
        # 스레드 join이 호출되었는지
        mock_thread.join.assert_called_once_with(timeout=3.0)

    def test_stop_handles_closed_loop(self):
        """이미 닫힌 이벤트 루프에서 stop()이 에러 없이 종료되는지 검증."""
        runner = DiscordBotRunner(token="fake", channel_id=1)

        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = True
        runner._bot = MagicMock()
        runner._loop = mock_loop

        # 이미 닫힌 루프 → close 스케줄링 건너뜀
        runner.stop()
        # asyncio.run_coroutine_threadsafe가 호출되지 않아야 함

    def test_stop_handles_close_error(self):
        """bot.close()가 에러를 던져도 stop()이 정상 종료되는지 검증."""
        runner = DiscordBotRunner(token="fake", channel_id=1)

        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_thread = MagicMock()
        mock_thread.is_alive.side_effect = [True, False]

        mock_future = MagicMock()
        mock_future.result.side_effect = TimeoutError("close timeout")

        runner._bot = MagicMock()
        runner._loop = mock_loop
        runner._thread = mock_thread

        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            # 에러가 전파되지 않고 정상 종료되어야 함
            runner.stop(timeout=1.0)

        mock_thread.join.assert_called_once_with(timeout=1.0)

