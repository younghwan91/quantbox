"""
Tests for Discord Bot command system.

검증 포인트:
1. BotCommand / CommandType 데이터 구조
2. TradingBot._parse_command() 파싱 로직
3. DiscordBotRunner 큐 동작 (get_command, get_all_commands)
4. RubberBandScalper 명령 처리 (_handle_command, _cmd_*)
5. _cycle에서 봇 명령 통합 (pause, stop)
"""

import asyncio
import queue
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta, timezone

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


# ============================================================
# RubberBandScalper command handlers
# ============================================================


class TestScalperCommandHandlers:
    """RubberBandScalper의 _cmd_* 메서드 테스트."""

    @pytest.fixture
    def scalper(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        with patch("quantbox.scalper.Client"):
            from quantbox.scalper import RubberBandScalper
            s = RubberBandScalper(paper_mode=True)
            s.discord = MagicMock()
            s._initial_balance = 1000.0
            return s

    def test_cmd_status_no_positions(self, scalper):
        result = scalper._cmd_status()
        assert "QuantBox Status" in result
        assert "No open positions" in result
        assert "0/" in result

    def test_cmd_status_with_position(self, scalper):
        scalper.strategy.open_position(
            symbol='BTCUSDT', signal=1, entry_price=50000.0,
            notional=1000.0, leverage=2, move_size=0.04,
        )
        result = scalper._cmd_status()
        assert "BTCUSDT" in result
        assert "L" in result
        assert "1/" in result

    def test_cmd_status_paused(self, scalper):
        scalper._paused = True
        result = scalper._cmd_status()
        assert "PAUSED" in result

    def test_cmd_pnl_no_trades(self, scalper):
        result = scalper._cmd_pnl()
        assert "No trades" in result

    def test_cmd_pnl_with_trades(self, scalper):
        scalper._trade_history = [
            {'symbol': 'BTCUSDT', 'pnl_usd': 5.0, 'pnl_pct': 0.5,
             'signal': 1, 'entry_price': 50000, 'exit_price': 50250,
             'notional': 1000, 'bars_held': 3, 'reason': 'target'},
            {'symbol': 'ETHUSDT', 'pnl_usd': -2.0, 'pnl_pct': -0.2,
             'signal': -1, 'entry_price': 3000, 'exit_price': 3006,
             'notional': 1000, 'bars_held': 5, 'reason': 'stop'},
        ]
        scalper.get_balance = MagicMock(return_value=1003.0)
        result = scalper._cmd_pnl()
        assert "$+3.00" in result
        assert "Win Rate" in result
        assert "50.0%" in result
        assert "BTCUSDT" in result

    def test_cmd_pause(self, scalper):
        assert scalper._paused is False
        result = scalper._cmd_pause()
        assert scalper._paused is True
        assert "paused" in result.lower()

    def test_cmd_pause_already(self, scalper):
        scalper._paused = True
        result = scalper._cmd_pause()
        assert "Already" in result

    def test_cmd_resume(self, scalper):
        scalper._paused = True
        result = scalper._cmd_resume()
        assert scalper._paused is False
        assert "resumed" in result.lower()

    def test_cmd_resume_already(self, scalper):
        result = scalper._cmd_resume()
        assert "Already" in result

    def test_cmd_stop(self, scalper):
        scalper.download_klines = MagicMock(return_value=None)
        result = scalper._cmd_stop()
        assert scalper._stop_requested is True
        assert "Stopping" in result

    def test_cmd_close_all_no_positions(self, scalper):
        result = scalper._cmd_close_all()
        assert "No positions" in result

    def test_cmd_close_slot_invalid(self, scalper):
        result = scalper._cmd_close_slot(99)
        assert "No positions" in result

    def test_cmd_blacklist_view(self, scalper):
        scalper.config.exclude_symbols.append('XYZUSDT')
        result = scalper._cmd_blacklist()
        assert "XYZUSDT" in result
        assert "Blacklist" in result

    def test_cmd_blacklist_clear(self, scalper):
        scalper.config.exclude_symbols.append('XYZUSDT')
        original_len = len(scalper.config.exclude_symbols)
        result = scalper._cmd_blacklist({'action': 'clear'})
        assert "cleared" in result.lower()
        assert len(scalper.config.exclude_symbols) == 0

    def test_handle_command_dispatches(self, scalper):
        """_handle_command가 각 CommandType을 올바른 핸들러로 보낸다."""
        result = scalper._handle_command(
            BotCommand(command=CommandType.STATUS)
        )
        assert "QuantBox Status" in result

        result = scalper._handle_command(
            BotCommand(command=CommandType.PAUSE)
        )
        assert scalper._paused is True

    def test_process_bot_commands_no_runner(self, scalper):
        """_bot_runner=None이면 에러 없이 건너뛴다."""
        scalper._bot_runner = None
        scalper._process_bot_commands()  # should not raise

    def test_process_bot_commands_with_queue(self, scalper):
        """큐에 쌓인 명령이 순서대로 처리된다."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        scalper._bot_runner = runner

        runner.command_queue.put(BotCommand(command=CommandType.PAUSE))
        runner.command_queue.put(BotCommand(command=CommandType.STATUS))

        scalper._process_bot_commands()
        assert scalper._paused is True


# ============================================================
# Polling loop (_wait_for_bar / _next_bar_time) tests
# ============================================================


class TestPollingLoop:
    """_wait_for_bar의 1분 폴링 루프 테스트."""

    @pytest.fixture
    def scalper(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        with patch("quantbox.scalper.Client"):
            from quantbox.scalper import RubberBandScalper
            s = RubberBandScalper(paper_mode=True)
            s.discord = MagicMock()
            s._bot_runner = None
            return s

    def test_next_bar_time_aligned(self):
        """14:03 → 다음 바 14:15:05."""
        from quantbox.scalper import RubberBandScalper
        fake_now = datetime(2026, 2, 17, 14, 3, 0, tzinfo=timezone.utc)
        with patch("quantbox.scalper.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # We need a scalper instance; create minimal mock
            s = MagicMock(spec=RubberBandScalper)
            s._next_bar_time = RubberBandScalper._next_bar_time.__get__(s)
            target = s._next_bar_time()
        assert target.minute == 15
        assert target.second == 5

    def test_next_bar_time_near_hour(self):
        """14:47 → 다음 바 15:00:05."""
        from quantbox.scalper import RubberBandScalper
        fake_now = datetime(2026, 2, 17, 14, 47, 0, tzinfo=timezone.utc)
        with patch("quantbox.scalper.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            s = MagicMock(spec=RubberBandScalper)
            s._next_bar_time = RubberBandScalper._next_bar_time.__get__(s)
            target = s._next_bar_time()
        assert target.hour == 15
        assert target.minute == 0
        assert target.second == 5

    def test_wait_for_bar_polls_bot_commands(self, scalper):
        """_wait_for_bar가 대기 중 봇 명령을 처리하는지 확인."""
        # Set up bot runner with a PAUSE command queued
        runner = DiscordBotRunner(token="fake", channel_id=1)
        scalper._bot_runner = runner
        runner.command_queue.put(BotCommand(command=CommandType.PAUSE))

        # Target time is 2 seconds from "now" — should poll once then exit
        fake_times = [
            datetime(2026, 2, 17, 14, 14, 58, tzinfo=timezone.utc),  # _next_bar_time
            datetime(2026, 2, 17, 14, 14, 58, tzinfo=timezone.utc),  # first loop check
            datetime(2026, 2, 17, 14, 14, 58, tzinfo=timezone.utc),  # remaining calc
            datetime(2026, 2, 17, 14, 15, 6, tzinfo=timezone.utc),   # after sleep: past target
        ]
        call_idx = [0]

        def fake_now(*args, **kwargs):
            idx = min(call_idx[0], len(fake_times) - 1)
            call_idx[0] += 1
            return fake_times[idx]

        with patch("quantbox.scalper.datetime") as mock_dt, \
             patch("quantbox.scalper.time") as mock_time:
            mock_dt.now = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic

            scalper._wait_for_bar()

        assert scalper._paused is True  # bot command was processed
        mock_time.sleep.assert_called()  # slept at least once

    def test_wait_for_bar_stops_on_stop_request(self, scalper):
        """_stop_requested=True면 즉시 대기 종료."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        scalper._bot_runner = runner
        # Queue a stop command
        runner.command_queue.put(BotCommand(command=CommandType.STOP))
        scalper.download_klines = MagicMock(return_value=None)

        fake_now_val = datetime(2026, 2, 17, 14, 1, 0, tzinfo=timezone.utc)

        with patch("quantbox.scalper.datetime") as mock_dt, \
             patch("quantbox.scalper.time") as mock_time:
            mock_dt.now.return_value = fake_now_val
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_time.sleep = MagicMock()
            mock_time.monotonic = time.monotonic

            scalper._wait_for_bar()

        assert scalper._stop_requested is True
        # Should NOT have slept the full 14 minutes
        if mock_time.sleep.called:
            assert mock_time.sleep.call_args[0][0] <= 60.0

    def test_wait_for_bar_syncs_server_orders(self, scalper):
        """대기 중 _sync_server_orders가 호출되는지 확인."""
        scalper._sync_server_orders = MagicMock()

        fake_times = [
            datetime(2026, 2, 17, 14, 14, 50, tzinfo=timezone.utc),  # _next_bar_time
            datetime(2026, 2, 17, 14, 14, 50, tzinfo=timezone.utc),  # loop check
            datetime(2026, 2, 17, 14, 14, 50, tzinfo=timezone.utc),  # remaining
            datetime(2026, 2, 17, 14, 15, 6, tzinfo=timezone.utc),   # past target
        ]
        call_idx = [0]

        def fake_now(*args, **kwargs):
            idx = min(call_idx[0], len(fake_times) - 1)
            call_idx[0] += 1
            return fake_times[idx]

        # Simulate monotonic clock jumping past SYNC_INTERVAL (60s)
        mono_values = [0.0, 61.0, 61.0]
        mono_idx = [0]

        def fake_monotonic():
            idx = min(mono_idx[0], len(mono_values) - 1)
            mono_idx[0] += 1
            return mono_values[idx]

        with patch("quantbox.scalper.datetime") as mock_dt, \
             patch("quantbox.scalper.time") as mock_time:
            mock_dt.now = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_time.sleep = MagicMock()
            mock_time.monotonic = fake_monotonic

            scalper._wait_for_bar()

        scalper._sync_server_orders.assert_called()
