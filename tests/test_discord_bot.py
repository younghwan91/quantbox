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
from quantbox.core.trader_slot import (
    TraderSlotState,
    get_idle_slots,
    get_active_slots,
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
# LivePairTrader command handlers
# ============================================================


class TestLiveTraderCommandHandlers:
    """LivePairTrader의 _cmd_* 메서드 테스트."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        monkeypatch.setenv("MAX_TRADERS", "3")

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            trader = LivePairTrader(paper_mode=True)
            trader.get_available_balance = MagicMock(return_value=5000.0)
            trader.get_total_balance = MagicMock(return_value=5000.0)
            trader.discord = MagicMock()
            return trader

    def test_cmd_status_all_idle(self, trader):
        result = trader._cmd_status()
        assert "QuantBox Status" in result
        assert "idle" in result
        assert "0 active" in result

    def test_cmd_status_with_active(self, trader):
        trader.slots[0].position_open = True
        trader.slots[0].current_pair = ("BTCUSDT", "ETHUSDT")
        trader.slot_positions[0] = {
            'entry_zscore': 2.5,
            'capital': 200,
        }

        result = trader._cmd_status()
        assert "BTCUSDT/ETHUSDT" in result
        assert "1 active" in result

    def test_cmd_status_paused(self, trader):
        trader.paused = True
        result = trader._cmd_status()
        assert "PAUSED" in result

    def test_cmd_close_all_no_positions(self, trader):
        result = trader._cmd_close_all()
        assert "없음" in result

    def test_cmd_close_all_with_positions(self, trader):
        trader.slots[0].position_open = True
        trader.slots[0].current_pair = ("A", "B")
        trader.slot_positions[0] = {'symbol1': 'A', 'symbol2': 'B', 'qty1': 0.1, 'qty2': -0.1}
        trader.close_all_positions = MagicMock()

        result = trader._cmd_close_all()
        assert "1개 슬롯" in result
        assert "청산 완료" in result
        trader.close_all_positions.assert_called_once()

    def test_cmd_close_slot_valid(self, trader):
        trader.slots[1].position_open = True
        trader.slots[1].current_pair = ("SOLUSDT", "AVAXUSDT")
        trader.slot_positions[1] = {'symbol1': 'SOLUSDT', 'symbol2': 'AVAXUSDT', 'qty1': 0.1, 'qty2': -0.1}
        trader.close_slot_positions = MagicMock()

        result = trader._cmd_close_slot(1)
        assert "슬롯 1 청산 완료" in result
        trader.close_slot_positions.assert_called_once_with(1)

    def test_cmd_close_slot_invalid(self, trader):
        result = trader._cmd_close_slot(99)
        assert "유효하지 않은" in result

    def test_cmd_close_slot_idle(self, trader):
        result = trader._cmd_close_slot(0)
        assert "비어있음" in result

    def test_cmd_pause(self, trader):
        assert trader.paused is False
        result = trader._cmd_pause()
        assert trader.paused is True
        assert "중지" in result

    def test_cmd_pause_already_paused(self, trader):
        trader.paused = True
        result = trader._cmd_pause()
        assert "이미" in result

    def test_cmd_resume(self, trader):
        trader.paused = True
        result = trader._cmd_resume()
        assert trader.paused is False
        assert "재개" in result

    def test_cmd_resume_already_active(self, trader):
        result = trader._cmd_resume()
        assert "이미" in result

    def test_cmd_stop(self, trader):
        assert trader._stop_requested is False
        result = trader._cmd_stop()
        assert trader._stop_requested is True
        assert "종료" in result

    def test_cmd_pnl_no_trades(self, trader):
        result = trader._cmd_pnl()
        assert "없음" in result

    def test_cmd_pnl_with_trades(self, trader):
        trader.trade_history = [
            {'pnl': 5.0, 'pair': ('A', 'B')},
            {'pnl': -2.0, 'pair': ('C', 'D')},
            {'pnl': 3.0, 'pair': ('E', 'F')},
        ]
        result = trader._cmd_pnl()
        assert "$+6.00" in result
        assert "67%" in result  # 2/3 win rate
        assert "3" in result


# ============================================================
# run_cycle integration with bot commands
# ============================================================


class TestRunCycleBotIntegration:
    """run_cycle에서 봇 명령이 올바르게 처리되는지 검증."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        monkeypatch.setenv("MAX_TRADERS", "3")
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DISCORD_CHANNEL_ID", raising=False)

    @pytest.fixture
    def trader(self, mock_env):
        with patch("quantbox.live_trader.Client"):
            from quantbox.live_trader import LivePairTrader
            with patch("quantbox.live_trader.get_settings") as mock_settings:
                s = mock_settings.return_value
                s.binance_api_key = "test_key"
                s.binance_api_secret = "test_secret"
                s.binance_testnet = True
                s.max_traders = 3
                s.leverage = 2
                s.position_size_pct = -1
                s.entry_zscore = 2.0
                s.exit_zscore = 0.5
                s.stop_loss_zscore = 3.5
                s.trading_fee = 0.0005
                s.slippage = 0.001
                s.funding_rate_8h = 0.0001
                s.n_coins = 30
                s.paper_balance = 1000.0
                s.discord_webhook_url = None
                s.discord_bot_token = None
                s.discord_channel_id = None
                s.to_trader_config.return_value = {
                    'quote_currency': 'USDT',
                    'n_coins': 30,
                    'interval': '15m',
                    'formation_hours': 168,
                    'entry_zscore': 2.0,
                    'exit_zscore': 0.5,
                    'stop_loss_zscore': 3.5,
                    'max_holding_hours': 24,
                    'position_size_pct': 0.333,
                    'trading_fee': 0.0005,
                    'slippage': 0.001,
                    'funding_rate_8h': 0.0001,
                    'leverage': 2,
                    'max_traders': 3,
                    'min_correlation': 0.5,
                    'max_cointegration_pvalue': 0.05,
                    'min_half_life': 3,
                    'max_half_life': 60,
                    'max_hurst': 0.5,
                    'check_interval_seconds': 900,
                    'retry_attempts': 3,
                    'retry_delay': 2,
                    'lookback_bars': 200,
                }
                trader = LivePairTrader(paper_mode=True)
            trader.get_available_balance = MagicMock(return_value=5000.0)
            trader.get_total_balance = MagicMock(return_value=5000.0)
            trader.get_trading_coins = MagicMock(return_value=["BTCUSDT", "ETHUSDT"])
            trader.download_multi_symbol_data = MagicMock(return_value={})
            trader.discord = MagicMock()
            return trader

    def test_stop_command_returns_false(self, trader):
        """!stop 명령 시 run_cycle()이 False를 반환한다."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        trader._bot_runner = runner

        # 큐에 stop 명령 넣기
        cmd = BotCommand(command=CommandType.STOP)
        runner.command_queue.put(cmd)

        result = trader.run_cycle()
        assert result is False
        assert trader._stop_requested is True

    def test_pause_skips_new_entries(self, trader):
        """!pause 상태에서 신규 진입을 건너뛴다."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        trader._bot_runner = runner

        # pause 명령
        runner.command_queue.put(BotCommand(command=CommandType.PAUSE))

        # Mock data 준비 (충분한 데이터)
        import numpy as np
        import pandas as pd

        def make_df():
            close = 100 + np.arange(200) * 0.01
            return pd.DataFrame({'Open': close, 'High': close, 'Low': close, 'Close': close, 'Volume': np.ones(200)})

        trader.download_multi_symbol_data = MagicMock(return_value={
            'BTCUSDT': make_df(), 'ETHUSDT': make_df(),
            'SOLUSDT': make_df(), 'BNBUSDT': make_df(),
        })
        trader.pair_selector.select_pairs = MagicMock()

        result = trader.run_cycle()
        assert result is True
        assert trader.paused is True
        # select_pairs가 호출되지 않아야 함 (pause로 건너뜀)
        trader.pair_selector.select_pairs.assert_not_called()

    def test_no_bot_runner_works_normally(self, trader):
        """_bot_runner=None이면 봇 명령 처리를 건너뛴다."""
        assert trader._bot_runner is None

        # download가 충분한 데이터를 줘야 cycle이 '정상' 진행
        import numpy as np
        import pandas as pd

        def make_df():
            close = 100 + np.arange(200) * 0.01
            return pd.DataFrame({'Open': close, 'High': close, 'Low': close, 'Close': close, 'Volume': np.ones(200)})

        trader.download_multi_symbol_data = MagicMock(return_value={
            'BTCUSDT': make_df(), 'ETHUSDT': make_df(),
            'SOLUSDT': make_df(), 'BNBUSDT': make_df(),
        })
        trader.pair_selector.select_pairs = MagicMock(return_value=[])

        result = trader.run_cycle()
        assert result is True

    def test_handle_command_dispatches_correctly(self, trader):
        """_handle_command가 각 CommandType을 올바른 핸들러로 보낸다."""
        trader._cmd_status = MagicMock(return_value="status_ok")
        trader._cmd_close_all = MagicMock(return_value="close_all_ok")
        trader._cmd_pause = MagicMock(return_value="pause_ok")

        assert trader._handle_command(BotCommand(command=CommandType.STATUS)) == "status_ok"
        assert trader._handle_command(BotCommand(command=CommandType.CLOSE_ALL)) == "close_all_ok"
        assert trader._handle_command(BotCommand(command=CommandType.PAUSE)) == "pause_ok"

    def test_multiple_commands_processed_in_order(self, trader):
        """큐에 쌓인 여러 명령이 순서대로 처리된다."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        trader._bot_runner = runner

        # pause → status 순서로 넣기
        runner.command_queue.put(BotCommand(command=CommandType.PAUSE))
        runner.command_queue.put(BotCommand(command=CommandType.STATUS))

        # _process_bot_commands만 호출
        trader._process_bot_commands()

        assert trader.paused is True  # pause가 먼저 처리됨

    def test_interruptible_sleep_processes_commands(self, trader):
        """_interruptible_sleep 중에도 봇 명령이 처리된다."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        trader._bot_runner = runner

        import threading
        import time

        # 1초 후에 STATUS 명령 큐에 넣기
        def delayed_cmd():
            time.sleep(1)
            runner.command_queue.put(BotCommand(command=CommandType.PAUSE))

        t = threading.Thread(target=delayed_cmd)
        t.start()

        # 4초짜리 interruptible sleep (2초마다 poll)
        trader._interruptible_sleep(4)
        t.join()

        # sleep 중에 PAUSE 명령이 처리되었어야 함
        assert trader.paused is True

    def test_interruptible_sleep_stops_on_stop_request(self, trader):
        """_interruptible_sleep 중 !stop이 오면 즉시 빠져나온다."""
        runner = DiscordBotRunner(token="fake", channel_id=1)
        trader._bot_runner = runner

        import threading
        import time

        # 0.5초 후에 STOP 명령
        def delayed_stop():
            time.sleep(0.5)
            runner.command_queue.put(BotCommand(command=CommandType.STOP))

        t = threading.Thread(target=delayed_stop)
        t.start()

        start = time.time()
        trader._interruptible_sleep(30)  # 30초 요청했지만 빨리 끝나야 함
        elapsed = time.time() - start
        t.join()

        assert trader._stop_requested is True
        assert elapsed < 10  # 30초 다 안 기다림
