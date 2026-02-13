"""
Discord Bot for QuantBox — 양방향 원격 명령 시스템.

Discord 채팅창에서 실시간으로 트레이더를 제어한다.
기존 DiscordNotifier(Webhook, 단방향)와 병행하여 동작한다.

지원 명령:
    !status        슬롯 상태 + 잔고 + 미실현 손익
    !close all     전체 포지션 청산
    !close <N>     슬롯 N번 청산
    !pause         신규 진입 중지 (기존 포지션 모니터링만)
    !resume        신규 진입 재개
    !stop          전체 청산 후 트레이더 종료
    !pnl           세션 손익 요약
    !help          명령어 목록

환경변수:
    DISCORD_BOT_TOKEN: Discord Bot 토큰 (필수)
    DISCORD_CHANNEL_ID: 명령을 수신할 채널 ID (필수)

아키텍처:
    - discord.py Client가 별도 스레드에서 asyncio 이벤트 루프를 돌림
    - queue.Queue로 메인 트레이딩 루프에 명령 전달
    - 메인 루프가 매 사이클 시작 시 큐를 확인하고 명령 실행
    - 결과를 asyncio 스레드에서 Discord 채널로 응답
"""

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import discord

logger = logging.getLogger('DiscordBot')


class CommandType(Enum):
    """봇 명령 타입."""
    STATUS = auto()
    CLOSE_ALL = auto()
    CLOSE_SLOT = auto()
    PAUSE = auto()
    RESUME = auto()
    STOP = auto()
    PNL = auto()
    BLACKLIST = auto()
    HELP = auto()


@dataclass
class BotCommand:
    """큐에 넣을 명령 객체."""
    command: CommandType
    args: dict[str, Any] | None = None
    response_future: asyncio.Future | None = None
    loop: asyncio.AbstractEventLoop | None = None

    def set_response(self, text: str):
        """비동기 스레드에서 응답을 설정."""
        if self.response_future and self.loop:
            self.loop.call_soon_threadsafe(
                self.response_future.set_result, text
            )


HELP_TEXT = """```
📋 QuantBox 명령어
─────────────────────────
!status       슬롯 상태 + 잔고
!close all    전체 포지션 청산
!close <N>    슬롯 N번 청산
!pause        신규 진입 중지
!resume       신규 진입 재개
!stop         전체 청산 + 종료
!pnl          세션 손익 요약
!blacklist    블랙리스트 조회
!blacklist clear  블랙리스트 초기화
!help         이 메시지
─────────────────────────
```"""


class TradingBot(discord.Client):
    """Discord Bot 클라이언트.

    메시지를 파싱하여 BotCommand로 변환하고 command_queue에 넣는다.
    메인 트레이딩 루프가 명령을 실행한 뒤 응답을 돌려보낸다.

    Args:
        command_queue: 메인 스레드로 명령을 전달하는 큐.
        channel_id: 명령을 수신할 Discord 채널 ID.
    """

    def __init__(self, command_queue: queue.Queue, channel_id: int):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.command_queue = command_queue
        self.channel_id = channel_id

    async def on_ready(self):
        logger.info(f"Discord Bot connected as {self.user}")
        channel = self.get_channel(self.channel_id)
        if channel:
            await channel.send("🟢 **QuantBox Bot 연결됨** — `!help`로 명령어 확인")

    async def on_message(self, message: discord.Message):
        # 봇 자신의 메시지 무시
        if message.author == self.user:
            return

        # 지정된 채널에서만 수신
        if message.channel.id != self.channel_id:
            return

        content = message.content.strip()
        if not content.startswith('!'):
            return

        parts = content.split()
        cmd = parts[0].lower()

        try:
            bot_cmd = self._parse_command(cmd, parts)
        except ValueError as e:
            await message.channel.send(f"❌ {e}")
            return

        if bot_cmd is None:
            return

        # !help는 즉시 응답 (큐 불필요)
        if bot_cmd.command == CommandType.HELP:
            await message.channel.send(HELP_TEXT)
            return

        # 응답용 Future 설정
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bot_cmd.response_future = future
        bot_cmd.loop = loop

        # 명령을 큐에 넣고 응답 대기
        self.command_queue.put(bot_cmd)
        await message.channel.send(f"⏳ `{content}` 처리 중...")

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            await message.channel.send(response)
        except asyncio.TimeoutError:
            await message.channel.send("⚠️ 명령 타임아웃 (30초)")

    def _parse_command(
        self, cmd: str, parts: list[str]
    ) -> BotCommand | None:
        """메시지를 BotCommand로 파싱.

        Args:
            cmd: 명령어 (e.g. '!status').
            parts: 전체 메시지 split 결과.

        Returns:
            BotCommand 또는 None (알 수 없는 명령).

        Raises:
            ValueError: 인자가 잘못된 경우.
        """
        if cmd == '!status':
            return BotCommand(command=CommandType.STATUS)
        elif cmd == '!close':
            if len(parts) < 2:
                raise ValueError("`!close all` 또는 `!close <슬롯번호>` 형식으로 입력")
            arg = parts[1].lower()
            if arg == 'all':
                return BotCommand(command=CommandType.CLOSE_ALL)
            else:
                try:
                    slot_id = int(arg)
                except ValueError:
                    raise ValueError(f"슬롯 번호는 숫자여야 합니다: `{arg}`")
                return BotCommand(
                    command=CommandType.CLOSE_SLOT,
                    args={'slot_id': slot_id},
                )
        elif cmd == '!pause':
            return BotCommand(command=CommandType.PAUSE)
        elif cmd == '!resume':
            return BotCommand(command=CommandType.RESUME)
        elif cmd == '!stop':
            return BotCommand(command=CommandType.STOP)
        elif cmd == '!pnl':
            return BotCommand(command=CommandType.PNL)
        elif cmd == '!blacklist':
            if len(parts) >= 2 and parts[1].lower() == 'clear':
                return BotCommand(
                    command=CommandType.BLACKLIST,
                    args={'action': 'clear'},
                )
            return BotCommand(command=CommandType.BLACKLIST)
        elif cmd == '!help':
            return BotCommand(command=CommandType.HELP)
        else:
            return None


class DiscordBotRunner:
    """Discord Bot을 별도 스레드에서 실행하는 러너.

    메인 트레이딩 루프에서 사용:
        runner = DiscordBotRunner(token, channel_id)
        runner.start()          # 봇 스레드 시작
        ...
        cmd = runner.get_command()  # 매 사이클마다 큐 확인
        if cmd:
            result = process(cmd)
            cmd.set_response(result)
        ...
        runner.stop()           # 종료 시

    Args:
        token: Discord Bot 토큰.
        channel_id: 명령 수신 채널 ID.
    """

    def __init__(self, token: str, channel_id: int):
        self.token = token
        self.channel_id = channel_id
        self.command_queue: queue.Queue[BotCommand] = queue.Queue()
        self._bot: TradingBot | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self):
        """봇 스레드를 시작한다."""
        self._thread = threading.Thread(
            target=self._run_bot,
            daemon=True,
            name='discord-bot',
        )
        self._thread.start()
        logger.info("Discord Bot thread started")

    def _run_bot(self):
        """별도 스레드에서 asyncio 이벤트 루프를 실행."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._bot = TradingBot(
            command_queue=self.command_queue,
            channel_id=self.channel_id,
        )

        try:
            self._loop.run_until_complete(self._bot.start(self.token))
        except Exception as e:
            logger.error(f"Discord Bot error: {e}")
        finally:
            # 잔여 태스크 정리 — "Task was destroyed but it is pending" 방지
            pending = asyncio.all_tasks(self._loop)
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def get_command(self) -> BotCommand | None:
        """큐에서 명령을 가져온다 (non-blocking).

        Returns:
            BotCommand 또는 None (큐가 비어있을 때).
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def get_all_commands(self) -> list[BotCommand]:
        """큐에 쌓인 모든 명령을 가져온다.

        Returns:
            BotCommand 리스트.
        """
        commands = []
        while True:
            cmd = self.get_command()
            if cmd is None:
                break
            commands.append(cmd)
        return commands

    def stop(self, timeout: float = 5.0):
        """봇을 안전하게 종료한다.

        discord.py Client.close()를 호출하고 스레드가 정리될 때까지 대기한다.
        대기하지 않으면 asyncio 커넥터/태스크 리소스 누수가 발생한다.

        Args:
            timeout: 종료 대기 최대 시간(초). 초과 시 강제 진행.
        """
        if self._bot and self._loop and not self._loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._bot.close(), self._loop
                )
                future.result(timeout=timeout)
            except Exception as e:
                logger.warning(f"Discord Bot close error (non-fatal): {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "Discord Bot thread did not terminate within "
                    f"{timeout}s — proceeding anyway"
                )

        logger.info("Discord Bot stopped")

    @property
    def is_running(self) -> bool:
        """봇 스레드가 살아있는지 확인."""
        return self._thread is not None and self._thread.is_alive()
