"""한 심볼의 호가·체결을 정해진 기간 기록한다.

기존 수집기(`scripts/collect_book_data.py`)를 재사용하지 않는다. 심볼을 추가하려면
재시작해야 하고, 그 프로세스는 2026-08-10부터 복구 불가능한 데이터를 쌓고 있다.
저장 레이아웃만 공유해 기존 분석 코드가 그대로 동작하게 한다.

재연결·백오프는 기존 수집기에서 검증된 패턴을 따른다 — python-binance 래퍼는
내부 큐가 100건이라 15스트림에서 17초 만에 넘쳤으므로 raw websockets를 쓴다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import websockets

from quantbox.collect.rotating_writer import RotatingWriter

logger = logging.getLogger("listing.recorder")

STREAM_BASE = "wss://fstream.binance.com"

#: 구독할 스트림. `depth20@500ms`는 실측으로 고른 값이다 (2026-08-13, DOGEUSDT):
#: 기본 250ms 3.73건/s·58MB/일, **500ms 1.97건/s·31MB/일**, 100ms 8.33건/s·130MB/일.
#: 시간대별 중앙값을 재는 데 초당 2표본이면 충분하고 100ms는 10배 용량으로 같은 답을 준다.
STREAM_SUFFIXES: tuple[str, ...] = ("bookTicker", "trade", "depth20@500ms")

RECONNECT_AFTER_S = 23 * 3600  # 바이낸스는 24h에 강제 종료한다 — 그 전에 먼저 재연결
FLUSH_INTERVAL_S = 5.0
DISK_CHECK_INTERVAL_S = 60.0

#: 소켓 수신 대기 시간. 종료 신호는 이 대기를 **중간에 깨지 못하므로**, 이 값이
#: 곧 종료 지연의 상한이다. 아래 `JOIN_TIMEOUT_S`가 이보다 커야 한다.
RECV_TIMEOUT_S = 30.0
JOIN_TIMEOUT_S = RECV_TIMEOUT_S + 5.0


def stream_kind(stream_name: str) -> str:
    """스트림 이름 → 저장 디렉터리 이름.

    `"btcusdt@depth20@500ms"` → `"depth20"`. 갱신 주기를 경로에 넣지 않는다 —
    주기를 바꿨을 때 같은 심볼이 두 디렉터리로 갈리면 분석이 조용히 반쪽이 된다.
    """
    parts = stream_name.split("@")
    return parts[1] if len(parts) > 1 else stream_name


def record(
    symbol: str,
    out_root: Path,
    duration_s: float,
    max_gb: float = 5.0,
    stop: threading.Event | None = None,
) -> dict[str, int]:
    """`symbol`을 `duration_s`초 동안 기록하고 스트림별 건수를 반환한다.

    Args:
        symbol: 대문자 심볼 (예: `"DOSUSDT"`).
        out_root: `data/socket` 루트. 하위에 `<stream>/<symbol>/<hour>.jsonl.gz`.
        duration_s: 기록 시간. 스펙 기본은 7일(604800).
        max_gb: 이 이벤트가 쓸 수 있는 상한. 폭주 방어용.
        stop: 외부 중단 신호.
    """
    symbol = symbol.upper()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    writer = RotatingWriter(out_root)
    stop = stop or threading.Event()

    streams = [f"{symbol.lower()}@{s}" for s in STREAM_SUFFIXES]
    logger.info(f"{symbol} 기록 시작 — {len(streams)}스트림, {duration_s / 3600:.1f}시간, 상한 {max_gb}GB")

    def handle(msg: dict) -> None:
        recv_ns = time.time_ns()
        data = msg.get("data", msg)
        name = msg.get("stream", "")
        if not name:
            logger.warning(f"스트림 이름이 비어 있다 — 건너뜀: {str(msg)[:200]}")
            return
        if data.get("e") == "error":
            logger.warning(f"스트림 에러: {str(data)[:200]}")
            return
        writer.submit(stream_kind(name), data.get("s", symbol), recv_ns, data)

    async def consume() -> None:
        backoff = 1.0
        url = STREAM_BASE + "/stream?streams=" + "/".join(streams)
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=None) as ws:
                    logger.info(f"{symbol} 연결됨")
                    backoff = 1.0
                    deadline = time.monotonic() + RECONNECT_AFTER_S
                    while not stop.is_set() and time.monotonic() < deadline:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            logger.warning(f"{symbol} {RECV_TIMEOUT_S:.0f}초간 수신 없음")
                            continue
                        handle(json.loads(raw))
            except Exception as e:
                if stop.is_set():
                    return
                logger.warning(f"{symbol} 연결 끊김: {type(e).__name__} — {backoff:.0f}s 후 재시도")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    t = threading.Thread(target=lambda: asyncio.run(consume()), name=f"ws-{symbol}", daemon=True)
    t.start()

    started = time.monotonic()
    last_disk = started
    try:
        while not stop.is_set() and time.monotonic() - started < duration_s:
            time.sleep(FLUSH_INTERVAL_S)
            writer.flush()
            now = time.monotonic()
            if now - last_disk >= DISK_CHECK_INTERVAL_S:
                last_disk = now
                # `disk_used_gb(out_root)`를 쓰면 안 된다 — out_root는 가동 중인 수집기와
                # 공유하는 `data/socket`이고 거기엔 이미 3.7GB가 쌓여 있다(2026-08-13 실측).
                # 트리 전체를 재면 신규 기록이 시작하자마자 상한의 75%를 쓴 상태가 되고,
                # 하루 뒤면 이벤트가 와도 60초 만에 중단된다. **이 기록이 쓴 바이트만** 센다.
                gb = writer.bytes_written / 1e9
                if gb >= max_gb:
                    logger.error(f"{symbol} 이벤트 디스크 상한 {max_gb}GB 도달 — 기록 중단")
                    break
    finally:
        stop.set()
        # 종료 신호는 대기 중인 recv를 깨지 못한다 — recv 타임아웃보다 넉넉히 기다려야
        # 스레드가 살아 있는 채로 writer를 닫는 일이 없다.
        t.join(timeout=JOIN_TIMEOUT_S)
        if t.is_alive():
            logger.warning(
                f"{symbol} 소켓 스레드가 {JOIN_TIMEOUT_S:.0f}초 안에 끝나지 않았다 — 일부 수신이 유실될 수 있다"
            )
        writer.flush()
        writer.close()
        logger.info(f"{symbol} 기록 종료 — {dict(writer.counts)}")

    return dict(writer.counts)
