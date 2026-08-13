#!/usr/bin/env python3
"""호가/체결/청산 소켓 데이터 수집기 (연구 전용).

공개 마켓 스트림만 구독하므로 **API 키가 필요 없고**, 계정 레이트리밋과도 무관하다.
트레이딩 로직을 일절 건드리지 않는다 — 순수 데이터 축적용.

배경:
    Binance는 USDT-M `bookTicker` **과거 아카이브**를 2024-03-30자로 중단했다
    (HTTP 검증: 03-30 → 200, 03-31 → 404). 라이브 스트림은 정상 동작한다.
    따라서 최우선호가 기반 연구는 "지금부터 직접 모으는" 방법밖에 없으며,
    그렇게 모은 데이터는 공개 시장에서 구매 불가능한 독점 자산이 된다.

수집 스트림:
    <symbol>@bookTicker  — 최우선 매수/매도 호가 + 수량 (실시간, 갱신마다)
    <symbol>@trade       — 개별 체결. `m`(매수자=메이커)로 테이커 방향 도출 → 주문흐름 불균형(OFI)

이 환경에서 확인된 스트림 가용성 (2026-08-02 실측, /ws/ 및 /stream? 양쪽 동일):
    @bookTicker    ✅ 동작
    @trade         ✅ 동작
    @aggTrade      ❌ 0건
    @kline_1m      ❌ 0건
    @markPrice@1s  ❌ 0건
    !forceOrder@arr ❌ 0건

    kline_1m처럼 기본적인 스트림까지 막힌 것으로 보아 네트워크/지역 레벨의 선택적 차단으로
    판단된다. 코드로 우회 불가. 그래서 aggTrade 대신 **@trade**를 쓴다 — 비집계 버전이라
    데이터가 오히려 더 상세하며 OFI 계산에는 동등하다.

    ⚠️ forceOrder(청산)가 막혀 있어 **청산 캐스케이드 연구는 이 환경에서 불가**하다.
       다른 네트워크에서 재확인이 필요하다.

저장:
    data/socket/<stream>/<symbol>/<YYYY-MM-DD-HH>.jsonl.gz  (시간 단위 로테이션)
    각 줄 = {"recv_ns": <로컬 수신 시각>, "d": <원본 페이로드>}

    recv_ns는 **로컬 수신 시각**이다. 거래소 타임스탬프는 페이로드 안에 그대로 보존하되,
    정렬·지연 측정에는 로컬 시각을 쓴다 (클럭 스큐 방지).

사용:
    uv run python scripts/collect_book_data.py                 # 기본 7심볼
    uv run python scripts/collect_book_data.py --symbols BTCUSDT,ETHUSDT
    uv run python scripts/collect_book_data.py --no-aggtrade   # 디스크 절약
    uv run python scripts/collect_book_data.py --max-gb 50     # 디스크 상한
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import asyncio

import websockets

from quantbox.collect.rotating_writer import (  # noqa: F401  (수집기 로그가 _warn_throttled를 쓴다)
    RotatingWriter,
    _warn_throttled,
    disk_used_gb,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "socket"

STREAM_BASE = "wss://fstream.binance.com"
CHUNK = 50  # 연결당 스트림 수
RECONNECT_AFTER_S = 23 * 3600  # Binance는 24h에 강제 종료한다 — 그 전에 먼저 재연결

# 메이저 2 + 중소형 5.
# 얇은 호가일수록 불균형의 정보량이 크다는 가설을 검증하려면 두 그룹이 모두 필요하다.
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "DOGEUSDT",
    "BANDUSDT",
    "BELUSDT",
    "DASHUSDT",
    "ENJUSDT",
    "ALGOUSDT",
]

FLUSH_INTERVAL_S = 5.0
STATS_INTERVAL_S = 60.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collector")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="쉼표 구분 심볼")
    ap.add_argument("--no-bookticker", action="store_true")
    ap.add_argument("--no-trade", action="store_true", help="체결 스트림 제외 (용량 절약)")
    ap.add_argument("--forceorder", action="store_true", help="청산 스트림 시도 (이 환경에선 0건 확인됨)")
    ap.add_argument("--max-gb", type=float, default=100.0, help="누적 상한(GB). 초과 시 정지")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("심볼이 비어 있음")
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    writer = RotatingWriter(out_root)

    streams: list[str] = []
    if not args.no_bookticker:
        streams += [f"{s.lower()}@bookTicker" for s in symbols]
    if not args.no_trade:
        streams += [f"{s.lower()}@trade" for s in symbols]
    if args.forceorder:
        streams.append("!forceOrder@arr")

    if not streams:
        logger.error("구독할 스트림이 없음")
        return 1

    logger.info(f"심볼 {len(symbols)}개: {', '.join(symbols)}")
    logger.info(f"스트림 {len(streams)}개 구독 시작 → {out_root}")
    logger.info("공개 마켓 스트림 — API 키 불필요, 계정 레이트리밋 무관")

    stop = threading.Event()

    def handle(msg: dict) -> None:
        recv_ns = time.time_ns()
        try:
            # multiplex 페이로드: {"stream": "btcusdt@bookTicker", "data": {...}}
            data = msg.get("data", msg)
            stream_name = msg.get("stream", "")

            if "@bookTicker" in stream_name:
                kind, sym = "bookTicker", data.get("s", "UNKNOWN")
            elif "@trade" in stream_name:
                kind, sym = "trade", data.get("s", "UNKNOWN")
            elif "forceOrder" in stream_name:
                kind = "forceOrder"
                sym = (data.get("o") or {}).get("s", "UNKNOWN")
            else:
                if data.get("e") == "error":
                    _warn_throttled(f"스트림 에러: {str(data)[:200]}", key="stream_error")
                    return
                kind, sym = "other", data.get("s", "UNKNOWN")

            writer.submit(kind, sym, recv_ns, data)
        except Exception as e:  # 수집기는 어떤 경우에도 죽지 않는다
            _warn_throttled(f"핸들러 예외: {e}", key="handler_exc")

    # raw websockets를 쓴다. python-binance의 ThreadedWebsocketManager는 내부 asyncio 큐가
    # 기본 100건이라 15스트림에서 17초 만에 넘쳤고(BinanceWebsocketQueueOverflow),
    # 키워도 aggTrade가 끝내 도착하지 않았다. raw 엔드포인트로 직접 붙어도 동일했으므로
    # 래퍼가 얻어주는 게 없다 — 대신 재연결·백오프를 직접 통제한다.
    async def consume(chunk: list[str]) -> None:
        backoff = 1.0
        while not stop.is_set():
            url = STREAM_BASE + "/stream?streams=" + "/".join(chunk)
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=None) as ws:
                    logger.info(f"연결됨 ({len(chunk)}스트림)")
                    backoff = 1.0
                    deadline = time.monotonic() + RECONNECT_AFTER_S
                    while not stop.is_set() and time.monotonic() < deadline:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            _warn_throttled("30초간 수신 없음", key="idle")
                            continue
                        handle(json.loads(raw))
                    logger.info("예방적 재연결 (24h 강제종료 회피)")
            except Exception as e:
                if stop.is_set():
                    return
                _warn_throttled(f"연결 끊김: {type(e).__name__} {str(e)[:80]} — {backoff:.0f}s 후 재시도", key="reconn")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def run_all() -> None:
        # Binance는 연결당 스트림 수에 상한이 있다. 넉넉히 나눠 붙는다.
        chunks = [streams[i : i + CHUNK] for i in range(0, len(streams), CHUNK)]
        await asyncio.gather(*(consume(c) for c in chunks))

    def ws_thread() -> None:
        try:
            asyncio.run(run_all())
        except Exception as e:
            logger.error(f"소켓 스레드 종료: {e}")

    t = threading.Thread(target=ws_thread, name="ws", daemon=True)
    t.start()

    def shutdown(signum, frame) -> None:  # noqa: ANN001
        logger.info(f"시그널 {signum} 수신 — 종료 중")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    started = time.monotonic()
    last_stats = started
    last_counts: dict[str, int] = {}

    try:
        while not stop.is_set():
            time.sleep(FLUSH_INTERVAL_S)
            writer.flush()

            now = time.monotonic()
            if now - last_stats >= STATS_INTERVAL_S:
                elapsed = now - started
                gb = disk_used_gb(out_root)
                parts = []
                for k in sorted(writer.counts):
                    total = writer.counts[k]
                    delta = total - last_counts.get(k, 0)
                    last_counts[k] = total
                    parts.append(f"{k}={total:,}(+{delta:,})")
                extra = f" | q={writer.qsize:,} 디스크 {gb:.2f}GB"
                if writer.dropped:
                    extra += f" | ⚠ 드랍 {writer.dropped:,}"
                logger.info(f"[{elapsed / 60:.0f}분] " + " ".join(parts) + extra)
                if gb >= args.max_gb:
                    logger.error(f"디스크 상한 {args.max_gb}GB 도달 — 정지")
                    break
                if not parts:
                    logger.warning("아직 수신 0건 — 연결 확인 필요")
                last_stats = now
    finally:
        logger.info("소켓 종료 중...")
        stop.set()
        t.join(timeout=10)
        writer.flush()
        writer.close()
        total = sum(writer.counts.values())
        logger.info(f"종료. 총 {total:,}건 / {disk_used_gb(out_root):.2f}GB → {out_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
