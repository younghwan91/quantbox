#!/usr/bin/env python3
"""신규 크립토 무기한 상장 감시기 (연구 전용, 주문 없음).

`exchangeInfo`를 60초마다 조회해 처음 보는 크립토 USDT 무기한을 감지하면
그 심볼만 구독하는 기록 스레드를 띄운다. 공개 엔드포인트라 API 키가 필요 없고
계정 레이트리밋과 무관하다(가중치 1).

기존 수집기(tmux `collector`)와 **완전히 분리된 프로세스**다. 저장 레이아웃만
공유하므로 기존 분석 코드가 그대로 동작한다.

사용:
    uv run python scripts/listing_probe.py              # 상시 감시 (tmux 'listing')
    uv run python scripts/listing_probe.py --once       # 한 번만 확인하고 종료
    uv run python scripts/listing_probe.py --days 7     # 이벤트당 기록 기간
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from quantbox.listing.events import EventLog
from quantbox.listing.recorder import record
from quantbox.listing.universe import (
    SeenStore,
    detect_new,
    fetch_exchange_info,
    target_symbols,
    unknown_types,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "socket"
STATE_DIR = ROOT / "data" / "listing"

POLL_INTERVAL_S = 60.0
PENDING_TIMEOUT_S = 48 * 3600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("listing.probe")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="한 번만 확인하고 종료")
    ap.add_argument("--days", type=float, default=7.0, help="이벤트당 기록 기간(일)")
    ap.add_argument("--max-gb", type=float, default=5.0, help="이벤트당 디스크 상한(GB)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    seen_store = SeenStore(STATE_DIR / "seen_symbols.json")
    events = EventLog(STATE_DIR / "events.json")

    stop = threading.Event()

    def shutdown(signum, frame) -> None:  # noqa: ANN001
        logger.info(f"시그널 {signum} 수신 — 종료 중")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("상장 감시 시작 — exchangeInfo 60초 폴링, 주문 없음")
    logger.info(f"기존 'collector' 세션은 건드리지 않는다. 기록 → {out_root}")

    # 처음 본 심볼이 아직 TRADING이 아니면 여기서 대기시킨다.
    pending: dict[str, float] = {}
    recorders: list[threading.Thread] = []
    #: 이 프로세스가 이미 기록을 시작한 심볼. 같은 심볼에 두 번째 스레드가 뜨면
    #: 두 RotatingWriter가 같은 .jsonl.gz에 각자 버퍼로 append해 gzip 멤버가
    #: 엇갈리고 파일이 깨진다. 마지막 방어선이다.
    recording: set[str] = set()

    def remember(symbols: set[str]) -> None:
        """본 심볼을 저장하되, 실패해도 감시를 멈추지 않는다.

        `bootstrap`의 저장 실패는 기동을 막아야 하지만(중복 방지가 아예 성립하지
        않으므로) 이벤트 감지 중의 실패는 다르다. 기록을 놓치는 것보다 중복 위험을
        감수하는 편이 낫다 — 상장은 주당 1건뿐이고 놓치면 몇 주를 다시 기다린다.
        """
        try:
            seen_store.save(symbols)
        except OSError as e:
            logger.error(f"본 심볼 저장 실패({e}) — 재시작 시 이 이벤트를 다시 기록할 수 있다")

    while not stop.is_set():
        try:
            info = fetch_exchange_info()
        except Exception as e:
            logger.warning(f"exchangeInfo 조회 실패: {type(e).__name__} {str(e)[:80]}")
            if args.once:
                return 1
            stop.wait(POLL_INTERVAL_S)
            continue

        if unknown := unknown_types(info):
            logger.error(f"⚠ 처음 보는 underlyingType: {unknown} — 분류 체계가 바뀌었다. KNOWN_TYPES 갱신 필요")

        seen = seen_store.bootstrap(info)
        targets = target_symbols(info)
        by_sym = {s["symbol"]: s for s in info["symbols"]}

        for sym, meta in detect_new(targets, seen).items():
            pending.setdefault(sym, time.monotonic())
            logger.info(f"신규 대상 감지: {sym} status={meta.get('status')}")

        for sym in list(pending):
            meta = by_sym.get(sym)
            if meta is None:
                continue
            if meta.get("status") == "TRADING":
                del pending[sym]
                if sym in recording:
                    continue  # 이미 기록 중 — 두 번째 스레드는 파일을 깨뜨린다
                logger.info(f"▶ {sym} 거래 시작 — 기록 개시 ({args.days}일)")
                events.add(sym, meta, time.time_ns())
                # `seen`을 제자리에서 갱신한 뒤 저장한다. `remember(seen | {sym})`처럼
                # 사본을 넘기면 같은 주기에 두 심볼이 감지됐을 때 두 번째 저장이 첫 번째를
                # 지운다 — 다음 폴링에서 첫 심볼이 신규로 재감지돼 기록 스레드가 하나 더 뜬다.
                seen.add(sym)
                remember(seen)
                recording.add(sym)
                t = threading.Thread(
                    target=record,
                    args=(sym, out_root, args.days * 86400, args.max_gb, stop),
                    name=f"rec-{sym}",
                    daemon=True,
                )
                t.start()
                recorders.append(t)
            elif time.monotonic() - pending[sym] > PENDING_TIMEOUT_S:
                logger.warning(f"{sym} 48시간 동안 TRADING이 되지 않았다 — 포기")
                seen.add(sym)
                remember(seen)
                del pending[sym]

        if args.once:
            logger.info(f"1회 확인 완료 — 대상 {len(targets)}개, 대기 {len(pending)}개")
            return 0

        stop.wait(POLL_INTERVAL_S)

    logger.info("종료 대기 중 (기록 스레드)...")
    for t in recorders:
        t.join(timeout=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
