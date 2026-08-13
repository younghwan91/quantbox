"""스트림×심볼별 시간 단위 gzip JSONL writer.

`scripts/collect_book_data.py`에서 추출했다. 상장 비용 측정기가 같은 저장
레이아웃을 써야 기존 분석 코드가 그대로 동작하고, 복사본 두 벌이 서서히
어긋나는 일이 없다.

소켓 콜백은 절대 블로킹하면 안 된다 — bookTicker는 심볼당 초당 수십 건이 오므로
콜백 안에서 gzip 압축을 하면 큐가 넘친다. 콜백은 deque에 append만 하고(O(1),
GIL 하에서 원자적) 압축·디스크 쓰기는 별도 스레드가 맡는다.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

MAX_QUEUE = 500_000  # 메모리 상한. 초과분은 드랍하고 카운트한다 (조용한 유실 방지)
WARN_THROTTLE_S = 10.0

logger = logging.getLogger("collect.writer")

_last_warn: dict[str, float] = {}


def _warn_throttled(msg: str, key: str | None = None) -> None:
    """동일 경고의 폭주를 막는다.

    첫 테스트에서 메시지마다 경고를 찍다가 콜백이 느려져 소켓 큐가 넘쳤다.
    로깅이 수집을 망가뜨리면 안 된다.
    """
    k = key or msg[:60]
    now = time.monotonic()
    if now - _last_warn.get(k, 0.0) >= WARN_THROTTLE_S:
        _last_warn[k] = now
        logger.warning(msg)


class RotatingWriter:
    """스트림×심볼별 시간 단위 gzip JSONL writer.

    소켓 콜백은 절대 블로킹되면 안 된다 — python-binance의 내부 큐는 100건에서
    넘치고(BinanceWebsocketQueueOverflow) 그 뒤 메시지는 유실된다. bookTicker는
    심볼당 초당 수십 건이 오므로 콜백 안에서 gzip 압축을 하면 확실히 터진다.

    따라서 콜백은 deque에 append만 하고(O(1), GIL 하에서 원자적), 압축·디스크 쓰기는
    별도 스레드가 담당한다.
    """

    def __init__(self, out_root: Path) -> None:
        self.out_root = out_root
        self._files: dict[tuple[str, str], tuple[str, object]] = {}
        self._q: deque[tuple[str, str, bytes]] = deque()
        self.counts: dict[str, int] = defaultdict(int)
        self.dropped = 0
        self.bytes_written = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain_loop, name="writer", daemon=True)
        self._thread.start()

    @staticmethod
    def _hour_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")

    def submit(self, stream: str, symbol: str, recv_ns: int, payload: dict) -> None:
        """콜백에서 호출. 직렬화만 하고 즉시 반환한다."""
        if len(self._q) > MAX_QUEUE:
            self.dropped += 1
            return
        line = json.dumps({"recv_ns": recv_ns, "d": payload}, separators=(",", ":")) + "\n"
        self._q.append((stream, symbol, line.encode()))

    def _handle(self, stream: str, symbol: str, enc: bytes) -> None:
        hour = self._hour_key()
        key = (stream, symbol)
        cur = self._files.get(key)
        if cur is None or cur[0] != hour:
            if cur is not None:
                cur[1].close()  # type: ignore[attr-defined]
            path = self.out_root / stream / symbol
            path.mkdir(parents=True, exist_ok=True)
            # 바이너리 append — 텍스트 모드("at")는 str만 받으므로 bytes 쓰기에 실패한다
            fh = gzip.open(path / f"{hour}.jsonl.gz", "ab", compresslevel=6)
            self._files[key] = (hour, fh)
            cur = self._files[key]
        cur[1].write(enc)  # type: ignore[attr-defined]
        self.counts[stream] += 1
        self.bytes_written += len(enc)

    def _drain_loop(self) -> None:
        while not self._stop.is_set() or self._q:
            try:
                item = self._q.popleft()
            except IndexError:
                time.sleep(0.02)
                continue
            try:
                self._handle(*item)
            except Exception as e:
                _warn_throttled(f"쓰기 실패: {e}")

    def flush(self) -> None:
        for _, fh in list(self._files.values()):
            try:
                fh.flush()  # type: ignore[attr-defined]
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=20)
        for _, fh in list(self._files.values()):
            try:
                fh.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._files.clear()

    @property
    def qsize(self) -> int:
        return len(self._q)


def disk_used_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*.gz")) / 1e9
