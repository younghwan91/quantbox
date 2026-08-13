"""기록한 상장 이벤트의 메타데이터.

소켓 파일만으로는 "상장 후 몇 시간"을 알 수 없다. 진입 창(1~24시간)이 판정의
정의 자체이므로 상장 시각을 이벤트와 함께 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path


class EventLog:
    """`{symbol: 레코드}` JSON 저장소."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def all(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return dict(json.loads(self.path.read_text()))
        except (json.JSONDecodeError, OSError):
            return {}

    def add(self, symbol: str, info: dict, detected_ns: int) -> None:
        """이벤트를 등록한다. 이미 있으면 **덮지 않는다** — 첫 감지 시각이 진실이다."""
        events = self.all()
        if symbol in events:
            return
        events[symbol] = {
            "symbol": symbol,
            "onboard_ms": info.get("onboardDate"),
            "detected_ns": detected_ns,
            "underlying_type": info.get("underlyingType"),
            "status_at_detect": info.get("status"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(events, indent=2, sort_keys=True))
        tmp.replace(self.path)
