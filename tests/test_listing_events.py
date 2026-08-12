"""기록한 상장 이벤트의 메타데이터.

`onboard_ms`가 없으면 분석기가 "상장 후 몇 시간"을 계산할 수 없고, 진입 창
(1~24시간) 자체가 정의되지 않는다. 소켓 파일만 남기면 3주 뒤에 이 사실을 알게 된다.
"""

from __future__ import annotations

from pathlib import Path

from quantbox.listing.events import EventLog


def test_records_onboard_time(tmp_path: Path) -> None:
    """진입 창 계산에 필요한 상장 시각이 남는다."""
    log = EventLog(tmp_path / "events.json")
    log.add("DOSUSDT", {"onboardDate": 1754924400000, "underlyingType": "COIN", "status": "TRADING"}, 999)

    rec = log.all()["DOSUSDT"]
    assert rec["onboard_ms"] == 1754924400000
    assert rec["underlying_type"] == "COIN"
    assert rec["detected_ns"] == 999


def test_survives_restart(tmp_path: Path) -> None:
    """재시작해도 이벤트가 남는다."""
    p = tmp_path / "events.json"
    EventLog(p).add("AAAUSDT", {"onboardDate": 1, "underlyingType": "COIN", "status": "TRADING"}, 1)
    EventLog(p).add("BBBUSDT", {"onboardDate": 2, "underlyingType": "COIN", "status": "TRADING"}, 2)

    assert set(EventLog(p).all()) == {"AAAUSDT", "BBBUSDT"}


def test_first_detection_wins(tmp_path: Path) -> None:
    """같은 심볼을 두 번 넣어도 첫 감지 시각을 유지한다 — 재시작이 시각을 덮으면 안 된다."""
    log = EventLog(tmp_path / "events.json")
    log.add("AAAUSDT", {"onboardDate": 1, "underlyingType": "COIN", "status": "TRADING"}, 100)
    log.add("AAAUSDT", {"onboardDate": 1, "underlyingType": "COIN", "status": "TRADING"}, 200)

    assert log.all()["AAAUSDT"]["detected_ns"] == 100
