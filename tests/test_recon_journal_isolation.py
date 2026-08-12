"""테스트가 프로덕션 리컨실 저널을 오염시키지 않는가.

2026-08-12 발견: pytest 1회 실행마다 `data/journal/recon.db`에 53행이 쌓이고 있었다.
누적 22,836건 중 **31%가 `TESTUSDT`** — 운영 기록이 테스트 데이터로 덮여
가동률·사고 분석이 조용히 틀린 상태였다.

원인은 `core/lifecycle.py:107`과 `execution/recon_journal.py`의 하드코딩 경로
`Path("data/journal/recon.db")`이고, 진짜 scalper를 만드는 테스트가 거래 저널만
MagicMock으로 막고 리컨실 저널은 두었기 때문이다.

`conftest._isolate_recon_journal`(autouse)가 생성 시점에 경로를 가로챈다.
그 가드가 살아있는지 여기서 고정한다.
"""

from __future__ import annotations

from pathlib import Path

from quantbox.execution.recon_journal import ReconEventType, ReconJournal

PROD_DB = Path("data/journal/recon.db")


def _prod_rows() -> int:
    """프로덕션 DB의 현재 행 수 (없으면 -1)."""
    if not PROD_DB.exists():
        return -1
    import sqlite3

    conn = sqlite3.connect(str(PROD_DB))
    try:
        return int(conn.execute("select count(*) from recon_events").fetchone()[0])
    finally:
        conn.close()


def test_hardcoded_prod_path_is_redirected() -> None:
    """프로덕션 경로로 생성해도 실제 파일에는 쓰이지 않는다."""
    before = _prod_rows()

    journal = ReconJournal(PROD_DB)
    journal.record(
        event_type=ReconEventType.GHOST_DETECTED,
        symbol="TESTUSDT",
        action="isolation guard probe",
        result="success",
    )

    assert _prod_rows() == before, "테스트가 프로덕션 recon.db에 기록했다 — 가드가 죽었다"


def test_redirected_journal_still_functions() -> None:
    """격리는 하되 기능은 살아있어야 한다 — 무력화가 아니라 우회다."""
    journal = ReconJournal(PROD_DB)
    journal.record(
        event_type=ReconEventType.GHOST_DETECTED,
        symbol="TESTUSDT",
        action="probe",
        result="success",
    )

    assert journal.count_by_type(ReconEventType.GHOST_DETECTED, hours=1) >= 1, (
        "격리된 저널이 기록을 못 읽는다 — 가드가 기능까지 죽였다"
    )
