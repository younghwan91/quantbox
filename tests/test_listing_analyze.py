"""기록 파일 → 시간대별 스프레드 표 → 사전 등록된 판정.

임계값을 결과를 보고 바꾸지 않는다. 2026-08-12에 과거 리포트에서 합격 기준이
사후 하향된 사례를 발견했고(OOS/IS 0.85→0.50, WR 42%→40%), 같은 날 나 자신도
두 번 그 함정에 빠졌다.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from quantbox.listing.analyze import (
    ENTRY_WINDOW_H,
    EXTEND_BP,
    MAX_UNFILLABLE_FRAC,
    PASS_BP,
    hourly_profile,
    read_depth_samples,
    verdict,
)

SOCKET = Path(__file__).resolve().parents[1] / "data" / "socket"


def test_thresholds_match_the_spec() -> None:
    """사전 등록된 값이다 — 바뀌면 스펙과 대조해야 한다."""
    assert (PASS_BP, EXTEND_BP, MAX_UNFILLABLE_FRAC) == (25.0, 50.0, 0.10)
    assert ENTRY_WINDOW_H == (1.0, 24.0)


def test_verdict_boundaries() -> None:
    """경계값이 어느 쪽에 속하는지 못 박는다 — 25.0bp가 진행인가 재검토인가."""
    assert verdict([(2.0, 10.0, True)] * 200)["decision"] == "진행"
    assert verdict([(2.0, 25.0, True)] * 200)["decision"] == "진행"  # 이하 = 진행
    assert verdict([(2.0, 25.1, True)] * 200)["decision"] == "재검토"
    assert verdict([(2.0, 50.0, True)] * 200)["decision"] == "재검토"  # 이하 = 재검토
    assert verdict([(2.0, 50.1, True)] * 200)["decision"] == "종료"


def test_unfillable_overrides_a_good_spread() -> None:
    """체결 불가가 10%를 넘으면 스프레드가 아무리 좋아도 종료다."""
    samples = [(2.0, 1.0, True)] * 170 + [(2.0, 1.0, False)] * 30  # 15% 체결 불가
    v = verdict(samples)

    assert v["unfillable_frac"] == pytest.approx(0.15)
    assert v["decision"] == "종료", "스프레드 1bp에 가려 체결 불가가 무시됐다"


def test_unfillable_samples_do_not_sweeten_the_median() -> None:
    """체결 불가 표본은 중앙값에 끼지 않는다.

    부분 체결 표본의 bp는 부분 VWAP에서 나오므로 비용을 크게 과소평가한다 — 비싼 호가를
    못 먹어서 평균 단가가 좋아 보일 뿐이다. 실측(2026-08-13): 얇은 호가에서 24달러 전체
    체결이 91.8bp인 상황이 최우선 호가만으로는 1.0bp로 보고됐다. 그 값이 섞이면
    "스프레드 좋음"이라는 낙관적 오판을 만든다.
    """
    samples = [(2.0, 40.0, True)] * 190 + [(2.0, 1.0, False)] * 10  # 체결불가 5% (게이트 통과)
    v = verdict(samples)

    assert v["unfillable_frac"] == pytest.approx(0.05)
    assert v["median_bp"] == pytest.approx(40.0), "체결 불가 1bp가 중앙값을 끌어내렸다"
    assert v["decision"] == "재검토", "낙관적 중앙값 때문에 '진행'으로 오판했다"


def test_nothing_fillable_in_window_is_not_a_pass() -> None:
    """창 안에서 한 번도 못 채웠으면 스프레드 값과 무관하게 종료다."""
    assert verdict([(2.0, 1.0, False)] * 200)["decision"] == "종료"


def test_only_entry_window_counts() -> None:
    """진입 창(1~24h) 밖 표본은 판정에 안 들어간다."""
    samples = [(0.5, 500.0, True)] * 200 + [(5.0, 10.0, True)] * 200  # 상장 30분은 제외
    v = verdict(samples)

    assert v["n"] == 200
    assert v["median_bp"] == pytest.approx(10.0)


def test_too_few_samples_is_not_a_pass() -> None:
    """표본이 적으면 '진행'이 아니라 '표본부족'이다 — 우연히 좋은 값이 통과하면 안 된다."""
    assert verdict([(2.0, 1.0, True)] * 3)["decision"] == "표본부족"


def test_hourly_profile_buckets_by_hour() -> None:
    """시간대별로 나뉘고 각 구간의 분위수가 나온다."""
    samples = [(1.5, 10.0, True), (1.9, 20.0, True), (3.2, 40.0, True)]
    rows = hourly_profile(samples)

    assert [r["hour"] for r in rows] == [1, 3]
    assert rows[0]["n"] == 2
    assert rows[0]["median_bp"] == pytest.approx(15.0)


def test_reads_recorded_depth_files(tmp_path: Path) -> None:
    """기록 형식을 그대로 읽는다 — 기록기 출력과 분석기 입력이 어긋나면 안 된다."""
    d = tmp_path / "depth20" / "TESTUSDT"
    d.mkdir(parents=True)
    onboard_ms = 1_754_924_400_000
    rec = {
        "recv_ns": (onboard_ms + 2 * 3_600_000) * 1_000_000,  # 상장 2시간 후
        "d": {"b": [["100.0", "10.0"]], "a": [["100.1", "10.0"]]},
    }
    with gzip.open(d / "2026-08-11-17.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rec) + "\n")

    samples = read_depth_samples("TESTUSDT", tmp_path, onboard_ms)

    assert len(samples) == 1
    hours, bp, filled = samples[0]
    assert hours == pytest.approx(2.0)
    assert filled
    assert bp == pytest.approx(5.0, rel=1e-3)  # 스프레드 0.1/100.05 → 편도 ≈ 5bp


@pytest.mark.slow
@pytest.mark.skipif(not (SOCKET / "bookTicker").is_dir(), reason="소켓 데이터 없음")
def test_runs_on_real_collected_data() -> None:
    """기존 58종목 실데이터에서 파이프라인이 끝까지 돈다.

    depth20은 없으므로(수집기가 최우선 호가만 받는다) bookTicker를 1레벨 호가창으로
    읽어 경로만 검증한다. 실효 스프레드 로직 자체는 test_listing_book.py의 실구독이 맡는다.
    """
    from quantbox.listing.analyze import read_bookticker_samples

    sym = next(p.name for p in (SOCKET / "bookTicker").iterdir() if p.is_dir())
    samples = read_bookticker_samples(sym, SOCKET, onboard_ms=0, max_files=1)

    assert samples, f"{sym}에서 표본이 하나도 안 나왔다"
    rows = hourly_profile(samples)
    assert rows and all(r["n"] > 0 for r in rows)
