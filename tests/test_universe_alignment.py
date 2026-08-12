"""타임스탬프 기반 정렬 — 폐지 종목을 섞어도 어긋나지 않는가.

구 캐시에는 타임스탬프가 없어 정렬이 `offset = max_bars - len`, 즉
**모든 심볼이 같은 시점에 끝난다**는 가정으로 위치 기반이었다.
상장폐지 종목에는 성립하지 않는다 — 2024년에 끝난 계열이 현재까지 거래된 것처럼
배치되어 BTC 게이트·슬롯 경합·일손실 경계가 전부 잘못된 시각으로 계산된다.

조용히 틀리는 종류라 테스트로 고정한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.research.harness.data import BAR_MS, MarketData

T0 = 1_600_000_000_000  # 임의 기준 시각 (15m 격자에 맞춰 사용)


def _series(start_bar: int, n: int, *, with_ts: bool = True) -> dict:
    """``start_bar``번째 봉부터 ``n``개짜리 계열."""
    d = {
        "close": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "volume": np.full(n, 1.0),
    }
    if with_ts:
        base = T0 - (T0 % BAR_MS)
        d["timestamps"] = base + (start_bar + np.arange(n)) * BAR_MS
    return d


def test_delisted_symbol_keeps_its_own_end_time() -> None:
    """일찍 끝난 계열이 '현재까지 거래된 것'으로 밀리지 않는다.

    live: 0~999봉, delisted: 0~499봉. 둘 다 같은 시점에 시작했으므로 offset은 0이어야
    한다. 우측 정렬이면 delisted가 offset=500으로 밀려 **시작 시각이 500봉 늦은 것처럼**
    보인다 — 폐지 시점이 아니라 상장 시점이 조작되는 셈이다.
    """
    data = {"LIVE": _series(0, 1000), "DEAD": _series(0, 500)}
    MarketData._align(data)

    assert data["LIVE"]["_offset"] == 0
    assert data["DEAD"]["_offset"] == 0, "폐지 종목이 뒤로 밀렸다 — 시각이 어긋난다"


def test_late_listing_gets_positive_offset() -> None:
    """늦게 상장한 종목은 그만큼 오프셋을 갖는다."""
    data = {"OLD": _series(0, 1000), "NEW": _series(300, 700)}
    MarketData._align(data)

    assert data["OLD"]["_offset"] == 0
    assert data["NEW"]["_offset"] == 300


def test_delisted_and_late_listing_together() -> None:
    """상장/폐지 시점이 제각각이어도 각자의 실제 시각에 놓인다."""
    data = {
        "OLD_LIVE": _series(0, 1000),
        "OLD_DEAD": _series(0, 400),
        "NEW_LIVE": _series(600, 400),
        "MID_DEAD": _series(200, 300),
    }
    MarketData._align(data)

    assert [data[k]["_offset"] for k in ("OLD_LIVE", "OLD_DEAD", "NEW_LIVE", "MID_DEAD")] == [0, 0, 600, 200]


def test_legacy_cache_falls_back_to_right_align() -> None:
    """타임스탬프가 없으면 기존 동작을 유지한다 — 구 캐시 결과가 바뀌면 안 된다."""
    data = {"A": _series(0, 1000, with_ts=False), "B": _series(0, 400, with_ts=False)}
    MarketData._align(data)

    assert data["A"]["_offset"] == 0
    assert data["B"]["_offset"] == 600  # 1000 - 400


def test_offset_must_land_on_bar_grid() -> None:
    """격자에서 벗어난 시각은 조용히 반올림하지 않고 거부한다."""
    a = _series(0, 100)
    b = _series(0, 100)
    b["timestamps"] = b["timestamps"] + BAR_MS // 3
    with pytest.raises(ValueError, match="격자"):
        MarketData._align({"A": a, "B": b})


def test_load_preserves_timestamps_for_alignment(tmp_path) -> None:
    """`load()`를 거쳐도 시각 정렬이 살아있어야 한다.

    회귀 방어: 위 테스트들은 전부 `_align`을 직접 호출해 타임스탬프를 손수 넣었다.
    그래서 `load()`가 `timestamps`를 버려 **우측 정렬로 조용히 떨어지던 버그**를
    아무도 못 잡았다 (2026-08-12 발견). 경로 전체를 지나가는 테스트가 있어야 한다.
    """
    for name, start_bar, n in (("OLD", 0, 900), ("DEAD", 0, 600), ("NEW", 300, 600)):
        np.savez(tmp_path / f"{name}.npz", **_series(start_bar, n))

    md = MarketData.load(cache_dir=tmp_path, pattern="*.npz", include_market_ref=False)

    assert "timestamps" in md["OLD"], "load()가 타임스탬프를 버렸다 — 정렬이 우측으로 떨어진다"
    # 우측 정렬이었다면 DEAD는 300(=900-600)으로 밀린다. 시각 정렬이면 0이다.
    assert md["DEAD"]["_offset"] == 0, "폐지 종목이 뒤로 밀렸다 — 우측 정렬로 떨어졌다"
    assert md["OLD"]["_offset"] == 0
    assert md["NEW"]["_offset"] == 300


def test_contiguity_report_flags_internal_gaps() -> None:
    """내부 결측을 보고한다 — 위치 기반 인덱싱은 연속성을 전제한다."""
    good = _series(0, 100)
    holed = _series(0, 100)
    holed["timestamps"] = np.delete(holed["timestamps"], 50)
    holed["close"] = np.delete(holed["close"], 50)

    rep = MarketData._contiguity_report({"GOOD": good, "HOLED": holed})
    assert "GOOD" not in rep
    assert rep["HOLED"] == 1
