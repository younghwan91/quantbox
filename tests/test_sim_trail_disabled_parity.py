"""백테스트가 라이브의 `ma_cluster_trail_enabled=False`를 따르는가.

2026-08-10 `d3f14bb`이 `ma_cluster_trail_enabled`를 신설하고 라이브 4개 경로
(진입·지연배치·실패복구·리컨실)에 배선했다. 그런데 `quantbox/backtest/` 어디에도
`is_trail_enabled` 호출이 없어, **라이브는 트레일을 안 거는데 백테스트는 걸고 있었다**
(2026-08-12 발견). 트레일 수익의 95.4%가 봉 집계 아티팩트라는 측정 위에서
플래그를 껐는데 정작 측정 도구가 그 플래그를 무시하면 이후 결론이 전부 어긋난다.

함정 하나 더: `sim_drive`의 발동 조건은 `eff_peak >= trail_activate`라
`trail_activate=0`은 트레일을 끄는 게 아니라 **즉시 발동**시킨다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantbox.backtest.sim_core import (
    EXIT_TRAIL,
    KIND_ACTIVE,
    TRAIL_OFF,
    run_ma_cluster_live,
    run_ma_cluster_live_15m_exits,
    sim_drive,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"


def _ramp(n: int = 400) -> dict:
    """트레일이 반드시 걸리는 계열 — 올랐다가 되돌린다."""
    t = np.arange(n, dtype=np.float64)
    close = 100.0 * (1.0 + 0.004 * np.sin(t / 9.0) + 0.0015 * t)
    return {
        "close": close,
        "high": close * 1.004,
        "low": close * 0.996,
        "volume": np.full(n, 1.0),
        "_offset": 0,
    }


def test_trail_activate_zero_does_not_disable() -> None:
    """0은 끄는 값이 아니다 — 이 오해가 이 버그의 두 번째 함정이었다."""
    d = _ramp()
    n = len(d["close"])
    kw = dict(
        bar_kind=np.full(n, KIND_ACTIVE, dtype=np.int8),
        entry_ok=np.zeros(n, dtype=np.bool_),
        direction=np.zeros(n, dtype=np.int8),
    )
    kw["entry_ok"][50] = True
    kw["direction"][50] = 1

    _, _, _, _, rsn_zero = sim_drive(
        d["close"],
        d["high"],
        d["low"],
        kw["bar_kind"],
        kw["entry_ok"],
        kw["direction"],
        1,
        0,
        0.02,
        0.08,
        0.0,
        0.005,
        24,
        0.0008,
        10,
    )
    _, _, _, _, rsn_off = sim_drive(
        d["close"],
        d["high"],
        d["low"],
        kw["bar_kind"],
        kw["entry_ok"],
        kw["direction"],
        1,
        0,
        0.02,
        0.08,
        TRAIL_OFF,
        0.005,
        24,
        0.0008,
        10,
    )

    assert EXIT_TRAIL in list(rsn_zero), "trail_activate=0이 트레일을 껐다고 오해했다"
    assert EXIT_TRAIL not in list(rsn_off), "TRAIL_OFF인데 트레일로 청산됐다"


def _real_data() -> dict:
    """실데이터만 쓴다 (프로젝트 규칙: No Mock Data). SMA200@4h 워밍업 때문에도 필요하다."""
    from quantbox.research.harness import MarketData

    return MarketData.load(["BTCUSDT", "ETHUSDT", "SOLUSDT"]).data


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
@pytest.mark.parametrize("runner", [run_ma_cluster_live, run_ma_cluster_live_15m_exits])
def test_backtest_honors_trail_enabled_flag(monkeypatch, runner) -> None:
    """설정에서 트레일을 끄면 백테스트 결과가 달라져야 한다 — 무시하면 안 된다."""
    data = _real_data()

    monkeypatch.setenv("VS_MA_CLUSTER_TRAIL_ENABLED", "true")
    on = runner(data)
    monkeypatch.setenv("VS_MA_CLUSTER_TRAIL_ENABLED", "false")
    off = runner(data)

    assert on, "테스트 계열에서 거래가 안 나왔다 — 케이스가 무의미하다"
    assert [t[3] for t in on] != [t[3] for t in off], "트레일 플래그가 결과를 안 바꿨다 — 무시되고 있다"


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_explicit_trail_activate_overrides_flag(monkeypatch) -> None:
    """명시 인자는 플래그보다 우선한다 — 스윕이 플래그에 막히면 안 된다."""
    data = _real_data()
    monkeypatch.setenv("VS_MA_CLUSTER_TRAIL_ENABLED", "false")

    forced = run_ma_cluster_live(data, trail_activate=0.01)
    defaulted = run_ma_cluster_live(data)

    assert [t[3] for t in forced] != [t[3] for t in defaulted], "명시 trail_activate가 무시됐다"
