"""G6 파리티: 트레일의 봉-내 순서 가정을 드러낸다.

라이브는 Binance 서버측 `TRAILING_STOP_MARKET`이 **틱 단위**로 고점을 추적한다
(`brackets.py:374`). 백테스트 `sim_drive`는 봉의 high/low만 본다.

문제는 같은 봉에서 **고점 갱신과 트레일 발동이 동시에** 일어나는 경우다. 현재 구현은
`high[j]`로 peak을 올린 뒤 그 peak 기준 트레일선을 `low[j]`와 비교한다 — 즉 봉 안에서
**고점이 저점보다 먼저** 왔다고 가정한다. 실제 순서가 반대였다면 그 시점의 고점은 더
낮았고, 트레일선도 더 낮아 체결가가 달랐거나 아예 발동하지 않았다.

봉 데이터로는 순서를 알 수 없다. 그래서 이 갭은 **없앨 수 없고 경계만 잴 수 있다**:
  - `trail_peak_lag=0` (현행): 고점 우선 = 낙관 경계
  - `trail_peak_lag=1`: 직전 봉까지의 고점만 사용 = 비관 경계
두 값의 차이가 이 가정이 만드는 불확실성 폭이다.
"""

from __future__ import annotations

import numpy as np

from quantbox.backtest.sim_core import KIND_ACTIVE, sim_drive


def _one_shot(close: np.ndarray, high: np.ndarray, low: np.ndarray, **kw) -> tuple:
    """봉 0에서 LONG 1건만 진입시키는 최소 구동."""
    n = len(close)
    bar_kind = np.full(n, KIND_ACTIVE, dtype=np.int8)
    entry_ok = np.zeros(n, dtype=bool)
    entry_ok[0] = True
    direction = np.zeros(n, dtype=np.int8)
    direction[0] = 1
    return sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,  # min_duration
        0,  # max_duration
        0.05,  # sl
        0.50,  # tp (닿지 않게)
        0.01,  # trail_activate
        0.01,  # trail_distance
        10,  # max_hold
        0.0,  # cost
        0,  # warmup
        **kw,
    )


def test_same_bar_peak_and_trigger_is_optimistic_by_default() -> None:
    """고점 갱신과 트레일 발동이 같은 봉 → 현행은 그 봉의 고점을 쓴다(낙관)."""
    # 봉1에서 +5% 고점 후 되돌림. 트레일선 = peak - 1% = +4%.
    close = np.array([100.0, 100.0, 100.0])
    high = np.array([100.0, 105.0, 100.0])
    low = np.array([100.0, 103.0, 100.0])  # 103 <= 104 → 같은 봉에서 발동

    pnl, ent, ext, _ = _one_shot(close, high, low)
    assert len(pnl) == 1
    assert pnl[0] == np.float64(0.05 - 0.01), "봉1 고점(+5%) 기준 트레일이 아니다"
    assert ext[0] == 1


def test_peak_lag_gives_pessimistic_bound() -> None:
    """`trail_peak_lag=1`은 직전 봉까지의 고점만 쓴다 — 같은 봉 고점을 신뢰하지 않는다.

    봉1의 고점을 못 쓰면 진입 시점 고점(0%)뿐이라 트레일이 활성화되지 않고,
    SL도 안 닿았으므로 트레이드는 max-hold까지 간다.
    """
    close = np.array([100.0, 100.0, 100.0])
    high = np.array([100.0, 105.0, 100.0])
    low = np.array([100.0, 103.0, 100.0])

    _, _, ext_opt, _ = _one_shot(close, high, low)
    _, _, ext_pess, _ = _one_shot(close, high, low, trail_peak_lag=1)

    # 이 케이스에서 관측되는 차이는 **청산 시점**이다. 봉1의 고점을 신뢰하지 않으므로
    # 트레일은 봉2에 가서야 발동한다. (PnL은 그 사이 고점이 안 바뀌어 우연히 같다 —
    # 실데이터에서는 이후 봉의 가격이 달라지므로 PnL도 갈린다.)
    assert ext_pess[0] > ext_opt[0], "지연을 켜도 청산 시점이 같다 — 파라미터가 무시됐다"


def test_peak_lag_zero_is_bit_identical_to_current() -> None:
    """기본값(0)은 현행과 비트 동일 — 기존 숫자를 건드리지 않는다."""
    rng_free_close = np.array([100.0, 101.0, 103.0, 102.0, 99.0, 104.0, 101.0])
    high = rng_free_close + 1.0
    low = rng_free_close - 1.0

    a = _one_shot(rng_free_close, high, low)
    b = _one_shot(rng_free_close, high, low, trail_peak_lag=0)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_lag_never_exits_earlier_than_optimistic() -> None:
    """비관 경계는 낙관 경계보다 먼저 청산될 수 없다 — 경계의 방향성 자체를 고정한다."""
    close = np.array([100.0, 102.0, 106.0, 103.0, 108.0, 104.0, 100.0, 101.0])
    high = close + 2.0
    low = close - 2.0

    _, _, ext_opt, _ = _one_shot(close, high, low)
    _, _, ext_pess, _ = _one_shot(close, high, low, trail_peak_lag=1)
    assert ext_pess[0] >= ext_opt[0]


def test_real_data_bounds_differ_measurably() -> None:
    """실데이터에서 두 경계가 실제로 갈린다 — 갭이 이론적 존재가 아님을 고정한다."""
    import pathlib

    import pytest

    cache = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache_extended"
    if not cache.is_dir():
        pytest.skip("확장 캐시 없음")

    from quantbox.backtest.sim_core import run_ma_cluster_live
    from quantbox.research.harness import MarketData

    md = MarketData.load(["BTCUSDT", "ETHUSDT"])
    opt = run_ma_cluster_live(md.data)
    pess = run_ma_cluster_live(md.data, trail_peak_lag=1)

    assert len(opt) > 0
    changed = sum(1 for a, b in zip(opt, pess, strict=False) if a[:3] != b[:3] or a[3] != b[3])
    assert changed > 0, "실데이터에서 두 경계가 완전히 같다 — 측정이 공허하다"
