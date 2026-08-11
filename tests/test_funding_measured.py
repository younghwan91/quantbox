"""실측 펀딩비 — 상수 8h 가정을 대체한다.

2026-08-12 발견 2건:
  1. `BARS_PER_8H_15M = 32` 고정 + 독스트링 "Binance는 8시간마다 청구"가 사실과 다르다.
     `fapi/v1/fundingInfo` 747심볼 중 **4h 443개 / 8h 302개 / 1h 2개**이고,
     캐시 타임스탬프 실측상 4h는 2024년 도입이다. 8h를 가정하면 4h 심볼에서 절반만 센다.
  2. 심볼별 수준 차이가 크다 (AXSUSDT −2.58bp ~ VETUSDT +1.50bp).

`funding.py`는 총 지불액÷총 시간으로 **시간당 비용률**을 얻으므로 주기를 몰라도 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantbox.backtest.funding import (
    HOURS_PER_15M_BAR,
    coverage,
    funding_cost,
    funding_rate_per_hour,
)

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_extended"
FALLBACK_8H = 0.0001


def test_missing_symbol_falls_back_to_constant() -> None:
    """실측이 없으면 상수를 8로 나눠 시간당으로 환산한다 — 기존 동작 보존."""
    r = funding_rate_per_hour("__NO_SUCH_SYMBOL__", FALLBACK_8H)
    assert r == pytest.approx(FALLBACK_8H / 8.0)

    # 32봉(=8시간) 보유면 정확히 상수 1회분이어야 한다.
    cost = funding_cost("__NO_SUCH_SYMBOL__", 32, FALLBACK_8H)
    assert cost == pytest.approx(FALLBACK_8H)


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_disabled_funding_beats_measured_table() -> None:
    """`fallback<=0`(호출자가 펀딩을 껐다)이면 실측이 있어도 0이어야 한다.

    이 분기가 없으면 `funding=False` 백테스트에 실측 펀딩이 되살아나
    "펀딩 제외" 비교가 조용히 오염된다 (구현 중 실제로 발생, g6 해상도 테스트가 잡아냄).
    """
    files = sorted(CACHE.glob("*_funding_rate_6yr.npz"))
    if not files:
        pytest.skip("펀딩 캐시 파일 없음")
    sym = files[0].name.split("_")[0]

    assert funding_rate_per_hour(sym, 0.0) == 0.0
    assert funding_cost(sym, 500, 0.0) == 0.0
    # 켜면 다시 살아나야 한다 — 항상 0을 주는 게 아니다.
    assert funding_cost(sym, 500, FALLBACK_8H) > 0


def test_cost_scales_with_hold_and_is_never_negative() -> None:
    """보유가 길수록 비례해 커지고, 부호는 항상 비용(0 이상)이다."""
    a = funding_cost("__NO_SUCH_SYMBOL__", 32, FALLBACK_8H)
    b = funding_cost("__NO_SUCH_SYMBOL__", 96, FALLBACK_8H)
    assert b == pytest.approx(3 * a)
    assert funding_cost("__NO_SUCH_SYMBOL__", 0, FALLBACK_8H) == 0.0
    assert funding_cost("__NO_SUCH_SYMBOL__", -5, FALLBACK_8H) == 0.0


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_measured_symbols_are_loaded() -> None:
    """실측 테이블이 실제로 채워진다 — 조용히 전부 폴백하면 고친 의미가 없다."""
    n, median_rate = coverage()
    if n == 0:
        pytest.skip("펀딩 캐시 파일 없음")
    assert n >= 10, f"실측 심볼이 {n}개뿐 — 로더가 대부분 실패했다"
    # 시간당 비율이므로 8h 상수(1bp)를 8로 나눈 값과 같은 자릿수여야 한다.
    assert 1e-7 < median_rate < 1e-4, f"시간당 비율이 비현실적이다: {median_rate}"


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_measured_differs_from_flat_constant() -> None:
    """실측이 상수와 실제로 다르다 — 같으면 배선이 안 된 것이다."""
    files = sorted(CACHE.glob("*_funding_rate_6yr.npz"))
    if not files:
        pytest.skip("펀딩 캐시 파일 없음")

    diffs = 0
    for f in files[:20]:
        sym = f.name.split("_")[0]
        if funding_rate_per_hour(sym, FALLBACK_8H) != pytest.approx(FALLBACK_8H / 8.0, rel=1e-3):
            diffs += 1
    assert diffs > 0, "모든 심볼이 상수와 동일 — 실측이 반영되지 않았다"


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_rate_matches_hand_computation() -> None:
    """로더가 하는 계산이 총지불액÷총시간과 일치한다."""
    files = sorted(CACHE.glob("*_funding_rate_6yr.npz"))
    if not files:
        pytest.skip("펀딩 캐시 파일 없음")

    f = files[0]
    sym = f.name.split("_")[0]
    npz = np.load(f, allow_pickle=False)
    hours = (float(npz["timestamps"][-1]) - float(npz["timestamps"][0])) / 3_600_000.0
    expected = float(np.sum(np.abs(npz["values"]))) / hours

    assert funding_rate_per_hour(sym, FALLBACK_8H) == pytest.approx(expected)
    # 96봉(=24시간) 보유 비용
    assert funding_cost(sym, 96, FALLBACK_8H) == pytest.approx(expected * 96 * HOURS_PER_15M_BAR)
