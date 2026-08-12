"""실측 펀딩비 — 심볼별 정산 이력에서 시간당 비용률을 얻는다.

기존 모델은 `default_funding_rate_8h` **상수**를 모든 심볼·모든 시점에 적용하고,
`BARS_PER_8H_15M = 32`로 나눠 "8시간마다 정산"을 가정했다. 둘 다 사실과 다르다
(2026-08-12 실측):

1. **주기가 8h가 아니다.** `fapi/v1/fundingInfo` 747심볼 중 4h가 443개, 8h가 302개,
   1h가 2개다. 캐시된 정산 타임스탬프로 보면 4h는 2024년에 도입됐다
   (2020~2023은 8h 100%, 2026년은 4h가 23%). 8h를 가정하면 4h 심볼에서 **펀딩을
   절반만 계산**한다.
2. **수준이 심볼마다 다르다.** 실측 평균이 AXSUSDT −2.58bp에서 VETUSDT +1.50bp까지
   갈린다. 보유 4일(정산 12~24회) 설계에서 이 차이는 거래당 수십 bp이고, 왕복
   통행료보다 크다.

여기서는 심볼별 **시간당 펀딩 비용률**을 실측에서 직접 얻는다. 총 지불액을 총 시간으로
나누므로 정산 주기가 몇 시간이든 자동으로 반영된다 — 주기를 따로 알 필요가 없다.

**부호 정책**: 절대값의 합을 쓴다. 즉 "항상 내는 쪽"으로 가정한다. 기존 모델도 항상
차감했으므로 보수성이 유지되고, 방향을 아는 경우의 수취(숏이 유리한 국면 등)는
반영하지 않는다 — 방향 인지 펀딩은 별도 작업이다.

데이터가 없는 심볼은 config 상수로 폴백한다. 현재 캐시는 34심볼뿐이다.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache

import numpy as np

#: 캐시 파일 패턴 — `{SYMBOL}_funding_rate_6yr.npz` (keys: timestamps[ms], values)
CACHE_GLOB = "data/cache_extended/*_funding_rate_6yr.npz"

#: 15m 봉 하나의 시간.
HOURS_PER_15M_BAR = 0.25

#: 이보다 짧은 이력은 시간당 환산이 불안정해 폴백한다.
MIN_SETTLEMENTS = 100


def _repo_root() -> str:
    from quantbox.research.harness.gitinfo import repo_root

    return str(repo_root())


@lru_cache(maxsize=1)
def _rate_table() -> dict[str, float]:
    """{symbol: 시간당 펀딩 비용률}. 캐시가 없으면 빈 dict.

    총 지불액 ÷ 총 경과시간이므로 정산 주기(4h/8h)가 자동 반영된다.
    """
    table: dict[str, float] = {}
    for path in glob.glob(os.path.join(_repo_root(), CACHE_GLOB)):
        sym = os.path.basename(path).split("_")[0]
        try:
            npz = np.load(path, allow_pickle=False)
            ts, val = npz["timestamps"], npz["values"]
        except Exception:
            continue
        if len(val) < MIN_SETTLEMENTS:
            continue
        hours = (float(ts[-1]) - float(ts[0])) / 3_600_000.0
        if hours <= 0:
            continue
        table[sym] = float(np.sum(np.abs(val))) / hours
    return table


def funding_rate_per_hour(symbol: str, fallback_8h: float) -> float:
    """심볼의 시간당 펀딩 비용률. 실측이 없으면 상수를 8로 나눠 폴백한다.

    `fallback_8h <= 0`은 **호출자가 펀딩을 끈 것**이므로 실측 테이블을 무시하고 0을
    돌려준다. 이 분기가 없으면 `funding=False`로 부른 백테스트에 실측 펀딩이 되살아나
    "펀딩 제외" 비교가 조용히 오염된다.

    Args:
        symbol: API 심볼 (`BTCUSDT`). 복합키를 넘기면 안 된다.
        fallback_8h: 실측이 없을 때 쓸 8시간당 비율 (config 상수). 0 이하면 펀딩 비활성.

    Returns:
        명목가 대비 시간당 비율.
    """
    if fallback_8h <= 0:
        return 0.0
    return _rate_table().get(symbol, fallback_8h / 8.0)


def funding_cost(symbol: str, hold_bars_15m: float, fallback_8h: float) -> float:
    """보유 기간 동안의 펀딩 비용 (명목가 대비 비율).

    Args:
        symbol: API 심볼.
        hold_bars_15m: 보유 기간 (15m 봉 수).
        fallback_8h: 실측이 없을 때 쓸 8시간당 비율.

    Returns:
        차감할 비용. 항상 0 이상이다.
    """
    if hold_bars_15m <= 0 or fallback_8h <= 0:
        return 0.0
    return funding_rate_per_hour(symbol, fallback_8h) * hold_bars_15m * HOURS_PER_15M_BAR


def coverage() -> tuple[int, float]:
    """(실측 심볼 수, 시간당 비용률 중앙값) — 진단·리포트용."""
    t = _rate_table()
    if not t:
        return 0, 0.0
    return len(t), float(np.median(list(t.values())))
