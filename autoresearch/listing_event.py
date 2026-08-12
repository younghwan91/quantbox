"""신규 상장 이벤트 — 상장 직후 거동에 체계적 패턴이 있는가.

이 방향을 1순위로 둔 이유는 **통행료 대비 진폭**이다. 왕복 12.6bp를 넘어야 하는데,
상장 초기 구간은 퍼센트 급으로 움직인다. 진폭이 통행료의 100배면 비용 가정의
오차가 결론을 바꾸지 못한다 — 오늘 죽인 전략들이 전부 그 반대였다.

데이터는 이미 있다: `cache_universe` 760심볼, 타임스탬프 보유, 폐지 종목 포함
(생존편향 없음). 각 심볼의 첫 봉이 상장 시점이다.

검정: 심볼 단위 부호검정. 상장은 심볼당 1회뿐이므로 심볼 = 독립 관측이고,
이 구조에서는 부호검정이 특히 적절하다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BARS_PER_DAY = 96  # 15m
COST = 12.6e-4  # 왕복 (단일 다리, 실측)
HORIZONS_D = (1, 3, 7, 14, 30, 60)


def main() -> None:
    md = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz")
    data = md.data
    print(f"# 심볼 {len(data)}")

    # 상장 시각 = 첫 봉. 유니버스 시작(2020)부터 있는 심볼은 '신규 상장'이 아니다.
    t0_global = min(int(d["_offset"]) for d in data.values())
    rows = []
    for s, d in data.items():
        off = int(d["_offset"])
        if off <= t0_global + BARS_PER_DAY:  # 처음부터 있던 종목 제외
            continue
        c = d["close"]
        if len(c) < BARS_PER_DAY * 70:
            continue
        rows.append((s, off, c))
    print(f"# 신규 상장(중도 진입) {len(rows)}심볼")

    # 상장 연도 분포 — 국면 편중 확인
    yr = {}
    for _s, off, _c in rows:
        y = 2020 + int(off / (BARS_PER_DAY * 365))
        yr[y] = yr.get(y, 0) + 1
    print("# 상장연도(근사):", dict(sorted(yr.items())))

    print()
    print(f"{'지평':>6}{'심볼':>6}{'중앙수익%':>11}{'평균%':>9}{'양수%':>8}{'부호p':>11}{'숏 순EV_bp':>12}")
    print("-" * 64)
    for h in HORIZONS_D:
        n = h * BARS_PER_DAY
        r = []
        for _s, _off, c in rows:
            if len(c) > n and c[0] > 0:
                r.append(c[n] / c[0] - 1.0)
        if len(r) < 20:
            continue
        a = np.array(r)
        pos = int((a > 0).sum())
        # 양방향 검정 — 오르는지 내리는지 모르므로
        p = sps.binomtest(pos, len(a), 0.5, alternative="two-sided").pvalue
        # 숏 진입 시 순손익 (하락이면 이익)
        short_ev = (-float(np.median(a)) - COST) * 1e4
        print(
            f"{h:>5}일{len(a):>6}{np.median(a) * 100:>11.2f}{a.mean() * 100:>9.2f}"
            f"{pos / len(a) * 100:>7.1f}%{p:>11.2e}{short_ev:>12.1f}"
        )

    print()
    print("# 숏 순EV = (−중앙수익 − 통행료 12.6bp). 양수면 상장 후 숏이 유효.")
    print("# 주의: 상장 초기는 대여 물량·증거금 제한이 걸릴 수 있다(미확인).")


if __name__ == "__main__":
    main()
