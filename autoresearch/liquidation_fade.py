"""청산 캐스케이드 역지정가 — 통행료를 내지 않고 받는 쪽.

오늘 다섯 계열이 전부 같은 이유로 죽었다: **엣지 < 통행료**. 방향성 단타, 마켓메이킹,
펀딩 캐리 양방향, 캐시앤캐리, 신규 상장 숏. 다섯 번 같은 사인이면 구조다.

그래서 통행료를 **받는** 쪽만 본다.

## 앞선 마켓메이킹 측정과 무엇이 다른가

앞서 잰 것은 **호가 최전선**의 지정가였고, 실현 반스프레드가 ≈0이었다 — 역선택이
전액을 먹었다. 상대가 나보다 정보가 많을 때 체결되기 때문이다.

호가에서 **멀리** 떨어진 지정가는 성격이 다르다. 평소엔 체결되지 않고, 가격이 급격히
움직일 때만 체결된다. 그리고 크립토에서 그런 급락의 상당 부분은 **강제 청산**이다 —
파는 쪽이 정보를 가져서가 아니라 증거금이 부족해 **가격을 보지 않고 던진다**.
정보 없는 매도자를 상대하는 것이므로 역선택의 부호가 뒤집힌다.

## 측정

  각 봉에서 직전 종가 대비 -D% 지정가 매수를 걸어둔다고 본다.
  그 봉의 **저가**가 그 수준을 건드리면 체결로 간주하고, 이후 H봉 뒤 종가에 청산한다.

  체결가는 지정가이므로 **스프레드를 지불하지 않는다**. 오히려 메이커 수수료(2bp)만
  낸다. 청산은 테이커(5bp)로 보수적으로 잡는다.

한계: 봉 저가가 지정가를 건드렸다고 반드시 체결되는 건 아니다(큐 우선순위).
그래서 D를 크게 잡을수록 체결 가정이 안전해진다 — 급락이 지정가를 관통하기 때문이다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
MAKER_FEE = 2e-4  # 진입 (지정가)
TAKER_FEE = 5e-4  # 청산 (시장가, 보수적)


def simulate(d: dict, depth: float, hold: int) -> np.ndarray:
    """직전 종가 대비 -depth 지정가 매수 → hold봉 뒤 종가 청산. 순수익 배열."""
    c, lo = d["close"], d["low"]
    n = len(c)
    if n < hold + 10:
        return np.array([])

    ref = c[:-1]  # 직전 종가
    limit = ref * (1.0 - depth)
    filled = lo[1:] <= limit  # 이 봉의 저가가 지정가를 관통 → 체결
    idx = np.flatnonzero(filled)
    idx = idx[idx + hold < n - 1]
    if len(idx) == 0:
        return np.array([])

    entry = limit[idx]
    exit_px = c[idx + 1 + hold]
    return exit_px / entry - 1.0 - MAKER_FEE - TAKER_FEE


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    # 유동성 상위만 — 지정가 체결 가정이 성립하려면 실제 거래가 있어야 한다
    liq = {}
    for s, d in data.items():
        v, c = d["volume"], d["close"]
        if len(v) < BPD * 90:
            continue
        if float(np.median(v[-BPD * 30 :] * c[-BPD * 30 :])) * BPD >= 1e7:
            liq[s] = d
    print(f"# 유동성 통과 {len(liq)}/{len(data)}심볼")
    print()
    print(f"{'깊이':>6}{'보유':>6}{'체결N':>9}{'심볼':>6}{'EV_bp':>9}{'t':>7}{'양수심볼%':>10}{'부호p':>10}")
    print("-" * 63)

    for depth in (0.01, 0.02, 0.03, 0.05):
        for hold in (4, 16, 96):  # 1h, 4h, 1일 (15m 봉)
            per: dict[str, np.ndarray] = {}
            for s, d in liq.items():
                r = simulate(d, depth, hold)
                if len(r) >= 5:
                    per[s] = r
            if len(per) < 30:
                continue
            allr = np.concatenate(list(per.values()))
            means = np.array([v.mean() for v in per.values()])
            pos = int((means > 0).sum())
            p = sps.binomtest(pos, len(means), 0.5, alternative="greater").pvalue
            ev = float(np.mean(allr))
            t = ev / (float(np.std(allr, ddof=1)) / np.sqrt(len(allr)))
            print(
                f"{depth:>6.0%}{hold:>6}{len(allr):>9}{len(means):>6}{ev * 1e4:>9.2f}"
                f"{t:>7.2f}{pos / len(means) * 100:>9.1f}%{p:>10.2e}"
            )


if __name__ == "__main__":
    main()
