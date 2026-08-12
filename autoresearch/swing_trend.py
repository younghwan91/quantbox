"""시계열 추세추종 (스윙) — 지금까지 안 재본 계열.

오늘 장기 지평에서 잰 것은 **횡단면**(종목 간 순위)이었다. 스윙은 보통
**시계열 추세추종**이다 — "이 종목이 자기 이동평균 위인가". 다른 신호이고
아직 검정하지 않았다.

  롱온리   추세 위일 때만 보유. 알트 하락 드리프트를 피하는 게 목적.
  롱숏     추세 위면 롱, 아래면 숏. 시장 방향에 양방향 노출.

두 변형을 모두 잰다. 회전이 낮아 통행료 12.6bp가 희석되는 게 이 계열의 강점이다.

## 대조군을 반드시 둔다

오늘 청산 역지정가에서 "초과수익"이 알고 보니 진입 할인의 산술 결과였다.
같은 실수를 막기 위해 **단순 보유(buy&hold)** 를 대조군으로 함께 낸다.
추세추종이 단순 보유를 못 이기면 신호가 아니라 베타다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
COST = 12.6e-4


def sma(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        c = np.concatenate([[0.0], np.cumsum(x)])
        out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def run(close: np.ndarray, win: int, long_short: bool) -> tuple[float, int, float]:
    """(총로그수익, 거래수, 단순보유 로그수익). 신호는 직전 봉까지로 판정 — 룩어헤드 없음."""
    m = sma(close, win)
    ok = np.isfinite(m)
    if ok.sum() < 100:
        return 0.0, 0, 0.0
    i0 = int(np.argmax(ok))
    c = close[i0:]
    mm = m[i0:]
    # t 시점 포지션은 t-1 종가와 t-1 이동평균으로 정한다
    sig = np.zeros(len(c))
    above = c[:-1] > mm[:-1]
    sig[1:] = np.where(above, 1.0, -1.0 if long_short else 0.0)

    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1.0
    pos_ret = sig * r
    flips = int(np.sum(np.abs(np.diff(sig)) > 0))
    gross = float(np.sum(np.log1p(np.clip(pos_ret, -0.99, None))))
    return gross - flips * COST, flips, float(np.sum(np.log1p(np.clip(r, -0.99, None))))


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    print(f"# 심볼 {len(data)}  타임라인 {end / (BPD * 365):.2f}년  왕복 {COST * 1e4:.1f}bp")
    print()
    print(f"{'MA기간':>8}{'방식':>8}{'심볼':>6}{'연%중앙':>10}{'단순보유':>10}{'초과':>9}{'승리심볼':>9}{'부호p':>10}")
    print("-" * 70)

    for days in (10, 20, 50, 100, 200):
        win = days * BPD
        for ls, lab in ((False, "롱온리"), (True, "롱숏")):
            strat, bh = [], []
            for _s, d in data.items():
                c = d["close"]
                if len(c) < win + BPD * 180 or c[0] <= 0:
                    continue
                g, _n, b = run(c, win, ls)
                if g == 0.0 and b == 0.0:
                    continue
                yrs = len(c) / (BPD * 365)
                strat.append(np.expm1(g / yrs))
                bh.append(np.expm1(b / yrs))
            if len(strat) < 50:
                continue
            a, b_ = np.array(strat), np.array(bh)
            diff = a - b_
            p = sps.binomtest(int((diff > 0).sum()), len(diff), 0.5, alternative="greater").pvalue
            print(
                f"{days:>6}일{lab:>8}{len(a):>6}{np.median(a) * 100:>10.1f}{np.median(b_) * 100:>10.1f}"
                f"{np.median(diff) * 100:>9.1f}{(diff > 0).mean() * 100:>8.1f}%{p:>10.2e}"
            )
        print()


if __name__ == "__main__":
    main()
