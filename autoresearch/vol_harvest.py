"""변동성 수확 — 예측 없이 수익이 나오는가.

오늘 기각한 13개는 **전부 예측 문제**였다. 무엇이 오를지 맞히려 했고 매번 통행료 아래였다.

변동성 수확은 다르다. 동일가중 바스켓을 정기 리밸런싱하면, 개별 자산이 평균회귀할 때
기하평균이 개별 기하평균의 가중합보다 높아진다. **어느 것이 오를지 몰라도 된다.**
크립토는 변동성이 극단적이라 이 효과가 클 수 있다.

  수확량 ≈ (개별 분산 평균 − 포트폴리오 분산) / 2

## 결정적 반론

알트는 하락 드리프트가 강하다(오늘 측정: 4시간에 -285bp). 롱온리면 수확이 드리프트에
먹힌다. 그래서 **반드시 무리밸런싱 대조군**과 비교해 초과분만 본다.

## 비용

리밸런싱마다 왕복 12.6bp × 회전율. 빈도가 수확량과 비용을 동시에 결정하므로
그 균형점을 격자로 찾되, **최량만 보지 않고 분포를 본다**.
"""

from __future__ import annotations

import numpy as np

from quantbox.research.harness.data import MarketData

BPD = 96
COST = 12.6e-4


def paths(px: np.ndarray, start: int, end_: int, k: int, rebal: int, cost: float) -> tuple[float, float]:
    """(리밸런싱 기하수익, 무리밸런싱 기하수익) — 같은 종목 집합에서."""
    # 시작 시점에 살아있는 종목 중 무작위가 아니라 **가장 오래 살아남을** 것을 고르면
    # 생존편향이다. 시작 시점 정보만으로 고른다: 그때 거래되는 종목 전부에서 앞 k개.
    alive = np.flatnonzero(np.isfinite(px[:, start]) & (px[:, start] > 0))
    if len(alive) < k:
        return np.nan, np.nan
    sel = alive[:k]

    # 무리밸런싱: 초기 동일가중 후 방치
    p0, p1 = px[sel, start], px[sel, end_]
    ok = np.isfinite(p1) & (p1 > 0)
    if ok.sum() < k // 2:
        return np.nan, np.nan
    # 죽은 종목은 마지막 관측가로 청산 처리
    ratio = np.ones(k)
    ratio[ok] = p1[ok] / p0[ok]
    for i in np.flatnonzero(~ok):
        row = px[sel[i], start:end_]
        good = np.isfinite(row) & (row > 0)
        ratio[i] = row[good][-1] / p0[i] if good.any() else 0.0
    buyhold = float(np.mean(ratio))

    # 리밸런싱: rebal 봉마다 동일가중 복원
    eq = 1.0
    t = start
    while t < end_:
        u = min(t + rebal, end_)
        a, b = px[sel, t], px[sel, u]
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if m.sum() >= k // 2:
            eq *= float(np.mean(b[m] / a[m])) * (1.0 - cost)
        t = u
    return eq, buyhold


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        o = int(d["_offset"])
        px[i, o : o + len(d["close"])] = d["close"]
    yrs = end / (BPD * 365)
    print(f"# 심볼 {len(syms)}  {yrs:.2f}년  비용 {COST * 1e4:.1f}bp/리밸런싱")
    print()
    print(
        f"{'종목수':>6}{'리밸런싱':>10}{'구간N':>7}{'리밸 연%':>10}{'무리밸 연%':>12}{'초과%p':>9}{'초과>0 비율':>12}"
    )
    print("-" * 68)

    # 1년 구간을 겹치지 않게 잘라 반복 — 시작 시점 편향 완화
    seg = BPD * 365
    starts = list(range(BPD * 30, end - seg, seg // 2))

    for k in (10, 30, 60):
        for rb_d, lab in ((1, "1일"), (7, "7일"), (30, "30일")):
            rb = rb_d * BPD
            re_, bh_ = [], []
            for s0 in starts:
                a, b = paths(px, s0, s0 + seg, k, rb, COST)
                if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                    re_.append(a - 1.0)
                    bh_.append(b - 1.0)
            if len(re_) < 5:
                continue
            A, B = np.array(re_), np.array(bh_)
            d_ = A - B
            print(
                f"{k:>6}{lab:>10}{len(A):>7}{np.median(A) * 100:>10.1f}"
                f"{np.median(B) * 100:>12.1f}{np.median(d_) * 100:>9.1f}{(d_ > 0).mean() * 100:>11.0f}%"
            )
        print()

    print("# 초과가 0 근처면 수확 효과가 비용에 먹힌 것이다.")
    print("# 무리밸런싱도 같은 종목·같은 구간이므로 종목 선택 편향은 상쇄된다.")


if __name__ == "__main__":
    main()
