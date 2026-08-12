"""캘린더 효과 — 크립토는 24시간이지만 참가자는 아니다.

다른 데이터가 필요 없다. 크립토 타임스탬프만으로 검정 가능하다.

## 사전 지정 셀 (다중검정 억제)

요일 7 + 세션 3(아시아/유럽/미국) + 월말주 1 + 미국장 개장·마감 2 = **13셀**만 본다.
격자를 넓히면(요일×시간=168셀) 우연이 반드시 나온다.

## 대조군

각 셀의 수익을 **전체 평균**과 비교한다. 절대 수익이 양수여도 전체 평균과 같으면
신호가 아니다. 오늘 청산 역지정가가 그 함정으로 죽었다.

## 비용

캘린더 전략은 회전이 높다. 요일 전략이면 주 1회 진입·청산 = 왕복 12.6bp/주.
**비용 후 수익만 보고한다.**
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
COST = 12.6e-4


def nw_t(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def report(lab: str, r: np.ndarray, base: float, lag: int, cost: float) -> None:
    """비용 차감 후 초과수익 보고."""
    if len(r) < 30:
        print(f"  {lab:<18}{len(r):>7}  표본부족")
        return
    net = r - cost
    exc = net - base
    p = sps.binomtest(int((exc > 0).sum()), len(exc), 0.5, alternative="greater").pvalue
    print(
        f"  {lab:<18}{len(r):>7}{net.mean() * 1e4:>10.1f}{base * 1e4:>10.1f}"
        f"{exc.mean() * 1e4:>10.1f}{nw_t(exc, lag):>8.2f}{p:>10.2e}"
    )


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        o = int(d["_offset"])
        px[i, o : o + len(d["close"])] = d["close"]

    b = data["BTCUSDT"]
    t0 = float(b["timestamps"][0]) - int(b["_offset"]) * 900_000.0
    ts = t0 + np.arange(end) * 900_000.0
    when = [dt.datetime.utcfromtimestamp(t / 1000) for t in ts]
    print(f"# 심볼 {len(syms)}  {when[0].date()} ~ {when[-1].date()}  비용 {COST * 1e4:.1f}bp")

    # 알트 중앙 수익 (동일가중 바스켓 대용)
    def basket(i: int, j: int) -> float:
        c0, c1 = px[:, i], px[:, j]
        ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0)
        return float(np.median(c1[ok] / c0[ok] - 1.0)) if ok.sum() >= 30 else np.nan

    # ── 1. 요일 (일 단위, 00:00 UTC 진입 → 24h 보유) ──
    day_idx = [i for i in range(0, end - BPD, BPD) if when[i].hour == 0 and when[i].minute == 0]
    rets, dows = [], []
    for i in day_idx:
        r = basket(i, i + BPD)
        if np.isfinite(r):
            rets.append(r)
            dows.append(when[i].weekday())
    rets, dows = np.array(rets), np.array(dows)
    base = float((rets - COST).mean())
    print(f"\n=== 요일 (00:00 UTC 진입, 24h 보유) — {len(rets)}일, 7셀")
    print(f"  {'셀':<18}{'N':>7}{'비용후bp':>10}{'전체평균':>10}{'초과bp':>10}{'t(NW)':>8}{'부호p':>10}")
    names = ["월", "화", "수", "목", "금", "토", "일"]
    for k in range(7):
        report(names[k], rets[dows == k], base, 1, COST)

    # ── 2. 세션 (8시간 구간) ──
    sess = {"아시아 00-08": 0, "유럽 08-16": 8, "미국 16-24": 16}
    print(f"\n=== 세션 (8h 보유) — 3셀")
    print(f"  {'셀':<18}{'N':>7}{'비용후bp':>10}{'전체평균':>10}{'초과bp':>10}{'t(NW)':>8}{'부호p':>10}")
    allr = []
    per_sess: dict[str, list[float]] = {k: [] for k in sess}
    for lab, h0 in sess.items():
        for i in range(0, end - 32, 32):
            if when[i].hour == h0 and when[i].minute == 0:
                r = basket(i, i + 32)
                if np.isfinite(r):
                    per_sess[lab].append(r)
                    allr.append(r)
    base_s = float((np.array(allr) - COST).mean())
    for lab in sess:
        report(lab, np.array(per_sess[lab]), base_s, 1, COST)

    # ── 3. 월말/월초 (일 단위) ──
    print(f"\n=== 월말·월초 (24h 보유) — 2셀")
    print(f"  {'셀':<18}{'N':>7}{'비용후bp':>10}{'전체평균':>10}{'초과bp':>10}{'t(NW)':>8}{'부호p':>10}")
    dom = np.array([when[i].day for i in day_idx if np.isfinite(basket(i, i + BPD))])
    if len(dom) == len(rets):
        report("월말 25일~", rets[dom >= 25], base, 1, COST)
        report("월초 1~5일", rets[dom <= 5], base, 1, COST)

    print("\n# 사전 지정 13셀. Bonferroni 임계 |t| ≈ 3.0 (α=0.05)")


if __name__ == "__main__":
    main()
