"""잔차 상대가치 — 공통 요인을 명시적으로 제거한 뒤 남는 것.

단순 횡단면 순위(리버설·모멘텀)는 죽었다. 이유는 순위에 **시장 베타가 섞여** 있었기
때문일 수 있다. 많이 빠진 종목은 그냥 베타가 큰 종목일 뿐이다.

여기서는 각 종목의 수익을 시장(BTC 또는 동일가중 지수)에 회귀해 **잔차만** 남긴다.
잔차 순위로 롱숏하면 베타 노출이 제거되고 특이 성분만 거래하게 된다.

## 룩어헤드 차단

베타는 **진입 시점까지의 과거 창**으로만 추정한다. 전체 표본 회귀는 미래를 쓴다.

## 비용

방금 실측 정정한 **왕복 4bp**(양쪽 지정가, 유동성 상위). 양다리라 8bp.
보유가 며칠이라 지정가 체결이 현실적이다(최우선호가 5분 체결률 98.5%).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
COST_LEG = 4.0e-4  # 실측 정정: 양쪽 지정가 왕복
TOP_LIQ = 200
K = 40


def nw_t(x: np.ndarray, lag: int) -> float:
    n = len(x)
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    dv = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        o = int(d["_offset"])
        n = len(d["close"])
        px[i, o : o + n] = d["close"]
        dv[i, o : o + n] = d["volume"] * d["close"]
    print(f"# 심볼 {len(syms)}  비용 다리당 왕복 {COST_LEG * 1e4:.1f}bp (실측 정정)")

    # 일 단위 격자
    di = np.arange(0, end - BPD, BPD)
    P = px[:, di]
    R = np.full(P.shape, np.nan)
    R[:, 1:] = P[:, 1:] / P[:, :-1] - 1.0
    mkt = np.nanmedian(R, axis=0)  # 동일가중 시장 수익

    print()
    print(
        f"{'베타창':>7}{'신호창':>7}{'보유':>6}{'N':>7}{'EV_bp':>9}{'t(NW)':>8}{'최근EV':>9}{'최근t':>8}{'부호p':>10}"
    )
    print("-" * 71)

    for beta_w in (60, 120):
        for sig_d, hold_d in ((3, 3), (7, 7)):
            rets, times = [], []
            for t in range(beta_w + 5, len(di) - hold_d):
                # 유동성 상위만 (지정가·저스프레드가 성립하는 범위)
                v = np.nanmedian(dv[:, max(0, di[t] - BPD * 30) : di[t]], axis=1)
                live = np.flatnonzero(np.isfinite(P[:, t]) & (P[:, t] > 0) & np.isfinite(v))
                if len(live) < TOP_LIQ // 2:
                    continue
                pool = live[np.argsort(-v[live])[:TOP_LIQ]]

                # 베타: 과거 beta_w일 회귀 (룩어헤드 없음)
                win = slice(t - beta_w, t)
                m = mkt[win]
                mv = m - np.nanmean(m)
                den = float(np.nansum(mv * mv))
                if den <= 0:
                    continue
                keep, resid = [], []
                for i in pool:
                    y = R[i, win]
                    ok = np.isfinite(y) & np.isfinite(m)
                    if ok.sum() < beta_w * 0.7:
                        continue
                    beta = float(np.nansum((y[ok] - np.nanmean(y[ok])) * mv[ok]) / max(np.nansum(mv[ok] ** 2), 1e-12))
                    # 신호: 최근 sig_d일 잔차 누적
                    ys = R[i, t - sig_d : t]
                    ms = mkt[t - sig_d : t]
                    if not (np.isfinite(ys).all() and np.isfinite(ms).all()):
                        continue
                    keep.append(i)
                    resid.append(float(np.sum(ys - beta * ms)))
                if len(keep) < 2 * K + 10:
                    continue

                keep = np.array(keep)
                resid = np.array(resid)
                fwd = P[keep, t + hold_d] / P[keep, t] - 1.0
                ok = np.isfinite(fwd)
                if ok.sum() < 2 * K + 10:
                    continue
                kk, ff = resid[ok], fwd[ok]
                o = np.argsort(kk)
                # 잔차가 낮은(과매도) 쪽 롱, 높은 쪽 숏 — 잔차 평균회귀
                rets.append(float(ff[o[:K]].mean() - ff[o[-K:]].mean()) - 2 * COST_LEG)
                times.append(di[t])

            if len(rets) < 200:
                continue
            r = np.array(rets)
            tt = np.array(times)
            rl = r[tt >= end * 0.75]
            p = sps.binomtest(int((r > 0).sum()), len(r), 0.5, alternative="greater").pvalue
            print(
                f"{beta_w:>7}{sig_d:>7}{hold_d:>6}{len(r):>7}{r.mean() * 1e4:>9.1f}"
                f"{nw_t(r, hold_d):>8.2f}{rl.mean() * 1e4:>9.1f}"
                f"{nw_t(rl, hold_d) if len(rl) > 30 else 0:>8.2f}{p:>10.2e}"
            )


if __name__ == "__main__":
    main()
