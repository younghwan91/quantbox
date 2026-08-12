"""횡단면 롱숏 — 시장 방향을 제거하고 구조만 남긴다.

직전 측정에서 알트 숏이 최근 구간 +270.9bp였으나 3분위에서는 −516.9bp였다.
즉 **국면 베팅**이고, 국면을 예측하는 엣지는 오늘 하나도 못 찾았다.

시장 방향을 제거하는 방법은 롱숏이다. 매 리밸런스마다 순위를 매겨
상위를 롱, 하위를 숏하면 공통 요인(시장)이 상쇄되고 **횡단면 분산만** 남는다.
알트 감쇠가 구조적(토큰 언락·희석·상장 후 물량)이라면 그 구조가 순위에 실릴 것이고,
단순 방향 베팅과 달리 국면에 덜 의존해야 한다.

## 검정 설계

  매 REBAL일마다: 그 시점까지의 정보로 전 심볼을 순위 → 상위 K% 롱, 하위 K% 숏
  포지션은 동일 가중, 롱숏 명목 균형(시장 중립)
  비용: 양다리이므로 왕복 12.6bp × 2 = 25.2bp

순위 신호는 사전에 세 개만 정한다(다중검정 억제). 각각 독립적인 가설이다.
검정은 리밸런스 시점 단위 부호검정 — 시점 = 독립 관측이다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
COST_PER_LEG = 12.6e-4
REBAL_D = 7
TOP_FRAC = 0.2


def build_panel(data: dict) -> tuple[np.ndarray, list[str], int]:
    """공통 타임라인 가격 행렬 (심볼 × 봉). 없는 구간은 nan."""
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        off = int(d["_offset"])
        c = d["close"]
        px[i, off : off + len(c)] = c
    return px, syms, end


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    px, syms, end = build_panel(data)
    print(f"# 심볼 {len(syms)}  타임라인 {end / (BPD * 365):.2f}년")

    step = REBAL_D * BPD
    rebal = np.arange(BPD * 60, end - step, step)  # 워밍업 60일
    print(f"# 리밸런스 {len(rebal)}회 ({REBAL_D}일 주기), 상·하위 각 {TOP_FRAC:.0%}")
    print()

    # 사전 지정 신호 3개 — 각각 독립 가설
    lookbacks = {"리버설_1d": 1, "리버설_3d": 3, "리버설_7d": 7, "리버설_14d": 14}

    print(f"{'신호':<14}{'구간':<10}{'시점N':>7}{'평균종목':>9}{'EV_bp':>9}{'t':>7}{'양수시점%':>10}{'부호p':>10}")
    print("-" * 76)

    for name, lb in lookbacks.items():
        back = lb * BPD
        rets_all, times = [], []
        for t in rebal:
            if t - back < 0 or t + step >= end:
                continue
            now, past, fwd = px[:, t], px[:, t - back], px[:, t + step]
            ok = np.isfinite(now) & np.isfinite(past) & np.isfinite(fwd) & (past > 0) & (now > 0)
            if ok.sum() < 30:
                continue
            sig = now[ok] / past[ok] - 1.0
            f = fwd[ok] / now[ok] - 1.0
            k = max(3, int(ok.sum() * TOP_FRAC))
            order = np.argsort(sig)
            short_leg = f[order[:k]].mean()  # 하위 = 약한 종목 → 숏
            long_leg = f[order[-k:]].mean()  # 상위 = 강한 종목 → 롱
            # 모멘텀이 전 구간 음수로 나왔다 → 리버설을 정면으로 검정한다.
            # 약한 종목을 롱, 강한 종목을 숏. 비용은 방향과 무관하게 항상 지불.
            rets_all.append(short_leg - long_leg - 2 * COST_PER_LEG)
            times.append(t)
        if len(rets_all) < 30:
            continue
        r = np.array(rets_all)
        tt = np.array(times)
        for lo, hi, lab in ((0.0, 1.01, "전체"), (0.5, 0.75, "3분위"), (0.75, 1.01, "4분위(최근)")):
            m = (tt >= end * lo) & (tt < end * hi)
            if m.sum() < 20:
                continue
            sub = r[m]
            pos = int((sub > 0).sum())
            p = sps.binomtest(pos, len(sub), 0.5, alternative="greater").pvalue
            t_stat = sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))) if sub.std(ddof=1) > 0 else 0.0
            print(
                f"{name if lab == '전체' else '':<14}{lab:<10}{len(sub):>7}{'':>9}"
                f"{sub.mean() * 1e4:>9.1f}{t_stat:>7.2f}{pos / len(sub) * 100:>9.1f}%{p:>10.2e}"
            )
        print()


if __name__ == "__main__":
    main()
