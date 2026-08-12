"""알트 하락 드리프트 — 이게 진짜 엣지인가, 아니면 또 다른 착시인가.

앞선 여섯 번의 기각 과정에서 반복해 찍힌 값이 있다: 알트 단순 보유 수익이
4시간에 **−285bp**이고, 시대가 갈수록 커진다(−119 → −268 → −218 → −285).
676심볼 22만 관측에서 나온 수치다.

이게 실재하면 **아무 신호 없이 그냥 숏**만 해도 돈이 된다. 그래서 검정한다.

## 이번엔 처음부터 함정을 막는다

앞서 두 번 당한 것:
  - 유동성 필터를 표본 **끝** 기준으로 걸면 살아남은 종목만 남는다 → 진입 시점 정보만 쓴다
  - 진입 할인 같은 **기계적 성분**을 알파로 착각한다 → 무조건 보유를 대조군으로 둔다

여기서는 신호가 없다. "무작위 시점에 숏"이 곧 전략이므로 기계적 성분이 끼어들 여지가
없고, 대신 **비용과 펀딩**이 결정적이다. 숏은 펀딩이 양수면 받고 음수면 낸다.

## 왜 공짜가 아닌가

알트가 내리는 만큼 숏에 수요가 몰리면 펀딩이 음수가 되어 숏이 비용을 낸다.
그 상쇄가 얼마인지가 이 전략의 전부다. 펀딩 데이터로 직접 뺀다.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
ROUND_TRIP = 12.6e-4
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-young-Documents-git-quantbox/657b1aaf-a7b9-47a2-bc8f-2835a111cb9b/scratchpad"
)


def load_funding() -> dict:
    p = SCRATCH / "funding_all.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    fund = load_funding()
    print(f"# 심볼 {len(data)}  펀딩 이력 {len(fund)}심볼")

    for hold_d in (1, 7, 30):
        hold = hold_d * BPD
        per: dict[str, np.ndarray] = {}
        perf: dict[str, np.ndarray] = {}
        for s, d in data.items():
            c = d["close"]
            off = int(d["_offset"])
            if len(c) < hold * 3:
                continue
            # 비중첩 구간으로 숏 수익 (신호 없음 — 그냥 주기적으로 숏)
            idx = np.arange(0, len(c) - hold - 1, hold)
            if len(idx) < 3:
                continue
            r = 1.0 - c[idx + hold] / c[idx] - ROUND_TRIP
            per[s] = r
            # 같은 구간의 펀딩 (숏은 양수 펀딩을 받는다 → +sum)
            if s in fund and fund[s]:
                fr = np.array(fund[s], dtype=np.float64)
                ft, fv = fr[:, 0], fr[:, 1]
                # 봉 인덱스 → 시각: timestamps가 없으므로 오프셋 기반 근사 불가.
                # 대신 심볼 전체 평균 펀딩률을 보유기간으로 환산한다(보수적 근사).
                days = (ft[-1] - ft[0]) / 86_400_000.0
                if days > 30:
                    perf[s] = np.full(len(r), float(np.sum(fv)) / days * hold_d)

        if len(per) < 50:
            continue
        allr = np.concatenate(list(per.values()))
        means = np.array([v.mean() for v in per.values()])
        pos = int((means > 0).sum())
        p = sps.binomtest(pos, len(means), 0.5, alternative="greater").pvalue
        print(f"\n보유 {hold_d}일  (비중첩, 신호 없음, 왕복 {ROUND_TRIP * 1e4:.1f}bp)")
        print(
            f"  가격만      N={len(allr):>7}  심볼 {len(means):>3}  EV {allr.mean() * 1e4:>9.2f}bp  "
            f"양수심볼 {pos / len(means) * 100:>5.1f}%  p={p:.2e}"
        )

        # 펀딩 반영
        common = [s for s in per if s in perf]
        if len(common) >= 50:
            withf = np.concatenate([per[s] + perf[s] for s in common])
            mf = np.array([(per[s] + perf[s]).mean() for s in common])
            posf = int((mf > 0).sum())
            pf = sps.binomtest(posf, len(mf), 0.5, alternative="greater").pvalue
            print(
                f"  가격+펀딩   N={len(withf):>7}  심볼 {len(mf):>3}  EV {withf.mean() * 1e4:>9.2f}bp  "
                f"양수심볼 {posf / len(mf) * 100:>5.1f}%  p={pf:.2e}"
            )

        # 시대 분할 — 최근에도 서는가
        print("  시대별(가격만):", end="")
        for lo, hi, lab in ((0.5, 0.75, "3분위"), (0.75, 1.01, "4분위")):
            sub = []
            for s, d in data.items():
                if s not in per:
                    continue
                off = int(d["_offset"])
                idx = np.arange(0, len(d["close"]) - hold - 1, hold)
                bars = off + idx
                m = (bars >= end * lo) & (bars < end * hi)
                if m.any():
                    sub.append(per[s][m].mean())
            if len(sub) >= 30:
                a = np.array(sub)
                pp = sps.binomtest(int((a > 0).sum()), len(a), 0.5, alternative="greater").pvalue
                print(f"  {lab} EV {a.mean() * 1e4:>7.1f}bp 양수 {(a > 0).mean() * 100:>4.1f}% p={pp:.1e}", end="")
        print()


if __name__ == "__main__":
    main()
