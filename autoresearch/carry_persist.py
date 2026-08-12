"""조건부 펀딩 캐리 — 지속성 검정.

무조건 보유는 죽었다(299심볼 중 48.2% 양수, p=0.756, 순 중앙 -0.62bp/일).
그러나 이 전략은 **예측이 필요 없다**: 진입 시점에 펀딩률도 차입금리도 관측 가능하다.

그래서 진짜 질문은 하나다 — **지금 유리한 조건이 보유 기간 동안 유지되는가.**

  진입 조건: 직전 정산 펀딩이 음수이고 |펀딩|/일 > 차입/일 × (1+margin)
  보유:      H일
  실현:      보유 구간의 **실제** 펀딩 합 − 차입 × H − 왕복 28bp

룩어헤드 없음: 진입 판단은 t 시점 관측치로만, 실현은 t 이후 실제 펀딩으로.
차입금리는 현재 스냅샷이라 시대 불일치가 남는다(한계로 명시).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats as sps

SP = pathlib.Path(__file__).parent
ROUND_TRIP = 28e-4
DAY_MS = 86_400_000.0


def main() -> None:
    exe = json.loads((SP / "exe.json").read_text())
    borrow = exe["borrow"]
    fund = json.loads((SP / "funding_all.json").read_text())

    print(f"{'조건':<22}{'H일':>5}{'거래N':>8}{'심볼':>6}{'순EV_bp':>10}{'t':>7}{'양수심볼%':>10}{'부호p':>10}")
    print("-" * 78)

    for margin in (0.0, 0.5, 1.0):
        for H in (1, 3, 7):
            per_sym: dict[str, list[float]] = {}
            for s, rows in fund.items():
                if s not in borrow or len(rows) < 200:
                    continue
                ts = np.array([r[0] for r in rows], dtype=np.float64)
                v = np.array([r[1] for r in rows], dtype=np.float64)
                days_tot = (ts[-1] - ts[0]) / DAY_MS
                if days_tot < 60:
                    continue
                per_day = len(v) / days_tot
                b = borrow[s]
                need = b * (1.0 + margin) / per_day  # 정산 1회당 요구 수취액
                horizon = max(1, int(round(H * per_day)))

                out = []
                i = 0
                while i < len(v) - horizon:
                    if v[i] < 0 and abs(v[i]) > need:
                        recv = float(np.sum(np.maximum(-v[i + 1 : i + 1 + horizon], 0.0)))
                        cost = b * H + ROUND_TRIP
                        out.append(recv - cost)
                        i += horizon  # 비중첩
                    else:
                        i += 1
                if out:
                    per_sym[s] = out

            allp = np.concatenate([np.array(x) for x in per_sym.values()]) if per_sym else np.array([])
            if len(allp) < 50:
                print(f"{'|f|>차입x' + f'{1 + margin:.1f}':<22}{H:>5}{len(allp):>8}  표본부족")
                continue
            ev = float(np.mean(allp))
            t = ev / (float(np.std(allp, ddof=1)) / np.sqrt(len(allp)))
            means = np.array([np.mean(x) for x in per_sym.values()])
            pos = int((means > 0).sum())
            pb = sps.binomtest(pos, len(means), 0.5, alternative="greater").pvalue
            print(f"{'|f|>차입x' + f'{1 + margin:.1f}':<22}{H:>5}{len(allp):>8}{len(means):>6}"
                  f"{ev * 1e4:>10.2f}{t:>7.2f}{pos / len(means) * 100:>9.1f}%{pb:>10.2e}")


if __name__ == "__main__":
    main()
