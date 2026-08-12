"""델타중립 펀딩 캐리 — 실행 가능 353심볼 전수 평가.

## 구조

  롱 퍼프 + 숏 현물(마진 차입). 펀딩이 **음수**일 때 숏이 롱에게 내므로 롱 퍼프가 받는다.
  현물 숏은 코인을 빌리는 것이므로 그 자산의 차입 이자를 낸다.

    순수익/일 = 펀딩수취/일 − 차입이자/일
    진입 1회당 왕복 = 현물 테이커 10bp×2 + 선물 4bp×2 = 28bp

## 검정

거래 단위 t검정은 소수 이상치에 지배된다(오늘 방향성 단타가 그렇게 죽었다).
**심볼 단위 부호검정**으로 "몇 개 심볼이 독립적으로 양수인가"를 본다 — 분포 가정이 없고
광범위성을 직접 잰다.

## 미반영 (정직하게 남긴다)

  베이시스 위험(퍼프-현물 괴리), 차입 한도·가용성, 현물 체결 비용의 심볼별 차이,
  펀딩 주기 변경 이력, 청산 위험.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats as sps

SP = pathlib.Path(__file__).parent
SPOT_ROUND_TRIP = 28e-4  # 양다리 왕복
DAY_MS = 86_400_000.0


def load() -> tuple[dict, dict, dict]:
    exe = json.loads((SP / "exe.json").read_text())
    fund = json.loads((SP / "funding_all.json").read_text())
    return exe["borrow"], exe.get("limit", {}), fund


def per_symbol(fund_rows: list, borrow_daily: float) -> dict | None:
    """심볼 하나의 캐리 경제성. 데이터가 얕으면 None."""
    if len(fund_rows) < 200:
        return None
    ts = np.array([r[0] for r in fund_rows], dtype=np.float64)
    v = np.array([r[1] for r in fund_rows], dtype=np.float64)
    days = (ts[-1] - ts[0]) / DAY_MS
    if days < 60:
        return None
    per_day = len(v) / days  # 하루 정산 횟수 (4h면 6, 8h면 3)

    neg = v[v < 0]
    if len(neg) < 30:
        return None
    # 음수 펀딩 구간에만 포지션을 잡는다고 보고, 그 구간의 평균 수취율을 쓴다.
    fund_day = float(np.mean(np.abs(neg))) * per_day
    net_day = fund_day - borrow_daily
    frac_neg = len(neg) / len(v)

    return {
        "fund_day_bp": fund_day * 1e4,
        "borrow_day_bp": borrow_daily * 1e4,
        "net_day_bp": net_day * 1e4,
        "frac_neg": frac_neg,
        "n_settle": len(v),
        "days": days,
        "per_day": per_day,
    }


def main() -> None:
    borrow, limit, fund = load()
    rows = {}
    for s, rws in fund.items():
        if s not in borrow:
            continue
        r = per_symbol(rws, borrow[s])
        if r is not None:
            rows[s] = r
    if not rows:
        print("데이터 없음")
        return

    net = np.array([r["net_day_bp"] for r in rows.values()])
    n, pos = len(net), int((net > 0).sum())
    pb = sps.binomtest(pos, n, 0.5, alternative="greater").pvalue
    w = sps.wilcoxon(net, alternative="greater")

    print(f"=== 실행 가능 심볼 {n}개 (무기한 ∩ 현물 ∩ 차입가능, 펀딩 이력 충분)")
    print(f"이력 길이 중앙 {np.median([r['days'] for r in rows.values()]):.0f}일")
    print()
    print(f"순(펀딩수취/일 − 차입이자/일) 양수  {pos}/{n} = {pos / n * 100:.1f}%")
    print(f"  이항검정  p = {pb:.3e}")
    print(f"  윌콕슨    p = {w.pvalue:.3e}")
    print(f"  순 중앙 {np.median(net):>7.2f}bp/일   평균 {net.mean():>7.2f}bp/일")
    print(f"  펀딩수취 중앙 {np.median([r['fund_day_bp'] for r in rows.values()]):>6.2f}  "
          f"차입 중앙 {np.median([r['borrow_day_bp'] for r in rows.values()]):>6.2f} bp/일")

    med = float(np.median(net))
    if med > 0:
        print(f"  왕복 {SPOT_ROUND_TRIP * 1e4:.0f}bp 회수 필요 보유 {SPOT_ROUND_TRIP * 1e4 / med:.1f}일")
        print(f"  상시보유 연환산 {(1 + med / 1e4) ** 365 - 1:.1%}")

    print()
    print("=== 순수익 상위 15")
    top = sorted(rows.items(), key=lambda kv: -kv[1]["net_day_bp"])[:15]
    print(f"{'심볼':<14}{'펀딩/일':>9}{'차입/일':>9}{'순/일':>8}{'음수비율':>9}{'한도':>14}")
    for s, r in top:
        print(f"{s:<14}{r['fund_day_bp']:>9.2f}{r['borrow_day_bp']:>9.2f}{r['net_day_bp']:>8.2f}"
              f"{r['frac_neg'] * 100:>8.0f}%{limit.get(s, 0):>14,.0f}")

    # 상위 K개만 운용할 때의 기대치 — 실제로는 전 종목을 돌리지 않는다
    print()
    print("=== 상위 K개 운용 시 (순/일 중앙)")
    for k in (5, 10, 20, 50):
        sub = np.sort(net)[::-1][:k]
        if len(sub) < k:
            continue
        print(f"  상위 {k:>3}: 중앙 {np.median(sub):>6.2f}bp/일  연환산 {(1 + np.median(sub) / 1e4) ** 365 - 1:>7.1%}")


if __name__ == "__main__":
    main()
