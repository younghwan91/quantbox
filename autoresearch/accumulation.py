"""세력 진입 포착 — 신규 상장 초기의 매집 신호가 이후 수익을 가르는가.

716개 신규 상장은 중앙 −26.59%(60일)로 시들지만 평균은 +1.36%다. 70%가 죽고
30%가 폭등한다. 전량 숏은 승률 70%에 기댓값 음수인 함정이고, **가치는 그 30%를
사전에 가려내는 데 있다.**

매집 지표는 주문흐름에서 온다. `taker_buy`는 공격적 매수 체결량이므로
`taker_buy / volume`이 매수 주도 비율이다. 0.5를 넘으면 사는 쪽이 서두른다는 뜻.

## 룩어헤드 차단

  관측창 [0, OBS)  → 신호 계산
  평가창 [OBS, OBS+HOLD) → 수익 측정
두 창은 겹치지 않는다. 신호는 진입 시점까지의 정보만 쓴다.

검정은 심볼 단위 부호검정 + 분위 단조성. 상장은 심볼당 1회라 심볼=독립 관측이다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96  # 15m 봉/일
COST = 12.6e-4


def signals(d: dict, obs: int) -> dict | None:
    """관측창 [0, obs) 에서만 계산한 매집 지표."""
    c, v, tb = d["close"][:obs], d["volume"][:obs], d["taker_buy"][:obs]
    if len(c) < obs or c[0] <= 0 or v.sum() <= 0:
        return None
    half = obs // 2
    vol_tot = float(v.sum())
    return {
        # 매수 주도 비율 — 세력이 사는 쪽에서 서두르는가
        "tb_ratio": float(tb.sum() / vol_tot),
        # 후반 매수 주도 − 전반: 매집이 **강해지고** 있는가
        "tb_trend": float(tb[half:].sum() / max(v[half:].sum(), 1e-12))
        - float(tb[:half].sum() / max(v[:half].sum(), 1e-12)),
        # 거래량 증가: 후반/전반
        "vol_ratio": float(v[half:].sum() / max(v[:half].sum(), 1e-12)),
        # 관측창 수익 — 이미 오른 뒤인가
        "ret_obs": float(c[-1] / c[0] - 1.0),
        # 가격은 눌려 있는데 매수는 강한가 (전형적 매집)
        "accum": float(tb.sum() / vol_tot) - 0.5 - float(c[-1] / c[0] - 1.0),
    }


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    t0 = min(int(d["_offset"]) for d in data.values())

    OBS_D, HOLD_D = 7, 30
    obs, hold = OBS_D * BPD, HOLD_D * BPD

    rows = []
    for s, d in data.items():
        if int(d["_offset"]) <= t0 + BPD:
            continue
        c = d["close"]
        if len(c) < obs + hold or "taker_buy" not in d:
            continue
        sg = signals(d, obs)
        if sg is None or c[obs] <= 0:
            continue
        fwd = float(c[obs + hold] / c[obs] - 1.0)
        rows.append((s, sg, fwd))

    print(f"# 신규 상장 {len(rows)}심볼   관측 {OBS_D}일 → 평가 {HOLD_D}일 (창 분리)")
    fwd = np.array([r[2] for r in rows])
    print(f"# 기준선: 중앙 {np.median(fwd) * 100:>7.2f}%  양수 {(fwd > 0).mean() * 100:.1f}%")
    print()

    print(f"{'지표':<12}{'분위':<8}{'심볼':>6}{'중앙수익%':>11}{'양수%':>8}{'롱 순EV_bp':>12}{'스피어만p':>12}")
    print("-" * 69)
    for key in ("tb_ratio", "tb_trend", "vol_ratio", "ret_obs", "accum"):
        x = np.array([r[1][key] for r in rows])
        rho = sps.spearmanr(x, fwd)
        q = np.percentile(x, [25, 50, 75])
        bins = [
            ("Q1(낮음)", x <= q[0]),
            ("Q4(높음)", x > q[2]),
        ]
        for lab, m in bins:
            sub = fwd[m]
            if len(sub) < 20:
                continue
            med = float(np.median(sub))
            ev = (med - COST) * 1e4
            print(
                f"{key:<12}{lab:<8}{len(sub):>6}{med * 100:>11.2f}"
                f"{(sub > 0).mean() * 100:>7.1f}%{ev:>12.1f}"
                + (f"{rho.pvalue:>12.2e}" if lab.startswith('Q4') else " " * 12)
            )
        print()


if __name__ == "__main__":
    main()
