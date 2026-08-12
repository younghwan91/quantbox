"""매크로 타이밍 크립토 전략 — 바이낸스 실행용.

신호: 미국 주식시장 20일 실현변동성(EQ_VOL). 하위 20%일 때 이후 크립토가 크게 빠진다
(20일 알트 -6.74% vs 무조건 -1.80%, t=-5.15, p=3.3e-10).

전략: EQ_VOL이 하위 임계 아래면 **알트 숏**, 아니면 현금. 20일 보유.

## 오늘 다섯 번 빠진 함정을 처음부터 막는다

  대조군    무조건 숏·무조건 롱을 나란히 낸다. 초과가 없으면 신호가 아니다.
  비용      왕복 12.6bp 실측치. 20일 보유라 회전이 낮아 영향이 작다.
  다중검정  임계값 3개만 사전 지정(하위 10/20/30%). 결과는 전부 보고한다.
  독립표본  시대 분할로 최근 구간을 따로 낸다.
  룩어헤드  신호는 그날 미국 장 마감 시점 정보만. 진입은 그 이후 봉.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import stats as sps

BPD = 96
COST = 12.6e-4
HOLD_D = 20
THRESHOLDS = (0.10, 0.20, 0.30)


def main() -> None:
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from macro_timing import equity_macro

    from quantbox.research.harness.data import MarketData

    macro = equity_macro("/home/user/data/us_micro.duckdb")
    md = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz")
    data = md.data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        o = int(d["_offset"])
        px[i, o : o + len(d["close"])] = d["close"]
    b = data["BTCUSDT"]
    ts = (float(b["timestamps"][0]) - int(b["_offset"]) * 900_000.0) + np.arange(end) * 900_000.0
    day = np.array([dt.datetime.utcfromtimestamp(t / 1000).date() for t in ts])
    hours = ((ts / 1000) % 86400) / 3600
    close_bars = {day[i]: i for i in np.flatnonzero(np.abs(hours - 21) < 0.13)}

    hold = HOLD_D * BPD
    rows = []
    for d, m in sorted(macro.items()):
        v = m.get("EQ_VOL")
        if v is None or not np.isfinite(v) or d not in close_bars:
            continue
        i = close_bars[d]
        j = i + hold
        if j >= end:
            continue
        c0, c1 = px[:, i], px[:, j]
        ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0)
        if ok.sum() < 30:
            continue
        rows.append((i, v, float(np.median(c1[ok] / c0[ok] - 1.0))))

    if len(rows) < 300:
        print("표본 부족")
        return
    bars = np.array([r[0] for r in rows])
    vol = np.array([r[1] for r in rows])
    alt = np.array([r[2] for r in rows])
    print(f"# 관측 {len(rows)}일  보유 {HOLD_D}일  비용 {COST * 1e4:.1f}bp")
    print(f"# 무조건 롱 {alt.mean() * 100:+.2f}%  무조건 숏 {-alt.mean() * 100:+.2f}% (비용 전)")
    print()
    print(f"{'임계':<10}{'구간':<12}{'진입N':>7}{'숏수익%':>10}{'연%':>8}{'승률':>8}{'t':>7}{'부호p':>10}")
    print("-" * 64)

    for th in THRESHOLDS:
        cut = float(np.quantile(vol, th))
        sel = vol <= cut
        for lo, hi, lab in ((0.0, 1.01, "전체"), (0.0, 0.75, "전반 3/4"), (0.75, 1.01, "최근 1/4")):
            m = sel & (bars >= end * lo) & (bars < end * hi)
            if m.sum() < 30:
                continue
            r = -alt[m] - COST  # 숏 수익
            t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
            p = sps.binomtest(int((r > 0).sum()), len(r), 0.5, alternative="greater").pvalue
            # 중첩 관측이므로 연환산은 비중첩 기준(365/20)으로 보수적으로
            print(
                f"{'하위' + f'{th:.0%}':<10}{lab:<12}{len(r):>7}{r.mean() * 100:>10.2f}"
                f"{r.mean() * (365 / HOLD_D) * 100:>8.1f}{(r > 0).mean() * 100:>7.1f}%{t:>7.2f}{p:>10.2e}"
            )
        print()


if __name__ == "__main__":
    main()
