"""주식 → 크립토 교차자산 신호.

지금까지 크립토 가격만 봤고 11개 계열이 전부 죽었다. 이번엔 **다른 자산의 정보**를 쓴다.

## 왜 이 방향인가

미국 주식은 09:30~16:00 ET에만 거래되고 크립토는 24시간이다. 두 시장이 같은 위험선호를
공유한다면, **주식이 닫혀 있는 동안 크립토가 먼저 움직이거나 그 반대**가 된다.
특히 크립토 연관주(COIN·MSTR·채굴주)는 크립토 노출을 그대로 담고 있어,
그 종가 움직임이 다음 크립토 구간에 정보를 줄 수 있다.

## 검정

  신호  미국 장 마감(16:00 ET) 시점의 크립토 연관주 당일 수익률
  대상  그 시점 이후 크립토 수익률 (다음 마감까지)
  대조군 반드시 둔다 — 같은 구간 BTC 단순 보유. 초과가 없으면 베타일 뿐이다.

주식 일봉은 미국 거래일이고 크립토는 매일이므로, **주식 거래일만** 짝짓는다.
16:00 ET = 20:00/21:00 UTC (서머타임). 보수적으로 21:00 UTC를 마감으로 본다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import stats as sps

BPD = 96  # 15m
CLOSE_UTC_H = 21  # 미국 장 마감 근사 (서머타임 고려해 늦은 쪽)
CRYPTO_EQ = ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "CIFR", "BTBT", "HIVE"]


def load_equity(db: str, tickers: list[str]) -> dict[str, dict[dt.date, float]]:
    """티커별 {날짜: 당일 수익률}. 수정종가 기준."""
    import duckdb

    c = duckdb.connect(db, read_only=True)
    out: dict[str, dict[dt.date, float]] = {}
    for t in tickers:
        rows = c.execute("select date, close from prices where ticker=? and close>0 order by date", [t]).fetchall()
        if len(rows) < 200:
            continue
        d = [r[0] for r in rows]
        p = np.array([float(r[1]) for r in rows])
        r_ = p[1:] / p[:-1] - 1.0
        out[t] = dict(zip(d[1:], r_, strict=True))
    c.close()
    return out


def crypto_series(cache_dir: str = "data/cache_universe") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(시각ms, 종가행렬, 심볼). 15m."""
    from quantbox.research.harness.data import MarketData

    md = MarketData.load(cache_dir=cache_dir, pattern="*.npz")
    data = md.data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    t0 = None
    for i, s in enumerate(syms):
        d = data[s]
        o = int(d["_offset"])
        px[i, o : o + len(d["close"])] = d["close"]
        if s == "BTCUSDT":
            t0 = float(d["timestamps"][0]) - o * 900_000.0
    ts = t0 + np.arange(end) * 900_000.0
    return ts, px, syms


def main() -> None:
    eq = load_equity("/home/user/data/us_micro.duckdb", CRYPTO_EQ)
    print(f"# 크립토 연관주 {len(eq)}종목: {sorted(eq)}")

    ts, px, syms = crypto_series()
    print(
        f"# 크립토 {len(syms)}종목  {dt.datetime.utcfromtimestamp(ts[0] / 1000).date()} ~ "
        f"{dt.datetime.utcfromtimestamp(ts[-1] / 1000).date()}"
    )

    # 각 미국 거래일의 21:00 UTC 봉 인덱스
    hours = ((ts / 1000) % 86400) / 3600
    dates = np.array([dt.datetime.utcfromtimestamp(t / 1000).date() for t in ts])
    close_idx = np.flatnonzero(np.abs(hours - CLOSE_UTC_H) < 0.13)
    print(f"# 마감시각({CLOSE_UTC_H}:00 UTC) 봉 {len(close_idx)}개")

    bi = syms.index("BTCUSDT")
    rows = []
    for k in range(len(close_idx) - 1):
        i, j = close_idx[k], close_idx[k + 1]
        d = dates[i]
        sig = [eq[t][d] for t in eq if d in eq[t]]
        if len(sig) < 3:
            continue
        s = float(np.median(sig))  # 연관주 당일 중앙 수익률

        b0, b1 = px[bi, i], px[bi, j]
        if not (np.isfinite(b0) and np.isfinite(b1) and b0 > 0):
            continue
        btc = b1 / b0 - 1.0

        col0, col1 = px[:, i], px[:, j]
        ok = np.isfinite(col0) & np.isfinite(col1) & (col0 > 0)
        alt = float(np.median(col1[ok] / col0[ok] - 1.0)) if ok.sum() >= 30 else np.nan
        rows.append((s, btc, alt))

    if len(rows) < 200:
        print("표본 부족")
        return
    S = np.array([r[0] for r in rows])
    B = np.array([r[1] for r in rows])
    A = np.array([r[2] for r in rows])

    print(f"# 짝지은 거래일 {len(S)}일\n")
    print(f"{'대상':<10}{'신호구간':<14}{'N':>6}{'평균%':>9}{'단순보유%':>11}{'초과bp':>9}{'t':>7}{'상관':>8}")
    print("-" * 76)
    for name, F in (("BTC", B), ("알트중앙", A)):
        ok = np.isfinite(F)
        f_, s_ = F[ok], S[ok]
        rho = sps.spearmanr(s_, f_)
        base = f_.mean()
        q = np.percentile(s_, [20, 80])
        for lab, m in (("연관주 하위20%", s_ <= q[0]), ("연관주 상위20%", s_ >= q[1])):
            sub = f_[m]
            if len(sub) < 30:
                continue
            t = (sub.mean() - base) / (sub.std(ddof=1) / np.sqrt(len(sub)))
            print(
                f"{name:<10}{lab:<14}{len(sub):>6}{sub.mean() * 100:>9.3f}{base * 100:>11.3f}"
                f"{(sub.mean() - base) * 1e4:>9.1f}{t:>7.2f}{rho.statistic:>8.3f}"
            )
        print(f"{'':<10}{'스피어만 p':<14}{rho.pvalue:>10.2e}")
        print()


if __name__ == "__main__":
    main()
