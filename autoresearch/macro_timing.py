"""매크로·인덱스 → 크립토 스윙 타이밍.

크립토 연관주(COIN·MSTR·채굴주)는 신호가 없었다(스피어만 |rho|<0.03). 그건 개별
종목이라 크립토와 **같은 것**을 담고 있어서, 선행 정보가 아니라 동시 정보였다.

매크로는 다르다. 크립토가 고베타 위험자산이라면 **주식시장 전체의 위험선호**가
크립토에 선행할 수 있다. 개별 종목이 아니라 시장 구조에서 뽑는다.

## 사전 지정 신호 3개 (다중검정 억제)

  BREADTH   50일 이평 위 종목 비율 — 시장 폭. 위험선호의 직접 측정.
  SIZE_SPR  소형주 − 대형주 20일 수익 — 위험선호가 클수록 소형주가 앞선다.
  EQ_VOL    동일가중 지수의 20일 실현변동성 — 스트레스 지표.

각 신호를 상·하위 20%로 나눠 **이후 크립토 수익**을 본다.
대조군은 무조건 보유. 초과가 없으면 베타일 뿐이다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import stats as sps

BPD = 96
HORIZONS_D = (5, 20)
MIN_STOCKS = 500


def equity_macro(db: str) -> dict[dt.date, dict[str, float]]:
    """일자별 매크로 신호. 그날까지의 정보만 사용."""
    import duckdb

    c = duckdb.connect(db, read_only=True)
    # 유동성 있는 보통주만 — 잡주 노이즈 제거
    rows = c.execute("""
        select p.date, p.ticker, p.close, p.mcap
        from prices p join tickers t using(ticker)
        where t.category like '%Common Stock%' and p.close>5 and p.volume*p.close>1e6
          and p.date >= '2019-06-01'
        order by p.date
    """).fetchall()
    c.close()

    by_date: dict[dt.date, list[tuple[str, float, float]]] = {}
    for d, tk, cl, mc in rows:
        by_date.setdefault(d, []).append((tk, float(cl), float(mc or 0)))
    dates = sorted(by_date)
    print(f"# 주식 일자 {len(dates)}일  종목/일 중앙 {int(np.median([len(by_date[d]) for d in dates]))}")

    hist: dict[str, list[float]] = {}
    idx_ret: list[float] = []
    out: dict[dt.date, dict[str, float]] = {}
    prev: dict[str, float] = {}

    for d in dates:
        cur = {tk: cl for tk, cl, _ in by_date[d]}
        mc = {tk: m for tk, _, m in by_date[d]}
        if len(cur) < MIN_STOCKS:
            prev = cur
            continue

        # 종목별 종가 이력 (이평용)
        for tk, cl in cur.items():
            hist.setdefault(tk, []).append(cl)
            if len(hist[tk]) > 60:
                hist[tk].pop(0)

        # 일간 수익 (전일 대비)
        rets = {tk: cur[tk] / prev[tk] - 1.0 for tk in cur if tk in prev and prev[tk] > 0}
        if len(rets) < MIN_STOCKS:
            prev = cur
            continue

        above = [1.0 if len(hist[tk]) >= 50 and cur[tk] > np.mean(hist[tk][-50:]) else 0.0 for tk in cur]
        breadth = float(np.mean(above))

        # 소형 vs 대형 (시총 하위/상위 30%)
        caps = np.array([mc[tk] for tk in rets])
        rr = np.array([rets[tk] for tk in rets])
        ok = caps > 0
        size_spr = np.nan
        if ok.sum() >= MIN_STOCKS:
            q = np.percentile(caps[ok], [30, 70])
            small = rr[ok][caps[ok] <= q[0]].mean()
            large = rr[ok][caps[ok] >= q[1]].mean()
            size_spr = float(small - large)

        idx_ret.append(float(np.mean(rr)))
        if len(idx_ret) > 20:
            idx_ret.pop(0)
        eq_vol = float(np.std(idx_ret)) if len(idx_ret) >= 20 else np.nan

        out[d] = {"BREADTH": breadth, "SIZE_SPR": size_spr, "EQ_VOL": eq_vol}
        prev = cur
    return out


def main() -> None:
    from quantbox.research.harness.data import MarketData

    macro = equity_macro("/home/user/data/us_micro.duckdb")
    print(f"# 매크로 신호 일자 {len(macro)}일")

    md = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz")
    data = md.data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)
    px = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        o = int(d["_offset"])
        px[i, o : o + len(d["close"])] = d["close"]
    btc = data["BTCUSDT"]
    t0 = float(btc["timestamps"][0]) - int(btc["_offset"]) * 900_000.0
    ts = t0 + np.arange(end) * 900_000.0
    day = np.array([dt.datetime.utcfromtimestamp(t / 1000).date() for t in ts])

    # 각 매크로 일자의 21:00 UTC 봉
    hours = ((ts / 1000) % 86400) / 3600
    close_bars = {day[i]: i for i in np.flatnonzero(np.abs(hours - 21) < 0.13)}

    bi = syms.index("BTCUSDT")
    print()
    print(
        f"{'신호':<10}{'지평':>5}{'대상':<8}{'구간':<12}{'N':>6}{'수익%':>9}{'무조건%':>10}{'초과bp':>9}{'t':>7}{'p':>10}"
    )
    print("-" * 82)

    for key in ("BREADTH", "SIZE_SPR", "EQ_VOL"):
        for H in HORIZONS_D:
            pairs = []
            for d, m in macro.items():
                v = m.get(key)
                if v is None or not np.isfinite(v) or d not in close_bars:
                    continue
                i = close_bars[d]
                j = i + H * BPD
                if j >= end:
                    continue
                b0, b1 = px[bi, i], px[bi, j]
                if not (np.isfinite(b0) and np.isfinite(b1) and b0 > 0):
                    continue
                c0, c1 = px[:, i], px[:, j]
                ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0)
                alt = float(np.median(c1[ok] / c0[ok] - 1.0)) if ok.sum() >= 30 else np.nan
                pairs.append((v, b1 / b0 - 1.0, alt))
            if len(pairs) < 200:
                continue
            V = np.array([p[0] for p in pairs])
            for nm, F in (("BTC", np.array([p[1] for p in pairs])), ("알트", np.array([p[2] for p in pairs]))):
                ok = np.isfinite(F)
                v_, f_ = V[ok], F[ok]
                base = f_.mean()
                rho = sps.spearmanr(v_, f_)
                q = np.percentile(v_, [20, 80])
                for lab, msk in (("하위20%", v_ <= q[0]), ("상위20%", v_ >= q[1])):
                    sub = f_[msk]
                    if len(sub) < 30:
                        continue
                    t = (sub.mean() - base) / (sub.std(ddof=1) / np.sqrt(len(sub)))
                    print(
                        f"{key:<10}{H:>4}d{nm:<8}{lab:<12}{len(sub):>6}{sub.mean() * 100:>9.2f}"
                        f"{base * 100:>10.2f}{(sub.mean() - base) * 1e4:>9.0f}{t:>7.2f}{rho.pvalue:>10.2e}"
                    )
            print()


if __name__ == "__main__":
    main()
