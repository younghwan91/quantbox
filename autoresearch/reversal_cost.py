"""횡단면 리버설 — 선택 종목의 실제 비용으로 재계산.

앞서 다섯 계열이 전부 비용에서 죽었다. 리버설은 25.2bp 가정에서 +107.8bp지만,
**하위 20%를 롱**하는 구조라 급락한 저유동 종목이 그쪽에 몰린다. 그 종목들의
실제 스프레드가 가정보다 크면 결론이 바뀐다.

비용은 심볼별·시점별로 Corwin-Schultz(2012)로 추정한다. 이 추정량은 소켓
bookTicker 실측 46심볼에 대조해 **넓은 스프레드 구간에서 배수 1.05로 정확**함을
확인했다(좁은 구간은 7.81배 과대). 리버설이 고르는 종목은 넓은 쪽이므로 적용 가능하다.

비용 = 스프레드 크로싱(반스프레드 × 2다리 × 진입청산) + 수수료
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from quantbox.research.harness.data import MarketData

BPD = 96
FEE_ROUND_TRIP = 8.6e-4  # 실측 수수료 (양쪽 taker)
CS_CALIB = 0.952  # 소켓 실측 대조로 얻은 보정계수
TOP_FRAC = 0.2
LB_D = 7
HOLD_D = 7


def cs_spread(high: np.ndarray, low: np.ndarray) -> float:
    """Corwin-Schultz 스프레드 추정 (비율). 표본 부족·계산 불가면 nan."""
    ok = (high > 0) & (low > 0) & (high >= low)
    h, lo = high[ok], low[ok]
    if len(h) < 20:
        return np.nan
    b = np.log(h[:-1] / lo[:-1]) ** 2 + np.log(h[1:] / lo[1:]) ** 2
    g = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    a = (np.sqrt(2.0 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    s = 2.0 * (np.exp(a) - 1.0) / (1.0 + np.exp(a))
    s = s[np.isfinite(s)]
    return float(np.mean(np.maximum(s, 0.0))) * CS_CALIB if len(s) else np.nan


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    syms = sorted(data)

    px = np.full((len(syms), end), np.nan)
    hi = np.full((len(syms), end), np.nan)
    lo_ = np.full((len(syms), end), np.nan)
    for i, s in enumerate(syms):
        d = data[s]
        o, c = int(d["_offset"]), d["close"]
        px[i, o : o + len(c)] = c
        hi[i, o : o + len(c)] = d["high"]
        lo_[i, o : o + len(c)] = d["low"]

    lb, hold = LB_D * BPD, HOLD_D * BPD
    rebal = np.arange(BPD * 60, end - hold, BPD)
    print(f"# 심볼 {len(syms)}  리밸런스 {len(rebal)}회 (중첩, 매일)")

    flat, real, times, sp_long, sp_short = [], [], [], [], []
    for t in rebal:
        if t - lb < 0:
            continue
        now, past, fwd = px[:, t], px[:, t - lb], px[:, t + hold]
        ok = np.isfinite(now) & np.isfinite(past) & np.isfinite(fwd) & (past > 0) & (now > 0)
        if ok.sum() < 30:
            continue
        idx = np.flatnonzero(ok)
        sig = now[idx] / past[idx] - 1.0
        f = fwd[idx] / now[idx] - 1.0
        k = max(3, int(len(idx) * TOP_FRAC))
        order = np.argsort(sig)
        li, si = idx[order[:k]], idx[order[-k:]]  # 롱=하위(약함), 숏=상위(강함)

        gross = f[order[:k]].mean() - f[order[-k:]].mean()

        # 선택된 종목의 스프레드 (진입 직전 30일 구간으로 추정 — 미래 미사용)
        w0 = max(0, t - BPD * 30)
        sl = np.nanmedian([cs_spread(hi[i, w0:t], lo_[i, w0:t]) for i in li])
        ss = np.nanmedian([cs_spread(hi[i, w0:t], lo_[i, w0:t]) for i in si])
        if not (np.isfinite(sl) and np.isfinite(ss)):
            continue
        # 각 다리: 반스프레드를 진입·청산 두 번 지불 = 스프레드 1개분
        cost_real = sl + ss + FEE_ROUND_TRIP * 2

        flat.append(gross - 25.2e-4)
        real.append(gross - cost_real)
        sp_long.append(sl)
        sp_short.append(ss)
        times.append(t)

    r_flat, r_real = np.array(flat), np.array(real)
    tt = np.array(times)
    print(f"# 유효 시점 {len(tt)}")
    print()
    print(f"롱 다리(하위20%) 스프레드 중앙 {np.median(sp_long) * 1e4:>7.2f}bp")
    print(f"숏 다리(상위20%) 스프레드 중앙 {np.median(sp_short) * 1e4:>7.2f}bp")
    print(
        f"→ 실제 왕복 비용 중앙 {(np.median(sp_long) + np.median(sp_short) + FEE_ROUND_TRIP * 2) * 1e4:>7.2f}bp  (가정 25.2bp)"
    )
    print()

    def nw_t(x: np.ndarray, lag: int = 7) -> float:
        n = len(x)
        e = x - x.mean()
        s = (e @ e) / n
        for l_ in range(1, lag + 1):
            s += 2 * (1 - l_ / (lag + 1)) * ((e[l_:] @ e[:-l_]) / n)
        return float(x.mean() / np.sqrt(max(s, 1e-18) / n))

    print(f"{'비용':<12}{'구간':<12}{'N':>6}{'EV_bp':>9}{'t(NW)':>8}{'양수%':>8}{'연환산':>9}")
    print("-" * 60)
    for lab, r in (("가정 25.2bp", r_flat), ("실측 추정", r_real)):
        for lo2, hi2, era in ((0.0, 1.01, "전체"), (0.75, 1.01, "최근 1/4")):
            m = (tt >= end * lo2) & (tt < end * hi2)
            if m.sum() < 30:
                continue
            sub = r[m]
            print(
                f"{lab if era == '전체' else '':<12}{era:<12}{len(sub):>6}{sub.mean() * 1e4:>9.1f}"
                f"{nw_t(sub):>8.2f}{(sub > 0).mean() * 100:>7.1f}%{sub.mean() * 52 * 100:>8.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
