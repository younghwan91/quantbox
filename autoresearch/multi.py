"""다중 신호 계열 — 표본을 늘려 검정력을 얻는다.

ma_cluster 단독으로는 최근 구간 t=1.52에 그친다. 구성을 더 시도해도 t는 오르지 않는다:
t는 √N으로 자라므로 **독립 표본을 더하는 것**이 유일한 정공법이다.

여기서는 같은 처방(TP 천장 제거 + 유동성 필터)을 여러 계열에 적용하고,
계열별 최근 구간 EV와 **풀링 검정**을 함께 낸다.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quantbox.backtest.sim_core import run_ma_cluster_live_fine_exits
from quantbox.backtest.trade_sim import build_resampled, collect_donchian_trades, portfolio_sim
from quantbox.research.harness.data import MarketData
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

BPY = 96 * 365
MIN_DOLLAR_VOL = 1e7
SLOTS = 5


def liquid(data: dict) -> dict:
    """일 거래대금 하한 통과 심볼만. 평평한 비용 가정이 성립하는 범위로 좁힌다."""
    out = {}
    for s, d in data.items():
        v, c = d.get("volume"), d.get("close")
        if v is None or c is None or len(v) < 96:
            continue
        if float(np.median(v[-9600:] * c[-9600:])) * 96 >= MIN_DOLLAR_VOL:
            out[s] = d
    return out


def stats(p: np.ndarray) -> tuple[int, float, float]:
    n = len(p)
    if n < 2:
        return n, 0.0, 0.0
    ev = float(np.mean(p))
    sd = float(np.std(p, ddof=1))
    return n, ev * 1e4, (ev / (sd / np.sqrt(n)) if sd > 0 else 0.0)


def era(pnl: np.ndarray, eb: np.ndarray, end_bar: int, frac: float = 0.5) -> np.ndarray:
    """후반 구간만."""
    return pnl[eb >= int(end_bar * (1.0 - frac))]


def main() -> None:
    t0 = time.time()
    cfg = VolStraddleConfig()
    data_all = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    data = liquid(data_all)
    end_bar = max(int(d["_offset"]) + len(d["close"]) for d in data_all.values())
    years = end_bar / BPY
    print(f"# 유동성 통과 {len(data)}/{len(data_all)}  타임라인 {years:.2f}년  비용 {cfg.round_trip_cost * 1e4:.2f}bp")

    data4 = build_resampled(data, 16)
    pools: dict[str, np.ndarray] = {}

    # ── ma_cluster (TP 천장 제거) ──
    tr, dr = run_ma_cluster_live_fine_exits(
        data, data4, exit_ratio=1, tp=1.0, max_duration=48, min_momentum=0.005, return_directions=True
    )
    pnl = np.asarray(portfolio_sim(tr, SLOTS, directions=dr, cfg=cfg, dir_diversity=cfg.dir_diversity_enabled))
    eb = np.array([t[1] for t in tr])[: len(pnl)] if len(tr) == len(pnl) else None
    _, ev, t = stats(pnl)
    print(f"ma_cluster : 전체 N={len(pnl):>5} EV={ev:>7.2f}bp t={t:>5.2f}")
    pools["ma_cluster"] = pnl

    # ── donchian (같은 처방: TP 천장 제거) ──
    for tp_val, tag in ((None, "prod"), (1.0, "tp-off")):
        kw = {} if tp_val is None else {"tp": tp_val}
        dtr = collect_donchian_trades(data, data4, **kw)
        if not dtr:
            print(f"donchian/{tag}: 거래 없음")
            continue
        dpnl = np.asarray(portfolio_sim(dtr, SLOTS, cfg=cfg))
        n, ev, t = stats(dpnl)
        deb = np.array([x[1] for x in dtr])
        late = era(np.asarray([x[3] for x in dtr]), deb, end_bar)
        _, lev, lt = stats(late)
        print(
            f"donchian/{tag:<7}: 전체 N={n:>5} EV={ev:>7.2f}bp t={t:>5.2f}  |  후반(원거래) N={len(late):>5} EV={lev:>7.2f} t={lt:>5.2f}"
        )
        pools[f"donchian_{tag}"] = dpnl

    # ── 풀링 ──
    print("\n# 풀링 (계열 독립 가정 — 상관이 있으면 t는 과대추정)")
    for combo in (("ma_cluster", "donchian_tp-off"), ("ma_cluster", "donchian_prod")):
        arrs = [pools[k] for k in combo if k in pools]
        if len(arrs) < 2:
            continue
        pooled = np.concatenate(arrs)
        n, ev, t = stats(pooled)
        print(f"  {' + '.join(combo):<34} N={n:>5} EV={ev:>7.2f}bp t={t:>5.2f}")

    print(f"\nelapsed_s: {time.time() - t0:.1f}")


if __name__ == "__main__":
    sys.exit(main())
