"""r122c — 숏의 엣지는 신호인가, 알트 드리프트인가.

r122: MA Cluster 숏 EV +23.6bp / 롱 −13.5bp (전 종목 760, 6.58y).
이 비대칭을 그대로 믿고 롱을 끄면, 그건 "신호를 다듬은 것"이 아니라 "알트 하락에
레버리지로 베팅한 것"일 수 있다. 알트 퍼프는 USDT 대비 장기 우하향 드리프트가 있고,
아무 시점에나 숏을 잡아도 평균이 양수로 나올 수 있기 때문이다.

가르는 방법 — 같은 심볼·같은 보유기간·같은 비용의 **무조건 숏** 대조군:
  신호 숏 EV ≈ 대조군 EV   →  엣지 = 드리프트. 신호는 기여가 없다.
  신호 숏 EV >> 대조군 EV  →  타이밍이 실제로 정보를 갖고 있다.

대조군은 진입 시점만 무작위가 아니라 **전 봉**을 쓴다(표본 최대, 선택 없음).
비용은 cfg.round_trip_cost로 동일하게 뺀다.
"""

from __future__ import annotations

import numpy as np

from quantbox.backtest.sim_core import run_ma_cluster_live
from quantbox.backtest.trade_sim import build_resampled
from quantbox.research.harness import MarketData, experiment
from quantbox.research.harness.data import UNIVERSE_CACHE_DIR, UNIVERSE_PATTERN

RATIO = 16


@experiment(name="r122c_drift_baseline")
def run(cfg):
    data = MarketData.load(None, cache_dir=UNIVERSE_CACHE_DIR, pattern=UNIVERSE_PATTERN)
    trades, dirs = run_ma_cluster_live(data.data, return_directions=True)

    holds_4h = [(t[2] - t[1]) // RATIO for t, d in zip(trades, dirs, strict=True) if d == -1]
    h = int(np.median(holds_4h))
    short_pnl = np.array([t[3] for t, d in zip(trades, dirs, strict=True) if d == -1])
    long_pnl = np.array([t[3] for t, d in zip(trades, dirs, strict=True) if d == 1])

    # 대조군: 같은 심볼 집합, 모든 4h 봉에서 h봉 무조건 숏, 동일 비용.
    d4h = build_resampled(data.data, RATIO)
    cost = cfg.round_trip_cost
    base_all, per_sym = [], {}
    for sym, dd in d4h.items():
        c = dd["close"]
        if len(c) < h + 2:
            continue
        r = -(c[h:] - c[:-h]) / c[:-h] - cost  # 숏 수익률
        per_sym[sym] = float(r.mean())
        base_all.append(r)
    base = np.concatenate(base_all)

    def stats(p):
        return {
            "N": int(len(p)),
            "EV_bp": round(float(p.mean()) * 1e4, 2),
            "t": round(float(p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))), 2),
        }

    # 신호 숏 vs 대조군: Welch t
    a, b = short_pnl, base
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {
        "median_hold_4h": h,
        "signal_short": stats(short_pnl),
        "signal_long": stats(long_pnl),
        "baseline_uncond_short": stats(base),
        "excess_bp": round(float(diff) * 1e4, 2),
        "welch_t": round(float(diff / se), 2),
        "symbols_with_positive_short_drift_pct": round(
            100 * float(np.mean([v > 0 for v in per_sym.values()])), 1
        ),
        "n_symbols": len(per_sym),
    }


if __name__ == "__main__":
    r = run().value
    print(f"\n보유 중앙값: {r['median_hold_4h']} × 4h ({r['median_hold_4h'] * 4}h)")
    print(f"신호 숏         : N={r['signal_short']['N']:6d}  EV={r['signal_short']['EV_bp']:8.2f}bp  t={r['signal_short']['t']:6.2f}")
    print(f"신호 롱         : N={r['signal_long']['N']:6d}  EV={r['signal_long']['EV_bp']:8.2f}bp  t={r['signal_long']['t']:6.2f}")
    print(f"무조건 숏 대조군: N={r['baseline_uncond_short']['N']:6d}  EV={r['baseline_uncond_short']['EV_bp']:8.2f}bp  t={r['baseline_uncond_short']['t']:6.2f}")
    print(f"\n초과 EV = {r['excess_bp']}bp   Welch t = {r['welch_t']}")
    print(f"무조건 숏이 양수인 심볼 비율: {r['symbols_with_positive_short_drift_pct']}%  (N={r['n_symbols']})")
