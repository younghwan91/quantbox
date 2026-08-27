"""r122b — BTC 게이트가 롱의 음(-) EV를 구제하는가.

r122에서 소지(素地) 상태의 MA Cluster 롱은 EV −11.3bp (전 종목, t=−2.19)였다.
그런데 프로덕션은 롱에만 BTC 추세 게이트를 두 겹으로 걸고 있다:
  G6            btc_long_ma_period=50 @ 15m  (12.5h)  — 전 전략 공통
  ma_cluster    ma_cluster_btc_ma_period_4h=50 @ 4h   (8.3일) — r92, 3월 라이브 실패의 근인

따라서 "롱을 꺼야 하는가"는 게이트를 **켠 상태**에서 다시 물어야 한다. 게이트가
음의 구간을 잘라내고 있다면 롱은 살아 있는 것이고, 켜도 여전히 음수면 그때는
게이트가 아니라 방향 자체가 문제다.

4h 게이트는 라이브 `compute_sma`로 계산한다 — np.mean과 1.5e-9 어긋나 실제 진입이
뒤집힌 전례가 있다(project_compute_sma_parity). G6는 프로덕션 portfolio_sim의
`btc_gate=True` 경로를 그대로 쓴다(risk_guard.allow_long).
"""

from __future__ import annotations

import numpy as np

from quantbox.backtest.sim_core import run_ma_cluster_live
from quantbox.backtest.trade_sim import build_resampled, compute_metrics, portfolio_sim
from quantbox.research.harness import MarketData, experiment, report
from quantbox.research.harness.data import UNIVERSE_CACHE_DIR, UNIVERSE_PATTERN
from quantbox.strategy.vol_straddle.signals import compute_sma

RATIO = 16  # 15m → 4h


def _btc_4h_gate_mask(btc_4h_close: np.ndarray, cfg) -> np.ndarray:
    """4h 봉별 '롱 허용' 마스크. BTC 4h 종가 >= SMA50(4h)."""
    sma = compute_sma(btc_4h_close, cfg.ma_cluster_btc_ma_period_4h)
    return btc_4h_close >= sma


def _apply_4h_gate(trades, dirs, mask, offset_ok=True):
    """롱만 4h 게이트로 거른다. 숏은 라이브와 동일하게 통과."""
    out_t, out_d = [], []
    for t, d in zip(trades, dirs, strict=True):
        if d == 1:
            idx = t[1] // RATIO  # entry_bar(15m) → 4h index
            if idx >= len(mask) or not mask[idx]:
                continue
        out_t.append(t)
        out_d.append(d)
    return out_t, out_d


@experiment(name="r122b_btc_gate_long")
def run(cfg):
    results = {}
    for label, kw, restrict in (
        ("production-60", {}, True),
        ("universe-all", {"cache_dir": UNIVERSE_CACHE_DIR, "pattern": UNIVERSE_PATTERN}, False),
    ):
        data = MarketData.load(cfg.symbol_whitelist if restrict else None, **kw)
        trades, dirs = run_ma_cluster_live(data.data, return_directions=True)

        btc15 = data.data["BTCUSDT"]["close"]
        btc4h = build_resampled({"BTCUSDT": data.data["BTCUSDT"]}, RATIO)["BTCUSDT"]["close"]
        mask4h = _btc_4h_gate_mask(btc4h, cfg)
        years = report.data_years(data)

        variants = {}
        # (라벨, 4h 게이트, G6 게이트)
        for vlabel, use4h, useG6 in (
            ("gate: none", False, False),
            ("gate: 4h only", True, False),
            ("gate: G6 only", False, True),
            ("gate: 4h + G6 (프로덕션)", True, True),
        ):
            t, d = (_apply_4h_gate(trades, dirs, mask4h) if use4h else (trades, dirs))
            longs = [(tt, dd) for tt, dd in zip(t, d, strict=True) if dd == 1]
            sub_t = [x[0] for x in longs]
            sub_d = [x[1] for x in longs]
            if not sub_t:
                variants[vlabel] = None
                continue
            pnl = portfolio_sim(
                sub_t,
                cfg.max_concurrent,
                None,
                directions=sub_d if useG6 else None,
                btc_close=btc15 if useG6 else None,
                cfg=cfg,
                btc_gate=useG6,
            )
            m = compute_metrics(pnl, years, portfolio_size=cfg.max_concurrent)
            raw = np.array([x[3] for x in sub_t], dtype=np.float64)
            m["n_raw"] = len(sub_t)
            m["n_filled"] = len(pnl)
            m["raw_EV_bp"] = round(float(raw.mean()) * 1e4, 2)
            variants[vlabel] = m
        results[label] = {"years": round(years, 2), "n_symbols": len(data.data), "long": variants}
    return results


if __name__ == "__main__":
    res = run()
    for label, r in res.value.items():
        print(f"\n{'=' * 72}\n{label} — LONG only, {r['n_symbols']} symbols, {r['years']}y\n{'=' * 72}")
        for vlabel, m in r["long"].items():
            if m is None:
                print(f"  {vlabel:26s} 트레이드 없음")
                continue
            print(
                f"  {vlabel:26s} N={m['n_filled']:5d}  EV={m['EV_bp']:8.2f}bp  "
                f"PF={m['PF']:6.3f}  t={m['t']:6.2f}  WR={m['WR']:5.1f}%  maxDD={m['max_dd_pct']:5.1f}%"
            )
