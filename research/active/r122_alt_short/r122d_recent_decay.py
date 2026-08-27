"""r122d — 숏 엣지의 감쇠가 어디까지 왔는가.

r122에서 숏의 워크포워드 PF가 1.635 → 1.718 → 1.214 → 1.033으로 단조 감쇠했다.
1.033은 사실상 손익분기다. "롱을 끄고 숏만 켠다"는 결정은 **최근 구간에도 엣지가
남아 있는가**에 달려 있으므로, 달력 연도별로 다시 자른다.

r122c와 같은 판정 기준을 각 구간에 적용한다 — 절대 EV가 아니라 **드리프트 대조군
대비 초과 EV**를 본다. 절대 EV는 그 해 알트가 어느 방향으로 흘렀는지에 오염되지만,
초과 EV는 "그 해에도 진입 타이밍이 정보를 갖고 있었나"만 묻는다.

구간 표본이 작아 초과 EV 하나로는 판정하지 않는다. Welch t와 N을 같이 읽는다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from quantbox.backtest.sim_core import run_ma_cluster_live
from quantbox.backtest.trade_sim import build_resampled, compute_metrics, portfolio_sim
from quantbox.research.harness import MarketData, experiment, report
from quantbox.research.harness.data import BAR_MS, UNIVERSE_CACHE_DIR, UNIVERSE_PATTERN

RATIO = 16
HOLD_4H = 5  # r122c 실측 보유 중앙값


def _global_t0(data: dict) -> float:
    """정렬된 공통 타임라인의 0번 봉 timestamp(ms)."""
    cands = [d["timestamps"][0] - d["_offset"] * BAR_MS for d in data.values() if "timestamps" in d]
    if not cands:
        raise ValueError("timestamps 없는 캐시 — cache_universe를 써야 한다")
    return float(min(cands))


def _year_of(bar: int, t0: float) -> int:
    return dt.datetime.fromtimestamp((t0 + bar * BAR_MS) / 1000, dt.UTC).year


@experiment(name="r122d_recent_decay")
def run(cfg):
    data = MarketData.load(None, cache_dir=UNIVERSE_CACHE_DIR, pattern=UNIVERSE_PATTERN)
    trades, dirs = run_ma_cluster_live(data.data, return_directions=True)
    t0 = _global_t0(data.data)
    years_total = report.data_years(data)

    # 무조건 숏 대조군을 연도별로 — 같은 보유기간·같은 비용.
    d4h = build_resampled(data.data, RATIO)
    cost = cfg.round_trip_cost
    base_by_year: dict[int, list[np.ndarray]] = {}
    for sym, dd in d4h.items():
        c = dd["close"]
        if len(c) < HOLD_4H + 2:
            continue
        off = int(data.data[sym]["_offset"])
        r = -(c[HOLD_4H:] - c[:-HOLD_4H]) / c[:-HOLD_4H] - cost
        yrs = np.array([_year_of(i * RATIO + off, t0) for i in range(len(r))])
        for y in np.unique(yrs):
            base_by_year.setdefault(int(y), []).append(r[yrs == y])

    tagged = [(t, d, _year_of(t[1], t0)) for t, d in zip(trades, dirs, strict=True)]
    all_years = sorted({y for _, _, y in tagged})

    def block(sel_trades, sel_dirs, base):
        raw = np.array([t[3] for t in sel_trades], dtype=np.float64) if sel_trades else np.empty(0)
        out = {"N": int(len(raw))}
        if len(raw) < 2:
            return out | {"EV_bp": None, "excess_bp": None, "welch_t": None, "PF": None}
        pnl = portfolio_sim(sel_trades, cfg.max_concurrent, None)
        m = compute_metrics(pnl, max(years_total / max(len(all_years), 1), 1e-9), portfolio_size=cfg.max_concurrent)
        diff = float(raw.mean() - base.mean())
        se = float(np.sqrt(raw.var(ddof=1) / len(raw) + base.var(ddof=1) / len(base)))
        return {
            "N": int(len(raw)),
            "N_slot": int(len(pnl)),
            "EV_bp": round(float(raw.mean()) * 1e4, 2),
            "excess_bp": round(diff * 1e4, 2),
            "welch_t": round(diff / se, 2) if se > 0 else None,
            "PF": m["PF"],
        }

    results = {}
    for y in all_years:
        base = np.concatenate(base_by_year.get(y, [np.zeros(1)]))
        row = {"baseline_short_EV_bp": round(float(base.mean()) * 1e4, 2), "baseline_N": int(len(base))}
        for name, want in (("short", -1), ("long", 1)):
            sel = [(t, d) for t, d, yy in tagged if yy == y and d == want]
            row[name] = block([x[0] for x in sel], [x[1] for x in sel], base)
        results[y] = row

    # 최근 2년 합산 (연 단위 표본이 얇을 때의 판정용)
    recent = all_years[-2:]
    base_recent = np.concatenate([np.concatenate(base_by_year[y]) for y in recent if y in base_by_year])
    agg = {}
    for name, want in (("short", -1), ("long", 1)):
        sel = [(t, d) for t, d, yy in tagged if yy in recent and d == want]
        agg[name] = block([x[0] for x in sel], [x[1] for x in sel], base_recent)
    results["recent2y"] = {"years": recent, "baseline_short_EV_bp": round(float(base_recent.mean()) * 1e4, 2), **agg}
    return results


if __name__ == "__main__":
    res = run().value
    hdr = f"{'연도':>8} | {'대조군':>9} | {'숏 N':>6} {'숏 EV':>9} {'초과':>9} {'t':>6} {'PF':>6} | {'롱 N':>6} {'롱 EV':>9} {'초과':>9} {'t':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for y, r in res.items():
        if y == "recent2y":
            continue
        s, lo = r["short"], r["long"]
        print(
            f"{y:>8} | {r['baseline_short_EV_bp']:>9.2f} | "
            f"{s['N']:>6} {str(s['EV_bp']):>9} {str(s['excess_bp']):>9} {str(s['welch_t']):>6} {str(s['PF']):>6} | "
            f"{lo['N']:>6} {str(lo['EV_bp']):>9} {str(lo['excess_bp']):>9} {str(lo['welch_t']):>6}"
        )
    r = res["recent2y"]
    print(f"\n최근 2년 {r['years']} 합산  (무조건 숏 대조군 {r['baseline_short_EV_bp']}bp)")
    for k in ("short", "long"):
        m = r[k]
        print(f"  {k:6s} N={m['N']:5d}  EV={m['EV_bp']}bp  초과={m['excess_bp']}bp  Welch t={m['welch_t']}  PF={m['PF']}")
