"""r85: S_d8 Straddle — 2D Trail Sweep (cliff edge landscape).

목적:
  r77에서 발견된 cliff edge (s_trail_activate ±0.003 → PF -60~75%) 의
  전체 landscape를 2D 그리드로 매핑.
  - X축: s_trail_activate (0.005 ~ 0.030, 6 steps)
  - Y축: s_trail_distance  (0.0005 ~ 0.004, 6 steps)
  - 각 셀: N, PF, EV(bp), MaxDD, Calmar

결론 목표:
  1) 현재 프로덕션 값 (0.015, 0.001)이 peak인가, 아니면 더 좋은 영역이 있는가?
  2) peak 주변의 gradient — robust한 plateau인가, 날카로운 cliff인가?
  3) OOS 안정성을 위해 약간 낮은 EV를 감수하고 더 평평한 구간을 선택할 가치가 있나?

사용:
    uv run python research/active/r81_r85/r85_trail_2d_sweep.py

현재 프로덕션 (config.py):
  s_trail_activate=0.015  s_trail_distance=0.001  → Calmar=10,175 (r84)
  ⚠️  r77: ±0.003 변화 시 PF 60-75% 붕괴 (cliff 확인)
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    WARMUP_15m,
    WARMUP_1h,
    WARMUP_4h,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    gen_vr_dur_signal,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

WARMUP_30m: int = 100
MAX_CONCURRENT = 5  # STRD slot parity: 1 straddle = 2 positions

cfg = VolStraddleConfig()

# ── Grid ────────────────────────────────────────────────────────────────────
TRAIL_ACTIVATE_GRID = [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.021, 0.025, 0.030]
TRAIL_DISTANCE_GRID = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]

PROD_ACT = cfg.s_trail_activate    # 0.015
PROD_DIST = cfg.s_trail_distance   # 0.001


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_15m(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in set(whitelist):
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def _dedup(sym: str, trades: list) -> list:
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_strd_trail(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    trail_act: float,
    trail_dist: float,
) -> list[tuple[str, int, int, float]]:
    """S_d8 G_4TF collection with custom trail params."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    mh_30m = cfg.max_hold_15m // 2
    g4tf = [
        ("30m", 2, WARMUP_30m, mh_30m * 2),
        ("15m", 1, WARMUP_15m, cfg.max_hold_15m * 2),
        ("1h",  4, WARMUP_1h,  cfg.max_hold_15m // 4 * 2),
        ("4h",  16, WARMUP_4h, cfg.max_hold_15m // 16 * 2),
    ]
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl = cfg.s_sl
    tp = cfg.s_tp

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, mh_s in g4tf:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE, trail_act, trail_dist,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.perf_counter()
    whitelist = cfg.symbol_whitelist
    assert len(whitelist) == 60

    print("=" * 70)
    print("r85: S_d8 Straddle — 2D Trail Sweep (cliff edge landscape)")
    print("=" * 70)
    print(f"\n프로덕션 기준: s_trail_activate={PROD_ACT}  s_trail_distance={PROD_DIST}")
    print(f"Grid: act={TRAIL_ACTIVATE_GRID}")
    print(f"      dist={TRAIL_DISTANCE_GRID}")
    print(f"총 {len(TRAIL_ACTIVATE_GRID) * len(TRAIL_DISTANCE_GRID)} 조합")

    print("\n데이터 로드 중...")
    data_15m = load_15m(whitelist)
    assert len(data_15m) >= 58
    sample_bars = max(len(d["close"]) for d in data_15m.values())
    data_years = sample_bars / 35040.0
    print(f"  {len(data_15m)} 심볼  |  {sample_bars:,} bars  |  {data_years:.2f}년")

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    vr_pairs = [(cfg.vr_short, cfg.vr_long)]
    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], vr_pairs)
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)
    print(f"  전처리 완료: {time.perf_counter() - t0:.1f}s\n")

    # ── Grid sweep ──
    results: dict[tuple[float, float], dict] = {}
    n_total = len(TRAIL_ACTIVATE_GRID) * len(TRAIL_DISTANCE_GRID)
    done = 0

    for act, dist in product(TRAIL_ACTIVATE_GRID, TRAIL_DISTANCE_GRID):
        t1 = time.perf_counter()
        trades = collect_strd_trail(data_15m, data_30m, data_1h, data_4h, act, dist)
        if not trades:
            results[(act, dist)] = {"N": 0, "PF": 0.0, "EV_bp": 0.0, "MaxDD": 0.0, "Calmar": 0.0}
            done += 1
            continue

        pnl = portfolio_sim(trades, max_concurrent=MAX_CONCURRENT)
        m = compute_metrics(pnl, data_years)
        boot_d = bootstrap_pf(pnl, n_boot=300)

        results[(act, dist)] = {
            "N": m["N"],
            "PF": m["PF"],
            "EV_bp": m["EV_bp"],
            "MaxDD": m["max_dd_pct"],
            "Calmar": m["calmar"],
            "P_PF_ge_1": boot_d["P_PF_ge_1"],
        }
        done += 1
        prod_mark = " ← PROD" if (abs(act - PROD_ACT) < 1e-6 and abs(dist - PROD_DIST) < 1e-6) else ""
        print(
            f"  [{done:2d}/{n_total}] act={act:.3f} dist={dist:.4f} → "
            f"N={m['N']:4d} PF={m['PF']:6.2f} EV={m['EV_bp']:+.1f}bp "
            f"DD={m['max_dd_pct']:.1f}% Calmar={m['calmar']:.0f} "
            f"({time.perf_counter() - t1:.1f}s){prod_mark}"
        )

    # ── Summary tables ──
    print("\n" + "=" * 70)
    print("PF 히트맵 (행=trail_activate, 열=trail_distance)")
    print("=" * 70)
    dist_header = "  act\\dist " + "".join(f"  {d:.4f}" for d in TRAIL_DISTANCE_GRID)
    print(dist_header)
    for act in TRAIL_ACTIVATE_GRID:
        row = f"  {act:.3f}     "
        for dist in TRAIL_DISTANCE_GRID:
            r = results.get((act, dist), {})
            pf = r.get("PF", 0.0)
            mark = "*" if (abs(act - PROD_ACT) < 1e-6 and abs(dist - PROD_DIST) < 1e-6) else " "
            row += f"  {pf:5.2f}{mark}"
        print(row)

    print("\n" + "=" * 70)
    print("EV(bp) 히트맵")
    print("=" * 70)
    print(dist_header)
    for act in TRAIL_ACTIVATE_GRID:
        row = f"  {act:.3f}     "
        for dist in TRAIL_DISTANCE_GRID:
            r = results.get((act, dist), {})
            ev = r.get("EV_bp", 0.0)
            mark = "*" if (abs(act - PROD_ACT) < 1e-6 and abs(dist - PROD_DIST) < 1e-6) else " "
            row += f"  {ev:+5.1f}{mark}"
        print(row)

    print("\n" + "=" * 70)
    print("Calmar 히트맵")
    print("=" * 70)
    print(dist_header)
    for act in TRAIL_ACTIVATE_GRID:
        row = f"  {act:.3f}     "
        for dist in TRAIL_DISTANCE_GRID:
            r = results.get((act, dist), {})
            cal = r.get("Calmar", 0.0)
            mark = "*" if (abs(act - PROD_ACT) < 1e-6 and abs(dist - PROD_DIST) < 1e-6) else " "
            row += f"  {cal:5.0f}{mark}"
        print(row)

    # ── Best configs ──
    print("\n" + "=" * 70)
    print("상위 10 조합 (Calmar 기준)")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: x[1].get("Calmar", 0), reverse=True)
    print(f"{'act':>6s} {'dist':>7s}  {'N':>5s}  {'PF':>7s}  {'EV(bp)':>8s}  {'MaxDD%':>7s}  {'Calmar':>8s}")
    print("-" * 60)
    for (act, dist), r in ranked[:10]:
        prod_mark = " ← PROD" if (abs(act - PROD_ACT) < 1e-6 and abs(dist - PROD_DIST) < 1e-6) else ""
        print(
            f"  {act:.3f}  {dist:.4f}  {r['N']:>5d}  {r['PF']:>7.2f}  "
            f"{r['EV_bp']:>+8.1f}  {r['MaxDD']:>7.1f}  {r['Calmar']:>8.0f}{prod_mark}"
        )

    # ── Prod vs best ──
    prod_r = results.get((PROD_ACT, PROD_DIST), {})
    best_key, best_r = ranked[0]
    print(f"\n프로덕션 ({PROD_ACT}, {PROD_DIST}): PF={prod_r.get('PF', 0):.2f}  "
          f"EV={prod_r.get('EV_bp', 0):+.1f}bp  Calmar={prod_r.get('Calmar', 0):.0f}")
    print(f"최고 ({best_key[0]}, {best_key[1]}): PF={best_r['PF']:.2f}  "
          f"EV={best_r['EV_bp']:+.1f}bp  Calmar={best_r['Calmar']:.0f}")
    if prod_r.get("Calmar", 0) > 0:
        print(f"프로덕션 대비 최고: Calmar {best_r['Calmar']/prod_r['Calmar']:.2f}×")

    print(f"\n총 실행: {time.perf_counter() - t0:.1f}s")
    print(f"\nParity Notes:")
    print(f"  config.py 읽음 ✅  symbol_whitelist 사용 ✅  룩어헤드 없음 ✅")
    print(f"  universe: {len(data_15m)} syms  기간: {data_years:.2f}년")
    print(f"  STRD 슬롯 패리티: max_concurrent=5 ✅")


if __name__ == "__main__":
    main()
