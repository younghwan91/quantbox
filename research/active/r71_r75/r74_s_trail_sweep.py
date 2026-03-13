"""r74: S_d8 trail_activate 스윕 — straddle 최적 trail 파라미터 탐색.

r73 confirmed S_d8 trail PF=4.948 (N=7730) as Phase 0 baseline using the r66
canonical trail_activate=0.005 (0.5%).  Live config changed d_trail_activate
from 0.005→0.010 in r68/r69 for D_mom8.  S_d8 straddle dynamics differ; this
script validates whether 1.0% (or another value) is better for straddle.

Sweep:
  Phase 1: trail_activate ∈ [0.003, 0.005, 0.007, 0.010, 0.012, 0.015]
  Phase 2: trail_distance ∈ [0.001, 0.002, 0.003, 0.004] @ best trail_activate

Usage:
    uv run python research/active/r71_r75/r74_s_trail_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal
from research.active.r41_r45.r43_exit_improvement import sim_straddle_trailing

# ── Sweep grid ────────────────────────────────────────────────────────────────
TRAIL_ACTIVATE_GRID = [0.003, 0.005, 0.007, 0.010, 0.012, 0.015]
TRAIL_DISTANCE_GRID = [0.001, 0.002, 0.003, 0.004]

# r73 / r66 canonical baseline (Phase 0 reference).
# r73 used cfg.s_trail_activate = 0.015 → N=7730, PF=4.948.
# Note: task description stated 0.005 but that was d_trail_activate's old value;
# s_trail_activate has been 0.015 since r53.
BASELINE_TRAIL_ACT  = 0.015
BASELINE_N_REF      = 7730
BASELINE_PF_REF     = 4.948

# Candidate under consideration: d_trail_activate was optimised to 1.0% in r68/r69.
# We mark 1.0% with * to see if that value would also suit S_d8 straddle.
LIVE_TRAIL_ACT = 0.010

# Warmup for 30m TF
WARMUP_30m: int = 100


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def _get_g4tf(cfg: VolStraddleConfig) -> list:
    """G_4TF: 15m/30m/1h/4h — matches r66/r73 canonical."""
    mh_30m = cfg.max_hold_15m // 2
    tf_30m = ("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)
    return [tf_30m] + list(G_3TF)


def load_15m(whitelist: list[str]) -> dict:
    """Load 60 whitelist symbols with _offset alignment."""
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
    """Per-symbol time dedup: reject overlapping (entry/exit) trades."""
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_s_trail(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
    trail_activate: float,
    trail_distance: float,
) -> list[tuple[str, int, int, float]]:
    """S_d8 trail-aware trade collection (G_4TF: 15m/30m/1h/4h, matching r66/r73)."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    g4tf = _get_g4tf(cfg)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl    = cfg.s_sl
    tp    = cfg.s_tp
    s_vr  = cfg.s_vr_threshold
    s_dur = cfg.s_squeeze_dur

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, _mhd, mh_s in g4tf:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE,
                trail_activate, trail_distance,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def run_one(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
    trail_activate: float,
    trail_distance: float,
    data_years: float,
) -> dict:
    """Run one (trail_activate, trail_distance) combination and return metrics."""
    trades = collect_s_trail(
        data_15m, data_30m, data_1h, data_4h, cfg, trail_activate, trail_distance,
    )
    pnl = portfolio_sim(trades, cfg.max_concurrent)
    m   = compute_metrics(pnl, data_years)
    b   = bootstrap_pf(pnl, 3000)
    return {**m, "boot": b, "trail_activate": trail_activate, "trail_distance": trail_distance}


def print_row(r: dict, mark: str = "") -> None:
    ta  = r["trail_activate"] * 100
    td  = r["trail_distance"] * 100
    tag = f"{ta:.1f}%{mark}"
    print(
        f"  {tag:<12}  {r['N']:>6}  {r['PF']:>7.3f}  {r['WR']:>5.1f}%  "
        f"{r['EV_bp']:>+7.1f}  {r['max_dd_pct']:>6.1f}%  {r['calmar']:>8.1f}"
    )


def main() -> None:
    commit = get_commit()
    t0 = time.time()

    print(f"\n{'='*78}")
    print(f"  r74: S_d8 trail_activate 스윕  [{commit}]")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*78}")

    cfg = VolStraddleConfig()
    vr_ln = cfg.base_leverage / cfg.max_concurrent

    print(f"\n  Config (from VolStraddleConfig):")
    print(f"    s_sl={cfg.s_sl}  s_tp={cfg.s_tp}  "
          f"s_vr={cfg.s_vr_threshold}  s_dur={cfg.s_squeeze_dur}")
    print(f"    s_trail_activate (cfg)={cfg.s_trail_activate}  "
          f"s_trail_distance (cfg)={cfg.s_trail_distance}")
    print(f"    max_concurrent={cfg.max_concurrent}  L/N={vr_ln:.1f}")
    print(f"    f5_tbr_enabled={cfg.f5_tbr_enabled}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data (60 symbols, symbol_whitelist)...")
    data_15m = load_15m(cfg.symbol_whitelist)
    if not data_15m:
        print("❌ No data loaded — check data/cache_extended/")
        return

    data_30m = build_resampled(data_15m, 2)
    data_1h  = build_resampled(data_15m, 4)
    data_4h  = build_resampled(data_15m, 16)

    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg.vr_short, cfg.vr_long)])
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)

    bars_list  = [len(d["close"]) for d in data_15m.values()]
    avg_bars   = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  "
          f"{data_years:.2f} years  |  load: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 0: Baseline validation (trail_activate=0.005, r66 canonical)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print(f"  PHASE 0: BASELINE (trail_activate={BASELINE_TRAIL_ACT*100:.1f}%, r66 canonical)")
    print(f"  Expected: N≈{BASELINE_N_REF}, PF≈{BASELINE_PF_REF}")
    print(f"{'─'*78}")

    b0 = run_one(
        data_15m, data_30m, data_1h, data_4h, cfg,
        trail_activate=BASELINE_TRAIL_ACT,
        trail_distance=cfg.s_trail_distance,
        data_years=data_years,
    )
    n_drift  = abs(b0["N"]  - BASELINE_N_REF)  / BASELINE_N_REF  * 100
    pf_drift = abs(b0["PF"] - BASELINE_PF_REF) / BASELINE_PF_REF * 100
    status   = "✅ PASS" if n_drift < 5.0 and pf_drift < 5.0 else "⚠️  DRIFT"

    print(f"  Result:  N={b0['N']}  PF={b0['PF']:.3f}  WR={b0['WR']:.1f}%  "
          f"EV={b0['EV_bp']:+.1f}bp")
    print(f"  vs r73:  N_drift={n_drift:.1f}%  PF_drift={pf_drift:.1f}%  {status}")

    if n_drift >= 10.0 or pf_drift >= 10.0:
        print("\n  ❌ Drift >10% — stopping. Check data/config alignment with r73.")
        return

    if n_drift >= 5.0 or pf_drift >= 5.0:
        print("  ⚠️  Drift 5-10% — proceeding with caution. Investigate if results look off.")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: trail_activate sweep (fixed trail_distance=cfg.s_trail_distance)
    # ══════════════════════════════════════════════════════════════════════
    fixed_trail_dist = cfg.s_trail_distance  # 0.002 (live)

    print(f"\n{'─'*78}")
    print(f"  PHASE 1: trail_activate SWEEP  (trail_distance fixed={fixed_trail_dist*100:.1f}%)")
    print(f"  Values: {[f'{v*100:.1f}%' for v in TRAIL_ACTIVATE_GRID]}")
    print(f"  * = current live (d_trail_activate={LIVE_TRAIL_ACT*100:.1f}%)")
    print(f"{'─'*78}")

    hdr = (f"  {'trail_act':<12}  {'N':>6}  {'PF':>7}  {'WR':>6}  "
           f"{'EV(bp)':>7}  {'MaxDD':>7}  {'Calmar':>8}")
    print(hdr)
    print(f"  {'─'*70}")

    phase1_results: list[dict] = []
    for ta in TRAIL_ACTIVATE_GRID:
        r = run_one(
            data_15m, data_30m, data_1h, data_4h, cfg,
            trail_activate=ta,
            trail_distance=fixed_trail_dist,
            data_years=data_years,
        )
        phase1_results.append(r)
        mark = " *" if abs(ta - LIVE_TRAIL_ACT) < 1e-6 else ""
        print_row(r, mark)

    phase1_sorted = sorted(phase1_results, key=lambda x: x["calmar"], reverse=True)
    best_ta   = phase1_sorted[0]["trail_activate"]
    second_ta = phase1_sorted[1]["trail_activate"]

    print(f"\n  Sorted by Calmar:")
    print(hdr)
    print(f"  {'─'*70}")
    for r in phase1_sorted:
        mark = " *" if abs(r["trail_activate"] - LIVE_TRAIL_ACT) < 1e-6 else ""
        print_row(r, mark)

    print(f"\n  → Best: trail_activate={best_ta*100:.1f}%  "
          f"(PF={phase1_sorted[0]['PF']:.3f}, Calmar={phase1_sorted[0]['calmar']:.1f})")
    print(f"  → 2nd:  trail_activate={second_ta*100:.1f}%  "
          f"(PF={phase1_sorted[1]['PF']:.3f}, Calmar={phase1_sorted[1]['calmar']:.1f})")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: trail_distance sweep (top 2 trail_activate from Phase 1)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*78}")
    print(f"  PHASE 2: trail_distance SWEEP  (top 2 trail_activate values)")
    print(f"  Values: {[f'{v*100:.1f}%' for v in TRAIL_DISTANCE_GRID]}")
    print(f"{'─'*78}")

    phase2_results: dict[float, list[dict]] = {}
    for ta in [best_ta, second_ta]:
        print(f"\n  [trail_activate={ta*100:.1f}%]")
        print(hdr)
        print(f"  {'─'*70}")
        rows: list[dict] = []
        for td in TRAIL_DISTANCE_GRID:
            r = run_one(
                data_15m, data_30m, data_1h, data_4h, cfg,
                trail_activate=ta,
                trail_distance=td,
                data_years=data_years,
            )
            rows.append(r)
            mark = " *" if abs(td - fixed_trail_dist) < 1e-6 else ""
            # print trail_distance instead of trail_activate
            ta_pct = r["trail_activate"] * 100
            td_pct = r["trail_distance"] * 100
            tag    = f"{td_pct:.1f}%{mark}"
            print(
                f"  {tag:<12}  {r['N']:>6}  {r['PF']:>7.3f}  {r['WR']:>5.1f}%  "
                f"{r['EV_bp']:>+7.1f}  {r['max_dd_pct']:>6.1f}%  {r['calmar']:>8.1f}"
            )
        phase2_results[ta] = rows

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")

    live_r = next(
        (r for r in phase1_results if abs(r["trail_activate"] - LIVE_TRAIL_ACT) < 1e-6),
        None,
    )
    best_p1 = phase1_sorted[0]

    print(f"\n  Phase 0 baseline  (trail_act={BASELINE_TRAIL_ACT*100:.1f}%): "
          f"N={b0['N']}  PF={b0['PF']:.3f}  Calmar={b0['calmar']:.1f}")
    if live_r:
        print(f"  Live config*       (trail_act={LIVE_TRAIL_ACT*100:.1f}%): "
              f"N={live_r['N']}  PF={live_r['PF']:.3f}  Calmar={live_r['calmar']:.1f}")
    print(f"  Phase 1 best      (trail_act={best_ta*100:.1f}%):  "
          f"N={best_p1['N']}  PF={best_p1['PF']:.3f}  Calmar={best_p1['calmar']:.1f}")

    if live_r:
        pf_delta = best_p1["PF"] - live_r["PF"]
        print(f"\n  vs live config:  PF delta={pf_delta:+.3f}  "
              f"Calmar delta={best_p1['calmar']-live_r['calmar']:+.1f}")
        if abs(best_ta - LIVE_TRAIL_ACT) < 1e-6:
            print("  → ✅ Current live 1.0% IS the best value for straddle. 변경 불필요.")
        elif best_p1["PF"] > live_r["PF"] * 1.02:
            print(f"  → 📈 best={best_ta*100:.1f}% PF≥2% better — 변경 권장")
        else:
            print(f"  → 📊 Best={best_ta*100:.1f}% marginally better (<2% PF diff) — 현상 유지 권장")

    # Phase 2 best overall
    all_p2: list[dict] = [r for rows in phase2_results.values() for r in rows]
    best_p2 = max(all_p2, key=lambda x: x["calmar"])
    print(f"\n  Phase 2 best: trail_activate={best_p2['trail_activate']*100:.1f}%"
          f"  trail_distance={best_p2['trail_distance']*100:.1f}%"
          f"  PF={best_p2['PF']:.3f}  Calmar={best_p2['calmar']:.1f}")

    print(f"\n  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}"
          f"  Commit: {commit}  Runtime: {time.time()-t0:.1f}s")
    print(f"  Note: 1x compound. Multiply EV by L/N={vr_ln:.1f} for production sizing.")
    print(f"  * = d_trail_activate={LIVE_TRAIL_ACT*100:.1f}% (r68/r69 D_mom8 optimum — candidate for S_d8)")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
