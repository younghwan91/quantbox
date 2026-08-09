"""r81: STRD s_sl / s_trail 비율 최적화 스윕.

**배경:**
  현재 s_sl=0.015 = s_trail_activate=0.015 → 동일한 이동폭.
  이기는 쪽 MFE가 1.76%+ 되어야 손익분기 (비용 포함).
  라이브 2/2 완성 페어 모두 net 음수 (MFE가 손익분기 미달).

**핵심 질문:**
  1. s_sl을 0.010~0.014로 낮추면 손익분기 개선되는가?
  2. s_trail_activate는 0.015 고정 (r77 cliff) → sl만 변경하면 cliff 영향 없는가?
  3. s_trail_distance 0.001 vs 0.002 비교

사용:
    uv run python research/active/r81_r85/r81_strd_sl_trail_ratio.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    G_3TF,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal

CACHE_DIR = ROOT / "data" / "cache_extended"
WARMUP_30m = 100
cfg = VolStraddleConfig()
MAX_CONCURRENT = 5  # STRD standalone: 1 straddle=2 slots → max 5 pairs


def load_all_data():
    """Load and precompute all timeframe data (same as r80)."""
    whitelist = set(cfg.symbol_whitelist)
    raw: dict = {}
    max_bars = 0
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        npz = np.load(str(f), allow_pickle=False)
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        raw[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
        if len(c) > max_bars:
            max_bars = len(c)

    data_15m: dict = {}
    for sym, d in raw.items():
        d["_offset"] = max_bars - len(d["close"])
        data_15m[sym] = d

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg.vr_short, cfg.vr_long)])
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)

    bars_list = [len(d["close"]) for d in data_15m.values()]
    data_years = int(np.mean(bars_list)) / (4 * 24 * 365)

    return data_15m, data_30m, data_1h, data_4h, data_years


def _dedup(sym: str, trades: list) -> list:
    result = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_strd(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    *,
    sl: float,
    trail_activate: float,
    trail_distance: float,
    s_vr: float | None = None,
    s_dur: int | None = None,
) -> list:
    """Collect STRD trades with explicit SL/trail params (multi-TF)."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    mh_30m = cfg.max_hold_15m // 2
    tf_30m = ("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)
    g4tf = [tf_30m] + list(G_3TF)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    tp = cfg.s_tp
    _s_vr = s_vr if s_vr is not None else cfg.s_vr_threshold
    _s_dur = s_dur if s_dur is not None else cfg.s_squeeze_dur
    mh_15m = cfg.max_hold_15m

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, _mhd, _mhs in g4tf:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, _s_vr, _s_dur, warmup)
            mh_d = int(mh_15m // ratio)
            mh_s = max(1, int(mh_d * 2))
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


def run_sweep(
    data_15m, data_30m, data_1h, data_4h, data_years,
    sl, trail_activate, trail_distance,
) -> dict:
    """Run one STRD config and return metrics."""
    trades = collect_strd(
        data_15m, data_30m, data_1h, data_4h,
        sl=sl, trail_activate=trail_activate, trail_distance=trail_distance,
    )
    if not trades:
        return {"N": 0, "PF": 0, "WR": 0, "EV_bp": 0, "max_dd_pct": 0, "calmar": 0}
    pnl = portfolio_sim(trades, MAX_CONCURRENT)
    return compute_metrics(pnl, data_years)


def main() -> None:
    t0 = time.time()
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(f"r81: STRD s_sl/trail ratio sweep — commit {sha}")
    print(f"Config: s_vr={cfg.s_vr_threshold}, s_dur={cfg.s_squeeze_dur}")
    print(f"Universe: {len(cfg.symbol_whitelist)} syms, max_concurrent={MAX_CONCURRENT}")
    print()

    print("Loading & precomputing data...")
    data_15m, data_30m, data_1h, data_4h, data_years = load_all_data()
    print(f"Loaded {len(data_15m)} symbols, {data_years:.2f} years")
    print()

    # ── Phase 0: Baseline ─────────────────────────────────────────────────
    print("=" * 80)
    print("Phase 0: Baseline (current production params)")
    print("=" * 80)
    baseline = run_sweep(
        data_15m, data_30m, data_1h, data_4h, data_years,
        sl=cfg.s_sl, trail_activate=cfg.s_trail_activate, trail_distance=cfg.s_trail_distance,
    )
    print(f"  s_sl={cfg.s_sl} trail_act={cfg.s_trail_activate} trail_dist={cfg.s_trail_distance}")
    print(f"  N={baseline['N']} PF={baseline['PF']:.3f} WR={baseline['WR']*100:.1f}% "
          f"EV={baseline['EV_bp']:.1f}bp MaxDD={baseline['max_dd_pct']*100:.1f}% "
          f"Calmar={baseline['calmar']:.1f}")
    assert abs(baseline["N"] - 6290) / 6290 < 0.15, f"N drift: {baseline['N']}"
    print("  ✅ Baseline verified")
    print()

    # ── Phase 1: s_sl sweep (trail_activate=0.015 fixed) ──────────────────
    SL_GRID = [0.008, 0.010, 0.012, 0.013, 0.014, 0.015, 0.018, 0.020]
    print("=" * 80)
    print("Phase 1: s_sl sweep (s_trail_activate=0.015 fixed, trail_dist=0.001)")
    print("=" * 80)
    print(f"{'s_sl':>8s} {'N':>6s} {'PF':>8s} {'WR':>6s} {'EV(bp)':>8s} {'max_dd_pct':>8s} {'calmar':>10s} {'BEP%':>6s} {'★':>2s}")
    print("-" * 65)

    sl_results = []
    for sl in SL_GRID:
        m = run_sweep(data_15m, data_30m, data_1h, data_4h, data_years,
                       sl=sl, trail_activate=0.015, trail_distance=0.001)
        bep = (sl + COST_STRADDLE + 0.001) * 100
        star = "★" if sl == cfg.s_sl else ""
        print(f"{sl:>8.3f} {m['N']:>6d} {m['PF']:>8.3f} {m['WR']*100:>5.1f}% "
              f"{m['EV_bp']:>7.1f} {m['max_dd_pct']*100:>7.1f}% {m['calmar']:>10.1f} {bep:>5.2f}% {star}")
        sl_results.append({"s_sl": sl, "bep": bep, **m})
    print()

    # ── Phase 2: s_trail_activate sweep (cliff test) ──────────────────────
    TRAIL_ACT_GRID = [0.010, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.020]
    print("=" * 80)
    print("Phase 2: s_trail_activate sweep (s_sl=0.015 fixed, cliff test)")
    print("=" * 80)
    print(f"{'trail_act':>10s} {'N':>6s} {'PF':>8s} {'WR':>6s} {'EV(bp)':>8s} {'max_dd_pct':>8s} {'calmar':>10s} {'★':>2s}")
    print("-" * 65)

    for ta in TRAIL_ACT_GRID:
        m = run_sweep(data_15m, data_30m, data_1h, data_4h, data_years,
                       sl=0.015, trail_activate=ta, trail_distance=0.001)
        star = "★" if ta == cfg.s_trail_activate else ""
        print(f"{ta:>10.3f} {m['N']:>6d} {m['PF']:>8.3f} {m['WR']*100:>5.1f}% "
              f"{m['EV_bp']:>7.1f} {m['max_dd_pct']*100:>7.1f}% {m['calmar']:>10.1f} {star}")
    print()

    # ── Phase 3: 2D sweep (s_sl × s_trail_activate) ──────────────────────
    sl_focus = [0.010, 0.012, 0.013, 0.014, 0.015]
    ta_focus = [0.013, 0.014, 0.015, 0.016, 0.017]
    print("=" * 80)
    print("Phase 3: 2D sweep — s_sl × s_trail_activate (trail_dist=0.001)")
    print("=" * 80)
    print(f"{'s_sl':>6s} {'t_act':>6s} {'N':>6s} {'PF':>8s} {'WR':>6s} {'EV(bp)':>8s} "
          f"{'max_dd_pct':>8s} {'calmar':>10s} {'BEP%':>6s}")
    print("-" * 72)

    sweep_2d = []
    for sl in sl_focus:
        for ta in ta_focus:
            m = run_sweep(data_15m, data_30m, data_1h, data_4h, data_years,
                           sl=sl, trail_activate=ta, trail_distance=0.001)
            bep = (sl + COST_STRADDLE + 0.001) * 100
            star = " ★" if sl == cfg.s_sl and ta == cfg.s_trail_activate else ""
            print(f"{sl:>6.3f} {ta:>6.3f} {m['N']:>6d} {m['PF']:>8.3f} {m['WR']*100:>5.1f}% "
                  f"{m['EV_bp']:>7.1f} {m['max_dd_pct']*100:>7.1f}% {m['calmar']:>10.1f} {bep:>5.2f}%{star}")
            sweep_2d.append({"s_sl": sl, "trail_act": ta, "bep": bep, **m})
    print()

    # ── Phase 4: trail_distance comparison ────────────────────────────────
    print("=" * 80)
    print("Phase 4: trail_distance 0.001 vs 0.002 (production s_sl/trail_act)")
    print("=" * 80)
    for td in [0.001, 0.002, 0.003]:
        m = run_sweep(data_15m, data_30m, data_1h, data_4h, data_years,
                       sl=cfg.s_sl, trail_activate=cfg.s_trail_activate, trail_distance=td)
        star = " ★" if td == cfg.s_trail_distance else ""
        print(f"  trail_dist={td:.3f}: N={m['N']} PF={m['PF']:.3f} WR={m['WR']*100:.1f}% "
              f"EV={m['EV_bp']:.1f}bp MaxDD={m['max_dd_pct']*100:.1f}% Calmar={m['calmar']:.1f}{star}")

    best_sl_row = max(sl_results, key=lambda x: x["calmar"])
    if best_sl_row["s_sl"] != cfg.s_sl:
        print(f"\n  Also with best s_sl={best_sl_row['s_sl']} from Phase 1:")
        for td in [0.001, 0.002]:
            m = run_sweep(data_15m, data_30m, data_1h, data_4h, data_years,
                           sl=best_sl_row["s_sl"], trail_activate=0.015, trail_distance=td)
            print(f"  s_sl={best_sl_row['s_sl']:.3f} trail_dist={td:.3f}: N={m['N']} "
                  f"PF={m['PF']:.3f} Calmar={m['calmar']:.1f}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    best_2d = max(sweep_2d, key=lambda x: x["calmar"])
    print(f"\nBaseline: s_sl={cfg.s_sl} trail_act={cfg.s_trail_activate} trail_dist={cfg.s_trail_distance}")
    print(f"  N={baseline['N']} PF={baseline['PF']:.3f} Calmar={baseline['calmar']:.1f} BEP=1.76%")

    print(f"\nBest 2D (Calmar): s_sl={best_2d['s_sl']} trail_act={best_2d['trail_act']}")
    print(f"  N={best_2d['N']} PF={best_2d['PF']:.3f} Calmar={best_2d['calmar']:.1f} BEP={best_2d['bep']:.2f}%")

    if baseline["calmar"] > 0 and best_2d["calmar"] > 0:
        calmar_chg = (best_2d["calmar"] / baseline["calmar"] - 1) * 100
        pf_chg = (best_2d["PF"] / baseline["PF"] - 1) * 100
        print(f"  Calmar: {calmar_chg:+.1f}%, PF: {pf_chg:+.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
