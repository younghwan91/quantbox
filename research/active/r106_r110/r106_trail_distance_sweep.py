"""r106: Trail Distance Sweep — Rule #8 Compliance + Optimization.

목적:
  D_mom8 (0.2%)와 MA Cluster (0.3%) trail_distance가 Binance rule #8
  (callbackRate >= 0.5%)을 위반. 각 전략별 최적 trail_distance를 찾는다.
  trail_activate는 이미 OOS 검증됨 (r69/r53/r75) → 고정.

실험:
  T1: D_mom8 distance sweep [0.5%, 0.8%, 1.0%, 1.5%] (activate=1.0%)
  T2: S_d8 distance sweep [0.5%, 0.8%, 1.0%, 1.5%] (activate=1.5%)
  T3: MA Cluster distance sweep [0.5%, 0.8%, 1.0%, 1.5%, 2.0%] (activate=2.0%)

수용 기준:
  Winner = max Calmar, PF >= 1.2, bootstrap CI_lo >= 1.0, N >= 200
  Cliff flag: PF 변화 > 20% between adjacent grid points

사용:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r106_r110/r106_trail_distance_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (  # noqa: E402
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_4TF,
    bootstrap_pf,
    build_resampled,
    collect_ma_cluster_trades,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trailing,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import (  # noqa: E402
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading (r88 pattern)
# ══════════════════════════════════════════════════════════════════════════════


def load_all_data(cfg: VolStraddleConfig) -> tuple[dict, dict, dict, dict, float, int]:
    """Load 15m/30m/1h/4h from cache, precompute VR/momentum.

    Returns:
        (data_15m, data_30m, data_1h, data_4h, data_years, max_bars)
    """
    cache_dir = ROOT / "data" / "cache_extended"
    whitelist = set(cfg.symbol_whitelist)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    data_15m: dict = {}
    max_bars = 0
    raw: dict = {}
    for sym, (fpath, _nb) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        raw[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
        if len(c) > max_bars:
            max_bars = len(c)

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
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)

    print(f"  Loaded: {len(data_15m)} symbols, avg {avg_bars} bars, {data_years:.2f} years")
    return data_15m, data_30m, data_1h, data_4h, data_years, max_bars


# ══════════════════════════════════════════════════════════════════════════════
#  VR Trade Collection with Trailing (D_mom8 / S_d8)
# ══════════════════════════════════════════════════════════════════════════════


def _dedup(sym: str, trades: list[tuple[int, int, float]]) -> list[tuple[str, int, int, float]]:
    """Per-symbol time dedup: sort by entry, reject overlapping."""
    result: list[tuple[str, int, int, float]] = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_vr_trades_trailing(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    trail_activate: float,
    trail_distance: float,
) -> list[tuple[str, int, int, float]]:
    """Collect D_mom8 or S_d8 trades using trailing sim functions.

    Scans G_4TF (15m/30m/1h/4h), converts to 15m base, deduplicates.
    """
    sp = cfg.vr_short
    lp = cfg.vr_long
    vr_key = f"vr_{sp}_{lp}"
    is_straddle = strat == "S_d8"

    sl = cfg.s_sl if is_straddle else cfg.d_sl
    tp = cfg.s_tp if is_straddle else cfg.d_tp
    cost = COST_STRADDLE if is_straddle else COST_DIRECTIONAL

    tf_data_map: dict[int, dict] = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []

        for _tf_label, ratio, warmup, mh_d, mh_s in G_4TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            mh = mh_s if is_straddle else mh_d

            if is_straddle:
                sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
                p, e, x = sim_straddle_trailing(
                    c, h, lo, sig, sl, tp, mh, cost, trail_activate, trail_distance
                )
            else:
                sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trailing(
                    c, h, lo, sig, dire, sl, tp, mh, cost, trail_activate, trail_distance
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        all_trades.extend(_dedup(sym, sym_trades))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Sweep Engine
# ══════════════════════════════════════════════════════════════════════════════


def run_sweep_vr(
    label: str,
    distance_grid: list[float],
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    trail_activate: float,
    data_years: float,
    straddle_syms: frozenset[str] | None = None,
) -> list[dict]:
    """Sweep trail_distance for D_mom8 or S_d8."""
    results = []
    for dist in distance_grid:
        t0 = time.time()
        trades = collect_vr_trades_trailing(
            cfg, data_15m, data_30m, data_1h, data_4h, strat, trail_activate, dist
        )
        pnl = portfolio_sim(trades, cfg.max_concurrent, straddle_syms=straddle_syms)
        elapsed = time.time() - t0
        results.append(_compute_result(pnl, dist, data_years, elapsed))
    return results


def run_sweep_ma(
    distance_grid: list[float],
    data_15m: dict,
    data_4h: dict,
    trail_activate: float,
    max_concurrent: int,
    data_years: float,
) -> list[dict]:
    """Sweep trail_distance for MA Cluster."""
    results = []
    for dist in distance_grid:
        t0 = time.time()
        trades = collect_ma_cluster_trades(
            data_15m, data_4h, trail_activate=trail_activate, trail_distance=dist
        )
        pnl = portfolio_sim(trades, max_concurrent)
        elapsed = time.time() - t0
        results.append(_compute_result(pnl, dist, data_years, elapsed))
    return results


def _compute_result(pnl: np.ndarray, dist: float, data_years: float, elapsed: float) -> dict:
    """Compute metrics for one sweep point."""
    n = len(pnl)
    if n < 10:
        return {
            "distance": dist, "distance_pct": f"{dist*100:.1f}%",
            "n": n, "pf": 0.0, "calmar": 0.0, "max_dd_pct": 0.0,
            "wr_pct": 0.0, "ann_ret_pct": 0.0,
            "ci_lo": 0.0, "ci_hi": 0.0, "elapsed": elapsed, "pass": False,
        }
    m = compute_metrics(pnl, data_years)
    bs = bootstrap_pf(pnl)
    return {
        "distance": dist, "distance_pct": f"{dist*100:.1f}%",
        "n": n, "pf": m["PF"], "calmar": m["calmar"],
        "max_dd_pct": m["max_dd_pct"], "wr_pct": m["WR"],
        "ann_ret_pct": m["ann_ret_pct"],
        "ci_lo": bs["CI_lo"], "ci_hi": bs["CI_hi"],
        "elapsed": elapsed,
        "pass": m["PF"] >= 1.2 and bs["CI_lo"] >= 1.0 and n >= 200,
    }


def print_results(label: str, results: list[dict]) -> dict | None:
    """Print sweep results table and return the winner."""
    print(f"\n{'='*80}")
    print(f"  {label} — Trail Distance Sweep")
    print(f"{'='*80}")
    print(f"  {'Dist':>6}  {'N':>5}  {'PF':>6}  {'Calmar':>7}  {'MaxDD%':>7}  {'WR%':>6}  "
          f"{'AnnRet%':>8}  {'CI_lo':>6}  {'CI_hi':>6}  {'Pass':>5}  {'Cliff':>5}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")

    best = None
    for i, r in enumerate(results):
        cliff = ""
        if i > 0 and results[i - 1]["pf"] > 0:
            pf_change = abs(r["pf"] - results[i - 1]["pf"]) / results[i - 1]["pf"]
            if pf_change > 0.20:
                cliff = "!CLIFF"

        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['distance_pct']:>6}  {r['n']:>5}  {r['pf']:>6.3f}  {r['calmar']:>7.1f}  "
              f"{r['max_dd_pct']:>7.1f}  {r['wr_pct']:>6.1f}  {r.get('ann_ret_pct', 0):>8.1f}  "
              f"{r['ci_lo']:>6.3f}  {r['ci_hi']:>6.3f}  {status:>5}  {cliff:>5}")

        if r["pass"] and (best is None or r["calmar"] > best["calmar"]):
            best = r

    if best:
        print(f"\n  >> Winner: distance={best['distance_pct']}, "
              f"PF={best['pf']:.3f}, Calmar={best['calmar']:.1f}, "
              f"CI=[{best['ci_lo']:.3f}, {best['ci_hi']:.3f}]")
    else:
        print("\n  >> No distance meets acceptance criteria (PF>=1.2, CI_lo>=1.0, N>=200)")
        by_pf = sorted([r for r in results if r["n"] >= 50], key=lambda r: r["pf"], reverse=True)
        if by_pf:
            r = by_pf[0]
            print(f"     Best available: distance={r['distance_pct']}, "
                  f"PF={r['pf']:.3f}, Calmar={r['calmar']:.1f}, N={r['n']}")

    return best


# ══════════════════════════════════════════════════════════════════════════════
#  T1: D_mom8 Distance Sweep
# ══════════════════════════════════════════════════════════════════════════════


def run_d_mom8(
    cfg: VolStraddleConfig,
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    data_years: float,
) -> dict | None:
    """Sweep D_mom8 trail_distance with fixed activate=1.0% (r69 OOS validated)."""
    print("\n\n" + "=" * 80)
    print("  T1: D_mom8 — Trail Distance Sweep")
    print(f"  trail_activate = {cfg.d_trail_activate} (fixed, r69 OOS validated)")
    print(f"  Current distance = {cfg.d_trail_distance}")
    print("=" * 80)

    grid = [0.005, 0.008, 0.010, 0.015]
    results = run_sweep_vr(
        "D_mom8", grid, cfg,
        data_15m, data_30m, data_1h, data_4h,
        "D_mom8", cfg.d_trail_activate, data_years,
    )
    return print_results("D_mom8", results)


# ══════════════════════════════════════════════════════════════════════════════
#  T2: S_d8 Straddle Distance Sweep
# ══════════════════════════════════════════════════════════════════════════════


def run_s_d8(
    cfg: VolStraddleConfig,
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    data_years: float,
) -> dict | None:
    """Sweep S_d8 trail_distance with fixed activate=1.5% (r53 OOS validated)."""
    print("\n\n" + "=" * 80)
    print("  T2: S_d8 (Straddle) — Trail Distance Sweep")
    print(f"  trail_activate = {cfg.s_trail_activate} (fixed, r53 OOS validated)")
    print(f"  Current distance = {cfg.s_trail_distance} (already compliant)")
    print("=" * 80)

    grid = [0.005, 0.008, 0.010, 0.015]

    # Pre-collect straddle syms for slot counting
    baseline = collect_vr_trades_trailing(
        cfg, data_15m, data_30m, data_1h, data_4h,
        "S_d8", cfg.s_trail_activate, cfg.s_trail_distance,
    )
    strd_syms = frozenset(sym for sym, *_ in baseline)

    results = run_sweep_vr(
        "S_d8", grid, cfg,
        data_15m, data_30m, data_1h, data_4h,
        "S_d8", cfg.s_trail_activate, data_years,
        straddle_syms=strd_syms,
    )
    return print_results("S_d8 (Straddle)", results)


# ══════════════════════════════════════════════════════════════════════════════
#  T3: MA Cluster Distance Sweep
# ══════════════════════════════════════════════════════════════════════════════


def run_ma_cluster(
    cfg: VolStraddleConfig,
    data_15m: dict, data_4h: dict,
    data_years: float,
) -> dict | None:
    """Sweep MA Cluster trail_distance with fixed activate=2.0% (r75 OOS validated)."""
    print("\n\n" + "=" * 80)
    print("  T3: MA Cluster (4h) — Trail Distance Sweep")
    print(f"  trail_activate = {cfg.ma_cluster_trail_activate} (fixed, r75 OOS validated)")
    print(f"  Current distance = {cfg.ma_cluster_trail_distance}")
    print("  Note: 4h TF = 16x wider candles, wider grid range")
    print("=" * 80)

    grid = [0.005, 0.008, 0.010, 0.015, 0.020]
    results = run_sweep_ma(
        grid, data_15m, data_4h,
        cfg.ma_cluster_trail_activate, cfg.max_concurrent, data_years,
    )
    return print_results("MA Cluster (4h)", results)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: 2D Sweep (activate × distance) for D_mom8 and S_d8
# ══════════════════════════════════════════════════════════════════════════════


def run_2d_sweep_vr(
    label: str,
    activate_grid: list[float],
    distance_grid: list[float],
    cfg: VolStraddleConfig,
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    strat: str,
    data_years: float,
    straddle_syms: frozenset[str] | None = None,
) -> list[dict]:
    """2D sweep: activate × distance for D_mom8 or S_d8."""
    results = []
    for act in activate_grid:
        for dist in distance_grid:
            if dist >= act:
                # distance must be < activate (config validator)
                continue
            t0 = time.time()
            trades = collect_vr_trades_trailing(
                cfg, data_15m, data_30m, data_1h, data_4h, strat, act, dist
            )
            pnl = portfolio_sim(trades, cfg.max_concurrent, straddle_syms=straddle_syms)
            elapsed = time.time() - t0
            r = _compute_result(pnl, dist, data_years, elapsed)
            r["activate"] = act
            r["activate_pct"] = f"{act*100:.1f}%"
            r["headroom_pct"] = f"{(act - dist)*100:.1f}%"
            results.append(r)
    return results


def print_2d_results(label: str, results: list[dict]) -> dict | None:
    """Print 2D sweep results and return winner."""
    print(f"\n{'='*90}")
    print(f"  {label} — 2D Trail Sweep (activate x distance)")
    print(f"{'='*90}")
    print(f"  {'Act':>5}  {'Dist':>5}  {'Head':>5}  {'N':>5}  {'PF':>6}  {'Calmar':>7}  "
          f"{'MaxDD%':>7}  {'WR%':>6}  {'AnnRet%':>8}  {'CI_lo':>6}  {'Pass':>5}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}  "
          f"{'-'*7}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*5}")

    best = None
    for r in sorted(results, key=lambda x: (x["activate"], x["distance"])):
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['activate_pct']:>5}  {r['distance_pct']:>5}  {r['headroom_pct']:>5}  "
              f"{r['n']:>5}  {r['pf']:>6.3f}  {r['calmar']:>7.1f}  "
              f"{r['max_dd_pct']:>7.1f}  {r['wr_pct']:>6.1f}  {r.get('ann_ret_pct', 0):>8.1f}  "
              f"{r['ci_lo']:>6.3f}  {status:>5}")

        if r["pass"] and (best is None or r["calmar"] > best["calmar"]):
            best = r

    if best:
        print(f"\n  >> Winner: activate={best['activate_pct']}, distance={best['distance_pct']}, "
              f"headroom={best['headroom_pct']}")
        print(f"     PF={best['pf']:.3f}, Calmar={best['calmar']:.1f}, "
              f"CI=[{best['ci_lo']:.3f}, {best['ci_hi']:.3f}], N={best['n']}")
    else:
        print(f"\n  >> No combo meets acceptance criteria")
        by_pf = sorted([r for r in results if r["n"] >= 50], key=lambda r: r["pf"], reverse=True)
        if by_pf:
            r = by_pf[0]
            print(f"     Best: activate={r['activate_pct']}, distance={r['distance_pct']}, "
                  f"PF={r['pf']:.3f}, Calmar={r['calmar']:.1f}")
    return best


def run_phase3(
    cfg: VolStraddleConfig,
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    data_years: float,
) -> tuple[dict | None, dict | None]:
    """Phase 3: 2D activate × distance sweep for D_mom8 and S_d8."""
    print("\n\n" + "#" * 90)
    print("  PHASE 3: 2D Trail Sweep — Raising activate to create headroom")
    print("#" * 90)

    # D_mom8: activate was 1.0%, need more headroom above 0.5% distance floor
    # Sweep activate from 1.5% to 4.0% (gives 1.0-3.5% headroom)
    print("\n  D_mom8: current activate=1.0% gives only 0.5% headroom at 0.5% distance")
    print("  Sweeping activate=[1.5%, 2.0%, 2.5%, 3.0%, 4.0%] x distance=[0.5%, 0.8%, 1.0%]")

    d_results = run_2d_sweep_vr(
        "D_mom8",
        activate_grid=[0.015, 0.020, 0.025, 0.030, 0.040],
        distance_grid=[0.005, 0.008, 0.010],
        cfg=cfg,
        data_15m=data_15m, data_30m=data_30m, data_1h=data_1h, data_4h=data_4h,
        strat="D_mom8", data_years=data_years,
    )
    d_winner = print_2d_results("D_mom8", d_results)

    # S_d8: activate was 1.5%, same problem
    # Sweep activate from 2.0% to 4.0%
    print("\n  S_d8: current activate=1.5% gives 1.0% headroom at 0.5% distance")
    print("  Sweeping activate=[2.0%, 2.5%, 3.0%, 4.0%] x distance=[0.5%, 0.8%, 1.0%]")

    # Get straddle syms
    baseline = collect_vr_trades_trailing(
        cfg, data_15m, data_30m, data_1h, data_4h,
        "S_d8", cfg.s_trail_activate, cfg.s_trail_distance,
    )
    strd_syms = frozenset(sym for sym, *_ in baseline)

    s_results = run_2d_sweep_vr(
        "S_d8",
        activate_grid=[0.020, 0.025, 0.030, 0.040],
        distance_grid=[0.005, 0.008, 0.010],
        cfg=cfg,
        data_15m=data_15m, data_30m=data_30m, data_1h=data_1h, data_4h=data_4h,
        strat="S_d8", data_years=data_years,
        straddle_syms=strd_syms,
    )
    s_winner = print_2d_results("S_d8 (Straddle)", s_results)

    return d_winner, s_winner


def main() -> None:
    commit = get_commit()
    cfg = VolStraddleConfig()

    print("r106: Trail Distance Sweep — Rule #8 Compliance")
    print(f"  commit: {commit}")
    print(f"  date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  max_concurrent: {cfg.max_concurrent}")
    print("\n  Current config:")
    print(f"    D_mom8:     activate={cfg.d_trail_activate}, distance={cfg.d_trail_distance}")
    print(f"    S_d8:       activate={cfg.s_trail_activate}, distance={cfg.s_trail_distance}")
    print(f"    MA Cluster: activate={cfg.ma_cluster_trail_activate}, distance={cfg.ma_cluster_trail_distance}")

    print(f"\n{'='*80}")
    print("  Loading data...")
    t0 = time.time()
    data_15m, data_30m, data_1h, data_4h, data_years, _max_bars = load_all_data(cfg)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Run sweeps
    d_winner = run_d_mom8(cfg, data_15m, data_30m, data_1h, data_4h, data_years)
    s_winner = run_s_d8(cfg, data_15m, data_30m, data_1h, data_4h, data_years)
    ma_winner = run_ma_cluster(cfg, data_15m, data_4h, data_years)

    # Summary
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY — r106 Trail Distance Sweep ({commit})")
    print(f"{'='*80}")
    print("\n  Acceptance: max Calmar, PF >= 1.2, CI_lo >= 1.0, N >= 200\n")

    for name, winner, current in [
        ("D_mom8", d_winner, cfg.d_trail_distance),
        ("S_d8", s_winner, cfg.s_trail_distance),
        ("MA Cluster", ma_winner, cfg.ma_cluster_trail_distance),
    ]:
        if winner:
            change = "CHANGE" if winner["distance"] != current else "KEEP"
            print(f"  {name:>12}: distance={winner['distance_pct']} "
                  f"(current={current*100:.1f}%) -> {change}")
            print(f"               PF={winner['pf']:.3f} Calmar={winner['calmar']:.1f} "
                  f"CI=[{winner['ci_lo']:.3f}, {winner['ci_hi']:.3f}] N={winner['n']}")
        else:
            print(f"  {name:>12}: NO WINNER — all distances below acceptance criteria")
            print(f"               Action: keep current={current*100:.1f}%, investigate further")

    # Phase 3: 2D sweep for D_mom8 and S_d8 (activate too close to 0.5% floor)
    d2_winner, s2_winner = run_phase3(cfg, data_15m, data_30m, data_1h, data_4h, data_years)

    # Final summary
    print(f"\n\n{'='*90}")
    print(f"  FINAL SUMMARY — r106 Trail Distance + 2D Sweep ({commit})")
    print(f"{'='*90}")
    print("\n  Phase 2 (distance-only):")
    print(f"    MA Cluster: distance=0.5% -> KEEP (PF=1.911, Calmar=11.8)")
    print(f"    D_mom8: FAIL at all distances (activate=1.0% too close to 0.5% floor)")
    print(f"    S_d8:   FAIL at all distances (activate=1.5% too close to 0.5% floor)")
    print("\n  Phase 3 (2D activate x distance):")
    for name, w in [("D_mom8", d2_winner), ("S_d8", s2_winner)]:
        if w:
            print(f"    {name}: activate={w['activate_pct']}, distance={w['distance_pct']} "
                  f"(PF={w['pf']:.3f}, Calmar={w['calmar']:.1f}, CI_lo={w['ci_lo']:.3f})")
        else:
            print(f"    {name}: NO WINNER found in 2D sweep")

    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
