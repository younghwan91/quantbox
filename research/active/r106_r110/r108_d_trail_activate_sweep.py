"""r108: D_mom8 Trail Activate Sweep — Straddle OFF 전환 최적값 검증.

목적:
  Straddle 비활성화 후 D_mom8 전용 운영 시 d_trail_activate 최적값을 찾는다.
  r106에서 4%로 결정했으나 현재 low-vol 장에서 너무 보수적일 수 있음.
  4%는 r106 검증값이므로 "home advantage" 적용:
  대체값은 Calmar 15% 이상 개선 시에만 채택.

실험:
  d_trail_activate sweep [2%, 3%, 4%] with d_trail_distance=0.5% (fixed, rule #8)
  strat = D_mom8 only (straddle disabled)

수용 기준:
  Winner = max Calmar, PF >= 1.2, bootstrap CI_lo >= 1.0, N >= 200
  Home advantage: 4% override only if alternative Calmar > 4%_Calmar * 1.15

사용:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r106_r110/r108_d_trail_activate_sweep.py
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
    G_4TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import (  # noqa: E402
    gen_vr_signal,
    make_direction_momentum,
)

# Reuse r106 data loading
from r106_trail_distance_sweep import load_all_data  # noqa: E402


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  D_mom8 Trade Collection (from r106, trail_activate as parameter)
# ════════════��═════════════════════════════════════════════════════════════════


def _dedup(sym: str, trades: list[tuple[int, int, float]]) -> list[tuple[str, int, int, float]]:
    result: list[tuple[str, int, int, float]] = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_d_mom8_trailing(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    trail_activate: float,
    trail_distance: float,
) -> list[tuple[str, int, int, float]]:
    """Collect D_mom8 trades with specified trail params, scanning G_4TF."""
    sp = cfg.vr_short
    lp = cfg.vr_long
    vr_key = f"vr_{sp}_{lp}"

    tf_data_map: dict[int, dict] = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []

        for _tf_label, ratio, warmup, mh_d, _mh_s in G_4TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire,
                cfg.d_sl, cfg.d_tp, mh_d,
                COST_DIRECTIONAL, trail_activate, trail_distance,
            )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        all_trades.extend(_dedup(sym, sym_trades))

    return all_trades


# ════════════════════════════════════��═════════════════════════════════════════
#  Sweep Engine
# ══════════════════════════════════════════════════════════════════════════════


def run_activate_sweep(
    cfg: VolStraddleConfig,
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    activate_grid: list[float],
    trail_distance: float,
    data_years: float,
) -> list[dict]:
    """Sweep d_trail_activate for D_mom8 with fixed trail_distance."""
    results = []
    for act in activate_grid:
        t0 = time.time()
        trades = collect_d_mom8_trailing(
            cfg, data_15m, data_30m, data_1h, data_4h, act, trail_distance,
        )
        pnl = portfolio_sim(trades, cfg.max_concurrent)
        elapsed = time.time() - t0

        n = len(pnl)
        if n < 10:
            results.append({
                "activate": act, "activate_pct": f"{act*100:.1f}%",
                "n": n, "pf": 0.0, "calmar": 0.0, "max_dd_pct": 0.0,
                "wr_pct": 0.0, "ann_ret_pct": 0.0,
                "ci_lo": 0.0, "ci_hi": 0.0, "elapsed": elapsed, "pass": False,
            })
            continue

        m = compute_metrics(pnl, data_years)
        bs = bootstrap_pf(pnl)
        results.append({
            "activate": act, "activate_pct": f"{act*100:.1f}%",
            "n": n, "pf": m["PF"], "calmar": m["calmar"],
            "max_dd_pct": m["max_dd_pct"], "wr_pct": m["WR"],
            "ann_ret_pct": m["ann_ret_pct"],
            "ci_lo": bs["CI_lo"], "ci_hi": bs["CI_hi"],
            "elapsed": elapsed,
            "pass": m["PF"] >= 1.2 and bs["CI_lo"] >= 1.0 and n >= 200,
        })
    return results


def print_results(results: list[dict], current_activate: float) -> dict | None:
    """Print sweep results and apply home advantage logic."""
    print(f"\n{'='*85}")
    print(f"  D_mom8 Trail Activate Sweep (distance=0.5% fixed)")
    print(f"{'='*85}")
    print(f"  {'Act':>5}  {'N':>5}  {'PF':>6}  {'Calmar':>7}  {'MaxDD%':>7}  {'WR%':>6}  "
          f"{'AnnRet%':>8}  {'CI_lo':>6}  {'CI_hi':>6}  {'Pass':>5}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}")

    best_calmar = None
    current_result = None

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        marker = " <-- current" if abs(r["activate"] - current_activate) < 0.001 else ""
        print(f"  {r['activate_pct']:>5}  {r['n']:>5}  {r['pf']:>6.3f}  {r['calmar']:>7.1f}  "
              f"{r['max_dd_pct']:>7.1f}  {r['wr_pct']:>6.1f}  {r.get('ann_ret_pct', 0):>8.1f}  "
              f"{r['ci_lo']:>6.3f}  {r['ci_hi']:>6.3f}  {status:>5}{marker}")

        if r["pass"] and (best_calmar is None or r["calmar"] > best_calmar["calmar"]):
            best_calmar = r
        if abs(r["activate"] - current_activate) < 0.001:
            current_result = r

    # Home advantage: 4% gets preference unless alternative beats it by >15%
    print(f"\n  --- Home Advantage Analysis ---")
    if current_result and current_result["pass"]:
        print(f"  Current 4%: Calmar={current_result['calmar']:.1f}, PF={current_result['pf']:.3f}")
        if best_calmar and best_calmar["activate"] != current_activate:
            improvement = (best_calmar["calmar"] - current_result["calmar"]) / current_result["calmar"]
            print(f"  Best alternative: {best_calmar['activate_pct']} Calmar={best_calmar['calmar']:.1f} "
                  f"(improvement: {improvement*100:+.1f}%)")
            if improvement > 0.15:
                print(f"  >> OVERRIDE: {best_calmar['activate_pct']} beats 4% by {improvement*100:.1f}% (>15% threshold)")
                return best_calmar
            else:
                print(f"  >> KEEP 4%: improvement {improvement*100:.1f}% < 15% threshold")
                return current_result
        else:
            print(f"  >> KEEP 4%: it is already the best")
            return current_result
    elif best_calmar:
        print(f"  Current 4% does not PASS. Best: {best_calmar['activate_pct']}")
        print(f"  >> SELECT: {best_calmar['activate_pct']} (PF={best_calmar['pf']:.3f}, "
              f"Calmar={best_calmar['calmar']:.1f})")
        return best_calmar
    else:
        print(f"  >> NO WINNER: no activate value meets acceptance criteria")
        by_pf = sorted([r for r in results if r["n"] >= 50], key=lambda r: r["pf"], reverse=True)
        if by_pf:
            r = by_pf[0]
            print(f"     Best available: {r['activate_pct']}, PF={r['pf']:.3f}, "
                  f"Calmar={r['calmar']:.1f}, N={r['n']}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    cfg = VolStraddleConfig()

    print("r108: D_mom8 Trail Activate Sweep — Straddle OFF Optimization")
    print(f"  commit: {commit}")
    print(f"  date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  straddle_enabled: {cfg.straddle_enabled}")
    print(f"  max_concurrent: {cfg.max_concurrent}")
    print(f"\n  Current D_mom8 config:")
    print(f"    d_vr_threshold:  {cfg.d_vr_threshold}")
    print(f"    d_sl:            {cfg.d_sl}")
    print(f"    d_trail_activate: {cfg.d_trail_activate}")
    print(f"    d_trail_distance: {cfg.d_trail_distance}")
    print(f"\n  Sweep: trail_activate = [2%, 3%, 4%], trail_distance = 0.5% (fixed, rule #8)")
    print(f"  Home advantage: 4% override only if alternative Calmar > 15% better")

    print(f"\n  Loading data...")
    t0 = time.time()
    data_15m, data_30m, data_1h, data_4h, data_years, _max_bars = load_all_data(cfg)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Run sweep
    activate_grid = [0.02, 0.03, 0.04]
    trail_distance = 0.005  # Rule #8 floor

    results = run_activate_sweep(
        cfg, data_15m, data_30m, data_1h, data_4h,
        activate_grid, trail_distance, data_years,
    )

    winner = print_results(results, current_activate=cfg.d_trail_activate)

    # Final recommendation
    print(f"\n{'='*85}")
    print(f"  RECOMMENDATION")
    print(f"{'='*85}")
    if winner:
        if abs(winner["activate"] - cfg.d_trail_activate) < 0.001:
            print(f"  KEEP current d_trail_activate={cfg.d_trail_activate} (4%)")
            print(f"  No .env override needed (config.py default is correct)")
        else:
            print(f"  CHANGE d_trail_activate to {winner['activate']} ({winner['activate_pct']})")
            print(f"  Add to .env: VS_D_TRAIL_ACTIVATE={winner['activate']}")
        print(f"\n  Winner: activate={winner['activate_pct']}, "
              f"PF={winner['pf']:.3f}, Calmar={winner['calmar']:.1f}, "
              f"CI=[{winner['ci_lo']:.3f}, {winner['ci_hi']:.3f}], N={winner['n']}")
    else:
        print(f"  WARNING: No activate value meets acceptance criteria")
        print(f"  KEEP current d_trail_activate={cfg.d_trail_activate} as safest option")

    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
