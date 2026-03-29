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

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

from quantbox.backtest.trade_sim import (  # noqa: E402
    bootstrap_pf,
    compute_metrics,
    portfolio_sim,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402

from r106_trail_distance_sweep import (  # noqa: E402
    collect_vr_trades_trailing,
    get_commit,
    load_all_data,
)

ACTIVATE_GRID = [0.02, 0.03, 0.04]
HOME_ADVANTAGE_THRESHOLD = 0.15  # 15% Calmar improvement required to override current


# ══════════════════════════════════════════════════════════════════════════════
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
        trades = collect_vr_trades_trailing(
            cfg, data_15m, data_30m, data_1h, data_4h,
            strat="D_mom8", trail_activate=act, trail_distance=trail_distance,
        )
        pnl = portfolio_sim(trades, cfg.max_concurrent)
        elapsed = time.time() - t0

        n = len(pnl)
        if n < 10:
            results.append({
                "activate": act, "n": n, "pf": 0.0, "calmar": 0.0,
                "max_dd_pct": 0.0, "wr_pct": 0.0, "ann_ret_pct": 0.0,
                "ci_lo": 0.0, "ci_hi": 0.0, "elapsed": elapsed, "pass": False,
            })
            continue

        m = compute_metrics(pnl, data_years)
        bs = bootstrap_pf(pnl)
        results.append({
            "activate": act, "n": n, "pf": m["PF"], "calmar": m["calmar"],
            "max_dd_pct": m["max_dd_pct"], "wr_pct": m["WR"],
            "ann_ret_pct": m["ann_ret_pct"],
            "ci_lo": bs["CI_lo"], "ci_hi": bs["CI_hi"],
            "elapsed": elapsed,
            "pass": m["PF"] >= 1.2 and bs["CI_lo"] >= 1.0 and n >= 200,
        })
    return results


def print_results(results: list[dict], current_activate: float, trail_distance: float) -> dict | None:
    """Print sweep results and apply home advantage logic."""
    print(f"\n{'='*85}")
    print(f"  D_mom8 Trail Activate Sweep (distance={trail_distance*100:.1f}% fixed)")
    print(f"{'='*85}")
    print(f"  {'Act':>5}  {'N':>5}  {'PF':>6}  {'Calmar':>7}  {'MaxDD%':>7}  {'WR%':>6}  "
          f"{'AnnRet%':>8}  {'CI_lo':>6}  {'CI_hi':>6}  {'Pass':>5}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}")

    best_calmar = None
    current_result = None

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        act_pct = f"{r['activate']*100:.1f}%"
        marker = " <-- current" if abs(r["activate"] - current_activate) < 0.001 else ""
        print(f"  {act_pct:>5}  {r['n']:>5}  {r['pf']:>6.3f}  {r['calmar']:>7.1f}  "
              f"{r['max_dd_pct']:>7.1f}  {r['wr_pct']:>6.1f}  {r.get('ann_ret_pct', 0):>8.1f}  "
              f"{r['ci_lo']:>6.3f}  {r['ci_hi']:>6.3f}  {status:>5}{marker}")

        if r["pass"] and (best_calmar is None or r["calmar"] > best_calmar["calmar"]):
            best_calmar = r
        if abs(r["activate"] - current_activate) < 0.001:
            current_result = r

    # Home advantage: current value gets preference unless alternative beats it significantly
    print(f"\n  --- Home Advantage Analysis ---")
    if current_result and current_result["pass"]:
        cur_pct = f"{current_activate*100:.0f}%"
        print(f"  Current {cur_pct}: Calmar={current_result['calmar']:.1f}, PF={current_result['pf']:.3f}")
        if best_calmar and best_calmar["activate"] != current_activate:
            improvement = (best_calmar["calmar"] - current_result["calmar"]) / current_result["calmar"]
            alt_pct = f"{best_calmar['activate']*100:.1f}%"
            print(f"  Best alternative: {alt_pct} Calmar={best_calmar['calmar']:.1f} "
                  f"(improvement: {improvement*100:+.1f}%)")
            if improvement > HOME_ADVANTAGE_THRESHOLD:
                print(f"  >> OVERRIDE: {alt_pct} beats {cur_pct} by {improvement*100:.1f}% "
                      f"(>{HOME_ADVANTAGE_THRESHOLD*100:.0f}% threshold)")
                return best_calmar
            else:
                print(f"  >> KEEP {cur_pct}: improvement {improvement*100:.1f}% "
                      f"< {HOME_ADVANTAGE_THRESHOLD*100:.0f}% threshold")
                return current_result
        else:
            print(f"  >> KEEP {cur_pct}: it is already the best")
            return current_result
    elif best_calmar:
        alt_pct = f"{best_calmar['activate']*100:.1f}%"
        print(f"  Current does not PASS. Best: {alt_pct}")
        print(f"  >> SELECT: {alt_pct} (PF={best_calmar['pf']:.3f}, Calmar={best_calmar['calmar']:.1f})")
        return best_calmar
    else:
        print(f"  >> NO WINNER: no activate value meets acceptance criteria")
        by_pf = sorted([r for r in results if r["n"] >= 50], key=lambda r: r["pf"], reverse=True)
        if by_pf:
            r = by_pf[0]
            print(f"     Best available: {r['activate']*100:.1f}%, PF={r['pf']:.3f}, "
                  f"Calmar={r['calmar']:.1f}, N={r['n']}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    cfg = VolStraddleConfig()
    trail_distance = cfg.d_trail_distance

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
    print(f"\n  Sweep: trail_activate = {[f'{a*100:.0f}%' for a in ACTIVATE_GRID]}, "
          f"trail_distance = {trail_distance*100:.1f}% (fixed, rule #8)")
    print(f"  Home advantage: override only if Calmar > {HOME_ADVANTAGE_THRESHOLD*100:.0f}% better")

    print(f"\n  Loading data...")
    wall_t0 = time.time()
    data_15m, data_30m, data_1h, data_4h, data_years, _max_bars = load_all_data(cfg)
    print(f"  Data loaded in {time.time() - wall_t0:.1f}s")

    results = run_activate_sweep(
        cfg, data_15m, data_30m, data_1h, data_4h,
        ACTIVATE_GRID, trail_distance, data_years,
    )

    winner = print_results(results, current_activate=cfg.d_trail_activate, trail_distance=trail_distance)

    # Final recommendation
    print(f"\n{'='*85}")
    print(f"  RECOMMENDATION")
    print(f"{'='*85}")
    if winner:
        if abs(winner["activate"] - cfg.d_trail_activate) < 0.001:
            print(f"  KEEP current d_trail_activate={cfg.d_trail_activate}")
            print(f"  No .env override needed (config.py default is correct)")
        else:
            print(f"  CHANGE d_trail_activate to {winner['activate']} ({winner['activate']*100:.1f}%)")
            print(f"  Add to .env: VS_D_TRAIL_ACTIVATE={winner['activate']}")
        print(f"\n  Winner: activate={winner['activate']*100:.1f}%, "
              f"PF={winner['pf']:.3f}, Calmar={winner['calmar']:.1f}, "
              f"CI=[{winner['ci_lo']:.3f}, {winner['ci_hi']:.3f}], N={winner['n']}")
    else:
        print(f"  WARNING: No activate value meets acceptance criteria")
        print(f"  KEEP current d_trail_activate={cfg.d_trail_activate} as safest option")

    print(f"\n  Total elapsed: {time.time() - wall_t0:.1f}s")


if __name__ == "__main__":
    main()
