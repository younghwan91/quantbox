"""r46 — Monte Carlo Simulation for QuantBox v14.3.0 Combined Portfolio

Bootstrap resampling of actual v14.3.0 trade returns with correct per-trade leverage.
  - VR (F5 TBR) trades: L/N = 3.0  (base_leverage=30, max_concurrent=10)
  - MA Cluster trades:  L/N = 2.1  (ma_cluster_leverage=21, max_concurrent=10)

Baseline: backtest_v14_3_0.py (canonical trade collection — do NOT reimplement).
Per-trade equity model: equity *= (1 + pnl * L/N)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import numba

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled,
    precompute,
    precompute_all_vr,
    collect_trades,
    portfolio_sim_with_details,
)
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m,
    collect_d_trades_trailing,
    collect_d_trades_with_tbr,
    VR_LN,
    MA_LN,
    MA_CLUSTER_PCT,
    MA_MIN_DURATION,
    MA_MAX_DURATION,
    MA_MIN_MOMENTUM,
    MA_SL,
    MA_TP,
    MA_TF_RESTRICTION,
)
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r31_r40.scaling_r38 import CFG_V13


# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 2_000.0  # USD starting balance
N_SIMS: int = 10_000              # Monte Carlo paths
SEED: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Numba helper
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _compute_max_streaks(is_loss: np.ndarray, n_sims: int, n_trades: int) -> np.ndarray:
    """Max consecutive losing streak per simulation path."""
    result = np.zeros(n_sims, dtype=np.int64)
    for i in range(n_sims):
        streak = 0
        best = 0
        for j in range(n_trades):
            if is_loss[i, j]:
                streak += 1
                if streak > best:
                    best = streak
            else:
                streak = 0
        result[i] = best
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo core
# ═══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    pnl_arr: np.ndarray,
    ln_arr: np.ndarray,
    initial_capital: float,
    n_sims: int,
    trades_per_year: int,
) -> None:
    """Bootstrap Monte Carlo with per-trade leverage.

    Args:
        pnl_arr: Per-trade raw PnL fractions (e.g. 0.035 = +3.5%).
        ln_arr: Per-trade L/N ratio (3.0 for VR, 2.1 for MA).
        initial_capital: Starting balance ($).
        n_sims: Number of simulation paths.
        trades_per_year: Trades drawn per simulation (annualized).
    """
    n_pool = len(pnl_arr)
    rng = np.random.default_rng(seed=SEED)

    print(f"\n{'=' * 78}")
    print(f"  MONTE CARLO SIMULATION — v14.3.0 Combined Portfolio")
    print(f"  {n_sims:,} paths × {trades_per_year} trades/year  |  pool: {n_pool} observed trades")
    print(f"{'=' * 78}")
    print(f"  VR L/N={VR_LN:.1f}  MA L/N={MA_LN:.1f}  "
          f"(blended pool avg L/N={np.mean(ln_arr):.3f})")
    mean_pct = np.mean(pnl_arr) * 100
    std_pct = np.std(pnl_arr) * 100
    print(f"  Pool EV: {mean_pct:+.3f}%  Std: {std_pct:.3f}%")
    print(f"  Running {n_sims:,} simulations (vectorized)...\n")

    # Draw trade indices: (n_sims, trades_per_year)
    idx = rng.integers(0, n_pool, size=(n_sims, trades_per_year))

    sampled_pnl = pnl_arr[idx]        # (n_sims, trades_per_year)
    sampled_ln  = ln_arr[idx]         # (n_sims, trades_per_year)

    # Per-trade equity growth: equity *= (1 + pnl * L/N)
    growth_factors = 1.0 + sampled_pnl * sampled_ln   # (n_sims, trades_per_year)

    # Cumulative equity paths
    cum_equity = initial_capital * np.cumprod(growth_factors, axis=1)  # (n_sims, trades_per_year)

    final_balances = cum_equity[:, -1]
    final_returns  = (final_balances / initial_capital - 1.0) * 100

    # Max drawdown per path
    running_max  = np.maximum.accumulate(cum_equity, axis=1)
    drawdowns    = (running_max - cum_equity) / running_max
    max_drawdowns = np.max(drawdowns, axis=1)

    # Ruin: equity < 50% of initial at any point
    ruin_mask  = np.any(cum_equity < initial_capital * 0.5, axis=1)
    ruin_count = int(np.sum(ruin_mask))

    # Loss streaks (numba)
    is_loss = sampled_pnl <= 0
    max_streaks = _compute_max_streaks(is_loss, n_sims, trades_per_year)

    # ── Output ────────────────────────────────────────────────────────────
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print(f"  {'─' * 65}")
    print(f"  FINAL BALANCE DISTRIBUTION (after {trades_per_year} trades, 1 year)")
    print(f"  {'─' * 65}")
    for p in pctiles:
        bal = np.percentile(final_balances, p)
        ret = np.percentile(final_returns, p)
        print(f"    P{p:2d}:  ${bal:>14,.0f}  ({ret:+,.0f}%)")
    print(f"    Mean: ${np.mean(final_balances):>14,.0f}  ({np.mean(final_returns):+,.0f}%)")

    print(f"\n  {'─' * 65}")
    print(f"  MAX DRAWDOWN DISTRIBUTION (per-year)")
    print(f"  {'─' * 65}")
    for p in [5, 25, 50, 75, 95]:
        dd = np.percentile(max_drawdowns, p) * 100
        print(f"    P{p:2d}:  {dd:.1f}%")
    print(f"    Mean: {np.mean(max_drawdowns)*100:.1f}%")

    print(f"\n  {'─' * 65}")
    print(f"  RISK METRICS")
    print(f"  {'─' * 65}")
    prob_profit = np.mean(final_returns > 0) * 100
    prob_2x     = np.mean(final_balances >= initial_capital * 2) * 100
    prob_5x     = np.mean(final_balances >= initial_capital * 5) * 100
    prob_10x    = np.mean(final_balances >= initial_capital * 10) * 100
    prob_100x   = np.mean(final_balances >= initial_capital * 100) * 100
    print(f"    P(profit):     {prob_profit:.1f}%")
    print(f"    P(2x capital): {prob_2x:.1f}%    (≥${initial_capital*2:,.0f})")
    print(f"    P(5x capital): {prob_5x:.1f}%    (≥${initial_capital*5:,.0f})")
    print(f"    P(10x):        {prob_10x:.1f}%    (≥${initial_capital*10:,.0f})")
    print(f"    P(100x):       {prob_100x:.1f}%   (≥${initial_capital*100:,.0f})")
    print(f"    P(ruin <50%):  {ruin_count/n_sims*100:.2f}%")

    print(f"\n  {'─' * 65}")
    print(f"  CONSECUTIVE LOSS STREAKS")
    print(f"  {'─' * 65}")
    for p in [50, 75, 90, 95, 99]:
        cl = np.percentile(max_streaks, p)
        print(f"    P{p:2d}:  {cl:.0f} trades")
    print(f"    Max:  {np.max(max_streaks):.0f} trades")

    p5_bal  = np.percentile(final_balances, 5)
    p50_bal = np.percentile(final_balances, 50)
    p95_bal = np.percentile(final_balances, 95)
    p95_dd  = np.percentile(max_drawdowns, 95) * 100

    print(f"\n  {'=' * 65}")
    print(f"  90% CONFIDENCE INTERVAL ({trades_per_year} trades/year, per-trade L/N leverage)")
    print(f"  {'=' * 65}")
    print(f"    Lower (P5):   ${p5_bal:>12,.0f}  ({(p5_bal/initial_capital-1)*100:+,.0f}%)")
    print(f"    Median (P50): ${p50_bal:>12,.0f}  ({(p50_bal/initial_capital-1)*100:+,.0f}%)")
    print(f"    Upper (P95):  ${p95_bal:>12,.0f}  ({(p95_bal/initial_capital-1)*100:+,.0f}%)")
    print(f"    Worst DD (P95): {p95_dd:.1f}%")
    print(f"    Ruin probability: {ruin_count/n_sims*100:.2f}%")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    print(f"\n{'=' * 78}")
    print(f"  r46 — Monte Carlo Simulation  |  QuantBox v14.3.0")
    print(f"  Commit: {commit}  |  Capital: ${INITIAL_CAPITAL:,.0f}  |  Sims: {N_SIMS:,}")
    print(f"{'=' * 78}")

    # ── Load data (identical to backtest_v14_3_0.py) ──────────────────────
    print("\n  Loading 15m data...")
    data_15m = load_15m()
    if not data_15m:
        print("❌ No data. Check data/cache_extended/")
        return

    avg_bars   = int(np.mean([len(v["close"]) for v in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  "
          f"{data_years:.2f} years  |  load: {time.time()-t0:.1f}s")

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)

    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    # ── Collect trades (same as backtest_v14_3_0.py Phase 2 & 3) ─────────
    print("\n  Collecting MA Cluster trades...")
    ma_trades = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT,
        min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION,
        min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=MA_TF_RESTRICTION,
    )
    print(f"  MA raw: {len(ma_trades)} trades")

    print("\n  Collecting VR Squeeze (F5 TBR) trades...")
    d_trades_filtered, n_blocked = collect_d_trades_with_tbr(
        CFG_V13, data_15m, data_1h, data_4h,
    )
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    vr_trades = d_trades_filtered + s_trades
    print(f"  VR raw: {len(vr_trades)} trades  ({n_blocked} blocked by F5 TBR)")

    # ── Combined portfolio (slot contention) ──────────────────────────────
    print("\n  Running portfolio_sim (10 slots)...")
    comb_trades = vr_trades + ma_trades
    comb_pnl, comb_accepted = portfolio_sim_with_details(comb_trades, 10)

    n_total  = len(comb_accepted)
    print(f"  Accepted: {n_total} trades  |  {n_total/data_years:.0f} trades/year")

    # ── Tag each accepted trade as VR or MA → assign L/N ─────────────────
    vr_key_set: set[tuple[str, int, int]] = {
        (sym, entry, exit_b) for sym, entry, exit_b, _ in vr_trades
    }

    pnl_list: list[float] = []
    ln_list:  list[float] = []
    n_vr_accepted = 0
    n_ma_accepted = 0

    for sym, entry, exit_b, pnl_val in comb_accepted:
        pnl_list.append(float(pnl_val))
        if (sym, entry, exit_b) in vr_key_set:
            ln_list.append(VR_LN)
            n_vr_accepted += 1
        else:
            ln_list.append(MA_LN)
            n_ma_accepted += 1

    pnl_arr = np.array(pnl_list, dtype=np.float64)
    ln_arr  = np.array(ln_list,  dtype=np.float64)

    blended_ln = np.mean(ln_arr)
    print(f"  VR accepted: {n_vr_accepted}  MA accepted: {n_ma_accepted}  "
          f"blended L/N: {blended_ln:.3f}")

    trades_per_year = max(1, round(n_total / data_years))
    print(f"  Trades per MC year: {trades_per_year}")

    # ── Run Monte Carlo ───────────────────────────────────────────────────
    run_monte_carlo(
        pnl_arr=pnl_arr,
        ln_arr=ln_arr,
        initial_capital=INITIAL_CAPITAL,
        n_sims=N_SIMS,
        trades_per_year=trades_per_year,
    )

    print(f"  Total runtime: {time.time()-t0:.1f}s  |  Commit: {commit}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
