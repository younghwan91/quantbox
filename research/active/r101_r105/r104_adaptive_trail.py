"""r104: Adaptive Trail — VR-Binned trail_distance sweep for S_d8 straddle.

Goal:
  Test whether trail_distance should vary by entry VR level.
  Low VR (deep squeeze) → tighter trail; rising VR (breakout) → wider trail.

Method:
  1. Collect all S_d8 straddle trades with entry VR recorded per trade
  2. Bin trades by entry VR (4 bins, S_d8 only: all entries VR < 0.57)
  3. For each bin, sweep trail_distance [0.0005, 0.001, 0.0015, 0.002, 0.003]
  4. Check: monotonic PF > 1.0 across all bins, cliff sensitivity < 30%
  5. Apply r101 bias correction (stratified: trail=15bp, SL=2bp)

Trigger gate (from consensus plan):
  Only proceed if straddle trail exits show >20bp live-vs-backtest divergence.
  This script runs regardless to establish the baseline; deployment is gated.

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r101_r105/r104_adaptive_trail.py
"""

from __future__ import annotations

import sys
import time
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
DATA_YEARS = 6.0
MIN_N_PER_BIN = 50  # Consensus plan: sparse bins fall back to fixed 0.001

cfg = VolStraddleConfig()

# ── VR bins (S_d8 straddle only: entries at VR < s_vr_threshold=0.57) ──
VR_BIN_EDGES = [0.0, 0.30, 0.40, 0.50, 0.57]
VR_BIN_LABELS = ["<0.30", "0.30-0.40", "0.40-0.50", "0.50-0.57"]

# ── Trail distance sweep grid ──
TRAIL_DIST_GRID = [0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003]
PROD_DIST = cfg.s_trail_distance  # 0.001
PROD_ACT = cfg.s_trail_activate  # 0.015

# ── Bias correction (stratified per r101) ──
BIAS_TRAIL_BP = 15.0
BIAS_SL_BP = 2.0
# straddle: ~54% trail exits, ~39% SL exits, ~7% other
# weighted average: 0.54*15 + 0.39*2 + 0.07*3 ≈ 9.1bp
BIAS_WEIGHTED_BP = 9.1


# ── Data loading (reuse r85 pattern) ──────────────────────────────────────

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
    for rec in sorted(trades, key=lambda t: t[0]):
        e, x = rec[0], rec[1]
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, *rec))
            accepted.append((e, x))
    return result


# ── Trade collection with entry VR ────────────────────────────────────────

def collect_strd_with_vr(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    trail_act: float,
    trail_dist: float,
) -> list[tuple[str, int, int, float, float]]:
    """Collect S_d8 trades with entry VR per trade.

    Returns: [(sym, entry_15m, exit_15m, pnl, entry_vr), ...]
    Entry VR is looked up from the precomputed VR array at the entry bar index.
    """
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    mh_30m = cfg.max_hold_15m // 2
    g4tf = [
        ("30m", 2, WARMUP_30m, mh_30m * 2),
        ("15m", 1, WARMUP_15m, cfg.max_hold_15m * 2),
        ("1h", 4, WARMUP_1h, cfg.max_hold_15m // 4 * 2),
        ("4h", 16, WARMUP_4h, cfg.max_hold_15m // 16 * 2),
    ]
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl = cfg.s_sl
    tp = cfg.s_tp

    all_trades: list[tuple[str, int, int, float, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float, float]] = []
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
                entry_bar = int(e[k])
                # Look up VR at entry bar from precomputed array
                entry_vr = float(vr[entry_bar]) if entry_bar < len(vr) else 0.0
                sym_trades.append((
                    entry_bar * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                    entry_vr,
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def bin_trades(
    trades: list[tuple[str, int, int, float, float]],
) -> dict[str, list[tuple[str, int, int, float]]]:
    """Bin trades by entry VR into the defined bins."""
    bins: dict[str, list[tuple[str, int, int, float]]] = {lbl: [] for lbl in VR_BIN_LABELS}
    for sym, e, x, pnl, vr in trades:
        for i in range(len(VR_BIN_EDGES) - 1):
            if VR_BIN_EDGES[i] <= vr < VR_BIN_EDGES[i + 1]:
                bins[VR_BIN_LABELS[i]].append((sym, e, x, pnl))
                break
    return bins


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    whitelist = cfg.symbol_whitelist

    print("=" * 70)
    print("r104: Adaptive Trail — VR-Binned trail_distance Sweep (S_d8)")
    print("=" * 70)
    print(f"Production: s_trail_activate={PROD_ACT}  s_trail_distance={PROD_DIST}")
    print(f"VR bins: {VR_BIN_LABELS}")
    print(f"Trail distance grid: {TRAIL_DIST_GRID}")
    print(f"Min N per bin: {MIN_N_PER_BIN}")

    # ── Load & precompute ──
    print("\nLoading data...")
    data_15m = load_15m(whitelist)
    sample_bars = max(len(d["close"]) for d in data_15m.values())
    data_years = sample_bars / 35040.0
    print(f"  {len(data_15m)} symbols  |  {sample_bars:,} bars  |  {data_years:.2f}yr")

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    vr_pairs = [(cfg.vr_short, cfg.vr_long)]
    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], vr_pairs)
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)
    print(f"  Precompute done: {time.perf_counter() - t0:.1f}s\n")

    # ── Section 1: VR distribution at production params ──
    print("─" * 70)
    print("SECTION 1: VR Distribution (production trail params)")
    print("─" * 70)

    trades_prod = collect_strd_with_vr(data_15m, data_30m, data_1h, data_4h, PROD_ACT, PROD_DIST)
    print(f"  Total trades: {len(trades_prod)}")

    binned_prod = bin_trades(trades_prod)
    print(f"\n  {'VR Bin':>12} {'N':>6} {'%':>6} {'Avg PnL(bp)':>12} {'WR%':>6}")
    print("  " + "-" * 46)
    for lbl in VR_BIN_LABELS:
        bt = binned_prod[lbl]
        n = len(bt)
        pct = n / max(len(trades_prod), 1) * 100
        if n > 0:
            pnls = np.array([t[3] for t in bt])
            avg_bp = pnls.mean() * 10000
            wr = (pnls > 0).sum() / n * 100
            sufficient = "✓" if n >= MIN_N_PER_BIN else "✗ sparse"
            print(f"  {lbl:>12} {n:>6} {pct:>5.1f}% {avg_bp:>+11.1f} {wr:>5.1f}  {sufficient}")
        else:
            print(f"  {lbl:>12} {n:>6} {pct:>5.1f}%       —     —")

    # ── Section 2: Per-bin trail_distance sweep ──
    print("\n" + "─" * 70)
    print("SECTION 2: Per-Bin Trail Distance Sweep")
    print("─" * 70)

    # For each trail_distance, collect trades, bin, compute per-bin metrics
    sweep_results: dict[str, dict[float, dict]] = {lbl: {} for lbl in VR_BIN_LABELS}
    combined_results: dict[float, dict] = {}

    for td in TRAIL_DIST_GRID:
        t1 = time.perf_counter()
        trades = collect_strd_with_vr(data_15m, data_30m, data_1h, data_4h, PROD_ACT, td)
        binned = bin_trades(trades)

        for lbl in VR_BIN_LABELS:
            bt = binned[lbl]
            n = len(bt)
            if n < 10:
                sweep_results[lbl][td] = {"N": n, "PF": 0.0, "EV_bp": 0.0}
                continue
            pnl_arr = portfolio_sim(bt, max_concurrent=MAX_CONCURRENT)
            m = compute_metrics(pnl_arr, data_years)
            sweep_results[lbl][td] = {
                "N": m["N"], "PF": m["PF"], "WR": m["WR"],
                "EV_bp": m.get("EV_bp", 0), "MaxDD": m.get("max_dd_pct", 0),
            }

        # Combined (all bins together)
        if trades:
            # Use only (sym, entry, exit, pnl) for portfolio_sim
            combined = [(t[0], t[1], t[2], t[3]) for t in trades]
            pnl_all = portfolio_sim(combined, max_concurrent=MAX_CONCURRENT)
            m_all = compute_metrics(pnl_all, data_years)
            boot = bootstrap_pf(pnl_all, n_boot=500)
            combined_results[td] = {
                "N": m_all["N"], "PF": m_all["PF"], "WR": m_all["WR"],
                "EV_bp": m_all.get("EV_bp", 0), "MaxDD": m_all.get("max_dd_pct", 0),
                "boot_p95": boot.get("p_ge_1", 0),
            }
        dt = time.perf_counter() - t1
        print(f"  td={td:.4f}: N={len(trades):>5}  PF={combined_results.get(td, {}).get('PF', 0):.3f}  ({dt:.1f}s)")

    # ── Heatmap ──
    print("\n" + "─" * 70)
    print("SECTION 3: Heatmap — PF(VR_bin, trail_distance)")
    print("─" * 70)

    header = f"  {'VR Bin':>12}" + "".join(f" {td:.4f}" for td in TRAIL_DIST_GRID)
    print(header)
    print("  " + "-" * (12 + 8 * len(TRAIL_DIST_GRID)))

    for lbl in VR_BIN_LABELS:
        row = f"  {lbl:>12}"
        for td in TRAIL_DIST_GRID:
            r = sweep_results[lbl].get(td, {})
            n = r.get("N", 0)
            pf = r.get("PF", 0)
            if n < MIN_N_PER_BIN:
                row += f"  {'—':>5}n"
            else:
                row += f" {pf:>6.2f}"
        print(row)

    print(f"\n  {'Combined':>12}", end="")
    for td in TRAIL_DIST_GRID:
        r = combined_results.get(td, {})
        print(f" {r.get('PF', 0):>6.2f}", end="")
    print()

    # ── Section 4: Per-bin optimal + cliff sensitivity ──
    print("\n" + "─" * 70)
    print("SECTION 4: Per-Bin Optimal + Cliff Sensitivity")
    print("─" * 70)

    optimal_per_bin: dict[str, float] = {}
    all_monotonic = True

    for lbl in VR_BIN_LABELS:
        results = sweep_results[lbl]
        # Find optimal (highest PF with N >= MIN_N)
        valid = {td: r for td, r in results.items() if r.get("N", 0) >= MIN_N_PER_BIN}
        if not valid:
            print(f"  {lbl}: SPARSE (N < {MIN_N_PER_BIN}) → fallback to fixed {PROD_DIST}")
            optimal_per_bin[lbl] = PROD_DIST
            continue

        best_td = max(valid, key=lambda td: valid[td].get("PF", 0))
        best_pf = valid[best_td]["PF"]
        optimal_per_bin[lbl] = best_td

        # Cliff sensitivity: check PF at ±1 grid step
        idx = TRAIL_DIST_GRID.index(best_td)
        neighbors = []
        if idx > 0:
            neighbors.append(TRAIL_DIST_GRID[idx - 1])
        if idx < len(TRAIL_DIST_GRID) - 1:
            neighbors.append(TRAIL_DIST_GRID[idx + 1])

        worst_neighbor_pf = min(
            valid.get(n, {}).get("PF", 0) for n in neighbors if n in valid
        ) if neighbors else best_pf
        cliff_pct = abs(best_pf - worst_neighbor_pf) / max(best_pf, 0.001) * 100

        monotonic = all(valid.get(n, {}).get("PF", 0) > 1.0 for n in neighbors if n in valid)
        if not monotonic:
            all_monotonic = False

        print(f"  {lbl}: optimal td={best_td:.4f}  PF={best_pf:.3f}  "
              f"cliff={cliff_pct:.0f}%  neighbors_PF>1={monotonic}")

    # ── Section 5: Bias-corrected combined results ──
    print("\n" + "─" * 70)
    print("SECTION 5: Bias-Corrected Combined Results")
    print("─" * 70)

    print(f"  {'td':>8} {'N':>5} {'PF_raw':>7} {'PF_corr':>8} {'EV_raw':>8} {'EV_corr':>9} {'P(PF≥1)':>8} {'MaxDD':>7}")
    print("  " + "-" * 65)
    for td in TRAIL_DIST_GRID:
        r = combined_results.get(td, {})
        n = r.get("N", 0)
        pf = r.get("PF", 0)
        ev = r.get("EV_bp", 0)
        boot = r.get("boot_p95", 0)
        maxdd = r.get("MaxDD", 0)
        # Bias correction: subtract weighted average
        ev_corr = ev - BIAS_WEIGHTED_BP
        # Approximate PF correction: PF_corr ≈ PF * (1 - bias/EV) if EV > 0
        pf_corr = pf * (ev_corr / ev) if ev > 0 else pf
        marker = " ◀ PROD" if abs(td - PROD_DIST) < 1e-6 else ""
        print(f"  {td:>8.4f} {n:>5} {pf:>7.3f} {pf_corr:>8.3f} {ev:>+7.1f} {ev_corr:>+8.1f} {boot:>7.0f}% {maxdd:>6.1f}%{marker}")

    # ── Decision ──
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    print(f"  Optimal per bin: {optimal_per_bin}")
    print(f"  All bins PF > 1.0 (monotonic): {all_monotonic}")
    print(f"  Bias correction: {BIAS_WEIGHTED_BP:.1f}bp weighted average")

    # Check if adaptive is better than fixed
    prod_pf = combined_results.get(PROD_DIST, {}).get("PF", 0)
    best_combined_td = max(combined_results, key=lambda td: combined_results[td].get("PF", 0))
    best_combined_pf = combined_results[best_combined_td]["PF"]

    if best_combined_pf > prod_pf * 1.05:  # 5% improvement threshold
        print(f"\n  📈 Adaptive trail shows {(best_combined_pf/prod_pf - 1)*100:.1f}% PF improvement")
        print(f"     Best combined: td={best_combined_td:.4f} PF={best_combined_pf:.3f}")
        print(f"     Production:    td={PROD_DIST:.4f} PF={prod_pf:.3f}")
        print(f"  → Proceed to r105 OOS validation")
    else:
        print(f"\n  ➡️  No significant improvement over production ({prod_pf:.3f})")
        print(f"  → Keep fixed td={PROD_DIST}. Track 2 closed as 'not needed'.")

    print(f"\n  Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
