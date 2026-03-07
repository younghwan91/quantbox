"""r46b — Monte Carlo Visualization for QuantBox v14.3.0

Generates two figures:
  1. Return distribution histograms: 일/주/월/년 (2×2 grid)
  2. Equity curve fan: percentile bands over 1 year (P5/P25/P50/P75/P95)

Saves to docs/backtest_results/202603/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled, precompute, precompute_all_vr,
    collect_trades, portfolio_sim_with_details,
)
from research.active.r41_r45.backtest_v14_3_0 import (
    load_15m, collect_d_trades_trailing, collect_d_trades_with_tbr,
    VR_LN, MA_LN,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
)
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r31_r40.scaling_r38 import CFG_V13

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 2_000.0
N_SIMS: int = 10_000
SEED: int = 42
OUT_DIR = ROOT / "docs" / "backtest_results" / "202603"

PALETTE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "grid":    "#2a2d3a",
    "text":    "#e0e0e0",
    "muted":   "#7a7d8e",
    "green":   "#00c896",
    "red":     "#ff4f6a",
    "blue":    "#4fa3ff",
    "yellow":  "#ffd166",
    "purple":  "#c07bff",
    "fan": ["#1a3a5c", "#1e5799", "#2980b9", "#5dade2", "#aed6f1"],  # P5→P95
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sim(pnl_arr, ln_arr, n_trades, rng):
    """Return (n_sims, n_trades) cumulative equity paths and final returns %."""
    idx = rng.integers(0, len(pnl_arr), size=(N_SIMS, n_trades))
    growth = 1.0 + pnl_arr[idx] * ln_arr[idx]
    cum_eq = INITIAL_CAPITAL * np.cumprod(growth, axis=1)
    ret_pct = (cum_eq[:, -1] / INITIAL_CAPITAL - 1.0) * 100
    return cum_eq, ret_pct


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Return distribution histograms (일/주/월/년)
# ═══════════════════════════════════════════════════════════════════════════════

def _hist_panel(ax, ret_pct, label, n_trades, color):
    """Draw a single horizon histogram with KDE and percentile markers."""
    p5, p25, p50, p75, p95 = np.percentile(ret_pct, [5, 25, 50, 75, 95])
    p_profit = np.mean(ret_pct > 0) * 100

    # Clip for readability (year has extreme right tail)
    lo = np.percentile(ret_pct, 1)
    hi = np.percentile(ret_pct, 99)
    clipped = np.clip(ret_pct, lo, hi)

    bins = 80
    neg = clipped[clipped <= 0]
    pos = clipped[clipped > 0]

    if len(neg):
        ax.hist(neg, bins=bins, color=PALETTE["red"],   alpha=0.55,
                density=True, zorder=2, label="_nolegend_")
    if len(pos):
        ax.hist(pos, bins=bins, color=PALETTE["green"], alpha=0.55,
                density=True, zorder=2, label="_nolegend_")

    # KDE
    try:
        kde = gaussian_kde(clipped, bw_method=0.15)
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, kde(xs), color=color, lw=2, zorder=3)
    except Exception:
        pass

    # Percentile vlines
    for pv, ls, lbl in [(p5, "--", "P5"), (p50, "-", "P50"), (p95, "--", "P95")]:
        ax.axvline(pv, color=PALETTE["yellow"], lw=1.2, ls=ls, zorder=4, alpha=0.85)

    ax.axvline(0, color=PALETTE["muted"], lw=1, ls=":", zorder=1)

    # Annotations
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
    for pv, lbl in [(p5, f"P5\n{p5:+.0f}%"), (p50, f"P50\n{p50:+.0f}%"), (p95, f"P95\n{p95:+.0f}%")]:
        ax.text(pv, ax.get_ylim()[1] * 0.92, lbl, ha="center", va="top",
                fontsize=7.5, color=PALETTE["yellow"],
                bbox=dict(fc=PALETTE["panel"], ec="none", alpha=0.7, pad=2))

    # Title & labels
    ax.set_title(f"{label}  ({n_trades} trades)\nP(+) = {p_profit:.1f}%",
                 color=PALETTE["text"], fontsize=11, pad=6)
    ax.set_xlabel("Return (%)", color=PALETTE["muted"], fontsize=9)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.set_facecolor(PALETTE["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])
    ax.yaxis.set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.5, zorder=0)

    # x-axis formatting
    if abs(hi) > 500:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f}K%" if abs(v) >= 1000 else f"{v:+.0f}%"
        ))


def fig1_distributions(horizons_data: dict, out_path: Path) -> None:
    labels = ["1 Day", "1 Week", "1 Month", "1 Year"]
    colors = [PALETTE["blue"], PALETTE["purple"], PALETTE["green"], PALETTE["yellow"]]
    keys   = ["day", "week", "month", "year"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "QuantBox v14.3.0 — Monte Carlo Return Distributions\n"
        f"10,000 paths | Bootstrap resampling | VR L/N=3.0 / MA L/N=2.1 (blended 2.626) | Capital $2,000",
        color=PALETTE["text"], fontsize=12, y=0.99
    )

    for ax, key, label, color in zip(axes.flat, keys, labels, colors):
        ret_pct, n_trades = horizons_data[key]
        _hist_panel(ax, ret_pct, label, n_trades, color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Equity curve fan (1 year)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_equity_fan(cum_eq_year: np.ndarray, trades_per_year: int, out_path: Path) -> None:
    """Fan chart: percentile equity bands across 785 trades (1 year)."""
    pctiles = [5, 25, 50, 75, 95]
    fan_colors = PALETTE["fan"]   # dark→light = P5→P95

    x = np.arange(trades_per_year + 1)
    # Prepend initial capital column
    full = np.concatenate(
        [np.full((cum_eq_year.shape[0], 1), INITIAL_CAPITAL), cum_eq_year], axis=1
    )
    bands = {p: np.percentile(full, p, axis=0) for p in pctiles}

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    # Shaded regions
    ax.fill_between(x, bands[5],  bands[25], color=fan_colors[0], alpha=0.55, label="P5–P25")
    ax.fill_between(x, bands[25], bands[50], color=fan_colors[1], alpha=0.55, label="P25–P50")
    ax.fill_between(x, bands[50], bands[75], color=fan_colors[2], alpha=0.55, label="P50–P75")
    ax.fill_between(x, bands[75], bands[95], color=fan_colors[3], alpha=0.55, label="P75–P95")

    # Percentile lines
    ls_map = {5: "--", 25: "-.", 50: "-", 75: "-.", 95: "--"}
    lw_map = {5: 1.5, 25: 1.5, 50: 2.5, 75: 1.5, 95: 1.5}
    for i, p in enumerate(pctiles):
        ax.plot(x, bands[p], color=fan_colors[i], lw=lw_map[p], ls=ls_map[p],
                label=f"P{p}", zorder=5)

    # Initial capital line
    ax.axhline(INITIAL_CAPITAL, color=PALETTE["muted"], lw=1, ls=":", zorder=1,
               label=f"Initial ${INITIAL_CAPITAL:,.0f}")

    # Ruin threshold
    ax.axhline(INITIAL_CAPITAL * 0.5, color=PALETTE["red"], lw=1, ls=":", zorder=1,
               alpha=0.7, label="Ruin (<50%)")

    # Log scale for y (compound returns are multiplicative)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}" if v < 1_000_000 else f"${v/1_000_000:.1f}M"
    ))

    # Final value annotations (right edge)
    x_ann = trades_per_year + trades_per_year * 0.01
    for i, p in enumerate(pctiles):
        val = bands[p][-1]
        label_str = f"P{p}: ${val:,.0f}" if val < 1_000_000 else f"P{p}: ${val/1_000_000:.1f}M"
        ax.text(x_ann, val, label_str, va="center", ha="left",
                color=fan_colors[i], fontsize=9, fontweight="bold")

    # x-axis: approximate weeks
    week_ticks = np.arange(0, trades_per_year + 1, trades_per_year // 12)
    ax.set_xticks(week_ticks)
    ax.set_xticklabels([f"Mo.{i+1}" if i < 12 else "Year" for i in range(len(week_ticks))],
                       fontsize=9, color=PALETTE["muted"])

    ax.set_xlim(0, trades_per_year * 1.18)
    ax.set_xlabel("Month (within 1 year)", color=PALETTE["muted"], fontsize=10)
    ax.set_ylabel("Portfolio Value ($)", color=PALETTE["muted"], fontsize=10)
    ax.tick_params(colors=PALETTE["muted"])
    ax.grid(color=PALETTE["grid"], lw=0.5, alpha=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])

    ax.set_title(
        "QuantBox v14.3.0 — Equity Curve Fan (1 Year, 10,000 paths)\n"
        "Bootstrap resampling | Per-trade leverage VR 3.0× / MA 2.1× | Log scale",
        color=PALETTE["text"], fontsize=12, pad=10
    )
    ax.legend(loc="upper left", framealpha=0.3, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print("  r46b — Monte Carlo Visualization  |  QuantBox v14.3.0")
    print(f"{'=' * 70}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data...")
    data_15m = load_15m()
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    print(f"  Data loaded: {time.time()-t0:.1f}s")

    # ── Collect trades ────────────────────────────────────────────────────
    ma_trades = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )
    d_trades_filtered, _ = collect_d_trades_with_tbr(CFG_V13, data_15m, data_1h, data_4h)
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    vr_trades = d_trades_filtered + s_trades

    comb_trades = vr_trades + ma_trades
    _, comb_accepted = portfolio_sim_with_details(comb_trades, 10)

    avg_bars   = int(np.mean([len(v["close"]) for v in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)

    # Build pnl/ln arrays with per-trade leverage tagging
    vr_key_set = {(sym, e, x) for sym, e, x, _ in vr_trades}
    pnl_list, ln_list = [], []
    for sym, e, x, pnl_val in comb_accepted:
        pnl_list.append(float(pnl_val))
        ln_list.append(VR_LN if (sym, e, x) in vr_key_set else MA_LN)

    pnl_arr = np.array(pnl_list, dtype=np.float64)
    ln_arr  = np.array(ln_list,  dtype=np.float64)
    trades_per_year = max(1, round(len(comb_accepted) / data_years))

    print(f"  Trades: {len(comb_accepted)}  |  {trades_per_year}/yr  |  "
          f"blended L/N={np.mean(ln_arr):.3f}  |  {time.time()-t0:.1f}s")

    # ── Run simulations ───────────────────────────────────────────────────
    print("\n  Running simulations...")
    rng = np.random.default_rng(seed=SEED)

    tpd = max(1, round(trades_per_year / 365))
    tpw = max(1, round(trades_per_year / 52))
    tpm = max(1, round(trades_per_year / 12))
    tpy = trades_per_year

    horizons_data: dict[str, tuple[np.ndarray, int]] = {}
    for key, n in [("day", tpd), ("week", tpw), ("month", tpm), ("year", tpy)]:
        _, ret = _sim(pnl_arr, ln_arr, n, rng)
        horizons_data[key] = (ret, n)

    # For equity fan: full year paths
    rng2 = np.random.default_rng(seed=SEED)
    cum_eq_year, _ = _sim(pnl_arr, ln_arr, tpy, rng2)

    print(f"  Simulations done: {time.time()-t0:.1f}s")

    # ── Generate figures ──────────────────────────────────────────────────
    print("\n  Generating figures...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1_path = OUT_DIR / "20260307_r46_mc_distributions.png"
    fig2_path = OUT_DIR / "20260307_r46_mc_equity_fan.png"

    fig1_distributions(horizons_data, fig1_path)
    fig2_equity_fan(cum_eq_year, tpy, fig2_path)

    print(f"\n  Done in {time.time()-t0:.1f}s")
    print(f"  Output: {OUT_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
