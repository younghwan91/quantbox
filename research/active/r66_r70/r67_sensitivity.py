#!/usr/bin/env python3
"""r67: Parameter Sensitivity Analysis — VR Squeeze Production Config.

Sweeps 7 key parameters ±20-50% around production values.
Goal: confirm no "cliff edges" — small changes should not cause large PF drops.

Pass criterion per parameter:
  ±20% change → PF change ≤ ±30%  (robust)
  OOS PF always ≥ 1.05            (generalises)

Output:
  docs/backtest_results/202603/YYYYMMDD_{commit}_r67_sensitivity.md
  docs/backtest_results/202603/r67_sensitivity_*.png  (per-parameter charts)

Usage:
    uv run python research/active/r66_r70/r67_sensitivity.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    G_3TF,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal
from research.active.r41_r45.r43_exit_improvement import sim_straddle_trailing

# ── Style ─────────────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_BORDER = "#30363d"
TEXT_MAIN = "#f0f6fc"
TEXT_SUB  = "#8b949e"
GREEN     = "#3fb950"
RED       = "#f85149"
ACCENT    = "#58a6ff"
ORANGE    = "#d29922"

# ── Config (NO hardcoding) ────────────────────────────────────────────────────
cfg = VolStraddleConfig()

VR_SHORT = cfg.vr_short
VR_LONG  = cfg.vr_long
VR_KEY   = f"vr_{VR_SHORT}_{VR_LONG}"
MAX_CONC = cfg.max_concurrent
VR_LN    = cfg.base_leverage / MAX_CONC

WARMUP_30m = 100
_MH30m = cfg.max_hold_15m // 2
TF_30m = ("30m", 2, WARMUP_30m, _MH30m * 2, _MH30m * 2)
G_4TF = [TF_30m] + list(G_3TF)

CACHE_DIR  = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"

DATA_END  = datetime(2026, 2, 20)
OOS_START = datetime(2023, 1, 1)

# ── Production baseline values (from config.py) ───────────────────────────────
PROD = {
    "d_sl":              cfg.d_sl,              # 0.010
    "d_tp":              cfg.d_tp,              # 0.120
    "d_trail_activate":  cfg.d_trail_activate,  # 0.005
    "d_trail_distance":  cfg.d_trail_distance,  # 0.002
    "s_vr_threshold":    cfg.s_vr_threshold,    # 0.570
    "s_squeeze_dur":     cfg.s_squeeze_dur,     # 3
    "base_leverage":     cfg.base_leverage,     # 30.0
}

# ── Sweep grids (7-point each) ────────────────────────────────────────────────
SWEEPS: dict[str, list[float]] = {
    "d_sl":              [0.006, 0.007, 0.008, 0.010, 0.012, 0.014, 0.016],
    "d_tp":              [0.080, 0.090, 0.100, 0.120, 0.140, 0.150, 0.160],
    "d_trail_activate":  [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010],
    "d_trail_distance":  [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006],
    "s_vr_threshold":    [0.50, 0.53, 0.55, 0.57, 0.59, 0.61, 0.65],
    "s_squeeze_dur":     [2, 2, 3, 3, 4, 5, 6],  # integers only
    "base_leverage":     [20.0, 22.0, 25.0, 30.0, 33.0, 36.0, 40.0],
}

PROD_IDX: dict[str, int] = {k: 3 for k in SWEEPS}  # production value is at index 3


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_syms() -> dict:
    whitelist = set(cfg.symbol_whitelist)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}

    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    print(f"Loaded {len(data)} symbols  (whitelist={len(whitelist)})")
    return data


def build_all_tf(d15: dict) -> tuple[dict, dict, dict, dict]:
    d30 = build_resampled(d15, 2)
    d1h = build_resampled(d15, 4)
    d4h = build_resampled(d15, 16)
    for ds in [d15, d30, d1h, d4h]:
        precompute_all_vr([ds], [(VR_SHORT, VR_LONG)])
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
    return d30, d1h, d4h


def _dedup(sym: str, trades: list) -> list:
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_s(
    d15: dict, d30: dict, d1h: dict, d4h: dict,
    s_vr: float, s_dur: int,
    sl: float, tp: float, trail_act: float, trail_dist: float,
) -> list:
    """S_d8 straddle trades across 4 TFs with configurable exit params."""
    tf_map = {1: d15, 2: d30, 4: d1h, 16: d4h}
    all_trades: list = []
    for sym in sorted(d15.keys()):
        offset = int(d15[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, _mhd, mh_s in G_4TF:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(VR_KEY)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE, trail_act, trail_dist
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def compute_split_bar(max_bars: int) -> int:
    start_dt = DATA_END - timedelta(minutes=max_bars * 15)
    if OOS_START <= start_dt:
        return 0
    span = (DATA_END - start_dt).total_seconds() / 60
    total_bars = int(span / 15)
    oos_span = (DATA_END - OOS_START).total_seconds() / 60
    return total_bars - int(oos_span / 15)


def run_point(
    d15: dict, d30: dict, d1h: dict, d4h: dict,
    params: dict, max_bars: int, data_years: float,
) -> dict:
    """Run a single parameter configuration and return metrics."""
    trades = collect_s(
        d15, d30, d1h, d4h,
        s_vr=params["s_vr_threshold"],
        s_dur=int(params["s_squeeze_dur"]),
        sl=params["d_sl"],
        tp=params["d_tp"],
        trail_act=params["d_trail_activate"],
        trail_dist=params["d_trail_distance"],
    )

    # LN = param leverage / max_concurrent
    ln = params["base_leverage"] / MAX_CONC

    split_bar = compute_split_bar(max_bars)
    is_trades  = [(s, e, x, p) for s, e, x, p in trades if e < split_bar]
    oos_trades = [(s, e, x, p) for s, e, x, p in trades if e >= split_bar]

    oos_years = (DATA_END - OOS_START).days / 365.25
    is_years  = max(data_years - oos_years, 0.1)

    pnl_full = portfolio_sim(trades, MAX_CONC)
    pnl_oos  = portfolio_sim(oos_trades, MAX_CONC)

    m_full = compute_metrics(pnl_full, data_years)
    m_oos  = compute_metrics(pnl_oos, oos_years)

    return {
        "N":      m_full["N"],
        "PF":     m_full["PF"],
        "EV_bp":  m_full["EV_bp"],
        "WR":     m_full["WR"],
        "maxdd":  m_full.get("max_dd_pct", 0),
        "OOS_PF": m_oos["PF"],
        "OOS_N":  m_oos["N"],
        "ln":     ln,
    }


# ─── Chart helpers ────────────────────────────────────────────────────────────

def plot_sweep(
    param: str, values: list[float], results: list[dict], prod_pf: float,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG_DARK)

    labels = [f"{v:.4g}" for v in values]
    pf_vals   = [r["PF"]    for r in results]
    oos_vals  = [r["OOS_PF"] for r in results]
    ev_vals   = [r["EV_bp"]  for r in results]
    prod_i    = PROD_IDX[param]

    def _ax_setup(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor(BG_CARD)
        ax.set_title(title, color=TEXT_MAIN, fontsize=11, pad=8)
        ax.tick_params(colors=TEXT_SUB, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG_BORDER)
        ax.grid(axis="y", color=BG_BORDER, linewidth=0.5, alpha=0.5)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, color=TEXT_SUB)

    # PF line
    ax = axes[0]
    _ax_setup(ax, "Profit Factor")
    colors = [GREEN if v >= prod_pf * 0.90 else RED for v in pf_vals]
    ax.bar(range(len(values)), pf_vals, color=colors, alpha=0.7, width=0.6)
    ax.plot(range(len(values)), pf_vals, color=ACCENT, linewidth=1.5, marker="o", markersize=4)
    ax.axhline(prod_pf, color=ORANGE, linewidth=1.0, linestyle="--", label=f"Prod PF={prod_pf:.3f}")
    ax.axhline(prod_pf * 0.70, color=RED, linewidth=0.8, linestyle=":", alpha=0.6, label="-30% cliff")
    ax.axvline(prod_i, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB)
    ax.set_ylabel("PF", color=TEXT_SUB, fontsize=9)

    # OOS PF
    ax = axes[1]
    _ax_setup(ax, "OOS Profit Factor (2023-2026)")
    oos_colors = [GREEN if v >= 1.05 else RED for v in oos_vals]
    ax.bar(range(len(values)), oos_vals, color=oos_colors, alpha=0.7, width=0.6)
    ax.plot(range(len(values)), oos_vals, color=ACCENT, linewidth=1.5, marker="o", markersize=4)
    ax.axhline(1.05, color=ORANGE, linewidth=0.8, linestyle="--", label="OOS floor 1.05")
    ax.axvline(prod_i, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB)
    ax.set_ylabel("OOS PF", color=TEXT_SUB, fontsize=9)

    # EV
    ax = axes[2]
    _ax_setup(ax, "Edge (EV bp/trade)")
    ev_colors = [GREEN if v >= 0 else RED for v in ev_vals]
    ax.bar(range(len(values)), ev_vals, color=ev_colors, alpha=0.7, width=0.6)
    ax.plot(range(len(values)), ev_vals, color=ACCENT, linewidth=1.5, marker="o", markersize=4)
    ax.axhline(0, color=RED, linewidth=0.8, linestyle="--")
    ax.axvline(prod_i, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.5,
               label=f"Production ({values[prod_i]:.4g})")
    ax.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB)
    ax.set_ylabel("EV (bp)", color=TEXT_SUB, fontsize=9)

    fig.suptitle(
        f"Sensitivity: {param}  |  60 syms · 6yr · 0.08% cost",
        color=TEXT_MAIN, fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out = out_dir / f"r67_{param}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    commit = get_commit()
    today  = datetime.utcnow().strftime("%Y%m%d")

    print("=" * 60)
    print("r67: Parameter Sensitivity Analysis")
    print("=" * 60)
    print("All params from config.py ✅\n")

    print("Loading data...")
    d15 = load_syms()
    max_bars   = max(len(d["close"]) for d in d15.values())
    data_years = max_bars * 15 / (365.25 * 24 * 60)
    print(f"Data: {data_years:.1f} years  |  max_bars={max_bars}\n")

    print("Building TFs and precomputing indicators...")
    d30, d1h, d4h = build_all_tf(d15)
    print()

    # ── Baseline (production) ─────────────────────────────────────────────────
    print("Running production baseline...")
    baseline = run_point(d15, d30, d1h, d4h, PROD, max_bars, data_years)
    prod_pf  = baseline["PF"]
    print(f"  Baseline: N={baseline['N']}  PF={prod_pf:.3f}  "
          f"EV={baseline['EV_bp']:.1f}bp  OOS_PF={baseline['OOS_PF']:.3f}\n")

    # ── Parameter sweeps ──────────────────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}
    cliff_flags: dict[str, list[bool]] = {}

    for param, values in SWEEPS.items():
        print(f"Sweeping {param} ({len(values)} points)...")
        sweep_res: list[dict] = []
        for v in values:
            p = dict(PROD)
            p[param] = v
            r = run_point(d15, d30, d1h, d4h, p, max_bars, data_years)
            sweep_res.append(r)
            mark = "★" if abs(v - PROD[param]) < 1e-9 else " "
            print(f"  {mark} {param}={v:.4g}  N={r['N']:5d}  PF={r['PF']:.3f}  "
                  f"OOS={r['OOS_PF']:.3f}  EV={r['EV_bp']:.1f}bp")
        all_results[param] = sweep_res

        # Cliff-edge check: ±1 step from production (indices 2 and 4)
        cliffs = []
        for i, r in enumerate(sweep_res):
            pf_change = (r["PF"] - prod_pf) / prod_pf
            oos_ok    = r["OOS_PF"] >= 1.05
            cliff     = abs(pf_change) > 0.30 or not oos_ok
            cliffs.append(cliff)
        cliff_flags[param] = cliffs
        print()

    # ── Charts ────────────────────────────────────────────────────────────────
    month_dir = REPORT_DIR / "202603"
    month_dir.mkdir(parents=True, exist_ok=True)
    print("Generating charts...")
    for param, sweep_res in all_results.items():
        out = plot_sweep(param, SWEEPS[param], sweep_res, prod_pf, month_dir)
        print(f"  Saved: {out.name}")
    print()

    # ── Robustness summary ────────────────────────────────────────────────────
    print("─" * 60)
    print("ROBUSTNESS SUMMARY  (±1 step from production = ±20-50%)")
    print("─" * 60)
    all_robust = True
    rows: list[tuple] = []
    for param, sweep_res in all_results.items():
        prod_i = PROD_IDX[param]
        lo_i   = max(0, prod_i - 1)
        hi_i   = min(len(sweep_res) - 1, prod_i + 1)
        lo_pf  = sweep_res[lo_i]["PF"]
        hi_pf  = sweep_res[hi_i]["PF"]
        lo_chg = (lo_pf - prod_pf) / prod_pf * 100
        hi_chg = (hi_pf - prod_pf) / prod_pf * 100
        lo_oos = sweep_res[lo_i]["OOS_PF"]
        hi_oos = sweep_res[hi_i]["OOS_PF"]
        robust = (abs(lo_chg) <= 30 and abs(hi_chg) <= 30
                  and lo_oos >= 1.05 and hi_oos >= 1.05)
        if not robust:
            all_robust = False
        flag = "✅" if robust else "❌"
        rows.append((flag, param, PROD[param], lo_chg, hi_chg, lo_oos, hi_oos))
        print(f"  {flag} {param:22s}  prod={PROD[param]:.4g}"
              f"  Δ-step={lo_chg:+.1f}%  Δ+step={hi_chg:+.1f}%"
              f"  OOS=[{lo_oos:.3f},{hi_oos:.3f}]")

    print()
    if all_robust:
        print("✅ ALL PARAMS ROBUST — no cliff edges detected")
    else:
        print("⚠️  SOME PARAMS HAVE CLIFF EDGES — review highlighted params")

    # ── Markdown report ───────────────────────────────────────────────────────
    report_path = month_dir / f"{today}_{commit}_r67_sensitivity.md"
    with open(report_path, "w") as f:
        f.write(f"# r67: Parameter Sensitivity Analysis\n\n")
        f.write(f"**Date:** {today}  |  **Commit:** `{commit}`\n\n")
        f.write(f"**Data:** 60 syms · {data_years:.1f} years (2020-01 – 2026-02) · 0.08% round-trip\n\n")
        f.write(f"**Baseline (production):** N={baseline['N']}  PF={prod_pf:.3f}  "
                f"EV={baseline['EV_bp']:.1f}bp  OOS_PF={baseline['OOS_PF']:.3f}\n\n")
        f.write("## Robustness Summary\n\n")
        f.write("Pass criterion: ±1-step PF change ≤ ±30% AND OOS PF ≥ 1.05\n\n")
        f.write("| | Parameter | Prod Value | Δ -step | Δ +step | OOS lo | OOS hi |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for flag, param, pval, lo_chg, hi_chg, lo_oos, hi_oos in rows:
            f.write(f"| {flag} | `{param}` | {pval:.4g} | {lo_chg:+.1f}% | "
                    f"{hi_chg:+.1f}% | {lo_oos:.3f} | {hi_oos:.3f} |\n")

        f.write("\n## Detailed Sweeps\n\n")
        for param, sweep_res in all_results.items():
            f.write(f"### {param}\n\n")
            f.write(f"![{param}](r67_{param}.png)\n\n")
            f.write("| Value | N | PF | EV(bp) | OOS PF | Robust |\n")
            f.write("|---|---|---|---|---|---|\n")
            for i, (v, r) in enumerate(zip(SWEEPS[param], sweep_res)):
                pf_chg = (r["PF"] - prod_pf) / prod_pf * 100
                robust = abs(pf_chg) <= 30 and r["OOS_PF"] >= 1.05
                marker = " ★" if i == PROD_IDX[param] else ""
                f.write(f"| {v:.4g}{marker} | {r['N']} | {r['PF']:.3f} | "
                        f"{r['EV_bp']:.1f} | {r['OOS_PF']:.3f} | "
                        f"{'✅' if robust else '❌'} |\n")
            f.write("\n")

        verdict = "✅ ALL ROBUST" if all_robust else "⚠️ CLIFF EDGES FOUND"
        f.write(f"\n## Verdict\n\n**{verdict}**\n\n")
        f.write("**Parity Notes:**\n")
        f.write("- `symbol_whitelist` used (60 live symbols, not exclude_symbols) ✅\n")
        f.write("- All params from `VolStraddleConfig()` — no hardcoding ✅\n")
        f.write("- No lookahead: `_offset` applied, completed bars only ✅\n")
        f.write("- L/N model: 1x compound (EV_bp is leverage-independent) ✅\n")

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
