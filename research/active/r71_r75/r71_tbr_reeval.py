#!/usr/bin/env python3
"""r71: D_mom8 TBR Gate Re-evaluation with trail_activate=1.0%.

Background:
    r47 disabled f5_tbr_enabled because trail_activate=0.5% handled bad trades
    directly (early trail exit). With trail_activate now 1.0%, the trail only
    kicks in after a 1% gain — bad entries can run to SL(1%) unfiltered.
    TBR gate becomes MORE necessary, not less.

TBR gate logic:
    Blocks LONG when tbr_z > threshold (taker buyers already aggressively active)
    Blocks SHORT when tbr_z < -threshold (taker sellers already aggressively active)
    Intuition: if the crowd is already acting, the squeeze breakout may be exhausted.

Question:
    Does TBR on/off make a consistent difference with trail_activate=1.0%?
    Tested on 2023 / 2024 / 2025 year-by-year OOS.

Pass criterion (ALL three years must pass):
    TBR_on PF > TBR_off PF for that year   AND   N_on >= 0.70 × N_off

Output:
    docs/backtest_results/202603/{YYYYMMDD}_{commit}_r71_tbr_reeval.md
    docs/backtest_results/202603/r71_tbr_reeval.png

Usage:
    uv run python research/active/r71_r75/r71_tbr_reeval.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    G_3TF,
    COST_DIRECTIONAL,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trades,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_signal,
    make_direction_momentum,
)

# ── Guard: verify config source ───────────────────────────────────────────────
cfg = VolStraddleConfig()
assert cfg.d_trail_activate == 0.010, (
    f"Expected d_trail_activate=0.010 from config.py, got {cfg.d_trail_activate}. "
    "Do not hardcode — read from VolStraddleConfig()."
)

VR_SHORT = cfg.vr_short
VR_LONG  = cfg.vr_long
VR_KEY   = f"vr_{VR_SHORT}_{VR_LONG}"
MAX_CONC = cfg.max_concurrent
VR_LN    = cfg.base_leverage / MAX_CONC
TBR_THRESH = cfg.f5_tbr_z_threshold   # 0.5
TBR_WIN    = cfg.f5_tbr_z_window      # 192

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

CACHE_DIR  = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"

DATA_END   = datetime(2026, 2, 20)

# Year-by-year windows (OOS validation)
YEAR_WINDOWS = [
    ("2023", datetime(2023, 1, 1), datetime(2023, 12, 31)),
    ("2024", datetime(2024, 1, 1), datetime(2024, 12, 31)),
    ("2025", datetime(2025, 1, 1), datetime(2025, 12, 31)),
    ("All (2020-2026)", datetime(2020, 1, 1), DATA_END),
]


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data_15m() -> dict:
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
        entry = {k: npz[k] for k in npz.files}
        data[sym] = entry

    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])

    print(f"Loaded {len(data)} symbols  (whitelist={len(whitelist)})")
    return data


def build_htf_with_taker(d15: dict, ratio: int) -> dict:
    """Resample 15m → higher TF, including taker_buy sum aggregation."""
    d_htf = build_resampled(d15, ratio)
    # Resample taker_buy (sum per period, matching volume aggregation)
    for sym, d in d_htf.items():
        if sym not in d15 or "taker_buy" not in d15[sym]:
            continue
        tb15 = d15[sym]["taker_buy"]
        n15 = len(tb15)
        n_htf = n15 // ratio
        # Trim to aligned length
        trim = n_htf * ratio
        tb_trimmed = tb15[n15 - trim:]
        d["taker_buy"] = tb_trimmed.reshape(n_htf, ratio).sum(axis=1)
    return d_htf


# ─── TBR z-score ──────────────────────────────────────────────────────────────

def compute_tbr_zscore_vec(
    taker_buy: np.ndarray,
    volume: np.ndarray,
    w: int,
) -> np.ndarray:
    """Fully-vectorised TBR z-score — no lookahead, O(n) via sliding window.

    For bar i: z[i] = (tbr[i] - mean(tbr[i-w:i])) / std(tbr[i-w:i])
    Window excludes the current bar to match strategy._compute_tbr_zscore().

    Returns 0.0 for bars with insufficient history.
    """
    n = len(taker_buy)
    safe_vol = np.where(volume > 0, volume, 1.0)
    tbr = taker_buy / safe_vol  # shape (n,)

    if n <= w:
        return np.zeros(n, dtype=np.float64)

    # sliding_window_view over tbr[0:n-1] (exclude current bar by shifting by 1)
    # windows[j] = tbr[j : j+w]  → corresponds to bar i = j+w (current bar)
    # so for bar i, window = tbr[i-w : i]
    src = tbr[:-1]  # length n-1 (exclude last bar to avoid 1-off edge)
    if len(src) < w:
        return np.zeros(n, dtype=np.float64)

    wins = np.lib.stride_tricks.sliding_window_view(src, w)  # shape (n-1-w+1, w) = (n-w, w)
    mu    = wins.mean(axis=1)      # shape (n-w,)
    sigma = wins.std(axis=1)       # shape (n-w,)

    # bar indices that have full windows: i = w, w+1, ..., n-1
    cur_tbr = tbr[w:]              # shape (n-w,) — current bar TBR values
    safe_sig = np.where(sigma < 1e-9, 1.0, sigma)
    z_vals   = np.where(sigma < 1e-9, 0.0, (cur_tbr - mu) / safe_sig)

    z = np.zeros(n, dtype=np.float64)
    z[w:] = z_vals
    return z


# ─── D_mom8 trade collection with optional TBR gate ──────────────────────────

def collect_dmom8(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    tbr_enabled: bool,
    tbr_thresh: float = TBR_THRESH,
    tbr_win: int = TBR_WIN,
) -> list[tuple[str, int, int, float]]:
    """Collect D_mom8 trades with optional TBR gate.

    Args:
        data_15m / data_1h / data_4h: Precomputed OHLCV + indicators.
        tbr_enabled: Whether to apply F5-TBR gate.
        tbr_thresh: TBR z-score threshold (from config).
        tbr_win: TBR rolling window (from config).

    Returns:
        List of (symbol, entry_15m, exit_15m, pnl).
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    d_sl  = cfg.d_sl
    d_tp  = cfg.d_tp
    d_vr  = cfg.d_vr_threshold

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []

        for _lbl, ratio, warmup, mh_d, _mh_s in G_3TF:
            d = tf_data_map.get(ratio)
            if d is None or sym not in d:
                continue
            sd = d[sym]
            vr   = sd.get(VR_KEY)
            mom8 = sd.get("mom_8")
            if vr is None or mom8 is None:
                continue

            c  = sd["close"]
            h  = sd["high"]
            lo = sd["low"]

            # Raw VR signal → direction from momentum
            sig  = gen_vr_signal(vr, d_vr, warmup)
            dire = make_direction_momentum(sig, mom8)

            # Optional TBR gate: filter out blocked entries
            if tbr_enabled and "taker_buy" in sd and "volume" in sd:
                tbr_z = compute_tbr_zscore_vec(sd["taker_buy"], sd["volume"], tbr_win)
                filtered_dire = dire.copy()
                for i in range(len(dire)):
                    if dire[i] == 0:
                        continue
                    is_long = dire[i] == 1
                    if is_long and tbr_z[i] > tbr_thresh:
                        filtered_dire[i] = 0  # block LONG
                    elif not is_long and tbr_z[i] < -tbr_thresh:
                        filtered_dire[i] = 0  # block SHORT
                dire = filtered_dire

            p, e, x = sim_directional_trades(
                c, h, lo, sig, dire, d_sl, d_tp, mh_d, COST_DIRECTIONAL
            )
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio + offset
                exit_15m  = int(x[k]) * ratio + (ratio - 1) + offset
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        # Per-symbol time dedup: sort by entry, reject overlapping
        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            if not any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted):
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades


def filter_year(
    trades: list[tuple[str, int, int, float]],
    max_bars: int,
    start: datetime,
    end: datetime,
) -> list[tuple[str, int, int, float]]:
    """Filter trades to a specific calendar window (approximate via bar index)."""
    data_start = DATA_END - __import__("datetime").timedelta(minutes=max_bars * 15)

    def bar_to_dt(bar: int) -> datetime:
        return data_start + __import__("datetime").timedelta(minutes=bar * 15)

    return [
        t for t in trades
        if start <= bar_to_dt(t[1]) < end
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    commit = get_commit()
    today  = datetime.utcnow().strftime("%Y%m%d")

    # ── Load + precompute ──
    print("Loading data...")
    d15 = load_data_15m()
    max_bars = max(len(d["close"]) for d in d15.values())
    data_years = max_bars * 15 / 60 / 24 / 365.25

    print("Building HTF resamples...")
    d1h = build_htf_with_taker(d15, 4)
    d4h = build_htf_with_taker(d15, 16)

    print("Precomputing indicators...")
    for ds in [d15, d1h, d4h]:
        precompute_all_vr([ds], [(VR_SHORT, VR_LONG)])
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)

    # ── Phase 1: baseline check ──
    print("\n[Phase 1] Baseline verification...")
    from quantbox.backtest.trade_sim import collect_trades
    baseline_cfg = {
        "sp": VR_SHORT, "lp": VR_LONG,
        "d_vr": cfg.d_vr_threshold, "s_vr": cfg.s_vr_threshold,
        "s_dur": cfg.s_squeeze_dur,
        "d_sl": cfg.d_sl, "s_sl": cfg.s_sl,
        "tp": cfg.d_tp,
    }
    bt_trades = collect_trades(baseline_cfg, d15, d1h, d4h, "D_mom8")
    bt_pnl    = portfolio_sim(bt_trades, MAX_CONC)
    bt_m      = compute_metrics(bt_pnl, data_years)
    print(f"  D_mom8 baseline (collect_trades): N={bt_m['N']}, PF={bt_m['PF']:.3f}")

    # ── Phase 2: TBR off vs on ──
    print("\n[Phase 2] D_mom8 TBR off vs on...")
    trades_off = collect_dmom8(d15, d1h, d4h, tbr_enabled=False)
    trades_on  = collect_dmom8(d15, d1h, d4h, tbr_enabled=True)

    pnl_off = portfolio_sim(trades_off, MAX_CONC)
    pnl_on  = portfolio_sim(trades_on,  MAX_CONC)
    m_off   = compute_metrics(pnl_off, data_years)
    m_on    = compute_metrics(pnl_on,  data_years)

    print(f"  TBR OFF: N={m_off['N']:>4}  PF={m_off['PF']:.3f}  WR={m_off['WR']:.1f}%  EV={m_off['EV_bp']:.1f}bp")
    print(f"  TBR ON:  N={m_on['N']:>4}  PF={m_on['PF']:.3f}  WR={m_on['WR']:.1f}%  EV={m_on['EV_bp']:.1f}bp")
    print(f"  ΔPF = {m_on['PF'] - m_off['PF']:+.3f}  ΔN = {m_on['N'] - m_off['N']:+d} ({(m_on['N']-m_off['N'])/max(m_off['N'],1)*100:.1f}%)")

    # ── Phase 3: year-by-year ──
    print("\n[Phase 3] Year-by-year OOS breakdown...")
    year_results: list[dict] = []
    for label, y_start, y_end in YEAR_WINDOWS:
        yr_off = filter_year(trades_off, max_bars, y_start, y_end)
        yr_on  = filter_year(trades_on,  max_bars, y_start, y_end)
        days = (min(y_end, DATA_END) - y_start).days
        years = max(days / 365.25, 0.01)

        pnl_yr_off = portfolio_sim(yr_off, MAX_CONC)
        pnl_yr_on  = portfolio_sim(yr_on,  MAX_CONC)
        m_yr_off   = compute_metrics(pnl_yr_off, years)
        m_yr_on    = compute_metrics(pnl_yr_on,  years)

        n_ratio = m_yr_on["N"] / max(m_yr_off["N"], 1)
        pass_pf = m_yr_on["PF"] > m_yr_off["PF"]
        pass_n  = n_ratio >= 0.70
        passed  = pass_pf and pass_n

        year_results.append({
            "label":   label,
            "N_off":   m_yr_off["N"],
            "N_on":    m_yr_on["N"],
            "PF_off":  m_yr_off["PF"],
            "PF_on":   m_yr_on["PF"],
            "WR_off":  m_yr_off["WR"],
            "WR_on":   m_yr_on["WR"],
            "passed":  passed,
        })
        mark = "✅" if passed else "❌"
        print(f"  {label}: OFF PF={m_yr_off['PF']:.3f}(N={m_yr_off['N']})  "
              f"ON PF={m_yr_on['PF']:.3f}(N={m_yr_on['N']})  "
              f"ΔPF={m_yr_on['PF']-m_yr_off['PF']:+.3f}  {mark}")

    oos_years = [r for r in year_results if r["label"] in ("2023", "2024", "2025")]
    all_pass  = all(r["passed"] for r in oos_years)
    verdict   = "UPGRADE ✅" if all_pass else "KEEP ❌"
    print(f"\n  Verdict: TBR → {verdict}")
    print(f"  (Pass = all 3 OOS years: PF improved AND N >= 70% of baseline)")

    # ── Chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG_DARK)

    oos_only = [r for r in year_results if r["label"] in ("2023", "2024", "2025")]
    labels   = [r["label"] for r in oos_only]
    x        = np.arange(len(labels))
    w        = 0.35

    # PF comparison
    ax1 = axes[0]
    ax1.set_facecolor(BG_CARD)
    bars_off = ax1.bar(x - w/2, [r["PF_off"] for r in oos_only], w,
                       label="TBR OFF", color=ACCENT, alpha=0.85)
    bars_on  = ax1.bar(x + w/2, [r["PF_on"]  for r in oos_only], w,
                       label="TBR ON",  color=GREEN, alpha=0.85)
    ax1.axhline(1.0, color=RED, lw=1, ls="--", alpha=0.7)
    ax1.set_title("Profit Factor by Year", color=TEXT_MAIN, fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color=TEXT_SUB)
    ax1.tick_params(colors=TEXT_SUB)
    ax1.spines[:].set_edgecolor(BG_BORDER)
    ax1.set_ylim(0, max(max(r["PF_off"] for r in oos_only),
                        max(r["PF_on"]  for r in oos_only)) * 1.25 + 0.1)
    ax1.legend(facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_MAIN)
    for bar in bars_off:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 color=TEXT_MAIN, fontsize=9)
    for bar in bars_on:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 color=TEXT_MAIN, fontsize=9)

    # Trade count comparison
    ax2 = axes[1]
    ax2.set_facecolor(BG_CARD)
    bars_n_off = ax2.bar(x - w/2, [r["N_off"] for r in oos_only], w,
                         label="TBR OFF", color=ACCENT, alpha=0.85)
    bars_n_on  = ax2.bar(x + w/2, [r["N_on"]  for r in oos_only], w,
                         label="TBR ON",  color=GREEN, alpha=0.85)
    ax2.set_title("Trade Count by Year", color=TEXT_MAIN, fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color=TEXT_SUB)
    ax2.tick_params(colors=TEXT_SUB)
    ax2.spines[:].set_edgecolor(BG_BORDER)
    ax2.legend(facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_MAIN)
    ax2.set_ylim(0, max(max(r["N_off"] for r in oos_only),
                        max(r["N_on"]  for r in oos_only)) * 1.25 + 5)
    for bar in bars_n_off:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha="center", va="bottom",
                 color=TEXT_MAIN, fontsize=9)
    for bar in bars_n_on:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha="center", va="bottom",
                 color=TEXT_MAIN, fontsize=9)

    fig.suptitle(
        f"r71: D_mom8 TBR Gate Re-evaluation  |  trail_activate={cfg.d_trail_activate*100:.1f}%  |  thresh={TBR_THRESH}",
        color=TEXT_MAIN, fontsize=13, y=0.98,
    )
    fig.text(
        0.5, 0.01,
        f"60 syms · 6yr · Binance USDT-M · 0.08% round-trip · {commit}",
        ha="center", color=TEXT_SUB, fontsize=9, style="italic",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_dir = REPORT_DIR / "202603"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / "r71_tbr_reeval.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"\nChart saved: {chart_path}")

    # ── Report ──
    report_path = out_dir / f"{today}_{commit}_r71_tbr_reeval.md"
    with open(report_path, "w") as f:
        f.write(f"""# r71: D_mom8 TBR Gate Re-evaluation

**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**Commit:** {commit}
**Parity:** config.py읽음 ✅ · 룩어헤드 없음 ✅ · symbol_whitelist(60sym) ✅

---

## Context

r47 disabled `f5_tbr_enabled` because `trail_activate=0.5%` handled bad trades via early exit.
After r69 upgrade (`trail_activate=1.0%`), the trail only kicks in after 1% gain.
Bad entries can now run unfiltered to SL(1%). TBR gate re-evaluation justified.

**Config (from VolStraddleConfig()):**
- `d_trail_activate = {cfg.d_trail_activate}` (1.0% — upgraded in r69)
- `f5_tbr_z_threshold = {cfg.f5_tbr_z_threshold}`
- `f5_tbr_z_window = {cfg.f5_tbr_z_window}` bars
- `d_sl = {cfg.d_sl}`, `d_tp = {cfg.d_tp}`
- `L/N = {VR_LN:.1f}` (VR squeeze)

---

## Phase 1: Baseline Sanity

| | D_mom8 (collect_trades) |
|--|--|
| N | {bt_m['N']} |
| PF | {bt_m['PF']:.3f} |

---

## Phase 2: Full Period Comparison

| | TBR OFF | TBR ON | Δ |
|--|--|--|--|
| N | {m_off['N']} | {m_on['N']} | {m_on['N']-m_off['N']:+d} ({(m_on['N']-m_off['N'])/max(m_off['N'],1)*100:.1f}%) |
| PF | {m_off['PF']:.3f} | {m_on['PF']:.3f} | {m_on['PF']-m_off['PF']:+.3f} |
| WR | {m_off['WR']:.1f}% | {m_on['WR']:.1f}% | {m_on['WR']-m_off['WR']:+.1f}pp |
| EV (bp) | {m_off['EV_bp']:.1f} | {m_on['EV_bp']:.1f} | {m_on['EV_bp']-m_off['EV_bp']:+.1f} |

---

## Phase 3: Year-by-Year OOS Validation

Pass criterion: PF_on > PF_off AND N_on >= 70% of N_off

| Year | N_off | N_on | PF_off | PF_on | ΔPF | WR_off | WR_on | Pass? |
|------|-------|------|--------|-------|-----|--------|-------|-------|
""")
        for r in year_results:
            mark = "✅" if r["passed"] else "❌"
            f.write(
                f"| {r['label']} | {r['N_off']} | {r['N_on']} | "
                f"{r['PF_off']:.3f} | {r['PF_on']:.3f} | "
                f"{r['PF_on']-r['PF_off']:+.3f} | "
                f"{r['WR_off']:.1f}% | {r['WR_on']:.1f}% | {mark} |\n"
            )
        f.write(f"""
---

## Verdict

**{verdict}**

Pass criterion: All 3 OOS years (2023/2024/2025) must show PF improvement AND N >= 70%.

""")
        if all_pass:
            f.write("→ Recommend setting `f5_tbr_enabled = True` in config.py.\n")
            f.write("→ Next: deploy in feature branch, run parity test, restart session.\n")
        else:
            f.write("→ TBR gate does not consistently improve PF with trail_activate=1.0%.\n")
            f.write("→ Keep `f5_tbr_enabled = False`.\n")

        f.write(f"""
---

## Data

- 60 symbols (symbol_whitelist)
- 6 years: 2020-01-01 ~ 2026-02-20
- Binance USDT-M perpetuals, 15m/1h/4h
- Cost: 0.08% round-trip (entry + exit)
- Slots: {MAX_CONC} (max_concurrent)
- L/N = {VR_LN:.1f}

**구현:** 미구현
""")

    print(f"Report saved: {report_path}")
    print(f"\n{'='*50}")
    print(f"Final Verdict: TBR → {verdict}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
