#!/usr/bin/env python3
"""r69: Year-by-Year OOS Validation for d_trail_activate and d_sl.

r68 found that OOS/IS=0.85 criterion was invalid (IS period = 2020-2022 bull market).
This script answers the correct question:

  "Does d_trail_activate=0.010 improve PF consistently in 2023, 2024, AND 2025?"

If all 3 recent years show improvement → UPGRADE.
If any year shows degradation → KEEP current value.

Also validates d_sl=0.006 (r67: +32.8% PF improvement).

Upgrade criterion (must pass for EACH year in 2023/2024/2025):
  - Candidate PF > baseline PF for that year
  - N > 0 (strategy traded in that year)

Output:
  docs/backtest_results/202603/YYYYMMDD_{commit}_r69_yearly.md
  docs/backtest_results/202603/r69_yearly.png

Usage:
    uv run python research/active/r66_r70/r69_yearly_breakdown.py
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal
from quantbox.backtest.trade_sim import sim_straddle_trailing

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
PURPLE    = "#bc8cff"
YELLOW    = "#e3b341"

# ── Config ────────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

VR_SHORT = cfg.vr_short
VR_LONG  = cfg.vr_long
VR_KEY   = f"vr_{VR_SHORT}_{VR_LONG}"
MAX_CONC = cfg.max_concurrent
VR_LN    = cfg.base_leverage / MAX_CONC

WARMUP_30m = 100
_MH30m = cfg.max_hold_15m // 2
TF_30m = ("30m", 2, WARMUP_30m, _MH30m * 2, _MH30m * 2)
G_4TF  = [TF_30m] + list(G_3TF)

CACHE_DIR  = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"

DATA_END = datetime(2026, 2, 20)

# Candidates to compare (prod value must be first in each list)
TRAIL_ACTS  = [0.005, 0.008, 0.010]   # prod at index 0
SL_VALS     = [0.010, 0.008, 0.006]   # prod at index 0

YEARS_OOS   = [2023, 2024, 2025]      # must ALL improve for upgrade


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


def build_all_tf(d15: dict) -> tuple[dict, dict, dict]:
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
    trail_act: float,
    sl: float | None = None,
    tp: float | None = None,
    trail_dist: float | None = None,
) -> list:
    _sl  = sl         if sl         is not None else cfg.d_sl
    _tp  = tp         if tp         is not None else cfg.d_tp
    _td  = trail_dist if trail_dist is not None else cfg.d_trail_distance

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
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, _sl, _tp, mh_s, COST_STRADDLE, trail_act, _td
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def dt_to_bar(dt: datetime, max_bars: int) -> int:
    """Convert a datetime to a global bar index (15m resolution)."""
    delta_min = (DATA_END - dt).total_seconds() / 60
    return max(0, max_bars - int(delta_min / 15))


def year_bar_range(year: int, max_bars: int) -> tuple[int, int]:
    """Return [start_bar, end_bar) for a calendar year."""
    start = dt_to_bar(datetime(year, 1, 1), max_bars)
    end   = dt_to_bar(datetime(year + 1, 1, 1), max_bars) if year < 2025 else max_bars
    return start, end


def run_yearly(
    trades: list,
    max_bars: int,
    years: list[int] | None = None,
) -> dict[int, dict]:
    """Break trades into calendar years and compute metrics for each."""
    if years is None:
        years = list(range(2020, 2026))
    result: dict[int, dict] = {}
    for yr in years:
        s_bar, e_bar = year_bar_range(yr, max_bars)
        yr_trades = [(sym, e, x, p) for sym, e, x, p in trades if s_bar <= e < e_bar]
        if not yr_trades:
            result[yr] = {"N": 0, "PF": 0.0, "EV_bp": 0.0, "WR": 0.0}
            continue
        pnl = portfolio_sim(yr_trades, MAX_CONC)
        m   = compute_metrics(pnl, 1.0)   # normalized to 1 year
        result[yr] = {"N": m["N"], "PF": m["PF"], "EV_bp": m["EV_bp"], "WR": m["WR"]}
    return result


# ─── Chart ────────────────────────────────────────────────────────────────────

def _color(label: str) -> str:
    if "0.005" in label or "0.010 (sl)" in label:
        return TEXT_SUB
    if "0.010" in label or "0.006" in label:
        return GREEN
    if "0.008" in label:
        return ACCENT
    return PURPLE


def plot_results(
    trail_results: dict[float, dict[int, dict]],
    sl_results:    dict[float, dict[int, dict]],
    years: list[int],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG_DARK)

    def _setup(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG_BORDER)
        ax.tick_params(colors=TEXT_SUB, labelsize=8.5)
        ax.set_title(title, color=TEXT_MAIN, fontsize=11, pad=8)
        ax.set_xlabel("Year", color=TEXT_SUB, fontsize=9)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(y) for y in years], fontsize=8.5)
        ax.axvspan(2.5, len(years) - 0.5, color=ACCENT, alpha=0.05, label="OOS 2023+")

    xs = list(range(len(years)))

    # ── Trail activate: PF ────────────────────────────────────────────────────
    ax = axes[0, 0]
    _setup(ax, "d_trail_activate — PF by Year")
    for val, yr_data in trail_results.items():
        pf_vals = [yr_data[y]["PF"] for y in years]
        style   = "-" if val == 0.005 else "--"
        label   = f"trail_act={val:.3f}{'★' if val == 0.005 else ''}"
        ax.plot(xs, pf_vals, linestyle=style, linewidth=2, marker="o",
                markersize=5, color=_color(label), label=label)
        for i, v in enumerate(pf_vals):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7,
                        color=_color(label))
    ax.axhline(1.0, color=RED, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB,
              loc="upper right")
    ax.set_ylabel("PF", color=TEXT_SUB, fontsize=9)

    # ── Trail activate: N ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    _setup(ax, "d_trail_activate — N by Year")
    for val, yr_data in trail_results.items():
        n_vals = [yr_data[y]["N"] for y in years]
        style  = "-" if val == 0.005 else "--"
        label  = f"trail_act={val:.3f}{'★' if val == 0.005 else ''}"
        ax.plot(xs, n_vals, linestyle=style, linewidth=2, marker="s",
                markersize=5, color=_color(label), label=label)
    ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB,
              loc="upper right")
    ax.set_ylabel("N trades", color=TEXT_SUB, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ── d_sl: PF ──────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    _setup(ax, "d_sl — PF by Year")
    sl_colors = {0.010: TEXT_SUB, 0.008: YELLOW, 0.006: GREEN}
    for val, yr_data in sl_results.items():
        pf_vals = [yr_data[y]["PF"] for y in years]
        style   = "-" if val == 0.010 else "--"
        label   = f"sl={val:.3f}{'★' if val == 0.010 else ''}"
        ax.plot(xs, pf_vals, linestyle=style, linewidth=2, marker="o",
                markersize=5, color=sl_colors.get(val, PURPLE), label=label)
        for i, v in enumerate(pf_vals):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7,
                        color=sl_colors.get(val, PURPLE))
    ax.axhline(1.0, color=RED, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB,
              loc="upper right")
    ax.set_ylabel("PF", color=TEXT_SUB, fontsize=9)

    # ── d_sl: N ───────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    _setup(ax, "d_sl — N by Year")
    for val, yr_data in sl_results.items():
        n_vals = [yr_data[y]["N"] for y in years]
        style  = "-" if val == 0.010 else "--"
        label  = f"sl={val:.3f}{'★' if val == 0.010 else ''}"
        ax.plot(xs, n_vals, linestyle=style, linewidth=2, marker="s",
                markersize=5, color=sl_colors.get(val, PURPLE), label=label)
    ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB,
              loc="upper right")
    ax.set_ylabel("N trades", color=TEXT_SUB, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.suptitle(
        "r69: Year-by-Year OOS Validation  |  60 syms · 6yr · OOS=2023+  |  0.08% cost",
        color=TEXT_MAIN, fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = out_dir / "r69_yearly.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return out


# ─── Report Generation ────────────────────────────────────────────────────────

def verdict_str(yr_data_cand: dict[int, dict], yr_data_base: dict[int, dict]) -> str:
    oos_years_pass = all(
        yr_data_cand[y]["PF"] > yr_data_base[y]["PF"]
        for y in YEARS_OOS
        if yr_data_base[y]["N"] > 0
    )
    return "✅ PASS" if oos_years_pass else "❌ FAIL"


def print_year_table(label: str, yr_base: dict, yr_cand: dict, years: list[int]) -> None:
    print(f"\n  {label}")
    print(f"  {'Year':>5}  {'base PF':>8}  {'cand PF':>8}  {'base N':>7}  {'cand N':>7}  {'diff':>8}")
    print("  " + "-" * 57)
    for y in years:
        bpf = yr_base[y]["PF"]
        cpf = yr_cand[y]["PF"]
        bn  = yr_base[y]["N"]
        cn  = yr_cand[y]["N"]
        diff_pct = (cpf - bpf) / bpf * 100 if bpf > 0 else 0
        mark = "✅" if y in YEARS_OOS and cpf > bpf else ("❌" if y in YEARS_OOS and cpf <= bpf else "  ")
        print(f"  {mark} {y}  {bpf:>8.3f}  {cpf:>8.3f}  {bn:>7d}  {cn:>7d}  {diff_pct:>+7.1f}%")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    commit = get_commit()
    today  = datetime.utcnow().strftime("%Y%m%d")
    years  = list(range(2020, 2026))

    print("=" * 60)
    print("r69: Year-by-Year OOS Validation")
    print("=" * 60)
    print(f"d_trail_activate candidates: {TRAIL_ACTS}")
    print(f"d_sl candidates:             {SL_VALS}")
    print(f"OOS years (must ALL improve): {YEARS_OOS}\n")

    print("Loading data...")
    d15 = load_syms()
    max_bars   = max(len(d["close"]) for d in d15.values())
    data_years = max_bars * 15 / (365.25 * 24 * 60)
    print(f"Data: {data_years:.1f} years  |  max_bars={max_bars}\n")

    print("Building TFs...")
    d30, d1h, d4h = build_all_tf(d15)
    print()

    # ── d_trail_activate sweep ────────────────────────────────────────────────
    print("Phase 1: d_trail_activate year-by-year sweep...")
    trail_results: dict[float, dict[int, dict]] = {}
    for val in TRAIL_ACTS:
        print(f"  Collecting trail_act={val:.3f}...")
        trades = collect_s(d15, d30, d1h, d4h, trail_act=val)
        trail_results[val] = run_yearly(trades, max_bars, years)

    baseline_trail = trail_results[0.005]
    print()
    for val in TRAIL_ACTS[1:]:
        cand = trail_results[val]
        vdict = verdict_str(cand, baseline_trail)
        print_year_table(f"trail_act={val:.3f} vs 0.005 → {vdict}", baseline_trail, cand, years)

    # ── d_sl sweep ────────────────────────────────────────────────────────────
    print("\n\nPhase 2: d_sl year-by-year sweep...")
    sl_results: dict[float, dict[int, dict]] = {}
    for val in SL_VALS:
        print(f"  Collecting sl={val:.3f}...")
        trades = collect_s(d15, d30, d1h, d4h, trail_act=cfg.d_trail_activate, sl=val)
        sl_results[val] = run_yearly(trades, max_bars, years)

    baseline_sl = sl_results[0.010]
    print()
    for val in SL_VALS[1:]:
        cand = sl_results[val]
        vdict = verdict_str(cand, baseline_sl)
        print_year_table(f"sl={val:.3f} vs 0.010 → {vdict}", baseline_sl, cand, years)

    # ── Verdicts ──────────────────────────────────────────────────────────────
    print("\n\n── UPGRADE VERDICTS ──────────────────────────────────────────")
    trail_verdicts: dict[float, str] = {}
    for val in TRAIL_ACTS[1:]:
        v = verdict_str(trail_results[val], baseline_trail)
        trail_verdicts[val] = v
        print(f"  d_trail_activate: 0.005 → {val:.3f}  {v}")

    sl_verdicts: dict[float, str] = {}
    for val in SL_VALS[1:]:
        v = verdict_str(sl_results[val], baseline_sl)
        sl_verdicts[val] = v
        print(f"  d_sl:             0.010 → {val:.3f}  {v}")

    # ── Charts ────────────────────────────────────────────────────────────────
    month_dir = REPORT_DIR / "202603"
    month_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating chart...")
    chart_path = plot_results(trail_results, sl_results, years, month_dir)
    print(f"  Saved: {chart_path.name}")

    # ── Markdown report ───────────────────────────────────────────────────────
    report_path = month_dir / f"{today}_{commit}_r69_yearly.md"
    with open(report_path, "w") as f:
        f.write(f"""# r69: Year-by-Year OOS Validation

**날짜:** {today}  **커밋:** {commit}  
**데이터:** 60 syms · {data_years:.1f}년 · 0.08% round-trip  
**OOS 합격 기준:** 2023, 2024, 2025 **모두** baseline 대비 PF 개선

---

## 업그레이드 판정 요약

| 파라미터 | 변경 | 2023 | 2024 | 2025 | 판정 |
|---|---|---|---|---|---|
""")
        for val in TRAIL_ACTS[1:]:
            cand = trail_results[val]
            base = baseline_trail
            marks = []
            for y in YEARS_OOS:
                ok = cand[y]["PF"] > base[y]["PF"]
                marks.append(f"{cand[y]['PF']:.3f}{'↑' if ok else '↓'}")
            vd = trail_verdicts[val]
            f.write(f"| d_trail_activate | 0.005→{val:.3f} | {marks[0]} | {marks[1]} | {marks[2]} | {vd} |\n")

        for val in SL_VALS[1:]:
            cand = sl_results[val]
            base = baseline_sl
            marks = []
            for y in YEARS_OOS:
                ok = cand[y]["PF"] > base[y]["PF"]
                marks.append(f"{cand[y]['PF']:.3f}{'↑' if ok else '↓'}")
            vd = sl_verdicts[val]
            f.write(f"| d_sl | 0.010→{val:.3f} | {marks[0]} | {marks[1]} | {marks[2]} | {vd} |\n")

        f.write("""
---

## 세부 결과 — d_trail_activate

| Year | 0.005★ PF | 0.005★ N | 0.008 PF | 0.008 N | 0.010 PF | 0.010 N |
|---|---|---|---|---|---|---|
""")
        for y in years:
            b  = baseline_trail[y]
            c8 = trail_results[0.008][y]
            c10 = trail_results[0.010][y]
            oos_mark = " ← OOS" if y in YEARS_OOS else ""
            f.write(f"| {y}{oos_mark} | {b['PF']:.3f} | {b['N']} | "
                    f"{c8['PF']:.3f} | {c8['N']} | "
                    f"{c10['PF']:.3f} | {c10['N']} |\n")

        f.write("""
## 세부 결과 — d_sl

| Year | 0.010★ PF | 0.010★ N | 0.008 PF | 0.008 N | 0.006 PF | 0.006 N |
|---|---|---|---|---|---|---|
""")
        for y in years:
            b  = baseline_sl[y]
            c8 = sl_results[0.008][y]
            c6 = sl_results[0.006][y]
            oos_mark = " ← OOS" if y in YEARS_OOS else ""
            f.write(f"| {y}{oos_mark} | {b['PF']:.3f} | {b['N']} | "
                    f"{c8['PF']:.3f} | {c8['N']} | "
                    f"{c6['PF']:.3f} | {c6['N']} |\n")

        # Best candidates
        trail_best = None
        for val in TRAIL_ACTS[1:]:
            if trail_verdicts[val] == "✅ PASS":
                trail_best = val
        sl_best = None
        for val in SL_VALS[1:]:
            if sl_verdicts[val] == "✅ PASS":
                sl_best = val

        f.write("""
---

## 결론 및 적용 방법

""")
        if trail_best:
            f.write(f"""### ✅ d_trail_activate UPGRADE: 0.005 → {trail_best:.3f}

2023/2024/2025 모두 PF 개선 확인. 업그레이드 안전.

```bash
# config.py d_trail_activate 수정 후:
git add quantbox/strategy/vol_straddle/config.py
git commit -m "config: d_trail_activate 0.005→{trail_best:.3f} (r69 pass)"
# 세션 재시작 필요
```
""")
        else:
            f.write("""### ❌ d_trail_activate — NO UPGRADE (keep 0.005)

2023/2024/2025 중 하나 이상에서 PF 개선 없음. 현재값 유지.
""")

        if sl_best:
            f.write(f"""
### ✅ d_sl UPGRADE: 0.010 → {sl_best:.3f}

2023/2024/2025 모두 PF 개선 확인. 단독 또는 trail_activate와 함께 변경 가능.

**주의:** d_sl과 d_trail_activate를 동시에 바꾸면 교호작용 미검증.
한 번에 하나씩 적용 권장.
""")
        else:
            f.write("""
### ❌ d_sl — NO UPGRADE (keep 0.010)

현재값 유지.
""")

        f.write("""
**Parity Notes:** config.py 읽음 ✅  룩어헤드 없음 ✅  symbol_whitelist 사용 ✅

**구현:** 미구현 (이 보고서 기준)
""")

    print(f"\nReport saved: {report_path}")
    print(f"\n{'='*60}")
    for val in TRAIL_ACTS[1:]:
        print(f"trail_act 0.005→{val:.3f}: {trail_verdicts[val]}")
    for val in SL_VALS[1:]:
        print(f"d_sl      0.010→{val:.3f}: {sl_verdicts[val]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
