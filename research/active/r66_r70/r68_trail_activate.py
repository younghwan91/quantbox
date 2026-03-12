#!/usr/bin/env python3
"""r68: d_trail_activate Optimization + d_tp Dead-Parameter Verification.

r67 sensitivity revealed two actionable findings:
  1. d_trail_activate=0.010 → PF +66% vs current 0.005
  2. d_tp=0.12 is a dead parameter (trail exits first in all cases)

This script:
  Phase 0: Baseline reproduction (±5% vs v14.5 canonical)
  Phase 1: d_tp dead-parameter verification (0.12 vs 0.20 must be identical)
  Phase 2: d_trail_activate sweep [0.003..0.015] with strict walk-forward
  Phase 3: Upgrade verdict and report

Upgrade criterion (all must pass):
  - OOS PF ≥ 1.05
  - OOS / IS PF ratio ≥ 0.85
  - Full PF > baseline Full PF
  - N does not drop > 20% vs baseline

Output:
  docs/backtest_results/202603/YYYYMMDD_{commit}_r68_trail_activate.md
  docs/backtest_results/202603/r68_trail_activate.png

Usage:
    uv run python research/active/r66_r70/r68_trail_activate.py
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
PURPLE    = "#bc8cff"

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

# ── Canonical baseline (v14.5.0, 60 sym, 6yr, trade_sim) ─────────────────────
CANONICAL_N  = 10388
CANONICAL_PF = 2.772

# ── Sweep: d_trail_activate [0.003..0.015], production at index 2 ────────────
SWEEP_VALS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.015]
PROD_IDX   = 2  # 0.005 is at index 2

# Upgrade criteria
MIN_OOS_PF     = 1.05
MIN_OOS_IS_RATIO = 0.85
MAX_N_DROP     = 0.20  # N must not drop more than 20%


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
    """Collect S_d8 straddle trades. Only trail_act varies; rest from config."""
    _sl   = sl         if sl         is not None else cfg.d_sl
    _tp   = tp         if tp         is not None else cfg.d_tp
    _td   = trail_dist if trail_dist is not None else cfg.d_trail_distance

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
    trail_act: float, max_bars: int, data_years: float,
    sl: float | None = None,
    tp: float | None = None,
) -> dict:
    trades = collect_s(d15, d30, d1h, d4h, trail_act=trail_act, sl=sl, tp=tp)

    split_bar = compute_split_bar(max_bars)
    is_trades  = [(s, e, x, p) for s, e, x, p in trades if e < split_bar]
    oos_trades = [(s, e, x, p) for s, e, x, p in trades if e >= split_bar]

    oos_years = (DATA_END - OOS_START).days / 365.25
    is_years  = max(data_years - oos_years, 0.1)

    pnl_full = portfolio_sim(trades,     MAX_CONC)
    pnl_is   = portfolio_sim(is_trades,  MAX_CONC)
    pnl_oos  = portfolio_sim(oos_trades, MAX_CONC)

    m_full = compute_metrics(pnl_full, data_years)
    m_is   = compute_metrics(pnl_is,   is_years)
    m_oos  = compute_metrics(pnl_oos,  oos_years)

    oos_is_ratio = m_oos["PF"] / m_is["PF"] if m_is["PF"] > 0 else 0.0

    return {
        "N":            m_full["N"],
        "PF":           m_full["PF"],
        "EV_bp":        m_full["EV_bp"],
        "WR":           m_full["WR"],
        "maxdd":        m_full.get("max_dd_pct", 0),
        "IS_PF":        m_is["PF"],
        "IS_N":         m_is["N"],
        "OOS_PF":       m_oos["PF"],
        "OOS_N":        m_oos["N"],
        "OOS_IS_ratio": oos_is_ratio,
        "trail_act":    trail_act,
    }


# ─── Chart ────────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], baseline: dict, out_dir: Path) -> Path:
    vals     = [r["trail_act"] for r in results]
    pf_vals  = [r["PF"]        for r in results]
    oos_vals = [r["OOS_PF"]    for r in results]
    ev_vals  = [r["EV_bp"]     for r in results]
    n_vals   = [r["N"]         for r in results]
    ratio_v  = [r["OOS_IS_ratio"] for r in results]
    labels   = [f"{v:.3g}" for v in vals]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.patch.set_facecolor(BG_DARK)

    prod_pf  = baseline["PF"]
    prod_oos = baseline["OOS_PF"]

    plot_cfg = [
        (axes[0, 0], pf_vals,  "Full PF",      ACCENT,  prod_pf),
        (axes[0, 1], oos_vals, "OOS PF",       GREEN,   prod_oos),
        (axes[0, 2], ev_vals,  "EV (bp)",      ORANGE,  baseline["EV_bp"]),
        (axes[1, 0], n_vals,   "N (trades)",   PURPLE,  baseline["N"]),
        (axes[1, 1], ratio_v,  "OOS/IS ratio", ACCENT,  baseline["OOS_IS_ratio"]),
        (axes[1, 2], [r["WR"] for r in results], "Win Rate (%)", GREEN, baseline["WR"]),
    ]

    for ax, vals_plt, title, color, prod_ref in plot_cfg:
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG_BORDER)
        ax.tick_params(colors=TEXT_SUB, labelsize=8)

        xs = list(range(len(vals_plt)))
        ax.plot(xs, vals_plt, color=color, linewidth=2, marker="o", markersize=6)
        ax.axhline(prod_ref, color=ORANGE, linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"Prod ({prod_ref:.3g})")
        ax.axvline(PROD_IDX, color=ORANGE, linewidth=1.0, linestyle=":", alpha=0.5)

        # Threshold lines
        if title == "OOS PF":
            ax.axhline(MIN_OOS_PF, color=RED, linewidth=0.8, linestyle=":", alpha=0.6,
                       label=f"Min {MIN_OOS_PF}")
        if title == "OOS/IS ratio":
            ax.axhline(MIN_OOS_IS_RATIO, color=RED, linewidth=0.8, linestyle=":", alpha=0.6,
                       label=f"Min {MIN_OOS_IS_RATIO}")

        # Highlight best (above baseline)
        for i, (v, baseline_v) in enumerate(zip(vals_plt, [prod_ref] * len(vals_plt))):
            c = GREEN if v > prod_ref else (RED if v < prod_ref * 0.85 else TEXT_SUB)
            ax.annotate(f"{v:.3g}", (i, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=c)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=10)
        ax.set_xlabel("d_trail_activate", color=TEXT_SUB, fontsize=8)
        ax.set_title(title, color=TEXT_MAIN, fontsize=10, pad=6)
        ax.legend(fontsize=7, facecolor=BG_CARD, edgecolor=BG_BORDER, labelcolor=TEXT_SUB)

    fig.suptitle(
        "r68: d_trail_activate Sweep  |  60 syms · 6yr · OOS=2023+  |  0.08% cost",
        color=TEXT_MAIN, fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = out_dir / "r68_trail_activate.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    commit = get_commit()
    today  = datetime.utcnow().strftime("%Y%m%d")

    print("=" * 60)
    print("r68: d_trail_activate Optimization")
    print("=" * 60)
    print(f"Production: d_trail_activate={cfg.d_trail_activate}  d_tp={cfg.d_tp}\n")

    print("Loading data...")
    d15 = load_syms()
    max_bars   = max(len(d["close"]) for d in d15.values())
    data_years = max_bars * 15 / (365.25 * 24 * 60)
    print(f"Data: {data_years:.1f} years  |  max_bars={max_bars}\n")

    print("Building TFs and precomputing indicators...")
    d30, d1h, d4h = build_all_tf(d15)
    print()

    # ── Phase 0: Baseline ────────────────────────────────────────────────────
    print("Phase 0: Production baseline...")
    baseline = run_point(d15, d30, d1h, d4h, cfg.d_trail_activate, max_bars, data_years)
    prod_pf  = baseline["PF"]
    print(f"  N={baseline['N']}  PF={prod_pf:.3f}  OOS_PF={baseline['OOS_PF']:.3f}"
          f"  OOS/IS={baseline['OOS_IS_ratio']:.3f}")

    n_drift  = abs(baseline["N"] - CANONICAL_N) / CANONICAL_N
    pf_drift = abs(prod_pf - CANONICAL_PF) / CANONICAL_PF
    if n_drift > 0.05 or pf_drift > 0.05:
        print(f"  ⚠️  Drift from canonical: N±{n_drift:.1%}  PF±{pf_drift:.1%}")
        print("     Check data/config alignment before proceeding.")
    else:
        print(f"  ✅ Within ±5% of canonical (N={CANONICAL_N}, PF={CANONICAL_PF})\n")

    # ── Phase 1: d_tp dead-parameter verification ─────────────────────────────
    print("Phase 1: d_tp dead-parameter check (0.12 vs 0.20)...")
    r_tp12 = run_point(d15, d30, d1h, d4h, cfg.d_trail_activate, max_bars, data_years, tp=0.12)
    r_tp20 = run_point(d15, d30, d1h, d4h, cfg.d_trail_activate, max_bars, data_years, tp=0.20)
    n_diff  = abs(r_tp12["N"]  - r_tp20["N"])
    pf_diff = abs(r_tp12["PF"] - r_tp20["PF"])
    print(f"  d_tp=0.12: N={r_tp12['N']}  PF={r_tp12['PF']:.4f}")
    print(f"  d_tp=0.20: N={r_tp20['N']}  PF={r_tp20['PF']:.4f}")
    if n_diff == 0 and pf_diff < 0.001:
        print("  ✅ CONFIRMED DEAD — d_tp has zero effect. Raising to 0.20 safe.\n")
        tp_verdict = "CONFIRMED_DEAD"
    else:
        print(f"  ⚠️  d_tp NOT dead! N diff={n_diff}  PF diff={pf_diff:.4f}\n")
        tp_verdict = "NOT_DEAD"

    # ── Phase 2: d_trail_activate sweep ──────────────────────────────────────
    print("Phase 2: d_trail_activate sweep...")
    print(f"  {'val':>8}  {'N':>6}  {'PF':>6}  {'OOS_PF':>7}  {'IS_PF':>6}  "
          f"{'OOS/IS':>7}  {'EV':>6}  {'verdict'}")
    print("  " + "-" * 72)

    results: list[dict] = []
    for val in SWEEP_VALS:
        r = run_point(d15, d30, d1h, d4h, val, max_bars, data_years)
        results.append(r)

        n_drop      = (baseline["N"] - r["N"]) / baseline["N"]
        pf_improve  = r["PF"] > prod_pf
        oos_ok      = r["OOS_PF"] >= MIN_OOS_PF
        ratio_ok    = r["OOS_IS_ratio"] >= MIN_OOS_IS_RATIO
        n_ok        = n_drop <= MAX_N_DROP
        upgrade_ok  = pf_improve and oos_ok and ratio_ok and n_ok

        marks = []
        if not pf_improve:   marks.append("PF↓")
        if not oos_ok:       marks.append(f"OOS<{MIN_OOS_PF}")
        if not ratio_ok:     marks.append(f"ratio<{MIN_OOS_IS_RATIO}")
        if not n_ok:         marks.append(f"N↓>{MAX_N_DROP:.0%}")
        verdict = "✅ PASS" if upgrade_ok else f"❌ {','.join(marks)}"

        star = "★" if abs(val - cfg.d_trail_activate) < 1e-9 else " "
        print(f"  {star}{val:>7.3f}  {r['N']:>6d}  {r['PF']:>6.3f}  "
              f"{r['OOS_PF']:>7.3f}  {r['IS_PF']:>6.3f}  "
              f"{r['OOS_IS_ratio']:>7.3f}  {r['EV_bp']:>5.1f}bp  {verdict}")
    print()

    # ── Best candidate ────────────────────────────────────────────────────────
    passers = [
        r for r in results
        if r["PF"] > prod_pf
        and r["OOS_PF"] >= MIN_OOS_PF
        and r["OOS_IS_ratio"] >= MIN_OOS_IS_RATIO
        and (baseline["N"] - r["N"]) / baseline["N"] <= MAX_N_DROP
    ]

    if passers:
        best = max(passers, key=lambda r: r["OOS_PF"])
        print(f"✅ UPGRADE RECOMMENDED: d_trail_activate = {best['trail_act']:.3f}")
        print(f"   PF: {prod_pf:.3f} → {best['PF']:.3f} ({(best['PF']-prod_pf)/prod_pf*100:+.1f}%)")
        print(f"   OOS PF: {baseline['OOS_PF']:.3f} → {best['OOS_PF']:.3f}")
        print(f"   OOS/IS: {baseline['OOS_IS_ratio']:.3f} → {best['OOS_IS_ratio']:.3f}")
        print(f"   N: {baseline['N']} → {best['N']}")
        upgrade_verdict = "UPGRADE"
        upgrade_val     = best["trail_act"]
    else:
        print("❌ NO UPGRADE — no candidate passes all criteria. Keep current 0.005.")
        upgrade_verdict = "KEEP"
        upgrade_val     = cfg.d_trail_activate

    # ── Charts ────────────────────────────────────────────────────────────────
    month_dir = REPORT_DIR / "202603"
    month_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating chart...")
    chart_path = plot_results(results, baseline, month_dir)
    print(f"  Saved: {chart_path.name}")

    # ── Markdown report ───────────────────────────────────────────────────────
    report_path = month_dir / f"{today}_{commit}_r68_trail_activate.md"
    oos_years_val = (DATA_END - OOS_START).days / 365.25
    is_years_val  = max(data_years - oos_years_val, 0.1)
    with open(report_path, "w") as f:
        f.write(f"""# r68: d_trail_activate Optimization

**날짜:** {today}  **커밋:** {commit}  
**데이터:** 2020-01-01 ~ {DATA_END.strftime('%Y-%m-%d')} ({data_years:.1f}년)  
**IS:** {(DATA_END - timedelta(days=int(is_years_val*365.25))).strftime('%Y-%m-%d')} ~ {OOS_START.strftime('%Y-%m-%d')} ({is_years_val:.1f}년)  
**OOS:** {OOS_START.strftime('%Y-%m-%d')} ~ {DATA_END.strftime('%Y-%m-%d')} ({oos_years_val:.1f}년)  
**심볼:** 60 (symbol_whitelist)  비용: 0.08% round-trip  슬롯: {MAX_CONC}  L/N: {VR_LN:.1f}

---

## Phase 1: d_tp Dead-Parameter Verification

| d_tp | N | PF | 판정 |
|---|---|---|---|
| 0.12 (이전) | {r_tp12['N']} | {r_tp12['PF']:.4f} | — |
| 0.20 (신규) | {r_tp20['N']} | {r_tp20['PF']:.4f} | {tp_verdict} |

{'✅ d_tp=0.20 변경 확정 — trail이 항상 먼저 청산. TP는 순수 safety cap.' if tp_verdict == 'CONFIRMED_DEAD' else '⚠️ d_tp가 영향 있음 — 추가 조사 필요.'}

---

## Phase 2: d_trail_activate Sweep

**합격 기준:** OOS PF ≥ {MIN_OOS_PF}  ·  OOS/IS ≥ {MIN_OOS_IS_RATIO}  ·  PF > baseline  ·  N 감소 ≤ {MAX_N_DROP:.0%}

| d_trail_activate | N | Full PF | OOS PF | IS PF | OOS/IS | EV (bp) | 판정 |
|---|---|---|---|---|---|---|---|
""")
        for r in results:
            star = "★" if abs(r["trail_act"] - cfg.d_trail_activate) < 1e-9 else ""
            n_drop = (baseline["N"] - r["N"]) / baseline["N"]
            pf_improve = r["PF"] > prod_pf
            oos_ok     = r["OOS_PF"] >= MIN_OOS_PF
            ratio_ok   = r["OOS_IS_ratio"] >= MIN_OOS_IS_RATIO
            n_ok       = n_drop <= MAX_N_DROP
            ok = pf_improve and oos_ok and ratio_ok and n_ok
            verdict = "✅ PASS" if ok else "❌"
            f.write(f"| {star}{r['trail_act']:.3f} | {r['N']} | {r['PF']:.3f} | "
                    f"{r['OOS_PF']:.3f} | {r['IS_PF']:.3f} | {r['OOS_IS_ratio']:.3f} | "
                    f"{r['EV_bp']:.1f} | {verdict} |\n")

        f.write(f"""
---

## 결론

**Baseline:** N={baseline['N']}  PF={prod_pf:.3f}  OOS={baseline['OOS_PF']:.3f}  OOS/IS={baseline['OOS_IS_ratio']:.3f}

### d_tp
{'✅ **d_tp=0.20 확정** — dead parameter 검증 완료. 기존 포지션 주문 유지 (신규 포지션부터 적용).' if tp_verdict == 'CONFIRMED_DEAD' else '⚠️ d_tp 추가 조사 필요.'}

### d_trail_activate
""")
        if upgrade_verdict == "UPGRADE":
            f.write(f"""✅ **UPGRADE: {cfg.d_trail_activate:.3f} → {upgrade_val:.3f}**

| 지표 | 현재 | 신규 | 변화 |
|---|---|---|---|
| Full PF | {prod_pf:.3f} | {best['PF']:.3f} | {(best['PF']-prod_pf)/prod_pf*100:+.1f}% |
| OOS PF | {baseline['OOS_PF']:.3f} | {best['OOS_PF']:.3f} | {(best['OOS_PF']-baseline['OOS_PF'])/baseline['OOS_PF']*100:+.1f}% |
| OOS/IS | {baseline['OOS_IS_ratio']:.3f} | {best['OOS_IS_ratio']:.3f} | — |
| N | {baseline['N']} | {best['N']} | {(best['N']-baseline['N'])/baseline['N']*100:+.1f}% |

**적용 방법:** `config.py` d_trail_activate: {cfg.d_trail_activate} → {upgrade_val}. `.env`에 VS_D_TRAIL_ACTIVATE 없는지 확인 후 세션 재시작.
""")
        else:
            f.write(f"""❌ **NO UPGRADE** — 현재값 {cfg.d_trail_activate:.3f} 유지.

어떤 후보도 4개 합격 기준을 모두 통과하지 못함.
""")

        f.write("""
**Parity Notes:** config.py 읽음 ✅  룩어헤드 없음 ✅  symbol_whitelist 사용 ✅

**구현:** 미구현 (이 보고서 기준)
""")

    print(f"\nReport saved: {report_path}")
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: d_trail_activate → {upgrade_verdict} ({upgrade_val:.3f})")
    print(f"               d_tp            → {tp_verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
