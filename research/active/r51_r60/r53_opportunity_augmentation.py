#!/usr/bin/env python3
"""r53: v14.4.x 기회 증강 (Opportunity Augmentation).

Phase 0: Pre-flight config 확인
Phase 1: Baseline 재현 (N≈4097, PF≈1.521, Sharpe≈5.06, TBR OFF)
Phase 2: S_d8 trailing stop sweep (현재 flat TP=5%)
Phase 3: Universe expansion (50→100→150→all)
Phase 4: VR threshold sweep
Phase 5: Time-of-day / Day-of-week 필터 분석
Phase 6: 최종 Combined best config
"""
from __future__ import annotations
import sys, time, subprocess, itertools
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import (
    portfolio_sim, compute_metrics, bootstrap_pf,
    build_resampled, precompute, precompute_all_vr,
    collect_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF,
)
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_signal, gen_vr_dur_signal, make_direction_momentum,
)
from research.active.r41_r45.backtest_v14_3_0 import (
    D_SL, D_TP, D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
    collect_d_trades_trailing,
)
from quantbox.backtest.trade_sim import sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Phase 0: Config 확인 ──────────────────────────────────────────────────────
cfg = VolStraddleConfig()
assert cfg.d_sl == D_SL,   f"d_sl mismatch: {cfg.d_sl} vs {D_SL}"
assert cfg.d_tp == D_TP,   f"d_tp mismatch: {cfg.d_tp} vs {D_TP}"
assert not cfg.f5_tbr_enabled, "TBR must be OFF for v14.4.x baseline"

VR_LN    = cfg.base_leverage / cfg.max_concurrent    # 3.0
MA_LN    = cfg.ma_cluster_leverage / cfg.max_concurrent  # 2.1
MAX_CONC = cfg.max_concurrent  # 10

CFG_BL = {
    "sp": 8, "lp": 56,
    "d_vr": cfg.d_vr_threshold, "s_vr": cfg.s_vr_threshold,
    "s_dur": cfg.s_squeeze_dur,
    "d_sl": D_SL, "s_sl": cfg.s_sl, "tp": cfg.s_tp,
}

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
except Exception:
    COMMIT = "unknown"

print("=" * 72)
print("r53: Opportunity Augmentation — v14.4.x")
print(f"  commit={COMMIT}")
print(f"  [Phase 0] d_sl={D_SL}  d_tp={D_TP}")
print(f"  [Phase 0] d_trail=({D_TRAIL_ACTIVATE},{D_TRAIL_DISTANCE})")
print(f"  [Phase 0] s_sl={cfg.s_sl}  s_tp={cfg.s_tp} (flat TP, no trailing)")
print(f"  [Phase 0] CFG_BL={CFG_BL}")
print(f"  [Phase 0] VR_LN={VR_LN:.1f}  MA_LN={MA_LN:.1f}  TBR={cfg.f5_tbr_enabled}")
print(f"  [Phase 0] G_3TF={G_3TF}  MA_TF={MA_TF_RESTRICTION}")
print("=" * 72)

# ── Data Loading ──────────────────────────────────────────────────────────────
CACHE_DIR = ROOT / "data" / "cache_extended"
EXCLUDE   = set(cfg.exclude_symbols)

def _bar_count(f: Path) -> int:
    return int(np.load(f, allow_pickle=False)["close"].shape[0])

def load_top_n(n: int | None, min_years: float = 5.0) -> tuple[dict, str]:
    """Load top-N symbols by bar count. Returns (data_15m, ref_fpath)."""
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym in EXCLUDE:
            continue
        nb = _bar_count(f)
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    min_bars = int(min_years * 365 * 24 * 4)
    ranked = sorted(
        [(s, p, nb) for s, (p, nb) in seen.items() if nb >= min_bars],
        key=lambda x: x[2], reverse=True,
    )
    if n is not None:
        ranked = ranked[:n]

    data: dict = {}
    ref_file = str(ranked[0][1]) if ranked else ""
    for sym, fpath, _ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}

    max_bars = max(len(d["close"]) for d in data.values())
    for d in data.values():
        d["_offset"] = max_bars - len(d["close"])
    return data, ref_file

t0 = time.time()
print(f"\nLoading all available symbols (min 5yr)...")
data_15m_all, ref_file = load_top_n(n=None, min_years=5.0)
all_syms  = list(data_15m_all.keys())
MAX_BARS  = max(len(d["close"]) for d in data_15m_all.values())
DATA_YEARS = MAX_BARS / (4 * 24 * 365)
print(f"  {len(all_syms)} symbols | {DATA_YEARS:.2f} years | {time.time()-t0:.1f}s")

# Derive bar-0 datetime from filename for time-of-day analysis
def _get_bar0_dt(fpath: str) -> datetime:
    fname = Path(fpath).name
    end_str = fname.split("_")[-1].replace(".npz", "")
    end_dt  = datetime.strptime(end_str, "%Y%m%d")
    days    = int(fname.split("_")[2].replace("d", ""))
    return end_dt - timedelta(days=days)

BAR0_DT = _get_bar0_dt(ref_file)
print(f"  Bar-0 = {BAR0_DT.strftime('%Y-%m-%d')} UTC  (for time-of-day mapping)")

# ── Precompute helpers ─────────────────────────────────────────────────────────
def build_all_tf(data_15m: dict) -> tuple[dict, dict]:
    d1h = build_resampled(data_15m, 4)
    d4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_15m, d1h, d4h], [(8, 56)])
    for ds in [data_15m, d1h, d4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
    precompute_mas(data_15m)
    precompute_mas(d1h)
    precompute_mas(d4h)
    return d1h, d4h

# ── collect S_d8 with trailing ─────────────────────────────────────────────────
def _dedup_append(sym: str, sym_trades: list, out: list) -> None:
    sym_trades.sort(key=lambda x: x[0])
    prev_exit = -1
    for entry_bar, exit_bar, pnl in sym_trades:
        if entry_bar > prev_exit:
            out.append((sym, entry_bar, exit_bar, pnl))
            prev_exit = exit_bar

def collect_s_trailing(
    data_15m: dict, data_1h: dict, data_4h: dict,
    s_act: float, s_dist: float,
) -> list:
    vr_key = "vr_8_56"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    out: list = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list = []
        for _, ratio, warmup, _, mh_s in G_3TF:
            ds = tf_map.get(ratio)
            if ds is None or sym not in ds:
                continue
            d   = ds[sym]
            vr  = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig,
                cfg.s_sl, cfg.s_tp, mh_s, COST_STRADDLE,
                s_act, s_dist,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + (ratio - 1),
                    float(p[k]),
                ))
        _dedup_append(sym, sym_trades, out)
    return out

# ── Collect MA trades ──────────────────────────────────────────────────────────
def collect_ma(d15: dict, d1h: dict, d4h: dict) -> list:
    return collect_enhanced_trades(
        d15, d1h, d4h,
        cluster_pct=MA_CLUSTER_PCT,
        min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION,
        min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=MA_TF_RESTRICTION,
    )

# ── run_combined ───────────────────────────────────────────────────────────────
def run_combined(
    d15: dict, d1h: dict, d4h: dict,
    s_trail_act: float | None = None,
    s_trail_dist: float | None = None,
    d_vr_override: float | None = None,
    label: str = "",
) -> dict:
    t = time.time()
    bl_cfg = dict(CFG_BL)
    if d_vr_override is not None:
        bl_cfg["d_vr"] = d_vr_override

    d_trades  = collect_d_trades_trailing(bl_cfg, d15, d1h, d4h)

    if s_trail_act is not None:
        s_trades = collect_s_trailing(d15, d1h, d4h, s_trail_act, s_trail_dist)
    else:
        s_trades = collect_trades(bl_cfg, d15, d1h, d4h, "S_d8")

    ma_trades = collect_ma(d15, d1h, d4h)

    all_trades = d_trades + s_trades + ma_trades
    pnl = portfolio_sim(all_trades, MAX_CONC)
    m   = compute_metrics(pnl, DATA_YEARS)
    bt  = bootstrap_pf(pnl, n_boot=300)

    m["label"]   = label
    m["elapsed"] = round(time.time() - t, 1)
    m["ci"]      = f"[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]"
    m["n_syms"]  = len(d15)
    m["raw_d"]   = len(d_trades)
    m["raw_s"]   = len(s_trades)
    m["raw_ma"]  = len(ma_trades)
    return m

def fmt(m: dict) -> str:
    return (f"N={m['N']:5d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
            f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.2f}  "
            f"DD={m['max_dd_pct']:.1f}%  CI={m.get('ci','?')}  [{m.get('elapsed',0)}s]")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Baseline
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
N_BASELINE_SYMS = min(50, len(all_syms))
print(f"Phase 1: Baseline ({N_BASELINE_SYMS} symbols, TBR OFF)")
print("─" * 72)

data_50 = {k: data_15m_all[k] for k in all_syms[:N_BASELINE_SYMS]}
d1h_50, d4h_50 = build_all_tf(data_50)
bl = run_combined(data_50, d1h_50, d4h_50, label=f"BASELINE_{N_BASELINE_SYMS}")

print(f"  {fmt(bl)}")
print(f"  D={bl['raw_d']}  S={bl['raw_s']}  MA={bl['raw_ma']}")
print(f"  Expected (50 sym): N≈4097  PF≈1.521  Sharpe≈5.06")

n_ok  = abs(bl["N"] - 4097) / 4097 < 0.15   # wider tolerance if <50 syms
pf_ok = abs(bl["PF"] - 1.521) / 1.521 < 0.10
print("  " + ("✅ Baseline OK" if n_ok and pf_ok
     else f"⚠️  DRIFT  N_Δ={abs(bl['N']-4097)/4097:.1%}  PF_Δ={abs(bl['PF']-1.521)/1.521:.1%}"))

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: S_d8 Trailing Sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("Phase 2: S_d8 Trailing Stop Sweep")
print("─" * 72)
print(f"  {'label':<24}  {fmt(bl)}")
print(f"  {'-'*24}  {'-'*75}")

s_acts  = [0.010, 0.015, 0.020, 0.025, 0.030]
s_dists = [0.002, 0.003, 0.005]

p2_results = []
for act, dist in itertools.product(s_acts, s_dists):
    label = f"act={act:.1%}_dist={dist:.1%}"
    m = run_combined(data_50, d1h_50, d4h_50,
                     s_trail_act=act, s_trail_dist=dist, label=label)
    p2_results.append(m)
    flag = " ★" if m["PF"] > bl["PF"] else ""
    print(f"  {label:<24}  {fmt(m)}{flag}")

best_p2 = max(p2_results, key=lambda x: x["PF"])
print(f"\n  Best P2: {best_p2['label']}  PF={best_p2['PF']:.3f}  "
      f"vs baseline {bl['PF']:.3f}  Δ={best_p2['PF']-bl['PF']:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Universe Expansion
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("Phase 3: Universe Expansion")
print("─" * 72)

universe_sizes = sorted(set([N_BASELINE_SYMS, min(100, len(all_syms)),
                              min(150, len(all_syms)), len(all_syms)]))
p3_results: list[dict] = []
for n_u in universe_sizes:
    if n_u == N_BASELINE_SYMS:
        m = dict(bl)  # reuse Phase 1 result for baseline size
        p3_results.append(m)
        print(f"  {n_u:>3d} syms: {fmt(m)}  (Phase 1 reuse)")
        continue
    data_n = {k: data_15m_all[k] for k in all_syms[:n_u]}
    d1h_n, d4h_n = build_all_tf(data_n)
    m = run_combined(data_n, d1h_n, d4h_n, label=f"UNIV_{n_u}")
    p3_results.append(m)
    flag = " ★" if m["PF"] > bl["PF"] else ""
    print(f"  {n_u:>3d} syms: {fmt(m)}{flag}")

best_p3 = max(p3_results, key=lambda x: x["PF"])
print(f"\n  Best P3: {best_p3['label']}  PF={best_p3['PF']:.3f}  "
      f"Δ={best_p3['PF']-bl['PF']:+.3f}")

# Pick the best universe data for remaining phases
best_n = int(best_p3["label"].split("_")[1])
if best_n != 50:
    data_best_u  = {k: data_15m_all[k] for k in all_syms[:best_n]}
    d1h_best_u, d4h_best_u = build_all_tf(data_best_u)
    print(f"  → Using {best_n} symbols for Phases 4+")
else:
    data_best_u, d1h_best_u, d4h_best_u = data_50, d1h_50, d4h_50

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: VR Threshold Sweep (d_vr)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("Phase 4: VR Threshold Sweep (d_vr: 0.40 / 0.45 / 0.48 / 0.50)")
print("─" * 72)

d_vr_vals = [0.40, 0.45, 0.48, 0.50]
p4_results: list[dict] = []
for dvr in d_vr_vals:
    m = run_combined(data_best_u, d1h_best_u, d4h_best_u,
                     d_vr_override=dvr, label=f"d_vr={dvr:.2f}")
    p4_results.append(m)
    flag = " ★" if m["PF"] > bl["PF"] and dvr != 0.50 else ""
    print(f"  d_vr={dvr:.2f}: {fmt(m)}{flag}")

non_baseline_p4 = [r for r in p4_results if r["label"] != "d_vr=0.50"]
best_p4 = max(non_baseline_p4, key=lambda x: x["PF"]) if non_baseline_p4 else bl
print(f"\n  Best P4: {best_p4['label']}  PF={best_p4['PF']:.3f}  "
      f"Δ={best_p4['PF']-bl['PF']:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Time-of-Day / Day-of-Week Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("Phase 5: Time-of-Day & Day-of-Week Analysis")
print("─" * 72)

d_raw  = collect_d_trades_trailing(CFG_BL, data_50, d1h_50, d4h_50)
s_raw  = collect_trades(CFG_BL, data_50, d1h_50, d4h_50, "S_d8")
ma_raw = collect_ma(data_50, d1h_50, d4h_50)
all_raw = d_raw + s_raw + ma_raw

DAYS_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hour_pnl:    dict[int, list[float]] = {h: [] for h in range(24)}
weekday_pnl: dict[int, list[float]] = {d: [] for d in range(7)}
session_pnl: dict[str, list[float]] = {
    "Asia    (00-08 UTC)": [],
    "London  (08-16 UTC)": [],
    "NY      (16-24 UTC)": [],
}

for sym, entry_bar, exit_bar, pnl in all_raw:
    dt = BAR0_DT + timedelta(minutes=15 * entry_bar)
    h, wd = dt.hour, dt.weekday()
    hour_pnl[h].append(pnl)
    weekday_pnl[wd].append(pnl)
    if h < 8:
        session_pnl["Asia    (00-08 UTC)"].append(pnl)
    elif h < 16:
        session_pnl["London  (08-16 UTC)"].append(pnl)
    else:
        session_pnl["NY      (16-24 UTC)"].append(pnl)

# overall EV for reference
all_ev = np.mean([p for _, _, _, p in all_raw]) * 10000

print(f"\n  Overall EV = {all_ev:+.1f}bp  (N={len(all_raw)})")
print(f"\n  [Hour-of-Day EV]  (UTC)")
print(f"  {'Hr':>2}  {'N':>5}  {'EV(bp)':>8}  {'WR%':>6}  bar")
for h in range(24):
    vals = hour_pnl[h]
    if len(vals) < 10:
        continue
    ev  = np.mean(vals) * 10000
    wr  = np.mean([v > 0 for v in vals]) * 100
    bar = "█" * int(max(0, ev) / 3) + ("░" * int(max(0, -ev) / 3) if ev < 0 else "")
    flag = " ◀" if ev == max(np.mean(v) for v in hour_pnl.values() if v) * 10000 else ""
    print(f"  {h:02d}  {len(vals):>5}  {ev:>+8.1f}  {wr:>6.1f}%  {bar}{flag}")

print(f"\n  [Day-of-Week EV]")
print(f"  {'Day':>5}  {'N':>5}  {'EV(bp)':>8}  {'WR%':>6}")
max_dow_ev = max(np.mean(v)*10000 for v in weekday_pnl.values() if v)
for wd in range(7):
    vals = weekday_pnl[wd]
    if not vals:
        continue
    ev  = np.mean(vals) * 10000
    wr  = np.mean([v > 0 for v in vals]) * 100
    flag = " ◀ best" if abs(ev - max_dow_ev) < 0.1 else ""
    print(f"  {DAYS_EN[wd]:>5}  {len(vals):>5}  {ev:>+8.1f}  {wr:>6.1f}%{flag}")

print(f"\n  [Trading Session]")
for sess, vals in session_pnl.items():
    if not vals:
        continue
    ev = np.mean(vals) * 10000
    wr = np.mean([v > 0 for v in vals]) * 100
    flag = " ◀" if ev == max(np.mean(v)*10000 for v in session_pnl.values() if v) else ""
    print(f"  {sess}  N={len(vals):>5}  EV={ev:>+7.1f}bp  WR={wr:.1f}%{flag}")

# Find best hours by EV
ev_by_h = {h: np.mean(v)*10000 for h, v in hour_pnl.items() if len(v) >= 30}
sorted_h = sorted(ev_by_h.items(), key=lambda x: x[1], reverse=True)
best_hours  = {h for h, _ in sorted_h[:8]}
worst_hours = {h for h, _ in sorted_h[-8:]}
print(f"\n  Top-8 hours: {sorted([f'{h:02d}:00' for h in best_hours])}")
print(f"  Bot-8 hours: {sorted([f'{h:02d}:00' for h in worst_hours])}")

# Run time-filtered backtests
def filter_by_time(trades: list, allowed_h: set, allowed_wd: set | None = None) -> list:
    out = []
    for sym, eb, xb, pnl in trades:
        dt = BAR0_DT + timedelta(minutes=15 * eb)
        if dt.hour not in allowed_h:
            continue
        if allowed_wd and dt.weekday() not in allowed_wd:
            continue
        out.append((sym, eb, xb, pnl))
    return out

# Best-8h filter
bh_trades = filter_by_time(all_raw, best_hours)
pnl_bh    = portfolio_sim(bh_trades, MAX_CONC)
m_bh      = compute_metrics(pnl_bh, DATA_YEARS)
bt_bh     = bootstrap_pf(pnl_bh, n_boot=300)
m_bh["ci"] = f"[{bt_bh['CI_lo']:.3f},{bt_bh['CI_hi']:.3f}]"

# Worst-8h exclusion (remove worst hours)
excl_trades = filter_by_time(all_raw, set(range(24)) - worst_hours)
pnl_excl    = portfolio_sim(excl_trades, MAX_CONC)
m_excl      = compute_metrics(pnl_excl, DATA_YEARS)
bt_excl     = bootstrap_pf(pnl_excl, n_boot=300)
m_excl["ci"] = f"[{bt_excl['CI_lo']:.3f},{bt_excl['CI_hi']:.3f}]"

# London session only
lon_trades = filter_by_time(all_raw, set(range(8, 16)))
pnl_lon    = portfolio_sim(lon_trades, MAX_CONC)
m_lon      = compute_metrics(pnl_lon, DATA_YEARS)
bt_lon     = bootstrap_pf(pnl_lon, n_boot=300)
m_lon["ci"] = f"[{bt_lon['CI_lo']:.3f},{bt_lon['CI_hi']:.3f}]"

# Weekday-only (Mon-Fri)
wd_trades = filter_by_time(all_raw, set(range(24)), allowed_wd={0,1,2,3,4})
pnl_wd    = portfolio_sim(wd_trades, MAX_CONC)
m_wd      = compute_metrics(pnl_wd, DATA_YEARS)
bt_wd     = bootstrap_pf(pnl_wd, n_boot=300)
m_wd["ci"] = f"[{bt_wd['CI_lo']:.3f},{bt_wd['CI_hi']:.3f}]"

print(f"\n  Time-filter comparison:")
print(f"  {'Config':<30}  {'N':>5}  {'PF':>6}  {'EV':>7}  {'Sharpe':>7}  CI")
rows_t = [
    ("Baseline (all)", bl),
    ("Best-8h filter",  {**m_bh,   "n_syms": 50}),
    ("Worst-8h exclude",{**m_excl, "n_syms": 50}),
    ("London (08-16)",  {**m_lon,  "n_syms": 50}),
    ("Weekday only",    {**m_wd,   "n_syms": 50}),
]
for lbl, m in rows_t:
    flag = " ★" if m["PF"] > bl["PF"] and lbl != "Baseline (all)" else ""
    print(f"  {lbl:<30}  {m['N']:>5}  {m['PF']:>6.3f}  {m['EV_bp']:>+7.1f}  "
          f"{m['calmar']:>7.2f}  {m.get('ci','')}{flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FINAL SUMMARY — All Phases vs Baseline")
print("=" * 72)
print(f"  {'Config':<35}  {'N':>5}  {'PF':>6}  {'EV(bp)':>7}  {'Calmar':>7}  {'DD%':>6}")
print(f"  {'─'*35}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}")

p6_rows: list[tuple[str, dict]] = [
    (f"BASELINE ({N_BASELINE_SYMS} syms, production)", bl),
    (f"P2: S_d8 trail best ({best_p2['label']})", best_p2),
]
for r in p3_results[1:]:
    p6_rows.append((f"P3: {r['label']}", r))
p6_rows.append((f"P4: {best_p4['label']}", best_p4))
p6_rows.append(("P5: Best-8h filter", {**m_bh, "n_syms": N_BASELINE_SYMS}))
p6_rows.append(("P5: Worst-8h exclude", {**m_excl, "n_syms": N_BASELINE_SYMS}))
p6_rows.append(("P5: Weekday only", {**m_wd, "n_syms": N_BASELINE_SYMS}))

for lbl, m in p6_rows:
    if m is None:
        continue
    flag = " ★" if m["PF"] > bl["PF"] and "BASELINE" not in lbl else ""
    print(f"  {lbl:<35}  {m['N']:>5}  {m['PF']:.3f}  {m['EV_bp']:>+7.1f}  "
          f"{m['calmar']:>7.2f}  {m['max_dd_pct']:>6.1f}%{flag}")

print(f"\n  commit={COMMIT}  Completed.")
