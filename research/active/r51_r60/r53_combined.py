#!/usr/bin/env python3
"""r53 Phase 6: Combined — S_d8 trailing (act=1.5%/dist=0.2%) + Time filter (worst-8h exclude).

Runs on:
  A) Current 34 syms (5yr+)
  B) 37 syms (4yr+)  
  C) Expanded universe after download (all available 5yr+)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

ROOT = Path('/home/user/Documents/git/quantbox')
sys.path.insert(0, str(ROOT))

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import (
    portfolio_sim, compute_metrics, bootstrap_pf,
    build_resampled, precompute, precompute_all_vr,
    collect_trades, COST_STRADDLE, G_3TF,
)
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal
from research.active.r41_r45.backtest_v14_3_0 import (
    D_SL, D_TP, D_TRAIL_ACTIVATE, D_TRAIL_DISTANCE,
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
    collect_d_trades_trailing,
)
from quantbox.backtest.trade_sim import sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

cfg = VolStraddleConfig()
MAX_CONC = cfg.max_concurrent  # 10
CFG_BL = {"sp": 8, "lp": 56, "d_vr": cfg.d_vr_threshold, "s_vr": cfg.s_vr_threshold,
           "s_dur": cfg.s_squeeze_dur, "d_sl": D_SL, "s_sl": cfg.s_sl, "tp": cfg.s_tp}

# Best params from Phase 2/5
S_TRAIL_ACT  = 0.015
S_TRAIL_DIST = 0.002
# Worst hours to exclude (from Phase 5)
WORST_HOURS = {2, 7, 9, 11, 13, 14, 15, 23}

CACHE_DIR = ROOT / "data" / "cache_extended"
EXCLUDE   = set(cfg.exclude_symbols)

def load_syms(min_years: float) -> tuple[dict, str]:
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym in EXCLUDE: continue
        nb = int(np.load(f, allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    min_bars = int(min_years * 365 * 24 * 4)
    ranked = sorted([(s,p,nb) for s,(p,nb) in seen.items() if nb >= min_bars],
                    key=lambda x: x[2], reverse=True)
    data: dict = {}
    ref = str(ranked[0][1]) if ranked else ""
    for sym, fpath, _ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    mb = max(len(d["close"]) for d in data.values()) if data else 0
    for d in data.values():
        d["_offset"] = mb - len(d["close"])
    return data, ref

def get_bar0(fpath: str) -> datetime:
    fname = Path(fpath).name
    end_dt = datetime.strptime(fname.split("_")[-1].replace(".npz",""), "%Y%m%d")
    days   = int(fname.split("_")[2].replace("d",""))
    return end_dt - timedelta(days=days)

def build_all_tf(d15: dict) -> tuple[dict, dict]:
    d1h = build_resampled(d15, 4); d4h = build_resampled(d15, 16)
    precompute_all_vr([d15, d1h, d4h], [(8, 56)])
    for ds in [d15, d1h, d4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
    precompute_mas(d15); precompute_mas(d1h); precompute_mas(d4h)
    return d1h, d4h

def _dedup_append(sym, sym_trades, out):
    sym_trades.sort(key=lambda x: x[0])
    prev = -1
    for eb, xb, pnl in sym_trades:
        if eb > prev:
            out.append((sym, eb, xb, pnl)); prev = xb

def collect_s_trail(d15, d1h, d4h) -> list:
    tf_map = {1: d15, 4: d1h, 16: d4h}
    out = []
    for sym in sorted(d15):
        st = []
        for _, ratio, warmup, _, mh_s in G_3TF:
            ds = tf_map.get(ratio)
            if ds is None or sym not in ds: continue
            d = ds[sym]; vr = d.get("vr_8_56")
            if vr is None: continue
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x = sim_straddle_trailing(
                d["close"], d["high"], d["low"], sig,
                cfg.s_sl, cfg.s_tp, mh_s, COST_STRADDLE,
                S_TRAIL_ACT, S_TRAIL_DIST,
            )
            for k in range(len(p)):
                st.append((int(e[k])*ratio, int(x[k])*ratio+(ratio-1), float(p[k])))
        _dedup_append(sym, st, out)
    return out

def collect_ma(d15, d1h, d4h) -> list:
    return collect_enhanced_trades(
        d15, d1h, d4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )

def filter_hours(trades: list, bar0: datetime, exclude_h: set) -> list:
    return [(sym, eb, xb, pnl) for sym, eb, xb, pnl in trades
            if (bar0 + timedelta(minutes=15*eb)).hour not in exclude_h]

def run_scenario(d15, d1h, d4h, bar0: datetime,
                 use_trail: bool, use_time_filter: bool, label: str) -> dict:
    t0 = time.time()
    data_years = max(len(d["close"]) for d in d15.values()) / (4*24*365)

    d_tr  = collect_d_trades_trailing(CFG_BL, d15, d1h, d4h)
    s_tr  = collect_s_trail(d15, d1h, d4h) if use_trail else collect_trades(CFG_BL, d15, d1h, d4h, "S_d8")
    ma_tr = collect_ma(d15, d1h, d4h)
    all_tr = d_tr + s_tr + ma_tr

    if use_time_filter:
        all_tr = filter_hours(all_tr, bar0, WORST_HOURS)

    pnl = portfolio_sim(all_tr, MAX_CONC)
    m   = compute_metrics(pnl, data_years)
    bt  = bootstrap_pf(pnl, n_boot=400)
    m["label"] = label; m["n_syms"] = len(d15)
    m["elapsed"] = round(time.time()-t0, 1)
    m["ci"] = f"[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]"
    m["raw_N"] = len(all_tr)
    return m

def fmt(m):
    return (f"N={m['N']:5d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
            f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.2f}  DD={m['max_dd_pct']:.1f}%  "
            f"CI={m['ci']}  syms={m['n_syms']}  [{m['elapsed']}s]")

print("="*72)
print("r53 Phase 6: Combined Scenarios")
print(f"  S_trail: act={S_TRAIL_ACT:.1%} dist={S_TRAIL_DIST:.1%}")
print(f"  Excl hours: {sorted(WORST_HOURS)}")
print("="*72)

all_results = []

for min_yr, label_prefix in [(5.0, "5yr"), (4.0, "4yr")]:
    print(f"\n{'─'*72}")
    print(f"Universe: min_years={min_yr}")
    print(f"{'─'*72}")

    data_15m, ref_file = load_syms(min_yr)
    if not data_15m:
        print(f"  No symbols found for min_years={min_yr}"); continue
    d1h, d4h = build_all_tf(data_15m)
    bar0 = get_bar0(ref_file)
    print(f"  {len(data_15m)} symbols | bar0={bar0.date()}")

    scenarios = [
        (False, False, "Baseline (no trail, no filter)"),
        (True,  False, "S_d8 trail only"),
        (False, True,  "Time filter only (worst-8h excl)"),
        (True,  True,  "Trail + Time filter [COMBINED]"),
    ]

    for use_trail, use_tf, name in scenarios:
        label = f"{label_prefix}/{name}"
        m = run_scenario(data_15m, d1h, d4h, bar0, use_trail, use_tf, label)
        all_results.append(m)
        flag = " ★★" if use_trail and use_tf else (" ★" if use_trail or use_tf else "")
        print(f"  [{name:<37}] {fmt(m)}{flag}")

print("\n" + "="*72)
print("SUMMARY — Combined vs Individual (5yr universe)")
print("="*72)
print(f"  {'Scenario':<42}  {'N':>5}  {'PF':>6}  {'EV':>7}  {'Calmar':>7}  {'DD%':>6}  CI")
print(f"  {'─'*42}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}")

for m in all_results:
    if "5yr" not in m["label"]: continue
    n = m["label"].replace("5yr/","")
    flag = " ★★" if "COMBINED" in n else (" ★" if "only" in n else "")
    print(f"  {n:<42}  {m['N']:>5}  {m['PF']:>6.3f}  {m['EV_bp']:>+7.1f}  "
          f"{m['calmar']:>7.2f}  {m['max_dd_pct']:>6.1f}%  {m['ci']}{flag}")

print("\nDone.")
