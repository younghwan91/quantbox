#!/usr/bin/env python3
"""r53 추가 조합 테스트: trail × 다양한 시간 필터 조합."""
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
    D_SL, MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
    collect_d_trades_trailing,
)
from quantbox.backtest.trade_sim import sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

cfg = VolStraddleConfig()
MAX_CONC = cfg.max_concurrent
CFG_BL = {"sp": 8, "lp": 56, "d_vr": cfg.d_vr_threshold, "s_vr": cfg.s_vr_threshold,
           "s_dur": cfg.s_squeeze_dur, "d_sl": D_SL, "s_sl": cfg.s_sl, "tp": cfg.s_tp}
CACHE_DIR = ROOT / "data" / "cache_extended"

def load_syms(min_years=5.0):
    seen = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym in cfg.exclude_symbols: continue
        nb = int(np.load(f, allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]: seen[sym] = (f, nb)
    min_bars = int(min_years * 365 * 24 * 4)
    ranked = sorted([(s,p,nb) for s,(p,nb) in seen.items() if nb>=min_bars], key=lambda x:-x[2])
    data = {}; ref = str(ranked[0][1]) if ranked else ""
    for sym, fpath, _ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    mb = max(len(d["close"]) for d in data.values()) if data else 0
    for d in data.values(): d["_offset"] = mb - len(d["close"])
    return data, ref

def get_bar0(fpath):
    fname = Path(fpath).name
    end = datetime.strptime(fname.split("_")[-1].replace(".npz",""), "%Y%m%d")
    return end - timedelta(days=int(fname.split("_")[2].replace("d","")))

def build_tf(d15):
    d1h = build_resampled(d15, 4); d4h = build_resampled(d15, 16)
    precompute_all_vr([d15, d1h, d4h], [(8, 56)])
    for ds in [d15, d1h, d4h]: precompute(ds, vr_sp=8, vr_lp=56)
    precompute_mas(d15); precompute_mas(d1h); precompute_mas(d4h)
    return d1h, d4h

def dedup(sym, st, out):
    st.sort(key=lambda x: x[0]); prev = -1
    for eb, xb, p in st:
        if eb > prev: out.append((sym, eb, xb, p)); prev = xb

def s_trail(d15, d1h, d4h, act, dist):
    tf = {1: d15, 4: d1h, 16: d4h}; out = []
    for sym in sorted(d15):
        st = []
        for _, ratio, warmup, _, mh in G_3TF:
            ds = tf.get(ratio)
            if ds is None or sym not in ds: continue
            d = ds[sym]; vr = d.get("vr_8_56")
            if vr is None: continue
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x = sim_straddle_trailing(d["close"],d["high"],d["low"],sig,
                cfg.s_sl,cfg.s_tp,mh,COST_STRADDLE,act,dist)
            for k in range(len(p)): st.append((int(e[k])*ratio,int(x[k])*ratio+(ratio-1),float(p[k])))
        dedup(sym, st, out)
    return out

def ma(d15, d1h, d4h):
    return collect_enhanced_trades(d15,d1h,d4h,
        cluster_pct=MA_CLUSTER_PCT,min_duration=MA_MIN_DURATION,max_duration=MA_MAX_DURATION,
        min_momentum=MA_MIN_MOMENTUM,sl=MA_SL,tp=MA_TP,tf_restriction=MA_TF_RESTRICTION)

def filt(trades, bar0, allowed_h=None, allowed_wd=None):
    out = []
    for sym,eb,xb,pnl in trades:
        dt = bar0 + timedelta(minutes=15*eb)
        if allowed_h is not None and dt.hour not in allowed_h: continue
        if allowed_wd is not None and dt.weekday() not in allowed_wd: continue
        out.append((sym,eb,xb,pnl))
    return out

def run(trades, data_years, label, n_syms):
    pnl = portfolio_sim(trades, MAX_CONC)
    m = compute_metrics(pnl, data_years)
    bt = bootstrap_pf(pnl, n_boot=400)
    m.update({"label": label, "n_syms": n_syms, "ci": f"[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]"})
    return m

def fmt(m):
    return (f"PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  WR={m['WR']:.1f}%  "
            f"Calmar={m['calmar']:.1f}  DD={m['max_dd_pct']:.1f}%  N={m['N']:4d}  {m['ci']}")

# ── Load 5yr data ──────────────────────────────────────────────────────────────
print("Loading 34 syms (5yr+)...")
d15, ref = load_syms(5.0)
d1h, d4h = build_tf(d15)
bar0 = get_bar0(ref)
DATA_YR = max(len(d["close"]) for d in d15.values()) / (4*24*365)
N_SYM = len(d15)
print(f"  {N_SYM} syms | {DATA_YR:.2f}yr | bar0={bar0.date()}")

# pre-collect
print("Collecting trades...")
d_tr  = collect_d_trades_trailing(CFG_BL, d15, d1h, d4h)
s_flat= collect_trades(CFG_BL, d15, d1h, d4h, "S_d8")
s_tr  = s_trail(d15, d1h, d4h, 0.015, 0.002)
ma_tr = ma(d15, d1h, d4h)
print(f"  D={len(d_tr)}  S_flat={len(s_flat)}  S_trail={len(s_tr)}  MA={len(ma_tr)}")

# time definitions
WORST_H   = {2,7,9,11,13,14,15,23}
BEST_H    = {0,1,3,8,16,17,18,21}
NY_H      = set(range(16,24))
ASIA_H    = set(range(0,8))
WEEKDAY   = {0,1,2,3,4}  # Mon-Fri
MON_FRI_H = None          # all hours, weekdays only

print("\n" + "="*72)
print("r53 Extended Combo Matrix (34 syms, 5yr+, trail act=1.5%/dist=0.2%)")
print("="*72)
print(f"{'Scenario':<42} {'PF':>6}  {'EV':>7}  {'Calmar':>7}  {'DD%':>6}  {'N':>5}  CI  flag")
print(f"{'─'*42} {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}")

results = []

scenarios = [
    # (label, use_trail, h_filter, wd_filter)
    ("Baseline (flat S)",         False, None,    None),
    ("Trail only",                True,  None,    None),
    ("Worst-8h excl",             False, None,    None),   # built below
    ("Trail + Worst-8h excl",     True,  None,    None),   # built below
    ("Trail + Best-8h only",      True,  BEST_H,  None),
    ("Trail + NY sess (16-24)",    True,  NY_H,    None),
    ("Trail + Asia sess (00-08)",  True,  ASIA_H,  None),
    ("Trail + Weekday (M-F)",     True,  None,    WEEKDAY),
    ("Trail + Worst-excl + M-F",  True,  None,    WEEKDAY),   # built below
    ("Trail + NY + M-F",          True,  NY_H,    WEEKDAY),
]

BL_PF = None
for label, use_trail, allowed_h, allowed_wd in scenarios:
    s_src = s_tr if use_trail else s_flat
    raw = d_tr + s_src + ma_tr

    # Special cases for worst-hour exclusion
    if "Worst-8h excl" in label and "Trail" not in label:
        raw = filt(d_tr + s_flat + ma_tr, bar0, allowed_h=set(range(24))-WORST_H)
    elif "Worst-8h excl" in label:
        raw = filt(d_tr + s_tr + ma_tr, bar0, allowed_h=set(range(24))-WORST_H)
    elif "Worst-excl + M-F" in label:
        raw = filt(d_tr + s_tr + ma_tr, bar0, allowed_h=set(range(24))-WORST_H, allowed_wd=WEEKDAY)
    elif allowed_h is not None or allowed_wd is not None:
        raw = filt(raw, bar0, allowed_h=allowed_h, allowed_wd=allowed_wd)

    m = run(raw, DATA_YR, label, N_SYM)
    results.append(m)
    if BL_PF is None: BL_PF = m["PF"]

    flag = ""
    if m["PF"] > BL_PF * 1.20: flag = " ★★★"
    elif m["PF"] > BL_PF * 1.10: flag = " ★★"
    elif m["PF"] > BL_PF: flag = " ★"
    print(f"  {label:<40} {fmt(m)}{flag}")

# Ranking
print("\n" + "─"*72)
print("Ranked by Calmar:")
for m in sorted(results, key=lambda x: x["calmar"], reverse=True):
    print(f"  {m['label']:<40} Calmar={m['calmar']:.1f}  PF={m['PF']:.3f}  DD={m['max_dd_pct']:.1f}%")

print("\nDone.")
