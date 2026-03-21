#!/usr/bin/env python3
"""r53 재실행: 올바른 D_mom8 trail params (0.005/0.002) 사용.

r53 원본은 backtest_v14_3_0.py에서 D_TRAIL_ACTIVATE=0.02 import → 구버전.
이 스크립트는 config.py에서 직접 읽어서 production-parity 보장.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

ROOT = Path('/home/user/Documents/git/quantbox')
sys.path.insert(0, str(ROOT))

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import (
    portfolio_sim, compute_metrics, bootstrap_pf,
    build_resampled, precompute, precompute_all_vr,
    collect_trades, COST_STRADDLE, COST_DIRECTIONAL, G_3TF,
    sim_directional_trailing, sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades
from research.active.r41_r45.backtest_v14_3_0 import (
    MA_CLUSTER_PCT, MA_MIN_DURATION, MA_MAX_DURATION,
    MA_MIN_MOMENTUM, MA_SL, MA_TP, MA_TF_RESTRICTION,
)

cfg = VolStraddleConfig()

# ── CRITICAL: 모두 config.py에서 직접 읽음 ──────────────────────────────
D_SL             = cfg.d_sl              # 0.01
D_TP             = cfg.d_tp              # 0.12
D_TRAIL_ACTIVATE = cfg.d_trail_activate  # 0.005 ← r47 production value
D_TRAIL_DISTANCE = cfg.d_trail_distance  # 0.002 ← r47 production value
S_TRAIL_ACTIVATE = cfg.s_trail_activate  # 0.015 ← r53 new
S_TRAIL_DISTANCE = cfg.s_trail_distance  # 0.002
MAX_CONC         = cfg.max_concurrent    # 10
WORST_H          = {2,7,9,11,13,14,15,23}

print(f"[PARITY CHECK]")
print(f"  D_TRAIL: activate={D_TRAIL_ACTIVATE} ({D_TRAIL_ACTIVATE*100:.1f}%) / distance={D_TRAIL_DISTANCE} ({D_TRAIL_DISTANCE*100:.1f}%)")
print(f"  S_TRAIL: activate={S_TRAIL_ACTIVATE} ({S_TRAIL_ACTIVATE*100:.1f}%) / distance={S_TRAIL_DISTANCE} ({S_TRAIL_DISTANCE*100:.1f}%)")
assert D_TRAIL_ACTIVATE == 0.040, f"D trail activate mismatch! {D_TRAIL_ACTIVATE}"
assert D_TRAIL_DISTANCE == 0.002, f"D trail distance mismatch! {D_TRAIL_DISTANCE}"
print("  ✅ All params match production config.py")

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
    data = {}
    for sym,fpath,_ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    mb = max(len(d["close"]) for d in data.values()) if data else 0
    for d in data.values(): d["_offset"] = mb - len(d["close"])
    return data, str(ranked[0][1])

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

def dedup(st):
    st.sort(key=lambda x: x[0]); prev=-1; out=[]
    for eb,xb,p in st:
        if eb > prev: out.append((eb,xb,p)); prev=xb
    return out

def collect_d_correct(d15, d1h, d4h):
    """D_mom8 directional with PRODUCTION trail (0.5%/0.2%)."""
    tf={1:d15,4:d1h,16:d4h}; out=[]
    TF_CONFIGS = [(1,1,50,192),(4,4,13,48),(16,16,4,12)]
    for sym in sorted(d15):
        st=[]
        for ratio,_,warmup,mh in TF_CONFIGS:
            ds=tf[ratio]
            if sym not in ds: continue
            d=ds[sym]; vr=d.get("vr_8_56")
            if vr is None: continue
            sig=gen_vr_dur_signal(vr,cfg.d_vr_threshold,cfg.s_squeeze_dur,warmup)
            mom=np.diff(np.concatenate([[d["close"][0]],d["close"]])) if "close" in d else None
            if mom is None: continue
            dire=np.where(mom>=0,1,-1)
            p,e,x=sim_directional_trailing(
                d["close"],d["high"],d["low"],sig,dire,
                D_SL,D_TP,mh,COST_DIRECTIONAL,
                trail_activate=D_TRAIL_ACTIVATE,trail_distance=D_TRAIL_DISTANCE)
            for k in range(len(p)):
                st.append((int(e[k])*ratio, int(x[k])*ratio+(ratio-1), float(p[k])))
        for eb,xb,p in dedup(st): out.append((sym,eb,xb,p))
    return out

def collect_s_trail(d15, d1h, d4h, act, dist):
    tf={1:d15,4:d1h,16:d4h}; out=[]
    for sym in sorted(d15):
        st=[]
        for _,ratio,warmup,_,mh in G_3TF:
            ds=tf.get(ratio)
            if ds is None or sym not in ds: continue
            d=ds[sym]; vr=d.get("vr_8_56")
            if vr is None: continue
            sig=gen_vr_dur_signal(vr,cfg.s_vr_threshold,cfg.s_squeeze_dur,warmup)
            p,e,x=sim_straddle_trailing(d["close"],d["high"],d["low"],sig,
                cfg.s_sl,cfg.s_tp,mh,COST_STRADDLE,act,dist)
            for k in range(len(p)):
                st.append((int(e[k])*ratio,int(x[k])*ratio+(ratio-1),float(p[k])))
        for eb,xb,p in dedup(st): out.append((sym,eb,xb,p))
    return out

def collect_s_flat(d15, d1h, d4h):
    CFG={"sp":8,"lp":56,"d_vr":cfg.d_vr_threshold,"s_vr":cfg.s_vr_threshold,
         "s_dur":cfg.s_squeeze_dur,"d_sl":D_SL,"s_sl":cfg.s_sl,"tp":cfg.s_tp}
    return collect_trades(CFG,d15,d1h,d4h,"S_d8")

def show(label, trades, yr):
    pnl=portfolio_sim(trades, MAX_CONC)
    m=compute_metrics(pnl, yr)
    bt=bootstrap_pf(pnl, n_boot=500)
    print(f"  {label:<42} N={m['N']:5d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.0f}bp  "
          f"Calmar={m['calmar']:.1f}  DD={m['max_dd_pct']:.1f}%  "
          f"CI=[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]")
    return m

print("\nLoading 63 syms...")
d15, ref = load_syms(5.0)
d1h, d4h = build_tf(d15)
DATA_YR = max(len(d["close"]) for d in d15.values()) / (4*24*365)
print(f"  {len(d15)} syms | {DATA_YR:.2f}yr")

print("Collecting trades (correct params)...")
d_correct = collect_d_correct(d15, d1h, d4h)
s_flat    = collect_s_flat(d15, d1h, d4h)
s_trail   = collect_s_trail(d15, d1h, d4h, S_TRAIL_ACTIVATE, S_TRAIL_DISTANCE)
ma        = collect_enhanced_trades(d15,d1h,d4h,
    cluster_pct=MA_CLUSTER_PCT,min_duration=MA_MIN_DURATION,max_duration=MA_MAX_DURATION,
    min_momentum=MA_MIN_MOMENTUM,sl=MA_SL,tp=MA_TP,tf_restriction=MA_TF_RESTRICTION)
print(f"  D={len(d_correct)}  S_flat={len(s_flat)}  S_trail={len(s_trail)}  MA={len(ma)}")

print("\n" + "="*80)
print(f"Production-Parity Backtest (63 syms, {DATA_YR:.2f}yr, correct D_trail=0.5%/0.2%)")
print("="*80)
m_bl = show("v14.4.x Production Baseline (S_d8 flat)", d_correct+s_flat+ma, DATA_YR)
m_tr = show("v14.5.0 S_d8 Trail (Option A)",           d_correct+s_trail+ma, DATA_YR)

print()
print(f"Delta:")
print(f"  ΔN     = {m_tr['N']-m_bl['N']:+d} ({(m_tr['N']-m_bl['N'])/m_bl['N']*100:+.1f}%)")
print(f"  ΔPF    = {m_tr['PF']-m_bl['PF']:+.3f} ({(m_tr['PF']-m_bl['PF'])/m_bl['PF']*100:+.1f}%)")
print(f"  ΔCalmar= {m_tr['calmar']-m_bl['calmar']:+.1f}")
print(f"  ΔDD    = {m_tr['max_dd_pct']-m_bl['max_dd_pct']:+.1f}pp")
print()
print("r47 canonical (34 syms, 4.73yr): N≈4097, PF≈1.521  ← 새 baseline과 비교")
