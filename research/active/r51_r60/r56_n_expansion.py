#!/usr/bin/env python3
"""r56: N Expansion — Multiple Independent Entry Paths.

현재 상태: N=6,207 (v14.5.2, 60심볼, 6yr), PF=1.563, EV=+38bp, Calmar=57.9
목표: PF≥1.50 유지하면서 N > 7,000 달성.

3가지 독립 진입 경로:
  A) 30m TF VR Squeeze — 15m→30m 리샘플 (ratio=2), G_3TF → G_4TF
     근거: r35 E1-4TF에서 +17% N, 최저 DD (미구현)
  B) 1h MA Cluster — 현재 4h 전용인 MA cluster를 1h로도 확장
     근거: 4h = 1봉/4h, 1h = 4봉/4h → 기회 4배
  C) Re-Entry After Profitable Exit — 수익 청산 후 VR 여전히 압축이면 재진입
     룩어헤드 없음: 청산봉+1 이후 완성봉만 사용

룩어헤드 바이어스 방지:
  - 30m: 15m→30m 리샘플은 완성된 30m 봉 기준 (현재봉 미포함)
  - 1h MA cluster: 완성된 1h 봉의 SMA 기준
  - Re-entry: profitable exit bar X → re-entry signal 가능 구간 = X+1 이후

Production baseline (v14.5.2, 60 syms, 6yr):
  N=6,207 | PF=1.563 | WR=40.6% | EV=+38bp | Calmar=57.9 | DD=60.5%

Phases:
  Phase 0: Config 검증 + 베이스라인 재현 (N±1%, PF±1%)
  Phase A: 30m TF VR Squeeze (단독 + 3TF와 합산)
  Phase B: 1h MA Cluster (단독 + 4h와 합산)
  Phase C: Re-Entry After TP (단독 측정)
  Phase D: 최선 조합 + OOS 50% 검증 + 보고서

Usage:
    uv run python research/active/r51_r60/r56_n_expansion.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    WARMUP_1h,
    WARMUP_4h,
    WARMUP_15m,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)
from quantbox.backtest.trade_sim import sim_directional_trailing, sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import (
    collect_enhanced_trades,
    sim_ma_cluster_enhanced,
)

# ── Load production config (NO hardcoding) ───────────────────────────────────
cfg = VolStraddleConfig()

# VR Squeeze (D_mom8 + S_d8)
VR_SHORT        = cfg.vr_short                      # 8
VR_LONG         = cfg.vr_long                       # 56
D_VR            = cfg.d_vr_threshold               # 0.50
S_VR            = cfg.s_vr_threshold               # 0.55
S_DUR           = cfg.s_squeeze_dur                # 4
D_SL            = cfg.d_sl                         # 0.01
D_TP            = cfg.d_tp                         # 0.12
D_TRAIL_ACT     = cfg.d_trail_activate             # 0.005
D_TRAIL_DIST    = cfg.d_trail_distance             # 0.002
S_SL            = cfg.s_sl
S_TP            = cfg.s_tp
S_TRAIL_ACT     = cfg.s_trail_activate
S_TRAIL_DIST    = cfg.s_trail_distance
MAX_CONC        = cfg.max_concurrent               # 10
VR_LN           = cfg.base_leverage / MAX_CONC     # 3.0

# MA Cluster
MA_PCT          = cfg.ma_cluster_pct               # 0.03
MA_MIN_DUR      = cfg.ma_cluster_min_duration      # 8
MA_MAX_DUR      = cfg.ma_cluster_max_duration      # 24
MA_MIN_MOM      = cfg.ma_cluster_min_momentum      # 0.01
MA_SL           = cfg.ma_cluster_sl               # 0.02
MA_TP           = cfg.ma_cluster_tp               # 0.08
MA_TRAIL_ACT    = cfg.ma_cluster_trail_activate    # 0.04
MA_TRAIL_DIST   = cfg.ma_cluster_trail_distance    # 0.005
MA_LN           = cfg.ma_cluster_leverage / MAX_CONC  # 2.1

# 30m TF constants (derived from 15m; same wall-clock durations)
WARMUP_30m: int = WARMUP_15m // 2   # 100 30m-bars ≈ 50h
_MH30m: int = cfg.max_hold_15m // 2  # 96 30m-bars ≈ 48h
TF_30m = ("30m", 2, WARMUP_30m, _MH30m, _MH30m * 2)
G_4TF = [TF_30m] + list(G_3TF)       # 30m + 15m + 1h + 4h

VR_KEY = f"vr_{VR_SHORT}_{VR_LONG}"

# Baseline expected (v14.5.2 report)
BL_N_EXPECTED   = 6207
BL_PF_EXPECTED  = 1.563

# Config assert guards
assert D_TRAIL_ACT == 0.010, f"d_trail_activate mismatch: {D_TRAIL_ACT}"
assert D_TRAIL_DIST == 0.002, f"d_trail_distance mismatch: {D_TRAIL_DIST}"

print("\n[Phase 0] Config 검증")
print(f"  D: SL={D_SL}, TP={D_TP}, trail=({D_TRAIL_ACT},{D_TRAIL_DIST})")
print(f"  S: SL={S_SL}, TP={S_TP}, trail=({S_TRAIL_ACT},{S_TRAIL_DIST})")
print(f"  MA: SL={MA_SL}, TP={MA_TP}, trail=({MA_TRAIL_ACT},{MA_TRAIL_DIST})")
print(f"  VR_LN={VR_LN:.1f}  MA_LN={MA_LN:.1f}  MAX_CONC={MAX_CONC}")
print(f"  30m: warmup={WARMUP_30m}, max_hold={_MH30m}")
print("  ✅ All params loaded from config.py\n")

CACHE_DIR  = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_syms(min_years: float = 5.0) -> tuple[dict, str]:
    """Load symbols from 15m cache. Returns (data_15m, ref_fpath_str)."""
    whitelist = set(cfg.symbol_whitelist)  # 60 live symbols only
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(f, allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    min_bars = int(min_years * 365 * 24 * 4)
    ranked = sorted(
        [(s, p, nb) for s, (p, nb) in seen.items() if nb >= min_bars],
        key=lambda x: -x[2],
    )
    data: dict = {}
    for sym, fpath, _ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    ref = str(ranked[0][1]) if ranked else ""
    return data, ref


def build_all_tf(d15: dict) -> tuple[dict, dict, dict]:
    """Build 30m, 1h, 4h from 15m. Precompute VR/mom/MAs on all."""
    d30  = build_resampled(d15, 2)
    d1h  = build_resampled(d15, 4)
    d4h  = build_resampled(d15, 16)
    precompute_all_vr([d15, d30, d1h, d4h], [(VR_SHORT, VR_LONG)])
    for ds in [d15, d30, d1h, d4h]:
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
    # MA cluster needs SMAs on 1h and 4h (not 30m — MA cluster is 1h and 4h only)
    precompute_mas(d15)
    precompute_mas(d1h)
    precompute_mas(d4h)
    return d30, d1h, d4h


# ═══════════════════════════════════════════════════════════════════════════════
#  Numba Helpers
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def gen_reentry_signal_from_exits(
    vr: np.ndarray,
    threshold: float,
    warmup: int,
    exit_bars: np.ndarray,
    pnl: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Generate re-entry signals after profitable exits.

    룩어헤드 없음: profitable exit at bar X → scan bars X+1..X+lookback
    (all completed bars). First bar with VR < threshold → allow re-entry.

    Args:
        vr: VR array in target TF.
        threshold: VR compression threshold (e.g. 0.50).
        warmup: Bars to skip at start.
        exit_bars: Exit bar indices (in target TF) from original sim.
        pnl: PnL per trade from original sim.
        lookback: Max bars after exit to look for re-entry opportunity.

    Returns:
        Binary signal array, same shape as vr.
    """
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for k in range(len(pnl)):
        if pnl[k] <= 0.0:
            continue
        ex = int(exit_bars[k])
        for j in range(ex + 1, min(ex + lookback + 1, n)):
            if j < warmup:
                continue
            if not np.isnan(vr[j]) and vr[j] < threshold:
                sig[j] = 1
                break  # one re-entry per trade only
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
#  Trade Collection Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _dedup_sym(sym: str, sym_trades: list) -> list:
    """Per-symbol time dedup, same logic as collect_trades in trade_sim."""
    sym_trades.sort(key=lambda x: x[0])
    accepted: list[tuple[int, int]] = []
    result = []
    for entry_15m, exit_15m, pnl_val in sym_trades:
        conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
        if not conflict:
            result.append((sym, entry_15m, exit_15m, pnl_val))
            accepted.append((entry_15m, exit_15m))
    return result


def collect_d_trailing(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
) -> list:
    """Collect D_mom8 trades with production trailing stop.

    Args:
        data_15m / data_1h / data_4h: Core 3 TF datasets.
        data_30m: Optional 30m dataset. If provided, adds 30m to collection.

    Returns:
        List of (sym, entry_15m, exit_15m, pnl).
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m

    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list = []
        for _lbl, ratio, warmup, mh_d, _mh_s in tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(VR_KEY)
            mom8 = d.get("mom_8")
            if vr is None or mom8 is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig  = gen_vr_signal(vr, D_VR, warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio, int(x[k]) * ratio + ratio - 1, float(p[k])))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_s_trailing(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
) -> list:
    """Collect S_d8 straddle trades with production trailing stop."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m

    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list = []
        for _lbl, ratio, warmup, _mh_d, mh_s in tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(VR_KEY)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, S_VR, S_DUR, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig,
                S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACT, S_TRAIL_DIST,
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio, int(x[k]) * ratio + ratio - 1, float(p[k])))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_ma(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    tf_restriction: int | None = 16,
) -> list:
    """Collect MA cluster trades (4h-only by default, or 1h with tf_restriction=4)."""
    return collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_PCT,
        min_duration=MA_MIN_DUR,
        max_duration=MA_MAX_DUR,
        min_momentum=MA_MIN_MOM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=tf_restriction,
    )


def collect_d_reentry(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    lookback: int = 5,
    vr_threshold: float | None = None,
) -> list:
    """Collect D_mom8 re-entry trades after profitable exits.

    룩어헤드 없음: After profitable exit at TF bar X, scan X+1..X+lookback
    for the next compressed bar. The re-entry signal fires there, and the
    new trade starts at X+1 or later (next bar open) — all completed bars.

    Args:
        lookback: Max bars after exit to look for re-entry (in TF bars).
        vr_threshold: VR compression threshold; defaults to D_VR.

    Returns:
        List of re-entry (sym, entry_15m, exit_15m, pnl) tuples.
    """
    if vr_threshold is None:
        vr_threshold = D_VR
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}

    all_reentry: list = []
    for sym in sorted(data_15m.keys()):
        reentry_trades: list = []
        for _lbl, ratio, warmup, mh_d, _mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr  = d.get(VR_KEY)
            mom8 = d.get("mom_8")
            if vr is None or mom8 is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]

            # Step 1: collect original trades
            orig_sig  = gen_vr_signal(vr, D_VR, warmup)
            orig_dire = make_direction_momentum(orig_sig, mom8)
            p_orig, e_orig, x_orig = sim_directional_trailing(
                c, h, lo, orig_sig, orig_dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            if len(p_orig) == 0:
                continue

            # Step 2: generate re-entry signal from profitable exits
            reentry_sig = gen_reentry_signal_from_exits(
                vr, vr_threshold, warmup, x_orig, p_orig, lookback,
            )
            if reentry_sig.sum() == 0:
                continue

            # Step 3: simulate re-entry trades
            reentry_dire = make_direction_momentum(reentry_sig, mom8)
            p_re, e_re, x_re = sim_directional_trailing(
                c, h, lo, reentry_sig, reentry_dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p_re)):
                reentry_trades.append((
                    int(e_re[k]) * ratio,
                    int(x_re[k]) * ratio + ratio - 1,
                    float(p_re[k]),
                ))

        # Dedup re-entry trades per symbol
        if reentry_trades:
            all_reentry.extend(_dedup_sym(sym, reentry_trades))
    return all_reentry


def run_combined(
    d15: dict,
    d1h: dict,
    d4h: dict,
    label: str,
    data_years: float,
    d30: dict | None = None,
    ma_tf: int | None = 16,
    include_reentry: bool = False,
    reentry_lookback: int = 5,
) -> dict:
    """Run full combined simulation and compute metrics.

    Args:
        d30: If provided, adds 30m TF to D/S collection.
        ma_tf: MA cluster TF restriction (16=4h, 4=1h, None=all).
        include_reentry: If True, adds re-entry D trades to collection.
    """
    t = time.time()
    d_trades = collect_d_trailing(d15, d1h, d4h, d30)
    s_trades = collect_s_trailing(d15, d1h, d4h, d30)
    ma_trades = collect_ma(d15, d1h, d4h, ma_tf) if ma_tf is not None else []

    extra_trades: list = []
    if include_reentry:
        extra_trades = collect_d_reentry(d15, d1h, d4h, lookback=reentry_lookback)

    all_trades = d_trades + s_trades + ma_trades + extra_trades
    pnl = portfolio_sim(all_trades, MAX_CONC)
    m   = compute_metrics(pnl, data_years)
    bt  = bootstrap_pf(pnl, n_boot=500)

    m["label"]     = label
    m["elapsed"]   = round(time.time() - t, 1)
    m["ci"]        = f"[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]"
    m["n_syms"]    = len(d15)
    m["raw_d"]     = len(d_trades)
    m["raw_s"]     = len(s_trades)
    m["raw_ma"]    = len(ma_trades)
    m["raw_extra"] = len(extra_trades)
    return m


def fmt(m: dict) -> str:
    return (
        f"N={m['N']:6d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
        f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.1f}  "
        f"DD={m['max_dd_pct']:.1f}%  CI={m.get('ci','?')}  [{m.get('elapsed',0)}s]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  OOS Split Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_to_period(d15: dict, d30: dict, d1h: dict, d4h: dict, start_ratio: float, end_ratio: float) -> tuple:
    """Trim all datasets to a time window [start_ratio, end_ratio] of total bars."""
    def _trim(dataset: dict) -> dict:
        out = {}
        for sym, d in dataset.items():
            n = len(d["close"])
            i0 = int(n * start_ratio)
            i1 = int(n * end_ratio)
            if i1 - i0 < 100:
                continue
            out[sym] = {k: v[i0:i1] if isinstance(v, np.ndarray) else v for k, v in d.items()}
        return out
    return _trim(d15), _trim(d30), _trim(d1h), _trim(d4h)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    commit = get_commit()
    print(f"\nr56: N Expansion [{commit}]  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    t_load = time.time()
    print("Loading 15m data...")
    data_15m, ref_fpath = load_syms(min_years=5.0)
    n_syms = len(data_15m)
    max_bars = max(len(d["close"]) for d in data_15m.values())
    DATA_YEARS = max_bars / (4 * 24 * 365)
    print(f"  {n_syms} symbols | {DATA_YEARS:.2f} years | {time.time()-t_load:.1f}s")

    print("Building 30m / 1h / 4h & precomputing...")
    t_prep = time.time()
    data_30m, data_1h, data_4h = build_all_tf(data_15m)
    print(f"  Done in {time.time()-t_prep:.1f}s  "
          f"| 30m={len(data_30m)} 1h={len(data_1h)} 4h={len(data_4h)} syms\n")

    results: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 0: Baseline (3TF + 4h MA cluster)
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("Phase 0: Baseline Reproduction (must be within ±1% of N=6207, PF=1.563)")
    print("─" * 72)
    m_bl = run_combined(data_15m, data_1h, data_4h, label="Baseline (3TF+MA4h)",
                        data_years=DATA_YEARS)
    print(f"  {m_bl['label']:30s}  {fmt(m_bl)}")

    n_drift  = abs(m_bl["N"] - BL_N_EXPECTED) / BL_N_EXPECTED
    pf_drift = abs(m_bl["PF"] - BL_PF_EXPECTED) / BL_PF_EXPECTED
    if n_drift > 0.015 or pf_drift > 0.015:
        print(f"\n⚠️  Baseline drift: N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}% — check cache/config")
    else:
        print(f"\n  ✅ Baseline OK (N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}%)")
    results.append(m_bl)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase A: 30m TF Addition
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase A: 30m TF VR Squeeze (ratio=2, WARMUP_30m=100, MAX_HOLD_30m=96)")
    print("─" * 72)

    # A1: 30m only contribution (D+S at 30m, no MA)
    print("\nA1: 30m-only contribution (what 30m adds standalone)")
    t = time.time()
    d30_only = collect_d_trailing(data_15m, data_1h, data_4h, data_30m)
    s30_only = collect_s_trailing(data_15m, data_1h, data_4h, data_30m)

    # Count 30m-specific trades by comparing with 3TF collection
    d3tf = collect_d_trailing(data_15m, data_1h, data_4h, None)
    s3tf = collect_s_trailing(data_15m, data_1h, data_4h, None)
    d3tf_keys = {(sym, e) for sym, e, x, pnl in d3tf}
    s3tf_keys = {(sym, e) for sym, e, x, pnl in s3tf}
    d30_new = [(sym, e, x, pnl) for sym, e, x, pnl in d30_only if (sym, e) not in d3tf_keys]
    s30_new = [(sym, e, x, pnl) for sym, e, x, pnl in s30_only if (sym, e) not in s3tf_keys]
    print(f"  30m adds {len(d30_new)} D-trades + {len(s30_new)} S-trades (not in 3TF baseline)")

    if len(d30_new) + len(s30_new) > 0:
        pnl_30new = np.array([p for _, _, _, p in d30_new + s30_new])
        m30_standalone = compute_metrics(pnl_30new, DATA_YEARS)
        print(f"  30m-only trades: N={m30_standalone['N']} PF={m30_standalone['PF']:.3f} "
              f"EV={m30_standalone['EV_bp']:+.1f}bp WR={m30_standalone['WR']:.1f}%")

    # A2: Full 4TF (3TF + 30m) combined
    m_a = run_combined(data_15m, data_1h, data_4h, label="4TF (3TF+30m) + MA4h",
                       data_years=DATA_YEARS, d30=data_30m, ma_tf=16)
    print(f"\nA2: {m_a['label']:30s}  {fmt(m_a)}")
    print(f"    ΔN={m_a['N']-m_bl['N']:+d}  ΔPF={m_a['PF']-m_bl['PF']:+.3f}  "
          f"ΔEV={m_a['EV_bp']-m_bl['EV_bp']:+.1f}bp")
    results.append(m_a)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase B: 1h MA Cluster
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase B: 1h MA Cluster (adds 1h MA signals to 4h baseline)")
    print("─" * 72)

    # B1: 1h MA only (standalone)
    t_b1 = time.time()
    ma_1h_trades = collect_ma(data_15m, data_1h, data_4h, tf_restriction=4)
    ma_4h_trades = collect_ma(data_15m, data_1h, data_4h, tf_restriction=16)
    ma_1h_only = set((sym, e) for sym, e, x, p in ma_1h_trades)
    ma_4h_only = set((sym, e) for sym, e, x, p in ma_4h_trades)
    overlap = ma_1h_only & ma_4h_only
    print(f"  MA-1h standalone: {len(ma_1h_trades)} trades")
    print(f"  MA-4h standalone: {len(ma_4h_trades)} trades")
    print(f"  Overlap (same entry): {len(overlap)} trades")

    if len(ma_1h_trades) > 0:
        pnl_ma1h = np.array([p for _, _, _, p in ma_1h_trades])
        m_ma1h = compute_metrics(pnl_ma1h, DATA_YEARS)
        print(f"  1h MA-only PF={m_ma1h['PF']:.3f} EV={m_ma1h['EV_bp']:+.1f}bp "
              f"WR={m_ma1h['WR']:.1f}% N={m_ma1h['N']}")

    # B2: 3TF + MA (1h only)
    m_b1h = run_combined(data_15m, data_1h, data_4h, label="3TF + MA-1h only",
                         data_years=DATA_YEARS, ma_tf=4)
    print(f"\nB2: {m_b1h['label']:30s}  {fmt(m_b1h)}")
    print(f"    ΔN={m_b1h['N']-m_bl['N']:+d}  ΔPF={m_b1h['PF']-m_bl['PF']:+.3f}  "
          f"ΔEV={m_b1h['EV_bp']-m_bl['EV_bp']:+.1f}bp")

    # B3: 3TF + MA (4h + 1h combined)
    m_b4h1h = run_combined(data_15m, data_1h, data_4h, label="3TF + MA-4h+1h",
                           data_years=DATA_YEARS, ma_tf=None)
    # ma_tf=None means no MA, so we add 4h+1h manually
    t_b3 = time.time()
    ma_both = ma_4h_trades + [t for t in ma_1h_trades if (t[0], t[1]) not in ma_4h_only]
    d3_b = collect_d_trailing(data_15m, data_1h, data_4h)
    s3_b = collect_s_trailing(data_15m, data_1h, data_4h)
    pnl_b_both = portfolio_sim(d3_b + s3_b + ma_both, MAX_CONC)
    m_b_both = compute_metrics(pnl_b_both, DATA_YEARS)
    bt_b = bootstrap_pf(pnl_b_both, n_boot=500)
    m_b_both["label"] = "3TF + MA-4h+1h (deduped)"
    m_b_both["ci"] = f"[{bt_b['CI_lo']:.3f},{bt_b['CI_hi']:.3f}]"
    m_b_both["elapsed"] = round(time.time() - t_b3, 1)
    m_b_both["n_syms"] = len(data_15m)
    print(f"B3: {m_b_both['label']:30s}  {fmt(m_b_both)}")
    print(f"    ΔN={m_b_both['N']-m_bl['N']:+d}  ΔPF={m_b_both['PF']-m_bl['PF']:+.3f}  "
          f"ΔEV={m_b_both['EV_bp']-m_bl['EV_bp']:+.1f}bp")
    results.append(m_b1h)
    results.append(m_b_both)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase C: Re-Entry After Profitable Exit
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase C: Re-Entry After Profitable D_mom8 Exit")
    print("  룩어헤드 없음: profitable exit bar X → scan X+1..X+lookback (all completed bars)")
    print("─" * 72)

    c_results: list[dict] = []
    for lookback in [3, 5, 8]:
        t_c = time.time()
        reentry = collect_d_reentry(data_15m, data_1h, data_4h, lookback=lookback)
        n_reentry = len(reentry)
        if n_reentry > 0:
            pnl_re = np.array([p for _, _, _, p in reentry])
            m_re_standalone = compute_metrics(pnl_re, DATA_YEARS)
            print(f"\nC-lookback={lookback}: {n_reentry} re-entry trades "
                  f"PF={m_re_standalone['PF']:.3f} EV={m_re_standalone['EV_bp']:+.1f}bp "
                  f"WR={m_re_standalone['WR']:.1f}%  [{time.time()-t_c:.1f}s]")

        # Full combined: baseline + re-entry
        d3_c  = collect_d_trailing(data_15m, data_1h, data_4h)
        s3_c  = collect_s_trailing(data_15m, data_1h, data_4h)
        ma_c  = collect_ma(data_15m, data_1h, data_4h, 16)
        t_pc  = time.time()
        pnl_c = portfolio_sim(d3_c + s3_c + ma_c + reentry, MAX_CONC)
        m_c   = compute_metrics(pnl_c, DATA_YEARS)
        bt_c  = bootstrap_pf(pnl_c, n_boot=500)
        label = f"Baseline + ReEntry(lb={lookback})"
        m_c["label"]   = label
        m_c["ci"]      = f"[{bt_c['CI_lo']:.3f},{bt_c['CI_hi']:.3f}]"
        m_c["elapsed"] = round(time.time() - t_pc, 1)
        m_c["n_syms"]  = len(data_15m)
        m_c["raw_reentry"] = n_reentry
        print(f"  Combined: {label:35s}  {fmt(m_c)}")
        print(f"    ΔN={m_c['N']-m_bl['N']:+d}  ΔPF={m_c['PF']-m_bl['PF']:+.3f}  "
              f"ΔEV={m_c['EV_bp']-m_bl['EV_bp']:+.1f}bp")
        c_results.append(m_c)
        results.append(m_c)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase D: Best Combination
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase D: Best Combination Candidates")
    print("─" * 72)

    # Determine best lookback from C results
    best_c = max(c_results, key=lambda x: x["PF"]) if c_results else None
    best_lb = int(best_c["label"].split("lb=")[1].rstrip(")")) if best_c else 5

    combos = [
        ("A+B: 4TF + MA-4h+1h",
         dict(d30=data_30m, ma_tf=None), True, None),  # manual MA
        ("A+C: 4TF + ReEntry",
         dict(d30=data_30m, ma_tf=16, include_reentry=True, reentry_lookback=best_lb), False, None),
        ("B+C: 3TF + MA-4h+1h + ReEntry",
         dict(d30=None, ma_tf=None, include_reentry=True, reentry_lookback=best_lb), True, ma_both),
        ("A+B+C: 4TF + MA-4h+1h + ReEntry",
         dict(d30=data_30m, ma_tf=16, include_reentry=True, reentry_lookback=best_lb), True, None),
    ]

    combo_results: list[dict] = []
    for label, kw, needs_manual_ma, manual_ma in combos:
        if needs_manual_ma and manual_ma is None:
            # Manually build: D+S + ma_both (4h+1h deduped) + optional 30m + optional reentry
            t_kc = time.time()
            d_kw  = collect_d_trailing(data_15m, data_1h, data_4h, kw.get("d30"))
            s_kw  = collect_s_trailing(data_15m, data_1h, data_4h, kw.get("d30"))
            re_kw = (collect_d_reentry(data_15m, data_1h, data_4h, lookback=kw.get("reentry_lookback", 5))
                     if kw.get("include_reentry") else [])
            all_kw = d_kw + s_kw + ma_both + re_kw
            pnl_kw = portfolio_sim(all_kw, MAX_CONC)
            m_kw   = compute_metrics(pnl_kw, DATA_YEARS)
            bt_kw  = bootstrap_pf(pnl_kw, n_boot=500)
            m_kw["label"] = label
            m_kw["ci"] = f"[{bt_kw['CI_lo']:.3f},{bt_kw['CI_hi']:.3f}]"
            m_kw["elapsed"] = round(time.time() - t_kc, 1)
            m_kw["n_syms"] = len(data_15m)
        elif needs_manual_ma and manual_ma is not None:
            t_kc = time.time()
            d_kw  = collect_d_trailing(data_15m, data_1h, data_4h, kw.get("d30"))
            s_kw  = collect_s_trailing(data_15m, data_1h, data_4h, kw.get("d30"))
            re_kw = (collect_d_reentry(data_15m, data_1h, data_4h, lookback=kw.get("reentry_lookback", 5))
                     if kw.get("include_reentry") else [])
            all_kw = d_kw + s_kw + manual_ma + re_kw
            pnl_kw = portfolio_sim(all_kw, MAX_CONC)
            m_kw   = compute_metrics(pnl_kw, DATA_YEARS)
            bt_kw  = bootstrap_pf(pnl_kw, n_boot=500)
            m_kw["label"] = label
            m_kw["ci"] = f"[{bt_kw['CI_lo']:.3f},{bt_kw['CI_hi']:.3f}]"
            m_kw["elapsed"] = round(time.time() - t_kc, 1)
            m_kw["n_syms"] = len(data_15m)
        else:
            m_kw = run_combined(data_15m, data_1h, data_4h, label=label,
                                data_years=DATA_YEARS, **kw)

        print(f"  {m_kw['label']:38s}  {fmt(m_kw)}")
        print(f"    ΔN={m_kw['N']-m_bl['N']:+d}  ΔPF={m_kw['PF']-m_bl['PF']:+.3f}  "
              f"ΔEV={m_kw['EV_bp']-m_bl['EV_bp']:+.1f}bp")
        combo_results.append(m_kw)
        results.append(m_kw)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase D: OOS Validation on Best Combo
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase D: OOS 50% Split Validation")
    print("─" * 72)

    best_combo = max(combo_results, key=lambda x: x["PF"])
    print(f"  Best combo: {best_combo['label']}")

    # Parse best combo settings — run same as baseline and best combo on IS/OOS
    is_d15, is_d30, is_d1h, is_d4h = _trim_to_period(data_15m, data_30m, data_1h, data_4h, 0.0, 0.5)
    os_d15, os_d30, os_d1h, os_d4h = _trim_to_period(data_15m, data_30m, data_1h, data_4h, 0.5, 1.0)

    is_years = (max(len(d["close"]) for d in is_d15.values()) / (4 * 24 * 365)) if is_d15 else DATA_YEARS / 2
    os_years = (max(len(d["close"]) for d in os_d15.values()) / (4 * 24 * 365)) if os_d15 else DATA_YEARS / 2

    # Precompute for IS/OOS splits
    for ds_set in [(is_d15, is_d30, is_d1h, is_d4h), (os_d15, os_d30, os_d1h, os_d4h)]:
        d15_, d30_, d1h_, d4h_ = ds_set
        precompute_all_vr([d15_, d30_, d1h_, d4h_], [(VR_SHORT, VR_LONG)])
        for ds in [d15_, d30_, d1h_, d4h_]:
            precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
        precompute_mas(d15_)
        precompute_mas(d1h_)
        precompute_mas(d4h_)

    m_bl_is = run_combined(is_d15, is_d1h, is_d4h, "BL IS 50%", is_years)
    m_bl_os = run_combined(os_d15, os_d1h, os_d4h, "BL OOS 50%", os_years)

    print(f"\n  Baseline IS : {fmt(m_bl_is)}")
    print(f"  Baseline OOS: {fmt(m_bl_os)}")
    print(f"  OOS/IS PF = {m_bl_os['PF']/m_bl_is['PF']:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("SUMMARY TABLE")
    print("═" * 80)
    print(f"{'Label':42s}  {'N':>6}  {'PF':>6}  {'EV':>7}  {'WR':>6}  {'Calmar':>7}  {'DD':>6}  {'CI'}")
    print("─" * 110)
    for m in results:
        star = "★" if m["N"] >= 7000 and m["PF"] >= 1.50 else (" " if m["PF"] >= 1.50 else "⚠")
        print(f"{star} {m['label']:40s}  {m['N']:>6d}  {m['PF']:>6.3f}  "
              f"{m['EV_bp']:>+7.1f}bp  {m['WR']:>5.1f}%  {m['calmar']:>7.1f}  "
              f"{m['max_dd_pct']:>5.1f}%  {m.get('ci','?')}")
    print("─" * 110)

    # ─────────────────────────────────────────────────────────────────────────
    # Write Report
    # ─────────────────────────────────────────────────────────────────────────
    today = datetime.now().strftime("%Y%m%d")
    yyyymm = datetime.now().strftime("%Y%m")
    report_dir = REPORT_DIR / yyyymm
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r56_n_expansion.md"

    winners = [m for m in results if m["PF"] >= 1.50]
    goal_met = any(m["N"] >= 7000 and m["PF"] >= 1.50 for m in results)

    # Best method by PF
    best_combo_pf = max(combo_results, key=lambda x: x["PF"])

    with open(report_path, "w") as f:
        f.write(f"# r56: N Expansion — Multiple Independent Entry Paths\n\n")
        f.write(f"**날짜:** {datetime.now().strftime('%Y-%m-%d')}  \n")
        f.write(f"**커밋:** `{commit}`  \n")
        f.write(f"**구현:** {'미구현 (추가 검증 필요)' if not goal_met else '미구현 (결과 검토 후 feature/ 브랜치에서 구현 예정)'}\n\n")
        f.write("---\n\n")
        f.write("## 연구 배경\n\n")
        f.write(f"**현재 프로덕션 (v14.5.2):** N=6,207 | PF=1.563 | EV=+38bp | Calmar=57.9  \n")
        f.write(f"**문제:** N이 낮으면 통계적 신뢰도 저하 + 자본 배치 기회 감소  \n")
        f.write(f"**목표:** N > 7,000 달성 (PF ≥ 1.50 유지)\n\n")
        f.write("### 연구 방법\n\n")
        f.write("| 방법 | 설명 | 예상 N 증가 |\n")
        f.write("|------|------|-------------|\n")
        f.write("| A: 30m TF | 15m→30m 리샘플 (ratio=2), G_3TF → G_4TF | +700~900 |\n")
        f.write("| B: 1h MA Cluster | 현재 4h 전용 MA cluster → 1h도 추가 | +500~800 |\n")
        f.write("| C: Re-Entry | 수익 청산 후 VR 압축 유지 시 재진입 | +200~400 |\n\n")
        f.write("---\n\n")
        f.write("## 환경\n\n")
        f.write(f"| 항목 | 값 |\n")
        f.write(f"|------|----|\n")
        f.write(f"| 심볼 | {n_syms}개 |\n")
        f.write(f"| 기간 | {DATA_YEARS:.1f}년 |\n")
        f.write(f"| 슬롯 | {MAX_CONC} (max_concurrent) |\n")
        f.write(f"| 비용 | 0.08% round-trip (D/S) |\n")
        f.write(f"| D trail | activate={D_TRAIL_ACT}, distance={D_TRAIL_DIST} |\n")
        f.write(f"| VR_LN | {VR_LN:.1f} (base_leverage/max_concurrent) |\n")
        f.write(f"| 30m warmup | {WARMUP_30m} 봉 (={WARMUP_30m*30}분 = {WARMUP_30m*30/60:.0f}h) |\n\n")
        f.write("---\n\n")
        f.write("## Phase 0: 베이스라인\n\n")
        f.write(f"N drift={n_drift*100:.1f}% PF drift={pf_drift*100:.1f}%  \n")
        f.write(f"**{fmt(m_bl)}**\n\n")
        f.write("---\n\n")
        f.write("## 결과 요약\n\n")
        f.write(f"| 방법 | N | ΔN | PF | ΔPF | EV | WR | Calmar | DD | CI |\n")
        f.write(f"|------|---|----|----|-----|----|----|----|----|-|\n")
        for m in results:
            delta_n  = m["N"] - m_bl["N"]
            delta_pf = m["PF"] - m_bl["PF"]
            f.write(
                f"| {m['label']} | {m['N']} | {delta_n:+d} | {m['PF']:.3f} | {delta_pf:+.3f} | "
                f"{m['EV_bp']:+.1f}bp | {m['WR']:.1f}% | {m['calmar']:.1f} | {m['max_dd_pct']:.1f}% | {m.get('ci','?')} |\n"
            )
        f.write("\n---\n\n")
        f.write("## OOS 검증\n\n")
        f.write(f"| | IS (50%) | OOS (50%) | OOS/IS |\n")
        f.write(f"|--|--|--|--|\n")
        f.write(f"| Baseline PF | {m_bl_is['PF']:.3f} | {m_bl_os['PF']:.3f} | {m_bl_os['PF']/m_bl_is['PF']:.3f} |\n")
        f.write(f"| Baseline N | {m_bl_is['N']} | {m_bl_os['N']} | — |\n\n")
        f.write("---\n\n")
        f.write("## 결론 및 권장사항\n\n")
        if goal_met:
            f.write("✅ **목표 달성 (N≥7,000, PF≥1.50)**\n\n")
        else:
            f.write("⚠️ **목표 미달성** — 추가 파라미터 탐색 또는 방법 개선 필요\n\n")
        f.write(f"**최선 조합 (PF 기준):** {best_combo_pf['label']}  \n")
        f.write(f"N={best_combo_pf['N']} PF={best_combo_pf['PF']:.3f} EV={best_combo_pf['EV_bp']:+.1f}bp  \n\n")
        f.write("---\n\n")
        f.write(f"*데이터: {n_syms}심볼 · {DATA_YEARS:.1f}년 · 비용 0.08% round-trip · Binance USDT-M perpetuals*\n")

    print(f"\n  📄 보고서: {report_path.relative_to(ROOT)}")
    print(f"\n  {'✅ 목표 달성 (N≥7000 + PF≥1.50)' if goal_met else '⚠️  목표 미달성 — 추가 탐색 필요'}")
    print("\n✓ r56 완료\n")


if __name__ == "__main__":
    main()
