#!/usr/bin/env python3
"""R37: ATR Period & Stacking Grid Search — Maximize EV×N
==========================================================
Based on R36 findings:
  - VR absolute value = edge. ATR period shortening is the only
    valid frequency boost lever.
  - V3-VR5_40 was the sweet spot: N+85%, EV-5.5bp, EV×N +59%

User directive: DD is acceptable if total profit (EV×N) is higher.
  "MDD일때도 안전한 전략보다 이미 수익이 훨씬 큰상황이라면, 그걸 선택"

R37 explores:
  A: Fine-grained ATR period grid (4/30 through 6/45)
  B: Threshold relaxation on VR(5/40)
  C: Duration reduction on VR(5/40)
  D: Combined (ATR + threshold + duration)
  E: Aggressive max-profit combos

Acceptance criteria (no DD filter):
  EV > 0 AND PF > 1.0 AND t > 2.0 AND EV×N > baseline

Cost assumptions:
  - Directional: 2 × 0.04% = 0.08%
  - Straddle:    4 × 0.04% = 0.16%

Usage:
    PYTHONUNBUFFERED=1 .venv/bin/python research/active/frequency_boost_r37.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.active.mtf_vol_straddle_r31 import (
    WARMUP_15m,
    WARMUP_1h,
    WARMUP_4h,
    build_resampled,
    compute_momentum,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    load_15m,
    make_direction_momentum,
    pf_from_pnl,
    bootstrap_pf,
    precompute,
    vol_percentile,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

NUM_WORKERS = 4
COST_DIRECTIONAL = 0.0008   # 2 × 0.04%
COST_STRADDLE = 0.0016      # 4 × 0.04%

# TF configs: (label, ratio, warmup, max_hold_d, max_hold_s)
TF_15m = ("15m", 1, WARMUP_15m, 192, 384)
TF_1h  = ("1h",  4, WARMUP_1h,  48,  96)
TF_4h  = ("4h", 16, WARMUP_4h,  12,  24)
G_3TF  = [TF_15m, TF_1h, TF_4h]

# Unique ATR pairs to precompute (covers all variants)
ATR_PAIRS = [(4, 30), (4, 35), (4, 40), (5, 35), (5, 40),
             (6, 35), (6, 40), (6, 45), (7, 50)]


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Simulations (from R36, unchanged)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        l_sl = entry * (1.0 - sl)
        l_tp = entry * (1.0 + tp)
        s_sl = entry * (1.0 + sl)
        s_tp = entry * (1.0 - tp)
        la = True
        sa = True
        le = 0.0
        se = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= l_sl:
                    le = -sl
                    la = False
                elif high[j] >= l_tp:
                    le = tp
                    la = False
            if sa:
                if high[j] >= s_sl:
                    se = -sl
                    sa = False
                elif low[j] <= s_tp:
                    se = tp
                    sa = False
            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue
        d = dire[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
        else:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry
        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
    n = len(pnl)
    if n < 2:
        return {"N": n, "PF": 0.0, "EV_bp": 0.0, "t": 0.0, "WR": 0.0,
                "RR": 0.0, "ret_pct": 0.0, "max_dd_pct": 0.0, "tpd": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n * 100
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = abs(float(np.mean(losses))) if len(losses) > 0 else 1e-9
    rr = avg_win / avg_loss if avg_loss > 0 else 999.0
    pf = pf_from_pnl(pnl)
    ev = float(np.mean(pnl)) * 10000
    se = float(np.std(pnl) / np.sqrt(n)) * 10000
    t = ev / se if se > 0 else 0.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnl:
        equity *= (1.0 + p)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    total_ret = (equity - 1.0) * 100
    tpd = n / (data_years * 365) if data_years > 0 else 0.0

    return {
        "N": n, "PF": round(pf, 3), "EV_bp": round(ev, 1),
        "t": round(t, 2), "WR": round(wr, 1), "RR": round(rr, 2),
        "ret_pct": round(total_ret, 1), "max_dd_pct": round(max_dd * 100, 1),
        "tpd": round(tpd, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Global data (for ProcessPoolExecutor workers)
# ══════════════════════════════════════════════════════════════════════════════

_G: dict = {}


def _init_worker(g: dict) -> None:
    global _G
    _G = g


# ══════════════════════════════════════════════════════════════════════════════
#  Variant Definitions
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS: OrderedDict[str, dict] = OrderedDict()

# ── T0: Current deployed config (baseline) ──
VARIANTS["T0-Baseline"] = {
    "desc": "Deployed: VR(7/50) < 0.50",
    "sp": 7, "lp": 50, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
}

# ── R36 Winner (reference point) ──
VARIANTS["R36-VR540"] = {
    "desc": "R36 winner: VR(5/40) < 0.50",
    "sp": 5, "lp": 40, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
}

# ── A: ATR Period Fine Grid (fill gaps around 5/40) ──
for sp, lp in [(4, 30), (4, 35), (4, 40), (5, 35), (6, 35), (6, 40), (6, 45)]:
    VARIANTS[f"A-VR{sp}_{lp}"] = {
        "desc": f"ATR grid: VR({sp}/{lp}) < 0.50",
        "sp": sp, "lp": lp, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── B: Threshold Relaxation on VR(5/40) ──
for d_vr in [0.52, 0.55, 0.60]:
    s_vr = d_vr + 0.05
    VARIANTS[f"B-T{int(d_vr * 100)}"] = {
        "desc": f"VR(5/40) + threshold d<{d_vr} s<{s_vr}",
        "sp": 5, "lp": 40, "d_vr": d_vr, "s_vr": s_vr, "s_dur": 8,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── C: Duration Reduction on VR(5/40) ──
for dur in [6, 4, 1]:
    VARIANTS[f"C-D{dur}"] = {
        "desc": f"VR(5/40) + squeeze_dur={dur}",
        "sp": 5, "lp": 40, "d_vr": 0.50, "s_vr": 0.55, "s_dur": dur,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── D: Combined (best ATR + threshold + duration) ──
_D_COMBOS = [
    ("D1-T55D6", 5, 40, 0.55, 0.60, 6),
    ("D2-T55D4", 5, 40, 0.55, 0.60, 4),
    ("D3-T60D6", 5, 40, 0.60, 0.65, 6),
    ("D4-VR435T55D6", 4, 35, 0.55, 0.60, 6),
]
for name, sp, lp, d_vr, s_vr, dur in _D_COMBOS:
    VARIANTS[name] = {
        "desc": f"VR({sp}/{lp}) d<{d_vr} s<{s_vr} dur={dur}",
        "sp": sp, "lp": lp, "d_vr": d_vr, "s_vr": s_vr, "s_dur": dur,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

# ── E: Aggressive Max Profit ──
_E_COMBOS = [
    ("E1-VR430T55D4", 4, 30, 0.55, 0.60, 4),
    ("E2-VR430T60D4", 4, 30, 0.60, 0.65, 4),
]
for name, sp, lp, d_vr, s_vr, dur in _E_COMBOS:
    VARIANTS[name] = {
        "desc": f"VR({sp}/{lp}) d<{d_vr} s<{s_vr} dur={dur}",
        "sp": sp, "lp": lp, "d_vr": d_vr, "s_vr": s_vr, "s_dur": dur,
        "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    }

print(f"  Total variants defined: {len(VARIANTS)}")


# ══════════════════════════════════════════════════════════════════════════════
#  VR Precomputation — all ATR pairs computed ONCE per dataset
# ══════════════════════════════════════════════════════════════════════════════


def precompute_all_vr(datasets: list[dict]) -> None:
    """Precompute VR for all ATR_PAIRS in all datasets.

    Stores as d["vr_{sp}_{lp}"] for each symbol. The standard
    precompute() already stores d["vr"] = VR(7/50).
    """
    for dataset in datasets:
        for sym, d in dataset.items():
            c, h, lo = d["close"], d["high"], d["low"]
            for sp, lp in ATR_PAIRS:
                key = f"vr_{sp}_{lp}"
                if key not in d:
                    d[key] = compute_vr(c, h, lo, sp, lp)


# ══════════════════════════════════════════════════════════════════════════════
#  Core Engine — unified for all variants (all use custom VR + threshold)
# ══════════════════════════════════════════════════════════════════════════════


def _run_variant(
    var_cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
) -> np.ndarray:
    """Run one variant for one strategy across all symbols + TFs.

    All variants share the same logic:
      1. Look up precomputed VR by (sp, lp)
      2. Generate signal with threshold (+ duration for S_d8)
      3. Run sim

    Returns: PnL array (after per-symbol time-dedup).
    """
    sp = var_cfg["sp"]
    lp = var_cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    is_straddle = (strat == "S_d8")

    d_sl = var_cfg["d_sl"]
    s_sl = var_cfg["s_sl"]
    tp = var_cfg["tp"]
    d_vr = var_cfg["d_vr"]
    s_vr = var_cfg["s_vr"]
    s_dur = var_cfg["s_dur"]

    all_pnl: list[float] = []

    for sym in sorted(data_15m.keys()):
        # Collect trades from all TFs for overlap dedup
        sym_trades: list[tuple[int, int, float]] = []  # (entry_15m, exit_15m, pnl)

        for tf_label, ratio, warmup, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]

            vr = d.get(vr_key)
            if vr is None:
                continue

            sl = s_sl if is_straddle else d_sl
            mh = mh_s if is_straddle else mh_d

            if is_straddle:
                sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                p, e, x = sim_straddle_trades(c, h, lo, sig, sl, tp, mh, COST_STRADDLE)
            else:
                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trades(c, h, lo, sig, dire, sl, tp, mh, COST_DIRECTIONAL)

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        # Per-symbol dedup (time-based overlap removal)
        sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True
                    break
            if not conflict:
                all_pnl.append(pnl_val)
                accepted.append((entry_15m, exit_15m))

    return np.array(all_pnl) if all_pnl else np.empty(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation Workers
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_variant(args: tuple) -> dict:
    """Full-sample evaluation of one variant."""
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_years = _G["years"]

    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    d_pnl = _run_variant(var_cfg, data_15m, data_1h, data_4h, "D_mom8")
    s_pnl = _run_variant(var_cfg, data_15m, data_1h, data_4h, "S_d8")
    combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)

    for key, pnl in [("D_mom8", d_pnl), ("S_d8", s_pnl), ("combined", combined)]:
        m = compute_metrics(pnl, data_years)
        bs = bootstrap_pf(pnl, 1000)
        m["P_PF_ge_1"] = bs["P_PF_ge_1"]
        m["CI_lo"] = bs["CI_lo"]
        m["CI_hi"] = bs["CI_hi"]
        result[key] = m

    return result


def oos_evaluate_variant(args: tuple) -> dict:
    """OOS 67/33 evaluation using pre-split data from _G."""
    var_name, var_cfg = args

    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    for split_label in ["IS", "OOS"]:
        sd_15m = _G[f"{split_label}_15m"]
        sd_1h = _G[f"{split_label}_1h"]
        sd_4h = _G[f"{split_label}_4h"]
        years = _G[f"{split_label}_years"]

        d_pnl = _run_variant(var_cfg, sd_15m, sd_1h, sd_4h, "D_mom8")
        s_pnl = _run_variant(var_cfg, sd_15m, sd_1h, sd_4h, "S_d8")
        combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)

        for key, pnl in [("D_mom8", d_pnl), ("S_d8", s_pnl), ("combined", combined)]:
            m = compute_metrics(pnl, years)
            bs = bootstrap_pf(pnl, 1000)
            m["P_PF_ge_1"] = bs["P_PF_ge_1"]
            m["CI_lo"] = bs["CI_lo"]
            m["CI_hi"] = bs["CI_hi"]
            if key not in result:
                result[key] = {}
            result[key][split_label] = m

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  OOS Data Preparation (split ONCE, precompute all VR variants)
# ══════════════════════════════════════════════════════════════════════════════


def prepare_oos_splits(
    data_15m: dict, data_1h: dict, data_4h: dict,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Split data into IS/OOS and precompute all VR variants for each segment.

    Returns: (is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h)
    """
    is_15m: dict = {}
    oos_15m: dict = {}
    is_1h: dict = {}
    oos_1h: dict = {}
    is_4h: dict = {}
    oos_4h: dict = {}

    for sym in data_15m:
        n = len(data_15m[sym]["close"])
        cut = int(n * 0.67)

        sources = [
            (data_15m, is_15m, oos_15m),
            (data_1h, is_1h, oos_1h),
            (data_4h, is_4h, oos_4h),
        ]

        for src, is_dst, oos_dst in sources:
            if sym not in src:
                continue
            d = src[sym]
            n_tf = len(d["close"])
            ratio = n // n_tf if n_tf > 0 else 1
            cut_tf = cut // ratio

            for start, end_b, dst in [
                (0, cut_tf, is_dst),
                (cut_tf, n_tf, oos_dst),
            ]:
                seg_len = end_b - start
                if seg_len < 100:
                    continue
                d_seg: dict = {}
                for key in ("close", "high", "low", "volume"):
                    d_seg[key] = d[key][start:end_b].copy()
                d_seg["bars"] = seg_len
                c, h, lo = d_seg["close"], d_seg["high"], d_seg["low"]
                v = d_seg["volume"]

                # Standard indicators
                d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
                d_seg["mom_8"] = compute_momentum(c, 8)
                d_seg["vol_pct"] = vol_percentile(v, min(200, seg_len // 2))
                d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)

                # All VR variants
                for sp, lp in ATR_PAIRS:
                    d_seg[f"vr_{sp}_{lp}"] = compute_vr(c, h, lo, sp, lp)

                dst[sym] = d_seg

    return is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h


# ══════════════════════════════════════════════════════════════════════════════
#  Printing
# ══════════════════════════════════════════════════════════════════════════════


def print_table(results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 110}")
    print(f"  {label}")
    print(f"{'─' * 110}")
    header = (
        f"  {'Variant':<22} {'N':>6} {'PF':>6} {'EV_bp':>7} {'t':>5} "
        f"{'WR%':>5} {'R:R':>5} {'Ret%':>9} {'DD%':>5} {'TPD':>5} {'P≥1':>4}"
    )
    print(header)
    print("  " + "─" * 98)
    for r in results:
        m = r[strat_key]
        print(
            f"  {r['name']:<22} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
            f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['RR']:>5.2f} "
            f"{m['ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
            f"{m['tpd']:>5.2f} "
            f"{m.get('P_PF_ge_1', 0):>4.0f}%"
        )


def print_oos_table(results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 115}")
    print(f"  {label}")
    print(f"{'─' * 115}")
    print(
        f"  {'Variant':<22} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} "
        f"{'OOS_N':>6} {'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} "
        f"{'P≥1':>5} {'CI_lo':>6} {'CI_hi':>6} {'Ratio':>5}"
    )
    print("  " + "─" * 100)
    for r in results:
        if strat_key not in r or "IS" not in r[strat_key]:
            continue
        is_m = r[strat_key]["IS"]
        oos_m = r[strat_key]["OOS"]
        ratio = oos_m["PF"] / is_m["PF"] if is_m["PF"] > 0 else 0
        print(
            f"  {r['name']:<22} {is_m['N']:>6} {is_m['PF']:>6.3f} "
            f"{is_m['EV_bp']:>+7.1f} {oos_m['N']:>6} {oos_m['PF']:>6.3f} "
            f"{oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
            f"{oos_m.get('P_PF_ge_1', 0):>4.0f}% "
            f"{oos_m.get('CI_lo', 0):>6.3f} {oos_m.get('CI_hi', 0):>6.3f} "
            f"{ratio:>5.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 110)
    print("  R37: ATR PERIOD & STACKING GRID SEARCH — MAXIMIZE EV×N")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Base: VR(7/50)<0.50 deployed, R36 winner: VR(5/40)")
    print(f"  Variants: {len(VARIANTS)}  |  ATR pairs: {len(ATR_PAIRS)}")
    print(f"  DD filter: NONE (maximize total profit)")
    print("=" * 110)

    # ── Load data ──
    print("\n  Loading 15m OHLCV data...")
    data_15m = load_15m()
    total_15m = sum(d["bars"] for d in data_15m.values())
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    print(f"  Loaded {len(data_15m)} symbols, {total_15m:,} bars, ~{data_years:.1f} years")

    print("  Resampling 15m → 1h...")
    data_1h = build_resampled(data_15m, 4)
    print(f"  1h: {len(data_1h)} symbols, {sum(d['bars'] for d in data_1h.values()):,} bars")

    print("  Resampling 15m → 4h...")
    data_4h = build_resampled(data_15m, 16)
    print(f"  4h: {len(data_4h)} symbols, {sum(d['bars'] for d in data_4h.values()):,} bars")

    print("  Precomputing standard indicators...")
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)

    print(f"  Precomputing {len(ATR_PAIRS)} VR variants per symbol...")
    precompute_all_vr([data_15m, data_1h, data_4h])
    print("  Done.\n")

    # ── Global data ──
    global _G
    _G = {
        "15m": data_15m, "1h": data_1h, "4h": data_4h,
        "years": data_years,
    }

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 110)
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON")
    print("=" * 110)

    eval_args = [(vn, vc) for vn, vc in VARIANTS.items()]

    print(f"\n  Running {len(eval_args)} variants across {NUM_WORKERS} workers...")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(evaluate_variant, eval_args):
            results.append(r)
            print(f"    ✓ {r['name']:<22}  N={r['combined']['N']:>6}")

    results.sort(key=lambda x: x["name"])
    base_m = next(r for r in results if r["name"] == "T0-Baseline")["combined"]
    base_n = base_m["N"]
    base_ev = base_m["EV_bp"]
    base_evn = base_ev * base_n

    print_table(results, "FULL-SAMPLE: Combined D+S (deduped)", "combined")

    # ── Ranking by EV×N (primary metric) ──
    print(f"\n{'─' * 110}")
    print("  RANKING by EV×N (total expected profit) — PRIMARY METRIC")
    print(f"{'─' * 110}")
    ranked_evn = sorted(results, key=lambda x: x["combined"]["EV_bp"] * x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_evn, 1):
        m = r["combined"]
        evn = m["EV_bp"] * m["N"]
        delta_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        n_boost = (m["N"] / base_n - 1) * 100 if base_n > 0 else 0
        marker = " ◄" if r["name"] == "T0-Baseline" else ""
        marker += " ★" if r["name"] == "R36-VR540" else ""
        print(
            f"  #{i:>2} {r['name']:<22} EV×N={evn:>+10.0f}  "
            f"(EV={m['EV_bp']:>+.1f} × N={m['N']})  N+{n_boost:>+.0f}%  "
            f"PF={m['PF']:.3f}  DD={m['max_dd_pct']:.1f}%  "
            f"Δ={delta_pct:>+6.1f}%{marker}"
        )

    # ── Ranking by N ──
    print(f"\n{'─' * 110}")
    print("  RANKING by N (trade count)")
    print(f"{'─' * 110}")
    ranked_n = sorted(results, key=lambda x: x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_n, 1):
        m = r["combined"]
        n_boost = (m["N"] / base_n - 1) * 100 if base_n > 0 else 0
        delta_ev = m["EV_bp"] - base_ev
        marker = " ◄" if r["name"] == "T0-Baseline" else ""
        print(
            f"  #{i:>2} {r['name']:<22} N={m['N']:>6} (+{n_boost:>+.0f}%)  "
            f"EV={m['EV_bp']:>+7.1f}bp (Δ{delta_ev:>+.1f})  "
            f"PF={m['PF']:.3f}  t={m['t']:.2f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: OOS evaluation (pre-split data)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("  PHASE 2: OOS TEST (67/33 split)")
    print("=" * 110)

    print("\n  Preparing IS/OOS splits with all VR variants...")
    is_15m, is_1h, is_4h, oos_15m, oos_1h, oos_4h = prepare_oos_splits(
        data_15m, data_1h, data_4h,
    )
    is_years = data_years * 0.67
    oos_years = data_years * 0.33

    _G["IS_15m"] = is_15m
    _G["IS_1h"] = is_1h
    _G["IS_4h"] = is_4h
    _G["IS_years"] = is_years
    _G["OOS_15m"] = oos_15m
    _G["OOS_1h"] = oos_1h
    _G["OOS_4h"] = oos_4h
    _G["OOS_years"] = oos_years
    print("  Done.\n")

    oos_args = [(vn, VARIANTS[vn]) for vn in sorted(VARIANTS.keys())]

    print(f"  Running OOS for {len(oos_args)} variants...")
    oos_results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(oos_evaluate_variant, oos_args):
            oos_results.append(r)
            oos_n = r.get("combined", {}).get("OOS", {}).get("N", 0)
            print(f"    ✓ {r['name']:<22}  OOS_N={oos_n:>5}")

    oos_results.sort(key=lambda x: x["name"])

    print_oos_table(oos_results, "OOS: Combined D+S", "combined")

    # ── OOS Rankings ──
    oos_base = next(r for r in oos_results if r["name"] == "T0-Baseline")
    bl_oos = oos_base["combined"]["OOS"]
    bl_oos_ev = bl_oos["EV_bp"]
    bl_oos_n = bl_oos["N"]
    bl_oos_evn = bl_oos_ev * bl_oos_n

    # Primary: EV×N
    print(f"\n{'─' * 110}")
    print("  OOS RANKING by EV×N (total expected profit) — PRIMARY")
    print(f"{'─' * 110}")
    oos_ranked_evn = sorted(
        oos_results,
        key=lambda x: x.get("combined", {}).get("OOS", {}).get("EV_bp", 0)
                     * x.get("combined", {}).get("OOS", {}).get("N", 0),
        reverse=True,
    )
    for i, r in enumerate(oos_ranked_evn, 1):
        if "combined" not in r or "OOS" not in r["combined"]:
            continue
        oos_m = r["combined"]["OOS"]
        evn = oos_m["EV_bp"] * oos_m["N"]
        delta_pct = (evn / bl_oos_evn - 1) * 100 if bl_oos_evn != 0 else 0
        n_boost = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        marker = " ◄" if r["name"] == "T0-Baseline" else ""
        marker += " ★" if r["name"] == "R36-VR540" else ""
        print(
            f"  #{i:>2} {r['name']:<22} EV×N={evn:>+10.0f}  "
            f"(EV={oos_m['EV_bp']:>+.1f} × N={oos_m['N']})  N+{n_boost:.0f}%  "
            f"PF={oos_m['PF']:.3f}  DD={oos_m['max_dd_pct']:.1f}%  "
            f"Δ={delta_pct:>+6.1f}%  t={oos_m['t']:.2f}{marker}"
        )

    # By EV (per-trade quality)
    print(f"\n{'─' * 110}")
    print("  OOS RANKING by EV (per-trade quality)")
    print(f"{'─' * 110}")
    oos_ranked_ev = sorted(
        oos_results,
        key=lambda x: x.get("combined", {}).get("OOS", {}).get("EV_bp", -999),
        reverse=True,
    )
    for i, r in enumerate(oos_ranked_ev, 1):
        if "combined" not in r or "OOS" not in r["combined"]:
            continue
        oos_m = r["combined"]["OOS"]
        delta_ev = oos_m["EV_bp"] - bl_oos_ev
        n_boost = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        marker = " ◄" if r["name"] == "T0-Baseline" else ""
        print(
            f"  #{i:>2} {r['name']:<22} OOS_EV={oos_m['EV_bp']:>+7.1f}bp  "
            f"PF={oos_m['PF']:.3f}  N={oos_m['N']:>5} (+{n_boost:.0f}%)  "
            f"t={oos_m['t']:.2f}  DD={oos_m['max_dd_pct']:.1f}%  "
            f"CI_lo={oos_m.get('CI_lo', 0):.3f}  ΔEV={delta_ev:>+.1f}bp{marker}"
        )

    # ── Winners: EV > 0, PF > 1.0, t > 2.0, EV×N > baseline ──
    print(f"\n{'─' * 110}")
    print("  WINNERS: OOS variants where EV > 0 AND PF > 1.0 AND t > 2.0 AND EV×N > baseline")
    print(f"{'─' * 110}")
    winners = []
    for r in oos_results:
        if "combined" not in r or "OOS" not in r["combined"]:
            continue
        if r["name"] == "T0-Baseline":
            continue
        oos_m = r["combined"]["OOS"]
        evn = oos_m["EV_bp"] * oos_m["N"]
        if oos_m["EV_bp"] > 0 and oos_m["PF"] > 1.0 and oos_m["t"] > 2.0 and evn > bl_oos_evn:
            winners.append(r)
    winners.sort(
        key=lambda x: x["combined"]["OOS"]["EV_bp"] * x["combined"]["OOS"]["N"],
        reverse=True,
    )
    if winners:
        for i, r in enumerate(winners, 1):
            oos_m = r["combined"]["OOS"]
            is_m = r["combined"]["IS"]
            evn = oos_m["EV_bp"] * oos_m["N"]
            n_boost = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
            evn_boost = (evn / bl_oos_evn - 1) * 100 if bl_oos_evn != 0 else 0
            ratio = oos_m["PF"] / is_m["PF"] if is_m["PF"] > 0 else 0
            print(
                f"  #{i} {r['name']:<22} EV×N={evn:>+.0f} (+{evn_boost:.0f}%)  "
                f"N={oos_m['N']} (+{n_boost:.0f}%)  EV={oos_m['EV_bp']:>+.1f}bp  "
                f"PF={oos_m['PF']:.3f}  t={oos_m['t']:.2f}  "
                f"DD={oos_m['max_dd_pct']:.1f}%  IS/OOS={ratio:.2f}"
            )
    else:
        print("  ⚠️  No variant meets all criteria. Check individual rankings.")

    # ── Equity comparison at MDD ──
    print(f"\n{'─' * 110}")
    print("  MDD EQUITY CHECK: Even at worst drawdown, is equity > baseline?")
    print(f"{'─' * 110}")
    if winners:
        bl_peak_equity = bl_oos["ret_pct"]
        print(f"  Baseline total return: {bl_peak_equity:>+.1f}%  (DD: {bl_oos['max_dd_pct']:.1f}%)")
        print()
        for r in winners:
            oos_m = r["combined"]["OOS"]
            # Equity at MDD: total_return * (1 - max_dd)
            # Approximate: if ret=+X% and DD=Y%, equity at MDD ≈ (1+X)*(1-Y)-1
            total_ret = oos_m["ret_pct"] / 100
            max_dd = oos_m["max_dd_pct"] / 100
            # Peak equity = 1 + total_ret (if DD hasn't recovered yet) or higher
            # At worst: equity_at_mdd = peak * (1 - max_dd)
            # Conservative: assume peak = final equity (underestimate)
            equity_at_mdd_pct = ((1 + total_ret) * (1 - max_dd) - 1) * 100
            bl_equity_at_mdd_pct = ((1 + bl_peak_equity / 100) * (1 - bl_oos["max_dd_pct"] / 100) - 1) * 100
            ahead = equity_at_mdd_pct > bl_equity_at_mdd_pct
            status = "✅ AHEAD" if ahead else "❌ BEHIND"
            print(
                f"  {r['name']:<22} ret={oos_m['ret_pct']:>+.1f}%  "
                f"DD={oos_m['max_dd_pct']:.1f}%  "
                f"equity@MDD≈{equity_at_mdd_pct:>+.1f}%  "
                f"vs baseline@MDD≈{bl_equity_at_mdd_pct:>+.1f}%  {status}"
            )

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 110}")
    print(
        f"  R37 COMPLETE  |  {elapsed:.0f}s  |  {len(VARIANTS)} variants  "
        f"|  {NUM_WORKERS} workers  |  commit: {commit}"
    )
    print(f"{'=' * 110}")

    print(f"\n  Deployed baseline (T0): N={bl_oos_n}  EV={bl_oos_ev:>+.1f}bp  "
          f"PF={bl_oos['PF']:.3f}  DD={bl_oos['max_dd_pct']:.1f}%  "
          f"EV×N={bl_oos_evn:>+.0f}")

    if winners:
        best = winners[0]
        bm = best["combined"]["OOS"]
        evn_val = bm["EV_bp"] * bm["N"]
        n_boost = (bm["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        evn_boost = (evn_val / bl_oos_evn - 1) * 100 if bl_oos_evn != 0 else 0
        print(f"\n  🏆 Best EV×N winner: {best['name']}")
        print(
            f"    N={bm['N']} (+{n_boost:.0f}%)  EV={bm['EV_bp']:>+.1f}bp  "
            f"PF={bm['PF']:.3f}  EV×N={evn_val:>+.0f} (+{evn_boost:.0f}%)  "
            f"DD={bm['max_dd_pct']:.1f}%  t={bm['t']:.2f}"
        )
    else:
        print("\n  ⚠️  No winner found. Baseline remains optimal.")

    print(f"\n  ⏱️  Runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
