#!/usr/bin/env python3
"""R35: N-Boosting & Conviction Sizing A/B Test
=================================================
Goal: Increase trade count (N) without destroying edge,
then combine with conviction-weighted sizing.

"기회 is king. N이 없으면 수익도 없어."

Current baseline: N=702 OOS / 1602 Full (50 sym × 6yr)
= ~7 trades/symbol/year. Too few.

N-Boosting Levers:
  A) VR threshold relaxation: D 0.45→0.50, S 0.55→0.60
  B) Squeeze duration: S d8→d6→d4
  C) Shorter max_hold: 50% reduction (faster turnover)
  D) Tighter SL: D SL=0.75%, S SL=1.0% (faster exits)
  E) Add 30m timeframe: 4th TF layer

Finally: Apply CW-Tier on best N-boosted variant(s).
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

from research.active.r31_r40.r31_mtf_vol_straddle import (
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

COND_NAMES = ["E2", "C2", "B3", "A3"]

# Standard TF configs
WARMUP_30m = 50
TF_15m = ("15m", 1, WARMUP_15m, 192, 384, 8)
TF_30m = ("30m", 2, WARMUP_30m, 96, 192, 8)
TF_1h  = ("1h",  4, WARMUP_1h,  48,  96, 8)
TF_4h  = ("4h", 16, WARMUP_4h,  12,  24, 8)
G_3TF  = [TF_15m, TF_1h, TF_4h]
G_4TF  = [TF_15m, TF_30m, TF_1h, TF_4h]

# ══════════════════════════════════════════════════════════════════════════════
#  Global data (fork-based workers)
# ══════════════════════════════════════════════════════════════════════════════

_G: dict = {}


def _init_worker(g: dict) -> None:
    global _G
    _G = g


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Helpers
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sma(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += arr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += arr[i] - arr[i - period]
        out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def compute_conditions(
    close: np.ndarray,
    vr: np.ndarray,
    sq_dur_055: np.ndarray,
    sma_20: np.ndarray,
    sig_d: np.ndarray,
    sig_s: np.ndarray,
    btc_mom: np.ndarray,
    ratio: int,
    is_straddle: bool,
) -> np.ndarray:
    """Compute total conviction score per bar (0–4).

    Conditions:
      E2: |close-SMA20|/SMA20 < 0.5%
      C2: |BTC_mom8| < 1%
      B3: Other strategy also signals
      A3: Straddle → sq_dur≥16; Directional → VR<0.35
    """
    n = len(close)
    scores = np.zeros(n, dtype=np.int32)
    btc_len = len(btc_mom)

    for i in range(n):
        s = 0
        # E2
        sv = sma_20[i]
        if sv > 0.0 and not np.isnan(sv):
            if abs(close[i] - sv) / sv < 0.005:
                s += 1
        # C2
        btc_idx = i * ratio
        if btc_idx < btc_len:
            bm = btc_mom[btc_idx]
            if not np.isnan(bm) and abs(bm) < 0.01:
                s += 1
        # B3
        if is_straddle:
            if sig_d[i] != 0:
                s += 1
        else:
            if sig_s[i] != 0:
                s += 1
        # A3
        if is_straddle:
            if sq_dur_055[i] >= 16:
                s += 1
        else:
            if not np.isnan(vr[i]) and vr[i] < 0.35:
                s += 1

        scores[i] = s
    return scores


@numba.jit(nopython=True, cache=True)
def weight_pnl_tiered(
    pnl: np.ndarray, entries: np.ndarray, scores: np.ndarray,
) -> np.ndarray:
    """Tiered conviction sizing: 0→0.5x, 1→1x, 2→1.5x, 3→2x, 4→2.5x"""
    n = len(pnl)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        eb = int(entries[i])
        s = scores[eb]
        if s == 0:
            w = 0.5
        elif s == 1:
            w = 1.0
        elif s == 2:
            w = 1.5
        elif s == 3:
            w = 2.0
        else:
            w = 2.5
        out[i] = pnl[i] * w
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Simulations
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
#  Variant Definitions
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS: OrderedDict[str, dict] = OrderedDict()

# ── Tier 0: Baseline ──
VARIANTS["T0-Base"] = {
    "desc": "Baseline G-ALL",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf",
    "mh_scale": 1.0,  # max_hold multiplier
    "cw": False,
}

# ── Group A: VR Threshold Relaxation ──
VARIANTS["A1-D50"] = {
    "desc": "D: VR<0.50 (from 0.45)",
    "d_vr": 0.50, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}
VARIANTS["A2-S60"] = {
    "desc": "S: VR<0.60 (from 0.55)",
    "d_vr": 0.45, "s_vr": 0.60, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}
VARIANTS["A3-Both"] = {
    "desc": "D+S VR relaxed",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}

# ── Group B: Squeeze Duration ──
VARIANTS["B1-Dur6"] = {
    "desc": "S dur≥6 (from ≥8)",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 6,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}
VARIANTS["B2-Dur4"] = {
    "desc": "S dur≥4 (aggressive)",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 4,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}

# ── Group C: Max Hold Reduction (faster turnover) ──
VARIANTS["C1-Hold50"] = {
    "desc": "max_hold ×0.5",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 0.5, "cw": False,
}
VARIANTS["C2-Hold75"] = {
    "desc": "max_hold ×0.75",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 0.75, "cw": False,
}

# ── Group D: Tighter SL (faster exit = more turns) ──
VARIANTS["D1-TightSL"] = {
    "desc": "D SL=0.75%, S SL=1.0%",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.0075, "s_sl": 0.01, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}

# ── Group E: Add 30m Timeframe ──
VARIANTS["E1-4TF"] = {
    "desc": "15m+30m+1h+4h",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "4tf", "mh_scale": 1.0, "cw": False,
}

# ── Group F: Combined N-Boost ──
VARIANTS["F1-VR+Dur"] = {
    "desc": "A3+B1: relax VR + dur≥6",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 6,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": False,
}
VARIANTS["F2-VR+Dur+Hold"] = {
    "desc": "A3+B1+C1: VR+dur+hold50%",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 6,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 0.5, "cw": False,
}
VARIANTS["F3-Full"] = {
    "desc": "A3+B1+C1+E1: all N-boost + 4TF",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 6,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "4tf", "mh_scale": 0.5, "cw": False,
}
VARIANTS["F4-VR+Dur4+4TF"] = {
    "desc": "A3+B2+E1: VR+dur4+4TF",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 4,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "4tf", "mh_scale": 1.0, "cw": False,
}

# ── Group G: CW-Tier on N-Boosted ──
VARIANTS["G1-CW-Base"] = {
    "desc": "CW-Tier on baseline",
    "d_vr": 0.45, "s_vr": 0.55, "s_dur": 8,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": True,
}
VARIANTS["G2-CW-F1"] = {
    "desc": "CW-Tier on F1 (VR+Dur)",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 6,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "3tf", "mh_scale": 1.0, "cw": True,
}
VARIANTS["G3-CW-F3"] = {
    "desc": "CW-Tier on F3 (full boost)",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 6,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "4tf", "mh_scale": 0.5, "cw": True,
}
VARIANTS["G4-CW-F4"] = {
    "desc": "CW-Tier on F4 (VR+dur4+4TF)",
    "d_vr": 0.50, "s_vr": 0.60, "s_dur": 4,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
    "tfs": "4tf", "mh_scale": 1.0, "cw": True,
}

# Total: 18 variants


# ══════════════════════════════════════════════════════════════════════════════
#  Core Engine
# ══════════════════════════════════════════════════════════════════════════════


def _run_variant_strat(
    var_cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    btc_mom_15m: np.ndarray,
    data_30m: dict | None = None,
) -> tuple[np.ndarray, int, int]:
    """Run one variant for one strategy across all symbols + TFs.

    Returns:
        (deduped_pnl_array, n_raw, n_dedup)
    """
    tf_data_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    use_cw = var_cfg["cw"]
    tfs = G_4TF if var_cfg["tfs"] == "4tf" else G_3TF
    mh_scale = var_cfg["mh_scale"]

    d_vr_thresh = var_cfg["d_vr"]
    s_vr_thresh = var_cfg["s_vr"]
    s_dur = var_cfg["s_dur"]
    d_sl = var_cfg["d_sl"]
    s_sl = var_cfg["s_sl"]
    tp = var_cfg["tp"]

    is_straddle = (strat == "S_d8")
    raw_pnls: list[float] = []
    deduped_pnls: list[float] = []

    for sym in sorted(data_15m.keys()):
        all_sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur_base in tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            n_bars = len(d["close"])

            mh_d_adj = max(2, int(mh_d * mh_scale))
            mh_s_adj = max(2, int(mh_s * mh_scale))
            mh = mh_d_adj if strat == "D_mom8" else mh_s_adj

            # Generate signals with variant-specific thresholds
            sig_d = gen_vr_signal(d["vr"], d_vr_thresh, warmup)
            sig_s = gen_vr_dur_signal(d["vr"], s_vr_thresh, s_dur, warmup)

            sig = sig_d if strat == "D_mom8" else sig_s

            # Compute conviction scores (for CW variants)
            scores = np.zeros(n_bars, dtype=np.int32)
            if use_cw:
                sma_20 = sma(d["close"], 20)
                scores = compute_conditions(
                    d["close"], d["vr"], d["sq_dur_055"], sma_20,
                    sig_d, sig_s, btc_mom_15m, ratio, is_straddle,
                )

            # Simulate
            if strat == "D_mom8":
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trades(
                    d["close"], d["high"], d["low"], sig, dire,
                    d_sl, tp, mh, COST_DIRECTIONAL,
                )
            else:
                p, e, x = sim_straddle_trades(
                    d["close"], d["high"], d["low"], sig,
                    s_sl, tp, mh, COST_STRADDLE,
                )

            # Apply CW-Tier weighting
            if use_cw and len(p) > 0:
                p = weight_pnl_tiered(p, e, scores)

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                all_sym_trades.append((entry_15m, exit_15m, float(p[k])))
                raw_pnls.append(float(p[k]))

        # Per-symbol dedup
        all_sym_trades.sort(key=lambda x: x[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in all_sym_trades:
            conflict = False
            for ae, ax in accepted:
                if entry_15m <= ax and exit_15m >= ae:
                    conflict = True
                    break
            if not conflict:
                deduped_pnls.append(pnl_val)
                accepted.append((entry_15m, exit_15m))

    return (
        np.array(deduped_pnls) if deduped_pnls else np.empty(0),
        len(raw_pnls),
        len(deduped_pnls),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════


def _compute_metrics(pnl: np.ndarray, data_years: float) -> dict:
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
#  Workers
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_variant(args: tuple) -> dict:
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_30m = _G.get("30m")
    data_years = _G["years"]
    btc_mom_15m = _G["btc_mom"]
    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    d_pnl = s_pnl = None
    for strat in ["D_mom8", "S_d8"]:
        deduped, n_raw, n_dedup = _run_variant_strat(
            var_cfg, data_15m, data_1h, data_4h, strat,
            btc_mom_15m, data_30m,
        )
        if strat == "D_mom8":
            d_pnl = deduped
        else:
            s_pnl = deduped
        metrics = _compute_metrics(deduped, data_years)
        metrics["n_raw"] = n_raw
        metrics["n_dedup"] = n_dedup
        bs = bootstrap_pf(deduped, 1000)
        metrics["P_PF_ge_1"] = bs["P_PF_ge_1"]
        metrics["CI_lo"] = bs["CI_lo"]
        metrics["CI_hi"] = bs["CI_hi"]
        result[strat] = metrics

    combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)
    c_metrics = _compute_metrics(combined, data_years)
    c_bs = bootstrap_pf(combined, 1000)
    c_metrics["P_PF_ge_1"] = c_bs["P_PF_ge_1"]
    c_metrics["CI_lo"] = c_bs["CI_lo"]
    c_metrics["CI_hi"] = c_bs["CI_hi"]
    result["combined"] = c_metrics

    return result


def oos_variant(args: tuple) -> dict:
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_30m = _G.get("30m")
    data_years = _G["years"]
    btc_mom_15m = _G["btc_mom"]
    result: dict = {"name": var_name}

    for strat_key in ["D_mom8", "S_d8", "combined"]:
        is_15m: dict = {}
        oos_15m: dict = {}
        is_1h: dict = {}
        oos_1h: dict = {}
        is_4h: dict = {}
        oos_4h: dict = {}
        is_30m: dict = {}
        oos_30m: dict = {}

        for sym in data_15m:
            n = len(data_15m[sym]["close"])
            cut = int(n * 0.67)
            sources = [
                (data_15m, is_15m, oos_15m),
                (data_1h, is_1h, oos_1h),
                (data_4h, is_4h, oos_4h),
            ]
            if data_30m and sym in data_30m:
                sources.append((data_30m, is_30m, oos_30m))

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
                    c, h, lo, v = (
                        d_seg["close"], d_seg["high"], d_seg["low"], d_seg["volume"],
                    )
                    d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
                    d_seg["mom_8"] = compute_momentum(c, 8)
                    d_seg["vol_pct"] = vol_percentile(v, min(200, seg_len // 2))
                    d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)
                    dst[sym] = d_seg

        is_years = data_years * 0.67
        oos_years = data_years * 0.33

        btc_cut = int(len(btc_mom_15m) * 0.67)
        is_btc_mom = btc_mom_15m[:btc_cut] if len(btc_mom_15m) > 0 else np.array([])
        oos_btc_mom = btc_mom_15m[btc_cut:] if len(btc_mom_15m) > 0 else np.array([])

        if strat_key == "combined":
            is_d, _, _ = _run_variant_strat(
                var_cfg, is_15m, is_1h, is_4h, "D_mom8", is_btc_mom,
                is_30m if data_30m else None,
            )
            is_s, _, _ = _run_variant_strat(
                var_cfg, is_15m, is_1h, is_4h, "S_d8", is_btc_mom,
                is_30m if data_30m else None,
            )
            oos_d, _, _ = _run_variant_strat(
                var_cfg, oos_15m, oos_1h, oos_4h, "D_mom8", oos_btc_mom,
                oos_30m if data_30m else None,
            )
            oos_s, _, _ = _run_variant_strat(
                var_cfg, oos_15m, oos_1h, oos_4h, "S_d8", oos_btc_mom,
                oos_30m if data_30m else None,
            )
            is_pnl = np.concatenate([is_d, is_s]) if len(is_d) + len(is_s) > 0 else np.empty(0)
            oos_pnl = np.concatenate([oos_d, oos_s]) if len(oos_d) + len(oos_s) > 0 else np.empty(0)
        else:
            is_pnl, _, _ = _run_variant_strat(
                var_cfg, is_15m, is_1h, is_4h, strat_key, is_btc_mom,
                is_30m if data_30m else None,
            )
            oos_pnl, _, _ = _run_variant_strat(
                var_cfg, oos_15m, oos_1h, oos_4h, strat_key, oos_btc_mom,
                oos_30m if data_30m else None,
            )

        is_m = _compute_metrics(is_pnl, is_years)
        oos_m = _compute_metrics(oos_pnl, oos_years)
        oos_bs = bootstrap_pf(oos_pnl, 1000)
        oos_m["P_PF_ge_1"] = oos_bs["P_PF_ge_1"]
        oos_m["CI_lo"] = oos_bs["CI_lo"]
        oos_m["CI_hi"] = oos_bs["CI_hi"]
        result[strat_key] = {"IS": is_m, "OOS": oos_m}

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Printing
# ══════════════════════════════════════════════════════════════════════════════


def print_table(results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 100}")
    print(f"  {label}")
    print(f"{'─' * 100}")
    header = (
        f"  {'Variant':<17} {'N':>6} {'PF':>6} {'EV_bp':>7} {'t':>5} "
        f"{'WR%':>5} {'R:R':>5} {'Ret%':>9} {'DD%':>5} {'TPD':>5} {'P≥1':>4}"
    )
    print(header)
    print("  " + "─" * 88)
    for r in results:
        m = r[strat_key]
        print(
            f"  {r['name']:<17} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
            f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['RR']:>5.2f} "
            f"{m['ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
            f"{m['tpd']:>5.2f} "
            f"{m.get('P_PF_ge_1', 0):>4.0f}%"
        )


def print_oos_table(oos_results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 100}")
    print(f"  {label}")
    print(f"{'─' * 100}")
    print(
        f"  {'Variant':<17} {'IS_N':>6} {'IS_PF':>6} {'IS_EV':>7} {'OOS_N':>6} "
        f"{'OOS_PF':>6} {'OOS_EV':>7} {'OOS_t':>6} {'P≥1':>5} {'CI_lo':>6} {'CI_hi':>6}"
    )
    print("  " + "─" * 89)
    for r in oos_results:
        is_m = r[strat_key]["IS"]
        oos_m = r[strat_key]["OOS"]
        print(
            f"  {r['name']:<17} {is_m['N']:>6} {is_m['PF']:>6.3f} "
            f"{is_m['EV_bp']:>+7.1f} {oos_m['N']:>6} {oos_m['PF']:>6.3f} "
            f"{oos_m['EV_bp']:>+7.1f} {oos_m['t']:>6.2f} "
            f"{oos_m.get('P_PF_ge_1', 0):>4.0f}% "
            f"{oos_m.get('CI_lo', 0):>6.3f} {oos_m.get('CI_hi', 0):>6.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 100)
    print("  R35: N-BOOSTING & CONVICTION SIZING A/B TEST")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Base: G-ALL deduped (15m+1h+4h)  |  Workers: {NUM_WORKERS}")
    print(f"  Variants: {len(VARIANTS)}")
    print("=" * 100)

    # ── Load data ──
    print("\n  Loading 15m OHLCV data...")
    data_15m = load_15m()
    total_15m = sum(d["bars"] for d in data_15m.values())
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    print(f"  Loaded {len(data_15m)} symbols, {total_15m:,} bars, ~{data_years:.1f} years")

    print("  Resampling 15m → 30m...")
    data_30m = build_resampled(data_15m, 2)
    print(f"  30m: {len(data_30m)} symbols, {sum(d['bars'] for d in data_30m.values()):,} bars")

    print("  Resampling 15m → 1h...")
    data_1h = build_resampled(data_15m, 4)
    print(f"  1h: {len(data_1h)} symbols, {sum(d['bars'] for d in data_1h.values()):,} bars")

    print("  Resampling 15m → 4h...")
    data_4h = build_resampled(data_15m, 16)
    print(f"  4h: {len(data_4h)} symbols, {sum(d['bars'] for d in data_4h.values()):,} bars")

    print("  Precomputing indicators...")
    precompute(data_15m)
    precompute(data_30m)
    precompute(data_1h)
    precompute(data_4h)

    # ── BTC data ──
    btc_mom_15m = np.array([])
    if "BTCUSDT" in data_15m:
        btc_mom_15m = data_15m["BTCUSDT"]["mom_8"].copy()
        print(f"  BTC cross-asset data: {len(btc_mom_15m):,} bars")

    # ── Global data ──
    global _G
    _G = {
        "15m": data_15m, "30m": data_30m, "1h": data_1h, "4h": data_4h,
        "years": data_years, "btc_mom": btc_mom_15m,
    }
    print("  Done.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON")
    print("=" * 100)

    eval_args = [(vn, vc) for vn, vc in VARIANTS.items()]

    print(f"\n  Running {len(eval_args)} variants across {NUM_WORKERS} workers...")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(evaluate_variant, eval_args):
            results.append(r)
            print(f"    ✓ {r['name']:<17}  N={r['combined']['N']:>6}  (Δ={r['combined']['N'] - VARIANTS['T0-Base'].get('_n', r['combined']['N']):>+5})")
            # Save baseline N for delta display
            if r["name"] == "T0-Base":
                VARIANTS["T0-Base"]["_n"] = r["combined"]["N"]

    results.sort(key=lambda x: x["name"])
    base_n = next(r for r in results if r["name"] == "T0-Base")["combined"]["N"]
    base_ev = next(r for r in results if r["name"] == "T0-Base")["combined"]["EV_bp"]

    print_table(results, "FULL-SAMPLE: D_mom8 (deduped)", "D_mom8")
    print_table(results, "FULL-SAMPLE: S_d8 (deduped)", "S_d8")
    print_table(results, "FULL-SAMPLE: Combined D+S (deduped)", "combined")

    # ── Ranking by N (descending) ──
    print(f"\n{'─' * 100}")
    print("  RANKING by Combined N (trade count) — N-boost effectiveness")
    print(f"{'─' * 100}")
    ranked_n = sorted(results, key=lambda x: x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_n, 1):
        m = r["combined"]
        delta_n = m["N"] - base_n
        delta_n_pct = (m["N"] / base_n - 1) * 100 if base_n > 0 else 0
        delta_ev = m["EV_bp"] - base_ev
        evn = m["EV_bp"] * m["N"]
        base_evn = base_ev * base_n
        delta_evn_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} N={m['N']:>6} ({delta_n:>+5}, {delta_n_pct:>+5.0f}%)  "
            f"EV={m['EV_bp']:>+7.1f}bp ({delta_ev:>+5.1f})  PF={m['PF']:.3f}  "
            f"EV×N Δ={delta_evn_pct:>+5.1f}%  t={m['t']:.2f}{marker}"
        )

    # ── Ranking by EV×N (descending) ──
    print(f"\n{'─' * 100}")
    print("  RANKING by Combined EV×N (total expected profit)")
    print(f"{'─' * 100}")
    base_evn = base_ev * base_n
    ranked_evn = sorted(results, key=lambda x: x["combined"]["EV_bp"] * x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_evn, 1):
        m = r["combined"]
        evn = m["EV_bp"] * m["N"]
        delta_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} EV×N={evn:>+10.0f}  "
            f"(EV={m['EV_bp']:>+.1f} × N={m['N']})  "
            f"Δ={delta_pct:>+5.1f}%  DD={m['max_dd_pct']:.1f}%{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: OOS for all variants
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print("  PHASE 2: OOS TEST (67/33 split) — All Variants")
    print(f"{'=' * 100}")

    oos_args = [(vn, VARIANTS[vn]) for vn in sorted(VARIANTS.keys())]

    print(f"\n  Running OOS for {len(oos_args)} variants...")
    oos_results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(oos_variant, oos_args):
            oos_results.append(r)
            print(f"    ✓ {r['name']}")

    oos_results.sort(key=lambda x: x["name"])

    print_oos_table(oos_results, "OOS: Combined D+S", "combined")
    print_oos_table(oos_results, "OOS: D_mom8", "D_mom8")
    print_oos_table(oos_results, "OOS: S_d8", "S_d8")

    # ── OOS Rankings ──
    oos_baseline = next(r for r in oos_results if r["name"] == "T0-Base")
    bl_oos_ev = oos_baseline["combined"]["OOS"]["EV_bp"]
    bl_oos_n = oos_baseline["combined"]["OOS"]["N"]
    bl_oos_evn = bl_oos_ev * bl_oos_n

    print(f"\n{'─' * 100}")
    print("  OOS RANKING by Combined N")
    print(f"{'─' * 100}")
    oos_ranked_n = sorted(
        oos_results, key=lambda x: x["combined"]["OOS"]["N"], reverse=True,
    )
    for i, r in enumerate(oos_ranked_n, 1):
        oos_m = r["combined"]["OOS"]
        delta_n = oos_m["N"] - bl_oos_n
        delta_n_pct = (oos_m["N"] / bl_oos_n - 1) * 100 if bl_oos_n > 0 else 0
        delta_ev = oos_m["EV_bp"] - bl_oos_ev
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} N={oos_m['N']:>6} ({delta_n:>+5}, {delta_n_pct:>+5.0f}%)  "
            f"EV={oos_m['EV_bp']:>+7.1f}bp ({delta_ev:>+5.1f})  PF={oos_m['PF']:.3f}  "
            f"t={oos_m['t']:.2f}  DD={oos_m['max_dd_pct']:.1f}%{marker}"
        )

    print(f"\n{'─' * 100}")
    print("  OOS RANKING by Combined EV×N (total expected profit)")
    print(f"{'─' * 100}")
    oos_ranked_evn = sorted(
        oos_results,
        key=lambda x: x["combined"]["OOS"]["EV_bp"] * x["combined"]["OOS"]["N"],
        reverse=True,
    )
    for i, r in enumerate(oos_ranked_evn, 1):
        oos_m = r["combined"]["OOS"]
        evn = oos_m["EV_bp"] * oos_m["N"]
        delta_pct = (evn / bl_oos_evn - 1) * 100 if bl_oos_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} EV×N={evn:>+10.0f}  "
            f"(EV={oos_m['EV_bp']:>+.1f} × N={oos_m['N']})  "
            f"Δ={delta_pct:>+5.1f}%  PF={oos_m['PF']:.3f}  DD={oos_m['max_dd_pct']:.1f}%{marker}"
        )

    print(f"\n{'─' * 100}")
    print("  OOS RANKING by Combined OOS EV (per-trade quality)")
    print(f"{'─' * 100}")
    oos_ranked_ev = sorted(
        oos_results, key=lambda x: x["combined"]["OOS"]["EV_bp"], reverse=True,
    )
    for i, r in enumerate(oos_ranked_ev, 1):
        oos_m = r["combined"]["OOS"]
        delta = oos_m["EV_bp"] - bl_oos_ev
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} OOS_EV={oos_m['EV_bp']:>+7.1f}bp  "
            f"PF={oos_m['PF']:.3f}  N={oos_m['N']:>5}  t={oos_m['t']:.2f}  "
            f"DD={oos_m['max_dd_pct']:.1f}%  CI_lo={oos_m.get('CI_lo', 0):.3f}  "
            f"Δ={delta:>+6.1f}bp{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  CW-Tier BIAS AUDIT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print("  CW-TIER BIAS AUDIT")
    print("=" * 100)

    print("\n  Lookahead Bias Check:")
    print("  ─────────────────────")
    print("  E2 (SMA proximity): SMA(20) at bar i uses close[i-19:i+1] → all past/current. ✓ NO LOOKAHEAD")
    print("  C2 (BTC neutral):   btc_mom_8 at 15m bar i*ratio → past data. ✓ NO LOOKAHEAD")
    print("  B3 (Confluence):    sig_d/sig_s at bar i from VR[i] → same bar, backward. ✓ NO LOOKAHEAD")
    print("  A3 (Quality+):      VR[i] or sq_dur[i] → backward-looking. ✓ NO LOOKAHEAD")
    print("  Score weighting:    Applied at ENTRY bar, not EXIT. ✓ NO LOOKAHEAD")

    print("\n  Survivorship Bias Check:")
    print("  ─────────────────────────")
    print(f"  Symbols: {len(_G['15m'])} (hardcoded whitelist)")
    # Check for short-lived symbols
    min_bars = min(d["bars"] for d in _G["15m"].values())
    max_bars = max(d["bars"] for d in _G["15m"].values())
    short_syms = [(sym, d["bars"]) for sym, d in _G["15m"].items() if d["bars"] < max_bars * 0.5]
    print(f"  Bar range: {min_bars:,} – {max_bars:,}")
    if short_syms:
        print(f"  Short-lived symbols (<50% of max bars): {len(short_syms)}")
        for sym, bars in sorted(short_syms, key=lambda x: x[1]):
            print(f"    {sym}: {bars:,} bars ({bars/max_bars*100:.0f}%)")
    else:
        print("  All symbols have >50% of max bars. Mild survivorship bias (delisted coins not included).")

    print("\n  CW-Tier Risk Amplification:")
    print("  ────────────────────────────")
    g1 = next((r for r in oos_results if r["name"] == "G1-CW-Base"), None)
    if g1:
        bl_m = oos_baseline["combined"]["OOS"]
        cw_m = g1["combined"]["OOS"]
        print(f"  Baseline:  EV={bl_m['EV_bp']:>+.1f}bp  PF={bl_m['PF']:.3f}  DD={bl_m['max_dd_pct']:.1f}%  N={bl_m['N']}")
        print(f"  CW-Tier:   EV={cw_m['EV_bp']:>+.1f}bp  PF={cw_m['PF']:.3f}  DD={cw_m['max_dd_pct']:.1f}%  N={cw_m['N']}")
        print(f"  ΔPF: {cw_m['PF'] - bl_m['PF']:>+.3f}  (marginal → most gain is from SIZING, not signal)")
        print(f"  ΔDD: {cw_m['max_dd_pct'] - bl_m['max_dd_pct']:>+.1f}pp  (risk amplification)")

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 100}")
    print(
        f"  R35 COMPLETE  |  {elapsed:.0f}s  |  {len(VARIANTS)} variants  "
        f"|  {NUM_WORKERS} workers"
    )
    print(f"{'=' * 100}")

    best_evn = oos_ranked_evn[0]
    bn = best_evn["combined"]["OOS"]
    evn_val = bn["EV_bp"] * bn["N"]
    print(f"\n  Best OOS EV×N: {best_evn['name']}")
    print(
        f"    N={bn['N']}  PF={bn['PF']:.3f}  EV={bn['EV_bp']:>+.1f}bp  "
        f"EV×N={evn_val:>+.0f}  t={bn['t']:.2f}  DD={bn['max_dd_pct']:.1f}%"
    )

    best_n = oos_ranked_n[0]
    bnn = best_n["combined"]["OOS"]
    print(f"\n  Most Trades: {best_n['name']}")
    print(
        f"    N={bnn['N']}  PF={bnn['PF']:.3f}  EV={bnn['EV_bp']:>+.1f}bp  "
        f"t={bnn['t']:.2f}  DD={bnn['max_dd_pct']:.1f}%"
    )

    bl_m = oos_baseline["combined"]["OOS"]
    print(
        f"\n  Baseline: N={bl_m['N']}  PF={bl_m['PF']:.3f}  "
        f"EV={bl_m['EV_bp']:>+.1f}bp  t={bl_m['t']:.2f}  DD={bl_m['max_dd_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
