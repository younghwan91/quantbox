#!/usr/bin/env python3
"""R34: Multi-Condition Portfolio Architecture
================================================
Treats each validated R33 filter as an independent conviction layer.
Tests all 2^4 = 16 specific combinations, threshold scoring, and
conviction-weighted sizing.

Core Insight (from user):
  "여러가지 전략적 컨디션을 동시에 검토해도 되거든...
   하나의 진입 컨디션에 픽스 시킬 필요없어.
   overlap = more conviction, independence = more opportunities."

4 Validated Conditions (from R33 OOS):
  E2: SMA proximity — |close - SMA(20)| / SMA(20) < 0.5%
  C2: BTC neutral  — |BTC_mom_8| < 1%
  B3: Confluence   — other strategy also signals
  A3: Quality+     — S: sq_dur≥16, D: VR<0.35

Score = sum of passing conditions (0–4).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations

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

# Condition names
COND_E2 = 0   # SMA proximity
COND_C2 = 1   # BTC neutral
COND_B3 = 2   # Confluence
COND_A3 = 3   # Quality+
NCOND = 4
COND_NAMES = ["E2", "C2", "B3", "A3"]

# TF configs: (label, ratio_to_15m, warmup, max_hold_D, max_hold_S, squeeze_min_dur)
TF_15m = ("15m", 1, WARMUP_15m, 192, 384, 8)
TF_1h = ("1h", 4, WARMUP_1h, 48, 96, 8)
TF_4h = ("4h", 16, WARMUP_4h, 12, 24, 8)
G_ALL_TFS = [TF_15m, TF_1h, TF_4h]


# ══════════════════════════════════════════════════════════════════════════════
#  Global data (fork-based workers)
# ══════════════════════════════════════════════════════════════════════════════

_G: dict = {}


def _init_worker(g: dict) -> None:
    global _G
    _G = g


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Helpers
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 4 condition masks + total score per bar.

    Returns:
        e2, c2, b3, a3 (int8 each), scores (int32)
    """
    n = len(close)
    e2 = np.zeros(n, dtype=np.int8)
    c2 = np.zeros(n, dtype=np.int8)
    b3 = np.zeros(n, dtype=np.int8)
    a3 = np.zeros(n, dtype=np.int8)
    scores = np.zeros(n, dtype=np.int32)

    btc_len = len(btc_mom)

    for i in range(n):
        s = 0

        # E2: SMA proximity — |close - SMA(20)| / SMA(20) < 0.5%
        sma_val = sma_20[i]
        if sma_val > 0.0 and not np.isnan(sma_val):
            if abs(close[i] - sma_val) / sma_val < 0.005:
                e2[i] = 1
                s += 1

        # C2: BTC neutral — |BTC_mom_8| < 1%
        btc_idx = i * ratio
        if btc_idx < btc_len:
            bm = btc_mom[btc_idx]
            if not np.isnan(bm) and abs(bm) < 0.01:
                c2[i] = 1
                s += 1

        # B3: Confluence — other strategy also signals
        if is_straddle:
            if sig_d[i] != 0:
                b3[i] = 1
                s += 1
        else:
            if sig_s[i] != 0:
                b3[i] = 1
                s += 1

        # A3: Quality+
        if is_straddle:
            # Long squeeze: sq_dur >= 16
            if sq_dur_055[i] >= 16:
                a3[i] = 1
                s += 1
        else:
            # Deep VR: VR < 0.35 (extra compressed)
            if not np.isnan(vr[i]) and vr[i] < 0.35:
                a3[i] = 1
                s += 1

        scores[i] = s

    return e2, c2, b3, a3, scores


@numba.jit(nopython=True, cache=True)
def filter_by_required(
    sig: np.ndarray,
    e2: np.ndarray, c2: np.ndarray, b3: np.ndarray, a3: np.ndarray,
    req_e2: bool, req_c2: bool, req_b3: bool, req_a3: bool,
) -> np.ndarray:
    """Keep signals where ALL required conditions pass."""
    n = len(sig)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if sig[i] == 0:
            continue
        ok = True
        if req_e2 and e2[i] == 0:
            ok = False
        if req_c2 and c2[i] == 0:
            ok = False
        if req_b3 and b3[i] == 0:
            ok = False
        if req_a3 and a3[i] == 0:
            ok = False
        if ok:
            out[i] = sig[i]
    return out


@numba.jit(nopython=True, cache=True)
def filter_by_score(
    sig: np.ndarray, scores: np.ndarray, min_score: int,
) -> np.ndarray:
    """Keep signals where conviction score >= min_score."""
    n = len(sig)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if sig[i] != 0 and scores[i] >= min_score:
            out[i] = sig[i]
    return out


@numba.jit(nopython=True, cache=True)
def weight_pnl_linear(
    pnl: np.ndarray, entries: np.ndarray, scores: np.ndarray,
) -> np.ndarray:
    """Conviction-weighted PnL: PnL × (1 + 0.5 × score).

    Score 0 → 1.0x, 1 → 1.5x, 2 → 2.0x, 3 → 2.5x, 4 → 3.0x
    """
    n = len(pnl)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        eb = int(entries[i])
        w = 1.0 + 0.5 * scores[eb]
        out[i] = pnl[i] * w
    return out


@numba.jit(nopython=True, cache=True)
def weight_pnl_tiered(
    pnl: np.ndarray, entries: np.ndarray, scores: np.ndarray,
) -> np.ndarray:
    """Tiered conviction sizing. De-weight unconfirmed, amplify confirmed.

    Score 0 → 0.5x, 1 → 1.0x, 2 → 1.5x, 3 → 2.0x, 4 → 2.5x
    """
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
#  NUMBA: Simulation
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim → (pnl, entries, exits)."""
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
    """Directional sim → (pnl, entries, exits)."""
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

# ── Baseline ──
VARIANTS["T0-Base"] = {
    "type": "combo", "required": frozenset(),
    "desc": "Baseline (no filter)",
}

# ── Singles (4) ──
VARIANTS["F-E2"] = {
    "type": "combo", "required": frozenset(["E2"]),
    "desc": "SMA proximity only",
}
VARIANTS["F-C2"] = {
    "type": "combo", "required": frozenset(["C2"]),
    "desc": "BTC neutral only",
}
VARIANTS["F-B3"] = {
    "type": "combo", "required": frozenset(["B3"]),
    "desc": "Confluence only",
}
VARIANTS["F-A3"] = {
    "type": "combo", "required": frozenset(["A3"]),
    "desc": "Quality+ only",
}

# ── Pairs (6) ──
for _c1, _c2 in combinations(COND_NAMES, 2):
    _name = f"P-{_c1}{_c2}"
    VARIANTS[_name] = {
        "type": "combo", "required": frozenset([_c1, _c2]),
        "desc": f"{_c1}+{_c2}",
    }

# ── Triples (4) ──
for _c1, _c2, _c3 in combinations(COND_NAMES, 3):
    _name = f"X-{_c1}{_c2}{_c3}"
    VARIANTS[_name] = {
        "type": "combo", "required": frozenset([_c1, _c2, _c3]),
        "desc": f"{_c1}+{_c2}+{_c3}",
    }

# ── Quad ──
VARIANTS["Q-ALL"] = {
    "type": "combo", "required": frozenset(COND_NAMES),
    "desc": "All 4 conditions",
}

# ── Threshold (score >= k) ──
VARIANTS["T1-Union"] = {
    "type": "threshold", "min_score": 1,
    "desc": "Score ≥ 1 (any condition)",
}
VARIANTS["T2-Dual"] = {
    "type": "threshold", "min_score": 2,
    "desc": "Score ≥ 2",
}
VARIANTS["T3-Triple"] = {
    "type": "threshold", "min_score": 3,
    "desc": "Score ≥ 3",
}

# ── Conviction-Weighted ──
VARIANTS["CW-Lin"] = {
    "type": "cw", "mode": "linear",
    "desc": "PnL × (1 + 0.5·score)",
}
VARIANTS["CW-Tier"] = {
    "type": "cw", "mode": "tiered",
    "desc": "Tiered: 0→0.5x, 1→1x, 2→1.5x, 3→2x, 4→2.5x",
}

# Total: 1 + 4 + 6 + 4 + 1 + 3 + 2 = 21 variants


# ══════════════════════════════════════════════════════════════════════════════
#  Core Engine
# ══════════════════════════════════════════════════════════════════════════════


def _run_variant_strat(
    var_name: str,
    var_cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    btc_mom_15m: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Run one variant for one strategy (D_mom8 or S_d8) across all symbols + TFs.

    Returns:
        (deduped_pnl_array, n_raw, n_dedup)
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    vtype = var_cfg["type"]
    raw_pnls: list[float] = []
    deduped_pnls: list[float] = []

    is_straddle = (strat == "S_d8")

    for sym in sorted(data_15m.keys()):
        all_sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in G_ALL_TFS:
            d_tf = tf_data_map[ratio]
            if sym not in d_tf:
                continue
            d = d_tf[sym]
            n_bars = len(d["close"])

            mh = mh_d if strat == "D_mom8" else mh_s

            # ── Generate BOTH D and S signals (needed for B3 confluence) ──
            sig_d = gen_vr_signal(d["vr"], 0.45, warmup)
            sig_s = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)

            # The signal we're actually trading
            sig = sig_d if strat == "D_mom8" else sig_s

            # ── Compute condition masks + scores ──
            sma_20 = sma(d["close"], 20)
            e2, c2, b3, a3, scores = compute_conditions(
                d["close"], d["vr"], d["sq_dur_055"], sma_20,
                sig_d, sig_s, btc_mom_15m, ratio, is_straddle,
            )

            # ── Apply variant filtering ──
            if vtype == "combo":
                required = var_cfg["required"]
                if len(required) == 0:
                    sig_f = sig  # Baseline — no filtering
                else:
                    sig_f = filter_by_required(
                        sig, e2, c2, b3, a3,
                        "E2" in required, "C2" in required,
                        "B3" in required, "A3" in required,
                    )
            elif vtype == "threshold":
                sig_f = filter_by_score(sig, scores, var_cfg["min_score"])
            elif vtype == "cw":
                sig_f = sig  # All signals — PnL weighted later
            else:
                sig_f = sig

            # ── Simulation ──
            if strat == "D_mom8":
                dire = make_direction_momentum(sig_f, d["mom_8"])
                p, e, x = sim_directional_trades(
                    d["close"], d["high"], d["low"], sig_f, dire,
                    0.01, 0.05, mh, COST_DIRECTIONAL,
                )
            else:
                p, e, x = sim_straddle_trades(
                    d["close"], d["high"], d["low"], sig_f,
                    0.015, 0.05, mh, COST_STRADDLE,
                )

            # ── Apply conviction weighting if CW variant ──
            if vtype == "cw" and len(p) > 0:
                if var_cfg["mode"] == "linear":
                    p = weight_pnl_linear(p, e, scores)
                elif var_cfg["mode"] == "tiered":
                    p = weight_pnl_tiered(p, e, scores)

            # ── Collect trades with 15m-index mapping ──
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                all_sym_trades.append((entry_15m, exit_15m, float(p[k])))
                raw_pnls.append(float(p[k]))

        # ── Per-symbol dedup: chronological first-come-first-served ──
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
#  Score Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════


def compute_score_distribution(
    data_15m: dict, data_1h: dict, data_4h: dict, btc_mom_15m: np.ndarray,
) -> dict[str, dict]:
    """Compute score distribution for all baseline signals.

    Returns:
        {"D_mom8": {"total": N, "dist": [n0, n1, ..., n4], "cond_pct": [e2%, c2%, b3%, a3%]},
         "S_d8": {...},
         "cooccurrence": 4x4 matrix of pairwise co-occurrence rates}
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    result = {}

    for strat in ["D_mom8", "S_d8"]:
        is_straddle = (strat == "S_d8")
        dist = [0] * 5  # score 0,1,2,3,4
        cond_counts = [0] * 4  # E2, C2, B3, A3
        cooccur = np.zeros((4, 4), dtype=np.int64)  # pairwise co-occurrence
        total = 0

        for sym in sorted(data_15m.keys()):
            for tf_label, ratio, warmup, mh_d, mh_s, min_dur in G_ALL_TFS:
                d_tf = tf_data_map[ratio]
                if sym not in d_tf:
                    continue
                d = d_tf[sym]
                sig_d = gen_vr_signal(d["vr"], 0.45, warmup)
                sig_s = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)
                sig = sig_d if strat == "D_mom8" else sig_s
                sma_20 = sma(d["close"], 20)
                e2, c2, b3, a3, scores = compute_conditions(
                    d["close"], d["vr"], d["sq_dur_055"], sma_20,
                    sig_d, sig_s, btc_mom_15m, ratio, is_straddle,
                )
                # Count signals
                for i in range(len(sig)):
                    if sig[i] != 0:
                        total += 1
                        s = int(scores[i])
                        if 0 <= s <= 4:
                            dist[s] += 1
                        conds = [e2[i], c2[i], b3[i], a3[i]]
                        for ci in range(4):
                            if conds[ci]:
                                cond_counts[ci] += 1
                            for cj in range(ci, 4):
                                if conds[ci] and conds[cj]:
                                    cooccur[ci, cj] += 1
                                    if ci != cj:
                                        cooccur[cj, ci] += 1

        cond_pct = [c / total * 100 if total > 0 else 0 for c in cond_counts]
        dist_pct = [d / total * 100 if total > 0 else 0 for d in dist]

        result[strat] = {
            "total": total,
            "dist": dist,
            "dist_pct": dist_pct,
            "cond_pct": cond_pct,
            "cond_counts": cond_counts,
            "cooccurrence": cooccur,
        }

    return result


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
    """Full-sample evaluation of one variant. Uses global _G."""
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
    data_years = _G["years"]
    btc_mom_15m = _G["btc_mom"]
    result: dict = {"name": var_name, "desc": var_cfg["desc"]}

    d_pnl = s_pnl = None
    for strat in ["D_mom8", "S_d8"]:
        deduped, n_raw, n_dedup = _run_variant_strat(
            var_name, var_cfg, data_15m, data_1h, data_4h, strat, btc_mom_15m,
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

    # Combined
    combined = np.concatenate([d_pnl, s_pnl]) if len(d_pnl) + len(s_pnl) > 0 else np.empty(0)
    c_metrics = _compute_metrics(combined, data_years)
    c_bs = bootstrap_pf(combined, 1000)
    c_metrics["P_PF_ge_1"] = c_bs["P_PF_ge_1"]
    c_metrics["CI_lo"] = c_bs["CI_lo"]
    c_metrics["CI_hi"] = c_bs["CI_hi"]
    result["combined"] = c_metrics

    return result


def oos_variant(args: tuple) -> dict:
    """OOS test for one variant (67/33 split). Uses global _G."""
    var_name, var_cfg = args
    data_15m = _G["15m"]
    data_1h = _G["1h"]
    data_4h = _G["4h"]
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

        for sym in data_15m:
            n = len(data_15m[sym]["close"])
            cut = int(n * 0.67)
            for src, is_dst, oos_dst in [
                (data_15m, is_15m, oos_15m),
                (data_1h, is_1h, oos_1h),
                (data_4h, is_4h, oos_4h),
            ]:
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

        # Split BTC mom too
        btc_cut = int(len(btc_mom_15m) * 0.67)
        is_btc_mom = btc_mom_15m[:btc_cut] if len(btc_mom_15m) > 0 else np.array([])
        oos_btc_mom = btc_mom_15m[btc_cut:] if len(btc_mom_15m) > 0 else np.array([])

        if strat_key == "combined":
            is_d, _, _ = _run_variant_strat(
                var_name, var_cfg, is_15m, is_1h, is_4h, "D_mom8", is_btc_mom,
            )
            is_s, _, _ = _run_variant_strat(
                var_name, var_cfg, is_15m, is_1h, is_4h, "S_d8", is_btc_mom,
            )
            oos_d, _, _ = _run_variant_strat(
                var_name, var_cfg, oos_15m, oos_1h, oos_4h, "D_mom8", oos_btc_mom,
            )
            oos_s, _, _ = _run_variant_strat(
                var_name, var_cfg, oos_15m, oos_1h, oos_4h, "S_d8", oos_btc_mom,
            )
            is_pnl = np.concatenate([is_d, is_s]) if len(is_d) + len(is_s) > 0 else np.empty(0)
            oos_pnl = np.concatenate([oos_d, oos_s]) if len(oos_d) + len(oos_s) > 0 else np.empty(0)
        else:
            is_pnl, _, _ = _run_variant_strat(
                var_name, var_cfg, is_15m, is_1h, is_4h, strat_key, is_btc_mom,
            )
            oos_pnl, _, _ = _run_variant_strat(
                var_name, var_cfg, oos_15m, oos_1h, oos_4h, strat_key, oos_btc_mom,
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
#  Printing helpers
# ══════════════════════════════════════════════════════════════════════════════


def print_table(results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 95}")
    print(f"  {label}")
    print(f"{'─' * 95}")
    header = (
        f"  {'Variant':<17} {'N':>5} {'PF':>6} {'EV_bp':>7} {'t':>5} "
        f"{'WR%':>5} {'R:R':>5} {'Ret%':>9} {'DD%':>5} {'P≥1':>4}"
    )
    print(header)
    print("  " + "─" * 83)
    for r in results:
        m = r[strat_key]
        print(
            f"  {r['name']:<17} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+7.1f} "
            f"{m['t']:>5.2f} {m['WR']:>5.1f} {m['RR']:>5.2f} "
            f"{m['ret_pct']:>+9.1f} {m['max_dd_pct']:>5.1f} "
            f"{m.get('P_PF_ge_1', 0):>4.0f}%"
        )


def print_oos_table(oos_results: list[dict], label: str, strat_key: str) -> None:
    print(f"\n{'─' * 95}")
    print(f"  {label}")
    print(f"{'─' * 95}")
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

    print("=" * 95)
    print("  R34: MULTI-CONDITION PORTFOLIO ARCHITECTURE")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Base: G-ALL deduped (15m+1h+4h)  |  Workers: {NUM_WORKERS}")
    print(f"  Conditions: {', '.join(COND_NAMES)}")
    print(f"  Variants: {len(VARIANTS)}")
    print("=" * 95)

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

    print("  Precomputing indicators...")
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)

    # ── BTC data ──
    btc_mom_15m = np.array([])
    if "BTCUSDT" in data_15m:
        btc_mom_15m = data_15m["BTCUSDT"]["mom_8"].copy()
        print(f"  BTC cross-asset data: {len(btc_mom_15m):,} bars")
    else:
        print("  ⚠ BTCUSDT not found — C2 filter will be a no-op")

    # ── Global data ──
    global _G
    _G = {
        "15m": data_15m, "1h": data_1h, "4h": data_4h,
        "years": data_years, "btc_mom": btc_mom_15m,
    }
    print("  Done.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  SCORE DISTRIBUTION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 95)
    print("  SCORE DISTRIBUTION ANALYSIS")
    print("=" * 95)

    score_dist = compute_score_distribution(data_15m, data_1h, data_4h, btc_mom_15m)

    for strat in ["D_mom8", "S_d8"]:
        sd = score_dist[strat]
        print(f"\n  {strat} — {sd['total']:,} total signal bars")
        print(f"  {'Score':<8} {'Count':>8} {'%':>7}")
        print("  " + "─" * 25)
        for s in range(5):
            print(f"  {s:<8} {sd['dist'][s]:>8,} {sd['dist_pct'][s]:>6.1f}%")

        print(f"\n  Condition pass rates:")
        for ci, cn in enumerate(COND_NAMES):
            print(f"    {cn}: {sd['cond_pct'][ci]:.1f}% ({sd['cond_counts'][ci]:,})")

        print(f"\n  Co-occurrence matrix (% of signals where BOTH conditions pass):")
        print(f"  {'':>4} ", end="")
        for cn in COND_NAMES:
            print(f"  {cn:>5}", end="")
        print()
        cooc = sd["cooccurrence"]
        total = sd["total"]
        for ci, cn in enumerate(COND_NAMES):
            print(f"  {cn:>4} ", end="")
            for cj in range(4):
                pct = cooc[ci, cj] / total * 100 if total > 0 else 0
                print(f"  {pct:>5.1f}", end="")
            print()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample evaluation
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 95}")
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON")
    print(f"{'=' * 95}")

    eval_args = [(vn, vc) for vn, vc in VARIANTS.items()]

    print(f"\n  Running {len(eval_args)} variants across {NUM_WORKERS} workers...")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker, initargs=(_G,),
    ) as executor:
        for r in executor.map(evaluate_variant, eval_args):
            results.append(r)
            print(f"    ✓ {r['name']}")

    results.sort(key=lambda x: x["name"])

    print_table(results, "FULL-SAMPLE: D_mom8 (deduped)", "D_mom8")
    print_table(results, "FULL-SAMPLE: S_d8 (deduped)", "S_d8")
    print_table(results, "FULL-SAMPLE: Combined D+S (deduped)", "combined")

    # ── Ranking ──
    print(f"\n{'─' * 95}")
    print("  RANKING by Combined EV (bp)")
    print(f"{'─' * 95}")
    baseline_ev = next(r for r in results if r["name"] == "T0-Base")["combined"]["EV_bp"]
    ranked = sorted(results, key=lambda x: x["combined"]["EV_bp"], reverse=True)
    for i, r in enumerate(ranked, 1):
        m = r["combined"]
        delta = m["EV_bp"] - baseline_ev
        marker = " ◄" if r["name"] == "T0-Base" else ""
        n_cond = len(VARIANTS[r["name"]].get("required", set()))
        print(
            f"  #{i:>2} {r['name']:<17} EV={m['EV_bp']:>+7.1f}bp  PF={m['PF']:.3f}  "
            f"N={m['N']:>5}  t={m['t']:.2f}  DD={m['max_dd_pct']:.1f}%  "
            f"Δ_vs_base={delta:>+6.1f}bp{marker}"
        )

    # ── Portfolio EV × N analysis ──
    print(f"\n{'─' * 95}")
    print("  PORTFOLIO METRIC: EV × N  (total expected bp)")
    print(f"{'─' * 95}")
    ranked_evn = sorted(results, key=lambda x: x["combined"]["EV_bp"] * x["combined"]["N"], reverse=True)
    for i, r in enumerate(ranked_evn, 1):
        m = r["combined"]
        evn = m["EV_bp"] * m["N"]
        base_evn = baseline_ev * next(rr for rr in results if rr["name"] == "T0-Base")["combined"]["N"]
        delta_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} EV×N={evn:>+9.0f}  "
            f"(EV={m['EV_bp']:>+.1f} × N={m['N']})  "
            f"Δ_vs_base={delta_pct:>+5.1f}%{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: OOS for top variants + baseline
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 95}")
    print("  PHASE 2: OOS TEST (67/33 split) — Top 10 + Baseline")
    print(f"{'=' * 95}")

    oos_names = {"T0-Base"}
    # Add top 8 by EV and top 2 by EV×N that aren't already included
    for r in ranked[:8]:
        oos_names.add(r["name"])
    for r in ranked_evn[:4]:
        oos_names.add(r["name"])
    # Always include threshold & CW variants
    oos_names |= {"T1-Union", "T2-Dual", "T3-Triple", "CW-Lin", "CW-Tier"}

    oos_args = [(vn, VARIANTS[vn]) for vn in sorted(oos_names)]

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

    # ── OOS Ranking ──
    print(f"\n{'─' * 95}")
    print("  OOS RANKING by Combined OOS EV (bp)")
    print(f"{'─' * 95}")
    oos_baseline = next(r for r in oos_results if r["name"] == "T0-Base")
    bl_oos_ev = oos_baseline["combined"]["OOS"]["EV_bp"]
    bl_oos_n = oos_baseline["combined"]["OOS"]["N"]
    oos_ranked = sorted(
        oos_results, key=lambda x: x["combined"]["OOS"]["EV_bp"], reverse=True,
    )
    for i, r in enumerate(oos_ranked, 1):
        oos_m = r["combined"]["OOS"]
        delta = oos_m["EV_bp"] - bl_oos_ev
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} OOS_EV={oos_m['EV_bp']:>+7.1f}bp  "
            f"PF={oos_m['PF']:.3f}  N={oos_m['N']:>5}  t={oos_m['t']:.2f}  "
            f"DD={oos_m['max_dd_pct']:.1f}%  Δ={delta:>+6.1f}bp{marker}"
        )

    # ── OOS EV × N Ranking ──
    print(f"\n{'─' * 95}")
    print("  OOS PORTFOLIO METRIC: EV × N")
    print(f"{'─' * 95}")
    oos_ranked_evn = sorted(
        oos_results,
        key=lambda x: x["combined"]["OOS"]["EV_bp"] * x["combined"]["OOS"]["N"],
        reverse=True,
    )
    base_evn = bl_oos_ev * bl_oos_n
    for i, r in enumerate(oos_ranked_evn, 1):
        oos_m = r["combined"]["OOS"]
        evn = oos_m["EV_bp"] * oos_m["N"]
        delta_pct = (evn / base_evn - 1) * 100 if base_evn != 0 else 0
        marker = " ◄" if r["name"] == "T0-Base" else ""
        print(
            f"  #{i:>2} {r['name']:<17} EV×N={evn:>+9.0f}  "
            f"(EV={oos_m['EV_bp']:>+.1f} × N={oos_m['N']})  "
            f"Δ_vs_base={delta_pct:>+5.1f}%{marker}"
        )

    # ── Marginal Condition Value (OOS) ──
    print(f"\n{'─' * 95}")
    print("  MARGINAL CONDITION VALUE (OOS Combined)")
    print("  = How much does each condition improve EV when it's the only filter?")
    print(f"{'─' * 95}")
    for cn in COND_NAMES:
        fn = f"F-{cn}"
        fr = next((r for r in oos_results if r["name"] == fn), None)
        if fr:
            oos_m = fr["combined"]["OOS"]
            delta = oos_m["EV_bp"] - bl_oos_ev
            n_kept = oos_m["N"]
            n_pct = n_kept / bl_oos_n * 100 if bl_oos_n > 0 else 0
            print(
                f"  {cn:<4}  ΔEV={delta:>+6.1f}bp  N={n_kept:>5} ({n_pct:.0f}% kept)  "
                f"PF={oos_m['PF']:.3f}  t={oos_m['t']:.2f}"
            )

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 95}")
    print(
        f"  R34 COMPLETE  |  {elapsed:.0f}s  |  {len(VARIANTS)} variants  "
        f"|  {NUM_WORKERS} workers"
    )
    print(f"{'=' * 95}")

    best_oos = oos_ranked[0]
    bm = best_oos["combined"]["OOS"]
    print(f"\n  Best OOS Combined (by EV): {best_oos['name']}")
    print(
        f"    N={bm['N']}  PF={bm['PF']:.3f}  EV={bm['EV_bp']:>+.1f}bp  "
        f"t={bm['t']:.2f}  WR={bm['WR']:.1f}%  DD={bm['max_dd_pct']:.1f}%"
    )

    best_evn = oos_ranked_evn[0]
    bn = best_evn["combined"]["OOS"]
    evn_val = bn["EV_bp"] * bn["N"]
    print(f"\n  Best OOS Combined (by EV×N): {best_evn['name']}")
    print(
        f"    N={bn['N']}  PF={bn['PF']:.3f}  EV={bn['EV_bp']:>+.1f}bp  "
        f"EV×N={evn_val:>+.0f}  t={bn['t']:.2f}"
    )

    bl_m = oos_baseline["combined"]["OOS"]
    print(
        f"\n  Baseline OOS: N={bl_m['N']}  PF={bl_m['PF']:.3f}  "
        f"EV={bl_m['EV_bp']:>+.1f}bp  t={bl_m['t']:.2f}  DD={bl_m['max_dd_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
