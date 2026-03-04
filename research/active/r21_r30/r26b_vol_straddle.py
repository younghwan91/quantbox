#!/usr/bin/env python3
"""
R26b: Deep-Dive on VolRatio Signal (V8 from R26)
==================================================
R26 found: V8:VolRatio<0.5 + Fix:SL1/TP4/48h → PF=1.163, N=1,307

This script stress-tests that finding:
  1. Threshold sweep: 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6
  2. Period sweep: short period (5,7,10,14) × long period (30,50,70,100)
  3. Exit sweep: more SL/TP combos around the working config
  4. Per-symbol breakdown: is it concentrated in a few coins or broad?
  5. Time stability: first half vs second half
  6. Combination: VolRatio + duration filter, VolRatio + BBsqueeze
  7. Baseline: random signal for each config (null hypothesis)
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COST_PER_SIDE = 0.0004
FEE_4LEGS = COST_PER_SIDE * 4


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA indicators
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _atr_pct(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = (high[0] - low[0]) / close[0] if close[0] > 0 else 0.0
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        mx = max(hl, max(hc, lc))
        tr[i] = mx / close[i] if close[i] > 0 else 0.0
    if n >= period:
        atr = 0.0
        for k in range(period):
            atr += tr[k]
        atr /= period
        out[period - 1] = atr
        for i in range(period, n):
            atr = (atr * (period - 1) + tr[i]) / period
            out[i] = atr
    return out


@numba.jit(nopython=True, cache=True)
def _bbw(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += close[i - period + 1 + k]
        mean = s / period
        v = 0.0
        for k in range(period):
            d = close[i - period + 1 + k] - mean
            v += d * d
        v /= period
        if mean > 0:
            out[i] = 4.0 * np.sqrt(v) / mean
    return out


@numba.jit(nopython=True, cache=True)
def _pct_rank(arr: np.ndarray, lookback: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, 50.0)
    for i in range(lookback - 1, n):
        cur = arr[i]
        if np.isnan(cur):
            continue
        below = 0
        valid = 0
        for k in range(lookback):
            v = arr[i - lookback + 1 + k]
            if not np.isnan(v):
                valid += 1
                if v < cur:
                    below += 1
        if valid > 0:
            out[i] = below / valid * 100.0
    return out


@numba.jit(nopython=True, cache=True)
def _consecutive_below(arr: np.ndarray, threshold: float) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.int32)
    cnt = 0
    for i in range(n):
        if not np.isnan(arr[i]) and arr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        out[i] = cnt
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: VolRatio signal generation
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def gen_volratio_signal(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    short_period: int, long_period: int, threshold: float,
) -> np.ndarray:
    """Return int8 signal array: 1 where ATR(short)/ATR(long) < threshold."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    atr_s = _atr_pct(close, high, low, short_period)
    atr_l = _atr_pct(close, high, low, long_period)
    warmup = long_period + 10
    for i in range(warmup, n):
        if not np.isnan(atr_s[i]) and not np.isnan(atr_l[i]) and atr_l[i] > 0:
            if atr_s[i] / atr_l[i] < threshold:
                sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_volratio_plus_bbsq(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    short_period: int, long_period: int, vr_threshold: float,
    bb_period: int, pct_lookback: int, bb_pct_threshold: float,
) -> np.ndarray:
    """VolRatio + BB squeeze combined."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    atr_s = _atr_pct(close, high, low, short_period)
    atr_l = _atr_pct(close, high, low, long_period)
    bbw = _bbw(close, bb_period)
    bbw_rank = _pct_rank(bbw, pct_lookback)
    warmup = max(long_period, pct_lookback) + 10
    for i in range(warmup, n):
        if np.isnan(atr_s[i]) or np.isnan(atr_l[i]) or atr_l[i] <= 0:
            continue
        vr = atr_s[i] / atr_l[i]
        if vr < vr_threshold and bbw_rank[i] < bb_pct_threshold:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_volratio_with_duration(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    short_period: int, long_period: int, threshold: float,
    min_duration: int,
) -> np.ndarray:
    """VolRatio signal only after being below threshold for min_duration bars."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    atr_s = _atr_pct(close, high, low, short_period)
    atr_l = _atr_pct(close, high, low, long_period)

    # Compute vol ratio
    vr = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_s[i]) and not np.isnan(atr_l[i]) and atr_l[i] > 0:
            vr[i] = atr_s[i] / atr_l[i]

    dur = _consecutive_below(vr, threshold)
    warmup = long_period + min_duration + 10
    for i in range(warmup, n):
        if dur[i] >= min_duration:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_random_signal(n: int, interval: int, warmup: int) -> np.ndarray:
    sig = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if (i - warmup) % interval == 0:
            sig[i] = 1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Straddle simulation (fixed SL/TP)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    oc = np.empty(cap, dtype=np.int64)
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
        if le >= tp * 0.9:
            o = 1
        elif se >= tp * 0.9:
            o = 2
        elif le <= -sl * 0.9 and se <= -sl * 0.9:
            o = 0
        else:
            o = 3
        if cnt < cap:
            pnl[cnt] = net
            oc[cnt] = o
            cnt += 1
        i = eb + 1
    return pnl[:cnt], oc[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_data() -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings
    settings = QuantBoxSettings()
    whitelist = settings.rb_symbol_whitelist.split(",")
    data: dict = {}
    for sym in whitelist:
        candidates = list(cache_dir.glob(f"{sym}_15m_*.npz"))
        if not candidates:
            continue
        best = max(candidates, key=_extract_days)
        npz = np.load(best)
        required = {"close", "high", "low", "volume"}
        if not required.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        h = npz["high"].astype(np.float64)
        lo = npz["low"].astype(np.float64)
        v = npz["volume"].astype(np.float64)
        if len(c) < 300:
            continue
        data[sym] = {"close": c, "high": h, "low": lo, "volume": v, "bars": len(c)}
    return data


def stats(pnl_arr: np.ndarray, oc_arr: np.ndarray) -> dict | None:
    n = len(pnl_arr)
    if n < 5:
        return None
    wins = int(np.sum((oc_arr == 1) | (oc_arr == 2)))
    whips = int(np.sum(oc_arr == 0))
    pos = pnl_arr[pnl_arr > 0]
    neg = pnl_arr[pnl_arr < 0]
    gp = float(np.sum(pos)) if len(pos) > 0 else 0.0
    gn = float(np.abs(np.sum(neg))) if len(neg) > 0 else 1e-9
    pf = gp / gn
    ev = float(np.mean(pnl_arr)) * 10_000
    se = float(np.std(pnl_arr) / np.sqrt(n)) * 10_000 if n > 1 else 999.0
    t_stat = ev / se if se > 0 else 0.0
    return {
        "N": n, "Win%": wins / n * 100, "Whip%": whips / n * 100,
        "PF": pf, "EV_bp": ev, "SE_bp": se, "t": t_stat,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    now = datetime.now(timezone.utc)

    print("=" * 95)
    print("  R26b: VOLRATIO SIGNAL DEEP-DIVE & ROBUSTNESS CHECK")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Starting point: V8:VolRatio<0.5 + Fix:SL1/TP4/48h → PF=1.163, N=1,307 (R26)")
    print("  Tests: threshold sweep, period sweep, exit sweep, time split, combinations")
    print("=" * 95)
    print()

    data = load_data()
    if not data:
        print("  ❌ No data!")
        return
    print(f"  Loaded: {len(data)} symbols, {sum(d['bars'] for d in data.values()):,} total bars")
    print()
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 1: Threshold sweep (short=7, long=50)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ TEST 1: THRESHOLD SWEEP (ATR7/ATR50) ══")
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    exit_configs = [
        ("SL0.5/TP2/48h", 0.005, 0.020, 192),
        ("SL0.5/TP3/48h", 0.005, 0.030, 192),
        ("SL1.0/TP3/48h", 0.010, 0.030, 192),
        ("SL1.0/TP4/48h", 0.010, 0.040, 192),
        ("SL1.0/TP5/48h", 0.010, 0.050, 192),
        ("SL1.5/TP4/48h", 0.015, 0.040, 192),
        ("SL1.5/TP5/96h", 0.015, 0.050, 384),
        ("SL2.0/TP5/96h", 0.020, 0.050, 384),
        ("SL2.0/TP6/96h", 0.020, 0.060, 384),
    ]

    # Also generate random baseline for each exit
    random_baseline: dict[str, dict] = {}

    hdr = f"  {'Thr':<6}"
    for ec in exit_configs:
        hdr += f"  {ec[0]:>14}"
    print(hdr)
    print("  " + "─" * (6 + 16 * len(exit_configs)))

    for thr in thresholds:
        row = f"  {thr:<6.2f}"
        for ec_label, sl, tp, mh in exit_configs:
            all_pnl = []
            all_oc = []
            # Random for baseline (only once per exit config)
            if ec_label not in random_baseline:
                rnd_pnl = []
                rnd_oc = []

            for sym, d in data.items():
                c, h, lo = d["close"], d["high"], d["low"]
                sig = gen_volratio_signal(c, h, lo, 7, 50, thr)
                p, o = sim_straddle(c, h, lo, sig, sl, tp, mh, FEE_4LEGS)
                all_pnl.append(p)
                all_oc.append(o)

                if ec_label not in random_baseline:
                    rsig = gen_random_signal(len(c), 96, 60)
                    rp, ro = sim_straddle(c, h, lo, rsig, sl, tp, mh, FEE_4LEGS)
                    rnd_pnl.append(rp)
                    rnd_oc.append(ro)

            if ec_label not in random_baseline:
                rnd_all = np.concatenate(rnd_pnl)
                rnd_oc_all = np.concatenate(rnd_oc)
                random_baseline[ec_label] = stats(rnd_all, rnd_oc_all) or {"PF": 0}

            merged_pnl = np.concatenate(all_pnl)
            merged_oc = np.concatenate(all_oc)
            s = stats(merged_pnl, merged_oc)
            if s:
                m = "✓" if s["PF"] >= 1.0 else " "
                row += f"  {m}{s['PF']:.3f}({s['N']:>5})"
            else:
                row += f"       N/A      "
        print(row)

    # Print random baseline
    row = f"  {'RAND':<6}"
    for ec_label, _, _, _ in exit_configs:
        if ec_label in random_baseline:
            rb = random_baseline[ec_label]
            row += f"   {rb.get('PF', 0):.3f}({rb.get('N', 0):>5})"
        else:
            row += f"       N/A      "
    print(row)
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 2: Period sweep (threshold=0.5)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ TEST 2: PERIOD SWEEP (threshold=0.5, exit=SL1/TP4/48h) ══")
    short_periods = [3, 5, 7, 10, 14, 20]
    long_periods = [30, 50, 70, 100, 150]
    sl, tp, mh = 0.01, 0.04, 192

    hdr = f"  {'S\\L':<6}"
    for lp in long_periods:
        hdr += f"  {'ATR'+str(lp):>14}"
    print(hdr)
    print("  " + "─" * (6 + 16 * len(long_periods)))

    for sp in short_periods:
        row = f"  ATR{sp:<3}"
        for lp in long_periods:
            if sp >= lp:
                row += f"       ---      "
                continue
            all_pnl = []
            all_oc = []
            for sym, d in data.items():
                c, h, lo = d["close"], d["high"], d["low"]
                sig = gen_volratio_signal(c, h, lo, sp, lp, 0.5)
                p, o = sim_straddle(c, h, lo, sig, sl, tp, mh, FEE_4LEGS)
                all_pnl.append(p)
                all_oc.append(o)
            merged = np.concatenate(all_pnl)
            m_oc = np.concatenate(all_oc)
            s = stats(merged, m_oc)
            if s:
                m = "✓" if s["PF"] >= 1.0 else " "
                row += f"  {m}{s['PF']:.3f}({s['N']:>5})"
            else:
                row += f"       N/A      "
        print(row)
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 3: Per-symbol breakdown (best config from TEST 1)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ TEST 3: PER-SYMBOL BREAKDOWN (ATR7/ATR50 < 0.5, SL1/TP4/48h) ══")
    print(f"  {'Symbol':<16} {'N':>5}  {'Win%':>5}  {'Whip%':>5}  {'PF':>6}  {'EV(bp)':>8}  {'t-stat':>6}")
    print("  " + "─" * 60)

    sym_results = []
    for sym, d in sorted(data.items()):
        c, h, lo = d["close"], d["high"], d["low"]
        sig = gen_volratio_signal(c, h, lo, 7, 50, 0.5)
        p, o = sim_straddle(c, h, lo, sig, 0.01, 0.04, 192, FEE_4LEGS)
        s = stats(p, o)
        if s and s["N"] >= 5:
            sym_results.append((sym, s))
            m = "✅" if s["PF"] >= 1.0 else "  "
            print(
                f"  {m}{sym:<14} {s['N']:>5}  {s['Win%']:>5.1f}  "
                f"{s['Whip%']:>5.1f}  {s['PF']:>6.3f}  {s['EV_bp']:>+8.1f}  {s['t']:>6.2f}"
            )

    if sym_results:
        pf_above = sum(1 for _, s in sym_results if s["PF"] >= 1.0)
        print(f"\n  Symbols with PF ≥ 1.0: {pf_above}/{len(sym_results)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 4: Time stability (first half vs second half)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ══ TEST 4: TIME STABILITY (first half vs second half) ══")
    for thr in [0.4, 0.45, 0.5, 0.55]:
        for ec_label, sl, tp, mh in [
            ("SL1/TP4/48h", 0.01, 0.04, 192),
            ("SL1/TP5/48h", 0.01, 0.05, 192),
            ("SL1.5/TP5/96h", 0.015, 0.05, 384),
        ]:
            h1_pnl, h1_oc = [], []
            h2_pnl, h2_oc = [], []
            for sym, d in data.items():
                c, h, lo = d["close"], d["high"], d["low"]
                sig = gen_volratio_signal(c, h, lo, 7, 50, thr)
                mid = len(c) // 2

                # First half
                p1, o1 = sim_straddle(c[:mid], h[:mid], lo[:mid], sig[:mid], sl, tp, mh, FEE_4LEGS)
                h1_pnl.append(p1)
                h1_oc.append(o1)
                # Second half
                p2, o2 = sim_straddle(c[mid:], h[mid:], lo[mid:], sig[mid:], sl, tp, mh, FEE_4LEGS)
                h2_pnl.append(p2)
                h2_oc.append(o2)

            s1 = stats(np.concatenate(h1_pnl), np.concatenate(h1_oc))
            s2 = stats(np.concatenate(h2_pnl), np.concatenate(h2_oc))
            if s1 and s2:
                m1 = "✅" if s1["PF"] >= 1.0 else "❌"
                m2 = "✅" if s2["PF"] >= 1.0 else "❌"
                print(
                    f"  VR<{thr:.2f} {ec_label:<14} "
                    f"H1: {m1} PF={s1['PF']:.3f} N={s1['N']:>5}  "
                    f"H2: {m2} PF={s2['PF']:.3f} N={s2['N']:>5}  "
                    f"ΔPF={s2['PF']-s1['PF']:>+.3f}"
                )

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 5: Combinations (VolRatio + BB, VolRatio + duration)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ══ TEST 5: COMBINATIONS ══")

    combo_configs = [
        ("VR<0.5+BB<10%", 0.5, 10.0),
        ("VR<0.5+BB<5%", 0.5, 5.0),
        ("VR<0.5+BB<3%", 0.5, 3.0),
        ("VR<0.45+BB<10%", 0.45, 10.0),
        ("VR<0.45+BB<5%", 0.45, 5.0),
    ]

    dur_configs = [
        ("VR<0.5 dur≥5", 0.5, 5),
        ("VR<0.5 dur≥10", 0.5, 10),
        ("VR<0.5 dur≥20", 0.5, 20),
        ("VR<0.45 dur≥5", 0.45, 5),
        ("VR<0.45 dur≥10", 0.45, 10),
    ]

    sl, tp, mh = 0.01, 0.04, 192

    print(f"  {'Label':<22} {'N':>6}  {'Win%':>5}  {'PF':>6}  {'EV(bp)':>8}  {'t-stat':>6}")
    print("  " + "─" * 60)

    # VR + BB combinations
    for label, vr_thr, bb_pct in combo_configs:
        all_pnl, all_oc = [], []
        for sym, d in data.items():
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_volratio_plus_bbsq(c, h, lo, 7, 50, vr_thr, 20, 200, bb_pct)
            p, o = sim_straddle(c, h, lo, sig, sl, tp, mh, FEE_4LEGS)
            all_pnl.append(p)
            all_oc.append(o)
        merged = np.concatenate(all_pnl)
        m_oc = np.concatenate(all_oc)
        s = stats(merged, m_oc)
        if s:
            m = "✅" if s["PF"] >= 1.0 else "❌"
            print(
                f"  {m}{label:<20} {s['N']:>6}  {s['Win%']:>5.1f}  "
                f"{s['PF']:>6.3f}  {s['EV_bp']:>+8.1f}  {s['t']:>6.2f}"
            )

    # VR + duration combinations
    for label, vr_thr, min_dur in dur_configs:
        all_pnl, all_oc = [], []
        for sym, d in data.items():
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_volratio_with_duration(c, h, lo, 7, 50, vr_thr, min_dur)
            p, o = sim_straddle(c, h, lo, sig, sl, tp, mh, FEE_4LEGS)
            all_pnl.append(p)
            all_oc.append(o)
        merged = np.concatenate(all_pnl)
        m_oc = np.concatenate(all_oc)
        s = stats(merged, m_oc)
        if s:
            m = "✅" if s["PF"] >= 1.0 else "❌"
            print(
                f"  {m}{label:<20} {s['N']:>6}  {s['Win%']:>5.1f}  "
                f"{s['PF']:>6.3f}  {s['EV_bp']:>+8.1f}  {s['t']:>6.2f}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST 6: Bootstrap confidence interval for best config
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ══ TEST 6: BOOTSTRAP CI (ATR7/ATR50<0.5, SL1/TP4/48h) ══")
    all_pnl_best = []
    all_oc_best = []
    for sym, d in data.items():
        c, h, lo = d["close"], d["high"], d["low"]
        sig = gen_volratio_signal(c, h, lo, 7, 50, 0.5)
        p, o = sim_straddle(c, h, lo, sig, 0.01, 0.04, 192, FEE_4LEGS)
        all_pnl_best.append(p)
        all_oc_best.append(o)
    best_pnl = np.concatenate(all_pnl_best)
    best_oc = np.concatenate(all_oc_best)

    if len(best_pnl) >= 20:
        n_boot = 10000
        boot_pfs = np.empty(n_boot)
        rng = np.random.default_rng(42)
        for b in range(n_boot):
            idx = rng.integers(0, len(best_pnl), size=len(best_pnl))
            sample = best_pnl[idx]
            pos = sample[sample > 0]
            neg = sample[sample < 0]
            gp = float(np.sum(pos)) if len(pos) > 0 else 0.0
            gn = float(np.abs(np.sum(neg))) if len(neg) > 0 else 1e-9
            boot_pfs[b] = gp / gn

        p5 = np.percentile(boot_pfs, 5)
        p25 = np.percentile(boot_pfs, 25)
        p50 = np.percentile(boot_pfs, 50)
        p75 = np.percentile(boot_pfs, 75)
        p95 = np.percentile(boot_pfs, 95)
        prob_above_1 = np.mean(boot_pfs >= 1.0) * 100

        print(f"    Bootstrap (N={len(best_pnl)}, 10K resamples):")
        print(f"    PF  5th={p5:.3f}  25th={p25:.3f}  50th={p50:.3f}  75th={p75:.3f}  95th={p95:.3f}")
        print(f"    P(PF ≥ 1.0) = {prob_above_1:.1f}%")

        s_best = stats(best_pnl, best_oc)
        if s_best:
            print(f"    Point estimate: PF={s_best['PF']:.3f}  EV={s_best['EV_bp']:>+.1f}bp  t={s_best['t']:.2f}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  Final summary
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"{'='*95}")
    print(f"  RUNTIME: {elapsed:.1f}s")

    # Check if any config is truly robust
    s_best = stats(best_pnl, best_oc)
    if s_best:
        pf = s_best["PF"]
        t_val = s_best["t"]
        if pf >= 1.0 and t_val >= 2.0:
            print(f"  ✅ STATISTICALLY SIGNIFICANT: PF={pf:.3f}, t={t_val:.2f}")
        elif pf >= 1.0 and t_val >= 1.0:
            print(f"  ⚠️  MARGINAL SIGNIFICANCE: PF={pf:.3f}, t={t_val:.2f}")
        elif pf >= 1.0:
            print(f"  ⚠️  PF ≥ 1.0 but NOT significant: PF={pf:.3f}, t={t_val:.2f}")
        else:
            print(f"  ❌ PF < 1.0: {pf:.3f}, t={t_val:.2f}")
    print(f"{'='*95}")
    print()

    # Save
    out_path = ROOT / "logs" / "research" / "vol_straddle_r26b_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "commit": commit, "date": now.isoformat(),
        "runtime_s": round(elapsed, 1),
        "best_pf": round(s_best["PF"], 4) if s_best else 0,
        "best_t": round(s_best["t"], 2) if s_best else 0,
        "best_n": s_best["N"] if s_best else 0,
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  💾 {out_path}")


if __name__ == "__main__":
    main()
