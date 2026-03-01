#!/usr/bin/env python3
"""
R31: Multi-Timeframe Volatility Straddle A/B Test
==================================================
R30 proved: D_mom8 + S_d8 have real edge (OOS PF=2.58/1.53, 6/6 years positive).
BUT trades are rare: ~326+776 = 1,102 trades over 6yr / 50 symbols (~0.5/day).

Hypothesis: VR squeeze is a structural phenomenon that occurs INDEPENDENTLY
across timeframes. Running the same strategy on 15m, 1h, and 4h bars
independently should multiply frequency WITHOUT degrading per-trade edge.

Why this should work:
  - VR = ATR(short)/ATR(long) measures relative volatility compression
  - A squeeze on 15m (hours) is different from a squeeze on 4h (days)
  - These are genuinely independent events: a 15m squeeze can happen
    inside a 4h trend, or a 4h squeeze during active 15m trading
  - Each TF generates signals → combined = more trades

Approach:
  1. Resample 15m OHLCV → 1h OHLCV (4:1) and 4h OHLCV (16:1)
  2. Run SAME D_mom8 + S_d8 logic on each TF independently
  3. Scale max_hold to match real-time duration:
     - 15m: 192 bars = 48h (same as R30)
     - 1h:   48 bars = 48h
     - 4h:   12 bars = 48h
     And for S_d8 (96h hold):
     - 15m: 384 bars
     - 1h:   96 bars
     - 4h:   24 bars
  4. VR periods (7/50) stay the same — VR is a ratio, scale-invariant
  5. SL/TP are %-based → timeframe-invariant
  6. Compare: single-TF vs multi-TF combined

Variants:
  A: 15m only (R30 baseline)
  B: 1h only
  C: 4h only
  D: 15m + 1h combined
  E: 15m + 4h combined
  F: 1h + 4h combined
  G: 15m + 1h + 4h combined (ALL)
  H: 15m + 1h + 4h, VR periods scaled (ATR 28/200 for 1h, 112/800 for 4h)
  I: 15m + 1h + 4h, squeeze dur scaled (d≥2 for 1h, d≥1 for 4h)

For each variant, run both D_mom8 and S_d8 strategies.

**ZERO LOOKAHEAD**: Each TF computed independently from its own resampled OHLCV.
No cross-TF reference. Single-TF only.

Cost: Same as R30.
  - Directional: 2 × 0.04% = 0.08%
  - Straddle:    4 × 0.04% = 0.16%

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/mtf_vol_straddle_r31.py
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

COST_STRADDLE = 0.0004 * 4    # 0.16%
COST_DIRECTIONAL = 0.0004 * 2  # 0.08%
WARMUP_15m = 200
WARMUP_1h = 50    # 200/4
WARMUP_4h = 13    # 200/16 (round up)
N_BOOTSTRAP = 1000


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Indicators (from R30, verified zero-lookahead)
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
def compute_vr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sp: int, lp: int,
) -> np.ndarray:
    atr_s = _atr_pct(close, high, low, sp)
    atr_l = _atr_pct(close, high, low, lp)
    n = len(close)
    vr = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_s[i]) and not np.isnan(atr_l[i]) and atr_l[i] > 0:
            vr[i] = atr_s[i] / atr_l[i]
    return vr


@numba.jit(nopython=True, cache=True)
def compute_momentum(close: np.ndarray, lookback: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, 0.0)
    for i in range(lookback, n):
        if close[i - lookback] > 0:
            out[i] = close[i] / close[i - lookback] - 1.0
    return out


@numba.jit(nopython=True, cache=True)
def compute_squeeze_dur(vr: np.ndarray, threshold: float) -> np.ndarray:
    n = len(vr)
    dur = np.zeros(n, dtype=np.int64)
    cnt = 0
    for i in range(n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        dur[i] = cnt
    return dur


@numba.jit(nopython=True, cache=True)
def vol_percentile(volume: np.ndarray, lookback: int) -> np.ndarray:
    n = len(volume)
    out = np.full(n, 50.0)
    for i in range(lookback - 1, n):
        cur = volume[i]
        below = 0
        valid = 0
        for k in range(lookback):
            v = volume[i - lookback + 1 + k]
            if v > 0:
                valid += 1
                if v < cur:
                    below += 1
        if valid > 0:
            out[i] = below / valid * 100.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Signal generators (from R30)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def gen_vr_signal(vr: np.ndarray, threshold: float, warmup: int) -> np.ndarray:
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_vr_dur_signal(vr: np.ndarray, threshold: float, min_dur: int, warmup: int) -> np.ndarray:
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    cnt = 0
    for i in range(n):
        if not np.isnan(vr[i]) and vr[i] < threshold:
            cnt += 1
        else:
            cnt = 0
        if i >= warmup and cnt >= min_dur:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def make_direction_momentum(sig: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    n = len(sig)
    dire = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if sig[i] == 1:
            if momentum[i] > 0.0001:
                dire[i] = 1
            elif momentum[i] < -0.0001:
                dire[i] = -1
            else:
                dire[i] = 1
    return dire


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Simulations (from R30)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> np.ndarray:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> np.ndarray:
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
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
            cnt += 1
        i = eb + 1
    return pnl[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  OHLCV Resampling: 15m → 1h, 15m → 4h
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def resample_ohlcv(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray,
    ratio: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample OHLCV from lower TF to higher TF.
    ratio=4 for 15m→1h, ratio=16 for 15m→4h.
    Uses COMPLETED bars only (truncates remainder).
    """
    n = len(close)
    n_out = n // ratio
    c_out = np.empty(n_out, dtype=np.float64)
    h_out = np.empty(n_out, dtype=np.float64)
    l_out = np.empty(n_out, dtype=np.float64)
    v_out = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        start = i * ratio
        end = start + ratio
        # Close = last bar's close
        c_out[i] = close[end - 1]
        # High = max high in window
        mx = high[start]
        for k in range(start + 1, end):
            if high[k] > mx:
                mx = high[k]
        h_out[i] = mx
        # Low = min low in window
        mn = low[start]
        for k in range(start + 1, end):
            if low[k] < mn:
                mn = low[k]
        l_out[i] = mn
        # Volume = sum
        vs = 0.0
        for k in range(start, end):
            vs += volume[k]
        v_out[i] = vs
    return c_out, h_out, l_out, v_out


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_15m() -> dict:
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
        if len(c) < 500:
            continue
        d: dict = {
            "close": c, "high": h, "low": lo, "volume": v,
            "bars": len(c), "days": _extract_days(best),
        }
        data[sym] = d
    return data


def build_resampled(data_15m: dict, ratio: int) -> dict:
    """Resample all symbols from 15m to higher TF."""
    data_htf: dict = {}
    for sym, d in data_15m.items():
        c, h, lo, v = resample_ohlcv(d["close"], d["high"], d["low"], d["volume"], ratio)
        if len(c) < 100:
            continue
        data_htf[sym] = {
            "close": c, "high": h, "low": lo, "volume": v,
            "bars": len(c), "days": d.get("days", 0),
        }
    return data_htf


def precompute(data: dict, vr_sp: int = 7, vr_lp: int = 50,
               mom_lb: int = 8, sq_thresh: float = 0.55) -> None:
    """Precompute indicators for a dataset."""
    for sym, d in data.items():
        c, h, lo, vol = d["close"], d["high"], d["low"], d["volume"]
        d["vr"] = compute_vr(c, h, lo, vr_sp, vr_lp)
        d["mom_8"] = compute_momentum(c, mom_lb)
        d["vol_pct"] = vol_percentile(vol, min(200, len(vol) // 2))
        d["sq_dur_055"] = compute_squeeze_dur(d["vr"], sq_thresh)


# ══════════════════════════════════════════════════════════════════════════════
#  Stats (from R30)
# ══════════════════════════════════════════════════════════════════════════════


def pf_from_pnl(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    pos = pnl[pnl > 0].sum()
    neg = abs(pnl[pnl < 0].sum())
    return pos / neg if neg > 0 else 999.0


def full_stats(pnl: np.ndarray) -> dict:
    n = len(pnl)
    if n < 2:
        return {"N": n, "PF": 0.0, "EV_bp": 0.0, "t": 0.0, "Win%": 0.0}
    pf = pf_from_pnl(pnl)
    ev = float(np.mean(pnl)) * 10_000
    se = float(np.std(pnl) / np.sqrt(n)) * 10_000
    t = ev / se if se > 0 else 0.0
    win = float(np.mean(pnl > 0)) * 100
    return {"N": n, "PF": round(pf, 3), "EV_bp": round(ev, 1),
            "t": round(t, 2), "Win%": round(win, 1)}


def fmt(s: dict) -> str:
    return (f"N={s['N']:>5}  PF={s['PF']:.3f}  EV={s['EV_bp']:>+6.1f}bp  "
            f"t={s['t']:>5.2f}  Win={s['Win%']:.1f}%")


def bootstrap_pf(pnl: np.ndarray, n_boot: int = 1000) -> dict:
    if len(pnl) < 10:
        return {"P_PF_ge_1": 0.0, "CI_lo": 0.0, "CI_hi": 0.0, "median": 0.0}
    rng = np.random.default_rng(42)
    n = len(pnl)
    pfs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pfs[b] = pf_from_pnl(pnl[idx])
    return {
        "P_PF_ge_1": round(float(np.mean(pfs >= 1.0)) * 100, 1),
        "CI_lo": round(float(np.percentile(pfs, 2.5)), 3),
        "CI_hi": round(float(np.percentile(pfs, 97.5)), 3),
        "median": round(float(np.median(pfs)), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy runners (parameterized by TF)
# ══════════════════════════════════════════════════════════════════════════════


def run_d_mom8_tf(d: dict, warmup: int, max_hold: int) -> np.ndarray:
    """D_mom8: VR<0.45, momentum-8 directional, SL1%/TP5%."""
    sig = gen_vr_signal(d["vr"], 0.45, warmup)
    dire = make_direction_momentum(sig, d["mom_8"])
    return sim_directional(
        d["close"], d["high"], d["low"], sig, dire,
        0.01, 0.05, max_hold, COST_DIRECTIONAL)


def run_s_d8_tf(d: dict, warmup: int, max_hold: int, min_dur: int = 8) -> np.ndarray:
    """S_d8: VR<0.55 squeeze duration >= min_dur, straddle, SL1.5%/TP5%."""
    sig = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)
    return sim_straddle(
        d["close"], d["high"], d["low"], sig,
        0.015, 0.05, max_hold, COST_STRADDLE)


def run_strat_on_data(
    data: dict, strat: str, warmup: int, max_hold_d: int, max_hold_s: int,
    min_dur_s: int = 8,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run strategy across all symbols. Returns (per_sym, aggregated)."""
    per_sym: dict[str, np.ndarray] = {}
    for sym, d in data.items():
        if strat == "D_mom8":
            pnl = run_d_mom8_tf(d, warmup, max_hold_d)
        elif strat == "S_d8":
            pnl = run_s_d8_tf(d, warmup, max_hold_s, min_dur_s)
        else:
            continue
        if len(pnl) > 0:
            per_sym[sym] = pnl
    arrs = [v for v in per_sym.values() if len(v) > 0]
    agg = np.concatenate(arrs) if arrs else np.array([], dtype=np.float64)
    return per_sym, agg


# ══════════════════════════════════════════════════════════════════════════════
#  Per-year breakdown + OOS
# ══════════════════════════════════════════════════════════════════════════════


def per_year_pfs(data: dict, strat: str, warmup: int, max_hold_d: int,
                 max_hold_s: int, bars_per_year: int, min_dur_s: int = 8) -> list[dict]:
    """Split each symbol into yearly chunks and aggregate."""
    max_bars = max(d["bars"] for d in data.values())
    n_years = max_bars // bars_per_year + 1
    year_pnls: list[list] = [[] for _ in range(n_years)]

    for sym, d_full in data.items():
        n = len(d_full["close"])
        for yr in range(n_years):
            start = yr * bars_per_year
            end = min((yr + 1) * bars_per_year, n)
            if end - start < max(100, warmup + 50):
                continue
            d_yr: dict = {}
            for key in ("close", "high", "low", "volume"):
                d_yr[key] = d_full[key][start:end].copy()
            d_yr["bars"] = end - start
            c, h, lo, vol = d_yr["close"], d_yr["high"], d_yr["low"], d_yr["volume"]
            d_yr["vr"] = compute_vr(c, h, lo, 7, 50)
            d_yr["mom_8"] = compute_momentum(c, 8)
            d_yr["vol_pct"] = vol_percentile(vol, min(200, len(vol) // 2))
            d_yr["sq_dur_055"] = compute_squeeze_dur(d_yr["vr"], 0.55)

            if strat == "D_mom8":
                pnl = run_d_mom8_tf(d_yr, warmup, max_hold_d)
            else:
                pnl = run_s_d8_tf(d_yr, warmup, max_hold_s, min_dur_s)
            if len(pnl) > 0:
                year_pnls[yr].append(pnl)

    results = []
    for yr, arrs in enumerate(year_pnls):
        if not arrs:
            continue
        agg = np.concatenate(arrs)
        s = full_stats(agg)
        s["year"] = 2020 + yr
        results.append(s)
    return results


def oos_test(data: dict, strat: str, warmup: int, max_hold_d: int,
             max_hold_s: int, min_dur_s: int = 8) -> dict:
    """True OOS: 67/33 split, recompute indicators independently."""
    is_pnls: list = []
    oos_pnls: list = []

    for sym, d_full in data.items():
        n = len(d_full["close"])
        cut = int(n * 0.67)
        for seg_start, seg_end, bucket in [(0, cut, is_pnls), (cut, n, oos_pnls)]:
            seg_len = seg_end - seg_start
            if seg_len < max(100, warmup + 50):
                continue
            d_seg: dict = {}
            for key in ("close", "high", "low", "volume"):
                d_seg[key] = d_full[key][seg_start:seg_end].copy()
            d_seg["bars"] = seg_len
            c, h, lo, vol = d_seg["close"], d_seg["high"], d_seg["low"], d_seg["volume"]
            d_seg["vr"] = compute_vr(c, h, lo, 7, 50)
            d_seg["mom_8"] = compute_momentum(c, 8)
            d_seg["vol_pct"] = vol_percentile(vol, min(200, seg_len // 2))
            d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)

            if strat == "D_mom8":
                pnl = run_d_mom8_tf(d_seg, warmup, max_hold_d)
            else:
                pnl = run_s_d8_tf(d_seg, warmup, max_hold_s, min_dur_s)
            if len(pnl) > 0:
                bucket.append(pnl)

    agg_is = np.concatenate(is_pnls) if is_pnls else np.empty(0)
    agg_oos = np.concatenate(oos_pnls) if oos_pnls else np.empty(0)
    return {"IS": full_stats(agg_is), "OOS": full_stats(agg_oos),
            "OOS_bootstrap": bootstrap_pf(agg_oos)}


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════


VARIANTS = {
    # ── Single Timeframe ──
    "A-15m": {
        "desc": "15m only (R30 baseline)",
        "tfs": [("15m", 1, WARMUP_15m, 192, 384, 8)],
        # tfs: list of (label, resample_ratio, warmup, max_hold_D, max_hold_S, min_dur_S)
        # resample_ratio: 1=native 15m, 4=1h, 16=4h
    },
    "B-1h": {
        "desc": "1h only",
        "tfs": [("1h", 4, WARMUP_1h, 48, 96, 8)],
    },
    "C-4h": {
        "desc": "4h only",
        "tfs": [("4h", 16, WARMUP_4h, 12, 24, 8)],
    },
    # ── Dual Timeframe ──
    "D-15m+1h": {
        "desc": "15m + 1h combined",
        "tfs": [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("1h", 4, WARMUP_1h, 48, 96, 8),
        ],
    },
    "E-15m+4h": {
        "desc": "15m + 4h combined",
        "tfs": [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ],
    },
    "F-1h+4h": {
        "desc": "1h + 4h combined",
        "tfs": [
            ("1h", 4, WARMUP_1h, 48, 96, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ],
    },
    # ── Triple Timeframe ──
    "G-ALL": {
        "desc": "15m + 1h + 4h combined",
        "tfs": [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("1h", 4, WARMUP_1h, 48, 96, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ],
    },
    # ── Triple with scaled squeeze duration ──
    # 1h squeeze d≥8 means 8 hours. For 15m d≥8 means 2 hours.
    # To equalize real-time duration: 15m d≥32 = 1h d≥8 = 4h d≥2
    "H-ALL-DurEq": {
        "desc": "15m+1h+4h, squeeze dur equalized (8h real: 15m=32, 1h=8, 4h=2)",
        "tfs": [
            ("15m", 1, WARMUP_15m, 192, 384, 32),
            ("1h", 4, WARMUP_1h, 48, 96, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 2),
        ],
    },
    # ── Loose squeeze on higher TFs for more signals ──
    "I-ALL-LooseHTF": {
        "desc": "15m d=8 + 1h d=4 + 4h d=2 (loose higher TFs for more signals)",
        "tfs": [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("1h", 4, WARMUP_1h, 48, 96, 4),
            ("4h", 16, WARMUP_4h, 12, 24, 2),
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_variant(
    var_name: str, var_cfg: dict, data_15m: dict,
    data_1h: dict, data_4h: dict,
) -> dict:
    """Evaluate one variant for both D_mom8 and S_d8."""
    result: dict = {"name": var_name, "desc": var_cfg["desc"], "strategies": {}}

    for strat in ["D_mom8", "S_d8"]:
        all_pnls: list = []

        tf_details: list = []
        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in var_cfg["tfs"]:
            # Select data
            if ratio == 1:
                data_tf = data_15m
                bpy = 365 * 24 * 4   # bars per year for 15m
            elif ratio == 4:
                data_tf = data_1h
                bpy = 365 * 24       # bars per year for 1h
            else:
                data_tf = data_4h
                bpy = 365 * 6        # bars per year for 4h

            mh = mh_d if strat == "D_mom8" else mh_s
            md = 8 if strat == "D_mom8" else min_dur  # D_mom8 doesn't use min_dur

            _, pnl = run_strat_on_data(data_tf, strat, warmup, mh_d, mh_s, md)

            tf_stats = full_stats(pnl)
            tf_stats["tf"] = tf_label
            tf_details.append(tf_stats)

            if len(pnl) > 0:
                all_pnls.append(pnl)

        # Combine all TF PnLs
        combined = np.concatenate(all_pnls) if all_pnls else np.empty(0)
        combined_stats = full_stats(combined)
        bs = bootstrap_pf(combined)

        result["strategies"][strat] = {
            "per_tf": tf_details,
            "combined": combined_stats,
            "bootstrap": bs,
        }

    return result


def oos_variant(
    var_name: str, var_cfg: dict, data_15m: dict,
    data_1h: dict, data_4h: dict,
) -> dict:
    """OOS test for a variant."""
    result: dict = {}

    for strat in ["D_mom8", "S_d8"]:
        all_is: list = []
        all_oos: list = []

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in var_cfg["tfs"]:
            if ratio == 1:
                data_tf = data_15m
            elif ratio == 4:
                data_tf = data_1h
            else:
                data_tf = data_4h

            md = 8 if strat == "D_mom8" else min_dur
            r = oos_test(data_tf, strat, warmup, mh_d, mh_s, md)

            # Recover raw PnLs for aggregation (re-run IS/OOS separately)
            for sym, d_full in data_tf.items():
                n = len(d_full["close"])
                cut = int(n * 0.67)
                for seg_start, seg_end, bucket in [(0, cut, all_is), (cut, n, all_oos)]:
                    seg_len = seg_end - seg_start
                    if seg_len < max(100, warmup + 50):
                        continue
                    d_seg: dict = {}
                    for key in ("close", "high", "low", "volume"):
                        d_seg[key] = d_full[key][seg_start:seg_end].copy()
                    d_seg["bars"] = seg_len
                    cc, hh, ll, vv = d_seg["close"], d_seg["high"], d_seg["low"], d_seg["volume"]
                    d_seg["vr"] = compute_vr(cc, hh, ll, 7, 50)
                    d_seg["mom_8"] = compute_momentum(cc, 8)
                    d_seg["vol_pct"] = vol_percentile(vv, min(200, seg_len // 2))
                    d_seg["sq_dur_055"] = compute_squeeze_dur(d_seg["vr"], 0.55)
                    if strat == "D_mom8":
                        pnl = run_d_mom8_tf(d_seg, warmup, mh_d)
                    else:
                        pnl = run_s_d8_tf(d_seg, warmup, mh_s, md)
                    if len(pnl) > 0:
                        bucket.append(pnl)

        agg_is = np.concatenate(all_is) if all_is else np.empty(0)
        agg_oos = np.concatenate(all_oos) if all_oos else np.empty(0)
        result[strat] = {
            "IS": full_stats(agg_is),
            "OOS": full_stats(agg_oos),
            "OOS_bootstrap": bootstrap_pf(agg_oos),
        }

    return result


def yearly_variant(
    var_name: str, var_cfg: dict, data_15m: dict,
    data_1h: dict, data_4h: dict,
) -> dict:
    """Per-year breakdown for a variant."""
    result: dict = {}

    for strat in ["D_mom8", "S_d8"]:
        # Aggregate per-year PnLs across TFs
        max_years = 7  # 2020-2026
        year_pnls: list[list] = [[] for _ in range(max_years)]

        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in var_cfg["tfs"]:
            if ratio == 1:
                data_tf = data_15m
                bpy = 365 * 24 * 4
            elif ratio == 4:
                data_tf = data_1h
                bpy = 365 * 24
            else:
                data_tf = data_4h
                bpy = 365 * 6

            md = 8 if strat == "D_mom8" else min_dur
            yrs = per_year_pfs(data_tf, strat, warmup, mh_d, mh_s, bpy, md)
            for y in yrs:
                yr_idx = y["year"] - 2020
                if 0 <= yr_idx < max_years:
                    # Hack: re-run to get raw PnLs for this year
                    # But we already have the stats. Just store them.
                    pass

            result.setdefault(strat, [])

        # Simpler approach: just run per_year for each TF and aggregate stats
        # We want combined N and PF per year across TFs
        for tf_label, ratio, warmup, mh_d, mh_s, min_dur in var_cfg["tfs"]:
            if ratio == 1:
                data_tf = data_15m
                bpy = 365 * 24 * 4
            elif ratio == 4:
                data_tf = data_1h
                bpy = 365 * 24
            else:
                data_tf = data_4h
                bpy = 365 * 6

            md = 8 if strat == "D_mom8" else min_dur

            max_bars = max(d["bars"] for d in data_tf.values())
            n_years_tf = max_bars // bpy + 1

            for yr in range(min(n_years_tf, max_years)):
                yr_arrs: list = []
                for sym, d_full in data_tf.items():
                    n = len(d_full["close"])
                    start = yr * bpy
                    end_b = min((yr + 1) * bpy, n)
                    if end_b - start < max(100, warmup + 50):
                        continue
                    d_yr: dict = {}
                    for key in ("close", "high", "low", "volume"):
                        d_yr[key] = d_full[key][start:end_b].copy()
                    d_yr["bars"] = end_b - start
                    cc, hh, ll, vv = d_yr["close"], d_yr["high"], d_yr["low"], d_yr["volume"]
                    d_yr["vr"] = compute_vr(cc, hh, ll, 7, 50)
                    d_yr["mom_8"] = compute_momentum(cc, 8)
                    d_yr["vol_pct"] = vol_percentile(vv, min(200, (end_b - start) // 2))
                    d_yr["sq_dur_055"] = compute_squeeze_dur(d_yr["vr"], 0.55)
                    if strat == "D_mom8":
                        pnl = run_d_mom8_tf(d_yr, warmup, mh_d)
                    else:
                        pnl = run_s_d8_tf(d_yr, warmup, mh_s, md)
                    if len(pnl) > 0:
                        yr_arrs.append(pnl)

                if yr_arrs:
                    agg_yr = np.concatenate(yr_arrs)
                    s_yr = full_stats(agg_yr)
                    s_yr["year"] = 2020 + yr
                    year_pnls[yr].append(agg_yr)

        # Now combine year_pnls across TFs
        yearly_results: list = []
        for yr in range(max_years):
            if year_pnls[yr]:
                agg = np.concatenate(year_pnls[yr])
                s = full_stats(agg)
                s["year"] = 2020 + yr
                yearly_results.append(s)

        result[strat] = yearly_results

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    commit = get_commit_short()

    print("=" * 80)
    print("  R31: MULTI-TIMEFRAME VOLATILITY STRADDLE A/B TEST")
    print(f"  Commit: {commit}  |  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Goal: Increase trade frequency via independent multi-TF signals")
    print("=" * 80)

    # ── Load base 15m data ──
    print("\n  Loading 15m OHLCV data...")
    data_15m = load_15m()
    total_15m = sum(d["bars"] for d in data_15m.values())
    print(f"  Loaded {len(data_15m)} symbols, {total_15m:,} total 15m bars")

    # ── Resample to 1h and 4h ──
    print("  Resampling 15m → 1h (4:1)...")
    data_1h = build_resampled(data_15m, 4)
    total_1h = sum(d["bars"] for d in data_1h.values())
    print(f"  1h: {len(data_1h)} symbols, {total_1h:,} bars")

    print("  Resampling 15m → 4h (16:1)...")
    data_4h = build_resampled(data_15m, 16)
    total_4h = sum(d["bars"] for d in data_4h.values())
    print(f"  4h: {len(data_4h)} symbols, {total_4h:,} bars")

    # ── Precompute indicators ──
    print("  Precomputing indicators...")
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)
    print("  Done.")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Full-sample comparison (all variants)
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("  PHASE 1: FULL-SAMPLE A/B COMPARISON")
    print("=" * 80)

    all_results: dict = {}
    for var_name in sorted(VARIANTS.keys()):
        var_cfg = VARIANTS[var_name]
        print(f"\n  {'─' * 74}")
        print(f"  {var_name}: {var_cfg['desc']}")
        print(f"  {'─' * 74}")

        r = evaluate_variant(var_name, var_cfg, data_15m, data_1h, data_4h)
        all_results[var_name] = r

        for strat in ["D_mom8", "S_d8"]:
            sr = r["strategies"][strat]
            # Per-TF details
            for tf_s in sr["per_tf"]:
                print(f"    {strat} [{tf_s['tf']:>3s}]: {fmt(tf_s)}")
            # Combined
            cs = sr["combined"]
            bs = sr["bootstrap"]
            print(f"    {strat} [CMB]: {fmt(cs)}  "
                  f"P(PF≥1)={bs['P_PF_ge_1']:.0f}%  "
                  f"CI=[{bs['CI_lo']:.3f},{bs['CI_hi']:.3f}]")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: Summary comparison table
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("  PHASE 2: SUMMARY COMPARISON TABLE")
    print("=" * 80)

    header = f"  {'Variant':<16} {'Strat':<8} {'N':>6} {'PF':>7} {'EV(bp)':>8} {'t':>6} {'Win%':>6} {'P(PF≥1)':>8} {'CI_lo':>7} {'CI_hi':>7}"
    print(header)
    print("  " + "─" * len(header))

    for var_name in sorted(all_results.keys()):
        r = all_results[var_name]
        for strat in ["D_mom8", "S_d8"]:
            sr = r["strategies"][strat]
            cs = sr["combined"]
            bs = sr["bootstrap"]
            line = (f"  {var_name:<16} {strat:<8} {cs['N']:>6} {cs['PF']:>7.3f} "
                    f"{cs['EV_bp']:>+8.1f} {cs['t']:>6.2f} {cs['Win%']:>6.1f} "
                    f"{bs['P_PF_ge_1']:>7.0f}% {bs['CI_lo']:>7.3f} {bs['CI_hi']:>7.3f}")
            print(line)

    # Frequency comparison
    print("\n  FREQUENCY ANALYSIS (trades/year equivalent):")
    print(f"  {'Variant':<16} {'D_mom8 N':>10} {'S_d8 N':>10} {'Total N':>10} {'Trades/day':>12}")
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    for var_name in sorted(all_results.keys()):
        r = all_results[var_name]
        n_d = r["strategies"]["D_mom8"]["combined"]["N"]
        n_s = r["strategies"]["S_d8"]["combined"]["N"]
        total_n = n_d + n_s
        tpd = total_n / (data_years * 365) if data_years > 0 else 0
        print(f"  {var_name:<16} {n_d:>10} {n_s:>10} {total_n:>10} {tpd:>12.2f}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: OOS + Per-Year for top variants
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("  PHASE 3: OOS + PER-YEAR BREAKDOWN (Top Variants)")
    print("=" * 80)

    # Pick top variants: A (baseline), plus best multi-TF by combined PF
    top_vars = ["A-15m"]  # always include baseline
    # Find best multi-TF
    best_mtf = None
    best_mtf_pf = 0
    for var_name in sorted(all_results.keys()):
        if var_name == "A-15m":
            continue
        r = all_results[var_name]
        # Use D_mom8+S_d8 combined PF as ranking metric
        d_pf = r["strategies"]["D_mom8"]["combined"]["PF"]
        s_pf = r["strategies"]["S_d8"]["combined"]["PF"]
        avg = (d_pf + s_pf) / 2
        if avg > best_mtf_pf:
            best_mtf_pf = avg
            best_mtf = var_name
    if best_mtf and best_mtf not in top_vars:
        top_vars.append(best_mtf)
    # Always include G-ALL
    if "G-ALL" not in top_vars:
        top_vars.append("G-ALL")
    # Include best between D/E/F
    for v in ["D-15m+1h", "E-15m+4h", "F-1h+4h"]:
        if v not in top_vars:
            top_vars.append(v)
            break

    for var_name in top_vars:
        var_cfg = VARIANTS[var_name]
        print(f"\n  {'═' * 74}")
        print(f"  {var_name}: {var_cfg['desc']}")
        print(f"  {'═' * 74}")

        # OOS test
        print(f"\n  [OOS TEST] 67/33 split:")
        oos_r = oos_variant(var_name, var_cfg, data_15m, data_1h, data_4h)
        for strat in ["D_mom8", "S_d8"]:
            r = oos_r[strat]
            print(f"    {strat} IS:  {fmt(r['IS'])}")
            print(f"    {strat} OOS: {fmt(r['OOS'])}  "
                  f"P(PF≥1)={r['OOS_bootstrap']['P_PF_ge_1']:.0f}%  "
                  f"CI=[{r['OOS_bootstrap']['CI_lo']:.3f},{r['OOS_bootstrap']['CI_hi']:.3f}]")
            if r['IS']['PF'] > 0:
                deg = (r['OOS']['PF'] - r['IS']['PF']) / r['IS']['PF'] * 100
                mark = "✅" if r['OOS']['PF'] >= 1.0 and deg > -30 else "⚠️"
                print(f"    {strat} IS→OOS: {deg:+.1f}%  {mark}")

        # Per-year
        print(f"\n  [PER-YEAR BREAKDOWN]:")
        yr_r = yearly_variant(var_name, var_cfg, data_15m, data_1h, data_4h)
        for strat in ["D_mom8", "S_d8"]:
            yrs = yr_r.get(strat, [])
            n_pos = sum(1 for y in yrs if y["PF"] >= 1.0)
            print(f"    {strat}: {n_pos}/{len(yrs)} profitable years")
            for y in yrs:
                mark = "✅" if y["PF"] >= 1.0 else "❌"
                print(f"      {y['year']}: N={y['N']:>4}  PF={y['PF']:.3f}  "
                      f"EV={y['EV_bp']:>+6.1f}bp  t={y['t']:>5.2f}  {mark}")

    # ══════════════════════════════════════════════════════════════════════
    #  GRAND SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n\n" + "=" * 80)
    print("  GRAND SUMMARY")
    print("=" * 80)

    print(f"\n  Baseline (A-15m): "
          f"D_mom8 PF={all_results['A-15m']['strategies']['D_mom8']['combined']['PF']:.3f} "
          f"N={all_results['A-15m']['strategies']['D_mom8']['combined']['N']}, "
          f"S_d8 PF={all_results['A-15m']['strategies']['S_d8']['combined']['PF']:.3f} "
          f"N={all_results['A-15m']['strategies']['S_d8']['combined']['N']}")

    print(f"\n  Best multi-TF: {best_mtf}")
    if best_mtf:
        bm = all_results[best_mtf]
        print(f"    D_mom8 PF={bm['strategies']['D_mom8']['combined']['PF']:.3f} "
              f"N={bm['strategies']['D_mom8']['combined']['N']}, "
              f"S_d8 PF={bm['strategies']['S_d8']['combined']['PF']:.3f} "
              f"N={bm['strategies']['S_d8']['combined']['N']}")

    # Delta N
    n_base = (all_results["A-15m"]["strategies"]["D_mom8"]["combined"]["N"] +
              all_results["A-15m"]["strategies"]["S_d8"]["combined"]["N"])
    if best_mtf:
        n_best = (all_results[best_mtf]["strategies"]["D_mom8"]["combined"]["N"] +
                  all_results[best_mtf]["strategies"]["S_d8"]["combined"]["N"])
        print(f"\n  Frequency increase: {n_base} → {n_best} ({(n_best/n_base - 1)*100:+.1f}%)")

    # G-ALL summary
    g = all_results.get("G-ALL")
    if g:
        n_g = (g["strategies"]["D_mom8"]["combined"]["N"] +
               g["strategies"]["S_d8"]["combined"]["N"])
        tpd_g = n_g / (data_years * 365) if data_years > 0 else 0
        print(f"\n  G-ALL (15m+1h+4h):")
        print(f"    Total trades: {n_g} ({tpd_g:.2f}/day)")
        print(f"    D_mom8 PF={g['strategies']['D_mom8']['combined']['PF']:.3f}")
        print(f"    S_d8   PF={g['strategies']['S_d8']['combined']['PF']:.3f}")

    print(f"\n  RUNTIME: {elapsed:.1f}s")

    # ── Save JSON ──
    out_dir = ROOT / "logs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mtf_vol_straddle_r31_results.json"

    save_data: dict = {
        "commit": commit,
        "date": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(elapsed, 1),
        "variants": {},
    }
    for var_name in sorted(all_results.keys()):
        r = all_results[var_name]
        save_data["variants"][var_name] = {
            "desc": r["desc"],
            "strategies": r["strategies"],
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  💾 {out_path}")


if __name__ == "__main__":
    main()
