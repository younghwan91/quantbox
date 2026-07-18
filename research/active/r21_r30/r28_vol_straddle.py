#!/usr/bin/env python3
"""
R28: Multi-Angle Edge Discovery (Exhaustive)
=============================================
After R27 showed VR squeeze doesn't predict bigger moves but R26b proved
PF=1.164 from reduced whipsaw, we explore every remaining angle:

  A. 4H Timeframe Straddle — coarser bars, naturally less whipsaw
  B. Volume-Confirmed Squeeze — VR + low volume = stronger compression
  C. Taker Buy Directional — order flow → predict direction (2-leg = 0.08% cost)
  D. Momentum Directional — drift during squeeze → predict breakout direction
  E. Breakout After Squeeze — price confirmation before entry (2-leg)
  F. Squeeze-Exit Timing — enter when leaving squeeze (vol expanding), not entering
  G. Direction Predictability Audit — measure accuracy of each predictor

Core math: With SL=1% / TP=4% (4:1 RR), breakeven directional accuracy = 21.6%.
Cost: straddle = 4 × 0.04% = 0.16%, directional = 2 × 0.04% = 0.08%.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# ── Constants ────────────────────────────────────────────────────────────────
COST_STRADDLE = 0.0004 * 4  # 0.16%
COST_DIRECTIONAL = 0.0004 * 2  # 0.08%
WARMUP = 200


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Indicators
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
def rolling_taker_ratio(taker_buy: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    n = len(taker_buy)
    out = np.full(n, 0.5)
    for i in range(period - 1, n):
        tb = 0.0
        tv = 0.0
        for k in range(period):
            tb += taker_buy[i - period + 1 + k]
            tv += volume[i - period + 1 + k]
        if tv > 0:
            out[i] = tb / tv
    return out


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


@numba.jit(nopython=True, cache=True)
def compute_momentum(close: np.ndarray, lookback: int) -> np.ndarray:
    """Return signed momentum: close[i] / close[i-lookback] - 1."""
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
def compute_range_hl(high: np.ndarray, low: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling highest high and lowest low over lookback bars (excluding current)."""
    n = len(high)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(lookback, n):
        mx = high[i - lookback]
        mn = low[i - lookback]
        for k in range(i - lookback + 1, i):  # exclude current bar
            if high[k] > mx:
                mx = high[k]
            if low[k] < mn:
                mn = low[k]
        hh[i] = mx
        ll[i] = mn
    return hh, ll


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Signal generators
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
def gen_vr_volume_signal(
    vr: np.ndarray, vol_pct: np.ndarray,
    vr_thresh: float, vol_pct_thresh: float, warmup: int,
) -> np.ndarray:
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if not np.isnan(vr[i]) and vr[i] < vr_thresh and vol_pct[i] < vol_pct_thresh:
            sig[i] = 1
    return sig


@numba.jit(nopython=True, cache=True)
def gen_taker_directional(
    vr: np.ndarray, taker_ratio: np.ndarray,
    vr_thresh: float, taker_thresh: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """VR squeeze + taker buy ratio → directional signal.
    Returns (signal, direction) where direction: +1 LONG, -1 SHORT."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    dire = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if np.isnan(vr[i]) or vr[i] >= vr_thresh:
            continue
        tr = taker_ratio[i]
        if tr > taker_thresh:
            sig[i] = 1
            dire[i] = 1  # LONG — more aggressive buying
        elif tr < (1.0 - taker_thresh):
            sig[i] = 1
            dire[i] = -1  # SHORT — more aggressive selling
    return sig, dire


@numba.jit(nopython=True, cache=True)
def gen_momentum_directional(
    vr: np.ndarray, momentum: np.ndarray,
    vr_thresh: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """VR squeeze + momentum direction → directional signal."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    dire = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if np.isnan(vr[i]) or vr[i] >= vr_thresh:
            continue
        if momentum[i] > 0.0001:
            sig[i] = 1
            dire[i] = 1  # LONG — drifting up
        elif momentum[i] < -0.0001:
            sig[i] = 1
            dire[i] = -1  # SHORT — drifting down
    return sig, dire


@numba.jit(nopython=True, cache=True)
def gen_breakout_signal(
    close: np.ndarray, vr: np.ndarray,
    hh: np.ndarray, ll: np.ndarray,
    vr_thresh: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """VR squeeze + range breakout → directional signal.
    1 bar lag: breakout on bar i means close[i] > hh of previous bars."""
    n = len(close)
    sig = np.zeros(n, dtype=np.int8)
    dire = np.zeros(n, dtype=np.int8)
    for i in range(warmup, n):
        if np.isnan(vr[i]) or vr[i] >= vr_thresh:
            continue
        if np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        if close[i] > hh[i]:
            sig[i] = 1
            dire[i] = 1
        elif close[i] < ll[i]:
            sig[i] = 1
            dire[i] = -1
    return sig, dire


@numba.jit(nopython=True, cache=True)
def gen_squeeze_exit_signal(
    vr: np.ndarray, close: np.ndarray,
    threshold: float, min_dur: int, warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Signal fires when VR LEAVES squeeze (was below, now above).
    Direction from close change during squeeze."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    dire = np.zeros(n, dtype=np.int8)
    in_sq = False
    sq_start = 0
    sq_count = 0

    for i in range(warmup, n):
        if np.isnan(vr[i]):
            in_sq = False
            sq_count = 0
            continue
        if vr[i] < threshold:
            if not in_sq:
                sq_start = i
                in_sq = True
            sq_count += 1
        else:
            if in_sq and sq_count >= min_dur:
                sig[i] = 1
                # Direction from price drift during squeeze
                if close[i] > close[sq_start]:
                    dire[i] = 1
                else:
                    dire[i] = -1
            in_sq = False
            sq_count = 0
    return sig, dire


# ══════════════════════════════════════════════════════════════════════════════
#  NUMBA: Simulations
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
            o = 1  # LONG TP
        elif se >= tp * 0.9:
            o = 2  # SHORT TP
        elif le <= -sl * 0.9 and se <= -sl * 0.9:
            o = 0  # Whipsaw
        else:
            o = 3  # Timeout/other
        if cnt < cap:
            pnl[cnt] = net
            oc[cnt] = o
            cnt += 1
        i = eb + 1
    return pnl[:cnt], oc[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate directional trades. Returns (pnl, outcome).
    outcome: 0=SL, 1=TP, 2=timeout"""
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    oc_arr = np.empty(cap, dtype=np.int64)
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
        outcome = 2  # timeout
        for j in range(i + 1, end):
            if d == 1:  # LONG
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    outcome = 0
                    eb = j
                    break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    outcome = 1
                    eb = j
                    break
            else:  # SHORT
                if high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    outcome = 0
                    eb = j
                    break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    outcome = 1
                    eb = j
                    break
        if outcome == 2:
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry
        if cnt < cap:
            pnl_arr[cnt] = trade_pnl - cost
            oc_arr[cnt] = outcome
            cnt += 1
        i = eb + 1
    return pnl_arr[:cnt], oc_arr[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def _load_data(timeframe: str) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings
    settings = QuantBoxSettings()
    whitelist = settings.rb_symbol_whitelist.split(",")
    data: dict = {}
    for sym in whitelist:
        candidates = list(cache_dir.glob(f"{sym}_{timeframe}_*.npz"))
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
        d: dict = {"close": c, "high": h, "low": lo, "volume": v, "bars": len(c)}
        if "taker_buy" in npz.files:
            d["taker_buy"] = npz["taker_buy"].astype(np.float64)
        data[sym] = d
    return data


def load_15m() -> dict:
    return _load_data("15m")


def load_4h() -> dict:
    return _load_data("4h")


# ══════════════════════════════════════════════════════════════════════════════
#  Stats helper
# ══════════════════════════════════════════════════════════════════════════════


def pf_stats(pnl: np.ndarray, oc: np.ndarray | None = None) -> dict | None:
    n = len(pnl)
    if n < 5:
        return None
    pos = pnl[pnl > 0]
    neg = pnl[pnl < 0]
    gp = float(np.sum(pos)) if len(pos) > 0 else 0.0
    gn = float(np.abs(np.sum(neg))) if len(neg) > 0 else 1e-9
    pf = gp / gn
    ev = float(np.mean(pnl)) * 10_000
    se = float(np.std(pnl) / np.sqrt(n)) * 10_000 if n > 1 else 999.0
    t_stat = ev / se if se > 0 else 0.0
    win_rate = float(np.mean(pnl > 0)) * 100
    result: dict = {"N": n, "PF": pf, "EV_bp": ev, "t": t_stat, "Win%": win_rate}
    if oc is not None and len(oc) == n:
        result["TP%"] = float(np.mean((oc == 1) | (oc == 2))) * 100 if oc.max() > 0 else 0.0
        result["SL%"] = float(np.mean(oc == 0)) * 100
    return result


def fmt_row(label: str, s: dict | None, width: int = 25) -> str:
    if s is None:
        return f"  {label:<{width}}    N/A"
    m = "✅" if s["PF"] >= 1.0 else "  "
    return (
        f"  {m}{label:<{width-2}} {s['N']:>5}  "
        f"PF={s['PF']:.3f}  EV={s['EV_bp']:>+6.1f}bp  "
        f"t={s['t']:>5.2f}  Win={s['Win%']:>4.1f}%"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Precompute indicators
# ══════════════════════════════════════════════════════════════════════════════


def precompute_15m(data: dict) -> None:
    """Add precomputed indicators to each symbol dict."""
    for sym, d in data.items():
        c, h, lo, vol = d["close"], d["high"], d["low"], d["volume"]
        d["vr"] = compute_vr(c, h, lo, 7, 50)
        d["vol_pct"] = vol_percentile(vol, 200)
        if "taker_buy" in d:
            d["taker_16"] = rolling_taker_ratio(d["taker_buy"], vol, 16)
            d["taker_32"] = rolling_taker_ratio(d["taker_buy"], vol, 32)
            d["taker_48"] = rolling_taker_ratio(d["taker_buy"], vol, 48)
        d["mom_8"] = compute_momentum(c, 8)
        d["mom_16"] = compute_momentum(c, 16)
        d["mom_32"] = compute_momentum(c, 32)
        d["mom_64"] = compute_momentum(c, 64)
        # Range channels for breakout
        d["hh_24"], d["ll_24"] = compute_range_hl(h, lo, 24)
        d["hh_48"], d["ll_48"] = compute_range_hl(h, lo, 48)


def precompute_4h(data: dict) -> None:
    for sym, d in data.items():
        c, h, lo = d["close"], d["high"], d["low"]
        d["vr"] = compute_vr(c, h, lo, 7, 50)


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-symbol aggregation helpers
# ══════════════════════════════════════════════════════════════════════════════


def aggregate_straddle(data: dict, sig_fn, sl: float, tp: float, max_hold: int, cost: float) -> dict | None:
    """Run straddle across all symbols, aggregate, return stats."""
    all_pnl, all_oc = [], []
    for sym, d in data.items():
        sig = sig_fn(d)
        if sig is None:
            continue
        p, o = sim_straddle(d["close"], d["high"], d["low"], sig, sl, tp, max_hold, cost)
        all_pnl.append(p)
        all_oc.append(o)
    if not all_pnl:
        return None
    return pf_stats(np.concatenate(all_pnl), np.concatenate(all_oc))


def aggregate_directional(data: dict, sig_dir_fn, sl: float, tp: float, max_hold: int, cost: float) -> dict | None:
    """Run directional across all symbols, aggregate, return stats."""
    all_pnl, all_oc = [], []
    for sym, d in data.items():
        result = sig_dir_fn(d)
        if result is None:
            continue
        sig, dire = result
        p, o = sim_directional(d["close"], d["high"], d["low"], sig, dire, sl, tp, max_hold, cost)
        all_pnl.append(p)
        all_oc.append(o)
    if not all_pnl:
        return None
    return pf_stats(np.concatenate(all_pnl), np.concatenate(all_oc))


# ══════════════════════════════════════════════════════════════════════════════
#  Direction accuracy measurement
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def measure_dominant_direction(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, max_hold: int,
) -> np.ndarray:
    """For each signal=1 bar, determine dominant direction in next max_hold bars.
    Returns array of +1 (up dominant), -1 (down dominant), 0 (unknown).
    Non-overlapping."""
    n = len(close)
    cap = n // 2 + 1
    out = np.empty(cap, dtype=np.int8)
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
        end = min(i + 1 + max_hold, n)
        mx_up = 0.0
        mx_dn = 0.0
        for j in range(i + 1, end):
            up = (high[j] - entry) / entry
            dn = (entry - low[j]) / entry
            if up > mx_up:
                mx_up = up
            if dn > mx_dn:
                mx_dn = dn
        if mx_up > mx_dn * 1.1:
            out[cnt] = 1
        elif mx_dn > mx_up * 1.1:
            out[cnt] = -1
        else:
            out[cnt] = 0
        cnt += 1
        i = end
    return out[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    now = datetime.now(timezone.utc)

    print("=" * 90)
    print("  R28: MULTI-ANGLE EDGE DISCOVERY (EXHAUSTIVE)")
    print(f"  Commit: {commit}")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("  With 4:1 RR (SL1%/TP4%), directional breakeven = 21.6%")
    print("  Cost: straddle=0.16%, directional=0.08%")
    print("=" * 90)
    print()

    t0 = time.time()

    # ── Load data ──
    print("  Loading 15m data...", end=" ", flush=True)
    data_15m = load_15m()
    bars_15m = sum(d["bars"] for d in data_15m.values())
    has_taker = sum(1 for d in data_15m.values() if "taker_buy" in d)
    print(f"{len(data_15m)} symbols, {bars_15m:,} bars ({has_taker} with taker_buy)")

    print("  Loading 4h data...", end=" ", flush=True)
    data_4h = load_4h()
    bars_4h = sum(d["bars"] for d in data_4h.values())
    print(f"{len(data_4h)} symbols, {bars_4h:,} bars")

    print("  Precomputing 15m indicators...", end=" ", flush=True)
    precompute_15m(data_15m)
    print("done")
    print("  Precomputing 4h indicators...", end=" ", flush=True)
    precompute_4h(data_4h)
    print("done\n")

    results_summary: dict = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE A: 4H TIMEFRAME STRADDLE
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE A: 4H TIMEFRAME DIRECT STRADDLE ══")
    print("  (4h bars → naturally smoother, less whipsaw)")
    print()

    exit_4h = [
        ("SL2/TP6/6d",   0.02, 0.06, 36),
        ("SL2/TP8/8d",   0.02, 0.08, 48),
        ("SL3/TP8/8d",   0.03, 0.08, 48),
        ("SL3/TP10/10d", 0.03, 0.10, 60),
        ("SL3/TP12/12d", 0.03, 0.12, 72),
        ("SL4/TP10/10d", 0.04, 0.10, 60),
        ("SL4/TP12/12d", 0.04, 0.12, 72),
        ("SL5/TP15/14d", 0.05, 0.15, 84),
    ]
    vr_thresholds_4h = [0.40, 0.50, 0.60, 0.70]

    best_a: dict | None = None
    best_a_label = ""

    for vr_t in vr_thresholds_4h:
        print(f"  VR<{vr_t:.2f}:")
        for ec_label, sl, tp, mh in exit_4h:
            def sig_fn(d, _vr_t=vr_t):
                return gen_vr_signal(d["vr"], _vr_t, 60)

            s = aggregate_straddle(data_4h, sig_fn, sl, tp, mh, COST_STRADDLE)
            label = f"VR<{vr_t} {ec_label}"
            print(fmt_row(label, s, width=28))
            if s and (best_a is None or s["PF"] > best_a["PF"]):
                best_a = s
                best_a_label = label
        print()

    # Random baseline for 4h
    def rand_4h_sig(d):
        n = len(d["close"])
        sig = np.zeros(n, dtype=np.int8)
        for i in range(60, n, 24):
            sig[i] = 1
        return sig

    s_rand = aggregate_straddle(data_4h, rand_4h_sig, 0.03, 0.10, 60, COST_STRADDLE)
    print(fmt_row("RANDOM (4h)", s_rand, width=28))
    print()
    if best_a:
        results_summary["A_4h_straddle"] = {"label": best_a_label, **best_a}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE B: VOLUME-CONFIRMED SQUEEZE (15m)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE B: VOLUME-CONFIRMED SQUEEZE (15m) ══")
    print("  (VR squeeze + low volume = deeper compression)")
    print()

    vol_pct_thresholds = [10.0, 20.0, 30.0, 50.0]
    exit_15m = [
        ("SL1/TP4/48h",   0.01, 0.04, 192),
        ("SL1/TP5/48h",   0.01, 0.05, 192),
        ("SL1.5/TP5/96h", 0.015, 0.05, 384),
    ]
    best_b: dict | None = None
    best_b_label = ""

    for vr_t in [0.45, 0.50, 0.55]:
        for vp_t in vol_pct_thresholds:
            for ec_label, sl, tp, mh in exit_15m:
                def sig_fn(d, _vr_t=vr_t, _vp_t=vp_t):
                    return gen_vr_volume_signal(d["vr"], d["vol_pct"], _vr_t, _vp_t, WARMUP)

                s = aggregate_straddle(data_15m, sig_fn, sl, tp, mh, COST_STRADDLE)
                label = f"VR<{vr_t}+Vol<{vp_t:.0f}% {ec_label}"
                if s and s["N"] >= 20:
                    print(fmt_row(label, s, width=35))
                    if best_b is None or s["PF"] > best_b["PF"]:
                        best_b = s
                        best_b_label = label
    print()
    if best_b:
        results_summary["B_volume_confirmed"] = {"label": best_b_label, **best_b}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE C: TAKER BUY DIRECTIONAL (15m)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE C: TAKER BUY DIRECTIONAL (15m) ══")
    print("  (VR squeeze + taker ratio skew → LONG or SHORT only, 2-leg cost)")
    print()

    taker_thresholds = [0.52, 0.54, 0.56, 0.58, 0.60]
    taker_periods = [16, 32, 48]
    exit_dir = [
        ("SL1/TP3/48h",   0.01, 0.03, 192),
        ("SL1/TP4/48h",   0.01, 0.04, 192),
        ("SL1/TP5/48h",   0.01, 0.05, 192),
        ("SL1.5/TP5/96h", 0.015, 0.05, 384),
        ("SL2/TP6/96h",   0.02, 0.06, 384),
    ]
    best_c: dict | None = None
    best_c_label = ""

    for vr_t in [0.45, 0.50, 0.55]:
        for tp_name in taker_periods:
            for tk_t in taker_thresholds:
                for ec_label, sl, tp, mh in exit_dir:
                    tr_key = f"taker_{tp_name}"

                    def sig_dir_fn(d, _vr_t=vr_t, _tk_t=tk_t, _tr_key=tr_key):
                        if _tr_key not in d:
                            return None
                        return gen_taker_directional(d["vr"], d[_tr_key], _vr_t, _tk_t, WARMUP)

                    s = aggregate_directional(data_15m, sig_dir_fn, sl, tp, mh, COST_DIRECTIONAL)
                    label = f"VR<{vr_t}+TR{tp_name}>{tk_t} {ec_label}"
                    if s and s["N"] >= 20:
                        if s["PF"] >= 0.95:  # only print promising ones
                            print(fmt_row(label, s, width=38))
                        if best_c is None or s["PF"] > best_c["PF"]:
                            best_c = s
                            best_c_label = label
    if best_c:
        print(f"\n  ★ Best C: {best_c_label}")
    print()
    if best_c:
        results_summary["C_taker_directional"] = {"label": best_c_label, **best_c}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE D: MOMENTUM DIRECTIONAL (15m)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE D: MOMENTUM DIRECTIONAL (15m) ══")
    print("  (VR squeeze + price drift direction → LONG or SHORT, 2-leg cost)")
    print()

    mom_lookbacks = [8, 16, 32, 64]
    best_d: dict | None = None
    best_d_label = ""

    for vr_t in [0.45, 0.50, 0.55]:
        for mlb in mom_lookbacks:
            mom_key = f"mom_{mlb}"
            for ec_label, sl, tp, mh in exit_dir:
                def sig_dir_fn(d, _vr_t=vr_t, _mk=mom_key):
                    if _mk not in d:
                        return None
                    return gen_momentum_directional(d["vr"], d[_mk], _vr_t, WARMUP)

                s = aggregate_directional(data_15m, sig_dir_fn, sl, tp, mh, COST_DIRECTIONAL)
                label = f"VR<{vr_t}+Mom{mlb} {ec_label}"
                if s and s["N"] >= 20:
                    if s["PF"] >= 0.95:
                        print(fmt_row(label, s, width=35))
                    if best_d is None or s["PF"] > best_d["PF"]:
                        best_d = s
                        best_d_label = label
    if best_d:
        print(f"\n  ★ Best D: {best_d_label}")
    print()
    if best_d:
        results_summary["D_momentum_directional"] = {"label": best_d_label, **best_d}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE E: BREAKOUT AFTER SQUEEZE (15m)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE E: BREAKOUT AFTER SQUEEZE (15m) ══")
    print("  (VR squeeze + range breakout → directional entry, 2-leg cost)")
    print()

    range_lookbacks = [24, 48]
    best_e: dict | None = None
    best_e_label = ""

    for vr_t in [0.45, 0.50, 0.55, 0.60]:
        for rlb in range_lookbacks:
            hh_key = f"hh_{rlb}"
            ll_key = f"ll_{rlb}"
            for ec_label, sl, tp, mh in exit_dir:
                def sig_dir_fn(d, _vr_t=vr_t, _hk=hh_key, _lk=ll_key):
                    if _hk not in d:
                        return None
                    return gen_breakout_signal(d["close"], d["vr"], d[_hk], d[_lk], _vr_t, WARMUP)

                s = aggregate_directional(data_15m, sig_dir_fn, sl, tp, mh, COST_DIRECTIONAL)
                label = f"VR<{vr_t}+BO{rlb} {ec_label}"
                if s and s["N"] >= 20:
                    if s["PF"] >= 0.95:
                        print(fmt_row(label, s, width=35))
                    if best_e is None or s["PF"] > best_e["PF"]:
                        best_e = s
                        best_e_label = label
    if best_e:
        print(f"\n  ★ Best E: {best_e_label}")
    print()
    if best_e:
        results_summary["E_breakout_directional"] = {"label": best_e_label, **best_e}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE F: SQUEEZE-EXIT TIMING (15m)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE F: SQUEEZE-EXIT TIMING (15m) ══")
    print("  (Enter when VR leaves squeeze zone = vol expanding. Straddle + directional.)")
    print()

    min_durs = [3, 5, 8, 12, 20]
    best_f_strad: dict | None = None
    best_f_strad_label = ""
    best_f_dir: dict | None = None
    best_f_dir_label = ""

    for vr_t in [0.45, 0.50, 0.55]:
        for md in min_durs:
            # Straddle variant (enter when leaving squeeze)
            for ec_label, sl, tp, mh in [
                ("SL1/TP4/48h",   0.01, 0.04, 192),
                ("SL1/TP5/48h",   0.01, 0.05, 192),
                ("SL1.5/TP5/96h", 0.015, 0.05, 384),
            ]:
                def sig_fn(d, _vr_t=vr_t, _md=md):
                    sig, _ = gen_squeeze_exit_signal(d["vr"], d["close"], _vr_t, _md, WARMUP)
                    return sig

                s = aggregate_straddle(data_15m, sig_fn, sl, tp, mh, COST_STRADDLE)
                label = f"SqExit VR<{vr_t} d≥{md} STR {ec_label}"
                if s and s["N"] >= 15:
                    if s["PF"] >= 0.95:
                        print(fmt_row(label, s, width=40))
                    if best_f_strad is None or s["PF"] > best_f_strad["PF"]:
                        best_f_strad = s
                        best_f_strad_label = label

            # Directional variant (direction from drift during squeeze)
            for ec_label, sl, tp, mh in exit_dir:
                def sig_dir_fn(d, _vr_t=vr_t, _md=md):
                    return gen_squeeze_exit_signal(d["vr"], d["close"], _vr_t, _md, WARMUP)

                s = aggregate_directional(data_15m, sig_dir_fn, sl, tp, mh, COST_DIRECTIONAL)
                label = f"SqExit VR<{vr_t} d≥{md} DIR {ec_label}"
                if s and s["N"] >= 15:
                    if s["PF"] >= 0.95:
                        print(fmt_row(label, s, width=40))
                    if best_f_dir is None or s["PF"] > best_f_dir["PF"]:
                        best_f_dir = s
                        best_f_dir_label = label

    if best_f_strad:
        print(f"\n  ★ Best F (straddle): {best_f_strad_label}")
    if best_f_dir:
        print(f"  ★ Best F (directional): {best_f_dir_label}")
    print()
    if best_f_strad:
        results_summary["F_squeeze_exit_straddle"] = {"label": best_f_strad_label, **best_f_strad}
    if best_f_dir:
        results_summary["F_squeeze_exit_directional"] = {"label": best_f_dir_label, **best_f_dir}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE G: DIRECTION PREDICTABILITY AUDIT
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE G: DIRECTION PREDICTABILITY AUDIT ══")
    print("  (Can we predict which way price breaks after VR squeeze?)")
    print("  (Even 22% breakeven with 4:1 RR is enough)")
    print()

    # Collect actual dominant directions during VR<0.5 squeeze events
    all_actual_dirs: list = []
    all_taker_dirs: dict[str, list] = {f"taker_{p}": [] for p in taker_periods}
    all_mom_dirs: dict[str, list] = {f"mom_{lb}": [] for lb in mom_lookbacks}

    for sym, d in data_15m.items():
        c, h, lo = d["close"], d["high"], d["low"]
        vr_sig = gen_vr_signal(d["vr"], 0.50, WARMUP)
        actual = measure_dominant_direction(c, h, lo, vr_sig, 192)

        # We need to align: non-overlapping events
        # The measure_dominant_direction function already handles non-overlapping
        # We need predictors at the same event points
        event_indices: list[int] = []
        i = 0
        n = len(c)
        while i < n - 1:
            if vr_sig[i] == 0:
                i += 1
                continue
            event_indices.append(i)
            i += 193  # skip max_hold to match non-overlapping
        event_indices = event_indices[:len(actual)]

        for k, idx in enumerate(event_indices):
            if k >= len(actual):
                break
            act = int(actual[k])  # +1, -1, or 0
            all_actual_dirs.append(act)

            # Taker predictors
            for tp_name in taker_periods:
                tr_key = f"taker_{tp_name}"
                if tr_key in d:
                    tr = d[tr_key][idx]
                    pred = 1 if tr > 0.50 else -1
                    all_taker_dirs[tr_key].append(pred)
                else:
                    all_taker_dirs[tr_key].append(0)

            # Momentum predictors
            for mlb in mom_lookbacks:
                mk = f"mom_{mlb}"
                if mk in d:
                    mom = d[mk][idx]
                    pred = 1 if mom > 0 else -1
                    all_mom_dirs[mk].append(pred)
                else:
                    all_mom_dirs[mk].append(0)

    actual_arr = np.array(all_actual_dirs, dtype=np.int8)
    n_events = len(actual_arr)
    n_clear = int(np.sum(actual_arr != 0))
    pct_up = float(np.mean(actual_arr == 1)) * 100
    pct_dn = float(np.mean(actual_arr == -1)) * 100
    pct_flat = float(np.mean(actual_arr == 0)) * 100

    print(f"  Events (VR<0.5, non-overlapping, 48h window): {n_events}")
    print(f"  Dominant UP: {pct_up:.1f}%  |  DOWN: {pct_dn:.1f}%  |  FLAT (<10% diff): {pct_flat:.1f}%")
    print()

    # Only compare on events with clear direction
    clear_mask = actual_arr != 0
    actual_clear = actual_arr[clear_mask]

    print(f"  {'Predictor':<22} {'Accuracy':>8}  {'N_correct':>9}  {'N_clear':>7}  {'2σ':>5}  {'Verdict'}")
    print("  " + "─" * 70)

    # Random baseline: 50%
    print(f"  {'Random (baseline)':<22} {'50.0%':>8}  {n_clear//2:>9}  {n_clear:>7}  {'2.9%':>5}  baseline")

    # Test each predictor
    predictors: dict[str, np.ndarray] = {}
    for tr_key in all_taker_dirs:
        predictors[tr_key] = np.array(all_taker_dirs[tr_key], dtype=np.int8)
    for mk in all_mom_dirs:
        predictors[mk] = np.array(all_mom_dirs[mk], dtype=np.int8)

    for name, pred_arr in predictors.items():
        p_clear = pred_arr[clear_mask]
        if len(p_clear) == 0:
            continue
        correct = int(np.sum(p_clear == actual_clear))
        acc = correct / len(p_clear) * 100
        two_sigma = 2 * np.sqrt(0.25 / len(p_clear)) * 100
        verdict = "✅ SIGNIFICANT" if acc > 50 + two_sigma else ("⚠️ marginal" if acc > 50 else "❌ no edge")
        print(f"  {name:<22} {acc:>7.1f}%  {correct:>9}  {len(p_clear):>7}  {two_sigma:>4.1f}%  {verdict}")

    print()

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE H: STRADDLE ON VR SQUEEZE + DURATION (15m) — expanded sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE H: STRADDLE ON VR + DURATION (expanded sweep) ══")
    print("  (R26b found dur≥5 + VR<0.5 → PF=1.505, t=3.86. Extending sweep.)")
    print()

    dur_thresholds = [3, 5, 8, 12, 16, 20, 30]
    exit_configs_h = [
        ("SL0.5/TP2/48h",   0.005, 0.020, 192),
        ("SL0.5/TP3/48h",   0.005, 0.030, 192),
        ("SL0.75/TP3/48h",  0.0075, 0.030, 192),
        ("SL1/TP3/48h",     0.010, 0.030, 192),
        ("SL1/TP4/48h",     0.010, 0.040, 192),
        ("SL1/TP5/48h",     0.010, 0.050, 192),
        ("SL1/TP4/96h",     0.010, 0.040, 384),
        ("SL1.5/TP5/96h",   0.015, 0.050, 384),
        ("SL2/TP6/96h",     0.020, 0.060, 384),
    ]
    best_h: dict | None = None
    best_h_label = ""

    for vr_t in [0.45, 0.50, 0.55]:
        for dt in dur_thresholds:
            for ec_label, sl, tp, mh in exit_configs_h:
                def sig_fn(d, _vr_t=vr_t, _dt=dt):
                    vr_arr = d["vr"]
                    dur_arr = compute_squeeze_dur(vr_arr, _vr_t)
                    n = len(vr_arr)
                    sig = np.zeros(n, dtype=np.int8)
                    for ii in range(WARMUP, n):
                        if dur_arr[ii] >= _dt:
                            sig[ii] = 1
                    return sig

                s = aggregate_straddle(data_15m, sig_fn, sl, tp, mh, COST_STRADDLE)
                label = f"VR<{vr_t} d≥{dt} {ec_label}"
                if s and s["N"] >= 10:
                    if s["PF"] >= 1.0:
                        print(fmt_row(label, s, width=35))
                    if best_h is None or (s["PF"] > best_h["PF"] and s["N"] >= 20):
                        best_h = s
                        best_h_label = label

    if best_h:
        print(f"\n  ★ Best H: {best_h_label}")
    print()
    if best_h:
        results_summary["H_vr_duration_straddle"] = {"label": best_h_label, **best_h}

    # ══════════════════════════════════════════════════════════════════════════
    #  ANGLE I: SQUEEZE-EXIT + STRADDLE + DURATION
    # ══════════════════════════════════════════════════════════════════════════
    print("  ══ ANGLE I: SQUEEZE-EXIT + DURATION + STRADDLE ══")
    print("  (Wait for squeeze to END + minimum duration → enter straddle)")
    print()

    best_i: dict | None = None
    best_i_label = ""

    for vr_t in [0.45, 0.50, 0.55]:
        for md in [3, 5, 8, 12, 20]:
            for ec_label, sl, tp, mh in exit_configs_h:
                def sig_fn(d, _vr_t=vr_t, _md=md):
                    sig, _ = gen_squeeze_exit_signal(d["vr"], d["close"], _vr_t, _md, WARMUP)
                    return sig

                s = aggregate_straddle(data_15m, sig_fn, sl, tp, mh, COST_STRADDLE)
                label = f"SqEx VR<{vr_t} md≥{md} {ec_label}"
                if s and s["N"] >= 10:
                    if s["PF"] >= 1.0:
                        print(fmt_row(label, s, width=38))
                    if best_i is None or (s["PF"] > best_i["PF"] and s["N"] >= 15):
                        best_i = s
                        best_i_label = label

    if best_i:
        print(f"\n  ★ Best I: {best_i_label}")
    print()
    if best_i:
        results_summary["I_squeeze_exit_dur"] = {"label": best_i_label, **best_i}

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print()
    print("=" * 90)
    print("  GRAND SUMMARY")
    print("=" * 90)
    print()
    print(f"  {'Angle':<30} {'Label':<45} {'N':>5}  {'PF':>6}  {'EV':>7}  {'t':>5}")
    print("  " + "─" * 104)

    for angle, info in sorted(results_summary.items()):
        m = "✅" if info["PF"] >= 1.0 else "❌"
        print(
            f"  {m}{angle:<28} {info['label']:<45} "
            f"{info['N']:>5}  {info['PF']:>6.3f}  {info['EV_bp']:>+6.1f}  {info['t']:>5.2f}"
        )

    # Count total configs tested
    total_configs = (
        len(vr_thresholds_4h) * len(exit_4h)  # A
        + 3 * len(vol_pct_thresholds) * len(exit_15m)  # B
        + 3 * len(taker_periods) * len(taker_thresholds) * len(exit_dir)  # C
        + 3 * len(mom_lookbacks) * len(exit_dir)  # D
        + 4 * len(range_lookbacks) * len(exit_dir)  # E
        + 3 * len(min_durs) * 3  # F straddle
        + 3 * len(min_durs) * len(exit_dir)  # F directional
        + 3 * len(dur_thresholds) * len(exit_configs_h)  # H
        + 3 * 5 * len(exit_configs_h)  # I
    )

    print(f"\n  Total configs tested: {total_configs}")
    print(f"  Configs with PF ≥ 1.0: {sum(1 for v in results_summary.values() if v['PF'] >= 1.0)}")
    print(f"  RUNTIME: {elapsed:.1f}s")
    print("=" * 90)
    print()

    # Save results
    out_path = ROOT / "logs" / "research" / "vol_straddle_r28_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data: dict = {
        "commit": commit,
        "date": now.isoformat(),
        "runtime_s": round(elapsed, 1),
        "total_configs": total_configs,
        "results": {},
    }
    for k, v in results_summary.items():
        save_data["results"][k] = {key: (round(val, 4) if isinstance(val, float) else val) for key, val in v.items()}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  💾 {out_path}")


if __name__ == "__main__":
    main()
