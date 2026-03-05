"""R44b — High-Volatility Complement: Momentum/Trend-Following

R44 finding: ALL mean reversion vectors fail in crypto (PF < 1.0 universally).
Crypto is momentum-driven. High vol = trend continuation, not exhaustion.

Pivot: FOLLOW momentum during high-vol instead of fading it.

Strategy vectors:
  V6: VR Expansion + Momentum FOLLOWING (VR > threshold, go WITH momentum)
  V7: Strong Momentum Breakout (large move = trend start, ride it)
  V8: Multi-TF Trend Alignment (all TFs same direction during expansion)
  V9: VR > threshold + Squeeze Duration filter (just entered expansion)
  V10: Momentum + RSI Confirmation (strong move confirmed by RSI not extreme)

Rules: Same as R44 — real data, R38 infrastructure, baseline verification.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    compute_vr, compute_momentum,
    vol_percentile,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    COST_DIRECTIONAL, G_3TF, CFG_V13,
)
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics, get_year_boundaries,
)
from research.active.r41_r45.r44_high_vol_complement import (
    compute_rsi, _dedup_sym_trades, overlap_analysis,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  V6: VR Expansion + Momentum Following
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_vr_momentum_follow(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vr: np.ndarray, momentum: np.ndarray,
    vr_threshold: float, mom_threshold: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow momentum direction when VR signals expansion.

    Entry: VR > vr_threshold AND |momentum| > mom_threshold
    Direction: SAME as momentum (trend following)
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if np.isnan(vr[i]) or vr[i] <= vr_threshold:
            i += 1
            continue

        if abs(momentum[i]) < mom_threshold:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        # FOLLOW momentum direction
        d = 1 if momentum[i] > 0 else -1

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
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
#  V7: Strong Momentum Breakout
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_momentum_breakout(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    momentum: np.ndarray,
    mom_threshold: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ride strong momentum moves (breakout continuation).

    Entry: |momentum| > mom_threshold
    Direction: SAME as momentum
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if abs(momentum[i]) < mom_threshold:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
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
#  V8: Multi-TF Trend Alignment (uses 4h momentum confirmation)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_aligned_momentum(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    momentum: np.ndarray,
    htf_momentum: np.ndarray, ratio: int,
    mom_threshold: float, htf_mom_threshold: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enter when both current TF and higher TF momentum align.

    Entry: |momentum| > mom_threshold AND htf_momentum same direction
    Direction: shared direction
    """
    n = len(close)
    n_htf = len(htf_momentum)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if abs(momentum[i]) < mom_threshold:
            i += 1
            continue

        # Get higher TF momentum — use COMPLETED bar only (j-1)
        htf_idx = i // ratio - 1
        if htf_idx < 0 or htf_idx >= n_htf:
            i += 1
            continue

        htf_mom = htf_momentum[htf_idx]
        if abs(htf_mom) < htf_mom_threshold:
            i += 1
            continue

        # Must agree on direction
        if (momentum[i] > 0) != (htf_mom > 0):
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
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
#  V9: VR Expansion Onset (just crossed into expansion)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_vr_expansion_onset(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vr: np.ndarray, momentum: np.ndarray,
    vr_threshold: float, lookback: int,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enter when VR just crossed above threshold (expansion onset).

    Checks VR[i] > threshold AND min(VR[i-lookback:i]) < threshold
    (was recently compressed, just broke out).
    Direction: momentum following.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = max(warmup, lookback)

    while i < n - 1:
        if np.isnan(vr[i]) or vr[i] <= vr_threshold:
            i += 1
            continue

        # Check was recently below threshold
        was_below = False
        for k in range(1, lookback + 1):
            if i - k >= 0 and vr[i - k] < vr_threshold:
                was_below = True
                break
        if not was_below:
            i += 1
            continue

        if abs(momentum[i]) < 0.005:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
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
#  V10: Momentum + RSI Confirmation (not overbought/oversold)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_momentum_rsi_confirm(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    momentum: np.ndarray, rsi: np.ndarray,
    mom_threshold: float,
    rsi_low: float, rsi_high: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow momentum confirmed by RSI not being extreme.

    Entry: |momentum| > threshold AND RSI in [rsi_low, rsi_high]
    RSI filter: prevents chasing overextended moves
    Direction: SAME as momentum
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if abs(momentum[i]) < mom_threshold:
            i += 1
            continue

        # RSI must be in range (not extreme)
        r = rsi[i]
        if momentum[i] > 0:
            # Long: RSI should not be too high (overbought)
            if r > rsi_high or r < 40.0:
                i += 1
                continue
        else:
            # Short: RSI should not be too low (oversold)
            if r < rsi_low or r > 60.0:
                i += 1
                continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
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
#  Trade Collection
# ══════════════════════════════════════════════════════════════════════════════


def collect_v6(data_15m, data_1h, data_4h,
               vr_th, mom_th, sl, tp, sp=8, lp=56):
    """V6: VR expansion + momentum following."""
    vr_key = f"vr_{sp}_{lp}"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}
    all_t = []
    for sym in sorted(data_15m):
        st = []
        for _, ratio, _, mh_d, _ in G_3TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            p, e, x = sim_vr_momentum_follow(
                d["close"], d["high"], d["low"], vr, d["mom_8"],
                vr_th, mom_th, sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio])
            for k in range(len(p)):
                st.append((int(e[k]) * ratio, int(x[k]) * ratio + ratio - 1, float(p[k])))
        all_t.extend(_dedup_sym_trades(sym, st))
    return all_t


def collect_v7(data_15m, data_1h, data_4h, mom_th, sl, tp):
    """V7: Strong momentum breakout."""
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}
    all_t = []
    for sym in sorted(data_15m):
        st = []
        for _, ratio, _, mh_d, _ in G_3TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            p, e, x = sim_momentum_breakout(
                d["close"], d["high"], d["low"], d["mom_8"],
                mom_th, sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio])
            for k in range(len(p)):
                st.append((int(e[k]) * ratio, int(x[k]) * ratio + ratio - 1, float(p[k])))
        all_t.extend(_dedup_sym_trades(sym, st))
    return all_t


def collect_v8(data_15m, data_4h, mom_th, htf_th, sl, tp):
    """V8: Multi-TF aligned momentum (15m + 4h)."""
    warmup = WARMUP_15m
    max_hold = 96
    all_t = []
    for sym in sorted(data_15m):
        d_15m = data_15m.get(sym)
        d_4h = data_4h.get(sym)
        if d_15m is None or d_4h is None:
            continue
        p, e, x = sim_aligned_momentum(
            d_15m["close"], d_15m["high"], d_15m["low"],
            d_15m["mom_8"], d_4h["mom_8"], 16,
            mom_th, htf_th, sl, tp, max_hold,
            COST_DIRECTIONAL, warmup)
        for k in range(len(p)):
            all_t.append((sym, int(e[k]), int(x[k]), float(p[k])))
    return all_t


def collect_v9(data_15m, data_1h, data_4h,
               vr_th, lookback, sl, tp, sp=8, lp=56):
    """V9: VR expansion onset (recently crossed threshold)."""
    vr_key = f"vr_{sp}_{lp}"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}
    all_t = []
    for sym in sorted(data_15m):
        st = []
        for _, ratio, _, mh_d, _ in G_3TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            p, e, x = sim_vr_expansion_onset(
                d["close"], d["high"], d["low"], vr, d["mom_8"],
                vr_th, lookback, sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio])
            for k in range(len(p)):
                st.append((int(e[k]) * ratio, int(x[k]) * ratio + ratio - 1, float(p[k])))
        all_t.extend(_dedup_sym_trades(sym, st))
    return all_t


def collect_v10(data_15m, data_1h, data_4h,
                mom_th, rsi_lo, rsi_hi, sl, tp, rsi_period=14):
    """V10: Momentum + RSI confirmation."""
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}
    all_t = []
    for sym in sorted(data_15m):
        st = []
        for _, ratio, _, mh_d, _ in G_3TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            rsi = compute_rsi(d["close"], rsi_period)
            p, e, x = sim_momentum_rsi_confirm(
                d["close"], d["high"], d["low"], d["mom_8"], rsi,
                mom_th, rsi_lo, rsi_hi,
                sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio])
            for k in range(len(p)):
                st.append((int(e[k]) * ratio, int(x[k]) * ratio + ratio - 1, float(p[k])))
        all_t.extend(_dedup_sym_trades(sym, st))
    return all_t


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*100}")
    print(f"  R44b — High-Vol Complement: Momentum/Trend-Following")
    print(f"  Commit: {commit}  |  Pivot from R44 (MR failed universally)")
    print(f"{'='*100}\n")

    t0 = time.time()

    # ── Load data ──
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])

    year_bounds = get_year_boundaries(data_15m)
    SLOTS = 10

    # ── Phase 1: Baseline ──
    print(f"  Phase 1: Baseline (v13.1 VR Squeeze)")
    print(f"  {'─'*90}")

    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_all = bl_d + bl_s
    bl_pnl, bl_acc = portfolio_sim_with_details(bl_all, SLOTS)
    bl_m = compute_metrics(bl_pnl, data_years)
    bl_yr = year_metrics(bl_pnl, bl_acc, year_bounds)

    print(f"  Squeeze BL: N={bl_m['N']:>5}  PF={bl_m['PF']:.3f}  "
          f"EV={bl_m['EV_bp']:+.1f}bp  WR={bl_m['WR']:.1f}%  "
          f"Calmar={bl_m['calmar']:.1f}")

    if abs(bl_m["N"] - 2265) > 2265 * 0.01:
        print(f"  ⛔ BASELINE DRIFT: N={bl_m['N']} vs expected 2265")
        return
    print(f"  ✅ Baseline OK\n")

    # ── Phase 2: Vector Screening ──
    print(f"  Phase 2: Trend-Following Vector Screening")
    print(f"  {'─'*90}")
    header = (f"  {'Vector':<40} {'N':>6} {'PF':>6} {'EV(bp)':>8} "
              f"{'WR%':>5} {'Calmar':>7} {'V':>3}")
    print(header)
    print(f"  {'─'*85}")

    results: dict[str, tuple[list, dict]] = {}

    def screen(name, trades):
        pnl = portfolio_sim(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        v = "✅" if m["PF"] > 1.05 and m["N"] > 100 else "❌"
        print(f"  {name:<40} {m['N']:>6} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['WR']:>5.1f} {m['calmar']:>7.1f} {v:>3}")
        results[name] = (trades, m)

    # V6: VR Expansion + Momentum Following
    for vr_th in [0.8, 1.0, 1.2, 1.5]:
        for mom_th in [0.01, 0.02, 0.03]:
            for sl, tp in [(0.01, 0.02), (0.01, 0.03), (0.01, 0.05),
                           (0.015, 0.03), (0.02, 0.05)]:
                name = f"V6: VR>{vr_th} m>{mom_th*100:.0f}% sl{sl*100:.0f}/tp{tp*100:.0f}"
                screen(name, collect_v6(data_15m, data_1h, data_4h,
                                         vr_th, mom_th, sl, tp))

    # V7: Strong Momentum Breakout
    for mom_th in [0.02, 0.03, 0.05, 0.08]:
        for sl, tp in [(0.01, 0.02), (0.01, 0.03), (0.01, 0.05),
                       (0.015, 0.05), (0.02, 0.05)]:
            name = f"V7: mom>{mom_th*100:.0f}% sl{sl*100:.0f}/tp{tp*100:.0f}"
            screen(name, collect_v7(data_15m, data_1h, data_4h,
                                     mom_th, sl, tp))

    # V8: Multi-TF Trend Alignment
    for mom_th in [0.01, 0.02, 0.03]:
        for htf_th in [0.01, 0.02, 0.05]:
            for sl, tp in [(0.01, 0.03), (0.01, 0.05), (0.015, 0.05)]:
                name = f"V8: m>{mom_th*100:.0f}% htf>{htf_th*100:.0f}% sl{sl*100:.0f}/tp{tp*100:.0f}"
                screen(name, collect_v8(data_15m, data_4h,
                                         mom_th, htf_th, sl, tp))

    # V9: VR Expansion Onset
    for vr_th in [0.6, 0.7, 0.8]:
        for lb in [4, 8, 16]:
            for sl, tp in [(0.01, 0.03), (0.01, 0.05), (0.015, 0.05)]:
                name = f"V9: VR>{vr_th} lb{lb} sl{sl*100:.0f}/tp{tp*100:.0f}"
                screen(name, collect_v9(data_15m, data_1h, data_4h,
                                         vr_th, lb, sl, tp))

    # V10: Momentum + RSI Confirmation
    for mom_th in [0.02, 0.03, 0.05]:
        for rsi_lo, rsi_hi in [(30, 70), (25, 75), (35, 65)]:
            for sl, tp in [(0.01, 0.03), (0.01, 0.05), (0.015, 0.05)]:
                name = f"V10: m>{mom_th*100:.0f}% RSI{rsi_lo}/{rsi_hi} sl{sl*100:.0f}/tp{tp*100:.0f}"
                screen(name, collect_v10(data_15m, data_1h, data_4h,
                                          mom_th, rsi_lo, rsi_hi, sl, tp))

    # ── Phase 3: Top candidates combined ──
    passing = [(name, trades, m) for name, (trades, m) in results.items()
               if m["PF"] > 1.05 and m["N"] > 100]
    passing.sort(key=lambda x: -x[2]["calmar"])

    print(f"\n  Phase 3: Combined Performance (Top {min(10, len(passing))})")
    print(f"  {'─'*90}")

    if not passing:
        print(f"  ⚠️  No vectors passed PF>1.05 AND N>100.")
        elapsed = time.time() - t0
        print(f"\n  Runtime: {elapsed:.1f}s  |  Commit: {commit}")
        print(f"{'='*100}\n")
        return

    for name, trades, m in passing[:10]:
        combined = bl_all + trades
        comb_pnl, comb_acc = portfolio_sim_with_details(combined, SLOTS)
        comb_m = compute_metrics(comb_pnl, data_years)
        comb_yr = year_metrics(comb_pnl, comb_acc, year_bounds)

        ovl = overlap_analysis(bl_all, trades)
        delta_pf = comb_m["PF"] - bl_m["PF"]
        delta_calmar = comb_m["calmar"] - bl_m["calmar"]

        print(f"\n  {name}")
        print(f"    Standalone:  N={m['N']:>6}  PF={m['PF']:.3f}  "
              f"EV={m['EV_bp']:+.1f}bp  Calmar={m['calmar']:.1f}")
        print(f"    Combined:    N={comb_m['N']:>6}  PF={comb_m['PF']:.3f} "
              f"(Δ{delta_pf:+.3f})  EV={comb_m['EV_bp']:+.1f}bp  "
              f"Calmar={comb_m['calmar']:.1f} (Δ{delta_calmar:+.1f})")
        print(f"    Anti-corr:   {ovl['unique_pct']:.1f}% unique "
              f"({ovl['complement_total'] - ovl['overlap_trades']}"
              f"/{ovl['complement_total']} non-overlapping)")

        # Per-year combined
        print(f"    Yearly: ", end="")
        for yr in sorted(comb_yr.keys()):
            if yr in bl_yr:
                delta_ev = comb_yr[yr]["EV_bp"] - bl_yr[yr]["EV_bp"]
                marker = "✓" if delta_ev >= -3.0 else "✗"
                print(f"{yr}:{delta_ev:+.1f}{marker} ", end="")
        print()

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R44b SUMMARY")
    print(f"  {'─'*90}")
    print(f"  Tested: {len(results)} vectors  |  Passed: {len(passing)}")
    if passing:
        best = passing[0]
        print(f"  Best: {best[0]}")
        print(f"        PF={best[2]['PF']:.3f}  N={best[2]['N']}  "
              f"Calmar={best[2]['calmar']:.1f}")
    print(f"  Runtime: {elapsed:.1f}s  |  Commit: {commit}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
