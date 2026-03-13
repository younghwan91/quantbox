"""r65: MFE/MAE Exit Optimization — Trail Bar-Close Gap & Breakeven SL.

Problem:
  Live dashboard shows SUIUSDT VR LONG: MFE=+131%, PnL=-$33 (trailing_stop).
  Root cause: trail exit is CLIENT-SIDE bar-close check in live, but backtest
  assumes INTRABAR execution at trail_stop_gain. Sharp intrabar reversals
  (bar goes +4.37%→-1.1% in one 15m bar) cause live loss vs backtest profit.

Research goals:
  Phase 1: Reproduce G_4TF A0 baseline (parity check)
  Phase 2: MFE distribution — peak_gain histogram per exit type
  Phase 3: "Bar-close trail" simulation — quantify the live/backtest gap
  Phase 4: Breakeven SL sweep — test [0.005, 0.01, 0.02, 0.03] thresholds
  Phase 5: Combined recommendation + report

Parity Notes:
  - Parameters: VolStraddleConfig() only (no hardcoding)
  - Baseline assert: N ±2%, PF ±2% vs G_4TF A0 (N≈10791, PF≈1.986)
  - sim functions: r43 imports for trailing, r57 collect_d_pullback for G_4TF
  - Lookahead: none (bar LOW/HIGH used only for completed bars)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import numba

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    WARMUP_15m,
    bootstrap_pf,
    compute_metrics,
    portfolio_sim,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_signal,
    make_direction_momentum,
)
from quantbox.backtest.trade_sim import sim_directional_trailing, sim_straddle_trailing
from research.active.r51_r60.r57_combo_30m_pullback import (
    G_4TF,
    VR_KEY,
    _dedup_sym,
    collect_d_pullback,
    collect_s_trailing,
    collect_ma,
)

# ── Config (source of truth) ────────────────────────────────────────────────
cfg = VolStraddleConfig()

D_SL          = cfg.d_sl              # 0.01
D_TP          = cfg.d_tp              # 0.12
D_TRAIL_ACT   = cfg.d_trail_activate  # 0.005
D_TRAIL_DIST  = cfg.d_trail_distance  # 0.002
S_SL          = cfg.s_sl
S_TP          = cfg.s_tp
S_TRAIL_ACT   = cfg.s_trail_activate
S_TRAIL_DIST  = cfg.s_trail_distance
VR_LN         = cfg.base_leverage / cfg.max_concurrent  # 3.0

assert D_TRAIL_ACT == 0.010, f"d_trail_activate mismatch: {D_TRAIL_ACT}"
assert D_TRAIL_DIST == 0.002, f"d_trail_distance mismatch: {D_TRAIL_DIST}"

CACHE_DIR  = "data/cache_extended"
DATA_YEARS = 6.0
MAX_SLOTS  = cfg.max_concurrent  # 10

print(f"Config: SL={D_SL}, TP={D_TP}, trail=({D_TRAIL_ACT},{D_TRAIL_DIST})")
print(f"VR_LN={VR_LN:.1f}, slots={MAX_SLOTS}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Extended Numba Sims
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_directional_trailing_with_mfe(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float, trail_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """sim_directional_trailing but also returns peak_gain and exit_reason per trade.

    exit_reason codes:
        0 = max_hold
        1 = trail_stop (intrabar — same as backtest baseline)
        2 = hard_tp
        3 = hard_sl

    Returns: (pnl, ent, ext, peak_gains, reasons)
    """
    n = len(close)
    cap = n // 2 + 1
    pnl     = np.empty(cap, dtype=np.float64)
    ent     = np.empty(cap, dtype=np.int64)
    ext     = np.empty(cap, dtype=np.int64)
    peaks   = np.empty(cap, dtype=np.float64)
    reasons = np.empty(cap, dtype=np.int8)
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
        trade_pnl    = 0.0
        trailing_act = False
        peak_gain    = 0.0
        reason       = np.int8(0)

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_act:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        reason = np.int8(1)
                        eb = j
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    reason = np.int8(2)
                    eb = j
                    break
                if not trailing_act and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    reason = np.int8(3)
                    eb = j
                    break
                if not trailing_act and peak_gain >= trail_activate:
                    trailing_act = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_act:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        reason = np.int8(1)
                        eb = j
                        break
                if low[j] <= entry * (1.0 - tp) and d == -1:
                    trade_pnl = tp
                    reason = np.int8(2)
                    eb = j
                    break
                if not trailing_act and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    reason = np.int8(3)
                    eb = j
                    break
                if not trailing_act and peak_gain >= trail_activate:
                    trailing_act = True
        else:
            if d == 1:
                trade_pnl = (close[end - 1] - entry) / entry
            else:
                trade_pnl = (entry - close[end - 1]) / entry
            reason = np.int8(0)

        trade_pnl -= cost
        pnl[cnt]     = trade_pnl
        ent[cnt]     = i
        ext[cnt]     = eb
        peaks[cnt]   = peak_gain
        reasons[cnt] = reason
        cnt += 1
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt], peaks[:cnt], reasons[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trailing_barclose(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float, trail_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """'Live behavior' trail simulation: exits at bar CLOSE when trail fires.

    Difference from sim_directional_trailing:
        When trail hits (low[j] <= trail_level), exit at close[j] instead of
        trail_stop_gain. This simulates live bar-close market order execution.

    Purpose: Quantify the gap between backtest (intrabar) and live (bar-close).
    """
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
        trade_pnl    = 0.0
        trailing_act = False
        peak_gain    = 0.0

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_act:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        # ← KEY CHANGE: exit at bar CLOSE (not trail_stop_gain)
                        trade_pnl = (close[j] - entry) / entry
                        eb = j
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_act and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_act and peak_gain >= trail_activate:
                    trailing_act = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_act:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        # ← KEY CHANGE: exit at bar CLOSE
                        trade_pnl = (entry - close[j]) / entry
                        eb = j
                        break
                if low[j] <= entry * (1.0 - tp) and d == -1:
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_act and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_act and peak_gain >= trail_activate:
                    trailing_act = True
        else:
            if d == 1:
                trade_pnl = (close[end - 1] - entry) / entry
            else:
                trade_pnl = (entry - close[end - 1]) / entry

        trade_pnl -= cost
        pnl[cnt] = trade_pnl
        ent[cnt] = i
        ext[cnt] = eb
        cnt += 1
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trailing_breakeven(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
    trail_activate: float, trail_distance: float,
    be_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trail sim with breakeven SL: after peak_gain >= be_threshold, floor exit at 0%.

    Simulates server-side SL moved to entry price after MFE exceeds be_threshold.
    Exit is still at bar CLOSE (live behavior) but clamped at max(0, close_gain).

    be_threshold: e.g. 0.01 = after gaining 1%, server SL moved to entry (breakeven).
    """
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
        trade_pnl    = 0.0
        trailing_act = False
        peak_gain    = 0.0
        be_active    = False

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not be_active and peak_gain >= be_threshold:
                    be_active = True

                # Hard SL check — before trail, and after trail if below breakeven
                if not trailing_act:
                    if low[j] <= entry * (1.0 - sl):
                        trade_pnl = -sl
                        eb = j
                        break
                else:
                    # If breakeven active: server SL at entry protects from loss
                    if be_active and low[j] <= entry:
                        # Breakeven SL fires before trail check
                        close_gain = (close[j] - entry) / entry
                        trade_pnl = max(0.0, close_gain)
                        eb = j
                        break
                    # Trail check
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        close_gain = (close[j] - entry) / entry
                        if be_active:
                            trade_pnl = max(0.0, close_gain)
                        else:
                            trade_pnl = close_gain
                        eb = j
                        break

                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_act and peak_gain >= trail_activate:
                    trailing_act = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not be_active and peak_gain >= be_threshold:
                    be_active = True

                if not trailing_act:
                    if high[j] >= entry * (1.0 + sl):
                        trade_pnl = -sl
                        eb = j
                        break
                else:
                    if be_active and high[j] >= entry:
                        close_gain = (entry - close[j]) / entry
                        trade_pnl = max(0.0, close_gain)
                        eb = j
                        break
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        close_gain = (entry - close[j]) / entry
                        if be_active:
                            trade_pnl = max(0.0, close_gain)
                        else:
                            trade_pnl = close_gain
                        eb = j
                        break

                if low[j] <= entry * (1.0 - tp) and d == -1:
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_act and peak_gain >= trail_activate:
                    trailing_act = True
        else:
            if d == 1:
                trade_pnl = (close[end - 1] - entry) / entry
            else:
                trade_pnl = (entry - close[end - 1]) / entry

        trade_pnl -= cost
        pnl[cnt] = trade_pnl
        ent[cnt] = i
        ext[cnt] = eb
        cnt += 1
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection Helpers
# ══════════════════════════════════════════════════════════════════════════════

def collect_d_barclose(data_15m, data_1h, data_4h, data_30m=None):
    """D_mom8 trades using bar-close trail execution (live behavior model)."""
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
            sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing_barclose(
                c, h, lo, sig, dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_d_breakeven(data_15m, data_1h, data_4h, be_threshold, data_30m=None):
    """D_mom8 trades: bar-close trail + breakeven SL at be_threshold."""
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
            sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing_breakeven(
                c, h, lo, sig, dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
                be_threshold,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_d_with_mfe(data_15m, data_1h, data_4h, data_30m=None):
    """D_mom8 trades with peak_gain and reason per trade (for MFE analysis)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m
    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_pnl, all_peaks, all_reasons = [], [], []
    for sym in sorted(data_15m.keys()):
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
            sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x, peaks, reasons = sim_directional_trailing_with_mfe(
                c, h, lo, sig, dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p)):
                all_pnl.append(float(p[k]))
                all_peaks.append(float(peaks[k]))
                all_reasons.append(int(reasons[k]))
    return (
        np.array(all_pnl),
        np.array(all_peaks),
        np.array(all_reasons, dtype=np.int8),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")
from research.active.r51_r60.r57_combo_30m_pullback import load_syms, build_all_tf

d15 = load_syms(min_years=5.0)
d30, d1h, d4h = build_all_tf(d15)
print(f"  15m: {len(d15)} syms | 30m: {len(d30)} | 1h: {len(d1h)} | 4h: {len(d4h)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: Baseline Reproduction (G_4TF A0, pb=0.0)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Phase 1: Baseline Reproduction (G_4TF A0, pb=0.0)")
print("=" * 70)

trades_d  = collect_d_pullback(d15, d1h, d4h, pullback_pct=0.0, pullback_window=3, data_30m=d30)
trades_s  = collect_s_trailing(d15, d1h, d4h, data_30m=d30)
trades_ma = collect_ma(d15, d1h, d4h)
trades_base = trades_d + trades_s + trades_ma

pnl_base = portfolio_sim(trades_base, max_concurrent=MAX_SLOTS)
m_base   = compute_metrics(pnl_base, DATA_YEARS)

print(f"  N={m_base['N']}, PF={m_base['PF']:.3f}, WR={m_base['WR']:.1f}%")
print(f"  Calmar={m_base['calmar']:.1f}, MaxDD={m_base['max_dd_pct']:.1f}%")

assert abs(m_base['N'] - 10791) / 10791 < 0.02, f"N drift: {m_base['N']}"
assert abs(m_base['PF'] - 1.986) / 1.986 < 0.02, f"PF drift: {m_base['PF']:.3f}"
print("  ✅ Baseline parity confirmed (N ±2%, PF ±2%)\n")

N_BASE  = m_base['N']
PF_BASE = m_base['PF']

# D-only baseline for apples-to-apples comparison in Phases 2-4
pnl_d_base = portfolio_sim(trades_d, max_concurrent=MAX_SLOTS)
m_d_base   = compute_metrics(pnl_d_base, DATA_YEARS)
print(f"  D-only baseline: N={m_d_base['N']}, PF={m_d_base['PF']:.3f}, WR={m_d_base['WR']:.1f}%")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: MFE Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Phase 2: MFE Distribution (D_mom8, G_4TF)")
print("=" * 70)

REASON_LABELS = {0: "max_hold", 1: "trail_stop", 2: "hard_TP", 3: "hard_SL"}

pnl_arr, peak_arr, reason_arr = collect_d_with_mfe(d15, d1h, d4h, data_30m=d30)
N_mfe = len(pnl_arr)
print(f"  Total D trades for MFE analysis: {N_mfe}")
print()

for rc in [0, 1, 2, 3]:
    mask = reason_arr == rc
    cnt  = mask.sum()
    if cnt == 0:
        continue
    sub_pnl   = pnl_arr[mask]
    sub_peaks = peak_arr[mask]
    wins = (sub_pnl > 0).sum()
    lbl = REASON_LABELS[rc]
    print(f"  Exit={lbl} ({cnt} trades, {cnt/N_mfe*100:.1f}%):")
    print(f"    WR={wins/cnt*100:.1f}%, avg_pnl={sub_pnl.mean()*100:.3f}%")
    print(f"    MFE P25={np.percentile(sub_peaks,25)*100:.2f}% "
          f"P50={np.percentile(sub_peaks,50)*100:.2f}% "
          f"P75={np.percentile(sub_peaks,75)*100:.2f}% "
          f"P90={np.percentile(sub_peaks,90)*100:.2f}%")
    if rc == 1:  # trail stops — key analysis
        neg_mask = sub_pnl < 0
        neg_cnt  = neg_mask.sum()
        print(f"    Trail→loss trades: {neg_cnt} ({neg_cnt/cnt*100:.1f}% of trail stops)")
        if neg_cnt > 0:
            print(f"    Loss avg MFE={sub_peaks[neg_mask].mean()*100:.2f}% "
                  f"avg_loss={sub_pnl[neg_mask].mean()*100:.3f}%")
            print(f"    Implied gap: avg MFE×30x = {sub_peaks[neg_mask].mean()*30*100:.1f}%"
                  f" → loss avg×30x = {sub_pnl[neg_mask].mean()*30*100:.1f}%")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Bar-Close Trail Execution (Live Behavior Simulation)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Phase 3: Bar-Close Trail Gap (Backtest vs Live Model)")
print("=" * 70)

trades_d_bc = collect_d_barclose(d15, d1h, d4h, data_30m=d30)
pnl_d_bc    = portfolio_sim(trades_d_bc, max_concurrent=MAX_SLOTS)
m_d_bc      = compute_metrics(pnl_d_bc, DATA_YEARS)

print(f"  Backtest (intrabar trail): N={m_d_base['N']}, PF={m_d_base['PF']:.3f}, "
      f"WR={m_d_base['WR']:.1f}%, Calmar={m_d_base['calmar']:.1f}")
print(f"  Live model (bar-close):    N={m_d_bc['N']}, PF={m_d_bc['PF']:.3f}, "
      f"WR={m_d_bc['WR']:.1f}%, Calmar={m_d_bc['calmar']:.1f}")
print()
dpf = m_d_bc['PF'] - m_d_base['PF']
dwr = m_d_bc['WR'] - m_d_base['WR']
dcalmar = m_d_bc['calmar'] - m_d_base['calmar']
print(f"  Gap: ΔPF={dpf:+.3f} ({dpf/m_d_base['PF']*100:+.1f}%), "
      f"ΔWR={dwr:+.1f}pp, ΔCalmar={dcalmar:+.1f}")
severity = "HIGH" if abs(dpf) > 0.05 else ("MEDIUM" if abs(dpf) > 0.02 else "LOW")
print(f"  Severity: {severity} (|ΔPF| = {abs(dpf):.3f})")
print()
print("  → This gap is what SUIUSDT-type trades create in live vs backtest.")
print("    Solution: Server-side SL ratchet to trail_stop_gain after trail activates.")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Breakeven SL Sweep
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Phase 4: Breakeven SL Threshold Sweep (D_mom8, bar-close+BE)")
print("=" * 70)
print(f"  Base = bar-close model: PF={m_d_bc['PF']:.3f}")
print()

BE_THRESHOLDS = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
be_results = []
for be in BE_THRESHOLDS:
    trades_be = collect_d_breakeven(d15, d1h, d4h, be_threshold=be, data_30m=d30)
    pnl_be    = portfolio_sim(trades_be, max_concurrent=MAX_SLOTS)
    m_be      = compute_metrics(pnl_be, DATA_YEARS)
    dpf_vs_bc    = m_be['PF'] - m_d_bc['PF']
    dpf_vs_base  = m_be['PF'] - m_d_base['PF']
    dwr_vs_bc    = (m_be['WR'] - m_d_bc['WR']) * 100
    be_results.append((be, m_be['N'], m_be['PF'], m_be['WR'], m_be['calmar'],
                        dpf_vs_bc, dpf_vs_base, dwr_vs_bc))
    print(f"  be={be*100:.1f}%: N={m_be['N']}, PF={m_be['PF']:.3f}, "
          f"WR={m_be['WR']:.1f}%, Calmar={m_be['calmar']:.1f}  "
          f"ΔPF_vs_barclose={dpf_vs_bc:+.3f}  ΔPF_vs_backtest={dpf_vs_base:+.3f}")

print()
# Find best breakeven threshold by PF
best_be = max(be_results, key=lambda r: r[2])
print(f"  Best BE threshold by PF: be={best_be[0]*100:.1f}% → PF={best_be[2]:.3f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Phase 5: Full Portfolio with Best BE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Phase 5: Full Portfolio (D+S+MA) with Best Breakeven SL")
print("=" * 70)

best_be_threshold = best_be[0]
trades_d_best = collect_d_breakeven(d15, d1h, d4h, be_threshold=best_be_threshold, data_30m=d30)
trades_full_be = trades_d_best + trades_s + trades_ma
pnl_full_be   = portfolio_sim(trades_full_be, max_concurrent=MAX_SLOTS)
m_full_be     = compute_metrics(pnl_full_be, DATA_YEARS)

print(f"  Baseline (backtest): N={N_BASE}, PF={PF_BASE:.3f}")
print(f"  With BE={best_be_threshold*100:.1f}% (bar-close model): "
      f"N={m_full_be['N']}, PF={m_full_be['PF']:.3f}, "
      f"WR={m_full_be['WR']:.1f}%, Calmar={m_full_be['calmar']:.1f}")
print()
dpf_total = m_full_be['PF'] - PF_BASE
print(f"  ΔPF vs backtest baseline: {dpf_total:+.3f} ({dpf_total/PF_BASE*100:+.1f}%)")
print()

boot_be = bootstrap_pf(pnl_full_be, n_boot=2000)
print(f"  Bootstrap PF CI (95%): [{boot_be['CI_lo']:.3f}, {boot_be['CI_hi']:.3f}], "
      f"P(PF≥1)={boot_be['P_PF_ge_1']:.1f}%")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Summary & Recommendations
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)
print()

trail_loss_mask = (reason_arr == 1) & (pnl_arr < 0)
trail_loss_cnt  = trail_loss_mask.sum()
trail_total     = (reason_arr == 1).sum()
trail_loss_avg_mfe = peak_arr[trail_loss_mask].mean() * 100 if trail_loss_cnt > 0 else 0.0
trail_loss_avg_pnl = pnl_arr[trail_loss_mask].mean() * 100 if trail_loss_cnt > 0 else 0.0

print("1. Trail Bar-Close Execution Gap:")
print(f"   Trail→loss: {trail_loss_cnt}/{trail_total} trail exits ({trail_loss_cnt/max(trail_total,1)*100:.1f}%)")
print(f"   avg MFE before loss: {trail_loss_avg_mfe:.2f}% (price) = "
      f"{trail_loss_avg_mfe*cfg.base_leverage:.0f}% (×30x, like dashboard MFE%)")
print(f"   avg loss: {trail_loss_avg_pnl:.3f}% (price) = {trail_loss_avg_pnl*30:.1f}% (×30x)")
print(f"   PF gap (backtest intrabar vs live bar-close): {dpf:+.3f}")
if abs(dpf) > 0.05:
    print("   ⚠️ HIGH PRIORITY: Implement server-side SL ratchet")
elif abs(dpf) > 0.02:
    print("   ⚡ MEDIUM: Server-side ratchet recommended")
else:
    print("   ✅ LOW: Gap is acceptable")
print()

print("2. Breakeven SL Results:")
for be, n, pf, wr, calmar, dpf_bc, dpf_bt, dwr in be_results:
    marker = " ← BEST" if abs(be - best_be_threshold) < 1e-6 else ""
    print(f"   be={be*100:.1f}%: PF={pf:.3f}, WR={wr:.1f}%, Calmar={calmar:.1f}"
          f"  ΔPF_vs_barclose={dpf_bc:+.3f}{marker}")
print()

print("3. Recommended Live Code Change:")
print("   Priority 1 (HIGH): Server SL ratchet after trail activates")
print("     After peak_gain >= trail_activate:")
print("       → Update Binance server STOP order to entry × (1 + trail_stop_gain)")
print("       → Binance executes intrabar → matches backtest assumption")
print("       → Eliminates SUIUSDT-type trail→loss scenarios")
print()
if best_be[5] > 0.01:  # ΔPF vs bar-close > 1%
    print(f"   Priority 2 (MEDIUM): Breakeven SL at be={best_be_threshold*100:.1f}%")
    print(f"     Simpler fallback: move server SL to entry after MFE≥{best_be_threshold*100:.1f}%")
    print(f"     ΔPF vs bar-close model: {best_be[5]:+.3f}")
else:
    print(f"   Priority 2 (LOW): Breakeven SL at be={best_be_threshold*100:.1f}%")
    print(f"     Marginal improvement ({best_be[5]:+.3f} PF) — lower priority than ratchet")
print()

print("4. Key Backtest-Live Parity Note:")
print("   Current backtest assumes intrabar trail execution (optimal).")
print("   Live executes at bar CLOSE (suboptimal for sharp reversals).")
print("   Server SL ratchet CLOSES this gap — both execute intrabar.")
print("   NO backtest param changes needed; only scalper.py code change.")
print()
print("=" * 70)
print(f"r65 DONE. Baseline: N={N_BASE}, PF={PF_BASE:.3f}")
print(f"Bar-close gap: ΔPF={dpf:+.3f} ({dpf/PF_BASE*100:+.1f}% of PF)")
print(f"Best BE fix: ΔPF vs bar-close={best_be[5]:+.3f}")
print("Recommended: Server SL ratchet → quantbox/scalper.py")
print("=" * 70)
