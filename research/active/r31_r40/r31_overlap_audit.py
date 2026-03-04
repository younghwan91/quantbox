#!/usr/bin/env python3
"""
R31 Overlap & Lookahead Audit
==============================
1. Lookahead: Verify signal at bar i only uses data ≤ i
2. Temporal overlap: Check how many multi-TF trades overlap in time
   for the same symbol (i.e., are they the same trade counted twice?)

Goal: If 15m signal fires at 15m bar 1000, and 1h signal fires at 1h bar 250
(= 15m bar 1003), these trade the SAME move on the SAME coin. Count overlaps.
"""

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import everything from R31
from research.active.mtf_vol_straddle_r31 import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
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
    precompute,
    vol_percentile,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Modified simulators that return trade entry/exit bar indices
# ═══════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pnl, entry_bars, exit_bars) arrays."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    entries = np.empty(cap, dtype=np.int64)
    exits = np.empty(cap, dtype=np.int64)
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
            entries[cnt] = i
            exits[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], entries[:cnt], exits[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_trades(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sig: np.ndarray, dire: np.ndarray,
    sl: float, tp: float, max_hold: int, cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pnl, entry_bars, exit_bars) arrays."""
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    entries = np.empty(cap, dtype=np.int64)
    exits = np.empty(cap, dtype=np.int64)
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
            entries[cnt] = i
            exits[cnt] = eb
            cnt += 1
        i = eb + 1
    return pnl[:cnt], entries[:cnt], exits[:cnt]


def get_trades_for_tf(
    data: dict, strat: str, warmup: int, max_hold_d: int,
    max_hold_s: int, min_dur: int, ratio: int,
) -> dict[str, list[tuple[int, int, float]]]:
    """Get (entry_15m_bar, exit_15m_bar, pnl) for each symbol at given TF."""
    result: dict[str, list[tuple[int, int, float]]] = {}
    for sym, d in data.items():
        if strat == "D_mom8":
            sig = gen_vr_signal(d["vr"], 0.45, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            pnl, ent, ext = sim_directional_trades(
                d["close"], d["high"], d["low"], sig, dire,
                0.01, 0.05, max_hold_d, COST_DIRECTIONAL)
        else:  # S_d8
            sig = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)
            pnl, ent, ext = sim_straddle_trades(
                d["close"], d["high"], d["low"], sig,
                0.015, 0.05, max_hold_s, COST_STRADDLE)

        trades = []
        for k in range(len(pnl)):
            # Convert bar indices to 15m-equivalent
            entry_15m = int(ent[k]) * ratio
            exit_15m = int(ext[k]) * ratio + (ratio - 1)  # end of the HTF bar
            trades.append((entry_15m, exit_15m, float(pnl[k])))
        result[sym] = trades
    return result


def check_overlap(
    trades_a: list[tuple[int, int, float]],
    trades_b: list[tuple[int, int, float]],
) -> int:
    """Count how many trades in B overlap with any trade in A.
    Overlap = entry_b is within [entry_a, exit_a] OR entry_a is within [entry_b, exit_b]."""
    count = 0
    for eb, xb, _ in trades_b:
        for ea, xa, _ in trades_a:
            # Check if trading windows overlap
            if eb <= xa and xb >= ea:
                count += 1
                break
    return count


def main() -> None:
    print("=" * 80)
    print("  R31 OVERLAP & LOOKAHEAD AUDIT")
    print("=" * 80)

    # ── Load data ──
    print("\n  Loading data...")
    data_15m = load_15m()
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m)
    precompute(data_1h)
    precompute(data_4h)
    n_symbols = len(data_15m)
    print(f"  {n_symbols} symbols loaded")
    print(f"  Symbols: {sorted(data_15m.keys())}")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 1: LOOKAHEAD AUDIT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 1: LOOKAHEAD BIAS AUDIT")
    print("=" * 80)

    # Test: Truncate data at bar T, compute signal. Then extend to T+1.
    # Signal at bar T should not change.
    sym = sorted(data_15m.keys())[0]
    d = data_15m[sym]
    n = len(d["close"])
    T = n // 2  # midpoint

    # Compute signal on first T bars
    c_t = d["close"][:T].copy()
    h_t = d["high"][:T].copy()
    l_t = d["low"][:T].copy()
    v_t = d["volume"][:T].copy()
    vr_t = compute_vr(c_t, h_t, l_t, 7, 50)
    mom_t = compute_momentum(c_t, 8)
    sig_t = gen_vr_signal(vr_t, 0.45, WARMUP_15m)
    dur_t = compute_squeeze_dur(vr_t, 0.55)
    sig_dur_t = gen_vr_dur_signal(vr_t, 0.55, 8, WARMUP_15m)

    # Compute signal on first T+100 bars
    c_t2 = d["close"][:T + 100].copy()
    h_t2 = d["high"][:T + 100].copy()
    l_t2 = d["low"][:T + 100].copy()
    v_t2 = d["volume"][:T + 100].copy()
    vr_t2 = compute_vr(c_t2, h_t2, l_t2, 7, 50)
    mom_t2 = compute_momentum(c_t2, 8)
    sig_t2 = gen_vr_signal(vr_t2, 0.45, WARMUP_15m)
    dur_t2 = compute_squeeze_dur(vr_t2, 0.55)
    sig_dur_t2 = gen_vr_dur_signal(vr_t2, 0.55, 8, WARMUP_15m)

    # Check: signals at [0:T] should be identical
    vr_match = np.allclose(vr_t[~np.isnan(vr_t)], vr_t2[:T][~np.isnan(vr_t)])
    mom_match = np.allclose(mom_t, mom_t2[:T])
    sig_match = np.array_equal(sig_t, sig_t2[:T])
    dur_match = np.array_equal(dur_t, dur_t2[:T])
    sig_dur_match = np.array_equal(sig_dur_t, sig_dur_t2[:T])

    print(f"\n  Symbol: {sym}, T={T}")
    print(f"  VR[0:T] unchanged when future added: {'✅ PASS' if vr_match else '❌ FAIL'}")
    print(f"  Momentum[0:T] unchanged: {'✅ PASS' if mom_match else '❌ FAIL'}")
    print(f"  VR signal[0:T] unchanged: {'✅ PASS' if sig_match else '❌ FAIL'}")
    print(f"  Squeeze dur[0:T] unchanged: {'✅ PASS' if dur_match else '❌ FAIL'}")
    print(f"  VR+dur signal[0:T] unchanged: {'✅ PASS' if sig_dur_match else '❌ FAIL'}")

    # Verify entry at close[i], exit check starts at i+1
    print(f"\n  sim_straddle: entry at close[i], exit checks j in [i+1, i+1+max_hold)")
    print(f"  sim_directional: same pattern")
    print(f"  → Signal uses close[i], entry is close[i]. This is standard 'bar-close' entry.")
    print(f"  → In live: signal fires a few seconds before bar close → enter at close. ")
    print(f"    Cost (0.04%/leg) covers this micro-slippage. ✅ ACCEPTABLE")

    all_pass = vr_match and mom_match and sig_match and dur_match and sig_dur_match
    print(f"\n  LOOKAHEAD VERDICT: {'✅ NO LOOKAHEAD DETECTED' if all_pass else '❌ LOOKAHEAD FOUND!'}")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 2: TEMPORAL OVERLAP ACROSS TFs
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 2: TEMPORAL OVERLAP ACROSS TIMEFRAMES")
    print("=" * 80)

    print("\n  If 15m and 1h signals fire on the SAME symbol at overlapping times,")
    print("  they trade the SAME market move → NOT independent → inflated N count.\n")

    for strat in ["D_mom8", "S_d8"]:
        print(f"\n  ── {strat} ──")

        # Get trades per TF
        trades_15m = get_trades_for_tf(
            data_15m, strat, WARMUP_15m, 192, 384, 8, ratio=1)
        trades_1h = get_trades_for_tf(
            data_1h, strat, WARMUP_1h, 48, 96, 8, ratio=4)
        trades_4h = get_trades_for_tf(
            data_4h, strat, WARMUP_4h, 12, 24, 8, ratio=16)

        n_15m = sum(len(v) for v in trades_15m.values())
        n_1h = sum(len(v) for v in trades_1h.values())
        n_4h = sum(len(v) for v in trades_4h.values())

        print(f"  Trade counts: 15m={n_15m}, 1h={n_1h}, 4h={n_4h}")

        # Check 15m vs 1h overlap
        overlap_15m_1h = 0
        overlap_15m_4h = 0
        overlap_1h_4h = 0

        for sym in sorted(data_15m.keys()):
            t_15 = trades_15m.get(sym, [])
            t_1h = trades_1h.get(sym, [])
            t_4h = trades_4h.get(sym, [])

            if t_15 and t_1h:
                overlap_15m_1h += check_overlap(t_15, t_1h)
            if t_15 and t_4h:
                overlap_15m_4h += check_overlap(t_15, t_4h)
            if t_1h and t_4h:
                overlap_1h_4h += check_overlap(t_1h, t_4h)

        print(f"\n  Overlaps (B trades that overlap with any A trade):")
        print(f"    15m vs 1h:  {overlap_15m_1h}/{n_1h} 1h trades overlap with 15m "
              f"({overlap_15m_1h/n_1h*100:.1f}%)" if n_1h > 0 else "    15m vs 1h: N/A")
        print(f"    15m vs 4h:  {overlap_15m_4h}/{n_4h} 4h trades overlap with 15m "
              f"({overlap_15m_4h/n_4h*100:.1f}%)" if n_4h > 0 else "    15m vs 4h: N/A")
        print(f"    1h vs 4h:   {overlap_1h_4h}/{n_4h} 4h trades overlap with 1h "
              f"({overlap_1h_4h/n_4h*100:.1f}%)" if n_4h > 0 else "    1h vs 4h: N/A")

        # For G-ALL: compute truly unique trades
        total_raw = n_15m + n_1h + n_4h
        # Unique = 15m trades + 1h trades NOT overlapping 15m + 4h trades NOT overlapping 15m or 1h
        unique_1h = n_1h - overlap_15m_1h  # 1h trades not overlapping any 15m
        # For 4h: check overlap with (15m OR 1h)
        overlap_4h_any = 0
        for sym in sorted(data_15m.keys()):
            t_15 = trades_15m.get(sym, [])
            t_1h = trades_1h.get(sym, [])
            t_4h = trades_4h.get(sym, [])
            if not t_4h:
                continue
            # Check each 4h trade: does it overlap with ANY 15m or 1h trade?
            for eb, xb, _ in t_4h:
                overlapped = False
                for ea, xa, _ in t_15:
                    if eb <= xa and xb >= ea:
                        overlapped = True
                        break
                if not overlapped:
                    for ea, xa, _ in t_1h:
                        if eb <= xa and xb >= ea:
                            overlapped = True
                            break
                if overlapped:
                    overlap_4h_any += 1
        unique_4h = n_4h - overlap_4h_any

        truly_unique = n_15m + max(0, unique_1h) + max(0, unique_4h)
        print(f"\n  G-ALL uniqueness analysis:")
        print(f"    Raw total (15m+1h+4h): {total_raw}")
        print(f"    15m base trades: {n_15m}")
        print(f"    1h truly new (non-overlapping): {max(0, unique_1h)} "
              f"({max(0, unique_1h)/n_1h*100:.1f}% of 1h)" if n_1h > 0 else "")
        print(f"    4h truly new (non-overlapping): {max(0, unique_4h)} "
              f"({max(0, unique_4h)/n_4h*100:.1f}% of 4h)" if n_4h > 0 else "")
        print(f"    Truly unique trades: {truly_unique}")
        print(f"    Inflation rate: {(total_raw - truly_unique)/total_raw*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 3: DEDUP BACKTEST — "one position per symbol" rule
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 3: DEDUPED G-ALL BACKTEST (skip if symbol already has position)")
    print("=" * 80)
    print("\n  Rule: for each symbol, merge all TF trades chronologically.")
    print("  If a new signal fires while the previous trade is still open → skip it.")
    print("  Priority: earlier entry wins (first-come-first-served).\n")

    from research.active.mtf_vol_straddle_r31 import bootstrap_pf, full_stats, pf_from_pnl

    # Configs: (variant_name, tf_list)
    dedup_variants = [
        ("A-15m (baseline, no dedup needed)", [("15m", 1, WARMUP_15m, 192, 384, 8)]),
        ("E-15m+4h", [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ]),
        ("G-ALL", [
            ("15m", 1, WARMUP_15m, 192, 384, 8),
            ("1h", 4, WARMUP_1h, 48, 96, 8),
            ("4h", 16, WARMUP_4h, 12, 24, 8),
        ]),
    ]

    for var_name, tf_list in dedup_variants:
        print(f"\n  {'─' * 74}")
        print(f"  {var_name}")
        print(f"  {'─' * 74}")

        for strat in ["D_mom8", "S_d8"]:
            # Collect all trades across TFs per symbol, tagged with 15m-equivalent times
            tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}

            raw_pnls: list[float] = []
            deduped_pnls: list[float] = []

            for sym in sorted(data_15m.keys()):
                # Gather trades from all TFs for this symbol
                all_sym_trades: list[tuple[int, int, float, str]] = []  # (entry_15m, exit_15m, pnl, tf)

                for tf_label, ratio, warmup, mh_d, mh_s, min_dur in tf_list:
                    d_tf = tf_data_map[ratio]
                    if sym not in d_tf:
                        continue
                    d = d_tf[sym]

                    if strat == "D_mom8":
                        sig = gen_vr_signal(d["vr"], 0.45, warmup)
                        dire = make_direction_momentum(sig, d["mom_8"])
                        pnl, ent, ext = sim_directional_trades(
                            d["close"], d["high"], d["low"], sig, dire,
                            0.01, 0.05, mh_d, COST_DIRECTIONAL)
                    else:
                        sig = gen_vr_dur_signal(d["vr"], 0.55, min_dur, warmup)
                        pnl, ent, ext = sim_straddle_trades(
                            d["close"], d["high"], d["low"], sig,
                            0.015, 0.05, mh_s, COST_STRADDLE)

                    for k in range(len(pnl)):
                        entry_15m = int(ent[k]) * ratio
                        exit_15m = int(ext[k]) * ratio + (ratio - 1)
                        all_sym_trades.append((entry_15m, exit_15m, float(pnl[k]), tf_label))
                        raw_pnls.append(float(pnl[k]))

                # Sort by entry time (first-come-first-served)
                all_sym_trades.sort(key=lambda x: x[0])

                # Dedup: skip if overlapping with any already-accepted trade
                accepted: list[tuple[int, int]] = []  # (entry, exit) of accepted trades
                for entry_15m, exit_15m, pnl_val, tf_lbl in all_sym_trades:
                    # Check if this trade overlaps with any accepted trade
                    conflict = False
                    for ae, ax in accepted:
                        if entry_15m <= ax and exit_15m >= ae:
                            conflict = True
                            break
                    if not conflict:
                        deduped_pnls.append(pnl_val)
                        accepted.append((entry_15m, exit_15m))

            raw_arr = np.array(raw_pnls)
            ded_arr = np.array(deduped_pnls)

            raw_s = full_stats(raw_arr)
            ded_s = full_stats(ded_arr)
            ded_bs = bootstrap_pf(ded_arr)

            removed = len(raw_pnls) - len(deduped_pnls)
            print(f"\n    {strat}:")
            print(f"      Raw:     N={raw_s['N']:>5}  PF={raw_s['PF']:.3f}  EV={raw_s['EV_bp']:>+6.1f}bp  t={raw_s['t']:>5.2f}")
            print(f"      Deduped: N={ded_s['N']:>5}  PF={ded_s['PF']:.3f}  EV={ded_s['EV_bp']:>+6.1f}bp  t={ded_s['t']:>5.2f}  "
                  f"P(PF≥1)={ded_bs['P_PF_ge_1']:.0f}%  CI=[{ded_bs['CI_lo']:.3f},{ded_bs['CI_hi']:.3f}]")
            print(f"      Removed: {removed} trades ({removed/len(raw_pnls)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("  AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
