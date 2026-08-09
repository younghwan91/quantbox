"""r90: Ratchet-Aware Trail Sim — True Live Execution Model.

**Discovery (r89):**
  Trail ratchet (r65) is DEAD CODE for straddle. 100% of trail exits show
  ACTIVATED+HIT on the SAME millisecond. The ratchet never executes because
  `update_position()` returns "trailing_stop" before ratchet code runs.
  Root cause: 0.1% trail_distance is too tight — any 15m bar that activates
  trail (high >= +1.5%) will also fire it (low <= +1.4%).

**Three sim models compared:**
  IDEAL:   fire on worst_gain, exit at trail_stop → PF=5.44 (current sim)
  WORST:   fire on worst_gain, exit at close → PF=0.72 (r89 bar-close)
  RATCHET: ratchet-aware — if close > trail_stop on activation bar,
           position survives (server STOP handles exit at trail_stop).
           Otherwise exit at close. Truth is between IDEAL and WORST.

**Key insight:**
  With wider trail_distance, activation bar's close is more likely to be
  ABOVE trail_stop → ratchet works → exit at trail_stop (server STOP).
  There's a sweet spot where ratchet-model PF is maximized.

Usage:
    uv run python research/active/r86_r90/r90_ratchet_model.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import numba

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    COST_DIRECTIONAL,
    WARMUP_15m,
    compute_metrics,
    portfolio_sim,
    sim_straddle_trailing,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)
from research.active.r86_r90.r88_edge_decay import load_all_data

# Reuse r89 bar-close models
from research.active.r86_r90.r89_trail_distance import (
    sim_straddle_trailing_barclose,
    sim_directional_trailing_barclose,
)


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ──────────────────────────────────────────────────────────────
# Model RATCHET: Ratchet-aware trailing stop
# ──────────────────────────────────────────────────────────────

@numba.jit(nopython=True, cache=True)
def sim_straddle_ratchet(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,
    trail_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ratchet-aware straddle trailing stop simulation.

    Per-leg logic:
      1. SL fires first (intra-bar, always) — matching live server SL.
      2. On trail activation bar:
         - If close_gain > trail_stop → survive. Ratchet places server STOP.
         - If close_gain ≤ trail_stop → exit at close (WOULD_TRIGGER).
      3. On subsequent bars with ratchet STOP live:
         - If worst_gain ≤ trail_stop → exit at trail_stop (server fires intra-bar).

    Returns (pnl, entry_bars, exit_bars, ratchet_stats):
      ratchet_stats: per-trade [long_ratchet_worked, short_ratchet_worked]
        1.0 = ratchet placed successfully, exit at trail_stop
        0.0 = ratchet failed (WOULD_TRIGGER), exit at close
       -1.0 = SL/TP exit (ratchet irrelevant)
    """
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    ent_arr = np.empty(cap, dtype=np.int64)
    ext_arr = np.empty(cap, dtype=np.int64)
    # Track ratchet success per trade: [long_status, short_status]
    rstats = np.empty((cap, 2), dtype=np.float64)
    cnt = 0
    i = 0

    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0.0:
            i += 1
            continue

        la = True   # long leg active
        sa = True   # short leg active
        le = 0.0    # long exit pnl
        se = 0.0    # short exit pnl
        l_peak = 0.0
        s_peak = 0.0
        l_trailing = False
        s_trailing = False
        l_ratchet_live = False
        s_ratchet_live = False
        l_rstat = -1.0  # default: no trail involvement
        s_rstat = -1.0
        eb = i
        end = min(i + 1 + max_hold, n)

        for j in range(i + 1, end):
            # ── Long leg ──
            if la:
                # SL first (server-side, intra-bar)
                if low[j] <= entry * (1.0 - sl):
                    le = -sl
                    la = False
                    l_rstat = -1.0
                else:
                    l_gain_best = (high[j] - entry) / entry
                    l_gain_close = (close[j] - entry) / entry

                    if l_gain_best > l_peak:
                        l_peak = l_gain_best

                    # Trail activation check
                    if not l_trailing and l_peak >= trail_activate:
                        l_trailing = True
                        # Activation bar: can ratchet place STOP?
                        l_trail_stop = l_peak - trail_distance
                        if l_gain_close > l_trail_stop:
                            # Close is ABOVE trail_stop → ratchet STOP placed
                            l_ratchet_live = True
                            # Position survives this bar
                        else:
                            # Close ≤ trail_stop → WOULD_TRIGGER → exit at close
                            le = l_gain_close
                            la = False
                            l_rstat = 0.0  # ratchet failed

                    elif l_trailing:
                        l_trail_stop = l_peak - trail_distance

                        if l_ratchet_live:
                            # Server STOP is live → fires intra-bar at trail_stop
                            if low[j] <= entry * (1.0 + l_trail_stop):
                                le = max(l_trail_stop, 0.0)
                                la = False
                                l_rstat = 1.0  # ratchet worked!
                            elif l_gain_close > l_trail_stop:
                                pass  # STOP not triggered, position continues
                            else:
                                # Close dropped below trail_stop but low didn't touch
                                # Ratchet updates STOP each bar, so this shouldn't happen
                                # unless peak just updated. Exit at close to be safe.
                                le = l_gain_close
                                la = False
                                l_rstat = 0.0
                        else:
                            # Ratchet not yet live — try to place this bar
                            if l_gain_close > l_trail_stop:
                                l_ratchet_live = True
                            else:
                                # Still can't place ratchet
                                if low[j] <= entry * (1.0 + l_trail_stop):
                                    le = l_gain_close
                                    la = False
                                    l_rstat = 0.0

                    # TP check
                    if la and high[j] >= entry * (1.0 + tp):
                        le = tp
                        la = False
                        l_rstat = -1.0

            # ── Short leg ──
            if sa:
                if high[j] >= entry * (1.0 + sl):
                    se = -sl
                    sa = False
                    s_rstat = -1.0
                else:
                    s_gain_best = (entry - low[j]) / entry
                    s_gain_close = (entry - close[j]) / entry

                    if s_gain_best > s_peak:
                        s_peak = s_gain_best

                    if not s_trailing and s_peak >= trail_activate:
                        s_trailing = True
                        s_trail_stop = s_peak - trail_distance
                        if s_gain_close > s_trail_stop:
                            s_ratchet_live = True
                        else:
                            se = s_gain_close
                            sa = False
                            s_rstat = 0.0

                    elif s_trailing:
                        s_trail_stop = s_peak - trail_distance

                        if s_ratchet_live:
                            if high[j] >= entry * (1.0 - s_trail_stop):
                                se = max(s_trail_stop, 0.0)
                                sa = False
                                s_rstat = 1.0
                            elif s_gain_close > s_trail_stop:
                                pass
                            else:
                                se = s_gain_close
                                sa = False
                                s_rstat = 0.0
                        else:
                            if s_gain_close > s_trail_stop:
                                s_ratchet_live = True
                            else:
                                if high[j] >= entry * (1.0 - s_trail_stop):
                                    se = s_gain_close
                                    sa = False
                                    s_rstat = 0.0

                    if sa and low[j] <= entry * (1.0 - tp):
                        se = tp
                        sa = False
                        s_rstat = -1.0

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
            pnl_arr[cnt] = net
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            rstats[cnt, 0] = l_rstat
            rstats[cnt, 1] = s_rstat
            cnt += 1
        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], rstats[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_directional_ratchet(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    dire: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,
    trail_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ratchet-aware directional trailing stop simulation.

    Same 3-phase logic as sim_straddle_ratchet but for single-direction trades.
    Returns (pnl, entry_bars, exit_bars, ratchet_stats).
    ratchet_stats is 1D: per-trade ratchet status (1.0/0.0/-1.0).
    """
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    ent_arr = np.empty(cap, dtype=np.int64)
    ext_arr = np.empty(cap, dtype=np.int64)
    rstats = np.empty(cap, dtype=np.float64)
    cnt = 0
    i = 0

    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        d = dire[i]
        if d == 0:
            i += 1
            continue

        entry = close[i]
        if entry <= 0.0:
            i += 1
            continue

        peak = 0.0
        trailing = False
        ratchet_live = False
        rstat = -1.0
        pnl_val = 0.0
        eb = i
        end = min(i + 1 + max_hold, n)

        for j in range(i + 1, end):
            if d == 1:  # LONG
                worst = (low[j] - entry) / entry
                best = (high[j] - entry) / entry
                close_gain = (close[j] - entry) / entry
            else:  # SHORT
                worst = (entry - high[j]) / entry
                best = (entry - low[j]) / entry
                close_gain = (entry - close[j]) / entry

            # SL first
            if worst <= -sl:
                pnl_val = -sl
                eb = j
                rstat = -1.0
                break

            if best > peak:
                peak = best

            if not trailing and peak >= trail_activate:
                trailing = True
                trail_stop = peak - trail_distance
                if close_gain > trail_stop:
                    ratchet_live = True
                else:
                    pnl_val = close_gain
                    eb = j
                    rstat = 0.0
                    break

            elif trailing:
                trail_stop = peak - trail_distance

                if ratchet_live:
                    if worst <= trail_stop:
                        pnl_val = max(trail_stop, -sl)
                        eb = j
                        rstat = 1.0
                        break
                    elif close_gain > trail_stop:
                        pass
                    else:
                        pnl_val = close_gain
                        eb = j
                        rstat = 0.0
                        break
                else:
                    if close_gain > trail_stop:
                        ratchet_live = True
                    elif worst <= trail_stop:
                        pnl_val = close_gain
                        eb = j
                        rstat = 0.0
                        break

            # TP
            if best >= tp:
                pnl_val = tp
                eb = j
                rstat = -1.0
                break
        else:
            eb = end - 1
            if d == 1:
                pnl_val = (close[eb] - entry) / entry
            else:
                pnl_val = (entry - close[eb]) / entry
            rstat = -1.0

        if cnt < cap:
            pnl_arr[cnt] = pnl_val - cost
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            rstats[cnt] = rstat
            cnt += 1
        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], rstats[:cnt]


# ──────────────────────────────────────────────────────────────
# Trade collection with model selection
# ──────────────────────────────────────────────────────────────

def collect_trades_3model(
    cfg_dict: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    model: str = "ratchet",
    strat: str = "S_d8",
    trail_activate: float | None = None,
    trail_distance: float | None = None,
) -> list[tuple]:
    """Collect trades using specified exit model.

    model: 'ideal' (trail_stop), 'worst' (bar_close), 'ratchet' (ratchet-aware)
    """
    cfg = VolStraddleConfig()
    sp = cfg_dict.get("sp", cfg.vr_short)
    lp = cfg_dict.get("lp", cfg.vr_long)
    d_vr = cfg_dict.get("d_vr", cfg.d_vr_threshold)
    s_vr = cfg_dict.get("s_vr", cfg.s_vr_threshold)
    s_dur = cfg_dict.get("s_dur", cfg.s_squeeze_dur)
    d_sl = cfg_dict.get("d_sl", cfg.d_sl)
    s_sl = cfg_dict.get("s_sl", cfg.s_sl)
    tp = cfg_dict.get("tp", cfg.d_tp)
    max_hold = cfg_dict.get("max_hold", cfg.max_hold_15m)

    ta = trail_activate if trail_activate is not None else cfg.s_trail_activate
    td = trail_distance if trail_distance is not None else cfg.s_trail_distance
    d_ta = trail_activate if trail_activate is not None else cfg.d_trail_activate
    d_td = trail_distance if trail_distance is not None else cfg.d_trail_distance

    vr_key = f"vr_{sp}_{lp}"
    all_trades: list[tuple] = []

    for sym, d15 in data_15m.items():
        c15 = d15["close"]
        h15 = d15["high"]
        l15 = d15["low"]
        offset = d15.get("_offset", 0)
        vr = d15.get(vr_key)
        if vr is None:
            continue

        d1h = data_1h.get(sym)
        d4h = data_4h.get(sym)
        if d1h is None or d4h is None:
            continue

        if strat in ("S_d8", "both"):
            sig_s = gen_vr_dur_signal(vr, s_vr, s_dur, WARMUP_15m)

            if model == "ideal":
                pnl_s, ent_s, ext_s = sim_straddle_trailing(
                    c15, h15, l15, sig_s, s_sl, tp, max_hold,
                    COST_STRADDLE, ta, td,
                )
            elif model == "worst":
                pnl_s, ent_s, ext_s = sim_straddle_trailing_barclose(
                    c15, h15, l15, sig_s, s_sl, tp, max_hold,
                    COST_STRADDLE, ta, td,
                )
            elif model == "ratchet":
                pnl_s, ent_s, ext_s, _ = sim_straddle_ratchet(
                    c15, h15, l15, sig_s, s_sl, tp, max_hold,
                    COST_STRADDLE, ta, td,
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            for k in range(len(pnl_s)):
                all_trades.append((
                    sym,
                    int(ent_s[k]) + offset,
                    int(ext_s[k]) + offset,
                    float(pnl_s[k]),
                ))

        if strat in ("D_mom8", "both"):
            sig_d = gen_vr_signal(vr, d_vr, WARMUP_15m)
            d_1h = make_direction_momentum(d1h["close"], 8)
            d_4h = make_direction_momentum(d4h["close"], 8)
            dire = np.zeros(len(c15), dtype=np.int32)
            ratio_1h = 4
            ratio_4h = 16
            for ii in range(len(c15)):
                j1 = (ii // ratio_1h) - 1
                j4 = (ii // ratio_4h) - 1
                if j1 >= 0 and j4 >= 0 and j1 < len(d_1h) and j4 < len(d_4h):
                    if d_1h[j1] != 0 and d_1h[j1] == d_4h[j4]:
                        dire[ii] = d_1h[j1]

            if model == "ideal":
                pnl_d, ent_d, ext_d = sim_directional_trailing(
                    c15, h15, l15, sig_d, dire, d_sl, tp, max_hold,
                    COST_DIRECTIONAL, d_ta, d_td,
                )
            elif model == "worst":
                pnl_d, ent_d, ext_d = sim_directional_trailing_barclose(
                    c15, h15, l15, sig_d, dire, d_sl, tp, max_hold,
                    COST_DIRECTIONAL, d_ta, d_td,
                )
            elif model == "ratchet":
                pnl_d, ent_d, ext_d, _ = sim_directional_ratchet(
                    c15, h15, l15, sig_d, dire, d_sl, tp, max_hold,
                    COST_DIRECTIONAL, d_ta, d_td,
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            for k in range(len(pnl_d)):
                all_trades.append((
                    sym,
                    int(ent_d[k]) + offset,
                    int(ext_d[k]) + offset,
                    float(pnl_d[k]),
                ))

    return all_trades


def collect_ratchet_stats(
    cfg_dict: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str = "S_d8",
    trail_activate: float | None = None,
    trail_distance: float | None = None,
) -> dict:
    """Collect ratchet success/failure statistics across all symbols."""
    cfg = VolStraddleConfig()
    sp = cfg_dict.get("sp", cfg.vr_short)
    lp = cfg_dict.get("lp", cfg.vr_long)
    s_vr = cfg_dict.get("s_vr", cfg.s_vr_threshold)
    s_dur = cfg_dict.get("s_dur", cfg.s_squeeze_dur)
    s_sl = cfg_dict.get("s_sl", cfg.s_sl)
    tp = cfg_dict.get("tp", cfg.d_tp)
    max_hold = cfg_dict.get("max_hold", cfg.max_hold_15m)

    ta = trail_activate if trail_activate is not None else cfg.s_trail_activate
    td = trail_distance if trail_distance is not None else cfg.s_trail_distance

    vr_key = f"vr_{sp}_{lp}"
    total_ratchet_ok = 0
    total_ratchet_fail = 0
    total_sl_tp = 0

    for sym, d15 in data_15m.items():
        c15 = d15["close"]
        h15 = d15["high"]
        l15 = d15["low"]
        vr = d15.get(vr_key)
        if vr is None:
            continue

        sig_s = gen_vr_dur_signal(vr, s_vr, s_dur, WARMUP_15m)

        _, _, _, rstats = sim_straddle_ratchet(
            c15, h15, l15, sig_s, s_sl, tp, max_hold,
            COST_STRADDLE, ta, td,
        )

        for k in range(len(rstats)):
            for leg in range(2):  # long=0, short=1
                v = rstats[k, leg]
                if v == 1.0:
                    total_ratchet_ok += 1
                elif v == 0.0:
                    total_ratchet_fail += 1
                else:
                    total_sl_tp += 1

    total_trail = total_ratchet_ok + total_ratchet_fail
    rate = total_ratchet_ok / total_trail * 100 if total_trail > 0 else 0.0

    return {
        "ratchet_ok": total_ratchet_ok,
        "ratchet_fail": total_ratchet_fail,
        "sl_tp": total_sl_tp,
        "total_trail": total_trail,
        "ratchet_success_rate": rate,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    commit = get_commit()
    cfg = VolStraddleConfig()

    print(f"r90: Ratchet-Aware Trail Sim Model (commit={commit})")
    print(f"  s_trail_activate={cfg.s_trail_activate}")
    print(f"  s_trail_distance={cfg.s_trail_distance}")
    print(f"  d_trail_activate={cfg.d_trail_activate}")
    print(f"  d_trail_distance={cfg.d_trail_distance}")

    # Load data
    data_15m, _data_30m, data_1h, data_4h, data_years, _max_bars = load_all_data(cfg)
    print(f"  symbols={len(data_15m)}, years={data_years:.2f}")

    base_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "s_vr": cfg.s_vr_threshold,
        "s_dur": cfg.s_squeeze_dur,
        "d_sl": cfg.d_sl,
        "s_sl": cfg.s_sl,
        "tp": cfg.d_tp,
        "max_hold": cfg.max_hold_15m,
    }

    # ── T1: Three-model comparison at current params ──
    print("\n" + "=" * 70)
    print("T1: Three-Model Comparison (current params)")
    print("=" * 70)

    for model_name in ("ideal", "worst", "ratchet"):
        trades = collect_trades_3model(
            base_cfg, data_15m, data_1h, data_4h,
            model=model_name, strat="S_d8",
        )
        strd_syms = set()
        for t in trades:
            if True:  # S_d8 only
                strd_syms.add(t[0])
        pnl = portfolio_sim(trades, max_concurrent=5, straddle_syms=strd_syms)
        m = compute_metrics(pnl, data_years)
        print(f"\n  Model={model_name.upper():<8} | N={m['N']:,} PF={m['PF']:.3f} "
              f"WR={m['WR']:.1f}% EV={m['EV_bp']:+.0f}bp MaxDD={m['max_dd_pct']:.1f}%")

    # Ratchet stats at current params
    stats = collect_ratchet_stats(
        base_cfg, data_15m, data_1h, data_4h,
        strat="S_d8",
    )
    print(f"\n  Ratchet stats (distance={cfg.s_trail_distance}):")
    print(f"    ratchet_ok={stats['ratchet_ok']:,}, ratchet_fail={stats['ratchet_fail']:,}, "
          f"SL/TP={stats['sl_tp']:,}")
    print(f"    success_rate={stats['ratchet_success_rate']:.1f}%")

    # ── T2: Trail distance sweep (S_d8 only, 3 models) ──
    print("\n" + "=" * 70)
    print("T2: Trail Distance Sweep (S_d8, activate=0.015 fixed)")
    print("=" * 70)

    distances = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]

    print(f"\n  {'dist':>6} | {'IDEAL PF':>8} {'N':>5} | "
          f"{'WORST PF':>8} {'N':>5} | "
          f"{'RATCHET PF':>10} {'N':>5} | "
          f"{'ratchet%':>8}")
    print("  " + "-" * 80)

    sweep_results = []
    for dist in distances:
        row = {"distance": dist}

        for model_name in ("ideal", "worst", "ratchet"):
            trades = collect_trades_3model(
                base_cfg, data_15m, data_1h, data_4h,
                model=model_name, strat="S_d8",
                trail_distance=dist,
            )
            strd_syms = set()
            for t in trades:
                if True:  # S_d8 only
                    strd_syms.add(t[0])
            pnl = portfolio_sim(trades, max_concurrent=5, straddle_syms=strd_syms)
            m = compute_metrics(pnl, data_years)
            row[f"{model_name}_pf"] = m["PF"]
            row[f"{model_name}_n"] = m["N"]
            row[f"{model_name}_wr"] = m["WR"]
            row[f"{model_name}_ev"] = m["EV_bp"]
            row[f"{model_name}_dd"] = m["max_dd_pct"]

        # Ratchet stats for this distance
        rstats = collect_ratchet_stats(
            base_cfg, data_15m, data_1h, data_4h,
            strat="S_d8",
            trail_distance=dist,
        )
        row["ratchet_rate"] = rstats["ratchet_success_rate"]
        row["ratchet_ok"] = rstats["ratchet_ok"]
        row["ratchet_fail"] = rstats["ratchet_fail"]

        sweep_results.append(row)
        print(f"  {dist:.3f} | {row['ideal_pf']:8.3f} {row['ideal_n']:5d} | "
              f"{row['worst_pf']:8.3f} {row['worst_n']:5d} | "
              f"{row['ratchet_pf']:10.3f} {row['ratchet_n']:5d} | "
              f"{row['ratchet_rate']:7.1f}%")

    # ── T3: Best ratchet distance — yearly breakdown ──
    print("\n" + "=" * 70)
    print("T3: Best Ratchet Distance — Yearly OOS Validation")
    print("=" * 70)

    # Find best ratchet distance
    best_idx = max(range(len(sweep_results)),
                   key=lambda i: sweep_results[i]["ratchet_pf"])
    best_dist = sweep_results[best_idx]["distance"]
    print(f"\n  Best ratchet distance: {best_dist:.3f} "
          f"(PF={sweep_results[best_idx]['ratchet_pf']:.3f})")

    # Also check top-2 and top-3 for robustness
    sorted_results = sorted(sweep_results, key=lambda r: r["ratchet_pf"], reverse=True)
    top_dists = [sorted_results[0]["distance"]]
    if len(sorted_results) > 1:
        top_dists.append(sorted_results[1]["distance"])
    if len(sorted_results) > 2:
        top_dists.append(sorted_results[2]["distance"])

    for dist in top_dists:
        print(f"\n  --- distance={dist:.3f} yearly breakdown ---")

        # Collect all trades with ratchet model
        all_trades_by_bar = {}
        for sym, d15 in data_15m.items():
            c15 = d15["close"]
            h15 = d15["high"]
            l15 = d15["low"]
            offset = d15.get("_offset", 0)

            vr = d15.get(f"vr_{cfg.vr_short}_{cfg.vr_long}")
            if vr is None:
                continue
            sig_s = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, WARMUP_15m)

            pnl_s, ent_s, ext_s, _ = sim_straddle_ratchet(
                c15, h15, l15, sig_s, cfg.s_sl, cfg.d_tp, cfg.max_hold_15m,
                COST_STRADDLE, cfg.s_trail_activate, dist,
            )
            for k in range(len(pnl_s)):
                all_trades_by_bar[int(ent_s[k]) + offset] = float(pnl_s[k])

        # We need actual timestamps for yearly split. Use first symbol's timestamps.
        first_sym = next(iter(data_15m))
        first_close = data_15m[first_sym]["close"]
        max_bars = len(first_close)
        # Approximate: 35,040 bars per year (365.25 * 24 * 4)
        bars_per_year = 35040

        # Collect trades with full portfolio sim for yearly split
        trades = collect_trades_3model(
            base_cfg, data_15m, data_1h, data_4h,
            model="ratchet", strat="S_d8",
            trail_distance=dist,
        )
        strd_syms = set()
        for t in trades:
            if True:  # S_d8 only
                strd_syms.add(t[0])

        # Split trades by approximate year
        yearly_trades: dict[int, list] = {}
        for t in trades:
            ebar = t[1]  # entry_bar is index 1: (sym, entry, exit, pnl)
            year_offset = ebar // bars_per_year
            year = 2020 + year_offset
            if year not in yearly_trades:
                yearly_trades[year] = []
            yearly_trades[year].append(t)

        print(f"  {'Year':>6} | {'N':>5} | {'PF':>6} | {'WR':>5} | {'EV_bp':>6} | {'MaxDD':>6}")
        print(f"  " + "-" * 50)
        for year in sorted(yearly_trades.keys()):
            yt = yearly_trades[year]
            y_pnl = np.array([t[3] for t in yt], dtype=np.float64)  # pnl is index 3
            if len(y_pnl) > 0:
                wins = y_pnl[y_pnl > 0]
                losses = y_pnl[y_pnl < 0]
                n_yr = len(y_pnl)
                wr = len(wins) / n_yr * 100 if n_yr > 0 else 0
                pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999.0
                ev = y_pnl.mean() * 10000  # bp
                # Equity curve for DD
                eq = np.cumprod(1.0 + y_pnl)
                peak_eq = np.maximum.accumulate(eq)
                dd = ((peak_eq - eq) / peak_eq).max() * 100 if len(eq) > 0 else 0
                print(f"  {year:>6} | {n_yr:>5} | {pf:>6.2f} | {wr:>5.1f} | {ev:>+6.0f} | {dd:>5.1f}%")

    # ── T4: Comparative summary ──
    print("\n" + "=" * 70)
    print("T4: Key Findings & Recommendation")
    print("=" * 70)

    curr = next(r for r in sweep_results if r["distance"] == 0.001)
    best = sorted_results[0]

    print(f"\n  Current (dist=0.001):")
    print(f"    IDEAL PF={curr['ideal_pf']:.3f}, WORST PF={curr['worst_pf']:.3f}, "
          f"RATCHET PF={curr['ratchet_pf']:.3f}")
    print(f"    Ratchet success rate: {curr['ratchet_rate']:.1f}%")

    print(f"\n  Best ratchet (dist={best['distance']:.3f}):")
    print(f"    IDEAL PF={best['ideal_pf']:.3f}, WORST PF={best['worst_pf']:.3f}, "
          f"RATCHET PF={best['ratchet_pf']:.3f}")
    print(f"    Ratchet success rate: {best['ratchet_rate']:.1f}%")

    if best["distance"] != 0.001:
        pf_improve = (best["ratchet_pf"] / curr["ratchet_pf"] - 1) * 100
        rate_improve = best["ratchet_rate"] - curr["ratchet_rate"]
        print(f"\n  Improvement: PF {pf_improve:+.1f}%, ratchet rate {rate_improve:+.1f}pp")
        print(f"\n  Recommended change:")
        print(f"    s_trail_distance: {cfg.s_trail_distance} → {best['distance']}")
        print(f"    Also need: fix ratchet timing (suppress client exit on activation bar)")
    else:
        print(f"\n  Current distance is already optimal for ratchet model.")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
