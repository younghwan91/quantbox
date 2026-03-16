"""r89: Trail Distance & Execution Gap — Independent Distance Sweep.

**목적:**
  라이브 거래에서 trail exit PnL이 백테스트보다 체계적으로 낮은 원인 분석.
  sim은 trail_stop 수준(peak-dist)으로 청산 가정, 라이브는 bar close에서 청산.
  평균 execution gap = -0.50% (trail distance 0.1%의 5배).
  s_trail_distance만 독립적으로 sweep하여 최적값을 찾는다.

**실험:**
  T1: Sim parity gap — bar-close exit model vs trail_stop exit model
  T2: Independent trail distance sweep (activate=0.015 고정, distance 변화)
  T3: Post-activation MFE 분석 — trail 활성화 후 가격이 얼마나 더 가는가?

사용:
    uv run python research/active/r86_r90/r89_trail_distance.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import numba

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    COST_DIRECTIONAL,
    G_3TF,
    G_4TF,
    bootstrap_pf,
    build_resampled,
    collect_trades,
    compute_metrics,
    get_straddle_syms,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)

# Reuse r88's data loading
from research.active.r86_r90.r88_edge_decay import load_all_data


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  T1: Bar-Close Exit Sim Functions
#  Same as production sim but trail exit uses close[j] instead of trail_stop
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_trailing_barclose(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with bar-close exit for trail stops.

    Identical to sim_straddle_trailing except trail exit PnL uses
    the actual bar close price instead of the trail_stop formula.
    This models live execution more realistically.
    """
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

        la = True
        sa = True
        le = 0.0
        se = 0.0
        l_peak = 0.0
        s_peak = 0.0
        l_trailing = False
        s_trailing = False
        l_exit_bar = -1
        s_exit_bar = -1
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            # Long leg — SL checked FIRST
            if la:
                if low[j] <= entry * (1.0 - sl):
                    le = -sl
                    la = False
                    l_exit_bar = j
                else:
                    l_gain = (high[j] - entry) / entry
                    if l_gain > l_peak:
                        l_peak = l_gain
                    if not l_trailing and l_peak >= trail_activate:
                        l_trailing = True
                    if l_trailing:
                        trail_lvl = entry * (1.0 + l_peak - trail_distance)
                        if low[j] <= trail_lvl:
                            # BAR-CLOSE EXIT: use close price, not trail_stop
                            le = (close[j] - entry) / entry
                            la = False
                            l_exit_bar = j
                    if la and high[j] >= entry * (1.0 + tp):
                        le = tp
                        la = False
                        l_exit_bar = j

            # Short leg — SL checked FIRST
            if sa:
                if high[j] >= entry * (1.0 + sl):
                    se = -sl
                    sa = False
                    s_exit_bar = j
                else:
                    s_gain = (entry - low[j]) / entry
                    if s_gain > s_peak:
                        s_peak = s_gain
                    if not s_trailing and s_peak >= trail_activate:
                        s_trailing = True
                    if s_trailing:
                        trail_lvl = entry * (1.0 - s_peak + trail_distance)
                        if high[j] >= trail_lvl:
                            # BAR-CLOSE EXIT: use close price, not trail_stop
                            se = (entry - close[j]) / entry
                            sa = False
                            s_exit_bar = j
                    if sa and low[j] <= entry * (1.0 - tp):
                        se = tp
                        sa = False
                        s_exit_bar = j

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
def sim_directional_trailing_barclose(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with bar-close exit for trail stops."""
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
        trailing_active = False
        peak_gain = 0.0

        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        # BAR-CLOSE EXIT
                        trade_pnl = (close[j] - entry) / entry
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
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        # BAR-CLOSE EXIT
                        trade_pnl = (entry - close[j]) / entry
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
#  T3: MFE Analysis Sim — Track post-activation MFE
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_straddle_mfe_analysis(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim that also tracks per-leg MFE after trail activation.

    Returns:
        (pnl, ent, ext, l_post_act_mfe, s_post_act_mfe)
        l/s_post_act_mfe: max gain of long/short leg AFTER trail activated.
            -1.0 if trail never activated for that leg.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    l_mfe_arr = np.empty(cap, dtype=np.float64)
    s_mfe_arr = np.empty(cap, dtype=np.float64)
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

        la = True
        sa = True
        le = 0.0
        se = 0.0
        l_peak = 0.0
        s_peak = 0.0
        l_trailing = False
        s_trailing = False
        l_post_mfe = -1.0
        s_post_mfe = -1.0
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if la:
                if low[j] <= entry * (1.0 - sl):
                    le = -sl
                    la = False
                else:
                    l_gain = (high[j] - entry) / entry
                    if l_gain > l_peak:
                        l_peak = l_gain
                    if not l_trailing and l_peak >= trail_activate:
                        l_trailing = True
                        l_post_mfe = l_peak
                    if l_trailing:
                        if l_peak > l_post_mfe:
                            l_post_mfe = l_peak
                        trail_lvl = entry * (1.0 + l_peak - trail_distance)
                        if low[j] <= trail_lvl:
                            le = max(l_peak - trail_distance, -sl)
                            la = False
                    if la and high[j] >= entry * (1.0 + tp):
                        le = tp
                        la = False

            if sa:
                if high[j] >= entry * (1.0 + sl):
                    se = -sl
                    sa = False
                else:
                    s_gain = (entry - low[j]) / entry
                    if s_gain > s_peak:
                        s_peak = s_gain
                    if not s_trailing and s_peak >= trail_activate:
                        s_trailing = True
                        s_post_mfe = s_peak
                    if s_trailing:
                        if s_peak > s_post_mfe:
                            s_post_mfe = s_peak
                        trail_lvl = entry * (1.0 - s_peak + trail_distance)
                        if high[j] >= trail_lvl:
                            se = max(s_peak - trail_distance, -sl)
                            sa = False
                    if sa and low[j] <= entry * (1.0 - tp):
                        se = tp
                        sa = False

            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
            if l_trailing and l_peak > l_post_mfe:
                l_post_mfe = l_peak
        if sa:
            se = (entry - close[eb]) / entry
            if s_trailing and s_peak > s_post_mfe:
                s_post_mfe = s_peak
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            l_mfe_arr[cnt] = l_post_mfe
            s_mfe_arr[cnt] = s_post_mfe
            cnt += 1
        i = eb + 1
    return pnl[:cnt], ent[:cnt], ext[:cnt], l_mfe_arr[:cnt], s_mfe_arr[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection (parametric)
# ══════════════════════════════════════════════════════════════════════════════


def _dedup(sym: str, trades: list) -> list[tuple[str, int, int, float]]:
    result = []
    last_exit = -1
    for item in sorted(trades, key=lambda t: t[0]):
        e = item[0]
        x = item[1]
        p = item[2]
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_trades_parametric(
    cfg_obj: VolStraddleConfig,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    *,
    tf_group: list | None = None,
    data_30m: dict | None = None,
    trail_act: float | None = None,
    trail_dist: float | None = None,
    use_barclose: bool = False,
) -> list[tuple[str, int, int, float]]:
    """Collect trades with optional bar-close exit model."""
    sp = cfg_obj.vr_short
    lp = cfg_obj.vr_long
    vr_key = f"vr_{sp}_{lp}"
    is_straddle = strat == "S_d8"

    d_sl = cfg_obj.d_sl
    s_sl = cfg_obj.s_sl
    d_tp = cfg_obj.d_tp
    s_tp = cfg_obj.s_tp
    d_vr = cfg_obj.d_vr_threshold
    s_vr = cfg_obj.s_vr_threshold
    s_dur = cfg_obj.s_squeeze_dur

    if is_straddle:
        _trail_act = trail_act if trail_act is not None else cfg_obj.s_trail_activate
        _trail_dist = trail_dist if trail_dist is not None else cfg_obj.s_trail_distance
    else:
        _trail_act = trail_act if trail_act is not None else cfg_obj.d_trail_activate
        _trail_dist = trail_dist if trail_dist is not None else cfg_obj.d_trail_distance

    active_tfs = tf_group if tf_group is not None else G_3TF
    tf_data_map: dict[int, dict] = {1: data_15m, 4: data_1h, 16: data_4h}
    if any(ratio == 2 for _, ratio, *_ in active_tfs):
        if data_30m is None:
            data_30m = build_resampled(data_15m, 2)
            precompute_all_vr([data_30m], [(sp, lp)])
            precompute(data_30m, vr_sp=sp, vr_lp=lp)
        tf_data_map[2] = data_30m

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []

        for tf_label, ratio, warmup, mh_d, mh_s in active_tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            sl = s_sl if is_straddle else d_sl
            tp = s_tp if is_straddle else d_tp
            mh = mh_s if is_straddle else mh_d
            cost = COST_STRADDLE if is_straddle else COST_DIRECTIONAL

            if is_straddle:
                sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                if use_barclose:
                    p, e, x = sim_straddle_trailing_barclose(
                        c, h, lo, sig, sl, tp, mh, cost, _trail_act, _trail_dist
                    )
                else:
                    p, e, x = sim_straddle_trailing(
                        c, h, lo, sig, sl, tp, mh, cost, _trail_act, _trail_dist
                    )
            else:
                sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                if use_barclose:
                    p, e, x = sim_directional_trailing_barclose(
                        c, h, lo, sig, dire, sl, tp, mh, cost, _trail_act, _trail_dist
                    )
                else:
                    p, e, x = sim_directional_trailing(
                        c, h, lo, sig, dire, sl, tp, mh, cost, _trail_act, _trail_dist
                    )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio + offset
                exit_15m = int(x[k]) * ratio + (ratio - 1) + offset
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        all_trades.extend(_dedup(sym, sym_trades))

    return all_trades


def collect_mfe_data(
    cfg_obj: VolStraddleConfig,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    *,
    tf_group: list | None = None,
    data_30m: dict | None = None,
) -> dict:
    """Collect post-activation MFE data for straddle trades."""
    sp = cfg_obj.vr_short
    lp = cfg_obj.vr_long
    vr_key = f"vr_{sp}_{lp}"
    s_sl = cfg_obj.s_sl
    s_tp = cfg_obj.s_tp
    s_vr = cfg_obj.s_vr_threshold
    s_dur = cfg_obj.s_squeeze_dur
    trail_act = cfg_obj.s_trail_activate
    trail_dist = cfg_obj.s_trail_distance

    active_tfs = tf_group if tf_group is not None else G_3TF
    tf_data_map: dict[int, dict] = {1: data_15m, 4: data_1h, 16: data_4h}
    if any(ratio == 2 for _, ratio, *_ in active_tfs):
        if data_30m is None:
            data_30m = build_resampled(data_15m, 2)
            precompute_all_vr([data_30m], [(sp, lp)])
            precompute(data_30m, vr_sp=sp, vr_lp=lp)
        tf_data_map[2] = data_30m

    all_l_mfe: list[float] = []
    all_s_mfe: list[float] = []

    for sym in sorted(data_15m.keys()):
        for tf_label, ratio, warmup, mh_d, mh_s in active_tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            _, _, _, l_mfe, s_mfe = sim_straddle_mfe_analysis(
                c, h, lo, sig, s_sl, s_tp, mh_s, COST_STRADDLE, trail_act, trail_dist
            )

            for v in l_mfe:
                if v >= 0:
                    all_l_mfe.append(v)
            for v in s_mfe:
                if v >= 0:
                    all_s_mfe.append(v)

    return {"long_post_act_mfe": np.array(all_l_mfe), "short_post_act_mfe": np.array(all_s_mfe)}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 80)
    print("r89: Trail Distance & Execution Gap Investigation")
    print("=" * 80)

    cfg = VolStraddleConfig()
    commit = get_commit()
    print(f"\nCommit: {commit}")
    print(f"s_trail_activate = {cfg.s_trail_activate} ({cfg.s_trail_activate*100:.1f}%)")
    print(f"s_trail_distance = {cfg.s_trail_distance} ({cfg.s_trail_distance*100:.2f}%)")
    print(f"s_sl             = {cfg.s_sl} ({cfg.s_sl*100:.1f}%)")
    print(f"Asymmetry: SL={cfg.s_sl*100:.1f}% / trail_dist={cfg.s_trail_distance*100:.2f}% = {cfg.s_sl/cfg.s_trail_distance:.0f}:1")

    # ── Phase 0: Data Loading ────────────────────────────────────────────
    print("\n── Phase 0: Loading data ──")
    data_15m, data_30m, data_1h, data_4h, data_years, max_bars = load_all_data(cfg)
    print(f"  data_years = {data_years:.2f}")

    # ── Phase 1: Baseline ────────────────────────────────────────────────
    print("\n── Phase 1: Baseline (current params) ──")
    strd_trades_ts = collect_trades_parametric(
        cfg, data_15m, data_1h, data_4h, "S_d8",
        tf_group=G_4TF, data_30m=data_30m,
        use_barclose=False,
    )
    dmom_trades_ts = collect_trades_parametric(
        cfg, data_15m, data_1h, data_4h, "D_mom8",
        tf_group=G_3TF, data_30m=data_30m,
        use_barclose=False,
    )

    # Combined (production: D+S)
    strd_syms = get_straddle_syms(strd_trades_ts)
    all_trades_ts = dmom_trades_ts + strd_trades_ts
    combined_pnl_ts = portfolio_sim(all_trades_ts, 10, straddle_syms=strd_syms)
    m_ts = compute_metrics(combined_pnl_ts, data_years)

    print(f"  [trail_stop exit] Combined D+S: N={m_ts['N']}, PF={m_ts['PF']:.3f}, "
          f"WR={m_ts['WR']:.1f}%, MaxDD={m_ts['max_dd_pct']:.1f}%")

    # ── T1: Bar-Close Parity Gap ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("T1: Sim Parity Gap — trail_stop vs bar_close exit model")
    print("=" * 80)

    strd_trades_bc = collect_trades_parametric(
        cfg, data_15m, data_1h, data_4h, "S_d8",
        tf_group=G_4TF, data_30m=data_30m,
        use_barclose=True,
    )
    dmom_trades_bc = collect_trades_parametric(
        cfg, data_15m, data_1h, data_4h, "D_mom8",
        tf_group=G_3TF, data_30m=data_30m,
        use_barclose=True,
    )

    # S_d8 standalone comparison
    s_pnl_ts = portfolio_sim(strd_trades_ts, 5)
    s_m_ts = compute_metrics(s_pnl_ts, data_years)
    s_pnl_bc = portfolio_sim(strd_trades_bc, 5)
    s_m_bc = compute_metrics(s_pnl_bc, data_years)

    print(f"\n  S_d8 standalone (max_concurrent=5):")
    print(f"  {'Model':<18s} {'N':>6s} {'PF':>7s} {'WR%':>6s} {'EV(bp)':>8s} {'MaxDD%':>8s} {'AvgWin%':>9s} {'AvgLoss%':>10s}")
    print(f"  {'trail_stop':<18s} {s_m_ts['N']:>6d} {s_m_ts['PF']:>7.3f} {s_m_ts['WR']:>6.1f} {s_m_ts['EV_bp']:>8.1f} {s_m_ts['max_dd_pct']:>8.1f} {s_m_ts['avg_win_pct']:>9.3f} {s_m_ts['avg_loss_pct']:>10.3f}")
    print(f"  {'bar_close':<18s} {s_m_bc['N']:>6d} {s_m_bc['PF']:>7.3f} {s_m_bc['WR']:>6.1f} {s_m_bc['EV_bp']:>8.1f} {s_m_bc['max_dd_pct']:>8.1f} {s_m_bc['avg_win_pct']:>9.3f} {s_m_bc['avg_loss_pct']:>10.3f}")

    pf_gap = (s_m_bc['PF'] - s_m_ts['PF']) / s_m_ts['PF'] * 100
    ev_gap = s_m_bc['EV_bp'] - s_m_ts['EV_bp']
    print(f"\n  PF gap: {pf_gap:+.1f}%  (bar_close is {'worse' if pf_gap < 0 else 'better'})")
    print(f"  EV gap: {ev_gap:+.1f}bp")

    # D_mom8 standalone comparison
    d_pnl_ts = portfolio_sim(dmom_trades_ts, 10)
    d_m_ts = compute_metrics(d_pnl_ts, data_years)
    d_pnl_bc = portfolio_sim(dmom_trades_bc, 10)
    d_m_bc = compute_metrics(d_pnl_bc, data_years)

    print(f"\n  D_mom8 standalone:")
    print(f"  {'trail_stop':<18s} {d_m_ts['N']:>6d} {d_m_ts['PF']:>7.3f} {d_m_ts['WR']:>6.1f} {d_m_ts['EV_bp']:>8.1f} {d_m_ts['max_dd_pct']:>8.1f}")
    print(f"  {'bar_close':<18s} {d_m_bc['N']:>6d} {d_m_bc['PF']:>7.3f} {d_m_bc['WR']:>6.1f} {d_m_bc['EV_bp']:>8.1f} {d_m_bc['max_dd_pct']:>8.1f}")

    # Combined comparison
    strd_syms_bc = get_straddle_syms(strd_trades_bc)
    combined_pnl_bc = portfolio_sim(dmom_trades_bc + strd_trades_bc, 10, straddle_syms=strd_syms_bc)
    m_bc = compute_metrics(combined_pnl_bc, data_years)

    print(f"\n  Combined D+S (max_concurrent=10):")
    print(f"  {'trail_stop':<18s} {m_ts['N']:>6d} {m_ts['PF']:>7.3f} {m_ts['WR']:>6.1f} {m_ts['EV_bp']:>8.1f} {m_ts['max_dd_pct']:>8.1f}")
    print(f"  {'bar_close':<18s} {m_bc['N']:>6d} {m_bc['PF']:>7.3f} {m_bc['WR']:>6.1f} {m_bc['EV_bp']:>8.1f} {m_bc['max_dd_pct']:>8.1f}")

    # Yearly comparison for S_d8
    print(f"\n  S_d8 Yearly PF comparison:")
    bars_per_year = 4 * 24 * 365
    for year in range(2020, 2026):
        yr_start = (year - 2020) * bars_per_year
        yr_end = yr_start + bars_per_year
        ts_yr = [p for p in s_pnl_ts if True]  # we need actual yearly splits
        # Use raw trades for yearly
        ts_yr_trades = [(s, e, x, p) for s, e, x, p in strd_trades_ts
                        if yr_start <= e < yr_end]
        bc_yr_trades = [(s, e, x, p) for s, e, x, p in strd_trades_bc
                        if yr_start <= e < yr_end]
        if not ts_yr_trades:
            continue
        ts_yr_pnl = portfolio_sim(ts_yr_trades, 5)
        bc_yr_pnl = portfolio_sim(bc_yr_trades, 5)
        if len(ts_yr_pnl) < 5:
            continue
        ts_m = compute_metrics(ts_yr_pnl, 1.0)
        bc_m = compute_metrics(bc_yr_pnl, 1.0)
        diff = (bc_m['PF'] - ts_m['PF']) / ts_m['PF'] * 100
        print(f"  {year}: trail_stop PF={ts_m['PF']:.3f}  bar_close PF={bc_m['PF']:.3f}  gap={diff:+.1f}%  N={ts_m['N']}")

    # ── T2: Independent Distance Sweep ───────────────────────────────────
    print("\n" + "=" * 80)
    print("T2: Independent Trail Distance Sweep (s_trail_activate=0.015 FIXED)")
    print("=" * 80)

    distances = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.008]

    print(f"\n  {'dist':>8s} {'model':<12s} {'N':>6s} {'PF':>7s} {'WR%':>6s} {'EV(bp)':>8s} {'MaxDD%':>8s} {'AvgWin%':>9s} {'2023-25 PF':>11s} {'2023-25 N':>10s}")
    print("  " + "-" * 100)

    bars_per_year = 4 * 24 * 365
    yr_2023_start = (2023 - 2020) * bars_per_year

    for dist in distances:
        for use_bc, model_name in [(False, "trail_stop"), (True, "bar_close")]:
            strd_t = collect_trades_parametric(
                cfg, data_15m, data_1h, data_4h, "S_d8",
                tf_group=G_4TF, data_30m=data_30m,
                trail_dist=dist, use_barclose=use_bc,
            )
            pnl_arr = portfolio_sim(strd_t, 5)
            m = compute_metrics(pnl_arr, data_years)

            # 2023-2025 subset
            recent_trades = [(s, e, x, p) for s, e, x, p in strd_t if e >= yr_2023_start]
            if recent_trades:
                recent_pnl = portfolio_sim(recent_trades, 5)
                recent_m = compute_metrics(recent_pnl, 3.0)
                recent_pf = f"{recent_m['PF']:.3f}"
                recent_n = f"{recent_m['N']}"
            else:
                recent_pf = "N/A"
                recent_n = "0"

            marker = " ◀ current" if abs(dist - cfg.s_trail_distance) < 1e-6 and not use_bc else ""
            if abs(dist - cfg.s_trail_distance) < 1e-6 and use_bc:
                marker = " ◀ current+bc"
            print(f"  {dist:>8.4f} {model_name:<12s} {m['N']:>6d} {m['PF']:>7.3f} {m['WR']:>6.1f} "
                  f"{m['EV_bp']:>8.1f} {m['max_dd_pct']:>8.1f} {m['avg_win_pct']:>9.3f} "
                  f"{recent_pf:>11s} {recent_n:>10s}{marker}")

    # ── T3: Post-Activation MFE Analysis ─────────────────────────────────
    print("\n" + "=" * 80)
    print("T3: Post-Activation MFE Analysis")
    print("=" * 80)

    mfe_data = collect_mfe_data(
        cfg, data_15m, data_1h, data_4h,
        tf_group=G_4TF, data_30m=data_30m,
    )

    for label, mfe_arr in [("Long legs", mfe_data["long_post_act_mfe"]),
                           ("Short legs", mfe_data["short_post_act_mfe"])]:
        activated = mfe_arr[mfe_arr >= 0]
        if len(activated) == 0:
            continue
        extension = (activated - cfg.s_trail_activate) * 100  # how far beyond activate

        print(f"\n  {label}: {len(activated)} legs activated trail (of {len(mfe_arr)} total legs)")
        print(f"  Post-activation MFE (peak gain after trail activated):")
        print(f"    Mean:   {np.mean(activated)*100:.3f}%")
        print(f"    Median: {np.median(activated)*100:.3f}%")
        print(f"    P25:    {np.percentile(activated, 25)*100:.3f}%")
        print(f"    P75:    {np.percentile(activated, 75)*100:.3f}%")
        print(f"    P90:    {np.percentile(activated, 90)*100:.3f}%")
        print(f"    Max:    {np.max(activated)*100:.3f}%")

        print(f"\n  Extension beyond activate ({cfg.s_trail_activate*100:.1f}%):")
        print(f"    Mean:   {np.mean(extension):.3f}%")
        print(f"    Median: {np.median(extension):.3f}%")

        # What % would benefit from wider distance?
        for wider_dist in [0.002, 0.003, 0.005]:
            # A leg benefits from wider distance if its peak exceeds activate + wider_dist
            # (meaning the trail wouldn't have fired at the tighter distance's trigger point)
            # But it's complex — let's just show extension distribution
            pct_above = np.mean(extension > wider_dist * 100) * 100
            print(f"    Extension > {wider_dist*100:.1f}% (would benefit from dist={wider_dist}): {pct_above:.1f}%")

    # ── T2b: Combined D+S sweep with bar_close ───────────────────────────
    print("\n" + "=" * 80)
    print("T2b: Combined D+S Sweep (bar_close model, straddle dist varies, D fixed)")
    print("=" * 80)

    # D_mom8 with bar-close (fixed, reuse)
    dmom_bc = dmom_trades_bc

    print(f"\n  {'s_dist':>8s} {'N':>6s} {'PF':>7s} {'WR%':>6s} {'EV(bp)':>8s} {'MaxDD%':>8s} {'2023-25 PF':>11s}")
    print("  " + "-" * 70)

    for dist in distances:
        strd_t = collect_trades_parametric(
            cfg, data_15m, data_1h, data_4h, "S_d8",
            tf_group=G_4TF, data_30m=data_30m,
            trail_dist=dist, use_barclose=True,
        )
        strd_s = get_straddle_syms(strd_t)
        combined = dmom_bc + strd_t
        pnl_arr = portfolio_sim(combined, 10, straddle_syms=strd_s)
        m = compute_metrics(pnl_arr, data_years)

        recent_trades = [(s, e, x, p) for s, e, x, p in strd_t if e >= yr_2023_start]
        recent_d = [(s, e, x, p) for s, e, x, p in dmom_bc if e >= yr_2023_start]
        if recent_trades:
            recent_strd_s = get_straddle_syms(recent_trades)
            recent_pnl = portfolio_sim(recent_d + recent_trades, 10, straddle_syms=recent_strd_s)
            recent_m = compute_metrics(recent_pnl, 3.0)
            recent_pf = f"{recent_m['PF']:.3f}"
        else:
            recent_pf = "N/A"

        marker = " ◀ current" if abs(dist - cfg.s_trail_distance) < 1e-6 else ""
        print(f"  {dist:>8.4f} {m['N']:>6d} {m['PF']:>7.3f} {m['WR']:>6.1f} "
              f"{m['EV_bp']:>8.1f} {m['max_dd_pct']:>8.1f} {recent_pf:>11s}{marker}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Trail params: activate={cfg.s_trail_activate*100:.1f}%, distance={cfg.s_trail_distance*100:.2f}%")
    print(f"  SL = {cfg.s_sl*100:.1f}%")
    print(f"  Asymmetry ratio: {cfg.s_sl/cfg.s_trail_distance:.0f}:1")
    print(f"\n  T1: Sim parity gap (trail_stop vs bar_close):")
    print(f"    S_d8 PF: {s_m_ts['PF']:.3f} → {s_m_bc['PF']:.3f} ({pf_gap:+.1f}%)")
    print(f"  T2: See distance sweep above")
    print(f"  T3: See MFE analysis above")


if __name__ == "__main__":
    main()
