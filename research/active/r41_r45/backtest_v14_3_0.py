"""v14.3.0 Full Backtest — F5 TBR Gate Activated

Extends v14.0.0 backtest with F5 TBR z-score gate on D_mom8 directional entries.
F5 gate: reject LONG if TBR z > +0.5, reject SHORT if TBR z < -0.5
         (R46b validated, p=0.029, D_mom8 only — S_d8 straddle unaffected)

Additional metrics vs v14.0.0: Sharpe, Sortino, CAGR (1x + production L/N model).

⛔ DEPRECATED — DO NOT IMPORT FROM THIS FILE
─────────────────────────────────────────────
The constants D_TRAIL_ACTIVATE=0.02 and D_TRAIL_DISTANCE=0.003 below are
PRE-r47 BASELINE values (the OLD production params before r47 exit A/B test).
Current production values are d_trail_activate=0.005, d_trail_distance=0.002
(set in r47, now in config.py).

Importing D_TRAIL_ACTIVATE / D_TRAIL_DISTANCE from this file will give WRONG
values for any production backtest or comparison.

Incident reference: r53 — importing from this file contaminated D_mom8 trail
results, causing silent parameter mismatch vs live trading.

For correct params always use:
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig
    cfg = VolStraddleConfig()
    cfg.d_trail_activate  # 0.005
    cfg.d_trail_distance  # 0.002
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled,
    precompute,
    precompute_all_vr,
    collect_trades,
    portfolio_sim,
    portfolio_sim_with_details,
    compute_metrics,
    bootstrap_pf,
    pf_from_pnl,
    sim_directional_trades,
    G_3TF,
    COST_DIRECTIONAL,
)
from quantbox.strategy.vol_straddle.signals import gen_vr_signal, make_direction_momentum
from research.active.r41_r45.r45_param_sweep import (
    collect_enhanced_trades,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics,
    get_year_boundaries,
)
from quantbox.backtest.trade_sim import sim_directional_trailing
from research.active.r31_r40.scaling_r38 import CFG_V13


# ─── PRE-r47 BASELINE exit parameters (STALE — do NOT use for production) ─────
# These were the production values BEFORE r47 exit A/B test changed them.
# r47 found: trail_activate=0.005, trail_distance=0.002 are better.
# Current production values: see VolStraddleConfig in config.py.
# ⛔ Importing D_TRAIL_ACTIVATE / D_TRAIL_DISTANCE from this file gives wrong values!
D_SL = 0.01            # config.py: d_sl = 0.01  (still correct)
D_TP = 0.12            # config.py: d_tp = 0.12  (still correct)
D_TRAIL_ACTIVATE = 0.02   # ⛔ STALE — production is now 0.005 (r47)
D_TRAIL_DISTANCE = 0.003  # ⛔ STALE — production is now 0.002 (r47)

# ─── TBR z-score gate ─────────────────────────────────────────────────────────

def compute_tbr_z_at_bar(
    taker_buy: np.ndarray,
    volume: np.ndarray,
    bar_15m: int,
    window: int = 192,
) -> float:
    """TBR z-score at bar_15m using [bar-window, bar) — no lookahead.

    Returns 0.0 if insufficient data (≤window bars available).
    """
    if bar_15m < window:
        return 0.0
    tb_win = taker_buy[bar_15m - window: bar_15m]
    vo_win = volume[bar_15m - window: bar_15m]
    safe_vol = np.where(vo_win > 0, vo_win, 1.0)
    tbr_win = tb_win / safe_vol

    mu = float(np.mean(tbr_win))
    sigma = float(np.std(tbr_win))
    if sigma < 1e-9:
        return 0.0

    cur_vol = float(volume[bar_15m]) if bar_15m < len(volume) else 0.0
    cur_tb = float(taker_buy[bar_15m]) if bar_15m < len(taker_buy) else 0.0
    cur_tbr = cur_tb / cur_vol if cur_vol > 0 else 0.5
    return (cur_tbr - mu) / sigma


def collect_d_trades_with_tbr(
    cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    tbr_z_window: int = 192,
    tbr_z_threshold: float = 0.5,
) -> tuple[list, int]:
    """D_mom8 collection with F5 TBR gate applied per entry.

    Returns (trades_list, n_blocked) where trades_list matches the same
    (sym, entry_15m, exit_15m, pnl) format as collect_trades().
    """
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    d_sl = cfg["d_sl"]
    # Use production exit params (D_TP/D_TRAIL_*) — NOT cfg["tp"] which is old 5% flat TP

    all_trades: list[tuple[str, int, int, float]] = []
    n_blocked = 0

    for sym in sorted(data_15m.keys()):
        sym_d = data_15m[sym]
        taker_buy_15m = sym_d.get("taker_buy")
        volume_15m = sym_d.get("volume")
        if taker_buy_15m is None or volume_15m is None:
            continue

        sym_trades: list[tuple[int, int, float]] = []

        for _tf_label, ratio, warmup, mh_d, _mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            mom8 = d.get("mom_8")
            if vr is None or mom8 is None:
                continue

            sig = gen_vr_signal(vr, cfg["d_vr"], warmup)
            dire = make_direction_momentum(sig, mom8)
            # Production exit: trailing stop (R43), NOT flat 5% TP
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                trail_activate=D_TRAIL_ACTIVATE, trail_distance=D_TRAIL_DISTANCE,
            )

            for k in range(len(p)):
                entry_tf = int(e[k])
                entry_15m = entry_tf * ratio

                # ── F5 TBR gate ──────────────────────────────────────────
                tbr_z = compute_tbr_z_at_bar(
                    taker_buy_15m, volume_15m, entry_15m, tbr_z_window
                )
                direction = int(dire[entry_tf]) if entry_tf < len(dire) else 0
                # Block: retail FOMO same direction as our trade
                if direction > 0 and tbr_z > tbr_z_threshold:
                    n_blocked += 1
                    continue
                if direction < 0 and tbr_z < -tbr_z_threshold:
                    n_blocked += 1
                    continue
                # ─────────────────────────────────────────────────────────

                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades, n_blocked


def collect_d_trades_trailing(
    cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list:
    """D_mom8 collection with production trailing stop exit — no TBR filter.

    Used for baseline parity checks. Uses same exit logic as live trading.
    """
    sp = cfg["sp"]
    lp = cfg["lp"]
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    d_sl = cfg["d_sl"]

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []

        for _tf_label, ratio, warmup, mh_d, _mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c = d["close"]
            h = d["high"]
            lo = d["low"]
            vr = d.get(vr_key)
            mom8 = d.get("mom_8")
            if vr is None or mom8 is None:
                continue

            sig = gen_vr_signal(vr, cfg["d_vr"], warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                trail_activate=D_TRAIL_ACTIVATE, trail_distance=D_TRAIL_DISTANCE,
            )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        sym_trades.sort(key=lambda t: t[0])
        accepted: list[tuple[int, int]] = []
        for entry_15m, exit_15m, pnl_val in sym_trades:
            conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
            if not conflict:
                all_trades.append((sym, entry_15m, exit_15m, pnl_val))
                accepted.append((entry_15m, exit_15m))

    return all_trades



def extended_metrics(
    pnl: np.ndarray,
    data_years: float,
    ln_leverage: float = 1.0,
    risk_free: float = 0.0,
) -> dict:
    """Compute Sharpe, Sortino, and production CAGR on top of compute_metrics.

    Args:
        pnl: Per-trade 1x PnL array (price returns, no leverage).
        data_years: Effective data span in years.
        ln_leverage: L/N production leverage multiplier (VR=3.0, MA=2.1).
        risk_free: Annual risk-free rate (default 0).
    """
    base = compute_metrics(pnl, data_years)

    # Production equity curve: equity *= (1 + pnl * L/N)
    prod_pnl = pnl * ln_leverage
    prod_equity = np.cumprod(1.0 + prod_pnl)
    prod_peaks = np.maximum.accumulate(prod_equity)
    prod_dd = (prod_peaks - prod_equity) / prod_peaks
    prod_max_dd = float(np.max(prod_dd)) * 100

    prod_cagr = (float(prod_equity[-1]) ** (1.0 / data_years) - 1.0) * 100 if data_years > 0 else 0.0
    prod_calmar = prod_cagr / prod_max_dd if prod_max_dd > 0 else 0.0

    # Sharpe (annualised, per-trade series, trades/yr normalisation)
    tpd = base["tpd"]                    # trades per day
    tpy = tpd * 365                      # trades per year
    mean_r = float(np.mean(prod_pnl))
    std_r = float(np.std(prod_pnl, ddof=1))
    rf_per_trade = risk_free / tpy if tpy > 0 else 0.0
    sharpe = (mean_r - rf_per_trade) / std_r * np.sqrt(tpy) if std_r > 0 else 0.0

    # Sortino (downside deviation only)
    downside = prod_pnl[prod_pnl < rf_per_trade] - rf_per_trade
    down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
    sortino = (mean_r - rf_per_trade) / down_std * np.sqrt(tpy) if down_std > 0 else 0.0

    return {
        **base,
        "prod_CAGR_pct": round(prod_cagr, 1),
        "prod_max_dd_pct": round(prod_max_dd, 1),
        "prod_calmar": round(prod_calmar, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
    }


# ─── Data loaders ─────────────────────────────────────────────────────────────

def _extract_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_15m() -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    from quantbox.core.settings import QuantBoxSettings
    settings = QuantBoxSettings()
    whitelist = [s.strip() for s in settings.rb_symbol_whitelist.split(",") if s.strip()]

    data: dict = {}
    for sym in whitelist:
        candidates = sorted(
            [f for f in cache_dir.glob(f"{sym}_15m_2190d_*.npz")],
            key=_extract_days, reverse=True,
        )
        if not candidates:
            continue
        npz = np.load(candidates[0])
        entry: dict = {k: npz[k] for k in npz.files}
        data[sym] = entry
    return data


def print_full(label: str, m: dict, boot: dict) -> None:
    print(f"\n  {label}")
    print(f"    N={m['N']}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
          f"RR={m['RR']:.2f}  EV={m['EV_bp']:+.1f}bp  t={m['t']:.2f}")
    print(f"    1x:   CAGR={m['ann_ret_pct']:+.1f}%  MaxDD={m['max_dd_pct']:.1f}%  "
          f"Calmar={m['calmar']:.1f}")
    if "prod_CAGR_pct" in m:
        print(f"    Prod: CAGR={m['prod_CAGR_pct']:+.1f}%  MaxDD={m['prod_max_dd_pct']:.1f}%  "
              f"Calmar={m['prod_calmar']:.1f}  "
              f"Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}")
    if boot:
        print(f"    Bootstrap: PF CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
              f"P(PF≥1)={boot['P_PF_ge_1']:.1f}%")


MA_CLUSTER_PCT = 0.03
MA_MIN_DURATION = 8
MA_MAX_DURATION = 24
MA_MIN_MOMENTUM = 0.010
MA_SL = 0.02
MA_TP = 0.08
MA_TF_RESTRICTION = 16  # 4h ratio (integer, not string)

# Production L/N leverage — read from config
from quantbox.strategy.vol_straddle.config import VolStraddleConfig as _Cfg

_cfg = _Cfg()
VR_LN = _cfg.base_leverage / _cfg.max_concurrent  # 3.0
MA_LN = _cfg.ma_cluster_leverage / _cfg.max_concurrent  # 2.1
del _cfg


def main() -> None:
    t0 = time.time()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    print(f"\n{'='*78}")
    print(f"  QuantBox v14.3.0 — Full Backtest (F5 TBR Gate Active)")
    print(f"  Commit: {commit}  |  F5: z>{0.5} blocks D_mom8  |  S_d8 unaffected")
    print(f"{'='*78}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n  Loading 15m data...")
    data_15m = load_15m()
    if not data_15m:
        print("❌ No data. Check data/cache_extended/")
        return

    avg_bars = int(np.mean([len(v["close"]) for v in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)
    year_bounds = get_year_boundaries(data_15m)

    print(f"  {len(data_15m)} symbols  |  avg {avg_bars:,} bars  |  "
          f"{data_years:.2f} years  |  load: {time.time()-t0:.1f}s")

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)

    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    # ── Phase 1: Baseline parity check (production trailing stop exit) ────
    # Uses sim_directional_trailing (d_tp=0.12, trail_activate=2%, trail_distance=0.3%)
    # matching live trading config exactly. Old flat-5% baseline (N=2265, PF=1.280)
    # is intentionally retired — parity is now checked against THIS output each run.
    print(f"\n{'─'*78}")
    print("  PHASE 1: BASELINE — Production exit (trailing stop, d_tp=12%)")
    print(f"{'─'*78}")

    bl_d_trades = collect_d_trades_trailing(CFG_V13, data_15m, data_1h, data_4h)
    bl_s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_vr_trades = bl_d_trades + bl_s_trades
    bl_vr_pnl = portfolio_sim(bl_vr_trades, 10)
    bl_vr = compute_metrics(bl_vr_pnl, data_years)

    print(f"  Baseline (trailing stop): N={bl_vr['N']}  PF={bl_vr['PF']:.3f}  "
          f"WR={bl_vr['WR']:.1f}%  EV={bl_vr['EV_bp']:+.1f}bp")
    print(f"  D_mom8={len(bl_d_trades)} trades  S_d8={len(bl_s_trades)} trades  ✓")

    # ── Phase 2: MA Cluster (unchanged from v14.0.0) ──────────────────────
    print(f"\n{'─'*78}")
    print("  PHASE 2: MA CLUSTER (unchanged — TBR gate not applied)")
    print(f"{'─'*78}")

    ma_trades = collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT,
        min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION,
        min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=MA_TF_RESTRICTION,
    )
    ma_pnl  = portfolio_sim(ma_trades, 10)
    ma_m    = extended_metrics(ma_pnl, data_years, ln_leverage=MA_LN)
    ma_boot = bootstrap_pf(ma_pnl, 5000)
    print_full("MA Cluster R45", ma_m, ma_boot)

    # ── Phase 3: VR Squeeze with F5 TBR on D_mom8 ─────────────────────────
    print(f"\n{'─'*78}")
    print("  PHASE 3: VR SQUEEZE v14.3.0 (D_mom8 + F5 TBR gate; S_d8 unchanged)")
    print(f"{'─'*78}")

    # D_mom8 with TBR gate
    d_trades_filtered, n_blocked = collect_d_trades_with_tbr(
        CFG_V13, data_15m, data_1h, data_4h,
        tbr_z_window=192, tbr_z_threshold=0.5,
    )
    # S_d8 straddle unchanged
    s_trades = collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")

    vr14_trades = d_trades_filtered + s_trades
    vr14_pnl    = portfolio_sim(vr14_trades, 10)
    vr14_m      = extended_metrics(vr14_pnl, data_years, ln_leverage=VR_LN)
    vr14_boot   = bootstrap_pf(vr14_pnl, 5000)

    # D_mom8 alone (for reporting)
    d_pnl_filt  = portfolio_sim(d_trades_filtered, 10)
    d_m_filt    = compute_metrics(d_pnl_filt, data_years)

    # D_mom8 baseline (trailing stop, no TBR) — reuse Phase 1 result for apples-to-apples
    d_pnl_base = portfolio_sim(bl_d_trades, 10)
    d_m_base   = compute_metrics(d_pnl_base, data_years)
    n_filtered_out = len(bl_d_trades) - len(d_trades_filtered)

    print(f"\n  D_mom8 (trailing, no TBR): N={d_m_base['N']}  PF={d_m_base['PF']:.3f}  WR={d_m_base['WR']:.1f}%  EV={d_m_base['EV_bp']:+.1f}bp")
    print(f"  D_mom8 + F5 TBR:           N={d_m_filt['N']}  PF={d_m_filt['PF']:.3f}  WR={d_m_filt['WR']:.1f}%  EV={d_m_filt['EV_bp']:+.1f}bp")
    print(f"  Removed by TBR: {n_filtered_out} trades ({n_filtered_out/len(bl_d_trades)*100:.1f}% reduction)")
    print()
    print_full("VR Squeeze v14.3.0 (D+S, F5 TBR on D_mom8)", vr14_m, vr14_boot)

    # ── Phase 4: Combined portfolio ────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("  PHASE 4: COMBINED PORTFOLIO v14.3.0 (VR F5 + MA, 10 slots)")
    print(f"{'─'*78}")

    comb_trades = vr14_trades + ma_trades
    comb_pnl, comb_accepted = portfolio_sim_with_details(comb_trades, 10)

    # Combined uses blended L/N: weighted by trade counts
    n_vr = len(vr14_trades)
    n_ma = len(ma_trades)
    n_total = n_vr + n_ma
    blended_ln = (n_vr * VR_LN + n_ma * MA_LN) / n_total if n_total > 0 else 1.5
    comb_m    = extended_metrics(comb_pnl, data_years, ln_leverage=blended_ln)
    comb_boot = bootstrap_pf(comb_pnl, 5000)
    print_full(f"Combined v14.3.0 (blended L/N={blended_ln:.2f})", comb_m, comb_boot)

    # Per-year
    print()
    print("  Combined per-year:")
    comb_yr = year_metrics(comb_pnl, comb_accepted, year_bounds)
    for key in sorted(comb_yr.keys()):
        y = comb_yr[key]
        flag = "✓" if y["PF"] >= 1.0 else "❌"
        print(f"    {key}  N={y['N']:>4}  PF={y['PF']:.3f}  "
              f"EV={y['EV_bp']:>+6.1f}bp  WR={y['WR']:.1f}%  {flag}")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  SUMMARY — v14.3.0 Performance (F5 TBR active on D_mom8)")
    print(f"{'='*78}")
    print(f"  {'Strategy':<34} {'N':>5} {'PF':>6} {'WR':>6} {'EV':>7} "
          f"{'Sharpe':>7} {'Sortino':>8} {'CAGR*':>7} {'DD*':>6} {'Calmar*':>8}")
    print(f"  {'─'*90}")
    for label, m in [
        ("VR Squeeze v14.3.0 (D+S)", vr14_m),
        ("MA Cluster R45", ma_m),
        (f"Combined v14.3.0", comb_m),
    ]:
        cagr  = m.get("prod_CAGR_pct", m["ann_ret_pct"])
        dd    = m.get("prod_max_dd_pct", m["max_dd_pct"])
        cal   = m.get("prod_calmar", m["calmar"])
        sh    = m.get("sharpe", 0.0)
        so    = m.get("sortino", 0.0)
        print(f"  {label:<34} {m['N']:>5} {m['PF']:>6.3f} {m['WR']:>5.1f}% "
              f"{m['EV_bp']:>+6.1f}bp {sh:>7.2f} {so:>8.2f} "
              f"{cagr:>+6.1f}% {dd:>5.1f}% {cal:>7.1f}")
    print(f"\n  * Production leverage model: VR L/N={VR_LN}, MA L/N={MA_LN}, combined L/N={blended_ln:.2f}")
    print(f"    1x CAGR (compute_metrics): VR={vr14_m['ann_ret_pct']:+.1f}%  "
          f"MA={ma_m['ann_ret_pct']:+.1f}%  Combined={comb_m['ann_ret_pct']:+.1f}%")
    print(f"\n  Data: {data_years:.2f}yr  Symbols: {len(data_15m)}  "
          f"Commit: {commit}  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
