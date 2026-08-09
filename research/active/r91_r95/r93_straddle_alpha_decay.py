"""r93: S_d8 Straddle Alpha Decay — 4 New Analyses.

Answers the key question r88 left open:
  Is EV genuinely decaying, or just ATR-compressed in a calmer market?

Analyses:
  A1: ATR-normalized EV by year — 'skill' decomposition
  A2: LONG vs SHORT leg attribution by year — directional exposure
  A3: Monthly PF heatmap — seasonality / bad-month detection
  A4: Symbol concentration — fragility of alpha source

Pre-flight (from config):
  s_trail_activate / s_trail_distance / s_sl / s_tp / s_vr_threshold / s_squeeze_dur

Usage:
    uv run python research/active/r91_r95/r93_straddle_alpha_decay.py
"""

from __future__ import annotations

import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    G_4TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_all_data(
    cfg: VolStraddleConfig,
) -> tuple[dict, dict, dict, dict, float, int, datetime]:
    """Load and precompute all TF data from cache_extended.

    Returns:
        (data_15m, data_30m, data_1h, data_4h, data_years, max_bars, start_date)
    """
    cache_dir = ROOT / "data" / "cache_extended"
    whitelist = set(cfg.symbol_whitelist)
    seen: dict[str, tuple[Path, int]] = {}
    collection_date: datetime | None = None

    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
        if sym == "BTCUSDT":
            m = re.search(r"(\d{8})", f.stem)
            if m:
                cd = datetime.strptime(m.group(1), "%Y%m%d")
                if collection_date is None or cd > collection_date:
                    collection_date = cd

    data_15m: dict = {}
    max_bars = 0
    raw: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        raw[sym] = {k: npz[k].astype(np.float64) for k in npz.files}
        if len(c) > max_bars:
            max_bars = len(c)

    for sym, d in raw.items():
        d["_offset"] = max_bars - len(d["close"])
        data_15m[sym] = d

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg.vr_short, cfg.vr_long)])
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)

    avg_bars = int(np.mean([len(d["close"]) for d in data_15m.values()]))
    data_years = avg_bars / (4 * 24 * 365)

    if collection_date is None:
        collection_date = datetime(2026, 2, 20)
    start_date = collection_date - timedelta(minutes=(max_bars - 1) * 15)

    print(f"  Loaded: {len(data_15m)} symbols, avg {avg_bars} bars, {data_years:.2f} years")
    print(f"  Data period: {start_date.strftime('%Y-%m-%d')} → {collection_date.strftime('%Y-%m-%d')}")
    return data_15m, data_30m, data_1h, data_4h, data_years, max_bars, start_date


# ══════════════════════════════════════════════════════════════════════════════
#  Canonical Baseline (Numba) — N/PF verification only
# ══════════════════════════════════════════════════════════════════════════════


def collect_trades_canonical(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list[tuple[str, int, int, float]]:
    """Collect straddle trades using canonical Numba sim (returns aggregate pnl only)."""
    sp, lp = cfg.vr_short, cfg.vr_long
    vr_key = f"vr_{sp}_{lp}"
    sl, tp = cfg.s_sl, cfg.s_tp
    trail_act, trail_dist = cfg.s_trail_activate, cfg.s_trail_distance
    s_vr, s_dur = cfg.s_vr_threshold, cfg.s_squeeze_dur
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _tf, ratio, warmup, _mh_d, mh in G_4TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                d["close"], d["high"], d["low"], sig, sl, tp, mh, COST_STRADDLE, trail_act, trail_dist
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio + offset, int(x[k]) * ratio + (ratio - 1) + offset, float(p[k])))
        # Dedup by entry bar
        seen_exit = -1
        for entry_b, exit_b, pnl in sorted(sym_trades):
            if entry_b > seen_exit:
                all_trades.append((sym, entry_b, exit_b, pnl))
                seen_exit = exit_b
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Detailed Python Sim — returns per-leg PnL
# ══════════════════════════════════════════════════════════════════════════════


def _sim_straddle_py(
    c: np.ndarray, h: np.ndarray, lo: np.ndarray,
    sig: np.ndarray, sl: float, tp: float, mh: int,
    cost: float, trail_act: float, trail_dist: float,
) -> list[tuple[int, int, float, float, float]]:
    """Pure-Python straddle sim returning (entry_bar, exit_bar, net_pnl, long_pnl, short_pnl).

    Same logic as sim_straddle_trailing but exposed per-leg data.
    """
    trades = []
    n = len(c)
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = c[i]
        if entry <= 0.0:
            i += 1
            continue

        la = True; sa = True
        le = 0.0; se = 0.0
        l_peak = 0.0; s_peak = 0.0
        l_trailing = False; s_trailing = False
        eb = i
        end = min(i + 1 + mh, n)

        for j in range(i + 1, end):
            if la:
                if lo[j] <= entry * (1.0 - sl):
                    le = -sl; la = False
                else:
                    l_gain = (h[j] - entry) / entry
                    if l_gain > l_peak:
                        l_peak = l_gain
                    if not l_trailing and l_peak >= trail_act:
                        l_trailing = True
                    if l_trailing:
                        if lo[j] <= entry * (1.0 + l_peak - trail_dist):
                            le = max(l_peak - trail_dist, -sl)
                            la = False
                    if la and h[j] >= entry * (1.0 + tp):
                        le = tp; la = False
            if sa:
                if h[j] >= entry * (1.0 + sl):
                    se = -sl; sa = False
                else:
                    s_gain = (entry - lo[j]) / entry
                    if s_gain > s_peak:
                        s_peak = s_gain
                    if not s_trailing and s_peak >= trail_act:
                        s_trailing = True
                    if s_trailing:
                        if h[j] >= entry * (1.0 - s_peak + trail_dist):
                            se = max(s_peak - trail_dist, -sl)
                            sa = False
                    if sa and lo[j] <= entry * (1.0 - tp):
                        se = tp; sa = False
            if not la and not sa:
                eb = j; break
        else:
            eb = end - 1

        if la:
            le = (c[eb] - entry) / entry
        if sa:
            se = (entry - c[eb]) / entry

        trades.append((i, eb, le + se - cost, le, se))
        i = eb + 1
    return trades


def collect_trades_detailed(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list[tuple[str, int, int, float, float, float]]:
    """Collect all S_d8 straddle trades with per-leg attribution.

    Returns list of (sym, entry_15m_global, exit_15m_global, net_pnl, long_pnl, short_pnl).
    """
    sp, lp = cfg.vr_short, cfg.vr_long
    vr_key = f"vr_{sp}_{lp}"
    sl, tp = cfg.s_sl, cfg.s_tp
    trail_act, trail_dist = cfg.s_trail_activate, cfg.s_trail_distance
    s_vr, s_dur = cfg.s_vr_threshold, cfg.s_squeeze_dur
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}

    all_trades: list[tuple[str, int, int, float, float, float]] = []
    for sym_idx, sym in enumerate(sorted(data_15m.keys())):
        if sym_idx % 10 == 0:
            print(f"  [{sym_idx+1}/60] {sym}...")
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float, float, float]] = []
        for _tf, ratio, warmup, _mh_d, mh in G_4TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            local_trades = _sim_straddle_py(
                d["close"], d["high"], d["low"], sig, sl, tp, mh, COST_STRADDLE, trail_act, trail_dist
            )
            for ebar, xbar, net, le, se in local_trades:
                entry_15m = ebar * ratio + offset
                exit_15m = xbar * ratio + (ratio - 1) + offset
                sym_trades.append((entry_15m, exit_15m, net, le, se))
        # Dedup
        seen_exit = -1
        for entry_b, exit_b, net, le, se in sorted(sym_trades):
            if entry_b > seen_exit:
                all_trades.append((sym, entry_b, exit_b, net, le, se))
                seen_exit = exit_b
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis helpers
# ══════════════════════════════════════════════════════════════════════════════


def compute_pf(pnl_list: list[float]) -> float:
    wins = sum(p for p in pnl_list if p > 0)
    losses = sum(-p for p in pnl_list if p < 0)
    return wins / losses if losses > 0 else float("inf")


def compute_ev_bp(pnl_list: list[float]) -> float:
    if not pnl_list:
        return 0.0
    return float(np.mean(pnl_list)) * 10_000


def year_of(bar: int, start_date: datetime) -> int:
    return (start_date + timedelta(minutes=bar * 15)).year


def month_of(bar: int, start_date: datetime) -> tuple[int, int]:
    dt = start_date + timedelta(minutes=bar * 15)
    return dt.year, dt.month


def compute_atr_by_year(data_15m: dict, start_date: datetime, max_bars: int) -> dict[int, float]:
    """Compute median 15m ATR% (high-low)/close across all symbols, grouped by year."""
    year_atrs: dict[int, list[float]] = defaultdict(list)
    for sym, d in data_15m.items():
        c = d["close"]
        h = d["high"]
        lo = d["low"]
        offset = int(d.get("_offset", 0))
        for i in range(len(c)):
            if c[i] <= 0:
                continue
            atr = (h[i] - lo[i]) / c[i]
            global_bar = i + offset
            yr = year_of(global_bar, start_date)
            year_atrs[yr].append(atr)
    return {yr: float(np.median(atrs)) for yr, atrs in sorted(year_atrs.items())}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    print("\n" + "=" * 72)
    print(f"r93: S_d8 Straddle Alpha Decay Analysis  [commit={commit}]")
    print("=" * 72)

    # ── Phase 0: Config ─────────────────────────────────────────────────────
    cfg = VolStraddleConfig()
    print(f"\n[Config — read from VolStraddleConfig()]")
    print(f"  s_sl={cfg.s_sl}  s_tp={cfg.s_tp}")
    print(f"  s_trail_activate={cfg.s_trail_activate}  s_trail_distance={cfg.s_trail_distance}")
    print(f"  s_vr_threshold={cfg.s_vr_threshold}  s_squeeze_dur={cfg.s_squeeze_dur}")

    # ── Phase 1: Data ────────────────────────────────────────────────────────
    print("\n[Phase 0] Loading data...")
    t0 = time.time()
    data_15m, data_30m, data_1h, data_4h, data_years, max_bars, start_date = load_all_data(cfg)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Phase 2: Canonical baseline verification ─────────────────────────────
    print("\n[Phase 1] Canonical Numba sim — baseline check...")
    t0 = time.time()
    canon_trades = collect_trades_canonical(cfg, data_15m, data_30m, data_1h, data_4h)
    canon_pnl = np.array([p for _, _, _, p in canon_trades])
    canon_sim = portfolio_sim([(sym, e, x, p) for sym, e, x, p in canon_trades], max_concurrent=5)
    canon_m = compute_metrics(canon_sim, data_years)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Canonical:  N={canon_m['N']}  PF={canon_m['PF']:.3f}  WR={canon_m['WR']:.1f}%  "
          f"EV={canon_m['EV_bp']:+.1f}bp  MaxDD={canon_m['max_dd_pct']:.1f}%")

    # ── Phase 3: Detailed Python sim (A1–A4) ─────────────────────────────────
    print("\n[Phase 2] Detailed Python sim (per-leg attribution, ~60–90s)...")
    t0 = time.time()
    trades = collect_trades_detailed(cfg, data_15m, data_30m, data_1h, data_4h)
    print(f"  Done in {time.time()-t0:.1f}s  N={len(trades)}")

    if len(trades) == 0:
        print("ERROR: No trades — check config/data.")
        return

    # Verify Python sim matches canonical
    py_pnl = np.array([net for _, _, _, net, _, _ in trades])
    py_sim = portfolio_sim([(sym, e, x, net) for sym, e, x, net, _, _ in trades], max_concurrent=5)
    py_m = compute_metrics(py_sim, data_years)
    pf_ratio = py_m["PF"] / canon_m["PF"] if canon_m["PF"] > 0 else 0.0
    print(f"  Python sim: N={py_m['N']}  PF={py_m['PF']:.3f}  PF-ratio={pf_ratio:.3f}  "
          f"(should be ~1.0; gap = Python dedup vs Numba order)")

    # ── A0: Pre-compute ATR by year ──────────────────────────────────────────
    print("\n[Pre-computing ATR by year (across all 60 symbols)...]")
    t0 = time.time()
    atr_by_year = compute_atr_by_year(data_15m, start_date, max_bars)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Assign year/month to each trade
    trade_years = [year_of(e, start_date) for _, e, _, _, _, _ in trades]
    trade_months = [month_of(e, start_date) for _, e, _, _, _, _ in trades]

    # ── A1: ATR-Normalized EV ────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("A1: ATR-Normalized EV by Year  (skill = EV / ATR)")
    print("─" * 72)
    print(f"  Year │   N  │   PF  │  WR%  │ EV(bp) │ ATR_med(bp) │ Skill(EV/ATR)")
    print(f"  ─────┼──────┼───────┼───────┼────────┼─────────────┼──────────────")

    year_trades: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for (sym, e, x, net, le, se), yr in zip(trades, trade_years):
        year_trades[yr].append((net, le, se))

    a1_rows = []
    for yr in sorted(year_trades.keys()):
        pnl_list = [t[0] for t in year_trades[yr]]
        n = len(pnl_list)
        pf = compute_pf(pnl_list)
        wr = 100 * sum(1 for p in pnl_list if p > 0) / n
        ev = compute_ev_bp(pnl_list)
        atr_med = atr_by_year.get(yr, float("nan")) * 10_000  # convert to bp
        skill = ev / atr_med if atr_med > 0 else float("nan")
        a1_rows.append((yr, n, pf, wr, ev, atr_med, skill))
        pf_str = f"{pf:.3f}" if pf != float("inf") else "  ∞  "
        print(f"  {yr} │ {n:4d} │ {pf_str} │ {wr:5.1f} │ {ev:+7.1f} │ {atr_med:11.1f} │ {skill:13.3f}")

    # ATR-normalized comparison: first year vs last year
    if len(a1_rows) >= 2:
        first_skill = a1_rows[0][-1]
        last_skill = a1_rows[-1][-1]
        skill_change = (last_skill - first_skill) / first_skill * 100 if first_skill > 0 else float("nan")
        print(f"\n  Skill ratio ({a1_rows[-1][0]}/{a1_rows[0][0]}): "
              f"{last_skill:.3f} / {first_skill:.3f} = {last_skill/first_skill:.2f}×  "
              f"(Δ{skill_change:+.1f}%)")
        if abs(skill_change) < 20:
            verdict = "✅ STABLE — EV decay is just ATR compression, not signal erosion"
        elif skill_change < -30:
            verdict = "⚠️  DECLINING — genuine alpha erosion (>30% skill loss)"
        else:
            verdict = "⚠️  MILD DECLINE — 20–30% skill degradation; monitor"
        print(f"  Verdict: {verdict}")

    # ── A2: LONG vs SHORT Attribution ────────────────────────────────────────
    print("\n" + "─" * 72)
    print("A2: LONG vs SHORT Leg Attribution by Year")
    print("─" * 72)
    print(f"  Year │   N  │ L-wins% │ AvgWin-L │ AvgLoss-L │ AvgWin-S │ AvgLoss-S │ SL-both%")
    print(f"  ─────┼──────┼─────────┼──────────┼───────────┼──────────┼───────────┼─────────")

    sl_thresh = -cfg.s_sl + 0.0001  # slightly above -sl (to catch SL hits)

    for yr in sorted(year_trades.keys()):
        entries = year_trades[yr]
        n = len(entries)
        l_wins = 0
        l_win_pnls, l_loss_pnls = [], []
        s_win_pnls, s_loss_pnls = [], []
        both_sl = 0

        for net, le, se in entries:
            if le > se:
                l_wins += 1
            # Long leg wins/losses
            if le > 0:
                l_win_pnls.append(le)
            else:
                l_loss_pnls.append(le)
            # Short leg wins/losses
            if se > 0:
                s_win_pnls.append(se)
            else:
                s_loss_pnls.append(se)
            # Both legs hit SL?
            if le <= sl_thresh and se <= sl_thresh:
                both_sl += 1

        lw_pct = 100 * l_wins / n
        avg_lwin = 10000 * np.mean(l_win_pnls) if l_win_pnls else 0.0
        avg_lloss = 10000 * np.mean(l_loss_pnls) if l_loss_pnls else 0.0
        avg_swin = 10000 * np.mean(s_win_pnls) if s_win_pnls else 0.0
        avg_sloss = 10000 * np.mean(s_loss_pnls) if s_loss_pnls else 0.0
        sl_both_pct = 100 * both_sl / n

        print(f"  {yr} │ {n:4d} │  {lw_pct:5.1f}%  │  {avg_lwin:7.0f}  │   {avg_lloss:7.0f}  │  {avg_swin:7.0f}  │   {avg_sloss:7.0f}  │  {sl_both_pct:5.1f}%")

    print("\n  Interpretation:")
    print("  L-wins% = fraction of trades where LONG leg made more than SHORT leg")
    print("  SL-both% = trades where BOTH legs hit their stop-loss (maximum loss scenario)")
    print("  If L-wins% drifts from ~50% → directional bias in recent market regime")

    # ── A3: Monthly PF Heatmap ───────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("A3: Monthly PF Heatmap  (year × month)")
    print("─" * 72)

    month_pnl: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (sym, e, x, net, le, se), (yr, mo) in zip(trades, trade_months):
        month_pnl[(yr, mo)].append(net)

    all_years = sorted({yr for yr, mo in month_pnl})
    months = list(range(1, 13))
    month_names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

    # Header
    print("       " + "  ".join(f"{m:>5}" for m in month_names))
    for yr in all_years:
        row_parts = []
        for mo in months:
            pnls = month_pnl.get((yr, mo), [])
            if not pnls:
                row_parts.append("    - ")
            else:
                pf = compute_pf(pnls)
                n = len(pnls)
                if pf == float("inf"):
                    tag = f" ∞({n})"
                elif pf >= 2.0:
                    tag = f" {pf:.1f}★"
                elif pf >= 1.0:
                    tag = f" {pf:.1f} "
                else:
                    tag = f"[{pf:.1f}]"
                row_parts.append(f"{tag:>6}")
        print(f"  {yr} " + "  ".join(row_parts))

    print("\n  Legend: ★=PF≥2  [.]=PF<1  -=no trades  number in () = trade count where PF=inf")

    # ── A4: Symbol Concentration ─────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("A4: Symbol Concentration — alpha source fragility")
    print("─" * 72)

    sym_pnl: dict[str, list[float]] = defaultdict(list)
    for sym, e, x, net, le, se in trades:
        sym_pnl[sym].append(net)

    sym_ev = [(sym, sum(pnls), len(pnls)) for sym, pnls in sym_pnl.items()]
    sym_ev.sort(key=lambda x: -x[1])  # sort by total pnl descending

    total_positive = sum(ev for _, ev, _ in sym_ev if ev > 0)
    total_n = sum(n for _, _, n in sym_ev)

    print(f"  Total positive EV pool: {total_positive*10000:.0f}bp-trades across {len(sym_ev)} symbols")
    print(f"\n  Rank │ Symbol         │  N  │ Cum.EV(bp) │ Running% of positive pool")

    cumulative = 0.0
    for rank, (sym, ev, n) in enumerate(sym_ev[:20], 1):
        if ev > 0:
            cumulative += ev
        cum_pct = 100 * cumulative / total_positive if total_positive > 0 else 0.0
        bar = "█" * int(cum_pct / 5)
        print(f"  {rank:4d} │ {sym:<14s} │ {n:3d} │ {ev*10000:+10.0f} │ {cum_pct:5.1f}%  {bar}")

    # Concentration metrics
    top5_pct = 100 * sum(max(ev, 0) for _, ev, _ in sym_ev[:5]) / total_positive if total_positive > 0 else 0
    top10_pct = 100 * sum(max(ev, 0) for _, ev, _ in sym_ev[:10]) / total_positive if total_positive > 0 else 0
    neg_syms = sum(1 for _, ev, _ in sym_ev if ev < 0)
    print(f"\n  Top-5  symbols: {top5_pct:.1f}% of total positive EV")
    print(f"  Top-10 symbols: {top10_pct:.1f}% of total positive EV")
    print(f"  Negative-EV symbols: {neg_syms}/{len(sym_ev)}")

    if top5_pct > 60:
        conc_verdict = "⚠️  FRAGILE — top-5 drive >60% of alpha"
    elif top5_pct > 40:
        conc_verdict = "⚠️  CONCENTRATED — top-5 drive 40-60% of alpha"
    else:
        conc_verdict = "✅ DIVERSIFIED — alpha spread across many symbols"
    print(f"  Concentration: {conc_verdict}")

    # ── A4b: Alpha by year per symbol (top 10 symbols) ───────────────────────
    print("\n  EV by year for top-10 symbols (bp / year):")
    top10_syms = [sym for sym, _, _ in sym_ev[:10]]
    sym_year_pnl: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (sym, e, x, net, le, se), yr in zip(trades, trade_years):
        if sym in top10_syms:
            sym_year_pnl[(sym, yr)].append(net)

    hdr = "  Symbol        " + "  ".join(f"{yr}" for yr in all_years)
    print(f"  {hdr}")
    for sym in top10_syms:
        row_str = f"  {sym:<14s} "
        for yr in all_years:
            pnls = sym_year_pnl.get((sym, yr), [])
            ev = compute_ev_bp(pnls) if pnls else 0.0
            n = len(pnls)
            row_str += f"  {ev:+6.0f}({n:2d})"
        print(row_str)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Canonical sim baseline: N={canon_m['N']}  PF={canon_m['PF']:.3f}  EV={canon_m['EV_bp']:+.1f}bp")
    boot = bootstrap_pf(np.array([p for _, _, _, p in canon_trades]), n_boot=2000)
    print(f"  Bootstrap CI (95%): PF [{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
          f"P(PF≥1.0)={boot['P_PF_ge_1']:.1%}")
    print()

    if a1_rows:
        print("  A1 ATR-normalized skill by year:")
        for yr, n, pf, wr, ev, atr, skill in a1_rows:
            print(f"    {yr}: EV={ev:+.0f}bp  ATR={atr:.0f}bp  Skill={skill:.3f}")

    print()
    print("  Decision matrix:")
    if len(a1_rows) >= 2:
        recent_skill = np.mean([r[-1] for r in a1_rows[-2:]])
        early_skill = np.mean([r[-1] for r in a1_rows[:2]])
        skill_decay = (recent_skill - early_skill) / early_skill * 100

        if abs(skill_decay) < 20:
            print("  → EV/ATR stable: alpha lives, invest in server trail precision")
        elif skill_decay < -30:
            print("  → EV/ATR declining >30%: genuine erosion — set 12-month sunset")
        else:
            print("  → EV/ATR mild decline (20-30%): monitor monthly, no action yet")

    recent_sl_pct = np.mean([
        100 * sum(1 for net, le, se in year_trades[yr] if le <= sl_thresh and se <= sl_thresh) / len(year_trades[yr])
        for yr in sorted(year_trades.keys())[-2:]
        if year_trades[yr]
    ])
    print(f"  SL-both% (recent 2yr avg): {recent_sl_pct:.1f}%")
    if recent_sl_pct > 15:
        print("  → High double-SL rate: post-squeeze volatility spike problem")
    else:
        print("  → Double-SL rate acceptable")

    print(f"\n  Commit: {commit}")
    print(f"  Data: {start_date.strftime('%Y-%m-%d')} → 2026-02-20  ({data_years:.1f}yr)")
    print(f"  Config: s_trail_activate={cfg.s_trail_activate}  s_trail_distance={cfg.s_trail_distance}")
    print("=" * 72)


if __name__ == "__main__":
    main()
