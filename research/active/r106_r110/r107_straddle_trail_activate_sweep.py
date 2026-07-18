"""r107: Straddle Trail Activation Threshold Sweep.

목적:
  현재 s_trail_activate=3.0% (r106에서 1.5%→3.0%로 인상)로 인해
  straddle 다리가 실제로 trail을 활성화하지 못하고 고정 SL로만 청산됨.
  Live에서 MFE가 +1.5~2.0% 수준인데, 3.0% activate는 사실상 dead parameter.

  s_trail_activate를 [1.5%, 2.0%, 3.0%(baseline)] 비교하여 최적 값을 찾는다.
  s_trail_distance=0.5%는 고정 (Binance callbackRate >= 0.5%, rule #8).

실험:
  각 activate 값에 대해:
  - 총 straddle trade 수
  - Win rate, Profit Factor, Net PnL
  - Trail activation rate (각 다리 기준)
  - 평균 MFE at exit

사용:
    cd /home/user/Documents/git/quantbox
    source .venv/bin/activate && PYTHONPATH=. python3 research/active/r106_r110/r107_straddle_trail_activate_sweep.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (  # noqa: E402
    COST_STRADDLE,
    G_4TF,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import (  # noqa: E402
    gen_vr_dur_signal,
)


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Detailed Straddle Sim — tracks per-leg trail activation + MFE
# ══════════════════════════════════════════════════════════════════════════════


def sim_straddle_trailing_detailed(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Straddle sim with trailing — returns extra detail arrays.

    Same logic as trade_sim.sim_straddle_trailing, but also returns:
      - long_mfe: max favorable excursion per trade (long leg)
      - short_mfe: max favorable excursion per trade (short leg)
      - long_trail_activated: 1 if long leg's trail activated, 0 otherwise
      - short_trail_activated: 1 if short leg's trail activated, 0 otherwise

    NOTE: This duplicates sim_straddle_trailing logic for analysis only.
    Production code must use the canonical sim from trade_sim.py.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    l_mfe_arr = np.empty(cap, dtype=np.float64)
    s_mfe_arr = np.empty(cap, dtype=np.float64)
    l_trail_arr = np.empty(cap, dtype=np.int64)
    s_trail_arr = np.empty(cap, dtype=np.int64)
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
        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            # Long leg — identical to trade_sim.sim_straddle_trailing
            if la:
                l_gain = (high[j] - entry) / entry
                if l_gain > l_peak:
                    l_peak = l_gain
                if not l_trailing and l_peak >= trail_activate:
                    l_trailing = True
                if l_trailing:
                    trail_lvl = entry * (1.0 + l_peak - trail_distance)
                    if low[j] <= trail_lvl:
                        le = max(l_peak - trail_distance, -sl)
                        la = False
                else:
                    if low[j] <= entry * (1.0 - sl):
                        le = -sl
                        la = False
                if la and high[j] >= entry * (1.0 + tp):
                    le = tp
                    la = False

            # Short leg — identical to trade_sim.sim_straddle_trailing
            if sa:
                s_gain = (entry - low[j]) / entry
                if s_gain > s_peak:
                    s_peak = s_gain
                if not s_trailing and s_peak >= trail_activate:
                    s_trailing = True
                if s_trailing:
                    trail_lvl = entry * (1.0 - s_peak + trail_distance)
                    if high[j] >= trail_lvl:
                        se = max(s_peak - trail_distance, -sl)
                        sa = False
                else:
                    if high[j] >= entry * (1.0 + sl):
                        se = -sl
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
        if sa:
            se = (entry - close[eb]) / entry
        net = le + se - cost
        if cnt < cap:
            pnl[cnt] = net
            ent[cnt] = i
            ext[cnt] = eb
            l_mfe_arr[cnt] = l_peak
            s_mfe_arr[cnt] = s_peak
            l_trail_arr[cnt] = 1 if l_trailing else 0
            s_trail_arr[cnt] = 1 if s_trailing else 0
            cnt += 1
        i = eb + 1
    return (
        pnl[:cnt], ent[:cnt], ext[:cnt],
        l_mfe_arr[:cnt], s_mfe_arr[:cnt],
        l_trail_arr[:cnt], s_trail_arr[:cnt],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading (r106 pattern)
# ══════════════════════════════════════════════════════════════════════════════


def load_all_data(cfg: VolStraddleConfig) -> tuple[dict, dict, dict, dict, float, int]:
    """Load 15m/30m/1h/4h from cache, precompute VR/momentum."""
    cache_dir = ROOT / "data" / "cache_extended"
    whitelist = set(cfg.symbol_whitelist)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    data_15m: dict = {}
    max_bars = 0
    raw: dict = {}
    for sym, (fpath, _nb) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        raw[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
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

    bars_list = [len(d["close"]) for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)

    print(f"  Loaded: {len(data_15m)} symbols, avg {avg_bars} bars, {data_years:.2f} years")
    return data_15m, data_30m, data_1h, data_4h, data_years, max_bars


# ══════════════════════════════════════════════════════════════════════════════
#  Detailed Trade Collection
# ══════════════════════════════════════════════════════════════════════════════


def _dedup_detailed(
    sym: str,
    trades: list[tuple[int, int, float, float, float, int, int]],
) -> list[tuple[str, int, int, float, float, float, int, int]]:
    """Per-symbol time dedup for detailed trades.

    Input tuple: (entry_15m, exit_15m, pnl, l_mfe, s_mfe, l_trail, s_trail)
    Output tuple: (sym, entry_15m, exit_15m, pnl, l_mfe, s_mfe, l_trail, s_trail)
    """
    result: list[tuple[str, int, int, float, float, float, int, int]] = []
    last_exit = -1
    for row in sorted(trades, key=lambda t: t[0]):
        if row[0] > last_exit:
            result.append((sym, *row))
            last_exit = row[1]
    return result


def collect_straddle_detailed(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    trail_activate: float,
    trail_distance: float,
) -> list[tuple[str, int, int, float, float, float, int, int]]:
    """Collect S_d8 straddle trades with detailed per-leg info.

    Returns list of (sym, entry_15m, exit_15m, pnl, l_mfe, s_mfe, l_trail, s_trail).
    """
    sp = cfg.vr_short
    lp = cfg.vr_long
    vr_key = f"vr_{sp}_{lp}"
    sl = cfg.s_sl
    tp = cfg.s_tp

    tf_data_map: dict[int, dict] = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}

    all_trades: list[tuple[str, int, int, float, float, float, int, int]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float, float, float, int, int]] = []

        for _tf_label, ratio, warmup, _mh_d, mh_s in G_4TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            c, h, lo = d["close"], d["high"], d["low"]
            vr = d.get(vr_key)
            if vr is None:
                continue

            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x, l_mfe, s_mfe, l_tr, s_tr = sim_straddle_trailing_detailed(
                c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE, trail_activate, trail_distance
            )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m = int(x[k]) * ratio + (ratio - 1)
                sym_trades.append((
                    entry_15m, exit_15m, float(p[k]),
                    float(l_mfe[k]), float(s_mfe[k]),
                    int(l_tr[k]), int(s_tr[k]),
                ))

        all_trades.extend(_dedup_detailed(sym, sym_trades))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio Sim with Details (slot-filtered)
# ══════════════════════════════════════════════════════════════════════════════


def portfolio_sim_detailed(
    all_trades: list[tuple[str, int, int, float, float, float, int, int]],
    max_concurrent: int,
) -> list[tuple[str, int, int, float, float, float, int, int]]:
    """Slot-filtered portfolio sim for detailed trades.

    Each straddle consumes 2 slots. Returns accepted trades only.
    """
    trades = sorted(all_trades, key=lambda x: x[1])
    accepted: list[tuple[str, int, int, float, float, float, int, int]] = []
    active: list[tuple[str, int]] = []

    for row in trades:
        sym, entry, exit_b = row[0], row[1], row[2]
        # Expire finished
        active = [(s, e) for s, e in active if e > entry]
        # Same-symbol block
        if any(s == sym for s, _ in active):
            continue
        # Straddle = 2 slots
        if len(active) + 2 > max_concurrent:
            continue
        active.extend([(sym, exit_b)] * 2)
        accepted.append(row)

    return accepted


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis
# ══════════════════════════════════════════════════════════════════════════════


def analyze_sweep_point(
    cfg: VolStraddleConfig,
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    trail_activate: float,
    trail_distance: float,
    data_years: float,
) -> dict:
    """Run one sweep point and return detailed metrics."""
    t0 = time.time()

    all_trades = collect_straddle_detailed(
        cfg, data_15m, data_30m, data_1h, data_4h,
        trail_activate, trail_distance,
    )

    accepted = portfolio_sim_detailed(all_trades, cfg.max_concurrent)
    elapsed = time.time() - t0

    if len(accepted) < 5:
        return {
            "activate": trail_activate,
            "activate_pct": f"{trail_activate*100:.1f}%",
            "n": len(accepted),
            "wr_pct": 0.0,
            "pf": 0.0,
            "net_pnl_pct": 0.0,
            "trail_rate_long_pct": 0.0,
            "trail_rate_short_pct": 0.0,
            "trail_rate_any_pct": 0.0,
            "avg_mfe_long_pct": 0.0,
            "avg_mfe_short_pct": 0.0,
            "avg_mfe_best_pct": 0.0,
            "calmar": 0.0,
            "max_dd_pct": 0.0,
            "ann_ret_pct": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "elapsed": elapsed,
        }

    pnl_arr = np.array([t[3] for t in accepted])
    l_mfe_arr = np.array([t[4] for t in accepted])
    s_mfe_arr = np.array([t[5] for t in accepted])
    l_trail_arr = np.array([t[6] for t in accepted])
    s_trail_arr = np.array([t[7] for t in accepted])

    n = len(pnl_arr)
    wins = np.sum(pnl_arr > 0)
    wr = 100.0 * wins / n

    m = compute_metrics(pnl_arr, data_years)
    bs = bootstrap_pf(pnl_arr)

    # Trail activation rates
    l_trail_rate = 100.0 * np.sum(l_trail_arr) / n
    s_trail_rate = 100.0 * np.sum(s_trail_arr) / n
    any_trail = np.sum((l_trail_arr + s_trail_arr) > 0)
    any_trail_rate = 100.0 * any_trail / n

    # MFE stats
    avg_l_mfe = 100.0 * np.mean(l_mfe_arr)
    avg_s_mfe = 100.0 * np.mean(s_mfe_arr)
    best_mfe = np.maximum(l_mfe_arr, s_mfe_arr)
    avg_best_mfe = 100.0 * np.mean(best_mfe)

    # Compound equity for net PnL
    equity = 1.0
    for p in pnl_arr:
        equity *= (1.0 + p)
    net_pnl_pct = (equity - 1.0) * 100.0

    return {
        "activate": trail_activate,
        "activate_pct": f"{trail_activate*100:.1f}%",
        "n": n,
        "wr_pct": wr,
        "pf": m["PF"],
        "net_pnl_pct": net_pnl_pct,
        "trail_rate_long_pct": l_trail_rate,
        "trail_rate_short_pct": s_trail_rate,
        "trail_rate_any_pct": any_trail_rate,
        "avg_mfe_long_pct": avg_l_mfe,
        "avg_mfe_short_pct": avg_s_mfe,
        "avg_mfe_best_pct": avg_best_mfe,
        "calmar": m["calmar"],
        "max_dd_pct": m["max_dd_pct"],
        "ann_ret_pct": m.get("ann_ret_pct", 0.0),
        "ci_lo": bs["CI_lo"],
        "ci_hi": bs["CI_hi"],
        "elapsed": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    cfg = VolStraddleConfig()

    print("=" * 90)
    print("  r107: Straddle Trail Activation Threshold Sweep")
    print("=" * 90)
    print(f"  commit: {commit}")
    print(f"  date:   {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  max_concurrent: {cfg.max_concurrent}")
    print(f"\n  Current config:")
    print(f"    s_trail_activate = {cfg.s_trail_activate} ({cfg.s_trail_activate*100:.1f}%)")
    print(f"    s_trail_distance = {cfg.s_trail_distance} ({cfg.s_trail_distance*100:.1f}%)")
    print(f"    s_sl = {cfg.s_sl} ({cfg.s_sl*100:.1f}%)")
    print(f"    s_tp = {cfg.s_tp} ({cfg.s_tp*100:.1f}%)")

    # Sweep grid: compare 1.0%, 1.5%, 2.0%, 3.0% (current)
    activate_grid = [0.010, 0.015, 0.020, 0.030]
    trail_distance = cfg.s_trail_distance  # Fixed at 0.5% (rule #8)

    print(f"\n  Sweep: s_trail_activate = {[f'{a*100:.1f}%' for a in activate_grid]}")
    print(f"  Fixed: s_trail_distance = {trail_distance*100:.1f}% (rule #8 floor)")

    print(f"\n  Loading data...")
    t0 = time.time()
    data_15m, data_30m, data_1h, data_4h, data_years, _max_bars = load_all_data(cfg)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Run sweep
    results = []
    for act in activate_grid:
        print(f"\n  Running s_trail_activate={act*100:.1f}%...")
        r = analyze_sweep_point(
            cfg, data_15m, data_30m, data_1h, data_4h,
            trail_activate=act, trail_distance=trail_distance,
            data_years=data_years,
        )
        results.append(r)
        print(f"    N={r['n']}, PF={r['pf']:.3f}, WR={r['wr_pct']:.1f}%, "
              f"Trail={r['trail_rate_any_pct']:.1f}%, elapsed={r['elapsed']:.1f}s")

    # ── Comparison Table ──
    print(f"\n\n{'='*100}")
    print(f"  COMPARISON TABLE — Straddle Trail Activation Sweep")
    print(f"{'='*100}")
    print(f"  Fixed: s_trail_distance={trail_distance*100:.1f}%, s_sl={cfg.s_sl*100:.1f}%, s_tp={cfg.s_tp*100:.1f}%")
    print()

    header = (
        f"  {'Activate':>8}  {'N':>5}  {'WR%':>6}  {'PF':>6}  {'Calmar':>7}  "
        f"{'NetPnL%':>9}  {'MaxDD%':>7}  {'AnnRet%':>8}  "
        f"{'TrailL%':>8}  {'TrailS%':>8}  {'TrailAny%':>9}  "
        f"{'MFE_L%':>7}  {'MFE_S%':>7}  {'MFE_best%':>9}  "
        f"{'CI_lo':>6}  {'CI_hi':>6}"
    )
    print(header)
    sep = "  " + "  ".join([
        "-" * 8, "-" * 5, "-" * 6, "-" * 6, "-" * 7,
        "-" * 9, "-" * 7, "-" * 8,
        "-" * 8, "-" * 8, "-" * 9,
        "-" * 7, "-" * 7, "-" * 9,
        "-" * 6, "-" * 6,
    ])
    print(sep)

    for r in results:
        marker = " <-- current" if r["activate"] == cfg.s_trail_activate else ""
        print(
            f"  {r['activate_pct']:>8}  {r['n']:>5}  {r['wr_pct']:>6.1f}  {r['pf']:>6.3f}  "
            f"{r['calmar']:>7.1f}  {r['net_pnl_pct']:>9.1f}  {r['max_dd_pct']:>7.1f}  "
            f"{r['ann_ret_pct']:>8.1f}  "
            f"{r['trail_rate_long_pct']:>8.1f}  {r['trail_rate_short_pct']:>8.1f}  "
            f"{r['trail_rate_any_pct']:>9.1f}  "
            f"{r['avg_mfe_long_pct']:>7.2f}  {r['avg_mfe_short_pct']:>7.2f}  "
            f"{r['avg_mfe_best_pct']:>9.2f}  "
            f"{r['ci_lo']:>6.3f}  {r['ci_hi']:>6.3f}{marker}"
        )

    # ── MFE Distribution ──
    print(f"\n\n{'='*80}")
    print(f"  MFE DISTRIBUTION — How often do legs reach various thresholds?")
    print(f"{'='*80}")
    print(f"  (Based on baseline run at s_trail_activate=3.0%)")

    # Use the 3.0% run (last) for MFE distribution since trail_activate doesn't
    # affect MFE itself — it only affects whether the trail gets activated.
    # Actually MFE is affected by trail since trail changes when the leg closes.
    # Use the 1.5% activate for the widest trail data.
    # We need raw unfiltered trades for true MFE distribution. Run one more
    # collection with the current config to show MFE percentiles.
    print(f"\n  Re-running with current config for MFE percentile analysis...")
    baseline_trades = collect_straddle_detailed(
        cfg, data_15m, data_30m, data_1h, data_4h,
        trail_activate=cfg.s_trail_activate, trail_distance=trail_distance,
    )
    baseline_accepted = portfolio_sim_detailed(baseline_trades, cfg.max_concurrent)

    if baseline_accepted:
        l_mfe = np.array([t[4] for t in baseline_accepted]) * 100
        s_mfe = np.array([t[5] for t in baseline_accepted]) * 100
        best_mfe = np.maximum(l_mfe, s_mfe)

        pcts = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n  {'Pct':>5}  {'Long MFE%':>10}  {'Short MFE%':>11}  {'Best Leg MFE%':>14}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*11}  {'-'*14}")
        for p in pcts:
            print(f"  {p:>5}  {np.percentile(l_mfe, p):>10.2f}  "
                  f"{np.percentile(s_mfe, p):>11.2f}  {np.percentile(best_mfe, p):>14.2f}")

        # What fraction reach each activate threshold?
        print(f"\n  Fraction of best-leg MFE reaching each activate threshold:")
        for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            frac = 100.0 * np.mean(best_mfe >= thresh)
            frac_l = 100.0 * np.mean(l_mfe >= thresh)
            frac_s = 100.0 * np.mean(s_mfe >= thresh)
            print(f"    >= {thresh:>4.1f}%:  best_leg={frac:>5.1f}%  long={frac_l:>5.1f}%  short={frac_s:>5.1f}%")

    # ── Summary & Recommendation ──
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY — r107 Straddle Trail Activate Sweep ({commit})")
    print(f"{'='*90}")
    print(f"\n  Current config: s_trail_activate={cfg.s_trail_activate*100:.1f}%")
    print()

    # Find best by Calmar among those with PF >= 1.0
    viable = [r for r in results if r["pf"] >= 1.0 and r["n"] >= 50]
    if viable:
        by_calmar = sorted(viable, key=lambda r: r["calmar"], reverse=True)
        best = by_calmar[0]
        print(f"  Best by Calmar (PF>=1.0, N>=50):")
        print(f"    s_trail_activate = {best['activate_pct']}")
        print(f"    PF={best['pf']:.3f}, Calmar={best['calmar']:.1f}, WR={best['wr_pct']:.1f}%")
        print(f"    Trail activation rate: {best['trail_rate_any_pct']:.1f}% of trades")
        print(f"    CI=[{best['ci_lo']:.3f}, {best['ci_hi']:.3f}]")

        by_pf = sorted(viable, key=lambda r: r["pf"], reverse=True)
        best_pf = by_pf[0]
        if best_pf["activate"] != best["activate"]:
            print(f"\n  Best by PF:")
            print(f"    s_trail_activate = {best_pf['activate_pct']}")
            print(f"    PF={best_pf['pf']:.3f}, Calmar={best_pf['calmar']:.1f}, WR={best_pf['wr_pct']:.1f}%")
            print(f"    Trail activation rate: {best_pf['trail_rate_any_pct']:.1f}% of trades")

        # Highlight the trail activation rate difference
        current_r = next((r for r in results if r["activate"] == cfg.s_trail_activate), None)
        if current_r and best["activate"] != cfg.s_trail_activate:
            print(f"\n  Current ({cfg.s_trail_activate*100:.1f}%) vs Best ({best['activate_pct']}):")
            print(f"    Trail rate: {current_r['trail_rate_any_pct']:.1f}% -> {best['trail_rate_any_pct']:.1f}%")
            print(f"    PF:         {current_r['pf']:.3f} -> {best['pf']:.3f}")
            print(f"    Calmar:     {current_r['calmar']:.1f} -> {best['calmar']:.1f}")
    else:
        print("  No viable configuration found (PF>=1.0, N>=50)")

    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
