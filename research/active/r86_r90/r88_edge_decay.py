"""r88: Edge Decay Deep Research — Adaptive Entry/Exit to Maintain Edge.

**목적:**
  EV per trade가 6년간 66% 감소 (+65bp→+22bp). 정적 VR 임계값이
  변화하는 시장 체제에 적응하지 못함. 적응형 진입/청산으로 엣지를
  유지할 수 있는지 실험한다.

**연구 질문:**
  1. 엣지 감소의 정확한 원인? (WR 하락? 수익 축소? 손실 확대?)
  2. 적응형 VR 임계값이 2023-2025 PF를 개선하면서 2020-2022를 유지하는가?
  3. 체제 적응형 trail 파라미터가 저변동성 시장에서 도움이 되는가?
  4. Conviction 필터가 최근 연도에서 신호 품질을 분리하는가?

**Phases:**
  Phase 0: Config + Data
  Phase 1: Baseline reproduction
  Phase 2: T1 — Edge decomposition by year
  Phase 3: T2 — Rolling VR percentile threshold (adaptive entry)
  Phase 4: T3 — Regime-scaled trail params (adaptive exit)
  Phase 5: T4 — Conviction filter by year
  Phase 6: Summary

사용:
    uv run python research/active/r86_r90/r88_edge_decay.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import numba

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
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
    sim_directional_trailing,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading (matches r86 pattern)
# ══════════════════════════════════════════════════════════════════════════════


def load_all_data(cfg: VolStraddleConfig) -> tuple[dict, dict, dict, dict, float, int]:
    """Load 15m/30m/1h/4h from cache, precompute VR/momentum.

    Returns:
        (data_15m, data_30m, data_1h, data_4h, data_years, max_bars)
    """
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
#  Trade Collection (from r86 — production exit logic)
# ══════════════════════════════════════════════════════════════════════════════


def _dedup(sym: str, trades: list[tuple[int, int, float]]) -> list[tuple[str, int, int, float]]:
    result = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_trades_trailing(
    cfg_obj: VolStraddleConfig,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    *,
    tf_group: list | None = None,
    tf_priority: bool = False,
    data_30m: dict | None = None,
    s_vr_override: float | None = None,
    d_vr_override: float | None = None,
    trail_act_override: float | None = None,
    trail_dist_override: float | None = None,
    adaptive_vr_arr: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect trades using trail-aware sim functions.

    Supports overrides for research experiments:
    - s_vr_override/d_vr_override: override VR threshold
    - trail_act_override/trail_dist_override: override trail params
    - adaptive_vr_arr: per-symbol adaptive VR threshold arrays (T2)
    """
    sp = cfg_obj.vr_short
    lp = cfg_obj.vr_long
    vr_key = f"vr_{sp}_{lp}"
    is_straddle = strat == "S_d8"

    d_sl = cfg_obj.d_sl
    s_sl = cfg_obj.s_sl
    d_tp = cfg_obj.d_tp
    s_tp = cfg_obj.s_tp
    d_vr = d_vr_override if d_vr_override is not None else cfg_obj.d_vr_threshold
    s_vr = s_vr_override if s_vr_override is not None else cfg_obj.s_vr_threshold
    s_dur = cfg_obj.s_squeeze_dur

    if is_straddle:
        trail_act = trail_act_override if trail_act_override is not None else cfg_obj.s_trail_activate
        trail_dist = trail_dist_override if trail_dist_override is not None else cfg_obj.s_trail_distance
    else:
        trail_act = trail_act_override if trail_act_override is not None else cfg_obj.d_trail_activate
        trail_dist = trail_dist_override if trail_dist_override is not None else cfg_obj.d_trail_distance

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
                if adaptive_vr_arr is not None and sym in adaptive_vr_arr:
                    sig = _gen_vr_dur_adaptive(vr, adaptive_vr_arr[sym], s_dur, warmup, ratio)
                else:
                    sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
                p, e, x = sim_straddle_trailing(
                    c, h, lo, sig, sl, tp, mh, cost, trail_act, trail_dist
                )
            else:
                if adaptive_vr_arr is not None and sym in adaptive_vr_arr:
                    sig = _gen_vr_adaptive(vr, adaptive_vr_arr[sym], warmup, ratio)
                else:
                    sig = gen_vr_signal(vr, d_vr, warmup)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trailing(
                    c, h, lo, sig, dire, sl, tp, mh, cost, trail_act, trail_dist
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio + offset
                exit_15m = int(x[k]) * ratio + (ratio - 1) + offset
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        all_trades.extend(_dedup(sym, sym_trades))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  T2: Adaptive VR Signal Generation
# ══════════════════════════════════════════════════════════════════════════════


def compute_adaptive_vr_thresholds(
    data: dict,
    vr_key: str,
    percentile: float = 20.0,
    window_bars: int = 17520,
    step: int = 96,
) -> dict[str, np.ndarray]:
    """Compute rolling percentile threshold of VR for each symbol.

    Instead of static threshold (e.g., 0.57), uses rolling Nth percentile
    of VR over trailing window. In calm regimes, threshold auto-adjusts upward
    to capture "relatively compressed" periods.

    Optimized: computes every `step` bars (default=96=1 day) and forward-fills.

    Args:
        data: Symbol→data dict (must contain vr_key)
        vr_key: e.g. "vr_8_56"
        percentile: Percentile to use (lower = stricter squeeze)
        window_bars: Lookback window in bars
        step: Compute percentile every N bars (forward-fill between)
    """
    result: dict[str, np.ndarray] = {}
    for sym, d in data.items():
        vr = d.get(vr_key)
        if vr is None:
            continue
        n = len(vr)
        thresholds = np.full(n, np.nan, dtype=np.float64)
        last_val = np.nan
        for i in range(window_bars, n, step):
            window = vr[i - window_bars:i]
            valid = window[~np.isnan(window)]
            if len(valid) > 10:
                last_val = np.percentile(valid, percentile)
            thresholds[i:min(i + step, n)] = last_val
        result[sym] = thresholds
    return result


@numba.jit(nopython=True, cache=True)
def _gen_vr_dur_adaptive(
    vr: np.ndarray,
    adaptive_thresh: np.ndarray,
    dur: int,
    warmup: int,
    ratio: int,
) -> np.ndarray:
    """gen_vr_dur_signal variant using per-bar adaptive threshold."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int64)
    count = 0
    for i in range(warmup, n):
        if np.isnan(adaptive_thresh[i]) or np.isnan(vr[i]):
            count = 0
            continue
        # For resampled data, index into adaptive_thresh array by mapping
        # through ratio (adaptive_thresh is at 15m resolution, vr is at TF resolution)
        # When ratio>1, adaptive threshold is already at TF-level (computed on TF data)
        if vr[i] < adaptive_thresh[i]:
            count += 1
        else:
            count = 0
        if count == dur:
            sig[i] = 1
            count = 0
    return sig


@numba.jit(nopython=True, cache=True)
def _gen_vr_adaptive(
    vr: np.ndarray,
    adaptive_thresh: np.ndarray,
    warmup: int,
    ratio: int,
) -> np.ndarray:
    """gen_vr_signal variant using per-bar adaptive threshold."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int64)
    for i in range(warmup, n):
        if np.isnan(adaptive_thresh[i]) or np.isnan(vr[i]):
            continue
        if vr[i] < adaptive_thresh[i]:
            sig[i] = 1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  T3: Regime-Scaled Trail Parameters
# ══════════════════════════════════════════════════════════════════════════════


def compute_yearly_atr_stats(
    data_15m: dict,
    data_years: float,
    atr_period: int = 56,
) -> dict[int, float]:
    """Compute median ATR(56) per year across all symbols.

    Returns dict: {year_label: median_atr_pct}
    """
    all_entries = []
    for sym, d in data_15m.items():
        c = d["close"]
        h = d["high"]
        lo = d["low"]
        n = len(c)
        for i in range(atr_period, n):
            tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            atr_pct = tr / c[i] if c[i] > 0 else 0
            all_entries.append((i, atr_pct))

    if not all_entries:
        return {}

    # Estimate bars per year
    max_bar = max(e[0] for e in all_entries)
    min_bar = min(e[0] for e in all_entries)
    total_range = max_bar - min_bar + 1
    bars_per_year = total_range / data_years if data_years > 0 else 35040

    # Sample: take 1% of data points to speed up (we just need medians)
    rng = np.random.default_rng(42)
    sample_size = min(len(all_entries), 500_000)
    indices = rng.choice(len(all_entries), size=sample_size, replace=False)
    sampled = [(all_entries[i][0], all_entries[i][1]) for i in indices]

    results = {}
    for year_offset in range(int(data_years) + 1):
        yr_start = min_bar + int(year_offset * bars_per_year)
        yr_end = min_bar + int((year_offset + 1) * bars_per_year)
        yr_atrs = [atr for bar, atr in sampled if yr_start <= bar < yr_end]
        if yr_atrs:
            results[2020 + year_offset] = float(np.median(yr_atrs))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting Helpers
# ══════════════════════════════════════════════════════════════════════════════


def print_metrics(label: str, m: dict, boot: dict | None = None) -> None:
    print(
        f"  [{label:25s}] N={m['N']:5d}  PF={m['PF']:.3f}  "
        f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
        f"MaxDD={m['max_dd_pct']:.1f}%  Calmar={m.get('calmar', 0):.0f}  "
        f"AvgWin={m.get('avg_win_pct', 0):.3f}%  AvgLoss={m.get('avg_loss_pct', 0):.3f}%  "
        f"RR={m.get('RR', 0):.2f}"
        + (f"  Boot P(PF≥1)={boot['P_PF_ge_1']:.1%}" if boot else "")
    )


def yearly_decomposition(
    trades: list[tuple[str, int, int, float]],
    data_years: float,
    max_concurrent: int,
    straddle_syms: frozenset[str] | None = None,
) -> dict[int, dict]:
    """Detailed per-year decomposition: N, PF, WR, AvgWin, AvgLoss, EV, hold bars."""
    all_entries = [t[1] for t in trades]
    if not all_entries:
        return {}
    min_bar = min(all_entries)
    max_bar = max(all_entries)
    total_range = max_bar - min_bar + 1
    bars_per_year = total_range / data_years if data_years > 0 else 35040

    results = {}
    for year_offset in range(int(data_years) + 1):
        yr_start = min_bar + int(year_offset * bars_per_year)
        yr_end = min_bar + int((year_offset + 1) * bars_per_year)
        yr_trades = [(s, e, x, p) for s, e, x, p in trades if yr_start <= e < yr_end]
        if not yr_trades:
            continue

        pnl = portfolio_sim(yr_trades, max_concurrent, straddle_syms)
        if len(pnl) == 0:
            continue

        m = compute_metrics(pnl, 1.0)
        year_label = 2020 + year_offset

        # Holding period stats (in 15m bars)
        hold_bars = [x - e for _, e, x, _ in yr_trades]
        avg_hold = float(np.mean(hold_bars)) if hold_bars else 0
        med_hold = float(np.median(hold_bars)) if hold_bars else 0

        # Exit type approximation for straddle (combined PnL)
        pnl_arr = np.array([t[3] for t in yr_trades])
        n_big_loss = int(np.sum(pnl_arr < -0.02))  # >2% loss
        n_big_win = int(np.sum(pnl_arr > 0.02))     # >2% win

        m["avg_hold_bars"] = round(avg_hold, 0)
        m["med_hold_bars"] = round(med_hold, 0)
        m["n_big_loss"] = n_big_loss
        m["n_big_win"] = n_big_win
        results[year_label] = m

    return results


def print_yearly_table(results: dict[int, dict], label: str = "") -> None:
    """Print formatted yearly decomposition table."""
    print(f"\n  {'Year':>4s}  {'N':>5s}  {'PF':>6s}  {'WR%':>5s}  {'EV(bp)':>7s}  "
          f"{'AvgWin%':>7s}  {'AvgLoss%':>8s}  {'R:R':>5s}  {'AvgHold':>7s}  "
          f"{'BigWin':>6s}  {'BigLoss':>7s}  {'MaxDD%':>6s}")
    print("  " + "-" * 95)
    for yr in sorted(results.keys()):
        m = results[yr]
        flag = "✅" if m["PF"] > 1.0 else "❌"
        print(
            f"  {yr:4d}  {m['N']:5d}  {m['PF']:6.3f}  {m['WR']:5.1f}  {m['EV_bp']:+7.1f}  "
            f"{m['avg_win_pct']:7.3f}  {m['avg_loss_pct']:8.3f}  {m['RR']:5.2f}  "
            f"{m['avg_hold_bars']:7.0f}  {m['n_big_win']:6d}  {m['n_big_loss']:7d}  "
            f"{m['max_dd_pct']:6.1f}  {flag}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    cfg = VolStraddleConfig()

    print("=" * 100)
    print(f"r88: Edge Decay Deep Research — commit {commit}")
    print("=" * 100)

    # ── Phase 0: Config Verification ──
    print("\n── Phase 0: Config Verification ──")
    print(f"  D_mom8: d_sl={cfg.d_sl}  d_tp={cfg.d_tp}  d_trail_act={cfg.d_trail_activate}  d_trail_dist={cfg.d_trail_distance}")
    print(f"  S_d8:   s_sl={cfg.s_sl}  s_tp={cfg.s_tp}  s_trail_act={cfg.s_trail_activate}  s_trail_dist={cfg.s_trail_distance}")
    print(f"  VR:     sp={cfg.vr_short}  lp={cfg.vr_long}  d_vr={cfg.d_vr_threshold}  s_vr={cfg.s_vr_threshold}  s_dur={cfg.s_squeeze_dur}")
    print(f"  Sizing: base_lev={cfg.base_leverage}  max_concurrent={cfg.max_concurrent}  L/N={cfg.base_leverage/cfg.max_concurrent:.1f}")
    print(f"  CW:     cw_enabled={cfg.cw_enabled}  min_conviction={cfg.min_conviction_score}")
    print(f"  Universe: {len(cfg.symbol_whitelist)} symbols")

    # ── Data Loading ──
    print("\n── Data Loading ──")
    t0 = time.perf_counter()
    data_15m, data_30m, data_1h, data_4h, data_years, max_bars = load_all_data(cfg)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: Baseline Reproduction
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("Phase 1: Baseline Reproduction")
    print("=" * 100)

    t1 = time.perf_counter()
    strd_trades = collect_trades_trailing(
        cfg, data_15m, data_1h, data_4h, "S_d8",
        tf_group=list(G_4TF), data_30m=data_30m,
    )
    vr_trades = collect_trades_trailing(
        cfg, data_15m, data_1h, data_4h, "D_mom8",
        tf_group=list(G_3TF), data_30m=data_30m,
    )
    strd_syms = get_straddle_syms(strd_trades)

    pnl_strd = portfolio_sim(strd_trades, 5)
    m_strd = compute_metrics(pnl_strd, data_years)
    print_metrics("S_d8 standalone", m_strd)

    pnl_vr = portfolio_sim(vr_trades, cfg.max_concurrent)
    m_vr = compute_metrics(pnl_vr, data_years)
    print_metrics("D_mom8 standalone", m_vr)

    combined = vr_trades + strd_trades
    pnl_combined = portfolio_sim(combined, cfg.max_concurrent, straddle_syms=strd_syms)
    m_combined = compute_metrics(pnl_combined, data_years)
    boot = bootstrap_pf(pnl_combined, n_boot=5000)
    print_metrics("Combined D+S", m_combined, boot)
    print(f"  Time: {time.perf_counter() - t1:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: T1 — Edge Decomposition by Year
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("Phase 2: T1 — Edge Decomposition by Year")
    print("=" * 100)
    print("\nQuestion: WHERE is the edge leaking? Smaller wins? Lower WR? Bigger losses?")

    # S_d8 yearly
    print("\n  ── S_d8 (Straddle) Yearly ──")
    yr_strd = yearly_decomposition(strd_trades, data_years, 5)
    print_yearly_table(yr_strd, "S_d8")

    # D_mom8 yearly
    print("\n  ── D_mom8 (Directional) Yearly ──")
    yr_vr = yearly_decomposition(vr_trades, data_years, cfg.max_concurrent)
    print_yearly_table(yr_vr, "D_mom8")

    # Combined yearly
    print("\n  ── Combined D+S Yearly ──")
    yr_combined = yearly_decomposition(combined, data_years, cfg.max_concurrent, strd_syms)
    print_yearly_table(yr_combined, "Combined")

    # Decay analysis
    print("\n  ── Decay Diagnosis ──")
    years = sorted(yr_combined.keys())
    if len(years) >= 2:
        first_yr = yr_combined[years[0]]
        last_yr = yr_combined[years[-1]]
        print(f"  EV decay:     {first_yr['EV_bp']:+.1f}bp → {last_yr['EV_bp']:+.1f}bp  "
              f"({(last_yr['EV_bp'] - first_yr['EV_bp']) / abs(first_yr['EV_bp']) * 100:+.0f}%)")
        print(f"  WR change:    {first_yr['WR']:.1f}% → {last_yr['WR']:.1f}%  "
              f"({last_yr['WR'] - first_yr['WR']:+.1f}pp)")
        print(f"  AvgWin:       {first_yr['avg_win_pct']:.3f}% → {last_yr['avg_win_pct']:.3f}%  "
              f"({(last_yr['avg_win_pct'] - first_yr['avg_win_pct']) / first_yr['avg_win_pct'] * 100:+.0f}%)" if first_yr['avg_win_pct'] > 0 else "")
        print(f"  AvgLoss:      {first_yr['avg_loss_pct']:.3f}% → {last_yr['avg_loss_pct']:.3f}%  "
              f"({(last_yr['avg_loss_pct'] - first_yr['avg_loss_pct']) / first_yr['avg_loss_pct'] * 100:+.0f}%)" if first_yr['avg_loss_pct'] > 0 else "")
        print(f"  R:R change:   {first_yr['RR']:.2f} → {last_yr['RR']:.2f}")
        print(f"  N change:     {first_yr['N']} → {last_yr['N']}")

    # ATR regime analysis
    print("\n  ── Yearly ATR Regime (15m bars) ──")
    yearly_atr = compute_yearly_atr_stats(data_15m, data_years)
    if yearly_atr:
        base_atr = list(yearly_atr.values())[0]
        for yr, atr in sorted(yearly_atr.items()):
            ratio = atr / base_atr if base_atr > 0 else 0
            print(f"    {yr}: median ATR% = {atr*100:.4f}%  (ratio vs {min(yearly_atr.keys())}: {ratio:.2f}x)")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 3: T2 — Rolling VR Percentile Threshold (Adaptive Entry)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("Phase 3: T2 — Rolling VR Percentile Threshold (Adaptive Entry)")
    print("=" * 100)
    print("\nHypothesis: In calm 2023+ regime, static VR threshold (0.57) is too strict.")
    print("Using rolling percentile auto-adapts to what 'squeeze' means in current vol regime.")

    t3 = time.perf_counter()

    # Sweep percentiles: 15th, 20th, 25th, 30th
    percentiles = [15, 20, 25, 30]
    # Window: 6 months (17520 15m bars) and 3 months (8760)
    windows = [
        (8760, "3mo"),
        (17520, "6mo"),
    ]

    print(f"\n  Sweep: percentiles={percentiles}, windows={[w[1] for w in windows]}")
    print(f"  Static baseline: s_vr={cfg.s_vr_threshold}, d_vr={cfg.d_vr_threshold}")

    adaptive_results: list[dict] = []

    for window_bars, window_label in windows:
        # Compute adaptive thresholds for each TF
        for pctile in percentiles:
            label = f"P{pctile}_{window_label}"
            print(f"\n  Computing {label}...", end=" ", flush=True)

            # Compute adaptive VR for each TF's data
            adaptive_15m = compute_adaptive_vr_thresholds(data_15m, vr_key, pctile, window_bars)
            adaptive_30m = compute_adaptive_vr_thresholds(data_30m, vr_key, pctile, window_bars // 2)
            adaptive_1h = compute_adaptive_vr_thresholds(data_1h, vr_key, pctile, window_bars // 4)
            adaptive_4h = compute_adaptive_vr_thresholds(data_4h, vr_key, pctile, window_bars // 16)

            # Collect straddle trades with adaptive thresholds
            # For each TF, pass the corresponding adaptive array
            strd_adaptive = _collect_trades_adaptive_multi_tf(
                cfg, data_15m, data_30m, data_1h, data_4h,
                "S_d8", list(G_4TF),
                {1: adaptive_15m, 2: adaptive_30m, 4: adaptive_1h, 16: adaptive_4h},
            )

            vr_adaptive = _collect_trades_adaptive_multi_tf(
                cfg, data_15m, data_30m, data_1h, data_4h,
                "D_mom8", list(G_3TF),
                {1: adaptive_15m, 2: adaptive_30m, 4: adaptive_1h, 16: adaptive_4h},
            )

            strd_syms_a = get_straddle_syms(strd_adaptive)
            combined_a = vr_adaptive + strd_adaptive
            pnl_a = portfolio_sim(combined_a, cfg.max_concurrent, straddle_syms=strd_syms_a)
            m_a = compute_metrics(pnl_a, data_years)
            print(f"N={m_a['N']}, PF={m_a['PF']:.3f}, EV={m_a['EV_bp']:+.1f}bp")

            # Yearly breakdown
            yr_a = yearly_decomposition(combined_a, data_years, cfg.max_concurrent, strd_syms_a)

            recent_pf = 0.0
            recent_n = 0
            recent_ev = 0.0
            for yr in [2023, 2024, 2025]:
                if yr in yr_a:
                    recent_pf += yr_a[yr]["PF"] * yr_a[yr]["N"]
                    recent_n += yr_a[yr]["N"]
                    recent_ev += yr_a[yr]["EV_bp"] * yr_a[yr]["N"]
            if recent_n > 0:
                recent_pf /= recent_n
                recent_ev /= recent_n

            adaptive_results.append({
                "label": label,
                "percentile": pctile,
                "window": window_label,
                "N": m_a["N"],
                "PF": m_a["PF"],
                "EV_bp": m_a["EV_bp"],
                "MaxDD": m_a["max_dd_pct"],
                "recent_pf": recent_pf,
                "recent_n": recent_n,
                "recent_ev": recent_ev,
                "yearly": yr_a,
            })

    # Comparison table
    print(f"\n  ── T2 Results: Adaptive VR Threshold vs Static ──")
    print(f"  {'Config':>12s}  {'N':>5s}  {'PF':>6s}  {'EV(bp)':>7s}  {'MaxDD%':>6s}  "
          f"{'2023-25 PF':>10s}  {'2023-25 N':>9s}  {'2023-25 EV':>10s}")
    print("  " + "-" * 85)
    print(f"  {'Static':>12s}  {m_combined['N']:5d}  {m_combined['PF']:6.3f}  "
          f"{m_combined['EV_bp']:+7.1f}  {m_combined['max_dd_pct']:6.1f}  "
          f"{'—':>10s}  {'—':>9s}  {'—':>10s}")
    for r in adaptive_results:
        print(f"  {r['label']:>12s}  {r['N']:5d}  {r['PF']:6.3f}  "
              f"{r['EV_bp']:+7.1f}  {r['MaxDD']:6.1f}  "
              f"{r['recent_pf']:10.3f}  {r['recent_n']:9d}  {r['recent_ev']:+10.1f}")

    # Print yearly breakdown for best adaptive config
    if adaptive_results:
        best = max(adaptive_results, key=lambda r: r["recent_pf"] * min(r["recent_n"], 500))
        print(f"\n  ── Best adaptive config: {best['label']} ──")
        print_yearly_table(best["yearly"], best["label"])

    print(f"\n  T2 time: {time.perf_counter() - t3:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 4: T3 — Regime-Scaled Trail Parameters (Adaptive Exit)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("Phase 4: T3 — Regime-Scaled Trail Parameters (Adaptive Exit)")
    print("=" * 100)
    print("\nHypothesis: In low-vol 2023+, static trail_activate=1.5% is too wide.")
    print("Scaling trail params by ATR ratio could capture smaller but valid moves.")

    t4 = time.perf_counter()

    # Reference ATR: median across all years
    if yearly_atr:
        ref_atr = float(np.median(list(yearly_atr.values())))
        print(f"  Reference ATR: {ref_atr*100:.4f}% (median all years)")
    else:
        ref_atr = 0.003
        print(f"  Reference ATR: {ref_atr*100:.4f}% (default)")

    # Trail multipliers to test (relative to current static params)
    trail_multipliers = [0.5, 0.67, 0.80, 1.0, 1.25, 1.5]
    print(f"  Sweep: trail_act multipliers = {trail_multipliers}")
    print(f"  Base S_d8: trail_act={cfg.s_trail_activate}, trail_dist={cfg.s_trail_distance}")
    print(f"  Base D_mom8: trail_act={cfg.d_trail_activate}, trail_dist={cfg.d_trail_distance}")

    trail_results: list[dict] = []

    for mult in trail_multipliers:
        s_act = round(cfg.s_trail_activate * mult, 4)
        s_dist = round(cfg.s_trail_distance * mult, 4)
        d_act = round(cfg.d_trail_activate * mult, 4)
        d_dist = round(cfg.d_trail_distance * mult, 4)

        label = f"×{mult:.2f}"

        strd_t = collect_trades_trailing(
            cfg, data_15m, data_1h, data_4h, "S_d8",
            tf_group=list(G_4TF), data_30m=data_30m,
            trail_act_override=s_act, trail_dist_override=s_dist,
        )
        vr_t = collect_trades_trailing(
            cfg, data_15m, data_1h, data_4h, "D_mom8",
            tf_group=list(G_3TF), data_30m=data_30m,
            trail_act_override=d_act, trail_dist_override=d_dist,
        )
        strd_syms_t = get_straddle_syms(strd_t)
        combined_t = vr_t + strd_t
        pnl_t = portfolio_sim(combined_t, cfg.max_concurrent, straddle_syms=strd_syms_t)
        m_t = compute_metrics(pnl_t, data_years)

        yr_t = yearly_decomposition(combined_t, data_years, cfg.max_concurrent, strd_syms_t)

        # Recent period (2023-2025) weighted PF
        recent_pf = 0.0
        recent_n = 0
        for yr in [2023, 2024, 2025]:
            if yr in yr_t:
                recent_pf += yr_t[yr]["PF"] * yr_t[yr]["N"]
                recent_n += yr_t[yr]["N"]
        if recent_n > 0:
            recent_pf /= recent_n

        trail_results.append({
            "label": label,
            "mult": mult,
            "s_act": s_act, "s_dist": s_dist,
            "d_act": d_act, "d_dist": d_dist,
            "N": m_t["N"],
            "PF": m_t["PF"],
            "EV_bp": m_t["EV_bp"],
            "MaxDD": m_t["max_dd_pct"],
            "recent_pf": recent_pf,
            "recent_n": recent_n,
            "yearly": yr_t,
        })

    # Comparison table
    print(f"\n  ── T3 Results: Trail Parameter Scaling ──")
    print(f"  {'Mult':>6s}  {'s_act':>7s}  {'s_dist':>7s}  {'N':>5s}  {'PF':>6s}  "
          f"{'EV(bp)':>7s}  {'MaxDD%':>6s}  {'2023-25 PF':>10s}  {'2023-25 N':>9s}")
    print("  " + "-" * 80)
    for r in trail_results:
        marker = " ← current" if r["mult"] == 1.0 else ""
        print(f"  {r['label']:>6s}  {r['s_act']:7.4f}  {r['s_dist']:7.4f}  "
              f"{r['N']:5d}  {r['PF']:6.3f}  {r['EV_bp']:+7.1f}  {r['MaxDD']:6.1f}  "
              f"{r['recent_pf']:10.3f}  {r['recent_n']:9d}{marker}")

    # Best trail config
    if trail_results:
        best_trail = max(trail_results, key=lambda r: r["recent_pf"] if r["recent_n"] > 50 else 0)
        print(f"\n  ── Best trail config: {best_trail['label']} (s_act={best_trail['s_act']}, s_dist={best_trail['s_dist']}) ──")
        print_yearly_table(best_trail["yearly"], best_trail["label"])

    print(f"\n  T3 time: {time.perf_counter() - t4:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 5: T4 — Conviction Filter by Year
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("Phase 5: T4 — VR Threshold Sweep (Entry Quality)")
    print("=" * 100)
    print("\nApproach: Tighter VR thresholds = stricter squeeze definition = fewer but higher-quality signals.")
    print("Prior finding: r58 showed s_vr 0.55→0.59 improved both N AND PF simultaneously.")

    t5 = time.perf_counter()

    # Sweep s_vr thresholds
    s_vr_values = [0.50, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63]
    print(f"  Sweep: s_vr = {s_vr_values}")
    print(f"  Current: s_vr = {cfg.s_vr_threshold}")

    vr_sweep_results: list[dict] = []

    for s_vr_test in s_vr_values:
        strd_t = collect_trades_trailing(
            cfg, data_15m, data_1h, data_4h, "S_d8",
            tf_group=list(G_4TF), data_30m=data_30m,
            s_vr_override=s_vr_test,
        )
        strd_syms_t = get_straddle_syms(strd_t)
        pnl_t = portfolio_sim(strd_t, 5)
        m_t = compute_metrics(pnl_t, data_years)

        yr_t = yearly_decomposition(strd_t, data_years, 5)

        recent_pf = 0.0
        recent_n = 0
        for yr in [2023, 2024, 2025]:
            if yr in yr_t:
                recent_pf += yr_t[yr]["PF"] * yr_t[yr]["N"]
                recent_n += yr_t[yr]["N"]
        if recent_n > 0:
            recent_pf /= recent_n

        vr_sweep_results.append({
            "s_vr": s_vr_test,
            "N": m_t["N"],
            "PF": m_t["PF"],
            "EV_bp": m_t["EV_bp"],
            "MaxDD": m_t["max_dd_pct"],
            "WR": m_t["WR"],
            "recent_pf": recent_pf,
            "recent_n": recent_n,
            "yearly": yr_t,
        })

    print(f"\n  ── T4 Results: VR Threshold Sweep (S_d8 standalone) ──")
    print(f"  {'s_vr':>6s}  {'N':>5s}  {'PF':>6s}  {'WR%':>5s}  {'EV(bp)':>7s}  "
          f"{'MaxDD%':>6s}  {'2023-25 PF':>10s}  {'2023-25 N':>9s}")
    print("  " + "-" * 70)
    for r in vr_sweep_results:
        marker = " ← current" if abs(r["s_vr"] - cfg.s_vr_threshold) < 0.001 else ""
        print(f"  {r['s_vr']:6.3f}  {r['N']:5d}  {r['PF']:6.3f}  {r['WR']:5.1f}  "
              f"{r['EV_bp']:+7.1f}  {r['MaxDD']:6.1f}  "
              f"{r['recent_pf']:10.3f}  {r['recent_n']:9d}{marker}")

    print(f"\n  T4 time: {time.perf_counter() - t5:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 6: Summary & Recommendations
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("Phase 6: Summary & Recommendations")
    print("=" * 100)

    print(f"\n  Total runtime: {time.perf_counter() - t0:.0f}s")

    # T1 Summary
    print("\n  ── T1 Edge Decomposition ──")
    if years:
        print(f"  Edge leak primary driver: ", end="")
        wr_change = yr_combined.get(years[-1], {}).get("WR", 0) - yr_combined.get(years[0], {}).get("WR", 0)
        rr_change = yr_combined.get(years[-1], {}).get("RR", 0) - yr_combined.get(years[0], {}).get("RR", 0)
        if abs(rr_change) > abs(wr_change / 10):
            print("R:R compression (avg_win shrinking relative to avg_loss)")
        else:
            print("WR decline (signal accuracy decreasing)")

    # T2 Summary
    print("\n  ── T2 Adaptive VR ──")
    if adaptive_results:
        best_a = max(adaptive_results, key=lambda r: r["recent_pf"] * min(r["recent_n"], 500))
        print(f"  Best: {best_a['label']} — 2023-25 PF={best_a['recent_pf']:.3f} (N={best_a['recent_n']})")
        improvement = (best_a["recent_pf"] - 1.0) / max(m_combined.get("PF", 1.0) - 1.0, 0.01) * 100
        print(f"  vs static: overall PF {m_combined['PF']:.3f} → {best_a['PF']:.3f}")

    # T3 Summary
    print("\n  ── T3 Trail Scaling ──")
    if trail_results:
        best_t = max(trail_results, key=lambda r: r["recent_pf"] if r["recent_n"] > 50 else 0)
        current = next((r for r in trail_results if r["mult"] == 1.0), None)
        print(f"  Best: {best_t['label']} — 2023-25 PF={best_t['recent_pf']:.3f} (N={best_t['recent_n']})")
        if current:
            print(f"  vs current (×1.00): PF {current['PF']:.3f} → {best_t['PF']:.3f}")

    # T4 Summary
    print("\n  ── T4 VR Threshold Sweep ──")
    if vr_sweep_results:
        best_v = max(vr_sweep_results, key=lambda r: r["recent_pf"] * min(r["recent_n"], 300))
        print(f"  Best: s_vr={best_v['s_vr']:.3f} — 2023-25 PF={best_v['recent_pf']:.3f} (N={best_v['recent_n']})")
        current_v = next((r for r in vr_sweep_results if abs(r["s_vr"] - cfg.s_vr_threshold) < 0.001), None)
        if current_v:
            print(f"  vs current (s_vr={cfg.s_vr_threshold}): PF {current_v['PF']:.3f} → {best_v['PF']:.3f}")

    # Overall recommendation
    print("\n  ══ Overall Recommendation ══")
    print("  See report in docs/backtest_results/ for detailed analysis.")

    print("\n" + "=" * 100)
    print("r88 complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: Collect trades with per-TF adaptive VR thresholds
# ══════════════════════════════════════════════════════════════════════════════


def _collect_trades_adaptive_multi_tf(
    cfg_obj: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    active_tfs: list,
    adaptive_by_tf: dict[int, dict[str, np.ndarray]],
) -> list[tuple[str, int, int, float]]:
    """Collect trades using per-TF adaptive VR thresholds."""
    sp = cfg_obj.vr_short
    lp = cfg_obj.vr_long
    vr_key = f"vr_{sp}_{lp}"
    is_straddle = strat == "S_d8"

    d_sl = cfg_obj.d_sl
    s_sl = cfg_obj.s_sl
    d_tp = cfg_obj.d_tp
    s_tp = cfg_obj.s_tp
    s_dur = cfg_obj.s_squeeze_dur

    if is_straddle:
        trail_act = cfg_obj.s_trail_activate
        trail_dist = cfg_obj.s_trail_distance
    else:
        trail_act = cfg_obj.d_trail_activate
        trail_dist = cfg_obj.d_trail_distance

    tf_data_map: dict[int, dict] = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}

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

            # Get adaptive threshold for this TF and symbol
            adaptive_arr = adaptive_by_tf.get(ratio, {}).get(sym)
            if adaptive_arr is None:
                continue

            sl = s_sl if is_straddle else d_sl
            tp = s_tp if is_straddle else d_tp
            mh = mh_s if is_straddle else mh_d
            cost = COST_STRADDLE if is_straddle else COST_DIRECTIONAL

            if is_straddle:
                sig = _gen_vr_dur_adaptive(vr, adaptive_arr, s_dur, warmup, ratio)
                p, e, x = sim_straddle_trailing(
                    c, h, lo, sig, sl, tp, mh, cost, trail_act, trail_dist
                )
            else:
                sig = _gen_vr_adaptive(vr, adaptive_arr, warmup, ratio)
                dire = make_direction_momentum(sig, d["mom_8"])
                p, e, x = sim_directional_trailing(
                    c, h, lo, sig, dire, sl, tp, mh, cost, trail_act, trail_dist
                )

            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio + offset
                exit_15m = int(x[k]) * ratio + (ratio - 1) + offset
                sym_trades.append((entry_15m, exit_15m, float(p[k])))

        all_trades.extend(_dedup(sym, sym_trades))

    return all_trades


if __name__ == "__main__":
    main()
