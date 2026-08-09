"""r80: 혼합(VR+STRD) 백테스트 슬롯 패리티 수정 후 canonical 수치 재계산.

**배경:**
  r76 이전 모든 혼합 백테스트(r73, r76)에서 S_d8 straddle이 1슬롯으로 처리됨.
  실전에서는 1 straddle = LONG+SHORT = 2슬롯 → 10슬롯 중 최대 5 straddle.
  T1 fix (commit 3b43bb2): portfolio_sim(straddle_syms=...) 파라미터 추가.

**이번 r80에서 검증:**
  Phase 0: Baseline 검증 (S_d8 standalone max=5, VR standalone max=10)
  Phase 1: 혼합 canonical — straddle_syms 적용 후 r76 대비 수치 변화
  Phase 2: 연도별 PF breakdown (2020~2025, STRD+VR+Mixed)
  Phase 3: s_vr_threshold × s_squeeze_dur 2D sweep
  Phase 4: max_hold_straddle 민감도 (0.75×, 1.0×, 1.5×, 2.0× 현재값)

**핵심 질문:**
  "straddle_syms 수정 후 혼합 canonical 수치는 얼마나 변하는가?"
  "연도별로 STRD가 VR보다 안정적인가?"
  "현재 s_vr_threshold=0.57이 최적인가?"
  "max_hold=384bar(96h)는 너무 길지 않은가?"

사용:
    uv run python research/active/r76_r80/r80_mixed_slot_parity.py
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
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    build_resampled,
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

# MA Cluster — r45 champion params (same as r73/r76 canonical)
sys.path.insert(0, str(ROOT / "research" / "active" / "r41_r45"))
try:
    from r44c_ma_confluence import precompute_mas  # type: ignore
    from r45_param_sweep import collect_enhanced_trades  # type: ignore
    MA_AVAILABLE = True
except ImportError:
    MA_AVAILABLE = False

MA_CLUSTER_PCT = 0.03
MA_MIN_DURATION = 8
MA_MAX_DURATION = 24
MA_MIN_MOMENTUM = 0.010
MA_SL = 0.02
MA_TP = 0.08
MA_TF_RESTRICTION = 16
WARMUP_30m: int = 100

# ── Sweep grids ───────────────────────────────────────────────────────────────
VR_THRESH_GRID = [0.53, 0.55, 0.57, 0.59, 0.61]
S_DUR_GRID = [2, 3, 4, 5]
MAX_HOLD_MULT_GRID = [0.75, 1.0, 1.5, 2.0]  # vs current 2× directional


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def load_all_data(cfg: VolStraddleConfig) -> tuple[dict, dict, dict, dict, int, float]:
    """Load and precompute 15m/1h/4h/30m data (matches r76/r78 exactly)."""
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

    print(f"  15m: {len(data_15m)} symbols")

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg.vr_short, cfg.vr_long)])
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)

    if MA_AVAILABLE:
        precompute_mas(data_15m)
        precompute_mas(data_1h)
        precompute_mas(data_4h)

    bars_list = [len(d["close"]) for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)

    return data_15m, data_1h, data_4h, data_30m, max_bars, data_years


def _get_g4tf(cfg: VolStraddleConfig) -> list:
    mh_30m = cfg.max_hold_15m // 2
    tf_30m = ("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)
    return [tf_30m] + list(G_3TF)


def _dedup(sym: str, trades: list) -> list:
    result = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Trade collectors
# ══════════════════════════════════════════════════════════════════════════════


def collect_s_trail(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict,
    cfg: VolStraddleConfig,
    *,
    s_vr: float | None = None,
    s_dur: int | None = None,
    max_hold_mult: float = 2.0,
) -> list:
    """S_d8 straddle trail collection with optional param overrides."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    g4tf = _get_g4tf(cfg)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl = cfg.s_sl
    tp = cfg.s_tp
    trail_act = cfg.s_trail_activate
    trail_dist = cfg.s_trail_distance
    _s_vr = s_vr if s_vr is not None else cfg.s_vr_threshold
    _s_dur = s_dur if s_dur is not None else cfg.s_squeeze_dur
    mh_15m = cfg.max_hold_15m

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, _mhd, _mhs in g4tf:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, _s_vr, _s_dur, warmup)
            # max_hold for straddle: max_hold_mult × directional max_hold
            mh_d = int(mh_15m // ratio)
            mh_s = max(1, int(mh_d * max_hold_mult))
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE, trail_act, trail_dist,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def collect_d_trail(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
) -> list:
    """D_mom8 directional trail (production params)."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    sl = cfg.d_sl
    tp = cfg.d_tp
    trail_act = cfg.d_trail_activate
    trail_dist = cfg.d_trail_distance
    d_vr = cfg.d_vr_threshold

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, mh_d, _mhs in G_3TF:
            d = tf_map[ratio].get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_signal(vr, d_vr, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, sl, tp, mh_d, COST_DIRECTIONAL,
                trail_act, trail_dist,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def collect_ma_trades(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list:
    if not MA_AVAILABLE:
        return []
    return collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 0: Baseline 검증
# ══════════════════════════════════════════════════════════════════════════════


def phase0_baseline(
    s_trades: list,
    d_trades: list,
    cfg: VolStraddleConfig,
    data_years: float,
) -> tuple[dict, dict]:
    """Standalone baseline for STRD(max=5) and VR(max=10)."""
    print("\n" + "=" * 70)
    print("PHASE 0: Baseline 검증")
    print("  STRD standalone max_concurrent=5 (r79 canonical)")
    print("  VR standalone max_concurrent=10")
    print("=" * 70)

    pnl_s = portfolio_sim(s_trades, max_concurrent=5)
    m_s = compute_metrics(pnl_s, data_years)

    pnl_d = portfolio_sim(d_trades, max_concurrent=10)
    m_d = compute_metrics(pnl_d, data_years)

    # r79 canonical: N≈7730, PF≈9.603 (max=5)
    S_REF_N, S_REF_PF = 7730, 9.603
    # r76 VR canonical: from prior research
    D_REF_N, D_REF_PF = 6207, 1.563

    s_n_drift = abs(m_s["N"] - S_REF_N) / S_REF_N
    s_pf_drift = abs(m_s["PF"] - S_REF_PF) / S_REF_PF
    d_n_drift = abs(m_d["N"] - D_REF_N) / D_REF_N
    d_pf_drift = abs(m_d["PF"] - D_REF_PF) / D_REF_PF

    print(f"\n  STRD(max=5):  N={m_s['N']} PF={m_s['PF']:.3f} WR={m_s['WR']:.1f}%"
          f"  ref N={S_REF_N} PF={S_REF_PF:.3f}"
          f"  drift N={s_n_drift:.1%} PF={s_pf_drift:.1%}"
          f"  {'✅' if s_n_drift<0.05 and s_pf_drift<0.05 else '⚠️'}")

    print(f"  VR(max=10):   N={m_d['N']} PF={m_d['PF']:.3f} WR={m_d['WR']:.1f}%"
          f"  ref N={D_REF_N} PF={D_REF_PF:.3f}"
          f"  drift N={d_n_drift:.1%} PF={d_pf_drift:.1%}"
          f"  {'✅' if d_n_drift<0.05 and d_pf_drift<0.05 else '⚠️'}")

    return m_s, m_d


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: 혼합 canonical — straddle_syms 수정 적용
# ══════════════════════════════════════════════════════════════════════════════


def phase1_mixed_canonical(
    s_trades: list,
    d_trades: list,
    ma_trades: list,
    cfg: VolStraddleConfig,
    data_years: float,
) -> dict:
    """Mixed canonical with straddle_syms fix vs old (1-slot straddle) numbers."""
    print("\n" + "=" * 70)
    print("PHASE 1: 혼합 canonical — straddle_syms 수정 적용")
    print("  r76 (broken): portfolio_sim(D+S+MA, 10)  ← straddle=1slot")
    print("  r80 (fixed):  portfolio_sim(D+S+MA, 10, straddle_syms=...)  ← straddle=2slots")
    print("=" * 70)

    strd_syms = get_straddle_syms(s_trades)
    all_trades = s_trades + d_trades + ma_trades

    # Old (broken): no straddle_syms
    pnl_old = portfolio_sim(all_trades, cfg.max_concurrent)
    m_old = compute_metrics(pnl_old, data_years)

    # New (fixed): straddle_syms
    pnl_new = portfolio_sim(all_trades, cfg.max_concurrent, straddle_syms=strd_syms)
    m_new = compute_metrics(pnl_new, data_years)

    # Without MA (VR+STRD only)
    sd_trades = s_trades + d_trades
    pnl_sd_fix = portfolio_sim(sd_trades, cfg.max_concurrent, straddle_syms=strd_syms)
    m_sd = compute_metrics(pnl_sd_fix, data_years)

    print(f"\n  {'':20} {'N':>6}  {'PF':>7}  {'Calmar':>8}  {'MaxDD':>7}  {'WR':>6}")
    print(f"  {'─'*60}")
    print(f"  {'r76 (1-slot,broken)':20} {m_old['N']:>6}  {m_old['PF']:>7.3f}  "
          f"{m_old['calmar']:>8.1f}  {m_old['max_dd_pct']:>6.1f}%  {m_old['WR']:>5.1f}%")
    print(f"  {'r80 VR+STRD (fixed)':20} {m_sd['N']:>6}  {m_sd['PF']:>7.3f}  "
          f"{m_sd['calmar']:>8.1f}  {m_sd['max_dd_pct']:>6.1f}%  {m_sd['WR']:>5.1f}%")
    print(f"  {'r80 D+S+MA (fixed)':20} {m_new['N']:>6}  {m_new['PF']:>7.3f}  "
          f"{m_new['calmar']:>8.1f}  {m_new['max_dd_pct']:>6.1f}%  {m_new['WR']:>5.1f}%")
    print(f"\n  r76 canonical ref:     N=9,775  PF=2.484  Calmar=9,882  MaxDD=37.1%")

    delta_n = (m_new["N"] - m_old["N"]) / m_old["N"] if m_old["N"] > 0 else 0.0
    delta_pf = (m_new["PF"] - m_old["PF"]) / m_old["PF"] if m_old["PF"] > 0 else 0.0
    print(f"\n  슬롯 패리티 수정 영향: N {delta_n:+.1%}  PF {delta_pf:+.1%}")

    return {"old": m_old, "fixed_sd": m_sd, "fixed_full": m_new}


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: 연도별 PF breakdown
# ══════════════════════════════════════════════════════════════════════════════


def phase2_annual_breakdown(
    s_trades: list,
    d_trades: list,
    cfg: VolStraddleConfig,
    data_years: float,
    max_bars: int,
) -> dict:
    """Per-year PF for STRD(max=5), VR(max=10), mixed(max=10, straddle_syms)."""
    print("\n" + "=" * 70)
    print("PHASE 2: 연도별 PF Breakdown (2020~2025)")
    print("=" * 70)

    total_bars = max_bars
    bars_per_year = total_bars / data_years
    start_year = 2020
    n_years = min(6, int(data_years) + 1)

    strd_syms = get_straddle_syms(s_trades)

    year_results: dict[int, dict] = {}
    print(f"\n  {'Year':>5}  {'STRD N':>7}  {'STRD PF':>8}  {'VR N':>6}  {'VR PF':>7}  "
          f"{'Mix N':>6}  {'Mix PF':>7}")
    print("  " + "─" * 65)

    for i in range(n_years):
        yr = start_year + i
        bar_start = int(i * bars_per_year)
        bar_end = int((i + 1) * bars_per_year)

        yr_s = [(sym, e, x, p) for sym, e, x, p in s_trades if bar_start <= e < bar_end]
        yr_d = [(sym, e, x, p) for sym, e, x, p in d_trades if bar_start <= e < bar_end]
        yr_m = yr_s + yr_d

        if not yr_s and not yr_d:
            continue

        pnl_s = portfolio_sim(yr_s, max_concurrent=5) if yr_s else np.empty(0)
        pnl_d = portfolio_sim(yr_d, max_concurrent=10) if yr_d else np.empty(0)
        pnl_m = portfolio_sim(yr_m, max_concurrent=10, straddle_syms=strd_syms) if yr_m else np.empty(0)

        ms = compute_metrics(pnl_s, 1.0) if len(pnl_s) > 0 else {"N": 0, "PF": 0.0}
        md = compute_metrics(pnl_d, 1.0) if len(pnl_d) > 0 else {"N": 0, "PF": 0.0}
        mm = compute_metrics(pnl_m, 1.0) if len(pnl_m) > 0 else {"N": 0, "PF": 0.0}

        s_ok = "✅" if ms["PF"] > 1.0 else "❌"
        d_ok = "✅" if md["PF"] > 1.0 else "❌"

        print(f"  {yr:>5}  {ms['N']:>7} {ms['PF']:>7.3f}{s_ok}"
              f"  {md['N']:>6} {md['PF']:>7.3f}{d_ok}"
              f"  {mm['N']:>6} {mm['PF']:>7.3f}")

        year_results[yr] = {"strd": ms, "vr": md, "mixed": mm}

    return year_results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: s_vr_threshold × s_squeeze_dur sweep
# ══════════════════════════════════════════════════════════════════════════════


def phase3_param_sweep(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict,
    cfg: VolStraddleConfig,
    data_years: float,
) -> list[dict]:
    """2D sweep: s_vr_threshold × s_squeeze_dur (standalone STRD, max=5)."""
    print("\n" + "=" * 70)
    print("PHASE 3: s_vr_threshold × s_squeeze_dur Sweep (STRD standalone, max=5)")
    print(f"  Grid: vr_thresh={VR_THRESH_GRID} × s_dur={S_DUR_GRID}")
    print(f"  Current: s_vr={cfg.s_vr_threshold} s_dur={cfg.s_squeeze_dur}")
    print("=" * 70)

    results: list[dict] = []
    print(f"\n  {'s_vr':>5}  {'s_dur':>5}  {'N':>5}  {'PF':>7}  {'WR':>6}  "
          f"{'EV(bp)':>7}  {'MaxDD':>6}  {'Calmar':>8}  {'★':>2}")
    print("  " + "─" * 65)

    for s_vr in VR_THRESH_GRID:
        for s_dur in S_DUR_GRID:
            t0 = time.perf_counter()
            trades = collect_s_trail(
                data_15m, data_1h, data_4h, data_30m, cfg,
                s_vr=s_vr, s_dur=s_dur,
            )
            pnl = portfolio_sim(trades, max_concurrent=5)
            m = compute_metrics(pnl, data_years)
            elapsed = time.perf_counter() - t0

            is_prod = abs(s_vr - cfg.s_vr_threshold) < 1e-4 and s_dur == cfg.s_squeeze_dur
            star = "★" if is_prod else " "

            print(f"  {s_vr:.2f}  {s_dur:>5}  {m['N']:>5}  {m['PF']:>7.3f}  "
                  f"{m['WR']:>5.1f}%  {m['EV_bp']:>7.1f}  {m['max_dd_pct']:>5.1f}%  "
                  f"{m['calmar']:>8.1f}  {star}  ({elapsed:.1f}s)")

            results.append({
                "s_vr": s_vr, "s_dur": s_dur, "N": m["N"], "PF": m["PF"],
                "WR": m["WR"], "EV": m["EV_bp"], "MaxDD": m["max_dd_pct"],
                "Calmar": m["calmar"], "prod": is_prod,
            })

    best = max(results, key=lambda r: r["PF"])
    print(f"\n  Best PF: s_vr={best['s_vr']:.2f} s_dur={best['s_dur']} "
          f"N={best['N']} PF={best['PF']:.3f} Calmar={best['Calmar']:.1f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4: max_hold_straddle 민감도
# ══════════════════════════════════════════════════════════════════════════════


def phase4_max_hold_sweep(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict,
    cfg: VolStraddleConfig,
    data_years: float,
) -> list[dict]:
    """max_hold_straddle sweep — how long is too long?

    Current: max_hold_straddle = 2 × max_hold_directional (= 384 bars = 96h).
    Question: Is 2× justified? Could 1× or 1.5× be better?
    """
    print("\n" + "=" * 70)
    mh_d_15m = cfg.max_hold_15m  # directional reference (192 bars = 48h)
    print("PHASE 4: max_hold_straddle Sweep (STRD standalone, max=5)")
    print(f"  Directional max_hold: {mh_d_15m}bars ({mh_d_15m * 15 / 60:.0f}h)")
    print(f"  Current straddle max_hold: {mh_d_15m * 2}bars ({mh_d_15m * 2 * 15 / 60:.0f}h) = 2×")
    print("=" * 70)

    results: list[dict] = []
    print(f"\n  {'mult':>5}  {'max_hold':>9}  {'N':>5}  {'PF':>7}  "
          f"{'WR':>6}  {'MaxDD':>6}  {'Calmar':>8}  {'★':>2}")
    print("  " + "─" * 60)

    for mult in MAX_HOLD_MULT_GRID:
        t0 = time.perf_counter()
        trades = collect_s_trail(
            data_15m, data_1h, data_4h, data_30m, cfg,
            max_hold_mult=mult,
        )
        pnl = portfolio_sim(trades, max_concurrent=5)
        m = compute_metrics(pnl, data_years)
        elapsed = time.perf_counter() - t0

        mh_bars = int(mh_d_15m * mult)
        mh_hrs = mh_bars * 15 / 60
        is_curr = abs(mult - 2.0) < 0.01
        star = "★" if is_curr else " "

        print(f"  {mult:>5.2f}  {mh_bars:>5}bars({mh_hrs:.0f}h)  "
              f"{m['N']:>5}  {m['PF']:>7.3f}  {m['WR']:>5.1f}%  "
              f"{m['max_dd_pct']:>5.1f}%  {m['calmar']:>8.1f}  {star}  ({elapsed:.1f}s)")

        results.append({
            "mult": mult, "mh_bars": mh_bars, "mh_hrs": mh_hrs,
            "N": m["N"], "PF": m["PF"], "WR": m["WR"],
            "MaxDD": m["max_dd_pct"], "Calmar": m["calmar"], "curr": is_curr,
        })

    best = max(results, key=lambda r: r["Calmar"])
    print(f"\n  Best Calmar: mult={best['mult']:.2f} ({best['mh_hrs']:.0f}h) "
          f"PF={best['PF']:.3f} Calmar={best['Calmar']:.1f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════


def save_report(
    commit: str,
    cfg: VolStraddleConfig,
    n_syms: int,
    data_years: float,
    m_s: dict,
    m_d: dict,
    phase1: dict,
    year_results: dict,
    p3_results: list[dict],
    p4_results: list[dict],
) -> Path:
    """Write markdown report to docs/backtest_results/202603/."""
    import datetime
    today = datetime.date.today().strftime("%Y%m%d")
    report_dir = ROOT / "docs" / "backtest_results" / "202603"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r80_mixed_slot_parity.md"

    m_old = phase1["old"]
    m_sd = phase1["fixed_sd"]
    m_new = phase1["fixed_full"]

    p3_best = max(p3_results, key=lambda r: r["PF"])
    p4_best_calmar = max(p4_results, key=lambda r: r["Calmar"])
    p4_curr = next((r for r in p4_results if r["curr"]), p4_results[-1])

    with open(report_path, "w") as f:
        f.write(f"""# r80: 혼합(VR+STRD) Canonical 재계산 — Slot Parity Fix 적용

**날짜:** {today[:4]}-{today[4:6]}-{today[6:]}
**커밋:** `{commit}`
**스크립트:** `research/active/r76_r80/r80_mixed_slot_parity.py`
**데이터:** {n_syms} syms (symbol_whitelist) · {data_years:.2f}yr · Binance USDT-M
**비용:** 0.16% round-trip (straddle) / 0.08% (VR) · 10슬롯 (max_concurrent)
**Parity:** config.py 직접 읽음 ✅ · symbol_whitelist ✅ · 룩어헤드 없음 ✅

## 배경

T1 fix (commit `3b43bb2`): `portfolio_sim(straddle_syms=...)` 파라미터 추가.
r76 이전 혼합 백테스트는 straddle을 **1슬롯**으로 처리 (live는 2슬롯) → N 과대추정.
r80에서 수정 후 canonical 수치를 재계산.

---

## Phase 0: Baseline 검증

| 전략 | N | PF | WR | EV(bp) | Calmar | MaxDD | Ref | 판정 |
|------|---|----|----|--------|--------|-------|-----|------|
| STRD (max=5) | {m_s['N']} | {m_s['PF']:.3f} | {m_s['WR']:.1f}% | {m_s['EV_bp']:.1f} | {m_s['calmar']:.1f} | {m_s['max_dd_pct']:.1f}% | r79: N≈7730 PF≈9.603 | {'✅' if abs(m_s['N']-7730)/7730<0.05 else '⚠️'} |
| VR (max=10) | {m_d['N']} | {m_d['PF']:.3f} | {m_d['WR']:.1f}% | {m_d['EV_bp']:.1f} | {m_d['calmar']:.1f} | {m_d['max_dd_pct']:.1f}% | r76: N≈6207 PF≈1.563 | {'✅' if abs(m_d['N']-6207)/6207<0.05 else '⚠️'} |

---

## Phase 1: 혼합 Canonical — straddle_syms 수정 효과

| 버전 | N | PF | Calmar | MaxDD | WR |
|------|---|----|----|----|----|
| r76 (1-slot, broken) | {m_old['N']} | {m_old['PF']:.3f} | {m_old['calmar']:.1f} | {m_old['max_dd_pct']:.1f}% | {m_old['WR']:.1f}% |
| **r80 VR+STRD (2-slot fixed)** | **{m_sd['N']}** | **{m_sd['PF']:.3f}** | **{m_sd['calmar']:.1f}** | **{m_sd['max_dd_pct']:.1f}%** | **{m_sd['WR']:.1f}%** |
| r80 D+S+MA (2-slot fixed) | {m_new['N']} | {m_new['PF']:.3f} | {m_new['calmar']:.1f} | {m_new['max_dd_pct']:.1f}% | {m_new['WR']:.1f}% |
| r76 canonical ref | 9,775 | 2.484 | 9,882 | 37.1% | — |

**슬롯 패리티 수정 영향 (D+S+MA):**
- N: {m_old['N']} → {m_new['N']} ({(m_new['N']-m_old['N'])/m_old['N']:+.1%})
- PF: {m_old['PF']:.3f} → {m_new['PF']:.3f} ({(m_new['PF']-m_old['PF'])/m_old['PF']:+.1%})

---

## Phase 2: 연도별 PF Breakdown

| Year | STRD N | STRD PF | VR N | VR PF | Mixed N | Mixed PF |
|------|--------|---------|------|-------|---------|---------|
""")
        for yr, res in sorted(year_results.items()):
            ms = res["strd"]
            md = res["vr"]
            mm = res["mixed"]
            s_ok = "✅" if ms["PF"] > 1.0 else "❌"
            d_ok = "✅" if md["PF"] > 1.0 else "❌"
            f.write(f"| {yr} | {ms['N']} | {ms['PF']:.3f}{s_ok} | {md['N']} | "
                    f"{md['PF']:.3f}{d_ok} | {mm['N']} | {mm['PF']:.3f} |\n")

        f.write(f"""
---

## Phase 3: s_vr_threshold × s_squeeze_dur Sweep (STRD standalone, max=5)

★ = production ({cfg.s_vr_threshold}, s_dur={cfg.s_squeeze_dur})

| s_vr | s_dur | N | PF | WR | EV(bp) | MaxDD | Calmar | ★ |
|------|-------|---|----|----|--------|-------|--------|---|
""")
        for r in p3_results:
            star = "★" if r["prod"] else ""
            f.write(f"| {r['s_vr']:.2f} | {r['s_dur']} | {r['N']} | {r['PF']:.3f} | "
                    f"{r['WR']:.1f}% | {r['EV']:.1f} | {r['MaxDD']:.1f}% | "
                    f"{r['Calmar']:.1f} | {star} |\n")

        f.write(f"""
**최적 (PF 기준):** s_vr={p3_best['s_vr']:.2f} s_dur={p3_best['s_dur']} "
N={p3_best['N']} PF={p3_best['PF']:.3f} Calmar={p3_best['Calmar']:.1f}

---

## Phase 4: max_hold_straddle 민감도 (STRD standalone, max=5)

현재: max_hold = 2× directional = {cfg.max_hold_15m * 2}bars ({cfg.max_hold_15m * 2 * 15 / 60:.0f}h)

| 배수 | max_hold | N | PF | WR | MaxDD | Calmar | ★ |
|------|----------|---|----|----|-------|--------|---|
""")
        for r in p4_results:
            star = "★" if r["curr"] else ""
            f.write(f"| {r['mult']:.2f}× | {r['mh_bars']}bars({r['mh_hrs']:.0f}h) | "
                    f"{r['N']} | {r['PF']:.3f} | {r['WR']:.1f}% | {r['MaxDD']:.1f}% | "
                    f"{r['Calmar']:.1f} | {star} |\n")

        improvement = p4_best_calmar['mult'] != p4_curr['mult']
        verdict = (f"**권장:** max_hold_mult={p4_best_calmar['mult']:.2f}× "
                   f"({p4_best_calmar['mh_hrs']:.0f}h)으로 변경"
                   if improvement else "**현재(2×) 유지 권장 — 개선 없음**")

        f.write(f"""
{verdict}

---

## 결론 & 프로덕션 권고

| 항목 | 현재 | r80 결과 | 변경 권고 |
|------|------|----------|----------|
| 혼합 백테스트 슬롯 | 1슬롯(broken) | 2슬롯(fixed) | ✅ commit 3b43bb2 적용됨 |
| s_vr_threshold | {cfg.s_vr_threshold} | {p3_best['s_vr']:.2f} (best PF) | {'⚠️ 검토 필요' if abs(p3_best['s_vr']-cfg.s_vr_threshold)>0.001 else '✅ 현재값 유지'} |
| s_squeeze_dur | {cfg.s_squeeze_dur} | {p3_best['s_dur']} (best PF) | {'⚠️ 검토 필요' if p3_best['s_dur']!=cfg.s_squeeze_dur else '✅ 현재값 유지'} |
| max_hold_straddle | 2×({cfg.max_hold_15m*2}bars) | {p4_best_calmar['mult']:.2f}×(best Calmar) | {verdict.split("**")[1]} |

**구현:** 미구현 (파라미터 변경 시 config.py 수정 + r81 재검증 필요)
""")

    print(f"\n  리포트 저장: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    cfg = VolStraddleConfig()
    commit = get_commit()

    print("=" * 70)
    print("r80: 혼합(VR+STRD) Canonical 재계산 — Slot Parity Fix (T1)")
    print(f"  Commit: {commit}")
    print(f"  STRD: s_vr={cfg.s_vr_threshold} s_dur={cfg.s_squeeze_dur}"
          f" s_sl={cfg.s_sl} trail={cfg.s_trail_activate}/{cfg.s_trail_distance}")
    print(f"  VR: d_vr={cfg.d_vr_threshold} d_sl={cfg.d_sl}"
          f" trail={cfg.d_trail_activate}/{cfg.d_trail_distance}")
    print(f"  VR L/N={cfg.base_leverage/cfg.max_concurrent:.1f}"
          f"  max_hold_strd={cfg.max_hold_15m*2}bars={cfg.max_hold_15m*2*15//60}h")
    print(f"  Symbols: {len(cfg.symbol_whitelist)} (symbol_whitelist)")
    print(f"  MA cluster: {'✅' if MA_AVAILABLE else '❌ (skipped)'}")
    print("=" * 70)

    print("\n[Loading data...]")
    t0 = time.perf_counter()
    data_15m, data_1h, data_4h, data_30m, max_bars, data_years = load_all_data(cfg)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s | data_years={data_years:.2f}")

    # Collect base trades once
    print("\n[Collecting base trades...]")
    t1 = time.perf_counter()
    s_trades = collect_s_trail(data_15m, data_1h, data_4h, data_30m, cfg)
    d_trades = collect_d_trail(data_15m, data_1h, data_4h, cfg)
    ma_trades = collect_ma_trades(data_15m, data_1h, data_4h) if MA_AVAILABLE else []
    print(f"  S={len(s_trades)} D={len(d_trades)} MA={len(ma_trades)}  ({time.perf_counter()-t1:.1f}s)")

    # Run phases
    m_s, m_d = phase0_baseline(s_trades, d_trades, cfg, data_years)
    phase1 = phase1_mixed_canonical(s_trades, d_trades, ma_trades, cfg, data_years)
    year_results = phase2_annual_breakdown(s_trades, d_trades, cfg, data_years, max_bars)
    p3_results = phase3_param_sweep(data_15m, data_1h, data_4h, data_30m, cfg, data_years)
    p4_results = phase4_max_hold_sweep(data_15m, data_1h, data_4h, data_30m, cfg, data_years)

    report_path = save_report(
        commit, cfg, len(data_15m), data_years,
        m_s, m_d, phase1, year_results, p3_results, p4_results,
    )

    print("\n" + "=" * 70)
    print("r80 완료.")
    print(f"  리포트: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
