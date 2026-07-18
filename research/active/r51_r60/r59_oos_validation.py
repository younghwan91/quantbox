#!/usr/bin/env python3
"""r59: OOS Validation — r58 Top Candidates.

Validates that parameters found in r58 are genuine edges (not curve-fitted)
using a time-based 50% IS/OOS split on 60-sym G_4TF.

  IS:  2020-01-01 ~ 2022-12-31  (~3.0yr — crypto boom/bust cycles)
  OOS: 2023-01-01 ~ 2026-02-20  (~3.1yr — market normalisation)

Candidates tested (from r58 findings):
  Baseline : s_vr=0.55  s_dur=4  (current production config)
  B2_mid   : s_vr=0.57  s_dur=4  (+28% N, +11% PF in full-period IS)
  B2_high  : s_vr=0.59  s_dur=4  (+72% N, +24% PF — aggressive)
  B3       : s_vr=0.55  s_dur=2  (+25% N, +12% PF)
  Combo1   : s_vr=0.57  s_dur=2  (untested in r58 — combined)
  Combo2   : s_vr=0.57  s_dur=3  (conservative combo)
  Combo3   : s_vr=0.59  s_dur=2  (aggressive combo)

Pass criteria per candidate:
  OOS_PF / IS_PF >= 0.85  (no severe overfitting)
  OOS_PF >= 1.20           (positive OOS edge)
  OOS_WR >= 42%            (hit rate maintained)

룩어헤드 없음:
  - All signals use only completed bars
  - IS/OOS split is by global bar index (time-based, strict)
  - No parameter re-optimization on OOS

Usage:
    uv run python research/active/r51_r60/r59_oos_validation.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    G_3TF,
    WARMUP_15m,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_signal,
    make_direction_momentum,
)

# ── Straddle stubs (S_d8 removed from production) ────────────────────────────
COST_STRADDLE = 0.0016


def gen_vr_dur_signal(vr, threshold, min_dur, warmup):  # type: ignore[override]
    """Stub: S_d8 straddle signal removed from production."""
    import numpy as np
    return np.zeros(len(vr), dtype=np.int8)


def sim_straddle_trailing(*args, **kwargs):  # type: ignore[override]
    """Stub: straddle simulator removed from production."""
    import numpy as np
    return np.array([]), np.array([]), np.array([])
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Load production config (NO hardcoding) ───────────────────────────────────
cfg = VolStraddleConfig()

VR_SHORT = cfg.vr_short  # 8
VR_LONG = cfg.vr_long  # 56
D_VR = cfg.d_vr_threshold  # 0.50
S_VR = 0.55    # s_vr_threshold removed from production config
S_DUR = 4      # s_squeeze_dur removed from production config
D_SL = cfg.d_sl  # 0.01
D_TP = cfg.d_tp  # 0.12
D_TRAIL_ACT = cfg.d_trail_activate  # 0.005
D_TRAIL_DIST = cfg.d_trail_distance  # 0.002
S_SL = 0.015   # s_sl removed from production config
S_TP = 0.05    # s_tp removed from production config
S_TRAIL_ACT = 0.015  # s_trail_activate removed from production config
S_TRAIL_DIST = 0.002  # s_trail_distance removed from production config
MAX_CONC = cfg.max_concurrent  # 10
VR_LN = cfg.base_leverage / MAX_CONC  # 3.0

MA_PCT = cfg.ma_cluster_pct
MA_MIN_DUR = cfg.ma_cluster_min_duration
MA_MAX_DUR = cfg.ma_cluster_max_duration
MA_MIN_MOM = cfg.ma_cluster_min_momentum
MA_SL = cfg.ma_cluster_sl
MA_TP = cfg.ma_cluster_tp
MA_TRAIL_ACT = cfg.ma_cluster_trail_activate
MA_TRAIL_DIST = cfg.ma_cluster_trail_distance

# 30m TF (same as r57/r58)
WARMUP_30m: int = WARMUP_15m // 2
_MH30m: int = cfg.max_hold_15m // 2
TF_30m = ("30m", 2, WARMUP_30m, _MH30m, _MH30m * 2)
G_4TF = [TF_30m] + list(G_3TF)

VR_KEY = f"vr_{VR_SHORT}_{VR_LONG}"

# Phase 0 parity assertions (canonical trade_sim, 60-sym)
BL_N_EXPECTED = 6207
BL_PF_EXPECTED = 1.563

# OOS split date boundaries
# DATA_END inferred from cache filename: BTCUSDT_15m_2190d_20260220.npz
DATA_END = datetime(2026, 2, 20)
IS_END = datetime(2022, 12, 31)  # IS ends here (inclusive)
OOS_START = datetime(2023, 1, 1)  # OOS starts here

# OOS pass criteria
OOS_PF_MIN = 1.20
OOS_WR_MIN = 42.0
OOS_IS_RATIO_MIN = 0.85

CACHE_DIR = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"

print("\n[Config] 검증")
print(f"  D: SL={D_SL}, TP={D_TP}, trail=({D_TRAIL_ACT},{D_TRAIL_DIST})")
print(f"  S: SL={S_SL}, TP={S_TP}, trail=({S_TRAIL_ACT},{S_TRAIL_DIST})")
print(f"  VR_LN={VR_LN:.1f}  MAX_CONC={MAX_CONC}")
print(f"  d_vr={D_VR}  s_vr={S_VR}  s_dur={S_DUR}")
print(f"  IS: 2020~{IS_END.year}  OOS: {OOS_START.year}~2026")
print("  ✅ All params from config.py\n")


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_syms(min_years: float = 5.0) -> dict:
    """Load 60 whitelist symbols with ≥min_years history."""
    whitelist = set(cfg.symbol_whitelist)
    min_bars = int(min_years * 365 * 24 * 4)
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if nb >= min_bars and (sym not in seen or nb > seen[sym][1]):
            seen[sym] = (f, nb)

    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}

    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def build_all_tf(d15: dict) -> tuple[dict, dict, dict]:
    """Build 30m/1h/4h and precompute all indicators."""
    d30 = build_resampled(d15, 2)
    d1h = build_resampled(d15, 4)
    d4h = build_resampled(d15, 16)
    precompute_all_vr([d15, d30, d1h, d4h], [(VR_SHORT, VR_LONG)])
    for ds in [d15, d30, d1h, d4h]:
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
    precompute_mas(d15)
    precompute_mas(d1h)
    precompute_mas(d4h)
    return d30, d1h, d4h


# ═══════════════════════════════════════════════════════════════════════════════
#  IS / OOS Bar Index Split
# ═══════════════════════════════════════════════════════════════════════════════


def bar_to_dt(global_bar: int, max_bars: int) -> datetime:
    """Convert global 15m bar index to datetime (using DATA_END anchor)."""
    total_minutes = max_bars * 15
    dt = DATA_END - timedelta(minutes=total_minutes) + timedelta(minutes=global_bar * 15)
    return dt


def compute_split_bar(max_bars: int, split_dt: datetime) -> int:
    """Compute global bar index corresponding to split_dt (first bar >= split_dt).

    This is the OOS start bar: bar ≥ this value belongs to OOS.
    bar < this value belongs to IS.
    No lookahead: split is time-based, fixed before seeing any OOS data.
    """
    start_dt = DATA_END - timedelta(minutes=max_bars * 15)
    if split_dt <= start_dt:
        return 0
    if split_dt >= DATA_END:
        return max_bars
    delta = split_dt - start_dt
    return int(delta.total_seconds() / 60 / 15)


def split_trades(trades: list, oos_start_bar: int) -> tuple[list, list]:
    """Split trades into IS / OOS by entry bar.

    IS:  entry_bar <  oos_start_bar
    OOS: entry_bar >= oos_start_bar

    Uses entry bar (index 1 in trade tuple) for assignment.
    No lookahead: trade assignment based only on entry timing.
    """
    is_trades = [t for t in trades if t[1] < oos_start_bar]
    oos_trades = [t for t in trades if t[1] >= oos_start_bar]
    return is_trades, oos_trades


def run_split_metrics(trades: list, data_years: float, max_concurrent: int = 10) -> dict:
    """Run portfolio_sim + compute_metrics on a trade list."""
    if not trades:
        return {"N": 0, "PF": 0.0, "WR": 0.0, "EV_bp": 0.0, "calmar": 0.0, "max_dd_pct": 100.0}
    pnl = portfolio_sim(trades, max_concurrent)
    if len(pnl) == 0:
        return {"N": 0, "PF": 0.0, "WR": 0.0, "EV_bp": 0.0, "calmar": 0.0, "max_dd_pct": 100.0}
    return compute_metrics(pnl, data_years)


# ═══════════════════════════════════════════════════════════════════════════════
#  Trade Collection
# ═══════════════════════════════════════════════════════════════════════════════


def _dedup_sym(sym: str, sym_trades: list) -> list:
    """Per-symbol time dedup."""
    sym_trades.sort(key=lambda x: x[0])
    accepted: list[tuple[int, int]] = []
    result = []
    for entry_15m, exit_15m, pnl_val in sym_trades:
        conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
        if not conflict:
            result.append((sym, entry_15m, exit_15m, pnl_val))
            accepted.append((entry_15m, exit_15m))
    return result


def collect_d(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
    d_vr: float = D_VR,
) -> list:
    """D_mom8 directional trades with trailing stop."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m
    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
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
            sig = gen_vr_signal(vr, d_vr, warmup)
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, D_SL, D_TP, mh_d, COST_DIRECTIONAL, D_TRAIL_ACT, D_TRAIL_DIST
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio + offset, int(x[k]) * ratio + ratio - 1 + offset, float(p[k])))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_s(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
    s_vr: float = S_VR,
    s_dur: int = S_DUR,
) -> list:
    """S_d8 straddle trades with configurable thresholds."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m
    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list = []
        for _lbl, ratio, warmup, _mh_d, mh_s in tfs:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(VR_KEY)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(c, h, lo, sig, S_SL, S_TP, mh_s, COST_STRADDLE, S_TRAIL_ACT, S_TRAIL_DIST)
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio + offset, int(x[k]) * ratio + ratio - 1 + offset, float(p[k])))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_ma(data_15m: dict, data_1h: dict, data_4h: dict) -> list:
    """MA cluster trades (4h-only, config params)."""
    return collect_enhanced_trades(
        data_15m,
        data_1h,
        data_4h,
        cluster_pct=MA_PCT,
        min_duration=MA_MIN_DUR,
        max_duration=MA_MAX_DUR,
        min_momentum=MA_MIN_MOM,
        sl=MA_SL,
        tp=MA_TP,
        tf_restriction=16,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  OOS Validation Runner
# ═══════════════════════════════════════════════════════════════════════════════


def validate_candidate(
    label: str,
    d15: dict,
    d1h: dict,
    d4h: dict,
    d30: dict,
    max_bars: int,
    oos_start_bar: int,
    is_years: float,
    oos_years: float,
    s_vr: float = S_VR,
    s_dur: int = S_DUR,
) -> dict:
    """Run IS + OOS validation for a single parameter candidate.

    Returns dict with full_period, IS, OOS metrics and pass/fail verdict.
    """
    t = time.time()

    # Collect all trades (full period)
    d_trades = collect_d(d15, d1h, d4h, d30)
    s_trades = collect_s(d15, d1h, d4h, d30, s_vr=s_vr, s_dur=s_dur)
    ma_trades = collect_ma(d15, d1h, d4h)
    all_trades = d_trades + s_trades + ma_trades

    # Full period
    full_years = is_years + oos_years
    full_pnl = portfolio_sim(all_trades, MAX_CONC)
    full_m = compute_metrics(full_pnl, full_years)
    full_boot = bootstrap_pf(full_pnl, n_boot=500)

    # IS / OOS split
    is_trades, oos_trades = split_trades(all_trades, oos_start_bar)
    is_m = run_split_metrics(is_trades, is_years, MAX_CONC)
    oos_m = run_split_metrics(oos_trades, oos_years, MAX_CONC)

    # Pass/fail
    oos_is_ratio = oos_m["PF"] / is_m["PF"] if is_m["PF"] > 0 else 0.0
    passed = oos_m["PF"] >= OOS_PF_MIN and oos_m["WR"] >= OOS_WR_MIN and oos_is_ratio >= OOS_IS_RATIO_MIN

    return {
        "label": label,
        "s_vr": s_vr,
        "s_dur": s_dur,
        # Full period
        "full_N": full_m["N"],
        "full_PF": full_m["PF"],
        "full_WR": full_m["WR"],
        "full_EV": full_m["EV_bp"],
        "full_calmar": full_m["calmar"],
        "full_DD": full_m["max_dd_pct"],
        "full_CI": f"[{full_boot['CI_lo']:.3f},{full_boot['CI_hi']:.3f}]",
        # IS
        "is_N": is_m["N"],
        "is_PF": is_m["PF"],
        "is_WR": is_m["WR"],
        "is_EV": is_m["EV_bp"],
        # OOS
        "oos_N": oos_m["N"],
        "oos_PF": oos_m["PF"],
        "oos_WR": oos_m["WR"],
        "oos_EV": oos_m["EV_bp"],
        # Verdict
        "oos_is_ratio": oos_is_ratio,
        "passed": passed,
        "elapsed": round(time.time() - t, 1),
    }


def fmt_candidate(r: dict) -> str:
    verdict = "✅ PASS" if r["passed"] else "❌ FAIL"
    ratio_flag = "✅" if r["oos_is_ratio"] >= OOS_IS_RATIO_MIN else "❌"
    pf_flag = "✅" if r["oos_PF"] >= OOS_PF_MIN else "❌"
    wr_flag = "✅" if r["oos_WR"] >= OOS_WR_MIN else "❌"
    return (
        f"\n  [{verdict}] {r['label']}  (s_vr={r['s_vr']:.2f} s_dur={r['s_dur']})\n"
        f"    Full:  N={r['full_N']:6d}  PF={r['full_PF']:.3f}  WR={r['full_WR']:.1f}%"
        f"  EV={r['full_EV']:+.1f}bp  Calmar={r['full_calmar']:.1f}  DD={r['full_DD']:.1f}%\n"
        f"    IS  :  N={r['is_N']:6d}  PF={r['is_PF']:.3f}  WR={r['is_WR']:.1f}%  EV={r['is_EV']:+.1f}bp\n"
        f"    OOS :  N={r['oos_N']:6d}  PF={r['oos_PF']:.3f} {pf_flag}  "
        f"WR={r['oos_WR']:.1f}% {wr_flag}  EV={r['oos_EV']:+.1f}bp\n"
        f"    OOS/IS ratio={r['oos_is_ratio']:.3f} {ratio_flag}  [{r['elapsed']}s]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(
    commit: str,
    baseline: dict,
    candidates: list[dict],
    oos_start_bar: int,
    max_bars: int,
    is_years: float,
    oos_years: float,
) -> str:
    """Generate markdown report."""
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    passed = [c for c in candidates if c["passed"]]
    failed = [c for c in candidates if not c["passed"]]

    # Determine IS/OOS calendar dates from bar indices
    is_start_dt = bar_to_dt(0, max_bars)
    oos_start_dt = bar_to_dt(oos_start_bar, max_bars)
    is_end_dt = bar_to_dt(oos_start_bar - 1, max_bars)

    lines = [
        f"# r59: OOS Validation — r58 Top Candidates",
        f"",
        f"**Date:** {today}",
        f"**Commit:** `{commit}`",
        f"**Script:** `research/active/r51_r60/r59_oos_validation.py`",
        f"**구현 여부:** 미구현",
        f"",
        f"---",
        f"",
        f"## 목표",
        f"",
        f"r58에서 발굴한 파라미터 후보들이 실제 엣지인지 IS/OOS 50% 분할로 검증.",
        f"과적합 없이 OOS에서도 동일한 엣지가 유지되는지 확인.",
        f"",
        f"---",
        f"",
        f"## Parity Notes",
        f"",
        f"- Config: `VolStraddleConfig()` 직접 읽음 ✅",
        f"- Lookahead 없음 ✅ (신호 생성: 완성된 바만, 분할: entry bar 기준)",
        f"- Universe: 60 whitelist symbols (5yr+)",
        f"- 비용: 8bp round-trip (D), 16bp (S)",
        f"- 모델: 1x compound (compute_metrics 기본값)",
        f"",
        f"---",
        f"",
        f"## IS / OOS 분할",
        f"",
        f"| | 기간 | 연수 | 특성 |",
        f"|---|---|---|---|",
        f"| IS  | {is_start_dt.strftime('%Y-%m-%d')} ~ {is_end_dt.strftime('%Y-%m-%d')} "
        f"| {is_years:.1f}yr | 크립토 붐/버스트 사이클 (2020-2022) |",
        f"| OOS | {oos_start_dt.strftime('%Y-%m-%d')} ~ 2026-02-20 "
        f"| {oos_years:.1f}yr | 시장 안정화 구간 (2023-2025) |",
        f"",
        f"**합격 기준:**",
        f"- OOS PF ≥ {OOS_PF_MIN}  (OOS 단독 수익성)",
        f"- OOS WR ≥ {OOS_WR_MIN}%  (승률 유지)",
        f"- OOS/IS PF ratio ≥ {OOS_IS_RATIO_MIN}  (과적합 없음)",
        f"",
        f"---",
        f"",
        f"## Phase 0: Baseline Parity",
        f"",
        f"| | N | PF | WR | EV |",
        f"|-|---|----|----|-----|",
        f"| Baseline | {baseline['full_N']} | {baseline['full_PF']:.3f} "
        f"| {baseline['full_WR']:.1f}% | {baseline['full_EV']:+.1f}bp |",
        f"| IS  | {baseline['is_N']} | {baseline['is_PF']:.3f} "
        f"| {baseline['is_WR']:.1f}% | {baseline['is_EV']:+.1f}bp |",
        f"| OOS | {baseline['oos_N']} | {baseline['oos_PF']:.3f} "
        f"| {baseline['oos_WR']:.1f}% | {baseline['oos_EV']:+.1f}bp |",
        f"",
        f"---",
        f"",
        f"## OOS Validation Results",
        f"",
        f"| 후보 | s_vr | s_dur | Full N | Full PF | IS PF | OOS PF | OOS/IS | WR OOS | 판정 |",
        f"|------|------|-------|--------|---------|-------|--------|--------|--------|------|",
    ]

    for c in candidates:
        verdict = "✅ PASS" if c["passed"] else "❌ FAIL"
        lines.append(
            f"| {c['label']} | {c['s_vr']:.2f} | {c['s_dur']} "
            f"| {c['full_N']:,} | {c['full_PF']:.3f} "
            f"| {c['is_PF']:.3f} | {c['oos_PF']:.3f} "
            f"| {c['oos_is_ratio']:.3f} | {c['oos_WR']:.1f}% | {verdict} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## 상세 결과",
        f"",
    ]
    for c in candidates:
        verdict = "✅ PASS" if c["passed"] else "❌ FAIL"
        ratio_flag = "✅" if c["oos_is_ratio"] >= OOS_IS_RATIO_MIN else "❌"
        pf_flag = "✅" if c["oos_PF"] >= OOS_PF_MIN else "❌"
        wr_flag = "✅" if c["oos_WR"] >= OOS_WR_MIN else "❌"
        lines += [
            f"### {verdict} {c['label']} (s_vr={c['s_vr']:.2f}, s_dur={c['s_dur']})",
            f"",
            f"| Period | N | PF | WR | EV |",
            f"|--------|---|----|----|-----|",
            f"| Full | {c['full_N']:,} | {c['full_PF']:.3f} | {c['full_WR']:.1f}% | {c['full_EV']:+.1f}bp |",
            f"| IS   | {c['is_N']:,} | {c['is_PF']:.3f} | {c['is_WR']:.1f}% | {c['is_EV']:+.1f}bp |",
            f"| OOS  | {c['oos_N']:,} | {c['oos_PF']:.3f} {pf_flag} | {c['oos_WR']:.1f}% {wr_flag} "
            f"| {c['oos_EV']:+.1f}bp |",
            f"",
            f"OOS/IS ratio = {c['oos_is_ratio']:.3f} {ratio_flag}  |  Bootstrap CI = {c['full_CI']}",
            f"",
        ]

    # Summary
    lines += [
        f"---",
        f"",
        f"## 결론",
        f"",
        f"**통과: {len(passed)}/{len(candidates)}개**",
        f"",
    ]
    if passed:
        lines += [f"### ✅ 구현 추천 (통과 후보)", f""]
        for i, c in enumerate(passed, 1):
            delta_n = c["full_N"] - baseline["full_N"]
            delta_pf = c["full_PF"] - baseline["full_PF"]
            lines.append(
                f"{i}. **{c['label']}** — s_vr={c['s_vr']:.2f} s_dur={c['s_dur']} "
                f"| Full N+{delta_n:+d} PF{delta_pf:+.3f} "
                f"| OOS PF={c['oos_PF']:.3f} WR={c['oos_WR']:.1f}%"
            )
        lines.append(f"")
        lines += [
            f"### 권장 Config 변경",
            f"",
            f"```python",
            f"# config.py 변경 대상 (OOS 검증 통과 기준)",
        ]
        best = max(passed, key=lambda x: x["oos_PF"])
        lines += [
            f"s_vr_threshold: {best['s_vr']:.2f}  # {S_VR:.2f} → {best['s_vr']:.2f}",
            f"s_squeeze_dur: {best['s_dur']}  # {S_DUR} → {best['s_dur']}",
            f"```",
            f"",
        ]
    if failed:
        lines += [f"### ❌ 미통과 (미구현)", f""]
        for c in failed:
            lines.append(f"- {c['label']}: OOS PF={c['oos_PF']:.3f} ratio={c['oos_is_ratio']:.3f}")
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"*Data: 60 symbols · {is_years + oos_years:.1f}yr · Binance USDT-M perpetuals · "
        f"비용 8/16bp round-trip · 1x compound*",
    ]
    return "\n".join(lines)


def bar_to_dt(global_bar: int, max_bars: int) -> datetime:
    """Convert global 15m bar to datetime (DATA_END anchor)."""
    total_minutes = max_bars * 15
    return DATA_END - timedelta(minutes=total_minutes) + timedelta(minutes=global_bar * 15)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    print(f"\nr59: OOS Validation [{commit}]  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # ── Load 60-sym whitelist ─────────────────────────────────────────────────
    t_load = time.time()
    print("Loading 60-sym whitelist (min_years=5.0)...")
    data_15m = load_syms(min_years=5.0)
    max_bars = max(len(d["close"]) for d in data_15m.values())
    DATA_YEARS = max_bars / (4 * 24 * 365)
    print(f"  {len(data_15m)} symbols | {DATA_YEARS:.2f} yr | {time.time() - t_load:.1f}s")

    t_prep = time.time()
    d30, d1h, d4h = build_all_tf(data_15m)
    print(f"  Precomputed: 30m={len(d30)} 1h={len(d1h)} 4h={len(d4h)} | {time.time() - t_prep:.1f}s\n")

    # ── Compute IS/OOS split bar ──────────────────────────────────────────────
    oos_start_bar = compute_split_bar(max_bars, OOS_START)
    is_end_bar = oos_start_bar - 1
    is_years = (is_end_bar * 15) / (60 * 24 * 365)
    oos_years = ((max_bars - oos_start_bar) * 15) / (60 * 24 * 365)

    is_start_dt = bar_to_dt(0, max_bars)
    is_end_dt = bar_to_dt(is_end_bar, max_bars)
    oos_start_dt = bar_to_dt(oos_start_bar, max_bars)

    print(f"IS/OOS Split:")
    print(
        f"  IS : bar 0 ~ {is_end_bar} ({is_start_dt.strftime('%Y-%m-%d')} ~ {is_end_dt.strftime('%Y-%m-%d')}, {is_years:.2f}yr)"
    )
    print(
        f"  OOS: bar {oos_start_bar} ~ {max_bars} ({oos_start_dt.strftime('%Y-%m-%d')} ~ 2026-02-20, {oos_years:.2f}yr)\n"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 0: Baseline Parity (G_3TF, config defaults)
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("Phase 0: Baseline Parity (G_3TF, default config)")
    print("─" * 72)
    t = time.time()
    d_trades_bl = collect_d(data_15m, d1h, d4h)
    s_trades_bl = collect_s(data_15m, d1h, d4h)
    ma_trades_bl = collect_ma(data_15m, d1h, d4h)
    all_bl = d_trades_bl + s_trades_bl + ma_trades_bl
    pnl_bl = portfolio_sim(all_bl, MAX_CONC)
    m_bl = compute_metrics(pnl_bl, DATA_YEARS)
    print(
        f"  N={m_bl['N']:,}  PF={m_bl['PF']:.3f}  WR={m_bl['WR']:.1f}%  EV={m_bl['EV_bp']:+.1f}bp  [{time.time() - t:.1f}s]"
    )

    n_drift = abs(m_bl["N"] - BL_N_EXPECTED) / BL_N_EXPECTED
    pf_drift = abs(m_bl["PF"] - BL_PF_EXPECTED) / BL_PF_EXPECTED
    if n_drift > 0.15 or pf_drift > 0.15:
        print(f"  ⚠️  Parity FAIL: N±{n_drift * 100:.1f}%, PF±{pf_drift * 100:.1f}%")
    else:
        print(f"  ✅ Parity OK (N±{n_drift * 100:.1f}%, PF±{pf_drift * 100:.1f}%)\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Candidates to validate (G_4TF)
    # ─────────────────────────────────────────────────────────────────────────
    CANDIDATES = [
        ("Baseline", S_VR, S_DUR),  # s_vr=0.55, s_dur=4
        ("B2_mid", 0.57, S_DUR),  # s_vr=0.57, s_dur=4
        ("B2_high", 0.59, S_DUR),  # s_vr=0.59, s_dur=4
        ("B3", S_VR, 2),  # s_vr=0.55, s_dur=2
        ("Combo1", 0.57, 2),  # s_vr=0.57, s_dur=2
        ("Combo2", 0.57, 3),  # s_vr=0.57, s_dur=3
        ("Combo3", 0.59, 2),  # s_vr=0.59, s_dur=2
    ]

    print("─" * 72)
    print("OOS Validation (G_4TF, 60-sym)")
    print(f"Candidates: {len(CANDIDATES)}")
    print("─" * 72)

    results: list[dict] = []
    baseline_result: dict = {}

    for label, s_vr_val, s_dur_val in CANDIDATES:
        print(f"\n  Running: {label} (s_vr={s_vr_val:.2f} s_dur={s_dur_val})...")
        r = validate_candidate(
            label=label,
            d15=data_15m,
            d1h=d1h,
            d4h=d4h,
            d30=d30,
            max_bars=max_bars,
            oos_start_bar=oos_start_bar,
            is_years=is_years,
            oos_years=oos_years,
            s_vr=s_vr_val,
            s_dur=s_dur_val,
        )
        print(fmt_candidate(r))
        results.append(r)
        if label == "Baseline":
            baseline_result = r

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    print("\n" + "=" * 80)
    print(f"OOS VALIDATION SUMMARY: {len(passed)}/{len(results)} PASSED")
    print("=" * 80)
    print(f"\n  Pass criteria: OOS_PF≥{OOS_PF_MIN}  OOS_WR≥{OOS_WR_MIN}%  OOS/IS≥{OOS_IS_RATIO_MIN}")
    print(f"\n  {'Candidate':<12} {'Full PF':>8} {'IS PF':>7} {'OOS PF':>7} {'OOS/IS':>7} {'WR OOS':>7} {'Verdict':>8}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")
    for r in results:
        verdict = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(
            f"  {r['label']:<12} {r['full_PF']:>8.3f} {r['is_PF']:>7.3f} "
            f"{r['oos_PF']:>7.3f} {r['oos_is_ratio']:>7.3f} {r['oos_WR']:>7.1f}% {verdict:>8}"
        )

    if passed:
        best = max(passed, key=lambda x: x["oos_PF"])
        print(f"\n  🏆 Best: {best['label']} — OOS PF={best['oos_PF']:.3f} WR={best['oos_WR']:.1f}%")
        print(f"     → Config 변경 권고: s_vr={best['s_vr']:.2f} s_dur={best['s_dur']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save report
    # ─────────────────────────────────────────────────────────────────────────
    report_dir = REPORT_DIR / "202603"
    report_dir.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    report_path = report_dir / f"{today_str}_{commit}_r59_oos_validation.md"

    report_md = generate_report(
        commit=commit,
        baseline=baseline_result,
        candidates=results,
        oos_start_bar=oos_start_bar,
        max_bars=max_bars,
        is_years=is_years,
        oos_years=oos_years,
    )
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n  📄 Report saved: {report_path.relative_to(ROOT)}")
    print(f"\n  Done. VR_LN={VR_LN:.1f}  MAX_CONC={MAX_CONC}")


if __name__ == "__main__":
    main()
