#!/usr/bin/env python3
"""r99: Volume Filters for VR Squeeze & MA Cluster.

r98에서 vol≥1.5× gate가 PF=1.795, DD=61.5%를 보여줌.
기존 VR/MA 전략에 다양한 볼륨 필터를 적용해 품질 개선 가능성 검증.

접근법:
  - 프로덕션 collect_trades()로 기본 트레이드 생성
  - Post-filter: 진입 바의 볼륨 조건으로 필터링
  - 프로덕션 시뮬레이터 수정 불필요

패리티:
  - symbol_whitelist 60개 사용 (exclude_symbols 금지)
  - collect_trades (G_4TF) + portfolio_sim (canonical)
  - config.py 파라미터 직접 읽기 (하드코딩 금지)
  - 1x compound + L/N 별도 표시
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (  # noqa: E402
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim,
    precompute_all_vr,
    G_4TF,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import (  # noqa: E402
    compute_momentum,
)

# ── Config ───────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

# Build cfg dict for collect_trades (must use exact keys)
CFG_DICT = {
    "sp": cfg.vr_short,
    "lp": cfg.vr_long,
    "d_vr": cfg.d_vr_threshold,
    "s_vr": cfg.s_vr_threshold,
    "s_dur": cfg.s_squeeze_dur,
    "d_sl": cfg.d_sl,
    "s_sl": cfg.s_sl,
    "tp": cfg.d_tp,
}

# Portfolio
MAX_SLOTS = cfg.max_concurrent  # 10
VR_LN = cfg.base_leverage / MAX_SLOTS  # 3.0
MA_LN = cfg.ma_cluster_leverage / MAX_SLOTS  # 2.1

# Data
DATA_YEARS = 6.0
CACHE_DIR = ROOT / "data" / "cache_extended"
WHITELIST = set(cfg.symbol_whitelist)  # 60 symbols


# ══════════════════════════════════════════════════════════════════════════════
#  Volume Filter Functions (Numba JIT)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def _vol_expansion(vol_arr: np.ndarray, bar: int, thresh: float) -> bool:
    """Type A: Volume expansion - recent 5 bars vs previous 20 bars."""
    if bar < 25:
        return True
    vol_recent = 0.0
    for k in range(bar - 5, bar):
        vol_recent += vol_arr[k]
    vol_recent /= 5.0
    vol_base = 0.0
    for k in range(bar - 25, bar - 5):
        vol_base += vol_arr[k]
    vol_base /= 20.0
    if vol_base <= 0:
        return True
    return vol_recent >= vol_base * thresh


@numba.jit(nopython=True, cache=True)
def _vol_contraction(vol_arr: np.ndarray, bar: int, thresh: float) -> bool:
    """Type B: Volume contraction - recent 10 < baseline × threshold."""
    if bar < 30:
        return True
    vol_recent = 0.0
    for k in range(bar - 10, bar):
        vol_recent += vol_arr[k]
    vol_recent /= 10.0
    vol_base = 0.0
    for k in range(bar - 30, bar - 10):
        vol_base += vol_arr[k]
    vol_base /= 20.0
    if vol_base <= 0:
        return True
    return vol_recent < vol_base * thresh


@numba.jit(nopython=True, cache=True)
def _vol_spike(vol_arr: np.ndarray, bar: int, thresh: float) -> bool:
    """Type C: Volume spike - entry bar vs 20-bar average."""
    if bar < 20:
        return True
    vol_entry = vol_arr[bar]
    vol_avg = 0.0
    for k in range(bar - 20, bar):
        vol_avg += vol_arr[k]
    vol_avg /= 20.0
    if vol_avg <= 0:
        return True
    return vol_entry >= vol_avg * thresh


@numba.jit(nopython=True, cache=True)
def _vol_trend(vol_arr: np.ndarray, bar: int, _thresh: float) -> bool:
    """Type D: Volume trend - fast MA > slow MA."""
    if bar < 20:
        return True
    vol_fast = 0.0
    for k in range(bar - 5, bar):
        vol_fast += vol_arr[k]
    vol_fast /= 5.0
    vol_slow = 0.0
    for k in range(bar - 20, bar):
        vol_slow += vol_arr[k]
    vol_slow /= 20.0
    return vol_fast > vol_slow


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_data_15m() -> tuple[dict, int]:
    """Load 15m data for all whitelist symbols.

    Returns:
        data_15m: {sym: {close, high, low, volume, _offset}}
        max_bars: Maximum bar count across all symbols.
    """
    data = {}
    max_bars = 0

    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in WHITELIST:
            continue
        d = np.load(f, allow_pickle=False)
        bars = len(d["close"])
        if bars > max_bars:
            max_bars = bars
        data[sym] = {
            "close": d["close"].astype(np.float64),
            "high": d["high"].astype(np.float64),
            "low": d["low"].astype(np.float64),
            "volume": d["volume"].astype(np.float64),
        }

    # Add offset and mom_8 for collect_trades
    mom_lb = 8  # momentum lookback
    for sym in data:
        data[sym]["_offset"] = max_bars - len(data[sym]["close"])
        data[sym]["mom_8"] = compute_momentum(data[sym]["close"], mom_lb)

    return data, max_bars


def build_htf_data(data_15m: dict) -> tuple[dict, dict]:
    """Build 1h and 4h resampled data with mom_8."""
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    # Add mom_8 for collect_trades
    mom_lb = 8
    for d in data_1h.values():
        d["mom_8"] = compute_momentum(d["close"], mom_lb)
    for d in data_4h.values():
        d["mom_8"] = compute_momentum(d["close"], mom_lb)
    return data_1h, data_4h


# ══════════════════════════════════════════════════════════════════════════════
#  Post-Filter Functions
# ══════════════════════════════════════════════════════════════════════════════


def apply_vol_filter(
    trades: list[tuple[str, int, int, float]],
    data_15m: dict,
    vol_filter_type: int,
    vol_threshold: float,
) -> list[tuple[str, int, int, float]]:
    """Apply volume filter to trade list (post-filter).

    Args:
        trades: List of (symbol, entry_bar_15m, exit_bar_15m, pnl).
        data_15m: 15m data with volume.
        vol_filter_type: 0=none, 1=expansion, 2=contraction, 3=spike, 4=trend.
        vol_threshold: Threshold for volume filter.

    Returns:
        Filtered trade list.
    """
    if vol_filter_type == 0:
        return trades

    filtered = []
    for sym, entry, exit_b, pnl in trades:
        if sym not in data_15m:
            continue
        vol_arr = data_15m[sym]["volume"]
        offset = data_15m[sym].get("_offset", 0)
        local_bar = entry - offset

        if local_bar < 0 or local_bar >= len(vol_arr):
            continue

        # Check volume filter
        passes = True
        if vol_filter_type == 1:
            passes = _vol_expansion(vol_arr, local_bar, vol_threshold)
        elif vol_filter_type == 2:
            passes = _vol_contraction(vol_arr, local_bar, vol_threshold)
        elif vol_filter_type == 3:
            passes = _vol_spike(vol_arr, local_bar, vol_threshold)
        elif vol_filter_type == 4:
            passes = _vol_trend(vol_arr, local_bar, vol_threshold)

        if passes:
            filtered.append((sym, entry, exit_b, pnl))

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics Helper
# ══════════════════════════════════════════════════════════════════════════════


def _metrics(trades: list) -> dict:
    """Compute metrics from trade list."""
    if not trades:
        return {
            "N": 0,
            "PF": 0.0,
            "WR": 0.0,
            "EV_bp": 0.0,
            "max_dd_pct": 100.0,
            "ann_ret_pct": 0.0,
            "avg_hold": 0.0,
        }
    pnl = portfolio_sim(trades, MAX_SLOTS)
    m = compute_metrics(pnl, DATA_YEARS)
    # Add avg_hold
    holds = [ext - ent for _, ent, ext, _ in trades]
    m["avg_hold"] = np.mean(holds) / 96 if holds else 0  # 15m bars to days
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  Phase Runners
# ══════════════════════════════════════════════════════════════════════════════

FILTER_NAMES = {0: "none", 1: "expansion", 2: "contraction", 3: "spike", 4: "trend"}


def run_phase0(data_15m: dict, data_1h: dict, data_4h: dict):
    """Phase 0: Baseline verification using canonical collect_trades."""
    print("=" * 70)
    print("Phase 0: 베이스라인 검증 (collect_trades G_4TF)")
    print("=" * 70)

    # VR baseline (D_mom8)
    vr_trades = collect_trades(CFG_DICT, data_15m, data_1h, data_4h, "D_mom8", tf_group=G_4TF)
    vr_m = _metrics(vr_trades)
    print(
        f"  VR Squeeze (D_mom8, G_4TF): N={vr_m['N']} PF={vr_m['PF']:.3f} "
        f"WR={vr_m['WR']:.1f}% EV={vr_m['EV_bp']:+.1f}bp DD={vr_m['max_dd_pct']:.1f}%"
    )

    # S_d8 baseline
    strd_trades = collect_trades(CFG_DICT, data_15m, data_1h, data_4h, "S_d8", tf_group=G_4TF)
    strd_m = _metrics(strd_trades)
    print(
        f"  Straddle (S_d8, G_4TF): N={strd_m['N']} PF={strd_m['PF']:.3f} "
        f"WR={strd_m['WR']:.1f}% EV={strd_m['EV_bp']:+.1f}bp DD={strd_m['max_dd_pct']:.1f}%"
    )

    # Verify VR baseline matches canonical (N=6207 ± 5%)
    expected_n = 6207
    if abs(vr_m["N"] - expected_n) / expected_n > 0.05:
        print(f"  ⚠️ VR N={vr_m['N']} differs from canonical {expected_n} by > 5%")
    else:
        print("  ✅ VR baseline matches canonical (±5%)")

    print("  ✅ Phase 0 완료")
    return {"vr_base": vr_m, "strd_base": strd_m, "vr_trades": vr_trades, "strd_trades": strd_trades}


def run_phase1(vr_trades: list, data_15m: dict, vr_base: dict):
    """Phase 1: VR Squeeze + Volume Filters (post-filter)."""
    print("\n" + "=" * 70)
    print("Phase 1: VR Squeeze + Volume Filters (post-filter)")
    print("=" * 70)

    results = []
    thresholds = {
        1: [1.2, 1.5, 2.0],  # expansion
        2: [0.5, 0.7, 0.8],  # contraction
        3: [1.2, 1.5, 2.0],  # spike
        4: [0.0],  # trend (binary)
    }

    print(
        f"  {'Filter':<20} {'Thresh':>6} {'N':>6} {'N%':>5} {'PF':>6} {'ΔPF':>6} "
        f"{'EV(bp)':>8} {'ΔEV':>6} {'DD%':>6} {'ΔDD':>6}"
    )
    print("  " + "-" * 85)

    # Baseline row
    print(
        f"  {'baseline':<20} {'-':>6} {vr_base['N']:>6} {'100%':>5} "
        f"{vr_base['PF']:>6.3f} {'-':>6} {vr_base['EV_bp']:>+8.1f} {'-':>6} "
        f"{vr_base['max_dd_pct']:>6.1f} {'-':>6}"
    )

    for ftype, thresh_list in thresholds.items():
        fname = FILTER_NAMES[ftype]
        for thresh in thresh_list:
            filtered = apply_vol_filter(vr_trades, data_15m, ftype, thresh)
            m = _metrics(filtered)
            if vr_base["N"] > 0:
                n_pct = m["N"] / vr_base["N"] * 100
                d_pf = m["PF"] - vr_base["PF"]
                d_ev = m["EV_bp"] - vr_base["EV_bp"]
                d_dd = m["max_dd_pct"] - vr_base["max_dd_pct"]
            else:
                n_pct, d_pf, d_ev, d_dd = 0, 0, 0, 0

            print(
                f"  {fname:<20} {thresh:>6.1f} {m['N']:>6} {n_pct:>4.0f}% "
                f"{m['PF']:>6.3f} {d_pf:>+6.3f} {m['EV_bp']:>+8.1f} {d_ev:>+6.1f} "
                f"{m['max_dd_pct']:>6.1f} {d_dd:>+6.1f}"
            )

            results.append(
                {
                    "strategy": "VR",
                    "filter": fname,
                    "threshold": thresh,
                    "ftype": ftype,
                    "m": m,
                    "n_pct": n_pct,
                    "d_pf": d_pf,
                    "d_ev": d_ev,
                    "d_dd": d_dd,
                }
            )

    # Find best (ΔPF ≥ +5% and N% ≥ 70% and ΔDD ≤ 0)
    candidates = [r for r in results if r["d_pf"] >= 0.05 * vr_base["PF"] and r["n_pct"] >= 70 and r["d_dd"] <= 0]
    if candidates:
        best = max(candidates, key=lambda x: x["d_pf"])
        print(
            f"\n  ✅ VR 최적: {best['filter']} t={best['threshold']:.1f} "
            f"→ ΔPF={best['d_pf']:+.3f} ({best['d_pf'] / vr_base['PF'] * 100:+.1f}%), N%={best['n_pct']:.0f}%"
        )
    else:
        print("\n  ❌ VR: 통과 기준 충족하는 필터 없음")
        best = None

    return {"results": results, "best": best}


def run_phase2(strd_trades: list, data_15m: dict, strd_base: dict):
    """Phase 2: Straddle (S_d8) + Volume Filters (post-filter)."""
    print("\n" + "=" * 70)
    print("Phase 2: Straddle (S_d8) + Volume Filters (post-filter)")
    print("=" * 70)

    results = []
    thresholds = {
        1: [1.2, 1.5, 2.0],
        2: [0.5, 0.7, 0.8],
        3: [1.2, 1.5, 2.0],
        4: [0.0],
    }

    print(
        f"  {'Filter':<20} {'Thresh':>6} {'N':>6} {'N%':>5} {'PF':>6} {'ΔPF':>6} "
        f"{'EV(bp)':>8} {'ΔEV':>6} {'DD%':>6} {'ΔDD':>6}"
    )
    print("  " + "-" * 85)

    # Baseline row
    print(
        f"  {'baseline':<20} {'-':>6} {strd_base['N']:>6} {'100%':>5} "
        f"{strd_base['PF']:>6.3f} {'-':>6} {strd_base['EV_bp']:>+8.1f} {'-':>6} "
        f"{strd_base['max_dd_pct']:>6.1f} {'-':>6}"
    )

    for ftype, thresh_list in thresholds.items():
        fname = FILTER_NAMES[ftype]
        for thresh in thresh_list:
            filtered = apply_vol_filter(strd_trades, data_15m, ftype, thresh)
            m = _metrics(filtered)
            if strd_base["N"] > 0:
                n_pct = m["N"] / strd_base["N"] * 100
                d_pf = m["PF"] - strd_base["PF"]
                d_ev = m["EV_bp"] - strd_base["EV_bp"]
                d_dd = m["max_dd_pct"] - strd_base["max_dd_pct"]
            else:
                n_pct, d_pf, d_ev, d_dd = 0, 0, 0, 0

            print(
                f"  {fname:<20} {thresh:>6.1f} {m['N']:>6} {n_pct:>4.0f}% "
                f"{m['PF']:>6.3f} {d_pf:>+6.3f} {m['EV_bp']:>+8.1f} {d_ev:>+6.1f} "
                f"{m['max_dd_pct']:>6.1f} {d_dd:>+6.1f}"
            )

            results.append(
                {
                    "strategy": "STRD",
                    "filter": fname,
                    "threshold": thresh,
                    "ftype": ftype,
                    "m": m,
                    "n_pct": n_pct,
                    "d_pf": d_pf,
                    "d_ev": d_ev,
                    "d_dd": d_dd,
                }
            )

    # Find best
    candidates = [r for r in results if r["d_pf"] >= 0.05 * strd_base["PF"] and r["n_pct"] >= 70 and r["d_dd"] <= 0]
    if candidates:
        best = max(candidates, key=lambda x: x["d_pf"])
        print(
            f"\n  ✅ STRD 최적: {best['filter']} t={best['threshold']:.1f} "
            f"→ ΔPF={best['d_pf']:+.3f} ({best['d_pf'] / strd_base['PF'] * 100:+.1f}%), N%={best['n_pct']:.0f}%"
        )
    else:
        print("\n  ❌ STRD: 통과 기준 충족하는 필터 없음")
        best = None

    return {"results": results, "best": best}


def run_phase3_oos(vr_trades: list, strd_trades: list, data_15m: dict, vr_best: dict | None, strd_best: dict | None):
    """Phase 3: OOS yearly validation for best filters."""
    print("\n" + "=" * 70)
    print("Phase 3: OOS 연도별 검증")
    print("=" * 70)

    DATA_START_YEAR = 2020.17  # Feb 2020

    def yearly_metrics(trades: list, year: int) -> dict:
        """Filter trades to specific year and compute metrics."""
        year_trades = []
        for sym, ent, ext, pnl in trades:
            entry_year = DATA_START_YEAR + ent / (96 * 365.25)
            if int(entry_year) == year:
                year_trades.append((sym, ent, ext, pnl))
        if not year_trades:
            return {"N": 0, "PF": 0.0}
        pnl_arr = portfolio_sim(year_trades, MAX_SLOTS)
        return compute_metrics(pnl_arr, 1.0)

    results = {}

    # VR best filter
    if vr_best:
        print(f"\n  VR: {vr_best['filter']} t={vr_best['threshold']:.1f}")
        filtered = apply_vol_filter(vr_trades, data_15m, vr_best["ftype"], vr_best["threshold"])
        print(f"  {'Year':<6} {'N':>6} {'PF':>6} {'pass?':>6}")
        print("  " + "-" * 28)
        pass_years = 0
        for year in range(2020, 2026):
            ym = yearly_metrics(filtered, year)
            passed = "✅" if ym["N"] > 0 and ym["PF"] >= 1.0 else "❌"
            if ym["N"] > 0 and ym["PF"] >= 1.0:
                pass_years += 1
            print(f"  {year:<6} {ym['N']:>6} {ym['PF']:>6.3f} {passed:>6}")
        results["vr_oos"] = pass_years
        print(f"  VR OOS: {pass_years}/6년")
    else:
        print("\n  VR: 최적 필터 없음, OOS 스킵")
        results["vr_oos"] = None

    # STRD best filter
    if strd_best:
        print(f"\n  STRD: {strd_best['filter']} t={strd_best['threshold']:.1f}")
        filtered = apply_vol_filter(strd_trades, data_15m, strd_best["ftype"], strd_best["threshold"])
        print(f"  {'Year':<6} {'N':>6} {'PF':>6} {'pass?':>6}")
        print("  " + "-" * 28)
        pass_years = 0
        for year in range(2020, 2026):
            ym = yearly_metrics(filtered, year)
            passed = "✅" if ym["N"] > 0 and ym["PF"] >= 1.0 else "❌"
            if ym["N"] > 0 and ym["PF"] >= 1.0:
                pass_years += 1
            print(f"  {year:<6} {ym['N']:>6} {ym['PF']:>6.3f} {passed:>6}")
        results["strd_oos"] = pass_years
        print(f"  STRD OOS: {pass_years}/6년")
    else:
        print("\n  STRD: 최적 필터 없음, OOS 스킵")
        results["strd_oos"] = None

    return results


def run_phase4_combined(
    vr_trades: list,
    strd_trades: list,
    data_15m: dict,
    vr_best: dict | None,
    strd_best: dict | None,
    vr_base: dict,
    strd_base: dict,
):
    """Phase 4: Combined portfolio impact."""
    print("\n" + "=" * 70)
    print("Phase 4: 혼합 포트폴리오 영향")
    print("=" * 70)

    # Baseline: VR + STRD without filters
    combined_base = vr_trades + strd_trades
    m_base = _metrics(combined_base)
    print(f"  VR+STRD (no filter): N={m_base['N']} PF={m_base['PF']:.3f} DD={m_base['max_dd_pct']:.1f}%")

    # With best filters (if any)
    if vr_best:
        vr_filtered = apply_vol_filter(vr_trades, data_15m, vr_best["ftype"], vr_best["threshold"])
    else:
        vr_filtered = vr_trades

    if strd_best:
        strd_filtered = apply_vol_filter(strd_trades, data_15m, strd_best["ftype"], strd_best["threshold"])
    else:
        strd_filtered = strd_trades

    combined_filt = vr_filtered + strd_filtered
    m_filt = _metrics(combined_filt)
    d_pf = m_filt["PF"] - m_base["PF"]
    d_dd = m_filt["max_dd_pct"] - m_base["max_dd_pct"]

    vr_label = f"{vr_best['filter']}@{vr_best['threshold']}" if vr_best else "none"
    strd_label = f"{strd_best['filter']}@{strd_best['threshold']}" if strd_best else "none"
    print(
        f"  VR+STRD (VR:{vr_label}, STRD:{strd_label}): N={m_filt['N']} PF={m_filt['PF']:.3f} "
        f"DD={m_filt['max_dd_pct']:.1f}%"
    )
    print(f"  ΔPF={d_pf:+.3f}, ΔDD={d_dd:+.1f}%p")

    if d_pf > 0 and d_dd <= 0:
        print("  ✅ 혼합 포트폴리오 개선")
    else:
        print("  ❌ 혼합 포트폴리오 미개선 (또는 DD 악화)")


def main():
    """Run all phases."""
    print("=" * 70)
    print("r99: Volume Filters for VR Squeeze & Straddle")
    print("Data: cache_extended/*_15m_2190d_*.npz | 6yr")
    print(f"VR: L/N={VR_LN} | MAX_SLOTS={MAX_SLOTS} | cost=0.08%")
    print("collect_trades() G_4TF | portfolio_sim() canonical")
    print("=" * 70)

    # Load data
    print("\n데이터 로드 중...")
    data_15m, max_bars = load_data_15m()
    print(f"  심볼 수: {len(data_15m)}")
    print(f"  MAX_BARS: {max_bars} (15m) = {max_bars / 96 / 365.25:.2f}년")

    # Build HTF data
    print("  HTF 리샘플링 (1h, 4h)...")
    data_1h, data_4h = build_htf_data(data_15m)

    # Precompute VR
    print("  VR precompute...")
    precompute_all_vr([data_15m, data_1h, data_4h], [(cfg.vr_short, cfg.vr_long)])

    # Phase 0: Baseline
    baselines = run_phase0(data_15m, data_1h, data_4h)

    # Phase 1: VR + Vol filters
    phase1 = run_phase1(baselines["vr_trades"], data_15m, baselines["vr_base"])

    # Phase 2: STRD + Vol filters
    phase2 = run_phase2(baselines["strd_trades"], data_15m, baselines["strd_base"])

    # Phase 3: OOS
    phase3 = run_phase3_oos(
        baselines["vr_trades"], baselines["strd_trades"], data_15m, phase1.get("best"), phase2.get("best")
    )

    # Phase 4: Combined
    run_phase4_combined(
        baselines["vr_trades"],
        baselines["strd_trades"],
        data_15m,
        phase1.get("best"),
        phase2.get("best"),
        baselines["vr_base"],
        baselines["strd_base"],
    )

    # Final summary
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)

    vr_best = phase1.get("best")
    strd_best = phase2.get("best")

    if vr_best:
        print(
            f"  VR 최적: {vr_best['filter']} t={vr_best['threshold']:.1f} "
            f"→ ΔPF={vr_best['d_pf']:+.3f}, OOS={phase3.get('vr_oos', '?')}/6"
        )
    else:
        print("  VR: 통과하는 필터 없음")

    if strd_best:
        print(
            f"  STRD 최적: {strd_best['filter']} t={strd_best['threshold']:.1f} "
            f"→ ΔPF={strd_best['d_pf']:+.3f}, OOS={phase3.get('strd_oos', '?')}/6"
        )
    else:
        print("  STRD: 통과하는 필터 없음")

    if not vr_best and not strd_best:
        print("\n  최종 판정: ❌ FAIL — 볼륨 필터 부가가치 없음")
    elif (vr_best and phase3.get("vr_oos", 0) >= 5) or (strd_best and phase3.get("strd_oos", 0) >= 5):
        print("\n  최종 판정: ✅ PASS — config 추가 권장")
    else:
        print("\n  최종 판정: ⚠️ CONDITIONAL — OOS 미충족, 추가 검증 필요")


if __name__ == "__main__":
    main()
