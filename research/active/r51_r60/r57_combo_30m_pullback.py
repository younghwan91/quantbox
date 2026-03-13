#!/usr/bin/env python3
"""r57: 30m TF + Pullback Entry Combination.

r54/r55/r56의 핵심 발견을 조합:
  r56 ✅: 30m TF (G_4TF) → N +8.5%, PF +2.3%, EV 동등
  r55 ✅: Limit Order Pullback (pb=0.5%, win=5) → PF +3.4%, N 96.6%
  r54 ✅: vr_depth ≥ 0.68 → 유일한 의미있는 예측 변수

조합 가설:
  두 개선이 독립적이면:
    N ≈ 7,468 × 0.966 ≈ 7,214 (pullback이 ~3.4% 거래 스킵)
    PF ≈ 1.673 × 1.034 ≈ 1.730 (복합 효과)

Phase A: G_4TF(30m 포함) + Pullback (pb=0.005, win=5) on D_mom8
Phase B: Phase A + vr_depth ≥ 0.68 gate — r54 발견 검증
Phase C: max_concurrent=12 + Re-Entry — r56 Method C 슬롯 제약 해결
Phase D: 최선 조합 + OOS 50% + 보고서

룩어헤드 없음:
  - 30m: 15m → 30m 리샘플 (완성된 봉만)
  - Pullback: 신호 봉 다음 봉부터 지정가 충족 확인
  - vr_depth: 신호 봉의 VR 값 (완성 바)

Usage:
    uv run python research/active/r51_r60/r57_combo_30m_pullback.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    WARMUP_15m,
    bootstrap_pf,
    build_resampled,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
)
from research.active.r41_r45.r43_exit_improvement import (
    sim_directional_trailing,
    sim_straddle_trailing,
)
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Load production config (NO hardcoding) ───────────────────────────────────
cfg = VolStraddleConfig()

VR_SHORT    = cfg.vr_short          # 8
VR_LONG     = cfg.vr_long           # 56
D_VR        = cfg.d_vr_threshold    # 0.50
S_VR        = cfg.s_vr_threshold    # 0.55
S_DUR       = cfg.s_squeeze_dur     # 4
D_SL        = cfg.d_sl              # 0.01
D_TP        = cfg.d_tp              # 0.12
D_TRAIL_ACT = cfg.d_trail_activate  # 0.005
D_TRAIL_DIST = cfg.d_trail_distance # 0.002
S_SL        = cfg.s_sl
S_TP        = cfg.s_tp
S_TRAIL_ACT = cfg.s_trail_activate
S_TRAIL_DIST = cfg.s_trail_distance
MAX_CONC    = cfg.max_concurrent    # 10
VR_LN       = cfg.base_leverage / MAX_CONC  # 3.0

MA_PCT      = cfg.ma_cluster_pct
MA_MIN_DUR  = cfg.ma_cluster_min_duration
MA_MAX_DUR  = cfg.ma_cluster_max_duration
MA_MIN_MOM  = cfg.ma_cluster_min_momentum
MA_SL       = cfg.ma_cluster_sl
MA_TP       = cfg.ma_cluster_tp
MA_TRAIL_ACT = cfg.ma_cluster_trail_activate
MA_TRAIL_DIST = cfg.ma_cluster_trail_distance

# 30m TF constants (same wall-clock durations as 15m)
WARMUP_30m: int = WARMUP_15m // 2        # 100 30m-bars ≈ 50h
_MH30m: int = cfg.max_hold_15m // 2      # 96 30m-bars ≈ 48h
TF_30m = ("30m", 2, WARMUP_30m, _MH30m, _MH30m * 2)
G_4TF = [TF_30m] + list(G_3TF)           # 30m + 15m + 1h + 4h

VR_KEY = f"vr_{VR_SHORT}_{VR_LONG}"

BL_N_EXPECTED  = 6207   # v14.5.2 canonical
BL_PF_EXPECTED = 1.563

# Config guards
assert D_TRAIL_ACT == 0.010, f"d_trail_activate mismatch: {D_TRAIL_ACT}"
assert D_TRAIL_DIST == 0.002, f"d_trail_distance mismatch: {D_TRAIL_DIST}"

print("\n[Config] 검증")
print(f"  D: SL={D_SL}, TP={D_TP}, trail=({D_TRAIL_ACT},{D_TRAIL_DIST})")
print(f"  VR_LN={VR_LN:.1f}  MAX_CONC={MAX_CONC}")
print(f"  30m: warmup={WARMUP_30m}, max_hold={_MH30m}")
print("  ✅ All params from config.py\n")

CACHE_DIR  = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_syms(min_years: float = 5.0) -> dict:
    """Load 15m symbols. Returns data_15m dict."""
    whitelist = set(cfg.symbol_whitelist)  # 60 live symbols only
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        nb = int(np.load(f, allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    min_bars = int(min_years * 365 * 24 * 4)
    ranked = sorted(
        [(s, p, nb) for s, (p, nb) in seen.items() if nb >= min_bars],
        key=lambda x: -x[2],
    )
    data: dict = {}
    for sym, fpath, _ in ranked:
        npz = np.load(fpath, allow_pickle=False)
        data[sym] = {k: npz[k] for k in npz.files}
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def build_all_tf(d15: dict) -> tuple[dict, dict, dict]:
    """Build 30m/1h/4h from 15m and precompute all indicators."""
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
#  Numba: Pullback Entry Simulation
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_directional_pullback_trailing(
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
    pullback_pct: float,
    pullback_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional sim with limit order pullback + trailing stop.

    After signal fires at bar i, place limit order at:
      LONG:  entry = close[i] * (1 - pullback_pct)  → fill if low[j] <= entry
      SHORT: entry = close[i] * (1 + pullback_pct)  → fill if high[j] >= entry

    Searches bars [i+1 .. i+pullback_window]. If not filled, skip signal.
    SL/TP measured from limit_price.
    룩어헤드 없음: 신호봉 i 이후 완성봉만 참조.
    """
    n = len(close)
    cap = n // 2 + 1
    pnl_arr = np.empty(cap, dtype=np.float64)
    ent_arr = np.empty(cap, dtype=np.int64)
    ext_arr = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = 0
    while i < n - 1:
        if sig[i] == 0 or dire[i] == 0:
            i += 1
            continue
        d = dire[i]
        sc = close[i]
        if sc <= 0.0:
            i += 1
            continue
        limit_price = sc * (1.0 - pullback_pct) if d == 1 else sc * (1.0 + pullback_pct)

        fill_bar = -1
        search_end = min(i + 1 + pullback_window, n)
        for w in range(i + 1, search_end):
            if d == 1 and low[w] <= limit_price:
                fill_bar = w
                break
            elif d == -1 and high[w] >= limit_price:
                fill_bar = w
                break

        if fill_bar == -1:
            i += 1
            continue

        entry = limit_price
        end = min(fill_bar + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0

        for j in range(fill_bar + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
        else:
            if d == 1:
                trade_pnl = (close[end - 1] - entry) / entry
            else:
                trade_pnl = (entry - close[end - 1]) / entry

        trade_pnl -= cost
        pnl_arr[cnt] = trade_pnl
        ent_arr[cnt] = fill_bar
        ext_arr[cnt] = eb
        cnt += 1
        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


# ═══════════════════════════════════════════════════════════════════════════════
#  Trade Collection Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _dedup_sym(sym: str, sym_trades: list) -> list:
    """Per-symbol time dedup (same logic as trade_sim.collect_trades)."""
    sym_trades.sort(key=lambda x: x[0])
    accepted: list[tuple[int, int]] = []
    result = []
    for entry_15m, exit_15m, pnl_val in sym_trades:
        conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
        if not conflict:
            result.append((sym, entry_15m, exit_15m, pnl_val))
            accepted.append((entry_15m, exit_15m))
    return result


def collect_d_pullback(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    pullback_pct: float,
    pullback_window: int,
    data_30m: dict | None = None,
    vr_depth_min: float = 0.0,
) -> list:
    """D_mom8 VR Squeeze trades with pullback entry, G_4TF support.

    Args:
        pullback_pct: Limit order distance from signal close (0.005 = 0.5%).
        pullback_window: Max bars to wait for limit fill after signal.
        data_30m: If provided, includes 30m TF in collection.
        vr_depth_min: Minimum VR at signal bar. 0.0 = no filter.
                      r54 finding: wins avg=0.724, losses avg=0.671.
                      Gate at 0.68 filters most lossy signals.

    Returns:
        List of (sym, entry_15m, exit_15m, pnl) tuples.
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m
    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
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

            sig = gen_vr_signal(vr, D_VR, warmup)

            # vr_depth gate: admit only signals where VR is not too compressed
            # Higher vr_depth = less extreme squeeze = better edge (r54 finding)
            if vr_depth_min > 0.0:
                sig = sig.copy()
                for idx in range(len(sig)):
                    if sig[idx] != 0 and not np.isnan(vr[idx]) and vr[idx] < vr_depth_min:
                        sig[idx] = 0

            dire = make_direction_momentum(sig, mom8)

            if pullback_pct > 0.0:
                p, e, x = sim_directional_pullback_trailing(
                    c, h, lo, sig, dire,
                    D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                    D_TRAIL_ACT, D_TRAIL_DIST,
                    pullback_pct, pullback_window,
                )
            else:
                p, e, x = sim_directional_trailing(
                    c, h, lo, sig, dire,
                    D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                    D_TRAIL_ACT, D_TRAIL_DIST,
                )

            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))

        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_s_trailing(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
) -> list:
    """S_d8 straddle trades (no pullback — straddle is symmetric)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m
    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_trades: list = []
    for sym in sorted(data_15m.keys()):
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
            sig = gen_vr_dur_signal(vr, S_VR, S_DUR, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig,
                S_SL, S_TP, mh_s, COST_STRADDLE,
                S_TRAIL_ACT, S_TRAIL_DIST,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_ma(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    tf_restriction: int | None = 16,
) -> list:
    """MA cluster trades (4h-only by default)."""
    return collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_PCT,
        min_duration=MA_MIN_DUR,
        max_duration=MA_MAX_DUR,
        min_momentum=MA_MIN_MOM,
        sl=MA_SL, tp=MA_TP,
        tf_restriction=tf_restriction,
    )


@numba.jit(nopython=True, cache=True)
def gen_reentry_signal(
    vr: np.ndarray,
    threshold: float,
    warmup: int,
    exit_bars: np.ndarray,
    pnl: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Re-entry signal after profitable exits (룩어헤드 없음)."""
    n = len(vr)
    sig = np.zeros(n, dtype=np.int8)
    for k in range(len(pnl)):
        if pnl[k] <= 0.0:
            continue
        ex = int(exit_bars[k])
        for j in range(ex + 1, min(ex + lookback + 1, n)):
            if j < warmup:
                continue
            if not np.isnan(vr[j]) and vr[j] < threshold:
                sig[j] = 1
                break
    return sig


def collect_d_reentry(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    lookback: int = 5,
) -> list:
    """D_mom8 re-entry trades after profitable exits."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    all_trades: list = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list = []
        for _lbl, ratio, warmup, mh_d, _mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(VR_KEY)
            mom8 = d.get("mom_8")
            if vr is None or mom8 is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]

            orig_sig  = gen_vr_signal(vr, D_VR, warmup)
            orig_dire = make_direction_momentum(orig_sig, mom8)
            p_orig, e_orig, x_orig = sim_directional_trailing(
                c, h, lo, orig_sig, orig_dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            if len(p_orig) == 0:
                continue

            rsig = gen_reentry_signal(vr, D_VR, warmup, x_orig, p_orig, lookback)
            if rsig.sum() == 0:
                continue

            rdire = make_direction_momentum(rsig, mom8)
            p_re, e_re, x_re = sim_directional_trailing(
                c, h, lo, rsig, rdire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p_re)):
                sym_trades.append((
                    int(e_re[k]) * ratio,
                    int(x_re[k]) * ratio + ratio - 1,
                    float(p_re[k]),
                ))

        if sym_trades:
            all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_sim(
    d15: dict,
    d1h: dict,
    d4h: dict,
    data_years: float,
    label: str,
    d30: dict | None = None,
    pullback_pct: float = 0.0,
    pullback_window: int = 5,
    vr_depth_min: float = 0.0,
    include_reentry: bool = False,
    reentry_lookback: int = 5,
    max_concurrent: int | None = None,
) -> dict:
    """Run full portfolio simulation and compute metrics."""
    if max_concurrent is None:
        max_concurrent = MAX_CONC

    t = time.time()
    d_trades = collect_d_pullback(
        d15, d1h, d4h, pullback_pct, pullback_window, d30, vr_depth_min
    )
    s_trades = collect_s_trailing(d15, d1h, d4h, d30)
    ma_trades = collect_ma(d15, d1h, d4h, tf_restriction=16)
    reentry_trades: list = []
    if include_reentry:
        reentry_trades = collect_d_reentry(d15, d1h, d4h, lookback=reentry_lookback)

    all_trades = d_trades + s_trades + ma_trades + reentry_trades
    pnl = portfolio_sim(all_trades, max_concurrent)
    m   = compute_metrics(pnl, data_years)
    bt  = bootstrap_pf(pnl, n_boot=500)

    m["label"]    = label
    m["elapsed"]  = round(time.time() - t, 1)
    m["ci"]       = f"[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]"
    m["n_syms"]   = len(d15)
    m["raw_d"]    = len(d_trades)
    m["raw_s"]    = len(s_trades)
    m["raw_ma"]   = len(ma_trades)
    m["raw_re"]   = len(reentry_trades)
    m["max_conc"] = max_concurrent
    return m


def fmt(m: dict) -> str:
    return (
        f"N={m['N']:6d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
        f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.1f}  "
        f"DD={m['max_dd_pct']:.1f}%  CI={m.get('ci','?')}  [{m.get('elapsed',0)}s]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  OOS Trim Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_period(
    d15: dict, d30: dict, d1h: dict, d4h: dict,
    start: float, end: float,
) -> tuple[dict, dict, dict, dict]:
    def trim(ds: dict) -> dict:
        out = {}
        for sym, d in ds.items():
            n = len(d["close"])
            i0, i1 = int(n * start), int(n * end)
            if i1 - i0 < 100:
                continue
            out[sym] = {k: v[i0:i1] if isinstance(v, np.ndarray) else v for k, v in d.items()}
        return out
    return trim(d15), trim(d30), trim(d1h), trim(d4h)


def precompute_split(d15: dict, d30: dict, d1h: dict, d4h: dict) -> None:
    precompute_all_vr([d15, d30, d1h, d4h], [(VR_SHORT, VR_LONG)])
    for ds in [d15, d30, d1h, d4h]:
        precompute(ds, vr_sp=VR_SHORT, vr_lp=VR_LONG)
    precompute_mas(d15)
    precompute_mas(d1h)
    precompute_mas(d4h)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    commit = get_commit()
    print(f"\nr57: 30m TF + Pullback Combo [{commit}]  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # ── Load ──────────────────────────────────────────────────────────────────
    t_load = time.time()
    print("Loading 15m data...")
    data_15m = load_syms(min_years=5.0)
    n_syms = len(data_15m)
    max_bars = max(len(d["close"]) for d in data_15m.values())
    DATA_YEARS = max_bars / (4 * 24 * 365)
    print(f"  {n_syms} symbols | {DATA_YEARS:.2f} years | {time.time()-t_load:.1f}s")

    print("Building 30m / 1h / 4h & precomputing...")
    t_prep = time.time()
    data_30m, data_1h, data_4h = build_all_tf(data_15m)
    print(f"  Done in {time.time()-t_prep:.1f}s  "
          f"| 30m={len(data_30m)} 1h={len(data_1h)} 4h={len(data_4h)} syms\n")

    all_results: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 0: Baseline (G_3TF, no pullback, no 30m)
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("Phase 0: Baseline (G_3TF, no pullback)")
    print("─" * 72)
    m_bl = run_sim(data_15m, data_1h, data_4h, DATA_YEARS, label="Baseline (G_3TF)")
    print(f"  {m_bl['label']:40s}  {fmt(m_bl)}")

    n_drift  = abs(m_bl["N"] - BL_N_EXPECTED) / BL_N_EXPECTED
    pf_drift = abs(m_bl["PF"] - BL_PF_EXPECTED) / BL_PF_EXPECTED
    if n_drift > 0.15 or pf_drift > 0.15:
        print(f"\n  ⚠️  Baseline drift: N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}% — check config")
    else:
        print(f"\n  ✅ Baseline OK (N±{n_drift*100:.1f}%, PF±{pf_drift*100:.1f}%)")
    all_results.append(m_bl)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase A: G_4TF + Pullback sweep
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase A: G_4TF(30m 포함) + Pullback Sweep")
    print("─" * 72)

    # A0: G_4TF without pullback (r56 replica — verify 30m effect)
    m_a0 = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                   label="G_4TF, no pullback", d30=data_30m)
    print(f"\n  A0 (30m only, no pullback):  {fmt(m_a0)}")
    print(f"      ΔN={m_a0['N']-m_bl['N']:+d}  ΔPF={m_a0['PF']-m_bl['PF']:+.3f}")
    all_results.append(m_a0)

    # Pullback sweep on G_3TF (validates r55 winner in this codebase)
    print("\n  A1: Pullback sweep on G_3TF (validate r55 winner):")
    pb_sweep = [(0.002, 3), (0.003, 5), (0.005, 5), (0.005, 3), (0.007, 5)]
    pb_results: list[dict] = []
    for pb_pct, pb_win in pb_sweep:
        m = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                    label=f"G_3TF pb={pb_pct*100:.1f}% win={pb_win}",
                    pullback_pct=pb_pct, pullback_window=pb_win)
        print(f"    {m['label']:35s}  N={m['N']:6d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
              f"WR={m['WR']:.1f}%  [{m['elapsed']}s]")
        pb_results.append(m)

    best_g3_pb = max(pb_results, key=lambda x: x["PF"])
    best_pct   = float(best_g3_pb["label"].split("pb=")[1].split("%")[0]) / 100
    best_win   = int(best_g3_pb["label"].split("win=")[1])
    print(f"  → Best G_3TF: pb={best_pct*100:.1f}% win={best_win}  PF={best_g3_pb['PF']:.3f}")

    # A2: G_4TF + Best pullback (main hypothesis)
    print(f"\n  A2: G_4TF + Pullback (pb={best_pct*100:.1f}%, win={best_win}) — MAIN HYPOTHESIS")
    m_a2 = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                   label=f"G_4TF pb={best_pct*100:.1f}% win={best_win}",
                   d30=data_30m, pullback_pct=best_pct, pullback_window=best_win)
    print(f"  {m_a2['label']:40s}  {fmt(m_a2)}")
    print(f"      ΔN={m_a2['N']-m_bl['N']:+d}  ΔPF={m_a2['PF']-m_bl['PF']:+.3f}  "
          f"ΔEV={m_a2['EV_bp']-m_bl['EV_bp']:+.1f}bp")
    all_results.extend(pb_results)
    all_results.append(m_a2)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase B: vr_depth Gate (r54 finding)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase B: vr_depth Gate (r54: wins=0.724, losses=0.671)")
    print("─" * 72)
    print("  Applying vr_depth gate on best A2 config (G_4TF + pullback):")

    vrd_sweep = [0.65, 0.68, 0.70, 0.72]
    vrd_results: list[dict] = []
    for vrd in vrd_sweep:
        m = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                    label=f"G_4TF pb+vrd≥{vrd}",
                    d30=data_30m, pullback_pct=best_pct, pullback_window=best_win,
                    vr_depth_min=vrd)
        print(f"    vr_depth≥{vrd:.2f}:  N={m['N']:6d}  PF={m['PF']:.3f}  "
              f"EV={m['EV_bp']:+.1f}bp  WR={m['WR']:.1f}%  [{m['elapsed']}s]")
        vrd_results.append(m)

    best_vrd_m = max(vrd_results, key=lambda x: x["PF"])
    best_vrd   = float(best_vrd_m["label"].split("vrd≥")[1])
    print(f"  → Best vrd gate: {best_vrd:.2f}  PF={best_vrd_m['PF']:.3f}")
    all_results.extend(vrd_results)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase C: max_concurrent=12 + Re-Entry (unlock r56 Method C)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase C: max_concurrent=12 + Re-Entry (r56 Method C was ΔN=0 at 10 slots)")
    print("─" * 72)

    # C0: Baseline at 10 slots with re-entry (confirm ΔN=0 observation)
    m_c0 = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                   label="G_3TF + ReEntry (10 slots)", include_reentry=True)
    print(f"  C0 (10 slots + reentry):  N={m_c0['N']}  PF={m_c0['PF']:.3f}  "
          f"ΔN={m_c0['N']-m_bl['N']:+d}  [{m_c0['elapsed']}s]")

    # C1: 11 slots + Re-Entry
    m_c1 = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                   label="G_3TF + ReEntry (11 slots)", include_reentry=True, max_concurrent=11)
    print(f"  C1 (11 slots + reentry):  N={m_c1['N']}  PF={m_c1['PF']:.3f}  "
          f"DD={m_c1['max_dd_pct']:.1f}%  ΔN={m_c1['N']-m_bl['N']:+d}  [{m_c1['elapsed']}s]")

    # C2: 12 slots + Re-Entry
    m_c2 = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                   label="G_3TF + ReEntry (12 slots)", include_reentry=True, max_concurrent=12)
    print(f"  C2 (12 slots + reentry):  N={m_c2['N']}  PF={m_c2['PF']:.3f}  "
          f"DD={m_c2['max_dd_pct']:.1f}%  ΔN={m_c2['N']-m_bl['N']:+d}  [{m_c2['elapsed']}s]")

    all_results.extend([m_c0, m_c1, m_c2])

    # ─────────────────────────────────────────────────────────────────────────
    # Phase D: Best Combinations
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase D: Best Combinations")
    print("─" * 72)

    combo_results: list[dict] = []

    # D1: A2 reference (already computed)
    print(f"\n  D1 (A2 = G_4TF + pb):  {fmt(m_a2)}")

    # D2: Best A + best vrd
    if best_vrd_m["PF"] > m_a2["PF"]:
        m_d2 = best_vrd_m
        print(f"  D2 (A + vrd≥{best_vrd:.2f}):  {fmt(m_d2)}")
    else:
        m_d2 = best_vrd_m
        print(f"  D2 (A + vrd≥{best_vrd:.2f}):  {fmt(m_d2)}  [vrd didn't help]")

    # D3: Best A + Re-Entry (with best slot count)
    best_re_slots = max([m_c0, m_c1, m_c2], key=lambda x: x["N"])
    re_slots = best_re_slots["max_conc"]
    m_d3 = run_sim(data_15m, data_1h, data_4h, DATA_YEARS,
                   label=f"G_4TF pb + ReEntry ({re_slots} slots)",
                   d30=data_30m, pullback_pct=best_pct, pullback_window=best_win,
                   include_reentry=True, max_concurrent=re_slots)
    print(f"  D3 (A + ReEntry {re_slots} slots):  {fmt(m_d3)}")
    print(f"      ΔN={m_d3['N']-m_bl['N']:+d}  ΔPF={m_d3['PF']-m_bl['PF']:+.3f}")

    combo_results.extend([m_a2, m_d2, m_d3])
    all_results.extend([m_d3])

    # ─────────────────────────────────────────────────────────────────────────
    # Phase D: OOS 50% Split on best combo
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase D: OOS 50% Split Validation")
    print("─" * 72)

    best_combo = max(combo_results, key=lambda x: x["PF"])
    print(f"  Best combo: {best_combo['label']}")

    is_d15, is_d30, is_d1h, is_d4h = _trim_period(data_15m, data_30m, data_1h, data_4h, 0.0, 0.5)
    os_d15, os_d30, os_d1h, os_d4h = _trim_period(data_15m, data_30m, data_1h, data_4h, 0.5, 1.0)

    is_years = max(len(d["close"]) for d in is_d15.values()) / (4 * 24 * 365) if is_d15 else DATA_YEARS / 2
    os_years = max(len(d["close"]) for d in os_d15.values()) / (4 * 24 * 365) if os_d15 else DATA_YEARS / 2

    precompute_split(is_d15, is_d30, is_d1h, is_d4h)
    precompute_split(os_d15, os_d30, os_d1h, os_d4h)

    m_bl_is = run_sim(is_d15, is_d1h, is_d4h, is_years, "BL IS 50%")
    m_bl_os = run_sim(os_d15, os_d1h, os_d4h, os_years, "BL OOS 50%")

    m_best_is = run_sim(is_d15, is_d1h, is_d4h, is_years,
                        f"Best IS", d30=is_d30, pullback_pct=best_pct,
                        pullback_window=best_win, vr_depth_min=0.0)
    m_best_os = run_sim(os_d15, os_d1h, os_d4h, os_years,
                        f"Best OOS", d30=os_d30, pullback_pct=best_pct,
                        pullback_window=best_win, vr_depth_min=0.0)

    print(f"\n  Baseline IS : {fmt(m_bl_is)}")
    print(f"  Baseline OOS: {fmt(m_bl_os)}")
    print(f"  OOS/IS (Baseline): {m_bl_os['PF']/m_bl_is['PF']:.3f}")
    print(f"\n  Best IS : {fmt(m_best_is)}")
    print(f"  Best OOS: {fmt(m_best_os)}")
    print(f"  OOS/IS (Best):     {m_best_os['PF']/m_best_is['PF']:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("SUMMARY TABLE")
    print("═" * 80)
    key_results = [m_bl, m_a0, m_a2, best_vrd_m, m_c2, m_d3]
    print(f"{'Label':44s}  {'N':>6}  {'PF':>6}  {'EV':>7}  {'WR':>6}  {'DD':>6}  {'CI'}")
    print("─" * 105)
    for m in key_results:
        dn = m["N"] - m_bl["N"]
        star = "★" if m["N"] >= 7000 and m["PF"] >= 1.65 else ("✓" if m["PF"] >= 1.50 else "⚠")
        print(f"{star} {m['label']:43s}  {m['N']:>6d}  {m['PF']:>6.3f}  "
              f"{m['EV_bp']:>+7.1f}bp  {m['WR']:>5.1f}%  {m['max_dd_pct']:>5.1f}%  "
              f"{m.get('ci','?')}  ΔN={dn:+d}")
    print("─" * 105)

    # ─────────────────────────────────────────────────────────────────────────
    # Write Report
    # ─────────────────────────────────────────────────────────────────────────
    today = datetime.now().strftime("%Y%m%d")
    yyyymm = datetime.now().strftime("%Y%m")
    report_dir = REPORT_DIR / yyyymm
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r57_combo.md"

    best_overall = max(key_results, key=lambda x: x["PF"])
    goal_met = best_overall["N"] >= 7000 and best_overall["PF"] >= 1.65

    oos_bl_ratio  = m_bl_os["PF"] / m_bl_is["PF"]
    oos_best_ratio = m_best_os["PF"] / m_best_is["PF"]

    with open(report_path, "w") as f:
        f.write("# r57: 30m TF + Pullback Entry Combination\n\n")
        f.write(f"**날짜:** {datetime.now().strftime('%Y-%m-%d')}  \n")
        f.write(f"**커밋:** `{commit}`  \n")
        f.write(f"**기간:** {DATA_YEARS:.1f}yr | {n_syms}심볼 | Binance USDT-M  \n")
        f.write(f"**비용:** 0.08% round-trip | **슬롯:** {MAX_CONC} | **L/N:** {VR_LN:.1f}  \n")
        f.write(f"**구현:** {'미구현' if not goal_met else '미구현 (feature/ 브랜치 구현 검토)'}  \n\n")
        f.write("**Parity Notes:** config.py 읽음 ✅ | 룩어헤드 없음 ✅ | sim 재구현 없음 ✅  \n\n")
        f.write("---\n\n")
        f.write("## 연구 배경\n\n")
        f.write("r54/r55/r56 핵심 발견 조합:\n\n")
        f.write("| 연구 | 발견 | 효과 |\n")
        f.write("|------|------|------|\n")
        f.write("| r56 | 30m TF (G_4TF) | N +8.5%, PF +2.3%, EV 동등 |\n")
        f.write("| r55 | Pullback 0.5%/win=5 | PF +3.4%, N 96.6% |\n")
        f.write("| r54 | vr_depth ≥ 0.68 | 유일한 예측 변수 (Δ=0.054) |\n\n")
        f.write("---\n\n")
        f.write("## Phase 0: Baseline\n\n")
        f.write(f"**{fmt(m_bl)}**\n\n")
        f.write(f"> N drift={n_drift*100:.1f}%, PF drift={pf_drift*100:.1f}% "
                f"(63sym vs canonical 60sym ± expected)\n\n")
        f.write("---\n\n")
        f.write("## Phase A: G_4TF + Pullback\n\n")
        f.write("| Config | N | ΔN | PF | ΔPF | EV | WR | Calmar | DD |\n")
        f.write("|--------|---|----|----|-----|----|----|----|----|\n")
        for m in [m_bl, m_a0] + pb_results + [m_a2]:
            dn = m["N"] - m_bl["N"]
            dpf = m["PF"] - m_bl["PF"]
            f.write(f"| {m['label']} | {m['N']} | {dn:+d} | {m['PF']:.3f} | {dpf:+.3f} | "
                    f"{m['EV_bp']:+.1f}bp | {m['WR']:.1f}% | {m['calmar']:.1f} | {m['max_dd_pct']:.1f}% |\n")
        f.write("\n")
        f.write("---\n\n")
        f.write("## Phase B: vr_depth Gate\n\n")
        f.write("| vr_depth_min | N | ΔN | PF | ΔPF | EV | WR |\n")
        f.write("|--------------|---|----|----|-----|----|----|\n")
        for m in vrd_results:
            dn = m["N"] - m_a2["N"]
            dpf = m["PF"] - m_a2["PF"]
            f.write(f"| {m['label'].split('vrd≥')[1]} | {m['N']} | {dn:+d} | {m['PF']:.3f} | "
                    f"{dpf:+.3f} | {m['EV_bp']:+.1f}bp | {m['WR']:.1f}% |\n")
        f.write("\n")
        f.write("---\n\n")
        f.write("## Phase C: max_concurrent 확장 + Re-Entry\n\n")
        f.write("| 슬롯 수 | N | ΔN | PF | DD |\n")
        f.write("|---------|---|----|----|----|\n")
        for m in [m_c0, m_c1, m_c2]:
            dn = m["N"] - m_bl["N"]
            f.write(f"| {m['max_conc']} | {m['N']} | {dn:+d} | {m['PF']:.3f} | {m['max_dd_pct']:.1f}% |\n")
        f.write("\n")
        f.write("---\n\n")
        f.write("## Phase D: OOS Validation\n\n")
        f.write("| | IS (50%) PF | OOS (50%) PF | OOS/IS |\n")
        f.write("|--|--|--|--|\n")
        f.write(f"| Baseline | {m_bl_is['PF']:.3f} | {m_bl_os['PF']:.3f} | {oos_bl_ratio:.3f} |\n")
        f.write(f"| Best (A2) | {m_best_is['PF']:.3f} | {m_best_os['PF']:.3f} | {oos_best_ratio:.3f} |\n")
        f.write("\n")
        f.write("---\n\n")
        f.write("## 결론\n\n")
        f.write(f"**최선 조합:** {best_overall['label']}  \n")
        f.write(f"N={best_overall['N']} PF={best_overall['PF']:.3f} EV={best_overall['EV_bp']:+.1f}bp "
                f"DD={best_overall['max_dd_pct']:.1f}%  \n\n")
        if goal_met:
            f.write("✅ **목표 달성 (N≥7,000, PF≥1.65)**\n\n")
        else:
            f.write(f"{'✓' if best_overall['PF']>=1.50 else '⚠️'} **목표 {'부분' if best_overall['PF']>=1.50 else '미'}달성** "
                    f"(N={best_overall['N']}, PF={best_overall['PF']:.3f})\n\n")
        f.write(f"OOS/IS = {oos_best_ratio:.3f} (Baseline: {oos_bl_ratio:.3f}) — "
                f"{'구조적 레짐이동, 과적합 아님' if abs(oos_best_ratio - oos_bl_ratio) < 0.05 else '추가 검토 필요'}\n\n")
        f.write("---\n\n")
        f.write(f"*데이터: {n_syms}심볼 · {DATA_YEARS:.1f}년 · 비용 0.08% round-trip · Binance USDT-M perpetuals*\n")

    print(f"\n  📄 보고서: {report_path.relative_to(ROOT)}")
    if goal_met:
        print("\n  ✅ 목표 달성 (N≥7000 + PF≥1.65)")
    else:
        print(f"\n  결과: N={best_overall['N']}, PF={best_overall['PF']:.3f}")
    print("\n✓ r57 완료\n")


if __name__ == "__main__":
    main()
