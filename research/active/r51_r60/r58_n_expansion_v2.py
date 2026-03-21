#!/usr/bin/env python3
"""r58: N Expansion v2 — Universe Widening + VR Threshold Sweep.

Starting from r57 A0 (G_4TF, 60-sym, N=7,119, PF=1.676, Calmar=127.2),
find ways to increase N by ≥20% (target N ≥ 8,540).

Three research phases:
  Phase A: Universe expansion — add non-whitelist symbols with ≥3yr cache data
            Candidates: HBAR, 1000SHIB, DYDX, APE, MATIC, OP, INJ, LDO, ICP, APT, FET
            → New DeFi/L2 regime-native assets. Use _offset for exact time alignment.

  Phase B: VR threshold sweep
            B1: d_vr_threshold: 0.50 → 0.52 / 0.54 / 0.56 (higher = more D signals)
            B2: s_vr_threshold: 0.55 → 0.57 / 0.59
            B3: s_squeeze_dur: 4 → 3 / 2

  Phase C: Combined best A + best B (universe + threshold)
            Yearly breakdown 2020-2025: all years positive, 2023+ stronger.

룩어헤드 없음:
  - Non-whitelist symbols: _offset = max_bars - local_bars → time-aligned
  - No future VR, no future close in signal generation

Usage:
    uv run python research/active/r51_r60/r58_n_expansion_v2.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
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
from quantbox.backtest.trade_sim import sim_directional_trailing, sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Load production config (NO hardcoding) ───────────────────────────────────
cfg = VolStraddleConfig()

VR_SHORT = cfg.vr_short  # 8
VR_LONG = cfg.vr_long  # 56
D_VR = cfg.d_vr_threshold  # 0.50
S_VR = cfg.s_vr_threshold  # 0.55
S_DUR = cfg.s_squeeze_dur  # 4
D_SL = cfg.d_sl  # 0.01
D_TP = cfg.d_tp  # 0.12
D_TRAIL_ACT = cfg.d_trail_activate  # 0.005
D_TRAIL_DIST = cfg.d_trail_distance  # 0.002
S_SL = cfg.s_sl
S_TP = cfg.s_tp
S_TRAIL_ACT = cfg.s_trail_activate
S_TRAIL_DIST = cfg.s_trail_distance
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

# 30m TF (same as r57)
WARMUP_30m: int = WARMUP_15m // 2
_MH30m: int = cfg.max_hold_15m // 2
TF_30m = ("30m", 2, WARMUP_30m, _MH30m, _MH30m * 2)
G_4TF = [TF_30m] + list(G_3TF)

VR_KEY = f"vr_{VR_SHORT}_{VR_LONG}"

# Phase 1 baseline assertions (r57 60-sym results)
BL_N_EXPECTED = 6207
BL_PF_EXPECTED = 1.563

# Config guards
assert D_TRAIL_ACT == 0.040, f"d_trail_activate mismatch: {D_TRAIL_ACT}"
assert D_TRAIL_DIST == 0.005, f"d_trail_distance mismatch: {D_TRAIL_DIST}"  # r106: 0.002→0.005 (rule #8)

print("\n[Config] 검증")
print(f"  D: SL={D_SL}, TP={D_TP}, trail=({D_TRAIL_ACT},{D_TRAIL_DIST})")
print(f"  VR_LN={VR_LN:.1f}  MAX_CONC={MAX_CONC}")
print(f"  d_vr={D_VR}  s_vr={S_VR}  s_dur={S_DUR}")
print("  ✅ All params from config.py\n")

CACHE_DIR = ROOT / "data" / "cache_extended"
REPORT_DIR = ROOT / "docs" / "backtest_results"

# Cache date → DATA_END for bar-to-year mapping
# Inferred from filename: BTCUSDT_15m_2190d_20260220.npz → Feb 20, 2026
DATA_END = datetime(2026, 2, 20)

# Non-whitelist candidates confirmed in cache with ≥3yr data
EXTRA_SYMS_3YR = [
    "HBARUSDT",  # 4.9yr — Hedera (DeFi)
    "1000SHIBUSDT",  # 4.8yr — Shib meme (high vol)
    "DYDXUSDT",  # 4.5yr — DeFi DEX
    "APEUSDT",  # 3.9yr — NFT gaming
    "MATICUSDT",  # 3.9yr — Polygon L2
    "OPUSDT",  # 3.7yr — Optimism L2
    "INJUSDT",  # 3.5yr — Injective DeFi
    "LDOUSDT",  # 3.4yr — Lido staking
    "ICPUSDT",  # 3.4yr — Internet Computer
    "APTUSDT",  # 3.3yr — Aptos L1
    "FETUSDT",  # 3.1yr — Fetch.ai AI
]


def get_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def _load_sym_dict(allowed: set[str], min_bars: int) -> dict:
    """Core loader: finds best (longest) NPZ per symbol, sets _offset."""
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in allowed:
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


def load_syms(min_years: float = 5.0) -> dict:
    """Load whitelist-only (60 live symbols) with ≥min_years history."""
    whitelist = set(cfg.symbol_whitelist)
    min_bars = int(min_years * 365 * 24 * 4)
    return _load_sym_dict(whitelist, min_bars)


def load_extended_syms(extra: list[str], min_years: float = 3.0) -> dict:
    """Load whitelist + extra non-whitelist symbols (min_years=3.0 for new syms).

    _offset is set relative to global max_bars across ALL loaded symbols,
    ensuring time alignment even when non-whitelist symbols have shorter history.
    """
    allowed = set(cfg.symbol_whitelist) | set(extra)
    min_bars = int(min_years * 365 * 24 * 4)
    return _load_sym_dict(allowed, min_bars)


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
#  Time Mapping (for yearly breakdown)
# ═══════════════════════════════════════════════════════════════════════════════


def bar_to_year(global_15m_bar: int, max_bars: int) -> int:
    """Convert global 15m bar index to calendar year.

    Uses DATA_END (2026-02-20) as anchor: bar 0 = DATA_END - max_bars*15min.
    No lookahead: uses completed bar index (entry bar).
    """
    minutes_from_start = global_15m_bar * 15
    total_minutes = max_bars * 15
    dt = DATA_END - timedelta(minutes=total_minutes) + timedelta(minutes=minutes_from_start)
    return dt.year


def yearly_breakdown(
    all_trades: list,
    max_bars: int,
    max_concurrent: int = 10,
) -> dict[int, dict]:
    """Compute per-year N, PF, WR, EV by running portfolio_sim per year.

    Args:
        all_trades: List of (sym, entry_15m_global, exit_15m_global, pnl).
        max_bars: Total bars in dataset (for year mapping).
        max_concurrent: Slot limit.

    Returns:
        {year: {N, PF, WR, EV_bp}} dict sorted by year.
    """
    by_year: dict[int, list] = defaultdict(list)
    for trade in all_trades:
        year = bar_to_year(trade[1], max_bars)
        by_year[year].append(trade)

    results: dict[int, dict] = {}
    for year in sorted(by_year.keys()):
        yr_trades = by_year[year]
        pnl_arr = portfolio_sim(yr_trades, max_concurrent)
        if len(pnl_arr) == 0:
            continue
        N = int(len(pnl_arr))
        wins = int((pnl_arr > 0).sum())
        losses = int((pnl_arr < 0).sum())
        WR = wins / N * 100
        gw = float(pnl_arr[pnl_arr > 0].sum()) if wins > 0 else 0.0
        gl = float(-pnl_arr[pnl_arr < 0].sum()) if losses > 0 else 0.0
        PF = gw / gl if gl > 0 else float("inf")
        EV = float(pnl_arr.mean()) * 10_000
        results[year] = {"N": N, "PF": PF, "WR": WR, "EV_bp": EV}
    return results


def fmt_yearly(by_year: dict) -> str:
    """Format yearly breakdown as compact table string."""
    lines = []
    for yr, m in sorted(by_year.items()):
        ok = "✅" if m["PF"] >= 1.0 else "❌"
        lines.append(f"    {yr} {ok}  N={m['N']:5d}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  EV={m['EV_bp']:+.0f}bp")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Trade Collection (with _offset for time alignment)
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


def collect_d_trailing(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
    d_vr: float = D_VR,
) -> list:
    """D_mom8 VR squeeze directional trades with trailing stop.

    Applies _offset (from data_15m[sym]) for correct time-alignment of symbols
    with different history lengths. Critical for universe expansion correctness.

    Args:
        d_vr: VR threshold for D signal. Default from config (0.50).
    """
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
                c,
                h,
                lo,
                sig,
                dire,
                D_SL,
                D_TP,
                mh_d,
                COST_DIRECTIONAL,
                D_TRAIL_ACT,
                D_TRAIL_DIST,
            )
            for k in range(len(p)):
                # Apply offset: local bar → global 15m bar
                sym_trades.append(
                    (
                        int(e[k]) * ratio + offset,
                        int(x[k]) * ratio + ratio - 1 + offset,
                        float(p[k]),
                    )
                )
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_s_trailing(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
    s_vr: float = S_VR,
    s_dur: int = S_DUR,
) -> list:
    """S_d8 straddle trades with configurable vr threshold and squeeze duration."""
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
            p, e, x = sim_straddle_trailing(
                c,
                h,
                lo,
                sig,
                S_SL,
                S_TP,
                mh_s,
                COST_STRADDLE,
                S_TRAIL_ACT,
                S_TRAIL_DIST,
            )
            for k in range(len(p)):
                sym_trades.append(
                    (
                        int(e[k]) * ratio + offset,
                        int(x[k]) * ratio + ratio - 1 + offset,
                        float(p[k]),
                    )
                )
        all_trades.extend(_dedup_sym(sym, sym_trades))
    return all_trades


def collect_ma(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
) -> list:
    """MA cluster trades (4h-only)."""
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
#  Simulation Runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_sim(
    d15: dict,
    d1h: dict,
    d4h: dict,
    data_years: float,
    label: str,
    d30: dict | None = None,
    d_vr: float = D_VR,
    s_vr: float = S_VR,
    s_dur: int = S_DUR,
    max_concurrent: int | None = None,
    show_yearly: bool = False,
) -> dict:
    """Run full portfolio simulation + compute metrics."""
    if max_concurrent is None:
        max_concurrent = MAX_CONC

    t = time.time()
    d_trades = collect_d_trailing(d15, d1h, d4h, d30, d_vr=d_vr)
    s_trades = collect_s_trailing(d15, d1h, d4h, d30, s_vr=s_vr, s_dur=s_dur)
    ma_trades = collect_ma(d15, d1h, d4h)

    all_trades = d_trades + s_trades + ma_trades
    pnl = portfolio_sim(all_trades, max_concurrent)
    m = compute_metrics(pnl, data_years)
    bt = bootstrap_pf(pnl, n_boot=500)

    max_bars = max(len(d["close"]) for d in d15.values()) if d15 else 0
    m["label"] = label
    m["elapsed"] = round(time.time() - t, 1)
    m["ci"] = f"[{bt['CI_lo']:.3f},{bt['CI_hi']:.3f}]"
    m["n_syms"] = len(d15)
    m["raw_d"] = len(d_trades)
    m["raw_s"] = len(s_trades)
    m["raw_ma"] = len(ma_trades)
    m["max_conc"] = max_concurrent

    if show_yearly and max_bars > 0:
        m["yearly"] = yearly_breakdown(all_trades, max_bars, max_concurrent)

    return m


def fmt(m: dict) -> str:
    return (
        f"N={m['N']:6d}  PF={m['PF']:.3f}  EV={m['EV_bp']:+.1f}bp  "
        f"WR={m['WR']:.1f}%  Calmar={m['calmar']:.1f}  "
        f"DD={m['max_dd_pct']:.1f}%  CI={m.get('ci', '?')}  "
        f"[{m['n_syms']}sym/{m.get('elapsed', 0)}s]"
    )


def all_years_positive(yearly: dict) -> bool:
    """Check if all years (2020-2025) have PF≥1.0."""
    for yr, ym in yearly.items():
        if 2020 <= yr <= 2025 and ym["PF"] < 1.0:
            return False
    return True


def recent_calmar_better(yearly: dict) -> str:
    """Check if 2023+ EV is ≥ 2020-2022 average EV."""
    early = [ym["EV_bp"] for yr, ym in yearly.items() if 2021 <= yr <= 2022]
    late = [ym["EV_bp"] for yr, ym in yearly.items() if 2023 <= yr <= 2025]
    if not early or not late:
        return "N/A"
    avg_e, avg_l = sum(early) / len(early), sum(late) / len(late)
    diff = avg_l - avg_e
    return f"{diff:+.0f}bp ({'✅ 개선' if diff >= 0 else '⚠️ 하락'})"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit()
    print(f"\nr58: N Expansion v2 [{commit}]  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # ── Load 60-sym whitelist (min_years=5.0) ─────────────────────────────────
    t_load = time.time()
    print("Loading 60-sym whitelist (min_years=5.0)...")
    data_15m = load_syms(min_years=5.0)
    max_bars = max(len(d["close"]) for d in data_15m.values())
    DATA_YEARS_60 = max_bars / (4 * 24 * 365)
    print(f"  60-sym set: {len(data_15m)} symbols | {DATA_YEARS_60:.2f} yr | {time.time() - t_load:.1f}s")

    t_prep = time.time()
    d30_60, d1h_60, d4h_60 = build_all_tf(data_15m)
    print(f"  Precomputed: 30m={len(d30_60)} 1h={len(d1h_60)} 4h={len(d4h_60)} | {time.time() - t_prep:.1f}s\n")

    all_results: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 0: Baseline (60-sym G_3TF) — parity check
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("Phase 0: Baseline (60-sym G_3TF, parity check)")
    print("─" * 72)
    m_bl = run_sim(data_15m, d1h_60, d4h_60, DATA_YEARS_60, label="Baseline 60-sym G_3TF")
    print(f"  {fmt(m_bl)}")

    n_drift = abs(m_bl["N"] - BL_N_EXPECTED) / BL_N_EXPECTED
    pf_drift = abs(m_bl["PF"] - BL_PF_EXPECTED) / BL_PF_EXPECTED
    if n_drift > 0.15 or pf_drift > 0.15:
        print(f"  ⚠️  Parity FAIL: N±{n_drift * 100:.1f}%, PF±{pf_drift * 100:.1f}%")
    else:
        print(f"  ✅ Parity OK (N±{n_drift * 100:.1f}%, PF±{pf_drift * 100:.1f}%)")
    all_results.append(m_bl)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 0b: A0 reference (60-sym G_4TF) — matches r57 A0
    # ─────────────────────────────────────────────────────────────────────────
    print("\n─" * 72)
    print("Phase 0b: A0 Reference (60-sym G_4TF) — r57 A0 재현")
    print("─" * 72)
    m_a0_ref = run_sim(data_15m, d1h_60, d4h_60, DATA_YEARS_60, label="A0-ref 60-sym G_4TF", d30=d30_60)
    print(f"  {fmt(m_a0_ref)}")
    print(f"  vs r57 A0: N=7119 PF=1.676 WR=48.0% Calmar=127.2")
    print(f"  ΔN={m_a0_ref['N'] - 7119:+d}  ΔPF={m_a0_ref['PF'] - 1.676:+.3f}")
    TARGET_N = int(m_a0_ref["N"] * 1.20)
    print(f"\n  🎯 Target N ≥ {TARGET_N} (+20% from {m_a0_ref['N']})")
    all_results.append(m_a0_ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase A: Universe Expansion
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase A: Universe Expansion (+11 symbols, min_years=3.0)")
    print(f"  Extra: {', '.join(EXTRA_SYMS_3YR)}")
    print("─" * 72)

    t_ext = time.time()
    data_15m_ext = load_extended_syms(EXTRA_SYMS_3YR, min_years=3.0)
    max_bars_ext = max(len(d["close"]) for d in data_15m_ext.values())
    DATA_YEARS_EXT = max_bars_ext / (4 * 24 * 365)
    n_extra = len(data_15m_ext) - len(data_15m)
    print(
        f"\n  Extended set: {len(data_15m_ext)} syms (+{n_extra} new) | "
        f"{DATA_YEARS_EXT:.2f} yr | {time.time() - t_ext:.1f}s"
    )
    print(f"  New symbols loaded: {sorted(set(data_15m_ext) - set(data_15m))}")

    t_prep_ext = time.time()
    d30_ext, d1h_ext, d4h_ext = build_all_tf(data_15m_ext)
    print(f"  Precomputed ext: {time.time() - t_prep_ext:.1f}s\n")

    # A1: Extended universe with G_3TF
    print("  A1: Extended universe, G_3TF")
    m_a1 = run_sim(data_15m_ext, d1h_ext, d4h_ext, DATA_YEARS_EXT, label="A1: G_3TF ext-universe", show_yearly=True)
    print(f"  {fmt(m_a1)}")
    print(f"  ΔN={m_a1['N'] - m_bl['N']:+d}  ΔPF={m_a1['PF'] - m_bl['PF']:+.3f}")
    if "yearly" in m_a1:
        print(fmt_yearly(m_a1["yearly"]))
        ok = all_years_positive(m_a1["yearly"])
        rec = recent_calmar_better(m_a1["yearly"])
        print(f"  모든 해 수익: {'✅' if ok else '❌'}  최근 트렌드: {rec}")
    all_results.append(m_a1)

    # A2: Extended universe with G_4TF (30m added)
    print("\n  A2: Extended universe, G_4TF (30m + ext symbols)")
    m_a2 = run_sim(
        data_15m_ext, d1h_ext, d4h_ext, DATA_YEARS_EXT, label="A2: G_4TF ext-universe", d30=d30_ext, show_yearly=True
    )
    print(f"  {fmt(m_a2)}")
    print(f"  ΔN from bl={m_a2['N'] - m_bl['N']:+d}  ΔN from A0={m_a2['N'] - m_a0_ref['N']:+d}")
    print(f"  ΔPF={m_a2['PF'] - m_bl['PF']:+.3f}")
    if "yearly" in m_a2:
        print(fmt_yearly(m_a2["yearly"]))
        ok = all_years_positive(m_a2["yearly"])
        rec = recent_calmar_better(m_a2["yearly"])
        print(f"  모든 해 수익: {'✅' if ok else '❌'}  최근 트렌드: {rec}")
    hit = (
        "🎯 TARGET HIT!"
        if m_a2["N"] >= TARGET_N
        else f"({m_a2['N']}/{TARGET_N}, +{(m_a2['N'] - m_a0_ref['N']) / m_a0_ref['N'] * 100:.1f}%)"
    )
    print(f"  {hit}")
    all_results.append(m_a2)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase B: VR Threshold Sweep (on 60-sym G_4TF)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase B: VR Threshold Sweep (60-sym G_4TF — isolate threshold effect)")
    print("─" * 72)

    # B1: d_vr_threshold sweep (D signal)
    print("\n  B1: d_vr_threshold sweep (higher = more D signals)")
    d_vr_grid = [0.50, 0.52, 0.54, 0.56]
    b1_results: list[dict] = []
    for dvr in d_vr_grid:
        m = run_sim(data_15m, d1h_60, d4h_60, DATA_YEARS_60, label=f"B1 d_vr={dvr:.2f}", d30=d30_60, d_vr=dvr)
        star = "←base" if dvr == D_VR else ""
        print(
            f"    d_vr={dvr:.2f}: N={m['N']:6d}  PF={m['PF']:.3f}  "
            f"EV={m['EV_bp']:+.1f}bp  WR={m['WR']:.1f}%  ΔN={m['N'] - m_a0_ref['N']:+d} {star}"
        )
        b1_results.append(m)
        all_results.append(m)

    best_b1 = max(b1_results, key=lambda x: x["N"] if x["PF"] >= 1.55 else 0)
    best_dvr = float(best_b1["label"].split("=")[1])
    print(f"  → Best d_vr (N-max, PF≥1.55): {best_dvr:.2f}  N={best_b1['N']}")

    # B2: s_vr_threshold sweep (S signal)
    print("\n  B2: s_vr_threshold sweep (higher = more S signals)")
    s_vr_grid = [0.55, 0.57, 0.59]
    b2_results: list[dict] = []
    for svr in s_vr_grid:
        m = run_sim(data_15m, d1h_60, d4h_60, DATA_YEARS_60, label=f"B2 s_vr={svr:.2f}", d30=d30_60, s_vr=svr)
        star = "←base" if svr == S_VR else ""
        print(
            f"    s_vr={svr:.2f}: N={m['N']:6d}  PF={m['PF']:.3f}  "
            f"EV={m['EV_bp']:+.1f}bp  WR={m['WR']:.1f}%  ΔN={m['N'] - m_a0_ref['N']:+d} {star}"
        )
        b2_results.append(m)
        all_results.append(m)

    best_b2 = max(b2_results, key=lambda x: x["N"] if x["PF"] >= 1.55 else 0)
    best_svr = float(best_b2["label"].split("=")[1])
    print(f"  → Best s_vr: {best_svr:.2f}  N={best_b2['N']}")

    # B3: s_squeeze_dur sweep (S signal)
    print("\n  B3: s_squeeze_dur sweep (lower = more S signals)")
    s_dur_grid = [4, 3, 2]
    b3_results: list[dict] = []
    for sdur in s_dur_grid:
        m = run_sim(data_15m, d1h_60, d4h_60, DATA_YEARS_60, label=f"B3 s_dur={sdur}", d30=d30_60, s_dur=sdur)
        star = "←base" if sdur == S_DUR else ""
        print(
            f"    s_dur={sdur}:   N={m['N']:6d}  PF={m['PF']:.3f}  "
            f"EV={m['EV_bp']:+.1f}bp  WR={m['WR']:.1f}%  ΔN={m['N'] - m_a0_ref['N']:+d} {star}"
        )
        b3_results.append(m)
        all_results.append(m)

    best_b3 = max(b3_results, key=lambda x: x["N"] if x["PF"] >= 1.55 else 0)
    best_sdur = int(best_b3["label"].split("=")[1])
    print(f"  → Best s_dur: {best_sdur}  N={best_b3['N']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase C: Combined A + B
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Phase C: Combined A (ext universe) + B (best thresholds)")
    print("─" * 72)

    # C1: Ext universe + best d_vr
    print(f"\n  C1: Ext universe + d_vr={best_dvr:.2f}")
    m_c1 = run_sim(
        data_15m_ext,
        d1h_ext,
        d4h_ext,
        DATA_YEARS_EXT,
        label=f"C1: G_4TF ext + d_vr={best_dvr:.2f}",
        d30=d30_ext,
        d_vr=best_dvr,
        show_yearly=True,
    )
    print(f"  {fmt(m_c1)}")
    print(f"  ΔN from A0={m_c1['N'] - m_a0_ref['N']:+d}  +{(m_c1['N'] - m_a0_ref['N']) / m_a0_ref['N'] * 100:.1f}%")
    if "yearly" in m_c1:
        print(fmt_yearly(m_c1["yearly"]))
        ok = all_years_positive(m_c1["yearly"])
        rec = recent_calmar_better(m_c1["yearly"])
        print(f"  모든 해 수익: {'✅' if ok else '❌'}  최근 트렌드: {rec}")
    hit = "🎯 TARGET HIT!" if m_c1["N"] >= TARGET_N else f"({m_c1['N']}/{TARGET_N})"
    print(f"  {hit}")
    all_results.append(m_c1)

    # C2: Ext universe + best s_vr + best s_dur
    print(f"\n  C2: Ext universe + s_vr={best_svr:.2f} + s_dur={best_sdur}")
    m_c2 = run_sim(
        data_15m_ext,
        d1h_ext,
        d4h_ext,
        DATA_YEARS_EXT,
        label=f"C2: G_4TF ext + s_vr={best_svr:.2f}/dur={best_sdur}",
        d30=d30_ext,
        s_vr=best_svr,
        s_dur=best_sdur,
        show_yearly=True,
    )
    print(f"  {fmt(m_c2)}")
    print(f"  ΔN from A0={m_c2['N'] - m_a0_ref['N']:+d}  +{(m_c2['N'] - m_a0_ref['N']) / m_a0_ref['N'] * 100:.1f}%")
    if "yearly" in m_c2:
        print(fmt_yearly(m_c2["yearly"]))
        ok = all_years_positive(m_c2["yearly"])
        rec = recent_calmar_better(m_c2["yearly"])
        print(f"  모든 해 수익: {'✅' if ok else '❌'}  최근 트렌드: {rec}")
    hit = "🎯 TARGET HIT!" if m_c2["N"] >= TARGET_N else f"({m_c2['N']}/{TARGET_N})"
    print(f"  {hit}")
    all_results.append(m_c2)

    # C3: Full combination (best A + best d_vr + best s params)
    print(f"\n  C3: Full combo — ext + d_vr={best_dvr:.2f} + s_vr={best_svr:.2f} + s_dur={best_sdur}")
    m_c3 = run_sim(
        data_15m_ext,
        d1h_ext,
        d4h_ext,
        DATA_YEARS_EXT,
        label=f"C3: Full-combo",
        d30=d30_ext,
        d_vr=best_dvr,
        s_vr=best_svr,
        s_dur=best_sdur,
        show_yearly=True,
    )
    print(f"  {fmt(m_c3)}")
    print(f"  ΔN from A0={m_c3['N'] - m_a0_ref['N']:+d}  +{(m_c3['N'] - m_a0_ref['N']) / m_a0_ref['N'] * 100:.1f}%")
    if "yearly" in m_c3:
        print(fmt_yearly(m_c3["yearly"]))
        ok = all_years_positive(m_c3["yearly"])
        rec = recent_calmar_better(m_c3["yearly"])
        print(f"  모든 해 수익: {'✅' if ok else '❌'}  최근 트렌드: {rec}")
    hit = "🎯 TARGET HIT!" if m_c3["N"] >= TARGET_N else f"({m_c3['N']}/{TARGET_N})"
    print(f"  {hit}")
    all_results.append(m_c3)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("SUMMARY TABLE")
    print("═" * 80)

    key_results = [m_bl, m_a0_ref, m_a1, m_a2, best_b1, best_b2, best_b3, m_c1, m_c2, m_c3]
    print(f"{'Label':46s}  {'N':>6}  {'PF':>6}  {'EV':>7}  {'WR':>6}  {'Calmar':>7}  {'DD':>6}")
    print("─" * 110)
    for m in key_results:
        dn = m["N"] - m_a0_ref["N"]
        star = "🎯" if m["N"] >= TARGET_N and m["PF"] >= 1.55 else ("✓" if m["PF"] >= 1.55 else "⚠")
        print(
            f"{star} {m['label']:45s}  {m['N']:>6d}  {m['PF']:>6.3f}  "
            f"{m['EV_bp']:>+7.1f}bp  {m['WR']:>5.1f}%  {m['calmar']:>7.1f}  "
            f"{m['max_dd_pct']:>5.1f}%  ΔN={dn:+d}"
        )
    print("─" * 110)

    # Best config
    best = max(key_results, key=lambda x: x["N"] if x["PF"] >= 1.50 else 0)
    goal = best["N"] >= TARGET_N and best["PF"] >= 1.55
    print(f"\n{'🎯 GOAL MET' if goal else '⚠️  GOAL NOT MET'}: Best={best['label']}  N={best['N']}  PF={best['PF']:.3f}")
    print(
        f"Target: N≥{TARGET_N}, PF≥1.55  "
        f"{'✅ N OK' if best['N'] >= TARGET_N else '❌ N short'}  "
        f"{'✅ PF OK' if best['PF'] >= 1.55 else '❌ PF low'}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Write Report
    # ─────────────────────────────────────────────────────────────────────────
    today = datetime.now().strftime("%Y%m%d")
    yyyymm = datetime.now().strftime("%Y%m")
    report_dir = REPORT_DIR / yyyymm
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r58_n_expansion.md"

    def yearly_md(m: dict) -> str:
        if "yearly" not in m:
            return "*yearly breakdown not computed*"
        rows = [
            "| Year | N | PF | WR | EV_bp |",
            "|------|---|----|----|-------|",
        ]
        for yr, ym in sorted(m["yearly"].items()):
            if 2020 <= yr <= 2026:
                ok = "✅" if ym["PF"] >= 1.0 else "❌"
                rows.append(f"| {yr} {ok} | {ym['N']} | {ym['PF']:.3f} | {ym['WR']:.1f}% | {ym['EV_bp']:+.0f} |")
        return "\n".join(rows)

    report_content = f"""# r58: N Expansion v2 — Universe + VR Threshold

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Commit:** `{commit}`
**Script:** `research/active/r51_r60/r58_n_expansion_v2.py`
**구현 여부:** 미구현

---

## 연구 목표

A0 (G_4TF, 60-sym, N=7,119) 기준 **N +20%** 달성  
Target N ≥ {TARGET_N} | PF ≥ 1.55 | 모든 연도(2020-2025) 수익 | 2023+ ≥ 이전

---

## Parity Notes

- Config: `VolStraddleConfig()` 직접 읽음 ✅
- Lookahead 없음 ✅ (`_offset` 적용으로 신규 심볼 시간 정렬)
- Universe: 60 whitelist (5yr+) + 11 extra (3yr+) = {len(data_15m_ext)} sym
- 비용: {COST_DIRECTIONAL * 10000:.0f}bp round-trip (D), {COST_STRADDLE * 10000:.0f}bp (S)
- 모델: 1x compound (compute_metrics 기본값)

---

## Phase 0: 60-sym Baseline (G_3TF)

| | N | PF | EV | WR | DD |
|-|---|----|----|----|----|
| Baseline | {m_bl["N"]} | {m_bl["PF"]:.3f} | {m_bl["EV_bp"]:+.1f}bp | {m_bl["WR"]:.1f}% | {m_bl["max_dd_pct"]:.1f}% |

**Parity:** N±{abs(m_bl["N"] - BL_N_EXPECTED) / BL_N_EXPECTED * 100:.1f}%  PF±{abs(m_bl["PF"] - BL_PF_EXPECTED) / BL_PF_EXPECTED * 100:.1f}%

---

## A0 Reference (60-sym G_4TF = r57 A0)

| | N | PF | WR | Calmar | DD |
|-|---|----|----|--------|----|
| A0-ref | {m_a0_ref["N"]} | {m_a0_ref["PF"]:.3f} | {m_a0_ref["WR"]:.1f}% | {m_a0_ref["calmar"]:.1f} | {m_a0_ref["max_dd_pct"]:.1f}% |

---

## Phase A: Universe Expansion

Extra symbols: {", ".join(EXTRA_SYMS_3YR)}

| Config | N | ΔN | PF | WR | Calmar | DD |
|--------|---|----|----|----|----|-----|
| A1 G_3TF ext | {m_a1["N"]} | {m_a1["N"] - m_a0_ref["N"]:+d} | {m_a1["PF"]:.3f} | {m_a1["WR"]:.1f}% | {m_a1["calmar"]:.1f} | {m_a1["max_dd_pct"]:.1f}% |
| A2 G_4TF ext | {m_a2["N"]} | {m_a2["N"] - m_a0_ref["N"]:+d} | {m_a2["PF"]:.3f} | {m_a2["WR"]:.1f}% | {m_a2["calmar"]:.1f} | {m_a2["max_dd_pct"]:.1f}% |

### A2 Yearly Breakdown

{yearly_md(m_a2)}

---

## Phase B: VR Threshold Sweep (60-sym G_4TF)

### B1: d_vr_threshold
| d_vr | N | ΔN | PF | WR |
|------|---|----|----|-----|
"""
    for m in b1_results:
        dvr = float(m["label"].split("=")[1])
        report_content += f"| {dvr:.2f} | {m['N']} | {m['N'] - m_a0_ref['N']:+d} | {m['PF']:.3f} | {m['WR']:.1f}% |\n"

    report_content += f"""
### B2: s_vr_threshold
| s_vr | N | ΔN | PF | WR |
|------|---|----|----|-----|
"""
    for m in b2_results:
        svr = float(m["label"].split("=")[1])
        report_content += f"| {svr:.2f} | {m['N']} | {m['N'] - m_a0_ref['N']:+d} | {m['PF']:.3f} | {m['WR']:.1f}% |\n"

    report_content += f"""
### B3: s_squeeze_dur
| s_dur | N | ΔN | PF | WR |
|-------|---|----|----|-----|
"""
    for m in b3_results:
        sdur = int(m["label"].split("=")[1])
        report_content += f"| {sdur} | {m['N']} | {m['N'] - m_a0_ref['N']:+d} | {m['PF']:.3f} | {m['WR']:.1f}% |\n"

    report_content += f"""
---

## Phase C: Combined A + B

| Config | N | ΔN from A0 | PF | WR | Calmar | DD | 모든해수익 |
|--------|---|-----------|----|----|--------|----|-----------|
| C1: ext+d_vr={best_dvr:.2f} | {m_c1["N"]} | {m_c1["N"] - m_a0_ref["N"]:+d} | {m_c1["PF"]:.3f} | {m_c1["WR"]:.1f}% | {m_c1["calmar"]:.1f} | {m_c1["max_dd_pct"]:.1f}% | {"✅" if "yearly" in m_c1 and all_years_positive(m_c1["yearly"]) else "❌"} |
| C2: ext+s_vr={best_svr:.2f}/dur={best_sdur} | {m_c2["N"]} | {m_c2["N"] - m_a0_ref["N"]:+d} | {m_c2["PF"]:.3f} | {m_c2["WR"]:.1f}% | {m_c2["calmar"]:.1f} | {m_c2["max_dd_pct"]:.1f}% | {"✅" if "yearly" in m_c2 and all_years_positive(m_c2["yearly"]) else "❌"} |
| C3: Full combo | {m_c3["N"]} | {m_c3["N"] - m_a0_ref["N"]:+d} | {m_c3["PF"]:.3f} | {m_c3["WR"]:.1f}% | {m_c3["calmar"]:.1f} | {m_c3["max_dd_pct"]:.1f}% | {"✅" if "yearly" in m_c3 and all_years_positive(m_c3["yearly"]) else "❌"} |

### C3 (Full Combo) Yearly Breakdown

{yearly_md(m_c3)}

---

## 결론

- **목표 N ≥ {TARGET_N}:** {"달성 ✅" if best["N"] >= TARGET_N else "미달성 ❌"}
- **Best 설정:** {best["label"]}
- **N={best["N"]}  PF={best["PF"]:.3f}  WR={best["WR"]:.1f}%  Calmar={best["calmar"]:.1f}**

{"### 주요 발견" if goal else "### 추가 연구 방향"}
{"- Universe 확장으로 N 증가 확인" if m_a2["N"] > m_a0_ref["N"] else "- Universe 확장 효과 제한적"}
{"- VR threshold 완화로 추가 N 확보" if best_dvr > D_VR or best_svr > S_VR else "- VR threshold 완화 효과 없음 — 현재값 최적"}
"""

    report_path.write_text(report_content, encoding="utf-8")
    print(f"\n✅ Report saved: {report_path}")


if __name__ == "__main__":
    main()
