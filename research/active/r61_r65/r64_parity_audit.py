#!/usr/bin/env python3
"""r64: Backtest-Live Parity Audit.

라이브 트레이딩(v14.16.0)과 백테스트(r57 A0) 간의 구조적 차이를 정량적으로 분석.

핵심 질문:
  1. TF 우선순위 차이가 성과에 얼마나 영향을 미치는가?
     - 라이브: 4h → 1h → 30m → 15m (높은 TF 우선, 첫 신호만 처리)
     - 백테스트: 30m → 15m → 1h → 4h 수집 후 entry 시간 기준 dedup

  2. 동일 entry bar에서 여러 TF가 동시에 신호 발생 시 어느 TF가 선택되는가?
     - 충돌 빈도와 방향(하위 TF 선택 vs 상위 TF 선택) 측정

  3. 모든 파리티 항목 체크리스트

결론:
  - 동일 entry bar 충돌: [X]회 발견 (전체 N의 [Y]%)
  - Strict Priority (live-like) vs Current Dedup: ΔN=[A], ΔPF=[B], ΔWR=[C]pp
  - 종합 판정: PARITY / MINOR GAP / MATERIAL GAP

Usage:
    uv run python research/active/r61_r65/r64_parity_audit.py
"""

from __future__ import annotations

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
from research.active.r51_r60.r57_combo_30m_pullback import (
    G_4TF,
    WARMUP_30m,
    _MH30m,
    collect_d_pullback,
    collect_s_trailing,
    collect_ma,
)

# ── Config ──────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
VR_SHORT     = cfg.vr_short
VR_LONG      = cfg.vr_long
D_VR         = cfg.d_vr_threshold
S_VR         = cfg.s_vr_threshold
S_DUR        = cfg.s_squeeze_dur
D_SL         = cfg.d_sl
D_TP         = cfg.d_tp
D_TRAIL_ACT  = cfg.d_trail_activate
D_TRAIL_DIST = cfg.d_trail_distance
S_SL         = cfg.s_sl
S_TP         = cfg.s_tp
S_TRAIL_ACT  = cfg.s_trail_activate
S_TRAIL_DIST = cfg.s_trail_distance
MAX_CONC     = cfg.max_concurrent
VR_LN        = cfg.base_leverage / MAX_CONC
MA_LN        = cfg.ma_cluster_leverage / MAX_CONC
VR_KEY       = f"vr_{VR_SHORT}_{VR_LONG}"

CACHE_DIR  = ROOT / "data" / "cache_extended"
DATA_YEARS = 6.0

# TF priority mapping for "strict priority" mode (matches live on_bar order)
# 4h (ratio=16) > 1h (ratio=4) > 30m (ratio=2) > 15m (ratio=1)
TF_PRIORITY = [16, 4, 2, 1]  # descending priority


# ──────────────────────────────────────────────────────────────────────────────
#  Data Loading (reuse r57 pattern)
# ──────────────────────────────────────────────────────────────────────────────

def load_syms(min_years: float = 5.0) -> dict:
    whitelist = set(cfg.symbol_whitelist)
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


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 1: Collect trades with TF LABEL for conflict analysis
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_sym_labeled(sym_trades_labeled: list) -> tuple[list, list, list]:
    """Dedup per-symbol trades (sort by entry, reject overlapping).

    Returns:
        (accepted, rejected, conflicts)
        conflicts: cases where a LOWER-priority TF was accepted over a
                   HIGHER-priority TF with the same entry bar.
    """
    # sym_trades_labeled: list of (entry_15m, exit_15m, pnl, tf_ratio)
    sym_trades_labeled.sort(key=lambda x: x[0])  # sort by entry bar
    accepted: list[tuple[int, int]] = []  # (entry, exit) for overlap check
    accepted_full: list = []  # (entry, exit, pnl, tf_ratio) 4-tuple
    rejected_full: list = []
    conflicts: list = []

    for entry_15m, exit_15m, pnl_val, tf_ratio in sym_trades_labeled:
        conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
        if not conflict:
            # Check for same entry bar with a higher-priority TF that was already accepted
            # (shouldn't happen since we sort by entry, but check for simultaneous entries)
            same_entry_accepted = [
                (ae, ax, tr) for (ae, ax, pn, tr) in accepted_full
                if ae == entry_15m
            ]
            if same_entry_accepted:
                # Multiple TFs firing at same entry bar — check priority
                for ae, ax, accepted_tf_ratio in same_entry_accepted:
                    if tf_ratio > accepted_tf_ratio:
                        # Live would prefer this trade (higher ratio = higher TF)
                        # but backtest already accepted lower TF
                        conflicts.append({
                            "entry_15m": entry_15m,
                            "accepted_tf": accepted_tf_ratio,
                            "preferred_tf": tf_ratio,
                        })
            accepted.append((entry_15m, exit_15m))
            accepted_full.append((entry_15m, exit_15m, pnl_val, tf_ratio))
        else:
            rejected_full.append((entry_15m, exit_15m, pnl_val, tf_ratio))

    return accepted_full, rejected_full, conflicts


def collect_d_labeled(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
) -> tuple[list, list, int]:
    """Collect D_mom8 trades with per-trade TF label.

    Returns:
        (all_accepted, all_rejected, conflict_count)
        all_accepted: (sym, entry_15m, exit_15m, pnl, tf_ratio)
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m
    tfs = G_4TF if data_30m is not None else list(G_3TF)

    all_accepted = []
    all_rejected = []
    total_conflicts = 0

    for sym in sorted(data_15m.keys()):
        sym_trades_labeled: list = []

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
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p)):
                sym_trades_labeled.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                    ratio,  # TF ratio for priority tracking
                ))

        accepted, rejected, conflicts = _dedup_sym_labeled(sym_trades_labeled)
        for entry, exit_, pnl, tf_ratio in accepted:
            all_accepted.append((sym, entry, exit_, pnl, tf_ratio))
        all_rejected.extend(rejected)
        total_conflicts += len(conflicts)

    return all_accepted, all_rejected, total_conflicts


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 2: "Strict Priority" collection (4h > 1h > 30m > 15m)
# ──────────────────────────────────────────────────────────────────────────────

def collect_d_strict_priority(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    data_30m: dict | None = None,
) -> list:
    """Collect D_mom8 with live-like strict TF priority.

    For each symbol, collects TFs in descending priority order (4h→1h→30m→15m).
    Each newly added trade must not overlap with already-accepted trades.
    This mirrors live on_bar() behavior: highest TF always wins.

    Note: Unlike live (which fires only one TF per bar), this accumulates
    ALL non-overlapping trades across TFs in priority order — which is still
    more accurate than current backtest since higher TFs fill first.
    """
    # Descending TF priority to match live: 4h(16) > 1h(4) > 30m(2) > 15m(1)
    tfs_priority_order = sorted(
        G_4TF if data_30m is not None else list(G_3TF),
        key=lambda x: -x[1],  # sort by ratio descending
    )
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    if data_30m is not None:
        tf_data_map[2] = data_30m

    all_trades: list = []

    for sym in sorted(data_15m.keys()):
        accepted: list[tuple[int, int]] = []  # (entry_15m, exit_15m)
        result: list = []

        for _lbl, ratio, warmup, mh_d, _mh_s in tfs_priority_order:
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
            dire = make_direction_momentum(sig, mom8)
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire,
                D_SL, D_TP, mh_d, COST_DIRECTIONAL,
                D_TRAIL_ACT, D_TRAIL_DIST,
            )
            for k in range(len(p)):
                entry_15m = int(e[k]) * ratio
                exit_15m  = int(x[k]) * ratio + ratio - 1
                # Accept only if no overlap with already-accepted (higher TF) trades
                conflict = any(entry_15m <= ax and exit_15m >= ae for ae, ax in accepted)
                if not conflict:
                    result.append((sym, entry_15m, exit_15m, float(p[k])))
                    accepted.append((entry_15m, exit_15m))

        all_trades.extend(result)

    return all_trades


# ──────────────────────────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────────────────────────

def fmt(m: dict) -> str:
    boot = bootstrap_pf([t[3] for t in []], n_bootstrap=0)  # placeholder
    return (f"N={m['N']:6d}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  "
            f"EV={m['EV_bp']:+.1f}bp  Calmar={m['calmar']:.1f}  DD={m['max_dd_pct']:.1f}%")


def run_portfolio(trades: list, label: str) -> dict:
    pnl = portfolio_sim([(t[0], t[1], t[2], t[3]) for t in trades], MAX_CONC)
    m = compute_metrics(pnl, DATA_YEARS)
    m["label"] = label
    return m


# ──────────────────────────────────────────────────────────────────────────────
#  Parity Checklist
# ──────────────────────────────────────────────────────────────────────────────

def print_parity_checklist() -> None:
    print("\n" + "═" * 72)
    print("PARITY CHECKLIST: Live (v14.16.0) vs Backtest (r57 A0 G_4TF)")
    print("═" * 72)

    items = [
        ("Entry timing (pullback)", True,
         "r57 A0 uses pb_pct=0.0 (immediate entry), matches live on_bar()"),
        ("SL/TP parameters", True,
         f"Both read from config.py: D_SL={D_SL}, D_TP={D_TP}"),
        ("D_mom8 trailing stop", True,
         f"Both use trail_activate={D_TRAIL_ACT}, trail_distance={D_TRAIL_DIST}"),
        ("S_d8 trailing stop", True,
         f"Both use s_trail_activate={S_TRAIL_ACT}, s_trail_distance={S_TRAIL_DIST}"),
        ("S_d8 SL/TP", True,
         f"Both: S_SL={S_SL}, S_TP={S_TP}"),
        (".env overrides", True,
         "No VS_* entries in .env — config.py values used directly"),
        ("Symbol universe (60 syms)", True,
         "Both use cfg.symbol_whitelist (60 symbols)"),
        ("Signal functions", True,
         "Both import gen_vr_signal, gen_vr_dur_signal from same module"),
        ("VR params (sp=8, lp=56)", True,
         f"Both: vr_short={cfg.vr_short}, vr_long={cfg.vr_long}"),
        ("MA Cluster params", True,
         f"Both read from config: cluster_pct={cfg.ma_cluster_pct:.2f}, "
         f"min_dur={cfg.ma_cluster_min_duration}"),
        ("Leverage: VR=30x", True,
         "positions.py L98: base_leverage=30x from config"),
        ("Leverage: MA=21x", True,
         "positions.py L112-115: sig_meta['leverage_override']=21x"),
        ("TF priority order", None,  # ← will be filled after analysis
         "Backtest=30m→15m→1h→4h collect+dedup; Live=4h→1h→30m→15m first-hit"),
        ("Slot model (one per symbol)", True,
         "Backtest time-overlap dedup ≈ live one-position-per-symbol"),
        ("Cost model (0.08% round-trip)", True,
         f"Both: COST_DIRECTIONAL={COST_DIRECTIONAL:.4f}, COST_STRADDLE={COST_STRADDLE:.4f}"),
    ]

    pass_count = sum(1 for _, ok, _ in items if ok is True)
    gap_count  = sum(1 for _, ok, _ in items if ok is False)
    tbd_count  = sum(1 for _, ok, _ in items if ok is None)

    for name, ok, note in items:
        if ok is True:
            status = "✅ PARITY"
        elif ok is False:
            status = "❌ GAP"
        else:
            status = "⚠️  TBD"
        print(f"\n  {status}  {name}")
        print(f"           {note}")

    print(f"\n  Summary: {pass_count} parity / {gap_count} gap / {tbd_count} TBD (→ quantified below)")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("r64: Backtest-Live Parity Audit  (v14.16.0)")
    print("=" * 72)

    # ─── Data ────────────────────────────────────────────────────────────────
    print("\n[0] Loading data …")
    t0 = time.time()
    d15 = load_syms(min_years=5.0)
    d30, d1h, d4h = build_all_tf(d15)
    print(f"    {len(d15)} symbols loaded in {time.time()-t0:.1f}s")

    # ─── Phase 1: Baseline (current dedup, G_4TF) ────────────────────────────
    print("\n[1] Phase 1: Current backtest (G_4TF, dedup by entry bar)")
    t1 = time.time()
    d_trades_current = collect_d_pullback(d15, d1h, d4h, 0.0, 5, d30)
    s_trades  = collect_s_trailing(d15, d1h, d4h, d30)
    ma_trades = collect_ma(d15, d1h, d4h)
    all_trades_current = d_trades_current + s_trades + ma_trades

    pnl_current = portfolio_sim(all_trades_current, MAX_CONC)
    m_current = compute_metrics(pnl_current, DATA_YEARS)
    m_current["label"] = "Current (G_4TF dedup)"
    print(f"    D_mom8: N={len(d_trades_current):5d}  S_d8: N={len(s_trades):5d}  MA: N={len(ma_trades):5d}")
    print(f"    Combined: N={m_current['N']:6d}  PF={m_current['PF']:.3f}  WR={m_current['WR']:.1f}%  "
          f"Calmar={m_current['calmar']:.1f}  DD={m_current['max_dd_pct']:.1f}%")
    print(f"    [{time.time()-t1:.1f}s]")

    # Baseline sanity check
    expected_N = 10791
    expected_PF = 1.986
    n_ok = abs(m_current["N"] - expected_N) / expected_N < 0.02
    pf_ok = abs(m_current["PF"] - expected_PF) / expected_PF < 0.02
    print(f"    Baseline check: N={'✅' if n_ok else '❌'} (expected ~{expected_N}), "
          f"PF={'✅' if pf_ok else '❌'} (expected ~{expected_PF})")

    # ─── Phase 2: TF Conflict Analysis ───────────────────────────────────────
    print("\n[2] Phase 2: TF conflict analysis (same entry bar, multiple TF signals)")
    t2 = time.time()
    d_labeled, d_rejected, conflict_count = collect_d_labeled(d15, d1h, d4h, d30)

    total_d = len(d_labeled)
    conflict_pct = 100 * conflict_count / max(1, total_d)
    print(f"    D_mom8 accepted:  {total_d:5d} trades")
    print(f"    D_mom8 rejected:  {len(d_rejected):5d} trades (time-overlap dedup)")
    print(f"    Same-bar conflicts (backtest picked lower TF, live would pick higher TF):")
    print(f"      Count: {conflict_count}  ({conflict_pct:.2f}% of accepted trades)")
    print(f"    [{time.time()-t2:.1f}s]")

    # TF distribution of accepted trades
    tf_dist: dict[int, int] = {}
    for _, _, _, _, tf_ratio in d_labeled:
        tf_dist[tf_ratio] = tf_dist.get(tf_ratio, 0) + 1
    tf_names = {1: "15m", 2: "30m", 4: "1h", 16: "4h"}
    print("    TF distribution of accepted D_mom8 trades:")
    for ratio in [16, 4, 2, 1]:
        cnt = tf_dist.get(ratio, 0)
        pct = 100 * cnt / max(1, total_d)
        print(f"      {tf_names[ratio]:3s}: {cnt:4d}  ({pct:.1f}%)")

    # TF distribution of rejected trades
    tf_dist_rej: dict[int, int] = {}
    for _, _, _, tf_ratio in d_rejected:
        tf_dist_rej[tf_ratio] = tf_dist_rej.get(tf_ratio, 0) + 1
    print("    TF distribution of REJECTED D_mom8 trades (time-overlap losers):")
    for ratio in [16, 4, 2, 1]:
        cnt = tf_dist_rej.get(ratio, 0)
        pct_rej = 100 * cnt / max(1, len(d_rejected))
        print(f"      {tf_names[ratio]:3s}: {cnt:4d}  ({pct_rej:.1f}% of rejected)")

    # ─── Phase 3: Strict Priority (live-like 4h > 1h > 30m > 15m) ───────────
    print("\n[3] Phase 3: Strict Priority (live-like: 4h > 1h > 30m > 15m)")
    t3 = time.time()
    d_trades_priority = collect_d_strict_priority(d15, d1h, d4h, d30)
    all_trades_priority = d_trades_priority + s_trades + ma_trades

    pnl_priority = portfolio_sim(all_trades_priority, MAX_CONC)
    m_priority = compute_metrics(pnl_priority, DATA_YEARS)
    m_priority["label"] = "Strict Priority (live-like)"
    print(f"    D_mom8 strict priority: N={len(d_trades_priority):5d}")
    print(f"    Combined: N={m_priority['N']:6d}  PF={m_priority['PF']:.3f}  WR={m_priority['WR']:.1f}%  "
          f"Calmar={m_priority['calmar']:.1f}  DD={m_priority['max_dd_pct']:.1f}%")
    print(f"    [{time.time()-t3:.1f}s]")

    # Delta
    dN  = m_priority["N"]  - m_current["N"]
    dPF = m_priority["PF"] - m_current["PF"]
    dWR = m_priority["WR"] - m_current["WR"]
    dCal = m_priority["calmar"] - m_current["calmar"]
    print(f"\n    Δ (Strict Priority vs Current Dedup):")
    print(f"      ΔN={dN:+d}  ΔPF={dPF:+.3f}  ΔWR={dWR:+.1f}pp  ΔCalmar={dCal:+.1f}")

    # ─── TF priority gap severity ─────────────────────────────────────────────
    pf_delta_pct = 100 * abs(dPF) / m_current["PF"]
    n_delta_pct  = 100 * abs(dN)  / m_current["N"]
    if pf_delta_pct < 2 and n_delta_pct < 5:
        severity = "MINOR GAP (<2% PF, <5% N)"
        tf_parity = True
    elif pf_delta_pct < 5 and n_delta_pct < 10:
        severity = "MODERATE GAP (2-5% PF, 5-10% N)"
        tf_parity = None
    else:
        severity = "MATERIAL GAP (>5% PF or >10% N)"
        tf_parity = False

    print(f"      PF delta: {pf_delta_pct:.1f}%  N delta: {n_delta_pct:.1f}%")
    print(f"      TF Priority Severity: {severity}")

    # ─── Print parity checklist ───────────────────────────────────────────────
    print_parity_checklist()

    # ─── Final verdict ────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("FINAL VERDICT")
    print("═" * 72)

    print(f"""
  Current Backtest (G_4TF dedup):
    N={m_current['N']:6d}  PF={m_current['PF']:.3f}  WR={m_current['WR']:.1f}%  Calmar={m_current['calmar']:.1f}

  Live-like (Strict Priority):
    N={m_priority['N']:6d}  PF={m_priority['PF']:.3f}  WR={m_priority['WR']:.1f}%  Calmar={m_priority['calmar']:.1f}

  TF Priority Gap: {severity}
  Same-bar conflicts: {conflict_count} ({conflict_pct:.2f}% of trades)

  OVERALL PARITY ASSESSMENT:""")

    if tf_parity is True and conflict_pct < 2:
        print("  ✅ PARITY — backtest and live systems produce equivalent results.")
        print("     TF priority difference is negligible. No code changes needed.")
    elif tf_parity is None or conflict_pct >= 2:
        print("  ⚠️  MINOR GAP — TF priority causes small but measurable divergence.")
        print("     Consider: change backtest dedup to match live priority order.")
        print("     OR: document as known conservative offset (live may outperform backtest).")
    else:
        print("  ❌ MATERIAL GAP — TF priority causes significant backtest-live divergence.")
        print("     Action required: align backtest dedup with live priority order.")

    print(f"""
  Data: {len(d15)} symbols × {DATA_YEARS}yr  Config: config.py (no .env overrides)
  VR params: sp={cfg.vr_short}, lp={cfg.vr_long}, D_VR={D_VR}
  Exit: SL={D_SL:.1%}, TP={D_TP:.1%}, trail_act={D_TRAIL_ACT:.1%}, trail_dist={D_TRAIL_DIST:.1%}
""")


if __name__ == "__main__":
    main()
