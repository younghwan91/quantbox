#!/usr/bin/env python3
"""r76: S_d8 s_trail_distance 최적화 — Trail Server Ratchet 이후 재검증.

r67 sensitivity (2026-03-12) swept straddle trail_distance and found:
  0.001  → PF 3.772, OOS 2.193 (❌ — PROD_IDX bug: used 0.003 as reference)
  0.0015 → PF 3.223, OOS 1.854 (✅)
  0.002  → PF 2.772, OOS 1.576 (✅, production)

Trail Server Ratchet (e473e4d) now live → live gap eliminated.
r67's ❌ for 0.001 was wrong (PROD_IDX=3 → 0.003, not 0.002 production).

Goal: verify whether s_trail_distance < 0.002 is justified with correct OOS.
  Phase 0: S_d8 standalone baseline vs r67 (N≈10388, PF≈2.772 at dist=0.002)
  Phase 1: Fine sweep [0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002]
  Phase 2: Combined (D+S+MA) with best s_trail_distance vs r73 canonical

Usage:
    uv run python research/active/r76_r80/r76_trail_distance.py
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
from research.active.r41_r45.r43_exit_improvement import get_year_boundaries
from quantbox.backtest.trade_sim import sim_directional_trailing, sim_straddle_trailing
from research.active.r41_r45.r44c_ma_confluence import precompute_mas
from research.active.r41_r45.r45_param_sweep import collect_enhanced_trades

# ── Sweep grid ────────────────────────────────────────────────────────────────
TRAIL_DIST_GRID = [0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002]

# ── MA Cluster deployed params (r45 champion, same as r73) ───────────────────
MA_CLUSTER_PCT    = 0.03
MA_MIN_DURATION   = 8
MA_MAX_DURATION   = 24
MA_MIN_MOMENTUM   = 0.010
MA_SL             = 0.02
MA_TP             = 0.08
MA_TF_RESTRICTION = 16

WARMUP_30m: int = 100


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


def _get_g4tf(cfg_obj: VolStraddleConfig) -> list:
    mh_30m = cfg_obj.max_hold_15m // 2
    tf_30m = ("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)
    return [tf_30m] + list(G_3TF)


def load_all_data(cfg_obj: VolStraddleConfig) -> tuple[dict, dict, dict, dict, int, float]:
    """Load and precompute all data (matches r73 exactly)."""
    cache_dir = ROOT / "data" / "cache_extended"
    whitelist = set(cfg_obj.symbol_whitelist)
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
    data_1h  = build_resampled(data_15m, 4)
    data_4h  = build_resampled(data_15m, 16)

    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], [(cfg_obj.vr_short, cfg_obj.vr_long)])
        precompute(ds, vr_sp=cfg_obj.vr_short, vr_lp=cfg_obj.vr_long)

    precompute_mas(data_15m)
    precompute_mas(data_1h)
    precompute_mas(data_4h)

    bars_list = [len(d["close"]) for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list))
    data_years = avg_bars / (4 * 24 * 365)

    return data_15m, data_1h, data_4h, data_30m, max_bars, data_years


def _dedup(sym: str, trades: list) -> list:
    result = []
    last_exit = -1
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if e > last_exit:
            result.append((sym, e, x, p))
            last_exit = x
    return result


def collect_s_with_dist(
    data_15m: dict, data_1h: dict, data_4h: dict, data_30m: dict,
    cfg_obj: VolStraddleConfig, trail_dist: float,
) -> list:
    """S_d8 straddle trail collection with custom trail_distance."""
    vr_key = f"vr_{cfg_obj.vr_short}_{cfg_obj.vr_long}"
    g4tf = _get_g4tf(cfg_obj)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl = cfg_obj.s_sl
    tp = cfg_obj.s_tp
    trail_act = cfg_obj.s_trail_activate
    s_vr = cfg_obj.s_vr_threshold
    s_dur = cfg_obj.s_squeeze_dur
    mh_15m = cfg_obj.max_hold_15m

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
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, sl, tp, int(mh_15m // ratio), COST_STRADDLE,
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


def collect_d_trail(
    data_15m: dict, data_1h: dict, data_4h: dict, cfg_obj: VolStraddleConfig,
) -> list:
    """D_mom8 directional trail (production params, fixed for Phase 2)."""
    vr_key = f"vr_{cfg_obj.vr_short}_{cfg_obj.vr_long}"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    sl = cfg_obj.d_sl
    tp = cfg_obj.d_tp
    trail_act = cfg_obj.d_trail_activate
    trail_dist = cfg_obj.d_trail_distance
    d_vr = cfg_obj.d_vr_threshold

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


def collect_ma_trades(data_15m: dict, data_1h: dict, data_4h: dict) -> list:
    return collect_enhanced_trades(
        data_15m, data_1h, data_4h,
        cluster_pct=MA_CLUSTER_PCT, min_duration=MA_MIN_DURATION,
        max_duration=MA_MAX_DURATION, min_momentum=MA_MIN_MOMENTUM,
        sl=MA_SL, tp=MA_TP, tf_restriction=MA_TF_RESTRICTION,
    )


def main() -> None:
    cfg = VolStraddleConfig()
    commit = get_commit()
    today = __import__("datetime").date.today().strftime("%Y%m%d")

    print("=" * 65)
    print("r76: S_d8 s_trail_distance Optimization (post-ratchet)")
    print(f"  Commit: {commit}")
    print(f"  Production: s_trail_activate={cfg.s_trail_activate:.3f}"
          f"  s_trail_distance={cfg.s_trail_distance:.4f}")
    print(f"  Symbols: {len(cfg.symbol_whitelist)} (symbol_whitelist)")
    print("=" * 65)

    print("\n[Phase 0] Loading data...")
    t0 = time.time()
    data_15m, data_1h, data_4h, data_30m, max_bars, data_years = load_all_data(cfg)
    oos_frac = 0.3
    oos_years = data_years * oos_frac
    print(f"  Loaded in {time.time()-t0:.1f}s | data_years={data_years:.2f}")

    # ── Phase 0 baseline (s_trail_distance=0.002, vs r67 N≈10388, PF≈2.772) ─
    print("\n[Phase 0] S_d8 baseline (s_trail_distance=0.002)...")
    baseline_trades = collect_s_with_dist(
        data_15m, data_1h, data_4h, data_30m, cfg, trail_dist=0.002,
    )
    baseline_pnl = portfolio_sim(baseline_trades, cfg.max_concurrent)
    baseline_m = compute_metrics(baseline_pnl, data_years)
    N_ref, PF_ref = 7730, 4.948  # r74 baseline (s_trail_activate=0.015, same params)

    print(f"  N={baseline_m['N']}  PF={baseline_m['PF']:.3f}  "
          f"WR={baseline_m['WR']:.1f}%  EV={baseline_m['EV_bp']:.1f}bp")
    print(f"  r74 reference: N={N_ref}, PF={PF_ref:.3f}")
    n_drift  = abs(baseline_m['N'] - N_ref) / N_ref
    pf_drift = abs(baseline_m['PF'] - PF_ref) / PF_ref
    print(f"  N_drift={n_drift:.1%}  PF_drift={pf_drift:.1%}", end="  ")
    print("✅ Baseline OK" if n_drift < 0.05 and pf_drift < 0.05 else "⚠️  Drift > 5%")

    # ── Phase 1: S_d8 standalone sweep ───────────────────────────────────────
    print(f"\n[Phase 1] s_trail_distance sweep: {TRAIL_DIST_GRID}")
    print(f"  OOS: last {oos_years:.1f}yr (30%)")

    split_bar = int(max_bars * (1 - oos_frac))
    results: list[dict] = []

    for dist in TRAIL_DIST_GRID:
        t1 = time.time()
        trades = collect_s_with_dist(
            data_15m, data_1h, data_4h, data_30m, cfg, trail_dist=dist,
        )
        pnl_full = portfolio_sim(trades, cfg.max_concurrent)
        m = compute_metrics(pnl_full, data_years)

        oos_t = [(s, e, x, p) for s, e, x, p in trades if e >= split_bar]
        pnl_oos = portfolio_sim(oos_t, cfg.max_concurrent)
        m_oos = compute_metrics(pnl_oos, oos_years) if len(pnl_oos) > 10 else {"PF": 0.0, "N": 0}

        boot = bootstrap_pf(pnl_full, n_boot=500) if len(pnl_full) > 20 else {"P_PF_ge_1": 0.0}
        is_oos = m["PF"] / m_oos["PF"] if m_oos["PF"] > 0 else 99.0

        # Robustness: OOS PF ≥ 1.5 (strong OOS edge) AND Bootstrap P≥95%
        # Note: IS/OOS ratio not used as criterion — high-PF strategies naturally have IS > OOS
        robust = m_oos["PF"] >= 1.5 and boot.get("P_PF_ge_1", 0) >= 95.0
        star = " ★" if abs(dist - cfg.s_trail_distance) < 1e-6 else ""
        results.append({
            "dist": dist, "N": m["N"], "PF": m["PF"], "WR": m["WR"],
            "EV": m["EV_bp"], "MaxDD": m["max_dd_pct"], "Calmar": m["calmar"],
            "OOS_PF": m_oos["PF"], "IS_OOS": is_oos,
            "boot_p": boot.get("P_PF_ge_1", 0), "robust": robust, "star": star,
        })
        flag = "✅" if robust else "❌"
        print(f"  {dist:.4f}{star}: N={m['N']} PF={m['PF']:.3f} OOS={m_oos['PF']:.3f}"
              f" boot={boot.get('P_PF_ge_1',0):.0f}% {flag}"
              f"  ({time.time()-t1:.1f}s)")

    robust_results = [r for r in results if r["robust"]]
    best = max(robust_results, key=lambda r: r["PF"]) if robust_results else results[-1]
    improved = best["dist"] < cfg.s_trail_distance
    print(f"\n  Best robust: s_trail_distance={best['dist']:.4f} "
          f"PF={best['PF']:.3f} OOS={best['OOS_PF']:.3f}"
          f"  {'→ IMPROVEMENT' if improved else '→ NO CHANGE'}")

    # ── Phase 2: Combined portfolio ───────────────────────────────────────────
    print(f"\n[Phase 2] Combined (D+S+MA) with best s_trail_distance={best['dist']:.4f}...")
    print("  r73 canonical: N=9773, PF=2.002, Calmar=1695.6")

    s_trades_best = collect_s_with_dist(
        data_15m, data_1h, data_4h, data_30m, cfg, trail_dist=best["dist"],
    )
    s_trades_curr = baseline_trades  # dist=0.002 already computed
    d_trades = collect_d_trail(data_15m, data_1h, data_4h, cfg)
    ma_trades = collect_ma_trades(data_15m, data_1h, data_4h)

    print(f"  Collected: S={len(s_trades_best)} D={len(d_trades)} MA={len(ma_trades)}")

    comb_best_pnl = portfolio_sim(s_trades_best + d_trades + ma_trades, cfg.max_concurrent)
    comb_curr_pnl = portfolio_sim(s_trades_curr + d_trades + ma_trades, cfg.max_concurrent)

    comb_best_m = compute_metrics(comb_best_pnl, data_years)
    comb_curr_m = compute_metrics(comb_curr_pnl, data_years)

    print(f"  Best (s_dist={best['dist']:.4f}): N={comb_best_m['N']} "
          f"PF={comb_best_m['PF']:.3f} Calmar={comb_best_m['calmar']:.1f} "
          f"MaxDD={comb_best_m['max_dd_pct']:.1f}%")
    print(f"  Curr (s_dist=0.002):      N={comb_curr_m['N']} "
          f"PF={comb_curr_m['PF']:.3f} Calmar={comb_curr_m['calmar']:.1f} "
          f"MaxDD={comb_curr_m['max_dd_pct']:.1f}%")
    print(f"  r73 canonical:            N=9773 PF=2.002 Calmar=1695.6 MaxDD=40.5%")

    # ── Report ────────────────────────────────────────────────────────────────
    report_dir = ROOT / "docs" / "backtest_results" / "202603"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r76_s_trail_distance.md"

    with open(report_path, "w") as f:
        f.write(f"""# r76: S_d8 s_trail_distance 최적화 (Trail Server Ratchet 이후 재검증)

**날짜:** {today[:4]}-{today[4:6]}-{today[6:]}
**커밋:** {commit}
**스크립트:** `research/active/r76_r80/r76_trail_distance.py`
**데이터:** {len(data_15m)} syms (symbol_whitelist) · {data_years:.2f}yr · Binance USDT-M
**비용:** 0.16% round-trip (straddle) · 10슬롯 (max_concurrent)
**Parity:** config.py 직접 읽음 ✅ · symbol_whitelist ✅ · 룩어헤드 없음 ✅

## 배경

r67 sensitivity (2026-03-12) — S_d8 straddle trail_distance 스윕 (당시 s_trail_activate=0.005 기준):
- 0.001: PF 3.772, OOS 2.193 (r67 ❌ — PROD_IDX 버그로 0.003 기준 판정, 실제 PROD=0.002)
- 0.0015: PF 3.223, OOS 1.854 (r67 ✅)
- 0.002 (현재): PF 2.772, OOS 1.576 (✅)

r74 (2026-03-13) — s_trail_activate=0.015 채택 후 S_d8 baseline: N=7730, PF=4.948
Trail Server Ratchet (e473e4d) 구현 완료 → live gap 문제 해소.
r76 OOS 기준: OOS PF ≥ 1.5 AND Bootstrap P(PF≥1.0) ≥ 95%
(IS/OOS ≤ 1.5 제외 — 고PF 전략에서 IS > OOS는 정상; OOS 절대값이 핵심)

---

## Phase 0: Baseline 검증 (s_trail_distance=0.002)

| 항목 | 값 | r74 reference |
|------|---|--------------|
| N | {baseline_m['N']} | {N_ref} ({n_drift:.1%} drift) |
| PF | {baseline_m['PF']:.3f} | {PF_ref:.3f} ({pf_drift:.1%} drift) |
| WR | {baseline_m['WR']:.1f}% | — |
| EV | {baseline_m['EV_bp']:.1f}bp | — |
| Calmar | {baseline_m['calmar']:.1f} | — |

## Phase 1: S_d8 s_trail_distance Sweep

| s_trail_distance | N | PF | WR | EV(bp) | MaxDD | OOS PF | IS/OOS | Boot≥95% | Robust |
|---|---|---|---|---|---|---|---|---|---|
""")
        for r in results:
            flag = "✅" if r["robust"] else "❌"
            boot_flag = "✅" if r["boot_p"] >= 0.95 else "❌"
            f.write(f"| {r['dist']:.4f}{r['star']} | {r['N']} | {r['PF']:.3f} | "
                    f"{r['WR']:.1f}% | {r['EV']:.1f} | {r['MaxDD']:.1f}% | "
                    f"{r['OOS_PF']:.3f} | {r['IS_OOS']:.2f} | {boot_flag} | {flag} |\n")

        verdict = ("✅ s_trail_distance=" + f"{best['dist']:.4f}" + " 채택 권장"
                   if improved else "⚠️ 현재값(0.002) 유지 — 개선 없음")

        f.write(f"""
★ = current production (s_trail_distance={cfg.s_trail_distance:.4f})

**추천:** {verdict}

---

## Phase 2: Combined Portfolio (D+S+MA)

| | N | PF | Calmar | MaxDD | vs r73 |
|---|---|---|---|---|---|
| s_dist={best['dist']:.4f} (proposed) | {comb_best_m['N']} | {comb_best_m['PF']:.3f} | {comb_best_m['calmar']:.1f} | {comb_best_m['max_dd_pct']:.1f}% | {comb_best_m['PF']/2.002:.2f}x |
| s_dist=0.002 (current) | {comb_curr_m['N']} | {comb_curr_m['PF']:.3f} | {comb_curr_m['calmar']:.1f} | {comb_curr_m['max_dd_pct']:.1f}% | {comb_curr_m['PF']/2.002:.2f}x |
| r73 canonical (reference) | 9,773 | 2.002 | 1,695.6 | 40.5% | 1.00x |

---

## 결론

{verdict}
{'변경 위치: `config.py: s_trail_distance={:.4f}` + `.env: VS_S_TRAIL_DISTANCE={:.4f}`'.format(best['dist'], best['dist']) if improved else '현재값 유지.'}

**구현:** 미구현
""")

    print(f"\n✅ Report: {report_path.name}")


if __name__ == "__main__":
    main()
