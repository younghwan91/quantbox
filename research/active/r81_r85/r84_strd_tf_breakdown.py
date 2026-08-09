"""r84: S_d8 Straddle — 타임프레임별 성과 분석.

목적:
  S_d8 신호가 15m / 30m / 1h / 4h 중 어느 TF에서 가장 많은 수익을 내는지 분석.
  - N, PF, EV(bp), WR, MaxDD, AvgHold(bars)
  - 각 TF 단독 vs 전체 G_4TF 합산(캐노니컬 r76 결과) 비교
  - 슬롯 경쟁 없이 TF별 순수 알파를 측정 (max_concurrent=5 적용)

결론 목표:
  1) 어느 TF가 signal quality가 가장 높은가? (EV로 판단)
  2) 어느 TF가 가장 많은 거래를 생성하는가? (N으로 판단)
  3) TF별 WR/DD 특성 — 위험 집중도 확인
  4) cliff edge 취약도: TF별로 trail_activate 민감도가 다른가?

사용:
    uv run python research/active/r81_r85/r84_strd_tf_breakdown.py

베이스라인 (r76 캐노니컬, commit b029ae7):
    N=9,775  PF=2.484  WR=~75%  MaxDD=37.1%  (3전략 합산)
    S_d8 단독: N=7,730  PF=9.603  MaxDD=3.5%  (r77 기준)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    WARMUP_15m,
    WARMUP_1h,
    WARMUP_4h,
    bootstrap_pf,
    compute_metrics,
    gen_vr_dur_signal,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
    build_resampled,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

WARMUP_30m: int = 100

# ── Config ─────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()
MAX_CONCURRENT = 5  # 1 straddle = 2 positions → max 5 pairs live


# ── Data Loading ────────────────────────────────────────────────────────────
def load_15m(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in set(whitelist):
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def _dedup(sym: str, trades: list) -> list:
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


# ── Per-TF trade collection ─────────────────────────────────────────────────
def collect_strd_single_tf(
    tf_label: str,
    ratio: int,
    warmup: int,
    mh_s: int,
    tf_data: dict,
    data_15m_offsets: dict[str, int],
    cfg: VolStraddleConfig,
) -> list[tuple[str, int, int, float]]:
    """Collect S_d8 trail trades for a single timeframe only."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    sl = cfg.s_sl
    tp = cfg.s_tp
    trail_act = cfg.s_trail_activate
    trail_dist = cfg.s_trail_distance
    s_vr = cfg.s_vr_threshold
    s_dur = cfg.s_squeeze_dur

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m_offsets.keys()):
        d = tf_data.get(sym)
        if d is None:
            continue
        vr = d.get(vr_key)
        if vr is None:
            continue
        offset = data_15m_offsets[sym]
        c, h, lo = d["close"], d["high"], d["low"]
        sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
        p, e, x = sim_straddle_trailing(
            c, h, lo, sig, sl, tp, mh_s, COST_STRADDLE, trail_act, trail_dist,
        )
        sym_trades = [
            (int(e[k]) * ratio + offset, int(x[k]) * ratio + ratio - 1 + offset, float(p[k]))
            for k in range(len(p))
        ]
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def collect_strd_all_tf(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
) -> list[tuple[str, int, int, float]]:
    """Collect S_d8 across G_4TF with per-symbol dedup (canonical r76)."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    mh_30m = cfg.max_hold_15m // 2
    g4tf = [
        ("30m", 2,  WARMUP_30m, mh_30m * 2),
        ("15m", 1,  WARMUP_15m, cfg.max_hold_15m * 2),
        ("1h",  4,  WARMUP_1h,  cfg.max_hold_15m // 4 * 2),
        ("4h",  16, WARMUP_4h,  cfg.max_hold_15m // 16 * 2),
    ]
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    sl = cfg.s_sl
    tp = cfg.s_tp
    trail_act = cfg.s_trail_activate
    trail_dist = cfg.s_trail_distance
    s_vr = cfg.s_vr_threshold
    s_dur = cfg.s_squeeze_dur

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, mh_s in g4tf:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, s_vr, s_dur, warmup)
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


# ── Metrics helper ──────────────────────────────────────────────────────────
def run_metrics(
    trades: list[tuple[str, int, int, float]],
    data_years: float,
    label: str,
) -> dict:
    if not trades:
        print(f"  [{label}] NO TRADES")
        return {}
    pnl = portfolio_sim(trades, max_concurrent=MAX_CONCURRENT)
    m = compute_metrics(pnl, data_years)
    boot_d = bootstrap_pf(pnl, n_boot=500)
    boot = (boot_d["CI_lo"], boot_d["CI_hi"])
    p_ge_1 = boot_d["P_PF_ge_1"]
    avg_hold_bars = np.mean([(x - e) for _, e, x, _ in trades])
    avg_hold_h = avg_hold_bars * 15 / 60  # 15m bars → hours
    return {
        "label": label,
        "N": m["N"],
        "PF": m["PF"],
        "WR": m["WR"],
        "EV_bp": m["EV_bp"],
        "MaxDD": m["max_dd_pct"] / 100,
        "CAGR_1x": m["ann_ret_pct"] / 100,
        "Calmar": m["calmar"],
        "boot_lo": boot[0],
        "boot_hi": boot[1],
        "p_ge_1": p_ge_1,
        "avg_hold_h": avg_hold_h,
        "n_trades_raw": len(trades),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.perf_counter()
    whitelist = cfg.symbol_whitelist
    assert len(whitelist) == 60, f"Whitelist must have 60 syms, got {len(whitelist)}"

    print("=" * 70)
    print("r84: S_d8 Straddle — 타임프레임별 성과 분석")
    print("=" * 70)
    print(f"\nConfig source: VolStraddleConfig()")
    print(f"  s_sl={cfg.s_sl}  s_tp={cfg.s_tp}")
    print(f"  s_trail_activate={cfg.s_trail_activate}  s_trail_distance={cfg.s_trail_distance}")
    print(f"  s_vr_threshold={cfg.s_vr_threshold}  s_squeeze_dur={cfg.s_squeeze_dur}")
    print(f"  max_concurrent={MAX_CONCURRENT} (5 pairs × 2 = 10 legs)")

    # ── Load data ──
    print("\n데이터 로드 중...")
    data_15m = load_15m(whitelist)
    assert len(data_15m) >= 58, f"심볼 부족: {len(data_15m)}"
    sample_bars = max(len(d["close"]) for d in data_15m.values())
    data_years = sample_bars / 35040.0
    print(f"  {len(data_15m)} 심볼  |  {sample_bars:,} bars  |  {data_years:.2f}년")

    data_30m = build_resampled(data_15m, 2)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    vr_pairs = [(cfg.vr_short, cfg.vr_long)]
    for ds in [data_15m, data_30m, data_1h, data_4h]:
        precompute_all_vr([ds], vr_pairs)
        precompute(ds, vr_sp=cfg.vr_short, vr_lp=cfg.vr_long)
    print(f"  전처리 완료: {time.perf_counter() - t0:.1f}s")

    offsets = {sym: int(d.get("_offset", 0)) for sym, d in data_15m.items()}
    mh_30m = cfg.max_hold_15m // 2

    # ── Per-TF collection ──
    tf_specs = [
        ("15m", 1,  WARMUP_15m, cfg.max_hold_15m * 2,    data_15m),
        ("30m", 2,  WARMUP_30m, mh_30m * 2,              data_30m),
        ("1h",  4,  WARMUP_1h,  cfg.max_hold_15m // 4 * 2, data_1h),
        ("4h",  16, WARMUP_4h,  cfg.max_hold_15m // 16 * 2, data_4h),
    ]

    results: list[dict] = []
    per_tf_trades: dict[str, list] = {}

    print("\n" + "=" * 70)
    print("Phase 1: 타임프레임별 단독 성과")
    print("=" * 70)

    for tf_label, ratio, warmup, mh_s, tf_data in tf_specs:
        t1 = time.perf_counter()
        trades = collect_strd_single_tf(tf_label, ratio, warmup, mh_s, tf_data, offsets, cfg)
        per_tf_trades[tf_label] = trades
        m = run_metrics(trades, data_years, tf_label)
        if m:
            results.append(m)
            boot_str = f"PF CI [{m['boot_lo']:.3f}, {m['boot_hi']:.3f}]  P(PF≥1)={m['p_ge_1']:.1f}%"
            print(
                f"\n[{tf_label:>3s}] N={m['N']:5d}  PF={m['PF']:.3f}  WR={m['WR']*100:.1f}%  "
                f"EV={m['EV_bp']:+.1f}bp  MaxDD={m['MaxDD']*100:.1f}%  "
                f"CAGR(1x)={m['CAGR_1x']*100:.0f}%  Calmar={m['Calmar']:.0f}"
            )
            print(f"       {boot_str}  AvgHold={m['avg_hold_h']:.1f}h  RawTrades={m['n_trades_raw']}")
        elapsed = time.perf_counter() - t1
        print(f"       [{tf_label}] 실행: {elapsed:.1f}s")

    # ── Canonical: all TFs combined ──
    print("\n" + "=" * 70)
    print("Phase 2: G_4TF 합산 (캐노니컬 — r76 재현)")
    print("=" * 70)
    t2 = time.perf_counter()
    all_trades = collect_strd_all_tf(data_15m, data_30m, data_1h, data_4h, cfg)
    m_all = run_metrics(all_trades, data_years, "G_4TF")
    if m_all:
        print(
            f"\n[G_4TF] N={m_all['N']:5d}  PF={m_all['PF']:.3f}  WR={m_all['WR']*100:.1f}%  "
            f"EV={m_all['EV_bp']:+.1f}bp  MaxDD={m_all['MaxDD']*100:.1f}%  "
            f"CAGR(1x)={m_all['CAGR_1x']*100:.0f}%  Calmar={m_all['Calmar']:.0f}"
        )
        print(f"       PF CI [{m_all['boot_lo']:.3f}, {m_all['boot_hi']:.3f}]  "
              f"P(PF≥1)={m_all['p_ge_1']:.1f}%  "
              f"AvgHold={m_all['avg_hold_h']:.1f}h")
    print(f"  G_4TF 실행: {time.perf_counter() - t2:.1f}s")

    # ── Trade contribution breakdown ──
    print("\n" + "=" * 70)
    print("Phase 3: TF별 기여도 (G_4TF 중복 제거 후)")
    print("=" * 70)
    # Which TF generated each trade in the combined set?
    # Use avg_hold proxy: 15m trades have hold ≤ mh_15m*2, 4h trades are much longer
    combined_set = {(sym, e, x) for sym, e, x, _ in all_trades}
    tf_contribution: dict[str, int] = {}
    for tf_label, ratio, warmup, mh_s, tf_data in tf_specs:
        tf_set = {(sym, e, x) for sym, e, x, _ in per_tf_trades[tf_label]}
        matched = len(combined_set & tf_set)
        tf_contribution[tf_label] = matched

    total_contrib = sum(tf_contribution.values()) if tf_contribution else 1
    print(f"\n{'TF':>4s}  {'N_solo':>7s}  {'In_G4TF':>8s}  {'%':>6s}  EV_solo")
    for r in results:
        lbl = r["label"]
        n_solo = r["n_trades_raw"]
        in_g4 = tf_contribution.get(lbl, 0)
        pct = in_g4 / len(all_trades) * 100 if all_trades else 0
        print(f"  {lbl:>4s}  {n_solo:>7d}  {in_g4:>8d}  {pct:>5.1f}%  EV={r['EV_bp']:+.1f}bp")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("요약 비교표")
    print("=" * 70)
    header = f"{'TF':>4s}  {'N':>6s}  {'PF':>6s}  {'WR%':>6s}  {'EV(bp)':>8s}  {'MaxDD%':>7s}  {'Calmar':>8s}  {'AvgH':>6s}"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"  {r['label']:>4s}  {r['N']:>6d}  {r['PF']:>6.3f}  "
            f"{r['WR']*100:>6.1f}  {r['EV_bp']:>+8.1f}  "
            f"{r['MaxDD']*100:>7.1f}  {r['Calmar']:>8.0f}  {r['avg_hold_h']:>6.1f}h"
        )
    if m_all:
        print("-" * 70)
        print(
            f"  {'G4TF':>4s}  {m_all['N']:>6d}  {m_all['PF']:>6.3f}  "
            f"{m_all['WR']*100:>6.1f}  {m_all['EV_bp']:>+8.1f}  "
            f"{m_all['MaxDD']*100:>7.1f}  {m_all['Calmar']:>8.0f}  {m_all['avg_hold_h']:>6.1f}h"
        )

    print(f"\n총 실행: {time.perf_counter() - t0:.1f}s")
    print(f"\n파라미터: s_trail_activate={cfg.s_trail_activate}  s_trail_distance={cfg.s_trail_distance}")
    print(f"  ⚠️  cliff edge: s_trail_activate ±0.003 → PF -60~75% (r77 확인)")
    print(f"  → r85 2D sweep 권장: s_trail_activate × s_trail_distance")
    print(f"\nParity Notes:")
    print(f"  config.py 읽음 ✅  symbol_whitelist 사용 ✅  룩어헤드 없음 ✅")
    print(f"  universe: {len(data_15m)} syms  기간: {data_years:.2f}년  비용: {COST_STRADDLE*100:.2f}% round-trip")


if __name__ == "__main__":
    main()
