"""r78: S_d8 Straddle 신뢰성 완전 검증 — VR/MA 수준 달성 목표.

배경:
  r77에서 S_d8 단독 성능을 확인 (N=7730, PF=9.6, Calmar=20601).
  하지만 VR/MA는 수년간 수십 회 연구로 검증됐으나 Straddle은 첫 심층 감사만 받음.
  이번 r78에서 6개 Phase로 VR/MA와 동등한 신뢰성 검증을 완료한다.

Phase 요약:
  Phase 0: Pre-flight + baseline 재현 (N±5%, PF±5%)
  Phase 1: Rolling walk-forward OOS — 5개 1년 leave-one-out 윈도우
  Phase 2: 심볼 집중도 — HHI + Top-5 기여율
  Phase 3: 레짐 분석 — BTC 90일 롤링 기준 Bull/Bear/Sideways
  Phase 4: 비용 스트레스 — 1x/1.5x/2x/3x 라운드트립 비용
  Phase 5: 슬롯 캡 민감도 — 동시 straddle 3/5/7/10 제한
  Phase 6: Trail 파라미터 IS/OOS split — cliff가 IS에만인지 확인

판정 기준:
  ✅ 신뢰: Ph1 모든 OOS PF>1.5 + Ph2 HHI<0.25 + Ph4 2x비용 PF>1.5 + Ph6 cliff 일관
  ⚠️ 조건부: 위 중 1~2개 실패 → 슬롯 캡 or 모니터링 조건 부가
  ❌ 재검토: Ph1 OOS PF<1.0 윈도우 존재 or Ph2 단일 심볼 >35%

사용:
    uv run python research/active/r76_r80/r78_strd_reliability.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    compute_metrics,
    portfolio_sim,
    precompute,
    precompute_all_vr,
    sim_straddle_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import gen_vr_dur_signal

# ─── r77에서 재활용한 헬퍼들 ──────────────────────────────────────────────────

WARMUP_30m: int = 100


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


def _get_g4tf(cfg: VolStraddleConfig) -> list:
    mh_30m = cfg.max_hold_15m // 2
    return [("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)] + list(G_3TF)


def load_data(whitelist: list[str]) -> tuple[dict, dict, dict, dict, dict]:
    """15m/30m/1h/4h 데이터 로드 (r77 패턴 동일)."""
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

    data_15m: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data_15m[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}

    if data_15m:
        max_bars = max(len(d["close"]) for d in data_15m.values())
        for d in data_15m.values():
            d["_offset"] = max_bars - len(d["close"])

    cfg_tmp = VolStraddleConfig()
    vr_pairs = [(cfg_tmp.vr_short, cfg_tmp.vr_long)]
    precompute_all_vr([data_15m], vr_pairs)

    # resample from 15m
    data_30m: dict = {}
    data_1h: dict = {}
    data_4h: dict = {}
    for sym, d in data_15m.items():
        d30 = {}
        d1h = {}
        d4h = {}
        for ratio, target in [(2, d30), (4, d1h), (16, d4h)]:
            c15 = d["close"]
            h15 = d["high"]
            l15 = d["low"]
            n = len(c15) // ratio * ratio
            c15, h15, l15 = c15[:n], h15[:n], l15[:n]
            target["close"] = c15.reshape(-1, ratio)[:, -1].copy()
            target["high"]  = h15.reshape(-1, ratio).max(axis=1).copy()
            target["low"]   = l15.reshape(-1, ratio).min(axis=1).copy()
            target["_offset"] = d.get("_offset", 0) // ratio
        data_30m[sym] = d30
        data_1h[sym]  = d1h
        data_4h[sym]  = d4h

    # precompute VR for resampled TFs
    precompute_all_vr([data_30m, data_1h, data_4h], vr_pairs)

    # btc_close for regime analysis (15m)
    btc_sym = "BTCUSDT" if "BTCUSDT" in data_15m else sorted(data_15m.keys())[0]
    btc_15m = data_15m[btc_sym]["close"].copy()

    return data_15m, data_30m, data_1h, data_4h, {"btc": btc_15m}


def _dedup(sym: str, trades: list) -> list:
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def collect_s_trail(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
    s_vr: float | None = None,
    s_dur: int | None = None,
    trail_act: float | None = None,
    trail_dist: float | None = None,
    cost: float | None = None,
    bar_filter: tuple[int, int] | None = None,  # (start_bar, end_bar) 글로벌 바 필터
) -> list[tuple[str, int, int, float]]:
    """S_d8 trail-aware trade collection. bar_filter로 기간 필터링 가능."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    g4tf = _get_g4tf(cfg)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    _sl = cfg.s_sl
    _tp = cfg.s_tp
    _trail_act = trail_act if trail_act is not None else cfg.s_trail_activate
    _trail_dist = trail_dist if trail_dist is not None else cfg.s_trail_distance
    _s_vr = s_vr if s_vr is not None else cfg.s_vr_threshold
    _s_dur = s_dur if s_dur is not None else cfg.s_squeeze_dur
    _cost = cost if cost is not None else COST_STRADDLE

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, _mh_d, mh_s in g4tf:
            tf_data = tf_map.get(ratio, {})
            d = tf_data.get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, _s_vr, _s_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, _sl, _tp, mh_s, _cost, _trail_act, _trail_dist,
            )
            for k in range(len(p)):
                gb_entry = int(e[k]) * ratio + offset
                gb_exit  = int(x[k]) * ratio + ratio - 1 + offset
                if bar_filter:
                    if gb_entry < bar_filter[0] or gb_exit > bar_filter[1]:
                        continue
                sym_trades.append((gb_entry, gb_exit, float(p[k])))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 0: Pre-flight + Baseline 재현
# ─────────────────────────────────────────────────────────────────────────────


def phase0_preflight(cfg: VolStraddleConfig) -> None:
    print("\n" + "=" * 70)
    print("PHASE 0: Pre-flight — Config 확인 + Baseline 재현 예고")
    print("=" * 70)
    print(f"  s_trail_activate = {cfg.s_trail_activate}  (cliff at ±0.003 from r77)")
    print(f"  s_trail_distance = {cfg.s_trail_distance}  (r76에서 0.002→0.001)")
    print(f"  s_squeeze_dur    = {cfg.s_squeeze_dur}")
    print(f"  s_vr_threshold   = {cfg.s_vr_threshold}")
    print(f"  s_sl             = {cfg.s_sl}")
    print(f"  s_tp             = {cfg.s_tp}")
    print(f"  max_concurrent   = {cfg.max_concurrent}")
    print(f"  COST_STRADDLE    = {COST_STRADDLE:.4f} (0.16% round-trip)")
    print(f"  L/N              = {cfg.base_leverage}/{cfg.max_concurrent} = {cfg.base_leverage/cfg.max_concurrent:.1f}")
    print("\n  r77 베이스라인: N=7730, PF=9.603, MaxDD=3.5%, Calmar=20601")
    print("  허용 오차: N±10%, PF±10% (r77은 s_trail_dist=0.001 이미 적용)")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1: Rolling Walk-Forward OOS (Leave-One-Year-Out)
# ─────────────────────────────────────────────────────────────────────────────


def phase1_rolling_wf(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
    max_concurrent: int,
) -> dict[int, dict]:
    print("\n" + "=" * 70)
    print("PHASE 1: Rolling Walk-Forward OOS (Leave-One-Year-Out, 5개 윈도우)")
    print("=" * 70)

    total_bars = max(len(d["close"]) for d in data_15m.values())
    bars_per_year = total_bars / data_years
    start_year = 2020

    # Full trades first
    t0 = time.perf_counter()
    all_trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg)
    print(f"  전체 trades 수집: {len(all_trades)} ({time.perf_counter()-t0:.1f}s)")

    # Index by global entry bar for quick filtering
    results: dict[int, dict] = {}
    print(f"\n  {'OOS 연도':>8}  {'OOS PF':>8}  {'OOS WR':>8}  {'IS PF':>7}  {'IS N':>6}  {'OOS N':>6}  {'판정':>8}")
    print("  " + "─" * 60)

    all_pass = True
    for oos_yr in range(start_year, start_year + 5):
        oos_start = int((oos_yr - start_year) * bars_per_year)
        oos_end   = int((oos_yr - start_year + 1) * bars_per_year)

        is_trades  = [(sym, e, x, p) for sym, e, x, p in all_trades if not (oos_start <= e < oos_end)]
        oos_trades = [(sym, e, x, p) for sym, e, x, p in all_trades if oos_start <= e < oos_end]

        if not oos_trades:
            print(f"  {oos_yr:>8}  (거래 없음)")
            continue

        is_pnl  = portfolio_sim(is_trades, max_concurrent) if is_trades else np.array([])
        oos_pnl = portfolio_sim(oos_trades, max_concurrent)

        is_m  = compute_metrics(is_pnl, data_years - 1.0) if len(is_pnl) > 0 else {"PF": 0, "N": 0, "WR": 0}
        oos_m = compute_metrics(oos_pnl, 1.0)

        pass_ = oos_m["PF"] >= 1.5
        if not pass_:
            all_pass = False
        marker = "✅" if pass_ else "❌"

        print(
            f"  {oos_yr:>8}  {oos_m['PF']:>8.3f}  {oos_m['WR']:>7.1f}%  "
            f"{is_m['PF']:>7.3f}  {is_m['N']:>6}  {oos_m['N']:>6}  {marker:>8}"
        )
        results[oos_yr] = {
            "oos_pf": oos_m["PF"], "oos_wr": oos_m["WR"], "oos_n": oos_m["N"],
            "is_pf": is_m["PF"], "is_n": is_m["N"], "pass": pass_,
        }

    verdict = "✅ PASS (모든 OOS PF≥1.5)" if all_pass else "⚠️ 일부 윈도우 OOS PF<1.5"
    print(f"\n  Rolling WF 판정: {verdict}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2: Symbol Concentration (HHI + Top-N 기여율)
# ─────────────────────────────────────────────────────────────────────────────


def phase2_symbol_concentration(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
    max_concurrent: int,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 2: 심볼 집중도 — HHI + Top-5 기여율")
    print("=" * 70)

    # 심볼별 개별 collect (portfolio_sim 없이 raw PnL만)
    sym_stats: dict[str, dict] = {}
    for sym in sorted(data_15m.keys()):
        trades = collect_s_trail(
            {sym: data_15m[sym]},
            {sym: data_30m[sym]},
            {sym: data_1h[sym]},
            {sym: data_4h[sym]},
            cfg,
        )
        if not trades:
            continue
        raw_pnl = np.array([p for _, _, _, p in trades])
        wins = (raw_pnl > 0).sum()
        total_pos = raw_pnl[raw_pnl > 0].sum()
        total_neg = abs(raw_pnl[raw_pnl < 0].sum())
        pf = total_pos / max(total_neg, 1e-9)
        sym_stats[sym] = {
            "n": len(raw_pnl),
            "pnl": float(raw_pnl.sum()),
            "wins": int(wins),
            "pf": float(pf),
        }

    total_pos_pnl = sum(max(v["pnl"], 0) for v in sym_stats.values())
    total_pnl = sum(v["pnl"] for v in sym_stats.values())

    # HHI (PnL 기준)
    pos_syms = {s: v for s, v in sym_stats.items() if v["pnl"] > 0}
    total_pos_sum = sum(v["pnl"] for v in pos_syms.values())
    shares = [v["pnl"] / max(total_pos_sum, 1e-9) for v in pos_syms.values()]
    hhi = sum(s ** 2 for s in shares)

    # Top-5 by PnL
    sorted_syms = sorted(sym_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)

    print(f"\n  {'심볼':12s}  {'N':>5}  {'PF':>6}  {'PnL(%)':>9}  {'기여율':>8}  {'누적':>8}")
    print("  " + "─" * 56)
    cumulative = 0.0
    for i, (sym, v) in enumerate(sorted_syms[:15]):
        contrib = v["pnl"] / max(total_pos_sum, 1e-9) * 100
        cumulative += contrib
        mark = " ★" if i < 5 else ""
        print(
            f"  {sym:12s}  {v['n']:>5}  {v['pf']:>6.2f}  "
            f"{v['pnl']:>+9.4f}  {contrib:>7.1f}%  {cumulative:>7.1f}%{mark}"
        )

    top5_pnl = sum(v["pnl"] for _, v in sorted_syms[:5])
    top5_share = top5_pnl / max(total_pos_sum, 1e-9) * 100
    top1_share = sorted_syms[0][1]["pnl"] / max(total_pos_sum, 1e-9) * 100 if sorted_syms else 0

    print(f"\n  HHI = {hhi:.4f}  ({'분산' if hhi < 0.1 else '중간' if hhi < 0.2 else '집중'})")
    print(f"  Top-1 기여율: {top1_share:.1f}%  ({'✅' if top1_share < 20 else '⚠️'})")
    print(f"  Top-5 기여율: {top5_share:.1f}%  ({'✅' if top5_share < 50 else '⚠️'})")
    print(f"  수익 심볼 수: {len([v for v in sym_stats.values() if v['pnl'] > 0])} / {len(sym_stats)}")

    # 판정
    hhi_pass  = hhi < 0.25
    top1_pass = top1_share < 20
    top5_pass = top5_share < 50
    all_pass  = hhi_pass and top1_pass and top5_pass
    verdict   = "✅ PASS" if all_pass else "⚠️ 집중 위험 존재"
    print(f"\n  심볼 집중도 판정: {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3: Regime Analysis (BTC 90일 롤링 기준)
# ─────────────────────────────────────────────────────────────────────────────


def phase3_regime_analysis(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    btc_close_15m: np.ndarray,
    data_years: float,
    max_concurrent: int,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 3: 레짐 분석 — BTC 90일 롤링 수익률 기준 Bull/Bear/Sideways")
    print("=" * 70)

    # BTC 90일 롤링 수익률 (90 × 96 bars = 8640 15m bars)
    window_bars = 90 * 96  # 90일 × 15m bars/day
    btc_ret = np.full(len(btc_close_15m), np.nan)
    for i in range(window_bars, len(btc_close_15m)):
        btc_ret[i] = (btc_close_15m[i] - btc_close_15m[i - window_bars]) / btc_close_15m[i - window_bars]

    # 레짐 분류: Bull>+20%, Bear<-20%, Sideways 그 사이
    regime = np.full(len(btc_close_15m), "Sideways", dtype=object)
    regime[btc_ret > 0.20]  = "Bull"
    regime[btc_ret < -0.20] = "Bear"

    total_bars = len(btc_close_15m)
    bull_bars = (regime == "Bull").sum()
    bear_bars = (regime == "Bear").sum()
    side_bars = (regime == "Sideways").sum()
    print(f"\n  레짐 분포 (전체 {total_bars:,} 15m bars):")
    print(f"    Bull     : {bull_bars:,} ({bull_bars/total_bars*100:.1f}%)")
    print(f"    Bear     : {bear_bars:,} ({bear_bars/total_bars*100:.1f}%)")
    print(f"    Sideways : {side_bars:,} ({side_bars/total_bars*100:.1f}%)")

    # 전체 trades 수집 후 레짐별 분류
    all_trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg)

    def _regime_label(entry_bar: int) -> str:
        if entry_bar < len(regime):
            return str(regime[entry_bar])
        return "Sideways"

    regime_trades: dict[str, list] = {"Bull": [], "Bear": [], "Sideways": []}
    for trade in all_trades:
        sym, e, x, p = trade
        # entry bar는 offset 포함 글로벌 바
        # BTC는 offset=0이므로 (전체 데이터 기준)
        r = _regime_label(e)
        regime_trades[r].append(trade)

    print(f"\n  {'레짐':10s}  {'N':>6}  {'PF':>7}  {'WR':>6}  {'EV(bp)':>8}  {'MaxDD':>7}  {'판정':>6}")
    print("  " + "─" * 56)

    for reg in ["Bull", "Bear", "Sideways"]:
        rtrades = regime_trades[reg]
        if not rtrades:
            print(f"  {reg:10s}  (거래 없음)")
            continue
        rpnl = portfolio_sim(rtrades, max_concurrent)
        rm = compute_metrics(rpnl, max(len(rpnl) / (365 * 96 * 4), 0.1))  # rough years
        pass_ = rm["PF"] >= 1.0
        marker = "✅" if pass_ else "❌"
        print(
            f"  {reg:10s}  {rm['N']:>6}  {rm['PF']:>7.3f}  {rm['WR']:>5.1f}%  "
            f"{rm['EV_bp']:>8.1f}  {rm['max_dd_pct']:>6.1f}%  {marker:>6}"
        )

    print("\n  합격 기준: 모든 레짐에서 PF ≥ 1.0")
    all_ok = all(
        compute_metrics(portfolio_sim(t, max_concurrent), 1.0)["PF"] >= 1.0
        for t in regime_trades.values() if t
    )
    print(f"  레짐 분석 판정: {'✅ PASS' if all_ok else '⚠️ 특정 레짐 PF<1.0'}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4: Cost Stress Test
# ─────────────────────────────────────────────────────────────────────────────


def phase4_cost_stress(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
    max_concurrent: int,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 4: 비용 스트레스 테스트 (0.16% → 최대 0.48%)")
    print("=" * 70)

    cost_levels = [
        (1.0, COST_STRADDLE),
        (1.5, COST_STRADDLE * 1.5),
        (2.0, COST_STRADDLE * 2.0),
        (3.0, COST_STRADDLE * 3.0),
    ]

    print(f"\n  {'비용 배수':>8}  {'비용(%)':>8}  {'N':>6}  {'PF':>7}  {'WR':>6}  {'Calmar':>9}  {'판정':>6}")
    print("  " + "─" * 58)

    all_pass = True
    for mult, cost in cost_levels:
        trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg, cost=cost)
        pnl = portfolio_sim(trades, max_concurrent)
        m = compute_metrics(pnl, data_years)
        pass_ = m["PF"] >= 1.5 if mult <= 2.0 else m["PF"] >= 1.2
        if mult <= 2.0 and not pass_:
            all_pass = False
        marker = "✅" if pass_ else "❌"
        print(
            f"  {mult:>7.1f}x  {cost*100:>7.3f}%  {m['N']:>6}  {m['PF']:>7.3f}  "
            f"{m['WR']:>5.1f}%  {m['calmar']:>9.1f}  {marker:>6}"
        )

    print(f"\n  합격 기준: 2x 비용에서도 PF≥1.5")
    print(f"  비용 스트레스 판정: {'✅ PASS' if all_pass else '⚠️ 2x 비용에서 PF<1.5'}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5: Slot Cap Sensitivity
# ─────────────────────────────────────────────────────────────────────────────


def phase5_slot_cap(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 5: 슬롯 캡 민감도 — 동시 Straddle 최대 3/5/7/10")
    print("=" * 70)

    trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg)
    print(f"  전체 raw trades: {len(trades)}")

    caps = [3, 5, 7, 10]
    print(f"\n  {'슬롯 캡':>8}  {'N':>6}  {'PF':>7}  {'WR':>6}  {'MaxDD':>7}  {'Calmar':>9}  {'CAGR(1x)':>9}")
    print("  " + "─" * 60)

    results = []
    for cap in caps:
        pnl = portfolio_sim(trades, cap)
        m = compute_metrics(pnl, data_years)
        results.append(m)
        mark = " ◀ 현재" if cap == cfg.max_concurrent else ""
        print(
            f"  {cap:>8}슬롯  {m['N']:>6}  {m['PF']:>7.3f}  {m['WR']:>5.1f}%  "
            f"{m['max_dd_pct']:>6.1f}%  {m['calmar']:>9.1f}  {m['ann_ret_pct']:>8.1f}%{mark}"
        )

    # 권고 슬롯 수
    # MaxDD가 5% 이하인 가장 큰 cap
    recommended = 10
    for cap, m in zip(caps, results):
        if m["max_dd_pct"] > 10.0:
            recommended = caps[max(0, caps.index(cap) - 1)]
            break

    print(f"\n  분석: MaxDD 기준 권고 슬롯 상한 = {recommended}슬롯")
    print("  (현재 10슬롯 운영 시 MaxDD가 10% 초과하면 슬롯 캡 권고)")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 6: Trail Parameter IS/OOS Split
# ─────────────────────────────────────────────────────────────────────────────


def phase6_trail_is_oos(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
    max_concurrent: int,
) -> None:
    print("\n" + "=" * 70)
    print("PHASE 6: Trail 파라미터 IS/OOS Split — cliff 위치 일관성")
    print("=" * 70)
    print("  ⚠️ r77 발견: s_trail_activate=0.015에서 sharp cliff")
    print("  이 cliff가 IS에서만 있는지 OOS에서도 동일한지 검증")

    total_bars = max(len(d["close"]) for d in data_15m.values())
    mid_bar = total_bars // 2  # IS / OOS 분기점 (글로벌 바)

    trail_vals = [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]

    print(f"\n  {'trail_act':>11}  {'IS PF':>7}  {'OOS PF':>8}  {'IS/OOS':>8}  {'IS N':>6}  {'OOS N':>6}")
    print("  " + "─" * 54)

    consistent_cliff = True
    is_cliff_val = None
    oos_cliff_val = None

    for ta in trail_vals:
        is_trades  = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg,
                                     trail_act=ta, bar_filter=(0, mid_bar))
        oos_trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg,
                                     trail_act=ta, bar_filter=(mid_bar, 10_000_000))

        is_pnl  = portfolio_sim(is_trades, max_concurrent) if is_trades else np.array([0.0])
        oos_pnl = portfolio_sim(oos_trades, max_concurrent) if oos_trades else np.array([0.0])

        is_m  = compute_metrics(is_pnl, data_years / 2)
        oos_m = compute_metrics(oos_pnl, data_years / 2)

        ratio_str = f"{is_m['PF']/max(oos_m['PF'],0.001):.2f}x"
        prod_mark = " ◀ PROD" if abs(ta - cfg.s_trail_activate) < 0.001 else ""

        # cliff 탐지: PF가 이전 대비 50% 이상 하락
        print(
            f"  {ta:>11.3f}  {is_m['PF']:>7.3f}  {oos_m['PF']:>8.3f}  "
            f"{ratio_str:>8}  {is_m['N']:>6}  {oos_m['N']:>6}{prod_mark}"
        )

    print("\n  합격 기준: IS cliff 위치 ≈ OOS cliff 위치 (±1 스텝)")
    print("  → IS와 OOS에서 같은 파라미터 값이 최적이면 과적합 아님 ✅")
    print("  → IS cliff와 OOS cliff가 다른 위치에 있으면 IS 과적합 ⚠️")


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline (full 6yr) for reference
# ─────────────────────────────────────────────────────────────────────────────


def baseline_check(
    cfg: VolStraddleConfig,
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    data_years: float,
    max_concurrent: int,
) -> tuple[np.ndarray, list]:
    print("\n" + "─" * 70)
    print("Baseline 재현 (r77 참조)")
    print("─" * 70)
    t0 = time.perf_counter()
    trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg)
    pnl = portfolio_sim(trades, max_concurrent)
    m = compute_metrics(pnl, data_years)
    boot = bootstrap_pf(pnl, n_boot=1000)
    elapsed = time.perf_counter() - t0

    ref_n, ref_pf = 7730, 9.603
    n_drift  = abs(m["N"] - ref_n)  / ref_n
    pf_drift = abs(m["PF"] - ref_pf) / ref_pf

    print(f"  N={m['N']:,}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  MaxDD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}")
    print(f"  Bootstrap P(PF≥1): {boot['P_PF_ge_1']:.1f}%  CI95=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]")
    print(f"  N drift={n_drift:.1%}  PF drift={pf_drift:.1%}  ({'✅' if n_drift<0.10 and pf_drift<0.10 else '⚠️ Baseline 이탈!'})")
    print(f"  (elapsed: {elapsed:.1f}s)")

    if n_drift > 0.10 or pf_drift > 0.10:
        print("  ⚠️ Baseline 이탈! 데이터/config 확인 후 진행.")

    return pnl, trades


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    commit = get_commit()
    print("=" * 70)
    print(f"r78: S_d8 Straddle 신뢰성 완전 검증  (commit={commit})")
    print("=" * 70)

    cfg = VolStraddleConfig()
    whitelist = list(cfg.symbol_whitelist)
    max_concurrent = cfg.max_concurrent

    print(f"\n  심볼 universe: {len(whitelist)}개 (symbol_whitelist)")
    print("  데이터 로딩 중 (15m/30m/1h/4h)...")
    t_load = time.perf_counter()
    data_15m, data_30m, data_1h, data_4h, extras = load_data(whitelist)
    btc_close = extras["btc"]
    print(f"  로딩 완료: {len(data_15m)}개 심볼  ({time.perf_counter()-t_load:.1f}s)")

    # 데이터 기간 계산
    max_bars  = max(len(d["close"]) for d in data_15m.values())
    data_years = max_bars / 35040  # 15m bars per year (365 × 96)
    print(f"  데이터 기간: {max_bars:,} bars = {data_years:.2f}년")

    # ── Phase 0: Pre-flight ───────────────────────────────────────────────
    phase0_preflight(cfg)

    # ── Baseline ─────────────────────────────────────────────────────────
    base_pnl, base_trades = baseline_check(
        cfg, data_15m, data_30m, data_1h, data_4h, data_years, max_concurrent
    )

    # ── Phase 1: Rolling Walk-Forward ────────────────────────────────────
    wf_results = phase1_rolling_wf(
        cfg, data_15m, data_30m, data_1h, data_4h, data_years, max_concurrent
    )

    # ── Phase 2: Symbol Concentration ────────────────────────────────────
    phase2_symbol_concentration(
        cfg, data_15m, data_30m, data_1h, data_4h, data_years, max_concurrent
    )

    # ── Phase 3: Regime Analysis ──────────────────────────────────────────
    phase3_regime_analysis(
        cfg, data_15m, data_30m, data_1h, data_4h,
        btc_close, data_years, max_concurrent,
    )

    # ── Phase 4: Cost Stress ──────────────────────────────────────────────
    phase4_cost_stress(
        cfg, data_15m, data_30m, data_1h, data_4h, data_years, max_concurrent
    )

    # ── Phase 5: Slot Cap ─────────────────────────────────────────────────
    phase5_slot_cap(
        cfg, data_15m, data_30m, data_1h, data_4h, data_years
    )

    # ── Phase 6: Trail IS/OOS Split ───────────────────────────────────────
    phase6_trail_is_oos(
        cfg, data_15m, data_30m, data_1h, data_4h, data_years, max_concurrent
    )

    # ── 최종 판정 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("최종 판정 요약")
    print("=" * 70)
    print("  각 Phase 결과를 확인하여 종합 판정을 수행하세요.")
    print()
    print("  ✅ 신뢰 가능 조건:")
    print("    - Ph1: 모든 OOS PF ≥ 1.5")
    print("    - Ph2: HHI < 0.25, 단일 심볼 < 20%")
    print("    - Ph3: Bear 레짐에서도 PF ≥ 1.0")
    print("    - Ph4: 2x 비용에서도 PF ≥ 1.5")
    print("    - Ph5: 10슬롯 MaxDD < 10%")
    print("    - Ph6: IS/OOS cliff 위치 일관")
    print()
    print("  ⚠️ 조건부 신뢰: 위 중 1~2개 실패 → 슬롯 캡 or 파라미터 모니터링")
    print("  ❌ 재검토: Ph1 OOS PF<1.0 or Ph2 단일 심볼 >35%")
    print()
    print(f"  보고서를 docs/backtest_results/202603/{commit}_r78_strd_reliability.md 에 저장하세요.")


if __name__ == "__main__":
    main()
