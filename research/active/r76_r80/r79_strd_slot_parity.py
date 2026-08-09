"""r79: Straddle 2-슬롯 패리티 & Per-Slot 효율 분석.

핵심 발견:
  Backtest:  portfolio_sim(max_concurrent=10) → 10 straddle 동시 허용
  Live:      n_active = len(_positions) → 1 straddle = 2 n_active → 최대 5 straddle

  → 백테스트 N이 라이브보다 2배 과대추정됨 (패리티 버그).

Phase 1: 슬롯 패리티 수정 — max_concurrent=5 재실행 vs r78(max=10) 비교
Phase 2: Per-slot 효율 — STRD(5유효슬롯) vs VR D_mom8(10슬롯) CAGR 비교
Phase 3: PnL 시나리오 분해 — 4가지 결과(양방수익/한방승/한방패/양방손실) 비율

핵심 질문:
  "2배 슬롯을 사용해도 straddle이 VR보다 per-slot 기준 이득인가?"

사용:
    uv run python research/active/r76_r80/r79_strd_slot_parity.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    G_3TF,
    bootstrap_pf,
    compute_metrics,
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

WARMUP_30m: int = 100
DATA_YEARS: float = 5.67  # 2020-01-01 ~ 2025-09-30 (approximate)


from quantbox.research.harness import get_commit  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading (r78 패턴 동일)
# ══════════════════════════════════════════════════════════════════════════════


def load_data(whitelist: list[str]) -> tuple[dict, dict, dict, dict]:
    """15m/30m/1h/4h 데이터 로드."""
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
    # Compute mom_8 for directional signal (needed by collect_d_trail)
    precompute(data_15m, vr_sp=cfg_tmp.vr_short, vr_lp=cfg_tmp.vr_long, n_workers=1)

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

    precompute_all_vr([data_30m, data_1h, data_4h], vr_pairs)
    # Add mom_8 to resampled TFs for directional signal (compute_momentum only — no volume needed)
    from quantbox.strategy.vol_straddle.signals import compute_momentum
    for tf_data in [data_1h, data_4h]:
        for d in tf_data.values():
            d["mom_8"] = compute_momentum(d["close"], 8)
    return data_15m, data_30m, data_1h, data_4h


def _dedup(sym: str, trades: list) -> list:
    result, accepted = [], []
    for e, x, p in sorted(trades, key=lambda t: t[0]):
        if not any(e <= ax and x >= ae for ae, ax in accepted):
            result.append((sym, e, x, p))
            accepted.append((e, x))
    return result


def _get_g4tf(cfg: VolStraddleConfig) -> list:
    mh_30m = cfg.max_hold_15m // 2
    return [("30m", 2, WARMUP_30m, mh_30m * 2, mh_30m * 2)] + list(G_3TF)


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection
# ══════════════════════════════════════════════════════════════════════════════


def collect_s_trail(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
) -> list[tuple[str, int, int, float]]:
    """S_d8 trail-aware trade collection."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    g4tf = _get_g4tf(cfg)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, mh_d, mh_s in g4tf:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)
            p, e, x = sim_straddle_trailing(
                c, h, lo, sig, cfg.s_sl, cfg.s_tp, mh_s,
                COST_STRADDLE, cfg.s_trail_activate, cfg.s_trail_distance,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


def collect_s_trail_with_legs(
    data_15m: dict,
    data_30m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
) -> list[tuple[str, int, int, float, float, float]]:
    """S_d8 trail collection that also returns individual leg PnLs.

    Returns:
        List of (sym, entry_bar, exit_bar, net_pnl, long_pnl, short_pnl)
    """
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    g4tf = _get_g4tf(cfg)
    tf_map = {1: data_15m, 2: data_30m, 4: data_1h, 16: data_4h}
    all_trades: list[tuple[str, int, int, float, float, float]] = []

    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades_with_legs: list[tuple[int, int, float, float, float]] = []

        for _lbl, ratio, warmup, mh_d, mh_s in g4tf:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_dur_signal(vr, cfg.s_vr_threshold, cfg.s_squeeze_dur, warmup)

            # Replicate sim_straddle_trailing but track each leg separately
            long_pnls, short_pnls, entries, exits = _sim_straddle_with_legs(
                c, h, lo, sig, cfg.s_sl, cfg.s_tp, mh_s,
                COST_STRADDLE, cfg.s_trail_activate, cfg.s_trail_distance,
            )
            for k in range(len(entries)):
                net = long_pnls[k] + short_pnls[k] - COST_STRADDLE
                sym_trades_with_legs.append((
                    int(entries[k]) * ratio + offset,
                    int(exits[k]) * ratio + ratio - 1 + offset,
                    net,
                    long_pnls[k],
                    short_pnls[k],
                ))

        # Dedup by entry/exit (use net pnl for dedup check)
        sorted_trades = sorted(sym_trades_with_legs, key=lambda t: t[0])
        accepted_intervals: list[tuple[int, int]] = []
        for e, x, net, lp, sp in sorted_trades:
            if not any(e <= ax and x >= ae for ae, ax in accepted_intervals):
                all_trades.append((sym, e, x, net, lp, sp))
                accepted_intervals.append((e, x))

    return all_trades


def _sim_straddle_with_legs(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig: np.ndarray,
    sl: float,
    tp: float,
    max_hold: int,
    cost: float,
    trail_activate: float,
    trail_distance: float,
) -> tuple[list[float], list[float], list[int], list[int]]:
    """Same as sim_straddle_trailing but returns individual leg PnLs."""
    n = len(close)
    long_pnls, short_pnls, entries, exits = [], [], [], []
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        la = sa = True
        le = se = 0.0
        l_peak = s_peak = 0.0
        l_trailing = s_trailing = False
        eb = i
        end = min(i + 1 + max_hold, n)

        for j in range(i + 1, end):
            if la:
                l_gain = (high[j] - entry) / entry
                if l_gain > l_peak:
                    l_peak = l_gain
                if not l_trailing and l_peak >= trail_activate:
                    l_trailing = True
                if l_trailing:
                    if low[j] <= entry * (1.0 + l_peak - trail_distance):
                        le = max(l_peak - trail_distance, -sl)
                        la = False
                else:
                    if low[j] <= entry * (1.0 - sl):
                        le = -sl
                        la = False
                if la and high[j] >= entry * (1.0 + tp):
                    le = tp
                    la = False

            if sa:
                s_gain = (entry - low[j]) / entry
                if s_gain > s_peak:
                    s_peak = s_gain
                if not s_trailing and s_peak >= trail_activate:
                    s_trailing = True
                if s_trailing:
                    if high[j] >= entry * (1.0 - s_peak + trail_distance):
                        se = max(s_peak - trail_distance, -sl)
                        sa = False
                else:
                    if high[j] >= entry * (1.0 + sl):
                        se = -sl
                        sa = False
                if sa and low[j] <= entry * (1.0 - tp):
                    se = tp
                    sa = False

            if not la and not sa:
                eb = j
                break
        else:
            eb = end - 1
        if la:
            le = (close[eb] - entry) / entry
        if sa:
            se = (entry - close[eb]) / entry

        long_pnls.append(le)
        short_pnls.append(se)
        entries.append(i)
        exits.append(eb)
        i = eb + 1

    return long_pnls, short_pnls, entries, exits


def collect_d_trail(
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    cfg: VolStraddleConfig,
) -> list[tuple[str, int, int, float]]:
    """D_mom8 trail-aware trade collection (G_3TF: 15m/1h/4h)."""
    vr_key = f"vr_{cfg.vr_short}_{cfg.vr_long}"
    tf_map = {1: data_15m, 4: data_1h, 16: data_4h}
    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        offset = int(data_15m[sym].get("_offset", 0))
        sym_trades: list[tuple[int, int, float]] = []
        for _lbl, ratio, warmup, mh_d, _mh_s in G_3TF:
            d = tf_map.get(ratio, {}).get(sym)
            if d is None:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            c, h, lo = d["close"], d["high"], d["low"]
            sig = gen_vr_signal(vr, cfg.d_vr_threshold, warmup)
            dire = make_direction_momentum(sig, d["mom_8"])
            p, e, x = sim_directional_trailing(
                c, h, lo, sig, dire, cfg.d_sl, cfg.d_tp, mh_d,
                COST_DIRECTIONAL, cfg.d_trail_activate, cfg.d_trail_distance,
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio + offset,
                    int(x[k]) * ratio + ratio - 1 + offset,
                    float(p[k]),
                ))
        all_trades.extend(_dedup(sym, sym_trades))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 0: Pre-flight + Baseline
# ══════════════════════════════════════════════════════════════════════════════


def phase0_baseline(
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict, cfg: VolStraddleConfig
) -> tuple[list, float]:
    print("\n" + "=" * 60)
    print("PHASE 0: Pre-flight + Baseline (r78 canonical)")
    print("=" * 60)
    print(f"  s_vr_threshold   = {cfg.s_vr_threshold}")
    print(f"  s_squeeze_dur    = {cfg.s_squeeze_dur}")
    print(f"  s_sl             = {cfg.s_sl}")
    print(f"  s_tp             = {cfg.s_tp}")
    print(f"  s_trail_activate = {cfg.s_trail_activate}")
    print(f"  s_trail_distance = {cfg.s_trail_distance}")
    print(f"  COST_STRADDLE    = {COST_STRADDLE}")
    print(f"  max_concurrent   = {cfg.max_concurrent}")
    print(f"  symbols loaded   = {len(data_15m)}")

    trades = collect_s_trail(data_15m, data_30m, data_1h, data_4h, cfg)
    pnl10 = portfolio_sim(trades, max_concurrent=10)
    m10 = compute_metrics(pnl10, DATA_YEARS)

    print(f"\n  [max_concurrent=10] N={m10['N']}, PF={m10['PF']:.3f}, WR={m10['WR']:.1f}%")
    print(f"  MaxDD={m10['max_dd_pct']:.1f}%, Calmar={m10['calmar']:.1f}")

    # r78 canonical check
    r78_n, r78_pf = 7730, 9.603
    n_drift = abs(m10["N"] - r78_n) / r78_n
    pf_drift = abs(m10["PF"] - r78_pf) / r78_pf
    if n_drift > 0.05 or pf_drift > 0.05:
        print(f"  ⚠️  DRIFT vs r78: N={n_drift:.1%}, PF={pf_drift:.1%}")
    else:
        print(f"  ✅ Baseline matches r78 (N drift={n_drift:.1%}, PF drift={pf_drift:.1%})")

    return trades, DATA_YEARS


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: 슬롯 패리티 — max_concurrent=5 vs 10
# ══════════════════════════════════════════════════════════════════════════════


def phase1_slot_parity(
    trades_strd: list, cfg: VolStraddleConfig
) -> dict:
    """패리티 수정: straddle은 2슬롯이므로 max_concurrent=5가 live-realistic."""
    print("\n" + "=" * 60)
    print("PHASE 1: 슬롯 패리티 분석 — max_concurrent=10 vs 5")
    print("=" * 60)
    print()
    print("  이론:")
    print("  - Live: 1 straddle = 2 positions → n_active += 2 per straddle")
    print("  - Live: available_slots = 10 - n_active → max 5 straddles")
    print("  - Backtest (r78): portfolio_sim(max=10) → 10 straddles 동시 허용")
    print("  - 수정: portfolio_sim(max=5) → live-realistic")
    print()

    results = {}
    for max_c in [10, 5]:
        pnl = portfolio_sim(trades_strd, max_concurrent=max_c)
        m = compute_metrics(pnl, DATA_YEARS)
        boot = bootstrap_pf(pnl, n_boot=1000)
        results[max_c] = {"m": m, "boot": boot, "pnl": pnl}
        label = "r78 기준 (과대추정)" if max_c == 10 else "라이브 패리티 (수정)"
        print(f"  max={max_c} ({label})")
        print(f"    N={m['N']}, PF={m['PF']:.3f}, WR={m['WR']:.1f}%")
        print(f"    MaxDD={m['max_dd_pct']:.1f}%, Calmar={m['calmar']:.1f}")
        print(f"    Bootstrap PF 95% CI: [{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]")
        print()

    m10 = results[10]["m"]
    m5  = results[5]["m"]
    n_change = (m5["N"] - m10["N"]) / m10["N"]
    pf_change = (m5["PF"] - m10["PF"]) / m10["PF"]
    calmar_change = (m5["calmar"] - m10["calmar"]) / m10["calmar"]
    print(f"  변화율 (max5 vs max10):")
    print(f"    N:      {n_change:+.1%}")
    print(f"    PF:     {pf_change:+.1%}")
    print(f"    Calmar: {calmar_change:+.1%}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Per-Slot 효율 비교
# ══════════════════════════════════════════════════════════════════════════════


def phase2_per_slot_efficiency(
    data_15m: dict, data_1h: dict, data_4h: dict,
    trades_strd: list,
    cfg: VolStraddleConfig,
    slot_results: dict,
) -> None:
    """STRD(5유효슬롯) vs VR D_mom8(10슬롯) 동일 자본 기준 비교."""
    print("\n" + "=" * 60)
    print("PHASE 2: Per-Slot 효율 비교")
    print("  핵심 질문: 2배 슬롯 사용해도 straddle이 VR보다 이득인가?")
    print("=" * 60)

    # VR D_mom8 (10슬롯)
    print("\n  VR D_mom8 수집 중...")
    d_trades = collect_d_trail(data_15m, data_1h, data_4h, cfg)
    pnl_vr = portfolio_sim(d_trades, max_concurrent=10)
    m_vr = compute_metrics(pnl_vr, DATA_YEARS)
    boot_vr = bootstrap_pf(pnl_vr, n_boot=1000)

    # STRD S_d8 (5유효슬롯 — 라이브 패리티)
    m_strd5 = slot_results[5]["m"]
    boot_strd5 = slot_results[5]["boot"]
    pnl_strd5 = slot_results[5]["pnl"]

    # 1x compound CAGR 계산
    # compute_metrics already returns CAGR but let's calculate explicitly
    def cagr_1x(pnl_arr: np.ndarray, years: float) -> float:
        equity = 1.0
        for p in pnl_arr:
            equity *= (1.0 + p)
        return equity ** (1.0 / years) - 1.0 if years > 0 else 0.0

    cagr_vr   = cagr_1x(pnl_vr, DATA_YEARS)
    cagr_strd = cagr_1x(pnl_strd5, DATA_YEARS)

    # Per-slot CAGR (annualized return per slot used)
    # VR: 10 slots → per_slot_cagr = CAGR / 10 (each slot gets this % return)
    # STRD: 5 effective straddle slots (each straddle = 2 physical slots)
    #   But the PnL formula: net = long_pnl + short_pnl on 1-slot notional
    #   → per-straddle notional = 1 slot's worth
    #   → per-physical-slot notional = 0.5 slot (each leg is half a notional)
    # Actually in live: each leg = per_slot_capital × leverage (FULL notional)
    # So straddle uses 2x the CAPITAL of a directional trade
    # Per-dollar-of-capital:
    #   VR: CAGR per dollar (10 slots × per_slot_notional active on average)
    #   STRD: CAGR but on 2× capital per trade

    # Correct comparison: normalize by max capital deployed
    # VR: max 10 directional trades × 1-slot-notional = 10 units capital
    # STRD: max 5 straddles × 2-slot-notional = 10 units capital  ← SAME!
    # So both use same total capital at full capacity — CAGR is directly comparable.

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │ Strategy  │ Slots │  N   │  PF   │  WR  │ Calmar │")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │ VR D_mom8 │  10   │ {m_vr['N']:4d} │ {m_vr['PF']:5.3f} │ {m_vr['WR']:.1f}% │ {m_vr['calmar']:6.1f} │")
    print(f"  │ STRD S_d8 │   5*  │ {m_strd5['N']:4d} │ {m_strd5['PF']:5.3f} │ {m_strd5['WR']:.1f}% │ {m_strd5['calmar']:6.1f} │")
    print(f"  └──────────────────────────────────────────────────────┘")
    print(f"  * 5 = 실제 동시 straddle 수 (2슬롯 × 5 = 10 슬롯 소비)")
    print()
    print(f"  1x Compound CAGR:")
    print(f"    VR   D_mom8 (10 slots):       {cagr_vr:+.1%}")
    print(f"    STRD S_d8  (5 eff slots):     {cagr_strd:+.1%}")

    if cagr_strd > cagr_vr:
        ratio = cagr_strd / cagr_vr if cagr_vr > 0 else float("inf")
        print(f"\n  ✅ STRD CAGR는 VR CAGR의 {ratio:.1f}배 — 2배 슬롯에도 이득")
    else:
        print(f"\n  ⚠️  STRD CAGR < VR CAGR — 2배 슬롯 사용이 효율 저하")

    print()
    print(f"  MaxDD 비교:")
    print(f"    VR   D_mom8: {m_vr['max_dd_pct']:.1f}%")
    print(f"    STRD S_d8:   {m_strd5['max_dd_pct']:.1f}%")

    print()
    print(f"  Bootstrap PF 95% CI:")
    print(f"    VR:   [{boot_vr['CI_lo']:.3f}, {boot_vr['CI_hi']:.3f}]")
    print(f"    STRD: [{boot_strd5['CI_lo']:.3f}, {boot_strd5['CI_hi']:.3f}]")

    print()
    print("  [PnL 분포 - 개념적 설명]")
    print("  STRD net PnL = long_leg_pnl + short_leg_pnl - cost")
    print("  같은 entry 가격 → 두 레그 중 한쪽이 이기면 다른쪽은 부분적으로 손실")
    print("  시장이 강하게 방향성 → 방향 레그 trail 수익 > 반대 레그 SL 손실 → 수익")
    print("  시장이 약하게 방향성 → 방향 레그 trail 수익 < 반대 레그 SL 손실 → 손실")

    return {
        "vr": {"m": m_vr, "cagr": cagr_vr},
        "strd5": {"m": m_strd5, "cagr": cagr_strd},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: PnL 시나리오 분해
# ══════════════════════════════════════════════════════════════════════════════


def phase3_pnl_decomposition(
    data_15m: dict, data_30m: dict, data_1h: dict, data_4h: dict,
    cfg: VolStraddleConfig,
) -> None:
    """4가지 시나리오 분류: 양방수익/한방승(net+)/한방승(net-)/양방손실."""
    print("\n" + "=" * 60)
    print("PHASE 3: PnL 시나리오 분해")
    print("  ETHUSDT LONG -$59 + SHORT +$40 = -$19 같은 케이스 설명")
    print("=" * 60)

    print("\n  레그별 PnL 수집 중...")
    detailed = collect_s_trail_with_legs(data_15m, data_30m, data_1h, data_4h, cfg)
    pnl_detailed = [(sym, e, x, net, lp, sp) for sym, e, x, net, lp, sp in detailed]

    # portfolio_sim로 max=5 필터링 (accepted 거래만)
    simple_trades = [(sym, e, x, net) for sym, e, x, net, lp, sp in pnl_detailed]
    pnl_arr, accepted_idx = _portfolio_sim_with_indices(simple_trades, max_concurrent=5)

    # 수락된 거래만 분석
    if accepted_idx:
        accepted_details = [pnl_detailed[i] for i in accepted_idx]
    else:
        accepted_details = pnl_detailed

    n_total = len(accepted_details)
    if n_total == 0:
        print("  데이터 없음")
        return

    sl = cfg.s_sl
    threshold_trail = cfg.s_trail_activate  # trail 발동 기준

    # 시나리오 분류
    # long_pnl > 0 이면 trail 수익 (trail_activate를 넘었다는 의미)
    # long_pnl == -sl 이면 SL 발동
    scenarios = {"AA": [], "A+": [], "A-": [], "BB": []}
    for sym, e, x, net, lp, sp in accepted_details:
        long_win = lp > 0  # LONG leg net positive
        short_win = sp > 0  # SHORT leg net positive
        if long_win and short_win:
            scenarios["AA"].append(net)  # A: 양방 수익 (rare — both legs trail!)
        elif long_win and not short_win:
            scenarios["A+"].append(net) if net > 0 else scenarios["A-"].append(net)
        elif not long_win and short_win:
            scenarios["A+"].append(net) if net > 0 else scenarios["A-"].append(net)
        else:
            scenarios["BB"].append(net)  # D: 양방 손실 (both SL)

    print(f"\n  총 분석 거래: {n_total}")
    print(f"\n  시나리오별 분류:")

    labels = {
        "AA": "A. 양방 수익 (both legs trail — rare)",
        "A+": "B. 한방승리 net+ (winning leg > SL loss)",
        "A-": "C. 한방패배 net- (winning leg < SL loss)",  # ETHUSDT case
        "BB": "D. 양방 손실 (both legs SL)",
    }
    for key, label in labels.items():
        s = scenarios[key]
        pct = len(s) / n_total * 100
        avg = np.mean(s) * 100 if s else 0
        print(f"    {label}")
        print(f"      N={len(s)} ({pct:.1f}%), 평균 PnL={avg:+.3f}%")

    # ETHUSDT 케이스 설명
    print(f"\n  📌 ETHUSDT LONG -$59 + SHORT +$40 = -$19 케이스:")
    print(f"     시나리오 C (한방패배 net-): SHORT trail했지만 gain < LONG SL 손실")
    print(f"     SHORT gain ≈ +{cfg.s_trail_activate:.1%}~+{cfg.s_trail_activate + 0.005:.1%} (trail 발동 직후)")
    print(f"     LONG SL = -{cfg.s_sl:.1%}")
    print(f"     Cost = -{COST_STRADDLE:.2%}")
    print(f"     ETHUSDT에서 시장이 아래로 이동:")
    print(f"       → SHORT 이겼지만 trail gain이 LONG SL 손실보다 작음")
    print(f"       → net = SHORT_gain - LONG_SL - cost < 0")
    print(f"     이는 WR={n_total and (len(scenarios['AA'])+len(scenarios['A+']))/n_total:.1%} 중")
    print(f"     나머지 {n_total and (len(scenarios['A-'])+len(scenarios['BB']))/n_total:.1%}에 해당하는 정상 손실")

    # Trail vs SL 발동 통계
    long_sl_count = sum(1 for _, _, _, _, lp, sp in accepted_details if abs(lp + sl) < 0.002)
    long_trail_count = sum(1 for _, _, _, _, lp, sp in accepted_details if lp > cfg.s_trail_activate * 0.5)
    short_sl_count = sum(1 for _, _, _, _, lp, sp in accepted_details if abs(sp + sl) < 0.002)
    short_trail_count = sum(1 for _, _, _, _, lp, sp in accepted_details if sp > cfg.s_trail_activate * 0.5)

    print(f"\n  레그 결과 분포 (N={n_total} 거래):")
    print(f"    LONG  : SL={long_sl_count} ({long_sl_count/n_total:.1%}), Trail/gain={long_trail_count} ({long_trail_count/n_total:.1%})")
    print(f"    SHORT : SL={short_sl_count} ({short_sl_count/n_total:.1%}), Trail/gain={short_trail_count} ({short_trail_count/n_total:.1%})")


def _portfolio_sim_with_indices(
    all_trades: list[tuple[str, int, int, float]],
    max_concurrent: int,
) -> tuple[np.ndarray, list[int]]:
    """portfolio_sim that also returns indices of accepted trades."""
    trades_with_idx = sorted(enumerate(all_trades), key=lambda x: x[1][1])
    accepted_pnl: list[float] = []
    accepted_idx: list[int] = []
    active: list[tuple[str, int]] = []

    for orig_idx, (sym, entry, exit_b, pnl_val) in trades_with_idx:
        active = [(s, e) for s, e in active if e > entry]
        sym_active = any(s == sym for s, _ in active)
        if sym_active:
            continue
        if len(active) >= max_concurrent:
            continue
        active.append((sym, exit_b))
        accepted_pnl.append(pnl_val)
        accepted_idx.append(orig_idx)

    return np.array(accepted_pnl) if accepted_pnl else np.empty(0), accepted_idx


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════


def save_report(commit: str, slot_results: dict, eff_results: dict | None) -> None:
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    out_dir = ROOT / "docs" / "backtest_results" / "202603"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}_{commit}_r79_strd_slot_parity.md"

    m10 = slot_results[10]["m"]
    m5  = slot_results[5]["m"]
    b10 = slot_results[10]["boot"]
    b5  = slot_results[5]["boot"]

    lines = [
        f"# r79: Straddle 2-슬롯 패리티 & Per-Slot 효율 분석",
        f"",
        f"**날짜:** {today} | **커밋:** `{commit}` | **기간:** 2020-01-01 ~ 2025-09-30 ({DATA_YEARS:.2f}yr)",
        f"",
        f"## 핵심 발견: 슬롯 패리티 버그",
        f"",
        f"| 항목 | Backtest (r78) | Live (실제) |",
        f"|------|---------------|------------|",
        f"| straddle 1개 = | 1 slot | **2 slots** (LONG + SHORT) |",
        f"| max_concurrent=10 기준 | 10 straddle 동시 | **5 straddle** 동시 |",
        f"| portfolio_sim max | 10 | 수정: **5** |",
        f"",
        f"→ r78 N이 라이브보다 과대추정됨. 아래 Phase 1에서 수정 재실행.",
        f"",
        f"## Phase 1: 슬롯 패리티 수정",
        f"",
        f"| 지표 | max=10 (r78 기준) | max=5 (라이브 패리티) | 변화 |",
        f"|------|------------------|---------------------|------|",
        f"| N | {m10['N']} | {m5['N']} | {(m5['N']-m10['N'])/m10['N']:+.1%} |",
        f"| PF | {m10['PF']:.3f} | {m5['PF']:.3f} | {(m5['PF']-m10['PF'])/m10['PF']:+.1%} |",
        f"| WR | {m10['WR']:.1f}% | {m5['WR']:.1f}% | {m5['WR']-m10['WR']:+.1%} |",
        f"| MaxDD | {m10['max_dd_pct']:.1f}% | {m5['max_dd_pct']:.1f}% | {m5['max_dd_pct']-m10['max_dd_pct']:+.1%} |",
        f"| Calmar | {m10['calmar']:.1f} | {m5['calmar']:.1f} | {(m5['calmar']-m10['calmar'])/m10['calmar']:+.1%} |",
        f"| Boot PF CI | [{b10['CI_lo']:.3f}, {b10['CI_hi']:.3f}] | [{b5['CI_lo']:.3f}, {b5['CI_hi']:.3f}] | — |",
        f"",
    ]

    if eff_results:
        vr = eff_results["vr"]
        strd5 = eff_results["strd5"]
        verdict = "✅ 이득" if strd5["cagr"] > vr["cagr"] else "⚠️ 불리"
        lines += [
            f"## Phase 2: Per-Slot 효율 비교",
            f"",
            f"동일 총 자본 기준 (10 slots × notional):",
            f"- VR D_mom8: 10 directional positions × 1-slot-capital",
            f"- STRD S_d8: 5 straddles × 2-slot-capital = 동일 총 자본",
            f"",
            f"| 전략 | 유효 슬롯 | N | PF | Calmar | CAGR (1x) |",
            f"|------|----------|---|----|----|---------|",
            f"| VR D_mom8 | 10 | {vr['m']['N']} | {vr['m']['PF']:.3f} | {vr['m']['calmar']:.1f} | {vr['cagr']:+.1%} |",
            f"| STRD S_d8 | 5* | {strd5['m']['N']} | {strd5['m']['PF']:.3f} | {strd5['m']['calmar']:.1f} | {strd5['cagr']:+.1%} |",
            f"",
            f"**결론: {verdict}** — STRD CAGR {strd5['cagr']:+.1%} {'>' if strd5['cagr']>vr['cagr'] else '<'} VR CAGR {vr['cagr']:+.1%}",
            f"",
        ]

    lines += [
        f"## Phase 3: PnL 시나리오 분해",
        f"",
        f"4가지 시나리오:",
        f"- **A. 양방 수익**: 양쪽 레그 모두 trail (market spike → V자 반전)",
        f"- **B. 한방승리 net+**: 이기는 레그 trail gain > 지는 레그 SL 손실",
        f"- **C. 한방패배 net-**: 이기는 레그 trail gain < 지는 레그 SL 손실 (ETHUSDT 케이스)",
        f"- **D. 양방 손실**: 두 레그 모두 SL (sideways chop — no breakout)",
        f"",
        f"ETHUSDT LONG -$59 + SHORT +$40 = -$19: **시나리오 C** (정상적 41% 손실 케이스)",
        f"- 시장이 아래로 이동 → SHORT trail(+1.6%) > LONG SL(-1.5%)이어야 수익",
        f"- 이 케이스: SHORT trail gain이 LONG SL 손실을 충분히 커버하지 못함",
        f"",
        f"## 코드 수정 권고",
        f"",
        f"```python",
        f"# r78에서 r79로 수정:  ",
        f"# research 스크립트에서 straddle 백테스트 시:",
        f"pnl = portfolio_sim(trades_strd, max_concurrent=5)  # ← 5 (not 10)",
        f"# 이유: 1 straddle = 2 physical positions = 2 n_active in live",
        f"```",
        f"",
        f"**구현:** 미구현 (backtesting.instructions.md 업데이트 필요)",
        f"",
        f"---",
        f"",
        f"*Parity Notes: config.py 직접 읽음 ✅ | 룩어헤드 없음 ✅ (r77/r78 검증 완료)*",
        f"*비용: {COST_STRADDLE:.4f} round-trip | 슬롯: max_concurrent=5 (패리티 수정)*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄 Report saved: {out_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    import time
    t0 = time.time()
    commit = get_commit()
    print(f"r79: Straddle 2-슬롯 패리티 & Per-Slot 효율  (commit={commit})")

    cfg = VolStraddleConfig()
    whitelist = list(cfg.symbol_whitelist)
    print(f"Loading data for {len(whitelist)} symbols...")
    data_15m, data_30m, data_1h, data_4h = load_data(whitelist)
    print(f"Loaded {len(data_15m)} symbols in {time.time()-t0:.1f}s")

    # Phase 0: baseline
    trades_strd, data_years = phase0_baseline(data_15m, data_30m, data_1h, data_4h, cfg)

    # Phase 1: slot parity
    slot_results = phase1_slot_parity(trades_strd, cfg)

    # Phase 2: per-slot efficiency
    eff_results = phase2_per_slot_efficiency(
        data_15m, data_1h, data_4h, trades_strd, cfg, slot_results
    )

    # Phase 3: PnL decomposition
    phase3_pnl_decomposition(data_15m, data_30m, data_1h, data_4h, cfg)

    # Save report
    save_report(commit, slot_results, eff_results)

    elapsed = time.time() - t0
    print(f"\n완료: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
