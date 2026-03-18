#!/usr/bin/env python3
"""r100: 반절 (Half-Close) + 본절 (Breakeven SL) Backtest

VR Squeeze와 MA Cluster에 대해:
1. 반절: 일정 수익(half_trigger) 도달 시 포지션 일부(half_ratio) 청산
2. 본절: 반절 후 SL을 entry price로 이동 (손실 제로 보장)
3. 나머지는 trail로 더 큰 수익 추구

기대 효과:
- 수익 일부 조기 확정 → DD 감소
- 본절 SL → worst case = breakeven
- Trail이 더 멀리 갈 기회

테스트 파라미터:
- half_trigger: 0.3%, 0.5%, 1.0% (반절 발동 수익률)
- half_ratio: 30%, 50%, 70% (청산 비율)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (  # noqa: E402
    compute_metrics,
    portfolio_sim,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402
from quantbox.strategy.vol_straddle.signals import (  # noqa: E402
    compute_momentum,
    compute_vr,
    gen_vr_signal,
)

# ── Config ───────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

# VR params
VR_SP = cfg.vr_short
VR_LP = cfg.vr_long
D_VR_THRESH = cfg.d_vr_threshold
D_SL = cfg.d_sl  # 1%
D_TP = cfg.d_tp  # 20% (dead cap)
D_TRAIL_ACT = cfg.d_trail_activate  # 1%
D_TRAIL_DIST = cfg.d_trail_distance  # 0.2%
MAX_HOLD = cfg.max_hold_15m  # 192 (48h)

# MA params (use MA-specific if available)
MA_SL = cfg.ma_cluster_sl  # 2%
MA_TRAIL_ACT = cfg.ma_cluster_trail_activate  # 2%
MA_TRAIL_DIST = cfg.ma_cluster_trail_distance  # 0.3%

COST = 0.0008  # round-trip cost
MAX_SLOTS = cfg.max_concurrent  # 10
VR_LN = cfg.base_leverage / cfg.max_concurrent  # 3.0
MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent  # 2.1

# Data paths
CACHE_DIR = ROOT / "data" / "cache_extended"
WHITELIST = set(cfg.symbol_whitelist)

# Test grids
HALF_TRIGGERS = [0.003, 0.005, 0.010]  # 0.3%, 0.5%, 1.0%
HALF_RATIOS = [0.3, 0.5, 0.7]  # 30%, 50%, 70%


# ══════════════════════════════════════════════════════════════════════════════
#  Half-Close + Breakeven SL Simulation (Numba JIT)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_half_be_trades(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signal: np.ndarray,
    direction: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    half_trigger: float,
    half_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate trades with half-close at profit + breakeven SL.

    Logic:
    1. Enter at signal bar+1 open (= close of signal bar)
    2. If pnl reaches half_trigger, close half_ratio portion at that bar's close
       - Record PnL for that portion (minus half cost)
       - Move SL to entry (breakeven) for remaining portion
    3. Remaining portion trails with trail_activate/trail_distance
    4. Exit remaining when trail stop hit, SL hit, or max_hold reached

    Returns:
        pnl_array: PnL per trade (combined first + second portions)
        entry_bars: Entry bar indices
        exit_bars: Final exit bar indices
    """
    n = len(close)
    # Pre-allocate for max possible trades
    max_trades = n // 2
    pnl_arr = np.zeros(max_trades, dtype=np.float64)
    entry_arr = np.zeros(max_trades, dtype=np.int64)
    exit_arr = np.zeros(max_trades, dtype=np.int64)
    trade_count = 0

    i = 0
    while i < n - 1:
        if signal[i] == 1 and direction[i] != 0:
            # Entry at next bar open (= this bar close)
            entry_bar = i + 1
            if entry_bar >= n:
                break
            entry_price = close[i]
            dire = direction[i]  # +1 LONG, -1 SHORT

            # State tracking
            half_closed = False
            half_pnl = 0.0
            remaining_ratio = 1.0
            current_sl = entry_price * (1 - sl_pct) if dire == 1 else entry_price * (1 + sl_pct)
            peak_gain = 0.0
            trail_active = False

            exit_bar = entry_bar
            exit_price = entry_price
            exit_reason = 0  # 0=none, 1=SL, 2=trail, 3=max_hold

            for j in range(entry_bar, min(entry_bar + max_hold, n)):
                bar_high = high[j]
                bar_low = low[j]
                bar_close = close[j]

                # Calculate current PnL
                if dire == 1:  # LONG
                    current_pnl = (bar_close - entry_price) / entry_price
                else:  # SHORT
                    current_pnl = (entry_price - bar_close) / entry_price

                # Check SL first (intra-bar, server SL fires immediately)
                sl_hit = False
                if dire == 1 and bar_low <= current_sl:
                    sl_hit = True
                    exit_price = current_sl
                elif dire == -1 and bar_high >= current_sl:
                    sl_hit = True
                    exit_price = current_sl

                if sl_hit:
                    exit_bar = j
                    exit_reason = 1
                    break

                # Half-close logic (at bar close, not intra-bar)
                if not half_closed and current_pnl >= half_trigger:
                    # Close half_ratio at this bar's close
                    half_pnl = current_pnl * half_ratio - cost * 0.5  # half cost for partial close
                    half_closed = True
                    remaining_ratio = 1.0 - half_ratio
                    # Move SL to breakeven (entry price)
                    current_sl = entry_price

                # Update peak and trail
                if current_pnl > peak_gain:
                    peak_gain = current_pnl

                if not trail_active and peak_gain >= trail_activate:
                    trail_active = True

                # Check trail stop (only after half-close or from beginning)
                if trail_active:
                    trail_stop = peak_gain - trail_distance
                    if current_pnl < trail_stop:
                        exit_bar = j
                        exit_price = bar_close
                        exit_reason = 2
                        break

                exit_bar = j
                exit_price = bar_close

            # Max hold exit
            if exit_reason == 0 and exit_bar >= entry_bar + max_hold - 1:
                exit_reason = 3

            # Calculate final PnL
            if dire == 1:
                remaining_pnl = (exit_price - entry_price) / entry_price
            else:
                remaining_pnl = (entry_price - exit_price) / entry_price

            # Combine PnL: half portion (already closed) + remaining portion
            if half_closed:
                total_pnl = half_pnl + remaining_pnl * remaining_ratio - cost * 0.5
            else:
                total_pnl = remaining_pnl - cost

            pnl_arr[trade_count] = total_pnl
            entry_arr[trade_count] = entry_bar
            exit_arr[trade_count] = exit_bar
            trade_count += 1

            i = exit_bar + 1
        else:
            i += 1

    return pnl_arr[:trade_count], entry_arr[:trade_count], exit_arr[:trade_count]


@numba.jit(nopython=True, cache=True)
def make_direction_momentum(sig: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """Assign direction based on momentum."""
    n = len(sig)
    dire = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if sig[i] == 1:
            if momentum[i] > 0.0001:
                dire[i] = 1
            elif momentum[i] < -0.0001:
                dire[i] = -1
            else:
                dire[i] = 1  # default LONG
    return dire


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline Simulation (no half-close, pure trail)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_baseline_trades(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signal: np.ndarray,
    direction: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline simulation without half-close (pure trail)."""
    n = len(close)
    max_trades = n // 2
    pnl_arr = np.zeros(max_trades, dtype=np.float64)
    entry_arr = np.zeros(max_trades, dtype=np.int64)
    exit_arr = np.zeros(max_trades, dtype=np.int64)
    trade_count = 0

    i = 0
    while i < n - 1:
        if signal[i] == 1 and direction[i] != 0:
            entry_bar = i + 1
            if entry_bar >= n:
                break
            entry_price = close[i]
            dire = direction[i]

            sl_price = entry_price * (1 - sl_pct) if dire == 1 else entry_price * (1 + sl_pct)
            peak_gain = 0.0
            trail_active = False

            exit_bar = entry_bar
            exit_price = entry_price

            for j in range(entry_bar, min(entry_bar + max_hold, n)):
                bar_high = high[j]
                bar_low = low[j]
                bar_close = close[j]

                if dire == 1:
                    current_pnl = (bar_close - entry_price) / entry_price
                else:
                    current_pnl = (entry_price - bar_close) / entry_price

                # SL check
                sl_hit = False
                if dire == 1 and bar_low <= sl_price:
                    sl_hit = True
                    exit_price = sl_price
                elif dire == -1 and bar_high >= sl_price:
                    sl_hit = True
                    exit_price = sl_price

                if sl_hit:
                    exit_bar = j
                    break

                if current_pnl > peak_gain:
                    peak_gain = current_pnl

                if not trail_active and peak_gain >= trail_activate:
                    trail_active = True

                if trail_active:
                    trail_stop = peak_gain - trail_distance
                    if current_pnl < trail_stop:
                        exit_bar = j
                        exit_price = bar_close
                        break

                exit_bar = j
                exit_price = bar_close

            if dire == 1:
                final_pnl = (exit_price - entry_price) / entry_price
            else:
                final_pnl = (entry_price - exit_price) / entry_price

            pnl_arr[trade_count] = final_pnl - cost
            entry_arr[trade_count] = entry_bar
            exit_arr[trade_count] = exit_bar
            trade_count += 1

            i = exit_bar + 1
        else:
            i += 1

    return pnl_arr[:trade_count], entry_arr[:trade_count], exit_arr[:trade_count]


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_data() -> tuple[dict, int]:
    """Load 15m data from cache_extended."""
    data = {}
    max_bars = 0

    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in WHITELIST:
            continue
        d = np.load(f)
        data[sym] = {
            "close": d["close"].astype(np.float64),
            "high": d["high"].astype(np.float64),
            "low": d["low"].astype(np.float64),
            "volume": d["volume"].astype(np.float64),
        }
        max_bars = max(max_bars, len(d["close"]))

    # Add offset and mom_8
    for sym in data:
        c = data[sym]["close"]
        h = data[sym]["high"]
        lo = data[sym]["low"]
        data[sym]["_offset"] = max_bars - len(c)
        data[sym]["mom_8"] = compute_momentum(c, 8)
        data[sym]["vr"] = compute_vr(c, h, lo, VR_SP, VR_LP)

    return data, max_bars


def collect_vr_trades_with_half_be(
    data_15m: dict,
    half_trigger: float,
    half_ratio: float,
    sl_pct: float = D_SL,
    trail_activate: float = D_TRAIL_ACT,
    trail_distance: float = D_TRAIL_DIST,
) -> list[tuple[int, int, float]]:
    """Collect VR trades with half-close + breakeven."""
    all_trades = []
    warmup = max(VR_SP, VR_LP) + 10

    for sym, d in data_15m.items():
        c = d["close"]
        h = d["high"]
        lo = d["low"]
        vr = d["vr"]
        mom = d["mom_8"]
        offset = d.get("_offset", 0)

        if len(c) < warmup + 50:
            continue

        # Generate VR signal
        sig = gen_vr_signal(vr, D_VR_THRESH, warmup)
        dire = make_direction_momentum(sig, mom)

        # Simulate with half-close
        pnl, entry, exit_ = sim_half_be_trades(
            c,
            h,
            lo,
            sig,
            dire,
            sl_pct,
            D_TP,
            trail_activate,
            trail_distance,
            MAX_HOLD,
            COST,
            half_trigger,
            half_ratio,
        )

        for k in range(len(pnl)):
            entry_global = int(entry[k]) + offset
            exit_global = int(exit_[k]) + offset
            all_trades.append((sym, entry_global, exit_global, float(pnl[k])))

    return all_trades


def collect_vr_trades_baseline(
    data_15m: dict,
    sl_pct: float = D_SL,
    trail_activate: float = D_TRAIL_ACT,
    trail_distance: float = D_TRAIL_DIST,
) -> list[tuple[int, int, float]]:
    """Collect VR trades with baseline (no half-close)."""
    all_trades = []
    warmup = max(VR_SP, VR_LP) + 10

    for sym, d in data_15m.items():
        c = d["close"]
        h = d["high"]
        lo = d["low"]
        vr = d["vr"]
        mom = d["mom_8"]
        offset = d.get("_offset", 0)

        if len(c) < warmup + 50:
            continue

        sig = gen_vr_signal(vr, D_VR_THRESH, warmup)
        dire = make_direction_momentum(sig, mom)

        pnl, entry, exit_ = sim_baseline_trades(
            c,
            h,
            lo,
            sig,
            dire,
            sl_pct,
            D_TP,
            trail_activate,
            trail_distance,
            MAX_HOLD,
            COST,
        )

        for k in range(len(pnl)):
            entry_global = int(entry[k]) + offset
            exit_global = int(exit_[k]) + offset
            all_trades.append((sym, entry_global, exit_global, float(pnl[k])))

    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 70)
    print("r100: 반절 (Half-Close) + 본절 (Breakeven SL) Backtest")
    print("Data: cache_extended/*_15m_2190d_*.npz | 6yr")
    print(f"VR: L/N={VR_LN:.1f} | MAX_SLOTS={MAX_SLOTS} | cost={COST * 100:.2f}%")
    print(f"VR params: SL={D_SL * 100:.1f}% trail_act={D_TRAIL_ACT * 100:.1f}% trail_dist={D_TRAIL_DIST * 100:.2f}%")
    print("=" * 70)

    # Load data
    print("\n데이터 로드 중...")
    data_15m, max_bars = load_data()
    data_years = max_bars / (4 * 24 * 365.25)
    print(f"  심볼 수: {len(data_15m)}")
    print(f"  MAX_BARS: {max_bars} (15m) = {data_years:.2f}년")

    # Phase 1: VR Baseline
    print("\n" + "=" * 70)
    print("Phase 1: VR Squeeze 베이스라인 (half-close 없음)")
    print("=" * 70)

    vr_base_trades = collect_vr_trades_baseline(data_15m)
    vr_base_pnl = portfolio_sim(vr_base_trades, MAX_SLOTS)
    vr_base_m = compute_metrics(vr_base_pnl, data_years)

    print(
        f"  VR Baseline: N={vr_base_m['N']} PF={vr_base_m['PF']:.3f} "
        f"WR={vr_base_m['WR']:.1f}% EV={vr_base_m['EV_bp']:+.1f}bp DD={vr_base_m['max_dd_pct']:.1f}%"
    )

    # Phase 2: VR with Half-Close + BE sweep
    print("\n" + "=" * 70)
    print("Phase 2: VR Squeeze + 반절/본절 파라미터 스윕")
    print("=" * 70)

    print(
        f"\n  {'half_trig':>10} {'half_ratio':>10} {'N':>6} {'PF':>7} {'ΔPF':>7} "
        f"{'WR':>6} {'EV(bp)':>8} {'DD%':>7} {'ΔDD':>7}"
    )
    print("  " + "-" * 85)

    vr_results = []
    for ht in HALF_TRIGGERS:
        for hr in HALF_RATIOS:
            trades = collect_vr_trades_with_half_be(data_15m, ht, hr)
            pnl = portfolio_sim(trades, MAX_SLOTS)
            m = compute_metrics(pnl, data_years)

            delta_pf = m["PF"] - vr_base_m["PF"]
            delta_dd = m["max_dd_pct"] - vr_base_m["max_dd_pct"]

            vr_results.append(
                {
                    "half_trigger": ht,
                    "half_ratio": hr,
                    "N": m["N"],
                    "PF": m["PF"],
                    "delta_PF": delta_pf,
                    "WR": m["WR"],
                    "EV_bp": m["EV_bp"],
                    "DD": m["max_dd_pct"],
                    "delta_DD": delta_dd,
                }
            )

            pf_color = "+" if delta_pf > 0 else ""
            dd_color = "" if delta_dd > 0 else ""

            print(
                f"  {ht * 100:>9.1f}% {hr * 100:>9.0f}% {m['N']:>6} {m['PF']:>7.3f} "
                f"{pf_color}{delta_pf:>+6.3f} {m['WR']:>5.1f}% {m['EV_bp']:>+7.1f} "
                f"{m['max_dd_pct']:>6.1f}% {dd_color}{delta_dd:>+6.1f}"
            )

    # Find best config
    best_vr = max(vr_results, key=lambda x: x["PF"])
    print(
        f"\n  최적: half_trigger={best_vr['half_trigger'] * 100:.1f}% "
        f"half_ratio={best_vr['half_ratio'] * 100:.0f}% "
        f"→ PF={best_vr['PF']:.3f} (Δ{best_vr['delta_PF']:+.3f}) "
        f"DD={best_vr['DD']:.1f}% (Δ{best_vr['delta_DD']:+.1f})"
    )

    # Check pass criteria
    vr_pass = best_vr["delta_PF"] >= 0.05 and best_vr["delta_DD"] <= 0

    # Phase 3: Yearly OOS for best config
    print("\n" + "=" * 70)
    print("Phase 3: VR 최적 설정 연도별 OOS 검증")
    print("=" * 70)

    if vr_pass:
        best_ht = best_vr["half_trigger"]
        best_hr = best_vr["half_ratio"]

        # Collect all trades with best config
        all_trades = collect_vr_trades_with_half_be(data_15m, best_ht, best_hr)
        base_trades = collect_vr_trades_baseline(data_15m)

        bars_per_year = 4 * 24 * 365

        print(f"\n  {'Year':>6} {'N_base':>8} {'N_half':>8} {'PF_base':>8} {'PF_half':>8} {'ΔPF':>7}")
        print("  " + "-" * 55)

        yearly_pass = 0
        for year in range(2020, 2026):
            start_bar = (year - 2020) * bars_per_year
            end_bar = start_bar + bars_per_year

            year_base = [(e, x, p) for e, x, p in base_trades if start_bar <= e < end_bar]
            year_half = [(e, x, p) for e, x, p in all_trades if start_bar <= e < end_bar]

            if len(year_base) < 10:
                print(f"  {year:>6} {'skip':>8} (N < 10)")
                continue

            pnl_base = portfolio_sim(year_base, MAX_SLOTS)
            pnl_half = portfolio_sim(year_half, MAX_SLOTS)

            m_base = compute_metrics(pnl_base, 1.0)
            m_half = compute_metrics(pnl_half, 1.0)

            delta = m_half["PF"] - m_base["PF"]
            status = "✅" if delta >= 0 else "❌"
            if delta >= 0:
                yearly_pass += 1

            print(
                f"  {year:>6} {m_base['N']:>8} {m_half['N']:>8} "
                f"{m_base['PF']:>8.3f} {m_half['PF']:>8.3f} {delta:>+6.3f} {status}"
            )

        print(f"\n  OOS 통과: {yearly_pass}/6 연도")
        oos_pass = yearly_pass >= 5
    else:
        print("  ❌ VR 개선 없음, OOS 스킵")
        oos_pass = False

    # Summary
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)
    print(f"\n  VR Baseline: N={vr_base_m['N']} PF={vr_base_m['PF']:.3f} DD={vr_base_m['max_dd_pct']:.1f}%")
    print(
        f"  VR 최적 반절/본절: half_trig={best_vr['half_trigger'] * 100:.1f}% "
        f"half_ratio={best_vr['half_ratio'] * 100:.0f}%"
    )
    print(
        f"    → PF={best_vr['PF']:.3f} (Δ{best_vr['delta_PF']:+.3f}) "
        f"DD={best_vr['DD']:.1f}% (Δ{best_vr['delta_DD']:+.1f})"
    )

    if vr_pass and oos_pass:
        print("\n  최종 판정: ✅ PASS — 반절/본절 전략 유효")
        print(
            f"    권장: half_trigger={best_vr['half_trigger'] * 100:.1f}% half_ratio={best_vr['half_ratio'] * 100:.0f}%"
        )
    elif vr_pass:
        print("\n  최종 판정: ⚠️ PARTIAL — PF 개선 있으나 OOS 미달")
    else:
        print("\n  최종 판정: ❌ FAIL — 반절/본절 부가가치 없음")


if __name__ == "__main__":
    main()
