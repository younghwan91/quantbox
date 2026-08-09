"""R44d — MA Cluster Squeeze: Refined Variants

R44c finding:
  V11 (MA Cluster Breakout) → 30/36 configs pass PF ≥ 1.0.
  Best: cluster=1% sl=2% tp=8% → PF=1.155, EV=+21.5bp, N=23,353.
  BUT: 13+ trades/day, maxDD=99.7%. Signal is statistically valid but too NOISY.

  V12-V15 (MA Bounce, MTF Pullback, VWMA, Perfect Order) → ALL fail.
  Bounce-off-MA signals have no edge in crypto.

Insight: MA cluster = compression → expansion. Same structural phenomenon as
VR squeeze, measured differently. But V11 fires on EVERY bar where MAs are
clustered and price is outside. Needs selectivity.

Refined vectors:
  V16: MA Cluster + Duration Filter + Momentum Direction
       — Require cluster to persist for ≥ D bars before entry (like sq_dur ≥ 4).
       — Use momentum for direction (like D_mom8 in VR squeeze).
       — Reset after each trade (one signal per cluster episode).
  V17: V16 but 4h-only (SMA200 on 4h = 33 days → very meaningful clusters)
  V18: MA Cluster + VR complement (V16 + restrict to VR > 0.5 periods only)
  V19: MA Proximity to Cluster (price within tight range of clustered MAs)

Goal: Get N down to ~2000-5000 range with PF > 1.1 and maxDD < 50%.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.active.r31_r40.r31_mtf_vol_straddle import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    compute_momentum,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    collect_trades as r38_collect_trades,
    COST_DIRECTIONAL, G_3TF, CFG_V13,
)
from research.active.r41_r45.r44c_ma_confluence import (
    compute_sma, compute_vwma, precompute_mas,
)
from research.active.r41_r45.r44_high_vol_complement import (
    _dedup_sym_trades,
)


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  V16: MA Cluster Squeeze — Duration + Momentum (refined V11)
#
#  Key differences from V11:
#    1. Duration filter: cluster must persist ≥ min_duration bars
#    2. Momentum direction: use momentum(8) instead of price position
#    3. One signal per cluster episode: reset after each trade
#  → Much more selective, should reduce 23K → 2-5K trades
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_squeeze(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MA cluster squeeze with duration filter and momentum direction.

    Track how long all 4 MAs have been within cluster_pct of each other.
    Once duration >= min_duration AND close breaks out: enter with momentum dir.
    Reset counter after trade (one signal per cluster episode).
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            i += 1
            continue

        # Compute MA band spread
        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            i += 1
            continue

        # Duration filter: cluster must have persisted
        if cluster_count < min_duration:
            i += 1
            continue

        # Breakout: close must be outside the cluster zone
        if close[i] <= ma_max and close[i] >= ma_min:
            i += 1
            continue

        # Momentum direction (like VR squeeze's D_mom8)
        if abs(momentum[i]) < 0.003:
            i += 1
            continue
        d = 1 if momentum[i] > 0 else -1

        entry = close[i]

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1

        # Reset cluster after trade — one signal per cluster episode
        cluster_count = 0
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V19: MA Proximity Entry (price near cluster, not breakout)
#
#  Different hypothesis: don't wait for breakout, enter WHEN price first
#  approaches the cluster zone. The MA cluster acts as a "magnet" — price
#  gets pulled toward it. Enter when price is within the cluster and
#  momentum gives direction. If all MAs are at one level and price is there
#  too, that IS "관심이 모이는 자리".
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_entry(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enter when price is WITHIN the MA cluster zone.

    Instead of waiting for breakout (V16), enter when price touches the
    cluster zone from either side. Momentum gives direction.
    The cluster IS the attention zone — enter there with tight stop.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0
    was_outside = True  # price was outside cluster → can enter on re-entry

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            was_outside = True
            i += 1
            continue

        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            was_outside = True
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue

        # Check if price is within the extended cluster zone
        zone_upper = ma_max * (1.0 + cluster_pct * 0.5)
        zone_lower = ma_min * (1.0 - cluster_pct * 0.5)
        in_zone = close[i] >= zone_lower and close[i] <= zone_upper

        if in_zone and not was_outside:
            # Already in zone, don't re-enter
            i += 1
            continue

        if not in_zone:
            was_outside = True
            i += 1
            continue

        # Price just entered the zone — signal!
        was_outside = False

        if abs(momentum[i]) < 0.003:
            i += 1
            continue
        d = 1 if momentum[i] > 0 else -1

        entry = close[i]

        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1

        cluster_count = 0
        was_outside = True
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Collect functions
# ══════════════════════════════════════════════════════════════════════════════


def collect_v16_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    cluster_pct: float, min_duration: int,
    sl: float, tp: float,
    tf_restriction: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect V16 trades: MA Cluster Squeeze with duration + momentum.

    tf_restriction: None = all TFs, 16 = 4h only, etc.
    """
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 192, 4: 48, 16: 12}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            if tf_restriction is not None and ratio != tf_restriction:
                continue
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            if "sma_200" not in d or "mom_8" not in d:
                continue
            p, e, x = sim_ma_cluster_squeeze(
                d["close"], d["high"], d["low"],
                d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"],
                d["mom_8"],
                cluster_pct, min_duration,
                sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


def collect_v18_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    cluster_pct: float, min_duration: int,
    sl: float, tp: float,
    vr_min: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V18: V16 restricted to VR > vr_min (high-vol complement mode)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 192, 4: 48, 16: 12}
    vr_key = "vr_8_56"

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            if "sma_200" not in d or "mom_8" not in d:
                continue
            vr = d.get(vr_key)
            if vr is None:
                continue
            # Run sim but pre-filter: only bars where VR > vr_min
            p, e, x = sim_ma_cluster_squeeze_vr(
                d["close"], d["high"], d["low"],
                d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"],
                d["mom_8"], vr,
                cluster_pct, min_duration, vr_min,
                sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


def collect_v19_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    cluster_pct: float, min_duration: int,
    sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V19: MA Cluster Zone Entry (not breakout)."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 192, 4: 48, 16: 12}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            if "sma_200" not in d or "mom_8" not in d:
                continue
            p, e, x = sim_ma_cluster_entry(
                d["close"], d["high"], d["low"],
                d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"],
                d["mom_8"],
                cluster_pct, min_duration,
                sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  V18 helper: sim with VR filter
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_squeeze_vr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    momentum: np.ndarray, vr: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    vr_min: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V16 but only fires when VR > vr_min (complement to VR squeeze)."""
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0

    while i < n - 1:
        if np.isnan(sma200[i]) or np.isnan(vr[i]):
            cluster_count = 0
            i += 1
            continue

        # VR complement filter: only trade when VR is high
        if vr[i] <= vr_min:
            cluster_count = 0
            i += 1
            continue

        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue

        if close[i] <= ma_max and close[i] >= ma_min:
            i += 1
            continue

        if abs(momentum[i]) < 0.003:
            i += 1
            continue
        d = 1 if momentum[i] > 0 else -1

        entry = close[i]
        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        cluster_count = 0
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V20: Fewer-MA cluster (SMA50 + SMA100 + SMA200 only)
#
#  Hypothesis: SMA20 is too fast and breaks clusters prematurely.
#  Using only the 3 slower MAs (50, 100, 200) gives more meaningful clusters.
#  On 4h: SMA50=8d, SMA100=16d, SMA200=33d — these are weekly/biweekly/monthly.
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma3_cluster_squeeze(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma50: np.ndarray, sma100: np.ndarray, sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-MA cluster (50, 100, 200) with duration + momentum.

    Using only slower MAs for more meaningful confluence zones.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            i += 1
            continue

        ma_max = max(sma50[i], max(sma100[i], sma200[i]))
        ma_min = min(sma50[i], min(sma100[i], sma200[i]))
        if close[i] <= 0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue

        if close[i] <= ma_max and close[i] >= ma_min:
            i += 1
            continue

        if abs(momentum[i]) < 0.003:
            i += 1
            continue
        d = 1 if momentum[i] > 0 else -1

        entry = close[i]
        if d == 1:
            sl_price = entry * (1.0 - sl)
            tp_price = entry * (1.0 + tp)
        else:
            sl_price = entry * (1.0 + sl)
            tp_price = entry * (1.0 - tp)

        eb = i
        end = min(i + 1 + max_hold, n)
        for j in range(i + 1, end):
            if d == 1:
                if low[j] <= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if high[j] >= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
            else:
                if high[j] >= sl_price:
                    trade_pnl = -sl
                    eb = j
                    break
                if low[j] <= tp_price:
                    trade_pnl = tp
                    eb = j
                    break
        else:
            eb = end - 1
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        cluster_count = 0
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


def collect_v20_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    cluster_pct: float, min_duration: int,
    sl: float, tp: float,
    tf_restriction: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """Collect V20: 3-MA cluster squeeze."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 192, 4: 48, 16: 12}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            if tf_restriction is not None and ratio != tf_restriction:
                continue
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            if "sma_200" not in d or "mom_8" not in d:
                continue
            p, e, x = sim_ma3_cluster_squeeze(
                d["close"], d["high"], d["low"],
                d["sma_50"], d["sma_100"], d["sma_200"],
                d["mom_8"],
                cluster_pct, min_duration,
                sl, tp, max_holds[ratio],
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((
                    int(e[k]) * ratio,
                    int(x[k]) * ratio + ratio - 1,
                    float(p[k]),
                ))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"R44d — MA Cluster Squeeze Refined [{commit}]")
    print("=" * 70)

    t0 = time.time()
    print("Loading 15m data...")
    data_15m = load_15m()
    print(f"  {len(data_15m)} symbols loaded")

    print("Resampling to 1h / 4h...")
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    print("Precomputing VR + MAs...")
    vr_pairs = [(8, 56)]
    precompute_all_vr([data_15m, data_1h, data_4h], vr_pairs)
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)
        precompute_mas(ds)

    bars_list = [d["bars"] for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list)) if bars_list else 0
    data_years = avg_bars / (4 * 24 * 365)
    print(f"  avg bars = {avg_bars}, data_years = {data_years:.1f}")
    print(f"  Load time: {time.time() - t0:.1f}s")
    print()

    # ── Baseline ──
    print("=" * 70)
    print("Baseline (CFG_V13)")
    print("=" * 70)
    bl_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_pnl = portfolio_sim(bl_trades, 10)
    bl_m = compute_metrics(bl_pnl, data_years)
    print(f"  N={bl_m['N']}  PF={bl_m['PF']:.3f}  WR={bl_m['WR']:.1f}%  "
          f"EV={bl_m['EV_bp']:+.1f}bp  DD={bl_m['max_dd_pct']:.1f}%  "
          f"Calmar={bl_m['calmar']:.1f}")
    print()

    results: list[tuple[str, dict]] = []
    PASS_THRESHOLD = 1.05

    # ══════════════════════════════════════════════════════════════════════
    #  V16: MA Cluster Squeeze (all TFs)
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 70)
    print("V16: MA Cluster Squeeze (duration + momentum, all TFs)")
    print("=" * 70)
    for cluster_pct in [0.005, 0.01, 0.015, 0.02]:
        for min_dur in [4, 8, 16, 32]:
            for sl, tp in [(0.01, 0.05), (0.01, 0.08), (0.015, 0.05),
                           (0.015, 0.08), (0.02, 0.08), (0.02, 0.12)]:
                label = (f"V16 c={cluster_pct*100:.1f}% d={min_dur} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v16_trades(
                    data_15m, data_1h, data_4h,
                    cluster_pct, min_dur, sl, tp)
                if not trades:
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"DD={m['max_dd_pct']:.1f}%  RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ══════════════════════════════════════════════════════════════════════
    #  V17: 4h-only MA Cluster Squeeze
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("V17: MA Cluster Squeeze (4h only — SMA200=33d)")
    print("=" * 70)
    for cluster_pct in [0.005, 0.01, 0.015, 0.02, 0.03]:
        for min_dur in [2, 4, 8, 16]:
            for sl, tp in [(0.01, 0.05), (0.01, 0.08), (0.015, 0.08),
                           (0.02, 0.08), (0.02, 0.12)]:
                label = (f"V17 c={cluster_pct*100:.1f}% d={min_dur} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v16_trades(
                    data_15m, data_1h, data_4h,
                    cluster_pct, min_dur, sl, tp,
                    tf_restriction=16)
                if not trades:
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"DD={m['max_dd_pct']:.1f}%  RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ══════════════════════════════════════════════════════════════════════
    #  V18: MA Cluster + VR complement (VR > 0.5 only)
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("V18: MA Cluster Squeeze → VR complement (VR > 0.5 only)")
    print("=" * 70)
    for cluster_pct in [0.005, 0.01, 0.015, 0.02]:
        for min_dur in [4, 8, 16]:
            for sl, tp in [(0.01, 0.05), (0.01, 0.08), (0.015, 0.08),
                           (0.02, 0.08), (0.02, 0.12)]:
                for vr_min in [0.5, 0.7]:
                    label = (f"V18 c={cluster_pct*100:.1f}% d={min_dur} "
                             f"vr>{vr_min} sl={sl*100:.1f}% tp={tp*100:.0f}%")
                    trades = collect_v18_trades(
                        data_15m, data_1h, data_4h,
                        cluster_pct, min_dur, sl, tp, vr_min)
                    if not trades:
                        continue
                    pnl = portfolio_sim(trades, 10)
                    m = compute_metrics(pnl, data_years)
                    flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                    print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                          f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                          f"DD={m['max_dd_pct']:.1f}%  RR={m['RR']:.2f}  {flag}")
                    results.append((label, m))

    # ══════════════════════════════════════════════════════════════════════
    #  V19: MA Cluster Zone Entry (price enters the cluster)
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("V19: MA Cluster Zone Entry (touch cluster, not breakout)")
    print("=" * 70)
    for cluster_pct in [0.01, 0.015, 0.02]:
        for min_dur in [4, 8, 16]:
            for sl, tp in [(0.01, 0.05), (0.01, 0.08), (0.015, 0.08),
                           (0.02, 0.08)]:
                label = (f"V19 c={cluster_pct*100:.1f}% d={min_dur} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v19_trades(
                    data_15m, data_1h, data_4h,
                    cluster_pct, min_dur, sl, tp)
                if not trades:
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"DD={m['max_dd_pct']:.1f}%  RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ══════════════════════════════════════════════════════════════════════
    #  V20: 3-MA Cluster (SMA 50/100/200 — slower MAs only)
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("V20: 3-MA Cluster (SMA50+100+200, drop SMA20)")
    print("=" * 70)
    for cluster_pct in [0.005, 0.01, 0.015, 0.02]:
        for min_dur in [4, 8, 16]:
            for sl, tp in [(0.01, 0.05), (0.01, 0.08), (0.015, 0.08),
                           (0.02, 0.08), (0.02, 0.12)]:
                label = (f"V20 c={cluster_pct*100:.1f}% d={min_dur} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v20_trades(
                    data_15m, data_1h, data_4h,
                    cluster_pct, min_dur, sl, tp)
                if not trades:
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"DD={m['max_dd_pct']:.1f}%  RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passing = [(l, m) for l, m in results if m["PF"] >= PASS_THRESHOLD]
    total = len(results)
    print(f"\n  Total configs: {total}")
    print(f"  Passing (PF ≥ {PASS_THRESHOLD}): {len(passing)}")

    if passing:
        # Sort by PF descending
        passing.sort(key=lambda x: -x[1]["PF"])
        print("\n  ── Top 15 by PF ──")
        for label, m in passing[:15]:
            print(f"    {label}")
            print(f"      N={m['N']:>5}  PF={m['PF']:.3f}  "
                  f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                  f"DD={m['max_dd_pct']:.1f}%  RR={m['RR']:.2f}  "
                  f"Calmar={m['calmar']:.1f}")

        # Also show top by Calmar (practical viability)
        viable = [(l, m) for l, m in passing
                  if m["max_dd_pct"] < 60 and m["N"] > 500]
        if viable:
            viable.sort(key=lambda x: -x[1]["calmar"])
            print("\n  ── Top 10 viable (DD<60%, N>500) by Calmar ──")
            for label, m in viable[:10]:
                print(f"    {label}")
                print(f"      N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"DD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.1f}")
    else:
        print("\n  No config passed.")
        if results:
            results.sort(key=lambda x: -x[1]["PF"])
            print("\n  ── Top 10 by PF (all below threshold) ──")
            for label, m in results[:10]:
                print(f"    {label}: PF={m['PF']:.3f}  N={m['N']}  "
                      f"WR={m['WR']:.1f}%  DD={m['max_dd_pct']:.1f}%")

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
    print(f"  Commit: {commit}")


if __name__ == "__main__":
    main()
