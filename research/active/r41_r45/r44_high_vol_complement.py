"""R44 — High-Volatility Complement Strategy Research

VR Squeeze (v13.1) trades only when VR < 0.5 (volatility compression).
This leaves the bot idle when volatility is already high (VR > 1.0).

Goal: Find a complementary strategy that is ACTIVE when VR squeeze is INACTIVE,
      so the combined system trades in more market conditions.

Strategy vectors to test:
  V1: VR Expansion Mean Reversion (VR > threshold → fade the overextension)
  V2: RSI Extreme Mean Reversion (RSI oversold/overbought → counter-trade)
  V3: Bollinger Band Mean Reversion (price outside bands → fade)
  V4: Momentum Exhaustion (large move + volume spike → reversal)
  V5: VR Regime Switch (VR crosses from high→low = compression starting)

Constraints:
  - Must use same data (50 symbols, 15m/1h/4h, 6yr)
  - Must pass through portfolio_sim with same slot management
  - Combined with v13.1 must not degrade existing VR squeeze performance
  - Anti-correlation with squeeze trades preferred (trades when squeeze doesn't)

Rules followed:
  - No mock data. Real Binance data only.
  - Uses R38 canonical sim (collect_trades, portfolio_sim)
  - Baseline verification (Phase 1)
  - Zero lookahead: all indicators on completed bars only
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
    compute_vr, compute_momentum,
    vol_percentile,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    portfolio_sim_with_details,
    collect_trades as r38_collect_trades,
    sim_directional_trades, sim_straddle_trades,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF, CFG_V13,
)
from research.active.r41_r45.r43_exit_improvement import (
    year_metrics, get_year_boundaries,
)


from quantbox.research.harness import get_commit_short  # 이주(P2): 재구현 제거


# ══════════════════════════════════════════════════════════════════════════════
#  Indicators (numba-accelerated)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI on close prices."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta

    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period

    if avg_gain + avg_loss > 0:
        rsi[period] = 100.0 * avg_gain / (avg_gain + avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_gain + avg_loss > 0:
            rsi[i] = 100.0 * avg_gain / (avg_gain + avg_loss)
        else:
            rsi[i] = 50.0

    return rsi


@numba.jit(nopython=True, cache=True)
def compute_bollinger(
    close: np.ndarray, period: int = 20, mult: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: upper, middle, lower."""
    n = len(close)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        s = 0.0
        for k in range(period):
            s += close[i - period + 1 + k]
        m = s / period
        v = 0.0
        for k in range(period):
            diff = close[i - period + 1 + k] - m
            v += diff * diff
        std = (v / period) ** 0.5
        middle[i] = m
        upper[i] = m + mult * std
        lower[i] = m - mult * std

    return upper, middle, lower


@numba.jit(nopython=True, cache=True)
def compute_move_pct(close: np.ndarray, lookback: int) -> np.ndarray:
    """Absolute percent move over lookback periods."""
    n = len(close)
    move = np.zeros(n)
    for i in range(lookback, n):
        if close[i - lookback] > 0:
            move[i] = abs(close[i] / close[i - lookback] - 1.0)
    return move


# ══════════════════════════════════════════════════════════════════════════════
#  V1: VR Expansion Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_vr_expansion_mr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vr: np.ndarray, momentum: np.ndarray,
    vr_threshold: float, sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean reversion when VR > threshold: fade the momentum direction.

    Entry: VR > vr_threshold (volatility expansion)
    Direction: OPPOSITE of momentum (mean reversion bet)
    SL/TP/max_hold: standard exit
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if np.isnan(vr[i]) or vr[i] <= vr_threshold:
            i += 1
            continue

        # Must have meaningful momentum to fade
        if abs(momentum[i]) < 0.005:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        # Fade: if momentum is positive (up move), SHORT (mean revert down)
        # if momentum is negative (down move), LONG (mean revert up)
        d = -1 if momentum[i] > 0 else 1

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
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V2: RSI Extreme Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_rsi_extreme_mr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    rsi: np.ndarray,
    rsi_oversold: float, rsi_overbought: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean reversion at RSI extremes.

    Entry: RSI < oversold → LONG, RSI > overbought → SHORT
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        d = 0
        if rsi[i] < rsi_oversold:
            d = 1  # LONG on oversold
        elif rsi[i] > rsi_overbought:
            d = -1  # SHORT on overbought

        if d == 0:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

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
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V3: Bollinger Band Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_bollinger_mr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    bb_upper: np.ndarray, bb_lower: np.ndarray,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean reversion at Bollinger Band extremes.

    Entry: close > upper → SHORT, close < lower → LONG
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            i += 1
            continue

        d = 0
        if close[i] > bb_upper[i]:
            d = -1  # SHORT above upper band
        elif close[i] < bb_lower[i]:
            d = 1  # LONG below lower band

        if d == 0:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

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
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V4: Momentum Exhaustion (large move + volume)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_momentum_exhaustion(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    momentum: np.ndarray, volume: np.ndarray,
    vol_pct: np.ndarray,
    move_threshold: float, vol_pct_threshold: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fade large moves on high volume (exhaustion signal).

    Entry: |momentum| > move_threshold AND volume > vol_pct_threshold
    Direction: OPPOSITE of momentum (reversal bet)
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if abs(momentum[i]) < move_threshold:
            i += 1
            continue

        if vol_pct[i] < vol_pct_threshold:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        # Fade the momentum
        d = -1 if momentum[i] > 0 else 1

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
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V5: VR Regime Switch (VR crosses from high → compression starting)
# ══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def sim_vr_regime_switch(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vr: np.ndarray, momentum: np.ndarray,
    vr_high: float, vr_low: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enter when VR crosses from expansion into contraction.

    Entry: VR[i-1] > vr_high AND VR[i] < vr_low (regime switch detected)
    Direction: momentum direction (trend following into compression)
    """
    n = len(close)
    cap = n // 2 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup + 1

    while i < n - 1:
        if np.isnan(vr[i]) or np.isnan(vr[i-1]):
            i += 1
            continue

        # Regime switch: was high, now low
        if not (vr[i-1] > vr_high and vr[i] < vr_low):
            i += 1
            continue

        if abs(momentum[i]) < 0.001:
            i += 1
            continue

        entry = close[i]
        if entry <= 0:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

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
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  Trade Collection — O(n log n) dedup via sweep-line
# ══════════════════════════════════════════════════════════════════════════════


def _dedup_sym_trades(
    sym: str,
    sym_trades: list[tuple[int, int, float]],
) -> list[tuple[str, int, int, float]]:
    """Deduplicate overlapping trades for one symbol using sweep-line O(n log n).

    Given sorted trades from multiple TFs, keep the earliest
    non-overlapping set (greedy interval scheduling).
    """
    if not sym_trades:
        return []
    sym_trades.sort(key=lambda x: x[0])  # sort by entry
    result: list[tuple[str, int, int, float]] = []
    max_exit = -1
    for entry_15m, exit_15m, pnl_val in sym_trades:
        if entry_15m > max_exit:
            result.append((sym, entry_15m, exit_15m, pnl_val))
            max_exit = exit_15m
    return result


def collect_v1_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    vr_threshold: float, sl: float, tp: float,
    sp: int = 8, lp: int = 56,
) -> list[tuple[str, int, int, float]]:
    """Collect V1 trades: VR expansion mean reversion."""
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}

    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(vr_key)
            if vr is None:
                continue
            p, e, x = sim_vr_expansion_mr(
                d["close"], d["high"], d["low"],
                vr, d["mom_8"],
                vr_threshold, sl, tp, max_holds.get(ratio, 96),
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio,
                                   int(x[k]) * ratio + ratio - 1,
                                   float(p[k])))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


def collect_v2_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    rsi_oversold: float, rsi_overbought: float,
    sl: float, tp: float, rsi_period: int = 14,
) -> list[tuple[str, int, int, float]]:
    """Collect V2 trades: RSI extreme mean reversion."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            rsi = compute_rsi(d["close"], rsi_period)
            p, e, x = sim_rsi_extreme_mr(
                d["close"], d["high"], d["low"], rsi,
                rsi_oversold, rsi_overbought,
                sl, tp, max_holds.get(ratio, 96),
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio,
                                   int(x[k]) * ratio + ratio - 1,
                                   float(p[k])))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


def collect_v3_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    bb_period: int, bb_mult: float,
    sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V3 trades: Bollinger Band mean reversion."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            bb_up, _, bb_lo = compute_bollinger(d["close"], bb_period, bb_mult)
            p, e, x = sim_bollinger_mr(
                d["close"], d["high"], d["low"], bb_up, bb_lo,
                sl, tp, max_holds.get(ratio, 96),
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio,
                                   int(x[k]) * ratio + ratio - 1,
                                   float(p[k])))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


def collect_v4_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    move_threshold: float, vol_pct_threshold: float,
    sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V4 trades: Momentum exhaustion on volume."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 96, 4: 24, 16: 6}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            p, e, x = sim_momentum_exhaustion(
                d["close"], d["high"], d["low"],
                d["mom_8"], d["volume"], d["vol_pct"],
                move_threshold, vol_pct_threshold,
                sl, tp, max_holds.get(ratio, 96),
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio,
                                   int(x[k]) * ratio + ratio - 1,
                                   float(p[k])))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


def collect_v5_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    vr_high: float, vr_low: float,
    sl: float, tp: float,
    sp: int = 8, lp: int = 56,
) -> list[tuple[str, int, int, float]]:
    """Collect V5 trades: VR regime switch."""
    vr_key = f"vr_{sp}_{lp}"
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            vr = d.get(vr_key)
            if vr is None:
                continue
            p, e, x = sim_vr_regime_switch(
                d["close"], d["high"], d["low"],
                vr, d["mom_8"],
                vr_high, vr_low,
                sl, tp, mh_d,
                COST_DIRECTIONAL, warmups[ratio],
            )
            for k in range(len(p)):
                sym_trades.append((int(e[k]) * ratio,
                                   int(x[k]) * ratio + ratio - 1,
                                   float(p[k])))
        all_trades.extend(_dedup_sym_trades(sym, sym_trades))
    return all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  Overlap Analysis (lightweight — count-based, not set-of-bars)
# ══════════════════════════════════════════════════════════════════════════════


def overlap_analysis(
    squeeze_trades: list[tuple[str, int, int, float]],
    complement_trades: list[tuple[str, int, int, float]],
) -> dict:
    """Analyze temporal overlap between squeeze and complement trades."""
    sq_total_bars = sum(exit_b - entry + 1 for _, entry, exit_b, _ in squeeze_trades)
    co_total_bars = sum(exit_b - entry + 1 for _, entry, exit_b, _ in complement_trades)

    # Count trades that overlap by checking interval intersection
    overlap_count = 0
    sq_by_sym: dict[str, list[tuple[int, int]]] = {}
    for sym, entry, exit_b, _ in squeeze_trades:
        sq_by_sym.setdefault(sym, []).append((entry, exit_b))

    for sym, entry, exit_b, _ in complement_trades:
        if sym in sq_by_sym:
            for sq_e, sq_x in sq_by_sym[sym]:
                if entry <= sq_x and exit_b >= sq_e:
                    overlap_count += 1
                    break

    return {
        "squeeze_bars": sq_total_bars,
        "complement_bars": co_total_bars,
        "overlap_trades": overlap_count,
        "complement_total": len(complement_trades),
        "unique_pct": round((1 - overlap_count / max(1, len(complement_trades))) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"\n{'='*100}")
    print(f"  R44 — High-Volatility Complement Strategy Research")
    print(f"  Commit: {commit}  |  Baseline: v13.1 P1 (VR squeeze + trail)")
    print(f"{'='*100}\n")

    t0 = time.time()

    # ── Load data ──
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(8, 56)])

    year_bounds = get_year_boundaries(data_15m)
    SLOTS = 10

    # ── Phase 1: Baseline Verification ──
    print(f"  Phase 1: Baseline Verification (v13.1 VR Squeeze)")
    print(f"  {'─'*90}")

    bl_d = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_s = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "S_d8")
    bl_all = bl_d + bl_s
    bl_pnl, bl_acc = portfolio_sim_with_details(bl_all, SLOTS)
    bl_m = compute_metrics(bl_pnl, data_years)
    bl_yr = year_metrics(bl_pnl, bl_acc, year_bounds)

    print(f"  Squeeze BL: N={bl_m['N']:>5}  PF={bl_m['PF']:.3f}  "
          f"EV={bl_m['EV_bp']:+.1f}bp  WR={bl_m['WR']:.1f}%  "
          f"Calmar={bl_m['calmar']:.1f}")

    if abs(bl_m["N"] - 2265) > 2265 * 0.01:
        print(f"  ⛔ BASELINE DRIFT: N={bl_m['N']} vs expected 2265")
        return
    print(f"  ✅ Baseline OK\n")

    # ── Phase 2: Individual Vector Screening ──
    print(f"  Phase 2: Vector Screening (standalone)")
    print(f"  {'─'*90}")

    header = (f"  {'Vector':<35} {'N':>5} {'PF':>6} {'EV(bp)':>8} "
              f"{'WR%':>5} {'Calmar':>7} {'Verdict':>10}")
    print(header)
    print(f"  {'─'*80}")

    results: dict[str, tuple[list, dict, dict]] = {}

    # V1: VR Expansion MR — parameter sweep
    v1_configs = [
        ("V1: VR>1.0 MR sl1/tp1",  1.0, 0.01, 0.01),
        ("V1: VR>1.0 MR sl1/tp2",  1.0, 0.01, 0.02),
        ("V1: VR>1.2 MR sl1/tp1",  1.2, 0.01, 0.01),
        ("V1: VR>1.2 MR sl1.5/tp1", 1.2, 0.015, 0.01),
        ("V1: VR>1.5 MR sl1/tp1",  1.5, 0.01, 0.01),
        ("V1: VR>0.8 MR sl1/tp1",  0.8, 0.01, 0.01),
        ("V1: VR>0.8 MR sl0.5/tp1", 0.8, 0.005, 0.01),
    ]

    for name, vr_th, sl, tp in v1_configs:
        trades = collect_v1_trades(data_15m, data_1h, data_4h, vr_th, sl, tp)
        pnl = portfolio_sim(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, [], year_bounds)  # simplified
        verdict = "✅" if m["PF"] > 1.05 and m["N"] > 100 else "❌"
        print(f"  {name:<35} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['WR']:>5.1f} {m['calmar']:>7.1f} {verdict:>10}")
        results[name] = (trades, m, yr)

    # V2: RSI Extreme MR
    v2_configs = [
        ("V2: RSI <25/>75 sl1/tp1", 25.0, 75.0, 0.01, 0.01),
        ("V2: RSI <20/>80 sl1/tp1", 20.0, 80.0, 0.01, 0.01),
        ("V2: RSI <30/>70 sl1/tp2", 30.0, 70.0, 0.01, 0.02),
        ("V2: RSI <20/>80 sl1/tp2", 20.0, 80.0, 0.01, 0.02),
        ("V2: RSI <15/>85 sl1/tp1", 15.0, 85.0, 0.01, 0.01),
    ]

    for name, oversold, overbought, sl, tp in v2_configs:
        trades = collect_v2_trades(data_15m, data_1h, data_4h,
                                   oversold, overbought, sl, tp)
        pnl = portfolio_sim(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, [], year_bounds)
        verdict = "✅" if m["PF"] > 1.05 and m["N"] > 100 else "❌"
        print(f"  {name:<35} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['WR']:>5.1f} {m['calmar']:>7.1f} {verdict:>10}")
        results[name] = (trades, m, yr)

    # V3: Bollinger Band MR
    v3_configs = [
        ("V3: BB(20,2.0) sl1/tp1", 20, 2.0, 0.01, 0.01),
        ("V3: BB(20,2.5) sl1/tp1", 20, 2.5, 0.01, 0.01),
        ("V3: BB(20,3.0) sl1/tp1", 20, 3.0, 0.01, 0.01),
        ("V3: BB(20,2.0) sl0.5/tp0.5", 20, 2.0, 0.005, 0.005),
        ("V3: BB(50,2.0) sl1/tp1", 50, 2.0, 0.01, 0.01),
    ]

    for name, period, mult, sl, tp in v3_configs:
        trades = collect_v3_trades(data_15m, data_1h, data_4h,
                                   period, mult, sl, tp)
        pnl = portfolio_sim(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, [], year_bounds)
        verdict = "✅" if m["PF"] > 1.05 and m["N"] > 100 else "❌"
        print(f"  {name:<35} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['WR']:>5.1f} {m['calmar']:>7.1f} {verdict:>10}")
        results[name] = (trades, m, yr)

    # V4: Momentum Exhaustion
    v4_configs = [
        ("V4: mom>3% vol>P80 sl1/tp1",  0.03, 80.0, 0.01, 0.01),
        ("V4: mom>5% vol>P80 sl1/tp1",  0.05, 80.0, 0.01, 0.01),
        ("V4: mom>3% vol>P90 sl1/tp1",  0.03, 90.0, 0.01, 0.01),
        ("V4: mom>5% vol>P90 sl1/tp2",  0.05, 90.0, 0.01, 0.02),
        ("V4: mom>3% vol>P70 sl1/tp1",  0.03, 70.0, 0.01, 0.01),
    ]

    for name, move_th, vol_th, sl, tp in v4_configs:
        trades = collect_v4_trades(data_15m, data_1h, data_4h,
                                   move_th, vol_th, sl, tp)
        pnl = portfolio_sim(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, [], year_bounds)
        verdict = "✅" if m["PF"] > 1.05 and m["N"] > 100 else "❌"
        print(f"  {name:<35} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['WR']:>5.1f} {m['calmar']:>7.1f} {verdict:>10}")
        results[name] = (trades, m, yr)

    # V5: VR Regime Switch
    v5_configs = [
        ("V5: VR 1.0→0.8 sl1/tp5",  1.0, 0.8, 0.01, 0.05),
        ("V5: VR 1.2→0.8 sl1/tp5",  1.2, 0.8, 0.01, 0.05),
        ("V5: VR 1.0→0.6 sl1/tp5",  1.0, 0.6, 0.01, 0.05),
        ("V5: VR 1.2→0.5 sl1/tp5",  1.2, 0.5, 0.01, 0.05),
    ]

    for name, vr_h, vr_l, sl, tp in v5_configs:
        trades = collect_v5_trades(data_15m, data_1h, data_4h,
                                   vr_h, vr_l, sl, tp)
        pnl = portfolio_sim(trades, SLOTS)
        m = compute_metrics(pnl, data_years)
        yr = year_metrics(pnl, [], year_bounds)
        verdict = "✅" if m["PF"] > 1.05 and m["N"] > 100 else "❌"
        print(f"  {name:<35} {m['N']:>5} {m['PF']:>6.3f} {m['EV_bp']:>+8.1f} "
              f"{m['WR']:>5.1f} {m['calmar']:>7.1f} {verdict:>10}")
        results[name] = (trades, m, yr)

    # ── Phase 3: Top candidates — combined with squeeze ──
    print(f"\n  Phase 3: Combined Performance (Squeeze + Complement)")
    print(f"  {'─'*90}")

    # Find top-3 passing vectors
    passing = [(name, trades, m) for name, (trades, m, yr) in results.items()
               if m["PF"] > 1.05 and m["N"] > 100]
    passing.sort(key=lambda x: -x[2]["calmar"])

    if not passing:
        print(f"  ⚠️  No vectors passed PF>1.05 AND N>100 threshold.")
        print(f"  Consider mean reversion may not work well in crypto.")
        elapsed = time.time() - t0
        print(f"\n  Runtime: {elapsed:.1f}s  |  Commit: {commit}")
        return

    print(f"\n  Top {min(5, len(passing))} passing vectors:")
    for name, trades, m in passing[:5]:
        # Combine with squeeze trades
        combined = bl_all + trades
        comb_pnl, comb_acc = portfolio_sim_with_details(combined, SLOTS)
        comb_m = compute_metrics(comb_pnl, data_years)
        comb_yr = year_metrics(comb_pnl, comb_acc, year_bounds)

        # Overlap analysis
        ovl = overlap_analysis(bl_all, trades)

        delta_pf = comb_m["PF"] - bl_m["PF"]
        delta_calmar = comb_m["calmar"] - bl_m["calmar"]

        print(f"\n  {name}")
        print(f"    Standalone:  N={m['N']:>5}  PF={m['PF']:.3f}  "
              f"EV={m['EV_bp']:+.1f}bp  Calmar={m['calmar']:.1f}")
        print(f"    Combined:    N={comb_m['N']:>5}  PF={comb_m['PF']:.3f} "
              f"(Δ{delta_pf:+.3f})  EV={comb_m['EV_bp']:+.1f}bp  "
              f"Calmar={comb_m['calmar']:.1f} (Δ{delta_calmar:+.1f})")
        print(f"    Overlap:     {ovl['overlap_pct']:.1f}% bar overlap  "
              f"(squeeze {ovl['squeeze_bars']} bars, "
              f"complement {ovl['complement_bars']} bars)")

        # Year-by-year for combined
        print(f"    Yearly ΔEV: ", end="")
        all_better = True
        for yr in sorted(comb_yr.keys()):
            if yr in bl_yr:
                delta_ev = comb_yr[yr]["EV_bp"] - bl_yr[yr]["EV_bp"]
                marker = "✓" if delta_ev >= -3.0 else "✗"
                if delta_ev < -3.0:
                    all_better = False
                print(f"{yr}:{delta_ev:+.1f}{marker} ", end="")
        status = " ✅ All years OK" if all_better else " ⚠️ Degraded years"
        print(status)

    # ── Phase 4: Anti-correlation check ──
    print(f"\n  Phase 4: Anti-Correlation Analysis")
    print(f"  {'─'*90}")

    best_name, best_trades, best_m = passing[0]
    print(f"  Best vector: {best_name}")

    # Check: when VR squeeze is idle (VR > 0.7 everywhere),
    # does complement still generate trades?
    complement_during_expansion = 0
    complement_during_squeeze = 0
    for sym, entry, exit_b, pnl_val in best_trades:
        # Check if squeeze was also active around the same time
        squeeze_overlap = False
        for sq_sym, sq_entry, sq_exit, sq_pnl in bl_all:
            if sq_sym == sym and entry <= sq_exit and exit_b >= sq_entry:
                squeeze_overlap = True
                break
        if squeeze_overlap:
            complement_during_squeeze += 1
        else:
            complement_during_expansion += 1

    total_complement = len(best_trades)
    pct_unique = complement_during_expansion / total_complement * 100 if total_complement > 0 else 0
    print(f"  Complement trades during VR squeeze idle: "
          f"{complement_during_expansion}/{total_complement} ({pct_unique:.1f}%)")
    print(f"  Complement trades overlapping squeeze: "
          f"{complement_during_squeeze}/{total_complement}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R44 SUMMARY")
    print(f"  {'─'*90}")
    print(f"  {len(passing)} vectors passed (PF>1.05, N>100)")
    if passing:
        best = passing[0]
        print(f"  Best: {best[0]} → PF={best[2]['PF']:.3f}, "
              f"N={best[2]['N']}, Calmar={best[2]['calmar']:.1f}")
    print(f"  Runtime: {elapsed:.1f}s  |  Commit: {commit}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
