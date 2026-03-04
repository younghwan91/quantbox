"""R44c — Moving Average Confluence Strategy Research

Core thesis (user-driven):
  "이동평균선이야말로 모든 심리가 모이는 구간. 큰 이동평균일수록, 큰 타임프레임일수록 더더욱.
   여러 이동평균선이 좁게 모이면 강한 지지/저항. VWMA-100도 잘 쓰인다.
   큰 R:R로 타점을 이룬다. 관심이 모이는 자리를 아는 것이 우승.
   여러 타임프레임에서 컨피던스를 챙긴다."

Key MAs: SMA(20), SMA(50), SMA(100), SMA(200), VWMA(100)
MA periods in real-time:
  15m → SMA200 ≈ 50h (2 days)
  1h  → SMA200 ≈ 200h (8 days)
  4h  → SMA200 ≈ 800h (33 days) ← VERY significant

Strategy vectors:
  V11: MA Cluster Breakout — all MAs converge → breakout direction
  V12: Key MA Bounce — pullback to SMA50/100/200 in trend → bounce
  V13: MTF MA Pullback — 4h SMA200 trend + 15m SMA pullback entry
  V14: VWMA Bounce — price bounces off VWMA(50/100)
  V15: MA Perfect Order — SMA20>50>100>200 → pullback to nearest MA

Design principles:
  - MAs = where trader attention converges → real S/R
  - Tight SL (at MA level), Wide TP → high R:R (3:1 to 8:1)
  - Multi-TF confidence: higher TF MA = more significant
  - All conditions non-lookahead (use bar i data, enter at close[i])

Rules: Real data only, R38 infrastructure, baseline verification.
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

from research.active.mtf_vol_straddle_r31 import (
    WARMUP_15m, WARMUP_1h, WARMUP_4h,
    load_15m, build_resampled, precompute, pf_from_pnl,
)
from research.active.scaling_r38 import (
    compute_metrics, precompute_all_vr, portfolio_sim,
    collect_trades as r38_collect_trades,
    COST_DIRECTIONAL, G_3TF, CFG_V13,
)
from research.active.high_vol_complement_r44 import (
    _dedup_sym_trades,
)


def get_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Indicators: Moving Averages (numba-accelerated)
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average — rolling mean of close."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += close[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += close[i] - close[i - period]
        out[i] = s / period
    return out


@numba.jit(nopython=True, cache=True)
def compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — Wilder smoothing."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s = 0.0
    for i in range(period):
        s += close[i]
    out[period - 1] = s / period
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = close[i] * alpha + out[i - 1] * (1.0 - alpha)
    return out


@numba.jit(nopython=True, cache=True)
def compute_vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average.

    VWMA = sum(close * volume, n) / sum(volume, n)
    Represents where volume actually traded — real consensus price.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < period:
        return out
    s_cv = 0.0  # sum(close * volume)
    s_v = 0.0   # sum(volume)
    for i in range(period):
        s_cv += close[i] * volume[i]
        s_v += volume[i]
    if s_v > 0:
        out[period - 1] = s_cv / s_v
    for i in range(period, n):
        s_cv += close[i] * volume[i] - close[i - period] * volume[i - period]
        s_v += volume[i] - volume[i - period]
        if s_v > 0:
            out[i] = s_cv / s_v
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Precompute MAs for all symbols
# ══════════════════════════════════════════════════════════════════════════════


def precompute_mas(data: dict) -> None:
    """Add SMA and VWMA arrays to each symbol dict."""
    for sym, d in data.items():
        c = d["close"]
        v = d["volume"]
        d["sma_20"] = compute_sma(c, 20)
        d["sma_50"] = compute_sma(c, 50)
        d["sma_100"] = compute_sma(c, 100)
        d["sma_200"] = compute_sma(c, 200)
        d["vwma_50"] = compute_vwma(c, v, 50)
        d["vwma_100"] = compute_vwma(c, v, 100)


# ══════════════════════════════════════════════════════════════════════════════
#  V11: MA Cluster Breakout
#
#  When multiple MAs converge (spread < cluster_pct), it's compression.
#  Price breaking above/below the cluster = directional breakout.
#  Tight SL on the other side of cluster, wide TP.
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_cluster_breakout(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    cluster_pct: float,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Breakout when all 4 MAs converge within cluster_pct.

    Entry: MAs spread < cluster_pct AND close breaks above/below the MA band.
    Direction: breakout side.
    SL/TP: fixed percentage.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = warmup

    while i < n - 1:
        if np.isnan(sma200[i]):
            i += 1
            continue

        # Compute MA band spread
        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread > cluster_pct:
            i += 1
            continue

        # MAs are clustered — check breakout direction
        if close[i] > ma_max:
            d = 1   # bullish breakout above cluster
        elif close[i] < ma_min:
            d = -1  # bearish breakout below cluster
        else:
            i += 1  # price inside cluster, no clear break
            continue

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
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V12: Key MA Bounce — pullback to MA in trend direction
#
#  Trend filter: SMA200.  Bounce target: SMA50 or SMA100.
#  Long: close > SMA200, price dips to bounce_ma and closes above → bounce.
#  Short: close < SMA200, price rallies to bounce_ma and closes below → reject.
#  "큰 이동평균일수록 더더욱" — SMA200 defines trend, smaller MAs give entry.
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_key_ma_bounce(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    bounce_ma: np.ndarray,
    trend_ma: np.ndarray,
    use_trend: bool,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounce off a key MA level with optional trend filter.

    Long: was above bounce_ma, wick touches it, closes above → support bounce.
    Short: was below bounce_ma, wick touches it, closes below → resistance reject.
    Trend filter: close must be above/below trend_ma for long/short.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = max(warmup, 1)  # need i-1

    while i < n - 1:
        if np.isnan(bounce_ma[i]) or np.isnan(bounce_ma[i - 1]):
            i += 1
            continue
        if use_trend and np.isnan(trend_ma[i]):
            i += 1
            continue

        d = 0

        # Long bounce: was above MA, wick touched, closed above
        if (close[i - 1] >= bounce_ma[i - 1]
                and low[i] <= bounce_ma[i]
                and close[i] > bounce_ma[i]):
            if use_trend:
                if close[i] > trend_ma[i]:
                    d = 1
            else:
                d = 1

        # Short rejection: was below MA, wick touched, closed below
        elif (close[i - 1] <= bounce_ma[i - 1]
              and high[i] >= bounce_ma[i]
              and close[i] < bounce_ma[i]):
            if use_trend:
                if close[i] < trend_ma[i]:
                    d = -1
            else:
                d = -1

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
#  V13: Multi-Timeframe MA Pullback
#
#  "여러 타임프레임에서 컨피던스를 챙긴다"
#  4h SMA200 (≈33 days) defines macro trend direction.
#  15m SMA50 (≈12h) gives pullback entry point.
#  Runs on 15m bars, references completed 4h bars only (j = i//16 - 1).
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_mtf_ma_pullback(
    close_15m: np.ndarray, high_15m: np.ndarray, low_15m: np.ndarray,
    entry_ma_15m: np.ndarray,
    trend_ma_4h: np.ndarray,
    close_4h: np.ndarray,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int, ratio: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MTF pullback entry: 4h trend + 15m MA bounce.

    Long: 4h close > 4h SMA200 (bullish macro) AND
          15m pullback to 15m entry_ma (wick touch + close above).
    Short: 4h close < 4h SMA200 (bearish macro) AND
           15m rally to 15m entry_ma (wick touch + close below).
    """
    n = len(close_15m)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = max(warmup, 1)

    while i < n - 1:
        if np.isnan(entry_ma_15m[i]) or np.isnan(entry_ma_15m[i - 1]):
            i += 1
            continue

        # Completed 4h bar index
        j = (i // ratio) - 1
        if j < 0 or j >= len(close_4h) or np.isnan(trend_ma_4h[j]):
            i += 1
            continue

        d = 0

        # 4h bullish + 15m pullback to MA → LONG
        if close_4h[j] > trend_ma_4h[j]:
            if (close_15m[i - 1] >= entry_ma_15m[i - 1]
                    and low_15m[i] <= entry_ma_15m[i]
                    and close_15m[i] > entry_ma_15m[i]):
                d = 1

        # 4h bearish + 15m rally to MA → SHORT
        elif close_4h[j] < trend_ma_4h[j]:
            if (close_15m[i - 1] <= entry_ma_15m[i - 1]
                    and high_15m[i] >= entry_ma_15m[i]
                    and close_15m[i] < entry_ma_15m[i]):
                d = -1

        if d == 0:
            i += 1
            continue

        entry = close_15m[i]
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
        for j2 in range(i + 1, end):
            if d == 1:
                if low_15m[j2] <= sl_price:
                    trade_pnl = -sl
                    eb = j2
                    break
                if high_15m[j2] >= tp_price:
                    trade_pnl = tp
                    eb = j2
                    break
            else:
                if high_15m[j2] >= sl_price:
                    trade_pnl = -sl
                    eb = j2
                    break
                if low_15m[j2] <= tp_price:
                    trade_pnl = tp
                    eb = j2
                    break
        else:
            eb = end - 1
            if d == 1:
                trade_pnl = (close_15m[eb] - entry) / entry
            else:
                trade_pnl = (entry - close_15m[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            cnt += 1
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt]


# ══════════════════════════════════════════════════════════════════════════════
#  V14: VWMA Bounce
#
#  "100 volume weighted moving average도 잘 쓰인다"
#  VWMA represents where VOLUME actually traded — real consensus price.
#  Price bouncing off VWMA = respecting the volume-weighted consensus.
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_vwma_bounce(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    vwma: np.ndarray,
    trend_ma: np.ndarray,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounce off VWMA with SMA200 trend filter.

    Long: close > SMA200 (uptrend), wick dips to VWMA, closes above → bounce.
    Short: close < SMA200 (downtrend), wick rallies to VWMA, closes below → reject.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = max(warmup, 1)

    while i < n - 1:
        if np.isnan(vwma[i]) or np.isnan(vwma[i - 1]) or np.isnan(trend_ma[i]):
            i += 1
            continue

        d = 0

        # Long bounce off VWMA in uptrend
        if close[i] > trend_ma[i]:
            if (close[i - 1] >= vwma[i - 1]
                    and low[i] <= vwma[i]
                    and close[i] > vwma[i]):
                d = 1

        # Short rejection at VWMA in downtrend
        elif close[i] < trend_ma[i]:
            if (close[i - 1] <= vwma[i - 1]
                    and high[i] >= vwma[i]
                    and close[i] < vwma[i]):
                d = -1

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
#  V15: MA Perfect Order + Pullback
#
#  Perfect bullish: SMA20 > SMA50 > SMA100 > SMA200 → strong trend.
#  Perfect bearish: SMA20 < SMA50 < SMA100 < SMA200 → strong downtrend.
#  Entry on pullback to nearest significant MA.
#  "관심이 모이는 자리를 아는 것이 우승"
# ══════════════════════════════════════════════════════════════════════════════


@numba.jit(nopython=True, cache=True)
def sim_ma_order_pullback(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    sma20: np.ndarray, sma50: np.ndarray,
    sma100: np.ndarray, sma200: np.ndarray,
    pullback_to_50: bool,
    sl: float, tp: float, max_hold: int,
    cost: float, warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trade pullbacks in perfectly ordered MA structure.

    Perfect bullish order → pullback to SMA20 or SMA50 → LONG.
    Perfect bearish order → rally to SMA20 or SMA50 → SHORT.
    pullback_to_50=True: enter at SMA50 (deeper pullback, less frequent).
    pullback_to_50=False: enter at SMA20 (shallower pullback, more frequent).
    """
    n = len(close)
    cap = n // 4 + 1
    pnl = np.empty(cap, dtype=np.float64)
    ent = np.empty(cap, dtype=np.int64)
    ext = np.empty(cap, dtype=np.int64)
    cnt = 0
    i = max(warmup, 1)

    while i < n - 1:
        if np.isnan(sma200[i]) or np.isnan(sma200[i - 1]):
            i += 1
            continue

        d = 0

        # Check perfect bullish order: 20 > 50 > 100 > 200
        if sma20[i] > sma50[i] > sma100[i] > sma200[i]:
            if pullback_to_50:
                target_ma = sma50[i]
                target_prev = sma50[i - 1]
            else:
                target_ma = sma20[i]
                target_prev = sma20[i - 1]
            # Pullback: was above target MA, wick touched, closed above
            if (close[i - 1] >= target_prev
                    and low[i] <= target_ma
                    and close[i] > target_ma):
                d = 1

        # Check perfect bearish order: 20 < 50 < 100 < 200
        elif sma20[i] < sma50[i] < sma100[i] < sma200[i]:
            if pullback_to_50:
                target_ma = sma50[i]
                target_prev = sma50[i - 1]
            else:
                target_ma = sma20[i]
                target_prev = sma20[i - 1]
            # Rally to target: was below target MA, wick touched, closed below
            if (close[i - 1] <= target_prev
                    and high[i] >= target_ma
                    and close[i] < target_ma):
                d = -1

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
#  Collect functions — aggregate trades across TFs, per-symbol dedup
# ══════════════════════════════════════════════════════════════════════════════


def collect_v11_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    cluster_pct: float, sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V11 trades: MA Cluster Breakout, all 3 TFs."""
    tf_data_map = {1: data_15m, 4: data_1h, 16: data_4h}
    warmups = {1: WARMUP_15m, 4: WARMUP_1h, 16: WARMUP_4h}
    max_holds = {1: 192, 4: 48, 16: 12}  # 48h each TF

    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        sym_trades: list[tuple[int, int, float]] = []
        for tf_label, ratio, warmup_base, mh_d, mh_s in G_3TF:
            d_tf = tf_data_map.get(ratio)
            if d_tf is None or sym not in d_tf:
                continue
            d = d_tf[sym]
            p, e, x = sim_ma_cluster_breakout(
                d["close"], d["high"], d["low"],
                d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"],
                cluster_pct, sl, tp, max_holds[ratio],
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


def collect_v12_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    bounce_key: str, use_trend: bool,
    sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V12 trades: Key MA Bounce, all 3 TFs."""
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
            bounce_ma = d.get(bounce_key)
            trend_ma = d.get("sma_200")
            if bounce_ma is None or trend_ma is None:
                continue
            p, e, x = sim_key_ma_bounce(
                d["close"], d["high"], d["low"],
                bounce_ma, trend_ma, use_trend,
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


def collect_v13_trades(
    data_15m: dict, data_4h: dict,
    entry_ma_key: str, trend_ma_key: str,
    sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V13 trades: MTF MA Pullback.

    Only runs on 15m bars; references 4h data for trend direction.
    No per-TF iteration — the MTF is internal to the sim.
    """
    all_trades: list[tuple[str, int, int, float]] = []
    for sym in sorted(data_15m.keys()):
        if sym not in data_4h:
            continue
        d15 = data_15m[sym]
        d4h = data_4h[sym]
        entry_ma_15m = d15.get(entry_ma_key)
        trend_ma_4h = d4h.get(trend_ma_key)
        if entry_ma_15m is None or trend_ma_4h is None:
            continue
        p, e, x = sim_mtf_ma_pullback(
            d15["close"], d15["high"], d15["low"],
            entry_ma_15m, trend_ma_4h, d4h["close"],
            sl, tp, 192, COST_DIRECTIONAL, WARMUP_15m, 16,
        )
        for k in range(len(p)):
            all_trades.append((sym, int(e[k]), int(x[k]), float(p[k])))
    return all_trades


def collect_v14_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    vwma_key: str, sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V14 trades: VWMA Bounce with SMA200 trend filter, all 3 TFs."""
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
            vwma = d.get(vwma_key)
            trend_ma = d.get("sma_200")
            if vwma is None or trend_ma is None:
                continue
            p, e, x = sim_vwma_bounce(
                d["close"], d["high"], d["low"],
                vwma, trend_ma,
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


def collect_v15_trades(
    data_15m: dict, data_1h: dict, data_4h: dict,
    pullback_to_50: bool, sl: float, tp: float,
) -> list[tuple[str, int, int, float]]:
    """Collect V15 trades: MA Perfect Order + Pullback, all 3 TFs."""
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
            p, e, x = sim_ma_order_pullback(
                d["close"], d["high"], d["low"],
                d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"],
                pullback_to_50,
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
#  Main: Sweep all vectors
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    commit = get_commit_short()
    print(f"R44c — MA Confluence Research [{commit}]")
    print("=" * 70)

    # ── Load data ──
    t0 = time.time()
    print("Loading 15m data...")
    data_15m = load_15m()
    print(f"  {len(data_15m)} symbols loaded")

    print("Resampling to 1h / 4h...")
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)

    # ── Precompute VR (for baseline) ──
    print("Precomputing VR indicators...")
    vr_pairs = [(8, 56)]
    precompute_all_vr([data_15m, data_1h, data_4h], vr_pairs)
    for ds in [data_15m, data_1h, data_4h]:
        precompute(ds, vr_sp=8, vr_lp=56)

    # ── Precompute MAs ──
    print("Precomputing Moving Averages (SMA 20/50/100/200, VWMA 50/100)...")
    t_ma = time.time()
    for ds in [data_15m, data_1h, data_4h]:
        precompute_mas(ds)
    print(f"  MAs computed in {time.time() - t_ma:.1f}s")

    # ── Estimate data span ──
    bars_list = [d["bars"] for d in data_15m.values()]
    avg_bars = int(np.mean(bars_list)) if bars_list else 0
    data_years = avg_bars / (4 * 24 * 365)  # 15m bars → years
    print(f"  avg bars = {avg_bars}, data_years = {data_years:.1f}")
    print(f"  Total load time: {time.time() - t0:.1f}s")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1: Baseline verification
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 70)
    print("Phase 1: Baseline verification (CFG_V13)")
    print("=" * 70)
    bl_trades = r38_collect_trades(CFG_V13, data_15m, data_1h, data_4h, "D_mom8")
    bl_pnl = portfolio_sim(bl_trades, 10)
    bl_m = compute_metrics(bl_pnl, data_years)
    print(f"  Baseline: N={bl_m['N']}  PF={bl_m['PF']:.3f}  "
          f"WR={bl_m['WR']:.1f}%  EV={bl_m['EV_bp']:+.1f}bp  "
          f"Calmar={bl_m['calmar']:.1f}")
    # Sanity: N should be ~2265 ±1%, PF ~1.280 ±1%
    if abs(bl_m['N'] - 2265) > 2265 * 0.01:
        print(f"  ⚠️  Baseline N drift: {bl_m['N']} vs expected 2265")
    if abs(bl_m['PF'] - 1.280) > 1.280 * 0.01:
        print(f"  ⚠️  Baseline PF drift: {bl_m['PF']:.3f} vs expected 1.280")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2: MA Strategy Sweep
    # ══════════════════════════════════════════════════════════════════════

    print("=" * 70)
    print("Phase 2: MA Strategy Sweep")
    print("=" * 70)

    results: list[tuple[str, dict]] = []
    PASS_THRESHOLD = 1.0  # PF >= 1.0 to pass

    # ── V11: MA Cluster Breakout ──
    print("\n── V11: MA Cluster Breakout ──")
    print("  MAs converge (spread < cluster_pct) → breakout direction")
    for cluster_pct in [0.01, 0.015, 0.02, 0.03]:
        for sl in [0.01, 0.015, 0.02]:
            for tp in [0.03, 0.05, 0.08]:
                label = (f"V11 cluster={cluster_pct*100:.1f}% "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v11_trades(
                    data_15m, data_1h, data_4h, cluster_pct, sl, tp)
                if not trades:
                    print(f"  {label}: 0 trades  ❌")
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ── V12: Key MA Bounce ──
    print("\n── V12: Key MA Bounce ──")
    print("  Pullback to SMA, trend filter from SMA200")
    for bounce_key, use_trend, trend_label in [
        ("sma_50", True, "SMA50+trend"),
        ("sma_100", True, "SMA100+trend"),
        ("sma_200", False, "SMA200 (no filter)"),
    ]:
        for sl in [0.005, 0.01, 0.015]:
            for tp in [0.03, 0.05, 0.08]:
                label = (f"V12 {trend_label} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v12_trades(
                    data_15m, data_1h, data_4h,
                    bounce_key, use_trend, sl, tp)
                if not trades:
                    print(f"  {label}: 0 trades  ❌")
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ── V13: MTF MA Pullback ──
    print("\n── V13: MTF MA Pullback ──")
    print("  4h SMA trend → 15m MA pullback entry")
    for trend_key, trend_label in [
        ("sma_100", "4h_SMA100"),
        ("sma_200", "4h_SMA200"),
    ]:
        for entry_key, entry_label in [
            ("sma_20", "15m_SMA20"),
            ("sma_50", "15m_SMA50"),
        ]:
            for sl in [0.005, 0.01]:
                for tp in [0.03, 0.05, 0.08]:
                    label = (f"V13 {trend_label}+{entry_label} "
                             f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                    trades = collect_v13_trades(
                        data_15m, data_4h,
                        entry_key, trend_key, sl, tp)
                    if not trades:
                        print(f"  {label}: 0 trades  ❌")
                        continue
                    pnl = portfolio_sim(trades, 10)
                    m = compute_metrics(pnl, data_years)
                    flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                    print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                          f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                          f"RR={m['RR']:.2f}  {flag}")
                    results.append((label, m))

    # ── V14: VWMA Bounce ──
    print("\n── V14: VWMA Bounce ──")
    print("  VWMA-50/100 bounce with SMA200 trend filter")
    for vwma_key, vwma_label in [
        ("vwma_50", "VWMA50"),
        ("vwma_100", "VWMA100"),
    ]:
        for sl in [0.005, 0.01, 0.015]:
            for tp in [0.03, 0.05, 0.08]:
                label = (f"V14 {vwma_label} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v14_trades(
                    data_15m, data_1h, data_4h,
                    vwma_key, sl, tp)
                if not trades:
                    print(f"  {label}: 0 trades  ❌")
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ── V15: MA Perfect Order + Pullback ──
    print("\n── V15: MA Perfect Order + Pullback ──")
    print("  SMA20>50>100>200 → pullback to nearest MA")
    for pullback_to_50, pb_label in [
        (False, "pullback_SMA20"),
        (True, "pullback_SMA50"),
    ]:
        for sl in [0.005, 0.01, 0.015]:
            for tp in [0.03, 0.05, 0.08]:
                label = (f"V15 {pb_label} "
                         f"sl={sl*100:.1f}% tp={tp*100:.0f}%")
                trades = collect_v15_trades(
                    data_15m, data_1h, data_4h,
                    pullback_to_50, sl, tp)
                if not trades:
                    print(f"  {label}: 0 trades  ❌")
                    continue
                pnl = portfolio_sim(trades, 10)
                m = compute_metrics(pnl, data_years)
                flag = "✓" if m["PF"] >= PASS_THRESHOLD else "❌"
                print(f"  {label}: N={m['N']:>5}  PF={m['PF']:.3f}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                      f"RR={m['RR']:.2f}  {flag}")
                results.append((label, m))

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 3: Summary
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 70)
    print("Phase 3: Summary")
    print("=" * 70)

    passing = [(l, m) for l, m in results if m["PF"] >= PASS_THRESHOLD]
    total = len(results)
    print(f"\n  Total configs tested: {total}")
    print(f"  Passing (PF ≥ 1.0):  {len(passing)}")

    if passing:
        print("\n  ── Passing configs (sorted by PF) ──")
        passing.sort(key=lambda x: -x[1]["PF"])
        for label, m in passing[:20]:
            print(f"    {label}")
            print(f"      N={m['N']:>5}  PF={m['PF']:.3f}  "
                  f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp  "
                  f"RR={m['RR']:.2f}  Calmar={m['calmar']:.1f}")

        # Deep analysis of top 3
        print("\n  ── Top 3 deep analysis ──")
        for label, m in passing[:3]:
            print(f"\n  {label}")
            for k, v in m.items():
                print(f"    {k}: {v}")
    else:
        print("\n  No config passed PF ≥ 1.0.")
        # Show top 5 by PF anyway
        if results:
            results.sort(key=lambda x: -x[1]["PF"])
            print("\n  ── Top 5 by PF (all below 1.0) ──")
            for label, m in results[:5]:
                print(f"    {label}: PF={m['PF']:.3f}  N={m['N']}  "
                      f"WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp")

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
    print(f"  Commit: {commit}")


if __name__ == "__main__":
    main()
