#!/usr/bin/env python3
"""
Alpha Search R11: VWMA + VPVR/POC — R:R Maximization for Leverage Confidence
=============================================================================
Building on R10's discovery that VWMA(100) distance is the #1 trade quality
predictor (r=+0.136, mono↑, 2.1× EV spread), this round introduces:

  1. **VPVR Point of Control (POC):** Rolling volume profile computes the
     price level where most volume occurred. POC is a natural S/R magnet.

  2. **Value Area (VA):** The price range containing 70% of total volume.
     VA High/Low act as dynamic support/resistance boundaries.

  3. **VWMA + POC synergy:** Two independent volume-weighted references
     that together define entry quality, stop anchoring, and target magnets.

Goal: Maximize R:R to build confidence for higher leverage.

Part A: DIAGNOSTIC — Factor analysis of POC distance, VA position, VWMA dist
Part B: 22 VARIANTS — entry gates, stops, targets, combos, R:R sizing

Data: 6yr (2190d), 50 symbols, exact production logic.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r11.py
"""

import sys
import time
import json
import subprocess
import numba
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
    compute_atr_pct,
)

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'
DATA_DAYS = 2190

# ═══════════════════════════════════════════════════════════════════
# Production parameters (v12.4 I-HybTight baseline)
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032
BTC_MOM_PERIOD = 48
BTC_REGIME_PERIOD = 96
ASSET_TREND_PERIOD = 192
MIN_CONFIDENCE = 3
BASE_LEVERAGE = 1.5
MTF_LEV_MULT = 2.5
EFFECTIVE_LEV = BASE_LEVERAGE * MTF_LEV_MULT
RSI_PERIOD = 14
VOL_PERIOD = 20
STOCH_ENABLED = True
STOCH_K = 3
STOCH_D = 3
MACD_ENABLED = True
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 14
ATR_PERIOD = 48
ATR_MULT = 2.0
THR_FLOOR = 0.008
THR_CAP = 0.06
MTF_SMA_PERIOD = 12

INITIAL_CAPITAL = 2095.0
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.01 / 100 / 32
BARS_PER_YEAR = 96 * 365

CAP_MULT = {2: 0.3, 3: 0.5, 4: 1.5, 5: 2.0}
SOFT_EQ = [(0.08, 0.50), (0.05, 0.70), (0.03, 0.85)]

SYMBOLS_50 = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT', 'EGLDUSDT',
    'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT',
    'CRVUSDT', 'LDOUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'VETUSDT',
    'EOSUSDT', 'APEUSDT',
]

# VPVR parameters
VPVR_LOOKBACK = 672       # 672 × 15m = 7 days of rolling volume profile
VPVR_NBINS = 50           # 50 price bins for volume profile histogram
VPVR_VA_PCT = 0.70        # Value Area = 70% of total volume


# ═══════════════════════════════════════════════════════════════════
# Variant Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class VConfig:
    """Variant configuration for VWMA + VPVR/POC R:R testing.

    Base exits are production I-HybTight. VWMA, POC, and VA features
    are layered on top.
    """
    name: str = 'A-Baseline'
    desc: str = ''

    # Base exit params (production)
    stop_mult: float = 0.382
    target_retrace: float = 1.5
    max_hold: int = 25

    # ── VWMA entry gate ──
    vwma_entry_min: float = 0.0        # min directional VWMA dist to enter (0=off)

    # ── VWMA stop ──
    vwma_stop: str = 'off'             # 'off', 'replace', 'tight', 'wide'
    vwma_stop_mult: float = 0.5        # multiplier on VWMA distance for stop

    # ── VWMA target ──
    vwma_target: str = 'off'           # 'off', 'replace', 'wide'
    vwma_target_mult: float = 1.0      # multiplier on VWMA distance for target

    # ── POC-based stop ──
    poc_stop: str = 'off'              # 'off', 'replace', 'tight', 'wide'
    poc_stop_mult: float = 0.5         # multiplier on POC distance for stop

    # ── POC-based target ──
    poc_target: str = 'off'            # 'off', 'replace', 'wide'
    poc_target_mult: float = 1.0       # multiplier on POC distance for target

    # ── VA entry gate ──
    va_entry_gate: bool = False        # only enter when price is OUTSIDE Value Area

    # ── VA stop: anchor to VA boundary ──
    va_stop: str = 'off'               # 'off', 'replace', 'tight', 'wide'
    va_stop_margin: float = 0.002      # margin beyond VA boundary for stop

    # ── VA target: target opposite VA boundary ──
    va_target: str = 'off'             # 'off', 'replace', 'wide'

    # ── R:R-gated sizing ──
    rr_size: bool = False              # scale size by implied R:R from POC/VA
    rr_size_min: float = 0.5           # min size factor (poor R:R)
    rr_size_max: float = 2.0           # max size factor (great R:R)

    # ── PBT (constant across all variants) ──
    partial_exit: bool = True
    partial_frac: float = 0.5
    trail_atr: float = 1.0
    be_after_tp1: bool = True
    runner_mh_mult: float = 3.0

    # ── Portfolio (constant) ──
    max_slots: int = 2
    dir_div: bool = True
    steep_cap: bool = True
    vol_adj: bool = True
    soft_eq: bool = True


# ═══════════════════════════════════════════════════════════════════
# Data loading (identical to R10)
# ═══════════════════════════════════════════════════════════════════
def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    pattern = f"{symbol}_{interval}_{days}d_*.npz"
    matches = sorted(CACHE_DIR.glob(pattern))
    if matches:
        return matches[-1]
    pattern2 = f"{symbol}_{interval}_*.npz"
    matches2 = sorted(CACHE_DIR.glob(pattern2))
    return matches2[-1] if matches2 else None


def load_15m(symbol: str) -> dict | None:
    cache = _find_cache(symbol, '15m', DATA_DAYS)
    if cache is None:
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', DATA_DAYS)
    if cache is None:
        return None
    return np.load(cache)['close']


# ═══════════════════════════════════════════════════════════════════
# Entry scanner (production-identical, from R10)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_entries(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period, btc_regime_period,
    asset_trend_period, min_confidence,
    rsi_period, vol_period,
    stoch_enabled, stoch_k_period, stoch_d_period,
    macd_enabled, macd_fast, macd_slow, macd_signal_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
):
    """Production-identical entry scanner (symmetric conf gate)."""
    n = min(len(close), len(btc_close))
    dirs = np.zeros(n)
    confs = np.zeros(n)
    moves = np.zeros(n)

    btc_sma = _compute_sma_inline(btc_close[:n], btc_regime_period)
    rsi = _compute_rsi_inline(close[:n], rsi_period)
    vol_sma = _compute_sma_inline(volume[:n], vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    if stoch_enabled:
        stoch_k, stoch_d = _compute_stochastic_inline(
            high[:n], low[:n], close[:n], stoch_k_period, stoch_d_period)
    else:
        stoch_k = np.full(n, 50.0)
        stoch_d = np.full(n, 50.0)

    if macd_enabled:
        _, _, macd_hist = _compute_macd_inline(
            close[:n], macd_fast, macd_slow, macd_signal_period)
    else:
        macd_hist = np.full(n, 0.0)

    if atr_period > 0:
        atr_pct = compute_atr_pct(high[:n], low[:n], close[:n], atr_period)
    else:
        atr_pct = np.full(n, np.nan)

    start_bar = max(lookback, btc_mom_period, btc_regime_period,
                    asset_trend_period, rsi_period, vol_period)
    if atr_period > 0 and atr_period > start_bar:
        start_bar = atr_period

    for i in range(start_bar, n):
        ret = close[i] / close[i - lookback] - 1.0

        if atr_period > 0:
            atr_val = atr_pct[i]
            if np.isnan(atr_val):
                thr = threshold
            else:
                thr = max(threshold_floor, min(atr_multiplier * atr_val, threshold_cap))
        else:
            thr = threshold

        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
        direction = 0
        confidence = 0

        if abs(ret) > thr:
            if ret < -thr and btc_mom < 0:
                direction = 1
                if rsi[i] < 35: confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]: confidence += 1
                if taker_ratio[i] < 0.42: confidence += 1
                if btc_mom < -0.01: confidence += 1
                if stoch_enabled and i > 0:
                    if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]:
                        confidence += 1
                if macd_enabled and i > 0:
                    if macd_hist[i] > macd_hist[i-1] and macd_hist[i-1] < 0:
                        confidence += 1
                if confidence < min_confidence:
                    direction = 0

            elif ret > thr and btc_mom > 0:
                direction = -1
                if rsi[i] > 65: confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]: confidence += 1
                if taker_ratio[i] > 0.58: confidence += 1
                if btc_mom > 0.01: confidence += 1
                if stoch_enabled and i > 0:
                    if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]:
                        confidence += 1
                if macd_enabled and i > 0:
                    if macd_hist[i] < macd_hist[i-1] and macd_hist[i-1] > 0:
                        confidence += 1
                if confidence < min_confidence:
                    direction = 0

        if direction != 0:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            moves[i] = abs(ret)

    return dirs, confs, moves


def apply_mtf_filter(dirs, confs, moves, close_4h, sma_period=12, ratio=16):
    """Zero out entries against the 4h trend."""
    n_4h = len(close_4h)
    sma_4h = np.full(n_4h, np.nan)
    s = 0.0
    for j in range(n_4h):
        s += close_4h[j]
        if j >= sma_period:
            s -= close_4h[j - sma_period]
        if j >= sma_period - 1:
            sma_4h[j] = s / sma_period

    od, oc, om = dirs.copy(), confs.copy(), moves.copy()
    for i in range(len(dirs)):
        if dirs[i] == 0.0:
            continue
        j = min(i // ratio, n_4h - 1)
        if j < sma_period or np.isnan(sma_4h[j]):
            continue
        trend_up = close_4h[j] > sma_4h[j]
        if dirs[i] > 0 and not trend_up:
            od[i] = 0.0; oc[i] = 0.0; om[i] = 0.0
        elif dirs[i] < 0 and trend_up:
            od[i] = 0.0; oc[i] = 0.0; om[i] = 0.0
    return od, oc, om


# ═══════════════════════════════════════════════════════════════════
# VWMA computation (from R10)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def compute_vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume-Weighted Moving Average.

    VWMA[i] = sum(close[j]*volume[j] for j in [i-period+1..i]) / sum(volume[j])
    Returns NaN before warmup or when total volume in window is zero.
    """
    n = len(close)
    vwma = np.full(n, np.nan)
    pv_sum = 0.0
    v_sum = 0.0
    for i in range(n):
        pv_sum += close[i] * volume[i]
        v_sum += volume[i]
        if i >= period:
            pv_sum -= close[i - period] * volume[i - period]
            v_sum -= volume[i - period]
        if i >= period - 1 and v_sum > 0:
            vwma[i] = pv_sum / v_sum
    return vwma


# ═══════════════════════════════════════════════════════════════════
# VPVR: Volume Profile Visible Range — POC + Value Area
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def compute_vpvr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    lookback: int,
    n_bins: int,
    va_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling Volume Profile: POC, VA High, VA Low.

    For each bar i, looks back `lookback` bars and builds a volume histogram
    across `n_bins` price bins. The bin with highest volume is the POC.
    The Value Area is the narrowest contiguous range of bins containing
    `va_pct` (70%) of total volume.

    Returns:
        poc: array of POC prices (NaN before warmup)
        va_high: array of VA High prices
        va_low: array of VA Low prices
    """
    n = len(close)
    poc = np.full(n, np.nan)
    va_high = np.full(n, np.nan)
    va_low = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        # Find price range in the lookback window
        start = i - lookback + 1
        p_min = low[start]
        p_max = high[start]
        for j in range(start + 1, i + 1):
            if low[j] < p_min:
                p_min = low[j]
            if high[j] > p_max:
                p_max = high[j]

        price_range = p_max - p_min
        if price_range < 1e-10:
            poc[i] = close[i]
            va_high[i] = close[i]
            va_low[i] = close[i]
            continue

        bin_size = price_range / n_bins

        # Build volume histogram across bins
        # Distribute each bar's volume across all bins that the bar's range covers
        vol_hist = np.zeros(n_bins)
        for j in range(start, i + 1):
            bar_vol = volume[j]
            if bar_vol <= 0:
                continue
            # Which bins does this bar span?
            lo_bin = int((low[j] - p_min) / bin_size)
            hi_bin = int((high[j] - p_min) / bin_size)
            if lo_bin < 0:
                lo_bin = 0
            if hi_bin >= n_bins:
                hi_bin = n_bins - 1
            if lo_bin > hi_bin:
                lo_bin = hi_bin
            n_covered = hi_bin - lo_bin + 1
            per_bin_vol = bar_vol / n_covered
            for b in range(lo_bin, hi_bin + 1):
                vol_hist[b] += per_bin_vol

        # POC: bin with maximum volume
        max_vol = -1.0
        poc_bin = 0
        for b in range(n_bins):
            if vol_hist[b] > max_vol:
                max_vol = vol_hist[b]
                poc_bin = b
        poc[i] = p_min + (poc_bin + 0.5) * bin_size

        # Value Area: expand from POC bin outward until 70% volume captured
        total_vol = 0.0
        for b in range(n_bins):
            total_vol += vol_hist[b]

        if total_vol <= 0:
            va_high[i] = p_max
            va_low[i] = p_min
            continue

        va_vol_target = total_vol * va_pct
        va_lo_bin = poc_bin
        va_hi_bin = poc_bin
        captured = vol_hist[poc_bin]

        while captured < va_vol_target:
            # Expand to the side with more volume
            can_lo = va_lo_bin > 0
            can_hi = va_hi_bin < n_bins - 1
            if not can_lo and not can_hi:
                break

            lo_vol = vol_hist[va_lo_bin - 1] if can_lo else -1.0
            hi_vol = vol_hist[va_hi_bin + 1] if can_hi else -1.0

            if lo_vol >= hi_vol:
                va_lo_bin -= 1
                captured += vol_hist[va_lo_bin]
            else:
                va_hi_bin += 1
                captured += vol_hist[va_hi_bin]

        va_low[i] = p_min + va_lo_bin * bin_size
        va_high[i] = p_min + (va_hi_bin + 1) * bin_size

    return poc, va_high, va_low


# ═══════════════════════════════════════════════════════════════════
# Factor precomputation
# ═══════════════════════════════════════════════════════════════════
def compute_dual_confirm_regime(btc_4h_close: np.ndarray, n_15m: int) -> np.ndarray:
    """Dual-confirm regime: +1 bull, -1 bear, 0 choppy."""
    n_4h = len(btc_4h_close)
    sma50 = np.full(n_4h, np.nan)
    sma200 = np.full(n_4h, np.nan)
    s50, s200 = 0.0, 0.0
    for j in range(n_4h):
        s50 += btc_4h_close[j]
        s200 += btc_4h_close[j]
        if j >= 50: s50 -= btc_4h_close[j - 50]
        if j >= 200: s200 -= btc_4h_close[j - 200]
        if j >= 49: sma50[j] = s50 / 50
        if j >= 199: sma200[j] = s200 / 200

    regime_4h = np.zeros(n_4h)
    for j in range(n_4h):
        if np.isnan(sma50[j]) or np.isnan(sma200[j]):
            regime_4h[j] = 0
        elif btc_4h_close[j] > sma50[j] and btc_4h_close[j] > sma200[j]:
            regime_4h[j] = 1
        elif btc_4h_close[j] < sma50[j] and btc_4h_close[j] < sma200[j]:
            regime_4h[j] = -1
        else:
            regime_4h[j] = 0

    regime_15m = np.zeros(n_15m)
    for i in range(n_15m):
        j = min(i // 16, n_4h - 1)
        regime_15m[i] = regime_4h[j]
    return regime_15m


# ═══════════════════════════════════════════════════════════════════
# Part A: Factor Independence Analysis
# ═══════════════════════════════════════════════════════════════════
def factor_analysis(trades: list[dict]):
    """Analyze factors: VWMA dist, POC dist, VA position, POC-implied R:R."""
    w = 110
    print(f"\n{'═' * w}")
    print(f"  PART A: FACTOR ANALYSIS — {len(trades)} trades")
    print(f"{'═' * w}")

    factor_names = [
        ('f_vwma_dist', 'VWMA(100) Dist'),
        ('f_poc_dist', 'POC Distance'),
        ('f_va_pos', 'VA Position'),
        ('f_poc_rr', 'POC Implied R:R'),
        ('f_vwma_poc_agree', 'VWMA-POC Agreement'),
    ]

    for fkey, flabel in factor_names:
        vals = [t[fkey] for t in trades if not np.isnan(t.get(fkey, np.nan))]
        if not vals:
            print(f"\n  {flabel}: NO DATA")
            continue

        vals_arr = np.array(vals)
        print(f"\n  ── {flabel} ──")
        print(f"  Distribution: min={vals_arr.min():.4f}  p25={np.percentile(vals_arr, 25):.4f}  "
              f"med={np.median(vals_arr):.4f}  p75={np.percentile(vals_arr, 75):.4f}  "
              f"max={vals_arr.max():.4f}")

        # Special handling for VA Position (categorical-ish)
        if fkey == 'f_va_pos':
            groups = [
                ('Below VA', [t for t in trades if t.get(fkey, np.nan) < -0.5]),
                ('Inside VA', [t for t in trades if -0.5 <= t.get(fkey, np.nan) <= 0.5]),
                ('Above VA', [t for t in trades if t.get(fkey, np.nan) > 0.5]),
            ]
        elif fkey == 'f_vwma_poc_agree':
            groups = [
                ('Disagree (0)', [t for t in trades if t.get(fkey, np.nan) < 0.5]),
                ('Agree (1)', [t for t in trades if t.get(fkey, np.nan) >= 0.5]),
            ]
        else:
            p33 = np.percentile(vals_arr, 33.3)
            p67 = np.percentile(vals_arr, 66.7)
            valid_trades = [t for t in trades if not np.isnan(t.get(fkey, np.nan))]
            groups = [
                (f'Lo (<{p33:.3f})', [t for t in valid_trades if t[fkey] < p33]),
                (f'Mid', [t for t in valid_trades if p33 <= t[fkey] <= p67]),
                (f'Hi (>{p67:.3f})', [t for t in valid_trades if t[fkey] > p67]),
            ]

        print(f"  {'Tercile':<22} {'#Tr':>6} {'WR%':>6} {'PF':>6} {'R:R':>6} {'EV%':>8} {'Total$':>10}")
        print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")

        ev_values = []
        for glabel, gtrades in groups:
            if not gtrades:
                print(f"  {glabel:<22} {'—':>6}")
                ev_values.append(0.0)
                continue
            n = len(gtrades)
            w_ = [t for t in gtrades if t['pnl_usd'] > 0]
            l_ = [t for t in gtrades if t['pnl_usd'] <= 0]
            wr = len(w_) / n * 100
            gp = sum(t['pnl_usd'] for t in w_)
            gl = sum(abs(t['pnl_usd']) for t in l_)
            pf = gp / gl if gl > 0 else 999.0
            aw = float(np.mean([t['pnl_pct'] for t in w_])) if w_ else 0
            al = float(np.mean([abs(t['pnl_pct']) for t in l_])) if l_ else 1
            rr = aw / al if al > 0 else 999.0
            ev = float(np.mean([t['pnl_pct'] for t in gtrades]))
            total = sum(t['pnl_usd'] for t in gtrades)
            print(f"  {glabel:<22} {n:>6d} {wr:>5.1f}% {pf:>5.2f} {rr:>5.2f} {ev:>7.3f}% {total:>10.1f}")
            ev_values.append(ev)

        # Monotonic check
        if len(ev_values) >= 3 and all(v != 0 for v in ev_values):
            if ev_values[0] < ev_values[1] < ev_values[2]:
                print(f"  ⚡ MONOTONIC INCREASING — factor predicts higher PnL ↑")
            elif ev_values[0] > ev_values[1] > ev_values[2]:
                print(f"  ⚡ MONOTONIC DECREASING — factor predicts lower PnL ↓")
            else:
                spread = max(ev_values) - min(ev_values)
                print(f"  ⚡ Non-monotonic — spread: {spread:.3f}%")

    # Pairwise correlation matrix
    print(f"\n  ── Pairwise Correlations ──")
    fkeys = [fk for fk, _ in factor_names]
    factor_arrays = {}
    for fk in fkeys:
        arr = np.array([t.get(fk, np.nan) for t in trades])
        factor_arrays[fk] = arr

    print(f"  {'':>20}", end='')
    for _, fl in factor_names:
        print(f" {fl[:12]:>12}", end='')
    print()

    for i, (fk_i, fl_i) in enumerate(factor_names):
        print(f"  {fl_i:>20}", end='')
        for j, (fk_j, fl_j) in enumerate(factor_names):
            ai = factor_arrays[fk_i]
            aj = factor_arrays[fk_j]
            mask = ~np.isnan(ai) & ~np.isnan(aj)
            if mask.sum() > 10:
                corr = np.corrcoef(ai[mask], aj[mask])[0, 1]
                print(f" {corr:>11.3f} ", end='')
            else:
                print(f" {'—':>12}", end='')
        print()

    # Factor-PnL correlation
    print(f"\n  ── Factor → PnL Correlation ──")
    pnl_arr = np.array([t['pnl_pct'] for t in trades])
    for fk, fl in factor_names:
        arr = factor_arrays[fk]
        mask = ~np.isnan(arr)
        if mask.sum() > 10:
            corr = np.corrcoef(arr[mask], pnl_arr[mask])[0, 1]
            strength = "STRONG" if abs(corr) > 0.05 else "MODERATE" if abs(corr) > 0.02 else "WEAK"
            print(f"  {fl:<24} r = {corr:+.4f}  ({strength})")
        else:
            print(f"  {fl:<24} — insufficient data")


# ═══════════════════════════════════════════════════════════════════
# Simulation engine (VWMA + POC/VA support)
# ═══════════════════════════════════════════════════════════════════
def simulate(
    sym_close, sym_high, sym_low,
    sym_dirs, sym_confs, sym_moves,
    cfg: VConfig,
    regime: np.ndarray,
    sym_atr=None,
    sym_vwma=None,
    sym_poc=None,
    sym_va_high=None,
    sym_va_low=None,
) -> dict:
    """Run portfolio sim with VWMA + VPVR/POC adaptive parameters.

    New vs R10:
    - POC stop/target: size by distance to POC
    - VA entry gate: only enter when outside Value Area
    - VA stop: anchor stop to VA boundary
    - VA target: target opposite VA boundary
    - R:R sizing: scale position by implied R:R
    """
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)
    n_bars = min(n_bars, len(regime))

    balance = INITIAL_CAPITAL
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance
    max_dd = 0.0

    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for i in range(1, n_bars):
        to_close: list[str] = []

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            ep = pos['entry_price']
            lev = pos['leverage']
            d = pos['direction']
            gain = (c - ep) / ep if d == 1 else (ep - c) / ep

            # ═══ STOP COMPUTATION ═══
            base_stop = pos['move_size'] * cfg.stop_mult

            # VWMA stop layer
            vw_dist = pos.get('vwma_dist', 0.0)
            if cfg.vwma_stop != 'off' and vw_dist > 0:
                vw_stop = max(vw_dist * cfg.vwma_stop_mult, 0.003)
                if cfg.vwma_stop == 'replace':
                    base_stop = vw_stop
                elif cfg.vwma_stop == 'tight':
                    base_stop = min(base_stop, vw_stop)
                elif cfg.vwma_stop == 'wide':
                    base_stop = max(base_stop, vw_stop)

            # POC stop layer
            poc_dist = pos.get('poc_dist', 0.0)
            if cfg.poc_stop != 'off' and poc_dist > 0:
                p_stop = max(poc_dist * cfg.poc_stop_mult, 0.003)
                if cfg.poc_stop == 'replace':
                    base_stop = p_stop
                elif cfg.poc_stop == 'tight':
                    base_stop = min(base_stop, p_stop)
                elif cfg.poc_stop == 'wide':
                    base_stop = max(base_stop, p_stop)

            # VA stop layer: use distance to VA boundary
            va_stop_dist = pos.get('va_stop_dist', 0.0)
            if cfg.va_stop != 'off' and va_stop_dist > 0:
                va_s = va_stop_dist + cfg.va_stop_margin
                if cfg.va_stop == 'replace':
                    base_stop = va_s
                elif cfg.va_stop == 'tight':
                    base_stop = min(base_stop, va_s)
                elif cfg.va_stop == 'wide':
                    base_stop = max(base_stop, va_s)

            effective_stop = base_stop

            # ═══ TARGET COMPUTATION ═══
            base_target = pos['move_size'] * cfg.target_retrace

            # VWMA target layer
            if cfg.vwma_target != 'off' and vw_dist > 0:
                vw_tgt = max(vw_dist * cfg.vwma_target_mult, 0.003)
                if cfg.vwma_target == 'replace':
                    base_target = vw_tgt
                elif cfg.vwma_target == 'wide':
                    base_target = max(base_target, vw_tgt)

            # POC target layer
            if cfg.poc_target != 'off' and poc_dist > 0:
                p_tgt = max(poc_dist * cfg.poc_target_mult, 0.003)
                if cfg.poc_target == 'replace':
                    base_target = p_tgt
                elif cfg.poc_target == 'wide':
                    base_target = max(base_target, p_tgt)

            # VA target layer: target opposite VA boundary
            va_tgt_dist = pos.get('va_tgt_dist', 0.0)
            if cfg.va_target != 'off' and va_tgt_dist > 0:
                if cfg.va_target == 'replace':
                    base_target = va_tgt_dist
                elif cfg.va_target == 'wide':
                    base_target = max(base_target, va_tgt_dist)

            target = base_target
            max_hold = cfg.max_hold

            # ═══ PBT Phase 1: Pre-TP1 ═══
            if cfg.partial_exit and not pos.get('tp1_hit', False):
                if gain >= target:
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain
                    tp1_not = pos['notional'] * cfg.partial_frac
                    rem_not = pos['notional'] - tp1_not
                    funding = lev * FUNDING_PER_BAR * tp1_not * pos['hold_count']
                    cost = lev * COST_PER_SIDE * tp1_not
                    net = gain * lev * tp1_not - cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': d,
                        'confidence': pos['confidence'],
                        'exit_reason': 'tp1',
                        'pnl_usd': net,
                        'pnl_pct': (net / tp1_not) * 100,
                        'bars_held': pos['hold_count'],
                        'entry_bar': pos['entry_bar'],
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0 if cfg.be_after_tp1 else -effective_stop
                elif gain < -effective_stop:
                    _close(balance, trades, pos, sym, gain, lev, 'stop', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= max_hold:
                    _close(balance, trades, pos, sym, gain, lev, 'time', to_close, i)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        _close(balance, trades, pos, sym, gain, lev, 'trail', to_close, i)
                        balance = trades[-1]['_b']

            # ═══ PBT Phase 2: Runner (post-TP1) ═══
            elif cfg.partial_exit and pos.get('tp1_hit', False):
                if gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain

                atr_val = 0.02
                if sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        atr_val = aa[i]

                trail_stop = pos['highest_since_tp1'] - cfg.trail_atr * atr_val
                floor_val = pos.get('trail_floor', -effective_stop)
                if trail_stop < floor_val:
                    trail_stop = floor_val

                exit_runner = False
                exit_reason = None
                if gain <= trail_stop:
                    exit_runner = True; exit_reason = 'trail_runner'
                elif pos['hold_count'] >= max_hold * cfg.runner_mh_mult:
                    exit_runner = True; exit_reason = 'time_runner'

                if exit_runner:
                    tp1_bar = pos.get('tp1_bar', pos['entry_bar'])
                    funding = lev * FUNDING_PER_BAR * pos['notional'] * (i - tp1_bar)
                    cost = lev * COST_PER_SIDE * pos['notional']
                    net = gain * lev * pos['notional'] - cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': d,
                        'confidence': pos['confidence'],
                        'exit_reason': exit_reason,
                        'pnl_usd': net,
                        'pnl_pct': (net / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                        'entry_bar': pos['entry_bar'],
                        '_b': balance,
                    })
                    to_close.append(sym)
            else:
                # Non-PBT fallback
                exit_reason = None
                if gain >= target:
                    exit_reason = 'target'
                elif gain < -effective_stop:
                    exit_reason = 'stop'
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        exit_reason = 'trail'
                    elif pos['hold_count'] >= max_hold:
                        exit_reason = 'time'
                if exit_reason:
                    _close(balance, trades, pos, sym, gain, lev, exit_reason, to_close, i)
                    balance = trades[-1]['_b']

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Entry phase ═══
        if len(positions) < cfg.max_slots:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                d_arr = sym_dirs[sym]
                if i >= len(d_arr) or d_arr[i] == 0.0:
                    continue
                direction = int(d_arr[i])
                confidence = int(sym_confs[sym][i])
                move_size = float(sym_moves[sym][i])

                c_entry = sym_close[sym][i]

                # Read VWMA at entry
                vwma_val = np.nan
                vwma_dist_entry = 0.0
                if sym_vwma is not None:
                    vwma_arr = sym_vwma.get(sym)
                    if vwma_arr is not None and i < len(vwma_arr):
                        vwma_val = vwma_arr[i]
                if not np.isnan(vwma_val):
                    if direction == 1:
                        vwma_dist_entry = (vwma_val - c_entry) / c_entry
                    else:
                        vwma_dist_entry = (c_entry - vwma_val) / c_entry
                    vwma_dist_entry = max(vwma_dist_entry, 0.001)

                # VWMA entry gate
                if cfg.vwma_entry_min > 0.0:
                    if np.isnan(vwma_val):
                        continue
                    dir_vwma = (vwma_val - c_entry) / c_entry if direction == 1 else (c_entry - vwma_val) / c_entry
                    if dir_vwma < cfg.vwma_entry_min:
                        continue

                # Read POC / VA at entry
                poc_val = np.nan
                va_hi_val = np.nan
                va_lo_val = np.nan
                poc_dist_entry = 0.0
                va_stop_dist_entry = 0.0
                va_tgt_dist_entry = 0.0

                if sym_poc is not None:
                    poc_arr = sym_poc.get(sym)
                    if poc_arr is not None and i < len(poc_arr):
                        poc_val = poc_arr[i]
                if sym_va_high is not None:
                    vh_arr = sym_va_high.get(sym)
                    if vh_arr is not None and i < len(vh_arr):
                        va_hi_val = vh_arr[i]
                if sym_va_low is not None:
                    vl_arr = sym_va_low.get(sym)
                    if vl_arr is not None and i < len(vl_arr):
                        va_lo_val = vl_arr[i]

                # POC distance (directional: positive = POC is in the reversion direction)
                if not np.isnan(poc_val) and c_entry > 0:
                    if direction == 1:  # long: POC above = room to revert up
                        poc_dist_entry = (poc_val - c_entry) / c_entry
                    else:  # short: POC below = room to revert down
                        poc_dist_entry = (c_entry - poc_val) / c_entry
                    poc_dist_entry = max(poc_dist_entry, 0.001)

                # VA position: -1=below VA, 0=inside VA, +1=above VA (directional)
                va_pos = 0.0
                if not np.isnan(va_hi_val) and not np.isnan(va_lo_val):
                    if c_entry < va_lo_val:
                        va_pos = -1.0  # below VA
                    elif c_entry > va_hi_val:
                        va_pos = 1.0   # above VA
                    else:
                        va_pos = 0.0   # inside VA

                    # VA stop distance: for long, distance below VA low
                    # for short, distance above VA high
                    if direction == 1:
                        va_stop_dist_entry = (c_entry - va_lo_val) / c_entry
                        va_stop_dist_entry = max(va_stop_dist_entry, 0.001)
                        # VA target: reach VA high
                        va_tgt_dist_entry = (va_hi_val - c_entry) / c_entry
                        va_tgt_dist_entry = max(va_tgt_dist_entry, 0.001)
                    else:
                        va_stop_dist_entry = (va_hi_val - c_entry) / c_entry
                        va_stop_dist_entry = max(va_stop_dist_entry, 0.001)
                        # VA target: reach VA low
                        va_tgt_dist_entry = (c_entry - va_lo_val) / c_entry
                        va_tgt_dist_entry = max(va_tgt_dist_entry, 0.001)

                # VA entry gate: only enter when outside Value Area
                if cfg.va_entry_gate:
                    if not np.isnan(va_hi_val) and not np.isnan(va_lo_val):
                        if va_lo_val <= c_entry <= va_hi_val:
                            continue  # inside VA → skip
                    else:
                        continue  # no VA data → skip

                # Direction diversity
                if cfg.dir_div:
                    same = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same >= 2:
                        continue

                candidates.append((
                    sym, direction, confidence, move_size,
                    vwma_dist_entry, poc_dist_entry,
                    va_stop_dist_entry, va_tgt_dist_entry,
                    va_pos,
                ))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for (sym, direction, confidence, move_size,
                 vwma_dist_e, poc_dist_e, va_stop_e, va_tgt_e, va_pos_e) in candidates:
                if len(positions) >= cfg.max_slots:
                    break

                per_slot = INITIAL_CAPITAL / max(1, len(positions) + 1)
                per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if per_slot < 10.0:
                    continue

                if cfg.steep_cap:
                    mult = CAP_MULT.get(confidence, CAP_MULT.get(min(confidence, 5), 0.3))
                    per_slot *= mult
                    per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if cfg.vol_adj and sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]) and aa[i] > 0:
                        vf = max(0.4, min(0.01 / aa[i], 2.0))
                        per_slot *= vf
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if cfg.soft_eq:
                    eq_now = equity_curve[i-1] if i > 0 else INITIAL_CAPITAL
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    for dd_thr, scale in SOFT_EQ:
                        if dd_pct > dd_thr:
                            per_slot *= scale
                            break

                # R:R-gated sizing
                if cfg.rr_size and poc_dist_e > 0:
                    # Implied R:R = target_dist / stop_dist using POC references
                    implied_target = poc_dist_e  # POC distance as target
                    implied_stop = move_size * cfg.stop_mult  # base stop
                    if implied_stop > 0:
                        implied_rr = implied_target / implied_stop
                        # Scale: rr=1 → 1.0x, rr=2 → 1.5x, rr=0.5 → 0.75x
                        rr_factor = max(cfg.rr_size_min, min(0.5 + 0.5 * implied_rr, cfg.rr_size_max))
                        per_slot *= rr_factor
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost

                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': sym_close[sym][i],
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': EFFECTIVE_LEV,
                    'entry_regime': int(regime[i]),
                    'vwma_dist': vwma_dist_e,
                    'poc_dist': poc_dist_e,
                    'va_stop_dist': va_stop_e,
                    'va_tgt_dist': va_tgt_e,
                }

        # Equity curve
        total_unr = 0.0
        for sym, pos in positions.items():
            c = sym_close[sym][i]
            ep = pos['entry_price']
            g = (c - ep) / ep if pos['direction'] == 1 else (ep - c) / ep
            total_unr += g * pos['leverage'] * pos['notional']
        equity_curve[i] = balance + total_unr
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return _compute_stats(equity_curve, trades, max_dd)


def _close(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx):
    funding = lev * FUNDING_PER_BAR * pos['notional'] * pos['hold_count']
    cost = lev * COST_PER_SIDE * pos['notional']
    net = gain * lev * pos['notional'] - cost - funding
    new_b = balance + net
    trades.append({
        'symbol': sym, 'direction': pos['direction'],
        'confidence': pos['confidence'],
        'exit_reason': reason,
        'pnl_usd': net,
        'pnl_pct': (net / pos['notional']) * 100,
        'bars_held': pos['hold_count'],
        'entry_bar': pos['entry_bar'],
        '_b': new_b,
    })
    to_close.append(sym)


def _compute_stats(eq, trades, max_dd):
    n = len(trades)
    if n == 0:
        return {'n_trades': 0}

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    long_t = [t for t in trades if t['direction'] == 1]
    short_t = [t for t in trades if t['direction'] == -1]

    total_ret = (eq[-1] / INITIAL_CAPITAL - 1.0) * 100
    gp = sum(t['pnl_usd'] for t in wins)
    gl = sum(abs(t['pnl_usd']) for t in losses)
    pf = gp / gl if gl > 0 else 999.0

    eq_rets = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)) if len(eq_rets) > 0 and np.std(eq_rets) > 0 else 0.0

    aw = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0
    al = float(np.mean([abs(t['pnl_pct']) for t in losses])) if losses else 1.0
    rr = aw / al if al > 0 else 999.0
    n_days = len(eq) / 96

    def _dir_stats(tl):
        if not tl:
            return {'count': 0, 'wr': 0, 'avg_pnl': 0, 'rr': 0, 'pf': 0, 'total_pnl': 0}
        w = [t for t in tl if t['pnl_usd'] > 0]
        l = [t for t in tl if t['pnl_usd'] <= 0]
        wr = len(w) / len(tl) * 100
        gp_d = sum(t['pnl_usd'] for t in w)
        gl_d = sum(abs(t['pnl_usd']) for t in l)
        pf_d = gp_d / gl_d if gl_d > 0 else 999.0
        aw_d = float(np.mean([t['pnl_pct'] for t in w])) if w else 0
        al_d = float(np.mean([abs(t['pnl_pct']) for t in l])) if l else 1
        rr_d = aw_d / al_d if al_d > 0 else 999.0
        avg_d = float(np.mean([t['pnl_pct'] for t in tl]))
        total_d = sum(t['pnl_usd'] for t in tl)
        return {
            'count': len(tl), 'wr': round(wr, 1),
            'avg_pnl': round(avg_d, 3), 'rr': round(rr_d, 2),
            'pf': round(pf_d, 2), 'total_pnl': round(total_d, 2),
        }

    return {
        'total_return_pct': round(total_ret, 1),
        'final_balance': round(eq[-1], 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(total_ret / (max_dd * 100), 1) if max_dd > 0 else 0.0,
        'n_trades': n,
        'trades_per_day': round(n / n_days, 2) if n_days > 0 else 0.0,
        'win_rate': round(len(wins) / n * 100, 1),
        'profit_factor': round(pf, 2),
        'rr': round(rr, 2),
        'long': _dir_stats(long_t),
        'short': _dir_stats(short_t),
    }


# ═══════════════════════════════════════════════════════════════════
# Baseline trade extraction with factor tagging
# ═══════════════════════════════════════════════════════════════════
def get_tagged_trades(
    sym_close, sym_high, sym_low,
    sym_dirs, sym_confs, sym_moves,
    sym_atr, sym_vwma,
    sym_poc, sym_va_high, sym_va_low,
) -> list[dict]:
    """Run baseline simulation and tag each trade with factor values at entry."""
    cfg = VConfig(name='TAG-Baseline')
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    balance = INITIAL_CAPITAL
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance
    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for i in range(1, n_bars):
        to_close: list[str] = []

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            ep = pos['entry_price']
            lev = pos['leverage']
            d = pos['direction']
            gain = (c - ep) / ep if d == 1 else (ep - c) / ep

            target = pos['move_size'] * cfg.target_retrace
            effective_stop = pos['move_size'] * cfg.stop_mult

            if cfg.partial_exit and not pos.get('tp1_hit', False):
                if gain >= target:
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain
                    tp1_not = pos['notional'] * cfg.partial_frac
                    rem_not = pos['notional'] - tp1_not
                    funding = lev * FUNDING_PER_BAR * tp1_not * pos['hold_count']
                    cost = lev * COST_PER_SIDE * tp1_not
                    net = gain * lev * tp1_not - cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': d,
                        'confidence': pos['confidence'],
                        'exit_reason': 'tp1',
                        'pnl_usd': net, 'pnl_pct': (net / tp1_not) * 100,
                        'bars_held': pos['hold_count'],
                        'entry_bar': pos['entry_bar'],
                        **{k: pos.get(k, np.nan) for k in [
                            'f_vwma_dist', 'f_poc_dist', 'f_va_pos',
                            'f_poc_rr', 'f_vwma_poc_agree',
                        ]},
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0
                elif gain < -effective_stop:
                    _close_tagged(balance, trades, pos, sym, gain, lev, 'stop', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= cfg.max_hold:
                    _close_tagged(balance, trades, pos, sym, gain, lev, 'time', to_close, i)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        _close_tagged(balance, trades, pos, sym, gain, lev, 'trail', to_close, i)
                        balance = trades[-1]['_b']
            elif cfg.partial_exit and pos.get('tp1_hit', False):
                if gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain
                atr_val = 0.02
                if sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        atr_val = aa[i]
                trail_stop = pos['highest_since_tp1'] - cfg.trail_atr * atr_val
                floor_val = pos.get('trail_floor', -effective_stop)
                if trail_stop < floor_val:
                    trail_stop = floor_val
                exit_runner = False; exit_reason = None
                if gain <= trail_stop:
                    exit_runner = True; exit_reason = 'trail_runner'
                elif pos['hold_count'] >= cfg.max_hold * cfg.runner_mh_mult:
                    exit_runner = True; exit_reason = 'time_runner'
                if exit_runner:
                    tp1_bar = pos.get('tp1_bar', pos['entry_bar'])
                    funding = lev * FUNDING_PER_BAR * pos['notional'] * (i - tp1_bar)
                    cost = lev * COST_PER_SIDE * pos['notional']
                    net = gain * lev * pos['notional'] - cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': d,
                        'confidence': pos['confidence'],
                        'exit_reason': exit_reason,
                        'pnl_usd': net, 'pnl_pct': (net / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                        'entry_bar': pos['entry_bar'],
                        **{k: pos.get(k, np.nan) for k in [
                            'f_vwma_dist', 'f_poc_dist', 'f_va_pos',
                            'f_poc_rr', 'f_vwma_poc_agree',
                        ]},
                        '_b': balance,
                    })
                    to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        if len(positions) < cfg.max_slots:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                d_arr = sym_dirs[sym]
                if i >= len(d_arr) or d_arr[i] == 0.0:
                    continue
                direction = int(d_arr[i])
                confidence = int(sym_confs[sym][i])
                move_size = float(sym_moves[sym][i])
                if cfg.dir_div:
                    same = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same >= 2:
                        continue
                candidates.append((sym, direction, confidence, move_size))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size in candidates:
                if len(positions) >= cfg.max_slots:
                    break
                per_slot = INITIAL_CAPITAL / max(1, len(positions) + 1)
                per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if per_slot < 10.0:
                    continue
                if cfg.steep_cap:
                    mult = CAP_MULT.get(confidence, CAP_MULT.get(min(confidence, 5), 0.3))
                    per_slot *= mult; per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if cfg.vol_adj and sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]) and aa[i] > 0:
                        vf = max(0.4, min(0.01 / aa[i], 2.0))
                        per_slot *= vf; per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if cfg.soft_eq:
                    eq_now = equity_curve[i-1] if i > 0 else INITIAL_CAPITAL
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    for dd_thr, scale in SOFT_EQ:
                        if dd_pct > dd_thr:
                            per_slot *= scale; break

                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost

                c_entry = sym_close[sym][i]

                # Tag VWMA distance
                f_vwma = np.nan
                if sym_vwma is not None:
                    va = sym_vwma.get(sym)
                    if va is not None and i < len(va) and not np.isnan(va[i]):
                        vw = va[i]
                        if direction == 1:
                            f_vwma = (vw - c_entry) / c_entry
                        else:
                            f_vwma = (c_entry - vw) / c_entry

                # Tag POC distance (directional)
                f_poc = np.nan
                poc_val = np.nan
                if sym_poc is not None:
                    pa = sym_poc.get(sym)
                    if pa is not None and i < len(pa):
                        poc_val = pa[i]
                if not np.isnan(poc_val) and c_entry > 0:
                    if direction == 1:
                        f_poc = (poc_val - c_entry) / c_entry
                    else:
                        f_poc = (c_entry - poc_val) / c_entry

                # Tag VA position: convert to directional
                # For longs: below VA (-1) is favorable (dislocation below fair range)
                # For shorts: above VA (+1) is favorable
                f_va_pos = np.nan
                va_hi_val = np.nan
                va_lo_val = np.nan
                if sym_va_high is not None and sym_va_low is not None:
                    vh = sym_va_high.get(sym)
                    vl = sym_va_low.get(sym)
                    if vh is not None and vl is not None and i < len(vh) and i < len(vl):
                        va_hi_val = vh[i]
                        va_lo_val = vl[i]
                if not np.isnan(va_hi_val) and not np.isnan(va_lo_val):
                    if c_entry < va_lo_val:
                        raw_pos = -1.0  # below VA
                    elif c_entry > va_hi_val:
                        raw_pos = 1.0   # above VA
                    else:
                        raw_pos = 0.0   # inside VA
                    # Directional: "favorable dislocation" should be positive
                    if direction == 1:
                        f_va_pos = -raw_pos  # below VA → positive for longs
                    else:
                        f_va_pos = raw_pos   # above VA → positive for shorts

                # Tag POC-implied R:R
                f_poc_rr = np.nan
                base_stop_dist = move_size * cfg.stop_mult
                if not np.isnan(f_poc) and base_stop_dist > 0:
                    poc_target = max(f_poc, 0.001)
                    f_poc_rr = poc_target / base_stop_dist

                # Tag VWMA-POC agreement (both point same direction)
                f_agree = np.nan
                if not np.isnan(f_vwma) and not np.isnan(f_poc):
                    f_agree = 1.0 if (f_vwma > 0 and f_poc > 0) else 0.0

                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': c_entry,
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': EFFECTIVE_LEV,
                    'f_vwma_dist': f_vwma,
                    'f_poc_dist': f_poc,
                    'f_va_pos': f_va_pos,
                    'f_poc_rr': f_poc_rr,
                    'f_vwma_poc_agree': f_agree,
                }

        total_unr = 0.0
        for sym, pos in positions.items():
            c = sym_close[sym][i]
            ep = pos['entry_price']
            g = (c - ep) / ep if pos['direction'] == 1 else (ep - c) / ep
            total_unr += g * pos['leverage'] * pos['notional']
        equity_curve[i] = balance + total_unr
        if equity_curve[i] > peak:
            peak = equity_curve[i]

    return trades


def _close_tagged(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx):
    funding = lev * FUNDING_PER_BAR * pos['notional'] * pos['hold_count']
    cost = lev * COST_PER_SIDE * pos['notional']
    net = gain * lev * pos['notional'] - cost - funding
    new_b = balance + net
    trades.append({
        'symbol': sym, 'direction': pos['direction'],
        'confidence': pos['confidence'],
        'exit_reason': reason,
        'pnl_usd': net, 'pnl_pct': (net / pos['notional']) * 100,
        'bars_held': pos['hold_count'],
        'entry_bar': pos['entry_bar'],
        **{k: pos.get(k, np.nan) for k in [
            'f_vwma_dist', 'f_poc_dist', 'f_va_pos',
            'f_poc_rr', 'f_vwma_poc_agree',
        ]},
        '_b': new_b,
    })
    to_close.append(sym)


# ═══════════════════════════════════════════════════════════════════
# Variant Definitions — 22 variants
# ═══════════════════════════════════════════════════════════════════
def build_variants() -> list[VConfig]:
    v = []

    # ── CONTROL ──
    v.append(VConfig(
        name='A-Baseline',
        desc='Symmetric I-HybTight (prod, no adaptation)',
    ))

    # ═══ G1: VWMA innovations (R10 winners, refined) ═══
    v.append(VConfig(
        name='B-VWStW',
        desc='R10 winner: blend-wide stop max(base, vwma×0.50)',
        vwma_stop='wide', vwma_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='C-VWTgW',
        desc='R10 winner: blend-wide tgt max(base, vwma×0.80)',
        vwma_target='wide', vwma_target_mult=0.80,
    ))
    v.append(VConfig(
        name='D-VWSW+TW',
        desc='R10 combo: blend-wide stop + blend-wide target',
        vwma_stop='wide', vwma_stop_mult=0.50,
        vwma_target='wide', vwma_target_mult=0.80,
    ))

    # ═══ G2: POC-based stop ═══
    v.append(VConfig(
        name='E-POCSt50',
        desc='POC stop (replace): poc_dist × 0.50',
        poc_stop='replace', poc_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='F-POCStW',
        desc='POC stop (blend-wide): max(base, poc×0.50)',
        poc_stop='wide', poc_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='G-POCStT',
        desc='POC stop (blend-tight): min(base, poc×0.50)',
        poc_stop='tight', poc_stop_mult=0.50,
    ))

    # ═══ G3: POC-based target ═══
    v.append(VConfig(
        name='H-POCTg10',
        desc='POC target (replace): poc_dist × 1.0 (reach POC)',
        poc_target='replace', poc_target_mult=1.0,
    ))
    v.append(VConfig(
        name='I-POCTgW',
        desc='POC target (blend-wide): max(base, poc×0.80)',
        poc_target='wide', poc_target_mult=0.80,
    ))

    # ═══ G4: VA-based entry gate ═══
    v.append(VConfig(
        name='J-VAGate',
        desc='VA entry gate: only enter when outside Value Area',
        va_entry_gate=True,
    ))

    # ═══ G5: VA-based stop ═══
    v.append(VConfig(
        name='K-VAStR',
        desc='VA stop (replace): anchor at VA boundary + 0.2%',
        va_stop='replace', va_stop_margin=0.002,
    ))
    v.append(VConfig(
        name='L-VAStW',
        desc='VA stop (blend-wide): max(base, VA boundary+0.2%)',
        va_stop='wide', va_stop_margin=0.002,
    ))

    # ═══ G6: VA-based target ═══
    v.append(VConfig(
        name='M-VATgR',
        desc='VA target (replace): target opp VA boundary',
        va_target='replace',
    ))
    v.append(VConfig(
        name='N-VATgW',
        desc='VA target (blend-wide): max(base, opp VA boundary)',
        va_target='wide',
    ))

    # ═══ G7: VWMA + POC combos ═══
    v.append(VConfig(
        name='O-VW+POC_S',
        desc='VWMA wide stop + POC wide stop (double anchor)',
        vwma_stop='wide', vwma_stop_mult=0.50,
        poc_stop='wide', poc_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='P-VW+POC_T',
        desc='VWMA wide tgt + POC wide tgt (double magnet)',
        vwma_target='wide', vwma_target_mult=0.80,
        poc_target='wide', poc_target_mult=0.80,
    ))
    v.append(VConfig(
        name='Q-VW+POC_F',
        desc='Full: VW+POC wide stop + VW+POC wide target',
        vwma_stop='wide', vwma_stop_mult=0.50,
        vwma_target='wide', vwma_target_mult=0.80,
        poc_stop='wide', poc_stop_mult=0.50,
        poc_target='wide', poc_target_mult=0.80,
    ))

    # ═══ G8: VWMA + VA combos ═══
    v.append(VConfig(
        name='R-VWE+VA_S',
        desc='VWMA entry(>0.5%) + VA stop(wide)',
        vwma_entry_min=0.005,
        va_stop='wide', va_stop_margin=0.002,
    ))
    v.append(VConfig(
        name='S-VAG+VWT',
        desc='VA gate + VWMA wide target',
        va_entry_gate=True,
        vwma_target='wide', vwma_target_mult=0.80,
    ))

    # ═══ G9: R:R sizing ═══
    v.append(VConfig(
        name='T-RRSize',
        desc='POC R:R sizing: scale pos by implied R:R',
        rr_size=True,
    ))
    v.append(VConfig(
        name='U-RR+VWS',
        desc='R:R sizing + VWMA wide stop',
        rr_size=True,
        vwma_stop='wide', vwma_stop_mult=0.50,
    ))

    # ═══ G10: Kitchen sink ═══
    v.append(VConfig(
        name='V-Kitchen',
        desc='Best-of: VW wide stop + POC wide tgt + VA wide stop',
        vwma_stop='wide', vwma_stop_mult=0.50,
        poc_target='wide', poc_target_mult=0.80,
        va_stop='wide', va_stop_margin=0.002,
    ))

    return v


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    commit = _get_git_commit()
    t0 = time.time()
    w = 110

    print("=" * w)
    print("  ALPHA SEARCH R11: VWMA + VPVR/POC — R:R Maximization")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baseline: v12.4 I-HybTight (symmetric T150_S38_H25_PBT)")
    print(f"  VWMA: 100-bar on 15m | VPVR: {VPVR_LOOKBACK}-bar lookback, {VPVR_NBINS} bins, {VPVR_VA_PCT:.0%} VA")
    print(f"  Data: {DATA_DAYS}d ({DATA_DAYS//365}yr), 50 symbols")
    print("=" * w)

    # ── Load data ──
    print(f"\n📊 Loading data...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC data not found"); sys.exit(1)
    btc_close = btc_data['close']

    btc_4h_close = load_4h('BTCUSDT')
    if btc_4h_close is None:
        print("ERROR: BTC 4h data not found"); sys.exit(1)

    all_data, all_4h = {}, {}
    for sym in SYMBOLS_50:
        d = load_15m(sym)
        c4h = load_4h(sym)
        if d and c4h is not None:
            all_data[sym] = d
            all_4h[sym] = c4h
        else:
            print(f"  SKIP {sym}")
    print(f"  Loaded {len(all_data)} symbols")

    # ── Scan entries ──
    print(f"\n🔍 Scanning entries (conf≥{MIN_CONFIDENCE})...")
    sym_dirs, sym_confs, sym_moves = {}, {}, {}
    sym_close_d, sym_high_d, sym_low_d, sym_atr_d = {}, {}, {}, {}
    sym_volume_d: dict[str, np.ndarray] = {}

    for sym, d in all_data.items():
        n = min(len(d['close']), len(btc_close))
        cn = d['close'][:n]; hn = d['high'][:n]; ln = d['low'][:n]
        vn = d['volume'][:n]; tn = d['taker_buy'][:n]; bn = btc_close[:n]

        dirs, confs, moves = scan_entries(
            cn, hn, ln, vn, tn, bn,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE,
            RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
        )
        dirs, confs, moves = apply_mtf_filter(dirs, confs, moves, all_4h[sym], MTF_SMA_PERIOD)

        sym_dirs[sym] = dirs; sym_confs[sym] = confs; sym_moves[sym] = moves
        sym_close_d[sym] = cn; sym_high_d[sym] = hn; sym_low_d[sym] = ln
        sym_atr_d[sym] = compute_atr_pct(hn, ln, cn, ATR_PERIOD)
        sym_volume_d[sym] = vn

    # Left-pad all arrays to same length
    max_bars = max(len(sym_close_d[s]) for s in sym_close_d)
    all_dicts = [
        (sym_close_d, 'first'), (sym_high_d, 'first'), (sym_low_d, 'first'),
        (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
        (sym_atr_d, 'nan'), (sym_volume_d, 'zero'),
    ]
    for sym in list(sym_close_d.keys()):
        for d_ref, fill in all_dicts:
            arr = d_ref.get(sym)
            if arr is None:
                continue
            if len(arr) < max_bars:
                pad_n = max_bars - len(arr)
                if fill == 'first':
                    fv = arr[0]
                elif fill == 'nan':
                    fv = np.nan
                else:
                    fv = 0.0
                d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

    raw_signals = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    print(f"  Left-padded to {max_bars} bars ({max_bars//96} days)")
    print(f"  Raw signals: {raw_signals:,}")

    # ── Compute VWMA(100) per symbol ──
    print(f"\n📈 Computing VWMA(100) per symbol...")
    sym_vwma: dict[str, np.ndarray] = {}
    vwma_period = 100
    for sym in sym_close_d:
        sym_vwma[sym] = compute_vwma(sym_close_d[sym], sym_volume_d[sym], vwma_period)
    vwma_valid = sum(1 for s in sym_vwma if np.any(~np.isnan(sym_vwma[s])))
    print(f"  VWMA computed for {vwma_valid}/{len(sym_vwma)} symbols")

    # ── Compute VPVR (POC + VA) per symbol ──
    print(f"\n📊 Computing VPVR ({VPVR_LOOKBACK}-bar lookback, {VPVR_NBINS} bins) per symbol...")
    sym_poc: dict[str, np.ndarray] = {}
    sym_va_high: dict[str, np.ndarray] = {}
    sym_va_low: dict[str, np.ndarray] = {}
    vpvr_t0 = time.time()

    for idx, sym in enumerate(sorted(sym_close_d.keys())):
        poc_arr, vah_arr, val_arr = compute_vpvr(
            sym_close_d[sym], sym_high_d[sym], sym_low_d[sym],
            sym_volume_d[sym], VPVR_LOOKBACK, VPVR_NBINS, VPVR_VA_PCT,
        )
        sym_poc[sym] = poc_arr
        sym_va_high[sym] = vah_arr
        sym_va_low[sym] = val_arr
        if (idx + 1) % 10 == 0:
            elapsed_vpvr = time.time() - vpvr_t0
            eta = elapsed_vpvr / (idx + 1) * (len(sym_close_d) - idx - 1)
            print(f"  [{idx+1}/{len(sym_close_d)}] {sym} done ({elapsed_vpvr:.1f}s, ETA {eta:.0f}s)")

    vpvr_elapsed = time.time() - vpvr_t0
    poc_valid = sum(1 for s in sym_poc if np.any(~np.isnan(sym_poc[s])))
    print(f"  VPVR computed for {poc_valid}/{len(sym_poc)} symbols in {vpvr_elapsed:.1f}s")

    # Check POC coverage: what % of entry bars have valid POC
    entry_bars = set()
    for sym in sym_dirs:
        for i in range(len(sym_dirs[sym])):
            if sym_dirs[sym][i] != 0:
                entry_bars.add((sym, i))
    poc_covered = sum(1 for sym, i in entry_bars
                      if sym in sym_poc and i < len(sym_poc[sym]) and not np.isnan(sym_poc[sym][i]))
    print(f"  POC coverage at entries: {poc_covered}/{len(entry_bars)} ({poc_covered/max(1,len(entry_bars))*100:.1f}%)")

    # ── Compute BTC dual-confirm regime ──
    print(f"\n🌐 Computing regime...")
    regime = compute_dual_confirm_regime(btc_4h_close, max_bars)

    # ═══════════════════ PART A: FACTOR ANALYSIS ═══════════════════
    print(f"\n🔬 Running baseline simulation for factor tagging...")
    tagged_trades = get_tagged_trades(
        sym_close_d, sym_high_d, sym_low_d,
        sym_dirs, sym_confs, sym_moves,
        sym_atr_d, sym_vwma,
        sym_poc, sym_va_high, sym_va_low,
    )
    print(f"  Tagged {len(tagged_trades)} trades with 5 factor values")

    factor_analysis(tagged_trades)

    # ═══════════════════ PART B: VARIANT TEST ═══════════════════
    variants = build_variants()
    print(f"\n{'═' * w}")
    print(f"  PART B: VWMA + VPVR/POC A/B TEST — {len(variants)} VARIANTS")
    print(f"{'═' * w}")

    results = {}
    for vi, cfg in enumerate(variants):
        t_v = time.time()
        print(f"\n🔬 [{vi+1}/{len(variants)}] {cfg.name}: {cfg.desc}")

        res = simulate(
            sym_close_d, sym_high_d, sym_low_d,
            sym_dirs, sym_confs, sym_moves,
            cfg, regime, sym_atr_d, sym_vwma,
            sym_poc, sym_va_high, sym_va_low,
        )

        elapsed_v = time.time() - t_v
        res['elapsed'] = round(elapsed_v, 1)
        results[cfg.name] = res

        print(f"  Ret: {res['total_return_pct']:,.1f}%  DD: {res['max_dd_pct']:.1f}%  "
              f"Sharpe: {res['sharpe']:.2f}  Calmar: {res['calmar']:.1f}  "
              f"WR: {res['win_rate']:.1f}%  PF: {res['profit_factor']:.2f}  R:R: {res['rr']:.2f}  "
              f"#Tr: {res['n_trades']}")

    # ── Summary table ──
    elapsed_total = time.time() - t0
    print(f"\n{'=' * w}")
    print("  RESULTS — Alpha Search R11: VWMA + VPVR/POC R:R Maximization")
    print(f"{'=' * w}")
    print(f"  {'Variant':<12} {'Desc':<52} {'Ret%':>8} {'DD%':>5} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WR':>5} {'PF':>5} {'R:R':>5} {'#Tr':>6}")
    print(f"  {'─'*12} {'─'*52} {'─'*8} {'─'*5} {'─'*7} {'─'*7} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*6}")

    base = results.get('A-Baseline', {})
    base_dd = base.get('max_dd_pct', 100)
    base_sharpe = base.get('sharpe', 0)
    base_ret = base.get('total_return_pct', 0)
    base_rr = base.get('rr', 0)

    for cfg in variants:
        r = results[cfg.name]
        marks = ''
        if r['max_dd_pct'] < base_dd:
            marks += '★'
        if r['sharpe'] > base_sharpe:
            marks += '◆'
        if r['total_return_pct'] > base_ret:
            marks += '▲'
        if r['rr'] > base_rr:
            marks += '⬆'
        marks = marks.ljust(4)

        print(f" {marks}{cfg.name:<12} {cfg.desc:<52} "
              f"{r['total_return_pct']:>7,.1f}% {r['max_dd_pct']:>4.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.1f} "
              f"{r['win_rate']:>4.1f}% {r['profit_factor']:>4.2f} {r['rr']:>4.2f} {r['n_trades']:>6d}")

    print(f"\n  ★ = better DD  ◆ = better Sharpe  ▲ = better Return  ⬆ = better R:R")
    print(f"  Total time: {elapsed_total:.1f}s")

    # ── Winners ──
    best_sharpe = max(results.items(), key=lambda x: x[1].get('sharpe', 0))
    best_dd = min(results.items(), key=lambda x: x[1].get('max_dd_pct', 100))
    best_calmar = max(results.items(), key=lambda x: x[1].get('calmar', 0))
    best_rr = max(results.items(), key=lambda x: x[1].get('rr', 0))
    best_ret = max(results.items(), key=lambda x: x[1].get('total_return_pct', 0))

    print(f"\n  🏆 Best Sharpe:  {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    print(f"  🏆 Best DD:      {best_dd[0]} ({best_dd[1]['max_dd_pct']:.1f}%)")
    print(f"  🏆 Best Calmar:  {best_calmar[0]} ({best_calmar[1]['calmar']:.1f})")
    print(f"  🏆 Best R:R:     {best_rr[0]} ({best_rr[1]['rr']:.2f})")
    print(f"  🏆 Best Return:  {best_ret[0]} ({best_ret[1]['total_return_pct']:,.1f}%)")

    # ── Save JSON ──
    output = {
        'commit': commit,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'data_days': DATA_DAYS,
        'n_symbols': len(all_data),
        'vwma_period': vwma_period,
        'vpvr_lookback': VPVR_LOOKBACK,
        'vpvr_nbins': VPVR_NBINS,
        'vpvr_va_pct': VPVR_VA_PCT,
        'vpvr_compute_time': round(vpvr_elapsed, 1),
        'factor_summary': {
            'n_tagged_trades': len(tagged_trades),
            'poc_coverage_pct': round(poc_covered / max(1, len(entry_bars)) * 100, 1),
        },
        'variants': {k: {kk: vv for kk, vv in v.items() if kk != '_b'}
                     for k, v in results.items()},
        'best': {
            'sharpe': best_sharpe[0], 'dd': best_dd[0],
            'calmar': best_calmar[0], 'rr': best_rr[0], 'ret': best_ret[0],
        },
        'elapsed': round(elapsed_total, 1),
    }

    out_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r11_results.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {out_path}")


if __name__ == '__main__':
    main()
