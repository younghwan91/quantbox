#!/usr/bin/env python3
"""
Alpha Search R10: Multi-Factor Adaptive + VWMA S/R Innovation
=============================================================
Extends the research by identifying INDEPENDENT market variables that
predict trade quality, then using them for adaptive entry/stop/target.

Key innovation: VWMA(100) as dynamic support/resistance reference for
stop placement, entry validation, and target sizing.

Part A: FACTOR INDEPENDENCE ANALYSIS
  Tag each baseline trade with 5 factors at entry time:
  F1: VWMA(100) directional distance — price dislocation from volume-weighted fair value
  F2: BTC dual-confirm regime — macro trend state (+1/0/-1)
  F3: BTC realized volatility — ATR ratio vs rolling average
  F4: Asset ATR percentile — current vol vs recent history
  F5: Market breadth — % of universe above SMA(200) on 15m
  → Tercile analysis for each, pairwise correlations.

Part B: VWMA VARIANTS (17 total)
  G1: VWMA entry gate (only enter when price is on correct side of VWMA)
  G2: VWMA dynamic stop (size stop by VWMA distance instead of move_size)
  G3: VWMA dynamic target (target = revert to VWMA)
  G4: VWMA full combo
  G5: Dual-confirm + VWMA
  G6: Multi-factor sizing (BTC vol, breadth)

Data: 6yr (2190d), 50 symbols, exact production logic.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r10.py
"""

import sys
import time
import json
import subprocess
import numba
import numpy as np
from dataclasses import dataclass
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


# ═══════════════════════════════════════════════════════════════════
# Variant Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class VConfig:
    """Variant configuration for multi-factor + VWMA testing.

    Base exits are production I-HybTight. VWMA and multi-factor
    features are layered on top.
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

    # ── Regime adaptation (dual-confirm from R9) ──
    regime_adapt: bool = False
    al_stop: float = 0.45              # aligned stop mult
    mis_stop: float = 0.30             # misaligned stop mult

    # ── BTC vol adaptive stop ──
    vol_adapt: bool = False
    vol_hi_stop: float = 0.45          # stop in high BTC vol
    vol_lo_stop: float = 0.30          # stop in low BTC vol

    # ── Market breadth sizing ──
    breadth_size: bool = False

    # ── PBT (constant across all variants) ──
    partial_exit: bool = True
    partial_frac: float = 0.5
    trail_atr: float = 1.0
    be_after_tp1: bool = True
    runner_mh_mult: float = 3.0
    stop_decay: float = 0.0

    # ── Portfolio (constant) ──
    max_slots: int = 2
    dir_div: bool = True
    steep_cap: bool = True
    vol_adj: bool = True
    soft_eq: bool = True


# ═══════════════════════════════════════════════════════════════════
# Data loading
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
# Entry scanner (production-identical)
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
# VWMA computation (core innovation)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def compute_vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume-Weighted Moving Average.

    VWMA[i] = sum(close[j]*volume[j] for j in [i-period+1..i]) / sum(volume[j])
    Returns NaN before warmup or when total volume in window is zero.
    """
    n = len(close)
    vwma = np.full(n, np.nan)
    pv_sum = 0.0   # running sum of price × volume
    v_sum = 0.0    # running sum of volume
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
# BTC dual-confirm regime (best signal from R9)
# ═══════════════════════════════════════════════════════════════════
def compute_dual_confirm_regime(btc_4h_close: np.ndarray, n_15m: int) -> np.ndarray:
    """Dual-confirm regime: bull only when BTC > BOTH SMA(50) and SMA(200) on 4h.

    +1 = strong bull, -1 = strong bear, 0 = choppy.
    """
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
# Factor precomputation
# ═══════════════════════════════════════════════════════════════════
def compute_btc_vol_ratio(btc_close: np.ndarray, btc_high: np.ndarray,
                          btc_low: np.ndarray) -> np.ndarray:
    """BTC realized volatility ratio: ATR(96) / SMA(ATR, 2880).

    >1 = above-average vol, <1 = below-average vol.
    96 bars = 1 day of 15m data. SMA(2880) = 30-day average.
    """
    atr = compute_atr_pct(btc_high, btc_low, btc_close, 96)
    n = len(atr)

    # Forward-fill NaN for SMA computation
    atr_safe = atr.copy()
    first_valid = -1
    for i in range(n):
        if not np.isnan(atr_safe[i]):
            first_valid = i
            break
    if first_valid < 0:
        return np.full(n, np.nan)
    for i in range(first_valid):
        atr_safe[i] = atr_safe[first_valid]
    for i in range(first_valid + 1, n):
        if np.isnan(atr_safe[i]):
            atr_safe[i] = atr_safe[i - 1]

    atr_sma = _compute_sma_inline(atr_safe, 2880)

    ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_sma[i]) and atr_sma[i] > 1e-10 and not np.isnan(atr[i]):
            ratio[i] = atr[i] / atr_sma[i]
    return ratio


def compute_market_breadth(sym_close: dict[str, np.ndarray],
                           sma_period: int = 200) -> np.ndarray:
    """Fraction of symbols with close > SMA(sma_period) on 15m.

    Returns array of length n_bars, values in [0, 1].
    """
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    # Compute SMA for each symbol
    sma_arrays = []
    close_arrays = []
    for sym in symbols:
        c = sym_close[sym][:n_bars]
        sma = _compute_sma_inline(c, sma_period)
        close_arrays.append(c)
        sma_arrays.append(sma)

    # Stack and vectorize
    all_close = np.array(close_arrays)   # (n_sym, n_bars)
    all_sma = np.array(sma_arrays)       # (n_sym, n_bars)
    valid = ~np.isnan(all_sma)
    above = (all_close > all_sma) & valid
    breadth = above.sum(axis=0).astype(np.float64) / len(symbols)
    return breadth


def compute_asset_atr_ratio(atr: np.ndarray) -> np.ndarray:
    """Asset ATR ratio: current ATR / SMA(ATR, 200). >1 = above avg vol."""
    n = len(atr)
    atr_safe = atr.copy()
    first_valid = -1
    for i in range(n):
        if not np.isnan(atr_safe[i]):
            first_valid = i
            break
    if first_valid < 0:
        return np.full(n, np.nan)
    for i in range(first_valid):
        atr_safe[i] = atr_safe[first_valid]
    for i in range(first_valid + 1, n):
        if np.isnan(atr_safe[i]):
            atr_safe[i] = atr_safe[i - 1]

    atr_sma = _compute_sma_inline(atr_safe, 200)
    ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_sma[i]) and atr_sma[i] > 1e-10 and not np.isnan(atr[i]):
            ratio[i] = atr[i] / atr_sma[i]
    return ratio


# ═══════════════════════════════════════════════════════════════════
# Part A: Factor Independence Analysis
# ═══════════════════════════════════════════════════════════════════
def factor_analysis(trades: list[dict], n_bars: int):
    """Analyze 5 independent factors: tercile split + pairwise correlations.

    Each trade must have been tagged with factor values at entry time.
    """
    w = 110
    print(f"\n{'═' * w}")
    print(f"  PART A: MULTI-FACTOR INDEPENDENCE ANALYSIS — {len(trades)} trades")
    print(f"{'═' * w}")

    factor_names = [
        ('f_vwma_dist', 'VWMA(100) Dist'),
        ('f_dual_regime', 'Dual-Confirm Regime'),
        ('f_btc_vol', 'BTC Vol Ratio'),
        ('f_asset_atr', 'Asset ATR Ratio'),
        ('f_breadth', 'Market Breadth'),
    ]

    # Compute per-factor stats
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

        # Tercile split
        if fkey == 'f_dual_regime':
            # Categorical: -1, 0, +1
            groups = [
                ('Bear (-1)', [t for t in trades if t.get(fkey) == -1]),
                ('Choppy (0)', [t for t in trades if t.get(fkey) == 0]),
                ('Bull (+1)', [t for t in trades if t.get(fkey) == 1]),
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
        if len(ev_values) == 3 and all(v != 0 for v in ev_values):
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
    n_factors = len(fkeys)

    # Build matrix
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
# Simulation engine (modified for VWMA + multi-factor support)
# ═══════════════════════════════════════════════════════════════════
def simulate(
    sym_close, sym_high, sym_low,
    sym_dirs, sym_confs, sym_moves,
    cfg: VConfig,
    regime: np.ndarray,
    sym_atr=None,
    sym_vwma=None,
    btc_vol_ratio=None,
    breadth=None,
) -> dict:
    """Run portfolio sim with VWMA + multi-factor adaptive parameters.

    New features vs adaptive_direction_ab.py:
    - VWMA entry gate: skip entries where price is on wrong side of VWMA
    - VWMA stop: size stop by VWMA distance instead of move_size
    - VWMA target: target = fraction of VWMA distance
    - BTC vol adaptive stop
    - Market breadth sizing
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
        r = regime[i]

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            ep = pos['entry_price']
            lev = pos['leverage']
            d = pos['direction']
            gain = (c - ep) / ep if d == 1 else (ep - c) / ep

            # ═══ STOP COMPUTATION ═══
            # Base stop from move_size
            time_factor = 1.0 - cfg.stop_decay * (pos['hold_count'] / cfg.max_hold)
            if time_factor < 0.1:
                time_factor = 0.1

            # Determine stop_mult (regime or vol-adaptive)
            s_mult = cfg.stop_mult
            if cfg.regime_adapt:
                entry_regime = pos.get('entry_regime', 0)
                aligned = (d == entry_regime) if entry_regime != 0 else True
                s_mult = cfg.al_stop if aligned else cfg.mis_stop
            elif cfg.vol_adapt and btc_vol_ratio is not None:
                vr = btc_vol_ratio[min(i, len(btc_vol_ratio) - 1)]
                if not np.isnan(vr):
                    if vr > 1.3:
                        s_mult = cfg.vol_hi_stop
                    elif vr < 0.7:
                        s_mult = cfg.vol_lo_stop

            base_stop = pos['move_size'] * s_mult * time_factor

            # VWMA stop
            vw_dist = pos.get('vwma_dist', 0.0)
            if cfg.vwma_stop == 'off':
                effective_stop = base_stop
            else:
                vw_stop = max(vw_dist * cfg.vwma_stop_mult, 0.003) * time_factor
                if cfg.vwma_stop == 'replace':
                    effective_stop = vw_stop
                elif cfg.vwma_stop == 'tight':
                    effective_stop = min(base_stop, vw_stop)
                elif cfg.vwma_stop == 'wide':
                    effective_stop = max(base_stop, vw_stop)
                else:
                    effective_stop = base_stop

            # ═══ TARGET COMPUTATION ═══
            base_target = pos['move_size'] * cfg.target_retrace
            if cfg.vwma_target == 'off':
                target = base_target
            else:
                vw_tgt = max(vw_dist * cfg.vwma_target_mult, 0.003)
                if cfg.vwma_target == 'replace':
                    target = vw_tgt
                elif cfg.vwma_target == 'wide':
                    target = max(base_target, vw_tgt)
                else:
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

                # Regime gating (for dual-confirm variants)
                if cfg.regime_adapt:
                    is_aligned = (direction == r) if r != 0 else True
                else:
                    is_aligned = True

                # VWMA entry gate
                if cfg.vwma_entry_min > 0.0 and sym_vwma is not None:
                    vwma_arr = sym_vwma.get(sym)
                    if vwma_arr is not None and i < len(vwma_arr) and not np.isnan(vwma_arr[i]):
                        c_entry = sym_close[sym][i]
                        vwma_val = vwma_arr[i]
                        if direction == 1:  # long: want VWMA above entry (room to revert up)
                            dir_vwma_dist = (vwma_val - c_entry) / c_entry
                        else:  # short: want VWMA below entry (room to revert down)
                            dir_vwma_dist = (c_entry - vwma_val) / c_entry
                        if dir_vwma_dist < cfg.vwma_entry_min:
                            continue  # skip: insufficient VWMA distance
                    else:
                        continue  # no VWMA data → skip conservatively

                # Direction diversity
                if cfg.dir_div:
                    same = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same >= 2:
                        continue

                candidates.append((sym, direction, confidence, move_size, is_aligned))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size, is_aligned in candidates:
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

                # Breadth sizing
                if cfg.breadth_size and breadth is not None:
                    b = breadth[min(i, len(breadth) - 1)]
                    if not np.isnan(b):
                        if b < 0.35:       # panic → bigger position
                            per_slot *= 1.3
                        elif b > 0.65:     # euphoria → smaller position
                            per_slot *= 0.7
                    per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost

                # Compute VWMA distance at entry
                vwma_dist_entry = 0.0
                if sym_vwma is not None:
                    vwma_arr = sym_vwma.get(sym)
                    if vwma_arr is not None and i < len(vwma_arr) and not np.isnan(vwma_arr[i]):
                        c_entry = sym_close[sym][i]
                        vwma_val = vwma_arr[i]
                        if direction == 1:
                            vwma_dist_entry = (vwma_val - c_entry) / c_entry
                        else:
                            vwma_dist_entry = (c_entry - vwma_val) / c_entry
                        vwma_dist_entry = max(vwma_dist_entry, 0.001)

                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': sym_close[sym][i],
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': EFFECTIVE_LEV,
                    'entry_regime': int(r),
                    'vwma_dist': vwma_dist_entry,
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
# Baseline trade extraction (for factor tagging)
# ═══════════════════════════════════════════════════════════════════
def get_tagged_trades(
    sym_close, sym_high, sym_low,
    sym_dirs, sym_confs, sym_moves,
    regime, sym_atr, sym_vwma,
    btc_vol_ratio, breadth, asset_atr_ratios,
) -> list[dict]:
    """Run baseline simulation and tag each trade with 5 factor values at entry."""
    cfg = VConfig(name='A-Baseline')
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)
    n_bars = min(n_bars, len(regime))

    balance = INITIAL_CAPITAL
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance
    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for i in range(1, n_bars):
        to_close: list[str] = []
        r = regime[i]

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
                        'f_vwma_dist': pos.get('f_vwma_dist', np.nan),
                        'f_dual_regime': pos.get('f_dual_regime', np.nan),
                        'f_btc_vol': pos.get('f_btc_vol', np.nan),
                        'f_asset_atr': pos.get('f_asset_atr', np.nan),
                        'f_breadth': pos.get('f_breadth', np.nan),
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
                        'f_vwma_dist': pos.get('f_vwma_dist', np.nan),
                        'f_dual_regime': pos.get('f_dual_regime', np.nan),
                        'f_btc_vol': pos.get('f_btc_vol', np.nan),
                        'f_asset_atr': pos.get('f_asset_atr', np.nan),
                        'f_breadth': pos.get('f_breadth', np.nan),
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

                # Tag factors at entry
                f_vwma = np.nan
                if sym_vwma is not None:
                    va = sym_vwma.get(sym)
                    if va is not None and i < len(va) and not np.isnan(va[i]):
                        c_e = sym_close[sym][i]
                        vw = va[i]
                        if direction == 1:
                            f_vwma = (vw - c_e) / c_e
                        else:
                            f_vwma = (c_e - vw) / c_e

                f_regime = float(r)

                f_btc_vol = np.nan
                if btc_vol_ratio is not None and i < len(btc_vol_ratio):
                    f_btc_vol = btc_vol_ratio[i]

                f_asset_atr = np.nan
                if sym in asset_atr_ratios:
                    aar = asset_atr_ratios[sym]
                    if i < len(aar) and not np.isnan(aar[i]):
                        f_asset_atr = aar[i]

                f_breadth = np.nan
                if breadth is not None and i < len(breadth):
                    f_breadth = breadth[i]

                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': sym_close[sym][i],
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': EFFECTIVE_LEV,
                    'entry_regime': int(r),
                    'f_vwma_dist': f_vwma,
                    'f_dual_regime': f_regime,
                    'f_btc_vol': f_btc_vol,
                    'f_asset_atr': f_asset_atr,
                    'f_breadth': f_breadth,
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
        'f_vwma_dist': pos.get('f_vwma_dist', np.nan),
        'f_dual_regime': pos.get('f_dual_regime', np.nan),
        'f_btc_vol': pos.get('f_btc_vol', np.nan),
        'f_asset_atr': pos.get('f_asset_atr', np.nan),
        'f_breadth': pos.get('f_breadth', np.nan),
        '_b': new_b,
    })
    to_close.append(sym)


# ═══════════════════════════════════════════════════════════════════
# Variant Definitions
# ═══════════════════════════════════════════════════════════════════
def build_variants() -> list[VConfig]:
    v = []

    # ── CONTROL ──
    v.append(VConfig(
        name='A-Baseline',
        desc='Symmetric I-HybTight (prod, no adaptation)',
    ))

    # ═══ G1: VWMA Entry Gate ═══
    # Only enter when price is on the correct side of VWMA (room for reversion)
    v.append(VConfig(
        name='B-VWEntry0',
        desc='VWMA entry gate: directional dist > 0%',
        vwma_entry_min=0.001,
    ))
    v.append(VConfig(
        name='C-VWE05',
        desc='VWMA entry gate: directional dist > 0.5%',
        vwma_entry_min=0.005,
    ))
    v.append(VConfig(
        name='D-VWE10',
        desc='VWMA entry gate: directional dist > 1.0%',
        vwma_entry_min=0.01,
    ))

    # ═══ G2: VWMA Dynamic Stop ═══
    # Size stop by VWMA distance instead of move_size
    v.append(VConfig(
        name='E-VWSt50',
        desc='VWMA stop (replace): vwma_dist × 0.50',
        vwma_stop='replace', vwma_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='F-VWSt75',
        desc='VWMA stop (replace): vwma_dist × 0.75',
        vwma_stop='replace', vwma_stop_mult=0.75,
    ))
    v.append(VConfig(
        name='G-VWStT',
        desc='VWMA stop (blend-tight): min(normal, vwma×0.50)',
        vwma_stop='tight', vwma_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='H-VWStW',
        desc='VWMA stop (blend-wide): max(normal, vwma×0.50)',
        vwma_stop='wide', vwma_stop_mult=0.50,
    ))

    # ═══ G3: VWMA Dynamic Target ═══
    # Target = revert to VWMA (or fraction thereof)
    v.append(VConfig(
        name='I-VWTg10',
        desc='VWMA target (replace): vwma_dist × 1.0 (reach VWMA)',
        vwma_target='replace', vwma_target_mult=1.0,
    ))
    v.append(VConfig(
        name='J-VWTg08',
        desc='VWMA target (replace): vwma_dist × 0.8',
        vwma_target='replace', vwma_target_mult=0.8,
    ))
    v.append(VConfig(
        name='K-VWTgW',
        desc='VWMA target (blend-wide): max(normal, vwma×0.80)',
        vwma_target='wide', vwma_target_mult=0.80,
    ))

    # ═══ G4: VWMA Full Combo ═══
    v.append(VConfig(
        name='L-VWFull',
        desc='VWMA: entry(>0) + stop(×0.50) + target(×1.0)',
        vwma_entry_min=0.001,
        vwma_stop='replace', vwma_stop_mult=0.50,
        vwma_target='replace', vwma_target_mult=1.0,
    ))

    # ═══ G5: Dual-Confirm + VWMA ═══
    v.append(VConfig(
        name='M-DuVWS',
        desc='Dual-confirm stops(.45/.30) + VWMA stop blend-tight',
        regime_adapt=True, al_stop=0.45, mis_stop=0.30,
        vwma_stop='tight', vwma_stop_mult=0.50,
    ))
    v.append(VConfig(
        name='N-DuVWE',
        desc='Dual-confirm + VWMA entry gate(>0.5%)',
        regime_adapt=True, al_stop=0.45, mis_stop=0.30,
        vwma_entry_min=0.005,
    ))
    v.append(VConfig(
        name='O-DuVWF',
        desc='Dual-confirm + VWMA entry + stop + target',
        regime_adapt=True, al_stop=0.45, mis_stop=0.30,
        vwma_entry_min=0.001,
        vwma_stop='tight', vwma_stop_mult=0.50,
        vwma_target='wide', vwma_target_mult=0.80,
    ))

    # ═══ G6: Multi-Factor ═══
    v.append(VConfig(
        name='P-VolSt',
        desc='BTC vol adaptive stop: high→0.45, low→0.30',
        vol_adapt=True, vol_hi_stop=0.45, vol_lo_stop=0.30,
    ))
    v.append(VConfig(
        name='Q-BrdSz',
        desc='Market breadth sizing: panic→1.3×, euphoria→0.7×',
        breadth_size=True,
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
    print("  ALPHA SEARCH R10: Multi-Factor Adaptive + VWMA S/R Innovation")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baseline: v12.4 I-HybTight (symmetric T150_S38_H25_PBT)")
    print(f"  VWMA: 100-bar volume-weighted moving average on 15m")
    print(f"  Data: {DATA_DAYS}d ({DATA_DAYS//365}yr), 50 symbols")
    print("=" * w)

    # ── Load data ──
    print(f"\n📊 Loading data...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC data not found"); sys.exit(1)
    btc_close = btc_data['close']
    btc_high = btc_data['high']
    btc_low = btc_data['low']

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

    # Left-pad all arrays to same length
    max_bars = max(len(sym_close_d[s]) for s in sym_close_d)
    all_dicts = [
        (sym_close_d, 'first'), (sym_high_d, 'first'), (sym_low_d, 'first'),
        (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
        (sym_atr_d, 'nan'),
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
        vol_arr = all_data[sym]['volume']
        # Left-pad volume too
        if len(vol_arr) < max_bars:
            pad_n = max_bars - len(vol_arr)
            vol_arr = np.concatenate([np.full(pad_n, 0.0), vol_arr])
        sym_vwma[sym] = compute_vwma(sym_close_d[sym], vol_arr, vwma_period)

    # Check VWMA coverage
    vwma_valid = sum(1 for s in sym_vwma if np.any(~np.isnan(sym_vwma[s])))
    print(f"  VWMA computed for {vwma_valid}/{len(sym_vwma)} symbols")

    # ── Compute BTC dual-confirm regime ──
    print(f"\n🌐 Computing regime & factor arrays...")
    regime = compute_dual_confirm_regime(btc_4h_close, max_bars)
    dc_bull = np.sum(regime > 0) / max_bars * 100
    dc_bear = np.sum(regime < 0) / max_bars * 100
    dc_chop = np.sum(regime == 0) / max_bars * 100
    print(f"  Dual-confirm: {dc_bull:.1f}% bull / {dc_bear:.1f}% bear / {dc_chop:.1f}% choppy")

    # ── Compute BTC vol ratio ──
    # Left-pad BTC arrays
    btc_h = btc_high.copy()
    btc_l = btc_low.copy()
    btc_c = btc_close.copy()
    if len(btc_h) < max_bars:
        pn = max_bars - len(btc_h)
        btc_h = np.concatenate([np.full(pn, btc_h[0]), btc_h])
        btc_l = np.concatenate([np.full(pn, btc_l[0]), btc_l])
        btc_c = np.concatenate([np.full(pn, btc_c[0]), btc_c])

    btc_vol_ratio_arr = compute_btc_vol_ratio(btc_c, btc_h, btc_l)
    bvr_valid = np.sum(~np.isnan(btc_vol_ratio_arr))
    print(f"  BTC vol ratio: {bvr_valid}/{max_bars} valid bars, "
          f"mean={np.nanmean(btc_vol_ratio_arr):.3f}")

    # ── Compute market breadth ──
    breadth_arr = compute_market_breadth(sym_close_d, 200)
    if len(breadth_arr) < max_bars:
        breadth_arr = np.concatenate([np.full(max_bars - len(breadth_arr), np.nan), breadth_arr])
    br_valid = np.sum(~np.isnan(breadth_arr))
    print(f"  Market breadth: {br_valid}/{max_bars} valid, mean={np.nanmean(breadth_arr):.3f}")

    # ── Compute per-asset ATR ratio ──
    asset_atr_ratios: dict[str, np.ndarray] = {}
    for sym in sym_atr_d:
        asset_atr_ratios[sym] = compute_asset_atr_ratio(sym_atr_d[sym])

    # ═══════════════════ PART A: FACTOR ANALYSIS ═══════════════════
    print(f"\n🔬 Running baseline simulation for factor tagging...")
    tagged_trades = get_tagged_trades(
        sym_close_d, sym_high_d, sym_low_d,
        sym_dirs, sym_confs, sym_moves,
        regime, sym_atr_d, sym_vwma,
        btc_vol_ratio_arr, breadth_arr, asset_atr_ratios,
    )
    print(f"  Tagged {len(tagged_trades)} trades with 5 factor values")

    factor_analysis(tagged_trades, max_bars)

    # ═══════════════════ PART B: VARIANT TEST ═══════════════════
    variants = build_variants()
    print(f"\n{'═' * w}")
    print(f"  PART B: VWMA + MULTI-FACTOR A/B TEST — {len(variants)} VARIANTS")
    print(f"{'═' * w}")

    results = {}
    for vi, cfg in enumerate(variants):
        t_v = time.time()
        print(f"\n🔬 [{vi+1}/{len(variants)}] {cfg.name}: {cfg.desc}")

        res = simulate(
            sym_close_d, sym_high_d, sym_low_d,
            sym_dirs, sym_confs, sym_moves,
            cfg, regime, sym_atr_d, sym_vwma,
            btc_vol_ratio_arr, breadth_arr,
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
    print("  RESULTS — Alpha Search R10: Multi-Factor + VWMA")
    print(f"{'=' * w}")
    print(f"  {'Variant':<12} {'Desc':<52} {'Ret%':>8} {'DD%':>5} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WR':>5} {'PF':>5} {'R:R':>5} {'#Tr':>6}")
    print(f"  {'─'*12} {'─'*52} {'─'*8} {'─'*5} {'─'*7} {'─'*7} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*6}")

    base = results.get('A-Baseline', {})
    base_dd = base.get('max_dd_pct', 100)
    base_sharpe = base.get('sharpe', 0)
    base_ret = base.get('total_return_pct', 0)

    for cfg in variants:
        r = results[cfg.name]
        marks = ''
        if r['max_dd_pct'] < base_dd:
            marks += '★'
        if r['sharpe'] > base_sharpe:
            marks += '◆'
        if r['total_return_pct'] > base_ret:
            marks += '▲'
        marks = marks.ljust(3)

        print(f" {marks}{cfg.name:<12} {cfg.desc:<52} "
              f"{r['total_return_pct']:>7,.1f}% {r['max_dd_pct']:>4.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.1f} "
              f"{r['win_rate']:>4.1f}% {r['profit_factor']:>4.2f} {r['rr']:>4.2f} {r['n_trades']:>6d}")

    print(f"\n  ★ = better DD than baseline  ◆ = better Sharpe  ▲ = better Return")
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
        'factor_summary': {
            'n_tagged_trades': len(tagged_trades),
            'btc_vol_ratio_mean': round(float(np.nanmean(btc_vol_ratio_arr)), 4),
            'breadth_mean': round(float(np.nanmean(breadth_arr)), 4),
        },
        'variants': {k: {kk: vv for kk, vv in v.items() if kk != '_b'}
                     for k, v in results.items()},
        'best': {
            'sharpe': best_sharpe[0], 'dd': best_dd[0],
            'calmar': best_calmar[0], 'rr': best_rr[0], 'ret': best_ret[0],
        },
        'elapsed': round(elapsed_total, 1),
    }

    out_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r10_results.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {out_path}")


if __name__ == '__main__':
    main()
