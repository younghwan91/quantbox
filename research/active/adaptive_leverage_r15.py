#!/usr/bin/env python3
"""
R15: Adaptive Leverage — Per-Position Leverage Conditioning
============================================================
R14 found D-KR4 (ConfHold + R4 base) as overall winner:
  - D-KR4:  6,719%, Calmar 1062, Sharpe 6.01, R:R 1.89
  - Baseline (A-Prod): 6,427%, Calmar 1004, Sharpe 5.77, R:R 1.76

Current leverage is FLAT 3.75x (1.5 base × 2.5 MTF mult) for ALL trades.
The production system has confidence/regime leverage infrastructure but
it's entirely disabled (scale=0.0, step=0.0).

R15 hypothesis: Selective leverage allocation boosts returns while
controlling DD — use high leverage on high-conviction setups and
low leverage on weak signals.

Leverage adaptation axes:
  - Confidence: low conf=2x, high conf=7.5x (signal quality gate)
  - Regime: bull=5x, chop=3.75x, bear=2x (macro environment)
  - Conf+Regime: compound both axes
  - Vol-inverse: low vol=5x, high vol=2.5x (calm market = safer lev)
  - VWMA-distance: far=6x rubberband = higher conviction

Tested on BOTH bases: A-Prod (baseline) and D-KR4 (R14 winner)
to see if leverage adaptation generalizes or is base-dependent.

Variants (16):
  G0: Controls (A-Prod 3.75x, D-KR4 3.75x)
  G1: Flat leverage sweep (2x, 5x, 5x on DKR4)
  G2: Confidence-adaptive (both bases + aggressive DKR4)
  G3: Regime-adaptive (both bases)
  G4: Conf+Regime combined (both bases)
  G5: Vol-inverse + VWMA-distance (DKR4)
  G6: Full stack (both bases)

Data: 6yr (2190d), 50 symbols, R12-identical entry logic.
VPVR cached, all R14 adaptive exits inherited.

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/adaptive_leverage_r15.py
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import numba
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
    compute_atr_pct,
)

CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache_extended'
VPVR_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'vpvr'
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
BARS_PER_DAY = 96

CAP_MULT = {2: 0.3, 3: 0.5, 4: 1.5, 5: 2.0}
SOFT_EQ = [(0.08, 0.50), (0.05, 0.70), (0.03, 0.85)]

VPVR_LOOKBACK = 672
VPVR_NBINS = 50
VPVR_VA_PCT = 0.70

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
# Variant Configuration — R13: Adaptive exits
# ═══════════════════════════════════════════════════════════════════
@dataclass
class VConfig:
    """Variant config. Defaults = production I-HybTight + U-RR+VWS."""
    name: str = 'A-Base'
    desc: str = ''

    # ── Base exit params (production) ──
    target_retrace: float = 1.5
    stop_mult: float = 0.382
    max_hold: int = 25
    pre_tp1_activation: float = 0.5
    pre_tp1_factor: float = 0.3

    # PBT
    partial_exit: bool = True
    partial_frac: float = 0.5
    be_after_tp1: bool = True
    trail_atr: float = 1.0
    runner_mh_mult: float = 3.0

    # Portfolio
    max_slots: int = 2
    dir_div: bool = True
    steep_cap: bool = True
    vol_adj: bool = True
    soft_eq: bool = True

    # U-RR+VWS
    rr_size: bool = True
    rr_size_min: float = 0.5
    rr_size_max: float = 2.0
    vwma_stop: str = 'wide'
    vwma_stop_mult: float = 0.50

    # ── ADAPTIVE EXIT (R13+R14 innovation) ──
    # Modes: 'off', 'regime', 'vol', 'conf', 'vwma', 'regime+vol', 'regime+conf'
    adapt_target: str = 'off'
    adapt_hold: str = 'off'
    adapt_trail: str = 'off'   # 'off', 'regime', 'vwma'
    adapt_stop: str = 'off'    # 'off', 'vol', 'conf'
    adapt_activation: str = 'off'  # R14: 'off', 'conf'

    # Regime-adaptive params: bull(+1) / choppy(0) / bear(-1)
    reg_tgt_bull: float = 1.8
    reg_tgt_chop: float = 1.5
    reg_tgt_bear: float = 1.2
    reg_hold_bull: int = 30
    reg_hold_chop: int = 25
    reg_hold_bear: int = 18
    reg_trail_bull: float = 0.15
    reg_trail_chop: float = 0.30
    reg_trail_bear: float = 0.40

    # Volatility-adaptive params (scale relative to vol_ref)
    vol_ref: float = 0.015       # reference ATR%
    vol_tgt_min: float = 0.8
    vol_tgt_max: float = 2.5
    vol_hold_min: int = 15
    vol_hold_max: int = 40
    vol_stop_min: float = 0.20
    vol_stop_max: float = 0.60

    # Confidence-adaptive params (conf 3/4/5/6)
    conf_tgt_3: float = 1.0
    conf_tgt_4: float = 1.3
    conf_tgt_5: float = 1.8
    conf_tgt_6: float = 2.0
    conf_hold_3: int = 18
    conf_hold_4: int = 22
    conf_hold_5: int = 28
    conf_hold_6: int = 32

    # R14: Confidence-adaptive stop (low conf = tighter stop)
    conf_stop_3: float = 0.25
    conf_stop_4: float = 0.32
    conf_stop_5: float = 0.38
    conf_stop_6: float = 0.45

    # R14: Confidence-adaptive pre-TP1 activation
    conf_act_3: float = 0.35
    conf_act_4: float = 0.45
    conf_act_5: float = 0.55
    conf_act_6: float = 0.65

    # R14: VWMA-distance adaptive params
    # Reference: median VWMA distance ~ 2% for good trades
    vwma_ref: float = 0.02
    vwma_tgt_min: float = 0.8   # target mult for small VWMA dist
    vwma_tgt_max: float = 2.5   # target mult for large VWMA dist
    vwma_hold_min: int = 15
    vwma_hold_max: int = 40
    vwma_trail_tight: float = 0.40  # trail factor for small VWMA dist
    vwma_trail_loose: float = 0.15  # trail factor for large VWMA dist

    # ── ADAPTIVE LEVERAGE (R15 innovation) ──
    # Modes: 'off', 'conf', 'regime', 'conf_regime', 'vol', 'vwma'
    base_lev: float = 3.75       # base effective leverage (replaces global EFFECTIVE_LEV)
    adapt_lev: str = 'off'       # leverage adaptation mode

    # Confidence leverage: confidence → leverage mapping
    conf_lev_3: float = 2.0
    conf_lev_4: float = 3.0
    conf_lev_5: float = 5.0
    conf_lev_6: float = 7.5

    # Regime leverage: direct regime → leverage
    regime_lev_bull: float = 5.0
    regime_lev_chop: float = 3.75
    regime_lev_bear: float = 2.0

    # Regime multiplier (for conf_regime mode, applied ON TOP of conf_lev)
    regime_mult_bull: float = 1.2
    regime_mult_chop: float = 1.0
    regime_mult_bear: float = 0.6

    # Volatility-inverse leverage (low ATR% → high lev)
    vol_lev_ref: float = 0.015    # ATR% reference
    vol_lev_low: float = 5.0      # leverage when vol << ref
    vol_lev_high: float = 2.5     # leverage when vol >> ref

    # VWMA-distance leverage (far from VWMA → high lev)
    vwma_lev_near: float = 2.5    # leverage when close to VWMA
    vwma_lev_far: float = 6.0     # leverage when far from VWMA
    vwma_lev_ref: float = 0.03    # 3% distance = max leverage

    # Hard cap
    max_lev: float = 20.0


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent.parent),
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
# Entry scanner (production-identical, from R12)
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
    """Production-identical entry scanner."""
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
# VWMA / VPVR / Regime — identical to R4
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def compute_vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
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


@numba.jit(nopython=True, cache=True)
def compute_vpvr(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, lookback: int, n_bins: int, va_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    poc = np.full(n, np.nan)
    va_high = np.full(n, np.nan)
    va_low = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        start = i - lookback + 1
        p_min = low[start]
        p_max = high[start]
        for j in range(start + 1, i + 1):
            if low[j] < p_min: p_min = low[j]
            if high[j] > p_max: p_max = high[j]

        price_range = p_max - p_min
        if price_range < 1e-10:
            poc[i] = close[i]; va_high[i] = close[i]; va_low[i] = close[i]
            continue

        bin_size = price_range / n_bins
        vol_hist = np.zeros(n_bins)
        for j in range(start, i + 1):
            bar_vol = volume[j]
            if bar_vol <= 0: continue
            lo_bin = int((low[j] - p_min) / bin_size)
            hi_bin = int((high[j] - p_min) / bin_size)
            if lo_bin < 0: lo_bin = 0
            if hi_bin >= n_bins: hi_bin = n_bins - 1
            if lo_bin > hi_bin: lo_bin = hi_bin
            n_covered = hi_bin - lo_bin + 1
            per_bin_vol = bar_vol / n_covered
            for b in range(lo_bin, hi_bin + 1):
                vol_hist[b] += per_bin_vol

        max_vol = -1.0
        poc_bin = 0
        for b in range(n_bins):
            if vol_hist[b] > max_vol:
                max_vol = vol_hist[b]; poc_bin = b
        poc[i] = p_min + (poc_bin + 0.5) * bin_size

        total_vol = 0.0
        for b in range(n_bins): total_vol += vol_hist[b]
        if total_vol <= 0:
            va_high[i] = p_max; va_low[i] = p_min; continue

        va_vol_target = total_vol * va_pct
        va_lo_bin = poc_bin; va_hi_bin = poc_bin
        captured = vol_hist[poc_bin]
        while captured < va_vol_target:
            can_lo = va_lo_bin > 0
            can_hi = va_hi_bin < n_bins - 1
            if not can_lo and not can_hi: break
            lo_vol = vol_hist[va_lo_bin - 1] if can_lo else -1.0
            hi_vol = vol_hist[va_hi_bin + 1] if can_hi else -1.0
            if lo_vol >= hi_vol:
                va_lo_bin -= 1; captured += vol_hist[va_lo_bin]
            else:
                va_hi_bin += 1; captured += vol_hist[va_hi_bin]

        va_low[i] = p_min + va_lo_bin * bin_size
        va_high[i] = p_min + (va_hi_bin + 1) * bin_size

    return poc, va_high, va_low


def compute_dual_confirm_regime(btc_4h_close: np.ndarray, n_15m: int) -> np.ndarray:
    n_4h = len(btc_4h_close)
    sma50 = np.full(n_4h, np.nan)
    sma200 = np.full(n_4h, np.nan)
    s50, s200 = 0.0, 0.0
    for j in range(n_4h):
        s50 += btc_4h_close[j]; s200 += btc_4h_close[j]
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
# Adaptive param computation (R13+R14 core innovation)
# ═══════════════════════════════════════════════════════════════════
def _resolve_adaptive_params(
    cfg: VConfig, regime_val: int, entry_atr: float, confidence: int,
    vwma_dist: float = 0.0,
) -> tuple[float, int, float, float, float]:
    """Compute per-position effective exit params at entry time.

    Returns:
        (eff_target_retrace, eff_max_hold, eff_trail_factor, eff_stop_mult,
         eff_activation)
    """
    eff_tgt = cfg.target_retrace
    eff_hold = cfg.max_hold
    eff_trail = cfg.pre_tp1_factor
    eff_stop = cfg.stop_mult
    eff_act = cfg.pre_tp1_activation

    # ── Target adaptation ──
    at = cfg.adapt_target
    if at == 'regime':
        if regime_val == 1:    eff_tgt = cfg.reg_tgt_bull
        elif regime_val == 0:  eff_tgt = cfg.reg_tgt_chop
        else:                  eff_tgt = cfg.reg_tgt_bear
    elif at == 'vol':
        vol_scale = entry_atr / cfg.vol_ref if cfg.vol_ref > 0 else 1.0
        eff_tgt = cfg.target_retrace * vol_scale
        eff_tgt = max(cfg.vol_tgt_min, min(eff_tgt, cfg.vol_tgt_max))
    elif at == 'conf':
        ct = {3: cfg.conf_tgt_3, 4: cfg.conf_tgt_4,
              5: cfg.conf_tgt_5, 6: cfg.conf_tgt_6}
        eff_tgt = ct.get(min(confidence, 6), cfg.target_retrace)
    elif at == 'vwma':
        # R14: Large VWMA dist → wider target, small → tighter
        vw_scale = vwma_dist / cfg.vwma_ref if cfg.vwma_ref > 0 else 1.0
        eff_tgt = cfg.target_retrace * vw_scale
        eff_tgt = max(cfg.vwma_tgt_min, min(eff_tgt, cfg.vwma_tgt_max))
    elif at == 'regime+vol':
        if regime_val == 1:    base_t = cfg.reg_tgt_bull
        elif regime_val == 0:  base_t = cfg.reg_tgt_chop
        else:                  base_t = cfg.reg_tgt_bear
        vol_scale = entry_atr / cfg.vol_ref if cfg.vol_ref > 0 else 1.0
        eff_tgt = base_t * vol_scale
        eff_tgt = max(cfg.vol_tgt_min, min(eff_tgt, cfg.vol_tgt_max))
    elif at == 'regime+conf':
        if regime_val == 1:    base_t = cfg.reg_tgt_bull
        elif regime_val == 0:  base_t = cfg.reg_tgt_chop
        else:                  base_t = cfg.reg_tgt_bear
        conf_mult = {3: 0.7, 4: 0.9, 5: 1.1, 6: 1.3}.get(
            min(confidence, 6), 1.0)
        eff_tgt = base_t * conf_mult
        eff_tgt = max(0.5, min(eff_tgt, 3.0))

    # ── Hold adaptation ──
    ah = cfg.adapt_hold
    if ah == 'regime':
        if regime_val == 1:    eff_hold = cfg.reg_hold_bull
        elif regime_val == 0:  eff_hold = cfg.reg_hold_chop
        else:                  eff_hold = cfg.reg_hold_bear
    elif ah == 'vol':
        vol_scale = entry_atr / cfg.vol_ref if cfg.vol_ref > 0 else 1.0
        eff_hold = int(cfg.max_hold * vol_scale)
        eff_hold = max(cfg.vol_hold_min, min(eff_hold, cfg.vol_hold_max))
    elif ah == 'conf':
        ch = {3: cfg.conf_hold_3, 4: cfg.conf_hold_4,
              5: cfg.conf_hold_5, 6: cfg.conf_hold_6}
        eff_hold = ch.get(min(confidence, 6), cfg.max_hold)
    elif ah == 'vwma':
        # R14: Large VWMA dist → longer hold (higher EV trades need more room)
        vw_scale = vwma_dist / cfg.vwma_ref if cfg.vwma_ref > 0 else 1.0
        eff_hold = int(cfg.max_hold * vw_scale)
        eff_hold = max(cfg.vwma_hold_min, min(eff_hold, cfg.vwma_hold_max))

    # ── Trail adaptation ──
    if cfg.adapt_trail == 'regime':
        if regime_val == 1:    eff_trail = cfg.reg_trail_bull
        elif regime_val == 0:  eff_trail = cfg.reg_trail_chop
        else:                  eff_trail = cfg.reg_trail_bear
    elif cfg.adapt_trail == 'vwma':
        # R14: Large VWMA dist → looser trail (more rope for high-EV trades)
        vw_scale = vwma_dist / cfg.vwma_ref if cfg.vwma_ref > 0 else 1.0
        # Interpolate: scale=0.5 → tight, scale=2 → loose
        alpha = max(0.0, min((vw_scale - 0.5) / 1.5, 1.0))
        eff_trail = cfg.vwma_trail_tight + alpha * (
            cfg.vwma_trail_loose - cfg.vwma_trail_tight)

    # ── Stop adaptation ──
    if cfg.adapt_stop == 'vol':
        vol_scale = entry_atr / cfg.vol_ref if cfg.vol_ref > 0 else 1.0
        eff_stop = cfg.stop_mult * vol_scale
        eff_stop = max(cfg.vol_stop_min, min(eff_stop, cfg.vol_stop_max))
    elif cfg.adapt_stop == 'conf':
        # R14: Low confidence → tighter stop (cut losers faster)
        cs = {3: cfg.conf_stop_3, 4: cfg.conf_stop_4,
              5: cfg.conf_stop_5, 6: cfg.conf_stop_6}
        eff_stop = cs.get(min(confidence, 6), cfg.stop_mult)

    # ── Activation adaptation (R14) ──
    if cfg.adapt_activation == 'conf':
        # Low conf → earlier trail activation (grab profits sooner)
        ca = {3: cfg.conf_act_3, 4: cfg.conf_act_4,
              5: cfg.conf_act_5, 6: cfg.conf_act_6}
        eff_act = ca.get(min(confidence, 6), cfg.pre_tp1_activation)

    return eff_tgt, eff_hold, eff_trail, eff_stop, eff_act


# ═══════════════════════════════════════════════════════════════════
# Adaptive leverage computation (R15 core innovation)
# ═══════════════════════════════════════════════════════════════════
def _resolve_leverage(
    cfg: VConfig, regime_val: int, entry_atr: float, confidence: int,
    vwma_dist: float = 0.0,
) -> float:
    """Compute per-position effective leverage at entry time.

    Returns:
        Effective leverage for this position, capped at cfg.max_lev.
    """
    if cfg.adapt_lev == 'off':
        return cfg.base_lev

    if cfg.adapt_lev == 'conf':
        lev_map = {3: cfg.conf_lev_3, 4: cfg.conf_lev_4,
                   5: cfg.conf_lev_5, 6: cfg.conf_lev_6}
        lev = lev_map.get(min(confidence, 6), cfg.base_lev)

    elif cfg.adapt_lev == 'regime':
        if regime_val == 1:    lev = cfg.regime_lev_bull
        elif regime_val == -1: lev = cfg.regime_lev_bear
        else:                  lev = cfg.regime_lev_chop

    elif cfg.adapt_lev == 'conf_regime':
        # Confidence sets base leverage, regime multiplies
        lev_map = {3: cfg.conf_lev_3, 4: cfg.conf_lev_4,
                   5: cfg.conf_lev_5, 6: cfg.conf_lev_6}
        lev = lev_map.get(min(confidence, 6), cfg.base_lev)
        if regime_val == 1:    lev *= cfg.regime_mult_bull
        elif regime_val == -1: lev *= cfg.regime_mult_bear
        else:                  lev *= cfg.regime_mult_chop

    elif cfg.adapt_lev == 'vol':
        # Low ATR% → high leverage (mean-rev environment = safer)
        ratio = cfg.vol_lev_ref / max(entry_atr, 0.001)
        # ratio > 1 = low vol, < 1 = high vol
        alpha = max(0.0, min((ratio - 0.5) / 1.5, 1.0))
        lev = cfg.vol_lev_high + alpha * (cfg.vol_lev_low - cfg.vol_lev_high)

    elif cfg.adapt_lev == 'vwma':
        # Far from VWMA = deep rubberband = higher conviction → higher lev
        alpha = min(vwma_dist / cfg.vwma_lev_ref, 1.0)
        lev = cfg.vwma_lev_near + alpha * (cfg.vwma_lev_far - cfg.vwma_lev_near)

    else:
        lev = cfg.base_lev

    return min(lev, cfg.max_lev)


# ═══════════════════════════════════════════════════════════════════
# Portfolio Simulation — R12-parity + adaptive per-position exits
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
    """Run portfolio sim with per-position adaptive exit params.

    R12-parity: close-based gain for ALL exit checks.
    R13 innovation: target/hold/trail/stop adapt per-position at entry time.
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

            # ═══ STOP (per-position adaptive) ═══
            base_stop = pos['move_size'] * pos['eff_stop_mult']
            vw_dist = pos.get('vwma_dist', 0.0)
            if cfg.vwma_stop != 'off' and vw_dist > 0:
                vw_stop = max(vw_dist * cfg.vwma_stop_mult, 0.003)
                if cfg.vwma_stop == 'replace':
                    base_stop = vw_stop
                elif cfg.vwma_stop == 'tight':
                    base_stop = min(base_stop, vw_stop)
                elif cfg.vwma_stop == 'wide':
                    base_stop = max(base_stop, vw_stop)
            effective_stop = base_stop

            # ═══ TARGET + HOLD (per-position adaptive) ═══
            target = pos['move_size'] * pos['eff_target_retrace']
            max_hold = pos['eff_max_hold']

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
                        'target_gain': target,
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0 if cfg.be_after_tp1 else -effective_stop
                    pos['stepup_active'] = False
                elif gain < -effective_stop:
                    _close(balance, trades, pos, sym, gain, lev,
                           'stop', to_close, i, target)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= max_hold:
                    _close(balance, trades, pos, sym, gain, lev,
                           'time', to_close, i, target)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    # Trail factor + activation are per-position adaptive (R14)
                    if (pos['best_pnl'] > target * pos.get('eff_activation', cfg.pre_tp1_activation)
                            and gain < pos['best_pnl'] * pos['eff_trail_factor']):
                        _close(balance, trades, pos, sym, gain, lev,
                               'trail', to_close, i, target)
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

                bars_since_tp1 = i - pos.get('tp1_bar', pos['entry_bar'])
                target_val = pos['move_size'] * pos['eff_target_retrace']

                trail_stop = pos['highest_since_tp1'] - cfg.trail_atr * atr_val
                floor_val = pos.get('trail_floor', 0.0)
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
                        'target_gain': target_val,
                        'highest_gain': pos.get('highest_since_tp1', gain),
                        'runner_bars': bars_since_tp1,
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
                    if (pos['best_pnl'] > target * pos.get('eff_activation', cfg.pre_tp1_activation)
                            and gain < pos['best_pnl'] * pos['eff_trail_factor']):
                        exit_reason = 'trail'
                    elif pos['hold_count'] >= max_hold:
                        exit_reason = 'time'
                if exit_reason:
                    _close(balance, trades, pos, sym, gain, lev,
                           exit_reason, to_close, i, target)
                    balance = trades[-1]['_b']

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Entry phase (U-RR+VWS + adaptive param resolution) ═══
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

                # Read POC at entry
                poc_val = np.nan
                poc_dist_entry = 0.0
                if sym_poc is not None:
                    poc_arr = sym_poc.get(sym)
                    if poc_arr is not None and i < len(poc_arr):
                        poc_val = poc_arr[i]
                if not np.isnan(poc_val) and c_entry > 0:
                    if direction == 1:
                        poc_dist_entry = (poc_val - c_entry) / c_entry
                    else:
                        poc_dist_entry = (c_entry - poc_val) / c_entry
                    poc_dist_entry = max(poc_dist_entry, 0.001)

                # Read ATR at entry (for adaptive params)
                entry_atr = 0.015
                if sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        entry_atr = aa[i]

                if cfg.dir_div:
                    same = sum(1 for p in positions.values()
                               if p['direction'] == direction)
                    if same >= 2:
                        continue

                candidates.append((
                    sym, direction, confidence, move_size,
                    vwma_dist_entry, poc_dist_entry, entry_atr,
                ))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for (sym, direction, confidence, move_size,
                 vwma_dist_e, poc_dist_e, entry_atr) in candidates:
                if len(positions) >= cfg.max_slots:
                    break

                per_slot = INITIAL_CAPITAL / max(1, len(positions) + 1)
                per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if per_slot < 10.0:
                    continue

                if cfg.steep_cap:
                    mult = CAP_MULT.get(confidence, CAP_MULT.get(
                        min(confidence, 5), 0.3))
                    per_slot *= mult
                    per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if cfg.vol_adj and sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        if aa[i] > 0:
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

                if cfg.rr_size and poc_dist_e > 0:
                    implied_target = poc_dist_e
                    implied_stop = move_size * cfg.stop_mult
                    if implied_stop > 0:
                        implied_rr = implied_target / implied_stop
                        rr_factor = max(cfg.rr_size_min,
                                        min(0.5 + 0.5 * implied_rr, cfg.rr_size_max))
                        per_slot *= rr_factor
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost

                # ═══ R14: Resolve adaptive exit params at entry ═══
                eff_tgt, eff_hold, eff_trail, eff_stop, eff_act = _resolve_adaptive_params(
                    cfg, int(regime[i]), entry_atr, confidence, vwma_dist_e,
                )

                # ═══ R15: Resolve adaptive leverage at entry ═══
                eff_lev = _resolve_leverage(
                    cfg, int(regime[i]), entry_atr, confidence, vwma_dist_e,
                )
                # Recompute entry cost with actual leverage
                balance += entry_cost  # undo the flat-rate cost
                entry_cost = eff_lev * COST_PER_SIDE * per_slot
                balance -= entry_cost

                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': sym_close[sym][i],
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': eff_lev,
                    'entry_regime': int(regime[i]),
                    'vwma_dist': vwma_dist_e,
                    'poc_dist': poc_dist_e,
                    # Per-position adaptive params (R14)
                    'eff_target_retrace': eff_tgt,
                    'eff_max_hold': eff_hold,
                    'eff_trail_factor': eff_trail,
                    'eff_stop_mult': eff_stop,
                    'eff_activation': eff_act,
                }

        # Equity update
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

    return _compute_stats(equity_curve, trades, max_dd, n_bars)


def _close(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx,
           target_gain=0.0):
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
        'target_gain': target_gain,
        '_b': new_b,
    })
    to_close.append(sym)


# ═══════════════════════════════════════════════════════════════════
# Stats computation — identical to R4
# ═══════════════════════════════════════════════════════════════════
def _compute_stats(eq, trades, max_dd, n_bars):
    n = len(trades)
    if n == 0:
        return {'n_trades': 0, 'yearly': {}}

    overall = _calc_metrics(eq, trades, max_dd)

    bars_per_year = BARS_PER_YEAR
    yearly = {}
    for yr in range(6):
        yr_start = yr * bars_per_year
        yr_end = min((yr + 1) * bars_per_year, n_bars)
        if yr_start >= n_bars:
            break
        yr_trades = [t for t in trades if yr_start <= t['entry_bar'] < yr_end]
        if not yr_trades:
            yearly[f'Y{yr+1}'] = {'n_trades': 0}
            continue
        yr_eq = eq[yr_start:yr_end]
        if len(yr_eq) < 2:
            continue
        yr_start_val = yr_eq[0] if yr_eq[0] > 0 else INITIAL_CAPITAL
        yr_peak = yr_start_val
        yr_max_dd = 0.0
        for val in yr_eq:
            if val > yr_peak: yr_peak = val
            dd_v = (yr_peak - val) / yr_peak if yr_peak > 0 else 0
            if dd_v > yr_max_dd: yr_max_dd = dd_v
        yearly[f'Y{yr+1}'] = _calc_metrics_from_trades(
            yr_trades, yr_eq, yr_max_dd, yr_start_val)

    overall['yearly'] = yearly

    exit_counts = {}
    for t in trades:
        r = t['exit_reason']
        exit_counts[r] = exit_counts.get(r, 0) + 1
    overall['exit_breakdown'] = exit_counts

    tp1_trades = [t for t in trades if t['exit_reason'] == 'tp1']
    runner_trades = [t for t in trades if t['exit_reason'] in (
        'trail_runner', 'time_runner')]
    total_entries = len(tp1_trades) + len([
        t for t in trades if t['exit_reason'] in (
            'stop', 'time', 'trail', 'target')])
    overall['tp1_hit_rate'] = round(
        len(tp1_trades) / total_entries * 100, 1) if total_entries > 0 else 0.0
    overall['tp1_count'] = len(tp1_trades)
    overall['runner_count'] = len(runner_trades)

    if runner_trades:
        r_wins = [t for t in runner_trades if t['pnl_usd'] > 0]
        r_pnls = [t['pnl_pct'] for t in runner_trades]
        r_bars = [t.get('runner_bars', 0) for t in runner_trades if 'runner_bars' in t]
        overall['runner'] = {
            'count': len(runner_trades),
            'win_rate': round(len(r_wins) / len(runner_trades) * 100, 1),
            'avg_pnl_pct': round(float(np.mean(r_pnls)), 3),
            'avg_bars': round(float(np.mean(r_bars)), 1) if r_bars else 0.0,
        }
    else:
        overall['runner'] = {'count': 0}

    return overall


def _calc_metrics(eq, trades, max_dd):
    n = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    total_ret = (eq[-1] / INITIAL_CAPITAL - 1.0) * 100
    gp = sum(t['pnl_usd'] for t in wins)
    gl = sum(abs(t['pnl_usd']) for t in losses)
    pf = gp / gl if gl > 0 else 999.0
    eq_rets = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(
        BARS_PER_YEAR)) if len(eq_rets) > 0 and np.std(eq_rets) > 0 else 0.0
    aw = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0
    al = float(np.mean([abs(t['pnl_pct']) for t in losses])) if losses else 1.0
    rr = aw / al if al > 0 else 999.0
    n_days = len(eq) / BARS_PER_DAY
    return {
        'total_return_pct': round(total_ret, 1),
        'final_balance': round(eq[-1], 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(total_ret / (max_dd * 100), 1) if max_dd > 0 else 0.0,
        'n_trades': n,
        'trades_per_day': round(n / n_days, 2) if n_days > 0 else 0.0,
        'win_rate': round(len(wins) / n * 100, 1) if n > 0 else 0.0,
        'profit_factor': round(pf, 2),
        'rr': round(rr, 2),
        'avg_win_pct': round(aw, 3),
        'avg_loss_pct': round(al, 3),
    }


def _calc_metrics_from_trades(trades, eq_seg, max_dd, start_val):
    n = len(trades)
    if n == 0:
        return {'n_trades': 0}
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    total_pnl = sum(t['pnl_usd'] for t in trades)
    ret_pct = (total_pnl / start_val) * 100 if start_val > 0 else 0.0
    gp = sum(t['pnl_usd'] for t in wins)
    gl = sum(abs(t['pnl_usd']) for t in losses)
    pf = gp / gl if gl > 0 else 999.0
    aw = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0
    al = float(np.mean([abs(t['pnl_pct']) for t in losses])) if losses else 1.0
    rr = aw / al if al > 0 else 999.0
    eq_rets = np.diff(eq_seg) / np.maximum(eq_seg[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(
        BARS_PER_YEAR)) if len(eq_rets) > 0 and np.std(eq_rets) > 0 else 0.0
    n_days = len(eq_seg) / BARS_PER_DAY
    tp1_count = sum(1 for t in trades if t['exit_reason'] == 'tp1')
    entry_count = sum(1 for t in trades if t['exit_reason'] in (
        'tp1', 'stop', 'time', 'trail', 'target'))
    tp1_rate = round(tp1_count / entry_count * 100, 1) if entry_count > 0 else 0.0
    return {
        'n_trades': n,
        'ret_pct': round(ret_pct, 1),
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'win_rate': round(len(wins) / n * 100, 1) if n > 0 else 0.0,
        'profit_factor': round(pf, 2),
        'rr': round(rr, 2),
        'trades_per_day': round(n / n_days, 2) if n_days > 0 else 0.0,
        'tp1_hit_rate': tp1_rate,
    }


# ═══════════════════════════════════════════════════════════════════
# Variant Definitions — R13: Adaptive Exit Parameters
# ═══════════════════════════════════════════════════════════════════
# D-KR4 base params (R14 winner) — reusable dict
_DKR4 = dict(pre_tp1_factor=0.15, stop_mult=0.30, adapt_hold='conf')


def build_variants() -> list[VConfig]:
    """R15: Adaptive Leverage — Per-position leverage conditioning.

    R14 findings:
    - D-KR4: 6,719%, Calmar 1062, Sharpe 6.01, R:R 1.89 (★ WINNER)
    - A-Prod: 6,427%, Calmar 1004, Sharpe 5.77, R:R 1.76 (baseline)
    - Current leverage: FLAT 3.75x for all trades (1.5 * 2.5 MTF)
    - Max DD: 6.3-6.4% at 3.75x → room to experiment up to ~15% DD budget

    R15 hypothesis: Selectively allocating leverage by signal quality
    (confidence, regime, vol, VWMA distance) concentrates risk on
    high-conviction trades, boosting returns while capping DD.

    Design: Each axis tested on BOTH A-Prod and D-KR4 bases.
    """
    v = []

    # ═══ G0: CONTROLS (flat 3.75x, same as R14) ═══
    v.append(VConfig(
        name='A-Prod',
        desc='Prod flat 3.75x (baseline)',
    ))
    v.append(VConfig(
        name='B-DKR4',
        desc='DKR4 flat 3.75x (R14 winner)',
        **_DKR4,
    ))

    # ═══ G1: FLAT LEVERAGE SWEEP (risk/return calibration) ═══
    v.append(VConfig(
        name='C-P2x',
        desc='Prod flat 2.0x (conservative)',
        base_lev=2.0,
    ))
    v.append(VConfig(
        name='D-P5x',
        desc='Prod flat 5.0x (aggressive)',
        base_lev=5.0,
    ))
    v.append(VConfig(
        name='E-D5x',
        desc='DKR4 flat 5.0x (aggressive)',
        **_DKR4, base_lev=5.0,
    ))

    # ═══ G2: CONFIDENCE-ADAPTIVE (signal quality → leverage) ═══
    v.append(VConfig(
        name='F-PConf',
        desc='Prod conf→lev (3→2x 4→3x 5→5x 6→7.5x)',
        adapt_lev='conf',
    ))
    v.append(VConfig(
        name='G-DConf',
        desc='DKR4 conf→lev (3→2x 4→3x 5→5x 6→7.5x)',
        **_DKR4, adapt_lev='conf',
    ))
    v.append(VConfig(
        name='H-DConfAg',
        desc='DKR4 conf→lev AGGR (3→1.5 6→12)',
        **_DKR4, adapt_lev='conf',
        conf_lev_3=1.5, conf_lev_4=3.75, conf_lev_5=7.5, conf_lev_6=12.0,
    ))

    # ═══ G3: REGIME-ADAPTIVE (macro environment → leverage) ═══
    v.append(VConfig(
        name='I-PReg',
        desc='Prod regime→lev (B5.0/C3.75/R2.0)',
        adapt_lev='regime',
    ))
    v.append(VConfig(
        name='J-DReg',
        desc='DKR4 regime→lev (B5.0/C3.75/R2.0)',
        **_DKR4, adapt_lev='regime',
    ))

    # ═══ G4: CONF+REGIME COMBINED (dual conditioning) ═══
    v.append(VConfig(
        name='K-PCR',
        desc='Prod conf×regime (bull×1.2 bear×0.6)',
        adapt_lev='conf_regime',
    ))
    v.append(VConfig(
        name='L-DCR',
        desc='DKR4 conf×regime (bull×1.2 bear×0.6)',
        **_DKR4, adapt_lev='conf_regime',
    ))

    # ═══ G5: VOL-INVERSE + VWMA-DISTANCE (DKR4 only) ═══
    v.append(VConfig(
        name='M-DVol',
        desc='DKR4 vol-inv lev (lo=5x hi=2.5x)',
        **_DKR4, adapt_lev='vol',
    ))
    v.append(VConfig(
        name='N-DVwma',
        desc='DKR4 VWMA-dist lev (near=2.5 far=6)',
        **_DKR4, adapt_lev='vwma',
    ))

    # ═══ G6: FULL STACKS (best combo + leverage axes) ═══
    v.append(VConfig(
        name='O-PStack',
        desc='Prod conf+regime lev + ConfHold exit',
        adapt_lev='conf_regime', adapt_hold='conf',
    ))
    v.append(VConfig(
        name='P-DStack',
        desc='DKR4 conf+regime lev (full adaptive)',
        **_DKR4, adapt_lev='conf_regime',
    ))

    return v


# ═══════════════════════════════════════════════════════════════════
# Multiprocessing helpers (pool initializer pattern)
# ═══════════════════════════════════════════════════════════════════
_pool_shared: dict = {}


def _pool_init(shared: dict):
    """Called once per worker process. Store shared data in global."""
    global _pool_shared
    _pool_shared = shared


def _pool_simulate(cfg: VConfig) -> dict:
    """Worker function — reads shared data from global, only cfg is pickled."""
    d = _pool_shared
    return simulate(
        d['sym_close'], d['sym_high'], d['sym_low'],
        d['sym_dirs'], d['sym_confs'], d['sym_moves'],
        cfg, d['regime'], d['sym_atr'], d['sym_vwma'],
        d['sym_poc'], d['sym_va_high'], d['sym_va_low'],
    )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    commit = _get_git_commit()
    t0 = time.time()
    w = 120

    print("=" * w)
    print("  R15: ADAPTIVE LEVERAGE — Per-Position Leverage Conditioning")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baseline: A-Prod 3.75x flat + D-KR4 3.75x flat (R14 winner)")
    print(f"  Innovation: Per-position adaptive leverage (conf/regime/vol/vwma)")
    print(f"  Data: {DATA_DAYS}d ({DATA_DAYS//365}yr), 50 symbols")
    print(f"  Scoring: Recency-weighted (Y5-6 = 2× weight)")
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
        dirs, confs, moves = apply_mtf_filter(
            dirs, confs, moves, all_4h[sym], MTF_SMA_PERIOD)

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
                if fill == 'first': fv = arr[0]
                elif fill == 'nan': fv = np.nan
                else: fv = 0.0
                d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

    raw_signals = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    print(f"  Left-padded to {max_bars} bars ({max_bars//96} days)")
    print(f"  Raw signals: {raw_signals:,}")

    # ── Compute VWMA(100) ──
    print(f"\n📈 Computing VWMA(100) per symbol...")
    sym_vwma: dict[str, np.ndarray] = {}
    for sym in sym_close_d:
        sym_vwma[sym] = compute_vwma(sym_close_d[sym], sym_volume_d[sym], 100)
    vwma_valid = sum(1 for s in sym_vwma if np.any(~np.isnan(sym_vwma[s])))
    print(f"  VWMA computed for {vwma_valid}/{len(sym_vwma)} symbols")

    # ── Compute VPVR (with disk cache) ──
    print(f"\n📊 Computing VPVR ({VPVR_LOOKBACK}-bar lookback, {VPVR_NBINS} bins)...")
    sym_poc: dict[str, np.ndarray] = {}
    sym_va_high: dict[str, np.ndarray] = {}
    sym_va_low: dict[str, np.ndarray] = {}
    vpvr_t0 = time.time()
    VPVR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vpvr_cached = 0
    vpvr_computed = 0

    for idx, sym in enumerate(sorted(sym_close_d.keys())):
        # Build cache key from data hash + VPVR params
        data_hash = hashlib.sha256(
            sym_close_d[sym].tobytes()
            + sym_high_d[sym].tobytes()
            + sym_low_d[sym].tobytes()
            + sym_volume_d[sym].tobytes()
        ).hexdigest()[:16]
        cache_key = f"{sym}_vpvr_{VPVR_LOOKBACK}_{VPVR_NBINS}_{VPVR_VA_PCT}_{data_hash}"
        cache_path = VPVR_CACHE_DIR / f"{cache_key}.npz"

        if cache_path.exists():
            cached = np.load(cache_path)
            sym_poc[sym] = cached['poc']
            sym_va_high[sym] = cached['va_high']
            sym_va_low[sym] = cached['va_low']
            vpvr_cached += 1
        else:
            poc_arr, vah_arr, val_arr = compute_vpvr(
                sym_close_d[sym], sym_high_d[sym], sym_low_d[sym],
                sym_volume_d[sym], VPVR_LOOKBACK, VPVR_NBINS, VPVR_VA_PCT,
            )
            sym_poc[sym] = poc_arr
            sym_va_high[sym] = vah_arr
            sym_va_low[sym] = val_arr
            np.savez_compressed(cache_path, poc=poc_arr, va_high=vah_arr, va_low=val_arr)
            vpvr_computed += 1

        if (idx + 1) % 10 == 0:
            elapsed_vpvr = time.time() - vpvr_t0
            eta = elapsed_vpvr / (idx + 1) * (len(sym_close_d) - idx - 1)
            print(f"  [{idx+1}/{len(sym_close_d)}] {sym} done "
                  f"({elapsed_vpvr:.1f}s, ETA {eta:.0f}s)")

    vpvr_elapsed = time.time() - vpvr_t0
    poc_valid = sum(1 for s in sym_poc if np.any(~np.isnan(sym_poc[s])))
    print(f"  VPVR: {vpvr_cached} cached + {vpvr_computed} computed "
          f"for {poc_valid}/{len(sym_poc)} symbols in {vpvr_elapsed:.1f}s")

    # ── Compute regime ──
    print(f"\n🌐 Computing regime...")
    regime = compute_dual_confirm_regime(btc_4h_close, max_bars)

    # Print regime distribution
    bull = np.sum(regime == 1)
    bear = np.sum(regime == -1)
    chop = np.sum(regime == 0)
    total_r = bull + bear + chop
    print(f"  Bull: {bull/total_r*100:.1f}%  Choppy: {chop/total_r*100:.1f}%  "
          f"Bear: {bear/total_r*100:.1f}%")

    # ═══════════════════ VARIANT TEST (parallel) ═══════════════════
    variants = build_variants()
    print(f"\n{'═' * w}")
    print(f"  R15 A/B TEST — {len(variants)} VARIANTS (parallel)")
    print(f"{'═' * w}")

    sim_t0 = time.time()
    n_cores = os.cpu_count() or 1
    n_workers = min(len(variants), max(1, n_cores))

    # Shared data dict — passed once to pool initializer, not pickled per task
    shared = {
        'sym_close': sym_close_d, 'sym_high': sym_high_d, 'sym_low': sym_low_d,
        'sym_dirs': sym_dirs, 'sym_confs': sym_confs, 'sym_moves': sym_moves,
        'regime': regime, 'sym_atr': sym_atr_d, 'sym_vwma': sym_vwma,
        'sym_poc': sym_poc, 'sym_va_high': sym_va_high, 'sym_va_low': sym_va_low,
    }

    results = {}
    if n_workers >= 2:
        print(f"  Workers: {n_workers} (CPU cores: {n_cores})")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_init,
            initargs=(shared,),
        ) as executor:
            future_map = {
                executor.submit(_pool_simulate, cfg): cfg
                for cfg in variants
            }
            for future in future_map:
                cfg = future_map[future]
                results[cfg.name] = future.result()
    else:
        print(f"  Serial mode (CPU cores: {n_cores})")
        for vi, cfg in enumerate(variants):
            t_v = time.time()
            res = simulate(
                sym_close_d, sym_high_d, sym_low_d,
                sym_dirs, sym_confs, sym_moves,
                cfg, regime, sym_atr_d, sym_vwma,
                sym_poc, sym_va_high, sym_va_low,
            )
            elapsed_v = time.time() - t_v
            results[cfg.name] = res
            tp1r = res.get('tp1_hit_rate', 0)
            print(f"  [{vi+1}/{len(variants)}] {cfg.name}: "
                  f"Ret={res['total_return_pct']:,.1f}%  DD={res['max_dd_pct']:.1f}%  "
                  f"Sharpe={res['sharpe']:.2f}  Calmar={res['calmar']:.1f}  "
                  f"WR={res['win_rate']:.1f}%  PF={res['profit_factor']:.2f}  "
                  f"R:R={res['rr']:.2f}  TP1%={tp1r:.0f}%  "
                  f"#Tr={res['n_trades']}  [{elapsed_v:.1f}s]")

    sim_elapsed = time.time() - sim_t0
    print(f"\n  All {len(variants)} variants completed in {sim_elapsed:.1f}s")

    # Print summary for parallel mode (serial prints per-variant above)
    if n_workers >= 2:
        for vi, cfg in enumerate(variants):
            res = results[cfg.name]
            tp1r = res.get('tp1_hit_rate', 0)
            print(f"  [{vi+1}/{len(variants)}] {cfg.name}: "
                  f"Ret={res['total_return_pct']:,.1f}%  DD={res['max_dd_pct']:.1f}%  "
                  f"Sharpe={res['sharpe']:.2f}  Calmar={res['calmar']:.1f}  "
                  f"WR={res['win_rate']:.1f}%  PF={res['profit_factor']:.2f}  "
                  f"R:R={res['rr']:.2f}  TP1%={tp1r:.0f}%  "
                  f"#Tr={res['n_trades']}")

    elapsed_total = time.time() - t0

    # ═══════════════════ RESULTS TABLE ═══════════════════
    base = results.get('A-Prod', {})
    base_dd = base.get('max_dd_pct', 100)
    base_sharpe = base.get('sharpe', 0)
    base_ret = base.get('total_return_pct', 0)
    base_rr = base.get('rr', 0)
    base_calmar = base.get('calmar', 0)

    print(f"\n{'=' * w}")
    print("  RESULTS — R15 Adaptive Leverage (6yr, 50 symbols)")
    print(f"{'=' * w}")
    print(f"  {'Var':<15} {'Desc':<42} {'Ret%':>8} {'DD%':>5} "
          f"{'Sharpe':>7} {'Calmar':>7} {'WR':>5} {'PF':>5} "
          f"{'R:R':>5} {'TP1%':>5} {'#Tr':>6}")
    print(f"  {'─'*15} {'─'*42} {'─'*8} {'─'*5} {'─'*7} {'─'*7} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*6}")

    for cfg in variants:
        r = results[cfg.name]
        marks = ''
        if r['max_dd_pct'] < base_dd: marks += '★'
        if r['sharpe'] > base_sharpe: marks += '◆'
        if r['total_return_pct'] > base_ret: marks += '▲'
        if r['rr'] > base_rr: marks += '⬆'
        if r['calmar'] > base_calmar: marks += '●'
        marks = marks.ljust(5)

        tp1r = r.get('tp1_hit_rate', 0)
        print(f" {marks}{cfg.name:<15} {cfg.desc:<42} "
              f"{r['total_return_pct']:>7,.1f}% {r['max_dd_pct']:>4.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.1f} "
              f"{r['win_rate']:>4.1f}% {r['profit_factor']:>4.2f} "
              f"{r['rr']:>4.2f} {tp1r:>4.0f}% {r['n_trades']:>6d}")

    print(f"\n  ★=betterDD  ◆=betterSharpe  ▲=betterRet  ⬆=betterR:R  ●=betterCalmar")

    # ═══════════════════ EXIT BREAKDOWN ═══════════════════
    print(f"\n{'=' * w}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'=' * w}")
    print(f"  {'Var':<15} {'stop':>6} {'time':>6} {'trail':>6} "
          f"{'tp1':>6} {'t_run':>6} {'tm_run':>6}")
    print(f"  {'─'*15} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for cfg in variants:
        r = results[cfg.name]
        eb = r.get('exit_breakdown', {})
        print(f"  {cfg.name:<15} "
              f"{eb.get('stop', 0):>6d} {eb.get('time', 0):>6d} "
              f"{eb.get('trail', 0):>6d} {eb.get('tp1', 0):>6d} "
              f"{eb.get('trail_runner', 0):>6d} {eb.get('time_runner', 0):>6d}")

    # ═══════════════════ YEARLY BREAKDOWN ═══════════════════
    print(f"\n{'=' * w}")
    print("  YEARLY PERFORMANCE (Sharpe / R:R / TP1%)")
    print(f"{'=' * w}")
    yr_labels = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6']
    header = f"  {'Var':<15}"
    for yl in yr_labels:
        header += f" {'':>2}{yl}Sh{'':<1} {yl}RR{'':<1} {yl}TP{'':<1}"
    print(header)
    print(f"  {'─'*15}" + " ─────── ────── ─────" * 6)

    for cfg in variants:
        r = results[cfg.name]
        yr_data = r.get('yearly', {})
        line = f"  {cfg.name:<15}"
        for yl in yr_labels:
            yd = yr_data.get(yl, {})
            sh = yd.get('sharpe', 0)
            rr_v = yd.get('rr', 0)
            tp1r = yd.get('tp1_hit_rate', 0)
            line += f"  {sh:>5.1f} {rr_v:>5.2f} {tp1r:>4.0f}%"
        print(line)

    # ═══════════════════ RECENCY SCORE ═══════════════════
    print(f"\n{'=' * w}")
    print("  RECENCY-WEIGHTED COMPOSITE SCORE")
    print(f"  Weights: Y1-4 = 1×, Y5-6 = 2× | "
          f"Sharpe 30%, PF 25%, R:R 20%, inv(DD) 15%, TP1% 10%")
    print(f"{'=' * w}")

    scores = {}
    for cfg in variants:
        r = results[cfg.name]
        yr_data = r.get('yearly', {})

        yr_sharpes, yr_pfs, yr_rrs = [], [], []
        for yi, yl in enumerate(yr_labels):
            yd = yr_data.get(yl, {})
            if yd.get('n_trades', 0) == 0:
                continue
            weight = 2.0 if yi >= 4 else 1.0
            yr_sharpes.append((yd.get('sharpe', 0), weight))
            yr_pfs.append((yd.get('profit_factor', 0), weight))
            yr_rrs.append((yd.get('rr', 0), weight))

        if not yr_sharpes:
            scores[cfg.name] = 0.0
            continue

        def wavg(vals_weights):
            tw = sum(w for _, w in vals_weights)
            return sum(v * w for v, w in vals_weights) / tw if tw > 0 else 0

        w_sharpe = wavg(yr_sharpes)
        w_pf = wavg(yr_pfs)
        w_rr = wavg(yr_rrs)
        inv_dd = 1.0 / max(r.get('max_dd_pct', 1), 0.1)
        tp1_rate = r.get('tp1_hit_rate', 0) / 100.0

        b_sharpe = base.get('sharpe', 1)
        b_pf = base.get('profit_factor', 1)
        b_rr = base.get('rr', 1)
        b_inv_dd = 1.0 / max(base.get('max_dd_pct', 1), 0.1)
        b_tp1 = base.get('tp1_hit_rate', 0) / 100.0

        score = (
            0.30 * (w_sharpe / max(b_sharpe, 0.01))
            + 0.25 * (w_pf / max(b_pf, 0.01))
            + 0.20 * (w_rr / max(b_rr, 0.01))
            + 0.15 * (inv_dd / max(b_inv_dd, 0.01))
            + 0.10 * (tp1_rate / max(b_tp1, 0.001))
        )
        scores[cfg.name] = round(score, 3)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Rank':>4} {'Var':<15} {'Score':>7} {'vs Base':>8} {'Verdict':<15}")
    print(f"  {'─'*4} {'─'*15} {'─'*7} {'─'*8} {'─'*15}")
    for rank, (name, score) in enumerate(ranked, 1):
        delta = score - 1.0
        verdict = '★ WINNER' if rank == 1 else (
            'beat base' if delta > 0 else 'below base')
        print(f"  {rank:>4} {name:<15} {score:>7.3f} {delta:>+7.1%} {verdict}")

    # ═══════════════════ WINNERS ═══════════════════
    best_sharpe = max(results.items(), key=lambda x: x[1].get('sharpe', 0))
    best_dd = min(results.items(), key=lambda x: x[1].get('max_dd_pct', 100))
    best_calmar = max(results.items(), key=lambda x: x[1].get('calmar', 0))
    best_rr = max(results.items(), key=lambda x: x[1].get('rr', 0))
    best_ret = max(results.items(), key=lambda x: x[1].get('total_return_pct', 0))
    best_pf = max(results.items(), key=lambda x: x[1].get('profit_factor', 0))
    best_tp1 = max(results.items(), key=lambda x: x[1].get('tp1_hit_rate', 0))

    print(f"\n  🏆 Best R:R:     {best_rr[0]}  ({best_rr[1]['rr']:.2f})")
    print(f"  🏆 Best Calmar:  {best_calmar[0]}  ({best_calmar[1]['calmar']:.1f})")
    print(f"  🏆 Best Sharpe:  {best_sharpe[0]}  ({best_sharpe[1]['sharpe']:.2f})")
    print(f"  🏆 Best DD:      {best_dd[0]}  ({best_dd[1]['max_dd_pct']:.1f}%)")
    print(f"  🏆 Best PF:      {best_pf[0]}  ({best_pf[1]['profit_factor']:.2f})")
    print(f"  🏆 Best TP1%:    {best_tp1[0]}  ({best_tp1[1]['tp1_hit_rate']:.1f}%)")
    print(f"  🏆 Best Return:  {best_ret[0]}  ({best_ret[1]['total_return_pct']:,.1f}%)")
    print(f"  🏆 Recency:      {ranked[0][0]}  (score={ranked[0][1]:.3f})")

    # ═══════════════════ KEY FINDINGS ═══════════════════
    print(f"\n{'=' * w}")
    print("  KEY COMPARISON: Production vs Top Adaptive Variants")
    print(f"{'=' * w}")
    top_names = ['A-Prod'] + [name for name, _ in ranked[:5] if name != 'A-Prod']
    top_names = list(dict.fromkeys(top_names))[:6]
    for name in top_names:
        if name not in results:
            continue
        r = results[name]
        tp1r = r.get('tp1_hit_rate', 0)
        eb = r.get('exit_breakdown', {})
        cfg_match = [c for c in variants if c.name == name]
        cfg_desc = cfg_match[0].desc if cfg_match else ''
        print(f"\n  {name}: {cfg_desc}")
        print(f"    Ret={r['total_return_pct']:,.1f}%  DD={r['max_dd_pct']:.1f}%  "
              f"Sharpe={r['sharpe']:.2f}  Calmar={r['calmar']:.1f}")
        print(f"    WR={r['win_rate']:.1f}%  PF={r['profit_factor']:.2f}  "
              f"R:R={r['rr']:.2f}  TP1 hit={tp1r:.0f}%")
        print(f"    Exits: stop={eb.get('stop', 0)} time={eb.get('time', 0)} "
              f"trail={eb.get('trail', 0)} tp1={eb.get('tp1', 0)} "
              f"runner={eb.get('trail_runner', 0)+eb.get('time_runner', 0)}")
        yr_data = r.get('yearly', {})
        for yl in ['Y5', 'Y6']:
            yd = yr_data.get(yl, {})
            if yd.get('n_trades', 0) > 0:
                print(f"    {yl}: Sharpe={yd['sharpe']:.1f}  "
                      f"PF={yd['profit_factor']:.2f}  "
                      f"R:R={yd['rr']:.2f}  "
                      f"TP1%={yd.get('tp1_hit_rate', 0):.0f}%  "
                      f"#Tr={yd['n_trades']}")

    print(f"\n  Total time: {elapsed_total:.1f}s")

    # ── Save JSON ──
    output = {
        'commit': commit,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'data_days': DATA_DAYS,
        'n_symbols': len(all_data),
        'baseline': 'U-RR+VWS (R12-parity, close-based gain)',
        'focus': 'R15: Per-position adaptive leverage (conf/regime/vol/vwma) on Prod + DKR4 bases',
        'recency_scores': scores,
        'variants': {},
        'elapsed': round(elapsed_total, 1),
    }
    for k, v_res in results.items():
        clean = {kk: vv for kk, vv in v_res.items() if kk != '_b'}
        output['variants'][k] = clean

    out_path = Path(__file__).parent.parent.parent / 'logs' / 'research'
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / 'adaptive_leverage_r15_results.json'
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {json_path}")


if __name__ == '__main__':
    main()
