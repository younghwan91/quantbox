#!/usr/bin/env python3
"""
Alpha Search R12: Trailing Stop Innovation — Maximize Runner R:R
================================================================
U-RR+VWS (R11 winner, Calmar 1004) is the new baseline. It achieved:
  DD 6.4%, PF 1.66, Sharpe 5.77, WR 58.2%, R:R 1.76

Problem: R:R dropped from 1.91 (baseline) to 1.76 due to wider VWMA stops.
Goal: Recover and exceed R:R through smarter trailing stop mechanics.
Philosophy: "When direction is confirmed (TP1 hit), take as much profit as we can."

Current runner mechanics (ATR×1.0 trail, BE floor, 75-bar max):
  - After TP1 hit, 50% exits. Runner keeps remaining 50%.
  - trail_stop = highest_since_tp1 - ATR×1.0
  - Floor = break-even (0%)
  - Max hold = 75 bars (25×3.0)

R12 innovations tested:
  G1: Trail width (ATR multiplier 0.7, 1.5, 2.0)
  G2: Step-up floor ratcheting (lock in % of best gain at milestones)
  G3: Multi-phase trailing (loose early → tight after profit milestone)
  G4: Time-based tightening (linear/step ATR decay over runner hold)
  G5: VWMA dynamic floor (VWMA as volume-weighted support level)
  G6: POC dynamic floor (POC as volume-profile support level)
  G7: Gain-adaptive width (wider trail when deeper in profit)
  G8: Runner hold extension (4.0×, 5.0× max hold)
  G9: Best combos

Part A: Runner Diagnostic — analyze baseline runner behavior
Part B: 24 variants — isolated + combo trailing innovations

Data: 6yr (2190d), 50 symbols, exact production logic.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r12.py
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

# VPVR parameters (for POC/VA — needed by U-RR+VWS baseline)
VPVR_LOOKBACK = 672
VPVR_NBINS = 50
VPVR_VA_PCT = 0.70


# ═══════════════════════════════════════════════════════════════════
# Variant Configuration — U-RR+VWS as default baseline
# ═══════════════════════════════════════════════════════════════════
@dataclass
class VConfig:
    """Variant configuration for trailing stop innovation testing.

    Defaults match U-RR+VWS (R11 winner). All trailing params are
    layered on top.
    """
    name: str = 'A-Base'
    desc: str = ''

    # Base exit params (production I-HybTight)
    stop_mult: float = 0.382
    target_retrace: float = 1.5
    max_hold: int = 25

    # U-RR+VWS features (always on for R12)
    rr_size: bool = True
    rr_size_min: float = 0.5
    rr_size_max: float = 2.0
    vwma_stop: str = 'wide'
    vwma_stop_mult: float = 0.50

    # ── PBT ──
    partial_exit: bool = True
    partial_frac: float = 0.5
    be_after_tp1: bool = True

    # ── Runner Trailing (THE FOCUS OF R12) ──
    trail_atr: float = 1.0            # ATR multiplier for runner trail
    runner_mh_mult: float = 3.0       # runner max hold = max_hold × this

    # ── Step-up floor ratcheting ──
    stepup_enabled: bool = False
    stepup_trigger: float = 2.0       # gain/target ratio to activate step-up
    stepup_floor_pct: float = 0.5     # floor = pct × highest_since_tp1

    # ── Multi-phase trailing ──
    multiphase: str = 'off'           # 'off', 'tight_after', 'progressive'
    mp_phase2_trigger: float = 2.0    # gain/target ratio for phase 2
    mp_phase1_atr: float = 1.5        # ATR mult in phase 1 (loose)
    mp_phase2_atr: float = 0.5        # ATR mult in phase 2 (tight)

    # ── Time-based tightening ──
    time_tighten: str = 'off'         # 'off', 'linear', 'step'
    tt_start_atr: float = 1.5         # starting ATR mult (linear)
    tt_end_atr: float = 0.3           # ending ATR mult (linear)
    tt_step_bar: int = 25             # bars post-TP1 for step trigger
    tt_step_atr: float = 0.5          # tighter ATR after step

    # ── VWMA dynamic trail floor ──
    vwma_trail_floor: bool = False    # Use VWMA as dynamic trail floor
    vwma_floor_mult: float = 0.5     # floor = max(0, vwma_gain × mult)

    # ── POC trail floor ──
    poc_trail_floor: str = 'off'      # 'off', 'static', 'dynamic'
    poc_floor_mult: float = 0.3       # floor mult on POC distance

    # ── Gain-adaptive trail width ──
    gain_adaptive: bool = False       # widen trail when deep in profit
    ga_base_atr: float = 0.5         # base ATR multiplier
    ga_gain_mult: float = 0.5        # trail_atr = base + mult × (gain/target)

    # ── Portfolio (constant across all variants) ──
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
# VWMA computation
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def compute_vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume-Weighted Moving Average."""
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
    """Compute rolling Volume Profile: POC, VA High, VA Low."""
    n = len(close)
    poc = np.full(n, np.nan)
    va_high = np.full(n, np.nan)
    va_low = np.full(n, np.nan)

    for i in range(lookback - 1, n):
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
        vol_hist = np.zeros(n_bins)
        for j in range(start, i + 1):
            bar_vol = volume[j]
            if bar_vol <= 0:
                continue
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

        max_vol = -1.0
        poc_bin = 0
        for b in range(n_bins):
            if vol_hist[b] > max_vol:
                max_vol = vol_hist[b]
                poc_bin = b
        poc[i] = p_min + (poc_bin + 0.5) * bin_size

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
# BTC regime
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
# Simulation engine — enhanced runner trailing
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
    """Run portfolio sim with U-RR+VWS baseline + enhanced runner trailing.

    R12 additions vs R11:
    - Step-up floor ratcheting
    - Multi-phase trailing (loose→tight)
    - Time-based tightening
    - VWMA dynamic trail floor
    - POC dynamic trail floor
    - Gain-adaptive trail width
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

            # ═══ STOP COMPUTATION (U-RR+VWS: VWMA wide stop) ═══
            base_stop = pos['move_size'] * cfg.stop_mult

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

            # ═══ TARGET COMPUTATION ═══
            target = pos['move_size'] * cfg.target_retrace
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
                        'target_gain': target,
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0 if cfg.be_after_tp1 else -effective_stop
                    pos['stepup_active'] = False
                elif gain < -effective_stop:
                    _close(balance, trades, pos, sym, gain, lev, 'stop', to_close, i, target)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= max_hold:
                    _close(balance, trades, pos, sym, gain, lev, 'time', to_close, i, target)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        _close(balance, trades, pos, sym, gain, lev, 'trail', to_close, i, target)
                        balance = trades[-1]['_b']

            # ═══ PBT Phase 2: ENHANCED RUNNER (post-TP1) ═══
            elif cfg.partial_exit and pos.get('tp1_hit', False):
                if gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain

                # Read ATR
                atr_val = 0.02
                if sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        atr_val = aa[i]

                bars_since_tp1 = i - pos.get('tp1_bar', pos['entry_bar'])
                target_val = pos['move_size'] * cfg.target_retrace
                gain_ratio = pos['highest_since_tp1'] / target_val if target_val > 0 else 0.0

                # ── Determine effective trail ATR multiplier ──
                trail_mult = cfg.trail_atr

                # Multi-phase: override trail_mult based on profit phase
                if cfg.multiphase == 'tight_after':
                    if gain_ratio >= cfg.mp_phase2_trigger:
                        trail_mult = cfg.mp_phase2_atr
                    else:
                        trail_mult = cfg.mp_phase1_atr
                elif cfg.multiphase == 'progressive':
                    progress = min(gain_ratio / cfg.mp_phase2_trigger, 1.0)
                    trail_mult = cfg.mp_phase1_atr + progress * (cfg.mp_phase2_atr - cfg.mp_phase1_atr)

                # Time-based tightening: override trail_mult based on time
                if cfg.time_tighten == 'linear':
                    runner_duration = cfg.max_hold * (cfg.runner_mh_mult - 1.0)
                    t_progress = min(bars_since_tp1 / max(runner_duration, 1.0), 1.0)
                    trail_mult = cfg.tt_start_atr + t_progress * (cfg.tt_end_atr - cfg.tt_start_atr)
                elif cfg.time_tighten == 'step':
                    if bars_since_tp1 >= cfg.tt_step_bar:
                        trail_mult = cfg.tt_step_atr

                # Gain-adaptive: override trail_mult based on current profit
                if cfg.gain_adaptive:
                    current_ratio = gain / target_val if target_val > 0 else 0.0
                    trail_mult = cfg.ga_base_atr + cfg.ga_gain_mult * max(current_ratio, 0.0)

                # Compute trail stop from highest peak
                trail_stop = pos['highest_since_tp1'] - trail_mult * atr_val

                # ── Determine floor ──
                floor_val = pos.get('trail_floor', 0.0)

                # Step-up floor ratcheting
                if cfg.stepup_enabled:
                    if not pos.get('stepup_active', False):
                        if gain_ratio >= cfg.stepup_trigger:
                            pos['stepup_active'] = True
                    if pos.get('stepup_active', False):
                        new_floor = pos['highest_since_tp1'] * cfg.stepup_floor_pct
                        current_floor = pos.get('trail_floor', 0.0)
                        if new_floor > current_floor:
                            pos['trail_floor'] = new_floor
                    floor_val = pos.get('trail_floor', 0.0)

                # VWMA dynamic floor: use current VWMA as support
                if cfg.vwma_trail_floor and sym_vwma is not None:
                    vwma_arr = sym_vwma.get(sym)
                    if vwma_arr is not None and i < len(vwma_arr):
                        vwma_val = vwma_arr[i]
                        if not np.isnan(vwma_val):
                            if d == 1:
                                vwma_gain = (vwma_val - ep) / ep
                            else:
                                vwma_gain = (ep - vwma_val) / ep
                            vwma_fl = max(0.0, vwma_gain * cfg.vwma_floor_mult)
                            if vwma_fl > floor_val:
                                floor_val = vwma_fl

                # POC trail floor
                if cfg.poc_trail_floor == 'static':
                    poc_dist_entry = pos.get('poc_dist', 0.0)
                    if poc_dist_entry > 0:
                        poc_fl = poc_dist_entry * cfg.poc_floor_mult
                        poc_fl = max(poc_fl, 0.0)
                        if poc_fl > floor_val:
                            floor_val = poc_fl
                elif cfg.poc_trail_floor == 'dynamic' and sym_poc is not None:
                    poc_arr = sym_poc.get(sym)
                    if poc_arr is not None and i < len(poc_arr):
                        poc_val = poc_arr[i]
                        if not np.isnan(poc_val):
                            if d == 1:
                                poc_gain = (poc_val - ep) / ep
                            else:
                                poc_gain = (ep - poc_val) / ep
                            poc_fl = max(0.0, poc_gain * cfg.poc_floor_mult)
                            if poc_fl > floor_val:
                                floor_val = poc_fl

                # Apply floor
                if trail_stop < floor_val:
                    trail_stop = floor_val

                # ── Check exit ──
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
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        exit_reason = 'trail'
                    elif pos['hold_count'] >= max_hold:
                        exit_reason = 'time'
                if exit_reason:
                    _close(balance, trades, pos, sym, gain, lev, exit_reason, to_close, i, target)
                    balance = trades[-1]['_b']

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Entry phase (identical to R11 U-RR+VWS) ═══
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

                # Read POC at entry (for R:R sizing)
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

                # Direction diversity
                if cfg.dir_div:
                    same = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same >= 2:
                        continue

                candidates.append((
                    sym, direction, confidence, move_size,
                    vwma_dist_entry, poc_dist_entry,
                ))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for (sym, direction, confidence, move_size,
                 vwma_dist_e, poc_dist_e) in candidates:
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

                # R:R-gated sizing (from U-RR+VWS)
                if cfg.rr_size and poc_dist_e > 0:
                    implied_target = poc_dist_e
                    implied_stop = move_size * cfg.stop_mult
                    if implied_stop > 0:
                        implied_rr = implied_target / implied_stop
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


def _close(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx, target_gain=0.0):
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

    # ── Runner-specific stats ──
    tp1_trades = [t for t in trades if t['exit_reason'] == 'tp1']
    runner_trades = [t for t in trades if t['exit_reason'] in ('trail_runner', 'time_runner')]
    pre_tp1_trades = [t for t in trades if t['exit_reason'] in ('stop', 'time', 'trail', 'target')]

    runner_stats = {}
    if runner_trades:
        r_wins = [t for t in runner_trades if t['pnl_usd'] > 0]
        r_losses = [t for t in runner_trades if t['pnl_usd'] <= 0]
        r_pnls = [t['pnl_pct'] for t in runner_trades]
        r_highest = [t.get('highest_gain', 0) * EFFECTIVE_LEV * 100 for t in runner_trades if 'highest_gain' in t]
        r_bars = [t.get('runner_bars', 0) for t in runner_trades if 'runner_bars' in t]
        r_be = sum(1 for t in runner_trades if abs(t['pnl_pct']) < 0.5)  # near break-even

        runner_stats = {
            'count': len(runner_trades),
            'win_rate': round(len(r_wins) / len(runner_trades) * 100, 1),
            'avg_pnl_pct': round(float(np.mean(r_pnls)), 3),
            'med_pnl_pct': round(float(np.median(r_pnls)), 3),
            'total_pnl_usd': round(sum(t['pnl_usd'] for t in runner_trades), 2),
            'avg_highest_pct': round(float(np.mean(r_highest)), 3) if r_highest else 0.0,
            'avg_runner_bars': round(float(np.mean(r_bars)), 1) if r_bars else 0.0,
            'pct_near_be': round(r_be / len(runner_trades) * 100, 1),
            'trail_exits': sum(1 for t in runner_trades if t['exit_reason'] == 'trail_runner'),
            'time_exits': sum(1 for t in runner_trades if t['exit_reason'] == 'time_runner'),
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
        'runner': runner_stats,
        'tp1_count': len(tp1_trades),
        'runner_count': len(runner_trades),
        'pre_tp1_count': len(pre_tp1_trades),
    }


# ═══════════════════════════════════════════════════════════════════
# Runner Diagnostic (Part A)
# ═══════════════════════════════════════════════════════════════════
def runner_diagnostic(base_result: dict, w: int = 110):
    """Analyze baseline runner behavior to understand optimization targets."""
    print(f"\n{'═' * w}")
    print(f"  PART A: RUNNER DIAGNOSTIC — U-RR+VWS Baseline")
    print(f"{'═' * w}")

    n_total = base_result['n_trades']
    n_tp1 = base_result.get('tp1_count', 0)
    n_runner = base_result.get('runner_count', 0)
    n_pre = base_result.get('pre_tp1_count', 0)

    print(f"\n  ── Trade Breakdown ──")
    print(f"  Total trades:     {n_total:>6d}")
    print(f"  Pre-TP1 exits:    {n_pre:>6d}  ({n_pre/n_total*100:.1f}%)  [stop/time/trail before TP1]")
    print(f"  TP1 exits:        {n_tp1:>6d}  ({n_tp1/n_total*100:.1f}%)  [partial 50% at target]")
    print(f"  Runner exits:     {n_runner:>6d}  ({n_runner/n_total*100:.1f}%)  [trailing 50% post-TP1]")

    rs = base_result.get('runner', {})
    if rs:
        print(f"\n  ── Runner Performance ──")
        print(f"  Runner WR:           {rs.get('win_rate', 0):>6.1f}%")
        print(f"  Runner avg PnL:      {rs.get('avg_pnl_pct', 0):>6.3f}%")
        print(f"  Runner median PnL:   {rs.get('med_pnl_pct', 0):>6.3f}%")
        print(f"  Runner total $:      {rs.get('total_pnl_usd', 0):>10.1f}")
        print(f"  Avg highest gain:    {rs.get('avg_highest_pct', 0):>6.3f}%  [peak before trail exit]")
        print(f"  Avg runner bars:     {rs.get('avg_runner_bars', 0):>6.1f}")
        print(f"  Near break-even:     {rs.get('pct_near_be', 0):>6.1f}%  [|pnl| < 0.5%]")
        print(f"  Trail exits:         {rs.get('trail_exits', 0):>6d}")
        print(f"  Time exits:          {rs.get('time_exits', 0):>6d}")

        # Runner efficiency: how much of the peak is captured at exit?
        if rs.get('avg_highest_pct', 0) > 0 and rs.get('avg_pnl_pct', 0) != 0:
            capture_eff = rs['avg_pnl_pct'] / rs['avg_highest_pct'] * 100
            giveback = rs['avg_highest_pct'] - rs['avg_pnl_pct']
            print(f"  Capture efficiency:  {capture_eff:>6.1f}%  [exit_pnl / peak_pnl]")
            print(f"  Avg giveback:        {giveback:>6.3f}%  [peak - exit]")

    print(f"\n  ── Key Metrics ──")
    print(f"  Overall R:R:         {base_result.get('rr', 0):>6.2f}  [TARGET: > 1.91]")
    print(f"  Overall WR:          {base_result.get('win_rate', 0):>6.1f}%")
    print(f"  Overall PF:          {base_result.get('profit_factor', 0):>6.2f}")
    print(f"  Calmar:              {base_result.get('calmar', 0):>6.1f}")

    print(f"\n  ── Optimization Targets ──")
    print(f"  1. Reduce near-BE runners → capture more from confirmed direction")
    print(f"  2. Reduce giveback → tighter trail after profit milestone")
    print(f"  3. Extend hold for strong trends → more time for winners to develop")
    print(f"  4. Use VWMA/POC as dynamic floor → volume-weighted S/R support")


# ═══════════════════════════════════════════════════════════════════
# Variant Definitions — 24 variants
# ═══════════════════════════════════════════════════════════════════
def build_variants() -> list[VConfig]:
    v = []

    # ── CONTROL: U-RR+VWS baseline ──
    v.append(VConfig(
        name='A-Base',
        desc='U-RR+VWS baseline (trail ATR×1.0, floor=BE, MH×3)',
    ))

    # ═══ G1: Trail Width (ATR multiplier) ═══
    v.append(VConfig(
        name='B-T07',
        desc='Tighter trail: ATR×0.7 (faster lock-in)',
        trail_atr=0.7,
    ))
    v.append(VConfig(
        name='C-T15',
        desc='Looser trail: ATR×1.5 (more room to breathe)',
        trail_atr=1.5,
    ))
    v.append(VConfig(
        name='D-T20',
        desc='Very loose trail: ATR×2.0 (max room)',
        trail_atr=2.0,
    ))

    # ═══ G2: Step-up Floor Ratcheting ═══
    v.append(VConfig(
        name='E-SU30',
        desc='Step-up: floor=30% of best after 2× target',
        stepup_enabled=True, stepup_trigger=2.0, stepup_floor_pct=0.3,
    ))
    v.append(VConfig(
        name='F-SU50',
        desc='Step-up: floor=50% of best after 2× target',
        stepup_enabled=True, stepup_trigger=2.0, stepup_floor_pct=0.5,
    ))
    v.append(VConfig(
        name='G-SU50E',
        desc='Step-up: floor=50% of best after 1.5× target (early)',
        stepup_enabled=True, stepup_trigger=1.5, stepup_floor_pct=0.5,
    ))

    # ═══ G3: Multi-phase Trailing ═══
    v.append(VConfig(
        name='H-MP15',
        desc='Phase 1→2: ATR 1.5→0.5 at 2× target',
        multiphase='tight_after', mp_phase2_trigger=2.0,
        mp_phase1_atr=1.5, mp_phase2_atr=0.5,
    ))
    v.append(VConfig(
        name='I-MP20',
        desc='Phase 1→2: ATR 2.0→0.3 at 3× target',
        multiphase='tight_after', mp_phase2_trigger=3.0,
        mp_phase1_atr=2.0, mp_phase2_atr=0.3,
    ))
    v.append(VConfig(
        name='J-MPrg',
        desc='Progressive: ATR 2.0→0.3 linear to 3× target',
        multiphase='progressive', mp_phase2_trigger=3.0,
        mp_phase1_atr=2.0, mp_phase2_atr=0.3,
    ))

    # ═══ G4: Time-based Tightening ═══
    v.append(VConfig(
        name='K-TtLn',
        desc='Time linear: ATR 1.5→0.3 over runner hold',
        time_tighten='linear', tt_start_atr=1.5, tt_end_atr=0.3,
    ))
    v.append(VConfig(
        name='L-TtSp',
        desc='Time step: ATR→0.5 after 25 bars post-TP1',
        time_tighten='step', tt_step_bar=25, tt_step_atr=0.5,
    ))

    # ═══ G5: VWMA Dynamic Floor ═══
    v.append(VConfig(
        name='M-VwF5',
        desc='VWMA floor: floor=max(0, vwma_gain×0.5)',
        vwma_trail_floor=True, vwma_floor_mult=0.5,
    ))
    v.append(VConfig(
        name='N-VwF3',
        desc='VWMA floor: floor=max(0, vwma_gain×0.3)',
        vwma_trail_floor=True, vwma_floor_mult=0.3,
    ))

    # ═══ G6: POC Dynamic Floor ═══
    v.append(VConfig(
        name='O-PcSt',
        desc='POC static floor: entry poc_dist×0.3 as floor',
        poc_trail_floor='static', poc_floor_mult=0.3,
    ))
    v.append(VConfig(
        name='P-PcDy',
        desc='POC dynamic floor: live poc_gain×0.3 as floor',
        poc_trail_floor='dynamic', poc_floor_mult=0.3,
    ))

    # ═══ G7: Gain-adaptive Width ═══
    v.append(VConfig(
        name='Q-GAm',
        desc='Gain-adaptive: ATR=0.5+0.5×(gain/target)',
        gain_adaptive=True, ga_base_atr=0.5, ga_gain_mult=0.5,
    ))
    v.append(VConfig(
        name='R-GAa',
        desc='Gain-adaptive: ATR=0.3+1.0×(gain/target) aggressive',
        gain_adaptive=True, ga_base_atr=0.3, ga_gain_mult=1.0,
    ))

    # ═══ G8: Runner Hold Extension ═══
    v.append(VConfig(
        name='S-MH4',
        desc='Runner max hold ×4.0 (100 bars)',
        runner_mh_mult=4.0,
    ))
    v.append(VConfig(
        name='T-MH5',
        desc='Runner max hold ×5.0 (125 bars)',
        runner_mh_mult=5.0,
    ))

    # ═══ G9: Best Combos ═══
    v.append(VConfig(
        name='U-T15S',
        desc='Loose(1.5) + step-up(50% at 2×)',
        trail_atr=1.5,
        stepup_enabled=True, stepup_trigger=2.0, stepup_floor_pct=0.5,
    ))
    v.append(VConfig(
        name='V-MPM4',
        desc='Multi-phase(1.5→0.5) + MH×4.0',
        multiphase='tight_after', mp_phase2_trigger=2.0,
        mp_phase1_atr=1.5, mp_phase2_atr=0.5,
        runner_mh_mult=4.0,
    ))
    v.append(VConfig(
        name='W-VwGA',
        desc='VWMA floor(0.5) + gain-adaptive(0.5+0.5×)',
        vwma_trail_floor=True, vwma_floor_mult=0.5,
        gain_adaptive=True, ga_base_atr=0.5, ga_gain_mult=0.5,
    ))
    v.append(VConfig(
        name='X-Full',
        desc='Loose(1.5)+SU(50%)+VWF(0.5)+MH×4',
        trail_atr=1.5,
        stepup_enabled=True, stepup_trigger=2.0, stepup_floor_pct=0.5,
        vwma_trail_floor=True, vwma_floor_mult=0.5,
        runner_mh_mult=4.0,
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
    print("  ALPHA SEARCH R12: Trailing Stop Innovation — Maximize Runner R:R")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baseline: U-RR+VWS (R11 winner, Calmar 1004, R:R 1.76)")
    print(f"  Target: R:R > 1.91 while maintaining DD ≤ 7%, Calmar ≥ 800")
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

    # ── Compute VPVR (POC) per symbol — needed for R:R sizing ──
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

    # ── Compute BTC regime ──
    print(f"\n🌐 Computing regime...")
    regime = compute_dual_confirm_regime(btc_4h_close, max_bars)

    # ═══════════════════ VARIANT TEST ═══════════════════
    variants = build_variants()
    print(f"\n{'═' * w}")
    print(f"  PART B: TRAILING STOP A/B TEST — {len(variants)} VARIANTS")
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

        # Print runner-specific info for each variant
        rs = res.get('runner', {})
        runner_info = ''
        if rs:
            runner_info = f"  RunRR: {rs.get('avg_pnl_pct', 0):.3f}%  RunBE: {rs.get('pct_near_be', 0):.0f}%"

        print(f"  Ret: {res['total_return_pct']:,.1f}%  DD: {res['max_dd_pct']:.1f}%  "
              f"Sharpe: {res['sharpe']:.2f}  Calmar: {res['calmar']:.1f}  "
              f"WR: {res['win_rate']:.1f}%  PF: {res['profit_factor']:.2f}  R:R: {res['rr']:.2f}  "
              f"#Tr: {res['n_trades']}{runner_info}")

    # ── Part A: Runner diagnostic from baseline ──
    runner_diagnostic(results.get('A-Base', {}), w)

    # ── Summary table ──
    elapsed_total = time.time() - t0
    print(f"\n{'=' * w}")
    print("  RESULTS — Alpha Search R12: Trailing Stop Innovation")
    print(f"{'=' * w}")
    print(f"  {'Variant':<10} {'Desc':<48} {'Ret%':>8} {'DD%':>5} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WR':>5} {'PF':>5} {'R:R':>5} {'#Tr':>6} {'RunPnl':>7} {'RunBE':>5}")
    print(f"  {'─'*10} {'─'*48} {'─'*8} {'─'*5} {'─'*7} {'─'*7} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*7} {'─'*5}")

    base = results.get('A-Base', {})
    base_dd = base.get('max_dd_pct', 100)
    base_sharpe = base.get('sharpe', 0)
    base_ret = base.get('total_return_pct', 0)
    base_rr = base.get('rr', 0)
    base_calmar = base.get('calmar', 0)

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
        if r['calmar'] > base_calmar:
            marks += '●'
        marks = marks.ljust(5)

        rs = r.get('runner', {})
        run_pnl = f"{rs.get('avg_pnl_pct', 0):>6.3f}%" if rs else '    N/A'
        run_be = f"{rs.get('pct_near_be', 0):>4.0f}%" if rs else '  N/A'

        print(f" {marks}{cfg.name:<10} {cfg.desc:<48} "
              f"{r['total_return_pct']:>7,.1f}% {r['max_dd_pct']:>4.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.1f} "
              f"{r['win_rate']:>4.1f}% {r['profit_factor']:>4.2f} {r['rr']:>4.2f} "
              f"{r['n_trades']:>6d} {run_pnl} {run_be}")

    print(f"\n  ★=betterDD  ◆=betterSharpe  ▲=betterRet  ⬆=betterR:R  ●=betterCalmar")
    print(f"  RunPnl=avg runner PnL%  RunBE=% runners near break-even")
    print(f"  Total time: {elapsed_total:.1f}s")

    # ── Winners ──
    best_sharpe = max(results.items(), key=lambda x: x[1].get('sharpe', 0))
    best_dd = min(results.items(), key=lambda x: x[1].get('max_dd_pct', 100))
    best_calmar = max(results.items(), key=lambda x: x[1].get('calmar', 0))
    best_rr = max(results.items(), key=lambda x: x[1].get('rr', 0))
    best_ret = max(results.items(), key=lambda x: x[1].get('total_return_pct', 0))
    best_pf = max(results.items(), key=lambda x: x[1].get('profit_factor', 0))

    print(f"\n  🏆 Best R:R:     {best_rr[0]} ({best_rr[1]['rr']:.2f})")
    print(f"  🏆 Best Calmar:  {best_calmar[0]} ({best_calmar[1]['calmar']:.1f})")
    print(f"  🏆 Best Sharpe:  {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    print(f"  🏆 Best DD:      {best_dd[0]} ({best_dd[1]['max_dd_pct']:.1f}%)")
    print(f"  🏆 Best PF:      {best_pf[0]} ({best_pf[1]['profit_factor']:.2f})")
    print(f"  🏆 Best Return:  {best_ret[0]} ({best_ret[1]['total_return_pct']:,.1f}%)")

    # ── R:R ranking (THE KEY METRIC) ──
    print(f"\n  ── R:R RANKING (vs baseline {base_rr:.2f}) ──")
    rr_sorted = sorted(results.items(), key=lambda x: x[1].get('rr', 0), reverse=True)
    for rank, (name, r) in enumerate(rr_sorted[:10], 1):
        delta = r['rr'] - base_rr
        arrow = '↑' if delta > 0 else '↓' if delta < 0 else '='
        feasible = '✓' if r['max_dd_pct'] <= 7.5 and r['calmar'] >= 700 else '✗'
        print(f"    {rank:>2}. {feasible} {name:<10} R:R={r['rr']:.2f} ({arrow}{abs(delta):.2f})  "
              f"DD={r['max_dd_pct']:.1f}%  Cal={r['calmar']:.1f}  PF={r['profit_factor']:.2f}")

    # ── Runner comparison table ──
    print(f"\n  ── RUNNER COMPARISON ──")
    print(f"  {'Variant':<10} {'R:R':>5} {'RunWR':>6} {'RunAvg':>8} {'RunMed':>8} {'RunBE%':>6} "
          f"{'Trail#':>7} {'Time#':>6} {'AvgBars':>8} {'Giveback':>9}")
    print(f"  {'─'*10} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*6} {'─'*8} {'─'*9}")

    for cfg in variants:
        r = results[cfg.name]
        rs = r.get('runner', {})
        if not rs:
            continue
        giveback = rs.get('avg_highest_pct', 0) - rs.get('avg_pnl_pct', 0)
        print(f"  {cfg.name:<10} {r['rr']:>4.2f} "
              f"{rs.get('win_rate', 0):>5.1f}% "
              f"{rs.get('avg_pnl_pct', 0):>7.3f}% "
              f"{rs.get('med_pnl_pct', 0):>7.3f}% "
              f"{rs.get('pct_near_be', 0):>5.1f}% "
              f"{rs.get('trail_exits', 0):>7d} "
              f"{rs.get('time_exits', 0):>6d} "
              f"{rs.get('avg_runner_bars', 0):>7.1f} "
              f"{giveback:>8.3f}%")

    # ── Save JSON ──
    output = {
        'commit': commit,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'data_days': DATA_DAYS,
        'n_symbols': len(all_data),
        'baseline': 'U-RR+VWS (R11 winner)',
        'focus': 'Trailing stop innovation for R:R maximization',
        'vpvr_compute_time': round(vpvr_elapsed, 1),
        'variants': {},
        'best': {
            'rr': best_rr[0], 'sharpe': best_sharpe[0],
            'dd': best_dd[0], 'calmar': best_calmar[0],
            'ret': best_ret[0], 'pf': best_pf[0],
        },
        'elapsed': round(elapsed_total, 1),
    }

    for k, v_res in results.items():
        output['variants'][k] = {kk: vv for kk, vv in v_res.items() if kk != '_b'}

    out_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r12_results.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {out_path}")


if __name__ == '__main__':
    main()
