#!/usr/bin/env python3
"""
Adaptive Direction A/B Test — BTC 4h SMA Regime-Driven Parameter Switching
===========================================================================
HYPOTHESIS: The long/short performance asymmetry is NOT static — it flips
with BTC macro regime. When BTC is above its 200 SMA on 4h (bull regime),
longs outperform; below it (bear regime), shorts outperform.

Instead of STATIC asymmetric params (which overfit to a vanishing edge),
we use an ADAPTIVE mechanism: at each bar, detect BTC regime and give
the "aligned" direction (long in bull, short in bear) favorable params.

Part A: REGIME DIAGNOSTIC — does BTC 200 SMA predict direction edge?
Part B: ADAPTIVE A/B TEST — 14 variants with regime-switching params

Regime detection: BTC 4h close vs SMA(N) on 4h timeframe.
  - Bull: BTC 4h close > SMA(N) → longs "aligned", shorts "misaligned"
  - Bear: BTC 4h close < SMA(N) → shorts "aligned", longs "misaligned"

Data: 6yr (2190d), 50 symbols, exact production logic.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/adaptive_direction_ab.py
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
# Adaptive Config
# ═══════════════════════════════════════════════════════════════════
@dataclass
class AdaptiveConfig:
    """Regime-adaptive variant configuration.

    "Aligned" = long in bull regime OR short in bear regime.
    "Misaligned" = long in bear regime OR short in bull regime.
    """
    name: str = 'A-Baseline'
    desc: str = ''

    # ── Regime detection ──
    btc_sma_period: int = 200          # SMA period on 4h chart

    # ── Exit params by alignment ──
    aligned_target: float = 1.5        # target_retrace for aligned direction
    misaligned_target: float = 1.5     # target_retrace for misaligned direction
    aligned_stop: float = 0.382        # stop_mult for aligned direction
    misaligned_stop: float = 0.382     # stop_mult for misaligned direction
    aligned_max_hold: int = 25
    misaligned_max_hold: int = 25

    # ── Entry gating ──
    misaligned_conf_boost: int = 0     # require +N more confidence for misaligned
    block_misaligned: bool = False     # completely block misaligned entries

    # ── PBT (constant across all) ──
    partial_exit: bool = True
    partial_fraction: float = 0.5
    trail_atr_mult: float = 1.0
    breakeven_after_tp1: bool = True
    runner_max_hold_mult: float = 3.0
    stop_decay: float = 0.0

    # ── Portfolio (constant across all) ──
    max_slots: int = 2
    dir_diversity: bool = True
    steep_cap: bool = True
    vol_adjusted: bool = True
    soft_equity: bool = True


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
# Entry scanner (production-identical, symmetric confidence gate)
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
# BTC Regime computation — builds 15m-indexed regime array
# ═══════════════════════════════════════════════════════════════════
def compute_btc_regime(btc_4h_close: np.ndarray, sma_period: int, n_15m: int) -> np.ndarray:
    """Compute BTC regime: +1 = bull (BTC > SMA on 4h), -1 = bear.

    Returns an array of length n_15m, aligned to 15m bar indices.
    No look-ahead: bar i uses SMA up to 4h bar i//16.
    """
    n_4h = len(btc_4h_close)
    sma_4h = np.full(n_4h, np.nan)
    s = 0.0
    for j in range(n_4h):
        s += btc_4h_close[j]
        if j >= sma_period:
            s -= btc_4h_close[j - sma_period]
        if j >= sma_period - 1:
            sma_4h[j] = s / sma_period

    regime_4h = np.zeros(n_4h)
    for j in range(n_4h):
        if np.isnan(sma_4h[j]):
            regime_4h[j] = 0  # undefined
        elif btc_4h_close[j] > sma_4h[j]:
            regime_4h[j] = 1   # bull
        else:
            regime_4h[j] = -1  # bear

    # Expand to 15m resolution
    regime_15m = np.zeros(n_15m)
    for i in range(n_15m):
        j = min(i // 16, n_4h - 1)
        regime_15m[i] = regime_4h[j]
    return regime_15m


# ═══════════════════════════════════════════════════════════════════
# Adaptive simulator — regime-aware parameter switching
# ═══════════════════════════════════════════════════════════════════
def simulate(
    sym_close, sym_high, sym_low,
    sym_dirs, sym_confs, sym_moves,
    cfg: AdaptiveConfig,
    regime: np.ndarray,
    sym_atr=None,
) -> dict:
    """Run portfolio sim with regime-adaptive parameters.

    At each bar, regime[i] = +1 (bull) or -1 (bear).
    For direction d: aligned = (d == regime[i]).
    Aligned positions get favorable params (wider stops, etc).
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
        r = regime[i]  # current regime: +1 bull, -1 bear, 0 undefined

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            ep = pos['entry_price']
            lev = pos['leverage']
            d = pos['direction']

            if d == 1:
                gain = (c - ep) / ep
            else:
                gain = (ep - c) / ep

            # Regime-adaptive parameters — use regime at ENTRY time
            entry_regime = pos.get('entry_regime', 0)
            aligned = (d == entry_regime) if entry_regime != 0 else True
            if aligned:
                target_retrace = cfg.aligned_target
                stop_mult = cfg.aligned_stop
                max_hold = cfg.aligned_max_hold
            else:
                target_retrace = cfg.misaligned_target
                stop_mult = cfg.misaligned_stop
                max_hold = cfg.misaligned_max_hold

            target = pos['move_size'] * target_retrace
            time_factor = 1.0 - cfg.stop_decay * (pos['hold_count'] / max_hold)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * stop_mult * time_factor

            # ═══ PBT Phase 1 ═══
            if cfg.partial_exit and not pos.get('tp1_hit', False):
                if gain >= target:
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain
                    tp1_not = pos['notional'] * cfg.partial_fraction
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
                        'aligned': aligned,
                        'entry_regime': entry_regime,
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0 if cfg.breakeven_after_tp1 else -effective_stop
                elif gain < -effective_stop:
                    _close(balance, trades, pos, sym, gain, lev, 'stop', to_close, i, aligned, entry_regime)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= max_hold:
                    _close(balance, trades, pos, sym, gain, lev, 'time', to_close, i, aligned, entry_regime)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        _close(balance, trades, pos, sym, gain, lev, 'trail', to_close, i, aligned, entry_regime)
                        balance = trades[-1]['_b']

            # ═══ PBT Phase 2 (runner) ═══
            elif cfg.partial_exit and pos.get('tp1_hit', False):
                if gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain

                atr_val = 0.02
                if sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        atr_val = aa[i]

                trail_stop = pos['highest_since_tp1'] - cfg.trail_atr_mult * atr_val
                floor_val = pos.get('trail_floor', -effective_stop)
                if trail_stop < floor_val:
                    trail_stop = floor_val

                exit_runner = False
                exit_reason = None

                if gain <= trail_stop:
                    exit_runner = True; exit_reason = 'trail_runner'
                elif pos['hold_count'] >= max_hold * cfg.runner_max_hold_mult:
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
                        'aligned': aligned,
                        'entry_regime': entry_regime,
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
                    _close(balance, trades, pos, sym, gain, lev, exit_reason, to_close, i, aligned, entry_regime)
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

                # Regime-aware entry gating
                is_aligned = (direction == r) if r != 0 else True

                if cfg.block_misaligned and not is_aligned:
                    continue

                # Misaligned entries need extra confidence
                eff_min_conf = MIN_CONFIDENCE
                if not is_aligned:
                    eff_min_conf += cfg.misaligned_conf_boost
                if confidence < eff_min_conf:
                    continue

                if cfg.dir_diversity:
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

                if cfg.vol_adjusted and sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]) and aa[i] > 0:
                        vf = max(0.4, min(0.01 / aa[i], 2.0))
                        per_slot *= vf
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if cfg.soft_equity:
                    eq_now = equity_curve[i-1] if i > 0 else INITIAL_CAPITAL
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    for dd_thr, scale in SOFT_EQ:
                        if dd_pct > dd_thr:
                            per_slot *= scale
                            break

                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost

                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': sym_close[sym][i],
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': EFFECTIVE_LEV,
                    'entry_regime': int(r),  # lock regime at entry
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


def _close(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx, aligned, entry_regime):
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
        'aligned': aligned,
        'entry_regime': entry_regime,
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
    aligned_t = [t for t in trades if t.get('aligned', True)]
    misaligned_t = [t for t in trades if not t.get('aligned', True)]

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

    reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        reasons[r] = reasons.get(r, 0) + 1

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
        'aligned': _dir_stats(aligned_t),
        'misaligned': _dir_stats(misaligned_t),
        'exit_reasons': reasons,
    }


# ═══════════════════════════════════════════════════════════════════
# Part A: Regime Diagnostic
# ═══════════════════════════════════════════════════════════════════
def regime_diagnostic(trades: list[dict], regime: np.ndarray, n_bars: int, sma_period: int):
    """Analyze baseline trades tagged by BTC regime alignment."""
    w = 110
    print(f"\n{'═' * w}")
    print(f"  PART A: REGIME DIAGNOSTIC — BTC 4h SMA({sma_period})")
    print(f"{'═' * w}")

    # Time in each regime
    bull_bars = int(np.sum(regime[:n_bars] > 0))
    bear_bars = int(np.sum(regime[:n_bars] < 0))
    undef_bars = int(np.sum(regime[:n_bars] == 0))
    total = bull_bars + bear_bars + undef_bars
    print(f"\n  Time in Bull: {bull_bars/total*100:.1f}%  |  Bear: {bear_bars/total*100:.1f}%  |  Undefined: {undef_bars/total*100:.1f}%")

    # Tag trades by regime at entry
    bull_long = [t for t in trades if t.get('entry_regime', 0) > 0 and t['direction'] == 1]
    bull_short = [t for t in trades if t.get('entry_regime', 0) > 0 and t['direction'] == -1]
    bear_long = [t for t in trades if t.get('entry_regime', 0) < 0 and t['direction'] == 1]
    bear_short = [t for t in trades if t.get('entry_regime', 0) < 0 and t['direction'] == -1]

    aligned = bull_long + bear_short
    misaligned = bull_short + bear_long

    def _s(tl, label):
        if not tl:
            return {'label': label, 'n': 0, 'wr': 0, 'pf': 0, 'rr': 0, 'ev': 0, 'total': 0}
        n = len(tl)
        w_ = [t for t in tl if t['pnl_usd'] > 0]
        l_ = [t for t in tl if t['pnl_usd'] <= 0]
        wr = len(w_) / n * 100
        gp = sum(t['pnl_usd'] for t in w_)
        gl = sum(abs(t['pnl_usd']) for t in l_)
        pf = gp / gl if gl > 0 else 999.0
        aw = float(np.mean([t['pnl_pct'] for t in w_])) if w_ else 0
        al = float(np.mean([abs(t['pnl_pct']) for t in l_])) if l_ else 1
        rr = aw / al if al > 0 else 999.0
        ev = float(np.mean([t['pnl_pct'] for t in tl]))
        total = sum(t['pnl_usd'] for t in tl)
        return {'label': label, 'n': n, 'wr': round(wr, 1), 'pf': round(pf, 2),
                'rr': round(rr, 2), 'ev': round(ev, 3), 'total': round(total, 2)}

    bl = _s(bull_long, 'Bull-Long')
    bs = _s(bull_short, 'Bull-Short')
    brl = _s(bear_long, 'Bear-Long')
    brs = _s(bear_short, 'Bear-Short')
    al_s = _s(aligned, 'ALIGNED')
    mis_s = _s(misaligned, 'MISALIGNED')

    print(f"\n  {'Category':<18} {'#Trades':>8} {'WR%':>7} {'PF':>6} {'R:R':>6} {'EV%':>8} {'Total$':>10}")
    print(f"  {'─'*18} {'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")
    for s in [bl, bs, brl, brs]:
        print(f"  {s['label']:<18} {s['n']:>8d} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['rr']:>5.2f} {s['ev']:>7.3f}% {s['total']:>10.2f}")
    print(f"  {'─'*18} {'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")
    for s in [al_s, mis_s]:
        print(f"  {s['label']:<18} {s['n']:>8d} {s['wr']:>6.1f}% {s['pf']:>5.2f} {s['rr']:>5.2f} {s['ev']:>7.3f}% {s['total']:>10.2f}")

    al_ev, mis_ev = al_s['ev'], mis_s['ev']
    gap = al_ev - mis_ev
    pct_gap = gap / abs(mis_ev) * 100 if mis_ev != 0 else 0
    print(f"\n  ⚡ Aligned − Misaligned gap: EV {gap:+.3f}% ({pct_gap:+.1f}%), "
          f"WR {al_s['wr']-mis_s['wr']:+.1f}pp, PF {al_s['pf']-mis_s['pf']:+.2f}, "
          f"R:R {al_s['rr']-mis_s['rr']:+.2f}")

    signal = "STRONG" if abs(gap) > 0.5 else "MODERATE" if abs(gap) > 0.2 else "WEAK"
    direction = "ALIGNED WINS" if gap > 0 else "MISALIGNED WINS" if gap < 0 else "NEUTRAL"
    print(f"  ⚡ Regime signal strength: {signal} — {direction}")

    # Yearly regime breakdown
    print(f"\n  {'Year':<18} {'Bull%':>7} {'L(bull)':>10} {'S(bull)':>10} {'L(bear)':>10} {'S(bear)':>10} {'Al EV':>8} {'Mis EV':>8}")
    print(f"  {'─'*18} {'─'*7} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for yr in range(6):
        bar_start = yr * BARS_PER_YEAR
        bar_end = (yr + 1) * BARS_PER_YEAR
        yr_trades = [t for t in trades if bar_start <= t.get('entry_bar', 0) < bar_end]
        if not yr_trades:
            continue

        yr_bull = int(np.sum(regime[bar_start:min(bar_end, n_bars)] > 0))
        yr_total = min(bar_end, n_bars) - bar_start
        yr_bull_pct = yr_bull / yr_total * 100 if yr_total > 0 else 0

        yr_bl = [t for t in yr_trades if t.get('entry_regime', 0) > 0 and t['direction'] == 1]
        yr_bs = [t for t in yr_trades if t.get('entry_regime', 0) > 0 and t['direction'] == -1]
        yr_rl = [t for t in yr_trades if t.get('entry_regime', 0) < 0 and t['direction'] == 1]
        yr_rs = [t for t in yr_trades if t.get('entry_regime', 0) < 0 and t['direction'] == -1]
        yr_aligned = yr_bl + yr_rs
        yr_misaligned = yr_bs + yr_rl

        def _ev(tl):
            return round(float(np.mean([t['pnl_pct'] for t in tl])), 3) if tl else 0.0

        def _wr(tl):
            return round(len([t for t in tl if t['pnl_usd'] > 0]) / len(tl) * 100, 1) if tl else 0.0

        label = f"Yr{yr+1} ({2020+yr}-{2021+yr})"
        print(f"  {label:<18} {yr_bull_pct:>6.1f}% "
              f"{len(yr_bl):>4d}|{_wr(yr_bl):>4.0f}% "
              f"{len(yr_bs):>4d}|{_wr(yr_bs):>4.0f}% "
              f"{len(yr_rl):>4d}|{_wr(yr_rl):>4.0f}% "
              f"{len(yr_rs):>4d}|{_wr(yr_rs):>4.0f}% "
              f"{_ev(yr_aligned):>7.3f}% {_ev(yr_misaligned):>7.3f}%")

    return {'aligned': al_s, 'misaligned': mis_s, 'gap_ev': gap}


# ═══════════════════════════════════════════════════════════════════
# Part B: Variant Definitions
# ═══════════════════════════════════════════════════════════════════
def build_variants() -> list[AdaptiveConfig]:
    """Build adaptive A/B variants."""
    v = []

    # ── CONTROL ──
    v.append(AdaptiveConfig(
        name='A-Baseline',
        desc='Symmetric I-HybTight (prod, no adaptation)',
    ))

    # ═══ GROUP 1: Stop Adaptation (the strongest single lever from R1/R2) ═══

    v.append(AdaptiveConfig(
        name='B-Stop200',
        desc='SMA(200) 4h: aligned .45 / misaligned .30 stop',
        btc_sma_period=200,
        aligned_stop=0.45, misaligned_stop=0.30,
    ))

    v.append(AdaptiveConfig(
        name='C-Stop100',
        desc='SMA(100) 4h: aligned .45 / misaligned .30 stop',
        btc_sma_period=100,
        aligned_stop=0.45, misaligned_stop=0.30,
    ))

    v.append(AdaptiveConfig(
        name='D-Stop50',
        desc='SMA(50) 4h: aligned .45 / misaligned .30 stop',
        btc_sma_period=50,
        aligned_stop=0.45, misaligned_stop=0.30,
    ))

    # ═══ GROUP 2: Target Adaptation ═══

    v.append(AdaptiveConfig(
        name='E-Tgt200',
        desc='SMA(200) 4h: aligned 2.0 / misaligned 1.0 target',
        btc_sma_period=200,
        aligned_target=2.0, misaligned_target=1.0,
    ))

    # ═══ GROUP 3: Stop + Target Combined ═══

    v.append(AdaptiveConfig(
        name='F-StopTgt200',
        desc='SMA(200) stop(.45/.30) + target(2.0/1.0)',
        btc_sma_period=200,
        aligned_stop=0.45, misaligned_stop=0.30,
        aligned_target=2.0, misaligned_target=1.0,
    ))

    v.append(AdaptiveConfig(
        name='G-StopTgt100',
        desc='SMA(100) stop(.45/.30) + target(2.0/1.0)',
        btc_sma_period=100,
        aligned_stop=0.45, misaligned_stop=0.30,
        aligned_target=2.0, misaligned_target=1.0,
    ))

    # ═══ GROUP 4: Max Hold Adaptation ═══

    v.append(AdaptiveConfig(
        name='H-MH200',
        desc='SMA(200) 4h: aligned MH30 / misaligned MH20',
        btc_sma_period=200,
        aligned_max_hold=30, misaligned_max_hold=20,
    ))

    # ═══ GROUP 5: Full Adaptive (stop + target + MH) ═══

    v.append(AdaptiveConfig(
        name='I-Full200',
        desc='SMA(200) stop+tgt+MH adaptive',
        btc_sma_period=200,
        aligned_stop=0.45, misaligned_stop=0.30,
        aligned_target=2.0, misaligned_target=1.0,
        aligned_max_hold=30, misaligned_max_hold=20,
    ))

    v.append(AdaptiveConfig(
        name='J-Full100',
        desc='SMA(100) stop+tgt+MH adaptive',
        btc_sma_period=100,
        aligned_stop=0.45, misaligned_stop=0.30,
        aligned_target=2.0, misaligned_target=1.0,
        aligned_max_hold=30, misaligned_max_hold=20,
    ))

    # ═══ GROUP 6: Entry Gating ═══

    v.append(AdaptiveConfig(
        name='K-ConfGate200',
        desc='SMA(200) misaligned needs +1 extra conf',
        btc_sma_period=200,
        misaligned_conf_boost=1,
    ))

    v.append(AdaptiveConfig(
        name='L-Block200',
        desc='SMA(200) block misaligned entries entirely',
        btc_sma_period=200,
        block_misaligned=True,
    ))

    # ═══ GROUP 7: Moderate / Conservative ═══

    v.append(AdaptiveConfig(
        name='M-Mild200',
        desc='SMA(200) mild stop: aligned .40 / misaligned .35',
        btc_sma_period=200,
        aligned_stop=0.40, misaligned_stop=0.35,
    ))

    v.append(AdaptiveConfig(
        name='N-MildFull200',
        desc='SMA(200) mild stop(.40/.35) + tgt(1.8/1.2)',
        btc_sma_period=200,
        aligned_stop=0.40, misaligned_stop=0.35,
        aligned_target=1.8, misaligned_target=1.2,
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
    print("  ADAPTIVE DIRECTION A/B TEST — BTC 4h SMA Regime-Driven")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baseline: v12.4 I-HybTight (symmetric T150_S38_H25_PBT)")
    print(f"  Regime: BTC 4h close vs SMA(N) — bull=longs aligned, bear=shorts aligned")
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

    # ── Scan entries (once, shared by all variants with same conf gate) ──
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
    print(f"  Left-padded to {max_bars} bars ({max_bars//32} days)")
    print(f"  Raw signals: {raw_signals:,}")

    # ── Pre-compute regime arrays for all SMA periods we'll test ──
    sma_periods = sorted(set(v.btc_sma_period for v in build_variants()))
    regime_cache: dict[int, np.ndarray] = {}
    for sp in sma_periods:
        regime_cache[sp] = compute_btc_regime(btc_4h_close, sp, max_bars)
        bull_pct = np.sum(regime_cache[sp][:max_bars] > 0) / max_bars * 100
        print(f"  BTC SMA({sp}) regime: {bull_pct:.1f}% bull / {100-bull_pct:.1f}% bear")

    # ═══════════════════ PART A: DIAGNOSTIC ═══════════════════
    print(f"\n🔬 Running baseline simulation for diagnostic...")
    baseline_cfg = AdaptiveConfig(name='A-Baseline', desc='Symmetric I-HybTight (prod)')
    regime_200 = regime_cache[200]
    baseline_res = simulate(
        sym_close_d, sym_high_d, sym_low_d,
        sym_dirs, sym_confs, sym_moves,
        baseline_cfg, regime_200, sym_atr_d,
    )

    # Extract trade list for diagnostic
    # We need to re-run with a modified simulate that returns trades
    # Actually, _compute_stats already computed aligned/misaligned stats from trades
    # But we need the raw trades for the detailed diagnostic.
    # Let's do a small hack: re-run simulate just to get the trade list.
    baseline_trades = _get_trades(
        sym_close_d, sym_high_d, sym_low_d,
        sym_dirs, sym_confs, sym_moves,
        baseline_cfg, regime_200, sym_atr_d,
    )

    # Run diagnostic for SMA(200) — the main one
    diag = regime_diagnostic(baseline_trades, regime_200, max_bars, 200)

    # Also show SMA(100) and SMA(50) diagnostic
    for sp in [100, 50]:
        if sp in regime_cache:
            r_sp = regime_cache[sp]
            trades_sp = _get_trades(
                sym_close_d, sym_high_d, sym_low_d,
                sym_dirs, sym_confs, sym_moves,
                AdaptiveConfig(btc_sma_period=sp), r_sp, sym_atr_d,
            )
            regime_diagnostic(trades_sp, r_sp, max_bars, sp)

    # ═══════════════════ PART B: A/B TEST ═══════════════════
    print(f"\n{'═' * w}")
    print(f"  PART B: ADAPTIVE A/B TEST — {len(build_variants())} VARIANTS")
    print(f"{'═' * w}")

    variants = build_variants()
    results = {}

    for vi, cfg in enumerate(variants):
        t_v = time.time()
        print(f"\n🔬 [{vi+1}/{len(variants)}] {cfg.name}: {cfg.desc}")

        regime = regime_cache[cfg.btc_sma_period]
        res = simulate(
            sym_close_d, sym_high_d, sym_low_d,
            sym_dirs, sym_confs, sym_moves,
            cfg, regime, sym_atr_d,
        )

        elapsed_v = time.time() - t_v
        res['elapsed'] = round(elapsed_v, 1)
        res['config'] = {
            'btc_sma': cfg.btc_sma_period,
            'al_stop': cfg.aligned_stop, 'mis_stop': cfg.misaligned_stop,
            'al_tgt': cfg.aligned_target, 'mis_tgt': cfg.misaligned_target,
            'al_mh': cfg.aligned_max_hold, 'mis_mh': cfg.misaligned_max_hold,
            'mis_conf_boost': cfg.misaligned_conf_boost,
            'block_mis': cfg.block_misaligned,
        }
        results[cfg.name] = res

        l = res.get('long', {})
        s = res.get('short', {})
        al = res.get('aligned', {})
        mis = res.get('misaligned', {})
        print(f"  Ret: {res['total_return_pct']:,.1f}%  DD: {res['max_dd_pct']:.1f}%  "
              f"Sharpe: {res['sharpe']:.2f}  Calmar: {res['calmar']:.1f}  "
              f"WR: {res['win_rate']:.1f}%  PF: {res['profit_factor']:.2f}  R:R: {res['rr']:.2f}")
        print(f"  ALIGNED: {al.get('count',0)} tr, WR {al.get('wr',0):.1f}%, EV {al.get('avg_pnl',0):.3f}%, PF {al.get('pf',0):.2f}")
        print(f"  MISALIGNED: {mis.get('count',0)} tr, WR {mis.get('wr',0):.1f}%, EV {mis.get('avg_pnl',0):.3f}%, PF {mis.get('pf',0):.2f}")

    # ── Summary table ──
    elapsed_total = time.time() - t0
    print(f"\n{'=' * w}")
    print("  RESULTS COMPARISON — Adaptive Direction A/B Test")
    print(f"{'=' * w}")
    print(f"  {'Variant':<16} {'Desc':<46} {'Ret%':>8} {'DD%':>5} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WR':>5} {'PF':>5} {'R:R':>5} {'#Tr':>6}")
    print(f"  {'─'*16} {'─'*46} {'─'*8} {'─'*5} {'─'*7} {'─'*7} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*6}")

    base = results.get('A-Baseline', {})
    base_dd = base.get('max_dd_pct', 100)
    base_sharpe = base.get('sharpe', 0)
    base_calmar = base.get('calmar', 0)
    base_ret = base.get('total_return_pct', 0)

    for cfg in variants:
        r = results[cfg.name]
        dd_mark = '★' if r['max_dd_pct'] < base_dd else ' '
        sh_mark = '★' if r['sharpe'] > base_sharpe else ' '
        print(f" {dd_mark}{sh_mark}{cfg.name:<15} {cfg.desc:<46} "
              f"{r['total_return_pct']:>7,.1f}% {r['max_dd_pct']:>4.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.1f} "
              f"{r['win_rate']:>4.1f}% {r['profit_factor']:>4.2f} {r['rr']:>4.2f} {r['n_trades']:>6d}")

    # Aligned/Misaligned detail table
    print(f"\n  {'Variant':<16} {'Al#':>5} {'Al_WR':>6} {'Al_EV':>7} {'Al_PF':>6} "
          f"{'Mis#':>5} {'Mis_WR':>7} {'Mis_EV':>8} {'Mis_PF':>7} {'Gap_EV':>8}")
    print(f"  {'─'*16} {'─'*5} {'─'*6} {'─'*7} {'─'*6} {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*8}")
    for cfg in variants:
        r = results[cfg.name]
        al = r.get('aligned', {})
        mis = r.get('misaligned', {})
        gap = al.get('avg_pnl', 0) - mis.get('avg_pnl', 0)
        print(f"  {cfg.name:<16} "
              f"{al.get('count',0):>5d} {al.get('wr',0):>5.1f}% {al.get('avg_pnl',0):>6.3f}% {al.get('pf',0):>5.2f} "
              f"{mis.get('count',0):>5d} {mis.get('wr',0):>6.1f}% {mis.get('avg_pnl',0):>7.3f}% {mis.get('pf',0):>6.2f} "
              f"{gap:>+7.3f}%")

    print(f"\n  ★ = better than baseline on that metric")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"{'=' * w}")

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
        'diagnostic': {
            'sma200': {k: v for k, v in diag.items() if k != 'gap_ev'} | {'gap_ev': diag['gap_ev']},
        },
        'variants': {k: {kk: vv for kk, vv in v.items() if kk != '_b'} for k, v in results.items()},
        'best': {
            'sharpe': best_sharpe[0], 'dd': best_dd[0],
            'calmar': best_calmar[0], 'rr': best_rr[0], 'ret': best_ret[0],
        },
        'elapsed': round(elapsed_total, 1),
    }

    out_path = Path(__file__).parent.parent / 'logs' / 'adaptive_direction_ab_results.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {out_path}")


def _get_trades(sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves,
                cfg, regime, sym_atr) -> list[dict]:
    """Run simulation and extract raw trade list (for diagnostic)."""
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

            entry_regime = pos.get('entry_regime', 0)
            aligned = (d == entry_regime) if entry_regime != 0 else True
            target_retrace = cfg.aligned_target if aligned else cfg.misaligned_target
            stop_mult = cfg.aligned_stop if aligned else cfg.misaligned_stop
            max_hold = cfg.aligned_max_hold if aligned else cfg.misaligned_max_hold

            target = pos['move_size'] * target_retrace
            effective_stop = pos['move_size'] * stop_mult

            if cfg.partial_exit and not pos.get('tp1_hit', False):
                if gain >= target:
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain
                    tp1_not = pos['notional'] * cfg.partial_fraction
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
                        'aligned': aligned, 'entry_regime': entry_regime,
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0 if cfg.breakeven_after_tp1 else -effective_stop
                elif gain < -effective_stop:
                    _close(balance, trades, pos, sym, gain, lev, 'stop', to_close, i, aligned, entry_regime)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= max_hold:
                    _close(balance, trades, pos, sym, gain, lev, 'time', to_close, i, aligned, entry_regime)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        _close(balance, trades, pos, sym, gain, lev, 'trail', to_close, i, aligned, entry_regime)
                        balance = trades[-1]['_b']
            elif cfg.partial_exit and pos.get('tp1_hit', False):
                if gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain
                atr_val = 0.02
                if sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                        atr_val = aa[i]
                trail_stop = pos['highest_since_tp1'] - cfg.trail_atr_mult * atr_val
                floor_val = pos.get('trail_floor', -effective_stop)
                if trail_stop < floor_val:
                    trail_stop = floor_val
                exit_runner = False; exit_reason = None
                if gain <= trail_stop:
                    exit_runner = True; exit_reason = 'trail_runner'
                elif pos['hold_count'] >= max_hold * cfg.runner_max_hold_mult:
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
                        'aligned': aligned, 'entry_regime': entry_regime,
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
                is_aligned = (direction == r) if r != 0 else True
                if cfg.block_misaligned and not is_aligned:
                    continue
                eff_min_conf = MIN_CONFIDENCE + (cfg.misaligned_conf_boost if not is_aligned else 0)
                if confidence < eff_min_conf:
                    continue
                if cfg.dir_diversity:
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
                if cfg.vol_adjusted and sym_atr is not None:
                    aa = sym_atr.get(sym)
                    if aa is not None and i < len(aa) and not np.isnan(aa[i]) and aa[i] > 0:
                        vf = max(0.4, min(0.01 / aa[i], 2.0))
                        per_slot *= vf
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if cfg.soft_equity:
                    eq_now = equity_curve[i-1] if i > 0 else INITIAL_CAPITAL
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    for dd_thr, scale in SOFT_EQ:
                        if dd_pct > dd_thr:
                            per_slot *= scale
                            break
                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost
                positions[sym] = {
                    'direction': direction, 'confidence': confidence,
                    'move_size': move_size, 'entry_price': sym_close[sym][i],
                    'entry_bar': i, 'hold_count': 0, 'best_pnl': 0.0,
                    'notional': per_slot, 'leverage': EFFECTIVE_LEV,
                    'entry_regime': int(r),
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


if __name__ == '__main__':
    main()
