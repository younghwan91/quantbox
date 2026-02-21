#!/usr/bin/env python3
"""
Reversal A/B Test — MR vs Semi-Reversal vs Full Reversal
=========================================================
Tests whether converting QuantBox from pure mean-reversion to
a semi-reversal or full reversal strategy improves R:R.

Strategy spectrum tested:
  A) MR-Current:   target_retrace=1.0, stop=0.382, max_hold=20 (production)
  B) MR-Wide:      target_retrace=1.5, stop=0.382, max_hold=30 (wider target)
  C) Semi-Rev:     target_retrace=2.0, stop=0.5, max_hold=40, PBT trail
  D) Semi-Rev-Wide: target_retrace=3.0, stop=0.5, max_hold=50, PBT trail
  E) Rev-ATR:      No fixed target, ATR trailing only, max_hold=60
  F) Rev-ATR-Wide: No fixed target, 2.5×ATR trail, max_hold=80
  G) MR→Rev Hybrid: TP1 at 1.0× (MR), then runner targets 2-3× (reversal)
  H) Momentum:     Entry WITH the move (direction flip), wider targets

All share the same confidence-gated entry signals from production.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/reversal_ab_test.py
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

# ═══════════════════════════════════════════════════════════════════
# Production parameters (v12.3.3)
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032
TARGET_RETRACE = 1.0
STOP_MULT = 0.382
MAX_HOLD = 20
STOP_DECAY = 0.0
BTC_MOM_PERIOD = 48
BTC_REGIME_PERIOD = 96
ASSET_TREND_PERIOD = 192
MIN_CONFIDENCE = 3
BASE_LEVERAGE = 1.5
MTF_LEV_MULT = 2.5
EFFECTIVE_LEV = BASE_LEVERAGE * MTF_LEV_MULT  # 3.75
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


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _find_cache(symbol: str, tf: str, days: int) -> Path | None:
    pattern = f"{symbol}_{tf}_{days}d_*.npz"
    matches = sorted(CACHE_DIR.glob(pattern))
    if matches:
        return matches[-1]
    pattern2 = f"{symbol}_{tf}_*.npz"
    matches2 = sorted(CACHE_DIR.glob(pattern2))
    return matches2[-1] if matches2 else None


def load_15m(symbol: str, days: int = 2190) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None:
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int = 2190) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None:
        return None
    return np.load(cache)['close']


# ═══════════════════════════════════════════════════════════════════
# Entry Scanner (reuse from R5 — same production signals)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_entries(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period, btc_regime_period,
    asset_trend_period, min_confidence, rsi_period, vol_period,
    stoch_enabled, stoch_k_period, stoch_d_period,
    macd_enabled, macd_fast, macd_slow, macd_signal_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
):
    """Scan ALL entry opportunities without position tracking."""
    n = min(len(close), len(btc_close))
    dirs = np.zeros(n)
    confs = np.zeros(n)
    moves = np.zeros(n)

    btc_sma = _compute_sma_inline(btc_close[:n], btc_regime_period)
    asset_sma = _compute_sma_inline(close[:n], asset_trend_period)
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
                if rsi[i] < 35:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] < 0.42:
                    confidence += 1
                if btc_mom < -0.01:
                    confidence += 1
                if stoch_enabled and i > 0:
                    if (stoch_k[i] < 20 and stoch_k[i] > stoch_d[i]
                            and stoch_k[i - 1] <= stoch_d[i - 1]):
                        confidence += 1
                if macd_enabled and i > 0:
                    if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                        confidence += 1

            elif ret > thr and btc_mom > 0:
                direction = -1
                if rsi[i] > 65:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] > 0.58:
                    confidence += 1
                if btc_mom > 0.01:
                    confidence += 1
                if stoch_enabled and i > 0:
                    if (stoch_k[i] > 80 and stoch_k[i] < stoch_d[i]
                            and stoch_k[i - 1] >= stoch_d[i - 1]):
                        confidence += 1
                if macd_enabled and i > 0:
                    if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                        confidence += 1

        if direction != 0 and confidence >= min_confidence:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            moves[i] = abs(ret)

    return dirs, confs, moves


# ═══════════════════════════════════════════════════════════════════
# Momentum entry scanner — SAME direction as the move
# (Dropped price → SHORT expecting continuation; Pumped → LONG)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_momentum_entries(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period, rsi_period, vol_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
):
    """Momentum continuation entries — trade WITH the large move.

    If price dropped >threshold → SHORT (expect further drop / trend continuation)
    If price pumped >threshold → LONG (expect further pump)
    Gate: BTC must agree with direction.
    """
    n = min(len(close), len(btc_close))
    dirs = np.zeros(n)
    confs = np.zeros(n)
    moves = np.zeros(n)

    rsi = _compute_rsi_inline(close[:n], rsi_period)
    vol_sma = _compute_sma_inline(volume[:n], vol_period)

    taker_ratio = np.full(n, 0.5)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy[i] / volume[i]

    if atr_period > 0:
        atr_pct = compute_atr_pct(high[:n], low[:n], close[:n], atr_period)
    else:
        atr_pct = np.full(n, np.nan)

    start_bar = max(lookback, btc_mom_period, rsi_period, vol_period)
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
            # MOMENTUM: trade WITH the move (opposite of MR)
            if ret < -thr and btc_mom < 0:
                # Price dropped AND BTC weak → SHORT (continue drop)
                direction = -1
                if rsi[i] < 30:
                    confidence += 1  # deeply oversold = strong momentum
                if not np.isnan(vol_sma[i]) and volume[i] > 2.0 * vol_sma[i]:
                    confidence += 1  # high volume confirms
                if taker_ratio[i] < 0.40:
                    confidence += 1  # aggressive sellers
                if btc_mom < -0.02:
                    confidence += 1  # strong BTC downtrend

            elif ret > thr and btc_mom > 0:
                # Price pumped AND BTC strong → LONG (continue pump)
                direction = 1
                if rsi[i] > 70:
                    confidence += 1  # overbought = strong momentum
                if not np.isnan(vol_sma[i]) and volume[i] > 2.0 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] > 0.60:
                    confidence += 1  # aggressive buyers
                if btc_mom > 0.02:
                    confidence += 1

        if direction != 0 and confidence >= 2:
            dirs[i] = float(direction)
            confs[i] = float(confidence)
            moves[i] = abs(ret)

    return dirs, confs, moves


def apply_mtf_filter(
    dirs, confs, moves, close_4h, sma_period=12, ratio=16,
):
    """Zero out entries that go against the 4h trend."""
    n_4h = len(close_4h)
    sma_4h = np.full(n_4h, np.nan)
    s = 0.0
    for j in range(n_4h):
        s += close_4h[j]
        if j >= sma_period:
            s -= close_4h[j - sma_period]
        if j >= sma_period - 1:
            sma_4h[j] = s / sma_period

    od = dirs.copy()
    oc = confs.copy()
    om = moves.copy()

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
# Variant Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class RevConfig:
    """Reversal A/B test variant configuration."""
    name: str = 'Baseline'
    desc: str = ''

    # Entry type
    entry_mode: str = 'mr'  # 'mr' = mean reversion, 'momentum' = trend continuation

    # Core exit params
    target_retrace: float = TARGET_RETRACE
    stop_mult: float = STOP_MULT
    stop_decay: float = STOP_DECAY
    max_hold: int = MAX_HOLD
    trail_activation: float = 0.5
    trail_factor: float = 0.3

    # Y-Kitchen innovations (always on)
    min_conf_gate: int = MIN_CONFIDENCE
    dir_diversity: bool = True
    steep_cap_curve: bool = True
    vol_adjusted_size: bool = True
    soft_equity_scale: bool = True

    # Steep cap multipliers
    cap_mult_conf2: float = 0.3
    cap_mult_conf3: float = 0.5
    cap_mult_conf4: float = 1.5
    cap_mult_conf5: float = 2.0

    # Soft equity thresholds
    soft_eq_dd1: float = 0.03
    soft_eq_scale1: float = 0.85
    soft_eq_dd2: float = 0.05
    soft_eq_scale2: float = 0.70
    soft_eq_dd3: float = 0.08
    soft_eq_scale3: float = 0.50

    # PBT (Partial-Breakeven-Trail)
    partial_exit: bool = True
    partial_fraction: float = 0.5
    trail_atr_mult: float = 1.5
    breakeven_after_tp1: bool = True

    # Reversal-specific: runner target (0 = no fixed target, pure ATR trail)
    runner_target_mult: float = 0.0  # if > 0, runner exits at this × move_size
    runner_max_hold_mult: float = 2.0  # runner max_hold = this × max_hold

    # Max slots
    max_slots: int = 2


# ═══════════════════════════════════════════════════════════════════
# Portfolio Simulator
# ═══════════════════════════════════════════════════════════════════
def simulate(
    sym_close: dict[str, np.ndarray],
    sym_high: dict[str, np.ndarray],
    sym_low: dict[str, np.ndarray],
    sym_dirs: dict[str, np.ndarray],
    sym_confs: dict[str, np.ndarray],
    sym_moves: dict[str, np.ndarray],
    cfg: RevConfig,
    sym_atr: dict[str, np.ndarray] | None = None,
) -> dict:
    """Portfolio simulator with the full R5 framework."""
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    balance = INITIAL_CAPITAL
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance
    max_dd = 0.0

    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for i in range(1, n_bars):
        # ═══ Exit phase ═══
        to_close: list[str] = []

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            ep = pos['entry_price']
            lev = pos['leverage']

            if pos['direction'] == 1:
                gain_close = (c - ep) / ep
            else:
                gain_close = (ep - c) / ep

            target = pos['move_size'] * cfg.target_retrace
            time_factor = 1.0 - cfg.stop_decay * (pos['hold_count'] / cfg.max_hold)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * cfg.stop_mult * time_factor

            # ════════════════════════════════════════════════
            # PBT mode: Phase 1 (pre-TP1) and Phase 2 (runner)
            # ════════════════════════════════════════════════
            if cfg.partial_exit and not pos.get('tp1_hit', False):
                # Phase 1: Watch for TP1
                if gain_close >= target:
                    # TP1 HIT → partial close, enter runner
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain_close

                    tp1_notional = pos['notional'] * cfg.partial_fraction
                    remaining_notional = pos['notional'] - tp1_notional
                    funding = lev * FUNDING_PER_BAR * tp1_notional * pos['hold_count']
                    close_cost = lev * COST_PER_SIDE * tp1_notional
                    gross_pnl = gain_close * lev * tp1_notional
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl

                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': 'tp1',
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / tp1_notional) * 100,
                        'bars_held': pos['hold_count'],
                    })

                    pos['notional'] = remaining_notional
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain_close

                    if cfg.breakeven_after_tp1:
                        pos['trail_floor'] = 0.0
                    else:
                        pos['trail_floor'] = -effective_stop

                elif gain_close < -effective_stop:
                    self_close(balance, trades, pos, sym, gain_close, lev, 'stop', to_close, i)
                    balance = trades[-1]['_bal_after']

                elif pos['hold_count'] >= cfg.max_hold:
                    self_close(balance, trades, pos, sym, gain_close, lev, 'time', to_close, i)
                    balance = trades[-1]['_bal_after']

                else:
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * cfg.trail_activation
                            and gain_close < pos['best_pnl'] * cfg.trail_factor):
                        self_close(balance, trades, pos, sym, gain_close, lev, 'trail', to_close, i)
                        balance = trades[-1]['_bal_after']

            elif cfg.partial_exit and pos.get('tp1_hit', False):
                # Phase 2: Runner with ATR trailing
                if gain_close > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain_close

                atr_val = 0.02
                if sym_atr is not None:
                    atr_arr = sym_atr.get(sym)
                    if atr_arr is not None and i < len(atr_arr) and not np.isnan(atr_arr[i]):
                        atr_val = atr_arr[i]

                trail_stop = pos['highest_since_tp1'] - cfg.trail_atr_mult * atr_val
                floor_val = pos.get('trail_floor', -effective_stop)
                if trail_stop < floor_val:
                    trail_stop = floor_val

                exit_runner = False
                exit_reason = None

                if gain_close <= trail_stop:
                    exit_runner = True
                    exit_reason = 'trail_runner'
                elif pos['hold_count'] >= cfg.max_hold * cfg.runner_max_hold_mult:
                    exit_runner = True
                    exit_reason = 'time_runner'
                elif cfg.runner_target_mult > 0:
                    runner_target = pos['move_size'] * cfg.runner_target_mult
                    if gain_close >= runner_target:
                        exit_runner = True
                        exit_reason = 'target_runner'

                if exit_runner:
                    funding = lev * FUNDING_PER_BAR * pos['notional'] * (i - pos.get('tp1_bar', pos['entry_bar']))
                    close_cost = lev * COST_PER_SIDE * pos['notional']
                    gross_pnl = gain_close * lev * pos['notional']
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': exit_reason,
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                        '_bal_after': balance,
                    })
                    to_close.append(sym)

            else:
                # Non-PBT: standard exit
                exit_reason = None

                if gain_close >= target:
                    exit_reason = 'target'
                elif gain_close < -effective_stop:
                    exit_reason = 'stop'
                else:
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * cfg.trail_activation
                            and gain_close < pos['best_pnl'] * cfg.trail_factor):
                        exit_reason = 'trail'
                    elif pos['hold_count'] >= cfg.max_hold:
                        exit_reason = 'time'

                if exit_reason is not None:
                    self_close(balance, trades, pos, sym, gain_close, lev, exit_reason, to_close, i)
                    balance = trades[-1]['_bal_after']

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Entry phase ═══
        n_active = len(positions)
        if n_active < cfg.max_slots:
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

                if confidence < cfg.min_conf_gate:
                    continue

                if cfg.dir_diversity:
                    same_dir = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same_dir >= 2:
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

                if cfg.steep_cap_curve:
                    if confidence >= 5:
                        per_slot *= cfg.cap_mult_conf5
                    elif confidence >= 4:
                        per_slot *= cfg.cap_mult_conf4
                    elif confidence == 3:
                        per_slot *= cfg.cap_mult_conf3
                    else:
                        per_slot *= cfg.cap_mult_conf2
                    per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if cfg.vol_adjusted_size and sym_atr is not None:
                    atr_arr = sym_atr.get(sym)
                    if atr_arr is not None and i < len(atr_arr) and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
                        vol_factor = 0.01 / atr_arr[i]
                        vol_factor = max(0.4, min(vol_factor, 2.0))
                        per_slot *= vol_factor
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if cfg.soft_equity_scale:
                    eq_now = equity_curve[i - 1] if i > 0 else INITIAL_CAPITAL
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    if dd_pct > cfg.soft_eq_dd3:
                        per_slot *= cfg.soft_eq_scale3
                    elif dd_pct > cfg.soft_eq_dd2:
                        per_slot *= cfg.soft_eq_scale2
                    elif dd_pct > cfg.soft_eq_dd1:
                        per_slot *= cfg.soft_eq_scale1

                entry_cost = EFFECTIVE_LEV * COST_PER_SIDE * per_slot
                balance -= entry_cost

                positions[sym] = {
                    'direction': direction,
                    'confidence': confidence,
                    'move_size': move_size,
                    'entry_price': sym_close[sym][i],
                    'entry_bar': i,
                    'hold_count': 0,
                    'best_pnl': 0.0,
                    'notional': per_slot,
                    'leverage': EFFECTIVE_LEV,
                }

        # ═══ Update equity curve ═══
        total_unrealized = 0.0
        for sym, pos in positions.items():
            c = sym_close[sym][i]
            ep = pos['entry_price']
            if pos['direction'] == 1:
                gain = (c - ep) / ep
            else:
                gain = (ep - c) / ep
            total_unrealized += gain * pos['leverage'] * pos['notional']

        equity_curve[i] = balance + total_unrealized
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return _compute_stats(equity_curve, trades, INITIAL_CAPITAL, max_dd)


def self_close(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx):
    """Helper to close full position (or non-PBT position)."""
    funding = lev * FUNDING_PER_BAR * pos['notional'] * pos['hold_count']
    close_cost = lev * COST_PER_SIDE * pos['notional']
    gross_pnl = gain * lev * pos['notional']
    net_pnl = gross_pnl - close_cost - funding
    new_balance = balance + net_pnl
    trades.append({
        'symbol': sym, 'direction': pos['direction'],
        'confidence': pos['confidence'],
        'exit_reason': reason,
        'pnl_usd': net_pnl,
        'pnl_pct': (net_pnl / pos['notional']) * 100,
        'bars_held': pos['hold_count'],
        '_bal_after': new_balance,
    })
    to_close.append(sym)


def _compute_stats(eq, trades, capital, max_dd):
    """Compute standard performance metrics."""
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]

    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0.0
    total_ret = (eq[-1] / capital - 1.0) * 100

    gp = sum(t['pnl_usd'] for t in wins)
    gl = sum(abs(t['pnl_usd']) for t in losses)
    pf = gp / gl if gl > 0 else 999.0

    eq_rets = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    sharpe = (float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25))
              if len(eq_rets) > 0 and np.std(eq_rets) > 0 else 0.0)
    calmar = total_ret / (max_dd * 100) if max_dd > 0 else 0.0

    aw = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0
    al = float(np.mean([abs(t['pnl_pct']) for t in losses])) if losses else 1.0
    rr = aw / al if al > 0 else 999.0
    avg_pnl = float(np.mean([t['pnl_pct'] for t in trades])) if trades else 0.0
    avg_bars = float(np.mean([t['bars_held'] for t in trades])) if trades else 0.0
    n_days = len(eq) / 96
    tpd = round(n_trades / n_days, 2) if n_days > 0 else 0.0

    reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        reasons[r] = reasons.get(r, 0) + 1

    # Separate runner vs tp1 stats
    tp1_trades = [t for t in trades if t['exit_reason'] == 'tp1']
    runner_trades = [t for t in trades if 'runner' in t.get('exit_reason', '')]
    runner_wins = [t for t in runner_trades if t['pnl_usd'] > 0]
    runner_wr = len(runner_wins) / len(runner_trades) * 100 if runner_trades else 0.0
    runner_avg = float(np.mean([t['pnl_pct'] for t in runner_trades])) if runner_trades else 0.0

    return {
        'total_return_pct': round(total_ret, 1),
        'final_balance': round(eq[-1], 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 1),
        'n_trades': n_trades,
        'trades_per_day': tpd,
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'avg_pnl_pct': round(avg_pnl, 3),
        'avg_win_pct': round(aw, 3),
        'avg_loss_pct': round(al, 3),
        'actual_rr': round(rr, 2),
        'avg_bars_held': round(avg_bars, 1),
        'exit_reasons': reasons,
        'n_runner_trades': len(runner_trades),
        'runner_win_rate': round(runner_wr, 1),
        'runner_avg_pnl': round(runner_avg, 3),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    commit = _get_git_commit()

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

    print("=" * 130)
    print("  REVERSAL A/B TEST — MR vs Semi-Reversal vs Full Reversal")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Question: Can we improve R:R by shifting from pure MR to reversal?")
    print(f"  Baseline: v12.3.3 production (MR: T100_S38_PBT + M-2SSTSX)")
    print("=" * 130)

    # ── Load data ──
    print(f"\n📊 Loading 50-symbol data (2190d / 6yr)...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        all_data[sym] = d
        h4 = load_4h(sym)
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # ── Scan MR entries (production: fade the move) ──
    print(f"\n🔍 Scanning MR entry signals (production)...")
    t0 = time.time()

    sym_dirs_mr: dict[str, np.ndarray] = {}
    sym_confs_mr: dict[str, np.ndarray] = {}
    sym_moves_mr: dict[str, np.ndarray] = {}
    sym_close: dict[str, np.ndarray] = {}
    sym_high: dict[str, np.ndarray] = {}
    sym_low: dict[str, np.ndarray] = {}

    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        c, h, l, v, t = d['close'][:n], d['high'][:n], d['low'][:n], d['volume'][:n], d['taker_buy'][:n]
        b = btc_close[:n]

        dirs, confs, moves = scan_entries(
            c, h, l, v, t, b,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
        )

        close_4h = all_4h.get(sym)
        if close_4h is not None:
            dirs, confs, moves = apply_mtf_filter(dirs, confs, moves, close_4h, MTF_SMA_PERIOD, ratio=16)

        sym_dirs_mr[sym] = dirs
        sym_confs_mr[sym] = confs
        sym_moves_mr[sym] = moves
        sym_close[sym] = c
        sym_high[sym] = h
        sym_low[sym] = l

    # ── Scan Momentum entries (trade WITH the move) ──
    print(f"🔍 Scanning Momentum entry signals...")
    sym_dirs_mom: dict[str, np.ndarray] = {}
    sym_confs_mom: dict[str, np.ndarray] = {}
    sym_moves_mom: dict[str, np.ndarray] = {}

    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        c, h, l, v, t = d['close'][:n], d['high'][:n], d['low'][:n], d['volume'][:n], d['taker_buy'][:n]
        b = btc_close[:n]

        dirs, confs, moves = scan_momentum_entries(
            c, h, l, v, t, b,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, RSI_PERIOD, VOL_PERIOD,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
        )

        close_4h = all_4h.get(sym)
        if close_4h is not None:
            dirs, confs, moves = apply_mtf_filter(dirs, confs, moves, close_4h, MTF_SMA_PERIOD, ratio=16)

        sym_dirs_mom[sym] = dirs
        sym_confs_mom[sym] = confs
        sym_moves_mom[sym] = moves

    # Pre-compute ATR per symbol
    sym_atr: dict[str, np.ndarray] = {}
    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        atr = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)
        sym_atr[sym] = atr

    # Left-pad all arrays to same length
    max_bars = max(len(sym_close[s]) for s in sym_close)
    all_dicts = [
        (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
        (sym_dirs_mr, 'zero'), (sym_confs_mr, 'zero'), (sym_moves_mr, 'zero'),
        (sym_dirs_mom, 'zero'), (sym_confs_mom, 'zero'), (sym_moves_mom, 'zero'),
        (sym_atr, 'nan'),
    ]
    for sym in list(sym_close.keys()):
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

    mr_entries = sum(np.count_nonzero(sym_dirs_mr[s]) for s in sym_dirs_mr)
    mom_entries = sum(np.count_nonzero(sym_dirs_mom[s]) for s in sym_dirs_mom)
    scan_time = time.time() - t0
    print(f"  MR entries: {mr_entries:,}  |  Momentum entries: {mom_entries:,}  ({scan_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # Define A/B Test Variants
    # ══════════════════════════════════════════════════════════════
    variants = [
        # ── A: Production MR (control) ──
        (
            RevConfig(
                name='A-MR-Prod',
                desc='v12.3.3 production: T100_S38_PBT, MR entry, 2 slots',
                target_retrace=1.0,
                stop_mult=0.382,
                max_hold=20,
                runner_max_hold_mult=2.0,
            ),
            'mr',
        ),
        # ── B: MR with wider target (1.5×) ──
        (
            RevConfig(
                name='B-MR-Wide',
                desc='MR entry, wider target 1.5× retrace, max_hold=30',
                target_retrace=1.5,
                stop_mult=0.382,
                max_hold=30,
                runner_max_hold_mult=2.0,
            ),
            'mr',
        ),
        # ── C: Semi-reversal (2× target, wider stop) ──
        (
            RevConfig(
                name='C-SemiRev',
                desc='MR entry, 2.0× target, wider stop 0.5, max_hold=40',
                target_retrace=2.0,
                stop_mult=0.5,
                max_hold=40,
                runner_max_hold_mult=2.5,
            ),
            'mr',
        ),
        # ── D: Deep reversal (3× target) ──
        (
            RevConfig(
                name='D-DeepRev',
                desc='MR entry, 3.0× target, stop 0.5, max_hold=50',
                target_retrace=3.0,
                stop_mult=0.5,
                max_hold=50,
                runner_max_hold_mult=3.0,
            ),
            'mr',
        ),
        # ── E: Pure ATR trail (no fixed target) ──
        (
            RevConfig(
                name='E-ATR-Only',
                desc='MR entry, no fixed target, 1.5×ATR trail, max_hold=60',
                target_retrace=99.0,  # effectively no target
                stop_mult=0.5,
                max_hold=60,
                trail_atr_mult=1.5,
                runner_max_hold_mult=1.0,  # no separate runner phase
                partial_exit=False,  # pure trail, no PBT
            ),
            'mr',
        ),
        # ── F: Wide ATR trail (no target) ──
        (
            RevConfig(
                name='F-ATR-Wide',
                desc='MR entry, no target, 2.5×ATR trail, max_hold=80',
                target_retrace=99.0,
                stop_mult=0.5,
                max_hold=80,
                trail_atr_mult=2.5,
                runner_max_hold_mult=1.0,
                partial_exit=False,
            ),
            'mr',
        ),
        # ── G: MR→Reversal Hybrid (TP1 at 1.0×, runner targets 3×) ──
        (
            RevConfig(
                name='G-Hybrid',
                desc='MR entry, TP1=1.0×, runner targets 3× (hybrid MR→Rev)',
                target_retrace=1.0,
                stop_mult=0.382,
                max_hold=20,
                runner_target_mult=3.0,
                runner_max_hold_mult=4.0,
                trail_atr_mult=2.0,
            ),
            'mr',
        ),
        # ── H: Hybrid with wider trail ──
        (
            RevConfig(
                name='H-HybWide',
                desc='MR entry, TP1=1.0×, runner 4×ATR trail, no runner target',
                target_retrace=1.0,
                stop_mult=0.382,
                max_hold=20,
                runner_target_mult=0.0,  # pure trail runner
                runner_max_hold_mult=4.0,
                trail_atr_mult=2.5,
            ),
            'mr',
        ),
        # ── I: Hybrid with tighter trail (capture reversal faster) ──
        (
            RevConfig(
                name='I-HybTight',
                desc='MR entry, TP1=1.5×, runner 1.0×ATR trail, max runner 3×hold',
                target_retrace=1.5,
                stop_mult=0.382,
                max_hold=25,
                runner_target_mult=0.0,
                runner_max_hold_mult=3.0,
                trail_atr_mult=1.0,
            ),
            'mr',
        ),
        # ── J: Momentum (pure reversal direction) ──
        (
            RevConfig(
                name='J-Momentum',
                desc='Momentum entry (WITH the move), 1.5× target, PBT',
                target_retrace=1.5,
                stop_mult=0.5,
                max_hold=30,
                runner_max_hold_mult=2.0,
                min_conf_gate=2,  # lower gate for momentum
            ),
            'momentum',
        ),
        # ── K: Momentum wide trail ──
        (
            RevConfig(
                name='K-Mom-Wide',
                desc='Momentum entry, 2.0×ATR trail, no fixed target, max_hold=60',
                target_retrace=99.0,
                stop_mult=0.5,
                max_hold=60,
                trail_atr_mult=2.0,
                runner_max_hold_mult=1.0,
                partial_exit=False,
                min_conf_gate=2,
            ),
            'momentum',
        ),
        # ── L: Semi-reversal with aggressive runners (1.5 TP, 3×ATR) ──
        (
            RevConfig(
                name='L-SemiAggr',
                desc='MR entry, TP1=1.5×, runner 3×ATR, max 5×hold',
                target_retrace=1.5,
                stop_mult=0.45,
                max_hold=25,
                runner_target_mult=0.0,
                runner_max_hold_mult=5.0,
                trail_atr_mult=3.0,
            ),
            'mr',
        ),
    ]

    # ── Run ──
    results: dict[str, dict] = {}
    baseline = None

    for cfg, entry_mode in variants:
        print(f"\n{'─' * 120}")
        print(f"  ⚡ {cfg.name}: {cfg.desc}")
        print(f"{'─' * 120}")

        if entry_mode == 'momentum':
            dirs_use = sym_dirs_mom
            confs_use = sym_confs_mom
            moves_use = sym_moves_mom
        else:
            dirs_use = sym_dirs_mr
            confs_use = sym_confs_mr
            moves_use = sym_moves_mr

        t0 = time.time()
        result = simulate(
            sym_close, sym_high, sym_low,
            dirs_use, confs_use, moves_use,
            cfg=cfg, sym_atr=sym_atr,
        )
        elapsed = time.time() - t0

        result['description'] = cfg.desc
        result['elapsed'] = round(elapsed, 2)
        results[cfg.name] = result

        if cfg.name == 'A-MR-Prod':
            baseline = result

        if baseline:
            wr_ok = '✓' if result['win_rate'] >= baseline['win_rate'] else '✗'
            rr_ok = '✓' if result['actual_rr'] >= baseline['actual_rr'] else '✗'
            dd_ok = '✓' if result['max_dd_pct'] <= baseline['max_dd_pct'] else '✗'
        else:
            wr_ok = rr_ok = dd_ok = '?'

        reasons_str = '  '.join(f"{k}={v}" for k, v in sorted(result['exit_reasons'].items()))
        print(f"  Ret: {result['total_return_pct']:+,.1f}%  "
              f"Sharpe: {result['sharpe']:.2f}  DD: {result['max_dd_pct']:.1f}%[{dd_ok}]  "
              f"Calmar: {result['calmar']:.1f}")
        print(f"  Trades: {result['n_trades']:,} ({result['trades_per_day']}/day)  "
              f"Bars: {result['avg_bars_held']:.1f}  "
              f"WR: {result['win_rate']:.1f}%[{wr_ok}]  R:R: {result['actual_rr']:.2f}[{rr_ok}]  "
              f"PF: {result['profit_factor']:.2f}  [{elapsed:.1f}s]")
        print(f"  Runners: {result['n_runner_trades']} (WR: {result['runner_win_rate']:.1f}%, "
              f"AvgPnL: {result['runner_avg_pnl']:+.3f}%)")
        print(f"  Exits: {reasons_str}")

    # ══════════════════════════════════════════════════════════════
    # Comparison Table
    # ══════════════════════════════════════════════════════════════
    names = list(results.keys())
    cw = 12

    print(f"\n\n{'=' * 160}")
    print(f"  REVERSAL A/B TEST RESULTS")
    if baseline:
        print(f"  Baseline (A-MR-Prod): WR {baseline['win_rate']}% R:R {baseline['actual_rr']} "
              f"DD {baseline['max_dd_pct']}% Sharpe {baseline['sharpe']} PF {baseline['profit_factor']}")
    print(f"{'=' * 160}\n")

    print(f"  {'Metric':<18}", end='')
    for n in names:
        print(f"  {n[:cw]:>{cw}}", end='')
    print()
    print("  " + "-" * (18 + (cw + 2) * len(names)))

    rows = [
        ('Return', 'total_return_pct', '{:+,.1f}%'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Calmar', 'calmar', '{:.1f}'),
        ('Max DD', 'max_dd_pct', '{:.1f}%'),
        ('Trades', 'n_trades', '{:,d}'),
        ('Trades/Day', 'trades_per_day', '{:.2f}'),
        ('Bars Held', 'avg_bars_held', '{:.1f}'),
        ('★ Win Rate', 'win_rate', '{:.1f}%'),
        ('★ R:R', 'actual_rr', '{:.2f}'),
        ('PF', 'profit_factor', '{:.2f}'),
        ('Avg Win', 'avg_win_pct', '{:+.3f}%'),
        ('Avg Loss', 'avg_loss_pct', '{:.3f}%'),
        ('Runner Count', 'n_runner_trades', '{:,d}'),
        ('Runner WR', 'runner_win_rate', '{:.1f}%'),
        ('Runner AvgPnL', 'runner_avg_pnl', '{:+.3f}%'),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<18}", end='')
        for n in names:
            s = fmt.format(results[n][key])
            print(f"  {s:>{cw}}", end='')
        print()

    # ── Delta vs Production ──
    if baseline:
        print(f"\n{'─' * 160}")
        print(f"  DELTA vs A-MR-Prod (Production)")
        print(f"{'─' * 160}\n")

        for n in names:
            if n == 'A-MR-Prod':
                continue
            r = results[n]
            d_wr = r['win_rate'] - baseline['win_rate']
            d_rr = r['actual_rr'] - baseline['actual_rr']
            d_ret = r['total_return_pct'] - baseline['total_return_pct']
            d_sh = r['sharpe'] - baseline['sharpe']
            d_dd = r['max_dd_pct'] - baseline['max_dd_pct']
            d_pf = r['profit_factor'] - baseline['profit_factor']

            wr_ok = '✓' if d_wr >= 0 else '✗'
            rr_ok = '✓' if d_rr >= 0 else '✗'
            dd_ok = '✓' if d_dd <= 0 else '✗'

            tags = []
            if d_rr > 0 and d_dd <= 0:
                tags.append('★ R:R+DD')
            if d_wr >= 0 and d_rr >= 0 and d_dd <= 0:
                tags.append('★★ TRIPLE WIN')
            if d_rr > 0.3:
                tags.append('🎯 R:R+0.3')
            tag = '  '.join(tags)

            print(f"  {n:<12} ΔWR={d_wr:>+5.1f}%p[{wr_ok}]  ΔR:R={d_rr:>+5.2f}[{rr_ok}]  "
                  f"ΔDD={d_dd:>+5.1f}%[{dd_ok}]  ΔRet={d_ret:>+8,.1f}%  "
                  f"ΔSharpe={d_sh:>+5.2f}  ΔPF={d_pf:>+5.2f}  {tag}")

    # ══════════════════════════════════════════════════════════════
    # R:R Focus Analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print(f"  R:R FOCUS RANKING (sorted by R:R)")
    print(f"{'=' * 80}\n")

    sorted_by_rr = sorted(results.items(), key=lambda x: x[1]['actual_rr'], reverse=True)
    for rank, (name, r) in enumerate(sorted_by_rr, 1):
        safe = '✓' if r['max_dd_pct'] <= (baseline['max_dd_pct'] + 2.0 if baseline else 15.0) else '✗'
        print(f"  #{rank:<3} {name:<14} R:R={r['actual_rr']:.2f}  WR={r['win_rate']:.1f}%  "
              f"PF={r['profit_factor']:.2f}  DD={r['max_dd_pct']:.1f}%[{safe}]  "
              f"Ret={r['total_return_pct']:+,.1f}%  Sharpe={r['sharpe']:.2f}")

    # ── Save results ──
    out_path = Path('logs/reversal_ab_results.json')
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
    with open(out_path, 'w') as f:
        json.dump({
            'commit': commit,
            'date': datetime.now(timezone.utc).isoformat(),
            'results': clean_results,
        }, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")

    out_txt = Path('logs/reversal_ab_output.txt')
    print(f"📊 Done! Output also at: {out_txt}")


if __name__ == '__main__':
    main()
