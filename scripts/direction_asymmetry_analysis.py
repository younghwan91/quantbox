#!/usr/bin/env python3
"""
Direction Asymmetry Analysis — Long vs Short Performance Deep Dive
===================================================================
Hypothesis: crypto futures are structurally asymmetric (upward bias,
faster drops, funding drag on longs) yet we treat long=short with
identical parameters. This script quantifies the asymmetry.

Uses EXACT production logic (v12.4 I-HybTight + M-2SSTSX + PBT)
on 6-year data (2190 days) covering COVID, Luna, FTX crashes.

Metrics computed per direction:
  - Trade count, win rate, profit factor
  - Avg win %, avg loss %, R:R, expected value per trade
  - Avg move_size at entry (are dips bigger than pumps?)
  - Avg bars held (do shorts close faster?)
  - Exit reason distribution (TP1/stop/trail/time by direction)
  - Confidence distribution (do longs get higher confidence?)
  - Yearly breakdown (does asymmetry shift with regime?)
  - Per-symbol breakdown (which coins have strongest L/S skew?)

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/direction_asymmetry_analysis.py
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
DATA_DAYS = 2190  # 6 years

# ═══════════════════════════════════════════════════════════════════
# Production parameters (v12.4 I-HybTight) — MUST match .env / on_bar
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032          # fallback (ATR adaptive overrides)
TARGET_RETRACE = 1.5       # v12.4 HybTight
STOP_MULT = 0.382          # Fibonacci stop
MAX_HOLD = 25              # v12.4 HybTight (was 20)
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
ATR_MULT = 2.0             # v12.2: 2.0× (was 2.5)
THR_FLOOR = 0.008
THR_CAP = 0.06
MTF_SMA_PERIOD = 12

# Cost model
INITIAL_CAPITAL = 2095.0
COST_PER_SIDE = 0.0004     # 0.04% maker/taker
FUNDING_PER_BAR = 0.01 / 100 / 32  # ~0.01% per 8h / 32 bars

# PBT (v12.4 I-HybTight)
PARTIAL_EXIT = True
PARTIAL_FRACTION = 0.5
TRAIL_ATR_MULT = 1.0       # v12.4 HybTight (was 1.5)
BREAKEVEN_AFTER_TP1 = True
RUNNER_MAX_HOLD_MULT = 3.0  # v12.4 HybTight (was 2.0)

# Sizing (M-2SSTSX)
MAX_SLOTS = 2
DIR_DIVERSITY = True
STEEP_CAP = True
VOL_ADJUSTED = True
SOFT_EQUITY = True

# Steep cap multipliers
CAP_MULT = {2: 0.3, 3: 0.5, 4: 1.5, 5: 2.0}
# Soft equity thresholds
SOFT_EQ = [(0.08, 0.50), (0.05, 0.70), (0.03, 0.85)]

# Year boundaries (approximate, in 15m bars from start)
BARS_PER_YEAR = 96 * 365  # 35040 bars/year


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
    # Fallback: any available file
    pattern2 = f"{symbol}_{interval}_*.npz"
    matches2 = sorted(CACHE_DIR.glob(pattern2))
    return matches2[-1] if matches2 else None


def load_15m(symbol: str, days: int = DATA_DAYS) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None:
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int = DATA_DAYS) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None:
        return None
    return np.load(cache)['close']


# ═══════════════════════════════════════════════════════════════════
# Entry Scanner — EXACT production Risk-Off logic (matches on_bar)
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
    """Scan ALL entry opportunities — Risk-Off only (production parity)."""
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
            # v12.4.1: Risk-Off ONLY (universal, validated across all regimes)
            if ret < -thr and btc_mom < 0:
                # LONG: alt dips with BTC → sympathy → mean revert
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
                # SHORT: alt pumps with BTC bounce → fades back
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
# Portfolio Simulator — EXACT production logic with trade-level tracking
# ═══════════════════════════════════════════════════════════════════
def simulate(
    sym_close: dict[str, np.ndarray],
    sym_high: dict[str, np.ndarray],
    sym_low: dict[str, np.ndarray],
    sym_dirs: dict[str, np.ndarray],
    sym_confs: dict[str, np.ndarray],
    sym_moves: dict[str, np.ndarray],
    sym_atr: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run portfolio simulation, return (equity_curve, trades).

    Each trade dict includes: symbol, direction, confidence, move_size,
    entry_bar, exit_bar, exit_reason, pnl_usd, pnl_pct, bars_held, notional.
    """
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    balance = INITIAL_CAPITAL
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance

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

            target = pos['move_size'] * TARGET_RETRACE
            time_factor = 1.0 - STOP_DECAY * (pos['hold_count'] / MAX_HOLD)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * STOP_MULT * time_factor

            # ═══ PBT Phase 1 (pre-TP1) ═══
            if PARTIAL_EXIT and not pos.get('tp1_hit', False):
                if gain_close >= target:
                    # TP1 HIT → partial close
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain_close

                    tp1_notional = pos['notional'] * PARTIAL_FRACTION
                    remaining_notional = pos['notional'] - tp1_notional
                    funding = lev * FUNDING_PER_BAR * tp1_notional * pos['hold_count']
                    close_cost = lev * COST_PER_SIDE * tp1_notional
                    gross_pnl = gain_close * lev * tp1_notional
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl

                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'move_size': pos['move_size'],
                        'entry_bar': pos['entry_bar'],
                        'exit_bar': i,
                        'exit_reason': 'tp1',
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / tp1_notional) * 100,
                        'bars_held': pos['hold_count'],
                        'notional': tp1_notional,
                        'gain_at_exit': gain_close,
                    })

                    pos['notional'] = remaining_notional
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain_close
                    pos['trail_floor'] = 0.0 if BREAKEVEN_AFTER_TP1 else -effective_stop

                elif gain_close < -effective_stop:
                    _close_pos(balance, trades, pos, sym, gain_close, lev, 'stop', to_close, i)
                    balance = trades[-1]['_bal']
                elif pos['hold_count'] >= MAX_HOLD:
                    _close_pos(balance, trades, pos, sym, gain_close, lev, 'time', to_close, i)
                    balance = trades[-1]['_bal']
                else:
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * 0.5
                            and gain_close < pos['best_pnl'] * 0.3):
                        _close_pos(balance, trades, pos, sym, gain_close, lev, 'trail', to_close, i)
                        balance = trades[-1]['_bal']

            # ═══ PBT Phase 2 (runner) ═══
            elif PARTIAL_EXIT and pos.get('tp1_hit', False):
                if gain_close > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain_close

                atr_val = 0.02
                if sym_atr is not None:
                    atr_arr = sym_atr.get(sym)
                    if atr_arr is not None and i < len(atr_arr) and not np.isnan(atr_arr[i]):
                        atr_val = atr_arr[i]

                trail_stop = pos['highest_since_tp1'] - TRAIL_ATR_MULT * atr_val
                floor_val = pos.get('trail_floor', -effective_stop)
                if trail_stop < floor_val:
                    trail_stop = floor_val

                exit_runner = False
                exit_reason = None

                if gain_close <= trail_stop:
                    exit_runner = True
                    exit_reason = 'trail_runner'
                elif pos['hold_count'] >= MAX_HOLD * RUNNER_MAX_HOLD_MULT:
                    exit_runner = True
                    exit_reason = 'time_runner'

                if exit_runner:
                    tp1_bar = pos.get('tp1_bar', pos['entry_bar'])
                    funding = lev * FUNDING_PER_BAR * pos['notional'] * (i - tp1_bar)
                    close_cost = lev * COST_PER_SIDE * pos['notional']
                    gross_pnl = gain_close * lev * pos['notional']
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'move_size': pos['move_size'],
                        'entry_bar': pos['entry_bar'],
                        'exit_bar': i,
                        'exit_reason': exit_reason,
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                        'notional': pos['notional'],
                        'gain_at_exit': gain_close,
                        '_bal': balance,
                    })
                    to_close.append(sym)
            else:
                # Non-PBT fallback
                exit_reason = None
                if gain_close >= target:
                    exit_reason = 'target'
                elif gain_close < -effective_stop:
                    exit_reason = 'stop'
                else:
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * 0.5
                            and gain_close < pos['best_pnl'] * 0.3):
                        exit_reason = 'trail'
                    elif pos['hold_count'] >= MAX_HOLD:
                        exit_reason = 'time'

                if exit_reason is not None:
                    _close_pos(balance, trades, pos, sym, gain_close, lev, exit_reason, to_close, i)
                    balance = trades[-1]['_bal']

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Entry phase ═══
        n_active = len(positions)
        if n_active < MAX_SLOTS:
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
                if confidence < MIN_CONFIDENCE:
                    continue
                if DIR_DIVERSITY:
                    same_dir = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same_dir >= 2:
                        continue
                candidates.append((sym, direction, confidence, move_size))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size in candidates:
                if len(positions) >= MAX_SLOTS:
                    break

                per_slot = INITIAL_CAPITAL / max(1, len(positions) + 1)
                per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if per_slot < 10.0:
                    continue

                if STEEP_CAP:
                    mult = CAP_MULT.get(confidence, CAP_MULT.get(min(confidence, 5), 0.3))
                    per_slot *= mult
                    per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if VOL_ADJUSTED and sym_atr is not None:
                    atr_arr = sym_atr.get(sym)
                    if atr_arr is not None and i < len(atr_arr) and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
                        vol_factor = 0.01 / atr_arr[i]
                        vol_factor = max(0.4, min(vol_factor, 2.0))
                        per_slot *= vol_factor
                        per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)

                if SOFT_EQUITY:
                    eq_now = equity_curve[i - 1] if i > 0 else INITIAL_CAPITAL
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
                }

        # ═══ Equity curve ═══
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

    return equity_curve, trades


def _close_pos(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx):
    """Close full position."""
    funding = lev * FUNDING_PER_BAR * pos['notional'] * pos['hold_count']
    close_cost = lev * COST_PER_SIDE * pos['notional']
    gross_pnl = gain * lev * pos['notional']
    net_pnl = gross_pnl - close_cost - funding
    new_bal = balance + net_pnl
    trades.append({
        'symbol': sym, 'direction': pos['direction'],
        'confidence': pos['confidence'],
        'move_size': pos['move_size'],
        'entry_bar': pos['entry_bar'],
        'exit_bar': bar_idx,
        'exit_reason': reason,
        'pnl_usd': net_pnl,
        'pnl_pct': (net_pnl / pos['notional']) * 100,
        'bars_held': pos['hold_count'],
        'notional': pos['notional'],
        'gain_at_exit': gain,
        '_bal': new_bal,
    })
    to_close.append(sym)


# ═══════════════════════════════════════════════════════════════════
# Direction-Aware Analysis Functions
# ═══════════════════════════════════════════════════════════════════
def analyze_by_direction(trades: list[dict]) -> dict:
    """Compute comprehensive per-direction metrics."""
    long_trades = [t for t in trades if t['direction'] == 1]
    short_trades = [t for t in trades if t['direction'] == -1]

    def _stats(tlist: list[dict], label: str) -> dict:
        if not tlist:
            return {'label': label, 'count': 0}

        wins = [t for t in tlist if t['pnl_usd'] > 0]
        losses = [t for t in tlist if t['pnl_usd'] <= 0]
        n = len(tlist)
        wr = len(wins) / n * 100
        gp = sum(t['pnl_usd'] for t in wins)
        gl = sum(abs(t['pnl_usd']) for t in losses)
        pf = gp / gl if gl > 0 else 999.0
        avg_win = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(t['pnl_pct']) for t in losses])) if losses else 1.0
        rr = avg_win / avg_loss if avg_loss > 0 else 999.0
        avg_pnl = float(np.mean([t['pnl_pct'] for t in tlist]))
        total_pnl = sum(t['pnl_usd'] for t in tlist)
        avg_bars = float(np.mean([t['bars_held'] for t in tlist]))
        avg_move = float(np.mean([t['move_size'] for t in tlist]))
        avg_gain_at_exit = float(np.mean([t['gain_at_exit'] for t in tlist]))

        # Exit reasons
        reasons = {}
        for t in tlist:
            r = t.get('exit_reason', 'unknown')
            reasons[r] = reasons.get(r, 0) + 1

        # Confidence distribution
        conf_dist = {}
        for t in tlist:
            c = t['confidence']
            if c not in conf_dist:
                conf_dist[c] = {'count': 0, 'wins': 0, 'pnl_sum': 0.0}
            conf_dist[c]['count'] += 1
            if t['pnl_usd'] > 0:
                conf_dist[c]['wins'] += 1
            conf_dist[c]['pnl_sum'] += t['pnl_usd']

        conf_breakdown = {}
        for c, cd in sorted(conf_dist.items()):
            conf_breakdown[str(c)] = {
                'count': cd['count'],
                'win_rate': round(cd['wins'] / cd['count'] * 100, 1),
                'total_pnl': round(cd['pnl_sum'], 2),
                'avg_pnl': round(cd['pnl_sum'] / cd['count'], 2),
            }

        return {
            'label': label,
            'count': n,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2),
            'avg_win_pct': round(avg_win, 3),
            'avg_loss_pct': round(avg_loss, 3),
            'rr': round(rr, 2),
            'avg_pnl_pct': round(avg_pnl, 3),
            'total_pnl_usd': round(total_pnl, 2),
            'avg_bars_held': round(avg_bars, 1),
            'avg_move_size_pct': round(avg_move * 100, 2),
            'avg_gain_at_exit_pct': round(avg_gain_at_exit * 100, 3),
            'exit_reasons': reasons,
            'confidence_breakdown': conf_breakdown,
        }

    return {
        'all': _stats(trades, 'ALL'),
        'long': _stats(long_trades, 'LONG'),
        'short': _stats(short_trades, 'SHORT'),
    }


def analyze_yearly(trades: list[dict], n_bars: int) -> list[dict]:
    """Break down direction metrics by year."""
    yearly = []
    for yr in range(6):
        bar_start = yr * BARS_PER_YEAR
        bar_end = (yr + 1) * BARS_PER_YEAR
        yr_trades = [t for t in trades if bar_start <= t['entry_bar'] < bar_end]
        if not yr_trades:
            continue

        long_t = [t for t in yr_trades if t['direction'] == 1]
        short_t = [t for t in yr_trades if t['direction'] == -1]

        def _wr(tl):
            if not tl:
                return 0.0
            return round(len([t for t in tl if t['pnl_usd'] > 0]) / len(tl) * 100, 1)

        def _avg_pnl(tl):
            if not tl:
                return 0.0
            return round(float(np.mean([t['pnl_pct'] for t in tl])), 3)

        def _pf(tl):
            if not tl:
                return 0.0
            gp = sum(t['pnl_usd'] for t in tl if t['pnl_usd'] > 0)
            gl = sum(abs(t['pnl_usd']) for t in tl if t['pnl_usd'] <= 0)
            return round(gp / gl, 2) if gl > 0 else 999.0

        def _avg_bars(tl):
            if not tl:
                return 0.0
            return round(float(np.mean([t['bars_held'] for t in tl])), 1)

        # Year labels (approx: data starts ~2020-02)
        start_year = 2020 + yr
        label = f"Yr{yr+1} ({start_year}-{start_year+1})"

        yearly.append({
            'label': label,
            'total_trades': len(yr_trades),
            'long_count': len(long_t),
            'short_count': len(short_t),
            'ls_ratio': round(len(long_t) / len(short_t), 2) if short_t else 999.0,
            'all_wr': _wr(yr_trades),
            'long_wr': _wr(long_t),
            'short_wr': _wr(short_t),
            'wr_gap_pp': round(_wr(long_t) - _wr(short_t), 1),
            'all_avg_pnl': _avg_pnl(yr_trades),
            'long_avg_pnl': _avg_pnl(long_t),
            'short_avg_pnl': _avg_pnl(short_t),
            'long_pf': _pf(long_t),
            'short_pf': _pf(short_t),
            'long_avg_bars': _avg_bars(long_t),
            'short_avg_bars': _avg_bars(short_t),
        })

    return yearly


def analyze_per_symbol(trades: list[dict]) -> list[dict]:
    """Per-symbol long vs short breakdown (top asymmetric coins)."""
    sym_data: dict[str, dict] = {}
    for t in trades:
        sym = t['symbol']
        if sym not in sym_data:
            sym_data[sym] = {'long': [], 'short': []}
        if t['direction'] == 1:
            sym_data[sym]['long'].append(t)
        else:
            sym_data[sym]['short'].append(t)

    results = []
    for sym, sd in sorted(sym_data.items()):
        long_t, short_t = sd['long'], sd['short']
        n_long, n_short = len(long_t), len(short_t)
        if n_long == 0 and n_short == 0:
            continue

        l_wr = round(len([t for t in long_t if t['pnl_usd'] > 0]) / n_long * 100, 1) if n_long > 0 else 0.0
        s_wr = round(len([t for t in short_t if t['pnl_usd'] > 0]) / n_short * 100, 1) if n_short > 0 else 0.0
        l_avg = round(float(np.mean([t['pnl_pct'] for t in long_t])), 3) if long_t else 0.0
        s_avg = round(float(np.mean([t['pnl_pct'] for t in short_t])), 3) if short_t else 0.0
        l_total = round(sum(t['pnl_usd'] for t in long_t), 2)
        s_total = round(sum(t['pnl_usd'] for t in short_t), 2)

        results.append({
            'symbol': sym,
            'long_trades': n_long,
            'short_trades': n_short,
            'ls_ratio': round(n_long / n_short, 2) if n_short > 0 else 999.0,
            'long_wr': l_wr,
            'short_wr': s_wr,
            'wr_gap': round(l_wr - s_wr, 1),
            'long_avg_pnl': l_avg,
            'short_avg_pnl': s_avg,
            'pnl_gap': round(l_avg - s_avg, 3),
            'long_total_pnl': l_total,
            'short_total_pnl': s_total,
        })

    # Sort by absolute WR gap (most asymmetric first)
    results.sort(key=lambda x: abs(x['wr_gap']), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# Formatted Output
# ═══════════════════════════════════════════════════════════════════
def print_report(
    dir_stats: dict,
    yearly: list[dict],
    per_symbol: list[dict],
    eq: np.ndarray,
    commit: str,
    elapsed: float,
    n_symbols: int,
):
    """Print formatted analysis report."""
    w = 130
    print()
    print("=" * w)
    print("  DIRECTION ASYMMETRY ANALYSIS — Long vs Short Performance (6yr)")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Config: v12.4 I-HybTight (T150_S38_H25_PBT + M-2SSTSX)")
    print(f"  Data: {DATA_DAYS}d ({DATA_DAYS/365:.1f}yr), {n_symbols} symbols")
    print(f"  Hypothesis: Long ≠ Short in crypto futures")
    print("=" * w)

    # ── Overall Direction Comparison ──
    a, l, s = dir_stats['all'], dir_stats['long'], dir_stats['short']
    print(f"\n{'─' * w}")
    print(f"  {'METRIC':<25} {'ALL':>12} {'LONG':>12} {'SHORT':>12} {'L−S GAP':>12}")
    print(f"{'─' * w}")

    def _row(name, all_v, long_v, short_v, fmt='.1f', gap=True):
        gap_v = f"{long_v - short_v:+{fmt}}" if gap and isinstance(long_v, (int, float)) and isinstance(short_v, (int, float)) else ''
        print(f"  {name:<25} {all_v:>12{fmt}} {long_v:>12{fmt}} {short_v:>12{fmt}} {gap_v:>12}")

    _row('Trade count', a['count'], l['count'], s['count'], 'd')
    _row('Win rate %', a['win_rate'], l['win_rate'], s['win_rate'])
    _row('Profit factor', a['profit_factor'], l['profit_factor'], s['profit_factor'], '.2f')
    _row('Avg win %', a['avg_win_pct'], l['avg_win_pct'], s['avg_win_pct'], '.3f')
    _row('Avg loss %', a['avg_loss_pct'], l['avg_loss_pct'], s['avg_loss_pct'], '.3f')
    _row('R:R', a['rr'], l['rr'], s['rr'], '.2f')
    _row('Avg PnL/trade %', a['avg_pnl_pct'], l['avg_pnl_pct'], s['avg_pnl_pct'], '.3f')
    _row('Total PnL $', a['total_pnl_usd'], l['total_pnl_usd'], s['total_pnl_usd'], '.2f')
    _row('Avg bars held', a['avg_bars_held'], l['avg_bars_held'], s['avg_bars_held'])
    _row('Avg entry move %', a['avg_move_size_pct'], l['avg_move_size_pct'], s['avg_move_size_pct'], '.2f')
    _row('Avg gain@exit %', a['avg_gain_at_exit_pct'], l['avg_gain_at_exit_pct'], s['avg_gain_at_exit_pct'], '.3f')

    ls_ratio = l['count'] / s['count'] if s['count'] > 0 else 0
    print(f"\n  Long:Short ratio = {ls_ratio:.2f}:1")
    print(f"  Expected value per trade: ALL={a['avg_pnl_pct']:.3f}%  LONG={l['avg_pnl_pct']:.3f}%  SHORT={s['avg_pnl_pct']:.3f}%")

    # ── Exit Reason Distribution ──
    print(f"\n{'─' * w}")
    print("  EXIT REASON DISTRIBUTION")
    print(f"{'─' * w}")
    all_reasons = set(l.get('exit_reasons', {}).keys()) | set(s.get('exit_reasons', {}).keys())
    print(f"  {'Reason':<20} {'LONG #':>8} {'LONG %':>8} {'SHORT #':>8} {'SHORT %':>8}")
    for reason in sorted(all_reasons):
        lc = l['exit_reasons'].get(reason, 0)
        sc = s['exit_reasons'].get(reason, 0)
        l_pct = lc / l['count'] * 100 if l['count'] > 0 else 0
        s_pct = sc / s['count'] * 100 if s['count'] > 0 else 0
        print(f"  {reason:<20} {lc:>8d} {l_pct:>7.1f}% {sc:>8d} {s_pct:>7.1f}%")

    # ── Confidence Distribution ──
    print(f"\n{'─' * w}")
    print("  CONFIDENCE DISTRIBUTION BY DIRECTION")
    print(f"{'─' * w}")
    all_confs = set(l.get('confidence_breakdown', {}).keys()) | set(s.get('confidence_breakdown', {}).keys())
    print(f"  {'Conf':<6} {'L count':>8} {'L WR%':>8} {'L avgPnL':>10} {'S count':>8} {'S WR%':>8} {'S avgPnL':>10}")
    for c in sorted(all_confs):
        lc = l['confidence_breakdown'].get(c, {})
        sc = s['confidence_breakdown'].get(c, {})
        print(f"  {c:<6} "
              f"{lc.get('count', 0):>8d} {lc.get('win_rate', 0):>7.1f}% {lc.get('avg_pnl', 0):>10.2f} "
              f"{sc.get('count', 0):>8d} {sc.get('win_rate', 0):>7.1f}% {sc.get('avg_pnl', 0):>10.2f}")

    # ── Yearly Breakdown ──
    print(f"\n{'─' * w}")
    print("  YEARLY DIRECTION BREAKDOWN")
    print(f"{'─' * w}")
    print(f"  {'Year':<22} {'L#':>6} {'S#':>6} {'L:S':>6} {'L WR':>7} {'S WR':>7} {'Gap':>6} "
          f"{'L avg%':>8} {'S avg%':>8} {'L PF':>6} {'S PF':>6} {'L bars':>7} {'S bars':>7}")
    for y in yearly:
        print(f"  {y['label']:<22} "
              f"{y['long_count']:>6d} {y['short_count']:>6d} {y['ls_ratio']:>5.1f}x "
              f"{y['long_wr']:>6.1f}% {y['short_wr']:>6.1f}% {y['wr_gap_pp']:>+5.1f} "
              f"{y['long_avg_pnl']:>7.3f}% {y['short_avg_pnl']:>7.3f}% "
              f"{y['long_pf']:>5.2f}x {y['short_pf']:>5.2f}x "
              f"{y['long_avg_bars']:>6.1f} {y['short_avg_bars']:>6.1f}")

    # ── Per-Symbol Top 15 Most Asymmetric ──
    print(f"\n{'─' * w}")
    print("  TOP 15 MOST DIRECTION-ASYMMETRIC SYMBOLS (by WR gap)")
    print(f"{'─' * w}")
    print(f"  {'Symbol':<18} {'L#':>5} {'S#':>5} {'L:S':>5} {'L WR':>7} {'S WR':>7} {'Gap':>7} "
          f"{'L avg%':>8} {'S avg%':>8} {'L $':>10} {'S $':>10}")
    for ps in per_symbol[:15]:
        print(f"  {ps['symbol']:<18} "
              f"{ps['long_trades']:>5d} {ps['short_trades']:>5d} {ps['ls_ratio']:>4.1f}x "
              f"{ps['long_wr']:>6.1f}% {ps['short_wr']:>6.1f}% {ps['wr_gap']:>+6.1f} "
              f"{ps['long_avg_pnl']:>7.3f}% {ps['short_avg_pnl']:>7.3f}% "
              f"{ps['long_total_pnl']:>10.2f} {ps['short_total_pnl']:>10.2f}")

    # ── Portfolio Stats ──
    total_ret = (eq[-1] / INITIAL_CAPITAL - 1.0) * 100
    peak_val = np.max(eq)
    max_dd = 0.0
    running_peak = eq[0]
    for v in eq:
        if v > running_peak:
            running_peak = v
        dd = (running_peak - v) / running_peak if running_peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    print(f"\n{'─' * w}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'─' * w}")
    print(f"  Total return: {total_ret:,.1f}%  |  Final: ${eq[-1]:,.2f}  |  Peak: ${peak_val:,.2f}")
    print(f"  Max DD: {max_dd * 100:.1f}%  |  Calmar: {total_ret / (max_dd * 100):.1f}" if max_dd > 0 else "  Max DD: 0.0%")
    print(f"  Total trades: {a['count']}  |  Long: {l['count']} ({l['count']/a['count']*100:.1f}%)  |  Short: {s['count']} ({s['count']/a['count']*100:.1f}%)")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * w)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    commit = _get_git_commit()
    t0 = time.time()

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
    print("  DIRECTION ASYMMETRY ANALYSIS — Long vs Short Performance")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Config: v12.4 I-HybTight (T150_S38_H25_PBT + M-2SSTSX)")
    print(f"  Data: {DATA_DAYS}d, 50 symbols")
    print("=" * 130)

    # ── Load data ──
    print(f"\n📊 Loading 50-symbol data ({DATA_DAYS}d / {DATA_DAYS/365:.0f}yr)...")
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
            print(f"  SKIP {sym} (no 15m data)")
            continue
        c4h = load_4h(sym)
        if c4h is None:
            print(f"  SKIP {sym} (no 4h data)")
            continue
        all_data[sym] = d
        all_4h[sym] = c4h

    n_loaded = len(all_data)
    print(f"  Loaded {n_loaded} symbols")

    # ── Scan entries ──
    print(f"\n🔍 Scanning entries (Risk-Off only, conf≥{MIN_CONFIDENCE})...")
    sym_dirs: dict[str, np.ndarray] = {}
    sym_confs: dict[str, np.ndarray] = {}
    sym_moves: dict[str, np.ndarray] = {}
    sym_close: dict[str, np.ndarray] = {}
    sym_high: dict[str, np.ndarray] = {}
    sym_low: dict[str, np.ndarray] = {}
    sym_atr: dict[str, np.ndarray] = {}

    total_raw_signals = 0
    for sym, d in all_data.items():
        n = min(len(d['close']), len(btc_close))
        close_n = d['close'][:n]
        high_n = d['high'][:n]
        low_n = d['low'][:n]
        vol_n = d['volume'][:n]
        taker_n = d['taker_buy'][:n]
        btc_n = btc_close[:n]

        dirs, confs, moves = scan_entries(
            close_n, high_n, low_n, vol_n, taker_n, btc_n,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
        )

        # MTF filter
        c4h = all_4h[sym]
        dirs, confs, moves = apply_mtf_filter(dirs, confs, moves, c4h, MTF_SMA_PERIOD)

        raw = int(np.sum(dirs != 0))
        total_raw_signals += raw

        # ATR for vol-adjusted sizing
        atr_arr = compute_atr_pct(high_n, low_n, close_n, ATR_PERIOD)

        sym_dirs[sym] = dirs
        sym_confs[sym] = confs
        sym_moves[sym] = moves
        sym_close[sym] = close_n
        sym_high[sym] = high_n
        sym_low[sym] = low_n
        sym_atr[sym] = atr_arr

    raw_longs = sum(int(np.sum(sym_dirs[s] > 0)) for s in sym_dirs)
    raw_shorts = sum(int(np.sum(sym_dirs[s] < 0)) for s in sym_dirs)
    print(f"  Raw signals: {total_raw_signals} (L:{raw_longs} S:{raw_shorts}, ratio {raw_longs/raw_shorts:.2f}:1)" if raw_shorts > 0 else f"  Raw signals: {total_raw_signals}")

    # ── Run simulation ──
    print(f"\n🚀 Running portfolio simulation (2-slot, PBT I-HybTight)...")
    eq, trades = simulate(
        sym_close, sym_high, sym_low,
        sym_dirs, sym_confs, sym_moves,
        sym_atr=sym_atr,
    )

    elapsed = time.time() - t0
    print(f"  Done. {len(trades)} trades in {elapsed:.1f}s")

    # ── Analyze ──
    print(f"\n📈 Analyzing direction asymmetry...")
    dir_stats = analyze_by_direction(trades)
    yearly = analyze_yearly(trades, len(eq))
    per_symbol = analyze_per_symbol(trades)

    # ── Print full report ──
    print_report(dir_stats, yearly, per_symbol, eq, commit, elapsed, n_loaded)

    # ── Save JSON ──
    output = {
        'commit': commit,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'config': {
            'target_retrace': TARGET_RETRACE,
            'stop_mult': STOP_MULT,
            'max_hold': MAX_HOLD,
            'min_confidence': MIN_CONFIDENCE,
            'trail_atr_mult': TRAIL_ATR_MULT,
            'runner_max_hold_mult': RUNNER_MAX_HOLD_MULT,
            'atr_mult': ATR_MULT,
            'partial_exit': PARTIAL_EXIT,
            'data_days': DATA_DAYS,
            'n_symbols': n_loaded,
        },
        'raw_signals': {
            'total': total_raw_signals,
            'long': raw_longs,
            'short': raw_shorts,
            'ls_ratio': round(raw_longs / raw_shorts, 2) if raw_shorts > 0 else 0,
        },
        'direction_stats': dir_stats,
        'yearly': yearly,
        'per_symbol': per_symbol,
        'portfolio': {
            'total_return_pct': round((eq[-1] / INITIAL_CAPITAL - 1.0) * 100, 1),
            'final_balance': round(eq[-1], 2),
        },
        'elapsed': round(elapsed, 1),
    }

    out_path = Path(__file__).parent.parent / 'logs' / 'direction_asymmetry_results.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {out_path}")


if __name__ == '__main__':
    main()
