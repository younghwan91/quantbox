#!/usr/bin/env python3
"""
Baseline 6-Year Backtest — Current Production (v12.4.1 HybTight)
=================================================================
Runs EXACTLY the current production entry logic (Risk-Off only, conf base=0,
min_confidence=3) over 6 years (2190 days) of cached data.

Key differences from R5_6yr.py:
  1. Risk-Off only entry (no Risk-On path) — v12.4.1
  2. Confidence base = 0 (not 1) — v12.4.1 parity fix
  3. min_confidence = 3 (production .env)
  4. target_retrace = 1.5 (HybTight)
  5. stop_mult = 0.382 (Fib stop)
  6. max_hold = 25 (HybTight)
  7. ATR mult = 2.0, floor = 0.008 (v12.2)
  8. Partial exit + breakeven + runner (HybTight)
  9. trail_atr_mult = 1.0 (HybTight)
  10. 2 slots, steep cap, soft equity, vol-adjusted sizing

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/baseline_6yr_backtest.py
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

from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
    compute_atr_pct,
)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'cache_extended'
DATA_DAYS = 2190  # 6 years

# ═══════════════════════════════════════════════════════════════════
# Production parameters (v12.4.1 HybTight) — from .env
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032
TARGET_RETRACE = 1.5          # v12.4 HybTight (was 0.382)
STOP_MULT = 0.382             # v12.3 Fib stop
MAX_HOLD = 25                 # v12.4 HybTight (was 20)
STOP_DECAY = 0.0
BTC_MOM_PERIOD = 48
BTC_REGIME_PERIOD = 96
ASSET_TREND_PERIOD = 192
MIN_CONFIDENCE = 3            # v12.3 production (was 2)
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
ATR_MULT = 2.0                # v12.2 (was 2.5)
THR_FLOOR = 0.008             # v12.2 (was 0.01)
THR_CAP = 0.06
MTF_SMA_PERIOD = 12

# HybTight exit params
TRAIL_ACTIVATION = 0.5
TRAIL_FACTOR = 0.3
PARTIAL_EXIT = True
PARTIAL_FRACTION = 0.5
TRAIL_ATR_MULT = 1.0          # v12.4 HybTight (was 1.5)
BREAKEVEN_AFTER_TP1 = True
RUNNER_MAX_HOLD_MULT = 3.0    # v12.4 HybTight

# Portfolio params
MAX_SLOTS = 2
INITIAL_CAPITAL = 2095.0

# Sizing
STEEP_CAP_CURVE = True
CAP_MULT_CONF2 = 0.3
CAP_MULT_CONF3 = 0.5
CAP_MULT_CONF4 = 1.5
CAP_MULT_CONF5 = 2.0
SOFT_EQUITY_SCALE = True
SOFT_EQ_DD1, SOFT_EQ_SCALE1 = 0.02, 0.80
SOFT_EQ_DD2, SOFT_EQ_SCALE2 = 0.04, 0.60
SOFT_EQ_DD3, SOFT_EQ_SCALE3 = 0.06, 0.40
VOL_ADJUSTED_SIZE = True
VOL_ADJ_MEDIAN_ATR = 0.01
VOL_ADJ_MIN = 0.4
VOL_ADJ_MAX = 2.0
DIR_DIVERSITY = True
MAX_SAME_DIR = 2


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
    matches = sorted(CACHE_DIR.glob(f'{symbol}_{interval}_{days}d_*.npz'), reverse=True)
    return matches[0] if matches else None


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
# Entry Scanner — v12.4.1 Risk-Off Only (production-identical)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_entries_risk_off_only(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray, taker_buy: np.ndarray, btc_close: np.ndarray,
    lookback: int, threshold: float,
    btc_mom_period: int, btc_regime_period: int, asset_trend_period: int,
    min_confidence: int, rsi_period: int, vol_period: int,
    stoch_enabled: bool, stoch_k_period: int, stoch_d_period: int,
    macd_enabled: bool, macd_fast: int, macd_slow: int, macd_signal_period: int,
    atr_period: int, atr_multiplier: float,
    threshold_floor: float, threshold_cap: float,
) -> tuple:
    """Scan entry opportunities using Risk-Off ONLY logic (v12.4.1).

    Key difference from R5: NO Risk-On path.
    Confidence base = 0 (not 1).
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

        # Adaptive threshold
        if atr_period > 0:
            atr_val = atr_pct[i]
            if np.isnan(atr_val):
                thr = threshold
            else:
                thr = atr_multiplier * atr_val
                if thr < threshold_floor:
                    thr = threshold_floor
                if thr > threshold_cap:
                    thr = threshold_cap
        else:
            thr = threshold

        if abs(ret) <= thr:
            continue

        # BTC momentum
        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0

        # v12.4.1: Risk-Off ONLY — confidence starts at 0
        confidence = 0
        direction = 0

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


# ═══════════════════════════════════════════════════════════════════
# MTF Filter
# ═══════════════════════════════════════════════════════════════════
def apply_mtf_filter(
    dirs: np.ndarray, confs: np.ndarray, moves: np.ndarray,
    close_4h: np.ndarray, sma_period: int = 12, ratio: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# Portfolio Simulator — v12.4.1 HybTight (partial + breakeven + runner)
# ═══════════════════════════════════════════════════════════════════
def simulate_portfolio(
    sym_close: dict[str, np.ndarray],
    sym_high: dict[str, np.ndarray],
    sym_low: dict[str, np.ndarray],
    sym_dirs: dict[str, np.ndarray],
    sym_confs: dict[str, np.ndarray],
    sym_moves: dict[str, np.ndarray],
    sym_atr: dict[str, np.ndarray],
    initial_capital: float = INITIAL_CAPITAL,
    max_slots: int = MAX_SLOTS,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
) -> dict:
    """Simulate portfolio with HybTight exit logic."""
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    balance = initial_capital
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance
    max_dd = 0.0

    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for i in range(1, n_bars):
        # ═══ Phase 1: Check exits ═══
        to_close: list[str] = []

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            h = sym_high[sym][i]
            l = sym_low[sym][i]
            ep = pos['entry_price']
            lev = pos['leverage']

            if pos['direction'] == 1:
                gain_close = (c - ep) / ep
                gain_best = (h - ep) / ep
                gain_worst = (l - ep) / ep
            else:
                gain_close = (ep - c) / ep
                gain_best = (ep - l) / ep
                gain_worst = (ep - h) / ep

            target = pos['move_size'] * TARGET_RETRACE
            time_factor = 1.0 - STOP_DECAY * (pos['hold_count'] / MAX_HOLD)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * STOP_MULT * time_factor

            # ──── Partial Exit (HybTight) ────
            if PARTIAL_EXIT and not pos.get('tp1_hit', False):
                # Pre-TP1 phase
                if gain_close >= target:
                    # TP1 hit → partial close
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain_close

                    tp1_notional = pos['notional'] * PARTIAL_FRACTION
                    remaining = pos['notional'] - tp1_notional
                    funding = lev * funding_per_bar * tp1_notional * pos['hold_count']
                    close_cost = lev * cost_per_side * tp1_notional
                    gross = gain_close * lev * tp1_notional
                    net = gross - close_cost - funding
                    balance += net

                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': 'tp1',
                        'pnl_usd': net,
                        'pnl_pct': (net / tp1_notional) * 100,
                        'bars_held': pos['hold_count'],
                    })

                    pos['notional'] = remaining
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain_close
                    pos['trail_floor'] = 0.0 if BREAKEVEN_AFTER_TP1 else -effective_stop

                elif gain_close < -effective_stop:
                    # Stop loss
                    funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                    close_cost = lev * cost_per_side * pos['notional']
                    gross = gain_close * lev * pos['notional']
                    net = gross - close_cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': 'stop',
                        'pnl_usd': net,
                        'pnl_pct': (net / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                    })
                    to_close.append(sym)

                elif pos['hold_count'] >= MAX_HOLD:
                    # Time exit (pre-TP1)
                    funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                    close_cost = lev * cost_per_side * pos['notional']
                    gross = gain_close * lev * pos['notional']
                    net = gross - close_cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': 'time',
                        'pnl_usd': net,
                        'pnl_pct': (net / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                    })
                    to_close.append(sym)

                else:
                    # Pre-TP1 trailing
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * TRAIL_ACTIVATION
                            and gain_close < pos['best_pnl'] * TRAIL_FACTOR):
                        funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                        close_cost = lev * cost_per_side * pos['notional']
                        gross = gain_close * lev * pos['notional']
                        net = gross - close_cost - funding
                        balance += net
                        trades.append({
                            'symbol': sym, 'direction': pos['direction'],
                            'confidence': pos['confidence'],
                            'exit_reason': 'trail',
                            'pnl_usd': net,
                            'pnl_pct': (net / pos['notional']) * 100,
                            'bars_held': pos['hold_count'],
                        })
                        to_close.append(sym)

            elif PARTIAL_EXIT and pos.get('tp1_hit', False):
                # Post-TP1 runner
                if gain_close > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain_close

                atr_val = 0.02
                a_arr = sym_atr.get(sym)
                if a_arr is not None and i < len(a_arr) and not np.isnan(a_arr[i]):
                    atr_val = float(a_arr[i])

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
                    bars_since_tp1 = i - pos.get('tp1_bar', pos['entry_bar'])
                    funding = lev * funding_per_bar * pos['notional'] * bars_since_tp1
                    close_cost = lev * cost_per_side * pos['notional']
                    gross = gain_close * lev * pos['notional']
                    net = gross - close_cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': exit_reason,
                        'pnl_usd': net,
                        'pnl_pct': (net / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                    })
                    to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Phase 2: New entries ═══
        n_active = len(positions)
        if n_active < max_slots:
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

                # Direction diversity
                if DIR_DIVERSITY:
                    same_dir = sum(
                        1 for p in positions.values()
                        if p['direction'] == direction
                    )
                    if same_dir >= MAX_SAME_DIR:
                        continue

                candidates.append((sym, direction, confidence, move_size))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size in candidates:
                if len(positions) >= max_slots:
                    break

                per_slot = initial_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, initial_capital * 0.5)
                if per_slot < 10.0:
                    continue

                # Steep cap curve
                if STEEP_CAP_CURVE:
                    if confidence >= 5:
                        per_slot *= CAP_MULT_CONF5
                    elif confidence >= 4:
                        per_slot *= CAP_MULT_CONF4
                    elif confidence == 3:
                        per_slot *= CAP_MULT_CONF3
                    else:
                        per_slot *= CAP_MULT_CONF2
                    per_slot = min(per_slot, initial_capital * 0.5)

                # Vol-adjusted sizing
                if VOL_ADJUSTED_SIZE:
                    a_arr = sym_atr.get(sym)
                    if a_arr is not None and i < len(a_arr) and not np.isnan(a_arr[i]) and a_arr[i] > 0:
                        vol_factor = VOL_ADJ_MEDIAN_ATR / a_arr[i]
                        vol_factor = max(VOL_ADJ_MIN, min(vol_factor, VOL_ADJ_MAX))
                        per_slot *= vol_factor
                        per_slot = min(per_slot, initial_capital * 0.5)

                # Soft equity scaling
                if SOFT_EQUITY_SCALE:
                    eq_now = equity_curve[i - 1] if i > 0 else initial_capital
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    if dd_pct > SOFT_EQ_DD3:
                        per_slot *= SOFT_EQ_SCALE3
                    elif dd_pct > SOFT_EQ_DD2:
                        per_slot *= SOFT_EQ_SCALE2
                    elif dd_pct > SOFT_EQ_DD1:
                        per_slot *= SOFT_EQ_SCALE1

                trade_lev = EFFECTIVE_LEV
                entry_cost = trade_lev * cost_per_side * per_slot
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
                    'leverage': trade_lev,
                }

        # ═══ Phase 3: Update equity curve ═══
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

    return _compute_stats(equity_curve, trades, initial_capital, max_dd)


def _compute_stats(eq: np.ndarray, trades: list, capital: float, max_dd: float) -> dict:
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
        'equity_curve': eq,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    commit = _get_git_commit()

    # Use OLD whitelist for cached data (new symbols don't have 6yr data)
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

    print("=" * 110)
    print("  BASELINE 6-YEAR BACKTEST — v12.4.1 HybTight (Current Production)")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Data: {DATA_DAYS}d (6 years)")
    print(f"  Entry: Risk-Off only (v12.4.1), conf base=0, min_conf={MIN_CONFIDENCE}")
    print(f"  Exit: HybTight (target={TARGET_RETRACE}×, stop={STOP_MULT}×, "
          f"hold={MAX_HOLD}, partial+BE+runner)")
    print(f"  Slots: {MAX_SLOTS}, Leverage: {EFFECTIVE_LEV}×")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}")
    print("=" * 110)

    # ── Load data ──
    print(f"\n📊 Loading 50-symbol data ({DATA_DAYS}d)...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']
    print(f"  BTC: {len(btc_close):,} bars ({len(btc_close)/96:.0f} days)")

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    data_lengths: dict[str, int] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym)
        if d is None:
            print(f"  SKIP {sym}: no {DATA_DAYS}d data")
            continue
        all_data[sym] = d
        data_lengths[sym] = len(d['close'])
        h4 = load_4h(sym)
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    if data_lengths:
        min_bars = min(data_lengths.values())
        max_bars = max(data_lengths.values())
        print(f"  Bar range: {min_bars:,} ({min_bars/96:.0f}d) — {max_bars:,} ({max_bars/96:.0f}d)")

    # ── Scan entries ──
    print(f"\n🔍 Scanning entries (Risk-Off only, conf≥{MIN_CONFIDENCE})...")
    t0 = time.time()

    sym_dirs: dict[str, np.ndarray] = {}
    sym_confs: dict[str, np.ndarray] = {}
    sym_moves: dict[str, np.ndarray] = {}
    sym_close: dict[str, np.ndarray] = {}
    sym_high: dict[str, np.ndarray] = {}
    sym_low: dict[str, np.ndarray] = {}

    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        c, h, l, v, t = d['close'][:n], d['high'][:n], d['low'][:n], d['volume'][:n], d['taker_buy'][:n]
        b = btc_close[:n]

        dirs, confs, moves = scan_entries_risk_off_only(
            c, h, l, v, t, b,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
        )

        close_4h = all_4h.get(sym)
        if close_4h is not None:
            dirs, confs, moves = apply_mtf_filter(
                dirs, confs, moves, close_4h, MTF_SMA_PERIOD, ratio=16)

        sym_dirs[sym] = dirs
        sym_confs[sym] = confs
        sym_moves[sym] = moves
        sym_close[sym] = c
        sym_high[sym] = h
        sym_low[sym] = l

    # Pre-compute ATR per symbol
    sym_atr: dict[str, np.ndarray] = {}
    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        atr = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)
        sym_atr[sym] = atr

    # Left-pad all arrays to same length
    max_n = max(len(sym_close[s]) for s in sym_close)
    for sym in list(sym_close.keys()):
        for d_ref, fill in [
            (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
            (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
            (sym_atr, 'nan'),
        ]:
            arr = d_ref[sym]
            if len(arr) < max_n:
                pad_n = max_n - len(arr)
                if fill == 'first':
                    fv = arr[0]
                elif fill == 'nan':
                    fv = np.nan
                else:
                    fv = 0.0
                d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

    total_entries = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    scan_time = time.time() - t0
    print(f"  Total entry opportunities: {total_entries:,} ({scan_time:.1f}s)")
    print(f"  Simulation window: {max_n:,} bars ({max_n/96:.0f} days)")

    # ── Simulate ──
    print(f"\n⚡ Running portfolio simulation...")
    t0 = time.time()
    result = simulate_portfolio(
        sym_close, sym_high, sym_low,
        sym_dirs, sym_confs, sym_moves,
        sym_atr=sym_atr,
    )
    elapsed = time.time() - t0

    # ── Report ──
    reasons_str = '  '.join(f"{k}={v}" for k, v in sorted(result['exit_reasons'].items()))

    print(f"\n{'=' * 110}")
    print(f"  BASELINE RESULTS — v12.4.1 HybTight × 6yr ({DATA_DAYS}d)")
    print(f"{'=' * 110}\n")
    print(f"  Return:       {result['total_return_pct']:+,.1f}%")
    print(f"  Final Balance:${result['final_balance']:,.2f} (from ${INITIAL_CAPITAL:,.0f})")
    print(f"  Max DD:       {result['max_dd_pct']:.1f}%")
    print(f"  Sharpe:       {result['sharpe']:.2f}")
    print(f"  Calmar:       {result['calmar']:.1f}")
    print(f"  Trades:       {result['n_trades']:,} ({result['trades_per_day']}/day)")
    print(f"  Avg Bars:     {result['avg_bars_held']:.1f}")
    print(f"  Win Rate:     {result['win_rate']:.1f}%")
    print(f"  Actual R:R:   {result['actual_rr']:.2f}")
    print(f"  Profit Factor:{result['profit_factor']:.2f}")
    print(f"  Avg Win:      {result['avg_win_pct']:+.3f}%")
    print(f"  Avg Loss:     {result['avg_loss_pct']:.3f}%")
    print(f"  Avg PnL:      {result['avg_pnl_pct']:+.3f}%")
    print(f"  Exit Reasons: {reasons_str}")
    print(f"  Elapsed:      {elapsed:.1f}s")

    # ── Yearly breakdown ──
    eq = result['equity_curve']
    bars_per_year = 96 * 365
    n_years = len(eq) // bars_per_year
    if n_years > 0:
        print(f"\n  {'Year':<8} {'Start':>10} {'End':>10} {'Return':>10} {'DD':>8}")
        print(f"  {'-'*50}")
        for y in range(n_years):
            s_idx = y * bars_per_year
            e_idx = min((y + 1) * bars_per_year, len(eq)) - 1
            start_eq = eq[s_idx]
            end_eq = eq[e_idx]
            yr_ret = (end_eq / start_eq - 1.0) * 100 if start_eq > 0 else 0.0

            # Year max DD
            yr_peak = start_eq
            yr_max_dd = 0.0
            for j in range(s_idx, e_idx + 1):
                if eq[j] > yr_peak:
                    yr_peak = eq[j]
                dd = (yr_peak - eq[j]) / yr_peak if yr_peak > 0 else 0.0
                if dd > yr_max_dd:
                    yr_max_dd = dd

            print(f"  Y{y+1:<6} ${start_eq:>9,.0f} ${end_eq:>9,.0f} {yr_ret:>+9.1f}% {yr_max_dd*100:>7.1f}%")

    # ── Save ──
    save_data = {k: v for k, v in result.items() if k != 'equity_curve'}
    save_data['commit'] = commit
    save_data['date'] = datetime.now(timezone.utc).isoformat()
    save_data['data_days'] = DATA_DAYS
    save_data['n_symbols'] = len(available)
    save_data['config'] = {
        'entry': 'risk_off_only_v12.4.1',
        'conf_base': 0,
        'min_confidence': MIN_CONFIDENCE,
        'target_retrace': TARGET_RETRACE,
        'stop_mult': STOP_MULT,
        'max_hold': MAX_HOLD,
        'atr_mult': ATR_MULT,
        'thr_floor': THR_FLOOR,
        'thr_cap': THR_CAP,
        'partial_exit': PARTIAL_EXIT,
        'trail_atr_mult': TRAIL_ATR_MULT,
        'breakeven_after_tp1': BREAKEVEN_AFTER_TP1,
        'runner_max_hold_mult': RUNNER_MAX_HOLD_MULT,
        'max_slots': MAX_SLOTS,
        'leverage': EFFECTIVE_LEV,
        'initial_capital': INITIAL_CAPITAL,
    }

    output_path = Path(__file__).resolve().parent.parent.parent / 'logs' / 'baseline_6yr_results.json'
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 110}")
    print("  DONE — BASELINE 6-YEAR BACKTEST COMPLETE")
    print(f"{'=' * 110}")


if __name__ == '__main__':
    main()
