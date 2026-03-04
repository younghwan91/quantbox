#!/usr/bin/env python3
"""
R22: Zero-Lookahead Alpha Diagnostic
======================================
Purpose: After fixing the 4h MTF lookahead bias (v12.6.1), determine
whether the rubber band mean-reversion signal has genuine alpha.

Key questions:
  Q1: Does the raw signal have edge at ZERO cost?
  Q2: How much edge does MTF (completed bars) add?
  Q3: At what cost level does the edge disappear?
  Q4: Can exit optimization improve the edge?
  Q5: Does changing entry filters help?

Approach:
  - Phase 1: Independent per-trade analysis (no portfolio constraints)
    For every entry signal, simulate the trade and record outcome.
    This measures RAW signal quality without portfolio effects.
  - Phase 2: Portfolio-level sim with max_slots=2 for realism.

All variants use 1x leverage to isolate signal quality from leverage
amplification. Leverage only scales existing edge — can't create it.

Data: 6yr (2190d), 50 symbols, cached in data/cache_extended/.
Lookahead: ZERO. MTF uses last completed 4h bar only.

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/alpha_diagnostic_r22.py
"""

import os
import sys
import time
import json
import numpy as np
import numba
from dataclasses import dataclass, field
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
DATA_DAYS = 2190

# ═══════════════════════════════════════════════════════════════════
# Production entry parameters (v12.5.3 baseline)
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032
BTC_MOM_PERIOD = 48
BTC_REGIME_PERIOD = 96
ASSET_TREND_PERIOD = 192
MIN_CONFIDENCE = 3
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
FUNDING_PER_BAR = 0.01 / 100 / 32  # 0.01% per 8h = per 32 bars
BARS_PER_YEAR = 96 * 365

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
class Variant:
    name: str
    desc: str
    # Entry
    min_confidence: int = 3
    use_mtf: bool = False
    mtf_sma_period: int = 12
    use_adaptive_thr: bool = True       # ATR-based adaptive threshold
    fixed_threshold: float = 0.032      # used when use_adaptive_thr=False
    risk_off_only: bool = True          # True=current (Risk-Off only), False=both regimes
    # Exit
    target_retrace: float = 1.5
    stop_mult: float = 0.382
    max_hold: int = 25
    use_pbt: bool = True                # PBT partial exit
    partial_frac: float = 0.5
    be_after_partial: bool = True
    trail_atr_mult: float = 1.0
    runner_mh_mult: float = 3.0
    pre_tp1_activation: float = 0.5
    pre_tp1_trail_factor: float = 0.3
    # Cost & leverage
    cost_per_side: float = 0.0          # 0 = zero cost
    leverage: float = 1.0
    # Portfolio
    max_slots: int = 2


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════
def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    pattern = f"{symbol}_{interval}_{days}d_"
    for f in sorted(CACHE_DIR.glob(f"{pattern}*.npz"), reverse=True):
        return f
    return None


def load_symbol(symbol: str) -> dict | None:
    """Load 15m + 4h data for a symbol."""
    f15 = _find_cache(symbol, '15m', DATA_DAYS)
    f4h = _find_cache(symbol, '4h', DATA_DAYS)
    if f15 is None:
        return None
    d = np.load(f15)
    result = {
        'close': d['close'].astype(np.float64),
        'high': d['high'].astype(np.float64),
        'low': d['low'].astype(np.float64),
        'volume': d['volume'].astype(np.float64),
        'taker_buy': d['taker_buy'].astype(np.float64),
    }
    if f4h is not None:
        d4 = np.load(f4h)
        result['close_4h'] = d4['close'].astype(np.float64)
    return result


# ═══════════════════════════════════════════════════════════════════
# Entry Scanner
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
    risk_off_only,
):
    """Production-identical entry scanner with Risk-Off toggle."""
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

        if abs(ret) <= thr:
            continue

        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
        direction = 0
        confidence = 0

        if risk_off_only:
            # ═══ RISK-OFF ONLY (v12.4.1 production) ═══
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
        else:
            # ═══ BOTH REGIMES (Risk-On + Risk-Off) ═══
            is_risk_on = btc_close[i] > btc_sma[i] if not np.isnan(btc_sma[i]) else False

            if is_risk_on:
                if not np.isnan(asset_sma[i]):
                    is_uptrend = close[i] > asset_sma[i]
                    if ret < -thr and is_uptrend:
                        direction = 1
                        if rsi[i] < 40: confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]: confidence += 1
                        if taker_ratio[i] < 0.45: confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02: confidence += 1
                        if stoch_enabled and i > 0:
                            if stoch_k[i] < 20 and stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]:
                                confidence += 1
                        if macd_enabled and i > 0:
                            if macd_hist[i] > macd_hist[i-1] and macd_hist[i-1] < 0:
                                confidence += 1
                        if confidence < min_confidence:
                            direction = 0

                    elif ret > thr and not is_uptrend:
                        direction = -1
                        if rsi[i] > 60: confidence += 1
                        if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]: confidence += 1
                        if taker_ratio[i] > 0.55: confidence += 1
                        dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                        if dist < 0.02: confidence += 1
                        if stoch_enabled and i > 0:
                            if stoch_k[i] > 80 and stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]:
                                confidence += 1
                        if macd_enabled and i > 0:
                            if macd_hist[i] < macd_hist[i-1] and macd_hist[i-1] > 0:
                                confidence += 1
                        if confidence < min_confidence:
                            direction = 0
            else:
                # Risk-Off (same as risk_off_only path)
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
    """Zero-lookahead MTF filter: uses last COMPLETED 4h bar only."""
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
        j_raw = min(i // ratio, n_4h - 1)
        j = j_raw - 1 if j_raw > 0 else 0
        if j < sma_period or np.isnan(sma_4h[j]):
            continue
        trend_up = close_4h[j] > sma_4h[j]
        if dirs[i] > 0 and not trend_up:
            od[i] = 0.0; oc[i] = 0.0; om[i] = 0.0
        elif dirs[i] < 0 and trend_up:
            od[i] = 0.0; oc[i] = 0.0; om[i] = 0.0
    return od, oc, om


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Per-Trade Independent Analysis
# ═══════════════════════════════════════════════════════════════════
def analyze_raw_trades(
    sym_close: dict, sym_high: dict, sym_low: dict,
    sym_dirs: dict, sym_confs: dict, sym_moves: dict,
    target_retrace: float = 1.5,
    stop_mult: float = 0.382,
    max_hold: int = 25,
) -> list[dict]:
    """Simulate each trade independently (no portfolio constraints).

    For each entry signal, simulate forward until exit.
    This gives us the UNIVERSE of all possible trades.
    """
    trades = []
    symbols = sorted(sym_close.keys())

    for sym in symbols:
        close = sym_close[sym]
        high = sym_high[sym]
        low = sym_low[sym]
        dirs = sym_dirs[sym]
        confs = sym_confs[sym]
        movs = sym_moves[sym]
        n = len(close)

        for entry_bar in range(n):
            if dirs[entry_bar] == 0.0:
                continue

            d = int(dirs[entry_bar])
            conf = int(confs[entry_bar])
            move = float(movs[entry_bar])
            ep = close[entry_bar]
            target = move * target_retrace
            stop = move * stop_mult

            best_pnl = 0.0
            exit_bar = -1
            exit_reason = 'time'
            exit_gain = 0.0
            mfe = 0.0  # max favorable excursion
            mae = 0.0  # max adverse excursion

            for j in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
                bars_held = j - entry_bar

                if d == 1:
                    gain = (close[j] - ep) / ep
                    worst = (low[j] - ep) / ep
                    best = (high[j] - ep) / ep
                else:
                    gain = (ep - close[j]) / ep
                    worst = (ep - high[j]) / ep
                    best = (ep - low[j]) / ep

                if best > mfe:
                    mfe = best
                if worst < mae:
                    mae = worst
                if best > best_pnl:
                    best_pnl = best

                # SL priority (intra-bar)
                if worst < -stop:
                    exit_bar = j
                    exit_reason = 'stop'
                    exit_gain = -stop  # assume stopped at exact level
                    break
                elif best >= target:
                    exit_bar = j
                    exit_reason = 'target'
                    exit_gain = target
                    break
                # Trailing
                elif best_pnl > target * 0.5 and worst < best_pnl * 0.3:
                    exit_bar = j
                    exit_reason = 'trail'
                    exit_gain = best_pnl * 0.3
                    break
                elif bars_held >= max_hold:
                    exit_bar = j
                    exit_reason = 'time'
                    exit_gain = gain
                    break

            if exit_bar == -1:
                # Ran out of data
                continue

            trades.append({
                'symbol': sym,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'direction': d,
                'confidence': conf,
                'move_size': move,
                'bars_held': exit_bar - entry_bar,
                'raw_gain': exit_gain,
                'mfe': mfe,
                'mae': mae,
                'exit_reason': exit_reason,
            })

    return trades


def compute_trade_stats(trades: list[dict], cost_per_side: float = 0.0,
                        leverage: float = 1.0) -> dict:
    """Compute statistics from raw trades under given cost/leverage."""
    if not trades:
        return {'n': 0, 'wr': 0, 'pf': 0, 'rr': 0, 'ev': 0, 'sharpe': 0}

    pnls = []
    for t in trades:
        raw = t['raw_gain']
        cost = 2 * cost_per_side  # roundtrip
        funding = FUNDING_PER_BAR * t['bars_held']
        net = (raw - cost - funding) * leverage
        pnls.append(net)

    pnls = np.array(pnls)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n * 100 if n > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    pf = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else 999
    ev = float(np.mean(pnls))
    total_ret = float(np.sum(pnls)) * 100
    sharpe = float(np.mean(pnls) / np.std(pnls)) * np.sqrt(BARS_PER_YEAR) if np.std(pnls) > 0 else 0

    # Break-even cost: find cost_per_side where EV = 0
    if cost_per_side == 0:
        raw_gains = np.array([t['raw_gain'] for t in trades])
        funding_costs = np.array([FUNDING_PER_BAR * t['bars_held'] for t in trades])
        avg_raw = np.mean(raw_gains)
        avg_fund = np.mean(funding_costs)
        be_cost = (avg_raw - avg_fund) / 2  # 2 sides
        be_cost = max(be_cost, 0)
    else:
        be_cost = 0

    return {
        'n': n,
        'wr': wr,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100,
        'rr': rr,
        'pf': pf,
        'ev': ev * 100,
        'total_ret': total_ret,
        'sharpe': sharpe,
        'be_cost_bps': be_cost * 10000 if cost_per_side == 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════
def simulate_portfolio(
    sym_close: dict, sym_high: dict, sym_low: dict,
    sym_dirs: dict, sym_confs: dict, sym_moves: dict,
    sym_atr: dict,
    v: Variant,
) -> dict:
    """Portfolio sim with max_slots, PBT, configurable cost/leverage."""
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    balance = INITIAL_CAPITAL
    equity = np.zeros(n_bars)
    equity[0] = balance
    peak = balance
    max_dd = 0.0
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    cost = v.cost_per_side
    lev = v.leverage

    for i in range(1, n_bars):
        to_close: list[str] = []

        # ═══ Manage existing positions ═══
        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            h = sym_high[sym][i]
            lo = sym_low[sym][i]
            ep = pos['entry_price']
            d = pos['direction']

            if d == 1:
                gain = (c - ep) / ep
                worst = (lo - ep) / ep
                best = (h - ep) / ep
            else:
                gain = (ep - c) / ep
                worst = (ep - h) / ep
                best = (ep - lo) / ep

            if best > pos['best_pnl']:
                pos['best_pnl'] = best

            target = pos['move_size'] * v.target_retrace
            stop_val = pos['move_size'] * v.stop_mult

            # ═══ PBT Phase 1: Pre-TP1 ═══
            if v.use_pbt and not pos.get('tp1_hit', False):
                if worst < -stop_val:
                    _close_pos(balance, trades, pos, sym, -stop_val, lev, cost, 'stop', to_close, i)
                    balance = trades[-1]['_b']
                elif best >= target:
                    # TP1 hit — partial close
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = best
                    tp1_not = pos['notional'] * v.partial_frac
                    rem_not = pos['notional'] - tp1_not
                    funding = lev * FUNDING_PER_BAR * tp1_not * pos['hold_count']
                    c_cost = lev * cost * tp1_not
                    net = target * lev * tp1_not - c_cost - funding
                    balance += net
                    trades.append({
                        'symbol': sym, 'direction': d,
                        'confidence': pos['confidence'],
                        'exit_reason': 'tp1',
                        'pnl_usd': net,
                        'pnl_pct': (net / tp1_not) * 100 if tp1_not > 0 else 0,
                        'bars_held': pos['hold_count'],
                        '_b': balance,
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = best
                    pos['trail_floor'] = 0.0 if v.be_after_partial else -stop_val
                elif pos['best_pnl'] > target * v.pre_tp1_activation and worst < pos['best_pnl'] * v.pre_tp1_trail_factor:
                    _close_pos(balance, trades, pos, sym, pos['best_pnl'] * v.pre_tp1_trail_factor, lev, cost, 'trail', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= v.max_hold:
                    _close_pos(balance, trades, pos, sym, gain, lev, cost, 'time', to_close, i)
                    balance = trades[-1]['_b']

            # ═══ PBT Phase 2: Runner ═══
            elif v.use_pbt and pos.get('tp1_hit', False):
                if best > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = best

                atr_val = 0.02
                aa = sym_atr.get(sym)
                if aa is not None and i < len(aa) and not np.isnan(aa[i]):
                    atr_val = float(aa[i])

                trail_stop = pos['highest_since_tp1'] - v.trail_atr_mult * atr_val
                floor_val = pos.get('trail_floor', 0.0)
                if trail_stop < floor_val:
                    trail_stop = floor_val

                if worst <= trail_stop:
                    _runner_close(balance, trades, pos, sym, trail_stop, lev, cost, 'runner_sl', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= v.max_hold * v.runner_mh_mult:
                    _runner_close(balance, trades, pos, sym, gain, lev, cost, 'runner_time', to_close, i)
                    balance = trades[-1]['_b']

            # ═══ Non-PBT simple exit ═══
            else:
                if worst < -stop_val:
                    _close_pos(balance, trades, pos, sym, -stop_val, lev, cost, 'stop', to_close, i)
                    balance = trades[-1]['_b']
                elif best >= target:
                    _close_pos(balance, trades, pos, sym, target, lev, cost, 'target', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['best_pnl'] > target * v.pre_tp1_activation and worst < pos['best_pnl'] * v.pre_tp1_trail_factor:
                    _close_pos(balance, trades, pos, sym, pos['best_pnl'] * v.pre_tp1_trail_factor, lev, cost, 'trail', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= v.max_hold:
                    _close_pos(balance, trades, pos, sym, gain, lev, cost, 'time', to_close, i)
                    balance = trades[-1]['_b']

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Entry ═══
        if len(positions) < v.max_slots:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                d_arr = sym_dirs[sym]
                if i >= len(d_arr) or d_arr[i] == 0.0:
                    continue
                candidates.append((
                    sym,
                    int(d_arr[i]),
                    int(sym_confs[sym][i]),
                    float(sym_moves[sym][i]),
                ))

            # Sort by confidence desc, move desc
            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size in candidates:
                if len(positions) >= v.max_slots:
                    break

                per_slot = INITIAL_CAPITAL / max(1, len(positions) + 1)
                per_slot = min(per_slot, INITIAL_CAPITAL * 0.5)
                if per_slot < 10.0:
                    continue

                entry_cost = lev * cost * per_slot
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
                }

        # Equity update
        total_unr = 0.0
        for sym, pos in positions.items():
            c = sym_close[sym][i]
            ep = pos['entry_price']
            g = (c - ep) / ep if pos['direction'] == 1 else (ep - c) / ep
            total_unr += g * lev * pos['notional']
        equity[i] = balance + total_unr
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Stats
    total_pnl = (equity[n_bars - 1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    sum_win = sum(t['pnl_usd'] for t in wins)
    sum_loss = abs(sum(t['pnl_usd'] for t in losses))
    pf = sum_win / sum_loss if sum_loss > 0 else 999

    # Yearly returns
    yearly_rets = []
    for yr in range(6):
        s = yr * BARS_PER_YEAR // 6  # approximate
        e = (yr + 1) * BARS_PER_YEAR // 6
        s = min(s, n_bars - 1)
        e = min(e, n_bars - 1)
        if s < e:
            yr_ret = (equity[e] - equity[s]) / max(equity[s], 1) * 100
        else:
            yr_ret = 0
        yearly_rets.append(yr_ret)

    # Per-exit-reason breakdown
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    return {
        'total_ret': total_pnl,
        'max_dd': max_dd * 100,
        'wr': wr,
        'pf': pf,
        'n_trades': n_trades,
        'sharpe': _calc_sharpe(equity),
        'reasons': reasons,
        'yearly_rets': yearly_rets,
    }


def _close_pos(balance, trades, pos, sym, gain, lev, cost_ps, reason, to_close, bar_idx):
    funding = lev * FUNDING_PER_BAR * pos['notional'] * pos['hold_count']
    c_cost = lev * cost_ps * pos['notional']
    net = gain * lev * pos['notional'] - c_cost - funding
    new_b = balance + net
    trades.append({
        'symbol': sym, 'direction': pos['direction'],
        'confidence': pos['confidence'],
        'exit_reason': reason,
        'pnl_usd': net,
        'pnl_pct': (net / pos['notional']) * 100 if pos['notional'] > 0 else 0,
        'bars_held': pos['hold_count'],
        '_b': new_b,
    })
    to_close.append(sym)


def _runner_close(balance, trades, pos, sym, gain, lev, cost_ps, reason, to_close, bar_idx):
    tp1_bar = pos.get('tp1_bar', pos['entry_bar'])
    bars_since = bar_idx - tp1_bar
    funding = lev * FUNDING_PER_BAR * pos['notional'] * bars_since
    c_cost = lev * cost_ps * pos['notional']
    net = gain * lev * pos['notional'] - c_cost - funding
    new_b = balance + net
    trades.append({
        'symbol': sym, 'direction': pos['direction'],
        'confidence': pos['confidence'],
        'exit_reason': reason,
        'pnl_usd': net,
        'pnl_pct': (net / pos['notional']) * 100 if pos['notional'] > 0 else 0,
        'bars_held': pos['hold_count'],
        '_b': new_b,
    })
    to_close.append(sym)


def _calc_sharpe(equity: np.ndarray) -> float:
    rets = np.diff(equity) / np.maximum(equity[:-1], 1.0)
    if len(rets) < 2 or np.std(rets) == 0:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * np.sqrt(BARS_PER_YEAR))


# ═══════════════════════════════════════════════════════════════════
# Variant Definitions
# ═══════════════════════════════════════════════════════════════════
def build_variants() -> list[Variant]:
    variants = []

    # ── Group 0: COST SENSITIVITY (same signal, vary cost) ──
    # Does the raw signal have edge?
    variants.append(Variant(
        name='A-ZeroCost', desc='Zero cost, no MTF, 1x (pure signal)',
        cost_per_side=0.0, leverage=1.0, use_mtf=False))
    variants.append(Variant(
        name='B-Maker', desc='Maker 0.02%/side, no MTF, 1x',
        cost_per_side=0.0002, leverage=1.0, use_mtf=False))
    variants.append(Variant(
        name='C-Full', desc='Full 0.04%/side, no MTF, 1x',
        cost_per_side=0.0004, leverage=1.0, use_mtf=False))
    variants.append(Variant(
        name='D-Prod', desc='Full cost, MTF, 3.75x (production)',
        cost_per_side=0.0004, leverage=3.75, use_mtf=True))

    # ── Group 1: MTF FILTER VALUE (all zero cost, 1x) ──
    variants.append(Variant(
        name='E-MTF12', desc='MTF SMA(12) completed, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=True, mtf_sma_period=12))
    variants.append(Variant(
        name='F-MTF24', desc='MTF SMA(24) completed, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=True, mtf_sma_period=24))
    variants.append(Variant(
        name='G-MTF6', desc='MTF SMA(6) completed, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=True, mtf_sma_period=6))

    # ── Group 2: ENTRY QUALITY (all zero cost, no MTF, 1x) ──
    variants.append(Variant(
        name='H-Conf2', desc='min_conf=2, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, min_confidence=2))
    variants.append(Variant(
        name='I-Conf4', desc='min_conf=4, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, min_confidence=4))
    variants.append(Variant(
        name='J-Both', desc='Both regimes (Risk-On+Off), zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, risk_off_only=False))
    variants.append(Variant(
        name='K-Fixed5', desc='Fixed 5% threshold, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False,
        use_adaptive_thr=False, fixed_threshold=0.05))
    variants.append(Variant(
        name='L-Fixed3', desc='Fixed 3.2% threshold, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False,
        use_adaptive_thr=False, fixed_threshold=0.032))

    # ── Group 3: EXIT STRATEGY (all zero cost, no MTF, 1x) ──
    variants.append(Variant(
        name='M-NoPBT', desc='No PBT (simple tgt/stop/trail/time), 0 cost',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, use_pbt=False))
    variants.append(Variant(
        name='N-TightSL', desc='Tight stop 0.20, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, stop_mult=0.20))
    variants.append(Variant(
        name='O-WideTgt', desc='Wide target 2.5x, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, target_retrace=2.5))
    variants.append(Variant(
        name='P-Quick', desc='Quick exit max_hold=12, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, max_hold=12))
    variants.append(Variant(
        name='Q-Patient', desc='Patient max_hold=50, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, max_hold=50))
    variants.append(Variant(
        name='R-TightTgt', desc='Tight target 0.5x, zero cost, 1x',
        cost_per_side=0.0, leverage=1.0, use_mtf=False, target_retrace=0.5))

    # ── Group 4: COMBINED BEST (if edge found) ──
    # Maker cost + best entry + best exit
    variants.append(Variant(
        name='S-C4Maker', desc='Conf4 + maker cost, 1x',
        cost_per_side=0.0002, leverage=1.0, use_mtf=False, min_confidence=4))
    variants.append(Variant(
        name='T-BothMkr', desc='Both regimes + maker, 1x',
        cost_per_side=0.0002, leverage=1.0, use_mtf=False, risk_off_only=False))

    return variants


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    import subprocess
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = 'unknown'

    print('=' * 80)
    print(f'  R22: ZERO-LOOKAHEAD ALPHA DIAGNOSTIC')
    print(f'  Commit: {commit}')
    print(f'  Date: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print(f'  Data: {DATA_DAYS}d (6yr), {len(SYMBOLS_50)} symbols')
    print(f'  Lookahead: ZERO (MTF uses last completed 4h bar)')
    print('=' * 80)

    # ── Load Data ──
    print('\n📊 Loading data...')
    sym_data = {}
    btc_data = None
    for sym in SYMBOLS_50:
        d = load_symbol(sym)
        if d is not None:
            sym_data[sym] = d
            if sym == 'BTCUSDT':
                btc_data = d
    print(f'  Loaded {len(sym_data)} symbols')

    if btc_data is None:
        print('  ❌ BTCUSDT not found. Aborting.')
        return

    btc_close = btc_data['close']

    # ── Pad to same length ──
    max_len = max(len(sym_data[s]['close']) for s in sym_data)
    for sym in sym_data:
        d = sym_data[sym]
        n = len(d['close'])
        if n < max_len:
            pad = max_len - n
            for k in ['close', 'high', 'low', 'volume', 'taker_buy']:
                d[k] = np.concatenate([np.full(pad, d[k][0]), d[k]])
            if 'close_4h' in d:
                n4 = len(d['close_4h'])
                pad4 = (max_len // 16 + 1) - n4
                if pad4 > 0:
                    d['close_4h'] = np.concatenate([np.full(pad4, d['close_4h'][0]), d['close_4h']])
    if len(btc_close) < max_len:
        pad = max_len - len(btc_close)
        btc_close = np.concatenate([np.full(pad, btc_close[0]), btc_close])
        btc_data['close'] = btc_close
    print(f'  Padded to {max_len} bars ({max_len / 96:.0f} days)')

    # ── Pre-compute entries for all confidence levels ──
    # We'll scan at min_conf=2 and filter later
    print('\n🔍 Scanning entries...')
    t0 = time.time()

    # Scan with min_conf=2 (captures all trades for conf 2,3,4,5)
    sym_dirs_ro = {}  # Risk-Off only
    sym_confs_ro = {}
    sym_moves_ro = {}
    sym_dirs_both = {}  # Both regimes
    sym_confs_both = {}
    sym_moves_both = {}
    sym_close_dict = {}
    sym_high_dict = {}
    sym_low_dict = {}
    sym_atr_dict = {}

    # Also prepare fixed-threshold variants
    sym_dirs_fix5 = {}
    sym_confs_fix5 = {}
    sym_moves_fix5 = {}
    sym_dirs_fix3 = {}
    sym_confs_fix3 = {}
    sym_moves_fix3 = {}

    for sym in sorted(sym_data.keys()):
        d = sym_data[sym]
        c = d['close']
        h = d['high']
        lo = d['low']
        v = d['volume']
        tb = d['taker_buy']

        sym_close_dict[sym] = c
        sym_high_dict[sym] = h
        sym_low_dict[sym] = lo
        sym_atr_dict[sym] = compute_atr_pct(h, lo, c, ATR_PERIOD)

        # Risk-Off only, adaptive threshold, min_conf=2
        dirs, confs, moves = scan_entries(
            c, h, lo, v, tb, btc_close,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, 2,  # min_conf=2 to capture all
            RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            True,  # risk_off_only
        )
        sym_dirs_ro[sym] = dirs
        sym_confs_ro[sym] = confs
        sym_moves_ro[sym] = moves

        # Both regimes, adaptive threshold, min_conf=2
        dirs2, confs2, moves2 = scan_entries(
            c, h, lo, v, tb, btc_close,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, 2,
            RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            False,  # both regimes
        )
        sym_dirs_both[sym] = dirs2
        sym_confs_both[sym] = confs2
        sym_moves_both[sym] = moves2

        # Fixed 5% threshold
        dirs5, confs5, moves5 = scan_entries(
            c, h, lo, v, tb, btc_close,
            LOOKBACK, 0.05, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, 2,
            RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            0, 0, 0, 0,  # no adaptive
            True,
        )
        sym_dirs_fix5[sym] = dirs5
        sym_confs_fix5[sym] = confs5
        sym_moves_fix5[sym] = moves5

        # Fixed 3.2% threshold
        dirs3, confs3, moves3 = scan_entries(
            c, h, lo, v, tb, btc_close,
            LOOKBACK, 0.032, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, 2,
            RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            0, 0, 0, 0,
            True,
        )
        sym_dirs_fix3[sym] = dirs3
        sym_confs_fix3[sym] = confs3
        sym_moves_fix3[sym] = moves3

    # Count raw signals
    n_signals_ro = sum(int(np.sum(np.abs(sym_dirs_ro[s]) > 0)) for s in sym_dirs_ro)
    n_signals_both = sum(int(np.sum(np.abs(sym_dirs_both[s]) > 0)) for s in sym_dirs_both)
    print(f'  Risk-Off signals (conf≥2): {n_signals_ro:,}')
    print(f'  Both regime signals (conf≥2): {n_signals_both:,}')
    print(f'  Scan time: {time.time() - t0:.1f}s')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Per-Trade Independent Analysis
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('  PHASE 1: PER-TRADE INDEPENDENT ANALYSIS')
    print('  (No portfolio constraints — all trades taken independently)')
    print('═' * 80)

    def filter_by_conf(dirs, confs, moves, min_c):
        od, oc, om = dirs.copy(), confs.copy(), moves.copy()
        for i in range(len(dirs)):
            if confs[i] > 0 and confs[i] < min_c:
                od[i] = 0.0; oc[i] = 0.0; om[i] = 0.0
        return od, oc, om

    # Analyze raw trades at different confidence levels
    for min_c in [2, 3, 4]:
        fd, fc, fm = {}, {}, {}
        for sym in sym_dirs_ro:
            d, c, m = filter_by_conf(sym_dirs_ro[sym], sym_confs_ro[sym], sym_moves_ro[sym], min_c)
            fd[sym] = d; fc[sym] = c; fm[sym] = m

        raw_trades = analyze_raw_trades(
            sym_close_dict, sym_high_dict, sym_low_dict,
            fd, fc, fm,
        )

        stats_zero = compute_trade_stats(raw_trades, cost_per_side=0.0)
        stats_maker = compute_trade_stats(raw_trades, cost_per_side=0.0002)
        stats_full = compute_trade_stats(raw_trades, cost_per_side=0.0004)

        print(f'\n  ── Risk-Off, conf≥{min_c}, PBT exit (target=1.5x, stop=0.382) ──')
        print(f'  {"Cost":<12} {"N":>6} {"WR%":>6} {"AvgW%":>7} {"AvgL%":>7} '
              f'{"R:R":>5} {"PF":>6} {"EV(bp)":>7} {"BE cost(bp)":>10}')
        print(f'  {"─" * 70}')
        for label, st in [('Zero', stats_zero), ('Maker', stats_maker), ('Full', stats_full)]:
            print(f'  {label:<12} {st["n"]:>6} {st["wr"]:>6.1f} {st["avg_win"]:>7.3f} '
                  f'{st["avg_loss"]:>7.3f} {st["rr"]:>5.2f} {st["pf"]:>6.3f} '
                  f'{st["ev"]:>7.2f} {st.get("be_cost_bps", 0):>10.1f}')

        # Exit reason breakdown
        reasons = {}
        for t in raw_trades:
            r = t['exit_reason']
            reasons[r] = reasons.get(r, 0) + 1
        parts = [f'{k}={v}' for k, v in sorted(reasons.items())]
        print(f'  Exits: {", ".join(parts)}')

    # Analyze BOTH regimes
    for min_c in [3]:
        fd, fc, fm = {}, {}, {}
        for sym in sym_dirs_both:
            d, c, m = filter_by_conf(sym_dirs_both[sym], sym_confs_both[sym], sym_moves_both[sym], min_c)
            fd[sym] = d; fc[sym] = c; fm[sym] = m

        raw_trades = analyze_raw_trades(
            sym_close_dict, sym_high_dict, sym_low_dict,
            fd, fc, fm,
        )
        stats_zero = compute_trade_stats(raw_trades, cost_per_side=0.0)
        print(f'\n  ── BOTH Regimes, conf≥{min_c}, zero cost ──')
        print(f'  N={stats_zero["n"]:,}  WR={stats_zero["wr"]:.1f}%  '
              f'R:R={stats_zero["rr"]:.2f}  PF={stats_zero["pf"]:.3f}  '
              f'EV={stats_zero["ev"]:.2f}bp  BE_cost={stats_zero["be_cost_bps"]:.1f}bp')

    # Analyze fixed thresholds
    for label, sd, sc, sm in [
        ('Fixed 5%', sym_dirs_fix5, sym_confs_fix5, sym_moves_fix5),
        ('Fixed 3.2%', sym_dirs_fix3, sym_confs_fix3, sym_moves_fix3),
    ]:
        fd, fc, fm = {}, {}, {}
        for sym in sd:
            d, c, m = filter_by_conf(sd[sym], sc[sym], sm[sym], 3)
            fd[sym] = d; fc[sym] = c; fm[sym] = m

        raw_trades = analyze_raw_trades(
            sym_close_dict, sym_high_dict, sym_low_dict,
            fd, fc, fm,
        )
        stats_zero = compute_trade_stats(raw_trades, cost_per_side=0.0)
        print(f'\n  ── {label}, conf≥3, zero cost ──')
        print(f'  N={stats_zero["n"]:,}  WR={stats_zero["wr"]:.1f}%  '
              f'R:R={stats_zero["rr"]:.2f}  PF={stats_zero["pf"]:.3f}  '
              f'EV={stats_zero["ev"]:.2f}bp  BE_cost={stats_zero["be_cost_bps"]:.1f}bp')

    # With MTF filter
    for mtf_period in [6, 12, 24]:
        fd, fc, fm = {}, {}, {}
        for sym in sym_dirs_ro:
            d, c, m = filter_by_conf(sym_dirs_ro[sym], sym_confs_ro[sym], sym_moves_ro[sym], 3)
            if 'close_4h' in sym_data[sym]:
                d, c, m = apply_mtf_filter(d, c, m, sym_data[sym]['close_4h'], sma_period=mtf_period)
            fd[sym] = d; fc[sym] = c; fm[sym] = m

        raw_trades = analyze_raw_trades(
            sym_close_dict, sym_high_dict, sym_low_dict,
            fd, fc, fm,
        )
        stats_zero = compute_trade_stats(raw_trades, cost_per_side=0.0)
        print(f'\n  ── MTF SMA({mtf_period}), conf≥3, zero cost ──')
        print(f'  N={stats_zero["n"]:,}  WR={stats_zero["wr"]:.1f}%  '
              f'R:R={stats_zero["rr"]:.2f}  PF={stats_zero["pf"]:.3f}  '
              f'EV={stats_zero["ev"]:.2f}bp  BE_cost={stats_zero["be_cost_bps"]:.1f}bp')

    # Different exit strategies
    print(f'\n  ── EXIT STRATEGY COMPARISON (conf≥3, Risk-Off, zero cost) ──')
    fd3, fc3, fm3 = {}, {}, {}
    for sym in sym_dirs_ro:
        d, c, m = filter_by_conf(sym_dirs_ro[sym], sym_confs_ro[sym], sym_moves_ro[sym], 3)
        fd3[sym] = d; fc3[sym] = c; fm3[sym] = m

    exit_configs = [
        ('PBT-1.5/0.382/25', 1.5, 0.382, 25),
        ('Tight-SL-0.20', 1.5, 0.20, 25),
        ('Wide-SL-0.50', 1.5, 0.50, 25),
        ('Quick-tgt-0.5x', 0.5, 0.382, 25),
        ('Wide-tgt-2.5x', 2.5, 0.382, 25),
        ('Quick-hold-12', 1.5, 0.382, 12),
        ('Patient-hold-50', 1.5, 0.382, 50),
        ('Asymm-2.0/0.20', 2.0, 0.20, 25),      # High R:R
        ('Scalp-0.3/0.15', 0.3, 0.15, 10),       # Quick scalp
    ]

    print(f'  {"Config":<20} {"N":>6} {"WR%":>6} {"R:R":>5} {"PF":>6} '
          f'{"EV(bp)":>7} {"BE(bp)":>6}')
    print(f'  {"─" * 60}')

    for name, tgt, stp, mh in exit_configs:
        raw_trades = analyze_raw_trades(
            sym_close_dict, sym_high_dict, sym_low_dict,
            fd3, fc3, fm3,
            target_retrace=tgt, stop_mult=stp, max_hold=mh,
        )
        st = compute_trade_stats(raw_trades, cost_per_side=0.0)
        print(f'  {name:<20} {st["n"]:>6} {st["wr"]:>6.1f} {st["rr"]:>5.2f} '
              f'{st["pf"]:>6.3f} {st["ev"]:>7.2f} {st.get("be_cost_bps", 0):>6.1f}')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: PORTFOLIO SIMULATION (top variants)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('  PHASE 2: PORTFOLIO SIMULATION (max_slots=2)')
    print('═' * 80)

    variants = build_variants()

    print(f'\n  {"Var":<12} {"Desc":<42} {"Ret%":>8} {"DD%":>6} '
          f'{"Sharpe":>7} {"WR%":>5} {"PF":>5} {"#Tr":>6}')
    print(f'  {"─" * 95}')

    for v in variants:
        # Select appropriate signal set
        if v.risk_off_only:
            if v.use_adaptive_thr:
                sd, sc, sm = sym_dirs_ro, sym_confs_ro, sym_moves_ro
            elif v.fixed_threshold == 0.05:
                sd, sc, sm = sym_dirs_fix5, sym_confs_fix5, sym_moves_fix5
            elif v.fixed_threshold == 0.032:
                sd, sc, sm = sym_dirs_fix3, sym_confs_fix3, sym_moves_fix3
            else:
                sd, sc, sm = sym_dirs_ro, sym_confs_ro, sym_moves_ro
        else:
            sd, sc, sm = sym_dirs_both, sym_confs_both, sym_moves_both

        # Filter by confidence
        fd, fc, fm = {}, {}, {}
        for sym in sd:
            d, c, m = filter_by_conf(sd[sym], sc[sym], sm[sym], v.min_confidence)
            # Apply MTF if enabled
            if v.use_mtf and 'close_4h' in sym_data.get(sym, {}):
                d, c, m = apply_mtf_filter(d, c, m, sym_data[sym]['close_4h'],
                                           sma_period=v.mtf_sma_period)
            fd[sym] = d; fc[sym] = c; fm[sym] = m

        result = simulate_portfolio(
            sym_close_dict, sym_high_dict, sym_low_dict,
            fd, fc, fm, sym_atr_dict, v,
        )

        print(f'  {v.name:<12} {v.desc:<42} {result["total_ret"]:>+8.1f} '
              f'{result["max_dd"]:>6.1f} {result["sharpe"]:>7.2f} '
              f'{result["wr"]:>5.1f} {result["pf"]:>5.2f} {result["n_trades"]:>6}')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: COST BREAK-EVEN ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('  PHASE 3: COST BREAK-EVEN ANALYSIS')
    print('  At what cost level does PF cross 1.0?')
    print('═' * 80)

    # Use raw trades from conf≥3, Risk-Off, no MTF
    fd3, fc3, fm3 = {}, {}, {}
    for sym in sym_dirs_ro:
        d, c, m = filter_by_conf(sym_dirs_ro[sym], sym_confs_ro[sym], sym_moves_ro[sym], 3)
        fd3[sym] = d; fc3[sym] = c; fm3[sym] = m

    raw_trades = analyze_raw_trades(
        sym_close_dict, sym_high_dict, sym_low_dict,
        fd3, fc3, fm3,
    )

    print(f'\n  {"Cost/side":>10} {"PF":>6} {"EV(bp)":>8} {"Total%":>10}')
    print(f'  {"─" * 40}')
    for cost_bps in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        cost = cost_bps / 10000
        st = compute_trade_stats(raw_trades, cost_per_side=cost, leverage=1.0)
        print(f'  {cost_bps:>8}bp {st["pf"]:>6.3f} {st["ev"]:>8.2f} {st["total_ret"]:>+10.1f}')

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: MFE/MAE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('  PHASE 4: MFE/MAE ANALYSIS (Maximum Favorable/Adverse Excursion)')
    print('  How much does the move continue after entry?')
    print('═' * 80)

    if raw_trades:
        mfes = np.array([t['mfe'] for t in raw_trades])
        maes = np.array([t['mae'] for t in raw_trades])
        gains = np.array([t['raw_gain'] for t in raw_trades])

        print(f'\n  {"":<10} {"Mean":>8} {"Median":>8} {"P25":>8} {"P75":>8} {"P90":>8}')
        print(f'  {"─" * 50}')
        for label, arr in [('MFE%', mfes * 100), ('MAE%', maes * 100), ('Gain%', gains * 100)]:
            print(f'  {label:<10} {np.mean(arr):>8.3f} {np.median(arr):>8.3f} '
                  f'{np.percentile(arr, 25):>8.3f} {np.percentile(arr, 75):>8.3f} '
                  f'{np.percentile(arr, 90):>8.3f}')

        # Win/loss MFE comparison
        winning = [t for t in raw_trades if t['raw_gain'] > 0]
        losing = [t for t in raw_trades if t['raw_gain'] <= 0]
        if winning and losing:
            w_mfe = np.mean([t['mfe'] for t in winning]) * 100
            l_mfe = np.mean([t['mfe'] for t in losing]) * 100
            w_mae = np.mean([t['mae'] for t in losing]) * 100
            print(f'\n  Winners ({len(winning)}): mean MFE = {w_mfe:.3f}%')
            print(f'  Losers ({len(losing)}): mean MFE = {l_mfe:.3f}%, mean MAE = {w_mae:.3f}%')

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('  SUMMARY')
    print('═' * 80)

    fd3, fc3, fm3 = {}, {}, {}
    for sym in sym_dirs_ro:
        d, c, m = filter_by_conf(sym_dirs_ro[sym], sym_confs_ro[sym], sym_moves_ro[sym], 3)
        fd3[sym] = d; fc3[sym] = c; fm3[sym] = m
    base_trades = analyze_raw_trades(
        sym_close_dict, sym_high_dict, sym_low_dict,
        fd3, fc3, fm3,
    )
    base_stats = compute_trade_stats(base_trades, cost_per_side=0.0)

    print(f'\n  Raw signal (conf≥3, Risk-Off, no MTF, zero cost):')
    print(f'    Trades: {base_stats["n"]:,}')
    print(f'    WR: {base_stats["wr"]:.1f}%')
    print(f'    R:R: {base_stats["rr"]:.2f}')
    print(f'    PF: {base_stats["pf"]:.3f}')
    print(f'    EV per trade: {base_stats["ev"]:.2f} bp')
    print(f'    Break-even cost: {base_stats["be_cost_bps"]:.1f} bp per side')
    print(f'    Verdict: {"✅ EDGE EXISTS" if base_stats["pf"] > 1.0 else "❌ NO EDGE"}')

    if base_stats['pf'] > 1.0:
        print(f'\n  The signal has theoretical edge ({base_stats["pf"]:.3f} PF).')
        print(f'  Break-even at {base_stats["be_cost_bps"]:.1f}bp/side.')
        if base_stats['be_cost_bps'] > 2.0:
            print(f'  Maker fees (2bp) would be survivable. ✅')
        else:
            print(f'  Even maker fees ({2.0:.0f}bp) exceed break-even. ❌')
    else:
        print(f'\n  ❌ The signal has NO theoretical edge (PF={base_stats["pf"]:.3f}).')
        print(f'  Mean reversion at 15m with these entry criteria does not work.')
        print(f'  Options: different entry logic, different timeframe, or different strategy.')

    # Save results
    log_dir = Path(__file__).parent.parent.parent / 'logs' / 'research'
    log_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'commit': commit,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'base_stats': base_stats,
    }
    with open(log_dir / 'alpha_diagnostic_r22_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n💾 Results saved to {log_dir / "alpha_diagnostic_r22_results.json"}')


if __name__ == '__main__':
    main()
