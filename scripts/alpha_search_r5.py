#!/usr/bin/env python3
"""
Alpha Search Round 5 — Killer Combo Refinement
================================================
R3+R4 found Y-Kitchen as TRIPLE WINNER:
  WR 67.1%, R:R 1.00, DD 7.2%, Sharpe 8.42, PF 2.07
  (conf≥3 + dir_diversity + steep_cap + vol_size + soft_equity)

R5 tests NEW dimensions on top of Y-Kitchen to find the ultimate combo:
  - Dynamic leverage stacking (Y-Kitchen has DD headroom: 7.2% vs 11.1% budget)
  - Partial exit + trailing (E-BE15 boosted R:R by +54% in prior test)
  - Slot count tuning (2 vs 4 slots)
  - Conf≥4 stricter gate
  - Threshold fine-tuning (soft equity / steep cap params)

Two baselines:
  A: v12.2 production (conf≥2, no innovations) via R3 A-Base
  B: Y-Kitchen (R4 champion)

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r5.py
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
# Production parameters (v12.2)
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032
TARGET_RETRACE = 0.382
STOP_MULT = 0.786
MAX_HOLD = 20
STOP_DECAY = 0.0
BTC_MOM_PERIOD = 48
BTC_REGIME_PERIOD = 96
ASSET_TREND_PERIOD = 192
MIN_CONFIDENCE = 2
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


def load_15m(symbol: str, days: int = 1095) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None:
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int = 1095) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None:
        return None
    return np.load(cache)['close']


# ═══════════════════════════════════════════════════════════════════
# Part 1: Entry Opportunity Scanner (reuse from R3)
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_entries(
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
                thr = atr_multiplier * atr_val
                if thr < threshold_floor:
                    thr = threshold_floor
                if thr > threshold_cap:
                    thr = threshold_cap
        else:
            thr = threshold

        if abs(ret) <= thr:
            continue

        if np.isnan(btc_sma[i]):
            continue

        is_risk_on = btc_close[i] > btc_sma[i]
        btc_mom = btc_close[i] / btc_close[i - btc_mom_period] - 1.0
        confidence = 1
        direction = 0

        if is_risk_on:
            if np.isnan(asset_sma[i]):
                continue
            is_uptrend = close[i] > asset_sma[i]

            if ret < -thr and is_uptrend:
                direction = 1
                if rsi[i] < 40:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] < 0.45:
                    confidence += 1
                dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                if dist < 0.02:
                    confidence += 1
                if stoch_enabled and i > 0:
                    if (stoch_k[i] < 20 and stoch_k[i] > stoch_d[i]
                            and stoch_k[i - 1] <= stoch_d[i - 1]):
                        confidence += 1
                if macd_enabled and i > 0:
                    if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                        confidence += 1

            elif ret > thr and not is_uptrend:
                direction = -1
                if rsi[i] > 60:
                    confidence += 1
                if not np.isnan(vol_sma[i]) and volume[i] > 1.5 * vol_sma[i]:
                    confidence += 1
                if taker_ratio[i] > 0.55:
                    confidence += 1
                dist = abs(close[i] - asset_sma[i]) / asset_sma[i]
                if dist < 0.02:
                    confidence += 1
                if stoch_enabled and i > 0:
                    if (stoch_k[i] > 80 and stoch_k[i] < stoch_d[i]
                            and stoch_k[i - 1] >= stoch_d[i - 1]):
                        confidence += 1
                if macd_enabled and i > 0:
                    if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                        confidence += 1

        else:
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
# Part 2: MTF Filter
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
# Part 3: Variant Configuration
# ═══════════════════════════════════════════════════════════════════
@dataclass
class R5Config:
    """R5 variant configuration — builds on Y-Kitchen foundation."""
    name: str = 'Baseline'
    desc: str = ''

    # Core exit params
    target_retrace: float = TARGET_RETRACE
    stop_mult: float = STOP_MULT
    stop_decay: float = STOP_DECAY
    max_hold: int = MAX_HOLD
    trail_activation: float = 0.5
    trail_factor: float = 0.3

    # Y-Kitchen innovations (default = Y-Kitchen ON)
    min_conf_gate: int = 3
    dir_diversity: bool = True
    steep_cap_curve: bool = True
    vol_adjusted_size: bool = True
    soft_equity_scale: bool = True

    # Steep cap multipliers (tunable)
    cap_mult_conf2: float = 0.5
    cap_mult_conf3: float = 0.7
    cap_mult_conf4: float = 1.2
    cap_mult_conf5: float = 1.5

    # Soft equity thresholds (tunable)
    soft_eq_dd1: float = 0.03   # 3% DD → scale 1
    soft_eq_scale1: float = 0.85
    soft_eq_dd2: float = 0.05   # 5% DD → scale 2
    soft_eq_scale2: float = 0.70
    soft_eq_dd3: float = 0.08   # 8% DD → scale 3
    soft_eq_scale3: float = 0.50

    # Dynamic leverage
    dynamic_leverage: bool = False
    dyn_lev_step: float = 0.25  # +step per confidence level
    dyn_lev_cap: float = 10.0

    # Partial exit + trailing
    partial_exit: bool = False
    partial_fraction: float = 0.5
    trail_atr_mult: float = 1.5
    breakeven_after_tp1: bool = False

    # Slot count
    max_slots: int = 3

    # Conf-adaptive trailing
    conf_adaptive_trail: bool = False


# ═══════════════════════════════════════════════════════════════════
# Part 4: Portfolio Simulator (R5 — extended with partial exit)
# ═══════════════════════════════════════════════════════════════════
def simulate_r5(
    sym_close: dict[str, np.ndarray],
    sym_high: dict[str, np.ndarray],
    sym_low: dict[str, np.ndarray],
    sym_dirs: dict[str, np.ndarray],
    sym_confs: dict[str, np.ndarray],
    sym_moves: dict[str, np.ndarray],
    cfg: R5Config,
    initial_capital: float = 2095.0,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    sym_atr: dict[str, np.ndarray] | None = None,
) -> dict:
    """Portfolio sim with all R4 innovations + partial exit + dynamic leverage."""
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
        # ═══ Phase 1: Check exits on existing positions ═══
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

            target = pos['move_size'] * cfg.target_retrace
            time_factor = 1.0 - cfg.stop_decay * (pos['hold_count'] / cfg.max_hold)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * cfg.stop_mult * time_factor

            # Trail params
            if cfg.conf_adaptive_trail:
                conf = pos['confidence']
                if conf >= 4:
                    t_act, t_fac = 0.35, 0.50
                elif conf <= 2:
                    t_act, t_fac = 0.30, 0.25
                else:
                    t_act, t_fac = cfg.trail_activation, cfg.trail_factor
            else:
                t_act = cfg.trail_activation
                t_fac = cfg.trail_factor

            # ════════════════════════════════════════════════
            # Partial exit mode
            # ════════════════════════════════════════════════
            if cfg.partial_exit and not pos.get('tp1_hit', False):
                # Pre-TP1: normal exit logic but watching for target
                if gain_close >= target:
                    # TP1 HIT — close partial_fraction
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain_close

                    # Close the first portion
                    tp1_notional = pos['notional'] * cfg.partial_fraction
                    remaining_notional = pos['notional'] - tp1_notional
                    funding = lev * funding_per_bar * tp1_notional * pos['hold_count']
                    close_cost = lev * cost_per_side * tp1_notional
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

                    # Update position for runner
                    pos['notional'] = remaining_notional
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain_close

                    if cfg.breakeven_after_tp1:
                        pos['trail_floor'] = 0.0  # breakeven
                    else:
                        pos['trail_floor'] = -effective_stop

                elif gain_close < -effective_stop:
                    # SL before TP1 — exit full position
                    funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                    close_cost = lev * cost_per_side * pos['notional']
                    gross_pnl = gain_close * lev * pos['notional']
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': 'stop',
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                    })
                    to_close.append(sym)

                elif pos['hold_count'] >= cfg.max_hold:
                    # Time exit before TP1
                    funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                    close_cost = lev * cost_per_side * pos['notional']
                    gross_pnl = gain_close * lev * pos['notional']
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': 'time',
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                    })
                    to_close.append(sym)

                else:
                    # Update trailing (pre-TP1 uses standard logic)
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * t_act
                            and gain_close < pos['best_pnl'] * t_fac):
                        funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                        close_cost = lev * cost_per_side * pos['notional']
                        gross_pnl = gain_close * lev * pos['notional']
                        net_pnl = gross_pnl - close_cost - funding
                        balance += net_pnl
                        trades.append({
                            'symbol': sym, 'direction': pos['direction'],
                            'confidence': pos['confidence'],
                            'exit_reason': 'trail',
                            'pnl_usd': net_pnl,
                            'pnl_pct': (net_pnl / pos['notional']) * 100,
                            'bars_held': pos['hold_count'],
                        })
                        to_close.append(sym)

            elif cfg.partial_exit and pos.get('tp1_hit', False):
                # Post-TP1: trailing the runner with ATR-based stop
                if gain_close > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain_close

                atr_val = 0.02  # fallback
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
                elif pos['hold_count'] >= cfg.max_hold * 2:
                    # Extended time limit for runner (2× normal)
                    exit_runner = True
                    exit_reason = 'time_runner'

                if exit_runner:
                    funding = lev * funding_per_bar * pos['notional'] * (i - pos.get('tp1_bar', pos['entry_bar']))
                    close_cost = lev * cost_per_side * pos['notional']
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
                    })
                    to_close.append(sym)

            else:
                # ════════════════════════════════════════════════
                # Standard exit (no partial) — same as R3/R4
                # ════════════════════════════════════════════════
                exit_reason = None
                exit_pnl = None

                if gain_close >= target:
                    exit_reason = 'target'
                    exit_pnl = gain_close
                elif gain_close < -effective_stop:
                    exit_reason = 'stop'
                    exit_pnl = gain_close
                else:
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (pos['best_pnl'] > target * t_act
                            and gain_close < pos['best_pnl'] * t_fac):
                        exit_reason = 'trail'
                        exit_pnl = gain_close
                    elif pos['hold_count'] >= cfg.max_hold:
                        exit_reason = 'time'
                        exit_pnl = gain_close

                if exit_reason is not None:
                    funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                    close_cost = lev * cost_per_side * pos['notional']
                    gross_pnl = exit_pnl * lev * pos['notional']
                    net_pnl = gross_pnl - close_cost - funding
                    balance += net_pnl
                    trades.append({
                        'symbol': sym, 'direction': pos['direction'],
                        'confidence': pos['confidence'],
                        'exit_reason': exit_reason,
                        'pnl_usd': net_pnl,
                        'pnl_pct': (net_pnl / pos['notional']) * 100,
                        'bars_held': pos['hold_count'],
                    })
                    to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Phase 2: New entries ═══
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
                    same_dir = sum(
                        1 for p in positions.values()
                        if p['direction'] == direction
                    )
                    if same_dir >= 2:
                        continue

                candidates.append((sym, direction, confidence, move_size))

            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size in candidates:
                if len(positions) >= cfg.max_slots:
                    break

                per_slot = initial_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, initial_capital * 0.5)
                if per_slot < 10.0:
                    continue

                # Steep cap curve
                if cfg.steep_cap_curve:
                    if confidence >= 5:
                        per_slot *= cfg.cap_mult_conf5
                    elif confidence >= 4:
                        per_slot *= cfg.cap_mult_conf4
                    elif confidence == 3:
                        per_slot *= cfg.cap_mult_conf3
                    else:
                        per_slot *= cfg.cap_mult_conf2
                    per_slot = min(per_slot, initial_capital * 0.5)

                # Vol-adjusted sizing
                if cfg.vol_adjusted_size and sym_atr is not None:
                    atr_arr = sym_atr.get(sym)
                    if atr_arr is not None and i < len(atr_arr) and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
                        median_atr = 0.01
                        vol_factor = median_atr / atr_arr[i]
                        vol_factor = max(0.4, min(vol_factor, 2.0))
                        per_slot *= vol_factor
                        per_slot = min(per_slot, initial_capital * 0.5)

                # Soft equity scaling
                if cfg.soft_equity_scale:
                    eq_now = equity_curve[i - 1] if i > 0 else initial_capital
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    if dd_pct > cfg.soft_eq_dd3:
                        per_slot *= cfg.soft_eq_scale3
                    elif dd_pct > cfg.soft_eq_dd2:
                        per_slot *= cfg.soft_eq_scale2
                    elif dd_pct > cfg.soft_eq_dd1:
                        per_slot *= cfg.soft_eq_scale1

                # Dynamic leverage
                trade_lev = EFFECTIVE_LEV
                if cfg.dynamic_leverage:
                    base_l = BASE_LEVERAGE + cfg.dyn_lev_step * (confidence - 1)
                    trade_lev = min(base_l * MTF_LEV_MULT, cfg.dyn_lev_cap)

                # Entry cost
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
    }


# ═══════════════════════════════════════════════════════════════════
# Part 5: Main
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

    print("=" * 110)
    print("  ALPHA SEARCH ROUND 5 — KILLER COMBO REFINEMENT")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baselines: A=v12.2 production / B=Y-Kitchen (R4 champion)")
    print(f"  New dimensions: DynLev, Partial Exit+Trail, Slot tuning, Conf≥4, Threshold tuning")
    print(f"  Target: Beat Y-Kitchen (WR 67.1%, R:R 1.00, DD 7.2%, Sharpe 8.42)")
    print("=" * 110)

    # ── Load data ──
    print(f"\n📊 Loading 50-symbol data (1095d)...")
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

    # ── Scan entries ──
    print(f"\n🔍 Scanning entry opportunities...")
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
    max_bars = max(len(sym_close[s]) for s in sym_close)
    for sym in list(sym_close.keys()):
        for d_ref, fill in [
            (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
            (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
            (sym_atr, 'nan'),
        ]:
            arr = d_ref[sym]
            if len(arr) < max_bars:
                pad_n = max_bars - len(arr)
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

    # ══════════════════════════════════════════════════════════════
    # Define R5 Variants
    # ══════════════════════════════════════════════════════════════
    variants = [
        # ── Baselines ──
        R5Config(
            name='A-v12.2',
            desc='v12.2 production (conf≥2, no innovations)',
            min_conf_gate=2,
            dir_diversity=False,
            steep_cap_curve=False,
            vol_adjusted_size=False,
            soft_equity_scale=False,
        ),
        R5Config(
            name='B-YKitch',
            desc='Y-Kitchen (R4 champion — conf≥3 + all 5 innovations)',
        ),

        # ── Dimension 1: Dynamic Leverage on Y-Kitchen ──
        R5Config(
            name='C-YK+DL25',
            desc='Y-Kitchen + dynamic leverage +0.25/conf',
            dynamic_leverage=True,
            dyn_lev_step=0.25,
        ),
        R5Config(
            name='D-YK+DL50',
            desc='Y-Kitchen + dynamic leverage +0.50/conf (aggressive)',
            dynamic_leverage=True,
            dyn_lev_step=0.50,
        ),

        # ── Dimension 2: Partial Exit + Trailing ──
        R5Config(
            name='E-YK+PT15',
            desc='Y-Kitchen + partial 50% trail 1.5×ATR',
            partial_exit=True,
            trail_atr_mult=1.5,
        ),
        R5Config(
            name='F-YK+BE15',
            desc='Y-Kitchen + partial 50% + breakeven + trail 1.5×ATR',
            partial_exit=True,
            trail_atr_mult=1.5,
            breakeven_after_tp1=True,
        ),
        R5Config(
            name='G-YK+PT20',
            desc='Y-Kitchen + partial 50% trail 2.0×ATR (wider)',
            partial_exit=True,
            trail_atr_mult=2.0,
        ),

        # ── Dimension 3: DynLev + Partial (double stack) ──
        R5Config(
            name='H-DL+BE15',
            desc='YK + dyn leverage 0.25 + partial + breakeven + trail 1.5×',
            dynamic_leverage=True,
            dyn_lev_step=0.25,
            partial_exit=True,
            trail_atr_mult=1.5,
            breakeven_after_tp1=True,
        ),
        R5Config(
            name='I-DL+PT20',
            desc='YK + dyn leverage 0.25 + partial + trail 2.0×',
            dynamic_leverage=True,
            dyn_lev_step=0.25,
            partial_exit=True,
            trail_atr_mult=2.0,
        ),

        # ── Dimension 4: Slot tuning ──
        R5Config(
            name='J-YK+2Slt',
            desc='Y-Kitchen with 2 slots (concentrated)',
            max_slots=2,
        ),
        R5Config(
            name='K-YK+4Slt',
            desc='Y-Kitchen with 4 slots (diversified)',
            max_slots=4,
        ),

        # ── Dimension 5: Stricter conf gate ──
        R5Config(
            name='L-YK+C4',
            desc='Y-Kitchen but conf≥4 gate (ultra-filter)',
            min_conf_gate=4,
        ),

        # ── Dimension 6: Threshold fine-tuning ──
        R5Config(
            name='M-SteepX',
            desc='YK + extreme steep cap (0.3/0.5/1.5/2.0)',
            cap_mult_conf2=0.3,
            cap_mult_conf3=0.5,
            cap_mult_conf4=1.5,
            cap_mult_conf5=2.0,
        ),
        R5Config(
            name='N-SoftTight',
            desc='YK + tighter soft equity (2%/4%/6% DD thresholds)',
            soft_eq_dd1=0.02, soft_eq_scale1=0.80,
            soft_eq_dd2=0.04, soft_eq_scale2=0.60,
            soft_eq_dd3=0.06, soft_eq_scale3=0.40,
        ),
        R5Config(
            name='O-SoftWide',
            desc='YK + wider soft equity (5%/8%/12% DD thresholds)',
            soft_eq_dd1=0.05, soft_eq_scale1=0.85,
            soft_eq_dd2=0.08, soft_eq_scale2=0.70,
            soft_eq_dd3=0.12, soft_eq_scale3=0.50,
        ),

        # ── Dimension 7: Best combos from above ──
        R5Config(
            name='P-DL+4Slt',
            desc='YK + dyn leverage 0.25 + 4 slots',
            dynamic_leverage=True,
            dyn_lev_step=0.25,
            max_slots=4,
        ),
        R5Config(
            name='Q-DL+StpX',
            desc='YK + dyn leverage 0.25 + extreme steep cap',
            dynamic_leverage=True,
            dyn_lev_step=0.25,
            cap_mult_conf2=0.3,
            cap_mult_conf3=0.5,
            cap_mult_conf4=1.5,
            cap_mult_conf5=2.0,
        ),

        # ── Dimension 8: Conf-adaptive trail on Y-Kitchen ──
        R5Config(
            name='R-YK+CfTr',
            desc='YK + conf-adaptive trailing',
            conf_adaptive_trail=True,
        ),

        # ── Kitchen sink++ (all promising bells & whistles) ──
        R5Config(
            name='S-Ultra',
            desc='YK + dyn lev 0.25 + breakeven trail 1.5× + steep X',
            dynamic_leverage=True,
            dyn_lev_step=0.25,
            partial_exit=True,
            trail_atr_mult=1.5,
            breakeven_after_tp1=True,
            cap_mult_conf2=0.3,
            cap_mult_conf3=0.5,
            cap_mult_conf4=1.5,
            cap_mult_conf5=2.0,
        ),
    ]

    # ── Run ──
    results: dict[str, dict] = {}
    yk_ref = None  # Y-Kitchen reference metrics

    for cfg in variants:
        print(f"\n{'─' * 100}")
        print(f"  ⚡ {cfg.name}: {cfg.desc}")
        print(f"{'─' * 100}")

        t0 = time.time()
        result = simulate_r5(
            sym_close, sym_high, sym_low,
            sym_dirs, sym_confs, sym_moves,
            cfg=cfg,
            sym_atr=sym_atr,
        )
        elapsed = time.time() - t0

        result['description'] = cfg.desc
        result['elapsed'] = round(elapsed, 2)
        results[cfg.name] = result

        if cfg.name == 'B-YKitch':
            yk_ref = result

        # Status marks vs Y-Kitchen
        if yk_ref:
            wr_ok = '✓' if result['win_rate'] >= yk_ref['win_rate'] else '✗'
            rr_ok = '✓' if result['actual_rr'] >= yk_ref['actual_rr'] else '✗'
            dd_ok = '✓' if result['max_dd_pct'] <= yk_ref['max_dd_pct'] else '✗'
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
        print(f"  Exits: {reasons_str}")

    # ── Comparison Table ──
    names = list(results.keys())
    cw = 11

    print(f"\n\n{'=' * 140}")
    print(f"  R5 COMPARATIVE RESULTS")
    if yk_ref:
        print(f"  Y-Kitchen ref: WR {yk_ref['win_rate']}% R:R {yk_ref['actual_rr']} "
              f"DD {yk_ref['max_dd_pct']}% Sharpe {yk_ref['sharpe']} PF {yk_ref['profit_factor']}")
    print(f"{'=' * 140}\n")

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
        ('Avg PnL', 'avg_pnl_pct', '{:+.3f}%'),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<18}", end='')
        for n in names:
            s = fmt.format(results[n][key])
            print(f"  {s:>{cw}}", end='')
        print()

    # ── Delta vs Y-Kitchen ──
    if yk_ref:
        base = yk_ref
        print(f"\n{'─' * 140}")
        print(f"  DELTA vs B-YKitch (Y-Kitchen)")
        print(f"{'─' * 140}\n")

        for n in names:
            if n == 'B-YKitch':
                continue
            r = results[n]
            d_wr = r['win_rate'] - base['win_rate']
            d_rr = r['actual_rr'] - base['actual_rr']
            d_ret = r['total_return_pct'] - base['total_return_pct']
            d_sh = r['sharpe'] - base['sharpe']
            d_dd = r['max_dd_pct'] - base['max_dd_pct']
            d_pf = r['profit_factor'] - base['profit_factor']

            wr_ok = '✓' if d_wr >= 0 else '✗'
            rr_ok = '✓' if d_rr >= 0 else '✗'
            dd_ok = '✓' if d_dd <= 0 else '✗'

            tags = []
            if d_wr >= 0 and d_rr >= 0 and d_dd <= 0 and n != 'A-v12.2':
                tags.append('★ DOMINATES YK')
            if d_sh > 0 and d_dd <= 0:
                tags.append('◆ SHARPE+DD')
            if r['total_return_pct'] > base['total_return_pct'] * 1.1:
                tags.append('💰 RET+10%')
            tag = '  '.join(tags)

            print(f"  {n:<12} ΔWR={d_wr:>+5.1f}%p[{wr_ok}]  ΔR:R={d_rr:>+5.2f}[{rr_ok}]  "
                  f"ΔDD={d_dd:>+5.1f}%[{dd_ok}]  ΔRet={d_ret:>+8,.1f}%  "
                  f"ΔSharpe={d_sh:>+5.2f}  ΔPF={d_pf:>+5.2f}  {tag}")

    # ── Also compare vs v12.2 ──
    v12_ref = results.get('A-v12.2')
    if v12_ref:
        print(f"\n{'─' * 140}")
        print(f"  DELTA vs A-v12.2 (Production)")
        print(f"{'─' * 140}\n")

        for n in names:
            if n == 'A-v12.2':
                continue
            r = results[n]
            d_wr = r['win_rate'] - v12_ref['win_rate']
            d_rr = r['actual_rr'] - v12_ref['actual_rr']
            d_dd = r['max_dd_pct'] - v12_ref['max_dd_pct']
            d_sh = r['sharpe'] - v12_ref['sharpe']

            triple = (d_wr > 0 and d_rr > 0 and d_dd < 0)
            tag = '★★★ TRIPLE WIN vs v12.2' if triple else ''

            print(f"  {n:<12} ΔWR={d_wr:>+5.1f}%p  ΔR:R={d_rr:>+5.2f}  "
                  f"ΔDD={d_dd:>+5.1f}%  ΔSharpe={d_sh:>+5.2f}  {tag}")

    # ── Verdict ──
    print(f"\n{'=' * 140}")
    print("  VERDICT")
    print(f"{'=' * 140}\n")

    if yk_ref:
        dominators = [n for n in names if n not in ('A-v12.2', 'B-YKitch') and (
            results[n]['win_rate'] >= yk_ref['win_rate']
            and results[n]['actual_rr'] >= yk_ref['actual_rr']
            and results[n]['max_dd_pct'] <= yk_ref['max_dd_pct']
        )]

        if dominators:
            print(f"  ★ VARIANTS THAT DOMINATE Y-KITCHEN: {', '.join(dominators)}")
            for n in dominators:
                r = results[n]
                print(f"    {n}: WR {r['win_rate']:.1f}% R:R {r['actual_rr']:.2f} "
                      f"DD {r['max_dd_pct']:.1f}% Sharpe {r['sharpe']:.2f} "
                      f"PF {r['profit_factor']:.2f} Ret {r['total_return_pct']:+,.1f}%")
        else:
            print("  No variant fully dominates Y-Kitchen on all 3 axes (WR, R:R, DD).")

        # Best by each metric
        non_base = [n for n in names if n not in ('A-v12.2', 'B-YKitch')]
        if non_base:
            best_sharpe = max(non_base, key=lambda n: results[n]['sharpe'])
            best_dd = min(non_base, key=lambda n: results[n]['max_dd_pct'])
            best_ret = max(non_base, key=lambda n: results[n]['total_return_pct'])
            best_rr = max(non_base, key=lambda n: results[n]['actual_rr'])
            best_pf = max(non_base, key=lambda n: results[n]['profit_factor'])
            print(f"\n  Category leaders:")
            print(f"    Best Sharpe: {best_sharpe} → {results[best_sharpe]['sharpe']:.2f}")
            print(f"    Best DD:     {best_dd} → {results[best_dd]['max_dd_pct']:.1f}%")
            print(f"    Best Return: {best_ret} → {results[best_ret]['total_return_pct']:+,.1f}%")
            print(f"    Best R:R:    {best_rr} → {results[best_rr]['actual_rr']:.2f}")
            print(f"    Best PF:     {best_pf} → {results[best_pf]['profit_factor']:.2f}")

    # ── Save ──
    output_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r5_results.json'
    save_results = {}
    for k, v in results.items():
        sv = {sk: sv for sk, sv in v.items() if sk != 'equity_curve'}
        save_results[k] = sv
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"\n{'=' * 140}")
    print("  DONE")
    print(f"{'=' * 140}")


if __name__ == '__main__':
    main()
