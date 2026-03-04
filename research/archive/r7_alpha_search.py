#!/usr/bin/env python3
"""
Alpha Search R7 — 3D EXIT GRID SWEEP (6-YEAR CRASH-ERA)
==========================================================
MOTIVATION: Live trading v12.1.0 showed R:R 0.56 (backtest expects 0.78).
  Target 0.382 cuts winners at +0.55% avg while SL 0.786 lets losers run to -1.15%.
  → "이기면 조금, 지면 많이" — the R:R asymmetry kills us.

This sweep explores the FULL TP × SL × EXIT_MODE space to find
aggressive exit configurations that achieve R:R ≥ 1.0+ while maintaining WR.

Base: M-2SSTSX (R6 champion: DD 8.0%, Sharpe 5.50)
  2 slots + SoftTight(2%/4%/6%) + SteepX(0.3/0.5/1.5/2.0)

Axes:
  1. target_retrace: [0.382, 0.50, 0.618, 0.786, 1.0]
  2. stop_mult:      [0.382, 0.50, 0.618, 0.786, 1.0]
  3. exit_mode:      [Std, Trail, Partial50+Trail, Partial50+BE+Trail]

Total: 5 × 5 × 4 = 100 variants

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r7.py
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
# Production parameters (v12.2) — identical to R5/R6
# ═══════════════════════════════════════════════════════════════════
LOOKBACK = 8
THRESHOLD = 0.032
TARGET_RETRACE = 0.382  # current production (overridden in sweep)
STOP_MULT = 0.786       # current production (overridden in sweep)
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
# Part 1: Entry Scanner — IDENTICAL to R5/R6
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
# Part 2: MTF Filter — IDENTICAL to R5/R6
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
# Part 3: R7 Grid Config
# ═══════════════════════════════════════════════════════════════════
@dataclass
class R7Config:
    """R7 exit grid configuration.

    Base: M-2SSTSX (2slt + SoftTight + SteepX) — fixed.
    Sweep: target_retrace × stop_mult × exit_mode.
    """
    name: str = ''
    desc: str = ''

    # ── Swept axes ──
    target_retrace: float = 0.382
    stop_mult: float = 0.786
    exit_mode: str = 'standard'  # standard | trail | partial | partial_be

    # ── Trail params (for trail/partial modes) ──
    trail_activation: float = 0.5
    trail_factor: float = 0.3

    # ── Partial exit params ──
    partial_fraction: float = 0.5
    trail_atr_mult: float = 1.5
    breakeven_after_tp1: bool = False

    # ── Fixed: M-2SSTSX base ──
    max_slots: int = 2
    min_conf_gate: int = 3
    dir_diversity: bool = True
    max_hold: int = 20
    stop_decay: float = 0.0

    # SteepX caps
    steep_cap_curve: bool = True
    cap_mult_conf2: float = 0.3
    cap_mult_conf3: float = 0.5
    cap_mult_conf4: float = 1.5
    cap_mult_conf5: float = 2.0

    # Vol-adjusted sizing
    vol_adjusted_size: bool = True

    # SoftTight equity
    soft_equity_scale: bool = True
    soft_eq_dd1: float = 0.02
    soft_eq_scale1: float = 0.80
    soft_eq_dd2: float = 0.04
    soft_eq_scale2: float = 0.60
    soft_eq_dd3: float = 0.06
    soft_eq_scale3: float = 0.40

    @property
    def theoretical_rr(self) -> float:
        """R:R implied by target/stop ratio."""
        return self.target_retrace / self.stop_mult if self.stop_mult > 0 else 999.0


# ═══════════════════════════════════════════════════════════════════
# Part 4: Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════
def simulate_r7(
    sym_close: dict[str, np.ndarray],
    sym_high: dict[str, np.ndarray],
    sym_low: dict[str, np.ndarray],
    sym_dirs: dict[str, np.ndarray],
    sym_confs: dict[str, np.ndarray],
    sym_moves: dict[str, np.ndarray],
    cfg: R7Config,
    initial_capital: float = 2095.0,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    sym_atr: dict[str, np.ndarray] | None = None,
) -> dict:
    """Portfolio sim with M-2SSTSX base + variable exit grid."""
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    is_trail = cfg.exit_mode in ('trail', 'partial', 'partial_be')
    is_partial = cfg.exit_mode in ('partial', 'partial_be')
    is_be = cfg.exit_mode == 'partial_be'

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

            target = pos['move_size'] * cfg.target_retrace
            time_factor = 1.0 - cfg.stop_decay * (pos['hold_count'] / cfg.max_hold)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * cfg.stop_mult * time_factor

            t_act = cfg.trail_activation
            t_fac = cfg.trail_factor

            # ════════════ PARTIAL EXIT MODE ════════════
            if is_partial and not pos.get('tp1_hit', False):
                if gain_close >= target:
                    # TP1 hit — close partial_fraction
                    pos['tp1_hit'] = True
                    pos['highest_since_tp1'] = gain_close

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

                    pos['notional'] = remaining_notional
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain_close

                    if is_be:
                        pos['trail_floor'] = 0.0  # breakeven floor
                    else:
                        pos['trail_floor'] = -effective_stop

                elif gain_close < -effective_stop:
                    # SL before TP1
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
                    # Trail check (pre-TP1 in partial mode)
                    if gain_close > pos['best_pnl']:
                        pos['best_pnl'] = gain_close
                    if (is_trail
                            and pos['best_pnl'] > target * t_act
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

            elif is_partial and pos.get('tp1_hit', False):
                # Post-TP1: ATR trailing on the runner
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
                elif pos['hold_count'] >= cfg.max_hold * 2:
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
                # ════════════ STANDARD / TRAIL EXIT ════════════
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
                    if (is_trail
                            and pos['best_pnl'] > target * t_act
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

                # SteepX sizing
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

                # SoftTight equity scaling
                if cfg.soft_equity_scale:
                    eq_now = equity_curve[i - 1] if i > 0 else initial_capital
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    if dd_pct > cfg.soft_eq_dd3:
                        per_slot *= cfg.soft_eq_scale3
                    elif dd_pct > cfg.soft_eq_dd2:
                        per_slot *= cfg.soft_eq_scale2
                    elif dd_pct > cfg.soft_eq_dd1:
                        per_slot *= cfg.soft_eq_scale1

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
# Part 5: Grid Generation
# ═══════════════════════════════════════════════════════════════════

TP_GRID = [0.382, 0.50, 0.618, 0.786, 1.0]
SL_GRID = [0.382, 0.50, 0.618, 0.786, 1.0]
EXIT_MODES = [
    ('Std', 'standard'),
    ('Trl', 'trail'),
    ('PT', 'partial'),       # partial 50% + ATR trail runner
    ('PBT', 'partial_be'),   # partial 50% + breakeven + ATR trail runner
]

# Short labels for TP/SL levels
TP_LABELS = {0.382: '38', 0.50: '50', 0.618: '62', 0.786: '79', 1.0: '100'}
SL_LABELS = {0.382: '38', 0.50: '50', 0.618: '62', 0.786: '79', 1.0: '100'}


def generate_grid() -> list[R7Config]:
    """Generate all 100 grid variants."""
    configs = []
    for tp in TP_GRID:
        for sl in SL_GRID:
            for mode_short, mode_full in EXIT_MODES:
                name = f"T{TP_LABELS[tp]}_S{SL_LABELS[sl]}_{mode_short}"
                desc = f"TP={tp:.3f} SL={sl:.3f} Exit={mode_full}"
                cfg = R7Config(
                    name=name,
                    desc=desc,
                    target_retrace=tp,
                    stop_mult=sl,
                    exit_mode=mode_full,
                    breakeven_after_tp1=(mode_full == 'partial_be'),
                )
                configs.append(cfg)
    return configs


# ═══════════════════════════════════════════════════════════════════
# Part 6: Main
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
    print("  ALPHA SEARCH R7 — 3D EXIT GRID SWEEP (6-YEAR CRASH-ERA)")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Data: {DATA_DAYS}d (covers COVID, Luna, FTX crashes)")
    print(f"  Base: M-2SSTSX (2slt + SoftTight + SteepX) — R6 champion")
    print(f"  Axes: TP=[{', '.join(f'{t:.3f}' for t in TP_GRID)}]")
    print(f"         SL=[{', '.join(f'{s:.3f}' for s in SL_GRID)}]")
    print(f"         Exit=[{', '.join(m[0] for m in EXIT_MODES)}]")
    print(f"  Total: {len(TP_GRID)} × {len(SL_GRID)} × {len(EXIT_MODES)} = "
          f"{len(TP_GRID) * len(SL_GRID) * len(EXIT_MODES)} variants")
    print(f"  GOAL: Find R:R ≥ 1.0+ while keeping DD ≤ 10%")
    print("=" * 130)

    # ── Load data (identical to R6) ──
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
        max_bars_d = max(data_lengths.values())
        print(f"  Bar range: {min_bars:,} ({min_bars/96:.0f}d) — {max_bars_d:,} ({max_bars_d/96:.0f}d)")

    # ── Scan entries (one-time, shared across grid) ──
    print(f"\n🔍 Scanning entry opportunities ({DATA_DAYS}d)...")
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

    # Pre-compute ATR
    sym_atr: dict[str, np.ndarray] = {}
    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        atr = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)
        sym_atr[sym] = atr

    # Left-pad
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
    print(f"  Simulation window: {max_bars:,} bars ({max_bars/96:.0f} days)")

    # ══════════════════════════════════════════════════════════════
    # Generate and run grid
    # ══════════════════════════════════════════════════════════════
    grid = generate_grid()

    # Add R6 M-2SSTSX reference (production config)
    ref_cfg = R7Config(
        name='★REF-M2SSTSX',
        desc='R6 champion (production): TP=0.382 SL=0.786 Std',
        target_retrace=0.382,
        stop_mult=0.786,
        exit_mode='standard',
    )
    all_configs = [ref_cfg] + grid

    print(f"\n🧪 Running {len(all_configs)} variants (1 ref + {len(grid)} grid)...")
    print(f"{'─' * 130}")

    results: dict[str, dict] = {}
    ref_result = None

    for idx, cfg in enumerate(all_configs):
        t1 = time.time()
        result = simulate_r7(
            sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves,
            cfg=cfg, sym_atr=sym_atr,
        )
        elapsed = time.time() - t1
        results[cfg.name] = result

        if cfg.name == '★REF-M2SSTSX':
            ref_result = result

        # Compact progress line
        rr_flag = '🏆' if result['actual_rr'] >= 1.0 else ('✓' if result['actual_rr'] >= 0.78 else ' ')
        dd_flag = '✓' if result['max_dd_pct'] <= 10.0 else '✗'
        print(f"  [{idx+1:3d}/{len(all_configs)}] {cfg.name:<22} "
              f"WR={result['win_rate']:5.1f}%  R:R={result['actual_rr']:.2f}{rr_flag}  "
              f"DD={result['max_dd_pct']:5.1f}%{dd_flag}  "
              f"Sharpe={result['sharpe']:5.2f}  PF={result['profit_factor']:.2f}  "
              f"Ret={result['total_return_pct']:+8,.1f}%  "
              f"Trades={result['n_trades']:,}  [{elapsed:.1f}s]")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════

    grid_names = [c.name for c in grid]
    grid_results = {n: results[n] for n in grid_names}

    # ── Heatmaps per exit mode ──
    print(f"\n\n{'=' * 130}")
    print(f"  R7 HEATMAPS — R:R BY TP × SL (per exit mode)")
    print(f"{'=' * 130}")

    for mode_short, mode_full in EXIT_MODES:
        print(f"\n  ── {mode_short} ({mode_full}) ──")
        print(f"  {'SL →':>12}", end='')
        for sl in SL_GRID:
            print(f"  {sl:>8.3f}", end='')
        print(f"  {'best_RR':>8}")
        print(f"  {'TP ↓':>12}  {'─' * (10 * len(SL_GRID) + 8)}")

        for tp in TP_GRID:
            print(f"  {tp:>12.3f}", end='')
            best_rr_row = 0.0
            for sl in SL_GRID:
                name = f"T{TP_LABELS[tp]}_S{SL_LABELS[sl]}_{mode_short}"
                r = results.get(name)
                if r:
                    rr = r['actual_rr']
                    if rr > best_rr_row:
                        best_rr_row = rr
                    marker = '★' if rr >= 1.0 else (' ' if rr >= 0.78 else '·')
                    print(f"  {rr:>7.2f}{marker}", end='')
                else:
                    print(f"  {'N/A':>8}", end='')
            print(f"  {best_rr_row:>7.2f}")

    # ── Heatmaps: Sharpe ──
    print(f"\n\n{'=' * 130}")
    print(f"  R7 HEATMAPS — SHARPE BY TP × SL (per exit mode)")
    print(f"{'=' * 130}")

    for mode_short, mode_full in EXIT_MODES:
        print(f"\n  ── {mode_short} ({mode_full}) ──")
        print(f"  {'SL →':>12}", end='')
        for sl in SL_GRID:
            print(f"  {sl:>8.3f}", end='')
        print()
        print(f"  {'TP ↓':>12}  {'─' * (10 * len(SL_GRID))}")

        for tp in TP_GRID:
            print(f"  {tp:>12.3f}", end='')
            for sl in SL_GRID:
                name = f"T{TP_LABELS[tp]}_S{SL_LABELS[sl]}_{mode_short}"
                r = results.get(name)
                if r:
                    sh = r['sharpe']
                    marker = '★' if sh >= 5.0 else (' ' if sh >= 4.0 else '·')
                    print(f"  {sh:>7.2f}{marker}", end='')
                else:
                    print(f"  {'N/A':>8}", end='')
            print()

    # ── Heatmaps: DD ──
    print(f"\n\n{'=' * 130}")
    print(f"  R7 HEATMAPS — MAX DD% BY TP × SL (per exit mode)")
    print(f"{'=' * 130}")

    for mode_short, mode_full in EXIT_MODES:
        print(f"\n  ── {mode_short} ({mode_full}) ──")
        print(f"  {'SL →':>12}", end='')
        for sl in SL_GRID:
            print(f"  {sl:>8.3f}", end='')
        print()
        print(f"  {'TP ↓':>12}  {'─' * (10 * len(SL_GRID))}")

        for tp in TP_GRID:
            print(f"  {tp:>12.3f}", end='')
            for sl in SL_GRID:
                name = f"T{TP_LABELS[tp]}_S{SL_LABELS[sl]}_{mode_short}"
                r = results.get(name)
                if r:
                    dd = r['max_dd_pct']
                    marker = '★' if dd <= 8.0 else (' ' if dd <= 10.0 else '!')
                    print(f"  {dd:>7.1f}{marker}", end='')
                else:
                    print(f"  {'N/A':>8}", end='')
            print()

    # ── Top performers ──
    print(f"\n\n{'=' * 130}")
    print(f"  R7 RANKINGS")
    print(f"{'=' * 130}")

    if ref_result:
        print(f"\n  REF (M-2SSTSX production): WR={ref_result['win_rate']:.1f}%  "
              f"R:R={ref_result['actual_rr']:.2f}  DD={ref_result['max_dd_pct']:.1f}%  "
              f"Sharpe={ref_result['sharpe']:.2f}  PF={ref_result['profit_factor']:.2f}  "
              f"Ret={ref_result['total_return_pct']:+,.1f}%")

    # Filter: DD ≤ 12% and at least 500 trades (meaningful sample)
    viable = {n: r for n, r in grid_results.items()
              if r['max_dd_pct'] <= 12.0 and r['n_trades'] >= 500}

    # ── By composite score (Sharpe × PF / DD) ──
    print(f"\n  🏆 Top 20 by Composite Score (Sharpe × PF / DD), DD ≤ 12%:")
    scored = []
    for n, r in viable.items():
        dd = max(r['max_dd_pct'], 0.1)
        score = r['sharpe'] * r['profit_factor'] / dd
        scored.append((n, score, r))
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'#':>3} {'Name':<22} {'Score':>7} {'WR':>6} {'R:R':>6} {'DD':>6} "
          f"{'Sharpe':>7} {'PF':>6} {'Ret':>10} {'Trades':>7} {'AvgW':>8} {'AvgL':>8} "
          f"{'theoRR':>7}")
    print(f"  {'─' * 120}")

    for rank, (n, score, r) in enumerate(scored[:20], 1):
        # Extract TP/SL from name to compute theoretical R:R
        parts = n.split('_')
        tp_val = int(parts[0][1:]) / 100 if parts[0][1:] != '100' else 1.0
        sl_val = int(parts[1][1:]) / 100 if parts[1][1:] != '100' else 1.0
        # Fix labels: 38→0.382, 50→0.50, 62→0.618, 79→0.786
        tp_map = {'38': 0.382, '50': 0.50, '62': 0.618, '79': 0.786, '100': 1.0}
        sl_map = {'38': 0.382, '50': 0.50, '62': 0.618, '79': 0.786, '100': 1.0}
        tp_v = tp_map.get(parts[0][1:], 0.382)
        sl_v = sl_map.get(parts[1][1:], 0.786)
        theo_rr = tp_v / sl_v

        dom_tag = ''
        if ref_result:
            if (r['win_rate'] >= ref_result['win_rate']
                    and r['actual_rr'] >= ref_result['actual_rr']
                    and r['max_dd_pct'] <= ref_result['max_dd_pct']):
                dom_tag = ' ★DOM'

        print(f"  {rank:>3} {n:<22} {score:>7.2f} {r['win_rate']:>5.1f}% "
              f"{r['actual_rr']:>5.2f} {r['max_dd_pct']:>5.1f}% "
              f"{r['sharpe']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['total_return_pct']:>+9,.1f}% {r['n_trades']:>6,} "
              f"{r['avg_win_pct']:>+7.3f}% {r['avg_loss_pct']:>7.3f}% "
              f"{theo_rr:>6.2f}{dom_tag}")

    # ── By R:R (high R:R achievers with viable DD) ──
    print(f"\n  📈 Top 20 by Actual R:R (DD ≤ 12%):")
    by_rr = sorted(viable.items(), key=lambda x: x[1]['actual_rr'], reverse=True)
    print(f"  {'#':>3} {'Name':<22} {'R:R':>6} {'WR':>6} {'DD':>6} "
          f"{'Sharpe':>7} {'PF':>6} {'Ret':>10} {'Trades':>7}")
    print(f"  {'─' * 90}")

    for rank, (n, r) in enumerate(by_rr[:20], 1):
        print(f"  {rank:>3} {n:<22} {r['actual_rr']:>5.2f} {r['win_rate']:>5.1f}% "
              f"{r['max_dd_pct']:>5.1f}% {r['sharpe']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['total_return_pct']:>+9,.1f}% {r['n_trades']:>6,}")

    # ── By Sharpe ──
    print(f"\n  ⚡ Top 20 by Sharpe (DD ≤ 12%):")
    by_sharpe = sorted(viable.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    print(f"  {'#':>3} {'Name':<22} {'Sharpe':>7} {'R:R':>6} {'WR':>6} {'DD':>6} "
          f"{'PF':>6} {'Ret':>10}")
    print(f"  {'─' * 80}")

    for rank, (n, r) in enumerate(by_sharpe[:20], 1):
        print(f"  {rank:>3} {n:<22} {r['sharpe']:>6.2f} {r['actual_rr']:>5.2f} "
              f"{r['win_rate']:>5.1f}% {r['max_dd_pct']:>5.1f}% "
              f"{r['profit_factor']:>5.2f} {r['total_return_pct']:>+9,.1f}%")

    # ── Dominators of REF ──
    if ref_result:
        print(f"\n  ★ VARIANTS THAT DOMINATE M-2SSTSX REF (WR≥, R:R≥, DD≤):")
        dominators = []
        for n, r in grid_results.items():
            if (r['win_rate'] >= ref_result['win_rate']
                    and r['actual_rr'] >= ref_result['actual_rr']
                    and r['max_dd_pct'] <= ref_result['max_dd_pct']):
                dominators.append((n, r))
        dominators.sort(key=lambda x: x[1]['sharpe'], reverse=True)

        if dominators:
            for n, r in dominators:
                d_rr = r['actual_rr'] - ref_result['actual_rr']
                d_dd = r['max_dd_pct'] - ref_result['max_dd_pct']
                d_sh = r['sharpe'] - ref_result['sharpe']
                print(f"    {n:<22} WR={r['win_rate']:.1f}% R:R={r['actual_rr']:.2f}(+{d_rr:.2f}) "
                      f"DD={r['max_dd_pct']:.1f}%({d_dd:+.1f}) Sharpe={r['sharpe']:.2f}({d_sh:+.2f}) "
                      f"PF={r['profit_factor']:.2f} Ret={r['total_return_pct']:+,.1f}%")
        else:
            print("    NONE — REF is Pareto-optimal in current grid")

    # ── R:R ≥ 1.0 achievers ──
    rr_high = {n: r for n, r in grid_results.items()
               if r['actual_rr'] >= 1.0 and r['max_dd_pct'] <= 12.0}
    if rr_high:
        print(f"\n  🎯 R:R ≥ 1.0 ACHIEVERS (DD ≤ 12%):")
        for n, r in sorted(rr_high.items(), key=lambda x: x[1]['sharpe'], reverse=True):
            print(f"    {n:<22} R:R={r['actual_rr']:.2f} WR={r['win_rate']:.1f}% "
                  f"DD={r['max_dd_pct']:.1f}% Sharpe={r['sharpe']:.2f} "
                  f"PF={r['profit_factor']:.2f} Ret={r['total_return_pct']:+,.1f}%")
    else:
        print(f"\n  🎯 R:R ≥ 1.0 ACHIEVERS (DD ≤ 12%): NONE")

    # ── R:R ≥ 1.5 achievers ──
    rr_very_high = {n: r for n, r in grid_results.items()
                    if r['actual_rr'] >= 1.5 and r['max_dd_pct'] <= 15.0}
    if rr_very_high:
        print(f"\n  🚀 R:R ≥ 1.5 ACHIEVERS (DD ≤ 15%):")
        for n, r in sorted(rr_very_high.items(), key=lambda x: x[1]['sharpe'], reverse=True):
            print(f"    {n:<22} R:R={r['actual_rr']:.2f} WR={r['win_rate']:.1f}% "
                  f"DD={r['max_dd_pct']:.1f}% Sharpe={r['sharpe']:.2f} "
                  f"PF={r['profit_factor']:.2f} Ret={r['total_return_pct']:+,.1f}%")

    # ── Best per exit mode ──
    print(f"\n  📊 Best per Exit Mode (by composite score, DD ≤ 12%):")
    for mode_short, mode_full in EXIT_MODES:
        mode_viable = {n: r for n, r in viable.items() if n.endswith(f'_{mode_short}')}
        if not mode_viable:
            print(f"    {mode_short}: no viable variants")
            continue
        best_n = max(mode_viable, key=lambda n: (
            mode_viable[n]['sharpe'] * mode_viable[n]['profit_factor']
            / max(mode_viable[n]['max_dd_pct'], 0.1)
        ))
        r = mode_viable[best_n]
        print(f"    {mode_short:<5} → {best_n:<22} R:R={r['actual_rr']:.2f} WR={r['win_rate']:.1f}% "
              f"DD={r['max_dd_pct']:.1f}% Sharpe={r['sharpe']:.2f} PF={r['profit_factor']:.2f} "
              f"Ret={r['total_return_pct']:+,.1f}%")

    # ── Exit reason breakdown for top 5 ──
    print(f"\n  🔧 Exit Reason Breakdown (top 5 by composite):")
    for rank, (n, score, r) in enumerate(scored[:5], 1):
        reasons = '  '.join(f"{k}={v}" for k, v in sorted(results[n].get('exit_reasons', {}).items()))
        print(f"    #{rank} {n:<22} {reasons}")

    # ── Save results ──
    output_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r7_results.json'
    save_results = {}
    for k, v in results.items():
        sv = {sk: sv for sk, sv in v.items() if sk != 'equity_curve'}
        save_results[k] = sv
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 130}")
    print("  DONE — R7 3D EXIT GRID SWEEP COMPLETE")
    print(f"{'=' * 130}")


if __name__ == '__main__':
    main()
