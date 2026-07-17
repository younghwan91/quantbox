#!/usr/bin/env python3
"""
Alpha Search Round 3 — Smart Exit Architecture
================================================
R1: Alternative signal generators → ALL failed (signal isn't the edge)
R2: Parameter refinement → B-Conf3 best (WR 65.3%, R:R 1.03) but MDD 13%

R3 tests STRUCTURAL innovations never attempted:
  A: Baseline (v12.2 exact replication via custom sim)
  B: Intrabar H/L exit (use high/low for TP/SL, not just close)
  C: Confidence-adaptive trailing (tighter trail for high-conf)
  D: Direction diversity (max 2 same-direction slots)
  E: B+C combined
  F: B+C+D combined
  G: B+C+D + equity curve feedback
  H: B+C+D + move-adaptive hold time
  I: Full stack (B+C+D + equity + move-hold)

Key insight: v12.2's edge is in its filter pipeline. These innovations
change HOW we manage trades, not WHICH trades we take.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r3.py
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

from dotenv import load_dotenv
load_dotenv()

from quantbox.strategy.rubberband import (
    _compute_sma_inline,
    _compute_rsi_inline,
    _compute_stochastic_inline,
    _compute_macd_inline,
    compute_atr_pct,
    apply_mtf_confirmation,
    apply_mtf_leverage,
)

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'

# ═══════════════════════════════════════════════════════════════════
# Production parameters (v12.2 from .env)
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
CONF_LEV_STEP = 0.0
MAX_LEVERAGE = 20.0
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
MTF_LEV_MULT = 2.5
EFFECTIVE_LEV = BASE_LEVERAGE * MTF_LEV_MULT  # 3.75


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
# Part 1: Entry Opportunity Scanner (Numba)
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
    """Scan ALL entry opportunities without position tracking.

    Returns:
        dirs: +1.0 (long) / -1.0 (short) / 0 per bar
        confs: confidence score (1-7) per bar (0 if no entry)
        moves: |return| that triggered the entry
    """
    n = min(len(close), len(btc_close))
    dirs = np.zeros(n)
    confs = np.zeros(n)
    moves = np.zeros(n)

    # Pre-compute all indicators (same as compute_signals_v9)
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
                # LONG: dip in uptrend during BTC bull
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
                # SHORT: surge in downtrend → exhaustion
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
            # RISK-OFF
            if ret < -thr and btc_mom < 0:
                # LONG: sympathy dip, BTC falling
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
                # SHORT: pump fades with BTC
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
# Part 2: MTF Filter (4h trend confirmation)
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
class ExitConfig:
    """Configurable exit innovations."""
    name: str = 'Baseline'
    desc: str = ''

    # Standard exit params (v12.2 defaults)
    target_retrace: float = TARGET_RETRACE
    stop_mult: float = STOP_MULT
    stop_decay: float = STOP_DECAY
    max_hold: int = MAX_HOLD
    trail_activation: float = 0.5     # activate trail after 50% of target
    trail_factor: float = 0.3         # exit when gain < peak * 0.3

    # Innovation flags
    intrabar_hl: bool = False         # use high/low for TP/SL detection (both)
    intrabar_tp_only: bool = False    # use high/low ONLY for TP, close for SL
    conf_adaptive: bool = False       # adjust trail by confidence
    dir_diversity: bool = False       # max 2 same-direction slots
    equity_feedback: bool = False     # reduce slots during drawdowns
    move_adaptive_hold: bool = False  # shorter hold for larger moves
    min_conf_gate: int = 2            # override min confidence at sim entry level
    cooldown_bars: int = 0            # bars to skip per-symbol after exit
    conf_capital_weight: bool = False # weight slot capital by confidence
    vol_adjusted_size: bool = False   # inverse-ATR position sizing
    soft_equity_scale: bool = False   # scale position size down during DD
    steep_cap_curve: bool = False     # more aggressive conf cap weighting
    dynamic_leverage: bool = False    # leverage scales with confidence

    # Confidence-adaptive trail params
    # High conf (≥4): tighter trail → lock in more profit → R:R ↑
    high_conf_trail_activation: float = 0.35  # activate earlier
    high_conf_trail_factor: float = 0.50      # keep more of peak
    # Low conf (≤2): quicker trail → protect small gains → WR ↑
    low_conf_trail_activation: float = 0.30   # activate even earlier
    low_conf_trail_factor: float = 0.25       # loose trail, just protect entry


# ═══════════════════════════════════════════════════════════════════
# Part 4: Smart Portfolio Simulator
# ═══════════════════════════════════════════════════════════════════
def simulate_r3(
    sym_close: dict[str, np.ndarray],
    sym_high: dict[str, np.ndarray],
    sym_low: dict[str, np.ndarray],
    sym_dirs: dict[str, np.ndarray],
    sym_confs: dict[str, np.ndarray],
    sym_moves: dict[str, np.ndarray],
    cfg: ExitConfig,
    initial_capital: float = 2095.0,
    max_slots: int = 3,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    sym_atr: dict[str, np.ndarray] | None = None,  # pre-computed ATR per symbol
) -> dict:
    """Portfolio sim with configurable exit innovations."""
    symbols = sorted(sym_close.keys())
    n_bars = min(len(sym_close[s]) for s in symbols)

    balance = initial_capital
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance
    peak = balance
    max_dd = 0.0

    # Position tracking: sym -> metadata dict
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    cooldown_until: dict[str, int] = {}  # sym -> bar after which entry is allowed

    for i in range(1, n_bars):
        # ═══ Phase 1: Check exits on existing positions ═══
        to_close: list[tuple[str, float, str]] = []  # (sym, exit_pnl_ratio, reason)

        for sym, pos in list(positions.items()):
            pos['hold_count'] += 1
            c = sym_close[sym][i]
            h = sym_high[sym][i]
            l = sym_low[sym][i]
            ep = pos['entry_price']

            # Compute directional gains
            if pos['direction'] == 1:
                gain_close = (c - ep) / ep
                gain_best = (h - ep) / ep           # best within bar
                gain_worst = (l - ep) / ep           # worst within bar
            else:
                gain_close = (ep - c) / ep
                gain_best = (ep - l) / ep            # best (price dropped to low)
                gain_worst = (ep - h) / ep           # worst (price rose to high)

            # Determine max_hold for this trade
            if cfg.move_adaptive_hold:
                # Larger moves → shorter hold (should revert faster)
                ref_thr = THR_FLOOR  # 0.008 reference
                ratio = min(1.0, ref_thr / max(pos['move_size'], 0.001))
                eff_max_hold = max(4, int(cfg.max_hold * ratio))
            else:
                eff_max_hold = cfg.max_hold

            # Target and stop
            target = pos['move_size'] * cfg.target_retrace
            time_factor = 1.0 - cfg.stop_decay * (pos['hold_count'] / eff_max_hold)
            if time_factor < 0.1:
                time_factor = 0.1
            effective_stop = pos['move_size'] * cfg.stop_mult * time_factor

            # Determine trailing params (confidence-adaptive)
            if cfg.conf_adaptive:
                conf = pos['confidence']
                if conf >= 4:
                    t_act = cfg.high_conf_trail_activation
                    t_fac = cfg.high_conf_trail_factor
                elif conf <= 2:
                    t_act = cfg.low_conf_trail_activation
                    t_fac = cfg.low_conf_trail_factor
                else:
                    t_act = cfg.trail_activation
                    t_fac = cfg.trail_factor
            else:
                t_act = cfg.trail_activation
                t_fac = cfg.trail_factor

            exit_reason = None
            exit_pnl = None

            if cfg.intrabar_hl:
                # ═══ INTRABAR EXIT: check H/L for BOTH TP and SL ═══
                # Conservative: check SL first (worst case)
                if gain_worst < -effective_stop:
                    exit_reason = 'stop'
                    exit_pnl = -effective_stop

                elif gain_best >= target:
                    exit_reason = 'target'
                    exit_pnl = target

                else:
                    if gain_best > pos['best_pnl']:
                        pos['best_pnl'] = gain_best
                    if (pos['best_pnl'] > target * t_act
                            and gain_close < pos['best_pnl'] * t_fac):
                        exit_reason = 'trail'
                        exit_pnl = gain_close
                    elif pos['hold_count'] >= eff_max_hold:
                        exit_reason = 'time'
                        exit_pnl = gain_close

            elif cfg.intrabar_tp_only:
                # ═══ TP-ONLY INTRABAR: H/L for TP, close-only for SL ═══
                # Check TP using intrabar high/low
                if gain_best >= target:
                    exit_reason = 'target'
                    exit_pnl = target
                # Check SL using close only (avoid false stop-outs)
                elif gain_close < -effective_stop:
                    exit_reason = 'stop'
                    exit_pnl = gain_close
                else:
                    # Update best_pnl with intrabar best (for tighter trailing)
                    if gain_best > pos['best_pnl']:
                        pos['best_pnl'] = gain_best
                    if (pos['best_pnl'] > target * t_act
                            and gain_close < pos['best_pnl'] * t_fac):
                        exit_reason = 'trail'
                        exit_pnl = gain_close
                    elif pos['hold_count'] >= eff_max_hold:
                        exit_reason = 'time'
                        exit_pnl = gain_close

            else:
                # ═══ STANDARD EXIT: close-only (matches v12.2) ═══
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

                    elif pos['hold_count'] >= eff_max_hold:
                        exit_reason = 'time'
                        exit_pnl = gain_close

            if exit_reason is not None:
                # Funding cost for all bars held
                lev = pos.get('leverage', EFFECTIVE_LEV)
                funding = lev * funding_per_bar * pos['notional'] * pos['hold_count']
                close_cost = lev * cost_per_side * pos['notional']
                gross_pnl = exit_pnl * lev * pos['notional']
                net_pnl = gross_pnl - close_cost - funding
                balance += net_pnl
                pnl_pct = (net_pnl / pos['notional']) * 100

                trades.append({
                    'symbol': sym,
                    'direction': pos['direction'],
                    'confidence': pos['confidence'],
                    'exit_reason': exit_reason,
                    'pnl_usd': net_pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': pos['hold_count'],
                })
                to_close.append(sym)
                # Set cooldown
                if cfg.cooldown_bars > 0:
                    cooldown_until[sym] = i + cfg.cooldown_bars

        for sym in to_close:
            positions.pop(sym, None)

        # ═══ Phase 2: New entries ═══
        # Equity feedback: reduce slots during drawdowns
        if cfg.equity_feedback:
            eq = equity_curve[i - 1] if i > 0 else initial_capital
            dd_ratio = eq / peak if peak > 0 else 1.0
            if dd_ratio < 0.90:
                eff_slots = 0  # pause during severe DD
            elif dd_ratio < 0.95:
                eff_slots = max(1, max_slots - 2)
            elif dd_ratio < 0.97:
                eff_slots = max(1, max_slots - 1)
            else:
                eff_slots = max_slots
        else:
            eff_slots = max_slots

        n_active = len(positions)
        if n_active < eff_slots:
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

                # Confidence gate (sim-level override)
                if confidence < cfg.min_conf_gate:
                    continue

                # Cooldown check
                if cfg.cooldown_bars > 0 and sym in cooldown_until:
                    if i < cooldown_until[sym]:
                        continue

                # Direction diversity: max 2 same direction
                if cfg.dir_diversity:
                    same_dir = sum(
                        1 for p in positions.values()
                        if p['direction'] == direction
                    )
                    if same_dir >= 2:
                        continue

                candidates.append((sym, direction, confidence, move_size))

            # Sort by confidence (higher first), then move_size (larger first)
            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

            for sym, direction, confidence, move_size in candidates:
                if len(positions) >= eff_slots:
                    break

                per_slot = initial_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, initial_capital * 0.5)
                if per_slot < 10.0:
                    continue

                # Confidence-weighted capital allocation
                if cfg.steep_cap_curve:
                    if confidence >= 5:
                        per_slot *= 1.5
                    elif confidence >= 4:
                        per_slot *= 1.2
                    elif confidence == 3:
                        per_slot *= 0.7
                    else:
                        per_slot *= 0.5
                    per_slot = min(per_slot, initial_capital * 0.5)
                elif cfg.conf_capital_weight:
                    if confidence >= 4:
                        per_slot *= 1.4  # +40% for high confidence
                    elif confidence <= 2:
                        per_slot *= 0.6  # -40% for low confidence
                    per_slot = min(per_slot, initial_capital * 0.5)

                # Volatility-adjusted sizing (inverse ATR)
                if cfg.vol_adjusted_size and sym_atr is not None:
                    atr_arr = sym_atr.get(sym)
                    if atr_arr is not None and i < len(atr_arr) and not np.isnan(atr_arr[i]) and atr_arr[i] > 0:
                        # Target constant risk: size ∝ 1/ATR
                        median_atr = 0.01  # 1% as reference
                        vol_factor = median_atr / atr_arr[i]
                        vol_factor = max(0.4, min(vol_factor, 2.0))  # clamp
                        per_slot *= vol_factor
                        per_slot = min(per_slot, initial_capital * 0.5)

                # Soft equity scaling (reduce size during DD, don't kill slots)
                if cfg.soft_equity_scale:
                    eq_now = equity_curve[i - 1] if i > 0 else initial_capital
                    dd_pct = (peak - eq_now) / peak if peak > 0 else 0.0
                    if dd_pct > 0.08:
                        per_slot *= 0.5
                    elif dd_pct > 0.05:
                        per_slot *= 0.7
                    elif dd_pct > 0.03:
                        per_slot *= 0.85

                # Dynamic leverage by confidence
                trade_lev = EFFECTIVE_LEV
                if cfg.dynamic_leverage:
                    base_l = BASE_LEVERAGE + 0.25 * (confidence - 1)
                    trade_lev = min(base_l * MTF_LEV_MULT, 10.0)

                # Entry cost
                trade_lev = EFFECTIVE_LEV
                if cfg.dynamic_leverage:
                    base_l = BASE_LEVERAGE + 0.25 * (confidence - 1)
                    trade_lev = min(base_l * MTF_LEV_MULT, 10.0)

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
            lev = pos.get('leverage', EFFECTIVE_LEV)
            total_unrealized += gain * lev * pos['notional']

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

    # Exit reason breakdown
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

    print("=" * 100)
    print("  ALPHA SEARCH ROUND 3 — SMART EXIT ARCHITECTURE")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Structural innovations: Intrabar H/L, Conf-Adaptive Trail,")
    print(f"  Direction Diversity, Equity Feedback, Move-Adaptive Hold")
    print(f"  Target: WR > 64.2% AND R:R > 0.98 AND MDD ≤ 11.1%")
    print("=" * 100)

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

    # ── Scan entry opportunities ──
    print(f"\n🔍 Scanning entry opportunities (Numba compile on first run)...")
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

        # Apply MTF filter
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

    # Pre-compute ATR% per symbol (for vol-adjusted sizing)
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

    # ── Define variants ──
    variants = [
        ExitConfig(
            name='A-Base',
            desc='Baseline (close-only exit, no innovations)',
        ),
        ExitConfig(
            name='B-TPonly',
            desc='TP-only intrabar (H/L for TP, close for SL)',
            intrabar_tp_only=True,
        ),
        ExitConfig(
            name='C-DirDiv',
            desc='Direction diversity (max 2 same-direction slots)',
            dir_diversity=True,
        ),
        ExitConfig(
            name='D-ConfTrail',
            desc='Conf-adaptive trailing (tighter trail for high-conf)',
            conf_adaptive=True,
        ),
        ExitConfig(
            name='E-TP+Dir',
            desc='TP-only intrabar + direction diversity',
            intrabar_tp_only=True,
            dir_diversity=True,
        ),
        ExitConfig(
            name='F-TP+Conf',
            desc='TP-only intrabar + conf-adaptive trail',
            intrabar_tp_only=True,
            conf_adaptive=True,
        ),
        ExitConfig(
            name='G-Triple',
            desc='TP-only + conf-trail + dir diversity',
            intrabar_tp_only=True,
            conf_adaptive=True,
            dir_diversity=True,
        ),
        ExitConfig(
            name='H-G+MoveH',
            desc='G + move-adaptive hold time',
            intrabar_tp_only=True,
            conf_adaptive=True,
            dir_diversity=True,
            move_adaptive_hold=True,
        ),
        ExitConfig(
            name='I-IntraHL',
            desc='Full intrabar H/L (both TP+SL) — R3 control',
            intrabar_hl=True,
        ),
        # === R3.2 Variants: focused refinements ===
        ExitConfig(
            name='J-Conf3',
            desc='Conf≥3 entry gate (filter low-quality entries)',
            min_conf_gate=3,
        ),
        ExitConfig(
            name='K-C3+Dir',
            desc='Conf≥3 + direction diversity',
            min_conf_gate=3,
            dir_diversity=True,
        ),
        ExitConfig(
            name='L-C3+CT+D',
            desc='Conf≥3 + conf-trail + dir diversity',
            min_conf_gate=3,
            conf_adaptive=True,
            dir_diversity=True,
        ),
        ExitConfig(
            name='M-Cool5',
            desc='5-bar cooldown per-symbol after exit',
            cooldown_bars=5,
        ),
        ExitConfig(
            name='N-CapWt',
            desc='Confidence-weighted capital (×1.4 high / ×0.6 low)',
            conf_capital_weight=True,
        ),
        ExitConfig(
            name='O-Best',
            desc='Conf≥3 + dir div + conf trail + cap weight',
            min_conf_gate=3,
            dir_diversity=True,
            conf_adaptive=True,
            conf_capital_weight=True,
        ),
        # === R3.3 Variants: the missing combos ===
        ExitConfig(
            name='P-C3DirCap',
            desc='Conf≥3 + dir div + cap weight (no conf trail)',
            min_conf_gate=3,
            dir_diversity=True,
            conf_capital_weight=True,
        ),
        ExitConfig(
            name='Q-DirCap',
            desc='Dir diversity + cap weight (no conf gate)',
            dir_diversity=True,
            conf_capital_weight=True,
        ),
        ExitConfig(
            name='R-C3Cap',
            desc='Conf≥3 + cap weight (no dir div)',
            min_conf_gate=3,
            conf_capital_weight=True,
        ),
        # === R4: MDD-focused innovations ===
        ExitConfig(
            name='S-VolSize',
            desc='Vol-adjusted sizing (inverse ATR)',
            vol_adjusted_size=True,
        ),
        ExitConfig(
            name='T-SoftEq',
            desc='Soft equity scaling (reduce size in DD)',
            soft_equity_scale=True,
        ),
        ExitConfig(
            name='U-SteepCap',
            desc='Steep cap curve (0.5/0.7/1.2/1.5× by conf)',
            steep_cap_curve=True,
        ),
        ExitConfig(
            name='V-DynLev',
            desc='Dynamic leverage by confidence (+0.25/level)',
            dynamic_leverage=True,
        ),
        ExitConfig(
            name='W-C3DV+SE',
            desc='Conf≥3 + dir div + vol sizing + soft equity',
            min_conf_gate=3,
            dir_diversity=True,
            vol_adjusted_size=True,
            soft_equity_scale=True,
        ),
        ExitConfig(
            name='X-C3DV+SC',
            desc='Conf≥3 + dir div + steep cap + soft equity',
            min_conf_gate=3,
            dir_diversity=True,
            steep_cap_curve=True,
            soft_equity_scale=True,
        ),
        ExitConfig(
            name='Y-Kitchen',
            desc='Conf≥3 + dir div + steep cap + vol size + soft eq',
            min_conf_gate=3,
            dir_diversity=True,
            steep_cap_curve=True,
            vol_adjusted_size=True,
            soft_equity_scale=True,
        ),
    ]

    # ── Run each variant ──
    results: dict[str, dict] = {}

    for cfg in variants:
        print(f"\n{'─' * 80}")
        print(f"  ⚡ {cfg.name}: {cfg.desc}")
        print(f"{'─' * 80}")

        t0 = time.time()
        result = simulate_r3(
            sym_close, sym_high, sym_low,
            sym_dirs, sym_confs, sym_moves,
            cfg=cfg,
            sym_atr=sym_atr,
        )
        elapsed = time.time() - t0

        wr_ok = '✓' if result['win_rate'] > 64.2 else '✗'
        rr_ok = '✓' if result['actual_rr'] > 0.98 else '✗'
        dd_ok = '✓' if result['max_dd_pct'] <= 11.1 else '✗'
        star = '★' if (result['win_rate'] > 64.2 and result['actual_rr'] > 0.98
                       and cfg.name != 'A-Base') else ''

        result['description'] = cfg.desc
        result['elapsed'] = round(elapsed, 2)
        result['innovations'] = {
            'intrabar_hl': cfg.intrabar_hl,
            'intrabar_tp_only': cfg.intrabar_tp_only,
            'conf_adaptive': cfg.conf_adaptive,
            'dir_diversity': cfg.dir_diversity,
            'equity_feedback': cfg.equity_feedback,
            'move_adaptive_hold': cfg.move_adaptive_hold,
        }
        results[cfg.name] = result

        # Exit reason breakdown
        reasons_str = '  '.join(f"{k}={v}" for k, v in sorted(result['exit_reasons'].items()))

        print(f"  Ret: {result['total_return_pct']:+,.1f}%  "
              f"Sharpe: {result['sharpe']:.2f}  DD: {result['max_dd_pct']:.1f}%[{dd_ok}]  "
              f"Calmar: {result['calmar']:.1f}")
        print(f"  Trades: {result['n_trades']:,} ({result['trades_per_day']}/day)  "
              f"Bars: {result['avg_bars_held']:.1f}  "
              f"WR: {result['win_rate']:.1f}%[{wr_ok}]  R:R: {result['actual_rr']:.2f}[{rr_ok}]  "
              f"PF: {result['profit_factor']:.2f}  {star} [{elapsed:.1f}s]")
        print(f"  Exits: {reasons_str}")

    # ── Comparison Table ──
    names = list(results.keys())
    base = results['A-Base']
    cw = 11

    print(f"\n\n{'=' * 130}")
    print(f"  R3 COMPARATIVE RESULTS")
    print(f"  Target: WR > {base['win_rate']}% AND R:R > {base['actual_rr']} AND DD ≤ {base['max_dd_pct']}%")
    print(f"{'=' * 130}\n")

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

    # ── Delta vs Base ──
    print(f"\n{'─' * 130}")
    print(f"  DELTA vs A-Base")
    print(f"{'─' * 130}\n")

    for n in names[1:]:
        r = results[n]
        d_wr = r['win_rate'] - base['win_rate']
        d_rr = r['actual_rr'] - base['actual_rr']
        d_ret = r['total_return_pct'] - base['total_return_pct']
        d_sh = r['sharpe'] - base['sharpe']
        d_dd = r['max_dd_pct'] - base['max_dd_pct']
        wr_ok = '✓' if d_wr > 0 else '✗'
        rr_ok = '✓' if d_rr > 0 else '✗'
        dd_ok = '✓' if d_dd <= 0 else '✗'
        triple = '★ TRIPLE WIN' if d_wr > 0 and d_rr > 0 and d_dd <= 0 else ''
        dual = '◆ DUAL WIN' if d_wr > 0 and d_rr > 0 and not triple else ''
        tag = triple or dual
        print(f"  {n:<12} ΔWR={d_wr:>+5.1f}%p[{wr_ok}]  ΔR:R={d_rr:>+5.2f}[{rr_ok}]  "
              f"ΔDD={d_dd:>+5.1f}%[{dd_ok}]  ΔRet={d_ret:>+8,.1f}%  ΔSharpe={d_sh:>+5.2f}  {tag}")

    # ── Verdict ──
    print(f"\n{'=' * 130}")
    print("  VERDICT")
    print(f"{'=' * 130}\n")

    triple_winners = [n for n in names[1:] if (
        results[n]['win_rate'] > base['win_rate']
        and results[n]['actual_rr'] > base['actual_rr']
        and results[n]['max_dd_pct'] <= base['max_dd_pct']
    )]
    dual_winners = [n for n in names[1:] if (
        results[n]['win_rate'] > base['win_rate']
        and results[n]['actual_rr'] > base['actual_rr']
        and n not in triple_winners
    )]

    if triple_winners:
        print(f"  ★ TRIPLE WINS (WR↑ + R:R↑ + DD↓): {', '.join(triple_winners)}")
        for n in triple_winners:
            r = results[n]
            print(f"    {n}: WR {r['win_rate']:.1f}% R:R {r['actual_rr']:.2f} "
                  f"DD {r['max_dd_pct']:.1f}% Sharpe {r['sharpe']:.2f} "
                  f"Ret {r['total_return_pct']:+,.1f}%")

    if dual_winners:
        print(f"  ◆ DUAL WINS (WR↑ + R:R↑, DD may increase): {', '.join(dual_winners)}")
        for n in dual_winners:
            r = results[n]
            print(f"    {n}: WR {r['win_rate']:.1f}% R:R {r['actual_rr']:.2f} "
                  f"DD {r['max_dd_pct']:.1f}% Sharpe {r['sharpe']:.2f} "
                  f"Ret {r['total_return_pct']:+,.1f}%")

    if not triple_winners and not dual_winners:
        print("  No variant achieves BOTH WR and R:R improvement.")
        best_rr = max(names[1:], key=lambda n: results[n]['actual_rr'])
        best_wr = max(names[1:], key=lambda n: results[n]['win_rate'])
        best_sh = max(names[1:], key=lambda n: results[n]['sharpe'])
        print(f"  Best R:R:    {best_rr} → {results[best_rr]['actual_rr']:.2f}")
        print(f"  Best WR:     {best_wr} → {results[best_wr]['win_rate']:.1f}%")
        print(f"  Best Sharpe: {best_sh} → {results[best_sh]['sharpe']:.2f}")

    # ── Save ──
    output_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r3_results.json'
    save_results = {}
    for k, v in results.items():
        sv = {sk: sv for sk, sv in v.items() if sk != 'equity_curve'}
        save_results[k] = sv
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"\n{'=' * 130}")
    print("  DONE")
    print(f"{'=' * 130}")


if __name__ == '__main__':
    main()
