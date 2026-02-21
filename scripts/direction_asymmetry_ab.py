#!/usr/bin/env python3
"""
Direction-Aware A/B Test — Asymmetric Long/Short Parameters
=============================================================
Tests direction-specific parameter modifications ranked by feature
importance from the direction asymmetry analysis (d3de815):

  Rank 1 (~70% of EV gap): R:R levers — target & stop by direction
  Rank 2 (~20% of EV gap): WR levers — confidence gate for shorts
  Rank 3 (~10% of EV gap): Hold levers — max_hold for shorts

Each lever is tested independently vs symmetric baseline, then
the best levers are combined.

Data: 6yr (2190d), 50 symbols, exact production logic.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/direction_asymmetry_ab.py
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

# Sizing (constant across variants)
CAP_MULT = {2: 0.3, 3: 0.5, 4: 1.5, 5: 2.0}
SOFT_EQ = [(0.08, 0.50), (0.05, 0.70), (0.03, 0.85)]


@dataclass
class DirConfig:
    """Direction-aware A/B variant configuration."""
    name: str = 'A-Baseline'
    desc: str = ''

    # ── Exit parameters (can differ by direction) ──
    long_target_retrace: float = 1.5
    short_target_retrace: float = 1.5
    long_stop_mult: float = 0.382
    short_stop_mult: float = 0.382
    long_max_hold: int = 25
    short_max_hold: int = 25
    stop_decay: float = 0.0

    # ── Entry gating (can differ by direction) ──
    long_min_confidence: int = 3
    short_min_confidence: int = 3

    # ── PBT (v12.4 HybTight) ──
    partial_exit: bool = True
    partial_fraction: float = 0.5
    trail_atr_mult: float = 1.0
    breakeven_after_tp1: bool = True
    runner_max_hold_mult: float = 3.0

    # ── Portfolio (M-2SSTSX, constant) ──
    max_slots: int = 2
    dir_diversity: bool = True
    steep_cap: bool = True
    vol_adjusted: bool = True
    soft_equity: bool = True


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
# Entry scanner — supports direction-specific min_confidence
# ═══════════════════════════════════════════════════════════════════
@numba.jit(nopython=True, cache=True)
def scan_entries(
    close, high, low, volume, taker_buy, btc_close,
    lookback, threshold, btc_mom_period, btc_regime_period,
    asset_trend_period, long_min_confidence, short_min_confidence,
    rsi_period, vol_period,
    stoch_enabled, stoch_k_period, stoch_d_period,
    macd_enabled, macd_fast, macd_slow, macd_signal_period,
    atr_period, atr_multiplier, threshold_floor, threshold_cap,
):
    """Scan entries with direction-specific confidence gates."""
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
                if confidence < long_min_confidence:
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
                if confidence < short_min_confidence:
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
# Portfolio Simulator — direction-aware exit parameters
# ═══════════════════════════════════════════════════════════════════
def simulate(
    sym_close, sym_high, sym_low,
    sym_dirs, sym_confs, sym_moves,
    cfg: DirConfig,
    sym_atr=None,
) -> dict:
    """Run portfolio sim with direction-aware parameters."""
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
        to_close: list[str] = []

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

            # Direction-aware parameters
            target_retrace = cfg.long_target_retrace if d == 1 else cfg.short_target_retrace
            stop_mult = cfg.long_stop_mult if d == 1 else cfg.short_stop_mult
            max_hold = cfg.long_max_hold if d == 1 else cfg.short_max_hold

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
                    })
                    pos['notional'] = rem_not
                    pos['tp1_bar'] = i
                    pos['best_pnl'] = gain
                    pos['trail_floor'] = 0.0 if cfg.breakeven_after_tp1 else -effective_stop
                elif gain < -effective_stop:
                    _close(balance, trades, pos, sym, gain, lev, 'stop', to_close, i)
                    balance = trades[-1]['_b']
                elif pos['hold_count'] >= max_hold:
                    _close(balance, trades, pos, sym, gain, lev, 'time', to_close, i)
                    balance = trades[-1]['_b']
                else:
                    if gain > pos['best_pnl']:
                        pos['best_pnl'] = gain
                    if pos['best_pnl'] > target * 0.5 and gain < pos['best_pnl'] * 0.3:
                        _close(balance, trades, pos, sym, gain, lev, 'trail', to_close, i)
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
                    _close(balance, trades, pos, sym, gain, lev, exit_reason, to_close, i)
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
                # Confidence gate is already applied in scan_entries
                if cfg.dir_diversity:
                    same = sum(1 for p in positions.values() if p['direction'] == direction)
                    if same >= 2:
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

    return _stats(equity_curve, trades, max_dd)


def _close(balance, trades, pos, sym, gain, lev, reason, to_close, bar_idx):
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
        '_b': new_b,
    })
    to_close.append(sym)


def _stats(eq, trades, max_dd):
    n = len(trades)
    if n == 0:
        return {'n_trades': 0}

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    long_t = [t for t in trades if t['direction'] == 1]
    short_t = [t for t in trades if t['direction'] == -1]
    long_w = [t for t in long_t if t['pnl_usd'] > 0]
    short_w = [t for t in short_t if t['pnl_usd'] > 0]

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
    avg_pnl = float(np.mean([t['pnl_pct'] for t in trades]))
    n_days = len(eq) / 96
    tpd = round(n / n_days, 2) if n_days > 0 else 0.0

    def _dir_stats(tl, wl):
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
            'avg_win': round(aw_d, 3), 'avg_loss': round(al_d, 3),
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
        'trades_per_day': tpd,
        'win_rate': round(len(wins) / n * 100, 1),
        'profit_factor': round(pf, 2),
        'avg_pnl_pct': round(avg_pnl, 3),
        'avg_win_pct': round(aw, 3),
        'avg_loss_pct': round(al, 3),
        'rr': round(rr, 2),
        'long': _dir_stats(long_t, long_w),
        'short': _dir_stats(short_t, short_w),
        'exit_reasons': reasons,
    }


# ═══════════════════════════════════════════════════════════════════
# Variant Definitions
# ═══════════════════════════════════════════════════════════════════
def build_variants() -> list[DirConfig]:
    """Build A/B variants ranked by feature importance."""
    variants = []

    # ── CONTROL ──
    variants.append(DirConfig(
        name='A-Baseline',
        desc='Symmetric v12.4 I-HybTight (production)',
    ))

    # ══════════════════════════════════════════════════
    # RANK 1: R:R LEVERS (~70% of EV gap)
    # The R:R gap (2.02 vs 1.61) is the dominant driver.
    # Avg win: L 12.17% vs S 10.69%, Avg loss: L 6.01% vs S 6.62%
    # ══════════════════════════════════════════════════

    # B: Wider target for longs (they run further)
    variants.append(DirConfig(
        name='B-LongWideT',
        desc='Long target 2.0× (short stays 1.5×)',
        long_target_retrace=2.0,
    ))

    # C: Tighter target for shorts (they fade less)
    variants.append(DirConfig(
        name='C-ShortTightT',
        desc='Short target 1.0× (long stays 1.5×)',
        short_target_retrace=1.0,
    ))

    # D: Both — wider long + tighter short
    variants.append(DirConfig(
        name='D-SplitTarget',
        desc='Long 2.0× / Short 1.0× target',
        long_target_retrace=2.0,
        short_target_retrace=1.0,
    ))

    # E: Tighter stop for shorts (failed shorts lose more)
    variants.append(DirConfig(
        name='E-ShortTightS',
        desc='Short stop 0.30× (long stays 0.382×)',
        short_stop_mult=0.30,
    ))

    # F: Both target + stop for shorts
    variants.append(DirConfig(
        name='F-ShortBoth',
        desc='Short target 1.0× + stop 0.30×',
        short_target_retrace=1.0,
        short_stop_mult=0.30,
    ))

    # G: Moderate split target (less extreme)
    variants.append(DirConfig(
        name='G-ModSplitT',
        desc='Long 1.8× / Short 1.2× target (moderate)',
        long_target_retrace=1.8,
        short_target_retrace=1.2,
    ))

    # H: Wider long stop (give longs more room)
    variants.append(DirConfig(
        name='H-LongWideS',
        desc='Long stop 0.45× (short stays 0.382×)',
        long_stop_mult=0.45,
    ))

    # ══════════════════════════════════════════════════
    # RANK 2: WIN RATE LEVERS (~20% of EV gap)
    # Long WR 55.9% vs Short WR 53.7% (+2.2pp gap)
    # ══════════════════════════════════════════════════

    # I: Higher confidence gate for shorts
    variants.append(DirConfig(
        name='I-ShortConf4',
        desc='Short conf≥4 (long stays conf≥3)',
        short_min_confidence=4,
    ))

    # J: ShortConf4 + tighter target
    variants.append(DirConfig(
        name='J-Conf4+TgtT',
        desc='Short conf≥4 + target 1.0×',
        short_min_confidence=4,
        short_target_retrace=1.0,
    ))

    # ══════════════════════════════════════════════════
    # RANK 3: HOLD TIME LEVERS (~10% of EV gap)
    # Avg bars identical (15.0) but shorts may grind longer
    # ══════════════════════════════════════════════════

    # K: Shorter max_hold for shorts
    variants.append(DirConfig(
        name='K-ShortMH20',
        desc='Short max_hold=20 (long stays 25)',
        short_max_hold=20,
    ))

    # L: Shorter max_hold + tighter stop for shorts
    variants.append(DirConfig(
        name='L-ShortFast',
        desc='Short max_hold=20 + stop 0.30×',
        short_max_hold=20,
        short_stop_mult=0.30,
    ))

    # ══════════════════════════════════════════════════
    # COMBINED: Best levers from each rank
    # ══════════════════════════════════════════════════

    # M: Moderate split + short tighter stop
    variants.append(DirConfig(
        name='M-ModCombo',
        desc='L1.8/S1.2 target + S stop 0.30×',
        long_target_retrace=1.8,
        short_target_retrace=1.2,
        short_stop_mult=0.30,
    ))

    # N: Full asymmetric (all levers)
    variants.append(DirConfig(
        name='N-FullAsym',
        desc='L2.0/S1.0 tgt, S stp 0.30, S conf4, S MH20',
        long_target_retrace=2.0,
        short_target_retrace=1.0,
        short_stop_mult=0.30,
        short_min_confidence=4,
        short_max_hold=20,
    ))

    # O: Moderate full asymmetric
    variants.append(DirConfig(
        name='O-ModFull',
        desc='L1.8/S1.2 tgt, S stp 0.30, S conf4',
        long_target_retrace=1.8,
        short_target_retrace=1.2,
        short_stop_mult=0.30,
        short_min_confidence=4,
    ))

    # P: Long-favored (wider long + wider stop)
    variants.append(DirConfig(
        name='P-LongFavor',
        desc='L tgt 2.0 + stp 0.45, S stays symmetric',
        long_target_retrace=2.0,
        long_stop_mult=0.45,
    ))

    return variants


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

    w = 130
    print("=" * w)
    print("  DIRECTION-AWARE A/B TEST — Asymmetric Long/Short Parameters")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Baseline: v12.4 I-HybTight (symmetric T150_S38_H25_PBT)")
    print(f"  Data: {DATA_DAYS}d ({DATA_DAYS/365:.0f}yr), 50 symbols")
    print("=" * w)

    # ── Load data ──
    print(f"\n📊 Loading data...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC data not found"); sys.exit(1)
    btc_close = btc_data['close']

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

    # ── Build variants ──
    variants = build_variants()
    results = {}

    for vi, cfg in enumerate(variants):
        t_v = time.time()
        print(f"\n🔬 [{vi+1}/{len(variants)}] {cfg.name}: {cfg.desc}")

        # Scan entries (direction-specific confidence gate)
        sym_dirs, sym_confs, sym_moves = {}, {}, {}
        sym_close, sym_high, sym_low, sym_atr_d = {}, {}, {}, {}

        for sym, d in all_data.items():
            n = min(len(d['close']), len(btc_close))
            cn = d['close'][:n]; hn = d['high'][:n]; ln = d['low'][:n]
            vn = d['volume'][:n]; tn = d['taker_buy'][:n]; bn = btc_close[:n]

            dirs, confs, moves = scan_entries(
                cn, hn, ln, vn, tn, bn,
                LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
                ASSET_TREND_PERIOD, cfg.long_min_confidence, cfg.short_min_confidence,
                RSI_PERIOD, VOL_PERIOD,
                STOCH_ENABLED, STOCH_K, STOCH_D,
                MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
            )
            dirs, confs, moves = apply_mtf_filter(dirs, confs, moves, all_4h[sym], MTF_SMA_PERIOD)

            sym_dirs[sym] = dirs; sym_confs[sym] = confs; sym_moves[sym] = moves
            sym_close[sym] = cn; sym_high[sym] = hn; sym_low[sym] = ln
            sym_atr_d[sym] = compute_atr_pct(hn, ln, cn, ATR_PERIOD)

        # Simulate
        res = simulate(sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, cfg, sym_atr_d)
        res['config'] = {
            'long_target': cfg.long_target_retrace, 'short_target': cfg.short_target_retrace,
            'long_stop': cfg.long_stop_mult, 'short_stop': cfg.short_stop_mult,
            'long_max_hold': cfg.long_max_hold, 'short_max_hold': cfg.short_max_hold,
            'long_min_conf': cfg.long_min_confidence, 'short_min_conf': cfg.short_min_confidence,
        }
        elapsed_v = time.time() - t_v
        res['elapsed'] = round(elapsed_v, 1)
        results[cfg.name] = res

        l = res.get('long', {})
        s = res.get('short', {})
        print(f"  Ret: {res['total_return_pct']:,.1f}%  DD: {res['max_dd_pct']:.1f}%  "
              f"Sharpe: {res['sharpe']:.2f}  Calmar: {res['calmar']:.1f}  "
              f"WR: {res['win_rate']:.1f}%  PF: {res['profit_factor']:.2f}  R:R: {res['rr']:.2f}")
        print(f"  LONG:  {l.get('count',0)} trades, WR {l.get('wr',0):.1f}%, EV {l.get('avg_pnl',0):.3f}%, PF {l.get('pf',0):.2f}, R:R {l.get('rr',0):.2f}")
        print(f"  SHORT: {s.get('count',0)} trades, WR {s.get('wr',0):.1f}%, EV {s.get('avg_pnl',0):.3f}%, PF {s.get('pf',0):.2f}, R:R {s.get('rr',0):.2f}")

    # ── Summary table ──
    elapsed_total = time.time() - t0
    print(f"\n{'=' * w}")
    print("  RESULTS COMPARISON — Direction-Aware A/B Test")
    print(f"{'=' * w}")
    print(f"  {'Variant':<16} {'Desc':<42} {'Ret%':>8} {'DD%':>5} {'Sharpe':>7} {'Calmar':>7} "
          f"{'WR':>5} {'PF':>5} {'R:R':>5} {'#Tr':>6} "
          f"{'L_WR':>5} {'S_WR':>5} {'L_EV':>6} {'S_EV':>6} {'L_PF':>5} {'S_PF':>5}")
    print(f"  {'─'*16} {'─'*42} {'─'*8} {'─'*5} {'─'*7} {'─'*7} "
          f"{'─'*5} {'─'*5} {'─'*5} {'─'*6} "
          f"{'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*5} {'─'*5}")

    base = results.get('A-Baseline', {})
    base_dd = base.get('max_dd_pct', 100)
    base_sharpe = base.get('sharpe', 0)
    base_calmar = base.get('calmar', 0)

    for name in [v.name for v in variants]:
        r = results[name]
        cfg_v = next(v for v in variants if v.name == name)
        l = r.get('long', {})
        s = r.get('short', {})

        # Color markers
        dd_mark = '★' if r['max_dd_pct'] < base_dd else ' '
        sh_mark = '★' if r['sharpe'] > base_sharpe else ' '

        print(f" {dd_mark}{sh_mark}{name:<15} {cfg_v.desc:<42} "
              f"{r['total_return_pct']:>7,.1f}% {r['max_dd_pct']:>4.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.1f} "
              f"{r['win_rate']:>4.1f}% {r['profit_factor']:>4.2f} {r['rr']:>4.2f} {r['n_trades']:>6d} "
              f"{l.get('wr',0):>4.1f}% {s.get('wr',0):>4.1f}% "
              f"{l.get('avg_pnl',0):>5.3f}% {s.get('avg_pnl',0):>5.3f}% "
              f"{l.get('pf',0):>4.2f} {s.get('pf',0):>4.2f}")

    print(f"\n  ★ = better than baseline on that metric")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"{'=' * w}")

    # ── Find winners ──
    # Best Sharpe, best DD, best Calmar
    best_sharpe = max(results.items(), key=lambda x: x[1].get('sharpe', 0))
    best_dd = min(results.items(), key=lambda x: x[1].get('max_dd_pct', 100))
    best_calmar = max(results.items(), key=lambda x: x[1].get('calmar', 0))
    best_rr = max(results.items(), key=lambda x: x[1].get('rr', 0))

    print(f"\n  🏆 Best Sharpe:  {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    print(f"  🏆 Best DD:      {best_dd[0]} ({best_dd[1]['max_dd_pct']:.1f}%)")
    print(f"  🏆 Best Calmar:  {best_calmar[0]} ({best_calmar[1]['calmar']:.1f})")
    print(f"  🏆 Best R:R:     {best_rr[0]} ({best_rr[1]['rr']:.2f})")

    # ── Save JSON ──
    output = {
        'commit': commit,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'data_days': DATA_DAYS,
        'n_symbols': len(all_data),
        'variants': results,
        'best': {
            'sharpe': best_sharpe[0],
            'dd': best_dd[0],
            'calmar': best_calmar[0],
            'rr': best_rr[0],
        },
        'elapsed': round(elapsed_total, 1),
    }

    out_path = Path(__file__).parent.parent / 'logs' / 'direction_asymmetry_ab_results.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n💾 Results saved to {out_path}")


if __name__ == '__main__':
    main()
