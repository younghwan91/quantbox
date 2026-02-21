#!/usr/bin/env python3
"""
Partial Exit + Trailing Stop A/B Test
======================================
Goal: Test if splitting exit (50% at target + 50% trailing) outperforms
the current all-or-nothing 0.382 exit.

The current strategy exits 100% at 0.382 retrace (~1.91% on a 5% move).
This test keeps half the position and trails the rest with ATR-based stops.

Variants:
  A: Baseline — 100% exit at 0.382 (current E-3Slot config)
  B: 50% exit at 0.382 + 50% ATR trailing (1.5× ATR, no cap)
  C: 50% exit at 0.382 + 50% ATR trailing (2.0× ATR, wider trail)
  D: 50% exit at 0.382 + 50% ATR trailing (1.0× ATR, tighter trail)
  E: 50% exit at 0.382 + 50% breakeven + ATR trailing (1.5× ATR)
     (move stop to breakeven after 0.382 hit, then trail from there)
  F: 50% exit at 0.500 + 50% ATR trailing (1.5× ATR)
     (test higher first-exit target)

Binance Server-Side Note:
  Binance USDT-M supports TRAILING_STOP_MARKET with callbackRate (0.1-5%)
  and optional activationPrice. This can be sent server-side after partial exit.
  Implementation plan if variants win:
    1. On target hit: market-close 50%, then place TRAILING_STOP_MARKET
       with callbackRate=1~2% & activationPrice=entry for remaining qty.
    2. Server handles the trailing — no client polling dependency.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/partial_exit_trailing_ab.py
"""

import sys
import time
import json
import subprocess
import numba
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_v9,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_atr_pct,
)

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'


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
    pattern = f'{symbol}_{interval}_{days}d_*.npz'
    matches = sorted(CACHE_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_15m(symbol: str, days: int) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None or not cache.exists():
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None or not cache.exists():
        return None
    d = np.load(cache)
    return d['close']


# ═══════════════════════════════════════════════════════════════════════
#  Signal Generation (reuse from filter_relaxation_ab.py)
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(
    data: dict, btc: np.ndarray, close_4h: np.ndarray | None,
    cfg: RubberBandConfig,
    atr_period: int, atr_mult: float, floor: float, cap: float,
    mtf_sma: int, mtf_lev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate signals with current E-3Slot config."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]

    sigs = compute_signals_v9(
        c, h, l, v, t, b,
        cfg.lookback, cfg.threshold, cfg.target_retrace, cfg.stop_mult,
        cfg.max_hold, cfg.stop_decay, cfg.btc_directional_period,
        cfg.btc_regime_period, cfg.asset_trend_period,
        cfg.min_confidence, cfg.base_leverage,
        cfg.confidence_leverage_step, cfg.max_leverage,
        cfg.rsi_period, cfg.vol_period,
        cfg.dynamic_target_enabled,
        cfg.target_conf3, cfg.target_conf4, cfg.target_conf5,
        cfg.stoch_enabled, cfg.stoch_k_period, cfg.stoch_d_period,
        cfg.macd_enabled, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal_period,
        atr_period=atr_period, atr_multiplier=atr_mult,
        threshold_floor=floor, threshold_cap=cap,
    )

    if close_4h is not None:
        sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
        sigs = apply_mtf_leverage(sigs, mtf_lev)

    return c, sigs


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Simulator with Partial Exit + Trailing
# ═══════════════════════════════════════════════════════════════════════

def simulate_portfolio_trailing(
    all_close: dict[str, np.ndarray],
    all_high: dict[str, np.ndarray],
    all_low: dict[str, np.ndarray],
    all_signals: dict[str, np.ndarray],
    all_atr_pct: dict[str, np.ndarray],
    initial_capital: float = 2095.0,
    max_slots: int = 3,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    # Partial exit params:
    partial_exit: bool = False,
    partial_fraction: float = 0.5,
    first_target_retrace: float = 0.382,
    # Trailing params:
    trail_atr_mult: float = 1.5,
    breakeven_after_tp1: bool = False,
) -> dict:
    """Portfolio sim with optional split-exit strategy.

    When partial_exit=False: behaves identically to current strategy
    (signals encode full entry/exit lifecycle).

    When partial_exit=True:
    - On signal entry: full position opened
    - Track per-bar gain & move_size from signal transitions
    - When gain >= move_size * first_target_retrace:
      → Close partial_fraction of position (record as trade)
      → Remaining half trails with ATR-based stop
      → If breakeven_after_tp1: trailing stop floor = entry price
    - Trailing stop: entry_price + direction * (highest_since_tp1 - trail_atr_mult * ATR)
    """
    symbols = sorted(all_close.keys())
    n_bars = min(len(all_close[s]) for s in symbols)

    balance = initial_capital
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance

    # Position tracking
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    peak = balance
    max_dd = 0.0

    for i in range(1, n_bars):
        # ── Phase 1: Update existing positions ──
        symbols_to_close = []
        for sym, pos in list(positions.items()):
            close_arr = all_close[sym]
            sig_arr = all_signals[sym]
            atr = all_atr_pct.get(sym)

            current_price = close_arr[i]
            prev_price = close_arr[i - 1]
            bar_ret = current_price / prev_price - 1.0

            direction = 1.0 if pos['signal'] > 0 else -1.0
            leverage = abs(pos['signal'])

            # PnL update for all active sub-positions
            for sub in pos['subs']:
                if sub['closed']:
                    continue
                pnl = direction * leverage * bar_ret * sub['notional']
                pnl -= abs(leverage) * funding_per_bar * sub['notional']
                sub['unrealized'] = sub.get('unrealized', 0.0) + pnl

            # Current gain from entry
            if direction > 0:
                gain = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                gain = (pos['entry_price'] - current_price) / pos['entry_price']

            if gain > pos['best_pnl']:
                pos['best_pnl'] = gain

            pos['bars_held'] += 1
            move_size = pos['move_size']
            target = move_size * first_target_retrace

            # ── Partial exit check ──
            if partial_exit and not pos.get('tp1_hit') and gain >= target:
                pos['tp1_hit'] = True
                pos['tp1_bar'] = i
                pos['tp1_price'] = current_price
                pos['highest_since_tp1'] = gain

                # Close the "first half" sub-position
                sub_tp1 = pos['subs'][0]  # first sub = TP1 target
                close_cost = leverage * cost_per_side * sub_tp1['notional']
                net_pnl = sub_tp1['unrealized'] - close_cost
                balance += net_pnl
                sub_tp1['closed'] = True

                pnl_pct = net_pnl / sub_tp1['notional'] * 100
                trades.append({
                    'symbol': sym, 'signal': pos['signal'],
                    'entry_bar': pos['entry_bar'], 'exit_bar': i,
                    'bars_held': pos['bars_held'],
                    'notional': sub_tp1['notional'],
                    'pnl_usd': net_pnl, 'pnl_pct': pnl_pct,
                    'exit_type': 'TP1',
                })

                if breakeven_after_tp1:
                    pos['trail_floor'] = 0.0  # gain >= 0 (breakeven)
                else:
                    pos['trail_floor'] = -move_size * 0.786  # keep original SL

            # ── Trailing stop update for runner sub ──
            if partial_exit and pos.get('tp1_hit'):
                if gain > pos.get('highest_since_tp1', 0):
                    pos['highest_since_tp1'] = gain

                # ATR trailing stop
                atr_val = 0.02  # fallback 2%
                if atr is not None and i < len(atr) and not np.isnan(atr[i]):
                    atr_val = atr[i]

                trail_stop = pos['highest_since_tp1'] - trail_atr_mult * atr_val

                # Apply floor (breakeven or original SL)
                floor_val = pos.get('trail_floor', -move_size * 0.786)
                if trail_stop < floor_val:
                    trail_stop = floor_val

                # Check if trailing stop hit
                runner_sub = pos['subs'][1] if len(pos['subs']) > 1 else None
                if runner_sub and not runner_sub['closed']:
                    if gain <= trail_stop:
                        close_cost = leverage * cost_per_side * runner_sub['notional']
                        net_pnl = runner_sub['unrealized'] - close_cost
                        balance += net_pnl
                        runner_sub['closed'] = True

                        pnl_pct = net_pnl / runner_sub['notional'] * 100
                        trades.append({
                            'symbol': sym, 'signal': pos['signal'],
                            'entry_bar': pos['entry_bar'], 'exit_bar': i,
                            'bars_held': pos['bars_held'],
                            'notional': runner_sub['notional'],
                            'pnl_usd': net_pnl, 'pnl_pct': pnl_pct,
                            'exit_type': 'TRAIL',
                        })
                        symbols_to_close.append(sym)

            # ── Full exit check (for non-partial or before TP1) ──
            current_sig = sig_arr[i] if i < len(sig_arr) else 0.0
            prev_sig = sig_arr[i - 1] if i - 1 < len(sig_arr) else 0.0

            if not partial_exit:
                # Original behavior: signal-driven exit
                if current_sig == 0.0 and prev_sig != 0.0:
                    symbols_to_close.append(sym)
                elif current_sig != 0.0 and np.sign(current_sig) != np.sign(pos['signal']):
                    symbols_to_close.append(sym)
            else:
                # Partial mode: before TP1 hit, still use signal exit
                if not pos.get('tp1_hit'):
                    if current_sig == 0.0 and prev_sig != 0.0:
                        symbols_to_close.append(sym)
                    elif current_sig != 0.0 and np.sign(current_sig) != np.sign(pos['signal']):
                        symbols_to_close.append(sym)
                else:
                    # After TP1: check if all subs are closed
                    all_closed = all(s['closed'] for s in pos['subs'])
                    if all_closed:
                        symbols_to_close.append(sym)
                    # Also force-close runner on signal reversal
                    elif current_sig != 0.0 and np.sign(current_sig) != np.sign(pos['signal']):
                        symbols_to_close.append(sym)

        # ── Close positions ──
        for sym in set(symbols_to_close):
            if sym not in positions:
                continue
            pos = positions.pop(sym)
            leverage = abs(pos['signal'])

            for sub in pos['subs']:
                if sub['closed']:
                    continue
                close_cost = leverage * cost_per_side * sub['notional']
                net_pnl = sub['unrealized'] - close_cost
                balance += net_pnl
                sub['closed'] = True

                pnl_pct = net_pnl / sub['notional'] * 100
                exit_type = 'SIGNAL' if not pos.get('tp1_hit') else 'TRAIL_FORCE'
                trades.append({
                    'symbol': sym, 'signal': pos['signal'],
                    'entry_bar': pos['entry_bar'], 'exit_bar': i,
                    'bars_held': pos['bars_held'],
                    'notional': sub['notional'],
                    'pnl_usd': net_pnl, 'pnl_pct': pnl_pct,
                    'exit_type': exit_type,
                })

        # ── Phase 2: New entries ──
        n_active = len(positions)
        if n_active < max_slots:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                sig_arr = all_signals[sym]
                if i >= len(sig_arr):
                    continue
                current_sig = sig_arr[i]
                prev_sig = sig_arr[i - 1] if i > 0 else 0.0
                if current_sig != 0.0 and prev_sig == 0.0:
                    candidates.append((sym, current_sig))

            candidates.sort(key=lambda x: abs(x[1]), reverse=True)

            for sym, sig in candidates:
                if len(positions) >= max_slots:
                    break
                per_slot = initial_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, initial_capital * 0.5)
                if per_slot < 10.0:
                    continue

                leverage = abs(sig)
                notional = per_slot
                entry_cost = leverage * cost_per_side * notional
                balance -= entry_cost

                close_arr = all_close[sym]
                entry_price = close_arr[i]

                # Compute move_size from lookback
                lookback = 8
                if i >= lookback:
                    move_size = abs(close_arr[i] / close_arr[i - lookback] - 1.0)
                else:
                    move_size = 0.03

                if partial_exit:
                    # Split into two sub-positions
                    half = notional * partial_fraction
                    subs = [
                        {'notional': half, 'unrealized': 0.0, 'closed': False, 'type': 'TP1'},
                        {'notional': notional - half, 'unrealized': 0.0, 'closed': False, 'type': 'RUNNER'},
                    ]
                else:
                    subs = [{'notional': notional, 'unrealized': 0.0, 'closed': False, 'type': 'FULL'}]

                positions[sym] = {
                    'signal': sig,
                    'entry_price': entry_price,
                    'entry_bar': i,
                    'move_size': move_size,
                    'bars_held': 0,
                    'best_pnl': 0.0,
                    'subs': subs,
                    'tp1_hit': False,
                    'highest_since_tp1': 0.0,
                    'trail_floor': -move_size * 0.786,
                }

        # Equity curve
        total_unrealized = 0.0
        for pos in positions.values():
            for sub in pos['subs']:
                if not sub['closed']:
                    total_unrealized += sub.get('unrealized', 0.0)
        equity_curve[i] = balance + total_unrealized

        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # ── Stats ──
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    n_wins = len(wins)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    total_return = (equity_curve[-1] / initial_capital - 1.0) * 100
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = sum(abs(t['pnl_usd']) for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    eq_rets = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    if len(eq_rets) > 0 and np.std(eq_rets) > 0:
        sharpe = np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)
    else:
        sharpe = 0.0

    calmar = total_return / (max_dd * 100) if max_dd > 0 else 0.0
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

    # Exit type breakdown
    exit_types = {}
    for t in trades:
        et = t.get('exit_type', 'SIGNAL')
        if et not in exit_types:
            exit_types[et] = {'count': 0, 'total_pnl': 0.0, 'pnl_pcts': []}
        exit_types[et]['count'] += 1
        exit_types[et]['total_pnl'] += t['pnl_usd']
        exit_types[et]['pnl_pcts'].append(t['pnl_pct'])

    for et in exit_types:
        pcts = exit_types[et]['pnl_pcts']
        exit_types[et]['avg_pnl_pct'] = round(np.mean(pcts), 3) if pcts else 0
        exit_types[et]['win_rate'] = round(
            len([p for p in pcts if p > 0]) / len(pcts) * 100, 1
        ) if pcts else 0
        del exit_types[et]['pnl_pcts']

    # Avg win/loss for R:R calculation
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 1
    actual_rr = avg_win / avg_loss if avg_loss > 0 else 999

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'total_return_pct': round(total_return, 1),
        'final_balance': round(equity_curve[-1], 2),
        'initial_capital': initial_capital,
        'max_dd_pct': round(max_dd * 100, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 1),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'avg_pnl_pct': round(avg_pnl_pct, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'actual_rr': round(actual_rr, 2),
        'exit_types': exit_types,
        'max_slots': max_slots,
    }


def yearly_breakdown(result: dict) -> list[dict]:
    bars_per_year = 96 * 365
    trades = result['trades']
    capital = result['initial_capital']
    years = []
    for y in range(3):
        start_bar = y * bars_per_year
        end_bar = (y + 1) * bars_per_year
        year_trades = [t for t in trades if start_bar <= t['entry_bar'] < end_bar]
        n_t = len(year_trades)
        wins = [t for t in year_trades if t['pnl_usd'] > 0]
        wr = len(wins) / n_t * 100 if n_t > 0 else 0
        total_pnl = sum(t['pnl_usd'] for t in year_trades)
        ret_pct = total_pnl / capital * 100

        eq = result['equity_curve']
        eq_slice = eq[start_bar:min(end_bar, len(eq))]
        if len(eq_slice) > 1:
            pk = np.maximum.accumulate(eq_slice)
            dd = (pk - eq_slice) / np.maximum(pk, 1e-10)
            year_max_dd = float(np.max(dd)) * 100
            eq_rets = np.diff(eq_slice) / np.maximum(eq_slice[:-1], 1e-10)
            eq_rets = eq_rets[~np.isnan(eq_rets)]
            yr_sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)) if np.std(eq_rets) > 0 else 0
        else:
            year_max_dd = 0
            yr_sharpe = 0

        years.append({
            'year': y + 1,
            'label': ['2023-02→2024-02', '2024-02→2025-02', '2025-02→2026-02'][y],
            'return_pct': round(ret_pct, 1),
            'trades': n_t,
            'win_rate': round(wr, 1),
            'max_dd_pct': round(year_max_dd, 1),
            'sharpe': round(yr_sharpe, 2),
        })
    return years


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    commit = _get_git_commit()
    settings = get_settings()

    print("=" * 90)
    print("  PARTIAL EXIT + TRAILING STOP A/B TEST")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Baseline: E-3Slot (conf≥2, ATR 2.0×, floor 0.8%, MTF block, 3 slots)")
    print("=" * 90)

    SYMBOLS_50 = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
        'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT', 'BCHUSDT',
        'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
        'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT',
        'EGLDUSDT', 'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT',
        'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'LDOUSDT', 'COMPUSDT',
        'MKRUSDT', 'SNXUSDT', 'VETUSDT', 'EOSUSDT', 'APEUSDT',
    ]

    DAYS = 1095
    CAPITAL = 2095.0
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # Config: E-3Slot
    cfg = RubberBandConfig()
    cfg.lookback = settings.rb_lookback
    cfg.threshold = settings.rb_threshold
    cfg.target_retrace = settings.rb_target_retrace
    cfg.stop_mult = settings.rb_stop_mult
    cfg.max_hold = settings.rb_max_hold
    cfg.stop_decay = settings.rb_stop_decay
    cfg.btc_directional_period = settings.rb_btc_directional_period
    cfg.btc_regime_period = settings.rb_btc_regime_period
    cfg.asset_trend_period = settings.rb_asset_trend_period
    cfg.base_leverage = settings.rb_base_leverage
    cfg.confidence_leverage_step = settings.rb_confidence_leverage_step
    cfg.max_leverage = settings.rb_max_leverage
    cfg.rsi_period = settings.rb_rsi_period
    cfg.vol_period = settings.rb_vol_period
    cfg.stoch_enabled = settings.rb_stoch_enabled
    cfg.stoch_k_period = settings.rb_stoch_k_period
    cfg.stoch_d_period = settings.rb_stoch_d_period
    cfg.macd_enabled = settings.rb_macd_enabled
    cfg.macd_fast = settings.rb_macd_fast
    cfg.macd_slow = settings.rb_macd_slow
    cfg.macd_signal_period = settings.rb_macd_signal_period
    cfg.dynamic_target_enabled = settings.rb_dynamic_target_enabled
    cfg.target_conf3 = settings.rb_target_conf3
    cfg.target_conf4 = settings.rb_target_conf4
    cfg.target_conf5 = settings.rb_target_conf5
    cfg.min_confidence = settings.rb_min_confidence  # 2 (E-3Slot)

    ATR_PERIOD = settings.rb_atr_period     # 48
    ATR_MULT = settings.rb_atr_multiplier   # 2.0
    FLOOR = settings.rb_threshold_floor     # 0.008
    CAP = settings.rb_threshold_cap         # 0.06
    MTF_SMA = settings.rb_mtf_sma_period    # 12
    MTF_LEV = settings.rb_mtf_leverage_mult # 2.5

    # ── Variant definitions ──
    # (name, desc, partial, fraction, first_target, trail_atr, breakeven)
    variants = [
        ('A-Base',    '100% exit at 0.382 (current)',
         False, 0.0, 0.382, 0.0, False),
        ('B-Trail15', '50% at 0.382 + 50% trail 1.5×ATR',
         True, 0.5, 0.382, 1.5, False),
        ('C-Trail20', '50% at 0.382 + 50% trail 2.0×ATR (wider)',
         True, 0.5, 0.382, 2.0, False),
        ('D-Trail10', '50% at 0.382 + 50% trail 1.0×ATR (tighter)',
         True, 0.5, 0.382, 1.0, False),
        ('E-BE15',    '50% at 0.382 + breakeven + trail 1.5×ATR',
         True, 0.5, 0.382, 1.5, True),
        ('F-High15',  '50% at 0.500 + 50% trail 1.5×ATR',
         True, 0.5, 0.500, 1.5, False),
    ]

    # ── Load data ──
    print(f"\n📊 Loading 50-symbol data ({DAYS}d)...")
    btc_data = load_15m('BTCUSDT', DAYS)
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        h4 = load_4h(sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # ── Generate signals ONCE (shared across all variants — same entry logic) ──
    print("\n⚡ Generating signals (shared across variants)...")
    sym_close: dict[str, np.ndarray] = {}
    sym_high: dict[str, np.ndarray] = {}
    sym_low: dict[str, np.ndarray] = {}
    sym_sigs: dict[str, np.ndarray] = {}
    sym_atr: dict[str, np.ndarray] = {}

    for sym in available:
        data = all_data[sym]
        close_4h = all_4h.get(sym)
        c, sigs = generate_signals(
            data, btc_close, close_4h, cfg,
            ATR_PERIOD, ATR_MULT, FLOOR, CAP, MTF_SMA, MTF_LEV,
        )
        sym_close[sym] = c
        sym_sigs[sym] = sigs

        # Also need high/low for ATR in trailing
        h = data['high'][:len(c)]
        l = data['low'][:len(c)]
        sym_high[sym] = h
        sym_low[sym] = l

        # Compute ATR% for trailing stop
        atr = compute_atr_pct(h, l, c, ATR_PERIOD)
        sym_atr[sym] = atr

    # Left-pad
    max_bars = max(len(v) for v in sym_close.values())
    price_ids = {id(sym_close), id(sym_high), id(sym_low)}
    sig_id = id(sym_sigs)
    for sym in list(sym_close.keys()):
        for arr_dict in [sym_close, sym_high, sym_low, sym_sigs, sym_atr]:
            arr = arr_dict[sym]
            if len(arr) < max_bars:
                pad_n = max_bars - len(arr)
                did = id(arr_dict)
                if did in price_ids:
                    fill = arr[0]
                elif did == sig_id:
                    fill = 0.0
                else:
                    fill = np.nan
                arr_dict[sym] = np.concatenate([np.full(pad_n, fill), arr])

    print(f"  {len(available)} symbols × {max_bars:,} bars ready")

    # ── Run variants ──
    results: dict[str, dict] = {}

    for name, desc, partial, fraction, first_tgt, trail_atr, be in variants:
        print(f"\n{'─' * 70}")
        print(f"  ⚡ {name}: {desc}")
        print(f"{'─' * 70}")

        t0 = time.time()
        result = simulate_portfolio_trailing(
            sym_close, sym_high, sym_low, sym_sigs, sym_atr,
            initial_capital=CAPITAL, max_slots=3,
            cost_per_side=COST, funding_per_bar=FUNDING,
            partial_exit=partial, partial_fraction=fraction,
            first_target_retrace=first_tgt,
            trail_atr_mult=trail_atr, breakeven_after_tp1=be,
        )
        elapsed = time.time() - t0

        n_days = len(result['equity_curve']) / 96
        trades_per_day = round(result['n_trades'] / n_days, 2)
        years = yearly_breakdown(result)

        summary = {
            'description': desc,
            'partial_exit': partial,
            'trail_atr_mult': trail_atr,
            'breakeven': be,
            'first_target': first_tgt,
            'total_return_pct': result['total_return_pct'],
            'final_balance': result['final_balance'],
            'max_dd_pct': result['max_dd_pct'],
            'sharpe': result['sharpe'],
            'calmar': result['calmar'],
            'n_trades': result['n_trades'],
            'trades_per_day': trades_per_day,
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'avg_pnl_pct': result['avg_pnl_pct'],
            'avg_win_pct': result['avg_win_pct'],
            'avg_loss_pct': result['avg_loss_pct'],
            'actual_rr': result['actual_rr'],
            'exit_types': result['exit_types'],
            'yearly': years,
            'elapsed': round(elapsed, 2),
        }
        results[name] = summary

        print(f"  Return: {result['total_return_pct']:+,.1f}%  Sharpe: {result['sharpe']:.2f}  "
              f"DD: {result['max_dd_pct']:.1f}%  Calmar: {result['calmar']:.1f}  "
              f"Trades: {result['n_trades']} ({trades_per_day}/day)  "
              f"WR: {result['win_rate']:.1f}%  PF: {result['profit_factor']:.2f}  "
              f"R:R: {result['actual_rr']:.2f}  [{elapsed:.1f}s]")

        # Exit type breakdown
        for et, info in result['exit_types'].items():
            print(f"    {et:>12}: {info['count']:,} trades, "
                  f"avg {info['avg_pnl_pct']:+.3f}%, "
                  f"WR {info['win_rate']:.1f}%")

    # ══════════════════════════════════════════════════════════════
    #   RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    names = list(results.keys())
    base = results['A-Base']

    print(f"\n\n{'=' * 100}")
    print("  PARTIAL EXIT + TRAILING A/B TEST — RESULTS")
    print(f"{'=' * 100}\n")

    col_w = 12
    print(f"  {'Metric':<20}", end='')
    for n in names:
        print(f"  {n:>{col_w}}", end='')
    print()
    print("  " + "-" * (20 + (col_w + 2) * len(names)))

    rows = [
        ('Partial Exit', 'partial_exit', '{}'),
        ('Trail ATR ×', 'trail_atr_mult', '{:.1f}'),
        ('Break-even', 'breakeven', '{}'),
        ('1st Target', 'first_target', '{:.3f}'),
        ('', None, None),
        ('Total Return', 'total_return_pct', '{:+,.1f}%'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Calmar', 'calmar', '{:.1f}'),
        ('Max Drawdown', 'max_dd_pct', '{:.1f}%'),
        ('Trades', 'n_trades', '{:,d}'),
        ('Trades/Day', 'trades_per_day', '{:.2f}'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Avg PnL/Trade', 'avg_pnl_pct', '{:+.3f}%'),
        ('', None, None),
        ('Avg Win %', 'avg_win_pct', '{:+.3f}%'),
        ('Avg Loss %', 'avg_loss_pct', '{:.3f}%'),
        ('Actual R:R', 'actual_rr', '{:.2f}'),
    ]

    for label, key, fmt in rows:
        if key is None:
            print()
            continue
        print(f"  {label:<20}", end='')
        for n in names:
            val = results[n][key]
            if isinstance(val, bool):
                s = 'Yes' if val else 'No'
            else:
                s = fmt.format(val)
            print(f"  {s:>{col_w}}", end='')
        print()

    # Delta
    print(f"\n{'─' * 100}")
    print("  DELTA vs A-Base")
    print(f"{'─' * 100}\n")
    delta_keys = [
        ('ΔReturn', 'total_return_pct'),
        ('ΔSharpe', 'sharpe'),
        ('ΔCalmar', 'calmar'),
        ('ΔMax DD', 'max_dd_pct'),
        ('ΔR:R', 'actual_rr'),
        ('ΔAvg PnL', 'avg_pnl_pct'),
    ]
    print(f"  {'Metric':<20}", end='')
    for n in names:
        print(f"  {n:>{col_w}}", end='')
    print()
    print("  " + "-" * (20 + (col_w + 2) * len(names)))
    for label, key in delta_keys:
        print(f"  {label:<20}", end='')
        for n in names:
            v = results[n][key]
            bv = base[key]
            if bv != 0:
                delta_pct = (v / bv - 1) * 100
                print(f"  {delta_pct:>+10.1f}%", end='  ')
            else:
                print(f"  {'—':>{col_w}}", end='')
        print()

    # Yearly
    print(f"\n{'─' * 100}")
    print("  YEARLY BREAKDOWN")
    print(f"{'─' * 100}")
    for y in range(3):
        yr = results[names[0]]['yearly'][y]
        print(f"\n  Year {y+1} ({yr['label']}):")
        print(f"  {'Metric':<18}", end='')
        for n in names:
            print(f"  {n:>{col_w}}", end='')
        print()
        for label, key, fmt in [
            ('Return', 'return_pct', '{:+,.1f}%'),
            ('Trades', 'trades', '{:,d}'),
            ('Win Rate', 'win_rate', '{:.1f}%'),
            ('Max DD', 'max_dd_pct', '{:.1f}%'),
            ('Sharpe', 'sharpe', '{:.2f}'),
        ]:
            print(f"    {label:<16}", end='')
            for n in names:
                val = results[n]['yearly'][y][key]
                print(f"  {fmt.format(val):>{col_w}}", end='')
            print()

    # ── Binance Server-Side Trailing ──
    print(f"\n{'=' * 100}")
    print("  BINANCE SERVER-SIDE TRAILING STOP")
    print(f"{'=' * 100}\n")
    print("  Binance USDT-M Futures supports TRAILING_STOP_MARKET order type:")
    print("    POST /fapi/v1/order  type=TRAILING_STOP_MARKET")
    print("    Required params:")
    print("      callbackRate: 0.1% ~ 5.0% (trailing distance as %)")
    print("      activationPrice: optional (price at which trailing starts)")
    print("      quantity, side, symbol, reduceOnly=true")
    print()
    print("  Implementation plan for live deployment:")
    print("    1. On TP1 hit: market-close 50% (existing _partial_close_position)")
    print("    2. Cancel old server STOP order")
    print("    3. Place TRAILING_STOP_MARKET for remaining qty:")
    print("       callbackRate = trail_atr_mult × ATR% × 100")
    print("       activationPrice = entry_price (breakeven activation)")
    print("       reduceOnly = true")
    print("    4. Server handles trailing — no client dependency")
    print("    5. Fallback: 15m polling detects fill → cleans state")

    # ── Recommendation ──
    print(f"\n{'=' * 100}")
    print("  RECOMMENDATION")
    print(f"{'=' * 100}\n")

    best_name = None
    best_score = -float('inf')
    for n, r in results.items():
        score = (
            (r['sharpe'] / max(base['sharpe'], 0.01)) * 0.25 +
            (r['calmar'] / max(base['calmar'], 0.01)) * 0.20 +
            (r['actual_rr'] / max(base['actual_rr'], 0.01)) * 0.20 +
            (r['total_return_pct'] / max(base['total_return_pct'], 0.01)) * 0.20 +
            (base['max_dd_pct'] / max(r['max_dd_pct'], 0.01)) * 0.15
        )
        if score > best_score:
            best_score = score
            best_name = n

    for n, r in results.items():
        score = (
            (r['sharpe'] / max(base['sharpe'], 0.01)) * 0.25 +
            (r['calmar'] / max(base['calmar'], 0.01)) * 0.20 +
            (r['actual_rr'] / max(base['actual_rr'], 0.01)) * 0.20 +
            (r['total_return_pct'] / max(base['total_return_pct'], 0.01)) * 0.20 +
            (base['max_dd_pct'] / max(r['max_dd_pct'], 0.01)) * 0.15
        )
        tag = ' ★ WINNER' if n == best_name else ''
        print(f"  {n:>12}: score={score:.3f}  Sharpe={r['sharpe']:.2f}  "
              f"Calmar={r['calmar']:.1f}  R:R={r['actual_rr']:.2f}  "
              f"DD={r['max_dd_pct']:.1f}%{tag}")

    print(f"\n  → {best_name} achieves the best balance.")

    # Save
    output_path = Path(__file__).parent.parent / 'logs' / 'partial_exit_trailing_ab_results.json'
    save_results = {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
