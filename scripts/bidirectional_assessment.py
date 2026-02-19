#!/usr/bin/env python3
"""
Bidirectional Strategy Assessment — 3-Year Portfolio Backtest
==============================================================
Tests whether allowing ALL signal directions (long + short) with MTF as a
leverage selector (not a blocker) improves trading frequency and returns.

Current problem:
  - MTF blocks 66.7% of raw signals (73.7% of longs, 37.4% of shorts)
  - Raw signal pool: 26,554 longs + 6,370 shorts = 32,924 total
  - After MTF blocking: 6,973 longs + 3,986 shorts = 10,959 total
  - In bear regimes, long blocking rate approaches 96%

Hypothesis: "Flipping" the strategy = allowing both directions
  - Counter-trend trades at base leverage (1.5×)
  - Trend-aligned trades at boosted leverage (3.75×)
  - Doubles (or triples) the entry opportunity set

Variants:
  B:  Conf≥3, MTF blocks counter-trend (current recommendation)
  D:  Conf≥3, MTF boost-only (counter-trend at 1.5×, trend-aligned at 3.75×)
  E:  Conf≥3, No MTF (all trades at flat 1.5×)
  F:  Conf≥3, MTF boost-only + 3 slots (more concurrent positions)
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
import numba
from datetime import datetime, timezone
from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_v9,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    _compute_sma_inline,
)

# ═══════════════════════════════════════════════════════════════════════
#  Git Commit Info
# ═══════════════════════════════════════════════════════════════════════

def _get_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'


def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    matches = sorted(CACHE_DIR.glob(f'{symbol}_{interval}_{days}d_*.npz'), reverse=True)
    return matches[0] if matches else None


def load_15m(symbol: str, days: int) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is None:
        return None
    d = np.load(cache)
    return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}


def load_4h(symbol: str, days: int) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is None:
        return None
    d = np.load(cache)
    return d['close']


# ═══════════════════════════════════════════════════════════════════════
#  MTF Boost-Only (new: don't block, just modulate leverage)
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def apply_mtf_boost_only(
    signals_15m: np.ndarray,
    close_4h: np.ndarray,
    sma_period_4h: int,
    leverage_mult: float,
    ratio: int = 16,
) -> np.ndarray:
    """MTF as leverage selector, NOT a blocker.

    - Trend-aligned signals: multiply leverage by leverage_mult
    - Counter-trend signals: keep at base leverage (unchanged)
    - No signals are zeroed out

    This preserves ALL entries while still rewarding trend-alignment.
    """
    sma_4h = _compute_sma_inline(close_4h, sma_period_4h)
    n = len(signals_15m)
    out = signals_15m.copy()

    for i in range(n):
        if out[i] == 0.0:
            continue
        j = min(i // ratio, len(close_4h) - 1)
        if j < sma_period_4h or np.isnan(sma_4h[j]):
            continue

        trend_up = close_4h[j] > sma_4h[j]

        # Boost trend-aligned signals
        if out[i] > 0 and trend_up:
            out[i] *= leverage_mult
        elif out[i] < 0 and not trend_up:
            out[i] *= leverage_mult
        # Counter-trend signals: left at base leverage (not zeroed)

    return out


# ═══════════════════════════════════════════════════════════════════════
#  Signal Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(
    data: dict, btc: np.ndarray, close_4h: np.ndarray | None,
    cfg: RubberBandConfig,
    atr_period: int, atr_mult: float, floor: float, cap: float,
    mtf_mode: str, mtf_sma: int, mtf_lev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate signals with configurable MTF mode.

    mtf_mode:
      'block'      = standard MTF (block counter-trend)
      'boost_only' = MTF as leverage selector (no blocking)
      'off'        = no MTF at all
    """
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
        if mtf_mode == 'block':
            sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
            sigs = apply_mtf_leverage(sigs, mtf_lev)
        elif mtf_mode == 'boost_only':
            sigs = apply_mtf_boost_only(sigs, close_4h, mtf_sma, mtf_lev, ratio=16)
        # else 'off': no MTF applied

    return c, sigs


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Simulator
# ═══════════════════════════════════════════════════════════════════════

def simulate_portfolio(
    all_close: dict[str, np.ndarray],
    all_signals: dict[str, np.ndarray],
    initial_capital: float = 2095.0,
    max_slots: int = 2,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    reinvest: bool = False,
) -> dict:
    symbols = sorted(all_close.keys())
    n_bars = min(len(all_close[s]) for s in symbols)

    balance = initial_capital
    equity_curve = np.zeros(n_bars)
    equity_curve[0] = balance

    positions: dict[str, dict] = {}
    trades: list[dict] = []
    peak = balance
    max_dd = 0.0

    for i in range(1, n_bars):
        symbols_to_close = []

        for sym, pos in list(positions.items()):
            close_arr = all_close[sym]
            sig_arr = all_signals[sym]

            current_price = close_arr[i]
            prev_price = close_arr[i - 1]
            bar_ret = current_price / prev_price - 1.0

            direction = 1.0 if pos['signal'] > 0 else -1.0
            leverage = abs(pos['signal'])
            pnl = direction * leverage * bar_ret * pos['notional']
            pnl -= abs(leverage) * funding_per_bar * pos['notional']

            pos['unrealized'] = pos.get('unrealized', 0.0) + pnl
            pos['bars_held'] += 1

            current_sig = sig_arr[i] if i < len(sig_arr) else 0.0
            prev_sig = sig_arr[i - 1] if i - 1 < len(sig_arr) else 0.0

            if current_sig == 0.0 and prev_sig != 0.0:
                symbols_to_close.append(sym)
            elif current_sig != 0.0 and np.sign(current_sig) != np.sign(pos['signal']):
                symbols_to_close.append(sym)

        for sym in symbols_to_close:
            pos = positions.pop(sym)
            leverage = abs(pos['signal'])
            close_cost = leverage * cost_per_side * pos['notional']
            net_pnl = pos['unrealized'] - close_cost
            balance += net_pnl

            pnl_pct = net_pnl / pos['notional'] * 100
            trades.append({
                'symbol': sym,
                'signal': pos['signal'],
                'entry_bar': pos['entry_bar'],
                'exit_bar': i,
                'bars_held': pos['bars_held'],
                'notional': pos['notional'],
                'pnl_usd': net_pnl,
                'pnl_pct': pnl_pct,
            })

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
                sizing_capital = balance if reinvest else initial_capital
                per_slot = sizing_capital / max(1, len(positions) + 1)
                per_slot = min(per_slot, sizing_capital * 0.5)
                if per_slot < 10.0:
                    continue

                leverage = abs(sig)
                notional = per_slot
                entry_cost = leverage * cost_per_side * notional
                balance -= entry_cost

                positions[sym] = {
                    'signal': sig,
                    'entry_price': all_close[sym][i],
                    'entry_bar': i,
                    'notional': notional,
                    'bars_held': 0,
                    'unrealized': 0.0,
                }

        total_unrealized = sum(p.get('unrealized', 0.0) for p in positions.values())
        equity_curve[i] = balance + total_unrealized

        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

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

    avg_bars = np.mean([t['bars_held'] for t in trades]) if trades else 0
    avg_pnl = np.mean([t['pnl_usd'] for t in trades]) if trades else 0
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

    # Long vs short breakdown
    long_trades = [t for t in trades if t['signal'] > 0]
    short_trades = [t for t in trades if t['signal'] < 0]
    n_long = len(long_trades)
    n_short = len(short_trades)
    long_wr = sum(1 for t in long_trades if t['pnl_usd'] > 0) / n_long * 100 if n_long > 0 else 0
    short_wr = sum(1 for t in short_trades if t['pnl_usd'] > 0) / n_short * 100 if n_short > 0 else 0
    long_pf = sum(t['pnl_usd'] for t in long_trades if t['pnl_usd'] > 0) / max(1, sum(abs(t['pnl_usd']) for t in long_trades if t['pnl_usd'] <= 0))
    short_pf = sum(t['pnl_usd'] for t in short_trades if t['pnl_usd'] > 0) / max(1, sum(abs(t['pnl_usd']) for t in short_trades if t['pnl_usd'] <= 0))
    long_avg_pnl = np.mean([t['pnl_pct'] for t in long_trades]) if long_trades else 0
    short_avg_pnl = np.mean([t['pnl_pct'] for t in short_trades]) if short_trades else 0

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
        'gross_profit': round(gross_profit, 0),
        'gross_loss': round(gross_loss, 0),
        'avg_bars_held': round(avg_bars, 1),
        'avg_pnl_usd': round(avg_pnl, 2),
        'avg_pnl_pct': round(avg_pnl_pct, 3),
        'max_slots': max_slots,
        'n_long': n_long,
        'n_short': n_short,
        'long_win_rate': round(long_wr, 1),
        'short_win_rate': round(short_wr, 1),
        'long_pf': round(long_pf, 2),
        'short_pf': round(short_pf, 2),
        'long_avg_pnl_pct': round(long_avg_pnl, 3),
        'short_avg_pnl_pct': round(short_avg_pnl, 3),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Yearly Breakdown
# ═══════════════════════════════════════════════════════════════════════

def yearly_breakdown(result: dict, days: int = 1095) -> list[dict]:
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
        n_wins = len(wins)
        wr = n_wins / n_t * 100 if n_t > 0 else 0.0

        gp = sum(t['pnl_usd'] for t in wins)
        gl = sum(abs(t['pnl_usd']) for t in year_trades if t['pnl_usd'] <= 0)
        pf = gp / gl if gl > 0 else 999.0

        total_pnl = sum(t['pnl_usd'] for t in year_trades)
        ret_pct = total_pnl / capital * 100

        # Long/Short in this year
        yr_long = [t for t in year_trades if t['signal'] > 0]
        yr_short = [t for t in year_trades if t['signal'] < 0]

        eq = result['equity_curve']
        eq_slice = eq[start_bar:min(end_bar, len(eq))]
        if len(eq_slice) > 0:
            peak = np.maximum.accumulate(eq_slice)
            dd = (peak - eq_slice) / np.maximum(peak, 1e-10)
            year_max_dd = float(np.max(dd)) * 100
        else:
            year_max_dd = 0.0

        if len(eq_slice) > 1:
            eq_rets = np.diff(eq_slice) / np.maximum(eq_slice[:-1], 1e-10)
            eq_rets = eq_rets[~np.isnan(eq_rets)]
            yr_sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)) if np.std(eq_rets) > 0 else 0.0
        else:
            yr_sharpe = 0.0

        years.append({
            'year': y + 1,
            'label': ['2023-02→2024-02 (Recovery/Bull)',
                       '2024-02→2025-02 (Bull/Mixed)',
                       '2025-02→2026-02 (Bear/Current)'][y],
            'return_pct': round(ret_pct, 1),
            'trades': n_t,
            'n_long': len(yr_long),
            'n_short': len(yr_short),
            'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2),
            'max_dd_pct': round(year_max_dd, 1),
            'sharpe': round(yr_sharpe, 2),
        })

    return years


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    commit = _get_git_commit()
    print(f"# Bidirectional Strategy Assessment — 3-Year Portfolio Backtest")
    print(f"# Commit: {commit}")
    print(f"# Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    settings = get_settings()

    SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
    ]

    DAYS = 1095
    CAPITAL = 2095.0
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # ── Load base config ──
    base_cfg = RubberBandConfig()
    for k in ['lookback', 'threshold', 'target_retrace', 'stop_mult', 'max_hold',
              'stop_decay', 'btc_directional_period', 'btc_regime_period',
              'asset_trend_period', 'confidence_leverage_step', 'max_leverage',
              'rsi_period', 'vol_period', 'stoch_enabled', 'stoch_k_period',
              'stoch_d_period', 'macd_enabled', 'macd_fast', 'macd_slow',
              'macd_signal_period', 'dynamic_target_enabled', 'target_conf3',
              'target_conf4', 'target_conf5']:
        setattr(base_cfg, k, getattr(settings, 'rb_' + k))

    # ── Define variants ──
    variants = {
        'B': {
            'name': 'Option B: Conf≥3, MTF Blocks (Baseline)',
            'desc': 'MTF blocks counter-trend, trend-aligned at 3.75×',
            'min_confidence': 3,
            'base_leverage': 1.5,
            'mtf_mode': 'block',
            'mtf_sma': 12,
            'mtf_lev': 2.5,
            'slots': 2,
        },
        'D': {
            'name': 'Option D: Conf≥3, MTF Boost-Only (Bidirectional)',
            'desc': 'All signals pass; trend-aligned 3.75×, counter-trend 1.5×',
            'min_confidence': 3,
            'base_leverage': 1.5,
            'mtf_mode': 'boost_only',
            'mtf_sma': 12,
            'mtf_lev': 2.5,
            'slots': 2,
        },
        'E': {
            'name': 'Option E: Conf≥3, No MTF (Flat 1.5×)',
            'desc': 'All signals at flat 1.5× leverage, no MTF at all',
            'min_confidence': 3,
            'base_leverage': 1.5,
            'mtf_mode': 'off',
            'mtf_sma': 12,
            'mtf_lev': 1.0,
            'slots': 2,
        },
        'F': {
            'name': 'Option F: Conf≥3, MTF Boost-Only, 3 Slots',
            'desc': 'Bidirectional + 3 concurrent positions',
            'min_confidence': 3,
            'base_leverage': 1.5,
            'mtf_mode': 'boost_only',
            'mtf_sma': 12,
            'mtf_lev': 2.5,
            'slots': 3,
        },
    }

    # ── Load data ──
    print(f"Loading {DAYS}-day data for {len(SYMBOLS)} symbols from cache...")
    btc_data = load_15m('BTCUSDT', DAYS)
    if btc_data is None:
        print("ERROR: BTC 15m data not found. Run download_extended_data.py first.")
        sys.exit(1)
    btc_close = btc_data['close']
    print(f"  BTC: {len(btc_close)} bars (15m)")

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS:
        d = load_15m(sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        h4 = load_4h(sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4
    print(f"  Loaded {len(all_data)} symbols\n")

    # ── Run each variant ──
    results = {}

    for key, var in variants.items():
        print(f"{'=' * 80}")
        print(f"  {var['name']}")
        print(f"  {var['desc']}")
        print(f"{'=' * 80}")

        cfg = RubberBandConfig()
        for attr in dir(base_cfg):
            if not attr.startswith('_'):
                try:
                    setattr(cfg, attr, getattr(base_cfg, attr))
                except (AttributeError, TypeError):
                    pass
        cfg.min_confidence = var['min_confidence']
        cfg.base_leverage = var['base_leverage']

        t0 = time.time()
        all_close = {}
        all_signals = {}
        for sym, data in all_data.items():
            close_4h = all_4h.get(sym)
            c, sigs = generate_signals(
                data, btc_close, close_4h, cfg,
                atr_period=48, atr_mult=2.5, floor=0.01, cap=0.06,
                mtf_mode=var['mtf_mode'], mtf_sma=var['mtf_sma'], mtf_lev=var['mtf_lev'],
            )
            all_close[sym] = c
            all_signals[sym] = sigs

        # Count entries & direction
        total_entries = 0
        long_entries = 0
        short_entries = 0
        for s in all_signals.values():
            for i in range(1, len(s)):
                if s[i] != 0 and s[i - 1] == 0:
                    total_entries += 1
                    if s[i] > 0:
                        long_entries += 1
                    else:
                        short_entries += 1

        print(f"  Signal entries: {total_entries} (L:{long_entries} S:{short_entries})  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        result = simulate_portfolio(
            all_close, all_signals,
            initial_capital=CAPITAL, max_slots=var['slots'],
            cost_per_side=COST, funding_per_bar=FUNDING,
            reinvest=False,
        )
        sim_time = time.time() - t0

        years = yearly_breakdown(result, DAYS)

        print(f"\n  Total Return:   {result['total_return_pct']:+,.1f}%")
        print(f"  Final Balance:  ${result['final_balance']:,.0f}")
        print(f"  Max Drawdown:   {result['max_dd_pct']:.1f}%")
        print(f"  Sharpe Ratio:   {result['sharpe']:.2f}")
        print(f"  Calmar Ratio:   {result['calmar']:.1f}")
        print(f"  Trades:         {result['n_trades']} (L:{result['n_long']} S:{result['n_short']})")
        print(f"  Win Rate:       {result['win_rate']:.1f}% (L:{result['long_win_rate']:.1f}% S:{result['short_win_rate']:.1f}%)")
        print(f"  Profit Factor:  {result['profit_factor']:.2f} (L:{result['long_pf']:.2f} S:{result['short_pf']:.2f})")
        print(f"  Avg PnL/Trade:  ${result['avg_pnl_usd']:.2f} ({result['avg_pnl_pct']:+.3f}%)")
        print(f"    Long avg:     {result['long_avg_pnl_pct']:+.3f}%")
        print(f"    Short avg:    {result['short_avg_pnl_pct']:+.3f}%")
        print(f"  Avg Bars Held:  {result['avg_bars_held']:.1f}")
        print(f"  Trades/day:     {result['n_trades'] / DAYS:.1f}")
        print(f"  Sim time:       {sim_time:.1f}s")

        print(f"\n  Yearly Breakdown:")
        for yr in years:
            print(f"    Y{yr['year']} ({yr['label']}):")
            print(f"      Return: {yr['return_pct']:+,.1f}%, Trades: {yr['trades']} (L:{yr['n_long']} S:{yr['n_short']}), "
                  f"Win: {yr['win_rate']:.1f}%, PF: {yr['profit_factor']:.2f}, "
                  f"DD: {yr['max_dd_pct']:.1f}%, Sharpe: {yr['sharpe']:.2f}")
        print()

        results[key] = {
            'name': var['name'],
            'desc': var['desc'],
            'config': {k: v for k, v in var.items() if k not in ('name', 'desc')},
            'summary': {k: v for k, v in result.items() if k != 'equity_curve' and k != 'trades'},
            'yearly': years,
            'signal_entries': total_entries,
            'signal_long': long_entries,
            'signal_short': short_entries,
        }

    # ── Comparison Table ──
    keys = ['B', 'D', 'E', 'F']
    labels = ['B (MTF Block)', 'D (MTF Boost)', 'E (No MTF)', 'F (Boost+3slot)']

    print(f"\n{'=' * 110}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 110}")
    header = f"{'Metric':<22}" + "".join(f"{l:>22}" for l in labels)
    print(header)
    print(f"{'─' * 110}")

    metrics = [
        ('total_return_pct', '{:+,.1f}%'),
        ('final_balance', '${:,.0f}'),
        ('max_dd_pct', '{:.1f}%'),
        ('sharpe', '{:.2f}'),
        ('calmar', '{:.1f}'),
        ('n_trades', '{:,}'),
        ('n_long', '{:,}'),
        ('n_short', '{:,}'),
        ('win_rate', '{:.1f}%'),
        ('long_win_rate', '{:.1f}%'),
        ('short_win_rate', '{:.1f}%'),
        ('profit_factor', '{:.2f}'),
        ('long_pf', '{:.2f}'),
        ('short_pf', '{:.2f}'),
        ('avg_pnl_pct', '{:+.3f}%'),
        ('long_avg_pnl_pct', '{:+.3f}%'),
        ('short_avg_pnl_pct', '{:+.3f}%'),
        ('avg_bars_held', '{:.1f}'),
    ]

    for metric, fmt in metrics:
        vals = []
        for k in keys:
            v = results[k]['summary'][metric]
            vals.append(fmt.format(v))
        label = metric.replace('_', ' ').title()
        print(f"  {label:<20}" + "".join(f"{v:>22}" for v in vals))

    # Trades per day
    print(f"  {'Trades/Day':<20}" + "".join(
        f"{results[k]['summary']['n_trades']/DAYS:>22.1f}" for k in keys))

    # Y3 comparison
    print(f"\n{'─' * 110}")
    print(f"  Y3 BEARISH REGIME (2025-02 → 2026-02)")
    print(f"{'─' * 110}")
    for metric, fmt in [
        ('return_pct', '{:+,.1f}%'),
        ('trades', '{:,}'),
        ('n_long', '{:,}'),
        ('n_short', '{:,}'),
        ('win_rate', '{:.1f}%'),
        ('profit_factor', '{:.2f}'),
        ('max_dd_pct', '{:.1f}%'),
        ('sharpe', '{:.2f}'),
    ]:
        vals = []
        for k in keys:
            y3 = results[k]['yearly'][2]
            vals.append(fmt.format(y3[metric]))
        label = metric.replace('_', ' ').title()
        print(f"  {label:<20}" + "".join(f"{v:>22}" for v in vals))

    print(f"\n{'=' * 110}")

    # Save JSON
    output_path = Path(__file__).parent.parent / 'logs' / 'bidirectional_assessment_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
