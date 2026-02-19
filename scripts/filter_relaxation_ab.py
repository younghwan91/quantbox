#!/usr/bin/env python3
"""
Filter Relaxation A/B Test — More Opportunities Without Sacrificing Performance
================================================================================
Goal: The B-50 universe + strict filters (conf≥3, MTF block, ATR 2.5×) produce
high Sharpe but low trade frequency. This test systematically relaxes filters
to find the Pareto-optimal balance between opportunity count and quality.

Variants (all use 50 symbols, 3-year data):
  A: Baseline (current B-50: conf≥3, ATR 2.5×, floor 1%, cap 6%, MTF block, 2 slots)
  B: conf≥2 (lower confidence gate)
  C: Wider ATR (2.0× mult, 0.8% floor → lower adaptive threshold)
  D: Combo (conf≥2 + ATR 2.0× + floor 0.8%)
  E: D + 3 slots (more capital deployed simultaneously)
  F: D + MTF soft (MTF leverage boost only, no counter-trend block)

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/filter_relaxation_ab.py
"""

import sys
import time
import json
import subprocess
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
#  Signal Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(
    data: dict, btc: np.ndarray, close_4h: np.ndarray | None,
    cfg: RubberBandConfig,
    atr_period: int, atr_mult: float, floor: float, cap: float,
    mtf_enabled: bool, mtf_block: bool, mtf_sma: int, mtf_lev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate signals with configurable MTF block vs leverage-only."""
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

    if mtf_enabled and close_4h is not None:
        if mtf_block:
            # Full MTF: block counter-trend + boost aligned
            sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
        # Always apply leverage boost for MTF-aligned trades
        sigs = apply_mtf_leverage(sigs, mtf_lev)

    return c, sigs


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Simulator (shared)
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

    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

    long_trades = [t for t in trades if t['signal'] > 0]
    short_trades = [t for t in trades if t['signal'] < 0]
    long_wins = [t for t in long_trades if t['pnl_usd'] > 0]
    short_wins = [t for t in short_trades if t['pnl_usd'] > 0]

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
        'avg_pnl_pct': round(avg_pnl_pct, 3),
        'max_slots': max_slots,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_wr': round(len(long_wins) / len(long_trades) * 100, 1) if long_trades else 0,
        'short_wr': round(len(short_wins) / len(short_trades) * 100, 1) if short_trades else 0,
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
        wr = len(wins) / n_t * 100 if n_t > 0 else 0.0

        gp = sum(t['pnl_usd'] for t in wins)
        gl = sum(abs(t['pnl_usd']) for t in year_trades if t['pnl_usd'] <= 0)
        pf = gp / gl if gl > 0 else 999.0

        total_pnl = sum(t['pnl_usd'] for t in year_trades)
        ret_pct = total_pnl / capital * 100

        eq = result['equity_curve']
        eq_slice = eq[start_bar:min(end_bar, len(eq))]
        if len(eq_slice) > 1:
            peak = np.maximum.accumulate(eq_slice)
            dd = (peak - eq_slice) / np.maximum(peak, 1e-10)
            year_max_dd = float(np.max(dd)) * 100
            eq_rets = np.diff(eq_slice) / np.maximum(eq_slice[:-1], 1e-10)
            eq_rets = eq_rets[~np.isnan(eq_rets)]
            yr_sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)) if np.std(eq_rets) > 0 else 0.0
        else:
            year_max_dd = 0.0
            yr_sharpe = 0.0

        years.append({
            'year': y + 1,
            'label': ['2023-02→2024-02', '2024-02→2025-02', '2025-02→2026-02'][y],
            'return_pct': round(ret_pct, 1),
            'trades': n_t,
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
    settings = get_settings()

    print("=" * 80)
    print("  FILTER RELAXATION A/B TEST — More Opportunities vs Performance")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  Baseline: B-50 (conf≥3, MTF block, ATR 2.5×, 2 slots)")
    print("=" * 80)

    # ── Symbol list (B-50) ──
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

    # ── Base config (shared across all variants) ──
    def make_cfg(min_conf: int = 3) -> RubberBandConfig:
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
        cfg.min_confidence = min_conf
        return cfg

    # ── Variant definitions ──
    # Each variant: (name, description, cfg, atr_period, atr_mult, floor, cap,
    #                mtf_enabled, mtf_block, mtf_sma, mtf_lev, slots)
    variants = [
        ('A-Base',    'conf≥3, ATR 2.5×, MTF block, 2 slots',
         make_cfg(3), 48, 2.5, 0.01, 0.06, True, True, 12, 2.5, 2),
        ('B-Conf2',   'conf≥2, ATR 2.5×, MTF block, 2 slots',
         make_cfg(2), 48, 2.5, 0.01, 0.06, True, True, 12, 2.5, 2),
        ('C-WideATR', 'conf≥3, ATR 2.0×, floor 0.8%, MTF block, 2 slots',
         make_cfg(3), 48, 2.0, 0.008, 0.06, True, True, 12, 2.5, 2),
        ('D-Combo',   'conf≥2, ATR 2.0×, floor 0.8%, MTF block, 2 slots',
         make_cfg(2), 48, 2.0, 0.008, 0.06, True, True, 12, 2.5, 2),
        ('E-3Slot',   'conf≥2, ATR 2.0×, floor 0.8%, MTF block, 3 slots',
         make_cfg(2), 48, 2.0, 0.008, 0.06, True, True, 12, 2.5, 3),
        ('F-NoBlk',   'conf≥2, ATR 2.0×, floor 0.8%, MTF lev only (no block), 2 slots',
         make_cfg(2), 48, 2.0, 0.008, 0.06, True, False, 12, 2.5, 2),
    ]

    # ── Load data (all cached from previous sweep) ──
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
            print(f"  SKIP {sym} (no 15m data)")
            continue
        h4 = load_4h(sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # ── Run all variants ──
    results: dict[str, dict] = {}

    for name, desc, cfg, atr_p, atr_m, flr, cap, mtf_en, mtf_blk, mtf_sma, mtf_lev, slots in variants:
        print(f"\n{'─' * 60}")
        print(f"  ⚡ {name}: {desc}")
        print(f"{'─' * 60}")

        # Generate signals for this variant
        sym_close: dict[str, np.ndarray] = {}
        sym_sigs: dict[str, np.ndarray] = {}
        total_entries = 0

        for sym in available:
            data = all_data[sym]
            close_4h = all_4h.get(sym)
            c, sigs = generate_signals(
                data, btc_close, close_4h, cfg,
                atr_p, atr_m, flr, cap,
                mtf_en, mtf_blk, mtf_sma, mtf_lev,
            )
            sym_close[sym] = c
            sym_sigs[sym] = sigs
            n_entries = int(np.sum((sigs[1:] != 0) & (sigs[:-1] == 0)))
            total_entries += n_entries

        # Left-pad
        max_bars = max(len(v) for v in sym_close.values())
        for sym in list(sym_close.keys()):
            c = sym_close[sym]
            s = sym_sigs[sym]
            if len(c) < max_bars:
                pad_n = max_bars - len(c)
                sym_close[sym] = np.concatenate([np.full(pad_n, c[0]), c])
                sym_sigs[sym] = np.concatenate([np.zeros(pad_n), s])

        print(f"  Raw signal entries: {total_entries:,}")

        # Simulate
        t0 = time.time()
        result = simulate_portfolio(
            sym_close, sym_sigs,
            initial_capital=CAPITAL, max_slots=slots,
            cost_per_side=COST, funding_per_bar=FUNDING,
            reinvest=False,
        )
        elapsed = time.time() - t0

        n_days = len(result['equity_curve']) / 96
        trades_per_day = round(result['n_trades'] / n_days, 2)
        years = yearly_breakdown(result)

        summary = {
            'description': desc,
            'symbols': len(available),
            'slots': slots,
            'min_confidence': cfg.min_confidence,
            'atr_mult': atr_m,
            'threshold_floor': flr,
            'mtf_block': mtf_blk,
            'total_return_pct': result['total_return_pct'],
            'final_balance': result['final_balance'],
            'max_dd_pct': result['max_dd_pct'],
            'sharpe': result['sharpe'],
            'calmar': result['calmar'],
            'n_trades': result['n_trades'],
            'trades_per_day': trades_per_day,
            'raw_signals': total_entries,
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'avg_pnl_pct': result['avg_pnl_pct'],
            'long_trades': result['long_trades'],
            'short_trades': result['short_trades'],
            'long_wr': result['long_wr'],
            'short_wr': result['short_wr'],
            'yearly': years,
            'elapsed': round(elapsed, 2),
        }
        results[name] = summary

        print(f"  Return: {result['total_return_pct']:+,.1f}%  Sharpe: {result['sharpe']:.2f}  "
              f"DD: {result['max_dd_pct']:.1f}%  Calmar: {result['calmar']:.1f}  "
              f"Trades: {result['n_trades']} ({trades_per_day}/day)  "
              f"WR: {result['win_rate']:.1f}%  PF: {result['profit_factor']:.2f}  [{elapsed:.1f}s]")

    # ══════════════════════════════════════════════════════════════
    #   RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    names = list(results.keys())
    base = results['A-Base']

    print(f"\n\n{'=' * 100}")
    print("  FILTER RELAXATION A/B TEST — RESULTS")
    print(f"{'=' * 100}\n")

    col_w = 12
    print(f"  {'Metric':<20}", end='')
    for n in names:
        print(f"  {n:>{col_w}}", end='')
    print()
    print("  " + "-" * (20 + (col_w + 2) * len(names)))

    rows = [
        ('Slots', 'slots', '{:d}'),
        ('Min Confidence', 'min_confidence', '{:d}'),
        ('ATR Mult', 'atr_mult', '{:.1f}×'),
        ('Thres Floor', 'threshold_floor', '{:.1%}'),
        ('MTF Block', 'mtf_block', '{}'),
        ('', None, None),  # separator
        ('Raw Signals', 'raw_signals', '{:,d}'),
        ('Total Trades', 'n_trades', '{:,d}'),
        ('Trades/Day', 'trades_per_day', '{:.2f}'),
        ('', None, None),
        ('Total Return', 'total_return_pct', '{:+,.1f}%'),
        ('Final Balance', 'final_balance', '${:,.0f}'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Calmar', 'calmar', '{:.1f}'),
        ('Max Drawdown', 'max_dd_pct', '{:.1f}%'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Avg PnL/Trade', 'avg_pnl_pct', '{:+.3f}%'),
        ('', None, None),
        ('Longs', 'long_trades', '{:,d}'),
        ('Shorts', 'short_trades', '{:,d}'),
        ('Long WR', 'long_wr', '{:.1f}%'),
        ('Short WR', 'short_wr', '{:.1f}%'),
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

    # ── Delta vs Baseline ──
    print(f"\n{'─' * 100}")
    print("  DELTA vs A-Base")
    print(f"{'─' * 100}\n")

    delta_rows = [
        ('ΔTrades/Day', 'trades_per_day'),
        ('ΔReturn', 'total_return_pct'),
        ('ΔSharpe', 'sharpe'),
        ('ΔCalmar', 'calmar'),
        ('ΔMax DD', 'max_dd_pct'),
        ('ΔAvg PnL/Trade', 'avg_pnl_pct'),
    ]

    print(f"  {'Metric':<20}", end='')
    for n in names:
        print(f"  {n:>{col_w}}", end='')
    print()
    print("  " + "-" * (20 + (col_w + 2) * len(names)))

    for label, key in delta_rows:
        print(f"  {label:<20}", end='')
        for n in names:
            val = results[n][key]
            base_val = base[key]
            if base_val != 0:
                delta_pct = (val / base_val - 1) * 100
                print(f"  {delta_pct:>+10.1f}%", end='  ')
            else:
                print(f"  {'—':>{col_w}}", end='')
        print()

    # ── Yearly breakdown ──
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
            ('Profit Factor', 'profit_factor', '{:.2f}'),
            ('Max DD', 'max_dd_pct', '{:.1f}%'),
            ('Sharpe', 'sharpe', '{:.2f}'),
        ]:
            print(f"    {label:<16}", end='')
            for n in names:
                val = results[n]['yearly'][y][key]
                print(f"  {fmt.format(val):>{col_w}}", end='')
            print()

    # ── Recommendation ──
    print(f"\n{'=' * 100}")
    print("  RECOMMENDATION")
    print(f"{'=' * 100}\n")

    # Score: weighted combination (Sharpe×0.3 + Calmar×0.2 + trades_per_day×0.2 + return×0.15 + -DD×0.15)
    best_name = None
    best_score = -float('inf')
    for n, r in results.items():
        score = (
            (r['sharpe'] / max(base['sharpe'], 0.01)) * 0.30 +
            (r['calmar'] / max(base['calmar'], 0.01)) * 0.20 +
            (r['trades_per_day'] / max(base['trades_per_day'], 0.01)) * 0.20 +
            (r['total_return_pct'] / max(base['total_return_pct'], 0.01)) * 0.15 +
            (base['max_dd_pct'] / max(r['max_dd_pct'], 0.01)) * 0.15
        )
        suffix = ' ★ WINNER' if score == best_score else ''
        if score > best_score:
            best_score = score
            best_name = n

    for n, r in results.items():
        score = (
            (r['sharpe'] / max(base['sharpe'], 0.01)) * 0.30 +
            (r['calmar'] / max(base['calmar'], 0.01)) * 0.20 +
            (r['trades_per_day'] / max(base['trades_per_day'], 0.01)) * 0.20 +
            (r['total_return_pct'] / max(base['total_return_pct'], 0.01)) * 0.15 +
            (base['max_dd_pct'] / max(r['max_dd_pct'], 0.01)) * 0.15
        )
        tag = ' ★ WINNER' if n == best_name else ''
        print(f"  {n:>12}: score={score:.3f}  Sharpe={r['sharpe']:.2f}  "
              f"Calmar={r['calmar']:.1f}  trades/day={r['trades_per_day']:.2f}  "
              f"DD={r['max_dd_pct']:.1f}%{tag}")

    print(f"\n  → {best_name} achieves the best balance of opportunity and performance.")

    # ── Save results ──
    output_path = Path(__file__).parent.parent / 'logs' / 'filter_relaxation_ab_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 100}")
    print("  DONE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
