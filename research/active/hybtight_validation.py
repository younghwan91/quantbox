#!/usr/bin/env python3
"""
I-HybTight Production Validation Backtest
==========================================
Runs the EXACT I-HybTight config on 6yr data with the SAME PBT logic
as live update_position(). This is the final validation before deployment.

Reuses entry scanning + portfolio simulator from reversal_ab_test.py,
running ONLY the I-HybTight variant to produce clean, verifiable metrics.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/hybtight_validation.py
"""

import sys
import time
import subprocess
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# Import from reversal_ab_test (reuse its scanner + simulator)
from scripts.reversal_ab_test import (
    RevConfig,
    simulate,
    scan_entries,
    scan_momentum_entries,
    apply_mtf_filter,
    load_15m,
    load_4h,
    _compute_stats,
    LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
    ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
    STOCH_ENABLED, STOCH_K, STOCH_D,
    MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
    MTF_SMA_PERIOD, INITIAL_CAPITAL,
)
from quantbox.strategy.rubberband import compute_atr_pct

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


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def main():
    commit = _get_git_commit()
    print("=" * 90)
    print("  I-HybTight PRODUCTION VALIDATION BACKTEST")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Config: target=1.5, stop=0.382, max_hold=25, trail_atr=1.0, runner_mult=3.0")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f} | Leverage: 3.75x | Slots: 2")
    print("=" * 90)

    # ── Load data ──
    print(f"\n📊 Loading 50-symbol data (6yr / 2190d)...")
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
            continue
        all_data[sym] = d
        h4 = load_4h(sym)
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # ── Scan MR entries ──
    print(f"\n🔍 Scanning MR entry signals...")
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
        c, h, l, v, t = (
            d['close'][:n], d['high'][:n], d['low'][:n],
            d['volume'][:n], d['taker_buy'][:n],
        )
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
                dirs, confs, moves, close_4h, MTF_SMA_PERIOD, ratio=16,
            )

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

    # Left-pad arrays to same length
    max_bars = max(len(sym_close[s]) for s in sym_close)
    all_dicts = [
        (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
        (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
        (sym_atr, 'nan'),
    ]
    for sym in list(sym_close.keys()):
        for d_ref, fill in all_dicts:
            arr = d_ref.get(sym)
            if arr is None:
                continue
            if len(arr) < max_bars:
                pad_n = max_bars - len(arr)
                fv = arr[0] if fill == 'first' else (np.nan if fill == 'nan' else 0.0)
                d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

    mr_entries = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    scan_time = time.time() - t0
    print(f"  MR entries: {mr_entries:,}  ({scan_time:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    # Run I-HybTight and A-MR-Prod (control) for comparison
    # ══════════════════════════════════════════════════════════════
    configs = [
        RevConfig(
            name='A-MR-Prod (control)',
            desc='v12.3.3 production: T100_S38_PBT',
            target_retrace=1.0,
            stop_mult=0.382,
            max_hold=20,
            runner_max_hold_mult=2.0,
            trail_atr_mult=1.5,
        ),
        RevConfig(
            name='I-HybTight (v12.4)',
            desc='Production candidate: deeper target + tighter trail + longer runner',
            target_retrace=1.5,
            stop_mult=0.382,
            max_hold=25,
            runner_max_hold_mult=3.0,
            trail_atr_mult=1.0,
        ),
    ]

    results = []
    for cfg in configs:
        print(f"\n{'─'*90}")
        print(f"  Running: {cfg.name}")
        print(f"  Params: target={cfg.target_retrace}, stop={cfg.stop_mult}, "
              f"max_hold={cfg.max_hold}, trail_atr={cfg.trail_atr_mult}, "
              f"runner_mult={cfg.runner_max_hold_mult}")
        print(f"{'─'*90}")
        t0 = time.time()

        stats = simulate(
            sym_close, sym_high, sym_low,
            sym_dirs, sym_confs, sym_moves,
            cfg,
            sym_atr=sym_atr,
        )

        elapsed = time.time() - t0
        results.append((cfg.name, stats))

        print(f"\n  📊 {cfg.name} Results ({elapsed:.1f}s):")
        print(f"  {'─'*50}")
        print(f"  Return:        {stats['total_return_pct']:>+10.1f}%")
        print(f"  Final Balance: ${stats['final_balance']:>10,.2f}")
        print(f"  Max DD:        {stats['max_dd_pct']:>10.1f}%")
        print(f"  Sharpe:        {stats['sharpe']:>10.2f}")
        print(f"  Calmar:        {stats['calmar']:>10.1f}")
        print(f"  Profit Factor: {stats['profit_factor']:>10.2f}")
        print(f"  Win Rate:      {stats['win_rate']:>10.1f}%")
        print(f"  R:R (actual):  {stats['actual_rr']:>10.2f}")
        print(f"  Trades:        {stats['n_trades']:>10,}")
        print(f"  Trades/Day:    {stats['trades_per_day']:>10.2f}")
        print(f"  Avg PnL:       {stats['avg_pnl_pct']:>+10.3f}%")
        print(f"  Avg Win:       {stats['avg_win_pct']:>+10.3f}%")
        print(f"  Avg Loss:      {stats['avg_loss_pct']:>10.3f}%")
        print(f"  Avg Hold:      {stats['avg_bars_held']:>10.1f} bars")
        print(f"  Runner Trades: {stats['n_runner_trades']:>10,}")
        print(f"  Runner WR:     {stats['runner_win_rate']:>10.1f}%")
        print(f"  Runner Avg:    {stats['runner_avg_pnl']:>+10.3f}%")
        print(f"  Exit Reasons:  {stats['exit_reasons']}")

    # ══════════════════════════════════════════════════════════════
    # Comparison
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  COMPARISON: A-MR-Prod vs I-HybTight")
    print(f"{'='*90}")
    print(f"  {'Metric':<20} {'A-MR-Prod':>15} {'I-HybTight':>15} {'Delta':>15}")
    print(f"  {'─'*65}")

    a, b = results[0][1], results[1][1]
    metrics = [
        ('Return %', 'total_return_pct', '+.1f'),
        ('Max DD %', 'max_dd_pct', '.1f'),
        ('Sharpe', 'sharpe', '.2f'),
        ('Calmar', 'calmar', '.1f'),
        ('PF', 'profit_factor', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('R:R', 'actual_rr', '.2f'),
        ('Trades', 'n_trades', ','),
        ('Trades/Day', 'trades_per_day', '.2f'),
        ('Avg PnL %', 'avg_pnl_pct', '+.3f'),
        ('Runner Trades', 'n_runner_trades', ','),
        ('Runner WR %', 'runner_win_rate', '.1f'),
    ]
    for label, key, fmt in metrics:
        va, vb = a[key], b[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            sign = '+' if delta > 0 else ''
            print(f"  {label:<20} {format(va, fmt):>15} {format(vb, fmt):>15} {sign}{format(delta, fmt):>14}")
        else:
            print(f"  {label:<20} {format(va, fmt):>15} {format(vb, fmt):>15}")

    # ══════════════════════════════════════════════════════════════
    # Breakeven Analysis
    # ══════════════════════════════════════════════════════════════
    ht = b  # HybTight stats
    if ht['avg_win_pct'] > 0 and ht['avg_loss_pct'] > 0:
        be_wr = ht['avg_loss_pct'] / (ht['avg_win_pct'] + ht['avg_loss_pct']) * 100
        margin = ht['win_rate'] - be_wr
        print(f"\n  📐 Breakeven Analysis (I-HybTight):")
        print(f"     Breakeven WR: {be_wr:.1f}%")
        print(f"     Actual WR:    {ht['win_rate']:.1f}%")
        print(f"     Margin:       {margin:+.1f}pp")

    print(f"\n{'='*90}")
    print(f"  Commit: {commit}")
    print(f"  Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
