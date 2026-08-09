#!/usr/bin/env python3
"""
Universe Expansion Sweep v2 — B-30 vs B-40 vs B-50
====================================================
Extends the previous sweep (20→30 winner) to test whether going wider
to 40 or 50 symbols further improves risk-adjusted returns.

Variants tested:
  B-30:  30 symbols, 2 slots (previous winner — baseline)
  B-40:  40 symbols, 2 slots (adds 10 mid-cap alts)
  B-50:  50 symbols, 2 slots (adds 20 mid-cap alts)

All variants use conf≥3, MTF block, 3.75× trend-aligned leverage.
All data: 1095 days (3 years), left-padded for shorter histories.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/universe_expansion_sweep_v2.py
"""

import sys
import time
import json
import subprocess
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


# ═══════════════════════════════════════════════════════════════════════
#  Data Download & Loading
# ═══════════════════════════════════════════════════════════════════════

def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    pattern = f'{symbol}_{interval}_{days}d_*.npz'
    matches = sorted(CACHE_DIR.glob(pattern), reverse=True)
    return matches[0] if matches else None


def _cache_path(symbol: str, interval: str, days: int) -> Path:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return CACHE_DIR / f'{symbol}_{interval}_{days}d_{today}.npz'


def download_15m(client, symbol: str, days: int) -> dict | None:
    cache = _find_cache(symbol, '15m', days)
    if cache is not None and cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}

    cache = _cache_path(symbol, '15m', days)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    all_klines, cur = [], start_ms
    while cur < end_ms:
        klines = client.futures_klines(
            symbol=symbol, interval='15m',
            startTime=cur, endTime=end_ms, limit=1500,
        )
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        time.sleep(0.35)

    if len(all_klines) < 200:
        return None
    all_klines = all_klines[:-1]

    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    high = np.array([float(k[2]) for k in all_klines], dtype=np.float64)
    low = np.array([float(k[3]) for k in all_klines], dtype=np.float64)
    volume = np.array([float(k[5]) for k in all_klines], dtype=np.float64)
    taker_buy = np.array([float(k[9]) for k in all_klines], dtype=np.float64)

    np.savez_compressed(cache, close=close, high=high, low=low,
                        volume=volume, taker_buy=taker_buy)
    print(f"    ✓ Saved 15m: {len(close)} bars")
    return {'close': close, 'high': high, 'low': low,
            'volume': volume, 'taker_buy': taker_buy}


def download_4h(client, symbol: str, days: int) -> np.ndarray | None:
    cache = _find_cache(symbol, '4h', days)
    if cache is not None and cache.exists():
        d = np.load(cache)
        return d['close']

    cache = _cache_path(symbol, '4h', days)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    all_klines, cur = [], start_ms
    while cur < end_ms:
        klines = client.futures_klines(
            symbol=symbol, interval='4h',
            startTime=cur, endTime=end_ms, limit=1500,
        )
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        time.sleep(0.35)

    if len(all_klines) < 12:
        return None
    all_klines = all_klines[:-1]
    close = np.array([float(k[4]) for k in all_klines], dtype=np.float64)
    np.savez_compressed(cache, close=close)
    print(f"    ✓ Saved 4h: {len(close)} bars")
    return close


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
#  Signal Generation (IDENTICAL to performance_assessment_abc.py)
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(
    data: dict, btc: np.ndarray, close_4h: np.ndarray | None,
    cfg: RubberBandConfig,
    atr_period: int, atr_mult: float, floor: float, cap: float,
    mtf_enabled: bool, mtf_sma: int, mtf_lev: float,
) -> tuple[np.ndarray, np.ndarray]:
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
        sigs = apply_mtf_confirmation(sigs, close_4h, mtf_sma, ratio=16)
        sigs = apply_mtf_leverage(sigs, mtf_lev)

    return c, sigs


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Simulator (IDENTICAL to performance_assessment_abc.py)
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
        'avg_bars_held': round(avg_bars, 1),
        'avg_pnl_usd': round(avg_pnl, 2),
        'avg_pnl_pct': round(avg_pnl_pct, 3),
        'max_slots': max_slots,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_wins': len(long_wins),
        'short_wins': len(short_wins),
        'long_wr': round(len(long_wins) / len(long_trades) * 100, 1) if long_trades else 0,
        'short_wr': round(len(short_wins) / len(short_trades) * 100, 1) if short_trades else 0,
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

        long_t = [t for t in year_trades if t['signal'] > 0]
        short_t = [t for t in year_trades if t['signal'] < 0]

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
            if len(eq_rets) > 0 and np.std(eq_rets) > 0:
                yr_sharpe = float(np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25))
            else:
                yr_sharpe = 0.0
        else:
            yr_sharpe = 0.0

        years.append({
            'year': y + 1,
            'label': [
                '2023-02→2024-02 (Recovery/Bull)',
                '2024-02→2025-02 (Bull/Mixed)',
                '2025-02→2026-02 (Bear/Current)',
            ][y],
            'return_pct': round(ret_pct, 1),
            'trades': n_t,
            'longs': len(long_t),
            'shorts': len(short_t),
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

    print("=" * 70)
    print("  UNIVERSE EXPANSION SWEEP v2 — B-30 / B-40 / B-50")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  conf≥3 + MTF block + Adaptive 2.5×_f1, 2 slots")
    print("=" * 70)

    # ── Symbol Lists ──
    # Tier 1: Original 20
    SYMBOLS_20 = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
    ]
    # Tier 2: +10 from previous sweep (v1 winner)
    TIER2_10 = [
        'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT', 'BCHUSDT',
        'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    ]
    # Tier 3: +10 mid-cap alts with full 3yr history (ranked by mcap/volume)
    TIER3_10 = [
        'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT',
        'EGLDUSDT', 'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT',
    ]
    # Tier 4: +10 more (ranked by mcap/volume, all 3yr)
    TIER4_10 = [
        'RUNEUSDT', 'DYDXUSDT', 'CRVUSDT', 'LDOUSDT', 'COMPUSDT',
        'MKRUSDT', 'SNXUSDT', 'VETUSDT', 'EOSUSDT', 'APEUSDT',
    ]

    SYMBOLS_30 = SYMBOLS_20 + TIER2_10
    SYMBOLS_40 = SYMBOLS_30 + TIER3_10
    SYMBOLS_50 = SYMBOLS_40 + TIER4_10
    ALL_SYMBOLS = SYMBOLS_50  # superset

    DAYS = 1095
    CAPITAL = 2095.0
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # ── Config (Option B: conf≥3, MTF block) ──
    base_cfg = RubberBandConfig()
    base_cfg.lookback = settings.rb_lookback
    base_cfg.threshold = settings.rb_threshold
    base_cfg.target_retrace = settings.rb_target_retrace
    base_cfg.stop_mult = settings.rb_stop_mult
    base_cfg.max_hold = settings.rb_max_hold
    base_cfg.stop_decay = settings.rb_stop_decay
    base_cfg.btc_directional_period = settings.rb_btc_directional_period
    base_cfg.btc_regime_period = settings.rb_btc_regime_period
    base_cfg.asset_trend_period = settings.rb_asset_trend_period
    base_cfg.base_leverage = settings.rb_base_leverage
    base_cfg.confidence_leverage_step = settings.rb_confidence_leverage_step
    base_cfg.max_leverage = settings.rb_max_leverage
    base_cfg.rsi_period = settings.rb_rsi_period
    base_cfg.vol_period = settings.rb_vol_period
    base_cfg.stoch_enabled = settings.rb_stoch_enabled
    base_cfg.stoch_k_period = settings.rb_stoch_k_period
    base_cfg.stoch_d_period = settings.rb_stoch_d_period
    base_cfg.macd_enabled = settings.rb_macd_enabled
    base_cfg.macd_fast = settings.rb_macd_fast
    base_cfg.macd_slow = settings.rb_macd_slow
    base_cfg.macd_signal_period = settings.rb_macd_signal_period
    base_cfg.dynamic_target_enabled = settings.rb_dynamic_target_enabled
    base_cfg.target_conf3 = settings.rb_target_conf3
    base_cfg.target_conf4 = settings.rb_target_conf4
    base_cfg.target_conf5 = settings.rb_target_conf5
    base_cfg.min_confidence = 3  # Option B

    ATR_PERIOD = 48
    ATR_MULT = 2.5
    FLOOR = 0.01
    CAP = 0.06
    MTF_SMA = 12
    MTF_LEV = 2.5

    # ── Step 1: Download missing data ──
    print(f"\n📥 Checking data for {len(ALL_SYMBOLS)} symbols × {DAYS}d...")

    need_download = []
    for sym in ALL_SYMBOLS:
        if _find_cache(sym, '15m', DAYS) is None or _find_cache(sym, '4h', DAYS) is None:
            need_download.append(sym)

    if need_download:
        print(f"  Need to download {len(need_download)}: {', '.join(need_download)}")
        from binance.client import Client
        client = Client(settings.binance_api_key, settings.binance_api_secret)

        for i, sym in enumerate(need_download):
            print(f"  [{i+1}/{len(need_download)}] {sym}:")
            if _find_cache(sym, '15m', DAYS) is None:
                print(f"    Downloading 15m ({DAYS}d)...")
                download_15m(client, sym, DAYS)
            if _find_cache(sym, '4h', DAYS) is None:
                print(f"    Downloading 4h ({DAYS}d)...")
                download_4h(client, sym, DAYS)
    else:
        print("  All data cached ✓")

    # ── Step 2: Load all data ──
    print(f"\n📊 Loading data...")

    btc_data = load_15m('BTCUSDT', DAYS)
    if btc_data is None:
        print("ERROR: BTC 15m data not found")
        sys.exit(1)
    btc_close = btc_data['close']

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in ALL_SYMBOLS:
        d = load_15m(sym, DAYS)
        if d is None:
            print(f"  SKIP {sym} (no 15m data)")
            continue
        h4 = load_4h(sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4

    available_30 = [s for s in SYMBOLS_30 if s in all_data]
    available_40 = [s for s in SYMBOLS_40 if s in all_data]
    available_50 = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  30-universe: {len(available_30)} loaded")
    print(f"  40-universe: {len(available_40)} loaded")
    print(f"  50-universe: {len(available_50)} loaded")

    # ── Step 3: Generate signals for all symbols ──
    print(f"\n⚡ Generating signals (conf≥3 + MTF)...")

    all_close_map: dict[str, np.ndarray] = {}
    all_signals_map: dict[str, np.ndarray] = {}

    new_syms = set(TIER3_10 + TIER4_10)
    for sym in available_50:
        data = all_data[sym]
        close_4h = all_4h.get(sym)
        c, sigs = generate_signals(
            data, btc_close, close_4h, base_cfg,
            ATR_PERIOD, ATR_MULT, FLOOR, CAP,
            True, MTF_SMA, MTF_LEV,
        )
        all_close_map[sym] = c
        all_signals_map[sym] = sigs

        n_entries = int(np.sum((sigs[1:] != 0) & (sigs[:-1] == 0)))
        n_long = int(np.sum((sigs[1:] > 0) & (sigs[:-1] == 0)))
        n_short = int(np.sum((sigs[1:] < 0) & (sigs[:-1] == 0)))
        tier = ''
        if sym in TIER3_10:
            tier = ' [T3]'
        elif sym in TIER4_10:
            tier = ' [T4]'
        print(f"  {sym:<16} {n_entries:>4} entries (L:{n_long} S:{n_short})  [{len(c)} bars]{tier}")

    # ── Step 3b: Left-pad shorter symbols ──
    max_bars = max(len(v) for v in all_close_map.values())
    padded_count = 0
    for sym in list(all_close_map.keys()):
        c = all_close_map[sym]
        s = all_signals_map[sym]
        if len(c) < max_bars:
            pad_n = max_bars - len(c)
            all_close_map[sym] = np.concatenate([np.full(pad_n, c[0]), c])
            all_signals_map[sym] = np.concatenate([np.zeros(pad_n), s])
            padded_count += 1
            print(f"  ⚠ {sym}: left-padded {pad_n} bars ({pad_n/96:.0f}d)")

    if padded_count > 0:
        print(f"  Aligned {padded_count} symbols to {max_bars} bars ({max_bars/96:.0f}d)")
    else:
        print(f"  All symbols already {max_bars} bars ✓")

    # ── Step 4: Run 3 variants ──
    variants = {
        'B-30': {'symbols': available_30, 'slots': 2},
        'B-40': {'symbols': available_40, 'slots': 2},
        'B-50': {'symbols': available_50, 'slots': 2},
    }

    results: dict[str, dict] = {}
    for name, var in variants.items():
        syms = var['symbols']
        slots = var['slots']
        print(f"\n🔄 Simulating {name} ({len(syms)} symbols, {slots} slots)...")

        sym_close = {s: all_close_map[s] for s in syms}
        sym_sigs = {s: all_signals_map[s] for s in syms}

        t0 = time.time()
        result = simulate_portfolio(
            sym_close, sym_sigs,
            initial_capital=CAPITAL, max_slots=slots,
            cost_per_side=COST, funding_per_bar=FUNDING,
            reinvest=False,
        )
        elapsed = time.time() - t0

        years = yearly_breakdown(result, DAYS)
        n_days = len(result['equity_curve']) / 96

        result_summary = {
            'symbols': len(syms),
            'slots': slots,
            'total_return_pct': result['total_return_pct'],
            'final_balance': result['final_balance'],
            'max_dd_pct': result['max_dd_pct'],
            'sharpe': result['sharpe'],
            'calmar': result['calmar'],
            'n_trades': result['n_trades'],
            'trades_per_day': round(result['n_trades'] / n_days, 2),
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
        results[name] = result_summary

        print(f"  Return: {result['total_return_pct']:+,.1f}%  Sharpe: {result['sharpe']:.2f}  "
              f"DD: {result['max_dd_pct']:.1f}%  Trades: {result['n_trades']} "
              f"({result_summary['trades_per_day']}/day)  WR: {result['win_rate']:.1f}%  "
              f"PF: {result['profit_factor']:.2f}  [{elapsed:.1f}s]")

    # ══════════════════════════════════════════════════════════════
    #   RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    names = list(results.keys())
    print(f"\n\n{'=' * 80}")
    print("  UNIVERSE EXPANSION SWEEP v2 — RESULTS")
    print(f"{'=' * 80}\n")

    print(f"  {'Metric':<20}", end='')
    for n in names:
        print(f"  {n:>14}", end='')
    print()
    print("  " + "-" * (20 + 16 * len(names)))

    rows = [
        ('Symbols', 'symbols', '{:d}'),
        ('Slots', 'slots', '{:d}'),
        ('Total Return', 'total_return_pct', '{:+,.1f}%'),
        ('Final Balance', 'final_balance', '${:,.0f}'),
        ('Sharpe', 'sharpe', '{:.2f}'),
        ('Calmar', 'calmar', '{:.1f}'),
        ('Max Drawdown', 'max_dd_pct', '{:.1f}%'),
        ('Total Trades', 'n_trades', '{:,d}'),
        ('Trades/Day', 'trades_per_day', '{:.2f}'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Avg PnL/Trade', 'avg_pnl_pct', '{:+.3f}%'),
        ('Longs', 'long_trades', '{:,d}'),
        ('Shorts', 'short_trades', '{:,d}'),
        ('Long WR', 'long_wr', '{:.1f}%'),
        ('Short WR', 'short_wr', '{:.1f}%'),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<20}", end='')
        for n in names:
            val = results[n][key]
            print(f"  {fmt.format(val):>14}", end='')
        print()

    # ── Yearly breakdown ──
    print(f"\n{'─' * 80}")
    print("  YEARLY BREAKDOWN")
    print(f"{'─' * 80}")

    for y in range(3):
        print(f"\n  Year {y+1} ({results[names[0]]['yearly'][y]['label']}):")
        print(f"  {'Metric':<18}", end='')
        for n in names:
            print(f"  {n:>14}", end='')
        print()

        year_rows = [
            ('Return', 'return_pct', '{:+,.1f}%'),
            ('Trades', 'trades', '{:,d}'),
            ('Longs', 'longs', '{:,d}'),
            ('Shorts', 'shorts', '{:,d}'),
            ('Win Rate', 'win_rate', '{:.1f}%'),
            ('Profit Factor', 'profit_factor', '{:.2f}'),
            ('Max DD', 'max_dd_pct', '{:.1f}%'),
            ('Sharpe', 'sharpe', '{:.2f}'),
        ]

        for label, key, fmt in year_rows:
            print(f"    {label:<16}", end='')
            for n in names:
                val = results[n]['yearly'][y][key]
                print(f"  {fmt.format(val):>14}", end='')
            print()

    # ── Per-symbol signal counts (all 50) ──
    print(f"\n{'─' * 80}")
    print("  PER-SYMBOL SIGNAL COUNTS (conf≥3, MTF blocked)")
    print(f"{'─' * 80}\n")

    signal_counts = []
    for sym in available_50:
        sigs = all_signals_map[sym]
        n_entries = int(np.sum((sigs[1:] != 0) & (sigs[:-1] == 0)))
        n_long = int(np.sum((sigs[1:] > 0) & (sigs[:-1] == 0)))
        n_short = int(np.sum((sigs[1:] < 0) & (sigs[:-1] == 0)))
        signal_counts.append((sym, n_entries, n_long, n_short))

    signal_counts.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Symbol':<16} {'Total':>8} {'Long':>8} {'Short':>8} {'L/S':>8}  Tier")
    for sym, total, lng, sht in signal_counts:
        ls_ratio = lng / sht if sht > 0 else float('inf')
        if sym in TIER4_10:
            tier = 'T4'
        elif sym in TIER3_10:
            tier = 'T3'
        elif sym in TIER2_10:
            tier = 'T2'
        else:
            tier = 'T1'
        print(f"  {sym:<16} {total:>8} {lng:>8} {sht:>8} {ls_ratio:>8.1f}  {tier}")

    # Tier signal totals
    t1_total = sum(t for s, t, _, _ in signal_counts if s in SYMBOLS_20)
    t2_total = sum(t for s, t, _, _ in signal_counts if s in TIER2_10)
    t3_total = sum(t for s, t, _, _ in signal_counts if s in TIER3_10)
    t4_total = sum(t for s, t, _, _ in signal_counts if s in TIER4_10)
    print(f"\n  T1 (orig 20): {t1_total:,} signals")
    print(f"  T2 (+10 v1):  {t2_total:,} signals")
    print(f"  T3 (+10 v2):  {t3_total:,} signals")
    print(f"  T4 (+10 v2):  {t4_total:,} signals")
    print(f"  Total 50:     {t1_total + t2_total + t3_total + t4_total:,} signals")

    # ── Analysis ──
    print(f"\n{'=' * 80}")
    print("  ANALYSIS")
    print(f"{'=' * 80}")

    b30 = results['B-30']

    print(f"\n  Trade frequency change (vs B-30 baseline):")
    for name, r in results.items():
        delta = (r['trades_per_day'] / b30['trades_per_day'] - 1) * 100 if b30['trades_per_day'] > 0 else 0
        print(f"    {name:>10}: {r['trades_per_day']:.2f} trades/day ({delta:+.0f}%)")

    print(f"\n  Return comparison:")
    for name, r in results.items():
        delta = (r['total_return_pct'] / b30['total_return_pct'] - 1) * 100 if b30['total_return_pct'] != 0 else 0
        print(f"    {name:>10}: {r['total_return_pct']:+,.1f}% ({delta:+.1f}% vs B-30)")

    print(f"\n  Sharpe comparison:")
    for name, r in results.items():
        delta = (r['sharpe'] / b30['sharpe'] - 1) * 100 if b30['sharpe'] != 0 else 0
        print(f"    {name:>10}: {r['sharpe']:.2f} ({delta:+.1f}% vs B-30)")

    print(f"\n  Quality check (avg PnL/trade):")
    for name, r in results.items():
        print(f"    {name:>10}: {r['avg_pnl_pct']:+.3f}% per trade")

    # ── Save results ──
    output_path = Path(__file__).parent.parent / 'logs' / 'universe_expansion_v2_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 80}")
    print("  DONE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
