#!/usr/bin/env python3
"""
Performance Assessment: Option A / B / C — 3-Year Portfolio Backtest
=====================================================================
Runs full portfolio-level simulations (shared capital, 2 slots, 1095 days)
for three configuration variants to assess the MTF / confidence trade-off.

Option A (Status Quo):
  - MTF ON, SMA=12, base_lev=1.5×, mtf_lev=2.5× (eff 3.75×)
  - min_confidence=1, adaptive 2.5x_f1

Option B (Confidence Gate ≥ 3):
  - Same as A but min_confidence=3
  - Fewer but higher-quality trades, ~25% trade reduction
  - 3-year single-dim test: Sharpe +0.46, DD -3.5pp, PF +0.30

Option C (Confidence ≥ 3 + Conservative Leverage):
  - min_confidence=3, base_lev=1.0×, mtf_lev=2.0× (eff 2.0×)
  - Maximum risk reduction: fewer trades + less exposure
  - Tests whether reduced leverage + quality filter compensates for lower return

All three use IDENTICAL data, pipeline, and portfolio simulator.
Each starts fresh with $2,095 × 2 slots across 20 symbols × 1095 days.
Yearly breakdowns show regime-specific performance (Y1=bull, Y2=mixed, Y3=bear).

Uses existing cached data (data/cache_extended/*_1095d_*.npz).
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
from binance.client import Client
from quantbox.core.settings import get_settings
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    compute_signals_v9,
    apply_mtf_confirmation,
    apply_mtf_leverage,
    compute_atr_pct,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

# ═══════════════════════════════════════════════════════════════════════
#  Git Commit Info
# ═══════════════════════════════════════════════════════════════════════

def _get_git_commit() -> str:
    import subprocess
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'


def _find_cache(symbol: str, interval: str, days: int) -> Path | None:
    """Find the most recent cache file for symbol/interval/days."""
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
#  Signal Generation (production-identical)
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(
    data: dict, btc: np.ndarray, close_4h: np.ndarray | None,
    cfg: RubberBandConfig,
    atr_period: int, atr_mult: float, floor: float, cap: float,
    mtf_enabled: bool, mtf_sma: int, mtf_lev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate signals. Returns (close, signals)."""
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
#  Portfolio Simulator (same as portfolio_backtest_2slot.py)
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
    """Simulate portfolio with shared capital and slot limits."""
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
    }


# ═══════════════════════════════════════════════════════════════════════
#  Yearly Breakdown (split equity curve into Y1/Y2/Y3)
# ═══════════════════════════════════════════════════════════════════════

def yearly_breakdown(result: dict, days: int = 1095) -> list[dict]:
    """Split trades into yearly buckets and compute per-year metrics."""
    bars_per_year = 96 * 365  # 15m bars per year
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

        avg_pnl = np.mean([t['pnl_usd'] for t in year_trades]) if year_trades else 0
        avg_bars = np.mean([t['bars_held'] for t in year_trades]) if year_trades else 0

        # Year-specific equity slice for drawdown
        eq = result['equity_curve']
        eq_slice = eq[start_bar:min(end_bar, len(eq))]
        if len(eq_slice) > 0:
            peak = np.maximum.accumulate(eq_slice)
            dd = (peak - eq_slice) / np.maximum(peak, 1e-10)
            year_max_dd = float(np.max(dd)) * 100
        else:
            year_max_dd = 0.0

        # Year-specific Sharpe
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
            'label': ['2023-02→2024-02 (Recovery/Bull)', '2024-02→2025-02 (Bull/Mixed)', '2025-02→2026-02 (Bear/Current)'][y],
            'return_pct': round(ret_pct, 1),
            'trades': n_t,
            'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2),
            'max_dd_pct': round(year_max_dd, 1),
            'sharpe': round(yr_sharpe, 2),
            'avg_pnl_usd': round(avg_pnl, 2),
            'avg_bars_held': round(avg_bars, 1),
        })

    return years


# ═══════════════════════════════════════════════════════════════════════
#  Monte Carlo (fast vectorized)
# ═══════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _compute_max_streaks(is_loss: np.ndarray, n_sims: int, n_trades: int) -> np.ndarray:
    result = np.zeros(n_sims, dtype=np.int64)
    for i in range(n_sims):
        streak = 0
        best = 0
        for j in range(n_trades):
            if is_loss[i, j]:
                streak += 1
                if streak > best:
                    best = streak
            else:
                streak = 0
        result[i] = best
    return result


def monte_carlo(trades: list[dict], capital: float, n_sims: int = 10_000) -> dict:
    """Fast Monte Carlo for summary stats. Returns dict with percentiles."""
    pnl_pcts = np.array([t['pnl_pct'] for t in trades], dtype=np.float64)
    n_t = len(trades)

    rng = np.random.default_rng(seed=42)
    idx = rng.integers(0, len(pnl_pcts), size=(n_sims, n_t))
    sampled = pnl_pcts[idx]
    growth = 1.0 + sampled / 100.0 * 0.5
    cum_eq = capital * np.cumprod(growth, axis=1)
    finals = cum_eq[:, -1]

    running_max = np.maximum.accumulate(cum_eq, axis=1)
    drawdowns = (running_max - cum_eq) / running_max
    max_dds = np.max(drawdowns, axis=1)

    ruin_mask = np.any(cum_eq < capital * 0.5, axis=1)

    is_loss = sampled <= 0
    max_streaks = _compute_max_streaks(is_loss, n_sims, n_t)

    return {
        'p5_return': round(float((np.percentile(finals, 5) / capital - 1) * 100), 1),
        'p25_return': round(float((np.percentile(finals, 25) / capital - 1) * 100), 1),
        'p50_return': round(float((np.percentile(finals, 50) / capital - 1) * 100), 1),
        'p75_return': round(float((np.percentile(finals, 75) / capital - 1) * 100), 1),
        'p95_return': round(float((np.percentile(finals, 95) / capital - 1) * 100), 1),
        'p95_max_dd': round(float(np.percentile(max_dds, 95) * 100), 1),
        'prob_profit': round(float(np.mean(finals > capital) * 100), 1),
        'prob_2x': round(float(np.mean(finals >= capital * 2) * 100), 1),
        'prob_ruin': round(float(np.sum(ruin_mask) / n_sims * 100), 2),
        'p95_max_streak': int(np.percentile(max_streaks, 95)),
        'max_streak': int(np.max(max_streaks)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main: Run 3 variants
# ═══════════════════════════════════════════════════════════════════════

def main():
    commit = _get_git_commit()
    print(f"# Performance Assessment A/B/C — 3-Year Portfolio Backtest")
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
    SLOTS = 2
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # ── Load base config from production ──
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

    # ── Define 3 variants ──
    variants = {
        'A': {
            'name': 'Option A: Status Quo (PROD)',
            'description': 'MTF ON SMA12, conf≥1, base 1.5× / MTF 2.5× (eff 3.75×)',
            'min_confidence': 1,
            'base_leverage': 1.5,
            'mtf_enabled': True,
            'mtf_sma': 12,
            'mtf_lev': 2.5,
            'atr_period': 48,
            'atr_mult': 2.5,
            'floor': 0.01,
            'cap': 0.06,
        },
        'B': {
            'name': 'Option B: Confidence Gate ≥ 3',
            'description': 'MTF ON SMA12, conf≥3, base 1.5× / MTF 2.5× (eff 3.75×)',
            'min_confidence': 3,
            'base_leverage': 1.5,
            'mtf_enabled': True,
            'mtf_sma': 12,
            'mtf_lev': 2.5,
            'atr_period': 48,
            'atr_mult': 2.5,
            'floor': 0.01,
            'cap': 0.06,
        },
        'C': {
            'name': 'Option C: Confidence ≥ 3 + Conservative Leverage',
            'description': 'MTF ON SMA12, conf≥3, base 1.0× / MTF 2.0× (eff 2.0×)',
            'min_confidence': 3,
            'base_leverage': 1.0,
            'mtf_enabled': True,
            'mtf_sma': 12,
            'mtf_lev': 2.0,
            'atr_period': 48,
            'atr_mult': 2.5,
            'floor': 0.01,
            'cap': 0.06,
        },
    }

    # ── Load 3-year data from cache ──
    print(f"Loading {DAYS}-day data for {len(SYMBOLS)} symbols from cache...")
    btc_data = load_15m('BTCUSDT', DAYS)
    if btc_data is None:
        print("ERROR: BTC 15m data not found in cache. Run download_extended_data.py first.")
        sys.exit(1)
    btc_close = btc_data['close']
    print(f"  BTC: {len(btc_close)} bars (15m)")

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS:
        d = load_15m(sym, DAYS)
        if d is None:
            print(f"  SKIP {sym} (no cached data)")
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
        print(f"  {var['description']}")
        print(f"{'=' * 80}")

        # Clone config and apply variant params
        cfg = RubberBandConfig()
        for attr in dir(base_cfg):
            if not attr.startswith('_'):
                try:
                    setattr(cfg, attr, getattr(base_cfg, attr))
                except (AttributeError, TypeError):
                    pass
        cfg.min_confidence = var['min_confidence']
        cfg.base_leverage = var['base_leverage']

        # Generate signals for all symbols
        t0 = time.time()
        all_close = {}
        all_signals = {}
        for sym, data in all_data.items():
            close_4h = all_4h.get(sym)
            c, sigs = generate_signals(
                data, btc_close, close_4h, cfg,
                var['atr_period'], var['atr_mult'], var['floor'], var['cap'],
                var['mtf_enabled'], var['mtf_sma'], var['mtf_lev'],
            )
            all_close[sym] = c
            all_signals[sym] = sigs

        total_entries = sum(
            int(np.sum((s[1:] != 0) & (s[:-1] == 0))) for s in all_signals.values()
        )
        print(f"  Signal entries: {total_entries}  ({time.time()-t0:.1f}s)")

        # Portfolio simulation (flat sizing)
        t0 = time.time()
        result = simulate_portfolio(
            all_close, all_signals,
            initial_capital=CAPITAL, max_slots=SLOTS,
            cost_per_side=COST, funding_per_bar=FUNDING,
            reinvest=False,
        )
        sim_time = time.time() - t0

        # Yearly breakdown
        years = yearly_breakdown(result, DAYS)

        # Monte Carlo
        mc = monte_carlo(result['trades'], CAPITAL) if result['trades'] else {}

        # Print summary
        print(f"\n  Total Return:   {result['total_return_pct']:+,.1f}%")
        print(f"  Final Balance:  ${result['final_balance']:,.0f}")
        print(f"  Max Drawdown:   {result['max_dd_pct']:.1f}%")
        print(f"  Sharpe Ratio:   {result['sharpe']:.2f}")
        print(f"  Calmar Ratio:   {result['calmar']:.1f}")
        print(f"  Trades:         {result['n_trades']}")
        print(f"  Win Rate:       {result['win_rate']:.1f}%")
        print(f"  Profit Factor:  {result['profit_factor']:.2f}")
        print(f"  Avg Bars Held:  {result['avg_bars_held']:.1f}")
        print(f"  Avg PnL/Trade:  ${result['avg_pnl_usd']:.2f} ({result['avg_pnl_pct']:+.3f}%)")
        print(f"  Sim time:       {sim_time:.1f}s")

        print(f"\n  Yearly Breakdown:")
        for yr in years:
            print(f"    Y{yr['year']} ({yr['label']}):")
            print(f"      Return: {yr['return_pct']:+,.1f}%, Trades: {yr['trades']}, "
                  f"Win: {yr['win_rate']:.1f}%, PF: {yr['profit_factor']:.2f}, "
                  f"DD: {yr['max_dd_pct']:.1f}%, Sharpe: {yr['sharpe']:.2f}")

        if mc:
            print(f"\n  Monte Carlo (10k sims):")
            print(f"    P5:  {mc['p5_return']:+,.1f}%   P50: {mc['p50_return']:+,.1f}%   P95: {mc['p95_return']:+,.1f}%")
            print(f"    P(profit): {mc['prob_profit']:.1f}%   P(2×): {mc['prob_2x']:.1f}%   P(ruin): {mc['prob_ruin']:.2f}%")
            print(f"    P95 max DD: {mc['p95_max_dd']:.1f}%   P95 loss streak: {mc['p95_max_streak']}")

        print()

        results[key] = {
            'name': var['name'],
            'description': var['description'],
            'config': {k: v for k, v in var.items() if k not in ('name', 'description')},
            'summary': {k: v for k, v in result.items() if k != 'equity_curve' and k != 'trades'},
            'yearly': years,
            'monte_carlo': mc,
        }

    # ── Comparison Table ──
    print(f"\n{'=' * 100}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 100}")
    header = f"{'Metric':<22} {'Option A (Status Quo)':>22} {'Option B (Conf≥3)':>22} {'Option C (Conf≥3+Low Lev)':>26}"
    print(header)
    print(f"{'─' * 100}")

    for metric, fmt in [
        ('total_return_pct', '{:+,.1f}%'),
        ('final_balance', '${:,.0f}'),
        ('max_dd_pct', '{:.1f}%'),
        ('sharpe', '{:.2f}'),
        ('calmar', '{:.1f}'),
        ('n_trades', '{:,}'),
        ('win_rate', '{:.1f}%'),
        ('profit_factor', '{:.2f}'),
        ('avg_pnl_usd', '${:.2f}'),
        ('avg_pnl_pct', '{:+.3f}%'),
        ('avg_bars_held', '{:.1f}'),
    ]:
        vals = []
        for key in ['A', 'B', 'C']:
            v = results[key]['summary'][metric]
            vals.append(fmt.format(v))
        label = metric.replace('_', ' ').title()
        print(f"  {label:<20} {vals[0]:>22} {vals[1]:>22} {vals[2]:>26}")

    # Y3 (bearish regime) comparison
    print(f"\n{'─' * 100}")
    print(f"  Y3 BEARISH REGIME (2025-02 → 2026-02) — The period that matters most")
    print(f"{'─' * 100}")
    for metric, fmt in [
        ('return_pct', '{:+,.1f}%'),
        ('trades', '{:,}'),
        ('win_rate', '{:.1f}%'),
        ('profit_factor', '{:.2f}'),
        ('max_dd_pct', '{:.1f}%'),
        ('sharpe', '{:.2f}'),
    ]:
        vals = []
        for key in ['A', 'B', 'C']:
            y3 = results[key]['yearly'][2]
            vals.append(fmt.format(y3[metric]))
        label = metric.replace('_', ' ').title()
        print(f"  {label:<20} {vals[0]:>22} {vals[1]:>22} {vals[2]:>26}")

    # Monte Carlo comparison
    print(f"\n{'─' * 100}")
    print(f"  MONTE CARLO RISK PROFILE (10,000 simulations)")
    print(f"{'─' * 100}")
    for metric, label, fmt in [
        ('p5_return', 'P5 Return (worst 5%)', '{:+,.1f}%'),
        ('p50_return', 'P50 Return (median)', '{:+,.1f}%'),
        ('p95_return', 'P95 Return (best 5%)', '{:+,.1f}%'),
        ('p95_max_dd', 'P95 Max Drawdown', '{:.1f}%'),
        ('prob_profit', 'P(Profit)', '{:.1f}%'),
        ('prob_ruin', 'P(Ruin <50%)', '{:.2f}%'),
        ('p95_max_streak', 'P95 Loss Streak', '{}'),
    ]:
        vals = []
        for key in ['A', 'B', 'C']:
            mc = results[key].get('monte_carlo', {})
            v = mc.get(metric, 0)
            vals.append(fmt.format(v))
        print(f"  {label:<24} {vals[0]:>22} {vals[1]:>22} {vals[2]:>26}")

    print(f"\n{'=' * 100}")

    # Save results to JSON
    output_path = Path(__file__).parent.parent / 'logs' / 'performance_abc_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Save full output
    return results


if __name__ == '__main__':
    main()
