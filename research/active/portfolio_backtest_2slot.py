#!/usr/bin/env python3
"""
Portfolio-Level Backtest: Realistic $2,280 / 2-slot Simulation
==============================================================
Models the ACTUAL live scalper behavior:
  - Shared capital pool ($2,280 starting)
  - Max N concurrent positions (default 2)
  - Dynamic capital allocation: balance / max(1, n_active+1)
  - Per-position leverage (effective 3.75x when MTF confirms)
  - Round-trip costs + funding
  - Adaptive threshold 2.5x_f1 (production config)

This is NOT a per-symbol independent backtest. It simulates the portfolio
exactly as the live scalper would run it, bar by bar across all 20 symbols.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import time
import numpy as np
import numba
from datetime import datetime, timedelta, timezone
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
#  Data Loading (reuse existing cache)
# ═══════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache_extended'


def _cache_path(symbol: str, interval: str, days: int) -> Path:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    return CACHE_DIR / f'{symbol}_{interval}_{days}d_{today}.npz'


def download_15m(client: Client, symbol: str, days: int) -> dict | None:
    cache = _cache_path(symbol, '15m', days)
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in ['close', 'high', 'low', 'volume', 'taker_buy']}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines = []
    cur = start_ms
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
    return {'close': close, 'high': high, 'low': low,
            'volume': volume, 'taker_buy': taker_buy}


def download_4h(client: Client, symbol: str, days: int) -> np.ndarray | None:
    cache = _cache_path(symbol, '4h', days)
    if cache.exists():
        d = np.load(cache)
        return d['close']

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_klines = []
    cur = start_ms
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
    return close


# ═══════════════════════════════════════════════════════════════════════
#  Adaptive Signal Generation (production-identical)
# ═══════════════════════════════════════════════════════════════════════

def generate_signals(
    data: dict, btc: np.ndarray, close_4h: np.ndarray | None,
    cfg: RubberBandConfig,
    atr_period: int, atr_mult: float, floor: float, cap: float,
    mtf_enabled: bool, mtf_sma: int, mtf_lev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate adaptive signals with MTF pipeline. Returns (close, signals)."""
    c = data['close']
    h = data['high']
    l = data['low']
    v = data['volume']
    t = data['taker_buy']
    n = min(len(c), len(btc))
    c, h, l, v, t, b = c[:n], h[:n], l[:n], v[:n], t[:n], btc[:n]

    # compute_signals_v9 handles adaptive threshold internally when atr_period > 0
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
#  Portfolio Simulator (bar-by-bar, shared capital, slot-limited)
# ═══════════════════════════════════════════════════════════════════════

def simulate_portfolio(
    all_close: dict[str, np.ndarray],
    all_signals: dict[str, np.ndarray],
    initial_capital: float = 2280.0,
    max_slots: int = 2,
    cost_per_side: float = 0.0004,
    funding_per_bar: float = 0.01 / 100 / 32,
    reinvest: bool = False,
) -> dict:
    """Simulate portfolio with shared capital and slot limits.

    Args:
        all_close: {symbol: close_array} for all symbols.
        all_signals: {symbol: signal_array} for all symbols.
        initial_capital: Starting balance ($).
        max_slots: Maximum concurrent positions.
        cost_per_side: Trading fee per side.
        funding_per_bar: Funding cost per bar per unit position.
        reinvest: If False (default), position sizes are ALWAYS based on
                  initial_capital (flat sizing). Profits accumulate in balance
                  but don't inflate position sizes. This is realistic for
                  small accounts. If True, sizes grow with equity (compound).

    Returns:
        Dict with equity curve, trade history, and summary metrics.
    """
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
        # 1. Update unrealized PnL and check exits
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

        # 2. Close exiting positions
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

        # 3. Open new positions (if slots available)
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

                # ── Position sizing ──
                # sizing_capital: base for allocation (initial or current equity)
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

        # 4. Calculate portfolio equity
        total_unrealized = sum(p.get('unrealized', 0.0) for p in positions.values())
        equity_curve[i] = balance + total_unrealized

        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # ── Compute summary metrics ──
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    n_wins = len(wins)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    total_return = (equity_curve[-1] / initial_capital - 1.0) * 100
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = sum(abs(t['pnl_usd']) for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Sharpe (annualized from bar returns)
    eq_rets = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    eq_rets = eq_rets[~np.isnan(eq_rets)]
    if len(eq_rets) > 0 and np.std(eq_rets) > 0:
        sharpe = np.mean(eq_rets) / np.std(eq_rets) * np.sqrt(96 * 365.25)
    else:
        sharpe = 0.0

    calmar = total_return / (max_dd * 100) if max_dd > 0 else 0.0

    avg_bars = np.mean([t['bars_held'] for t in trades]) if trades else 0
    avg_pnl = np.mean([t['pnl_usd'] for t in trades]) if trades else 0

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'total_return_pct': total_return,
        'final_balance': equity_curve[-1],
        'initial_capital': initial_capital,
        'max_dd_pct': max_dd * 100,
        'sharpe': sharpe,
        'calmar': calmar,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'avg_bars_held': avg_bars,
        'avg_pnl_usd': avg_pnl,
        'max_slots': max_slots,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    settings = get_settings()
    client = Client(settings.binance_api_key, settings.binance_api_secret)
    print("Connected to Binance\n")

    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
        'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
        'ICPUSDT', 'FILUSDT',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365, help='Lookback days (default: 365)')
    parser.add_argument('--capital', type=float, default=2095.0, help='Starting capital (default: 2095)')
    parser.add_argument('--slots', type=int, default=2, help='Max concurrent slots (default: 2)')
    args = parser.parse_args()

    DAYS = args.days
    COST = 0.0004
    FUNDING = 0.01 / 100 / 32

    # Load production config
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
    cfg.min_confidence = settings.rb_min_confidence
    cfg.rsi_period = settings.rb_rsi_period
    cfg.vol_period = settings.rb_vol_period
    cfg.base_leverage = settings.rb_base_leverage
    cfg.confidence_leverage_step = settings.rb_confidence_leverage_step
    cfg.max_leverage = settings.rb_max_leverage
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

    mtf_enabled = settings.rb_mtf_confirmation_enabled
    mtf_sma = settings.rb_mtf_sma_period
    mtf_lev = settings.rb_mtf_leverage_mult

    # Adaptive threshold config: 2.5x_f1 (production)
    atr_period = settings.rb_atr_period      # 48
    atr_mult = settings.rb_atr_multiplier    # 2.5
    floor = settings.rb_threshold_floor      # 0.01
    cap = settings.rb_threshold_cap          # 0.06

    print(f"Production config: 2.5x_f1")
    print(f"  ATR period={atr_period}, mult={atr_mult}, floor={floor*100:.1f}%, cap={cap*100:.1f}%")
    print(f"  Leverage: base={cfg.base_leverage}x, MTF mult={mtf_lev}x → eff={cfg.base_leverage*mtf_lev}x")
    print(f"  MTF: enabled={mtf_enabled}, sma={mtf_sma}")
    print()

    # ── Download data ──
    print(f"Loading {DAYS}-day data for {len(symbols)} symbols...")
    btc_data = download_15m(client, 'BTCUSDT', DAYS)
    btc_close = btc_data['close']
    btc_4h = download_4h(client, 'BTCUSDT', DAYS)
    print(f"  BTC: {len(btc_close)} bars (15m)")

    all_data = {}
    all_4h = {}
    for sym in symbols:
        d = download_15m(client, sym, DAYS)
        if d is None:
            print(f"  SKIP {sym}")
            continue
        h4 = download_4h(client, sym, DAYS)
        all_data[sym] = d
        if h4 is not None:
            all_4h[sym] = h4
        time.sleep(0.05)
    print(f"  Loaded {len(all_data)} symbols\n")

    # ── Generate signals for all symbols ──
    print("Generating signals (adaptive 2.5x_f1 + MTF)...")
    all_close = {}
    all_signals = {}
    for sym, data in all_data.items():
        close_4h = all_4h.get(sym)
        c, sigs = generate_signals(
            data, btc_close, close_4h, cfg,
            atr_period, atr_mult, floor, cap,
            mtf_enabled, mtf_sma, mtf_lev,
        )
        all_close[sym] = c
        all_signals[sym] = sigs

    total_entries = sum(
        np.sum(np.diff(np.sign(s)) != 0) for s in all_signals.values()
    )
    print(f"  Total signal transitions: {total_entries}\n")

    # ── Run portfolio simulations ──
    # PRIMARY: Flat sizing (no reinvestment) — realistic for small accounts
    # SECONDARY: Compound (reinvest) — theoretical upper bound

    CAPITAL = args.capital
    SLOTS = args.slots

    print("=" * 80)
    print(f"PORTFOLIO BACKTEST: ${CAPITAL:,.0f} / {SLOTS} slots ({DAYS}d, Adaptive 2.5x_f1, {len(all_data)} symbols)")
    print("=" * 80)

    # ── 1. FLAT SIZING (the truth for your account) ──
    result_flat = simulate_portfolio(
        all_close, all_signals,
        initial_capital=CAPITAL, max_slots=SLOTS,
        cost_per_side=COST, funding_per_bar=FUNDING,
        reinvest=False,
    )

    # ── 2. COMPOUND SIZING (theoretical if you reinvest everything) ──
    result_comp = simulate_portfolio(
        all_close, all_signals,
        initial_capital=CAPITAL, max_slots=SLOTS,
        cost_per_side=COST, funding_per_bar=FUNDING,
        reinvest=True,
    )

    # ── Print Results ──
    for label, result, mode in [
        ("FLAT SIZING (realistic — no reinvestment)", result_flat, "flat"),
        ("COMPOUND SIZING (reinvest all profits)", result_comp, "compound"),
    ]:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")
        print(f"  Starting Capital:  ${CAPITAL:,.0f}")
        print(f"  Final Balance:     ${result['final_balance']:,.0f}")
        print(f"  Total Return:      {result['total_return_pct']:+,.1f}%")
        print(f"  Max Drawdown:      {result['max_dd_pct']:.1f}%")
        print(f"  Sharpe Ratio:      {result['sharpe']:.2f}")
        print(f"  Calmar Ratio:      {result['calmar']:.1f}")
        print(f"  Trades:            {result['n_trades']}")
        print(f"  Win Rate:          {result['win_rate']:.1f}%")
        print(f"  Profit Factor:     {result['profit_factor']:.2f}")
        print(f"  Avg Bars Held:     {result['avg_bars_held']:.1f}")
        print(f"  Avg PnL/Trade:     ${result['avg_pnl_usd']:.2f}")
        print(f"  Gross Profit:      ${result['gross_profit']:,.0f}")
        print(f"  Gross Loss:        ${result['gross_loss']:,.0f}")

        # Per-trade return distribution
        tr = result['trades']
        if tr:
            pcts = [t['pnl_pct'] for t in tr]
            pcts_sorted = sorted(pcts)
            median_pct = pcts_sorted[len(pcts_sorted) // 2]
            p25 = pcts_sorted[int(len(pcts_sorted) * 0.25)]
            p75 = pcts_sorted[int(len(pcts_sorted) * 0.75)]
            avg_win = np.mean([t['pnl_pct'] for t in tr if t['pnl_usd'] > 0]) if any(t['pnl_usd'] > 0 for t in tr) else 0
            avg_loss = np.mean([t['pnl_pct'] for t in tr if t['pnl_usd'] <= 0]) if any(t['pnl_usd'] <= 0 for t in tr) else 0
            print(f"\n  Per-Trade Return Distribution:")
            print(f"    Mean:    {np.mean(pcts):+.3f}%")
            print(f"    Median:  {median_pct:+.3f}%")
            print(f"    P25:     {p25:+.3f}%")
            print(f"    P75:     {p75:+.3f}%")
            print(f"    Avg Win: {avg_win:+.3f}%")
            print(f"    Avg Loss:{avg_loss:+.3f}%")

        # Monthly breakdown
        bars_per_month = 96 * 30
        n_bars_eq = len(result['equity_curve'])
        eq = result['equity_curve']
        n_months = max(1, DAYS // 30)
        print(f"\n  Monthly Equity Snapshots (every 3 months if >12):")
        step = 3 if n_months > 12 else 1
        for m in range(step, n_months + 1, step):
            idx = min(m * bars_per_month, n_bars_eq - 1)
            monthly_eq = eq[idx]
            monthly_ret = (monthly_eq / CAPITAL - 1.0) * 100
            print(f"    Month {m:2d}: ${monthly_eq:>12,.0f}  ({monthly_ret:+,.1f}%)")

    # ── Summary: What to actually expect ──
    flat_trades = result_flat['trades']
    if flat_trades:
        avg_pnl_pct = np.mean([t['pnl_pct'] for t in flat_trades])
        trades_per_day = result_flat['n_trades'] / DAYS
        daily_expected = avg_pnl_pct * trades_per_day
        monthly_expected_pct = daily_expected * 30
        annual_expected_pct = result_flat['total_return_pct'] * (365 / DAYS)
        annual_expected_usd = CAPITAL * annual_expected_pct / 100

        print(f"\n{'=' * 80}")
        print(f"REALISTIC EXPECTATION (flat sizing, ${CAPITAL:,.0f}, {SLOTS} slots, {DAYS}d)")
        print(f"{'=' * 80}")
        print(f"  Trades/day:        {trades_per_day:.1f}")
        print(f"  Avg return/trade:  {avg_pnl_pct:+.3f}% on margin")
        print(f"  Expected daily:    {daily_expected:+.3f}% → ${CAPITAL*daily_expected/100:+.2f}")
        print(f"  Expected monthly:  {monthly_expected_pct:+.1f}% → ${CAPITAL*monthly_expected_pct/100:+.0f}")
        print(f"  Expected annual:   {annual_expected_pct:+,.1f}% → ${annual_expected_usd:+,.0f}")
        print(f"  Max Drawdown:      {result_flat['max_dd_pct']:.1f}%")
        print(f"  Win Rate:          {result_flat['win_rate']:.1f}%")
        print(f"  Profit Factor:     {result_flat['profit_factor']:.2f}")
        print()
        print(f"  ⚠ LIVE DISCOUNT: Expect 40-60% of backtest Sharpe due to slippage,")
        print(f"    partial fills, latency. Halve the annual return for conservative estimate.")
        print(f"    Conservative annual: {annual_expected_pct*0.5:+,.1f}% → ${annual_expected_usd*0.5:+,.0f}")

    # ══════════════════════════════════════════════════════════════════
    #  MONTE CARLO SIMULATION (compound, trade-level resampling)
    # ══════════════════════════════════════════════════════════════════
    comp_trades = result_comp['trades']
    if comp_trades:
        monte_carlo_analysis(comp_trades, CAPITAL, n_sims=10_000, trades_per_year=len(comp_trades))

    print(f"\n{'=' * 80}")
    print("DONE")


# ═══════════════════════════════════════════════════════════════════════
#  Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════

def monte_carlo_analysis(
    trades: list[dict],
    initial_capital: float,
    n_sims: int = 10_000,
    trades_per_year: int | None = None,
):
    """Monte Carlo resampling of actual trade returns (fully vectorized).

    Method: Bootstrap resampling with replacement from observed trade PnL%.
    Each simulation draws `trades_per_year` trades, compounds them via
    cumulative product, then computes max drawdown and loss streaks.

    Fully vectorized with numpy — no Python loops over trades.
    Only the drawdown/streak computation uses numba for speed.

    Args:
        trades: List of trade dicts with 'pnl_pct' field.
        initial_capital: Starting balance ($).
        n_sims: Number of Monte Carlo paths (10,000 default).
        trades_per_year: Trades per simulation path. Defaults to len(trades).
    """
    pnl_pcts = np.array([t['pnl_pct'] for t in trades], dtype=np.float64)
    n_trades = trades_per_year or len(trades)

    print(f"\n{'=' * 80}")
    print(f"MONTE CARLO SIMULATION ({n_sims:,} paths, {n_trades} trades/year, compound)")
    print(f"{'=' * 80}")
    print(f"  Trade pool: {len(pnl_pcts)} observed trades")
    print(f"  Mean return: {np.mean(pnl_pcts):+.3f}%, Std: {np.std(pnl_pcts):.3f}%")
    print(f"  Skew: {_skew(pnl_pcts):.2f}, Kurtosis: {_kurtosis(pnl_pcts):.2f}")
    print(f"  Running {n_sims:,} simulations (vectorized)...\n")

    rng = np.random.default_rng(seed=42)

    # ── Vectorized core: (n_sims, n_trades) ──
    # Draw all random trade returns at once
    idx = rng.integers(0, len(pnl_pcts), size=(n_sims, n_trades))
    sampled_returns = pnl_pcts[idx]  # (n_sims, n_trades) PnL% values

    # Growth factors: each trade uses ~50% of equity (dynamic allocation)
    growth_factors = 1.0 + sampled_returns / 100.0 * 0.5  # (n_sims, n_trades)

    # Cumulative equity paths via cumprod
    cum_equity = initial_capital * np.cumprod(growth_factors, axis=1)  # (n_sims, n_trades)

    # Final balances
    final_balances = cum_equity[:, -1]

    # ── Max drawdown per path (vectorized) ──
    running_max = np.maximum.accumulate(cum_equity, axis=1)
    drawdowns = (running_max - cum_equity) / running_max
    max_drawdowns = np.max(drawdowns, axis=1)

    # ── Ruin count: equity < 50% of initial at any point ──
    ruin_mask = np.any(cum_equity < initial_capital * 0.5, axis=1)
    ruin_count = int(np.sum(ruin_mask))

    # ── Consecutive loss streaks (use numba for speed) ──
    is_loss = sampled_returns <= 0  # (n_sims, n_trades) bool
    max_consecutive_losses = _compute_max_streaks(is_loss, n_sims, n_trades)

    # ── Compute statistics ──
    final_returns = (final_balances / initial_capital - 1.0) * 100
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print(f"  {'─' * 65}")
    print(f"  FINAL BALANCE DISTRIBUTION (after {n_trades} trades)")
    print(f"  {'─' * 65}")
    for p in pctiles:
        bal = np.percentile(final_balances, p)
        ret = np.percentile(final_returns, p)
        print(f"    P{p:2d}:  ${bal:>14,.0f}  ({ret:+,.0f}%)")
    print(f"    Mean: ${np.mean(final_balances):>14,.0f}  ({np.mean(final_returns):+,.0f}%)")

    print(f"\n  {'─' * 65}")
    print(f"  MAX DRAWDOWN DISTRIBUTION")
    print(f"  {'─' * 65}")
    for p in [5, 25, 50, 75, 95]:
        dd = np.percentile(max_drawdowns, p) * 100
        print(f"    P{p:2d}:  {dd:.1f}%")
    print(f"    Mean: {np.mean(max_drawdowns)*100:.1f}%")

    print(f"\n  {'─' * 65}")
    print(f"  RISK METRICS")
    print(f"  {'─' * 65}")

    prob_profit = np.mean(final_returns > 0) * 100
    prob_2x = np.mean(final_balances >= initial_capital * 2) * 100
    prob_5x = np.mean(final_balances >= initial_capital * 5) * 100
    prob_10x = np.mean(final_balances >= initial_capital * 10) * 100
    prob_100x = np.mean(final_balances >= initial_capital * 100) * 100

    print(f"    P(profit):     {prob_profit:.1f}%")
    print(f"    P(2x capital): {prob_2x:.1f}%    (≥${initial_capital*2:,.0f})")
    print(f"    P(5x capital): {prob_5x:.1f}%    (≥${initial_capital*5:,.0f})")
    print(f"    P(10x):        {prob_10x:.1f}%    (≥${initial_capital*10:,.0f})")
    print(f"    P(100x):       {prob_100x:.1f}%   (≥${initial_capital*100:,.0f})")
    print(f"    P(ruin <50%):  {ruin_count/n_sims*100:.2f}%")

    print(f"\n  {'─' * 65}")
    print(f"  CONSECUTIVE LOSS STREAKS")
    print(f"  {'─' * 65}")
    for p in [50, 75, 90, 95, 99]:
        cl = np.percentile(max_consecutive_losses, p)
        print(f"    P{p:2d}:  {cl:.0f} trades")
    print(f"    Max:  {np.max(max_consecutive_losses):.0f} trades")

    # ── Confidence interval summary ──
    p5_bal = np.percentile(final_balances, 5)
    p50_bal = np.percentile(final_balances, 50)
    p95_bal = np.percentile(final_balances, 95)
    p5_dd = np.percentile(max_drawdowns, 95) * 100

    print(f"\n  {'=' * 65}")
    print(f"  90% CONFIDENCE INTERVAL (compound, {n_trades} trades/year)")
    print(f"  {'=' * 65}")
    print(f"    Lower (P5):   ${p5_bal:>12,.0f}  ({(p5_bal/initial_capital-1)*100:+,.0f}%)")
    print(f"    Median (P50): ${p50_bal:>12,.0f}  ({(p50_bal/initial_capital-1)*100:+,.0f}%)")
    print(f"    Upper (P95):  ${p95_bal:>12,.0f}  ({(p95_bal/initial_capital-1)*100:+,.0f}%)")
    print(f"    Worst DD (P95): {p5_dd:.1f}%")
    print(f"    Ruin probability: {ruin_count/n_sims*100:.2f}%")


@numba.jit(nopython=True, cache=True)
def _compute_max_streaks(is_loss: np.ndarray, n_sims: int, n_trades: int) -> np.ndarray:
    """Compute max consecutive loss streak per simulation path."""
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


def _skew(x: np.ndarray) -> float:
    """Compute skewness."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


if __name__ == '__main__':
    main()
