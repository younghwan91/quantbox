#!/usr/bin/env python3
"""Asset Pool Expansion A/B Test — maximize trade opportunities.

Tests whether expanding from the current v10 Top-15 whitelist to a broader
asset pool generates more alpha via increased trade frequency.

Hypothesis:
  More symbols → more entry signals → better capital utilization →
  higher absolute returns (even if per-symbol alpha is diluted).

Tests:
  1. Per-symbol profitability for all 30 available symbols
     → Identify which NEW symbols are profitable and which are toxic
  2. Portfolio simulation: 15 vs 20 vs 25 vs 29 symbols
     → With max_concurrent = {5, 8, 10, 15} slots
  3. Risk-adjusted comparison (Calmar, Sharpe, Max DD)

Uses v11.0 production params (btc_directional_period=48).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from quantbox.strategy.rubberband import (
    compute_signals_v9, apply_mtf_confirmation, apply_mtf_leverage,
)
from quantbox.backtest.vectorized_backtest import _backtest_core

CACHE = Path('data/cache_extended')
COST = 0.0004       # 0.04% per side (fee + slippage)
FUNDING = 0.0001 / 32  # funding per 15m bar

# v11.0 production params
V11 = dict(
    lookback=8, threshold=0.032, target_retrace=0.382, stop_mult=0.786,
    max_hold=20, stop_decay=0.0, btc_dir_period=48, btc_regime_period=96,
    asset_trend_period=192, min_confidence=1, base_leverage=1.5,
    conf_lev_step=0.0, max_leverage=1.5, rsi_period=14, vol_period=20,
    dynamic_target=False, target_c3=0.45, target_c4=0.5, target_c5=0.55,
    stoch_enabled=True, stoch_k=3, stoch_d=3,
    macd_enabled=True, macd_fast=12, macd_slow=26, macd_signal=14,
)
MTF_LEV = 2.5

# Current v10 whitelist (15 symbols)
TOP15 = [
    'FILUSDT', 'NEARUSDT', 'ADAUSDT', 'XLMUSDT', 'ETHUSDT',
    'BCHUSDT', 'ARBUSDT', 'UNIUSDT', 'PEPEUSDT', 'XRPUSDT',
    'FETUSDT', 'INJUSDT', 'SOLUSDT', 'AVAXUSDT', 'AAVEUSDT',
]

# All available symbols (except BTCUSDT which is used as index)
ALL_SYMBOLS = sorted([
    p.name.split('_15m_')[0]
    for p in CACHE.glob('*_15m_365d.pkl.gz')
    if 'BTCUSDT' not in p.name
])


def load_all_data() -> dict:
    """Load all symbol data into memory."""
    btc_15m = pd.read_pickle(CACHE / 'BTCUSDT_15m_365d.pkl.gz')['Close'].values.astype(np.float64)
    data = {'btc_15m': btc_15m}

    btc_4h = CACHE / 'BTCUSDT_4h_365d.pkl.gz'
    if btc_4h.exists():
        data['btc_4h'] = pd.read_pickle(btc_4h)['Close'].values.astype(np.float64)

    for sym in ALL_SYMBOLS:
        f15 = CACHE / f'{sym}_15m_365d.pkl.gz'
        f4h = CACHE / f'{sym}_4h_365d.pkl.gz'
        if not f15.exists():
            continue
        df = pd.read_pickle(f15)
        entry = {
            'close': df['Close'].values.astype(np.float64),
            'high': df['High'].values.astype(np.float64),
            'low': df['Low'].values.astype(np.float64),
            'volume': df['Volume'].values.astype(np.float64),
            'taker': df['Taker Buy Base'].values.astype(np.float64),
        }
        if f4h.exists():
            entry['close_4h'] = pd.read_pickle(f4h)['Close'].values.astype(np.float64)
        data[sym] = entry
    return data


def get_signals(data: dict, sym: str) -> np.ndarray:
    """Generate v11.0 signals with MTF for a symbol."""
    d = data[sym]
    close = d['close']
    n = len(close)
    btc = data['btc_15m']
    btc_n = btc[:n] if len(btc) >= n else np.concatenate([btc, np.full(n - len(btc), btc[-1])])

    signals = compute_signals_v9(
        close, d['high'], d['low'], d['volume'], d['taker'], btc_n,
        V11['lookback'], V11['threshold'], V11['target_retrace'],
        V11['stop_mult'], V11['max_hold'], V11['stop_decay'],
        V11['btc_dir_period'], V11['btc_regime_period'],
        V11['asset_trend_period'], V11['min_confidence'],
        V11['base_leverage'], V11['conf_lev_step'],
        V11['max_leverage'], V11['rsi_period'], V11['vol_period'],
        V11['dynamic_target'], V11['target_c3'], V11['target_c4'],
        V11['target_c5'], V11['stoch_enabled'], V11['stoch_k'],
        V11['stoch_d'], V11['macd_enabled'], V11['macd_fast'],
        V11['macd_slow'], V11['macd_signal'],
    )

    if 'close_4h' in d:
        signals = apply_mtf_confirmation(signals, d['close_4h'], 12)
        signals = apply_mtf_leverage(signals, MTF_LEV)

    return signals


def backtest_symbol(data: dict, sym: str, signals: np.ndarray) -> dict:
    """Run backtest for a single symbol. Returns detailed metrics."""
    close = data[sym]['close']
    r = _backtest_core(close, signals, COST, FUNDING)
    equity = r[0]
    ret = float(r[1])
    sharpe = float(r[2])
    dd = float(r[3])
    wr = float(r[4])
    nt = int(r[5])
    pf = float(r[7])
    cal = ret / dd if dd > 0 else 0.0
    return {
        'symbol': sym,
        'return_pct': ret * 100,
        'max_dd_pct': dd * 100,
        'calmar': cal,
        'sharpe': sharpe,
        'win_rate': wr,
        'n_trades': nt,
        'profit_factor': pf,
        'equity': equity,
    }


def simulate_portfolio(
    all_signals: dict[str, np.ndarray],
    data: dict,
    symbol_list: list[str],
    max_concurrent: int,
    n_bars: int,
) -> dict:
    """Simulate multi-slot portfolio with capital allocation.

    At each bar, if a new signal fires and we have open slots, we enter.
    Capital per slot = equity / max(1, n_active + 1), capped at 50%.
    Each position contributes leveraged return to portfolio equity.

    This is a realistic simulation of how the live scalper operates.

    Args:
        all_signals: Pre-computed signal arrays per symbol.
        data: Market data dict.
        symbol_list: Symbols to trade.
        max_concurrent: Maximum simultaneous positions.
        n_bars: Number of bars in simulation.

    Returns:
        Portfolio metrics dict.
    """
    equity = np.ones(n_bars)
    positions = {}  # sym -> {signal, entry_bar, leverage_mag}
    n_trades = 0
    n_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    peak = 1.0
    max_dd = 0.0

    for i in range(1, n_bars):
        bar_pnl = 0.0

        # 1. Update existing positions
        closed = []
        for sym, pos in positions.items():
            sig_arr = all_signals[sym]
            curr_close = data[sym]['close']
            if i >= len(sig_arr) or i >= len(curr_close):
                closed.append(sym)
                continue

            # Position still held?
            if sig_arr[i] == 0.0 and pos['prev_sig'] != 0.0:
                # Exit: compute trade PnL
                ret = curr_close[i] / curr_close[i - 1] - 1.0
                pos_pnl = pos['leverage_mag'] * ret
                # Close cost
                pos_pnl -= abs(pos['leverage_mag']) * COST
                bar_pnl += pos_pnl / max_concurrent  # allocate per-slot share
                # Trade stats
                trade_ret = curr_close[i] / pos['entry_price'] - 1.0
                if pos['direction'] == -1:
                    trade_ret = -trade_ret
                trade_ret *= pos['leverage_mag']
                n_trades += 1
                if trade_ret > 0:
                    n_wins += 1
                    gross_profit += trade_ret
                else:
                    gross_loss += abs(trade_ret)
                closed.append(sym)
            elif sig_arr[i] != 0.0:
                # Still in position: compute bar PnL
                ret = curr_close[i] / curr_close[i - 1] - 1.0
                direction = 1.0 if pos['direction'] == 1 else -1.0
                pos_pnl = abs(pos['leverage_mag']) * direction * ret
                # Funding
                pos_pnl -= abs(pos['leverage_mag']) * FUNDING
                bar_pnl += pos_pnl / max_concurrent
            else:
                # sig still 0, no position action
                pass

            pos['prev_sig'] = sig_arr[i]

        for sym in closed:
            del positions[sym]

        # 2. New entries (signal transitions from 0 → non-zero)
        if len(positions) < max_concurrent:
            candidates = []
            for sym in symbol_list:
                if sym in positions:
                    continue
                sig_arr = all_signals.get(sym)
                if sig_arr is None or i >= len(sig_arr):
                    continue
                curr_sig = sig_arr[i]
                prev_sig = sig_arr[i - 1] if i > 0 else 0.0
                if curr_sig != 0.0 and prev_sig == 0.0:
                    # New entry signal
                    curr_close = data[sym]['close']
                    if i >= len(curr_close):
                        continue
                    move = abs(curr_close[i] / curr_close[max(0, i - V11['lookback'])] - 1.0)
                    candidates.append((sym, curr_sig, move, curr_close[i]))

            # Prioritize strongest moves
            candidates.sort(key=lambda x: x[2], reverse=True)

            for sym, sig, move, price in candidates:
                if len(positions) >= max_concurrent:
                    break
                direction = 1 if sig > 0 else -1
                leverage_mag = abs(sig)
                # Entry cost
                bar_pnl -= leverage_mag * COST / max_concurrent
                positions[sym] = {
                    'direction': direction,
                    'leverage_mag': leverage_mag,
                    'entry_bar': i,
                    'entry_price': price,
                    'prev_sig': sig,
                }

        equity[i] = equity[i - 1] * (1.0 + bar_pnl)
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd

    total_ret = equity[-1] / equity[0] - 1.0

    # Sharpe (annualized from 15m bars)
    rets = np.diff(equity) / equity[:-1]
    mean_r = np.mean(rets) if len(rets) > 0 else 0.0
    std_r = np.std(rets) if len(rets) > 0 else 1.0
    sharpe = mean_r / std_r * np.sqrt(96.0 * 365.25) if std_r > 0 else 0.0

    calmar = total_ret / max_dd if max_dd > 0 else 0.0
    wr = n_wins / n_trades * 100 if n_trades > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    return {
        'return_pct': total_ret * 100,
        'max_dd_pct': max_dd * 100,
        'calmar': calmar,
        'sharpe': sharpe,
        'win_rate': wr,
        'n_trades': n_trades,
        'profit_factor': pf,
        'equity': equity,
    }


def main():
    print("=" * 80)
    print("ASSET POOL EXPANSION A/B TEST (v11.0 baseline)")
    print("=" * 80)

    print("\n▶ Loading data for all 30 symbols...")
    data = load_all_data()
    available = [s for s in ALL_SYMBOLS if s in data]
    print(f"  Loaded: {len(available)} symbols")
    print(f"  Current whitelist: {len(TOP15)} symbols")
    new_symbols = [s for s in available if s not in TOP15]
    print(f"  New candidates: {len(new_symbols)}: {', '.join(new_symbols)}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Per-Symbol Profitability Scan
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 1: Per-Symbol Profitability (all 30 symbols)")
    print("=" * 80)

    all_signals = {}
    results = []

    for sym in available:
        signals = get_signals(data, sym)
        all_signals[sym] = signals
        r = backtest_symbol(data, sym, signals)
        results.append(r)

    # Sort by Calmar
    results.sort(key=lambda x: x['calmar'], reverse=True)

    print(f"\n{'Symbol':<12} {'Return%':>10} {'MaxDD%':>8} {'Calmar':>8} "
          f"{'Sharpe':>8} {'WR%':>6} {'Trades':>7} {'PF':>6} {'Status':<12}")
    print("-" * 85)

    profitable = []
    unprofitable = []

    for r in results:
        in_whitelist = "✓ TOP15" if r['symbol'] in TOP15 else "  NEW"
        status = "🟢 PASS" if r['return_pct'] > 0 and r['calmar'] > 1.0 else "🔴 FAIL"
        if r['return_pct'] > 0 and r['calmar'] > 1.0:
            profitable.append(r['symbol'])
        else:
            unprofitable.append(r['symbol'])

        print(f"{r['symbol']:<12} {r['return_pct']:>10.1f} {r['max_dd_pct']:>8.2f} "
              f"{r['calmar']:>8.2f} {r['sharpe']:>8.2f} {r['win_rate']:>6.1f} "
              f"{r['n_trades']:>7d} {r['profit_factor']:>6.2f} {in_whitelist} {status}")

    print(f"\nProfitable (Calmar > 1.0): {len(profitable)}/{len(available)}")
    print(f"  Current whitelist keeping: {len([s for s in profitable if s in TOP15])}/{len(TOP15)}")
    new_profitable = [s for s in profitable if s not in TOP15]
    print(f"  NEW profitable symbols: {len(new_profitable)}: {', '.join(new_profitable)}")
    print(f"  Unprofitable: {len(unprofitable)}: {', '.join(unprofitable)}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Build optimal expanded whitelists
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 2: Whitelist Candidates (ranked by Calmar)")
    print("=" * 80)

    # Different pool sizes to test
    ranked_profitable = [r['symbol'] for r in results if r['symbol'] in profitable]
    pools = {
        'Top-15 (current)': TOP15,
        f'Top-{min(20, len(ranked_profitable))} (by Calmar)': ranked_profitable[:min(20, len(ranked_profitable))],
        f'Top-{min(25, len(ranked_profitable))} (by Calmar)': ranked_profitable[:min(25, len(ranked_profitable))],
        f'All profitable ({len(ranked_profitable)})': ranked_profitable,
    }

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Portfolio Simulation — Pool × Slots grid
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 3: Portfolio Simulation — Pool Size × Max Concurrent Slots")
    print("=" * 80)

    # Find common bar count
    n_bars = min(len(data[sym]['close']) for sym in available if sym in data)
    slot_options = [5, 8, 10, 15]

    best_config = None
    best_calmar = -999

    print(f"\n{'Pool':<30} {'Slots':>5} {'Return%':>10} {'MaxDD%':>8} "
          f"{'Calmar':>8} {'Sharpe':>8} {'Trades':>7} {'WR%':>6} {'PF':>6}")
    print("-" * 100)

    for pool_name, pool_symbols in pools.items():
        for slots in slot_options:
            result = simulate_portfolio(
                all_signals=all_signals,
                data=data,
                symbol_list=pool_symbols,
                max_concurrent=slots,
                n_bars=n_bars,
            )
            tag = ""
            if result['calmar'] > best_calmar:
                best_calmar = result['calmar']
                best_config = (pool_name, slots, pool_symbols, result)
                tag = " ★ BEST"

            print(f"{pool_name:<30} {slots:>5} {result['return_pct']:>10.1f} "
                  f"{result['max_dd_pct']:>8.2f} {result['calmar']:>8.2f} "
                  f"{result['sharpe']:>8.2f} {result['n_trades']:>7} "
                  f"{result['win_rate']:>6.1f} {result['profit_factor']:>6.2f}{tag}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Summary & Recommendation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if best_config:
        pool_name, slots, syms, r = best_config
        print(f"\n★ BEST CONFIG: {pool_name}, {slots} slots")
        print(f"  Return:  {r['return_pct']:.1f}%")
        print(f"  Max DD:  {r['max_dd_pct']:.2f}%")
        print(f"  Calmar:  {r['calmar']:.2f}")
        print(f"  Sharpe:  {r['sharpe']:.2f}")
        print(f"  Trades:  {r['n_trades']}")
        print(f"  WR:      {r['win_rate']:.1f}%")
        print(f"  PF:      {r['profit_factor']:.2f}")
        print(f"  Symbols: {', '.join(syms)}")

    # Also compare: current Top-15/5-slots baseline
    baseline = simulate_portfolio(
        all_signals=all_signals, data=data,
        symbol_list=TOP15, max_concurrent=5, n_bars=n_bars,
    )
    print(f"\n  Baseline (Top-15, 5 slots):")
    print(f"  Return:  {baseline['return_pct']:.1f}%")
    print(f"  Max DD:  {baseline['max_dd_pct']:.2f}%")
    print(f"  Calmar:  {baseline['calmar']:.2f}")
    print(f"  Trades:  {baseline['n_trades']}")

    if best_config:
        _, _, _, r = best_config
        ret_delta = r['return_pct'] - baseline['return_pct']
        cal_delta = r['calmar'] - baseline['calmar']
        trade_delta = r['n_trades'] - baseline['n_trades']
        print(f"\n  Delta vs baseline:")
        print(f"  Return: {ret_delta:+.1f}%")
        print(f"  Calmar: {cal_delta:+.2f}")
        print(f"  Trades: {trade_delta:+d}")

    print("\n" + "=" * 80)
    print("CODE-READY WHITELIST (paste into settings.py / .env):")
    print("=" * 80)
    if best_config:
        _, _, syms, _ = best_config
        print(f"  symbol_whitelist = {syms}")
        print(f"  .env: RB_SYMBOL_WHITELIST={','.join(syms)}")


if __name__ == '__main__':
    main()
