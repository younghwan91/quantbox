#!/usr/bin/env python3
"""Market-Cap Pool A/B Test — Maeoknam style (Top-30 by mcap) vs Curated.

Hypothesis:
  매억남 방식처럼 시총 Top-30을 무조건 모두 거래하면,
  개별 심볼 백테스트 Calmar 기준으로 큐레이션한 것보다 나을까?

  Key difference:
  - 현재 Top-20: 개별 Calmar > 1.0 기준으로 선별 (losers 제외)
  - 매억남 Top-30: 시총 순으로 필터 없이 전부 투입

Variants:
  A) Top-20 Curated (current v11.1 production)
  B) All-29 by Market Cap (Maeoknam style, no performance filter)
  C) Top-20 by Market Cap (시총 상위 20개만)
  D) Top-25 by Market Cap (시총 상위 25개만)

  Each variant tested with 5 slots (proven optimal).

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

# ═══════════════════════════════════════════════════════════════
# Market Cap ranking (approximate, 2026 Q1)
# This simulates Maeoknam's approach: pick top N coins by mcap
# BTCUSDT excluded as it's used as the index/regime filter
# ═══════════════════════════════════════════════════════════════
MCAP_RANKED = [
    'ETHUSDT',    # #2
    'BNBUSDT',    # #3
    'XRPUSDT',    # #4
    'SOLUSDT',    # #5
    'ADAUSDT',    # #6
    'DOGEUSDT',   # #7
    'TRXUSDT',    # #8
    'LINKUSDT',   # #9
    'AVAXUSDT',   # #10
    'SUIUSDT',    # #11
    'SHIBUSDT',   # #12
    'DOTUSDT',    # #13
    'BCHUSDT',    # #14
    'LTCUSDT',    # #15
    'UNIUSDT',    # #16
    'NEARUSDT',   # #17
    'APTUSDT',    # #18
    'PEPEUSDT',   # #19
    'ICPUSDT',    # #20
    'TONUSDT',    # #21
    'FILUSDT',    # #22
    'ATOMUSDT',   # #23
    'ARBUSDT',    # #24
    'OPUSDT',     # #25
    'INJUSDT',    # #26
    'HBARUSDT',   # #27
    'XLMUSDT',    # #28
    'FETUSDT',    # #29
    'AAVEUSDT',   # #30
]

# Current v11.1 production whitelist (Top-20 by Calmar)
TOP20_CURATED = [
    'FILUSDT', 'PEPEUSDT', 'XRPUSDT', 'XLMUSDT', 'ADAUSDT',
    'FETUSDT', 'ETHUSDT', 'ARBUSDT', 'SHIBUSDT', 'LINKUSDT',
    'ATOMUSDT', 'OPUSDT', 'DOTUSDT', 'NEARUSDT', 'AAVEUSDT',
    'BCHUSDT', 'HBARUSDT', 'SOLUSDT', 'ICPUSDT', 'BNBUSDT',
]


def load_all_data() -> dict:
    """Load all symbol data into memory."""
    btc_15m = pd.read_pickle(CACHE / 'BTCUSDT_15m_365d.pkl.gz')['Close'].values.astype(np.float64)
    data = {'btc_15m': btc_15m}

    btc_4h = CACHE / 'BTCUSDT_4h_365d.pkl.gz'
    if btc_4h.exists():
        data['btc_4h'] = pd.read_pickle(btc_4h)['Close'].values.astype(np.float64)

    for sym in MCAP_RANKED:
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
    """Run backtest for a single symbol."""
    close = data[sym]['close']
    r = _backtest_core(close, signals, COST, FUNDING)
    ret = float(r[1])
    dd = float(r[3])
    return {
        'symbol': sym,
        'return_pct': ret * 100,
        'max_dd_pct': dd * 100,
        'calmar': ret / dd if dd > 0 else 0.0,
        'sharpe': float(r[2]),
        'win_rate': float(r[4]),
        'n_trades': int(r[5]),
        'profit_factor': float(r[7]),
    }


def simulate_portfolio(
    all_signals: dict[str, np.ndarray],
    data: dict,
    symbol_list: list[str],
    max_concurrent: int,
    n_bars: int,
) -> dict:
    """Simulate multi-slot portfolio with capital allocation."""
    equity = np.ones(n_bars)
    positions = {}
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

            if sig_arr[i] == 0.0 and pos['prev_sig'] != 0.0:
                ret = curr_close[i] / curr_close[i - 1] - 1.0
                pos_pnl = pos['leverage_mag'] * ret
                pos_pnl -= abs(pos['leverage_mag']) * COST
                bar_pnl += pos_pnl / max_concurrent

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
                ret = curr_close[i] / curr_close[i - 1] - 1.0
                direction = 1.0 if pos['direction'] == 1 else -1.0
                pos_pnl = abs(pos['leverage_mag']) * direction * ret
                pos_pnl -= abs(pos['leverage_mag']) * FUNDING
                bar_pnl += pos_pnl / max_concurrent

            pos['prev_sig'] = sig_arr[i]

        for sym in closed:
            del positions[sym]

        # 2. New entries
        if len(positions) < max_concurrent:
            candidates = []
            for sym in symbol_list:
                if sym in positions or sym not in all_signals:
                    continue
                sig_arr = all_signals[sym]
                if sig_arr is None or i >= len(sig_arr):
                    continue
                curr_sig = sig_arr[i]
                prev_sig = sig_arr[i - 1] if i > 0 else 0.0
                if curr_sig != 0.0 and prev_sig == 0.0:
                    curr_close = data[sym]['close']
                    if i >= len(curr_close):
                        continue
                    move = abs(curr_close[i] / curr_close[max(0, i - V11['lookback'])] - 1.0)
                    candidates.append((sym, curr_sig, move, curr_close[i]))

            candidates.sort(key=lambda x: x[2], reverse=True)

            for sym, sig, move, price in candidates:
                if len(positions) >= max_concurrent:
                    break
                direction = 1 if sig > 0 else -1
                leverage_mag = abs(sig)
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
    print("MARKET-CAP POOL A/B TEST — Maeoknam Style vs Curated")
    print("=" * 80)

    print("\n▶ Loading data for all symbols...")
    data = load_all_data()
    available = [s for s in MCAP_RANKED if s in data]
    print(f"  Loaded: {len(available)} symbols (by market cap rank)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Per-Symbol Scan (show which mcap-ranked symbols are losers)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 1: Per-Symbol Profitability (Market Cap Order)")
    print("  Shows which top-mcap coins HURT the portfolio")
    print("=" * 80)

    all_signals = {}
    results = []

    for sym in available:
        signals = get_signals(data, sym)
        all_signals[sym] = signals
        r = backtest_symbol(data, sym, signals)
        results.append(r)

    # Print in market-cap order (not Calmar order) to highlight the difference
    print(f"\n{'Rank':>4} {'Symbol':<12} {'Return%':>10} {'MaxDD%':>8} {'Calmar':>8} "
          f"{'Sharpe':>8} {'WR%':>6} {'Trades':>7} {'PF':>6} {'Curated?':<10} {'Verdict'}")
    print("-" * 110)

    losers = []
    for i, sym in enumerate(available):
        r = next(x for x in results if x['symbol'] == sym)
        in_curated = "✓ TOP20" if sym in TOP20_CURATED else "  ---"
        verdict = "PROFIT" if r['return_pct'] > 0 and r['calmar'] > 1.0 else "☠ LOSER"
        if r['return_pct'] <= 0 or r['calmar'] <= 1.0:
            losers.append(sym)
        print(f"#{i+1:>3} {r['symbol']:<12} {r['return_pct']:>10.1f} {r['max_dd_pct']:>8.2f} "
              f"{r['calmar']:>8.2f} {r['sharpe']:>8.2f} {r['win_rate']:>6.1f} "
              f"{r['n_trades']:>7d} {r['profit_factor']:>6.2f} {in_curated:<10} {verdict}")

    print(f"\n  ☠ Losers in Top-30 mcap: {len(losers)} → {', '.join(losers)}")
    print(f"  These coins DRAG the Maeoknam-style pool down.")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Head-to-Head A/B Test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 2: Portfolio Simulation — Head-to-Head A/B")
    print("  All variants use 5 concurrent slots (proven optimal)")
    print("=" * 80)

    n_bars = min(len(data[sym]['close']) for sym in available if sym in data)

    # Define variants
    variants = {
        'A) Top-20 Curated (v11.1)': TOP20_CURATED,
        'B) All-29 McapRank (Maeoknam)': available,  # All 29 coins by mcap
        'C) Top-20 by Mcap': available[:20],
        'D) Top-25 by Mcap': available[:25],
        'E) All-29 minus losers': [s for s in available if s not in losers],
    }

    # Also test different slot counts for the Maeoknam-style pool
    slot_options = [5, 8, 10]

    print(f"\n{'Variant':<35} {'Slots':>5} {'#Syms':>5} {'Return%':>10} {'MaxDD%':>8} "
          f"{'Calmar':>8} {'Sharpe':>8} {'Trades':>7} {'WR%':>6} {'PF':>6}")
    print("-" * 115)

    all_results = []

    for variant_name, pool in variants.items():
        for slots in slot_options:
            r = simulate_portfolio(
                all_signals=all_signals,
                data=data,
                symbol_list=pool,
                max_concurrent=slots,
                n_bars=n_bars,
            )
            r['variant'] = variant_name
            r['slots'] = slots
            r['n_symbols'] = len(pool)
            all_results.append(r)

            tag = ""
            if variant_name == 'A) Top-20 Curated (v11.1)' and slots == 5:
                tag = " ← CURRENT"
            if variant_name == 'B) All-29 McapRank (Maeoknam)' and slots == 5:
                tag = " ← MAEOKNAM"

            print(f"{variant_name:<35} {slots:>5} {len(pool):>5} {r['return_pct']:>10.1f} "
                  f"{r['max_dd_pct']:>8.2f} {r['calmar']:>8.2f} {r['sharpe']:>8.2f} "
                  f"{r['n_trades']:>7} {r['win_rate']:>6.1f} {r['profit_factor']:>6.2f}{tag}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Direct Comparison Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 3: HEAD-TO-HEAD COMPARISON (5 slots)")
    print("=" * 80)

    # Get 5-slot results for key variants
    def get_5slot(name):
        return next(r for r in all_results if r['variant'] == name and r['slots'] == 5)

    curated = get_5slot('A) Top-20 Curated (v11.1)')
    maeoknam = get_5slot('B) All-29 McapRank (Maeoknam)')
    mcap20 = get_5slot('C) Top-20 by Mcap')

    print(f"\n{'Metric':<20} {'Curated Top-20':>18} {'Maeoknam All-29':>18} {'McapTop-20':>18}")
    print("-" * 78)

    metrics = [
        ('Return %', 'return_pct', '.1f'),
        ('Max DD %', 'max_dd_pct', '.2f'),
        ('Calmar', 'calmar', '.2f'),
        ('Sharpe', 'sharpe', '.2f'),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Trades', 'n_trades', 'd'),
        ('Profit Factor', 'profit_factor', '.2f'),
        ('# Symbols', 'n_symbols', 'd'),
    ]

    for label, key, fmt in metrics:
        v_c = curated[key]
        v_m = maeoknam[key]
        v_20 = mcap20[key]
        best = max(v_c, v_m, v_20) if key not in ('max_dd_pct',) else min(v_c, v_m, v_20)

        def mark(v, b):
            return " ★" if v == b else ""

        print(f"{label:<20} {v_c:>{16}{fmt}}{mark(v_c, best)} {v_m:>{16}{fmt}}{mark(v_m, best)} "
              f"{v_20:>{16}{fmt}}{mark(v_20, best)}")

    # Delta
    ret_delta = maeoknam['return_pct'] - curated['return_pct']
    cal_delta = maeoknam['calmar'] - curated['calmar']
    dd_delta = maeoknam['max_dd_pct'] - curated['max_dd_pct']

    print(f"\n  Maeoknam vs Curated Delta:")
    print(f"    Return:  {ret_delta:+.1f}%")
    print(f"    Calmar:  {cal_delta:+.2f}")
    print(f"    MaxDD:   {dd_delta:+.2f}%")

    # Final verdict
    print("\n" + "=" * 80)
    winner = "Maeoknam All-29" if maeoknam['calmar'] > curated['calmar'] else "Curated Top-20"
    print(f"★ VERDICT: {winner} WINS on risk-adjusted basis (Calmar)")

    if winner == "Maeoknam All-29":
        print(f"\n  → 매억남 방식(시총 Top-30 전부) = 우리 전략에도 적합!")
        print(f"  → Losers ({', '.join(losers)}) 포함해도 portfolio 수준에선 상쇄")
        print(f"  → 심볼 필터링 없이 시총 Top-30 전부 투입 권장")
        print(f"\n  CODE-READY:")
        print(f"    symbol_whitelist = {available}")
        print(f"    .env: RB_SYMBOL_WHITELIST={','.join(available)}")
    else:
        print(f"\n  → 큐레이션이 승리! 개별 심볼 성능 필터링이 중요")
        print(f"  → Losers ({', '.join(losers)}) 가 포트폴리오를 갉아먹음")
        print(f"  → 현재 Top-20 Curated 유지 권장")

        # Check if mcap-minus-losers is better
        minus_losers = get_5slot('E) All-29 minus losers')
        if minus_losers['calmar'] > curated['calmar']:
            syms = [s for s in available if s not in losers]
            print(f"\n  ⚡ BUT: McapAll minus losers ({len(syms)} syms) beats curated!")
            print(f"    Return: {minus_losers['return_pct']:.1f}%  Calmar: {minus_losers['calmar']:.2f}")
            print(f"    symbol_whitelist = {syms}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
