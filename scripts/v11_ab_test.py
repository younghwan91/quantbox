#!/usr/bin/env python3
"""v11 A/B Test — Mining winning_strategy.md for remaining alpha.

Based on v10 production params (threshold=0.032, mc=1, mh=20, sk=3).
Tests concepts from winning_strategy.md NOT previously tested:

  H1: BTC 4h EMA200 Macro Gate — "200 EMA as trend lifeline"
      Long only when BTC > 4h EMA200, short only when BTC < 4h EMA200.
      Different from btc_directional (8h momentum) — this is structural trend.

  H2: Trailing BE Stop — "50% TP + trail remainder"
      After position reaches 50% of target profit, activate break-even stop.
      Captures extended moves while protecting captured profit.

  H3: Confidence-Scaled Leverage — "deploy more when confirmed"
      conf≥3 → 2x leverage, conf≥4 → 2.5x, instead of flat 1.5x.
      (Not same as dynamic_target which was rejected — this scales SIZE not TARGET)

  H4: Dual Lookback Confluence — "Top-Down fractal analysis"
      Both fast (4-bar) and slow (12-bar) lookbacks must show moves in same direction.
      Reduces false signals at cost of fewer trades.

  H5: Volatility-Gated Entry — "deploy at statistically favorable coordinates"
      Only enter when ATR(20) / ATR(100) > 1.5 (volatility expanding).
      Mean reversion strongest during volatility expansion.

Previously rejected (DO NOT re-test):
  - RSI Divergence (v8.0 -18%), Vol Squeeze (v8.0 no effect)
  - Reversal Confirmation (v8.0 -76%), EMA Regime (v9.0 -4% to -9%)
  - Capital Preservation / Partial Exit (v4.4 -24%)
  - Confluence Gate (v9.0 -90%), lookback < 8 (v10.0 destroys perf)
  - dynamic_target (v9.0 — Fib discipline outperforms)
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
COST = 0.0004
FUNDING = 0.0001 / 32

# v10 Top-15 whitelist
TOP15 = [
    'FILUSDT', 'NEARUSDT', 'ADAUSDT', 'XLMUSDT', 'ETHUSDT',
    'BCHUSDT', 'ARBUSDT', 'UNIUSDT', 'PEPEUSDT', 'XRPUSDT',
    'FETUSDT', 'INJUSDT', 'SOLUSDT', 'AVAXUSDT', 'AAVEUSDT',
]

# v10 production baseline
V10 = dict(
    lookback=8, threshold=0.032, target_retrace=0.382, stop_mult=0.786,
    max_hold=20, stop_decay=0.0, btc_dir_period=0, btc_regime_period=96,
    asset_trend_period=192, min_confidence=1, base_leverage=1.5,
    conf_lev_step=0.0, max_leverage=1.5, rsi_period=14, vol_period=20,
    dynamic_target=False, target_c3=0.45, target_c4=0.5, target_c5=0.55,
    stoch_enabled=True, stoch_k=3, stoch_d=3,
    macd_enabled=True, macd_fast=12, macd_slow=26, macd_signal=14,
)


def load_data() -> dict:
    """Load all symbol data into memory."""
    btc_15m = pd.read_pickle(CACHE / 'BTCUSDT_15m_365d.pkl.gz')['Close'].values.astype(np.float64)
    data = {'btc_15m': btc_15m}

    # BTC 4h data for macro gate
    btc_4h_path = CACHE / 'BTCUSDT_4h_365d.pkl.gz'
    if btc_4h_path.exists():
        data['btc_4h'] = pd.read_pickle(btc_4h_path)['Close'].values.astype(np.float64)

    for sym in TOP15:
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


def compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA for an array."""
    ema = np.empty_like(close)
    ema[0] = close[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR for volatility gating."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    # SMA of TR
    atr = np.empty(n)
    atr[:period] = np.nan
    atr[period] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])
    return atr


def apply_btc_ema200_gate(signals: np.ndarray, btc_4h: np.ndarray, ratio: int = 16) -> np.ndarray:
    """H1: BTC 4h EMA200 macro gate.
    
    Long only when BTC > EMA200 (uptrend), short only when BTC < EMA200.
    ratio: 15m bars per 4h bar (16).
    """
    ema200 = compute_ema(btc_4h, 200)
    out = signals.copy()
    n15 = len(signals)
    n4h = len(btc_4h)

    for i in range(n15):
        if out[i] == 0:
            continue
        j = min(i // ratio, n4h - 1)
        btc_above_ema = btc_4h[j] > ema200[j]
        # Long only if BTC in uptrend, short only if BTC in downtrend
        if out[i] > 0 and not btc_above_ema:
            out[i] = 0
        elif out[i] < 0 and btc_above_ema:
            out[i] = 0
    return out


def apply_trailing_be_stop(signals: np.ndarray, close: np.ndarray,
                           target_retrace: float = 0.382,
                           be_activation: float = 0.5) -> np.ndarray:
    """H2: Trailing break-even stop after reaching partial target.
    
    Once price moves 50% toward target, set a break-even stop.
    If price returns to entry, close the trade.
    """
    out = signals.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        direction = 1 if out[i] > 0 else -1
        entry_price = close[i]
        
        # Find the lookback price to compute move
        lookback = 8  # hardcoded since we can't know exact lookback from signal
        lb_idx = max(0, i - lookback)
        lb_price = close[lb_idx]
        move = abs(entry_price / lb_price - 1.0)
        target_move = move * target_retrace
        be_threshold = target_move * be_activation  # 50% of target
        
        be_activated = False
        j = i + 1
        while j < n and out[j] != 0:
            pnl_pct = direction * (close[j] / entry_price - 1.0)
            if pnl_pct >= be_threshold:
                be_activated = True
            if be_activated and pnl_pct <= 0:
                # Break-even stop hit — zero out remaining position
                for k in range(j, n):
                    if out[k] == 0:
                        break
                    out[k] = 0
                break
            j += 1
        i = j + 1 if j < n else n
    return out


def apply_confidence_leverage(signals: np.ndarray, scale: str = 'moderate') -> np.ndarray:
    """H3: Scale leverage by confidence level encoded in signal magnitude.
    
    signals encode confidence: |sig| = base_leverage * (1 + confidence_bonus)
    Moderate: conf≥3 → 2x, conf≥4 → 2.5x
    Aggressive: conf≥3 → 2.5x, conf≥4 → 3x
    """
    out = signals.copy()
    for i in range(len(out)):
        if out[i] == 0:
            continue
        mag = abs(out[i])
        # Current v10 with MTF: mag is typically 1.5 * 2.0 = 3.0 for MTF-confirmed
        # Without MTF: mag = 1.5
        # We'll scale based on whether MTF was applied (mag > 2.0)
        if scale == 'moderate':
            if mag > 2.5:  # MTF confirmed, likely high confidence
                out[i] = np.sign(out[i]) * mag * 1.33  # 33% more leverage
            # else: keep as is
        elif scale == 'aggressive':
            if mag > 2.5:
                out[i] = np.sign(out[i]) * mag * 1.67  # 67% more leverage
    return out


def apply_dual_lookback(signals: np.ndarray, close: np.ndarray,
                        fast_lb: int = 4, slow_lb: int = 12,
                        threshold: float = 0.032) -> np.ndarray:
    """H4: Dual lookback confluence — both fast and slow must agree."""
    out = signals.copy()
    for i in range(slow_lb + 1, len(out)):
        if out[i] == 0:
            continue
        fast_move = close[i] / close[i - fast_lb] - 1.0
        slow_move = close[i] / close[i - slow_lb] - 1.0
        # Both must show threshold-level moves in same direction
        if not (abs(fast_move) > threshold * 0.5 and abs(slow_move) > threshold):
            out[i] = 0
            continue
        if np.sign(fast_move) != np.sign(slow_move):
            out[i] = 0
    return out


def apply_vol_expansion_gate(signals: np.ndarray, high: np.ndarray,
                              low: np.ndarray, close: np.ndarray,
                              fast_period: int = 20, slow_period: int = 100,
                              min_ratio: float = 1.3) -> np.ndarray:
    """H5: Only enter when volatility is expanding (ATR fast/slow > min_ratio).
    
    Mean reversion is strongest during vol expansion (sharp moves revert harder).
    """
    atr_fast = compute_atr(high, low, close, fast_period)
    atr_slow = compute_atr(high, low, close, slow_period)
    out = signals.copy()
    for i in range(slow_period + 1, len(out)):
        if out[i] == 0:
            continue
        if np.isnan(atr_fast[i]) or np.isnan(atr_slow[i]) or atr_slow[i] == 0:
            continue
        ratio = atr_fast[i] / atr_slow[i]
        if ratio < min_ratio:
            out[i] = 0
    return out


def get_signals(data: dict, sym: str, params: dict) -> np.ndarray:
    """Generate v10 signals with MTF for a symbol."""
    d = data[sym]
    close = d['close']
    n = len(close)
    btc = data['btc_15m']
    btc_n = btc[:n] if len(btc) >= n else np.concatenate([btc, np.full(n - len(btc), btc[-1])])

    signals = compute_signals_v9(
        close, d['high'], d['low'], d['volume'], d['taker'], btc_n,
        params['lookback'], params['threshold'], params['target_retrace'],
        params['stop_mult'], params['max_hold'], params['stop_decay'],
        params['btc_dir_period'], params['btc_regime_period'],
        params['asset_trend_period'], params['min_confidence'],
        params['base_leverage'], params['conf_lev_step'],
        params['max_leverage'], params['rsi_period'], params['vol_period'],
        params['dynamic_target'], params['target_c3'], params['target_c4'],
        params['target_c5'], params['stoch_enabled'], params['stoch_k'],
        params['stoch_d'], params['macd_enabled'], params['macd_fast'],
        params['macd_slow'], params['macd_signal'],
    )

    if 'close_4h' in d:
        signals = apply_mtf_confirmation(signals, d['close_4h'], 12)
        signals = apply_mtf_leverage(signals, 2.0)

    return signals


def backtest_symbol(data: dict, sym: str, signals: np.ndarray) -> tuple:
    """Run backtest for a symbol with given signals."""
    close = data[sym]['close']
    r = _backtest_core(close, signals, COST, FUNDING)
    ret, dd, nt, wr = float(r[1]), float(r[3]), int(r[5]), float(r[4])
    cal = ret / dd if dd > 0 else 0.0
    return ret * 100, dd * 100, cal, nt, wr


def run_hypothesis(data: dict, name: str, transform_fn, params: dict = None) -> dict:
    """Run a hypothesis across Top-15 with optional signal transform."""
    if params is None:
        params = V10
    results = []
    for sym in TOP15:
        if sym not in data:
            continue
        signals = get_signals(data, sym, params)
        if transform_fn is not None:
            signals = transform_fn(signals, data, sym)
        ret, dd, cal, nt, wr = backtest_symbol(data, sym, signals)
        results.append((sym, ret, dd, cal, nt, wr))

    total_ret = sum(r[1] for r in results)
    total_trades = sum(r[4] for r in results)
    avg_calmar = np.mean([r[3] for r in results]) if results else 0
    profitable = sum(1 for r in results if r[1] > 0)
    losers = sum(1 for r in results if r[1] <= 0)
    worst_dd = max(r[2] for r in results) if results else 0

    return {
        'name': name,
        'total_ret': total_ret,
        'trades': total_trades,
        'avg_calmar': avg_calmar,
        'profitable': profitable,
        'losers': losers,
        'worst_dd': worst_dd,
        'details': results,
    }


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data) - 2} symbols + BTC\n")

    # ── Baseline: v10 production ──
    baseline = run_hypothesis(data, "v10 Baseline", None)
    print(f"{'='*80}")
    print(f"BASELINE (v10 production): Ret={baseline['total_ret']:.1f}% | "
          f"Trades={baseline['trades']} | Calmar={baseline['avg_calmar']:.2f} | "
          f"Profitable={baseline['profitable']}/{baseline['profitable']+baseline['losers']}")
    print(f"{'='*80}\n")

    hypotheses = []

    # ── H1: BTC 4h EMA200 Macro Gate ──
    if 'btc_4h' in data:
        for label, ema_period in [('EMA100', 100), ('EMA200', 200), ('EMA50', 50)]:
            def make_h1(p):
                def tf(signals, d, sym):
                    return apply_btc_ema200_gate(signals, d['btc_4h'], ratio=16)
                return tf
            # Need to adjust ema period in the function — use closure
            def h1_transform(ema_p):
                def tf(signals, d, sym):
                    ema = compute_ema(d['btc_4h'], ema_p)
                    out = signals.copy()
                    n15 = len(signals)
                    n4h = len(d['btc_4h'])
                    for i in range(n15):
                        if out[i] == 0:
                            continue
                        j = min(i // 16, n4h - 1)
                        btc_above = d['btc_4h'][j] > ema[j]
                        if out[i] > 0 and not btc_above:
                            out[i] = 0
                        elif out[i] < 0 and btc_above:
                            out[i] = 0
                    return out
                return tf

            r = run_hypothesis(data, f"H1: BTC {label} Gate", h1_transform(int(label.replace('EMA', ''))))
            hypotheses.append(r)
    else:
        print("WARNING: No BTC 4h data found, skipping H1\n")

    # ── H2: Trailing Break-Even Stop ──
    for be_act in [0.3, 0.5, 0.7]:
        def h2_transform(act):
            def tf(signals, d, sym):
                return apply_trailing_be_stop(signals, d[sym]['close'],
                                              target_retrace=0.382, be_activation=act)
            return tf
        r = run_hypothesis(data, f"H2: Trailing BE (act={be_act})", h2_transform(be_act))
        hypotheses.append(r)

    # ── H3: Confidence-Scaled Leverage ──
    for scale in ['moderate', 'aggressive']:
        def h3_transform(s):
            def tf(signals, d, sym):
                return apply_confidence_leverage(signals, scale=s)
            return tf
        r = run_hypothesis(data, f"H3: Conf-Scaled Lev ({scale})", h3_transform(scale))
        hypotheses.append(r)

    # ── H4: Dual Lookback Confluence ──
    for fast_lb, slow_lb in [(4, 12), (4, 16), (6, 12)]:
        def h4_transform(f, s):
            def tf(signals, d, sym):
                return apply_dual_lookback(signals, d[sym]['close'], fast_lb=f, slow_lb=s,
                                          threshold=0.032)
            return tf
        r = run_hypothesis(data, f"H4: Dual LB ({fast_lb}/{slow_lb})", h4_transform(fast_lb, slow_lb))
        hypotheses.append(r)

    # ── H5: Volatility Expansion Gate ──
    for min_ratio in [1.2, 1.3, 1.5]:
        def h5_transform(mr):
            def tf(signals, d, sym):
                return apply_vol_expansion_gate(signals, d[sym]['high'], d[sym]['low'],
                                                d[sym]['close'], min_ratio=mr)
            return tf
        r = run_hypothesis(data, f"H5: Vol Gate (ratio>{min_ratio})", h5_transform(min_ratio))
        hypotheses.append(r)

    # ── Print results ──
    print(f"\n{'='*80}")
    print(f"{'Hypothesis':<35} {'Ret%':>8} {'Trades':>7} {'Calmar':>8} {'P/L':>6} {'ΔRet':>8} {'ΔCal':>8}")
    print(f"{'='*80}")

    base_ret = baseline['total_ret']
    base_cal = baseline['avg_calmar']

    for h in hypotheses:
        d_ret = h['total_ret'] - base_ret
        d_cal = h['avg_calmar'] - base_cal
        flag = " ★" if d_ret > 0 and d_cal > 0 else " ✗" if d_ret < 0 and d_cal < 0 else ""
        print(f"{h['name']:<35} {h['total_ret']:>7.1f}% {h['trades']:>7} "
              f"{h['avg_calmar']:>8.2f} {h['profitable']:>2}/{h['profitable']+h['losers']:>2} "
              f"{d_ret:>+7.1f}% {d_cal:>+7.2f}{flag}")

    # ── Identify winners ──
    winners = [h for h in hypotheses if h['total_ret'] > base_ret and h['avg_calmar'] > base_cal]
    if winners:
        print(f"\n{'='*80}")
        print("WINNERS (beat baseline on BOTH Ret AND Calmar):")
        for w in sorted(winners, key=lambda x: x['avg_calmar'], reverse=True):
            print(f"  {w['name']}: Ret={w['total_ret']:.1f}% (+{w['total_ret']-base_ret:.1f}%), "
                  f"Calmar={w['avg_calmar']:.2f} (+{w['avg_calmar']-base_cal:.2f}), "
                  f"Trades={w['trades']}, {w['profitable']}/{w['profitable']+w['losers']} profitable")
            # Print per-symbol details for winners
            print(f"    {'Symbol':<12} {'Ret%':>8} {'DD%':>8} {'Calmar':>8} {'Trades':>7}")
            for sym, ret, dd, cal, nt, wr in w['details']:
                flag = "✓" if ret > 0 else "✗"
                print(f"    {sym:<12} {ret:>7.1f}% {dd:>7.1f}% {cal:>8.2f} {nt:>7} {flag}")
    else:
        print("\nNo clear winners found. v10 baseline is already optimal.")

    # ── Strictly better (same/more trades, higher returns, same/lower DD) ──
    strict = [h for h in hypotheses
              if h['total_ret'] > base_ret * 1.05  # 5%+ better
              and h['losers'] <= baseline['losers']]
    if strict:
        print(f"\nSTRICTLY BETTER (≥5% more ret, ≤ losers):")
        for s in sorted(strict, key=lambda x: x['total_ret'], reverse=True):
            print(f"  {s['name']}: Ret={s['total_ret']:.1f}%, Losers={s['losers']}")


if __name__ == '__main__':
    main()
