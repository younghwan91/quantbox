#!/usr/bin/env python3
"""Maeoknam Strategy Gap A/B Test — 4 biggest differences from winning_strategy.md.

Based on v10.2 production params (threshold=0.032, mc=1, mh=20, sk=3, mtf_lev=2.5).
Tests the 4 biggest gaps between QuantBox and the Maeoknam championship strategy:

  GAP1: Fractional Entry (Pilot 2% + Main 8%)
        Maeoknam enters 2% to test the waters, then 8% on confirmation.
        Implementation: 20% leverage for first 3 bars, then full leverage if
        position is still alive (not stopped out).

  GAP2: Pyramiding on Confirmed Moves
        Maeoknam adds to winners after Wave 3/C confirmation.
        Implementation: When unrealized PnL > 30% of target, add 50% more size.
        Trail stop to break-even after adding.

  GAP3: Trail Profit Runners (50% TP + Trail Remainder)
        Maeoknam takes 50% profit at target, trails the rest until exhaustion.
        Implementation: At target, reduce signal to 50%, then trail until
        a reversal to break-even or max_hold * 2.

  GAP4: BTC Macro Regime (Re-enable btc_directional_period)
        Maeoknam treats BTC as mandatory market index.
        Implementation: Re-enable btc_directional_period={16,32,48} to see
        if Risk-Off trades (sympathy pullback) add alpha.

Previously tested & rejected (DO NOT re-test):
  - BTC EMA200 macro gate (v11 H1 — killed too many trades)
  - BE stop without partial TP (v11 H2 — no consistent winner)
  - Confidence-scaled leverage (v11 H3 — marginal)
  - Capital preservation / partial exit at target (v4.4 — -24%)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
from quantbox.strategy.rubberband import (
    compute_signals_v9, apply_mtf_confirmation, apply_mtf_leverage,
)
from quantbox.backtest.vectorized_backtest import _backtest_core, _backtest_core_partial

CACHE = Path('data/cache_extended')
COST = 0.0004
FUNDING = 0.0001 / 32

# v10 Top-15 whitelist
TOP15 = [
    'FILUSDT', 'NEARUSDT', 'ADAUSDT', 'XLMUSDT', 'ETHUSDT',
    'BCHUSDT', 'ARBUSDT', 'UNIUSDT', 'PEPEUSDT', 'XRPUSDT',
    'FETUSDT', 'INJUSDT', 'SOLUSDT', 'AVAXUSDT', 'AAVEUSDT',
]

# v10.2 production baseline (mtf_leverage_mult=2.5)
V10 = dict(
    lookback=8, threshold=0.032, target_retrace=0.382, stop_mult=0.786,
    max_hold=20, stop_decay=0.0, btc_dir_period=0, btc_regime_period=96,
    asset_trend_period=192, min_confidence=1, base_leverage=1.5,
    conf_lev_step=0.0, max_leverage=1.5, rsi_period=14, vol_period=20,
    dynamic_target=False, target_c3=0.45, target_c4=0.5, target_c5=0.55,
    stoch_enabled=True, stoch_k=3, stoch_d=3,
    macd_enabled=True, macd_fast=12, macd_slow=26, macd_signal=14,
)

MTF_LEV = 2.5  # v10.1 production


def load_data() -> dict:
    """Load all symbol data into memory."""
    btc_15m = pd.read_pickle(CACHE / 'BTCUSDT_15m_365d.pkl.gz')['Close'].values.astype(np.float64)
    data = {'btc_15m': btc_15m}

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


def get_signals(data: dict, sym: str, params: dict = None,
                mtf_lev: float = MTF_LEV) -> np.ndarray:
    """Generate v10.2 signals with MTF for a symbol."""
    if params is None:
        params = V10
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
        signals = apply_mtf_leverage(signals, mtf_lev)

    return signals


def backtest_symbol(data: dict, sym: str, signals: np.ndarray,
                    use_partial: bool = False) -> tuple:
    """Run backtest for a symbol with given signals."""
    close = data[sym]['close']
    if use_partial:
        r = _backtest_core_partial(close, signals, COST, FUNDING)
    else:
        r = _backtest_core(close, signals, COST, FUNDING)
    ret, dd, nt, wr = float(r[1]), float(r[3]), int(r[5]), float(r[4])
    cal = ret / dd if dd > 0 else 0.0
    return ret * 100, dd * 100, cal, nt, wr


# ──────────────── GAP 1: Fractional Entry ────────────────
def apply_fractional_entry(signals: np.ndarray, close: np.ndarray,
                           pilot_fraction: float = 0.2,
                           confirm_bars: int = 3) -> np.ndarray:
    """Maeoknam-style pilot entry: small initial size, scale up if trade lives.

    Enter with pilot_fraction (20%) of full leverage initially.
    If trade survives confirm_bars bars and is profitable, scale up to full.
    If trade is stopped out during pilot phase, loss is 80% smaller.

    Args:
        signals: Full signals from compute_signals_v9 + MTF
        close: Price array
        pilot_fraction: Fraction of full leverage for pilot (0.2 = 20%)
        confirm_bars: Bars to wait before scaling up
    """
    out = signals.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue

        # Trade opened at bar i
        full_mag = out[i]
        direction = 1.0 if full_mag > 0 else -1.0
        entry_price = close[i]

        # Pilot phase: reduce to pilot_fraction of full size
        out[i] = full_mag * pilot_fraction

        j = i + 1
        scaled_up = False
        while j < n and out[j] != 0:
            if not scaled_up and (j - i) >= confirm_bars:
                # Check if we're profitable
                pnl_pct = direction * (close[j] / entry_price - 1.0)
                if pnl_pct > 0:
                    # Scale up to full size
                    out[j] = full_mag
                    scaled_up = True
                else:
                    # Not profitable after confirm_bars, keep pilot size
                    # (will likely get stopped out soon at reduced loss)
                    out[j] = full_mag * pilot_fraction
            elif not scaled_up:
                out[j] = full_mag * pilot_fraction
            # Else: already scaled up, keep full signal
            j += 1

        i = j
    return out


# ──────────────── GAP 2: Pyramiding ────────────────
def apply_pyramiding(signals: np.ndarray, close: np.ndarray,
                     add_threshold: float = 0.3,
                     pyramid_mult: float = 0.5,
                     target_retrace: float = 0.382) -> np.ndarray:
    """Maeoknam-style pyramiding: add to winners after confirmation.

    When unrealized PnL exceeds add_threshold * target_move,
    increase position by pyramid_mult (50% more).

    Args:
        signals: Full signals from compute_signals_v9 + MTF
        close: Price array
        add_threshold: Fraction of target where we add (0.3 = 30% of way to target)
        pyramid_mult: How much to add (0.5 = 50% more on top)
        target_retrace: Same as strategy target_retrace
    """
    out = signals.copy()
    n = len(out)
    lookback = 8

    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue

        full_mag = out[i]
        direction = 1.0 if full_mag > 0 else -1.0
        entry_price = close[i]

        # Estimate move size for target calculation
        lb_idx = max(0, i - lookback)
        move = abs(close[i] / close[lb_idx] - 1.0)
        target_threshold = move * target_retrace * add_threshold

        j = i + 1
        pyramided = False
        while j < n and out[j] != 0:
            pnl_pct = direction * (close[j] / entry_price - 1.0)
            if not pyramided and pnl_pct >= target_threshold:
                # Add pyramid: increase signal magnitude
                out[j] = full_mag * (1.0 + pyramid_mult)
                pyramided = True
            elif pyramided:
                # Keep pyramided size
                out[j] = full_mag * (1.0 + pyramid_mult)
            j += 1

        i = j
    return out


# ──────────────── GAP 3: Trail Profit Runners ────────────────
def apply_trail_runners(signals: np.ndarray, close: np.ndarray,
                        target_retrace: float = 0.382,
                        trail_pct: float = 0.5,
                        trail_stop_retrace: float = 0.5,
                        max_trail_bars: int = 40) -> np.ndarray:
    """Maeoknam-style: 50% TP at target + trail remaining 50%.

    At target, reduce position to 50%. Then trail with a stop at 50%
    of max unrealized profit. Close on either stop or max_trail_bars.

    Uses _backtest_core_partial for correct delta-based cost accounting.

    Args:
        signals: Full signals from compute_signals_v9 + MTF
        close: Price array
        target_retrace: Strategy target retrace
        trail_pct: Fraction to keep trailing (0.5 = keep 50%)
        trail_stop_retrace: Trail stop as fraction of best PnL (0.5 = lose half gains → close)
        max_trail_bars: Max additional bars for trailing phase
    """
    out = signals.copy()
    n = len(out)
    lookback = 8

    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue

        full_mag = out[i]
        direction = 1.0 if full_mag > 0 else -1.0
        entry_price = close[i]

        lb_idx = max(0, i - lookback)
        move = abs(close[i] / close[lb_idx] - 1.0)
        target_move = move * target_retrace

        j = i + 1
        hit_target = False
        best_pnl = 0.0
        trail_bars = 0

        while j < n and out[j] != 0:
            pnl_pct = direction * (close[j] / entry_price - 1.0)

            if not hit_target and pnl_pct >= target_move:
                # Hit target: reduce to trail_pct of full position
                hit_target = True
                out[j] = full_mag * trail_pct
                best_pnl = pnl_pct
                trail_bars = 0
            elif hit_target:
                trail_bars += 1
                if pnl_pct > best_pnl:
                    best_pnl = pnl_pct

                # Trail stop: if PnL drops below trail_stop_retrace of best
                trail_stop = best_pnl * trail_stop_retrace
                if pnl_pct < trail_stop or trail_bars >= max_trail_bars:
                    # Close trailing position
                    for k in range(j, n):
                        if out[k] == 0:
                            break
                        out[k] = 0
                    break
                else:
                    out[j] = full_mag * trail_pct
            j += 1

        i = j + 1 if j < n else n
    return out


# ──────────────── GAP 4: BTC Macro Regime ────────────────
# No signal transform needed — just change btc_dir_period in params


def run_hypothesis(data: dict, name: str, transform_fn=None,
                   params: dict = None, use_partial: bool = False,
                   mtf_lev: float = MTF_LEV) -> dict:
    """Run a hypothesis across Top-15 with optional signal transform."""
    if params is None:
        params = V10
    results = []
    for sym in TOP15:
        if sym not in data:
            continue
        signals = get_signals(data, sym, params, mtf_lev=mtf_lev)
        if transform_fn is not None:
            signals = transform_fn(signals, data, sym)
        ret, dd, cal, nt, wr = backtest_symbol(data, sym, signals, use_partial=use_partial)
        results.append((sym, ret, dd, cal, nt, wr))

    total_ret = sum(r[1] for r in results)
    total_trades = sum(r[4] for r in results)
    avg_calmar = np.mean([r[3] for r in results]) if results else 0
    profitable = sum(1 for r in results if r[1] > 0)
    losers = sum(1 for r in results if r[1] <= 0)
    worst_dd = max(r[2] for r in results) if results else 0
    avg_wr = np.mean([r[5] for r in results]) if results else 0

    return {
        'name': name,
        'total_ret': total_ret,
        'trades': total_trades,
        'avg_calmar': avg_calmar,
        'profitable': profitable,
        'losers': losers,
        'worst_dd': worst_dd,
        'avg_wr': avg_wr,
        'details': results,
    }


def main():
    print("=" * 90)
    print("MAEOKNAM STRATEGY GAP A/B TEST")
    print("Testing 4 biggest differences from winning_strategy.md vs QuantBox v10.2")
    print("=" * 90)
    print("\nLoading data...")
    data = load_data()
    loaded = [s for s in TOP15 if s in data]
    print(f"Loaded {len(loaded)} symbols + BTC\n")

    # ── Baseline: v10.2 production ──
    baseline = run_hypothesis(data, "v10.2 Baseline")
    print(f"{'=' * 90}")
    print(f"BASELINE (v10.2): Ret={baseline['total_ret']:.1f}% | "
          f"Trades={baseline['trades']} | AvgCalmar={baseline['avg_calmar']:.2f} | "
          f"P/L={baseline['profitable']}/{baseline['profitable']+baseline['losers']} | "
          f"WR={baseline['avg_wr']:.1f}%")
    print(f"{'=' * 90}\n")

    hypotheses = []

    # ══════════════════════════════════════════════════════════════
    # GAP 1: Fractional Entry (Maeoknam's 2% pilot → 8% main)
    # ══════════════════════════════════════════════════════════════
    print("GAP 1: Fractional Entry (pilot → full)")
    for pilot_frac, confirm_bars in [(0.2, 2), (0.2, 3), (0.3, 2), (0.3, 3),
                                      (0.5, 2), (0.5, 3)]:
        def make_frac_entry(pf, cb):
            def tf(signals, d, sym):
                return apply_fractional_entry(signals, d[sym]['close'],
                                              pilot_fraction=pf,
                                              confirm_bars=cb)
            return tf

        r = run_hypothesis(data, f"G1: pilot={pilot_frac:.0%} confirm={confirm_bars}b",
                           make_frac_entry(pilot_frac, confirm_bars),
                           use_partial=True)
        hypotheses.append(('GAP1', r))

    # ══════════════════════════════════════════════════════════════
    # GAP 2: Pyramiding on Confirmed Moves
    # ══════════════════════════════════════════════════════════════
    print("GAP 2: Pyramiding (add to winners)")
    for add_thr, pyr_mult in [(0.2, 0.3), (0.2, 0.5), (0.3, 0.3),
                               (0.3, 0.5), (0.4, 0.5), (0.5, 0.5)]:
        def make_pyramid(at, pm):
            def tf(signals, d, sym):
                return apply_pyramiding(signals, d[sym]['close'],
                                        add_threshold=at,
                                        pyramid_mult=pm,
                                        target_retrace=0.382)
            return tf

        r = run_hypothesis(data, f"G2: add@{add_thr:.0%} +{pyr_mult:.0%}",
                           make_pyramid(add_thr, pyr_mult),
                           use_partial=True)
        hypotheses.append(('GAP2', r))

    # ══════════════════════════════════════════════════════════════
    # GAP 3: Trail Profit Runners (50% TP + trail rest)
    # ══════════════════════════════════════════════════════════════
    print("GAP 3: Trail Profit Runners (partial TP + trail)")
    for trail_pct, trail_stop, max_tb in [(0.5, 0.3, 20), (0.5, 0.5, 20),
                                           (0.5, 0.3, 40), (0.5, 0.5, 40),
                                           (0.3, 0.5, 30), (0.7, 0.3, 30)]:
        def make_trail(tp, ts, mt):
            def tf(signals, d, sym):
                return apply_trail_runners(signals, d[sym]['close'],
                                           target_retrace=0.382,
                                           trail_pct=tp,
                                           trail_stop_retrace=ts,
                                           max_trail_bars=mt)
            return tf

        r = run_hypothesis(data, f"G3: keep={trail_pct:.0%} ts={trail_stop:.0%} max={max_tb}",
                           make_trail(trail_pct, trail_stop, max_tb),
                           use_partial=True)
        hypotheses.append(('GAP3', r))

    # ══════════════════════════════════════════════════════════════
    # GAP 4: BTC Macro Regime (re-enable btc_directional_period)
    # ══════════════════════════════════════════════════════════════
    print("GAP 4: BTC Macro Regime (re-enable directional alignment)")
    for bdp in [16, 32, 48, 64, 96]:
        params_bdp = dict(V10, btc_dir_period=bdp)
        r = run_hypothesis(data, f"G4: btc_dir_period={bdp}",
                           params=params_bdp)
        hypotheses.append(('GAP4', r))

    # ══════════════════════════════════════════════════════════════
    # COMBO: Best from each gap (if any winners found)
    # ══════════════════════════════════════════════════════════════

    # ── Print group results ──
    base_ret = baseline['total_ret']
    base_cal = baseline['avg_calmar']
    base_wr = baseline['avg_wr']

    gap_order = ['GAP1', 'GAP2', 'GAP3', 'GAP4']
    gap_labels = {
        'GAP1': 'Fractional Entry (Pilot → Full)',
        'GAP2': 'Pyramiding (Add to Winners)',
        'GAP3': 'Trail Profit Runners (50% TP + Trail)',
        'GAP4': 'BTC Macro Regime (Directional Alignment)',
    }

    all_winners = []

    for gap in gap_order:
        group = [h for g, h in hypotheses if g == gap]
        print(f"\n{'═' * 90}")
        print(f"  {gap}: {gap_labels[gap]}")
        print(f"{'═' * 90}")
        print(f"  {'Variant':<35} {'Ret%':>8} {'ΔRet%':>8} {'Calmar':>8} "
              f"{'ΔCal':>8} {'WR%':>6} {'Trades':>7} {'P/L':>6}")
        print(f"  {'-'*35} {'-----':>8} {'-----':>8} {'------':>8} "
              f"{'-----':>8} {'----':>6} {'------':>7} {'---':>6}")

        for h in group:
            d_ret = h['total_ret'] - base_ret
            d_cal = h['avg_calmar'] - base_cal
            wins = h['profitable']
            total = wins + h['losers']
            flag = " ★" if d_ret > 0 and d_cal > 0 else ""
            if d_ret > 0 and d_cal > 0:
                all_winners.append(h)
            print(f"  {h['name']:<35} {h['total_ret']:>7.1f}% {d_ret:>+7.1f}% "
                  f"{h['avg_calmar']:>8.2f} {d_cal:>+7.2f} {h['avg_wr']:>5.1f}% "
                  f"{h['trades']:>7} {wins:>2}/{total:>2}{flag}")

    # ── Final Summary ──
    print(f"\n{'═' * 90}")
    print("FINAL SUMMARY")
    print(f"{'═' * 90}")
    print(f"\n  Baseline: Ret={base_ret:.1f}%, Calmar={base_cal:.2f}, WR={base_wr:.1f}%\n")

    if all_winners:
        print("  WINNERS (beat baseline on BOTH Ret AND Calmar):")
        for w in sorted(all_winners, key=lambda x: x['avg_calmar'], reverse=True):
            d_ret = w['total_ret'] - base_ret
            d_cal = w['avg_calmar'] - base_cal
            print(f"    ★ {w['name']}: Ret={w['total_ret']:.1f}% ({d_ret:>+.1f}%), "
                  f"Calmar={w['avg_calmar']:.2f} ({d_cal:>+.2f}), "
                  f"WR={w['avg_wr']:.1f}%")
            print(f"      {'Symbol':<12} {'Ret%':>8} {'DD%':>8} {'Calmar':>8} {'Trades':>7} {'WR%':>6}")
            for sym, ret, dd, cal, nt, wr in w['details']:
                flag = "✓" if ret > 0 else "✗"
                print(f"      {sym:<12} {ret:>7.1f}% {dd:>7.1f}% {cal:>8.2f} {nt:>7} {wr:>5.1f}% {flag}")
        print()
    else:
        print("  No winners. QuantBox v10.2 is already optimal against Maeoknam gaps.\n")

    # ── Per-gap verdict ──
    print("  PER-GAP VERDICT:")
    for gap in gap_order:
        group = [h for g, h in hypotheses if g == gap]
        gap_winners = [h for h in group if h['total_ret'] > base_ret and h['avg_calmar'] > base_cal]
        best = max(group, key=lambda x: x['avg_calmar'])
        if gap_winners:
            best_w = max(gap_winners, key=lambda x: x['avg_calmar'])
            print(f"    {gap}: ✓ ALPHA FOUND — best={best_w['name']}, "
                  f"Calmar={best_w['avg_calmar']:.2f} (+{best_w['avg_calmar']-base_cal:.2f})")
        else:
            print(f"    {gap}: ✗ NO ALPHA — best was {best['name']} "
                  f"(Ret={best['total_ret']:.1f}%, Calmar={best['avg_calmar']:.2f})")

    print(f"\n{'═' * 90}")


if __name__ == '__main__':
    main()
