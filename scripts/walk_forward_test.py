#!/usr/bin/env python3
"""
Walk-Forward Validation of Rubber Band Strategy

Tests the strategy across multiple rolling windows to ensure
alpha is not regime-dependent.

Windows (8640 bars = 90 days):
  W1: IS 0-2880, OOS 2880-4320  (month 1-2 train, month 2-3 test)  
  W2: IS 1440-4320, OOS 4320-5760
  W3: IS 2880-5760, OOS 5760-7200
  W4: IS 4320-7200, OOS 7200-8640

If the strategy works in ALL windows, it's robust.
If only in some, it's regime-dependent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import numba
import time
from quantbox.data.real_data_loader import RealDataLoader


COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.01 / 100 / 32


@numba.jit(nopython=True, cache=True)
def _ema(data, period):
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


@numba.jit(nopython=True, cache=True)
def _atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    s = 0.0
    for i in range(period):
        s += tr[i]
    atr[period - 1] = s / period
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


@numba.jit(nopython=True, cache=True)
def rubberband_signals(close, lookback, threshold, target_retrace, stop_mult, max_hold):
    n = len(close)
    sig = np.zeros(n)
    position = 0
    entry_price = 0.0
    move_size = 0.0
    hold_count = 0
    best_pnl = 0.0
    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0
        if position == 0:
            if ret < -threshold:
                position = 1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
            elif ret > threshold:
                position = -1
                entry_price = close[i]
                move_size = abs(ret)
                hold_count = 0
                best_pnl = 0.0
        else:
            hold_count += 1
            if position == 1:
                gain = (close[i] - entry_price) / entry_price
            else:
                gain = (entry_price - close[i]) / entry_price
            if gain > best_pnl:
                best_pnl = gain
            if gain >= move_size * target_retrace:
                position = 0
            elif gain < -move_size * stop_mult:
                position = 0
            elif hold_count >= max_hold:
                position = 0
            elif best_pnl > move_size * target_retrace * 0.5 and gain < best_pnl * 0.3:
                position = 0
        sig[i] = position
    return sig


@numba.jit(nopython=True, cache=True)
def vectorized_backtest(close, signals, cost_per_side, funding_per_bar):
    n = len(close)
    equity = np.ones(n)
    position = 0.0
    prev_signal = 0.0
    n_trades = 0
    n_wins = 0
    peak = 1.0
    max_dd = 0.0
    trade_entry_eq = 1.0
    for i in range(1, n):
        ret = close[i] / close[i - 1] - 1.0
        bar_pnl = position * ret
        new_pos = signals[i]
        if new_pos != prev_signal:
            cost = abs(prev_signal) * cost_per_side + abs(new_pos) * cost_per_side
            bar_pnl -= cost
            if prev_signal != 0.0:
                n_trades += 1
                if equity[i - 1] * (1.0 + bar_pnl) > trade_entry_eq:
                    n_wins += 1
            if new_pos != 0.0:
                trade_entry_eq = equity[i - 1] * (1.0 + bar_pnl)
        if position != 0.0:
            bar_pnl -= abs(position) * funding_per_bar
        equity[i] = equity[i - 1] * (1.0 + bar_pnl)
        position = new_pos
        prev_signal = new_pos
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
    if position != 0.0:
        n_trades += 1
    total_return = equity[-1] / equity[0] - 1.0
    rets = np.empty(n - 1)
    for i in range(n - 1):
        rets[i] = equity[i + 1] / equity[i] - 1.0
    mean_r = np.mean(rets)
    std_r = np.std(rets)
    sharpe = mean_r / std_r * np.sqrt(96.0 * 365.25) if std_r > 0 else 0.0
    win_rate = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0
    return total_return, sharpe, max_dd, win_rate, n_trades


def main():
    print("=" * 100)
    print("WALK-FORWARD VALIDATION — Rubber Band Strategy")
    print("=" * 100)

    loader = RealDataLoader()
    data = loader.load_all(min_bars=1000, align_data=True)
    symbols = sorted(data.keys())
    n_bars = len(next(iter(data.values())))

    # Pre-compute
    all_close = {}
    for sym in symbols:
        all_close[sym] = data[sym]['Close'].values.astype(np.float64)

    # Warmup numba
    dummy = np.random.randn(200).astype(np.float64) + 100
    rubberband_signals(dummy, 6, 0.03, 0.3, 1.0, 8)
    vectorized_backtest(dummy, np.zeros(200), 0.0004, 0.0)

    # Walk-forward windows
    is_len = 2880  # 30 days
    oos_len = 1440  # 15 days
    step = 1440     # 15 days

    windows = []
    start = 0
    while start + is_len + oos_len <= n_bars:
        windows.append((start, start + is_len, start + is_len, start + is_len + oos_len))
        start += step

    print(f"\nData: {len(symbols)} symbols, {n_bars} bars")
    print(f"Walk-forward: {len(windows)} windows, IS={is_len} bars, OOS={oos_len} bars, step={step}")

    # Configs to test (best and neighbors from refinement)
    configs = [
        ("L6_T30_R30_S10_H8",  6, 0.030, 0.30, 1.0, 8),
        ("L8_T30_R30_S10_H8",  8, 0.030, 0.30, 1.0, 8),
        ("L6_T30_R30_S15_H8",  6, 0.030, 0.30, 1.5, 8),
        ("L8_T30_R40_S10_H12", 8, 0.030, 0.40, 1.0, 12),
        ("L8_T25_R50_S10_H8",  8, 0.025, 0.50, 1.0, 8),
        ("L8_T35_R30_S10_H8",  8, 0.035, 0.30, 1.0, 8),
        ("L10_T35_R40_S10_H8", 10, 0.035, 0.40, 1.0, 8),
    ]

    # Also test with higher costs for extra conservatism
    cost_levels = [
        (0.0004, "0.04%"),
        (0.0006, "0.06%"),
        (0.0008, "0.08%"),
    ]

    print(f"\nTesting {len(configs)} configs × {len(windows)} windows × {len(cost_levels)} cost levels\n")

    for cost, cost_label in cost_levels:
        print(f"\n{'='*100}")
        print(f"COST: {cost_label} per side")
        print(f"{'='*100}")

        header = (f"{'Config':<24}")
        for wi, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
            header += f" {'W'+str(wi+1)+' Ret%':>9} {'Shrp':>6}"
        header += f" {'AvgOOS%':>9} {'AvgShrp':>8} {'%Win':>6}"
        print(header)
        print("-" * len(header))

        for label, lb, thr, tgt, sm, mh in configs:
            row = f"{label:<24}"
            oos_rets = []
            oos_sharpes = []
            wins = 0

            for wi, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
                # Average across symbols
                sym_rets = []
                sym_sharpes = []
                for sym in symbols:
                    c = all_close[sym]
                    sig = rubberband_signals(c, lb, thr, tgt, sm, mh)
                    # Only test OOS
                    r, s, d, w, t = vectorized_backtest(
                        c[oos_s:oos_e], sig[oos_s:oos_e], cost, FUNDING_PER_BAR
                    )
                    sym_rets.append(r)
                    sym_sharpes.append(s)

                avg_ret = np.mean(sym_rets) * 100
                avg_sharpe = np.mean(sym_sharpes)
                oos_rets.append(avg_ret)
                oos_sharpes.append(avg_sharpe)
                if avg_ret > 0:
                    wins += 1

                row += f" {avg_ret:>+9.2f} {avg_sharpe:>6.2f}"

            avg_oos = np.mean(oos_rets)
            avg_sh = np.mean(oos_sharpes)
            win_pct = wins / len(windows) * 100
            flag = "★" if win_pct >= 60 and avg_oos > 0 else ""
            row += f" {avg_oos:>+9.2f} {avg_sh:>8.2f} {win_pct:>5.0f}% {flag}"
            print(row)

    # ── Also: full-period backtest (non-walk-forward) ──
    print(f"\n{'='*100}")
    print("FULL-PERIOD BACKTEST (all 8640 bars, no IS/OOS split)")
    print(f"{'='*100}")

    header2 = f"{'Config':<24} {'Ret%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WR%':>7} {'Trades':>7} {'%+Sym':>7}"
    print(header2)
    print("-" * len(header2))

    for label, lb, thr, tgt, sm, mh in configs:
        sym_rets = []
        sym_sharpes = []
        sym_dds = []
        sym_wrs = []
        sym_trades = []
        sym_positive = 0

        for sym in symbols:
            c = all_close[sym]
            sig = rubberband_signals(c, lb, thr, tgt, sm, mh)
            r, s, d, w, t = vectorized_backtest(c, sig, COST_PER_SIDE, FUNDING_PER_BAR)
            sym_rets.append(r * 100)
            sym_sharpes.append(s)
            sym_dds.append(d * 100)
            sym_wrs.append(w)
            sym_trades.append(t)
            if r > 0:
                sym_positive += 1

        avg_ret = np.mean(sym_rets)
        avg_sh = np.mean(sym_sharpes)
        max_dd = np.max(sym_dds)
        avg_wr = np.mean(sym_wrs)
        total_trades = sum(sym_trades)
        pct_pos = sym_positive / len(symbols) * 100
        flag = "★" if pct_pos >= 50 and avg_ret > 0 else ""

        print(f"{label:<24} {avg_ret:>+9.2f} {avg_sh:>8.2f} {max_dd:>8.1f} "
              f"{avg_wr:>7.1f} {total_trades:>7d} {pct_pos:>6.0f}% {flag}")

    # ── Portfolio equity curve for best config ──
    print(f"\n{'='*100}")
    print("PORTFOLIO EQUITY: Equal-weight all 30 symbols (L6_T30_R30_S10_H8)")
    print(f"{'='*100}")

    lb, thr, tgt, sm, mh = 6, 0.03, 0.3, 1.0, 8
    n = n_bars
    portfolio_eq = np.ones(n)
    n_active = 0

    for sym in symbols:
        c = all_close[sym]
        sig = rubberband_signals(c, lb, thr, tgt, sm, mh)
        # Per-symbol equity
        sym_eq = np.ones(n)
        position = 0.0
        prev_signal = 0.0
        for i in range(1, n):
            ret = c[i] / c[i-1] - 1.0
            bar_pnl = position * ret
            new_pos = sig[i]
            if new_pos != prev_signal:
                bar_pnl -= abs(prev_signal) * COST_PER_SIDE + abs(new_pos) * COST_PER_SIDE
            if position != 0.0:
                bar_pnl -= abs(position) * FUNDING_PER_BAR
            sym_eq[i] = sym_eq[i-1] * (1.0 + bar_pnl)
            position = new_pos
            prev_signal = new_pos
        # Equal weight contribution
        portfolio_eq *= (sym_eq ** (1.0 / len(symbols)))
        n_active += 1

    final_ret = (portfolio_eq[-1] - 1.0) * 100

    # Rolling drawdown
    peak = np.maximum.accumulate(portfolio_eq)
    drawdown = (peak - portfolio_eq) / peak
    max_dd = np.max(drawdown) * 100

    # Monthly returns
    bars_per_month = 96 * 30
    print(f"\nPortfolio final return: {final_ret:+.2f}%")
    print(f"Portfolio max drawdown: {max_dd:.2f}%")
    print(f"\nMonthly (approx) returns:")
    for m in range(0, n, bars_per_month):
        end = min(m + bars_per_month, n)
        month_ret = (portfolio_eq[end-1] / portfolio_eq[m] - 1.0) * 100
        print(f"  Month {m // bars_per_month + 1}: {month_ret:+.2f}%")


if __name__ == "__main__":
    main()
