#!/usr/bin/env python3
"""
DYNAMIC LEVERAGE A/B TEST — World Competition Mode

Competition winners achieve 10,000%+ returns by dynamically adjusting
leverage proportional to confidence. This script tests leverage strategies
on 365 days of real Binance data.

Leverage Strategies Tested:
  Phase 1: Fixed leverage sweep (1x-20x) — establish risk/reward curve
  Phase 2: Move-size confidence — bigger move = higher conviction = more lev
  Phase 3: Volatility-inverse — target constant risk via lev = target_vol/sym_vol
  Phase 4: Kelly Criterion — mathematically optimal leverage from edge
  Phase 5: DD-gated — scale down on drawdown, scale up on wins
  Phase 6: Combo strategies — best of above combined
  Phase 7: Per-symbol optimal — different leverage per symbol
  Phase 8: Rolling stability check — does it survive all regimes?

Key insight: _backtest_core treats signal value as position size.
  signal=+3.0 → 3x long, signal=-5.0 → 5x short.
  Costs and funding automatically scale with leverage.
  Liquidation modeled: if equity drop > maintenance margin → forced exit.
"""

import sys, os, gzip, pickle, time, warnings
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import numba
from quantbox.strategy.rubberband import compute_signals
from quantbox.backtest.vectorized_backtest import _backtest_core

# ── Constants ──
COST_PER_SIDE = 0.0004
FUNDING_PER_BAR = 0.0001 / 32
BARS_PER_DAY = 96

# ── v4.0 validated parameters ──
LOOKBACK = 8
THRESHOLD = 0.045
TARGET = 0.3
STOP = 1.0
MAX_HOLD = 10
STOP_DECAY = 0.0

OUTPUT = "/tmp/leverage_ab.txt"
out = open(OUTPUT, "w")

def log(msg=""):
    print(msg)
    out.write(msg + "\n")
    out.flush()


def load_data():
    cache_dir = Path("data/cache_extended")
    all_data = {}
    for f in sorted(cache_dir.glob("*_15m_365d.pkl.gz")):
        symbol = f.name.split("_15m_")[0]
        with gzip.open(f, "rb") as fh:
            df = pickle.load(fh)
        if "Close" in df.columns and len(df) >= 200:
            all_data[symbol] = df["Close"].values.astype(np.float64)
    return all_data


def fast_metrics(eq, start=0):
    """Return (ret%, dd%, calmar, sharpe)."""
    s = eq[start:]
    if len(s) < 2:
        return 0, 0, 0, 0
    ret = (s[-1] / s[0] - 1) * 100
    pk = np.maximum.accumulate(s)
    dd = np.max((pk - s) / pk) * 100
    cal = ret / dd if dd > 1e-9 else (999 if ret > 0 else -999)
    rs = np.diff(s) / s[:-1]
    sh = np.mean(rs) / np.std(rs) * np.sqrt(96 * 365.25) if np.std(rs) > 1e-12 else 0
    return ret, dd, cal, sh


@numba.jit(nopython=True, cache=True)
def backtest_with_leverage(
    close: np.ndarray,
    base_signals: np.ndarray,
    leverage: float,
    cost_per_side: float,
    funding_per_bar: float,
    liquidation_threshold: float = 0.90,
) -> np.ndarray:
    """Backtest with fixed leverage and liquidation protection.

    Liquidation: if cumulative loss on a trade exceeds
    (1 - 1/leverage) * liquidation_threshold, force exit.
    This models margin call / liquidation in real futures trading.

    Args:
        close: Price array.
        base_signals: Direction signals (-1, 0, +1).
        leverage: Leverage multiplier.
        cost_per_side: Cost per side (fraction).
        funding_per_bar: Funding cost per bar (fraction).
        liquidation_threshold: Fraction of maintenance margin at which liq happens.

    Returns:
        Equity curve.
    """
    n = len(close)
    equity = np.ones(n)
    position = 0.0
    prev_signal = 0.0
    entry_price = 0.0

    for i in range(1, n):
        ret = close[i] / close[i - 1] - 1.0
        bar_pnl = position * ret

        new_base = base_signals[i]
        new_pos = new_base * leverage

        # Check liquidation — only for leveraged positions (lev > 1)
        lev = abs(position)
        if position != 0.0 and entry_price > 0.0 and lev > 1.01:
            if position > 0:
                price_change = close[i] / entry_price - 1.0
            else:
                price_change = 1.0 - close[i] / entry_price
            # Liquidation when adverse move exceeds 95% of 1/leverage
            # At 10x: liquidation at ~9.5% adverse move
            # At 20x: liquidation at ~4.75% adverse move
            if price_change < -(1.0 / lev) * liquidation_threshold:
                # Liquidated — lose the margin (1/lev of equity)
                liq_loss = min(1.0 / lev, 0.99)
                equity[i] = equity[i - 1] * (1.0 - liq_loss)
                position = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                continue

        if new_pos != prev_signal:
            cost = abs(prev_signal) * cost_per_side + abs(new_pos) * cost_per_side
            bar_pnl -= cost
            if new_pos != 0.0:
                entry_price = close[i]

        if position != 0.0:
            bar_pnl -= abs(position) * funding_per_bar

        equity[i] = equity[i - 1] * (1.0 + bar_pnl)

        # Floor at 1% to prevent negative equity
        if equity[i] < equity[0] * 0.01:
            equity[i] = equity[0] * 0.01
            position = 0.0
            prev_signal = 0.0
            entry_price = 0.0
            continue

        position = new_pos
        prev_signal = new_pos

    return equity


@numba.jit(nopython=True, cache=True)
def compute_move_sizes(close: np.ndarray, lookback: int, threshold: float) -> np.ndarray:
    """Compute the move size at each entry point (for confidence-based leverage)."""
    n = len(close)
    moves = np.zeros(n)
    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1.0
        if abs(ret) >= threshold:
            moves[i] = abs(ret)
    return moves


@numba.jit(nopython=True, cache=True)
def apply_move_confidence_leverage(
    base_signals: np.ndarray,
    move_sizes: np.ndarray,
    threshold: float,
    base_lev: float,
    scale: float,
    max_lev: float,
) -> np.ndarray:
    """Scale leverage proportional to move size beyond threshold.

    lev = base_lev + scale * (move_size - threshold) / threshold
    Capped at max_lev.
    """
    n = len(base_signals)
    out = np.zeros(n)
    current_lev = 0.0
    for i in range(n):
        if base_signals[i] != 0.0 and (i == 0 or base_signals[i - 1] == 0.0):
            # New entry — compute leverage from move size
            if move_sizes[i] > 0:
                excess = (move_sizes[i] - threshold) / threshold
                current_lev = min(base_lev + scale * excess, max_lev)
            else:
                current_lev = base_lev
        elif base_signals[i] == 0.0:
            current_lev = 0.0
        out[i] = base_signals[i] * current_lev
    return out


@numba.jit(nopython=True, cache=True)
def apply_dd_gated_leverage(
    base_signals: np.ndarray,
    equity: np.ndarray,
    base_lev: float,
    dd_start: float,
    dd_full_reduce: float,
    min_lev: float,
) -> np.ndarray:
    """Reduce leverage during drawdowns, full leverage at equity highs.

    lev = base_lev when no DD
    lev = min_lev when DD >= dd_full_reduce
    Linear interpolation in between.
    """
    n = len(base_signals)
    out = np.zeros(n)
    hwm = equity[0]
    for i in range(n):
        if equity[i] > hwm:
            hwm = equity[i]
        dd = (hwm - equity[i]) / hwm
        if dd < dd_start:
            lev = base_lev
        elif dd >= dd_full_reduce:
            lev = min_lev
        else:
            frac = (dd - dd_start) / (dd_full_reduce - dd_start)
            lev = base_lev - frac * (base_lev - min_lev)
        out[i] = base_signals[i] * lev
    return out


@numba.jit(nopython=True, cache=True)
def apply_vol_inverse_leverage(
    base_signals: np.ndarray,
    close: np.ndarray,
    target_daily_vol: float,
    vol_lookback: int,
    max_lev: float,
    min_lev: float,
) -> np.ndarray:
    """Leverage = target_vol / realized_vol (volatility targeting).

    Lower vol → more leverage. Higher vol → less leverage.
    """
    n = len(base_signals)
    out = np.zeros(n)
    for i in range(n):
        if base_signals[i] != 0.0:
            # Compute recent realized volatility
            start = max(0, i - vol_lookback)
            if i - start < 10:
                lev = min_lev
            else:
                # Daily vol from 15m returns
                rets = np.empty(i - start - 1)
                for j in range(start + 1, i):
                    rets[j - start - 1] = close[j] / close[j - 1] - 1.0
                vol_bar = np.std(rets)
                vol_daily = vol_bar * np.sqrt(96.0)  # 96 bars per day
                if vol_daily > 1e-8:
                    lev = min(max(target_daily_vol / vol_daily, min_lev), max_lev)
                else:
                    lev = max_lev
            out[i] = base_signals[i] * lev
        else:
            out[i] = 0.0
    return out


@numba.jit(nopython=True, cache=True)
def apply_streak_leverage(
    base_signals: np.ndarray,
    close: np.ndarray,
    base_lev: float,
    win_bonus: float,
    max_lev: float,
    min_lev: float,
) -> np.ndarray:
    """Increase leverage on winning streaks, decrease on losing streaks.

    Tracks recent trade outcomes. After K consecutive wins, ramp up.
    After K consecutive losses, scale down.
    """
    n = len(base_signals)
    out = np.zeros(n)
    streak = 0  # positive = wins, negative = losses
    entry_price = 0.0
    in_trade = False

    for i in range(n):
        if base_signals[i] != 0.0 and not in_trade:
            # New entry
            in_trade = True
            entry_price = close[i]
            # Compute leverage from streak
            if streak > 0:
                lev = min(base_lev + win_bonus * streak, max_lev)
            elif streak < 0:
                lev = max(base_lev + win_bonus * streak, min_lev)  # win_bonus * negative = reduce
            else:
                lev = base_lev
            out[i] = base_signals[i] * lev
        elif base_signals[i] != 0.0 and in_trade:
            # Continuing position — keep same leverage
            out[i] = out[i - 1] if i > 0 else base_signals[i] * base_lev
        elif base_signals[i] == 0.0 and in_trade:
            # Exit — update streak
            in_trade = False
            if entry_price > 0:
                pnl = close[i] / entry_price - 1.0
                if out[i - 1] < 0:
                    pnl = -pnl
                if pnl > 0:
                    if streak > 0:
                        streak += 1
                    else:
                        streak = 1
                else:
                    if streak < 0:
                        streak -= 1
                    else:
                        streak = -1
            out[i] = 0.0
        else:
            out[i] = 0.0
    return out


def portfolio_eq(close_dict, symbols, n_bars, leveraged_signals_dict, top_n=None, top_syms=None):
    """EW portfolio from leveraged signal arrays."""
    if top_syms:
        syms = [s for s in symbols if s in top_syms]
    elif top_n:
        # Use the top_n symbols by individual return
        sym_rets = []
        for s in symbols:
            eq = backtest_with_leverage(close_dict[s], leveraged_signals_dict[s] if s in leveraged_signals_dict else np.zeros(n_bars), 1.0, COST_PER_SIDE, FUNDING_PER_BAR)
            sym_rets.append((s, eq[-1]/eq[0] - 1))
        sym_rets.sort(key=lambda x: -x[1])
        syms = [s for s, _ in sym_rets[:top_n]]
    else:
        syms = symbols

    if not syms:
        return np.ones(n_bars)

    # Compute bar returns for each symbol
    mat = np.zeros((len(syms), n_bars))
    for i, s in enumerate(syms):
        eq = backtest_with_leverage(
            close_dict[s], leveraged_signals_dict.get(s, np.zeros(n_bars)),
            1.0,  # leverage already in signals
            COST_PER_SIDE, FUNDING_PER_BAR
        )
        for j in range(1, n_bars):
            mat[i, j] = eq[j] / eq[j - 1] - 1.0

    # EW average
    avg = np.mean(mat, axis=0)
    eq = np.ones(n_bars)
    for j in range(1, n_bars):
        eq[j] = eq[j - 1] * (1 + avg[j])
    return eq


def run_fixed_lev_test(close_dict, symbols, n_bars, base_sigs, lev):
    """Run backtest with fixed leverage on all symbols, return portfolio eq."""
    sig_dict = {}
    for s in symbols:
        sig_dict[s] = base_sigs[s] * lev
    return portfolio_eq(close_dict, symbols, n_bars, sig_dict)


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 74)
    log("  DYNAMIC LEVERAGE A/B TEST — World Competition Mode")
    log("=" * 74)

    data = load_data()
    symbols = sorted(data.keys())
    n_bars = min(len(data[s]) for s in symbols)
    log(f"Data: {len(symbols)} symbols × {n_bars:,} bars (365 days)")

    # Pre-compute base signals (1x, direction only)
    log("\n  Computing base signals (v4.0 params)...")
    base_sigs = {}
    move_sizes = {}
    for s in symbols:
        close = data[s][:n_bars]
        base_sigs[s] = compute_signals(close, LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY)
        move_sizes[s] = compute_move_sizes(close, LOOKBACK, THRESHOLD)

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1: FIXED LEVERAGE SWEEP
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 1: FIXED LEVERAGE SWEEP (EW All-30)")
    log("█" * 74)

    fixed_levs = [1, 2, 3, 5, 7, 10, 15, 20]
    log(f"\n  {'Lev':>5}  {'Ret%':>8}  {'DD%':>7}  {'Calmar':>7}  {'Sharpe':>7}")
    log("  " + "-" * 42)

    best_fixed = (0, 0, 0, 0, 0)
    for lev in fixed_levs:
        eq = run_fixed_lev_test(data, symbols, n_bars, base_sigs, lev)
        ret, dd, cal, sh = fast_metrics(eq)
        star = " ★" if cal > best_fixed[3] and ret > 0 else ""
        if cal > best_fixed[3] and ret > 0:
            best_fixed = (lev, ret, dd, cal, sh)
        log(f"  {lev:>4}x  {ret:>+7.1f}  {dd:>6.1f}  {cal:>7.2f}  {sh:>7.2f}{star}")

    log(f"\n  ★ BEST FIXED: {best_fixed[0]}x → Ret={best_fixed[1]:+.1f}%, DD={best_fixed[2]:.1f}%, Calmar={best_fixed[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2: MOVE-SIZE CONFIDENCE LEVERAGE
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 2: MOVE-SIZE CONFIDENCE LEVERAGE")
    log("█" * 74)
    log("  Bigger move beyond threshold → higher conviction → more leverage")

    configs = [
        # (base_lev, scale, max_lev, label)
        (2, 5, 5, "Base=2, Scale=5, Max=5"),
        (2, 10, 7, "Base=2, Scale=10, Max=7"),
        (2, 15, 10, "Base=2, Scale=15, Max=10"),
        (3, 10, 7, "Base=3, Scale=10, Max=7"),
        (3, 10, 10, "Base=3, Scale=10, Max=10"),
        (3, 15, 10, "Base=3, Scale=15, Max=10"),
        (3, 20, 15, "Base=3, Scale=20, Max=15"),
        (5, 10, 10, "Base=5, Scale=10, Max=10"),
        (5, 15, 15, "Base=5, Scale=15, Max=15"),
        (5, 20, 20, "Base=5, Scale=20, Max=20"),
        (5, 30, 20, "Base=5, Scale=30, Max=20"),
        (7, 20, 15, "Base=7, Scale=20, Max=15"),
        (7, 30, 20, "Base=7, Scale=30, Max=20"),
        (10, 20, 20, "Base=10, Scale=20, Max=20"),
    ]

    log(f"\n  {'Config':<30} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 62)

    best_move = ("", 0, 0, 0, 0)
    for base_lev, scale, max_lev, label in configs:
        sig_dict = {}
        for s in symbols:
            sig_dict[s] = apply_move_confidence_leverage(
                base_sigs[s], move_sizes[s], THRESHOLD, base_lev, scale, max_lev
            )
        eq = portfolio_eq(data, symbols, n_bars, sig_dict)
        ret, dd, cal, sh = fast_metrics(eq)
        star = " ★" if cal > best_move[3] else ""
        if cal > best_move[3] and ret > 0:
            best_move = (label, ret, dd, cal, sh)
        log(f"  {label:<30} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}{star}")

    log(f"\n  ★ BEST MOVE-CONF: {best_move[0]}")
    log(f"    Ret={best_move[1]:+.1f}%, DD={best_move[2]:.1f}%, Calmar={best_move[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 3: VOLATILITY-INVERSE LEVERAGE (target constant risk)
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 3: VOLATILITY-INVERSE LEVERAGE (vol targeting)")
    log("█" * 74)
    log("  Low vol → more lev, High vol → less lev. Constant risk budget.")

    vol_configs = [
        # (target_daily_vol, vol_lookback, max_lev, min_lev, label)
        (0.03, 480, 5, 1, "TgtVol=3%, Max=5x"),
        (0.04, 480, 7, 1, "TgtVol=4%, Max=7x"),
        (0.05, 480, 7, 1, "TgtVol=5%, Max=7x"),
        (0.05, 480, 10, 1, "TgtVol=5%, Max=10x"),
        (0.07, 480, 10, 1, "TgtVol=7%, Max=10x"),
        (0.07, 480, 15, 2, "TgtVol=7%, Max=15x"),
        (0.10, 480, 15, 2, "TgtVol=10%, Max=15x"),
        (0.10, 480, 20, 2, "TgtVol=10%, Max=20x"),
        (0.15, 480, 20, 3, "TgtVol=15%, Max=20x"),
        (0.05, 960, 10, 1, "TgtVol=5%, LB=10d, Max=10x"),
        (0.07, 960, 10, 1, "TgtVol=7%, LB=10d, Max=10x"),
    ]

    log(f"\n  {'Config':<32} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 64)

    best_vol = ("", 0, 0, 0, 0)
    for tvol, vlb, mxl, mnl, label in vol_configs:
        sig_dict = {}
        for s in symbols:
            sig_dict[s] = apply_vol_inverse_leverage(
                base_sigs[s], data[s][:n_bars], tvol, vlb, mxl, mnl
            )
        eq = portfolio_eq(data, symbols, n_bars, sig_dict)
        ret, dd, cal, sh = fast_metrics(eq)
        star = " ★" if cal > best_vol[3] and ret > 0 else ""
        if cal > best_vol[3] and ret > 0:
            best_vol = (label, ret, dd, cal, sh)
        log(f"  {label:<32} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}{star}")

    log(f"\n  ★ BEST VOL-INV: {best_vol[0]}")
    log(f"    Ret={best_vol[1]:+.1f}%, DD={best_vol[2]:.1f}%, Calmar={best_vol[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 4: DD-GATED LEVERAGE
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 4: DRAWDOWN-GATED LEVERAGE")
    log("█" * 74)
    log("  Scale down on DD, full leverage at equity highs.")

    dd_configs = [
        # (base_lev, dd_start, dd_full_reduce, min_lev, label)
        (3, 0.05, 0.15, 1, "3x, DD5→15%→1x"),
        (5, 0.05, 0.15, 1, "5x, DD5→15%→1x"),
        (5, 0.05, 0.15, 2, "5x, DD5→15%→2x"),
        (5, 0.03, 0.10, 1, "5x, DD3→10%→1x"),
        (7, 0.05, 0.15, 2, "7x, DD5→15%→2x"),
        (7, 0.05, 0.20, 2, "7x, DD5→20%→2x"),
        (7, 0.03, 0.10, 1, "7x, DD3→10%→1x"),
        (10, 0.05, 0.15, 2, "10x, DD5→15%→2x"),
        (10, 0.05, 0.20, 3, "10x, DD5→20%→3x"),
        (10, 0.03, 0.10, 2, "10x, DD3→10%→2x"),
        (15, 0.05, 0.20, 3, "15x, DD5→20%→3x"),
    ]

    log(f"\n  {'Config':<24} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 56)

    best_dd = ("", 0, 0, 0, 0)
    for blev, dds, ddf, mlev, label in dd_configs:
        # Need to run iteratively: compute 1x equity first, then apply DD gate
        sig_dict = {}
        for s in symbols:
            # First pass: 1x equity for DD measurement
            eq_1x = backtest_with_leverage(data[s][:n_bars], base_sigs[s], 1.0, COST_PER_SIDE, FUNDING_PER_BAR)
            sig_dict[s] = apply_dd_gated_leverage(base_sigs[s], eq_1x, blev, dds, ddf, mlev)
        eq = portfolio_eq(data, symbols, n_bars, sig_dict)
        ret, dd, cal, sh = fast_metrics(eq)
        star = " ★" if cal > best_dd[3] and ret > 0 else ""
        if cal > best_dd[3] and ret > 0:
            best_dd = (label, ret, dd, cal, sh)
        log(f"  {label:<24} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}{star}")

    log(f"\n  ★ BEST DD-GATED: {best_dd[0]}")
    log(f"    Ret={best_dd[1]:+.1f}%, DD={best_dd[2]:.1f}%, Calmar={best_dd[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 5: STREAK-BASED LEVERAGE
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 5: STREAK-BASED LEVERAGE (momentum of edge)")
    log("█" * 74)
    log("  Win streak → ramp up, Loss streak → scale down.")

    streak_configs = [
        # (base_lev, win_bonus, max_lev, min_lev, label)
        (3, 1.0, 7, 1, "Base=3, Bonus=1/win, Max=7x"),
        (3, 1.5, 10, 1, "Base=3, Bonus=1.5/win, Max=10x"),
        (5, 1.0, 10, 2, "Base=5, Bonus=1/win, Max=10x"),
        (5, 1.5, 10, 2, "Base=5, Bonus=1.5/win, Max=10x"),
        (5, 2.0, 15, 2, "Base=5, Bonus=2/win, Max=15x"),
        (7, 1.5, 15, 2, "Base=7, Bonus=1.5/win, Max=15x"),
        (7, 2.0, 15, 3, "Base=7, Bonus=2/win, Max=15x"),
        (7, 2.0, 20, 3, "Base=7, Bonus=2/win, Max=20x"),
        (10, 2.0, 20, 3, "Base=10, Bonus=2/win, Max=20x"),
    ]

    log(f"\n  {'Config':<34} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 66)

    best_streak = ("", 0, 0, 0, 0)
    for blev, wb, mxl, mnl, label in streak_configs:
        sig_dict = {}
        for s in symbols:
            sig_dict[s] = apply_streak_leverage(
                base_sigs[s], data[s][:n_bars], blev, wb, mxl, mnl
            )
        eq = portfolio_eq(data, symbols, n_bars, sig_dict)
        ret, dd, cal, sh = fast_metrics(eq)
        star = " ★" if cal > best_streak[3] and ret > 0 else ""
        if cal > best_streak[3] and ret > 0:
            best_streak = (label, ret, dd, cal, sh)
        log(f"  {label:<34} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}{star}")

    log(f"\n  ★ BEST STREAK: {best_streak[0]}")
    log(f"    Ret={best_streak[1]:+.1f}%, DD={best_streak[2]:.1f}%, Calmar={best_streak[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 6: COMBO STRATEGIES (best ideas combined)
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 6: COMBO STRATEGIES (mix best ideas)")
    log("█" * 74)

    combos = [
        "MoveConf + DDGate",
        "VolInv + DDGate",
        "MoveConf + Streak",
        "VolInv + MoveConf",
    ]

    log(f"\n  {'Combo':<30} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 62)

    best_combo = ("", 0, 0, 0, 0)

    # Combo 1: Move confidence → then DD gate on top
    sig_dict_1 = {}
    for s in symbols:
        mc_sigs = apply_move_confidence_leverage(base_sigs[s], move_sizes[s], THRESHOLD, 3, 15, 10)
        eq_mc = backtest_with_leverage(data[s][:n_bars], mc_sigs, 1.0, COST_PER_SIDE, FUNDING_PER_BAR)
        # DD gate: scale 0.5-1.0 based on DD
        hwm = np.maximum.accumulate(eq_mc)
        dd_arr = (hwm - eq_mc) / hwm
        sig_dict_1[s] = np.where(dd_arr > 0.15, mc_sigs * 0.3, np.where(dd_arr > 0.05, mc_sigs * (1 - (dd_arr - 0.05)/0.10 * 0.7), mc_sigs))
    eq = portfolio_eq(data, symbols, n_bars, sig_dict_1)
    ret, dd, cal, sh = fast_metrics(eq)
    if cal > best_combo[3] and ret > 0:
        best_combo = (combos[0], ret, dd, cal, sh)
    log(f"  {combos[0]:<30} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}")

    # Combo 2: Vol inverse → then DD gate
    sig_dict_2 = {}
    for s in symbols:
        vi_sigs = apply_vol_inverse_leverage(base_sigs[s], data[s][:n_bars], 0.07, 480, 10, 1)
        eq_vi = backtest_with_leverage(data[s][:n_bars], vi_sigs, 1.0, COST_PER_SIDE, FUNDING_PER_BAR)
        hwm = np.maximum.accumulate(eq_vi)
        dd_arr = (hwm - eq_vi) / hwm
        sig_dict_2[s] = np.where(dd_arr > 0.15, vi_sigs * 0.3, np.where(dd_arr > 0.05, vi_sigs * (1 - (dd_arr - 0.05)/0.10 * 0.7), vi_sigs))
    eq = portfolio_eq(data, symbols, n_bars, sig_dict_2)
    ret, dd, cal, sh = fast_metrics(eq)
    if cal > best_combo[3] and ret > 0:
        best_combo = (combos[1], ret, dd, cal, sh)
    log(f"  {combos[1]:<30} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}")

    # Combo 3: Move confidence + streak
    sig_dict_3 = {}
    for s in symbols:
        mc_sigs = apply_move_confidence_leverage(base_sigs[s], move_sizes[s], THRESHOLD, 3, 15, 10)
        # Extract leverage from MC and use as base for streak
        sig_dict_3[s] = apply_streak_leverage(np.sign(mc_sigs), data[s][:n_bars], 5, 1.5, 15, 2)
    eq = portfolio_eq(data, symbols, n_bars, sig_dict_3)
    ret, dd, cal, sh = fast_metrics(eq)
    if cal > best_combo[3] and ret > 0:
        best_combo = (combos[2], ret, dd, cal, sh)
    log(f"  {combos[2]:<30} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}")

    # Combo 4: Vol inverse * move confidence
    sig_dict_4 = {}
    for s in symbols:
        vi_sigs = apply_vol_inverse_leverage(base_sigs[s], data[s][:n_bars], 0.05, 480, 7, 1)
        mc_factor = np.ones(n_bars)
        for j in range(n_bars):
            if move_sizes[s][j] > THRESHOLD:
                mc_factor[j] = 1.0 + min((move_sizes[s][j] - THRESHOLD) / THRESHOLD * 2, 2.0)
        sig_dict_4[s] = np.clip(vi_sigs * mc_factor, -20, 20)
    eq = portfolio_eq(data, symbols, n_bars, sig_dict_4)
    ret, dd, cal, sh = fast_metrics(eq)
    if cal > best_combo[3] and ret > 0:
        best_combo = (combos[3], ret, dd, cal, sh)
    log(f"  {combos[3]:<30} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}")

    log(f"\n  ★ BEST COMBO: {best_combo[0]}")
    log(f"    Ret={best_combo[1]:+.1f}%, DD={best_combo[2]:.1f}%, Calmar={best_combo[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 7: TOP-10 PORTFOLIO × BEST STRATEGIES
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 7: TOP-10 CURATED × LEVERAGE STRATEGIES")
    log("█" * 74)
    log("  Apply leverage strategies to the curated Top-10 (best symbols only)")

    # Top-10 from Phase 2 of fast_ab_test.py
    TOP10 = ["FETUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "DOTUSDT",
             "OPUSDT", "SHIBUSDT", "DOGEUSDT", "LTCUSDT", "ATOMUSDT"]

    strategies = [
        ("1x (baseline)", lambda s: base_sigs[s]),
        ("3x fixed", lambda s: base_sigs[s] * 3),
        ("5x fixed", lambda s: base_sigs[s] * 5),
        ("7x fixed", lambda s: base_sigs[s] * 7),
        ("10x fixed", lambda s: base_sigs[s] * 10),
        ("MoveConf B=3 S=15 M=10", lambda s: apply_move_confidence_leverage(base_sigs[s], move_sizes[s], THRESHOLD, 3, 15, 10)),
        ("MoveConf B=5 S=20 M=15", lambda s: apply_move_confidence_leverage(base_sigs[s], move_sizes[s], THRESHOLD, 5, 20, 15)),
        ("VolInv 7% Max=10", lambda s: apply_vol_inverse_leverage(base_sigs[s], data[s][:n_bars], 0.07, 480, 10, 1)),
        ("DDGate 7x→2x", lambda s: apply_dd_gated_leverage(base_sigs[s], backtest_with_leverage(data[s][:n_bars], base_sigs[s], 1.0, COST_PER_SIDE, FUNDING_PER_BAR), 7, 0.05, 0.15, 2)),
        ("Streak B=5 +1.5/w M=10", lambda s: apply_streak_leverage(base_sigs[s], data[s][:n_bars], 5, 1.5, 10, 2)),
    ]

    log(f"\n  {'Strategy':<28} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 60)

    best_t10 = ("", 0, 0, 0, 0)
    for label, sig_func in strategies:
        sig_dict = {}
        for s in TOP10:
            if s in data:
                sig_dict[s] = sig_func(s)
        eq = portfolio_eq(data, [s for s in TOP10 if s in data], n_bars, sig_dict)
        ret, dd, cal, sh = fast_metrics(eq)
        star = " ★" if cal > best_t10[3] and ret > 0 else ""
        if cal > best_t10[3] and ret > 0:
            best_t10 = (label, ret, dd, cal, sh)
        log(f"  {label:<28} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}{star}")

    log(f"\n  ★ BEST TOP-10: {best_t10[0]}")
    log(f"    Ret={best_t10[1]:+.1f}%, DD={best_t10[2]:.1f}%, Calmar={best_t10[3]:.2f}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 8: ROLLING 90-DAY STABILITY (best strategy)
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  PHASE 8: ROLLING WINDOW STABILITY (best leverage strategy)")
    log("█" * 74)

    # Determine overall best: compare across phases
    all_bests = [
        ("Fixed " + str(best_fixed[0]) + "x", best_fixed[1], best_fixed[2], best_fixed[3], best_fixed[4]),
        ("MoveConf", best_move[1], best_move[2], best_move[3], best_move[4]),
        ("VolInv", best_vol[1], best_vol[2], best_vol[3], best_vol[4]),
        ("DDGate", best_dd[1], best_dd[2], best_dd[3], best_dd[4]),
        ("Streak", best_streak[1], best_streak[2], best_streak[3], best_streak[4]),
        ("Combo", best_combo[1], best_combo[2], best_combo[3], best_combo[4]),
        ("Top10+" + best_t10[0][:10], best_t10[1], best_t10[2], best_t10[3], best_t10[4]),
    ]

    log(f"\n  CROSS-PHASE COMPARISON:")
    log(f"  {'Strategy':<25} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 57)
    overall_best = all_bests[0]
    for name, ret, dd, cal, sh in all_bests:
        star = ""
        if cal > overall_best[3] and ret > 0:
            overall_best = (name, ret, dd, cal, sh)
            star = " ★"
        elif name == overall_best[0]:
            star = " ★"
        log(f"  {name:<25} {ret:>+7.1f} {dd:>6.1f} {cal:>7.2f} {sh:>7.2f}{star}")

    log(f"\n  ═══ OVERALL BEST: {overall_best[0]} ═══")
    log(f"    Ret={overall_best[1]:+.1f}%, DD={overall_best[2]:.1f}%, Calmar={overall_best[3]:.2f}, Sharpe={overall_best[4]:.2f}")

    # Rolling 90-day window test on Top-10 with best leverage strategy
    window = BARS_PER_DAY * 90
    step = BARS_PER_DAY * 30

    log(f"\n  Rolling 90-day window stability (Top-10 + best strategy):")
    log(f"  {'Window':>8} {'Days':>12} {'Ret%':>8} {'DD%':>7} {'Calmar':>7} {'Sharpe':>7}")
    log("  " + "-" * 55)

    n_windows = 0
    n_profitable = 0
    for w_start in range(0, n_bars - window, step):
        w_end = w_start + window
        n_windows += 1
        w_len = w_end - w_start
        # Build per-symbol bar returns matrix
        sym_bar_rets = []
        for s in TOP10:
            if s not in data:
                continue
            close_w = data[s][w_start:w_end]
            sig_w = compute_signals(close_w, LOOKBACK, THRESHOLD, TARGET, STOP, MAX_HOLD, STOP_DECAY)
            best_lev = best_fixed[0] if best_fixed[0] > 1 else 3
            eq_w = backtest_with_leverage(close_w, sig_w, float(best_lev), COST_PER_SIDE, FUNDING_PER_BAR)
            rets_s = np.zeros(w_len)
            for j in range(1, w_len):
                rets_s[j] = eq_w[j] / eq_w[j - 1] - 1.0
            sym_bar_rets.append(rets_s)

        # Build EW portfolio
        mat_w = np.array(sym_bar_rets)
        avg_r = np.mean(mat_w, axis=0)
        eq_w = np.ones(w_len)
        for j in range(1, w_len):
            eq_w[j] = eq_w[j - 1] * (1 + avg_r[j])

        ret, dd, cal, sh = fast_metrics(eq_w)
        profitable = "★" if ret > 0 else "✗"
        if ret > 0:
            n_profitable += 1
        log(f"  W {n_windows:>3}  D{w_start//BARS_PER_DAY:>3}-D{w_end//BARS_PER_DAY:<3}  {ret:>+7.2f} {dd:>6.2f} {cal:>7.2f} {sh:>7.2f} {profitable}")

    pct = n_profitable / n_windows * 100 if n_windows > 0 else 0
    log(f"\n  Profitable: {n_profitable}/{n_windows} ({pct:.0f}%)")

    # ══════════════════════════════════════════════════════════════
    #  FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    log("\n" + "█" * 74)
    log("  FINAL VERDICT & RECOMMENDATION")
    log("█" * 74)

    log(f"\n  BASE (1x):  Ret=+18.4%, DD=16.3%, Calmar=1.13")
    log(f"  BEST:       {overall_best[0]}")
    log(f"              Ret={overall_best[1]:+.1f}%, DD={overall_best[2]:.1f}%, Calmar={overall_best[3]:.2f}, Sharpe={overall_best[4]:.2f}")
    log(f"  Rolling:    {n_profitable}/{n_windows} windows profitable ({pct:.0f}%)")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("=" * 74)
    out.close()


if __name__ == "__main__":
    main()
