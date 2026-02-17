"""
Concurrent Slot Analysis — How many slots do we ACTUALLY need?
Simulates v9.0 signals across all 30 symbols and measures:
1. Per-symbol trade count & performance
2. Concurrent position demand over time
3. Slot-constrained portfolio simulation (5, 8, 10, 15, 20 slots)
4. Revenue impact of slot limits
"""
import numpy as np
import pandas as pd
from pathlib import Path
from quantbox.strategy.rubberband import compute_signals_v9
from quantbox.backtest.vectorized_backtest import _backtest_core

CACHE = Path("data/cache_extended")

# v9.0 production params
LB, TH, TR, SM, MH, SD = 8, 0.045, 0.382, 0.786, 10, 0.0
BTC_MOM_P = 0  # btc_mom_period (legacy HTF, disabled)
REGIME_P, TREND_P, MIN_CONF = 96, 192, 2
BASE_LEV, CONF_LEV_STEP, MAX_LEV = 1.5, 0.0, 1.5
RSI_P, VOL_P = 14, 20
DYN_TARGET = False
TARGET_C3, TARGET_C4, TARGET_C5 = 0.45, 0.50, 0.55
MACD_EN, MACD_F, MACD_S, MACD_SIG = True, 12, 26, 14
STOCH_EN, STOCH_K, STOCH_D = True, 5, 3
COST, FUNDING = 0.0004, 0.0001 / 32


def load_data(sym: str):
    """Load 15m and 4h data for a symbol."""
    f15 = CACHE / f"{sym}_15m_365d.pkl.gz"
    f4h = CACHE / f"{sym}_4h_365d.pkl.gz"
    if not f15.exists():
        return None
    df = pd.read_pickle(f15)
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    vol = df["Volume"].values.astype(np.float64)
    taker = df["Taker Buy Base"].values.astype(np.float64)
    close_4h = (
        pd.read_pickle(f4h)["Close"].values.astype(np.float64)
        if f4h.exists()
        else np.ones(100)
    )
    return close, high, low, vol, taker, close_4h


def run_signals(close, high, low, vol, taker, close_4h, btc, btc_4h):
    """Run v9.0 signal generation."""
    n = len(close)
    btc_n = btc[:n] if len(btc) >= n else np.concatenate(
        [btc, np.full(n - len(btc), btc[-1])]
    )
    return compute_signals_v9(
        close, high, low, vol, taker, btc_n,
        LB, TH, TR, SM, MH, SD,
        BTC_MOM_P,
        REGIME_P, TREND_P, MIN_CONF,
        BASE_LEV, CONF_LEV_STEP, MAX_LEV,
        RSI_P, VOL_P,
        DYN_TARGET, TARGET_C3, TARGET_C4, TARGET_C5,
        STOCH_EN, STOCH_K, STOCH_D,
        MACD_EN, MACD_F, MACD_S, MACD_SIG,
    )


def extract_trades(signals):
    """Extract (entry_bar, exit_bar, direction) tuples from signal array."""
    trades = []
    in_trade = False
    entry_bar = 0
    direction = 0
    for i in range(len(signals)):
        if not in_trade and signals[i] != 0:
            in_trade = True
            entry_bar = i
            direction = int(signals[i])
        elif in_trade and signals[i] == 0:
            in_trade = False
            trades.append((entry_bar, i, direction))
    return trades


def simulate_slot_constrained(all_entries, max_slots, n_bars):
    """Simulate slot-constrained portfolio.

    all_entries: list of (bar, sym_idx, trade_idx, move_strength) for entries
    Returns: number of accepted trades, number of rejected trades.
    """
    # Track how many slots are occupied at each bar
    slots_used = np.zeros(n_bars, dtype=np.int32)
    accepted = 0
    rejected = 0

    # Sort entries by bar, then by move_strength descending
    sorted_entries = sorted(all_entries, key=lambda x: (x[0], -x[3]))

    for bar, sym_idx, trade_idx, move_str in sorted_entries:
        if slots_used[bar] < max_slots:
            # Accept: mark bars as occupied
            _, exit_bar, _ = trade_data[sym_idx][trade_idx]
            for b in range(bar, min(exit_bar, n_bars)):
                slots_used[b] += 1
            accepted += 1
        else:
            rejected += 1

    return accepted, rejected, slots_used


# ── Main ──
print("=" * 78)
print("  Concurrent Slot Analysis — v9.0 across ALL 30 symbols")
print("=" * 78)

# Load BTC
btc_data = load_data("BTCUSDT")
btc = btc_data[0]  # close
btc_4h_data = pd.read_pickle(CACHE / "BTCUSDT_4h_365d.pkl.gz")
btc_4h = btc_4h_data["Close"].values.astype(np.float64)

# Get all non-BTC symbols
all_files = sorted(CACHE.glob("*_15m_365d.pkl.gz"))
all_syms = [f.stem.split("_")[0] for f in all_files if "BTCUSDT" not in f.stem]
print(f"Symbols: {len(all_syms)}")

# Run signals per symbol
results = []
trade_data = {}  # sym_idx -> list of (entry, exit, direction)
all_signals = {}  # sym -> signals array
sym_closes = {}  # sym -> close array

for idx, sym in enumerate(all_syms):
    data = load_data(sym)
    if data is None:
        continue
    close, high, low, vol, taker, close_4h = data

    signals = run_signals(close, high, low, vol, taker, close_4h, btc, btc_4h)
    result = _backtest_core(close, signals, COST, FUNDING)
    # result = (equity, total_return, sharpe, max_dd, win_rate,
    #           n_trades, bars_in_market, profit_factor, avg_ret, trade_returns)
    equity_curve = result[0]
    ret = float(result[1])
    dd = float(result[3])
    n_trades = int(result[5])
    wr = float(result[4])
    calmar = ret / dd if dd > 0 else 0.0
    results.append((sym, ret * 100, dd * 100, calmar, n_trades, wr))

    trades = extract_trades(signals)
    trade_data[idx] = trades
    all_signals[sym] = signals
    sym_closes[sym] = close

# ── Concurrent position analysis ──
n_bars = 35040  # 365 days × 96 bars/day
concurrent = np.zeros(n_bars, dtype=np.int32)
all_entries = []

for idx, sym in enumerate(all_syms):
    if idx not in trade_data:
        continue
    close = sym_closes.get(sym)
    if close is None:
        continue
    for t_idx, (entry, exit_b, direction) in enumerate(trade_data[idx]):
        for b in range(entry, min(exit_b, n_bars)):
            concurrent[b] += 1
        # Move strength for prioritization
        lb = min(LB, entry)
        if lb > 0 and entry > 0:
            move = abs(close[entry] / close[entry - lb] - 1.0)
        else:
            move = 0.0
        all_entries.append((entry, idx, t_idx, move))

total_trades = sum(int(r[4]) for r in results)
peak = int(np.max(concurrent))
active_mask = concurrent > 0
avg_conc = float(np.mean(concurrent[active_mask])) if np.any(active_mask) else 0
p50 = int(np.percentile(concurrent[active_mask], 50)) if np.any(active_mask) else 0
p95 = int(np.percentile(concurrent[active_mask], 95)) if np.any(active_mask) else 0

print(f"\n{'=' * 78}")
print(f"  SECTION 1: Unconstrained Results (no slot limit)")
print(f"{'=' * 78}")
print(f"Total trades (all symbols): {total_trades}")
print(f"Peak concurrent positions: {peak}")
print(f"Avg concurrent (when active): {avg_conc:.1f}")
print(f"Median concurrent: {p50}")
print(f"95th pctile concurrent: {p95}")
sum_ret = sum(r[1] for r in results)
avg_calmar = np.mean([r[3] for r in results])
print(f"Sum of per-symbol returns: {sum_ret:.1f}%")
print(f"Avg Calmar (per-symbol): {avg_calmar:.2f}")

# Concurrency distribution
from collections import Counter
c = Counter(concurrent[active_mask].tolist())
print(f"\nConcurrency distribution (active bars only):")
for k in sorted(c.keys()):
    bars = c[k]
    pct = bars / sum(c.values()) * 100
    bar_str = "█" * max(1, int(pct / 2))
    print(f"  {k:>2} simultaneous: {bars:>5} bars ({pct:>5.1f}%) {bar_str}")

# ── Per-symbol ranking ──
print(f"\n{'=' * 78}")
print(f"  SECTION 2: Per-symbol Performance (ranked by Calmar)")
print(f"{'=' * 78}")
results.sort(key=lambda x: -x[3])
print(f"{'Sym':<12} {'Ret%':>7} {'DD%':>7} {'Cal':>7} {'#T':>5} {'WR%':>6}")
print("-" * 50)
profitable = 0
for sym, ret, dd, cal, nt, wr in results:
    flag = "✓" if cal > 0 else "✗"
    print(f"{sym:<12} {ret:>7.1f} {dd:>7.1f} {cal:>7.2f} {nt:>5} {wr:>6.1f} {flag}")
    if ret > 0:
        profitable += 1
print(f"\nProfitable symbols: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")

# ── Slot-constrained simulation ──
print(f"\n{'=' * 78}")
print(f"  SECTION 3: Slot-Constrained Portfolio Simulation")
print(f"{'=' * 78}")
print(f"  (Equal-weight capital per slot, prioritize by move_strength)\n")

for max_slots in [3, 5, 8, 10, 15, 20, 29]:
    # Simple simulation: count how many bars are over limit
    bars_over = int(np.sum(concurrent > max_slots))
    pct_over = bars_over / n_bars * 100

    # Estimate trade loss from slot constraint
    # When concurrent > max_slots, the excess trades get rejected
    excess_bars = np.maximum(concurrent - max_slots, 0)
    total_excess_bar_trades = int(np.sum(excess_bars))
    total_bar_trades = int(np.sum(concurrent))
    rejection_rate = total_excess_bar_trades / total_bar_trades * 100 if total_bar_trades > 0 else 0

    # Capital efficiency: what fraction of capital is deployed
    utilized = np.minimum(concurrent, max_slots) / max_slots
    avg_utilization = float(np.mean(utilized)) * 100

    print(
        f"  max_slots={max_slots:>2}: "
        f"bars_over={bars_over:>5} ({pct_over:>4.1f}%)  "
        f"bar·trade rejection={rejection_rate:>5.1f}%  "
        f"capital_util={avg_utilization:>4.1f}%"
    )

# ── Capital deployment analysis ──
print(f"\n{'=' * 78}")
print(f"  SECTION 4: Optimal Slot Count Analysis")
print(f"{'=' * 78}")
print(f"  Key insight: More slots = more trades captured, but each gets less capital")
print(f"  Sweet spot = maximize (trades_captured × per_slot_return)\n")

# For each slot count, estimate portfolio return
# Each slot gets 1/N of capital. Return = sum of per-trade returns × (1/N)
# With slot constraint, some trades are rejected (weakest move_strength first)
for max_slots in [3, 5, 8, 10, 15, 20]:
    per_slot_frac = 1.0 / max_slots
    # How many bars have trades we can take?
    capped = np.minimum(concurrent, max_slots)
    # Fraction of all trade-bars we capture
    captured_pct = float(np.sum(capped)) / float(np.sum(concurrent)) * 100
    # Effective capital deployed per slot
    # With leverage 1.5x, effective exposure = per_slot_frac * 1.5
    exposure = per_slot_frac * 1.5
    
    print(
        f"  slots={max_slots:>2}: "
        f"per_slot={per_slot_frac*100:>5.1f}% capital  "
        f"leverage_exposure={exposure*100:>5.1f}%  "
        f"trades_captured={captured_pct:>5.1f}%  "
        f"effective_multiplier={captured_pct/100 * max_slots * exposure:>5.2f}x"
    )

print(f"\n{'=' * 78}")
print(f"  SECTION 5: Revenue Estimation (per $10,000 capital)")
print(f"{'=' * 78}")
# Average per-trade return across all symbols
all_per_trade_rets = []
for sym, ret, dd, cal, nt, wr in results:
    if nt > 0:
        all_per_trade_rets.append(ret / nt)  # avg % return per trade
avg_per_trade = np.mean(all_per_trade_rets)
print(f"  Avg per-trade return (across all symbols): {avg_per_trade:.3f}%")
print(f"  Total annual trades (unconstrained): {total_trades}")
print()

capital = 10000
for max_slots in [5, 8, 10, 15]:
    per_slot = capital / max_slots
    # Estimate captured trades
    capped = np.minimum(concurrent, max_slots)
    capture_ratio = float(np.sum(capped)) / float(np.sum(concurrent))
    est_trades = int(total_trades * capture_ratio)
    # Per-trade PnL with leverage
    per_trade_pnl = per_slot * 1.5 * (avg_per_trade / 100)
    est_annual = per_trade_pnl * est_trades
    est_ret_pct = est_annual / capital * 100
    print(
        f"  slots={max_slots:>2}: "
        f"~{est_trades:>4} trades/yr  "
        f"${per_trade_pnl:>6.1f}/trade  "
        f"est.annual=${est_annual:>8.0f} ({est_ret_pct:>6.1f}%)"
    )

print()
