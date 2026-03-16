<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-16 -->

# quantbox/backtest

## Purpose

Dual backtesting engines with strict parity to live trading. The canonical engine (TradeEventSim) models discrete trades with intra-bar SL/TP resolution and portfolio slot contention. VectorizedBacktest is a fast alternative for signal-array strategies. All backtests use real Binance data; no lookahead; no synthetic pricing.

## Files

| File | Purpose | Key Exports |
| --- | --- | --- |
| `trade_sim.py` | TradeEventSim — discrete trade simulation with intra-bar SL/TP | `collect_trades()`, `portfolio_sim()`, `compute_metrics()` |
| `vectorized_backtest.py` | VectorizedBacktest — signal array engine (close-to-close PnL) | `VectorizedBacktest`, `BacktestConfig`, `BacktestResult` |
| `__init__.py` | Module exports | — |

## Dual Engines: When to Use Which

### TradeEventSim (Canonical)

**Use for:**

- VolStraddle (VR Squeeze) strategy
- Any strategy with explicit stop-loss and take-profit exits
- Research requiring portfolio slot contention modeling
- Production-grade backtest metrics (PF, EV, Calmar ratio, etc.)

**Characteristics:**

- Discrete trade events (entry, SL hit, TP hit, max-hold exit)
- Intra-bar SL/TP resolution using high/low prices
- Cross-symbol slot contention (portfolio_sim)
- Supports straddle pairs (simultaneous LONG + SHORT)
- Cost model: 0.05% fee + slippage (round-trip = 0.0008 per side)

**Result:** Production metrics matching live trading reality.

### VectorizedBacktest (Fast)

**Use for:**

- RubberBand strategy
- Strategies that output a pre-computed ±1 position series
- Quick exploratory backtests
- Signal array testing

**Characteristics:**

- Close-to-close PnL only (no intra-bar SL/TP)
- Signal array input (must be pre-baked with all entry/exit logic)
- Per-symbol backtests (no portfolio slot contention)
- Fast execution (100 symbols × 5 years in seconds)

**Result:** Quick feedback on signal quality; not suitable for detailed metrics.

## Parity Doctrine

**Core principle:** Backtest logic ≡ live logic. Zero lookahead. Completed bars only.

### No Lookahead

- Backtest never uses future prices
- Signals based on prices up to bar close
- SL/TP exits evaluated using only O/H/L/C of current bar
- Entry cannot happen until bar closes (no mid-bar entry)

### Completed Bars Only

- All signal arrays and prices are for **closed** candles
- Current (still-forming) bar is skipped
- Cross-TF references use completed bar indices: `j = (i // ratio) - 1`

**Example:** At 15m bar 100 (closed), 1h bar index is `(100 // 4) - 1 = 24` (bar 24 is closed; bar 25 is current).

### No Hardcoded Parameters

- All config params (leverage, SL/TP distance, max-hold bars, costs) come from `VolStraddleConfig`
- Research scripts read config dynamically, never hardcode
- Changing `.env` updates config; backtest picks it up on next run

## Cost Model

**Fee:** 0.02% per side (Binance maker rate)

**Slippage:** 0.02% per side (limit order execution cost)

**Round-trip cost (entry + exit):** 0.0008 per notional (2 × 0.0004)

**Straddle (4-sided) cost:** 0.0016 per notional (4 × 0.0004)

**Application:**

- Direct trades: multiply notional by cost
- Straddle pairs: both LONG and SHORT share portfolio capital; total cost is 4× (entry long, exit long, entry short, exit short)
- Trailing stops: cost only applied to filled portion

## Research Integration

**Canonical pipeline:**

1. Research script generates signals via `gen_vr_signal()` or similar
2. Import `collect_trades()` and `portfolio_sim()` from `trade_sim.py`
3. Simulate trades across all symbols and timeframes
4. Call `compute_metrics()` for final results
5. Commit result as `(ref: rNN)` — trace lineage back to research branch

**Golden rule:** Never reimplement `collect_trades()` or `portfolio_sim()`. Subtle details (momentum=0 handling, squeeze counter reset, trade dedup order) make it easy to produce different numbers.

**Code template:**

```python
from quantbox.backtest.trade_sim import (
    collect_trades, portfolio_sim, compute_metrics
)
from quantbox.strategy.vol_straddle.signals import gen_vr_signal

# Generate signals for all symbols/TFs
trades = collect_trades(signals_dict, symbol_list, tf_list)

# Filter by slot constraints
portfolio_trades = portfolio_sim(trades, max_concurrent=10)

# Compute metrics
metrics = compute_metrics(portfolio_trades)
print(f"Profit Factor: {metrics['profit_factor']}")
```

## Critical Rules for Agents

1. **Trade_sim.py is canonical.** Research must use `collect_trades()` and `portfolio_sim()` from here. Never reimplement them locally.

2. **No lookahead.** Signals must be computed using only data up to bar close. Never use future prices, never skip forward.

3. **Completed bars only.** If current bar index is 100, use data from bars 0-99 for signals. Skip bar 100 (still forming).

4. **Cross-TF indices use completed bar formula.** At 15m bar i, the corresponding 1h bar is `j = (i // 4) - 1`. At 15m bar 100, 1h bar is `j = 24` (the 25th bar, indexed from 0).

5. **Symbol whitelist drives data loading.** Download only symbols in `cfg.symbol_whitelist`. Never hardcode symbol lists for backtests.

6. **Cost model is fixed.** Entry + exit cost = 0.0008 per notional. Straddle cost = 0.0016. These are exact; don't approximate.

7. **Intra-bar SL/TP uses high/low.** Long SL is triggered when `low <= sl_price`. Short SL is triggered when `high >= sl_price`. TP similarly uses high/low.

8. **Slot contention is real.** `portfolio_sim()` enforces max_concurrent slot limit. Overlapping trades for different symbols compete for slots. Excess trades are skipped (not queued).

9. **Max-hold timeout is a hard exit.** After max_hold bars (e.g., 192 for 15m = 48h), position closes at next bar's close price, regardless of SL/TP.

10. **Straddle pairs must have matching lifespan.** LONG and SHORT legs of a straddle pair open and close together. Both hit SL/TP at the same time; both exit on max-hold timeout.

11. **Research branches use (ref: rNN) commit messages.** When promoting a research result to main via a feature commit, include `(ref: r42)` in the commit message to trace lineage.

## For AI Agents

### Task: Run a backtest for a symbol

```python
from quantbox.backtest.trade_sim import collect_trades, portfolio_sim, compute_metrics
from quantbox.data.real_data_loader import RealDataLoader
from quantbox.strategy.vol_straddle.signals import gen_vr_signal
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

cfg = VolStraddleConfig()
loader = RealDataLoader()

# Download data
ohlcv = loader.load("BTCUSDT", interval="15m", n_bars=5000)
close = ohlcv[:, 4]

# Generate signals
signals = gen_vr_signal(close, cfg.vr_window, cfg.vr_threshold, cfg.momentum_window)

# Collect trades
trades = collect_trades(
    {"BTCUSDT": {"15m": signals}},
    symbol_list=["BTCUSDT"],
    tf_list=["15m"]
)

# Compute metrics
metrics = compute_metrics(trades)
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

### Task: Validate parity (backtest vs live)

1. Run backtest on historical data using `trade_sim.py`
2. Compare to live trade results from `_trade_history` on same date range
3. If PnL differs by > 2%, investigate:
   - Are fees identical? (backtest uses config fee, live may vary)
   - Are SL/TP distances identical?
   - Are the same symbols traded?
   - Did live trades hit slippage not modeled in backtest?

### Task: Add a new signal to signals.py

1. Implement signal function in `strategy/vol_straddle/signals.py`
2. Use Numba `@jit(nopython=True)` for performance
3. Return signal array: -1 (short), 0 (no signal), +1 (long), or +2 (straddle)
4. Test with `tests/test_signals_unit.py` (unit tests, no coverage)
5. Backtest with `collect_trades()` to validate before committing

### Task: Research a parameter variation

1. Create branch: `research/param_study_rNN`
2. Modify `VolStraddleConfig` defaults or environment variables
3. Run backtest via `trade_sim.py` across symbol list and date range
4. Save results to `docs/backtest_results/YYYYMM/YYYYMMDD_short_description.md`
5. Commit with `git commit -m "research(rNN): parameter study — X vs Y results"` (not squashed)
6. If results look good, create feature branch: `feature/param_update`
7. Promote to main with squash: `git merge --squash research/param_study_rNN`
8. Commit message: `feat(signals): update VR window to 50 (ref: r42)`

### Task: Debug a backtest divergence

If backtest metrics don't match live results:

1. Check symbol whitelist: `cfg.symbol_whitelist`
2. Check date range: backtest start/end vs live period
3. Check cost model: backtest fee (0.0004) vs live fee (may vary)
4. Check signal generation: same threshold, window size as live?
5. Check slot contention: does live also enforce max_concurrent?
6. Print trade-by-trade comparison: side, entry_price, exit_price, pnl_pct
7. If still diverged, file issue with specific trade mismatch

## Parity Validation Procedure

**Before promoting backtest results to production config:**

1. Run backtest on 2-year history with current config
2. Run live paper trading on same symbols for 2+ weeks
3. Compare cumulative PnL: should be within 2% (fees, slippage account for 1%)
4. Compare win rate: should be within 1 percentage point
5. Compare max drawdown: should be within 2 percentage points
6. If validated, update production config with confidence

## Testing Considerations

- **Real data only.** All backtest fixtures use real Binance klines from `data/` directory.
- **Completed bar enforcement.** Test that signal generation skips the current (incomplete) bar.
- **Cost model.** Verify all trades include 0.0008 round-trip cost (0.0004 per side).
- **Slot contention.** Test that portfolio_sim() respects max_concurrent limit.
- **SL/TP resolution.** Test intra-bar hits: low touch on long SL, high touch on short SL.
- **Straddle pairing.** Test that LONG and SHORT legs of a straddle pair exit together.
- **Max-hold timeout.** Test that positions close after max_hold bars regardless of PnL.
- **Lookahead detection.** Static analysis: verify no future-bar references in signal code.
