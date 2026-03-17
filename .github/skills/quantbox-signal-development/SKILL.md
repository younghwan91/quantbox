---
name: quantbox-signal-development
description: |
  Workflow for developing and validating new trading signals: data download → engine selection → baseline validation.
  Use when creating new signal functions or testing signal hypotheses.
  Triggers: signal development, numba, signal generation, data download, backtest baseline, jit
license: MIT
---

# Signal Development Workflow

## Purpose

Signals are the core logic driving trade entry. This workflow ensures signal correctness before backtest/live deployment.

## Complete Workflow (4 Phases)

### Phase 1: Data Preparation
```python
from quantbox.data.downloader import BinanceKlineDownloader
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# Initialize downloader with rate limiting
cfg = VolStraddleConfig()
downloader = BinanceKlineDownloader(rate_limit_qpm=2400)

# Download for all symbols in whitelist (don't hardcode)
symbols = cfg.symbol_whitelist  # ← Use config, never hardcode
klines = {}
for symbol in symbols:
    klines[symbol] = downloader.download_klines(symbol, interval='15m', limit=1000)
    print(f"Loaded {len(klines[symbol])} bars for {symbol}")

# Verify data integrity
for symbol, bars in klines.items():
    assert len(bars) >= cfg.warmup_15m, f"{symbol} has insufficient warmup bars"
    assert bars[-1]['time'] <= now, f"{symbol} has future bars (lookahead risk)"
```

**Checklist**:
- ❏ Data sourced from Binance API (no synthetic/mock data)
- ❏ Symbols sourced from `VolStraddleConfig().symbol_whitelist` (never hardcoded)
- ❏ Download includes sufficient warmup period (`cfg.warmup_15m` = 200 bars by default)
- ❏ Most recent bar timestamp < now (no forward-looking data)

### Phase 2: Signal Function Implementation

#### Option A: Single-Timeframe Signal (15m)
```python
import numba

@numba.jit(nopython=True)
def my_signal_15m(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """
    Generate binary entry signal (1 = long bias, -1 = short bias, 0 = neutral).

    Args:
        closes, highs, lows: OHLC bar data (completed bars only, no lookahead)

    Returns:
        signal: binary array of same length as inputs
    """
    n = len(closes)
    signal = np.zeros(n, dtype=np.int64)

    for i in range(20, n):  # Start after warmup
        # Example: simple SMA crossover
        sma_short = np.mean(closes[i-5:i])
        sma_long = np.mean(closes[i-20:i])

        if sma_short > sma_long:
            signal[i] = 1
        elif sma_short < sma_long:
            signal[i] = -1

    return signal

# Verify Numba compiles
signal_15m = my_signal_15m(klines['SUIUSDT']['close'], ...)
assert len(signal_15m) == len(klines['SUIUSDT'])
print("✓ Numba JIT compiled successfully")
```

#### Option B: Multi-Timeframe Signal (15m + 1h + 4h)
```python
@numba.jit(nopython=True)
def my_signal_mtf(closes_15m, closes_1h, closes_4h, ratio_1h=4, ratio_4h=16) -> np.ndarray:
    """Multi-TF signal with completed bar rule."""
    n = len(closes_15m)
    signal = np.zeros(n, dtype=np.int64)

    for i in range(50, n):  # Warmup
        # 15m baseline
        sma_15m = np.mean(closes_15m[i-5:i])

        # 1h confirmation: MUST use completed bar
        j_1h = (i // ratio_1h) - 1  # ← Critical: -1 for completed bar
        if j_1h >= 20:  # Ensure enough data
            sma_1h = np.mean(closes_1h[j_1h-20:j_1h])
        else:
            continue

        # 4h trend: MUST use completed bar
        j_4h = (i // ratio_4h) - 1  # ← Critical: -1 for completed bar
        if j_4h >= 5:
            trend_4h = closes_4h[j_4h] > np.mean(closes_4h[j_4h-5:j_4h])
        else:
            continue

        # Multi-TF logic
        if sma_15m > sma_1h and trend_4h:
            signal[i] = 1
        elif sma_15m < sma_1h and not trend_4h:
            signal[i] = -1

    return signal
```

**Critical Rule**: For multi-TF, always use `j = (i // ratio) - 1` (subtract 1 for completed bar). See CLAUDE.md Rule #2.

**Checklist**:
- ❏ Function decorated with `@numba.jit(nopython=True)`
- ❏ Only numpy operations (no loops that can't be JIT'd)
- ❏ Multi-TF uses `j = (i // ratio) - 1` (not `j = i // ratio`)
- ❏ No lookahead: no access to `closes[i+1]` or future bars
- ❏ Warmup period handled: signals only from `i >= warmup`

### Phase 3: Baseline Validation

Compare against existing working signal for regression.

```python
# Load existing baseline signal
from quantbox.strategy.vol_straddle.signals import gen_vr_signal

baseline_signal = gen_vr_signal(closes_15m, highs_15m, lows_15m)
my_signal = my_signal_15m(closes_15m, highs_15m, lows_15m)

# Compare agreement
agreement = np.mean(baseline_signal == my_signal) * 100
print(f"Agreement with baseline: {agreement:.1f}%")

# Minimum acceptable agreement: 70% (some novelty expected)
# If < 50%, something is wrong (lookahead, wrong data, off-by-one)
assert agreement > 50, "Signal diverges too much from baseline"
```

### Phase 4: Backtest with Canonical Engine

Run trade_sim with your signal.

```python
from quantbox.backtest.trade_sim import collect_trades, portfolio_sim, compute_metrics
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

cfg = VolStraddleConfig()
cost = cfg.round_trip_cost * 2  # 4 sides for straddle

# Run backtest
trades = collect_trades(
    symbol='SUIUSDT',
    klines=klines['SUIUSDT'],
    signals=my_signal,
    params={'cost': cost}
)

# Portfolio-level sim (slot contention)
pf, ev, calmar, dd = portfolio_sim(trades, slot_limit=cfg.max_concurrent)

# Acceptance criteria
print(f"Profit Factor: {pf:.2f}")
print(f"Expected Value: {ev:.4f}")
print(f"Calmar Ratio: {calmar:.2f}")
print(f"Max Drawdown: {dd:.1%}")

# Minimum viable signal
assert pf >= 1.5, "Profit factor too low (< 1.5)"
assert ev > 0.0005, "Expected value too low (< 0.05%)"
assert calmar > 1.0, "Calmar ratio too low (< 1.0)"
```

## Decision Tree

```
Developing a new signal?

├─ Single-timeframe (15m only)?
│  ├─ YES: Use Phase 1-4 (simpler path)
│  └─ NO: Continue
│
├─ Multi-timeframe (15m + 1h or 4h)?
│  ├─ YES: Use Phase 2B with j = (i // ratio) - 1 rule
│  └─ Critical: Any off-by-one error causes lookahead
│
├─ Before backtest, ask:
│  ├─ Is data real (from Binance) or synthetic?
│  ├─ Are symbols from config or hardcoded?
│  ├─ Is warmup period sufficient?
│  └─ Does baseline agreement > 50%?
│
└─ If all yes: Proceed to production
```

## Test Command

```bash
# Unit test (Numba compilation + correctness)
uv run pytest tests/test_signals_unit.py -v

# Integration test (backtest parity)
uv run pytest tests/test_backtest_live_parity.py -v

# Multi-TF lookahead detection
uv run pytest tests/test_mtf_lookahead.py -v
```

## References
- **Signal functions**: `quantbox/strategy/vol_straddle/signals.py` (9 Numba JIT functions)
- **Signal reference**: `quantbox/strategy/vol_straddle/AGENTS.md` (lines 157-209)
- **4-phase research structure**: `.github/instructions/backtesting.instructions.md` Section 9 (lines 288-317)
- **Multi-TF rule**: CLAUDE.md Rule #2 (line 99)
- **Tests**: `tests/test_signals_unit.py`, `tests/test_mtf_lookahead.py`
- **Config**: `quantbox/strategy/vol_straddle/config.py` (VolStraddleConfig class)

## When to Use This Skill

- Creating a new signal function for VolStraddle strategy
- Debugging signal output (use baseline comparison)
- Migrating a signal from RubberBand to VolStraddle
- Validating multi-TF signal logic (high lookahead risk)
- Before deploying signal changes to live trading
