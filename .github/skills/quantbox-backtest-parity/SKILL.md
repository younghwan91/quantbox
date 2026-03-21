---
name: quantbox-backtest-parity
description: |
  Multi-timeframe parity diff workflow for debugging backtest vs live divergence.
  Use when backtest results don't match live trading numbers.
  Triggers: parity, backtest divergence, live mismatch, lookahead, incomplete candle, backtest accuracy
license: MIT
---

# Backtest-Live Parity Verification

## Purpose

Backtest logic MUST equal live logic. This skill provides the diagnostic workflow when they diverge.

## Divergence Workflow (Step-by-Step)

### Step 1: Establish Baseline Mismatch
```python
# Compare backtest PnL vs live PnL
backtest_pnl = 125.43  # From trade_sim run
live_pnl = 87.52       # From Discord alerts + state file

delta_pct = abs(backtest_pnl - live_pnl) / live_pnl * 100
# delta_pct = 43.3% → DIVERGENCE CONFIRMED
```

**Decision**:
- Delta < 5%: Acceptable (fee/slippage variance)
- Delta 5-20%: Investigate cost model or fills
- Delta > 20%: Investigate lookahead, signals, or multi-TF logic

### Step 2: Isolate Timeframe
If divergence > 5%, which TF caused it?

```python
# Run backtest for each TF separately
from quantbox.backtest.trade_sim import collect_trades

# 15m signals only
trades_15m = collect_trades('SUIUSDT', klines_15m, signals_15m, cost=0.0008)
pnl_15m = sum(t['pnl'] for t in trades_15m)

# 1h signals only (requires resampling)
trades_1h = collect_trades('SUIUSDT', klines_1h, signals_1h, cost=0.0008)
pnl_1h = sum(t['pnl'] for t in trades_1h)

# 4h signals only
trades_4h = collect_trades('SUIUSDT', klines_4h, signals_4h, cost=0.0008)
pnl_4h = sum(t['pnl'] for t in trades_4h)

print(f"15m: {pnl_15m}, 1h: {pnl_1h}, 4h: {pnl_4h}")
# One TF will show large divergence
```

### Step 3: Check for Lookahead (Most Common)

Lookahead = using tomorrow's data to signal today. Causes 50%+ backtest overperformance (r55: 83% backtest vs 36% live).

**Checklist**:
- ❏ All multi-TF references use `j = (i // ratio) - 1` (completed bar, not current)?
- ❏ No access to `klines[i+1]` or future bars?
- ❏ Resampling uses `close[j]` for crossovers, not `close[j+1]`?

**Verification command**:
```bash
# Search for off-by-one errors in signal generation
grep -n "// ratio\|ratio\|resample" quantbox/strategy/vol_straddle/signals.py
# Look for any `j + 1` or `j + 2` references to klines
```

**Common lookahead pattern**:
```python
# ✗ WRONG: Current bar (lookahead)
j = i // 4  # Gets current 1h bar
sma = np.mean(klines[j]['close'])  # Includes current bar's close (not finalized)

# ✓ CORRECT: Completed bar
j = (i // 4) - 1  # Gets previous 1h bar (completed)
sma = np.mean(klines[j]['close'])  # Includes only finished candles
```

### Step 4: Check for Incomplete Candles

Live trading only signals on bar CLOSE. Backtest must match.

**Verification**:
```python
# In backtest code:
for i in range(warmup, len(klines)):
    bar = klines[i]
    # At this point, is bar['close'] finalized or still updating?
    # MUST be: finished bar only
```

**Checklist**:
- ❏ Signals generated AFTER bar close, not during bar?
- ❏ Warmup period skips enough bars to avoid sparse data?
- ❏ No signals on the current incomplete bar?

### Step 5: Verify Cost Model Parity

Cost model mismatch causes 5-15% divergence.

```python
# Backtest cost model
from quantbox.backtest.trade_sim import COST_STRADDLE
backtest_cost = COST_STRADDLE  # Should be 0.0016

# Live cost model
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
live_cost = cfg.round_trip_cost * 2  # 4 sides (long+short entry+exit)

# Must match
assert backtest_cost == live_cost, f"Cost mismatch: {backtest_cost} vs {live_cost}"
```

### Step 6: Check Symbol Whitelist

Symbol loading mismatch causes subset divergence.

```python
# Backtest must use whitelist (not exclude_symbols)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
print(f"Backtest symbols: {cfg.symbol_whitelist}")

# Live system uses same whitelist
# Verify no hardcoded symbol lists
grep -n "symbol_whitelist\|exclude" quantbox/strategy/*.py
```

## Quick Checklist (All Points)

```
Backtest vs Live Divergence?
├─ Delta < 5%?
│  └─ ACCEPTABLE (fee/slippage variance, OK to commit)
│
├─ Delta 5-20%?
│  ├─ Check cost model parity (Step 5)
│  ├─ Check symbol whitelist (Step 6)
│  └─ If still divergent, check fills + slippage assumptions
│
└─ Delta > 20%?
   ├─ CHECK LOOKAHEAD FIRST (Step 3)
   │  └─ grep for "j + 1" or "// ratio" errors
   ├─ Then check incomplete candles (Step 4)
   ├─ Then isolate by TF (Step 2)
   └─ If still unresolved, review CLAUDE.md parity section
```

## Test Command

```bash
# Full parity verification test
uv run pytest tests/test_backtest_live_parity.py -v -s

# If failing, also run:
uv run pytest tests/test_volstraddle_lookahead.py -v
uv run pytest tests/test_mtf_lookahead.py -v
```

## References
- **Parity doctrine**: ``quantbox/backtest/trade_sim.py` (parity docstrings)
- **Lookahead checklist**: `CLAUDE.md` (Backtest-Live Parity + Incident Rules)
- **Multi-TF rules**: CLAUDE.md Rule #2 (line 99)
- **Tests**: `tests/test_backtest_live_parity.py`, `tests/test_*_lookahead.py`
- **Cost model**: `quantbox/backtest/trade_sim.py` (lines 70-79)

## When to Use This Skill

- Backtest results show >5% divergence from live
- Deploying a new signal to production (verify parity first)
- Optimizing strategy parameters (parity must hold through optimization)
- Multi-timeframe signal development (high lookahead risk)
