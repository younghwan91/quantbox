---
name: quantbox-trade-event-sim
description: |
  Decision tree for canonical backtest engine selection (TradeEventSim vs VectorizedBacktest).
  Use when running backtests or considering simulation reimplementation.
  Triggers: trade sim, collect_trades, canonical, straddle backtest, reimplement
license: MIT
---

# Trade Event Sim — Canonical Engine Selection

## Purpose

There are 2 backtest engines in QuantBox. Choose correctly and never reimplement — subtle details make divergence easy.

## Decision Tree: Which Engine?

```
What strategy are you backtesting?

├─ VolStraddle (VR Squeeze, straddle pairs)?
│  └─ MUST USE: TradeEventSim (quantbox/backtest/trade_sim.py)
│     • Intra-bar SL/TP resolution (high/low)
│     • Slot contention (portfolio_sim)
│     • Production metrics (PF, Calmar, EV)
│     • Cost model: 0.0008 round-trip per leg
│     • ✓ Canonical baseline for VR Squeeze
│
├─ RubberBand (directional fade, single-leg)?
│  └─ MAY USE: VectorizedBacktest (legacy, deprecating)
│     • Signal arrays, close-to-close PnL
│     • Fast iteration (pre-baked signals)
│     • NOT recommended for new work
│
└─ Uncertain?
   └─ DEFAULT: TradeEventSim
      • Always safe for straddle work
```

## Import Rule (Absolute)

### ✓ CORRECT: Import from Trade Sim
```python
from quantbox.backtest.trade_sim import collect_trades, portfolio_sim, compute_metrics

trades = collect_trades(symbol, klines, signals, params)
pf, ev, calmar = portfolio_sim(trades, slot_limit=10)
```

### ✗ WRONG: Reimplement
```python
# DO NOT DO THIS — you will diverge from production
def my_collect_trades(symbol, klines, signals):
    # Custom trade logic here
    # Likely to miss edge cases in:
    # - momentum=0 handling
    # - squeeze counter reset
    # - SL/TP dedup order
    ...
```

## Why Never Reimplement

The `trade_sim.py` file contains 23 years of incident-driven refinement:
- **Momentum=0 handling** (r53): Trades expire on direction change
- **Squeeze counter reset** (r50): Resets when volatility crosses threshold
- **Dedup order** (trade_sim.py:180-195): Filters duplicate SL triggers
- **Intra-bar SL/TP** (trade_sim.py:142-165): High/low prices, not close only
- **Slot contention** (trade_sim.py:510-540): Portfolio-level slot limits
- **Cost model parity** (trade_sim.py:70-79): Matches live execution (0.0008 per leg)

**If you reimplement any of these, you WILL diverge from backtest results.**

Reference: `quantbox/backtest/trade_sim.py` docstring (lines 1-26)

## Cost Model Verification

```python
from quantbox.backtest.trade_sim import COST_DIRECTIONAL, COST_STRADDLE

# Verify before running backtest:
assert COST_DIRECTIONAL == 0.0008, "Directional cost (2x entry+exit)"
assert COST_STRADDLE == 0.0016, "Straddle cost (4x: long+short entry+exit)"

# These must match VolStraddleConfig:
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
assert cfg.round_trip_cost == 0.0008, "Config must match trade_sim"
```

## Pipeline Order (Checklist)

When running a straddle backtest:
1. ❏ Download real klines (no synthetic data)
2. ❏ Generate signals (gen_vr_signal, gen_vr_dur_signal from vol_straddle/signals.py)
3. ❏ Call `collect_trades(symbol, klines, signals, cost=COST_STRADDLE)`
4. ❏ Call `portfolio_sim(trades, slot_limit=10)` (slot contention)
5. ❏ Call `compute_metrics()` (PF, EV, Calmar, DD)
6. ❏ Verify cost model: sum of trade costs ≈ PF * profit (cost impact visible)

## Testing Verification

Run the canonical baseline test:
```bash
uv run pytest tests/test_trade_sim.py -v
uv run pytest tests/test_backtest_live_parity.py -v
```

If either fails, the cost model or trade logic has diverged.

## References
- **Canonical engine**: `quantbox/backtest/trade_sim.py` (lines 1-26 docstring)
- **Engine comparison**: ``quantbox/backtest/trade_sim.py` (engine docstrings)
- **Cost model**: `trade_sim.py:70-79` (COST_DIRECTIONAL, COST_STRADDLE)
- **Tests**: `tests/test_trade_sim.py`, `tests/test_pbt_vol_straddle.py`

## When to Use This Skill

- Before starting a backtest (choose engine correctly)
- When considering a custom backtest implementation (check this first — probably not needed)
- When backtest/live numbers diverge unexpectedly (verify cost model and import path)
- When debugging trade-level issues (use collect_trades directly, examine output)
