---
name: quantbox-incident-rules
description: |
  Unified pre-commit checklist for QuantBox's 8 incident-driven rules that prevent trading losses.
  Use when reviewing code changes in scalper.py, brackets.py, or execution modules.
  Triggers: leverage, reduce_only, CANCELED, rule #1-#8, SL missing, -2028, -1022, bracket
license: MIT
---

# Incident Rule Enforcement

## Purpose

All 8 rules in CLAUDE.md exist because of real capital losses. Before committing, verify every code change against this checklist.

## Pre-Commit Verification (Checklist)

### Rule #1: Leverage Only in Entry Paths
- **File**: `scalper.py:517-531` (entry), `scalper.py:1323` (close guard)
- **Check**: Is `futures_change_leverage()` called only in entry logic?
- **Violation**: Changing leverage on close/reduce causes liquidation
- ❏ Search for `futures_change_leverage` in modified files
- ❏ Verify all calls are inside entry conditions (`if signal and not position`)
- ❏ Verify close paths have leverage guard: `if not reduce_only and leverage > 1:`
- ❏ Run: `grep -n "futures_change_leverage" quantbox/scalper.py`

### Rule #2: Cross-TF Refs Use Completed Bars Only
- **File**: Multi-TF signal generation in `vol_straddle/signals.py`
- **Check**: Multi-timeframe candle index calculation
- **Violation**: Incomplete candles cause lookahead bias (r55: 83% backtest vs 36% live)
- ❏ For all `i // ratio` calculations, use `j = (i // ratio) - 1` (subtract 1 for completed)
- ❏ Verify no `j = i // ratio` (off by 1, includes current bar)
- ❏ Run: `grep -n "// ratio" quantbox/strategy/vol_straddle/signals.py`

### Rule #3: Slot Sizing Tests Include Active State
- **File**: Tests in `tests/test_pbt_sizing_parity.py`
- **Check**: Slot sizing with `n_active > 0` state
- **Violation**: Sizing logic fails when portfolio is full
- ❏ All slot tests include scenarios with `n_active` close to `max_concurrent`
- ❏ Tests verify dd_scaling applies when slots constrained

### Rule #4: Order Fill Check Handles Both TRIGGERED and FINISHED
- **File**: `execution/inflight.py` order fill detection
- **Check**: SL/TP trigger status detection
- **Violation**: Missed fills if only checking FINISHED status
- ❏ Verify `_check_order_filled()` checks both `TRIGGERED` and `FINISHED` states
- ❏ Trailing stop triggers on TRIGGERED, regular SL on FINISHED

### Rule #5: Hedge Mode Close Uses Guard
- **File**: `brackets.py:68-72` (positionSide + reduce_only)
- **Check**: Hedge mode close safety
- **Violation**: Changing leverage in hedge close reverses position
- ❏ Verify: `if not reduce_only and leverage > 1: raise`
- ❏ Verify `positionSide` is set correctly (LONG vs SHORT)
- ❏ Run: `grep -n "reduce_only=False" quantbox/execution/brackets.py`

### Rule #6: Config Params Never Hardcoded
- **File**: `strategy/vol_straddle/config.py` (VolStraddleConfig)
- **Check**: All parameter values sourced from config
- **Violation**: Hardcoded values diverge from env/.env defaults (r39: hardcoded symbol list)
- ❏ Search modified files for numeric literals (0.0008, 30, 21, 200, etc.)
- ❏ Verify all params read from `VolStraddleConfig()`, not hardcoded
- ❏ Run: `grep -E "\b(0\.0008|30|21|200|192|100|50)\b" quantbox/*.py` (check for hardcodes)

### Rule #7: CANCELED Spelling (US, 1 L)
- **File**: `brackets.py:700` (Binance Algo API uses American spelling)
- **Check**: Order status comparison
- **Violation**: Missing cancel events if only checking `CANCELLED` (2 L)
- ❏ Verify Algo API cancel checks include both: `"CANCELED"` and `"CANCELLED"`
- ❏ Binance documentation uses inconsistent spelling
- ❏ Run: `grep -i "CANCEL" quantbox/execution/brackets.py | head -20`

### Rule #8: TRAILING_STOP_MARKET (Algo API Only)
- **File**: `brackets.py:27-56` (`_place_algo_order` method)
- **Check**: Trailing stop order type selection
- **Violation**: Using regular API trailing stop returns error -4120
- ❏ Verify: TRAILING_STOP_MARKET via Algo API only (not regular API)
- ❏ Verify: `activationPrice=entry×(1±trail_activate)` — future price OK (verified live 2026-03-19). `callbackRate≥0.5%` (0.1% fires on tick noise)
- ❏ Verify: Regular API has no trailing stop support
- ❏ Run: `grep -n "TRAILING_STOP" quantbox/execution/brackets.py`

## Decision Tree

```
Modified scalper.py or brackets.py?
├─ YES: Check Rules #1, #2, #5, #7, #8 above
└─ NO: Continue to next rule check

Modified vol_straddle/signals.py?
├─ YES: Check Rule #2 (multi-TF bar indexing)
└─ NO: Continue

Modified strategy/vol_straddle/config.py?
├─ YES: Check Rule #6 (no hardcodes)
└─ NO: Continue

Modified execution/inflight.py?
├─ YES: Check Rule #4 (TRIGGERED + FINISHED)
└─ NO: All clear

Run full test suite before commit:
  uv run pytest tests/ -v
  uv run mypy
```

## References
- **Full rules**: CLAUDE.md:94-105
- **Incident details**: `CLAUDE.md` (Incident-Driven Rules #1-#8)
- **Test suite**: `tests/test_critical_path_guards.py`

## When to Use This Skill

- Before every commit to production-critical paths (scalper.py, brackets.py, signals.py, config.py)
- When debugging order failures (check Rule #7, #8)
- When backtest/live numbers diverge (check Rule #2, #6)
- When leverage/margin errors occur (check Rule #1, #5, #6)
