---
name: quantbox-reconciliation-ghost
description: |
  Ghost position detection flowchart and diagnostic workflow for reconciliation failures.
  Use when positions exist in state but not on exchange, or vice versa.
  Triggers: ghost detection, reconciliation, orphan, counterpart SL, reverse ghost, exchange sync, phantom position
license: MIT
---

# Ghost Position Detection & Recovery

## Purpose

Ghost positions = local state diverges from exchange reality. Causes failed closes, duplicate entries, margin calls. This skill provides diagnostic and recovery workflows.

## Ghost Type Detection (Decision Tree)

```
Position exists in state file but NOT on Binance?

├─ Is the position VERY OLD (> max_hold duration)?
│  └─ ORPHAN SL: Entry filled, but SL never triggered
│     └─ Recovery: Manual close or let SL execute
│
├─ Is the position RECENT (< max_hold)?
│  │
│  └─ Check if SL was placed on Binance:
│     ├─ SL order found on Binance? (check order status)
│     │  └─ COUNTERPART SL RACE: SL placed but cancelled by us
│     │     └─ Recovery: Re-place SL, re-sync state
│     │
│     ├─ SL order NOT found?
│     │  ├─ Did we EVER try to place SL? (check logs)
│     │  │  ├─ YES: FORWARD GHOST (SL request never reached Binance)
│     │  │  │  └─ Recovery: Re-create bracket orders
│     │  │  │
│     │  │  └─ NO: INCOMPLETE ENTRY (position opened but SL skipped)
│     │  │     └─ Recovery: Emergency close position immediately
│     │  │
│     │  └─ SL exists but position MISSING?
│     │     └─ REVERSE GHOST (2026-03-13 incident: position closed but SL remains)
│     │        └─ Recovery: Cancel orphan SL, sync state

└─ If unsure, assume WORST CASE and close immediately
```

## Diagnostic Steps (Checklist)

### Step 1: Identify Ghost Positions

```python
import json
from datetime import datetime, timedelta

# Load state file
with open('data/state/scalper_state.json') as f:
    state = json.load(f)

# Load Binance positions via API
binance_positions = client.futures_position_information()
binance_symbols = {p['symbol'] for p in binance_positions if float(p['positionAmt']) != 0}

# State vs Binance comparison
state_symbols = set(state['positions'].keys())
ghosts = state_symbols - binance_symbols  # In state, not on Binance

print(f"Ghost positions: {ghosts}")
for pos_key in ghosts:
    pos = state['positions'][pos_key]
    entry_time = datetime.fromtimestamp(pos['entry_time'])
    age_hours = (datetime.now() - entry_time).total_seconds() / 3600
    print(f"  {pos_key}: age={age_hours:.1f}h, size={pos.get('qty')}, entry_price={pos.get('entry_price')}")
```

### Step 2: Check for Counterpart SL Race

Most common ghost cause: Our code tries to close while SL simultaneously triggers.

```python
# Check if SL exists on Binance (even though position is gone)
symbol = ghosts.pop()  # e.g., 'SUIUSDT'
orders = client.futures_get_all_orders(symbol=symbol, limit=100)

# Find the most recent SL order for this position
sl_orders = [o for o in orders if 'STOP' in o['type']]
if sl_orders:
    latest_sl = max(sl_orders, key=lambda o: o['updateTime'])
    print(f"Found SL order: {latest_sl['orderId']} status={latest_sl['status']} type={latest_sl['type']}")

    if latest_sl['status'] == 'FILLED':
        print("→ LIKELY: SL executed while we were closing (race condition)")
        print("→ RECOVERY: Verify position actually closed, update state file")
    elif latest_sl['status'] == 'CANCELED':
        print("→ LIKELY: We cancelled SL before position closed (counterpart SL race)")
        print("→ RECOVERY: Re-place SL immediately and retry close")
```

### Step 3: Check Logs for SL Placement

```bash
# Search for SL placement attempts in the past 10 minutes
grep -i "place.*stop\|stop.*loss\|activationPrice" data/logs/*.log | tail -20

# Look for errors in order placement
grep -i "error\|-1022\|-2021\|-4120" data/logs/*.log | tail -20

# Check Discord alerts for clues
# Manual inspection of #trading channel for position updates
```

### Step 4: Determine Ghost Type

#### FORWARD GHOST (SL never placed)
```
Symptoms:
- Position in state, NOT on Binance
- No SL order found on Binance
- Logs show entry filled but NO SL placement attempt

Cause: Network failure, race condition, or SL placement code skipped

Recovery:
1. Close position manually on Binance:
   client.futures_create_order(symbol='SUIUSDT', side='SELL', positionSide='LONG',
                               type='MARKET', quantity=...)
2. Update state: state['positions'].pop('SUIUSDT:LONG')
3. Save state: json.dump(state, ...)
4. Restart bot and let reconciliation verify
```

#### COUNTERPART SL RACE (Both close simultaneously)
```
Symptoms:
- Position in state, NOT on Binance
- SL order exists but is FILLED or CANCELLED
- Logs show concurrent close + SL trigger

Cause: Position closed while SL also triggered (legitimate race)

Recovery:
1. Verify position is actually closed on Binance:
   positions = client.futures_position_information(symbol='SUIUSDT')
   if float(positions[0]['positionAmt']) == 0: print("✓ Closed")
2. Update state and PnL to match SL execution
3. Restart bot
```

#### REVERSE GHOST (2026-03-13 incident)
```
Symptoms:
- SL order exists on Binance (TRIGGERED or FILLED)
- Position quantity is now 0
- State file still shows open position

Cause: SL executed closing the position, but our state update failed

Recovery:
1. Cancel orphan SL order (won't work if already filled):
   client.futures_cancel_order(symbol='SUIUSDT', orderId=sl_order_id)
2. Clear state: state['positions'].pop('SUIUSDT:LONG')
3. Verify PnL: Look for SL execution in trade_journal.csv
4. Restart bot
```

#### ORPHAN SL (Entry filled, SL never triggered)
```
Symptoms:
- Position in state but very old (> max_hold_15m = 192 bars = 48 hours)
- SL still placed on Binance but untouched

Cause: Position hit max_hold limit but our close logic was overridden

Recovery:
1. Manually close the position:
   client.futures_create_order(symbol='..', side='SELL', type='MARKET', quantity=...)
2. Cancel orphan SL:
   client.futures_cancel_order(...)
3. Log incident and analyze why max_hold wasn't enforced
4. Update state
```

## Quick Recovery Script

```python
# Emergency ghost recovery (use only if position is confirmed missing on Binance)
import json

with open('data/state/scalper_state.json') as f:
    state = json.load(f)

# Remove ghost position
ghost_pos = 'SUIUSDT:LONG'
if ghost_pos in state['positions']:
    pos = state['positions'].pop(ghost_pos)
    print(f"Removed ghost: {ghost_pos} (entry: {pos['entry_price']}, qty: {pos['qty']})")

# Backup and save
import shutil
shutil.copy('data/state/scalper_state.json', 'data/state/scalper_state.json.backup')
with open('data/state/scalper_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print("State updated. Restart bot with: python -m quantbox.scalper --paper --once")
```

## Verification Tests

```bash
# Reconciliation hardening tests
uv run pytest tests/test_reconciliation.py -v
uv run pytest tests/test_reconciliation_hardening.py -v
uv run pytest tests/test_reconciliation_reverse_ghost.py -v

# If any fail, ghost detection logic has regressed
```

## References
- **Reconciliation logic**: `quantbox/execution/reconciliation.py` (lines 41-75 counterpart SL, lines 554-930 ghost detection)
- **Ghost detection guide**: `.github/instructions/debugging.instructions.md` Section 2.2 (lines 244-260)
- **Incident #6 details**: 2026-03-13 reverse ghost (state file shows position, Binance shows 0, SL was filled)
- **Related incidents**: Incident #4 (compound key), Incident #7 (CANCELED spelling)
- **Tests**: `tests/test_reconciliation.py`, `tests/test_reconciliation_reverse_ghost.py`
- **Source**: `quantbox/execution/reconciliation.py` (full implementation)

## When to Use This Skill

- Position exists in Discord alerts but not on state/Binance (forward ghost)
- Position closed but SL still exists on Binance (reverse ghost)
- Reconciliation loop is spinning (multiple sync attempts failing)
- Position safety is uncertain (emergency close needed)
- Debugging a failed close or duplicate entry

## Warning

⚠️ **Ghost positions are account-level emergencies.** If unresolved, they can:
- Lock up margin (inflated risk)
- Cause duplicate entries (margin call risk)
- Prevent graceful shutdown

If you cannot diagnose in 5 minutes, **close the position manually on Binance immediately** and investigate later.
