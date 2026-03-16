<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-16 -->

# quantbox/core

## Purpose

Core infrastructure for the RubberBandScalper orchestrator: type system, protocol definitions, state persistence, configuration, and domain mixins. This module enforces production safety through structural typing and atomic state management.

## Files

| File | Purpose | Key Exports |
| --- | --- | --- |
| `types.py` | Domain type definitions (NewTypes, TypedDicts, Literals) | `ApiSymbol`, `PositionKey`, `PositionDict`, `TradeRecord`, `SymInfoDict` |
| `protocol.py` | ScalperProtocol — structural type for composed Scalper | `ScalperProtocol` (47 methods + 20 attributes) |
| `settings.py` | Environment variable validation (pydantic-settings) | `QuantBoxSettings`, `get_settings()` |
| `config.py` | Static app configuration (paths, logging, exchange) | `Config`, `config` singleton |
| `scalper_state.py` | Atomic JSON persistence, backup rotation | `save_scalper_state()`, `load_scalper_state()` |
| `loop_mixin.py` | Timing, bar waiting, cycle logging | `LoopMixin` class |
| `commands.py` | Discord bot command dispatch (pause, close-all, status) | `BotCommand`, `CommandsMixin` |
| `bot_utils.py` | Discord message formatting, notification utilities | `format_trade_update()`, etc. |
| `heartbeat.py` | Health check interval timer | `HeartbeatMonitor` |
| `rate_limiter.py` | Request throttling for Binance API | `RateLimiter` |
| `limit_chaser.py` | LimitOrderChaser — order fill retries with live price | `LimitOrderChaser` |
| `divergence_monitor.py` | Track P&L divergence between internal state + exchange | `DivergenceMonitor` |
| `trade_journal.py` | CSV trade logging | `TradeJournal` |
| `events.py` | Event dataclasses (Position, Trade, etc.) | `PositionEvent`, `TradeEvent` |
| `terminal_ui.py` | Real-time position display (curses) | `TerminalUI` |
| `sd_notify.py` | systemd readiness protocol | `sd_notify()` |
| `__init__.py` | Module exports | — |

## Design Patterns

### 1. Domain Types (Single Source of Truth)

**Files:** `types.py`

Hedge mode requires compound keys internally (`"SUIUSDT:LONG"` / `"SUIUSDT:SHORT"`), but Binance API expects pure symbols (`"SUIUSDT"`). The type system prevents mixing them:

- `ApiSymbol` = `NewType("ApiSymbol", str)` — must be pure symbol, no colon
- `PositionKey` = `NewType("PositionKey", str)` — may have colon (internal use only)
- `Side` = `Literal["BUY", "SELL"]` — order direction (API-compatible)
- `PositionSide` = `Literal["LONG", "SHORT", "BOTH"]` — hedge mode side

mypy catches cross-boundary mistakes at pre-commit time, not at 3am during live trading.

### 2. Structural Protocol (Mixin Resolution)

**Files:** `protocol.py`

RubberBandScalper is composed from 6 mixins (LoopMixin, DataMixin, BracketMixin, ReconciliationMixin, PositionMixin, CommandsMixin). Each mixin calls methods from others via `self`, creating circular reference problems.

Solution: `ScalperProtocol` declares the **complete interface** of the composed class (47 methods, 20 attributes). Mixins annotate cross-mixin methods with `self: ScalperProtocol`:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from quantbox.core.protocol import ScalperProtocol

class ReconciliationMixin:
    def _sync_server_orders(self: ScalperProtocol) -> None:
        self.strategy.get_positions()  # mypy resolves .strategy
        self._save_state()             # mypy resolves ._save_state()
```

This avoids circular imports and lets each mixin type-check independently.

### 3. Atomic State Persistence (Crash Recovery)

**Files:** `scalper_state.py`

Positions and trade history are persisted to disk atomically using the tmpfile → `os.replace()` pattern. This prevents corruption if the process crashes mid-write.

**Flow:**

1. Build state dict (positions, trade history, balance)
2. Compute SHA-256 checksum of all state keys (stored as `_checksum` field)
3. Write to temporary file in same directory
4. Atomically rename tmpfile → live state file
5. Back up old file with timestamp; keep 5 most recent backups
6. Attempt optional remote backup to `STATE_REMOTE_BACKUP_DIR`

**Load recovery:**

On app restart, load state file and verify checksum. If corrupt, fall back to most recent backup. Positions are restored with all metadata (entry price, SL/TP, bars_held, etc.).

### 4. Environment Validation (Fail-Fast)

**Files:** `settings.py`, `config.py`

Two-tier config:

- `QuantBoxSettings` (pydantic-settings) — validates required Binance API keys, leverage, position sizing, Discord webhook URL. Reads from `.env` file. Raises `ValidationError` at app startup if anything is missing or wrong type.
- `Config` singleton — static app config (paths, logging level, exchange metadata). Set at module load time; immutable thereafter.

This fails fast: bad config → error message → exit. Never runs with silent defaults.

### 5. Protocol Enforcement (Type Safety)

**Files:** `protocol.py`, `types.py`

Every method that accesses cross-mixin state is typed with the Protocol. Every position operation uses `PositionKey` or `ApiSymbol` explicitly. The result:

- mypy `--strict` catches typos and boundary crossings at pre-commit
- No `attr-defined` errors masking real bugs
- Clear which data is internal (PositionKey) vs. API-bound (ApiSymbol)

### 6. Command Dispatch (Bot Integration)

**Files:** `commands.py`, `bot_utils.py`

Discord commands (status, close-all, pause, resume, stop, pnl) are dispatched via a command registry. Each command is a method `_cmd_*()` annotated with `self: ScalperProtocol`. CommandsMixin processes them in the main loop via `_process_bot_commands()`.

## Critical Rules for Agents

1. **Never hardcode domain types.** Use `ApiSymbol`, `PositionKey`, `PositionDict`, `TradeRecord` from `types.py` everywhere. No raw `str` for symbols.

2. **Compound keys must be split before API calls.** If you have `pos_key = "SUIUSDT:LONG"`, extract the symbol as `symbol = pos_key.split(':')[0]` to get `ApiSymbol("SUIUSDT")` before calling `futures_create_order()`.

3. **Cross-mixin methods use `self: ScalperProtocol`.** If a method in one mixin calls another mixin's method, annotate the method signature: `def my_method(self: ScalperProtocol) -> None:`.

4. **Config is immutable after init.** Read from `QuantBoxSettings` at app startup; don't call it in hot loops. Cache the instance if needed.

5. **State is atomic.** Always call `_save_state()` after modifying `self.strategy._positions` or `self._trade_history`. Never do partial writes.

6. **Types enforce boundaries.** If a function expects `ApiSymbol`, pass an `ApiSymbol`. If it expects `PositionKey`, pass a `PositionKey`. Let mypy catch mixing them.

7. **PositionDict has optional fields.** Use `.get(key, default)` for strategy-specific fields (vol_straddle-only: `is_straddle`, `server_sl_order_id`, etc.). Never assume a key exists.

8. **Leverage changes only in entry paths.** `futures_change_leverage()` is called exactly once per symbol per entry cycle. Never in close/reduce paths. Use `leverage_guard` in hedge mode closes.

9. **Checksum validates state integrity.** When loading state, verify the `_checksum` field. If it doesn't match, the file is corrupt; fall back to backups.

10. **Type annotations are enforceable contracts.** If a test or external code passes the wrong type, mypy will reject it at pre-commit. Use this to prevent runtime errors.

11. **Settings are environment-sourced only.** Never modify `QuantBoxSettings` at runtime. Create a new instance if overrides are needed for testing.

## For AI Agents

### Task: Add a new mixin method

**Protocol first.** Add the method signature to `ScalperProtocol` in `protocol.py`, then implement it in the mixin. This ensures other mixins can call it with type safety.

### Task: Modify position state

**Use PositionDict.** Import from `types.py`, access fields with `.get()` for optional ones, always call `_save_state()` when done.

### Task: Add a command

**Use CommandsMixin.** Add a `_cmd_*` method, annotate it with `self: ScalperProtocol`, add it to `ScalperProtocol`, then register it in `commands.py`.

### Task: Read config at runtime

**Use QuantBoxSettings.** Call `get_settings()` once at app startup, cache the instance, and pass it as a parameter if needed in hot loops.

### Task: Persist new state fields

**Update PositionDict and TradeRecord TypedDicts** in `types.py`, then update `scalper_state.py` load/save logic to handle the new fields. Test crash recovery with the backup system.

## Dependency Graph

```
types.py (no imports from quantbox/)
  ↓
protocol.py (imports types)
  ↓
settings.py, config.py (no cross-dependencies)
  ↓
scalper_state.py (imports types)
  ↓
All Mixins (LoopMixin, DataMixin, BracketMixin, etc.)
  ├── Annotate cross-mixin calls with self: ScalperProtocol
  ├── Import only from types, settings, config
  ├── Call _save_state() after mutations
  └──→ scalper.py (composes all mixins)
```

## Atomic State Persistence Explained

**Why atomic writes matter:**

If the process crashes while writing 50MB of position state, the on-disk file is corrupted. Next restart tries to load garbage → crash loop.

**How quantbox solves it:**

1. Write to temp file in the same directory (same filesystem, atomic rename guaranteed)
2. Rename tmpfile → live file only after successful write (atomic operation)
3. Backup old file before overwriting (kept for 5 generations)
4. Verify checksum on load; if corrupt, fall back to backup
5. Optional remote backup (best-effort, never fails the write)

**Result:** Zero data loss on crash. Positions are always recoverable from backups.

## Testing Considerations

- **Type checking:** All core modules must pass `mypy --strict`.
- **State recovery:** Test `load_scalper_state()` with corrupted files; verify fallback to backups.
- **Protocol compliance:** Verify all Mixin classes have method signatures in ScalperProtocol.
- **Config validation:** Test `.env` parsing with missing required fields; expect `ValidationError`.
- **Atomic writes:** Kill process during `_save_state()` and verify crash recovery restores positions.
