# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantBox is a production algorithmic trading system for Binance USDT-M perpetual futures. It runs a dual-axis squeeze strategy: **VR Squeeze** (VolStraddle, 30x leverage) and **MA Cluster Squeeze** (21x leverage) sharing a 10-slot portfolio. This is a live system managing real capital — all rules exist because of past losses or bugs.

**Top Rule: Never guess numbers or facts. If you don't know, say so. Always read the code/config first.**

## Build & Development Commands

```bash
# Install dependencies
uv sync

# Run all tests (excludes @slow by default via pyproject.toml)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_vol_straddle.py -v

# Run slow tests (full backtests with real data)
uv run pytest tests/ -m slow -v

# Lint & format
uv run ruff check quantbox/ --fix
uv run ruff format quantbox/ tests/

# Type check
uv run mypy

# Run the bot (paper trading)
python -m quantbox.scalper --strategy vol_straddle --paper

# Run a single cycle
python -m quantbox.scalper --strategy vol_straddle --paper --once
```

Pre-commit hooks run: ruff format → ruff check → mypy → pytest-fast → research-param-guard (on research/*.py).

## Architecture

### Mixin-Based Orchestrator

`RubberBandScalper` in `scalper.py` composes 6 mixins:
- `LoopMixin` — timing, bar waiting, logging
- `DataMixin` — kline download, symbol info
- `BracketMixin` — SL/TP bracket orders
- `ReconciliationMixin` — exchange state sync, ghost detection
- `PositionMixin` — entry, exit, sizing
- `CommandsMixin` — Discord bot commands

### Strategy Protocol

`TradingStrategy` (14-member `@runtime_checkable` Protocol in `strategy/protocol.py`) defines the interface. `VolStraddleStrategy` and `RubberBandStrategy` implement it via structural subtyping — no inheritance required.

### Hedge Mode Compound Keys

Positions use compound keys: `"SUIUSDT:LONG"` / `"SUIUSDT:SHORT"` internally. API calls must split: `symbol = pos_key.split(':')[0]`. Type system distinguishes `ApiSymbol` (no colon) from `PositionKey` (may have colon) in `core/types.py`.

### Dual Backtest Engines

- **VectorizedBacktest** (`backtest/vectorized_backtest.py`) — signal arrays, close-to-close PnL (RubberBand, fast)
- **TradeEventSim** (`backtest/trade_sim.py`) — discrete trades with intra-bar SL/TP, slot contention, trailing stops (VolStraddle, production metrics). This is the canonical sim — research scripts must import `collect_trades()` + `portfolio_sim()` from here, never reimplement.

### Config as Source of Truth

All parameters live in `VolStraddleConfig` (`strategy/vol_straddle/config.py`), a pydantic-settings class reading from env vars (prefix `VS_`). **Never hardcode parameters.** Always read from config:
```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
VR_LN = cfg.base_leverage / cfg.max_concurrent  # 30/10=3.0
```

`pydantic-settings` makes `.env` values override `settings.py` defaults. Changing `settings.py` alone without updating `.env` has no effect.

### Numba JIT

Signal functions in `strategy/vol_straddle/signals.py` use `@numba.jit(nopython=True)`. These are invisible to coverage.py — verified separately via `tests/test_signals_unit.py`.

### State Persistence

Atomic JSON persistence via `core/scalper_state.py`: tmpfile → `os.replace()`. Crash recovery restores positions + balance on restart.

## Core Principles

1. **No Mock Data** — backtests use only real Binance data. No `np.random` synthetic data.
2. **Backtest-Live Parity** — backtest logic = live logic. No lookahead, no incomplete candles, no hardcoded params. Symbol loading uses `cfg.symbol_whitelist` (not `exclude_symbols`).
3. **Test-Code Parity** — every feature needs pytest tests. Untested code = incomplete.
4. **Execution Reality** — always account for fees (0.05%), funding, slippage (round-trip = 0.0008).
5. **Code First** — read code/config before answering questions about parameters.

## Incident-Driven Rules (Absolute — violations cause real losses)

| # | Rule |
|---|------|
| #1 | `futures_change_leverage()` only in entry paths, never close/reduce |
| #2 | Cross-TF refs use completed bars only (`j = (i // ratio) - 1`) |
| #3 | Slot sizing tests must include `n_active > 0` state |
| #4 | `_check_order_filled()` handles both TRIGGERED and FINISHED |
| #5 | Hedge Mode close uses `reduce_only=False` + leverage guard: `if not reduce_only and leverage > 1:` |
| #6 | Config params always read from `VolStraddleConfig()`, never hardcoded |
| #7 | `CANCELED` (US spelling, 1 L) — Binance Algo API uses American spelling, check both |
| #8 | TRAILING_STOP_MARKET: Algo API only (`-4120` on regular API). `activationPrice=entry×(1±trail_activate)` — Binance keeps trail in NEW until price reaches activation (verified live 2026-03-19). `callbackRate≥0.5%` — tighter fires on tick noise (r104b: 0.1% has 81% bias) |

## API Testing Policy

**API 변경 시 반드시 실제 소액 주문으로 검증.** Mock 테스트만으로 커밋 금지.
- 테스트 심볼: DOGEUSDT (50 DOGE ≈ $5)
- 주문 → 체결 확인 → 취소/청산까지 full lifecycle
- 문서/라이브러리 소스보다 실제 API 응답이 source of truth (Binance 문서가 부정확한 사례 확인됨)

## Code Quality

- **Line length**: 120 (ruff enforced)
- **Type hints**: Python 3.13+ generics (`list[T]`, `dict[K,V]`)
- **mypy**: strict on core types/protocol/config/execution modules (see pyproject.toml `[tool.mypy]`)
- **Docstrings**: Google style
- **Language**: Korean in comments/Discord messages; code identifiers in English

## Modification Impact Matrix

When modifying these files, always check the paired files:

| Modified | Must also check |
|----------|----------------|
| `settings.py` params | `.env.example`, `.env` |
| `vol_straddle/signals.py` | `trade_sim.py`, `test_signals_unit.py` |
| `vol_straddle/config.py` | All backtest scripts, `.env` |
| `scalper.py` trading logic | `trade_sim.py` parity |
| `core/scalper_state.py` | `scalper.py`, `conftest.py`, state-related tests |
| Any new module | `README.md` structure, `__init__.py`, test file, `AGENTS.md` |
| `AGENTS.md` | `.github/skills/` cross-references for staleness |

## Git Flow

| Branch | Purpose | Merge style |
|--------|---------|-------------|
| `main` | Production (no direct commits) | `release` → `--no-ff` + version tag |
| `develop` | Integration | feature/fix merge here |
| `feature/xxx` | New features | `--squash` |
| `research/xxx` | Backtest exploration | `--squash` + archive |
| `fix/xxx` | Bug fixes | `--squash` |
| `hotfix/xxx` | Urgent (from main) | `--no-ff` to both main + develop |

Research → feature commits use `(ref: rNN)` to trace lineage.

## Backtest Reports

Location: `docs/backtest_results/YYYYMM/` only. Naming: `{YYYYMMDD}_{commit_short}_{description}.md`.

## Risk Levels for Changes

| Level | Examples | Action |
|-------|----------|--------|
| Low | Log messages, docstrings | Execute directly |
| Medium | New filters, notification changes | Execute + verify tests |
| High | Signal/exit/cost/sizing logic | User confirmation + parity verification required |
| Forbidden | Remove safety guards, use mock data, break parity | Never execute |

## Project Skills (GitHub Copilot CLI & Claude Code)

Project-specific skills live in `.github/skills/*/SKILL.md` (Copilot CLI-compatible format).

**Loading in Copilot CLI**: Skills auto-load based on semantic matching in description field. Use `/skills reload` if manually added.

**Loading in Claude Code**: Reference a skill with `cat .github/skills/<skill-name>/SKILL.md` in your prompt, or Claude will suggest it automatically.

### Current Priority Skills (5)

1. **quantbox-incident-rules** (116 lines)
   - Unified checklist for all 8 incident-driven rules
   - Use: Before every production commit
   - Triggers: `leverage`, `reduce_only`, `CANCELED`, `rule #1-#8`, `SL missing`, `-2028`, `-1022`

2. **quantbox-trade-event-sim** (122 lines)
   - Decision tree for canonical backtest engine (TradeEventSim vs VectorizedBacktest)
   - Use: When selecting backtest engine or considering reimplementation
   - Triggers: `trade sim`, `collect_trades`, `canonical`, `straddle backtest`, `reimplement`

3. **quantbox-backtest-parity** (177 lines)
   - Multi-TF parity diff workflow for debugging backtest vs live divergence
   - Use: When backtest results > 5% differ from live
   - Triggers: `parity`, `backtest divergence`, `live mismatch`, `lookahead`

4. **quantbox-signal-development** (231 lines)
   - 4-phase signal development workflow: data → engine selection → validation → backtest
   - Use: When creating new signals or debugging signal output
   - Triggers: `signal development`, `numba`, `signal generation`, `data download`, `backtest baseline`

5. **quantbox-reconciliation-ghost** (240 lines)
   - Ghost position detection flowchart and recovery workflow
   - Use: When positions exist in state but not on exchange, or vice versa
   - Triggers: `ghost detection`, `reconciliation`, `orphan`, `counterpart SL`, `reverse ghost`, `exchange sync`

6. **quantbox-live-ops-agent** (450+ lines)
   - Autonomous live session diagnostics: tmux capture, Binance API verification, discrepancy detection, strategy performance analysis, error/bug detection, remediation plan
   - Use: Morning desk check, post-deploy verification, drawdown investigation, weekly performance audit
   - Triggers: `live ops`, `trading audit`, `session audit`, `desk check`, `risk check`, `discrepancy`, `account audit`, `ops agent`
   - **Claude Code slash command:** `/live-ops` (`.claude/commands/live-ops.md`)

### Phase 2 Deferred (Existing coverage in `.instructions.md`)

- quantbox-state-persistence (90% covered by debugging.instructions.md Section 3)
- quantbox-position-sizing (100% covered by position_sizing.instructions.md)
- quantbox-bracket-orders (85% covered by debugging.instructions.md Section 2.1)
- quantbox-hedge-mode (80% covered by debugging.instructions.md Section 1)

Add Phase 2 skills when: (a) agent misses guidance in that domain, AND (b) existing `.instructions.md` coverage is demonstrably insufficient.
