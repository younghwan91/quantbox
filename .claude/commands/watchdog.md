---
description: "Autonomous watchdog: diagnose → fix → commit → restart. For cron/loop use."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
---

# Watchdog — Autonomous Session Guardian

You are an autonomous watchdog for the live QuantBox trading session.
Run the full /diagnose flow, then **act on problems automatically**.

## Phase 1: Diagnose (same as /diagnose)

Run the standard diagnostics:
1. Check tmux session & process alive
2. Check log age (HEALTHY < 15min, STALE 15-30min, DEAD > 30min)
3. Scan for ERROR/CRITICAL/Exception in latest log
4. Check state file (positions, daily_loss_blocked, restart_count)
5. Verify brackets (SL must exist for all positions)
6. Check cycle timing (gap > 300s = stalled)

## Phase 2: Triage

Classify the session:

| Status | Action |
|--------|--------|
| HEALTHY + no errors | → Phase 3b: Housekeeping (1 trivial improvement) |
| DEAD (no process or log > 30min) | → Phase 3: Restart |
| STALE (log 15-30min) | Wait one more check, restart if still stale |
| Code error (ImportError, SyntaxError, AttributeError) | → Phase 3: Fix + Restart |
| API error (-2019, -4028, etc.) | Log only, do NOT restart (transient) |
| Missing SL bracket | → Phase 3: Restart (bracket repair happens on startup) |
| Ghost detected 3+ times | Log warning, do NOT auto-fix |

## Phase 3: Auto-Fix (only if Phase 2 triggers it)

### For code errors:
1. Read the traceback from the log
2. Read the source file at the error location
3. Fix the bug using Edit
4. Run `uv run pytest tests/ -q --tb=line` (timeout 120s)
5. If tests pass: commit with message `hotfix: {description}`
6. If tests fail: revert with `git checkout -- .` and report "AUTO-FIX FAILED"

### For restart (DEAD, code fix applied, missing brackets):
```bash
tmux kill-session -t quantbox 2>/dev/null
sleep 2
tmux new-session -d -s quantbox -c /home/user/Documents/git/quantbox \
  "uv run python -m quantbox.scalper --strategy vol_straddle"
```
Wait 10s, verify process is running.

## Phase 4: Report

Always output a status line:

```
✓ Watchdog: clean (cycle #N, $balance, DD X%)
```
or
```
⚠ Watchdog: RESTARTED — {reason}
```
or
```
🔧 Watchdog: FIXED + RESTARTED — {description of fix}
```
or
```
❌ Watchdog: AUTO-FIX FAILED — {error}. Manual intervention needed.
```

## Phase 3b: Housekeeping (when session is HEALTHY)

When the session is clean, do ONE small improvement per cycle. Pick from this list in order:

### Trivial (auto-fix + commit immediately):
1. `grep -rn "straddle" quantbox/ tests/ --include="*.py" | grep -v __pycache__ | grep -v straddle_pair_id | grep -v STRATEGY_ALIASES | grep -v "straddle_syms\|vr_straddle.*vr_directional\|vol_straddle"` — clean remaining straddle references
2. `grep -rn "TODO\|FIXME\|HACK\|XXX" quantbox/ --include="*.py" | grep -v __pycache__` — resolve trivial TODOs
3. `uv run ruff check quantbox/ 2>&1 | head -5` — fix lint warnings
4. Unused imports (check with ruff)
5. Dead code removal (unreachable branches, unused variables)
6. Comment cleanup (stale version references, narrating-the-obvious)

### Small (log to .omc/watchdog-backlog.md, don't fix):
- New test cases for uncovered paths
- Helper extraction (>50 line methods)
- Config cleanup (dead parameters)
- Documentation updates

### Process:
1. Pick the FIRST item from Trivial list that has work to do
2. Make ONE change (not multiple)
3. Run `uv run ruff check quantbox/ --fix && uv run ruff format quantbox/ tests/`
4. Run `uv run pytest tests/ -q --tb=line` (timeout 120s)
5. If pass: `git add -u && git commit -m "chore: {description}"` + push
6. If fail: `git checkout -- .` and skip
7. Report: `🔧 Watchdog: clean + {what was fixed}`

If nothing trivial remains, check Small list and append to `.omc/watchdog-backlog.md`.
If backlog is empty too: `✓ Watchdog: clean, nothing to improve`

## Rules
1. NEVER modify trading logic (signals, sizing, brackets). Only fix crashes/imports/typos/dead-code.
2. NEVER call Binance API directly.
3. NEVER restart if positions are open (check state file first) unless process is DEAD.
4. Keep output to 1-3 lines. This runs on a timer — be quiet.
5. If unsure, do NOT fix. Log to backlog and let the user decide.
6. ONE fix per cycle max. Don't batch.
