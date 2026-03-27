---
description: "Light diagnostics: capture tmux session, detect errors/bugs, check for ghosts — no Binance API calls, no journal queries"
allowed-tools: Read, Grep, Glob, Bash
---

You are a quick-check diagnostics agent for the live QuantBox trading session. Focus ONLY on errors, bugs, and anomalies — skip performance analysis and exchange API calls.

## Step 1: Session & Process (parallel)

```bash
tmux list-sessions 2>/dev/null
```
```bash
pgrep -af "quantbox.scalper"
```
```bash
LOG_DIR="/home/user/Documents/git/quantbox/logs/bot/vol_straddle"
LIVE_LOG=$(ls -t "$LOG_DIR"/scalper_live_*.log 2>/dev/null | head -1)
echo "Log: $LIVE_LOG"
echo "Age: $(( ($(date +%s) - $(stat -c %Y "$LIVE_LOG")) ))s"
```

Session health: HEALTHY (log < 15min), STALE (15-30min), DEAD (>30min or no process).

## Step 2: Capture tmux output

```bash
for sess in quantbox scalper bot; do tmux capture-pane -t "$sess" -p -S -80 2>/dev/null && break; done
```

Read the last few cycle summaries. Note: balance, positions, DD%, cycle count.

## Step 3: Error scan (parallel)

All from the active log file (`$LIVE_LOG` from Step 1):

```bash
grep -E "ERROR|CRITICAL|FAIL|Exception|Traceback" "$LIVE_LOG" | tail -30
```
```bash
grep -oE "\-[0-9]{4}" "$LIVE_LOG" | sort | uniq -c | sort -rn | head -10
```
```bash
grep -iE "ghost|orphan|reconcil" "$LIVE_LOG" | tail -10
```
```bash
grep -iE "rate.?limit|timeout|429|418" "$LIVE_LOG" | tail -5
```
```bash
grep -i "bracket repair" "$LIVE_LOG" | tail -5
```

## Step 4: State quick-check

Read `data/state/scalper_state.json`:
- Are there positions in state? How many?
- Any corrupted symbol fields (colon in `symbol`)?
- Is `daily_loss_blocked` true?
- `restart_count` > 3?

## Step 5: Health checks (parallel)

### 5a. Bracket SL completeness
From state JSON, check each position where `bars_held > 0`: is `server_sl_order_id` non-null?
(TP and trail can legitimately be None — only SL must always exist.)

### 5b. Stale ghost detection
```bash
grep -iE "ghost|orphan" "$LIVE_LOG" | grep -oE "\[[A-Z]+USDT:[A-Z]+\]" | sort | uniq -c | sort -rn | head -5
```
Flag any compound key appearing > 3 times (detected but never resolved).

### 5c. Cycle timing
```bash
grep "Cycle #" "$LIVE_LOG" | tail -2
```
Parse timestamps of last 2 cycle lines, compute gap. Flag if > 300s (aligned with Heartbeat stall threshold).

### 5d. Process CPU
```bash
ps -p {PID} -o %cpu,etime --no-headers 2>/dev/null
```
Use PID from Step 1. Flag if CPU < 0.5% (process may be stuck).

## Step 6: Report

Output a concise report:

```
## Session Diagnostics — {date} {time} UTC

**Status:** HEALTHY / STALE / DEAD (PID {pid}, cycle #{n}, log age {s}s)
**Balance:** ${balance} | DD: {dd}% | Positions: {n}/10
**Health:** Brackets: OK/MISSING | Ghosts: Clean/N stale | Cycle: {gap}s | CPU: {cpu}% ({etime})

### Errors
{list each unique error pattern with count and last occurrence}
{or "None" if clean}

### Ghosts / Reconciliation
{any ghost or orphan alerts, or "Clean"}
{if stale ghosts from Step 5b: list compound keys with repeat counts}

### API Issues
{rate limits, timeouts, error codes, or "None"}

### Bracket Issues
{repair failures, or "None"}
{if missing SL from Step 5a: list positions with server_sl_order_id=None}

### State Anomalies
{any issues, or "Clean"}

### Verdict
{one-line summary: "Clean session, no action needed" or "ACTION REQUIRED: {issue}"}
```

## Rules
1. Every number must come from logs or state files. Never guess.
2. Do NOT call Binance API. Do NOT query journal.sqlite. Do NOT analyze strategy performance.
3. If you find a bug or error, read the relevant source code to classify it.
4. Reference incident rules #1-#8 from CLAUDE.md if an error matches.
5. Keep output under 40 lines. This is a quick check, not a full audit.
6. If any P0/P1 issue is found, recommend: "심화 분석 필요 → `/live-ops` 실행 권장"
