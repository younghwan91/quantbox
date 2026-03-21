#!/bin/bash
# Paper trading watchdog — runs every hour via cron
# Checks health, restarts if dead, logs status
#
# Install: crontab -e → 0 * * * * /home/user/Documents/git/quantbox/scripts/paper_watchdog.sh
# Logs: /home/user/Documents/git/quantbox/logs/watchdog.log

set -euo pipefail

PROJECT_DIR="/home/user/Documents/git/quantbox"
LOG="$PROJECT_DIR/logs/watchdog.log"
TMUX_SESSION="quantbox"
LOG_DIR="$PROJECT_DIR/logs/bot/vol_straddle"
STATE_FILE="$PROJECT_DIR/data/state/scalper_state_paper.json"

mkdir -p "$(dirname "$LOG")"

log() {
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | $1" >> "$LOG"
}

# ── Check if paper process is alive ──
PAPER_PID=$(pgrep -f "quantbox.scalper.*--paper" || true)

if [ -z "$PAPER_PID" ]; then
    log "DEAD — no paper process found. Restarting..."

    # Check if tmux session exists
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        log "Creating tmux session $TMUX_SESSION"
        tmux new-session -d -s "$TMUX_SESSION"
    fi

    # Find an available pane or create one
    cd "$PROJECT_DIR"
    tmux send-keys -t "$TMUX_SESSION" "cd $PROJECT_DIR && python -m quantbox.scalper --strategy vol_straddle --paper" Enter
    sleep 5

    NEW_PID=$(pgrep -f "quantbox.scalper.*--paper" || true)
    if [ -n "$NEW_PID" ]; then
        log "RESTARTED — PID=$NEW_PID"
    else
        log "RESTART FAILED — check manually"
    fi
    exit 0
fi

# ── Check log freshness ──
PAPER_LOG=$(ls -t "$LOG_DIR"/scalper_paper_*.log 2>/dev/null | head -1)
if [ -z "$PAPER_LOG" ]; then
    log "WARNING — no paper log found (PID=$PAPER_PID running)"
    exit 0
fi

LOG_AGE=$(( $(date +%s) - $(stat -c %Y "$PAPER_LOG") ))

if [ "$LOG_AGE" -gt 1800 ]; then
    # Log stale > 30 min — process might be hung
    log "STALE — log age=${LOG_AGE}s (>30min). Killing PID=$PAPER_PID and restarting..."
    kill "$PAPER_PID" 2>/dev/null || true
    sleep 3

    cd "$PROJECT_DIR"
    tmux send-keys -t "$TMUX_SESSION" "cd $PROJECT_DIR && python -m quantbox.scalper --strategy vol_straddle --paper" Enter
    sleep 5

    NEW_PID=$(pgrep -f "quantbox.scalper.*--paper" || true)
    if [ -n "$NEW_PID" ]; then
        log "RESTARTED — PID=$NEW_PID (was stale)"
    else
        log "RESTART FAILED after stale kill"
    fi
    exit 0
fi

# ── Check for errors in last hour ──
ERROR_COUNT=$(grep -c "ERROR\|CRITICAL" "$PAPER_LOG" 2>/dev/null || echo 0)
LAST_CYCLE=$(grep "Cycle #" "$PAPER_LOG" 2>/dev/null | tail -1 | grep -oP 'Cycle #\K[0-9]+' || echo 0)
BALANCE=$(grep -oP 'Balance=\$\K[0-9.]+' "$PAPER_LOG" 2>/dev/null | tail -1 || echo "?")
DD=$(grep -oP 'DD=\K[0-9.]+' "$PAPER_LOG" 2>/dev/null | tail -1 || echo "?")
POSITIONS=$(grep -oP 'Positions=\K[0-9]+/[0-9]+' "$PAPER_LOG" 2>/dev/null | tail -1 || echo "?")

log "HEALTHY — PID=$PAPER_PID cycle=#$LAST_CYCLE bal=\$$BALANCE DD=${DD}% pos=$POSITIONS errors=$ERROR_COUNT age=${LOG_AGE}s"
