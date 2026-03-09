#!/usr/bin/env bash
# QuantBox watchdog — auto-restarts on crash, stops on clean exit (Ctrl+C)
# Usage: bash scripts/watchdog.sh
# Tmux:  tmux new-session -d -s quantbox 'bash scripts/watchdog.sh'

BOT_DIR="/home/user/Documents/git/quantbox"
BOT_CMD=".venv/bin/python3 -m quantbox.scalper --strategy vol_straddle"
RESTART_DELAY=15
CLEAN_EXIT_CODES=(0 130 143)   # 0=OK, 130=Ctrl+C, 143=SIGTERM → do NOT restart

cd "$BOT_DIR"

while true; do
    echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') — starting bot..."
    $BOT_CMD
    EXIT_CODE=$?

    for clean in "${CLEAN_EXIT_CODES[@]}"; do
        if [ "$EXIT_CODE" -eq "$clean" ]; then
            echo "[watchdog] Bot exited cleanly (code=$EXIT_CODE). Stopping."
            exit 0
        fi
    done

    echo "[watchdog] Bot crashed (code=$EXIT_CODE). Restarting in ${RESTART_DELAY}s..."
    sleep "$RESTART_DELAY"
done
