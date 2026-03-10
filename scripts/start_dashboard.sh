#!/usr/bin/env bash
# QuantBox Dashboard 시작/중지 스크립트
# 사용: ./scripts/start_dashboard.sh [start|stop|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/logs/dashboard.pid"
LOG_FILE="$PROJECT_DIR/logs/dashboard.log"
PORT=8501

start() {
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE")
        if [ -n "$EXISTING_PID" ] && [ -d "/proc/$EXISTING_PID" ]; then
            echo "대시보드 이미 실행 중 (PID=$EXISTING_PID)"
            echo "→ http://localhost:$PORT"
            return
        fi
    fi

    mkdir -p "$PROJECT_DIR/logs"
    cd "$PROJECT_DIR"
    nohup uv run streamlit run scripts/dashboard.py \
        --server.port "$PORT" \
        --server.headless true \
        --server.address 0.0.0.0 \
        > "$LOG_FILE" 2>&1 &

    DASHBOARD_PID=$!
    echo "$DASHBOARD_PID" > "$PID_FILE"
    sleep 2

    if [ -d "/proc/$DASHBOARD_PID" ]; then
        LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
        echo "✅ 대시보드 시작됨 (PID=$DASHBOARD_PID)"
        echo "   → http://localhost:$PORT  (이 컴퓨터)"
        echo "   → http://$LOCAL_IP:$PORT  (같은 네트워크)"
    else
        echo "❌ 시작 실패. 로그 확인: tail -20 $LOG_FILE"
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE")
        if [ -n "$EXISTING_PID" ] && [ -d "/proc/$EXISTING_PID" ]; then
            kill "$EXISTING_PID"
            rm -f "$PID_FILE"
            echo "⏹ 대시보드 종료됨"
            return
        fi
    fi
    echo "대시보드가 실행 중이 아닙니다"
    rm -f "$PID_FILE" 2>/dev/null
}

status() {
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE")
        if [ -n "$EXISTING_PID" ] && [ -d "/proc/$EXISTING_PID" ]; then
            LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
            echo "✅ 실행 중 (PID=$EXISTING_PID)"
            echo "   → http://localhost:$PORT"
            echo "   → http://$LOCAL_IP:$PORT"
            return
        fi
    fi
    echo "⏹ 실행 중 아님"
}

restart() {
    stop
    sleep 1
    start
}

case "${1:-start}" in
    start)   start   ;;
    stop)    stop    ;;
    status)  status  ;;
    restart) restart ;;
    *) echo "사용법: $0 [start|stop|restart|status]" ;;
esac
