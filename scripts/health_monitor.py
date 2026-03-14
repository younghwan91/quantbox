#!/usr/bin/env python3
"""QuantBox Health Monitor — 5분마다 이상 징후 감지 → Discord 즉시 알림.

감지 항목:
  1. Stale state      — scalper가 30분 이상 state를 저장 안 함 (멈춤 의심)
  2. Journal drift    — 최근 24h state 거래 vs journal.sqlite 건수 불일치
  3. Straddle orphan  — LONG/SHORT 페어 중 한쪽만 존재 (2바 이상 지속)
  4. DD 임계 초과     — 8% 경고 / 12% 위험
  5. Ghost position   — state에는 있지만 거래소에 없는 포지션

Usage:
    # 한 번만 실행 (cron용)
    python3 scripts/health_monitor.py --once

    # 루프 (5분 간격, 기본값)
    python3 scripts/health_monitor.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
STATE_PATH = ROOT / "data" / "state" / "scalper_state.json"
JOURNAL_PATH = ROOT / "data" / "journal.sqlite"

CHECK_INTERVAL = 300  # 5분

# DD 경보 임계값
DD_WARN_PCT = 0.08   # 8%  — 경고 (SoftTight 최소 단계 이미 진입)
DD_CRIT_PCT = 0.12   # 12% — 위험 (max_dd=15% 접근)

# Journal drift — 24h 기준 허용 오차
DRIFT_WARN = 2       # 2건 초과 → 경고
DRIFT_CRIT = 5       # 5건 초과 → 위험


# ── Discord ──────────────────────────────────────────────────────────────────

def _get_notifier():
    """Return DiscordNotifier instance (lazy import to avoid circular deps)."""
    try:
        from quantbox.notifications.discord import DiscordNotifier
        return DiscordNotifier()
    except Exception:
        return None


_notifier = None  # initialized once on first use


def alert(title: str, body: str, fields: list[dict] | None = None, critical: bool = False) -> None:
    global _notifier
    if _notifier is None:
        _notifier = _get_notifier()

    emoji = "🚨" if critical else "⚠️"
    color = 0xFF1744 if critical else 0xFF9800

    if _notifier and _notifier.enabled:
        embed = {
            "title": f"{emoji} {title}",
            "description": body,
            "color": color,
            "fields": fields or [],
            "footer": {"text": f"health_monitor • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"},
        }
        _notifier.send(embed=embed)
    else:
        print(f"{emoji} [ALERT] {title}: {body}", file=sys.stderr)

    print(f"[ALERT] {title}: {body}")


def ok_msg(msg: str) -> None:
    print(f"[OK]    {msg}")


# ── Binance helpers ───────────────────────────────────────────────────────────

def _get_client():
    try:
        from binance.client import Client
        return Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_API_SECRET"])
    except Exception:
        return None


def get_exchange_positions() -> dict[str, float] | None:
    """Return {SYMBOL:SIDE → notional_qty} for non-zero exchange positions."""
    c = _get_client()
    if not c:
        return None
    try:
        result = {}
        for p in c.futures_position_information():
            amt = float(p.get("positionAmt", 0))
            if abs(amt) > 1e-9:
                key = f"{p['symbol']}:{p['positionSide']}"
                result[key] = amt
        return result
    except Exception as e:
        print(f"[MONITOR] Binance positions error: {e}", file=sys.stderr)
        return None


def get_live_balance() -> float | None:
    """Wallet balance from Binance futures account."""
    c = _get_client()
    if not c:
        return None
    try:
        balances = c.futures_account_balance()
        usdt = next((b for b in balances if b["asset"] == "USDT"), None)
        return float(usdt["crossWalletBalance"]) if usdt else None
    except Exception as e:
        print(f"[MONITOR] Binance balance error: {e}", file=sys.stderr)
        return None


# ── State helpers ─────────────────────────────────────────────────────────────

def load_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception as e:
        print(f"[MONITOR] State load error: {e}", file=sys.stderr)
        return None


def _journal_count_24h() -> int:
    """Count live trades in journal.sqlite closed in the last 24h."""
    if not JOURNAL_PATH.exists():
        return 0
    try:
        cut = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        conn = sqlite3.connect(JOURNAL_PATH)
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_time >= ?", (cut,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _state_count_24h(history: list[dict]) -> int:
    """Count state trade_history entries closed in the last 24h."""
    cut = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    return sum(1 for t in history if t.get("time", "") >= cut)


# ── Health Checks ─────────────────────────────────────────────────────────────

def check_stale_state(state: dict) -> bool:
    """Alert if scalper hasn't saved state in > 30 minutes."""
    saved_at_str = state.get("saved_at", "")
    if not saved_at_str:
        return False
    try:
        saved_at = datetime.fromisoformat(saved_at_str)
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - saved_at
        age_min = int(age.total_seconds() / 60)
        if age > timedelta(minutes=30):
            alert(
                "Scalper 응답 없음",
                f"마지막 state 저장이 **{age_min}분** 전입니다. 스칼퍼가 멈췄을 수 있습니다.",
                [{"name": "saved_at", "value": saved_at_str[:19], "inline": True},
                 {"name": "경과", "value": f"{age_min}분", "inline": True}],
                critical=True,
            )
            return True
    except Exception:
        pass
    ok_msg(f"State 최신 (saved {saved_at_str[:19]})")
    return False


def check_journal_drift(state: dict) -> bool:
    """Alert if recent 24h trades in state significantly exceed journal count."""
    history = state.get("trade_history", [])
    s_count = _state_count_24h(history)
    j_count = _journal_count_24h()
    drift = s_count - j_count

    if drift > DRIFT_CRIT:
        alert(
            "Journal Drift 심각",
            f"최근 24h: state={s_count}건, DB={j_count}건. **{drift}건** 누락.",
            [{"name": "state(24h)", "value": str(s_count), "inline": True},
             {"name": "DB(24h)", "value": str(j_count), "inline": True},
             {"name": "누락", "value": str(drift), "inline": True}],
            critical=True,
        )
        return True
    elif drift > DRIFT_WARN:
        alert(
            "Journal Drift 경고",
            f"최근 24h: state={s_count}건, DB={j_count}건. {drift}건 누락.",
            [{"name": "state(24h)", "value": str(s_count), "inline": True},
             {"name": "DB(24h)", "value": str(j_count), "inline": True},
             {"name": "누락", "value": str(drift), "inline": True}],
        )
        return True
    ok_msg(f"Journal 정상 (24h state={s_count}, DB={j_count}, drift={drift})")
    return False


def check_straddle_orphans(state: dict) -> bool:
    """Alert if a straddle position has no matching pair (persisted ≥ 2 bars)."""
    positions = state.get("positions", {})
    straddle_pos = {
        k: v for k, v in positions.items()
        if v.get("is_straddle") or v.get("strategy_type") == "straddle"
    }

    by_sym: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, pos in straddle_pos.items():
        sym = key.split(":")[0]
        side = key.split(":")[1] if ":" in key else "UNKNOWN"
        bars = pos.get("bars_held", 0)
        by_sym[sym].append((side, bars))

    orphans = []
    for sym, entries in by_sym.items():
        if len(entries) == 1:
            side, bars = entries[0]
            # Only alert if persisted ≥ 2 bars (not just opened and waiting)
            if bars >= 2:
                orphans.append(f"{sym}:{side}(bars={bars})")

    if orphans:
        alert(
            "Straddle 고아 포지션",
            f"페어 없이 {len(orphans)}개 포지션이 방치 중: {', '.join(orphans)}",
            [{"name": "고아", "value": "\n".join(orphans), "inline": False}],
            critical=True,
        )
        return True
    ok_msg(f"Straddle 페어 정상 ({len(by_sym)}쌍)")
    return False


def check_drawdown(state: dict) -> bool:
    """Alert when DD approaches critical thresholds."""
    hwm = state.get("high_water_mark", 0)
    if not hwm or hwm <= 0:
        return False

    paper_mode = state.get("paper_mode", True)

    # Live mode: always use Binance API balance
    if not paper_mode:
        balance = get_live_balance()
        src = "Binance API"
    else:
        balance = state.get("paper_balance")
        src = "paper_balance"

    if not balance:
        ok_msg("DD 확인 불가 (balance 조회 실패)")
        return False

    dd = (hwm - balance) / hwm
    dd_pct = dd * 100

    if dd >= DD_CRIT_PCT:
        alert(
            f"DD 위험 {dd_pct:.1f}%",
            f"HWM ${hwm:,.0f} → 현재 ${balance:,.0f} ({src}). max_dd=15% 접근 중!",
            [{"name": "HWM", "value": f"${hwm:,.2f}", "inline": True},
             {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
             {"name": "DD", "value": f"{dd_pct:.1f}%", "inline": True}],
            critical=True,
        )
        return True
    elif dd >= DD_WARN_PCT:
        alert(
            f"DD 경고 {dd_pct:.1f}%",
            f"SoftTight 0.4× 작동 중. HWM ${hwm:,.0f} → 현재 ${balance:,.0f} ({src}).",
            [{"name": "HWM", "value": f"${hwm:,.2f}", "inline": True},
             {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
             {"name": "DD", "value": f"{dd_pct:.1f}%", "inline": True}],
        )
        return True
    ok_msg(f"DD 정상 {dd_pct:.1f}% (HWM ${hwm:,.0f} / bal ${balance:,.0f} via {src})")
    return False


def check_ghost_positions(state: dict) -> bool:
    """Alert if state has positions not found on exchange (≥ 2 bars old)."""
    state_positions = state.get("positions", {})
    exchange_pos = get_exchange_positions()

    if exchange_pos is None:
        ok_msg("Ghost 체크 건너뜀 (Binance API 미응답)")
        return False

    ghosts = []
    for pos_key, pos in state_positions.items():
        sym = pos_key.split(":")[0] if ":" in pos_key else pos_key
        side = pos_key.split(":")[1] if ":" in pos_key else None
        ex_key = f"{sym}:{side}" if side else sym
        bars = pos.get("bars_held", 0)
        if ex_key not in exchange_pos and bars >= 2:
            ghosts.append(f"{pos_key}(bars={bars})")

    if ghosts:
        alert(
            "Ghost 포지션 감지",
            f"state에 있지만 거래소에 없는 포지션 {len(ghosts)}개: {', '.join(ghosts)}",
            [{"name": "Ghost", "value": "\n".join(ghosts), "inline": False}],
            critical=True,
        )
        return True
    ok_msg(f"Ghost 없음 (state={len(state_positions)}, exchange={len(exchange_pos)})")
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run_checks() -> int:
    """Run all health checks. Returns number of alerts fired."""
    print(f"\n{'='*58}")
    print(f"[MONITOR] {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*58}")

    state = load_state()
    if state is None:
        alert(
            "State 파일 없음",
            f"`{STATE_PATH}` 를 찾을 수 없습니다. 스칼퍼가 한 번도 실행되지 않았거나 경로가 잘못됐습니다.",
            critical=True,
        )
        return 1

    n_alerts = 0
    n_alerts += int(check_stale_state(state))
    n_alerts += int(check_journal_drift(state))
    n_alerts += int(check_straddle_orphans(state))
    n_alerts += int(check_drawdown(state))
    n_alerts += int(check_ghost_positions(state))

    if n_alerts == 0:
        print("[MONITOR] ✅ All checks passed")
    else:
        print(f"[MONITOR] ⚠️  {n_alerts} alert(s) fired → Discord 전송 완료")
    return n_alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantBox Health Monitor")
    parser.add_argument("--once", action="store_true", help="Run once and exit (cron 용)")
    parser.add_argument(
        "--interval", type=int, default=CHECK_INTERVAL,
        help=f"점검 주기 (초, 기본값: {CHECK_INTERVAL})",
    )
    args = parser.parse_args()

    if args.once:
        sys.exit(min(run_checks(), 1))

    print(f"[MONITOR] 연속 모드 시작 (주기={args.interval}s). Ctrl+C로 종료.")
    while True:
        try:
            run_checks()
        except KeyboardInterrupt:
            print("\n[MONITOR] 종료")
            break
        except Exception as e:
            print(f"[MONITOR] 예기치 않은 오류: {e}", file=sys.stderr)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
