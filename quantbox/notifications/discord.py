"""
Discord webhook notification system for QuantBox.

Provides rich embed notifications for live trading events:
- Bot start/stop
- Position open/close (enhanced with slot, leverage, balance info)
- Cycle summary (slot status dashboard)
- Daily report (cumulative statistics)
- Warnings (API errors, insufficient balance, drawdown alerts)

환경변수:
    DISCORD_WEBHOOK_URL: Discord 웹훅 URL (미설정 시 알림 비활성화)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from quantbox.core.divergence_monitor import DivergenceAlert

logger = logging.getLogger("DiscordNotifier")

# Discord embed color palette
COLOR_GREEN = 0x00C853  # 수익 / 정상
COLOR_RED = 0xFF1744  # 손실 / 위험
COLOR_YELLOW = 0xFFD600  # 경고
COLOR_BLUE = 0x2979FF  # 정보
COLOR_GRAY = 0x9E9E9E  # 중립


class DiscordNotifier:
    """Discord 웹훅 알림 전송.

    모든 send 계열 메서드는 webhook_url이 없으면 조용히 무시한다.
    네트워크 에러가 트레이딩 로직을 방해해서는 안 되므로 모든 전송은
    try/except로 보호한다.

    Args:
        webhook_url: Discord 웹훅 URL. None이면 환경변수에서 읽는다.
    """

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
        # 일일 리포트 추적용
        self._last_daily_report: datetime | None = None
        if self.enabled:
            logger.info("Discord notifications enabled")

    def send(self, content: str | None = None, embed: dict | None = None) -> bool:
        """메시지 전송 (low-level).

        Args:
            content: 일반 텍스트 메시지.
            embed: Discord embed dict.

        Returns:
            True if 전송 성공, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            data: dict = {}
            if content:
                data["content"] = content
            if embed:
                data["embeds"] = [embed]

            r = requests.post(self.webhook_url, json=data, timeout=5)
            return r.status_code == 204
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")
            return False

    # ==================== Bot Lifecycle ====================

    def notify_start(self, mode: str, balance: float, leverage: int = 1, max_traders: int = 5):
        """봇 시작 알림.

        Args:
            mode: PAPER 또는 LIVE.
            balance: 시작 잔고.
            leverage: 레버리지 배수.
            max_traders: 최대 동시 트레이더 수.
        """
        self.send(
            embed={
                "title": "🚀 QuantBox 트레이딩 시작",
                "color": COLOR_GREEN,
                "fields": [
                    {"name": "모드", "value": mode, "inline": True},
                    {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                    {"name": "레버리지", "value": f"{leverage}x", "inline": True},
                    {"name": "최대 슬롯", "value": str(max_traders), "inline": True},
                    {"name": "시작 시간", "value": _now_str(), "inline": True},
                ],
            }
        )

    def notify_stop(self, reason: str, total_pnl: float = 0.0, n_trades: int = 0):
        """봇 종료 알림.

        Args:
            reason: 종료 사유 (user interrupt, error, etc.).
            total_pnl: 총 누적 PnL.
            n_trades: 총 트레이드 수.
        """
        color = COLOR_GREEN if total_pnl >= 0 else COLOR_RED
        self.send(
            embed={
                "title": "🛑 QuantBox 트레이딩 종료",
                "color": color,
                "fields": [
                    {"name": "사유", "value": reason, "inline": True},
                    {"name": "총 PnL", "value": f"${total_pnl:+,.2f}", "inline": True},
                    {"name": "총 거래", "value": str(n_trades), "inline": True},
                    {"name": "종료 시간", "value": _now_str(), "inline": True},
                ],
            }
        )

    # ==================== Position Events ====================

    def notify_open(
        self,
        symbol1: str,
        symbol2: str,
        direction: str,
        zscore: float,
        capital: float,
        slot_id: int = 0,
        leverage: int = 1,
        margin: float = 0.0,
        available_after: float = 0.0,
    ):
        """포지션 오픈 알림 (강화).

        Args:
            symbol1: 첫 번째 심볼.
            symbol2: 두 번째 심볼.
            direction: 방향 문자열 (e.g. "[S0] LONG Spread").
            zscore: 진입 z-score.
            capital: 노셔널 (레버리지 반영 투자금).
            slot_id: 슬롯 ID.
            leverage: 레버리지 배수.
            margin: 마진 (실제 잠긴 자본).
            available_after: 진입 후 가용 잔고.
        """
        color = COLOR_GREEN if "LONG" in direction.upper() else COLOR_RED
        fields = [
            {"name": "슬롯", "value": f"S{slot_id}", "inline": True},
            {"name": "페어", "value": f"{symbol1}/{symbol2}", "inline": True},
            {"name": "방향", "value": direction, "inline": True},
            {"name": "Z-Score", "value": f"{zscore:.2f}", "inline": True},
            {"name": "노셔널", "value": f"${capital:,.2f}", "inline": True},
            {"name": "마진", "value": f"${margin:,.2f}", "inline": True},
        ]
        if leverage > 1:
            fields.append({"name": "레버리지", "value": f"{leverage}x", "inline": True})
        if available_after > 0:
            fields.append({"name": "잔여 가용", "value": f"${available_after:,.2f}", "inline": True})

        self.send(
            embed={
                "title": "📈 포지션 오픈",
                "color": color,
                "fields": fields,
                "footer": {"text": _now_str()},
            }
        )

    def notify_close(
        self,
        symbol1: str,
        symbol2: str,
        pnl: float,
        pnl_pct: float,
        reason: str,
        slot_id: int = 0,
        holding_time: str = "",
        cumulative_pnl: float = 0.0,
        balance_after: float = 0.0,
    ):
        """포지션 청산 알림 (강화).

        Args:
            symbol1: 첫 번째 심볼.
            symbol2: 두 번째 심볼.
            pnl: 해당 트레이드 PnL ($).
            pnl_pct: 해당 트레이드 PnL (%).
            reason: 청산 사유.
            slot_id: 슬롯 ID.
            holding_time: 보유 시간 문자열 (e.g. "4.2h").
            cumulative_pnl: 세션 누적 PnL.
            balance_after: 청산 후 잔고.
        """
        color = COLOR_GREEN if pnl >= 0 else COLOR_RED
        emoji = "✅" if pnl >= 0 else "❌"

        fields = [
            {"name": "슬롯", "value": f"S{slot_id}", "inline": True},
            {"name": "페어", "value": f"{symbol1}/{symbol2}", "inline": True},
            {"name": "PnL", "value": f"${pnl:+,.2f} ({pnl_pct:+.2f}%)", "inline": True},
            {"name": "사유", "value": reason, "inline": True},
        ]
        if holding_time:
            fields.append({"name": "보유 시간", "value": holding_time, "inline": True})
        if cumulative_pnl != 0.0:
            fields.append({"name": "누적 PnL", "value": f"${cumulative_pnl:+,.2f}", "inline": True})
        if balance_after > 0:
            fields.append({"name": "잔고", "value": f"${balance_after:,.2f}", "inline": True})

        self.send(
            embed={
                "title": f"{emoji} 포지션 청산",
                "color": color,
                "fields": fields,
                "footer": {"text": _now_str()},
            }
        )

    # ==================== Cycle Summary ====================

    def notify_cycle_summary(
        self,
        slots_info: list[dict],
        available_capital: float,
        total_capital: float,
        total_unrealized_pnl: float = 0.0,
        session_pnl: float = 0.0,
        n_trades_session: int = 0,
    ):
        """사이클 요약 알림 — 매 run_cycle 완료 시.

        슬롯 상태를 한 눈에 보여주는 대시보드.

        Args:
            slots_info: 슬롯별 상태 리스트.
                Each dict: {slot_id, status, pair, unrealized_pnl, holding_hours, entry_z, current_z}
            available_capital: 가용 잔고.
            total_capital: 총 자본.
            total_unrealized_pnl: 전체 미실현 PnL 합계.
            session_pnl: 세션 누적 실현 PnL.
            n_trades_session: 세션 내 거래 수.
        """
        # Build slot status lines
        slot_lines = []
        for s in slots_info:
            if s["status"] == "active":
                pnl_str = f"${s.get('unrealized_pnl', 0):+,.2f}"
                hours = s.get("holding_hours", 0)
                pair = s.get("pair", "?/?")
                z_str = f"z={s.get('current_z', 0):.1f}"
                slot_lines.append(f"🟢 S{s['slot_id']}: {pair} | {pnl_str} | {hours:.1f}h | {z_str}")
            else:
                slot_lines.append(f"⚪ S{s['slot_id']}: 대기")

        slots_text = "\n".join(slot_lines) if slot_lines else "없음"

        # Decide color: green if unrealized+session positive, red if negative
        net = total_unrealized_pnl + session_pnl
        color = COLOR_GREEN if net >= 0 else COLOR_RED

        capital_pct = (available_capital / total_capital * 100) if total_capital > 0 else 0

        fields = [
            {"name": "📊 슬롯 상태", "value": f"```\n{slots_text}\n```", "inline": False},
            {"name": "가용 자본", "value": f"${available_capital:,.2f} ({capital_pct:.0f}%)", "inline": True},
            {"name": "총 자본", "value": f"${total_capital:,.2f}", "inline": True},
            {"name": "미실현 PnL", "value": f"${total_unrealized_pnl:+,.2f}", "inline": True},
        ]
        if session_pnl != 0.0 or n_trades_session > 0:
            fields.append({"name": "세션 PnL", "value": f"${session_pnl:+,.2f} ({n_trades_session}건)", "inline": True})

        self.send(
            embed={
                "title": "📋 사이클 요약",
                "color": color,
                "fields": fields,
                "footer": {"text": _now_str()},
            }
        )

    # ==================== Daily Report ====================

    def notify_daily_report(
        self,
        total_balance: float,
        daily_pnl: float,
        daily_return_pct: float,
        cumulative_pnl: float,
        cumulative_return_pct: float,
        n_trades_today: int,
        n_trades_total: int,
        win_rate: float,
        max_drawdown_pct: float,
        active_slots: int,
        max_slots: int,
        leverage: int = 1,
    ):
        """24시간 일일 리포트.

        Args:
            total_balance: 현재 총 잔고.
            daily_pnl: 당일 PnL ($).
            daily_return_pct: 당일 수익률 (%).
            cumulative_pnl: 누적 PnL ($).
            cumulative_return_pct: 누적 수익률 (%).
            n_trades_today: 당일 거래 수.
            n_trades_total: 총 거래 수.
            win_rate: 전체 승률 (%).
            max_drawdown_pct: 최대 낙폭 (%).
            active_slots: 현재 활성 슬롯 수.
            max_slots: 최대 슬롯 수.
            leverage: 레버리지 배수.
        """
        color = COLOR_GREEN if daily_pnl >= 0 else COLOR_RED
        emoji = "📈" if daily_pnl >= 0 else "📉"

        self.send(
            embed={
                "title": f"{emoji} 일일 리포트",
                "color": color,
                "fields": [
                    {"name": "총 잔고", "value": f"${total_balance:,.2f}", "inline": True},
                    {"name": "당일 PnL", "value": f"${daily_pnl:+,.2f} ({daily_return_pct:+.2f}%)", "inline": True},
                    {
                        "name": "누적 PnL",
                        "value": f"${cumulative_pnl:+,.2f} ({cumulative_return_pct:+.2f}%)",
                        "inline": True,
                    },
                    {"name": "당일 거래", "value": str(n_trades_today), "inline": True},
                    {"name": "총 거래", "value": str(n_trades_total), "inline": True},
                    {"name": "승률", "value": f"{win_rate:.1f}%", "inline": True},
                    {"name": "최대 DD", "value": f"{max_drawdown_pct:.2f}%", "inline": True},
                    {"name": "슬롯", "value": f"{active_slots}/{max_slots}", "inline": True},
                    {"name": "레버리지", "value": f"{leverage}x", "inline": True},
                ],
                "footer": {"text": _now_str()},
            }
        )
        self._last_daily_report = datetime.now(timezone.utc)

    def should_send_daily_report(self) -> bool:
        """24시간마다 일일 리포트 전송 여부 확인.

        Returns:
            True if 마지막 리포트로부터 24시간 이상 경과.
        """
        if self._last_daily_report is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_daily_report).total_seconds()
        return elapsed >= 24 * 3600

    # ==================== Warnings ====================

    def notify_error(self, error: str):
        """에러 알림.

        Args:
            error: 에러 메시지 (500자 제한).
        """
        self.send(
            embed={
                "title": "🚨 에러 발생",
                "color": COLOR_RED,
                "description": error[:500],
                "footer": {"text": _now_str()},
            }
        )

    def notify_warning(self, title: str, message: str):
        """경고 알림.

        Args:
            title: 경고 제목.
            message: 경고 상세 메시지.
        """
        self.send(
            embed={
                "title": f"⚠️ {title}",
                "color": COLOR_YELLOW,
                "description": message[:500],
                "footer": {"text": _now_str()},
            }
        )

    def notify_insufficient_balance(self, available: float, required: float):
        """잔고 부족 경고.

        Args:
            available: 가용 잔고.
            required: 최소 필요 금액.
        """
        self.send(
            embed={
                "title": "⚠️ 잔고 부족",
                "color": COLOR_YELLOW,
                "fields": [
                    {"name": "가용 잔고", "value": f"${available:,.2f}", "inline": True},
                    {"name": "최소 필요", "value": f"${required:,.2f}", "inline": True},
                ],
                "footer": {"text": _now_str()},
            }
        )

    def notify_all_slots_occupied(self, slots_info: list[dict]):
        """모든 슬롯 사용 중 알림.

        Args:
            slots_info: 슬롯별 상태 리스트.
        """
        slot_lines = []
        for s in slots_info:
            pair = s.get("pair", "?/?")
            hours = s.get("holding_hours", 0)
            slot_lines.append(f"S{s['slot_id']}: {pair} ({hours:.1f}h)")

        self.send(
            embed={
                "title": "🔒 모든 슬롯 사용 중",
                "color": COLOR_YELLOW,
                "description": "\n".join(slot_lines),
                "footer": {"text": _now_str()},
            }
        )

    def notify_drawdown_alert(self, current_dd_pct: float, threshold_pct: float, balance: float):
        """최대 낙폭 임계치 경고.

        Args:
            current_dd_pct: 현재 낙폭 (%).
            threshold_pct: 경고 임계치 (%).
            balance: 현재 잔고.
        """
        self.send(
            embed={
                "title": "🔴 낙폭 경고",
                "color": COLOR_RED,
                "fields": [
                    {"name": "현재 낙폭", "value": f"{current_dd_pct:.2f}%", "inline": True},
                    {"name": "임계치", "value": f"{threshold_pct:.2f}%", "inline": True},
                    {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                ],
                "footer": {"text": _now_str()},
            }
        )

    def notify_divergence_alert(self, alert: "DivergenceAlert") -> None:
        """라이브-백테스트 성과 괴리 경보.

        Args:
            alert: DivergenceAlert from DivergenceMonitor.check().
        """
        triggered = ", ".join(t.upper() for t in alert.triggered_by)
        pf_gap = alert.pf_live - alert.pf_baseline
        wr_gap_pp = (alert.wr_live - alert.wr_baseline) * 100
        self.send(
            embed={
                "title": f"⚠️ Live-Backtest 괴리 경보 [{triggered}]",
                "description": alert.message,
                "color": COLOR_RED,
                "fields": [
                    {
                        "name": "PF (라이브 / 백테스트)",
                        "value": f"{alert.pf_live:.3f} / {alert.pf_baseline:.3f}  ({pf_gap:+.3f})",
                        "inline": True,
                    },
                    {
                        "name": "WR (라이브 / 백테스트)",
                        "value": f"{alert.wr_live * 100:.1f}% / {alert.wr_baseline * 100:.1f}%  ({wr_gap_pp:+.1f}pp)",
                        "inline": True,
                    },
                    {
                        "name": "거래 수",
                        "value": str(alert.n_trades),
                        "inline": True,
                    },
                ],
                "footer": {"text": _now_str()},
            }
        )


def _now_str() -> str:
    """현재 UTC 시각 문자열."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
