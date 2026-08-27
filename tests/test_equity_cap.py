"""원금 캡핑 — 선물 계좌 시드 고정 + 초과 수익 출금 알림.

트레이더 룰북(2026-08-27)의 자금관리 항목: "선물 계좌에는 정해진 최소 시드만 유지하고
수익금은 상시 출금해 강제청산 리스크를 원천 차단한다."

봇이 대신 출금할 수는 없다(출금 권한을 API 키에 주는 것 자체가 더 큰 위험이다).
대신 **사이징 기준을 상한으로 묶는다** — 잔고가 캡을 넘어도 그 위로는 포지션을 키우지
않고, 넘은 만큼을 디스코드로 알려 사람이 빼게 한다.

불변식 2개:
  1. 캡은 **사이징에만** 걸린다. 드로다운·HWM 계산은 실제 잔고를 봐야 한다
     (캡 잔고로 DD를 재면 손실을 과소평가한다).
  2. 캡이 0이면 완전히 비활성 — 기존 동작과 1비트도 달라지지 않는다.
"""

from __future__ import annotations

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig


@pytest.fixture
def scalper(vs_scalper):
    return vs_scalper


class TestSizingBalanceCap:
    def test_disabled_by_default(self):
        assert VolStraddleConfig().equity_cap_usdt == 0.0

    def test_zero_cap_is_passthrough(self, scalper):
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=0.0)
        for bal in (0.0, 100.0, 1_000_000.0):
            assert scalper._sizing_balance(bal) == bal

    def test_balance_below_cap_untouched(self, scalper):
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=1000.0)
        assert scalper._sizing_balance(750.0) == 750.0

    def test_balance_above_cap_is_clamped(self, scalper):
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=1000.0)
        assert scalper._sizing_balance(1800.0) == 1000.0

    def test_exactly_at_cap(self, scalper):
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=1000.0)
        assert scalper._sizing_balance(1000.0) == 1000.0

    def test_negative_cap_rejected_by_config(self):
        with pytest.raises(Exception):
            VolStraddleConfig(equity_cap_usdt=-1.0)


class TestWithdrawalAlert:
    def test_alert_sent_when_over_cap(self, scalper, monkeypatch):
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=1000.0)
        sent: list[tuple[str, str]] = []
        monkeypatch.setattr(scalper.discord, "notify_warning", lambda t, m: sent.append((t, m)))

        scalper._sizing_balance(1500.0)

        assert len(sent) == 1
        assert "500" in sent[0][1], "초과분 금액이 메시지에 있어야 사람이 얼마를 뺄지 안다"

    def test_no_alert_when_under_cap(self, scalper, monkeypatch):
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=1000.0)
        sent = []
        monkeypatch.setattr(scalper.discord, "notify_warning", lambda t, m: sent.append((t, m)))
        scalper._sizing_balance(900.0)
        assert sent == []

    def test_alert_respects_cooldown(self, scalper, monkeypatch):
        """매 사이클(15m)마다 같은 알림을 쏘면 알림이 무의미해진다."""
        scalper.vs_config = VolStraddleConfig(equity_cap_usdt=1000.0)
        sent = []
        monkeypatch.setattr(scalper.discord, "notify_warning", lambda t, m: sent.append((t, m)))
        for _ in range(5):
            scalper._sizing_balance(1500.0)
        assert len(sent) == 1


class TestCapDoesNotDistortRisk:
    def test_drawdown_uses_real_balance(self, scalper):
        """캡은 사이징 전용 — DD/HWM 경로에 스며들면 손실을 과소평가한다."""
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[1] / "quantbox" / "scalper.py").read_text()
        cycle = src.split("def _cycle(self")[1].split("def _scan_entries")[0]
        assert "_check_risk_and_halt(balance)" in cycle, "DD 체크가 실제 잔고를 안 받는다"
        assert "_log_cycle_header(balance)" in cycle
        assert "_sizing_balance(balance)" in cycle, "진입 경로가 캡 잔고를 안 쓴다"
