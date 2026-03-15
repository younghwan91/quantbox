"""r85 실행 품질 분석 스크립트 테스트."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

# research 스크립트를 직접 import하기 위해 sys.path 설정은 conftest에서 처리
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from research.active.r81_r85.r85_execution_quality import (
    compute_execution_gaps,
    compute_pair_metrics,
    generate_report,
    load_live_trades,
    load_recon_events,
)


# ── 샘플 데이터 픽스처 ──────────────────────────────────────────────────────────


def _sample_trades() -> list[dict]:
    """테스트용 스트래들 거래 샘플."""
    return [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "strategy_type": "vol_straddle",
            "direction": "LONG",
            "entry_price": 50000.0,
            "exit_price": 52500.0,
            "pnl_pct": 0.05,
            "pnl_usd": 50.0,
            "mfe_pct": 0.06,
            "mae_pct": -0.005,
            "notional": 1000.0,
            "bars_held": 10,
            "entry_time": "2026-03-01T10:00:00",
            "exit_time": "2026-03-01T12:00:00",
            "exit_reason": "trail_stop",
            "balance_before": 1000.0,
            "balance_after": 1050.0,
            "session": "live",
            "commit_sha": "abc1234",
        },
        {
            "id": 2,
            "symbol": "BTCUSDT",
            "strategy_type": "vol_straddle",
            "direction": "SHORT",
            "entry_price": 50000.0,
            "exit_price": 49250.0,
            "pnl_pct": -0.015,
            "pnl_usd": -15.0,
            "mfe_pct": 0.002,
            "mae_pct": -0.016,
            "notional": 1000.0,
            "bars_held": 8,
            "entry_time": "2026-03-01T10:00:00",
            "exit_time": "2026-03-01T11:50:00",
            "exit_reason": "sl_hit",
            "balance_before": 1000.0,
            "balance_after": 985.0,
            "session": "live",
            "commit_sha": "abc1234",
        },
        # 두 번째 페어 (다른 심볼)
        {
            "id": 3,
            "symbol": "ETHUSDT",
            "strategy_type": "vol_straddle",
            "direction": "LONG",
            "entry_price": 3000.0,
            "exit_price": 2955.0,
            "pnl_pct": -0.015,
            "pnl_usd": -15.0,
            "mfe_pct": 0.001,
            "mae_pct": -0.016,
            "notional": 1000.0,
            "bars_held": 5,
            "entry_time": "2026-03-01T14:00:00",
            "exit_time": "2026-03-01T15:15:00",
            "exit_reason": "sl_hit",
            "balance_before": 985.0,
            "balance_after": 970.0,
            "session": "live",
            "commit_sha": "abc1234",
        },
        {
            "id": 4,
            "symbol": "ETHUSDT",
            "strategy_type": "vol_straddle",
            "direction": "SHORT",
            "entry_price": 3000.0,
            "exit_price": 2850.0,
            "pnl_pct": 0.05,
            "pnl_usd": 50.0,
            "mfe_pct": 0.055,
            "mae_pct": -0.002,
            "notional": 1000.0,
            "bars_held": 12,
            "entry_time": "2026-03-01T14:00:00",
            "exit_time": "2026-03-01T16:00:00",
            "exit_reason": "trail_stop",
            "balance_before": 985.0,
            "balance_after": 1035.0,
            "session": "live",
            "commit_sha": "abc1234",
        },
    ]


def _sample_recon_events() -> list[dict]:
    """테스트용 정산 이벤트 샘플."""
    return [
        {
            "id": 1,
            "timestamp": 1740830400.0,
            "event_type": "ghost_detected",
            "symbol": "SOLUSDT",
            "action": "ghost detected",
            "result": "success",
            "details": {},
            "session_id": "session1",
        },
        {
            "id": 2,
            "timestamp": 1740830500.0,
            "event_type": "ghost_closed",
            "symbol": "SOLUSDT",
            "action": "auto-closed at market",
            "result": "success",
            "details": {"pnl_usd": -8.5},
            "session_id": "session1",
        },
        {
            "id": 3,
            "timestamp": 1740831000.0,
            "event_type": "bracket_repair",
            "symbol": "BTCUSDT",
            "action": "re-placed SL",
            "result": "success",
            "details": {},
            "session_id": "session1",
        },
        {
            "id": 4,
            "timestamp": 1740831100.0,
            "event_type": "bracket_repair_fail",
            "symbol": "ETHUSDT",
            "action": "re-place failed",
            "result": "failure",
            "details": {},
            "session_id": "session1",
        },
        {
            "id": 5,
            "timestamp": 1740831200.0,
            "event_type": "server_sl_fill",
            "symbol": "BTCUSDT",
            "action": "server SL triggered",
            "result": "success",
            "details": {},
            "session_id": "session1",
        },
        {
            "id": 6,
            "timestamp": 1740831300.0,
            "event_type": "server_tp_fill",
            "symbol": "ETHUSDT",
            "action": "server TP triggered",
            "result": "success",
            "details": {},
            "session_id": "session1",
        },
    ]


# ── load_live_trades 테스트 ────────────────────────────────────────────────────


class TestLoadLiveTrades:
    def test_missing_db_returns_empty(self, tmp_path: Path) -> None:
        """DB 파일이 없으면 빈 리스트 반환."""
        result = load_live_trades(str(tmp_path / "nonexistent.sqlite"))
        assert result == []

    def test_loads_straddle_trades(self, tmp_path: Path) -> None:
        """strategy_type이 straddle/vol_straddle인 거래만 반환."""
        db = tmp_path / "journal.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY, symbol TEXT, strategy_type TEXT,
                direction TEXT, entry_price REAL, exit_price REAL,
                pnl_pct REAL, pnl_usd REAL, mfe_pct REAL, mae_pct REAL,
                notional REAL, bars_held INTEGER, entry_time TEXT,
                exit_time TEXT, exit_reason TEXT, balance_before REAL,
                balance_after REAL, session TEXT, commit_sha TEXT
            )
        """)
        # 스트래들 거래 2건 + 다른 전략 1건
        conn.execute(
            "INSERT INTO trades VALUES (1,'BTCUSDT','vol_straddle','LONG',"
            "50000,52500,0.05,50,0.06,-0.005,1000,10,'2026-03-01T10:00:00',"
            "'2026-03-01T12:00:00','trail_stop',1000,1050,'live','abc')"
        )
        conn.execute(
            "INSERT INTO trades VALUES (2,'BTCUSDT','vr_squeeze','LONG',"
            "50000,52500,0.05,50,0.06,-0.005,1000,10,'2026-03-01T10:00:00',"
            "'2026-03-01T12:00:00','trail_stop',1000,1050,'live','abc')"
        )
        conn.commit()
        conn.close()

        # load_live_trades는 절대경로를 받으므로 ROOT-relative가 아닌 직접 테스트
        import importlib
        import research.active.r81_r85.r85_execution_quality as mod

        original_root = mod.ROOT
        mod.ROOT = tmp_path
        try:
            result = mod.load_live_trades("journal.sqlite")
        finally:
            mod.ROOT = original_root

        assert len(result) == 1
        assert result[0]["strategy_type"] == "vol_straddle"

    def test_invalid_db_returns_empty(self, tmp_path: Path) -> None:
        """손상된 DB는 빈 리스트 반환."""
        bad_db = tmp_path / "bad.sqlite"
        bad_db.write_bytes(b"not a sqlite file")

        import research.active.r81_r85.r85_execution_quality as mod

        original_root = mod.ROOT
        mod.ROOT = tmp_path
        try:
            result = mod.load_live_trades("bad.sqlite")
        finally:
            mod.ROOT = original_root

        assert result == []


# ── load_recon_events 테스트 ───────────────────────────────────────────────────


class TestLoadReconEvents:
    def test_missing_db_returns_empty(self, tmp_path: Path) -> None:
        """DB 파일이 없으면 빈 리스트 반환."""
        result = load_recon_events(str(tmp_path / "nonexistent.db"))
        assert result == []

    def test_loads_recon_events(self, tmp_path: Path) -> None:
        """recon.db에서 이벤트를 로드한다."""
        import json

        db_dir = tmp_path / "journal"
        db_dir.mkdir()
        db = db_dir / "recon.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE recon_events (
                id INTEGER PRIMARY KEY, timestamp REAL, event_type TEXT,
                symbol TEXT, action TEXT, result TEXT,
                details_json TEXT, session_id TEXT
            )
        """)
        conn.execute(
            "INSERT INTO recon_events VALUES (1, 1740830400.0, 'ghost_detected',"
            "'BTCUSDT', 'detected', 'success', '{}', 'sess1')"
        )
        conn.commit()
        conn.close()

        import research.active.r81_r85.r85_execution_quality as mod

        original_root = mod.ROOT
        mod.ROOT = tmp_path
        try:
            result = mod.load_recon_events("journal/recon.db")
        finally:
            mod.ROOT = original_root

        assert len(result) == 1
        assert result[0]["event_type"] == "ghost_detected"
        assert isinstance(result[0]["details"], dict)


# ── compute_pair_metrics 테스트 ────────────────────────────────────────────────


class TestComputePairMetrics:
    def test_full_pair_grouping(self) -> None:
        """동일 심볼 + 같은 시각의 LONG+SHORT가 하나의 페어로 묶인다."""
        trades = _sample_trades()
        pairs = compute_pair_metrics(trades)
        full_pairs = [p for p in pairs if p["pair_type"] == "full"]
        assert len(full_pairs) == 2

    def test_pair_pnl_sum(self) -> None:
        """페어 gross_pnl = long_pnl + short_pnl."""
        trades = _sample_trades()
        pairs = compute_pair_metrics(trades)
        full_pairs = [p for p in pairs if p["pair_type"] == "full"]
        for p in full_pairs:
            assert abs(p["gross_pnl_usd"] - (p["long_pnl_usd"] + p["short_pnl_usd"])) < 1e-9

    def test_fee_positive(self) -> None:
        """수수료는 항상 양수."""
        trades = _sample_trades()
        pairs = compute_pair_metrics(trades)
        for p in pairs:
            assert p["fee_usd"] >= 0.0

    def test_net_pnl_equals_gross_minus_fee(self) -> None:
        """net_pnl = gross_pnl - fee."""
        trades = _sample_trades()
        pairs = compute_pair_metrics(trades)
        for p in pairs:
            expected = p["gross_pnl_usd"] - p["fee_usd"]
            assert abs(p["net_pnl_usd"] - expected) < 1e-9

    def test_winning_losing_leg(self) -> None:
        """winning_leg >= losing_leg."""
        trades = _sample_trades()
        pairs = compute_pair_metrics(trades)
        full_pairs = [p for p in pairs if p["pair_type"] == "full"]
        for p in full_pairs:
            assert p["winning_leg_pnl"] >= p["losing_leg_pnl"]

    def test_orphan_single_leg(self) -> None:
        """LONG만 있는 경우 orphan으로 처리된다."""
        trades = [_sample_trades()[0]]  # LONG만
        pairs = compute_pair_metrics(trades)
        orphans = [p for p in pairs if p["pair_type"] == "orphan"]
        assert len(orphans) == 1
        assert orphans[0]["symbol"] == "BTCUSDT"

    def test_empty_trades(self) -> None:
        """빈 거래 목록은 빈 페어 목록 반환."""
        assert compute_pair_metrics([]) == []


# ── compute_execution_gaps 테스트 ─────────────────────────────────────────────


class TestComputeExecutionGaps:
    def test_counts_ghost_events(self) -> None:
        """ghost 감지/청산 건수가 정확히 집계된다."""
        recon = _sample_recon_events()
        gaps = compute_execution_gaps([], recon)
        assert gaps["ghost_detections"] == 1
        assert gaps["ghost_closed_count"] == 1

    def test_ghost_pnl_accumulated(self) -> None:
        """ghost 청산 PnL이 합산된다."""
        recon = _sample_recon_events()
        gaps = compute_execution_gaps([], recon)
        assert abs(float(gaps["total_ghost_pnl_usd"]) - (-8.5)) < 1e-9

    def test_counts_bracket_events(self) -> None:
        """bracket 수리 성공/실패가 집계된다."""
        recon = _sample_recon_events()
        gaps = compute_execution_gaps([], recon)
        assert gaps["bracket_repairs"] == 1
        assert gaps["bracket_repair_fails"] == 1

    def test_counts_server_fills(self) -> None:
        """서버 SL/TP 체결이 집계된다."""
        recon = _sample_recon_events()
        gaps = compute_execution_gaps([], recon)
        assert gaps["server_sl_fills"] == 1
        assert gaps["server_tp_fills"] == 1

    def test_ratchet_miss_detection(self) -> None:
        """MFE가 trail_activate 초과 + max_hold 청산 → ratchet 미스로 감지."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        trail_activate = VolStraddleConfig().s_trail_activate
        trades = [
            {
                "mfe_pct": trail_activate + 0.01,  # trail 활성화 수준 이상
                "exit_reason": "max_hold_exit",
            }
        ]
        gaps = compute_execution_gaps(trades, [])
        assert gaps["ratchet_miss_count"] == 1

    def test_no_ratchet_miss_when_sl_exit(self) -> None:
        """SL 청산은 ratchet 미스로 집계되지 않는다."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        trail_activate = VolStraddleConfig().s_trail_activate
        trades = [
            {
                "mfe_pct": trail_activate + 0.01,
                "exit_reason": "sl_hit",
            }
        ]
        gaps = compute_execution_gaps(trades, [])
        assert gaps["ratchet_miss_count"] == 0

    def test_empty_inputs(self) -> None:
        """빈 입력도 정상 처리된다."""
        gaps = compute_execution_gaps([], [])
        assert gaps["bracket_repairs"] == 0
        assert gaps["ghost_detections"] == 0


# ── generate_report 테스트 ────────────────────────────────────────────────────


class TestGenerateReport:
    def test_creates_markdown_file(self, tmp_path: Path) -> None:
        """리포트 파일이 생성된다."""
        output = tmp_path / "report.md"
        generate_report(compute_pair_metrics(_sample_trades()), {}, output)
        assert output.exists()

    def test_report_contains_required_sections(self, tmp_path: Path) -> None:
        """필수 섹션이 모두 포함된다: 요약, 실행 갭, 판정."""
        output = tmp_path / "report.md"
        pairs = compute_pair_metrics(_sample_trades())
        gaps = compute_execution_gaps(_sample_trades(), _sample_recon_events())
        generate_report(pairs, gaps, output)

        content = output.read_text(encoding="utf-8")
        assert "## 요약" in content
        assert "## 실행 갭 분류" in content
        assert "### 판정" in content

    def test_report_contains_pair_detail_section(self, tmp_path: Path) -> None:
        """페어별 상세 섹션이 포함된다."""
        output = tmp_path / "report.md"
        pairs = compute_pair_metrics(_sample_trades())
        generate_report(pairs, {}, output)
        content = output.read_text(encoding="utf-8")
        assert "## 페어별 상세" in content

    def test_report_is_valid_markdown(self, tmp_path: Path) -> None:
        """리포트가 비어있지 않고 마크다운 헤더로 시작한다."""
        output = tmp_path / "report.md"
        generate_report([], {}, output)
        content = output.read_text(encoding="utf-8")
        assert content.startswith("# r85:")
        assert len(content) > 100

    def test_report_shows_win_rate(self, tmp_path: Path) -> None:
        """승률이 리포트에 표시된다."""
        output = tmp_path / "report.md"
        pairs = compute_pair_metrics(_sample_trades())
        generate_report(pairs, {}, output)
        content = output.read_text(encoding="utf-8")
        assert "승률" in content

    def test_report_shows_execution_gap_numbers(self, tmp_path: Path) -> None:
        """실행 갭 건수가 리포트에 표시된다."""
        output = tmp_path / "report.md"
        gaps = compute_execution_gaps(_sample_trades(), _sample_recon_events())
        generate_report([], gaps, output)
        content = output.read_text(encoding="utf-8")
        assert "Ghost" in content
        assert "Bracket" in content

    def test_report_handles_empty_data_gracefully(self, tmp_path: Path) -> None:
        """데이터가 없어도 오류 없이 리포트가 생성된다."""
        output = tmp_path / "report.md"
        generate_report([], {}, output)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "데이터 없음" in content or "## 요약" in content
