"""Tests for straddle pair coordination config params (Step 0.2).

Verifies that all new retry/orphan config params have correct defaults
and can be read from environment variables.
"""

from __future__ import annotations

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig


class TestPairConfigDefaults:
    def test_entry_retry_count_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.entry_retry_count == 2

    def test_entry_retry_delay_s_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.entry_retry_delay_s == 2.0

    def test_rollback_retry_count_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.rollback_retry_count == 2

    def test_rollback_retry_delay_s_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.rollback_retry_delay_s == 2.0

    def test_sl_cancel_retry_count_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.sl_cancel_retry_count == 2

    def test_sl_cancel_retry_delay_s_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.sl_cancel_retry_delay_s == 1.0

    def test_orphan_auto_close_cycles_default(self) -> None:
        cfg = VolStraddleConfig()
        assert cfg.orphan_auto_close_cycles == 4


class TestPairConfigEnvVars:
    def test_entry_retry_count_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_ENTRY_RETRY_COUNT", "5")
        cfg = VolStraddleConfig()
        assert cfg.entry_retry_count == 5

    def test_entry_retry_delay_s_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_ENTRY_RETRY_DELAY_S", "3.5")
        cfg = VolStraddleConfig()
        assert cfg.entry_retry_delay_s == 3.5

    def test_rollback_retry_count_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_ROLLBACK_RETRY_COUNT", "3")
        cfg = VolStraddleConfig()
        assert cfg.rollback_retry_count == 3

    def test_rollback_retry_delay_s_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_ROLLBACK_RETRY_DELAY_S", "1.5")
        cfg = VolStraddleConfig()
        assert cfg.rollback_retry_delay_s == 1.5

    def test_sl_cancel_retry_count_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_SL_CANCEL_RETRY_COUNT", "4")
        cfg = VolStraddleConfig()
        assert cfg.sl_cancel_retry_count == 4

    def test_sl_cancel_retry_delay_s_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_SL_CANCEL_RETRY_DELAY_S", "0.5")
        cfg = VolStraddleConfig()
        assert cfg.sl_cancel_retry_delay_s == 0.5

    def test_orphan_auto_close_cycles_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VS_ORPHAN_AUTO_CLOSE_CYCLES", "8")
        cfg = VolStraddleConfig()
        assert cfg.orphan_auto_close_cycles == 8
