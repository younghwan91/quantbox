"""Tests for state schema versioning and migration (Phase 2.4)."""

from __future__ import annotations

import json
from pathlib import Path

from quantbox.core.scalper_state import (
    CURRENT_SCHEMA_VERSION,
    migrate_state,
    save_scalper_state,
    load_scalper_state,
)


class TestMigrateState:
    """migrate_state() upgrades old state to current version."""

    def test_v0_to_v1_adds_version(self):
        """Old state (no _schema_version) gets upgraded to v1."""
        state = {"positions": {}, "balance": 1000.0}
        migrated = migrate_state(state)
        assert migrated["_schema_version"] == 1
        assert migrated["positions"] == {}
        assert migrated["balance"] == 1000.0

    def test_current_version_is_noop(self):
        """State at current version passes through unchanged."""
        state = {"positions": {}, "_schema_version": CURRENT_SCHEMA_VERSION}
        migrated = migrate_state(state)
        assert migrated is state  # same object, no copy

    def test_future_version_passes_through(self):
        """State from a future version (e.g., v99) is not downgraded."""
        state = {"positions": {}, "_schema_version": 99}
        migrated = migrate_state(state)
        assert migrated["_schema_version"] == 99


class TestSaveLoadRoundTrip:
    """Schema version persists through save/load cycle."""

    def test_save_includes_schema_version(self, tmp_path: Path):
        filepath = tmp_path / "test_state.json"
        state = {"positions": {"BTCUSDT": {"entry_price": 50000}}}
        save_scalper_state(state, filepath=filepath)

        raw = json.loads(filepath.read_text())
        assert raw["_schema_version"] == CURRENT_SCHEMA_VERSION
        assert "_checksum" in raw

    def test_load_old_state_migrates(self, tmp_path: Path):
        """Loading a pre-versioning state file adds _schema_version."""
        filepath = tmp_path / "old_state.json"
        old_state = {"positions": {}, "balance": 500.0}
        filepath.write_text(json.dumps(old_state))

        loaded = load_scalper_state(filepath=filepath)
        assert loaded is not None
        assert loaded["_schema_version"] == CURRENT_SCHEMA_VERSION
        assert loaded["balance"] == 500.0

    def test_save_load_roundtrip_preserves_data(self, tmp_path: Path):
        filepath = tmp_path / "rt_state.json"
        state = {
            "positions": {"ETHUSDT:LONG": {"entry_price": 3000, "signal": 1}},
            "trade_history": [{"pnl": 0.05}],
        }
        save_scalper_state(state, filepath=filepath)
        loaded = load_scalper_state(filepath=filepath)

        assert loaded is not None
        assert loaded["positions"]["ETHUSDT:LONG"]["entry_price"] == 3000
        assert loaded["trade_history"][0]["pnl"] == 0.05
        assert loaded["_schema_version"] == CURRENT_SCHEMA_VERSION
