"""Tests for SHA-256 checksum in state file save/load (T3-3)."""

import json
import logging

import pytest

from quantbox.core.scalper_state import (
    _compute_checksum,
    load_scalper_state,
    save_scalper_state,
)

SAMPLE_STATE = {
    "positions": {
        "BTCUSDT:LONG": {
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "entry_price": 95000.0,
            "notional": 3000.0,
        }
    },
    "high_water_mark": 1050.0,
    "saved_at": "2026-01-01T00:00:00Z",
}


def test_save_adds_checksum(tmp_path):
    """save_scalper_state must write a _checksum field to the JSON file."""
    fp = tmp_path / "state.json"
    save_scalper_state(SAMPLE_STATE, filepath=fp)

    raw = json.loads(fp.read_text())
    assert "_checksum" in raw
    assert isinstance(raw["_checksum"], str)
    assert len(raw["_checksum"]) == 64  # SHA-256 hex digest


def test_save_does_not_mutate_caller_state(tmp_path):
    """Caller's dict must not be modified by save_scalper_state."""
    state = {"positions": {}, "balance": 1000.0}
    fp = tmp_path / "state.json"
    save_scalper_state(state, filepath=fp)
    assert "_checksum" not in state


def test_valid_checksum_loads_normally(tmp_path):
    """Round-trip: saved state is loaded without error, positions intact."""
    fp = tmp_path / "state.json"
    save_scalper_state(SAMPLE_STATE, filepath=fp)

    loaded = load_scalper_state(filepath=fp)
    assert loaded is not None
    assert "_checksum" not in loaded  # stripped by loader
    assert loaded["positions"] == SAMPLE_STATE["positions"]
    assert loaded["high_water_mark"] == pytest.approx(1050.0)


def test_corrupted_state_detected_falls_back_to_backup(tmp_path, caplog):
    """Checksum mismatch triggers CRITICAL log and backup restore."""
    fp = tmp_path / "state.json"

    # Save a clean backup-like state (simulate an older good save)
    good_state = {**SAMPLE_STATE, "high_water_mark": 900.0}
    backup_ts = "20260101_120000"
    backup_fp = tmp_path / f"{fp.stem}_backup_{backup_ts}.json"
    save_scalper_state(good_state, filepath=backup_fp)

    # Now write a corrupted primary file (tamper with a price AFTER checksum)
    save_scalper_state(SAMPLE_STATE, filepath=fp)
    raw = json.loads(fp.read_text())
    raw["positions"]["BTCUSDT:LONG"]["entry_price"] = 99999.0  # tamper
    fp.write_text(json.dumps(raw))

    with caplog.at_level(logging.CRITICAL, logger="Scalper"):
        loaded = load_scalper_state(filepath=fp)

    # Should have detected mismatch and logged CRITICAL
    assert any("CHECKSUM MISMATCH" in r.message for r in caplog.records)
    # Should fall back to backup
    assert loaded is not None
    assert loaded["high_water_mark"] == pytest.approx(900.0)


def test_corrupted_primary_and_all_backups_returns_primary(tmp_path, caplog):
    """If primary AND all backups are corrupt, return primary with CRITICAL."""
    fp = tmp_path / "state.json"
    save_scalper_state(SAMPLE_STATE, filepath=fp)

    # Tamper with primary checksum
    raw = json.loads(fp.read_text())
    raw["high_water_mark"] = 99999.0
    fp.write_text(json.dumps(raw))

    with caplog.at_level(logging.CRITICAL, logger="Scalper"):
        loaded = load_scalper_state(filepath=fp)

    assert loaded is not None  # must not return None / crash
    assert any("CHECKSUM MISMATCH" in r.message for r in caplog.records)


def test_old_state_without_checksum_loads(tmp_path, caplog):
    """State files without _checksum (pre-T3-3) must load normally."""
    fp = tmp_path / "state.json"
    old_state = {**SAMPLE_STATE}  # no _checksum key
    fp.write_text(json.dumps(old_state))

    with caplog.at_level(logging.DEBUG, logger="Scalper"):
        loaded = load_scalper_state(filepath=fp)

    assert loaded is not None
    assert loaded["positions"] == SAMPLE_STATE["positions"]
    # Debug message should mention absence of checksum
    assert any("no checksum" in r.message.lower() for r in caplog.records)


def test_nonexistent_file_returns_none(tmp_path):
    """load_scalper_state returns None when file does not exist."""
    result = load_scalper_state(filepath=tmp_path / "missing.json")
    assert result is None


def test_compute_checksum_ignores_existing_checksum_field():
    """_compute_checksum must be idempotent w.r.t. _checksum field."""
    state = {"a": 1, "b": 2}
    cs1 = _compute_checksum(state)
    cs2 = _compute_checksum({**state, "_checksum": "whatever"})
    assert cs1 == cs2


def test_checksum_detects_value_change():
    """Changing any value must produce a different checksum."""
    state = {"entry_price": 100.0, "notional": 3000.0}
    cs_orig = _compute_checksum(state)
    cs_tampered = _compute_checksum({**state, "entry_price": 100.1})
    assert cs_orig != cs_tampered
