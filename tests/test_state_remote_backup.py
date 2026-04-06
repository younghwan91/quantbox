"""Tests for the remote/secondary backup path feature in scalper_state.py.

The remote backup is best-effort: it must never raise, never slow down the
primary save, and must create/copy the file when the env-var is configured.
"""

import json

import pytest

from quantbox.core.scalper_state import (
    _try_remote_backup,
    save_scalper_state,
)

# ---------------------------------------------------------------------------
# Minimal valid state fixture
# ---------------------------------------------------------------------------

SAMPLE_STATE: dict = {
    "positions": {},
    "trade_records": [],
    "paper_balance": 1000.0,
    "high_water_mark": 1000.0,
    "saved_at": "2026-01-01T00:00:00",
}


# ---------------------------------------------------------------------------
# _try_remote_backup unit tests (env-var controlled)
# ---------------------------------------------------------------------------


def test_remote_backup_created_when_env_set(tmp_path, monkeypatch):
    """When STATE_REMOTE_BACKUP_DIR is set, file is copied to remote dir."""
    src = tmp_path / "source" / "scalper_state.json"
    src.parent.mkdir()
    src.write_text('{"test": 1}')

    remote_dir = tmp_path / "remote"
    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    _try_remote_backup(src)

    dest = remote_dir / "scalper_state.json"
    assert dest.exists(), "Remote backup file should exist"
    assert dest.read_text() == '{"test": 1}', "Content should match source"


def test_remote_backup_skipped_when_env_unset(tmp_path, monkeypatch):
    """When STATE_REMOTE_BACKUP_DIR is not set, no remote dir is created."""
    src = tmp_path / "scalper_state.json"
    src.write_text('{"test": 1}')

    monkeypatch.delenv("STATE_REMOTE_BACKUP_DIR", raising=False)

    remote_dir = tmp_path / "remote_should_not_exist"
    _try_remote_backup(src)

    assert not remote_dir.exists(), "Remote dir should not be created when env is unset"


def test_remote_backup_skipped_when_env_empty(tmp_path, monkeypatch):
    """When STATE_REMOTE_BACKUP_DIR is empty string, backup is skipped."""
    src = tmp_path / "scalper_state.json"
    src.write_text('{"test": 1}')

    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", "")

    # Should silently do nothing — no exception
    _try_remote_backup(src)


def test_remote_backup_does_not_raise_on_failure(tmp_path, monkeypatch):
    """Permission error (or any failure) must NOT propagate — best-effort only."""
    src = tmp_path / "scalper_state.json"
    src.write_text('{"test": 1}')

    remote_dir = tmp_path / "remote"
    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    # Simulate copy failure (e.g. permission denied on destination)
    with pytest.raises(Exception):
        # Sanity check: confirm the mock actually raises
        raise PermissionError("simulated")

    import unittest.mock as mock

    with mock.patch("shutil.copy2", side_effect=PermissionError("permission denied")):
        # Must not raise — failure must be swallowed
        _try_remote_backup(src)


def test_remote_backup_same_filename_as_source(tmp_path, monkeypatch):
    """Destination filename must equal source filename (not timestamped)."""
    src = tmp_path / "source" / "scalper_state_paper.json"
    src.parent.mkdir()
    src.write_text('{"paper": true}')

    remote_dir = tmp_path / "remote"
    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    _try_remote_backup(src)

    assert (remote_dir / "scalper_state_paper.json").exists()
    assert not any(f.name != "scalper_state_paper.json" for f in remote_dir.iterdir()), (
        "Only the exact filename should be in the remote dir"
    )


def test_remote_backup_creates_missing_dir(tmp_path, monkeypatch):
    """Remote dir (including nested parents) is created automatically."""
    src = tmp_path / "scalper_state.json"
    src.write_text('{"test": 1}')

    nested_remote = tmp_path / "a" / "b" / "c" / "remote"
    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(nested_remote))

    _try_remote_backup(src)

    assert nested_remote.is_dir()
    assert (nested_remote / "scalper_state.json").exists()


def test_remote_backup_overwrites_previous(tmp_path, monkeypatch):
    """Each backup overwrites (not appends) the previous remote copy."""
    src = tmp_path / "source" / "scalper_state.json"
    src.parent.mkdir()

    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    existing = remote_dir / "scalper_state.json"
    existing.write_text("old content")

    src.write_text("new content")
    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    _try_remote_backup(src)

    assert existing.read_text() == "new content"


# ---------------------------------------------------------------------------
# Integration: save_scalper_state end-to-end with remote backup
# ---------------------------------------------------------------------------


def test_save_writes_remote_backup(tmp_path, monkeypatch):
    """save_scalper_state() triggers remote backup when env-var is set."""
    primary = tmp_path / "state" / "scalper_state.json"
    remote_dir = tmp_path / "remote"

    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    save_scalper_state(SAMPLE_STATE, filepath=primary)

    assert primary.exists(), "Primary state file must be written"
    dest = remote_dir / "scalper_state.json"
    assert dest.exists(), "Remote backup must be written by save_scalper_state"

    remote_data = json.loads(dest.read_text())
    assert remote_data.get("paper_balance") == 1000.0


def test_save_no_remote_backup_when_env_unset(tmp_path, monkeypatch):
    """save_scalper_state() leaves the remote dir untouched when env-var is absent."""
    primary = tmp_path / "state" / "scalper_state.json"
    remote_dir = tmp_path / "remote"

    monkeypatch.delenv("STATE_REMOTE_BACKUP_DIR", raising=False)

    save_scalper_state(SAMPLE_STATE, filepath=primary)

    assert primary.exists()
    assert not remote_dir.exists(), "Remote dir must not be created when env is unset"


def test_save_succeeds_even_if_remote_fails(tmp_path, monkeypatch):
    """Primary save must succeed even when the remote backup fails."""
    import unittest.mock as mock

    primary = tmp_path / "state" / "scalper_state.json"
    remote_dir = tmp_path / "remote"
    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    with mock.patch("shutil.copy2", side_effect=OSError("disk full")):
        # Must not raise
        save_scalper_state(SAMPLE_STATE, filepath=primary)

    assert primary.exists(), "Primary state file must exist despite remote failure"
    data = json.loads(primary.read_text())
    assert "_checksum" in data, "Checksum must be present in primary state"


def test_save_remote_backup_paper_state(tmp_path, monkeypatch):
    """Remote backup works for the paper state file too."""
    paper = tmp_path / "state" / "scalper_state_paper.json"
    remote_dir = tmp_path / "remote"

    monkeypatch.setenv("STATE_REMOTE_BACKUP_DIR", str(remote_dir))

    save_scalper_state({**SAMPLE_STATE, "paper_balance": 500.0}, filepath=paper)

    assert (remote_dir / "scalper_state_paper.json").exists()
    remote_data = json.loads((remote_dir / "scalper_state_paper.json").read_text())
    assert remote_data.get("paper_balance") == 500.0
