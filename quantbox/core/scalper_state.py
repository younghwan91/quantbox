"""Scalper state file management: atomic save, load, rotating backups.

Extracted from scalper.py to reduce file size and isolate state I/O concerns.
Used by RubberBandScalper._save_state() and _load_state() class methods.

State files:
    data/state/scalper_state.json       — live trading
    data/state/scalper_state_paper.json — paper trading
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("Scalper")

STATE_DIR = Path("data/state")
STATE_FILE = STATE_DIR / "scalper_state.json"
PAPER_STATE_FILE = STATE_DIR / "scalper_state_paper.json"


def rotate_state_backups(filepath: Path, keep: int = 5) -> None:
    """Delete oldest backup files, keeping only the N most recent."""
    pattern = filepath.stem + "_backup_"
    backups = sorted(filepath.parent.glob(f"{pattern}*.json"))
    for old in backups[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def _compute_checksum(state: dict) -> str:
    """SHA-256 of canonical JSON (sorted keys, _checksum field excluded)."""
    state_copy = {k: v for k, v in state.items() if k != "_checksum"}
    canonical = json.dumps(state_copy, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_scalper_state(state: dict, filepath: Path | None = None) -> None:
    """Atomic JSON write with rotating backup (keeps 5 most recent).

    Adds a SHA-256 ``_checksum`` field covering all other state keys so
    that ``load_scalper_state`` can detect on-disk corruption.

    Uses tmp → rename pattern to prevent corruption on crash mid-write.
    Backs up existing file before overwriting.
    """
    if filepath is None:
        filepath = STATE_FILE
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = filepath.parent / f"{filepath.stem}_backup_{ts}.json"
        try:
            shutil.copy2(filepath, backup)
            rotate_state_backups(filepath, keep=5)
        except OSError:
            pass

    # Build the dict to write: original state + fresh checksum.
    # Checksum is computed BEFORE adding the field so it covers all real data.
    state_to_write = {k: v for k, v in state.items() if k != "_checksum"}
    state_to_write["_checksum"] = _compute_checksum(state_to_write)

    fd, tmp = tempfile.mkstemp(dir=str(filepath.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state_to_write, f, indent=2, default=str)
        os.replace(tmp, str(filepath))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_state_from_file(filepath: Path) -> dict | None:
    """Read and JSON-parse a state file; return None on IO/parse error."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"State load failed ({filepath.name}): {e}")
        return None


def _verify_and_strip_checksum(state: dict, source: str) -> dict | None:
    """Verify ``_checksum``; return state without the field on success.

    Returns None if the checksum is present but does not match (corruption).
    If the field is absent (old state file) the state is returned as-is after
    logging a debug message.
    """
    if "_checksum" not in state:
        logger.debug(f"State loaded from {source} has no checksum (pre-T3-3 file).")
        return state

    stored = state["_checksum"]
    expected = _compute_checksum(state)
    if stored != expected:
        logger.critical(
            f"State file CHECKSUM MISMATCH ({source})! "
            f"Expected={expected[:16]}… Stored={stored[:16]}… "
            "File may be corrupted — attempting backup restore."
        )
        return None

    clean = {k: v for k, v in state.items() if k != "_checksum"}
    return clean


def load_scalper_state(filepath: Path | None = None) -> dict | None:
    """Load state JSON with SHA-256 checksum verification.

    On checksum mismatch the function attempts to restore from the most
    recent backup file.  If the backup also fails verification, the original
    (potentially corrupted) state is returned with a CRITICAL log warning so
    the caller can decide how to proceed rather than crashing.

    Returns None if the primary file does not exist.
    """
    if filepath is None:
        filepath = STATE_FILE
    if not filepath.exists():
        return None

    raw = _load_state_from_file(filepath)
    if raw is not None:
        verified = _verify_and_strip_checksum(raw, filepath.name)
        if verified is not None:
            return verified

    # Primary failed — try the most recent backup.
    pattern = filepath.stem + "_backup_"
    backups = sorted(filepath.parent.glob(f"{pattern}*.json"))
    for backup_path in reversed(backups):  # newest first
        logger.warning(f"Attempting state restore from backup: {backup_path.name}")
        backup_raw = _load_state_from_file(backup_path)
        if backup_raw is None:
            continue
        backup_verified = _verify_and_strip_checksum(backup_raw, backup_path.name)
        if backup_verified is not None:
            logger.warning(f"State restored successfully from backup: {backup_path.name}")
            return backup_verified

    # All backups exhausted — return raw primary state with CRITICAL warning.
    if raw is not None:
        logger.critical(
            "All state backups failed checksum or were unreadable. "
            "Returning potentially corrupted primary state. "
            "Manual inspection of positions is STRONGLY recommended."
        )
        return {k: v for k, v in raw.items() if k != "_checksum"}

    logger.error("State file unreadable and no valid backups found.")
    return None
