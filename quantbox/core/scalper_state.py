"""Scalper state file management: atomic save, load, rotating backups.

Extracted from scalper.py to reduce file size and isolate state I/O concerns.
Used by RubberBandScalper._save_state() and _load_state() class methods.

State files:
    data/state/scalper_state.json       — live trading
    data/state/scalper_state_paper.json — paper trading
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger('Scalper')

STATE_DIR = Path('data/state')
STATE_FILE = STATE_DIR / 'scalper_state.json'
PAPER_STATE_FILE = STATE_DIR / 'scalper_state_paper.json'


def rotate_state_backups(filepath: Path, keep: int = 5) -> None:
    """Delete oldest backup files, keeping only the N most recent."""
    pattern = filepath.stem + '_backup_'
    backups = sorted(filepath.parent.glob(f'{pattern}*.json'))
    for old in backups[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def save_scalper_state(state: dict, filepath: Path | None = None) -> None:
    """Atomic JSON write with rotating backup (keeps 5 most recent).

    Uses tmp → rename pattern to prevent corruption on crash mid-write.
    Backs up existing file before overwriting.
    """
    if filepath is None:
        filepath = STATE_FILE
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = filepath.parent / f'{filepath.stem}_backup_{ts}.json'
        try:
            shutil.copy2(filepath, backup)
            rotate_state_backups(filepath, keep=5)
        except OSError:
            pass

    fd, tmp = tempfile.mkstemp(dir=str(filepath.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, str(filepath))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_scalper_state(filepath: Path | None = None) -> dict | None:
    """Load state JSON. Returns None if file doesn't exist or is corrupt."""
    if filepath is None:
        filepath = STATE_FILE
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"State load failed: {e}")
        return None
