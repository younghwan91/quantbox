"""Bot-level utilities: git commit tracking and logging setup.

These are module-level helpers used by scalper.py at startup.
Extracted from scalper.py to reduce file size and isolate concerns.
"""

import logging
import logging.handlers
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_commit() -> dict[str, str]:
    """Get current git commit info for log traceability.

    Returns:
        Dict with 'short', 'full', 'message', 'date' keys.
        All values are empty strings if git is unavailable.
    """
    info = {"short": "", "full": "", "message": "", "date": ""}
    try:
        info["short"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        info["full"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        info["message"] = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        info["date"] = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        pass
    return info


def setup_logging(
    strategy_name: str = "vol_straddle",
    paper_mode: bool = False,
    git_commit: dict[str, str] | None = None,
) -> Path:
    """Configure rotating file + stdout logging. Returns log file path.

    Args:
        strategy_name: Used to choose log subdirectory.
        paper_mode: If True, log file is named 'paper', else 'live'.
        git_commit: Optional dict from get_git_commit(). Written as header.
    """
    log_dir = Path("logs/bot") / strategy_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "paper" if paper_mode else "live"
    log_file = log_dir / f"scalper_{mode}_{ts}.log"

    if git_commit and git_commit.get("short"):
        with open(log_file, "w") as f:
            f.write(f"# Commit: {git_commit['short']} ({git_commit['date']})\n")
            f.write(f"# Message: {git_commit['message']}\n")
            f.write(f"# Full SHA: {git_commit['full']}\n")
            f.write("#\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                str(log_file),
                mode="a",
                maxBytes=50 * 1024 * 1024,
                backupCount=5,
            ),
        ],
    )
    for name in ("urllib3", "requests", "binance"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


# Module-level git commit info — computed once at import time
GIT_COMMIT: dict[str, str] = get_git_commit()
