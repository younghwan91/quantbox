"""Bot-level utilities: git commit tracking and logging setup.

These are module-level helpers used by scalper.py at startup.
Extracted from scalper.py to reduce file size and isolate concerns.
"""

import logging
import logging.handlers
import subprocess
import sys
import time
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


def cleanup_old_logs(log_dir: Path, max_age_days: int = 30) -> int:
    """Delete log files older than max_age_days. Returns count of deleted files.

    Safe: only deletes .log files. Never raises — failure is logged as warning.

    Args:
        log_dir: Directory to scan for old log files.
        max_age_days: Files older than this are deleted. 0 = delete files older than 1 hour.
    """
    if not log_dir.exists():
        return 0

    deleted = 0
    now = time.time()
    cutoff = now - 3600 if max_age_days == 0 else now - max_age_days * 86400

    try:
        for f in log_dir.iterdir():
            if not f.is_file() or f.suffix != ".log":
                continue
            try:
                mtime = f.stat().st_mtime
                # Skip files modified within the last 10 seconds (current session)
                if mtime > now - 10:
                    continue
                if mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except Exception as e:
                logging.getLogger("Scalper").warning(f"Could not delete log file {f.name}: {e}")
    except Exception as e:
        logging.getLogger("Scalper").warning(f"Log cleanup scan failed for {log_dir}: {e}")

    return deleted


def cleanup_all_log_dirs(base: Path = Path("logs"), max_age_days: int = 30) -> int:
    """Walk all subdirs under base and delete .log files older than max_age_days.

    Args:
        base: Root directory to scan (default: logs/).
        max_age_days: Passed to cleanup_old_logs for each subdir.

    Returns:
        Total count of deleted files across all subdirs.
    """
    if not base.exists():
        return 0
    total = 0
    dirs = [base] + [p for p in base.rglob("*") if p.is_dir()]
    for d in dirs:
        total += cleanup_old_logs(d, max_age_days=max_age_days)
    return total


def setup_logging(
    strategy_name: str = "vol_straddle",
    paper_mode: bool = False,
    git_commit: dict[str, str] | None = None,
    max_age_days: int = 30,
) -> Path:
    """Configure rotating file + stdout logging. Returns log file path.

    Args:
        strategy_name: Used to choose log subdirectory.
        paper_mode: If True, log file is named 'paper', else 'live'.
        git_commit: Optional dict from get_git_commit(). Written as header.
        max_age_days: Delete log files older than this many days. 0 = 1-hour cutoff.
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

    deleted = cleanup_old_logs(log_dir, max_age_days=max_age_days)
    if deleted > 0:
        logging.getLogger("Scalper").info(f"Cleaned up {deleted} old log files")

    return log_file


# Module-level git commit info — computed once at import time
GIT_COMMIT: dict[str, str] = get_git_commit()
