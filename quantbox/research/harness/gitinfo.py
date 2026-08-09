"""커밋 해시 조회 — 리포트에 코드 버전을 각인하기 위한 유일 구현.

research/ 전역에 `get_commit` 33벌, `get_commit_short` 31벌이 복제돼 있었다.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

UNKNOWN = "unknown"


def repo_root() -> Path:
    """레포 루트. 이 파일 위치 기준이라 cwd에 무관하다."""
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=4)
def get_commit(short: bool = True) -> str:
    """현재 HEAD 커밋 해시. git이 없거나 레포가 아니면 "unknown".

    Args:
        short: True면 `--short` 해시.

    Returns:
        커밋 해시 문자열 또는 "unknown".
    """
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.check_output(cmd, cwd=str(repo_root()), text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, OSError):
        return UNKNOWN


def get_commit_short() -> str:
    """짧은 커밋 해시 (기존 research 스크립트의 이름 호환용)."""
    return get_commit(short=True)
