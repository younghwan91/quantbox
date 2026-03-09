"""
State Persistence — JSON 기반 슬롯 상태 저장/복구.

프로세스 크래시 시 거래소에 남은 고아 포지션을 복구하기 위해
슬롯 상태를 주기적으로 디스크에 저장한다.

파일 위치: data/state/live_state.json (기본값)

설계 원칙:
    - 상태 저장은 포지션 진입/청산 시 즉시 수행 (이벤트 기반).
    - 복구 시 거래소 실제 포지션과 대조(reconcile)하여 불일치를 감지.
    - 파일 쓰기는 atomic write (tmp → rename) 패턴으로 corruption 방지.
    - JSON을 사용하여 사람이 읽을 수 있고 디버깅이 용이함.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("LiveTrader")

# 기본 상태 파일 경로
DEFAULT_STATE_DIR = Path("data/state")
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "live_state.json"


def save_state(
    slots: list[dict],
    slot_positions: dict[int, dict],
    slot_entry_times: dict[int, datetime],
    trade_history: list[dict],
    paper_balance: float,
    paper_mode: bool,
    initial_balance: float = 0.0,
    session_high_water_mark: float = 0.0,
    filepath: Path = DEFAULT_STATE_FILE,
) -> None:
    """슬롯 상태를 JSON 파일로 저장 (atomic write).

    Args:
        slots: 슬롯 상태 리스트 [{'slot_id': int, 'position_open': bool, ...}].
        slot_positions: slot_id → position dict 매핑.
        slot_entry_times: slot_id → entry datetime 매핑.
        trade_history: 완료된 트레이드 이력.
        paper_balance: 페이퍼 트레이딩 잔고.
        paper_mode: 페이퍼 모드 여부.
        initial_balance: 세션 시작 잔고 (drawdown 기준).
        session_high_water_mark: 세션 최고 잔고 (DD 계산용).
        filepath: 저장 파일 경로.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "version": "1.8.0",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "paper_mode": paper_mode,
        "paper_balance": paper_balance,
        "initial_balance": initial_balance,
        "session_high_water_mark": session_high_water_mark,
        "slots": _serialize_slots(slots),
        "positions": _serialize_positions(slot_positions),
        "entry_times": _serialize_entry_times(slot_entry_times),
        "trade_history": _serialize_trade_history(trade_history),
    }

    # Atomic write: tmp 파일에 쓰고 rename (OS-level atomic on same filesystem)
    fd, tmp_path = tempfile.mkstemp(dir=str(filepath.parent), suffix=".tmp", prefix="state_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, str(filepath))
        logger.debug(f"State saved to {filepath}")
    except Exception:
        # 실패 시 tmp 파일 정리
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(filepath: Path = DEFAULT_STATE_FILE) -> dict[str, Any] | None:
    """저장된 상태 파일 로드.

    Args:
        filepath: 상태 파일 경로.

    Returns:
        상태 dict 또는 파일이 없으면 None.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.info(f"No state file found at {filepath}")
        return None

    try:
        with open(filepath, "r") as f:
            state = json.load(f)
        logger.info(f"State loaded from {filepath} (saved_at={state.get('saved_at', 'unknown')})")
        return state
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load state from {filepath}: {e}")
        return None


def clear_state(filepath: Path = DEFAULT_STATE_FILE) -> None:
    """상태 파일 삭제 (정상 종료 시).

    Args:
        filepath: 상태 파일 경로.
    """
    filepath = Path(filepath)
    if filepath.exists():
        filepath.unlink()
        logger.info(f"State file cleared: {filepath}")


def _serialize_slots(slots: list) -> list[dict]:
    """TraderSlotState 리스트를 JSON-safe dict 리스트로 변환."""
    result = []
    for s in slots:
        if hasattr(s, "slot_id"):
            # TraderSlotState dataclass
            result.append(
                {
                    "slot_id": s.slot_id,
                    "position_open": s.position_open,
                    "current_pair": list(s.current_pair) if s.current_pair else None,
                }
            )
        else:
            # 이미 dict
            result.append(s)
    return result


def _serialize_positions(positions: dict[int, dict]) -> dict[str, dict]:
    """slot_positions를 JSON-safe로 변환 (int key → str key)."""
    result = {}
    for slot_id, pos in positions.items():
        safe_pos = {}
        for k, v in pos.items():
            # DataFrame은 저장하지 않음
            if k.startswith("_"):
                continue
            safe_pos[k] = v
        result[str(slot_id)] = safe_pos
    return result


def _serialize_entry_times(entry_times: dict[int, datetime]) -> dict[str, str]:
    """slot_entry_times를 ISO format 문자열로 변환."""
    return {
        str(slot_id): dt.isoformat() if isinstance(dt, datetime) else str(dt) for slot_id, dt in entry_times.items()
    }


def _serialize_trade_history(trades: list[dict]) -> list[dict]:
    """trade_history의 datetime 필드를 문자열로 변환."""
    result = []
    for t in trades:
        safe_t = {}
        for k, v in t.items():
            if isinstance(v, datetime):
                safe_t[k] = v.isoformat()
            else:
                safe_t[k] = v
        result.append(safe_t)
    return result


def deserialize_entry_times(raw: dict[str, str]) -> dict[int, datetime]:
    """ISO format 문자열을 datetime으로 복원."""
    result = {}
    for slot_id_str, dt_str in raw.items():
        try:
            dt = datetime.fromisoformat(dt_str)
            # timezone-aware 보장
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            result[int(slot_id_str)] = dt
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse entry time for slot {slot_id_str}: {e}")
    return result


def deserialize_positions(raw: dict[str, dict]) -> dict[int, dict]:
    """str key를 int로 복원."""
    return {int(k): v for k, v in raw.items()}
