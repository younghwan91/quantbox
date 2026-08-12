"""기록 파일 → 경과시간별 스프레드 표 → 사전 등록된 판정.

임계값은 데이터가 오기 전에 정해졌고 결과를 보고 조정하지 않는다. 근거는
어제 실측한 손익분기 왕복 100bp이며, 편도 46bp + 수수료 8.6bp가 그 구성이다.
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
import os
from pathlib import Path

import numpy as np

from quantbox.listing.book import effective_half_spread_bp, parse_levels

logger = logging.getLogger("listing.analyze")

#: 진입 창 — 상장 후 이 구간의 스프레드로 판정한다.
ENTRY_WINDOW_H: tuple[float, float] = (1.0, 24.0)

#: 판정 기준 (사전 등록, 스펙과 일치해야 한다)
TARGET_NOTIONAL_USD = 24.0
PASS_BP = 25.0
EXTEND_BP = 50.0
MAX_UNFILLABLE_FRAC = 0.10
MIN_SAMPLES = 100

Sample = tuple[float, float, bool]
"""``(상장 후 경과 시간(h), 편도 실효 스프레드(bp), 체결 여부)``."""


def _samples_from_file(path: str, onboard_ms: int, extract) -> list[Sample]:
    out: list[Sample] = []
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bids, asks = extract(rec["d"])
                bp, filled = effective_half_spread_bp(bids, asks, TARGET_NOTIONAL_USD)
                if bp != bp:  # nan — 교차·빈 호가창
                    continue
                hours = (rec["recv_ns"] / 1e6 - onboard_ms) / 3_600_000.0
                out.append((hours, bp, filled))
    except (OSError, EOFError, gzip.BadGzipFile) as e:
        # 수집기가 지금 쓰고 있는 파일은 gzip 스트림이 미완결이다 — 흔한 일이므로 건너뛴다.
        logger.debug(f"{path} 읽기 건너뜀: {e}")
    return out


def read_depth_samples(symbol: str, socket_root: Path, onboard_ms: int, max_files: int = 0) -> list[Sample]:
    """`depth20` 기록에서 표본을 뽑는다."""
    files = sorted(glob.glob(os.path.join(str(socket_root), "depth20", symbol, "*.jsonl.gz")))
    if max_files:
        files = files[:max_files]

    def extract(d: dict):
        return parse_levels(d["b"]), parse_levels(d["a"])

    return [s for f in files for s in _samples_from_file(f, onboard_ms, extract)]


def read_bookticker_samples(symbol: str, socket_root: Path, onboard_ms: int, max_files: int = 0) -> list[Sample]:
    """`bookTicker`를 1레벨 호가창으로 읽는다.

    기존 58종목에는 depth20이 없으므로 파이프라인 검증·대조용으로만 쓴다.
    최우선 호가 물량을 넘는 주문은 여기서 체결 불가로 나온다.
    """
    files = sorted(glob.glob(os.path.join(str(socket_root), "bookTicker", symbol, "*.jsonl.gz")))
    files = files[:-1] if len(files) > 1 else files  # 마지막 파일은 쓰는 중일 수 있다
    if max_files:
        files = files[:max_files]

    def extract(d: dict):
        return [(float(d["b"]), float(d["B"]))], [(float(d["a"]), float(d["A"]))]

    return [s for f in files for s in _samples_from_file(f, onboard_ms, extract)]


def _in_window(samples: list[Sample]) -> list[Sample]:
    lo, hi = ENTRY_WINDOW_H
    return [s for s in samples if lo <= s[0] <= hi]


def hourly_profile(samples: list[Sample]) -> list[dict]:
    """시간대별 스프레드 분포. 진입 창 밖도 포함한다 — 프로파일 전체를 보기 위해서다."""
    buckets: dict[int, list[Sample]] = {}
    for s in samples:
        buckets.setdefault(int(s[0]), []).append(s)

    rows = []
    for hour in sorted(buckets):
        group = buckets[hour]
        bps = np.array([b for _, b, _ in group], dtype=float)
        rows.append(
            {
                "hour": hour,
                "n": len(group),
                "median_bp": float(np.median(bps)),
                "p25_bp": float(np.percentile(bps, 25)),
                "p75_bp": float(np.percentile(bps, 75)),
                "unfillable_frac": float(np.mean([not f for _, _, f in group])),
            }
        )
    return rows


def verdict(samples: list[Sample]) -> dict:
    """진입 창 표본으로 사전 등록된 판정을 낸다."""
    window = _in_window(samples)
    n = len(window)
    if n < MIN_SAMPLES:
        return {"n": n, "median_bp": float("nan"), "unfillable_frac": float("nan"), "decision": "표본부족"}

    median_bp = float(np.median([b for _, b, _ in window]))
    unfillable = float(np.mean([not f for _, _, f in window]))

    if unfillable > MAX_UNFILLABLE_FRAC:
        decision = "종료"  # 스프레드와 무관하다 — 체결이 안 되면 전략이 성립하지 않는다
    elif median_bp <= PASS_BP:
        decision = "진행"
    elif median_bp <= EXTEND_BP:
        decision = "재검토"
    else:
        decision = "종료"

    return {"n": n, "median_bp": median_bp, "unfillable_frac": unfillable, "decision": decision}
