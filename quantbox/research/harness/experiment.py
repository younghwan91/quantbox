"""실험 진입점 데코레이터 — 부트스트랩 보일러플레이트의 유일 구현.

실험 스크립트마다 반복되던 것: 커밋 각인, config 생성, 타이머, 헤더 출력.
`@experiment`가 이걸 흡수하므로 실험 본문은 '아이디어'만 남는다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from quantbox.research.harness.gitinfo import get_commit
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentResult:
    """실험 1회 실행의 결과 + 재현에 필요한 메타."""

    name: str
    commit: str
    elapsed_s: float
    value: Any

    def __str__(self) -> str:
        return f"[{self.name}] commit={self.commit} elapsed={self.elapsed_s:.1f}s"


def experiment(
    name: str,
    *,
    quiet: bool = False,
) -> Callable[[Callable[[VolStraddleConfig], T]], Callable[[], ExperimentResult]]:
    """실험 함수를 감싸 커밋 각인 + 타이밍을 붙인다.

    감싸인 함수는 `cfg: VolStraddleConfig` 하나만 받는다 — 파라미터의 유일 출처가
    config임을 구조로 강제하기 위해서다(하드코딩된 상수를 받을 자리가 없다).

    Args:
        name: 실험 식별자 (예: "r121_my_idea"). 리포트 파일명·헤더에 쓰인다.
        quiet: True면 배너를 출력하지 않는다 (테스트용).

    Returns:
        인자 없이 호출하면 ExperimentResult를 돌려주는 함수.

    Example:
        >>> @experiment(name="r121_my_idea")
        ... def run(cfg):
        ...     data = MarketData.load(cfg.symbol_whitelist)
        ...     trades = run_ma_cluster(data.data, cfg=cfg)
        ...     return report.standard(trades, data, cfg)
    """

    def decorator(fn: Callable[[VolStraddleConfig], T]) -> Callable[[], ExperimentResult]:
        @wraps(fn)
        def wrapper() -> ExperimentResult:
            cfg = VolStraddleConfig()
            commit = get_commit()
            if not quiet:
                print(f"═══ {name} @ {commit} ═══")
            t0 = time.perf_counter()
            value = fn(cfg)
            elapsed = time.perf_counter() - t0
            result = ExperimentResult(name=name, commit=commit, elapsed_s=elapsed, value=value)
            if not quiet:
                print(f"═══ done in {elapsed:.1f}s ═══")
            return result

        return wrapper

    return decorator


def run_parallel(
    experiment_fns: list,
    *,
    max_workers: int = 4,
) -> dict[str, ExperimentResult]:
    """여러 @experiment 함수를 병렬 실행한다 (Phase 5 — 멀티에이전트 실행 인프라).

    **실행만 병렬** — 공용코어(sim_core)는 read-only로 공유되므로 안전하다(코어를
    동시 수정하지 않는다, deep-interview R4 결정). 각 실험은 독립적으로 도는 순수
    함수라 병렬 결과 = 순차 결과(결정론). 이게 '실험은 병렬, 코어는 순차' 원칙의 구현이다.

    Args:
        experiment_fns: @experiment로 감싼 무인자 함수들.
        max_workers: 동시 실행 수.

    Returns:
        {experiment_name: ExperimentResult}.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: dict[str, ExperimentResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn) for fn in experiment_fns]
        for f in futures:
            r = f.result()
            results[r.name] = r
    return results
