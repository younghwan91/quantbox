"""병렬 실험 인프라 검증 (Phase 5) — 실행만 병렬, 코어는 read-only 공유.

run_parallel은 여러 @experiment를 동시 실행한다. 공용코어(sim_core)를 수정하지 않고
공유하므로 병렬 결과 = 순차 결과(결정론). '실험은 병렬, 코어는 순차' 원칙의 구현.
"""

from __future__ import annotations

from quantbox.research.harness import ExperimentResult, experiment, run_parallel


def test_parallel_runs_all_experiments():
    @experiment(name="e1", quiet=True)
    def e1(cfg):
        return 1

    @experiment(name="e2", quiet=True)
    def e2(cfg):
        return 2

    @experiment(name="e3", quiet=True)
    def e3(cfg):
        return 3

    r = run_parallel([e1, e2, e3])
    assert set(r) == {"e1", "e2", "e3"}
    assert r["e1"].value == 1 and r["e2"].value == 2 and r["e3"].value == 3


def test_parallel_matches_sequential():
    """병렬 결과 == 순차 결과 (결정론)."""

    @experiment(name="ea", quiet=True)
    def ea(cfg):
        return sum(range(1000))

    seq = ea().value
    par = run_parallel([ea])["ea"].value
    assert par == seq


def test_parallel_returns_experiment_results():
    @experiment(name="e", quiet=True)
    def e(cfg):
        return 42

    r = run_parallel([e])
    assert isinstance(r["e"], ExperimentResult)
    assert r["e"].value == 42
    assert r["e"].name == "e"


def test_parallel_experiments_receive_config():
    """각 실험이 독립적으로 config를 주입받는다(파라미터 단일소스)."""
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    seen = {}

    @experiment(name="cfg_check", quiet=True)
    def cfg_check(cfg):
        seen["type"] = type(cfg).__name__
        return cfg.max_concurrent

    r = run_parallel([cfg_check])
    assert seen["type"] == "VolStraddleConfig"
    assert r["cfg_check"].value == VolStraddleConfig().max_concurrent
