"""실험 harness — 실험이 **다시 짜면 안 되는 것들**의 유일 출처.

새 실험은 이 패키지만 import하면 된다. sim/로딩/지표/리포트를 직접 구현하지 않는다:

    from quantbox.research.harness import MarketData, experiment, report

    @experiment(name="r121_my_idea")
    def run(cfg):
        data = MarketData.load(cfg.symbol_whitelist)
        trades = run_ma_cluster(data.data, cfg=cfg)
        return report.standard(trades, data, cfg)

    if __name__ == "__main__":
        print(report.render(run().value))

여기서 재수출하는 sim/지표 함수는 **라이브가 쓰는 바로 그 함수**다. 그래서 백테스트와
실거래의 parity가 구조적으로 보장된다(재구현이 없으면 어긋날 수가 없다).
"""

from __future__ import annotations

from quantbox.backtest.sim_core import (
    directional_conditions,
    donchian_conditions,
    ma_cluster_conditions,
    run_ma_cluster,
    sim_drive,
)

# 프로덕션 유일 구현 — 재수출만 한다. 실험은 이걸 쓰고, 절대 다시 짜지 않는다.
from quantbox.backtest.trade_sim import (
    bootstrap_pf,
    build_resampled,
    collect_donchian_trades,
    collect_ma_cluster_trades,
    collect_trades,
    compute_metrics,
    portfolio_sim,
)
from quantbox.research.harness import report
from quantbox.research.harness.data import MarketData
from quantbox.research.harness.experiment import ExperimentResult, experiment, run_parallel
from quantbox.research.harness.gitinfo import get_commit, get_commit_short, repo_root
from quantbox.strategy.vol_straddle.signals import (
    check_donchian_breakout,
    check_ma_cluster,
    compute_momentum,
    compute_sma,
    compute_vr,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)

__all__ = [
    # harness
    "MarketData",
    "experiment",
    "ExperimentResult",
    "run_parallel",
    "report",
    "get_commit",
    "get_commit_short",
    "repo_root",
    # 제네릭 sim 코어
    "sim_drive",
    "run_ma_cluster",
    "ma_cluster_conditions",
    "directional_conditions",
    "donchian_conditions",
    # 골든 레퍼런스 sim
    "collect_trades",
    "collect_ma_cluster_trades",
    "collect_donchian_trades",
    # 지표
    "portfolio_sim",
    "compute_metrics",
    "bootstrap_pf",
    # 지표/데이터 헬퍼
    "build_resampled",
    "compute_sma",
    "compute_momentum",
    "resample_ohlcv",
    # 라이브 신호 함수 (진입 규칙 단일소스 — research는 재구현 말고 이걸 import)
    "check_ma_cluster",
    "check_donchian_breakout",
    "gen_vr_signal",
    "make_direction_momentum",
    "compute_vr",
]
