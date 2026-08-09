"""G9 파리티: 라이브 포트폴리오 필터를 백테스트에 이식.

라이브는 진입 직전 `RiskGuard`로 세 가지를 더 본다 (`execution/risk_guard.py`):
  - 방향 분산: 같은 방향 포지션이 `max_same_direction`(=5) 이상이면 차단
  - BTC 트렌드 게이트(G6): BTC가 MA50 아래면 LONG 차단
  - 일손실 차단기(G7): 당일 실현손실이 잔고의 `daily_loss_limit_pct`를 넘으면 신규 진입 차단

`portfolio_sim`은 슬롯 상한과 동일 심볼 중복만 본다. 나머지 셋은 **없다** — 백테스트가
라이브라면 거절했을 트레이드를 계속 세고 있었다는 뜻이다.

필터는 전부 **명시적 opt-in**이다. 기본값을 켜면 기존 리서치 baseline이 전부 무효화되고,
`portfolio_sim`은 골든/파리티 테스트가 걸려 있는 공용 경로다.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbox.backtest.trade_sim import portfolio_sim, portfolio_sim_with_details
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

#: 15m 봉 하루치.
BARS_PER_DAY = 96


def _overlapping(n: int, pnl: float = 0.01, hold: int = 50) -> list[tuple[str, int, int, float]]:
    """서로 겹치는 n건의 트레이드 — 심볼이 다르므로 동일 심볼 규칙엔 안 걸린다."""
    return [(f"SYM{i}", 10 + i, 10 + i + hold, pnl) for i in range(n)]


def test_direction_diversity_blocks_beyond_max_same_direction() -> None:
    """같은 방향이 max_same_direction에 도달하면 그 다음 동일 방향 진입은 거절된다."""
    cfg = VolStraddleConfig()
    n = cfg.max_same_direction + 3
    trades = _overlapping(n)
    dirs = [1] * n  # 전부 LONG

    pnl, accepted = portfolio_sim_with_details(
        trades, max_concurrent=n + 5, directions=dirs, dir_diversity=True, cfg=cfg
    )
    assert len(accepted) == cfg.max_same_direction, f"{cfg.max_same_direction}건에서 멈춰야 하는데 {len(accepted)}건"
    assert len(pnl) == len(accepted)


def test_direction_diversity_counts_each_direction_separately() -> None:
    """LONG 상한에 걸려도 SHORT는 별도로 센다 — 라이브 `allow_direction`과 동일."""
    cfg = VolStraddleConfig()
    n_long = cfg.max_same_direction + 2
    trades = _overlapping(n_long + 2)
    dirs = [1] * n_long + [-1, -1]

    _, accepted = portfolio_sim_with_details(trades, max_concurrent=99, directions=dirs, dir_diversity=True, cfg=cfg)
    accepted_syms = {t[0] for t in accepted}
    # LONG은 상한까지만, SHORT 2건은 통과
    assert len(accepted) == cfg.max_same_direction + 2
    assert f"SYM{n_long}" in accepted_syms and f"SYM{n_long + 1}" in accepted_syms


def test_btc_gate_blocks_longs_below_ma() -> None:
    """BTC가 MA50 아래면 LONG은 거절되고 SHORT는 통과한다."""
    cfg = VolStraddleConfig()
    n_bars = 400
    # 계속 하락하는 BTC → 모든 봉에서 종가 < MA50
    btc = np.linspace(100.0, 50.0, n_bars)

    trades = [("AAA", 200, 250, 0.01), ("BBB", 210, 260, 0.01)]
    dirs = [1, -1]

    _, accepted = portfolio_sim_with_details(
        trades, max_concurrent=99, directions=dirs, btc_close=btc, btc_gate=True, cfg=cfg
    )
    assert [t[0] for t in accepted] == ["BBB"], "BTC 하락 구간에서 LONG이 통과했다"


def test_btc_gate_allows_longs_above_ma() -> None:
    """상승 구간에서는 LONG도 통과한다 — 게이트가 항상 막는 게 아니다."""
    cfg = VolStraddleConfig()
    btc = np.linspace(50.0, 100.0, 400)
    trades = [("AAA", 200, 250, 0.01)]

    _, accepted = portfolio_sim_with_details(
        trades, max_concurrent=99, directions=[1], btc_close=btc, btc_gate=True, cfg=cfg
    )
    assert len(accepted) == 1


def test_daily_loss_breaker_blocks_rest_of_day() -> None:
    """당일 실현손실이 한도를 넘으면 그날 남은 진입은 막히고, 다음 날 풀린다."""
    cfg = VolStraddleConfig()
    slots = cfg.max_concurrent
    # 한 건당 자기자본 영향 = pnl / slots. 한도를 확실히 넘기도록 큰 손실을 쓴다.
    big_loss = -cfg.daily_loss_limit_pct * slots * 2

    trades = [
        ("AAA", 1, 2, big_loss),  # 0일차: 손실 확정 → 이후 차단
        ("BBB", 10, 20, 0.05),  # 0일차: 차단돼야 함
        ("CCC", BARS_PER_DAY + 5, BARS_PER_DAY + 10, 0.05),  # 1일차: 다시 허용
    ]

    _, accepted = portfolio_sim_with_details(trades, max_concurrent=slots, daily_loss=True, cfg=cfg)
    assert [t[0] for t in accepted] == ["AAA", "CCC"], f"차단기가 잘못 동작: {[t[0] for t in accepted]}"


def test_daily_loss_breaker_stays_off_when_within_limit() -> None:
    """한도 안이면 아무것도 막지 않는다."""
    cfg = VolStraddleConfig()
    trades = [("AAA", 1, 2, -0.0001), ("BBB", 10, 20, 0.05)]
    _, accepted = portfolio_sim_with_details(trades, max_concurrent=cfg.max_concurrent, daily_loss=True, cfg=cfg)
    assert len(accepted) == 2


def test_filters_are_opt_in() -> None:
    """인자를 안 주면 기존 동작 그대로 — 공용 경로의 기존 숫자를 지킨다."""
    cfg = VolStraddleConfig()
    n = cfg.max_same_direction + 5
    trades = _overlapping(n)

    baseline = portfolio_sim(trades, max_concurrent=99)
    assert len(baseline) == n, "필터가 기본으로 켜져 기존 숫자를 바꿨다"

    pnl, accepted = portfolio_sim_with_details(trades, max_concurrent=99)
    assert len(accepted) == n


def test_direction_filter_requires_directions() -> None:
    """방향 배열 없이 방향 필터를 켜면 조용히 통과시키지 않고 실패한다."""
    with pytest.raises(ValueError, match="directions"):
        portfolio_sim_with_details(_overlapping(3), max_concurrent=99, dir_diversity=True)


# ──────────────────────────────────────────────────────────────────────────
#  방향 배선 — sim이 방향을 버리지 않는다
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "cache_extended").is_dir(),
    reason="확장 캐시 없음",
)
def test_sim_returns_directions_aligned_with_trades() -> None:
    """`return_directions=True`가 트레이드와 1:1로 정렬된 ±1 배열을 준다.

    방향 없이는 방향 분산·BTC 게이트 필터를 아예 적용할 수 없다.
    """
    from quantbox.backtest.sim_core import run_ma_cluster_live
    from quantbox.research.harness import MarketData

    md = MarketData.load(["BTCUSDT"])
    trades_only = run_ma_cluster_live(md.data)
    trades, dirs = run_ma_cluster_live(md.data, return_directions=True)

    assert trades == trades_only, "반환 형태가 트레이드 자체를 바꿨다"
    assert len(dirs) == len(trades)
    assert set(dirs) <= {1, -1}, f"방향이 ±1이 아니다: {set(dirs)}"
    assert len(set(dirs)) == 2, "한 방향만 나왔다 — 배선이 상수를 반환하는지 의심"
