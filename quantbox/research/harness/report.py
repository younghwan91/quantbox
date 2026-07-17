"""표준 성과 리포트 — 지표 계산의 유일 경로.

프로덕션 `compute_metrics` / `bootstrap_pf` / `portfolio_sim`을 **재사용**한다.
harness가 더하는 것은 계산이 아니라 **올바른 기본값**이다. 관측된 실제 함정 둘:

1. `compute_metrics(portfolio_size=)` 기본값이 1(legacy 1x 복리)이다. research에서
   `compute_metrics`를 호출하는 92개 파일 중 이 인자를 넘기는 건 5개뿐이었다.
   나머지는 매 트레이드가 전액 복리로 굴러 연수익이 폭발한다(실측: ann_ret 약 17억%,
   Calmar 6100만 — trade_sim.compute_metrics docstring에 기록된 r116 baseline 아티팩트).
   여기서는 config의 `max_concurrent`가 기본값이다 — 라이브 슬롯 사이징과 같은 의미.

2. `data_years` 산출식이 6종으로 갈라져 있었다(`avg_bars/(4*24*365)`,
   `max_bars*15/(365.25*24*60)`, `sample_bars/35040.0`, `max(days)/365` ...).
   이 값은 연수익·Calmar를 직접 스케일하므로, 같은 트레이드 셋이 스크립트마다
   다른 연수익으로 찍혔다. 여기서는 정렬된 공통 타임라인(max_bars) 기준 하나로 고정한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantbox.backtest.trade_sim import bootstrap_pf, compute_metrics, portfolio_sim
from quantbox.research.harness.data import MarketData
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

#: 15m 봉 기준 1년 봉 수 (365.25일 = 윤년 평균).
BARS_PER_YEAR_15M = 365.25 * 24 * 60 / 15

Trade = tuple[str, int, int, float]


def data_years(data: MarketData) -> float:
    """정렬된 공통 타임라인 길이를 연 단위로 환산한다.

    포트폴리오 백테스트가 실제로 커버하는 구간은 정렬 후 **가장 긴** 심볼의 길이다
    (짧은 심볼은 앞쪽이 `_offset`으로 비어 있을 뿐 같은 타임라인 위에 있다).
    평균 봉수를 쓰면 기간을 과소평가해 연수익이 부풀려진다 — 갈라진 구현 중 다수가
    그렇게 하고 있었다.
    """
    if len(data) == 0:
        return 0.0
    max_bars = max(len(d["close"]) + d["_offset"] for d in data.data.values())
    return max_bars / BARS_PER_YEAR_15M


def standard(
    trades: list[Trade],
    data: MarketData,
    cfg: VolStraddleConfig | None = None,
    *,
    apply_portfolio_sim: bool = True,
    straddle_syms: frozenset[str] | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """트레이드 리스트 → 표준 지표 묶음.

    Args:
        trades: `(symbol, entry_bar, exit_bar, pnl)` 튜플 리스트.
        data: 기간 산출용 MarketData.
        cfg: 파라미터 출처. None이면 `VolStraddleConfig()`.
        apply_portfolio_sim: 슬롯 경합(max_concurrent)을 적용할지. 라이브가 슬롯을
            공유하므로 기본 True — 끄면 동시에 무한히 들어갈 수 있는 비현실적 수치가 된다.
        straddle_syms: 스트래들 심볼 집합 (portfolio_sim에 전달).
        n_boot: 부트스트랩 표본 수. 0 이하면 부트스트랩을 생략한다(스윕처럼 수천 번
            돌릴 때 비용이 크므로). 생략하면 `bootstrap` 키는 빈 dict.

    Returns:
        `compute_metrics` 지표 + `bootstrap` + `years` / `n_raw` / `n_filled`.
    """
    cfg = cfg or VolStraddleConfig()
    years = data_years(data)

    if apply_portfolio_sim:
        pnl = portfolio_sim(trades, cfg.max_concurrent, straddle_syms)
    else:
        pnl = np.array([t[3] for t in trades], dtype=np.float64)

    metrics = compute_metrics(pnl, years, portfolio_size=cfg.max_concurrent)
    metrics["bootstrap"] = bootstrap_pf(pnl, n_boot=n_boot) if n_boot > 0 else {}
    metrics["years"] = round(years, 2)
    metrics["n_raw"] = len(trades)
    metrics["n_filled"] = len(pnl)
    return metrics


def render(metrics: dict[str, Any], title: str = "") -> str:
    """지표를 사람이 읽는 한 덩어리 텍스트로. 리포트 마크다운에 그대로 붙일 수 있다."""
    b = metrics.get("bootstrap", {})
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines += [
        f"N={metrics['N']}  (raw {metrics.get('n_raw', '?')} → slot-filled {metrics.get('n_filled', '?')})",
        f"PF={metrics['PF']}  EV={metrics['EV_bp']}bp  t={metrics['t']}  WR={metrics['WR']}%  RR={metrics['RR']}",
        f"ret={metrics['ret_pct']}%  maxDD={metrics['max_dd_pct']}%  ann={metrics['ann_ret_pct']}%  "
        f"calmar={metrics['calmar']}  ({metrics.get('years', '?')}y)",
    ]
    if b:
        lines.append(f"bootstrap PF: median={b['median']}  CI=[{b['CI_lo']}, {b['CI_hi']}]  P(PF≥1)={b['P_PF_ge_1']}%")
    return "\n".join(lines)
