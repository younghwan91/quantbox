"""autoresearch — 폭발적 성장의 수학적 설계.

목표: 단타로 월 20% 이상의 **실제** 수익. 모든 제약 자유(레버리지 포함).

## 목적함수를 왜 이렇게 잡았나

레버리지가 자유일 때 "월 수익률 기댓값 최대화"는 **내부 최적점이 없다**. 기댓값은
레버리지에 선형으로 커지는데, 로그 자산의 기댓값은 f*mu - f^2*sigma^2/2 로 위로 볼록해
어느 지점을 넘으면 **중앙값 경로가 0으로 간다**. 즉 기댓값을 좇으면 "평균은 무한대,
실제로 겪는 결과는 파산"이 된다. 실제로 얻는 성장을 최대화하는 유일한 목적함수는
로그 자산(켈리)이다.

그래서 primary_metric은 **중앙값 월 성장률**이다. 레버리지는 자유변수로 두되, 이 지표가
자기파괴 구간을 스스로 걸러낸다. 파산확률·MDD는 부수 지표로 함께 보고한다.

## 구조

  [FIXED]  데이터 로딩 · 포트폴리오 · 평가 — 수정하지 않는다.
  [SEARCH] 신호 생성 · 사이징 — 실험마다 여기만 바꾼다.

정본 경로만 쓴다. `sim_drive`/`portfolio_sim`/`compute_metrics` 재구현 금지.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from quantbox.backtest.sim_core import run_ma_cluster_live_fine_exits
from quantbox.backtest.trade_sim import portfolio_sim_with_details
from quantbox.research.harness.data import MarketData
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ══════════════════════════════════════════════════════════════════════════════
#  [FIXED] 상수 · 데이터 — 수정하지 않는다
# ══════════════════════════════════════════════════════════════════════════════

BARS_PER_YEAR_15M = 96 * 365
MONTHS_PER_YEAR = 12.0

#: 파산 판정 — 초기 자본 대비 이 아래로 내려가면 그 경로는 죽은 것으로 본다.
RUIN_FRACTION = 0.10

#: 부트스트랩 경로 수 (거래 순서를 섞어 파산확률을 추정).
N_PATHS = 400

#: 레버리지 탐색 격자. 성장최적점을 이 안에서 찾는다.
LEVERAGE_GRID = np.concatenate([np.arange(0.5, 10.01, 0.5), np.arange(11.0, 61.0, 2.0)])

_CACHE = os.path.join(os.path.dirname(__file__), "_universe_cache.npz")


def load_universe() -> tuple[dict, float]:
    """760심볼 유니버스와 타임라인 길이(년). 시각 정렬 보장."""
    md = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz")
    data = md.data
    end = max(int(d["_offset"]) + len(d["close"]) for d in data.values())
    return data, end / BARS_PER_YEAR_15M


def evaluate(pnl: np.ndarray, years: float, seed: int = 0) -> dict:
    """[FIXED] 거래별 손익 → 성장 지표.

    거래 손익은 **명목가 대비**다. 레버리지 L은 자기자본 대비 노출 배수이며,
    슬롯 분할(max_concurrent)로 나눈 뒤 적용한다 — 한 거래가 계좌 전체를 쓰지 않는다.

    각 L에 대해:
      1. 실제 순서 경로로 중앙값 월 성장률을 얻고,
      2. 순서를 섞은 N_PATHS 경로로 파산확률을 추정한다.
    성장최적 L(중앙값 월 성장률 최대)을 골라 보고한다.

    Args:
        pnl: 채택된 거래별 손익 (명목 대비 비율, 시간순).
        years: 타임라인 길이 (년).
        seed: 부트스트랩 시드.

    Returns:
        지표 dict. 거래가 부족하면 전부 0.
    """
    cfg = VolStraddleConfig()
    n = len(pnl)
    months = years * MONTHS_PER_YEAR
    if n < 50 or months <= 0:
        return {
            "median_monthly_pct": 0.0,
            "best_leverage": 0.0,
            "p_ruin": 1.0,
            "max_dd_pct": 100.0,
            "ann_ret_pct": 0.0,
            "N": n,
            "ev_bp": 0.0,
            "t": 0.0,
            "trades_per_year": 0.0,
        }

    per_trade = pnl / (SLOTS if SLOTS is not None else cfg.max_concurrent)  # 슬롯 분할 후 자기자본 대비
    rng = np.random.default_rng(seed)
    order = np.array([rng.permutation(n) for _ in range(N_PATHS)])

    best = None
    for lev in LEVERAGE_GRID:
        r = 1.0 + lev * per_trade
        if np.any(r <= 0):  # 한 거래로 계좌가 0 이하 — 이 레버리지는 정의상 파산
            continue
        logr = np.log(r)

        eq = np.exp(np.cumsum(logr))
        final = float(eq[-1])
        peak = np.maximum.accumulate(eq)
        mdd = float(np.max(1.0 - eq / peak)) * 100.0
        if final <= 0:
            continue
        monthly = final ** (1.0 / months) - 1.0

        # 부트스트랩: 순서를 섞어도 살아남는가 + 단일 경로 점추정의 신뢰구간
        #
        # primary_metric은 **실제 순서 한 경로**의 값이라 점추정이다. 실험 간 차이가
        # 진짜인지 경로 우연인지 구분하려면 분포가 필요하다. 지표 정의는 그대로 두고
        # 부트스트랩 5~95 백분위를 부수 지표로 함께 보고한다 — 사후 골대 이동 방지.
        paths = np.exp(np.cumsum(logr[order], axis=1))
        ruin = float(np.mean(np.min(paths, axis=1) <= RUIN_FRACTION))
        boot_monthly = paths[:, -1] ** (1.0 / months) - 1.0

        cand = {
            "median_monthly_pct": monthly * 100.0,
            "boot_p05": float(np.percentile(boot_monthly, 5)) * 100.0,
            "boot_p50": float(np.percentile(boot_monthly, 50)) * 100.0,
            "boot_p95": float(np.percentile(boot_monthly, 95)) * 100.0,
            "best_leverage": float(lev),
            "p_ruin": ruin,
            "max_dd_pct": mdd,
            "ann_ret_pct": (final ** (1.0 / years) - 1.0) * 100.0,
        }
        if best is None or cand["median_monthly_pct"] > best["median_monthly_pct"]:
            best = cand

    if best is None:
        best = {
            "median_monthly_pct": 0.0,
            "boot_p05": 0.0,
            "boot_p50": 0.0,
            "boot_p95": 0.0,
            "best_leverage": 0.0,
            "p_ruin": 1.0,
            "max_dd_pct": 100.0,
            "ann_ret_pct": 0.0,
        }

    ev = float(np.mean(pnl))
    t = ev / (float(np.std(pnl, ddof=1)) / np.sqrt(n)) if n > 1 else 0.0
    best |= {"N": n, "ev_bp": ev * 1e4, "t": t, "trades_per_year": n / years}
    return best


def report(tag: str, m: dict, elapsed: float) -> None:
    """[FIXED] grep 가능한 요약 블록."""
    print("---")
    print(f"primary_metric: {m['median_monthly_pct']:.4f}")
    print(f"best_leverage:  {m['best_leverage']:.1f}")
    print(f"boot_p05:       {m['boot_p05']:.4f}")
    print(f"boot_p50:       {m['boot_p50']:.4f}")
    print(f"boot_p95:       {m['boot_p95']:.4f}")
    print(f"p_ruin:         {m['p_ruin']:.4f}")
    print(f"max_dd_pct:     {m['max_dd_pct']:.2f}")
    print(f"ann_ret_pct:    {m['ann_ret_pct']:.2f}")
    print(f"N:              {m['N']}")
    print(f"trades_per_year:{m['trades_per_year']:.0f}")
    print(f"ev_bp:          {m['ev_bp']:.2f}")
    print(f"t:              {m['t']:.2f}")
    print(f"tag:            {tag}")
    print(f"elapsed_s:      {elapsed:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
#  [SEARCH] 신호 생성 — 실험마다 여기만 바꾼다
# ══════════════════════════════════════════════════════════════════════════════

TAG = "exp15-slots5"

#: [SEARCH] 포트폴리오 노브 — main()이 읽는다.
SLOTS: int | None = 5
DIR_DIVERSITY: bool | None = None  # None이면 cfg 기본값
SYMBOL_FILTER = None  # callable(sym, d) -> bool, None이면 전부


def generate(data: dict) -> tuple[list, list]:
    """거래 후보 생성. (trades, directions) 반환.

    baseline: 라이브 ma_cluster 진입 + 15m 청산 해상도, 프로덕션 파라미터 그대로.
    """
    # TP 8%는 최적화 대상이 아니라 천장이다 — 스윕이 격자 끝까지 단조 증가하며
    # 내부 최적점이 없다(exit-audit). 26% 승률/RR 3:1 구조는 소수 대형 승리로 먹고사는데
    # 그 꼬리를 자르고 있다. 사실상 제거해본다.
    return run_ma_cluster_live_fine_exits(
        data, exit_ratio=1, tp=1.0, max_duration=48, min_momentum=0.005, return_directions=True
    )


# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    cfg = VolStraddleConfig()
    data, years = load_universe()
    print(f"# 심볼 {len(data)}  타임라인 {years:.2f}년  비용 {cfg.round_trip_cost * 1e4:.2f}bp", flush=True)

    if SYMBOL_FILTER is not None:
        data = {k: v for k, v in data.items() if SYMBOL_FILTER(k, v)}
        print(f"# 심볼 필터 후 {len(data)}", flush=True)

    trades, dirs = generate(data)
    print(f"# 원거래 {len(trades)}", flush=True)

    slots = SLOTS if SLOTS is not None else cfg.max_concurrent
    dd = DIR_DIVERSITY if DIR_DIVERSITY is not None else cfg.dir_diversity_enabled
    pnl, _acc = portfolio_sim_with_details(
        trades,
        slots,
        directions=dirs,
        cfg=cfg,
        dir_diversity=dd,
    )
    pnl = np.asarray(pnl, dtype=np.float64)
    print(f"# 채택 {len(pnl)}", flush=True)

    report(TAG, evaluate(pnl, years), time.time() - t0)


if __name__ == "__main__":
    sys.exit(main())
