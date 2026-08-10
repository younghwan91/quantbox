"""Conviction-weighted (CW) sizing — 진입 확신도 판정의 **단일 소스**.

라이브(`strategy.py`)와 백테스트(`backtest/sim_core.py`)가 **같은 함수를 호출**한다.
이전에는 라이브가 인라인으로 판정하고 백테스트가 그걸 벡터화해 다시 짰다 — 사본이었고,
라이브 조건이 바뀌면 조용히 갈라졌다. 실제로 `close <= 0` 경계에서 이미 갈라져 있었다
(라이브는 spread를 0.0으로 폴백해 조건1을 통과시키고, 사본은 NaN이라 탈락시켰다).

판정은 진입 봉의 정보만 쓴다 — 과거 20봉 거래량 평균을 포함해 lookahead가 없다.
"""

from __future__ import annotations

import numpy as np

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

__all__ = ["ma_cluster_cw_multiplier", "ma_cluster_cw_score", "ma_cluster_spread_pct"]

#: 거래량 조건은 20봉이 모여야 평가된다.
CW_VOL_WINDOW = 20


def ma_cluster_spread_pct(
    close: float,
    sma20: float,
    sma50: float,
    sma100: float,
    sma200: float,
) -> float:
    """MA 클러스터 스프레드를 **퍼센트 단위**로 계산한다.

    Args:
        close: 진입 봉 종가.
        sma20: SMA(20) 값.
        sma50: SMA(50) 값.
        sma100: SMA(100) 값.
        sma200: SMA(200) 값.

    Returns:
        `(max(MA) - min(MA)) / close * 100`. `close <= 0`이면 0.0
        (라이브 `strategy.py`의 폴백과 동일 — 값이 아니라 동작을 맞춘다).
    """
    ma_vals = (float(sma20), float(sma50), float(sma100), float(sma200))
    if close <= 0:
        return 0.0
    return (max(ma_vals) - min(ma_vals)) / close * 100.0


def ma_cluster_cw_score(
    spread_pct: float,
    mom_val: float,
    volume: np.ndarray | None,
    cfg: VolStraddleConfig | None = None,
) -> int:
    """진입 봉의 CW 점수(0~3)를 센다.

    세 조건은 각각 1점이다:
      1. 좁은 스프레드 — `spread_pct < ma_cluster_cw_spread_threshold * 100`
      2. 거래량 스파이크 — `volume[-1] > ma_cluster_cw_volume_mult × nanmean(직전 20봉)`
      3. 강한 모멘텀 — `|mom_val| > ma_cluster_cw_mom_mult × ma_cluster_min_momentum`

    Args:
        spread_pct: `ma_cluster_spread_pct` 결과 (퍼센트 단위).
        mom_val: 진입 봉 모멘텀. NaN은 호출자가 0.0으로 정규화해 넘긴다.
        volume: 진입 봉까지의 거래량 배열 (마지막 원소가 진입 봉). 20봉 미만이거나
            None이면 조건2는 평가하지 않는다.
        cfg: 설정. 생략 시 `VolStraddleConfig()` (Rule #6 — 하드코딩 금지).

    Returns:
        0~3 정수 점수. `ma_cluster_cw_enabled`가 꺼져 있으면 항상 0.
    """
    cfg = cfg if cfg is not None else VolStraddleConfig()
    if not cfg.ma_cluster_cw_enabled:
        return 0

    score = 0
    # 1) tight spread
    if spread_pct < cfg.ma_cluster_cw_spread_threshold * 100:
        score += 1
    # 2) volume spike
    if volume is not None and len(volume) >= CW_VOL_WINDOW:
        avg_vol = float(np.nanmean(volume[-CW_VOL_WINDOW:]))
        if avg_vol > 0 and float(volume[-1]) > cfg.ma_cluster_cw_volume_mult * avg_vol:
            score += 1
    # 3) strong momentum
    if abs(mom_val) > cfg.ma_cluster_cw_mom_mult * cfg.ma_cluster_min_momentum:
        score += 1
    return score


def ma_cluster_cw_multiplier(score: int, cfg: VolStraddleConfig | None = None) -> float:
    """CW 점수를 명목가 배수로 바꾼다.

    Args:
        score: `ma_cluster_cw_score` 결과.
        cfg: 설정. 생략 시 `VolStraddleConfig()`.

    Returns:
        `ma_cluster_cw_multipliers[min(score, len-1)]` — 표 길이를 넘으면 클램프한다.
    """
    cfg = cfg if cfg is not None else VolStraddleConfig()
    table = cfg.ma_cluster_cw_multipliers
    return float(table[min(max(score, 0), len(table) - 1)])
