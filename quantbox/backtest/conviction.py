"""Conviction-weighted sizing — 라이브 판정을 백테스트에서 재사용한다 (G8).

라이브 `strategy.py:_check_ma_cluster`는 진입 봉에서 3개 조건을 세어 CW 점수(0~3)를
매기고, `ma_cluster_cw_multipliers`로 명목가를 스케일한다. 백테스트에는 이 축이 없어
모든 트레이드를 1.0배로 계산해 왔다 — 라이브와 사이징 축이 통째로 달랐다.

여기서는 라이브의 조건을 **봉 배열 전체에 대해 미리 계산**한다. 세 조건 모두 진입 봉의
정보만 쓰므로(과거 20봉 거래량 평균 포함) 사전 계산해도 lookahead가 생기지 않는다.

트레이드 PnL이 자기자본 대비 **비율**이므로, 명목가를 m배 하는 것은 비용까지 포함한
순 수익률에 m을 곱하는 것과 같다 — 적용은 `pnl * mult[entry_bar]`.
"""

import numpy as np

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

__all__ = ["ma_cluster_cw_mults", "ma_cluster_cw_scores"]

#: 라이브 `strategy.py:508` — 거래량 조건은 20봉이 모여야 평가된다.
_VOL_WINDOW = 20


def ma_cluster_cw_scores(
    close: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    volume: np.ndarray,
    cfg: VolStraddleConfig | None = None,
) -> np.ndarray:
    """봉마다 MA cluster CW 점수(0~3)를 계산한다 — 라이브와 동일 판정.

    라이브 대응 (`strategy.py:500-514`):
      1. 좁은 스프레드: ``spread% < ma_cluster_cw_spread_threshold * 100``
      2. 거래량 스파이크: ``volume[i] > ma_cluster_cw_volume_mult × mean(직전 20봉)``
      3. 강한 모멘텀: ``|mom[i]| > ma_cluster_cw_mom_mult × ma_cluster_min_momentum``

    Args:
        close: TF 종가 배열.
        sma20: SMA(20) — 클러스터 밴드 구성.
        sma50: SMA(50).
        sma100: SMA(100).
        sma200: SMA(200).
        momentum: 모멘텀 배열 (라이브는 `compute_momentum(close, 8)`).
        volume: TF 거래량 배열.
        cfg: 설정. 생략 시 `VolStraddleConfig()` (Rule #6 — 하드코딩 금지).

    Returns:
        길이 ``len(close)``의 int8 점수 배열. `ma_cluster_cw_enabled`가 꺼져 있으면
        전부 0 (라이브도 이때 점수를 세지 않는다).
    """
    cfg = cfg if cfg is not None else VolStraddleConfig()
    n = len(close)
    scores: np.ndarray = np.zeros(n, dtype=np.int8)
    if not cfg.ma_cluster_cw_enabled:
        return scores

    # ── 1) 좁은 스프레드 ── 라이브는 퍼센트 단위로 비교한다 (`threshold * 100`).
    ma_max = np.maximum(np.maximum(sma20, sma50), np.maximum(sma100, sma200))
    ma_min = np.minimum(np.minimum(sma20, sma50), np.minimum(sma100, sma200))
    with np.errstate(invalid="ignore", divide="ignore"):
        spread_pct = np.where(close > 0, (ma_max - ma_min) / close * 100.0, np.nan)
        scores += (spread_pct < cfg.ma_cluster_cw_spread_threshold * 100.0).astype(np.int8)

    # ── 2) 거래량 스파이크 ── 진입 봉 포함 직전 20봉 평균 대비.
    if n >= _VOL_WINDOW:
        # cumsum 대신 sliding window 평균 — NaN 거래량도 라이브 nanmean과 같게 다룬다.
        windows = np.lib.stride_tricks.sliding_window_view(volume, _VOL_WINDOW)
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(windows, axis=1)
        spike: np.ndarray = np.zeros(n, dtype=bool)
        tail = volume[_VOL_WINDOW - 1 :]
        spike[_VOL_WINDOW - 1 :] = (avg > 0) & (tail > cfg.ma_cluster_cw_volume_mult * avg)
        scores += spike.astype(np.int8)

    # ── 3) 강한 모멘텀 ──
    with np.errstate(invalid="ignore"):
        strong = np.abs(momentum) > cfg.ma_cluster_cw_mom_mult * cfg.ma_cluster_min_momentum
    scores += strong.astype(np.int8)

    return scores


def ma_cluster_cw_mults(scores: np.ndarray, cfg: VolStraddleConfig | None = None) -> np.ndarray:
    """CW 점수를 명목가 배수로 바꾼다 — 라이브 `strategy.py:516`과 동일한 클램프.

    Args:
        scores: `ma_cluster_cw_scores` 결과.
        cfg: 설정. 생략 시 `VolStraddleConfig()`.

    Returns:
        길이가 같은 float64 배수 배열.
    """
    cfg = cfg if cfg is not None else VolStraddleConfig()
    table = np.asarray(cfg.ma_cluster_cw_multipliers, dtype=np.float64)
    idx = np.clip(scores, 0, len(table) - 1)
    return table[idx]
