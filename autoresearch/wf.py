"""워크포워드 거래 선별 — 과거로 학습, 미래에서만 검증.

이 레포는 홀드아웃을 이미 소진했다(2023-01-01 이후를 r47~r120이 수십 번 재조회).
그래서 "전 구간에서 좋았다"는 아무 정보도 아니다. 유일하게 의미 있는 검정은
**각 시점에서 그때까지의 정보만으로 고른 거래가 그 다음 구간에서 벌었는가**이다.

모델은 릿지 회귀다. 트리/부스팅이 없어서가 아니라, 표본 대비 자유도가 낮아
과적합이 덜하기 때문이다 — 오늘 하루가 과적합으로 무너진 결론의 연속이었다.

절차 (fold마다):
  1. train = fold 시작 이전 전부, test = 그 fold
  2. train에서 표준화 + 릿지 적합 (pnl 예측)
  3. train의 예측값 분위로 임계 결정 → test에 그대로 적용
  4. test에서 선별된 거래의 EV/t 측정
학습 구간이 최소 MIN_TRAIN개 미만이면 그 fold는 건너뛴다.
"""

from __future__ import annotations

import numpy as np

MIN_TRAIN = 300
RIDGE_LAMBDA = 10.0


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """표준화 후 릿지 적합. (beta, mu, sd) 반환 — mu/sd는 test에 그대로 적용한다."""
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    z = (x - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    zi = np.hstack([np.ones((len(z), 1)), z])
    a = zi.T @ zi + lam * np.eye(zi.shape[1])
    a[0, 0] -= lam  # 절편은 벌하지 않는다
    beta = np.linalg.solve(a, zi.T @ y)
    return beta, mu, sd


def _predict(x: np.ndarray, beta: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([np.ones((len(z), 1)), z]) @ beta


def stats(p: np.ndarray) -> tuple[int, float, float]:
    """(N, EV_bp, t)."""
    n = len(p)
    if n < 2:
        return n, 0.0, 0.0
    ev = float(np.mean(p))
    sd = float(np.std(p, ddof=1))
    return n, ev * 1e4, (ev / (sd / np.sqrt(n)) if sd > 0 else 0.0)


def walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    eb: np.ndarray,
    end_bar: int,
    n_folds: int = 8,
    keep_frac: float = 0.5,
    xb: np.ndarray | None = None,
    lookback: int | None = None,
) -> dict:
    """워크포워드 선별 결과.

    Args:
        x: 피처 (n, F).
        y: 거래 손익.
        eb: 진입 봉 (시간 순서 기준).
        end_bar: 타임라인 마지막 봉.
        n_folds: 분할 수.
        keep_frac: 예측 상위 몇 %를 취할지.

    Returns:
        선별/전체 비교 지표.
    """
    order = np.argsort(eb, kind="stable")
    x, y, eb = x[order], y[order], eb[order]
    # 학습 자격은 **청산 시각** 기준이다. 진입만 과거면, fold 시작 시점에 아직 열려 있어
    # 손익을 모르는 거래로 학습하게 된다 — 미묘하지만 확실한 룩어헤드다.
    xb_sorted = eb if xb is None else np.asarray(xb)[order]

    sel, base, folds = [], [], []
    for k in range(1, n_folds):
        lo, hi = end_bar * k // n_folds, end_bar * (k + 1) // n_folds
        tr = xb_sorted < lo
        if lookback is not None:
            # 비정상성 대응: 전 이력이 아니라 **직전 lookback개**로만 학습한다.
            # 옛 체제에서 배운 관계가 최근 구간에서 역방향이 되는 것을 확인했으므로,
            # 창을 좁혀 현재 체제에 적응시킨다.
            idx = np.flatnonzero(tr)
            if len(idx) > lookback:
                tr = np.zeros_like(tr)
                tr[idx[-lookback:]] = True
        te = (eb >= lo) & (eb < hi)
        if tr.sum() < MIN_TRAIN or te.sum() < 30:
            continue

        beta, mu, sd = _fit_ridge(x[tr], y[tr], RIDGE_LAMBDA)
        thr = float(np.quantile(_predict(x[tr], beta, mu, sd), 1.0 - keep_frac))
        pick = _predict(x[te], beta, mu, sd) >= thr

        if pick.sum() >= 10:
            sel.append(y[te][pick])
            base.append(y[te])
            folds.append(
                (k, int(pick.sum()), int(te.sum()), float(np.mean(y[te][pick]) * 1e4), float(np.mean(y[te]) * 1e4))
            )

    if not sel:
        return {"n_sel": 0, "ev_sel": 0.0, "t_sel": 0.0, "n_base": 0, "ev_base": 0.0, "t_base": 0.0, "folds": []}

    s, b = np.concatenate(sel), np.concatenate(base)
    ns, evs, ts = stats(s)
    nb, evb, tb = stats(b)
    return {
        "n_sel": ns,
        "ev_sel": evs,
        "t_sel": ts,
        "n_base": nb,
        "ev_base": evb,
        "t_base": tb,
        "folds": folds,
        "pnl_sel": s,
    }
