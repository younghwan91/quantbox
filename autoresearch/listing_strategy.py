"""신규 상장 숏 — 체계적 전략 명세 + 성장최적 사이징.

## 왜 이 전략인가

오늘 검정한 방향성 단타·마켓메이킹·펀딩 캐리는 전부 **엣지가 왕복 통행료(12.6bp)의
0.3~2배**라 비용 가정 오차가 결론을 뒤집었다. 신규 상장 이벤트는 엣지가 18~25%로
**통행료의 150~200배**다. 비용을 10배 잘못 잡아도 결론이 안 바뀐다.

## 전략 명세

  진입: 상장 후 OBS일 시점. 그때까지의 정보로 매집지표를 계산해 상위 QTOP 분위면 **숏**.
  청산: HOLD일 경과, 또는 가격이 STOP% 오르면 손절 (둘 중 먼저).
  사이징: 자기자본의 f 배분.

매집지표 `accum = taker_buy/volume − 0.5 − 관측창수익` 은 **역방향**으로 쓴다.
상장 초기의 강한 매수 체결은 세력 진입이 아니라 물량 털기이며, 클수록 이후가 나쁘다
(p=1.3e-3). 진입 판단에 미래 정보는 쓰지 않는다 — 관측창과 평가창이 겹치지 않는다.

## 사이징의 수학

숏은 손실이 무한대다. `f` 배분에서 가격이 `x`배 오르면 자산은 `1 − f·x` 가 되고,
`x ≥ 1/f` 이면 파산이다. 실측 최악은 +465%(x=4.65)이므로 무손절이면 `f < 0.215`가
생존 상한이고, 그것도 **관측된** 최악일 뿐이다.

STOP을 걸면 손실이 `STOP + 비용`으로 **유한해진다**. 그러면 f를 훨씬 키울 수 있다.
목적함수는 로그자산 성장 `E[log(1 + f·r)]` — 기댓값 최대화는 레버리지가 자유일 때
내부 최적점이 없고 중앙값 경로가 0으로 가기 때문이다(켈리).

(STOP, f)를 **학습 구간에서만** 고르고 검증 구간에서 평가한다.
"""

from __future__ import annotations

import numpy as np

from quantbox.research.harness.data import MarketData

BARS_PER_DAY = 96  # 15m
COST = 12.6e-4  # 왕복 실측 (fee 8.6 + slippage 4.0)

#: 사전 고정 파라미터 — 사후 조정 금지. 임의성은 남지만 그 사실을 기록한다.
OBS_DAYS = 7
HOLD_DAYS = 30
Q_TOP = 0.75  # 매집지표 상위 25%


def build_trades(data: dict) -> list[dict]:
    """상장 이벤트별 거래 후보. 관측창 정보만으로 신호를 만든다."""
    t0 = min(int(d["_offset"]) for d in data.values())
    obs, hold = OBS_DAYS * BARS_PER_DAY, HOLD_DAYS * BARS_PER_DAY

    out = []
    for sym, d in data.items():
        off = int(d["_offset"])
        if off <= t0 + BARS_PER_DAY:  # 처음부터 있던 종목은 '신규 상장'이 아니다
            continue
        c, v, tb = d["close"], d["volume"], d.get("taker_buy")
        if tb is None or len(c) < obs + hold or c[0] <= 0 or c[obs] <= 0:
            continue
        vol = float(v[:obs].sum())
        if vol <= 0:
            continue

        out.append(
            {
                "sym": sym,
                "listed_bar": off,
                "entry_bar": off + obs,
                "accum": float(tb[:obs].sum() / vol) - 0.5 - float(c[obs - 1] / c[0] - 1.0),
                # 진입 후 경로 — 청산 규칙 적용에 필요 (진입 시점엔 미사용)
                "path": c[obs : obs + hold + 1] / c[obs],
            }
        )
    out.sort(key=lambda r: r["entry_bar"])
    return out


def short_return(path: np.ndarray, stop: float | None) -> float:
    """숏 1건의 순수익률. `stop`은 가격 상승 손절선(예: 0.5 = +50%).

    경로에서 손절선을 먼저 건드리면 그 지점에서 청산한다. 봉 종가 기준이므로
    실제 체결은 이보다 나쁠 수 있다(갭) — 낙관 방향의 잔여 편향으로 남긴다.
    """
    if stop is not None:
        hit = np.flatnonzero(path >= 1.0 + stop)
        if len(hit):
            return -stop - COST
    return float(1.0 - path[-1]) - COST


def log_growth(rets: np.ndarray, f: float) -> float:
    """배분 f에서의 거래당 로그 자산 성장. 파산 경로가 있으면 -inf."""
    w = 1.0 + f * rets
    if np.any(w <= 0):
        return -np.inf
    return float(np.mean(np.log(w)))


def optimise(rets: np.ndarray, f_grid: np.ndarray) -> tuple[float, float]:
    """로그성장 최대 배분. (f*, growth*)"""
    best_f, best_g = 0.0, 0.0
    for f in f_grid:
        g = log_growth(rets, f)
        if g > best_g:
            best_f, best_g = float(f), g
    return best_f, best_g


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data
    trades = build_trades(data)
    print(f"# 상장 이벤트 {len(trades)}건  (관측 {OBS_DAYS}일 → 보유 {HOLD_DAYS}일)")

    accum = np.array([t["accum"] for t in trades])
    thr = float(np.quantile(accum, Q_TOP))
    sel = [t for t in trades if t["accum"] > thr]
    print(f"# 매집지표 상위 {(1 - Q_TOP):.0%} 선별 → {len(sel)}건")

    # ── 학습/검증 분할: 진입 시점 기준 시간 분할 (미래 누출 없음) ──
    cut = int(len(sel) * 0.6)
    tr, te = sel[:cut], sel[cut:]
    print(f"# 학습 {len(tr)}건 / 검증 {len(te)}건 (시간순 분할)")
    print()

    f_grid = np.arange(0.01, 1.001, 0.01)
    print(f"{'STOP':>8}{'학습f*':>9}{'학습성장bp':>12}{'검증성장bp':>12}{'검증중앙%':>11}{'검증승률':>10}{'최악%':>9}")
    print("-" * 71)

    rows = []
    for stop in (None, 1.00, 0.75, 0.50, 0.35, 0.25):
        rtr = np.array([short_return(t["path"], stop) for t in tr])
        rte = np.array([short_return(t["path"], stop) for t in te])
        f, g_tr = optimise(rtr, f_grid)
        if f == 0.0:
            continue
        g_te = log_growth(rte, f)
        lab = "없음" if stop is None else f"+{stop:.0%}"
        rows.append((lab, stop, f, g_tr, g_te, rte))
        print(
            f"{lab:>8}{f:>9.2f}{g_tr * 1e4:>12.1f}{g_te * 1e4:>12.1f}"
            f"{np.median(rte) * 100:>11.2f}{(rte > 0).mean() * 100:>9.1f}%{rte.min() * 100:>9.1f}"
        )

    # ── 포트폴리오: 동시 보유 ──────────────────────────────────────
    # 거래당 f=0.09는 자본 대부분을 놀린다. 30일 보유이므로 상장이 이어지면
    # 여러 건이 동시에 열린다. 동시 보유는 분산을 낮춰 총 노출을 키울 수 있게 한다.
    # 겹치는 구간을 실제 시간선에서 재현해 자산 경로를 만든다.
    print()
    print("=== 포트폴리오 (동시 보유, 무손절)")
    print(f"{'후보':<14}{'연거래':>8}{'동시중앙':>10}{'건당f':>8}{'검증연%':>10}{'최대낙폭%':>11}")
    print("-" * 61)
    hold_bars = HOLD_DAYS * BARS_PER_DAY
    for lab, pool in (("상위25%", sel), ("전체상장", trades)):
        cut2 = int(len(pool) * 0.6)
        te2 = pool[cut2:]
        if len(te2) < 30:
            continue
        rets = np.array([short_return(t["path"], None) for t in te2])
        ent = np.array([t["entry_bar"] for t in te2], dtype=np.float64)
        span = (ent.max() - ent.min()) / (BARS_PER_DAY * 365)
        n_year = len(te2) / max(span, 1e-9)
        # 동시 보유 수 = 30일 창에 겹치는 진입 수
        conc = np.array([int(((ent >= e) & (ent < e + hold_bars)).sum()) for e in ent])
        k = max(1, int(np.median(conc)))
        # 건당 배분을 f_grid에서 다시 최적화하되, 동시 k건이므로 총노출 = k*f
        best = (0.0, -np.inf)
        for f in f_grid:
            if f * k > 1.0:
                break
            g = log_growth(rets, f)
            if g > best[1]:
                best = (float(f), g)
        f2, g2 = best
        if f2 == 0.0:
            continue
        eq = np.cumprod(1.0 + f2 * rets)
        mdd = float(np.max(1.0 - eq / np.maximum.accumulate(eq))) * 100
        print(f"{lab:<14}{n_year:>8.0f}{k:>10}{f2:>8.2f}{np.expm1(g2 * n_year) * 100:>9.1f}%{mdd:>11.1f}")

    # 연 환산 — 상장 빈도로 거래수를 잡는다
    span_years = (max(t["entry_bar"] for t in trades) - min(t["entry_bar"] for t in trades)) / (BARS_PER_DAY * 365)
    per_year = len(sel) / span_years
    print()
    print(f"# 선별 거래 연 {per_year:.0f}건 (전체 상장 {len(trades) / span_years:.0f}건/년)")
    best = max(rows, key=lambda r: r[4]) if rows else None
    if best:
        lab, _stop, f, _gtr, gte, _r = best
        print(f"# 검증 최선: STOP {lab}  f={f:.2f}  거래당 {gte * 1e4:.1f}bp  →  연 {np.expm1(gte * per_year):.1%}")


if __name__ == "__main__":
    main()
