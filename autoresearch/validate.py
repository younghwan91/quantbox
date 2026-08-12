"""신규 상장 숏 — 편향 전수 검증.

`listing_strategy.py`의 포트폴리오 수치(연 61.8%)는 **오염됐다**: 배분 `f`를 검증셋에서
최적화했다. 여기서는 그것을 포함해 이 전략에 걸릴 수 있는 편향을 하나씩 공격한다.

## 공격 목록

1. 사이징 누출  — f를 학습에서만 정하고 검증에 적용
2. 단일 분할     — 워크포워드 다중 fold로 대체
3. 생존 편향     — 보유 중 상장폐지된 종목의 처리
4. 비용 가정     — 12.6bp는 신규 상장에 낙관적. 50~300bp까지 스윕
5. 파라미터 선택 — OBS×HOLD×분위 격자의 **분포**를 보고, 최량만 보지 않는다
6. 상장일 정의   — 첫 봉이 정말 상장인가
"""

from __future__ import annotations

import numpy as np

from quantbox.research.harness.data import MarketData

BPD = 96
F_GRID = np.arange(0.01, 0.51, 0.01)


def collect(data: dict, obs_d: int, hold_d: int) -> tuple[list[dict], dict]:
    """상장 이벤트 + 제외 사유 집계 (생존 편향 점검용)."""
    t0 = min(int(d["_offset"]) for d in data.values())
    obs, hold = obs_d * BPD, hold_d * BPD
    out, drop = [], {"기존종목": 0, "보유중_데이터끊김": 0, "관측창부족": 0, "무거래": 0}

    for sym, d in data.items():
        off = int(d["_offset"])
        if off <= t0 + BPD:
            drop["기존종목"] += 1
            continue
        c, v, tb = d["close"], d["volume"], d.get("taker_buy")
        if tb is None or len(c) < obs + 1 or c[0] <= 0:
            drop["관측창부족"] += 1
            continue
        if float(v[:obs].sum()) <= 0 or c[obs] <= 0:
            drop["무거래"] += 1
            continue
        truncated = len(c) < obs + hold + 1
        if truncated:
            drop["보유중_데이터끊김"] += 1
        path = c[obs : obs + hold + 1] / c[obs]
        out.append(
            {
                "sym": sym,
                "entry_bar": off + obs,
                "accum": float(tb[:obs].sum() / float(v[:obs].sum())) - 0.5 - float(c[obs - 1] / c[0] - 1.0),
                "path": path,
                "truncated": truncated,
            }
        )
    out.sort(key=lambda r: r["entry_bar"])
    return out, drop


def ret(t: dict, cost: float) -> float:
    """숏 순수익. 보유 중 데이터가 끊긴 종목은 마지막 관측가로 청산한 것으로 본다."""
    return float(1.0 - t["path"][-1]) - cost


def log_growth(r: np.ndarray, f: float) -> float:
    w = 1.0 + f * r
    return -np.inf if np.any(w <= 0) else float(np.mean(np.log(w)))


def best_f(r: np.ndarray) -> float:
    g = [(log_growth(r, f), f) for f in F_GRID]
    g = [x for x in g if np.isfinite(x[0]) and x[0] > 0]
    return max(g)[1] if g else 0.0


def walk_forward(trades: list[dict], cost: float, n_folds: int = 5) -> dict:
    """fold마다 과거로 f를 정하고 그 fold에서만 평가. 사이징 누출 없음."""
    n = len(trades)
    if n < 100:
        return {"n": 0}
    edges = [int(n * k / n_folds) for k in range(n_folds + 1)]
    gs, ns, fs = [], 0, []
    for k in range(2, n_folds):  # 최소 2 fold 학습 후 시작
        tr = trades[: edges[k]]
        te = trades[edges[k] : edges[k + 1]]
        if len(tr) < 50 or len(te) < 20:
            continue
        f = best_f(np.array([ret(t, cost) for t in tr]))
        if f == 0.0:
            continue
        rte = np.array([ret(t, cost) for t in te])
        g = log_growth(rte, f)
        if np.isfinite(g):
            gs.append(g * len(te))
            ns += len(te)
            fs.append(f)
    if ns == 0:
        return {"n": 0}
    span = (trades[-1]["entry_bar"] - trades[0]["entry_bar"]) / (BPD * 365)
    per_year = n / max(span, 1e-9)
    g_per = sum(gs) / ns
    return {
        "n": ns,
        "g_bp": g_per * 1e4,
        "f_med": float(np.median(fs)),
        "ann": np.expm1(g_per * per_year),
        "per_year": per_year,
    }


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data

    # ── 3. 생존 편향 점검 ──
    tr7, drop = collect(data, 7, 30)
    trunc = sum(1 for t in tr7 if t["truncated"])
    print("=== 3. 생존 편향")
    print(f"  이벤트 {len(tr7)}건  |  제외: {drop}")
    print(f"  보유 중 데이터 끊김 {trunc}건 ({trunc / max(len(tr7), 1) * 100:.1f}%) — 마지막 관측가로 청산 처리")
    if trunc:
        a = np.array([ret(t, 12.6e-4) for t in tr7 if t["truncated"]])
        b = np.array([ret(t, 12.6e-4) for t in tr7 if not t["truncated"]])
        print(f"  끊긴 종목 숏수익 중앙 {np.median(a) * 100:>7.2f}%  |  정상 {np.median(b) * 100:>7.2f}%")
        print("  → 끊긴 쪽이 나으면 제외 시 **보수적**, 나쁘면 낙관 편향")

    # ── 1+2. 사이징 누출 제거 + 워크포워드 ──
    print("\n=== 1+2. 워크포워드 (f는 과거에서만 결정)")
    print(f"{'후보':<12}{'평가N':>7}{'f중앙':>8}{'거래당bp':>10}{'연%':>9}")
    print("-" * 46)
    acc = np.array([t["accum"] for t in tr7])
    thr = float(np.quantile(acc, 0.75))
    for lab, pool in (("전체상장", tr7), ("상위25%", [t for t in tr7 if t["accum"] > thr])):
        r = walk_forward(pool, 12.6e-4)
        if r["n"]:
            print(f"{lab:<12}{r['n']:>7}{r['f_med']:>8.2f}{r['g_bp']:>10.1f}{r['ann'] * 100:>8.1f}%")

    # ── 4. 비용 민감도 ──
    print("\n=== 4. 비용 민감도 (신규 상장 스프레드는 넓다)")
    print(f"{'왕복비용':<10}{'전체상장 연%':>14}{'상위25% 연%':>14}")
    print("-" * 38)
    sel = [t for t in tr7 if t["accum"] > thr]
    for c_bp in (12.6, 50, 100, 200, 300):
        a = walk_forward(tr7, c_bp * 1e-4)
        b = walk_forward(sel, c_bp * 1e-4)
        fa = f"{a['ann'] * 100:>13.1f}%" if a["n"] else f"{'—':>14}"
        fb = f"{b['ann'] * 100:>13.1f}%" if b["n"] else f"{'—':>14}"
        print(f"{c_bp:<10.1f}{fa}{fb}")

    # ── 5. 파라미터 격자의 분포 ──
    print("\n=== 5. 파라미터 민감도 (최량이 아니라 분포를 본다)")
    anns = []
    for obs_d in (3, 5, 7, 10, 14):
        for hold_d in (14, 30, 45, 60):
            tt, _ = collect(data, obs_d, hold_d)
            if len(tt) < 200:
                continue
            r = walk_forward(tt, 12.6e-4)
            if r["n"]:
                anns.append((obs_d, hold_d, r["ann"] * 100))
    if anns:
        v = np.array([a[2] for a in anns])
        print(
            f"  격자 {len(anns)}개  연수익 중앙 {np.median(v):.1f}%  "
            f"최소 {v.min():.1f}%  최대 {v.max():.1f}%  양수 {(v > 0).mean() * 100:.0f}%"
        )
        for o, h, a in sorted(anns, key=lambda x: -x[2])[:3]:
            print(f"    최상위: obs{o}일/hold{h}일 → {a:.1f}%")
        for o, h, a in sorted(anns, key=lambda x: x[2])[:3]:
            print(f"    최하위: obs{o}일/hold{h}일 → {a:.1f}%")


if __name__ == "__main__":
    main()
