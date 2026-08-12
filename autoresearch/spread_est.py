"""신규 상장 스프레드 추정 — Corwin-Schultz 고저가 추정량.

신규 상장 숏 전략의 손익분기 왕복 비용은 **100bp**인데, 상장 직후 스프레드를 실측한
적이 없다. 소켓 수집은 2026-08-09부터라 과거 상장 시점과 겹치지 않는다.

Corwin & Schultz (2012, JF)는 **고가/저가만으로** 스프레드를 추정한다. 아이디어:
고저 범위는 (1) 진짜 변동성과 (2) 매수/매도호가 스프레드를 함께 담는데, 변동성은
기간에 비례하고 스프레드는 비례하지 않는다. 1기간 범위와 2기간 범위를 비교하면
둘을 분리할 수 있다.

**추정량을 그냥 믿지 않는다.** 소켓 bookTicker 실측이 있는 44심볼에서 먼저 대조해
편향을 재고, 그 보정을 신규 상장에 적용한다.
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np

from quantbox.research.harness.data import MarketData

BPD = 96
SOCKET = "data/socket/bookTicker"


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> float:
    """왕복 스프레드 추정치 (비율). 음수 추정은 0으로 절단해 평균낸다.

    Args:
        high, low: 같은 길이의 고가/저가 배열 (연속 봉).

    Returns:
        스프레드 비율 (0.001 = 10bp). 계산 불가면 nan.
    """
    h, lo = np.asarray(high, float), np.asarray(low, float)
    ok = (h > 0) & (lo > 0) & (h >= lo)
    if ok.sum() < 20:
        return float("nan")
    h, lo = h[ok], lo[ok]
    if len(h) < 2:
        return float("nan")

    b = np.log(h[:-1] / lo[:-1]) ** 2 + np.log(h[1:] / lo[1:]) ** 2
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(lo[:-1], lo[1:])
    g = np.log(h2 / l2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    return float(np.mean(np.maximum(s, 0.0)))


def realised_spread_from_socket(sym: str, max_files: int = 3) -> float:
    """소켓 bookTicker에서 실측 스프레드 중앙값 (비율). 없으면 nan."""
    # 마지막 파일은 수집기가 아직 쓰는 중이라 gzip 스트림이 미완결이다 — 제외한다.
    files = sorted(glob.glob(os.path.join(SOCKET, sym, "*.jsonl.gz")))[:-1][:max_files]
    if not files:
        return float("nan")
    vals = []
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                for i, line in enumerate(fh):
                    if i % 50:  # 표본추출 — 전수는 느리다
                        continue
                    d = json.loads(line)["d"]
                    bid, ask = float(d["b"]), float(d["a"])
                    if bid > 0 and ask > bid:
                        vals.append((ask - bid) / ((ask + bid) / 2))
                    if len(vals) > 40000:
                        break
        except Exception:
            # 손상·미완결 gzip은 건너뛴다. 수집기가 상시 쓰고 있어 흔하다.
            continue
    return float(np.median(vals)) if vals else float("nan")


def main() -> None:
    data = MarketData.load(cache_dir="data/cache_universe", pattern="*.npz").data

    # ── 1. 추정량 검증: 소켓 실측이 있는 심볼에서 대조 ──
    print("=== 1. 추정량 검증 (소켓 실측 대조)")
    syms = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SOCKET, "*")))
    rows = []
    for s in syms:
        if s not in data:
            continue
        real = realised_spread_from_socket(s)
        if not np.isfinite(real):
            continue
        d = data[s]
        est = corwin_schultz(d["high"][-BPD * 30 :], d["low"][-BPD * 30 :])
        if np.isfinite(est):
            rows.append((s, real * 1e4, est * 1e4))
    if not rows:
        print("  소켓 데이터 없음 — 검증 불가")
        return

    real = np.array([r[1] for r in rows])
    est = np.array([r[2] for r in rows])
    ratio = est / np.maximum(real, 1e-9)
    print(f"  대조 심볼 {len(rows)}개")
    print(f"  실측 중앙 {np.median(real):>7.2f}bp   추정 중앙 {np.median(est):>7.2f}bp")
    print(
        f"  추정/실측 배수: 중앙 {np.median(ratio):>5.2f}  p25 {np.percentile(ratio, 25):>5.2f}  p75 {np.percentile(ratio, 75):>5.2f}"
    )
    corr = np.corrcoef(np.log(np.maximum(real, 1e-9)), np.log(np.maximum(est, 1e-9)))[0, 1]
    print(f"  로그 상관 {corr:.3f}")
    calib = float(np.median(ratio))
    print(f"  → 보정계수 {1 / calib:.3f} (추정치를 이 값으로 곱해 실측 수준에 맞춘다)")
    for s, a, b in sorted(rows, key=lambda r: r[1])[:3] + sorted(rows, key=lambda r: -r[1])[:3]:
        print(f"    {s:<12} 실측 {a:>7.2f}  추정 {b:>7.2f}")

    # ── 2. 신규 상장의 스프레드 ──
    print("\n=== 2. 신규 상장 스프레드 (진입 시점 = 상장 후 7일)")
    t0 = min(int(d["_offset"]) for d in data.values())
    obs = 7 * BPD
    ests = []
    for s, d in data.items():
        if int(d["_offset"]) <= t0 + BPD:
            continue
        h, lo = d["high"], d["low"]
        if len(h) < obs + BPD * 30:
            continue
        # 진입 전후 구간(상장 7일차부터 30일)의 스프레드
        e = corwin_schultz(h[obs : obs + BPD * 30], lo[obs : obs + BPD * 30])
        if np.isfinite(e):
            ests.append(e * 1e4 * calib_adjust(calib))
    if not ests:
        print("  추정 불가")
        return
    a = np.array(ests)
    print(f"  신규 상장 {len(a)}심볼  (보정 적용)")
    for p in (10, 25, 50, 75, 90):
        print(f"    p{p:<3} {np.percentile(a, p):>8.2f} bp")
    print()
    rt = np.median(a) * 2  # 왕복 = 스프레드 크로싱 2회
    print(f"  스프레드 중앙 {np.median(a):.1f}bp → 왕복 크로싱 {rt:.1f}bp + 수수료 8.6bp = **{rt + 8.6:.1f}bp**")
    print(f"  손익분기 100bp 대비: {'통과' if rt + 8.6 < 100 else '미달'}")


def calib_adjust(calib: float) -> float:
    """추정량 편향 보정 — 검증에서 얻은 배수의 역수."""
    return 1.0 / calib if calib > 0 else 1.0


if __name__ == "__main__":
    main()
