"""강건성 게이트 검증 (Phase 3) — 오버피팅 차단.

robust_gate는 단일 in-sample PF가 아니라 워크포워드 OOS 분할 + 부트스트랩으로
강건성을 본다. 시간 전 구간에 일관되게 좋으면 통과, 특정 구간에만 의존하면 차단.
이게 stochastic optimization 강제의 핵심이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quantbox.research.harness import MarketData, report


def _md(bars: int) -> MarketData:
    arr = np.linspace(1.0, 2.0, bars)
    data = {"AAA": {"close": arr, "high": arr * 1.01, "low": arr * 0.99}}
    MarketData._align(data)
    return MarketData(data=data, source=Path("t"))


def _trades(pattern) -> list:
    """pattern(k,i) -> pnl. 4구간 × 15트레이드, 안 겹치게 배치(gap>exit)."""
    out = []
    for k in range(4):
        base = k * 1000
        for i in range(15):
            e = base + i * 20
            out.append(("AAA", e, e + 5, pattern(k, i)))
    return out


def test_robust_gate_passes_consistent_edge():
    """전 구간 일관되게 PF>1 → 통과."""
    md = _md(4000)
    trades = _trades(lambda k, i: 0.02 if i < 10 else -0.01)  # 각 구간 PF≈4
    g = report.robust_gate(trades, md, n_walk=4, min_boot_p_pf=50.0)
    assert g["passed"], g
    assert g["walk_ok"], g["walkforward_pf"]


def test_robust_gate_fails_regime_dependent():
    """첫 구간만 좋고 나머지 손실 → 오버핏 차단(walk_ok False)."""
    md = _md(4000)
    trades = _trades(lambda k, i: 0.02 if (k == 0 and i < 12) else -0.02)
    g = report.robust_gate(trades, md, n_walk=4, min_boot_p_pf=50.0)
    assert not g["passed"], g
    assert not g["walk_ok"], g["walkforward_pf"]


def test_robust_gate_empty_fails():
    g = report.robust_gate([], _md(1000))
    assert not g["passed"]


def test_robust_gate_reports_per_segment_pf():
    """구간별 PF를 노출한다(진단 가능)."""
    md = _md(4000)
    trades = _trades(lambda k, i: 0.02 if i < 10 else -0.01)
    g = report.robust_gate(trades, md, n_walk=4, min_boot_p_pf=50.0)
    assert len(g["walkforward_pf"]) == 4
    assert all(p is None or p > 0 for p in g["walkforward_pf"])
    assert "bootstrap" in g
