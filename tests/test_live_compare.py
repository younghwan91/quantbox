"""라이브 대조 검증 (Phase 4) — 백테 vs 실현 discrepancy 감시.

compare_to_live는 백테스트 지표와 실현(라이브) 지표를 대조해 갭을 분해한다.
라이브 운영 재개 후 저널 실현 지표를 넣어 discrepancy를 감시한다(read-only).
"""

from __future__ import annotations

from quantbox.research.harness import report


def test_compare_aligned_within_tolerance():
    """작은 갭(슬리피지 수준)은 aligned=True."""
    bt = {"PF": 2.0, "WR": 68.0, "EV_bp": 60.0}
    live = {"PF": 1.9, "WR": 66.0, "EV_bp": 50.0}
    r = report.compare_to_live(bt, live)
    assert r["aligned"], r
    assert abs(r["discrepancy"]["PF"]["gap"] - (-0.1)) < 1e-9


def test_compare_flags_large_discrepancy():
    """큰 갭(백테가 라이브보다 훨씬 좋음)은 aligned=False → 알림 대상."""
    bt = {"PF": 2.0, "WR": 68.0, "EV_bp": 60.0}
    live = {"PF": 1.0, "WR": 50.0, "EV_bp": 10.0}
    r = report.compare_to_live(bt, live)
    assert not r["aligned"], r
    assert not r["pf_ok"] and not r["wr_ok"] and not r["ev_ok"]
    assert r["discrepancy"]["EV_bp"]["gap"] == -50.0


def test_compare_missing_keys_are_skipped():
    """실현 지표에 일부 키만 있어도 있는 것만 대조."""
    bt = {"PF": 2.0, "WR": 68.0, "EV_bp": 60.0}
    live = {"PF": 1.95}  # PF만
    r = report.compare_to_live(bt, live)
    assert "PF" in r["discrepancy"] and "WR" not in r["discrepancy"]
    assert r["aligned"]  # PF 갭 작음, 나머지 없음=ok


def test_compare_per_metric_gap_reported():
    bt = {"PF": 2.0, "WR": 68.0, "EV_bp": 60.0}
    live = {"PF": 1.8, "WR": 64.0, "EV_bp": 45.0}
    r = report.compare_to_live(bt, live)
    for k in ("PF", "WR", "EV_bp"):
        assert set(r["discrepancy"][k]) == {"backtest", "live", "gap"}
