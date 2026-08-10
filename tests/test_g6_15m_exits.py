"""G6 폭 좁히기: 4h 진입 + 15m 청산 재생의 계약을 고정한다.

봉 데이터로는 한 봉 안에서 고점과 저점 중 무엇이 먼저 왔는지 알 수 없다. 이 불확실성은
없앨 수 없지만 **봉을 잘게 쪼개면 좁아진다**. 진입은 라이브와 같이 4h 완성봉에서 하고,
청산만 15m 봉에서 판정한다.

청산 로직은 재구현하지 않는다 — 같은 `sim_drive`를 15m 배열에 태운다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"
RATIO = 16

pytestmark = pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")


def _load():
    from quantbox.research.harness import MarketData

    return MarketData.load(["BTCUSDT", "ETHUSDT", "SOLUSDT"]).data


def test_entries_are_identical_to_4h_path() -> None:
    """청산 격자만 바뀐다 — 진입은 라이브 4h 판정 그대로여야 한다.

    4h 경로는 진입을 그 4h 봉의 **첫** 15m 봉으로 보고하고, 15m 경로는 진입이
    실제로 일어나는 **마지막** 15m 봉(=4h 종가)에서 시작한다. 그래서 `ratio-1`만큼
    어긋나며, 이 보정을 빼면 두 집합은 완전히 같아야 한다.
    """
    from quantbox.backtest.sim_core import run_ma_cluster_live, run_ma_cluster_live_15m_exits

    data = _load()
    a = run_ma_cluster_live(data, cw_sizing=False)
    b = run_ma_cluster_live_15m_exits(data, cw_sizing=False)

    assert len(a) > 0
    assert len(a) == len(b), f"트레이드 수가 다르다: 4h={len(a)} 15m={len(b)}"
    assert {(s, eb) for s, eb, _, _ in a} == {(s, eb - (RATIO - 1)) for s, eb, _, _ in b}, (
        "진입 시점이 달라졌다 — 청산 격자 변경이 진입에 샜다"
    )


def test_15m_exits_are_mostly_earlier_than_4h_exits() -> None:
    """대부분 같거나 빨라지되, 일부는 **늦어질 수도 있다** — 그게 정상이다.

    SL만 보면 늦어질 수 없다: 4h 저가는 그 안의 15m 저가들의 최솟값이므로 4h에서
    닿았다면 15m에서도 같은 구간 안에 닿는다.

    하지만 트레일은 다르다. 4h는 봉 전체 고점으로 peak을 올린 뒤 그 기준선을 같은 봉의
    저가와 비교한다 — 고점이 먼저 왔다고 가정하는 셈이다. 15m에서 실제 순서가 반대면
    그 시점엔 트레일이 아직 안 걸려 트레이드가 더 살아남는다. **이 어긋남 자체가 G6가
    측정하려는 대상**이므로, 0건을 요구하면 오히려 측정이 사라진다.

    격자 정렬이 깨지면 이 비율이 무너지므로 하한만 고정한다.
    """
    from quantbox.backtest.sim_core import run_ma_cluster_live, run_ma_cluster_live_15m_exits

    data = _load()
    a = {(s, eb): xb for s, eb, xb, _ in run_ma_cluster_live(data, cw_sizing=False)}
    b = {(s, eb - (RATIO - 1)): xb for s, eb, xb, _ in run_ma_cluster_live_15m_exits(data, cw_sizing=False)}

    # 4h 청산봉은 그 봉의 **마지막** 15m 봉으로 보고되므로 같은 봉 안이면 동률이다.
    earlier = sum(1 for k in a if b[k] <= a[k])
    ratio = earlier / len(a)
    assert ratio >= 0.95, f"같거나 빠른 청산이 {ratio:.1%}뿐 — 격자 정렬이 어긋났을 가능성"


def test_cw_sizing_scales_without_moving_trades() -> None:
    """CW는 사이징이지 신호가 아니다 — 15m 경로에서도 진입/청산 봉은 불변."""
    from quantbox.backtest.sim_core import run_ma_cluster_live_15m_exits

    data = _load()
    base = run_ma_cluster_live_15m_exits(data, cw_sizing=False)
    cw = run_ma_cluster_live_15m_exits(data)

    assert [t[:3] for t in base] == [t[:3] for t in cw]
    allowed = {round(m, 10) for m in VolStraddleConfig().ma_cluster_cw_multipliers}
    ratios = {round(c[3] / b[3], 10) for b, c in zip(base, cw, strict=True) if b[3] != 0.0}
    assert ratios <= allowed, f"배수 테이블 밖의 스케일: {ratios - allowed}"


def test_directions_align_with_trades() -> None:
    """방향 배열이 트레이드와 1:1이어야 포트폴리오 필터를 걸 수 있다."""
    from quantbox.backtest.sim_core import run_ma_cluster_live_15m_exits

    data = _load()
    trades, dirs = run_ma_cluster_live_15m_exits(data, return_directions=True)
    assert len(dirs) == len(trades)
    assert set(dirs) <= {1, -1}
