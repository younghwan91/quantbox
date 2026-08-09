"""G4 파리티: 백테스트 상수를 모듈 로드 시점이 아니라 **호출 시점**에 읽는다.

`trade_sim.py`는 import 시점에 `COST_DIRECTIONAL` / `WARMUP_*` / `TF_*`를 전역으로
굳혀 놓았다. 그래서 실행 중에 설정을 바꿔도(스윕 스크립트, `.env` 수정, 테스트 픽스처)
이미 import된 sim은 **옛 값으로 계속 돈다** — 파라미터를 바꿨다고 믿는 백테스트가
실제로는 안 바뀐 채 돌아가는 조용한 오답의 원인이다.

`round_trip_cost = 2 × (fee_rate + slippage_rate)`이고 sim은 트레이드당 정확히
한 번 이 비용을 뺀다(`sim_core.sim_drive`: `pnl[cnt] = trade_pnl - cost`).
따라서 수수료를 Δ만큼 올리면 **모든 트레이드 PnL이 정확히 2Δ만큼** 낮아져야 한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"

#: 단일 심볼로 충분하다 — 검증 대상은 전파 여부지 성과가 아니다.
_SYM = "BTCUSDT"
_FEE_BUMP = 0.001


def _load_one() -> dict:
    from quantbox.research.harness import MarketData

    return MarketData.load([_SYM]).data


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_cost_change_propagates_to_ma_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """수수료를 올리면 재import 없이도 모든 트레이드 PnL이 정확히 2Δ 낮아진다."""
    from quantbox.backtest.sim_core import run_ma_cluster

    data = _load_one()
    base = run_ma_cluster(data)
    assert len(base) > 0, "심볼에 트레이드가 없다 — 테스트가 공허하다"

    monkeypatch.setenv("VS_FEE_RATE", str(VolStraddleConfig().fee_rate + _FEE_BUMP))
    bumped = run_ma_cluster(data)

    assert [t[:3] for t in base] == [t[:3] for t in bumped], "비용 변경이 트레이드 집합을 바꿨다"
    for b, x in zip(base, bumped, strict=True):
        assert b[3] - x[3] == pytest.approx(2 * _FEE_BUMP, abs=1e-12), "비용 변경이 sim에 전파되지 않았다"


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_cost_change_propagates_to_live_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """라이브 충실 경로도 같은 계약을 지킨다 — 단 비용도 명목가에 비례한다.

    이 경로는 CW 사이징이 기본 ON이므로 비용 증가분도 배수만큼 스케일된다
    (명목가를 0.75배로 태우면 수수료도 0.75배). 따라서 델타는 `2Δ × 배수` 집합에
    들어가야 한다 — 정확히 `2Δ`이면 오히려 사이징이 비용에 반영되지 않은 것이다.
    """
    from quantbox.backtest.sim_core import run_ma_cluster_live

    data = _load_one()
    base = run_ma_cluster_live(data)
    assert len(base) > 0

    monkeypatch.setenv("VS_FEE_RATE", str(VolStraddleConfig().fee_rate + _FEE_BUMP))
    bumped = run_ma_cluster_live(data)

    assert [t[:3] for t in base] == [t[:3] for t in bumped]
    allowed = [2 * _FEE_BUMP * m for m in VolStraddleConfig().ma_cluster_cw_multipliers]
    for b, x in zip(base, bumped, strict=True):
        delta = b[3] - x[3]
        assert any(delta == pytest.approx(a, abs=1e-12) for a in allowed), (
            f"비용 변경이 전파되지 않았거나 사이징과 무관하게 적용됐다: delta={delta}"
        )
    # 최소 한 건은 1.0배가 아니어야 한다 — 아니면 CW가 꺼진 것과 구별되지 않는다.
    assert any(b[3] - x[3] != pytest.approx(2 * _FEE_BUMP, abs=1e-12) for b, x in zip(base, bumped, strict=True))


@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_cost_change_propagates_to_donchian(monkeypatch: pytest.MonkeyPatch) -> None:
    """Donchian 수집기도 모듈 전역이 아니라 호출 시점 설정을 쓴다."""
    from quantbox.backtest.trade_sim import collect_donchian_trades

    data = _load_one()
    base = collect_donchian_trades(data)
    assert len(base) > 0

    monkeypatch.setenv("VS_FEE_RATE", str(VolStraddleConfig().fee_rate + _FEE_BUMP))
    bumped = collect_donchian_trades(data)

    assert [t[:3] for t in base] == [t[:3] for t in bumped]
    for b, x in zip(base, bumped, strict=True):
        assert b[3] - x[3] == pytest.approx(2 * _FEE_BUMP, abs=1e-12)


def test_module_globals_are_not_the_source_of_truth() -> None:
    """전역 상수는 남아 있어도 되지만, 값의 출처는 `get_trade_sim_config()`여야 한다.

    이 테스트는 회귀 방지용이다 — 새 sim이 다시 전역을 기본 인자로 굳히면 잡는다.
    """
    import inspect

    from quantbox.backtest import trade_sim

    for name in ("collect_ma_cluster_trades", "collect_donchian_trades"):
        sig = inspect.signature(getattr(trade_sim, name))
        cost_default = sig.parameters["cost"].default
        assert cost_default is None, f"{name}: cost 기본값이 import 시점에 굳어 있다 ({cost_default})"
