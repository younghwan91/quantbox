"""r122 — MA Cluster 롱 차단 게이트.

측정 근거: 6.58년/760심볼에서 ma_cluster 롱은 무조건 롱 대비 초과 EV ~1bp
(타이밍 정보 없음), 2024~2026 3년 연속 음수. 숏은 초과 +38.6bp(t=6.82).
(docs/backtest_results/202608/20260827_b3906aa_r122_long_short_split.md)

이 테스트가 지키는 불변식은 "롱이 막힌다"가 아니라 **라이브와 백테스트가 같은
config 필드를 읽는다**는 것이다. 한쪽만 막히면 파리티가 깨진다.
"""

from __future__ import annotations

import pytest

from quantbox.core.strategy_names import DONCHIAN, MA_CLUSTER, VR_DIRECTIONAL
from quantbox.execution.risk_guard import RiskGuard
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


@pytest.fixture(scope="module")
def real_market_data():
    """실데이터 4h 클러스터가 실제로 발화하는 심볼 묶음 (합성 데이터 금지 — 원칙 #1)."""
    from quantbox.research.harness import MarketData

    try:
        md = MarketData.load(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "ADAUSDT"])
    except Exception as e:  # pragma: no cover - 캐시 없는 환경
        pytest.skip(f"실데이터 캐시 없음: {e}")
    if len(md.data) < 3:
        pytest.skip("실데이터 캐시 부족")
    return md.data


@pytest.fixture
def cfg_long_off() -> VolStraddleConfig:
    return VolStraddleConfig(ma_cluster_long_enabled=False)


@pytest.fixture
def cfg_long_on() -> VolStraddleConfig:
    return VolStraddleConfig(ma_cluster_long_enabled=True)


class TestRiskGuardStrategyDirection:
    def test_ma_cluster_long_blocked_when_disabled(self, cfg_long_off):
        ok, reason = RiskGuard(cfg_long_off).allow_strategy_direction(MA_CLUSTER, 1)
        assert ok is False
        assert "long" in reason.lower()

    def test_ma_cluster_short_always_allowed(self, cfg_long_off):
        assert RiskGuard(cfg_long_off).allow_strategy_direction(MA_CLUSTER, -1)[0] is True

    def test_ma_cluster_long_allowed_when_enabled(self, cfg_long_on):
        assert RiskGuard(cfg_long_on).allow_strategy_direction(MA_CLUSTER, 1)[0] is True

    @pytest.mark.parametrize("strat", [VR_DIRECTIONAL, DONCHIAN, None])
    def test_other_strategies_unaffected(self, cfg_long_off, strat):
        """게이트는 ma_cluster 전용 — 다른 전략의 롱을 건드리면 안 된다."""
        assert RiskGuard(cfg_long_off).allow_strategy_direction(strat, 1)[0] is True

    def test_straddle_signal_unaffected(self, cfg_long_off):
        """스트래들(signal=2)은 방향 중립이라 차단 대상이 아니다."""
        assert RiskGuard(cfg_long_off).allow_strategy_direction(MA_CLUSTER, 2)[0] is True


class TestBacktestHonoursSameFlag:
    """백테스트 진입 생성기도 같은 플래그를 읽는지 — 파리티의 핵심."""

    def test_run_ma_cluster_live_drops_longs(self, real_market_data):
        from quantbox.backtest.sim_core import run_ma_cluster_live

        base_t, base_d = run_ma_cluster_live(real_market_data, return_directions=True, long_enabled=True)
        off_t, off_d = run_ma_cluster_live(real_market_data, return_directions=True, long_enabled=False)

        assert 1 in base_d, "픽스처에 롱 신호가 없으면 이 테스트는 아무것도 검증하지 못한다"
        assert 1 not in off_d
        assert set(off_d) <= {-1}

    def test_shorts_are_bit_identical(self, real_market_data):
        """롱을 끄는 것이 숏 체결을 바꾸면 안 된다 — 심볼별 독립 시뮬이므로 동일해야 한다."""
        from quantbox.backtest.sim_core import run_ma_cluster_live

        base_t, base_d = run_ma_cluster_live(real_market_data, return_directions=True, long_enabled=True)
        off_t, _ = run_ma_cluster_live(real_market_data, return_directions=True, long_enabled=False)

        base_shorts = [t for t, d in zip(base_t, base_d, strict=True) if d == -1]
        assert off_t == base_shorts

    def test_default_follows_config(self, real_market_data):
        """long_enabled=None이면 config 값을 따른다."""
        from quantbox.backtest import sim_core

        _, d_default = sim_core.run_ma_cluster_live(real_market_data, return_directions=True)
        cfg = VolStraddleConfig()
        if cfg.ma_cluster_long_enabled:
            assert 1 in d_default
        else:
            assert 1 not in d_default


def test_live_delegates_to_risk_guard() -> None:
    """라이브 진입 경로가 게이트를 위임하는지 — 사본을 두면 파리티가 갈라진다."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "quantbox" / "scalper.py").read_text()
    assert "allow_strategy_direction(" in src, "scalper가 방향 게이트를 위임하지 않는다"
    assert "ma_cluster_long_enabled" not in src, "scalper가 플래그를 직접 비교한다 — 판정을 다시 짜고 있다"


def test_backtest_and_live_read_the_same_field() -> None:
    """백테스트 live 경로와 RiskGuard가 같은 config 필드를 읽는지."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "quantbox"
    guard_src = (root / "execution" / "risk_guard.py").read_text()
    sim_src = (root / "backtest" / "sim_core.py").read_text()
    assert "cfg.ma_cluster_long_enabled" in guard_src
    assert "cfg.ma_cluster_long_enabled" in sim_src
