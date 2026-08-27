"""펀딩 부호 정책 — 기본은 보수적, 방향 인지는 명시적 선택.

2026-08-27 실측(34심볼): 시간당 |v| 2.39e-5 vs 부호 있는 평균 7.34e-6.
20h 보유면 기본 모델이 **양방향 모두에게** 4.78bp를 물리지만 실제로는 롱이 1.47bp를
내고 숏이 1.47bp를 받는다. 한쪽 편들기가 아니라 양쪽 과대 청구다.

이 테스트가 지키는 건 "정확한 펀딩"이 아니라 **기본값이 조용히 완화되지 않는 것**이다.
"""

from __future__ import annotations

from quantbox.backtest.funding import funding_cost, funding_rate_per_hour, signed_funding_rate_per_hour
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

SYM = "BTCUSDT"
HOLD = 80.0  # 80 × 15m = 20h
FB = 0.0001


class TestConservativeDefault:
    def test_default_is_unchanged_and_nonnegative(self):
        assert funding_cost(SYM, HOLD, FB) >= 0.0
        assert funding_cost(SYM, HOLD, FB) == funding_rate_per_hour(SYM, FB) * HOLD * 0.25

    def test_direction_ignored_unless_opted_in(self):
        """direction만 넘기고 플래그를 안 켜면 아무것도 바뀌면 안 된다."""
        base = funding_cost(SYM, HOLD, FB)
        assert funding_cost(SYM, HOLD, FB, direction=1) == base
        assert funding_cost(SYM, HOLD, FB, direction=-1) == base

    def test_config_default_is_off(self):
        assert VolStraddleConfig().funding_direction_aware is False

    def test_disabled_funding_still_zero(self):
        assert funding_cost(SYM, HOLD, 0.0, direction=-1, direction_aware=True) == 0.0
        assert funding_cost(SYM, 0.0, FB, direction=-1, direction_aware=True) == 0.0


class TestDirectionAware:
    def test_long_pays_short_receives(self):
        lo = funding_cost(SYM, HOLD, FB, direction=1, direction_aware=True)
        sh = funding_cost(SYM, HOLD, FB, direction=-1, direction_aware=True)
        assert lo == -sh, "롱 지불과 숏 수취는 크기가 같아야 한다"

    def test_direction_aware_never_harsher_than_default(self):
        """방향 인지는 완화 방향이다 — 어느 쪽도 기본보다 더 물리면 안 된다."""
        base = funding_cost(SYM, HOLD, FB)
        for d in (1, -1):
            assert funding_cost(SYM, HOLD, FB, direction=d, direction_aware=True) <= base + 1e-18

    def test_signed_rate_is_not_abs_rate(self):
        """부호 있는 테이블이 절대값 테이블의 사본이면 이 기능은 무의미하다."""
        assert signed_funding_rate_per_hour(SYM, FB) <= funding_rate_per_hour(SYM, FB) + 1e-18

    def test_zero_direction_falls_back_to_conservative(self):
        assert funding_cost(SYM, HOLD, FB, direction=0, direction_aware=True) == funding_cost(SYM, HOLD, FB)
