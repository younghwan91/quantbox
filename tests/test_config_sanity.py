"""Production config sanity tests.

These tests encode the live-trading production invariants.
Any .env VS_ override or accidental config.py change that breaks parity
will fail here at pre-commit time — before it reaches live trading.

Rules encoded:
  - VR L/N = base_leverage / max_concurrent must be 3.0
  - MA L/N = ma_cluster_leverage / max_concurrent must be 2.1
  - hedge_mode must be True (straddle requires Binance Hedge Mode)
  - dynamic_capital_enabled must be False (causes -2019 margin errors)
  - max_concurrent must be 10
  - Delisted symbols (MKRUSDT/FTMUSDT/EOSUSDT) must not be in whitelist
  - Whitelist must have between 50 and 75 symbols
  - d_trail_distance < d_trail_activate (trail logic validity)
  - s_trail_distance < s_trail_activate (straddle trail logic)
  - max_drawdown_pct >= 0.25 (r48 analysis: P99 session DD, must not be too tight)
"""

import pytest

from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ──────────────────────────────────────────────────────────────────────────────
#  Delisted symbols that MUST NOT appear in whitelist (r55-r57 incident)
# ──────────────────────────────────────────────────────────────────────────────
_DELISTED = {"MKRUSDT", "FTMUSDT", "EOSUSDT"}


@pytest.fixture(scope="module")
def cfg() -> VolStraddleConfig:
    return VolStraddleConfig()


class TestVRLeverage:
    def test_vr_ln_equals_3(self, cfg: VolStraddleConfig) -> None:
        """VR L/N = 30/10 = 3.0 exactly — backtest parity requires this."""
        ln = cfg.base_leverage / cfg.max_concurrent
        assert ln == pytest.approx(3.0, abs=0.01), (
            f"VR L/N={ln:.2f} ≠ 3.0. Check .env VS_BASE_LEVERAGE / VS_MAX_CONCURRENT overrides."
        )

    def test_ma_ln_equals_2_1(self, cfg: VolStraddleConfig) -> None:
        """MA L/N = 21/10 = 2.1 exactly."""
        ln = cfg.ma_cluster_leverage / cfg.max_concurrent
        assert ln == pytest.approx(2.1, abs=0.01), f"MA L/N={ln:.2f} ≠ 2.1. Check .env VS_MA_CLUSTER_LEVERAGE override."

    def test_max_concurrent_is_10(self, cfg: VolStraddleConfig) -> None:
        """Slot count must be 10 — Kelly derivation based on this."""
        assert cfg.max_concurrent == 10, (
            f"max_concurrent={cfg.max_concurrent} ≠ 10. Changing slot count invalidates Kelly leverage derivation."
        )

    def test_base_leverage_is_30(self, cfg: VolStraddleConfig) -> None:
        """VR leverage must be 30x."""
        assert cfg.base_leverage == pytest.approx(30.0, abs=0.1), f"base_leverage={cfg.base_leverage} ≠ 30.0"


class TestRiskMode:
    def test_hedge_mode_enabled(self, cfg: VolStraddleConfig) -> None:
        """Hedge mode must be True — straddle requires Binance dualSidePosition."""
        assert cfg.hedge_mode is True, (
            "hedge_mode=False will break straddle (S_d8) position management. "
            "Binance requires dualSidePosition=true for simultaneous LONG+SHORT."
        )

    def test_dynamic_capital_disabled(self, cfg: VolStraddleConfig) -> None:
        """Dynamic capital must be False — causes -2019 margin errors (Incident #3)."""
        assert cfg.dynamic_capital_enabled is False, (
            "dynamic_capital_enabled=True causes -2019 APIError on 10-slot config. "
            "Incident #3 (2026-03-05): balance/(n+1) formula over-concentrates first slot."
        )

    def test_dd_halt_not_too_tight(self, cfg: VolStraddleConfig) -> None:
        """DD halt must be ≥ 10% — STRD-dominant era (MaxDD=2.6% BT, r82)."""
        assert cfg.max_drawdown_pct >= 0.10, (
            f"max_drawdown_pct={cfg.max_drawdown_pct:.0%} is too tight. "
            f"STRD-dominant portfolio MaxDD=2.6% (r82). "
            f"15% halt gives 6x headroom. Minimum recommended: 0.10 (10%)."
        )


class TestTrailLogic:
    def test_vr_trail_distance_less_than_activate(self, cfg: VolStraddleConfig) -> None:
        """Trail distance must be less than activation — otherwise trail triggers immediately."""
        assert cfg.d_trail_distance < cfg.d_trail_activate, (
            f"d_trail_distance={cfg.d_trail_distance} >= d_trail_activate={cfg.d_trail_activate}. "
            f"Trail would trigger at activation — effectively a fixed TP."
        )

    def test_straddle_trail_distance_less_than_activate(self, cfg: VolStraddleConfig) -> None:
        """Straddle trail distance must be less than straddle activation."""
        assert cfg.s_trail_distance < cfg.s_trail_activate, (
            f"s_trail_distance={cfg.s_trail_distance} >= s_trail_activate={cfg.s_trail_activate}."
        )

    def test_vr_sl_is_positive(self, cfg: VolStraddleConfig) -> None:
        assert 0 < cfg.d_sl < 0.10, f"d_sl={cfg.d_sl} out of valid range (0, 0.10)"

    def test_straddle_sl_is_positive(self, cfg: VolStraddleConfig) -> None:
        assert 0 < cfg.s_sl < 0.10, f"s_sl={cfg.s_sl} out of valid range (0, 0.10)"


class TestSymbolUniverse:
    def test_no_delisted_symbols_in_whitelist(self, cfg: VolStraddleConfig) -> None:
        """Delisted symbols must not be in whitelist (r55-r57 incident: 63 vs 60 syms)."""
        found = _DELISTED & set(cfg.symbol_whitelist)
        assert not found, (
            f"Delisted symbol(s) found in whitelist: {found}. "
            f"r55-r57 incident: exclude_symbols misses delisted symbols. "
            f"Remove from symbol_whitelist in config.py."
        )

    def test_whitelist_count_in_range(self, cfg: VolStraddleConfig) -> None:
        """Whitelist should have 50-75 symbols — alert on unexpected drift."""
        n = len(cfg.symbol_whitelist)
        assert 50 <= n <= 75, (
            f"symbol_whitelist has {n} symbols — expected 50-75. "
            f"Current production: 60 symbols. "
            f"If intentional, update this test's bounds."
        )

    def test_whitelist_all_usdt_pairs(self, cfg: VolStraddleConfig) -> None:
        """All whitelist symbols must end with USDT."""
        bad = [s for s in cfg.symbol_whitelist if not s.endswith("USDT")]
        assert not bad, f"Non-USDT symbols in whitelist: {bad}"
