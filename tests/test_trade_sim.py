"""Tests for quantbox.backtest.trade_sim — Trade-Event Backtesting Engine.

Covers:
  - sim_directional_trades: SL/TP/max_hold, long/short, edge cases
  - sim_straddle_trades: dual-leg exit, cost deduction
  - collect_trades: multi-TF assembly, per-symbol dedup
  - portfolio_sim: slot contention, same-symbol block
  - compute_metrics: mathematical correctness on known data
  - pf_from_pnl: profit factor edge cases
  - bootstrap_pf: confidence interval output shape
"""

import numpy as np
import pytest

from quantbox.backtest.trade_sim import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    WARMUP_15m,
    bootstrap_pf,
    collect_trades,
    compute_metrics,
    pf_from_pnl,
    portfolio_sim,
    portfolio_sim_with_details,
    sim_directional_trades,
    sim_directional_trailing,
    sim_straddle_trades,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _flat_price(n: int, base: float = 100.0) -> tuple:
    """Flat price series (no movement)."""
    close = np.full(n, base)
    high = np.full(n, base)
    low = np.full(n, base)
    return close, high, low


def _trending_up(n: int, base: float = 100.0, step: float = 0.1) -> tuple:
    """Monotonically increasing price series."""
    close = base + np.arange(n) * step
    high = close + 0.05
    low = close - 0.05
    return close, high, low


# ═══════════════════════════════════════════════════════════════════════════
#  sim_directional_trades
# ═══════════════════════════════════════════════════════════════════════════


class TestSimDirectionalTrades:
    """Tests for the Numba-compiled directional trade simulator."""

    def test_no_signal_no_trades(self):
        """No signals → no trades."""
        close, high, low = _flat_price(100)
        sig = np.zeros(100, dtype=np.int8)
        dire = np.zeros(100, dtype=np.int8)
        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0008,
        )
        assert len(pnl) == 0

    def test_long_tp_hit(self):
        """Long trade hits TP via high price."""
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        # Signal on bar 0
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1  # long
        # Bar 5: high spikes above TP (5%)
        high[5] = 106.0  # +6% from entry

        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 1
        assert ent[0] == 0
        assert ext[0] == 5
        assert pnl[0] == pytest.approx(0.05, abs=1e-9)  # TP = +5%

    def test_long_sl_hit(self):
        """Long trade hits SL via low price."""
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1
        # Bar 3: low dips below SL (1%)
        low[3] = 98.0  # -2% from entry

        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 1
        assert pnl[0] == pytest.approx(-0.01, abs=1e-9)  # SL = -1%
        assert ext[0] == 3

    def test_short_tp_hit(self):
        """Short trade hits TP when low drops."""
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = -1  # short
        # Bar 4: low drops to 94 → TP for short (5%)
        low[4] = 94.0

        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 1
        assert pnl[0] == pytest.approx(0.05, abs=1e-9)

    def test_short_sl_hit(self):
        """Short trade hits SL when high rises."""
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = -1
        # Bar 2: high spikes → SL for short
        high[2] = 102.0

        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 1
        assert pnl[0] == pytest.approx(-0.01, abs=1e-9)

    def test_max_hold_exit(self):
        """Trade exits at max_hold if no SL/TP hit."""
        n = 20
        close, high, low = _flat_price(n)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1

        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            5,
            0.0,
        )
        assert len(pnl) == 1
        # Flat price → PnL ≈ 0 (close-based exit)
        assert abs(pnl[0]) < 1e-9
        # Exit at bar 5 (entry=0, max_hold=5 → bar 0+1..5)
        assert ext[0] == 5

    def test_cost_deducted(self):
        """Transaction cost is subtracted from trade PnL."""
        n = 20
        close, high, low = _flat_price(n)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1
        cost = 0.001

        pnl, _, _ = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            5,
            cost,
        )
        assert pnl[0] == pytest.approx(-cost, abs=1e-9)

    def test_dire_zero_skipped(self):
        """Signal with dire=0 is skipped (no trade)."""
        n = 20
        close, high, low = _flat_price(n)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        # dire[0] = 0 → should be skipped

        pnl, _, _ = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 0

    def test_multiple_trades_sequential(self):
        """Multiple trades fire sequentially (no overlap)."""
        n = 50
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        sig = np.zeros(n, dtype=np.int8)
        dire = np.zeros(n, dtype=np.int8)
        # Trade 1: bar 0, TP at bar 2
        sig[0] = 1
        dire[0] = 1
        high[2] = 106.0
        # Trade 2: bar 10, SL at bar 12
        sig[10] = 1
        dire[10] = 1
        low[12] = 98.0

        pnl, ent, ext = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 2
        assert pnl[0] == pytest.approx(0.05)
        assert pnl[1] == pytest.approx(-0.01)

    def test_sl_checked_before_tp_long(self):
        """For long: SL is checked (low) before TP (high) on same bar."""
        n = 10
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1
        # Bar 1: both SL and TP could trigger
        low[1] = 98.0  # SL hit
        high[1] = 106.0  # TP hit
        # SL is checked first → trade should end with -1%

        pnl, _, _ = sim_directional_trades(
            close,
            high,
            low,
            sig,
            dire,
            0.01,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 1
        assert pnl[0] == pytest.approx(-0.01)


# ═══════════════════════════════════════════════════════════════════════════
#  sim_straddle_trades
# ═══════════════════════════════════════════════════════════════════════════


class TestSimStraddleTrades:
    """Tests for straddle (long+short simultaneous) simulator."""

    def test_no_signal_no_trades(self):
        close, high, low = _flat_price(100)
        sig = np.zeros(100, dtype=np.int8)
        pnl, _, _ = sim_straddle_trades(
            close,
            high,
            low,
            sig,
            0.015,
            0.05,
            50,
            0.0,
        )
        assert len(pnl) == 0

    def test_flat_price_both_legs_breakeven(self):
        """Flat price → both legs close at max_hold → net ≈ 0."""
        n = 20
        close, high, low = _flat_price(n)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1

        pnl, _, _ = sim_straddle_trades(
            close,
            high,
            low,
            sig,
            0.015,
            0.05,
            5,
            0.0,
        )
        assert len(pnl) == 1
        # Both legs exit at same price → long = 0%, short = 0%, net = 0
        assert abs(pnl[0]) < 1e-9

    def test_straddle_cost_deducted(self):
        """Straddle cost is properly deducted."""
        n = 20
        close, high, low = _flat_price(n)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        cost = COST_STRADDLE

        pnl, _, _ = sim_straddle_trades(
            close,
            high,
            low,
            sig,
            0.015,
            0.05,
            5,
            cost,
        )
        assert pnl[0] == pytest.approx(-cost, abs=1e-9)

    def test_straddle_one_leg_tp(self):
        """One leg hits TP, other leg hits SL → net = TP - SL - cost."""
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        # Price goes up → long TP, short SL
        high[3] = 106.0  # long TP at +5%
        # Short SL triggers at entry*(1+sl) = 100*1.015 = 101.5
        # high[3]=106 > 101.5 → short SL also fires

        sl, tp = 0.015, 0.05
        pnl, _, _ = sim_straddle_trades(
            close,
            high,
            low,
            sig,
            sl,
            tp,
            50,
            0.0,
        )
        assert len(pnl) == 1
        # Half-size per leg: Long TP = +5%, Short SL = -1.5%, net = 0.5*(+5%) + 0.5*(-1.5%) = 1.75%
        assert pnl[0] == pytest.approx(0.5 * tp + 0.5 * (-sl), abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
#  portfolio_sim
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioSim:
    """Tests for slot-based portfolio simulation."""

    def test_empty_trades(self):
        pnl = portfolio_sim([], 5)
        assert len(pnl) == 0

    def test_single_trade(self):
        trades = [("BTCUSDT", 0, 10, 0.05)]
        pnl = portfolio_sim(trades, 5)
        assert len(pnl) == 1
        assert pnl[0] == 0.05

    def test_max_concurrent_blocks(self):
        """Trades exceeding max_concurrent are rejected."""
        trades = [
            ("BTCUSDT", 0, 100, 0.01),
            ("ETHUSDT", 1, 100, 0.02),
            ("SOLUSDT", 2, 100, 0.03),  # should be blocked (max=2)
        ]
        pnl = portfolio_sim(trades, 2)
        assert len(pnl) == 2
        assert pnl[0] == 0.01
        assert pnl[1] == 0.02

    def test_same_symbol_blocked(self):
        """Can't have two positions on same symbol simultaneously."""
        trades = [
            ("BTCUSDT", 0, 100, 0.01),
            ("BTCUSDT", 50, 150, 0.02),  # same symbol, overlapping
        ]
        pnl = portfolio_sim(trades, 10)
        assert len(pnl) == 1
        assert pnl[0] == 0.01

    def test_same_symbol_after_close(self):
        """Same symbol can re-enter after previous trade closes."""
        trades = [
            ("BTCUSDT", 0, 10, 0.01),
            ("BTCUSDT", 20, 30, 0.02),  # after first closes
        ]
        pnl = portfolio_sim(trades, 10)
        assert len(pnl) == 2

    def test_slot_freed_after_exit(self):
        """Slot is freed after trade exits, allowing new trade."""
        trades = [
            ("BTCUSDT", 0, 5, 0.01),
            ("ETHUSDT", 0, 5, 0.02),
            ("SOLUSDT", 10, 20, 0.03),  # after both close, slot free
        ]
        pnl = portfolio_sim(trades, 2)
        assert len(pnl) == 3

    def test_with_details_returns_metadata(self):
        """portfolio_sim_with_details returns both PnL and trade tuples."""
        trades = [
            ("BTCUSDT", 0, 10, 0.01),
            ("ETHUSDT", 5, 15, 0.02),
        ]
        pnl, accepted = portfolio_sim_with_details(trades, 5)
        assert len(pnl) == 2
        assert len(accepted) == 2
        assert accepted[0][0] == "BTCUSDT"
        assert accepted[1][0] == "ETHUSDT"

    def test_chronological_order(self):
        """Trades are processed in entry-time order regardless of input order."""
        trades = [
            ("ETHUSDT", 10, 20, 0.02),
            ("BTCUSDT", 0, 5, 0.01),  # earlier but listed second
        ]
        pnl = portfolio_sim(trades, 5)
        assert len(pnl) == 2
        # BTC should be first (entry=0)
        assert pnl[0] == 0.01
        assert pnl[1] == 0.02

    def test_straddle_syms_consumes_2_slots(self):
        """Straddle trades consume 2 slots, blocking a 3rd trade when max=2."""
        # Straddle on ETCUSDT takes 2 of 2 slots → SOLUSDT blocked
        trades = [
            ("ETCUSDT", 0, 100, 0.05),  # straddle: uses 2 slots
            ("SOLUSDT", 5, 50, 0.02),  # directional: needs 1 slot → blocked
        ]
        strd_syms = frozenset(["ETCUSDT"])
        pnl = portfolio_sim(trades, 2, straddle_syms=strd_syms)
        assert len(pnl) == 1
        assert pnl[0] == pytest.approx(0.05)

    def test_straddle_syms_allows_when_slots_available(self):
        """Straddle + directional both accepted when enough slots exist."""
        trades = [
            ("ETCUSDT", 0, 100, 0.05),  # straddle: uses 2 slots (of 3)
            ("SOLUSDT", 5, 50, 0.02),  # directional: uses 1 slot (of 3)
        ]
        strd_syms = frozenset(["ETCUSDT"])
        pnl = portfolio_sim(trades, 3, straddle_syms=strd_syms)
        assert len(pnl) == 2

    def test_straddle_syms_mixed_with_vr(self):
        """Mixed VR+STRD: straddle counts as 2, max_concurrent=10 → max 5 straddle."""
        # 5 straddle = 10 slots exactly, 6th straddle blocked
        strd_syms = frozenset([f"SYM{i}" for i in range(6)])
        trades = [(f"SYM{i}", i, i + 50, 0.01) for i in range(6)]
        pnl = portfolio_sim(trades, 10, straddle_syms=strd_syms)
        assert len(pnl) == 5  # 6th blocked: would need 12 slots

    def test_straddle_syms_none_backward_compat(self):
        """Without straddle_syms, original single-slot behaviour is preserved."""
        trades = [
            ("ETCUSDT", 0, 100, 0.05),
            ("SOLUSDT", 5, 50, 0.02),
            ("BTCUSDT", 10, 80, 0.03),
        ]
        # All fit in 3 slots (1 each), classic behaviour
        pnl = portfolio_sim(trades, 3)
        assert len(pnl) == 3


# ═══════════════════════════════════════════════════════════════════════════
#  compute_metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeMetrics:
    """Tests for performance metrics computation."""

    def test_empty_pnl(self):
        m = compute_metrics(np.array([]), 1.0)
        assert m["N"] == 0
        assert m["PF"] == 0.0

    def test_single_trade(self):
        m = compute_metrics(np.array([0.05]), 1.0)
        assert m["N"] == 1
        assert m["PF"] == 0.0  # N < 2 → early return

    def test_all_winners(self):
        pnl = np.array([0.01, 0.02, 0.03])
        m = compute_metrics(pnl, 1.0)
        assert m["WR"] == 100.0
        assert m["PF"] == 999.0  # no losses

    def test_all_losers(self):
        pnl = np.array([-0.01, -0.02, -0.03])
        m = compute_metrics(pnl, 1.0)
        assert m["WR"] == 0.0
        assert m["PF"] == 0.0  # no winners

    def test_known_pnl_series(self):
        """Verify metrics against hand-calculated values."""
        pnl = np.array([0.05, -0.01, 0.03, -0.01, 0.02])
        m = compute_metrics(pnl, 1.0)

        assert m["N"] == 5
        assert m["WR"] == pytest.approx(60.0)  # 3/5

        # PF = sum(wins) / |sum(losses)| = 0.10 / 0.02 = 5.0
        assert m["PF"] == pytest.approx(5.0, abs=0.01)

        # EV = mean(pnl) * 10000 = 0.016 * 10000 = 160 bp
        assert m["EV_bp"] == pytest.approx(160.0, abs=1.0)

        # Equity: (1.05)(0.99)(1.03)(0.99)(1.02) = 1.08127...
        expected_equity = 1.05 * 0.99 * 1.03 * 0.99 * 1.02
        expected_ret = (expected_equity - 1.0) * 100
        assert m["ret_pct"] == pytest.approx(expected_ret, abs=0.5)

    def test_tpd_calculation(self):
        """Trades per day = N / (years * 365)."""
        pnl = np.array([0.01] * 365)
        m = compute_metrics(pnl, 1.0)
        assert m["tpd"] == pytest.approx(1.0, abs=0.01)

    def test_max_dd(self):
        """Max drawdown is correctly computed."""
        # Sequence: +10%, -20%, +5%
        # Equity: 1.1, 0.88, 0.924
        # Peak: 1.1, DD at bar 2 = (1.1 - 0.88)/1.1 = 20%
        pnl = np.array([0.10, -0.20, 0.05])
        m = compute_metrics(pnl, 1.0)
        assert m["max_dd_pct"] == pytest.approx(20.0, abs=0.1)

    def test_annualized_return(self):
        """Annualized return for 2-year data."""
        # 100% total return over 2 years → ann = (2.0)^(1/2) - 1 = 41.4%
        pnl = np.array([1.0])  # +100% in one trade
        # N=1 → early return (N<2)
        # Use 3 trades instead
        pnl = np.array([0.30, 0.30, 0.30])  # each +30%
        m = compute_metrics(pnl, 2.0)
        expected_equity = 1.3 * 1.3 * 1.3  # = 2.197
        expected_ann = (expected_equity**0.5 - 1.0) * 100
        assert m["ann_ret_pct"] == pytest.approx(expected_ann, abs=0.5)


# ═══════════════════════════════════════════════════════════════════════════
#  pf_from_pnl
# ═══════════════════════════════════════════════════════════════════════════


class TestPfFromPnl:
    def test_empty(self):
        assert pf_from_pnl(np.array([])) == 0.0

    def test_all_positive(self):
        assert pf_from_pnl(np.array([0.01, 0.02])) == 999.0

    def test_all_negative(self):
        assert pf_from_pnl(np.array([-0.01, -0.02])) == 0.0

    def test_equal_wins_losses(self):
        pnl = np.array([0.10, -0.10])
        assert pf_from_pnl(pnl) == pytest.approx(1.0)

    def test_known_value(self):
        pnl = np.array([0.05, 0.03, -0.01, -0.02])
        # PF = 0.08 / 0.03 = 2.667
        assert pf_from_pnl(pnl) == pytest.approx(2.667, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
#  bootstrap_pf
# ═══════════════════════════════════════════════════════════════════════════


class TestBootstrapPf:
    def test_too_few_trades(self):
        result = bootstrap_pf(np.array([0.01] * 5))
        assert result["P_PF_ge_1"] == 0.0

    def test_output_keys(self):
        pnl = np.array([0.05, -0.01, 0.03, -0.01, 0.02] * 5)
        result = bootstrap_pf(pnl, n_boot=100)
        assert "P_PF_ge_1" in result
        assert "CI_lo" in result
        assert "CI_hi" in result
        assert "median" in result

    def test_deterministic(self):
        """Same seed → same result."""
        pnl = np.array([0.05, -0.01, 0.03, -0.02, 0.01] * 10)
        r1 = bootstrap_pf(pnl, n_boot=200)
        r2 = bootstrap_pf(pnl, n_boot=200)
        assert r1["median"] == r2["median"]


# ═══════════════════════════════════════════════════════════════════════════
#  collect_trades
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectTrades:
    """Tests for multi-TF trade collection with synthetic data."""

    def _make_symbol_data(self, n: int = 500):
        """Create minimal symbol data dict for testing."""
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 100.5, dtype=np.float64)
        low = np.full(n, 99.5, dtype=np.float64)
        # VR below threshold on a few bars to trigger entries
        vr = np.full(n, 0.8, dtype=np.float64)
        vr[250] = 0.30  # trigger
        mom = np.full(n, 0.001, dtype=np.float64)  # positive momentum
        return {
            "close": close,
            "high": high,
            "low": low,
            "volume": np.ones(n, dtype=np.float64),
            "vr_8_56": vr,
            "mom_8": mom,
            "bars": n,
        }

    def test_empty_data(self):
        cfg = {"sp": 8, "lp": 56, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4, "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05}
        trades = collect_trades(cfg, {}, {}, {}, "D_mom8")
        assert trades == []

    def test_15m_only_directional(self):
        """Collect trades from 15m data only (1h/4h empty)."""
        cfg = {"sp": 8, "lp": 56, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4, "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05}
        data_15m = {"BTCUSDT": self._make_symbol_data()}
        trades = collect_trades(cfg, data_15m, {}, {}, "D_mom8")
        # Should find at least 1 trade from the VR trigger at bar 250
        assert len(trades) >= 1
        # All trades should have symbol = "BTCUSDT"
        assert all(t[0] == "BTCUSDT" for t in trades)

    def test_no_overlap_per_symbol(self):
        """Per-symbol dedup prevents overlapping trades."""
        cfg = {"sp": 8, "lp": 56, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4, "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05}
        d = self._make_symbol_data(1000)
        # Multiple VR triggers close together
        d["vr_8_56"][250] = 0.30
        d["vr_8_56"][251] = 0.30
        d["vr_8_56"][252] = 0.30
        data_15m = {"BTCUSDT": d}
        trades = collect_trades(cfg, data_15m, {}, {}, "D_mom8")
        # Verify no overlapping intervals for same symbol
        btc_trades = [(t[1], t[2]) for t in trades if t[0] == "BTCUSDT"]
        for i in range(len(btc_trades)):
            for j in range(i + 1, len(btc_trades)):
                e1, x1 = btc_trades[i]
                e2, x2 = btc_trades[j]
                # Should not overlap
                assert not (e1 <= x2 and e2 <= x1), f"Overlap detected: ({e1},{x1}) vs ({e2},{x2})"

    def test_trade_tuple_format(self):
        """Each trade is (symbol, entry_bar, exit_bar, pnl)."""
        cfg = {"sp": 8, "lp": 56, "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4, "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05}
        data_15m = {"BTCUSDT": self._make_symbol_data()}
        trades = collect_trades(cfg, data_15m, {}, {}, "D_mom8")
        if trades:
            sym, entry, exit_b, pnl_val = trades[0]
            assert isinstance(sym, str)
            assert isinstance(entry, int)
            assert isinstance(exit_b, int)
            assert isinstance(pnl_val, float)
            assert entry < exit_b


# ═══════════════════════════════════════════════════════════════════════════
#  sim_directional_trailing: trailing stop for D_mom8 parity
# ═══════════════════════════════════════════════════════════════════════════


class TestSimDirectionalTrailing:
    """Tests for the Numba-compiled directional trailing stop simulator."""

    def test_trail_activates_and_exits(self):
        """Price rises past trail_activate, then retraces → trail exit."""
        n = 30
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)

        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1  # long

        # entry at bar 0, close=100
        # Bars 3-5: price rises to +5% with tight range (no trail trigger)
        # Peak from high = 5%, trail activates (>4%), trail level = 5%-0.5%=4.5%
        # low must stay above trail level: 100*(1+0.045) = 104.5
        for i in range(3, 6):
            close[i] = 105.0
            high[i] = 105.0
            low[i] = 104.6  # +4.6% > trail level 4.5% → holds

        # Bar 6: new peak at +6%, then retrace below trail level
        # Peak updates to 6%, trail level = 6%-0.5% = 5.5%
        # low at 104.0 = +4% which is < 5.5% → trail fires
        high[6] = 106.0
        close[6] = 104.0
        low[6] = 104.0

        pnl, ent, ext = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            sl=0.01,
            tp=0.20,
            max_hold=50,
            cost=0.0,
            trail_activate=0.04,
            trail_distance=0.005,
        )
        assert len(pnl) == 1
        assert ent[0] == 0
        assert ext[0] == 6  # trail exit on bar 6
        # PnL should be peak - distance = 6% - 0.5% = 5.5%
        assert pnl[0] == pytest.approx(0.055, abs=0.005)

    def test_no_trail_without_activation(self):
        """If price never reaches trail_activate, use normal SL/TP."""
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)

        sig = np.zeros(n, dtype=np.int8)
        sig[0] = 1
        dire = np.zeros(n, dtype=np.int8)
        dire[0] = 1  # long

        # Price rises only +2% (below trail_activate=4%), then hits SL
        high[3] = 102.0
        low[5] = 98.5  # -1.5% → SL at 1%

        pnl, ent, ext = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            sl=0.01,
            tp=0.20,
            max_hold=50,
            cost=0.0,
            trail_activate=0.04,
            trail_distance=0.005,
        )
        assert len(pnl) == 1
        # Should exit at SL, not trail
        assert pnl[0] == pytest.approx(-0.01, abs=1e-9)

    def test_collect_trades_d_mom8_uses_trailing(self):
        """collect_trades for D_mom8 now uses trailing stop (parity fix)."""
        n = 500
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 100.0, dtype=np.float64)
        low = np.full(n, 100.0, dtype=np.float64)
        vr = np.full(n, 0.8, dtype=np.float64)
        mom = np.full(n, 0.001, dtype=np.float64)

        # Trigger at bar 250 with price ramp that would activate trail
        vr[250] = 0.30
        # Ramp price up after entry to activate trail (>4%)
        for i in range(252, 260):
            close[i] = 105.0
            high[i] = 105.0
            low[i] = 105.0
        # Then retrace to trigger trail exit
        for i in range(260, 270):
            close[i] = 103.0
            high[i] = 103.5
            low[i] = 103.0

        data_15m = {
            "BTCUSDT": {
                "close": close,
                "high": high,
                "low": low,
                "volume": np.ones(n, dtype=np.float64),
                "vr_8_56": vr,
                "mom_8": mom,
                "bars": n,
            }
        }
        cfg = {
            "sp": 8,
            "lp": 56,
            "d_vr": 0.50,
            "s_vr": 0.55,
            "s_dur": 4,
            "d_sl": 0.01,
            "s_sl": 0.015,
            "tp": 0.20,
            "d_trail_activate": 0.04,
            "d_trail_distance": 0.005,
        }
        trades = collect_trades(cfg, data_15m, {}, {}, "D_mom8")
        assert len(trades) >= 1
        # With trail, PnL should be > SL (trail exit preserves gains)
        for _sym, _e, _x, pnl_val in trades:
            # Trail exit from +5% peak should give positive PnL, not -1% SL
            assert pnl_val > 0.0, f"Expected positive trail exit, got {pnl_val}"


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: sim → portfolio → metrics pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """End-to-end test: sim produces trades → portfolio filters → metrics."""

    def test_full_pipeline(self):
        """Synthetic but complete pipeline test."""
        # Create trades manually
        trades = [
            ("BTCUSDT", 0, 10, 0.05),  # win
            ("ETHUSDT", 5, 15, -0.01),  # loss (overlaps with BTC)
            ("BTCUSDT", 20, 30, -0.01),  # loss
            ("SOLUSDT", 25, 35, 0.03),  # win (overlaps with BTC#2)
            ("ETHUSDT", 40, 50, 0.02),  # win
        ]
        # With max_concurrent=2
        pnl = portfolio_sim(trades, 2)
        assert len(pnl) >= 3  # at least some accepted

        m = compute_metrics(pnl, 1.0)
        assert m["N"] == len(pnl)
        assert m["PF"] > 0
        assert 0 <= m["WR"] <= 100
        assert m["max_dd_pct"] >= 0

    def test_portfolio_sim_with_details_matches_sim(self):
        """portfolio_sim and portfolio_sim_with_details produce same PnL."""
        trades = [
            ("BTCUSDT", 0, 10, 0.05),
            ("ETHUSDT", 5, 15, -0.01),
            ("SOLUSDT", 2, 8, 0.03),
        ]
        pnl1 = portfolio_sim(trades, 2)
        pnl2, _ = portfolio_sim_with_details(trades, 2)
        np.testing.assert_array_equal(pnl1, pnl2)
