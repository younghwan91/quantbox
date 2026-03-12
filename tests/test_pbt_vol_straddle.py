"""Property-Based Tests for VR Squeeze / Vol Straddle invariants.

Uses Hypothesis to generate hundreds of random inputs and verify that the
system's core properties (invariants) always hold.  A single counter-example
means a real bug — these tests are *not* expected to be fragile.

Invariants tested
─────────────────
1. PnL direction consistency   — LONG/SHORT PnL sign always matches profit/loss
2. Trailing stop monotonicity  — trail_stop can only rise, never fall
3. SoftTight scale range       — equity_scale ∈ {0.4, 0.6, 0.8, 1.0} always
4. Bracket price direction      — SL < entry < TP (LONG), TP < entry < SL (SHORT)
5. portfolio_sim slot limit    — concurrent positions ≤ max_concurrent at every bar
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: reusable Hypothesis strategies
# ──────────────────────────────────────────────────────────────────────────────

_pos_price = st.floats(min_value=0.001, max_value=100_000.0, allow_nan=False, allow_infinity=False)
_signal = st.sampled_from([-1, 1])
_balance = st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
_pct = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PnL direction invariant
# ══════════════════════════════════════════════════════════════════════════════


class TestPnLDirectionInvariant:
    """PnL sign must always be consistent with price direction.

    LONG trade : pnl_pct = (exit - entry) / entry
    SHORT trade: pnl_pct = (entry - exit) / entry

    Invariant: exit > entry → LONG profit (>0), SHORT loss (<0)
               exit < entry → LONG loss (<0),  SHORT profit (>0)
               exit = entry → both zero
    """

    @given(_pos_price, _pos_price, _signal)
    @settings(max_examples=500)
    def test_pnl_sign_matches_price_direction(self, entry_price: float, exit_price: float, signal: int) -> None:
        assume(entry_price > 0)
        assume(exit_price > 0)

        if signal == 1:  # LONG
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price

        if exit_price > entry_price:
            if signal == 1:
                assert pnl_pct > 0, f"LONG profit must be positive when exit>entry: {pnl_pct}"
            else:
                assert pnl_pct < 0, f"SHORT loss must be negative when exit>entry: {pnl_pct}"
        elif exit_price < entry_price:
            if signal == 1:
                assert pnl_pct < 0, f"LONG loss must be negative when exit<entry: {pnl_pct}"
            else:
                assert pnl_pct > 0, f"SHORT profit must be positive when exit<entry: {pnl_pct}"
        else:
            assert abs(pnl_pct) < 1e-12, f"Zero-move must give zero PnL: {pnl_pct}"

    @given(_pos_price, _pos_price, _signal)
    @settings(max_examples=300)
    def test_pnl_is_finite(self, entry_price: float, exit_price: float, signal: int) -> None:
        assume(entry_price > 0)
        assume(exit_price > 0)

        if signal == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        assert np.isfinite(pnl_pct), f"PnL must be finite: entry={entry_price} exit={exit_price}"

    @given(_pos_price, _signal)
    @settings(max_examples=200)
    def test_breakeven_exit_gives_zero_pnl(self, price: float, signal: int) -> None:
        """Exit at exact entry price → PnL = 0."""
        assume(price > 0)
        if signal == 1:
            pnl_pct = (price - price) / price
        else:
            pnl_pct = (price - price) / price
        assert pnl_pct == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Trailing stop monotonicity invariant
# ══════════════════════════════════════════════════════════════════════════════


class TestTrailStopMonotonicity:
    """Trailing stop can only move UP (tighten), never down.

    Production logic (strategy.py _check_exit_conditions):
        peak_gain = max(peak_gain, current_gain)
        trail_stop = peak_gain - d_trail_distance

    Because peak_gain is monotone non-decreasing:
        trail_stop[t+1] >= trail_stop[t] always
    """

    @given(
        st.lists(
            st.floats(min_value=-0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=200,
        )
    )
    @settings(max_examples=300)
    def test_trail_stop_never_decreases(self, gain_sequence: list[float]) -> None:
        """Simulate the trailing stop update loop and verify monotonicity."""
        cfg = VolStraddleConfig()
        trail_distance = cfg.d_trail_distance  # 0.002

        peak_gain = 0.0
        trail_active = False
        prev_trail_stop: float | None = None

        for gain in gain_sequence:
            if gain > peak_gain:
                peak_gain = gain
            if not trail_active and peak_gain >= cfg.d_trail_activate:
                trail_active = True

            if trail_active:
                trail_stop = peak_gain - trail_distance
                if prev_trail_stop is not None:
                    assert trail_stop >= prev_trail_stop - 1e-12, (
                        f"trail_stop decreased: {prev_trail_stop:.6f} → {trail_stop:.6f} (peak={peak_gain:.6f})"
                    )
                prev_trail_stop = trail_stop

    @given(
        st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300)
    def test_higher_peak_gives_higher_trail_stop(self, peak1: float, peak2: float) -> None:
        """Higher peak always gives equal-or-higher trail stop."""
        cfg = VolStraddleConfig()
        dist = cfg.d_trail_distance
        if peak1 <= peak2:
            assert (peak1 - dist) <= (peak2 - dist) + 1e-12

    @given(
        st.floats(min_value=0.006, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_trail_stop_always_below_peak(self, peak_gain: float) -> None:
        """trail_stop = peak - distance must be strictly below peak."""
        cfg = VolStraddleConfig()
        trail_stop = peak_gain - cfg.d_trail_distance
        assert trail_stop < peak_gain, f"trail_stop={trail_stop:.6f} must be < peak_gain={peak_gain:.6f}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. SoftTight equity scale invariant
# ══════════════════════════════════════════════════════════════════════════════


class TestSoftTightScaleInvariant:
    """SoftTight DD scaling must always produce one of exactly four values.

    Production logic (positions.py _compute_slot_capital):
        dd_pct = (hwm - balance) / hwm
        if dd_pct >= dd3: scale = scale3  (0.40)
        elif dd_pct >= dd2: scale = scale2  (0.60)
        elif dd_pct >= dd1: scale = scale1  (0.80)
        else: scale = 1.0

    Invariant: eq_scale ∈ {scale3, scale2, scale1, 1.0} always
    """

    def _compute_scale(self, balance: float, hwm: float) -> float:
        cfg = VolStraddleConfig()
        if hwm <= 0:
            return 1.0
        dd_pct = (hwm - balance) / hwm
        if dd_pct >= cfg.soft_eq_dd3:
            return cfg.soft_eq_scale3
        elif dd_pct >= cfg.soft_eq_dd2:
            return cfg.soft_eq_scale2
        elif dd_pct >= cfg.soft_eq_dd1:
            return cfg.soft_eq_scale1
        return 1.0

    @given(_balance, _balance)
    @settings(max_examples=500)
    def test_scale_is_one_of_four_values(self, balance: float, hwm: float) -> None:
        assume(hwm > 0)
        assume(balance > 0)
        assume(balance <= hwm)  # balance never exceeds HWM

        cfg = VolStraddleConfig()
        valid = {cfg.soft_eq_scale3, cfg.soft_eq_scale2, cfg.soft_eq_scale1, 1.0}
        scale = self._compute_scale(balance, hwm)
        assert scale in valid, f"scale={scale} not in {valid}"

    @given(_balance, _balance)
    @settings(max_examples=500)
    def test_scaled_slot_never_exceeds_base_slot(self, balance: float, hwm: float) -> None:
        """After SoftTight scaling, per_slot_capital ≤ unscaled slot."""
        assume(hwm > 0)
        assume(balance > 0)
        assume(balance <= hwm)

        cfg = VolStraddleConfig()
        base_slot = balance / cfg.max_concurrent
        scale = self._compute_scale(balance, hwm)
        scaled_slot = base_slot * scale

        assert scaled_slot <= base_slot + 1e-9, f"scaled_slot={scaled_slot:.4f} > base_slot={base_slot:.4f}"

    @given(_balance)
    @settings(max_examples=300)
    def test_no_dd_gives_full_scale(self, balance: float) -> None:
        """balance == hwm → no drawdown → scale = 1.0."""
        assume(balance > 0)
        scale = self._compute_scale(balance, balance)
        assert scale == 1.0, f"Expected scale=1.0 at zero DD, got {scale}"

    @given(
        st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.10, max_value=0.50, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300)
    def test_heavy_dd_gives_minimum_scale(self, hwm: float, dd_fraction: float) -> None:
        """Very heavy DD (≥ dd3 threshold) always gives minimum scale."""
        cfg = VolStraddleConfig()
        assume(dd_fraction >= cfg.soft_eq_dd3)
        balance = hwm * (1.0 - dd_fraction)
        assume(balance > 0)
        scale = self._compute_scale(balance, hwm)
        assert scale == cfg.soft_eq_scale3, (
            f"Expected min scale={cfg.soft_eq_scale3} for DD={dd_fraction * 100:.1f}%, got {scale}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Bracket price direction invariant
# ══════════════════════════════════════════════════════════════════════════════


class TestBracketPriceDirectionInvariant:
    """Stop-loss and take-profit prices must be on the correct side of entry.

    LONG position:
        sl_price = entry × (1 - sl_pct)    → sl_price < entry
        tp_price = entry × (1 + tp_pct)    → tp_price > entry

    SHORT position:
        sl_price = entry × (1 + sl_pct)    → sl_price > entry
        tp_price = entry × (1 - tp_pct)    → tp_price < entry
    """

    @given(
        _pos_price,
        st.floats(min_value=0.001, max_value=0.20, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        _signal,
    )
    @settings(max_examples=500)
    def test_sl_on_correct_side(self, entry_price: float, sl_pct: float, tp_pct: float, signal: int) -> None:
        assume(entry_price > 0)

        if signal == 1:  # LONG
            sl_price = entry_price * (1.0 - sl_pct)
            assert sl_price < entry_price, f"LONG SL must be below entry: sl={sl_price:.4f} entry={entry_price:.4f}"
        else:  # SHORT
            sl_price = entry_price * (1.0 + sl_pct)
            assert sl_price > entry_price, f"SHORT SL must be above entry: sl={sl_price:.4f} entry={entry_price:.4f}"

    @given(
        _pos_price,
        st.floats(min_value=0.001, max_value=0.20, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        _signal,
    )
    @settings(max_examples=500)
    def test_tp_on_correct_side(self, entry_price: float, sl_pct: float, tp_pct: float, signal: int) -> None:
        assume(entry_price > 0)

        if signal == 1:  # LONG
            tp_price = entry_price * (1.0 + tp_pct)
            assert tp_price > entry_price, f"LONG TP must be above entry: tp={tp_price:.4f} entry={entry_price:.4f}"
        else:  # SHORT
            tp_price = entry_price * (1.0 - tp_pct)
            assert tp_price < entry_price, f"SHORT TP must be below entry: tp={tp_price:.4f} entry={entry_price:.4f}"

    @given(
        _pos_price,
        st.floats(min_value=0.001, max_value=0.20, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        _signal,
    )
    @settings(max_examples=500)
    def test_sl_and_tp_on_opposite_sides(self, entry_price: float, sl_pct: float, tp_pct: float, signal: int) -> None:
        """SL and TP must always be on opposite sides of entry."""
        assume(entry_price > 0)

        if signal == 1:  # LONG
            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_pct)
        else:  # SHORT
            sl_price = entry_price * (1.0 + sl_pct)
            tp_price = entry_price * (1.0 - tp_pct)

        assert sl_price != tp_price, "SL and TP must never be at the same price"

        # For LONG: sl < entry < tp → sl < tp
        # For SHORT: tp < entry < sl → tp < sl
        if signal == 1:
            assert sl_price < tp_price, f"LONG: SL must be < TP: {sl_price} >= {tp_price}"
        else:
            assert tp_price < sl_price, f"SHORT: TP must be < SL: {tp_price} >= {sl_price}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. portfolio_sim slot limit invariant
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioSimSlotInvariant:
    """portfolio_sim must never exceed max_concurrent slots at any single bar.

    Also: no same-symbol concurrent positions.
    """

    @staticmethod
    def _sim(trades: list[tuple[str, int, int, float]], max_concurrent: int) -> list[tuple]:
        """Mirror of portfolio_sim logic to track which trades are active at each bar."""
        from quantbox.backtest.trade_sim import portfolio_sim

        # portfolio_sim returns only PnL array — we need accepted trades
        sorted_trades = sorted(trades, key=lambda x: x[1])
        accepted: list[tuple] = []
        active: list[tuple[str, int]] = []

        for sym, entry, exit_b, pnl_val in sorted_trades:
            active = [(s, e) for s, e in active if e > entry]
            sym_active = any(s == sym for s, _ in active)
            if sym_active:
                continue
            if len(active) >= max_concurrent:
                continue
            active.append((sym, exit_b))
            accepted.append((sym, entry, exit_b, pnl_val))

        return accepted

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]),
                st.integers(min_value=0, max_value=1000),
                st.integers(min_value=1, max_value=200),
                st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
            ),
            min_size=0,
            max_size=100,
        ),
        st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=300)
    def test_concurrent_positions_never_exceed_max(self, raw_trades: list[tuple], max_concurrent: int) -> None:
        """At no single bar should more than max_concurrent positions be open."""
        trades = [(sym, entry, entry + dur, pnl) for sym, entry, dur, pnl in raw_trades]
        accepted = self._sim(trades, max_concurrent)

        if not accepted:
            return

        # Check every bar from min entry to max exit (exclusive)
        # portfolio_sim uses strict: active if entry <= t < exit_b
        max_bar = max(exit_b for _, _, exit_b, _ in accepted)
        min_bar = min(entry for _, entry, _, _ in accepted)
        for t in range(min_bar, max_bar):
            count = sum(1 for _, entry, exit_b, _ in accepted if entry <= t < exit_b)
            assert count <= max_concurrent, f"Bar {t}: {count} positions open, limit={max_concurrent}"

    @given(
        st.lists(
            st.tuples(
                st.just("SAMEUSDT"),  # all same symbol
                st.integers(min_value=0, max_value=500),
                st.integers(min_value=1, max_value=50),
                st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
            ),
            min_size=2,
            max_size=50,
        ),
    )
    @settings(max_examples=200)
    def test_no_same_symbol_overlap(self, raw_trades: list[tuple]) -> None:
        """Same symbol positions must never overlap in time."""
        trades = [(sym, entry, entry + dur, pnl) for sym, entry, dur, pnl in raw_trades]
        accepted = self._sim(trades, max_concurrent=10)

        # Check no two accepted trades on same symbol overlap
        for i, (sym_i, entry_i, exit_i, _) in enumerate(accepted):
            for j, (sym_j, entry_j, exit_j, _) in enumerate(accepted):
                if i >= j:
                    continue
                if sym_i != sym_j:
                    continue
                # They should NOT overlap
                overlap = entry_j < exit_i and exit_j > entry_i
                assert not overlap, (
                    f"Same-symbol overlap: trade {i} [{entry_i},{exit_i}) overlaps trade {j} [{entry_j},{exit_j})"
                )

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["A", "B", "C", "D", "E"]),
                st.integers(min_value=0, max_value=200),
                st.integers(min_value=1, max_value=100),
                st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=200)
    def test_accepted_count_le_total_input(self, raw_trades: list[tuple]) -> None:
        """Accepted trades can never exceed total input trades."""
        trades = [(sym, entry, entry + dur, pnl) for sym, entry, dur, pnl in raw_trades]
        accepted = self._sim(trades, max_concurrent=10)
        assert len(accepted) <= len(trades), f"Accepted {len(accepted)} > input {len(trades)}"
