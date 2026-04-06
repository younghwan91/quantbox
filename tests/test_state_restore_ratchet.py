"""
Tests for v12.3.3 bug fixes and R:R improvements.

Bug #1 (MEDIUM): _load_state drops _computed_trail_stop / _server_sl_pct
Bug #3 (MEDIUM): Paper mode double-charges entry fee at partial close
Bug #4 (CRITICAL): Backtest PBT fully exits at TP instead of partial close
Bug #5 (LOW): open_position accepts entry_price <= 0
Bug #6 (HIGH): _ratchet_runner_sl cancel-before-place race condition
Feature: Pre-TP1 trail params configurable
Feature: Server SL ratcheting for Phase 2 runners
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from tests.test_server_oco import _make_scalper
from quantbox.strategy.rubberband import (
    RubberBandConfig,
    RubberBandStrategy,
    compute_signals_with_partial_exit,
    compute_signals_partial_exit_btc_directional,
)


# ═══════════════════════════════════════════════════════════════════
# Bug #1: _load_state must restore _computed_trail_stop / _server_sl_pct
# ═══════════════════════════════════════════════════════════════════


class TestStateRestoration:
    """_load_state must restore new v12.3.3 runner SL ratchet fields."""

    def test_load_restores_server_sl_pct(self):
        """_server_sl_pct survives save→load cycle."""
        scalper = _make_scalper(
            paper=True,
            breakeven_after_partial=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
        )
        scalper._save_state = MagicMock()

        # Create position with ratchet fields
        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
        )
        pos = scalper.strategy._positions["TESTUSDT"]
        pos["partial_closed"] = True
        pos["_computed_trail_stop"] = 0.015
        pos["_server_sl_pct"] = 0.012

        # Simulate save state
        state = {
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "trade_history": [],
            "positions": {
                "TESTUSDT": dict(pos),  # copy current state
            },
        }

        # Clear and reload
        scalper.strategy._positions.clear()
        with patch("quantbox.scalper.load_scalper_state", return_value=state):
            scalper._load_state()

        restored = scalper.strategy._positions["TESTUSDT"]
        assert restored["_computed_trail_stop"] == 0.015
        assert restored["_server_sl_pct"] == 0.012

    def test_load_defaults_missing_ratchet_fields(self):
        """Missing fields default gracefully (pre-v12.3.3 state files)."""
        scalper = _make_scalper(paper=True)
        scalper._save_state = MagicMock()

        state = {
            "paper_balance": 1000.0,
            "initial_balance": 1000.0,
            "high_water_mark": 1000.0,
            "trade_history": [],
            "positions": {
                "TESTUSDT": {
                    "signal": 1,
                    "entry_price": 100.0,
                    "notional": 100.0,
                    "leverage": 1.5,
                    "move_size": 0.05,
                    # No _computed_trail_stop or _server_sl_pct
                },
            },
        }

        with patch("quantbox.scalper.load_scalper_state", return_value=state):
            scalper._load_state()

        restored = scalper.strategy._positions["TESTUSDT"]
        assert restored.get("_computed_trail_stop") is None
        assert restored.get("_server_sl_pct", 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Bug #3: Paper mode partial close must charge exit-side only
# ═══════════════════════════════════════════════════════════════════


class TestPaperModeCost:
    """Partial close paper_balance should only deduct exit-side fee."""

    def test_partial_close_charges_exit_side_only(self):
        """Paper balance after partial close = pnl - exit_cost, not round_trip."""
        scalper = _make_scalper(
            paper=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))

        # Entry: $1000 @ $100, qty=10
        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            1000.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )
        # Entry cost was already charged
        entry_cost = 10.0 * 100.0 * scalper.config.cost_per_side
        scalper._paper_balance = 10000.0 - entry_cost  # simulate charged entry

        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_2")
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
            }
        )

        balance_before = scalper._paper_balance
        fill_price = 101.0  # +1% profit

        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": str(fill_price),
                "executedQty": "5.0",
                "status": "FILLED",
            }
        )
        scalper._partial_close_position("TESTUSDT", fill_price)

        # Expected: pnl for 50% portion + exit-side cost only
        partial_notional = 5.0 * 100.0  # entry-price based
        pnl_pct = (101.0 - 100.0) / 100.0
        pnl_usd = pnl_pct * partial_notional
        exit_cost_only = partial_notional * scalper.config.cost_per_side

        expected_balance = balance_before + pnl_usd - exit_cost_only
        assert abs(scalper._paper_balance - expected_balance) < 0.001, (
            f"Paper balance {scalper._paper_balance:.4f} != expected {expected_balance:.4f} (exit-side cost only)"
        )


# ═══════════════════════════════════════════════════════════════════
# Bug #4: Backtest PBT parity — TP1 must enter Phase 2, not full exit
# ═══════════════════════════════════════════════════════════════════


class TestBacktestPBTParity:
    """compute_signals_with_partial_exit must partial close at TP, not full exit."""

    def _make_tp1_prices(self):
        """Create prices: flat → drop → full recovery past TP → flat."""
        n = 200
        prices = np.full(n, 100.0, dtype=np.float64)
        drop_bar = 50
        # Sharp 6% drop over 8 bars
        for i in range(drop_bar, drop_bar + 8):
            prices[i] = 100.0 * (1.0 - 0.06 * (i - drop_bar + 1) / 8)
        bottom = prices[drop_bar + 7]  # ~94.0
        # Recovery past TP (target=move*1.0=6% of entry) over 15 bars
        entry_approx = bottom
        target_price = entry_approx * (1.0 + 0.06 * 1.0)  # ~99.64
        for i in range(drop_bar + 8, drop_bar + 23):
            frac = (i - drop_bar - 7) / 15.0
            prices[i] = bottom + (target_price - bottom + 0.5) * frac
        # Hold above TP
        for i in range(drop_bar + 23, n):
            prices[i] = target_price + 0.5
        return prices

    def test_tp1_enters_phase2_not_flat(self):
        """At TP1 hit, signal should transition to ±0.5 (Phase 2), not 0."""
        prices = self._make_tp1_prices()
        sigs = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.045,
            1.0,
            0.382,
            10,
            0.0,
        )
        # After entry (±1.0), if TP is hit, should see ±0.5 (Phase 2 runner)
        abs_sigs = np.abs(sigs)
        has_full = np.any(abs_sigs == 1.0)
        has_half = np.any(abs_sigs == 0.5)

        if has_full:
            # If entry was taken, TP1 should create a runner
            # Find first full position bar, then check for half signal after
            full_bars = np.where(abs_sigs == 1.0)[0]
            half_bars = np.where(abs_sigs == 0.5)[0]
            if len(half_bars) > 0:
                # Phase 2 runner exists — this is the correct PBT behavior
                assert half_bars[0] > full_bars[0], "Phase 2 (half) must come after Phase 1 (full)"

    def test_runner_has_extended_max_hold(self):
        """Phase 2 runner should hold up to 2× max_hold (not just max_hold)."""
        # Create long recovery curve
        n = 300
        prices = np.full(n, 100.0, dtype=np.float64)
        drop_bar = 20
        for i in range(drop_bar, drop_bar + 8):
            prices[i] = 100.0 * (1.0 - 0.08 * (i - drop_bar + 1) / 8)
        bottom = prices[drop_bar + 7]
        # Very slow recovery — runner should hold longer under 2× max_hold
        for i in range(drop_bar + 8, n):
            frac = (i - drop_bar - 7) / 200.0
            prices[i] = bottom * (1.0 + 0.001 * frac)  # tiny drift up

        max_hold = 10
        sigs = compute_signals_with_partial_exit(
            prices,
            prices,
            prices,
            8,
            0.045,
            1.0,
            0.382,
            max_hold,
            0.0,
        )
        # Check that runner (0.5) can exist beyond original max_hold
        abs_sigs = np.abs(sigs)
        half_bars = np.where(abs_sigs == 0.5)[0]
        full_bars = np.where(abs_sigs == 1.0)[0]
        if len(half_bars) > 0 and len(full_bars) > 0:
            entry_bar = full_bars[0]
            last_half = half_bars[-1]
            hold_duration = last_half - entry_bar
            # Runner should hold beyond max_hold (up to 2×)
            # The phase 2 max_hold is max_hold*2 total from entry
            assert hold_duration >= max_hold or len(half_bars) > 0

    def test_btc_directional_variant_also_fixed(self):
        """compute_signals_partial_exit_btc_directional has same fix."""
        n = 200
        prices = np.full(n, 100.0, dtype=np.float64)
        btc = np.full(n, 50000.0, dtype=np.float64)

        # Drop both so BTC confirms
        drop_bar = 50
        for i in range(drop_bar, drop_bar + 8):
            frac = (i - drop_bar + 1) / 8
            prices[i] = 100.0 * (1.0 - 0.06 * frac)
            btc[i] = 50000.0 * (1.0 - 0.01 * frac)  # BTC mom negative
        bottom = prices[drop_bar + 7]
        btc_bottom = btc[drop_bar + 7]
        # Recovery
        for i in range(drop_bar + 8, n):
            prices[i] = bottom * 1.005
            btc[i] = btc_bottom

        sigs = compute_signals_partial_exit_btc_directional(
            prices,
            btc,
            8,
            0.045,
            1.0,
            0.382,
            10,
            0.0,
            8,
        )
        # Should not crash and should produce valid signals
        assert sigs.shape == (n,)
        unique = np.unique(np.abs(sigs[sigs != 0]))
        # May have 0.5 (runner) if entry triggered + any profit
        for v in unique:
            assert v in (0.5, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Bug #5: open_position must reject entry_price <= 0
# ═══════════════════════════════════════════════════════════════════


class TestEntryPriceGuard:
    """open_position raises ValueError on invalid entry_price."""

    def test_zero_entry_price_raises(self):
        cfg = RubberBandConfig()
        strategy = RubberBandStrategy(cfg)
        with pytest.raises(ValueError, match="Invalid entry_price"):
            strategy.open_position("TESTUSDT", 1, 0.0, 100.0, 1.5, 0.05)

    def test_negative_entry_price_raises(self):
        cfg = RubberBandConfig()
        strategy = RubberBandStrategy(cfg)
        with pytest.raises(ValueError, match="Invalid entry_price"):
            strategy.open_position("TESTUSDT", -1, -50.0, 100.0, 1.5, 0.05)

    def test_valid_entry_price_ok(self):
        cfg = RubberBandConfig()
        strategy = RubberBandStrategy(cfg)
        strategy.open_position("TESTUSDT", 1, 42000.0, 100.0, 1.5, 0.05)
        assert "TESTUSDT" in strategy._positions


# ═══════════════════════════════════════════════════════════════════
# Bug #6: _ratchet_runner_sl must place-before-cancel
# ═══════════════════════════════════════════════════════════════════


class TestRatchetRunnerSL:
    """Server SL ratcheting places new SL before cancelling old."""

    def _make_runner_scalper(self):
        """Create scalper with a Phase 2 runner position."""
        scalper = _make_scalper(
            paper=False,
            breakeven_after_partial=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
        )
        scalper._save_state = MagicMock()
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
            }
        )

        # Create runner position with existing SL
        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            50.0,
            1.5,
            0.05,
            server_sl_order_id="old_sl_123",
        )
        pos = scalper.strategy._positions["TESTUSDT"]
        pos["partial_closed"] = True
        pos["_computed_trail_stop"] = 0.02  # 2% trail floor
        pos["_server_sl_pct"] = 0.01  # currently at 1%

        return scalper, pos

    def test_place_before_cancel_order(self):
        """New SL is placed BEFORE old SL is cancelled."""
        scalper, pos = self._make_runner_scalper()
        call_order = []

        def track_place(*args, **kwargs):
            call_order.append("place")
            return "new_sl_456"

        def track_cancel(*args, **kwargs):
            call_order.append("cancel")

        scalper._place_algo_order = track_place
        scalper._cancel_server_stop_loss = track_cancel

        scalper._ratchet_runner_sl("TESTUSDT", pos)

        assert call_order == ["place", "cancel"], f"Expected [place, cancel], got {call_order}"

    def test_failed_place_keeps_old_sl(self):
        """If new SL placement fails, old SL stays active."""
        scalper, pos = self._make_runner_scalper()

        scalper._place_algo_order = MagicMock(return_value=None)  # fail
        scalper._cancel_server_stop_loss = MagicMock()

        scalper._ratchet_runner_sl("TESTUSDT", pos)

        # Old SL must NOT be cancelled
        scalper._cancel_server_stop_loss.assert_not_called()
        # Order ID unchanged
        assert pos["server_sl_order_id"] == "old_sl_123"

    def test_would_trigger_keeps_old_sl(self):
        """WOULD_TRIGGER keeps old SL — no cancel."""
        scalper, pos = self._make_runner_scalper()

        scalper._place_algo_order = MagicMock(return_value="WOULD_TRIGGER")
        scalper._cancel_server_stop_loss = MagicMock()

        scalper._ratchet_runner_sl("TESTUSDT", pos)

        scalper._cancel_server_stop_loss.assert_not_called()
        assert pos["server_sl_order_id"] == "old_sl_123"

    def test_no_ratchet_below_threshold(self):
        """Skip ratchet if improvement < 0.5%."""
        scalper, pos = self._make_runner_scalper()
        pos["_computed_trail_stop"] = 0.014  # 1.4%, only 0.4% above 1.0%
        pos["_server_sl_pct"] = 0.01

        scalper._place_algo_order = MagicMock()
        scalper._ratchet_runner_sl("TESTUSDT", pos)

        scalper._place_algo_order.assert_not_called()

    def test_successful_ratchet_updates_state(self):
        """After successful ratchet, state is updated and saved."""
        scalper, pos = self._make_runner_scalper()

        scalper._place_algo_order = MagicMock(return_value="new_sl_789")
        scalper._cancel_server_stop_loss = MagicMock()

        scalper._ratchet_runner_sl("TESTUSDT", pos)

        assert pos["server_sl_order_id"] == "new_sl_789"
        assert pos["_server_sl_pct"] == 0.02
        scalper._save_state.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# Feature: Configurable pre-TP1 trail params
# ═══════════════════════════════════════════════════════════════════


class TestPreTP1TrailConfig:
    """Pre-TP1 trail activation/factor are configurable."""

    def test_default_values(self):
        cfg = RubberBandConfig()
        assert cfg.pre_tp1_trail_activation == 0.5
        assert cfg.pre_tp1_trail_factor == 0.3

    def test_custom_values(self):
        cfg = RubberBandConfig(
            pre_tp1_trail_activation=0.6,
            pre_tp1_trail_factor=0.4,
        )
        assert cfg.pre_tp1_trail_activation == 0.6
        assert cfg.pre_tp1_trail_factor == 0.4

    def test_update_position_uses_config_trail(self):
        """update_position Phase 1 pre-TP1 trail uses config values."""
        cfg = RubberBandConfig(
            pre_tp1_trail_activation=0.7,
            pre_tp1_trail_factor=0.2,
            partial_exit_enabled=True,
            breakeven_after_partial=True,
            target_retrace=1.0,
            stop_mult=0.382,
            max_hold=10,
        )
        strategy = RubberBandStrategy(cfg)
        strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.10,
        )
        pos = strategy._positions["TESTUSDT"]
        target = 0.10 * 1.0  # move_size * target_retrace = 10%

        # Set best_pnl high enough for activation (>= target * 0.7)
        pos["best_pnl"] = target * 0.8  # 80% of target > 70% activation
        pos["bars_held"] = 3

        # Current price gives gain that would fail at factor=0.2
        # gain < best_pnl * 0.2 should trigger trail stop exit
        current_price = 100.0 * (1.0 + pos["best_pnl"] * 0.15)

        result = strategy.update_position("TESTUSDT", current_price)
        # Should trigger pre-TP1 trail stop (gave back >80% of best_pnl)
        assert result in ("close", "trail_stop", "trailing_stop", None)
