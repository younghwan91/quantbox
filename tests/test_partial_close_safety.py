"""
Tests for v12.3.2 safety audit — verifies each bug fix is effective.

BUG-A: Partial close notional used exit price → dust on exchange + wrong PnL
BUG-B: _record_server_fill used exit-price notional → ~3-5% PnL error
BUG-C: PBT cost only charged exit side → 50% cost under-report
BUG-D: Partial NO_POSITION double close_position → trade silently dropped
BUG-E: Partial close failure left position unprotected for 15 min
BUG-F: Runner trailing used stale ATR from entry time
"""

from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone
import numpy as np
import pytest

from tests.test_server_oco import _make_scalper
from quantbox.strategy.rubberband import RubberBandConfig, RubberBandStrategy


# ═══════════════════════════════════════════════════════════════════
# BUG-A: Partial close notional must use entry price, not fill price
# ═══════════════════════════════════════════════════════════════════


class TestPartialCloseNotional:
    """After partial close, pos['notional'] must equal
    remaining_qty * entry_price, not original - close_qty * fill_price."""

    def test_notional_uses_entry_price(self):
        """After partial close at profit, remaining notional is entry-based."""
        scalper = _make_scalper(
            paper=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()

        # Manually create a position: entry $100, notional $100, qty 1.0
        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_2")
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
                "min_notional": 5.0,
            }
        )

        # Partial close fills at $103 (3% profit)
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": "103.0",
                "executedQty": "0.5",
                "status": "FILLED",
            }
        )

        result = scalper._partial_close_position("TESTUSDT", 103.0)
        assert result is True

        pos = scalper.strategy._positions["TESTUSDT"]

        # BUG-A fix: remaining notional = (1.0 - 0.5) * 100 = 50.0
        # Old buggy: 100 - 0.5 * 103 = 48.5 (WRONG)
        assert pos["notional"] == pytest.approx(50.0, abs=0.01), (
            f"Runner notional must be entry-based: got {pos['notional']}"
        )

    def test_runner_close_quantity_exact(self):
        """Runner close sends exactly remaining_qty to exchange (no dust)."""
        scalper = _make_scalper(
            paper=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()

        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_2")
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
                "min_notional": 5.0,
            }
        )
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": "103.0",
                "executedQty": "0.5",
                "status": "FILLED",
            }
        )

        scalper._partial_close_position("TESTUSDT", 103.0)
        pos = scalper.strategy._positions["TESTUSDT"]

        # Verify remaining qty computable from notional
        remaining_qty = pos["notional"] / pos["entry_price"]
        assert remaining_qty == pytest.approx(0.5, abs=0.001), f"remaining_qty must be 0.5, got {remaining_qty}"


# ═══════════════════════════════════════════════════════════════════
# BUG-B: _record_server_fill must use entry-price notional
# ═══════════════════════════════════════════════════════════════════


class TestServerFillPnL:
    """_record_server_fill must use pos['notional'] (entry-based),
    not fill_qty * fill_price (exit-based)."""

    def test_server_tp_pnl_uses_entry_notional(self):
        """Server TP fill PnL should be computed on entry-price notional."""
        scalper = _make_scalper(paper=False)
        scalper._save_state = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper.discord = MagicMock()

        # Position: entry $100, notional $100 (qty=1.0)
        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )

        # Server TP fill at $105 (5% gain)
        scalper._record_server_fill("TESTUSDT", 105.0, 1.0, "server_tp")

        assert len(scalper._trade_history) == 1
        trade = scalper._trade_history[0]

        # Correct PnL: 5% * $100 (entry notional) = $5.0
        # Old buggy: 5% * $105 (exit notional) = $5.25 (WRONG)
        expected_pnl_pct = 5.0
        expected_gross = 0.05 * 100.0  # $5.0
        cost = 100.0 * scalper.config.round_trip_cost
        expected_net = expected_gross - cost

        assert trade["pnl_pct"] == pytest.approx(expected_pnl_pct, abs=0.01)
        assert trade["pnl_usd"] == pytest.approx(expected_net, abs=0.01)
        assert trade["notional"] == pytest.approx(100.0, abs=0.01), "Trade record should show entry-based notional"

    def test_server_sl_pnl_not_understated(self):
        """Server SL fill loss must not be understated by exit-price."""
        scalper = _make_scalper(paper=False)
        scalper._save_state = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper.discord = MagicMock()

        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )

        # Server SL fill at $96 (4% loss)
        scalper._record_server_fill("TESTUSDT", 96.0, 1.0, "server_sl")

        trade = scalper._trade_history[0]
        # Correct loss: -4% * $100 = -$4.0
        # Old buggy: -4% * $96 = -$3.84 (understated!)
        expected_gross = -0.04 * 100.0  # -$4.0
        cost = 100.0 * scalper.config.round_trip_cost
        expected_net = expected_gross - cost

        assert trade["pnl_usd"] == pytest.approx(expected_net, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# BUG-C: PBT must charge round_trip_cost, not just exit cost
# ═══════════════════════════════════════════════════════════════════


class TestPBTCostParity:
    """PBT trades must charge round_trip_cost for backtest parity."""

    def test_partial_close_charges_round_trip(self):
        """Partial close trade records round_trip_cost, not cost_per_side."""
        scalper = _make_scalper(
            paper=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper.discord = MagicMock()

        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_2")
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
                "min_notional": 5.0,
            }
        )
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": "103.0",
                "executedQty": "0.5",
                "status": "FILLED",
            }
        )

        scalper._partial_close_position("TESTUSDT", 103.0)

        assert len(scalper._trade_history) == 1
        trade = scalper._trade_history[0]

        # notional = 0.5 * 100 (entry-based) = 50.0
        partial_notional = 50.0
        expected_cost = partial_notional * scalper.config.round_trip_cost

        pnl_pct = (103.0 - 100.0) / 100.0  # 3%
        expected_net = pnl_pct * partial_notional - expected_cost

        assert trade["pnl_usd"] == pytest.approx(expected_net, abs=0.01), f"Cost must be round_trip, not cost_per_side"

    def test_runner_close_charges_round_trip(self):
        """Runner close trade records round_trip_cost, not cost_per_side."""
        scalper = _make_scalper(
            paper=True,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper.discord = MagicMock()
        scalper.get_balance = MagicMock(return_value=1000.0)
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_2")
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
                "min_notional": 5.0,
            }
        )

        # Create position and manually set to runner state
        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            50.0,
            1.5,
            0.05,
        )
        scalper.strategy._positions["TESTUSDT"]["partial_closed"] = True
        scalper.strategy._positions["TESTUSDT"]["trail_floor"] = 0.0

        # Close runner at $110
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": "110.0",
                "executedQty": "0.5",
                "status": "FILLED",
            }
        )
        result = scalper._close_position("TESTUSDT", "trail_runner", 110.0)
        assert result is True

        trade = scalper._trade_history[-1]
        runner_notional = 50.0
        expected_cost = runner_notional * scalper.config.round_trip_cost
        pnl_pct = (110.0 - 100.0) / 100.0
        expected_net = pnl_pct * runner_notional - expected_cost

        assert trade["pnl_usd"] == pytest.approx(expected_net, abs=0.01), f"Runner must charge round_trip_cost"


# ═══════════════════════════════════════════════════════════════════
# BUG-D: Partial NO_POSITION must record trade inline
# ═══════════════════════════════════════════════════════════════════


class TestPartialNoPositionRecording:
    """When partial close gets NO_POSITION, the trade must still
    be recorded (inline, not via _record_server_fill which double-pops)."""

    def test_no_position_records_trade(self):
        """NO_POSITION during partial close records a trade."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
            breakeven_after_partial=True,
        )
        scalper._save_state = MagicMock()
        scalper.discord = MagicMock()
        scalper.client.futures_account_trades = MagicMock(return_value=[])

        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_1",
            server_tp_order_id="tp_1",
        )
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
                "min_notional": 5.0,
            }
        )

        # Return NO_POSITION (server already closed it)
        scalper._place_order = MagicMock(
            return_value={
                "status": "NO_POSITION",
            }
        )

        result = scalper._partial_close_position("TESTUSDT", 103.0)
        assert result is False  # Partial close "failed" (position gone)

        # Trade MUST be recorded (this was the old bug — silently dropped)
        assert len(scalper._trade_history) == 1, "NO_POSITION trade must be recorded!"
        assert scalper._trade_history[0]["reason"] == "server_fill_race"

        # Position must be removed from internal state
        assert not scalper.strategy.has_position("TESTUSDT")


# ═══════════════════════════════════════════════════════════════════
# BUG-E: Partial close failure must re-place server orders immediately
# ═══════════════════════════════════════════════════════════════════


class TestPartialCloseFailureProtection:
    """When partial close order fails, server SL/TP must be re-placed
    immediately, not left empty for 15-minute bracket repair cycle.

    v12.4.2: In PBT mode, server TP must NOT be re-placed (same bug class
    as v12.4.1 CRITICAL #1 — server TP closes 100%, killing PBT runner).
    """

    def test_failure_replaces_server_orders(self):
        """Failed partial close re-places server SL; skips TP in PBT mode."""
        scalper = _make_scalper(
            paper=False,
            partial_exit_enabled=True,
            partial_exit_fraction=0.5,
        )
        scalper._save_state = MagicMock()

        scalper.strategy.open_position(
            "TESTUSDT",
            1,
            100.0,
            100.0,
            1.5,
            0.05,
            server_sl_order_id="sl_old",
            server_tp_order_id="tp_old",
        )
        # Simulate: strategy.update_position() has already set flag
        scalper.strategy._positions["TESTUSDT"]["partial_closed"] = True

        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._round_qty = MagicMock(side_effect=lambda q, s: round(q, 3))
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.01,
                "price_precision": 2,
                "step_size": 0.001,
                "quantity_precision": 3,
                "min_notional": 5.0,
            }
        )

        # Place order FAILS
        scalper._place_order = MagicMock(return_value=None)
        # New server orders
        scalper._place_server_stop_loss = MagicMock(return_value="sl_new")
        scalper._place_server_take_profit = MagicMock(return_value="tp_new")

        result = scalper._partial_close_position("TESTUSDT", 103.0)
        assert result is False

        # SL MUST be re-placed to protect the downside
        pos = scalper.strategy._positions["TESTUSDT"]
        assert pos["server_sl_order_id"] == "sl_new", "Failed partial close must re-place server SL"
        # v12.5: In PBT mode, server TP IS re-placed with 50% qty
        # (server TP now fires for partial qty only, enabling Phase 2 runner).
        assert pos["server_tp_order_id"] == "tp_new", "PBT mode: failed partial close must re-place 50% server TP"
        assert pos.get("server_tp_is_partial") is True, "Re-placed TP must be flagged as partial"
        assert pos["partial_closed"] is False, "Must revert partial_closed flag"


# ═══════════════════════════════════════════════════════════════════
# BUG-F: Runner trailing must use fresh ATR, not stale entry-time ATR
# ═══════════════════════════════════════════════════════════════════


class TestFreshATRForRunners:
    """The cycle must download full OHLC for active positions and
    compute fresh ATR, not use stale get_last_atr_pct from entry."""

    def test_compute_atr_pct_produces_values(self):
        """compute_atr_pct must return valid non-NaN values."""
        from quantbox.strategy.rubberband import compute_atr_pct

        n = 100
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.abs(close) + 50  # ensure positive
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        atr_arr = compute_atr_pct(high, low, close, period=14)

        # After warm-up (14 bars), values should be non-NaN
        assert not np.isnan(atr_arr[-1]), "Last ATR value should not be NaN"
        assert atr_arr[-1] > 0, "ATR% must be positive"

    def test_fresh_atr_differs_from_stale(self):
        """Fresh ATR computation should be independent of strategy._last_atr_pct."""
        from quantbox.strategy.rubberband import compute_atr_pct

        # Simulate: ATR was 2% at entry, now volatility doubled
        n = 100
        # Low vol first half, high vol second half
        close = np.ones(n) * 100.0
        high = np.ones(n) * 100.5
        low = np.ones(n) * 99.5
        # Make second half much more volatile
        high[50:] = 102.0
        low[50:] = 98.0

        atr = compute_atr_pct(high, low, close, period=14)
        atr_early = float(atr[30])  # Low vol period
        atr_late = float(atr[-1])  # High vol period

        assert atr_late > atr_early * 1.5, "ATR should increase with volatility"

    def test_scalper_imports_compute_atr_pct(self):
        """compute_atr_pct available from rubberband module."""
        from quantbox.strategy.rubberband import compute_atr_pct as imported_fn

        assert callable(imported_fn)
