"""Tests for 7 critical path safety guards (audit fixes).

Each test verifies one specific guard added to prevent financial loss
from config corruption, state file corruption, or edge-case inputs.

Fix IDs: F2, F3, A1, A2, B3, C1, D1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _make_rb_scalper(paper: bool = False, **config_overrides):
    """Create a minimal RubberBandScalper for testing (reuse existing factory)."""
    from tests.test_server_oco import _make_scalper

    return _make_scalper(paper=paper, **config_overrides)


def _inject_rb_pos(scalper, sym: str = "BTCUSDT", signal: int = 1, entry: float = 50000.0):
    """Inject a basic position into RubberBandScalper."""
    scalper.strategy.open_position(
        symbol=sym,
        signal=signal,
        entry_price=entry,
        notional=300.0,
        leverage=3,
        move_size=0.03,
        server_sl_order_id="sl_001",
        server_tp_order_id="tp_001",
    )


# ─────────────────────────────────────────────
# F2: SHORT force-close uses askPrice (not bidPrice)
# ─────────────────────────────────────────────


class TestF2ShortForceClosePrice:
    """F2: WOULD_TRIGGER force-close must use askPrice for SHORT, bidPrice for LONG."""

    def _setup_would_trigger(self, signal: int):
        """Set up scalper with a WOULD_TRIGGER SL scenario."""
        scalper = _make_rb_scalper(paper=False)
        sym = "BTCUSDT"
        _inject_rb_pos(scalper, sym=sym, signal=signal)
        pos = scalper.strategy._positions[sym]
        pos["server_sl_order_id"] = None  # missing SL → triggers bracket repair

        bid_price = 49000.0
        ask_price = 49050.0
        scalper.client.futures_orderbook_ticker.return_value = {
            "bidPrice": str(bid_price),
            "askPrice": str(ask_price),
        }
        return scalper, sym, bid_price, ask_price

    def test_long_force_close_uses_bid_price(self):
        """LONG position WOULD_TRIGGER → force-close at bidPrice."""
        scalper, sym, bid_price, ask_price = self._setup_would_trigger(signal=1)

        closed_prices = []
        scalper._close_position = lambda s, r, p: closed_prices.append(p) or True
        scalper._place_server_stop_loss = MagicMock(return_value="WOULD_TRIGGER")
        scalper._sync_server_orders()

        assert closed_prices, "Force-close should have been called for LONG"
        assert closed_prices[0] == bid_price, f"LONG should close at bidPrice={bid_price}, got {closed_prices[0]}"

    def test_short_force_close_uses_ask_price(self):
        """SHORT position WOULD_TRIGGER → force-close at askPrice (not bidPrice)."""
        scalper, sym, bid_price, ask_price = self._setup_would_trigger(signal=-1)

        closed_prices = []
        scalper._close_position = lambda s, r, p: closed_prices.append(p) or True
        scalper._place_server_stop_loss = MagicMock(return_value="WOULD_TRIGGER")
        scalper._sync_server_orders()

        assert closed_prices, "Force-close should have been called for SHORT"
        assert closed_prices[0] == ask_price, f"SHORT should close at askPrice={ask_price}, got {closed_prices[0]}"


# ─────────────────────────────────────────────
# F3: bracket repair skipped when qty=0
# ─────────────────────────────────────────────


class TestF3BracketRepairQtyZero:
    """F3: If position quantity rounds to 0, bracket repair must skip (not call API)."""

    def test_zero_qty_skips_repair_and_alerts(self):
        """Tiny notional → rounded qty=0 → repair skipped, Discord alert sent."""
        scalper = _make_rb_scalper(paper=False)
        sym = "BTCUSDT"
        _inject_rb_pos(scalper, sym=sym)

        pos = scalper.strategy._positions[sym]
        pos["notional"] = 0.0000001  # rounds to qty=0
        pos["entry_price"] = 50000.0
        pos["server_sl_order_id"] = None  # needs SL repair

        discord_messages = []
        scalper.discord.send = lambda content=None, **kw: discord_messages.append(content or "")

        # Track algo API POST calls
        post_calls = []

        def mock_api(method, path, signed=True, data=None):
            if method == "post":
                post_calls.append(data)
            return {"algoStatus": "NEW"}

        scalper.client._request_futures_api = MagicMock(side_effect=mock_api)

        scalper._sync_server_orders()

        # Should alert via Discord about skipping
        assert any("qty=0" in m or "skip" in m.lower() for m in discord_messages), (
            f"Expected Discord alert about qty=0 skip, got: {discord_messages}"
        )
        # No algo order should be placed
        assert len(post_calls) == 0, "No algo order should be placed when qty=0"


# ─────────────────────────────────────────────
# A1: trade_history validation on restore
# ─────────────────────────────────────────────


class TestA1TradeHistoryValidation:
    """A1: Corrupt trade_history records must be dropped on restore, not crash."""

    def _load_with_history(self, scalper, records: list, balance: float = 1000.0):
        """Inject state with trade_history (include one position to avoid no-pos reset)."""
        import quantbox.scalper as scalper_mod

        fake_positions = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "signal": 1,
                "entry_price": 50000.0,
                "notional": 300.0,
                "leverage": 3,
                "move_size": 0.03,
                "sl_pct": 0.01,
                "tp_pct": 0.05,
                "bars_held": 2,
                "max_hold": 192,
                "sl": 0.01,
                "tp": 0.05,
                "trail_activate": 0.005,
                "trail_distance": 0.002,
                "trail_active": False,
                "peak_gain": 0.0,
                "server_sl_order_id": "sl_1",
                "server_tp_order_id": "tp_1",
                "partial_closed": False,
            }
        }
        fake_state = {
            "saved_at": "2024-01-01",
            "paper_balance": balance,
            "initial_balance": balance,
            "high_water_mark": balance,
            "trade_history": records,
            "positions": fake_positions,
        }
        scalper.get_balance = MagicMock(return_value=balance)
        with patch.object(scalper_mod, "load_scalper_state", return_value=fake_state):
            scalper._load_state()

    def test_valid_records_kept(self):
        """Valid pnl_usd and pnl_pct are preserved."""
        scalper = _make_rb_scalper(paper=True)
        records = [
            {"pnl_usd": 10.0, "pnl_pct": 0.01, "symbol": "BTCUSDT"},
            {"pnl_usd": -5.0, "pnl_pct": -0.005, "symbol": "ETHUSDT"},
        ]
        self._load_with_history(scalper, records)
        assert len(scalper._trade_history) == 2

    def test_corrupt_record_dropped(self):
        """Records with non-numeric pnl fields are silently dropped."""
        scalper = _make_rb_scalper(paper=True)
        records = [
            {"pnl_usd": "NaN_string", "pnl_pct": None, "symbol": "BTCUSDT"},
            {"pnl_usd": 10.0, "pnl_pct": 0.01, "symbol": "ETHUSDT"},
        ]
        self._load_with_history(scalper, records)
        assert len(scalper._trade_history) == 1
        assert scalper._trade_history[0]["symbol"] == "ETHUSDT"

    def test_non_dict_record_dropped(self):
        """Non-dict entries in trade_history are dropped."""
        scalper = _make_rb_scalper(paper=True)
        records = [
            "bad_string_record",
            {"pnl_usd": 5.0, "pnl_pct": 0.005, "symbol": "BTCUSDT"},
        ]
        self._load_with_history(scalper, records)
        assert len(scalper._trade_history) == 1

    def test_all_corrupt_gives_empty_history(self):
        """All corrupt records → empty history (not crash)."""
        scalper = _make_rb_scalper(paper=True)
        records = [
            {"pnl_usd": None, "pnl_pct": None},
            {"pnl_usd": "bad", "pnl_pct": "bad"},
        ]
        self._load_with_history(scalper, records)
        assert scalper._trade_history == []


# ─────────────────────────────────────────────
# A2: HWM/initial_balance negative guard
# ─────────────────────────────────────────────


class TestA2HWMNegativeGuard:
    """A2: Restored HWM or initial_balance ≤ 0 must be reset to current balance."""

    def _load_with_hwm(self, scalper, hwm: float, initial: float, balance: float = 1000.0):
        import quantbox.scalper as scalper_mod

        fake_positions = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "signal": 1,
                "entry_price": 50000.0,
                "notional": 300.0,
                "leverage": 3,
                "move_size": 0.03,
                "sl": 0.01,
                "tp": 0.05,
                "trail_activate": 0.005,
                "trail_distance": 0.002,
                "trail_active": False,
                "peak_gain": 0.0,
                "bars_held": 2,
                "max_hold": 192,
                "server_sl_order_id": "sl_1",
                "server_tp_order_id": "tp_1",
                "partial_closed": False,
            }
        }
        fake_state = {
            "saved_at": "2024-01-01",
            "paper_balance": balance,
            "initial_balance": initial,
            "high_water_mark": hwm,
            "trade_history": [],
            "positions": fake_positions,
        }
        scalper.get_balance = MagicMock(return_value=balance)
        with patch.object(scalper_mod, "load_scalper_state", return_value=fake_state):
            scalper._load_state()

    def test_negative_hwm_reset_to_balance(self):
        """HWM=-999 → reset to current balance (1000)."""
        scalper = _make_rb_scalper(paper=True)
        self._load_with_hwm(scalper, hwm=-999.0, initial=1000.0)
        assert scalper._high_water_mark == 1000.0, f"Expected HWM=1000.0 after reset, got {scalper._high_water_mark}"

    def test_zero_hwm_reset_to_balance(self):
        """HWM=0 → reset to current balance."""
        scalper = _make_rb_scalper(paper=True)
        self._load_with_hwm(scalper, hwm=0.0, initial=1000.0)
        assert scalper._high_water_mark == 1000.0

    def test_negative_initial_balance_reset(self):
        """initial_balance=-1 → reset to current balance."""
        scalper = _make_rb_scalper(paper=True)
        self._load_with_hwm(scalper, hwm=1000.0, initial=-1.0)
        assert scalper._initial_balance == 1000.0

    def test_valid_hwm_preserved(self):
        """Valid HWM=1200 with balance=1000 → preserved for DD tracking."""
        scalper = _make_rb_scalper(paper=True)
        self._load_with_hwm(scalper, hwm=1200.0, initial=1000.0)
        assert scalper._high_water_mark == 1200.0


# ─────────────────────────────────────────────
# B3: sl_pct=0 does NOT trigger SL exit
# ─────────────────────────────────────────────


class TestB3SlPctZeroGuard:
    """B3: sl_pct=0 must not trigger SL exit on any loss."""

    def test_sl_pct_zero_no_exit(self):
        """sl_pct=0 → SL check skipped even if position is deeply in loss."""
        from tests.conftest import _build_vs_scalper

        scalper = _build_vs_scalper(balance=1000.0)
        strategy = scalper.strategy
        sym = "BTCUSDT:LONG"

        strategy._positions[sym] = {
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "signal": 1,
            "entry_price": 50000.0,
            "notional": 300.0,
            "leverage": 30,
            "move_size": 0.03,
            "sl": 0.0,  # ← corrupt config: sl_pct=0
            "tp": 0.12,
            "trail_activate": 0.005,
            "trail_distance": 0.002,
            "trail_active": False,
            "peak_gain": 0.0,
            "bars_held": 1,
            "max_hold": 192,
            "server_sl_order_id": "sl_1",
            "server_tp_order_id": "tp_1",
            "partial_closed": False,
            "entry": 50000.0,
        }

        # -5% loss — would trigger a 1% SL if guard is absent
        result = strategy.update_position(
            symbol=sym,
            current_price=47500.0,
            current_high=50100.0,
            current_low=47500.0,
        )
        assert result != "stop_loss", "sl_pct=0 should NOT trigger SL exit (B3 guard violation)"

    def test_sl_pct_positive_exits(self):
        """sl_pct=0.01 triggers SL when loss exceeds it (guard should not block)."""
        from tests.conftest import _build_vs_scalper

        scalper = _build_vs_scalper(balance=1000.0)
        strategy = scalper.strategy
        sym = "BTCUSDT:LONG"

        strategy._positions[sym] = {
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "signal": 1,
            "entry_price": 50000.0,
            "notional": 300.0,
            "leverage": 30,
            "move_size": 0.03,
            "sl": 0.01,  # 1% SL
            "tp": 0.12,
            "trail_activate": 0.005,
            "trail_distance": 0.002,
            "trail_active": False,
            "peak_gain": 0.0,
            "bars_held": 1,
            "max_hold": 192,
            "server_sl_order_id": "sl_1",
            "server_tp_order_id": "tp_1",
            "partial_closed": False,
            "entry": 50000.0,
        }

        # -2% loss → should hit 1% SL
        result = strategy.update_position(
            symbol=sym,
            current_price=49000.0,
            current_high=50100.0,
            current_low=49000.0,
        )
        assert result == "stop_loss", "sl_pct=0.01 should trigger SL at -2% loss"


# ─────────────────────────────────────────────
# C1: stop_mult negative guard in _place_server_stop_loss
# ─────────────────────────────────────────────


class TestC1StopMultNegativeGuard:
    """C1: stop_mult < 0 must be rejected before placing server SL."""

    def test_negative_stop_mult_returns_none(self):
        """stop_mult=-0.5 → abort SL placement (inverted SL direction)."""
        scalper = _make_rb_scalper(paper=False)
        result = scalper._place_server_stop_loss(
            symbol="BTCUSDT",
            signal=1,
            quantity=0.01,
            entry_price=50000.0,
            move_size=0.03,
            stop_mult=-0.5,
        )
        assert result is None, "Negative stop_mult should return None (C1 guard)"

    def test_positive_stop_mult_proceeds(self):
        """stop_mult=1.0 → normal flow, not blocked by C1."""
        scalper = _make_rb_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {"algoId": "sl_normal"}
        scalper.get_symbol_info = MagicMock(
            return_value={
                "tick_size": 0.1,
                "price_precision": 1,
                "qty_precision": 3,
                "step_size": 0.001,
            }
        )
        result = scalper._place_server_stop_loss(
            symbol="BTCUSDT",
            signal=1,
            quantity=0.01,
            entry_price=50000.0,
            move_size=0.03,
            stop_mult=1.0,
        )
        assert result is not None, "Positive stop_mult should proceed past C1 guard"


# ─────────────────────────────────────────────
# D1: target_retrace guard in _place_server_take_profit
# ─────────────────────────────────────────────


class TestD1TargetRetraceGuard:
    """D1: target_retrace ≤ 0 or causing negative SHORT price must abort TP."""

    def _sym_info(self):
        return {
            "tick_size": 0.1,
            "price_precision": 1,
            "qty_precision": 3,
            "step_size": 0.001,
        }

    def test_negative_target_retrace_returns_none(self):
        """target_retrace=-0.5 → abort TP (would invert direction)."""
        scalper = _make_rb_scalper(paper=False)
        result = scalper._place_server_take_profit(
            symbol="BTCUSDT",
            signal=1,
            quantity=0.01,
            entry_price=50000.0,
            move_size=0.03,
            target_retrace=-0.5,
        )
        assert result is None, "Negative target_retrace should return None (D1 guard)"

    def test_zero_target_retrace_returns_none(self):
        """target_retrace=0 → abort TP (zero profit target)."""
        scalper = _make_rb_scalper(paper=False)
        result = scalper._place_server_take_profit(
            symbol="BTCUSDT",
            signal=1,
            quantity=0.01,
            entry_price=50000.0,
            move_size=0.03,
            target_retrace=0.0,
        )
        assert result is None, "Zero target_retrace should return None (D1 guard)"

    def test_large_move_exceeding_100pct_returns_none(self):
        """move_size=0.9 × target_retrace=2.0 = 1.8 ≥ 1.0 → abort for SHORT (price would go ≤ 0)."""
        scalper = _make_rb_scalper(paper=False)
        result = scalper._place_server_take_profit(
            symbol="BTCUSDT",
            signal=-1,  # SHORT
            quantity=0.01,
            entry_price=50000.0,
            move_size=0.9,
            target_retrace=2.0,  # 90% × 2.0 = 180% → negative price for SHORT
        )
        assert result is None, "move_size × target_retrace ≥ 1.0 → SHORT price ≤ 0 → must return None"

    def test_rubberband_default_target_retrace_allowed(self):
        """target_retrace=1.5 with move_size=0.03 → target_distance=0.045 < 1.0 → allowed (RubberBand default)."""
        scalper = _make_rb_scalper(paper=False)
        scalper.client._request_futures_api.return_value = {"algoId": "tp_rb"}
        scalper.get_symbol_info = MagicMock(return_value=self._sym_info())
        result = scalper._place_server_take_profit(
            symbol="BTCUSDT",
            signal=1,
            quantity=0.01,
            entry_price=50000.0,
            move_size=0.03,
            target_retrace=1.5,  # RubberBand default — should NOT be blocked
        )
        assert result is not None, "target_retrace=1.5 with small move_size=0.03 should be allowed"
