"""positions.py coverage tests — targeting uncovered sizing, bracket, and close paths.

Coverage targets:
  L57-291:  Vol_straddle position sizing (dynamic capital, SoftTight DD, CW-Tier, MA override)
  L319-454: Vol_straddle bracket placement (SL/TP server orders, WOULD_TRIGGER)
  L1377-1461: Close path (trade record, journal.record(), balance fields)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np

from quantbox.scalper import RubberBandScalper
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PRICE = 100.0  # Fixed fill price for all tests — makes qty = notional/100


def _make_vs_scalper(
    vs_config: VolStraddleConfig | None = None,
    paper: bool = False,
    balance: float = 1000.0,
) -> RubberBandScalper:
    """Create a vol_straddle scalper with mocked Binance client."""
    if vs_config is None:
        vs_config = VolStraddleConfig()

    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client"),
        patch("quantbox.scalper.DiscordNotifier"),
        patch("quantbox.scalper.DiscordBotRunner", None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="k",
            binance_api_secret="s",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token="",
            discord_channel_id="",
            paper_balance=balance,
        )
        scalper = RubberBandScalper(
            paper_mode=paper,
            strategy_name="vol_straddle",
            vs_config=vs_config,
        )

    # Pre-populate symbol info cache so _round_qty() never hits the API.
    # max_notional=None sentinel: leverage bracket already "tried" → no API call.
    scalper._symbol_info_cache["TESTUSDT"] = {
        "step_size": 0.001,
        "tick_size": 0.01,
        "quantity_precision": 3,
        "price_precision": 2,
        "min_notional": 5.0,
        "max_notional": None,  # sentinel: no cap by default
    }
    scalper._symbol_info_cache["SHORTUSDT"] = scalper._symbol_info_cache["TESTUSDT"]

    scalper.client = MagicMock()
    scalper.discord = MagicMock()
    scalper._paper_balance = balance
    scalper._high_water_mark = 0.0  # No DD by default

    # Default get_balance → paper balance
    scalper.get_balance = MagicMock(return_value=balance)
    scalper._save_state = MagicMock()

    return scalper


def _set_signal_meta(scalper: RubberBandScalper, **meta) -> None:
    """Inject signal metadata into the strategy mock."""
    defaults = {
        "sl": scalper.vs_config.d_sl,
        "tp": scalper.vs_config.d_tp,
        "tf": "15m",
        "straddle": False,
        "strategy_type": "vr_squeeze",
        "conviction_score": 1,
    }
    defaults.update(meta)
    scalper.strategy.get_last_signal_meta = MagicMock(return_value=defaults)


def _mock_open(
    scalper: RubberBandScalper,
    sl_id: str | None = "sl_111",
    tp_id: str | None = "tp_222",
) -> None:
    """Mock the order placement calls for a successful open."""
    scalper._place_order = MagicMock(
        return_value={
            "avgPrice": str(PRICE),
            "executedQty": "1.0",
            "status": "FILLED",
        }
    )
    scalper._place_server_stop_loss = MagicMock(return_value=sl_id)
    scalper._place_server_take_profit = MagicMock(return_value=tp_id)
    scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)


def _open(scalper: RubberBandScalper, balance: float = 1000.0) -> bool:
    closes = np.full(210, PRICE)
    return scalper._open_position("TESTUSDT", 1, closes, balance)


# ─────────────────────────────────────────────────────────────────────────────
# TestPositionSizing — L57-291
# ─────────────────────────────────────────────────────────────────────────────


class TestPositionSizing:
    """Vol_straddle position sizing logic: flat, dynamic, SoftTight, CW-Tier."""

    def test_flat_sizing_baseline(self):
        """Default: per_slot = balance/10, notional = per_slot × 30."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=False,
            base_leverage=30.0,
            max_concurrent=10,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper)

        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # per_slot=$100, notional=$3000, qty=3000/100=30
        assert abs(qty - 30.0) < 0.01, f"Expected qty≈30, got {qty}"

    def test_flat_sizing_different_balance(self):
        """Flat sizing scales linearly with balance."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=False,
            base_leverage=30.0,
            max_concurrent=10,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=2000.0)
        scalper.get_balance = MagicMock(return_value=2000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper)

        _open(scalper, 2000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # per_slot=$200, notional=$6000, qty=60
        assert abs(qty - 60.0) < 0.01, f"Expected qty≈60, got {qty}"

    def test_dynamic_capital_disabled(self):
        """dynamic_capital_enabled=False → always flat balance/max_concurrent."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg)
        # n_active is derived from _positions length — start with empty
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper)

        qty = float(scalper._place_order.call_args[0][2])
        # always flat: $100/slot
        assert abs(qty - 30.0) < 0.01

    def test_dynamic_capital_n0_capped_to_flat(self):
        """dynamic_capital n_active=0: balance/1=$1000 > flat $100 → capped."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=True,
            soft_equity_scale=False,
            cw_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        # n_active=0 (empty _positions) → per_slot=$1000, capped to $100
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # Capped at flat=$100 → notional=$3000 → qty=30
        assert abs(qty - 30.0) < 0.01

    def test_dynamic_capital_n9_capped_to_flat(self):
        """dynamic_capital n_active=9: balance/10=$100 = flat → no cap warning."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=True,
            soft_equity_scale=False,
            cw_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        # Add 9 fake positions to set n_active=9
        for i in range(9):
            scalper.strategy._positions[f"SYM{i}USDT:LONG"] = {
                "signal": 1,
                "entry_price": 100.0,
            }
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        assert abs(qty - 30.0) < 0.01

    # ── SoftTight DD Scaling ──────────────────────────────────────────────────

    def test_softtight_no_dd(self):
        """DD=0% → no scaling, full flat sizing."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=True,
            cw_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        scalper._high_water_mark = 1000.0  # DD = 0%
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # No scaling → notional=$3000 → qty=30
        assert abs(qty - 30.0) < 0.1

    def test_softtight_dd1_80pct_scale(self):
        """DD=2% → 0.80x scaling on per_slot_capital."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=True,
            cw_enabled=False,
            soft_eq_dd1=0.02,
            soft_eq_scale1=0.80,
        )
        balance = 980.0
        scalper = _make_vs_scalper(vs_config=cfg, balance=balance)
        scalper.get_balance = MagicMock(return_value=balance)
        scalper._high_water_mark = 1000.0  # DD=(1000-980)/1000=2%
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, balance)

        qty = float(scalper._place_order.call_args[0][2])
        # flat_per_slot=98, ×0.80=78.4, notional=78.4×30=2352, qty=23.52
        expected_qty = (balance / 10 * 0.80 * 30) / PRICE
        assert abs(qty - expected_qty) < 0.1, f"Expected ~{expected_qty:.2f}, got {qty:.3f}"

    def test_softtight_dd2_60pct_scale(self):
        """DD=4% → 0.60x scaling."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=True,
            cw_enabled=False,
            soft_eq_dd2=0.04,
            soft_eq_scale2=0.60,
        )
        balance = 960.0
        scalper = _make_vs_scalper(vs_config=cfg, balance=balance)
        scalper.get_balance = MagicMock(return_value=balance)
        scalper._high_water_mark = 1000.0  # DD=(1000-960)/1000=4%
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, balance)

        qty = float(scalper._place_order.call_args[0][2])
        expected_qty = (balance / 10 * 0.60 * 30) / PRICE
        assert abs(qty - expected_qty) < 0.1

    def test_softtight_dd3_40pct_scale(self):
        """DD=6% → 0.40x scaling."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=True,
            cw_enabled=False,
            soft_eq_dd3=0.06,
            soft_eq_scale3=0.40,
        )
        balance = 940.0
        scalper = _make_vs_scalper(vs_config=cfg, balance=balance)
        scalper.get_balance = MagicMock(return_value=balance)
        scalper._high_water_mark = 1000.0  # DD=6%
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, balance)

        qty = float(scalper._place_order.call_args[0][2])
        expected_qty = (balance / 10 * 0.40 * 30) / PRICE
        assert abs(qty - expected_qty) < 0.1

    def test_softtight_hwm_zero_uses_balance(self):
        """HWM=0 → DD=0%, no scaling (hwm fallback to balance)."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=True,
            cw_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        scalper._high_water_mark = 0.0  # triggers hwm = balance fallback
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        assert abs(qty - 30.0) < 0.1

    # ── CW-Tier Conviction Sizing ─────────────────────────────────────────────

    def test_cw_tier_score0_half_size(self):
        """CW conviction score=0 → 0.5x notional."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=0, strategy_type="vr_squeeze")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # notional=$3000 × 0.5=$1500 → qty=15
        assert abs(qty - 15.0) < 0.1, f"Expected ~15, got {qty}"

    def test_cw_tier_score1_baseline(self):
        """CW score=1 → 1.0x (baseline notional)."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=1, strategy_type="vr_squeeze")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        assert abs(qty - 30.0) < 0.1

    def test_cw_tier_score4_max(self):
        """CW score=4 → 2.5x notional."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=4, strategy_type="vr_squeeze")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # $3000 × 2.5=$7500 → qty=75
        assert abs(qty - 75.0) < 0.1, f"Expected ~75, got {qty}"

    def test_cw_tier_score_clamped_to_last(self):
        """CW score > len(multipliers) → clamped to last multiplier."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=99, strategy_type="vr_squeeze")  # clamped to 4
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        assert abs(qty - 75.0) < 0.1

    def test_cw_disabled_for_ma_cluster(self):
        """CW multiplier NOT applied for ma_cluster strategy_type."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=0, strategy_type="ma_cluster")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # CW not applied → baseline notional=$3000 → qty=30 (no 0.5x)
        assert abs(qty - 30.0) < 0.1, f"Expected ~30 (no CW), got {qty}"

    # ── MA Cluster Leverage Override ─────────────────────────────────────────

    def test_ma_cluster_leverage_override(self):
        """MA cluster: leverage_override=21 replaces base_leverage=30."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=False,
            base_leverage=30.0,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(
            scalper,
            strategy_type="ma_cluster",
            leverage_override=21,
            sl=0.02,
        )
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # per_slot=$100, notional=$100×21=$2100, qty=21
        assert abs(qty - 21.0) < 0.1, f"Expected ~21, got {qty}"

    def test_leverage_override_zero_clamped_to_1(self, caplog):
        """leverage_override=0 → clamped to 1.0 with WARNING (not ZeroDivision)."""
        import logging

        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=False,
            base_leverage=30.0,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, strategy_type="ma_cluster", leverage_override=0, sl=0.02)
        with caplog.at_level(logging.WARNING, logger="quantbox.execution.positions"):
            _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # clamped to 1 → notional=$100×1=$100 → qty=1 (price=100)
        assert qty >= 1.0, f"qty must be >= 1, got {qty}"
        assert any("clamped" in r.message for r in caplog.records)

    def test_leverage_override_exceeds_max_clamped_to_125(self, caplog):
        """leverage_override=200 → clamped to 125.0 with WARNING."""
        import logging

        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=False,
            base_leverage=30.0,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper, strategy_type="ma_cluster", leverage_override=200, sl=0.02)
        with caplog.at_level(logging.WARNING, logger="quantbox.execution.positions"):
            _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # clamped to 125 → notional=$100×125=$12500 → qty=125 (price=100)
        assert abs(qty - 125.0) < 0.5, f"Expected ~125, got {qty}"
        assert any("clamped" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# TestBracketPlacement — L319-454
# ─────────────────────────────────────────────────────────────────────────────


class TestBracketPlacement:
    """After _open_position: SL/TP bracket server orders, WOULD_TRIGGER handling."""

    def test_sl_and_tp_placed_after_open(self):
        """Both SL and TP server orders placed on successful open."""
        scalper = _make_vs_scalper()
        _mock_open(scalper, sl_id="sl_111", tp_id="tp_222")
        _set_signal_meta(scalper)
        result = _open(scalper)

        assert result is True
        scalper._place_server_stop_loss.assert_called_once()
        scalper._place_server_take_profit.assert_called_once()
        pos = scalper.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_sl_order_id"] == "sl_111"
        assert pos["server_tp_order_id"] == "tp_222"

    def test_sl_would_trigger_stored_as_none(self):
        """SL WOULD_TRIGGER → stored as None (bracket repair handles it next cycle)."""
        scalper = _make_vs_scalper()
        _mock_open(scalper, sl_id="WOULD_TRIGGER", tp_id="tp_222")
        _set_signal_meta(scalper)
        _open(scalper)

        pos = scalper.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_sl_order_id"] is None
        assert pos["server_tp_order_id"] == "tp_222"

    def test_tp_would_trigger_stored_as_none(self):
        """TP WOULD_TRIGGER → stored as None."""
        scalper = _make_vs_scalper()
        _mock_open(scalper, sl_id="sl_111", tp_id="WOULD_TRIGGER")
        _set_signal_meta(scalper)
        _open(scalper)

        pos = scalper.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_sl_order_id"] == "sl_111"
        assert pos["server_tp_order_id"] is None

    def test_position_recorded_with_correct_sl_tp_pct(self):
        """Position dict has sl/tp from signal metadata."""
        scalper = _make_vs_scalper()
        _mock_open(scalper)
        _set_signal_meta(scalper, sl=0.015, tp=0.15)
        _open(scalper)

        pos = scalper.strategy._positions["TESTUSDT:LONG"]
        assert abs(pos["sl"] - 0.015) < 1e-6
        assert abs(pos["tp"] - 0.15) < 1e-6

    def test_position_recorded_with_tf_label(self):
        """Position dict records the TF label from signal meta."""
        scalper = _make_vs_scalper()
        _mock_open(scalper)
        _set_signal_meta(scalper, tf="1h")
        _open(scalper)

        pos = scalper.strategy._positions["TESTUSDT:LONG"]
        assert pos["tf"] == "1h"

    def test_discord_notified_on_open(self):
        """Discord.send() called at least once after successful open."""
        scalper = _make_vs_scalper()
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper)

        scalper.discord.send.assert_called()

    def test_open_returns_false_when_place_order_fails(self):
        """_open_position returns False if _place_order returns None."""
        scalper = _make_vs_scalper()
        scalper._place_order = MagicMock(return_value=None)
        scalper._place_server_stop_loss = MagicMock(return_value="sl_111")
        scalper._place_server_take_profit = MagicMock(return_value="tp_222")
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)
        _set_signal_meta(scalper)

        result = _open(scalper)
        assert result is False

    def test_short_position_opened(self):
        """_open_position works for SHORT signal=-1."""
        scalper = _make_vs_scalper()
        _mock_open(scalper)
        _set_signal_meta(scalper)
        closes = np.full(210, PRICE)
        result = scalper._open_position("TESTUSDT", -1, closes, 1000.0)

        assert result is True
        pos = scalper.strategy._positions["TESTUSDT:SHORT"]
        assert pos["signal"] == -1


# ─────────────────────────────────────────────────────────────────────────────
# TestClosePaths — L1377-1461
# ─────────────────────────────────────────────────────────────────────────────


class TestClosePaths:
    """_close_position: trade record creation, journal.record() hook."""

    def _setup_position(self, scalper: RubberBandScalper, symbol: str = "TESTUSDT", signal: int = 1) -> None:
        """Inject a fake open position for close-path testing."""
        pos_key = f"{symbol}:LONG" if signal == 1 else f"{symbol}:SHORT"
        pos_side = "LONG" if signal == 1 else "SHORT"
        scalper.strategy._positions[pos_key] = {
            "symbol": symbol,
            "signal": signal,
            "entry_price": PRICE,
            "notional": 3000.0,
            "bars_held": 10,
            "peak_gain": 0.02,
            "best_pnl": 0.02,
            "worst_pnl": -0.005,
            "server_sl_order_id": "sl_123",
            "server_tp_order_id": "tp_456",
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": pos_side,
            "trail_active": False,
            "opened_at": "2026-03-09T10:00:00+00:00",
            "strategy_type": "vr_squeeze",
            "balance_before": 1234.0,
        }
        return pos_key

    def _make_close_scalper(self) -> RubberBandScalper:
        scalper = _make_vs_scalper()
        # Mock cancel methods
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        # Mock _place_order (close order)
        close_price = 105.0
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": str(close_price),
                "executedQty": "30.0",
                "status": "FILLED",
            }
        )
        scalper._journal = MagicMock()
        return scalper

    def test_close_creates_trade_record(self):
        """_close_position appends a trade to _trade_history."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)
        before_count = len(scalper._trade_history)

        result = scalper._close_position(pos_key, reason="trailing_stop", current_price=105.0)

        assert result is True
        assert len(scalper._trade_history) == before_count + 1

    def test_close_journal_record_called(self):
        """_close_position calls journal.record() exactly once."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)

        scalper._close_position(pos_key, reason="trailing_stop", current_price=105.0)

        scalper._journal.record.assert_called_once()

    def test_close_trade_has_correct_reason(self):
        """Trade record contains the close reason."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)
        scalper._close_position(pos_key, reason="max_hold", current_price=105.0)

        trade = scalper._trade_history[-1]
        assert trade["reason"] == "max_hold"

    def test_close_trade_pnl_calculation(self):
        """PnL calculated correctly for LONG position."""
        close_price = 110.0
        scalper = self._make_close_scalper()
        scalper._place_order = MagicMock(
            return_value={"avgPrice": str(close_price), "executedQty": "30.0", "status": "FILLED"}
        )
        pos_key = self._setup_position(scalper, signal=1)
        scalper._close_position(pos_key, reason="take_profit", current_price=close_price)

        trade = scalper._trade_history[-1]
        expected_pnl_pct = (close_price - PRICE) / PRICE * 100  # +10%
        assert abs(trade["pnl_pct"] - expected_pnl_pct) < 0.01

    def test_close_trade_pnl_calculation_short(self):
        """PnL calculated correctly for SHORT position."""
        close_price = 90.0
        scalper = self._make_close_scalper()
        scalper._place_order = MagicMock(
            return_value={"avgPrice": str(close_price), "executedQty": "30.0", "status": "FILLED"}
        )
        pos_key = self._setup_position(scalper, signal=-1)
        scalper._close_position(pos_key, reason="trailing_stop", current_price=close_price)

        trade = scalper._trade_history[-1]
        expected_pnl_pct = (PRICE - close_price) / PRICE * 100  # +10%
        assert abs(trade["pnl_pct"] - expected_pnl_pct) < 0.01

    def test_close_trade_has_mfe_mae(self):
        """Trade record includes mfe_pct and mae_pct from position."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)
        scalper._close_position(pos_key, reason="stop_loss", current_price=99.0)

        trade = scalper._trade_history[-1]
        assert "mfe_pct" in trade
        assert "mae_pct" in trade
        assert abs(trade["mfe_pct"] - 2.0) < 0.01  # best_pnl=0.02 → 2%
        assert abs(trade["mae_pct"] - (-0.5)) < 0.01  # worst_pnl=-0.005 → -0.5%

    def test_close_trade_has_entry_time(self):
        """Trade record includes entry_time from opened_at."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)
        scalper._close_position(pos_key, reason="drawdown_halt", current_price=105.0)

        trade = scalper._trade_history[-1]
        assert trade["entry_time"] == "2026-03-09T10:00:00+00:00"

    def test_close_balance_before_is_open_balance_not_close_balance(self):
        """balance_before in trade record must reflect the balance AT OPEN, not close.
        Regression test for bug where balance_before == balance_after (both close-time).
        """
        scalper = self._make_close_scalper()
        # Simulate open balance ($1234) stored on position at open time
        pos_key = self._setup_position(scalper)  # position has balance_before=1234.0
        # Simulate a different (lower) balance at close time
        scalper.get_balance = MagicMock(return_value=1220.0)

        scalper._close_position(pos_key, reason="take_profit", current_price=105.0)

        trade = scalper._trade_history[-1]
        # balance_before must be the open-time value (1234), not close-time (1220)
        assert trade["balance_before"] == pytest.approx(1234.0), (
            f"balance_before should be open-time balance 1234.0, got {trade['balance_before']}"
        )
        # balance_after is close-time balance (mocked to 1220)
        assert trade["balance_after"] == pytest.approx(1220.0), (
            f"balance_after should be close-time balance 1220.0, got {trade['balance_after']}"
        )
        # They should differ
        assert trade["balance_before"] != trade["balance_after"]

    def test_close_cancels_server_orders_first(self):
        """SL and TP server orders cancelled before close order is placed."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)
        scalper._close_position(pos_key, reason="max_hold", current_price=100.0)

        scalper._cancel_server_stop_loss.assert_called_once()
        scalper._cancel_server_take_profit.assert_called_once()

    def test_close_discord_called(self):
        """Discord notification sent after close."""
        scalper = self._make_close_scalper()
        pos_key = self._setup_position(scalper)
        scalper._close_position(pos_key, reason="stop_loss", current_price=99.0)
        scalper.discord.send.assert_called()

    def test_close_returns_false_for_unknown_symbol(self):
        """Returns False when symbol not in active positions."""
        scalper = self._make_close_scalper()
        result = scalper._close_position("UNKNOWN:LONG", reason="stop_loss", current_price=100.0)
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# TestExchangeDedup — L278-291
# ─────────────────────────────────────────────────────────────────────────────


class TestExchangeDedup:
    """Exchange-level dedup guard prevents double entry."""

    def test_dedup_skips_when_already_open_on_exchange(self):
        """Returns False when exchange already has LONG position open."""
        scalper = _make_vs_scalper(paper=False)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        # Exchange reports position already open
        scalper.client.futures_position_information = MagicMock(
            return_value=[
                {"positionSide": "LONG", "positionAmt": "30.0", "entryPrice": "100.0"},
            ]
        )
        result = _open(scalper)
        assert result is False
        scalper._place_order.assert_not_called()

    def test_dedup_skips_with_state_out_of_sync(self):
        """Logs warning and returns False even when internal state has no position."""
        scalper = _make_vs_scalper(paper=False)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        scalper.client.futures_position_information = MagicMock(
            return_value=[
                {"positionSide": "LONG", "positionAmt": "15.5"},
            ]
        )
        # No internal position → state out of sync path
        assert "TESTUSDT:LONG" not in scalper.strategy._positions
        result = _open(scalper)
        assert result is False

    def test_dedup_allows_entry_when_no_position_on_exchange(self):
        """Proceeds normally when exchange has zero position amount."""
        scalper = _make_vs_scalper(paper=False)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        scalper.client.futures_position_information = MagicMock(
            return_value=[
                {"positionSide": "LONG", "positionAmt": "0.0"},
            ]
        )
        result = _open(scalper)
        assert result is True

    def test_dedup_allows_entry_when_exchange_check_fails(self):
        """Proceeds if exchange API call raises (fail-safe behavior)."""
        scalper = _make_vs_scalper(paper=False)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        scalper.client.futures_position_information = MagicMock(side_effect=Exception("Network error"))
        result = _open(scalper)
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# TestPaperMode — L383-384 paper cost deduction
# ─────────────────────────────────────────────────────────────────────────────


class TestPaperMode:
    """Paper mode position sizing and cost deduction."""

    def test_paper_mode_deducts_cost_on_open(self):
        """Opening in paper mode deducts cost_per_side from _paper_balance."""
        scalper = _make_vs_scalper(paper=True, balance=1000.0)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        balance_before = scalper._paper_balance

        _open(scalper, 1000.0)

        # Cost should have been deducted
        assert scalper._paper_balance < balance_before


# ─────────────────────────────────────────────────────────────────────────────
# TestSteepXConfidence — L151-162
# ─────────────────────────────────────────────────────────────────────────────


class TestSteepXConfidence:
    """SteepX cap_curve confidence-level position sizing."""

    def _make_steep(self, conf_level: int) -> tuple:
        cfg = VolStraddleConfig(
            steep_cap_curve=True,
            soft_equity_scale=False,
            cw_enabled=False,
            dynamic_capital_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        scalper.strategy.get_last_confidence = MagicMock(return_value=conf_level)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper)
        qty = float(scalper._place_order.call_args[0][2])
        return scalper, qty

    def test_steep_conf5_uses_cap_mult_conf5(self):
        """conf_level >= 5 → cap_mult_conf5 applied."""
        cfg = VolStraddleConfig(
            steep_cap_curve=True,
            soft_equity_scale=False,
            cw_enabled=False,
            dynamic_capital_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        scalper.strategy.get_last_confidence = MagicMock(return_value=5)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper)
        # Should have called _place_order successfully (not return False)
        scalper._place_order.assert_called_once()

    def test_steep_conf0_skips_multiplier(self):
        """conf_level == 0 → steep_cap_curve skipped, flat sizing."""
        cfg = VolStraddleConfig(
            steep_cap_curve=True,
            soft_equity_scale=False,
            cw_enabled=False,
            dynamic_capital_enabled=False,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        scalper.strategy.get_last_confidence = MagicMock(return_value=0)
        _mock_open(scalper)
        _set_signal_meta(scalper)
        _open(scalper)
        qty = float(scalper._place_order.call_args[0][2])
        # conf=0 skips multiplier → flat: $100/slot × 30x / $100 = 30
        assert abs(qty - 30.0) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestPartialClose — L653, L662-663, L679
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialClose:
    """_partial_close_position: race guard, cancel orders, execute."""

    def _setup(self) -> tuple:
        scalper = _make_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_new")
        scalper._place_server_take_profit = MagicMock(return_value="tp_new")
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": "105.0",
                "executedQty": "15.0",
                "status": "FILLED",
            }
        )
        scalper.strategy._positions["TESTUSDT:LONG"] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": PRICE,
            "notional": 3000.0,
            "bars_held": 5,
            "peak_gain": 0.05,
            "server_sl_order_id": "sl_old",
            "server_tp_order_id": "tp_old",
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "move_size": 0.01,
            "entry_stop_mult": 1.0,
            "partial_closed": False,
            "_server_partial_tp_filled": False,
        }
        return scalper

    def test_partial_close_cancels_old_orders(self):
        """Old SL and TP orders cancelled before partial close order."""
        scalper = self._setup()
        scalper._partial_close_position("TESTUSDT:LONG", current_price=105.0)
        scalper._cancel_server_stop_loss.assert_called_once()
        scalper._cancel_server_take_profit.assert_called_once()

    def test_partial_close_server_filled_flag_skips(self):
        """_server_partial_tp_filled flag → skip partial close, return False."""
        scalper = self._setup()
        scalper.strategy._positions["TESTUSDT:LONG"]["_server_partial_tp_filled"] = True
        result = scalper._partial_close_position("TESTUSDT:LONG", current_price=105.0)
        assert result is False
        scalper._place_order.assert_not_called()

    def test_partial_close_unknown_symbol_returns_false(self):
        """Returns False for unknown symbol."""
        scalper = self._setup()
        result = scalper._partial_close_position("UNKNOWN:LONG", current_price=105.0)
        assert result is False

    def test_partial_close_succeeds(self):
        """Happy path: place_order returns result → True."""
        scalper = self._setup()
        result = scalper._partial_close_position("TESTUSDT:LONG", current_price=105.0)
        assert result is True
        scalper._place_order.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TestBracketNonPBT — L526-527
# ─────────────────────────────────────────────────────────────────────────────


class TestBracketNonPBT:
    """partial_exit_enabled=False (non-PBT) TP WOULD_TRIGGER path."""

    def test_non_pbt_tp_would_trigger_stored_as_none(self):
        """In non-PBT mode, TP WOULD_TRIGGER at entry → stored as None."""
        from quantbox.strategy.rubberband.config import RubberBandConfig

        cfg = RubberBandConfig(partial_exit_enabled=False)
        vs_cfg = VolStraddleConfig(cw_enabled=False, soft_equity_scale=False)

        scalper = _make_vs_scalper(vs_config=vs_cfg)
        scalper.config = cfg  # Override to non-PBT

        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": str(PRICE),
                "executedQty": "1.0",
                "status": "FILLED",
            }
        )
        scalper._place_server_stop_loss = MagicMock(return_value="sl_111")
        scalper._place_server_take_profit = MagicMock(return_value="WOULD_TRIGGER")
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)
        _set_signal_meta(scalper)

        result = _open(scalper)
        assert result is True
        pos = scalper.strategy._positions["TESTUSDT:LONG"]
        assert pos["server_tp_order_id"] is None
        assert pos["server_sl_order_id"] == "sl_111"


class TestCloseWithRemainingPositions:
    """_close_position Discord embed includes remaining positions summary."""

    def test_close_discord_shows_remaining_positions(self):
        """When other positions exist, Discord embed includes their summary lines."""
        scalper = _make_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_order = MagicMock(return_value={"avgPrice": "105.0", "executedQty": "30.0", "status": "FILLED"})
        scalper._journal = MagicMock()

        # Add the position to close
        scalper.strategy._positions["TESTUSDT:LONG"] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": PRICE,
            "notional": 3000.0,
            "bars_held": 5,
            "peak_gain": 0.02,
            "best_pnl": 0.02,
            "worst_pnl": -0.005,
            "server_sl_order_id": "sl_1",
            "server_tp_order_id": "tp_1",
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "trail_active": False,
            "opened_at": "2026-03-09T10:00:00+00:00",
            "strategy_type": "vr_squeeze",
        }

        # Add a second position that remains after close
        scalper.strategy._positions["ETHUSDT:SHORT"] = {
            "symbol": "ETHUSDT",
            "signal": -1,
            "entry_price": 3000.0,
            "notional": 3000.0,
            "bars_held": 2,
            "peak_gain": 0.01,
            "best_pnl": 0.01,
            "worst_pnl": -0.002,
            "server_sl_order_id": None,
            "server_tp_order_id": None,
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "SHORT",
            "trail_active": False,
        }

        result = scalper._close_position("TESTUSDT:LONG", reason="trailing_stop", current_price=105.0)
        assert result is True
        # Discord was called
        scalper.discord.send.assert_called()

    def test_partial_close_no_tp_order(self):
        """Partial close with no server TP — cancel not called for TP (L679 branch)."""
        scalper = _make_vs_scalper()
        scalper._cancel_server_stop_loss = MagicMock()
        scalper._cancel_server_take_profit = MagicMock()
        scalper._place_server_stop_loss = MagicMock(return_value="sl_new")
        scalper._place_server_take_profit = MagicMock(return_value="tp_new")
        scalper._place_order = MagicMock(return_value={"avgPrice": "105.0", "executedQty": "15.0", "status": "FILLED"})
        scalper.strategy._positions["TESTUSDT:LONG"] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": PRICE,
            "notional": 3000.0,
            "bars_held": 5,
            "peak_gain": 0.05,
            "server_sl_order_id": "sl_old",
            "server_tp_order_id": None,  # No TP order
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "move_size": 0.01,
            "entry_stop_mult": 1.0,
            "partial_closed": False,
            "_server_partial_tp_filled": False,
        }

        result = scalper._partial_close_position("TESTUSDT:LONG", current_price=105.0)
        assert result is True
        scalper._cancel_server_stop_loss.assert_called_once()
        scalper._cancel_server_take_profit.assert_not_called()  # No TP to cancel


# ─────────────────────────────────────────────────────────────────────────────
# TestRubberbandOpen — L455-533 rubberband else branch in _open_position
# ─────────────────────────────────────────────────────────────────────────────


def _make_rb_scalper(balance: float = 1000.0, rb_cfg=None) -> RubberBandScalper:
    """Create a rubberband scalper with mocked Binance client."""
    from quantbox.strategy.rubberband.config import RubberBandConfig

    if rb_cfg is None:
        rb_cfg = RubberBandConfig(
            lookback=10,
            threshold=0.01,
            partial_exit_enabled=False,  # simpler non-PBT
            dynamic_leverage_enabled=False,
            vol_adjusted_size=False,  # disable ATR-based sizing
            steep_cap_curve=False,  # disable confidence curve
            mtf_confirmation_enabled=False,
        )

    with (
        patch("quantbox.scalper.get_settings") as mock_settings,
        patch("quantbox.scalper.Client"),
        patch("quantbox.scalper.DiscordNotifier"),
        patch("quantbox.scalper.DiscordBotRunner", None),
    ):
        mock_settings.return_value = MagicMock(
            binance_api_key="k",
            binance_api_secret="s",
            binance_testnet=False,
            discord_webhook_url="",
            discord_bot_token="",
            discord_channel_id="",
            paper_balance=balance,
        )
        scalper = RubberBandScalper(
            paper_mode=True,
            strategy_name="rubberband",
            config=rb_cfg,
        )

    scalper._symbol_info_cache["TESTUSDT"] = {
        "step_size": 0.001,
        "tick_size": 0.01,
        "quantity_precision": 3,
        "price_precision": 2,
        "min_notional": 5.0,
        "min_qty": 0.001,
        "max_qty": 1000.0,
    }
    scalper.get_balance = MagicMock(return_value=balance)
    return scalper


class TestRubberbandOpen:
    """_open_position rubberband else branch (L455-533)."""

    def test_rubberband_open_long(self):
        """Rubberband strategy: _open_position places order and records position."""
        scalper = _make_rb_scalper()
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": str(PRICE),
                "executedQty": "1.0",
                "status": "FILLED",
            }
        )
        scalper._place_server_stop_loss = MagicMock(return_value="sl_rb")
        scalper._place_server_take_profit = MagicMock(return_value="tp_rb")
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)
        scalper.strategy.signal_meta = {
            "TESTUSDT": {
                "sl": 0.01,
                "tp": 0.12,
                "tf": "15m",
                "conviction_score": 3,
                "strategy_type": "rubberband",
            }
        }

        closes = np.array([95.0] * 69 + [100.0])  # 5% move for non-zero move_size
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)

        assert result is True
        assert "TESTUSDT" in scalper.strategy._positions
        """Rubberband strategy: SHORT signal=-1 opens position."""
        scalper = _make_rb_scalper()
        scalper._place_order = MagicMock(
            return_value={
                "avgPrice": str(PRICE),
                "executedQty": "1.0",
                "status": "FILLED",
            }
        )
        scalper._place_server_stop_loss = MagicMock(return_value="sl_rb")
        scalper._place_server_take_profit = MagicMock(return_value=None)
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)
        scalper.strategy.signal_meta = {
            "TESTUSDT": {
                "sl": 0.01,
                "tp": 0.12,
                "tf": "15m",
                "conviction_score": 2,
                "strategy_type": "rubberband",
            }
        }

        closes = np.array([105.0] * 69 + [100.0])  # 5% drop for non-zero move_size
        result = scalper._open_position("TESTUSDT", -1, closes, 1000.0)

        assert result is True
        pos = scalper.strategy._positions["TESTUSDT"]
        assert pos["signal"] == -1

    def test_rubberband_open_returns_false_on_failed_order(self):
        """_open_position returns False when _place_order returns None."""
        scalper = _make_rb_scalper()
        scalper._place_order = MagicMock(return_value=None)
        scalper._place_server_stop_loss = MagicMock(return_value="sl_rb")
        scalper._place_server_take_profit = MagicMock(return_value=None)
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)
        scalper.strategy.signal_meta = {
            "TESTUSDT": {
                "sl": 0.01,
                "tp": 0.12,
                "tf": "15m",
                "conviction_score": 3,
                "strategy_type": "rubberband",
            }
        }

        closes = np.array([95.0] * 69 + [100.0])  # non-zero move_size
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is False


class TestRubberbandSteepXAndLeverage:
    """Rubberband SteepX confidence sizing and dynamic leverage paths."""

    def _make_scalper(self, **extra_cfg) -> RubberBandScalper:
        from quantbox.strategy.rubberband.config import RubberBandConfig

        rb_cfg = RubberBandConfig(
            lookback=10,
            threshold=0.01,
            partial_exit_enabled=False,
            dynamic_leverage_enabled=False,
            vol_adjusted_size=False,
            mtf_confirmation_enabled=False,
            **extra_cfg,
        )

        with (
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client"),
            patch("quantbox.scalper.DiscordNotifier"),
            patch("quantbox.scalper.DiscordBotRunner", None),
        ):
            mock_settings.return_value = MagicMock(
                binance_api_key="k",
                binance_api_secret="s",
                binance_testnet=False,
                discord_webhook_url="",
                discord_bot_token="",
                discord_channel_id="",
                paper_balance=1000.0,
            )
            scalper = RubberBandScalper(
                paper_mode=True,
                strategy_name="rubberband",
                config=rb_cfg,
            )
        scalper._symbol_info_cache["TESTUSDT"] = {
            "step_size": 0.001,
            "tick_size": 0.01,
            "quantity_precision": 3,
            "price_precision": 2,
            "min_notional": 5.0,
            "min_qty": 0.001,
            "max_qty": 1000.0,
        }
        scalper.get_balance = MagicMock(return_value=1000.0)
        scalper._place_order = MagicMock(return_value={"avgPrice": "100.0", "executedQty": "1.0", "status": "FILLED"})
        scalper._place_server_stop_loss = MagicMock(return_value="sl_rb")
        scalper._place_server_take_profit = MagicMock(return_value="tp_rb")
        scalper.strategy.get_effective_stop_mult = MagicMock(return_value=1.0)
        scalper.strategy.signal_meta = {"TESTUSDT": {"sl": 0.01, "tp": 0.12, "tf": "15m", "conviction_score": 3}}
        return scalper

    def test_steep_cap_curve_conf5(self):
        """SteepX: conf_level >= 5 uses cap_mult_conf5."""
        scalper = self._make_scalper(steep_cap_curve=True)
        scalper.strategy.get_last_confidence = MagicMock(return_value=5)
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True

    def test_steep_cap_curve_conf4(self):
        """SteepX: conf_level == 4 uses cap_mult_conf4."""
        scalper = self._make_scalper(steep_cap_curve=True)
        scalper.strategy.get_last_confidence = MagicMock(return_value=4)
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True

    def test_steep_cap_curve_conf3(self):
        """SteepX: conf_level == 3 uses cap_mult_conf3."""
        scalper = self._make_scalper(steep_cap_curve=True)
        scalper.strategy.get_last_confidence = MagicMock(return_value=3)
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True

    def test_steep_cap_curve_conf2(self):
        """SteepX: conf_level == 2 uses cap_mult_conf2."""
        scalper = self._make_scalper(steep_cap_curve=True)
        scalper.strategy.get_last_confidence = MagicMock(return_value=2)
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True

    def test_steep_cap_curve_conf1_uses_conf2_mult(self):
        """SteepX: conf_level == 1 falls through to cap_mult_conf2 (else branch)."""
        scalper = self._make_scalper(steep_cap_curve=True)
        scalper.strategy.get_last_confidence = MagicMock(return_value=1)
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True

    def test_dynamic_leverage_confidence_path(self):
        """dynamic_leverage_enabled=True, rr_leverage_enabled=False → confidence leverage."""
        scalper = self._make_scalper()
        # Override config to enable dynamic leverage
        scalper.config.dynamic_leverage_enabled = True
        scalper.config.rr_leverage_enabled = False
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True


class TestRubberbandEdgeCases:
    """Cover rubberband-specific edge cases: SoftTight, MTF boost, small notional, TP WOULD_TRIGGER."""

    def _make_rb(self, **kwargs):
        """Rubberband scalper with sensible defaults."""
        from quantbox.strategy.rubberband.config import RubberBandConfig

        rb_cfg = RubberBandConfig(
            lookback=10,
            threshold=0.01,
            partial_exit_enabled=False,
            dynamic_leverage_enabled=False,
            vol_adjusted_size=False,
            steep_cap_curve=False,
            mtf_confirmation_enabled=False,
            soft_equity_scale=False,
        )
        for k, v in kwargs.items():
            setattr(rb_cfg, k, v)
        return _make_rb_scalper(rb_cfg=rb_cfg)

    def test_softtight_dd3_rubberband(self):
        """L169: DD=7% → soft_eq_scale3 (0.40x) in rubberband else block."""
        scalper = self._make_rb(soft_equity_scale=True)
        scalper._high_water_mark = 1000.0
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 930.0)  # DD=7%
        assert result is True

    def test_softtight_dd2_rubberband(self):
        """L171: DD=5% → soft_eq_scale2 (0.60x) in rubberband else block."""
        scalper = self._make_rb(soft_equity_scale=True)
        scalper._high_water_mark = 1000.0
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 950.0)  # DD=5%
        assert result is True

    def test_softtight_dd1_rubberband(self):
        """L173: DD=3% → soft_eq_scale1 (0.80x) in rubberband else block."""
        scalper = self._make_rb(soft_equity_scale=True)
        scalper._high_water_mark = 1000.0
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 970.0)  # DD=3%
        assert result is True

    def test_mtf_leverage_boost_rubberband(self):
        """L233-235: MTF confirmed → leverage multiplied by mtf_leverage_mult."""
        scalper = self._make_rb(
            mtf_confirmation_enabled=True,
            mtf_leverage_mult=2.0,
            dynamic_leverage_enabled=True,
        )
        scalper.strategy.was_mtf_confirmed = MagicMock(return_value=True)
        scalper.strategy.get_last_confidence = MagicMock(return_value=5)
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True

    def test_notional_too_small_returns_false(self):
        """L240-241: notional < $10 → return False."""
        scalper = self._make_rb()
        closes = np.array([95.0] * 69 + [100.0])
        # balance=0.05 → per_slot=$0.005 → notional too small
        result = scalper._open_position("TESTUSDT", 1, closes, 0.05)
        assert result is False

    def test_below_min_notional_returns_false(self):
        """L252-253: quantity * price < min_notional → return False."""
        scalper = self._make_rb()
        # Override get_symbol_info to return high min_notional
        scalper.get_symbol_info = MagicMock(return_value={"min_notional": 1_000_000.0})
        closes = np.array([95.0] * 69 + [100.0])
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is False

    def test_tp_would_trigger_rubberband_bracket(self):
        """L526-527: TP WOULD_TRIGGER sentinel in rubberband bracket → tp_order_id=None."""
        scalper = self._make_rb()
        closes = np.array([95.0] * 69 + [100.0])
        # SL returns valid id, TP returns WOULD_TRIGGER
        scalper._place_server_stop_loss = MagicMock(return_value="sl_001")
        scalper._place_server_take_profit = MagicMock(return_value="WOULD_TRIGGER")
        result = scalper._open_position("TESTUSDT", 1, closes, 1000.0)
        assert result is True
        key = list(scalper.strategy._positions.keys())[0]
        pos = scalper.strategy._positions[key]
        assert pos.get("server_tp_order_id") is None


class TestCloseEdgeCases:
    """L679 (partial qty=0), L1126 (close qty=0), L1377 (race condition)."""

    def _make_vs(self):
        return _make_vs_scalper()

    def test_partial_close_zero_qty(self):
        """L679: partial_exit_fraction × full_qty rounds to 0 → return False."""
        scalper = self._make_vs()
        key = "TESTUSDT:LONG"
        scalper.strategy._positions[key] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": 100.0,
            "notional": 3000.0,
            "bars_held": 0,
            "peak_gain": 0.0,
            "best_pnl": 0.0,
            "worst_pnl": 0.0,
            "server_sl_order_id": None,
            "server_tp_order_id": None,
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "trail_active": False,
            "opened_at": "2026-01-01T00:00:00+00:00",
            "strategy_type": "vr_squeeze",
        }
        scalper.config.partial_exit_fraction = 1e-10  # tiny → rounds to 0
        scalper._round_qty = MagicMock(return_value=0.0)
        result = scalper._partial_close_position("TESTUSDT:LONG", 100.0)
        assert result is False

    def test_close_zero_qty_returns_false(self):
        """L1126: quantity rounds to 0 in _close_position → return False."""
        scalper = self._make_vs()
        key = "TESTUSDT:LONG"
        scalper.strategy._positions[key] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": 100.0,
            "notional": 3000.0,
            "bars_held": 10,
            "peak_gain": 0.02,
            "best_pnl": 0.02,
            "worst_pnl": -0.005,
            "server_sl_order_id": None,
            "server_tp_order_id": None,
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "trail_active": False,
            "opened_at": "2026-01-01T00:00:00+00:00",
            "strategy_type": "vr_squeeze",
        }
        scalper._round_qty = MagicMock(return_value=0.0)
        result = scalper._close_position("TESTUSDT:LONG", "max_hold", 100.0)
        assert result is False

    def test_close_position_race_returns_false(self):
        """L1377: strategy.close_position() returns None (position already removed)."""
        scalper = self._make_vs()
        key = "TESTUSDT:LONG"
        scalper.strategy._positions[key] = {
            "symbol": "TESTUSDT",
            "signal": 1,
            "entry_price": 100.0,
            "notional": 3000.0,
            "bars_held": 10,
            "peak_gain": 0.02,
            "best_pnl": 0.02,
            "worst_pnl": -0.005,
            "server_sl_order_id": None,
            "server_tp_order_id": None,
            "sl": 0.01,
            "tp": 0.12,
            "tf": "15m",
            "position_side": "LONG",
            "trail_active": False,
            "opened_at": "2026-01-01T00:00:00+00:00",
            "strategy_type": "vr_squeeze",
        }
        # Fake order placed successfully, but strategy.close_position returns None (race)
        scalper.client.futures_create_order = MagicMock(
            return_value={"orderId": 9001, "avgPrice": "100.0", "executedQty": "30.0", "status": "FILLED"}
        )
        scalper.strategy.close_position = MagicMock(return_value=None)
        result = scalper._close_position("TESTUSDT:LONG", "max_hold", 100.0)
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# max_notional cap tests (prevents -2027)
# ─────────────────────────────────────────────────────────────────────────────


class TestMaxNotionalCap:
    """positions.py: notional is capped at symbol's Binance max_notional limit."""

    def test_cw_notional_capped_when_exceeds_max(self):
        """CW-Tier 1.5x notional이 max_notional 초과 시 캡핑되어 _place_order에 전달."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
            base_leverage=30,
            max_concurrent=10,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        # CW score=2 → 1.5x → notional = $3000 * 1.5 = $4500
        # max_notional = $4000 → should cap to $4000
        scalper._symbol_info_cache["TESTUSDT"]["max_notional"] = 4000.0
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=2, strategy_type="vr_squeeze")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # capped notional=$4000 @ price=100.0 → qty=40
        assert abs(qty - 40.0) < 0.1, f"Expected ~40 (capped), got {qty}"

    def test_notional_not_capped_when_within_limit(self):
        """max_notional 한도 내이면 캡핑 없이 그대로 사용."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
            base_leverage=30,
            max_concurrent=10,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        # CW score=2 → 1.5x → notional = $4500, max_notional = $10000 → no cap
        scalper._symbol_info_cache["TESTUSDT"]["max_notional"] = 10000.0
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=2, strategy_type="vr_squeeze")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # notional=$4500 @ price=100.0 → qty=45
        assert abs(qty - 45.0) < 0.1, f"Expected ~45 (uncapped), got {qty}"

    def test_no_cap_when_max_notional_absent(self):
        """max_notional=None (sentinel: fetch failed or unavailable) → 캡핑 없이 원래 notional 사용."""
        cfg = VolStraddleConfig(
            dynamic_capital_enabled=False,
            soft_equity_scale=False,
            cw_enabled=True,
            cw_multipliers=[0.5, 1.0, 1.5, 2.0, 2.5],
            base_leverage=30,
            max_concurrent=10,
        )
        scalper = _make_vs_scalper(vs_config=cfg, balance=1000.0)
        # max_notional=None sentinel means "bracket fetch failed" → no cap applied.
        # _make_vs_scalper already sets None; this is explicit for documentation.
        scalper._symbol_info_cache["TESTUSDT"]["max_notional"] = None
        _mock_open(scalper)
        _set_signal_meta(scalper, conviction_score=2, strategy_type="vr_squeeze")
        _open(scalper, 1000.0)

        qty = float(scalper._place_order.call_args[0][2])
        # original notional=$4500 @ price=100.0 → qty=45
        assert abs(qty - 45.0) < 0.1, f"Expected ~45 (no cap), got {qty}"
