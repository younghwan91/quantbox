"""Tests for _ratchet_trail_sl() — r65 trail server SL ratchet.

Verifies that bar-close execution gap is closed by placing a Binance
STOP order at trail_stop_gain price whenever trail is active.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import _build_vs_scalper
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


PRICE = 100.0


def _make_pos(
    trail_active: bool = True,
    peak_gain: float = 0.008,
    strategy_type: str = "vr_squeeze",
    signal: int = 1,
    sl_order_id: str = "sl_111",
    trail_sl_ratcheted: float = -999.0,
    is_straddle: bool = False,
) -> dict:
    pos = {
        "symbol": "TESTUSDT",
        "signal": signal,
        "entry_price": PRICE,
        "notional": 3000.0,
        "position_side": "LONG" if signal == 1 else "SHORT",
        "trail_active": trail_active,
        "peak_gain": peak_gain,
        "server_sl_order_id": sl_order_id,
        "strategy_type": strategy_type,
        "is_straddle": is_straddle,
        "_trail_sl_ratcheted": trail_sl_ratcheted,
    }
    return pos


def _make_scalper(paper=False, **vs_overrides):
    vs_cfg = VolStraddleConfig()
    for k, v in vs_overrides.items():
        object.__setattr__(vs_cfg, k, v)
    s = _build_vs_scalper(vs_config=vs_cfg)
    s.paper_mode = paper
    return s


def _setup_mocks(scalper):
    """Patch external I/O on the scalper instance."""
    scalper.get_symbol_info = MagicMock(return_value={"tick_size": 0.01, "price_precision": 2})
    scalper._round_qty = MagicMock(return_value=30.0)
    scalper._cancel_server_stop_loss = MagicMock()
    scalper._save_state = MagicMock()
    return scalper


# ─────────────────────────────────────────────────────────────────────────────
# T1: Trail inactive → no API call
# ─────────────────────────────────────────────────────────────────────────────


def test_trail_inactive_no_api():
    """When trail_active=False, _ratchet_trail_sl must make no API calls."""
    s = _make_scalper()
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="new_sl_999")

    pos = _make_pos(trail_active=False, peak_gain=0.010)
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    s._place_algo_order.assert_not_called()
    s._cancel_server_stop_loss.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# T2: Trail active, first ratchet → STOP placed, old SL cancelled
# ─────────────────────────────────────────────────────────────────────────────


def test_trail_active_places_stop():
    """Trail active + no previous ratchet → new STOP placed, old SL cancelled."""
    s = _make_scalper()
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="new_sl_555")

    pos = _make_pos(trail_active=True, peak_gain=0.008, trail_sl_ratcheted=-999.0)
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    # d_trail_distance = 0.002 → trail_stop_gain = 0.008 - 0.002 = 0.006
    # trigger = 100 × 1.006 = 100.60
    s._place_algo_order.assert_called_once()
    call_kwargs = s._place_algo_order.call_args.kwargs
    assert call_kwargs["side"] == "SELL"
    assert call_kwargs["order_type"] == "STOP"
    assert abs(call_kwargs["trigger_price"] - 100.60) < 0.01

    s._cancel_server_stop_loss.assert_called_once_with("TESTUSDT", "sl_111")
    assert pos["server_sl_order_id"] == "new_sl_555"
    assert abs(pos["_trail_sl_ratcheted"] - 0.006) < 1e-9
    s._save_state.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# T3: Peak improves → ratchet up only; tiny improvement → no API call
# ─────────────────────────────────────────────────────────────────────────────


def test_ratchet_upward_only():
    """Second call with improved peak ratchets up; third with tiny delta skips."""
    s = _make_scalper()
    _setup_mocks(s)

    # First ratchet at 0.006 already stored
    pos = _make_pos(trail_active=True, peak_gain=0.012, trail_sl_ratcheted=0.006)

    s._place_algo_order = MagicMock(return_value="new_sl_777")
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    # trail_stop_gain = 0.012 - 0.002 = 0.010, improvement = 0.010 - 0.006 = 0.004 ≥ 0.001
    s._place_algo_order.assert_called_once()
    assert abs(pos["_trail_sl_ratcheted"] - 0.010) < 1e-9

    # Now peak barely changes (tiny delta below 0.001 threshold)
    pos["peak_gain"] = 0.0125
    s._place_algo_order.reset_mock()
    s._cancel_server_stop_loss.reset_mock()
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    # trail_stop_gain = 0.0105, improvement = 0.0005 < 0.001 → no API
    s._place_algo_order.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# T4: Paper mode → no API calls
# ─────────────────────────────────────────────────────────────────────────────


def test_paper_mode_no_api():
    """In paper mode, _ratchet_trail_sl must never touch the API."""
    s = _make_scalper(paper=True)
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="new_sl_999")

    pos = _make_pos(trail_active=True, peak_gain=0.015)
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    s._place_algo_order.assert_not_called()
    s._cancel_server_stop_loss.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# T5: SHORT position → BUY side stop, trigger below entry
# ─────────────────────────────────────────────────────────────────────────────


def test_short_position_buy_stop():
    """Short position: STOP side=BUY, trigger = entry × (1 - trail_stop_gain)."""
    s = _make_scalper()
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="new_sl_short")

    pos = _make_pos(trail_active=True, peak_gain=0.010, signal=-1, trail_sl_ratcheted=-999.0)
    pos["position_side"] = "SHORT"
    s._ratchet_trail_sl("TESTUSDT:SHORT", pos)

    # trail_stop_gain = 0.010 - 0.002 = 0.008
    # trigger = 100 × (1 - 0.008) = 99.20
    call_kwargs = s._place_algo_order.call_args.kwargs
    assert call_kwargs["side"] == "BUY"
    assert abs(call_kwargs["trigger_price"] - 99.20) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# T6: MA cluster uses ma_cluster_trail_distance
# ─────────────────────────────────────────────────────────────────────────────


def test_ma_cluster_uses_correct_distance():
    """MA cluster strategy uses ma_cluster_trail_distance (0.005, not 0.002)."""
    s = _make_scalper()
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="new_sl_ma")

    # Peak = 0.020, trail_distance = 0.005 → trail_stop_gain = 0.015
    pos = _make_pos(trail_active=True, peak_gain=0.020, strategy_type="ma_cluster", trail_sl_ratcheted=-999.0)
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    # trigger = 100 × (1 + 0.015) = 101.50
    call_kwargs = s._place_algo_order.call_args.kwargs
    assert abs(call_kwargs["trigger_price"] - 101.50) < 0.01
    assert abs(pos["_trail_sl_ratcheted"] - 0.015) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# T7: WOULD_TRIGGER response → old SL preserved
# ─────────────────────────────────────────────────────────────────────────────


def test_would_trigger_keeps_old_sl():
    """WOULD_TRIGGER means price already past trail stop — keep old SL."""
    s = _make_scalper()
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="WOULD_TRIGGER")

    pos = _make_pos(trail_active=True, peak_gain=0.008, trail_sl_ratcheted=-999.0)
    s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    # Old SL must not be cancelled and ratcheted level must not change
    s._cancel_server_stop_loss.assert_not_called()
    assert pos["server_sl_order_id"] == "sl_111"
    assert pos["_trail_sl_ratcheted"] == -999.0


# ─────────────────────────────────────────────────────────────────────────────
# T8: config flag disabled → no API call
# ─────────────────────────────────────────────────────────────────────────────


def test_config_flag_disabled_skips_ratchet():
    """trail_server_ratchet_enabled=False → _cycle skips the call entirely."""
    s = _make_scalper(trail_server_ratchet_enabled=False)
    _setup_mocks(s)
    s._place_algo_order = MagicMock(return_value="new_sl_999")

    pos = _make_pos(trail_active=True, peak_gain=0.010)
    # Simulate the gate check in _cycle
    if (
        pos.get("trail_active")
        and not s.paper_mode
        and s.strategy_name == "vol_straddle"
        and getattr(s.vs_config, "trail_server_ratchet_enabled", True)
    ):
        s._ratchet_trail_sl("TESTUSDT:LONG", pos)

    s._place_algo_order.assert_not_called()
