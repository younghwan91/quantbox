"""DD halt profitable position protection tests.

Tests for the selective close behavior added in v14.40.0:
- Profitable positions are KEPT (trail/SL protection active)
- Losing positions are CLOSED
- Klines unavailable → precautionary close
"""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock

from tests.conftest import _build_vs_scalper


ENTRY_PRICE = 100.0


def _inject_position(scalper, key: str, signal: int, entry_price: float = ENTRY_PRICE) -> dict:
    pos_side = "LONG" if signal == 1 else "SHORT"
    pos = {
        "symbol": key.split(":")[0],
        "signal": signal,
        "entry_price": entry_price,
        "notional": 3000.0,
        "bars_held": 10,
        "peak_gain": 0.02,
        "best_pnl": 0.02,
        "worst_pnl": -0.005,
        "server_sl_order_id": "sl_111",
        "server_tp_order_id": "tp_222",
        "sl": 0.01,
        "tp": 0.12,
        "tf": "15m",
        "position_side": pos_side,
        "trail_active": False,
        "opened_at": "2026-01-01T00:00:00+00:00",
        "strategy_type": "vr_directional",
    }
    scalper.strategy._positions[key] = pos
    return pos


def _setup_dd_halt(scalper, balance: float = 800.0):
    """Configure scalper in DD halt state."""
    scalper._drawdown_halt = True
    scalper._high_water_mark = 1000.0
    scalper.get_balance = MagicMock(return_value=balance)
    scalper._cfg.max_drawdown_pct = 0.10  # 10% threshold, 20% actual DD
    scalper._close_position = MagicMock(return_value=True)
    scalper._sync_server_orders = MagicMock()
    scalper._process_bot_commands = MagicMock()
    scalper._reconcile_positions = MagicMock()
    scalper._stop_requested = False
    scalper.paper_mode = True  # skip reconciliation API calls


class TestDDHaltProfitableSkip:
    """DD halt keeps profitable positions, closes losing ones."""

    def test_profitable_long_kept(self):
        """LONG with price > entry → uPnL > 0 → NOT closed."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:LONG", signal=1, entry_price=100.0)
        # Price at 105 → LONG uPnL = (105/100 - 1) * 1 = +5%
        scalper.download_klines = MagicMock(return_value=np.full(2, 105.0))

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_not_called()

    def test_losing_long_closed(self):
        """LONG with price < entry → uPnL < 0 → closed."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:LONG", signal=1, entry_price=100.0)
        # Price at 95 → LONG uPnL = (95/100 - 1) * 1 = -5%
        scalper.download_klines = MagicMock(return_value=np.full(2, 95.0))

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:LONG", "drawdown_halt", 95.0)

    def test_profitable_short_kept(self):
        """SHORT with price < entry → uPnL > 0 → NOT closed."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:SHORT", signal=-1, entry_price=100.0)
        # Price at 95 → SHORT uPnL = (95/100 - 1) * -1 = +5%
        scalper.download_klines = MagicMock(return_value=np.full(2, 95.0))

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_not_called()

    def test_losing_short_closed(self):
        """SHORT with price > entry → uPnL < 0 → closed."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:SHORT", signal=-1, entry_price=100.0)
        # Price at 105 → SHORT uPnL = (105/100 - 1) * -1 = -5%
        scalper.download_klines = MagicMock(return_value=np.full(2, 105.0))

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:SHORT", "drawdown_halt", 105.0)

    def test_klines_unavailable_closes_as_precaution(self):
        """Klines download fails → close position as precaution."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:LONG", signal=1, entry_price=100.0)
        scalper.download_klines = MagicMock(return_value=None)

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:LONG", "drawdown_halt", 0.0)

    def test_mixed_positions_selective_close(self):
        """Multiple positions: profitable kept, losing closed."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "AAUSDT:LONG", signal=1, entry_price=100.0)  # will be profitable
        _inject_position(scalper, "BBUSDT:SHORT", signal=-1, entry_price=100.0)  # will be losing

        def klines_side_effect(symbol, *args, **kwargs):
            if symbol == "AAUSDT":
                return np.full(2, 110.0)  # LONG profitable
            elif symbol == "BBUSDT":
                return np.full(2, 110.0)  # SHORT losing
            return None

        scalper.download_klines = MagicMock(side_effect=klines_side_effect)

        scalper._cycle(["AAUSDT", "BBUSDT"])

        # Only BBUSDT:SHORT should be closed (losing)
        scalper._close_position.assert_called_once_with("BBUSDT:SHORT", "drawdown_halt", 110.0)

    def test_breakeven_position_not_kept(self):
        """Position at exactly entry price → uPnL = 0 → closed (not > 0)."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:LONG", signal=1, entry_price=100.0)
        scalper.download_klines = MagicMock(return_value=np.full(2, 100.0))

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:LONG", "drawdown_halt", 100.0)

    def test_entry_price_zero_safe(self):
        """Position with entry_price=0 → uPnL defaults to 0.0 → closed (not > 0)."""
        scalper = _build_vs_scalper()
        _setup_dd_halt(scalper)
        _inject_position(scalper, "TESTUSDT:LONG", signal=1, entry_price=0.0)
        scalper.download_klines = MagicMock(return_value=np.full(2, 100.0))

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:LONG", "drawdown_halt", 100.0)
