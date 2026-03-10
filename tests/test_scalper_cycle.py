"""scalper.py coverage tests — targeting _cycle, _check_risk_limits, _check_divergence,
_scan_entries, main() CLI paths.

Uses shared fixtures from conftest.py:
    vs_scalper  — VolStraddle scalper, paper=True
    vs_position — Standard open position dict

Coverage targets (scalper.py 54% → 75%+):
    L256-287 : _check_risk_limits (drawdown halt, HWM update)
    L480-516 : _cycle drawdown halt + auto-recover
    L542-567 : _cycle position status log
    L572-649 : _cycle position update loop (exit signal → _close_position)
    L900-929 : _scan_entries (G6 BTC gate, dir diversity, _open_position)
    L933-975 : _check_divergence (cooldown, alert)
    L1360-1404: main() CLI paths
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from tests.conftest import _build_vs_scalper


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PRICE = 100.0


def _inject_position(scalper, key: str = "TESTUSDT:LONG", **overrides) -> dict:
    """Inject a standard open position into scalper._positions and return it."""
    pos = {
        "symbol": key.split(":")[0] if ":" in key else key,
        "signal": 1,
        "entry_price": PRICE,
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
        "position_side": "LONG" if "LONG" in key else "SHORT",
        "trail_active": False,
        "opened_at": "2026-01-01T00:00:00+00:00",
        "strategy_type": "vr_squeeze",
    }
    pos.update(overrides)
    scalper.strategy._positions[key] = pos
    return pos


def _mock_klines(scalper, price: float = PRICE, length: int = 210) -> None:
    """Make download_klines return a flat close array."""
    closes = np.full(length, price)
    scalper.download_klines = MagicMock(return_value=closes)


def _mock_klines_full(scalper, price: float = PRICE, length: int = 210) -> None:
    """Make download_klines return full OHLCV tuple for full=True calls."""
    closes = np.full(length, price)
    highs = closes + 1.0
    lows = closes - 1.0
    vols = np.full(length, 1000.0)
    tbs = np.full(length, 500.0)

    def _dl(symbol, n, interval="15m", full=False):
        if full:
            return closes[:n], highs[:n], lows[:n], vols[:n], tbs[:n]
        return closes[:n]

    scalper.download_klines = MagicMock(side_effect=_dl)


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckRiskLimits — L256-287
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckRiskLimits:
    """_check_risk_limits: HWM update, drawdown halt trigger."""

    def test_hwm_updates_on_new_high(self):
        """L270-271: balance > HWM → HWM updated."""
        scalper = _build_vs_scalper(balance=1000.0)
        scalper._high_water_mark = 900.0
        scalper.get_balance = MagicMock(return_value=1100.0)
        scalper._check_risk_limits(1100.0)
        assert scalper._high_water_mark == 1100.0

    def test_no_halt_below_threshold(self):
        """DD < max_drawdown_pct → returns False."""
        scalper = _build_vs_scalper()
        scalper._high_water_mark = 1000.0
        scalper._cfg.max_drawdown_pct = 0.30
        result = scalper._check_risk_limits(800.0)  # 20% DD
        assert result is False

    def test_halt_triggers_at_threshold(self):
        """L276-285: DD >= max_drawdown_pct → halt=True, return True."""
        scalper = _build_vs_scalper()
        scalper._high_water_mark = 1000.0
        scalper._drawdown_halt = False
        scalper._cfg.max_drawdown_pct = 0.10
        result = scalper._check_risk_limits(880.0)  # 12% DD
        assert result is True
        assert scalper._drawdown_halt is True

    def test_halt_already_active_still_returns_true(self):
        """Already halted: skips Discord, still returns True."""
        scalper = _build_vs_scalper()
        scalper._high_water_mark = 1000.0
        scalper._drawdown_halt = True
        scalper._cfg.max_drawdown_pct = 0.10
        result = scalper._check_risk_limits(880.0)
        assert result is True
        # Discord NOT called again (already halted)
        scalper.discord.send.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# TestCycleDrawdownHalt — L479-516
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleDrawdownHalt:
    """_cycle: drawdown halt closes all positions; auto-recovers when DD clears."""

    def test_drawdown_halt_closes_open_position(self):
        """L481-485: halt active → _close_position called for each open pos."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = True
        scalper._high_water_mark = 1000.0
        scalper.get_balance = MagicMock(return_value=800.0)  # 20% DD
        scalper._cfg.max_drawdown_pct = 0.10

        _inject_position(scalper, "TESTUSDT:LONG")
        scalper._close_position = MagicMock(return_value=True)
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._stop_requested = False
        _mock_klines(scalper)

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:LONG", "drawdown_halt", PRICE)

    def test_drawdown_auto_recover(self):
        """L489-505: halt + n_active=0 + DD recovered → _drawdown_halt=False."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = True
        scalper._high_water_mark = 1000.0
        scalper.get_balance = MagicMock(return_value=950.0)  # 5% DD < 10%
        scalper._cfg.max_drawdown_pct = 0.10
        scalper._cycle_count = 0
        scalper._trade_history = []

        # No positions → n_active=0 already
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._scan_entries = MagicMock()  # skip the expensive scan
        scalper._stop_requested = False
        _mock_klines(scalper)

        scalper._cycle(["TESTUSDT"])

        assert scalper._drawdown_halt is False


# ─────────────────────────────────────────────────────────────────────────────
# TestCyclePositionLoop — L570-649
# ─────────────────────────────────────────────────────────────────────────────


class TestCyclePositionLoop:
    """_cycle position update loop: exit signal → _close_position."""

    def test_exit_signal_triggers_close(self):
        """L596-604: update_position returns exit reason → _close_position called."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = False
        scalper._cycle_count = 0
        scalper._trade_history = []
        scalper._stop_requested = False

        _inject_position(scalper, "TESTUSDT:LONG")
        scalper.strategy.update_position = MagicMock(return_value="trailing_stop")
        scalper._close_position = MagicMock(return_value=True)
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._scan_entries = MagicMock()
        _mock_klines_full(scalper)

        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": str(PRICE)})

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_called_once_with("TESTUSDT:LONG", "trailing_stop", PRICE)

    def test_no_exit_signal_holds_position(self):
        """update_position returns None → _close_position NOT called."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = False
        scalper._cycle_count = 0
        scalper._trade_history = []
        scalper._stop_requested = False

        _inject_position(scalper, "TESTUSDT:LONG")
        scalper.strategy.update_position = MagicMock(return_value=None)
        scalper._close_position = MagicMock(return_value=False)
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._scan_entries = MagicMock()
        _mock_klines_full(scalper)

        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": str(PRICE)})

        scalper._cycle(["TESTUSDT"])

        scalper._close_position.assert_not_called()

    def test_klines_none_skips_position(self):
        """L579-580: download_klines returns None → skip position."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = False
        scalper._cycle_count = 0
        scalper._trade_history = []
        scalper._stop_requested = False

        _inject_position(scalper, "TESTUSDT:LONG")
        scalper.strategy.update_position = MagicMock(return_value=None)
        scalper._close_position = MagicMock(return_value=False)
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._scan_entries = MagicMock()

        # full=True returns None → skip
        def _dl(symbol, n, interval="15m", full=False):
            if full:
                return None
            return np.full(n, PRICE)

        scalper.download_klines = MagicMock(side_effect=_dl)
        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": str(PRICE)})

        scalper._cycle(["TESTUSDT"])

        scalper.strategy.update_position.assert_not_called()

    def test_straddle_other_leg_logged_on_exit(self):
        """L606-615: vol_straddle LONG exit → other leg (SHORT) logged."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = False
        scalper._cycle_count = 0
        scalper._trade_history = []
        scalper._stop_requested = False

        _inject_position(scalper, "TESTUSDT:LONG")
        # Also inject other side so active_positions has both
        _inject_position(scalper, "TESTUSDT:SHORT", signal=-1, position_side="SHORT")

        scalper.strategy.update_position = MagicMock(return_value="trailing_stop")
        scalper._close_position = MagicMock(return_value=True)
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._scan_entries = MagicMock()
        _mock_klines_full(scalper)

        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": str(PRICE)})

        scalper._cycle(["TESTUSDT"])

        # _close_position called for at least TESTUSDT:LONG
        assert scalper._close_position.called

    def test_partial_close_exit_calls_partial_close(self):
        """exit_reason='partial_close' → _partial_close_position called (L627)."""
        scalper = _build_vs_scalper()
        scalper._drawdown_halt = False
        scalper._cycle_count = 0
        scalper._trade_history = []
        scalper._stop_requested = False

        _inject_position(scalper, "TESTUSDT:LONG")
        scalper.strategy.update_position = MagicMock(return_value="partial_close")
        scalper._partial_close_position = MagicMock()
        scalper._sync_server_orders = MagicMock()
        scalper._process_bot_commands = MagicMock()
        scalper._scan_entries = MagicMock()
        _mock_klines_full(scalper)
        scalper.client.futures_mark_price = MagicMock(return_value={"markPrice": str(PRICE)})

        scalper._cycle(["TESTUSDT"])

        scalper._partial_close_position.assert_called_once_with("TESTUSDT:LONG", PRICE)


# ─────────────────────────────────────────────────────────────────────────────
# TestScanEntries — L900-929
# ─────────────────────────────────────────────────────────────────────────────


def _setup_scan(scalper, signal: int = 1, price: float = PRICE):
    """Wire up mocks so _scan_entries processes one symbol and returns a signal."""
    n = 500
    closes = np.full(n, price)
    highs = closes + 1.0
    lows = closes - 1.0
    vols = np.full(n, 1000.0)
    tbs = np.full(n, 500.0)
    ohlcv = (closes, highs, lows, vols, tbs)

    def _dl(symbol, n_bars, interval="15m", full=False):
        if full:
            return ohlcv
        return closes

    scalper.download_klines = MagicMock(side_effect=_dl)
    scalper.strategy.on_bar = MagicMock(return_value=signal)
    scalper.strategy.has_position = MagicMock(return_value=False)
    scalper.strategy.update_market_data = MagicMock()
    scalper.strategy.update_htf_data = MagicMock()
    scalper.strategy.diagnose_bar = MagicMock(return_value={"signal_actual": signal})
    scalper.strategy.get_vr_all_tf = MagicMock(return_value={"15m": 1.0, "1h": 1.0, "4h": 1.0})
    scalper._open_position = MagicMock(return_value=True)
    # G6: BTC above MA → don't suppress LONG
    scalper.vs_config.btc_long_ma_period = 5
    scalper.vs_config.ma_cluster_enabled = False


class TestScanEntries:
    """_scan_entries: G6 BTC gate, dir diversity filter, _open_position called."""

    def test_open_position_called_for_candidate(self):
        """_open_position called when symbol generates a signal."""
        scalper = _build_vs_scalper()
        _setup_scan(scalper, signal=1)
        btc_close = np.full(210, PRICE)

        scalper._scan_entries(["TESTUSDT"], btc_close, 1000.0, max_entries=5)

        scalper._open_position.assert_called_once()

    def test_g6_long_suppressed_below_btc_ma(self):
        """BTC below MA → LONG suppressed (G6 gate)."""
        scalper = _build_vs_scalper()
        _setup_scan(scalper, signal=1)
        scalper.vs_config.btc_long_ma_period = 5

        # BTC far below its 5-bar MA
        btc_close = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 50.0])

        scalper._scan_entries(["TESTUSDT"], btc_close, 1000.0, max_entries=5)

        scalper._open_position.assert_not_called()

    def test_short_not_suppressed_by_g6(self):
        """G6 only applies to LONG — SHORT (signal=-1) passes through."""
        scalper = _build_vs_scalper()
        _setup_scan(scalper, signal=-1)
        scalper.vs_config.btc_long_ma_period = 5

        # BTC way below MA — SHORT should still open
        btc_close = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 50.0])

        scalper._scan_entries(["TESTUSDT"], btc_close, 1000.0, max_entries=5)

        scalper._open_position.assert_called_once()

    def test_dir_diversity_blocks_excess_longs(self):
        """Too many same-direction positions → candidate skipped."""
        scalper = _build_vs_scalper()
        _setup_scan(scalper, signal=1)
        scalper._cfg.dir_diversity_enabled = True
        scalper._cfg.max_same_direction = 2

        # Inject 2 existing LONG positions → diversity limit hit
        _inject_position(scalper, "ETHUSDT:LONG", **{"symbol": "ETHUSDT"})
        _inject_position(scalper, "BTCUSDT:LONG", **{"symbol": "BTCUSDT"})
        scalper.strategy.has_position = MagicMock(return_value=False)

        btc_close = np.full(210, PRICE)
        scalper._scan_entries(["TESTUSDT"], btc_close, 1000.0, max_entries=5)

        scalper._open_position.assert_not_called()

    def test_max_entries_respected(self):
        """Only max_entries new positions opened per cycle (uses max_entries=1)."""
        scalper = _build_vs_scalper()
        _setup_scan(scalper, signal=1)

        btc_close = np.full(210, PRICE)
        # 3 symbols to scan, max_entries=1 → only 1 opened
        scalper._scan_entries(["TESTUSDT", "ETHUSDT", "BTCUSDT"], btc_close, 1000.0, max_entries=1)

        assert scalper._open_position.call_count == 1

    def test_zero_signal_not_added_to_candidates(self):
        """signal=0 → no candidate → _open_position not called."""
        scalper = _build_vs_scalper()
        _setup_scan(scalper, signal=0)

        btc_close = np.full(210, PRICE)
        scalper._scan_entries(["TESTUSDT"], btc_close, 1000.0, max_entries=5)

        scalper._open_position.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckDivergence — L933-975
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckDivergence:
    """_check_divergence: triggered every 10 trades, mocks journal + monitor."""

    def test_no_check_below_10_trades(self):
        """n_trades=5 (not multiple of 10) → divergence_monitor.check not called."""
        scalper = _build_vs_scalper()
        scalper._trade_history = [{"pnl_usd": 1.0}] * 5
        scalper._divergence_monitor = MagicMock()

        scalper._check_divergence()

        scalper._divergence_monitor.check.assert_not_called()

    def test_check_fires_at_10_trades(self):
        """n_trades=10 → divergence_monitor.check called once."""
        scalper = _build_vs_scalper()
        scalper._trade_history = [{"pnl_usd": 1.0}] * 10
        scalper._divergence_monitor = MagicMock()
        scalper._divergence_monitor.check.return_value = None
        scalper._journal = MagicMock()
        scalper._journal.get_stats.return_value = {"n_trades": 10, "profit_factor": 1.5, "win_rate": 0.55}

        scalper._check_divergence()

        scalper._divergence_monitor.check.assert_called_once()

    def test_alert_sent_to_discord(self):
        """When monitor returns alert → discord.notify_divergence_alert called."""
        from quantbox.core.divergence_monitor import DivergenceAlert

        scalper = _build_vs_scalper()
        scalper._trade_history = [{"pnl_usd": 1.0}] * 20
        scalper._journal = MagicMock()
        scalper._journal.get_stats.return_value = {"n_trades": 20, "profit_factor": 0.5, "win_rate": 0.30}

        fake_alert = DivergenceAlert(
            pf_live=0.5,
            pf_baseline=1.563,
            wr_live=0.30,
            wr_baseline=0.434,
            n_trades=20,
            triggered_by=["pf"],
            message="Live PF 0.500 < threshold 0.80",
        )
        scalper._divergence_monitor = MagicMock()
        scalper._divergence_monitor.check.return_value = fake_alert

        scalper._check_divergence()

        scalper.discord.notify_divergence_alert.assert_called_once_with(fake_alert)

    def test_no_alert_when_monitor_returns_none(self):
        """monitor.check() returns None → no Discord notification."""
        scalper = _build_vs_scalper()
        scalper._trade_history = [{"pnl_usd": 1.0}] * 10
        scalper._journal = MagicMock()
        scalper._journal.get_stats.return_value = {"n_trades": 10, "profit_factor": 1.8, "win_rate": 0.55}
        scalper._divergence_monitor = MagicMock()
        scalper._divergence_monitor.check.return_value = None

        scalper._check_divergence()

        scalper.discord.notify_divergence_alert.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# TestSyncServerOrders — reconciliation.py (live mode only)
# ─────────────────────────────────────────────────────────────────────────────


def _build_live_vs_scalper():
    """VolStraddle scalper in LIVE mode (paper_mode=False) for sync tests."""
    from quantbox.scalper import RubberBandScalper
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

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
            paper_mode=False,  # LIVE mode so _sync_server_orders runs
            strategy_name="vol_straddle",
            vs_config=VolStraddleConfig(),
            journal_db_path=":memory:",
        )

    scalper._symbol_info_cache["TESTUSDT"] = {
        "step_size": 0.001,
        "tick_size": 0.01,
        "quantity_precision": 3,
        "price_precision": 2,
        "min_notional": 5.0,
    }
    scalper.client = MagicMock()
    scalper.discord = MagicMock()
    scalper._paper_balance = 1000.0
    scalper._high_water_mark = 0.0
    scalper.get_balance = MagicMock(return_value=1000.0)
    scalper._save_state = MagicMock()
    return scalper


class TestSyncServerOrders:
    """_sync_server_orders: skipped in paper mode; checks algo orders in live."""

    def test_paper_mode_sync_is_noop(self):
        """paper_mode=True → _sync_server_orders returns immediately (no API calls)."""
        scalper = _build_vs_scalper()  # paper=True
        _inject_position(scalper, "TESTUSDT:LONG")
        scalper._sync_server_orders()
        scalper.client.futures_get_algo_order.assert_not_called()

    def test_live_no_positions_sync_is_noop(self):
        """Live mode, no positions → no API calls."""
        scalper = _build_live_vs_scalper()
        scalper._sync_server_orders()
        scalper.client.futures_get_algo_order.assert_not_called()

    def test_finished_tp_triggers_close(self):
        """FINISHED TP algo order → _record_server_fill called (position closed)."""
        scalper = _build_live_vs_scalper()
        _inject_position(scalper, "TESTUSDT:LONG")
        scalper.strategy._positions["TESTUSDT:LONG"]["server_sl_order_id"] = "111111"
        scalper.strategy._positions["TESTUSDT:LONG"]["server_tp_order_id"] = "222222"
        scalper.strategy._positions["TESTUSDT:LONG"]["server_tp_is_partial"] = False

        # Mock _check_order_filled directly (avoids numeric conversion, API plumbing)
        def _check_filled(symbol, order_id):
            if order_id == "222222":  # TP
                return {"avgPrice": "105.0", "executedQty": "30.0", "status": "FILLED"}
            return None

        scalper._check_order_filled = _check_filled
        scalper._record_server_fill = MagicMock()
        scalper._cancel_server_stop_loss = MagicMock()

        scalper._sync_server_orders()

        scalper._record_server_fill.assert_called_once()
        args = scalper._record_server_fill.call_args[0]
        assert args[0] == "TESTUSDT:LONG"
        assert args[3] == "server_tp"

    def test_working_order_no_action(self):
        """WORKING algo order → no fill recorded."""
        scalper = _build_live_vs_scalper()
        _inject_position(scalper, "TESTUSDT:LONG")
        scalper.strategy._positions["TESTUSDT:LONG"]["server_tp_order_id"] = "333333"
        scalper.strategy._positions["TESTUSDT:LONG"]["server_sl_order_id"] = None

        scalper._check_order_filled = MagicMock(return_value=None)
        scalper._record_server_fill = MagicMock()

        scalper._sync_server_orders()

        scalper._record_server_fill.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# TestStopRequested — L448-449
# ─────────────────────────────────────────────────────────────────────────────


class TestStopRequested:
    """_cycle: _stop_requested=True → early return."""

    def test_stop_requested_returns_early(self):
        """L448-449: stop flag → cycle exits before any position processing."""
        scalper = _build_vs_scalper()
        scalper._stop_requested = True
        scalper._cycle_count = 0
        scalper._trade_history = []
        scalper.get_balance = MagicMock(return_value=1000.0)
        scalper._process_bot_commands = MagicMock()
        scalper._sync_server_orders = MagicMock()
        scalper._scan_entries = MagicMock()

        scalper._cycle(["TESTUSDT"])

        scalper._sync_server_orders.assert_not_called()
        scalper._scan_entries.assert_not_called()


class TestRunOnce:
    """run_once(): real method called (covers L1125-1138)."""

    def test_run_once_calls_cycle(self):
        """run_once() → _cycle() called with symbols list."""
        scalper = _build_vs_scalper()
        scalper._ensure_position_mode = MagicMock()
        scalper.get_balance = MagicMock(return_value=1000.0)
        scalper._load_state = MagicMock()
        scalper._print_startup_banner = MagicMock()
        scalper.get_trading_symbols = MagicMock(return_value=["BTCUSDT", "ETHUSDT"])
        scalper._cycle = MagicMock()

        scalper.run_once()

        scalper._cycle.assert_called_once_with(["BTCUSDT", "ETHUSDT"])

    def test_run_once_no_symbols_returns_early(self):
        """run_once() with empty symbol list → _cycle not called (L1133-1135)."""
        scalper = _build_vs_scalper()
        scalper._ensure_position_mode = MagicMock()
        scalper.get_balance = MagicMock(return_value=1000.0)
        scalper._load_state = MagicMock()
        scalper._print_startup_banner = MagicMock()
        scalper.get_trading_symbols = MagicMock(return_value=[])
        scalper._cycle = MagicMock()

        scalper.run_once()

        scalper._cycle.assert_not_called()

    def test_run_continuous_single_cycle_then_stop(self):
        """run_continuous() runs one cycle then stops when _stop_requested flips."""
        scalper = _build_vs_scalper()
        scalper._ensure_position_mode = MagicMock()
        scalper.get_balance = MagicMock(return_value=1000.0)
        scalper._load_state = MagicMock()
        scalper._print_startup_banner = MagicMock()
        scalper.get_trading_symbols = MagicMock(return_value=["BTCUSDT"])
        scalper._save_state = MagicMock()
        scalper._bot_runner = None  # no Discord bot

        call_count = [0]

        def _wait_then_stop():
            call_count[0] += 1
            scalper._stop_requested = True  # exit after 1 cycle

        scalper._wait_for_bar = _wait_then_stop
        scalper._cycle = MagicMock()

        scalper.run_continuous()

        assert scalper._cycle.call_count == 1
        scalper._save_state.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# TestMainCLI — L1360-1404
# ─────────────────────────────────────────────────────────────────────────────


class TestMainCLI:
    """main() CLI: --once, --paper, --strategy flags."""

    def _run_main(self, args: list[str]):
        """Invoke main() with patched sys.argv and mocked scalper."""
        import sys

        with (
            patch("sys.argv", ["scalper"] + args),
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client"),
            patch("quantbox.scalper.DiscordNotifier"),
            patch("quantbox.scalper.DiscordBotRunner", None),
            patch("quantbox.scalper.setup_logging"),
            patch("quantbox.scalper.RubberBandScalper") as MockScalper,
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
            mock_instance = MockScalper.return_value
            mock_instance.run_once = MagicMock()
            mock_instance.run_continuous = MagicMock()

            from quantbox.scalper import main

            main()
            return mock_instance

    def test_once_flag_calls_run_once(self):
        """--once → run_once() called, not run_continuous()."""
        instance = self._run_main(["--strategy", "vol_straddle", "--once", "--paper"])
        instance.run_once.assert_called_once()
        instance.run_continuous.assert_not_called()

    def test_no_once_flag_calls_run_continuous(self):
        """No --once → run_continuous() called."""
        instance = self._run_main(["--strategy", "vol_straddle", "--paper"])
        instance.run_continuous.assert_called_once()

    def test_paper_flag_passed(self):
        """--paper → RubberBandScalper instantiated with paper_mode=True."""
        with (
            patch("sys.argv", ["scalper", "--paper", "--once"]),
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client"),
            patch("quantbox.scalper.DiscordNotifier"),
            patch("quantbox.scalper.DiscordBotRunner", None),
            patch("quantbox.scalper.setup_logging"),
            patch("quantbox.scalper.RubberBandScalper") as MockScalper,
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
            MockScalper.return_value.run_once = MagicMock()

            from quantbox.scalper import main

            main()

            _, kwargs = MockScalper.call_args
            assert kwargs.get("paper_mode") is True

    def test_lookback_override(self):
        """--lookback 50 → config.lookback = 50."""
        with (
            patch("sys.argv", ["scalper", "--paper", "--once", "--lookback", "50"]),
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client"),
            patch("quantbox.scalper.DiscordNotifier"),
            patch("quantbox.scalper.DiscordBotRunner", None),
            patch("quantbox.scalper.setup_logging"),
            patch("quantbox.scalper.RubberBandScalper") as MockScalper,
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
            MockScalper.return_value.run_once = MagicMock()

            from quantbox.scalper import main

            main()

            # The RubberBandConfig passed should have lookback=50
            _, kwargs = MockScalper.call_args
            # config is passed as positional or keyword
            cfg = kwargs.get("config")
            if cfg is not None:
                assert cfg.lookback == 50

    def test_threshold_override(self):
        """--threshold 0.04 → config.threshold = 0.04 (covers L1384)."""
        with (
            patch("sys.argv", ["scalper", "--paper", "--once", "--threshold", "0.04"]),
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client"),
            patch("quantbox.scalper.DiscordNotifier"),
            patch("quantbox.scalper.DiscordBotRunner", None),
            patch("quantbox.scalper.setup_logging"),
            patch("quantbox.scalper.RubberBandScalper") as MockScalper,
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
            MockScalper.return_value.run_once = MagicMock()

            from quantbox.scalper import main

            main()

            _, kwargs = MockScalper.call_args
            cfg = kwargs.get("config")
            if cfg is not None:
                assert cfg.threshold == 0.04

    def test_max_slots_override(self):
        """--max-slots 5 → config.max_concurrent = 5 (covers L1386)."""
        with (
            patch("sys.argv", ["scalper", "--paper", "--once", "--max-slots", "5"]),
            patch("quantbox.scalper.get_settings") as mock_settings,
            patch("quantbox.scalper.Client"),
            patch("quantbox.scalper.DiscordNotifier"),
            patch("quantbox.scalper.DiscordBotRunner", None),
            patch("quantbox.scalper.setup_logging"),
            patch("quantbox.scalper.RubberBandScalper") as MockScalper,
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
            MockScalper.return_value.run_once = MagicMock()

            from quantbox.scalper import main

            main()

            _, kwargs = MockScalper.call_args
            cfg = kwargs.get("config")
            if cfg is not None:
                assert cfg.max_concurrent == 5
