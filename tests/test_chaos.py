"""T3-4: Chaos engineering tests — simulate API failures mid-operation.

Each test injects a failure at a specific API boundary and asserts that:
- No unhandled exception propagates
- Internal state remains consistent
- Correct sentinel values (None / False) are returned

Tests:
1. balance API 50% failure → _balance_fetch_ok=False
2. futures_position_information timeout → _reconcile_positions() safe
3. Order placement API failure → _place_order() returns None
4. Algo order API failure → _place_server_stop_loss() returns None
5. State save failure → _save_state() does not crash
6. All APIs fail simultaneously → full _cycle() completes without exception
"""

from unittest.mock import MagicMock, patch

from tests.conftest import _build_vs_scalper
from tests.test_server_oco import _make_scalper


# ─────────────────────────── Test 1: Balance API failure ──────────


def test_balance_fetch_failure_sets_flag() -> None:
    """When futures_account_balance raises, get_balance() sets _balance_fetch_ok=False.

    T1-4 guard: stale balance flag blocks new entries next cycle.
    """
    from quantbox.scalper import RubberBandScalper

    sc = _build_vs_scalper()
    # Restore the real get_balance and switch to live mode
    sc.get_balance = RubberBandScalper.get_balance.__get__(sc)
    sc.paper_mode = False
    sc.client.futures_account_balance.side_effect = Exception("connection timeout")

    result = sc.get_balance()

    assert sc._balance_fetch_ok is False
    # Should return 0.0 (no last_known_balance on first failure)
    assert result == 0.0


def test_balance_fetch_ok_is_true_in_paper_mode() -> None:
    """Paper mode always sets _balance_fetch_ok=True (no API call)."""
    sc = _build_vs_scalper()
    # Restore real get_balance; paper_mode stays True
    from quantbox.scalper import RubberBandScalper

    sc.get_balance = RubberBandScalper.get_balance.__get__(sc)
    # paper_mode is already True from _build_vs_scalper
    result = sc.get_balance()
    assert sc._balance_fetch_ok is True
    assert result == 1000.0  # _paper_balance


# ─────────────────────────── Test 2: Position info timeout ─────────


def test_reconcile_positions_handles_api_timeout() -> None:
    """futures_position_information raising must not crash _reconcile_positions."""
    sc = _build_vs_scalper()
    sc.paper_mode = False
    sc.client.futures_position_information.side_effect = Exception("read timeout")

    # Inject a position to ensure active_positions is not empty
    sc.strategy._positions["TESTUSDT:LONG"] = {
        "symbol": "TESTUSDT",
        "signal": 1,
        "entry_price": 100.0,
        "notional": 3000.0,
        "position_side": "LONG",
    }

    positions_before = set(sc.strategy.active_positions.keys())
    sc._reconcile_positions()  # Must not raise
    positions_after = set(sc.strategy.active_positions.keys())

    assert positions_before == positions_after, "Positions changed despite API timeout"


# ─────────────────────────── Test 3: Order placement failure ───────


def test_place_order_returns_none_on_api_failure() -> None:
    """When futures_orderbook_ticker raises, _place_order() returns None (not exception).

    In paper mode, _place_order() calls futures_orderbook_ticker to simulate fill price.
    """
    sc = _build_vs_scalper()  # paper_mode=True
    sc.client.futures_orderbook_ticker.side_effect = Exception("network error")

    result = sc._place_order("TESTUSDT", "BUY", 30.0)
    assert result is None, f"Expected None, got {result!r}"


def test_place_order_returns_none_on_chaser_exception() -> None:
    """Live mode: chaser raising is caught and returns None."""
    sc = _build_vs_scalper()
    sc.paper_mode = False
    # Replace _chaser with a MagicMock whose chase_limit_order raises
    sc._chaser = MagicMock()
    sc._chaser.chase_limit_order.side_effect = Exception("exchange down")
    # Prevent leverage-change call from failing before chaser is reached
    sc.client.futures_change_leverage.return_value = {"leverage": 30}

    result = sc._place_order("TESTUSDT", "BUY", 30.0, leverage=30)
    assert result is None, f"Expected None on chaser exception, got {result!r}"


# ─────────────────────────── Test 4: Algo order API failure ────────


def test_place_server_stop_loss_returns_none_on_api_failure() -> None:
    """When _request_futures_api raises (non-2021), _place_server_stop_loss returns None."""
    sc = _make_scalper(paper=False)
    sc.discord = MagicMock()
    sc.get_symbol_info = MagicMock(return_value={"tick_size": 0.01, "price_precision": 2})
    sc.client._request_futures_api.side_effect = Exception("API unavailable")

    result = sc._place_server_stop_loss(
        symbol="TESTUSDT",
        signal=1,
        quantity=30.0,
        entry_price=100.0,
        move_size=0.03,
        stop_mult=1.0,
        position_side="LONG",
    )
    assert result is None, f"Expected None on API failure, got {result!r}"


def test_place_server_stop_loss_returns_would_trigger_on_2021() -> None:
    """When API raises -2021, _place_server_stop_loss returns 'WOULD_TRIGGER'."""
    sc = _make_scalper(paper=False)
    sc.discord = MagicMock()
    sc.get_symbol_info = MagicMock(return_value={"tick_size": 0.01, "price_precision": 2})
    sc.client._request_futures_api.side_effect = Exception("APIError(code=-2021): would trigger")

    result = sc._place_server_stop_loss(
        symbol="TESTUSDT",
        signal=1,
        quantity=30.0,
        entry_price=100.0,
        move_size=0.03,
        stop_mult=1.0,
        position_side="LONG",
    )
    assert result == "WOULD_TRIGGER", f"Expected WOULD_TRIGGER sentinel, got {result!r}"


# ─────────────────────────── Test 5: State save failure ────────────


def test_save_state_does_not_raise_on_io_error() -> None:
    """When save_scalper_state raises (e.g. disk full), _save_state must not propagate."""
    from quantbox.scalper import RubberBandScalper

    sc = _build_vs_scalper()
    # Restore the real _save_state (conftest mocks it)
    sc._save_state = RubberBandScalper._save_state.__get__(sc)

    with patch("quantbox.scalper.save_scalper_state", side_effect=OSError("disk full")):
        # Must complete silently — exception is caught and logged internally
        sc._save_state()


def test_save_state_does_not_raise_on_permission_error() -> None:
    """PermissionError during state save is handled gracefully."""
    from quantbox.scalper import RubberBandScalper

    sc = _build_vs_scalper()
    sc._save_state = RubberBandScalper._save_state.__get__(sc)

    with patch("quantbox.scalper.save_scalper_state", side_effect=PermissionError("read-only")):
        sc._save_state()


# ─────────────────────────── Test 6: All APIs fail → full cycle ────


def test_full_cycle_survives_all_api_failures() -> None:
    """With all client APIs raising, _cycle() must complete without unhandled exception.

    This validates end-to-end resilience: balance fetch, reconciliation,
    and order placement all fail — the cycle must log errors and continue.
    """
    from quantbox.scalper import RubberBandScalper

    sc = _build_vs_scalper()

    # Make all main API calls fail
    sc.client.futures_account_balance.side_effect = Exception("timeout")
    sc.client.futures_position_information.side_effect = Exception("timeout")
    sc.client.futures_create_order.side_effect = Exception("timeout")
    sc.client.futures_orderbook_ticker.side_effect = Exception("timeout")
    sc.client._request_futures_api.side_effect = Exception("timeout")

    # Restore real get_balance (was mocked to MagicMock in conftest factory)
    sc.get_balance = RubberBandScalper.get_balance.__get__(sc)
    sc.paper_mode = False

    # download_klines returning None is handled gracefully (positions skipped)
    sc.download_klines = MagicMock(return_value=None)

    # Must NOT raise — all errors are caught internally
    sc._cycle([])  # no symbols → no entry scanning
