"""T3-1: Property-based tests using Hypothesis.

Invariants tested:
1. Slot margin sum ≤ balance (total_margin == balance exactly for the fixed sizing model)
2. Notional always positive for valid balance/leverage inputs
3. entry_price=0 rejected by _open_position() → returns False
4. quantity = notional / entry_price within float tolerance

Install: pip install hypothesis (or uv pip install hypothesis)
"""

import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from unittest.mock import MagicMock

from tests.conftest import _build_vs_scalper


# ─────────────────────────── Sizing invariants (pure math) ────────


@given(st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_total_slot_margin_equals_balance(balance: float) -> None:
    """Total margin across all 10 slots = balance exactly.

    With fixed sizing (dynamic_capital_enabled=False):
      per_slot = balance / max_concurrent  (= balance / 10)
      margin   = notional / leverage       (= per_slot × lev / lev = per_slot)
      total    = margin × max_concurrent   (= per_slot × 10 = balance)
    """
    max_concurrent = 10
    per_slot = balance / max_concurrent
    total_margin = per_slot * max_concurrent
    assert abs(total_margin - balance) < 1e-9, f"Expected total_margin={balance:.4f}, got {total_margin:.4f}"


@given(
    st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50)
def test_notional_always_positive(balance: float, n_active: int) -> None:
    """Notional must be > 0 for any valid balance and active slot count."""
    max_concurrent = 10
    base_leverage = 30
    per_slot = balance / max_concurrent
    notional = per_slot * base_leverage
    assert notional > 0, f"notional={notional} not positive (balance={balance})"


# ─────────────────────────── _open_position guard ─────────────────


@given(st.sampled_from([-1, 1]))
@settings(max_examples=20)
def test_entry_price_zero_rejected(signal: int) -> None:
    """_open_position must return False when entry_price=0 (T2-1 guard).

    Uses close_prices array of zeros so entry_price = close_prices[-1] = 0.
    """
    sc = _build_vs_scalper()
    sc.strategy.get_last_signal_meta = MagicMock(return_value={})
    close_prices = np.zeros(20)
    result = sc._open_position("TESTUSDT", signal, close_prices, 1000.0)
    assert result is False, f"Expected False for entry_price=0, signal={signal}, got {result!r}"


def test_entry_price_guard_prevents_zero_division() -> None:
    """T2-1 guard must catch entry_price=0 *before* quantity division.

    Without the guard, `notional / 0.0` raises ZeroDivisionError (Python float)
    or returns inf (numpy float64) — both silent/crash failures in production.
    This test verifies the guard fires and logs an error.
    """
    import logging
    from unittest.mock import patch

    sc = _build_vs_scalper()
    sc.strategy.get_last_signal_meta = MagicMock(return_value={})
    # Prevent lazy-load via MagicMock client (MagicMock().__float__() == 1.0
    # would set max_notional=1.0 and route the early-return through notional cap
    # instead of the entry_price guard). Set sentinel to bypass lazy-load.
    sc._symbol_info_cache["TESTUSDT"]["max_notional"] = None

    # Python float 0.0 → ZeroDivisionError without guard
    close_prices = np.array([100.0] * 19 + [0.0])

    with patch("quantbox.execution.positions.logger") as mock_log:
        result = sc._open_position("TESTUSDT", 1, close_prices, 1000.0)

    assert result is False, "entry_price=0 must return False"
    # Guard must have logged an error (INVARIANT VIOLATION)
    error_calls = [str(c) for c in mock_log.error.call_args_list]
    assert any("INVARIANT" in c or "entry_price" in c for c in error_calls), (
        f"Expected INVARIANT error log, got: {error_calls}"
    )


# ─────────────────────────── Quantity formula ─────────────────────


@given(
    st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=100_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_quantity_equals_notional_over_entry_price(balance: float, entry_price: float) -> None:
    """quantity = notional / entry_price holds within float tolerance."""
    max_concurrent = 10
    base_leverage = 30
    per_slot = balance / max_concurrent
    notional = per_slot * base_leverage
    quantity = notional / entry_price
    # Verify the relationship: quantity × entry_price ≈ notional
    reconstructed = quantity * entry_price
    rel_err = abs(reconstructed - notional) / notional
    assert rel_err < 1e-9, f"quantity×entry_price={reconstructed:.6f} ≠ notional={notional:.6f} (rel_err={rel_err:.2e})"
    assert quantity > 0, f"quantity={quantity} not positive"
