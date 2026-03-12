"""T3-5: Tests for _verify_invariants() — position invariant checker."""

import pytest
from tests.conftest import _build_vs_scalper


def _valid_pos(**overrides) -> dict:
    """Return a valid position dict, optionally overriding fields."""
    base = {
        "symbol": "TESTUSDT",
        "signal": 1,
        "entry_price": 100.0,
        "notional": 3000.0,
        "position_side": "LONG",
    }
    base.update(overrides)
    return base


# ── Clean state ─────────────────────────────────────────────────────────────


def test_no_violations_empty_positions():
    scalper = _build_vs_scalper(balance=1000.0)
    violations = scalper._verify_invariants(1000.0)
    assert violations == []


def test_no_violations_normal_state():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos()
    violations = scalper._verify_invariants(1000.0)
    assert violations == []


def test_no_discord_when_clean():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper._verify_invariants(1000.0)
    scalper.discord.send.assert_not_called()


# ── Entry price ──────────────────────────────────────────────────────────────


def test_detects_zero_entry_price():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(entry_price=0.0)
    violations = scalper._verify_invariants(1000.0)
    assert any("entry_price" in v for v in violations)


def test_detects_negative_entry_price():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(entry_price=-5.0)
    violations = scalper._verify_invariants(1000.0)
    assert any("entry_price" in v for v in violations)


def test_detects_huge_entry_price():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(entry_price=2e7)
    violations = scalper._verify_invariants(1000.0)
    assert any("entry_price" in v for v in violations)


# ── Symbol format ────────────────────────────────────────────────────────────


def test_detects_colon_in_symbol():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(symbol="TESTUSDT:LONG")
    violations = scalper._verify_invariants(1000.0)
    assert any("symbol" in v or "malformed" in v for v in violations)


def test_detects_lowercase_symbol():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(symbol="testusdt")
    violations = scalper._verify_invariants(1000.0)
    assert any("symbol" in v or "malformed" in v for v in violations)


def test_valid_short_symbol():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["BTCUSDT:LONG"] = _valid_pos(symbol="BTCUSDT")
    violations = scalper._verify_invariants(1000.0)
    assert violations == []


# ── Position count ───────────────────────────────────────────────────────────


def test_detects_position_count_overflow():
    scalper = _build_vs_scalper(balance=1000.0)
    for i in range(11):
        scalper.strategy._positions[f"COIN{i:02d}USDT:LONG"] = _valid_pos(symbol=f"COIN{i:02d}USDT")
    violations = scalper._verify_invariants(1000.0)
    assert any("position_count" in v or "max_concurrent" in v for v in violations)


def test_exactly_at_max_concurrent_is_ok():
    scalper = _build_vs_scalper(balance=1000.0)
    max_c = scalper._cfg.max_concurrent  # 10
    for i in range(max_c):
        scalper.strategy._positions[f"COIN{i:02d}USDT:LONG"] = _valid_pos(symbol=f"COIN{i:02d}USDT")
    violations = scalper._verify_invariants(1000.0)
    # position_count violation should NOT appear (count == max, not >)
    assert not any("position_count" in v for v in violations)


# ── Signal validity ──────────────────────────────────────────────────────────


def test_detects_invalid_signal():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(signal=99)
    violations = scalper._verify_invariants(1000.0)
    assert any("signal" in v for v in violations)


def test_signal_none_is_invalid():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(signal=None)
    violations = scalper._verify_invariants(1000.0)
    assert any("signal" in v for v in violations)


@pytest.mark.parametrize("sig", [-1, 0, 1, 2])
def test_valid_signals(sig):
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(signal=sig)
    violations = scalper._verify_invariants(1000.0)
    assert not any("signal" in v for v in violations)


# ── Notional ─────────────────────────────────────────────────────────────────


def test_detects_zero_notional():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(notional=0.0)
    violations = scalper._verify_invariants(1000.0)
    assert any("notional" in v for v in violations)


def test_detects_excessive_notional():
    scalper = _build_vs_scalper(balance=1000.0)
    # balance=1000 * max_concurrent=10 * max_leverage=125 = 1_250_000 → use 2x that
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(notional=3_000_000.0)
    violations = scalper._verify_invariants(1000.0)
    assert any("notional" in v for v in violations)


def test_zero_balance_guard():
    """balance=0 must not cause ZeroDivisionError (T1-4 API failure case)."""
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos()
    violations = scalper._verify_invariants(0.0)  # must not raise
    assert isinstance(violations, list)


# ── Discord alert ─────────────────────────────────────────────────────────────


def test_discord_sent_on_violation():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(signal=99)
    scalper._verify_invariants(1000.0)
    assert scalper.discord.send.called


def test_discord_single_call_for_multiple_violations():
    scalper = _build_vs_scalper(balance=1000.0)
    scalper.strategy._positions["TESTUSDT:LONG"] = _valid_pos(signal=99, entry_price=0.0)
    scalper._verify_invariants(1000.0)
    assert scalper.discord.send.call_count == 1  # one call, not one per violation
