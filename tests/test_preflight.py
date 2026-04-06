"""Tests for OrderPreflight — order pre-flight validation gate.

Verifies that each known failure mode is caught before any API call is made,
and that valid orders pass through without modification.
"""

from quantbox.execution.preflight import OrderPreflight, PreflightResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SYM_INFO = {"step_size": 0.01, "tick_size": 0.1, "price_precision": 2}


def _validate(**kwargs) -> PreflightResult:
    """Helper with sensible defaults for a valid BUY entry order."""
    defaults = dict(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.10,
        leverage=30,
        reduce_only=False,
        sym_info=VALID_SYM_INFO,
    )
    defaults.update(kwargs)
    return OrderPreflight.validate(**defaults)


# ── Happy path ────────────────────────────────────────────────────────────────


class TestPreflightPass:
    def test_valid_entry_order(self):
        result = _validate()
        assert result.ok is True
        assert result.reason is None

    def test_valid_sell_entry(self):
        result = _validate(side="SELL", quantity=1.0, leverage=21)
        assert result.ok is True

    def test_reduce_only_close(self):
        # reduce_only close with step_size-misaligned qty should still pass
        result = _validate(
            side="SELL",
            quantity=0.12345,  # not aligned to step_size=0.01
            leverage=1,
            reduce_only=True,
        )
        assert result.ok is True

    def test_empty_sym_info_no_step_check(self):
        # If sym_info has no step_size, step-size check is skipped gracefully
        result = _validate(sym_info={})
        assert result.ok is True


# ── Symbol format failures ─────────────────────────────────────────────────────


class TestSymbolValidation:
    """Incident #4 (2026-03-08): compound key 'SUIUSDT:LONG' sent as API
    symbol → -1022 signature error → SL/TP placement failed → UNPROTECTED."""

    def test_compound_key_rejected(self):
        result = _validate(symbol="SUIUSDT:LONG")
        assert result.ok is False
        assert "compound key" in result.reason
        assert "SUIUSDT:LONG" in result.reason

    def test_compound_key_short(self):
        result = _validate(symbol="WLDUSDT:SHORT")
        assert result.ok is False
        assert ":" in result.reason or "compound" in result.reason

    def test_empty_symbol_rejected(self):
        result = _validate(symbol="")
        assert result.ok is False
        assert "empty" in result.reason


# ── Side validation ───────────────────────────────────────────────────────────


class TestSideValidation:
    def test_invalid_side_long(self):
        # Common mistake: passing positionSide 'LONG' as order side
        result = _validate(side="LONG")
        assert result.ok is False
        assert "side" in result.reason.lower()

    def test_invalid_side_short(self):
        result = _validate(side="SHORT")
        assert result.ok is False

    def test_invalid_side_empty(self):
        result = _validate(side="")
        assert result.ok is False


# ── Quantity validation ───────────────────────────────────────────────────────


class TestQuantityValidation:
    def test_zero_quantity_rejected(self):
        result = _validate(quantity=0.0)
        assert result.ok is False
        assert "quantity" in result.reason.lower() or "non-positive" in result.reason

    def test_negative_quantity_rejected(self):
        result = _validate(quantity=-0.5)
        assert result.ok is False


# ── Leverage validation ───────────────────────────────────────────────────────


class TestLeverageValidation:
    def test_zero_leverage_rejected(self):
        result = _validate(leverage=0)
        assert result.ok is False
        assert "leverage" in result.reason.lower()

    def test_negative_leverage_rejected(self):
        result = _validate(leverage=-1)
        assert result.ok is False

    def test_leverage_one_entry_ok(self):
        # leverage=1 is valid (though unusual for entry)
        result = _validate(leverage=1)
        assert result.ok is True


# ── Step-size alignment ───────────────────────────────────────────────────────


class TestStepSizeValidation:
    def test_aligned_quantity_ok(self):
        result = _validate(quantity=0.10, sym_info={"step_size": 0.01})
        assert result.ok is True

    def test_misaligned_entry_rejected(self):
        # 0.1234 % 0.01 ≈ 0.0034 > 0.01 * 0.01 tolerance
        result = _validate(quantity=0.1234, sym_info={"step_size": 0.01})
        assert result.ok is False
        assert "step" in result.reason.lower() or "aligned" in result.reason.lower()

    def test_misaligned_reduce_only_allowed(self):
        # Reduce-only closes skip step-size check (partial close rounding)
        result = _validate(
            quantity=0.1234,
            sym_info={"step_size": 0.01},
            reduce_only=True,
        )
        assert result.ok is True

    def test_floating_point_tolerance(self):
        # 0.3 - 0.1 - 0.1 - 0.1 = ~1e-17 remainder due to float; should pass
        qty = 0.1 + 0.1 + 0.1  # 0.30000000000000004
        result = _validate(quantity=qty, sym_info={"step_size": 0.1})
        assert result.ok is True


# ── Paper mode integration ────────────────────────────────────────────────────


class TestPaperModeCoverage:
    """Verify preflight runs before paper_mode early return.

    Previously preflight was live-only.  Bugs like compound keys were only
    caught when the same path hit the live exchange for the first time.
    """

    def test_compound_key_caught_in_paper_mode(self):
        from tests.conftest import _build_vs_scalper

        scalper = _build_vs_scalper(balance=1000.0)
        scalper.paper_mode = True

        # _place_order with compound key must return None even in paper mode
        result = scalper._place_order(
            symbol="TESTUSDT:LONG",  # compound key — should be caught by preflight
            side="BUY",
            quantity=1.0,
            leverage=30,
        )
        assert result is None, "Preflight should reject compound key before paper fill"
