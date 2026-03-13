"""Order pre-flight validation.

Provides a centralised gate that runs before every live order placement.
All checks are performed without additional API calls — runtime constraints
(max_notional, available margin) are enforced earlier in the call chain
(positions.py) or reactively in limit_chaser.py.

Motivation: across v11–v14 several bugs were caused by new code paths
(CW-Tier, hedge-mode close, MA-cluster) bypassing guards that already
existed on other paths.  A single, always-called validate() ensures every
order passes the same baseline sanity checks regardless of which strategy
or execution branch generated it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantbox.core.types import ApiSymbol, Side


@dataclass
class PreflightResult:
    """Result of an order pre-flight check."""

    ok: bool
    reason: str | None = None


class OrderPreflight:
    """Stateless order validator called before every live order."""

    @staticmethod
    def validate(
        symbol: "ApiSymbol",
        side: "Side",
        quantity: float,
        leverage: int,
        reduce_only: bool,
        sym_info: dict,
    ) -> PreflightResult:
        """Validate order parameters before submission.

        Args:
            symbol: Trading symbol sent to Binance API (must not contain ':').
            side: Order side — 'BUY' or 'SELL'.
            quantity: Order quantity in base asset units.
            leverage: Leverage multiplier (≥ 1).
            reduce_only: True for close/reduce orders; step_size check is
                skipped for reduce-only orders to avoid rounding conflicts on
                partial-close quantities.
            sym_info: Symbol metadata dict from DataMixin.get_symbol_info().
                Expected keys: 'step_size' (float).

        Returns:
            PreflightResult with ok=True on success, or ok=False and a
            human-readable reason string on failure.
        """
        # ── Symbol format ──────────────────────────────────────────────────
        # Incident #4 (2026-03-08): compound key 'SUIUSDT:LONG' was passed
        # as the API symbol, causing -1022 signature errors and bracket
        # repair failures that left positions without SL/TP protection.
        if not symbol:
            return PreflightResult(False, "symbol is empty")
        if ":" in symbol:
            return PreflightResult(
                False,
                f"compound key passed as API symbol: {symbol!r} — use pos['symbol'] not pos_key",
            )

        # ── Side ───────────────────────────────────────────────────────────
        if side not in ("BUY", "SELL"):
            return PreflightResult(False, f"invalid side: {side!r} (expected 'BUY' or 'SELL')")

        # ── Quantity ───────────────────────────────────────────────────────
        if quantity <= 0:
            return PreflightResult(False, f"non-positive quantity: {quantity}")

        # ── Leverage ───────────────────────────────────────────────────────
        if leverage < 1:
            return PreflightResult(False, f"invalid leverage: {leverage} (must be ≥ 1)")

        # ── Step-size alignment (entry orders only) ────────────────────────
        # Skip for reduce-only: partial-close quantities may not align to
        # step_size if the original position was opened with a different
        # precision.  Binance accepts reduce-only quantities that are
        # slightly misaligned (it rounds internally).
        if not reduce_only:
            step_size = sym_info.get("step_size", 0.0)
            if step_size > 0:
                # Robust step-size check: compute nearest aligned value and
                # measure distance.  Using modulo directly is unreliable for
                # quantities that are exact multiples (e.g. 1.0 % 0.01 → 0.01
                # in CPython due to float representation of 0.01).
                steps = quantity / step_size
                aligned = round(steps) * step_size
                remainder = abs(quantity - aligned)
                tolerance = step_size * 0.01  # 1% of step — floating-point slack
                if remainder > tolerance:
                    return PreflightResult(
                        False,
                        f"quantity {quantity} not aligned to step_size {step_size} (remainder={remainder:.10g})",
                    )

        return PreflightResult(True)
