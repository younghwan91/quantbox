"""Tests for STARTUP CROSSCHECK message / alert-only parity fix.

US-URGENT-3: previously, the startup crosscheck log claimed
"Will be auto-closed in first reconciliation cycle" but the actual
reconciliation path routes to ORPHAN DETECTED → REVERSE GHOST (alert-only).
This mismatch caused operators to trust the misleading auto-close promise
and skip manual !close-orphan action.

This fix aligns the startup message with alert-only semantics, matching
user preference (feedback_no_shutdown.md: never auto-close without approval).
"""

from __future__ import annotations

from pathlib import Path

SCALPER_PY = Path(__file__).resolve().parents[1] / "quantbox" / "scalper.py"


def test_startup_crosscheck_message_is_alert_only():
    """The startup crosscheck log MUST NOT claim auto-close.

    It should mention Alert-only, !adopt, and !close-orphan as the user actions.
    """
    src = SCALPER_PY.read_text()
    # Find the STARTUP CROSSCHECK block (single occurrence expected)
    assert src.count("STARTUP CROSSCHECK") >= 1, "STARTUP CROSSCHECK message missing"
    # Must not promise auto-close in the log
    assert "Will be auto-closed" not in src, (
        "STARTUP CROSSCHECK log still claims auto-close; this contradicts actual reconciliation behavior (alert-only)."
    )
    # Must reference the user action commands
    assert "!adopt" in src
    assert "!close-orphan" in src


def test_discord_startup_message_is_alert_only():
    """The Discord startup notification must also be alert-only."""
    src = SCALPER_PY.read_text()
    # Find the startup Discord send block
    assert "Startup: Untracked position" in src
    # Must not claim "Auto-close pending" (pre-fix wording)
    assert "Auto-close pending" not in src, (
        "Discord startup message still claims auto-close pending; must be alert-only."
    )
    # Alert-only phrasing (case-insensitive check for safety)
    src_lower = src.lower()
    assert "alert-only" in src_lower


def test_orphan_detected_path_still_alert_only():
    """The ongoing-cycle ORPHAN DETECTED path remains alert-only (no change).

    This test pins the existing behavior: we did NOT change the reconciliation
    logic, only the startup crosscheck message.
    """
    # Check reconciliation module
    recon_py = Path(__file__).resolve().parents[1] / "quantbox" / "execution" / "reconciliation.py"
    rsrc = recon_py.read_text().lower()
    # Both markers should be present: "alert-only" descriptor and "no auto-close" clarification
    assert "alert-only" in rsrc, (
        "ORPHAN DETECTED path should still be alert-only; if this assertion "
        "fails, the reconciliation logic was changed and the startup message "
        "policy may need re-alignment."
    )
    assert "no auto-close" in rsrc, "Reconciliation should still document 'no auto-close' semantics"


def test_startup_crosscheck_still_warns_on_missing_sl():
    """Still log NO SL/TP concern (the safety warning is preserved)."""
    src = SCALPER_PY.read_text()
    # Original safety warning must remain
    assert "No SL/TP" in src
