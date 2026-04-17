"""Tests for r117 Donchian parity audit.

US-URGENT-1: verifies that r117_donchian_parity_audit.py correctly
extracts live Donchian trades from journal.sqlite, matches them to sim
results via bar-index tolerance, computes deltas, and emits a binary
PAUSE/KEEP recommendation against fixed thresholds.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "research" / "active" / "r116_willy_nfi" / "r117_donchian_parity_audit.py"

sys.path.insert(0, str(REPO_ROOT))


def test_script_exists_and_imports_canonical_sim():
    assert SCRIPT.exists()
    src = SCRIPT.read_text()
    assert "from quantbox.backtest.trade_sim import" in src
    assert "sim_donchian_trailing" in src
    assert "VolStraddleConfig" in src


def test_sql_query_correctness():
    """Loader must filter strategy_type='donchian' and pull required cols."""
    src = SCRIPT.read_text()
    assert "strategy_type='donchian'" in src
    # Required columns for parity comparison
    for col in [
        "symbol",
        "direction",
        "entry_price",
        "exit_price",
        "pnl_pct",
        "pnl_usd",
        "bars_held",
        "exit_reason",
        "entry_time",
        "exit_time",
        "notional",
    ]:
        assert col in src, f"SQL missing column: {col}"


def test_bar_match_tolerance_is_defined():
    """±1 bar tolerance must be explicitly named so future tuning is easy."""
    src = SCRIPT.read_text()
    m = re.search(r"BAR_MATCH_TOLERANCE\s*=\s*(\d+)", src)
    assert m, "BAR_MATCH_TOLERANCE not defined"
    tol = int(m.group(1))
    assert 0 <= tol <= 3, f"tolerance {tol} outside sane range 0-3"


def test_pause_thresholds_defined():
    """Decision thresholds must be named constants, not magic numbers buried in code."""
    src = SCRIPT.read_text()
    assert "PAUSE_AVG_DELTA_THRESHOLD" in src
    assert "PAUSE_WORST3_DELTA_THRESHOLD" in src


def test_find_matching_sim_trade_tolerance():
    from research.active.r116_willy_nfi.r117_donchian_parity_audit import (
        BAR_MATCH_TOLERANCE,
        find_matching_sim_trade,
    )

    sim = [(100, 110, 0.02), (200, 210, -0.01), (300, 310, 0.015)]
    # Exact match
    assert find_matching_sim_trade(sim, 200) == (200, 210, -0.01)
    # Within tolerance
    assert find_matching_sim_trade(sim, 200 + BAR_MATCH_TOLERANCE) is not None
    assert find_matching_sim_trade(sim, 200 - BAR_MATCH_TOLERANCE) is not None
    # Outside tolerance
    assert find_matching_sim_trade(sim, 200 + BAR_MATCH_TOLERANCE + 10) is None
    # Negative bar_idx (OUT_OF_CACHE sentinel)
    assert find_matching_sim_trade(sim, -1) is None


def test_classify_logic():
    from research.active.r116_willy_nfi.r117_donchian_parity_audit import classify

    # Identical (delta < 0.3%)
    assert classify({"pnl_pct": 1.5}, (10, 20, 0.015)) == "IDENTICAL"
    # Same sign, different mag
    assert classify({"pnl_pct": 1.5}, (10, 20, 0.050)) == "SAME_SIGN_DIFFERENT_MAG"
    # Opposite sign
    assert classify({"pnl_pct": -1.5}, (10, 20, 0.050)) == "OPPOSITE_SIGN"
    # No sim match
    assert classify({"pnl_pct": 1.5}, None) == "NO_SIM_SIGNAL"
    # Out of cache
    assert classify({"pnl_pct": 1.5, "_out_of_cache": True}, None) == "OUT_OF_CACHE"


def test_recommendation_thresholds():
    from research.active.r116_willy_nfi.r117_donchian_parity_audit import (
        make_recommendation,
    )

    # Case 1: no data
    r, _ = make_recommendation(
        {
            "delta_avg_pct": None,
            "delta_worst3_avg_pct": None,
            "n_live": 0,
            "n_matched_sim": 0,
        }
    )
    assert r == "INSUFFICIENT_DATA"

    # Case 2: both pause conditions met
    r, reasons = make_recommendation(
        {
            "delta_avg_pct": -15.0,
            "delta_worst3_avg_pct": -20.0,
            "n_live": 9,
            "n_matched_sim": 4,
        }
    )
    assert r == "PAUSE_DONCHIAN"
    assert any("2.0" in rs or "threshold" in rs.lower() for rs in reasons)

    # Case 3: only avg triggers — not enough
    r, _ = make_recommendation(
        {
            "delta_avg_pct": -5.0,
            "delta_worst3_avg_pct": -1.0,
            "n_live": 9,
            "n_matched_sim": 4,
        }
    )
    assert r == "KEEP_RUNNING"

    # Case 4: only worst-3 triggers — not enough
    r, _ = make_recommendation(
        {
            "delta_avg_pct": -1.0,
            "delta_worst3_avg_pct": -5.0,
            "n_live": 9,
            "n_matched_sim": 4,
        }
    )
    assert r == "KEEP_RUNNING"


def test_output_artifacts_exist():
    """After running the audit, JSON + report files should exist on disk."""
    # Artifact existence is confirmed by the script's last run
    json_path = REPO_ROOT / ".omc" / "state" / "r117" / "donchian_parity.json"
    assert json_path.exists(), f"Missing {json_path} — run the audit script first"

    data = json.loads(json_path.read_text())
    assert "recommendation" in data
    assert "aggregate" in data
    assert "per_trade" in data
    assert data["aggregate"]["n_live"] >= 0
