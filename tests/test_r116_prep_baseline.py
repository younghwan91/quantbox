"""Tests for r116_prep_baseline.py static properties.

These tests verify script correctness by inspecting source text and importing
module-level constants — no subprocess, no data files required, completes <5s.
"""

from __future__ import annotations

import ast
import importlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "research" / "active" / "r116_willy_nfi" / "r116_prep_baseline.py"
SCRIPT_SRC = SCRIPT_PATH.read_text()


# ── test_baseline_imports_canonical_sim ──────────────────────────────────────


def test_baseline_imports_canonical_sim():
    """Script must import collect_trades and portfolio_sim from canonical sim."""
    assert "from quantbox.backtest.trade_sim import" in SCRIPT_SRC, (
        "Script does not import from quantbox.backtest.trade_sim"
    )
    # Both names must appear in import context (not just anywhere)
    import_block_match = re.search(r"from quantbox\.backtest\.trade_sim import\s*\(([^)]+)\)", SCRIPT_SRC)
    assert import_block_match, "Could not find multi-line trade_sim import block"
    import_names = import_block_match.group(1)
    assert "collect_trades" in import_names, "collect_trades not in trade_sim import"
    assert "portfolio_sim" in import_names, "portfolio_sim not in trade_sim import"

    # No local redefinition of collect_trades or portfolio_sim
    tree = ast.parse(SCRIPT_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.name not in ("collect_trades", "portfolio_sim"), (
                f"Local redefinition of canonical function: {node.name}"
            )


# ── test_baseline_uses_config ────────────────────────────────────────────────


def test_baseline_uses_config():
    """Script must use VolStraddleConfig and not hardcode SL/TP floats by name."""
    assert "VolStraddleConfig()" in SCRIPT_SRC, "Script does not instantiate VolStraddleConfig()"
    # Ensure no bare hardcoded assignments for the key SL/TP param names
    forbidden_patterns = [
        r"\bd_sl\s*=\s*[\d.]+",
        r"\bd_tp\s*=\s*[\d.]+",
        r"\bma_cluster_sl\s*=\s*[\d.]+",
        r"\bma_cluster_tp\s*=\s*[\d.]+",
        r"\bdonchian_sl\s*=\s*[\d.]+",
        r"\bdonchian_tp\s*=\s*[\d.]+",
    ]
    for pat in forbidden_patterns:
        matches = re.findall(pat, SCRIPT_SRC)
        assert not matches, f"Hardcoded param found matching '{pat}': {matches}"


# ── test_baseline_window_no_leak ─────────────────────────────────────────────


def test_baseline_window_no_leak():
    """WINDOW_END must be at or before the cutoff 2026-04-12T16:18:00+00:00."""
    # Import the module constants directly
    import sys

    root = str(Path(__file__).parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

    spec = importlib.util.spec_from_file_location("r116_prep_baseline", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    cutoff = datetime(2026, 4, 12, 16, 18, 0, tzinfo=timezone.utc)
    assert hasattr(mod, "WINDOW_END"), "WINDOW_END not defined in module"
    assert mod.WINDOW_END <= cutoff, f"WINDOW_END {mod.WINDOW_END} exceeds cutoff {cutoff} — potential data leak"


# ── test_baseline_metrics_schema ─────────────────────────────────────────────


def test_baseline_metrics_schema():
    """Result dict from run_* functions must contain all required metric keys."""
    # Verify by inspecting the source AST for the required keys in result dicts
    required_keys = {
        "N",
        "WR_pct",
        "PF",
        "MDD_pct",
        "avg_win_pct",
        "avg_loss_pct",
        "sharpe",
        "calmar",
        "mfe_losing_avg_pct",
        "low_N",
    }

    # Collect all string constants in the source that appear as dict keys
    tree = ast.parse(SCRIPT_SRC)
    found_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    found_keys.add(key.value)

    missing = required_keys - found_keys
    assert not missing, f"Result dict missing required metric keys: {missing}"


# ── test_baseline_no_mock_data ────────────────────────────────────────────────


def test_baseline_no_mock_data():
    """Script must not use synthetic/mock data generation."""
    forbidden = [
        r"np\.random",
        r"\bsynthetic\b",
        r"\bfake_ohlcv\b",
    ]
    for pat in forbidden:
        matches = re.findall(pat, SCRIPT_SRC)
        assert not matches, f"Mock data pattern '{pat}' found in script: {matches}"


# ── test_ma_cluster_per_symbol_schema ─────────────────────────────────────────


def test_ma_cluster_per_symbol_schema():
    """run_ma_cluster must produce per_symbol breakdown (US-URGENT-5)."""
    import json as _json

    # Source-text check: run_ma_cluster should call compute_per_symbol_metrics.
    ma_func_src = re.search(
        r"def run_ma_cluster\(.*?(?=\ndef |\Z)",
        SCRIPT_SRC,
        flags=re.DOTALL,
    )
    assert ma_func_src is not None, "run_ma_cluster function not found in script"
    assert "compute_per_symbol_metrics" in ma_func_src.group(0), "run_ma_cluster does not compute per_symbol breakdown"
    assert '"per_symbol"' in ma_func_src.group(0), "run_ma_cluster result dict missing per_symbol key"

    # Artifact check: if baseline has been run, JSON file should include per_symbol
    artifact = Path(__file__).parents[1] / ".omc" / "state" / "r116_prep" / "ma_cluster_metrics.json"
    if artifact.exists():
        data = _json.loads(artifact.read_text())
        assert "per_symbol" in data, "ma_cluster_metrics.json missing per_symbol array"
        assert isinstance(data["per_symbol"], list)
        if data["per_symbol"]:
            row = data["per_symbol"][0]
            for key in ("symbol", "N", "WR_pct", "PF", "mfe_losing_avg_pct"):
                assert key in row, f"per_symbol row missing {key}"


def test_ma_cluster_blacklist_candidates_artifact_valid():
    """If candidates file exists, it must have the documented schema."""
    import json as _json

    path = Path(__file__).parents[1] / ".omc" / "state" / "r116_prep" / "ma_cluster_blacklist_candidates.json"
    if not path.exists():
        pytest.skip("ma_cluster_blacklist_candidates.json not generated yet")
    data = _json.loads(path.read_text())
    assert "criterion" in data
    assert "count" in data
    assert "candidates" in data
    for c in data["candidates"]:
        assert c["PF"] < 1.0
        assert c["N"] >= 10
