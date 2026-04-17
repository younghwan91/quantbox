"""Tests for r116b sweep scripts.

US-R116B-6: verifies that r116b_symbol_blacklist_sweep, r116b_conviction_sweep,
and r116b_combined_sweep use canonical sim, have no hardcoded params, correctly
filter by blacklist/conviction, produce valid JSON schemas, and combined sweep
correctly selects top-2 × top-2 combinations.

These tests run fast (< 60s total) by using source-code inspection and tiny
synthetic fixtures — not the full 2.3-year baseline runs (those live in the
research scripts themselves, validated during r116b sweep execution).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
R116B_DIR = REPO_ROOT / "research" / "active" / "r116_willy_nfi"

BLACKLIST_SCRIPT = R116B_DIR / "r116b_symbol_blacklist_sweep.py"
CONVICTION_SCRIPT = R116B_DIR / "r116b_conviction_sweep.py"
COMBINED_SCRIPT = R116B_DIR / "r116b_combined_sweep.py"
SCRIPTS = [BLACKLIST_SCRIPT, CONVICTION_SCRIPT, COMBINED_SCRIPT]


# ═══════════════════════════════════════════════════════════════════════════
# Script existence + canonical sim import
# ═══════════════════════════════════════════════════════════════════════════


class TestCanonicalSim:
    """Every sweep script must import from trade_sim (canonical), not reimplement."""

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_script_exists(self, script: Path):
        assert script.exists(), f"Missing sweep script: {script}"

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_imports_canonical_sim(self, script: Path):
        """Must import collect_trades or equivalent from quantbox.backtest.trade_sim."""
        src = script.read_text()
        assert "from quantbox.backtest.trade_sim import" in src, (
            f"{script.name} does not import from quantbox.backtest.trade_sim"
        )
        # compute_metrics + portfolio_sim (or sim helpers) must be imported
        assert "compute_metrics" in src
        # blacklist script uses portfolio_sim_with_details
        # conviction + combined scripts use it indirectly through filter
        assert "portfolio_sim_with_details" in src or "portfolio_sim" in src

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_no_local_sim_redefinition(self, script: Path):
        """Must not define local 'collect_trades' or 'portfolio_sim' that would shadow canonical."""
        src = script.read_text()
        # Allow 'collect_tagged_trades' (new name) but forbid 'def collect_trades('
        assert not re.search(r"^def collect_trades\(", src, flags=re.MULTILINE), (
            f"{script.name} redefines collect_trades locally"
        )
        assert not re.search(r"^def portfolio_sim\(", src, flags=re.MULTILINE), (
            f"{script.name} redefines portfolio_sim locally"
        )


# ═══════════════════════════════════════════════════════════════════════════
# No hardcoded strategy params (config-first rule)
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigFirst:
    """Scripts must read params from VolStraddleConfig, never hardcode."""

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_uses_config(self, script: Path):
        src = script.read_text()
        assert "VolStraddleConfig" in src, f"{script.name} does not use VolStraddleConfig"

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_no_hardcoded_sl_tp_floats(self, script: Path):
        """Guard against `d_sl = 0.01` or `d_tp = 0.20` style hardcodes on RHS.

        Strategy-critical params must be sourced from cfg attrs (cfg.d_sl / CFG.d_sl).
        Numeric literals elsewhere (blacklist thresholds, statistical bounds) are allowed.
        """
        src = script.read_text()
        # If the script references d_sl/d_tp/ma_cluster_*/donchian_*, it must also
        # reference the config attribute form (cfg.X or CFG.X).
        forbidden_keys = ["d_sl", "d_tp", "ma_cluster_sl", "ma_cluster_tp", "donchian_sl", "donchian_tp"]
        for key in forbidden_keys:
            if key not in src:
                continue
            assert f"cfg.{key}" in src or f"CFG.{key}" in src, (
                f"{script.name} references {key} but no cfg.{key}/CFG.{key} — possible hardcode"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Blacklist logic correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestBlacklistLogic:
    def test_compute_blacklist_empty_at_zero_threshold(self):
        from research.active.r116_willy_nfi.r116b_symbol_blacklist_sweep import compute_blacklist

        per_sym = [{"symbol": "BTCUSDT", "PF": 0.5, "N": 100}]
        # threshold=0 → no blacklist regardless of PF
        assert compute_blacklist(per_sym, 0.0, 0) == set()

    def test_compute_blacklist_filters_by_pf_and_n(self):
        from research.active.r116_willy_nfi.r116b_symbol_blacklist_sweep import compute_blacklist

        per_sym = [
            {"symbol": "LOSER", "PF": 0.5, "N": 20},  # in (PF<0.9 AND N>=10)
            {"symbol": "SMALL_N", "PF": 0.3, "N": 5},  # out (N<10)
            {"symbol": "WINNER", "PF": 1.5, "N": 30},  # out (PF>=0.9)
            {"symbol": "BORDER", "PF": 0.9, "N": 15},  # out (PF not strictly <0.9)
        ]
        bl = compute_blacklist(per_sym, 0.9, 10)
        assert bl == {"LOSER"}

    def test_blacklist_variants_tuples(self):
        """VARIANTS must have 4 entries: empty, tight, medium, loose."""
        from research.active.r116_willy_nfi.r116b_symbol_blacklist_sweep import VARIANTS

        names = [v[0] for v in VARIANTS]
        assert names == ["empty", "tight", "medium", "loose"]
        # empty has threshold=0 (no filter)
        assert VARIANTS[0][1] == 0.0
        # thresholds monotonic
        thresholds = [v[1] for v in VARIANTS[1:]]
        assert thresholds == sorted(thresholds)


# ═══════════════════════════════════════════════════════════════════════════
# Conviction gate correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestConvictionGate:
    def test_filter_by_min_excludes_below_threshold(self):
        from research.active.r116_willy_nfi.r116b_conviction_sweep import filter_by_min

        tagged = [
            ("A", 0, 10, 0.01, 0),
            ("B", 5, 15, -0.02, 1),
            ("C", 10, 20, 0.03, 2),
            ("D", 15, 25, 0.01, 3),
        ]
        filt = filter_by_min(tagged, 2)
        # Expect scores 2 and 3 retained
        assert len(filt) == 2
        assert {t[0] for t in filt} == {"C", "D"}
        # Result strips conviction score (4-tuple)
        assert all(len(t) == 4 for t in filt)

    def test_filter_by_min_zero_keeps_all(self):
        from research.active.r116_willy_nfi.r116b_conviction_sweep import filter_by_min

        tagged = [("A", 0, 10, 0.01, 0), ("B", 5, 15, -0.02, 1)]
        assert len(filter_by_min(tagged, 0)) == 2

    def test_filter_by_min_high_excludes_all(self):
        from research.active.r116_willy_nfi.r116b_conviction_sweep import filter_by_min

        tagged = [("A", 0, 10, 0.01, 0), ("B", 5, 15, -0.02, 1)]
        assert filter_by_min(tagged, 5) == []


# ═══════════════════════════════════════════════════════════════════════════
# JSON output schema
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputSchema:
    """Runtime artifacts from sweep execution must contain expected keys."""

    STATE_DIR = REPO_ROOT / ".omc" / "state" / "r116b"
    REQUIRED_KEYS = {
        "N",
        "WR_pct",
        "PF",
        "MDD_pct",
        "sharpe",
        "calmar",
        "ann_ret_pct",
        "mfe_losing_avg_pct",
    }

    def _check_json(self, path: Path) -> dict:
        assert path.exists(), f"Missing JSON artifact: {path}"
        with open(path) as f:
            data = json.load(f)
        missing = self.REQUIRED_KEYS - set(data.keys())
        assert not missing, f"{path.name} missing keys: {missing}"
        return data

    def test_blacklist_json_schema(self):
        for variant in ["empty", "tight", "medium", "loose"]:
            self._check_json(self.STATE_DIR / f"blacklist_{variant}_metrics.json")

    def test_conviction_json_schema(self):
        for ms in [0, 1, 2, 3]:
            self._check_json(self.STATE_DIR / f"conviction_{ms}_metrics.json")

    def test_combined_json_schema(self):
        # Run must have produced at least 4 combined JSONs (top-2 × top-2)
        combined_files = list(self.STATE_DIR.glob("combined_*_min*_metrics.json"))
        assert len(combined_files) >= 4, f"Expected ≥4 combined JSONs, found {len(combined_files)}"
        for p in combined_files:
            self._check_json(p)


# ═══════════════════════════════════════════════════════════════════════════
# Combined sweep top-2 × top-2 selection
# ═══════════════════════════════════════════════════════════════════════════


class TestCombinedSelection:
    def test_select_top_n_by_pf(self):
        from research.active.r116_willy_nfi.r116b_combined_sweep import select_top_n

        variants = [
            {"variant": "A", "PF": 1.10},
            {"variant": "B", "PF": 1.40},
            {"variant": "C", "PF": 1.25},
            {"variant": "D", "PF": 1.30},
        ]
        top2 = select_top_n(variants, 2)
        assert [v["variant"] for v in top2] == ["B", "D"]

    def test_combined_summary_csv_has_four_rows(self):
        summary = REPO_ROOT / ".omc" / "state" / "r116b" / "combined_summary.csv"
        assert summary.exists(), "combined_summary.csv missing"
        lines = summary.read_text().strip().split("\n")
        # Header + 4 combos
        assert len(lines) == 5, f"Expected 5 lines (header + 4), got {len(lines)}: {lines}"
