"""Research integrity guard tests.

These tests prevent two categories of silent failures:

1. **Stale import guard**: New research files importing from backtest_v14_3_0.py
   get wrong parameters (D_TRAIL_ACTIVATE=0.02 vs production 0.010).
   r53 incident: results invalidated after stale import discovered.

2. **Universe guard**: Research files using cfg.exclude_symbols instead of
   cfg.symbol_whitelist include delisted symbols (MKRUSDT/FTMUSDT/EOSUSDT).
   r55-r57 incident: 63 symbols vs live 60 symbols, backtest-live parity broken.

Whitelist of legacy files that are allowed to import backtest_v14_3_0:
  These are archived research scripts whose results are already documented.
  New scripts must use quantbox.backtest.trade_sim instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────────────
#  Legacy files ALLOWED to import backtest_v14_3_0 (already documented, archived)
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent

_LEGACY_STALE_IMPORT_ALLOWED: set[str] = {
    # The file itself
    "research/active/r41_r45/backtest_v14_3_0.py",
    # r46-r50: written before trade_sim.py existed
    "research/active/r46_r50/r46_monte_carlo.py",
    "research/active/r46_r50/r46b_plot.py",
    "research/active/r47_r50/r47_exit_ab_test.py",
    "research/active/r47_r50/r47b_comparison.py",
    "research/active/r47_r50/r48_dd_halt_analysis.py",
    "research/active/r47_r50/r49_signal_timeline.py",
    "research/active/r47_r50/r50_cross_sectional_momentum.py",
    "research/active/r47_r50/r51_post_squeeze_trend.py",
    # r53: documented incident, results superseded
    "research/active/r51_r60/r53_combined.py",
    "research/active/r51_r60/r53_extra_combos.py",
    "research/active/r51_r60/r53_opportunity_augmentation.py",
    "research/active/r51_r60/r53_rerun_correct.py",
}

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: find all research .py files with a given string
# ──────────────────────────────────────────────────────────────────────────────


def _find_research_py_files() -> list[Path]:
    """Return all .py files under research/ (excluding __pycache__)."""
    return [p for p in (REPO_ROOT / "research").rglob("*.py") if "__pycache__" not in p.parts]


def _file_imports_module(path: Path, module_name: str) -> bool:
    """Return True if the file has an import statement referencing module_name."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return module_name in source


def _file_has_pattern_outside_comments(path: Path, pattern: str) -> bool:
    """Return True if pattern appears in non-comment, non-docstring code."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    # Quick check: if pattern not in file at all, skip expensive AST walk
    if pattern not in source:
        return False
    # Line-by-line: only count lines that are not pure comments
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if pattern in line:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Test 1: No new stale backtest_v14_3_0 imports
# ──────────────────────────────────────────────────────────────────────────────


class TestNoStaleBacktestImports:
    def test_no_new_files_import_backtest_v14_3_0(self) -> None:
        """New research files must not import from backtest_v14_3_0.

        All results from scripts using that file are potentially invalidated:
        - D_TRAIL_ACTIVATE was 0.02 in that file vs 0.010 in production
        - d_tp was 0.05 vs 0.20 in production
        Use quantbox.backtest.trade_sim instead.
        """
        violators: list[str] = []
        for py_file in _find_research_py_files():
            rel = str(py_file.relative_to(REPO_ROOT))
            if rel in _LEGACY_STALE_IMPORT_ALLOWED:
                continue
            if _file_imports_module(py_file, "backtest_v14_3_0"):
                violators.append(rel)

        assert not violators, (
            f"New research files importing stale backtest_v14_3_0:\n"
            + "\n".join(f"  {v}" for v in sorted(violators))
            + "\n\nUse: from quantbox.backtest.trade_sim import collect_trades, portfolio_sim"
            + "\nSee backtesting.instructions.md Section 1."
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Test 2: No exclude_symbols in research scripts newer than r60
# ──────────────────────────────────────────────────────────────────────────────

# Research files from r61 onward must use symbol_whitelist, not exclude_symbols
_EXCLUDE_SYMBOLS_LEGACY_ALLOWED: set[str] = {
    # These predate the whitelist rule and are archived
    "research/active/r51_r60/r53_combined.py",
    "research/active/r51_r60/r53_extra_combos.py",
    "research/active/r51_r60/r53_opportunity_augmentation.py",
    "research/active/r51_r60/r53_rerun_correct.py",
    "research/active/r51_r60/r55_subtf_entry.py",
    "research/active/r51_r60/r56_universe_expansion.py",
    "research/active/r51_r60/r57_30m_tf.py",
    # r66/r67 use exclude_symbols in comments/log messages only
    "research/active/r66_r70/r66_prod_verify.py",
    "research/active/r66_r70/r67_sensitivity.py",
}

_POST_RULE_DIRS = {
    "r61_r65",
    "r66_r70",
    "r71_r75",
    "r76_r80",
}


class TestNoExcludeSymbolsInNewResearch:
    def test_no_exclude_symbols_in_r61_plus(self) -> None:
        """Research scripts from r61+ must use cfg.symbol_whitelist, not cfg.exclude_symbols.

        r55-r57 incident (2026-03-11): exclude_symbols only removes stablecoins,
        leaving delisted symbols (MKRUSDT/FTMUSDT/EOSUSDT) in the universe.
        Result: 63-symbol backtest vs 60-symbol live trading → parity broken.
        """
        violators: list[str] = []
        for py_file in _find_research_py_files():
            rel = str(py_file.relative_to(REPO_ROOT))
            if rel in _EXCLUDE_SYMBOLS_LEGACY_ALLOWED:
                continue
            # Only check r61+ directories
            if not any(d in rel for d in _POST_RULE_DIRS):
                continue
            if _file_has_pattern_outside_comments(py_file, "exclude_symbols"):
                violators.append(rel)

        assert not violators, (
            f"Research files using exclude_symbols (must use symbol_whitelist):\n"
            + "\n".join(f"  {v}" for v in sorted(violators))
            + "\n\nCorrect pattern:"
            + "\n  whitelist = set(cfg.symbol_whitelist)"
            + "\n  if sym not in whitelist: continue"
            + "\nSee backtesting.instructions.md Phase 0 — Universe Loading."
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Test 3: Whitelist consistency (no delisted symbols)
# ──────────────────────────────────────────────────────────────────────────────


class TestWhitelistConsistency:
    def test_delisted_not_in_whitelist(self) -> None:
        """MKRUSDT, FTMUSDT, EOSUSDT must never appear in symbol_whitelist."""
        from quantbox.strategy.vol_straddle.config import VolStraddleConfig

        cfg = VolStraddleConfig()
        delisted = {"MKRUSDT", "FTMUSDT", "EOSUSDT"}
        found = delisted & set(cfg.symbol_whitelist)
        assert not found, (
            f"Delisted symbols in whitelist: {found}. "
            f"These have been removed from Binance USDT-M perpetuals. "
            f"Remove from symbol_whitelist in config.py."
        )
