#!/usr/bin/env python3
"""Research parameter hardcoding guard.

Pre-commit hook that detects hardcoded parameter values in research scripts.
Run via .pre-commit-config.yaml on files matching research/**/*.py.

Catches:
  - VR_LN = <number>  (should come from cfg.base_leverage / cfg.max_concurrent)
  - MA_LN = <number>  (should come from cfg.ma_cluster_leverage / cfg.max_concurrent)
  - d_tp = <float>    (should come from cfg.d_tp)
  - New imports of backtest_v14_3_0 (stale parameters — r53 incident)

Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
#  Files allowed to have stale backtest_v14_3_0 imports (legacy, archived)
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent

_LEGACY_STALE_ALLOWED: set[str] = {
    "research/active/r41_r45/backtest_v14_3_0.py",
    "research/active/r46_r50/r46_monte_carlo.py",
    "research/active/r46_r50/r46b_plot.py",
    "research/active/r47_r50/r47_exit_ab_test.py",
    "research/active/r47_r50/r47b_comparison.py",
    "research/active/r47_r50/r48_dd_halt_analysis.py",
    "research/active/r47_r50/r49_signal_timeline.py",
    "research/active/r47_r50/r50_cross_sectional_momentum.py",
    "research/active/r47_r50/r51_post_squeeze_trend.py",
    "research/active/r51_r60/r53_combined.py",
    "research/active/r51_r60/r53_extra_combos.py",
    "research/active/r51_r60/r53_opportunity_augmentation.py",
    "research/active/r51_r60/r53_rerun_correct.py",
}

# ──────────────────────────────────────────────────────────────────────────────
#  Patterns: (regex, message)
# ──────────────────────────────────────────────────────────────────────────────
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^\s*VR_LN\s*=\s*[\d.]+\s*$"),
        "Hardcoded VR_LN. Use: VR_LN = cfg.base_leverage / cfg.max_concurrent",
    ),
    (
        re.compile(r"^\s*MA_LN\s*=\s*[\d.]+\s*$"),
        "Hardcoded MA_LN. Use: MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent",
    ),
    (
        re.compile(r"^\s*LN\s*=\s*[\d.]+\s*$"),
        "Hardcoded LN. Use: LN = cfg.base_leverage / cfg.max_concurrent",
    ),
    (
        re.compile(r"['\"]d_tp['\"]:\s*0\.\d+"),
        "Hardcoded d_tp value. Use: 'tp': cfg.d_tp (or cfg.s_tp for straddle)",
    ),
    (
        re.compile(r"['\"]d_sl['\"]:\s*0\.\d+(?!\s*#\s*from\s*cfg)"),
        "Hardcoded d_sl value. Use: 'd_sl': cfg.d_sl",
    ),
]

_STALE_IMPORT_PATTERN = re.compile(r"from\s+\S*backtest_v14_3_0\s+import|import\s+\S*backtest_v14_3_0")


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, line_text, message) for violations."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    rel = str(path.relative_to(REPO_ROOT))
    violations: list[tuple[int, str, str]] = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip comment-only lines
        if stripped.startswith("#"):
            continue

        # Stale import check (only for new files)
        if _STALE_IMPORT_PATTERN.search(line) and rel not in _LEGACY_STALE_ALLOWED:
            violations.append((
                lineno, line.rstrip(),
                "Stale import: backtest_v14_3_0 has wrong parameters "
                "(D_TRAIL_ACTIVATE=0.02 vs production 0.010). "
                "Use: from quantbox.backtest.trade_sim import collect_trades",
            ))

        # Hardcoded parameter patterns
        for pattern, msg in _PATTERNS:
            if pattern.search(line):
                violations.append((lineno, line.rstrip(), msg))

    return violations


def main(files: list[str]) -> int:
    """Check files for violations. Return exit code."""
    all_violations: dict[str, list[tuple[int, str, str]]] = {}

    for file_str in files:
        path = Path(file_str)
        if not path.exists() or path.suffix != ".py":
            continue
        v = check_file(path)
        if v:
            all_violations[file_str] = v

    if not all_violations:
        return 0

    print("\n[research-param-guard] Hardcoded parameters detected:\n", file=sys.stderr)
    for file_str, violations in sorted(all_violations.items()):
        rel = str(Path(file_str).relative_to(REPO_ROOT)) if Path(file_str).is_absolute() else file_str
        for lineno, line, msg in violations:
            print(f"  {rel}:{lineno}: {msg}", file=sys.stderr)
            print(f"    {line}", file=sys.stderr)
        print(file=sys.stderr)

    print(
        "Fix: Read ALL parameters from VolStraddleConfig().\n"
        "  from quantbox.strategy.vol_straddle.config import VolStraddleConfig\n"
        "  cfg = VolStraddleConfig()\n"
        "  VR_LN = cfg.base_leverage / cfg.max_concurrent  # 3.0\n"
        "See backtesting.instructions.md Phase 0.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
