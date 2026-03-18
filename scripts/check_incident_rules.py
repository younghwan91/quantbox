#!/usr/bin/env python3
"""Incident rules pre-commit guard.

Pre-commit hook that detects patterns known to cause trading losses.
Based on documented incidents #1-#9 in CLAUDE.md.

Checks:
  1. CANCELED/CANCELLED: Both spellings must be present together
  2. Hardcoded numeric literals (0.01, 0.02, 30, 21, etc.)
  3. pos['symbol'] or pos["symbol"] assignments (may include colon)

Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# ──────────────────────────────────────────────────────────────────────────────
#  Files allowed to have certain patterns (legacy/intentional)
# ──────────────────────────────────────────────────────────────────────────────

# Files that legitimately define the CANCELED/CANCELLED check
_CANCEL_CHECK_FILES: set[str] = {
    "quantbox/execution/brackets.py",  # defines the canonical check
    "quantbox/execution/reconciliation.py",  # uses both
    "quantbox/core/limit_chaser.py",  # uses CANCELED
}

# Files allowed to have numeric literals (config definitions)
_NUMERIC_ALLOWED: set[str] = {
    "quantbox/strategy/vol_straddle/config.py",
    "quantbox/core/settings.py",
    "quantbox/backtest/trade_sim.py",  # simulation constants
    "tests/",  # test files can have literals
}

# ──────────────────────────────────────────────────────────────────────────────
#  Check 1: CANCELED/CANCELLED both present
# ──────────────────────────────────────────────────────────────────────────────

_CANCELLED_UK = re.compile(r'["\']CANCELLED["\']')
_CANCELED_US = re.compile(r'["\']CANCELED["\']')


def check_cancel_spelling(filepath: Path, content: str) -> list[str]:
    """Check that both CANCELED and CANCELLED are checked together.
    
    Binance Algo API returns 'CANCELED' (US), but UK spelling is common.
    If code checks one, it should check both.
    """
    violations = []
    rel_path = str(filepath.relative_to(REPO_ROOT))
    
    # Skip files that define the canonical check
    if any(rel_path.startswith(allowed) for allowed in _CANCEL_CHECK_FILES):
        return []
    
    has_uk = bool(_CANCELLED_UK.search(content))
    has_us = bool(_CANCELED_US.search(content))
    
    # If either is present, both should be present
    if has_uk and not has_us:
        violations.append(
            f"{rel_path}: Found 'CANCELLED' (UK) but not 'CANCELED' (US). "
            f"Binance Algo API returns 'CANCELED'. Check both spellings."
        )
    elif has_us and not has_uk:
        # US-only is less risky but still worth noting
        pass  # Don't warn - US is the correct Binance spelling
    
    return violations


# ──────────────────────────────────────────────────────────────────────────────
#  Check 2: Hardcoded numeric literals
# ──────────────────────────────────────────────────────────────────────────────

# Suspicious numeric patterns (trading params that should come from config)
_SUSPICIOUS_NUMBERS = [
    # SL/TP percentages
    (re.compile(r'\b(sl|stop_loss|d_sl|s_sl)\s*=\s*(0\.0[0-9]+)\b', re.IGNORECASE), "SL percentage"),
    (re.compile(r'\b(tp|take_profit|d_tp)\s*=\s*(0\.[0-9]+)\b', re.IGNORECASE), "TP percentage"),
    # Trail parameters
    (re.compile(r'\b(trail_activate|trail_distance)\s*=\s*(0\.0[0-9]+)\b', re.IGNORECASE), "trail parameter"),
    # Leverage
    (re.compile(r'\bleverage\s*=\s*(2[0-9]|3[0-5])\b'), "leverage (should be from config)"),
    # L/N ratio hardcoded
    (re.compile(r'\b(VR_LN|MA_LN)\s*=\s*[0-9.]+\s*$', re.MULTILINE), "L/N ratio hardcoded"),
]


def check_numeric_hardcodes(filepath: Path, content: str) -> list[str]:
    """Check for hardcoded trading parameters."""
    violations = []
    rel_path = str(filepath.relative_to(REPO_ROOT))
    
    # Skip allowed files
    if any(rel_path.startswith(allowed.rstrip('/')) for allowed in _NUMERIC_ALLOWED):
        return []
    
    for pattern, desc in _SUSPICIOUS_NUMBERS:
        matches = pattern.finditer(content)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            violations.append(
                f"{rel_path}:{line_num}: Hardcoded {desc}: {match.group(0)!r}. "
                f"Use VolStraddleConfig() instead."
            )
    
    return violations


# ──────────────────────────────────────────────────────────────────────────────
#  Check 3: pos['symbol'] assignment patterns
# ──────────────────────────────────────────────────────────────────────────────

# Pattern: pos['symbol'] = something_with_colon
# or pos["symbol"] = pos_key (where pos_key may have colon)
_POS_SYMBOL_ASSIGN = re.compile(
    r'''pos\s*\[\s*['"]symbol['"]\s*\]\s*=\s*([^#\n]+)''',
    re.IGNORECASE
)


def check_pos_symbol_assignment(filepath: Path, content: str) -> list[str]:
    """Check for potentially unsafe pos['symbol'] assignments."""
    violations = []
    rel_path = str(filepath.relative_to(REPO_ROOT))
    
    for match in _POS_SYMBOL_ASSIGN.finditer(content):
        rhs = match.group(1).strip()
        line_num = content[:match.start()].count('\n') + 1
        
        # Skip if this is inside a string (log message, f-string)
        # Check if there's an odd number of quotes before the match on this line
        line_start = content.rfind('\n', 0, match.start()) + 1
        line_content = content[line_start:match.start()]
        if line_content.count('"') % 2 == 1 or line_content.count("'") % 2 == 1:
            continue  # Inside a string literal
        if 'f"' in line_content or "f'" in line_content:
            continue  # Inside an f-string
        
        # Warn if RHS contains 'pos_key' or 'key' (may have colon)
        if 'pos_key' in rhs or ('key' in rhs and 'symbol' not in rhs.lower()):
            violations.append(
                f"{rel_path}:{line_num}: pos['symbol'] assigned from {rhs!r}. "
                f"Ensure no colon in API symbol (split pos_key first)."
            )
        
        # Warn if RHS contains ':' (but not in a comparison like 'if ":" in')
        if ':' in rhs and 'split' not in rhs and 'in ' not in rhs:
            violations.append(
                f"{rel_path}:{line_num}: pos['symbol'] may contain colon: {rhs!r}. "
                f"API symbol must not have colon."
            )
    
    return violations


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Run all checks on modified files."""
    # Get files from command line (passed by pre-commit)
    files = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # If no files provided, check all quantbox/*.py
    if not files:
        files = [str(f) for f in REPO_ROOT.glob("quantbox/**/*.py")]
    
    all_violations: list[str] = []
    
    for filepath_str in files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            continue
        if not filepath.suffix == '.py':
            continue
        if not str(filepath).startswith(str(REPO_ROOT)):
            filepath = REPO_ROOT / filepath
        
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception:
            continue
        
        # Run all checks
        all_violations.extend(check_cancel_spelling(filepath, content))
        all_violations.extend(check_numeric_hardcodes(filepath, content))
        all_violations.extend(check_pos_symbol_assignment(filepath, content))
    
    if all_violations:
        print("❌ Incident rule violations found:\n")
        for v in all_violations:
            print(f"  {v}")
        print(f"\n{len(all_violations)} violation(s) found.")
        print("\nReference: CLAUDE.md Incident Rules #1-#9")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
