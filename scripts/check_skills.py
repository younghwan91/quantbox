#!/usr/bin/env python3
"""Skills audit: validate .github/instructions/*.instructions.md files.

Usage:
    python scripts/check_skills.py              # full audit
    python scripts/check_skills.py --post-fix   # post-fix checklist only
    python scripts/check_skills.py --ci         # CI mode (exit 1 on failure)

Checks:
    1. Each .instructions.md has valid applyTo frontmatter
    2. File size warning (>400 lines → flag for pruning)
    3. Post-fix: verify debugging.instructions.md was updated recently
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SKILLS_DIR = ROOT / ".github" / "instructions"
MAX_LINES = 400
WARN_LINES = 300

# Which skills should be updated for each fix type
FIX_SKILL_MAP = {
    "debug": "debugging.instructions.md",
    "trade": "live_trading.instructions.md",
    "siz": "position_sizing.instructions.md",    # sizing
    "backtest": "backtesting.instructions.md",
    "research": "backtesting.instructions.md",
    "incident": "debugging.instructions.md",
    "hotfix": "debugging.instructions.md",
}


def check_frontmatter(path: Path) -> list[str]:
    errors = []
    content = path.read_text()
    if not content.startswith("---"):
        errors.append(f"  ✗ Missing frontmatter in {path.name}")
        return errors
    match = re.search(r"^---\s*\napplyTo:\s*\"(.+?)\"\s*\n---", content, re.MULTILINE)
    if not match:
        errors.append(f"  ✗ Missing or invalid applyTo in {path.name}")
    else:
        errors.append(f"  ✓ applyTo: {match.group(1)[:60]}")
    return errors


def check_sizes() -> list[str]:
    results = []
    for f in sorted(SKILLS_DIR.glob("*.instructions.md")):
        lines = len(f.read_text().splitlines())
        if lines > MAX_LINES:
            results.append(f"  ⚠️  {f.name}: {lines} lines (>{MAX_LINES}) — prune needed")
        elif lines > WARN_LINES:
            results.append(f"  📏 {f.name}: {lines} lines (>{WARN_LINES}) — watch size")
        else:
            results.append(f"  ✓  {f.name}: {lines} lines")
    return results


def get_recent_fix_commits(n: int = 5) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "log", f"-{n}", "--oneline", "--format=%s"],
            capture_output=True, text=True, cwd=ROOT
        )
        return [
            line for line in result.stdout.splitlines()
            if re.match(r"^(fix|hotfix|bug)", line, re.IGNORECASE)
        ]
    except Exception:
        return []


def check_skill_updated_for_fix(commit_msg: str) -> list[str]:
    """Check if relevant skill was updated in same commit."""
    warnings = []
    msg_lower = commit_msg.lower()

    expected_skills = set()
    for keyword, skill_file in FIX_SKILL_MAP.items():
        if keyword in msg_lower:
            expected_skills.add(skill_file)

    if not expected_skills:
        expected_skills.add("debugging.instructions.md")

    try:
        # Check files changed in last commit
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "HEAD", "--name-only"],
            capture_output=True, text=True, cwd=ROOT
        )
        changed = result.stdout
    except Exception:
        changed = ""

    for skill in expected_skills:
        if skill not in changed:
            warnings.append(
                f"  ⚠️  Fix commit detected but '{skill}' not updated.\n"
                f"     Run: view .github/instructions/{skill} and add the new pattern."
            )
        else:
            warnings.append(f"  ✓  {skill} updated in this commit")

    return warnings


def post_fix_checklist() -> None:
    print("\n🔧 POST-FIX SKILL CHECKLIST")
    print("=" * 50)

    recent = get_recent_fix_commits(1)
    if not recent:
        print("  No recent fix commits detected.")
        return

    for commit in recent:
        print(f"\nFix: {commit}")
        warnings = check_skill_updated_for_fix(commit)
        for w in warnings:
            print(w)

    print("\n📋 Required actions after any bug fix:")
    print("  1. Add incident to debugging.instructions.md (Section 1)")
    print("  2. Add row to backtesting.instructions.md Section 11 (if backtest-related)")
    print("  3. Add rule to copilot-instructions.md incident table")
    print("  4. Commit skill updates in same PR as fix")


def full_audit(ci_mode: bool = False) -> int:
    exit_code = 0
    print("\n📚 QUANTBOX SKILLS AUDIT")
    print("=" * 50)

    print("\n[1] Frontmatter validation:")
    for f in sorted(SKILLS_DIR.glob("*.instructions.md")):
        results = check_frontmatter(f)
        for r in results:
            print(r)
            if "✗" in r:
                exit_code = 1

    print("\n[2] File size check (warn>{WARN_LINES}, prune>{MAX_LINES} lines):")
    for r in check_sizes():
        print(r)
        if "⚠️" in r and ci_mode:
            exit_code = 1

    print("\n[3] Recent fix commits without skill updates:")
    recent = get_recent_fix_commits(3)
    if not recent:
        print("  ✓ No recent fix commits, or all covered.")
    for commit in recent:
        results = check_skill_updated_for_fix(commit)
        for r in results:
            print(r)

    print("\n[4] Skills overview:")
    total_lines = 0
    for f in sorted(SKILLS_DIR.glob("*.instructions.md")):
        lines = len(f.read_text().splitlines())
        total_lines += lines
    print(f"  Total: {len(list(SKILLS_DIR.glob('*.instructions.md')))} skill files, {total_lines} lines")

    status = "✅ PASS" if exit_code == 0 else "❌ FAIL"
    print(f"\n{status}\n")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantBox skills audit")
    parser.add_argument("--post-fix", action="store_true", help="Post-fix checklist only")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit 1 on issues")
    args = parser.parse_args()

    if args.post_fix:
        post_fix_checklist()
    else:
        code = full_audit(ci_mode=args.ci)
        if args.ci:
            sys.exit(code)


if __name__ == "__main__":
    main()
