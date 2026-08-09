#!/usr/bin/env python3
"""Research parameter hardcoding guard.

Pre-commit hook that detects hardcoded parameter values in research scripts.
Run via .pre-commit-config.yaml on files matching research/**/*.py.

Catches:
  - VR_LN = <number>  (should come from cfg.base_leverage / cfg.max_concurrent)
  - MA_LN = <number>  (should come from cfg.ma_cluster_leverage / cfg.max_concurrent)
  - d_tp = <float>    (should come from cfg.d_tp)
  - d_sl = <float>    (should come from cfg.d_sl)
  - New imports of backtest_v14_3_0 (stale parameters — r53 incident)

동작 방식 (래칫, reimpl 가드와 동일):
  - 도입 시점에 이미 있던 하드코딩은 baseline(JSON)에 (파일 -> 시그니처 집합)로
    기록돼 **유예**된다. 옛 sweep 실험 파일을 마이그레이션으로 다시 스테이징해도
    커밋이 막히지 않는다. 다만 baseline은 늘어나지 않는다 — 파일에 **새 하드코딩
    값**(baseline에 없는 시그니처)이 들어오면 실패한다.
  - d_sl 은 줄 끝 `# from cfg` 주석으로도 개별 면제된다(기존 escape 유지).
  - stale import(backtest_v14_3_0)는 _LEGACY_STALE_ALLOWED 명시 목록으로 유예(기존 유지).

사용:
  check_research_params.py <file>...          # 검사 (pre-commit)
  check_research_params.py --update-baseline  # 현재 위반을 baseline으로 갱신

Exit code 0 = clean, 1 = 새 위반.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
#  Files allowed to have stale backtest_v14_3_0 imports (legacy, archived)
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
BASELINE_PATH = REPO_ROOT / "scripts" / "research_param_baseline.json"

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
#  Patterns: (regex with a value capture group, label, message)
#  label+value form the ratchet signature "label=value".
# ──────────────────────────────────────────────────────────────────────────────
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"^\s*VR_LN\s*=\s*([\d.]+)\s*$"),
        "VR_LN",
        "Hardcoded VR_LN. Use: VR_LN = cfg.base_leverage / cfg.max_concurrent",
    ),
    (
        re.compile(r"^\s*MA_LN\s*=\s*([\d.]+)\s*$"),
        "MA_LN",
        "Hardcoded MA_LN. Use: MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent",
    ),
    (
        re.compile(r"^\s*LN\s*=\s*([\d.]+)\s*$"),
        "LN",
        "Hardcoded LN. Use: LN = cfg.base_leverage / cfg.max_concurrent",
    ),
    (
        re.compile(r"['\"]d_tp['\"]:\s*(0\.\d+)"),
        "d_tp",
        "Hardcoded d_tp value. Use: 'tp': cfg.d_tp (or cfg.s_tp for straddle)",
    ),
    (
        re.compile(r"['\"]d_sl['\"]:\s*(0\.\d+)"),
        "d_sl",
        "Hardcoded d_sl value. Use: 'd_sl': cfg.d_sl",
    ),
]

# 줄 끝 이 주석이 있으면 해당 줄의 하드코딩 파라미터는 개별 면제(config에서 온 값임을 명시).
# (기존 negative-lookahead 방식은 \d+ 백트래킹으로 우회되던 버그가 있었다 → 줄 단위로 고정.)
_FROM_CFG_ESCAPE = "# from cfg"

_STALE_IMPORT_PATTERN = re.compile(r"from\s+\S*backtest_v14_3_0\s+import|import\s+\S*backtest_v14_3_0")

# Violation: (lineno, line_text, message, signature). sig is None for stale-import
# violations (governed by _LEGACY_STALE_ALLOWED, not the ratchet baseline).
Violation = tuple[int, str, str, str | None]


def _rel(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def check_file(path: Path) -> list[Violation]:
    """Return list of (line_no, line_text, message, sig) for violations."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    rel = _rel(path)
    violations: list[Violation] = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip comment-only lines
        if stripped.startswith("#"):
            continue

        # Stale import check (only for new files) — kept on the explicit allowlist.
        if _STALE_IMPORT_PATTERN.search(line) and rel not in _LEGACY_STALE_ALLOWED:
            violations.append(
                (
                    lineno,
                    line.rstrip(),
                    "Stale import: backtest_v14_3_0 has wrong parameters "
                    "(D_TRAIL_ACTIVATE=0.02 vs production 0.010). "
                    "Use: from quantbox.backtest.trade_sim import collect_trades",
                    None,
                )
            )

        # `# from cfg` 면제 — 이 줄의 하드코딩 파라미터 검사는 건너뛴다.
        if _FROM_CFG_ESCAPE in line:
            continue

        # Hardcoded parameter patterns — ratchet-tracked by signature.
        for pattern, label, msg in _PATTERNS:
            m = pattern.search(line)
            if m:
                value = m.group(1)
                violations.append((lineno, line.rstrip(), msg, f"{label}={value}"))

    return violations


# ──────────────────────────────────────────────────────────────────────────────
#  Ratchet baseline (mirror of check_research_reimpl.py)
# ──────────────────────────────────────────────────────────────────────────────
def load_baseline() -> dict[str, list[str]]:
    if not BASELINE_PATH.exists():
        return {}
    import json

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return data.get("grandfathered", {})


def build_baseline() -> dict[str, list[str]]:
    """현재 research/ 전체를 훑어 하드코딩 시그니처 목록을 만든다(stale은 제외)."""
    found: dict[str, list[str]] = {}
    for path in sorted(RESEARCH_DIR.rglob("*.py")):
        sigs = sorted({v[3] for v in check_file(path) if v[3] is not None})
        if sigs:
            found[_rel(path)] = sigs
    return found


def update_baseline() -> int:
    import json

    grandfathered = build_baseline()
    total = sum(len(v) for v in grandfathered.values())
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "가드 도입 시점에 이미 존재하던 하드코딩 파라미터 — 유예 목록. "
                    "여기 없는 새 값은 pre-commit이 막는다. 이 목록은 늘리지 않고 줄여 나간다. "
                    "갱신: uv run python scripts/check_research_params.py --update-baseline"
                ),
                "grandfathered": grandfathered,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"baseline 갱신: {len(grandfathered)} 파일, {total} 시그니처 -> {_rel(BASELINE_PATH)}")
    return 0


def main(argv: list[str]) -> int:
    """Check files for violations. Return exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    if args.update_baseline:
        return update_baseline()

    baseline = load_baseline()
    all_violations: dict[str, list[Violation]] = {}

    for file_str in args.files:
        path = Path(file_str)
        if not path.exists() or path.suffix != ".py":
            continue
        rel = _rel(path)
        allowed = set(baseline.get(rel, []))
        # sig is None (stale import) → always report; sig in baseline → grandfathered.
        fresh = [v for v in check_file(path) if v[3] is None or v[3] not in allowed]
        if fresh:
            all_violations[file_str] = fresh

    if not all_violations:
        return 0

    print("\n[research-param-guard] Hardcoded parameters detected:\n", file=sys.stderr)
    for file_str, violations in sorted(all_violations.items()):
        rel = _rel(Path(file_str))
        for lineno, line, msg, _sig in violations:
            print(f"  {rel}:{lineno}: {msg}", file=sys.stderr)
            print(f"    {line}", file=sys.stderr)
        print(file=sys.stderr)

    print(
        "Fix: Read ALL parameters from VolStraddleConfig().\n"
        "  from quantbox.strategy.vol_straddle.config import VolStraddleConfig\n"
        "  cfg = VolStraddleConfig()\n"
        "  VR_LN = cfg.base_leverage / cfg.max_concurrent  # 3.0\n"
        "기존(레거시) 하드코딩을 옮기기만 하는 마이그레이션이라면:\n"
        "  uv run python scripts/check_research_params.py --update-baseline\n"
        "See backtesting.instructions.md Phase 0.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
