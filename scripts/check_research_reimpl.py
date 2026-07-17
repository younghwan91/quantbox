#!/usr/bin/env python3
"""Research 재구현 가드 — harness가 제공하는 것을 다시 짜는 걸 막는다.

배경: research 전역에 `get_commit` 33벌, `load_data` 16벌(동작 6종으로 갈라짐,
그중 4벌은 `_offset` 정렬 누락), 자체 numba sim 47개가 복제돼 있었다. 실험마다
sim/로딩/지표를 새로 짜기 때문에 **같은 백테스트 버그가 반복 재발**한다.
CLAUDE.md의 "재구현 금지"는 문장일 뿐 강제되지 않았다 — 이 스크립트가 게이트다.

동작 방식 (래칫):
  - 도입 시점의 기존 위반은 baseline(JSON)에 기록돼 **유예**된다. 옛 파일을 손대도
    커밋이 막히지 않는다. 다만 baseline은 늘어나지 않는다 — 새 위반은 실패.
  - 정당한 예외는 `# harness-exempt: <이유>` 주석으로 명시한다. 이유가 없으면 실패.
    (우회 해킹을 막으면서 '의도적 예외'를 기록으로 남긴다.)

사용:
  check_research_reimpl.py <file>...          # 검사 (pre-commit)
  check_research_reimpl.py --update-baseline  # 현재 위반을 baseline으로 갱신

Exit code 0 = clean, 1 = 새 위반.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "scripts" / "research_reimpl_baseline.json"
RESEARCH_DIR = REPO_ROOT / "research"

EXEMPT_MARKER = "# harness-exempt:"

#: 재구현 금지 심볼 -> harness에서 가져오는 법.
BANNED: dict[str, str] = {
    "load_data": "from quantbox.research.harness import MarketData  # MarketData.load(...)",
    "load_all_data": "from quantbox.research.harness import MarketData  # MarketData.load(...)",
    "get_commit": "from quantbox.research.harness import get_commit",
    "get_commit_short": "from quantbox.research.harness import get_commit_short",
    "portfolio_sim": "from quantbox.research.harness import portfolio_sim",
    "compute_metrics": "from quantbox.research.harness import compute_metrics  # 또는 report.standard(...)",
    "bootstrap_pf": "from quantbox.research.harness import bootstrap_pf",
    "compute_sma": "from quantbox.research.harness import compute_sma",
    "compute_momentum": "from quantbox.research.harness import compute_momentum",
    "momentum8": "from quantbox.research.harness import compute_momentum  # compute_momentum(close, 8)",
    "build_resampled": "from quantbox.research.harness import build_resampled  # 또는 MarketData.resample(ratio)",
    "resample_ohlcv": "from quantbox.research.harness import resample_ohlcv",
}

#: 자체 sim 커널 정의 금지. 공용 드라이버 sim_drive를 쓴다.
SIM_PREFIX = "sim_"
SIM_HINT = "from quantbox.research.harness import sim_drive  # 전략별로는 무상태 조건 커널만 작성"

Violation = tuple[int, str, str]  # (lineno, symbol, hint)


def _exempt_reason(lines: list[str], node: ast.FunctionDef) -> str | None:
    """def 줄 또는 바로 윗줄들의 `# harness-exempt: <이유>` 를 찾는다."""
    candidates: list[str] = []
    idx = node.lineno - 1
    if 0 <= idx < len(lines):
        candidates.append(lines[idx])
    # 데코레이터/주석이 위에 붙는 경우 — def 위 3줄까지 훑는다
    for back in range(1, 4):
        j = idx - back
        if 0 <= j < len(lines):
            candidates.append(lines[j])
    for line in candidates:
        if EXEMPT_MARKER in line:
            reason = line.split(EXEMPT_MARKER, 1)[1].strip()
            return reason or None
    return None


def scan_file(path: Path) -> list[Violation]:
    """파일에서 재구현 위반을 찾는다 (exempt 처리 후)."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    lines = src.splitlines()
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if name in BANNED:
            hint = BANNED[name]
        elif name.startswith(SIM_PREFIX):
            hint = SIM_HINT
        else:
            continue
        if _exempt_reason(lines, node) is not None:
            continue
        out.append((node.lineno, name, hint))
    return out


def _rel(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_baseline() -> dict[str, list[str]]:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return data.get("grandfathered", {})


def build_baseline() -> dict[str, list[str]]:
    """현재 research/ 전체를 훑어 위반 목록을 만든다."""
    found: dict[str, list[str]] = {}
    for path in sorted(RESEARCH_DIR.rglob("*.py")):
        v = scan_file(path)
        if v:
            found[_rel(path)] = sorted({name for _, name, _ in v})
    return found


def update_baseline() -> int:
    grandfathered = build_baseline()
    total = sum(len(v) for v in grandfathered.values())
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "harness 도입 시점에 이미 존재하던 재구현 — 유예 목록. "
                    "여기 없는 새 재구현은 pre-commit이 막는다. 이 목록은 늘리지 않고 줄여 나간다. "
                    "갱신: uv run python scripts/check_research_reimpl.py --update-baseline"
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
    print(f"baseline 갱신: {len(grandfathered)} 파일, {total} 심볼 -> {_rel(BASELINE_PATH)}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    if args.update_baseline:
        return update_baseline()

    baseline = load_baseline()
    new_violations: dict[str, list[Violation]] = {}

    for f in args.files:
        path = Path(f)
        if path.suffix != ".py" or not path.exists():
            continue
        rel = _rel(path)
        allowed = set(baseline.get(rel, []))
        fresh = [v for v in scan_file(path) if v[1] not in allowed]
        if fresh:
            new_violations[rel] = fresh

    if not new_violations:
        return 0

    print("\n[research-reimpl-guard] 이미 있는 코드를 다시 짜고 있다:\n", file=sys.stderr)
    for rel, violations in sorted(new_violations.items()):
        for lineno, name, hint in violations:
            print(f"  {rel}:{lineno}: def {name}(...) — 재구현 금지", file=sys.stderr)
            print(f"      대신: {hint}", file=sys.stderr)
        print(file=sys.stderr)
    print(
        "이유: 실험마다 sim/로딩/지표를 새로 짜서 같은 백테스트 버그가 반복 재발했다.\n"
        "  (load_data 16벌이 동작 6종으로 갈라졌고, 4벌은 심볼 정렬 _offset이 누락됐다.)\n"
        "  harness는 라이브가 쓰는 바로 그 함수를 재수출한다 -> parity가 구조적으로 보장된다.\n\n"
        "  from quantbox.research.harness import MarketData, report, experiment\n\n"
        "정당한 예외라면 def 위에 이유를 남긴다:  # harness-exempt: <이유>",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
