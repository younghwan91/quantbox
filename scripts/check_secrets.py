"""시크릿 커밋 차단 가드 — 2026-08-27.

이 레포는 `.env`를 3개 커밋(1f7fc27 / 958fcfc / f0f3abd)에 통째로 올린 전력이 있다.
그 키들은 현재 라이브 키와 바이트 단위로 동일했고, 이미 GitHub origin/main에 도달
가능한 상태였다. 즉 `.gitignore`만으로는 부족했다 — 한 번 뚫리면 되돌릴 수 없다.

이 가드는 커밋 **직전**에 스테이징된 내용만 본다. 히스토리 정화와는 별개로, 앞으로
같은 사고가 나지 않게 하는 쪽이다.

탐지 대상:
  - 알려진 키 이름에 실값이 붙은 형태 (`BINANCE_API_SECRET=<20자 이상>`)
  - Discord 웹훅 URL / 봇 토큰 형태
  - PEM 개인키 헤더
  - `.env` 계열 파일이 스테이징된 경우 (내용 불문 차단)

플레이스홀더(`your_..._here`, `xxx`, `<...>`, `REDACTED`)는 통과시킨다 —
`.env.example`이 걸리면 가드가 꺼지기 때문이다.
"""

from __future__ import annotations

import re
import subprocess
import sys

#: 값이 붙으면 안 되는 환경변수 이름들.
SECRET_KEYS = (
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "DISCORD_BOT_TOKEN",
    "DISCORD_WEBHOOK_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

#: 이 중 하나라도 값에 들어 있으면 실제 시크릿이 아니라 예시로 본다.
PLACEHOLDERS = ("your_", "_here", "xxx", "<", "redacted", "example", "changeme", "dummy", "...")

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("환경변수에 실값이 붙었다", re.compile(rf"\b({'|'.join(SECRET_KEYS)})\s*=\s*[\"']?([^\s\"']{{20,}})")),
    ("Discord 웹훅 URL", re.compile(r"discord(?:app)?\.com/api/webhooks/\d{15,}/[\w\-]{50,}")),
    ("Discord 봇 토큰", re.compile(r"\b[MN][A-Za-z0-9_\-]{22,}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{25,}\b")),
    ("PEM 개인키", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

#: 내용과 무관하게 커밋을 막을 파일명 패턴 (`.env.example`만 예외).
FORBIDDEN_FILES = re.compile(r"(^|/)\.env(\.|$)|(^|/)[^/]*\.env$|(^|/)secrets\.json$|\.(pem|key|p12)$")


def _is_placeholder(value: str) -> bool:
    low = value.lower()
    return any(p in low for p in PLACEHOLDERS)


def scan_text(path: str, content: str) -> list[str]:
    """한 파일의 내용에서 시크릿 후보를 찾는다 — 가드와 감사가 공유하는 유일 판정기."""
    found: list[str] = []
    for label, pat in PATTERNS:
        for m in pat.finditer(content):
            value = m.group(2) if m.re.groups >= 2 else m.group(0)
            if _is_placeholder(value):
                continue
            line = content[: m.start()].count("\n") + 1
            found.append(f"{path}:{line}: {label}")
    return found


def scan_rev(rev: str) -> list[str]:
    """git rev의 트리 전체를 훑는다 — 공개 직전 감사용.

    스테이징이 아니라 **커밋된 트리**를 보므로, 공개하려는 브랜치가 실제로 깨끗한지
    확인할 수 있다. 판정은 `scan_text` 하나만 쓴다(가드와 갈라지면 감사가 무의미하다).
    """
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    out: list[str] = []
    for path in names:
        if not path.strip():
            continue
        blob = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, errors="ignore")
        if blob.returncode:
            continue
        if FORBIDDEN_FILES.search(path) and not path.endswith(".env.example"):
            out.append(f"{path}: 시크릿 파일이 트리에 있다")
            continue
        out.extend(scan_text(path, blob.stdout))
    return out


def _staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [f for f in out.stdout.splitlines() if f.strip()]


def _staged_content(path: str) -> str:
    out = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True, check=False)
    return out.stdout if out.returncode == 0 else ""


def main(argv: list[str]) -> int:
    # `--rev <ref>`: 커밋된 트리 전체 감사 (공개 직전 확인용).
    if len(argv) >= 3 and argv[1] == "--rev":
        violations = scan_rev(argv[2])
        if violations:
            print(f"🔴 {argv[2]} 트리에서 시크릿 후보 {len(violations)}건:\n", file=sys.stderr)
            for v in violations:
                print(f"  {v}", file=sys.stderr)
            return 1
        print(f"✅ {argv[2]} 트리 깨끗 — 시크릿 후보 0건")
        return 0

    files = argv[1:] or _staged_files()
    violations: list[str] = []

    for path in files:
        if path.endswith(".env.example"):
            continue
        if FORBIDDEN_FILES.search(path):
            violations.append(f"{path}: 시크릿 파일은 커밋할 수 없다 (.gitignore 확인)")
            continue

        content = _staged_content(path)
        if not content:
            continue
        violations.extend(scan_text(path, content))

    if violations:
        print("🔴 시크릿으로 보이는 내용이 스테이징됐다 — 커밋을 막는다:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\n한 번 커밋되면 원격에서 되돌릴 수 없다. 값을 .env(추적 안 됨)로 옮기고 다시 시도해라.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
