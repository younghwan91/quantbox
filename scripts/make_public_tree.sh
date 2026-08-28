#!/usr/bin/env bash
# 공개용 클린 트리를 현재 HEAD에서 재생성하고 감사한다 — 2026-08-27.
#
# `public/clean`은 브랜치를 따라가지 않는다. HEAD 트리의 **스냅샷**이라, 브랜치에
# 커밋해도 자동으로 갱신되지 않는다. 실제로 한 번 낡은 채로 "공개 준비 완료"라고
# 판단할 뻔했다(conftest 수정 이전 트리였다).
#
# 그래서 재생성과 감사를 한 명령으로 묶는다. 둘을 따로 두면 한쪽만 하게 된다 —
# 이 레포에서 반복된 실패 모양이 정확히 그것이다(ci.yml env ↔ pre-commit 훅).
#
# 사용: scripts/make_public_tree.sh
set -euo pipefail

BRANCH="public/clean"
MSG="QuantBox — 초기 공개 커밋

Binance USDT-M 무기한 선물 알고리즘 트레이딩 시스템.

히스토리 없이 시작한다. 이전 히스토리에는 실 API 자격증명이 평문으로 들어간
커밋이 있어 공개할 수 없다. 그 커밋들은 이 트리에 반영되지 않는다."

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠️  워킹트리에 커밋되지 않은 변경이 있다. 스냅샷은 HEAD 기준이므로" >&2
  echo "    그 변경은 공개 트리에 들어가지 않는다. 먼저 커밋해라." >&2
  git status --short >&2
  exit 1
fi

# 작성자 신원 게이트 — 회사/조직 이메일이 공개본에 실리면 안 된다.
# 2026-08-28: 다른 세션이 `git -c user.email=<회사주소>`로 커밋해 회사 계정이 author로
# 박혔고, cherry-pick이 그걸 승계했다. 레포 설정은 개인 이메일이었는데도 그랬다.
# 사용자에게는 고용 관련 위험이 걸린 항목이라, 사람의 기억이 아니라 게이트로 막는다.
# 차단할 도메인은 환경변수로 넘길 수 있다 (기본: 개인 도메인이 아닌 모든 것을 훑지는
# 않고, 알려진 회사 도메인만 본다).
BLOCKED_EMAIL_PATTERN="${BLOCKED_EMAIL_PATTERN:-employer.example\.com}"
BAD_AUTHORS=$(git log --format='%ae %ce' HEAD | tr ' ' '\n' | sort -u | grep -iE "$BLOCKED_EMAIL_PATTERN" || true)
if [ -n "$BAD_AUTHORS" ]; then
  echo "🔴 커밋 작성자/커미터에 차단 대상 이메일이 있다 — 공개 트리를 만들지 않는다:" >&2
  echo "$BAD_AUTHORS" | sed 's/^/    /' >&2
  echo "    해당 커밋: " >&2
  git log --format='%h %ae %ce %s' HEAD | grep -iE "$BLOCKED_EMAIL_PATTERN" | sed 's/^/    /' >&2
  exit 1
fi

SHA=$(git commit-tree "$(git rev-parse HEAD^{tree})" -m "$MSG")
git branch -f "$BRANCH" "$SHA"

echo "$BRANCH = $(git rev-parse --short "$BRANCH")  (HEAD $(git rev-parse --short HEAD) 스냅샷)"
echo "  커밋 수: $(git rev-list --count "$BRANCH")  파일: $(git ls-tree -r --name-only "$BRANCH" | wc -l)"
echo

# 공개 트리 자체의 작성자도 확인한다 (commit-tree는 현재 git 설정을 쓴다).
TREE_AUTHOR=$(git log -1 --format='%ae %ce' "$BRANCH")
if echo "$TREE_AUTHOR" | grep -iqE "$BLOCKED_EMAIL_PATTERN"; then
  echo "🔴 공개 트리 커밋 자체가 차단 이메일로 서명됐다: $TREE_AUTHOR" >&2
  exit 1
fi
echo "작성자: $TREE_AUTHOR"
echo

uv run python scripts/check_secrets.py --rev "$BRANCH"
echo
echo "자격증명 상태:"
uv run python scripts/check_credentials.py || {
  echo
  echo "🔴 유출된 자격증명이 살아 있다 — push 금지." >&2
  exit 1
}
echo
echo "✅ 트리·자격증명 모두 통과 — push 가능"
