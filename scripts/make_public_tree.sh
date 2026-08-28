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

SHA=$(git commit-tree "$(git rev-parse HEAD^{tree})" -m "$MSG")
git branch -f "$BRANCH" "$SHA"

echo "$BRANCH = $(git rev-parse --short "$BRANCH")  (HEAD $(git rev-parse --short HEAD) 스냅샷)"
echo "  커밋 수: $(git rev-list --count "$BRANCH")  파일: $(git ls-tree -r --name-only "$BRANCH" | wc -l)"
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
