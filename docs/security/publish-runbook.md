# 공개 레포 전환 런북

작성 2026-08-27. 대상: `younghwan91/quantbox` (현재 private).

## 판정: 현 상태로 공개 금지

차단 사유는 **하나**이고, 완전히 차단이다.

### 실 자격증명이 git 히스토리에 평문이고 이미 GitHub에 있다

| 커밋 | 내용 |
|---|---|
| `1f7fc27` | `.env` 전체 — 바이낸스 KEY/SECRET, 디스코드 웹훅, 봇 토큰 |
| `958fcfc` | 〃 |
| `f0f3abd` | 〃 |
| `54f9714` | `.env` 추적 해제 (이 시점부터 신규 노출 없음) |

- 유출된 4개 값은 **현재 `.env`의 값과 SHA-256 기준 동일**하다 (2026-08-27 대조).
- 해당 키는 **살아 있다** — 같은 날 잔고 조회 성공(1000.00 USDT).
- 이 커밋들은 `origin/main`을 포함해 **모든 원격 브랜치에서 도달 가능**하다.
  즉 이미 GitHub 서버에 있고, public 전환은 "노출"이 아니라 "검색 가능화"다.

신규 public 레포의 자격증명은 자동 스캐너가 수 분 내에 수집한다. 자금이 있는
선물 계좌 키이므로 실손실로 직결된다.

## 나머지 노출면 — 이상 없음 (2026-08-27 확인)

- 추적 파일 686개 시크릿 후보 **0건** (`scripts/check_secrets.py --rev`)
- `.gitignore` 정상: `.env`, `*.env`, `*.key`, `*.pem`, `/data/*`, `logs/`
- `.backup/` (미추적) 시크릿 없음 — **커밋하지 말 것**
- 저널/상태 DB, 계좌 잔고 문서 노출 없음. 팩 47.6MB로 정상

## 순서 (순서가 중요하다)

### 1. 키 폐기 — 사람만 할 수 있다 · 유일하게 위험을 실제로 없애는 단계

- [ ] 바이낸스 API 관리 → 해당 키 **삭제** 후 재발급
- [ ] 디스코드 웹훅 **삭제** 후 재생성
- [ ] 디스코드 봇 토큰 **리셋**
- [ ] 새 값을 로컬 `.env`에만 기록 (추적되지 않음)

히스토리를 아무리 정리해도 이 단계를 건너뛰면 무의미하다. 이미 클론·캐시됐을 수 있다.

### 2. 히스토리 — 재작성보다 새 레포를 권장

`git filter-repo`로 지우고 force push해도 GitHub는 도달 불가능해진 객체를 SHA로
한동안 계속 서빙한다(GitHub Support에 GC 요청이 필요). 공개가 목적이면 **새 레포**가
확실하고 빠르다.

준비 완료 상태다 — 로컬 브랜치 `public/clean`:
- 현재 트리 그대로, **부모 없는 단일 커밋** (히스토리 0)
- 감사 통과: `uv run python scripts/check_secrets.py --rev public/clean` → 0건

```bash
# 새 레포 생성 후 (예: quantbox-public)
git remote add public git@github.com:<user>/<new-repo>.git
git push public public/clean:main
```

히스토리를 살리고 싶다면 `git filter-repo --path .env --invert-paths` 경로로 가되
위 한계를 감수한다.

### 3. 공개 직전 재확인

- [ ] `uv run python scripts/check_secrets.py --rev <공개할 ref>` → 0건
- [ ] 키가 실제로 폐기됐는지 확인 (구 키로 API 호출 시 실패해야 정상)

## 재발 방지 (적용됨)

`scripts/check_secrets.py`가 pre-commit `secret-guard`로 등록돼 있다. 커밋 직전
스테이징을 검사해 실 키 형태·`.env` 계열·PEM을 차단한다. `.env.example`과
플레이스홀더는 통과시킨다(상시 실패하면 결국 가드를 끄게 되므로).

`--rev <ref>` 모드는 커밋된 트리 전체 감사용이며, 가드와 **동일한 판정기**(`scan_text`)를
쓴다 — 갈라지면 감사가 무의미해진다.

## 공개 시 별도 판단 사항 (차단 사유 아님)

- 전 히스토리에 커밋 작성자 이메일 `chyohw97@gmail.com`이 박혀 있다.
  `public/clean`은 단일 커밋이므로 여기서 정리하려면 그 커밋의 author를 바꾸면 된다.
- 라이브 전략 파라미터가 전부 공개된다. 알파 노출 문제이나, r122 측정에서 2026년
  엣지가 0으로 확인돼 실질 손실은 제한적이다. 판단은 운영자 몫.
