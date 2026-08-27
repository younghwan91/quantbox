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

신규 public 레포의 자격증명은 자동 스캐너가 수 분 내에 수집한다.

### 실제 위험도 — 자격증명별로 크게 다르다 (2026-08-27 실측)

**바이낸스 키: 즉시 자금 유출은 불가.**

| 권한 | 값 |
|---|---|
| `enableWithdrawals` | **False** — 출금 불가 |
| `ipRestrict` | **True** — 화이트리스트 IP 외 사용 불가 |
| `enableFutures` / `enableSpotAndMarginTrading` | True |
| `permitsUniversalTransfer` | True (계좌 **내부** 지갑 간 이동만) |

악용 흔적 없음:
- 2026-02-01 이후 **출금 0건** (유출 커밋은 2026-02-17)
- 계좌 밖으로 나간 이체 0건 (`TRANSFER` 4건 전부 2026-02-11 **입금**, 유출 이전)
- 2026-07-01 이후 선물 체결 0건 / income 0건

즉 이 키는 IP 제한과 출금 차단 두 겹에 막혀 있고, 실제로 악용된 적도 없다.
잔여 위험은 화이트리스트 IP가 풀리거나 그 IP에서 접근이 가능해질 때의 **거래 권한**이다.

**디스코드 자격증명: 이쪽이 더 위험하다. IP 제한이 걸리지 않는다.**

| 대상 | 상태 (2026-08-27 확인) |
|---|---|
| 웹훅 `Captain Hook` (channel `000000000000000000`) | 🔴 **살아 있음** — URL만 알면 누구나 그 채널에 임의 메시지 게시 |
| 봇 토큰 `QuantBox` (id `000000000000000000`) | 🔴 **살아 있음** — 봇 권한 전체 탈취 가능 |

두 값 모두 URL/토큰만 있으면 어디서든 쓸 수 있다. **폐기 우선순위는 바이낸스 키보다 높다.**

## 나머지 노출면 — 이상 없음 (2026-08-27 확인)

- 추적 파일 686개 시크릿 후보 **0건** (`scripts/check_secrets.py --rev`)
- `.gitignore` 정상: `.env`, `*.env`, `*.key`, `*.pem`, `/data/*`, `logs/`
- `.backup/` (미추적) 시크릿 없음 — **커밋하지 말 것**
- 저널/상태 DB, 계좌 잔고 문서 노출 없음. 팩 47.6MB로 정상

## 순서 (순서가 중요하다)

### 1. 자격증명 폐기 — 위험 순서대로

- [ ] **디스코드 봇 토큰 리셋** (최우선 — IP 제한 없음, 봇 전체 탈취 가능).
      developers.discord.com → Applications → Bot → Reset Token. 웹 콘솔 전용
      (Discord API에 토큰 리셋 엔드포인트가 없다).

      **임시 무력화는 API로 가능하다**: 이 토큰은 서버 1개(`QuantBox`,
      id `000000000000000000`)에만 들어가 있다. `DELETE /users/@me/guilds/{id}`로
      봇을 내보내면 소속 서버가 0이 되어 토큰이 사실상 무해해진다. 되돌리려면 OAuth
      초대 링크로 재초대하면 된다. 토큰 리셋이 가능하면 그쪽이 더 완전하다.
- [x] ~~**디스코드 웹훅 삭제**~~ — **2026-08-27 폐기 완료** (HTTP DELETE → 204, 재조회 404).
- [ ] **바이낸스 키 삭제 후 재발급** — 출금 차단 + IP 제한이라 긴급도는 위 둘보다 낮다.
      API 관리 웹 콘솔 전용(2FA 필요). **자기 키를 삭제하는 엔드포인트는 존재하지 않는다**
      (2026-08-27 확인: python-binance에 `get_account_api_permissions` 조회만 있고
      삭제/수정은 없다. `..._delete_broker_sub_account_api`는 브로커 마스터가 서브계정
      키를 지우는 것으로, 자기 키에는 적용되지 않는다).

      ⚠️ 다만 **IP 화이트리스트 조작 엔드포인트는 존재한다**
      (`POST/DELETE /sapi/v1/account/apiRestrictions/ipRestriction[/ipList]`).
      이론상 화이트리스트를 좁혀 무력화할 수 있으나, **IP 목록이 비었을 때 Binance가
      제한을 해제하는지 전면 차단하는지 확인되지 않았다.** 잘못 건드리면 지금 걸려 있는
      보호막이 풀린다. 자금이 있는 라이브 계좌에서 추측으로 실행하지 않는다 —
      웹 콘솔 삭제가 확실하고 더 빠르다.
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

### 3. 공개 직전 재확인 — 명령 두 개로 판정된다

```bash
uv run python scripts/check_secrets.py --rev public/clean   # 트리에 시크릿이 없는가
uv run python scripts/check_credentials.py                  # 유출된 값이 죽었는가
```

둘 다 exit 0이면 공개 전환의 기술적 조건은 충족이다.

`check_credentials.py`는 `.env`(현재 값)가 아니라 **유출 커밋에서 직접 뽑은 값**을
검사한다. 회전 후 `.env`에는 새 값이 들어가므로 현재 값을 보면 "옛 키가 죽었는지"를
영영 확인할 수 없기 때문이다. 값은 출력하지 않는다.

2026-08-27 기준 실행 결과:
```
  🔴 Discord 봇 토큰       아직 살아 있다 — 폐기 필요
  ✅ Discord 웹훅         폐기 확인
  🔴 바이낸스 API 키         아직 살아 있다 — 폐기 필요
```

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
