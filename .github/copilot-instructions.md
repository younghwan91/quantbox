# QuantBox Copilot Instructions (Strict Profitable Mode)

이 문서는 바이낸스 USDT-M 선물 **러버밴드 스캘핑 시스템** 'QuantBox'의 절대적인 개발 지침입니다.
우리의 목표는 **학술적 연구가 아닌, 실제 시장에서 수익을 내는 시스템**을 구축하는 것입니다.

Top Critical Rule: Always be absolutely honest and transparent about your answers, and never make up numbers or facts. If you are not 100% sure, say you don't know.

## 🚨 최우선 원칙 (Strict Rules)

1.  **Real Data Only (No Mock Data)**
    - 백테스트와 로직 검증에는 **반드시 실제 바이낸스 과거 데이터**만 사용한다.
    - `np.random` 등을 사용한 가짜(Synthetic) 가격 데이터 생성은 **절대 금지**한다.

2.  **Test-Code Parity (1:1 테스트 강제)**
    - 모든 기능 함수(Function)나 클래스 작성 시, 해당 기능을 검증하는 `pytest` 코드를 **반드시 함께 작성**한다.
    - 테스트 없는 코드는 '미완성'으로 간주한다.
    - `pytest-asyncio`를 사용하여 비동기 함수도 철저히 테스트한다.
    - 절대 많은 테스트가 중요한 것이 아니라 짧지만 효과적인 테스트가 중요하다.

3.  **Execution Reality (현실적 체결)**
    - 단순 시장가(Market) 주문보다는, 슬리피지를 최소화하는 지정가(Limit) 추격 로직을 우선 고려한다.
    - 백테스트 수익 계산 시 **거래 수수료(0.05%)** 뿐만 아니라 **펀딩비(Funding Fee)**와 **슬리피지**를 보수적으로 차감해야 한다.

4. **실거래와 백테스트의 일치**
    - 백테스트에서 사용한 로직과 동일한 로직이 실거래에서도 사용되어야 한다. 
    - 실거래와 백테스트 간의 괴리는 절대 발생해서는 안 된다.
    - 백테스트에서만 특별히 유리한 조건을 부여하는 코드는 허용되지 않는다.
    - 백테스팅은 창의적이면서, 단순히 시간이 긴 것이 중요한 것이 아니라 효율적인 탐색이 중요하다.
    - 모델을 고도화시켜 알파를 찾기를 두려워하지 말라. 하지만 하이퍼파라미터를 과최적화해선 안된다.

5. **모든 권한을 에이전트에게 위임**
    - 에이전트가 스스로 학습하고 개선할 수 있도록, 가능한 한 많은 권한과 자율성을 부여한다.
    - File write operations detected that cannot be auto approved: /tmp/* 라는 말이 절대 나오지 않도록 한다.

6. **Code First, Talk Later (코드 확인 후 발언)**
    - 전략 파라미터, 설정값, 현재 상태에 대해 답변할 때 **반드시 코드/설정/로그를 먼저 읽고** 답변한다.
    - 기억이나 추측으로 수치를 말하는 것은 **절대 금지**한다. 틀린 숫자는 잘못된 투자 판단을 유발한다.
    - 확실하지 않으면 "코드를 확인하겠습니다"라고 말하고, 확인 후 답변한다.
    
## 📝 문서 자동 관리 (Documentation Auto-Sync)

코드 변경 시 아래 항목을 **사용자가 요청하지 않아도 알아서** 최신 상태로 유지한다:

1. **README.md**: 새 기능 추가/삭제 시 주요 기능 섹션, 프로젝트 구조, 테스트 수를 업데이트한다.
2. **MANUAL.md**: 환경변수 추가, 전략 파라미터 변경, 새 모듈 추가 시 해당 섹션을 업데이트한다. 버전 번호와 최종 업데이트 날짜도 갱신한다.
3. **.env.example**: 새 환경변수 추가 시 **반드시** `.env.example`에도 주석과 함께 추가한다. 동시에 .env 파일에도 수정을 반영한다.

> ⚠️ **`.env` 우선순위 주의**: `pydantic-settings`는 `.env` 값이 `settings.py` 기본값보다 **항상 우선**한다. 파라미터를 변경할 때 `settings.py`만 수정하고 `.env`를 안 바꾸면 **실제 적용값은 변하지 않는다.** 반드시 `settings.py`, `.env`, `.env.example` **3곳을 동시에** 수정해야 한다.
4. **copilot-instructions.md**: 개발 원칙이나 워크플로우가 변경될 때 이 문서도 갱신한다.
5. **agents.md**: 아키텍처, 모듈 구조, 안전장치, 파라미터, 트레이딩 흐름 등이 변경될 때 `.github/agents.md`도 **반드시** 동기화한다. `copilot-instructions.md`와 `agents.md`는 항상 쌍으로 최신 상태를 유지해야 한다.

**판단 기준:**
- feature 브랜치 머지 시: README.md 기능 목록 + MANUAL.md 해당 섹션 확인
- 환경변수 추가 시: .env.example + MANUAL.md 환경변수 섹션
- 테스트 파일 추가/삭제 시: README.md 테스트 수 갱신
- 프로젝트 구조 변경 시 (파일 추가/삭제/이동): README.md 프로젝트 구조 트리 갱신

## 🔗 커밋 추적 (Commit Traceability)

모든 출력물(로그, 백테스트 결과, 문서)은 **어떤 커밋에서 생성되었는지** 추적 가능해야 한다.

1. **로그 파일**: `scalper.py`의 `_setup_logging()`이 모든 새 로그 파일 상단에 4줄 커밋 헤더를 자동 기록한다:
   ```
   # Commit: abc1234 (2026-02-19 00:00 +0900)
   # Message: feat: some feature description
   # Full SHA: abc1234567890abcdef...
   #
   ```
2. **백테스트 결과**: `docs/backtest_results/` 폴더에 `{commit_short}_{date}_{description}.md` 형식으로 저장한다. 새 백테스트 실행 시 반드시 해당 커밋의 결과 스냅샷을 생성하고, `docs/backtest_results/README.md` 인덱스를 갱신한다.
3. **Discord 시작 알림**: 봇 시작 시 Discord embed에 현재 커밋 SHA를 포함한다.
4. **스크립트 출력**: 백테스트 스크립트 실행 시 출력물 상단에 커밋 정보를 포함하는 것을 지향한다.

## 코딩 컨벤션 및 타입

Type Hinting: typing.List, typing.Dict 대신 Python 3.9+ 표준인 list[], dict[], tuple[] 사용을 지향한다.
Docstring: Google Style Docstring을 사용하여 입출력 데이터의 형상을 명시한다.
Numba: 루프가 10,000회 이상 반복되는 연산(백테스트)에는 반드시 @numba.jit(nopython=True)를 적용한다.

## 환경 설정

.env 파일에 BINANCE_API_KEY, BINANCE_API_SECRET이 없으면 실행되지 않도록 pydantic-settings (`quantbox.core.settings`)를 활용하여 검증한다.
로그 레벨은 개발 시 DEBUG, 운영 시 INFO로 설정하되, 주문 체결 로그는 별도 파일로 영구 보존한다.

## Git Flow 전략 (필수 준수)

- **main**: 프로덕션 릴리스 브랜치. 태그(v1.x.x)만 존재. 직접 커밋 금지.
- **develop**: 개발 통합 브랜치. 모든 feature가 머지되는 곳.
- **feature/xxx**: 새 기능 개발. `develop`에서 분기 → 완료 후 `develop`으로 머지.
- **fix/xxx**: 버그 수정. `develop`에서 분기 → 완료 후 `develop`으로 머지.
- **hotfix/xxx**: 긴급 수정. `main`에서 분기 → `main`과 `develop` 둘 다 머지.

**규칙:**
1. feature/fix 브랜치 작업 완료 시 **반드시 develop에 머지**한 뒤 push한다. 브랜치만 push하고 develop을 안 건드리면 안 된다.
2. develop에 직접 커밋하지 않고, 항상 feature/fix 브랜치를 경유한다.
3. main에는 릴리스 준비가 완료된 develop만 머지한다.
4. **백테스트/리서치 브랜치** (`docs/backtest_results/`에 리포트를 생성하는 경우): **`--no-ff`** 머지. 리포트 안의 커밋 해시가 머지 후에도 유효해야 하므로 원본 커밋을 보존한다.
5. **그 외 모든 브랜치** (fix, chore, config 등): **squash merge** (`git merge --squash`). 히스토리를 깔끔하게 유지한다.
6. develop → main 머지 시에는 **`--no-ff`** 옵션으로 머지 커밋을 남긴다.

**Agent Note:** 사용자는 엄격하며 공격적인 퀀트 트레이더입니다. 코드가 예쁘게 보이는 것보다, **돈을 잃지 않는 안전장치(Safety check)**가 구현되었는지를 가장 중요하게 생각합니다.

## 🩸 실수로부터의 교훈 (Lessons from Live Incidents)

> 아래 원칙들은 실제 자금 손실 또는 수익 기회 상실을 초래한 실전 버그에서 비롯되었다.
> **절대 반복해서는 안 된다.**

### Incident #1: Partial Close Leverage Crash (2026-02-21, v12.4.6)

**사건:** `_place_order()`가 `reduce_only=True`인 주문에도 `futures_change_leverage()`를 호출했다.
Cross 4x 포지션에서 leverage를 기본값 1로 변경 시도 → Binance APIError `-2028: insufficient margin balance` → **TP1 partial close 실패**. 가격이 TP1을 크게 넘어 수익 구간까지 갔다가 되돌아왔으나, 50% 이익 확정 기회를 완전히 상실.

**근본 원인:**
1. `reduce_only` 주문은 마진 요구사항을 변경하지 않으므로 leverage 변경이 불필요하다는 사실을 코드에 반영하지 않았다.
2. PBT 모드에서 서버 TP를 의도적으로 건너뛰는 로직이 bracket repair에 반영되지 않아, 매 분마다 `TP=MISSING` 로그를 무한 생성하여 진짜 문제를 노이즈 속에 묻어버렸다.
3. 해당 코드 경로(open position + reduce_only close)에 대한 테스트가 없었다.

**영구 원칙:**

1. **Reduce-Only = No Side Effects (감산 주문은 부작용 금지)**
   - `reduce_only=True` 주문은 **기존 포지션을 줄이기만** 한다. leverage 변경, margin mode 변경 등 포지션 구조를 바꾸는 API 호출을 절대 포함하지 않는다.

2. **Money-Path Full Coverage (돈이 오가는 경로는 100% 테스트)**
   - `_place_order()`, `_partial_close_position()`, `_close_position()` 등 실제 주문을 발생시키는 모든 함수는 반드시 `reduce_only=True`와 `reduce_only=False` 양쪽 경로를 테스트해야 한다.
   - Mock 레벨이 아닌, **실제 API 에러 코드(-2028, -4028, -2022 등)에 대한 대응 테스트**를 포함한다.

3. **Silent Failure = Critical Bug (조용한 실패는 치명적 버그)**
   - 안전장치(TP1 partial close, SL, bracket repair)가 실패했을 때 **ERROR 로그 + Discord 알림**을 즉시 발송해야 한다. 로그에 묻히는 실패는 없는 것과 같다.
   - bracket repair가 같은 문제를 반복 로깅할 때, 횟수 제한 없이 매 분 반복하는 것은 노이즈이며 실제 장애를 가린다. **반복 경고는 최대 3회 후 요약으로 전환**한다.

4. **Position-Mutating API Audit (포지션 변경 API 감사)**
   - `futures_change_leverage()`, `futures_change_margin_type()` 등 포지션 구조를 변경하는 API는 **오직 신규 진입(entry) 시에만 호출**한다. 포지션이 열려 있는 상태에서 호출하면 예측 불가능한 거부가 발생할 수 있다.