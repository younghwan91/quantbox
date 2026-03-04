# QuantBox Copilot Instructions

바이낸스 USDT-M 선물 VR 스퀴즈 시스템 'QuantBox'의 개발 지침.
목표: **실제 시장에서 수익을 내는 시스템**. 학술적 연구 아님.

**Top Rule: 수치/사실을 절대 추측하지 말라. 모르면 모른다고 하라.**

## 핵심 원칙

1. **No Mock Data** — 백테스트에 실제 바이낸스 데이터만 사용. `np.random` 합성 데이터 금지.

2. **Backtest-Live Parity** — 백테스트 로직 = 실거래 로직. 백테스트에만 유리한 조건 금지. **룩어헤드/미래 데이터 누수 절대 금지.**

3. **Test-Code Parity** — 모든 기능에 `pytest` 테스트 필수. 짧고 효과적인 테스트 우선. 테스트 없는 코드 = 미완성.

4. **Execution Reality** — 수수료(0.05%) + 펀딩비 + 슬리피지 보수적 차감. 지정가 추격 우선.

5. **Code First** — 파라미터, 설정값 답변 시 반드시 코드/설정을 먼저 읽고 답변. 추측 금지.

6. **Agent Autonomy** — 에이전트에게 최대한 자율성 부여. `/tmp/*` 쓰기 차단 경고가 나오지 않도록 한다.

## 문서 자동 관리

코드 변경 시 **요청 없이** 동기화:
- **README.md**: 기능 목록, 프로젝트 구조, 테스트 수
- **MANUAL.md**: 환경변수, 전략 파라미터, 모듈 추가/변경
- **.env.example + .env**: 새 환경변수 추가 시 3곳(`settings.py`, `.env`, `.env.example`) 동시 수정
- **agents.md**: 아키텍처/모듈/안전장치 변경 시 동기화

> ⚠️ `pydantic-settings`는 `.env` 값이 `settings.py` 기본값보다 항상 우선. `settings.py`만 바꾸고 `.env` 안 바꾸면 **적용 안 됨**.

## 커밋 추적

모든 출력물(로그, 백테스트, Discord)에 커밋 SHA 포함. 이미 `_setup_logging()`에 구현됨.
백테스트 결과: `docs/backtest_results/{commit}_{date}_{desc}.md` 형식.

## 코딩 컨벤션

- Type Hints: `list[]`, `dict[]`, `tuple[]` (Python 3.9+)
- Docstring: Google Style, 입출력 형상 명시
- Numba: 10,000회+ 반복 연산에 `@numba.jit(nopython=True)` 필수
- 환경: `BINANCE_API_KEY/SECRET` 미설정 시 실행 거부 (`pydantic-settings`)

## Git Flow

| 브랜치 | 용도 | 머지 방식 |
|--------|------|-----------|
| `main` | 프로덕션. 직접 커밋 금지 | `release` → `main` `--no-ff` + 버전 태그 |
| `develop` | 통합. 직접 커밋 금지 | feature/fix/release에서만 머지 |
| `feature/xxx` | 새 기능 (코드) | **`--squash`** — 커밋 정리 후 단일 커밋으로 |
| `research/xxx` | 리서치/백테스트 | **`--no-ff`** — 백테스트 리포트·탐색 이력 보존 |
| `release/xxx` | 릴리즈 준비 | `main` + `develop` 둘 다 **`--no-ff`** |
| `fix/xxx` | 버그 수정 | **`--squash`** |
| `hotfix/xxx` | 긴급 (`main`에서 분기) | `main` + `develop` 둘 다 **`--no-ff`** |

**반드시 develop에 머지 후 push.** 브랜치만 push하고 develop 안 건드리면 안 됨.

**main에 머지할 때는 반드시 버전 태그 생성 및 push.** `git tag -a vX.Y.Z -m "..."`

## 리서치 베이스라인 규칙

1. **Sim 함수 재구현 금지.** baseline은 반드시 `scaling_r38.py`의 `collect_trades()` + `portfolio_sim()`을 import. 새로 짜면 momentum=0 처리, squeeze 카운터 리셋 등 미묘한 차이로 수치가 틀어진다.
2. **Config는 `CFG_V13` import.** 하드코딩 금지.
3. **Phase 1에서 N, PF가 `docs/backtest_results/` 내 최신 정식 리포트와 ±1% 이내인지 확인.** 불일치 시 즉시 중단.
4. **슬롯 수 명시.** 15 slots ≠ 10 slots. Production은 10 slots.
5. **재구현 불가피 시** static params로 baseline 재현 검증을 **먼저** 실행.
6. **베이스라인 = 배포 수치. 예외 없음.** 리서치에서 출력한 baseline이 배포 리포트와 다르면 리서치 전체가 무효.

## 수익률 보고 규칙 (Performance Reporting Rules)

> 버전·보고서 간 수익률 불일치의 반복된 원인. 수치 보고 시 반드시 준수.

### 레버리지 모델 명시 필수

`scaling_r38.py`의 `compute_metrics()`는 **1x compound** (`equity *= (1+p)`)만 계산한다. Production은 `equity *= (1 + p × L/N)`이다. **이 둘을 절대 혼동하지 말 것.**

- 수익률 보고 시 **반드시 어떤 compound 모델인지 명시**: `1x` vs `L/N=2.5` vs `L/N=1.2`
- `compute_metrics()` 출력을 production 수익률로 제시하면 **거짓말**이 된다
- 결합 포트폴리오는 per-trade leverage (VR→L/N=2.5, MA→L/N=1.2) 적용

### 데이터 기간에 따른 compound 차이

동일 PF/EV라도 compound 기간이 다르면 연간수익률이 **극적으로** 달라진다:
- VR PF≈1.31: 4.7yr → +338%, 6yr → +687% (2배 차이)
- **버전 간 수익률 비교 시 반드시 데이터 기간을 같이 명시**
- PF와 per-trade EV가 레버리지·기간 무관한 **불변 엣지 지표**

### 체크리스트 (보고서 작성 전 확인)

1. ☐ compound 모델 명시? (1x / L/N=? / per-trade leverage)
2. ☐ 데이터 기간 명시? (몇 년, 시작~끝 날짜)
3. ☐ 이전 버전 수치와 **같은 조건**으로 비교?
4. ☐ `compute_metrics()` 출력을 그대로 production 수치로 쓰지 않았는가?
5. ☐ 이전 보고서의 **데이터 기간·파라미터(TP/SL)·리서치 스크립트**가 현재와 다르면, 다른 스크립트 출력을 같은 전략이라고 비교하지 않았는가?

## 실전 교훈 (Incident-Driven Rules)

> 실제 자금 손실에서 비롯된 규칙. 절대 반복 금지.

### #1: Reduce-Only Leverage Crash (2026-02-21)

`reduce_only=True` 주문에서 `futures_change_leverage()` 호출 → API -2028 → TP1 실패 → 수익 확정 기회 상실.

**규칙:**
- **reduce_only 주문은 leverage/margin 변경 금지.** 감산만 수행.
- **돈이 오가는 함수** (`_place_order`, `_close_position` 등)는 `reduce_only` 양쪽 경로 모두 테스트.
- **안전장치 실패 시** ERROR 로그 + Discord 즉시 알림. 반복 경고는 3회 후 요약 전환.
- **포지션 변경 API** (`futures_change_leverage` 등)는 **신규 진입 시에만** 호출.

### #2: 4h MTF Lookahead Bias (2026-02-26)

`i // ratio`가 developing bar의 final close 참조 → 백테스트 WR 83% (실제 36%).

**규칙:**
- **교차 타임프레임 참조는 완성된 바만 사용.** `j = (i // ratio) - 1`.
- 새 상위 TF 참조 추가 시: "이 바가 진행 중일 때 이 값을 알 수 있는가?" 반드시 검증.
- **백테스트 WR이 라이브의 2배 이상이면 lookahead 즉시 의심.**