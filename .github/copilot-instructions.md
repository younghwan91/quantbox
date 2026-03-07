# QuantBox Copilot Instructions

바이낸스 USDT-M 선물 VR 스퀴즈 시스템 'QuantBox'의 개발 지침.
목표: **실제 시장에서 수익을 내는 시스템**. 학술적 연구 아님.

**Top Rule: 수치/사실을 절대 추측하지 말라. 모르면 모른다고 하라.**

## 핵심 원칙

1. **No Mock Data** — 백테스트에 실제 바이낸스 데이터만 사용. `np.random` 합성 데이터 금지.

2. **Backtest-Live Parity** — 백테스트 로직 = 실거래 로직. 백테스트에만 유리한 조건 금지. **룩어헤드/미래 데이터 누수 절대 금지.**
   - **파라미터는 반드시 `config.py`를 직접 읽어서 사용.** 하드코딩 또는 기억에서 쓰지 말 것.
   - **체크리스트** — 백테스트 작성 전 반드시 `config.py`에서 확인:
     - exit 로직: flat TP vs trailing stop? (`d_tp`, `d_trail_activate`, `d_trail_distance`)
     - L/N: `base_leverage / max_concurrent` (VR=30/10=3.0, MA=21/10=2.1) — 하드코딩 금지
     - SL 값, 슬리피지/수수료 (round-trip = 0.0008)
   - **위반 선례 (2026-03-07):** `backtest_v14_3_0.py`가 `tp=0.05`(flat) + `VR_LN=2.5`, `MA_LN=1.2` 사용 →
     실제 프로덕션은 `tp=0.12` + trailing stop + `VR_LN=3.0`, `MA_LN=2.1`. N/PF/CAGR 전부 틀린 수치 보고.

3. **Test-Code Parity** — 모든 기능에 `pytest` 테스트 필수. 짧고 효과적인 테스트 우선. 테스트 없는 코드 = 미완성.

4. **Execution Reality** — 수수료(0.05%) + 펀딩비 + 슬리피지 보수적 차감. 지정가 추격 우선.

5. **Code First** — 파라미터, 설정값 답변 시 반드시 코드/설정을 먼저 읽고 답변. 추측 금지.

6. **Agent Autonomy** — 에이전트에게 최대한 자율성 부여. `/tmp/*` 쓰기 차단 경고가 나오지 않도록 한다.

## 문서 자동 관리

코드 변경 시 **요청 없이** 동기화:
- **README.md**: 기능 목록, 프로젝트 구조, 테스트 수
- **MANUAL.md**: 환경변수·실행 커맨드 변경 시만 (인프라 변경에만 해당; 전략 파라미터는 `config.py`가 source of truth)
- **.env.example + .env**: 새 환경변수 추가 시 3곳(`settings.py`, `.env`, `.env.example`) 동시 수정
- **agents.md**: 아키텍처/모듈/안전장치/트레이딩 흐름 변경 시 동기화

> ⚠️ `pydantic-settings`는 `.env` 값이 `settings.py` 기본값보다 항상 우선. `settings.py`만 바꾸고 `.env` 안 바꾸면 **적용 안 됨**.

## 문서 배치 규칙 (파일이 떠돌지 않게)

```
프로젝트 루트
├── README.md              # 프로젝트 개요, 퀵스타트 (루트에만)
├── MANUAL.md              # 운영 매뉴얼 (루트에만)
├── .github/
│   ├── copilot-instructions.md   # 에이전트 지침 (repo-wide, 자동 로드)
│   ├── agents.md                 # 아키텍처/모듈 설명
│   └── instructions/             # Path-specific instructions (Copilot CLI 공식, 자동 로드)
│       ├── backtesting.instructions.md      # research/**/*.py 에 자동 적용
│       ├── live_trading.instructions.md     # scalper.py, execution/**/*.py 에 자동 적용
│       ├── debugging.instructions.md        # quantbox/**/*.py, tests/**/*.py 에 자동 적용
│       └── position_sizing.instructions.md  # execution/**, strategy/**, *config* 에 자동 적용
└── docs/
    └── backtest_results/
        ├── README.md             # 인덱스 (이것만 루트에)
        ├── 202602/               # YYYYMM 폴더 안에만
        └── 202603/               # 신규 월 생기면 폴더 먼저
```

**배치 원칙:**
- 백테스트 리포트는 **반드시** `docs/backtest_results/YYYYMM/` 안에. 루트나 `docs/` 직하위 금지.
- 월이 바뀌면 폴더 **먼저** 만들고 파일 생성.
- `docs/backtest_results/` 루트에는 `README.md` 하나만 허용.
- `.github/`에는 에이전트·개발 지침만. 전략 결과 문서 금지.

**네이밍 규칙** (백테스트 리포트):
```
docs/backtest_results/YYYYMM/{YYYYMMDD}_{commit_short}_{description}.md
예: docs/backtest_results/202603/20260304_d4e70d8_r45_param_sweep.md
```

## 커밋 추적

모든 출력물(로그, 백테스트, Discord)에 커밋 SHA 포함. 이미 `_setup_logging()`에 구현됨.
백테스트 결과는 위 **문서 배치 규칙**의 네이밍 형식으로 YYYYMM 폴더에 저장.

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

### #3: Dynamic Capital 50% Cap (2026-03-05)

`VolStraddleConfig.dynamic_capital_enabled=True` → `balance / max(1, n_active+1)`, 50% cap → 첫 슬롯이 잔고의 50%($991) 독점 → 나머지 9개 슬롯 전부 `-2019: Margin is insufficient`.

**테스트 미검출 원인: 포지션 0개인 상태(`n_active=0`)에서 단일 진입만 검증했기 때문에, 2번째 슬롯 진입 시 마진 부족이 드러나지 않았다.**

**규칙:**
- **슬롯 sizing 테스트는 반드시 `n_active > 0` 상태(선행 포지션 보유 중)를 포함.**
- 포지션 sizing 로직 변경 시 `max_concurrent` 개 슬롯 전체를 순서대로 열었을 때 총 margin이 잔고를 초과하지 않는지 검증.
- **다른 전략에서 이식한 공식(RubberBand 2-slot → VolStraddle 10-slot)은 슬롯 수 역할이 바뀌므로 반드시 재검증.**

### #4: Algo Order FINISHED Status (2026-03-06)

`_check_order_filled()`가 `algoStatus == 'TRIGGERED'`만 검사 → 바이낸스 API는 조건부 주문이 트리거+체결 완료 후 `'FINISHED'`를 반환 → 모든 서버 SL/TP 체결을 감지 못함 → `_reconcile_positions()` 고스트 감지에 의존 → 대응 알고 주문(반대편 SL/TP) 미취소.

**바이낸스 조건부 주문 상태 전이:** `WORKING` → `TRIGGERED`(순간) → `FINISHED`(최종)

**규칙:**
- **바이낸스 API 상태값은 반드시 실제 API 응답으로 검증.** 문서/추측 금지.
- **`_check_order_filled()`는 `TRIGGERED`와 `FINISHED` 모두 체결로 처리.**
- **고스트 감지 시** `_reconcile_positions()`에서 **반드시 반대편 알고 주문도 취소**. 고아 주문 방치 시 다음 포지션에 영향.
- **서버 체결 감지 로그**는 `DEBUG` 아닌 `INFO` 레벨. 프로덕션에서 진단 가능해야 함.