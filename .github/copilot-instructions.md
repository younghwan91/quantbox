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

코드 변경 시 **요청 없이** 동기화하고, `.instructions.md` 파일이 300줄 초과 시 오래됐거나 중복된 내용을 **먼저 줄이고 추가**한다:
- **README.md**: 기능 목록, 프로젝트 구조, 테스트 수
- **MANUAL.md**: 환경변수·실행 커맨드 변경 시만 (인프라 변경에만 해당; 전략 파라미터는 `config.py`가 source of truth)
- **.env.example + .env**: 새 환경변수 추가 시 3곳(`settings.py`, `.env`, `.env.example`) 동시 수정
- **agents.md**: 아키텍처/모듈/안전장치/트레이딩 흐름 변경 시 동기화

### Skills 자율 업데이트 (요청 불필요)

목표: 백테스팅·디버깅·세션 관리가 반복할수록 더 안정적으로 실행되도록 skills를 점진적으로 다듬는다.

| 상황 | 업데이트 대상 |
|------|-------------|
| 버그 수정 / 인시던트 발생 | `debugging.instructions.md` + `copilot-instructions.md` 인시던트 테이블 |
| 백테스트에서 새 주의사항 발견 | `backtesting.instructions.md` |
| 세션 시작/종료 절차 변경 | `live_trading.instructions.md` |
| 포지션 사이징 공식 변경 | `position_sizing.instructions.md` |

**업데이트 기준:** 같은 실수를 두 번 하지 않을 만한 내용. 일회성 특이사항은 추가하지 않는다.

**자기학습 루프:** 주요 작업(백테스트, 디버깅, 세션 재시작) 완료 후 스스로 묻는다:
- 이번에 skill이 안내하지 못해서 직접 알아낸 게 있는가?
- 다음 번에 이 작업을 더 빠르고 정확하게 할 수 있게 할 내용인가?

→ **Yes이면 즉시 해당 skill 업데이트.** 오늘 r48 DD halt 분석처럼 — 버그가 아닌 분석에서 발견한 것도 skill로 남긴다.

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
| `research/xxx` | 리서치/백테스트 | **`--squash`** — 탐색 커밋 정리 후 단일 커밋으로. 머지 후 삭제 가능 (이력은 squash 커밋 + docs/backtest_results/ + `(ref: rNN)` 로 보존) |
| `release/xxx` | 릴리즈 준비 | `main` + `develop` 둘 다 **`--no-ff`** |
| `fix/xxx` | 버그 수정 | **`--squash`** |
| `hotfix/xxx` | 긴급 (`main`에서 분기) | `main` + `develop` 둘 다 **`--no-ff`** |

**반드시 develop에 머지 후 push.** 브랜치만 push하고 develop 안 건드리면 안 됨.

**main에 머지할 때는 반드시 버전 태그 생성 및 push.** `git tag -a vX.Y.Z -m "..."`

**리서치 → 전략 추적 관례:** 리서치 결과를 구현하는 feature/fix 커밋에 `(ref: rNN)` 명시.
```
# 예시
feat(strategy): D_mom8 pullback entry (ref: r55, pb=0.5% win=5)
fix(config): SL 파라미터 조정 (ref: r48 DD halt 분석)
```
반대 방향(어떤 버전에 반영됐는지)은 각 리서치 보고서 `.md`에 `**구현:** feature/xxx → vX.Y.Z` 한 줄 추가.

## 리서치·수익률 보고 규칙

> 상세 절차: `.github/instructions/backtesting.instructions.md` (research/**/*.py 자동 로드)

**핵심 3가지 — 위반 시 보고서 전체 무효:**
1. **Sim 재구현 금지** — `quantbox/backtest/trade_sim.py`의 `collect_trades()` + `portfolio_sim()` import 필수
2. **파라미터 하드코딩 금지** — 반드시 `VolStraddleConfig()` 읽어서 사용 (`VR_LN = cfg.base_leverage / cfg.max_concurrent`)
3. **compound 모델 명시** — `compute_metrics()` 출력(1x)을 production 수익률(L/N=3.0)로 제시 금지

## 실전 인시던트 규칙 (Incident-Driven Rules)

> 상세 진단: `.github/instructions/debugging.instructions.md` (quantbox/**/*.py 자동 로드)

실제 자금 손실에서 도출된 규칙. **위반 절대 금지.**

| # | 날짜 | 핵심 규칙 |
|---|------|-----------|
| #1 | 2026-02-21 | `futures_change_leverage()` 는 **신규 진입 시에만**. close/reduce_only 경로에서 호출 금지 |
| #2 | 2026-02-26 | 교차 TF 참조는 **완성된 바만** (`j = (i // ratio) - 1`). WR이 라이브의 2배면 lookahead 의심 |
| #3 | 2026-03-05 | slot sizing 테스트는 반드시 **`n_active > 0` 상태** 포함. 전체 10슬롯 마진 합계 검증 |
| #4 | 2026-03-06 | `_check_order_filled()` — `TRIGGERED` **+ `FINISHED`** 둘 다 체결 처리. ghost 발견 시 반대편 주문도 취소 |
| #5 | 2026-03-08 | Hedge Mode close는 `reduce_only=False` → leverage guard는 **`if not reduce_only and leverage > 1:`** |