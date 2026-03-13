# QuantBox Agent Instructions

> 코드 사실(Facts) 레퍼런스. 원칙은 `copilot-instructions.md` 참조.  
> **현재 프로덕션:** VolStraddle v14.24.0 + 5 신뢰성 모듈 (--strategy vol_straddle)

---

## 1. 시스템 개요

| 구성 요소 | 역할 |
|-----------|------|
| **RubberBandScalper** (`scalper.py`) | 실거래 오케스트레이터. 15분 사이클로 신호 → 진입 → 모니터링 → 청산. `--strategy` CLI로 전략 선택 |
| **RubberBandStrategy** (`rubberband.py`) | 레거시 평균회귀 (비프로덕션) |
| **ElliottStrategy** (`elliott.py`) | 레거시 Elliott Wave (비프로덕션) |
| **VolStraddleStrategy** (`vol_straddle/`) | VR 압축 감지 + D_mom8 방향 진입 + **S_d8 스트래들** (양방향 동시). 15m/1h/4h 멀티TF. **Hedge Mode** (`positionSide` LONG/SHORT). 복합 키: `SYM:LONG`/`SYM:SHORT`. **현재 프로덕션** |
| **AlphaEngine** (`alpha_engine.py`) | 멀티팩터 신호 엔진 |
| **VolatilitySelector** (`vol_selector.py`) | 30일 실현변동성 기준 Top-N 심볼 동적 선택 |
| **LimitOrderChaser** (`limit_chaser.py`) | 메이커 지정가 추격 주문. 타임아웃 시 시장가 폴백 |
| **State Persistence** (`state_persistence.py`) | JSON 기반 상태 영속화. 원자적 쓰기. 크래시 복구 지원 |
| **Drawdown / Daily Loss Breaker** | 세션 HWM 10% / 당일 5% 초과 시 자동 포지션 청산 + 일시정지 |
| **RateLimiter + CircuitBreaker** (`core/rate_limiter.py`) | API 호출 스로틀링 + 연속 에러 시 자동 차단 (v14.22.0) |
| **HeartbeatMonitor** (`core/heartbeat.py`) | 메인 루프 정지 감지, Discord 봇 재연결 (v14.22.0) |
| **InFlightTracker** (`execution/inflight.py`) | 미체결 주문 실시간 추적, 크래시 복구 지원 (v14.22.0) |
| **ExposureGuard** (`execution/exposure.py`) | 세션 노출도 상한선 (`max_exposure_multiplier=35.0`), 초과 시 신규 진입 차단 (v14.23.0) |
| **ReconJournal** (`execution/recon_journal.py`) | SQLite 감사 로그. 10종 이벤트 (ghost, bracket, exposure 등) (v14.23.0) |
| **TradingStrategy** (`strategy/protocol.py`) | 14멤버 전략 Protocol 인터페이스. `@runtime_checkable` (v14.24.0) |
| **Discord Bot/Notifier** (`bot.py`, `discord.py`) | 원격 제어(!status, !close all 등) 및 실시간 알림 |

모든 파라미터는 `quantbox/strategy/vol_straddle/config.py`의 `VolStraddleConfig()`에서 직접 읽어라. 하드코딩 금지.

---

## 2. 라이브 트레이딩 전체 흐름 (VolStraddleStrategy._cycle)

```
┌─────────────────────────────────────────────────────────────────┐
│  0. balance = get_balance()                                     │
│  1. _check_risk_limits(balance)                                 │
│     → HWM 대비 DD ≥ 10% → _drawdown_halt (전체 청산)           │
│  2. If drawdown_halt: close ALL positions → return              │
│  3. 기존 포지션 관리:                                            │
│     ∀ active position (SYM:LONG / SYM:SHORT):                  │
│       → update_position() → SL / hard TP / trailing / MaxHold  │
│       → FINISHED/TRIGGERED 알고 주문 감지 → 반대편 주문 취소    │
│  4. 신규 진입 스캔:                                              │
│     ∀ TF in [4h, 1h, 15m] (높은 TF 우선):                     │
│       ∀ symbol:                                                 │
│         → klines 다운로드 → compute_vr() → gen_vr_signal()     │
│         → D_mom8: VR<0.50 → 방향 결정 → open LONG or SHORT     │
│         → S_d8: VR<0.55 × 4바 → LONG+SHORT 동시 진입 (Hedge)  │
│         → F5 TBR Gate: tbr_zscore > 0.5 → D_mom8 차단         │
│         → MA Cluster: SMA20/50/100/200 수렴 ≤3%, 8바 → 브레이크아웃 |
│     → max_concurrent=10 슬롯 공유, dir_diversity max_same_direction=2 │
│  5. _save_state() — 원자적 JSON 저장                            │
│  6. 매 사이클 _log_summary() + Discord 알림                     │
└─────────────────────────────────────────────────────────────────┘
```

비용 모델: `position_sizing.instructions.md` Section 6 참조.

---

## 3. 핵심 모듈 맵

### 3.1 `quantbox/core/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `settings.py` | `QuantBoxSettings` | pydantic-settings 기반 환경변수 검증. `get_settings()` 싱글턴 |
| `config.py` | `Config`, `StrategyConfig`, `RiskConfig` | YAML/dict 기반 설정. 백테스트에서 주로 사용 |
| `events.py` | `SignalEvent`, `OrderEvent`, `FillEvent`, `EventQueue` | 이벤트 드리븐 아키텍처 기반 |
| `limit_chaser.py` | `LimitOrderChaser`, `ChaseResult` | 메이커 주문 추격 + 시장가 폴백 |
| `state_persistence.py` | `save_state`, `load_state`, `clear_state` | JSON 원자적 쓰기/읽기. 포지션 직렬화 |
| `bot_utils.py` | `get_git_commit()`, `setup_logging()`, `GIT_COMMIT` | 봇 시작 유틸리티. 커밋 SHA 추적, 로그 핸들러 설정. `GIT_COMMIT`은 import 시 1회 평가 |
| `scalper_state.py` | `save_scalper_state()`, `load_scalper_state()`, `rotate_state_backups()`, `STATE_FILE`, `PAPER_STATE_FILE` | 라이브 봇 JSON 상태 영속화. tmpfile→os.replace 원자적 쓰기. 크래시 복구 지원 |
| `loop_mixin.py` | `LoopMixin` | 타이밍/디스플레이. `_wait_for_bar()`, `_next_bar_time()`, `_log_summary()`, `_log_vr_distribution()`, `_print_startup_banner()` |
| `rate_limiter.py` | `RateLimiter`, `CircuitBreaker` | API 호출 스로틀링 + 연속 에러 차단 (v14.22.0) |
| `heartbeat.py` | `HeartbeatMonitor` | 메인 루프 정지 감지, Discord 봇 재연결 (v14.22.0) |

### 3.2 `quantbox/strategy/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `rubberband.py` | `RubberBandStrategy` | 레거시 평균회귀 (비프로덕션) |
| `elliott.py` | `ElliottStrategy` | 레거시 Elliott Wave (비프로덕션) |
| `vol_straddle/` | `VolStraddleConfig`, `VolStraddleStrategy`, `compute_vr`, `compute_momentum`, `gen_vr_signal`, `gen_vr_dur_signal`, `make_direction_momentum`, `resample_ohlcv` | VR 압축 + D_mom8 방향 + **S_d8 스트래들** (VR<0.55, 8바 스퀴즈). Hedge Mode 복합 키 (`SYM:LONG`/`SYM:SHORT`). 완성된 바만 참조 (zero lookahead) |
| `alpha_engine.py` | `AlphaEngine`, `AlphaConfig` | 멀티팩터 신호 |
| `vol_selector.py` | `VolatilitySelector` | 실현변동성 기반 심볼 선택 |
| `protocol.py` | `TradingStrategy` | 14멤버 Protocol 인터페이스. `@runtime_checkable`. 전략 확장 표준 (v14.24.0) |

### 3.3 `quantbox/backtest/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `vectorized_backtest.py` | `VectorizedBacktest`, `BacktestConfig`, `BacktestResult`, `_backtest_core_partial` | Numba 가속 백테스트. 단일/포트폴리오 지원. Delta-based 비용 모델 (부분 청산 대응) |
| `trade_sim.py` | `collect_trades`, `portfolio_sim`, `compute_metrics`, `sim_directional_trailing`, `sim_straddle_trailing` | 슬롯 경쟁 시뮬레이터. `sim_directional_trailing()` / `sim_straddle_trailing()` — v14.24.0에서 research/r43에서 이전됨 |

### 3.4 `quantbox/execution/`

| 파일 | 핵심 클래스 | 역할 |
|------|------------|------|
| `brackets.py` | `BracketMixin` | SL/TP 알고주문 배치/감지/repair |
| `positions.py` | `PositionMixin` | 포지션 진입/청산/사이징 |
| `reconciliation.py` | `ReconciliationMixin` | 거래소 상태 동기화, ghost 감지 |
| `data_mixin.py` | `DataMixin` | `get_trading_symbols()`, `get_symbol_info()`, `download_klines()`. 마지막 미완성 캔들 자동 제거 (backtest-live parity) |
| `inflight.py` | `InFlightTracker` | 미체결 주문 실시간 추적, 크래시 복구 지원 (v14.22.0) |
| `exposure.py` | `ExposureGuard` | 세션 노출도 상한선, max_exposure_multiplier=35.0 (v14.23.0) |
| `recon_journal.py` | `ReconJournal` | SQLite 감사 로그, 10종 이벤트 타입 (v14.23.0) |

### 3.5 `quantbox/data/`

| 파일 | 역할 |
|------|------|
| `downloader.py` | Binance API ← klines 다운로드. 1000개씩 배치 |
| `cache.py` | Memory(LRU) + Disk(gzip pickle) 2계층 캐시 |
| `real_data_loader.py` | 캐시된 parquet/csv 로딩. 백테스트 데이터 소스 |

### 3.6 `quantbox/notifications/`

| 파일 | 역할 |
|------|------|
| `discord.py` | Webhook 기반 알림 (진입/청산/에러/일일보고) |
| `bot.py` | Discord Bot 기반 원격 제어 (!status, !close, !pause 등) |

> **Discord 알림 언어:** 모든 Discord 메시지는 **한국어**로 작성. 서버/클라이언트 TP/SL 체결방식, 활성 포지션 목록, 세션 현황(누적PnL/승률/잔고)을 포함한 구조화된 embed 사용.

### 3.7 `quantbox/` (최상위)

| 파일 | 역할 |
|------|------|
| `scalper.py` | `RubberBandScalper(LoopMixin, DataMixin, BracketMixin, ReconciliationMixin, PositionMixin, CommandsMixin)`. CLI 진입점 (`python -m quantbox.scalper`). `_cycle()`, `_scan_entries()`, `_place_order()`, `_save_state()`, `_load_state()` 포함. |

> 안전장치는 `quantbox/execution/preflight.py` (OrderPreflight), `quantbox/execution/reconciliation.py` (ReconJournal)에 코드로 구현됨. 별도 수동 체크 불필요.

---

## 4. 파일 수정 시 영향 범위 매트릭스

코드를 수정할 때 반드시 영향받는 파일을 함께 확인/수정해야 한다:

| 수정 대상 | 반드시 함께 확인 |
|----------|-----------------|
| `settings.py` 파라미터 추가 | `.env.example`, `.env` |
| `rubberband.py` 신호 로직 변경 | `scalper.py`, `vectorized_backtest.py`, `test_rubberband.py` |
| `alpha_engine.py` 수정 | `test_alpha_engine.py`, `vectorized_backtest.py` |
| `vol_selector.py` 수정 | `test_vol_selector.py`, 백테스트 스크립트들 |
| `LimitOrderChaser` 수정 | `limit_chaser.py`, `scalper.py`, `test_limit_chaser.py` |
| `scalper.py` 트레이딩 로직 | `vectorized_backtest.py` 동기화 확인, `test_scalper.py` |
| `vectorized_backtest.py` 수정 | 모든 `scripts/` 백테스트, `scalper.py`와 패리티 확인 |
| `core/scalper_state.py` 수정 | `scalper.py` (`_save_state`/`_load_state`), `tests/conftest.py` (isolate fixture), `test_scalper.py`, `test_hwm_reset.py`, `test_state_restore_ratchet.py` |
| `core/loop_mixin.py` 수정 | `scalper.py`, `test_discord_bot.py` (datetime/time patch 경로) |
| `execution/data_mixin.py` 수정 | `scalper.py`, 백테스트 klines 로직 패리티 확인 |
| `core/bot_utils.py` 수정 | `scalper.py`, `loop_mixin.py` (GIT_COMMIT import) |
| Discord 명령어 추가 | `bot.py`, `test_discord_bot.py` |
| 새 모듈/파일 추가 | `README.md` 프로젝트 구조, 해당 `__init__.py`, 테스트 파일, `agents.md` |

---

## 5. 테스트 현황 (1011+ tests)

| 테스트 파일 | 검증 대상 |
|------------|----------|
| `test_vol_straddle` | Config/VR/신호/리샘플링/SL/TP/max_hold/멀티TF/상태 직렬화 (34개) |
| `test_hedge_mode` | Hedge mode 진입/청산/compound key 파싱 |
| `test_bracket_repair` | SL/TP 자동 배치, bracket repair 재시도 |
| `test_backtest_live_parity` | 신호/비용/사이징 일치 |
| `test_settings` | 환경변수 검증 |
| `test_scalper` | 라이브 사이클/포지션 관리/state 복원 |
| `test_limit_chaser` | 주문 추격, 시장가 폴백 |
| `test_state_persistence` | 원자적 쓰기, 라운드트립 |
| `test_discord_bot` | 원격 제어 명령 |

레거시 (`test_rubberband`, `test_mtf_*`, `test_v12*`, `test_v9*`, `test_elliott` 등) — 수정 불필요.

---

## 6. 수정 시 위험 수준

| 위험 수준 | 예시 | 에이전트 행동 |
|----------|------|-------------|
| 🟢 낮음 | 로그 메시지, docstring | 즉시 실행 |
| 🟡 중간 | 새 필터, 알림 변경 | 실행 + 테스트 확인 |
| 🔴 높음 | 신호/청산/비용/사이징 로직 | 사용자 확인 + 패리티 검증 필수 |
| ⛔ 금지 | 안전장치 제거, mock 데이터, 패리티 파괴 | 절대 실행 불가 |
