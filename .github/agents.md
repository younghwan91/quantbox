# QuantBox Agent Instructions

> 코드 사실(Facts) 레퍼런스. 원칙은 `copilot-instructions.md` 참조.  
> **현재 프로덕션:** VolStraddle v14.17.0 (--strategy vol_straddle)

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
| **Discord Bot/Notifier** (`bot.py`, `discord.py`) | 원격 제어(!status, !close all 등) 및 실시간 알림 |

### 1.1 프로덕션 전략 구성 (v14.17.x — VolStraddle, r73)

> ⚠️ **파라미터 source of truth = `quantbox/strategy/vol_straddle/config.py`.**  
> 아래는 빠른 참조용. 반드시 config.py 읽어서 확인.

**VolStraddle v14.17.x 핵심 파라미터:**
```
전략:        VolStraddle (--strategy vol_straddle)
레버리지:    VR base_leverage=30, MA ma_cluster_leverage=21, max_concurrent=10
L/N:         VR=3.0, MA=2.1
SL:          VR d_sl=1%, MA ma_cluster_sl=2%
Exit (VR):   hard TP=20% (dead cap), trailing activate=+1.0%, trail_distance=0.2%  ← r69 기준
TBR Gate:    f5_tbr_enabled=False
MaxHold:     48h (192바 @15m)
비용 (RT):   0.08% (0.04%/side × 2)
베이스라인:  r73 3전략 trail: PF=2.002, Calmar=1695.6 (D+S+MA, 60 syms, 5.67yr, 2026-03-13)
```

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

### 2.1 비용 모델

| 비용 항목 | 값 | 비고 |
|-----------|---|------|
| **수수료** | 0.02%/side (메이커) | LimitChaser로 메이커 체결 목표 |
| **슬리피지** | 0.02%/side | 보수적 추정 |
| **펀딩비** | 0.01%/8h | 48h 포지션 기준 약 0.06% |
| **왕복 비용** | ~0.08% + 펀딩비 | 트레이드당 고정 차감 |

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

### 3.2 `quantbox/strategy/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `rubberband.py` | `RubberBandStrategy` | 레거시 평균회귀 (비프로덕션) |
| `elliott.py` | `ElliottStrategy` | 레거시 Elliott Wave (비프로덕션) |
| `vol_straddle/` | `VolStraddleConfig`, `VolStraddleStrategy`, `compute_vr`, `compute_momentum`, `gen_vr_signal`, `gen_vr_dur_signal`, `make_direction_momentum`, `resample_ohlcv` | VR 압축 + D_mom8 방향 + **S_d8 스트래들** (VR<0.55, 8바 스퀴즈). Hedge Mode 복합 키 (`SYM:LONG`/`SYM:SHORT`). 완성된 바만 참조 (zero lookahead) |
| `alpha_engine.py` | `AlphaEngine`, `AlphaConfig` | 멀티팩터 신호 |
| `vol_selector.py` | `VolatilitySelector` | 실현변동성 기반 심볼 선택 |

### 3.3 `quantbox/backtest/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `vectorized_backtest.py` | `VectorizedBacktest`, `BacktestConfig`, `BacktestResult`, `_backtest_core_partial` | Numba 가속 백테스트. 단일/포트폴리오 지원. Delta-based 비용 모델 (부분 청산 대응) |

### 3.4 `quantbox/execution/`

| 파일 | 역할 |
|------|------|
| `brackets.py` | `BracketMixin` — SL/TP 알고주문 배치/감지/repair |
| `positions.py` | `PositionMixin` — 포지션 진입/청산/사이징 |
| `reconciliation.py` | `ReconciliationMixin` — 거래소 상태 동기화, ghost 감지 |
| `data_mixin.py` | `DataMixin` — `get_trading_symbols()`, `get_symbol_info()`, `download_klines()`. 마지막 미완성 캔들 자동 제거 (backtest-live parity) |

### 3.5 `quantbox/data/`

| 파일 | 역할 |
|------|------|
| `downloader.py` | Binance API ← klines 다운로드. 1000개씩 배치 |
| `cache.py` | Memory(LRU) + Disk(gzip pickle) 2계층 캐시 |
| `real_data_loader.py` | 캐시된 parquet/csv 로딩. 백테스트 데이터 소스 |

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

### 3.6 `quantbox/notifications/`

| 파일 | 역할 |
|------|------|
| `discord.py` | Webhook 기반 알림 (진입/청산/에러/일일보고) |
| `bot.py` | Discord Bot 기반 원격 제어 (!status, !close, !pause 등) |

> **Discord 알림 언어:** 모든 Discord 메시지는 **한국어**로 작성. 서버/클라이언트 TP/SL 체결방식, 활성 포지션 목록, 세션 현황(누적PnL/승률/잔고)을 포함한 구조화된 embed 사용.

### 3.7 `quantbox/` (최상위)

| 파일 | 역할 |
|------|------|
| `scalper.py` | `RubberBandScalper(LoopMixin, DataMixin, BracketMixin, ReconciliationMixin, PositionMixin, CommandsMixin)`. 1,371줄. CLI 진입점 (`python -m quantbox.scalper`). `_cycle()`, `_scan_entries()`, `_place_order()`, `_save_state()`, `_load_state()` 포함. |

---

## 4. 안전장치 목록 (Safety Checklist)

에이전트는 코드 수정 시 아래 안전장치들이 **절대 제거되거나 우회되지 않도록** 해야 한다:

### 4.1 주문 실행 안전
- [ ] LimitOrderChaser의 시장가 폴백 + 5회 체결 검증 로직
- [ ] ACK 응답(`executedQty=0`) 감지 및 재검증
- [ ] `_round_qty()` — step_size 정밀도 준수
- [ ] 최소 notional 검증 ($10 이상)

### 4.2 위험 관리
- [ ] Max Drawdown Breaker — 세션 HWM 대비 10% DD 시 전체 청산 + 정지
- [ ] **v12.4.6 HWM Reset** — 0 포지션 재시작 시 HWM/initial_balance/trade_history 리셋 (팬텀 DD halt 방지)
- [ ] Daily Loss Limit — 당일 잔고 대비 5% 손실 시 신규 진입 차단
- [ ] Max Concurrent — 동시 포지션 수 제한 (max_concurrent=10)

### 4.3 포지션 관리 (VolStraddle)
- [ ] SL / Hard TP / Trailing / MaxHold 4중 청산 조건
- [ ] 청산 실패 시 포지션 재등록 → 다음 사이클 재시도
- [ ] 동일 심볼 중복 진입 방지 (`has_position()`)
- [ ] max_hold 강제 청산 (192바 = 48h)

### 4.4 상태 영속화 & 복구
- [ ] `save_scalper_state()` (`core/scalper_state.py`) — tmpfile → os.replace 원자적 쓰기
- [ ] `load_scalper_state()` (`core/scalper_state.py`) — 시작 시 크래시 전 상태 복원
- [ ] compound key 파싱 (`SYM:LONG` → `api_sym=SYM, position_side=LONG`) — v14.4.1 수정
- [ ] 로그 파일 로테이션 — RotatingFileHandler 50MB/5백업
- [ ] **Bracket Repair** — 복원된 포지션의 누락 SL/TP 자동 배치 (`_sync_server_orders`)
- [ ] **Ghost Fill Audit** — ghost reconcile 발생 시 fill/orderId 전체 로깅 (`_reconcile_positions`)

### 4.5 실거래-백테스트 패리티 (VolStraddle)
- [x] 동일 신호 로직 (`compute_vr`, `gen_vr_signal`, `make_direction_momentum`)
- [x] 동일 비용 모델 (수수료 0.04% + 슬리피지 0.04% per side)
- [x] 완성된 바만 참조 — MTF 참조 시 `j = (i // ratio) - 1`
- [x] `portfolio_sim(max_concurrent=10)` — 슬롯 수 production과 동일
- [x] Algo order: `TRIGGERED` + `FINISHED` 모두 체결로 처리 (v14.3.x)

---

## 5. 파일 수정 시 영향 범위 매트릭스

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

## 6. 핵심 상수 & 기본값 참조

### 6.1 VolStraddle 전략 파라미터 (r47 기준)

> **Source of truth: `quantbox/strategy/vol_straddle/config.py`** — 반드시 파일 읽어서 확인

| 파라미터 | 값 | 용도 |
|---------|-----|------|
| `vr_short` | 8 | VR 단기 ATR 기간 |
| `vr_long` | 56 | VR 장기 ATR 기간 |
| `d_vr_threshold` | 0.50 | D_mom8 진입 VR 임계값 |
| `d_sl` | 0.01 (1%) | D 손절 |
| `d_tp` | 0.12 (12%) | D 하드 TP |
| `d_trail_activate` | **0.010 (+1.0%)** | 트레일링 활성화 수익 ← r69 변경 |
| `d_trail_distance` | **0.002 (0.2%)** | 트레일링 거리 ← r47 변경 |
| `s_vr_threshold` | 0.55 | S_d8 스트래들 VR 임계값 |
| `s_squeeze_dur` | 4 bars | 스트래들 최소 지속 기간 |
| `s_sl` | 0.015 (1.5%) | 스트래들 손절 (per leg) |
| `s_tp` | 0.05 (5%) | 스트래들 TP (per leg) |
| `momentum_period` | 8 | 방향 결정 모멘텀 lookback |
| `max_hold_15m` | 192 bars (48h) | 최대 보유 (15m TF) |
| `base_leverage` | 30.0 | VR Half-Kelly 레버리지 |
| `ma_cluster_leverage` | 21.0 | MA 레버리지 |
| `max_concurrent` | 10 | 최대 동시 포지션 |
| `f5_tbr_enabled` | **False** | TBR Gate ← r47 이후 비활성 |
| `dynamic_capital_enabled` | False | 동적 자본 배분 비활성 |
| `fee_rate` | 0.0002 | 메이커 수수료/side |
| `slippage_rate` | 0.0002 | 슬리피지/side |

---

## 7. 테스트 현황 (1011+ tests)

**VolStraddle 핵심 테스트:**

| 테스트 파일 | 검증 대상 |
|------------|----------|
| `test_vol_straddle` | Config/VR/신호/리샘플링/SL/TP/max_hold/멀티TF/상태 직렬화 (34개) |
| `test_hedge_mode` | Hedge mode 진입/청산/compound key 파싱 (`TestLoadStateCompoundKeyBug`, `TestHedgeCloseNoLeverageChange`) |
| `test_bracket_repair` | SL/TP 자동 배치, bracket repair 재시도 |
| `test_backtest_live_parity` | 신호/비용/사이징 일치 |

**인프라/공통 테스트:**

| 테스트 파일 | 검증 대상 |
|------------|----------|
| `test_settings` | 환경변수 검증 |
| `test_scalper` | 라이브 사이클/포지션 관리/state 복원 |
| `test_limit_chaser` | 주문 추격, 시장가 폴백 |
| `test_state_persistence` | 원자적 쓰기, 라운드트립 |
| `test_discord_bot` | 원격 제어 명령 |

**레거시 테스트** (RubberBand/Elliott — 수정 불필요):  
`test_rubberband`, `test_mtf_*`, `test_v12*`, `test_v9*`, `test_partial_exit`, `test_elliott` 등

---

## 8. 수정 시 위험 수준

| 위험 수준 | 예시 | 에이전트 행동 |
|----------|------|-------------|
| 🟢 낮음 | 로그 메시지, docstring | 즉시 실행 |
| 🟡 중간 | 새 필터, 알림 변경 | 실행 + 테스트 확인 |
| 🔴 높음 | 신호/청산/비용/사이징 로직 | 사용자 확인 + 패리티 검증 필수 |
| ⛔ 금지 | 안전장치 제거, mock 데이터, 패리티 파괴 | 절대 실행 불가 |
