# QuantBox Agent Instructions

> 이 문서는 AI 에이전트가 QuantBox 코드베이스를 이해하고, 정확한 작업을 수행하기 위한 **시스템 아키텍처 맵** 겸 **행동 규범**입니다.
> `copilot-instructions.md`가 "무엇을 지켜야 하는가"(원칙)를 정의한다면, 이 문서는 "코드가 실제로 어떻게 동작하는가"(사실)를 기술합니다.

---

## 1. 시스템 개요

QuantBox v1.5.0은 **바이낸스 USDT-M 선물** 위에서 동작하는 페어 트레이딩 시스템이다.

| 구성 요소 | 역할 |
|-----------|------|
| **LivePairTrader** | 실거래 오케스트레이터. 15분 사이클로 페어 탐색 → 진입 → 모니터링 → 청산 |
| **ContinuousPairBacktest** | 동일한 진입/청산/비용 로직을 과거 데이터에 적용 |
| **LimitOrderChaser** | 메이커 지정가 추격 주문. 타임아웃 시 시장가 폴백 |
| **FilterPipeline** | 통계적 페어 선별 (상관관계 → 공적분 → 반감기 → 허스트) |
| **Vol-Inverse Sizing** | 변동성 반비례 포지션 사이징. 동일 달러 손절액 보장 |
| **Discord Bot/Notifier** | 원격 제어(!status, !close all 등) 및 실시간 알림 |

---

## 2. 라이브 트레이딩 전체 흐름 (run_cycle)

```
┌────────────────────────────────────────────────────────────────────┐
│  1. _process_bot_commands()   ← Discord 큐에서 명령 수신          │
│  2. monitor_slot(i)           ← 활성 슬롯: Frozen z-score 모니터링 │
│     → 평균회귀(|z| ≤ exit_z) / 손절(z > stop_z) / 최대보유 초과  │
│  3. get_trading_coins(n)      ← 24h 거래량 상위 N개 코인          │
│  4. download_multi_symbol_data()  ← 15m OHLCV 다운로드 + 캐시    │
│  5. select_pairs() (1회/사이클 캐시) ← FilterPipeline 적용        │
│  6. 유휴 슬롯마다:                                                │
│     a. calculate_slot_capital()   → 풀 기반 자본 배분             │
│     b. find_best_pair(cached)     → 점수순 정렬, 중복 심볼 제외   │
│     c. execute_pair_trade()       → 5중 게이트 통과 후 4건 주문   │
│  7. notify_cycle_summary()    → Discord 요약                     │
│  8. 24시간마다 daily_report() → Discord 일간 리포트              │
└────────────────────────────────────────────────────────────────────┘
```

### 2.1 진입 5중 게이트 (execute_pair_trade)

모든 진입은 아래 5개 게이트를 **순서대로** 통과해야 한다. 하나라도 실패하면 진입 불가:

| # | 게이트 | 조건 | 기본값 |
|---|--------|------|--------|
| 1 | **BTC Crash Gate** | BTC 15m 수익률 절대값 > threshold → 차단 | ±3%, 8캔들 |
| 2 | **Z-Score Cap** | \|z\| > cap → 레짐 변화로 간주, 거부 | 4.0 |
| 3 | **Vol Expansion Gate** | √S / formation_std > gate → 구조 변화, 거부 | 2.0 |
| 4 | **Z-Score Threshold** | \|z\| < entry_zscore → 시그널 부족, 거부 | 2.0 |
| 5 | **자본 충분성** | slot_capital < 최소 주문 크기 → 거부 | — |

### 2.2 청산 조건 (monitor_slot)

| 조건 | 판정 | 비고 |
|------|------|------|
| **평균회귀** | \|z\| ≤ exit_zscore | Frozen z-score (formation 기간 통계) 사용 |
| **손절** | z가 진입 방향 반대로 > stop_loss_zscore | 3.5 기본값 |
| **최대보유** | bars_held > max_holding_bars | 시간 기반 강제 청산 |

**동적 청산**: `use_dynamic_exit=True`일 때 exit_zscore는 반감기에 비례하여 조절됨:
```
exit_z = base + ((hl - hl_min) / (hl_max - hl_min)) × range
```

### 2.3 주문 실행 (LimitOrderChaser)

```
1. get_best_price() → BBO (best bid/ask)
2. 지정가 POST_ONLY 주문 제출
3. chase_interval(1초)마다 체결 확인
4. 미체결 시 취소 → 새 BBO로 재제출 (최대 10회)
5. timeout(30초) 초과 시 → 시장가 폴백
6. 시장가 폴백 후 5회까지 체결 검증 (ACK 응답 대응)
```

### 2.4 자본 배분 (calculate_slot_capital)

```
total_book = available_capital + Σ(활성 슬롯 자본)
slot_target = total_book × position_size_pct    (기본: 1/max_traders = 20%)
margin = min(slot_target, available_capital)
notional = margin × leverage
```

### 2.5 Vol-Inverse Sizing (변동성 반비례 포지션 사이징)

사이클 내 모든 후보 페어의 `spread_std` 중 최솟값을 `anchor_std`로 설정하고, 각 페어의 notional을 변동성에 반비례하여 조절한다:

```
anchor_std = min(spread_std for all candidates in cycle)
vol_ratio  = min(anchor_std / spread_std, 1.0)
notional   = base_notional × vol_ratio
```

**핵심 성질**: 모든 페어의 달러 손절액이 동일해진다:
```
stop_loss_dollar = stop_z × spread_std × notional
                 = stop_z × spread_std × base × (anchor / spread_std)
                 = stop_z × anchor × base   ← CONSTANT
```

- Low-vol 페어(anchor): 풀 자본 배분 (vol_ratio = 1.0)
- High-vol 페어: 비례 축소 (vol_ratio < 1.0)
- 실거래(`live_trader.py`)와 백테스트(`continuous_backtest.py`) 양쪽에 동일 로직 적용

### 2.6 PnL 공식 (백테스트/실거래 통일)

```
# 실거래 (leg-split 방식):
s1_capital = notional / (1 + HR)
s2_capital = HR × s1_capital
pnl = qty1 × Δprice1 + qty2 × Δprice2

# 백테스트 (spread-based 방식 — 수학적으로 동일):
gross_pnl = signal × spread_change × notional / (1 + HR)
```

수익률 % 분모는 **margin** (= notional / leverage)으로 통일:
```
return_pct = net_pnl / margin × 100
```

### 2.7 비용 모델

| 비용 항목 | 공식 (레그당) | 비고 |
|-----------|---------------|------|
| **수수료** | notional × fee × 2 (진입+청산) | 메이커 0.02%, 테이커 0.04% |
| **슬리피지** | notional × slippage × 2 | 백테스트 0.1%, 실거래는 실제 체결가 |
| **펀딩비** | notional × rate × (hours/8) × 2 | 실거래: API로 실시간 조회 |

---

## 3. 핵심 모듈 맵

### 3.1 `quantbox/core/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `settings.py` | `QuantBoxSettings` | pydantic-settings 기반 환경변수 검증. `get_settings()` 싱글턴 |
| `config.py` | `Config`, `StrategyConfig`, `RiskConfig` | YAML/dict 기반 설정. 백테스트에서 주로 사용 |
| `events.py` | `SignalEvent`, `OrderEvent`, `FillEvent`, `EventQueue` | 이벤트 드리븐 아키텍처 기반 |
| `trader_slot.py` | `TraderSlotState`, `calculate_slot_capital` | 멀티슬롯 상태 관리 + 자본 배분 |
| `limit_chaser.py` | `LimitOrderChaser`, `ChaseResult` | 메이커 주문 추격 + 시장가 폴백 |

### 3.2 `quantbox/strategy/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|

| `filters.py` | `FilterPipeline`, 8개 필터 | 통계적 페어 검증 파이프라인 |
| `pair_selector.py` | `PairSelector`, `PairCandidate` | C(n,2) 페어 평가 + 가중 점수 랭킹 |
| `base.py` | `Strategy(ABC)` | 전략 추상 기반 클래스 |

### 3.3 `quantbox/backtest/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `continuous_backtest.py` | `ContinuousPairBacktest`, `BacktestState` | 연속 백테스트 엔진. 실거래와 동일한 로직 |
| `metrics.py` | `PerformanceMetrics`, `monte_carlo_analysis` | Sharpe, Sortino, Calmar, MDD, Monte Carlo |

### 3.4 `quantbox/data/`

| 파일 | 역할 |
|------|------|
| `downloader.py` | Binance API ← klines 다운로드. 1000개씩 배치 |
| `cache.py` | Memory(LRU) + Disk(gzip pickle) 2계층 캐시 |
| `preprocessor.py` | 헤지비율, 스프레드, 공적분, 허스트 계산 |
| `real_data_loader.py` | 캐시된 parquet/csv 로딩. 백테스트 데이터 소스 |

### 3.5 `quantbox/notifications/`

| 파일 | 역할 |
|------|------|
| `discord.py` | Webhook 기반 알림 (진입/청산/에러/일일보고) |
| `bot.py` | Discord Bot 기반 원격 제어 (!status, !close, !pause 등) |

### 3.6 `quantbox/utils/`

| 파일 | 역할 |
|------|------|
| `numba_ops.py` | `@numba.jit(nopython=True)` 최적화 함수 11개 |
| `costs.py` | `TransactionCosts` 비용 모델. 왕복 비용 추정 |

---

## 4. 안전장치 목록 (Safety Checklist)

에이전트는 코드 수정 시 아래 안전장치들이 **절대 제거되거나 우회되지 않도록** 해야 한다:

### 4.1 주문 실행 안전
- [ ] LimitOrderChaser의 시장가 폴백 + 5회 체결 검증 로직
- [ ] ACK 응답(`executedQty=0`) 감지 및 재검증
- [ ] `round_price()` / `round_quantity()` — tickSize, stepSize 정밀도 준수
- [ ] 고아 포지션 감지 (`close_slot_positions` 후 `get_current_positions` 확인)

### 4.2 레짐 보호
- [ ] BTC Crash Gate — `check_btc_crash_gate()`
- [ ] Z-Score Cap (4.0) — 극단 z-score 진입 차단
- [ ] Vol Expansion Gate (2.0) — 변동성 급변 차단
- [ ] Hurst Filter (< 0.5) — 평균회귀 아닌 페어 필터링

### 4.3 포지션 관리
- [ ] `close_all_positions()` — 에러 시 전체 청산
- [ ] 블랙리스트 TTL (3600초) — 실패 심볼 재진입 방지
- [ ] 동적 청산 z-score — 반감기 기반 적응형 청산
- [ ] 최대 보유 기간 강제 청산

### 4.4 자본 보호
- [ ] 풀 기반 자본 배분 — `calculate_slot_capital()`
- [ ] available_capital 초과 배분 방지 (`min(target, available)`)
- [ ] 잔고 부족 시 알림 + 진입 거부

### 4.5 실거래-백테스트 패리티
- [x] `_get_effective_exit_zscore()` — 양쪽 동일 공식
- [x] `ExitReason` enum 공유
- [x] 동일 비용 모델 (수수료 + 슬리피지 + 펀딩비)
- [x] 동일 자본 배분 로직
- [x] 동일 필터 파이프라인
- [x] BTC Crash Gate — 백테스트에 `_check_btc_crash_gate()` 구현 (v1.4.0)
- [x] Z-Score Cap — 백테스트 `_enter_slot()`에 구현 (v1.4.0)
- [x] Vol Expansion Gate — 백테스트 `_enter_slot()`에 구현 (v1.4.0)
- [x] Formation window 동기화 — `FORMATION_HOURS=60` (v1.4.0)
- [x] Vol-Inverse Sizing — 양쪽 동일 anchor_std 기반 비율 적용 (v1.5.0)
- [x] `test_backtest_live_parity.py`로 검증

---

## 5. 파일 수정 시 영향 범위 매트릭스

코드를 수정할 때 반드시 영향받는 파일을 함께 확인/수정해야 한다:

| 수정 대상 | 반드시 함께 확인 |
|----------|-----------------|
| `settings.py` 파라미터 추가 | `.env.example`, `MANUAL.md`, `to_trader_config()`, `live_trader.py`에서 해당 값 사용 여부 |
| 진입/청산 로직 변경 | `live_trader.py` **AND** `continuous_backtest.py` 양쪽 동기화 |
| `_get_effective_exit_zscore()` | `live_trader.py`, `continuous_backtest.py`, `test_dynamic_exit.py`, `test_backtest_live_parity.py` |
| `FilterPipeline` 수정 | `filters.py`, `pair_selector.py`, `test_filters.py` |
| `LimitOrderChaser` 수정 | `limit_chaser.py`, `live_trader.py`, `test_limit_chaser.py` |
| `TraderSlotState` 수정 | `trader_slot.py`, `live_trader.py`, `continuous_backtest.py`, `test_multi_trader.py` |
| Vol-Inverse Sizing 수정 | `live_trader.py`, `continuous_backtest.py`, `test_vol_inverse_sizing.py` |
| `TransactionCosts` 수정 | `costs.py`, `continuous_backtest.py`, `test_pnl_accuracy.py` |
| numba 함수 수정 | `numba_ops.py`, `__pycache__/` 캐시 무효화 필요, `test_numba_ops.py` |
| Discord 명령어 추가 | `bot.py`, `live_trader.py`의 `_handle_command()`, `test_discord_bot.py` |
| 새 모듈/파일 추가 | `README.md` 프로젝트 구조 트리, 해당 `__init__.py`, 테스트 파일 |

---

## 6. 핵심 상수 & 기본값 참조

| 파라미터 | 기본값 | 소스 | 용도 |
|---------|--------|------|------|
| `entry_zscore` | 2.0 | settings.py | 진입 임계값 |
| `exit_zscore` | 0.5 | settings.py | 청산 임계값 (동적 청산 시 base) |
| `stop_loss_zscore` | 3.5 | settings.py | 손절 임계값 |
| `z_score_cap` | 4.0 | settings.py | 극단 z-score 차단 |
| `vol_expansion_gate` | 2.0 | settings.py | 변동성 급변 차단 |
| `btc_crash_threshold` | 0.03 (3%) | settings.py | BTC 급락 게이트 |
| `btc_crash_window` | 8 | settings.py | BTC 게이트 관찰 캔들 수 |
| `max_traders` | 5 | settings.py | 동시 거래 슬롯 수 |
| `leverage` | 2 | settings.py | 레버리지 배수 |
| `trading_fee` | 0.0005 (0.05%) | settings.py | 거래 수수료 |
| `slippage` | 0.001 (0.1%) | settings.py | 슬리피지 추정치 |
| `funding_rate_8h` | 0.0001 (0.01%) | settings.py | 8시간 펀딩비 |
| `vol_inverse_sizing` | True | backtest config | 변동성 반비례 사이징 활성화 |
| `vol_sizing_max_multiplier` | 3.0 | backtest config | 사이징 최대 배율 제한 |
| `max_holding_hours` | 20 | settings.py | 최대 보유 시간 (80bars@15m) |
| `dynamic_exit_base` | 0.3 | settings.py | 동적 청산 기저값 |
| `dynamic_exit_range` | 0.7 | settings.py | 동적 청산 범위 |
| `BLACKLIST_TTL_SECONDS` | 3600 | live_trader.py | 실패 심볼 블랙리스트 TTL |
| `max_chase_ticks` | 10 | limit_chaser.py | 지정가 추격 최대 횟수 |
| `chase_timeout` | 30초 | limit_chaser.py | 추격 타임아웃 |
| `formation_hours` | 60 | settings.py | 페어 선별용 과거 데이터 길이 (60h = 240 bars @ 15m) |
| `formation_bars` | 240 | backtest | 형성 구간 (60시간 @ 15m) |

---

## 7. 테스트 현황 (20파일, ~534 테스트)

| 테스트 파일 | 검증 대상 | 핵심 관점 |
|------------|----------|----------|
| `test_backtest_live_parity` | 실거래-백테스트 로직 일치 | Hurst 필터, 청산 로직 동기화 |
| `test_pnl_accuracy` | PnL 계산 정확성 | 체결가, 비용 차감, 수익 계산 |
| `test_aggressive_sizing` | 자본 배분 | 레버리지, Kelly, slot_capital |
| `test_btc_crash_gate` | BTC 게이트 | 차단/통과, 임계값, 에지 케이스 |
| `test_dynamic_exit` | 동적 청산 | 반감기 기반 exit_zscore 계산 |
| `test_multi_trader` | 멀티슬롯 | 상태 관리, 자본 배분, 심볼 비중복 |
| `test_cached_pair_search` | 페어 캐싱 | select_pairs() 호출 횟수, 캐시 |
| `test_blacklist` | 블랙리스트 | TTL 만료, Discord 명령, 지속성 |
| `test_limit_chaser` | 주문 추격 | Paper/Live 체결, 시장가 폴백 |
| `test_vol_inverse_sizing` | Vol-Inverse Sizing | 비율 공식, anchor_std, 달러 손절 균등화 |
| `test_filters` | 통계 필터 | 개별 필터 + 파이프라인 조합 |
| `test_events` | 이벤트 시스템 | 생성, FIFO, 핸들러 디스패치 |
| `test_discord_bot` | Discord Bot | 명령어 파싱, 큐, 러너 생명주기 |
| `test_discord_notifications` | Discord 알림 | 웹훅 페이로드, 색상 코딩, 에러 복원 |
| `test_numba_ops` | Numba 함수 | z-score, half-life, Hurst, OLS |
| `test_metrics` | 성과 지표 | Sharpe, MDD, Monte Carlo, 월간 수익 |
| `test_settings` | 설정 검증 | 환경변수, 기본값, 별칭, 타입 에러 |
| `test_config` | Config | 생성, 검증, 직렬화 |
| `test_cache` | 캐시 | Memory LRU, Disk, 만료, Timeframed |

---

## 8. 에이전트 행동 규칙

### 8.1 코드 수정 전 필수 확인

1. **패리티 위반 금지**: 진입/청산/비용 로직을 한쪽만 수정하면 안 된다. `live_trader.py` ↔ `continuous_backtest.py` 양쪽을 반드시 동기화.
2. **안전장치 제거 금지**: 섹션 4의 체크리스트 항목을 제거하거나, 조건을 완화하는 코드는 사용자 명시 승인 없이 작성하지 않는다.
3. **테스트 동반 필수**: 새 함수/클래스 작성 시 pytest 테스트를 반드시 함께 작성한다. 테스트 없는 코드는 미완성이다.
4. **실제 데이터만 사용**: `np.random` 등을 사용한 합성 가격 데이터 생성은 절대 금지. 테스트에서의 mock은 구조적 mock(API 응답 형태)만 허용.

### 8.2 수정 시 영향 범위 체크

코드 수정 후 반드시:
1. 섹션 5의 영향 범위 매트릭스를 확인하여, 관련 파일을 모두 업데이트
2. 관련 테스트를 실행하여 통과 확인
3. `copilot-instructions.md`의 문서 자동 관리 규칙에 따라 README.md, MANUAL.md, .env.example 갱신

### 8.3 문서 동기화 필수

`copilot-instructions.md`와 `agents.md`는 **항상 쌍으로** 최신 상태를 유지해야 한다:
- 개발 원칙/워크플로우 변경 → `copilot-instructions.md` 갱신
- 아키텍처/모듈/안전장치/파라미터/트레이딩 흐름 변경 → `agents.md` 갱신
- 두 문서 중 하나만 수정하고 다른 하나를 방치하는 것은 **금지**한다.
- 코드 변경이 두 문서 모두에 영향을 줄 경우, 반드시 두 문서를 동시에 업데이트한다.

### 8.4 Git 작업

- 항상 feature/fix 브랜치에서 작업
- develop에 `--no-ff` 머지
- main에 직접 커밋 금지
- 커밋 메시지는 영어로, 변경 내용을 구체적으로 기술

### 8.5 위험 수준별 대응

| 위험 수준 | 예시 | 에이전트 행동 |
|----------|------|-------------|
| 🟢 낮음 | 로그 메시지 변경, docstring 수정 | 즉시 실행 |
| 🟡 중간 | 새 필터 추가, 알림 메시지 변경 | 실행 + 테스트 확인 |
| 🔴 높음 | 진입/청산 로직, 비용 계산, 자본 배분 변경 | 사용자 확인 요청 후 실행. 패리티 검증 필수 |
| ⛔ 금지 | 안전장치 제거, 실데이터 규칙 위반, 패리티 파괴 | 절대 실행 불가. 사용자에게 위험성 경고 |
