# QuantBox Agent Instructions

> 이 문서는 AI 에이전트가 QuantBox 코드베이스를 이해하고, 정확한 작업을 수행하기 위한 **시스템 아키텍처 맵** 겸 **행동 규범**입니다.
> `copilot-instructions.md`가 "무엇을 지켜야 하는가"(원칙)를 정의한다면, 이 문서는 "코드가 실제로 어떻게 동작하는가"(사실)를 기술합니다.

---

## 1. 시스템 개요

QuantBox v14.3.0는 **바이낸스 USDT-M 선물** 위에서 동작하는 **멀티 전략 스퀴즈 시스템**이다.
세 가지 전략을 지원한다:
1. **Rubber Band Scalper** (레거시): 단일 자산 평균회귀 — 급등/급락 후 스냅백 포착 (15m)
2. **Elliott Wave Strategy** (v11.6): ZigZag ABC 패턴 + Fibonacci 검증 + Timing Alpha Filter (4h)
3. **VolStraddle** (v14.3.0, 30x, L/N=3.0): VR 압축 감지 → 모멘텀 방향 진입 + **S_d8 스트래들** (15m/1h/4h 멀티TF, **Hedge Mode**, ATR 8/56, 30x leverage, max_concurrent=10) + **F5 TBR Gate**

| 구성 요소 | 역할 |
|-----------|------|
| **RubberBandScalper** (`scalper.py`) | 실거래 오케스트레이터. 15분 사이클로 신호 → 진입 → 모니터링 → 청산. `--strategy` CLI로 전략 선택 (rubberband/elliott/vol_straddle). v11.1: 서버사이드 STOP 손절 + 레버리지 검증 |
| **RubberBandStrategy** (`rubberband.py`) | 기본 평균회귀 신호 생성기 (Numba 가속, 3.2%+ 급등/급락 포착) + Fibonacci Exit (v9.0) + MACD/Stochastic Confluence (v9.0) + MTF 확인 + 독립 4H 트레이딩 + Symbol Whitelist (v10.0) |
| **ElliottStrategy** (`elliott.py`) | Elliott Wave ABC 패턴 전략 (Numba 가속). ZigZag 피봇 → ABC Fibonacci 검증 → 탐색적/확인 2단계 포지셔닝 → 50% 부분 익절 → 브레이크이븐 → 트레일링 (3×ATR). BTC 매크로 바이어스 (200/500 EMA) |
| **VolStraddleStrategy** (`vol_straddle/`) | VR 압축 감지 + D_mom8 방향 진입 + **S_d8 스트래들** (양방향 동시). 15m/1h/4h 멀티TF. **Hedge Mode** (`positionSide` LONG/SHORT). 복합 키: `SYM:LONG`/`SYM:SHORT`. SL=1%, hard TP=12%, trailing +2%/0.3%, MaxHold=48h. **F5 TBR Gate**: D_mom8 z-score 필터 (v14.3.0). **현재 프로덕션** |
| **AlphaEngine** (`alpha_engine.py`) | 멀티팩터 신호 엔진 (볼륨/모멘텀/ATR/트레일링스탑 필터 토글) |
| **VolatilitySelector** (`vol_selector.py`) | 30일 실현변동성 기준 Top-N 심볼 동적 선택 (10일 리밸런싱) |
| **VectorizedBacktest** (`vectorized_backtest.py`) | Numba 가속 백테스트, 비용/펀딩비 반영 |
| **LimitOrderChaser** (`limit_chaser.py`) | 메이커 지정가 추격 주문. 타임아웃 시 시장가 폴백 |
| **State Persistence** (`state_persistence.py`) | JSON 기반 상태 영속화. 원자적 쓰기. 크래시 복구 지원 |
| **Drawdown / Daily Loss Breaker** | 세션 HWM 10% / 당일 5% 초과 시 자동 포지션 청산 + 일시정지 |
| **Discord Bot/Notifier** (`bot.py`, `discord.py`) | 원격 제어(!status, !close all 등) 및 실시간 알림 |

### 1.1 프로덕션 전략 구성 (v14.3.0 — VolStraddle, 현재 Live)

> ⚠️ **현재 프로덕션은 VolStraddle v14.3.0** (RubberBand 아님).  
> 파라미터 source of truth = `quantbox/strategy/vol_straddle/config.py`.  
> 아래 RubberBand 구성은 레거시 히스토리 참고용.

**VolStraddle v14.3.0 핵심 파라미터:**
```
전략:        VolStraddle (--strategy vol_straddle)
레버리지:    VR base_leverage=30, MA ma_cluster_leverage=21, max_concurrent=10
L/N:         VR=3.0, MA=2.1
엣지 지표:   VR N=2,229, PF=1.308; MA N=1,631, PF=1.365; Combined PF=1.324, Sharpe=3.39
SL:          VR d_sl=1%, MA ma_cluster_sl=2%
Exit (VR):   hard TP=12%, trailing activate=+2%, trail_distance=0.3%
TBR Gate:    f5_tbr_enabled=True, threshold=0.5, window=192
MaxHold:     48h (192바 @15m)
비용 (RT):   0.08% (0.04%/side × 2)
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

### 3.2 `quantbox/strategy/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `rubberband.py` | `RubberBandStrategy`, `RubberBandConfig`, `compute_signals`, `compute_signals_with_btc_directional`, `compute_signals_regime_switch`, `compute_signals_confidence`, `compute_signals_with_partial_exit`, `compute_signals_partial_exit_btc_directional`, `compute_rr_gated_leverage`, `apply_rr_gated_leverage`, `apply_mtf_confirmation`, `apply_mtf_leverage`, `compute_signals_4h_independent`, `_compute_ema_inline`, `_compute_stochastic_inline`, `_compute_macd_inline`, `compute_signals_v9`, `compute_atr_pct` | 기본 평균회귀 신호 + BTC 레짐 스위치 + 컨피던스 필터링 + BTC 방향 정렬 + 부분 청산 + R:R-Gated 레버리지 + **MTF 확인 (v7.0)** + **독립 4H 트레이딩 (v7.0)** + **Fibonacci Exit + MACD/Stochastic Confluence (v9.0)** + **Adaptive Threshold (v11.4)** (Numba 컴파일) |
| `elliott.py` | `ElliottConfig`, `ElliottStrategy`, `compute_elliott_signals`, `compute_ema`, `compute_rsi`, `compute_macd`, `compute_stochastic`, `compute_atr`, `zigzag_pivots`, `detect_abc_patterns`, `detect_bullish_divergence`, `detect_bearish_divergence`, `check_macd_reversal`, `check_stoch_cross`, `detect_ending_diagonal`, `detect_triangle` | Elliott Wave ABC 패턴 전략 (Numba 가속). ZigZag + Fibonacci + 다이버전스/MACD/스토캐스틱 확인 + 웨지/삼각 감지 + 2단계 포지셔닝 + 자본 보존 익절 |
| `vol_straddle/` | `VolStraddleConfig`, `VolStraddleStrategy`, `compute_vr`, `compute_momentum`, `gen_vr_signal`, `gen_vr_dur_signal`, `make_direction_momentum`, `resample_ohlcv`, `_atr_pct` | VR 압축 + D_mom8 방향 + **S_d8 스트래들** (VR<0.55, 8바 스퀴즈). Hedge Mode 복합 키 (`SYM:LONG`/`SYM:SHORT`). 15m/1h/4h 멀티TF. 완성된 바만 참조 (zero lookahead) |
| `alpha_engine.py` | `AlphaEngine`, `AlphaConfig`, `compute_alpha_signals` | 멀티팩터 신호 (볼륨/모멘텀/ATR/트레일링) |
| `vol_selector.py` | `VolatilitySelector`, `VolSelectorConfig`, `compute_realized_vol` | 실현변동성 기반 심볼 선택 |

### 3.3 `quantbox/backtest/`

| 파일 | 핵심 클래스/함수 | 역할 |
|------|------------------|------|
| `vectorized_backtest.py` | `VectorizedBacktest`, `BacktestConfig`, `BacktestResult`, `_backtest_core_partial` | Numba 가속 백테스트. 단일/포트폴리오 지원. Delta-based 비용 모델 (부분 청산 대응) |

### 3.4 `quantbox/data/`

| 파일 | 역할 |
|------|------|
| `downloader.py` | Binance API ← klines 다운로드. 1000개씩 배치 |
| `cache.py` | Memory(LRU) + Disk(gzip pickle) 2계층 캐시 |
| `real_data_loader.py` | 캐시된 parquet/csv 로딩. 백테스트 데이터 소스 |

### 3.5 `quantbox/notifications/`

| 파일 | 역할 |
|------|------|
| `discord.py` | Webhook 기반 알림 (진입/청산/에러/일일보고) |
| `bot.py` | Discord Bot 기반 원격 제어 (!status, !close, !pause 등) |

> **Discord 알림 언어:** 모든 Discord 메시지는 **한국어**로 작성. 서버/클라이언트 TP/SL 체결방식, 활성 포지션 목록, 세션 현황(누적PnL/승률/잔고)을 포함한 구조화된 embed 사용.

### 3.6 `quantbox/` (최상위)

| 파일 | 역할 |
|------|------|
| `scalper.py` | `RubberBandScalper` 라이브 트레이더. CLI 진입점 (`python -m quantbox.scalper`). 커밋 자동추적(로그 헤더, Discord 시작알림). |

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
- [ ] BTC Crash Gate — regime filter로 극단적 시장 상황 시 진입 차단
- [ ] Max Concurrent — 동시 포지션 수 제한 (기본 2개, M-2SSTSX)
- [ ] Risk per Trade — 트레이드당 1% 리스크 사이징

### 4.3 포지션 관리
- [ ] Target/Stop/Trailing/MaxHold 4중 청산 조건
- [ ] **PBT Exit (v12.3)**: 50% partial at TP1 → breakeven floor → ATR trailing runner
- [ ] **v12.4.1–3 Server TP PBT Qty**: 6개 콜사이트 모두 PBT 모드 50% 수량 적용
- [ ] **v12.4.4 SL Ratchet Reset**: 러너 SL breakeven 재배치 후 `_server_sl_pct = 0.0` 리셋
- [ ] **v12.4.5 Partial Close Save**: 파셈 클로즈 성공 후 즉시 `_save_state()` 호출
- [ ] 청산 실패 시 포지션 재등록 → 다음 사이클 재시도
- [ ] 동일 심볼 중복 진입 방지 (`has_position()`)
- [ ] max_hold 강제 청산 (타임아웃)

### 4.4 상태 영속화 & 복구
- [ ] `_save_scalper_state()` — tmpfile → os.replace 원자적 쓰기
- [ ] `_load_scalper_state()` — 시작 시 크래시 전 상태 복원
- [ ] 포지션 + 잔고 + 거래 히스토리 직렬화
- [ ] **v12.4.6**: 0 포지션 복원 시 HWM/trade_history 리셋 (경과 손실 세션 전이 방지)
- [ ] 로그 파일 로테이션 — RotatingFileHandler 50MB/5백업
- [ ] **v11.4.2 Bracket Repair** — 복원된 포지션의 누락 SL/TP 자동 배치 (`_sync_server_orders`)
- [ ] **Ghost Fill Audit (P1)** — ghost reconcile 발생 시 fill time/qty/realizedPnl/commission/orderId + 트리거 주문 정보 전체 로깅 (`_reconcile_positions`)

### 4.5 실거래-백테스트 패리티
- [x] 동일 신호 로직 (`compute_signals_mtf`)
- [x] 동일 비용 모델 (수수료 + 슬리피지 + 펀딩비)
- [x] 동일 포지션 사이징 (1% risk per trade)
- [x] `vectorized_backtest.py`의 `_backtest_core`가 실거래 PnL 로직과 동일
- [x] `entry_stop_mult` 패리티: Numba의 HTF agree 시 `stop_mult * htf_agree_stop_mult`가 live `update_position()`에서도 동일 적용
- [x] `get_effective_stop_mult()`: 진입 시점의 HTF 상태를 미러링하여 정확한 stop 배수 계산
- [x] `settings.py` → `scalper.py main()` 완전 연동: `.env`의 14개 RB 파라미터가 라이브에 자동 적용
- [x] Partial exit: `_partial_close_position()` 라이브 부분 청산 ↔ `_backtest_core_partial()` 백테스트 부분 청산 패리티
- [x] R:R-gated leverage: `compute_rr_gated_leverage()` 라이브 ↔ `apply_rr_gated_leverage()` 백테스트 패리티
- [x] **v7.0 MTF Confirmation**: `apply_mtf_confirmation()` 백테스트 ↔ `_scan_entries()` 라이브 4h SMA 필터 패리티
- [x] **v7.0 MTF 4h Lookahead Fix (v12.6.1)**: `apply_mtf_confirmation()`에서 `j = i // ratio`가 현재 진행 중인 4h 바의 **최종 종가**를 사용 → 라이브는 developing price 사용 → 괴리 발생. Fix: `j = j_raw - 1` (마지막 **완성된** 4h 바 사용). `signals.py`, `backtest_live_period.py`, `r15`, `r16` 4곳 수정 완료. `test_mtf_lookahead.py` 3개 회귀 테스트.
- [x] **v7.0 MTF Leverage**: `apply_mtf_leverage()` 백테스트 ↔ 라이브 MTF 확인 시 2.0x 레버리지 부스트 패리티
- [x] **v7.0 Independent 4H**: `compute_signals_4h_independent()` 백테스트 ↔ `on_bar_4h()` 라이브 4h 신호 패리티
- [x] **v7.0 Live Fix**: volume/taker 데이터 `_scan_entries()` 연결, 4h kline 다운로드, `main()` v7.0 설정 전달
- [x] **v9.0 Fibonacci Exit**: `target_retrace=0.382`, `stop_mult=0.786` 백테스트 ↔ 라이브 동일 적용
- [x] **v9.0 MACD/Stochastic**: `compute_signals_v9()` 백테스트 ↔ `on_bar()` 라이브 v9 디스패치 패리티
- [x] **v9.0 Live Pipeline**: `download_klines(full=True)` 5-tuple (close, high, low, volume, taker_buy) 반환
- [x] **v11.4 Adaptive Threshold**: `compute_atr_pct()` + per-bar `thr` 백테스트 ↔ `on_bar()` 라이브 ATR 계산 패리티
- [x] **v12.4 I-HybTight**: target_retrace=1.5, max_hold=25, trail_atr_mult=1.0, runner_max_hold_mult=3.0 백테스트 ↔ 라이브 동일 적용

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
| State Persistence 수정 | `state_persistence.py`, `scalper.py`, `test_state_persistence.py` |
| Discord 명령어 추가 | `bot.py`, `test_discord_bot.py` |
| 새 모듈/파일 추가 | `README.md` 프로젝트 구조, 해당 `__init__.py`, 테스트 파일 |

---

## 6. 핵심 상수 & 기본값 참조

### 6.1 VolStraddle 전략 파라미터

> Source of truth: `quantbox/strategy/vol_straddle/config.py` (`VolStraddleConfig`)

| 파라미터 | 기본값 | 용도 |
|---------|--------|------|
| `vr_short` | 8 | VR 단기 ATR 기간 |
| `vr_long` | 56 | VR 장기 ATR 기간 |
| `d_vr_threshold` | 0.50 | D_mom8 진입 VR 임계값 |
| `d_sl` | 0.01 (1%) | D 손절 |
| `d_tp` | 0.12 (12%) | D 하드 TP |
| `d_trail_activate` | 0.02 (+2%) | 트레일링 활성화 수익 |
| `d_trail_distance` | 0.003 (0.3%) | 트레일링 거리 |
| `s_vr_threshold` | 0.55 | S_d8 스트래들 VR 임계값 |
| `s_squeeze_dur` | 4 bars | 스트래들 최소 지속 기간 |
| `s_sl` | 0.015 (1.5%) | 스트래들 손절 (per leg) |
| `s_tp` | 0.05 (5%) | 스트래들 TP (per leg) |
| `momentum_period` | 8 | 방향 결정 모멘텀 lookback |
| `max_hold_15m` | 192 bars (48h) | 최대 보유 (15m TF) |
| `max_hold_1h` | 48 bars (48h) | 최대 보유 (1h TF) |
| `max_hold_4h` | 12 bars (48h) | 최대 보유 (4h TF) |
| `base_leverage` | 30.0 | VR Half-Kelly 레버리지 |
| `max_concurrent` | 10 | 최대 동시 포지션 (VR L/N=3.0, MA L/N=2.1) |
| `max_drawdown_pct` | 0.10 (10%) | 세션 DD 서킷 브레이커 |
| `fee_rate` | 0.0002 (0.02%) | 메이커 수수료/side |
| `slippage_rate` | 0.0002 (0.02%) | 슬리피지/side |
| `dir_diversity_enabled` | True | 방향 다양성 (max_same_direction=2) |
| `dynamic_capital_enabled` | False | 동적 자본 배분 비활성 (마진 부족 방지) |

### 6.2 Alpha Engine 파라미터

| 파라미터 | 기본값 | 소스 | 용도 |
|---------|--------|------|------|
| `vol_filter` | True | `AlphaConfig` | 볼륨 확인 필터 |
| `vol_mult` | 1.5× | `AlphaConfig` | 평균 대비 최소 볼륨 배율 |
| `momentum_filter` | True | `AlphaConfig` | 모멘텀 방향 필터 |
| `adaptive_threshold` | True | `AlphaConfig` | ATR 기반 적응형 임계값 |
| `trailing_stop` | True | `AlphaConfig` | 동적 트레일링 스탑 |

### 6.3 Volatility Selector 파라미터

| 파라미터 | 기본값 | 소스 | 용도 |
|---------|--------|------|------|
| `top_n` | 5 | `VolSelectorConfig` | 선택 심볼 수 |
| `vol_lookback_bars` | 2880 (30일) | `VolSelectorConfig` | 변동성 계산 기간 |
| `rebalance_interval_bars` | 960 (10일) | `VolSelectorConfig` | 리밸런싱 주기 |

### 6.4 위험 관리 파라미터

| 파라미터 | 기본값 | 소스 | 용도 |
|---------|--------|------|------|
| `risk_per_trade` | 0.01 (1%) | `RubberBandConfig` | 트레이드당 리스크 |
| `max_concurrent` | 5 | `RubberBandConfig` | 최대 동시 포지션 |
| `max_drawdown_pct` | 0.10 (10%) | `RubberBandConfig` | 세션 DD 서킷 브레이커 |
| `fee_rate` | 0.0002 (0.02%) | `RubberBandConfig` | 메이커 수수료/side |
| `slippage_rate` | 0.0002 (0.02%) | `RubberBandConfig` | 슬리피지/side |
| `funding_rate_8h` | 0.0001 (0.01%) | `RubberBandConfig` | 8h 펀딩비 |

### 6.5 LimitOrderChaser 파라미터

| 파라미터 | 기본값 | 소스 | 용도 |
|---------|--------|------|------|
| `max_chase_ticks` | 10 | `ChaseConfig` | 지정가 추격 최대 횟수 |
| `chase_interval_seconds` | 1.0 | `ChaseConfig` | 추격 간격 |
| `timeout_seconds` | 30.0 | `ChaseConfig` | 추격 타임아웃 |
| `fallback_to_market` | True | `ChaseConfig` | 시장가 폴백 허용 |

---

## 7. 테스트 현황

| 테스트 파일 | 검증 대상 | 핵심 관점 |
|------------|----------|----------|
| `test_rubberband` | 러버밴드 신호 생성 | compute_signals 정확성, 진입/청산 로직, **stop_decay**, **DD-managed portfolio**, **MTF confirmation**, **MTF enhanced (agree discount + tighter stop)**, **entry_stop_mult 패리티 (3개)** |
| `test_mtf_confirmation` | MTF 확인 + 독립 4H | apply_mtf_confirmation, apply_mtf_leverage, compute_signals_4h_independent, on_bar_4h, update_htf_data (18개) |
| `test_alpha_engine` | Alpha Engine 멀티팩터 | 팩터 토글, 적응형 임계값, 트레일링 |
| `test_vol_selector` | 변동성 선택기 | 랭킹, 리밸런싱, 실현변동성 계산 |
| `test_scalper` | 라이브 스캘퍼 | 사이클, 포지션 관리, 상태 저장/복원 |
| `test_limit_chaser` | 주문 추격 | Paper/Live 체결, 시장가 폴백 |
| `test_events` | 이벤트 시스템 | 생성, FIFO, 핸들러 디스패치 |
| `test_discord_bot` | Discord Bot | 명령어 파싱, 큐, 러너 생명주기 |
| `test_discord_notifications` | Discord 알림 | 웹훅 페이로드, 색상 코딩, 에러 복원 |
| `test_settings` | 설정 검증 | 환경변수, 기본값, 별칭, 타입 에러 |
| `test_config` | Config | 생성, 검증, 직렬화 |
| `test_cache` | 캐시 | Memory LRU, Disk, 만료, Timeframed |
| `test_state_persistence` | 상태 영속화 | 직렬화, 원자적 쓰기, 라운드트립 |
| `test_dynamic_target` | Dynamic Target (v8.0) | 컨피던스별 타겟 스케일링, 파라미터 검증, 기각된 전략 확인 (10개) |
| `test_v9_confluence` | v9.0 S+R+M Confluence | Fibonacci Exit, MACD Histogram, Stochastic Cross, compute_signals_v9, 라이브 파이프라인 (20개) |
| `test_partial_exit` | 부분 청산 + R:R 레버리지 | Partial exit 신호, Break-even stop, R:R-gate, delta-based 비용, 라이브 패리티 (23개) |
| `test_hybtight` | I-HybTight 패리티 | target/stop/trail/runner 파라미터 검증 (15개) |
| `test_v123_parity` | v12.3 PBT 패리티 | Runner state restore, bracket repair guard, failed partial recovery |
| `test_v1231_hotfix` | v12.3.1 안전장치 | Cancelled order detection, WOULD_TRIGGER guard, runner BE repair |
| `test_v1232_safety` | v12.3.2 안전장치 | 추가 엣지케이스 커버 |
| `test_v1233_fixes` | v12.3.3 수정 | 인터사이클 동기화 수정 |
| `test_v1241_hotfix` | v12.4.1 Server TP PBT | TP 수량 50% 검증 (PBT mode) |
| `test_v1244_hotfix` | v12.4.4 SL ratchet | Stale _server_sl_pct 리셋 검증 (6개) |
| `test_v1245_hotfix` | v12.4.5 save_state | 파셜 클로즈 후 save_state 호출 검증 (3개) |
| `test_v1246_hotfix` | v12.4.6 HWM reset | 0 포지션 재시작 시 HWM/history 리셋 검증 (3개) |
| `test_backtest_live_parity` | 백테스트↔라이브 패리티 | 신호/비용/사이징 일치 |
| `test_btc_directional` | BTC 방향 정렬 | Risk-Off 모드 BTC 모멘텀 |
| `test_confidence` | 컨피던스 스코어링 | 멀티 증거 필터링 |
| `test_diagnostics` | 진단 로깅 | JSONL 진단 출력 |
| `test_dynamic_leverage` | 동적 레버리지 | Move-confidence 기반 |
| `test_regime_switch` | 레짐 스위치 | Risk-On/Off 전환 |
| `test_server_oco` | 서버 OCO 주문 | SL/TP 서버 주문 |
| `test_ghost_audit` | Ghost Fill 감사 | 고스트 포지션 감지 시 체결 상세 로깅 (4개) |
| `test_mtf_lookahead` | MTF 4h Lookahead 방지 | mid-bar/boundary/reversal 3가지 시나리오에서 완성된 바만 사용하는지 검증 (3개) |
| `test_vol_straddle` | VolStraddle 전략 전체 | Config R39-VR856 파라미터, VR/모멘텀/방향 신호, 리샘플링 lookahead 방지, on_bar 시그널, per-symbol dedup, SL/TP/max_hold 포지션 관리, 멀티TF 우선순위, 스칼퍼 인터페이스, 상태 직렬화 (34개) |
| `test_terminal_ui` | 터미널 UI | 실시간 상태 표시 |
| `test_elliott` | Elliott Wave | ZigZag, ABC, 품질 스코어 |

---

## 8. 에이전트 행동 규칙

### 8.1 코드 수정 전 필수 확인

1. **패리티 위반 금지**: 신호/비용/사이징 로직을 한쪽만 수정하면 안 된다. `scalper.py` ↔ `vectorized_backtest.py` 양쪽을 반드시 동기화.
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
| 🔴 높음 | 신호/청산 로직, 비용 계산, 포지션 사이징 변경 | 사용자 확인 요청 후 실행. 패리티 검증 필수 |
| ⛔ 금지 | 안전장치 제거, 실데이터 규칙 위반, 패리티 파괴 | 절대 실행 불가. 사용자에게 위험성 경고 |
