# QuantBox Agent Instructions

> 이 문서는 AI 에이전트가 QuantBox 코드베이스를 이해하고, 정확한 작업을 수행하기 위한 **시스템 아키텍처 맵** 겸 **행동 규범**입니다.
> `copilot-instructions.md`가 "무엇을 지켜야 하는가"(원칙)를 정의한다면, 이 문서는 "코드가 실제로 어떻게 동작하는가"(사실)를 기술합니다.

---

## 1. 시스템 개요

QuantBox v5.0은 **바이낸스 USDT-M 선물** 위에서 동작하는 **Rubber Band 스캘핑 시스템**이다.
단일 자산 평균회귀 전략으로, 급등/급락 후 스냅백(반등)을 포착하여 수익을 낸다.
**v5.0: Regime-Switch Dual Entry** — BTC 레짐에 따라 다른 진입 알고리즘을 사용한다.

| 구성 요소 | 역할 |
|-----------|------|
| **RubberBandScalper** (`scalper.py`) | 실거래 오케스트레이터. 15분 사이클로 신호 → 진입 → 모니터링 → 청산 |
| **RubberBandStrategy** (`rubberband.py`) | 기본 평균회귀 신호 생성기 (Numba 가속, 4.5%+ 급등/급락 포착) |
| **AlphaEngine** (`alpha_engine.py`) | 멀티팩터 신호 엔진 (볼륨/모멘텀/ATR/트레일링스탑 필터 토글) |
| **VolatilitySelector** (`vol_selector.py`) | 30일 실현변동성 기준 Top-N 심볼 동적 선택 (10일 리밸런싱) |
| **VectorizedBacktest** (`vectorized_backtest.py`) | Numba 가속 백테스트, 비용/펀딩비 반영 |
| **LimitOrderChaser** (`limit_chaser.py`) | 메이커 지정가 추격 주문. 타임아웃 시 시장가 폴백 |
| **State Persistence** (`state_persistence.py`) | JSON 기반 상태 영속화. 원자적 쓰기. 크래시 복구 지원 |
| **Drawdown / Daily Loss Breaker** | 세션 HWM 10% / 당일 5% 초과 시 자동 포지션 청산 + 일시정지 |
| **Discord Bot/Notifier** (`bot.py`, `discord.py`) | 원격 제어(!status, !close all 등) 및 실시간 알림 |

### 1.1 프로덕션 전략 구성 (Walk-Forward 검증 완료)

```
전략:     RB_original (러버밴드, Regime-Switch Dual Entry v5.0)
심볼선택: Top-10 curated (FETUSDT, FILUSDT, ICPUSDT, HBARUSDT, DOTUSDT,
                        OPUSDT, SHIBUSDT, DOGEUSDT, LTCUSDT, ATOMUSDT)
파라미터: lookback=8, threshold=0.045, target=0.4, stop=0.9, max_hold=10
         stop_decay=0.0, htf_period=0 (MTF DISABLED)
         entry_mode='regime_switch'
         btc_regime_period=96 (24h SMA for regime detection)
         asset_trend_period=192 (48h SMA for classic pullback)
         btc_directional_period=32 (8h BTC momentum for sympathy)
         base_leverage=1.5, leverage_scale=0.0, max_leverage=1.5 (flat 1.5x)
         partial_exit_enabled=false
비용:     0.04%/side + 펀딩비

성과 (365일, 10개 심볼 유니버스):
  Return:    +116.8%
  Sharpe:    1.89
  Max DD:    17.4%
  Calmar:    6.70
  Win Rate:  61.7%
  R:R:       1.25
  Trades:    832
  Rolling Windows: 13/13 profitable (100%)
```

---

## 2. 라이브 트레이딩 전체 흐름 (RubberBandScalper._cycle)

```
┌─────────────────────────────────────────────────────────────────┐
│  0. balance = get_balance()                                     │
│  1. _check_risk_limits(balance)                                 │
│     → HWM 대비 DD ≥ 10% → _drawdown_halt (전체 청산)           │
│     → 당일 손실 ≥ 5% → _daily_halt (신규 진입 차단)             │
│  2. If drawdown_halt: close ALL positions → return              │
│  3. BTC klines 다운로드 (regime filter용)                       │
│  4. 기존 포지션 관리:                                            │
│     ∀ active position:                                          │
│       → update_position(symbol, price) → exit?                  │
│       → TARGET / STOP_LOSS / TRAILING / MAX_HOLD                │
│  5. 신규 진입 스캔 (_scan_entries):                              │
│     ∀ symbol in universe (not already held):                    │
│       → download_klines(symbol) → on_bar(symbol, close, btc)   │
│       → 신호 +1/-1 → candidates 수집 → move 크기 정렬          │
│       → 상위 N개까지 _open_position()                           │
│  6. _save_state() — 원자적 JSON 저장                            │
│  7. 매 4사이클마다 _log_summary() → 시간별 요약                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 신호 생성 (RubberBandStrategy.on_bar)

단일 자산의 15m close 배열과 BTC close를 받아 `compute_signals_regime_switch()` (Numba 컴파일)로 신호를 생성한다:

**Regime-Switch Dual Entry (v5.0):**

| BTC 레짐 | 감지 방법 | 알고리즘 | 진입 조건 |
|----------|----------|---------|----------|
| **Risk-On** (BTC > SMA96) | BTC price > SMA(96) | Classic Pullback | 자산 dip + 자산 uptrend (price > SMA192) |
| **Risk-Off** (BTC < SMA96) | BTC price < SMA(96) | Sympathy Pullback | 자산 dip + BTC 단기 하락 (btc_mom < 0) |

- **Risk-On**: BTC 상승장 → 자산이 자체 상승추세에서 조정 → 매수 (클래식 풀백)
- **Risk-Off**: BTC 하락장 → BTC와 동반 하락하는 자산 → 매수 (동조 풀백, v4.4과 동일)
- **BTC crash gate**: BTC 15m close로 `check_regime_filter()` → 급변 시 차단
- 기존 포지션 보유 심볼은 중복 신호 무시

### 2.2 청산 조건 (update_position)

| 조건 | 판정 | 비고 |
|------|------|------|
| **Target 도달** | unrealized PnL ≥ target_retrace × move_size | 40% 되돌림 기본 |
| **Stop Loss** | unrealized PnL < -stop_mult × move_size × time_factor | time_factor = 1 - stop_decay × (bars_held / max_hold) |
| **Trailing Stop** | best_pnl > 0 && PnL < best_pnl × (1 - trail_pct) | 이익 확보 후 역전 방어 |
| **Max Hold** | bars_held ≥ max_hold | 타임아웃 강제 청산 (8 bars = 2h) |
| **Partial Close (v4.3, 비활성)** | gain > 0 && not partial_closed | v4.4에서 비활성화 (tighter stop이 DD 억제) |
| **Break-even Stop (v4.3, 비활성)** | partial_closed && gain ≤ 0 | v4.4에서 비활성화 |

### 2.3 주문 실행 (LimitOrderChaser)

```
1. get_best_price() → BBO (best bid/ask)
2. 지정가 POST_ONLY 주문 제출
3. chase_interval(1초)마다 체결 확인
4. 미체결 시 취소 → 새 BBO로 재제출 (최대 10회)
5. timeout(30초) 초과 시 → 시장가 폴백
6. 시장가 폴백 후 5회까지 체결 검증 (ACK 응답 대응)
```

### 2.4 포지션 사이징 (calculate_position_size + R:R-Gated Leverage)

```python
per_slot_capital = balance / max_concurrent   # 균등 배분 (5슬롯)

# ── R:R-Gated Leverage (v4.3) ──
# 손익비(R:R)가 충분히 높을 때만 레버리지 적용
if rr_leverage_enabled:
    expected_gain = move_size * target_retrace - 2 * cost_per_side
    expected_loss = move_size * stop_mult + 2 * cost_per_side
    rr = expected_gain / expected_loss
    if rr < min_rr_for_leverage:  # 기본 0.30
        leverage = 1.0  # 레버리지 비활성화
    else:
        excess = (move_size - threshold) / threshold
        leverage = base_leverage + leverage_scale * max(excess, 0)
        leverage = min(leverage, max_leverage)
    notional = per_slot_capital × leverage
else:
    # Fallback: 리스크 역산 방식
    stop_distance = move_size × stop_mult
    risk_amount = per_slot_capital × risk_per_trade
    position_size = risk_amount / stop_distance
    leverage = min(position_size / per_slot_capital, max_leverage)
    notional = per_slot_capital × leverage
```

> **핵심 발견 (v5.0)**: Regime-Switch Dual Entry가 단일 BTC directional 대비 Calmar 6.70 (vs 4.98). 두 레짐에서 각각 최적의 알고리즘을 사용하여 832 trades, WR 61.7%, R:R 1.25로 안정적 수익. 레버리지는 flat 1.5x가 최적 (2.5x에서 Calmar 6.94지만 리스크 상승).

### 2.5 비용 모델

| 비용 항목 | 값 | 비고 |
|-----------|---|------|
| **수수료** | 0.02%/side (메이커) | LimitChaser로 메이커 체결 목표 |
| **슬리피지** | 0.02%/side | 보수적 추정 |
| **펀딩비** | 0.01%/8h | 15m 바 기준 1/32 |
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
| `rubberband.py` | `RubberBandStrategy`, `RubberBandConfig`, `compute_signals`, `compute_signals_with_btc_directional`, `compute_signals_regime_switch`, `compute_signals_with_partial_exit`, `compute_signals_partial_exit_btc_directional`, `compute_rr_gated_leverage`, `apply_rr_gated_leverage` | 기본 평균회귀 신호 + BTC 레짐 스위치 + BTC 방향 정렬 + 부분 청산 + R:R-Gated 레버리지 (Numba 컴파일) |
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

### 3.6 `quantbox/` (최상위)

| 파일 | 역할 |
|------|------|
| `scalper.py` | `RubberBandScalper` 라이브 트레이더. CLI 진입점 (`python -m quantbox.scalper`) |

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
- [ ] Daily Loss Limit — 당일 잔고 대비 5% 손실 시 신규 진입 차단
- [ ] BTC Crash Gate — regime filter로 극단적 시장 상황 시 진입 차단
- [ ] Max Concurrent — 동시 포지션 수 제한 (기본 5개)
- [ ] Risk per Trade — 트레이드당 1% 리스크 사이징

### 4.3 포지션 관리
- [ ] Target/Stop/Trailing/MaxHold 4중 청산 조건
- [ ] Partial Close (v4.3, v4.4 비활성) — 부분 청산 기능 존재하나 v4.4에서 비활성화
- [ ] 청산 실패 시 포지션 재등록 → 다음 사이클 재시도
- [ ] 동일 심볼 중복 진입 방지 (`has_position()`)
- [ ] max_hold 강제 청산 (타임아웃)

### 4.4 상태 영속화 & 복구
- [ ] `_save_scalper_state()` — tmpfile → os.replace 원자적 쓰기
- [ ] `_load_scalper_state()` — 시작 시 크래시 전 상태 복원
- [ ] 포지션 + 잔고 + 거래 히스토리 직렬화
- [ ] 로그 파일 로테이션 — RotatingFileHandler 50MB/5백업

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

---

## 5. 파일 수정 시 영향 범위 매트릭스

코드를 수정할 때 반드시 영향받는 파일을 함께 확인/수정해야 한다:

| 수정 대상 | 반드시 함께 확인 |
|----------|-----------------|
| `settings.py` 파라미터 추가 | `.env.example`, `.env`, `MANUAL.md` |
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

### 6.1 Rubber Band 전략 파라미터

| 파라미터 | 기본값 | 소스 | 용도 |
|---------|--------|------|------|
| `lookback` | 8 bars (2h) | `RubberBandConfig` | 급등/급락 측정 기간 |
| `threshold` | 0.045 (4.5%) | `RubberBandConfig` | 최소 무브 크기 |
| `target_retrace` | 0.4 (40%) | `RubberBandConfig` | 이익 실현 목표 |
| `stop_mult` | 0.9 | `RubberBandConfig` | 손절 배수 (무브 크기 대비) |
| `max_hold` | 10 bars (2.5h) | `RubberBandConfig` | 최대 보유 기간 |
| `stop_decay` | 0.0 | `RubberBandConfig` | 시간감쇠 계수 (0=비활성) |
| `htf_period` | 0 (비활성) | `RubberBandConfig` | MTF SMA lookback (0=비활성) |
| `htf_block_threshold` | 0.01 (1%) | `RubberBandConfig` | HTF 트렌드 차단 임계값 |
| `htf_agree_discount` | 0.85 | `RubberBandConfig` | HTF 동의 시 threshold 배수 (15% 낮춤) |
| `htf_agree_stop_mult` | 0.95 | `RubberBandConfig` | HTF 동의 시 stop 배수 (5% 타이트) |
| `dynamic_leverage_enabled` | True | `RubberBandConfig` | Move-confidence 동적 레버리지 활성화 |
| `base_leverage` | 1.5 | `RubberBandConfig` | flat 1.5x 레버리지 |
| `leverage_scale` | 0.0 | `RubberBandConfig` | 컨피던스 스케일링 비활성화 |
| `max_leverage` | 1.5 | `RubberBandConfig` | 레버리지 안전 상한 |
| `btc_directional_enabled` | True | `RubberBandConfig` | BTC 방향 정렬 필터 활성화 (Risk-Off에서 사용) |
| `btc_directional_period` | 32 (8h) | `RubberBandConfig` | BTC 모멘텀 lookback (봉 수) |
| `entry_mode` | 'regime_switch' | `RubberBandConfig` | 진입 모드: regime_switch / btc_directional / raw |
| `btc_regime_period` | 96 (24h) | `RubberBandConfig` | BTC SMA period for regime detection |
| `asset_trend_period` | 192 (48h) | `RubberBandConfig` | Asset SMA period for classic pullback |
| `partial_exit_enabled` | False | `RubberBandConfig` | 부분 청산 비활성화 (v4.4) |
| `partial_exit_fraction` | 0.5 (50%) | `RubberBandConfig` | 부분 청산 비율 |
| `rr_leverage_enabled` | True | `RubberBandConfig` | R:R-Gated 레버리지 활성화 |
| `min_rr_for_leverage` | 0.30 | `RubberBandConfig` | 레버리지 허용 최소 R:R 값 |

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
| `daily_loss_limit_pct` | 0.05 (5%) | `RubberBandConfig` | 일일 손실 제한 |
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

## 7. 테스트 현황 (30+파일, 352개)

| 테스트 파일 | 검증 대상 | 핵심 관점 |
|------------|----------|----------|
| `test_rubberband` | 러버밴드 신호 생성 | compute_signals 정확성, 진입/청산 로직, **stop_decay**, **DD-managed portfolio**, **MTF confirmation**, **MTF enhanced (agree discount + tighter stop)**, **entry_stop_mult 패리티 (3개)** |
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
| `test_partial_exit` | 부분 청산 + R:R 레버리지 | Partial exit 신호, Break-even stop, R:R-gate, delta-based 비용, 라이브 패리티 (23개) |

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
