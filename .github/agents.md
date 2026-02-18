# QuantBox Agent Instructions

> 이 문서는 AI 에이전트가 QuantBox 코드베이스를 이해하고, 정확한 작업을 수행하기 위한 **시스템 아키텍처 맵** 겸 **행동 규범**입니다.
> `copilot-instructions.md`가 "무엇을 지켜야 하는가"(원칙)를 정의한다면, 이 문서는 "코드가 실제로 어떻게 동작하는가"(사실)를 기술합니다.

---

## 1. 시스템 개요

QuantBox v11.4는 **바이난스 USDT-M 선물** 위에서 동작하는 **멀티 전략 스켈핑 시스템**이다.
두 가지 전략을 지원한다:
1. **Rubber Band Scalper** (기본): 단일 자산 평균회귀 — 급등/급락 후 스냅백 포착 (15m)
2. **Elliott Wave Strategy** (v11.6): ZigZag ABC 패턴 + Fibonacci 검증 + Timing Alpha Filter (4h)
**v11.4: Adaptive Threshold + Leverage Optimization** — ATR 기반 적응형 임계값 (`clip(2.5 × ATR%(48), 1%, 6%)`). 18-variant walk-forward sweep + 14-variant leverage grid → 2.5x_f1 선택. OOS Return +5069%, MDD 19.0%, Calmar 266.6. MTF 레버리지 2.0→2.5× (실효 3.75×).
**v11.6: Elliott Wave Timing Alpha** — 97패턴 리서치 기반 3중 타이밍 필터 (impulse_accel≥0, corr_ratio≥0.20, w3w1≤2.5). Calmar 1.51→5.10, Sharpe 1.07→1.69. 파라미터 비감응 (-0.3~+0.3 동일 결과).
**v11.4: Elliott Wave Strategy** — ZigZag ABC 패턴 인식, 2단계 포지셔닝 (2% 탐색적 → 8% 확인), 50% 부분 익절 + 브레이크이븐 스탑, BTC 매크로 바이어스, Numba 가속 지표.
**v11.3: 1-Minute Polling Loop** — 15분 대기 대신 60초 간격 폴링으로 반응성 개선.
**v11.1: Critical Fixes + Pool Expansion** — 서버사이드 손절(STOP 주문), 레버리지 검증, Top-20 심볼 화이트리스트 (A/B validated: +130% return vs Top-15).
**v11.0: BTC Directional Alignment** — btc_directional_period=48 (12h BTC momentum).
**v10.0: Entry Optimization + Symbol Whitelist** — threshold 0.032 (3.2%), min_confidence=1 (MTF as gate), max_hold=20, stoch_k=3. 동적 자본 배분.
**v9.0: Fibonacci + S+R+M Confluence** — target 0.382, stop 0.786, MACD(12,26,14), Stochastic(3,3).
**v7.0: Multi-Timeframe Confirmation + Independent 4H Trading** — 4h SMA(12) 트렌드 필터 + 독립 4h 평균회귀 + MTF 레버리지 부스트.

| 구성 요소 | 역할 |
|-----------|------|
| **RubberBandScalper** (`scalper.py`) | 실거래 오케스트레이터. 15분 사이클로 신호 → 진입 → 모니터링 → 청산. `--strategy` CLI로 전략 선택 (rubberband/elliott). v11.1: 서버사이드 STOP 손절 + 레버리지 검증 |
| **RubberBandStrategy** (`rubberband.py`) | 기본 평균회귀 신호 생성기 (Numba 가속, 3.2%+ 급등/급락 포착) + Fibonacci Exit (v9.0) + MACD/Stochastic Confluence (v9.0) + MTF 확인 + 독립 4H 트레이딩 + Symbol Whitelist (v10.0) |
| **ElliottStrategy** (`elliott.py`) | Elliott Wave ABC 패턴 전략 (Numba 가속). ZigZag 피봇 → ABC Fibonacci 검증 → 탐색적/확인 2단계 포지셔닝 → 50% 부분 익절 → 브레이크이븐 → 트레일링 (3×ATR). BTC 매크로 바이어스 (200/500 EMA) |
| **AlphaEngine** (`alpha_engine.py`) | 멀티팩터 신호 엔진 (볼륨/모멘텀/ATR/트레일링스탑 필터 토글) |
| **VolatilitySelector** (`vol_selector.py`) | 30일 실현변동성 기준 Top-N 심볼 동적 선택 (10일 리밸런싱) |
| **VectorizedBacktest** (`vectorized_backtest.py`) | Numba 가속 백테스트, 비용/펀딩비 반영 |
| **LimitOrderChaser** (`limit_chaser.py`) | 메이커 지정가 추격 주문. 타임아웃 시 시장가 폴백 |
| **State Persistence** (`state_persistence.py`) | JSON 기반 상태 영속화. 원자적 쓰기. 크래시 복구 지원 |
| **Drawdown / Daily Loss Breaker** | 세션 HWM 10% / 당일 5% 초과 시 자동 포지션 청산 + 일시정지 |
| **Discord Bot/Notifier** (`bot.py`, `discord.py`) | 원격 제어(!status, !close all 등) 및 실시간 알림 |

### 1.1 프로덕션 전략 구성 (Walk-Forward 검증 완료)

```
전략:     RB_original (러버밴드, Confidence-Scored + MTF Confirmation v7.0 + Fibonacci + S+R+M Confluence v9.0 + Entry Optimization v10.0 + Critical Fixes v11.1 + Adaptive Threshold v11.4)
심볼선택: Top-20 whitelist (FILUSDT, PEPEUSDT, XRPUSDT, XLMUSDT, ADAUSDT,
                          FETUSDT, ETHUSDT, ARBUSDT, SHIBUSDT, LINKUSDT,
                          ATOMUSDT, OPUSDT, DOTUSDT, NEARUSDT, AAVEUSDT,
                          BCHUSDT, HBARUSDT, SOLUSDT, ICPUSDT, BNBUSDT)
파라미터: lookback=8, threshold=0.032 (adaptive fallback), target=0.382, stop=0.786, max_hold=20
         stop_decay=0.0
         entry_mode='confidence'
         min_confidence=1 (MTF is the real quality gate)
         btc_regime_period=96 (24h SMA for regime detection)
         asset_trend_period=192 (48h SMA for classic pullback)
         btc_directional_period=48 (12h BTC momentum for sympathy)
         base_leverage=1.5, leverage_scale=0.0, max_leverage=1.5 (flat 1.5x)
         # v11.4 Adaptive Threshold (2.5x_f1 variant, walk-forward validated)
         atr_period=48 (12h ATR lookback)
         atr_multiplier=2.5 (ATR × 2.5)
         threshold_floor=0.01 (1% minimum)
         threshold_cap=0.06 (6% maximum)
         partial_exit_enabled=false
         dynamic_capital_enabled=true (balance / max(1, n_active+1))
         # v9.0 Fibonacci Exit Levels
         target_retrace=0.382 (Fibonacci golden ratio, +29% Calmar vs 0.4)
         stop_mult=0.786 (Fibonacci deep correction, DD -8.5% vs 0.9)
         # v8.0 Dynamic Target (DISABLED in v9.0)
         dynamic_target_enabled=false
         target_conf3=0.45, target_conf4=0.50, target_conf5=0.55  # 유지되나 비활성
         # v9.0 MACD Histogram Evidence
         macd_enabled=true
         macd_fast=12, macd_slow=26, macd_signal_period=14
         # v9.0 Stochastic Cross Evidence
         stoch_enabled=true
         stoch_k_period=3, stoch_d_period=3  # v10.0: faster stochastic
         # v7.0 MTF Confirmation
         mtf_confirmation_enabled=true
         mtf_sma_period=12 (4h SMA, 2일)
         mtf_leverage_mult=2.5 (MTF 확인 시 실효 3.75x, v11.4 leverage grid validated)
         # v7.0 Independent 4H Rubber Band
         htf_independent_enabled=true
         htf_rb_lookback=3, htf_rb_threshold=0.10
         htf_rb_target_retrace=0.5, htf_rb_stop_mult=0.8
         htf_rb_max_hold=12, htf_rb_btc_sma_period=48
비용:     0.04%/side + 펀딩비

성과 (365일, v11.4 Adaptive Threshold 2.5x_f1, Top-20 유니버스, 5 slots):
  Return:    +5069.4% (OOS walk-forward)
  Max DD:    19.0%
  Calmar:    266.6
  Trades:    969
  Win Rate:  70.1%
  Profit Factor: 2.83
  Sharpe:    5.31
  Dynamic Capital: balance / max(1, n_active+1), 50% cap
  안전장치: 서버사이드 STOP 손절 (클라이언트 크래시 방어), 레버리지 검증
```

---

## 2. 라이브 트레이딩 전체 흐름 (RubberBandScalper._cycle)

```
┌─────────────────────────────────────────────────────────────────┐
│  0. balance = get_balance()                                     │
│  1. _check_risk_limits(balance)                                 │
│     → HWM 대비 DD ≥ 10% → _drawdown_halt (전체 청산)           │
│  2. If drawdown_halt: close ALL positions → return              │
│  3. BTC klines 다운로드 (regime filter용)                       │
│  3b. 4h klines 다운로드 (MTF confirmation + 독립 4H 트레이딩)   │
│  4. 기존 포지션 관리:                                            │
│     ∀ active position:                                          │
│       → update_position(symbol, price) → exit?                  │
│       → TARGET / STOP_LOSS / TRAILING / MAX_HOLD                │
│  5. 신규 진입 스캔 (_scan_entries):                              │
│     ∀ symbol in universe (not already held):                    │
│       → download_klines(symbol, full=True) → on_bar(symbol, close, btc, high, low) │
│       → MACD/Stochastic 활성화 시 compute_signals_v9() 디스패치 (v9.0) │
│       → MTF confirmation: 4h SMA(12) 트렌드 필터 적용 (v7.0)   │
│       → 신호 +1/-1 → candidates 수집 → move 크기 정렬          │
│       → MTF 확인 트레이드: 2.0x 레버리지 부스트 (v7.0)          │
│     ∀ symbol: on_bar_4h(symbol, close_4h, btc_4h) (v7.0)       │
│       → 독립 4H 러버밴드 신호 → 4h candidates 추가              │
│       → 상위 N개까지 _open_position()                           │
│  6. _save_state() — 원자적 JSON 저장                            │
│  7. 매 4사이클마다 _log_summary() → 시간별 요약                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 신호 생성 (RubberBandStrategy.on_bar)

단일 자산의 15m close/volume/taker_buy 배열과 BTC close, 그리고 선택적 high/low 배열을 받아 신호를 생성한다. MACD 또는 Stochastic 활성화 시 `compute_signals_v9()` (Numba 컴파일)로 디스패치하고, 그렇지 않으면 `compute_signals_confidence()` (Numba 컴파일)로 신호를 생성한다.

**v11.4 Adaptive Threshold:**
- `atr_period > 0` 이면 per-bar 적응형 임계값 활성화: `thr[i] = clip(atr_multiplier × ATR%(atr_period)[i], threshold_floor, threshold_cap)`
- ATR%는 EMA-smoothed ATR / close 비율 (`compute_atr_pct()` Numba 함수)
- 고변동 구간에서 자동으로 임계값 상승 → 노이즈 필터링, 저변동 구간에서 하락 → 민감도 증가
- `atr_period=0` 이면 기존 고정 threshold 사용 (backward compatible):

**Confidence-Scored Multi-Evidence Entry (v6.0):**

| BTC 레짐 | 감지 방법 | 알고리즘 | 진입 조건 |
|----------|----------|---------|----------|
| **Risk-On** (BTC > SMA96) | BTC price > SMA(96) | Classic Pullback | 자산 dip + 자산 uptrend (price > SMA192) + ≥min_confidence |
| **Risk-Off** (BTC < SMA96) | BTC price < SMA(96) | Sympathy Pullback | 자산 dip + BTC 단기 하락 (btc_mom < 0) + ≥min_confidence |

**컨피던스 증거 (5가지, 레짐별 임계값 다름):**
- +1: BTC 레짐 확인 (기본 1점)
- +1: RSI 과매도/과매수 (Risk-On: RSI<40, Risk-Off: RSI<35)
- +1: 거래량 > 1.5x 평균 (청산 캐스케이드 감지)
- +1: 테이커 매도 압력 (Risk-On: ratio<0.45, Risk-Off: ratio<0.42)
- +1: SMA 지지 근접 (Risk-On) / BTC 모멘텀 강도 (Risk-Off)
- +1: MACD 히스토그램 기울기 방향 일치 (v9.0, `macd_enabled=true`)
- +1: 스토캐스틱 %K/%D 크로스 방향 일치 (v9.0, `stoch_enabled=true`)
- **BTC crash gate**: BTC 15m close로 `check_regime_filter()` → 급변 시 차단
- 기존 포지션 보유 심볼은 중복 신호 무시

**v7.0 MTF Confirmation (apply_mtf_confirmation):**
- 4h SMA(12)로 15m 신호 필터링: long 시 close_4h > SMA_4h, short 시 close_4h < SMA_4h
- 역추세 트레이드 차단 → MDD 16.5% → 4.3% (-74%)
- MTF 확인된 트레이드에 `apply_mtf_leverage()` 적용 (2.0x 부스트, 실효 3.0x)

**v7.0 Independent 4H Rubber Band (compute_signals_4h_independent):**
- 4h 타임프레임에서 독립적으로 평균회귀 신호 생성
- lookback=3, threshold=0.10, target=0.5, stop=0.8, max_hold=12
- BTC SMA(48) 레짐 필터 적용
- `on_bar_4h()` → `update_htf_data()` → 4h 데이터 저장 및 신호 생성

### 2.2 청산 조건 (update_position)

| 조건 | 판정 | 비고 |
|------|------|------|
| **Target 도달** | unrealized PnL ≥ target_retrace × move_size | 38.2% 되돌림 기본 (Fibonacci golden ratio, v9.0) |
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

> **핵심 발견 (v6.0)**: Confidence-Scored Entry가 v5.0 대비 Calmar 8.76 (vs 8.03). min_confidence=2로 34개 저품질 트레이드 필터링 → 동일 DD(16.9%)에서 Sharpe 2.05, WR 63.3%. 모멘텀 전략 5종 테스트 결과 15m 크립토에서 모멘텀은 비효과적임을 확인. 졸업 레버리지(base=1.0, step=1.0, max=2.0) 적용 시 Calmar 9.99까지 상승 가능하나 DD 22.4%로 상승. 보수적 flat 1.5x가 라이브 트레이딩 기본값.

> **핵심 발견 (v7.0)**: MTF Confirmation(4h SMA12)이 v6.0 대비 MDD 16.5% → 4.3% (-74%), Calmar 32.42 (+275%), Sharpe 5.23 (+153%). 수익률은 -2.2% 소폭 감소하나 위험조정 수익 극적 개선. Walk-forward Q3 DD 16.5% → 3.0% (-82%). 독립 4H 트레이딩 단독으로 Ret +52.2%, DD 4.8%, Calmar 10.87. 15m+4h 하이브리드가 최적 구성.

> **핵심 발견 (v8.0)**: Dynamic Target — 컨피던스 ≥3일 때 타겟을 40%→45/50/55%로 확대하여 비대칭 R:R 확보. DD 불변(4.3%), Calmar +9.3% (27.71→30.29), Sharpe 안정. 10000% 대회 우승자의 "확실할 때 파도를 타라" 원칙 적용. **v9.0에서 비활성화** — 타이트한 Fib 스탑(0.786)에서 고정 0.382 타겟이 동적 스케일링보다 우수. **A/B 테스트 후 기각된 전략**: RSI Divergence, Volatility Squeeze, Reversal Confirmation — 15m 크립토에서 데이터가 지지하지 않음. 이 3가지를 재테스트하지 말 것.

> **핵심 발견 (v9.0)**: Fibonacci + S+R+M Confluence — target 0.4→0.382 (Fib golden ratio, +29% Calmar), stop 0.9→0.786 (Fib deep correction, DD -8.5%). R:R 0.444→0.486 (+9.5%). MACD(12,26,14) 히스토그램 기울기 증거 단독 +7.6% Calmar. Stochastic(5,3) 크로스 증거 (high/low 캔들 데이터 필요). `compute_signals_v9()` Numba 함수로 통합. **v9.0 A/B 테스트 후 기각**: EMA Regime(모든 기간 50/96/200) -4~-9% Calmar, Capital Preservation(50% 부분 청산) -32% Calmar, Confluence Gate(RSI+Stoch+MACD 동시 필수) -90% Calmar(트레이드 252→41). 이 3가지를 재테스트하지 말 것.

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
| `rubberband.py` | `RubberBandStrategy`, `RubberBandConfig`, `compute_signals`, `compute_signals_with_btc_directional`, `compute_signals_regime_switch`, `compute_signals_confidence`, `compute_signals_with_partial_exit`, `compute_signals_partial_exit_btc_directional`, `compute_rr_gated_leverage`, `apply_rr_gated_leverage`, `apply_mtf_confirmation`, `apply_mtf_leverage`, `compute_signals_4h_independent`, `_compute_ema_inline`, `_compute_stochastic_inline`, `_compute_macd_inline`, `compute_signals_v9`, `compute_atr_pct` | 기본 평균회귀 신호 + BTC 레짐 스위치 + 컨피던스 필터링 + BTC 방향 정렬 + 부분 청산 + R:R-Gated 레버리지 + **MTF 확인 (v7.0)** + **독립 4H 트레이딩 (v7.0)** + **Fibonacci Exit + MACD/Stochastic Confluence (v9.0)** + **Adaptive Threshold (v11.4)** (Numba 컴파일) |
| `elliott.py` | `ElliottConfig`, `ElliottStrategy`, `compute_elliott_signals`, `compute_ema`, `compute_rsi`, `compute_macd`, `compute_stochastic`, `compute_atr`, `zigzag_pivots`, `detect_abc_patterns`, `detect_bullish_divergence`, `detect_bearish_divergence`, `check_macd_reversal`, `check_stoch_cross`, `detect_ending_diagonal`, `detect_triangle` | Elliott Wave ABC 패턴 전략 (Numba 가속). ZigZag + Fibonacci + 다이버전스/MACD/스토캐스틱 확인 + 웨지/삼각 감지 + 2단계 포지셔닝 + 자본 보존 익절 |
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
- [x] **v7.0 MTF Confirmation**: `apply_mtf_confirmation()` 백테스트 ↔ `_scan_entries()` 라이브 4h SMA 필터 패리티
- [x] **v7.0 MTF Leverage**: `apply_mtf_leverage()` 백테스트 ↔ 라이브 MTF 확인 시 2.0x 레버리지 부스트 패리티
- [x] **v7.0 Independent 4H**: `compute_signals_4h_independent()` 백테스트 ↔ `on_bar_4h()` 라이브 4h 신호 패리티
- [x] **v7.0 Live Fix**: volume/taker 데이터 `_scan_entries()` 연결, 4h kline 다운로드, `main()` v7.0 설정 전달
- [x] **v9.0 Fibonacci Exit**: `target_retrace=0.382`, `stop_mult=0.786` 백테스트 ↔ 라이브 동일 적용
- [x] **v9.0 MACD/Stochastic**: `compute_signals_v9()` 백테스트 ↔ `on_bar()` 라이브 v9 디스패치 패리티
- [x] **v9.0 Live Pipeline**: `download_klines(full=True)` 5-tuple (close, high, low, volume, taker_buy) 반환
- [x] **v11.4 Adaptive Threshold**: `compute_atr_pct()` + per-bar `thr` 백테스트 ↔ `on_bar()` 라이브 ATR 계산 패리티
- [x] **v11.4 Leverage**: `mtf_leverage_mult=2.5` (실효 3.75×) 백테스트 ↔ 라이브 동일 적용

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
| `target_retrace` | 0.382 (38.2%) | `RubberBandConfig` | 이익 실현 목표 (Fibonacci golden ratio, v9.0) |
| `stop_mult` | 0.786 | `RubberBandConfig` | 손절 배수 (무브 크기 대비, Fibonacci deep correction, v9.0) |
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
| `btc_directional_period` | 48 (12h) | `RubberBandConfig` | BTC 모멘텀 lookback (봉 수) |
| `entry_mode` | 'confidence' | `RubberBandConfig` | 진입 모드: confidence / regime_switch / btc_directional / raw |
| `btc_regime_period` | 96 (24h) | `RubberBandConfig` | BTC SMA period for regime detection |
| `asset_trend_period` | 192 (48h) | `RubberBandConfig` | Asset SMA period for classic pullback |
| `min_confidence` | 2 | `RubberBandConfig` | 최소 증거 점수 (1-5, 2=최적) |
| `confidence_leverage_step` | 0.0 | `RubberBandConfig` | 컨피던스당 추가 레버리지 (0=고정) |
| `rsi_period` | 14 | `RubberBandConfig` | RSI 계산 기간 |
| `vol_period` | 20 | `RubberBandConfig` | 거래량 SMA 기간 |
| `btc_regime_period` | 96 (24h) | `RubberBandConfig` | BTC SMA period for regime detection |
| `asset_trend_period` | 192 (48h) | `RubberBandConfig` | Asset SMA period for classic pullback |
| `partial_exit_enabled` | False | `RubberBandConfig` | 부분 청산 비활성화 (v4.4) |
| `partial_exit_fraction` | 0.5 (50%) | `RubberBandConfig` | 부분 청산 비율 |
| `rr_leverage_enabled` | True | `RubberBandConfig` | R:R-Gated 레버리지 활성화 |
| `min_rr_for_leverage` | 0.30 | `RubberBandConfig` | 레버리지 허용 최소 R:R 값 |
| `mtf_confirmation_enabled` | True | `RubberBandConfig` | **v7.0** 4h 트렌드 확인 활성화 |
| `mtf_sma_period` | 12 (2일) | `RubberBandConfig` | **v7.0** 4h SMA 기간 |
| `mtf_leverage_mult` | 2.5 | `RubberBandConfig` | **v7.0** MTF 확인 시 레버리지 배수 (실효 3.75x, v11.4 leverage grid) |
| `htf_independent_enabled` | True | `RubberBandConfig` | **v7.0** 독립 4시간 트레이딩 활성화 |
| `htf_rb_lookback` | 3 (12h) | `RubberBandConfig` | **v7.0** 4h lookback |
| `htf_rb_threshold` | 0.10 (10%) | `RubberBandConfig` | **v7.0** 4h 최소 무브 크기 |
| `htf_rb_target_retrace` | 0.5 (50%) | `RubberBandConfig` | **v7.0** 4h 되돌림 목표 |
| `htf_rb_stop_mult` | 0.8 | `RubberBandConfig` | **v7.0** 4h 손절 배수 |
| `htf_rb_max_hold` | 12 (2일) | `RubberBandConfig` | **v7.0** 4h 최대 보유 기간 |
| `htf_rb_btc_sma_period` | 48 (8일) | `RubberBandConfig` | **v7.0** 4h BTC SMA 기간 |
| `dynamic_target_enabled` | False | `RubberBandConfig` | 비활성화 (v9.0: Fib 0.382 고정 > 동적 스케일링) |
| `macd_enabled` | True | `RubberBandConfig` | **v9.0** MACD 히스토그램 컨피던스 증거 활성화 |
| `macd_fast` | 12 | `RubberBandConfig` | **v9.0** MACD fast EMA 기간 |
| `macd_slow` | 26 | `RubberBandConfig` | **v9.0** MACD slow EMA 기간 |
| `macd_signal_period` | 14 | `RubberBandConfig` | **v9.0** MACD signal EMA 기간 |
| `stoch_enabled` | True | `RubberBandConfig` | **v9.0** 스토캐스틱 크로스 컨피던스 증거 활성화 |
| `stoch_k_period` | 5 | `RubberBandConfig` | **v9.0** 스토캐스틱 %K 기간 (very fast) |
| `stoch_d_period` | 3 | `RubberBandConfig` | **v9.0** 스토캐스틱 %D 기간 |
| `atr_period` | 48 (12h) | `RubberBandConfig` | **v11.4** 적응형 임계값 ATR lookback (0=비활성, 고정 threshold 사용) |
| `atr_multiplier` | 2.5 | `RubberBandConfig` | **v11.4** ATR × 배수 (walk-forward sweep 최적) |
| `threshold_floor` | 0.01 (1%) | `RubberBandConfig` | **v11.4** 적응형 임계값 하한 |
| `threshold_cap` | 0.06 (6%) | `RubberBandConfig` | **v11.4** 적응형 임계값 상한 |

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

## 7. 테스트 현황 (30+파일, 540개)

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
