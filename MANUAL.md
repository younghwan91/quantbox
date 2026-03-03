# QuantBox 실행 매뉴얼 📈

**암호화폐 트레이딩 프레임워크 완벽 가이드 — Rubber Band Scalper + Elliott Wave + VolStraddle**

> 버전: 13.0  
> 최종 업데이트: 2026년 3월 2일

---

## 📋 목차

1. [개요](#1-개요)
2. [설치 및 환경 설정](#2-설치-및-환경-설정)
3. [🔥 Rubber Band Scalper (v12.6)](#3-rubber-band-scalper)
3.5. [🌊 Elliott Wave Strategy (v11.6)](#35-elliott-wave-strategy)
3.6. [🎯 VolStraddle (A1-D50)](#36-volstraddle)
4. [아키텍처 이해](#4-아키텍처-이해)
5. [핵심 모듈 상세](#5-핵심-모듈-상세)
6. [데이터 관리](#6-데이터-관리)
7. [페어 선택 시스템](#7-페어-선택-시스템)
8. [전략 구현](#8-전략-구현)
9. [백테스팅](#9-백테스팅)
10. [파라미터 최적화](#10-파라미터-최적화)
11. [Walk-Forward 분석](#11-walk-forward-분석)
12. [실거래 연동](#12-실거래-연동)
13. [Discord Bot (원격 제어)](#13-discord-bot-원격-제어)
14. [성과 지표 해석](#14-성과-지표-해석)
15. [고급 사용법](#15-고급-사용법)
16. [트러블슈팅](#16-트러블슈팅)
17. [API 레퍼런스](#17-api-레퍼런스)

---

## 1. 개요

### 1.1 QuantBox란?

QuantBox는 암호화폐 트레이딩을 위한 종합 퀀트 프레임워크입니다. **Rubber Band Scalper**(15m 단일 자산 평균회귀)와 **Elliott Wave Strategy**(4h ABC 패턴) 두 가지 전략을 지원하며, 백테스팅과 실거래에서 동일한 코드를 사용합니다.

---

## 3. 🔥 Rubber Band Scalper (v12.6 — ATR Stop + Climax Filter)

### 3.1 전략 개요

Rubber Band Scalper는 **단일 자산 평균회귀** 전략입니다. ATR 기반 적응형 임계값으로 시장 변동성에 실시간 대응합니다.

**v12.6 핵심 개선 (R21, 2026-02-24):**
- **ATR-Based Stop**: `SL = clamp(2.0 × ATR_pct, 0.5%, 4.0%)` — 고정 0.382× 대신 적응형 스탑. Ghost-SL (노이즈 정도 파동에 의한 조기 스탑아웃) 해결.
- **Climax Volume Filter**: `vol > 3.0 × vol_SMA(20)` = +1 confidence. 극단적 거래량 폭발 시에만 진입 바이어스.
- **Stoch/MACD 비활성화**: 노이즈 필터 제거, 컨피던스 체계 단순화.
- **결과 (6yr, 50 심볼)**: Calmar **1,171** (+60.8%), DD **5.8%** (−24.7%), WR **62.2%** (+11.1pp), Sharpe **6.37**.

**v12.4 핵심 개선:**
- **Adaptive Threshold**: 고정 3.2% → `clip(2.5 × ATR%(48), 1%, 6%)` 적응형 (OOS Sharpe 8.23, OOS/IS 1.09)
  - Walk-forward 검증: 18개 변형 × 20개 심볼, IS/OOS 분리, 모든 변형에서 OOS/IS ≥ 1.04
  - 변동성 높은 시장에서는 임계값 자동 확대, 낮은 시장에서는 타이트하게 조정
  - 레버리지 최적화: 1×~5× 그리드 → 현행 3.75× 유지 결정
- **Server-Side Stop-Loss**: 바이낸스 Algo Order API로 클라이언트 크래시 방어
  - **v11.4.2 Bracket Repair**: 복원된 포지션의 누락된 SL/TP를 자동 복구
- **v12.6 ATR-Based Stop**: `SL = clamp(2.0 × ATR_pct, 0.5%, 4.0%)` — Ghost-SL 제거
- **v12.6 Climax Volume Filter**: vol > 3.0× vol_SMA(20) = +1 conf (R21 champion: Calmar 1171, DD 5.8%)
- **v12.6**: Stoch/MACD 비활성화 (RB_MACD_ENABLED=false, RB_STOCH_ENABLED=false)
- **v12.4 I-HybTight Exit**: target 1.5, stop 0.382, trail ATR×1.0, runner 3× max_hold (R:R 1.91, Calmar 834)
- **v12.4.1–v12.4.3 Server TP PBT audit**: 6 call-sites corrected to use 50% qty in PBT mode
- **v12.4.4**: Stale `_server_sl_pct` reset after runner SL re-placed at breakeven
- **v12.4.5**: Missing `_save_state()` after successful partial close
- **v12.4.6**: HWM + trade history reset on restart with 0 positions (prevents phantom DD halt)
- **v9.0 S+R+M Confluence**: MACD(12,26,14) + Stochastic(3,3) 컨피던스 증거
- **v7.0 MTF Confirmation**: 4h SMA(12) 트렌드 필터 — MDD -74%, Calmar +275%

**알파 소스:**
- 청산 캐스케이드(liquidation cascades)로 인한 오버슈트
- 스톱 헌팅 + 강제 청산 후 자연적 반등
- 2시간(8×15m 봉) 내 38.2% 피보나치 되돌림
- MTF 4h 트렌드 확인 시 역추세 진입 차단

**검증 결과 (Top-50 E-3Slot, 1095일, 15m 데이터, v12.2):**
- Portfolio Return: **+3384.3%**
- Sharpe: 5.02
- Max Drawdown: 21.8%
- Calmar: **155.22**
- Win Rate: 69.7%, Profit Factor: 2.66
- Trades: 936

### 3.2 실행 방법

```bash
# Paper trading (시뮬레이션, 실제 주문 없음)
python -m quantbox.scalper --paper

# 한 사이클만 실행 후 종료
python -m quantbox.scalper --paper --once

# 커스텀 파라미터
python -m quantbox.scalper --paper --lookback 10 --threshold 0.03 --max-slots 3

# LIVE (실거래, 주의!)
python -m quantbox.scalper
```

### 3.3 전략 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| lookback | 8 | 이동 측정 기간 (봉 수, 8×15m = 2시간) |
| threshold | 0.032 | ATR NaN 시 고정 임계값 폴백 (3.2%) |
| **atr_period** | **48** | **v11.4: ATR EMA 기간 (48 × 15분 = 12시간)** |
| **atr_multiplier** | **2.0** | **v12.2: ATR 배수 (threshold = 2.0 × ATR%, E-3Slot A/B validated)** |
| **threshold_floor** | **0.008** | **v12.2: 최소 임계값 (0.8%, E-3Slot A/B validated)** |
| **threshold_cap** | **0.06** | **v11.4: 최대 임계값 (6%)** |
| target_retrace | 1.5 | v12.4: 이익 실현 — 이동 폭의 150% 되돌림 (I-HybTight champion) |
| stop_mult | 0.382 | v12.3: 손절 기본값 (포지션 없을 때 폴백). **v12.6 ATR Stop 활성화 시 이 값은 무시됨.** |
| max_hold | 25 | v12.4: 최대 보유 기간 (25봉 = 6.25시간, I-HybTight patience) |
| **atr_stop_enabled** | **True** | **v12.6: ATR 기반 적응형 스탑 활성화** |
| **atr_stop_mult** | **2.0** | **v12.6: ATR 배수 (SL = 2.0 × ATR_pct)** |
| **atr_stop_floor** | **0.005** | **v12.6: 최소 스탑 거리 0.5%** |
| **atr_stop_cap** | **0.04** | **v12.6: 최대 스탑 거리 4.0%** |
| **climax_enabled** | **True** | **v12.6: Climax 거래량 필터 활성화** |
| **climax_vol_mult** | **3.0** | **v12.6: Climax 임계값 (vol > 3.0 × vol_SMA20 = +1 conf)** |
| stop_decay | 0.0 | 시간감쇠 비활성화 (장기 안정성 확보) |
| htf_period | 0 | MTF SMA 기간 (0=비활성화 — 장기 성능 저하) |
| htf_block_threshold | 0.01 | HTF 트렌드 차단 임계값 (1%, MTF 비활성 시 미사용) |
| htf_agree_discount | 0.85 | HTF 동의 시 진입 임계값 할인 (MTF 비활성 시 미사용) |
| htf_agree_stop_mult | 0.95 | HTF 동의 시 스톱 배수 (MTF 비활성 시 미사용) |
| max_concurrent | 2 | v12.3: M-2SSTSX 2슬롯 집중 |
| min_confidence | 3 | v12.3: 최소 증거 점수 3 (quality gate) |
| **partial_exit** | **True** | **v12.3: PBT 50% 부분 익절 활성화** |
| **trail_atr_mult** | **1.0** | **v12.4: 러너 ATR 트레일링 배수 (tighter trail locks gains faster)** |
| **runner_max_hold_mult** | **3.0** | **v12.4: 러너 최대 보유 배수 (max_hold × 3.0 = 75봉)** |
| **breakeven_after_partial** | **True** | **v12.3: 파셜 후 BE 스톱 + 러너** |
| **steep_cap_curve** | **True** | **v12.3: SteepX 컨피던스 사이징 (0.3/0.5/1.5/2.0×)** |
| **soft_equity_scale** | **True** | **v12.3: SoftTight DD 스케일링 (2/4/6%DD)** |
| **vol_adjusted_size** | **True** | **v12.3: 변동성 조정 사이징** |
| **dir_diversity** | **True** | **v12.3: 방향 다양성 필터 (max 2 same-dir)** |

### 3.4 안전장치

- **Max Drawdown Breaker**: 세션 HWM 대비 10% DD → 전체 청산
- **Daily Loss Limit**: 당일 잔고 대비 5% 손실 → 신규 진입 중단
- **Risk-Sized Positions**: 1% risk/trade, 최대 레버리지 20x
- **LimitOrderChaser**: maker-only 체결로 수수료 0.02% (taker 0.05% 대비 60% 절감)
- **Atomic State Persistence**: 크래시 후 자동 복구
- **v12.3 Runner State Restore**: 봇 재시작 시 러너 trail_floor/highest_since_tp1 보존
- **v12.3 Bracket Repair Guard**: 러너 포지션에 서버 TP 재배치 방지 (즉시 킬 방지)
- **v12.3 Failed Partial Recovery**: 파셜 클로즈 실패 시 stale order ID 제거 → 브래킷 리페어 강제
- **v12.3.1 Cancelled Order Detection**: 외부 취소/만료 알고 오더 감지 → 브래킷 리페어 자동 트리거
- **v12.3.1 WOULD_TRIGGER Guard**: SL/TP 센티넬 값 저장 방지 + 재배치 시 force close
- **v12.3.1 Runner BE Repair**: 러너 브래킷 리페어 시 breakeven stop(0.0001) 사용
- **v12.3.1 State Persistence**: 서버 필/인터사이클 싱크 후 즉시 save_state
- **Discord 알림**: 진입/청산, 드로다운 경고 전송

### 3.5 백테스트

```python
from quantbox.strategy.rubberband import compute_signals
from quantbox.backtest.vectorized_backtest import VectorizedBacktest, BacktestConfig

bt = VectorizedBacktest(BacktestConfig(
    initial_capital=10000,
    cost_per_side=0.0004,  # 0.04% per side
))
signals = compute_signals(close, 8, 0.025, 0.5, 1.0, 8)
result = bt.run_single(close, signals, "BTCUSDT")
print(f"Return: {result.total_return_pct:+.2f}%, Sharpe: {result.sharpe:.2f}")
```

### 3.5 Elliott Wave Strategy (v11.6)

Elliott Wave ABC 패턴 + 5차원 품질 스코어링 + Timing Alpha Filter + Risk-Sized Leverage 기반 전략.

#### v11.6 핵심 개선:
- **Timing Alpha Filter**: 97패턴 리서치 기반 3중 타이밍 게이트 (accel≥0, corr≥0.20, W3/W1≤2.5)
  - `impulse_accel`: W3/W1 속도 가속도 (감속 임펄스 제거 → 패자 40% 필터)
  - `corr_impulse_ratio`: ABC 정정 시간 / 임펄스 시간 (너무 짧은 정정 제거)
  - `max_w3_w1_time_ratio`: W3/W1 바 비율 (지연된 W3 제거)
  - 파라미터 비감응: accel -0.3~+0.3 동일 결과 → 과적합 아님

#### v11.5 핵심 개선:
- **Quality Scoring**: W2/W3/B retrace, C:A ratio, 구조적 규칙 5차원 0-100점 스코어. 임계값 40점 이하 패턴 제거
- **Risk-Sized Leverage**: `lev = min(2.0, risk_per_trade / stop_dist)` — 트레이드당 최대 2% 위험으로 일관된 리스크
- **Tight Stops at C**: C 완료 지점 ±0.5% — 예측이 틀리면 즉시 소규모 손절, 맞으면 높은 R:R
- **adj_close**: 스탑/TP 체결 시 인트라바 가격으로 PnL 계산 (close-to-close 대신)
- **Tighter Fibonacci Ranges**: W3 extension min 1.272, C/A 0.786-2.000, C Fib tolerance 0.10
- **Risk Filter**: entry-to-stop 거리 최대 3% 제한

#### 백테스트 결과 (365일, 18심볼, timing alpha, best_refined):
| 지표 | v11.5 | v11.6 (timing) |
|------|-------|----------------|
| Active Symbols | 8 | 12 |
| Total Trades | 8 | 21 |
| Portfolio Return | +4.84% | **+5.61%** |
| Portfolio MaxDD | 1.96% | 1.10% |
| Sharpe Ratio | 1.11 | **1.69** |
| Calmar Ratio | 2.47 | **5.10** |

#### 실행 방법

```bash
# Paper trading
python -m quantbox.scalper --strategy elliott --paper

# Live trading
python -m quantbox.scalper --strategy elliott
```

#### 전략 개요

1. **ZigZag 피봇 감지**: 4h 캔들에서 High/Low ZigZag 피봇 추출 (threshold 3%/1.5%)
2. **ABC 패턴 인식**: 5개 피봇 시퀀스에서 Fibonacci 비율 검증
   - B wave: 0.40–0.72 retracement (0.618 금 영역)
   - C/A: 0.80–1.80 ratio (1.0–1.618 이상적)
3. **2단계 포지셔닝**:
   - 탐색적 진입 (2% 자본) → 4봉 내 확인 시 8% 추가 (RSI 다이버전스/MACD 반전/스토캐스틱 크로스)
   - 미확인 시 탐색적 포지션 자동 종료
4. **자본 보존 익절**: 50% 부분 익절 → 브레이크이븐 스탑 → 트레일링 스탑 (3×ATR)
5. **BTC 매크로 바이어스**: EMA(200)/EMA(500) 기반 Long/Short/Neutral 필터링
6. **엔딩 다이애고널/트라이앵글 감지**: 웨지/삼각 패턴 식별 → 조기 청산 또는 진입 억제

#### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| zigzag_threshold | 0.02 | ZigZag 피봇 최소 변동 (2%, sweep-optimized: 0.02 > 0.03) |
| zigzag_threshold_minor | 0.015 | 소 패턴 ZigZag (1.5%) |
| ema_periods | [20,50,100,200,500] | EMA 기간 세트 |
| rsi_period / rsi_fast | 14 / 9 | RSI 기간 (표준/다이나믹) |
| macd_fast/slow/signal | 12/26/9 | MACD 파라미터 |
| stoch_k/stoch_d | 14/3 | Stochastic 파라미터 |
| b_retrace_min/max | 0.40/0.72 | B wave Fibonacci 리트레이스 범위 |
| ca_ratio_min/max | 0.80/1.80 | C/A 비율 범위 |
| exploratory_pct | 0.02 | 탐색적 진입 자본 (2%) |
| primary_pct | 0.08 | 확인 후 추가 자본 (8%) |
| partial_tp_pct | 0.50 | 부분 익절 비율 (50%) |
| nominal_leverage | 20 | 공칭 레버리지 |
| max_margin_pct | 0.10 | 총 자본 대비 최대 마진 (10%) |
| confirmation_bars | 4 | 확인 대기 봉 수 |
| max_hold_bars | 96 | 최대 보유 (4h×96 = 16일) |
| pyramid_enabled | True | 미실현 이익 피라미딩 |
| sl_buffer_pct | 0.005 | 스탑 버퍼 (C 지점 ±0.5%) |
| max_entry_risk_pct | 0.03 | 최대 entry-to-stop 거리 (3%) |
| min_pattern_quality | 40.0 | 최소 패턴 품질 점수 (0-100) |
| risk_per_trade | 0.02 | 트레이드당 자기자본 위험 (2%) |
| risk_sized_leverage | True | 스탑 거리 기반 레버리지 자동 조정 |
| trail_atr_mult | 1.5 | 트레일링 스탑 ATR 배수 (sweep-optimized: 1.5 > 2.0) |

#### 백테스트

```python
from quantbox.strategy.elliott import ElliottConfig, compute_elliott_signals
import numpy as np

# 4h OHLCV 데이터
config = ElliottConfig()
signals, adj_close = compute_elliott_signals(close_4h, high_4h, low_4h, btc_close)
# signals: leverage-weighted direction (-2.0 ~ +2.0), 0 = flat
# adj_close: fill-price adjusted close for accurate PnL
```

### 3.6 v7.0: Multi-Timeframe Confirmation + Independent 4H Trading

v7.0은 두 가지 핵심 업그레이드를 도입합니다:

#### 3.6.1 MTF Confirmation (4h 트렌드 확인)

4h SMA(12)로 15m 진입을 필터링합니다. 역추세 트레이드를 차단하여 MDD를 극적으로 줄입니다.

| 지표 | v6.0 (15m only) | v7.0 hybrid | 변화 |
|------|-----------------|-------------|------|
| Return | +143.0% | +139.8% | -2.2% |
| MDD | 16.5% | 4.3% | **-73.9%** |
| Calmar | 8.65 | 32.42 | **+274.8%** |
| Sharpe | 2.07 | 5.23 | **+153%** |

- Walk-forward Q3 (worst quarter) DD: 16.5% → 3.0% (-82%)
- `apply_mtf_confirmation()` — Numba 컴파일, 4h SMA 방향과 15m 신호 비교
- `apply_mtf_leverage()` — MTF 확인된 트레이드에 2.0x 레버리지 부스트 (실효 3.0x)

#### 3.6.2 Independent 4H Rubber Band

4h 타임프레임에서 독립적으로 평균회귀 트레이딩합니다. 15m과 별개로 동작합니다.

- 4h-only 성과: Ret +52.2%, DD 4.8%, Calmar 10.87
- `compute_signals_4h_independent()` — Numba 컴파일 4h 신호 생성
- `RubberBandStrategy.on_bar_4h()` — 4h 봉 데이터 시 신호 생성
- `RubberBandStrategy.update_htf_data()` — 4h 데이터 저장

### 3.7 v8.0: Dynamic Target (v9.0에서 비활성화)

v8.0은 컨피던스 점수에 따라 이익 실현 타겟을 동적으로 확대합니다. **v9.0에서 비활성화됨** — 타이트한 Fib 스탑(0.786)에서 고정 0.382 타겟이 동적 스케일링보다 우수합니다. conf3/4/5 필드는 유지되나 비활성 상태.

**원리**: 10000% 대회 우승자의 "확실할 때 파도를 타라" 전략에서 영감

| 컨피던스 | 타겟 | 설명 |
|----------|------|------|
| ≤ 2 | 40% (기본) | 기존과 동일 |
| 3 | 45% | 3개 증거 확인 시 |
| 4 | 50% | 4개 증거 확인 시 |
| 5 | 55% | 전 증거 확인 시 |

**A/B 검증 결과:**
- DD 불변: 4.3% (동일)
- Calmar +9.3%: 27.71 → 30.29
- Sharpe 안정

**테스트 후 기각된 전략:**
- RSI Divergence — 데이터가 지지하지 않음
- Volatility Squeeze — 데이터가 지지하지 않음
- Reversal Confirmation — 데이터가 지지하지 않음

#### 3.7.1 Dynamic Target 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| RB_DYNAMIC_TARGET_ENABLED | false | 비활성화 (v9.0: Fib 0.382 고정 > 동적 스케일링) |
| RB_TARGET_CONF3 | 0.45 | conf=3 타겟 (45%) — v9.0에서 비활성 |
| RB_TARGET_CONF4 | 0.50 | conf=4 타겟 (50%) — v9.0에서 비활성 |
| RB_TARGET_CONF5 | 0.55 | conf=5 타겟 (55%) — v9.0에서 비활성 |

#### 3.6.3 MTF 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| RB_MTF_CONFIRMATION_ENABLED | true | 4h 트렌드 확인 활성화 |
| RB_MTF_SMA_PERIOD | 12 | 4h SMA 기간 (12봉 = 2일) |
| RB_MTF_LEVERAGE_MULT | 2.0 | MTF 확인 후 레버리지 배수 (실효 3.0x) |
| RB_HTF_INDEPENDENT_ENABLED | true | 독립 4시간 트레이딩 활성화 |
| RB_HTF_RB_LOOKBACK | 3 | 4h lookback (12시간) |
| RB_HTF_RB_THRESHOLD | 0.10 | 10% 임계값 |
| RB_HTF_RB_TARGET_RETRACE | 0.5 | 50% 되돌림 목표 |
| RB_HTF_RB_STOP_MULT | 0.8 | 0.8x 손절 |
| RB_HTF_RB_MAX_HOLD | 12 | 12봉 (2일) |
| RB_HTF_RB_BTC_SMA_PERIOD | 48 | BTC SMA (8일) |

### 3.8 v9.0: Fibonacci + S+R+M Confluence

v9.0은 피보나치 기반 정밀 Exit Level과 MACD/Stochastic 컨플루언스 증거를 도입합니다. v8.0 대비 Calmar +33.6% (11.50→15.38), DD -8.5% (12.0%→11.0%).

#### 3.8.1 Fibonacci Exit Levels

기존 arbitrary 0.4/0.9 값을 피보나치 수열에서 파생된 값으로 교체:

| 파라미터 | v8.0 | v9.0 | 변화 |
|----------|------|------|------|
| target_retrace | 0.4 | **0.382** (Fibonacci golden ratio) | R:R개선, +29% Calmar |
| stop_mult | 0.9 | **0.786** (Fibonacci deep correction) | DD -8.5%, 타이트 스탑 |
| R:R | 0.444 | **0.486** | +9.5% |

#### 3.8.2 MACD Histogram Evidence

> ⚠️ **v12.6에서 비활성화** (`RB_MACD_ENABLED=false`). 노이즈 제거 — R21 검증에서 Stoch/MACD 제거 후 WR +11.1pp, Calmar +60.8%.

MACD 히스토그램 기울기가 트레이드 방향과 일치하면 +1 컨피던스를 추가합니다. 단독으로 +7.6% Calmar 개선.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| RB_MACD_ENABLED | **false** | **v12.6: 비활성화** (노이즈 제거) |
| RB_MACD_FAST | 12 | MACD fast EMA 기간 |
| RB_MACD_SLOW | 26 | MACD slow EMA 기간 |
| RB_MACD_SIGNAL_PERIOD | 14 | MACD signal EMA 기간 (14=최적) |

#### 3.8.3 Stochastic Cross Evidence

> ⚠️ **v12.6에서 비활성화** (`RB_STOCH_ENABLED=false`). R21 검증에서 MACD와 함께 제거 후 성능 대폭 개선.

스토캐스틱 %K가 %D를 트레이드 방향으로 크로스하면 +1 컨피던스를 추가합니다. Very fast 설정(3,3)은 스캘핑에 최적화 (v10.0: 5→3, +5% Calmar). high/low 캔들 데이터를 요구하며 라이브 파이프라인에 추가됨.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| RB_STOCH_ENABLED | **false** | **v12.6: 비활성화** (노이즈 제거) |
| RB_STOCH_K_PERIOD | 3 | v10.0: 스토캐스틱 %K 기간 (3=ultra fast, +5% Calmar vs 5) |
| RB_STOCH_D_PERIOD | 3 | 스토캐스틱 %D 기간 |
| **RB_CLIMAX_ENABLED** | **true** | **v12.6: Climax 거래량 필터 (vol > N× vol_SMA20 = +1 conf)** |
| **RB_CLIMAX_VOL_MULT** | **3.0** | **v12.6: Climax 임계값 (3.0 = R21 최적, Calmar 1171)** |
| **RB_ATR_STOP_ENABLED** | **true** | **v12.6: ATR 기반 적응형 스탑 활성화** |
| **RB_ATR_STOP_MULT** | **2.0** | **v12.6: ATR 배수 (SL = 2.0 × ATR_pct)** |
| **RB_ATR_STOP_FLOOR** | **0.005** | **v12.6: 최소 SL 거리 0.5% (소진 노이즈 Ghost-SL 방지)** |
| **RB_ATR_STOP_CAP** | **0.04** | **v12.6: 최대 SL 거리 4.0% (고변동성 기에서 과다한 노숨 방지)** |
| **RB_ATR_PERIOD** | **48** | **v11.4: ATR EMA 기간 (48 × 15분 = 12시간)** |
| **RB_ATR_MULTIPLIER** | **2.0** | **v12.2: ATR 배수 (threshold = 2.0 × ATR%, E-3Slot A/B)** |
| **RB_THRESHOLD_FLOOR** | **0.008** | **v12.2: 최소 임계값 (0.8%)** |
| **RB_THRESHOLD_CAP** | **0.06** | **v11.4: 최대 임계값 (6%)** |

#### 3.8.4 Live Pipeline 확장 (v9.0)

- `download_klines(full=True)`: 5-tuple 반환 (close, high, low, volume, taker_buy)
- `update_market_data()`: optional high/low 배열 수용
- `on_bar()`: MACD 또는 Stochastic 활성화 시 `compute_signals_v9()` 디스패치

#### 3.8.5 v9.0 Numba 함수

| 함수 | 반환 | 설명 |
|------|------|------|
| `_compute_ema_inline(arr, period)` | EMA array | SMA warm-up 포함 EMA 계산 |
| `_compute_stochastic_inline(high, low, close, k, d)` | (K, D) | 스토캐스틱 오실레이터 |
| `_compute_macd_inline(close, fast, slow, signal)` | (line, signal, histogram) | MACD 전체 |
| `compute_signals_v9(...)` | signals array | MACD/Stochastic 증거 포함 v9 신호 함수 |

#### 3.8.6 A/B 테스트 기각 기록 (v9.0 — DO NOT RE-TEST)

| 전략 | 결과 | 판정 |
|------|------|------|
| EMA Regime (50/96/200 모든 기간) | -4% ~ -9% Calmar | ❌ REJECTED |
| Capital Preservation (50% 부분 청산) | -32% Calmar | ❌ REJECTED |
| Confluence Gate (RSI+Stoch+MACD 동시 필수) | -90% Calmar (트레이드 252→41) | ❌ REJECTED |

> ⚠️ 위 3가지 전략은 v9.0 컨텍스트에서 철저한 A/B 테스트를 거쳐 기각됨. 재테스트 금지.

#### 3.6.4 Live Parity Fix (v7.0)

- volume/taker 데이터를 scalper의 `_scan_entries()`에 연결 (기존 누락 수정)
- 4h kline 다운로드 추가
- `main()`에서 v6.0/v7.0/v9.0 모든 설정을 전달하도록 수정

---

### 1.2 페어 트레이딩 개념

페어 트레이딩은 **공적분(Cointegration)** 관계에 있는 두 자산의 스프레드가 평균으로 회귀하는 성질을 이용하는 **시장 중립 전략**입니다.

```
스프레드 = log(자산1) - β × log(자산2)

β (헤지 비율): 두 자산 간의 관계를 나타내는 계수
```

**진입 조건:**
- Z-Score > +2: 스프레드 숏 (자산1 매도, 자산2 매수)
- Z-Score < -2: 스프레드 롱 (자산1 매수, 자산2 매도)

**청산 조건:**
- Z-Score가 0 근처로 회귀할 때

### 1.3 주요 기능

| 기능 | 설명 |
|------|------|
| 이벤트 기반 엔진 | 백테스트/실거래 통합 |
| 2단계 캐싱 | 메모리(LRU) + 디스크(gzip) |
| 필터 파이프라인 | 조합 가능한 페어 선별 필터 |
| 칼만 필터 | 동적 헤지 비율 추정 |
| 다양한 최적화 | Grid, Random, DE, Bayesian |
| Walk-Forward | Out-of-Sample 검증 |
| Numba 가속 | 성능 핵심 계산 JIT 컴파일 |

### 1.4 ⚠️ 중요: 데이터 정책

> **모든 백테스팅과 분석은 반드시 실제 시장 데이터만 사용합니다.**
> 
> - ❌ 합성 데이터(synthetic data) 사용 금지
> - ❌ 가짜 데이터(fake data) 생성 금지
> - ❌ 더미 데이터(dummy data) 테스트 금지
> - ✅ 바이낸스 선물 실제 가격 데이터만 사용

**실제 데이터 로딩 방법:**
```python
from quantbox.data.real_data_loader import RealDataLoader

# 실제 데이터만 로드하는 공식 방법
loader = RealDataLoader()
data = loader.load_data(
    symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    min_bars=500,
    align_data=True
)

# 사용 가능한 심볼 확인
symbols = loader.get_available_symbols()
```

이 정책은 백테스트 결과의 신뢰성을 보장하기 위함입니다.

---

## 2. 설치 및 환경 설정

### 2.1 시스템 요구사항

- Python 3.13 이상
- RAM: 8GB 이상 권장
- 디스크: 최소 5GB (데이터 캐시용)

### 2.2 설치

```bash
# 저장소 클론
git clone https://github.com/younghwan91/quantbox.git
cd quantbox

# 개발 모드 설치 (권장)
pip install -e .

# 또는 의존성만 설치
pip install -r requirements.txt
```

### 2.3 의존성 패키지

```toml
# pyproject.toml에 정의된 의존성
dependencies = [
    "cryptography>=46.0.1",      # API 보안
    "matplotlib>=3.10.6",        # 시각화
    "numba>=0.62.0",             # JIT 가속
    "pandas>=2.3.2",             # 데이터 처리
    "pykalman>=0.10.2",          # 칼만 필터
    "python-binance>=1.0.29",    # 바이낸스 API
    "requests>=2.32.5",          # HTTP 요청
    "statsmodels>=0.14.5",       # 통계 모델
    "ta>=0.11.0",                # 기술적 지표
    "tqdm>=4.67.1",              # 진행 표시
]
```

### 2.4 바이낸스 API 설정

```python
# config.json 생성
{
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "testnet": true  # 테스트넷 사용 시
}
```

**API 키 발급 방법:**
1. 바이낸스 계정 로그인
2. API Management → Create API
3. 스팟 거래/선물 거래 권한 활성화
4. IP 화이트리스트 설정 (권장)

### 2.5 디렉토리 구조 생성

```bash
# 자동 생성됨 (Config 초기화 시)
mkdir -p data/cache data/raw data/processed
```

### 2.6 환경변수 설정 (.env)

실거래/페이퍼 트레이딩 시 `.env` 파일에 아래 환경변수를 설정합니다:

```bash
# 필수: 바이낸스 API
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true          # 테스트넷 사용 시

# 레버리지 및 포지션 사이징
LEVERAGE=2                     # Fallback 레버리지 (동적 레버리지 미사용 시)
TARGET_R_PCT=0.02              # 트레이드당 목표 리스크 (밸런스의 2%, 0=동적 비활성)
MAX_LEVERAGE=125               # 동적 레버리지 안전 상한
MAX_TRADERS=10                 # 최대 동시 트레이더 수 (기본값: 10)
POSITION_SIZE_PCT=-1           # 슬롯당 자본 비율 (-1=자동: 1/MAX_TRADERS)

# 전략 파라미터
ENTRY_ZSCORE=2.0               # 진입 z-score 임계값
EXIT_ZSCORE=0.5                # 청산 z-score 임계값 (dynamic_exit 비활성 시)
STOP_LOSS_ZSCORE=3.5           # 손절 z-score
MAX_HOLDING_HOURS=20           # 최대 보유 시간 (초과 시 강제 청산, 백테스트 80bars@15m과 동일)
FORMATION_HOURS=60             # 페어 선별에 사용할 과거 데이터 길이 (60h = 240 bars @ 15m)

# Dynamic Exit: half-life 기반 청산 z-score 자동 조절
# USE_DYNAMIC_EXIT=true         # 활성화 시 EXIT_ZSCORE 대신 half-life 기반 동적 값 사용
# DYNAMIC_EXIT_BASE=0.3         # 빠른 페어 exit z-score (half_life=min)
# DYNAMIC_EXIT_RANGE=0.7        # 느린 페어까지의 범위 (max exit = base + range = 1.0)

# Regime Protection
Z_SCORE_CAP=4.0                # |z| > cap 이면 진입 거부 (레짐 변화 방어)
VOL_EXPANSION_GATE=2.0         # recent_std / formation_std > gate 이면 진입 거부

# Dynamic Leverage (v4.4: flat 1.5x, 365일 A/B 검증)
RB_DYNAMIC_LEVERAGE_ENABLED=true  # 레버리지 파이프라인 활성화
RB_BASE_LEVERAGE=1.5              # 전 트레이드 1.5x 고정 (최적 sweet spot)
RB_LEVERAGE_SCALE=0.0             # 컨피던스 스케일링 비활성화 (고정 레버리지)
RB_MAX_LEVERAGE=1.5               # base와 동일 (안전 상한)

# Regime-Switch + Confidence Scoring (v6.0, 365일 A/B 검증)
RB_ENTRY_MODE=confidence           # 진입 모드: confidence | regime_switch | btc_directional | raw
RB_BTC_REGIME_PERIOD=96            # BTC SMA period for regime detection (96봉 = 24시간)
RB_ASSET_TREND_PERIOD=192          # Asset SMA period for classic pullback (192봉 = 48시간)

# Confidence Scoring (v12.2 E-3Slot, filter relaxation A/B validated)
RB_MIN_CONFIDENCE=2                # v12.2: E-3Slot 최소 증거 점수 2 (Sharpe 6.41, Calmar 795)
RB_CONFIDENCE_LEVERAGE_STEP=0.0    # 컨피던스당 추가 레버리지 (0=고정, 1.0=moderate)
RB_RSI_PERIOD=14                   # RSI 계산 기간
RB_VOL_PERIOD=20                   # 거래량 SMA 기간 (스파이크 감지)

# v7.0 Multi-Timeframe Confirmation (365일 A/B 검증)
# 4h SMA(12) 트렌드 필터로 15m 역추세 진입 차단
# MDD 16.5% → 4.3% (-74%), Calmar 8.65 → 32.42 (+275%), Sharpe 2.07 → 5.23 (+153%)
RB_MTF_CONFIRMATION_ENABLED=true   # 4h 트렌드 확인 활성화
RB_MTF_SMA_PERIOD=12               # 4h SMA 기간 (12봉 = 2일)
RB_MTF_LEVERAGE_MULT=2.0           # MTF 확인 후 레버리지 배수 (2.0x → 실효 3.0x)

# v8.0 Dynamic Target (DISABLED in v9.0 — Fib discipline outperforms)
# 피보나치 0.382 고정 목표가 동적 스케일링보다 우수. 타이트한 스탑(0.786)에 빠른 TP 필요.
RB_DYNAMIC_TARGET_ENABLED=false    # 비활성화: Fib 0.382 고정 > 동적 스케일링
RB_TARGET_CONF3=0.45               # conf=3 타겟 (사용 안 함)
RB_TARGET_CONF4=0.50               # conf=4 타겟 (사용 안 함)
RB_TARGET_CONF5=0.55               # conf≥5 타겟 (사용 안 함)

# v9.0 S+R+M Confluence Pipeline (A/B validated, Calmar +33.6%)
# MACD 히스토그램 반전 + 스토캐스틱 %K/%D 크로스 = 추가 컨피던스 증거
RB_MACD_ENABLED=true               # MACD 히스토그램 기울기 컨피던스 증거
RB_MACD_FAST=12                    # MACD fast EMA 기간
RB_MACD_SLOW=26                    # MACD slow EMA 기간
RB_MACD_SIGNAL_PERIOD=14           # MACD signal EMA 기간 (14=최적)
RB_STOCH_ENABLED=true              # 스토캐스틱 크로스 컨피던스 증거
RB_STOCH_K_PERIOD=3                # v10.0: 스토캐스틱 %K 기간 (3=ultra fast, +5% Calmar vs 5)
RB_STOCH_D_PERIOD=3                # 스토캐스틱 %D 기간

# v11.4 Adaptive Threshold (Walk-Forward Validated)
RB_ATR_PERIOD=48                   # ATR EMA 기간 (48 × 15분 = 12시간)
RB_ATR_MULTIPLIER=2.0              # v12.2: ATR 배수 (2.5→2.0, E-3Slot A/B validated)
RB_THRESHOLD_FLOOR=0.008           # v12.2: 최소 임계값 (0.8%)
RB_THRESHOLD_CAP=0.06              # 최대 임계값 (6%)

# v7.0 Independent 4H Rubber Band
# 4h 타임프레임에서 독립적으로 평균회귀 트레이딩
# 4h-only: Ret +52.2%, DD 4.8%, Calmar 10.87
RB_HTF_INDEPENDENT_ENABLED=true    # 독립 4시간 트레이딩 활성화
RB_HTF_RB_LOOKBACK=3               # 4h lookback (12시간)
RB_HTF_RB_THRESHOLD=0.10           # 10% 임계값
RB_HTF_RB_TARGET_RETRACE=0.5       # 50% 되돌림 목표
RB_HTF_RB_STOP_MULT=0.8            # 0.8x 손절
RB_HTF_RB_MAX_HOLD=12              # 12봉 (2일)
RB_HTF_RB_BTC_SMA_PERIOD=48        # BTC SMA (8일)

# BTC Directional Alignment (v4.2, Risk-Off 모드에서 사용)
RB_BTC_DIRECTIONAL_ENABLED=true   # BTC 방향 정렬: Long→BTC하락, Short→BTC상승 시만 진입
RB_BTC_DIRECTIONAL_PERIOD=48      # BTC 모멘텀 lookback (48봉 = 12시간)

# v12.3 PBT Exit (Partial → Breakeven → Trailing Runner)
RB_PARTIAL_EXIT_ENABLED=true      # v12.3: PBT 활성화 (50% partial → BE floor → ATR trailing)
RB_PARTIAL_EXIT_FRACTION=0.5      # 부분 청산 비율 (0.5 = 50%)
RB_TRAIL_ATR_MULT=1.0             # v12.4: 러너 ATR 트레일링 배수 (1.0×ATR, tighter trail)
RB_BREAKEVEN_AFTER_PARTIAL=true   # 파셈 후 스톱을 브레이크이븐으로 이동
RB_RUNNER_MAX_HOLD_MULT=3.0       # v12.4: 러너 최대 보유 = max_hold × 3.0 (75봉)

# R:R-Gated Leverage (v4.4, 365일 A/B 검증)
RB_RR_LEVERAGE_ENABLED=true       # R:R 기준 레버리지 게이트: 손익비 높을 때만 레버리지
RB_MIN_RR_FOR_LEVERAGE=0.30       # 레버리지 허용 최소 R:R 비율

# 드로다운 제한
MAX_DRAWDOWN_PCT=0.10          # 세션 HWM 대비 최대 허용 드로다운 (10%)

# 비용 모델
TRADING_FEE=0.0005             # 거래 수수료 (0.05%)
SLIPPAGE=0.001                 # 슬리피지 (0.10%)
FUNDING_RATE_8H=0.0001         # 8시간 펀딩비 (0.01%)

# 유니버스 & 페이퍼 트레이딩
N_COINS=50                     # v12.2: E-3Slot 상위 50개 코인 스캔
PAPER_BALANCE=1000             # 페이퍼 트레이딩 시 가상 잔고 ($)

# Symbol Whitelist (50 symbols, 2026-02-24 업데이트)
# MATICUSDT/FTMUSDT/MKRUSDT/EOSUSDT → 상장폐지로 ENAUSDT/TAOUSDT/WLDUSDT/WIFUSDT로 교체
RB_SYMBOL_WHITELIST=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,XLMUSDT,DOTUSDT,1000PEPEUSDT,ARBUSDT,FETUSDT,1000SHIBUSDT,OPUSDT,ATOMUSDT,HBARUSDT,ICPUSDT,FILUSDT,TRXUSDT,SUIUSDT,TONUSDT,LTCUSDT,BCHUSDT,UNIUSDT,NEARUSDT,APTUSDT,AAVEUSDT,INJUSDT,ETCUSDT,GRTUSDT,STXUSDT,EGLDUSDT,AXSUSDT,SANDUSDT,THETAUSDT,ALGOUSDT,RUNEUSDT,DYDXUSDT,CRVUSDT,LDOUSDT,COMPUSDT,SNXUSDT,VETUSDT,APEUSDT,ENAUSDT,TAOUSDT,WLDUSDT,WIFUSDT
RB_MAX_CONCURRENT=2            # v12.3: M-2SSTSX (2 slots)

# 알림 (Discord Webhook - 단방향 알림)
DISCORD_WEBHOOK_URL=           # 미설정 시 알림 비활성화

# Discord Bot (양방향 원격 명령)
DISCORD_BOT_TOKEN=             # Bot 토큰 (미설정 시 봇 비활성화)
DISCORD_CHANNEL_ID=            # 명령 수신 채널 ID (숫자)
```

> **팁**: `.env.example` 파일을 복사하여 시작하세요: `cp .env.example .env`

---

## 3.6 🎯 VolStraddle — VR Compression Detector (R40-VR856-3x)

### 3.6.1 전략 개요

VolStraddle은 **변동성 압축(Volatility Squeeze)**을 감지하여 폭발적 움직임의 방향을 포착하는 전략입니다.

**핵심 원리:**
- **VR (Volatility Ratio)** = ATR(8) / ATR(56) — 단기 변동성 / 장기 변동성
- VR < 0.50 = 변동성 압축 감지 → 폭발 임박
- **Momentum(8)** 방향으로 진입 (양수 → LONG, 음수 → SHORT)
- **멀티 타임프레임**: 15m / 1h / 4h 독립 스캔, 높은 TF 우선

### 3.6.2 파라미터 (v13.0 Aggressive, 6yr 검증)

**v13.0 검증 결과:** N=2,265 @25x leverage ×10 slots (L/N=2.5), AnnRet=+687%, DD=68%, Calmar 10.12, 6년 연속 흑자.
25×25 Fine Grid + 4 Adaptive Model 분석 결과, 고정 L/N=2.5가 최적. Adaptive leverage는 고정 대비 열위.

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `VS_VR_SHORT` | 8 | 단기 ATR 기간 (R39: was 7) |
| `VS_VR_LONG` | 56 | 장기 ATR 기간 (R39: was 50) |
| `VS_D_VR_THRESHOLD` | 0.50 | D_mom8 VR 진입 임계값 |
| `VS_D_SL` | 0.01 | 1% 손절 |
| `VS_D_TP` | 0.12 | 12% 하드 TP 천장 (R43: trail이 먼저 작동) |
| `VS_D_TRAIL_ACTIVATE` | 0.02 | R43: +2% 이상 미실현 수익 시 트레일 활성화 |
| `VS_D_TRAIL_DISTANCE` | 0.003 | R43: 고점 대비 0.3% 되돌림 시 트레일 청산 |
| `VS_MOMENTUM_PERIOD` | 8 | 방향 결정 모멘텀 기간 |
| `VS_MAX_HOLD_15M` | 192 | 15m TF 최대 보유 (=48h) |
| `VS_MAX_HOLD_1H` | 48 | 1h TF 최대 보유 (=48h) |
| `VS_MAX_HOLD_4H` | 12 | 4h TF 최대 보유 (=48h) |
| `VS_BASE_LEVERAGE` | 25.0 | v13.0: 25x aggressive (L/N=2.5, Calmar 10.12) |
| `VS_MAX_LEVERAGE` | 25.0 | Hard cap (matches base_leverage) |
| `VS_MAX_CONCURRENT` | 10 | v13.0: 10 slots (L/N=25/10=2.5) |
| `VS_STRADDLE_ENABLED` | true | S_d8 스트래들 활성화 |
| `VS_HEDGE_MODE` | true | Binance Hedge Mode (dualSidePosition) |
| `VS_S_VR_THRESHOLD` | 0.55 | S_d8 VR 진입 임계값 |
| `VS_S_SQUEEZE_DUR` | 4 | S_d8 최소 스퀴즈 지속 바 수 (R39: was 8) |
| `VS_S_SL` | 0.015 | S_d8 1.5% 손절 (레그별) |
| `VS_S_TP` | 0.05 | S_d8 5% 익절 (레그별) |

### 3.6.3 CW-Tier 확신도 사이징 (R34 검증)

4가지 조건(E2, C2, B3, A3)의 충족 수(0~4)에 따라 포지션 크기를 조절합니다.

**조건:**
- **E2 (SMA 근접)**: `|close - SMA(20)| / SMA(20) < 0.5%` — 가격이 평균 근처일 때 스퀴즈 유효
- **C2 (BTC 중립)**: `|BTC momentum(8)| < 1%` — BTC 비추세 시 알트코인 스퀴즈 유효
- **B3 (컨플루언스)**: D_mom8과 S_d8이 동시에 발동 — 두 시그널 동시 확인
- **A3 (퀄리티+)**: D는 VR < 0.35 (깊은 압축), S는 squeeze ≥ 16바 (연장 스퀴즈)

**CW-Tier 배율:**

| Score | 배율 | 의미 |
|-------|------|------|
| 0 | 0.5x | 최소 확신 |
| 1 | 1.0x | 기본 |
| 2 | 1.5x | 보통 확신 |
| 3 | 2.0x | 높은 확신 |
| 4 | 2.5x | 최고 확신 |

**T3-Triple 모드**: `VS_MIN_CONVICTION_SCORE=3`으로 설정하면 score ≥ 3인 경우에만 진입 (고확신 필터).

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `VS_CW_ENABLED` | true | CW-Tier 확신도 사이징 활성화 |
| `VS_MIN_CONVICTION_SCORE` | 0 | 최소 확신도 점수 (0=전체, 3=T3모드) |
| `VS_E2_SMA_PERIOD` | 20 | E2: SMA 기간 |
| `VS_E2_THRESHOLD` | 0.005 | E2: 0.5% 최대 SMA 거리 |
| `VS_C2_BTC_MOM_THRESHOLD` | 0.01 | C2: 1% 최대 BTC 모멘텀 |
| `VS_A3_DEEP_VR_THRESHOLD` | 0.35 | A3: D 깊은 압축 임계값 |
| `VS_A3_LONG_SQUEEZE_DUR` | 16 | A3: S 연장 스퀴즈 기간 |

**OOS 검증 결과 (R34):**
- CW-Tier: PF=1.603, EV=+96.9bp, EV×N=+68,024 (기본 대비 +51.2%)
- T3-Triple: PF=1.943, EV=+85.9bp, DD=14.8%, CI_lo=1.520

### 3.6.4 실행

```bash
# VolStraddle 전략으로 실행
uv run python -m quantbox.scalper --strategy vol_straddle

# Paper 모드
uv run python -m quantbox.scalper --strategy vol_straddle --paper
```

### 3.6.5 백테스트 결과 (R35 OOS)

| D_mom8 (A1-D50) | 15m | 1h | 4h | 합계 |
|------------------|-----|-----|-----|------|
| N (트레이드 수) | 542 | 201 | 56 | 799 |
| EV (기대치) | +29.3bp | — | — | +29.3bp |
| PF (수익 팩터) | — | — | — | 1.376 |
| t-stat | — | — | — | 3.48 |

### 3.6.5 제로 룩어헤드 설계

- **15m**: `download_klines`가 개발 중인 바 제거 (`klines[:-1]`)
- **1h**: 15m 완성된 바만으로 리샘플링 (나머지 절삭)
- **4h**: 개별 다운로드, 개발 중 바 제거됨
- **교차 TF**: VR/모멘텀은 동일 TF 내 데이터만 사용

### 3.6.6 S_d8 스트래들 모드 (Hedge Mode)

**v12.7**: S_d8 스트래들이 활성화되었습니다. Binance **Hedge Mode** (`dualSidePosition=true`)가 필요합니다.

**동작 원리:**
- VR < 0.55 상태가 **8바 이상** 지속 → 양방향 동시 진입 (LONG + SHORT)
- 각 레그별 SL=1.5%, TP=5%
- `positionSide` 파라미터로 LONG/SHORT 방향 구분 (Hedge Mode 전용)
- 포지션 키는 복합 키 (`BTCUSDT:LONG`, `BTCUSDT:SHORT`)로 관리

**사전 준비 (필수):**
1. Binance 앱/웹 → Futures → Settings ⚙️ → Position Mode → **Hedge Mode** 활성화
2. `.env`에 `VS_HEDGE_MODE=true`, `VS_STRADDLE_ENABLED=true` 설정
3. 모든 기존 주문/포지션을 정리한 후 전환 (열린 주문이 있으면 `-4067` 에러)

---

## 3. 아키텍처 이해

### 3.1 전체 구조

```
quantbox/
├── core/                    # 핵심 컴포넌트
│   ├── config.py           # 설정 관리
│   ├── events.py           # 이벤트 시스템
│   ├── limit_chaser.py     # 지정가 추격 주문
│   ├── settings.py         # 환경변수 검증 (pydantic-settings)
│   └── state_persistence.py # 상태 영속화 (JSON, 크래시 복구)
├── data/                    # 데이터 계층
│   ├── cache.py            # 2단계 캐싱 (LRU + gzip)
│   ├── downloader.py       # 바이낸스 데이터 다운로드
│   └── real_data_loader.py # 실제 데이터 로더
├── strategy/               # 전략 계층
│   ├── rubberband.py       # Rubber Band Scalper (v12.4.6, Numba 가속)
│   ├── elliott.py          # Elliott Wave 전략 (v11.5, Numba 가속)
│   ├── alpha_engine.py     # 멀티팩터 Alpha Engine
│   └── vol_selector.py     # 변동성 기반 심볼 선택기
├── backtest/               # 백테스팅
│   └── vectorized_backtest.py  # 벡터화 백테스트 (Numba, bars_per_day 동적)
├── notifications/          # 알림 & 원격 제어
│   ├── discord.py          # Discord 웹훅 (단방향 알림)
│   └── bot.py              # Discord 봇 (양방향 명령)
├── scalper.py              # 멀티 전략 실거래 트레이더 (--strategy rubberband|elliott)
└── utils/                  # 유틸리티
```

### 3.2 이벤트 기반 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Market     │───▶│   Signal    │───▶│   Order     │
│  Event      │    │   Event     │    │   Event     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────┐
│                   Event Queue                        │
└─────────────────────────────────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Strategy   │    │  Portfolio  │    │  Execution  │
│             │    │             │    │   Handler   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**이벤트 타입:**
- `MARKET`: 새로운 시장 데이터
- `SIGNAL`: 전략 시그널 생성
- `ORDER`: 주문 생성
- `FILL`: 주문 체결

### 3.3 설정 계층 구조

```python
from quantbox.core.config import Config

Config
├── mode: TradingMode           # BACKTEST / PAPER / LIVE
├── interval: str               # "15m", "1h", etc.
├── exchange: ExchangeConfig    # API 설정
├── data: DataConfig            # 데이터 경로
├── strategy: StrategyConfig    # 전략 파라미터
├── risk: RiskConfig            # 리스크 관리
├── backtest: BacktestConfig    # 백테스트 설정
└── optimization: OptimizationConfig  # 최적화 설정
```

---

## 4. 핵심 모듈 상세

### 4.1 Config (설정 관리)

```python
from quantbox import Config, TradingMode

# 기본 설정 생성
config = Config()

# 커스텀 설정
config = Config(
    mode=TradingMode.BACKTEST,
    interval="1h"
)

# 전략 파라미터 조정
config.strategy.entry_zscore = 2.0
config.strategy.exit_zscore = 0.5
config.strategy.stop_loss_zscore = 3.5
config.strategy.max_holding_periods = 80
config.strategy.formation_periods = 240
config.strategy.trading_periods = 24

# 리스크 설정
config.risk.max_position_size = 0.1  # 10%
config.risk.trading_fee = 0.0005     # 0.05%
config.risk.slippage = 0.001         # 0.1%

# 백테스트 설정
config.backtest.initial_capital = 10000.0
config.backtest.walk_forward = True
```

### 4.2 TradingEngine (트레이딩 엔진)

```python
from quantbox import TradingEngine, Config

# 엔진 초기화
config = Config()
engine = TradingEngine(config)

# 전략 등록
engine.register_strategy("pair_trading", strategy)

# 콜백 등록
@engine.on_signal
def handle_signal(event):
    print(f"Signal: {event.signal_type}")

@engine.on_fill
def handle_fill(event):
    print(f"Fill: {event.symbol} @ {event.fill_price}")

# 백테스트 실행
results = engine.run_backtest(data, strategy_func)
```

### 4.3 Portfolio (포트폴리오 관리)

```python
from quantbox import Portfolio

portfolio = Portfolio(
    initial_capital=10000.0,
    trading_fee=0.0005,
    slippage=0.001
)

# 포지션 확인
pos = portfolio.get_position("BTCUSDT")
print(f"Quantity: {pos.quantity}")
print(f"Unrealized PnL: {pos.unrealized_pnl}")

# 포트폴리오 상태
print(f"Total Equity: {portfolio.equity}")
print(f"Cash: {portfolio.cash}")
print(f"Current Drawdown: {portfolio.current_drawdown}")
```

### 4.4 Events (이벤트 시스템)

```python
from quantbox.core.events import (
    SignalEvent, OrderEvent, SignalType, OrderSide
)

# 시그널 이벤트 생성
signal = SignalEvent(
    signal_type=SignalType.LONG,
    symbol1="BTCUSDT",
    symbol2="ETHUSDT",
    z_score=-2.5,
    hedge_ratio=0.85
)

# 주문 이벤트 생성
order = OrderEvent(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    quantity=0.1,
    is_pair_order=True,
    pair_symbol="ETHUSDT",
    hedge_ratio=0.85
)
```

---

## 5. 데이터 관리

### 5.1 DataDownloader (데이터 다운로드)

```python
from quantbox.data import DataDownloader
from quantbox import Config
from binance.client import Client

# 바이낸스 클라이언트 초기화
client = Client(api_key, api_secret)

# 다운로더 초기화
config = Config()
downloader = DataDownloader(client, config)

# 거래 가능 코인 목록 조회
coins = downloader.get_trading_coins(
    quote_currency="USDT",
    n_coins=50,
    sort_by_volume=True
)

# 단일 심볼 다운로드
from datetime import datetime, timedelta

data = downloader.download_klines(
    symbol="BTCUSDT",
    interval="1h",
    start=datetime.now() - timedelta(days=30),
    end=datetime.now(),
    use_cache=True
)

# 여러 심볼 다운로드
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
data = downloader.download_multiple(
    symbols=symbols,
    timeframe='1h',
    days=60
)

print(data['BTCUSDT'].head())
```

**반환 데이터 형식:**
```
                         Open      High       Low     Close      Volume
2025-01-01 00:00:00  42150.5  42230.0  42100.0  42180.0  1234567.89
2025-01-01 01:00:00  42180.0  42350.0  42150.0  42300.0  1345678.90
...
```

### 5.2 DataCache (캐싱 시스템)

**2단계 캐싱 아키텍처:**

```python
from quantbox.data.cache import DataCache, TimeframedCache

# 기본 캐시 초기화
cache = DataCache(
    cache_dir="data/cache",
    max_memory_entries=100,
    compress=True,
    expiry_hours=24
)

# 캐시 저장
cache.put(
    symbol="BTCUSDT",
    interval="1h",
    data=df,
    start=start_date,
    end=end_date
)

# 캐시 조회
cached_data = cache.get(
    symbol="BTCUSDT",
    interval="1h",
    start=start_date,
    end=end_date
)

# 캐시 통계
print(f"Memory cache size: {cache.memory_cache.size}")
print(f"Disk cache entries: {len(cache.disk_cache._index)}")
```

**타임프레임별 캐시:**

```python
# 멀티 타임프레임 캐시
tf_cache = TimeframedCache(
    cache_dir="data/cache",
    intervals=["15m", "1h", "4h", "1d"]
)

# 자동 타임프레임 감지
tf_cache.put("BTCUSDT", df, interval="1h")
data = tf_cache.get("BTCUSDT", interval="1h", days=30)
```

---

## 6. 페어 선택 시스템

### 6.1 필터 파이프라인

필터 파이프라인은 여러 필터를 순차적으로 적용하여 유효한 페어를 선별합니다.

```python
from quantbox.strategy import FilterPipeline
from quantbox.strategy.filters import (
    CorrelationFilter,
    CointegrationFilter,
    HalfLifeFilter,
    HurstFilter,
    VolumeFilter,
    StationarityFilter
)

# 필터 파이프라인 생성
pipeline = FilterPipeline([
    CorrelationFilter(
        min_correlation=0.6,    # 최소 상관계수
        max_correlation=0.98,   # 최대 상관계수 (1에 가까우면 동일 자산)
        use_returns=True        # 수익률 기반 계산
    ),
    CointegrationFilter(
        max_pvalue=0.05        # 공적분 p-value 임계값
    ),
    HalfLifeFilter(
        min_halflife=3,        # 최소 반감기 (너무 빠르면 노이즈)
        max_halflife=60        # 최대 반감기 (너무 느리면 비효율)
    ),
    HurstFilter(
        max_hurst=0.5          # < 0.5: 평균 회귀 성향
    ),
])

# 필터 적용
passed, score, results = pipeline.apply(series1, series2)

print(f"Passed: {passed}")
print(f"Combined Score: {score:.4f}")
for name, result in results.items():
    print(f"  {name}: {result.reason}")
```

### 6.2 개별 필터 상세

#### CorrelationFilter (상관관계 필터)
```python
CorrelationFilter(
    min_correlation=0.5,   # 너무 낮으면 관계 없음
    max_correlation=0.99,  # 너무 높으면 동일 자산
    use_returns=True       # 가격 대신 수익률 사용
)
```

#### CointegrationFilter (공적분 필터)
```python
CointegrationFilter(
    max_pvalue=0.05  # Engle-Granger 테스트 p-value
)
```
- p-value < 0.05: 공적분 관계 존재 (유의)
- p-value > 0.05: 공적분 관계 없음

#### HalfLifeFilter (반감기 필터)
```python
HalfLifeFilter(
    min_halflife=1,    # 최소 1 bar
    max_halflife=100   # 최대 100 bars
)
```
- 반감기: 스프레드가 평균의 절반까지 회귀하는 데 걸리는 시간
- 트레이딩 기간의 1/4 ~ 1/2이 이상적

#### HurstFilter (허스트 지수 필터)
```python
HurstFilter(
    min_hurst=0.0,
    max_hurst=0.5  # < 0.5 평균 회귀, = 0.5 랜덤워크, > 0.5 추세
)
```

### 6.3 PairSelector (페어 선택기)

```python
from quantbox.strategy import PairSelector

# 선택기 초기화
selector = PairSelector(
    filter_pipeline=pipeline,
    max_pairs=50,
    n_jobs=4  # 병렬 처리
)

# 페어 선택
candidates = selector.select_pairs(
    data=data,          # Dict[symbol, DataFrame]
    progress=True,
    min_data_points=100
)

# 결과 확인
for c in candidates[:10]:
    print(f"{c.symbol1}/{c.symbol2}:")
    print(f"  Score: {c.combined_score:.4f}")
    print(f"  Correlation: {c.correlation:.4f}")
    print(f"  Coint P-value: {c.cointegration_pvalue:.4f}")
    print(f"  Half-life: {c.half_life:.1f}")
    print(f"  Z-Score: {c.current_z_score:.2f}")

# 거래 가능 페어 필터링 (현재 Z-Score 기준)
tradeable = selector.select_tradeable_pairs(
    candidates,
    min_z_score=2.0,   # 최소 진입 Z-Score
    max_z_score=5.0    # 최대 (이상치 제외)
)

# 커스텀 가중치로 순위 재조정
ranked = selector.rank_pairs(
    candidates,
    weights={
        'combined_score': 0.3,
        'half_life_score': 0.2,
        'correlation_score': 0.15,
        'coint_score': 0.2,
        'z_score_strength': 0.15
    }
)
```

### 6.4 PairCandidate 데이터 클래스

```python
@dataclass
class PairCandidate:
    symbol1: str                    # 첫 번째 심볼
    symbol2: str                    # 두 번째 심볼
    passed: bool                    # 필터 통과 여부
    combined_score: float           # 종합 점수
    filter_results: Dict            # 개별 필터 결과
    
    # 통계
    correlation: float              # 상관계수
    cointegration_pvalue: float     # 공적분 p-value
    half_life: float               # 반감기
    hedge_ratio: float             # 헤지 비율
    hurst: float                   # 허스트 지수
    
    # 현재 상태
    current_z_score: float         # 현재 Z-Score
    spread_std: float              # 스프레드 표준편차
```

---

## 7. 전략 구현

### 7.1 기본 전략 클래스

```python
from quantbox.strategy.base import Strategy, StatefulStrategy

class MyStrategy(Strategy):
    """상태 없는 간단한 전략"""
    
    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
    
    def on_bar(self, timestamp, data):
        """새로운 바 데이터 처리"""
        pass
    
    def generate_signal(self, data):
        """시그널 생성"""
        pass

class MyStatefulStrategy(StatefulStrategy):
    """상태를 유지하는 전략"""
    
    def on_bar(self, timestamp, data):
        """상태 업데이트 및 시그널 생성"""
        self.update_state(timestamp, data)
        return self.check_signals()
```

### 7.2 PairTradingStrategy 상세

```python
from quantbox.strategy import PairTradingStrategy

strategy = PairTradingStrategy(
    name="MyPairTrading",
    config={
        'entry_zscore': 2.0,        # 진입 Z-Score
        'exit_zscore': 0.5,         # 청산 Z-Score
        'stop_loss_zscore': 3.5,    # 손절 Z-Score
        'max_holding_periods': 80,  # 최대 보유 기간 (20h @ 15m)
        'lookback_periods': 240,     # 통계 계산 기간
        'recalculate_hedge': True,   # 동적 헤지 비율
        'use_dynamic_zscore': False, # 동적 Z-Score 사용
        'dynamic_lookback': 50,      # 동적 계산 기간
    }
)

# 페어 초기화
strategy.initialize_pair(
    symbol1="BTCUSDT",
    symbol2="ETHUSDT",
    data1=btc_data,
    data2=eth_data
)

# 또는 PairCandidate로 추가
strategy.add_pair(candidate, btc_data, eth_data)

# 바 데이터 처리
bar_data = {
    'BTCUSDT': {'open': 42000, 'high': 42100, 'low': 41900, 'close': 42050, 'volume': 1000},
    'ETHUSDT': {'open': 2200, 'high': 2210, 'low': 2190, 'close': 2205, 'volume': 500}
}

signal = strategy.on_bar(timestamp, bar_data)

if signal:
    print(f"Signal Type: {signal['type']}")
    print(f"Z-Score: {signal['z_score']:.2f}")
```

### 7.3 시그널 로직

**진입 조건:**
```python
# Long Spread (자산1 매수, 자산2 매도)
if z_score <= -entry_zscore and not in_position:
    signal = SignalType.LONG

# Short Spread (자산1 매도, 자산2 매수)
if z_score >= entry_zscore and not in_position:
    signal = SignalType.SHORT
```

**청산 조건:**
```python
# 평균 회귀 청산
if in_long_position and z_score >= -exit_zscore:
    exit_reason = "mean_reversion"

if in_short_position and z_score <= exit_zscore:
    exit_reason = "mean_reversion"

# 손절
if in_long_position and z_score <= entry_z - stop_loss_zscore:
    exit_reason = "stop_loss"

if in_short_position and z_score >= entry_z + stop_loss_zscore:
    exit_reason = "stop_loss"

# Trailing Stop (Ratchet Stop)
# z가 entry 대비 activate_delta 이상 개선 후 entry 수준으로 역전 시 청산
if trailing_active and z reversed past entry_z ± buffer:
    exit_reason = "trailing_stop"

# 최대 보유 기간
if bars_held >= max_holding_periods:
    exit_reason = "max_holding"
```

### 7.4 Dynamic Exit (Half-Life 기반 청산)

진입 시 계산된 **half-life**를 활용하여 청산 z-score를 페어별로 자동 조절합니다.

**원리:**
- **빠른 페어** (낮은 half-life): 평균 회귀가 빠르므로 tight exit (z=0.3) → 더 깊은 수익 포착
- **느린 페어** (높은 half-life): 전부 돌아올 때까지 기다리면 스탑에 걸림 → loose exit (z=1.0) → 부분 수익 확정

**계산식:**
```
exit_z = base + clamp((hl - min_hl) / (max_hl - min_hl), 0, 1) × range
```

**기본값:** `base=0.3`, `range=0.7` → 출력 범위 [0.3, 1.0]

**A/B 백테스트 결과 (35심볼, 2877봉):**

| Metric | Static (exit=0.5) | Dynamic (hl-based) |
|--------|------------------:|-------------------:|
| Total Return | 0.25% | **3.38%** |
| Sharpe Ratio | 0.98 | **4.06** |
| Max Drawdown | 24.69% | **23.92%** |
| Stop Loss 횟수 | 11 | **10** |

**활성화:**
```bash
# .env
USE_DYNAMIC_EXIT=true
DYNAMIC_EXIT_BASE=0.3
DYNAMIC_EXIT_RANGE=0.7
```

> **주의**: 기본적으로 비활성화 상태입니다 (`USE_DYNAMIC_EXIT=false`). 활성화 시 `EXIT_ZSCORE` 값은 무시됩니다.

### 7.5 진입 방어 레이어 (Entry Gates)

진입 시 5개 레이어가 순서대로 실행됩니다. 하나라도 차단되면 진입하지 않습니다.
**라이브와 백테스트 모두 동일한 게이트를 적용합니다** (v1.4.0부터 패리티 달성).

| # | Gate | 대상 | 설정 | 기본값 |
|---|------|------|------|--------|
| 1 | **Z-Score Cap** | 개별 페어 | `Z_SCORE_CAP` | \|z\| > 4.0 |
| 2 | **Vol Expansion Gate** | 개별 페어 | `VOL_EXPANSION_GATE` | ratio > 2.0x |
| 3 | **Entry Z-Score Threshold** | 개별 페어 | `ENTRY_ZSCORE` | \|z\| >= 2.0 |
| 4 | **자본 충분성** | 슬롯 | - | min notional 이상 |

### 7.6 커스텀 전략 구현 예시

```python
from quantbox.strategy.base import StatefulStrategy
from quantbox.core.events import SignalType

class EnhancedPairStrategy(StatefulStrategy):
    """향상된 페어 트레이딩 전략"""
    
    DEFAULT_PARAMS = {
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
        'stop_loss_zscore': 3.5,
        'momentum_filter': True,
        'momentum_lookback': 20,
        'volume_filter': True,
        'min_volume_ratio': 0.8,
    }
    
    def __init__(self, name: str = "EnhancedPair", config: dict = None):
        params = {**self.DEFAULT_PARAMS, **(config or {})}
        super().__init__(name, params)
        self.momentum_history = {}
    
    def on_bar(self, timestamp, data):
        # 기본 Z-Score 계산
        z_score = self.calculate_z_score(data)
        
        # 모멘텀 필터
        if self.config['momentum_filter']:
            if not self.check_momentum(data):
                return None
        
        # 거래량 필터
        if self.config['volume_filter']:
            if not self.check_volume(data):
                return None
        
        # 시그널 생성
        return self.generate_signal(z_score)
    
    def check_momentum(self, data):
        """모멘텀 확인: 추세 반대 방향으로만 진입"""
        lookback = self.config['momentum_lookback']
        # 구현...
        return True
    
    def check_volume(self, data):
        """거래량 확인: 충분한 유동성 확인"""
        min_ratio = self.config['min_volume_ratio']
        # 구현...
        return True
```

---

## 8. 백테스팅

### 8.1 기본 백테스트

```python
from quantbox import BacktestRunner
from quantbox.backtest.runner import BacktestConfig

# 백테스트 설정
config = BacktestConfig(
    initial_capital=10000.0,      # 초기 자본
    trading_fee=0.0005,           # 거래 수수료 (0.05%)
    slippage=0.001,               # 슬리피지 (0.1%)
    
    # 윈도우 설정
    formation_periods=240,        # 형성 기간 (통계 계산용)
    trading_periods=24,           # 거래 기간
    window_shift=24,              # 윈도우 이동 간격
    
    # 전략 파라미터
    entry_zscore=2.0,
    exit_zscore=0.5,
    stop_loss_zscore=3.5,
    max_holding_periods=80,
    
    # 페어 선택
    max_pairs_per_window=10,
    min_correlation=0.5,
    max_cointegration_pvalue=0.05,
)

# 백테스트 실행
runner = BacktestRunner(config)
results = runner.run_backtest(
    data=data,                    # Dict[symbol, DataFrame]
    pair_selector=selector,       # 페어 선택기 (선택)
    strategy=strategy,            # 전략 (선택)
    progress=True                 # 진행 표시
)

# 결과 확인
print(results['metrics'].summary())
```

### 8.2 롤링 윈도우 백테스트

```
시간 축: ─────────────────────────────────────────────►

Window 1: [======Formation======][==Trading==]
Window 2:          [======Formation======][==Trading==]
Window 3:                   [======Formation======][==Trading==]
                   ◄──shift─►
```

```python
# 윈도우 생성 설정
config = BacktestConfig(
    formation_periods=240,  # 형성: 240 bars
    trading_periods=24,     # 거래: 24 bars
    window_shift=24,        # 이동: 24 bars
)

# 각 윈도우에서:
# 1. Formation Period: 페어 선택, 통계 계산
# 2. Trading Period: 실제 거래 시뮬레이션
```

### 8.3 결과 분석

```python
# 메트릭스
metrics = results['metrics']
print(f"Total Return: {metrics.total_return_pct:.2f}%")
print(f"Annualized Return: {metrics.annualized_return:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
print(f"Win Rate: {metrics.win_rate:.1f}%")
print(f"Profit Factor: {metrics.profit_factor:.2f}")

# 거래 내역
trades = results['trades']
for trade in trades[:5]:
    print(f"{trade.pair}: {trade.pnl:+.2f} ({trade.exit_reason})")

# 에쿼티 커브
equity_curve = results['equity_curve']
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(equity_curve['timestamp'], equity_curve['equity'])
plt.title('Equity Curve')
plt.xlabel('Time')
plt.ylabel('Equity ($)')
plt.show()

# 페어별 성과
pair_performance = results['pair_performance']
for pair, trades in pair_performance.items():
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len([t for t in trades if t.pnl > 0]) / len(trades) * 100
    print(f"{pair}: ${total_pnl:.2f}, WR: {win_rate:.1f}%")
```

### 8.4 TradeRecord 상세

```python
@dataclass
class TradeRecord:
    pair: Tuple[str, str]         # 페어 심볼
    entry_time: datetime          # 진입 시간
    exit_time: datetime           # 청산 시간
    entry_z_score: float          # 진입 Z-Score
    exit_z_score: float           # 청산 Z-Score
    entry_spread: float           # 진입 스프레드
    exit_spread: float            # 청산 스프레드
    signal: int                   # 1: Long, -1: Short
    pnl: float                    # 절대 손익
    pnl_pct: float               # 수익률
    bars_held: int               # 보유 기간
    exit_reason: str             # 청산 사유
    hedge_ratio: float           # 사용된 헤지 비율
```

---

## 9. 파라미터 최적화

### 9.1 Grid Search

```python
from quantbox.optimization import GridSearchOptimizer

# 파라미터 범위 정의
param_bounds = {
    'entry_zscore': (1.5, 3.0),
    'exit_zscore': (0.2, 1.0),
    'stop_loss_zscore': (2.5, 4.0),
}

# Grid Search 최적화
optimizer = GridSearchOptimizer(
    param_bounds=param_bounds,
    param_steps={'entry_zscore': 5, 'exit_zscore': 4, 'stop_loss_zscore': 4},
    maximize=True
)

result = optimizer.optimize(data, progress=True)

print(f"Best Parameters: {result.best_params}")
print(f"Best Score: {result.best_score:.4f}")
print(f"Total Evaluations: {result.num_evaluations}")

# 결과 DataFrame
df = result.to_dataframe()
print(df.head(10))
```

### 9.2 Random Search

```python
from quantbox.optimization import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    param_bounds=param_bounds,
    n_iterations=200,
    maximize=True
)

result = optimizer.optimize(data)
```

### 9.3 Differential Evolution

```python
from quantbox.optimization import DifferentialEvolutionOptimizer

optimizer = DifferentialEvolutionOptimizer(
    param_bounds=param_bounds,
    population_size=30,
    mutation=(0.5, 1.0),
    crossover=0.7,
    max_iterations=100,
    tolerance=1e-6,
    maximize=True
)

result = optimizer.optimize(data, progress=True)

# 수렴 히스토리
import matplotlib.pyplot as plt
plt.plot(result.convergence_history)
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.title('DE Convergence')
plt.show()
```

### 9.4 Bayesian Optimization

```python
from quantbox.optimization import BayesianOptimizer

optimizer = BayesianOptimizer(
    param_bounds=param_bounds,
    n_initial=20,        # 초기 랜덤 샘플
    n_iterations=80,     # 베이지안 반복
    acquisition='ei',    # Expected Improvement
    maximize=True
)

result = optimizer.optimize(data)
```

### 9.5 목적 함수 커스터마이징

```python
from quantbox.optimization.objective import (
    SharpeObjective, CalmarObjective, MultiObjective, create_objective
)

# 단일 목적 함수
sharpe_obj = SharpeObjective()
calmar_obj = CalmarObjective()

# 다목적 함수
multi_obj = MultiObjective([
    (SharpeObjective(), 0.5),
    (CalmarObjective(), 0.3),
    ('win_rate', 0.2)
])

# 커스텀 목적 함수
def custom_objective(metrics):
    """Sortino ratio를 최대화하되 drawdown 제한"""
    if metrics.max_drawdown > 20:  # 20% 이상 DD 페널티
        return -999
    return metrics.sortino_ratio

optimizer = GridSearchOptimizer(
    param_bounds=param_bounds,
    objective_func=custom_objective
)
```

### 9.6 민감도 분석

```python
from quantbox.optimization import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(
    base_params=result.best_params,
    param_ranges={
        'entry_zscore': (1.5, 3.0, 0.25),    # (min, max, step)
        'exit_zscore': (0.2, 1.0, 0.2),
        'stop_loss_zscore': (2.5, 4.0, 0.5),
    }
)

sensitivity = analyzer.analyze(data)

# 결과 시각화
analyzer.plot_sensitivity()
analyzer.plot_heatmap('entry_zscore', 'exit_zscore')

# 파라미터 안정성 점수
for param, score in sensitivity['stability_scores'].items():
    print(f"{param}: {score:.4f}")  # 높을수록 안정적
```

---

## 10. Walk-Forward 분석

### 10.1 기본 사용법

Walk-Forward 분석은 **과최적화 방지**와 **실제 성과 예측**을 위한 핵심 검증 방법입니다.

```
In-Sample (Training)    Out-of-Sample (Testing)
├─────────────────────┼───────────┤
│     최적화           │   검증     │
├─────────────────────┼───────────┤
        │     최적화           │   검증     │
        ├─────────────────────┼───────────┤
                │     최적화           │   검증     │
                └─────────────────────┴───────────┘
```

```python
from quantbox.backtest import WalkForwardAnalysis

wf = WalkForwardAnalysis(
    train_periods=500,    # 학습 기간 (bars)
    test_periods=100,     # 테스트 기간 (bars)
    step_periods=100,     # 이동 간격 (bars)
    min_trades=10         # 최소 거래 수
)

# 분석 실행
results = wf.run(
    data=data,
    param_ranges={
        'entry_zscore': (1.5, 3.0),
        'exit_zscore': (0.2, 1.0),
    },
    progress=True
)

# 결과 확인
for window in wf.windows:
    print(f"Window {window.window_id}:")
    print(f"  Train: {window.train_start} - {window.train_end}")
    print(f"  Test: {window.test_start} - {window.test_end}")
    print(f"  Best Params: {window.best_params}")
    print(f"  Train Return: {window.train_metrics.total_return_pct:.2f}%")
    print(f"  Test Return: {window.test_metrics.total_return_pct:.2f}%")
    print(f"  Efficiency Ratio: {window.efficiency_ratio:.2f}")
```

### 10.2 효율성 비율 (Efficiency Ratio)

```python
# 효율성 비율 = OOS Return / IS Return
# > 0.5: 좋음 (과최적화 낮음)
# < 0.3: 과최적화 가능성

avg_efficiency = np.mean([w.efficiency_ratio for w in wf.windows])
print(f"Average Efficiency Ratio: {avg_efficiency:.2f}")

if avg_efficiency < 0.3:
    print("Warning: Possible overfitting detected")
```

### 10.3 통합 결과

```python
# 전체 OOS 성과 (실제 예상 성과)
combined_results = results['combined']
print(f"Combined OOS Return: {combined_results['total_return']:.2f}%")
print(f"Combined Sharpe: {combined_results['sharpe_ratio']:.2f}")

# OOS 에쿼티 커브
combined_equity = wf.combined_equity
plt.plot(combined_equity['timestamp'], combined_equity['equity'])
plt.title('Walk-Forward Combined Equity (Out-of-Sample)')
plt.show()
```

---

## 11. 실거래 연동

### 11.1 실거래 설정

```python
from quantbox import Config, TradingEngine, TradingMode
from binance.client import Client

# 설정
config = Config(mode=TradingMode.LIVE)
config.exchange.api_key = "your_api_key"
config.exchange.api_secret = "your_api_secret"
config.exchange.testnet = False  # 실거래

# 바이낸스 클라이언트
client = Client(
    config.exchange.api_key,
    config.exchange.api_secret
)

# 엔진 초기화
engine = TradingEngine(config, client)
```

### 11.2 Paper Trading (모의 거래)

```python
config = Config(mode=TradingMode.PAPER)

# Paper trading은 실제 주문 없이 시뮬레이션
engine = TradingEngine(config)
engine.register_strategy("pair_trading", strategy)

# 실시간 데이터 피드 시뮬레이션
while True:
    # 데이터 업데이트
    data = fetch_latest_data()
    engine.update_market_data(datetime.now(), data)
    engine.process_events()
    
    time.sleep(60)  # 1분 대기
```

### 11.3 실거래 실행

```python
from quantbox.core.execution import LiveExecutionHandler

# 실행 핸들러
execution = LiveExecutionHandler(config, client)

# 콜백 설정
@engine.on_fill
def on_trade_fill(event):
    print(f"Order filled: {event.symbol} @ {event.fill_price}")
    # 알림 전송, 로깅 등

@engine.on_signal
def on_signal_generated(event):
    print(f"Signal: {event.signal_type} for {event.symbol1}/{event.symbol2}")

# 실행
engine.run_live(strategy)
```

### 11.4 리스크 관리

```python
# 리스크 파라미터 설정
config.risk.max_position_size = 0.05    # 최대 포지션: 5%
config.risk.max_drawdown = 0.15         # 최대 낙폭: 15%
config.risk.stop_loss_pct = 0.03        # 손절: 3%
config.risk.leverage = 2                 # Fallback 레버리지 (동적 미사용 시)
config.risk.target_r_pct = 0.02         # 트레이드당 목표 리스크: 밸런스의 2%
config.risk.max_leverage = 125          # 동적 레버리지 상한: 125x (바이난스 절대 상한)
config.risk.funding_rate_8h = 0.0001     # 8시간 펀딩비: 0.01%

# 자동 리스크 관리
if portfolio.current_drawdown > config.risk.max_drawdown:
    engine.close_all_positions()
    print("Max drawdown exceeded - all positions closed")
```

---

## 12. Discord Bot (원격 제어)

### 12.1 개요

Discord Bot을 통해 실거래 중인 시스템을 채팅으로 원격 제어할 수 있습니다. Webhook(단방향 알림)과 달리, Bot은 **양방향 통신**이 가능하여 상태 조회, 포지션 청산, 시스템 정지 등의 명령을 실행할 수 있습니다.

**아키텍처:**
```
메인 스레드 (트레이딩 루프)     데몬 스레드 (Discord Bot)
┌──────────────────┐          ┌──────────────────┐
│  run_cycle()     │◀─Queue──│  on_message()    │
│  _process_cmds() │──Reply──▶│  send response   │
└──────────────────┘          └──────────────────┘
```

### 12.2 Bot 생성 및 설정

**1단계: Discord Developer Portal에서 Bot 생성**

1. https://discord.com/developers/applications 접속
2. "New Application" → 이름 입력 → 생성
3. 좌측 "Bot" 메뉴 → "Reset Token" → **토큰 복사**
4. "Privileged Gateway Intents" → **Message Content Intent** 활성화

**2단계: Bot을 서버에 초대**

1. 좌측 "OAuth2" → "URL Generator"
2. Scopes: `bot` 선택
3. Bot Permissions: `Send Messages`, `Read Message History` 선택
4. 생성된 URL로 서버에 초대

**3단계: 채널 ID 확인**

1. Discord 설정 → 고급 → "개발자 모드" 활성화
2. 명령용 채널 우클릭 → "ID 복사"

**4단계: .env 설정**

```dotenv
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_CHANNEL_ID=1234567890123456789
```

### 12.3 사용 가능한 명령어

| 명령어 | 설명 | 응답 예시 |
|--------|------|----------|
| `!status` | 전체 시스템 상태 | 슬롯별 페어, PnL, z-score |
| `!pnl` | 손익 요약 | 총 PnL, 슬롯별 수익률 |
| `!pause` | 신규 진입 일시 정지 | 기존 포지션은 유지, 새 진입만 차단 |
| `!resume` | 일시 정지 해제 | 정상 운영 복귀 |
| `!close all` | 전체 포지션 청산 | 모든 슬롯 즉시 청산 |
| `!close N` | N번 슬롯 청산 | 특정 슬롯만 청산 (0부터 시작) |
| `!stop` | 시스템 종료 | 전체 포지션 청산 후 안전 종료 |
| `!help` | 명령어 목록 | 위 명령어 안내 |

> **안전장치**: `!close`와 `!stop`은 실제 바이낸스 주문을 트리거합니다. 반드시 의도한 것인지 확인하세요.

### 12.4 작동 방식

- Bot은 `run_continuous()` 호출 시 **데몬 스레드**로 자동 시작
- `DISCORD_BOT_TOKEN`과 `DISCORD_CHANNEL_ID`가 모두 설정된 경우에만 활성화
- 미설정 시 기존과 동일하게 봇 없이 운영 (하위 호환)
- 매 `run_cycle()` 시작 시 큐에 쌓인 명령을 처리
- 명령 응답은 3초 타임아웃 (봇 → 트레이딩 스레드 응답 대기)

```python
# 코드에서 직접 사용 (자동으로 통합됨)
from quantbox.live_trader import LivePairTrader
from quantbox.core.settings import Settings

settings = Settings()  # .env에서 자동 로드
trader = LivePairTrader(settings)
trader.run_continuous()  # Bot 자동 시작 (토큰 설정 시)
```

---

## 13. 성과 지표 해석

### 12.1 수익 지표

| 지표 | 계산 | 해석 |
|------|------|------|
| Total Return | (최종 자본 - 초기 자본) / 초기 자본 | 전체 수익률 |
| Annualized Return | (1 + Total Return)^(252/days) - 1 | 연환산 수익률 |
| CAGR | (최종/초기)^(1/years) - 1 | 복리 연평균 성장률 |

### 12.2 위험 지표

| 지표 | 계산 | 해석 |
|------|------|------|
| Volatility | std(returns) × √252 | 연환산 변동성 |
| Max Drawdown | max((peak - trough) / peak) | 최대 낙폭 |
| VaR (95%) | quantile(returns, 0.05) | 95% 신뢰수준 최대 손실 |

### 12.3 위험조정 수익률

| 지표 | 계산 | 기준 |
|------|------|------|
| Sharpe Ratio | (Return - Rf) / Volatility | > 1.0 양호, > 2.0 우수 |
| Sortino Ratio | (Return - Rf) / Downside Dev | > 1.5 양호 |
| Calmar Ratio | CAGR / Max Drawdown | > 1.0 양호 |

### 12.4 거래 통계

```python
metrics = results['metrics']

# 승률 분석
print(f"Win Rate: {metrics.win_rate:.1f}%")
print(f"Avg Win: ${metrics.avg_win:.2f}")
print(f"Avg Loss: ${metrics.avg_loss:.2f}")
print(f"Payoff Ratio: {abs(metrics.avg_win/metrics.avg_loss):.2f}")

# Profit Factor
print(f"Profit Factor: {metrics.profit_factor:.2f}")
# > 1.5: 양호, > 2.0: 우수

# 기대값
expectancy = metrics.win_rate/100 * metrics.avg_win + (1-metrics.win_rate/100) * metrics.avg_loss
print(f"Expectancy per Trade: ${expectancy:.2f}")
```

### 12.5 권장 기준

| 지표 | 최소 기준 | 양호 | 우수 |
|------|-----------|------|------|
| Sharpe Ratio | > 0.5 | > 1.0 | > 2.0 |
| Sortino Ratio | > 0.7 | > 1.5 | > 3.0 |
| Max Drawdown | < 30% | < 20% | < 10% |
| Win Rate | > 40% | > 50% | > 60% |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |
| WF Efficiency | > 0.3 | > 0.5 | > 0.7 |

---

## 14. 고급 사용법

### 14.1 Numba 가속 함수

```python
from quantbox.utils.numba_ops import (
    calculate_rolling_zscore,
    calculate_half_life,
    calculate_hurst_exponent,
    kelly_criterion,
    optimal_position_sizing
)

# 롤링 Z-Score (10x 빠름)
zscore = calculate_rolling_zscore(spread.values, lookback=50)

# 반감기 계산
half_life = calculate_half_life(spread.values)

# 허스트 지수
hurst = calculate_hurst_exponent(prices.values, max_lag=20)

# 켈리 기준
optimal_fraction = kelly_criterion(win_rate, avg_win, avg_loss)

# 최적 포지션 사이즈
position_size = optimal_position_sizing(
    capital=10000,
    risk_per_trade=0.02,
    stop_distance=0.05
)
```

### 14.2 몬테카를로 시뮬레이션

```python
from quantbox.backtest import MonteCarloSimulation

# 시뮬레이션 실행
mc = MonteCarloSimulation(
    trades=results['trades'],
    n_simulations=1000
)

mc_results = mc.run()

# 신뢰 구간
print(f"95% CI Return: [{mc_results['ci_95'][0]:.2f}%, {mc_results['ci_95'][1]:.2f}%]")
print(f"Probability of Loss: {mc_results['prob_loss']:.1f}%")
print(f"Expected Drawdown: {mc_results['expected_dd']:.2f}%")

# 분포 시각화
mc.plot_distribution()
```

### 14.3 Frozen 헤지 비율 (OLS)

```python
import numpy as np

def ols_hedge_ratio(price1, price2):
    """OLS를 이용한 고정 헤지 비율 추정 (formation 기간에 1회 계산)"""
    log_p1 = np.log(price1.values)
    log_p2 = np.log(price2.values)
    
    # Simple OLS: log_p1 = gamma * log_p2 + mu + epsilon
    A = np.column_stack([log_p2, np.ones(len(log_p2))])
    result = np.linalg.lstsq(A, log_p1, rcond=None)
    hedge_ratio, intercept = result[0]
    
    return hedge_ratio, intercept
```

### 14.4 다중 타임프레임 전략

```python
class MultiTimeframeStrategy(StatefulStrategy):
    """멀티 타임프레임 페어 트레이딩"""
    
    def __init__(self):
        self.tf_data = {
            '1h': {},    # 시그널 생성
            '4h': {},    # 트렌드 필터
            '1d': {},    # 장기 컨텍스트
        }
    
    def on_bar(self, timestamp, data):
        # 상위 타임프레임 트렌드 확인
        daily_trend = self.get_trend('1d')
        hourly_trend = self.get_trend('4h')
        
        # 트렌드 정렬 필터
        if daily_trend != hourly_trend:
            return None  # 트렌드 불일치 시 대기
        
        # 1시간 봉에서 시그널 생성
        z_score = self.calculate_z_score(data, '1h')
        
        return self.generate_signal(z_score, trend=daily_trend)
```

### 14.5 포트폴리오 레벨 리스크 관리

```python
class PortfolioRiskManager:
    """포트폴리오 전체 리스크 관리"""
    
    def __init__(self, max_pairs=5, max_correlation=0.7):
        self.max_pairs = max_pairs
        self.max_correlation = max_correlation
        self.active_pairs = []
    
    def can_add_pair(self, new_pair, existing_pairs):
        """새 페어 추가 가능 여부"""
        if len(existing_pairs) >= self.max_pairs:
            return False
        
        # 기존 페어들과의 상관관계 확인
        for pair in existing_pairs:
            corr = self.calculate_pair_correlation(new_pair, pair)
            if corr > self.max_correlation:
                return False
        
        return True
    
    def calculate_position_sizes(self, pairs, capital):
        """페어별 포지션 크기 계산 (동일 위험 기여)"""
        volatilities = [self.get_pair_volatility(p) for p in pairs]
        inv_vol = [1/v for v in volatilities]
        total = sum(inv_vol)
        
        sizes = {}
        for pair, iv in zip(pairs, inv_vol):
            sizes[pair] = capital * (iv / total) / len(pairs)
        
        return sizes
```

---

## 15. 트러블슈팅

### 15.1 일반적인 오류

#### API 연결 오류
```python
# 오류: BinanceAPIException: Invalid API-key
# 해결:
# 1. API 키 확인
# 2. IP 화이트리스트 확인
# 3. API 권한 확인

from binance.exceptions import BinanceAPIException
try:
    client.get_account()
except BinanceAPIException as e:
    print(f"API Error: {e.message}")
```

#### 데이터 누락
```python
# 오류: KeyError: timestamp not in index
# 해결: 데이터 정렬 확인

# 인덱스 정렬
df = df.sort_index()

# 누락 데이터 확인
missing = df.isnull().sum()
print(f"Missing values: {missing}")

# 보간
df = df.interpolate(method='time')
```

#### 메모리 부족
```python
# 해결: 청크 처리

def process_in_chunks(data, chunk_size=10000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        yield process_chunk(chunk)
```

### 15.2 성능 최적화

```python
# 1. Numba 캐시 활용
# 첫 실행은 느리지만 이후 빠름

# 2. 벡터화 연산
# 나쁜 예
for i in range(len(df)):
    df.iloc[i] = calculate(df.iloc[i])

# 좋은 예
df['result'] = df.apply(calculate, axis=1)

# 3. 필요한 컬럼만 로드
df = pd.read_csv('data.csv', usecols=['Open', 'High', 'Low', 'Close'])
```

### 15.3 디버깅 팁

```python
import logging

# 상세 로깅 활성화
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# 특정 모듈만 디버그
logging.getLogger('PairSelector').setLevel(logging.DEBUG)
logging.getLogger('BacktestRunner').setLevel(logging.INFO)
```

---

## 16. API 레퍼런스

### 16.1 주요 클래스

```python
# Core
from quantbox.core.config import RubberBandConfig
from quantbox.core.settings import get_settings
from quantbox.core.events import SignalEvent, OrderEvent, SignalType, OrderSide
from quantbox.core.limit_chaser import LimitChaser
from quantbox.core.state_persistence import save_state, load_state

# Data
from quantbox.data import (
    DataDownloader,          # 데이터 다운로드
    DataCache,               # 캐시 시스템
    RealDataLoader,          # 실제 데이터 로더
    get_real_data,           # 편의 함수
)

# Strategy
from quantbox.strategy.rubberband import RubberBandStrategy
from quantbox.strategy.alpha_engine import AlphaEngine
from quantbox.strategy.vol_selector import VolSelector

# Backtest
from quantbox.backtest import VectorizedBacktest, BacktestConfig, BacktestResult

# Notifications
from quantbox.notifications.discord import DiscordNotifier
from quantbox.notifications.bot import DiscordBotRunner, BotCommand, CommandType

# Scalper (Live Trading)
from quantbox.scalper import RubberBandScalper
```

### 16.2 설정 옵션 전체 목록

```python
@dataclass
class StrategyConfig:
    correlation_threshold: float = 0.5     # 최소 상관계수
    cointegration_pvalue: float = 0.05     # 공적분 p-value
    normality_pvalue: float = 0.05         # 정규성 p-value
    min_half_life: float = 1.0             # 최소 반감기
    max_half_life: float = 100.0           # 최대 반감기
    hedge_ratio_bounds: tuple = (0.25, 4.0) # 헤지 비율 범위
    entry_zscore: float = 2.0              # 진입 Z-Score
    exit_zscore: float = 0.0               # 청산 Z-Score
    stop_loss_zscore: float = 3.5          # 손절 Z-Score
    max_holding_periods: int = 80          # 최대 보유 기간 (20h @ 15m)
    formation_periods: int = 240           # 형성 기간
    trading_periods: int = 24              # 거래 기간

@dataclass
class RiskConfig:
    max_position_size: float = 0.1         # 최대 포지션 비율
    max_drawdown: float = 0.2              # 최대 낙폭
    leverage: int = 2                       # Fallback 레버리지 (동적 미사용 시)
    target_r_pct: float = 0.02             # 트레이드당 목표 리스크 (밸런스의 2%)
    max_leverage: int = 125                # 동적 레버리지 안전 상한 (바이난스 절대 상한)
    trading_fee: float = 0.0005            # 거래 수수료
    slippage: float = 0.001                # 슬리피지
    funding_rate_8h: float = 0.0001        # 8시간 펀딩비
    stop_loss_pct: float = 0.05            # 손절 비율
    take_profit_pct: float = 0.10          # 익절 비율

@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0       # 초기 자본
    start_date: Optional[str] = None       # 시작일
    end_date: Optional[str] = None         # 종료일
    window_shift_hours: int = 24           # 윈도우 이동
    train_test_split: float = 0.8          # 학습/테스트 분할
    walk_forward: bool = True              # Walk-Forward 사용
    monte_carlo_runs: int = 1000           # MC 시뮬레이션 횟수
```

---

## 📚 부록

### A. 페어 트레이딩 수학

**공적분 검정 (Engle-Granger)**
$$
y_t = \alpha + \beta x_t + \epsilon_t
$$

**ADF 검정 통계량**
$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t
$$

**Z-Score**
$$
z_t = \frac{s_t - \mu_s}{\sigma_s}
$$
여기서 $s_t$는 스프레드, $\mu_s$는 평균, $\sigma_s$는 표준편차

**반감기**
$$
\text{Half-Life} = -\frac{\ln(2)}{\ln(\theta)}
$$
여기서 $\theta$는 AR(1) 계수

**허스트 지수**
$$
E[R(n)/S(n)] = Cn^H
$$
- $H < 0.5$: 평균 회귀
- $H = 0.5$: 랜덤 워크
- $H > 0.5$: 추세 추종

### B. 용어 사전

| 용어 | 설명 |
|------|------|
| Cointegration | 두 비정상 시계열의 선형 결합이 정상인 관계 |
| Half-life | 평균으로 절반 회귀하는 데 걸리는 시간 |
| Hedge Ratio | 페어 간 최적 비율 |
| Spread | 두 자산의 가격 차이 |
| Z-Score | 표준화된 스프레드 값 |
| Walk-Forward | 롤링 최적화+검증 방법 |
| Sharpe Ratio | 위험 대비 초과 수익 |
| Drawdown | 고점 대비 하락률 |

### C. 참고 자료

1. Vidyamurthy, G. (2004). *Pairs Trading*
2. Gatev, E. et al. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*
3. Avellaneda, M. & Lee, J. (2010). *Statistical Arbitrage in the US Equities Market*

---

## 🎯 Quick Start 체크리스트

- [ ] Python 3.13+ 설치
- [ ] 의존성 패키지 설치 (`pip install -e .`)
- [ ] 바이낸스 API 키 발급 및 설정
- [ ] 샘플 데이터 다운로드
- [ ] 기본 백테스트 실행
- [ ] 결과 분석 및 시각화
- [ ] Walk-Forward 검증
- [ ] 파라미터 최적화
- [ ] Paper Trading 테스트
- [ ] Discord Bot 설정 (선택사항)

---

**문의 및 지원**: GitHub Issues를 통해 문의해주세요.

**라이선스**: MIT License

---

*이 문서는 QuantBox 12.4.6 기준으로 작성되었습니다. (2026-02-21)*
