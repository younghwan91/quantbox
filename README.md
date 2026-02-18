# QuantBox 📈

**암호화폐 Rubber Band Scalper + Elliott Wave — Binance USDT-M Futures**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

멀티 전략 스캘핑 프레임워크 (Rubber Band 15분봉 + Elliott Wave 4h봉, Numba 가속)

## 🌟 주요 기능

### 🔥 Rubber Band Scalper (v11.4 — Adaptive Threshold)
- **Alpha**: ATR 기반 적응형 임계값으로 시장 변동성에 실시간 대응하는 단일 자산 평균회귀 전략
- **v11.4 Adaptive Threshold**: 고정 3.2% → ATR(48)×2.5 적응형 (OOS Sharpe 8.23, OOS/IS 1.09, 제로 과적합)
  - 공식: `threshold = clip(2.5 × ATR%(48), 1%, 6%)` — 변동성 높으면 넓고, 낮으면 타이트
  - Walk-forward 검증: 18개 변형 × 20개 심볼, IS/OOS 분리, 모든 변형에서 OOS/IS ≥ 1.04
  - 레버리지 최적화: 1×~5× 그리드 → Sharpe 단조 감소, 현행 3.75× 유지 결정
- **v10.0 Entry Optimization**: threshold fallback 0.032 (ATR NaN 시), min_confidence 1 (MTF as quality gate)
- **v10.0 Extended Hold**: max_hold 10→20 (+24% ret, strictly better)
- **v10.0 Faster Stochastic**: stoch_k 5→3 (+5% Calmar)
- **v11.1 Server-Side Stop-Loss**: 바이난스 STOP 주문으로 클라이언트 크래시 방어 + 레버리지 설정 검증
- **v11.1 Pool Expansion**: Top-15→Top-20 (A/B validated: 3384% ret, Calmar 155.22 vs baseline 1470%)
- **v10.0 Symbol Whitelist**: Top-20 검증된 심볼만 거래 — 3384% ret, WR 69.7%, Calmar 155.22
- **v10.0 Dynamic Capital**: balance / max(1, n_active+1) — 유휴 자본 최소화, 50% cap 안전장치
- **Fibonacci Exit Levels (v9.0)**: 피보나치 기반 정밀 목표/손절 — target 0.382 (golden ratio), stop 0.786 (deep correction)
  - R:R 0.444→0.486 (+9.5%), Calmar +29% vs 기존 0.4/0.9
  - Dynamic Target 비활성화: 타이트한 Fib 스탑(0.786)에서 고정 0.382가 동적 스케일링보다 우수
- **MACD Histogram Evidence (v9.0)**: MACD(12,26,14) 히스토그램 기울기 방향 일치 시 +1 컨피던스
  - 단독 +7.6% Calmar 개선
- **Stochastic Cross Evidence (v9.0)**: 스토캐스틱(3,3) %K/%D 크로스 방향 일치 시 +1 컨피던스
  - Very fast 설정(3,3) — 스캘핑 최적 (v10.0: 5→3으로 가속), high/low 캔들 데이터 사용
- **S+R+M Confluence 종합**: Calmar 15.38 (+33.6% vs v8.0), DD 11.0% (-8.5% vs v8.0)
- **MTF Confirmation (v7.0)**: 4h SMA(12) 트렌드 필터로 15m 역추세 진입 차단
  - MDD 16.5% → 4.3% (-74%), Calmar 8.65 → 32.42 (+275%), Sharpe 2.07 → 5.23 (+153%)
  - Walk-forward Q3 (worst quarter) DD 16.5% → 3.0% (-82%)
- **Independent 4H Rubber Band (v7.0)**: 4h 타임프레임 독립 평균회귀 트레이딩
  - 4h-only: Ret +52.2%, DD 4.8%, Calmar 10.87
  - 15m + 4h 하이브리드: Ret +139.8%, DD 4.3%, Calmar 32.42
- **MTF Leverage Boost (v7.0)**: MTF 확인 통과 트레이드에 2.0x 레버리지 부스트 (실효 3.0x)
- **Confidence-Scored Entry (v6.0)**: 멀티 증거 기반 컨피던스 필터링 + 졸업 레버리지
  - RSI, 거래량 스파이크, 테이커 매수/매도 비율, SMA 거리, BTC 모멘텀, MACD 히스토그램(v9.0), 스토캐스틱 크로스(v9.0) — 최대 7가지 증거
  - min_confidence=1: MTF 4h 트렌드 필터가 실질적 품질 게이트 역할 (v10.0, +86.9% ret)
  - 선택적 졸업 레버리지: 증거 많을수록 포지션 확대
- **Regime-Switch Dual Entry (v5.0 base)**: BTC 레짐에 따라 다른 진입 알고리즘 사용
  - Risk-On (BTC > SMA96): Classic Pullback — 자산 자체 상승추세에서 조정 시 매수
  - Risk-Off (BTC < SMA96): Sympathy Pullback — BTC 하락 시 동반 하락하는 자산 매수
- **Fibonacci Stop/Target (v9.0)**: stop=0.786x (Fib deep correction), target=38.2% (Fib golden ratio) → R:R 0.486
- **Flat 1.5x Leverage**: R:R 1.25로 레버리지 정당화
  - R:R ≥ 0.30 안전 게이트 유지
- **Dynamic Leverage**: Move-confidence 기반 동적 레버리지 (R:R 게이트와 연동)
- **MTF (v7.0)**: 4h SMA(12) 트렌드 확인 활성화 — MDD -74%, Calmar +275%
- **Live/Backtest Parity (v7.0)**: volume/taker 데이터 _scan_entries() 연결 + 4h kline 다운로드 + main() v7.0 설정 전달
- **365-Day Validated**: 30 심볼 × 35,040봉, 7/7 견고성 점수, 9/10 롤링 윈도우 수익
- **v7.0 MTF Numba 함수**: `apply_mtf_confirmation()`, `apply_mtf_leverage()`, `compute_signals_4h_independent()` (Numba 가속)
- **v9.0 S+R+M Numba 함수**: `_compute_ema_inline()`, `_compute_stochastic_inline()`, `_compute_macd_inline()`, `compute_signals_v9()` (Numba 가속)
- **Volatility Selection**: 30일 실현변동성 기준 Top-N 심볼 동적 선택 (10일마다 리밸런싱)
- **DD-Managed Portfolio**: 역변동성 가중 + 포트폴리오 DD 기반 노출 축소
- **Numba 가속**: 신호 생성, 백테스트, 레버리지 스케일링, 변동성 계산 모두 `@numba.jit(nopython=True)` 컴파일
- **Alpha Engine**: 멀티팩터 신호 엔진 (볼륨, 모멘텀, ATR 적응, 트레일링 스탑, 테이커 비율)
- **LimitChaser 실행**: maker-only 체결로 슬리피지 최소화
- **위험 관리**: Max DD 차단기, 일일 손실 제한, 포지션 사이징 (1% risk/trade)
- **상태 복구**: 크래시 후 포지션 자동 복구 (atomic JSON write)

### 핵심 아키텍처
- **실거래/백테스트 일체형**: 동일한 Frozen z-score 로직이 백테스트와 실거래에서 동작
- **모듈형 설계**: 전략, 필터, 실행 컴포넌트 조합 가능
- **타입 안전성**: 전체 코드베이스에 타입 힌트 적용
- **안전장치 내장**: Market Fallback 검증, 알림 스로틀링, avg_price=0 거부 등

### 데이터 관리
- **2단계 캐싱**: 메모리(LRU) + 디스크(압축) 캐싱 시스템
- **멀티 타임프레임**: 1m, 5m, 15m, 1h, 4h, 1d 데이터 지원
- **바이낸스 연동**: Rate limiting과 배치 처리 지원

### 전략 프레임워크
- **Rubber Band Mean-Reversion (v11.1)**: 3.2%+ 급등/급락 후 피보나치 0.382 되돌림 포착 + MACD/Stochastic 컨플루언스 + Top-20 심볼 화이트리스트
- **Elliott Wave Strategy (v11.6)**: ZigZag ABC 패턴 + 5차원 품질 스코어링 + Timing Alpha Filter + Risk-Sized Leverage + 50% 부분 익절 + 브레이크이븐 스탑
  - 4h 타임프레임, EMA(20/50/100/200/500), RSI(14/9), MACD(12/26/9), Stochastic(14/3)
  - **Timing Alpha (v11.6)**: W3/W1 가속도 + 정정/충격 시간비율 + W3/W1 바 비율 필터 (Calmar 5.10, Sharpe 1.69)
  - **Quality Scoring (v11.5)**: W2 retrace/W3 extension/B retrace/C:A ratio/구조적 규칙 5차원 0-100점 (임계: 40점)
  - **Risk-Sized Leverage (v11.5)**: `lev = min(2.0, 2%/stop_dist)` — 트레이드당 최대 2% 자기자본 위험으로 일관된 리스크 관리
  - **Tight Stops at C (v11.5)**: 정정파 C 완료 지점 ±0.5% — 틀리면 즉시 소규모 손절, 맞으면 높은 RR
  - **adj_close (v11.5)**: 정밀한 백테스트를 위한 인트라바 체결가 보정
  - BTC 매크로 바이어스: 200/500 EMA 기반 롱/숏 필터링
  - **Sweep-Optimized (2026-02-18)**: zigzag 0.03→0.02, trail 2.0→1.5 (48+36 combos, 8 symbols)
  - 백테스트 (365일, 8심볼): 8 trades, WR 75%, +4.84%, MaxDD 1.96%, Sharpe 1.11, Calmar 2.47
  - Numba 가속 지표: EMA, RSI, MACD, Stochastic, ATR, ZigZag
- **Alpha Engine**: 볼륨/모멘텀/ATR 필터 토글 가능한 멀티팩터 엔진
- **Hybrid Strategy**: 트렌드 방향 + 러버밴드 타이밍 ("상승장에서 저가 매수")
- **Volatility Selector**: 실현변동성 기준 심볼 동적 선택 (10일 리밸런싱)
- **Walk-Forward 검증**: 365일 롤링 윈도우, v11.1 Top-20: 3384% ret, Calmar 155.22

### 실거래 안전장치
- **Server-Side Stop-Loss (v11.1)**: 바이난스 STOP 주문으로 클라이언트 크래시/네트워크 단절 시 손절 보장
- **Leverage Verification (v11.1)**: 레버리지 설정 실패 시 주문 중단 (알 수 없는 레버리지로 거래 방지)
- **Limit Chaser**: 지정가 추격 주문으로 슬리피지 최소화
- **Market Fallback 검증**: ACK 응답 감지 → 자동 재조회 (유령 체결 방지)
- **State Persistence**: 포지션 상태 JSON 저장 → 크래시 후 자동 복구 + 거래소 대조
- **Max Drawdown Breaker**: 세션 HWM 대비 10% 드로다운 시 전체 청산 + 자동 정지
- **Daily Loss Limit**: 당일 잔고 대비 5% 손실 시 신규 진입 중단
- **Consecutive Error Guard**: 연속 5회 에러 시에만 전체 청산 (일시적 API 에러에 과민 반응 방지)
- **Close Failure Retention**: 청산 실패 시 슬롯 유지 → 다음 사이클 재시도 (유령 포지션 방지)
- **알림 스로틀링**: 슬롯 포화/잔고 부족 알림 중복 전송 방지
- **Discord Bot**: 원격 모니터링 및 명령 (포지션 조회, 강제 청산 등)
- **Log Rotation**: 50MB/5파일 로테이션 (디스크 무한 성장 방지)

### 백테스팅
- **롤링 윈도우**: Formation/Trading 기간 시뮬레이션
- **종합 지표**: Sharpe, Sortino, Calmar, Profit Factor 등

## 📦 설치

\`\`\`bash
git clone https://github.com/younghwan91/quantbox.git
cd quantbox

# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e .
\`\`\`

## 🚀 빠른 시작

### Rubber Band Scalper (권장)

```bash
# Rubber Band (기본 전략)
python -m quantbox.scalper --paper

# Elliott Wave 전략
python -m quantbox.scalper --strategy elliott --paper

# 한 사이클만 실행
python -m quantbox.scalper --paper --once

# LIVE trading (실거래!)
python -m quantbox.scalper
python -m quantbox.scalper --strategy elliott
```

```python
# 백테스트
from quantbox.strategy.rubberband import compute_signals, RubberBandConfig
from quantbox.backtest.vectorized_backtest import VectorizedBacktest, BacktestConfig

bt = VectorizedBacktest(BacktestConfig())
signals = compute_signals(close_prices, lookback=8, threshold=0.025,
                          target_retrace=0.5, stop_mult=1.0, max_hold=8)
result = bt.run_single(close_prices, signals, symbol="BTCUSDT")
print(f"Return: {result.total_return_pct:+.2f}%, Sharpe: {result.sharpe:.2f}")
```

### 페어 트레이딩 (레거시)

> 페어 트레이딩 코드는 `quantbox-pairs` 별도 프로젝트로 분리되었습니다.

## 📁 프로젝트 구조

\`\`\`
quantbox/
├── core/                    # 핵심 컴포넌트
│   ├── config.py           # 설정 관리
│   ├── events.py           # 이벤트 시스템
│   ├── limit_chaser.py     # 지정가 추격 주문
│   ├── settings.py         # 환경변수 검증 (pydantic-settings)
│   └── state_persistence.py # 상태 영속화 (JSON, 크래시 복구)
├── data/                    # 데이터 처리
│   ├── cache.py            # 2단계 캐싱 (LRU + gzip)
│   ├── downloader.py       # 바이낸스 데이터 다운로드
│   └── real_data_loader.py # 실제 데이터 로더
├── strategy/               # 전략 컴포넌트
│   ├── rubberband.py       # Rubber Band Scalper (기본, Numba 가속)
│   ├── elliott.py          # Elliott Wave 전략 (Numba 가속)
│   ├── alpha_engine.py     # 멀티팩터 Alpha Engine (향상 RB + 하이브리드)
│   └── vol_selector.py     # 변동성 기반 심볼 선택기
├── backtest/               # 백테스팅 프레임워크
│   └── vectorized_backtest.py  # 벡터화 백테스트 (단일 자산, Numba)
├── notifications/          # 알림 시스템
│   ├── bot.py              # Discord Bot (양방향)
│   └── discord.py          # Discord Webhook (단방향)
└── scalper.py              # 멀티 전략 실거래 트레이더 (--strategy rubberband|elliott)

scripts/                    # 유틸리티 스크립트
├── download_15m_data.py    # 15분 캔들 데이터 캐시 갱신
├── download_extended_data.py # 장기 과거 데이터 다운로드
├── download_mtf_data.py    # 멀티 타임프레임 데이터 다운로드
├── elliott_backtest.py     # Elliott Wave 365일 백테스트
├── elliott_param_sweep.py  # Elliott 글로벌 파라미터 서치 + AB 테스트
├── elliott_wave_timing_research.py  # Elliott 타이밍 알파 피처 리서치
├── elliott_timing_backtest.py       # Elliott 타이밍 알파 Round 1 백테스트
└── elliott_timing_backtest_r2.py    # Elliott 타이밍 알파 Round 2 정제
tests/                      # 유닛 테스트 (567개)
\`\`\`

## 📈 성과 지표 (v11.1: Critical Fixes + Pool Expansion, Top-20, 365일)

| 지표 | v7.0 Hybrid | v8.0 DynTarget | v9.0 Fib+SRM | v10.0 Optimized | v11.1 Top-20 ★ |
|------|-------------|----------------|--------------|--------------------|--------------------|
| Portfolio Return | +139.8% | +139.8% | +169.2% | +982.4% | **+3384.3%** |
| Max Drawdown | 4.3% | 4.3% | 11.0% | 19.6% | **21.8%** |
| Calmar Ratio | 32.42 | 30.29 | 15.38 | 4.78 | **155.22** |
| Sharpe Ratio | 5.23 | 5.23 | 5.23 | 5.23 | **5.02** |
| Win Rate | 63.3% | 63.3% | 63.3% | 71.1% | **69.7%** |
| Trades | 798 | 798 | 798 | 318 | **936** |
| Profit Factor | N/A | N/A | N/A | N/A | **2.66** |

> **v11.1 핵심**: Top-15→Top-20 Pool Expansion (SHIBUSDT, LINKUSDT, ATOMUSDT, OPUSDT, DOTUSDT, HBARUSDT, ICPUSDT, BNBUSDT 추가). 5 slots 집중 전략이 8/10/15 slots 대비 압도적 성과. 서버사이드 STOP 손절로 클라이언트 크래시 방어.

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
