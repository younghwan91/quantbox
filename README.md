# QuantBox 📈

**암호화폐 Rubber Band Scalper — Binance USDT-M Futures**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

단일 자산 평균회귀 스캘핑 프레임워크 (15분봉, Numba 가속)

## 🌟 주요 기능

### 🔥 Rubber Band Scalper (v4.4 — Optimized R:R + Flat Leverage)
- **Alpha**: 4.5% 이상 급등/급락 후 반등을 포착하는 단일 자산 평균회귀 전략
- **Optimized Stop/Target (v4.4)**: stop=0.9x, target=40% → 실현 R:R 1.10, DD 12.7%
  - 타이트한 스톱(0.9x)이 DD를 15.4%→12.7%로 자연 억제
  - 넓은 타겟(40%)이 더 많은 리버전 수익 포착
  - Partial exit 비활성화: 타이트 스톱에서 오히려 성능 저하 (Calmar 3.75 vs 4.96)
- **Flat 1.5x Leverage (v4.4)**: R:R 1.10으로 레버리지 정당화
  - 1.5x에서 Calmar 최대 (4.98 ★), 2x부터 DD 급증
  - R:R ≥ 0.30 안전 게이트 유지 (현재 R:R=0.44, 항상 통과)
- **BTC Directional Alignment**: BTC 방향 정렬 필터 — 시장 전체가 같은 방향일 때만 진입
  - 개별 종목의 특이적 움직임(비복귀 가능) 대비 시장 전체 움직임(복귀 가능)을 구분
- **Dynamic Leverage**: Move-confidence 기반 동적 레버리지 (R:R 게이트와 연동)
- **MTF**: 비활성화 (장기 성능 저하 확인 — A/B 테스트 결과 No MTF이 최적)
- **Live/Backtest Parity**: entry_stop_mult 저장 + 동일 레버리지 계산으로 일치 보장
- **365-Day Validated**: 30 심볼 × 35,040봉, 7/7 견고성 점수, 9/10 롤링 윈도우 수익
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
- **Rubber Band Mean-Reversion**: 4.5%+ 급등/급락 후 반등 포착 (기본 전략, 365일 검증)
- **Alpha Engine**: 볼륨/모멘텀/ATR 필터 토글 가능한 멀티팩터 엔진
- **Hybrid Strategy**: 트렌드 방향 + 러버밴드 타이밍 ("상승장에서 저가 매수")
- **Volatility Selector**: 실현변동성 기준 심볼 동적 선택 (10일 리밸런싱)
- **Walk-Forward 검증**: 365일 롤링 윈도우 9/10 수익, Calmar 1.13, Sharpe 0.81

### 실거래 안전장치
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
# Paper trading (시뮬레이션)
python -m quantbox.scalper --paper

# 한 사이클만 실행
python -m quantbox.scalper --paper --once

# LIVE trading (실거래!)
python -m quantbox.scalper
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
│   ├── alpha_engine.py     # 멀티팩터 Alpha Engine (향상 RB + 하이브리드)
│   └── vol_selector.py     # 변동성 기반 심볼 선택기
├── backtest/               # 백테스팅 프레임워크
│   └── vectorized_backtest.py  # 벡터화 백테스트 (단일 자산, Numba)
├── notifications/          # 알림 시스템
│   ├── bot.py              # Discord Bot (양방향)
│   └── discord.py          # Discord Webhook (단방향)
└── scalper.py              # Rubber Band 실거래 트레이더

scripts/                    # 연구/분석 스크립트
tests/                      # 유닛 테스트 (338개)
\`\`\`

## 📈 성과 지표 (v4.4: Optimized R:R + Flat 1.5x Leverage, 365일 검증)

| 지표 | v4.2 Baseline | v4.3 (Partial) | v4.4 1x | v4.4 1.5x ★ |
|------|--------------|----------------|---------|-------------|
| Portfolio Return (Top-10) | +58.1% | +50.4% | +62.9% | **+95.0%** |
| Max Drawdown | 15.4% | 15.9% | **12.7%** | 19.1% |
| Calmar Ratio | 3.78 | 3.17 | 4.96 | **4.98** |
| Sharpe Ratio | 1.63 | 1.54 | **1.67** | 1.61 |
| Win Rate | 61.6% | 65.0% | 59.3% | 59.2% |
| R:R (Realized) | 0.99 | 0.90 | **1.10** | **1.10** |
| Rolling 90-day | 12/13 | 11/13 | **12/13** | **12/13** |

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
