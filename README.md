# QuantBox 📈

**암호화폐 페어 트레이딩 프레임워크**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

백테스팅과 실거래를 통합한 암호화폐 페어 트레이딩 프레임워크

## 🌟 주요 기능

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
- **유연한 필터 파이프라인**: 페어 선별을 위한 조합 가능한 필터
  - 상관관계 필터
  - 공적분 (Engle-Granger) 필터
  - Half-Life 필터
  - Hurst 지수 필터
  - 정상성 (ADF) 필터
  - 거래량 필터
- **페어 선택**: 통계적 페어 식별 및 순위화
- **Dynamic Exit**: Half-life 기반 청산 z-score 자동 조절 (빠른 페어 → tight exit, 느린 페어 → loose exit)
- **Leverage-Based Risk Parity**: 페어별 spread_std에 따라 레버리지를 조절하여 R=target_r_pct(기본 2%)를 정확히 달성 (고변동 → 낮은 레버리지, 저변동 → 높은 레버리지)
- **Dynamic Leverage**: 페어별 spread_std, HR, 비용을 반영하여 레버리지 자동 계산 (max 20x cap, 자본 100% 활용)
- **Z-Score Cap**: |z| > 4.0 진입 거부 (레짐 변화 방어)
- **Vol Expansion Gate**: 최근 변동성/포메이션 변동성 > 2.0 진입 거부
- **BTC Crash Gate**: BTC 급락/급등(±3%) 감지 시 전체 진입 차단 (매크로 방어)
- **Trailing Stop (Ratchet Stop)**: z가 개선 후 entry 수준으로 역전 시 조기 청산 — "Revert-Then-Reverse" 패턴 방어 (Leung & Zhang 2021)

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

### 기본 백테스트

\`\`\`python
from quantbox.data import DataDownloader
from quantbox.backtest import run_continuous_backtest, ContinuousBacktestConfig

# 데이터 다운로드
downloader = DataDownloader()
data = downloader.download_multiple(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
    timeframe='1h',
    days=90
)

# 백테스트 설정
config = ContinuousBacktestConfig(
    initial_capital=10000.0,
    trading_fee=0.0005,
    entry_zscore=2.0,
    exit_zscore=0.5,
)

# 백테스트 실행 (Frozen z-score + Vol-inverse sizing)
results = run_continuous_backtest(config, data)
print(results['metrics'].summary())
\`\`\`

### 페어 선택

\`\`\`python
from quantbox.strategy import FilterPipeline
from quantbox.strategy.filters import (
    CorrelationFilter, 
    CointegrationFilter,
    HalfLifeFilter
)

# 필터 파이프라인 생성
pipeline = FilterPipeline([
    CorrelationFilter(min_correlation=0.7),
    CointegrationFilter(max_pvalue=0.05),
    HalfLifeFilter(min_halflife=5, max_halflife=50),
])

# 페어 필터링
result = pipeline.apply(series1, series2)
print(f"통과 여부: {result.passed}")
\`\`\`

## 📁 프로젝트 구조

\`\`\`
quantbox/
├── core/                    # 핵심 컴포넌트
│   ├── config.py           # 설정 관리
│   ├── events.py           # 이벤트 시스템
│   ├── limit_chaser.py     # 지정가 추격 주문
│   ├── settings.py         # 환경변수 검증 (pydantic-settings)
│   └── trader_slot.py      # 트레이더 슬롯 관리 (Dynamic Leverage)
│   └── state_persistence.py # 상태 영속화 (JSON, 크래시 복구)
├── data/                    # 데이터 처리
│   ├── cache.py            # 2단계 캐싱 (LRU + gzip)
│   ├── downloader.py       # 바이낸스 데이터 다운로드
│   ├── preprocessor.py     # 데이터 전처리
│   └── real_data_loader.py # 실제 데이터 로더
├── strategy/               # 전략 컴포넌트
│   ├── base.py             # 전략 베이스 클래스
│   ├── filters.py          # 필터 파이프라인
│   └── pair_selector.py    # 페어 선택기
├── backtest/               # 백테스팅 프레임워크
│   ├── continuous_backtest.py  # 연속 백테스트 엔진 (Vol-inverse sizing, Trailing stop)
│   └── metrics.py          # 성과 지표
├── notifications/          # 알림 시스템
│   ├── bot.py              # Discord Bot (양방향)
│   └── discord.py          # Discord Webhook (단방향)
├── live_trader.py          # 실거래 트레이더 (Multi-slot)
└── utils/                  # 유틸리티
    ├── numba_ops.py        # Numba JIT 가속 함수
    └── costs.py            # 거래 비용 모델

scripts/                    # 스크립트
    ├── download_15m_data.py    # 15m 데이터 다운로드
    ├── sweep_max_holding.py    # Max holding 스윕
    └── dynamic_exit_ab_test.py # Dynamic exit A/B 테스트

tests/                      # 유닛 테스트 (648개, 기본 실행 644 + slow 4)
\`\`\`

## 📈 성과 지표

| 지표 | 설명 |
|------|------|
| Total Return | 기간 전체 수익률 |
| Sharpe Ratio | 위험 조정 수익률 |
| Sortino Ratio | 하방 위험 조정 수익률 |
| Max Drawdown | 최대 낙폭 |
| Win Rate | 수익 거래 비율 |
| Profit Factor | 총 수익 / 총 손실 |

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
