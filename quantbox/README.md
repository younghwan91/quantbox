# QuantBox 📈

**암호화폐 페어 트레이딩 프레임워크**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

백테스팅과 실거래를 통합한 암호화폐 페어 트레이딩 프레임워크

## 🌟 주요 기능

### 핵심 아키텍처
- **이벤트 기반 엔진**: 백테스트와 실거래에서 동일한 코드 사용
- **모듈형 설계**: 전략, 필터, 실행 컴포넌트 조합 가능
- **타입 안전성**: 전체 코드베이스에 타입 힌트 적용

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
- **칼만 필터**: 온라인 헤지 비율 추정

### 백테스팅
- **롤링 윈도우**: Formation/Trading 기간 시뮬레이션
- **Walk-Forward 분석**: Out-of-Sample 검증
- **몬테카를로 시뮬레이션**: 강건성 테스트
- **종합 지표**: Sharpe, Sortino, Calmar, Profit Factor 등

### 최적화
- **다양한 알고리즘**:
  - Grid Search
  - Random Search
  - Differential Evolution
  - Bayesian Optimization
- **민감도 분석**: 파라미터 강건성 테스트
- **다목적 최적화**: 복합 최적화 기준

## 📦 설치

\`\`\`bash
git clone https://github.com/younghwan91/trading_code.git
cd trading_code
pip install -e .
\`\`\`

## 🚀 빠른 시작

### 기본 백테스트

\`\`\`python
from quantbox import Config, BacktestRunner
from quantbox.data import DataDownloader

# 데이터 다운로드
downloader = DataDownloader()
data = downloader.download_multiple(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
    timeframe='1h',
    days=90
)

# 백테스트 설정
from quantbox.backtest.runner import BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,
    trading_fee=0.0005,
    entry_zscore=2.0,
    exit_zscore=0.5,
)

# 백테스트 실행
runner = BacktestRunner(config)
results = runner.run_backtest(data)
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
├── core/                    # 핵심 엔진 컴포넌트
│   ├── config.py           # 설정 관리
│   ├── engine.py           # 트레이딩 엔진
│   ├── events.py           # 이벤트 시스템
│   ├── execution.py        # 주문 실행
│   └── portfolio.py        # 포지션 관리
├── data/                    # 데이터 처리
│   ├── cache.py            # 캐싱 시스템
│   └── downloader.py       # 데이터 다운로드
├── strategy/               # 전략 컴포넌트
│   ├── filters.py          # 필터 구현
│   └── pair_strategy.py    # 페어 트레이딩 전략
├── backtest/               # 백테스팅 프레임워크
│   ├── runner.py           # 백테스트 실행
│   ├── metrics.py          # 성과 지표
│   └── walk_forward.py     # Walk-Forward 테스트
├── optimization/           # 파라미터 최적화
│   └── optimizer.py        # 최적화 알고리즘
└── utils/                  # 유틸리티
    ├── numba_ops.py        # Numba JIT 가속 함수
    └── costs.py            # 거래 비용 모델

tests/                      # 유닛 테스트 (117개)
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
pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
