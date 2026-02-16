# QuantBox 📈

**암호화폐 Rubber Band Scalper — Binance USDT-M Futures**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

단일 자산 평균회귀 스캘핑 프레임워크 (15분봉, Numba 가속)

## 🌟 주요 기능

### 🔥 Rubber Band Scalper (v3.2 — Multi-Timeframe)
- **Alpha**: 2~3% 이상 급등/급락 후 반등을 포착하는 단일 자산 평균회귀 전략
- **MTF Confirmation**: 고시간프레임(12h SMA) 트렌드 필터로 역트렌드 진입 차단 (Calmar 7.6× 개선)
- **Volatility Selection**: 30일 실현변동성 기준 Top-N 심볼 동적 선택 (10일마다 리밸런싱)
- **Time-Decaying Stop**: 보유 시간이 길어질수록 스탑 레벨을 점진적으로 타이트닝 (`stop_decay`)
- **DD-Managed Portfolio**: 역변동성 가중 + 포트폴리오 DD 기반 노출 축소
- **Numba 가속**: 신호 생성, 백테스트, 변동성 계산 모두 `@numba.jit(nopython=True)` 컴파일
- **Walk-Forward 검증**: 4-window walk-forward, OOS 3/4 수익, max OOS DD 7.02%
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
- **Rubber Band Mean-Reversion**: 2.5%+ 급등/급락 후 반등 포착 (기본 전략)
- **Alpha Engine**: 볼륨/모멘텀/ATR 필터 토글 가능한 멀티팩터 엔진
- **Hybrid Strategy**: 트렌드 방향 + 러버밴드 타이밍 ("상승장에서 저가 매수")
- **Volatility Selector**: 실현변동성 기준 심볼 동적 선택 (10일 리밸런싱)
- **Walk-Forward 검증**: OOS +34.31% (60일), 연환산 +502%, Sharpe 4.67

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
tests/                      # 유닛 테스트 (279개)
\`\`\`

## 📈 성과 지표 (v3.2: MTF + decay=0.3, Top-3)

| 지표 | 결과 |
|------|------|
| Portfolio Return (MTF) | +17.32% (90일, top-3 vol-selected) |
| Sharpe Ratio | 3.41 |
| Max Drawdown | 5.96% |
| Calmar Ratio | 2.91 (baseline 0.38 대비 7.6× 개선) |
| Win Rate | 58.9% |
| Coins Profitable | 18/30 (60%) |
| Total Trades (30코인) | 1,598 (baseline 4,409 대비 36%) |
| Time Window Robustness | 3/5 windows MTF 우위 |

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
