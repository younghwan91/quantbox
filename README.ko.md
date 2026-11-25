<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="quantbox" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

**바이낸스 USDT-M 선물 변동성 압축(VR Squeeze) 및 MA 클러스터 돌파 트레이딩 시스템**

Numba 백테스트와 실거래가 동일한 `TradingStrategy` 코드 경로에서 동작합니다. 교육·연구 목적의 코드입니다.

[![CI](https://github.com/younghwan91/quantbox/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/quantbox/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

[빠른 시작](#빠른-시작-api-키-불필요) · [가이드](docs/guide.ko.md) · [전략 레퍼런스](docs/STRATEGIES.md) · [매뉴얼](MANUAL.md) · [English](README.md)

</div>

## 빠른 시작 (API 키 불필요)

```bash
git clone https://github.com/younghwan91/quantbox.git && cd quantbox
uv sync --dev
uv run pytest tests/test_backtest_live_parity.py tests/test_backtest_plausibility.py -m "not slow" --no-cov -q
```

약 10초 안에 `38 passed`가 나옵니다. 첫 번째 파일은 실거래 신호 경로가 백테스트 엔진과 같은 판단(신호 방향, 멀티 타임프레임 게이팅, 레버리지, 진행 중인 캔들 제외)을 내리는지 검증합니다. 두 번째 파일은 룩어헤드가 의심되는 백테스트를 걸러내는 타당성 게이트를 검증합니다. 둘 다 메모리 내 테스트 데이터를 쓰므로 거래소 계정이나 내려받은 데이터가 필요하지 않습니다.

## 페이퍼 트레이딩 (Binance 키 필요)

```bash
cp .env.example .env   # BINANCE_API_KEY, BINANCE_API_SECRET 입력
python -m quantbox.scalper --strategy vol_straddle --paper --once
```

한 번의 페이퍼 트레이딩 사이클 동안 완성된 캔들을 평가하고 실주문 없이 신호를 기록합니다. 전체 테스트 스위트는 자격 증명 없이 실행됩니다(`uv run pytest tests/ -v`).

## 주요 기능

- **이중 변동성 돌파 신호** — `VolStraddle`(15m/1h/4h VR 압축, `D_mom8`·`S_d8`)과 `MA Cluster Squeeze`(4h SMA 20/50/100/200 수렴 돌파)가 10개 슬롯 풀을 공유합니다.
- **실거래·백테스트 일체형** — 동일한 `TradingStrategy` Protocol 신호 로직이 Numba 가속 백테스트 엔진(`vectorized_backtest.py`, `trade_sim.py`)과 실거래 스캘퍼에서 동작합니다.
- **다층 실거래 안전장치** — 서버사이드 STOP 브라켓, Limit Chaser, Market Fallback 검증, Max Drawdown Breaker, Daily Loss Limit, `ExposureGuard`, SQLite `ReconJournal`을 내장합니다.
- **운영 가시성** — Streamlit 실시간 대시보드, 트레이드 저널 CLI, Discord 양방향 봇·웹훅 알림을 제공합니다.

## 전략 구성

| 전략 | 타임프레임 | 동작 방식 | 레퍼런스 |
|---|---|---|---|
| **VolStraddle (`D_mom8` / `S_d8`)** | 15m · 1h · 4h | VR 압축(`ATR(8)/ATR(56)`) + 모멘텀 또는 스트래들 | [docs/STRATEGIES.md](docs/STRATEGIES.md) |
| **MA Cluster Squeeze** | 4h | SMA(20/50/100/200) 수렴 후 돌파 | [docs/STRATEGIES.md](docs/STRATEGIES.md) |
| **RubberBand (레거시)** | — | ATR 적응형 평균회귀 (비프로덕션) | `quantbox/strategy/rubberband/` |

## 환경 설정

API 키, Discord 토큰, 웹훅 URL, 채널 ID는 git에 포함되지 않는 `.env` 파일에서만 읽습니다. 필수 값이 비어 있거나 `.env.example`의 자리표시자(`your_...`) 그대로면 `--paper` 모드를 포함해 시작을 거부합니다.

전략 파라미터는 `.env`가 아닌 `quantbox/strategy/*/config.py`(`VolStraddleConfig`)에서 관리합니다.

## 대시보드 및 CLI

```bash
bash scripts/start_dashboard.sh start   # Streamlit 대시보드 시작 (http://localhost:8501)
bash scripts/start_dashboard.sh status  # 상태 확인
bash scripts/start_dashboard.sh stop    # 중지

python3 scripts/journal.py              # 오늘 트레이드 + 전체 요약
python3 scripts/journal.py symbols      # 심볼별 손익 랭킹
python3 scripts/journal.py last 10      # 최근 10건
python3 scripts/journal.py strategy     # 전략별 성과
```

## 유의사항

- **교육·연구 목적** — 실행 결과를 수익의 근거로 삼지 않으며, 암호화폐 선물 거래에는 상당한 손실 위험이 따릅니다.
- **실거래 전 Hedge Mode 필수** — `--paper` 없이 실행하면 실제 주문이 나갑니다. 실행 전 Binance Hedge Mode를 켜고 `.env`를 설정하세요.
- **완성된 바만 참조** — 룩어헤드 편향을 막기 위해 신호와 멀티 타임프레임 로직은 마감된 캔들만 읽습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [아키텍처 및 레퍼런스 가이드](docs/guide.ko.md) | 아키텍처 다이어그램, 안전장치, 듀얼 백테스트 엔진, 프로젝트 구조 |
| [전략 레퍼런스](docs/STRATEGIES.md) | 활성 전략, 진입·청산 규칙, 리스크 게이트 |
| [운영 매뉴얼](MANUAL.md) | 상세 설치, 실거래 체크리스트, 운영 명령어 |

## 검증 기록

표본 외 검증 결과는 좋아 보였다가 아니었던 사례까지 포함해 [crypto-quant-insights](https://github.com/younghwan91/crypto-quant-insights)에 공개합니다. 이 저장소의 백테스트 아카이브는 [docs/backtest_results](docs/backtest_results/README.md)에 있으며, 연구 기록일 뿐 운용 실적이 아닙니다.

## 같은 저자의 다른 저장소

- [crypto-quant-insights](https://github.com/younghwan91/crypto-quant-insights) - 이 시스템을 검증하며 얻은 차트 기반 교훈 열네 가지.
- [crypto-pair-trading](https://github.com/younghwan91/crypto-pair-trading) - 백테스트와 실거래 실행을 공유하는 이벤트 기반 통계적 차익거래 프레임워크.
- [kiwoom-client](https://github.com/younghwan91/kiwoom-client) - 국내 주식용 키움 REST API 클라이언트.
- [fin-checkup](https://github.com/younghwan91/fin-checkup) - DART 공시로 보는 국내 상장사 재무 건강검진.

## 기여

Pull Request를 열기 전에 테스트 스위트를 실행해 주세요.

```bash
uv run pytest tests/ -v
```

이슈 제보와 Pull Request를 환영합니다.

## 라이선스

[MIT](LICENSE) © 2026 Younghwan Chae
