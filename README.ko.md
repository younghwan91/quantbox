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

[빠른 시작](#빠른-시작) · [가이드](docs/guide.ko.md) · [전략 레퍼런스](docs/STRATEGIES.md) · [매뉴얼](MANUAL.md) · [English](README.md)

</div>

## 빠른 시작

```bash
git clone https://github.com/younghwan91/quantbox.git && cd quantbox
uv sync
cp .env.example .env   # BINANCE_API_KEY, BINANCE_API_SECRET 입력
python -m quantbox.scalper --strategy vol_straddle --paper --once
```

한 사이클 동안 완성된 캔들을 평가하고 실주문 없이 신호를 기록합니다. 단위 테스트와 백테스트는 `.env` 설정 없이 실행됩니다(`uv run pytest tests/ -v`).

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

## 기여

Pull Request를 열기 전에 테스트 스위트를 실행해 주세요.

```bash
uv run pytest tests/ -v
```

이슈 제보와 Pull Request를 환영합니다.

## 라이선스

[MIT](LICENSE) © 2026 Younghwan Chae
