# QuantBox 📈

**바이낸스 USDT-M 선물 변동성 압축(VR) 스퀴즈 시스템**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

바이낸스 USDT-M 선물에서 변동성 압축 뒤의 폭발을 노리는 시스템이다. VR Squeeze 와 MA Cluster Squeeze 두 신호가 슬롯 풀을 공유하며,
같은 전략 코드가 Numba 가속 백테스트와 페이퍼·실거래 엔진에서 동작한다. 설정은 pydantic-settings 로 환경변수에서 읽는다.
전략별 파라미터는 [docs/STRATEGIES.md](docs/STRATEGIES.md) 와 `quantbox/strategy/*/config.py` 에 있다.

> 교육·연구 목적의 코드다. 실행 결과를 수익의 근거로 삼지 않는다. 맨 아래 면책 조항 참조.

## 전략 구성

### VolStraddle — VR 압축 감지
- **Volatility Ratio(VR) 압축 감지**: 단기 ATR 대 장기 ATR 비율이 임계값 밑으로 내려가면 변동성 압축으로 본다.
- **멀티 타임프레임**: 15m/1h/4h 신호를 독립으로 보고 높은 타임프레임을 우선한다.
- **D_mom8**: 압축 신호가 나오면 8바 모멘텀 방향으로 단방향 진입한다.
- **S_d8 스트래들**: 압축이 일정 바 수 지속되면 양방향 동시 진입한다(Hedge Mode).
- **청산**: 손절, 안전 상한 익절, 트레일링 스탑, 최대 보유 시간.
- **룩어헤드 방지**: 완성된 바만 참조하고 진행 중인 바는 제외한다.

### MA Cluster Squeeze — 4h 이동평균 수렴 돌파
- **MA 클러스터 감지**: SMA(20/50/100/200) 4선이 좁게 수렴한 상태가 일정 시간 이어진 뒤의 돌파를 포착한다.
- **4h 전용**: 상위 타임프레임에서만 동작하고 VR 신호와 거의 겹치지 않는다.
- **방향**: 클러스터를 벗어나는 쪽의 모멘텀 방향으로 진입한다.
- **필터**: 너무 오래된 클러스터와 약한 모멘텀 신호를 차단한다.

### 레거시 전략 (비프로덕션)
- **Rubber Band Scalper**: ATR 적응형 평균회귀, 거래량 클라이맥스 필터 (`quantbox/strategy/rubberband/`).

## 핵심 아키텍처
- **실거래/백테스트 일체형**: 같은 신호 로직이 백테스트와 실거래에서 동작한다.
- **모듈형 설계**: 전략, 필터, 실행 컴포넌트를 조합한다.
- **타입 안전성**: 전체 코드베이스에 타입 힌트를 적용하고 전략은 `TradingStrategy` Protocol 로 확장한다.

### 데이터 관리
- **2단계 캐싱**: 메모리(LRU) + 디스크(압축).
- **멀티 타임프레임**: 1m, 5m, 15m, 1h, 4h, 1d.
- **바이낸스 연동**: 호출 제한과 배치 처리.

### 실거래 안전장치
- **서버사이드 손절**: 바이낸스 STOP 주문으로 클라이언트 크래시·네트워크 단절에도 손절을 보장한다.
- **레버리지 검증**: 레버리지 설정에 실패하면 주문을 중단한다.
- **Limit Chaser**: 지정가 추격 주문으로 슬리피지를 줄인다.
- **Market Fallback 검증**: ACK 응답을 감지하면 재조회해 유령 체결을 막는다.
- **상태 영속화**: 포지션 상태를 JSON 으로 저장해 크래시 뒤 거래소와 대조해 복구한다.
- **Max Drawdown Breaker · Daily Loss Limit**: 세션 고점 대비 낙폭과 일일 손실 한도를 넘으면 청산·신규 진입을 멈춘다.
- **Consecutive Error Guard · Close Failure Retention**: 연속 에러 시 전체 청산, 청산 실패 시 슬롯을 유지해 재시도한다.
- **Rate Limiter + Circuit Breaker · In-Flight Order Tracker**: API 호출 제한과 미체결 주문 추적.
- **ExposureGuard · ReconJournal**: 세션 노출도 상한과 SQLite 감사 로그.
- **Config Validators**: SL<TP, 레버리지≥1 같은 설정 교차 검증.
- **Discord Bot**: 원격 모니터링과 명령(포지션 조회, 강제 청산 등).

### 백테스팅 (Dual-Engine)
- **Signal-Array Engine** (`vectorized_backtest.py`): ±1 포지션 어레이 기반 단일자산 신속 스캔.
- **Trade-Event Engine** (`trade_sim.py`): 이산 트레이드 시뮬레이션. High/Low 기반 인트라바 SL/TP, N-슬롯 포트폴리오 경합, Numba 가속.
  `sim_directional_trades()` / `sim_straddle_trades()` → `collect_trades()` → `portfolio_sim()` → `compute_metrics()` 순서로 쓴다.
- **롤링 윈도우**: Formation/Trading 기간 시뮬레이션.
- **지표**: Sharpe, Sortino, Calmar, Profit Factor, Bootstrap CI.

## 📦 설치

```bash
git clone https://github.com/younghwan91/quantbox.git
cd quantbox

# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e .
```

## 🔐 설정 (비밀값은 환경변수로만)

API 키·디스코드 토큰·웹훅 URL·채널 ID 는 코드나 git 에 두지 않고 환경변수(또는 `.env`)에서만 읽는다.
`.env.example` 을 `.env` 로 복사해 `BINANCE_API_KEY`, `BINANCE_API_SECRET`(필수), `DISCORD_WEBHOOK_URL`,
`DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`(선택)를 채운다. `.env` 는 `.gitignore` 대상이다.
필수 값이 없거나 `.env.example` 의 자리표시자(`your_...`) 그대로면 설정 검증이 시작을 거부한다(페이퍼 모드 포함).
백테스트·테스트는 환경변수 없이 실행된다.

```bash
cp .env.example .env   # 편집기로 값 채우기 — 값은 절대 커밋하지 않는다
```

## 🚀 빠른 시작

```bash
# 페이퍼 트레이딩 (시뮬레이션, 주문 없음)
python -m quantbox.scalper --strategy vol_straddle --paper

# 한 사이클만 실행
python -m quantbox.scalper --strategy vol_straddle --paper --once
```

실거래는 `--paper` 없이 실행하며 실주문이 나간다. 실거래 전에 Binance Hedge Mode 를 켜고 `.env` 를 채워야 한다.
자세한 설정은 [MANUAL.md](MANUAL.md) 참조.

## 📊 라이브 대시보드

브라우저에서 실시간으로 포지션/잔고/트레이드 현황을 볼 수 있습니다.

```bash
# 시작 (백그라운드)
bash scripts/start_dashboard.sh start

# 중지
bash scripts/start_dashboard.sh stop

# 상태 확인
bash scripts/start_dashboard.sh status
```

접속: `http://localhost:8501` (같은 PC) 또는 `http://[PC_IP]:8501` (같은 네트워크)

- 30초마다 자동 새로고침
- 현재 포지션, 잔고, 전략별 성과, 최근 트레이드 확인

### 트레이드 내역 CLI

```bash
python3 scripts/journal.py              # 오늘 트레이드 + 전체 요약
python3 scripts/journal.py symbols      # 심볼별 손익 랭킹
python3 scripts/journal.py last 10      # 최근 10건
python3 scripts/journal.py strategy     # 전략별 성과
```

## 📁 프로젝트 구조

```
quantbox/
├── core/             # 설정(pydantic-settings), 이벤트 루프, 상태 영속화, 레이트 리미터, Discord 커맨드, 트레이드 저널
├── execution/        # 서버사이드 브라켓, 포지션, 거래소 대조, 미체결 추적, 노출 상한, 프리플라이트, 리스크 가드
├── data/             # 바이낸스 다운로더, 2단계 캐시, 데이터 로더
├── strategy/         # vol_straddle(VR 압축 + MA 클러스터), rubberband(레거시), vol_selector, TradingStrategy Protocol
├── backtest/         # vectorized_backtest(시그널 어레이), trade_sim(트레이드 이벤트)
├── notifications/    # Discord 봇(양방향)·웹훅(단방향)
├── utils/            # 진단 로깅
└── scalper.py        # 코어 엔진 진입점
scripts/              # 데이터 다운로드, 계좌·포지션 점검, 저널 조회, 대시보드, 리포트 생성
research/             # 리서치 스크립트(r번호 체계, active/ 와 archive/)
autoresearch/         # 자동 탐색 스크립트
docs/                 # STRATEGIES.md, 백테스트 리포트
tests/                # 유닛 테스트
```

## 🧪 테스트

```bash
uv run pytest tests/ -v
```

## 라이선스

[MIT](LICENSE) © 2026 Younghwan Chae

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
