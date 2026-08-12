# QuantBox 📈

**바이낸스 USDT-M 선물 VR 스퀴즈 시스템 — Dual-Axis Squeeze (VR + MA Cluster)**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-v14.35.0-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-1929%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-77%25-green.svg)]()
[![Branch](https://img.shields.io/badge/branch-main-blue.svg)]()
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

바이낸스 USDT-M 선물 전용 VR 압축 스퀴즈 시스템. VR Squeeze(30x)와 MA Cluster Squeeze(21x) 두 축이 10슬롯 공유 풀에서 동시 운용. Numba 가속 백테스트, pydantic-settings 환경 관리.

> **현재 상태**: Live Trading | Tag `v14.35.0` | 2339 tests passed

## 🌟 주요 기능

### 🎯 VolStraddle — VR Compression Detector (v14.35.0, 배포 중)
- **Volatility Ratio (VR) 압축 감지**: `VR = ATR(8)/ATR(56) < 0.50` — 변동성 압축 후 폭발 포착
- **멀티 타임프레임**: 15m/1h/4h 독립 신호, 높은 TF 우선 (4h > 1h > 15m)
- **D_mom8 방향**: 8바 모멘텀 기반 롱/숏 방향 결정 (D_mom8)
- **S_d8 스트래들**: VR < 0.55 × 4바 지속 → 양방향 동시 진입 (Hedge Mode)
- **파라미터**: SL=1%, 하드TP=20%(safety cap), Trailing activate +4.0%/distance 0.5%, MaxHold=48h
- **r106 Trail 조정**: D 4.0%/0.5%, S 3.0%/0.5%, MA 2.0%/0.5% (callbackRate≥0.5% 준수)
- **30x @10 slots** (L/N=3.0, Half-Kelly), 6년 연속 흑자, Bootstrap P(PF≥1.0)=100%
- **현재 엣지 (r73, 5.67yr, trail)**: D+S+MA 3전략, PF=2.002, Calmar=1695.6, 60 syms
- **제로 룩어헤드**: 완성된 바만 참조, 개발 중 바 제외, 교차 TF 감사 통과

### 📐 MA Cluster Squeeze — 4h SMA Convergence Breakout (V17, R45, 배포 중)
- **MA 클러스터 감지**: SMA(20/50/100/200) 4선 수렴도 ≤ 3% 지속 ≥ 8바 (32h) → 브레이크아웃 포착
- **4h 전용**: 상위 타임프레임에서만 동작, VR squeeze와 ~93% 비중복
- **Momentum(8) 방향**: 클러스터 이탈 시 모멘텀 방향으로 진입
- **Half-Kelly 레버리지**: 21x leverage × 2% SL → per-trade risk 4.2% (f*/2=4.33%)
- **Exit**: SL=2%, TP=8%(safety cap), Trail activate +2.0%/distance 0.5%, MaxHold=192바 (48h)
- **R45 Zombie Filter**: `max_duration=24` 바 초과 클러스터 차단 (오래된 클러스터 제거)
- **R45 Momentum Filter**: `abs(momentum) < 0.010` 차단 (거스름 없는 약한 신호 제거)
- **MA Cluster 현재 엣지**: N=2,329 accepted, PF=MA-class, 4h TF 독립, VR와 94.5% 비중복
- **결합 포트폴리오 (r73, 5.67yr)**: D+S+MA 3전략, PF=2.002, Calmar=1695.6, P(PF≥1.0)=100%, 5/5년 흑자

### 🔥 레거시 전략 (v14.0.0 이전, 비프로덕션)

- **Rubber Band Scalper (v12.6)**: ATR 적응형 평균회귀, Climax Volume Filter. v14.0.0에서 VolStraddle로 전략 전환.
- **Elliott Wave (v11.6)**: ZigZag ABC 패턴 + Timing Alpha. 레거시 보존.

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
- **Rubber Band Mean-Reversion (v12.2 E-3Slot)**: ATR 2.0× 적응형 임계값 + MACD/Stochastic 컨플루언스 + Top-50 심볼 + 3슬롯 분산
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
- **Walk-Forward 검증**: 365일 롤링 윈도우, v12.2 E-3Slot: 50 심볼, RB_MAX_CONCURRENT=3

### 실거래 안전장치
- **Server-Side Stop-Loss (v11.1)**: 바이난스 STOP 주문으로 클라이언트 크래시/네트워크 단절 시 손절 보장
- **Leverage Verification (v11.1)**: 레버리지 설정 실패 시 주문 중단 (알 수 없는 레버리지로 거래 방지)
- **Limit Chaser**: 지정가 추격 주문으로 슬리피지 최소화
- **Market Fallback 검증**: ACK 응답 감지 → 자동 재조회 (유령 체결 방지)
- **State Persistence**: 포지션 상태 JSON 저장 → 크래시 후 자동 복구 + 거래소 대조
- **Max Drawdown Breaker**: 세션 HWM 대비 10% 드로다운 시 전체 청산 + 자동 정지
  - v12.4.6: 0 포지션 재시작 시 HWM을 현재 잔고로 리셋 (확정 손실에 의한 팬텀 DD 방지)
- **Daily Loss Limit**: 당일 잔고 대비 5% 손실 시 신규 진입 중단
- **Consecutive Error Guard**: 연속 5회 에러 시에만 전체 청산 (일시적 API 에러에 과민 반응 방지)
- **Close Failure Retention**: 청산 실패 시 슬롯 유지 → 다음 사이클 재시도 (유령 포지션 방지)
- **알림 스로틀링**: 슬롯 포화/잔고 부족 알림 중복 전송 방지
- **Discord Bot**: 원격 모니터링 및 명령 (포지션 조회, 강제 청산 등)
- **Log Rotation**: 50MB/5파일 로테이션 (디스크 무한 성장 방지)
- **Config Validators (v14.22.0)**: 15개 교차 검증 — SL<TP, leverage≥1, trail_distance<trail_activate 등
- **Rate Limiter + Circuit Breaker (v14.22.0)**: API 호출 스로틀링. 연속 에러 시 자동 차단
- **In-Flight Order Tracker (v14.22.0)**: 미체결 주문 실시간 추적. 크래시 복구 시 자동 재동기화
- **ExposureGuard (v14.23.0)**: 세션 노출도 상한선 (`max_exposure_multiplier=35.0`). 초과 시 신규 진입 차단
- **ReconJournal (v14.23.0)**: SQLite 기반 10종 이벤트 감사 로그 (ghost, bracket repair, exposure 등)
- **TradingStrategy Protocol (v14.35.0)**: 14개 멤버 타입 안전 인터페이스. 전략 확장 표준화

### 백테스팅 (Dual-Engine)
- **Signal-Array Engine** (`vectorized_backtest.py`): RubberBand/Elliott 등 ±1 포지션 어레이 기반. Close-to-close PnL. 단일자산 신속 스캔용.
- **Trade-Event Engine** (`trade_sim.py`): VolStraddle/MA Cluster 등 이산 트레이드 시뮬레이션. High/Low 기반 인트라바 SL/TP. N-slot 포트폴리오 경합. Numba 가속.
  - `sim_directional_trades()` / `sim_straddle_trades()` → `collect_trades()` → `portfolio_sim()` → `compute_metrics()` 파이프라인
  - `sim_directional_trailing()` / `sim_straddle_trailing()` — trail-aware 시뮬레이션 (v14.35.0, r43에서 이전)
  - 리서치 스크립트의 canonical source (`scaling_r38.py`가 re-export)
- **롤링 윈도우**: Formation/Trading 기간 시뮬레이션
- **종합 지표**: Sharpe, Sortino, Calmar, Profit Factor, Bootstrap CI 등

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

```bash
# Paper trading (시뮬레이션)
python -m quantbox.scalper --strategy vol_straddle --paper

# 한 사이클만 실행
python -m quantbox.scalper --strategy vol_straddle --paper --once

# LIVE trading (실거래!)
python -m quantbox.scalper --strategy vol_straddle
```

> ⚠️ 실거래 전 `.env`에 `BINANCE_API_KEY/SECRET` 설정 및 Binance Hedge Mode 활성화 필요.  
> 자세한 설정: `MANUAL.md` 참조.

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
- CPU 사용: < 0.5% (대기 중), RAM: ~60MB — 트레이딩 성능에 영향 없음

### 트레이드 내역 CLI

```bash
python3 scripts/journal.py              # 오늘 트레이드 + 전체 요약
python3 scripts/journal.py symbols      # 심볼별 손익 랭킹
python3 scripts/journal.py last 10      # 최근 10건
python3 scripts/journal.py strategy     # 전략별 성과
```

## 📁 프로젝트 구조

\`\`\`
quantbox/
├── core/                    # 핵심 컴포넌트
│   ├── commands.py         # Discord 봇 커맨드 핸들러 (CommandsMixin)
│   ├── config.py           # 설정 관리
│   ├── events.py           # 이벤트 시스템
│   ├── limit_chaser.py     # 지정가 추격 주문
│   ├── settings.py         # 환경변수 검증 (pydantic-settings)
│   ├── state_persistence.py # 상태 영속화 (JSON, 크래시 복구)
│   ├── terminal_ui.py      # 터미널 UI (실시간 상태 표시)
│   ├── rate_limiter.py     # API 호출 스로틀링 + Circuit Breaker (v14.22.0)
│   └── heartbeat.py        # 메인 루프 정지 감지 + Discord 재연결 (v14.22.0)
├── execution/               # 주문 실행 및 포지션 관리
│   ├── brackets.py         # 서버사이드 브라켓 주문 (BracketMixin)
│   ├── positions.py        # 포지션 라이프사이클 (PositionMixin)
│   ├── reconciliation.py   # 거래소 상태 동기화 (ReconciliationMixin)
│   ├── inflight.py         # 미체결 주문 추적 + 크래시 복구 (v14.22.0)
│   ├── exposure.py         # 세션 노출도 상한선 ExposureGuard (v14.23.0)
│   └── recon_journal.py    # SQLite 감사 로그 10종 이벤트 (v14.23.0)
├── data/                    # 데이터 처리
│   ├── cache.py            # 2단계 캐싱 (LRU + gzip)
│   ├── downloader.py       # 바이낸스 데이터 다운로드
│   └── real_data_loader.py # 실제 데이터 로더
├── strategy/               # 전략 컴포넌트
│   ├── rubberband/         # Rubber Band Scalper 패키지
│   │   ├── config.py       # RubberBandConfig (전략 파라미터)
│   │   ├── signals.py      # Numba 시그널 엔진 (23개 함수)
│   │   └── strategy.py     # RubberBandStrategy 클래스
│   ├── vol_straddle/       # VolStraddle VR 압축 + MA 클러스터 스퀴즈
│   │   ├── config.py       # VolStraddleConfig (VR + MA cluster 파라미터)
│   │   ├── signals.py      # Numba VR/모멘텀/SMA/MA클러스터 (10개 함수)
│   │   └── strategy.py     # VolStraddleStrategy 클래스 (VR + MA cluster)
│   ├── elliott.py          # Elliott Wave 전략 (Numba 가속)
│   ├── alpha_engine.py     # 멀티팩터 Alpha Engine (향상 RB + 하이브리드)
│   ├── vol_selector.py     # 변동성 기반 심볼 선택기
│   └── protocol.py         # TradingStrategy Protocol 14멤버 (v14.35.0)
├── backtest/               # 듀얼 엔진 백테스팅 프레임워크
│   ├── vectorized_backtest.py  # Signal-Array 엔진 (RubberBand, close-to-close)
│   └── trade_sim.py            # Trade-Event 엔진 (VolStraddle, intra-bar SL/TP, slot 경합)
├── notifications/          # 알림 시스템
│   ├── bot.py              # Discord Bot (양방향)
│   └── discord.py          # Discord Webhook (단방향)
├── utils/                   # 유틸리티
│   └── diagnostics.py      # DiagnosticLogger (진단 로깅)
├── collect/                 # 소켓 수집 공용 부품 (RotatingWriter)
├── listing/                 # 신규 상장 비용 측정기 (연구 전용, 주문 없음)
└── scalper.py              # 코어 엔진 — init, cycle, scan, state (4개 Mixin 상속)

scripts/                    # 유틸리티 스크립트
├── download_15m_data.py        # 15분 캔들 데이터 캐시 갱신
├── download_6yr_data.py        # 6년 장기 데이터 다운로드
├── download_extended_data.py   # 장기 과거 데이터 다운로드
├── download_mtf_data.py        # 멀티 타임프레임 데이터 다운로드
├── download_new_symbols.py     # 신규 심볼 데이터 추가
├── backtest_live_period.py     # Live 기간 재백테스트 검증
├── check_account.py            # 계좌 상태 점검
├── win_rate_check.py           # 실시간 Win Rate 검증
├── generate_report_pdf.py      # 단일 마크다운 → PDF (Chrome headless)
└── generate_combined_report.py # 종합 PDF 리포트
research/                   # 리서치 스크립트 (r번호 체계)
├── active/
│   ├── misc/               # 베이스라인·유틸 (번호 없음)
│   ├── r12_r20/            # Alpha search R12-R16, leverage 실험
│   ├── r21_r30/            # Vol straddle 아키텍처 탐색 (R22-R30)
│   ├── r31_r40/            # 스케일링·최적화 (R31-R40, scaling_r38 canonical)
│   └── r41_r45/            # 현재 배포 전략 (R41-R45, r45_param_sweep 최신)
└── archive/                # 구버전 보관 (r2-r11, r_ 무번호)
docs/
└── backtest_results/
    ├── 202602/             # 2026-02 백테스트 리포트 (31개)
    └── 202603/             # 2026-03 백테스트 리포트 (64개, 현재)
tests/                      # 유닛 테스트 (1929개, 56 파일)
\`\`\`

## 📈 성과 지표 (v14.24.x — Dual-Axis Squeeze, r73 trail)

### 불변 엣지 지표 (leverage·기간 무관)

| 지표 | VR Squeeze (D_mom8) | VR Straddle (S_d8) | MA Cluster | 결합 (프로덕션) |
|------|---|---|---|---|
| N (trade 수) | ~292 accepted | ~7,152 accepted | ~2,329 accepted | **~9,773** |
| PF | — | **4.948** | — | **2.002** |
| Calmar | — | 1813 | — | **1695.6** |
| Bootstrap P(PF≥1.0) | 100% | 100% | 100% | 100% |
| 연속 흑자 | ✓ | ✓ | ✓ | ✓ |

> 데이터: 5.67yr (2020~2026-03), 60 syms, 0.08% RT 비용, trail-aware sim (r73)  
> **레버리지 구조**: VR 30x×1%SL → L/N=3.0 | MA 21x×2%SL → L/N=2.1 (Half-Kelly)  
> S_d8 dominant (82.4% 슬롯), MA 독립 (94.5%), D_mom8 cherry-pick (11.3%)

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
