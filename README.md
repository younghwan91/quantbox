# QuantBox 📈

**바이낸스 USDT-M 선물 VR 스퀴즈 시스템 — Dual-Axis Squeeze (VR + MA Cluster)**

> ## ⚠️ 면책 (Disclaimer)
>
> 이 저장소는 **교육·연구 목적**으로 공개된다. 투자 조언이 아니다.
>
> - 레버리지 선물 거래는 **원금 전액을 잃을 수 있고**, 손실이 입금액을 초과할 수 있다.
> - 백테스트 성과는 시뮬레이션이며 미래 수익을 보장하지 않는다. 이 저장소의 리포트에는
>   **엣지가 시간에 따라 소멸한 기록이 그대로 남아 있다** — 과거에 통했다는 것이
>   지금 통한다는 뜻이 아니라는 실례다.
> - 이 코드를 실행해 발생한 어떠한 손실에 대해서도 저자는 책임지지 않는다 (MIT, `LICENSE`).
> - 실행 전 반드시 페이퍼 모드로 검증하고, 잃어도 되는 돈만 쓸 것.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-v14.44.0-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-2482%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-79%25-green.svg)]()
[![Branch](https://img.shields.io/badge/branch-main-blue.svg)]()
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

바이낸스 USDT-M 선물 전용 VR 압축 스퀴즈 시스템. VR Directional(D_mom8)과 MA Cluster Squeeze 두 축이 10슬롯 공유 풀에서 동시 운용. Donchian Breakout은 코드에 있으나 기본 비활성(`donchian_enabled=False`). Numba 가속 백테스트, pydantic-settings 환경 관리.

> **현재 상태**: 라이브 중단 (마지막 라이브 트레이드 2026-07-18, `data/journal.sqlite`) | Tag `v14.44.0` | 2482 passed, 8 skipped

## 🌟 주요 기능

### 🎯 VolStraddle — VR Compression Detector (v14.44.0)
- **Volatility Ratio (VR) 압축 감지**: `VR = ATR(8)/ATR(56) < 0.50` — 변동성 압축 후 폭발 포착
- **멀티 타임프레임**: 15m/30m/1h/4h 독립 신호, 높은 TF 우선 (4h > 1h > 30m > 15m)
- **D_mom8 방향**: 8바 모멘텀 기반 롱/숏 방향 결정 (D_mom8)
- **파라미터**: SL=1%, 하드TP=20%(safety cap, 사실상 미발동), Trailing activate +1.5%/distance 0.5%, MaxHold=48h
- **Trail 조정**: D 1.5%/0.5% (r106 4.0%→라이브 1.5%; distance 하한은 Binance callbackRate≥0.5%)
- **사이징**: SL 폭에서 레버리지 역산 — `lev = risk_per_trade × max_concurrent / SL`. 현행 `risk_per_trade=0.005`, 10슬롯 → VR 5x (`max_leverage=30`은 상한일 뿐)
- **제로 룩어헤드**: 완성된 바만 참조, 개발 중 바 제외, 교차 TF 감사 통과 (`tests/test_mtf_lookahead.py`)

### 📐 MA Cluster Squeeze — 4h SMA Convergence Breakout (V17, R45)
- **MA 클러스터 감지**: SMA(20/50/100/200) 4선 수렴도 ≤ 3% 지속 ≥ 6바 (24h) → 브레이크아웃 포착
- **4h 전용**: 상위 타임프레임에서만 동작, VR squeeze와 ~93% 비중복
- **Momentum(8) 방향**: 클러스터 이탈 시 모멘텀 방향으로 진입
- **사이징**: VR과 동일한 SL 역산 — `risk_per_trade=0.005`, 2% SL, 10슬롯 → MA 2.5x (per-trade risk 0.5%)
- **Exit**: SL=2%, TP=8%(safety cap, 전 구간 0건), **트레일 비활성** (`ma_cluster_trail_enabled=False`), MaxHold=24바×4h (4일)
- **R45 Zombie Filter**: `max_duration=24` 바 초과 클러스터 차단 (오래된 클러스터 제거)
- **R45 Momentum Filter**: `abs(momentum) < 0.010` 차단 (거스름 없는 약한 신호 제거)
- **MA Cluster 현재 엣지 (2026-08-10 워크포워드, 펀딩 반영, risk=0.005)**: 검증 PF 중앙값 1.156, 연 15.56%, maxDD 13.4%, 5/5 구간 PF>1. **최근 구간은 PF 1.042 / 연 1.86%** — 엣지가 시간에 따라 감쇠 중이다 (`docs/backtest_results/202608/20260810_f3191ec_trail_removal_long_hold.md`)
- **4h TF 독립**: VR squeeze와 93.7% 비중복

### 🔥 레거시 전략 (v14.0.0 이전, 비프로덕션)

- **Rubber Band Scalper (v12.6)**: ATR 적응형 평균회귀, Climax Volume Filter. v14.0.0에서 VolStraddle로 전략 전환. Phase 2.5에서 deprecated — 모듈 파일은 참조용으로만 남아 있고 신규 코드에서 import 금지.
- **Elliott Wave (v11.6)**: ZigZag ABC 패턴 + Timing Alpha. **코드 삭제됨** (커밋 `70a7bca`).
- **Alpha Engine / Hybrid Strategy**: **코드 삭제됨** (커밋 `dfc2ea8`, 데드코드 정리).

### 핵심 아키텍처
- **실거래/백테스트 일체형**: 동일한 Frozen z-score 로직이 백테스트와 실거래에서 동작
- **모듈형 설계**: 전략, 필터, 실행 컴포넌트 조합 가능
- **타입 안전성**: 전체 코드베이스에 타입 힌트 적용
- **안전장치 내장**: Market Fallback 검증, 알림 스로틀링, avg_price=0 거부 등

### 데이터 관리
- **2단계 캐싱**: 메모리(LRU) + 디스크(압축) 캐싱 시스템
- **멀티 타임프레임**: 1m, 5m, 15m, 30m, 1h, 4h, 1d 데이터 지원
- **바이낸스 연동**: Rate limiting과 배치 처리 지원

### 전략 프레임워크
- **VR Directional (`vr_directional`, D_mom8)**: 프로덕션 전략. 15m/30m/1h/4h VR 압축 + 8바 모멘텀 방향
- **MA Cluster (`ma_cluster`)**: 프로덕션 전략. 4h SMA 4선 수렴 후 브레이크아웃
- **Donchian (`donchian`, DC_vol50)**: 4h 50바 채널 돌파 + 볼륨 스파이크. 코드는 있으나 `donchian_enabled=False` (라이브 검증 대기)
- **Rubber Band Mean-Reversion (v12.2 E-3Slot)**: ATR 2.0× 적응형 임계값 + MACD/Stochastic 컨플루언스. deprecated, 참조용 보존
- **Volatility Selector**: 실현변동성 기준 심볼 동적 선택 (백테스트 유틸)
- 전략 이름 문자열의 단일 진실 원천: `quantbox/core/strategy_names.py`

### 실거래 안전장치
- **Server-Side Stop-Loss (v11.1)**: 바이난스 STOP 주문으로 클라이언트 크래시/네트워크 단절 시 손절 보장
- **Leverage Verification (v11.1)**: 레버리지 설정 실패 시 주문 중단 (알 수 없는 레버리지로 거래 방지)
- **Limit Chaser**: 지정가 추격 주문으로 슬리피지 최소화
- **Market Fallback 검증**: ACK 응답 감지 → 자동 재조회 (유령 체결 방지)
- **State Persistence**: 포지션 상태 JSON 저장 → 크래시 후 자동 복구 + 거래소 대조
- **Max Drawdown Breaker**: 세션 HWM 대비 25% 드로다운 시 전체 청산 + 자동 정지 (`max_drawdown_pct=0.25`)
  - v12.4.6: 0 포지션 재시작 시 HWM을 현재 잔고로 리셋 (확정 손실에 의한 팬텀 DD 방지)
- **Daily Loss Limit**: 당일 잔고 대비 12% 손실 시 신규 진입 중단 (`daily_loss_limit_pct=0.12`, UTC 자정 리셋)
- **Consecutive Error Guard**: 연속 3회 사이클 에러 시 SAFE MODE — 신규 진입만 중단하고 기존 포지션은 계속 관리 (전체 청산 아님)
- **Close Failure Retention**: 청산 실패 시 슬롯 유지 → 다음 사이클 재시도 (유령 포지션 방지)
- **알림 스로틀링**: 슬롯 포화/잔고 부족 알림 중복 전송 방지
- **Discord Bot**: 원격 모니터링 및 명령 (포지션 조회, 강제 청산 등)
- **Log Rotation**: 50MB/5파일 로테이션 (디스크 무한 성장 방지)
- **Config Validators (v14.22.0)**: 22개 교차 검증 — SL<TP, leverage≥1, trail_distance<trail_activate, callbackRate≥0.5% 등 (`VolStraddleConfig._validate_safety_constraints`)
- **Rate Limiter + Circuit Breaker (v14.22.0)**: API 호출 스로틀링. 연속 에러 시 자동 차단
- **In-Flight Order Tracker (v14.22.0)**: 미체결 주문 실시간 추적. 크래시 복구 시 자동 재동기화
- **ExposureGuard (v14.23.0)**: 세션 노출도 상한선 (`max_exposure_multiplier=35.0`). 초과 시 신규 진입 차단
- **ReconJournal (v14.23.0)**: SQLite 기반 10종 이벤트 감사 로그 (ghost, bracket repair, exposure 등)
- **TradingStrategy Protocol (v14.35.0)**: 14개 멤버 타입 안전 인터페이스. 전략 확장 표준화

### 백테스팅 (Dual-Engine)
- **Signal-Array Engine** (`vectorized_backtest.py`): RubberBand 등 ±1 포지션 어레이 기반. Close-to-close PnL. 단일자산 신속 스캔용.
- **Trade-Event Engine** (`trade_sim.py` + `sim_core.py`): VolStraddle/MA Cluster/Donchian 이산 트레이드 시뮬레이션. High/Low 기반 인트라바 SL/TP. N-slot 포트폴리오 경합. Numba 가속.
  - `sim_directional_trades()` → `collect_trades()` → `portfolio_sim()` → `compute_metrics()` 파이프라인
  - `sim_directional_trailing()` / `sim_ma_cluster_trailing()` / `sim_donchian_trailing()` — trail-aware 시뮬레이션
  - `apply_funding_drag()` — 8h 펀딩 드래그 (2026-08-10 파리티 수정: MA cluster 경로가 펀딩 미차감이었다)
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

접속: `http://localhost:8501`

> ⚠️ **대시보드에는 인증이 없다.** 실계좌 잔고·포지션·손익을 그대로 보여주므로 기본
> 바인딩은 `127.0.0.1`(같은 PC 전용)이다. 다른 기기에서 봐야 한다면
> `DASHBOARD_BIND=0.0.0.0 ./scripts/start_dashboard.sh start`로 명시적으로 열되,
> 그 네트워크에 있는 누구나 인증 없이 계좌를 볼 수 있다는 뜻임을 알고 하라.
> 공개망에 노출하지 말 것.

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
│   ├── limit_chaser.py     # 지정가 추격 주문
│   ├── settings.py         # 환경변수 검증 (pydantic-settings)
│   ├── scalper_state.py    # 상태 영속화 (JSON 원자적 저장, 롤링 백업, 크래시 복구)
│   ├── loop_mixin.py       # 메인 사이클 루프 (LoopMixin)
│   ├── lifecycle.py        # 기동/종료 라이프사이클
│   ├── protocol.py         # ScalperProtocol — Mixin 합성 클래스의 구조적 타입
│   ├── strategy_names.py   # 전략 이름 단일 진실 원천 (vr_directional/ma_cluster/donchian)
│   ├── trade_journal.py    # SQLite 트레이드 저널 기록
│   ├── divergence_monitor.py # 라이브-백테스트 성과 괴리 감시
│   ├── terminal_ui.py      # 터미널 UI (실시간 상태 표시)
│   ├── rate_limiter.py     # API 호출 스로틀링 + Circuit Breaker (v14.22.0)
│   └── heartbeat.py        # 메인 루프 정지 감지 + Discord 재연결 (v14.22.0)
├── execution/               # 주문 실행 및 포지션 관리
│   ├── brackets.py         # 서버사이드 브라켓 주문 (BracketMixin)
│   ├── positions.py        # 포지션 라이프사이클 (PositionMixin)
│   ├── reconciliation.py   # 거래소 상태 동기화 (ReconciliationMixin)
│   ├── inflight.py         # 미체결 주문 추적 + 크래시 복구 (v14.22.0)
│   ├── exposure.py         # 세션 노출도 상한선 ExposureGuard (v14.23.0)
│   ├── risk_guard.py       # 일일 손실 한도 / DD 차단
│   ├── sizing.py           # SL 역산 레버리지 (risk_per_trade 기반)
│   ├── preflight.py        # 주문 전 사전 점검
│   ├── algo_api.py         # Binance Algo Order API 래퍼
│   ├── data_mixin.py       # 캔들 조회 (DataMixin)
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
│   ├── vol_straddle/       # VR 압축 + MA 클러스터 + Donchian
│   │   ├── config.py       # VolStraddleConfig (VR + MA cluster + Donchian 파라미터)
│   │   ├── signals.py      # Numba VR/모멘텀/SMA/MA클러스터 (10개 함수)
│   │   ├── conviction.py   # CW-tier conviction 스코어링/사이징
│   │   └── strategy.py     # VolStraddleStrategy 클래스
│   ├── vol_selector.py     # 변동성 기반 심볼 선택기 (백테스트 유틸)
│   └── protocol.py         # TradingStrategy Protocol 14멤버 (v14.35.0)
├── backtest/               # 듀얼 엔진 백테스팅 프레임워크
│   ├── vectorized_backtest.py  # Signal-Array 엔진 (RubberBand, close-to-close)
│   ├── trade_sim.py            # Trade-Event 엔진 (intra-bar SL/TP, slot 경합, 펀딩 드래그)
│   ├── sim_core.py             # 라이브 충실 시뮬레이션 코어
│   ├── funding.py              # 펀딩 비용 모델
│   └── meta_score.py           # MetaScore 파라미터 평가
├── notifications/          # 알림 시스템
│   ├── bot.py              # Discord Bot (양방향)
│   └── discord.py          # Discord Webhook (단방향)
├── utils/                   # 유틸리티
│   └── diagnostics.py      # DiagnosticLogger (진단 로깅)
├── collect/                 # 소켓 수집 공용 부품 (RotatingWriter)
├── listing/                 # 신규 상장 비용 측정기 (연구 전용, 주문 없음)
├── research/                # 리서치 하니스 (harness — 커밋 스탬프 등 공용 유틸)
└── scalper.py              # 코어 엔진 — init, cycle, scan, state (6개 Mixin 상속:
                            #   Loop/Data/Bracket/Reconciliation/Position/Commands)

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
│   ├── r12_r20/ … r31_r40/ # Alpha search·아키텍처 탐색·스케일링 (scaling_r38 canonical)
│   ├── r41_r45/ … r96_r100/# 배포 전략 계보 (r45_param_sweep, r73, r92 등)
│   ├── r101_r105/ … r116_willy_nfi/  # 트레일 스윕(r106-r108), CW 사이징(r115)
│   └── r121_harness/       # 최신 — MA cluster per-TF 하니스
└── archive/                # 구버전 보관 (r2-r11, r_ 무번호)
autoresearch/               # 자동 리서치·페이퍼 로깅 (crontab 운용)
                            #   paper_log.py — 매일 09:17
                            #   macro_live.py — 평일 10:07
docs/
└── backtest_results/
    ├── 202602/             # 2026-02 백테스트 리포트 (31개)
    ├── 202603/             # 2026-03 백테스트 리포트 (86개)
    ├── 202604/ 202607/     # (6개 / 2개)
    └── 202608/             # 2026-08 — 트레일 제거·사이징 재산정 (6개, 현재)
tests/                      # 유닛 테스트 (2482 passed / 8 skipped, 139 파일)
\`\`\`

## 📈 성과 지표 (2026-08-10 재측정 — MA Cluster 워크포워드)

> 이전 README의 r73 3전략 수치(PF 2.002 / Calmar 1695.6)는 S_d8 스트래들이 포함된
> 값이다. S_d8은 커밋 `9a83ebf`에서 코드에서 완전히 제거됐으므로 더 이상 유효하지 않다.
> 아래는 현행 설계(트레일 제거, MaxHold 24봉, `risk_per_trade=0.005`)로 재측정한 값이다.

### MA Cluster 워크포워드 (학습 구간에서만 h 선택, 펀딩 반영, risk=0.005 / 2.5x)

| 구간 | h | 검증 PF | 연수익 | maxDD | N |
|---|---|---|---|---|---|
| 1 | 24 | 2.204 | 48.16% | 4.6% | 109 |
| 2 | 36 | 1.380 | 20.57% | 13.4% | 169 |
| 3 | 36 | 1.156 | 15.56% | 14.2% | 347 |
| 4 | 36 | 1.118 | 6.73% | 13.4% | 243 |
| 5 | 24 | 1.042 | **1.86%** | 18.9% | 227 |

> **중앙값 PF 1.156 / 연 15.56% / maxDD 13.4%, 5/5 구간 PF > 1.**  
> **엣지는 시간에 따라 단조 감쇠한다 — 가장 최근 구간은 PF 1.042, 연 1.86%다.**
> 중앙값은 현재 기대치가 아니다.  
> 데이터: `data/cache_extended` 전 종목 6.0년, 15m 청산 격자, 펀딩 반영.  
> 근거: `docs/backtest_results/202608/20260810_f3191ec_trail_removal_long_hold.md`  
> **사이징**: `lev = risk_per_trade × max_concurrent / SL` — VR 5x, MA 2.5x.
> 이전 30x/21x 사이징은 같은 리포트에서 라이브 maxDD 77~98%로 측정돼 폐기됐다.

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.

## 📄 License

MIT — 자세한 내용은 [`LICENSE`](LICENSE) 참조.
