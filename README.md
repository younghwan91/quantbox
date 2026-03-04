# QuantBox 📈

**바이낸스 USDT-M 선물 VR 스퀴즈 시스템 — Dual-Axis Squeeze (VR + MA Cluster)**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-v14.0.0-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-891%20passed-brightgreen.svg)]()
[![Branch](https://img.shields.io/badge/branch-release%2Fv14.0.0-blue.svg)]()

바이낸스 USDT-M 선물 전용 VR 압축 스퀴즈 시스템. VR Squeeze(25x)와 MA Cluster Squeeze(12x) 두 축이 10슬롯 공유 풀에서 동시 운용. Numba 가속 백테스트, pydantic-settings 환경 관리.

> **현재 상태**: Live Trading | Commit `v14.0.1` | 891 tests passed

## 🌟 주요 기능

### 🎯 VolStraddle — VR Compression Detector (v14.0.0, 배포 중)
- **Volatility Ratio (VR) 압축 감지**: `VR = ATR(8)/ATR(56) < 0.50` — 변동성 압축 후 폭발 포착
- **멀티 타임프레임**: 15m/1h/4h 독립 신호, 높은 TF 우선 (4h > 1h > 15m)
- **D_mom8 방향**: 8바 모멘텀 기반 롱/숏 방향 결정 (D_mom8)
- **S_d8 스트래들**: VR < 0.55 × 4바 지속 → 양방향 동시 진입 (Hedge Mode)
- **파라미터**: SL=1%, 하드TP=12%, Trailing +2%/0.3%, MaxHold=48h (192/48/12바)
- **R43 Trailing**: d_tp=12% 하드 TP + activate +2% / trail_distance 0.3%
- **25x @10 slots** (L/N=2.5), 6년 연속 흑자, Bootstrap P(PF≥1.0)=100%
- **불변 엣지**: N=2,265 (D+S), PF=1.280, WR=43.4%, EV=+32.1bp — 레버리지/기간 무관
- **제로 룩어헤드**: 완성된 바만 참조, 개발 중 바 제외, 교차 TF 감사 통과

### 📐 MA Cluster Squeeze — 4h SMA Convergence Breakout (V17, R45, 배포 중)
- **MA 클러스터 감지**: SMA(20/50/100/200) 4선 수렴도 ≤ 3% 지속 ≥ 8바 (32h) → 브레이크아웃 포착
- **4h 전용**: 상위 타임프레임에서만 동작, VR squeeze와 ~93% 비중복
- **Momentum(8) 방향**: 클러스터 이탈 시 모멘텀 방향으로 진입
- **리스크 패리티**: 12x leverage × 2% SL ≈ 25x × 1% SL (동일 per-trade risk)
- **Simple Exit**: SL=2%, TP=8%, MaxHold=12바 (48h), 트레일링 스탑 없음
- **R45 Zombie Filter**: `max_duration=24` 바 초과 클러스터 차단 (오래된 클러스터 제거)
- **R45 Momentum Filter**: `abs(momentum) < 0.010` 차단 (거스름 없는 약한 신호 제거)
- **불변 엣지 (R45, 6yr)**: N=1,631, PF=1.365, WR=38.1%, EV=+42.6bp
- **결합 포트폴리오 (R45)**: N=3,748, PF=1.304, Calmar=18.8, P(PF≥1.0)=100%, 6/6년 흑자

### 🔥 Rubber Band Scalper — 레거시 (v12.6, 현재 비활성화)

> ⚠️ v14.0.0에서 VR Squeeze + MA Cluster로 전략 전환. 아래 내용은 참고용.

- **v12.6 Climax Volume Filter + ATR Stop (R21, 2026-02-24)**:
  - **ATR-Based Stop**: `SL = clamp(2.0 × ATR_pct, 0.5%, 4.0%)` — 고정 0.382×에서 적응형 스탑으로 전환, Ghost-SL 문제 해결
  - **Climax Volume Filter**: `vol > 3.0 × vol_SMA(20)` = +1 confidence — 극단적 거래량 폭발 시에만 진입 바이어스
  - Stoch/MACD 비활성화 (노이즈 제거), 전체 컨피던스 체계 단순화
  - 결과: Calmar **1,171** (+60.8% vs v12.5.3), DD **5.8%** (−24.7%), WR **62.2%** (+11.1pp), Sharpe **6.37**
  - commit `15b94d9` (feature) + `14feeeb` (fixes: param wiring + diagnose_bar parity)
- **v12.4 I-HybTight**: Reversal A/B test champion — deeper target + tighter trail + longer runner
  - target_retrace=1.5 (150% move), stop_mult=0.382, max_hold=25 → R:R 1.91
  - **PBT Exit**: 50% partial at TP1 → breakeven floor → ATR trailing runner
  - Runner exits via `trail_runner` (ATR×1.0 tighter trailing) or `time_runner` (3× max_hold)
  - Server TP as safety net; client-side trailing for precision
- **v12.3 M-2SSTSX**: R6 champion combo (2slt + SoftTight + SteepX + DirDiversity)
  - **SteepX Sizing**: conf2→0.3×, conf3→0.5×, conf4→1.5×, conf5+→2.0× — rewards conviction
  - **SoftTight DD Scaling**: 2%DD→0.8×, 4%DD→0.6×, 6%DD→0.4× — automatic position reduction during drawdowns
  - **Vol-Adjusted Sizing**: median_atr/current_atr clamped [0.4, 2.0] — normalizes position size by current volatility
  - **Direction Diversity**: max 2 same-direction positions — prevents cascade losses
  - min_confidence=3, max_concurrent=2
- **v12.3 Production Safety** (3 critical bug fixes):
  - State restore preserves runner `trail_floor` + `highest_since_tp1` across bot restarts
  - Bracket repair skips server TP for runners (prevents instant kill)
  - Failed partial close clears stale order IDs to trigger bracket repair
- **v12.4.1–v12.4.6 Production Hardening** (6 hotfixes):
  - v12.4.1: Server TP respects PBT mode — 50% qty at TP1, not 100%
  - v12.4.2: Runner-phase server TP guard — skip TP placement when trailing
  - v12.4.3: Remaining 4 call-sites TP qty audit (sync/repair/failover/recovery)
  - v12.4.4: Stale `_server_sl_pct` reset after runner SL re-placed at breakeven
  - v12.4.5: Missing `_save_state()` after successful partial close
  - v12.4.6: HWM + trade history reset on restart with 0 positions (prevents phantom DD halt)
- **Alpha**: ATR 기반 적응형 임계값으로 시장 변동성에 실시간 대응하는 단일 자산 평균회귀 전략
- **v11.4 Adaptive Threshold**: 고정 3.2% → ATR(48)×2.5 적응형 (OOS Sharpe 8.23, OOS/IS 1.09, 제로 과적합)
  - 공식: `threshold = clip(2.5 × ATR%(48), 1%, 6%)` — 변동성 높으면 넓고, 낮으면 타이트
  - Walk-forward 검증: 18개 변형 × 20개 심볼, IS/OOS 분리, 모든 변형에서 OOS/IS ≥ 1.04
  - 레버리지 최적화: 1×~5× 그리드 → Sharpe 단조 감소, 현행 3.75× 유지 결정
- **v10.0 Entry Optimization**: threshold fallback 0.032 (ATR NaN 시), min_confidence 2 (v12.2 E-3Slot quality gate)
- **v10.0 Extended Hold**: max_hold 10→20 (+24% ret, strictly better)
- **v10.0 Faster Stochastic**: stoch_k 5→3 (+5% Calmar)
- **v11.1 Server-Side Stop-Loss**: 바이난스 STOP 주문으로 클라이언트 크래시 방어 + 레버리지 설정 검증
- **v12.2 E-3Slot Universe**: Top-50 (4 tiers) + 필터 완화 (conf≥2, ATR 2.0×, floor 0.8%, 3 slots)
- **v10.0 Symbol Whitelist**: Top-50 검증된 심볼만 거래 (E-3Slot config, RB_MAX_CONCURRENT=3)
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
  - min_confidence=2: v12.2 E-3Slot 최적 필터 (Sharpe 6.41, Calmar 795, +138% trades/day)
  - 선택적 졸업 레버리지: 증거 많을수록 포지션 확대
- **Regime-Switch Dual Entry (v5.0 base)**: BTC 레짐에 따라 다른 진입 알고리즘 사용
  - Risk-On (BTC > SMA96): Classic Pullback — 자산 자체 상승추세에서 조정 시 매수
  - Risk-Off (BTC < SMA96): Sympathy Pullback — BTC 하락 시 동반 하락하는 자산 매수
- **Fibonacci Stop/Target (v9.0→v12.4)**: stop=0.382× (v12.3), target=1.5× (v12.4 HybTight) → R:R 1.91
- **Flat 1.5x Leverage**: R:R 1.25로 레버리지 정당화
  - R:R ≥ 0.30 안전 게이트 유지
- **Dynamic Leverage**: Move-confidence 기반 동적 레버리지 (R:R 게이트와 연동)
- **MTF (v7.0)**: 4h SMA(12) 트렌드 확인 활성화 — MDD -74%, Calmar +275%
- **Live/Backtest Parity (v7.0)**: volume/taker 데이터 _scan_entries() 연결 + 4h kline 다운로드 + main() v7.0 설정 전달
- **365-Day Validated**: 50 심볼 × 35,040봉, 7/7 견고성 점수, 9/10 롤링 윈도우 수익
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
│   ├── commands.py         # Discord 봇 커맨드 핸들러 (CommandsMixin)
│   ├── config.py           # 설정 관리
│   ├── events.py           # 이벤트 시스템
│   ├── limit_chaser.py     # 지정가 추격 주문
│   ├── settings.py         # 환경변수 검증 (pydantic-settings)
│   ├── state_persistence.py # 상태 영속화 (JSON, 크래시 복구)
│   └── terminal_ui.py      # 터미널 UI (실시간 상태 표시)
├── execution/               # 주문 실행 및 포지션 관리
│   ├── brackets.py         # 서버사이드 브라켓 주문 (BracketMixin)
│   ├── positions.py        # 포지션 라이프사이클 (PositionMixin)
│   └── reconciliation.py   # 거래소 상태 동기화 (ReconciliationMixin)
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
│   └── vol_selector.py     # 변동성 기반 심볼 선택기
├── backtest/               # 백테스팅 프레임워크
│   └── vectorized_backtest.py  # 벡터화 백테스트 (단일 자산, Numba)
├── notifications/          # 알림 시스템
│   ├── bot.py              # Discord Bot (양방향)
│   └── discord.py          # Discord Webhook (단방향)
├── utils/                   # 유틸리티
│   └── diagnostics.py      # DiagnosticLogger (진단 로깅)
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
    └── 202603/             # 2026-03 백테스트 리포트 (19개, 현재)
tests/                      # 유닛 테스트 (897개, 45 파일)
\`\`\`

## 📈 성과 지표 (현재 v14.0.0 — Dual-Axis Squeeze)

### 불변 엣지 지표 (leversage·기간 무관)

| 지표 | VR Squeeze (D+S) | MA Cluster (R45) | 결합 (R45) |
|------|---|---|---|
| N (trade 수) | 2,265 | 1,631 | **3,748** |
| PF | 1.280 | 1.365 | **1.304** |
| WR | 43.4% | 38.1% | 40.9% |
| EV/trade | +32.1bp | +42.6bp | +35.4bp |
| Bootstrap P(PF≥1.0) | 100.0% | 100.0% | 100.0% |
| 6/6년 흑자 | ✓ | ✓ | ✓ |

### 레버리지 적용 수익률 (6년, 50심볼)

| 지표 | v7.0 Hybrid | v13.0 VR (25x) | v14.0 Dual (R45) ★ |
|------|-------------|----------------|--------------------|

| AnnRet (backtest) | +139.8% | +687% | **+1,086% (1x) / +1,410% (L)** |
| AnnRet (realistic 60%) | — | +412% | **+846%** |
| Max Drawdown | 4.3% | 68% | **62%** |
| Calmar | 32.42 | 10.12 | **18.8** |
| Profit Factor | — | — | **1.304** |
| N/year | — | — | **~600** |

> **v14.0.0 R45 변경 (현재)**: MA Cluster min_momentum 0.003→0.010, max_duration=∞→24. 좀비·약모멘텀 트레이드 차단. Combined PF 1.278→1.304 (+2%), EV/trade +24%.
>
> **v14.0.0 R43 Trailing**: VR Squeeze d_tp 5%→12% (하드 TP) + trailing stop (activate +2%, trail 0.3%). Calmar 대폭 개선.
>
> **v13.0 → v14.0**: MA Cluster Squeeze 추가 (Dual-Axis). AnnRet +687%→+1,086% (1x compound), Calmar 10.12→18.8.
>
> **레버리지 구조**: VR 25x×1%SL = MA 12x×2%SL = per-trade risk 25% (리스크 패리티). 현재 잔고 $1,668 → 현실적 1yr 기대 $13,000–$16,000 (50-60% backtest 적용).

## 🧪 테스트

\`\`\`bash
uv run pytest tests/ -v
\`\`\`

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 사용해야 합니다. 암호화폐 거래는 상당한 손실 위험이 있습니다.
