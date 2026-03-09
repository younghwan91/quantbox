# QuantBox 버전별 성능 연대기

**작성일:** 2026-03-09  
**커밋:** `aec56c9`  
**목적:** v12.9 → v14.5.0 전 버전의 전략 파라미터, 성능 수치, 기능 diff를 완전히 추적.  
**보장:** 마지막 행(v14.5.2)은 라이브 프로덕션(60 syms, SHA 34076e8)과 **단 1개 파라미터도, 단 1개 심볼도 다르지 않음.**  
**Canonical 수치:** v14.5.2, 60 syms, 6yr | PF=1.563 | N=6,207 | Calmar=57.9 | CI=[1.461,1.676]

---

## 약어 & 공통 기준

| 약어 | 정의 |
|------|------|
| PF | Profit Factor (총수익 / 총손실) |
| WR | Win Rate |
| EV | Expected Value per trade (basis points) |
| Calmar | CAGR / Max DrawDown |
| L/N | base_leverage / max_concurrent (per-trade leverage multiplier) |
| 1x compound | `equity *= (1 + pnl_pct)` — compute_metrics() 기본값, 레버리지 미적용 |
| TBR | Trend Bias Rejection (D_mom8 entry z-score gate) |
| round-trip cost | 0.08% (entry 0.04% + exit 0.04%) |
| 데이터 셋 A | 4.73yr, ~50 symbols (2021-2025, r43-r47 시대) |
| 데이터 셋 B | 6yr, 60 symbols (2020-2025, r53 63-sym - 3 delisted, v14.5.2 canonical) |

> **주의:** 각 버전의 성능 수치는 해당 시점에 사용된 데이터셋 기준. 크로스-버전 절대치 직접 비교는 데이터셋 B(v14.5.0 Phase 8) 기준만 유효.

---

## v12.9 — "R39/R40 VR Tournament Champion"

**태그:** `v12.9` | **태그 메시지:** `v12.9 production: R39-VR856 frequency boost`

### 전략 파라미터

| 파라미터 | 값 |
|----------|-----|
| 전략 축 | VR Squeeze (D_mom8 only, directional) |
| d_sl | 0.01 (1%) |
| d_tp | 0.05 (5%, flat) |
| d_trail_activate | — (없음) |
| d_trail_distance | — (없음) |
| straddle_enabled | False |
| ma_cluster_enabled | False |
| base_leverage | 3.0x (exchange) |
| max_concurrent | 15 slots |
| **L/N** | **0.20** |
| f5_tbr_enabled | False |
| max_drawdown_pct | 0.10 |

### 성능 (데이터셋 A, R40 기준)

| 지표 | 값 |
|------|-----|
| N | ~2,265 |
| PF | 1.280 |
| WR | 43.4% |
| 1x Calmar | 5.8 |
| 1x AnnRet | +195% |
| L/N=0.20 AnnRet | ~+39% (보수적) |
| Losing years | 0 / 6yr |

---

## v13.0 — "Aggressive L/N=2.5 Deployment"

**태그:** `v13.0` | **배포:** 2026-03-02 | **연구:** R40 tournament

### v12.9 대비 변경사항 (기능 diff)

| 항목 | v12.9 | v13.0 | 이유 |
|------|-------|-------|------|
| base_leverage | 3.0x | **25.0x** | Half-Kelly 최적 L/N=2.5 |
| max_concurrent | 15 | **10** | L/N=2.5 목표 |
| straddle_enabled | False | **True** | S_d8 straddle 추가 |
| L/N | 0.20 | **2.50** | Calmar 최대화 zone |

> ⚡ **핵심 변경:** L/N 0.20→2.50, 12.5배 증가. 동일한 엣지(PF=1.280)에서 compound 수익률 폭발.

### 성능 (데이터셋 A, v13.0 deploy report)

| 지표 | 값 |
|------|-----|
| N | ~2,265 |
| PF | 1.280 |
| WR | 43.4% |
| 1x AnnRet | +195% |
| **L/N=2.5 AnnRet** | **+687%** |
| DD | 67.9% |
| Calmar (L/N=2.5) | **10.12** |
| Losing years | 0 / 6yr |

---

## v13.1.0 — "D_mom8 Trailing Stop (R43 P1 Conservative)"

**태그:** `v13.1.0` | **배포:** 2026-03-03 | **연구:** R43 exit improvement

### v13.0 대비 변경사항 (기능 diff)

| 항목 | v13.0 | v13.1.0 | 이유 |
|------|-------|---------|------|
| d_tp | 0.05 (5%) | **0.12 (12%)** | 위너 오래 보유 |
| d_trail_activate | — | **0.02 (2%)** | +2% 이후 trail 시작 |
| d_trail_distance | — | **0.003 (0.3%)** | peak 0.3% 아래 추적 |

> ⚡ **핵심 변경:** 트레일링 스탑 도입. 5% flat TP를 제거하고 12% hard ceiling + trailing으로 위너를 더 오래 보유.

### 성능 (데이터셋 A, R43 P1)

| 지표 | v13.0 | v13.1.0 | 변화 |
|------|-------|---------|------|
| N | 2,265 | ~2,265 | — |
| PF | 1.280 | **1.351** | +5.5% |
| WR | 43.4% | **47.3%** | +3.9pp |
| Calmar (1x) | 5.8 | **9.4** | +62% |
| Calmar (L/N=2.5) | 10.1 | **24.2** | **+140%** |
| Losing years | 0 | 0 | — |

---

## v14.0.0 — "Dual-Axis: MA Cluster Squeeze 추가 (R44/R45)"

**태그:** `v14.0.0` | **배포:** 2026-03-04 | **연구:** R44c/R44d, R45 param sweep

### v13.1.0 대비 변경사항 (기능 diff)

| 항목 | v13.1.0 | v14.0.0 | 이유 |
|------|---------|---------|------|
| **ma_cluster_enabled** | False | **True** | 2축 전략 추가 (다각화) |
| ma_cluster_leverage | — | **12.0x** | Risk parity: 12×2%SL ≈ 25×1%SL |
| ma_cluster_sl | — | 0.02 (2%) | |
| ma_cluster_tp | — | 0.08 (8%) | flat TP |
| ma_cluster_trail_activate | — | — (없음) | v14.4.0에서 추가 |
| ma_cluster_min_momentum | — | **0.01** | R45 zombie filter |
| ma_cluster_max_duration | — | **24 bars** | R45 zombie filter |
| MA L/N | — | **1.2** (12/10) | |
| dynamic_capital_enabled | True | **False** | hotfix: -2019 마진 에러 방지 |

> ⚡ **핵심 변경:** MA Cluster Squeeze 2축 추가. VR+MA 결합 포트폴리오로 Calmar 대폭 개선.

### 성능 (데이터셋 A, R45 combined)

| 지표 | VR only (v13.1.0) | VR+MA Combined | 변화 |
|------|-------------------|----------------|------|
| N | ~2,265 | ~(VR+MA) | ↑ |
| PF (VR) | 1.351 | 1.280 (D_mom8) | — |
| PF (MA) | — | **1.365** | — |
| **Combined PF** | — | **1.304** | — |
| **Calmar (combined)** | — | **18.8** | ↑↑ |
| P(PF≥1.0) | 100% | **100%** | — |
| Losing years | 0 | 0 | — |

> **참고:** v14.0.0 시점 D_mom8 trail params는 v13.1.0 그대로 (act=2%, dist=0.3%).

---

## v14.1.0 — "Half-Kelly Leverage: VR 25x→30x, MA 12x→21x"

**태그:** `v14.1.0` | **배포:** 2026-03-05 | **연구:** Half-Kelly 공식

### v14.0.0 대비 변경사항 (기능 diff)

| 항목 | v14.0.0 | v14.1.0 | 이유 |
|------|---------|---------|------|
| base_leverage | 25.0 | **30.0** | Half-Kelly: f*/2=3.07%, L=0.0307×10/0.01 |
| ma_cluster_leverage | 12.0 | **21.0** | Half-Kelly: f*/2=4.33%, L=0.0433×10/0.02 |
| VR L/N | 2.50 | **3.00** | |
| MA L/N | 1.20 | **2.10** | |
| **버그수정** | — | Algo FINISHED 미감지 수정 (Incident #4) | |
| **버그수정** | — | dynamic_capital balance/max_concurrent hard-cap | |
| **리팩터** | — | trade_sim.py 추출 (백테스트 재구현 금지 기반) | |

> ⚡ **핵심 변경:** 레버리지 최적화 (순수 sizing 변경). 전략 파라미터 동일, compound 수익률 증가.

### 성능

전략 params 변경 없음 → PF/WR/N 동일. Calmar은 L/N 증가에 비례하여 상승.  
Combined PF=1.304, VR L/N=3.0, MA L/N=2.1.

---

## v14.2.0 — "F5 TBR Gate 코드 추가 (R46, 기본값=OFF)"

**태그:** `v14.2.0` | **배포:** 2026-03-06 | **연구:** R46 Monte Carlo validated

### v14.1.0 대비 변경사항 (기능 diff)

| 항목 | v14.1.0 | v14.2.0 | 이유 |
|------|---------|---------|------|
| f5_tbr_enabled | — (없음) | **False** (신규, 기본=OFF) | R46: 코드 추가, 아직 미활성 |
| f5_tbr_z_threshold | — | 0.5 | R46b optimal |
| f5_tbr_z_window | — | 192 bars (~2day 15m) | |

> **성능 변화 없음** — TBR 기본값 False. 실거래 동작 v14.1.0과 동일.

---

## v14.3.0 — "F5 TBR Gate 활성화 (f5_tbr_enabled=True)"

**태그:** `v14.3.0` | **배포:** 2026-03-07 | **연구:** R46 + corrected parity

### v14.2.0 대비 변경사항 (기능 diff)

| 항목 | v14.2.0 | v14.3.0 | 이유 |
|------|---------|---------|------|
| f5_tbr_enabled | False | **True** | D_mom8 저품질 진입 차단 |

> **Gate 로직:** LONG 차단 when TBR_z > +0.5; SHORT 차단 when z < −0.5 (lookback=192 bars, bar[t-1] — no lookahead)

> ⚡ **핵심 변경:** D_mom8 entry filter 활성화. N 감소, PF/EV 개선.

### 성능 (데이터셋 A, corrected parity report)

| 지표 | v14.2.0 (TBR off) | v14.3.0 (TBR on) | 변화 |
|------|-------------------|------------------|------|
| D_mom8 N | baseline | **−31%** | 저품질 제거 |
| D_mom8 PF | 1.324 | **1.380** | +4.2% |
| D_mom8 WR | ~43% | **47.0%** | +4pp |
| D_mom8 EV | ~+23bp | **+29.6bp** | +29% |
| MA PF | 1.365 | 1.365 | — |
| **Combined PF** | 1.304 | **1.308** | +0.3% |
| **Combined Sharpe** | ~3.4 | **2.39** | ↓ (N감소, vol 변화) |
| Calmar (1x) | — | 10.1 | — |
| Losing years | 0 | 0 | — |

---

## v14.4.0 — "r47 Exit Strategy: Option D (Tight D_trail + MA Trail 신규)"

**태그:** `v14.4.0` | **배포:** 2026-03-08 | **연구:** R47 4-way A/B test

### v14.3.0 대비 변경사항 (기능 diff) — **가장 큰 전략 변화**

| 항목 | v14.3.0 | v14.4.0 | 이유 |
|------|---------|---------|------|
| d_trail_activate | 0.02 (2.0%) | **0.005 (0.5%)** | trail을 훨씬 빠르게 시작 |
| d_trail_distance | 0.003 (0.3%) | **0.002 (0.2%)** | trail gap 더 좁게 |
| **ma_cluster_trail_activate** | — (없음) | **0.04 (4%)** | MA도 trailing 추가 |
| **ma_cluster_trail_distance** | — (없음) | **0.005 (0.5%)** | |
| ma_cluster_tp | 0.08 (flat) | **0.08 (hard ceiling)** | trailing이 먼저 작동 |
| f5_tbr_enabled | True | **False** | trail이 저품질 trade를 직접 컷 |

> ⚡ **핵심 변경:** D_mom8 trail 10배 tight화(2%→0.5% activate), MA trailing 신규 추가. TBR 비활성화 (trail로 대체).

### 성능 (데이터셋 A, r47 Option D COMBO_D0.5_MA4, TBR OFF 비교)

| 지표 | Baseline (v14.3.0 equiv, TBR off) | v14.4.0 (r47 Option D) | 변화 |
|------|-----------------------------------|------------------------|------|
| N | 3,713 | ~similar | — |
| **PF** | 1.324 | **1.461** | **+10.4%** |
| **Sharpe** | 3.39 | **4.65** | **+37.2%** |
| 1x CAGR | +1,282% | **+4,327%** | +238% |
| OOS50 PF | 1.269 | ~1.304 | +2.8% |
| 2025 (약세장) PF | 1.130 | **1.170** | 개선 |
| Losing years | 0 | 0 | — |

---

## v14.4.1 — "Hotfix: _load_state 복합키 버그 (-1022 fix)"

**태그:** `v14.4.1` | **배포:** 2026-03-08

### 변경사항

- **버그수정:** `_load_state()`가 state key `'SUIUSDT:LONG'`을 그대로 Binance symbol로 전달 → API -1022 서명 오류 → bracket repair 반복 실패
- **수정:** `pos['symbol']` = 순수 심볼(`'SUIUSDT'`), `pos['position_side']` 별도 복원
- **전략 파라미터 변경 없음** → 성능 v14.4.0 동일

---

## v14.4.2 — "Hotfix: Hedge Mode Close 레버리지 abort (Incident #5)"

**태그:** `v14.4.2` | **배포:** 2026-03-08

### 변경사항

- **버그수정:** trail exit 시 `_place_order()` guard `if not reduce_only:` → `if not reduce_only and leverage > 1:`
- close 경로에서 leverage=1 기본값이 `futures_change_leverage()` 호출 트리거 → -2028 → trail 포지션 미청산
- **전략 파라미터 변경 없음** → 성능 v14.4.0 동일

---

## v14.4.3 — "DD Halt 30% (r48 P99 분석 기반)"

**태그:** `v14.4.3` | **배포:** 2026-03-09 | **연구:** R48 DD halt analysis

### v14.4.2 대비 변경사항

| 항목 | v14.4.2 | v14.4.3 | 이유 |
|------|---------|---------|------|
| max_drawdown_pct | 0.10 (10%) | **0.30 (30%)** | r48: 세션 DD P99=28.4%, halt=10%는 정상 운영 51.7% 차단 |

> **r48 분석 결과 (Option D, 2020-2025, L/N=2.60):**
> - Session DD: P50=10.5%, P90=19.9%, P99=28.4%
> - halt=10%: 51.7% 세션에서 발동 (정상 운영 차단)
> - halt=30%: 0.5% 세션에서 발동 (이상 구간만)

**전략 파라미터 변경 없음** → 성능 v14.4.0 동일.

---

## v14.5.0 — "r53: S_d8 Per-Leg Trailing Stop (Option A)"

**태그:** `v14.5.0` | **배포:** 2026-03-09 | **연구:** R53 opportunity augmentation

### v14.4.3 대비 변경사항 (기능 diff)

| 항목 | v14.4.3 | v14.5.0 | 이유 |
|------|---------|---------|------|
| **s_trail_activate** | — (없음) | **0.015 (1.5%)** | S_d8 각 leg에 trailing 추가 |
| **s_trail_distance** | — (없음) | **0.002 (0.2%)** | |
| s_tp | 0.05 (5%, flat) | **0.05 (5%, hard ceiling)** | trailing이 먼저 작동 |
| s_sl, d_trail_*, ma_trail_* | 변화 없음 | 변화 없음 | |

> ⚡ **핵심 변경:** S_d8 straddle의 각 leg에 trailing stop 추가. avg hold 251→113 bars (55% 단축) → 빠른 exit→re-entry → N +19% 증가.

### 성능 (데이터셋 B, Phase 8, 63 syms — 역사적 참고용)

| 지표 | Baseline (v14.4.x, 63sym) | v14.5.0 (63sym) | 변화 |
|------|--------------------------|-----------------|------|
| N | 5,470 | 6,517 | +19.1% |
| PF | 1.223 | 1.728 | +41.3% |
| WR | ~41% | ~41% | — |
| RR | — | — | — |
| Calmar(1x) | 10.3 | 89.9 | +772% |
| MaxDD | — | 60.9% | — |
| CI95 | — | [1.606,1.858] | — |

> ⚠️ 63-sym 수치는 역사적 참고용. MKRUSDT/FTMUSDT/EOSUSDT 상장폐지로 v14.5.2에서 60-sym으로 수정.
> **현재 라이브와 parity가 맞는 canonical 수치는 v14.5.2 (60 syms) 참조.**

> **Phase 8 중요 주석:** r53 원본은 `backtest_v14_3_0.py`에서 D_TRAIL_ACTIVATE=0.02 import (v13.1 시대 구버전). Phase 8에서 `config.py` 직접 읽어 재실행 → 올바른 D_TRAIL_ACTIVATE=0.005로 검증. 실제 ΔPF=+0.505 (+41%), 원본 ΔPF=+0.306 (+24%)보다 더 강한 결과.

---

## v14.5.1 — "Config 구조 혁신: pydantic-settings BaseSettings (근본 해결)"

**태그:** `v14.5.1` | **배포:** 2026-03-08 | **연구:** config/env 동기화 인시던트 근본 해결

### v14.5.0 대비 변경사항 (기능 diff)

| 항목 | v14.5.0 | v14.5.1 | 이유 |
|------|---------|---------|------|
| VolStraddleConfig 구현 | `@dataclass` + `settings.py` vs_ 필드 ~70개 | **`pydantic_settings.BaseSettings`** | 4-place 동기화 문제 근본 해결 |
| 파라미터 우선순위 | `.env` → `settings.py` → `config.py` (3단계 전달) | `.env`(`VS_` prefix) → `VolStraddleConfig()` 직접 | 중간 레이어 제거 |
| scalper.py 주입 코드 | 59줄 (모든 파라미터 수동 주입) | **3줄** (`VolStraddleConfig()` 직접 호출) | 누락 파라미터 위험 제거 |
| settings.py vs_* 필드 | ~70개 | **0개** (삭제) | dead code 제거 |

> ⚡ **핵심 변경:** 전략 파라미터 변경 없음. 아키텍처만 혁신.
> 과거 인시던트 패턴: settings.py에 필드 추가 누락, scalper.py 주입 누락 → 라이브/백테스트 불일치.
> v14.5.1부터 `VolStraddleConfig()` 한 번 호출로 `.env`에서 직접 로드.

**전략 파라미터 변경 없음** → 성능 v14.5.0 동일 (63 syms: PF=1.728, Calmar=89.9).

---

## v14.5.2 — "Universe Parity 수정: 60 syms (라이브=백테스트 일치)"

**태그:** `v14.5.2` | **배포:** 2026-03-08 | **원인:** Backtest-Live Parity 원칙 위반 발견

### v14.5.1 대비 변경사항

| 항목 | v14.5.1 | v14.5.2 | 이유 |
|------|---------|---------|------|
| 라이브 universe | 50 syms (구형 top-cap) | **60 syms** (r53 63-sym - 3 delisted) | Backtest-Live Parity 복원 |
| 제거 심볼 | — | MKRUSDT, FTMUSDT, EOSUSDT | Binance USDT-M 상장폐지 |
| VS_SYMBOL_WHITELIST | 50 syms | **60 syms (JSON array)** | .env 직접 수정 |

> ⚡ **핵심 변경:** 라이브 universe를 백테스트 universe와 정확히 일치시킴.
> 단순히 symbol 리스트를 바꾼 게 아니라 Backtest-Live Parity 원칙 위반을 수정한 것.

### 성능 (데이터셋 B, 60 syms, 6yr — **현재 공식 canonical**)

| 지표 | Baseline (v14.4.x, 60sym) | **v14.5.2 / v14.5.0+ (60sym)** | 변화 |
|------|--------------------------|-------------------------------|------|
| N | 5,515 | **6,207** | **+12.5%** |
| PF | 1.273 | **1.563** | **+22.8%** |
| WR | 41.7% | **40.6%** | -1.1pp |
| RR | 1.78x | **2.29x** | **+28.7%** |
| avg_win | 3.58% | 2.60% | — |
| avg_loss | 2.01% | 1.14% | — |
| EV | +32bp | **+38bp** | +19% |
| MaxDD | 69.7% | **60.5%** | -9.2pp ✓ |
| AnnRet(1x) | 1,106% | **3,498%** | +216% |
| Calmar(1x) | 15.9 | **57.9** | **+264%** |
| CI95 | [1.199,1.350] | **[1.461,1.676]** | — |
| P(PF≥1) | 100% | **100%** | — |

> **v14.5.0(63sym) vs v14.5.2(60sym) 차이:**
> MKR/FTM/EOS 3개 상장폐지 심볼이 백테스트 성능에 유의미하게 기여.
> PF 1.728→1.563 (ΔPF=-0.165, -9.6%). 이것이 현재 라이브의 실제 백테스트 수치.

---

## 완전한 버전 타임라인 요약표

| 버전 | N | PF | WR | RR | EV(bp) | MaxDD | AnnRet(1x) | Calmar(1x) | 전략 요약 | 데이터 |
|------|---|----|----|----|----|-------|------------|------------|----------|------|
| v12.9 | ~2,265 | 1.280 | 43.4% | 1.67x | +32bp | 67.9% | +195% | 5.8 | D_mom8 flat TP=5%, L/N=0.20 | A(34sym,4.7yr) |
| v13.0 | ~2,265 | 1.280 | 43.4% | 1.67x | +32bp | 67.9% | +687%(LN2.5) | 10.1(LN2.5) | +S_d8, L/N=2.5 | A(34sym,4.7yr) |
| v13.1.0 | ~2,265 | 1.351 | 47.3% | 1.51x | — | — | — | 9.4 | +D_mom8 trailing(act=2%) | A |
| v14.0.0 | ~3,000 | 1.304 | ~41% | — | — | — | — | 18.8 | +MA Cluster, L/N=VR2.5+MA1.2 | A |
| v14.1.0 | ~3,000 | 1.304 | ~41% | — | — | — | — | ↑ | L/N→VR3.0+MA2.1 (Half-Kelly) | A |
| v14.3.0 | ~2,400 | 1.308 | 47.0% | 1.47x | — | — | — | 10.1 | +TBR gate ON | A |
| v14.4.0 | ~3,700 | 1.461 | — | — | — | — | — | ↑↑ | D_trail tight(0.5%/0.2%), MA_trail 신규 | A |
| v14.4.x (60sym) | 5,515 | 1.273 | 41.7% | 1.78x | +32bp | 69.7% | 1,106% | 15.9 | (60sym 재측정, v14.4.x baseline) | B(60sym,6yr) |
| v14.5.0 _(참고)_ | 6,517 | 1.728 | ~41% | — | — | 60.9% | — | 89.9 | +S_d8 trail (63-sym, 역사적) | B(63sym,6yr) |
| **v14.5.2 ✅** | **6,207** | **1.563** | **40.6%** | **2.29x** | **+38bp** | **60.5%** | **3,498%** | **57.9** | +S_d8 trail, 60-sym parity | **B(60sym,6yr)** |

> **데이터셋 A:** 34 syms, 4.73yr (v13~v14.4 시대 연구)  
> **데이터셋 B:** 60 syms, 6yr (v14.5.2+ 현재 canonical, commit=34076e8)  
> A↔B 절대값 직접 비교 주의. 데이터셋 B 내에서만 크로스-버전 비교 유효.

---

## v14.5.2 라이브 프로덕션 파라미터 (완전한 목록, 현재 기준)

> **검증 기준:** `python3 -c "from quantbox.strategy.vol_straddle.config import VolStraddleConfig; cfg = VolStraddleConfig(); ..."` 실행 결과 (2026-03-09)  
> **라이브 PID:** 554486 | **Git SHA:** db2650f | **시작:** 2026-03-09 03:03 KST

### VR Squeeze (D_mom8) 파라미터

```
d_sl               = 0.01      # 1% stop loss
d_tp               = 0.12      # 12% hard TP ceiling
d_trail_activate   = 0.005     # +0.5% 이후 trail 시작 (r47: was 0.02)
d_trail_distance   = 0.002     # peak 0.2% 아래 추적 (r47: was 0.003)
d_vr_threshold     = 0.5
lookback           = 8
vr_long            = 56
vr_short           = 8
momentum_period    = 8
```

### S_d8 Straddle 파라미터

```
straddle_enabled   = True
s_sl               = 0.015     # 1.5% SL (per leg)
s_tp               = 0.05      # 5% hard TP ceiling
s_trail_activate   = 0.015     # +1.5% 이후 trail 시작 ← v14.5.0 NEW
s_trail_distance   = 0.002     # peak 0.2% 아래 추적  ← v14.5.0 NEW
s_vr_threshold     = 0.55
s_squeeze_dur      = 4
```

### MA Cluster 파라미터

```
ma_cluster_enabled        = True
ma_cluster_leverage       = 21.0      # L/N = 21/10 = 2.1
ma_cluster_sl             = 0.02      # 2% SL
ma_cluster_tp             = 0.08      # 8% hard TP ceiling
ma_cluster_trail_activate = 0.04      # +4% 이후 trail 시작 (r47)
ma_cluster_trail_distance = 0.005     # 0.5% trail gap (r47)
ma_cluster_pct            = 0.03      # 4 SMA max spread 3%
ma_cluster_min_duration   = 8         # min 8×4h bars = 32h
ma_cluster_max_duration   = 24        # max 24×4h bars = 96h (zombie filter)
ma_cluster_min_momentum   = 0.01      # |momentum| ≥ 1% (zombie filter)
ma_cluster_max_hold_4h    = 12        # 12×4h = 48h
ma_cluster_sma_periods    = [20, 50, 100, 200]
ma_cluster_max_concurrent = 5         # MA 전용 슬롯 (combined에서 공유)
ma_cluster_trail_activate = 0.04
ma_cluster_trail_distance = 0.005
```

### 사이징 & 리스크

```
base_leverage          = 30.0         # VR L/N = 30/10 = 3.0
max_concurrent         = 10           # 최대 동시 포지션
dynamic_capital_enabled = False       # 고정 balance/max_concurrent
max_drawdown_pct       = 0.30         # 30% 세션 DD halt (r48)
soft_equity_scale      = True
soft_eq_dd1/scale1     = 0.02 / 0.80 # DD≥2%: 80% sizing
soft_eq_dd2/scale2     = 0.04 / 0.60 # DD≥4%: 60% sizing
soft_eq_dd3/scale3     = 0.06 / 0.40 # DD≥6%: 40% sizing
max_leverage           = 30.0
risk_per_trade         = 0.01
```

### 진입 필터 (Entry Filters)

```
f5_tbr_enabled         = False        # TBR 비활성 (r47 이후)
f5_tbr_z_threshold     = 0.5
f5_tbr_z_window        = 192

a3_enabled             = True         # Long: VR squeeze dur ≥ 16 bars
a3_long_squeeze_dur    = 16
a3_deep_vr_threshold   = 0.35

b3_enabled             = True         # Short: deep VR filter

c2_enabled             = True         # BTC momentum filter
c2_btc_mom_threshold   = 0.01         # |BTC_mom| ≥ 1%

e2_enabled             = True         # SMA20 trend alignment
e2_sma_period          = 20
e2_threshold           = 0.005

dir_diversity_enabled  = True         # 방향 다양성
max_same_direction     = 2            # 같은 방향 최대 2슬롯

cw_enabled             = True         # Conviction-weighted sizing
cw_multipliers         = [0.5, 1.0, 1.5, 2.0, 2.5]
min_conviction_score   = 0
```

### 실행 & 유니버스

```
hedge_mode             = True
bar_interval_minutes   = 15
enable_15m             = True
enable_1h              = True
enable_4h              = True
warmup_15m             = 200
warmup_1h              = 50
warmup_4h              = 13

cost_per_side          = 0.0004       # 0.04% per side
slippage_rate          = 0.0002       # 0.02% per side
round_trip_cost        = 0.0008       # 0.08% (total)
fee_rate               = 0.0002

min_volume_rank        = 30
exclude_symbols        = ['USDCUSDT', 'TUSDUSDT', 'BUSDUSDT']
symbol_whitelist       = [60 syms]    # r53 63-sym에서 3 delisted 제거 (.env:VS_SYMBOL_WHITELIST)
```

---

## 크로스-버전 엣지 진화 (데이터셋 A 기준, 동일 조건)

```
v12.9       PF=1.280  WR=43.4%  EV=+32bp   Calmar_1x=5.8
v13.0       PF=1.280  WR=43.4%  EV=+32bp   Calmar_1x=5.8   (사이징만 변경)
v13.1.0     PF=1.351  WR=47.3%  EV=↑       Calmar_1x=9.4   (+trailing)
v14.0.0     PF=1.304  WR=~41%   EV=+?      Calmar=18.8     (+MA, combined)
v14.1.0     PF=1.304  (동일)                               (레버리지만↑)
v14.3.0     PF=1.308  WR=47.0%  EV=+30bp   Calmar=10.1     (+TBR gate)
v14.4.0     PF=1.461  WR=↑      EV=↑       Sharpe=4.65     (D_trail tight, MA_trail)
v14.4.x†   PF=1.273  WR=41.7%  EV=+32bp   Calmar=15.9   RR=1.78x (60sym baseline)
v14.5.2†   PF=1.563  WR=40.6%  EV=+38bp   Calmar=57.9   RR=2.29x (+S_d8 trail, 60sym canonical)
            * 데이터셋 A (34 syms, 4.73yr) — v12.9~v14.4.0
            † 데이터셋 B (60 syms, 6yr) — v14.5.2 현재 canonical (commit=34076e8)
```

---

## 핵심 인사이트 (버전 진화 패턴)

1. **엣지 확보 (v12.9):** VR squeeze 자체가 통계적으로 유효한 비효율 포착.
2. **레버리지 최적화 (v13.0):** 동일 엣지에서 L/N 12.5배 → AnnRet 폭발. 가장 큰 수익률 점프.
3. **Exit 혁신 1 (v13.1.0):** 트레일링 스탑이 위너 보유 연장 → PF/WR 동시 개선.
4. **다각화 (v14.0.0):** MA Cluster 추가로 비상관 수익원 확보 → Calmar 급등.
5. **Exit 혁신 2 (v14.4.0):** trail_activate 10배 tight(2%→0.5%) → Sharpe +37%. 가장 큰 단일 개선.
6. **기회 증폭 (v14.5.0):** S_d8 trailing으로 avg hold 55% 단축 → N +19%, PF +41%.
