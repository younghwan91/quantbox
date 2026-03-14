---
applyTo: "research/**/*.py,scripts/**/*.py,quantbox/research/**/*.py"
---
# Backtesting Skill — QuantBox VR Squeeze

> **목적:** AI 에이전트가 이 문서를 읽고 독립적으로 올바른 백테스트를 수행할 수 있도록.
> 과거 실패 사례에서 추출한 규칙. **모든 항목은 실제 수치 오류 또는 자금 손실에서 비롯됨.**

---

## 0. 시작 전 체크리스트 (Phase 0 — Pre-flight)

새 백테스트 스크립트를 작성하기 전에 **반드시** 이 항목들을 실제 파일에서 읽어 확인한다.

```python
# 1. 파라미터는 반드시 config.py에서 읽는다. 하드코딩 금지.
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()

D_TP          = cfg.d_tp                              # 0.20 (20% dead cap — trail exits first)
D_SL          = cfg.d_sl                              # 0.01 (1%)
D_TRAIL_ACT   = cfg.d_trail_activate                  # 0.01 (1.0%)
D_TRAIL_DIST  = cfg.d_trail_distance                  # 0.002 (0.2%)
S_TRAIL_ACT   = cfg.s_trail_activate                  # 0.015 (1.5%)
S_TRAIL_DIST  = cfg.s_trail_distance                  # 0.001 (0.1%, r76: was 0.002)
MA_TRAIL_ACT  = cfg.ma_cluster_trail_activate         # 0.02 (2%)
MA_TRAIL_DIST = cfg.ma_cluster_trail_distance         # 0.003 (0.3%, r75: was 0.005)
VR_LN         = cfg.base_leverage / cfg.max_concurrent  # 30/10=3.0
MA_LN         = cfg.ma_cluster_leverage / cfg.max_concurrent  # 21/10=2.1
TBR_ENABLED   = cfg.f5_tbr_enabled                   # False (r47 이후)
```

**체크 항목:**
- [ ] `d_tp` 확인 (0.20 — dead cap; 실제 청산은 trail이 전담. 0.05/0.12 아님)
- [ ] trailing stop 파라미터 확인 (`d_trail_activate`, `d_trail_distance`)
- [ ] `f5_tbr_enabled` 확인 (현재 False)
- [ ] `VR_LN = base_leverage / max_concurrent` (3.0). 하드코딩 2.5 등 금지
- [ ] `MA_LN = ma_cluster_leverage / max_concurrent` (2.1). 하드코딩 금지
- [ ] **심볼 universe = `cfg.symbol_whitelist`** (현재 60개). 하드코딩 금지
- [ ] **`cfg.exclude_symbols` 방식으로 로딩 금지** → 스테이블코인만 제외해 상장폐지 심볼 포함됨

> ⚠️ **Backtest-Live Universe Parity 원칙 (절대 규칙):**
> **라이브 트레이딩 universe는 항상 백테스트 universe와 동일해야 한다.**
> 두 개가 달라지는 순간 백테스트 수치는 무효. 그것이 "라이브가 달라서" 재실행하는 것이지,
> "universe를 바꿔서" 재실행하는 게 아니다. 둘의 차이가 생기면 무조건 재실행.
> - 상장폐지 발견 시: 라이브 제거 → 백테스트 재실행 → 공식 수치 업데이트 (순서 무관, 동시에)
> - 신규 심볼 추가 시: 백테스트 먼저 → 검증 통과 후 라이브 추가 (백테스트 선행 필수)

**심볼 로딩 코드 패턴 (반드시 이 방식):**
```python
# ✅ 올바른 방식 — 60 라이브 심볼만
whitelist = set(cfg.symbol_whitelist)
for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
    sym = f.name.split("_")[0]
    if sym not in whitelist:
        continue

# ❌ 금지 — exclude_symbols는 스테이블코인만 제외, 상장폐지 심볼 포함됨
exclude = set(cfg.exclude_symbols)  # USDCUSDT/TUSDUSDT/BUSDUSDT만 제외
# → MKRUSDT, FTMUSDT, EOSUSDT 포함됨 (상장폐지!) → 63sym
```

> ⚠️ **위반 선례 (2026-03-11):** r55/r56/r57이 `exclude_symbols` 방식으로 로딩 →
> 상장폐지 3개(MKR/FTM/EOS) 포함 63심볼로 백테스트. 라이브(60sym)와 조건 불일치.
> `symbol_whitelist` 방식으로 수정 후 전부 재실행함.

> ⚠️ **Universe 불일치 발생 시 즉시 해야 할 것:**
> 1. `cfg.symbol_whitelist`로 로딩하는 스크립트인지 먼저 확인
> 2. 라이브 60sym과 백테스트 sym 수가 다르면 → 결과 즉시 무효, 재실행
> 3. 신규 심볼 추가 시: 백테스트 먼저 → 검증 통과 → 라이브 추가 (순서 불변)

---

## 1. 시뮬레이션 함수 — 절대 재구현 금지

```python
# ✅ 반드시 이 경로에서 import
from quantbox.backtest.trade_sim import (
    collect_trades,         # 신호 → 개별 거래 목록
    portfolio_sim,          # 슬롯 경쟁 시뮬레이션 (N개 동시 보유 제한)
    compute_metrics,        # PF, EV, Sharpe, MaxDD, bootstrap CI
    bootstrap_pf,           # 부트스트랩 신뢰구간
    precompute_all_vr,      # VR 신호 사전계산
    sim_directional_trailing,   # trail-aware D_mom8 시뮬레이션 (v14.24.0, r43에서 이전)
    sim_straddle_trailing,      # trail-aware S_d8 시뮬레이션 (v14.24.0, r43에서 이전)
)
```

**왜 재구현 금지인가?**
- `momentum=0` 처리, squeeze counter 리셋, dedup 순서 등 미묘한 차이 → ±5-10% 수치 이탈
- `portfolio_sim`은 Numba 최적화. 재구현 시 성능 저하 + 버그 위험

**`compute_metrics()` 출력의 의미:**
```
compute_metrics(pnl, data_years) → {
    "N": 총 거래 수,
    "PF": profit factor (수익 합 / 손실 합),
    "WR": 승률 — **이미 퍼센트 스케일** (50.6, NOT 0.506). ×100 하면 5060% 버그!
    "EV_bp": 기댓값(bp) per trade — 레버리지 무관한 순수 엣지 지표,
    "RR": risk/reward ratio,
    "max_dd_pct": 최대 낙폭 (1x 복리 기준, %).
    "ann_ret_pct": 연간 수익률 (1x compound, %),
    "calmar": Calmar ratio,
    ...
}
```

> ⚠️ `compute_metrics()`는 **1x compound** (레버리지 미적용) 수익률만 계산.
> 프로덕션 수익률 ≠ `compute_metrics()` 출력. 아래 Section 4 참조.

---

## 2. Phase 1 — 베이스라인 검증 (필수, 건너뛰기 금지)

모든 백테스트 스크립트의 **첫 번째 출력**은 최신 정식 리포트와의 베이스라인 대조여야 한다.

```python
# Phase 1: 정적 파라미터로 베이스라인 재현
# ✅ collect_trades() 에 전달하는 cfg dict — 올바른 키 이름
static_cfg = {
    "sp":    cfg.vr_short,           # e.g. 4
    "lp":    cfg.vr_long,            # e.g. 24
    "d_vr":  cfg.d_vr_threshold,     # directional VR threshold
    "s_vr":  cfg.s_vr_threshold,     # straddle VR threshold
    "s_dur": cfg.s_squeeze_dur,      # squeeze duration filter
    "d_sl":  cfg.d_sl,               # directional SL (0.01)
    "s_sl":  cfg.s_sl,               # straddle SL (0.015)
    "tp":    cfg.d_tp,               # TP cap (0.20 — trail exits first)
}
baseline_trades = collect_trades(static_cfg, data_15m, data_1h, data_4h, strat="D_mom8")
baseline_pnl    = portfolio_sim(baseline_trades, max_concurrent=10)
baseline_m      = compute_metrics(baseline_pnl, data_years)

print(f"[Phase 1 Baseline] N={baseline_m['N']}, PF={baseline_m['PF']:.3f}")
# 최신 정식 리포트 값과 ±5% 이내인지 확인
```

**허용 기준:** 최신 리포트 대비 N ±1%, PF ±1%  
**이탈 시 → 즉시 중단.** 데이터 범위, config, 함수 버전 확인 먼저.

**최신 정식 베이스라인 (v14.21.0, 2026-03-14 기준, commit=b029ae7):**

| 버전 | 포함 전략 | 유니버스 | 기간 | N | PF | Calmar | DD | 비고 |
|------|---------|---------|------|---|----|--------|----|-----|
| **v14.5.0 (프로덕션)** | D_mom8+MA | **60 syms** | 6yr | **6,207** | **1.563** | **57.9** | 60.5% | canonical |
| r73 3전략 trail | **D+S+MA** | **60 syms** | 5.67yr | 9,773 | 2.002 | 1695.6 | 40.5% | s_trail_dist=0.002 기준 |
| **r76 3전략 trail** | **D+S+MA** | **60 syms** | **5.67yr** | **9,775** | **2.484** | **9882** | **37.1%** | **trail canonical (s_trail_dist=0.001)** |
| **r77 S_d8 단독** | **S_d8** | **60 syms** | **6yr** | **7,730** | **9.603** | **20,601** | **3.5%** | **룩어헤드 검증 완료 ✅** |
| r84 G_4TF (현행) | **S_d8** | **60 syms** | **6yr** | **6,290** | **10.23** | **10,175** | **2.6%** | max_concurrent=5 (슬롯 패리티) |

> **r76 trail = 현재 프로덕션 기준 수치 (v14.21.0+).** r73은 이전 기준.
> **r77 S_d8 감사 결과 (2026-03-14):**
> - 룩어헤드 없음 ✅ (4개 체크 모두 통과)
> - S_d8 Calmar=20,601 vs D_mom8 Calmar=17.5 → **1177배 우월**
> - 6년 모두 수익, MaxDD=3.5%, Bootstrap P(PF≥1.0)=100%
> - **⚠️ 핵심 cliff edge: s_trail_activate=0.015는 매우 sharp한 peak**
>   ±0.003 변화만으로 PF 60-75% 붕괴. 실거래 모니터링 필수.
> **r84 TF별 분석 (2026-03-15):** 4h TF가 가장 좋은 신호 품질(PF=61, EV=150bp)이나
> G_4TF가 8.1× 더 좋은 Calmar (슬롯 활용도 때문). 15m이 전체 N의 53.5% 기여.
> **r85 2D trail sweep (2026-03-15):** act=0.015는 genuine peak (양방향 cliff 확인).
> (act=0.015, dist=0.0005) IS Calmar=24,204 (+138%) — OOS 검증 전 변경 금지.
> **universe 변경 시 반드시 이 표와 아래 assert 값 업데이트.**

> **v14.24.0 추가:** `sim_directional_trailing()` 및 `sim_straddle_trailing()`이 `research/active/r41_r45/r43_exit_improvement.py`에서 `quantbox/backtest/trade_sim.py`로 이전됨 (v14.24.0).  
> 모든 연구 스크립트 import 자동 업데이트 완료. 기존 import 경로(r43)는 deprecation 경고 출력.

---

## 3. 룩어헤드 바이어스 — 체크리스트

> ⚠️ **실전 교훈 (2026-02-26):** `i // ratio`가 개발 중인 4h 바의 최종 close를 참조
> → 백테스트 WR 83%, 실전 WR 36% (2.3배 차이). 전략 완전 무효.

**멀티 타임프레임 참조 규칙:**

```python
# ❌ WRONG: 현재 15m 바 i가 속한 4h 바 (아직 진행 중)
j_4h = i // 16

# ✅ CORRECT: 완성된 이전 4h 바
j_4h = (i // 16) - 1
```

**바 진입 타이밍:**
```python
# 거래 진입: entry_bar+1의 open 가격 사용 (bar close 후 다음 바 시작)
# 거래 청산: exit_bar의 SL/TP/max_hold 체크 (intra-bar 해상도)
# → trade_sim.py가 이미 올바르게 구현. 재구현 없이 import.
```

**의심 신호 (발견 즉시 중단):**
- 백테스트 WR이 라이브의 2배 이상
- 보유 기간이 비현실적으로 짧음 (15m 전략에서 평균 1-2바)
- 특정 연도에 비정상적으로 높은 승률

---

## 4. 수익률 보고 — 레버리지 모델 명시 필수

### 두 모델의 차이

```
1x compound (compute_metrics 기본):
  equity *= (1 + pnl_pct)

Production L/N compound (프로덕션 실제):
  equity *= (1 + pnl_pct × LN)
  where LN = base_leverage / max_concurrent
        VR LN = 30/10 = 3.0
        MA LN = 21/10 = 2.1
        Blended ≈ 2.7 (VR+MA 혼합 시)
```

### 보고서에 반드시 포함할 항목

```
# 올바른 보고 예시
PF=1.521 | EV=+58bp (레버리지 무관 순수 엣지)
1x CAGR: +X%  (compute_metrics 출력값)
L/N=3.0 VR CAGR: +X% (4.7년 복리)
데이터: 2020-01-01 ~ 2025-12-31 (5.5년)
슬롯: 10 (max_concurrent=10)
비용: 0.08% round-trip (entry+exit)
```

> **자주 발생하는 실수:** `compute_metrics` 1x CAGR을 프로덕션 수익률로 제시.
> 동일 PF라도 기간이 다르면 CAGR이 2배 이상 차이날 수 있음. **기간 항상 명시.**

---

## 5. 포지션 사이징 검증

> ⚠️ **실전 교훈 (2026-03-05):** `n_active=0` 단일 진입만 테스트 → 2번째 슬롯에서 마진 부족 `-2019`.

**반드시:**
- `portfolio_sim(trades, max_concurrent=10)` — VR/MA 단독 또는 혼합 백테스트
- `portfolio_sim(trades, max_concurrent=5)` — **STRD(S_d8) 단독 백테스트** ← r79 발견
  
마진 공식: `balance / 10 × 30 (VR)` or `balance / 10 × 21 (MA)` ≤ available margin.

> ⚠️ **STRD 슬롯 패리티 규칙 (r79 발견 → T1에서 코드 수정 완료, commit 3b43bb2):**
> 1 straddle = 2 positions → `n_active += 2` → live에서 max 5 straddle 동시 가능.
> 백테스트에서 `max_concurrent=10` + `straddle_syms` 미사용 시 N이 **18.6% 과대추정** (r79 확인).
>
> **STRD 단독 백테스트:**
> ```python
> pnl = portfolio_sim(strd_trades, max_concurrent=5)
> ```
>
> **혼합(VR + STRD) 백테스트 — 반드시 `straddle_syms` 전달:**
> ```python
> from quantbox.backtest.trade_sim import get_straddle_syms
> strd_syms = get_straddle_syms(strd_trades)
> pnl = portfolio_sim(vr_trades + strd_trades, 10, straddle_syms=strd_syms)
> ```
> `straddle_syms` 없이 `max_concurrent=10`만 쓰면 straddle이 1슬롯으로 처리 → 수치 무효.

---

## 6. 비용 모델

```python
COST_DIRECTIONAL = 0.0008  # 0.08% round-trip
COST_STRADDLE    = 0.0016  # 0.16% round-trip (long+short)
# 펀딩비: 8h마다 ~0.01%, 48h 보유 시 ~0.06% 추가
```

---

## 7. 데이터 범위 및 워밍업

```python
WARMUP_15m = 200; WARMUP_1h = 50; WARMUP_4h = 13  # ceil(200/16)
# 권장: 2020-01-01 이후 (5년+). 그 이전은 Binance 유동성 부족.
```

---

## 8. 통계 유효성 기준

| 지표 | 최소 | 권장 |
|------|------|------|
| N | ≥ 500 | ≥ 2,000 |
| PF | > 1.05 | > 1.20 |
| Bootstrap P(PF≥1.0) | ≥ 90% | ≥ 95% |
| OOS50 PF | > 1.03 | > 1.15 |
| OOS/IS 괴리 | < 10% | < 5% |
| 손실 연도 수 | 0 | 0 |

OOS 검증: 50% split, `oos_pf / is_pf ≥ 0.85`이면 과적합 아님.

---

## 9. 연구 스크립트 4-Phase 구조

```python
"""r{N}: 목적 한 줄."""
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import collect_trades, portfolio_sim, compute_metrics, bootstrap_pf, precompute_all_vr

# Phase 0: Config 읽기 (하드코딩 금지)
cfg = VolStraddleConfig()
baseline_cfg = {"d_tp": cfg.d_tp, "d_sl": cfg.d_sl,
                "d_trail_activate": cfg.d_trail_activate, "d_trail_distance": cfg.d_trail_distance,
                "f5_tbr_enabled": cfg.f5_tbr_enabled}

# 심볼 로딩 — 반드시 symbol_whitelist 사용 (exclude_symbols 금지)
whitelist = set(cfg.symbol_whitelist)  # 60 live symbols

# Phase 1: 베이스라인 재현 → ±1% 확인 후 진행 (canonical: N=6207, PF=1.563)
b_pnl = portfolio_sim(collect_trades(baseline_cfg, ...), max_concurrent=cfg.max_concurrent)
b_m   = compute_metrics(b_pnl, DATA_YEARS)
assert abs(b_m['N'] - 6207) / 6207 < 0.05, f"N drift: {b_m['N']}"   # ±5% (연구 스크립트 구현 차이 허용)
assert abs(b_m['PF'] - 1.563) / 1.563 < 0.05, f"PF drift: {b_m['PF']:.3f}"

# Phase 2: 파라미터 스윕
for param_value in param_grid:
    m = compute_metrics(portfolio_sim(collect_trades({**baseline_cfg, "p": param_value}, ...), max_concurrent=10), DATA_YEARS)
    boot = bootstrap_pf(...)
    results.append({...})

# Phase 3: 보고서 → docs/backtest_results/YYYYMM/YYYYMMDD_{commit}_{desc}.md
```

---

## 10. 보고서 파일명 및 필수 섹션

**파일명:** `docs/backtest_results/YYYYMM/YYYYMMDD_{commit_short}_{description}.md`

**필수 포함:**
- 데이터 범위 (2020-01-01 ~ YYYY-MM-DD, X년)
- 슬롯: 10, 비용: 0.08% round-trip
- 베이스라인 N/PF (±1% 검증 통과 여부)
- 수익률 모델 명시: **1x compound** 또는 **L/N=3.0**
- Parity Notes: config.py 읽음 ✅, 룩어헤드 없음 ✅
- **구현:** `미구현` 또는 `feature/xxx → vX.Y.Z` (결과가 전략에 반영되면 업데이트)

---

## 11. 과거 실패 사례 요약 (Quick Reference)

| 날짜 | 실패 | 증상 | 원인 | 규칙 |
|------|------|------|------|------|
| 2026-03-07 | 하드코딩 파라미터 | N/PF 2배 차이 | `tp=0.05` ≠ `cfg.d_tp` (현재 0.20 dead cap) | 항상 `cfg = VolStraddleConfig()` |
| 2026-02-26 | 4h MTF 룩어헤드 | WR 83% vs 실전 36% | `i//16` (개발중 바) | `(i//16) - 1` (완성 바) |
| 2026-03-05 | 동적 자본 공식 오류 | 슬롯 2에서 마진 부족 | `balance/(n+1)` → 첫 슬롯 50% | 고정 `balance/max_concurrent` |
| 2026-03-06 | 알고 주문 상태 미검출 | SL/TP 체결 누락 | `TRIGGERED`만 확인, `FINISHED` 누락 | `TRIGGERED` + `FINISHED` 둘 다 |
| 2026-02-21 | Reduce-only 레버리지 | -2028 API 에러 | 청산 주문에 `change_leverage()` 호출 | 청산 시 레버리지 변경 금지 |
| 2026-03-08 | Hedge Mode trail abort | trail exit 실패 (-2028) | close가 `leverage=1` 기본값 전달 → 변경 시도 | `if not reduce_only and leverage > 1` (v14.4.2) |
| 2026-03-09 | `.env` VS_ 파라미터 덮어쓰기 | config.py 변경 적용 안 됨 | 38개 VS_ 중복 in `.env` (pydantic-settings 우선순위) | `.env`는 override만. 기본값은 `config.py`만. |
| 2026-03-11 | Universe 불일치 (r55/r56/r57) | 63sym 백테스트 vs 60sym 라이브 | `exclude_symbols`는 스테이블코인만 제외, 상장폐지 포함 | 반드시 `cfg.symbol_whitelist` 사용 |
| 2026-03-11 | _offset 미적용 (r58 이전) | 단축 히스토리 심볼 슬롯 충돌 오류 | 신규 심볼 bar 0 ≠ 기존 심볼 bar 0 (시간 불일치) | 신규 심볼은 `data_15m[sym]["_offset"]` 적용 필수 |
| 2026-03-15 | STRD 슬롯 패리티 오류 (r78) | N 18.6% 과대추정 | `portfolio_sim(max=10)` → 10 straddle 허용, 라이브는 max 5 | STRD 단독: `portfolio_sim(max_concurrent=5)` |

파라미터 민감도 분석: `research/active/r66_r70/r67_sensitivity.py` 및 `docs/backtest_results/202603/` 참조.

---

## 12. `.env` 운영 원칙 — 중복 추가 금지

> `config.py`가 source of truth. `.env`는 override 전용. 상세 규칙은 `copilot-instructions.md` 참조.

**요약:** `VS_*` 기본값과 같은 값을 `.env`에 중복 추가 금지. 추가하면 pydantic-settings가 config.py 변경을 무시함.

---

## 13. 연구 브랜치 컨벤션

Git flow 규칙은 `copilot-instructions.md` 참조. 요약: `research/r{N}-{desc}` 브랜치 → `--no-ff` 머지 (이력 보존).

---

## 14. Universe Expansion — _offset 적용 규칙 (r58)

whitelist 외 심볼(단축 히스토리)을 추가할 때 반드시 `_offset`을 적용해야 시간 정렬이 맞는다.

```python
# cache 로드 시 _offset 계산 (예: APTUSDT=3.3yr → max_bars=210,240 기준)
data_15m[sym]["_offset"] = max_bars - len(close_arr)  # e.g. 95,240

# collect 함수 내부 — trade bar를 global bar로 변환
offset = data_15m[sym].get("_offset", 0)
entry_global = local_bar + offset  # portfolio_sim에 전달하는 bar index
```

**미적용 시 증상:** 신규 심볼의 bar 0이 기존 심볼의 bar 0과 동일 시각으로 취급
→ portfolio_sim이 서로 다른 날짜의 거래를 같은 슬롯 경쟁으로 봄 → N 과대/과소 오류.

**r58 핵심 발견 (s_vr threshold):**
- `s_vr_threshold` 0.55→0.59: N+72%, PF+24% **동시** 개선 — "moderate squeeze"가 "deep squeeze"보다 더 좋은 엣지
- `s_squeeze_dur` 4→2: N+25%, PF+12% — 현재 `s_dur=4`는 너무 엄격, 2~3이 최적
- `d_vr_threshold` 완화: N 크게 증가하지만 EV 단조 감소 → 권장 안 함
- 연도별 EV 감소(2021→2025)는 시장 구조 변화(강세장→정상화)이며 전략 악화 아님

