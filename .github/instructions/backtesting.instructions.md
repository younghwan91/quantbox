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

D_TP          = cfg.d_tp                              # 0.12 (12%)
D_SL          = cfg.d_sl                              # 0.01 (1%)
D_TRAIL_ACT   = cfg.d_trail_activate                  # 0.005 (0.5%)
D_TRAIL_DIST  = cfg.d_trail_distance                  # 0.002 (0.2%)
MA_TRAIL_ACT  = cfg.ma_cluster_trail_activate         # 0.04 (4%)
MA_TRAIL_DIST = cfg.ma_cluster_trail_distance         # 0.005 (0.5%)
VR_LN         = cfg.base_leverage / cfg.max_concurrent  # 30/10=3.0
MA_LN         = cfg.ma_cluster_leverage / cfg.max_concurrent  # 21/10=2.1
TBR_ENABLED   = cfg.f5_tbr_enabled                   # False (r47 이후)
```

**체크 항목:**
- [ ] `d_tp` 확인 (0.12. 0.05 아님)
- [ ] trailing stop 파라미터 확인 (`d_trail_activate`, `d_trail_distance`)
- [ ] `f5_tbr_enabled` 확인 (현재 False)
- [ ] `VR_LN = base_leverage / max_concurrent` (3.0). 하드코딩 2.5 등 금지
- [ ] `MA_LN = ma_cluster_leverage / max_concurrent` (2.1). 하드코딩 금지

> ⚠️ **위반 선례 (2026-03-07):** `backtest_v14_3_0.py`가 `tp=0.05` + `VR_LN=2.5`, `MA_LN=1.2` 하드코딩 →
> 실제 프로덕션 `tp=0.12` + trailing + `VR_LN=3.0`, `MA_LN=2.1`. N/PF/CAGR 전부 틀린 수치 보고.

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
    "WR": 승률,
    "EV": 기댓값(bp) per trade — 레버리지 무관한 순수 엣지 지표,
    "Sharpe": 연간 샤프,
    "MaxDD": 최대 낙폭 (1x 복리 기준),
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
static_cfg = {
    "d_tp": 0.12, "d_sl": 0.01,
    "d_trail_activate": 0.005, "d_trail_distance": 0.002,
    ...  # config.py에서 읽은 값
}
baseline_trades = collect_trades(static_cfg, ...)
baseline_pnl    = portfolio_sim(baseline_trades, max_concurrent=10)
baseline_m      = compute_metrics(baseline_pnl, data_years)

print(f"[Phase 1 Baseline] N={baseline_m['N']}, PF={baseline_m['PF']:.3f}")
# 최신 정식 리포트 값과 ±1% 이내인지 확인
```

**허용 기준:** 최신 리포트 대비 N ±1%, PF ±1%  
**이탈 시 → 즉시 중단.** 데이터 범위, config, 함수 버전 확인 먼저.

**최신 정식 베이스라인 (v14.4.0, 2020-2025):**

| 조합 | N | PF | Sharpe |
|------|---|----|--------|
| VR+MA, TBR OFF (프로덕션) | ~4,097 | ~1.521 | ~5.06 |
| VR+MA, TBR ON | ~3,890 | ~1.466 | ~4.65 |

> 위 수치는 `r47b_comparison.py` 기준. 새 리포트 나오면 이 표 업데이트.

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

> ⚠️ **실전 교훈 (2026-03-05):** `n_active=0`에서 단일 진입만 테스트
> → 2번째 슬롯 진입 시 마진 부족 `-2019` 에러.

**백테스트에서 슬롯 수 명시:**

```python
# 잘못된 예: production은 10 slots인데 15 slots로 시뮬레이션
pnl = portfolio_sim(trades, max_concurrent=15)  # ❌

# 올바른 예
pnl = portfolio_sim(trades, max_concurrent=10)  # ✅ Production = 10
```

**마진 검증 체크:**
- 보고서에 `max_concurrent=10` 명시
- n_active=0, 1, 2, ..., 9 순서로 연속 진입 시 마진 < balance × 95% 확인
- VR: `balance × (1/10) × 30 ≤ available_margin`
- MA: `balance × (1/10) × 21 ≤ available_margin`

---

## 6. 비용 모델

```python
COST_DIRECTIONAL = 0.0008  # 0.08% round-trip (entry 0.04% + exit 0.04%)
COST_STRADDLE    = 0.0016  # 0.16% round-trip (long 0.08% + short 0.08%)
# 펀딩비: 8h마다 약 0.01%, 48h 보유 시 약 0.06% 추가 (pnl에서 차감)
```

이 값은 `trade_sim.py`의 `COST_DIRECTIONAL`, `COST_STRADDLE` 상수에서 읽는다. 변경 시 여기만.

---

## 7. 데이터 범위 및 워밍업

```python
WARMUP_15m = 200  # 최초 200 바는 신호 무시 (ATR 계산 안정화)
WARMUP_1h  = 50
WARMUP_4h  = 13   # ceil(200 / 16)

# 권장 데이터 범위: 2020-01-01 이후 (5년 이상)
# 2020 이전 데이터는 Binance 유동성 부족으로 비현실적 결과 가능
```

---

## 8. 통계 유효성 기준

| 지표 | 최소 기준 | 권장 |
|------|---------|------|
| N (거래 수) | ≥ 500 | ≥ 2,000 |
| PF | > 1.05 | > 1.20 |
| Bootstrap P(PF≥1.0) | ≥ 90% | ≥ 95% |
| OOS PF (50% holdout) | > 1.03 | > 1.15 |
| OOS/IS 괴리 | < 10% | < 5% |
| 손실 연도 수 | 0 | 0 |
| 연속 손실 거래 | < 15 | < 10 |

**OOS 검증:**
```python
split = int(len(all_data) * 0.5)
is_trades  = collect_trades(cfg, all_data[:split])
oos_trades = collect_trades(cfg, all_data[split:])
# OOS PF가 IS PF의 85% 이상이면 과적합 아님
```

---

## 9. 연구 스크립트 템플릿

새 연구 스크립트 (`research/active/r{N}_r{M}/r{N}_description.py`) 기본 구조:

```python
"""r{N}: 연구 목적 한 줄 설명.

Phase 0: Pre-flight config 확인
Phase 1: 베이스라인 검증 (±1%)
Phase 2: 실험 파라미터 스윕
Phase 3: 결과 요약 및 리포트
"""
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.backtest.trade_sim import (
    collect_trades, portfolio_sim, compute_metrics, bootstrap_pf, precompute_all_vr,
)

# ── Phase 0: Config 읽기 ──────────────────────────────────────────────────
cfg = VolStraddleConfig()
print(f"[Phase 0] d_tp={cfg.d_tp}, d_trail_activate={cfg.d_trail_activate}, "
      f"VR_LN={cfg.base_leverage/cfg.max_concurrent:.1f}, "
      f"TBR={cfg.f5_tbr_enabled}")

# ── Phase 0: 데이터 로드 ─────────────────────────────────────────────────
# data_15m = { symbol: {"close": np.ndarray, "high": ..., "low": ..., "open": ...} }
# 로드 방법: precompute_all_vr() 또는 기존 스크립트 참조

# ── Phase 1: 베이스라인 (정적 파라미터, 변경 금지) ─────────────────────
baseline_cfg = {
    "d_tp": cfg.d_tp,
    "d_sl": cfg.d_sl,
    "d_trail_activate": cfg.d_trail_activate,
    "d_trail_distance": cfg.d_trail_distance,
    "f5_tbr_enabled": cfg.f5_tbr_enabled,
    # ... 나머지 config
}
b_trades = collect_trades(baseline_cfg, data_15m, data_1h, data_4h, strat="combined")
b_pnl    = portfolio_sim(b_trades, max_concurrent=cfg.max_concurrent)
b_m      = compute_metrics(b_pnl, DATA_YEARS)

print(f"[Phase 1 Baseline] N={b_m['N']}, PF={b_m['PF']:.3f}, Sharpe={b_m['Sharpe']:.2f}")
# 최신 정식 수치 (v14.4.0): N≈4097, PF≈1.521
assert abs(b_m['N'] - 4097) / 4097 < 0.01, f"Baseline N drift! {b_m['N']} vs 4097"
assert abs(b_m['PF'] - 1.521) / 1.521 < 0.01, f"Baseline PF drift! {b_m['PF']:.3f} vs 1.521"
print("[Phase 1] ✅ Baseline verified")

# ── Phase 2: 실험 ────────────────────────────────────────────────────────
results = []
for param_value in param_grid:
    exp_cfg = {**baseline_cfg, "target_param": param_value}
    trades  = collect_trades(exp_cfg, ...)
    pnl     = portfolio_sim(trades, max_concurrent=cfg.max_concurrent)
    m       = compute_metrics(pnl, DATA_YEARS)
    boot    = bootstrap_pf(pnl)
    results.append({"param": param_value, **m, "ci_lo": boot["lo"], "ci_hi": boot["hi"]})

# ── Phase 3: 리포트 ──────────────────────────────────────────────────────
# docs/backtest_results/YYYYMM/YYYYMMDD_{commit}_{description}.md 에 저장
```

---

## 10. 보고서 파일명 규칙

```
docs/backtest_results/YYYYMM/YYYYMMDD_{commit_short}_{description}.md

예: docs/backtest_results/202603/20260308_81dec2e_r47_exit_ab_test.md
```

**보고서 필수 섹션:**

```markdown
## Summary
- 데이터: 2020-01-01 ~ 2025-12-31 (5.5년)
- 슬롯: 10 (max_concurrent=10), 비용: 0.08% round-trip
- 베이스라인 N={}, PF={} (±1% 검증 통과)
- 수익률 모델: 1x compound (레버리지 미적용)

## Results
| 파라미터 | N | PF | Sharpe | EV(bp) | MaxDD | OOS50 PF | Bootstrap CI |
|----------|---|----|--------|--------|-------|----------|-------------|

## Conclusion
- 최적 파라미터: ...
- 베이스라인 대비 PF 변화: +X.X%
- 프로덕션 적용 권고: Yes/No / 추가 검증 필요

## Backtest-Live Parity Notes
- config.py 읽음: ✅
- 룩어헤드 없음: ✅ (j = (i//ratio) - 1)
- 비용 모델: COST_DIRECTIONAL=0.0008, COST_STRADDLE=0.0016
- 슬롯 수: 10 (production과 동일)
```

---

## 11. 과거 실패 사례 요약 (Quick Reference)

| 날짜 | 실패 | 증상 | 원인 | 규칙 |
|------|------|------|------|------|
| 2026-03-07 | 하드코딩 파라미터 | N/PF 2배 차이 | `tp=0.05` ≠ `cfg.d_tp=0.12` | 항상 `cfg = VolStraddleConfig()` |
| 2026-02-26 | 4h MTF 룩어헤드 | WR 83% vs 실전 36% | `i//16` (개발중 바) | `(i//16) - 1` (완성 바) |
| 2026-03-05 | 동적 자본 공식 오류 | 슬롯 2에서 마진 부족 | `balance/(n+1)` → 첫 슬롯 50% | 고정 `balance/max_concurrent` |
| 2026-03-06 | 알고 주문 상태 미검출 | SL/TP 체결 누락 | `TRIGGERED`만 확인, `FINISHED` 누락 | `TRIGGERED` + `FINISHED` 둘 다 |
| 2026-02-21 | Reduce-only 레버리지 | -2028 API 에러 | 청산 주문에 `change_leverage()` 호출 | 청산 시 레버리지 변경 금지 |
| 2026-03-08 | Hedge Mode trail abort | trail exit 실패 (-2028) | close가 `leverage=1` 기본값 전달 → 변경 시도 | `if not reduce_only and leverage > 1` (v14.4.2) |
| (반복) | 복리 모델 혼동 | CAGR 보고 오류 | `compute_metrics` 1x ≠ 프로덕션 L/N | 레버리지 모델 항상 명시 |

---

## 12. 연구 브랜치 컨벤션

```bash
git checkout -b research/r{N}-{description}   # 브랜치 생성
git merge --no-ff develop                     # 머지 (squash 아님, 이력 보존)
git push origin research/r{N}-{description}

# 리포트 저장 후 develop 머지 (연구 완료 시):
git checkout develop
git merge --no-ff research/r{N}-{description}
git push origin develop
```
