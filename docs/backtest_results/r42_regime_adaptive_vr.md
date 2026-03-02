# R42: Regime-Adaptive VR Squeeze Parameters

**Commit**: bb7cb38  
**Date**: 2026-02-20  
**Verdict**: ❌ MARGINAL — Static params robust enough  

## Objective

BTC 변동성 레짐(low/mid/high)에 따라 VR squeeze entry/SL/TP를 조절하여
저변동성 기간(2023, 2024-2025)의 성과를 개선할 수 있는지 검증.

## Methodology

- **Regime classification**: BTC 4h ATR(56) rolling percentile (250-bar lookback)
  - Low-vol: percentile < 33%, Mid-vol: 33-67%, High-vol: > 67%
- **Data**: 50 symbols × 2,190d (6yr) × 15m = 210,335 bars
- **Sequential sweep**: Fix 2 regimes to baseline → sweep 1 regime → propagate best
  - Phase 2a: Sweep low-vol (80 variants) → Phase 2b: Sweep high-vol (64) → Phase 2c: Sweep mid-vol (80)
- **Costs**: Directional 0.08%, Straddle 0.16% (4-leg)

## Regime Distribution

| Year | Low% | Mid% | High% | Bars |
|------|------|------|-------|------|
| 2020 | 45.5 | 25.7 | 28.8 | 30,158 |
| 2021 | 43.3 | 33.3 | 23.4 | 35,056 |
| 2022 | 49.2 | 23.1 | 27.7 | 35,055 |
| 2023 | 35.4 | 30.0 | 34.6 | 35,056 |
| 2024 | 44.0 | 23.5 | 32.5 | 35,152 |
| 2025 | 44.2 | 28.3 | 27.5 | 35,055 |
| **Total** | **43.5** | **27.1** | **29.4** | **210,335** |

## Phase 1: Baseline (v13.0 Static Params)

| Metric | D_mom8 | S_d8 | Combined (10 slots) |
|--------|--------|------|---------------------|
| N | 1,569 | 2,039 | 2,248 |
| PF | 1.313 | 1.272 | 1.267 |
| EV | +25.0bp | +37.4bp | +30.8bp |
| WR% | 25.2 | 54.0 | 42.7 |
| AnnRet% | +78.7 | +200.4 | +171.6 |
| DD% | 31.7 | 57.9 | 36.3 |
| Calmar | 2.5 | 3.5 | **4.7** |

### Baseline Year-by-Year

| Year | N | PF | EV(bp) | WR% |
|------|---|------|--------|-----|
| 2020 | 895 | 1.36 | +39.9 | 40.3 |
| 2021 | 348 | 1.28 | +33.0 | 40.8 |
| 2022 | 591 | 1.24 | +28.3 | 40.8 |
| 2023 | 639 | 1.30 | +31.8 | 42.7 |
| 2024 | 540 | 1.20 | +22.8 | 42.2 |
| 2025 | 542 | 1.24 | +27.6 | 41.7 |

## Phase 2: Regime-Adaptive Sweep Results

### 2a: Low-Vol Sweep (80 variants)

**Best: LO_vr50_tp5_sl15 = BASELINE params**

| Variant | N | PF | EV | Calmar |
|---------|---|------|------|--------|
| **LO_vr50_tp5_sl15** | 3,552 | 1.270 | +31.0 | 7.6 |
| LO_vr50_tp5_sl10 | 3,608 | 1.285 | +32.0 | 7.5 |
| LO_vr60_tp5_sl15 | 13,483 | 1.105 | +13.5 | 7.5 |

**Finding**: 저변동성 기간에서 기존 파라미터가 이미 최적. VR > 0.55로 완화 시 PF 급락 (1.10~1.15).

### 2b: High-Vol Sweep (64 variants)

**Best: HI_vr40_tp5_sl15 — VR threshold 0.40 (baseline 0.50보다 타이트)**

| Variant | N | PF | EV | Calmar |
|---------|---|------|------|--------|
| **HI_vr40_tp5_sl15** | 2,592 | 1.341 | +38.2 | 9.2 |
| HI_vr40_tp7_sl15 | 2,592 | 1.335 | +37.8 | 9.0 |
| HI_vr35_tp7_sl15 | 2,542 | 1.338 | +38.0 | 8.8 |

**Finding**: 고변동성에서 VR 진입 기준을 0.40으로 강화하면 PF +0.07, EV +7bp 개선. 더 선택적인 진입이 도움.

### 2c: Mid-Vol Sweep (80 variants)

**Best: MID_vr50_tp5_sl10 — SL 1.0% (baseline 1.5%보다 타이트)**

| Variant | N | PF | EV | Calmar |
|---------|---|------|------|--------|
| **MID_vr50_tp5_sl10** | 2,592 | 1.341 | +38.2 | 9.2 |
| MID_vr50_tp7_sl10 | 2,548 | 1.313 | +38.1 | 8.9 |
| MID_vr50_tp5_sl8 | 2,660 | 1.343 | +36.7 | 8.7 |

**Finding**: 중간 변동성에서 SL을 1.0%로 좁히면 소폭 개선. 하지만 VR threshold는 baseline 0.50이 최적.

## Phase 3: Baseline vs Adaptive Comparison

### Best Adaptive Params Found

| Regime | d_vr | s_vr | d_sl | s_sl | TP | s_dur |
|--------|------|------|------|------|-----|-------|
| LOW | 0.50 | 0.55 | 1.5% | 2.2% | 5% | 4 |
| MID | 0.50 | 0.55 | 1.0% | 1.5% | 5% | 4 |
| HIGH | 0.40 | 0.45 | 1.5% | 2.2% | 5% | 4 |

- LOW = baseline 동일
- MID = VR 동일, SL만 타이트
- HIGH = VR만 타이트 (0.40 vs 0.50)

### Portfolio Comparison (10 slots)

| Mode | N | PF | EV(bp) | AnnRet% | DD% | **Calmar** |
|------|---|------|--------|---------|------|------------|
| BASELINE | 2,248 | 1.267 | +30.8 | +171.6 | 36.3 | **4.7** |
| ADAPTIVE | 1,628 | 1.311 | +35.7 | +134.0 | 42.6 | **3.1** |
| **Δ** | **-620** | **+0.044** | **+4.9** | **-37.6** | **+6.3** | **-1.6** |

### Year-by-Year (Unlimited Slots)

| Year | BL EV | ADP EV | Δ | |
|------|-------|--------|---|-|
| 2020 | +39.9 | +48.4 | +8.5 | ✓ |
| 2021 | +33.0 | +31.5 | -1.5 | — |
| 2022 | +28.3 | +22.5 | -5.8 | ✗ |
| 2023 | +31.8 | +38.3 | +6.5 | ✓ |
| **2024** | **+22.8** | **+38.6** | **+15.8** | **✓** |
| **2025** | **+27.6** | **+37.1** | **+9.5** | **✓** |
| 2026 | +86.1 | +61.5 | -24.6 | ✗ |

## Phase 4: Regime Threshold Sensitivity

| Thresholds | N | PF | EV | Calmar |
|-----------|---|------|------|--------|
| 0.25/0.75 | 2,767 | 1.329 | +36.8 | 9.0 |
| **0.30/0.70** | **2,655** | **1.338** | **+37.9** | **9.7** |
| 0.33/0.67 | 2,592 | 1.341 | +38.2 | 9.2 |
| 0.40/0.60 | 2,405 | 1.329 | +37.3 | 8.1 |

무관. 모든 threshold에서 비슷한 성능 → regime 분류 자체가 안정적.

## Conclusion

### 핵심 발견

1. **VR squeeze는 이미 regime-agnostic** — VR ratio 자체가 내장된 적응 메커니즘. 현재 threshold(0.50)이 저변동성에서도 이미 최적.

2. **고변동성에서 VR 0.40이 더 좋다** — 유일한 의미 있는 차이. 하지만 이를 적용하면 전체 거래 수 28% 감소 (2,248→1,628), 포트폴리오 분산 약화.

3. **Trade-off: 품질 vs 수량** — 트레이드당 EV는 +4.9bp 개선되지만, 거래 감소로 연간수익률 -37.6%p, Calmar -1.6.

4. **타겟 기간(2024-2025)은 개선** — 2024 EV +15.8bp, 2025 EV +9.5bp 로 원래 목표는 달성. 하지만 2022년 성과 악화(-5.8bp)로 전체 균형이 깨짐.

### Verdict: MARGINAL — 구현하지 않음

- **복잡성 비용 > 한계 이득**. 3개 regime × 6개 파라미터 = 관리할 변수 18개 추가.
- Static params가 6년간 모든 연도에서 PF > 1.20 유지 → 충분히 robust.
- VR ratio 자체가 이미 자연스러운 regime filter 역할 수행.

### 향후 고려

- 고변동성에서 VR threshold만 약간 강화(0.45~0.48) 해보는 것은 가능하나, 거래 감소 risk 존재.
- 레짐 기반 "포지션 사이징" 접근이 레짐 기반 "파라미터 스위칭"보다 더 유망할 수 있음 (향후 R43 후보).
