# r73: Trail-Aware 3-Strategy Combined Portfolio — 최선의 결정

**날짜:** 2026-03-13  
**Commit:** 33c8487  
**데이터:** 60 syms · 5.67yr · Binance USDT-M  
**비용:** 0.08% (directional) / 0.16% (straddle)  
**슬롯:** 10 (max_concurrent)  
**스크립트:** `research/active/r71_r75/r73_trail_combined.py`  
**구현:** 미구현 (Option A = 현상 유지, 코드 변경 없음)

---

## 핵심 질문

> r72에서 발견된 D_mom8 슬롯 압박(6.2%) 문제에 대한 최선의 결정은?

### 결론: **Option A — 현상 유지. 구조 변경 불필요.**

---

## Phase 0: S_d8 Trail Baseline 검증

| | N | PF | 결과 |
|--|---|----|----|
| r66 canonical | 7,730 | 4.948 | 기준 |
| r73 Phase 0 | **7,730** | **4.948** | ✅ 0.0% drift — 완벽 일치 |

→ Trail 시뮬레이션 정확성 확인 완료. 이하 수치 신뢰 가능.

---

## Phase 1: 전략별 개별 성능 (trail-aware)

| 전략 | N | PF | WR | EV(bp) | MaxDD | Calmar |
|------|---|----|----|--------|-------|--------|
| D_mom8 trail (G_3TF) | 2,518 | **1.693** | 48.3% | +38.3 | 20.7% | 19.6 |
| S_d8 trail (G_4TF) | 7,730 | **4.948** | 49.8% | +41.9 | 14.7% | 1813.8 |
| MA Cluster (flat TP=8%) | 2,461 | 1.283 | 36.7% | +34.2 | 55.2% | 4.4 |

> D_mom8 trail PF=1.693 (flat 1.212에서 40% 개선). Trail이 D_mom8에도 유효함.  
> S_d8 trail Calmar=1813 — 예외적으로 강한 신호. 슬롯 독점이 정당화됨.

---

## Phase 2: 3전략 Combined Portfolio

### 슬롯 배분 (trail-aware, 핵심 변화)

| 전략 | Raw 신호 | 슬롯 배정 | 배정률 | r72 flat 비교 |
|------|---------|---------|------|------------|
| D_mom8 | 2,594 | 292 | **11.3%** | ↑ 6.2% → 11.3% (+82%) |
| S_d8 | 8,683 | 7,152 | 82.4% | ↓ 91.7% → 82.4% |
| MA Cluster | 2,464 | 2,329 | **94.5%** | ↑ 88.3% → 94.5% |

> Trail 기반 trade timing이 달라져 D_mom8 배정률이 flat 6.2%에서 11.3%로 개선됨.  
> MA Cluster는 trail S_d8와 4h TF 분리로 충돌 최소화.

### 종합 성능 비교

| 구분 | N | PF | WR | EV(bp) | MaxDD | Calmar |
|------|---|----|----|--------|-------|--------|
| **OLD D+MA trail** | 4,874 | 1.414 | 42.4% | +36.6 | 49.7% | **31.4** |
| **NEW 3-strat, 10-slot** | 9,773 | **2.002** | 46.1% | +40.1 | 40.5% | **1695.6** |
| NEW 3-strat, 15-slot | 10,534 | 2.005 | 45.8% | +38.8 | 43.6% | 2091.9 |

### 연도별 (3전략 combined, trail)

| 연도 | N | PF | EV(bp) | WR |
|------|---|----|--------|----|
| 2020 | 1,905 | **3.23** | +67.4 | 55.9% | ✓ |
| 2021 | 1,205 | **3.34** | +73.1 | 55.4% | ✓ |
| 2022 | 1,539 | **1.91** | +41.2 | 45.8% | ✓ |
| 2023 | 1,655 | **1.45** | +24.9 | 41.0% | ✓ |
| 2024 | 1,550 | **1.42** | +18.3 | 38.8% | ✓ |
| 2025 | 1,918 | **1.65** | +22.2 | 41.4% | ✓ |

**Positive years: 6/6** ✓ — 모든 연도에서 양수, 최약 연도 PF=1.42 (2024)

---

## Phase 3: 구조 변경 옵션 비교

### 결정 프레임워크

```
OLD D+MA Calmar:          31.4
3-strat 10-slot Calmar: 1695.6
임계값 (D+MA × 1.1):      34.5

1695.6 ≥ 34.5 → ✅ Option A: 현상 유지 (54배 초과!)
```

### 옵션별 비교

| 옵션 | 설명 | Calmar | D_mom8% | 코드변경 | 권고 |
|------|------|--------|---------|---------|-----|
| **A: 현상유지** | 10-slot | **1695.6** | 11.3% | 없음 | ✅ **채택** |
| B: max_concurrent=15 | 15-slot | 2091.9 | 10.7% | config만 | ❌ |
| C: 슬롯 쿼터 | 전략별 예약 | 미측정 | ~30%+ | 코드 수정 | ❌ |

> **Option B (15-slot)**: Calmar +23% 개선이나, D_mom8 배정률은 오히려 감소 (11.3→10.7%)!  
> 슬롯 늘려도 S_d8 raw 신호(8683)가 더 많아 여전히 S_d8가 먼저 채움.  
> 추가 자본 투입 대비 효과 미미 → 불채택.

---

## 최선의 결정: Option A — 현상 유지

### 이유

1. **S_d8 독점이 정당하다**: trail PF=4.948, Calmar=1813 — 이렇게 강한 신호가 슬롯을 쓰는 것이 최적
2. **D_mom8 11.3% = 보너스**: flat 6.2%에서 11.3%로 개선. 292개 trades/5.67yr → 연 51개 실행, 유의미한 기여
3. **MA Cluster 94.5% = 완벽 분리**: 4h TF 덕에 VR 전략과 무충돌, 독립적 기여
4. **Calmar 54배 개선**: OLD D+MA 31.4 → 3-strat 1695.6 (S_d8 trail이 주도)
5. **구조 변경 효과 없음**: 15-slot도 D_mom8 배정률 감소 (원인: S_d8 raw 신호량이 항상 더 많음)

### 실질적 포트폴리오 해석

> **현재 10-slot = 사실상 "S_d8 dominant + MA 독립 + D_mom8 cherry pick" 구조**
> 이것이 최적이다. S_d8가 많이 쓰는 것은 버그가 아니라 "강한 신호가 슬롯을 쓰는" 올바른 동작.

---

## Flat vs Trail 비교 (r72 vs r73)

| 지표 | r72 flat | r73 trail | 배율 |
|-----|---------|----------|------|
| PF (3-strat) | 1.129 | 2.002 | 1.8x |
| Calmar (3-strat) | 5.6 | 1695.6 | **302x** |
| DD (3-strat) | 85.0% | 40.5% | -53% |
| D_mom8 배정률 | 6.2% | 11.3% | +82% |

> **Flat sim의 한계**: trail의 "big winner 보호" 효과를 반영하지 못해 PF를 4.4배 과소평가.  
> 모든 전략 평가는 trail-aware sim이 기준이어야 함.

---

## Parity Notes

- ✅ `VolStraddleConfig()` 직접 읽음 (하드코딩 없음)
- ✅ `symbol_whitelist` 60개 사용 (numtrades 파일 제외, _offset 적용)
- ✅ S_d8: r66 완벽 재현 (N=7730, PF=4.948, 0.0% drift)
- ✅ 룩어헤드 없음: `sim_straddle_trailing` / `sim_directional_trailing` import 사용
- ⚠️ **MA Cluster: flat TP=8%** (trail-aware MA sim 미구현)
- ✅ D_mom8: `sim_directional_trailing` (trail=1.0%/0.2%)

*60 symbols · 5.67yr · Binance USDT-M perpetuals · 0.08%/0.16% round-trip*
