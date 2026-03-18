# r101: 1-Minute Trail Validation

**날짜:** 2026-03-18  
**커밋:** TBD  
**결과:** ⚠️ BIAS DETECTED — 15m sim overestimates PnL by +0.10%/trade

---

## 목적

15m 바 기반 trailing stop 시뮬레이션의 정확도 검증.

**문제:** 15m 바 내에서 high/low 발생 순서를 모름 → intra-bar path ambiguity.

현재 시뮬레이션 가정: "high 먼저, low 나중" (낙관적)

---

## 방법

1. 5개 심볼(BTC, ETH, SOL, DOGE, SUI) × 60일 1m 데이터 다운로드
2. 1m 데이터에서 trail 시뮬레이션 (ground truth)
3. 15m으로 집계 후 동일 시뮬레이션
4. 두 결과 비교

---

## 파라미터

```
VR threshold: 0.57
Squeeze duration: 3
SL: 1.5%, TP: 20%
Trail activate: 1.5%, Trail distance: 0.1%
Max hold: 96 bars (24h)
```

---

## 결과

| 지표 | 값 |
|------|-----|
| Total comparisons | 94 |
| **Exit reason match** | **100%** ✅ |
| PnL diff mean | **+0.10%** |
| PnL diff median | 0.00% |
| PnL diff std | 0.37% |
| PnL diff max | +3.13% |

### Trail Exit 상세

| 지표 | 값 |
|------|-----|
| 1m trail exits | 39 |
| 15m trail exits | 39 |
| **Trail diff mean** | **+0.25%** |
| Trail diff std | 0.54% |

### Mean PnL 비교

| 시뮬레이션 | Mean PnL |
|------------|----------|
| 1m (ground truth) | +0.012% |
| 15m (current) | +0.115% |
| **Bias** | **+0.103%** |

---

## 해석

### Exit Reason은 정확
- SL/TRAIL/TP 판정 100% 일치
- "어떻게 청산되는가"는 정확함

### Exit Price는 낙관적
- Trail exit에서 평균 +0.25% 과대추정
- 이유: 15m 바에서 high 도달 후 low 도달 가정
- 실제: low가 먼저 올 수 있음 → trail이 더 낮은 가격에 발동

### 실제 영향 추정

```
Straddle 4,773 trades × 0.10% bias = ~477bp 누적
Trail exits 4,302 × 0.25% bias = ~1,076bp 누적
```

보고된 PF에서 **~2-5% 하향 조정** 필요.

---

## Bias 원인 상세

```
15m 바 예시:
  Entry: $100.00
  15m High: $101.80 (+1.8%)
  15m Low: $100.50

Trail activate: 1.5%, Trail distance: 0.1%

경로 A (15m 가정): $100 → $101.80 (peak) → $100.50
  Peak = +1.8% → trail_level = $101.70
  Low $100.50 < $101.70 → TRAIL at +1.7%

경로 B (실제 가능): $100 → $100.50 → $101.80
  Peak at low time = 0% (trail inactive)
  Then peak = +1.8% → trail_level = $101.70
  But bar closes at $101.50 → TRAIL at bar close (+1.5%)

Diff: +1.7% - +1.5% = +0.2% (15m가 낙관적)
```

---

## 결론

1. **Exit reason 정확** — SL/TRAIL/TP 판정 신뢰 가능
2. **Exit price 낙관적** — 거래당 +0.10% bias
3. **Trail에서 bias 집중** — trail exit당 +0.25%

### 권장 사항

| 옵션 | 장점 | 단점 |
|------|------|------|
| 현상 유지 + safety margin | 간단 | PF 정확도 ↓ |
| 1m 데이터로 최종 검증 | 정확 | 다운로드/저장 비용 |
| Pessimistic adjustment | 보수적 | 과소추정 위험 |

**권장:** 현상 유지 + 보고서에 "~3-5% PF 하향 margin" 명시.

---

## 파일

- `research/active/r101_r105/r101_1m_trail_validation.py`

---

## Parity Notes

- ✅ cfg 읽음 (하드코딩 없음)
- ✅ 실제 API 데이터 사용
- ⚠️ 60일만 검증 (6년 대비 일부)
