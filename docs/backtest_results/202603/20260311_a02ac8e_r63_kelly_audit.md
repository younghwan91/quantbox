# r63: Half-Kelly Leverage Audit

**날짜:** 2026-03-11  
**커밋:** a02ac8e (v14.16.0)  
**데이터:** 60 syms, 6yr, Combo2+G_4TF params  
**비용:** 0.08% round-trip  

---

## 결론 요약

| 전략 | Config | Kelly 계산 | 괴리 | 판정 |
|------|--------|-----------|------|------|
| MA Cluster | 21x | 20.1x | 4.1% | ✅ OK |
| VR (D+S 통합) | 30x | 이론상 훨씬 높음 | — | ✅ 보수적 안전 |

**MA는 현재 WR/RR과 완벽히 일치. VR 30x는 보수적으로 안전한 수준.**

---

## 전략별 실제 WR/RR (v14.16.0 기준)

| 전략 | N | WR | RR | avg_win | avg_loss | EV |
|------|---|----|----|---------|----------|----|
| D_mom8 (방향성) | 3,368 | 62.9% | 1.16 | 1.25% | 1.08% | — |
| S_d8 (스트래들) | 8,683 | 48.9% | 4.84 | 1.02% | 0.21% | — |
| MA Cluster | 2,464 | 36.7% | 2.21 | 4.23% | 1.91% | 34.4bp |

---

## Half-Kelly 재계산 상세

### 공식
```
f* = WR - (1-WR)/RR
L  = (f*/2) × N / SL       (N=10슬롯)
```

### MA Cluster (이상 없음)
- WR=36.7%, RR=2.21 → f*=8.06%, f*/2=4.03%
- Kelly L = 4.03% × 10 / 2% = **20.1x** (config: 21x) → 괴리 4.1% ✅

### VR 30x — 왜 "보수적 안전"인가

원본 Kelly 계산 (20260304, D_mom8 단독):
- WR=25.9%, RR=3.75 → f*=6.1% → L=**30.7x** → 30x 채택 ✅

현재 VR 통합 (D_mom8 + S_d8):
- D_mom8: WR=62.9%, RR=1.16 → Kelly L=154.6x
- S_d8: WR=48.9%, RR=4.84 → Kelly L=191.7x
- **→ Kelly 이론상 30x는 실제 최적의 1/5 수준**

#### WR 변화의 원인
1. **D_mom8 WR 상승** (25.9% → 62.9%): Pullback entry 필터(r55) + Combo2(s_vr=0.57) 도입으로 고품질 진입만 선택
2. **RR 하락** (3.75 → 1.16): `trail_activate=0.5%, trail_distance=0.2%` 도입 → avg_win이 ~1.25%로 제한 (TP=12%는 드물게 도달)
3. **S_d8 스트래들 추가**: 스트래들은 long+short 동시 진입으로 Kelly 이항 공식 직접 적용 불가

#### 실제 per-trade 자본 리스크 (30x 기준)
```
VR 슬롯당 리스크 = SL(1%) × leverage(30x) × per_slot(balance/10)
               = 0.01 × 30 × 0.10 × balance
               = 3.0% of total balance ← 안정적
```

---

## CW-Tier 멀티플라이어 정합성

```
cw_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5]  (score 0~4)
```

- 선형 비례 ✅ (score당 0.5x씩 증가)
- 기준 score=1에서 1.0x, 최대 2.5x (score=4)
- Kelly 비례 원칙: 더 높은 conviction → 더 높은 베팅 ✅
- 최대 레버리지: 30x × 2.5 = 75x (극히 드문 최고 conviction 신호)

---

## 코드 검증 (positions.py)

```python
# VolStraddle 레버리지 적용 코드 (확인 완료)
leverage = vs.base_leverage  # 30x (VR 기본)
if "leverage_override" in sig_meta:
    leverage = sig_meta["leverage_override"]  # 21x (MA Cluster)
if vs.cw_enabled:
    notional *= cw_mult  # CW-Tier 멀티플라이어
```

**모든 분기가 코드에서 정확히 구현됨.**  
레버리지는 설계 시점에 Half-Kelly로 계산 후 config에 고정(static). 런타임에 동적 재계산하지 않음.

---

## 최종 판정

| 항목 | 결과 |
|------|------|
| VR 30x 적용 여부 | ✅ 코드에서 정확히 적용됨 |
| MA 21x 적용 여부 | ✅ sig_meta leverage_override로 정확히 적용됨 |
| Kelly 공식 일치 (MA) | ✅ 4.1% 오차 (허용 범위) |
| Kelly 공식 일치 (VR) | ✅ 30x는 현재 WR/RR 기준 보수적 안전 구간 |
| CW-Tier 비례 원칙 | ✅ 선형 비례 |
| per-trade 리스크 | VR: 잔고의 3% / MA: 잔고의 4.2% |

**→ 현행 레버리지 설정 유지. 다음 대규모 파라미터 변경 시 재검토.**

---

*구현: 검증만 (config 변경 없음)*
