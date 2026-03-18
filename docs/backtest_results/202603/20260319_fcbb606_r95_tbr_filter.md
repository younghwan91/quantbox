# r95: Taker Buy Ratio (TBR) Filter — S_d8 Straddle + MA Cluster

**날짜:** 2026-03-19  
**커밋:** fcbb606  
**스크립트:** `research/active/r91_r95/r95_tbr_filter.py`  
**결론:** ❌ **양쪽 모두 FAIL — TBR 필터 구현 불필요**

---

## 배경 및 범위

**VR Squeeze (D_mom8): 제외** — r71 (2026-03-13) 이미 테스트 완료  
- TBR ON vs OFF: PF delta = -0.004, 2/3 OOS 연도 역효과  
- trail_activate=1.0% 환경에서 나쁜 진입을 trail이 조기 cut → TBR 불필요  

**테스트 대상:**
1. **S_d8 Straddle** — 방향 중립, 진입 시점 TBR extremeness 버킷 분석
2. **MA Cluster** — 방향성, TBR 방향 정렬 필터 스윕 (BTC gate 항상 ON)

**데이터:**
- `data/cache_extended/*_15m_2190d_*.npz` — `taker_buy` 컬럼 포함
- ⚠️ **31/60 whitelist 심볼만 로드** (cache_extended 커버리지 부분)
- 기간: 2020-01-01 ~ 2026-02-18 (6.0년)

**TBR 분포 (31심볼 샘플):**
- mean=0.490, std=0.088
- P10=0.383, P25=0.437, P50=0.491, P75=0.543, P90=0.596

---

## Section A: S_d8 Straddle TBR 버킷 분석

### 설계
진입 바의 `tbr_extremeness = |taker_buy/volume - 0.5|`를 4개 버킷으로 분리  
경쟁 가설:
- **H_extreme**: 강한 TBR 편향 → 실제 스퀴즈 → 스트래들 유리
- **H_neutral**: TBR ≈ 0.5 (균형) → 양방향 기회 → 스트래들 유리

**파라미터:** `s_sl=0.015, s_vr=0.57, s_dur=3, trail_activate=0.015, trail_distance=0.001`

### 결과

| Bucket | 범위 | N | PF | WR | EV(bp) | MaxDD |
|--------|------|---|----|----|--------|-------|
| Neutral | \|tbr-0.5\| < 0.03 | 1,737 | 2.729 | 45.3% | +15.0 | 5.3% |
| Mild | 0.03–0.06 | 1,677 | 2.878 | 45.2% | +16.3 | 3.3% |
| Moderate | 0.06–0.10 | 1,678 | 2.754 | 44.3% | +15.2 | 4.6% |
| Extreme | > 0.10 | 1,812 | 2.488 | 43.3% | +13.4 | 4.5% |
| **ALL (baseline)** | — | **2,782** | **3.061** | **44.8%** | **+17.7** | **4.3%** |

### 해석

**모든 버킷이 baseline(ALL)보다 낮다.**
- H_extreme 기각: extreme TBR(PF=2.488)이 baseline(PF=3.061) 대비 -19% 하락
- H_neutral 기각: neutral TBR(PF=2.729)도 baseline 대비 -11% 하락
- TBR extremeness는 스트래들 결과와 **역상관**: 더 극단적일수록 EV가 낮아지는 경향

**왜 모든 버킷이 baseline 미만인가?**  
버킷 시뮬레이션은 필터에 걸리는 신호를 skip하지만 `i += 1`로만 진행 → 그 자리에 다른 거래 가능.  
반면 baseline은 발동된 거래 직후 `i = eb + 1`로 점프 (자연스러운 cooldown).  
즉, 버킷 필터는 cooldown을 없애 overtrading 발생 → 개별 버킷 N이 증가하지만 품질 저하.  
**결론: 스트래들 TBR 버킷 분석은 "좋은 거래만 남긴다" 효과 없음.**

### 판정: ❌ FAIL

---

## Section B: MA Cluster TBR 방향 정렬 필터 스윕

### 설계
- **LONG**: `mean(tbr[-window:]) > threshold` 조건 충족 시만 진입
- **SHORT**: `mean(tbr[-window:]) < (1 - threshold)` 조건 충족 시만 진입
- BTC 4h gate: **항상 ON** (r92 대비 동일 조건)
- Grid: window × [1, 4, 16 bars] × threshold × [0.00, 0.50, 0.52, 0.55, 0.57, 0.60]

**파라미터:** `sl=0.02, trail_activate=0.02, trail_distance=0.003, max_hold=12×4h`

⚠️ **데이터 주의:** 31/60 심볼 → baseline N=428 vs r92 전체(649). 결과는 부분 우주 기준.

### 결과

| Config | N | PF | WR | EV(bp) | MaxDD | 상태 |
|--------|---|----|----|--------|-------|------|
| w=1 t=0.00 | 428 | 1.913 | 47.7% | +88.1 | 25.7% | ← baseline |
| w=1 t=0.50 | 264 | 1.747 | 47.7% | +71.6 | 21.9% | ✗ |
| w=1 t=0.52 | 147 | 1.745 | 46.3% | +71.8 | 18.0% | ✗ |
| w=1 t=0.55 | 34 | 1.406 | 41.2% | +41.5 | 11.6% | ✗ |
| w=4 t=0.50 | 275 | 1.913 | 49.5% | +83.8 | 29.3% | ~marginal |
| w=4 t=0.52 | 95 | 2.084 | 49.5% | +98.6 | 18.2% | ~marginal |
| w=4 t=0.55 | 13 | 1.147 | 30.8% | +20.3 | 8.1% | ✗ |
| w=16 t=0.00 | 428 | 1.913 | 47.7% | +88.1 | 25.7% | ← baseline |
| w=16 t=0.50 | 254 | 1.893 | 50.8% | +79.6 | 18.7% | ~marginal |
| w=16 t=0.52 | 68 | 1.138 | 41.2% | +16.1 | 18.0% | ✗ |
| w=16 t=0.55 | 4 | 7.122 | 75.0% | +318.4 | 2.1% | N=4 (무시) |

**Pass 기준:** PF ≥ baseline(1.913) AND EV ≥ baseline(88.1bp) AND N ≥ 266 (62%)

→ **해당 조건 충족 config 없음.**

### 해석

1. **TBR 필터를 강화할수록 N이 급감** (thresh=0.55에서 N=13~34 — 통계 의미 없음)
2. `w=4 t=0.52`가 PF=2.084(+9%), EV=+98.6bp(+12%)로 약간의 개선을 보이나 N=95 (baseline의 22%) — 너무 적음
3. MA cluster에서도 **TBR은 레이어드 필터로서 효과 없음**
4. 이미 BTC 4h gate(regime 필터)가 있어 나쁜 진입 상당수를 제거 → TBR의 추가 기여 미미

### w=16, t=0.55 (N=4, PF=7.122) 는 무시

4개 거래에서 나온 결과 — 통계적으로 의미 없음. 실제 배포 불가.

### 판정: ❌ FAIL

---

## 최종 판정 매트릭스

| 전략 | 테스트 | 결론 | 액션 |
|------|--------|------|------|
| VR Squeeze | r71 (제외) | FAIL | 변경 없음 |
| S_d8 Straddle | r95a — TBR 버킷 | ❌ FAIL | 미구현 |
| MA Cluster | r95b — TBR 방향 필터 | ❌ FAIL | 미구현 |

**3개 전략 모두 TBR 필터 불필요. 코드 변경 없음.**

---

## 실패 이유 분석

### VR Squeeze (r71): 이미 알려진 이유
- trail_activate=1.0% → 나쁜 진입이 빠르게 cut → TBR이 막으려는 거래를 trail이 이미 처리

### S_d8 Straddle: 방향 중립 전략의 한계
- 스트래들은 정의상 방향을 모름 → TBR이 "어느 레그가 이길지" 예측 불가
- 진입 시점 TBR extremeness는 post-squeeze 방향성과 무관
- **핵심 인사이트**: 스트래들의 알파는 TBR이 아닌 **VR 스퀴즈 강도**에서 나옴

### MA Cluster: BTC gate 중복 효과
- BTC 4h MA50 gate가 이미 나쁜 regime (BTC 하락 중)의 LONG을 차단
- TBR의 추가 신호 정보가 BTC gate 이후 남은 거래에서 미미함
- trail_activate=2.0% → trail이 1%보다 2배 늦게 발동하지만, BTC gate 이미 regime 필터링 완료

---

## 데이터 한계 (중요)

**31/60 심볼**: cache_extended에 `taker_buy` 데이터가 있는 심볼이 전체의 52%.  
MA cluster 결과(N=428, PF=1.913)는 r92 전체(N=649, PF=2.749) 대비 낮음.  
**하지만 결론에는 영향 없음**: 어떤 TBR 설정도 각 설정 내 자체 baseline을 못 넘음.

추후 cache_extended를 60심볼로 확장 시 재실행 가능하나 기대 결과 동일.

---

## 구현 결정

**코드 변경 없음.** `ma_cluster_tbr_enabled`, `strd_tbr_filter_enabled` config 추가 불필요.

현재 r95 스크립트만 커밋 (연구 기록).

**구현:** 미구현 — r95 FAIL
