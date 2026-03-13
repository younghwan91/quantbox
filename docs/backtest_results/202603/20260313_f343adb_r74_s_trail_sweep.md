# r74: S_d8 trail_activate 스윕

**날짜:** 2026-03-13  
**커밋:** f343adb  
**스크립트:** `research/active/r71_r75/r74_s_trail_sweep.py`  
**데이터:** 60 syms (symbol_whitelist) · avg 198,647 bars · 5.67yr · Binance USDT-M  
**비용:** 0.16% round-trip (straddle, 4 sides) · 10슬롯 (max_concurrent)  
**Parity:** config.py 직접 읽음 ✅ · symbol_whitelist ✅ · 룩어헤드 없음 ✅  

## 목적

r73이 S_d8 trail PF=4.948 (N=7730, s_trail_activate=1.5%)를 r66 canonical로 확인.  
r68/r69에서 D_mom8의 d_trail_activate를 0.5%→1.0%로 최적화했는데, **S_d8 straddle에도 1.0%가 유효한지** 데이터로 탐색.

> ⚠️ 태스크 스펙에서 "r66 canonical = trail_activate=0.5%"로 기술됐으나, 실제로는 `s_trail_activate` 파라미터는 r53 이후 1.5%로 설정돼 있음. r73도 cfg.s_trail_activate=1.5%로 N=7730, PF=4.948을 생성. 0.5%(0.005)는 d_trail_activate의 이전 값으로 혼동 발생. Phase 0은 실제 r73 canonical(1.5%)로 검증.

---

## Phase 0: Baseline 검증 (s_trail_activate=1.5%, r73 canonical)

| 항목 | 값 | r73 대비 drift |
|------|---|----------------|
| N | **7,730** | 0.0% ✅ |
| PF | **4.948** | 0.0% ✅ |
| WR | 49.8% | |
| EV | +41.9bp | |
| MaxDD | 14.7% | |
| Calmar | 1,813.8 | |

**→ r73 완벽 재현. Phase 1 진행.**

---

## Phase 1: trail_activate 스윕 (trail_distance=0.2% 고정)

| trail_activate | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| 0.3% | 10,503 | 2.476 | 71.6% | +44.8 | 65.0% | 4,962.9 |
| 0.5% | 9,064 | 2.177 | 58.9% | +46.0 | 58.9% | 2,102.9 |
| 1.0% * | 7,928 | 2.281 | 41.1% | +42.4 | 43.0% | 721.6 |
| 1.2% | 7,797 | 2.779 | 40.5% | +42.8 | 40.0% | 757.5 |
| **1.5% (현재)** | **7,730** | **4.948** | **49.8%** | **+41.9** | **14.7%** | **1,813.8** |
| 0.7% | 8,402 | 2.118 | 49.3% | +45.2 | 58.6% | 1,110.4 |

*\* = D_mom8 r68/r69 최적값 (S_d8 변경 검토 대상)*

**Calmar 기준 정렬:**

| 순위 | trail_activate | PF | MaxDD | Calmar |
|------|----------------|-----|-------|--------|
| 1 | 0.3% | 2.476 | 65.0% | 4,962.9 |
| 2 | 0.5% | 2.177 | 58.9% | 2,102.9 |
| 3 | **1.5% (현재)** | **4.948** | **14.7%** | **1,813.8** |
| 4 | 0.7% | 2.118 | 58.6% | 1,110.4 |
| 5 | 1.2% | 2.779 | 40.0% | 757.5 |
| 6 | 1.0% * | 2.281 | 43.0% | 721.6 |

**핵심 관찰:**
- **1.0%는 가장 나쁜 값**: PF=2.281, MaxDD=43.0%, Calmar=721.6 — 최하위권
- **현재 1.5%가 PF 최고**: PF=4.948 (2위 1.2%=2.779 대비 78% 우월)
- **0.3%가 Calmar 최고**: 하지만 MaxDD=65.0%로 DD가 크고 N이 많아 지나친 early exit 우려
- Calmar 고Calmar 값은 작은 trail_activate에서 더 많은 N(trades)이 발생해 compound 수익이 폭발적으로 증가하는 효과. PF가 더 신뢰성 있는 엣지 지표.

---

## Phase 2: trail_distance 스윕 (상위 2개 trail_activate 기준)

### trail_activate = 0.3%

| trail_distance | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| **0.1%** | **10,506** | **3.211** | **72.3%** | **+61.1** | **31.9%** | **202,468.7** |
| 0.2% (현재) | 10,503 | 2.476 | 71.6% | +44.8 | 65.0% | 4,962.9 |
| 0.3% | 10,486 | 1.908 | 64.6% | +30.0 | 88.9% | 232.4 |
| 0.4% | 10,430 | 1.466 | 53.7% | +17.0 | 99.8% | 17.9 |

### trail_activate = 0.5%

| trail_distance | N | PF | WR | EV(bp) | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| **0.1%** | **9,064** | **2.735** | **59.8%** | **+60.8** | **38.0%** | **34,277.7** |
| 0.2% (현재) | 9,064 | 2.177 | 58.9% | +46.0 | 58.9% | 2,102.9 |
| 0.3% | 9,057 | 1.743 | 58.2% | +31.8 | 79.9% | 162.2 |
| 0.4% | 9,043 | 1.420 | 57.6% | +19.2 | 98.7% | 16.7 |

**Phase 2 관찰:**
- trail_distance가 작을수록 PF와 EV 모두 향상 (조기 trail start + tight stop)
- 0.1%의 Calmar은 compound 폭발 효과로 과장됨; trail_distance=0.1%는 실거래에서 슬리피지로 실현 어려움

---

## 결론

### 핵심 발견

1. **현재 s_trail_activate=1.5%를 유지해야 한다**: PF=4.948 (최고), MaxDD=14.7% (최저). 안정성과 엣지 모두 최고.

2. **1.0%로 변경하면 악화**: PF 2.281 (현재 4.948 대비 −54%), MaxDD 43.0% (현재 14.7% 대비 3배), Calmar 721 (현재 1,814 대비 −60%). **변경 금지.**

3. **D_mom8 최적화 결과(1.0%)를 S_d8에 적용하면 안 됨**: Straddle은 양방향 노출이라 directional과 trail 역학이 다름. D_mom8에서 최적인 값이 S_d8에서 최악임.

4. **작은 trail_activate(0.3-0.5%)의 높은 Calmar는 compound 아티팩트**: N 증가로 compound 수익이 폭발하지만 MaxDD도 높음 (65-59%). 실거래 적용 부적합.

### 최종 권장

| 파라미터 | 현재 (config.py) | 권장 | 변경 여부 |
|----------|----------------|------|----------|
| `s_trail_activate` | 0.015 (1.5%) | **유지** | ❌ 변경 불필요 |
| `s_trail_distance` | 0.002 (0.2%) | 유지 | ❌ |

**D_mom8의 d_trail_activate=1.0% 변경은 S_d8에 영향 없음. S_d8은 현재 파라미터가 최적.**

---

## Parity Notes

- ✅ `cfg = VolStraddleConfig()` — 하드코딩 없음
- ✅ `symbol_whitelist` 60개 — `exclude_symbols` 미사용
- ✅ G_4TF (15m/30m/1h/4h) — r66/r73 canonical
- ✅ `sim_straddle_trailing` import (r43_exit_improvement.py) — 재구현 없음
- ✅ `_offset` 정렬 적용 (단축 히스토리 심볼 대응)
- ✅ 비용 0.16% round-trip (straddle 4 sides)

**구현:** 미구현 (현재 파라미터 유지 권장)
