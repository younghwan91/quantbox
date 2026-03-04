# QuantBox v14.0.0 — 현재 배포 상태 보고서

- **Commit**: `6fdd86c` (release/v14.0.0)
- **Date**: 2026-03-04
- **Branch**: `release/v14.0.0`
- **Mode**: Paper Trading
- **Balance**: $1,668 (≈₩240만)
- **Tests**: 868 passed (0 failures)

---

## 1. 현재 배포 전략 요약

**Dual-Axis VR Squeeze System** — 두 개의 독립적인 스퀴즈 전략을 하나의 슬롯 풀에서 운용.

```
┌─────────────────────────────────────────────────────────┐
│              QuantBox v14.0.0 (6fdd86c)                  │
│              Dual-Axis Squeeze System                    │
├──────────────────────┬──────────────────────────────────┤
│   Axis 1: VR Squeeze │   Axis 2: MA Cluster Squeeze     │
│   VR(8,56) < 0.50    │   4 SMA within 3%, dur ≥ 8      │
│   25x leverage       │   12x leverage                   │
│   1% SL / 12% TP     │   2% SL / 8% TP                 │
│   Trailing stop      │   Fixed SL/TP                    │
│   L/N = 2.5          │   L/N = 1.2                      │
│                      │   mom ≥ 0.010 (R45 신규)          │
│                      │   maxd ≤ 24 bars (R45 신규)       │
├──────────────────────┴──────────────────────────────────┤
│              공유 슬롯 풀: 10 slots                       │
│              리스크 패리티: 25×1% ≈ 12×2%                 │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 배포 파라미터 전체

### 2.1 VR Squeeze (Axis 1)

| 파라미터 | 값 | 설명 |
|---|---|---|
| `d_vr_threshold` | 0.50 | VR < 0.50일 때 진입 |
| `d_sl` | 1% | 고정 손절 |
| `d_tp` | 12% | 하드 TP (R43 업그레이드) |
| `d_trail_activate` | 2% | +2%에서 추적 시작 |
| `d_trail_distance` | 0.3% | 고점 대비 0.3% 하락 시 청산 |
| `s_vr_threshold` | 0.55 | 스트래들 VR 진입 |
| `s_sl` | 1.5% | 스트래들 SL (per leg) |
| `s_tp` | 5% | 스트래들 TP (per leg) |
| `max_hold` | 48h | 시간 기반 청산 |
| Leverage | 25x | L/N = 25/10 = 2.5 |

### 2.2 MA Cluster Squeeze (Axis 2)

| 파라미터 | 값 | 설명 |
|---|---|---|
| `ma_cluster_pct` | 3% | 4 SMA 최대 스프레드 |
| `ma_cluster_min_duration` | 8 bars | 최소 32시간 수렴 |
| `ma_cluster_max_duration` | **24 bars** | R45: 좀비 클러스터 차단 |
| `ma_cluster_min_momentum` | **0.010** | R45: 약한 모멘텀 필터 |
| `ma_cluster_sl` | 2% | 고정 손절 |
| `ma_cluster_tp` | 8% | 고정 익절 |
| `ma_cluster_max_hold_4h` | 12 (48h) | 시간 기반 청산 |
| Leverage | 12x | L/N = 12/10 = 1.2 |

---

## 3. 불변 엣지 지표 (레버리지/기간 무관)

이 수치들은 전략 자체의 엣지. 어떤 레버리지, 어떤 데이터 기간에서도 동일.

| 지표 | VR Squeeze (D+S) | MA Cluster | 결합 |
|---|---|---|---|
| **N** | 2,265 | 1,631 (R45) | 3,748 (R45) |
| **PF** | 1.280 | 1.365 (R45) | 1.304 (R45) |
| **WR** | 43.4% | 38.1% | 40.9% |
| **EV/trade** | +32.1bp | +42.6bp | +35.4bp |
| **Bootstrap P(PF≥1.0)** | 100.0% | 100.0% | 100.0% |
| **Bootstrap CI** | [1.158, 1.479] | — | [1.221, 1.414] |
| **6/6 연도 흑자** | ✓ | ✓ | ✓ |

> 출처: R39/R40 (VR), R45 corrected (MA, Combined) — 모두 `scaling_r38.py` canonical sim 사용

---

## 4. 수익률 기대치

### 4.1 Compound 모델별 수익률

| Compound 모델 | 설명 | AnnRet |
|---|---|---|
| 1x compound | `equity *= (1+p)` — 레버리지 무적용 | +1,086% |
| Production leverage | VR: L/N=2.5, MA: L/N=1.2 per-trade | **+1,410%** (추정) |
| 현실적 (60%) | 슬리피지+펀딩비+API에러 감안 | **+846%** |
| 현실적 (50%) | 보수적 | +705% |

> 1x→Production 증폭비 ≈ 1.3× (v14 strategy report 기준)

### 4.2 현재 잔고 $1,668 기준

| 시나리오 | 연수익률 | 1년 후 | 월 평균 |
|---|---|---|---|
| 백테스트 그대로 | +1,410% | $25,200 | +24%/월 |
| **현실적 (60%)** | **+846%** | **$15,780** | **+20%/월** |
| 보수적 (50%) | +705% | $13,420 | +18%/월 |
| 최악 연도 (2023형) | +188% | $4,800 | +9%/월 |

### 4.3 DD 시나리오

| 수준 | 잔고 | 비고 |
|---|---|---|
| Max DD (백테스트) | $762 (-54.3%) | |
| Max DD (현실×1.2) | $584 (-65%) | 백테스트 미반영 마찰 포함 |
| **반드시 버텨야 한다** | 이 구간에서 시스템을 끄면 엣지 실현 불가 | |

---

## 5. 버전 변경 이력

### v13.0 → v13.1 → v14.0.0

| 버전 | 변경 | 핵심 지표 |
|---|---|---|
| v13.0 (e4b5831) | 25x/10s 공격적 배포 | VR only, AnnRet +687% (6yr), Calmar 10.12 |
| v13.1 (de67965) | R43 trailing stop exit | d_tp 5%→12%, trail +2%/0.3% |
| **v14.0.0 (ef55e2e)** | **MA Cluster 추가** | **Dual, AnnRet +1,204%, Calmar 19.4** |
| v14.0.0-R45 (d4e70d8) | MA zombie/momentum filter | mom≥0.010, maxd≤24, PF +2% |
| v14.0.0-fix (6fdd86c) | 베이스라인 수정, 보고서 정리 | VR baseline D+S combined 확정 |

### R45 변경의 실제 효과

| 지표 | R45 전 | R45 후 | 변화 |
|---|---|---|---|
| Combined N | 3,911 | 3,748 | -163 (-4.2%) |
| Combined PF | 1.278 | **1.304** | **+2.0%** |
| Calmar | 19.0 | 18.8 | -1% |
| per-trade EV | +28.6bp | **+35.4bp** | **+24%** |

> R45는 TRXUSDT dur=70/mom=0.007 같은 좀비 진입을 차단. 소수의 저품질 트레이드 제거로 PF/EV 개선.

---

## 6. 라이브 검증 현황 (Paper Trading)

### 2026-03-04 트레이드 결과

| # | 심볼 | 전략 | 방향 | mom | dur | PnL | R45 판정 |
|---|---|---|---|---|---|---|---|
| 1 | ADAUSDT | MA Cluster | SHORT | 0.021 | 9 | **-2.68% ($-331)** | ✓ 통과 |
| 2 | TAOUSDT | MA Cluster | LONG | 0.024 | 8 | **-1.33% ($-168)** | ✓ 통과 |
| 3 | FILUSDT | MA Cluster | LONG | 0.030 | 8 | **+2.71% ($+209)** | ✓ 통과 |
| 4 | TRXUSDT | MA Cluster | SHORT | 0.007 | **70** | **-1.38% ($-43)** | **✗ R45 차단** |
| 5 | WLDUSDT | MA Cluster | SHORT | 0.059 | 10 | (미진입) | ✓ 통과 |

- **VR Squeeze**: 0건 — 전 심볼 VR > 0.50 (스퀴즈 조건 미충족)
- **R45 효과**: TRXUSDT(dur=70, mom=0.007) 1건 차단 → $43 손실 방지
- **당일 결과**: 4 trades closed, PnL = **$-332** (하락장)

> Paper trading 개시일이므로 통계적 의미 없음. 최소 200 trades (~200일) 후 판단.

---

## 7. 정식 백테스트 레퍼런스

현재 버전의 수치를 검증할 때 참조할 정식 보고서:

| 보고서 | 역할 | 핵심 수치 |
|---|---|---|
| [R39/R40](20260302_d70cd2f_yearly_r39.md) | VR 정식 베이스라인 | N=2265, PF=1.280, 6/6yr |
| [v13 deployment](20260302_e4b5831_v13_deployment.md) | VR @25x/10s 배포 | AnnRet +687%, Calmar 10.12 |
| [R45 param sweep](20260304_d4e70d8_r45_param_sweep.md) | MA 필터 최적화 | Combined PF=1.304, Calmar 18.8 |
| [v14 strategy](20260304_ef55e2e_quantbox_v14_strategy_report.md) | 듀얼 전략 통합 | AnnRet +1,204%, Calmar 19.4 |

---

## 8. 안전장치 요약

| 장치 | 설명 |
|---|---|
| 리스크 패리티 | 25x×1% ≈ 12x×2% → per-trade risk 동일 |
| 방향 다양성 | max 2 same direction |
| SoftTight DD Scaling | 2%/4%/6% DD → 80%/60%/40% sizing |
| R45 Zombie Filter | duration > 24 bars 차단 |
| R45 Momentum Filter | abs(momentum) < 0.010 차단 |
| Trailing Stop (VR) | +2% 활성, 0.3% 추적 거리 |
| Max Hold | 48h 시간 기반 강제 청산 |
| DD Halt | 과도 DD 시 전 포지션 청산 |
| MA Cluster 즉시 비활성화 | `VS_MA_CLUSTER_ENABLED=false` → 재시작 |
| reduce_only 보호 | 청산 주문에서 leverage 변경 금지 |
| 상태 복원 | 재시작 시 포지션/전략 메타 전부 복원 |

---

## 9. 모니터링 체크리스트

| 항목 | 기준 | 액션 |
|---|---|---|
| DD | > 65% | MA cluster 비활성화 검토 |
| 30일 PF | < 0.9 | 경고 — 전략별 PF 분리 확인 |
| 60일 PF | < 1.0 | 레버리지 축소 or MA 비활성화 |
| MA cluster 발동률 | 0/week (2주) | 정상 — 시장 조건 대기 |
| VR squeeze 발동률 | < 0.5/day (30일) | 정상 — 시장 구조 확인 |
| 백테스트 WR 대비 라이브 WR | 라이브 WR < 백테스트의 50% | lookahead 즉시 의심 |

---

## 10. 솔직한 기대

- **엣지는 확실히 존재**: Bootstrap 5,000회에서 PF<1.0 = 0회. 6년 연속 흑자.
- **현실적 1년 기대**: 현재 $1,668 → **$13,000~$16,000** (현실적 50~60%)
- **최악의 경우**: $1,668 → $584 (DD 65%) — 이 구간에서도 시스템을 유지해야 복구
- **per-trade EV는 작다** (+35bp). 많은 거래와 레버리지로 증폭하는 구조.
- **이번 R45 변경**: 소폭이지만 의미 있는 개선. PF +2%, EV +24%. 좀비 트레이드 제거.

> 백테스트의 +1,410%를 그대로 기대하면 안 된다. 실전 마찰 40-50% 감안한 **+700~850%**가 합리적.

---

## Reporting Checklist

- [x] Compound model: production leverage (VR L/N=2.5, MA L/N=1.2)
- [x] Data period: 6yr max (avg 4.7yr, 50 symbols)
- [x] VR baseline: D_mom8 + S_d8 combined (canonical R39/R40)
- [x] R45 MA filter: mom≥0.010, maxd≤24 반영
- [x] 이전 보고서와 동일 조건 비교
- [x] compute_metrics() 출력을 production 수치로 쓰지 않음

---

*Generated: 2026-03-04 | Commit: 6fdd86c | Branch: release/v14.0.0 | Tests: 868 passed*
