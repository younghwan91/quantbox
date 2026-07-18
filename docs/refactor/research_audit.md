# 리서치 실험 감사 — 재구현 드리프트

> 리팩토링 1단계의 부산물. "새 전략 테스트할 때마다 새로 짜서 같은 백테스트 버그가
> 재발한다"(사용자 통증 ②)를 **정량화**한 기록. 북극성: *틀리지 않은 백테스팅.*

## 왜 이 문서가 있나

harness 이주를 시작하며 활성 실험 하나(r119)를 공용 엔진(`sim_core`)으로 재현해
원본과 대조했다. 결과가 이주 전략 자체를 바꿨다:

> **개별 실험을 harness로 이주하는 것은 단순 리팩토링이 아니라 "결과 정정"이다.**
> 원본이 재구현한 sim이 canonical과 다르면, 이주하는 순간 그 실험의 발행 숫자가 바뀐다.

그래서 "152개 자동 이주"는 각각 숫자를 바꾸는 작업이라 일괄 처리할 수 없다. 각 실험은
(a) canonical과 일치하는지 먼저 감사하고, (b) 다르면 사용자 확인 후 정정해야 한다.

## 발견된 두 종류의 재구현 갭 (독립적)

### 1. 진입 재구현 갭 — MA Cluster (프로덕션, 21x)
`trade_sim`이 라이브 `check_ma_cluster`를 호출하지 않고 재구현 → 클러스터 에피소드당
1회여야 할 진입이 **2.32배** 발생(유령 트레이드 56.9%). 상세: `sim_core.run_ma_cluster_live`,
`tests/test_sim_core_live_parity.py`, 메모리 `ma-cluster-parity-break`.

### 2. exit 재구현 갭 — r119 (실험, 커밋 a205672)
r119를 harness로 재현해 원본과 비교(BTC MA50 게이트 ON, ATR OFF, trail=0.010):

| | 원본 r119 | harness (canonical exit) |
|---|---|---|
| 트레이드 | 1295 | 1296 |
| 진입 일치 | — | **100%** (원본에만 0) |
| 청산봉 다름 | — | **810 / 1295** |
| **PF** | **2.523** | **2.228** |

**진입은 완벽히 일치** — r119의 `in_cooldown` 재구현이 라이브와 같은 진입을 낸다.
**exit이 갈라진다**, 두 지점:

1. **trailing 활성화 타이밍.** r119는 봉에서 peak 갱신 → trail/TP/SL 체크 → *그 다음*
   활성화 판정. canonical `sim_drive`는 peak 갱신 → *먼저* 활성화 → 즉시 trail 체크.
   → r119는 trailing이 **1봉 늦다**.
2. **not-trailing SL/TP 순서.** r119는 TP 먼저, canonical은 SL 먼저.

효과: 발행 PF 2.523이 canonical(=라이브 기준)으로는 **2.228** (~12% 과대).

이 둘은 독립적이다: **r119는 진입은 맞췄지만 exit에서 틀렸다.** "진입만 맞추면 된다"가
아니라는 증거. 검증: `scratchpad/r119_migrate_check.py` (미커밋, 원본 r119 의존).

## 이주 전략 (이 감사가 함의하는 것)

| 대상 | 방침 | 근거 |
|---|---|---|
| **새 실험** | harness 강제 | 가드가 재구현 차단 → 갭이 구조적으로 불가능. 이미 시행 중. |
| **활성 실험 (자체 sim)** | 감사 → 선택적 정정 | 이주 = 숫자 변경. 개별 확인 필요. |
| **archive 실험** | 동결 | 죽은 실험. 이주 가치 없음. |

핵심: harness가 존재하는 이유는 **미래**를 막는 것이지 과거를 소급 정정하는 게 아니다.
과거 실험은 "그 숫자를 믿을 수 있나"를 감사하는 대상이지, 자동 이주 대상이 아니다.

## exit 드리프트 범위 (정적 스캔, 2026-07-18)

trailing exit을 가진 활성 실험 17개를 정규식 휴리스틱으로 스캔(활성화 판정 위치,
TP/SL 체크 순서). **13개가 canonical과 다른 exit 패턴 후보** — 대부분 r119와 같은
"trailing 1봉 지연 + TP-before-SL":

```
지연 TP먼저   r103_ma_cluster_parity, r115_quick_wins, r119_btc_atr, r120_per_tf,
             r43/r43b_exit_improvement, r55_subtf, r57_combo, r75_ma_cluster_sweep,
             r91_ma_cluster_parity, r92_ma_cluster_gate
지연         r43c_exit_improvement
TP먼저       r47_exit_ab_test
```

**신뢰도 한계 (중요):**
- 정규식 휴리스틱이다. **r119만 실측(재현) 확인됨.** 나머지 12개는 "후보"이지 확정 아님.
- ma_cluster 계열(r103/r115/r120/r75/r91/r92)이 전부 같은 패턴 → 이 계열은 **서로는
  일관**될 가능성이 높다. 그렇다면 계열 내 파라미터 비교(어느 게 나은가)는 유효하고,
  **절대 PF만** 프로덕션/라이브보다 높게 나온다(r119: 2.523 vs 2.228).
- **미해결 상위 질문: canonical(sim_drive)이 정말 라이브 Binance trailing과 같은가?**
  sim_drive는 프로덕션 `trade_sim`과 bit-parity이지만, 그 exit 시뮬레이션이 라이브
  브래킷 주문(activationPrice/callbackRate) 거동과 일치하는지는 별도 검증 대상이다.
  즉 "실험 vs canonical" 드리프트는 확인됐으나 "canonical vs 실제 체결"은 미확인.

## 상위 질문 조사 — canonical vs 라이브 trailing (2026-07-18, 코드 분석)

라이브 trailing 설정 코드를 읽어 canonical `sim_drive`와 대조했다.
(`brackets.py:_get_trail_params`, `positions.py`, Binance TRAILING_STOP_MARKET)

| | 청산가 (LONG) |
|---|---|
| 라이브 (Binance callbackRate) | `peak_price × (1 − trail_distance)` = peak_price − **peak_price**×td |
| canonical `sim_drive` | `entry × (1 + peak_gain − td)` = peak_price − **entry**×td |

**되돌림폭 base가 다르다: 라이브=peak_price, sim=entry.** peak_price > entry(활성화 시점)
이므로 라이브 청산가가 더 낮다 → **라이브가 더 늦게 청산, 백테스트가 보수적**.
크기 ≈ `trail_distance × peak_gain` (peak_gain 5% → ~2.5bp/트레이드, 20% → ~10bp).

**세 요소로 분리 (2026-07-18 정량화):**

1. **활성화 트리거 = 라이브와 일치.** activationPrice = entry×(1+trail_activate) ⟺
   sim의 peak_gain ≥ trail_activate (peak_price ≥ entry×(1+trail_activate)). canonical은
   이 봉에서 **즉시** 활성화 = 라이브와 같다. `config.py:76` "Matches IDEAL sim model".
2. **청산가 base 차이(entry vs peak) = 집계 영향 0.** sim_drive를 복사해 peak-base 변형을
   만들어 ma_cluster 3836 트레이드에 두 모델을 돌린 결과: 개별 트레이드 **2536/3836이
   달라지지만** 평균 PnL 53.63bp 동일(차이 0.000bp), **PF 1.904 / EV 53.6bp 완전 동일.**
   차이가 양방향 상쇄된다. → **백테스트 결론에 영향 없음. 실측 우선순위 낮음.**
   (`scratchpad/trail_base_quant.py`)
3. 결론: **canonical(sim_drive) trailing은 라이브 기준으로 신뢰 가능.** 활성화는 일치하고,
   base 차이는 집계에 안 나타난다. 아침의 "canonical이 라이브와 다를 수 있다" 우려는 해소.

## exit 드리프트 확정 — 활성 실험 10개, r119와 동일 코드 (2026-07-18)

후보 12개의 exit 블록을 canonical과 라인 비교. **10개가 r119와 정확히 같은
"활성화 1봉 지연 + TP-before-SL" 패턴** (코드 동일 → 동작 동일 → r119 실측이 대표):

```
지연+TP먼저 (10):  r103_ma_cluster_parity, r115_quick_wins, r120_per_tf,
                  r43/r43b_exit_improvement, r55_subtf, r57_combo,
                  r75_ma_cluster_sweep, r91_ma_cluster_parity, r92_ma_cluster_gate
지연만 (1):        r43c_exit_improvement
TP먼저만 (1):      r47_exit_ab_test
```

**핵심 구분:** r119류 드리프트의 주원인은 **활성화 1봉 지연**(canonical은 즉시=라이브).
이건 청산 시점 자체를 바꿔 집계에 큰 영향(r119: PF 2.523→2.228, ~12% 과대). base 차이
(요소 2)와 달리 상쇄되지 않는다. 즉 **canonical이 라이브에 맞고, 이 실험들이 틀렸다.**

영향받는 ma_cluster 계열(r103/r115/r120/r75/r91/r92)은 **서로는 일관**(같은 지연 패턴)
이므로 계열 내 파라미터 비교는 유효, **절대 PF만** ~12% 부풀려짐.

## 아직 안 한 것

- 정정 필요 실험 목록(위 10개) → 사용자 확인 → harness 재현으로 재발행.
- MA cluster 프로덕션 라이브 규칙 승격(진입 갭, 별건) → 사용자 결정.
- (선택) canonical trailing base DOGEUSDT 실측 — 집계 영향 0이라 우선순위 낮음.
