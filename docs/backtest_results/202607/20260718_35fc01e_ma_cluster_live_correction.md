# MA Cluster 백테스트 정정 — 라이브 규칙 vs 재구현

Date: 20260718, Commit: 35fc01e
Data: 실 Binance 캐시 59심볼, 6년 (data/cache_extended, 2190d)

## 요약

`collect_ma_cluster_trades`(백테스트)가 라이브 `signals.check_ma_cluster`를 재구현하며
**클러스터 에피소드당 재진입**으로 트레이드를 2.32배 냈다. **라이브 실거래는 이 함수를
쓰지 않는다** — `strategy.py`가 `check_ma_cluster`(에피소드당 1회)를 호출한다. 따라서
과대였던 것은 백테스트/연구 도구뿐이고, **라이브 실거래는 처음부터 올바랐다.**

`sim_core.run_ma_cluster_live`(라이브 함수를 직접 호출)로 정정 성과를 측정했다.

## 성과 비교 (portfolio_sim, max_concurrent=10)

| 지표 | 재구현 (현행 collect_ma_cluster_trades) | 라이브 규칙 (run_ma_cluster_live) |
|------|------:|------:|
| N (슬롯 후) | 3,831 | **1,652** |
| PF | 1.90 | **2.00** |
| 승률 % | 67.6 | 68.0 |
| EV (bp) | 53.5 | **59.9** |
| t-stat | 14.49 | 10.02 |
| RR | 0.91 | 0.94 |
| MaxDD % | 3.6 | 2.9 |
| 연수익 % | 40.5 | **17.8** |
| Calmar | 11.1 | 6.2 |
| bootstrap PF median [CI] | 1.90 [1.75, 2.05] | 2.00 [1.76, 2.26] |
| P(PF≥1) % | 100 | 100 |

**읽는 법:** 전략은 멀쩡하고 오히려 **트레이드당 엣지는 더 좋다**(PF 1.9→2.0, EV
53.5→59.9bp) — 유령 트레이드(같은 에피소드의 2~17번째 재진입)가 첫 진입보다 나빠
엣지를 희석하고 있었다. 하지만 **빈도가 2.3배 부풀려져 연수익 기대가 40.5%→17.8%로
절반 이하**다. 즉 라이브는 백테스트 발행 수익의 절반 수준을 낼 것이다 — 전략이 나빠서가
아니라, 백테스트가 라이브가 잡지 않을 거래를 세고 있었기 때문.

검증: `run_ma_cluster_live`의 1,652 트레이드는 라이브 `check_ma_cluster`를 독립적으로
리플레이한 시그널 수(1,652)와 정확히 일치. `tests/test_sim_core_live_parity.py`.

## 근본 원인

라이브: 시그널 시 `in_cooldown=1`, 클러스터 해체(spread>cluster_pct)까지 재발화 안 함.
재구현: 트레이드 후 `cluster_count=0`만 — 해체를 요구하지 않아 같은 에피소드에서
`min_duration`봉 뒤 재진입(최대 17회 관측). `max_duration` 좀비 필터도 같은 이유로 무력화.
재구현 주석은 "One signal per cluster episode"라 주장하나 코드가 구현하지 않았다.

## 조치 (이 커밋)

- `collect_ma_cluster_trades` docstring에 재구현 경고(2.32x, 라이브 기준은
  `run_ma_cluster_live`). 함수 자체는 기존 백테스트 숫자 재현용으로 유지(골든 불변).
- `autoresearch/train.py`: `collect_ma_cluster_trades` → `run_ma_cluster_live` 전환.
  자동 연구가 ma_cluster를 2.32x 과대평가해 잘못된 포트폴리오를 학습하고 있었다.

## 프로덕션 함의

- **라이브 실거래 로직은 변경 없음** — 이미 올바르다. 실자본 위험 없음.
- **백테스트 기반 파라미터 결정은 재검토 권장.** MA cluster의 SL/trail/cluster_pct/
  min_duration 등은 유령 트레이드 56.9%가 섞인 데이터로 튜닝됐을 수 있다. 라이브 규칙
  하에서의 재스윕은 별도 작업(메모리 제약: SL 낮추는 방향 최적화 금지).
- exit 드리프트(별건): 활성 실험 10개가 exit도 재구현으로 canonical과 다름
  (r119 PF 2.523→2.228). 상세 `docs/refactor/research_audit.md`.
