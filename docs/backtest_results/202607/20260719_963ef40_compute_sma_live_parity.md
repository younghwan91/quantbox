# compute_sma 재구현 → 라이브 정렬: research SMA parity 정정

- 날짜: 2026-07-19
- 베이스 커밋: 963ef40 (feature/shared-sim-core)
- 범위: research 8파일의 `compute_sma` 재구현 → `quantbox.research.harness import compute_sma`
- 라이브/프로덕션 백테스트 코어: **불변** (trade_sim / sim_core / signals / config 무변경)

## 배경 — 왜 봤나

reimpl 가드 baseline에서 남은 최대 중복 클러스터를 줄이는 과정. `compute_sma` 는 순수
수치 유틸이고 harness가 **프로덕션(라이브) 함수를 그대로 재수출**하므로, 이주는 원칙적으로
parity 교정(백테 SMA = 라이브 SMA)이다. 다만 "순수 유틸이니 당연히 같다"를 **가정하지
않고** 실데이터로 검증했다.

## 발견 — research SMA는 라이브와 실제로 어긋난다

research 8파일 중 6개는 SMA를 `np.mean(arr[i-p+1:i+1])`(윈도우마다 재합산, pairwise)로
계산한다. 라이브(`signals.compute_sma`, `@numba.jit`)는 **러닝섬**(`s += close[k] -
close[k-p]`)이다. 수학적으론 같지만 float64 누적 순서가 달라 결과가 갈린다.

실데이터 측정(83심볼 전체):

| 재구현 방식 | 파일 수 | 라이브 대비 max|diff| (118M 비교) |
|---|---|---|
| `np.mean` (pairwise) | 6 | **1.513e-09** |
| `np.sum` 초기화 + 러닝섬 (r111) | 1 | 2.910e-11 |
| 러닝섬 (r44c, 라이브와 알고리즘 동일) | 1 | 0 (비트동일) |

### 이 미세 오차가 실제 신호를 뒤집는가? — 그렇다

`sim_core.live_ma_cluster_signals`(라이브 `check_ma_cluster` 를 바별 재생)에 프로덕션 SMA
vs `np.mean` SMA를 넣어 진입신호를 대조:

- 83심볼 / 922,736개 4h바
- **진입신호 2건, 방향 2건 뒤집힘** (1심볼)
- 구체: **TRXUSDT** — 프로덕션은 bar 11654에서 진입, `np.mean`은 그 진입을 놓치고
  bar 11657에서 진입(**3바 지연**). cluster 스프레드가 `ma_cluster_pct` 경계로부터
  1.5e-9 이내에 놓인 경계 케이스.

즉 research의 `np.mean` SMA는 라이브가 내리는 진입 결정과 **실제로 어긋나는** 진짜
parity 갭이다(빈도는 극히 낮지만 0이 아님). `sim_core` 도크스트링이 이미 "monolith의
np.sum 초기화가 float64에서 라이브와 갈린다(~1e-15)"를 인지하고 있었고, 본 측정이 그
현상을 신호 수준까지 정량화했다.

## 조치

8파일 모두 `from quantbox.research.harness import compute_sma` 로 이주. 6개 `np.mean`
파일은 라이브와 정렬되어 TRX류 경계 어긋남이 교정된다. r44c/r111은 중복 제거(숫자 영향
무의미).

- 순수 def→import 치환 (+8/−56), numpy는 여전히 사용 → 고아 import 없음.
- reimpl 가드 baseline: 185→177 심볼 (compute_sma 8건 제거).
- 검증: 8파일 py_compile OK, parity/golden/guard 349 passed, 프로덕션 골든 불변.

## 발행 리포트 영향

없음. 과거 발행 리포트는 git 스냅샷이며 소급 변경되지 않는다. 본 이주는 **앞으로의**
research가 라이브와 같은 SMA를 쓰게 만드는 교정이다.

## 남은 것 (후속)

- `momentum8`(6), `compute_momentum`(4), `resample_ohlcv`/`build_resampled` 등 다른 순수
  유틸도 동일하게 실측 후 이주 가능.
- `load_data`(16)/`load_all_data`(9)는 동작 6종으로 갈린 데다 `_offset` 정렬 누락 이슈가
  얽혀 있어 별도 신중 작업(숫자 영향 큼, 게이트).
