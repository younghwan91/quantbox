# 설계도: 공용 코어 + 실험 Harness

> 전체 레포 리팩토링의 **1단계** 청사진. 라이브 런타임은 이 단계에서 건드리지 않는다.
> 목표: "새 전략 실험 = 신호함수 + config만 채우면 백테스트·리포트가 나온다. 같은 백테스트 버그를 두 번 만들 수 없다."
>
> **북극성 (최우선 원칙):** *틀리지 않은 백테스팅.* 그 다음 실험 가속화, 있는 코드 재사용(재작성 금지).
> 모든 설계 결정은 이 순서를 따른다 — parity by construction > bit-parity 검증 > 재구현 0줄.

## 1. 문제 → 목표

| 지금 | 목표 |
|------|------|
| 실험마다 sim/데이터/지표를 새로 짬 → 같은 버그 재발 (②) | 코어는 **유일 구현** 하나. 실험은 조합만 함 |
| `collect_trades`가 `strat=="D_mom8"` 강제, MA/Donchian은 별도 함수 | 전략은 **플러그**로 등록. sim은 전략에 무지 |
| 코어가 4h 전용 → r120이 sim 통째로 fork | sim이 **TF 파라미터**를 받음. fork 불필요 |
| "재구현 금지"는 CLAUDE.md 문장일 뿐, 강제 안 됨 | pre-commit 가드가 재구현을 **차단** |

**핵심 원리:** research는 라이브의 *바로 그* 함수를 import한다 → parity(백테스트=실거래)가 구조적으로 보장됨. (레포를 나누지 않는 이유이기도 함.)

## 2. 코어 API 계약 (`quantbox/backtest/`)

**핵심 제약:** numba `nopython`은 파이썬 함수를 인자로 못 받는다 → "신호함수 주입"은 불가.

**⚠️ parity 함정 (검증으로 발견, trade_sim.py:1256-1258):** 진입 스캔은 **상태 결합**이다.
트레이드 후 `cluster_count = 0`(리셋) + `i = eb + 1`(청산 다음 봉으로 점프) → 트레이드 중 봉들은
스캔되지 않아 진입 상태가 **동결**된다. 그래서 "entry_signals 배열을 트레이드와 무관하게 통짜로
미리계산"하면 트레이드 중에도 카운트가 올라가고 리셋도 안 돼 → **청산 이후 진입 신호가 달라짐 →
기존 백테스트 숫자 변경**. 순진한 2단계 분리는 bit-parity를 깬다.

**올바른 분리 지점:** entry의 "무상태 조건"과 "크로스-봉 상태"를 나눈다.

```
[전략별 무상태 per-bar 조건]  →  [단 하나의 공용 드라이버 커널]  →  trades
  spread/breakout/momentum/gate    episode·cooldown 상태 + 진입게이트
  (순수 배열, 트레이드 무관)         + exit-sim + 리셋/점프(i=eb+1) 전부 소유
                                    (모든 전략 공용, 유일 구현)
```

검증된 사실: (1) 기존 3개 sim의 **exit 부분은 글자까지 동일**(peak_gain→trail_activate→SL-first→TP).
(2) 크로스-봉 **진입 상태(연속 카운트 → min/max_duration 게이트 → 트레이드 후 리셋 → eb+1 점프)도
3개 전략이 동일 패턴.** → 공용 드라이버로 일반화 가능. 전략별로 남는 건 "이 봉이 조건을 만족하나?"라는
무상태 배열뿐. bit-parity는 이 분리에서만 성립.

```python
# 전략 = (무상태 조건 커널 + exit 파라미터)로 기술되는 플러그
@dataclass(frozen=True)
class StrategySpec:
    name: str                       # "vr_directional" | "ma_cluster" | "donchian"
    condition_fn: Callable          # numba: (feature arrays, params) -> per-bar 무상태 배열들:
                                    #   cond_ok[bool], direction[int8], gate_ok[bool]
                                    #   (연속카운트/쿨다운/리셋 없음 — 그건 드라이버 소유)
    required_features: list[str]    # ["vr_8_50", "mom_8", ...] — precompute가 채움
    exit_params: ExitParams         # sl/tp/trail_activate/trail_distance (config에서만)

# 유일한 드라이버 커널 (전략·TF 무지, 유일 구현) — 진입 상태 + exit-sim 전부 소유
@numba.jit(nopython=True, cache=True)
def sim_drive(close, high, low, cond_ok, direction, gate_ok,
              min_duration, max_duration,          # 연속카운트 게이트 (episode 상태)
              sl, tp, trail_activate, trail_distance, max_hold, cost, warmup) -> trades:
    # while i<n: cluster_count 증가/리셋, 게이트 통과 시 진입,
    #            inner exit-loop(peak_gain→trail→SL-first→TP), 트레이드 후 리셋 + i=eb+1
    ...

# 유일한 백테스트 진입점 (전략·TF 무지)
def run_backtest(
    spec: StrategySpec,
    data: MarketData,               # 심볼별 OHLCV + features, 이미 로드됨
    *,
    tf: TF,                         # 15m/30m/1h/4h — 파라미터. fork 금지
    cfg: VolStraddleConfig,         # 유일한 파라미터 출처
) -> TradeSet:                      # (symbol, entry_15m, exit_15m, pnl) + 메타
    # 1) spec.condition_fn 으로 무상태 배열 생성  2) sim_drive 로 체결  3) 15m bar로 환산
    ...
```

- `tf`가 인자가 되면 r120 같은 per-TF 질문에 fork가 필요 없다. (altitude 근본 해결)
- 기존 `sim_*_trailing` 3개는 이 구조 위에서 **얇은 어댑터**로 재현 → 기존 백테스트 숫자 불변(§6).

## 3. Harness API (`quantbox/research/harness/`)

harness는 **진짜 파이썬 패키지**로 둔다(`quantbox/research/`) — 기존 `sys.path.insert` 해킹 제거,
`import`만으로 재사용 강제. 실험 스크립트는 `research/active/`에 그대로 두되 아래처럼 **얇아진다.**
새 실험 = 아래 한 파일. sim/로딩/지표/리포트는 **손대지 않는다.**

```python
from quantbox.research.harness import experiment, MarketData, report

@experiment(name="r121_my_idea")
def run(cfg):
    data = MarketData.load(cfg.symbol_whitelist)     # load_data 33벌 대체
    spec = StrategySpec(name="ma_cluster", condition_fn=my_conditions, ...)
    trades = run_backtest(spec, data, tf="1h", cfg=cfg)
    return report.standard(trades, cfg)              # PF/WR/DD/bootstrap/연도별 자동
```

harness가 제공(= 실험이 다시 짜면 안 되는 것): `get_commit`, `load_data`, `resample`,
`compute_sma`, `momentum`, `portfolio_sim`, `compute_metrics`, `bootstrap_pf`, 리포트 렌더링, 표준 판정 기준.

## 4. 강제 (재발 방지)

기존 pre-commit `research-param-guard`를 확장한다:

- research 파일이 `def portfolio_sim` / `def compute_metrics` / `@numba ... def sim_` 를 **정의**하면 실패.
- research가 `compute_sma`/`load_data`/`get_commit`를 **로컬 정의**하면 실패 → harness import 요구.
- 즉 "재구현 금지"가 문장이 아니라 **CI 게이트**가 된다.
- **강도: 기본 차단 + 명시적 탈출구.** 정당한 프로토타입은 주석 한 줄 `# harness-exempt: <이유>`
  로 예외 허용 → 우회 해킹 방지하면서 '의도적 예외'를 기록으로 남긴다. (구현 시 조정 가능)

## 5. 범위 경계 (이 단계에서 하는 것 / 안 하는 것)

**한다:** 코어 제네릭화(2), harness(3), 가드(4), 신규 실험 1개를 harness로 작성해 검증.
**안 한다 (다음 단계):**
- 라이브 스파게티 = 전략 플러그화 (Mega v2 Phase 3) — 실거래 로직이라 별도.
- 기존 research 152파일 마이그레이션 — 동결해두고, 필요할 때 코어 위로 이식.

## 6. 검증 기준 (이 단계 "완료"의 정의)

- [x] `run_backtest`가 기존 `collect_trades(D_mom8)` / `collect_ma_cluster_trades` 결과와 **bit-parity** (기존 백테스트 숫자 안 변함) — 3개 전략 전부, 실데이터로 trade 리스트·PnL 완전 일치. → `sim_core.py` + `tests/test_sim_core_parity.py` (26 tests, 실데이터 254 트레이드)
- [x] **상태결합 회귀 테스트**: 트레이드 중 진입상태 동결 + 트레이드 후 `cluster_count=0` 리셋 + `i=eb+1` 점프가 드라이버에서 재현됨을 명시적으로 검증 (이걸 놓치면 청산 이후 신호가 어긋남 — parity 함정 §2). → `test_state_coupling_*` (naive 미리계산이 실제로 어긋남을 증명하는 테스트 포함)
- [x] r120의 per-TF 질문을 fork 없이 harness로 재현. → `research/active/r121_harness/` (`ratio` 인자만으로 4개 TF)
- [x] 신규 실험 파일이 sim/지표/로딩을 **0줄** 재구현. → r121 = 45줄, 가드 통과
- [x] 가드가 재구현 시도를 실제로 잡음 (테스트로 증명). → `scripts/check_research_reimpl.py` + `tests/test_research_reimpl_guard.py` (26 tests)
- [x] 라이브 런타임 코드·거동 **불변** (import 경로만 정리). → `trade_sim.py` 바이트 불변, 신규 코드는 전부 추가(additive)

### 실행 규약 (구현 중 확정)

실험은 **레포 루트에서 `-m`으로** 실행한다:

```bash
uv run python -m research.active.r121_harness.r121_ma_cluster_per_tf_harness
```

`python <경로>`로 실행하면 `sys.path[0]`이 스크립트 폴더라 `import quantbox`가 실패한다 —
research 파일 149곳에 `sys.path.insert`가 박혀 있던 이유가 바로 이것이다. `-m`은
`sys.path[0]`을 cwd(레포 루트)로 잡으므로 해킹 없이 import된다.

> 더 근본적인 해법은 `[build-system]`을 추가해 프로젝트를 editable 설치하는 것이다
> (그러면 cwd와 무관하게 import된다). 다만 이는 라이브 배포의 `uv sync` 동작을 바꾸므로
> 이 단계에서는 하지 않았다. 별도 판단 필요.

### 구현 중 발견 (설계 수정 사항)

1. `sim_drive`에 `cap_divisor` 필요 — 모놀리식 sim들의 사전할당 상한이 다르다
   (ma_cluster `n//4+1`, directional/donchian `n//2+1`). 절단 동작까지 맞추려면 파라미터여야 한다.
2. `sim_drive`에 `cooldown_bars` 필요 — donchian은 **감소하는 post-signal 쿨다운**을 쓴다.
   ma_cluster의 증가 카운터와 다른 상태 패턴이라, "3개 전략이 같은 진입상태 패턴"이라는
   §2의 주장은 2/3만 맞았다. 드라이버가 두 패턴을 모두 지원하도록 확장했다.
3. **지표 관행이 갈라져 있던 게 sim보다 더 심했다.** `compute_metrics`를 호출하는 research
   92개 파일 중 `portfolio_size`를 넘기는 건 5개뿐 — 나머지 87개는 기본값 1(매 트레이드
   전액 복리)로 수익률이 폭발한다. 실측: MA cluster 1h가 `ps=1`에서 연 204만%/총 7.13e27%,
   `ps=10`에서 연 198%. `data_years` 산출식도 6종으로 갈라져 연수익을 서로 다르게 스케일했다.
   → `report.standard`가 두 기본값을 올바른 쪽으로 고정한다.
4. `load_data` 16벌의 동작이 **6종으로 갈라져** 있었고, 그중 4벌은 `_offset`(심볼 간
   타임라인 정렬)을 누락했다 → 포트폴리오 sim이 심볼 간 봉이 어긋난 채 돌아간다.
   → `MarketData`가 정렬을 생성 시점 불변식으로 강제한다.

### 유예 장부 (래칫)

가드 도입 시점의 기존 위반은 `scripts/research_reimpl_baseline.json`에 기록해 유예했다
(**80개 파일, 237개 심볼** — 이 부채의 실측 규모다). 기존 파일을 손대도 커밋은 막히지 않지만,
**새 재구현은 차단**된다. 이 목록은 늘리지 않고 줄여 나간다.

---
*작성 2026-07-13. 인터뷰 결정: mono-repo, 공용코어+harness 우선, 설계도 먼저. 근거·전체 맥락은 메모리 `repo-refactor-goal` 참조.*
