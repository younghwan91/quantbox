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

- [ ] `run_backtest`가 기존 `collect_trades(D_mom8)` / `collect_ma_cluster_trades` 결과와 **bit-parity** (기존 백테스트 숫자 안 변함) — 3개 전략 전부, 실데이터로 trade 리스트·PnL 완전 일치.
- [ ] **상태결합 회귀 테스트**: 트레이드 중 진입상태 동결 + 트레이드 후 `cluster_count=0` 리셋 + `i=eb+1` 점프가 드라이버에서 재현됨을 명시적으로 검증 (이걸 놓치면 청산 이후 신호가 어긋남 — parity 함정 §2).
- [ ] r120의 per-TF 질문을 fork 없이 harness로 재현.
- [ ] 신규 실험 파일이 sim/지표/로딩을 **0줄** 재구현.
- [ ] 가드가 재구현 시도를 실제로 잡음 (테스트로 증명).
- [ ] 라이브 런타임 코드·거동 **불변** (import 경로만 정리).

---
*작성 2026-07-13. 인터뷰 결정: mono-repo, 공용코어+harness 우선, 설계도 먼저. 근거·전체 맥락은 메모리 `repo-refactor-goal` 참조.*
