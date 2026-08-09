# G11 (부분체결 / 더스트) — 모델링 불필요 + 보정 불가

- **일자**: 2026-08-10
- **브랜치**: `feature/mega-v2-phase5`
- **범위**: Mega Refactor v2 — Phase 5.3 (G11)

## 계획서 명세

> G11: Add partial fill / dust position modeling (configurable fill rate)

두 갈래로 나눠 각각 판정했다.

## 1. 더스트 / `min_notional` — **구속력 없음 (수치로 확정)**

라이브는 명목가가 거래소 `min_notional`(≈$5)보다 작으면 진입이 거부되고, 부분체결 잔여가
그 아래로 남으면 더스트가 되어 슬롯을 막는다(`positions.py:252-255`).

현재 사이징에서 최소 진입 명목가:

```
per_slot = balance / max_concurrent
leverage = clamp(risk_per_trade × max_concurrent / SL, 1, max_leverage)
notional = per_slot × leverage × min(CW 배수)
```

프로덕션 선언값(`risk_per_trade=0.06`, `max_concurrent=10`, `max_leverage=30`, CW 하한 0.5):

| 잔고 | 최소 명목가 | 거래소 하한 대비 |
|---|---|---|
| $1000 (라이브 실측) | $1,416 | **283배** |
| $200 (스트레스) | $300 | **60배** |

두 자릿수 배율로 떨어져 있다. **백테스트가 더스트를 무시해도 결과가 달라지지 않는다.**

여유가 사라지는 순간을 잡는 회귀 가드를 넣었다 (`tests/test_dust_threshold.py`).
잔고 급감이나 `risk_per_trade` / `max_concurrent` / CW 하한 변경으로 배율이 10배 아래로
내려가면 테스트가 먼저 깨진다. 그때는 백테스트에도 반영해야 한다.

> 주의: `conftest.py:68`이 테스트 환경에 `VS_RISK_PER_TRADE=0.01`을 주입하므로
> `VolStraddleConfig()` 인스턴스는 라이브 사이징이 아니다. 가드는 선언 기본값을 직접 읽는다.

## 2. 부분체결 (`fill_rate`) — **명세대로면 무의미, 의미 있는 버전은 보정 불가**

### 균일 체결률은 모델이 아니라 산수다

트레이드 PnL이 자기자본 대비 비율이므로 모든 진입에 체결률 *f*를 곱하면 결과는
**전 트레이드 PnL의 선형 리스케일**이다:

- PF, 승률, 트레이드 수 — **전부 불변**
- 연수익만 *f*배

파라미터를 만들 필요가 없다. 손으로 곱하면 되는 값에 노브를 다는 것은 없는 정밀도를
있는 것처럼 보이게 할 뿐이다.

### 의미 있는 부분은 선택적 체결이고, 그건 데이터가 없다

실제로 성과를 바꾸는 건 **역선택**이다 — 빠르게 움직이는 시장에서 체결이 더 나쁘고,
그 구간이 곧 큰 수익/손실 구간이다. 이건 균일 배수로 표현되지 않는다.

보정하려면 저널에 요청수량 / 체결수량 / 체이스 횟수 / 기준가가 있어야 하는데
**하나도 없다**:

```
컬럼: id, symbol, strategy_type, direction, entry_price, exit_price, pnl_pct,
      pnl_usd, mfe_pct, mae_pct, notional, bars_held, entry_time, exit_time,
      exit_reason, balance_before, balance_after, session, commit_sha,
      straddle_pair_id, created_at
```

2026-08-02 인계 노트 7절의 "집행 계측" 항목과 같은 결론이다.

## 결론

| 갈래 | 판정 | 조치 |
|---|---|---|
| 더스트 / min_notional | 구속력 없음 (60~283배 여유) | 회귀 가드만 추가 |
| 균일 체결률 | 선형 리스케일 — 모델 가치 없음 | 구현하지 않음 |
| 선택적(역선택) 체결 | 실질적이나 **보정 불가** | 저널 계측 선행 필요 |

**G11은 "미구현"이 아니라 "구현할 것이 없음 + 계측 선행"으로 종료한다.**
G2와 동일한 패턴이다 — 명세가 데이터 가용성을 확인하지 않고 작성됐다.
