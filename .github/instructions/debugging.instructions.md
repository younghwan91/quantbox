---
applyTo: "quantbox/**/*.py,tests/**/*.py"
---
# Debugging Skill — QuantBox API Error & Live Issue Guide

> **목적:** AI 에이전트가 라이브 트레이딩 중 발생하는 에러를 즉시 진단하고 대응할 수 있도록.
> 모든 에러 코드와 대응책은 실제 발생한 인시던트에서 추출됨.

---

## 1. Binance API 에러 코드 대응표

### -1022: Signature for this request is not valid

**증상:** 주문 API 호출 시 서명 불일치
```
APIError(code=-1022): Signature for this request is not valid.
params={'symbol': 'SUIUSDT:LONG', ...}  ← ':' 포함이 단서
```

**원인 A: `pos['symbol']`에 compound key 저장** ← Incident #4 (2026-03-08)
- `_load_state`가 `'SUIUSDT:LONG'`을 `symbol=`로 그대로 전달
- `pos['symbol'] = 'SUIUSDT:LONG'` → API 호출에 `:` 포함 → 서명 불일치

**진단:**
```python
import json
state = json.load(open('data/state/scalper_state.json'))
for sym, pos in state['positions'].items():
    print(sym, '→ symbol:', pos.get('symbol'))
# 'SUIUSDT:LONG' → symbol: 'SUIUSDT:LONG'  ← BUG (should be 'SUIUSDT')
```

**수정:**
```python
# state 파일 즉시 패치
for key, pos in state['positions'].items():
    sym_field = pos.get('symbol', '')
    if ':' in sym_field:
        parts = sym_field.rsplit(':', 1)
        if parts[1] in ('LONG', 'SHORT'):
            pos['symbol'] = parts[0]
            if pos.get('position_side') is None:
                pos['position_side'] = parts[1]
json.dump(state, open('data/state/scalper_state.json', 'w'), indent=2)
# 세션 재시작
```

**원인 B: timestamp 만료 (recvWindow)**
- 서버 시계와 로컬 시계 차이가 recvWindow(10000ms) 초과
- `quantbox/core/settings.py`에서 `recvWindow` 확인

---

### -2019: Margin is insufficient

**증상:** 포지션 오픈 시 마진 부족
```
APIError(code=-2019): Margin is insufficient.
```

**원인 A: dynamic_capital 50% cap** ← Incident #3 (2026-03-05)
- `dynamic_capital_enabled=True`일 때 `balance × 0.5` cap이 첫 슬롯에 집중
- 2번째 슬롯부터 마진 부족

**진단:**
```python
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
cfg = VolStraddleConfig()
print(f'dynamic_capital_enabled={cfg.dynamic_capital_enabled}')  # False 확인
# 전략의 현재 슬롯 수 확인
```

**원인 B: 레버리지가 심볼 최대치 초과**
- `scalper.py`의 `-4028` 에러 처리로 자동 fallback 있음
- 그래도 실패 시 해당 심볼의 Binance 최대 레버리지 확인

**원인 C: 기존 포지션 마진이 잔고를 초과**
```python
# 실제 사용 중인 마진 조회
c.futures_account()['totalInitialMargin']
c.futures_account()['availableBalance']
```

---

### -2028: Insufficient margin balance

**증상 A — One-Way close (Incident #1, 2026-02-21):**
```
APIError(code=-2028): Insufficient margin balance.
```
`reduce_only=True` 경로에서 `futures_change_leverage()` 호출. 레버리지 낮추면 포지션 유지 마진 증가 → 에러.
**규칙:** reduce_only 주문에서 절대 leverage/margin 변경 금지.

---

**증상 B — Hedge Mode close trail exit abort (Incident #5, 2026-03-08):**
```
[SUIUSDT] Leverage set failed: APIError(code=-2028) — ABORTING order
```
trail hit 후 포지션이 미청산 상태로 유지됨.

**원인 체인:**
1. trail hit → SL/TP 취소 → `_close_position()` 호출
2. Hedge Mode: `reduce_only=(pos_side is None)` → `pos_side='LONG'` → `reduce_only=False`
3. `_place_order()` guard `if not reduce_only:` → 통과 (leverage=1 기본값)
4. `futures_change_leverage(symbol, leverage=1)` 실행 → 30x 포지션에 1x 설정 → -2028

**Fix (v14.4.2):** `scalper.py L535: if not reduce_only and leverage > 1:`
- Entry만 `leverage=lev_int(30)` 명시 → 변경 실행
- Close/partial-close는 기본값 `leverage=1` → 변경 스킵

**자동복구:** SL/TP 취소 후 bracket repair가 다음 사이클에 자동 재배치.

**코드 확인:**
```bash
grep -n "futures_change_leverage\|leverage > 1" quantbox/scalper.py | head -5
# should show: if not reduce_only and leverage > 1:
```

---

### -4028: Leverage not supported for this symbol

**증상:** 레버리지가 심볼 최대치 초과
```
APIError(code=-4028): Leverage 30x is not supported for XYZUSDT.
```

**자동 처리:** `scalper.py _place_order()`의 fallback 로직 (`max(20, leverage // 2)`)
**수동 확인:**
```python
info = c.futures_exchange_info()
sym_info = next(s for s in info['symbols'] if s['symbol'] == 'XYZUSDT')
print(sym_info['filters'])  # MAX_NUM_ALGO_ORDERS, LEVERAGE 등
```

---

### -4059: No need to change position side

**증상:** positionSide 설정 시도 실패
```
APIError(code=-4059): No need to change position side.
```

**의미:** 이미 해당 모드(hedge/one-way)임. 에러가 아님, 무시해도 됨.

---

### -4068: Cannot change position side if there are open positions

**증상:** 포지션 보유 중 포지션 모드 변경 시도
```
APIError(code=-4068): Cannot change position side if there are open positions or pending orders.
```

**해결:** 모든 포지션/주문 종료 후 모드 변경. 라이브 중 절대 실행 금지.

---

### -1111: Quantity less than zero / Precision error

**증상:** 주문 수량 정밀도 오류
```
APIError(code=-1111): Precision is over the maximum defined for this asset.
```

**원인:** step_size/tick_size에 맞지 않는 수량/가격
**코드 확인:** `quantbox/execution/positions.py`의 `get_symbol_info()` 호출 확인

---

### Algo Order 상태: WORKING → TRIGGERED → FINISHED ← Incident #4 (2026-03-06)

**버그 패턴:** `_check_order_filled()`가 `TRIGGERED`만 체크, `FINISHED` 미처리
```
# Binance 조건부 주문 상태 전이
WORKING → TRIGGERED (순간) → FINISHED (최종 체결)
```

**코드 확인:**
```bash
grep -n "TRIGGERED\|FINISHED\|algoStatus" quantbox/execution/brackets.py | head -10
# 두 상태 모두 체결로 처리해야 함
```

---

## 2. 라이브 세션 이상 패턴 진단

### 2.1 bracket repair 반복 실패
```bash
grep "BRACKET REPAIR FAILED" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1) | wc -l
```
- 1회: 일시적 API 오류 (무시)
- 매 사이클 반복: state 파일 `symbol` 필드 점검 → -1022 원인

### 2.2 포지션 ghost 감지
```bash
grep "Ghost position\|RECONCILIATION" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1)
```
- ghost 감지 = 내부 state와 exchange 불일치
- 정상: 포지션이 server SL/TP로 체결됐으나 state 미업데이트 → reconciliation이 자동 처리
- 이상: 반복 ghost → reconciliation 로직 확인

### 2.3 WR 급락 (backtest vs live 乖離)
```bash
grep "Summary\|WR=" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1) | tail -5
```
- 라이브 WR이 백테스트의 50% 미만이면 lookahead bias 또는 파라미터 불일치 의심
- config.py 값과 실제 로그에서 사용되는 파라미터 비교

### 2.4 Discord 알림 미전달
```bash
grep "Discord\|Notifier" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1) | head -5
```
- `.env`에 `DISCORD_WEBHOOK_URL` 설정 확인

---

## 3. State 파일 부패 진단 & 복구

### 3.1 전체 state 덤프 (빠른 확인)
```python
import json
state = json.load(open('data/state/scalper_state.json'))
print(f"Positions: {list(state.get('positions', {}).keys())}")
print(f"Saved at: {state.get('saved_at')}")
print(f"Balance: {state.get('paper_balance')} / HWM: {state.get('high_water_mark')}")
for sym, pos in state.get('positions', {}).items():
    fields = ['symbol', 'position_side', 'entry_price', 'bars_held',
              'server_sl_order_id', 'server_tp_order_id', 'trail_active', 'peak_gain']
    print(f"\n{sym}:")
    for f in fields:
        print(f"  {f}={pos.get(f)}")
```

### 3.2 알려진 부패 패턴

| 필드 | 정상 | 비정상 | 원인 |
|------|------|--------|------|
| `symbol` | `'SUIUSDT'` | `'SUIUSDT:LONG'` | `_load_state` bug (fixed v14.4.1) |
| `position_side` | `'LONG'` / `'SHORT'` | `None` (hedge mode에서) | 위와 동일 |
| `server_sl_order_id` | `4000000xxxxxxx` / `None` | `'WOULD_TRIGGER'` | v12.3.1 이전 bug |
| `peak_gain` | `float` | `None` | `_load_state`에서 미복원 |

### 3.3 state 백업 & 복원
```bash
# 백업
cp data/state/scalper_state.json data/state/scalper_state_backup_$(date +%Y%m%d_%H%M%S).json

# 복원 (세션 다운 후)
cp data/state/scalper_state_backup_YYYYMMDD_HHMMSS.json data/state/scalper_state.json
```

---

## 4. 성능 이상 진단 체크리스트

백테스트 수치 대비 라이브 성과가 크게 다를 때:

1. **파라미터 불일치:** config.py vs .env vs 실제 로그 파라미터 비교
2. **슬리피지/수수료:** round-trip = 0.0008 (0.08%). 실제 체결가 vs 예상가 비교
3. **Lookahead bias:** 백테스트 WR이 라이브 2배 이상 → 즉시 의심
4. **타임프레임 혼용:** 4h 바를 15m 루프에서 참조 시 `j = (i // ratio) - 1` 확인
5. **포지션 크기:** L/N 값 확인 (VR=3.0, MA=2.1)
6. **데이터 기간:** 강세장만 백테스트 → 실제와 다른 시장 환경
7. **D_trail 파라미터 소스 오염 (r53 인시던트, 2026-03-08):** `backtest_v14_3_0.py`의 `D_TRAIL_ACTIVATE=0.02/D_TRAIL_DISTANCE=0.003` import 절대 금지. 항상 `cfg = VolStraddleConfig()`에서 직접 읽어야 함. r47 이후 프로덕션 값은 `0.005/0.002`. 구버전 파일에서 import 시 D_mom8 결과가 조용히 틀림.
8. **Trail bar-close 실행 갭 (r65 발견, 2026-03-11, ✅ ratchet 구현 완료):**
   - 기존 문제: 라이브는 bar CLOSE에서 MARKET 주문 → 급락 바에서 최대 $158/slot 손실 갭. D_mom8 ΔPF=-0.950 (48% 저하).
   - **해결:** Trail Server Ratchet 구현 완료 (commit `e473e4d`, `trail_server_ratchet_enabled=True`)
     - trail_active 활성화 후 매 bar마다 서버에 STOP 주문을 `peak_gain - trail_distance` 가격에 배치/업데이트
     - 최소 0.1% 개선 시에만 API 호출 (스팸 방지)
     - `quantbox/execution/brackets.py:574-670` `_ratchet_trail_sl()`
     - 현재 라이브 PF 갭은 ratchet 이전 포지션 or paper mode에서만 발생 가능

   **진단:** 대시보드에서 MFE >> |PnL|이고 trailing_stop 청산이면 이 갭이 원인 (ratchet 이전 포지션).
   ```
   SUIUSDT LONG VR: MFE=+131%, PnL=-$33 → 한 15m 바에 +4.37% → -1.1%
   백테스트: trail_stop_gain=4.17% → +$125 / 라이브: close=-1.1% → -$33
   ```

---

## 5. 로그 레벨별 의미

| 레벨 | 의미 | 액션 필요 |
|------|------|-----------|
| `DEBUG` | 상세 계산값 (기본 숨김) | 없음 |
| `INFO` | 정상 동작 기록 | 없음 |
| `WARNING` | 주의 필요하나 계속 실행 | 다음 사이클 확인 |
| `ERROR` | 실패 (포지션 위험 가능) | 즉시 확인 |
| `CRITICAL` | 시스템 중단 | 즉시 개입 |

**ERROR가 반복될 때** (3회 이상 동일 에러):
1. 로그에서 에러 코드 추출 → 이 문서에서 대응책 조회
2. 해결 불가 시 포지션 수동 점검 후 세션 재시작

---

## 6. 테스트 커버리지 확인 방법

버그 수정 후 반드시 실행:
```bash
# 수정된 모듈의 관련 테스트만 빠르게
python -m pytest tests/test_hedge_mode.py tests/test_bracket_repair.py -x -q

# 전체 (1130+ 테스트, ~35초)
python -m pytest tests/ -x -q
```

**테스트 없는 버그 수정 = 미완성.** 수정 후 반드시 regression test 추가.

---

## 7. Scalper 함수 시그니처 & 테스트 패턴 (이 세션에서 반복 실수 방지용)

### 7.1 자주 틀리는 함수 시그니처 (3회 이상 실수)

| 함수 | 올바른 시그니처 | 흔한 실수 |
|------|---------------|-----------|
| `_close_position` | `(pos_key, reason, price)` — 3개 | 4번째 arg `balance` 추가 → TypeError |
| `_scan_entries` | `(symbols, btc_close, balance, max_entries)` — 4개 필수 | 3개만 전달, max_entries 생략 |
| `_build_rb_scalper()` | `rb_cfg=None` param 있음 | kwarg 없이 정의하면 unexpected kwarg 에러 |

### 7.2 Compound Key vs API Symbol

```
pos_key  → "TESTUSDT:LONG"  (strategy._positions dict key, hedge mode)
api_sym  → "TESTUSDT"       (Binance API에 전달하는 순수 심볼)
pos['symbol'] → "TESTUSDT"  (pos dict 안 symbol 필드 — 절대 ':' 포함 금지)

# 올바른 추출
api_sym = pos_key.split(":")[0] if ":" in pos_key else pos_key
```

### 7.3 Rubberband vs VolStraddle Code Path

| 코드 경로 | RubberBand | VolStraddle |
|----------|-----------|-------------|
| `_cycle` L525-527 | ✅ (rubberband-specific) | ❌ skip |
| `strategy.on_bar()` signal | -1/0/1 | -1/0/1/2 (straddle=2) |
| position key format | `"TESTUSDT"` (no colon) | `"TESTUSDT:LONG"` / `"TESTUSDT:SHORT"` |
| `_place_order` position_side | `None` (One-Way) | `"LONG"` / `"SHORT"` (Hedge) |
| `_scan_entries` 4h download | only if `vs_config.ma_cluster_enabled` | always (vol_straddle) |

### 7.4 conftest.py 공유 Fixture 사용법

```python
from tests.conftest import _build_vs_scalper, _build_rb_scalper

# VolStraddle scalper (paper mode, mocked client)
scalper = _build_vs_scalper(balance=1000.0)

# RubberBand scalper
rb_scalper = _build_rb_scalper()  # rb_cfg= optional

# 포지션 주입 (hedge mode, LONG)
_inject_position(scalper, "TESTUSDT:LONG")
# pos['symbol'] = "TESTUSDT" (자동으로 colon 없이 설정됨)
```

> ⚠️ `_build_vs_scalper`는 `pytest.fixture`가 아님 — 함수로 직접 호출.
> `vs_scalper` fixture는 conftest.py에서 자동 주입되며, params 못 받음.

---

## 8. 새 코드 경로 추가 시 Guard 상속 체크리스트

> **목적:** 새 기능(진입 전략, 사이징 로직, 청산 경로)을 추가할 때 기존 Guard들이
> 새 경로에서도 동작하는지 확인. 이 체크를 건너뛰면 "기존 경로는 됐는데 새 경로에서 처음 에러" 패턴 반복.

**새 entry/exit/sizing 로직 추가 시 반드시 확인:**

- [ ] **Symbol 형식**: API에 `pos['symbol']` 전달 (`:` 포함 안 된 순수 심볼). `pos_key` 직접 전달 금지
  → 위반 시 -1022. `_place_order()` 진입 시 `OrderPreflight`가 자동 차단.

- [ ] **레버리지 변경 guard**: `if not reduce_only and leverage > 1:` 조건 있는가?
  → 위반 시 -2028 (Incident #1, #5). close/partial-close 경로에서 절대 leverage 변경 금지.

- [ ] **Notional 상한**: `max_notional` 캡 적용 경로를 지나는가? (`positions.py` 내부)
  → 위반 시 -2027 (ENJ/ICX Incident). CW-Tier, leverage_override 모두 cap 이후 적용됨.

- [ ] **주문 경로**: `_place_order()` 경유하는가?
  → 모든 진입/청산은 `_place_order()`를 통해야 `OrderPreflight`가 자동 적용됨.

- [ ] **경계 상태 테스트 포함**: n_active>0, trail_active=True, hedge mode, reduce_only=False
  → n_active=0만 테스트하면 실거래에서 슬롯 2+에서 에러 (Incident #3).

- [ ] **Binance 상태 전이 처리**: TRIGGERED + FINISHED 둘 다 체결로 처리
  → TRIGGERED만 처리하면 FINISHED 상태에서 ghost 발생 (Incident #4).

**새 코드 경로 추가 후 반드시 실행:**
```bash
python -m pytest tests/ -x -q  # 전체 (1586+ tests)
```

---

## 9. OrderPreflight — 중앙화된 주문 검증 게이트 (v14.5.x~)

`_place_order()` 진입 시 항상 `OrderPreflight.validate()`가 호출됨.
추가 검증이 필요할 때 `quantbox/execution/preflight.py`에 추가:

```python
from quantbox.execution.preflight import OrderPreflight, PreflightResult

# 현재 검증 항목:
# 1. symbol — ':' 포함 금지, 빈 문자열 금지
# 2. side — 'BUY' or 'SELL' 만
# 3. quantity > 0
# 4. leverage ≥ 1
# 5. step_size 정렬 (entry 전용; reduce_only=True는 skip)

# 새 체크 추가 예시:
@staticmethod
def validate(...) -> PreflightResult:
    ...
    # 6. 새 조건 추가
    if some_new_condition:
        return PreflightResult(False, "reason string")
    return PreflightResult(True)
```

`tests/test_preflight.py`에 각 조건마다 테스트 추가 필수.
