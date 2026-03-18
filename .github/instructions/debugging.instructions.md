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

### Incident #6: Reverse Ghost Position (2026-03-13)

**증상:** exchange에 포지션이 있지만 state에 없어 SL/TP 없이 방치됨.
**원인:** `_reconcile_positions()`가 exchange→state 역방향 체크 없음.
**수정 (v14.5.x):** `_reconcile_positions()` Phase 2 + `_startup_exchange_crosscheck()` 추가.

```bash
grep "State out of sync\|REVERSE GHOST" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1)
grep -n "REVERSE GHOST\|_startup_exchange_crosscheck" quantbox/execution/reconciliation.py quantbox/scalper.py
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
WORKING → TRIGGERED (순간) → FINISHED (최종 체결)
```
`_check_order_filled()` (brackets.py L564)에서 TRIGGERED + FINISHED 둘 다 체결 처리.

### Algo Order 취소 상태: `"CANCELED"` (US) vs `"CANCELLED"` (UK) ← Incident #7 (2026-03-17)

**증상:** SL이 Binance에서 취소됐으나 state에 stale ID 잔류 → bracket repair가 "이미 SL 있음"으로 판단 → SL 미재배치 → **포지션 SL 미보호**

**원인:** Binance Algo API는 `algoStatus: "CANCELED"` (미국식 1-L) 반환. 코드는 `"CANCELLED"` (영국식 2-L)만 체크.

**수정 (v14.25.1, commit 8403c72):**
```python
# brackets.py L647, L894 — 두 곳 모두 수정
if status in ("CANCELLED", "CANCELED", "EXPIRED"):
```

**진단:** SL이 없는 포지션이 발견되면:
```bash
grep "CANCELED\|CANCELLED\|bracket repair" $LOG | tail -10
```

### Incident #8: TRAILING_STOP_MARKET API 선택 (2026-03-17)

**증상:** Algo API (`POST /fapi/v1/algoOrder`)의 TRAILING_STOP_MARKET은 `activationPrice` 필수.
mark price가 activationPrice에 도달하지 않으면 `-2021: Order would immediately trigger` reject.
straddle 진입 시 trail 배치 불가 → 포지션이 SL까지 하락 → PF=0.715 (r89).

**수정:** **Regular API** (`POST /fapi/v1/order`) 사용. `activationPrice`가 **optional** (기본값=현재가).
미도달 가격이어도 Binance가 주문 접수 후 대기. -2021 reject 없음.

**현재 메서드:**
- `_place_trail_regular()` → `futures_create_order(type='TRAILING_STOP_MARKET', callbackRate=..., activationPrice=...)`
- `_cancel_trail_regular()` → `futures_cancel_order(orderId=...)`
- `_check_trail_regular()` → `futures_get_order(orderId=...)` → status: NEW/FILLED/CANCELED

**deprecated:** `_place_trailing_stop_market`, `_cancel_trailing_stop_market`, `_check_trail_order_status` (algo API)

**callbackRate 범위:** 0.1-5.0 (1=1%). `s_trail_distance=0.001` → `callbackRate=0.1` (최솟값).

**cross-cancel:** SL 체결 시 trail 취소 (regular), trail 체결 시 SL+TP 취소 (algo).

### Incident #9: Deferred Server Trail — Immediate Callback Fire (2026-03-18)

**증상:** CRVUSDT MA Cluster LONG, entry $0.2520, exit $0.2510, PnL=-0.40%. Trail이 entry 1분 후 바로 발동.

**원인:** `_open_position()`에서 `activationPrice=entry_price`로 trail 즉시 배치.
- Binance Algo API: `activationPrice` 미래 가격을 설정해도 **현재 mark price로 silently override**
- 결과: trail이 즉시 활성화 → 첫 callbackRate%(MA=0.3%, VR=0.2%) 하락에 즉시 trail stop 발동
- `trail_activate` 파라미터(MA=2%, VR=1%)가 완전히 무시됨

**수정 (v14.26.0, b91642d):**
- `positions.py`: `trail_server_native=True`일 때 entry에서 trail 배치 안 함 (`TRAIL=deferred` 로그)
- `scalper.py`: `_manage_active_positions()` HOLD 브랜치에 deferred 블록 추가
  - 조건: `trail_active=True AND server_trail_order_id=None AND not paper_mode AND trail_server_native=True`
  - 배치: `activation_price=current_close` (이미 trail_activate% 도달 by definition)

**MA cluster 실제 효과:** trail이 2% 수익 달성 전까지는 절대 발동 안 함 (기존: 0.3% 하락 즉시 발동)

**진단 방법:**
```bash
# 신규 거래 로그에서 "TRAIL=deferred" 확인 (entry에서 trail 없음 = 정상)
grep "TRAIL=" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1) | head -5
# 기대 출력: Server bracket: SL=OK (algoId=...) TRAIL=deferred (places when trail_activate% reached)

# trail이 deferred로 배치됐을 때 확인
grep "Deferred server trail PLACED" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1)
# 기대 출력: [SUIUSDT:LONG] Deferred server trail PLACED: algoId=... callbackRate=0.3% activationPrice=$...
```

### S_d8 Straddle 실행 갭 (2026-03-17, r89)

**발견:** S_d8 straddle의 전체 수익이 trail exit 가격 정밀도에 의존.
- IDEAL (trail_stop 가격에 정확히 exit): PF=5.44, WR=58.5%
- BAR-CLOSE (현실적 bar close 가격에 exit): **PF=0.715**, WR=32.7%, MaxDD=99.9%
- 갭: **-86.9%** — 모든 연도 손실 (2020-2025)
- **해결:** Regular API TRAILING_STOP_MARKET이 tick-by-tick 추적 → IDEAL에 근접 예상
- **straddle 재활성화됨** (`VS_STRADDLE_ENABLED=true`, `VS_TRAIL_SERVER_NATIVE=true`)

---

## 2. 라이브 세션 이상 패턴 진단

### 2.1 bracket repair 반복 실패
```bash
grep "BRACKET REPAIR FAILED" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1) | wc -l
```
- 1회: 일시적 API 오류 (무시)
- 매 사이클 반복: state 파일 `symbol` 필드 점검 → -1022 원인
- **SL이 Binance에서 취소(`CANCELED`)됐으나 state에 잔류 → 신규 SL 배치 불가 (Incident #7)**

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
7. **D_trail 파라미터 소스 오염 (r53 인시던트, 2026-03-08):** `backtest_v14_3_0.py`의 `D_TRAIL_ACTIVATE=0.02/D_TRAIL_DISTANCE=0.003` import 절대 금지. 항상 `cfg = VolStraddleConfig()`에서 직접 읽어야 함. **r69 이후 프로덕션 값은 `0.010/0.002`** (r47~r68: 0.005/0.002). 구버전 파일에서 import 시 D_mom8 결과가 조용히 틀림.
8. **Trail bar-close 실행 갭:** Trail ratchet 구현 완료 (commit `e473e4d`, v14.5+). MFE >> |PnL| + trailing_stop 청산이면 ratchet 이전 포지션. 신규 포지션은 자동 해결.
9. **Straddle 실행 갭 (r89, 2026-03-17):** S_d8 bar-close PF=0.715 vs IDEAL PF=5.44. 전략의 수익이 전적으로 trail exit 가격 정밀도에 의존. **straddle_enabled=False (프로덕션).**
10. **TRAILING_STOP_MARKET API (2026-03-17):** Algo API(`/fapi/v1/algoOrder`) → `-2021` reject. **Regular API**(`/fapi/v1/order`) 사용 필수. `_place_trail_regular()` / `_cancel_trail_regular()` / `_check_trail_regular()`. deprecated: `_place_trailing_stop_market` 등.
11. **Deferred server trail (Incident #9, 2026-03-18):** trail이 entry에서 즉시 발동하면 CRVUSDT처럼 0.4% 손실 후 종료됨. trail이 `callbackRate`% 하락에 즉시 반응하는지 확인. 로그에서 `TRAIL=deferred` 확인 필요. `trail_server_native=True` + deferred 배치 로직 확인.

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

## 7. Scalper 핵심 참조

- `pos_key` = `"TESTUSDT:LONG"` (hedge), `api_sym` = `"TESTUSDT"` (API용). `pos['symbol']`에 `:` 금지.
- `_build_vs_scalper()` / `_build_rb_scalper()` (conftest.py) — fixture 아님, 직접 호출.
- `_inject_position(scalper, "TESTUSDT:LONG")` — pos['symbol']은 자동 colon 제거.
- RubberBand: one-way, signal -1/0/1. VolStraddle: hedge, signal -1/0/1/2 (straddle=2).

---

## 8. 새 코드 경로 추가 시 Guard 체크리스트

- [ ] **Symbol**: API에 `pos['symbol']` (`:` 없는 순수 심볼). `OrderPreflight` 자동 차단.
- [ ] **Leverage guard**: `if not reduce_only and leverage > 1:` (Incident #1, #5)
- [ ] **Notional cap**: `max_notional` 적용 경로 확인 (positions.py)
- [ ] **주문 경로**: `_place_order()` 경유 필수 (Preflight 자동 적용)
- [ ] **경계 테스트**: n_active>0, trail_active=True, hedge mode (Incident #3)
- [ ] **Algo 상태**: TRIGGERED + FINISHED + CANCELED(US) + CANCELLED(UK) 모두 처리 (Incident #4, #7)

새 경로 추가 후: `python -m pytest tests/ -x -q`

---

## 9. 핵심 안전장치 요약 (Audit 2026-03)

| ID | 위치 | 가드 내용 |
|----|------|-----------|
| F2 | reconciliation.py | SHORT force-close → `askPrice` (LONG → `bidPrice`) |
| A2 | scalper.py | HWM ≤ 0 → balance로 리셋 |
| C1 | brackets.py | `stop_mult < 0` → `return None` |
| D1 | brackets.py | `target_retrace ≤ 0` or `move_size × target_retrace ≥ 1.0` → `return None` |
| B3 | strategy.py | `sl_pct > 0` 확인 후 SL 체크 (0이면 즉시 청산 버그) |
| F3 | reconciliation.py | bracket repair `qty ≤ 0` → skip (극소 notional) |

19개 테스트: `tests/test_critical_path_guards.py`
